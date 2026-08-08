"""
ROIF Validation Suite
Mechanical Scenario 02A — Active Probe Phase 1

Purpose
-------
Test whether the existing pre-stressed branching Scenario 02A can use the
new Active Cascade + APE + Vector Probe stack to resolve its first ambiguous
cascade transition without receiving the external expected role labels.

This test does NOT claim that D_root or Node* are resolved.

It tests the first active transition only:

    anchor_preload_loss
        ↓
    {anchor_to_primary, anchor_to_bypass}
        ↓
    REQUIRES_PROBE
        ↓
    one anchor perturbation
        ↓
    vector response + Δutilization on both branches
        ↓
    relation evidence
        ↓
    primary branch admitted as the next active cascade node

The local Probe response model is intentionally simple and transparent:

    transmitted_increment = probe_magnitude * |operator.gain|

    probe_load = baseline_load + transmitted_increment

    Δu = probe_load / capacity - baseline_load / capacity

The vector response is aligned with the target channel's declared mechanical
direction from Scenario 02A.

This is an integration/construct test using Scenario 02A's existing mechanical
parameters. It is not an independent finite-element or multibody ground truth.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from roif.active_cascade import (
    ActiveCascadeExplorer,
    ActiveCascadeState,
    CandidateTransition,
    TransitionStatus,
    start_active_cascade,
)

from roif.active_cascade_ape import (
    ActiveCascadeAPEBridge,
    ActiveCascadeAPEConfig,
    ActiveCascadeProbePlan,
    ActiveCascadeAPEError,
    AuthorizedRelationEvidence,
)

from roif.active_probe_engine import ActiveProbeEngine

from roif.probe_entities import (
    Perturbation,
    PerturbationType,
    ProbeDefinition,
    ProbeMethod,
    ProbePurpose,
    ProbeRegime,
)

from roif.probe_graph_adapter import (
    GraphTarget,
    GraphTargetKind,
    GraphUncertainty,
    IncompleteGraphSnapshot,
    ProbeGraphAdapter,
    ProbeGraphLink,
    ProbeGraphTarget,
)

from roif.probe_planner import ProbePlanner
from roif.probe_registry import ProbeRegistry

from roif.vector_probe import (
    CandidateEdgeGeometry,
    VectorEvidenceDecision,
    VectorEvidencePolicy,
    VectorProbeEvidence,
    build_vector_probe_evidence,
)

from validation.mechanical.prestressed_branching_case import (
    PrestressedBranchingValidationCase,
    build_prestressed_branching_case,
)


ANCHOR_CHANNEL = "anchor_preload_loss"
PRIMARY_CHANNEL = "primary_branch_stiffness"
BYPASS_CHANNEL = "bypass_branch_stiffness"

PRIMARY_RELATION = "anchor_to_primary"
BYPASS_RELATION = "anchor_to_bypass"

ANCHOR_PROBE_ID = "probe_02A_anchor_branch_response"

PROBE_MAGNITUDE = 0.20
MIN_MEANINGFUL_DELTA_UTILIZATION = 0.10


@pytest.fixture(scope="module")
def case() -> PrestressedBranchingValidationCase:
    return build_prestressed_branching_case()


def channel_lookup(
    case: PrestressedBranchingValidationCase,
) -> dict[str, Any]:
    return {
        channel.channel_id: channel
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }


def operator_lookup(
    case: PrestressedBranchingValidationCase,
) -> dict[str, Any]:
    return {
        operator.operator_id: operator
        for operator in case.pathological_system.operators
    }


def anchor_outgoing_operators(
    case: PrestressedBranchingValidationCase,
) -> tuple[Any, ...]:
    return tuple(
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (ANCHOR_CHANNEL,)
    )


def vector3_tuple(
    vector: Any,
) -> tuple[float, float, float]:
    return (
        float(vector.x),
        float(vector.y),
        float(vector.z),
    )


def normalized_vector3_tuple(
    vector: Any,
) -> tuple[float, float, float]:
    return vector3_tuple(vector.normalized())


def target_channel_for_operator(
    case: PrestressedBranchingValidationCase,
    operator: Any,
) -> Any:
    assert len(operator.target_ids) == 1
    return channel_lookup(case)[operator.target_ids[0]]


def candidate_from_operator(
    operator: Any,
) -> CandidateTransition:
    assert len(operator.source_ids) == 1
    assert len(operator.target_ids) == 1

    return CandidateTransition(
        source_id=operator.source_ids[0],
        target_id=operator.target_ids[0],
        relation_id=operator.operator_id,
        confidence=0.50,
        uncertainty=0.50,
        score=abs(float(operator.gain)),
        metadata={
            "source": "scenario_02A_operator",
            "expected_role_label_used": False,
        },
    )


def anchor_candidates(
    case: PrestressedBranchingValidationCase,
) -> tuple[CandidateTransition, ...]:
    return tuple(
        candidate_from_operator(operator)
        for operator in anchor_outgoing_operators(case)
    )


class Scenario02ACandidateProvider:
    def __init__(
        self,
        case: PrestressedBranchingValidationCase,
    ) -> None:
        self._case = case

    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ):
        return tuple(
            candidate_from_operator(operator)
            for operator in self._case.pathological_system.operators
            if operator.source_ids == (current_node_id,)
        )


def build_anchor_probe_definition() -> ProbeDefinition:
    return ProbeDefinition(
        identifier=ANCHOR_PROBE_ID,
        name="02A anchor branching response Probe",
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.DYNAMIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=PerturbationType.SIMULATED_INTERVENTION,
            magnitude=PROBE_MAGNITUDE,
            magnitude_units="normalized",
            duration_seconds=1.0,
        ),
        description=(
            "Small reversible synthetic perturbation of the 02A anchor "
            "used to compare directed branch responses."
        ),
        tags=(
            "validation",
            "mechanical_02A",
            "active_cascade",
            "vector_probe",
            "pre_stressed",
        ),
    )


def build_anchor_probe_registry() -> ProbeRegistry:
    return ProbeRegistry((build_anchor_probe_definition(),))


def build_anchor_probe_adapter() -> ProbeGraphAdapter:
    return ProbeGraphAdapter(
        (
            ProbeGraphLink(
                probe_identifier=ANCHOR_PROBE_ID,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=PRIMARY_RELATION,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.90,
                    ),
                    ProbeGraphTarget(
                        target_identifier=BYPASS_RELATION,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.90,
                    ),
                ),
                novelty=0.90,
                feasibility=1.0,
                estimate_confidence=0.95,
                metadata={
                    "validation_case": "mechanical_02A",
                    "hidden_role_labels_used": False,
                },
            ),
        )
    )


def build_anchor_snapshot() -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id="mechanical_02A_active_anchor",
        version="v0",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PRIMARY_RELATION,
                    source=ANCHOR_CHANNEL,
                    target=PRIMARY_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=BYPASS_RELATION,
                    source=ANCHOR_CHANNEL,
                    target=BYPASS_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
        ),
        metadata={
            "validation_case": "mechanical_02A",
            "expected_roles_included": False,
        },
    )


def build_bridge() -> ActiveCascadeAPEBridge:
    registry = build_anchor_probe_registry()
    adapter = build_anchor_probe_adapter()
    planner = ProbePlanner(registry=registry)
    engine = ActiveProbeEngine(registry=registry)

    config = ActiveCascadeAPEConfig(
        min_confirmation_confidence=0.80,
        vector_policy=VectorEvidencePolicy(
            min_forward_alignment=0.70,
            contradiction_alignment=-0.50,
            require_tension_or_displacement=True,
            min_delta_utilization=MIN_MEANINGFUL_DELTA_UTILIZATION,
            require_capacity_relevance=True,
        ),
        metadata={
            "validation_case": "mechanical_02A",
            "phase": "active_probe_1",
        },
    )

    return ActiveCascadeAPEBridge.create(
        registry=registry,
        adapter=adapter,
        planner=planner,
        engine=engine,
        config=config,
    )


def local_probe_evidence(
    case: PrestressedBranchingValidationCase,
    relation_id: str,
) -> VectorProbeEvidence:
    operators = operator_lookup(case)
    operator = operators[relation_id]

    target_channel = target_channel_for_operator(case, operator)
    target_direction = normalized_vector3_tuple(target_channel.direction)

    transmitted_increment = (
        PROBE_MAGNITUDE * abs(float(operator.gain))
    )

    probe_tension = tuple(
        component * transmitted_increment
        for component in target_direction
    )

    capacity_state = target_channel.capacity_state

    geometry = CandidateEdgeGeometry(
        source_id=operator.source_ids[0],
        target_id=operator.target_ids[0],
        source_position=(0.0, 0.0, 0.0),
        target_position=target_direction,
        metadata={
            "geometry_mode": "local_target_direction_from_02A",
        },
    )

    return build_vector_probe_evidence(
        relation_id=relation_id,
        geometry=geometry,
        baseline_tension=(0.0, 0.0, 0.0),
        probe_tension=probe_tension,
        baseline_demand=float(capacity_state.load),
        baseline_capacity=float(capacity_state.capacity),
        probe_demand=(
            float(capacity_state.load)
            + transmitted_increment
        ),
        probe_capacity=float(capacity_state.capacity),
        metadata={
            "probe_magnitude": PROBE_MAGNITUDE,
            "operator_gain": float(operator.gain),
            "expected_role_label_used": False,
        },
    )


def test_02a_has_two_anchor_branch_candidates(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert {
        operator.operator_id
        for operator in anchor_outgoing_operators(case)
    } == {
        PRIMARY_RELATION,
        BYPASS_RELATION,
    }


def test_02a_primary_and_bypass_use_distinct_declared_directions(
    case: PrestressedBranchingValidationCase,
) -> None:
    channels = channel_lookup(case)

    assert vector3_tuple(
        channels[PRIMARY_CHANNEL].direction
    ) != vector3_tuple(
        channels[BYPASS_CHANNEL].direction
    )


def test_02a_primary_has_lower_reserve_than_bypass(
    case: PrestressedBranchingValidationCase,
) -> None:
    channels = channel_lookup(case)

    primary = channels[PRIMARY_CHANNEL].capacity_state
    bypass = channels[BYPASS_CHANNEL].capacity_state

    assert (
        primary.capacity - primary.load
    ) < (
        bypass.capacity - bypass.load
    )


def test_active_probe_test_does_not_read_expected_roles(
    case: PrestressedBranchingValidationCase,
) -> None:
    candidates = anchor_candidates(case)
    snapshot = build_anchor_snapshot()

    assert all(
        candidate.metadata["expected_role_label_used"] is False
        for candidate in candidates
    )
    assert snapshot.metadata["expected_roles_included"] is False


def test_anchor_transition_requires_probe_even_though_primary_gain_is_larger(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario02ACandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        ANCHOR_CHANNEL,
        graph_version="v0",
    )

    decision = explorer.inspect_transition(
        state,
        graph_snapshot=build_anchor_snapshot(),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert decision.selected is None
    assert decision.candidates[0].score > decision.candidates[1].score


def test_real_ape_plans_single_anchor_probe(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario02ACandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        ANCHOR_CHANNEL,
        graph_version="v0",
    )

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=build_anchor_snapshot(),
    )

    assert isinstance(plan, ActiveCascadeProbePlan)
    assert plan.selected_probe_identifier == ANCHOR_PROBE_ID
    assert len(plan.estimates) == 1
    assert set(plan.candidate_relation_ids) == {
        PRIMARY_RELATION,
        BYPASS_RELATION,
    }


def test_same_anchor_probe_produces_larger_delta_utilization_in_primary(
    case: PrestressedBranchingValidationCase,
) -> None:
    primary = local_probe_evidence(case, PRIMARY_RELATION)
    bypass = local_probe_evidence(case, BYPASS_RELATION)

    assert primary.utilization is not None
    assert bypass.utilization is not None

    assert (
        primary.utilization.delta_utilization
        >
        bypass.utilization.delta_utilization
    )


def test_primary_probe_response_exceeds_meaningful_delta_utilization(
    case: PrestressedBranchingValidationCase,
) -> None:
    evidence = local_probe_evidence(case, PRIMARY_RELATION)

    assert evidence.utilization is not None
    assert (
        evidence.utilization.delta_utilization
        >= MIN_MEANINGFUL_DELTA_UTILIZATION
    )


def test_bypass_probe_response_remains_below_meaningful_delta_utilization(
    case: PrestressedBranchingValidationCase,
) -> None:
    evidence = local_probe_evidence(case, BYPASS_RELATION)

    assert evidence.utilization is not None
    assert (
        evidence.utilization.delta_utilization
        < MIN_MEANINGFUL_DELTA_UTILIZATION
    )


def test_both_branch_vector_responses_are_directionally_forward(
    case: PrestressedBranchingValidationCase,
) -> None:
    primary = local_probe_evidence(case, PRIMARY_RELATION)
    bypass = local_probe_evidence(case, BYPASS_RELATION)

    assert primary.tension is not None
    assert bypass.tension is not None

    assert primary.tension.alignment == pytest.approx(1.0)
    assert bypass.tension.alignment == pytest.approx(1.0)


def test_vector_policy_supports_primary_but_not_bypass(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    primary = bridge.vector_evidence.assess(
        local_probe_evidence(case, PRIMARY_RELATION)
    )
    bypass = bridge.vector_evidence.assess(
        local_probe_evidence(case, BYPASS_RELATION)
    )

    assert primary.decision is VectorEvidenceDecision.SUPPORTS
    assert bypass.decision is VectorEvidenceDecision.INSUFFICIENT


def test_only_primary_relation_is_eligible_for_authorization(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    primary = bridge.vector_evidence.assess(
        local_probe_evidence(case, PRIMARY_RELATION)
    )
    bypass = bridge.vector_evidence.assess(
        local_probe_evidence(case, BYPASS_RELATION)
    )

    assert primary.proposed_confirmed is True
    assert bypass.proposed_confirmed is False

    authorized = bridge.vector_evidence.authorize(
        primary,
        approve=True,
        metadata={
            "validation_case": "mechanical_02A",
        },
    )

    assert isinstance(authorized, AuthorizedRelationEvidence)
    assert authorized.confirmed is True

    with pytest.raises(ActiveCascadeAPEError):
        bridge.vector_evidence.authorize(
            bypass,
            approve=True,
        )


def test_authorized_primary_probe_advances_02a_to_primary_branch(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario02ACandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        ANCHOR_CHANNEL,
        graph_version="v0",
    )

    proposal = bridge.vector_evidence.assess(
        local_probe_evidence(case, PRIMARY_RELATION)
    )

    authorized = bridge.vector_evidence.authorize(
        proposal,
        approve=True,
    )

    assert authorized is not None

    next_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=build_anchor_snapshot(),
        graph_update=authorized,
        probe_identifier=ANCHOR_PROBE_ID,
        graph_version="v1",
    )

    assert next_state.current_node_id == PRIMARY_CHANNEL
    assert next_state.step_index == 1
    assert next_state.node_path == (
        ANCHOR_CHANNEL,
        PRIMARY_CHANNEL,
    )

    assert (
        next_state.confirmed_edges[0].relation_id
        == PRIMARY_RELATION
    )


def test_02a_active_probe_phase1_matches_external_d_fast_without_using_label(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario02ACandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        ANCHOR_CHANNEL,
        graph_version="v0",
    )

    proposal = bridge.vector_evidence.assess(
        local_probe_evidence(case, PRIMARY_RELATION)
    )

    authorized = bridge.vector_evidence.authorize(
        proposal,
        approve=True,
    )

    assert authorized is not None

    resolved = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=build_anchor_snapshot(),
        graph_update=authorized,
        probe_identifier=ANCHOR_PROBE_ID,
        graph_version="v1",
    )

    # External label is consulted only after active resolution.
    assert resolved.current_node_id == case.expected.d_fast


def test_phase1_does_not_claim_d_root_or_node_star_resolution(
    case: PrestressedBranchingValidationCase,
) -> None:
    metadata = MappingProxyType(
        {
            "d_fast_phase1_tested": True,
            "d_root_resolved": False,
            "node_star_resolved": False,
        }
    )

    assert metadata["d_fast_phase1_tested"] is True
    assert metadata["d_root_resolved"] is False
    assert metadata["node_star_resolved"] is False


def test_02a_active_probe_phase1_is_deterministic(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    left = bridge.vector_evidence.assess(
        local_probe_evidence(case, PRIMARY_RELATION)
    )

    right = bridge.vector_evidence.assess(
        local_probe_evidence(case, PRIMARY_RELATION)
    )

    assert left.relation_id == right.relation_id
    assert left.decision is right.decision
    assert left.proposed_confirmed == right.proposed_confirmed
    assert left.confidence == pytest.approx(right.confidence)
