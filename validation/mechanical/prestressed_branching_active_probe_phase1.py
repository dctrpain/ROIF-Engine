"""
ROIF Mechanical Validation
Scenario 02A — Active Probe Phase 1

Reusable Phase-1 orchestration extracted from the already passing
Scenario 02A Active Probe validation test.

This module resolves only the first ambiguous transition and does not
claim D_root or Node* resolution.
"""

from __future__ import annotations

from typing import Any

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



def run_phase1(
    case: PrestressedBranchingValidationCase,
) -> ActiveCascadeState:
    """Run the validated Scenario 02A Phase-1 active-probe transition."""
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

    if authorized is None:
        raise ActiveCascadeAPEError(
            "Phase 1 primary relation was not authorized."
        )

    return explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=build_anchor_snapshot(),
        graph_update=authorized,
        probe_identifier=ANCHOR_PROBE_ID,
        graph_version="v1",
    )


__all__ = [
    "ANCHOR_CHANNEL",
    "PRIMARY_CHANNEL",
    "BYPASS_CHANNEL",
    "PRIMARY_RELATION",
    "BYPASS_RELATION",
    "ANCHOR_PROBE_ID",
    "PROBE_MAGNITUDE",
    "MIN_MEANINGFUL_DELTA_UTILIZATION",
    "Scenario02ACandidateProvider",
    "channel_lookup",
    "operator_lookup",
    "anchor_outgoing_operators",
    "vector3_tuple",
    "normalized_vector3_tuple",
    "target_channel_for_operator",
    "candidate_from_operator",
    "anchor_candidates",
    "build_anchor_probe_definition",
    "build_anchor_probe_registry",
    "build_anchor_probe_adapter",
    "build_anchor_snapshot",
    "build_bridge",
    "local_probe_evidence",
    "run_phase1",
]
