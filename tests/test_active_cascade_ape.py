"""
Tests for roif.active_cascade_ape

Integration scope
-----------------
This suite connects:

    roif.active_cascade
    roif.active_cascade_ape
    roif.vector_probe
    existing APE public classes

The core invariant is:

    unresolved transition
        -> real APE planning
        -> typed vector evidence
        -> explicit authorization
        -> confirmed/rejected relation
        -> cascade advancement

Vector evidence itself must never silently authorize a relation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_cascade import (
    ActiveCascadeExplorer,
    ActiveCascadeState,
    CandidateTransition,
    TransitionStatus,
    start_active_cascade,
)

from roif.active_cascade_ape import (
    APEGraphUpdateResolver,
    APEProbePlannerBridge,
    APEVectorEvidenceBridge,
    ActiveCascadeAPEBridge,
    ActiveCascadeAPEConfig,
    ActiveCascadeAPEError,
    ActiveCascadeProbePlan,
    AuthorizedRelationEvidence,
    RelationEvidenceProposal,
)

from roif.active_probe_engine import (
    ActiveProbeEngine,
)

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

from roif.probe_planner import (
    ProbePlanner,
)

from roif.probe_registry import (
    ProbeRegistry,
)

from roif.vector_probe import (
    CandidateEdgeGeometry,
    VectorEvidenceDecision,
    build_vector_probe_evidence,
)


# =============================================================================
# Fixtures / helpers
# =============================================================================


FIRST_RELATION = "A_to_B"
SECOND_RELATION = "A_to_C"

FIRST_PROBE = "probe_A_to_B"
SECOND_PROBE = "probe_A_to_C"


def candidate(
    target: str,
    *,
    relation: str,
    confidence: float = 0.60,
    uncertainty: float = 0.40,
    score: float = 0.0,
) -> CandidateTransition:
    return CandidateTransition(
        source_id="A",
        target_id=target,
        relation_id=relation,
        confidence=confidence,
        uncertainty=uncertainty,
        score=score,
    )


def first_candidate() -> CandidateTransition:
    return candidate(
        "B",
        relation=FIRST_RELATION,
        confidence=0.60,
        uncertainty=0.40,
        score=10.0,
    )


def second_candidate() -> CandidateTransition:
    return candidate(
        "C",
        relation=SECOND_RELATION,
        confidence=0.59,
        uncertainty=0.41,
        score=1.0,
    )


def probe_definition(
    identifier: str,
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=identifier,
        name=identifier,
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.DYNAMIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=PerturbationType.SIMULATED_INTERVENTION,
            magnitude=1.0,
            magnitude_units="normalized",
            duration_seconds=1.0,
        ),
        tags=(
            "active_cascade",
            "validation",
        ),
    )


def make_registry() -> ProbeRegistry:
    return ProbeRegistry(
        (
            probe_definition(FIRST_PROBE),
            probe_definition(SECOND_PROBE),
        )
    )


def make_adapter() -> ProbeGraphAdapter:
    return ProbeGraphAdapter(
        (
            ProbeGraphLink(
                probe_identifier=FIRST_PROBE,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=FIRST_RELATION,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.85,
                    ),
                ),
                novelty=0.9,
                feasibility=1.0,
                estimate_confidence=0.95,
            ),
            ProbeGraphLink(
                probe_identifier=SECOND_PROBE,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=SECOND_RELATION,
                        sensitivity=0.8,
                        discrimination=0.8,
                        expected_uncertainty_reduction=0.50,
                    ),
                ),
                novelty=0.5,
                feasibility=1.0,
                estimate_confidence=0.90,
            ),
        )
    )


def make_snapshot() -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id="active_cascade_ape_test",
        version="v0",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=FIRST_RELATION,
                    source="A",
                    target="B",
                ),
                uncertainty=0.90,
                importance=1.0,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=SECOND_RELATION,
                    source="A",
                    target="C",
                ),
                uncertainty=0.60,
                importance=0.8,
            ),
        ),
    )


def make_planner(
    registry: ProbeRegistry,
) -> ProbePlanner:
    return ProbePlanner(
        registry=registry,
    )


def make_engine(
    registry: ProbeRegistry,
) -> ActiveProbeEngine:
    return ActiveProbeEngine(
        registry=registry,
    )


class StaticCandidateProvider:
    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot,
        state: ActiveCascadeState,
    ):
        if current_node_id == "A":
            return (
                first_candidate(),
                second_candidate(),
            )

        return ()


def make_bridge(
    *,
    config: ActiveCascadeAPEConfig | None = None,
) -> ActiveCascadeAPEBridge:
    registry = make_registry()
    adapter = make_adapter()
    planner = make_planner(
        registry
    )
    engine = make_engine(
        registry
    )

    return ActiveCascadeAPEBridge.create(
        registry=registry,
        adapter=adapter,
        planner=planner,
        engine=engine,
        config=config,
    )


# =============================================================================
# Configuration / data entities
# =============================================================================


def test_default_bridge_config_contract() -> None:
    config = ActiveCascadeAPEConfig()

    assert config.min_confirmation_confidence == pytest.approx(0.80)
    assert config.reject_on_explicit_negative is True
    assert config.require_exact_relation_target is True
    assert config.allow_probe_without_target_link is False
    assert config.allow_vector_support_to_propose_confirmation is True
    assert config.allow_vector_contradiction_to_propose_rejection is True
    assert isinstance(
        config.metadata,
        MappingProxyType,
    )


def test_bridge_config_rejects_invalid_confirmation_confidence() -> None:
    with pytest.raises(ActiveCascadeAPEError):
        ActiveCascadeAPEConfig(
            min_confirmation_confidence=1.1,
        )


def test_relation_evidence_proposal_is_frozen() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = bridge.assess(
        evidence
    )

    with pytest.raises(FrozenInstanceError):
        proposal.confidence = 0.0  # type: ignore[misc]


def test_authorized_relation_evidence_preserves_vector_assessment() -> None:
    vector_bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = vector_bridge.assess(
        evidence
    )
    authorized = vector_bridge.authorize(
        proposal,
        approve=True,
    )

    assert authorized is not None
    assert authorized.vector_assessment is proposal.assessment
    assert authorized.metadata[
        "authorized_from_vector_evidence"
    ] is True


# =============================================================================
# Real APE planner bridge
# =============================================================================


def test_real_registry_contains_two_probe_definitions() -> None:
    registry = make_registry()

    assert set(
        registry.identifiers()
    ) == {
        FIRST_PROBE,
        SECOND_PROBE,
    }


def test_real_adapter_estimates_candidate_probes() -> None:
    registry = make_registry()
    adapter = make_adapter()
    planner = make_planner(
        registry
    )

    bridge = APEProbePlannerBridge(
        registry=registry,
        adapter=adapter,
        planner=planner,
    )

    estimates = bridge.estimate_candidates(
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_snapshot=make_snapshot(),
    )

    assert len(estimates) == 2


def test_real_adapter_filters_estimates_to_local_relations() -> None:
    registry = make_registry()
    adapter = make_adapter()
    planner = make_planner(
        registry
    )

    bridge = APEProbePlannerBridge(
        registry=registry,
        adapter=adapter,
        planner=planner,
    )

    estimates = bridge.estimate_candidates(
        candidates=(
            first_candidate(),
        ),
        graph_snapshot=make_snapshot(),
    )

    assert len(estimates) == 1


def test_real_probe_planner_returns_active_cascade_plan() -> None:
    registry = make_registry()
    adapter = make_adapter()
    planner = make_planner(
        registry
    )

    bridge = APEProbePlannerBridge(
        registry=registry,
        adapter=adapter,
        planner=planner,
    )

    plan = bridge.plan_probe(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_snapshot=make_snapshot(),
        state=start_active_cascade(
            "A",
            graph_version="v0",
        ),
    )

    assert isinstance(
        plan,
        ActiveCascadeProbePlan,
    )
    assert plan.current_node_id == "A"
    assert set(
        plan.candidate_relation_ids
    ) == {
        FIRST_RELATION,
        SECOND_RELATION,
    }
    assert len(plan.estimates) == 2


def test_probe_planning_is_deterministic() -> None:
    registry = make_registry()
    adapter = make_adapter()
    planner = make_planner(
        registry
    )

    bridge = APEProbePlannerBridge(
        registry=registry,
        adapter=adapter,
        planner=planner,
    )

    kwargs = dict(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_snapshot=make_snapshot(),
        state=start_active_cascade(
            "A",
            graph_version="v0",
        ),
    )

    left = bridge.plan_probe(
        **kwargs
    )
    right = bridge.plan_probe(
        **kwargs
    )

    assert left.candidate_relation_ids == right.candidate_relation_ids
    assert left.selected_probe_identifier == right.selected_probe_identifier


def test_real_active_probe_engine_can_start_selected_plan() -> None:
    registry = make_registry()
    adapter = make_adapter()
    planner = make_planner(
        registry
    )
    engine = make_engine(
        registry
    )

    bridge = APEProbePlannerBridge(
        registry=registry,
        adapter=adapter,
        planner=planner,
        engine=engine,
    )

    plan = bridge.plan_probe(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_snapshot=make_snapshot(),
        state=start_active_cascade("A"),
    )

    run = bridge.start_selected_probe(
        plan,
        metadata={
            "case": "active_cascade_ape",
        },
    )

    assert run is not None


# =============================================================================
# Vector evidence bridge
# =============================================================================


def test_forward_vector_response_proposes_confirmation() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = bridge.assess(
        evidence
    )

    assert isinstance(
        proposal,
        RelationEvidenceProposal,
    )
    assert proposal.decision is VectorEvidenceDecision.SUPPORTS
    assert proposal.proposed_confirmed is True
    assert proposal.proposed_rejected is False
    assert proposal.confidence == pytest.approx(1.0)


def test_reverse_vector_response_proposes_rejection() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(-1.0, 0.0),
    )

    proposal = bridge.assess(
        evidence
    )

    assert proposal.decision is VectorEvidenceDecision.CONTRADICTS
    assert proposal.proposed_confirmed is False
    assert proposal.proposed_rejected is True
    assert proposal.confidence == pytest.approx(1.0)


def test_orthogonal_response_is_not_authorizable() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(0.0, 1.0),
    )

    proposal = bridge.assess(
        evidence
    )

    assert proposal.decision is VectorEvidenceDecision.INSUFFICIENT
    assert proposal.proposed_confirmed is False
    assert proposal.proposed_rejected is False

    with pytest.raises(
        ActiveCascadeAPEError,
        match="not eligible",
    ):
        bridge.authorize(
            proposal,
            approve=True,
        )


def test_no_response_is_not_authorizable() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(1.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = bridge.assess(
        evidence
    )

    assert proposal.decision is VectorEvidenceDecision.NO_RESPONSE

    with pytest.raises(ActiveCascadeAPEError):
        bridge.authorize(
            proposal,
            approve=True,
        )


def test_vector_evidence_never_authorizes_without_explicit_approve() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = bridge.assess(
        evidence
    )

    authorized = bridge.authorize(
        proposal,
        approve=False,
    )

    assert authorized is None


def test_authorized_support_becomes_positive_relation_evidence() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = bridge.assess(
        evidence
    )

    authorized = bridge.authorize(
        proposal,
        approve=True,
    )

    assert isinstance(
        authorized,
        AuthorizedRelationEvidence,
    )
    assert authorized.confirmed is True
    assert authorized.relation_id == FIRST_RELATION
    assert authorized.source_id == "A"
    assert authorized.target_id == "B"


def test_authorized_contradiction_becomes_negative_relation_evidence() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(-1.0, 0.0),
    )

    proposal = bridge.assess(
        evidence
    )

    authorized = bridge.authorize(
        proposal,
        approve=True,
    )

    assert authorized is not None
    assert authorized.confirmed is False


# =============================================================================
# Graph update resolver
# =============================================================================


def test_graph_update_resolver_confirms_authorized_relation() -> None:
    resolver = APEGraphUpdateResolver()

    authorized = AuthorizedRelationEvidence(
        relation_id=FIRST_RELATION,
        confirmed=True,
        confidence=0.95,
        source_id="A",
        target_id="B",
    )

    confirmed, rejected = resolver.resolve_update(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_update=authorized,
        state=start_active_cascade("A"),
    )

    assert confirmed is not None
    assert confirmed.relation_id == FIRST_RELATION
    assert rejected == ()


def test_graph_update_resolver_rejects_authorized_negative_relation() -> None:
    resolver = APEGraphUpdateResolver()

    authorized = AuthorizedRelationEvidence(
        relation_id=SECOND_RELATION,
        confirmed=False,
        confidence=0.95,
        source_id="A",
        target_id="C",
    )

    confirmed, rejected = resolver.resolve_update(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_update=authorized,
        state=start_active_cascade("A"),
    )

    assert confirmed is None
    assert rejected == (
        SECOND_RELATION,
    )


def test_low_confidence_positive_evidence_does_not_confirm() -> None:
    resolver = APEGraphUpdateResolver(
        config=ActiveCascadeAPEConfig(
            min_confirmation_confidence=0.80,
        )
    )

    authorized = AuthorizedRelationEvidence(
        relation_id=FIRST_RELATION,
        confirmed=True,
        confidence=0.50,
        source_id="A",
        target_id="B",
    )

    confirmed, rejected = resolver.resolve_update(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_update=authorized,
        state=start_active_cascade("A"),
    )

    assert confirmed is None
    assert rejected == ()


def test_affected_edges_alone_are_not_causal_confirmation() -> None:
    resolver = APEGraphUpdateResolver()

    graph_update = {
        "affected_edges": (
            ("A", "B"),
        ),
    }

    confirmed, rejected = resolver.resolve_update(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_update=graph_update,
        state=start_active_cascade("A"),
    )

    assert confirmed is None
    assert rejected == ()


def test_wrong_source_target_guard_does_not_confirm() -> None:
    resolver = APEGraphUpdateResolver()

    authorized = AuthorizedRelationEvidence(
        relation_id=FIRST_RELATION,
        confirmed=True,
        confidence=1.0,
        source_id="X",
        target_id="B",
    )

    confirmed, rejected = resolver.resolve_update(
        current_node_id="A",
        candidates=(
            first_candidate(),
        ),
        graph_update=authorized,
        state=start_active_cascade("A"),
    )

    assert confirmed is None
    assert rejected == ()


def test_multiple_positive_relations_remain_unresolved() -> None:
    resolver = APEGraphUpdateResolver()

    evidence = (
        AuthorizedRelationEvidence(
            relation_id=FIRST_RELATION,
            confirmed=True,
            confidence=1.0,
            source_id="A",
            target_id="B",
        ),
        AuthorizedRelationEvidence(
            relation_id=SECOND_RELATION,
            confirmed=True,
            confidence=1.0,
            source_id="A",
            target_id="C",
        ),
    )

    confirmed, rejected = resolver.resolve_update(
        current_node_id="A",
        candidates=(
            first_candidate(),
            second_candidate(),
        ),
        graph_update=evidence,
        state=start_active_cascade("A"),
    )

    assert confirmed is None
    assert rejected == ()


# =============================================================================
# Composite bridge
# =============================================================================


def test_composite_bridge_shares_same_config() -> None:
    config = ActiveCascadeAPEConfig(
        min_confirmation_confidence=0.90,
    )

    bridge = make_bridge(
        config=config
    )

    assert bridge.probe_planner.config is config
    assert bridge.vector_evidence.config is config
    assert bridge.graph_update_resolver.config is config


def test_composite_bridge_exposes_real_ape_components() -> None:
    bridge = make_bridge()

    assert isinstance(
        bridge.probe_planner,
        APEProbePlannerBridge,
    )
    assert isinstance(
        bridge.vector_evidence,
        APEVectorEvidenceBridge,
    )
    assert isinstance(
        bridge.graph_update_resolver,
        APEGraphUpdateResolver,
    )


# =============================================================================
# ActiveCascadeExplorer integration
# =============================================================================


def test_active_cascade_requests_real_probe_for_ambiguous_transition() -> None:
    bridge = make_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=StaticCandidateProvider(),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        "A",
        graph_version="v0",
    )

    decision = explorer.inspect_transition(
        state,
        graph_snapshot=make_snapshot(),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=make_snapshot(),
        decision=decision,
    )

    assert isinstance(
        plan,
        ActiveCascadeProbePlan,
    )
    assert len(plan.estimates) == 2


def test_vector_support_can_advance_active_cascade_after_authorization() -> None:
    bridge = make_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=StaticCandidateProvider(),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        "A",
        graph_version="v0",
    )

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    authorized = bridge.vector_evidence.authorize(
        proposal,
        approve=True,
        metadata={
            "reviewed": True,
        },
    )

    assert authorized is not None

    next_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=make_snapshot(),
        graph_update=authorized,
        probe_identifier=FIRST_PROBE,
        graph_version="v1",
    )

    assert next_state.current_node_id == "B"
    assert next_state.step_index == 1
    assert next_state.graph_version == "v1"
    assert next_state.node_path == (
        "A",
        "B",
    )
    assert (
        next_state.confirmed_edges[0].confirmation_source
        == "active_probe"
    )


def test_vector_contradiction_can_reject_branch_without_advancing() -> None:
    bridge = make_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=StaticCandidateProvider(),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade("A")

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="C",
        source_position=(0.0, 0.0),
        target_position=(0.0, 1.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=SECOND_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(0.0, -1.0),
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    authorized = bridge.vector_evidence.authorize(
        proposal,
        approve=True,
    )

    assert authorized is not None
    assert authorized.confirmed is False

    next_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=make_snapshot(),
        graph_update=authorized,
        probe_identifier=SECOND_PROBE,
        graph_version="v1",
    )

    assert next_state.current_node_id == "A"
    assert next_state.step_index == 0
    assert next_state.confirmed_edges == ()
    assert next_state.rejected_relations == (
        SECOND_RELATION,
    )


def test_unauthorized_vector_support_cannot_advance_cascade() -> None:
    bridge = make_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=StaticCandidateProvider(),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade("A")

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    assert proposal.proposed_confirmed is True

    # Proposal itself is not AuthorizedRelationEvidence.
    next_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=make_snapshot(),
        graph_update={
            "proposal": proposal,
        },
        probe_identifier=FIRST_PROBE,
        graph_version="v1",
    )

    assert next_state.current_node_id == "A"
    assert next_state.step_index == 0
    assert next_state.confirmed_edges == ()


def test_high_amplitude_orthogonal_response_cannot_advance_cascade() -> None:
    bridge = make_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=StaticCandidateProvider(),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade("A")

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(0.0, 1000.0),
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    assert proposal.decision is VectorEvidenceDecision.INSUFFICIENT

    with pytest.raises(ActiveCascadeAPEError):
        bridge.vector_evidence.authorize(
            proposal,
            approve=True,
        )

    assert state.current_node_id == "A"
    assert state.confirmed_edges == ()


# =============================================================================
# Full end-to-end active transition trial
# =============================================================================


def test_active_cascade_ape_vector_end_to_end() -> None:
    """
    Full local transition:

        ambiguous A -> {B, C}
            в†“
        REQUIRES_PROBE
            в†“
        real APE plans Probe*
            в†“
        vector response supports A->B
        vector response contradicts A->C
            в†“
        explicit authorization
            в†“
        resolver confirms A->B and rejects A->C
            в†“
        cascade advances to B
    """

    bridge = make_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=StaticCandidateProvider(),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    snapshot = make_snapshot()

    state = start_active_cascade(
        "A",
        graph_version="v0",
    )

    decision = explorer.inspect_transition(
        state,
        graph_snapshot=snapshot,
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=snapshot,
        decision=decision,
    )

    assert isinstance(
        plan,
        ActiveCascadeProbePlan,
    )
    assert len(plan.estimates) == 2

    geometry_b = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    geometry_c = CandidateEdgeGeometry(
        source_id="A",
        target_id="C",
        source_position=(0.0, 0.0),
        target_position=(0.0, 1.0),
    )

    evidence_b = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry_b,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=80.0,
        probe_capacity=100.0,
    )

    evidence_c = build_vector_probe_evidence(
        relation_id=SECOND_RELATION,
        geometry=geometry_c,
        baseline_tension=(0.0, 0.0),
        probe_tension=(0.0, -2.0),
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=70.0,
        probe_capacity=100.0,
    )

    proposals = bridge.vector_evidence.assess_many(
        (
            evidence_b,
            evidence_c,
        )
    )

    by_relation = {
        item.relation_id: item
        for item in proposals
    }

    assert (
        by_relation[FIRST_RELATION].decision
        is VectorEvidenceDecision.SUPPORTS
    )
    assert (
        by_relation[SECOND_RELATION].decision
        is VectorEvidenceDecision.CONTRADICTS
    )

    authorized_b = bridge.vector_evidence.authorize(
        by_relation[FIRST_RELATION],
        approve=True,
        metadata={
            "reviewed": True,
        },
    )

    authorized_c = bridge.vector_evidence.authorize(
        by_relation[SECOND_RELATION],
        approve=True,
        metadata={
            "reviewed": True,
        },
    )

    assert authorized_b is not None
    assert authorized_c is not None

    state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=snapshot,
        graph_update=(
            authorized_b,
            authorized_c,
        ),
        probe_identifier=(
            plan.selected_probe_identifier
            or FIRST_PROBE
        ),
        graph_version="v1",
    )

    assert state.current_node_id == "B"
    assert state.step_index == 1
    assert state.graph_version == "v1"
    assert state.node_path == (
        "A",
        "B",
    )
    assert state.rejected_relations == (
        SECOND_RELATION,
    )

    confirmed_edge = state.confirmed_edges[0]

    assert confirmed_edge.relation_id == FIRST_RELATION
    assert confirmed_edge.confirmation_source == "active_probe"


def test_active_cascade_ape_vector_pipeline_is_deterministic() -> None:
    bridge = APEVectorEvidenceBridge()

    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id=FIRST_RELATION,
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=80.0,
        probe_capacity=100.0,
    )

    left = bridge.assess(
        evidence
    )
    right = bridge.assess(
        evidence
    )

    assert left.relation_id == right.relation_id
    assert left.decision is right.decision
    assert left.proposed_confirmed == right.proposed_confirmed
    assert left.proposed_rejected == right.proposed_rejected
    assert left.confidence == pytest.approx(
        right.confidence
    )



