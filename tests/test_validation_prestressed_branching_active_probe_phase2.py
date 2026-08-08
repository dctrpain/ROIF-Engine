"""
Tests for Scenario 02A Active Probe Phase 2.

Phase 2 question
----------------
Can ROIF actively reconstruct load_sharing_junction as the common convergent
mediator of BOTH pre-stressed branches without reading the external D_root
label during inference?

Required chain:

    primary_branch_stiffness
        -> Probe*
        -> vector/utilization evidence
        -> SUPPORTS
        -> authorization
        -> load_sharing_junction

    bypass_branch_stiffness
        -> Probe*
        -> vector/utilization evidence
        -> SUPPORTS
        -> authorization
        -> load_sharing_junction

Only after both incoming relations independently resolve to the same target
may the test compare the reconstructed junction candidate with the external
Scenario 02A D_root label.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_cascade import (
    ActiveCascadeExplorer,
    TransitionStatus,
    start_active_cascade,
)

from roif.active_cascade_ape import (
    ActiveCascadeProbePlan,
    AuthorizedRelationEvidence,
)

from roif.vector_probe import (
    ResponsePolarity,
    VectorEvidenceDecision,
)

from validation.mechanical.prestressed_branching_active_probe_phase2 import (
    BYPASS_CHANNEL,
    BYPASS_PROBE_ID,
    BYPASS_TO_JUNCTION,
    JUNCTION_CHANNEL,
    MIN_JUNCTION_DELTA_UTILIZATION,
    PRIMARY_CHANNEL,
    PRIMARY_PROBE_ID,
    PRIMARY_TO_JUNCTION,
    PROBE_MAGNITUDE,
    BranchJunctionProbeResult,
    JunctionConvergenceResult,
    Phase2CandidateProvider,
    Phase2ValidationError,
    build_branch_junction_evidence,
    build_bridge,
    build_case,
    build_probe_adapter,
    build_probe_registry,
    build_snapshot,
    candidate_for_relation,
    channel_lookup,
    operator_lookup,
    reconstruct_junction_candidate,
    run_branch_probe,
)

from validation.mechanical.prestressed_branching_case import (
    PrestressedBranchingValidationCase,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def case() -> PrestressedBranchingValidationCase:
    return build_case()


# =============================================================================
# Scenario integrity
# =============================================================================


def test_phase2_uses_expected_branch_and_junction_ids() -> None:
    assert PRIMARY_CHANNEL == "primary_branch_stiffness"
    assert BYPASS_CHANNEL == "bypass_branch_stiffness"
    assert JUNCTION_CHANNEL == "load_sharing_junction"

    assert PRIMARY_TO_JUNCTION == "primary_to_junction"
    assert BYPASS_TO_JUNCTION == "bypass_to_junction"


def test_phase2_probe_magnitude_is_small_positive() -> None:
    assert 0.0 < PROBE_MAGNITUDE < 1.0


def test_phase2_minimum_utilization_change_is_positive() -> None:
    assert MIN_JUNCTION_DELTA_UTILIZATION > 0.0


def test_02a_contains_both_branch_to_junction_operators(
    case: PrestressedBranchingValidationCase,
) -> None:
    operators = operator_lookup(case)

    assert PRIMARY_TO_JUNCTION in operators
    assert BYPASS_TO_JUNCTION in operators

    assert operators[PRIMARY_TO_JUNCTION].source_ids == (
        PRIMARY_CHANNEL,
    )
    assert operators[PRIMARY_TO_JUNCTION].target_ids == (
        JUNCTION_CHANNEL,
    )

    assert operators[BYPASS_TO_JUNCTION].source_ids == (
        BYPASS_CHANNEL,
    )
    assert operators[BYPASS_TO_JUNCTION].target_ids == (
        JUNCTION_CHANNEL,
    )


def test_primary_branch_has_stronger_junction_gain_than_bypass(
    case: PrestressedBranchingValidationCase,
) -> None:
    operators = operator_lookup(case)

    assert (
        float(operators[PRIMARY_TO_JUNCTION].gain)
        >
        float(operators[BYPASS_TO_JUNCTION].gain)
    )


def test_junction_capacity_state_exists(
    case: PrestressedBranchingValidationCase,
) -> None:
    junction = channel_lookup(case)[JUNCTION_CHANNEL]

    assert junction.capacity_state.capacity > 0.0
    assert junction.capacity_state.load >= 0.0
    assert junction.capacity_state.load < junction.capacity_state.capacity


# =============================================================================
# Read-only lookup / data contracts
# =============================================================================


def test_channel_lookup_is_read_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    lookup = channel_lookup(case)

    assert isinstance(lookup, MappingProxyType)

    with pytest.raises(TypeError):
        lookup["x"] = object()  # type: ignore[index]


def test_operator_lookup_is_read_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    lookup = operator_lookup(case)

    assert isinstance(lookup, MappingProxyType)

    with pytest.raises(TypeError):
        lookup["x"] = object()  # type: ignore[index]


def test_candidate_metadata_does_not_use_expected_d_root(
    case: PrestressedBranchingValidationCase,
) -> None:
    primary = candidate_for_relation(
        case,
        PRIMARY_TO_JUNCTION,
    )
    bypass = candidate_for_relation(
        case,
        BYPASS_TO_JUNCTION,
    )

    assert primary.metadata[
        "expected_d_root_label_used"
    ] is False
    assert bypass.metadata[
        "expected_d_root_label_used"
    ] is False


def test_unknown_phase2_relation_is_rejected(
    case: PrestressedBranchingValidationCase,
) -> None:
    with pytest.raises(KeyError):
        candidate_for_relation(
            case,
            "not_a_real_relation",
        )


# =============================================================================
# Probe registry / graph mapping
# =============================================================================


def test_phase2_registry_contains_two_distinct_probes() -> None:
    registry = build_probe_registry()

    assert set(registry.identifiers()) == {
        PRIMARY_PROBE_ID,
        BYPASS_PROBE_ID,
    }


def test_phase2_adapter_maps_primary_probe_to_primary_relation() -> None:
    adapter = build_probe_adapter()

    link = adapter.get_link(
        PRIMARY_PROBE_ID
    )

    assert tuple(
        target.target_identifier
        for target in link.targets
    ) == (
        PRIMARY_TO_JUNCTION,
    )


def test_phase2_adapter_maps_bypass_probe_to_bypass_relation() -> None:
    adapter = build_probe_adapter()

    link = adapter.get_link(
        BYPASS_PROBE_ID
    )

    assert tuple(
        target.target_identifier
        for target in link.targets
    ) == (
        BYPASS_TO_JUNCTION,
    )


def test_primary_snapshot_contains_no_external_root_label() -> None:
    snapshot = build_snapshot(
        PRIMARY_CHANNEL
    )

    assert snapshot.metadata[
        "expected_d_root_label_used"
    ] is False
    assert len(snapshot.uncertainties) == 2
    assert (
        snapshot.uncertainties[0].target.identifier
        == PRIMARY_TO_JUNCTION
    )


def test_bypass_snapshot_contains_no_external_root_label() -> None:
    snapshot = build_snapshot(
        BYPASS_CHANNEL
    )

    assert snapshot.metadata[
        "expected_d_root_label_used"
    ] is False
    assert len(snapshot.uncertainties) == 2
    relation_ids = {
        uncertainty.target.identifier
        for uncertainty in snapshot.uncertainties
    }

    assert BYPASS_TO_JUNCTION in relation_ids


def test_snapshot_rejects_non_branch_source() -> None:
    with pytest.raises(Phase2ValidationError):
        build_snapshot(
            "terminal_displacement"
        )


# =============================================================================
# Active Cascade + APE planning
# =============================================================================


def test_primary_transition_requires_probe(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Phase2CandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        PRIMARY_CHANNEL,
        graph_version="v0",
    )

    decision = explorer.inspect_transition(
        state,
        graph_snapshot=build_snapshot(
            PRIMARY_CHANNEL
        ),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert decision.selected is None


def test_bypass_transition_requires_probe(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Phase2CandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        BYPASS_CHANNEL,
        graph_version="v0",
    )

    decision = explorer.inspect_transition(
        state,
        graph_snapshot=build_snapshot(
            BYPASS_CHANNEL
        ),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert decision.selected is None


def test_real_ape_selects_primary_to_junction_probe(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Phase2CandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        PRIMARY_CHANNEL,
        graph_version="v0",
    )

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=build_snapshot(
            PRIMARY_CHANNEL
        ),
    )

    assert isinstance(
        plan,
        ActiveCascadeProbePlan,
    )
    assert plan.selected_probe_identifier == PRIMARY_PROBE_ID
    assert plan.candidate_relation_ids == (
        PRIMARY_TO_JUNCTION,
    )


def test_real_ape_selects_bypass_to_junction_probe(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Phase2CandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        BYPASS_CHANNEL,
        graph_version="v0",
    )

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=build_snapshot(
            BYPASS_CHANNEL
        ),
    )

    assert isinstance(
        plan,
        ActiveCascadeProbePlan,
    )
    assert plan.selected_probe_identifier == BYPASS_PROBE_ID
    assert plan.candidate_relation_ids == (
        BYPASS_TO_JUNCTION,
    )


# =============================================================================
# Physical vector / utilization evidence
# =============================================================================


def test_primary_probe_response_is_forward_aligned(
    case: PrestressedBranchingValidationCase,
) -> None:
    evidence = build_branch_junction_evidence(
        case,
        PRIMARY_TO_JUNCTION,
    )

    assert evidence.tension is not None
    assert evidence.tension.polarity is ResponsePolarity.FORWARD
    assert evidence.tension.alignment == pytest.approx(1.0)


def test_bypass_probe_response_is_forward_aligned(
    case: PrestressedBranchingValidationCase,
) -> None:
    evidence = build_branch_junction_evidence(
        case,
        BYPASS_TO_JUNCTION,
    )

    assert evidence.tension is not None
    assert evidence.tension.polarity is ResponsePolarity.FORWARD
    assert evidence.tension.alignment == pytest.approx(1.0)


def test_primary_probe_is_utilization_relevant(
    case: PrestressedBranchingValidationCase,
) -> None:
    evidence = build_branch_junction_evidence(
        case,
        PRIMARY_TO_JUNCTION,
    )

    assert evidence.utilization is not None
    assert (
        evidence.utilization.delta_utilization
        >= MIN_JUNCTION_DELTA_UTILIZATION
    )


def test_bypass_probe_is_utilization_relevant(
    case: PrestressedBranchingValidationCase,
) -> None:
    evidence = build_branch_junction_evidence(
        case,
        BYPASS_TO_JUNCTION,
    )

    assert evidence.utilization is not None
    assert (
        evidence.utilization.delta_utilization
        >= MIN_JUNCTION_DELTA_UTILIZATION
    )


def test_primary_probe_changes_junction_more_than_bypass(
    case: PrestressedBranchingValidationCase,
) -> None:
    primary = build_branch_junction_evidence(
        case,
        PRIMARY_TO_JUNCTION,
    )
    bypass = build_branch_junction_evidence(
        case,
        BYPASS_TO_JUNCTION,
    )

    assert primary.utilization is not None
    assert bypass.utilization is not None

    assert (
        primary.utilization.delta_utilization
        >
        bypass.utilization.delta_utilization
    )


def test_both_vector_assessments_support_junction_relations(
    case: PrestressedBranchingValidationCase,
) -> None:
    bridge = build_bridge()

    primary = bridge.vector_evidence.assess(
        build_branch_junction_evidence(
            case,
            PRIMARY_TO_JUNCTION,
        )
    )
    bypass = bridge.vector_evidence.assess(
        build_branch_junction_evidence(
            case,
            BYPASS_TO_JUNCTION,
        )
    )

    assert primary.decision is VectorEvidenceDecision.SUPPORTS
    assert bypass.decision is VectorEvidenceDecision.SUPPORTS


# =============================================================================
# Authorized one-branch execution
# =============================================================================


def test_primary_probe_reaches_junction(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_branch_probe(
        case,
        PRIMARY_CHANNEL,
    )

    assert isinstance(
        result,
        BranchJunctionProbeResult,
    )
    assert result.relation_id == PRIMARY_TO_JUNCTION
    assert result.probe_identifier == PRIMARY_PROBE_ID
    assert result.decision is VectorEvidenceDecision.SUPPORTS
    assert result.reached_junction is True

    assert isinstance(
        result.authorized_evidence,
        AuthorizedRelationEvidence,
    )
    assert result.authorized_evidence.confirmed is True
    assert (
        result.authorized_evidence.target_id
        == JUNCTION_CHANNEL
    )


def test_bypass_probe_reaches_same_junction(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_branch_probe(
        case,
        BYPASS_CHANNEL,
    )

    assert result.relation_id == BYPASS_TO_JUNCTION
    assert result.probe_identifier == BYPASS_PROBE_ID
    assert result.decision is VectorEvidenceDecision.SUPPORTS
    assert result.reached_junction is True
    assert (
        result.authorized_evidence.target_id
        == JUNCTION_CHANNEL
    )


def test_branch_probe_result_metadata_preserves_no_label_boundary(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_branch_probe(
        case,
        PRIMARY_CHANNEL,
    )

    assert isinstance(
        result.metadata,
        MappingProxyType,
    )
    assert result.metadata[
        "expected_d_root_label_used"
    ] is False


def test_branch_probe_result_is_frozen(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_branch_probe(
        case,
        PRIMARY_CHANNEL,
    )

    with pytest.raises(FrozenInstanceError):
        result.reached_junction = False  # type: ignore[misc]


# =============================================================================
# Convergence reconstruction
# =============================================================================


def test_reconstruction_requires_both_supported_relations(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_junction_candidate(
        case
    )

    assert result.primary_result.reached_junction is True
    assert result.bypass_result.reached_junction is True
    assert result.both_relations_supported is True


def test_reconstruction_produces_common_junction_candidate(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_junction_candidate(
        case
    )

    assert isinstance(
        result,
        JunctionConvergenceResult,
    )
    assert result.junction_candidate_id == JUNCTION_CHANNEL


def test_reconstruction_metadata_does_not_use_external_d_root(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_junction_candidate(
        case
    )

    assert isinstance(
        result.metadata,
        MappingProxyType,
    )
    assert result.metadata[
        "expected_d_root_label_used"
    ] is False
    assert result.metadata[
        "evidence_basis"
    ] == "dual_branch_authorized_vector_probe"


def test_reconstructed_junction_matches_external_d_root_only_after_inference(
    case: PrestressedBranchingValidationCase,
) -> None:
    """
    The external expected D_root is read only in this final validation
    comparison, after the two active Probe paths have independently resolved.
    """

    result = reconstruct_junction_candidate(
        case
    )

    assert result.junction_candidate_id == case.expected.d_root


# =============================================================================
# Determinism
# =============================================================================


def test_primary_branch_probe_is_deterministic(
    case: PrestressedBranchingValidationCase,
) -> None:
    left = run_branch_probe(
        case,
        PRIMARY_CHANNEL,
    )
    right = run_branch_probe(
        case,
        PRIMARY_CHANNEL,
    )

    assert left.source_channel_id == right.source_channel_id
    assert left.relation_id == right.relation_id
    assert left.probe_identifier == right.probe_identifier
    assert left.decision is right.decision
    assert left.reached_junction == right.reached_junction

    assert (
        left.authorized_evidence.relation_id
        == right.authorized_evidence.relation_id
    )
    assert (
        left.authorized_evidence.confidence
        == pytest.approx(
            right.authorized_evidence.confidence
        )
    )


def test_complete_phase2_reconstruction_is_deterministic(
    case: PrestressedBranchingValidationCase,
) -> None:
    left = reconstruct_junction_candidate(
        case
    )
    right = reconstruct_junction_candidate(
        case
    )

    assert (
        left.junction_candidate_id
        == right.junction_candidate_id
    )
    assert (
        left.both_relations_supported
        == right.both_relations_supported
    )

    assert (
        left.primary_result.relation_id
        == right.primary_result.relation_id
    )
    assert (
        left.bypass_result.relation_id
        == right.bypass_result.relation_id
    )



def test_phase2_snapshot_contains_complete_local_uncertainty_field() -> None:
    snapshot = build_snapshot(
        PRIMARY_CHANNEL
    )

    relation_ids = {
        uncertainty.target.identifier
        for uncertainty in snapshot.uncertainties
    }

    assert relation_ids == {
        PRIMARY_TO_JUNCTION,
        BYPASS_TO_JUNCTION,
    }


def test_phase2_snapshot_is_not_conditioned_on_active_branch() -> None:
    primary_snapshot = build_snapshot(
        PRIMARY_CHANNEL
    )
    bypass_snapshot = build_snapshot(
        BYPASS_CHANNEL
    )

    primary_relations = {
        uncertainty.target.identifier
        for uncertainty in primary_snapshot.uncertainties
    }

    bypass_relations = {
        uncertainty.target.identifier
        for uncertainty in bypass_snapshot.uncertainties
    }

    assert primary_relations == bypass_relations
    assert primary_relations == {
        PRIMARY_TO_JUNCTION,
        BYPASS_TO_JUNCTION,
    }


