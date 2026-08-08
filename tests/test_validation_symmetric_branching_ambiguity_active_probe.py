"""
ROIF Mechanical Validation
Scenario 03B вЂ” Symmetric Branching Ambiguity Active Probe Tests

Control property
----------------
A non-discriminative Probe must preserve ambiguity.

Expected behavior:

    shared_preload_node
        -> branch A candidate
        -> branch B candidate
        -> Probe required
        -> both vector responses SUPPORT
        -> both may be individually authorizable
        -> graph-update resolver receives both
        -> no unique transition is selected
        -> cascade remains at shared_preload_node
        -> status remains REQUIRES_PROBE

No preferred branch or evaluator-side branch_truth is used during inference.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_cascade import (
    TransitionStatus,
)
from roif.active_cascade_ape import (
    AuthorizedRelationEvidence,
)
from roif.vector_probe import (
    VectorEvidenceDecision,
)

from validation.mechanical.symmetric_branching_ambiguity_case import (
    BRANCH_A_CHANNEL,
    BRANCH_B_CHANNEL,
    PRELOAD_CHANNEL,
    PRELOAD_TO_A,
    PRELOAD_TO_B,
    SymmetricBranchingValidationCase,
)

from validation.mechanical.symmetric_branching_ambiguity_active_probe import (
    AMBIGUITY_PROBE_ID,
    MAX_BRANCH_RESPONSE_GAP,
    MIN_MEANINGFUL_DELTA_UTILIZATION,
    PROBE_MAGNITUDE,
    AmbiguousBranchAssessment,
    Scenario03BAmbiguousCandidateProvider,
    SymmetricAmbiguityProbeResult,
    SymmetricBranchingActiveProbeError,
    assess_relation,
    authorized_support,
    branch_response_gap,
    build_bridge,
    build_case,
    build_probe_adapter,
    build_probe_definition,
    build_probe_registry,
    build_snapshot,
    candidate_from_operator,
    local_probe_evidence,
    operator_lookup,
    outgoing_branch_operators,
    run_ambiguity_probe,
)


@pytest.fixture(scope="module")
def case() -> SymmetricBranchingValidationCase:
    return build_case()


@pytest.fixture(scope="module")
def result(
    case: SymmetricBranchingValidationCase,
) -> SymmetricAmbiguityProbeResult:
    return run_ambiguity_probe(case)


# =============================================================================
# Configuration contract
# =============================================================================


def test_probe_magnitude_is_small_positive() -> None:
    assert PROBE_MAGNITUDE > 0.0
    assert PROBE_MAGNITUDE <= 0.20


def test_max_branch_response_gap_is_small_positive() -> None:
    assert MAX_BRANCH_RESPONSE_GAP > 0.0
    assert MAX_BRANCH_RESPONSE_GAP <= 0.05


def test_min_meaningful_delta_utilization_is_positive() -> None:
    assert MIN_MEANINGFUL_DELTA_UTILIZATION > 0.0


# =============================================================================
# Candidate graph contract
# =============================================================================


def test_preload_has_exactly_two_branch_operators(
    case: SymmetricBranchingValidationCase,
) -> None:
    operators = outgoing_branch_operators(case)

    assert len(operators) == 2

    assert {
        operator.operator_id
        for operator in operators
    } == {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    }


def test_candidate_a_contract(
    case: SymmetricBranchingValidationCase,
) -> None:
    operator = operator_lookup(case)[PRELOAD_TO_A]
    candidate = candidate_from_operator(operator)

    assert candidate.source_id == PRELOAD_CHANNEL
    assert candidate.target_id == BRANCH_A_CHANNEL
    assert candidate.relation_id == PRELOAD_TO_A
    assert candidate.confidence == pytest.approx(0.50)
    assert candidate.uncertainty == pytest.approx(0.50)


def test_candidate_b_contract(
    case: SymmetricBranchingValidationCase,
) -> None:
    operator = operator_lookup(case)[PRELOAD_TO_B]
    candidate = candidate_from_operator(operator)

    assert candidate.source_id == PRELOAD_CHANNEL
    assert candidate.target_id == BRANCH_B_CHANNEL
    assert candidate.relation_id == PRELOAD_TO_B
    assert candidate.confidence == pytest.approx(0.50)
    assert candidate.uncertainty == pytest.approx(0.50)


def test_candidate_metadata_contains_no_preferred_branch_label(
    case: SymmetricBranchingValidationCase,
) -> None:
    for relation_id in (
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    ):
        operator = operator_lookup(case)[relation_id]
        candidate = candidate_from_operator(operator)

        assert candidate.metadata[
            "preferred_branch_label_used"
        ] is False

        assert candidate.metadata[
            "expected_role_label_used"
        ] is False


def test_candidate_provider_returns_both_branches(
    case: SymmetricBranchingValidationCase,
) -> None:
    provider = Scenario03BAmbiguousCandidateProvider(case)

    from roif.active_cascade import start_active_cascade

    snapshot = build_snapshot()

    state = start_active_cascade(
        PRELOAD_CHANNEL,
        graph_version=snapshot.version,
    )

    candidates = provider.candidates_for(
        PRELOAD_CHANNEL,
        graph_snapshot=snapshot,
        state=state,
    )

    assert {
        candidate.relation_id
        for candidate in candidates
    } == {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    }


# =============================================================================
# Probe definition / registry / adapter
# =============================================================================


def test_probe_definition_contract() -> None:
    probe = build_probe_definition()

    assert probe.identifier == AMBIGUITY_PROBE_ID
    assert probe.perturbation.magnitude == pytest.approx(
        PROBE_MAGNITUDE
    )


def test_probe_registry_contains_single_probe() -> None:
    registry = build_probe_registry()

    assert len(registry) == 1


def test_probe_adapter_maps_probe_to_both_relations() -> None:
    adapter = build_probe_adapter()
    registry = build_probe_registry()
    snapshot = build_snapshot()

    estimate = adapter.estimate_probe(
        registry.get(
            AMBIGUITY_PROBE_ID
        ),
        snapshot,
    )

    assert (
        estimate.probe_identifier
        == AMBIGUITY_PROBE_ID
    )

    # The single Probe covers both uncertainties in the snapshot.
    assert estimate.graph_coverage == pytest.approx(
        1.0
    )

    assert estimate.uncertainty_reduction > 0.0
    assert estimate.hypothesis_discrimination > 0.0


def test_snapshot_contains_both_branch_uncertainties() -> None:
    snapshot = build_snapshot()

    identifiers = {
        uncertainty.target.identifier
        for uncertainty in snapshot.uncertainties
    }

    assert identifiers == {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    }


def test_snapshot_contains_no_external_branch_truth() -> None:
    snapshot = build_snapshot()

    assert snapshot.metadata[
        "preferred_branch_label_used"
    ] is False

    assert snapshot.metadata[
        "external_branch_truth_included"
    ] is False


# =============================================================================
# Vector Probe evidence
# =============================================================================


def test_branch_a_probe_response_is_forward_aligned(
    case: SymmetricBranchingValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PRELOAD_TO_A,
    )

    assert evidence.tension is not None
    assert evidence.tension.alignment == pytest.approx(1.0)


def test_branch_b_probe_response_is_forward_aligned(
    case: SymmetricBranchingValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PRELOAD_TO_B,
    )

    assert evidence.tension is not None
    assert evidence.tension.alignment == pytest.approx(1.0)


def test_branch_a_probe_has_meaningful_utilization_change(
    case: SymmetricBranchingValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PRELOAD_TO_A,
    )

    assert evidence.utilization is not None

    assert (
        evidence.utilization.delta_utilization
        >= MIN_MEANINGFUL_DELTA_UTILIZATION
    )


def test_branch_b_probe_has_meaningful_utilization_change(
    case: SymmetricBranchingValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PRELOAD_TO_B,
    )

    assert evidence.utilization is not None

    assert (
        evidence.utilization.delta_utilization
        >= MIN_MEANINGFUL_DELTA_UTILIZATION
    )


def test_branch_response_gap_remains_below_control_threshold(
    case: SymmetricBranchingValidationCase,
) -> None:
    gap = branch_response_gap(case)

    assert gap >= 0.0
    assert gap <= MAX_BRANCH_RESPONSE_GAP


def test_branch_probe_evidence_contains_no_preferred_branch_label(
    case: SymmetricBranchingValidationCase,
) -> None:
    for relation_id in (
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    ):
        evidence = local_probe_evidence(
            case,
            relation_id,
        )

        assert evidence.metadata[
            "preferred_branch_label_used"
        ] is False

        assert evidence.metadata[
            "expected_role_label_used"
        ] is False


# =============================================================================
# Vector assessments
# =============================================================================


def test_branch_a_assessment_supports_relation(
    case: SymmetricBranchingValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PRELOAD_TO_A,
    )

    assert isinstance(
        assessment,
        AmbiguousBranchAssessment,
    )

    assert (
        assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )

    assert assessment.proposed_confirmed is True
    assert assessment.proposed_rejected is False


def test_branch_b_assessment_supports_relation(
    case: SymmetricBranchingValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PRELOAD_TO_B,
    )

    assert (
        assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )

    assert assessment.proposed_confirmed is True
    assert assessment.proposed_rejected is False


def test_both_branch_assessments_are_supportive(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert (
        result.branch_a.decision
        is VectorEvidenceDecision.SUPPORTS
    )

    assert (
        result.branch_b.decision
        is VectorEvidenceDecision.SUPPORTS
    )


def test_both_assessments_preserve_no_label_boundary(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.branch_a.metadata[
        "preferred_branch_label_used"
    ] is False

    assert result.branch_b.metadata[
        "preferred_branch_label_used"
    ] is False


# =============================================================================
# Authorization boundary
# =============================================================================


def test_branch_a_can_be_individually_authorized(
    case: SymmetricBranchingValidationCase,
) -> None:
    authorized = authorized_support(
        build_bridge(),
        case,
        PRELOAD_TO_A,
    )

    assert isinstance(
        authorized,
        AuthorizedRelationEvidence,
    )

    assert authorized.confirmed is True


def test_branch_b_can_be_individually_authorized(
    case: SymmetricBranchingValidationCase,
) -> None:
    authorized = authorized_support(
        build_bridge(),
        case,
        PRELOAD_TO_B,
    )

    assert isinstance(
        authorized,
        AuthorizedRelationEvidence,
    )

    assert authorized.confirmed is True


def test_complete_probe_run_preserves_two_authorized_supports(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert len(
        result.authorized_evidence
    ) == 2

    assert {
        item.relation_id
        for item in result.authorized_evidence
    } == {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    }


def test_authorized_evidence_does_not_encode_unique_winner(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert {
        item.confirmed
        for item in result.authorized_evidence
    } == {
        True
    }

    assert len(
        {
            item.relation_id
            for item in result.authorized_evidence
        }
    ) == 2


# =============================================================================
# Active Cascade ambiguity preservation
# =============================================================================


def test_transition_requires_probe_before_evidence(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert (
        result.decision_before_probe.status
        is TransitionStatus.REQUIRES_PROBE
    )

    assert result.decision_before_probe.selected is None


def test_probe_plan_targets_both_candidate_relations(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert set(
        result.probe_plan.candidate_relation_ids
    ) == {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    }


def test_non_discriminative_probe_does_not_advance(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.advanced is False
    assert result.remains_on_preload is True


def test_final_state_remains_on_shared_preload(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert (
        result.final_state.current_node_id
        == PRELOAD_CHANNEL
    )


def test_final_state_has_no_confirmed_edges(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.final_state.confirmed_edges == ()


def test_final_state_has_no_rejected_relations(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.final_state.rejected_relations == ()


def test_transition_remains_requires_probe_after_evidence(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert (
        result.decision_after_probe.status
        is TransitionStatus.REQUIRES_PROBE
    )

    assert result.decision_after_probe.selected is None


def test_after_probe_still_contains_both_candidates(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert {
        candidate.relation_id
        for candidate in result.decision_after_probe.candidates
    } == {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    }


def test_no_micro_difference_is_promoted_to_unique_transition(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.metadata[
        "branch_response_gap"
    ] <= result.metadata[
        "max_allowed_branch_response_gap"
    ]

    assert result.final_state.confirmed_edges == ()


# =============================================================================
# Audit boundary
# =============================================================================


def test_result_metadata_declares_no_preferred_branch_label(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.metadata[
        "preferred_branch_label_used"
    ] is False


def test_result_metadata_declares_no_external_branch_truth_usage(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.metadata[
        "external_branch_truth_used"
    ] is False


def test_result_metadata_records_no_advance(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.metadata[
        "advanced"
    ] is False


def test_control_question_is_preserved(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert result.metadata[
        "control_question"
    ] == (
        "does_non_discriminative_probe_preserve_ambiguity"
    )


# =============================================================================
# Immutability
# =============================================================================


def test_result_metadata_is_read_only(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_branch_assessment_metadata_is_read_only(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    assert isinstance(
        result.branch_a.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.branch_a.metadata["x"] = 1  # type: ignore[index]


def test_branch_assessment_is_frozen(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    with pytest.raises(FrozenInstanceError):
        result.branch_a.relation_id = "x"  # type: ignore[misc]


def test_result_is_frozen(
    result: SymmetricAmbiguityProbeResult,
) -> None:
    with pytest.raises(FrozenInstanceError):
        result.final_state = result.initial_state  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_branch_response_gap_is_deterministic(
    case: SymmetricBranchingValidationCase,
) -> None:
    left = branch_response_gap(case)
    right = branch_response_gap(case)

    assert left == pytest.approx(
        right,
        abs=1e-15,
    )


def test_branch_assessments_are_deterministic(
    case: SymmetricBranchingValidationCase,
) -> None:
    left_a = assess_relation(
        build_bridge(),
        case,
        PRELOAD_TO_A,
    )
    right_a = assess_relation(
        build_bridge(),
        case,
        PRELOAD_TO_A,
    )

    left_b = assess_relation(
        build_bridge(),
        case,
        PRELOAD_TO_B,
    )
    right_b = assess_relation(
        build_bridge(),
        case,
        PRELOAD_TO_B,
    )

    assert left_a == right_a
    assert left_b == right_b


def test_complete_ambiguity_probe_is_deterministic(
    case: SymmetricBranchingValidationCase,
) -> None:
    left = run_ambiguity_probe(case)
    right = run_ambiguity_probe(case)

    assert left.initial_state == right.initial_state
    assert (
        left.decision_before_probe
        == right.decision_before_probe
    )
    assert left.branch_a == right.branch_a
    assert left.branch_b == right.branch_b
    left_authorized = tuple(
        (
            item.relation_id,
            item.confirmed,
            item.confidence,
            item.source_id,
            item.target_id,
            dict(item.metadata),
        )
        for item in left.authorized_evidence
    )

    right_authorized = tuple(
        (
            item.relation_id,
            item.confirmed,
            item.confidence,
            item.source_id,
            item.target_id,
            dict(item.metadata),
        )
        for item in right.authorized_evidence
    )

    assert left_authorized == right_authorized
    assert left.final_state == right.final_state
    assert (
        left.decision_after_probe
        == right.decision_after_probe
    )
    assert left.metadata == right.metadata

