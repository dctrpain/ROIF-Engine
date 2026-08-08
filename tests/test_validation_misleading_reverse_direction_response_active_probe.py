"""
ROIF Mechanical Validation
Scenario 03D — Misleading Reverse-Direction Response Active Probe Tests

Core control property
---------------------
A larger scalar response must be explicitly rejected when its vector response
is opposite to the candidate edge direction.

Expected behavior:

    shared_probe_node
        -> loud_reverse_branch      [larger scalar amplitude]
        -> aligned_forward_branch   [smaller scalar amplitude]

    before Probe:
        REQUIRES_PROBE

    vector evidence:
        reverse -> CONTRADICTS
        aligned -> SUPPORTS

    authorization:
        reverse -> confirmed=False
        aligned -> confirmed=True

    graph update:
        reverse relation enters rejected_relations
        aligned relation enters confirmed_edges

    final transition:
        shared_probe_node -> aligned_forward_branch

Evaluator-side amplitude_winner, contradiction_branch, and directional_winner
labels must not participate in inference.
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

from validation.mechanical.misleading_reverse_direction_response_case import (
    ALIGNED_BRANCH_CHANNEL,
    MisleadingReverseDirectionValidationCase,
    PROBE_NODE_CHANNEL,
    PROBE_TO_ALIGNED,
    PROBE_TO_REVERSE,
    REVERSE_BRANCH_CHANNEL,
)

from validation.mechanical.misleading_reverse_direction_response_active_probe import (
    MIN_MEANINGFUL_DELTA_UTILIZATION,
    MIN_REVERSE_TO_ALIGNED_AMPLITUDE_RATIO,
    PROBE_MAGNITUDE,
    REVERSE_PROBE_ID,
    MisleadingReverseDirectionProbeResult,
    ReverseBranchProbeAssessment,
    Scenario03DCandidateProvider,
    amplitude_ratio,
    assess_relation,
    authorize_relation,
    branch_target_id,
    build_bridge,
    build_case,
    build_probe_adapter,
    build_probe_definition,
    build_probe_registry,
    build_snapshot,
    candidate_from_operator,
    local_probe_evidence,
    operator_lookup,
    run_misleading_reverse_direction_probe,
    scalar_probe_amplitude,
)


@pytest.fixture(scope="module")
def case() -> MisleadingReverseDirectionValidationCase:
    return build_case()


@pytest.fixture(scope="module")
def result(
    case: MisleadingReverseDirectionValidationCase,
) -> MisleadingReverseDirectionProbeResult:
    return run_misleading_reverse_direction_probe(
        case
    )


# =============================================================================
# Configuration contract
# =============================================================================


def test_probe_magnitude_is_small_positive() -> None:
    assert PROBE_MAGNITUDE > 0.0
    assert PROBE_MAGNITUDE <= 0.20


def test_min_meaningful_delta_utilization_is_positive() -> None:
    assert (
        MIN_MEANINGFUL_DELTA_UTILIZATION
        > 0.0
    )


def test_required_reverse_amplitude_ratio_is_material() -> None:
    assert (
        MIN_REVERSE_TO_ALIGNED_AMPLITUDE_RATIO
        > 1.0
    )


# =============================================================================
# Candidate graph contract
# =============================================================================


def test_branch_target_lookup_is_exact() -> None:
    assert (
        branch_target_id(
            PROBE_TO_REVERSE
        )
        == REVERSE_BRANCH_CHANNEL
    )

    assert (
        branch_target_id(
            PROBE_TO_ALIGNED
        )
        == ALIGNED_BRANCH_CHANNEL
    )


def test_branch_target_lookup_rejects_unknown_relation() -> None:
    with pytest.raises(Exception):
        branch_target_id(
            "unknown_relation"
        )


def test_reverse_candidate_contract(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    operator = operator_lookup(
        case
    )[
        PROBE_TO_REVERSE
    ]

    candidate = candidate_from_operator(
        operator
    )

    assert (
        candidate.source_id
        == PROBE_NODE_CHANNEL
    )
    assert (
        candidate.target_id
        == REVERSE_BRANCH_CHANNEL
    )
    assert (
        candidate.relation_id
        == PROBE_TO_REVERSE
    )

    assert candidate.metadata[
        "scalar_score_only"
    ] is True


def test_aligned_candidate_contract(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    operator = operator_lookup(
        case
    )[
        PROBE_TO_ALIGNED
    ]

    candidate = candidate_from_operator(
        operator
    )

    assert (
        candidate.source_id
        == PROBE_NODE_CHANNEL
    )
    assert (
        candidate.target_id
        == ALIGNED_BRANCH_CHANNEL
    )
    assert (
        candidate.relation_id
        == PROBE_TO_ALIGNED
    )


def test_candidate_metadata_contains_no_external_role_labels(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    ):
        candidate = candidate_from_operator(
            operator_lookup(
                case
            )[
                relation_id
            ]
        )

        assert candidate.metadata[
            "preferred_branch_label_used"
        ] is False

        assert candidate.metadata[
            "expected_contradiction_label_used"
        ] is False

        assert candidate.metadata[
            "expected_directional_winner_used"
        ] is False


def test_candidate_provider_returns_both_competing_branches(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    from roif.active_cascade import (
        start_active_cascade,
    )

    provider = Scenario03DCandidateProvider(
        case
    )

    snapshot = build_snapshot()

    state = start_active_cascade(
        PROBE_NODE_CHANNEL,
        graph_version=snapshot.version,
    )

    candidates = provider.candidates_for(
        PROBE_NODE_CHANNEL,
        graph_snapshot=snapshot,
        state=state,
    )

    assert {
        candidate.relation_id
        for candidate
        in candidates
    } == {
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    }


# =============================================================================
# Probe / snapshot contract
# =============================================================================


def test_probe_definition_contract() -> None:
    probe = build_probe_definition()

    assert (
        probe.identifier
        == REVERSE_PROBE_ID
    )

    assert (
        probe.perturbation.magnitude
        == pytest.approx(
            PROBE_MAGNITUDE
        )
    )


def test_probe_registry_contains_single_probe() -> None:
    registry = build_probe_registry()

    assert len(
        registry
    ) == 1


def test_probe_adapter_covers_both_candidate_relations() -> None:
    adapter = build_probe_adapter()
    registry = build_probe_registry()
    snapshot = build_snapshot()

    estimate = adapter.estimate_probe(
        registry.get(
            REVERSE_PROBE_ID
        ),
        snapshot,
    )

    assert (
        estimate.probe_identifier
        == REVERSE_PROBE_ID
    )

    assert (
        estimate.graph_coverage
        == pytest.approx(
            1.0
        )
    )

    assert (
        estimate.uncertainty_reduction
        > 0.0
    )


def test_snapshot_contains_both_candidate_uncertainties() -> None:
    snapshot = build_snapshot()

    assert {
        uncertainty.target.identifier
        for uncertainty
        in snapshot.uncertainties
    } == {
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    }


def test_snapshot_contains_no_external_winner_labels() -> None:
    snapshot = build_snapshot()

    assert snapshot.metadata[
        "scalar_amplitude_winner_included"
    ] is False

    assert snapshot.metadata[
        "contradiction_branch_included"
    ] is False

    assert snapshot.metadata[
        "directional_winner_included"
    ] is False

    assert snapshot.metadata[
        "preferred_branch_label_used"
    ] is False


# =============================================================================
# Scalar amplitude conflict
# =============================================================================


def test_reverse_scalar_amplitude_exceeds_aligned(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    reverse = scalar_probe_amplitude(
        case,
        PROBE_TO_REVERSE,
    )

    aligned = scalar_probe_amplitude(
        case,
        PROBE_TO_ALIGNED,
    )

    assert reverse > aligned


def test_reverse_amplitude_ratio_exceeds_control_threshold(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        amplitude_ratio(
            case
        )
        >= MIN_REVERSE_TO_ALIGNED_AMPLITUDE_RATIO
    )


def test_scalar_amplitude_ratio_is_deterministic(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert amplitude_ratio(
        case
    ) == pytest.approx(
        amplitude_ratio(
            case
        ),
        abs=1e-15,
    )


# =============================================================================
# Vector Probe evidence
# =============================================================================


def test_reverse_probe_preserves_high_scalar_magnitude(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_REVERSE,
    )

    assert evidence.tension is not None

    assert scalar_probe_amplitude(
        case,
        PROBE_TO_REVERSE,
    ) > scalar_probe_amplitude(
        case,
        PROBE_TO_ALIGNED,
    )


def test_reverse_probe_response_is_strictly_opposed(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_REVERSE,
    )

    assert evidence.tension is not None

    assert (
        evidence.tension.alignment
        == pytest.approx(
            -1.0,
            abs=1e-12,
        )
    )


def test_reverse_probe_contains_no_utilization_support(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_REVERSE,
    )

    assert evidence.utilization is None

    assert evidence.metadata[
        "utilization_evidence_included"
    ] is False


def test_aligned_probe_response_is_forward_aligned(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_ALIGNED,
    )

    assert evidence.tension is not None

    assert (
        evidence.tension.alignment
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )


def test_aligned_probe_has_meaningful_utilization_change(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_ALIGNED,
    )

    assert evidence.utilization is not None

    assert (
        evidence.utilization.delta_utilization
        >= MIN_MEANINGFUL_DELTA_UTILIZATION
    )


def test_probe_evidence_contains_no_external_label_usage(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    ):
        evidence = local_probe_evidence(
            case,
            relation_id,
        )

        assert evidence.metadata[
            "preferred_branch_label_used"
        ] is False

        assert evidence.metadata[
            "expected_contradiction_label_used"
        ] is False

        assert evidence.metadata[
            "expected_directional_winner_used"
        ] is False


# =============================================================================
# Vector assessment
# =============================================================================


def test_reverse_assessment_contradicts_relation(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_REVERSE,
    )

    assert isinstance(
        assessment,
        ReverseBranchProbeAssessment,
    )

    assert (
        assessment.decision
        is VectorEvidenceDecision.CONTRADICTS
    )

    assert (
        assessment.proposed_confirmed
        is False
    )

    assert (
        assessment.proposed_rejected
        is True
    )


def test_aligned_assessment_supports_relation(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_ALIGNED,
    )

    assert (
        assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )

    assert (
        assessment.proposed_confirmed
        is True
    )

    assert (
        assessment.proposed_rejected
        is False
    )


def test_reverse_assessment_has_higher_scalar_amplitude_than_aligned(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.reverse_assessment.scalar_amplitude
        >
        result.aligned_assessment.scalar_amplitude
    )


def test_reverse_assessment_converts_opposite_direction_into_contradiction(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.reverse_assessment.alignment
        == pytest.approx(
            -1.0,
            abs=1e-12,
        )
    )

    assert (
        result.reverse_assessment.decision
        is VectorEvidenceDecision.CONTRADICTS
    )

    assert (
        result.reverse_assessment.proposed_rejected
        is True
    )


def test_aligned_assessment_is_only_supportive_branch(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.aligned_assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )

    assert (
        result.aligned_assessment.proposed_confirmed
        is True
    )


# =============================================================================
# Authorization boundary
# =============================================================================


def test_reverse_branch_authorizes_negative_evidence(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    authorized = authorize_relation(
        build_bridge(),
        case,
        PROBE_TO_REVERSE,
    )

    assert isinstance(
        authorized,
        AuthorizedRelationEvidence,
    )

    assert (
        authorized.confirmed
        is False
    )

    assert (
        authorized.relation_id
        == PROBE_TO_REVERSE
    )


def test_aligned_branch_authorizes_positive_evidence(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    authorized = authorize_relation(
        build_bridge(),
        case,
        PROBE_TO_ALIGNED,
    )

    assert isinstance(
        authorized,
        AuthorizedRelationEvidence,
    )

    assert (
        authorized.confirmed
        is True
    )

    assert (
        authorized.relation_id
        == PROBE_TO_ALIGNED
    )


def test_complete_run_contains_two_signed_authorizations(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert len(
        result.authorized_evidence
    ) == 2

    signed = {
        item.relation_id: item.confirmed
        for item
        in result.authorized_evidence
    }

    assert signed == {
        PROBE_TO_REVERSE: False,
        PROBE_TO_ALIGNED: True,
    }


def test_authorized_reverse_and_aligned_evidence_have_opposite_signs(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    values = {
        item.confirmed
        for item
        in result.authorized_evidence
    }

    assert values == {
        False,
        True,
    }


# =============================================================================
# Active Cascade transition
# =============================================================================


def test_transition_requires_probe_before_vector_evidence(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.decision_before_probe.status
        is TransitionStatus.REQUIRES_PROBE
    )

    assert (
        result.decision_before_probe.selected
        is None
    )


def test_probe_plan_preserves_both_initial_candidates(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert set(
        result.probe_plan.candidate_relation_ids
    ) == {
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    }


def test_complete_probe_run_advances(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert result.advanced is True


def test_complete_probe_run_selects_aligned_relation(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.selected_relation_id
        == PROBE_TO_ALIGNED
    )


def test_final_state_moves_to_aligned_branch(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.final_state.current_node_id
        == ALIGNED_BRANCH_CHANNEL
    )


def test_final_state_does_not_move_to_reverse_branch(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.final_state.current_node_id
        != REVERSE_BRANCH_CHANNEL
    )


def test_final_state_contains_exactly_one_confirmed_edge(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert len(
        result.final_state.confirmed_edges
    ) == 1


def test_confirmed_edge_is_aligned_relation(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    edge = result.final_state.confirmed_edges[
        0
    ]

    assert (
        edge.relation_id
        == PROBE_TO_ALIGNED
    )

    assert (
        edge.target_id
        == ALIGNED_BRANCH_CHANNEL
    )


def test_reverse_relation_is_explicitly_rejected(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        PROBE_TO_REVERSE
        in result.final_state.rejected_relations
    )


def test_rejected_relations_contains_only_reverse_relation(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.final_state.rejected_relations
        == (
            PROBE_TO_REVERSE,
        )
    )


def test_reverse_relation_is_not_confirmed(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        PROBE_TO_REVERSE
        not in {
            edge.relation_id
            for edge
            in result.final_state.confirmed_edges
        }
    )


# =============================================================================
# Audit / no-label boundary
# =============================================================================


def test_result_metadata_preserves_control_question(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert result.metadata[
        "control_question"
    ] == (
        "does_reverse_vector_evidence_trigger_explicit_rejection"
    )


def test_result_metadata_records_scalar_conflict(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.metadata[
            "reverse_scalar_amplitude"
        ]
        >
        result.metadata[
            "aligned_scalar_amplitude"
        ]
    )

    assert (
        result.metadata[
            "amplitude_ratio"
        ]
        >= MIN_REVERSE_TO_ALIGNED_AMPLITUDE_RATIO
    )


def test_result_metadata_records_reverse_rejection(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.metadata[
            "rejected_relation_id"
        ]
        == PROBE_TO_REVERSE
    )


def test_result_metadata_records_aligned_selection(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert (
        result.metadata[
            "selected_relation_id"
        ]
        == PROBE_TO_ALIGNED
    )

    assert (
        result.metadata[
            "selected_channel_id"
        ]
        == ALIGNED_BRANCH_CHANNEL
    )


def test_result_metadata_declares_no_external_label_usage(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert result.metadata[
        "preferred_branch_label_used"
    ] is False

    assert result.metadata[
        "external_amplitude_winner_used"
    ] is False

    assert result.metadata[
        "external_contradiction_label_used"
    ] is False

    assert result.metadata[
        "external_directional_winner_used"
    ] is False


def test_assessment_metadata_declares_no_external_label_usage(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    for assessment in (
        result.reverse_assessment,
        result.aligned_assessment,
    ):
        assert assessment.metadata[
            "preferred_branch_label_used"
        ] is False

        assert assessment.metadata[
            "expected_contradiction_label_used"
        ] is False

        assert assessment.metadata[
            "expected_directional_winner_used"
        ] is False


# =============================================================================
# Immutability
# =============================================================================


def test_result_metadata_is_read_only(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_assessment_metadata_is_read_only(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    assert isinstance(
        result.reverse_assessment.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.reverse_assessment.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_reverse_assessment_is_frozen(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.reverse_assessment.relation_id = "x"  # type: ignore[misc]


def test_complete_result_is_frozen(
    result: MisleadingReverseDirectionProbeResult,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.final_state = result.initial_state  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_scalar_probe_amplitudes_are_deterministic(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    ):
        left = scalar_probe_amplitude(
            case,
            relation_id,
        )

        right = scalar_probe_amplitude(
            case,
            relation_id,
        )

        assert left == pytest.approx(
            right,
            abs=1e-15,
        )


def test_vector_assessments_are_deterministic(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    ):
        left = assess_relation(
            build_bridge(),
            case,
            relation_id,
        )

        right = assess_relation(
            build_bridge(),
            case,
            relation_id,
        )

        assert left == right


def _authorized_signature(
    result: MisleadingReverseDirectionProbeResult,
) -> tuple[
    tuple[
        object,
        ...,
    ],
    ...,
]:
    return tuple(
        (
            item.relation_id,
            item.confirmed,
            item.confidence,
            item.source_id,
            item.target_id,
            dict(
                item.metadata
            ),
        )
        for item
        in result.authorized_evidence
    )


def test_complete_active_probe_pipeline_is_deterministic(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    left = run_misleading_reverse_direction_probe(
        case
    )

    right = run_misleading_reverse_direction_probe(
        case
    )

    assert (
        left.initial_state
        == right.initial_state
    )

    assert (
        left.decision_before_probe
        == right.decision_before_probe
    )

    assert (
        left.reverse_assessment
        == right.reverse_assessment
    )

    assert (
        left.aligned_assessment
        == right.aligned_assessment
    )

    assert (
        _authorized_signature(
            left
        )
        == _authorized_signature(
            right
        )
    )

    assert (
        left.final_state
        == right.final_state
    )

    assert (
        left.metadata
        == right.metadata
    )
