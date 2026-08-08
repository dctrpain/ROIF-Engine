"""
ROIF Mechanical Validation
Scenario 03C — Misleading High-Amplitude Branch Active Probe Tests

Core control property
---------------------
The larger scalar response must not win when its vector response is orthogonal
to the candidate edge.

Expected behavior:

    shared_probe_node
        -> loud_orthogonal_branch   [larger scalar amplitude]
        -> aligned_quiet_branch     [smaller scalar amplitude]

    before Probe:
        REQUIRES_PROBE

    vector evidence:
        loud    -> INSUFFICIENT
        aligned -> SUPPORTS

    authorization:
        only probe_to_aligned_quiet

    final transition:
        shared_probe_node -> aligned_quiet_branch

Evaluator-side winner labels must not participate in inference.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_cascade import (
    TransitionStatus,
)
from roif.active_cascade_ape import (
    ActiveCascadeAPEError,
    AuthorizedRelationEvidence,
)
from roif.vector_probe import (
    VectorEvidenceDecision,
)

from validation.mechanical.misleading_high_amplitude_branch_case import (
    ALIGNED_BRANCH_CHANNEL,
    LOUD_BRANCH_CHANNEL,
    PROBE_NODE_CHANNEL,
    PROBE_TO_ALIGNED,
    PROBE_TO_LOUD,
    MisleadingHighAmplitudeValidationCase,
)

from validation.mechanical.misleading_high_amplitude_branch_active_probe import (
    MIN_LOUD_TO_ALIGNED_AMPLITUDE_RATIO,
    MIN_MEANINGFUL_DELTA_UTILIZATION,
    MISLEADING_PROBE_ID,
    PROBE_MAGNITUDE,
    BranchProbeAssessment,
    MisleadingHighAmplitudeProbeResult,
    Scenario03CCandidateProvider,
    amplitude_ratio,
    assess_relation,
    authorize_if_supported,
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
    run_misleading_high_amplitude_probe,
    scalar_probe_amplitude,
)


@pytest.fixture(scope="module")
def case() -> MisleadingHighAmplitudeValidationCase:
    return build_case()


@pytest.fixture(scope="module")
def result(
    case: MisleadingHighAmplitudeValidationCase,
) -> MisleadingHighAmplitudeProbeResult:
    return run_misleading_high_amplitude_probe(
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


def test_required_scalar_amplitude_ratio_is_material() -> None:
    assert (
        MIN_LOUD_TO_ALIGNED_AMPLITUDE_RATIO
        > 1.0
    )


# =============================================================================
# Candidate graph contract
# =============================================================================


def test_branch_target_lookup_is_exact() -> None:
    assert (
        branch_target_id(
            PROBE_TO_LOUD
        )
        == LOUD_BRANCH_CHANNEL
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


def test_loud_candidate_contract(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    operator = operator_lookup(
        case
    )[
        PROBE_TO_LOUD
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
        == LOUD_BRANCH_CHANNEL
    )
    assert (
        candidate.relation_id
        == PROBE_TO_LOUD
    )

    assert candidate.metadata[
        "scalar_score_only"
    ] is True


def test_aligned_candidate_contract(
    case: MisleadingHighAmplitudeValidationCase,
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


def test_candidate_metadata_contains_no_directional_winner_label(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_LOUD,
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
            "expected_directional_winner_used"
        ] is False


def test_candidate_provider_returns_both_competing_branches(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    from roif.active_cascade import (
        start_active_cascade,
    )

    provider = Scenario03CCandidateProvider(
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
        for candidate in candidates
    } == {
        PROBE_TO_LOUD,
        PROBE_TO_ALIGNED,
    }


# =============================================================================
# Probe / snapshot contract
# =============================================================================


def test_probe_definition_contract() -> None:
    probe = build_probe_definition()

    assert (
        probe.identifier
        == MISLEADING_PROBE_ID
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
            MISLEADING_PROBE_ID
        ),
        snapshot,
    )

    assert (
        estimate.probe_identifier
        == MISLEADING_PROBE_ID
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
        PROBE_TO_LOUD,
        PROBE_TO_ALIGNED,
    }


def test_snapshot_contains_no_external_winner_labels() -> None:
    snapshot = build_snapshot()

    assert snapshot.metadata[
        "scalar_amplitude_winner_included"
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


def test_loud_scalar_amplitude_exceeds_aligned(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    loud = scalar_probe_amplitude(
        case,
        PROBE_TO_LOUD,
    )

    aligned = scalar_probe_amplitude(
        case,
        PROBE_TO_ALIGNED,
    )

    assert loud > aligned


def test_scalar_amplitude_ratio_exceeds_control_threshold(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert (
        amplitude_ratio(
            case
        )
        >= MIN_LOUD_TO_ALIGNED_AMPLITUDE_RATIO
    )


def test_scalar_amplitude_ratio_is_deterministic(
    case: MisleadingHighAmplitudeValidationCase,
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


def test_loud_probe_preserves_high_scalar_magnitude(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_LOUD,
    )

    assert evidence.tension is not None

    assert scalar_probe_amplitude(
        case,
        PROBE_TO_LOUD,
    ) > scalar_probe_amplitude(
        case,
        PROBE_TO_ALIGNED,
    )


def test_loud_probe_response_is_orthogonal(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_LOUD,
    )

    assert evidence.tension is not None

    assert (
        evidence.tension.alignment
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_loud_probe_contains_no_utilization_support(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    evidence = local_probe_evidence(
        case,
        PROBE_TO_LOUD,
    )

    assert evidence.utilization is None

    assert evidence.metadata[
        "utilization_evidence_included"
    ] is False


def test_aligned_probe_response_is_forward_aligned(
    case: MisleadingHighAmplitudeValidationCase,
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
    case: MisleadingHighAmplitudeValidationCase,
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


def test_probe_evidence_contains_no_expected_winner_usage(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_LOUD,
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
            "expected_directional_winner_used"
        ] is False


# =============================================================================
# Vector assessment
# =============================================================================


def test_loud_assessment_is_insufficient(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_LOUD,
    )

    assert isinstance(
        assessment,
        BranchProbeAssessment,
    )

    assert (
        assessment.decision
        is VectorEvidenceDecision.INSUFFICIENT
    )

    assert assessment.proposed_confirmed is False
    assert assessment.proposed_rejected is False


def test_aligned_assessment_supports_relation(
    case: MisleadingHighAmplitudeValidationCase,
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

    assert assessment.proposed_confirmed is True
    assert assessment.proposed_rejected is False


def test_loud_assessment_has_higher_scalar_amplitude_than_aligned(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert (
        result.loud_assessment.scalar_amplitude
        >
        result.aligned_assessment.scalar_amplitude
    )


def test_loud_assessment_does_not_convert_amplitude_into_support(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert (
        result.loud_assessment.decision
        is VectorEvidenceDecision.INSUFFICIENT
    )

    assert (
        result.loud_assessment.proposed_confirmed
        is False
    )


def test_aligned_assessment_is_only_supportive_branch(
    result: MisleadingHighAmplitudeProbeResult,
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


def test_loud_branch_is_not_authorizable_as_support(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    authorized = authorize_if_supported(
        build_bridge(),
        case,
        PROBE_TO_LOUD,
    )

    assert authorized is None


def test_direct_authorization_of_loud_insufficient_proposal_fails(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    bridge = build_bridge()

    proposal = bridge.vector_evidence.assess(
        local_probe_evidence(
            case,
            PROBE_TO_LOUD,
        )
    )

    with pytest.raises(
        ActiveCascadeAPEError
    ):
        bridge.vector_evidence.authorize(
            proposal,
            approve=True,
        )


def test_aligned_branch_is_authorizable(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    authorized = authorize_if_supported(
        build_bridge(),
        case,
        PROBE_TO_ALIGNED,
    )

    assert isinstance(
        authorized,
        AuthorizedRelationEvidence,
    )

    assert authorized.confirmed is True

    assert (
        authorized.relation_id
        == PROBE_TO_ALIGNED
    )


def test_complete_run_contains_exactly_one_authorized_relation(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert len(
        result.authorized_evidence
    ) == 1

    assert (
        result.authorized_evidence[
            0
        ].relation_id
        == PROBE_TO_ALIGNED
    )


# =============================================================================
# Active Cascade transition
# =============================================================================


def test_transition_requires_probe_before_vector_evidence(
    result: MisleadingHighAmplitudeProbeResult,
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
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert set(
        result.probe_plan.candidate_relation_ids
    ) == {
        PROBE_TO_LOUD,
        PROBE_TO_ALIGNED,
    }


def test_complete_probe_run_advances(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert result.advanced is True


def test_complete_probe_run_selects_aligned_relation(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert (
        result.selected_relation_id
        == PROBE_TO_ALIGNED
    )


def test_final_state_moves_to_aligned_branch(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert (
        result.final_state.current_node_id
        == ALIGNED_BRANCH_CHANNEL
    )


def test_final_state_does_not_move_to_loud_branch(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert (
        result.final_state.current_node_id
        != LOUD_BRANCH_CHANNEL
    )


def test_final_state_contains_exactly_one_confirmed_edge(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert len(
        result.final_state.confirmed_edges
    ) == 1


def test_confirmed_edge_is_aligned_relation(
    result: MisleadingHighAmplitudeProbeResult,
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


def test_loud_relation_is_not_confirmed(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert PROBE_TO_LOUD not in {
        edge.relation_id
        for edge
        in result.final_state.confirmed_edges
    }


# =============================================================================
# Audit / no-label boundary
# =============================================================================


def test_result_metadata_preserves_control_question(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert result.metadata[
        "control_question"
    ] == (
        "does_vector_evidence_reject_louder_orthogonal_response"
    )


def test_result_metadata_records_scalar_conflict(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert (
        result.metadata[
            "loud_scalar_amplitude"
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
        >= MIN_LOUD_TO_ALIGNED_AMPLITUDE_RATIO
    )


def test_result_metadata_records_aligned_selection(
    result: MisleadingHighAmplitudeProbeResult,
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
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert result.metadata[
        "preferred_branch_label_used"
    ] is False

    assert result.metadata[
        "external_amplitude_winner_used"
    ] is False

    assert result.metadata[
        "external_directional_winner_used"
    ] is False


def test_assessment_metadata_declares_no_external_winner_usage(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    for assessment in (
        result.loud_assessment,
        result.aligned_assessment,
    ):
        assert assessment.metadata[
            "preferred_branch_label_used"
        ] is False

        assert assessment.metadata[
            "expected_directional_winner_used"
        ] is False


# =============================================================================
# Immutability
# =============================================================================


def test_result_metadata_is_read_only(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_assessment_metadata_is_read_only(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    assert isinstance(
        result.loud_assessment.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.loud_assessment.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_branch_assessment_is_frozen(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.loud_assessment.relation_id = "x"  # type: ignore[misc]


def test_complete_result_is_frozen(
    result: MisleadingHighAmplitudeProbeResult,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.final_state = result.initial_state  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_scalar_probe_amplitudes_are_deterministic(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_LOUD,
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
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_LOUD,
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
    result: MisleadingHighAmplitudeProbeResult,
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
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    left = run_misleading_high_amplitude_probe(
        case
    )

    right = run_misleading_high_amplitude_probe(
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
        left.loud_assessment
        == right.loud_assessment
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
