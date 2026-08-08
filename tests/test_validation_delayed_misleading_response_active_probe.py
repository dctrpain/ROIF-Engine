"""
ROIF Mechanical Validation
Scenario 03E — Delayed Misleading Response Active Probe Tests

Core control property
---------------------
Temporal precedence must not be promoted directly to causal authorization.

Expected behavior:

    EARLY WINDOW
        fast_misleading_branch:
            visible=True
            VectorEvidenceDecision.SUPPORTS
            proposed_confirmed=False

        delayed_true_branch:
            visible=False
            VectorEvidenceDecision.NO_RESPONSE
            proposed_confirmed=False

        graph:
            no confirmed edges
            no advance

    FULL WINDOW
        fast_misleading_branch:
            visible=True
            SUPPORTS
            proposed_confirmed=False

        delayed_true_branch:
            visible=True
            SUPPORTS
            proposed_confirmed=True

        authorization:
            only probe_to_delayed_true

        final transition:
            shared_probe_node -> delayed_true_branch

Evaluator-side first_response_branch and directional_winner labels must not
participate in inference.
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

from validation.mechanical.delayed_misleading_response_case import (
    DELAYED_BRANCH_CHANNEL,
    DELAYED_RESPONSE_DELAY_SECONDS,
    EARLY_OBSERVATION_WINDOW_SECONDS,
    FAST_BRANCH_CHANNEL,
    FAST_RESPONSE_DELAY_SECONDS,
    FULL_OBSERVATION_WINDOW_SECONDS,
    PROBE_NODE_CHANNEL,
    PROBE_TO_DELAYED,
    PROBE_TO_FAST,
    DelayedMisleadingResponseValidationCase,
)

from validation.mechanical.delayed_misleading_response_active_probe import (
    DELAYED_PROBE_ID,
    MIN_DELAYED_SUPPORT_MARGIN,
    MIN_MEANINGFUL_DELTA_UTILIZATION,
    PROBE_MAGNITUDE,
    DelayedMisleadingResponseProbeResult,
    Scenario03ECandidateProvider,
    TemporalProbeAssessment,
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
    response_delay_seconds,
    run_delayed_misleading_response_probe,
    scalar_probe_amplitude,
)


@pytest.fixture(scope="module")
def case() -> DelayedMisleadingResponseValidationCase:
    return build_case()


@pytest.fixture(scope="module")
def result(
    case: DelayedMisleadingResponseValidationCase,
) -> DelayedMisleadingResponseProbeResult:
    return run_delayed_misleading_response_probe(
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


def test_delayed_support_margin_is_non_negative() -> None:
    assert (
        MIN_DELAYED_SUPPORT_MARGIN
        >= 0.0
    )


def test_early_window_precedes_delayed_response() -> None:
    assert (
        EARLY_OBSERVATION_WINDOW_SECONDS
        < DELAYED_RESPONSE_DELAY_SECONDS
    )


def test_full_window_contains_delayed_response() -> None:
    assert (
        FULL_OBSERVATION_WINDOW_SECONDS
        >= DELAYED_RESPONSE_DELAY_SECONDS
    )


# =============================================================================
# Candidate graph contract
# =============================================================================


def test_branch_target_lookup_is_exact() -> None:
    assert (
        branch_target_id(
            PROBE_TO_FAST
        )
        == FAST_BRANCH_CHANNEL
    )

    assert (
        branch_target_id(
            PROBE_TO_DELAYED
        )
        == DELAYED_BRANCH_CHANNEL
    )


def test_branch_target_lookup_rejects_unknown_relation() -> None:
    with pytest.raises(Exception):
        branch_target_id(
            "unknown_relation"
        )


def test_fast_candidate_contract(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    operator = operator_lookup(
        case
    )[
        PROBE_TO_FAST
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
        == FAST_BRANCH_CHANNEL
    )

    assert (
        candidate.relation_id
        == PROBE_TO_FAST
    )

    assert candidate.metadata[
        "temporal_precedence_not_confirmation"
    ] is True


def test_delayed_candidate_contract(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    operator = operator_lookup(
        case
    )[
        PROBE_TO_DELAYED
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
        == DELAYED_BRANCH_CHANNEL
    )

    assert (
        candidate.relation_id
        == PROBE_TO_DELAYED
    )


def test_candidate_metadata_contains_no_external_temporal_labels(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
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
            "expected_first_response_label_used"
        ] is False

        assert candidate.metadata[
            "expected_directional_winner_used"
        ] is False


def test_candidate_provider_returns_both_competing_branches(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    from roif.active_cascade import (
        start_active_cascade,
    )

    provider = Scenario03ECandidateProvider(
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
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
    }


# =============================================================================
# Probe / snapshot contract
# =============================================================================


def test_probe_definition_contract() -> None:
    probe = build_probe_definition()

    assert (
        probe.identifier
        == DELAYED_PROBE_ID
    )

    assert (
        probe.perturbation.magnitude
        == pytest.approx(
            PROBE_MAGNITUDE
        )
    )

    assert (
        probe.perturbation.duration_seconds
        == pytest.approx(
            FULL_OBSERVATION_WINDOW_SECONDS
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
            DELAYED_PROBE_ID
        ),
        snapshot,
    )

    assert (
        estimate.probe_identifier
        == DELAYED_PROBE_ID
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
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
    }


def test_snapshot_contains_no_external_temporal_labels() -> None:
    snapshot = build_snapshot()

    assert snapshot.metadata[
        "first_response_branch_included"
    ] is False

    assert snapshot.metadata[
        "directional_winner_included"
    ] is False

    assert snapshot.metadata[
        "preferred_branch_label_used"
    ] is False


# =============================================================================
# Temporal helper contract
# =============================================================================


def test_fast_response_delay_is_earlier() -> None:
    assert (
        response_delay_seconds(
            PROBE_TO_FAST
        )
        < response_delay_seconds(
            PROBE_TO_DELAYED
        )
    )


def test_response_delay_values_match_case_constants() -> None:
    assert response_delay_seconds(
        PROBE_TO_FAST
    ) == pytest.approx(
        FAST_RESPONSE_DELAY_SECONDS
    )

    assert response_delay_seconds(
        PROBE_TO_DELAYED
    ) == pytest.approx(
        DELAYED_RESPONSE_DELAY_SECONDS
    )


def test_active_probe_delay_helper_rejects_unknown_relation() -> None:
    with pytest.raises(Exception):
        response_delay_seconds(
            "unknown_relation"
        )


# =============================================================================
# Early-window evidence
# =============================================================================


def test_early_fast_response_is_visible(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_FAST,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert assessment.visible is True


def test_early_delayed_response_is_not_visible(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_DELAYED,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert assessment.visible is False


def test_early_fast_vector_layer_supports_relation(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_FAST,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert (
        assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )


def test_early_fast_temporal_gate_blocks_confirmation(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_FAST,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert (
        assessment.proposed_confirmed
        is False
    )

    assert (
        assessment.delta_utilization
        is None
    )


def test_early_delayed_vector_layer_reports_no_response(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assessment = assess_relation(
        build_bridge(),
        case,
        PROBE_TO_DELAYED,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert (
        assessment.decision
        is VectorEvidenceDecision.NO_RESPONSE
    )

    assert (
        assessment.proposed_confirmed
        is False
    )


def test_early_fast_is_not_authorizable(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    authorized = authorize_if_supported(
        build_bridge(),
        case,
        PROBE_TO_FAST,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert authorized is None


def test_early_delayed_is_not_authorizable(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    authorized = authorize_if_supported(
        build_bridge(),
        case,
        PROBE_TO_DELAYED,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert authorized is None


def test_early_state_does_not_advance(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert result.advanced_early is False

    assert (
        result.early_state.current_node_id
        == PROBE_NODE_CHANNEL
    )


def test_early_state_contains_no_confirmed_edges(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        len(
            result.early_state.confirmed_edges
        )
        == 0
    )


def test_early_state_preserves_no_rejected_relations(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.early_state.rejected_relations
        == ()
    )


# =============================================================================
# Full-window evidence
# =============================================================================


def test_full_fast_response_remains_visible(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.full_fast_assessment.visible
        is True
    )


def test_full_delayed_response_becomes_visible(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.full_delayed_assessment.visible
        is True
    )


def test_full_fast_vector_layer_still_supports_relation(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.full_fast_assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )


def test_full_fast_temporal_gate_still_blocks_confirmation(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.full_fast_assessment.proposed_confirmed
        is False
    )

    assert (
        result.full_fast_assessment.delta_utilization
        is None
    )


def test_full_delayed_vector_layer_supports_relation(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.full_delayed_assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )


def test_full_delayed_has_meaningful_utilization_change(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.full_delayed_assessment.delta_utilization
        is not None
    )

    assert (
        result.full_delayed_assessment.delta_utilization
        >= MIN_MEANINGFUL_DELTA_UTILIZATION
    )


def test_full_delayed_is_confirmable(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.full_delayed_assessment.proposed_confirmed
        is True
    )

    assert (
        result.full_delayed_assessment.proposed_rejected
        is False
    )


def test_full_fast_is_not_authorizable(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    authorized = authorize_if_supported(
        build_bridge(),
        case,
        PROBE_TO_FAST,
        observation_window_seconds=(
            FULL_OBSERVATION_WINDOW_SECONDS
        ),
    )

    assert authorized is None


def test_full_delayed_is_authorizable(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    authorized = authorize_if_supported(
        build_bridge(),
        case,
        PROBE_TO_DELAYED,
        observation_window_seconds=(
            FULL_OBSERVATION_WINDOW_SECONDS
        ),
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
        == PROBE_TO_DELAYED
    )


# =============================================================================
# Complete temporal pipeline
# =============================================================================


def test_transition_requires_probe_before_temporal_observation(
    result: DelayedMisleadingResponseProbeResult,
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
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert set(
        result.probe_plan.candidate_relation_ids
    ) == {
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
    }


def test_complete_run_preserves_early_fast_support_without_advance(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.early_fast_assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    )

    assert (
        result.early_fast_assessment.proposed_confirmed
        is False
    )

    assert result.advanced_early is False


def test_complete_run_authorizes_only_delayed_relation(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        len(
            result.authorized_evidence
        )
        == 1
    )

    assert (
        result.authorized_evidence[
            0
        ].relation_id
        == PROBE_TO_DELAYED
    )

    assert (
        result.authorized_evidence[
            0
        ].confirmed
        is True
    )


def test_complete_probe_run_advances_finally(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.advanced_final
        is True
    )


def test_complete_probe_run_selects_delayed_relation(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.selected_relation_id
        == PROBE_TO_DELAYED
    )


def test_final_state_moves_to_delayed_true_branch(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.final_state.current_node_id
        == DELAYED_BRANCH_CHANNEL
    )


def test_final_state_does_not_move_to_fast_branch(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.final_state.current_node_id
        != FAST_BRANCH_CHANNEL
    )


def test_final_state_contains_exactly_one_confirmed_edge(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        len(
            result.final_state.confirmed_edges
        )
        == 1
    )


def test_confirmed_edge_is_delayed_relation(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    edge = result.final_state.confirmed_edges[
        0
    ]

    assert (
        edge.relation_id
        == PROBE_TO_DELAYED
    )

    assert (
        edge.target_id
        == DELAYED_BRANCH_CHANNEL
    )


def test_fast_relation_is_not_confirmed(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        PROBE_TO_FAST
        not in {
            edge.relation_id
            for edge
            in result.final_state.confirmed_edges
        }
    )


def test_no_relation_is_rejected_in_temporal_control(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.final_state.rejected_relations
        == ()
    )


# =============================================================================
# Audit / no-label boundary
# =============================================================================


def test_result_metadata_preserves_control_question(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert result.metadata[
        "control_question"
    ] == (
        "does_active_probe_wait_for_delayed_evidence_before_advancing"
    )


def test_result_metadata_records_two_observation_windows(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.metadata[
            "early_window_seconds"
        ]
        == pytest.approx(
            EARLY_OBSERVATION_WINDOW_SECONDS
        )
    )

    assert (
        result.metadata[
            "full_window_seconds"
        ]
        == pytest.approx(
            FULL_OBSERVATION_WINDOW_SECONDS
        )
    )


def test_result_metadata_records_response_delays(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.metadata[
            "fast_response_delay_seconds"
        ]
        == pytest.approx(
            FAST_RESPONSE_DELAY_SECONDS
        )
    )

    assert (
        result.metadata[
            "delayed_response_delay_seconds"
        ]
        == pytest.approx(
            DELAYED_RESPONSE_DELAY_SECONDS
        )
    )


def test_result_metadata_records_no_early_advance(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.metadata[
            "advanced_early"
        ]
        is False
    )


def test_result_metadata_records_delayed_selection(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.metadata[
            "selected_relation_id"
        ]
        == PROBE_TO_DELAYED
    )

    assert (
        result.metadata[
            "selected_channel_id"
        ]
        == DELAYED_BRANCH_CHANNEL
    )


def test_result_metadata_declares_no_external_label_usage(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert (
        result.metadata[
            "preferred_branch_label_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "external_first_response_label_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "external_directional_winner_used"
        ]
        is False
    )


def test_temporal_assessment_metadata_declares_no_external_label_usage(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    for assessment in (
        result.early_fast_assessment,
        result.early_delayed_assessment,
        result.full_fast_assessment,
        result.full_delayed_assessment,
    ):
        assert (
            assessment.metadata[
                "preferred_branch_label_used"
            ]
            is False
        )

        assert (
            assessment.metadata[
                "expected_first_response_label_used"
            ]
            is False
        )

        assert (
            assessment.metadata[
                "expected_directional_winner_used"
            ]
            is False
        )


# =============================================================================
# Immutability
# =============================================================================


def test_result_metadata_is_read_only(
    result: DelayedMisleadingResponseProbeResult,
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
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    assert isinstance(
        result.early_fast_assessment.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.early_fast_assessment.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_temporal_assessment_is_frozen(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.early_fast_assessment.visible = False  # type: ignore[misc]


def test_complete_result_is_frozen(
    result: DelayedMisleadingResponseProbeResult,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.final_state = result.initial_state  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_scalar_probe_amplitudes_are_deterministic(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
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


def test_early_assessments_are_deterministic(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
    ):
        left = assess_relation(
            build_bridge(),
            case,
            relation_id,
            observation_window_seconds=(
                EARLY_OBSERVATION_WINDOW_SECONDS
            ),
        )

        right = assess_relation(
            build_bridge(),
            case,
            relation_id,
            observation_window_seconds=(
                EARLY_OBSERVATION_WINDOW_SECONDS
            ),
        )

        assert left == right


def test_full_assessments_are_deterministic(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    for relation_id in (
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
    ):
        left = assess_relation(
            build_bridge(),
            case,
            relation_id,
            observation_window_seconds=(
                FULL_OBSERVATION_WINDOW_SECONDS
            ),
        )

        right = assess_relation(
            build_bridge(),
            case,
            relation_id,
            observation_window_seconds=(
                FULL_OBSERVATION_WINDOW_SECONDS
            ),
        )

        assert left == right


def _authorized_signature(
    result: DelayedMisleadingResponseProbeResult,
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


def test_complete_temporal_probe_pipeline_is_deterministic(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    left = run_delayed_misleading_response_probe(
        case
    )

    right = run_delayed_misleading_response_probe(
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
        left.early_fast_assessment
        == right.early_fast_assessment
    )

    assert (
        left.early_delayed_assessment
        == right.early_delayed_assessment
    )

    assert (
        left.early_state
        == right.early_state
    )

    assert (
        left.full_fast_assessment
        == right.full_fast_assessment
    )

    assert (
        left.full_delayed_assessment
        == right.full_delayed_assessment
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
