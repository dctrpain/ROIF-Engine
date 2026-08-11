"""
Tests for roif.experience_trace

The suite fixes the first Experience Trace contract.

It validates:
- immutable trace identity and payload;
- consistency between decision, candidate, prediction, and prediction error;
- stabilization outcome classification;
- descriptive stabilization summary;
- NO_SAFE_ACTION episodes without fake prediction error;
- selected-candidate resolution;
- audit metadata;
- the core invariant:

      ExperienceTrace != Learning
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from roif.experience_trace import (
    ExperienceActionKind,
    ExperienceTrace,
    ExperienceTraceIdentity,
    IncompleteExperienceEpisodeError,
    InvalidExperienceTraceError,
    StabilizationOutcome,
    build_experience_trace,
    build_stabilization_summary,
    classify_stabilization_outcome,
    find_selected_evaluation,
    improved_stabilization,
    prediction_error_norm,
    selected_candidate_id,
    trace_is_learning_free,
)
from roif.predictive_control import (
    CandidateEvaluation,
    ControlCandidate,
    ControlCandidateKind,
    ControlReserve,
    DisturbanceEstimate,
    PredictedOutcome,
    PredictiveControlDecision,
    PredictiveControlDecisionKind,
    PredictiveState,
    StabilizationDemand,
)


# =============================================================================
# Helpers
# =============================================================================


def initial_state(
    *,
    residual: float = 1.0,
    uncertainty: float = 0.4,
) -> PredictiveState:
    return PredictiveState(
        values={
            "x": residual,
        },
        target_values={
            "x": 0.0,
        },
        timestamp=0.0,
        uncertainty=uncertainty,
        metadata={
            "external_expected_label_used": False,
        },
    )


def observed_state(
    *,
    residual: float = 0.4,
    uncertainty: float = 0.2,
) -> PredictiveState:
    return PredictiveState(
        values={
            "x": residual,
        },
        target_values={
            "x": 0.0,
        },
        timestamp=1.0,
        uncertainty=uncertainty,
        metadata={
            "external_expected_label_used": False,
        },
    )


def disturbance() -> DisturbanceEstimate:
    return DisturbanceEstimate(
        components={
            "d": 0.5,
        },
        confidence=0.9,
        metadata={
            "external_expected_label_used": False,
        },
    )


def reserve() -> ControlReserve:
    return ControlReserve(
        capacity=2.0,
        committed=0.5,
        recoverable_fraction=0.1,
        metadata={
            "external_expected_label_used": False,
        },
    )


def demand() -> StabilizationDemand:
    return StabilizationDemand(
        state_error=1.0,
        disturbance_load=0.5,
        uncertainty_load=0.1,
        urgency=1.0,
        metadata={
            "external_expected_label_used": False,
        },
    )


def corrective_candidate(
    *,
    candidate_id: str = "corrective",
    cost: float = 0.2,
) -> ControlCandidate:
    return ControlCandidate(
        candidate_id=candidate_id,
        kind=ControlCandidateKind.CORRECTIVE,
        control_delta={
            "u": -0.5,
        },
        estimated_cost=cost,
        reversibility=0.9,
        safety_risk=0.05,
        uncertainty=0.1,
        metadata={
            "external_expected_label_used": False,
        },
    )


def probe_candidate(
    *,
    candidate_id: str = "probe",
    cost: float = 0.05,
) -> ControlCandidate:
    return ControlCandidate(
        candidate_id=candidate_id,
        kind=ControlCandidateKind.PROBE,
        control_delta={
            "u": -0.05,
        },
        estimated_cost=cost,
        reversibility=1.0,
        safety_risk=0.01,
        uncertainty=0.05,
        metadata={
            "external_expected_label_used": False,
        },
    )


def predicted_outcome(
    candidate: ControlCandidate,
    *,
    residual: float = 0.3,
    uncertainty: float = 0.2,
) -> PredictedOutcome:
    state = PredictiveState(
        values={
            "x": residual,
        },
        target_values={
            "x": 0.0,
        },
        timestamp=1.0,
        uncertainty=uncertainty,
        metadata={
            "external_expected_label_used": False,
        },
    )

    return PredictedOutcome(
        candidate_id=candidate.candidate_id,
        predicted_state=state,
        predicted_residual_error=residual,
        predicted_control_cost=candidate.estimated_cost,
        predicted_uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
        metadata={
            "external_expected_label_used": False,
        },
    )


def candidate_evaluation(
    candidate: ControlCandidate,
    *,
    residual: float = 0.3,
    uncertainty: float = 0.2,
    reserve_after: float = 1.2,
    margin: float = -0.4,
    score: float = 1.0,
    safe: bool = True,
    viable: bool = True,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=candidate,
        outcome=predicted_outcome(
            candidate,
            residual=residual,
            uncertainty=uncertainty,
        ),
        reserve_after_action=reserve_after,
        stabilization_margin=margin,
        score=score,
        safe=safe,
        viable=viable,
        metadata={
            "external_expected_label_used": False,
        },
    )


def action_decision(
    evaluation: CandidateEvaluation,
) -> PredictiveControlDecision:
    return PredictiveControlDecision(
        kind=PredictiveControlDecisionKind.ACTION,
        selected_candidate_id=evaluation.candidate.candidate_id,
        selected_evaluation=evaluation,
        ranked_evaluations=(
            evaluation,
        ),
        reason="test_action",
        metadata={
            "external_expected_label_used": False,
        },
    )


def probe_decision(
    evaluation: CandidateEvaluation,
) -> PredictiveControlDecision:
    return PredictiveControlDecision(
        kind=PredictiveControlDecisionKind.PROBE,
        selected_candidate_id=evaluation.candidate.candidate_id,
        selected_evaluation=evaluation,
        ranked_evaluations=(
            evaluation,
        ),
        reason="test_probe",
        metadata={
            "external_expected_label_used": False,
        },
    )


def no_safe_action_decision(
) -> PredictiveControlDecision:
    return PredictiveControlDecision(
        kind=PredictiveControlDecisionKind.NO_SAFE_ACTION,
        selected_candidate_id=None,
        selected_evaluation=None,
        ranked_evaluations=(),
        reason="no_viable_candidate",
        metadata={
            "external_expected_label_used": False,
        },
    )


def identity(
    *,
    trace_id: str = "trace_0001",
    episode_index: int = 1,
) -> ExperienceTraceIdentity:
    return ExperienceTraceIdentity(
        trace_id=trace_id,
        episode_index=episode_index,
        sequence_id="sequence_A",
        timestamp_start=0.0,
        timestamp_end=1.0,
        metadata={
            "external_expected_label_used": False,
        },
    )


def complete_trace(
    *,
    observed_residual: float = 0.4,
) -> ExperienceTrace:
    candidate = corrective_candidate()
    evaluation = candidate_evaluation(candidate)
    decision = action_decision(evaluation)

    return build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(
            candidate,
        ),
        evaluations=(
            evaluation,
        ),
        decision=decision,
        observed_state=observed_state(
            residual=observed_residual
        ),
        metadata={
            "source": "unit_test",
        },
    )


# =============================================================================
# Identity
# =============================================================================


def test_identity_accepts_valid_episode() -> None:
    item = identity()

    assert item.trace_id == "trace_0001"
    assert item.episode_index == 1


def test_identity_rejects_empty_trace_id() -> None:
    with pytest.raises(
        InvalidExperienceTraceError
    ):
        ExperienceTraceIdentity(
            trace_id="",
            episode_index=0,
        )


def test_identity_rejects_negative_episode_index() -> None:
    with pytest.raises(
        InvalidExperienceTraceError
    ):
        ExperienceTraceIdentity(
            trace_id="x",
            episode_index=-1,
        )


def test_identity_rejects_end_before_start() -> None:
    with pytest.raises(
        InvalidExperienceTraceError
    ):
        ExperienceTraceIdentity(
            trace_id="x",
            episode_index=0,
            timestamp_start=2.0,
            timestamp_end=1.0,
        )


def test_identity_metadata_is_read_only() -> None:
    item = identity()

    assert isinstance(
        item.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        item.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_identity_is_frozen() -> None:
    item = identity()

    with pytest.raises(
        FrozenInstanceError
    ):
        item.trace_id = "other"  # type: ignore[misc]


# =============================================================================
# Stabilization classification
# =============================================================================


def test_stabilization_outcome_improved() -> None:
    result = classify_stabilization_outcome(
        initial_state(
            residual=1.0
        ),
        observed_state(
            residual=0.5
        ),
    )

    assert (
        result
        is StabilizationOutcome.IMPROVED
    )


def test_stabilization_outcome_unchanged() -> None:
    result = classify_stabilization_outcome(
        initial_state(
            residual=1.0
        ),
        observed_state(
            residual=1.0
        ),
    )

    assert (
        result
        is StabilizationOutcome.UNCHANGED
    )


def test_stabilization_outcome_worsened() -> None:
    result = classify_stabilization_outcome(
        initial_state(
            residual=1.0
        ),
        observed_state(
            residual=1.5
        ),
    )

    assert (
        result
        is StabilizationOutcome.WORSENED
    )


def test_stabilization_outcome_respects_tolerance() -> None:
    result = classify_stabilization_outcome(
        initial_state(
            residual=1.0
        ),
        observed_state(
            residual=1.0 + 1e-13
        ),
        tolerance=1e-12,
    )

    assert (
        result
        is StabilizationOutcome.UNCHANGED
    )


# =============================================================================
# Stabilization summary
# =============================================================================


def test_stabilization_summary_records_before_predicted_observed() -> None:
    candidate = corrective_candidate()
    evaluation = candidate_evaluation(
        candidate,
        residual=0.3,
    )

    summary = build_stabilization_summary(
        initial_state=initial_state(
            residual=1.0
        ),
        observed_state=observed_state(
            residual=0.4
        ),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=evaluation,
    )

    assert summary.residual_before == pytest.approx(
        1.0
    )

    assert summary.residual_predicted == pytest.approx(
        0.3
    )

    assert summary.residual_observed == pytest.approx(
        0.4
    )


def test_stabilization_summary_residual_change_is_observed_minus_before() -> None:
    summary = build_stabilization_summary(
        initial_state=initial_state(
            residual=1.0
        ),
        observed_state=observed_state(
            residual=0.4
        ),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=None,
    )

    assert summary.residual_change == pytest.approx(
        -0.6
    )


def test_stabilization_summary_records_selected_action_cost() -> None:
    candidate = corrective_candidate(
        cost=0.25
    )

    evaluation = candidate_evaluation(
        candidate
    )

    summary = build_stabilization_summary(
        initial_state=initial_state(),
        observed_state=observed_state(),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=evaluation,
    )

    assert summary.selected_action_cost == pytest.approx(
        0.25
    )


def test_stabilization_summary_records_margin_and_reserve_after() -> None:
    candidate = corrective_candidate()

    evaluation = candidate_evaluation(
        candidate,
        reserve_after=1.1,
        margin=-0.7,
    )

    summary = build_stabilization_summary(
        initial_state=initial_state(),
        observed_state=observed_state(),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=evaluation,
    )

    assert summary.reserve_after_selected_action == pytest.approx(
        1.1
    )

    assert summary.stabilization_margin == pytest.approx(
        -0.7
    )


def test_stabilization_summary_without_selection_uses_zero_cost() -> None:
    summary = build_stabilization_summary(
        initial_state=initial_state(),
        observed_state=observed_state(),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=None,
    )

    assert summary.selected_action_cost == 0.0
    assert summary.residual_predicted is None
    assert summary.reserve_after_selected_action is None
    assert summary.stabilization_margin is None
    assert summary.uncertainty_predicted is None


def test_stabilization_summary_metadata_is_read_only() -> None:
    summary = build_stabilization_summary(
        initial_state=initial_state(),
        observed_state=observed_state(),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=None,
        metadata={
            "x": 1,
        },
    )

    with pytest.raises(TypeError):
        summary.metadata[
            "x"
        ] = 2  # type: ignore[index]


def test_stabilization_summary_is_frozen() -> None:
    summary = build_stabilization_summary(
        initial_state=initial_state(),
        observed_state=observed_state(),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=None,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        summary.residual_before = 9.0  # type: ignore[misc]


# =============================================================================
# Selected evaluation resolution
# =============================================================================


def test_find_selected_evaluation_returns_decision_embedded_evaluation() -> None:
    candidate = corrective_candidate()
    evaluation = candidate_evaluation(candidate)
    decision = action_decision(evaluation)

    resolved = find_selected_evaluation(
        decision,
        (
            evaluation,
        ),
    )

    assert resolved == evaluation


def test_find_selected_evaluation_can_resolve_by_id() -> None:
    candidate = corrective_candidate()
    evaluation = candidate_evaluation(candidate)

    decision = PredictiveControlDecision(
        kind=PredictiveControlDecisionKind.ACTION,
        selected_candidate_id=candidate.candidate_id,
        selected_evaluation=None,
        ranked_evaluations=(
            evaluation,
        ),
        reason="resolve_by_id",
    )

    resolved = find_selected_evaluation(
        decision,
        (
            evaluation,
        ),
    )

    assert resolved == evaluation


def test_find_selected_evaluation_returns_none_when_no_selection() -> None:
    assert (
        find_selected_evaluation(
            no_safe_action_decision(),
            (),
        )
        is None
    )


def test_find_selected_evaluation_rejects_embedded_mismatch() -> None:
    a = corrective_candidate(
        candidate_id="a"
    )
    b = corrective_candidate(
        candidate_id="b"
    )

    eval_a = candidate_evaluation(a)

    decision = PredictiveControlDecision(
        kind=PredictiveControlDecisionKind.ACTION,
        selected_candidate_id="b",
        selected_evaluation=eval_a,
        ranked_evaluations=(
            eval_a,
        ),
        reason="mismatch",
    )

    with pytest.raises(
        IncompleteExperienceEpisodeError
    ):
        find_selected_evaluation(
            decision,
            (
                eval_a,
            ),
        )


def test_find_selected_evaluation_requires_exactly_one_match() -> None:
    candidate = corrective_candidate()
    evaluation = candidate_evaluation(candidate)

    decision = PredictiveControlDecision(
        kind=PredictiveControlDecisionKind.ACTION,
        selected_candidate_id=candidate.candidate_id,
        selected_evaluation=None,
        ranked_evaluations=(),
        reason="duplicate_test",
    )

    with pytest.raises(
        IncompleteExperienceEpisodeError
    ):
        find_selected_evaluation(
            decision,
            (
                evaluation,
                evaluation,
            ),
        )


# =============================================================================
# Complete trace
# =============================================================================


def test_build_experience_trace_returns_trace() -> None:
    trace = complete_trace()

    assert isinstance(
        trace,
        ExperienceTrace,
    )


def test_complete_trace_preserves_identity() -> None:
    trace = complete_trace()

    assert trace.identity.trace_id == "trace_0001"


def test_complete_trace_preserves_candidate_set() -> None:
    trace = complete_trace()

    assert len(
        trace.candidates
    ) == 1

    assert (
        trace.candidates[
            0
        ].candidate_id
        == "corrective"
    )


def test_complete_trace_preserves_evaluations() -> None:
    trace = complete_trace()

    assert len(
        trace.evaluations
    ) == 1


def test_complete_trace_resolves_selected_candidate() -> None:
    trace = complete_trace()

    assert trace.selected_candidate is not None

    assert (
        trace.selected_candidate.candidate_id
        == "corrective"
    )


def test_complete_trace_preserves_selected_prediction() -> None:
    trace = complete_trace()

    assert trace.selected_prediction is not None

    assert (
        trace.selected_prediction.candidate_id
        == "corrective"
    )


def test_complete_trace_computes_prediction_error() -> None:
    trace = complete_trace()

    assert trace.prediction_error is not None
    assert trace.prediction_error.l2_error > 0.0


def test_prediction_error_matches_observed_minus_predicted() -> None:
    trace = complete_trace(
        observed_residual=0.4
    )

    assert trace.prediction_error is not None

    # default predicted residual is 0.3
    assert (
        trace.prediction_error.variable_errors[
            "x"
        ]
        == pytest.approx(
            0.1
        )
    )


def test_complete_trace_action_kind_matches_action_decision() -> None:
    trace = complete_trace()

    assert (
        trace.action_kind
        is ExperienceActionKind.ACTION
    )


def test_probe_decision_maps_to_probe_action_kind() -> None:
    candidate = probe_candidate()
    evaluation = candidate_evaluation(candidate)
    decision = probe_decision(evaluation)

    trace = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(
            candidate,
        ),
        evaluations=(
            evaluation,
        ),
        decision=decision,
        observed_state=observed_state(),
    )

    assert (
        trace.action_kind
        is ExperienceActionKind.PROBE
    )


def test_complete_trace_stabilization_is_improved() -> None:
    trace = complete_trace(
        observed_residual=0.4
    )

    assert (
        trace.stabilization.outcome
        is StabilizationOutcome.IMPROVED
    )


def test_complete_trace_stabilization_can_be_worsened() -> None:
    trace = complete_trace(
        observed_residual=1.5
    )

    assert (
        trace.stabilization.outcome
        is StabilizationOutcome.WORSENED
    )


# =============================================================================
# NO_SAFE_ACTION
# =============================================================================


def test_no_safe_action_trace_has_no_selected_candidate() -> None:
    trace = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(),
        evaluations=(),
        decision=no_safe_action_decision(),
        observed_state=observed_state(
            residual=1.0
        ),
    )

    assert trace.selected_candidate is None


def test_no_safe_action_trace_has_no_selected_prediction() -> None:
    trace = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(),
        evaluations=(),
        decision=no_safe_action_decision(),
        observed_state=observed_state(
            residual=1.0
        ),
    )

    assert trace.selected_prediction is None


def test_no_safe_action_trace_has_no_fake_prediction_error() -> None:
    trace = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(),
        evaluations=(),
        decision=no_safe_action_decision(),
        observed_state=observed_state(
            residual=1.0
        ),
    )

    assert trace.prediction_error is None


def test_no_safe_action_maps_to_no_safe_action_kind() -> None:
    trace = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(),
        evaluations=(),
        decision=no_safe_action_decision(),
        observed_state=observed_state(
            residual=1.0
        ),
    )

    assert (
        trace.action_kind
        is ExperienceActionKind.NO_SAFE_ACTION
    )


# =============================================================================
# Trace consistency guards
# =============================================================================


def test_trace_rejects_action_kind_mismatch() -> None:
    trace = complete_trace()

    with pytest.raises(
        InvalidExperienceTraceError
    ):
        replace(
            trace,
            action_kind=ExperienceActionKind.PROBE,
        )


def test_trace_rejects_selected_candidate_when_decision_has_no_selection() -> None:
    candidate = corrective_candidate()

    base = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(),
        evaluations=(),
        decision=no_safe_action_decision(),
        observed_state=observed_state(),
    )

    with pytest.raises(
        InvalidExperienceTraceError
    ):
        replace(
            base,
            selected_candidate=candidate,
        )


def test_trace_rejects_selected_candidate_id_mismatch() -> None:
    trace = complete_trace()

    other = corrective_candidate(
        candidate_id="other"
    )

    with pytest.raises(
        InvalidExperienceTraceError
    ):
        replace(
            trace,
            selected_candidate=other,
        )


def test_trace_rejects_selected_prediction_mismatch() -> None:
    trace = complete_trace()

    other_candidate = corrective_candidate(
        candidate_id="other"
    )

    other_prediction = predicted_outcome(
        other_candidate
    )

    with pytest.raises(
        InvalidExperienceTraceError
    ):
        replace(
            trace,
            selected_prediction=other_prediction,
        )


def test_trace_rejects_missing_prediction_error_for_selected_candidate() -> None:
    trace = complete_trace()

    with pytest.raises(
        InvalidExperienceTraceError
    ):
        replace(
            trace,
            prediction_error=None,
        )


# =============================================================================
# Audit / no-learning invariant
# =============================================================================


def test_trace_metadata_marks_experience_trace_generated() -> None:
    trace = complete_trace()

    assert (
        trace.metadata[
            "experience_trace_generated"
        ]
        is True
    )


def test_trace_metadata_declares_learning_not_applied() -> None:
    trace = complete_trace()

    assert (
        trace.metadata[
            "learning_applied"
        ]
        is False
    )


def test_trace_metadata_declares_controller_memory_not_mutated() -> None:
    trace = complete_trace()

    assert (
        trace.metadata[
            "controller_memory_mutated"
        ]
        is False
    )


def test_trace_metadata_declares_predictive_preload_not_modified() -> None:
    trace = complete_trace()

    assert (
        trace.metadata[
            "predictive_preload_modified"
        ]
        is False
    )


def test_trace_metadata_declares_graph_not_mutated() -> None:
    trace = complete_trace()

    assert (
        trace.metadata[
            "graph_mutated"
        ]
        is False
    )


def test_trace_metadata_declares_no_external_expected_label_usage() -> None:
    trace = complete_trace()

    assert (
        trace.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_prediction_error_audit_declares_learning_not_applied() -> None:
    trace = complete_trace()

    assert trace.prediction_error is not None

    assert (
        trace.prediction_error.metadata[
            "learning_applied"
        ]
        is False
    )


def test_stabilization_summary_audit_declares_memory_not_mutated() -> None:
    trace = complete_trace()

    assert (
        trace.stabilization.metadata[
            "controller_memory_mutated"
        ]
        is False
    )


def test_trace_is_learning_free_returns_true() -> None:
    trace = complete_trace()

    assert trace_is_learning_free(
        trace
    ) is True


def test_trace_is_learning_free_detects_forbidden_learning_flag() -> None:
    trace = complete_trace()

    altered = replace(
        trace,
        metadata={
            **dict(
                trace.metadata
            ),
            "learning_applied": True,
        },
    )

    assert trace_is_learning_free(
        altered
    ) is False


# =============================================================================
# Inspection helpers
# =============================================================================


def test_selected_candidate_id_helper() -> None:
    trace = complete_trace()

    assert selected_candidate_id(
        trace
    ) == "corrective"


def test_selected_candidate_id_helper_returns_none_without_selection() -> None:
    trace = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(),
        evaluations=(),
        decision=no_safe_action_decision(),
        observed_state=observed_state(),
    )

    assert selected_candidate_id(
        trace
    ) is None


def test_prediction_error_norm_helper() -> None:
    trace = complete_trace()

    assert prediction_error_norm(
        trace
    ) == pytest.approx(
        trace.prediction_error.l2_error
    )


def test_prediction_error_norm_helper_returns_none_without_selection() -> None:
    trace = build_experience_trace(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(),
        evaluations=(),
        decision=no_safe_action_decision(),
        observed_state=observed_state(),
    )

    assert prediction_error_norm(
        trace
    ) is None


def test_improved_stabilization_helper_true() -> None:
    trace = complete_trace(
        observed_residual=0.4
    )

    assert improved_stabilization(
        trace
    ) is True


def test_improved_stabilization_helper_false_when_worsened() -> None:
    trace = complete_trace(
        observed_residual=1.4
    )

    assert improved_stabilization(
        trace
    ) is False


# =============================================================================
# Immutability
# =============================================================================


def test_trace_metadata_is_read_only() -> None:
    trace = complete_trace()

    assert isinstance(
        trace.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        trace.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_trace_candidates_are_tuple() -> None:
    trace = complete_trace()

    assert isinstance(
        trace.candidates,
        tuple,
    )


def test_trace_evaluations_are_tuple() -> None:
    trace = complete_trace()

    assert isinstance(
        trace.evaluations,
        tuple,
    )


def test_trace_is_frozen() -> None:
    trace = complete_trace()

    with pytest.raises(
        FrozenInstanceError
    ):
        trace.action_kind = ExperienceActionKind.HOLD  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_stabilization_classification_is_deterministic() -> None:
    before = initial_state(
        residual=1.0
    )

    after = observed_state(
        residual=0.4
    )

    left = classify_stabilization_outcome(
        before,
        after,
    )

    right = classify_stabilization_outcome(
        before,
        after,
    )

    assert left == right


def test_stabilization_summary_is_deterministic() -> None:
    candidate = corrective_candidate()
    evaluation = candidate_evaluation(candidate)

    kwargs = dict(
        initial_state=initial_state(),
        observed_state=observed_state(),
        demand=demand(),
        reserve=reserve(),
        selected_evaluation=evaluation,
    )

    left = build_stabilization_summary(
        **kwargs
    )

    right = build_stabilization_summary(
        **kwargs
    )

    assert left == right


def test_complete_experience_trace_is_deterministic() -> None:
    candidate = corrective_candidate()
    evaluation = candidate_evaluation(candidate)
    decision = action_decision(evaluation)

    kwargs = dict(
        identity=identity(),
        initial_state=initial_state(),
        disturbance=disturbance(),
        demand=demand(),
        reserve=reserve(),
        candidates=(
            candidate,
        ),
        evaluations=(
            evaluation,
        ),
        decision=decision,
        observed_state=observed_state(),
        metadata={
            "source": "unit_test",
        },
    )

    left = build_experience_trace(
        **kwargs
    )

    right = build_experience_trace(
        **kwargs
    )

    assert left == right
