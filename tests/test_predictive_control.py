"""
Tests for roif.predictive_control

The suite fixes the first predictive-control contract before integration with
Scenario 04A.

It validates:
- immutable state objects;
- residual/state-error calculation;
- disturbance magnitude;
- control reserve and utilization;
- stabilization demand;
- candidate/outcome evaluation;
- deterministic ranking;
- ACTION / PROBE / HOLD / NO_SAFE_ACTION decisions;
- uncertainty-driven Probe behavior;
- prediction-error comparison;
- no hidden expected-label dependency.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.predictive_control import (
    CandidateEvaluation,
    ControlCandidate,
    ControlCandidateKind,
    ControlReserve,
    DisturbanceEstimate,
    InvalidControlCandidateError,
    InvalidPredictiveStateError,
    PredictionError,
    PredictedOutcome,
    PredictiveControlConfig,
    PredictiveControlDecisionKind,
    PredictiveControlMode,
    PredictiveState,
    StabilizationDemand,
    compare_prediction,
    estimate_stabilization_demand,
    evaluate_candidate,
    evaluate_control_candidates,
    predict_outcome,
    rank_candidate_evaluations,
    select_control_action,
)


# =============================================================================
# Helpers
# =============================================================================


def base_state(
    *,
    error: float = 0.4,
    uncertainty: float = 0.1,
) -> PredictiveState:
    return PredictiveState(
        values={
            "course_error": error,
            "heel_error": 0.2,
        },
        target_values={
            "course_error": 0.0,
            "heel_error": 0.0,
        },
        timestamp=1.0,
        uncertainty=uncertainty,
        metadata={
            "expected_label_used": False,
        },
    )


def base_disturbance() -> DisturbanceEstimate:
    return DisturbanceEstimate(
        components={
            "wind": 0.3,
            "wave": 0.4,
        },
        confidence=0.8,
        metadata={
            "expected_label_used": False,
        },
    )


def base_reserve() -> ControlReserve:
    return ControlReserve(
        capacity=2.0,
        committed=0.5,
        recoverable_fraction=0.2,
        metadata={
            "expected_label_used": False,
        },
    )


def corrective_candidate(
    *,
    candidate_id: str = "corrective",
    estimated_cost: float = 0.2,
    uncertainty: float = 0.1,
    safety_risk: float = 0.05,
) -> ControlCandidate:
    return ControlCandidate(
        candidate_id=candidate_id,
        kind=ControlCandidateKind.CORRECTIVE,
        control_delta={
            "rudder": -0.2,
        },
        estimated_cost=estimated_cost,
        reversibility=0.9,
        safety_risk=safety_risk,
        uncertainty=uncertainty,
        metadata={
            "expected_label_used": False,
        },
    )


def probe_candidate(
    *,
    candidate_id: str = "probe",
    estimated_cost: float = 0.05,
    reversibility: float = 1.0,
    safety_risk: float = 0.01,
) -> ControlCandidate:
    return ControlCandidate(
        candidate_id=candidate_id,
        kind=ControlCandidateKind.PROBE,
        control_delta={
            "rudder": -0.02,
        },
        estimated_cost=estimated_cost,
        reversibility=reversibility,
        safety_risk=safety_risk,
        uncertainty=0.0,
        metadata={
            "expected_label_used": False,
        },
    )


def hold_candidate() -> ControlCandidate:
    return ControlCandidate(
        candidate_id="hold",
        kind=ControlCandidateKind.HOLD,
        control_delta={},
        estimated_cost=0.0,
        reversibility=1.0,
        safety_risk=0.0,
        uncertainty=0.0,
    )


def simple_prediction_model(
    state: PredictiveState,
    disturbance: DisturbanceEstimate,
    candidate: ControlCandidate,
) -> PredictiveState:
    course = state.values["course_error"]
    heel = state.values["heel_error"]

    disturbance_push = (
        disturbance.components.get("wind", 0.0)
        + disturbance.components.get("wave", 0.0)
    ) * 0.1

    action = candidate.control_delta.get(
        "rudder",
        0.0,
    )

    if candidate.kind is ControlCandidateKind.PROBE:
        action_effect = action * 0.2
        uncertainty = max(
            0.0,
            state.uncertainty - 0.30,
        )
    elif candidate.kind is ControlCandidateKind.HOLD:
        action_effect = 0.0
        uncertainty = state.uncertainty
    else:
        action_effect = action
        uncertainty = max(
            0.0,
            state.uncertainty - 0.05,
        )

    return PredictiveState(
        values={
            "course_error": (
                course
                + disturbance_push
                + action_effect
            ),
            "heel_error": heel,
        },
        target_values=dict(
            state.target_values
        ),
        timestamp=(
            None
            if state.timestamp is None
            else state.timestamp + 1.0
        ),
        uncertainty=uncertainty,
        metadata={
            "model": "simple_test_model",
            "expected_label_used": False,
        },
    )


def evaluation(
    candidate: ControlCandidate,
    *,
    residual: float,
    uncertainty: float,
    score_config: PredictiveControlConfig | None = None,
    reserve: ControlReserve | None = None,
    demand: StabilizationDemand | None = None,
) -> CandidateEvaluation:
    predicted_state = PredictiveState(
        values={
            "x": residual,
        },
        target_values={
            "x": 0.0,
        },
        uncertainty=uncertainty,
    )

    outcome = PredictedOutcome(
        candidate_id=candidate.candidate_id,
        predicted_state=predicted_state,
        predicted_residual_error=residual,
        predicted_control_cost=candidate.estimated_cost,
        predicted_uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
    )

    return evaluate_candidate(
        candidate,
        outcome,
        reserve
        or ControlReserve(
            capacity=5.0,
            committed=0.5,
        ),
        demand
        or StabilizationDemand(
            state_error=0.2,
            disturbance_load=0.2,
        ),
        config=score_config,
    )


# =============================================================================
# PredictiveState
# =============================================================================


def test_predictive_state_residuals_are_exact() -> None:
    state = PredictiveState(
        values={
            "a": 3.0,
            "b": -1.0,
        },
        target_values={
            "a": 1.0,
            "b": 1.0,
        },
    )

    assert state.residuals() == {
        "a": 2.0,
        "b": -2.0,
    }


def test_predictive_state_residual_norm_is_euclidean() -> None:
    state = PredictiveState(
        values={
            "a": 3.0,
            "b": 4.0,
        },
        target_values={
            "a": 0.0,
            "b": 0.0,
        },
    )

    assert state.residual_norm() == pytest.approx(
        5.0
    )


def test_predictive_state_rejects_empty_values() -> None:
    with pytest.raises(
        InvalidPredictiveStateError
    ):
        PredictiveState(
            values={},
            target_values={},
        )


def test_predictive_state_rejects_unknown_target_variable() -> None:
    with pytest.raises(
        InvalidPredictiveStateError
    ):
        PredictiveState(
            values={
                "a": 1.0,
            },
            target_values={
                "missing": 0.0,
            },
        )


def test_predictive_state_uncertainty_is_clamped() -> None:
    high = PredictiveState(
        values={
            "x": 1.0,
        },
        target_values={
            "x": 0.0,
        },
        uncertainty=2.0,
    )

    low = PredictiveState(
        values={
            "x": 1.0,
        },
        target_values={
            "x": 0.0,
        },
        uncertainty=-1.0,
    )

    assert high.uncertainty == 1.0
    assert low.uncertainty == 0.0


def test_predictive_state_values_are_read_only() -> None:
    state = base_state()

    assert isinstance(
        state.values,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        state.values[
            "course_error"
        ] = 0.0  # type: ignore[index]


def test_predictive_state_targets_are_read_only() -> None:
    state = base_state()

    assert isinstance(
        state.target_values,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        state.target_values[
            "course_error"
        ] = 1.0  # type: ignore[index]


def test_predictive_state_metadata_is_read_only() -> None:
    state = base_state()

    assert isinstance(
        state.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        state.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_predictive_state_is_frozen() -> None:
    state = base_state()

    with pytest.raises(
        FrozenInstanceError
    ):
        state.uncertainty = 0.9  # type: ignore[misc]


# =============================================================================
# Disturbance
# =============================================================================


def test_disturbance_magnitude_is_euclidean() -> None:
    disturbance = DisturbanceEstimate(
        components={
            "x": 3.0,
            "y": 4.0,
        }
    )

    assert disturbance.magnitude == pytest.approx(
        5.0
    )


def test_disturbance_confidence_is_clamped() -> None:
    disturbance = DisturbanceEstimate(
        components={
            "x": 1.0,
        },
        confidence=2.0,
    )

    assert disturbance.confidence == 1.0


def test_disturbance_components_are_read_only() -> None:
    disturbance = base_disturbance()

    with pytest.raises(TypeError):
        disturbance.components[
            "wind"
        ] = 2.0  # type: ignore[index]


# =============================================================================
# ControlReserve
# =============================================================================


def test_control_reserve_available_includes_recoverable_fraction() -> None:
    reserve = ControlReserve(
        capacity=10.0,
        committed=4.0,
        recoverable_fraction=0.5,
    )

    assert reserve.available == pytest.approx(
        8.0
    )


def test_control_reserve_utilization() -> None:
    reserve = ControlReserve(
        capacity=10.0,
        committed=4.0,
    )

    assert reserve.utilization == pytest.approx(
        0.4
    )


def test_zero_capacity_reserve_is_fully_utilized() -> None:
    reserve = ControlReserve(
        capacity=0.0,
        committed=0.0,
    )

    assert reserve.utilization == 1.0


def test_control_reserve_rejects_committed_above_capacity() -> None:
    with pytest.raises(
        ValueError
    ):
        ControlReserve(
            capacity=1.0,
            committed=2.0,
        )


def test_control_reserve_rejects_negative_capacity() -> None:
    with pytest.raises(
        ValueError
    ):
        ControlReserve(
            capacity=-1.0,
            committed=0.0,
        )


# =============================================================================
# StabilizationDemand
# =============================================================================


def test_stabilization_demand_total() -> None:
    demand = StabilizationDemand(
        state_error=1.0,
        disturbance_load=2.0,
        uncertainty_load=0.5,
        urgency=2.0,
    )

    assert demand.total == pytest.approx(
        7.0
    )


def test_stabilization_demand_reserve_ratio() -> None:
    demand = StabilizationDemand(
        state_error=1.0,
        disturbance_load=1.0,
    )

    reserve = ControlReserve(
        capacity=5.0,
        committed=1.0,
    )

    assert demand.reserve_ratio(
        reserve
    ) == pytest.approx(
        2.0
    )


def test_zero_demand_has_infinite_reserve_ratio() -> None:
    demand = StabilizationDemand(
        state_error=0.0,
        disturbance_load=0.0,
    )

    assert demand.reserve_ratio(
        base_reserve()
    ) == float("inf")


def test_estimate_stabilization_demand_uses_state_disturbance_and_uncertainty() -> None:
    state = PredictiveState(
        values={
            "x": 3.0,
            "y": 4.0,
        },
        target_values={
            "x": 0.0,
            "y": 0.0,
        },
        uncertainty=0.25,
    )

    disturbance = DisturbanceEstimate(
        components={
            "d": 2.0,
        },
        confidence=0.5,
    )

    demand = estimate_stabilization_demand(
        state,
        disturbance,
        uncertainty_weight=2.0,
        urgency=1.0,
    )

    assert demand.state_error == pytest.approx(
        5.0
    )

    assert demand.disturbance_load == pytest.approx(
        1.0
    )

    assert demand.uncertainty_load == pytest.approx(
        0.5
    )

    assert demand.total == pytest.approx(
        6.5
    )


# =============================================================================
# ControlCandidate
# =============================================================================


def test_control_candidate_rejects_empty_id() -> None:
    with pytest.raises(
        InvalidControlCandidateError
    ):
        ControlCandidate(
            candidate_id="",
            kind=ControlCandidateKind.CORRECTIVE,
            control_delta={},
            estimated_cost=0.0,
        )


def test_control_candidate_kind_is_normalized() -> None:
    candidate = ControlCandidate(
        candidate_id="x",
        kind="corrective",
        control_delta={},
        estimated_cost=0.0,
    )

    assert (
        candidate.kind
        is ControlCandidateKind.CORRECTIVE
    )


def test_control_candidate_probabilities_are_clamped() -> None:
    candidate = ControlCandidate(
        candidate_id="x",
        kind=ControlCandidateKind.CORRECTIVE,
        control_delta={},
        estimated_cost=0.0,
        reversibility=2.0,
        safety_risk=3.0,
        uncertainty=-1.0,
    )

    assert candidate.reversibility == 1.0
    assert candidate.safety_risk == 1.0
    assert candidate.uncertainty == 0.0


def test_control_candidate_control_delta_is_read_only() -> None:
    candidate = corrective_candidate()

    with pytest.raises(TypeError):
        candidate.control_delta[
            "rudder"
        ] = 1.0  # type: ignore[index]


# =============================================================================
# Prediction
# =============================================================================


def test_predict_outcome_returns_expected_contract() -> None:
    outcome = predict_outcome(
        base_state(),
        base_disturbance(),
        corrective_candidate(),
        model=simple_prediction_model,
    )

    assert isinstance(
        outcome,
        PredictedOutcome,
    )

    assert (
        outcome.candidate_id
        == "corrective"
    )

    assert (
        outcome.predicted_residual_error
        == pytest.approx(
            outcome.predicted_state.residual_norm()
        )
    )


def test_corrective_prediction_reduces_course_error_relative_to_hold() -> None:
    state = base_state()
    disturbance = base_disturbance()

    corrective = predict_outcome(
        state,
        disturbance,
        corrective_candidate(),
        model=simple_prediction_model,
    )

    hold = predict_outcome(
        state,
        disturbance,
        hold_candidate(),
        model=simple_prediction_model,
    )

    assert (
        corrective.predicted_state.values[
            "course_error"
        ]
        <
        hold.predicted_state.values[
            "course_error"
        ]
    )


def test_probe_prediction_reduces_uncertainty() -> None:
    state = base_state(
        uncertainty=0.8
    )

    outcome = predict_outcome(
        state,
        base_disturbance(),
        probe_candidate(),
        model=simple_prediction_model,
    )

    assert (
        outcome.predicted_uncertainty
        < state.uncertainty
    )


def test_predict_outcome_rejects_non_predictive_state_model_result() -> None:
    def bad_model(
        state,
        disturbance,
        candidate,
    ):
        return {
            "x": 1.0,
        }

    from roif.predictive_control import (
        PredictiveControlError,
    )

    with pytest.raises(
        PredictiveControlError
    ):
        predict_outcome(
            base_state(),
            base_disturbance(),
            corrective_candidate(),
            model=bad_model,
        )


# =============================================================================
# Candidate evaluation
# =============================================================================


def test_candidate_evaluation_reserve_after_action() -> None:
    reserve = ControlReserve(
        capacity=2.0,
        committed=0.5,
    )

    demand = StabilizationDemand(
        state_error=0.2,
        disturbance_load=0.2,
    )

    candidate = corrective_candidate(
        estimated_cost=0.3
    )

    outcome = predict_outcome(
        base_state(),
        base_disturbance(),
        candidate,
        model=simple_prediction_model,
    )

    result = evaluate_candidate(
        candidate,
        outcome,
        reserve,
        demand,
    )

    assert result.reserve_after_action == pytest.approx(
        1.2
    )


def test_candidate_is_safe_under_threshold() -> None:
    candidate = corrective_candidate(
        safety_risk=0.1
    )

    result = evaluation(
        candidate,
        residual=0.1,
        uncertainty=0.1,
        score_config=PredictiveControlConfig(
            max_safety_risk=0.2,
            min_stabilization_margin=-10.0,
        ),
    )

    assert result.safe is True


def test_candidate_is_unsafe_above_threshold() -> None:
    candidate = corrective_candidate(
        safety_risk=0.8
    )

    result = evaluation(
        candidate,
        residual=0.1,
        uncertainty=0.1,
        score_config=PredictiveControlConfig(
            max_safety_risk=0.2,
            min_stabilization_margin=-10.0,
        ),
    )

    assert result.safe is False
    assert result.viable is False


def test_candidate_is_not_viable_if_reserve_exhausted() -> None:
    candidate = corrective_candidate(
        estimated_cost=1.0
    )

    result = evaluation(
        candidate,
        residual=0.1,
        uncertainty=0.1,
        reserve=ControlReserve(
            capacity=1.0,
            committed=0.2,
        ),
        demand=StabilizationDemand(
            state_error=0.0,
            disturbance_load=0.0,
        ),
        score_config=PredictiveControlConfig(
            min_stabilization_margin=-10.0,
        ),
    )

    assert result.viable is False


def test_candidate_is_not_viable_if_stabilization_margin_is_negative() -> None:
    candidate = corrective_candidate(
        estimated_cost=0.1
    )

    result = evaluation(
        candidate,
        residual=0.1,
        uncertainty=0.1,
        reserve=ControlReserve(
            capacity=1.0,
            committed=0.1,
        ),
        demand=StabilizationDemand(
            state_error=0.5,
            disturbance_load=0.5,
        ),
        score_config=PredictiveControlConfig(
            min_stabilization_margin=0.0,
        ),
    )

    assert result.stabilization_margin < 0.0
    assert result.viable is False


def test_lower_predicted_residual_can_improve_score() -> None:
    config = PredictiveControlConfig(
        residual_weight=2.0,
        cost_weight=0.0,
        uncertainty_weight=0.0,
        reserve_weight=0.0,
        safety_weight=0.0,
        min_stabilization_margin=-10.0,
    )

    better = evaluation(
        corrective_candidate(
            candidate_id="better"
        ),
        residual=0.1,
        uncertainty=0.1,
        score_config=config,
    )

    worse = evaluation(
        corrective_candidate(
            candidate_id="worse"
        ),
        residual=0.8,
        uncertainty=0.1,
        score_config=config,
    )

    assert better.score > worse.score


# =============================================================================
# Ranking
# =============================================================================


def test_ranking_places_viable_before_nonviable() -> None:
    config = PredictiveControlConfig(
        min_stabilization_margin=-10.0,
    )

    viable = evaluation(
        corrective_candidate(
            candidate_id="viable",
            safety_risk=0.0,
        ),
        residual=0.5,
        uncertainty=0.1,
        score_config=config,
    )

    unsafe = evaluation(
        corrective_candidate(
            candidate_id="unsafe",
            safety_risk=1.0,
        ),
        residual=0.0,
        uncertainty=0.0,
        score_config=config,
    )

    ranked = rank_candidate_evaluations(
        (
            unsafe,
            viable,
        )
    )

    assert (
        ranked[
            0
        ].candidate.candidate_id
        == "viable"
    )


def test_ranking_uses_candidate_id_as_deterministic_tiebreak() -> None:
    config = PredictiveControlConfig(
        min_stabilization_margin=-10.0,
    )

    a = evaluation(
        corrective_candidate(
            candidate_id="a"
        ),
        residual=0.1,
        uncertainty=0.1,
        score_config=config,
    )

    b = evaluation(
        corrective_candidate(
            candidate_id="b"
        ),
        residual=0.1,
        uncertainty=0.1,
        score_config=config,
    )

    ranked = rank_candidate_evaluations(
        (
            b,
            a,
        )
    )

    assert [
        item.candidate.candidate_id
        for item
        in ranked
    ] == [
        "a",
        "b",
    ]


# =============================================================================
# Decision selection
# =============================================================================


def test_no_candidates_returns_hold() -> None:
    decision = select_control_action(
        ()
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.HOLD
    )

    assert (
        decision.reason
        == "no_candidates"
    )


def test_observe_only_mode_returns_hold() -> None:
    item = evaluation(
        corrective_candidate(),
        residual=0.1,
        uncertainty=0.0,
        score_config=PredictiveControlConfig(
            min_stabilization_margin=-10.0,
        ),
    )

    decision = select_control_action(
        (
            item,
        ),
        config=PredictiveControlConfig(
            mode=PredictiveControlMode.OBSERVE_ONLY,
            min_stabilization_margin=-10.0,
        ),
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.HOLD
    )

    assert (
        decision.reason
        == "observe_only_mode"
    )


def test_best_viable_corrective_candidate_returns_action() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.STABILIZE,
        min_stabilization_margin=-10.0,
    )

    item = evaluation(
        corrective_candidate(),
        residual=0.1,
        uncertainty=0.1,
        score_config=config,
    )

    decision = select_control_action(
        (
            item,
        ),
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.ACTION
    )

    assert (
        decision.selected_candidate_id
        == "corrective"
    )


def test_best_viable_hold_candidate_returns_hold() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.STABILIZE,
        min_stabilization_margin=-10.0,
    )

    item = evaluation(
        hold_candidate(),
        residual=0.0,
        uncertainty=0.0,
        score_config=config,
    )

    decision = select_control_action(
        (
            item,
        ),
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.HOLD
    )


def test_high_uncertainty_can_force_probe_over_corrective_action() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        uncertainty_probe_threshold=0.35,
        min_stabilization_margin=-10.0,
    )

    corrective = evaluation(
        corrective_candidate(
            candidate_id="strong_action"
        ),
        residual=0.05,
        uncertainty=0.8,
        score_config=config,
    )

    probe = evaluation(
        probe_candidate(),
        residual=0.4,
        uncertainty=0.1,
        score_config=config,
    )

    decision = select_control_action(
        (
            corrective,
            probe,
        ),
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.PROBE
    )

    assert (
        decision.selected_candidate_id
        == "probe"
    )

    assert (
        decision.reason
        == "uncertainty_requires_probe"
    )


def test_low_uncertainty_allows_best_corrective_action() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        uncertainty_probe_threshold=0.35,
        min_stabilization_margin=-10.0,
    )

    corrective = evaluation(
        corrective_candidate(
            candidate_id="action"
        ),
        residual=0.05,
        uncertainty=0.1,
        score_config=config,
    )

    probe = evaluation(
        probe_candidate(),
        residual=0.5,
        uncertainty=0.0,
        score_config=config,
    )

    decision = select_control_action(
        (
            corrective,
            probe,
        ),
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.ACTION
    )

    assert (
        decision.selected_candidate_id
        == "action"
    )


def test_no_viable_candidate_and_no_probe_returns_no_safe_action() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        max_safety_risk=0.1,
        min_stabilization_margin=0.0,
    )

    unsafe = evaluation(
        corrective_candidate(
            candidate_id="unsafe",
            safety_risk=1.0,
        ),
        residual=0.1,
        uncertainty=0.1,
        score_config=config,
    )

    decision = select_control_action(
        (
            unsafe,
        ),
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.NO_SAFE_ACTION
    )

    assert (
        decision.selected_candidate_id
        is None
    )


def test_no_viable_corrective_but_safe_probe_can_return_probe() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        max_safety_risk=0.2,
        min_stabilization_margin=10.0,
    )

    corrective = evaluation(
        corrective_candidate(
            candidate_id="not_viable"
        ),
        residual=0.1,
        uncertainty=0.8,
        score_config=config,
    )

    probe = evaluation(
        probe_candidate(),
        residual=0.5,
        uncertainty=0.1,
        score_config=config,
    )

    decision = select_control_action(
        (
            corrective,
            probe,
        ),
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.PROBE
    )

    assert (
        decision.reason
        == "no_viable_corrective_action_probe_available"
    )


def test_irreversible_probe_is_not_selected_as_fallback() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        min_reversibility_for_probe=0.8,
        min_stabilization_margin=10.0,
    )

    probe = evaluation(
        probe_candidate(
            reversibility=0.2
        ),
        residual=0.5,
        uncertainty=0.1,
        score_config=config,
    )

    decision = select_control_action(
        (
            probe,
        ),
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.NO_SAFE_ACTION
    )


# =============================================================================
# Prediction error
# =============================================================================


def test_compare_prediction_zero_error_for_identical_state() -> None:
    predicted = PredictiveState(
        values={
            "x": 1.0,
        },
        target_values={
            "x": 0.0,
        },
    )

    outcome = PredictedOutcome(
        candidate_id="x",
        predicted_state=predicted,
        predicted_residual_error=1.0,
        predicted_control_cost=0.0,
        predicted_uncertainty=0.0,
        confidence=1.0,
    )

    error = compare_prediction(
        outcome,
        predicted,
    )

    assert isinstance(
        error,
        PredictionError,
    )

    assert error.l2_error == 0.0
    assert error.normalized_error == 0.0
    assert error.confidence == 1.0


def test_compare_prediction_variable_error_sign_is_observed_minus_predicted() -> None:
    predicted = PredictiveState(
        values={
            "x": 1.0,
        },
        target_values={
            "x": 0.0,
        },
    )

    observed = PredictiveState(
        values={
            "x": 1.5,
        },
        target_values={
            "x": 0.0,
        },
    )

    outcome = PredictedOutcome(
        candidate_id="x",
        predicted_state=predicted,
        predicted_residual_error=1.0,
        predicted_control_cost=0.0,
        predicted_uncertainty=0.0,
        confidence=1.0,
    )

    error = compare_prediction(
        outcome,
        observed,
    )

    assert error.variable_errors[
        "x"
    ] == pytest.approx(
        0.5
    )


def test_compare_prediction_uses_only_shared_variables() -> None:
    predicted = PredictiveState(
        values={
            "x": 1.0,
            "predicted_only": 2.0,
        },
        target_values={
            "x": 0.0,
        },
    )

    observed = PredictiveState(
        values={
            "x": 1.2,
            "observed_only": 3.0,
        },
        target_values={
            "x": 0.0,
        },
    )

    outcome = PredictedOutcome(
        candidate_id="x",
        predicted_state=predicted,
        predicted_residual_error=1.0,
        predicted_control_cost=0.0,
        predicted_uncertainty=0.0,
        confidence=1.0,
    )

    error = compare_prediction(
        outcome,
        observed,
    )

    assert set(
        error.variable_errors
    ) == {
        "x",
    }


def test_compare_prediction_rejects_no_shared_variables() -> None:
    from roif.predictive_control import (
        PredictiveControlError,
    )

    predicted = PredictiveState(
        values={
            "x": 1.0,
        },
        target_values={},
    )

    observed = PredictiveState(
        values={
            "y": 1.0,
        },
        target_values={},
    )

    outcome = PredictedOutcome(
        candidate_id="x",
        predicted_state=predicted,
        predicted_residual_error=0.0,
        predicted_control_cost=0.0,
        predicted_uncertainty=0.0,
        confidence=1.0,
    )

    with pytest.raises(
        PredictiveControlError
    ):
        compare_prediction(
            outcome,
            observed,
        )


def test_prediction_error_metadata_is_read_only() -> None:
    predicted = PredictiveState(
        values={
            "x": 1.0,
        },
        target_values={
            "x": 0.0,
        },
    )

    outcome = PredictedOutcome(
        candidate_id="x",
        predicted_state=predicted,
        predicted_residual_error=1.0,
        predicted_control_cost=0.0,
        predicted_uncertainty=0.0,
        confidence=1.0,
    )

    error = compare_prediction(
        outcome,
        predicted,
        metadata={
            "expected_label_used": False,
        },
    )

    with pytest.raises(TypeError):
        error.metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# End-to-end evaluation
# =============================================================================


def test_evaluate_control_candidates_returns_demand_and_ranked_results() -> None:
    state = base_state(
        uncertainty=0.1
    )

    demand, ranked = evaluate_control_candidates(
        state,
        base_disturbance(),
        (
            corrective_candidate(),
            hold_candidate(),
        ),
        ControlReserve(
            capacity=5.0,
            committed=0.2,
        ),
        model=simple_prediction_model,
        config=PredictiveControlConfig(
            min_stabilization_margin=-10.0,
        ),
    )

    assert isinstance(
        demand,
        StabilizationDemand,
    )

    assert len(
        ranked
    ) == 2

    assert all(
        isinstance(
            item,
            CandidateEvaluation,
        )
        for item
        in ranked
    )


def test_end_to_end_evaluations_declare_no_external_expected_label_usage() -> None:
    _, ranked = evaluate_control_candidates(
        base_state(),
        base_disturbance(),
        (
            corrective_candidate(),
        ),
        ControlReserve(
            capacity=5.0,
            committed=0.2,
        ),
        model=simple_prediction_model,
        config=PredictiveControlConfig(
            min_stabilization_margin=-10.0,
        ),
    )

    assert ranked[
        0
    ].metadata[
        "external_expected_label_used"
    ] is False

    assert ranked[
        0
    ].outcome.metadata[
        "external_expected_label_used"
    ] is False


def test_end_to_end_predictive_control_selects_corrective_action_when_certain() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        uncertainty_probe_threshold=0.5,
        min_stabilization_margin=-10.0,
        residual_weight=3.0,
    )

    _, ranked = evaluate_control_candidates(
        base_state(
            uncertainty=0.1
        ),
        base_disturbance(),
        (
            corrective_candidate(),
            probe_candidate(),
            hold_candidate(),
        ),
        ControlReserve(
            capacity=5.0,
            committed=0.2,
        ),
        model=simple_prediction_model,
        config=config,
    )

    decision = select_control_action(
        ranked,
        config=config,
    )

    assert (
        decision.kind
        is PredictiveControlDecisionKind.ACTION
    )


# =============================================================================
# Immutability
# =============================================================================


def test_control_candidate_is_frozen() -> None:
    candidate = corrective_candidate()

    with pytest.raises(
        FrozenInstanceError
    ):
        candidate.estimated_cost = 10.0  # type: ignore[misc]


def test_control_reserve_is_frozen() -> None:
    reserve = base_reserve()

    with pytest.raises(
        FrozenInstanceError
    ):
        reserve.capacity = 10.0  # type: ignore[misc]


def test_stabilization_demand_is_frozen() -> None:
    demand = StabilizationDemand(
        state_error=1.0,
        disturbance_load=1.0,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        demand.urgency = 2.0  # type: ignore[misc]


def test_predicted_outcome_is_frozen() -> None:
    outcome = predict_outcome(
        base_state(),
        base_disturbance(),
        corrective_candidate(),
        model=simple_prediction_model,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        outcome.confidence = 0.0  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_prediction_is_deterministic() -> None:
    state = base_state()
    disturbance = base_disturbance()
    candidate = corrective_candidate()

    left = predict_outcome(
        state,
        disturbance,
        candidate,
        model=simple_prediction_model,
    )

    right = predict_outcome(
        state,
        disturbance,
        candidate,
        model=simple_prediction_model,
    )

    assert left == right


def test_stabilization_demand_is_deterministic() -> None:
    state = base_state()
    disturbance = base_disturbance()

    left = estimate_stabilization_demand(
        state,
        disturbance,
    )

    right = estimate_stabilization_demand(
        state,
        disturbance,
    )

    assert left == right


def test_candidate_ranking_is_deterministic() -> None:
    config = PredictiveControlConfig(
        min_stabilization_margin=-10.0,
    )

    items = (
        evaluation(
            corrective_candidate(
                candidate_id="b"
            ),
            residual=0.2,
            uncertainty=0.1,
            score_config=config,
        ),
        evaluation(
            corrective_candidate(
                candidate_id="a"
            ),
            residual=0.2,
            uncertainty=0.1,
            score_config=config,
        ),
    )

    left = rank_candidate_evaluations(
        items
    )

    right = rank_candidate_evaluations(
        items
    )

    assert left == right


def test_control_decision_is_deterministic() -> None:
    config = PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        uncertainty_probe_threshold=0.4,
        min_stabilization_margin=-10.0,
    )

    items = (
        evaluation(
            corrective_candidate(),
            residual=0.1,
            uncertainty=0.2,
            score_config=config,
        ),
        evaluation(
            probe_candidate(),
            residual=0.3,
            uncertainty=0.0,
            score_config=config,
        ),
    )

    left = select_control_action(
        items,
        config=config,
    )

    right = select_control_action(
        items,
        config=config,
    )

    assert left == right


def test_complete_candidate_evaluation_pipeline_is_deterministic() -> None:
    config = PredictiveControlConfig(
        min_stabilization_margin=-10.0,
    )

    args = dict(
        state=base_state(),
        disturbance=base_disturbance(),
        candidates=(
            corrective_candidate(),
            probe_candidate(),
            hold_candidate(),
        ),
        reserve=ControlReserve(
            capacity=5.0,
            committed=0.2,
        ),
        model=simple_prediction_model,
        config=config,
    )

    left = evaluate_control_candidates(
        **args
    )

    right = evaluate_control_candidates(
        **args
    )

    assert left == right
