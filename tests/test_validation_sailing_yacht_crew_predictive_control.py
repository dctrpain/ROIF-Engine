"""
Tests for Scenario 04A — Sailing Yacht + Crew Predictive Control

This suite validates the bridge between:
- validation.sailing.sailing_yacht_crew_coupled_case
- roif.predictive_control
- validation.sailing.sailing_yacht_crew_predictive_control

Core contract
-------------
1. Yacht/crew state is converted into a PredictiveState.
2. Environmental disturbance is represented separately.
3. Crew reserve is finite and derived from helm + trim utilization.
4. Five candidate actions are available:
   helm, trim, coupled, probe, hold.
5. Coupled action predicts the lowest residual error in the default benchmark.
6. Selection uses predictive-control logic without evaluator labels.
7. Stabilization demand may exceed available reserve and must remain visible
   in the audit result.
8. Selected prediction can be compared with a later observed state.
9. The complete 04A bridge is immutable and deterministic.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.predictive_control import (
    ControlCandidateKind,
    PredictiveControlDecisionKind,
    PredictiveState,
)

from validation.sailing.sailing_yacht_crew_coupled_case import (
    COURSE_ERROR_CHANNEL,
    HEEL_YAW_CHANNEL,
    HELM_CONTROL_DEMAND_CHANNEL,
    RUDDER_ACTION_CHANNEL,
    SAIL_LOAD_CHANNEL,
    SAIL_TRIM_ACTION_CHANNEL,
    TRIM_CONTROL_DEMAND_CHANNEL,
    WIND_DISTURBANCE_CHANNEL,
    build_sailing_yacht_crew_coupled_case,
)

from validation.sailing.sailing_yacht_crew_predictive_control import (
    COUPLED_CORRECTION_ID,
    DEFAULT_COUPLED_AUTHORITY,
    DEFAULT_CREW_RESERVE_CAPACITY,
    DEFAULT_HELM_AUTHORITY,
    DEFAULT_PROBE_MAGNITUDE,
    DEFAULT_TRIM_AUTHORITY,
    HELM_CORRECTION_ID,
    HOLD_ID,
    INFORMATION_PROBE_ID,
    TRIM_CORRECTION_ID,
    SailingPredictiveControlError,
    SailingPredictiveControlResult,
    build_case,
    build_control_candidates,
    build_crew_control_reserve,
    build_disturbance_estimate,
    build_predictive_control_config,
    build_predictive_state,
    channel_capacity,
    channel_load,
    channel_lookup,
    compare_selected_prediction_with_observation,
    run_sailing_yacht_crew_predictive_control,
    sailing_prediction_model,
)


@pytest.fixture(scope="module")
def case():
    return build_case()


@pytest.fixture(scope="module")
def result(
    case,
) -> SailingPredictiveControlResult:
    return run_sailing_yacht_crew_predictive_control(
        case
    )


# =============================================================================
# Basic case bridge
# =============================================================================


def test_build_case_matches_04a_case_factory() -> None:
    left = build_case()
    right = build_sailing_yacht_crew_coupled_case()

    assert left.case_id == right.case_id
    assert left.channel_ids == right.channel_ids


def test_channel_lookup_contains_all_case_channels(
    case,
) -> None:
    lookup = channel_lookup(
        case
    )

    assert set(
        lookup
    ) == set(
        case.channel_ids
    )


def test_channel_lookup_is_read_only(
    case,
) -> None:
    lookup = channel_lookup(
        case
    )

    assert isinstance(
        lookup,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        lookup[
            "x"
        ] = object()  # type: ignore[index]


def test_channel_load_matches_source_case(
    case,
) -> None:
    expected = float(
        channel_lookup(
            case
        )[
            COURSE_ERROR_CHANNEL
        ].capacity_state.load
    )

    assert channel_load(
        case,
        COURSE_ERROR_CHANNEL,
    ) == pytest.approx(
        expected
    )


def test_channel_capacity_matches_source_case(
    case,
) -> None:
    expected = float(
        channel_lookup(
            case
        )[
            HELM_CONTROL_DEMAND_CHANNEL
        ].capacity_state.capacity
    )

    assert channel_capacity(
        case,
        HELM_CONTROL_DEMAND_CHANNEL,
    ) == pytest.approx(
        expected
    )


# =============================================================================
# Predictive-state bridge
# =============================================================================


def test_predictive_state_contains_expected_observable_channels(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    assert set(
        state.values
    ) == {
        SAIL_LOAD_CHANNEL,
        HEEL_YAW_CHANNEL,
        COURSE_ERROR_CHANNEL,
        HELM_CONTROL_DEMAND_CHANNEL,
        TRIM_CONTROL_DEMAND_CHANNEL,
        RUDDER_ACTION_CHANNEL,
        SAIL_TRIM_ACTION_CHANNEL,
    }


def test_predictive_state_regulates_course_and_heel(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    assert set(
        state.target_values
    ) == {
        COURSE_ERROR_CHANNEL,
        HEEL_YAW_CHANNEL,
    }

    assert (
        state.target_values[
            COURSE_ERROR_CHANNEL
        ]
        == 0.0
    )

    assert (
        state.target_values[
            HEEL_YAW_CHANNEL
        ]
        == 0.0
    )


def test_predictive_state_preserves_nonzero_stressed_outputs(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    assert (
        state.values[
            COURSE_ERROR_CHANNEL
        ]
        > 0.0
    )

    assert (
        state.values[
            HEEL_YAW_CHANNEL
        ]
        > 0.0
    )


def test_predictive_state_declares_no_external_expected_label_usage(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    assert (
        state.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Disturbance
# =============================================================================


def test_disturbance_contains_wind_and_wave_components(
    case,
) -> None:
    disturbance = build_disturbance_estimate(
        case
    )

    assert set(
        disturbance.components
    ) == {
        "wind",
        "wave",
    }


def test_disturbance_wind_is_derived_from_case(
    case,
) -> None:
    disturbance = build_disturbance_estimate(
        case
    )

    assert (
        disturbance.components[
            "wind"
        ]
        > 0.0
    )


def test_disturbance_wave_component_is_positive(
    case,
) -> None:
    disturbance = build_disturbance_estimate(
        case
    )

    assert (
        disturbance.components[
            "wave"
        ]
        > 0.0
    )


def test_disturbance_metadata_marks_wave_as_normalized_benchmark(
    case,
) -> None:
    disturbance = build_disturbance_estimate(
        case
    )

    assert (
        disturbance.metadata[
            "wave_component_is_benchmark_normalized"
        ]
        is True
    )


# =============================================================================
# Crew reserve
# =============================================================================


def test_crew_reserve_capacity_uses_default(
    case,
) -> None:
    reserve = build_crew_control_reserve(
        case
    )

    assert reserve.capacity == pytest.approx(
        DEFAULT_CREW_RESERVE_CAPACITY
    )


def test_crew_reserve_committed_is_positive(
    case,
) -> None:
    reserve = build_crew_control_reserve(
        case
    )

    assert reserve.committed > 0.0


def test_crew_reserve_available_is_positive(
    case,
) -> None:
    reserve = build_crew_control_reserve(
        case
    )

    assert reserve.available > 0.0


def test_crew_reserve_available_is_below_total_capacity(
    case,
) -> None:
    reserve = build_crew_control_reserve(
        case
    )

    assert reserve.available < reserve.capacity


def test_crew_reserve_metadata_contains_helm_and_trim_utilization(
    case,
) -> None:
    reserve = build_crew_control_reserve(
        case
    )

    assert (
        reserve.metadata[
            "helm_utilization"
        ]
        > 0.0
    )

    assert (
        reserve.metadata[
            "trim_utilization"
        ]
        > 0.0
    )

    assert (
        reserve.metadata[
            "mean_utilization"
        ]
        > 0.0
    )


def test_crew_reserve_metadata_declares_no_external_label_usage(
    case,
) -> None:
    reserve = build_crew_control_reserve(
        case
    )

    assert (
        reserve.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Candidate set
# =============================================================================


def test_candidate_set_contains_exactly_five_actions() -> None:
    candidates = build_control_candidates()

    assert len(
        candidates
    ) == 5


def test_candidate_ids_are_complete() -> None:
    candidates = build_control_candidates()

    assert {
        candidate.candidate_id
        for candidate
        in candidates
    } == {
        HELM_CORRECTION_ID,
        TRIM_CORRECTION_ID,
        COUPLED_CORRECTION_ID,
        INFORMATION_PROBE_ID,
        HOLD_ID,
    }


def test_helm_candidate_is_corrective() -> None:
    candidate = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == HELM_CORRECTION_ID
    )

    assert (
        candidate.kind
        is ControlCandidateKind.CORRECTIVE
    )

    assert (
        RUDDER_ACTION_CHANNEL
        in candidate.control_delta
    )


def test_trim_candidate_is_corrective() -> None:
    candidate = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == TRIM_CORRECTION_ID
    )

    assert (
        candidate.kind
        is ControlCandidateKind.CORRECTIVE
    )

    assert (
        SAIL_TRIM_ACTION_CHANNEL
        in candidate.control_delta
    )


def test_coupled_candidate_controls_both_channels() -> None:
    candidate = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == COUPLED_CORRECTION_ID
    )

    assert (
        candidate.kind
        is ControlCandidateKind.CORRECTIVE
    )

    assert set(
        candidate.control_delta
    ) == {
        RUDDER_ACTION_CHANNEL,
        SAIL_TRIM_ACTION_CHANNEL,
    }


def test_information_probe_is_probe_kind() -> None:
    candidate = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == INFORMATION_PROBE_ID
    )

    assert (
        candidate.kind
        is ControlCandidateKind.PROBE
    )

    assert (
        candidate.reversibility
        == pytest.approx(
            1.0
        )
    )


def test_information_probe_uses_small_magnitude() -> None:
    candidate = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == INFORMATION_PROBE_ID
    )

    assert abs(
        candidate.control_delta[
            RUDDER_ACTION_CHANNEL
        ]
    ) == pytest.approx(
        DEFAULT_PROBE_MAGNITUDE
    )

    assert abs(
        candidate.control_delta[
            SAIL_TRIM_ACTION_CHANNEL
        ]
    ) == pytest.approx(
        DEFAULT_PROBE_MAGNITUDE
    )


def test_hold_candidate_is_hold_kind() -> None:
    candidate = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == HOLD_ID
    )

    assert (
        candidate.kind
        is ControlCandidateKind.HOLD
    )

    assert (
        candidate.control_delta
        == {}
    )


def test_all_candidates_declare_no_external_expected_label_usage() -> None:
    for candidate in build_control_candidates():
        assert (
            candidate.metadata[
                "external_expected_label_used"
            ]
            is False
        )


# =============================================================================
# Predictor mechanics
# =============================================================================


def test_positive_disturbance_increases_hold_course_error(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    hold = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == HOLD_ID
    )

    predicted = sailing_prediction_model(
        state,
        disturbance,
        hold,
    )

    assert (
        predicted.values[
            COURSE_ERROR_CHANNEL
        ]
        >
        state.values[
            COURSE_ERROR_CHANNEL
        ]
    )


def test_positive_disturbance_increases_hold_heel_yaw(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    hold = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == HOLD_ID
    )

    predicted = sailing_prediction_model(
        state,
        disturbance,
        hold,
    )

    assert (
        predicted.values[
            HEEL_YAW_CHANNEL
        ]
        >
        state.values[
            HEEL_YAW_CHANNEL
        ]
    )


def test_helm_action_reduces_course_error_relative_to_hold(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    candidates = {
        item.candidate_id: item
        for item
        in build_control_candidates()
    }

    hold = sailing_prediction_model(
        state,
        disturbance,
        candidates[
            HOLD_ID
        ],
    )

    helm = sailing_prediction_model(
        state,
        disturbance,
        candidates[
            HELM_CORRECTION_ID
        ],
    )

    assert (
        helm.values[
            COURSE_ERROR_CHANNEL
        ]
        <
        hold.values[
            COURSE_ERROR_CHANNEL
        ]
    )


def test_trim_action_reduces_heel_relative_to_hold(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    candidates = {
        item.candidate_id: item
        for item
        in build_control_candidates()
    }

    hold = sailing_prediction_model(
        state,
        disturbance,
        candidates[
            HOLD_ID
        ],
    )

    trim = sailing_prediction_model(
        state,
        disturbance,
        candidates[
            TRIM_CORRECTION_ID
        ],
    )

    assert (
        trim.values[
            HEEL_YAW_CHANNEL
        ]
        <
        hold.values[
            HEEL_YAW_CHANNEL
        ]
    )


def test_probe_reduces_uncertainty_more_than_hold(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    candidates = {
        item.candidate_id: item
        for item
        in build_control_candidates()
    }

    probe = sailing_prediction_model(
        state,
        disturbance,
        candidates[
            INFORMATION_PROBE_ID
        ],
    )

    hold = sailing_prediction_model(
        state,
        disturbance,
        candidates[
            HOLD_ID
        ],
    )

    assert (
        probe.uncertainty
        <
        hold.uncertainty
    )


def test_predictor_metadata_declares_no_hydrodynamic_claim(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    candidate = build_control_candidates()[
        0
    ]

    predicted = sailing_prediction_model(
        state,
        disturbance,
        candidate,
    )

    assert (
        predicted.metadata[
            "hydrodynamic_claim"
        ]
        is False
    )


# =============================================================================
# Default 04A full run
# =============================================================================


def test_complete_predictive_control_runs(
    result,
) -> None:
    assert isinstance(
        result,
        SailingPredictiveControlResult,
    )


def test_complete_result_uses_04a_case_id(
    result,
) -> None:
    assert (
        result.case_id
        == "sailing_04A_coupled_yacht_crew"
    )


def test_complete_run_has_five_evaluations(
    result,
) -> None:
    assert len(
        result.evaluations
    ) == 5


def test_complete_run_decision_is_action(
    result,
) -> None:
    assert (
        result.decision.kind
        is PredictiveControlDecisionKind.ACTION
    )


def test_complete_run_selects_coupled_action(
    result,
) -> None:
    assert (
        result.decision.selected_candidate_id
        == COUPLED_CORRECTION_ID
    )


def test_coupled_action_has_lowest_predicted_residual(
    result,
) -> None:
    residuals = {
        item.candidate.candidate_id:
        item.outcome.predicted_residual_error
        for item
        in result.evaluations
    }

    assert residuals[
        COUPLED_CORRECTION_ID
    ] == min(
        residuals.values()
    )


def test_coupled_action_beats_hold_on_predicted_residual(
    result,
) -> None:
    residuals = {
        item.candidate.candidate_id:
        item.outcome.predicted_residual_error
        for item
        in result.evaluations
    }

    assert (
        residuals[
            COUPLED_CORRECTION_ID
        ]
        <
        residuals[
            HOLD_ID
        ]
    )


def test_coupled_action_beats_probe_on_predicted_residual(
    result,
) -> None:
    residuals = {
        item.candidate.candidate_id:
        item.outcome.predicted_residual_error
        for item
        in result.evaluations
    }

    assert (
        residuals[
            COUPLED_CORRECTION_ID
        ]
        <
        residuals[
            INFORMATION_PROBE_ID
        ]
    )


def test_coupled_action_beats_helm_only_on_predicted_residual(
    result,
) -> None:
    residuals = {
        item.candidate.candidate_id:
        item.outcome.predicted_residual_error
        for item
        in result.evaluations
    }

    assert (
        residuals[
            COUPLED_CORRECTION_ID
        ]
        <
        residuals[
            HELM_CORRECTION_ID
        ]
    )


def test_coupled_action_beats_trim_only_on_predicted_residual(
    result,
) -> None:
    residuals = {
        item.candidate.candidate_id:
        item.outcome.predicted_residual_error
        for item
        in result.evaluations
    }

    assert (
        residuals[
            COUPLED_CORRECTION_ID
        ]
        <
        residuals[
            TRIM_CORRECTION_ID
        ]
    )


# =============================================================================
# Reserve-pressure contract
# =============================================================================


def test_stabilization_demand_is_positive(
    result,
) -> None:
    assert result.demand.total > 0.0


def test_control_reserve_is_finite(
    result,
) -> None:
    assert result.reserve.capacity > 0.0
    assert result.reserve.available > 0.0


def test_default_04a_demand_exceeds_available_reserve(
    result,
) -> None:
    """
    Important audit property.

    The current normalized 04A benchmark intentionally operates under
    reserve pressure: demand is larger than immediately available crew reserve.
    """

    assert (
        result.demand.total
        >
        result.reserve.available
    )


def test_default_04a_records_negative_raw_reserve_pressure(
    result,
) -> None:
    raw_pressure = (
        result.reserve.available
        - result.demand.total
    )

    assert raw_pressure < 0.0


def test_every_evaluation_preserves_stabilization_margin(
    result,
) -> None:
    assert all(
        isinstance(
            item.stabilization_margin,
            float,
        )
        for item
        in result.evaluations
    )


def test_selected_action_is_viable_under_current_config(
    result,
) -> None:
    selected = result.decision.selected_evaluation

    assert selected is not None
    assert selected.viable is True


def test_selected_action_still_records_negative_stabilization_margin(
    result,
) -> None:
    """
    This fixes the current architectural state:
    action may remain admissible under a permissive benchmark threshold,
    but reserve pressure must not disappear from the audit.
    """

    selected = result.decision.selected_evaluation

    assert selected is not None

    assert (
        selected.stabilization_margin
        < 0.0
    )


def test_config_explicitly_allows_negative_margin() -> None:
    config = build_predictive_control_config()

    assert (
        config.min_stabilization_margin
        < 0.0
    )


# =============================================================================
# No-label / provenance boundary
# =============================================================================


def test_result_metadata_preserves_published_topology_basis(
    result,
) -> None:
    assert (
        result.metadata[
            "published_topology_basis"
        ]
        is True
    )


def test_result_metadata_marks_predictive_parameters_as_normalized(
    result,
) -> None:
    assert (
        result.metadata[
            "normalized_predictive_parameters"
        ]
        is True
    )


def test_result_metadata_declares_memory_disabled(
    result,
) -> None:
    assert (
        result.metadata[
            "memory_enabled"
        ]
        is False
    )


def test_result_metadata_declares_learning_disabled(
    result,
) -> None:
    assert (
        result.metadata[
            "learning_enabled"
        ]
        is False
    )


def test_result_metadata_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_decision_metadata_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.decision.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_all_evaluations_declare_no_external_expected_label_usage(
    result,
) -> None:
    for item in result.evaluations:
        assert (
            item.metadata[
                "external_expected_label_used"
            ]
            is False
        )

        assert (
            item.outcome.metadata[
                "external_expected_label_used"
            ]
            is False
        )


# =============================================================================
# Prediction error helper
# =============================================================================


def test_compare_selected_prediction_with_identical_observation_is_zero(
    result,
) -> None:
    selected = result.decision.selected_evaluation

    assert selected is not None

    observed = selected.outcome.predicted_state

    error = compare_selected_prediction_with_observation(
        result,
        observed,
    )

    assert error.l2_error == 0.0
    assert error.normalized_error == 0.0
    assert error.confidence == 1.0


def test_compare_selected_prediction_detects_course_error_difference(
    result,
) -> None:
    selected = result.decision.selected_evaluation

    assert selected is not None

    predicted = selected.outcome.predicted_state

    values = dict(
        predicted.values
    )

    values[
        COURSE_ERROR_CHANNEL
    ] += 0.5

    observed = PredictiveState(
        values=values,
        target_values=dict(
            predicted.target_values
        ),
        timestamp=predicted.timestamp,
        uncertainty=predicted.uncertainty,
    )

    error = compare_selected_prediction_with_observation(
        result,
        observed,
    )

    assert error.l2_error > 0.0

    assert (
        error.variable_errors[
            COURSE_ERROR_CHANNEL
        ]
        == pytest.approx(
            0.5
        )
    )


def test_compare_selected_prediction_metadata_declares_no_external_label_usage(
    result,
) -> None:
    selected = result.decision.selected_evaluation

    assert selected is not None

    error = compare_selected_prediction_with_observation(
        result,
        selected.outcome.predicted_state,
    )

    assert (
        error.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_compare_selected_prediction_requires_selected_evaluation(
    result,
) -> None:
    from dataclasses import replace

    empty_decision = replace(
        result.decision,
        selected_candidate_id=None,
        selected_evaluation=None,
    )

    no_selection = replace(
        result,
        decision=empty_decision,
    )

    with pytest.raises(
        SailingPredictiveControlError
    ):
        compare_selected_prediction_with_observation(
            no_selection,
            result.predictive_state,
        )


# =============================================================================
# Immutability
# =============================================================================


def test_result_metadata_is_read_only(
    result,
) -> None:
    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_complete_result_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.case_id = "x"  # type: ignore[misc]


def test_candidates_are_tuple(
    result,
) -> None:
    assert isinstance(
        result.candidates,
        tuple,
    )


def test_evaluations_are_tuple(
    result,
) -> None:
    assert isinstance(
        result.evaluations,
        tuple,
    )


# =============================================================================
# Determinism
# =============================================================================


def test_predictive_state_bridge_is_deterministic(
    case,
) -> None:
    left = build_predictive_state(
        case
    )

    right = build_predictive_state(
        case
    )

    assert left == right


def test_disturbance_bridge_is_deterministic(
    case,
) -> None:
    left = build_disturbance_estimate(
        case
    )

    right = build_disturbance_estimate(
        case
    )

    assert left == right


def test_crew_reserve_bridge_is_deterministic(
    case,
) -> None:
    left = build_crew_control_reserve(
        case
    )

    right = build_crew_control_reserve(
        case
    )

    assert left == right


def test_candidate_factory_is_deterministic() -> None:
    assert (
        build_control_candidates()
        == build_control_candidates()
    )


def test_prediction_model_is_deterministic(
    case,
) -> None:
    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    candidate = next(
        item
        for item
        in build_control_candidates()
        if item.candidate_id
        == COUPLED_CORRECTION_ID
    )

    left = sailing_prediction_model(
        state,
        disturbance,
        candidate,
    )

    right = sailing_prediction_model(
        state,
        disturbance,
        candidate,
    )

    assert left == right


def _evaluation_signature(
    result: SailingPredictiveControlResult,
):
    return tuple(
        (
            item.candidate.candidate_id,
            item.outcome.predicted_residual_error,
            item.outcome.predicted_control_cost,
            item.outcome.predicted_uncertainty,
            item.reserve_after_action,
            item.stabilization_margin,
            item.score,
            item.safe,
            item.viable,
        )
        for item
        in result.evaluations
    )


def test_complete_predictive_control_pipeline_is_deterministic(
    case,
) -> None:
    left = run_sailing_yacht_crew_predictive_control(
        case
    )

    right = run_sailing_yacht_crew_predictive_control(
        case
    )

    assert (
        left.predictive_state
        == right.predictive_state
    )

    assert (
        left.disturbance
        == right.disturbance
    )

    assert (
        left.reserve
        == right.reserve
    )

    assert (
        left.demand
        == right.demand
    )

    assert (
        _evaluation_signature(
            left
        )
        == _evaluation_signature(
            right
        )
    )

    assert (
        left.decision
        == right.decision
    )

    assert (
        left.metadata
        == right.metadata
    )


# =============================================================================
# Authority sanity checks
# =============================================================================


def test_helm_authority_is_positive() -> None:
    assert DEFAULT_HELM_AUTHORITY > 0.0


def test_trim_authority_is_positive() -> None:
    assert DEFAULT_TRIM_AUTHORITY > 0.0


def test_coupled_authority_is_positive() -> None:
    assert DEFAULT_COUPLED_AUTHORITY > 0.0
