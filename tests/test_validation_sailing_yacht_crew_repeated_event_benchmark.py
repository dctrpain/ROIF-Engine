"""
Tests for Scenario 04B — Repeated-Event Yacht–Crew Benchmark

04B is the NO-MEMORY baseline for future controller-memory experiments.

Core contract
-------------
Repeated disturbance patterns must NOT improve merely because they were seen
before.

For repeated Pattern A events:

    error(A1) ~= error(A2) ~= error(A3)

because previous ExperienceTrace objects are not supplied to later
predictive-control decisions.

The suite validates:
- deterministic disturbance sequence;
- five repeated-event episodes;
- no memory / no learning boundary;
- ExperienceTrace generation for every episode;
- repeated Pattern A prediction-error equality;
- stable policy selection in the default benchmark;
- nonzero systematic prediction error;
- aggregate metrics;
- immutability;
- determinism.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import sqrt
from types import MappingProxyType

import pytest

from roif.experience_trace import (
    ExperienceTrace,
    StabilizationOutcome,
    trace_is_learning_free,
)
from roif.predictive_control import (
    PredictiveControlDecisionKind,
)

from validation.sailing.sailing_yacht_crew_predictive_control import (
    COUPLED_CORRECTION_ID,
)

from validation.sailing.sailing_yacht_crew_repeated_event_benchmark import (
    DEFAULT_SEQUENCE_ID,
    SCENARIO_ID,
    InvalidRepeatedEventScenarioError,
    RepeatedEventBenchmarkResult,
    RepeatedEventEpisodeResult,
    SailingDisturbanceEvent,
    all_traces_learning_free,
    build_default_disturbance_sequence,
    build_episode_initial_state,
    build_event_disturbance_estimate,
    observed_residual_series,
    prediction_error_series,
    run_repeated_event_benchmark,
    run_repeated_event_episode,
    selected_action_ids,
    synthesize_observed_state,
)

from validation.sailing.sailing_yacht_crew_predictive_control import (
    build_case,
    build_control_candidates,
    build_crew_control_reserve,
    build_predictive_control_config,
    sailing_prediction_model,
)

from roif.predictive_control import (
    estimate_stabilization_demand,
    evaluate_control_candidates,
    select_control_action,
)


@pytest.fixture(scope="module")
def case():
    return build_case()


@pytest.fixture(scope="module")
def result(
    case,
) -> RepeatedEventBenchmarkResult:
    return run_repeated_event_benchmark(
        case=case
    )


# =============================================================================
# Scenario identity
# =============================================================================


def test_scenario_id_is_04b() -> None:
    assert (
        SCENARIO_ID
        == "sailing_04B_repeated_event_no_memory"
    )


def test_default_sequence_id_is_stable() -> None:
    assert (
        DEFAULT_SEQUENCE_ID
        == "04B_default_sequence"
    )


# =============================================================================
# Disturbance event contract
# =============================================================================


def test_event_accepts_valid_values() -> None:
    event = SailingDisturbanceEvent(
        event_id="x",
        wind=0.2,
        wave=0.1,
    )

    assert event.event_id == "x"
    assert event.wind == pytest.approx(
        0.2
    )
    assert event.wave == pytest.approx(
        0.1
    )


def test_event_rejects_empty_id() -> None:
    with pytest.raises(
        InvalidRepeatedEventScenarioError
    ):
        SailingDisturbanceEvent(
            event_id="",
            wind=0.1,
            wave=0.1,
        )


def test_event_rejects_negative_wind() -> None:
    with pytest.raises(
        InvalidRepeatedEventScenarioError
    ):
        SailingDisturbanceEvent(
            event_id="x",
            wind=-0.1,
            wave=0.1,
        )


def test_event_rejects_negative_wave() -> None:
    with pytest.raises(
        InvalidRepeatedEventScenarioError
    ):
        SailingDisturbanceEvent(
            event_id="x",
            wind=0.1,
            wave=-0.1,
        )


def test_event_confidence_is_clamped_high() -> None:
    event = SailingDisturbanceEvent(
        event_id="x",
        wind=0.1,
        wave=0.1,
        confidence=2.0,
    )

    assert event.confidence == 1.0


def test_event_confidence_is_clamped_low() -> None:
    event = SailingDisturbanceEvent(
        event_id="x",
        wind=0.1,
        wave=0.1,
        confidence=-1.0,
    )

    assert event.confidence == 0.0


def test_event_metadata_is_read_only() -> None:
    event = SailingDisturbanceEvent(
        event_id="x",
        wind=0.1,
        wave=0.1,
        metadata={
            "family": "A",
        },
    )

    assert isinstance(
        event.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        event.metadata[
            "family"
        ] = "B"  # type: ignore[index]


def test_event_is_frozen() -> None:
    event = SailingDisturbanceEvent(
        event_id="x",
        wind=0.1,
        wave=0.1,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        event.wind = 1.0  # type: ignore[misc]


# =============================================================================
# Default sequence
# =============================================================================


def test_default_sequence_contains_five_events() -> None:
    events = build_default_disturbance_sequence()

    assert len(
        events
    ) == 5


def test_default_sequence_event_ids_are_exact() -> None:
    events = build_default_disturbance_sequence()

    assert tuple(
        event.event_id
        for event
        in events
    ) == (
        "event_01_moderate",
        "event_02_stronger",
        "event_03_repeat_moderate",
        "event_04_strong",
        "event_05_repeat_moderate",
    )


def test_default_sequence_pattern_families_are_a_b_a_c_a() -> None:
    events = build_default_disturbance_sequence()

    assert tuple(
        event.metadata[
            "pattern_family"
        ]
        for event
        in events
    ) == (
        "A",
        "B",
        "A",
        "C",
        "A",
    )


def test_pattern_a_events_share_same_course_bias() -> None:
    events = build_default_disturbance_sequence()

    pattern_a = tuple(
        event
        for event
        in events
        if (
            event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert len(
        pattern_a
    ) == 3

    assert all(
        event.observation_bias_course
        == pytest.approx(
            0.08
        )
        for event
        in pattern_a
    )


def test_pattern_a_events_share_same_heel_bias() -> None:
    events = build_default_disturbance_sequence()

    pattern_a = tuple(
        event
        for event
        in events
        if (
            event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert all(
        event.observation_bias_heel
        == pytest.approx(
            0.04
        )
        for event
        in pattern_a
    )


def test_pattern_b_bias_is_exact() -> None:
    event = build_default_disturbance_sequence()[
        1
    ]

    assert event.observation_bias_course == pytest.approx(
        -0.03
    )

    assert event.observation_bias_heel == pytest.approx(
        0.09
    )


def test_pattern_c_bias_is_exact() -> None:
    event = build_default_disturbance_sequence()[
        3
    ]

    assert event.observation_bias_course == pytest.approx(
        0.12
    )

    assert event.observation_bias_heel == pytest.approx(
        0.07
    )


def test_default_sequence_is_deterministic() -> None:
    assert (
        build_default_disturbance_sequence()
        == build_default_disturbance_sequence()
    )


# =============================================================================
# Disturbance bridge
# =============================================================================


def test_event_disturbance_preserves_wind_and_wave() -> None:
    event = SailingDisturbanceEvent(
        event_id="x",
        wind=0.3,
        wave=0.2,
        confidence=0.8,
    )

    disturbance = build_event_disturbance_estimate(
        event
    )

    assert disturbance.components[
        "wind"
    ] == pytest.approx(
        0.3
    )

    assert disturbance.components[
        "wave"
    ] == pytest.approx(
        0.2
    )

    assert disturbance.confidence == pytest.approx(
        0.8
    )


def test_event_disturbance_metadata_declares_no_external_label_usage() -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    disturbance = build_event_disturbance_estimate(
        event
    )

    assert (
        disturbance.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# No-memory initial-state contract
# =============================================================================


def test_episode_initial_state_marks_memory_input_unused(
    case,
) -> None:
    state = build_episode_initial_state(
        case,
        episode_index=1,
    )

    assert (
        state.metadata[
            "memory_input_used"
        ]
        is False
    )


def test_episode_initial_state_marks_previous_trace_unused(
    case,
) -> None:
    state = build_episode_initial_state(
        case,
        episode_index=1,
    )

    assert (
        state.metadata[
            "previous_trace_used"
        ]
        is False
    )


def test_episode_initial_state_marks_predictive_preload_unused(
    case,
) -> None:
    state = build_episode_initial_state(
        case,
        episode_index=1,
    )

    assert (
        state.metadata[
            "predictive_preload_used"
        ]
        is False
    )


def test_episode_initial_states_have_same_values_across_repeats(
    case,
) -> None:
    first = build_episode_initial_state(
        case,
        episode_index=1,
    )

    later = build_episode_initial_state(
        case,
        episode_index=5,
    )

    assert first.values == later.values
    assert (
        first.target_values
        == later.target_values
    )
    assert (
        first.uncertainty
        == later.uncertainty
    )


def test_episode_initial_state_timestamp_changes_only_episode_identity(
    case,
) -> None:
    first = build_episode_initial_state(
        case,
        episode_index=1,
    )

    second = build_episode_initial_state(
        case,
        episode_index=2,
    )

    assert first.timestamp == pytest.approx(
        1.0
    )

    assert second.timestamp == pytest.approx(
        2.0
    )

    assert first.values == second.values


# =============================================================================
# Observation model
# =============================================================================


def _run_one_evaluation(
    case,
    event: SailingDisturbanceEvent,
):
    state = build_episode_initial_state(
        case,
        episode_index=1,
    )

    d = build_event_disturbance_estimate(
        event
    )

    r = build_crew_control_reserve(
        case
    )

    dem = estimate_stabilization_demand(
        state,
        d,
        uncertainty_weight=0.50,
        urgency=1.0,
    )

    config = build_predictive_control_config()

    candidates = build_control_candidates()

    _, evaluations = evaluate_control_candidates(
        state,
        d,
        candidates,
        r,
        model=sailing_prediction_model,
        config=config,
        demand=dem,
    )

    decision = select_control_action(
        evaluations,
        config=config,
    )

    return (
        state,
        decision.selected_evaluation,
    )


def test_observation_model_adds_declared_course_bias(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    state, selected = _run_one_evaluation(
        case,
        event,
    )

    assert selected is not None

    observed = synthesize_observed_state(
        initial_state=state,
        selected_evaluation=selected,
        event=event,
    )

    predicted = selected.outcome.predicted_state

    assert (
        observed.values[
            "course_error"
        ]
        - predicted.values[
            "course_error"
        ]
    ) == pytest.approx(
        event.observation_bias_course
    )


def test_observation_model_adds_declared_heel_bias(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    state, selected = _run_one_evaluation(
        case,
        event,
    )

    assert selected is not None

    observed = synthesize_observed_state(
        initial_state=state,
        selected_evaluation=selected,
        event=event,
    )

    predicted = selected.outcome.predicted_state

    assert (
        observed.values[
            "heel_yaw_response"
        ]
        - predicted.values[
            "heel_yaw_response"
        ]
    ) == pytest.approx(
        event.observation_bias_heel
    )


def test_observation_model_is_learning_free(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    state, selected = _run_one_evaluation(
        case,
        event,
    )

    observed = synthesize_observed_state(
        initial_state=state,
        selected_evaluation=selected,
        event=event,
    )

    assert (
        observed.metadata[
            "learning_applied"
        ]
        is False
    )

    assert (
        observed.metadata[
            "controller_memory_mutated"
        ]
        is False
    )


# =============================================================================
# Single episode
# =============================================================================


def test_single_episode_returns_episode_result(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert isinstance(
        episode,
        RepeatedEventEpisodeResult,
    )


def test_single_episode_creates_experience_trace(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert isinstance(
        episode.trace,
        ExperienceTrace,
    )


def test_single_episode_trace_is_learning_free(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert trace_is_learning_free(
        episode.trace
    ) is True


def test_single_episode_metadata_marks_no_memory_baseline(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert (
        episode.metadata[
            "no_memory_baseline"
        ]
        is True
    )


def test_single_episode_metadata_marks_previous_trace_unused(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert (
        episode.metadata[
            "previous_trace_used"
        ]
        is False
    )


def test_single_episode_selects_action(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert (
        episode.decision.kind
        is PredictiveControlDecisionKind.ACTION
    )


def test_single_episode_default_action_is_coupled(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert (
        episode.decision.selected_candidate_id
        == COUPLED_CORRECTION_ID
    )


def test_single_episode_prediction_error_is_nonzero(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="test_sequence",
    )

    assert episode.trace.prediction_error is not None
    assert (
        episode.trace.prediction_error.l2_error
        > 0.0
    )


# =============================================================================
# Complete benchmark
# =============================================================================


def test_complete_benchmark_returns_result(
    result,
) -> None:
    assert isinstance(
        result,
        RepeatedEventBenchmarkResult,
    )


def test_complete_benchmark_scenario_id(
    result,
) -> None:
    assert result.scenario_id == SCENARIO_ID


def test_complete_benchmark_sequence_id(
    result,
) -> None:
    assert (
        result.sequence_id
        == DEFAULT_SEQUENCE_ID
    )


def test_complete_benchmark_contains_five_episodes(
    result,
) -> None:
    assert len(
        result.episodes
    ) == 5


def test_all_episodes_have_unique_trace_ids(
    result,
) -> None:
    trace_ids = tuple(
        episode.trace.identity.trace_id
        for episode
        in result.episodes
    )

    assert len(
        set(
            trace_ids
        )
    ) == len(
        trace_ids
    )


def test_all_episodes_are_learning_free(
    result,
) -> None:
    assert all_traces_learning_free(
        result
    ) is True


def test_all_episode_trace_metadata_declares_no_learning(
    result,
) -> None:
    assert all(
        episode.trace.metadata[
            "learning_applied"
        ]
        is False
        for episode
        in result.episodes
    )


def test_all_episode_initial_states_declare_no_memory_input(
    result,
) -> None:
    assert all(
        episode.initial_state.metadata[
            "memory_input_used"
        ]
        is False
        for episode
        in result.episodes
    )


def test_all_episode_initial_states_declare_no_previous_trace_usage(
    result,
) -> None:
    assert all(
        episode.initial_state.metadata[
            "previous_trace_used"
        ]
        is False
        for episode
        in result.episodes
    )


def test_benchmark_metadata_declares_memory_disabled(
    result,
) -> None:
    assert (
        result.metadata[
            "memory_enabled"
        ]
        is False
    )


def test_benchmark_metadata_declares_learning_disabled(
    result,
) -> None:
    assert (
        result.metadata[
            "learning_enabled"
        ]
        is False
    )


def test_benchmark_metadata_declares_previous_trace_not_used(
    result,
) -> None:
    assert (
        result.metadata[
            "previous_trace_used_for_next_decision"
        ]
        is False
    )


def test_benchmark_metadata_declares_predictive_preload_disabled(
    result,
) -> None:
    assert (
        result.metadata[
            "predictive_preload_enabled"
        ]
        is False
    )


def test_benchmark_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Baseline action-policy contract
# =============================================================================


def test_default_action_series_is_all_coupled(
    result,
) -> None:
    assert selected_action_ids(
        result
    ) == (
        COUPLED_CORRECTION_ID,
        COUPLED_CORRECTION_ID,
        COUPLED_CORRECTION_ID,
        COUPLED_CORRECTION_ID,
        COUPLED_CORRECTION_ID,
    )


def test_repeated_pattern_does_not_spontaneously_change_policy(
    result,
) -> None:
    actions = selected_action_ids(
        result
    )

    assert actions[
        0
    ] == actions[
        2
    ] == actions[
        4
    ]


# =============================================================================
# Prediction-error baseline
# =============================================================================


def test_prediction_error_series_has_five_values(
    result,
) -> None:
    errors = prediction_error_series(
        result
    )

    assert len(
        errors
    ) == 5

    assert all(
        value is not None
        for value
        in errors
    )


def test_all_prediction_errors_are_nonzero(
    result,
) -> None:
    errors = prediction_error_series(
        result
    )

    assert all(
        value is not None
        and value > 0.0
        for value
        in errors
    )


def test_pattern_a_error_matches_declared_bias_norm(
    result,
) -> None:
    expected = sqrt(
        0.08 ** 2
        + 0.04 ** 2
    )

    errors = prediction_error_series(
        result
    )

    assert errors[
        0
    ] == pytest.approx(
        expected
    )

    assert errors[
        2
    ] == pytest.approx(
        expected
    )

    assert errors[
        4
    ] == pytest.approx(
        expected
    )


def test_pattern_b_error_matches_declared_bias_norm(
    result,
) -> None:
    expected = sqrt(
        (-0.03) ** 2
        + 0.09 ** 2
    )

    errors = prediction_error_series(
        result
    )

    assert errors[
        1
    ] == pytest.approx(
        expected
    )


def test_pattern_c_error_matches_declared_bias_norm(
    result,
) -> None:
    expected = sqrt(
        0.12 ** 2
        + 0.07 ** 2
    )

    errors = prediction_error_series(
        result
    )

    assert errors[
        3
    ] == pytest.approx(
        expected
    )


def test_repeated_pattern_a_error_does_not_improve(
    result,
) -> None:
    """
    Central 04B no-memory baseline invariant.

    Seeing Pattern A before must not reduce future Pattern A prediction error.
    """

    errors = prediction_error_series(
        result
    )

    assert errors[
        0
    ] == pytest.approx(
        errors[
            2
        ]
    )

    assert errors[
        2
    ] == pytest.approx(
        errors[
            4
        ]
    )


def test_repeated_pattern_a_error_is_not_monotonically_reduced(
    result,
) -> None:
    errors = prediction_error_series(
        result
    )

    a1 = errors[
        0
    ]
    a2 = errors[
        2
    ]
    a3 = errors[
        4
    ]

    assert a1 is not None
    assert a2 is not None
    assert a3 is not None

    assert not (
        a1 > a2 > a3
    )


def test_mean_prediction_error_is_expected_value(
    result,
) -> None:
    expected = (
        sqrt(
            0.08 ** 2
            + 0.04 ** 2
        )
        + sqrt(
            (-0.03) ** 2
            + 0.09 ** 2
        )
        + sqrt(
            0.08 ** 2
            + 0.04 ** 2
        )
        + sqrt(
            0.12 ** 2
            + 0.07 ** 2
        )
        + sqrt(
            0.08 ** 2
            + 0.04 ** 2
        )
    ) / 5.0

    assert result.mean_prediction_error == pytest.approx(
        expected
    )


# =============================================================================
# Residual / stabilization metrics
# =============================================================================


def test_observed_residual_series_has_five_values(
    result,
) -> None:
    residuals = observed_residual_series(
        result
    )

    assert len(
        residuals
    ) == 5


def test_all_observed_residuals_are_positive(
    result,
) -> None:
    assert all(
        value > 0.0
        for value
        in observed_residual_series(
            result
        )
    )


def test_mean_observed_residual_matches_series_mean(
    result,
) -> None:
    residuals = observed_residual_series(
        result
    )

    assert result.mean_observed_residual == pytest.approx(
        sum(
            residuals
        )
        / len(
            residuals
        )
    )


def test_stabilization_outcome_counts_cover_all_episodes(
    result,
) -> None:
    assert (
        result.improved_episode_count
        + result.unchanged_episode_count
        + result.worsened_episode_count
        == len(
            result.episodes
        )
    )


def test_default_benchmark_has_three_improved_episodes(
    result,
) -> None:
    assert (
        result.improved_episode_count
        == 3
    )


def test_episode_stabilization_outcomes_match_aggregate_counts(
    result,
) -> None:
    improved = sum(
        1
        for episode
        in result.episodes
        if (
            episode.trace.stabilization.outcome
            is StabilizationOutcome.IMPROVED
        )
    )

    unchanged = sum(
        1
        for episode
        in result.episodes
        if (
            episode.trace.stabilization.outcome
            is StabilizationOutcome.UNCHANGED
        )
    )

    worsened = sum(
        1
        for episode
        in result.episodes
        if (
            episode.trace.stabilization.outcome
            is StabilizationOutcome.WORSENED
        )
    )

    assert improved == result.improved_episode_count
    assert unchanged == result.unchanged_episode_count
    assert worsened == result.worsened_episode_count


# =============================================================================
# Trace provenance
# =============================================================================


def test_each_trace_sequence_id_matches_benchmark(
    result,
) -> None:
    assert all(
        episode.trace.identity.sequence_id
        == result.sequence_id
        for episode
        in result.episodes
    )


def test_each_trace_episode_index_matches_episode(
    result,
) -> None:
    assert all(
        episode.trace.identity.episode_index
        == episode.episode_index
        for episode
        in result.episodes
    )


def test_each_trace_metadata_marks_no_memory_baseline(
    result,
) -> None:
    assert all(
        episode.trace.metadata[
            "no_memory_baseline"
        ]
        is True
        for episode
        in result.episodes
    )


def test_each_trace_marks_predictive_preload_unmodified(
    result,
) -> None:
    assert all(
        episode.trace.metadata[
            "predictive_preload_modified"
        ]
        is False
        for episode
        in result.episodes
    )


def test_each_trace_marks_controller_memory_unmodified(
    result,
) -> None:
    assert all(
        episode.trace.metadata[
            "controller_memory_mutated"
        ]
        is False
        for episode
        in result.episodes
    )


# =============================================================================
# Empty sequence guard
# =============================================================================


def test_empty_event_sequence_is_rejected(
    case,
) -> None:
    with pytest.raises(
        InvalidRepeatedEventScenarioError
    ):
        run_repeated_event_benchmark(
            case=case,
            events=(),
        )


# =============================================================================
# Immutability
# =============================================================================


def test_episode_metadata_is_read_only(
    result,
) -> None:
    episode = result.episodes[
        0
    ]

    assert isinstance(
        episode.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        episode.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_episode_result_is_frozen(
    result,
) -> None:
    episode = result.episodes[
        0
    ]

    with pytest.raises(
        FrozenInstanceError
    ):
        episode.episode_index = 99  # type: ignore[misc]


def test_benchmark_episodes_are_tuple(
    result,
) -> None:
    assert isinstance(
        result.episodes,
        tuple,
    )


def test_benchmark_metadata_is_read_only(
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


def test_benchmark_result_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.sequence_id = "changed"  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_single_episode_is_deterministic(
    case,
) -> None:
    event = build_default_disturbance_sequence()[
        0
    ]

    kwargs = dict(
        case=case,
        event=event,
        episode_index=1,
        sequence_id="deterministic",
    )

    left = run_repeated_event_episode(
        **kwargs
    )

    right = run_repeated_event_episode(
        **kwargs
    )

    assert left == right


def test_prediction_error_series_is_deterministic(
    case,
) -> None:
    left = run_repeated_event_benchmark(
        case=case
    )

    right = run_repeated_event_benchmark(
        case=case
    )

    assert prediction_error_series(
        left
    ) == prediction_error_series(
        right
    )


def test_action_series_is_deterministic(
    case,
) -> None:
    left = run_repeated_event_benchmark(
        case=case
    )

    right = run_repeated_event_benchmark(
        case=case
    )

    assert selected_action_ids(
        left
    ) == selected_action_ids(
        right
    )


def test_residual_series_is_deterministic(
    case,
) -> None:
    left = run_repeated_event_benchmark(
        case=case
    )

    right = run_repeated_event_benchmark(
        case=case
    )

    assert observed_residual_series(
        left
    ) == observed_residual_series(
        right
    )


def _benchmark_signature(
    item: RepeatedEventBenchmarkResult,
):
    return (
        item.scenario_id,
        item.sequence_id,
        tuple(
            (
                episode.episode_index,
                episode.event.event_id,
                episode.decision.kind,
                episode.decision.selected_candidate_id,
                (
                    None
                    if episode.trace.prediction_error
                    is None
                    else episode.trace.prediction_error.l2_error
                ),
                episode.observed_state.residual_norm(),
                episode.trace.stabilization.outcome,
                trace_is_learning_free(
                    episode.trace
                ),
            )
            for episode
            in item.episodes
        ),
        item.mean_prediction_error,
        item.mean_observed_residual,
        item.improved_episode_count,
        item.unchanged_episode_count,
        item.worsened_episode_count,
        tuple(
            sorted(
                item.metadata.items()
            )
        ),
    )


def test_complete_04b_pipeline_is_deterministic(
    case,
) -> None:
    left = run_repeated_event_benchmark(
        case=case
    )

    right = run_repeated_event_benchmark(
        case=case
    )

    assert _benchmark_signature(
        left
    ) == _benchmark_signature(
        right
    )
