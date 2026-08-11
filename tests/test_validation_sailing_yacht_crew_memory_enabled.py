"""
Tests for Scenario 04D — Memory-Enabled Sailing Benchmark

The suite fixes the first complete memory-enabled predictive-control contract.

Contrast:

    04B no memory:
        A1 ~= A2 ~= A3

    04D memory enabled:
        A1 ~= A2
        A3 << A1

Pipeline under test:

    Predictive Control
        -> Observation
        -> ExperienceTrace
        -> ControllerMemory
        -> MemoryScar
        -> PredictivePreload
        -> next Predictive Control cycle

Core invariants:

    Memory Retrieval != Action Selection
    PredictivePreload != Policy Override
    MemoryScar != Ground Truth
    Evaluator Pattern Labels != Controller Action Labels
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.controller_memory import (
    ControllerMemory,
    MemoryScar,
)
from roif.predictive_preload import (
    PredictivePreload,
    preload_is_policy_free,
)

from validation.sailing.sailing_yacht_crew_memory_enabled import (
    DEFAULT_SEQUENCE_ID,
    MEMORY_PATTERN_ID,
    SCENARIO_ID,
    MemoryEnabledEpisodeResult,
    MemoryEnabledSailingBenchmarkError,
    MemoryEnabledSailingBenchmarkResult,
    derive_pattern_a_scar,
    pattern_a_memory_usage,
    pattern_a_scar_confidence_series,
    run_memory_enabled_episode,
    run_memory_enabled_sailing_benchmark,
    selected_action_ids,
)

from validation.sailing.sailing_yacht_crew_repeated_event_benchmark import (
    build_default_disturbance_sequence,
    prediction_error_series as baseline_prediction_error_series,
    run_repeated_event_benchmark,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def result() -> MemoryEnabledSailingBenchmarkResult:
    return run_memory_enabled_sailing_benchmark()


@pytest.fixture(scope="module")
def baseline():
    return run_repeated_event_benchmark()


# =============================================================================
# Identity
# =============================================================================


def test_scenario_id_is_04d() -> None:
    assert (
        SCENARIO_ID
        == "sailing_04D_memory_enabled"
    )


def test_default_sequence_id_is_stable() -> None:
    assert (
        DEFAULT_SEQUENCE_ID
        == "04D_memory_enabled_sequence"
    )


def test_memory_pattern_id_is_a() -> None:
    assert MEMORY_PATTERN_ID == "A"


# =============================================================================
# Complete benchmark shape
# =============================================================================


def test_complete_run_returns_result(
    result,
) -> None:
    assert isinstance(
        result,
        MemoryEnabledSailingBenchmarkResult,
    )


def test_complete_run_contains_five_episodes(
    result,
) -> None:
    assert len(
        result.episodes
    ) == 5


def test_all_episode_results_have_expected_type(
    result,
) -> None:
    assert all(
        isinstance(
            episode,
            MemoryEnabledEpisodeResult,
        )
        for episode
        in result.episodes
    )


def test_final_memory_contains_all_five_traces(
    result,
) -> None:
    assert result.final_memory.size == 5


def test_final_memory_is_controller_memory(
    result,
) -> None:
    assert isinstance(
        result.final_memory,
        ControllerMemory,
    )


# =============================================================================
# Action policy remains unchanged
# =============================================================================


def test_all_actions_remain_coupled_correction(
    result,
) -> None:
    assert selected_action_ids(
        result
    ) == (
        "coupled_helm_trim_action",
        "coupled_helm_trim_action",
        "coupled_helm_trim_action",
        "coupled_helm_trim_action",
        "coupled_helm_trim_action",
    )


def test_result_declares_policy_override_disabled(
    result,
) -> None:
    assert (
        result.metadata[
            "policy_override_enabled"
        ]
        is False
    )


def test_each_episode_declares_memory_not_used_for_action_selection(
    result,
) -> None:
    assert all(
        episode.metadata[
            "memory_retrieval_used_for_action_selection"
        ]
        is False
        for episode
        in result.episodes
    )


def test_each_episode_declares_policy_not_modified_by_memory(
    result,
) -> None:
    assert all(
        episode.metadata[
            "policy_modified_by_memory"
        ]
        is False
        for episode
        in result.episodes
    )


# =============================================================================
# Pattern A memory timing
# =============================================================================


def test_pattern_a_memory_usage_is_false_false_true(
    result,
) -> None:
    assert pattern_a_memory_usage(
        result
    ) == (
        False,
        False,
        True,
    )


def test_first_a_has_no_preload(
    result,
) -> None:
    a_episodes = tuple(
        episode
        for episode
        in result.episodes
        if (
            episode.event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert a_episodes[
        0
    ].preload is None


def test_second_a_has_no_preload(
    result,
) -> None:
    a_episodes = tuple(
        episode
        for episode
        in result.episodes
        if (
            episode.event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert a_episodes[
        1
    ].preload is None


def test_third_a_has_predictive_preload(
    result,
) -> None:
    a_episodes = tuple(
        episode
        for episode
        in result.episodes
        if (
            episode.event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert isinstance(
        a_episodes[
            2
        ].preload,
        PredictivePreload,
    )


def test_third_a_preload_is_policy_free(
    result,
) -> None:
    a_episodes = tuple(
        episode
        for episode
        in result.episodes
        if (
            episode.event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert preload_is_policy_free(
        a_episodes[
            2
        ].preload
    ) is True


# =============================================================================
# Scar formation
# =============================================================================


def test_first_a_does_not_create_usable_scar_before_next_repeat(
    result,
) -> None:
    a_episodes = tuple(
        episode
        for episode
        in result.episodes
        if (
            episode.event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert a_episodes[
        0
    ].scar_after is None


def test_second_a_creates_memory_scar(
    result,
) -> None:
    a_episodes = tuple(
        episode
        for episode
        in result.episodes
        if (
            episode.event.metadata[
                "pattern_family"
            ]
            == "A"
        )
    )

    assert isinstance(
        a_episodes[
            1
        ].scar_after,
        MemoryScar,
    )


def test_second_a_scar_confidence_is_point_five(
    result,
) -> None:
    series = pattern_a_scar_confidence_series(
        result
    )

    assert series[
        1
    ] == pytest.approx(
        0.5
    )


def test_third_a_scar_confidence_is_point_six(
    result,
) -> None:
    series = pattern_a_scar_confidence_series(
        result
    )

    assert series[
        2
    ] == pytest.approx(
        0.6
    )


def test_pattern_a_scar_confidence_series_is_expected(
    result,
) -> None:
    assert pattern_a_scar_confidence_series(
        result
    ) == (
        None,
        pytest.approx(
            0.5
        ),
        pytest.approx(
            0.6
        ),
    )


# =============================================================================
# Error reduction
# =============================================================================


def test_pattern_a_error_series_has_three_values(
    result,
) -> None:
    assert len(
        result.pattern_a_error_series
    ) == 3


def test_first_two_a_errors_match_no_memory_baseline(
    result,
) -> None:
    assert result.pattern_a_error_series[
        0
    ] == pytest.approx(
        0.08944271909999162
    )

    assert result.pattern_a_error_series[
        1
    ] == pytest.approx(
        0.08944271909999157
    )


def test_third_a_error_is_reduced(
    result,
) -> None:
    assert (
        result.pattern_a_error_series[
            2
        ]
        <
        result.pattern_a_error_series[
            1
        ]
    )


def test_third_a_error_matches_regression(
    result,
) -> None:
    assert result.pattern_a_error_series[
        2
    ] == pytest.approx(
        0.022360679774998015
    )


def test_third_a_error_is_approximately_quarter_of_baseline(
    result,
) -> None:
    assert result.pattern_a_error_series[
        2
    ] == pytest.approx(
        result.pattern_a_error_series[
            0
        ]
        * 0.25
    )


def test_memory_reduces_a_error_by_about_75_percent(
    result,
) -> None:
    baseline_error = result.pattern_a_error_series[
        0
    ]

    learned_error = result.pattern_a_error_series[
        2
    ]

    reduction_fraction = (
        baseline_error
        - learned_error
    ) / baseline_error

    assert reduction_fraction == pytest.approx(
        0.75
    )


# =============================================================================
# Direct contrast with 04B
# =============================================================================


def test_04b_repeated_a_errors_do_not_improve(
    baseline,
) -> None:
    errors = baseline_prediction_error_series(
        baseline
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


def test_04d_third_a_beats_04b_third_a(
    result,
    baseline,
) -> None:
    baseline_errors = baseline_prediction_error_series(
        baseline
    )

    assert (
        result.pattern_a_error_series[
            2
        ]
        <
        baseline_errors[
            4
        ]
    )


def test_04d_first_a_matches_04b_first_a(
    result,
    baseline,
) -> None:
    baseline_errors = baseline_prediction_error_series(
        baseline
    )

    assert result.pattern_a_error_series[
        0
    ] == pytest.approx(
        baseline_errors[
            0
        ]
    )


# =============================================================================
# Preload magnitude
# =============================================================================


def test_pattern_a_preload_series_is_zero_zero_positive(
    result,
) -> None:
    assert result.pattern_a_preload_series[
        0
    ] == pytest.approx(
        0.0
    )

    assert result.pattern_a_preload_series[
        1
    ] == pytest.approx(
        0.0
    )

    assert result.pattern_a_preload_series[
        2
    ] > 0.0


def test_third_a_preload_matches_regression(
    result,
) -> None:
    assert result.pattern_a_preload_series[
        2
    ] == pytest.approx(
        0.0670820393249937
    )


def test_preload_plus_remaining_error_reconstructs_original_bias_norm(
    result,
) -> None:
    assert (
        result.pattern_a_preload_series[
            2
        ]
        +
        result.pattern_a_error_series[
            2
        ]
        == pytest.approx(
            result.pattern_a_error_series[
                0
            ]
        )
    )


# =============================================================================
# Independent evaluator-side observation
# =============================================================================


def test_memory_preload_is_not_used_to_generate_observation(
    result,
) -> None:
    memory_episode = next(
        episode
        for episode
        in result.episodes
        if episode.preload is not None
    )

    assert (
        memory_episode.observed_state.metadata[
            "memory_preload_used_to_generate_observation"
        ]
        is False
    )


def test_observation_source_is_independent_raw_state_evaluator(
    result,
) -> None:
    memory_episode = next(
        episode
        for episode
        in result.episodes
        if episode.preload is not None
    )

    assert (
        memory_episode.observed_state.metadata[
            "observation_source"
        ]
        == "independent_raw_state_plant_evaluator"
    )


# =============================================================================
# Memory growth
# =============================================================================


def test_memory_size_increases_one_per_episode(
    result,
) -> None:
    assert tuple(
        episode.memory_after.size
        for episode
        in result.episodes
    ) == (
        1,
        2,
        3,
        4,
        5,
    )


def test_memory_before_after_chain_is_continuous(
    result,
) -> None:
    for previous, current in zip(
        result.episodes,
        result.episodes[
            1:
        ],
    ):
        assert (
            current.memory_before
            == previous.memory_after
        )


def test_final_memory_equals_last_episode_memory_after(
    result,
) -> None:
    assert (
        result.final_memory
        == result.episodes[
            -1
        ].memory_after
    )


# =============================================================================
# Metadata / audit
# =============================================================================


def test_result_declares_memory_enabled(
    result,
) -> None:
    assert (
        result.metadata[
            "memory_enabled"
        ]
        is True
    )


def test_result_declares_predictive_preload_enabled(
    result,
) -> None:
    assert (
        result.metadata[
            "predictive_preload_enabled"
        ]
        is True
    )


def test_result_declares_pattern_labels_evaluator_only(
    result,
) -> None:
    assert (
        result.metadata[
            "pattern_labels_evaluator_only"
        ]
        is True
    )


def test_result_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_all_episode_metadata_declares_no_external_label_usage(
    result,
) -> None:
    assert all(
        episode.metadata[
            "external_expected_label_used"
        ]
        is False
        for episode
        in result.episodes
    )


# =============================================================================
# Guards
# =============================================================================


def test_empty_event_sequence_is_rejected() -> None:
    with pytest.raises(
        MemoryEnabledSailingBenchmarkError
    ):
        run_memory_enabled_sailing_benchmark(
            events=()
        )


def test_derive_pattern_a_scar_returns_none_for_empty_memory() -> None:
    assert derive_pattern_a_scar(
        ControllerMemory()
    ) is None


# =============================================================================
# Immutability
# =============================================================================


def test_result_episodes_are_tuple(
    result,
) -> None:
    assert isinstance(
        result.episodes,
        tuple,
    )


def test_pattern_a_error_series_is_tuple(
    result,
) -> None:
    assert isinstance(
        result.pattern_a_error_series,
        tuple,
    )


def test_pattern_a_preload_series_is_tuple(
    result,
) -> None:
    assert isinstance(
        result.pattern_a_preload_series,
        tuple,
    )


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


def test_result_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.sequence_id = "changed"  # type: ignore[misc]


def test_episode_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.episodes[
            0
        ].episode_index = 99  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def _signature(
    result: MemoryEnabledSailingBenchmarkResult,
):
    return (
        result.scenario_id,
        result.baseline_scenario_id,
        result.sequence_id,
        selected_action_ids(
            result
        ),
        result.pattern_a_error_series,
        result.pattern_a_preload_series,
        pattern_a_memory_usage(
            result
        ),
        pattern_a_scar_confidence_series(
            result
        ),
        result.final_memory.size,
        tuple(
            (
                episode.episode_index,
                episode.event.event_id,
                episode.decision.selected_candidate_id,
                episode.preload is not None,
                episode.trace.prediction_error.l2_error
                if episode.trace.prediction_error is not None
                else None,
                episode.memory_after.size,
            )
            for episode
            in result.episodes
        ),
        tuple(
            sorted(
                result.metadata.items()
            )
        ),
    )


def test_complete_memory_enabled_benchmark_is_deterministic() -> None:
    left = run_memory_enabled_sailing_benchmark()
    right = run_memory_enabled_sailing_benchmark()

    assert _signature(
        left
    ) == _signature(
        right
    )


def test_single_memory_enabled_episode_is_deterministic() -> None:
    events = build_default_disturbance_sequence()

    base_memory = ControllerMemory()

    left = run_memory_enabled_episode(
        memory=base_memory,
        event=events[
            0
        ],
        episode_index=1,
    )

    right = run_memory_enabled_episode(
        memory=base_memory,
        event=events[
            0
        ],
        episode_index=1,
    )

    assert left.trace.prediction_error == right.trace.prediction_error
    assert (
        left.decision.selected_candidate_id
        == right.decision.selected_candidate_id
    )
    assert left.memory_after == right.memory_after
