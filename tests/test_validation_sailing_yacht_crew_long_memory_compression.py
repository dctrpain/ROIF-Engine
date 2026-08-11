"""
Tests for Scenario 04C — Long-Memory Compression Benchmark

The suite fixes the first large-scale controller-memory compression contract.

Validated stack:

    Predictive Control
        -> ExperienceTrace
        -> ControllerHistoryAdapter
        -> StructuralSignature
        -> HierarchicalMemory
        -> Compression Metrics

Core invariants:

    Compression != Learning
    Compression != PredictivePreload
    Cluster Assignment != External Ground Truth
    Anomaly != Deletion Candidate

The tests use deterministic 04C streams and verify:
- event-family distribution;
- bounded cluster growth;
- increasing compression;
- preserved representatives;
- preserved anomalies;
- strong same/cross pattern separation;
- evaluator-only labels;
- preload-free memory;
- deterministic 100- and 1000-event results.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.hierarchical_memory import (
    hierarchical_memory_is_preload_free,
)

from validation.sailing.sailing_yacht_crew_long_memory_compression import (
    DEFAULT_EVENT_COUNT,
    DEFAULT_SEQUENCE_ID,
    PATTERN_LIBRARY,
    SCENARIO_ID,
    LongMemoryBenchmarkError,
    LongMemoryBenchmarkResult,
    LongMemoryCheckpoint,
    LongMemoryEventSpec,
    build_long_memory_event,
    build_long_memory_sequence,
    checkpoint_compression_series,
    cluster_anomaly_counts,
    cluster_sizes,
    event_to_controller_history_record,
    run_long_memory_compression_benchmark,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def result_100() -> LongMemoryBenchmarkResult:
    return run_long_memory_compression_benchmark(
        event_count=100,
        checkpoint_interval=20,
    )


@pytest.fixture(scope="module")
def result_1000() -> LongMemoryBenchmarkResult:
    return run_long_memory_compression_benchmark(
        event_count=1000,
        checkpoint_interval=100,
    )


# =============================================================================
# Scenario identity
# =============================================================================


def test_scenario_id_is_04c() -> None:
    assert (
        SCENARIO_ID
        == "sailing_04C_long_memory_compression"
    )


def test_default_event_count_is_1000() -> None:
    assert DEFAULT_EVENT_COUNT == 1000


def test_default_sequence_id_is_stable() -> None:
    assert (
        DEFAULT_SEQUENCE_ID
        == "04C_long_memory_sequence"
    )


# =============================================================================
# Pattern library
# =============================================================================


def test_pattern_library_contains_a_b_c_d() -> None:
    assert tuple(
        PATTERN_LIBRARY.keys()
    ) == (
        "A",
        "B",
        "C",
        "D",
    )


def test_pattern_library_is_read_only() -> None:
    assert isinstance(
        PATTERN_LIBRARY,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        PATTERN_LIBRARY[
            "X"
        ] = LongMemoryEventSpec(  # type: ignore[index]
            pattern_family="X",
            wind=0.1,
            wave=0.1,
            course_bias=0.1,
            heel_bias=0.1,
        )


def test_event_spec_rejects_empty_family() -> None:
    with pytest.raises(
        LongMemoryBenchmarkError
    ):
        LongMemoryEventSpec(
            pattern_family="",
            wind=0.1,
            wave=0.1,
            course_bias=0.1,
            heel_bias=0.1,
        )


def test_event_spec_is_frozen() -> None:
    spec = PATTERN_LIBRARY[
        "A"
    ]

    with pytest.raises(
        FrozenInstanceError
    ):
        spec.wind = 1.0  # type: ignore[misc]


# =============================================================================
# Deterministic event generation
# =============================================================================


def test_event_builder_rejects_nonpositive_index() -> None:
    with pytest.raises(
        LongMemoryBenchmarkError
    ):
        build_long_memory_event(
            0
        )


def test_first_event_is_family_a() -> None:
    event = build_long_memory_event(
        1
    )

    assert (
        event.metadata[
            "pattern_family"
        ]
        == "A"
    )


def test_second_event_is_family_b() -> None:
    event = build_long_memory_event(
        2
    )

    assert (
        event.metadata[
            "pattern_family"
        ]
        == "B"
    )


def test_fourth_event_is_family_c() -> None:
    event = build_long_memory_event(
        4
    )

    assert (
        event.metadata[
            "pattern_family"
        ]
        == "C"
    )


def test_every_97th_event_is_family_d() -> None:
    event = build_long_memory_event(
        97
    )

    assert (
        event.metadata[
            "pattern_family"
        ]
        == "D"
    )


def test_pattern_label_is_evaluator_only() -> None:
    event = build_long_memory_event(
        1
    )

    assert (
        event.metadata[
            "evaluator_only_pattern_label"
        ]
        is True
    )


def test_event_builder_declares_no_external_expected_label_usage() -> None:
    event = build_long_memory_event(
        1
    )

    assert (
        event.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_event_generation_is_deterministic() -> None:
    assert (
        build_long_memory_event(
            123
        )
        == build_long_memory_event(
            123
        )
    )


# =============================================================================
# Sequence generation
# =============================================================================


def test_sequence_rejects_zero_events() -> None:
    with pytest.raises(
        LongMemoryBenchmarkError
    ):
        build_long_memory_sequence(
            0
        )


def test_sequence_has_requested_length() -> None:
    sequence = build_long_memory_sequence(
        100
    )

    assert len(
        sequence
    ) == 100


def test_100_event_family_counts_are_expected() -> None:
    sequence = build_long_memory_sequence(
        100
    )

    counts = {
        family: sum(
            1
            for event
            in sequence
            if (
                event.metadata[
                    "pattern_family"
                ]
                == family
            )
        )
        for family
        in PATTERN_LIBRARY
    }

    assert counts == {
        "A": 49,
        "B": 33,
        "C": 17,
        "D": 1,
    }


def test_1000_event_family_counts_are_expected() -> None:
    sequence = build_long_memory_sequence(
        1000
    )

    counts = {
        family: sum(
            1
            for event
            in sequence
            if (
                event.metadata[
                    "pattern_family"
                ]
                == family
            )
        )
        for family
        in PATTERN_LIBRARY
    }

    assert counts == {
        "A": 495,
        "B": 330,
        "C": 165,
        "D": 10,
    }


# =============================================================================
# Real adapter path
# =============================================================================


def test_event_to_record_returns_controller_history_record() -> None:
    event = build_long_memory_event(
        1
    )

    record = event_to_controller_history_record(
        event=event,
        episode_index=1,
    )

    assert record.trace_id.endswith(
        event.event_id
    )


def test_event_to_record_preserves_evaluator_pattern_only_as_reporting_id() -> None:
    event = build_long_memory_event(
        1
    )

    record = event_to_controller_history_record(
        event=event,
        episode_index=1,
    )

    assert record.pattern_id == "A"


def test_event_to_record_remains_preload_free() -> None:
    event = build_long_memory_event(
        1
    )

    record = event_to_controller_history_record(
        event=event,
        episode_index=1,
    )

    assert (
        record.metadata[
            "predictive_preload_applied"
        ]
        is False
    )


# =============================================================================
# 100-event benchmark
# =============================================================================


def test_100_result_is_benchmark_result(
    result_100,
) -> None:
    assert isinstance(
        result_100,
        LongMemoryBenchmarkResult,
    )


def test_100_result_has_100_events(
    result_100,
) -> None:
    assert result_100.event_count == 100


def test_100_result_has_expected_family_counts(
    result_100,
) -> None:
    assert dict(
        result_100.family_counts
    ) == {
        "A": 49,
        "B": 33,
        "C": 17,
        "D": 1,
    }


def test_100_result_has_two_clusters(
    result_100,
) -> None:
    assert result_100.memory.cluster_count == 2


def test_100_cluster_sizes_are_50_50(
    result_100,
) -> None:
    assert cluster_sizes(
        result_100
    ) == (
        50,
        50,
    )


def test_100_anomaly_counts_are_one_one(
    result_100,
) -> None:
    assert cluster_anomaly_counts(
        result_100
    ) == (
        1,
        1,
    )


def test_100_compression_ratio_matches_regression(
    result_100,
) -> None:
    assert (
        result_100.compression.compression_ratio
        == pytest.approx(
            8.333333333333334
        )
    )


def test_100_same_family_distance_is_small(
    result_100,
) -> None:
    assert (
        result_100.mean_same_family_distance
        < 0.01
    )


def test_100_cross_family_distance_is_large(
    result_100,
) -> None:
    assert (
        result_100.mean_cross_family_distance
        > 1.0
    )


def test_100_separation_exceeds_500x(
    result_100,
) -> None:
    assert (
        result_100.separation_ratio
        > 500.0
    )


def test_100_checkpoint_series_is_monotonic(
    result_100,
) -> None:
    series = checkpoint_compression_series(
        result_100
    )

    ratios = tuple(
        ratio
        for _count, ratio
        in series
    )

    assert all(
        left < right
        for left, right
        in zip(
            ratios,
            ratios[
                1:
            ],
        )
    )


# =============================================================================
# 1000-event benchmark
# =============================================================================


def test_1000_result_has_1000_events(
    result_1000,
) -> None:
    assert result_1000.event_count == 1000


def test_1000_result_has_expected_family_counts(
    result_1000,
) -> None:
    assert dict(
        result_1000.family_counts
    ) == {
        "A": 495,
        "B": 330,
        "C": 165,
        "D": 10,
    }


def test_1000_cluster_count_remains_bounded(
    result_1000,
) -> None:
    assert result_1000.memory.cluster_count == 2


def test_1000_cluster_sizes_match_regression(
    result_1000,
) -> None:
    assert cluster_sizes(
        result_1000
    ) == (
        505,
        495,
    )


def test_1000_anomaly_counts_match_regression(
    result_1000,
) -> None:
    assert cluster_anomaly_counts(
        result_1000
    ) == (
        10,
        1,
    )


def test_1000_raw_record_count_is_1000(
    result_1000,
) -> None:
    assert (
        result_1000.compression.raw_record_count
        == 1000
    )


def test_1000_representative_count_is_10(
    result_1000,
) -> None:
    assert (
        result_1000.compression.representative_count
        == 10
    )


def test_1000_effective_retained_count_is_21(
    result_1000,
) -> None:
    assert (
        result_1000.compression.effective_retained_count
        == 21
    )


def test_1000_compression_ratio_exceeds_40x(
    result_1000,
) -> None:
    assert (
        result_1000.compression.compression_ratio
        > 40.0
    )


def test_1000_compression_ratio_matches_regression(
    result_1000,
) -> None:
    assert (
        result_1000.compression.compression_ratio
        == pytest.approx(
            47.61904761904762
        )
    )


def test_1000_same_family_distance_remains_small(
    result_1000,
) -> None:
    assert (
        result_1000.mean_same_family_distance
        < 0.01
    )


def test_1000_cross_family_distance_remains_large(
    result_1000,
) -> None:
    assert (
        result_1000.mean_cross_family_distance
        > 1.0
    )


def test_1000_separation_exceeds_500x(
    result_1000,
) -> None:
    assert (
        result_1000.separation_ratio
        > 500.0
    )


def test_1000_separation_matches_regression(
    result_1000,
) -> None:
    assert (
        result_1000.separation_ratio
        == pytest.approx(
            545.1663580623938
        )
    )


# =============================================================================
# Scaling behavior
# =============================================================================


def test_compression_improves_from_100_to_1000(
    result_100,
    result_1000,
) -> None:
    assert (
        result_1000.compression.compression_ratio
        >
        result_100.compression.compression_ratio
    )


def test_cluster_count_does_not_scale_with_raw_events(
    result_100,
    result_1000,
) -> None:
    assert (
        result_1000.memory.cluster_count
        == result_100.memory.cluster_count
    )


def test_representatives_remain_small_relative_to_raw_memory(
    result_1000,
) -> None:
    assert (
        result_1000.compression.representative_count
        / result_1000.compression.raw_record_count
        < 0.02
    )


def test_effective_retention_is_small_relative_to_raw_memory(
    result_1000,
) -> None:
    assert (
        result_1000.compression.effective_retained_count
        / result_1000.compression.raw_record_count
        < 0.03
    )


def test_rare_d_events_are_preserved_as_anomalies(
    result_1000,
) -> None:
    assert (
        sum(
            cluster_anomaly_counts(
                result_1000
            )
        )
        >= result_1000.family_counts[
            "D"
        ]
    )


# =============================================================================
# Checkpoints
# =============================================================================


def test_1000_has_ten_checkpoints(
    result_1000,
) -> None:
    assert len(
        result_1000.checkpoints
    ) == 10


def test_1000_checkpoint_event_counts_are_100_to_1000(
    result_1000,
) -> None:
    assert tuple(
        checkpoint.event_count
        for checkpoint
        in result_1000.checkpoints
    ) == (
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    )


def test_1000_checkpoint_compression_is_strictly_increasing(
    result_1000,
) -> None:
    ratios = tuple(
        checkpoint.compression_ratio
        for checkpoint
        in result_1000.checkpoints
    )

    assert all(
        left < right
        for left, right
        in zip(
            ratios,
            ratios[
                1:
            ],
        )
    )


def test_last_checkpoint_matches_final_compression(
    result_1000,
) -> None:
    last = result_1000.checkpoints[
        -1
    ]

    assert (
        last.compression_ratio
        == pytest.approx(
            result_1000.compression.compression_ratio
        )
    )


# =============================================================================
# Audit boundary
# =============================================================================


def test_1000_memory_is_preload_free(
    result_1000,
) -> None:
    assert hierarchical_memory_is_preload_free(
        result_1000.memory
    ) is True


def test_result_metadata_declares_cluster_assignment_label_free(
    result_1000,
) -> None:
    assert (
        result_1000.metadata[
            "cluster_assignment_uses_external_pattern_label"
        ]
        is False
    )


def test_result_metadata_declares_pattern_labels_evaluator_only(
    result_1000,
) -> None:
    assert (
        result_1000.metadata[
            "pattern_labels_evaluator_only"
        ]
        is True
    )


def test_result_metadata_declares_learning_disabled(
    result_1000,
) -> None:
    assert (
        result_1000.metadata[
            "learning_enabled"
        ]
        is False
    )


def test_result_metadata_declares_predictive_preload_disabled(
    result_1000,
) -> None:
    assert (
        result_1000.metadata[
            "predictive_preload_enabled"
        ]
        is False
    )


def test_result_metadata_preserves_representatives(
    result_1000,
) -> None:
    assert (
        result_1000.metadata[
            "representatives_preserved"
        ]
        is True
    )


def test_result_metadata_preserves_anomalies(
    result_1000,
) -> None:
    assert (
        result_1000.metadata[
            "anomalies_preserved"
        ]
        is True
    )


def test_result_declares_no_external_expected_label_usage(
    result_1000,
) -> None:
    assert (
        result_1000.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Immutability
# =============================================================================


def test_result_checkpoints_are_tuple(
    result_1000,
) -> None:
    assert isinstance(
        result_1000.checkpoints,
        tuple,
    )


def test_result_family_counts_are_read_only(
    result_1000,
) -> None:
    assert isinstance(
        result_1000.family_counts,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result_1000.family_counts[
            "A"
        ] = 0  # type: ignore[index]


def test_result_metadata_is_read_only(
    result_1000,
) -> None:
    assert isinstance(
        result_1000.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result_1000.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_checkpoint_is_frozen(
    result_1000,
) -> None:
    checkpoint = result_1000.checkpoints[
        0
    ]

    assert isinstance(
        checkpoint,
        LongMemoryCheckpoint,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        checkpoint.event_count = 1  # type: ignore[misc]


def test_result_is_frozen(
    result_1000,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result_1000.event_count = 1  # type: ignore[misc]


# =============================================================================
# Guards
# =============================================================================


def test_benchmark_rejects_zero_event_count() -> None:
    with pytest.raises(
        LongMemoryBenchmarkError
    ):
        run_long_memory_compression_benchmark(
            event_count=0
        )


def test_benchmark_rejects_zero_checkpoint_interval() -> None:
    with pytest.raises(
        LongMemoryBenchmarkError
    ):
        run_long_memory_compression_benchmark(
            event_count=10,
            checkpoint_interval=0,
        )


# =============================================================================
# Determinism
# =============================================================================


def _result_signature(
    result: LongMemoryBenchmarkResult,
):
    return (
        result.scenario_id,
        result.event_count,
        tuple(
            sorted(
                result.family_counts.items()
            )
        ),
        tuple(
            cluster_sizes(
                result
            )
        ),
        tuple(
            cluster_anomaly_counts(
                result
            )
        ),
        result.compression.raw_record_count,
        result.compression.cluster_count,
        result.compression.representative_count,
        result.compression.anomaly_count,
        result.compression.effective_retained_count,
        result.compression.compression_ratio,
        result.mean_same_family_distance,
        result.mean_cross_family_distance,
        result.separation_ratio,
        checkpoint_compression_series(
            result
        ),
        tuple(
            sorted(
                result.metadata.items()
            )
        ),
    )


def test_100_event_benchmark_is_deterministic() -> None:
    left = run_long_memory_compression_benchmark(
        event_count=100,
        checkpoint_interval=20,
    )

    right = run_long_memory_compression_benchmark(
        event_count=100,
        checkpoint_interval=20,
    )

    assert _result_signature(
        left
    ) == _result_signature(
        right
    )


def test_1000_event_benchmark_is_deterministic(
    result_1000,
) -> None:
    second = run_long_memory_compression_benchmark(
        event_count=1000,
        checkpoint_interval=100,
    )

    assert _result_signature(
        result_1000
    ) == _result_signature(
        second
    )
