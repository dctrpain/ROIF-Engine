"""
Tests for Scenario 04J — Tensor Node Capacity / Interference Benchmark.

The suite fixes the observed 2 -> 4 -> 8 -> 16 capacity curve and validates
the architectural boundaries without inventing unmeasured 32/64 regressions.

Observed smoke regression:
    unique profile fraction = 1.0 throughout
    ambiguity rises strongly
    interference rises strongly
    dominant margin collapses
    collisions remain zero through 16 images
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from validation.sailing.sailing_yacht_crew_tensor_node_capacity_interference import (
    DEFAULT_HIGH_AMBIGUITY_THRESHOLD,
    DEFAULT_IMAGE_COUNTS,
    DEFAULT_PROFILE_PRECISION,
    DEFAULT_QUERY_REPETITIONS,
    SCENARIO_ID,
    TensorNodeCapacityCheckpoint,
    TensorNodeCapacityError,
    TensorNodeCapacityResult,
    TensorNodeCapacitySummary,
    build_capacity_checkpoint,
    collision_fraction_series,
    dominant_coverage_series,
    dominant_margin_series,
    high_ambiguity_fraction_series,
    image_count_series,
    mean_ambiguity_series,
    mean_interference_series,
    normalize_image_counts,
    run_tensor_node_capacity_benchmark,
    summarize_capacity,
    unique_profile_fraction_series,
)


SMOKE_COUNTS = (2, 4, 8, 16)


@pytest.fixture(scope="module")
def result() -> TensorNodeCapacityResult:
    return run_tensor_node_capacity_benchmark(
        image_counts=SMOKE_COUNTS
    )


def test_scenario_id_is_04j() -> None:
    assert SCENARIO_ID == "sailing_04J_tensor_node_capacity_interference"


def test_default_image_counts() -> None:
    assert DEFAULT_IMAGE_COUNTS == (2, 4, 8, 16, 32, 64)


def test_default_query_repetitions_is_one() -> None:
    assert DEFAULT_QUERY_REPETITIONS == 1


def test_default_profile_precision_is_nonnegative() -> None:
    assert DEFAULT_PROFILE_PRECISION >= 0


def test_default_high_ambiguity_threshold_is_unit_interval() -> None:
    assert 0.0 <= DEFAULT_HIGH_AMBIGUITY_THRESHOLD <= 1.0


def test_normalize_image_counts_returns_tuple() -> None:
    assert normalize_image_counts([2, 4, 8]) == (2, 4, 8)


def test_normalize_image_counts_rejects_empty() -> None:
    with pytest.raises(TensorNodeCapacityError):
        normalize_image_counts(())


def test_normalize_image_counts_rejects_values_below_two() -> None:
    with pytest.raises(TensorNodeCapacityError):
        normalize_image_counts((1, 2))


def test_normalize_image_counts_rejects_duplicates() -> None:
    with pytest.raises(TensorNodeCapacityError):
        normalize_image_counts((2, 2))


def test_normalize_image_counts_rejects_descending() -> None:
    with pytest.raises(TensorNodeCapacityError):
        normalize_image_counts((4, 2))


def test_checkpoint_builder_returns_checkpoint() -> None:
    checkpoint = build_capacity_checkpoint(image_count=2)
    assert isinstance(checkpoint, TensorNodeCapacityCheckpoint)


def test_checkpoint_source_record_count_matches_capacity() -> None:
    checkpoint = build_capacity_checkpoint(image_count=4)
    assert checkpoint.source_record_count == 64


def test_checkpoint_query_count_matches_image_count() -> None:
    checkpoint = build_capacity_checkpoint(image_count=8)
    assert checkpoint.query_count == 8


def test_checkpoint_uses_same_physical_node() -> None:
    checkpoint = build_capacity_checkpoint(image_count=8)
    assert checkpoint.same_physical_node is True


def test_checkpoint_is_policy_free() -> None:
    checkpoint = build_capacity_checkpoint(image_count=8)
    assert checkpoint.policy_free is True


def test_checkpoint_rejects_image_count_below_two() -> None:
    with pytest.raises(TensorNodeCapacityError):
        build_capacity_checkpoint(image_count=1)


def test_checkpoint_rejects_zero_query_repetitions() -> None:
    with pytest.raises(TensorNodeCapacityError):
        build_capacity_checkpoint(
            image_count=2,
            query_repetitions=0,
        )


def test_checkpoint_rejects_negative_profile_precision() -> None:
    with pytest.raises(TensorNodeCapacityError):
        build_capacity_checkpoint(
            image_count=2,
            profile_precision=-1,
        )


def test_checkpoint_rejects_invalid_ambiguity_threshold() -> None:
    with pytest.raises(TensorNodeCapacityError):
        build_capacity_checkpoint(
            image_count=2,
            high_ambiguity_threshold=1.1,
        )


def test_result_is_capacity_result(result) -> None:
    assert isinstance(result, TensorNodeCapacityResult)


def test_result_contains_four_checkpoints(result) -> None:
    assert len(result.checkpoints) == 4


def test_image_count_series_regression(result) -> None:
    assert image_count_series(result) == (2, 4, 8, 16)


def test_unique_profile_fraction_regression(result) -> None:
    assert unique_profile_fraction_series(result) == pytest.approx(
        (1.0, 1.0, 1.0, 1.0)
    )


def test_dominant_coverage_regression(result) -> None:
    assert dominant_coverage_series(result) == pytest.approx(
        (1.0, 0.75, 0.5, 0.25)
    )


def test_mean_ambiguity_regression(result) -> None:
    assert mean_ambiguity_series(result) == pytest.approx(
        (
            0.0,
            0.605102716028717,
            0.9353431114487021,
            0.9808650588171764,
        )
    )


def test_high_ambiguity_fraction_regression(result) -> None:
    assert high_ambiguity_fraction_series(result) == pytest.approx(
        (0.0, 0.0, 1.0, 1.0)
    )


def test_mean_interference_regression(result) -> None:
    assert mean_interference_series(result) == pytest.approx(
        (
            0.0,
            0.3761527876079237,
            0.632987678738443,
            0.8392622234561513,
        )
    )


def test_dominant_margin_regression(result) -> None:
    assert dominant_margin_series(result) == pytest.approx(
        (
            0.6104397026000845,
            0.29728106727303694,
            0.047633476132696445,
            0.015492317517276012,
        )
    )


def test_collision_fraction_regression(result) -> None:
    assert collision_fraction_series(result) == pytest.approx(
        (0.0, 0.0, 0.0, 0.0)
    )


def test_ambiguity_strictly_increases(result) -> None:
    series = mean_ambiguity_series(result)
    assert all(a < b for a, b in zip(series, series[1:]))


def test_interference_strictly_increases(result) -> None:
    series = mean_interference_series(result)
    assert all(a < b for a, b in zip(series, series[1:]))


def test_dominant_margin_strictly_decreases(result) -> None:
    series = dominant_margin_series(result)
    assert all(a > b for a, b in zip(series, series[1:]))


def test_dominant_coverage_strictly_decreases(result) -> None:
    series = dominant_coverage_series(result)
    assert all(a > b for a, b in zip(series, series[1:]))


def test_profiles_remain_unique_through_16_images(result) -> None:
    assert all(
        value == pytest.approx(1.0)
        for value in unique_profile_fraction_series(result)
    )


def test_collisions_remain_absent_through_16_images(result) -> None:
    assert all(
        value == pytest.approx(0.0)
        for value in collision_fraction_series(result)
    )


def test_functional_discrimination_degrades_before_profile_collisions(result) -> None:
    assert mean_ambiguity_series(result)[-1] > 0.95
    assert mean_interference_series(result)[-1] > 0.80
    assert dominant_margin_series(result)[-1] < 0.02
    assert collision_fraction_series(result)[-1] == pytest.approx(0.0)


def test_high_ambiguity_reaches_full_fraction_by_eight_images(result) -> None:
    assert high_ambiguity_fraction_series(result)[2:] == pytest.approx(
        (1.0, 1.0)
    )


def test_summary_is_capacity_summary(result) -> None:
    assert isinstance(result.summary, TensorNodeCapacitySummary)


def test_summary_detects_ambiguity_increase(result) -> None:
    assert result.summary.ambiguity_increased is True


def test_summary_detects_interference_increase(result) -> None:
    assert result.summary.interference_increased is True


def test_summary_detects_margin_decrease(result) -> None:
    assert result.summary.dominant_margin_decreased is True


def test_summary_does_not_claim_collision_increase(result) -> None:
    assert result.summary.collisions_increased is False


def test_summary_first_last_counts(result) -> None:
    assert result.summary.first_image_count == 2
    assert result.summary.last_image_count == 16


def test_summary_first_last_ambiguity_regression(result) -> None:
    assert result.summary.first_mean_ambiguity == pytest.approx(0.0)
    assert result.summary.last_mean_ambiguity == pytest.approx(
        0.9808650588171764
    )


def test_summary_first_last_interference_regression(result) -> None:
    assert result.summary.first_mean_interference_ratio == pytest.approx(0.0)
    assert result.summary.last_mean_interference_ratio == pytest.approx(
        0.8392622234561513
    )


def test_summary_first_last_margin_regression(result) -> None:
    assert result.summary.first_mean_dominant_margin == pytest.approx(
        0.6104397026000845
    )
    assert result.summary.last_mean_dominant_margin == pytest.approx(
        0.015492317517276012
    )


def test_summary_recomputation_matches_result(result) -> None:
    assert summarize_capacity(result.checkpoints) == result.summary


def test_result_declares_single_shared_node_per_checkpoint(result) -> None:
    assert result.metadata["single_shared_physical_node_per_checkpoint"] is True


def test_result_declares_structural_geometry_response(result) -> None:
    assert result.metadata["response_derived_from_structural_geometry"] is True


def test_result_declares_memory_does_not_select_action(result) -> None:
    assert result.metadata["memory_selects_action"] is False


def test_result_declares_policy_override_disabled(result) -> None:
    assert result.metadata["policy_override_enabled"] is False


def test_result_declares_no_external_label_usage(result) -> None:
    assert result.metadata["external_expected_label_used"] is False


def test_every_checkpoint_is_same_node_and_policy_free(result) -> None:
    assert all(
        checkpoint.same_physical_node and checkpoint.policy_free
        for checkpoint in result.checkpoints
    )


def test_result_checkpoints_are_tuple(result) -> None:
    assert isinstance(result.checkpoints, tuple)


def test_result_metadata_is_read_only(result) -> None:
    assert isinstance(result.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_checkpoint_metadata_is_read_only(result) -> None:
    checkpoint = result.checkpoints[0]
    assert isinstance(checkpoint.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        checkpoint.metadata["x"] = 1  # type: ignore[index]


def test_summary_metadata_is_read_only(result) -> None:
    assert isinstance(result.summary.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        result.summary.metadata["x"] = 1  # type: ignore[index]


def test_result_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.scenario_id = "changed"  # type: ignore[misc]


def test_checkpoint_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.checkpoints[0].image_count = 99  # type: ignore[misc]


def test_summary_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.summary.last_image_count = 99  # type: ignore[misc]


def _signature(result: TensorNodeCapacityResult):
    return (
        result.scenario_id,
        tuple(
            (
                item.image_count,
                item.history_per_image,
                item.query_repetitions,
                item.source_record_count,
                item.query_count,
                item.unique_response_profile_count,
                item.unique_profile_fraction,
                item.unique_dominant_image_count,
                item.dominant_coverage_fraction,
                item.mean_response_energy,
                item.mean_top1_activation,
                item.mean_top2_activation,
                item.mean_dominant_margin,
                item.mean_ambiguity,
                item.high_ambiguity_fraction,
                item.mean_interference_ratio,
                item.collision_count,
                item.collision_fraction,
                item.same_physical_node,
                item.policy_free,
                tuple(sorted(item.metadata.items())),
            )
            for item in result.checkpoints
        ),
        result.summary,
        tuple(sorted(result.metadata.items())),
    )


def test_complete_04j_smoke_benchmark_is_deterministic() -> None:
    left = run_tensor_node_capacity_benchmark(
        image_counts=SMOKE_COUNTS
    )
    right = run_tensor_node_capacity_benchmark(
        image_counts=SMOKE_COUNTS
    )
    assert _signature(left) == _signature(right)


def test_single_checkpoint_is_deterministic() -> None:
    left = build_capacity_checkpoint(image_count=8)
    right = build_capacity_checkpoint(image_count=8)
    assert left == right


def test_custom_two_point_capacity_summary() -> None:
    result = run_tensor_node_capacity_benchmark(
        image_counts=(2, 16)
    )
    assert result.summary.ambiguity_increased is True
    assert result.summary.interference_increased is True
    assert result.summary.dominant_margin_decreased is True
