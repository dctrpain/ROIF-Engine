"""
Tests for Scenario 04H — Recursive Memory Scaling Benchmark.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from validation.sailing.sailing_yacht_crew_recursive_memory_scaling import (
    DEFAULT_EVENT_COUNTS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_QUERY_OFFSET,
    SCENARIO_ID,
    RecursiveMemoryScalingCheckpoint,
    RecursiveMemoryScalingError,
    RecursiveMemoryScalingResult,
    RecursiveMemoryScalingSummary,
    active_ancestry_series,
    active_context_series,
    base_cluster_series,
    build_scaling_checkpoint,
    normalize_event_counts,
    raw_record_series,
    raw_to_active_context_series,
    recursive_node_series,
    run_recursive_memory_scaling_benchmark,
    summarize_scaling,
    tree_depth_series,
)


SMOKE_COUNTS = (64, 128, 256, 512)


@pytest.fixture(scope="module")
def result() -> RecursiveMemoryScalingResult:
    return run_recursive_memory_scaling_benchmark(
        event_counts=SMOKE_COUNTS
    )


def test_scenario_id_is_04h() -> None:
    assert SCENARIO_ID == "sailing_04H_recursive_memory_scaling"


def test_default_event_counts() -> None:
    assert DEFAULT_EVENT_COUNTS == (64, 128, 256, 512, 1024, 2048)


def test_default_query_offset_is_one() -> None:
    assert DEFAULT_QUERY_OFFSET == 1


def test_default_max_depth_is_positive() -> None:
    assert DEFAULT_MAX_DEPTH > 0


def test_normalize_event_counts_returns_tuple() -> None:
    assert normalize_event_counts([64, 128, 256]) == (64, 128, 256)


def test_normalize_event_counts_rejects_empty() -> None:
    with pytest.raises(RecursiveMemoryScalingError):
        normalize_event_counts(())


def test_normalize_event_counts_rejects_too_small_count() -> None:
    with pytest.raises(RecursiveMemoryScalingError):
        normalize_event_counts((7, 64))


def test_normalize_event_counts_rejects_duplicate_count() -> None:
    with pytest.raises(RecursiveMemoryScalingError):
        normalize_event_counts((64, 64))


def test_normalize_event_counts_rejects_descending_counts() -> None:
    with pytest.raises(RecursiveMemoryScalingError):
        normalize_event_counts((128, 64))


def test_checkpoint_builder_returns_checkpoint() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    assert isinstance(checkpoint, RecursiveMemoryScalingCheckpoint)


def test_checkpoint_query_index_is_event_count_plus_one() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    assert checkpoint.query_index == 65


def test_checkpoint_rejects_zero_query_offset() -> None:
    with pytest.raises(RecursiveMemoryScalingError):
        build_scaling_checkpoint(event_count=64, query_offset=0)


def test_checkpoint_uses_memory() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    assert checkpoint.used_memory is True


def test_checkpoint_does_not_enable_policy_override() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    assert checkpoint.policy_override_enabled is False


def test_checkpoint_declares_no_external_expected_label_usage() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    assert checkpoint.metadata["external_expected_label_used"] is False


def test_checkpoint_declares_label_free_cluster_assignment() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    assert (
        checkpoint.metadata[
            "cluster_assignment_uses_external_pattern_label"
        ]
        is False
    )


def test_checkpoint_metadata_is_read_only() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    assert isinstance(checkpoint.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        checkpoint.metadata["x"] = 1  # type: ignore[index]


def test_checkpoint_is_frozen() -> None:
    checkpoint = build_scaling_checkpoint(event_count=64)
    with pytest.raises(FrozenInstanceError):
        checkpoint.event_count = 1  # type: ignore[misc]


def test_result_is_scaling_result(result) -> None:
    assert isinstance(result, RecursiveMemoryScalingResult)


def test_result_contains_four_smoke_checkpoints(result) -> None:
    assert len(result.checkpoints) == 4


def test_result_checkpoint_counts_match_request(result) -> None:
    assert tuple(c.event_count for c in result.checkpoints) == SMOKE_COUNTS


def test_raw_record_series_regression(result) -> None:
    assert raw_record_series(result) == (64, 128, 256, 512)


def test_base_cluster_series_regression(result) -> None:
    assert base_cluster_series(result) == (2, 2, 3, 3)


def test_recursive_node_series_regression(result) -> None:
    assert recursive_node_series(result) == (3, 3, 6, 6)


def test_tree_depth_series_regression(result) -> None:
    assert tree_depth_series(result) == (2, 2, 3, 3)


def test_active_ancestry_series_regression(result) -> None:
    assert active_ancestry_series(result) == (2, 2, 3, 3)


def test_active_context_series_regression(result) -> None:
    assert active_context_series(result) == (2, 2, 3, 3)


def test_raw_to_active_context_series_regression(result) -> None:
    assert raw_to_active_context_series(result) == pytest.approx(
        (32.0, 64.0, 85.33333333333333, 170.66666666666666)
    )


def test_raw_records_strictly_increase(result) -> None:
    series = raw_record_series(result)
    assert all(a < b for a, b in zip(series, series[1:]))


def test_active_context_does_not_grow_at_every_checkpoint(result) -> None:
    assert active_context_series(result) == (2, 2, 3, 3)


def test_tree_depth_grows_stepwise(result) -> None:
    assert tree_depth_series(result) == (2, 2, 3, 3)


def test_active_context_never_exceeds_tree_depth(result) -> None:
    assert all(
        active <= depth
        for active, depth in zip(
            active_context_series(result),
            tree_depth_series(result),
        )
    )


def test_raw_to_active_context_ratio_strictly_increases(result) -> None:
    series = raw_to_active_context_series(result)
    assert all(a < b for a, b in zip(series, series[1:]))


def test_summary_is_scaling_summary(result) -> None:
    assert isinstance(result.summary, RecursiveMemoryScalingSummary)


def test_raw_growth_factor_regression(result) -> None:
    assert result.summary.raw_growth_factor == pytest.approx(8.0)


def test_active_context_growth_factor_regression(result) -> None:
    assert result.summary.active_context_growth_factor == pytest.approx(1.5)


def test_active_ancestry_growth_factor_regression(result) -> None:
    assert result.summary.active_ancestry_growth_factor == pytest.approx(1.5)


def test_active_context_scaling_exponent_regression(result) -> None:
    assert result.summary.active_context_scaling_exponent == pytest.approx(
        0.19498750024038541
    )


def test_ancestry_scaling_exponent_regression(result) -> None:
    assert result.summary.ancestry_scaling_exponent == pytest.approx(
        0.19498750024038541
    )


def test_recursive_node_scaling_exponent_regression(result) -> None:
    assert result.summary.recursive_node_scaling_exponent == pytest.approx(
        0.33333333333333337
    )


def test_active_context_scaling_is_sublinear(result) -> None:
    assert result.summary.sublinear_active_context is True
    assert result.summary.active_context_scaling_exponent < 1.0


def test_active_ancestry_scaling_is_sublinear(result) -> None:
    assert result.summary.sublinear_active_ancestry is True
    assert result.summary.ancestry_scaling_exponent < 1.0


def test_active_context_exponent_is_below_recursive_node_exponent(result) -> None:
    assert (
        result.summary.active_context_scaling_exponent
        < result.summary.recursive_node_scaling_exponent
    )


def test_final_raw_to_active_context_ratio_regression(result) -> None:
    assert result.summary.final_raw_to_active_context_ratio == pytest.approx(
        170.66666666666666
    )


def test_final_tree_depth_is_three(result) -> None:
    assert result.summary.final_tree_depth == 3


def test_final_active_context_size_is_three(result) -> None:
    assert result.summary.final_active_context_size == 3


def test_summary_declares_first_last_log_scaling(result) -> None:
    assert result.summary.metadata["summary_method"] == "first_last_log_scaling"


def test_summary_declares_no_external_expected_label_usage(result) -> None:
    assert result.summary.metadata["external_expected_label_used"] is False


def test_summary_metadata_is_read_only(result) -> None:
    assert isinstance(result.summary.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        result.summary.metadata["x"] = 1  # type: ignore[index]


def test_summary_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.summary.final_tree_depth = 1  # type: ignore[misc]


def test_recomputed_summary_equals_result_summary(result) -> None:
    assert summarize_scaling(result.checkpoints) == result.summary


def test_result_declares_recursive_tree_enabled(result) -> None:
    assert result.metadata["recursive_tree_enabled"] is True


def test_result_declares_deep_recursive_retrieval(result) -> None:
    assert result.metadata["deep_recursive_retrieval"] is True


def test_result_declares_policy_override_disabled(result) -> None:
    assert result.metadata["policy_override_enabled"] is False


def test_result_declares_memory_does_not_select_action(result) -> None:
    assert result.metadata["action_selected_by_memory"] is False


def test_result_declares_label_free_cluster_assignment(result) -> None:
    assert (
        result.metadata[
            "cluster_assignment_uses_external_pattern_label"
        ]
        is False
    )


def test_result_declares_no_external_expected_label_usage(result) -> None:
    assert result.metadata["external_expected_label_used"] is False


def test_every_checkpoint_is_policy_free(result) -> None:
    assert all(
        checkpoint.policy_override_enabled is False
        for checkpoint in result.checkpoints
    )


def test_every_checkpoint_uses_memory(result) -> None:
    assert all(
        checkpoint.used_memory is True
        for checkpoint in result.checkpoints
    )


def test_result_checkpoints_are_tuple(result) -> None:
    assert isinstance(result.checkpoints, tuple)


def test_result_metadata_is_read_only(result) -> None:
    assert isinstance(result.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_result_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.scenario_id = "x"  # type: ignore[misc]


def _signature(result: RecursiveMemoryScalingResult):
    return (
        result.scenario_id,
        tuple(
            (
                c.event_count,
                c.query_index,
                c.raw_record_count,
                c.base_cluster_count,
                c.recursive_node_count,
                c.recursive_tree_depth,
                c.root_count,
                c.active_ancestry_length,
                c.active_contribution_count,
                c.raw_to_cluster_ratio,
                c.raw_to_recursive_node_ratio,
                c.raw_to_active_context_ratio,
                c.used_memory,
                c.policy_override_enabled,
                tuple(sorted(c.metadata.items())),
            )
            for c in result.checkpoints
        ),
        result.summary,
        tuple(sorted(result.metadata.items())),
    )


def test_complete_04h_smoke_benchmark_is_deterministic() -> None:
    left = run_recursive_memory_scaling_benchmark(
        event_counts=SMOKE_COUNTS
    )
    right = run_recursive_memory_scaling_benchmark(
        event_counts=SMOKE_COUNTS
    )
    assert _signature(left) == _signature(right)


def test_single_checkpoint_is_deterministic() -> None:
    left = build_scaling_checkpoint(event_count=64)
    right = build_scaling_checkpoint(event_count=64)
    assert left == right


def test_two_checkpoint_summary_detects_sublinear_active_context() -> None:
    result = run_recursive_memory_scaling_benchmark(
        event_counts=(64, 512)
    )
    assert result.summary.sublinear_active_context is True


def test_custom_event_counts_are_preserved() -> None:
    result = run_recursive_memory_scaling_benchmark(
        event_counts=(64, 128)
    )
    assert raw_record_series(result) == (64, 128)
