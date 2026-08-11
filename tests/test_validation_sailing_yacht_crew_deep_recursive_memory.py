"""
Tests for Scenario 04G — Deep Recursive Memory Benchmark
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.hierarchical_memory_retrieval import retrieval_is_policy_free
from roif.predictive_preload import hierarchy_levels_used, preload_is_policy_free
from roif.recursive_hierarchical_memory import (
    RecursiveMemoryTree,
    ancestry,
    tree_is_policy_free,
)

from validation.sailing.sailing_yacht_crew_deep_recursive_memory import (
    DEFAULT_ASSIGNMENT_RADIUS,
    DEFAULT_EVENT_COUNT,
    DEFAULT_MAX_CHILDREN,
    DEFAULT_QUERY_INDEX,
    DEFAULT_RECURSIVE_MERGE_RADIUS,
    SCENARIO_ID,
    DeepRecursiveMemoryBenchmarkError,
    DeepRecursiveMemoryMetrics,
    DeepRecursiveMemoryResult,
    build_deep_memory_event,
    build_deep_query_record,
    build_deep_recursive_history,
    build_deep_recursive_tree,
    compute_deep_recursive_metrics,
    leaf_cluster_ancestry,
    physical_ancestry_node_ids,
    retrieval_tree_levels,
    retrieval_tree_node_ids,
    run_deep_recursive_memory_benchmark,
    run_deep_recursive_retrieval,
)


@pytest.fixture(scope="module")
def result() -> DeepRecursiveMemoryResult:
    return run_deep_recursive_memory_benchmark()


@pytest.fixture(scope="module")
def memory():
    return build_deep_recursive_history()


@pytest.fixture(scope="module")
def tree(memory) -> RecursiveMemoryTree:
    return build_deep_recursive_tree(memory)


def test_scenario_id_is_04g() -> None:
    assert SCENARIO_ID == "sailing_04G_deep_recursive_memory"


def test_default_event_count_is_256() -> None:
    assert DEFAULT_EVENT_COUNT == 256


def test_default_query_index_is_257() -> None:
    assert DEFAULT_QUERY_INDEX == 257


def test_default_assignment_radius_is_small() -> None:
    assert DEFAULT_ASSIGNMENT_RADIUS == pytest.approx(1.0e-6)


def test_default_max_children_is_binary() -> None:
    assert DEFAULT_MAX_CHILDREN == 2


def test_default_recursive_merge_radius_is_positive() -> None:
    assert DEFAULT_RECURSIVE_MERGE_RADIUS > 0.0


def test_deep_event_builder_returns_unique_04g_id() -> None:
    event = build_deep_memory_event(1)
    assert event.event_id == "04G_event_00001_x00_y00_p00"


def test_deep_event_metadata_marks_deep_event() -> None:
    event = build_deep_memory_event(1)
    assert event.metadata["deep_recursive_event"] is True


def test_deep_event_metadata_has_lattice_coordinates() -> None:
    event = build_deep_memory_event(18)
    assert "deep_lattice_x" in event.metadata
    assert "deep_lattice_y" in event.metadata
    assert "deep_lattice_phase" in event.metadata


def test_deep_event_builder_declares_no_external_label_usage() -> None:
    event = build_deep_memory_event(1)
    assert event.metadata["external_expected_label_used"] is False


def test_deep_event_builder_rejects_nonpositive_index() -> None:
    with pytest.raises(DeepRecursiveMemoryBenchmarkError):
        build_deep_memory_event(0)


def test_deep_event_builder_is_deterministic() -> None:
    assert build_deep_memory_event(123) == build_deep_memory_event(123)


def test_deep_event_generator_changes_across_lattice() -> None:
    assert build_deep_memory_event(1) != build_deep_memory_event(2)


def test_history_contains_256_records(memory) -> None:
    assert memory.record_count == 256


def test_history_has_three_base_clusters(memory) -> None:
    assert memory.cluster_count == 3


def test_history_declares_no_predictive_preload(memory) -> None:
    assert memory.metadata["predictive_preload_applied"] is False


def test_history_declares_no_policy_modification(memory) -> None:
    assert memory.metadata["policy_modified"] is False


def test_history_declares_no_external_expected_label_usage(memory) -> None:
    assert memory.metadata["external_expected_label_used"] is False


def test_history_rejects_less_than_eight_events() -> None:
    with pytest.raises(DeepRecursiveMemoryBenchmarkError):
        build_deep_recursive_history(event_count=7)


def test_history_rejects_negative_assignment_radius() -> None:
    with pytest.raises(DeepRecursiveMemoryBenchmarkError):
        build_deep_recursive_history(event_count=8, assignment_radius=-1.0)


def test_tree_is_recursive_memory_tree(tree) -> None:
    assert isinstance(tree, RecursiveMemoryTree)


def test_tree_has_six_nodes(tree) -> None:
    assert tree.node_count == 6


def test_tree_depth_is_three(tree) -> None:
    assert tree.max_level == 3


def test_tree_has_one_root(tree) -> None:
    assert tree.root_count == 1


def test_tree_preserves_three_leaf_clusters(tree) -> None:
    assert len(tree.leaf_cluster_ids) == 3


def test_tree_is_policy_free(tree) -> None:
    assert tree_is_policy_free(tree) is True


def test_tree_rejects_max_depth_below_three(memory) -> None:
    with pytest.raises(DeepRecursiveMemoryBenchmarkError):
        build_deep_recursive_tree(memory, max_depth=2)


def test_query_record_uses_default_query_index() -> None:
    query = build_deep_query_record()
    assert "00257" in query.trace_id


def test_query_record_builder_rejects_nonpositive_index() -> None:
    with pytest.raises(DeepRecursiveMemoryBenchmarkError):
        build_deep_query_record(query_index=0)


def test_query_record_is_deterministic() -> None:
    left = build_deep_query_record(query_index=257)
    right = build_deep_query_record(query_index=257)

    # Runtime UUID fields are intentionally unique.
    # Determinism here means semantic equivalence.
    assert left.trace_id == right.trace_id
    assert left.sequence_id == right.sequence_id
    assert left.pattern_id == right.pattern_id
    assert left.source_trace == right.source_trace
    assert left.metadata == right.metadata

    assert tuple(event.kind for event in left.events) == tuple(
        event.kind for event in right.events
    )
    assert tuple(event.time for event in left.events) == tuple(
        event.time for event in right.events
    )
    assert tuple(event.deltas for event in left.events) == tuple(
        event.deltas for event in right.events
    )

    assert left.signature.source_pattern_id == right.signature.source_pattern_id
    assert left.signature.compact_vector() == right.signature.compact_vector()

def test_complete_run_returns_result(result) -> None:
    assert isinstance(result, DeepRecursiveMemoryResult)


def test_result_contains_metrics(result) -> None:
    assert isinstance(result.metrics, DeepRecursiveMemoryMetrics)


def test_result_uses_memory(result) -> None:
    assert result.preload.used_memory is True


def test_result_tree_is_policy_free(result) -> None:
    assert result.metadata["tree_policy_free"] is True


def test_raw_record_count_regression(result) -> None:
    assert result.metrics.raw_record_count == 256


def test_base_cluster_count_regression(result) -> None:
    assert result.metrics.base_cluster_count == 3


def test_recursive_node_count_regression(result) -> None:
    assert result.metrics.recursive_node_count == 6


def test_recursive_tree_depth_regression(result) -> None:
    assert result.metrics.recursive_tree_depth == 3


def test_root_count_regression(result) -> None:
    assert result.metrics.root_count == 1


def test_active_ancestry_length_regression(result) -> None:
    assert result.metrics.active_ancestry_length == 3


def test_active_contribution_count_regression(result) -> None:
    assert result.metrics.active_contribution_count == 3


def test_raw_to_cluster_ratio_regression(result) -> None:
    assert result.metrics.raw_to_cluster_ratio == pytest.approx(85.33333333333333)


def test_raw_to_recursive_node_ratio_regression(result) -> None:
    assert result.metrics.raw_to_recursive_node_ratio == pytest.approx(42.666666666666664)


def test_raw_to_active_context_ratio_regression(result) -> None:
    assert result.metrics.raw_to_active_context_ratio == pytest.approx(85.33333333333333)


def test_raw_memory_is_much_larger_than_active_context(result) -> None:
    assert result.metrics.raw_record_count > 80 * result.metrics.active_contribution_count


def test_active_context_is_less_than_two_percent_of_raw_memory(result) -> None:
    assert (
        result.metrics.active_contribution_count / result.metrics.raw_record_count
        < 0.02
    )


def test_retrieval_tree_levels_are_one_two_three(result) -> None:
    assert retrieval_tree_levels(result) == (1, 2, 3)


def test_retrieval_preload_levels_are_one_two(result) -> None:
    assert hierarchy_levels_used(result.preload) == (1, 2)


def test_physical_ancestry_has_three_nodes(result) -> None:
    assert len(physical_ancestry_node_ids(result)) == 3


def test_physical_ancestry_matches_retrieval_nodes(result) -> None:
    assert physical_ancestry_node_ids(result) == retrieval_tree_node_ids(result)


def test_physical_ancestry_matches_tree_ancestry(result) -> None:
    chain = ancestry(result.tree, result.retrieval.leaf_node_id)
    assert tuple(node.node_id for node in chain) == physical_ancestry_node_ids(result)


def test_leaf_cluster_ancestry_has_three_levels(result) -> None:
    assert len(leaf_cluster_ancestry(result)) == 3


def test_leaf_cluster_coverage_expands_or_stays_equal_up_tree(result) -> None:
    lineage = leaf_cluster_ancestry(result)
    sizes = tuple(len(leaf_ids) for _node_id, leaf_ids in lineage)
    assert all(left <= right for left, right in zip(sizes, sizes[1:]))


def test_retrieval_is_policy_free(result) -> None:
    assert retrieval_is_policy_free(result.retrieval) is True


def test_preload_is_policy_free(result) -> None:
    assert preload_is_policy_free(result.preload) is True


def test_result_declares_policy_override_disabled(result) -> None:
    assert result.metadata["policy_override_enabled"] is False


def test_result_declares_memory_does_not_select_action(result) -> None:
    assert result.metadata["action_selected_by_memory"] is False


def test_result_declares_no_external_label_usage(result) -> None:
    assert result.metadata["external_expected_label_used"] is False


def test_result_declares_label_free_cluster_assignment(result) -> None:
    assert result.metadata["cluster_assignment_uses_external_pattern_label"] is False


def test_recomputed_metrics_equal_result_metrics(result) -> None:
    recalculated = compute_deep_recursive_metrics(
        result.memory,
        result.tree,
        result.retrieval,
    )
    assert recalculated == result.metrics


def test_result_metadata_is_read_only(result) -> None:
    assert isinstance(result.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_result_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.query_index = 1  # type: ignore[misc]


def test_metrics_are_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.metrics.raw_record_count = 1  # type: ignore[misc]


def _signature(result: DeepRecursiveMemoryResult):
    return (
        result.scenario_id,
        result.memory.record_count,
        result.memory.cluster_count,
        result.tree.node_count,
        result.tree.root_ids,
        result.tree.max_level,
        result.query_index,
        result.retrieval.nearest_cluster_id,
        result.retrieval.nearest_distance,
        result.retrieval.leaf_node_id,
        retrieval_tree_levels(result),
        retrieval_tree_node_ids(result),
        physical_ancestry_node_ids(result),
        tuple(sorted(result.preload.applied_bias_by_variable.items())),
        result.preload.used_memory,
        result.metrics,
        tuple(sorted(result.metadata.items())),
    )


def test_complete_04g_benchmark_is_deterministic() -> None:
    left = run_deep_recursive_memory_benchmark()
    right = run_deep_recursive_memory_benchmark()
    assert _signature(left) == _signature(right)


def test_history_build_is_deterministic() -> None:
    left = build_deep_recursive_history(event_count=64)
    right = build_deep_recursive_history(event_count=64)

    assert left.record_count == right.record_count
    assert (
        tuple(cluster.member_record_ids for cluster in left.clusters)
        == tuple(cluster.member_record_ids for cluster in right.clusters)
    )


def test_recursive_tree_build_is_deterministic(memory) -> None:
    assert build_deep_recursive_tree(memory) == build_deep_recursive_tree(memory)


def test_recursive_retrieval_is_deterministic(memory, tree) -> None:
    left_retrieval, left_preload = run_deep_recursive_retrieval(memory, tree)
    right_retrieval, right_preload = run_deep_recursive_retrieval(memory, tree)

    assert left_retrieval.nearest_cluster_id == right_retrieval.nearest_cluster_id
    assert left_retrieval.nearest_distance == right_retrieval.nearest_distance
    assert (
        tuple(level.tree_node_id for level in left_retrieval.levels)
        == tuple(level.tree_node_id for level in right_retrieval.levels)
    )
    assert left_preload == right_preload
