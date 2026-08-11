"""
Tests for roif.hierarchical_memory_retrieval

Updated for explicit RecursiveMemoryTree retrieval.

Pipeline:

    HierarchicalMemory
        -> RecursiveMemoryTree
        -> nearest leaf cluster
        -> real ancestry: local -> parent -> grandparent
        -> MemoryScar[]
        -> MemoryPreloadContribution[]
        -> PredictivePreload

Core invariants:

    Retrieval != Action Selection
    Retrieval != Policy Override
    Recursive Tree != Ground Truth
    Cluster Match != Evaluator Label
    Retrieval != Predictive State Mutation
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from roif.hierarchical_memory import HierarchicalMemory
from roif.hierarchical_memory_retrieval import (
    HierarchicalMemoryRetrieval,
    HierarchicalMemoryRetrievalError,
    RetrievalConfig,
    RetrievalLevel,
    retrieve_hierarchical_memory,
    retrieval_is_policy_free,
    retrieval_levels,
    retrieval_tree_levels,
    retrieval_tree_node_ids,
)
from roif.predictive_preload import (
    MemoryPreloadContribution,
    apply_predictive_preload,
    hierarchy_levels_used,
    preload_is_policy_free,
)
from roif.recursive_hierarchical_memory import (
    RecursiveMemoryTree,
    ancestry,
    build_recursive_memory_tree,
    leaf_node_for_cluster,
)

from validation.sailing.sailing_yacht_crew_long_memory_compression import (
    build_long_memory_event,
    event_to_controller_history_record,
    run_long_memory_compression_benchmark,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def memory_100() -> HierarchicalMemory:
    return run_long_memory_compression_benchmark(
        event_count=100,
        checkpoint_interval=20,
    ).memory


@pytest.fixture(scope="module")
def tree_100(
    memory_100: HierarchicalMemory,
) -> RecursiveMemoryTree:
    return build_recursive_memory_tree(
        memory_100
    )


@pytest.fixture(scope="module")
def query_a():
    event = build_long_memory_event(101)

    return event_to_controller_history_record(
        event=event,
        episode_index=101,
        sequence_id="recursive_retrieval_query_A",
    )


@pytest.fixture(scope="module")
def query_b():
    event = build_long_memory_event(104)

    return event_to_controller_history_record(
        event=event,
        episode_index=104,
        sequence_id="recursive_retrieval_query_B",
    )


@pytest.fixture(scope="module")
def retrieval_a(
    memory_100,
    tree_100,
    query_a,
) -> HierarchicalMemoryRetrieval:
    return retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
    )


# =============================================================================
# Config
# =============================================================================


def test_default_config_is_valid() -> None:
    config = RetrievalConfig()

    assert config.max_hierarchy_levels == 3
    assert config.local_weight == pytest.approx(1.0)
    assert config.parent_weight == pytest.approx(1.0)
    assert config.grandparent_weight == pytest.approx(1.0)


def test_config_rejects_zero_max_hierarchy_levels() -> None:
    with pytest.raises(HierarchicalMemoryRetrievalError):
        RetrievalConfig(
            max_hierarchy_levels=0
        )


def test_config_rejects_minimum_members_below_two() -> None:
    with pytest.raises(HierarchicalMemoryRetrievalError):
        RetrievalConfig(
            minimum_cluster_members_for_scar=1
        )


def test_config_rejects_negative_local_weight() -> None:
    with pytest.raises(HierarchicalMemoryRetrievalError):
        RetrievalConfig(
            local_weight=-1.0
        )


def test_config_rejects_negative_parent_weight() -> None:
    with pytest.raises(HierarchicalMemoryRetrievalError):
        RetrievalConfig(
            parent_weight=-1.0
        )


def test_config_rejects_negative_grandparent_weight() -> None:
    with pytest.raises(HierarchicalMemoryRetrievalError):
        RetrievalConfig(
            grandparent_weight=-1.0
        )


def test_config_is_frozen() -> None:
    config = RetrievalConfig()

    with pytest.raises(FrozenInstanceError):
        config.local_weight = 2.0  # type: ignore[misc]


# =============================================================================
# Empty memory / tree
# =============================================================================


def test_empty_memory_and_tree_return_empty_retrieval(
    query_a,
) -> None:
    memory = HierarchicalMemory()
    tree = build_recursive_memory_tree(
        memory
    )

    retrieval = retrieve_hierarchical_memory(
        memory,
        tree,
        query_a,
    )

    assert retrieval.nearest_cluster_id is None
    assert retrieval.nearest_distance is None
    assert retrieval.leaf_node_id is None
    assert retrieval.levels == ()
    assert retrieval.contributions == ()


def test_empty_retrieval_marks_recursive_tree_used(
    query_a,
) -> None:
    memory = HierarchicalMemory()
    tree = build_recursive_memory_tree(
        memory
    )

    retrieval = retrieve_hierarchical_memory(
        memory,
        tree,
        query_a,
    )

    assert retrieval.metadata[
        "recursive_tree_used"
    ] is True


def test_empty_retrieval_is_policy_free(
    query_a,
) -> None:
    memory = HierarchicalMemory()
    tree = build_recursive_memory_tree(
        memory
    )

    retrieval = retrieve_hierarchical_memory(
        memory,
        tree,
        query_a,
    )

    assert retrieval_is_policy_free(
        retrieval
    ) is True


# =============================================================================
# Real recursive retrieval
# =============================================================================


def test_real_retrieval_returns_result(
    retrieval_a,
) -> None:
    assert isinstance(
        retrieval_a,
        HierarchicalMemoryRetrieval,
    )


def test_real_retrieval_finds_nearest_cluster(
    retrieval_a,
) -> None:
    assert retrieval_a.nearest_cluster_id is not None


def test_real_retrieval_resolves_leaf_node(
    retrieval_a,
) -> None:
    assert retrieval_a.leaf_node_id is not None


def test_leaf_node_matches_nearest_cluster(
    retrieval_a,
    tree_100,
) -> None:
    expected = leaf_node_for_cluster(
        tree_100,
        retrieval_a.nearest_cluster_id,
    )

    assert retrieval_a.leaf_node_id == expected.node_id


def test_real_retrieval_distance_is_nonnegative(
    retrieval_a,
) -> None:
    assert retrieval_a.nearest_distance is not None
    assert retrieval_a.nearest_distance >= 0.0


def test_real_retrieval_uses_recursive_tree(
    retrieval_a,
) -> None:
    assert retrieval_a.metadata[
        "recursive_tree_used"
    ] is True


# =============================================================================
# Real ancestry contract
# =============================================================================


def test_retrieval_levels_follow_available_tree_depth(
    retrieval_a,
) -> None:
    # 04C currently has 2 leaf clusters -> level-1 leaves + level-2 root.
    assert retrieval_levels(
        retrieval_a
    ) == (
        0,
        1,
    )


def test_retrieval_tree_levels_are_one_two(
    retrieval_a,
) -> None:
    assert retrieval_tree_levels(
        retrieval_a
    ) == (
        1,
        2,
    )


def test_retrieval_tree_node_ids_match_real_ancestry(
    retrieval_a,
    tree_100,
) -> None:
    chain = ancestry(
        tree_100,
        retrieval_a.leaf_node_id,
    )

    assert retrieval_tree_node_ids(
        retrieval_a
    ) == tuple(
        node.node_id
        for node
        in chain[:3]
    )


def test_local_retrieval_level_points_to_leaf_node(
    retrieval_a,
) -> None:
    local = retrieval_a.levels[
        0
    ]

    assert local.level == 0
    assert local.tree_level == 1
    assert local.tree_node_id == retrieval_a.leaf_node_id


def test_parent_retrieval_level_points_to_real_parent(
    retrieval_a,
    tree_100,
) -> None:
    local_chain = ancestry(
        tree_100,
        retrieval_a.leaf_node_id,
    )

    parent = retrieval_a.levels[
        1
    ]

    assert parent.level == 1
    assert parent.tree_node_id == local_chain[
        1
    ].node_id
    assert parent.tree_level == local_chain[
        1
    ].level


def test_parent_level_contains_all_root_leaf_clusters(
    retrieval_a,
    tree_100,
) -> None:
    root = ancestry(
        tree_100,
        retrieval_a.leaf_node_id,
    )[
        1
    ]

    parent = retrieval_a.levels[
        1
    ]

    assert set(
        parent.leaf_cluster_ids
    ) == set(
        root.leaf_cluster_ids
    )


def test_real_retrieval_builds_two_contributions_for_current_tree(
    retrieval_a,
) -> None:
    assert len(
        retrieval_a.contributions
    ) == 2


def test_all_retrieval_contributions_are_preload_contributions(
    retrieval_a,
) -> None:
    assert all(
        isinstance(
            contribution,
            MemoryPreloadContribution,
        )
        for contribution
        in retrieval_a.contributions
    )


def test_contribution_hierarchy_levels_match_retrieval_levels(
    retrieval_a,
) -> None:
    assert tuple(
        contribution.hierarchy_level
        for contribution
        in retrieval_a.contributions
    ) == retrieval_levels(
        retrieval_a
    )


def test_contribution_source_ids_are_real_tree_node_ids(
    retrieval_a,
) -> None:
    assert tuple(
        contribution.source_cluster_id
        for contribution
        in retrieval_a.contributions
    ) == retrieval_tree_node_ids(
        retrieval_a
    )


# =============================================================================
# RetrievalLevel semantics
# =============================================================================


def test_all_levels_are_retrieval_levels(
    retrieval_a,
) -> None:
    assert all(
        isinstance(
            level,
            RetrievalLevel,
        )
        for level
        in retrieval_a.levels
    )


def test_each_level_scar_is_retrieval_generated(
    retrieval_a,
) -> None:
    assert all(
        level.scar.metadata[
            "retrieval_generated"
        ] is True
        for level
        in retrieval_a.levels
    )


def test_each_level_scar_preserves_tree_node_id(
    retrieval_a,
) -> None:
    assert all(
        level.scar.metadata[
            "recursive_tree_node_id"
        ] == level.tree_node_id
        for level
        in retrieval_a.levels
    )


def test_each_level_scar_preserves_tree_level(
    retrieval_a,
) -> None:
    assert all(
        level.scar.metadata[
            "recursive_tree_level"
        ] == level.tree_level
        for level
        in retrieval_a.levels
    )


# =============================================================================
# No label leakage
# =============================================================================


def test_retrieval_metadata_declares_no_external_label_usage(
    retrieval_a,
) -> None:
    assert retrieval_a.metadata[
        "external_expected_label_used"
    ] is False


def test_each_level_declares_no_external_label_usage(
    retrieval_a,
) -> None:
    assert all(
        level.metadata[
            "external_expected_label_used"
        ] is False
        for level
        in retrieval_a.levels
    )


def test_each_contribution_declares_no_external_label_usage(
    retrieval_a,
) -> None:
    assert all(
        contribution.metadata[
            "external_expected_label_used"
        ] is False
        for contribution
        in retrieval_a.contributions
    )


# =============================================================================
# Policy-free boundary
# =============================================================================


def test_retrieval_does_not_select_action(
    retrieval_a,
) -> None:
    assert retrieval_a.metadata[
        "action_selected"
    ] is False


def test_retrieval_does_not_modify_policy(
    retrieval_a,
) -> None:
    assert retrieval_a.metadata[
        "policy_modified"
    ] is False


def test_retrieval_does_not_mutate_predictive_state(
    retrieval_a,
) -> None:
    assert retrieval_a.metadata[
        "predictive_state_mutated"
    ] is False


def test_retrieval_is_policy_free_returns_true(
    retrieval_a,
) -> None:
    assert retrieval_is_policy_free(
        retrieval_a
    ) is True


def test_policy_free_detects_forbidden_flag(
    retrieval_a,
) -> None:
    altered = replace(
        retrieval_a,
        metadata={
            **dict(
                retrieval_a.metadata
            ),
            "policy_modified": True,
        },
    )

    assert retrieval_is_policy_free(
        altered
    ) is False


# =============================================================================
# Direct retrieval -> preload integration
# =============================================================================


def test_retrieval_contributions_can_feed_predictive_preload(
    retrieval_a,
    query_a,
) -> None:
    preload = apply_predictive_preload(
        query_a.source_trace.initial_state,
        retrieval_a.contributions,
    )

    assert preload.used_memory is True


def test_retrieval_preload_uses_real_available_nested_levels(
    retrieval_a,
    query_a,
) -> None:
    preload = apply_predictive_preload(
        query_a.source_trace.initial_state,
        retrieval_a.contributions,
    )

    assert hierarchy_levels_used(
        preload
    ) == (
        0,
        1,
    )


def test_retrieval_preload_remains_policy_free(
    retrieval_a,
    query_a,
) -> None:
    preload = apply_predictive_preload(
        query_a.source_trace.initial_state,
        retrieval_a.contributions,
    )

    assert preload_is_policy_free(
        preload
    ) is True


def test_retrieval_preload_changes_known_variable(
    retrieval_a,
    query_a,
) -> None:
    preload = apply_predictive_preload(
        query_a.source_trace.initial_state,
        retrieval_a.contributions,
    )

    assert (
        preload.preloaded_state.values[
            "course_error"
        ]
        != pytest.approx(
            preload.original_state.values[
                "course_error"
            ]
        )
    )


def test_retrieval_preload_reduces_uncertainty(
    retrieval_a,
    query_a,
) -> None:
    preload = apply_predictive_preload(
        query_a.source_trace.initial_state,
        retrieval_a.contributions,
    )

    assert (
        preload.uncertainty_after
        <
        preload.uncertainty_before
    )


# =============================================================================
# Max hierarchy depth switch
# =============================================================================


def test_local_only_retrieval_uses_only_level_zero(
    memory_100,
    tree_100,
    query_a,
) -> None:
    retrieval = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
        config=RetrievalConfig(
            max_hierarchy_levels=1,
        ),
    )

    assert retrieval_levels(
        retrieval
    ) == (
        0,
    )


def test_two_level_retrieval_uses_local_and_parent(
    memory_100,
    tree_100,
    query_a,
) -> None:
    retrieval = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
        config=RetrievalConfig(
            max_hierarchy_levels=2,
        ),
    )

    assert retrieval_levels(
        retrieval
    ) == (
        0,
        1,
    )


def test_requesting_three_levels_does_not_invent_missing_grandparent(
    memory_100,
    tree_100,
    query_a,
) -> None:
    retrieval = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
        config=RetrievalConfig(
            max_hierarchy_levels=3,
        ),
    )

    assert retrieval_tree_levels(
        retrieval
    ) == (
        1,
        2,
    )


# =============================================================================
# Tree consistency guard
# =============================================================================


def test_retrieval_rejects_tree_without_nearest_cluster(
    memory_100,
    query_a,
) -> None:
    empty_tree = build_recursive_memory_tree(
        HierarchicalMemory()
    )

    with pytest.raises(
        HierarchicalMemoryRetrievalError
    ):
        retrieve_hierarchical_memory(
            memory_100,
            empty_tree,
            query_a,
        )


# =============================================================================
# Different queries
# =============================================================================


def test_a_and_b_queries_can_retrieve_different_nearest_clusters(
    memory_100,
    tree_100,
    query_a,
    query_b,
) -> None:
    a = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
    )

    b = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_b,
    )

    assert (
        a.nearest_cluster_id
        != b.nearest_cluster_id
        or a.nearest_distance
        != pytest.approx(
            b.nearest_distance
        )
    )


def test_b_query_retrieval_is_policy_free(
    memory_100,
    tree_100,
    query_b,
) -> None:
    retrieval = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_b,
    )

    assert retrieval_is_policy_free(
        retrieval
    ) is True


# =============================================================================
# Immutability
# =============================================================================


def test_retrieval_levels_are_tuple(
    retrieval_a,
) -> None:
    assert isinstance(
        retrieval_a.levels,
        tuple,
    )


def test_retrieval_contributions_are_tuple(
    retrieval_a,
) -> None:
    assert isinstance(
        retrieval_a.contributions,
        tuple,
    )


def test_retrieval_metadata_is_read_only(
    retrieval_a,
) -> None:
    assert isinstance(
        retrieval_a.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        retrieval_a.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_retrieval_is_frozen(
    retrieval_a,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        retrieval_a.query_record_id = "changed"  # type: ignore[misc]


def test_level_leaf_cluster_ids_are_tuple(
    retrieval_a,
) -> None:
    assert isinstance(
        retrieval_a.levels[
            0
        ].leaf_cluster_ids,
        tuple,
    )


def test_level_metadata_is_read_only(
    retrieval_a,
) -> None:
    level = retrieval_a.levels[
        0
    ]

    assert isinstance(
        level.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        level.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_level_is_frozen(
    retrieval_a,
) -> None:
    level = retrieval_a.levels[
        0
    ]

    with pytest.raises(
        FrozenInstanceError
    ):
        level.level = 99  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def _signature(
    retrieval: HierarchicalMemoryRetrieval,
):
    return (
        retrieval.query_record_id,
        retrieval.nearest_cluster_id,
        retrieval.nearest_distance,
        retrieval.leaf_node_id,
        tuple(
            (
                level.level,
                level.tree_node_id,
                level.tree_level,
                level.leaf_cluster_ids,
                level.scar.scar_id,
                tuple(
                    sorted(
                        level.scar.bias_by_variable.items()
                    )
                ),
                level.scar.confidence,
                level.scar.consistency,
            )
            for level
            in retrieval.levels
        ),
        tuple(
            (
                contribution.scar.scar_id,
                contribution.hierarchy_level,
                contribution.explicit_weight,
                contribution.source_cluster_id,
            )
            for contribution
            in retrieval.contributions
        ),
        tuple(
            sorted(
                retrieval.metadata.items()
            )
        ),
    )


def test_real_recursive_retrieval_is_deterministic(
    memory_100,
    tree_100,
    query_a,
) -> None:
    left = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
    )

    right = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
    )

    assert _signature(
        left
    ) == _signature(
        right
    )


def test_local_only_recursive_retrieval_is_deterministic(
    memory_100,
    tree_100,
    query_a,
) -> None:
    config = RetrievalConfig(
        max_hierarchy_levels=1,
    )

    left = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
        config=config,
    )

    right = retrieve_hierarchical_memory(
        memory_100,
        tree_100,
        query_a,
        config=config,
    )

    assert _signature(
        left
    ) == _signature(
        right
    )
