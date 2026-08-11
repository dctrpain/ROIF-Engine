"""
Tests for roif.recursive_hierarchical_memory

The suite fixes the first explicit recursive super-cluster tree contract.

Base structure:

    HierarchicalMemory.MemoryCluster
        -> RecursiveMemoryNode(level 1)
        -> RecursiveMemoryNode(level 2)
        -> RecursiveMemoryNode(level 3)
        -> ...

Core invariants:

    Recursive Grouping != Learning
    Super-Cluster != Ground Truth
    Tree Construction != PredictivePreload
    Parent Context != Policy Override

The tests use real 04C HierarchicalMemory and verify:
- level-1 leaf wrappers;
- real parent_id / child_ids links;
- root construction;
- ancestry traversal;
- preservation of all source clusters;
- prototype/radius aggregation;
- no mutation of source memory;
- immutability;
- policy-free metadata;
- deterministic tree construction.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.recursive_hierarchical_memory import (
    InvalidRecursiveMemoryNodeError,
    RecursiveMemoryConfig,
    RecursiveMemoryError,
    RecursiveMemoryNode,
    RecursiveMemoryTree,
    ancestry,
    build_recursive_memory_tree,
    children_of,
    leaf_node_for_cluster,
    parent_of,
    recursive_node_by_id,
    tree_is_policy_free,
)

from validation.sailing.sailing_yacht_crew_long_memory_compression import (
    run_long_memory_compression_benchmark,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def memory_100():
    return run_long_memory_compression_benchmark(
        event_count=100,
        checkpoint_interval=20,
    ).memory


@pytest.fixture(scope="module")
def tree(
    memory_100,
) -> RecursiveMemoryTree:
    return build_recursive_memory_tree(
        memory_100
    )


# =============================================================================
# Config
# =============================================================================


def test_default_config_is_valid() -> None:
    config = RecursiveMemoryConfig()

    assert config.merge_radius == pytest.approx(
        0.25
    )
    assert config.max_children == 4
    assert config.max_depth == 8


def test_config_rejects_negative_merge_radius() -> None:
    with pytest.raises(
        RecursiveMemoryError
    ):
        RecursiveMemoryConfig(
            merge_radius=-0.1
        )


def test_config_rejects_max_children_below_two() -> None:
    with pytest.raises(
        RecursiveMemoryError
    ):
        RecursiveMemoryConfig(
            max_children=1
        )


def test_config_rejects_max_depth_below_one() -> None:
    with pytest.raises(
        RecursiveMemoryError
    ):
        RecursiveMemoryConfig(
            max_depth=0
        )


def test_config_is_frozen() -> None:
    config = RecursiveMemoryConfig()

    with pytest.raises(
        FrozenInstanceError
    ):
        config.merge_radius = 1.0  # type: ignore[misc]


# =============================================================================
# Empty tree
# =============================================================================


def test_empty_memory_builds_empty_tree() -> None:
    from roif.hierarchical_memory import HierarchicalMemory

    tree = build_recursive_memory_tree(
        HierarchicalMemory()
    )

    assert tree.node_count == 0
    assert tree.root_count == 0
    assert tree.max_level == 0


def test_empty_tree_is_policy_free() -> None:
    from roif.hierarchical_memory import HierarchicalMemory

    tree = build_recursive_memory_tree(
        HierarchicalMemory()
    )

    assert tree_is_policy_free(
        tree
    ) is True


# =============================================================================
# Real 04C tree shape
# =============================================================================


def test_real_tree_returns_tree(
    tree,
) -> None:
    assert isinstance(
        tree,
        RecursiveMemoryTree,
    )


def test_real_tree_preserves_two_leaf_clusters(
    tree,
    memory_100,
) -> None:
    assert set(
        tree.leaf_cluster_ids
    ) == {
        cluster.cluster_id
        for cluster
        in memory_100.clusters
    }


def test_real_tree_has_two_level_one_leaf_wrappers(
    tree,
) -> None:
    leaves = tuple(
        node
        for node in tree.nodes
        if node.level == 1
    )

    assert len(
        leaves
    ) == 2


def test_level_one_nodes_are_leaf_wrappers(
    tree,
) -> None:
    leaves = tuple(
        node
        for node in tree.nodes
        if node.level == 1
    )

    assert all(
        node.is_leaf_wrapper
        for node
        in leaves
    )


def test_real_tree_has_one_root(
    tree,
) -> None:
    assert tree.root_count == 1


def test_real_tree_reaches_level_two(
    tree,
) -> None:
    assert tree.max_level == 2


def test_root_is_level_two(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    assert root.level == 2


def test_root_contains_both_leaf_clusters(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    assert set(
        root.leaf_cluster_ids
    ) == set(
        tree.leaf_cluster_ids
    )


def test_root_has_two_children(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    assert len(
        root.child_ids
    ) == 2


# =============================================================================
# Parent / child links
# =============================================================================


def test_root_children_exist(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    children = children_of(
        tree,
        root.node_id,
    )

    assert len(
        children
    ) == 2


def test_each_leaf_has_root_parent(
    tree,
) -> None:
    root_id = tree.root_ids[
        0
    ]

    leaves = tuple(
        node
        for node in tree.nodes
        if node.level == 1
    )

    assert all(
        node.parent_id
        == root_id
        for node
        in leaves
    )


def test_parent_of_leaf_returns_root(
    tree,
) -> None:
    leaf = next(
        node
        for node in tree.nodes
        if node.level == 1
    )

    parent = parent_of(
        tree,
        leaf.node_id,
    )

    assert parent is not None
    assert parent.node_id == tree.root_ids[
        0
    ]


def test_parent_of_root_is_none(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    assert parent_of(
        tree,
        root.node_id,
    ) is None


# =============================================================================
# Leaf lookup / ancestry
# =============================================================================


def test_leaf_node_for_each_source_cluster(
    tree,
    memory_100,
) -> None:
    for cluster in memory_100.clusters:
        leaf = leaf_node_for_cluster(
            tree,
            cluster.cluster_id,
        )

        assert leaf.leaf_cluster_ids == (
            cluster.cluster_id,
        )


def test_leaf_lookup_rejects_unknown_cluster(
    tree,
) -> None:
    with pytest.raises(
        KeyError
    ):
        leaf_node_for_cluster(
            tree,
            "unknown_cluster",
        )


def test_leaf_ancestry_contains_leaf_then_root(
    tree,
) -> None:
    leaf = next(
        node
        for node in tree.nodes
        if node.level == 1
    )

    chain = ancestry(
        tree,
        leaf.node_id,
    )

    assert tuple(
        node.level
        for node in chain
    ) == (
        1,
        2,
    )


def test_root_ancestry_contains_only_root(
    tree,
) -> None:
    root_id = tree.root_ids[
        0
    ]

    chain = ancestry(
        tree,
        root_id,
    )

    assert len(
        chain
    ) == 1

    assert chain[
        0
    ].node_id == root_id


# =============================================================================
# Prototype / radius aggregation
# =============================================================================


def test_root_prototype_vector_has_same_length_as_leaf(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    leaf = next(
        node
        for node in tree.nodes
        if node.level == 1
    )

    assert len(
        root.prototype_vector
    ) == len(
        leaf.prototype_vector
    )


def test_root_radius_is_positive(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    assert root.radius > 0.0


def test_root_member_count_equals_total_cluster_members(
    tree,
    memory_100,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    assert root.member_count == sum(
        cluster.size
        for cluster
        in memory_100.clusters
    )


def test_root_has_representative_cluster(
    tree,
) -> None:
    root = recursive_node_by_id(
        tree,
        tree.root_ids[
            0
        ],
    )

    assert (
        root.representative_cluster_id
        in tree.leaf_cluster_ids
    )


# =============================================================================
# Source memory non-mutation
# =============================================================================


def test_tree_construction_does_not_change_source_record_count(
    memory_100,
) -> None:
    before = memory_100.record_count

    build_recursive_memory_tree(
        memory_100
    )

    assert memory_100.record_count == before


def test_tree_construction_does_not_change_source_cluster_count(
    memory_100,
) -> None:
    before = memory_100.cluster_count

    build_recursive_memory_tree(
        memory_100
    )

    assert memory_100.cluster_count == before


def test_tree_construction_does_not_change_source_cluster_memberships(
    memory_100,
) -> None:
    before = tuple(
        cluster.member_record_ids
        for cluster
        in memory_100.clusters
    )

    build_recursive_memory_tree(
        memory_100
    )

    after = tuple(
        cluster.member_record_ids
        for cluster
        in memory_100.clusters
    )

    assert before == after


# =============================================================================
# Node validation
# =============================================================================


def test_node_rejects_empty_id() -> None:
    with pytest.raises(
        InvalidRecursiveMemoryNodeError
    ):
        RecursiveMemoryNode(
            node_id="",
            level=1,
            prototype_vector=(0.0,),
            radius=0.0,
            leaf_cluster_ids=(
                "cluster_1",
            ),
        )


def test_node_rejects_level_zero() -> None:
    with pytest.raises(
        InvalidRecursiveMemoryNodeError
    ):
        RecursiveMemoryNode(
            node_id="x",
            level=0,
            prototype_vector=(0.0,),
            radius=0.0,
            leaf_cluster_ids=(
                "cluster_1",
            ),
        )


def test_node_rejects_negative_radius() -> None:
    with pytest.raises(
        InvalidRecursiveMemoryNodeError
    ):
        RecursiveMemoryNode(
            node_id="x",
            level=1,
            prototype_vector=(0.0,),
            radius=-1.0,
            leaf_cluster_ids=(
                "cluster_1",
            ),
        )


def test_node_rejects_empty_leaf_cluster_ids() -> None:
    with pytest.raises(
        InvalidRecursiveMemoryNodeError
    ):
        RecursiveMemoryNode(
            node_id="x",
            level=1,
            prototype_vector=(0.0,),
            radius=0.0,
            leaf_cluster_ids=(),
        )


def test_node_rejects_duplicate_leaf_cluster_ids() -> None:
    with pytest.raises(
        InvalidRecursiveMemoryNodeError
    ):
        RecursiveMemoryNode(
            node_id="x",
            level=1,
            prototype_vector=(0.0,),
            radius=0.0,
            leaf_cluster_ids=(
                "cluster_1",
                "cluster_1",
            ),
        )


def test_node_rejects_duplicate_child_ids() -> None:
    with pytest.raises(
        InvalidRecursiveMemoryNodeError
    ):
        RecursiveMemoryNode(
            node_id="x",
            level=2,
            prototype_vector=(0.0,),
            radius=0.0,
            leaf_cluster_ids=(
                "cluster_1",
            ),
            child_ids=(
                "child",
                "child",
            ),
        )


# =============================================================================
# Lookup guards
# =============================================================================


def test_recursive_node_by_id_rejects_unknown(
    tree,
) -> None:
    with pytest.raises(
        KeyError
    ):
        recursive_node_by_id(
            tree,
            "unknown",
        )


def test_children_of_leaf_is_empty(
    tree,
) -> None:
    leaf = next(
        node
        for node in tree.nodes
        if node.level == 1
    )

    assert children_of(
        tree,
        leaf.node_id,
    ) == ()


# =============================================================================
# Policy-free boundary
# =============================================================================


def test_tree_is_policy_free(
    tree,
) -> None:
    assert tree_is_policy_free(
        tree
    ) is True


def test_tree_declares_no_predictive_preload(
    tree,
) -> None:
    assert (
        tree.metadata[
            "predictive_preload_applied"
        ]
        is False
    )


def test_tree_declares_no_policy_modification(
    tree,
) -> None:
    assert (
        tree.metadata[
            "policy_modified"
        ]
        is False
    )


def test_all_nodes_declare_no_predictive_preload(
    tree,
) -> None:
    assert all(
        node.metadata[
            "predictive_preload_applied"
        ]
        is False
        for node
        in tree.nodes
    )


def test_all_nodes_declare_no_policy_modification(
    tree,
) -> None:
    assert all(
        node.metadata[
            "policy_modified"
        ]
        is False
        for node
        in tree.nodes
    )


def test_tree_declares_no_external_expected_label_usage(
    tree,
) -> None:
    assert (
        tree.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Immutability
# =============================================================================


def test_tree_nodes_are_tuple(
    tree,
) -> None:
    assert isinstance(
        tree.nodes,
        tuple,
    )


def test_tree_root_ids_are_tuple(
    tree,
) -> None:
    assert isinstance(
        tree.root_ids,
        tuple,
    )


def test_tree_leaf_cluster_ids_are_tuple(
    tree,
) -> None:
    assert isinstance(
        tree.leaf_cluster_ids,
        tuple,
    )


def test_tree_metadata_is_read_only(
    tree,
) -> None:
    assert isinstance(
        tree.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        tree.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_tree_is_frozen(
    tree,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        tree.max_level = 99  # type: ignore[misc]


def test_node_metadata_is_read_only(
    tree,
) -> None:
    node = tree.nodes[
        0
    ]

    assert isinstance(
        node.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        node.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_node_is_frozen(
    tree,
) -> None:
    node = tree.nodes[
        0
    ]

    with pytest.raises(
        FrozenInstanceError
    ):
        node.parent_id = "changed"  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def _signature(
    tree: RecursiveMemoryTree,
):
    return (
        tree.root_ids,
        tree.leaf_cluster_ids,
        tree.max_level,
        tuple(
            (
                node.node_id,
                node.level,
                node.prototype_vector,
                node.radius,
                node.leaf_cluster_ids,
                node.child_ids,
                node.parent_id,
                node.member_count,
                node.representative_cluster_id,
                tuple(
                    sorted(
                        node.metadata.items()
                    )
                ),
            )
            for node
            in tree.nodes
        ),
        tuple(
            sorted(
                tree.metadata.items()
            )
        ),
    )


def test_recursive_tree_is_deterministic(
    memory_100,
) -> None:
    left = build_recursive_memory_tree(
        memory_100
    )

    right = build_recursive_memory_tree(
        memory_100
    )

    assert _signature(
        left
    ) == _signature(
        right
    )


def test_custom_config_tree_is_deterministic(
    memory_100,
) -> None:
    config = RecursiveMemoryConfig(
        merge_radius=0.05,
        max_children=2,
        max_depth=4,
    )

    left = build_recursive_memory_tree(
        memory_100,
        config=config,
    )

    right = build_recursive_memory_tree(
        memory_100,
        config=config,
    )

    assert _signature(
        left
    ) == _signature(
        right
    )
