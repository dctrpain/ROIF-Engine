"""
ROIF Recursive Hierarchical Memory
=================================

Purpose
-------
Build explicit recursive super-cluster nodes over existing HierarchicalMemory.

Existing first-level memory remains unchanged:

    ControllerHistoryRecord
        -> MemoryCluster (level 1)

This module adds:

    MemoryCluster
        -> RecursiveMemoryNode(level 1)
        -> RecursiveMemoryNode(level 2)
        -> RecursiveMemoryNode(level 3)
        -> ...

Architectural boundaries
------------------------

    Recursive Grouping != Learning
    Super-Cluster != Ground Truth
    Tree Construction != PredictivePreload
    Parent Context != Policy Override

The module is intentionally additive:
- it does not mutate HierarchicalMemory;
- it does not replace MemoryCluster;
- it creates an immutable recursive tree snapshot;
- it preserves cluster provenance;
- it uses prototype-vector geometry only.

The first implementation uses deterministic agglomerative grouping:
- level-1 nodes wrap existing MemoryCluster objects;
- higher levels group nearest nodes while distance <= merge_radius;
- a singleton may be promoted unchanged to the next level;
- grouping repeats until one root remains or max_depth is reached.

This provides the physical "Matryoshka" structure that retrieval can later use
directly instead of synthesizing parent/grandparent aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.hierarchical_memory import (
    HierarchicalMemory,
    MemoryCluster,
)


# =============================================================================
# Errors
# =============================================================================


class RecursiveMemoryError(RuntimeError):
    pass


class InvalidRecursiveMemoryNodeError(RecursiveMemoryError):
    pass


# =============================================================================
# Helpers
# =============================================================================


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _euclidean(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise RecursiveMemoryError(
            "prototype vectors must have equal length"
        )

    return sqrt(
        sum(
            (
                float(a)
                - float(b)
            ) ** 2
            for a, b in zip(
                left,
                right,
            )
        )
    )


def _centroid(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    if not vectors:
        raise RecursiveMemoryError(
            "cannot build centroid from no vectors"
        )

    size = len(vectors[0])

    if any(
        len(vector) != size
        for vector in vectors
    ):
        raise RecursiveMemoryError(
            "all prototype vectors must have equal length"
        )

    return tuple(
        sum(
            float(vector[index])
            for vector in vectors
        ) / len(vectors)
        for index in range(size)
    )


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True, slots=True)
class RecursiveMemoryConfig:
    merge_radius: float = 0.25
    max_children: int = 4
    max_depth: int = 8

    def __post_init__(self) -> None:
        if self.merge_radius < 0.0:
            raise RecursiveMemoryError(
                "merge_radius must be non-negative"
            )

        if self.max_children < 2:
            raise RecursiveMemoryError(
                "max_children must be >= 2"
            )

        if self.max_depth < 1:
            raise RecursiveMemoryError(
                "max_depth must be >= 1"
            )


# =============================================================================
# Node
# =============================================================================


@dataclass(frozen=True, slots=True)
class RecursiveMemoryNode:
    node_id: str
    level: int

    prototype_vector: tuple[float, ...]
    radius: float

    leaf_cluster_ids: tuple[str, ...]
    child_ids: tuple[str, ...] = ()
    parent_id: str | None = None

    member_count: int = 0
    representative_cluster_id: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise InvalidRecursiveMemoryNodeError(
                "node_id must not be empty"
            )

        if self.level < 1:
            raise InvalidRecursiveMemoryNodeError(
                "level must be >= 1"
            )

        if self.radius < 0.0:
            raise InvalidRecursiveMemoryNodeError(
                "radius must be non-negative"
            )

        leaf_ids = tuple(self.leaf_cluster_ids)
        child_ids = tuple(self.child_ids)

        if not leaf_ids:
            raise InvalidRecursiveMemoryNodeError(
                "node must contain at least one leaf cluster"
            )

        if len(set(leaf_ids)) != len(leaf_ids):
            raise InvalidRecursiveMemoryNodeError(
                "leaf cluster ids must be unique"
            )

        if len(set(child_ids)) != len(child_ids):
            raise InvalidRecursiveMemoryNodeError(
                "child ids must be unique"
            )

        object.__setattr__(
            self,
            "prototype_vector",
            tuple(
                float(value)
                for value in self.prototype_vector
            ),
        )

        object.__setattr__(
            self,
            "leaf_cluster_ids",
            leaf_ids,
        )

        object.__setattr__(
            self,
            "child_ids",
            child_ids,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )

    @property
    def is_leaf_wrapper(self) -> bool:
        return self.level == 1 and not self.child_ids


# =============================================================================
# Tree
# =============================================================================


@dataclass(frozen=True, slots=True)
class RecursiveMemoryTree:
    nodes: tuple[RecursiveMemoryNode, ...]
    root_ids: tuple[str, ...]
    leaf_cluster_ids: tuple[str, ...]
    max_level: int
    config: RecursiveMemoryConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        roots = tuple(self.root_ids)
        leaves = tuple(self.leaf_cluster_ids)

        ids = tuple(
            node.node_id
            for node in nodes
        )

        if len(set(ids)) != len(ids):
            raise RecursiveMemoryError(
                "recursive node ids must be unique"
            )

        node_id_set = set(ids)

        if not set(roots).issubset(node_id_set):
            raise RecursiveMemoryError(
                "root ids must reference existing nodes"
            )

        for node in nodes:
            if not set(node.child_ids).issubset(node_id_set):
                raise RecursiveMemoryError(
                    f"node {node.node_id} references unknown child"
                )

            if (
                node.parent_id is not None
                and node.parent_id not in node_id_set
            ):
                raise RecursiveMemoryError(
                    f"node {node.node_id} references unknown parent"
                )

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "root_ids", roots)
        object.__setattr__(self, "leaf_cluster_ids", leaves)
        object.__setattr__(self, "metadata", _readonly(self.metadata))

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def root_count(self) -> int:
        return len(self.root_ids)


# =============================================================================
# Lookup
# =============================================================================


def recursive_node_by_id(
    tree: RecursiveMemoryTree,
    node_id: str,
) -> RecursiveMemoryNode:
    for node in tree.nodes:
        if node.node_id == node_id:
            return node

    raise KeyError(node_id)


def children_of(
    tree: RecursiveMemoryTree,
    node_id: str,
) -> tuple[RecursiveMemoryNode, ...]:
    node = recursive_node_by_id(
        tree,
        node_id,
    )

    return tuple(
        recursive_node_by_id(
            tree,
            child_id,
        )
        for child_id in node.child_ids
    )


def parent_of(
    tree: RecursiveMemoryTree,
    node_id: str,
) -> RecursiveMemoryNode | None:
    node = recursive_node_by_id(
        tree,
        node_id,
    )

    if node.parent_id is None:
        return None

    return recursive_node_by_id(
        tree,
        node.parent_id,
    )


# =============================================================================
# Level-1 wrapping
# =============================================================================


def _leaf_node_from_cluster(
    cluster: MemoryCluster,
) -> RecursiveMemoryNode:
    return RecursiveMemoryNode(
        node_id=f"node_l1_{cluster.cluster_id}",
        level=1,
        prototype_vector=cluster.prototype_vector,
        radius=cluster.local_radius,
        leaf_cluster_ids=(
            cluster.cluster_id,
        ),
        child_ids=(),
        parent_id=None,
        member_count=cluster.size,
        representative_cluster_id=cluster.cluster_id,
        metadata={
            "node_role": "leaf_cluster_wrapper",
            "source_cluster_id": cluster.cluster_id,
            "learning_applied": False,
            "predictive_preload_applied": False,
            "policy_modified": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Grouping
# =============================================================================


def _node_distance(
    left: RecursiveMemoryNode,
    right: RecursiveMemoryNode,
) -> float:
    return _euclidean(
        left.prototype_vector,
        right.prototype_vector,
    )


def _representative_child(
    children: Sequence[RecursiveMemoryNode],
    prototype: Sequence[float],
) -> RecursiveMemoryNode:
    ranked = sorted(
        (
            (
                _euclidean(
                    child.prototype_vector,
                    prototype,
                ),
                child.node_id,
                child,
            )
            for child in children
        ),
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return ranked[0][2]


def _merge_nodes(
    children: Sequence[RecursiveMemoryNode],
    *,
    level: int,
    index: int,
) -> RecursiveMemoryNode:
    children = tuple(children)

    if not children:
        raise RecursiveMemoryError(
            "cannot merge zero children"
        )

    prototype = _centroid(
        tuple(
            child.prototype_vector
            for child in children
        )
    )

    radius = max(
        _euclidean(
            child.prototype_vector,
            prototype,
        )
        + child.radius
        for child in children
    )

    leaf_ids = tuple(
        dict.fromkeys(
            leaf_id
            for child in children
            for leaf_id in child.leaf_cluster_ids
        )
    )

    representative = _representative_child(
        children,
        prototype,
    )

    return RecursiveMemoryNode(
        node_id=f"node_l{level}_{index:04d}",
        level=level,
        prototype_vector=prototype,
        radius=radius,
        leaf_cluster_ids=leaf_ids,
        child_ids=tuple(
            child.node_id
            for child in children
        ),
        parent_id=None,
        member_count=sum(
            child.member_count
            for child in children
        ),
        representative_cluster_id=(
            representative.representative_cluster_id
        ),
        metadata={
            "node_role": "super_cluster",
            "child_count": len(children),
            "learning_applied": False,
            "predictive_preload_applied": False,
            "policy_modified": False,
            "external_expected_label_used": False,
        },
    )


def _build_groups(
    nodes: Sequence[RecursiveMemoryNode],
    *,
    config: RecursiveMemoryConfig,
) -> tuple[tuple[RecursiveMemoryNode, ...], ...]:
    """
    Deterministic greedy grouping.

    Start from lexicographically smallest remaining node.
    Add nearest compatible nodes while distance <= merge_radius and
    max_children is not exceeded.
    """

    remaining = list(
        sorted(
            nodes,
            key=lambda node: node.node_id,
        )
    )

    groups = []

    while remaining:
        seed = remaining.pop(0)
        group = [seed]

        while (
            remaining
            and len(group) < config.max_children
        ):
            group_prototype = _centroid(
                tuple(
                    node.prototype_vector
                    for node in group
                )
            )

            ranked = sorted(
                (
                    (
                        _euclidean(
                            group_prototype,
                            candidate.prototype_vector,
                        ),
                        candidate.node_id,
                        candidate,
                    )
                    for candidate in remaining
                ),
                key=lambda item: (
                    item[0],
                    item[1],
                ),
            )

            distance, _candidate_id, candidate = ranked[0]

            if distance > config.merge_radius:
                break

            group.append(candidate)
            remaining.remove(candidate)

        groups.append(
            tuple(group)
        )

    return tuple(groups)


# =============================================================================
# Tree construction
# =============================================================================


def build_recursive_memory_tree(
    memory: HierarchicalMemory,
    *,
    config: RecursiveMemoryConfig | None = None,
) -> RecursiveMemoryTree:
    config = config or RecursiveMemoryConfig()

    if not memory.clusters:
        return RecursiveMemoryTree(
            nodes=(),
            root_ids=(),
            leaf_cluster_ids=(),
            max_level=0,
            config=config,
            metadata={
                "recursive_memory_tree": True,
                "empty": True,
                "learning_applied": False,
                "predictive_preload_applied": False,
                "policy_modified": False,
                "external_expected_label_used": False,
            },
        )

    all_nodes: list[RecursiveMemoryNode] = []

    current_level = tuple(
        _leaf_node_from_cluster(
            cluster
        )
        for cluster in memory.clusters
    )

    all_nodes.extend(
        current_level
    )

    level = 1

    while (
        len(current_level) > 1
        and level < config.max_depth
    ):
        next_level_number = level + 1

        groups = _build_groups(
            current_level,
            config=config,
        )

        next_level = tuple(
            _merge_nodes(
                group,
                level=next_level_number,
                index=index,
            )
            for index, group in enumerate(
                groups,
                start=1,
            )
        )

        # If no grouping progress was possible, create a single final root.
        if (
            len(next_level)
            == len(current_level)
            and all(
                len(group) == 1
                for group in groups
            )
        ):
            next_level = (
                _merge_nodes(
                    current_level,
                    level=next_level_number,
                    index=1,
                ),
            )

        # Assign parent ids immutably by rebuilding child nodes.
        parent_map = {}

        for parent in next_level:
            for child_id in parent.child_ids:
                parent_map[child_id] = parent.node_id

        rebuilt = []

        for node in all_nodes:
            parent_id = parent_map.get(
                node.node_id,
                node.parent_id,
            )

            if parent_id != node.parent_id:
                rebuilt.append(
                    RecursiveMemoryNode(
                        node_id=node.node_id,
                        level=node.level,
                        prototype_vector=node.prototype_vector,
                        radius=node.radius,
                        leaf_cluster_ids=node.leaf_cluster_ids,
                        child_ids=node.child_ids,
                        parent_id=parent_id,
                        member_count=node.member_count,
                        representative_cluster_id=(
                            node.representative_cluster_id
                        ),
                        metadata=node.metadata,
                    )
                )
            else:
                rebuilt.append(node)

        all_nodes = rebuilt
        all_nodes.extend(next_level)

        current_level = next_level
        level = next_level_number

    root_ids = tuple(
        node.node_id
        for node in current_level
    )

    return RecursiveMemoryTree(
        nodes=tuple(all_nodes),
        root_ids=root_ids,
        leaf_cluster_ids=tuple(
            cluster.cluster_id
            for cluster in memory.clusters
        ),
        max_level=max(
            node.level
            for node in all_nodes
        ),
        config=config,
        metadata={
            "recursive_memory_tree": True,
            "empty": False,
            "source_cluster_count": memory.cluster_count,
            "learning_applied": False,
            "predictive_preload_applied": False,
            "policy_modified": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Traversal
# =============================================================================


def ancestry(
    tree: RecursiveMemoryTree,
    node_id: str,
) -> tuple[RecursiveMemoryNode, ...]:
    """
    Return node -> parent -> grandparent -> ... root.
    """

    chain = []
    current = recursive_node_by_id(
        tree,
        node_id,
    )

    while current is not None:
        chain.append(current)
        current = (
            None
            if current.parent_id is None
            else recursive_node_by_id(
                tree,
                current.parent_id,
            )
        )

    return tuple(chain)


def leaf_node_for_cluster(
    tree: RecursiveMemoryTree,
    cluster_id: str,
) -> RecursiveMemoryNode:
    for node in tree.nodes:
        if (
            node.level == 1
            and node.leaf_cluster_ids
            == (cluster_id,)
        ):
            return node

    raise KeyError(cluster_id)


def tree_is_policy_free(
    tree: RecursiveMemoryTree,
) -> bool:
    return (
        tree.metadata.get(
            "policy_modified",
            False,
        )
        is False
        and tree.metadata.get(
            "predictive_preload_applied",
            False,
        )
        is False
        and all(
            node.metadata.get(
                "policy_modified"
            )
            is False
            and node.metadata.get(
                "predictive_preload_applied"
            )
            is False
            for node in tree.nodes
        )
    )


# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "InvalidRecursiveMemoryNodeError",
    "RecursiveMemoryConfig",
    "RecursiveMemoryError",
    "RecursiveMemoryNode",
    "RecursiveMemoryTree",
    "ancestry",
    "build_recursive_memory_tree",
    "children_of",
    "leaf_node_for_cluster",
    "parent_of",
    "recursive_node_by_id",
    "tree_is_policy_free",
]
