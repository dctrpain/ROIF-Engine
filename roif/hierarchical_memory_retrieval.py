"""
ROIF Hierarchical Memory Retrieval
=================================

Purpose
-------
Retrieve nested memory directly from the explicit RecursiveMemoryTree.

Input:
    current ControllerHistoryRecord
    HierarchicalMemory
    RecursiveMemoryTree

Output:
    HierarchicalMemoryRetrieval
        -> local leaf node
        -> parent recursive node
        -> grandparent recursive node
        -> MemoryPreloadContribution[]

Architectural boundaries
------------------------

    Retrieval != Action Selection
    Retrieval != Policy Override
    Recursive Tree != Ground Truth
    Cluster Match != Evaluator Label
    Retrieval != Predictive State Mutation

Key change from the previous implementation
-------------------------------------------
Parent and grandparent context are NO LONGER synthesized from temporary
cluster aggregates.

Instead:

    nearest MemoryCluster
        -> leaf RecursiveMemoryNode
        -> parent_id
        -> grandparent parent_id
        -> ...

This makes the "Matryoshka" hierarchy explicit and physically stored in the
recursive memory tree.

HierarchicalMemory remains the source of raw ControllerHistoryRecord objects
for each leaf cluster. RecursiveMemoryTree supplies hierarchy topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.controller_memory import (
    MemoryPattern,
    MemoryScar,
    derive_memory_scar,
)
from roif.hierarchical_memory import (
    HierarchicalMemory,
    nearest_cluster,
    record_by_id,
)
from roif.history.controller_history_adapter import (
    ControllerHistoryRecord,
)
from roif.predictive_preload import (
    MemoryPreloadContribution,
)
from roif.recursive_hierarchical_memory import (
    RecursiveMemoryNode,
    RecursiveMemoryTree,
    ancestry,
    leaf_node_for_cluster,
    recursive_node_by_id,
)


# =============================================================================
# Errors
# =============================================================================


class HierarchicalMemoryRetrievalError(RuntimeError):
    pass


# =============================================================================
# Helpers
# =============================================================================


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {}
        if value is None
        else dict(value)
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """
    Retrieval configuration for explicit recursive memory.

    max_hierarchy_levels:
        Maximum number of hierarchy levels to expose beginning at local level 0.

        Example:
            1 -> local only
            2 -> local + parent
            3 -> local + parent + grandparent

    local_weight / parent_weight / grandparent_weight:
        Explicit contribution weights passed into PredictivePreload.
        PredictivePreload still applies its own hierarchy decay.

    minimum_cluster_members_for_scar:
        Minimum number of source traces required to derive a MemoryScar.
    """

    max_hierarchy_levels: int = 3

    local_weight: float = 1.0
    parent_weight: float = 1.0
    grandparent_weight: float = 1.0

    minimum_cluster_members_for_scar: int = 2

    def __post_init__(self) -> None:
        if self.max_hierarchy_levels < 1:
            raise HierarchicalMemoryRetrievalError(
                "max_hierarchy_levels must be >= 1"
            )

        if self.minimum_cluster_members_for_scar < 2:
            raise HierarchicalMemoryRetrievalError(
                "minimum_cluster_members_for_scar must be >= 2"
            )

        for name, value in (
            ("local_weight", self.local_weight),
            ("parent_weight", self.parent_weight),
            ("grandparent_weight", self.grandparent_weight),
        ):
            if value < 0.0:
                raise HierarchicalMemoryRetrievalError(
                    f"{name} must be non-negative"
                )


# =============================================================================
# Retrieval output
# =============================================================================


@dataclass(frozen=True, slots=True)
class RetrievalLevel:
    """
    One explicit hierarchy level used for memory retrieval.

    level:
        Preload hierarchy level:
            0 local
            1 parent
            2 grandparent
            ...

    tree_node_id:
        Actual RecursiveMemoryNode id.

    leaf_cluster_ids:
        All base MemoryCluster ids represented by this recursive node.
    """

    level: int
    tree_node_id: str
    tree_level: int
    leaf_cluster_ids: tuple[str, ...]
    scar: MemoryScar

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.level < 0:
            raise HierarchicalMemoryRetrievalError(
                "retrieval level must be non-negative"
            )

        if self.tree_level < 1:
            raise HierarchicalMemoryRetrievalError(
                "tree_level must be >= 1"
            )

        object.__setattr__(
            self,
            "leaf_cluster_ids",
            tuple(
                self.leaf_cluster_ids
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class HierarchicalMemoryRetrieval:
    query_record_id: str

    nearest_cluster_id: str | None
    nearest_distance: float | None

    leaf_node_id: str | None

    levels: tuple[
        RetrievalLevel,
        ...,
    ]

    contributions: tuple[
        MemoryPreloadContribution,
        ...,
    ]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "levels",
            tuple(
                self.levels
            ),
        )

        object.__setattr__(
            self,
            "contributions",
            tuple(
                self.contributions
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


# =============================================================================
# Source trace extraction
# =============================================================================


def _records_for_leaf_cluster_ids(
    memory: HierarchicalMemory,
    leaf_cluster_ids: Sequence[str],
) -> tuple[
    ControllerHistoryRecord,
    ...,
]:
    requested = set(
        leaf_cluster_ids
    )

    records = []

    for cluster in memory.clusters:
        if cluster.cluster_id not in requested:
            continue

        for record_id in cluster.member_record_ids:
            records.append(
                record_by_id(
                    memory,
                    record_id,
                )
            )

    return tuple(
        records
    )


def _scar_for_recursive_node(
    memory: HierarchicalMemory,
    node: RecursiveMemoryNode,
    *,
    scar_id: str,
    pattern_id: str,
    minimum_members: int,
) -> MemoryScar | None:
    records = _records_for_leaf_cluster_ids(
        memory,
        node.leaf_cluster_ids,
    )

    traces = tuple(
        record.source_trace
        for record in records
        if record.source_trace.prediction_error is not None
    )

    if len(
        traces
    ) < minimum_members:
        return None

    return derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id=pattern_id,
            metadata={
                "retrieval_generated": True,
                "recursive_tree_node_id": node.node_id,
                "recursive_tree_level": node.level,
            },
        ),
        traces=traces,
        scar_id=scar_id,
        metadata={
            "retrieval_generated": True,
            "recursive_tree_node_id": node.node_id,
            "recursive_tree_level": node.level,
            "source_leaf_cluster_ids": node.leaf_cluster_ids,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Weight helper
# =============================================================================


def _explicit_weight_for_level(
    level: int,
    config: RetrievalConfig,
) -> float:
    if level == 0:
        return config.local_weight

    if level == 1:
        return config.parent_weight

    if level == 2:
        return config.grandparent_weight

    # Deeper levels use grandparent_weight as the generic higher-context weight.
    return config.grandparent_weight


# =============================================================================
# Main retrieval
# =============================================================================


def retrieve_hierarchical_memory(
    memory: HierarchicalMemory,
    tree: RecursiveMemoryTree,
    query_record: ControllerHistoryRecord,
    *,
    config: RetrievalConfig | None = None,
) -> HierarchicalMemoryRetrieval:
    """
    Retrieve memory through explicit RecursiveMemoryTree ancestry.

    Steps
    -----
    1. Find nearest base MemoryCluster from query StructuralSignature.
    2. Resolve that cluster to its level-1 RecursiveMemoryNode.
    3. Traverse:
           leaf -> parent -> grandparent -> ...
    4. Derive one MemoryScar per recursive node from all source traces beneath it.
    5. Convert those scars into MemoryPreloadContribution objects.

    External pattern labels are not used in steps 1-5.
    """

    config = config or RetrievalConfig()

    match = nearest_cluster(
        memory,
        query_record,
    )

    if match is None:
        return HierarchicalMemoryRetrieval(
            query_record_id=query_record.trace_id,
            nearest_cluster_id=None,
            nearest_distance=None,
            leaf_node_id=None,
            levels=(),
            contributions=(),
            metadata={
                "retrieval_performed": True,
                "memory_empty": True,
                "recursive_tree_used": True,
                "action_selected": False,
                "policy_modified": False,
                "predictive_state_mutated": False,
                "external_expected_label_used": False,
            },
        )

    try:
        leaf_node = leaf_node_for_cluster(
            tree,
            match.cluster_id,
        )
    except KeyError as exc:
        raise HierarchicalMemoryRetrievalError(
            "nearest cluster is not represented in RecursiveMemoryTree"
        ) from exc

    chain = ancestry(
        tree,
        leaf_node.node_id,
    )

    chain = chain[
        : config.max_hierarchy_levels
    ]

    levels = []
    contributions = []

    for preload_level, node in enumerate(
        chain
    ):
        scar = _scar_for_recursive_node(
            memory,
            node,
            scar_id=(
                f"retrieval_node:"
                f"{node.node_id}"
            ),
            pattern_id=(
                f"recursive_context:"
                f"{node.node_id}"
            ),
            minimum_members=(
                config.minimum_cluster_members_for_scar
            ),
        )

        if scar is None:
            continue

        retrieval_level = RetrievalLevel(
            level=preload_level,
            tree_node_id=node.node_id,
            tree_level=node.level,
            leaf_cluster_ids=node.leaf_cluster_ids,
            scar=scar,
            metadata={
                "retrieval_level": preload_level,
                "recursive_tree_level": node.level,
                "recursive_tree_node_id": node.node_id,
                "external_expected_label_used": False,
            },
        )

        levels.append(
            retrieval_level
        )

        contributions.append(
            MemoryPreloadContribution(
                scar=scar,
                hierarchy_level=preload_level,
                explicit_weight=_explicit_weight_for_level(
                    preload_level,
                    config,
                ),
                source_cluster_id=node.node_id,
                metadata={
                    "retrieval_level": preload_level,
                    "recursive_tree_level": node.level,
                    "recursive_tree_node_id": node.node_id,
                    "external_expected_label_used": False,
                },
            )
        )

    return HierarchicalMemoryRetrieval(
        query_record_id=query_record.trace_id,
        nearest_cluster_id=match.cluster_id,
        nearest_distance=match.distance,
        leaf_node_id=leaf_node.node_id,
        levels=tuple(
            levels
        ),
        contributions=tuple(
            contributions
        ),
        metadata={
            "retrieval_performed": True,
            "memory_empty": False,
            "recursive_tree_used": True,
            "nearest_cluster_within_assignment_radius": (
                match.within_assignment_radius
            ),
            "nearest_cluster_within_anomaly_radius": (
                match.within_anomaly_radius
            ),
            "action_selected": False,
            "policy_modified": False,
            "predictive_state_mutated": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Inspection helpers
# =============================================================================


def retrieval_levels(
    retrieval: HierarchicalMemoryRetrieval,
) -> tuple[int, ...]:
    return tuple(
        level.level
        for level in retrieval.levels
    )


def retrieval_tree_node_ids(
    retrieval: HierarchicalMemoryRetrieval,
) -> tuple[str, ...]:
    return tuple(
        level.tree_node_id
        for level in retrieval.levels
    )


def retrieval_tree_levels(
    retrieval: HierarchicalMemoryRetrieval,
) -> tuple[int, ...]:
    return tuple(
        level.tree_level
        for level in retrieval.levels
    )


def retrieval_is_policy_free(
    retrieval: HierarchicalMemoryRetrieval,
) -> bool:
    return (
        retrieval.metadata.get(
            "action_selected"
        )
        is False
        and retrieval.metadata.get(
            "policy_modified"
        )
        is False
        and retrieval.metadata.get(
            "predictive_state_mutated"
        )
        is False
    )


# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "HierarchicalMemoryRetrieval",
    "HierarchicalMemoryRetrievalError",
    "RetrievalConfig",
    "RetrievalLevel",
    "retrieve_hierarchical_memory",
    "retrieval_is_policy_free",
    "retrieval_levels",
    "retrieval_tree_levels",
    "retrieval_tree_node_ids",
]
