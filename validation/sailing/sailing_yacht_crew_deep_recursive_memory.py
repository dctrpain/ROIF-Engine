"""
ROIF External-Domain Validation
Scenario 04G — Deep Recursive Memory Benchmark

Purpose
-------
Stress-test the physical RecursiveMemoryTree at greater depth than 04F.

Pipeline:

    many ExperienceTrace records
        -> ControllerHistoryAdapter
        -> fine-grained HierarchicalMemory
        -> many level-1 MemoryCluster leaves
        -> explicit RecursiveMemoryTree
        -> deep ancestry
        -> automatic retrieval
        -> PredictivePreload

Primary question
----------------
Can raw experience grow much faster than the active memory context required
for one prediction?

We measure:

    raw_record_count
    base_cluster_count
    recursive_node_count
    recursive_tree_depth
    active_ancestry_length
    active_contribution_count

The benchmark deliberately uses a very small assignment radius so repeated
events form multiple fine-grained base clusters. RecursiveMemoryTree then
compresses those leaves into higher-order super-clusters.

Architectural boundaries
------------------------

    Deep Memory != Policy Override
    Recursive Grouping != Learning
    Retrieval != Action Selection
    Cluster Geometry != Evaluator Ground Truth
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Mapping

from roif.hierarchical_memory import (
    HierarchicalMemory,
    HierarchicalMemoryConfig,
    append_controller_history_record,
)
from roif.hierarchical_memory_retrieval import (
    HierarchicalMemoryRetrieval,
    RetrievalConfig,
    retrieve_hierarchical_memory,
)
from roif.predictive_preload import (
    PredictivePreload,
    PredictivePreloadConfig,
    apply_predictive_preload,
)
from roif.recursive_hierarchical_memory import (
    RecursiveMemoryConfig,
    RecursiveMemoryTree,
    ancestry,
    build_recursive_memory_tree,
    leaf_node_for_cluster,
    tree_is_policy_free,
)

from validation.sailing.sailing_yacht_crew_long_memory_compression import (
    build_long_memory_event,
    event_to_controller_history_record,
)


SCENARIO_ID = "sailing_04G_deep_recursive_memory"

DEFAULT_EVENT_COUNT = 256
DEFAULT_QUERY_INDEX = 257

# Small enough to preserve fine geometric distinctions between repeated traces.
DEFAULT_ASSIGNMENT_RADIUS = 1.0e-6

# Binary recursive grouping intentionally creates several hierarchy levels.
DEFAULT_MAX_CHILDREN = 2

# Large merge radius lets the tree pair available nodes instead of collapsing
# immediately through the fallback root path.
DEFAULT_RECURSIVE_MERGE_RADIUS = 10.0


class DeepRecursiveMemoryBenchmarkError(RuntimeError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


@dataclass(frozen=True, slots=True)
class DeepRecursiveMemoryMetrics:
    raw_record_count: int
    base_cluster_count: int
    recursive_node_count: int
    recursive_tree_depth: int
    root_count: int

    active_ancestry_length: int
    active_contribution_count: int

    raw_to_cluster_ratio: float
    raw_to_recursive_node_ratio: float
    raw_to_active_context_ratio: float

    def __post_init__(self) -> None:
        for name in (
            "raw_record_count",
            "base_cluster_count",
            "recursive_node_count",
            "recursive_tree_depth",
            "root_count",
            "active_ancestry_length",
            "active_contribution_count",
        ):
            if getattr(self, name) < 0:
                raise DeepRecursiveMemoryBenchmarkError(
                    f"{name} must be non-negative"
                )


@dataclass(frozen=True, slots=True)
class DeepRecursiveMemoryResult:
    scenario_id: str

    memory: HierarchicalMemory
    tree: RecursiveMemoryTree

    query_index: int
    retrieval: HierarchicalMemoryRetrieval
    preload: PredictivePreload

    metrics: DeepRecursiveMemoryMetrics

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )



# =============================================================================
# Deep event-space generator
# =============================================================================


def build_deep_memory_event(
    index: int,
):
    """
    Build a deterministic multi-dimensional 04G event.

    The previous 04G reused the ordinary long-memory generator unchanged.
    That produced only two stable StructuralSignature basins even with a tiny
    assignment radius.

    Here we deliberately span a wider deterministic control-state lattice while
    preserving the original SailingDisturbanceEvent type.

    No evaluator class label is used for clustering.
    """

    if index <= 0:
        raise DeepRecursiveMemoryBenchmarkError(
            "index must be positive"
        )

    base = build_long_memory_event(
        index
    )

    available = {
        item.name
        for item in fields(base)
    }

    # 16 x 16 deterministic lattice. 256 events traverse the full lattice once.
    x = (index - 1) % 16
    y = ((index - 1) // 16) % 16

    # A slow phase adds a third deterministic dimension for runs > 256.
    phase = ((index - 1) // 256) % 8

    # Keep values bounded and separated enough to affect the downstream
    # prediction-error / structural-signature geometry.
    wind = 0.20 + 0.045 * x + 0.010 * phase
    wave = 0.15 + 0.040 * y + 0.008 * phase

    course_bias = (
        -0.18
        + 0.024 * x
        + 0.006 * phase
    )

    heel_bias = (
        -0.12
        + 0.016 * y
        - 0.004 * phase
    )

    updates = {}

    for name, value in (
        ("wind", wind),
        ("wave", wave),
        ("course_bias", course_bias),
        ("heel_bias", heel_bias),
    ):
        if name in available:
            updates[name] = value

    # Preserve the original metadata but make 04G provenance explicit.
    if "metadata" in available:
        metadata = dict(
            getattr(
                base,
                "metadata",
                {},
            )
        )

        metadata.update(
            {
                "scenario_id": SCENARIO_ID,
                "deep_recursive_event": True,
                "deep_lattice_x": x,
                "deep_lattice_y": y,
                "deep_lattice_phase": phase,
                "external_expected_label_used": False,
            }
        )

        updates[
            "metadata"
        ] = metadata

    # Give each generated event a unique 04G identity when the event type
    # exposes event_id.
    if "event_id" in available:
        updates[
            "event_id"
        ] = (
            f"04G_event_"
            f"{index:05d}_"
            f"x{x:02d}_y{y:02d}_p{phase:02d}"
        )

    event = replace(
        base,
        **updates,
    )

    # Fail loudly if the runtime event schema does not expose any of the four
    # control dimensions we intended to vary. This protects the benchmark from
    # silently degenerating back into the old two-cluster case.
    if not any(
        name in available
        for name in (
            "wind",
            "wave",
            "course_bias",
            "heel_bias",
        )
    ):
        raise DeepRecursiveMemoryBenchmarkError(
            "SailingDisturbanceEvent exposes none of the expected deep-memory "
            "dimensions: wind, wave, course_bias, heel_bias"
        )

    return event


# =============================================================================
# Memory construction
# =============================================================================


def build_deep_recursive_history(
    *,
    event_count: int = DEFAULT_EVENT_COUNT,
    assignment_radius: float = DEFAULT_ASSIGNMENT_RADIUS,
) -> HierarchicalMemory:
    if event_count < 8:
        raise DeepRecursiveMemoryBenchmarkError(
            "event_count must be >= 8"
        )

    if assignment_radius < 0.0:
        raise DeepRecursiveMemoryBenchmarkError(
            "assignment_radius must be non-negative"
        )

    memory = HierarchicalMemory(
        config=HierarchicalMemoryConfig(
            assignment_radius=assignment_radius,
            anomaly_radius=0.50,
            max_representatives=5,
            representative_refresh_interval=10,
        ),
        metadata={
            "scenario_id": SCENARIO_ID,
            "deep_recursive_history": True,
            "predictive_preload_applied": False,
            "policy_modified": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )

    for index in range(
        1,
        event_count + 1,
    ):
        event = build_deep_memory_event(
            index
        )

        record = event_to_controller_history_record(
            event=event,
            episode_index=index,
            sequence_id="04G_deep_recursive_history",
        )

        memory = append_controller_history_record(
            memory,
            record,
        )

    return memory


def build_deep_recursive_tree(
    memory: HierarchicalMemory,
    *,
    merge_radius: float = DEFAULT_RECURSIVE_MERGE_RADIUS,
    max_children: int = DEFAULT_MAX_CHILDREN,
    max_depth: int = 16,
) -> RecursiveMemoryTree:
    if max_depth < 3:
        raise DeepRecursiveMemoryBenchmarkError(
            "max_depth must be >= 3"
        )

    return build_recursive_memory_tree(
        memory,
        config=RecursiveMemoryConfig(
            merge_radius=merge_radius,
            max_children=max_children,
            max_depth=max_depth,
        ),
    )


# =============================================================================
# Query / retrieval
# =============================================================================


def build_deep_query_record(
    *,
    query_index: int = DEFAULT_QUERY_INDEX,
):
    if query_index <= 0:
        raise DeepRecursiveMemoryBenchmarkError(
            "query_index must be positive"
        )

    event = build_deep_memory_event(
        query_index
    )

    return event_to_controller_history_record(
        event=event,
        episode_index=query_index,
        sequence_id="04G_deep_recursive_query",
    )


def run_deep_recursive_retrieval(
    memory: HierarchicalMemory,
    tree: RecursiveMemoryTree,
    *,
    query_index: int = DEFAULT_QUERY_INDEX,
    max_hierarchy_levels: int = 16,
) -> tuple[
    HierarchicalMemoryRetrieval,
    PredictivePreload,
]:
    query_record = build_deep_query_record(
        query_index=query_index
    )

    retrieval = retrieve_hierarchical_memory(
        memory,
        tree,
        query_record,
        config=RetrievalConfig(
            max_hierarchy_levels=max_hierarchy_levels,
            local_weight=1.0,
            parent_weight=1.0,
            grandparent_weight=1.0,
            minimum_cluster_members_for_scar=2,
        ),
    )

    preload = apply_predictive_preload(
        query_record.source_trace.initial_state,
        retrieval.contributions,
        config=PredictivePreloadConfig(
            max_bias_fraction=0.75,
            uncertainty_reduction_scale=0.35,
            minimum_scar_confidence=0.25,
            minimum_scar_consistency=0.80,
            require_systematic_scar=True,
            local_level_decay=0.50,
        ),
        metadata={
            "scenario_id": SCENARIO_ID,
            "deep_recursive_retrieval": True,
            "external_expected_label_used": False,
        },
    )

    return retrieval, preload


# =============================================================================
# Metrics
# =============================================================================


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return float(numerator) / float(denominator)


def compute_deep_recursive_metrics(
    memory: HierarchicalMemory,
    tree: RecursiveMemoryTree,
    retrieval: HierarchicalMemoryRetrieval,
) -> DeepRecursiveMemoryMetrics:
    raw = memory.record_count
    clusters = memory.cluster_count
    nodes = tree.node_count

    active_ancestry = 0

    if retrieval.leaf_node_id is not None:
        active_ancestry = len(
            ancestry(
                tree,
                retrieval.leaf_node_id,
            )
        )

    active_context = len(
        retrieval.contributions
    )

    return DeepRecursiveMemoryMetrics(
        raw_record_count=raw,
        base_cluster_count=clusters,
        recursive_node_count=nodes,
        recursive_tree_depth=tree.max_level,
        root_count=tree.root_count,
        active_ancestry_length=active_ancestry,
        active_contribution_count=active_context,
        raw_to_cluster_ratio=_safe_ratio(
            raw,
            clusters,
        ),
        raw_to_recursive_node_ratio=_safe_ratio(
            raw,
            nodes,
        ),
        raw_to_active_context_ratio=_safe_ratio(
            raw,
            active_context,
        ),
    )


# =============================================================================
# Full benchmark
# =============================================================================


def run_deep_recursive_memory_benchmark(
    *,
    event_count: int = DEFAULT_EVENT_COUNT,
    query_index: int = DEFAULT_QUERY_INDEX,
    assignment_radius: float = DEFAULT_ASSIGNMENT_RADIUS,
    merge_radius: float = DEFAULT_RECURSIVE_MERGE_RADIUS,
    max_children: int = DEFAULT_MAX_CHILDREN,
    max_depth: int = 16,
) -> DeepRecursiveMemoryResult:
    memory = build_deep_recursive_history(
        event_count=event_count,
        assignment_radius=assignment_radius,
    )

    tree = build_deep_recursive_tree(
        memory,
        merge_radius=merge_radius,
        max_children=max_children,
        max_depth=max_depth,
    )

    retrieval, preload = run_deep_recursive_retrieval(
        memory,
        tree,
        query_index=query_index,
        max_hierarchy_levels=max_depth,
    )

    metrics = compute_deep_recursive_metrics(
        memory,
        tree,
        retrieval,
    )

    return DeepRecursiveMemoryResult(
        scenario_id=SCENARIO_ID,
        memory=memory,
        tree=tree,
        query_index=query_index,
        retrieval=retrieval,
        preload=preload,
        metrics=metrics,
        metadata={
            "benchmark_type": "deep_recursive_memory",
            "recursive_tree_enabled": True,
            "deep_recursive_retrieval": True,
            "policy_override_enabled": False,
            "action_selected_by_memory": False,
            "tree_policy_free": tree_is_policy_free(
                tree
            ),
            "cluster_assignment_uses_external_pattern_label": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Reporting helpers
# =============================================================================


def retrieval_tree_node_ids(
    result: DeepRecursiveMemoryResult,
) -> tuple[str, ...]:
    return tuple(
        level.tree_node_id
        for level in result.retrieval.levels
    )


def retrieval_tree_levels(
    result: DeepRecursiveMemoryResult,
) -> tuple[int, ...]:
    return tuple(
        level.tree_level
        for level in result.retrieval.levels
    )


def physical_ancestry_node_ids(
    result: DeepRecursiveMemoryResult,
) -> tuple[str, ...]:
    if result.retrieval.leaf_node_id is None:
        return ()

    return tuple(
        node.node_id
        for node in ancestry(
            result.tree,
            result.retrieval.leaf_node_id,
        )
    )


def leaf_cluster_ancestry(
    result: DeepRecursiveMemoryResult,
) -> tuple[
    tuple[str, tuple[str, ...]],
    ...,
]:
    if result.retrieval.leaf_node_id is None:
        return ()

    return tuple(
        (
            node.node_id,
            node.leaf_cluster_ids,
        )
        for node in ancestry(
            result.tree,
            result.retrieval.leaf_node_id,
        )
    )


__all__ = [
    "DEFAULT_ASSIGNMENT_RADIUS",
    "DEFAULT_EVENT_COUNT",
    "DEFAULT_MAX_CHILDREN",
    "DEFAULT_QUERY_INDEX",
    "DEFAULT_RECURSIVE_MERGE_RADIUS",
    "DeepRecursiveMemoryBenchmarkError",
    "DeepRecursiveMemoryMetrics",
    "DeepRecursiveMemoryResult",
    "SCENARIO_ID",
    "build_deep_memory_event",
    "build_deep_query_record",
    "build_deep_recursive_history",
    "build_deep_recursive_tree",
    "compute_deep_recursive_metrics",
    "leaf_cluster_ancestry",
    "physical_ancestry_node_ids",
    "retrieval_tree_levels",
    "retrieval_tree_node_ids",
    "run_deep_recursive_memory_benchmark",
    "run_deep_recursive_retrieval",
]
