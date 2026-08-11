"""
ROIF External-Domain Validation
Scenario 04H — Recursive Memory Scaling Benchmark

Purpose
-------
Measure how recursive-memory structure and active retrieval cost scale as
raw experience grows.

This benchmark builds on 04G and evaluates a deterministic sequence of memory
sizes:

    raw records
        -> HierarchicalMemory
        -> RecursiveMemoryTree
        -> one representative query
        -> active ancestry
        -> PredictivePreload

Primary question
----------------
Does active retrieval context grow substantially slower than accumulated raw
experience?

Default checkpoints:

    64
    128
    256
    512
    1024
    2048

Measured per checkpoint:

    raw_record_count
    base_cluster_count
    recursive_node_count
    recursive_tree_depth
    root_count
    active_ancestry_length
    active_contribution_count
    raw_to_cluster_ratio
    raw_to_recursive_node_ratio
    raw_to_active_context_ratio

Derived scaling metrics:

    raw_growth_factor
    cluster_growth_factor
    recursive_node_growth_factor
    depth_growth_factor
    active_ancestry_growth_factor
    active_context_growth_factor

    active_context_scaling_exponent
    ancestry_scaling_exponent
    recursive_node_scaling_exponent

Architectural boundaries
------------------------

    Scaling Benchmark != Learning
    Scaling Benchmark != Policy Override
    Retrieval Cost != Raw Memory Size
    Recursive Growth != Evaluator Ground Truth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from validation.sailing.sailing_yacht_crew_deep_recursive_memory import (
    DEFAULT_ASSIGNMENT_RADIUS,
    DEFAULT_MAX_CHILDREN,
    DEFAULT_RECURSIVE_MERGE_RADIUS,
    DeepRecursiveMemoryResult,
    run_deep_recursive_memory_benchmark,
)


SCENARIO_ID = "sailing_04H_recursive_memory_scaling"

DEFAULT_EVENT_COUNTS = (
    64,
    128,
    256,
    512,
    1024,
    2048,
)

DEFAULT_QUERY_OFFSET = 1
DEFAULT_MAX_DEPTH = 24


class RecursiveMemoryScalingError(RuntimeError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _growth_factor(
    first: float,
    last: float,
) -> float:
    if first <= 0.0:
        return 0.0
    return float(last) / float(first)


def _scaling_exponent(
    raw_first: float,
    raw_last: float,
    value_first: float,
    value_last: float,
) -> float:
    """
    Estimate exponent alpha in:

        value ~ raw ** alpha

    using only first and last checkpoints.
    """

    if (
        raw_first <= 0.0
        or raw_last <= 0.0
        or value_first <= 0.0
        or value_last <= 0.0
        or raw_first == raw_last
    ):
        return 0.0

    denominator = log(
        float(raw_last)
        / float(raw_first)
    )

    if abs(
        denominator
    ) <= 1.0e-12:
        return 0.0

    return log(
        float(value_last)
        / float(value_first)
    ) / denominator


# =============================================================================
# Checkpoint result
# =============================================================================


@dataclass(frozen=True, slots=True)
class RecursiveMemoryScalingCheckpoint:
    event_count: int
    query_index: int

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

    used_memory: bool
    policy_override_enabled: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.event_count <= 0:
            raise RecursiveMemoryScalingError(
                "event_count must be positive"
            )

        if self.query_index <= 0:
            raise RecursiveMemoryScalingError(
                "query_index must be positive"
            )

        for name in (
            "raw_record_count",
            "base_cluster_count",
            "recursive_node_count",
            "recursive_tree_depth",
            "root_count",
            "active_ancestry_length",
            "active_contribution_count",
        ):
            if getattr(
                self,
                name,
            ) < 0:
                raise RecursiveMemoryScalingError(
                    f"{name} must be non-negative"
                )

        for name in (
            "raw_to_cluster_ratio",
            "raw_to_recursive_node_ratio",
            "raw_to_active_context_ratio",
        ):
            value = float(
                getattr(
                    self,
                    name,
                )
            )

            if (
                not isfinite(value)
                or value < 0.0
            ):
                raise RecursiveMemoryScalingError(
                    f"{name} must be finite and non-negative"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


# =============================================================================
# Aggregate scaling summary
# =============================================================================


@dataclass(frozen=True, slots=True)
class RecursiveMemoryScalingSummary:
    raw_growth_factor: float
    cluster_growth_factor: float
    recursive_node_growth_factor: float
    depth_growth_factor: float
    active_ancestry_growth_factor: float
    active_context_growth_factor: float

    active_context_scaling_exponent: float
    ancestry_scaling_exponent: float
    recursive_node_scaling_exponent: float

    final_raw_to_active_context_ratio: float
    final_tree_depth: int
    final_active_context_size: int

    sublinear_active_context: bool
    sublinear_active_ancestry: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "raw_growth_factor",
            "cluster_growth_factor",
            "recursive_node_growth_factor",
            "depth_growth_factor",
            "active_ancestry_growth_factor",
            "active_context_growth_factor",
            "active_context_scaling_exponent",
            "ancestry_scaling_exponent",
            "recursive_node_scaling_exponent",
            "final_raw_to_active_context_ratio",
        ):
            value = float(
                getattr(
                    self,
                    name,
                )
            )

            if not isfinite(value):
                raise RecursiveMemoryScalingError(
                    f"{name} must be finite"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class RecursiveMemoryScalingResult:
    scenario_id: str

    checkpoints: tuple[
        RecursiveMemoryScalingCheckpoint,
        ...,
    ]

    summary: RecursiveMemoryScalingSummary

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        checkpoints = tuple(
            self.checkpoints
        )

        if not checkpoints:
            raise RecursiveMemoryScalingError(
                "at least one checkpoint is required"
            )

        object.__setattr__(
            self,
            "checkpoints",
            checkpoints,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


# =============================================================================
# Input validation
# =============================================================================


def normalize_event_counts(
    event_counts: Iterable[int],
) -> tuple[int, ...]:
    normalized = tuple(
        int(value)
        for value in event_counts
    )

    if not normalized:
        raise RecursiveMemoryScalingError(
            "event_counts must not be empty"
        )

    if any(
        value < 8
        for value in normalized
    ):
        raise RecursiveMemoryScalingError(
            "all event counts must be >= 8"
        )

    if any(
        right <= left
        for left, right in zip(
            normalized,
            normalized[
                1:
            ],
        )
    ):
        raise RecursiveMemoryScalingError(
            "event_counts must be strictly increasing"
        )

    return normalized


# =============================================================================
# One checkpoint
# =============================================================================


def build_scaling_checkpoint(
    *,
    event_count: int,
    query_offset: int = DEFAULT_QUERY_OFFSET,
    assignment_radius: float = DEFAULT_ASSIGNMENT_RADIUS,
    merge_radius: float = DEFAULT_RECURSIVE_MERGE_RADIUS,
    max_children: int = DEFAULT_MAX_CHILDREN,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> RecursiveMemoryScalingCheckpoint:
    if query_offset < 1:
        raise RecursiveMemoryScalingError(
            "query_offset must be >= 1"
        )

    query_index = (
        event_count
        + query_offset
    )

    benchmark = run_deep_recursive_memory_benchmark(
        event_count=event_count,
        query_index=query_index,
        assignment_radius=assignment_radius,
        merge_radius=merge_radius,
        max_children=max_children,
        max_depth=max_depth,
    )

    metrics = benchmark.metrics

    return RecursiveMemoryScalingCheckpoint(
        event_count=event_count,
        query_index=query_index,
        raw_record_count=metrics.raw_record_count,
        base_cluster_count=metrics.base_cluster_count,
        recursive_node_count=metrics.recursive_node_count,
        recursive_tree_depth=metrics.recursive_tree_depth,
        root_count=metrics.root_count,
        active_ancestry_length=metrics.active_ancestry_length,
        active_contribution_count=metrics.active_contribution_count,
        raw_to_cluster_ratio=metrics.raw_to_cluster_ratio,
        raw_to_recursive_node_ratio=metrics.raw_to_recursive_node_ratio,
        raw_to_active_context_ratio=metrics.raw_to_active_context_ratio,
        used_memory=benchmark.preload.used_memory,
        policy_override_enabled=bool(
            benchmark.metadata[
                "policy_override_enabled"
            ]
        ),
        metadata={
            "scenario_id": SCENARIO_ID,
            "source_scenario_id": benchmark.scenario_id,
            "recursive_tree_enabled": True,
            "deep_recursive_retrieval": True,
            "action_selected_by_memory": benchmark.metadata[
                "action_selected_by_memory"
            ],
            "cluster_assignment_uses_external_pattern_label": benchmark.metadata[
                "cluster_assignment_uses_external_pattern_label"
            ],
            "external_expected_label_used": benchmark.metadata[
                "external_expected_label_used"
            ],
        },
    )


# =============================================================================
# Summary
# =============================================================================


def summarize_scaling(
    checkpoints: Iterable[
        RecursiveMemoryScalingCheckpoint
    ],
) -> RecursiveMemoryScalingSummary:
    checkpoints = tuple(
        checkpoints
    )

    if not checkpoints:
        raise RecursiveMemoryScalingError(
            "cannot summarize zero checkpoints"
        )

    first = checkpoints[
        0
    ]

    last = checkpoints[
        -1
    ]

    raw_growth = _growth_factor(
        first.raw_record_count,
        last.raw_record_count,
    )

    cluster_growth = _growth_factor(
        first.base_cluster_count,
        last.base_cluster_count,
    )

    node_growth = _growth_factor(
        first.recursive_node_count,
        last.recursive_node_count,
    )

    depth_growth = _growth_factor(
        first.recursive_tree_depth,
        last.recursive_tree_depth,
    )

    ancestry_growth = _growth_factor(
        first.active_ancestry_length,
        last.active_ancestry_length,
    )

    active_growth = _growth_factor(
        first.active_contribution_count,
        last.active_contribution_count,
    )

    active_exponent = _scaling_exponent(
        first.raw_record_count,
        last.raw_record_count,
        first.active_contribution_count,
        last.active_contribution_count,
    )

    ancestry_exponent = _scaling_exponent(
        first.raw_record_count,
        last.raw_record_count,
        first.active_ancestry_length,
        last.active_ancestry_length,
    )

    node_exponent = _scaling_exponent(
        first.raw_record_count,
        last.raw_record_count,
        first.recursive_node_count,
        last.recursive_node_count,
    )

    return RecursiveMemoryScalingSummary(
        raw_growth_factor=raw_growth,
        cluster_growth_factor=cluster_growth,
        recursive_node_growth_factor=node_growth,
        depth_growth_factor=depth_growth,
        active_ancestry_growth_factor=ancestry_growth,
        active_context_growth_factor=active_growth,
        active_context_scaling_exponent=active_exponent,
        ancestry_scaling_exponent=ancestry_exponent,
        recursive_node_scaling_exponent=node_exponent,
        final_raw_to_active_context_ratio=(
            last.raw_to_active_context_ratio
        ),
        final_tree_depth=(
            last.recursive_tree_depth
        ),
        final_active_context_size=(
            last.active_contribution_count
        ),
        sublinear_active_context=(
            active_exponent
            < 1.0
        ),
        sublinear_active_ancestry=(
            ancestry_exponent
            < 1.0
        ),
        metadata={
            "summary_method": "first_last_log_scaling",
            "checkpoint_count": len(
                checkpoints
            ),
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Full benchmark
# =============================================================================


def run_recursive_memory_scaling_benchmark(
    *,
    event_counts: Iterable[int] = DEFAULT_EVENT_COUNTS,
    query_offset: int = DEFAULT_QUERY_OFFSET,
    assignment_radius: float = DEFAULT_ASSIGNMENT_RADIUS,
    merge_radius: float = DEFAULT_RECURSIVE_MERGE_RADIUS,
    max_children: int = DEFAULT_MAX_CHILDREN,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> RecursiveMemoryScalingResult:
    counts = normalize_event_counts(
        event_counts
    )

    checkpoints = tuple(
        build_scaling_checkpoint(
            event_count=event_count,
            query_offset=query_offset,
            assignment_radius=assignment_radius,
            merge_radius=merge_radius,
            max_children=max_children,
            max_depth=max_depth,
        )
        for event_count
        in counts
    )

    summary = summarize_scaling(
        checkpoints
    )

    return RecursiveMemoryScalingResult(
        scenario_id=SCENARIO_ID,
        checkpoints=checkpoints,
        summary=summary,
        metadata={
            "benchmark_type": "recursive_memory_scaling",
            "checkpoint_count": len(
                checkpoints
            ),
            "recursive_tree_enabled": True,
            "deep_recursive_retrieval": True,
            "policy_override_enabled": False,
            "action_selected_by_memory": False,
            "cluster_assignment_uses_external_pattern_label": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Reporting helpers
# =============================================================================


def raw_record_series(
    result: RecursiveMemoryScalingResult,
) -> tuple[int, ...]:
    return tuple(
        item.raw_record_count
        for item in result.checkpoints
    )


def base_cluster_series(
    result: RecursiveMemoryScalingResult,
) -> tuple[int, ...]:
    return tuple(
        item.base_cluster_count
        for item in result.checkpoints
    )


def recursive_node_series(
    result: RecursiveMemoryScalingResult,
) -> tuple[int, ...]:
    return tuple(
        item.recursive_node_count
        for item in result.checkpoints
    )


def tree_depth_series(
    result: RecursiveMemoryScalingResult,
) -> tuple[int, ...]:
    return tuple(
        item.recursive_tree_depth
        for item in result.checkpoints
    )


def active_ancestry_series(
    result: RecursiveMemoryScalingResult,
) -> tuple[int, ...]:
    return tuple(
        item.active_ancestry_length
        for item in result.checkpoints
    )


def active_context_series(
    result: RecursiveMemoryScalingResult,
) -> tuple[int, ...]:
    return tuple(
        item.active_contribution_count
        for item in result.checkpoints
    )


def raw_to_active_context_series(
    result: RecursiveMemoryScalingResult,
) -> tuple[float, ...]:
    return tuple(
        item.raw_to_active_context_ratio
        for item in result.checkpoints
    )


__all__ = [
    "DEFAULT_EVENT_COUNTS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_QUERY_OFFSET",
    "RecursiveMemoryScalingCheckpoint",
    "RecursiveMemoryScalingError",
    "RecursiveMemoryScalingResult",
    "RecursiveMemoryScalingSummary",
    "SCENARIO_ID",
    "active_ancestry_series",
    "active_context_series",
    "base_cluster_series",
    "build_scaling_checkpoint",
    "normalize_event_counts",
    "raw_record_series",
    "raw_to_active_context_series",
    "recursive_node_series",
    "run_recursive_memory_scaling_benchmark",
    "summarize_scaling",
    "tree_depth_series",
]
