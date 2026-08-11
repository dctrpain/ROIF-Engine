"""
ROIF External-Domain Validation
Scenario 04E — Nested / Matryoshka Memory Benchmark

Purpose
-------
Validate hierarchical memory influence across multiple nested levels:

    local MemoryScar
        -> parent MemoryScar
            -> grandparent MemoryScar

The benchmark measures:
- local-memory dominance;
- geometric hierarchy decay;
- parent-prior support when local memory is weak;
- stability under conflicting parent context;
- no direct action selection by memory;
- no policy override;
- deterministic preload behavior.

Architectural boundaries
------------------------

    Nested Memory != Action Selection
    Parent Prior != Local Override
    Higher-Order Context != Ground Truth
    PredictivePreload != Policy Mutation

Scenario structure
------------------

Case 1: ALIGNED hierarchy
    local, parent, grandparent biases point in the same direction.

Case 2: CONFLICTING parent
    local bias remains correct while parent/grandparent partially oppose it.

Case 3: WEAK local
    local contribution is weak; parent context is allowed to help.

The current scenario operates at the PredictivePreload layer and uses
deterministic MemoryScar objects. It does not yet perform automatic
HierarchicalMemory-to-MemoryScar retrieval; that remains a separate retrieval
adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from roif.controller_memory import (
    MemoryPattern,
    MemoryScar,
)
from roif.predictive_control import (
    PredictiveState,
)
from roif.predictive_preload import (
    MemoryPreloadContribution,
    PredictivePreload,
    PredictivePreloadConfig,
    apply_predictive_preload,
    hierarchy_levels_used,
)


SCENARIO_ID = "sailing_04E_nested_matryoshka_memory"


class NestedMemoryBenchmarkError(RuntimeError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


@dataclass(frozen=True, slots=True)
class NestedMemoryCaseResult:
    case_id: str
    preload: PredictivePreload
    local_weight: float
    parent_weight: float
    grandparent_weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class NestedMemoryBenchmarkResult:
    scenario_id: str
    aligned: NestedMemoryCaseResult
    conflicting_parent: NestedMemoryCaseResult
    weak_local: NestedMemoryCaseResult
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


# =============================================================================
# Deterministic state and scars
# =============================================================================


def build_reference_state() -> PredictiveState:
    return PredictiveState(
        values={
            "course_error": 1.0,
            "heel_error": 0.5,
        },
        target_values={
            "course_error": 0.0,
            "heel_error": 0.0,
        },
        timestamp=1.0,
        uncertainty=0.40,
        metadata={
            "scenario_id": SCENARIO_ID,
            "external_expected_label_used": False,
        },
    )


def build_memory_scar(
    *,
    scar_id: str,
    pattern_id: str,
    course_bias: float,
    heel_bias: float,
    recurrence_count: int,
    consistency: float,
    confidence: float,
) -> MemoryScar:
    return MemoryScar(
        scar_id=scar_id,
        pattern=MemoryPattern(
            pattern_id=pattern_id,
            metadata={
                "scenario_id": SCENARIO_ID,
                "hierarchical_memory": True,
            },
        ),
        trace_ids=tuple(
            f"{scar_id}_{index:03d}"
            for index in range(recurrence_count)
        ),
        bias_by_variable={
            "course_error": course_bias,
            "heel_error": heel_bias,
        },
        error_norm_mean=(
            course_bias ** 2
            + heel_bias ** 2
        ) ** 0.5,
        error_norm_last=(
            course_bias ** 2
            + heel_bias ** 2
        ) ** 0.5,
        recurrence_count=recurrence_count,
        consistency=consistency,
        confidence=confidence,
        improved_count=recurrence_count,
        unchanged_count=0,
        worsened_count=0,
        metadata={
            "scenario_id": SCENARIO_ID,
            "predictive_preload_applied": False,
            "predictive_state_mutated": False,
            "policy_modified": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def build_aligned_scars() -> tuple[MemoryScar, MemoryScar, MemoryScar]:
    local = build_memory_scar(
        scar_id="04E_local_A",
        pattern_id="A_local",
        course_bias=0.08,
        heel_bias=0.04,
        recurrence_count=6,
        consistency=1.0,
        confidence=0.75,
    )

    parent = build_memory_scar(
        scar_id="04E_parent_AB",
        pattern_id="AB_parent",
        course_bias=0.06,
        heel_bias=0.03,
        recurrence_count=20,
        consistency=0.95,
        confidence=0.85,
    )

    grandparent = build_memory_scar(
        scar_id="04E_grandparent_sailing",
        pattern_id="sailing_context",
        course_bias=0.04,
        heel_bias=0.02,
        recurrence_count=80,
        consistency=0.90,
        confidence=0.90,
    )

    return local, parent, grandparent


def build_conflicting_scars() -> tuple[MemoryScar, MemoryScar, MemoryScar]:
    local = build_memory_scar(
        scar_id="04E_local_A_conflict",
        pattern_id="A_local",
        course_bias=0.08,
        heel_bias=0.04,
        recurrence_count=6,
        consistency=1.0,
        confidence=0.75,
    )

    parent = build_memory_scar(
        scar_id="04E_parent_conflict",
        pattern_id="conflicting_parent",
        course_bias=-0.04,
        heel_bias=-0.02,
        recurrence_count=25,
        consistency=0.95,
        confidence=0.85,
    )

    grandparent = build_memory_scar(
        scar_id="04E_grandparent_conflict",
        pattern_id="conflicting_context",
        course_bias=-0.02,
        heel_bias=-0.01,
        recurrence_count=100,
        consistency=0.90,
        confidence=0.90,
    )

    return local, parent, grandparent


def build_weak_local_scars() -> tuple[MemoryScar, MemoryScar, MemoryScar]:
    local = build_memory_scar(
        scar_id="04E_weak_local",
        pattern_id="weak_local",
        course_bias=0.02,
        heel_bias=0.01,
        recurrence_count=2,
        consistency=0.85,
        confidence=0.30,
    )

    parent = build_memory_scar(
        scar_id="04E_strong_parent",
        pattern_id="parent_support",
        course_bias=0.06,
        heel_bias=0.03,
        recurrence_count=30,
        consistency=0.95,
        confidence=0.90,
    )

    grandparent = build_memory_scar(
        scar_id="04E_support_context",
        pattern_id="grandparent_support",
        course_bias=0.05,
        heel_bias=0.025,
        recurrence_count=100,
        consistency=0.90,
        confidence=0.95,
    )

    return local, parent, grandparent


def build_preload_config() -> PredictivePreloadConfig:
    return PredictivePreloadConfig(
        max_bias_fraction=0.75,
        uncertainty_reduction_scale=0.35,
        minimum_scar_confidence=0.25,
        minimum_scar_consistency=0.80,
        require_systematic_scar=True,
        local_level_decay=0.50,
    )


def _run_case(
    *,
    case_id: str,
    local: MemoryScar,
    parent: MemoryScar,
    grandparent: MemoryScar,
    local_explicit_weight: float = 1.0,
) -> NestedMemoryCaseResult:
    config = build_preload_config()

    contributions = (
        MemoryPreloadContribution(
            scar=local,
            hierarchy_level=0,
            explicit_weight=local_explicit_weight,
            source_cluster_id=f"{case_id}:local",
        ),
        MemoryPreloadContribution(
            scar=parent,
            hierarchy_level=1,
            explicit_weight=1.0,
            source_cluster_id=f"{case_id}:parent",
        ),
        MemoryPreloadContribution(
            scar=grandparent,
            hierarchy_level=2,
            explicit_weight=1.0,
            source_cluster_id=f"{case_id}:grandparent",
        ),
    )

    preload = apply_predictive_preload(
        build_reference_state(),
        contributions,
        config=config,
        metadata={
            "scenario_id": SCENARIO_ID,
            "case_id": case_id,
            "external_expected_label_used": False,
        },
    )

    weight_by_level = {
        item.hierarchy_level: item.effective_weight
        for item in preload.applied_contributions
    }

    return NestedMemoryCaseResult(
        case_id=case_id,
        preload=preload,
        local_weight=weight_by_level.get(0, 0.0),
        parent_weight=weight_by_level.get(1, 0.0),
        grandparent_weight=weight_by_level.get(2, 0.0),
        metadata={
            "scenario_id": SCENARIO_ID,
            "hierarchy_levels": hierarchy_levels_used(preload),
            "policy_override_enabled": False,
            "action_selected_by_memory": False,
            "external_expected_label_used": False,
        },
    )


def run_nested_memory_benchmark() -> NestedMemoryBenchmarkResult:
    aligned_local, aligned_parent, aligned_grandparent = (
        build_aligned_scars()
    )

    conflict_local, conflict_parent, conflict_grandparent = (
        build_conflicting_scars()
    )

    weak_local, weak_parent, weak_grandparent = (
        build_weak_local_scars()
    )

    aligned = _run_case(
        case_id="aligned",
        local=aligned_local,
        parent=aligned_parent,
        grandparent=aligned_grandparent,
    )

    conflicting = _run_case(
        case_id="conflicting_parent",
        local=conflict_local,
        parent=conflict_parent,
        grandparent=conflict_grandparent,
    )

    weak = _run_case(
        case_id="weak_local",
        local=weak_local,
        parent=weak_parent,
        grandparent=weak_grandparent,
        local_explicit_weight=0.35,
    )

    return NestedMemoryBenchmarkResult(
        scenario_id=SCENARIO_ID,
        aligned=aligned,
        conflicting_parent=conflicting,
        weak_local=weak,
        metadata={
            "benchmark_type": "nested_matryoshka_memory",
            "nested_memory_enabled": True,
            "policy_override_enabled": False,
            "action_selected_by_memory": False,
            "external_expected_label_used": False,
        },
    )


def case_bias(
    case: NestedMemoryCaseResult,
) -> tuple[float, float]:
    return (
        float(
            case.preload.applied_bias_by_variable.get(
                "course_error",
                0.0,
            )
        ),
        float(
            case.preload.applied_bias_by_variable.get(
                "heel_error",
                0.0,
            )
        ),
    )


def case_weight_series(
    case: NestedMemoryCaseResult,
) -> tuple[float, float, float]:
    return (
        case.local_weight,
        case.parent_weight,
        case.grandparent_weight,
    )


__all__ = [
    "NestedMemoryBenchmarkError",
    "NestedMemoryBenchmarkResult",
    "NestedMemoryCaseResult",
    "SCENARIO_ID",
    "build_aligned_scars",
    "build_conflicting_scars",
    "build_memory_scar",
    "build_preload_config",
    "build_reference_state",
    "build_weak_local_scars",
    "case_bias",
    "case_weight_series",
    "run_nested_memory_benchmark",
]
