"""
ROIF External-Domain Validation
Scenario 04C — Long-Memory Compression Benchmark

Purpose
-------
Stress-test the controller-history / hierarchical-memory stack on a long,
repeated event stream.

Pipeline:

    repeated sailing event
        ->
    predictive control
        ->
    ExperienceTrace
        ->
    ControllerHistoryRecord
        ->
    StructuralSignature
        ->
    HierarchicalMemory

04C measures:
- cluster growth;
- compression ratio;
- representatives retained;
- anomalies retained;
- same-pattern cohesion;
- cross-pattern separation;
- deterministic long-run behavior.

Architectural boundaries
------------------------

    Compression != Learning
    Compression != PredictivePreload
    Cluster Label != External Ground Truth Input
    Anomaly != Deletion Candidate

The pattern labels used by this validation module are evaluator-side metadata
for reporting only. Cluster assignment itself is performed exclusively from
StructuralSignature distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.hierarchical_memory import (
    HierarchicalMemory,
    HierarchicalMemoryConfig,
    MemoryCompressionStats,
    append_controller_history_record,
    hierarchical_memory_is_preload_free,
    memory_compression_stats,
)
from roif.history.controller_history_adapter import (
    ControllerHistoryRecord,
    adapt_experience_trace,
    controller_signature_distance,
)
from validation.sailing.sailing_yacht_crew_repeated_event_benchmark import (
    SailingDisturbanceEvent,
    run_repeated_event_episode,
)
from validation.sailing.sailing_yacht_crew_predictive_control import (
    build_case,
    build_predictive_control_config,
)


SCENARIO_ID = "sailing_04C_long_memory_compression"
DEFAULT_EVENT_COUNT = 1000
DEFAULT_SEQUENCE_ID = "04C_long_memory_sequence"


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class LongMemoryBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LongMemoryEventSpec:
    pattern_family: str
    wind: float
    wave: float
    course_bias: float
    heel_bias: float

    def __post_init__(self) -> None:
        if not self.pattern_family:
            raise LongMemoryBenchmarkError(
                "pattern_family must not be empty"
            )


@dataclass(frozen=True, slots=True)
class LongMemoryCheckpoint:
    event_count: int
    cluster_count: int
    representative_count: int
    anomaly_count: int
    effective_retained_count: int
    compression_ratio: float

    def __post_init__(self) -> None:
        if self.event_count <= 0:
            raise LongMemoryBenchmarkError(
                "event_count must be positive"
            )


@dataclass(frozen=True, slots=True)
class LongMemoryBenchmarkResult:
    scenario_id: str
    event_count: int
    memory: HierarchicalMemory
    compression: MemoryCompressionStats
    checkpoints: tuple[LongMemoryCheckpoint, ...]
    family_counts: Mapping[str, int]
    mean_same_family_distance: float
    mean_cross_family_distance: float
    separation_ratio: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoints",
            tuple(self.checkpoints),
        )
        object.__setattr__(
            self,
            "family_counts",
            MappingProxyType(dict(self.family_counts)),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


# =============================================================================
# Deterministic event stream
# =============================================================================


PATTERN_LIBRARY: Mapping[str, LongMemoryEventSpec] = MappingProxyType(
    {
        "A": LongMemoryEventSpec(
            pattern_family="A",
            wind=0.28,
            wave=0.18,
            course_bias=0.08,
            heel_bias=0.04,
        ),
        "B": LongMemoryEventSpec(
            pattern_family="B",
            wind=0.42,
            wave=0.26,
            course_bias=-0.03,
            heel_bias=0.09,
        ),
        "C": LongMemoryEventSpec(
            pattern_family="C",
            wind=0.48,
            wave=0.31,
            course_bias=0.12,
            heel_bias=0.07,
        ),
        "D": LongMemoryEventSpec(
            pattern_family="D",
            wind=0.18,
            wave=0.38,
            course_bias=-0.11,
            heel_bias=-0.06,
        ),
    }
)


def _deterministic_variation(
    index: int,
    scale: float,
) -> float:
    """
    Small bounded deterministic variation without random state.
    """
    cycle = (
        -2,
        -1,
        0,
        1,
        2,
        1,
        0,
        -1,
    )
    return cycle[
        index % len(cycle)
    ] * scale


def build_long_memory_event(
    index: int,
) -> SailingDisturbanceEvent:
    """
    Event distribution:
      A: common
      B: common
      C: less common
      D: rare / exceptional

    Every 97th event introduces D.
    Otherwise an A/B/A/C/A/B cycle is used.

    Small deterministic variations ensure clusters must tolerate realistic
    within-pattern variation instead of exact duplicates.
    """

    if index <= 0:
        raise LongMemoryBenchmarkError(
            "index must be positive"
        )

    if index % 97 == 0:
        family = "D"
    else:
        cycle = (
            "A",
            "B",
            "A",
            "C",
            "A",
            "B",
        )
        family = cycle[
            (index - 1) % len(cycle)
        ]

    spec = PATTERN_LIBRARY[
        family
    ]

    wind_variation = _deterministic_variation(
        index,
        0.002,
    )
    wave_variation = _deterministic_variation(
        index + 3,
        0.0015,
    )

    return SailingDisturbanceEvent(
        event_id=f"04C_event_{index:05d}_{family}",
        wind=max(
            0.0,
            spec.wind + wind_variation,
        ),
        wave=max(
            0.0,
            spec.wave + wave_variation,
        ),
        observation_bias_course=spec.course_bias,
        observation_bias_heel=spec.heel_bias,
        metadata={
            "scenario_id": SCENARIO_ID,
            "pattern_family": family,
            "evaluator_only_pattern_label": True,
            "external_expected_label_used": False,
        },
    )


def build_long_memory_sequence(
    event_count: int = DEFAULT_EVENT_COUNT,
) -> tuple[SailingDisturbanceEvent, ...]:
    if event_count <= 0:
        raise LongMemoryBenchmarkError(
            "event_count must be positive"
        )

    return tuple(
        build_long_memory_event(index)
        for index in range(
            1,
            event_count + 1,
        )
    )


# =============================================================================
# Adapter path
# =============================================================================


def event_to_controller_history_record(
    *,
    event: SailingDisturbanceEvent,
    episode_index: int,
    sequence_id: str = DEFAULT_SEQUENCE_ID,
) -> ControllerHistoryRecord:
    """
    Build a real predictive-control episode and adapt it to History.

    IMPORTANT:
    pattern_family is passed only into record reporting metadata/source id.
    HierarchicalMemory never reads pattern_family when assigning clusters.
    """

    case = build_case()
    config = build_predictive_control_config()

    episode = run_repeated_event_episode(
        case=case,
        event=event,
        episode_index=episode_index,
        sequence_id=sequence_id,
        config=config,
    )

    return adapt_experience_trace(
        episode.trace,
        pattern_id=event.metadata[
            "pattern_family"
        ],
    )


# =============================================================================
# Distance diagnostics
# =============================================================================


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(
        float(value)
        for value in values
    ) / len(values)


def _family_distance_metrics(
    records: Sequence[ControllerHistoryRecord],
) -> tuple[float, float]:
    """
    Use a bounded diagnostic sample to avoid O(N^2) long-run cost.
    """

    sample = tuple(
        records[
            : min(
                len(records),
                120,
            )
        ]
    )

    same = []
    cross = []

    for left_index, left in enumerate(sample):
        for right in sample[
            left_index + 1:
        ]:
            distance = controller_signature_distance(
                left,
                right,
            )

            if (
                left.pattern_id
                == right.pattern_id
            ):
                same.append(
                    distance
                )
            else:
                cross.append(
                    distance
                )

    return (
        _mean(same),
        _mean(cross),
    )


# =============================================================================
# Benchmark
# =============================================================================


def run_long_memory_compression_benchmark(
    *,
    event_count: int = DEFAULT_EVENT_COUNT,
    config: HierarchicalMemoryConfig | None = None,
    checkpoint_interval: int = 100,
) -> LongMemoryBenchmarkResult:
    if event_count <= 0:
        raise LongMemoryBenchmarkError(
            "event_count must be positive"
        )

    if checkpoint_interval <= 0:
        raise LongMemoryBenchmarkError(
            "checkpoint_interval must be positive"
        )

    config = (
        config
        or HierarchicalMemoryConfig(
            assignment_radius=0.10,
            anomaly_radius=0.50,
            max_representatives=5,
            representative_refresh_interval=10,
        )
    )

    memory = HierarchicalMemory(
        config=config,
        metadata={
            "scenario_id": SCENARIO_ID,
            "learning_applied": False,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )

    family_counts: dict[str, int] = {}
    records: list[ControllerHistoryRecord] = []
    checkpoints: list[LongMemoryCheckpoint] = []

    for index, event in enumerate(
        build_long_memory_sequence(
            event_count
        ),
        start=1,
    ):
        family = event.metadata[
            "pattern_family"
        ]

        family_counts[
            family
        ] = (
            family_counts.get(
                family,
                0,
            )
            + 1
        )

        record = event_to_controller_history_record(
            event=event,
            episode_index=index,
        )

        records.append(
            record
        )

        memory = append_controller_history_record(
            memory,
            record,
        )

        if (
            index % checkpoint_interval == 0
            or index == event_count
        ):
            stats = memory_compression_stats(
                memory
            )

            checkpoints.append(
                LongMemoryCheckpoint(
                    event_count=index,
                    cluster_count=stats.cluster_count,
                    representative_count=stats.representative_count,
                    anomaly_count=stats.anomaly_count,
                    effective_retained_count=stats.effective_retained_count,
                    compression_ratio=stats.compression_ratio,
                )
            )

    compression = memory_compression_stats(
        memory
    )

    same_distance, cross_distance = (
        _family_distance_metrics(
            records
        )
    )

    separation_ratio = (
        0.0
        if same_distance <= 1e-12
        else cross_distance
        / same_distance
    )

    if not hierarchical_memory_is_preload_free(
        memory
    ):
        raise LongMemoryBenchmarkError(
            "04C memory unexpectedly applied PredictivePreload"
        )

    return LongMemoryBenchmarkResult(
        scenario_id=SCENARIO_ID,
        event_count=event_count,
        memory=memory,
        compression=compression,
        checkpoints=tuple(
            checkpoints
        ),
        family_counts=family_counts,
        mean_same_family_distance=same_distance,
        mean_cross_family_distance=cross_distance,
        separation_ratio=separation_ratio,
        metadata={
            "benchmark_type": "long_memory_compression",
            "cluster_assignment_uses_external_pattern_label": False,
            "pattern_labels_evaluator_only": True,
            "learning_enabled": False,
            "predictive_preload_enabled": False,
            "raw_records_preserved_in_current_phase": True,
            "representatives_preserved": True,
            "anomalies_preserved": True,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Reporting helpers
# =============================================================================


def cluster_sizes(
    result: LongMemoryBenchmarkResult,
) -> tuple[int, ...]:
    return tuple(
        cluster.size
        for cluster
        in result.memory.clusters
    )


def cluster_anomaly_counts(
    result: LongMemoryBenchmarkResult,
) -> tuple[int, ...]:
    return tuple(
        len(
            cluster.anomaly_record_ids
        )
        for cluster
        in result.memory.clusters
    )


def checkpoint_compression_series(
    result: LongMemoryBenchmarkResult,
) -> tuple[tuple[int, float], ...]:
    return tuple(
        (
            checkpoint.event_count,
            checkpoint.compression_ratio,
        )
        for checkpoint
        in result.checkpoints
    )


__all__ = [
    "DEFAULT_EVENT_COUNT",
    "DEFAULT_SEQUENCE_ID",
    "LongMemoryBenchmarkError",
    "LongMemoryBenchmarkResult",
    "LongMemoryCheckpoint",
    "LongMemoryEventSpec",
    "PATTERN_LIBRARY",
    "SCENARIO_ID",
    "build_long_memory_event",
    "build_long_memory_sequence",
    "checkpoint_compression_series",
    "cluster_anomaly_counts",
    "cluster_sizes",
    "event_to_controller_history_record",
    "run_long_memory_compression_benchmark",
]
