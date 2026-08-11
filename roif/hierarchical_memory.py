"""
ROIF Hierarchical Memory
========================

Purpose
-------
Provide a compact, auditable hierarchical index over controller-history
StructuralSignature records.

Input:
    ControllerHistoryRecord

Output:
    HierarchicalMemory
        -> MemoryCluster[]
            -> prototype signature vector
            -> local radius
            -> member record ids
            -> representative record ids
            -> anomaly record ids

Architectural boundaries
------------------------

    HierarchicalMemory != Controller Learning
    Cluster Assignment != PredictivePreload
    Prototype != Ground Truth
    Compression != Deletion of Anomalies

This module:
- groups similar controller-history records by StructuralSignature distance;
- maintains immutable memory snapshots;
- keeps compact cluster statistics;
- preserves representative and anomalous records;
- supports nearest-cluster lookup;
- supports recursive higher-order grouping later.

This module does NOT:
- alter PredictiveState;
- change candidate ranking;
- modify control policy;
- apply MemoryScar to predictions;
- mutate graph structure;
- delete source records automatically;
- infer biological memory.

The current implementation is intentionally conservative and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.history.controller_history_adapter import (
    ControllerHistoryRecord,
    controller_signature_distance,
    record_is_preload_free,
)


# =============================================================================
# Errors
# =============================================================================


class HierarchicalMemoryError(RuntimeError):
    """Base hierarchical-memory error."""


class InvalidMemoryClusterError(HierarchicalMemoryError):
    """Raised when a cluster violates structural invariants."""


class DuplicateMemoryRecordError(HierarchicalMemoryError):
    """Raised when a record is inserted twice."""


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


def _finite_non_negative(
    value: float,
    *,
    name: str,
) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )
    if value < 0.0:
        raise ValueError(
            f"{name} must be non-negative"
        )
    return value


def _euclidean(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise HierarchicalMemoryError(
            "signature vectors must have equal length"
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
    vectors: Sequence[
        Sequence[float]
    ],
) -> tuple[float, ...]:
    if not vectors:
        raise HierarchicalMemoryError(
            "cannot build centroid from zero vectors"
        )

    size = len(
        vectors[0]
    )

    if any(
        len(vector) != size
        for vector in vectors
    ):
        raise HierarchicalMemoryError(
            "all vectors must have equal length"
        )

    return tuple(
        sum(
            float(vector[index])
            for vector
            in vectors
        )
        / len(vectors)
        for index in range(size)
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class HierarchicalMemoryConfig:
    """
    Initial local-clustering configuration.

    assignment_radius:
        Maximum distance from cluster prototype for normal membership.

    anomaly_radius:
        Distance above which a record is preserved as an anomaly.

    max_representatives:
        Maximum number of representative record IDs retained per cluster.

    representative_refresh_interval:
        Recompute representatives every N accepted members.
    """

    assignment_radius: float = 0.10
    anomaly_radius: float = 0.50
    max_representatives: int = 5
    representative_refresh_interval: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assignment_radius",
            _finite_non_negative(
                self.assignment_radius,
                name="assignment_radius",
            ),
        )

        object.__setattr__(
            self,
            "anomaly_radius",
            _finite_non_negative(
                self.anomaly_radius,
                name="anomaly_radius",
            ),
        )

        if (
            self.anomaly_radius
            < self.assignment_radius
        ):
            raise HierarchicalMemoryError(
                "anomaly_radius must be >= assignment_radius"
            )

        if self.max_representatives <= 0:
            raise HierarchicalMemoryError(
                "max_representatives must be positive"
            )

        if self.representative_refresh_interval <= 0:
            raise HierarchicalMemoryError(
                "representative_refresh_interval must be positive"
            )


# =============================================================================
# Cluster
# =============================================================================


@dataclass(frozen=True, slots=True)
class MemoryCluster:
    cluster_id: str
    member_record_ids: tuple[str, ...]
    prototype_vector: tuple[float, ...]
    local_radius: float
    representative_record_ids: tuple[str, ...]
    anomaly_record_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.cluster_id:
            raise InvalidMemoryClusterError(
                "cluster_id must not be empty"
            )

        members = tuple(
            self.member_record_ids
        )

        if not members:
            raise InvalidMemoryClusterError(
                "cluster must contain at least one member"
            )

        if len(
            set(
                members
            )
        ) != len(
            members
        ):
            raise InvalidMemoryClusterError(
                "cluster member ids must be unique"
            )

        representatives = tuple(
            self.representative_record_ids
        )

        if not set(
            representatives
        ).issubset(
            set(
                members
            )
        ):
            raise InvalidMemoryClusterError(
                "representatives must be cluster members"
            )

        anomalies = tuple(
            self.anomaly_record_ids
        )

        if not set(
            anomalies
        ).issubset(
            set(
                members
            )
        ):
            raise InvalidMemoryClusterError(
                "anomalies must be cluster members"
            )

        object.__setattr__(
            self,
            "member_record_ids",
            members,
        )

        object.__setattr__(
            self,
            "prototype_vector",
            tuple(
                float(value)
                for value
                in self.prototype_vector
            ),
        )

        object.__setattr__(
            self,
            "local_radius",
            _finite_non_negative(
                self.local_radius,
                name="local_radius",
            ),
        )

        object.__setattr__(
            self,
            "representative_record_ids",
            representatives,
        )

        object.__setattr__(
            self,
            "anomaly_record_ids",
            anomalies,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )

    @property
    def size(self) -> int:
        return len(
            self.member_record_ids
        )


# =============================================================================
# Memory snapshot
# =============================================================================


@dataclass(frozen=True, slots=True)
class HierarchicalMemory:
    """
    Immutable hierarchical-memory snapshot.

    records:
        Source records remain available during the early implementation phase.
        Future compression can move old raw records to archival storage while
        preserving representatives and anomalies.

    clusters:
        First-level local clusters.

    The current version implements one clustering level. The API is designed
    so higher-order cluster trees can be introduced without changing record
    semantics.
    """

    records: tuple[
        ControllerHistoryRecord,
        ...,
    ] = ()

    clusters: tuple[
        MemoryCluster,
        ...,
    ] = ()

    config: HierarchicalMemoryConfig = field(
        default_factory=HierarchicalMemoryConfig
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        records = tuple(
            self.records
        )

        clusters = tuple(
            self.clusters
        )

        ids = tuple(
            record.trace_id
            for record
            in records
        )

        if len(
            set(
                ids
            )
        ) != len(
            ids
        ):
            raise DuplicateMemoryRecordError(
                "duplicate record trace_id in hierarchical memory"
            )

        if not all(
            record_is_preload_free(
                record
            )
            for record
            in records
        ):
            raise HierarchicalMemoryError(
                "hierarchical memory accepts only preload-free records"
            )

        object.__setattr__(
            self,
            "records",
            records,
        )

        object.__setattr__(
            self,
            "clusters",
            clusters,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )

    @property
    def record_count(self) -> int:
        return len(
            self.records
        )

    @property
    def cluster_count(self) -> int:
        return len(
            self.clusters
        )

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(
            record.trace_id
            for record
            in self.records
        )


# =============================================================================
# Record lookup
# =============================================================================


def record_by_id(
    memory: HierarchicalMemory,
    record_id: str,
) -> ControllerHistoryRecord:
    for record in memory.records:
        if record.trace_id == record_id:
            return record

    raise KeyError(
        record_id
    )


def cluster_by_id(
    memory: HierarchicalMemory,
    cluster_id: str,
) -> MemoryCluster:
    for cluster in memory.clusters:
        if cluster.cluster_id == cluster_id:
            return cluster

    raise KeyError(
        cluster_id
    )


# =============================================================================
# Cluster statistics
# =============================================================================


def _record_vector(
    record: ControllerHistoryRecord,
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value
        in record.signature.compact_vector()
    )


def _cluster_records(
    memory: HierarchicalMemory,
    cluster: MemoryCluster,
) -> tuple[
    ControllerHistoryRecord,
    ...,
]:
    return tuple(
        record_by_id(
            memory,
            record_id,
        )
        for record_id
        in cluster.member_record_ids
    )


def _prototype_for_records(
    records: Sequence[
        ControllerHistoryRecord
    ],
) -> tuple[float, ...]:
    return _centroid(
        tuple(
            _record_vector(
                record
            )
            for record
            in records
        )
    )


def _radius_for_records(
    records: Sequence[
        ControllerHistoryRecord
    ],
    prototype: Sequence[float],
) -> float:
    if not records:
        return 0.0

    return max(
        _euclidean(
            _record_vector(
                record
            ),
            prototype,
        )
        for record
        in records
    )


def _representatives_for_records(
    records: Sequence[
        ControllerHistoryRecord
    ],
    prototype: Sequence[float],
    *,
    limit: int,
) -> tuple[str, ...]:
    ranked = sorted(
        (
            (
                _euclidean(
                    _record_vector(
                        record
                    ),
                    prototype,
                ),
                record.trace_id,
            )
            for record
            in records
        ),
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return tuple(
        record_id
        for _distance, record_id
        in ranked[
            :limit
        ]
    )


# =============================================================================
# Nearest-cluster lookup
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClusterMatch:
    cluster_id: str
    distance: float
    within_assignment_radius: bool
    within_anomaly_radius: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distance",
            _finite_non_negative(
                self.distance,
                name="distance",
            ),
        )


def nearest_cluster(
    memory: HierarchicalMemory,
    record: ControllerHistoryRecord,
) -> ClusterMatch | None:
    if not memory.clusters:
        return None

    vector = _record_vector(
        record
    )

    ranked = sorted(
        (
            (
                _euclidean(
                    vector,
                    cluster.prototype_vector,
                ),
                cluster.cluster_id,
            )
            for cluster
            in memory.clusters
        ),
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    distance, cluster_id = ranked[
        0
    ]

    return ClusterMatch(
        cluster_id=cluster_id,
        distance=distance,
        within_assignment_radius=(
            distance
            <= memory.config.assignment_radius
        ),
        within_anomaly_radius=(
            distance
            <= memory.config.anomaly_radius
        ),
    )


# =============================================================================
# Cluster creation / update
# =============================================================================


def _new_cluster(
    record: ControllerHistoryRecord,
    *,
    index: int,
) -> MemoryCluster:
    vector = _record_vector(
        record
    )

    return MemoryCluster(
        cluster_id=f"cluster_{index:04d}",
        member_record_ids=(
            record.trace_id,
        ),
        prototype_vector=vector,
        local_radius=0.0,
        representative_record_ids=(
            record.trace_id,
        ),
        anomaly_record_ids=(),
        metadata={
            "cluster_level": 1,
            "prototype_type": "centroid",
            "learning_applied": False,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def _updated_cluster(
    memory: HierarchicalMemory,
    cluster: MemoryCluster,
    record: ControllerHistoryRecord,
    *,
    mark_anomaly: bool,
) -> MemoryCluster:
    existing = _cluster_records(
        memory,
        cluster,
    )

    records = (
        existing
        + (
            record,
        )
    )

    prototype = _prototype_for_records(
        records
    )

    radius = _radius_for_records(
        records,
        prototype,
    )

    representatives = _representatives_for_records(
        records,
        prototype,
        limit=memory.config.max_representatives,
    )

    anomaly_ids = tuple(
        dict.fromkeys(
            cluster.anomaly_record_ids
            + (
                (
                    record.trace_id,
                )
                if mark_anomaly
                else ()
            )
        )
    )

    return MemoryCluster(
        cluster_id=cluster.cluster_id,
        member_record_ids=(
            cluster.member_record_ids
            + (
                record.trace_id,
            )
        ),
        prototype_vector=prototype,
        local_radius=radius,
        representative_record_ids=representatives,
        anomaly_record_ids=anomaly_ids,
        metadata={
            **dict(
                cluster.metadata
            ),
            "cluster_level": 1,
            "prototype_type": "centroid",
            "learning_applied": False,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def append_controller_history_record(
    memory: HierarchicalMemory,
    record: ControllerHistoryRecord,
) -> HierarchicalMemory:
    """
    Add a record to a NEW HierarchicalMemory snapshot.

    Rules:
    - duplicate trace_id -> error;
    - no clusters -> create first cluster;
    - nearest distance <= assignment_radius -> normal cluster member;
    - assignment_radius < distance <= anomaly_radius -> preserve in nearest
      cluster but mark as anomaly;
    - distance > anomaly_radius -> create a new cluster.

    No PredictivePreload or policy learning is performed.
    """

    if record.trace_id in memory.record_ids:
        raise DuplicateMemoryRecordError(
            f"duplicate record: {record.trace_id}"
        )

    if not record_is_preload_free(
        record
    ):
        raise HierarchicalMemoryError(
            "cannot store record that already applied PredictivePreload"
        )

    updated_records = (
        memory.records
        + (
            record,
        )
    )

    if not memory.clusters:
        updated_clusters = (
            _new_cluster(
                record,
                index=1,
            ),
        )

    else:
        match = nearest_cluster(
            memory,
            record,
        )

        assert match is not None

        if not match.within_anomaly_radius:
            updated_clusters = (
                memory.clusters
                + (
                    _new_cluster(
                        record,
                        index=(
                            len(
                                memory.clusters
                            )
                            + 1
                        ),
                    ),
                )
            )

        else:
            updated = []

            for cluster in memory.clusters:
                if (
                    cluster.cluster_id
                    == match.cluster_id
                ):
                    updated.append(
                        _updated_cluster(
                            memory,
                            cluster,
                            record,
                            mark_anomaly=(
                                not match.within_assignment_radius
                            ),
                        )
                    )
                else:
                    updated.append(
                        cluster
                    )

            updated_clusters = tuple(
                updated
            )

    return HierarchicalMemory(
        records=updated_records,
        clusters=updated_clusters,
        config=memory.config,
        metadata={
            **dict(
                memory.metadata
            ),
            "hierarchical_memory_snapshot": True,
            "record_count": len(
                updated_records
            ),
            "cluster_count": len(
                updated_clusters
            ),
            "learning_applied": False,
            "predictive_preload_applied": False,
            "predictive_state_mutated": False,
            "policy_modified": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def append_controller_history_records(
    memory: HierarchicalMemory,
    records: Sequence[
        ControllerHistoryRecord
    ],
) -> HierarchicalMemory:
    result = memory

    for record in records:
        result = append_controller_history_record(
            result,
            record,
        )

    return result


# =============================================================================
# Compression statistics
# =============================================================================


@dataclass(frozen=True, slots=True)
class MemoryCompressionStats:
    raw_record_count: int
    cluster_count: int
    representative_count: int
    anomaly_count: int
    effective_retained_count: int
    compression_ratio: float

    def __post_init__(self) -> None:
        if self.raw_record_count < 0:
            raise HierarchicalMemoryError(
                "raw_record_count must be non-negative"
            )

        object.__setattr__(
            self,
            "compression_ratio",
            _finite_non_negative(
                self.compression_ratio,
                name="compression_ratio",
            ),
        )


def memory_compression_stats(
    memory: HierarchicalMemory,
) -> MemoryCompressionStats:
    representative_ids = {
        record_id
        for cluster
        in memory.clusters
        for record_id
        in cluster.representative_record_ids
    }

    anomaly_ids = {
        record_id
        for cluster
        in memory.clusters
        for record_id
        in cluster.anomaly_record_ids
    }

    effective = len(
        representative_ids
        | anomaly_ids
    )

    if memory.record_count == 0:
        ratio = 1.0
    elif effective == 0:
        ratio = float(
            memory.record_count
        )
    else:
        ratio = (
            memory.record_count
            / effective
        )

    return MemoryCompressionStats(
        raw_record_count=memory.record_count,
        cluster_count=memory.cluster_count,
        representative_count=len(
            representative_ids
        ),
        anomaly_count=len(
            anomaly_ids
        ),
        effective_retained_count=effective,
        compression_ratio=ratio,
    )


# =============================================================================
# Audit helpers
# =============================================================================


def hierarchical_memory_is_preload_free(
    memory: HierarchicalMemory,
) -> bool:
    return (
        memory.metadata.get(
            "predictive_preload_applied",
            False,
        )
        is False
        and memory.metadata.get(
            "predictive_state_mutated",
            False,
        )
        is False
        and memory.metadata.get(
            "policy_modified",
            False,
        )
        is False
        and memory.metadata.get(
            "graph_mutated",
            False,
        )
        is False
        and all(
            cluster.metadata.get(
                "predictive_preload_applied"
            )
            is False
            for cluster
            in memory.clusters
        )
    )


# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "ClusterMatch",
    "DuplicateMemoryRecordError",
    "HierarchicalMemory",
    "HierarchicalMemoryConfig",
    "HierarchicalMemoryError",
    "InvalidMemoryClusterError",
    "MemoryCluster",
    "MemoryCompressionStats",
    "append_controller_history_record",
    "append_controller_history_records",
    "cluster_by_id",
    "hierarchical_memory_is_preload_free",
    "memory_compression_stats",
    "nearest_cluster",
    "record_by_id",
]
