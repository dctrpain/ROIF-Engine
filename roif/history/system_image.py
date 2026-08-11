"""
ROIF Engine
System Image Core

Purpose
-------
Represent a time-dependent system image reconstructed from heterogeneous
measurements, prestress state, adaptive connection state, and historical traces.

This module does NOT collapse heterogeneous quantities into one scalar.
It preserves layers explicitly and produces a deterministic composite image.

Concept
-------
A system image at time t is represented as:

    I_t = R(
        measures_t,
        prestress_t,
        adaptive_connections_t,
        history_t,
        context_t
    )

The image is not assumed to be additive or multiplicative.
It is a structured state container with deterministic signatures and
cross-layer summaries.

This layer does NOT:
- infer diagnosis
- infer biological truth
- learn hidden couplings
- mutate memory
- modify topology
- select action/policy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.adaptive_connection import (
    AdaptiveConnectionState,
    effective_transfer_gain,
)
from roif.history.prestress_redistribution import (
    PrestressNodeState,
)


SCHEMA_VERSION = "system_image_v1"


class SystemImageError(ValueError):
    """Raised when system-image inputs are invalid."""


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _validate_id(
    name: str,
    value: str,
) -> str:
    cleaned = str(value).strip()

    if not cleaned:
        raise SystemImageError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    number = float(value)

    if not isfinite(number):
        raise SystemImageError(
            f"{name} must be finite"
        )

    return number


@dataclass(frozen=True, slots=True)
class SystemMeasure:
    """
    One heterogeneous observation of the system.

    measure_type
        Semantic type, e.g. mass, volume, temperature, current, pressure.

    value
        Numeric value in its own native unit.

    unit
        Native unit string.

    domain
        Optional measurement domain/group.

    normalized_value
        Optional dimensionless representation for comparisons.
        This is NOT automatically derived because different measures require
        different normalization rules.
    """

    measure_id: str
    measure_type: str
    value: float
    unit: str

    domain: str = "unspecified"
    normalized_value: float | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measure_id",
            _validate_id(
                "measure_id",
                self.measure_id,
            ),
        )

        object.__setattr__(
            self,
            "measure_type",
            _validate_id(
                "measure_type",
                self.measure_type,
            ),
        )

        object.__setattr__(
            self,
            "unit",
            _validate_id(
                "unit",
                self.unit,
            ),
        )

        object.__setattr__(
            self,
            "domain",
            _validate_id(
                "domain",
                self.domain,
            ),
        )

        object.__setattr__(
            self,
            "value",
            _validate_finite(
                "value",
                self.value,
            ),
        )

        if self.normalized_value is not None:
            object.__setattr__(
                self,
                "normalized_value",
                _validate_finite(
                    "normalized_value",
                    self.normalized_value,
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
class HistoricalTrace:
    """
    Persistent historical mark included in the current system image.

    magnitude
        Dimensionless trace magnitude supplied by the caller.

    persistence
        [0, 1] indicator of how much of the trace remains relevant.

    relation_count
        Number of explicit relations this trace has to other traces/entities.
    """

    trace_id: str
    source_event_id: str
    trace_type: str

    magnitude: float
    persistence: float = 1.0
    relation_count: int = 0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trace_id",
            _validate_id(
                "trace_id",
                self.trace_id,
            ),
        )

        object.__setattr__(
            self,
            "source_event_id",
            _validate_id(
                "source_event_id",
                self.source_event_id,
            ),
        )

        object.__setattr__(
            self,
            "trace_type",
            _validate_id(
                "trace_type",
                self.trace_type,
            ),
        )

        magnitude = _validate_finite(
            "magnitude",
            self.magnitude,
        )

        persistence = _validate_finite(
            "persistence",
            self.persistence,
        )

        if magnitude < 0.0:
            raise SystemImageError(
                "magnitude must be nonnegative"
            )

        if not (
            0.0
            <= persistence
            <= 1.0
        ):
            raise SystemImageError(
                "persistence must lie in [0, 1]"
            )

        relation_count = int(
            self.relation_count
        )

        if relation_count < 0:
            raise SystemImageError(
                "relation_count must be nonnegative"
            )

        object.__setattr__(
            self,
            "magnitude",
            magnitude,
        )

        object.__setattr__(
            self,
            "persistence",
            persistence,
        )

        object.__setattr__(
            self,
            "relation_count",
            relation_count,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class SystemImageContext:
    context_id: str
    timestamp_label: str
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_id",
            _validate_id(
                "context_id",
                self.context_id,
            ),
        )

        object.__setattr__(
            self,
            "timestamp_label",
            _validate_id(
                "timestamp_label",
                self.timestamp_label,
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
class SystemImageSummary:
    measure_count: int
    prestress_node_count: int
    adaptive_connection_count: int
    trace_count: int

    normalized_measure_rms: float | None
    prestress_rms: float
    mean_effective_transfer_gain: float | None
    trace_load: float
    trace_relation_total: int

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "measure_count",
            "prestress_node_count",
            "adaptive_connection_count",
            "trace_count",
            "trace_relation_total",
        ):
            value = int(
                getattr(
                    self,
                    name,
                )
            )

            if value < 0:
                raise SystemImageError(
                    f"{name} must be nonnegative"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        if self.normalized_measure_rms is not None:
            normalized_measure_rms = _validate_finite(
                "normalized_measure_rms",
                self.normalized_measure_rms,
            )

            if normalized_measure_rms < 0.0:
                raise SystemImageError(
                    "normalized_measure_rms must be nonnegative"
                )

            object.__setattr__(
                self,
                "normalized_measure_rms",
                normalized_measure_rms,
            )

        for name in (
            "prestress_rms",
            "trace_load",
        ):
            value = _validate_finite(
                name,
                getattr(
                    self,
                    name,
                ),
            )

            if value < 0.0:
                raise SystemImageError(
                    f"{name} must be nonnegative"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        if self.mean_effective_transfer_gain is not None:
            object.__setattr__(
                self,
                "mean_effective_transfer_gain",
                _validate_finite(
                    "mean_effective_transfer_gain",
                    self.mean_effective_transfer_gain,
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
class SystemImage:
    image_id: str
    revision: int

    measures: tuple[SystemMeasure, ...]
    prestress_nodes: tuple[PrestressNodeState, ...]
    adaptive_connections: tuple[AdaptiveConnectionState, ...]
    historical_traces: tuple[HistoricalTrace, ...]

    context: SystemImageContext
    summary: SystemImageSummary

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_id",
            _validate_id(
                "image_id",
                self.image_id,
            ),
        )

        revision = int(
            self.revision
        )

        if revision < 0:
            raise SystemImageError(
                "revision must be nonnegative"
            )

        object.__setattr__(
            self,
            "revision",
            revision,
        )

        object.__setattr__(
            self,
            "measures",
            tuple(
                self.measures
            ),
        )

        object.__setattr__(
            self,
            "prestress_nodes",
            tuple(
                self.prestress_nodes
            ),
        )

        object.__setattr__(
            self,
            "adaptive_connections",
            tuple(
                self.adaptive_connections
            ),
        )

        object.__setattr__(
            self,
            "historical_traces",
            tuple(
                self.historical_traces
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _ensure_unique_ids(
    items: Sequence[Any],
    attr_name: str,
    label: str,
) -> None:
    seen = set()

    for item in items:
        value = getattr(
            item,
            attr_name,
        )

        if value in seen:
            raise SystemImageError(
                f"duplicate {label}: {value}"
            )

        seen.add(
            value
        )


def _rms(
    values: Sequence[float],
) -> float:
    if not values:
        return 0.0

    return sqrt(
        sum(
            value * value
            for value in values
        )
        / len(
            values
        )
    )


def summarize_system_image_layers(
    *,
    measures: Sequence[SystemMeasure],
    prestress_nodes: Sequence[PrestressNodeState],
    adaptive_connections: Sequence[AdaptiveConnectionState],
    historical_traces: Sequence[HistoricalTrace],
) -> SystemImageSummary:
    normalized_values = [
        measure.normalized_value
        for measure in measures
        if measure.normalized_value is not None
    ]

    normalized_measure_rms = (
        _rms(
            normalized_values
        )
        if normalized_values
        else None
    )

    prestress_rms = _rms(
        [
            node.prestress
            for node in prestress_nodes
        ]
    )

    transfer_gains = [
        effective_transfer_gain(
            connection
        )
        for connection in adaptive_connections
    ]

    mean_effective_transfer_gain = (
        sum(
            transfer_gains
        )
        / len(
            transfer_gains
        )
        if transfer_gains
        else None
    )

    trace_load = sum(
        trace.magnitude
        * trace.persistence
        for trace in historical_traces
    )

    trace_relation_total = sum(
        trace.relation_count
        for trace in historical_traces
    )

    return SystemImageSummary(
        measure_count=len(
            measures
        ),
        prestress_node_count=len(
            prestress_nodes
        ),
        adaptive_connection_count=len(
            adaptive_connections
        ),
        trace_count=len(
            historical_traces
        ),
        normalized_measure_rms=normalized_measure_rms,
        prestress_rms=prestress_rms,
        mean_effective_transfer_gain=mean_effective_transfer_gain,
        trace_load=trace_load,
        trace_relation_total=trace_relation_total,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "summary_mode": "layer_preserving",
        },
    )


def build_system_image(
    *,
    image_id: str,
    revision: int,
    measures: Iterable[SystemMeasure],
    prestress_nodes: Iterable[PrestressNodeState],
    adaptive_connections: Iterable[AdaptiveConnectionState],
    historical_traces: Iterable[HistoricalTrace],
    context: SystemImageContext,
    metadata: Mapping[str, Any] | None = None,
) -> SystemImage:
    """
    Build one immutable system image.

    Heterogeneous measures remain heterogeneous.
    No raw values with incompatible units are summed.
    """
    image_id = _validate_id(
        "image_id",
        image_id,
    )

    measure_items = tuple(
        measures
    )

    prestress_items = tuple(
        prestress_nodes
    )

    adaptive_items = tuple(
        adaptive_connections
    )

    trace_items = tuple(
        historical_traces
    )

    _ensure_unique_ids(
        measure_items,
        "measure_id",
        "measure_id",
    )

    _ensure_unique_ids(
        prestress_items,
        "node_id",
        "prestress node_id",
    )

    _ensure_unique_ids(
        adaptive_items,
        "connection_id",
        "adaptive connection_id",
    )

    _ensure_unique_ids(
        trace_items,
        "trace_id",
        "trace_id",
    )

    summary = summarize_system_image_layers(
        measures=measure_items,
        prestress_nodes=prestress_items,
        adaptive_connections=adaptive_items,
        historical_traces=trace_items,
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "system_image_mode": "structured_multilayer",
        "heterogeneous_measures_collapsed": False,
        "memory_mutated": False,
        "learning_applied": False,
        "topology_modified": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_truth_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return SystemImage(
        image_id=image_id,
        revision=revision,
        measures=measure_items,
        prestress_nodes=prestress_items,
        adaptive_connections=adaptive_items,
        historical_traces=trace_items,
        context=context,
        summary=summary,
        metadata=merged_metadata,
    )


def revise_system_image(
    *,
    source: SystemImage,
    target_image_id: str,
    measures: Iterable[SystemMeasure] | None = None,
    prestress_nodes: Iterable[PrestressNodeState] | None = None,
    adaptive_connections: Iterable[AdaptiveConnectionState] | None = None,
    historical_traces: Iterable[HistoricalTrace] | None = None,
    context: SystemImageContext | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SystemImage:
    """
    Create the next immutable image revision.

    Omitted layers are carried forward unchanged.
    """
    return build_system_image(
        image_id=target_image_id,
        revision=source.revision + 1,
        measures=(
            source.measures
            if measures is None
            else tuple(
                measures
            )
        ),
        prestress_nodes=(
            source.prestress_nodes
            if prestress_nodes is None
            else tuple(
                prestress_nodes
            )
        ),
        adaptive_connections=(
            source.adaptive_connections
            if adaptive_connections is None
            else tuple(
                adaptive_connections
            )
        ),
        historical_traces=(
            source.historical_traces
            if historical_traces is None
            else tuple(
                historical_traces
            )
        ),
        context=(
            source.context
            if context is None
            else context
        ),
        metadata=metadata,
    )


def measure_by_id(
    image: SystemImage,
    measure_id: str,
) -> SystemMeasure | None:
    for measure in image.measures:
        if measure.measure_id == measure_id:
            return measure

    return None


def prestress_node_by_id(
    image: SystemImage,
    node_id: str,
) -> PrestressNodeState | None:
    for node in image.prestress_nodes:
        if node.node_id == node_id:
            return node

    return None


def adaptive_connection_by_id(
    image: SystemImage,
    connection_id: str,
) -> AdaptiveConnectionState | None:
    for connection in image.adaptive_connections:
        if connection.connection_id == connection_id:
            return connection

    return None


def trace_by_id(
    image: SystemImage,
    trace_id: str,
) -> HistoricalTrace | None:
    for trace in image.historical_traces:
        if trace.trace_id == trace_id:
            return trace

    return None


def normalized_measure_vector(
    image: SystemImage,
) -> tuple[
    tuple[str, float],
    ...,
]:
    """
    Return only explicitly normalized measures.

    Native heterogeneous values are intentionally not collapsed.
    """
    return tuple(
        sorted(
            (
                measure.measure_id,
                measure.normalized_value,
            )
            for measure in image.measures
            if measure.normalized_value is not None
        )
    )


def system_image_signature(
    image: SystemImage,
) -> tuple[
    str,
    int,
    tuple[tuple[str, float | None], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float, float, int], ...],
]:
    return (
        image.image_id,
        image.revision,
        tuple(
            sorted(
                (
                    measure.measure_id,
                    measure.normalized_value,
                )
                for measure in image.measures
            )
        ),
        tuple(
            sorted(
                (
                    node.node_id,
                    node.prestress,
                )
                for node in image.prestress_nodes
            )
        ),
        tuple(
            sorted(
                (
                    connection.connection_id,
                    effective_transfer_gain(
                        connection
                    ),
                )
                for connection in image.adaptive_connections
            )
        ),
        tuple(
            sorted(
                (
                    trace.trace_id,
                    trace.magnitude,
                    trace.persistence,
                    trace.relation_count,
                )
                for trace in image.historical_traces
            )
        ),
    )


def system_image_is_policy_free(
    image: SystemImage,
) -> bool:
    return (
        image.metadata.get(
            "action_selected"
        )
        is False
        and image.metadata.get(
            "policy_modified"
        )
        is False
        and image.metadata.get(
            "learning_applied"
        )
        is False
        and image.metadata.get(
            "topology_modified"
        )
        is False
    )


__all__ = [
    "HistoricalTrace",
    "SCHEMA_VERSION",
    "SystemImage",
    "SystemImageContext",
    "SystemImageError",
    "SystemImageSummary",
    "SystemMeasure",
    "adaptive_connection_by_id",
    "build_system_image",
    "measure_by_id",
    "normalized_measure_vector",
    "prestress_node_by_id",
    "revise_system_image",
    "summarize_system_image_layers",
    "system_image_is_policy_free",
    "system_image_signature",
    "trace_by_id",
]
