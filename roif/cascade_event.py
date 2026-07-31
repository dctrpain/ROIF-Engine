"""
Discrete cascade-event model for the ROIF engine.

``PlaneKernel`` evolves continuous plane activation. This module converts
meaningful changes in that activation into immutable, serializable events.

The module deliberately separates three concerns:

1. ``CascadeEvent`` records one discrete event.
2. ``CascadeEventPolicy`` defines thresholds and classification rules.
3. ``detect_cascade_events`` converts a ``PlaneKernelResult`` into events.

No persistent history is maintained here. Storage, ordering across many
steps, causal graph reconstruction, and replay belong to later modules such
as ``history.py`` and ``cascade.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from types import MappingProxyType
from typing import Any, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .plane_kernel import PlaneKernelResult


class CascadeEventError(ValueError):
    """Raised when a cascade event or event policy is invalid."""


class CascadeEventKind(str, Enum):
    """Semantic category of a discrete cascade event."""

    ACTIVATION = "activation"
    DEACTIVATION = "deactivation"
    AMPLIFICATION = "amplification"
    ATTENUATION = "attenuation"
    SATURATION = "saturation"
    FLOOR_CONTACT = "floor_contact"
    HYSTERESIS_RETENTION = "hysteresis_retention"
    PROPAGATION = "propagation"


class CascadeEventSeverity(str, Enum):
    """Magnitude class of an event."""

    TRACE = "trace"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


def _normalize_non_empty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise CascadeEventError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise CascadeEventError(f"{name} cannot be empty.")

    return normalized


def _normalize_optional_string(
    value: str | None,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    return _normalize_non_empty_string(value, name=name)


def _finite_float(
    value: float,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    converted = float(value)

    if not math.isfinite(converted):
        raise CascadeEventError(f"{name} must be finite.")

    if minimum is not None and converted < minimum:
        raise CascadeEventError(
            f"{name} must be greater than or equal to {minimum}."
        )

    if maximum is not None and converted > maximum:
        raise CascadeEventError(
            f"{name} must be less than or equal to {maximum}."
        )

    return converted


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise CascadeEventError("metadata must be a mapping.")

    copied: dict[str, Any] = {}

    for key, value in metadata.items():
        normalized_key = _normalize_non_empty_string(
            key,
            name="metadata key",
        )
        copied[normalized_key] = value

    return MappingProxyType(copied)


def _coerce_kind(value: CascadeEventKind | str) -> CascadeEventKind:
    if isinstance(value, CascadeEventKind):
        return value

    try:
        return CascadeEventKind(value)
    except (TypeError, ValueError) as exc:
        raise CascadeEventError(
            f"Unknown cascade event kind {value!r}."
        ) from exc


def _coerce_severity(
    value: CascadeEventSeverity | str,
) -> CascadeEventSeverity:
    if isinstance(value, CascadeEventSeverity):
        return value

    try:
        return CascadeEventSeverity(value)
    except (TypeError, ValueError) as exc:
        raise CascadeEventError(
            f"Unknown cascade event severity {value!r}."
        ) from exc


def _stable_event_id(
    *,
    time: float,
    plane_id: str,
    kind: CascadeEventKind,
    activation_before: float,
    activation_after: float,
    source_plane_id: str | None,
    parent_event_id: str | None,
) -> str:
    payload = "|".join(
        (
            format(time, ".17g"),
            plane_id,
            kind.value,
            format(activation_before, ".17g"),
            format(activation_after, ".17g"),
            source_plane_id or "",
            parent_event_id or "",
        )
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"evt_{digest}"


@dataclass(frozen=True, slots=True)
class CascadeEvent:
    """Immutable record of one discrete change in a cascade."""

    time: float
    plane_id: str
    kind: CascadeEventKind | str
    severity: CascadeEventSeverity | str
    activation_before: float
    activation_after: float
    event_id: str | None = None
    source_plane_id: str | None = None
    parent_event_id: str | None = None
    clipped: bool = False
    retained_by_hysteresis: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _finite_float(self.time, name="time", minimum=0.0)
        plane_id = _normalize_non_empty_string(
            self.plane_id,
            name="plane_id",
        )
        kind = _coerce_kind(self.kind)
        severity = _coerce_severity(self.severity)
        activation_before = _finite_float(
            self.activation_before,
            name="activation_before",
            minimum=0.0,
            maximum=1.0,
        )
        activation_after = _finite_float(
            self.activation_after,
            name="activation_after",
            minimum=0.0,
            maximum=1.0,
        )
        source_plane_id = _normalize_optional_string(
            self.source_plane_id,
            name="source_plane_id",
        )
        parent_event_id = _normalize_optional_string(
            self.parent_event_id,
            name="parent_event_id",
        )

        if not isinstance(self.clipped, bool):
            raise CascadeEventError("clipped must be a bool.")

        if not isinstance(self.retained_by_hysteresis, bool):
            raise CascadeEventError(
                "retained_by_hysteresis must be a bool."
            )

        if source_plane_id == plane_id:
            source_plane_id = None

        event_id = self.event_id

        if event_id is None:
            event_id = _stable_event_id(
                time=time,
                plane_id=plane_id,
                kind=kind,
                activation_before=activation_before,
                activation_after=activation_after,
                source_plane_id=source_plane_id,
                parent_event_id=parent_event_id,
            )
        else:
            event_id = _normalize_non_empty_string(
                event_id,
                name="event_id",
            )

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(
            self,
            "activation_before",
            activation_before,
        )
        object.__setattr__(
            self,
            "activation_after",
            activation_after,
        )
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(
            self,
            "source_plane_id",
            source_plane_id,
        )
        object.__setattr__(
            self,
            "parent_event_id",
            parent_event_id,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def delta(self) -> float:
        return self.activation_after - self.activation_before

    @property
    def magnitude(self) -> float:
        return abs(self.delta)

    @property
    def is_increase(self) -> bool:
        return self.delta > 0.0

    @property
    def is_decrease(self) -> bool:
        return self.delta < 0.0

    @property
    def is_retained(self) -> bool:
        return self.retained_by_hysteresis

    @property
    def is_root(self) -> bool:
        return self.parent_event_id is None

    def with_parent(
        self,
        parent_event_id: str,
        *,
        source_plane_id: str | None = None,
    ) -> CascadeEvent:
        """Return an independent event linked to a causal parent."""

        return replace(
            self,
            event_id=None,
            parent_event_id=parent_event_id,
            source_plane_id=source_plane_id,
        )

    def with_metadata(self, **updates: Any) -> CascadeEvent:
        """Return an independent event with merged metadata."""

        merged = dict(self.metadata)
        merged.update(updates)
        return replace(self, metadata=merged)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "time": self.time,
            "plane_id": self.plane_id,
            "source_plane_id": self.source_plane_id,
            "parent_event_id": self.parent_event_id,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "activation_before": self.activation_before,
            "activation_after": self.activation_after,
            "delta": self.delta,
            "magnitude": self.magnitude,
            "clipped": self.clipped,
            "retained_by_hysteresis": (
                self.retained_by_hysteresis
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CascadeEventPolicy:
    """Thresholds used to convert continuous changes into events."""

    minimum_delta: float = 1e-12
    activation_threshold: float = 0.5
    deactivation_threshold: float = 0.5
    low_threshold: float = 0.05
    moderate_threshold: float = 0.15
    high_threshold: float = 0.35
    critical_threshold: float = 0.65
    emit_hysteresis_events: bool = True
    emit_clipping_events: bool = True

    def __post_init__(self) -> None:
        minimum_delta = _finite_float(
            self.minimum_delta,
            name="minimum_delta",
            minimum=0.0,
            maximum=1.0,
        )
        activation_threshold = _finite_float(
            self.activation_threshold,
            name="activation_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        deactivation_threshold = _finite_float(
            self.deactivation_threshold,
            name="deactivation_threshold",
            minimum=0.0,
            maximum=1.0,
        )

        thresholds = (
            _finite_float(
                self.low_threshold,
                name="low_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            _finite_float(
                self.moderate_threshold,
                name="moderate_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            _finite_float(
                self.high_threshold,
                name="high_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            _finite_float(
                self.critical_threshold,
                name="critical_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )

        if thresholds != tuple(sorted(thresholds)):
            raise CascadeEventError(
                "Severity thresholds must be non-decreasing."
            )

        if not isinstance(self.emit_hysteresis_events, bool):
            raise CascadeEventError(
                "emit_hysteresis_events must be a bool."
            )

        if not isinstance(self.emit_clipping_events, bool):
            raise CascadeEventError(
                "emit_clipping_events must be a bool."
            )

        object.__setattr__(self, "minimum_delta", minimum_delta)
        object.__setattr__(
            self,
            "activation_threshold",
            activation_threshold,
        )
        object.__setattr__(
            self,
            "deactivation_threshold",
            deactivation_threshold,
        )
        object.__setattr__(self, "low_threshold", thresholds[0])
        object.__setattr__(
            self,
            "moderate_threshold",
            thresholds[1],
        )
        object.__setattr__(self, "high_threshold", thresholds[2])
        object.__setattr__(
            self,
            "critical_threshold",
            thresholds[3],
        )

    def severity_for(
        self,
        magnitude: float,
    ) -> CascadeEventSeverity:
        value = _finite_float(
            magnitude,
            name="magnitude",
            minimum=0.0,
        )

        if value >= self.critical_threshold:
            return CascadeEventSeverity.CRITICAL
        if value >= self.high_threshold:
            return CascadeEventSeverity.HIGH
        if value >= self.moderate_threshold:
            return CascadeEventSeverity.MODERATE
        if value >= self.low_threshold:
            return CascadeEventSeverity.LOW
        return CascadeEventSeverity.TRACE

    def kind_for(
        self,
        *,
        activation_before: float,
        activation_after: float,
        clipped: bool = False,
        retained_by_hysteresis: bool = False,
        propagated: bool = False,
    ) -> CascadeEventKind | None:
        before = _finite_float(
            activation_before,
            name="activation_before",
            minimum=0.0,
            maximum=1.0,
        )
        after = _finite_float(
            activation_after,
            name="activation_after",
            minimum=0.0,
            maximum=1.0,
        )

        if retained_by_hysteresis:
            if self.emit_hysteresis_events:
                return CascadeEventKind.HYSTERESIS_RETENTION
            return None

        if clipped and self.emit_clipping_events:
            if after >= 1.0:
                return CascadeEventKind.SATURATION
            if after <= 0.0:
                return CascadeEventKind.FLOOR_CONTACT

        if (
            before < self.activation_threshold
            <= after
        ):
            return CascadeEventKind.ACTIVATION

        if (
            before >= self.deactivation_threshold
            > after
        ):
            return CascadeEventKind.DEACTIVATION

        delta = after - before

        magnitude = abs(delta)

        if magnitude == 0.0:
            return None

        if (
            magnitude < self.minimum_delta
            and not math.isclose(
                magnitude,
                self.minimum_delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return None

        if propagated:
            return CascadeEventKind.PROPAGATION

        if delta > 0.0:
            return CascadeEventKind.AMPLIFICATION

        return CascadeEventKind.ATTENUATION


@dataclass(frozen=True, slots=True)
class CascadeEventBatch:
    """Immutable collection of events produced by one kernel step."""

    time: float
    events: tuple[CascadeEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        time = _finite_float(self.time, name="time", minimum=0.0)

        normalized_events = tuple(self.events)

        for index, event in enumerate(normalized_events):
            if not isinstance(event, CascadeEvent):
                raise CascadeEventError(
                    f"events[{index}] must be a CascadeEvent."
                )

            if not math.isclose(
                event.time,
                time,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise CascadeEventError(
                    "All events in a batch must have the batch time."
                )

        event_ids = [event.event_id for event in normalized_events]

        if len(set(event_ids)) != len(event_ids):
            raise CascadeEventError(
                "Event IDs must be unique within a batch."
            )

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "events", normalized_events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __bool__(self) -> bool:
        return bool(self.events)

    @property
    def plane_ids(self) -> tuple[str, ...]:
        return tuple(event.plane_id for event in self.events)

    @property
    def critical_events(self) -> tuple[CascadeEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.severity is CascadeEventSeverity.CRITICAL
        )

    def for_plane(self, plane_id: str) -> tuple[CascadeEvent, ...]:
        normalized = _normalize_non_empty_string(
            plane_id,
            name="plane_id",
        )
        return tuple(
            event
            for event in self.events
            if event.plane_id == normalized
        )

    def by_kind(
        self,
        kind: CascadeEventKind | str,
    ) -> tuple[CascadeEvent, ...]:
        normalized = _coerce_kind(kind)
        return tuple(
            event
            for event in self.events
            if event.kind is normalized
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "event_count": len(self.events),
            "events": [event.as_dict() for event in self.events],
        }


def _dominant_source_plane(
    *,
    target_index: int,
    plane_ids: Sequence[str],
    activation_before: Sequence[float],
    coupling_matrix: Sequence[Sequence[float]] | None,
    minimum_influence: float,
) -> str | None:
    if coupling_matrix is None:
        return None

    size = len(plane_ids)

    if len(coupling_matrix) != size:
        raise CascadeEventError(
            "coupling_matrix row count must match plane_ids."
        )

    row = coupling_matrix[target_index]

    if len(row) != size:
        raise CascadeEventError(
            "coupling_matrix must be square and match plane_ids."
        )

    dominant_index: int | None = None
    dominant_magnitude = minimum_influence

    for source_index, (weight, activation) in enumerate(
        zip(row, activation_before, strict=True)
    ):
        if source_index == target_index:
            continue

        influence = abs(float(weight) * float(activation))

        if not math.isfinite(influence):
            raise CascadeEventError(
                "coupling_matrix and activation values must be finite."
            )

        if influence > dominant_magnitude:
            dominant_magnitude = influence
            dominant_index = source_index

    if dominant_index is None:
        return None

    return plane_ids[dominant_index]


def detect_cascade_events(
    result: PlaneKernelResult,
    *,
    policy: CascadeEventPolicy | None = None,
    coupling_matrix: Sequence[Sequence[float]] | None = None,
    source_event_ids: Mapping[str, str] | None = None,
    minimum_source_influence: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> CascadeEventBatch:
    """
    Convert one ``PlaneKernelResult`` into a batch of discrete events.

    ``coupling_matrix[i][j]`` follows the same convention as ``PlaneKernel``:
    source plane ``j`` influences target plane ``i``.
    """

    from .plane_kernel import PlaneKernelResult as RuntimePlaneKernelResult

    if not isinstance(result, RuntimePlaneKernelResult):
        raise CascadeEventError(
            "result must be a PlaneKernelResult."
        )

    if policy is None:
        policy = CascadeEventPolicy()

    if not isinstance(policy, CascadeEventPolicy):
        raise CascadeEventError(
            "policy must be a CascadeEventPolicy."
        )

    minimum_source_influence = _finite_float(
        minimum_source_influence,
        name="minimum_source_influence",
        minimum=0.0,
    )

    if source_event_ids is not None and not isinstance(
        source_event_ids,
        Mapping,
    ):
        raise CascadeEventError(
            "source_event_ids must be a mapping."
        )

    shared_metadata = dict(_freeze_metadata(metadata))
    events: list[CascadeEvent] = []

    for index, plane_id in enumerate(result.plane_ids):
        before = float(result.activation_before[index])
        after = float(result.activation_after[index])
        clipped = bool(result.clipped[index])
        retained = bool(result.retained_by_hysteresis[index])

        source_plane_id = _dominant_source_plane(
            target_index=index,
            plane_ids=result.plane_ids,
            activation_before=result.activation_before,
            coupling_matrix=coupling_matrix,
            minimum_influence=minimum_source_influence,
        )

        propagated = source_plane_id is not None
        kind = policy.kind_for(
            activation_before=before,
            activation_after=after,
            clipped=clipped,
            retained_by_hysteresis=retained,
            propagated=propagated,
        )

        if kind is None:
            continue

        parent_event_id: str | None = None

        if source_plane_id is not None and source_event_ids is not None:
            raw_parent = source_event_ids.get(source_plane_id)

            if raw_parent is not None:
                parent_event_id = _normalize_non_empty_string(
                    raw_parent,
                    name="source event ID",
                )

        event_metadata = dict(shared_metadata)
        event_metadata.update(
            {
                "interaction": float(result.interaction[index]),
                "rate": float(result.rate[index]),
                "dt": result.dt,
            }
        )

        event = CascadeEvent(
            time=result.time,
            plane_id=plane_id,
            source_plane_id=source_plane_id,
            parent_event_id=parent_event_id,
            kind=kind,
            severity=policy.severity_for(abs(after - before)),
            activation_before=before,
            activation_after=after,
            clipped=clipped,
            retained_by_hysteresis=retained,
            metadata=event_metadata,
        )
        events.append(event)

    return CascadeEventBatch(
        time=result.time,
        events=tuple(events),
    )


def link_event_chain(
    events: Iterable[CascadeEvent],
) -> tuple[CascadeEvent, ...]:
    """
    Link ordered events into a simple parent-child chain.

    The first event remains a root. Each following event receives the previous
    event as its parent and source plane.
    """

    normalized = tuple(events)

    for index, event in enumerate(normalized):
        if not isinstance(event, CascadeEvent):
            raise CascadeEventError(
                f"events[{index}] must be a CascadeEvent."
            )

    if not normalized:
        return ()

    linked: list[CascadeEvent] = [normalized[0]]

    for event in normalized[1:]:
        parent = linked[-1]
        linked.append(
            event.with_parent(
                parent.event_id,
                source_plane_id=parent.plane_id,
            )
        )

    return tuple(linked)


__all__ = [
    "CascadeEvent",
    "CascadeEventBatch",
    "CascadeEventError",
    "CascadeEventKind",
    "CascadeEventPolicy",
    "CascadeEventSeverity",
    "detect_cascade_events",
    "link_event_chain",
]
