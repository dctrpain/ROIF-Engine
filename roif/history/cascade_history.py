"""
Persistent cascade-event history for the ROIF engine.

``cascade_event.py`` creates immutable events for one PlaneKernel step.
This module stores those events across time, validates causal references,
supports deterministic querying, and reconstructs event ancestry.

The history itself is immutable: every write operation returns a new
``CascadeHistory`` instance.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
import math

from ..cascade_event import (
    CascadeEvent,
    CascadeEventBatch,
    CascadeEventKind,
    CascadeEventSeverity,
)


class CascadeHistoryError(ValueError):
    """Raised when a cascade history or history operation is invalid."""


def _normalize_non_empty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise CascadeHistoryError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise CascadeHistoryError(f"{name} cannot be empty.")

    return normalized


def _finite_non_negative(value: float, *, name: str) -> float:
    converted = float(value)

    if not math.isfinite(converted):
        raise CascadeHistoryError(f"{name} must be finite.")

    if converted < 0.0:
        raise CascadeHistoryError(
            f"{name} must be greater than or equal to 0.0."
        )

    return converted


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise CascadeHistoryError("metadata must be a mapping.")

    copied: dict[str, Any] = {}

    for key, value in metadata.items():
        normalized_key = _normalize_non_empty_string(
            key,
            name="metadata key",
        )
        copied[normalized_key] = value

    return MappingProxyType(copied)


def _coerce_kind(
    value: CascadeEventKind | str,
) -> CascadeEventKind:
    if isinstance(value, CascadeEventKind):
        return value

    try:
        return CascadeEventKind(value)
    except (TypeError, ValueError) as exc:
        raise CascadeHistoryError(
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
        raise CascadeHistoryError(
            f"Unknown cascade event severity {value!r}."
        ) from exc


def _severity_rank(severity: CascadeEventSeverity) -> int:
    order = {
        CascadeEventSeverity.TRACE: 0,
        CascadeEventSeverity.LOW: 1,
        CascadeEventSeverity.MODERATE: 2,
        CascadeEventSeverity.HIGH: 3,
        CascadeEventSeverity.CRITICAL: 4,
    }
    return order[severity]


@dataclass(frozen=True, slots=True)
class CascadeHistory:
    """
    Immutable ordered collection of cascade events.

    Events are stored in non-decreasing time order. Event IDs must be unique.
    Parent references may either point to an earlier event in this history or
    remain unresolved when ``allow_external_parents`` is enabled.
    """

    events: tuple[CascadeEvent, ...] = field(default_factory=tuple)
    allow_external_parents: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.allow_external_parents, bool):
            raise CascadeHistoryError(
                "allow_external_parents must be a bool."
            )

        normalized_events = tuple(self.events)
        event_ids: set[str] = set()
        last_time = -math.inf

        for index, event in enumerate(normalized_events):
            if not isinstance(event, CascadeEvent):
                raise CascadeHistoryError(
                    f"events[{index}] must be a CascadeEvent."
                )

            if event.time < last_time:
                raise CascadeHistoryError(
                    "Events must be ordered by non-decreasing time."
                )

            if event.event_id in event_ids:
                raise CascadeHistoryError(
                    f"Duplicate event ID {event.event_id!r}."
                )

            if (
                event.parent_event_id is not None
                and event.parent_event_id not in event_ids
                and not self.allow_external_parents
            ):
                raise CascadeHistoryError(
                    f"Event {event.event_id!r} references an "
                    f"unknown or future parent "
                    f"{event.parent_event_id!r}."
                )

            event_ids.add(event.event_id)
            last_time = event.time

        object.__setattr__(self, "events", normalized_events)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[CascadeEvent]:
        return iter(self.events)

    def __bool__(self) -> bool:
        return bool(self.events)

    def __contains__(self, event_id: object) -> bool:
        if not isinstance(event_id, str):
            return False

        normalized = event_id.strip()

        if not normalized:
            return False

        return any(
            event.event_id == normalized
            for event in self.events
        )

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)

    @property
    def plane_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []

        for event in self.events:
            if event.plane_id not in seen:
                seen.add(event.plane_id)
                ordered.append(event.plane_id)

        return tuple(ordered)

    @property
    def start_time(self) -> float | None:
        if not self.events:
            return None

        return self.events[0].time

    @property
    def end_time(self) -> float | None:
        if not self.events:
            return None

        return self.events[-1].time

    @property
    def duration(self) -> float:
        if len(self.events) < 2:
            return 0.0

        return self.events[-1].time - self.events[0].time

    @property
    def root_events(self) -> tuple[CascadeEvent, ...]:
        known_ids = set(self.event_ids)

        return tuple(
            event
            for event in self.events
            if event.parent_event_id is None
            or event.parent_event_id not in known_ids
        )

    @property
    def leaf_events(self) -> tuple[CascadeEvent, ...]:
        parent_ids = {
            event.parent_event_id
            for event in self.events
            if event.parent_event_id is not None
        }

        return tuple(
            event
            for event in self.events
            if event.event_id not in parent_ids
        )

    @property
    def critical_events(self) -> tuple[CascadeEvent, ...]:
        return self.at_or_above_severity(
            CascadeEventSeverity.CRITICAL
        )

    def get(self, event_id: str) -> CascadeEvent:
        normalized = _normalize_non_empty_string(
            event_id,
            name="event_id",
        )

        for event in self.events:
            if event.event_id == normalized:
                return event

        raise CascadeHistoryError(
            f"Unknown event ID {normalized!r}."
        )

    def index_of(self, event_id: str) -> int:
        normalized = _normalize_non_empty_string(
            event_id,
            name="event_id",
        )

        for index, event in enumerate(self.events):
            if event.event_id == normalized:
                return index

        raise CascadeHistoryError(
            f"Unknown event ID {normalized!r}."
        )

    def append(self, event: CascadeEvent) -> CascadeHistory:
        if not isinstance(event, CascadeEvent):
            raise CascadeHistoryError(
                "event must be a CascadeEvent."
            )

        return CascadeHistory(
            events=self.events + (event,),
            allow_external_parents=self.allow_external_parents,
            metadata=self.metadata,
        )

    def extend(
        self,
        events: Iterable[CascadeEvent],
    ) -> CascadeHistory:
        try:
            additions = tuple(events)
        except TypeError as exc:
            raise CascadeHistoryError(
                "events must be iterable."
            ) from exc

        return CascadeHistory(
            events=self.events + additions,
            allow_external_parents=self.allow_external_parents,
            metadata=self.metadata,
        )

    def append_batch(
        self,
        batch: CascadeEventBatch,
    ) -> CascadeHistory:
        if not isinstance(batch, CascadeEventBatch):
            raise CascadeHistoryError(
                "batch must be a CascadeEventBatch."
            )

        return self.extend(batch.events)

    def with_metadata(self, **updates: Any) -> CascadeHistory:
        merged = dict(self.metadata)
        merged.update(updates)

        return CascadeHistory(
            events=self.events,
            allow_external_parents=self.allow_external_parents,
            metadata=merged,
        )

    def for_plane(
        self,
        plane_id: str,
    ) -> tuple[CascadeEvent, ...]:
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

    def by_severity(
        self,
        severity: CascadeEventSeverity | str,
    ) -> tuple[CascadeEvent, ...]:
        normalized = _coerce_severity(severity)

        return tuple(
            event
            for event in self.events
            if event.severity is normalized
        )

    def at_or_above_severity(
        self,
        severity: CascadeEventSeverity | str,
    ) -> tuple[CascadeEvent, ...]:
        normalized = _coerce_severity(severity)
        minimum_rank = _severity_rank(normalized)

        return tuple(
            event
            for event in self.events
            if _severity_rank(event.severity) >= minimum_rank
        )

    def between(
        self,
        start_time: float,
        end_time: float,
        *,
        inclusive: bool = True,
    ) -> tuple[CascadeEvent, ...]:
        start = _finite_non_negative(
            start_time,
            name="start_time",
        )
        end = _finite_non_negative(
            end_time,
            name="end_time",
        )

        if start > end:
            raise CascadeHistoryError(
                "start_time cannot exceed end_time."
            )

        if not isinstance(inclusive, bool):
            raise CascadeHistoryError(
                "inclusive must be a bool."
            )

        if inclusive:
            return tuple(
                event
                for event in self.events
                if start <= event.time <= end
            )

        return tuple(
            event
            for event in self.events
            if start < event.time < end
        )

    def children_of(
        self,
        event_id: str,
    ) -> tuple[CascadeEvent, ...]:
        parent = self.get(event_id)

        return tuple(
            event
            for event in self.events
            if event.parent_event_id == parent.event_id
        )

    def parent_of(
        self,
        event_id: str,
    ) -> CascadeEvent | None:
        event = self.get(event_id)

        if event.parent_event_id is None:
            return None

        try:
            return self.get(event.parent_event_id)
        except CascadeHistoryError:
            if self.allow_external_parents:
                return None
            raise

    def ancestors_of(
        self,
        event_id: str,
    ) -> tuple[CascadeEvent, ...]:
        event = self.get(event_id)
        ancestors: list[CascadeEvent] = []
        visited: set[str] = {event.event_id}
        current = event

        while current.parent_event_id is not None:
            parent_id = current.parent_event_id

            if parent_id in visited:
                raise CascadeHistoryError(
                    "Causal cycle detected in history."
                )

            try:
                parent = self.get(parent_id)
            except CascadeHistoryError:
                if self.allow_external_parents:
                    break
                raise

            ancestors.append(parent)
            visited.add(parent.event_id)
            current = parent

        ancestors.reverse()
        return tuple(ancestors)

    def descendants_of(
        self,
        event_id: str,
    ) -> tuple[CascadeEvent, ...]:
        root = self.get(event_id)
        descendants: list[CascadeEvent] = []
        frontier = [root.event_id]
        visited = {root.event_id}

        while frontier:
            parent_id = frontier.pop(0)

            for child in self.events:
                if child.parent_event_id != parent_id:
                    continue

                if child.event_id in visited:
                    raise CascadeHistoryError(
                        "Causal cycle detected in history."
                    )

                visited.add(child.event_id)
                descendants.append(child)
                frontier.append(child.event_id)

        return tuple(descendants)

    def lineage_of(
        self,
        event_id: str,
    ) -> tuple[CascadeEvent, ...]:
        event = self.get(event_id)
        return self.ancestors_of(event_id) + (event,)

    def causal_depth(self, event_id: str) -> int:
        return len(self.ancestors_of(event_id))

    def maximum_causal_depth(self) -> int:
        if not self.events:
            return 0

        return max(
            self.causal_depth(event.event_id)
            for event in self.events
        )

    def count_by_plane(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}

        for event in self.events:
            counts[event.plane_id] = (
                counts.get(event.plane_id, 0) + 1
            )

        return MappingProxyType(counts)

    def count_by_kind(self) -> Mapping[CascadeEventKind, int]:
        counts: dict[CascadeEventKind, int] = {}

        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1

        return MappingProxyType(counts)

    def count_by_severity(
        self,
    ) -> Mapping[CascadeEventSeverity, int]:
        counts: dict[CascadeEventSeverity, int] = {}

        for event in self.events:
            counts[event.severity] = (
                counts.get(event.severity, 0) + 1
            )

        return MappingProxyType(counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "allow_external_parents": (
                self.allow_external_parents
            ),
            "metadata": dict(self.metadata),
            "events": [
                event.as_dict()
                for event in self.events
            ],
        }


def merge_histories(
    *histories: CascadeHistory,
    allow_external_parents: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CascadeHistory:
    """
    Merge histories into deterministic time order.

    Equal-time events preserve the order of the supplied histories and the
    order inside each history.
    """

    for index, history in enumerate(histories):
        if not isinstance(history, CascadeHistory):
            raise CascadeHistoryError(
                f"histories[{index}] must be a CascadeHistory."
            )

    if allow_external_parents is None:
        allow_external = any(
            history.allow_external_parents
            for history in histories
        )
    else:
        if not isinstance(allow_external_parents, bool):
            raise CascadeHistoryError(
                "allow_external_parents must be a bool."
            )
        allow_external = allow_external_parents

    indexed_events: list[tuple[float, int, int, CascadeEvent]] = []

    for history_index, history in enumerate(histories):
        for event_index, event in enumerate(history.events):
            indexed_events.append(
                (
                    event.time,
                    history_index,
                    event_index,
                    event,
                )
            )

    indexed_events.sort(
        key=lambda item: (item[0], item[1], item[2])
    )

    if metadata is None:
        merged_metadata: dict[str, Any] = {}
        for history in histories:
            merged_metadata.update(history.metadata)
    else:
        merged_metadata = dict(_freeze_metadata(metadata))

    return CascadeHistory(
        events=tuple(item[3] for item in indexed_events),
        allow_external_parents=allow_external,
        metadata=merged_metadata,
    )


def build_history(
    events: Iterable[CascadeEvent],
    *,
    sort_by_time: bool = False,
    allow_external_parents: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> CascadeHistory:
    """Build a validated history from an event iterable."""

    if not isinstance(sort_by_time, bool):
        raise CascadeHistoryError(
            "sort_by_time must be a bool."
        )

    try:
        normalized = tuple(events)
    except TypeError as exc:
        raise CascadeHistoryError(
            "events must be iterable."
        ) from exc

    if sort_by_time:
        normalized = tuple(
            event
            for _, event in sorted(
                enumerate(normalized),
                key=lambda item: (
                    getattr(item[1], "time", math.inf),
                    item[0],
                ),
            )
        )

    return CascadeHistory(
        events=normalized,
        allow_external_parents=allow_external_parents,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "CascadeHistory",
    "CascadeHistoryError",
    "build_history",
    "merge_histories",
]

