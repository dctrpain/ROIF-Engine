"""
Immutable full-state snapshot for the ROIF engine.

A ``StateSnapshot`` binds together the numerical output of one
``PlaneKernel`` step, the discrete events detected for that step, and the
persistent cascade history after those events have been recorded.

The class validates cross-module invariants so downstream simulation,
serialization, metrics, replay, and visualization layers can consume one
coherent object instead of several loosely related values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from types import MappingProxyType
from typing import Any
import math

from .cascade_event import CascadeEvent, CascadeEventBatch
from .history import CascadeHistory
from .plane_kernel import PlaneKernelResult


class StateSnapshotError(ValueError):
    """Raised when a state snapshot is internally inconsistent."""


_TIME_ABS_TOLERANCE = 1e-12


def _finite_non_negative(value: float, *, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise StateSnapshotError(
            f"{name} must be a real number."
        ) from exc

    if not math.isfinite(converted):
        raise StateSnapshotError(f"{name} must be finite.")

    if converted < 0.0:
        raise StateSnapshotError(
            f"{name} must be greater than or equal to 0.0."
        )

    return converted


def _normalize_optional_string(
    value: str | None,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise StateSnapshotError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise StateSnapshotError(f"{name} cannot be empty.")

    return normalized


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise StateSnapshotError("metadata must be a mapping.")

    copied: dict[str, Any] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise StateSnapshotError(
                "metadata key must be a string."
            )

        normalized_key = key.strip()

        if not normalized_key:
            raise StateSnapshotError(
                "metadata key cannot be empty."
            )

        copied[normalized_key] = value

    return MappingProxyType(copied)


def _times_equal(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=_TIME_ABS_TOLERANCE,
    )


def _stable_snapshot_id(
    *,
    time: float,
    step_index: int | None,
    plane_ids: tuple[str, ...],
    event_ids: tuple[str, ...],
    history_event_ids: tuple[str, ...],
) -> str:
    payload = "|".join(
        (
            format(time, ".17g"),
            "" if step_index is None else str(step_index),
            ",".join(plane_ids),
            ",".join(event_ids),
            ",".join(history_event_ids),
        )
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"snap_{digest}"


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """
    Immutable coherent state of one ROIF simulation step.

    Invariants
    ----------
    * ``time`` equals ``kernel_result.time`` and ``event_batch.time``.
    * Every event in ``event_batch`` belongs to the current time.
    * Every event in ``event_batch`` is already present in ``history``.
    * History cannot contain events later than the snapshot time.
    * If ``step_index`` is supplied, it is a non-negative integer.
    """

    kernel_result: PlaneKernelResult
    event_batch: CascadeEventBatch
    history: CascadeHistory
    time: float | None = None
    step_index: int | None = None
    snapshot_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_result, PlaneKernelResult):
            raise StateSnapshotError(
                "kernel_result must be a PlaneKernelResult."
            )

        if not isinstance(self.event_batch, CascadeEventBatch):
            raise StateSnapshotError(
                "event_batch must be a CascadeEventBatch."
            )

        if not isinstance(self.history, CascadeHistory):
            raise StateSnapshotError(
                "history must be a CascadeHistory."
            )

        resolved_time = (
            self.kernel_result.time
            if self.time is None
            else _finite_non_negative(self.time, name="time")
        )

        if not _times_equal(
            resolved_time,
            self.kernel_result.time,
        ):
            raise StateSnapshotError(
                "time must match kernel_result.time."
            )

        if not _times_equal(
            resolved_time,
            self.event_batch.time,
        ):
            raise StateSnapshotError(
                "time must match event_batch.time."
            )

        if self.step_index is not None:
            if (
                isinstance(self.step_index, bool)
                or not isinstance(self.step_index, int)
            ):
                raise StateSnapshotError(
                    "step_index must be an integer or None."
                )

            if self.step_index < 0:
                raise StateSnapshotError(
                    "step_index must be non-negative."
                )

        if (
            self.history.end_time is not None
            and self.history.end_time
            > resolved_time + _TIME_ABS_TOLERANCE
        ):
            raise StateSnapshotError(
                "history cannot contain events later than "
                "the snapshot time."
            )

        history_by_id = {
            event.event_id: event
            for event in self.history.events
        }

        for event in self.event_batch.events:
            if not _times_equal(event.time, resolved_time):
                raise StateSnapshotError(
                    f"Event {event.event_id!r} does not belong "
                    "to the snapshot time."
                )

            historical = history_by_id.get(event.event_id)

            if historical is None:
                raise StateSnapshotError(
                    f"Current event {event.event_id!r} is missing "
                    "from history."
                )

            if historical != event:
                raise StateSnapshotError(
                    f"History contains a different event under "
                    f"ID {event.event_id!r}."
                )

        metadata = _freeze_metadata(self.metadata)
        snapshot_id = _normalize_optional_string(
            self.snapshot_id,
            name="snapshot_id",
        )

        if snapshot_id is None:
            snapshot_id = _stable_snapshot_id(
                time=resolved_time,
                step_index=self.step_index,
                plane_ids=self.kernel_result.plane_ids,
                event_ids=tuple(
                    event.event_id
                    for event in self.event_batch.events
                ),
                history_event_ids=self.history.event_ids,
            )

        object.__setattr__(self, "time", resolved_time)
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "metadata", metadata)

    @property
    def plane_ids(self) -> tuple[str, ...]:
        """Plane identifiers in kernel order."""

        return self.kernel_result.plane_ids

    @property
    def plane_count(self) -> int:
        """Number of planes represented by the snapshot."""

        return self.kernel_result.plane_count

    @property
    def event_count(self) -> int:
        """Number of events detected at this step."""

        return len(self.event_batch)

    @property
    def history_event_count(self) -> int:
        """Total number of events accumulated in history."""

        return len(self.history)

    @property
    def changed(self) -> bool:
        """Whether the continuous kernel state changed."""

        return self.kernel_result.changed

    @property
    def changed_plane_ids(self) -> tuple[str, ...]:
        """Plane identifiers whose activation changed."""

        return self.kernel_result.changed_plane_ids

    @property
    def has_events(self) -> bool:
        """Whether at least one discrete event occurred."""

        return bool(self.event_batch)

    @property
    def current_events(self) -> tuple[CascadeEvent, ...]:
        """Events detected at this snapshot time."""

        return self.event_batch.events

    @property
    def activation_before(self) -> Mapping[str, float]:
        """Read-only activation mapping before the kernel step."""

        return MappingProxyType(
            {
                plane_id: float(value)
                for plane_id, value in zip(
                    self.kernel_result.plane_ids,
                    self.kernel_result.activation_before,
                    strict=True,
                )
            }
        )

    @property
    def activation_after(self) -> Mapping[str, float]:
        """Read-only activation mapping after the kernel step."""

        return MappingProxyType(
            {
                plane_id: float(value)
                for plane_id, value in zip(
                    self.kernel_result.plane_ids,
                    self.kernel_result.activation_after,
                    strict=True,
                )
            }
        )

    def activation_for(self, plane_id: str) -> float:
        """Return resulting activation for one plane."""

        return self.kernel_result.activation_for(plane_id)

    def events_for_plane(
        self,
        plane_id: str,
    ) -> tuple[CascadeEvent, ...]:
        """Return current-step events for one plane."""

        if not isinstance(plane_id, str):
            raise StateSnapshotError(
                "plane_id must be a string."
            )

        normalized = plane_id.strip()

        if not normalized:
            raise StateSnapshotError(
                "plane_id cannot be empty."
            )

        return tuple(
            event
            for event in self.event_batch.events
            if event.plane_id == normalized
        )

    def with_metadata(self, **updates: Any) -> StateSnapshot:
        """Return an independent snapshot with merged metadata."""

        merged = dict(self.metadata)
        merged.update(updates)
        return replace(self, metadata=merged)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return {
            "snapshot_id": self.snapshot_id,
            "time": self.time,
            "step_index": self.step_index,
            "plane_count": self.plane_count,
            "event_count": self.event_count,
            "history_event_count": self.history_event_count,
            "changed": self.changed,
            "changed_plane_ids": list(
                self.changed_plane_ids
            ),
            "metadata": dict(self.metadata),
            "kernel_result": self.kernel_result.as_dict(),
            "event_batch": self.event_batch.as_dict(),
            "history": self.history.as_dict(),
        }


def create_state_snapshot(
    kernel_result: PlaneKernelResult,
    event_batch: CascadeEventBatch,
    history_before: CascadeHistory | None = None,
    *,
    step_index: int | None = None,
    snapshot_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StateSnapshot:
    """
    Create a snapshot and append the current batch to prior history.

    ``history_before`` must end no later than the current kernel time. The
    returned snapshot contains the new accumulated history.
    """

    if not isinstance(kernel_result, PlaneKernelResult):
        raise StateSnapshotError(
            "kernel_result must be a PlaneKernelResult."
        )

    if not isinstance(event_batch, CascadeEventBatch):
        raise StateSnapshotError(
            "event_batch must be a CascadeEventBatch."
        )

    if history_before is None:
        history = CascadeHistory()
    elif isinstance(history_before, CascadeHistory):
        history = history_before
    else:
        raise StateSnapshotError(
            "history_before must be a CascadeHistory or None."
        )

    if (
        history.end_time is not None
        and history.end_time
        > kernel_result.time + _TIME_ABS_TOLERANCE
    ):
        raise StateSnapshotError(
            "history_before cannot contain events later "
            "than kernel_result.time."
        )

    try:
        accumulated_history = history.append_batch(event_batch)
    except ValueError as exc:
        raise StateSnapshotError(
            "Cannot append event_batch to history_before."
        ) from exc

    return StateSnapshot(
        kernel_result=kernel_result,
        event_batch=event_batch,
        history=accumulated_history,
        step_index=step_index,
        snapshot_id=snapshot_id,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "StateSnapshot",
    "StateSnapshotError",
    "create_state_snapshot",
]
