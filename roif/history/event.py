from __future__ import annotations

"""
Fundamental event model for the ROIF Structural Memory Engine.

A HistoryEvent records one state transition that may leave a trace in the
structure. It does not itself modify a Network, Material, Capacity state, or
causal graph.

The event model is intentionally domain-independent. The same class can
represent:

- mechanical impact;
- material damage;
- fatigue accumulation;
- residual strain;
- corrosion;
- biological remodeling;
- reflex activation;
- temperature exposure;
- influence-plane state changes;
- capacity loss;
- induced edge collapse;
- intervention.

HistoryEvent is immutable after construction so previously recorded history
cannot be silently rewritten.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any
from uuid import uuid4


class HistoryValidationError(ValueError):
    """Raised when a history event contains invalid or inconsistent data."""


class HistoryEventKind(str, Enum):
    """Stable categories of events recorded by Structural Memory."""

    EXTERNAL_PERTURBATION = "external_perturbation"
    INFLUENCE_PLANE_CHANGE = "influence_plane_change"
    LOAD_CHANGE = "load_change"
    CAPACITY_CHANGE = "capacity_change"
    RESERVE_THRESHOLD = "reserve_threshold"
    MATERIAL_STATE_CHANGE = "material_state_change"
    DAMAGE = "damage"
    FATIGUE = "fatigue"
    CREEP = "creep"
    PLASTIC_DEFORMATION = "plastic_deformation"
    RESIDUAL_STRAIN = "residual_strain"
    RESIDUAL_STRESS = "residual_stress"
    REMODELING = "remodeling"
    CORROSION = "corrosion"
    AGING = "aging"
    GEOMETRY_CHANGE = "geometry_change"
    TOPOLOGY_CHANGE = "topology_change"
    EDGE_COLLAPSE = "edge_collapse"
    INDUCED_COLLAPSE = "induced_collapse"
    SENSOR_SIGNAL = "sensor_signal"
    REFLEX_ACTIVATION = "reflex_activation"
    POLICY_ACTION = "policy_action"
    INTERVENTION = "intervention"
    RECOVERY = "recovery"
    CUSTOM = "custom"


class TimeScale(str, Enum):
    """
    Human-readable temporal grouping.

    The numerical characteristic time remains the primary scientific value.
    TimeScale is used for grouping, scheduling, filtering, and visualization.
    """

    INSTANT = "instant"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    VERY_SLOW = "very_slow"


class ReversibilityClass(str, Enum):
    """Qualitative classification of event reversibility."""

    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class HistoryTargetKind(str, Enum):
    """Type of structural object affected by an event."""

    SYSTEM = "system"
    NETWORK = "network"
    NODE = "node"
    EDGE = "edge"
    ELEMENT = "element"
    MATERIAL = "material"
    PLANE = "plane"
    CONTROLLER = "controller"
    AGENT = "agent"
    REGION = "region"
    CUSTOM = "custom"


def _require_nonempty_string(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise HistoryValidationError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise HistoryValidationError(
            f"{field_name} must not be empty"
        )

    return normalized


def _validate_nonnegative_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoryValidationError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise HistoryValidationError(
            f"{field_name} must be finite"
        )

    if numeric < 0.0:
        raise HistoryValidationError(
            f"{field_name} must be non-negative"
        )

    return numeric


def _validate_unit_interval(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _validate_nonnegative_finite(
        value,
        field_name=field_name,
    )

    if numeric > 1.0:
        raise HistoryValidationError(
            f"{field_name} must be in the interval [0, 1]"
        )

    return numeric


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise HistoryValidationError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, Any] = {}

    for raw_key, item in value.items():
        key = _require_nonempty_string(
            raw_key,
            field_name=f"{field_name} key",
        )
        normalized[key] = item

    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class HistoryTarget:
    """
    Reference to the structural object affected by a HistoryEvent.

    The history layer stores stable identifiers instead of direct mutable
    object references. This keeps recordings serializable and prevents past
    events from changing when live objects change.
    """

    kind: HistoryTargetKind
    target_id: str
    label: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        try:
            kind = HistoryTargetKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise HistoryValidationError(
                f"unsupported target kind: {self.kind!r}"
            ) from exc

        target_id = _require_nonempty_string(
            self.target_id,
            field_name="target_id",
        )

        label = self.label

        if label is not None:
            label = _require_nonempty_string(
                label,
                field_name="label",
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe dictionary."""

        return {
            "kind": self.kind.value,
            "target_id": self.target_id,
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> HistoryTarget:
        """Create a target from serialized data."""

        if not isinstance(data, Mapping):
            raise HistoryValidationError(
                "target data must be a mapping"
            )

        return cls(
            kind=HistoryTargetKind(data["kind"]),
            target_id=str(data["target_id"]),
            label=data.get("label"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class StateDelta:
    """
    One named state transition produced by an event.

    Example:

        StateDelta(
            quantity="capacity",
            before=0.92,
            after=0.71,
            units="normalized",
        )
    """

    quantity: str
    before: Any
    after: Any
    units: str | None = None
    channel: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        quantity = _require_nonempty_string(
            self.quantity,
            field_name="quantity",
        )

        units = self.units

        if units is not None:
            units = _require_nonempty_string(
                units,
                field_name="units",
            )

        channel = self.channel

        if channel is not None:
            channel = _require_nonempty_string(
                channel,
                field_name="channel",
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "metadata", metadata)

    @property
    def numeric_change(self) -> float | None:
        """
        Return after - before when both values are finite real numbers.

        Non-numeric state transitions correctly return None.
        """

        try:
            before = float(self.before)
            after = float(self.after)
        except (TypeError, ValueError):
            return None

        if not isfinite(before) or not isfinite(after):
            return None

        return after - before

    @property
    def changed(self) -> bool:
        """Return True when the stored values differ."""

        return self.before != self.after

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe dictionary."""

        return {
            "quantity": self.quantity,
            "before": self.before,
            "after": self.after,
            "units": self.units,
            "channel": self.channel,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> StateDelta:
        """Create a state delta from serialized data."""

        if not isinstance(data, Mapping):
            raise HistoryValidationError(
                "state-delta data must be a mapping"
            )

        return cls(
            quantity=str(data["quantity"]),
            before=data.get("before"),
            after=data.get("after"),
            units=data.get("units"),
            channel=data.get("channel"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class HistoryEvent:
    """
    Immutable event recorded in the structural history.

    Parameters
    ----------
    kind:
        Stable semantic category of the event.

    time:
        Simulation or observation time in seconds.

    target:
        Structural object directly affected by the event.

    deltas:
        State transitions caused or observed during the event.

    cause_event_ids:
        Identifiers of direct predecessor events. This forms a causal DAG
        without storing mutable references.

    plane_ids:
        Influence planes participating in the event.

    agent_ids:
        External or internal agents participating in the event.

    characteristic_time:
        Characteristic process time in seconds.

    persistence:
        Normalized persistence of the structural trace. Zero means no
        persistent trace; one means fully persistent for the modeled horizon.

    reversibility:
        Qualitative reversibility classification.

    capacity_effect:
        Signed normalized change in effective capacity. Negative values
        reduce capacity, positive values improve capacity.

    confidence:
        Confidence that the event and causal attribution are correct.
    """

    kind: HistoryEventKind
    time: float
    target: HistoryTarget
    deltas: tuple[StateDelta, ...] = ()
    cause_event_ids: tuple[str, ...] = ()
    plane_ids: tuple[str, ...] = ()
    agent_ids: tuple[str, ...] = ()
    time_scale: TimeScale = TimeScale.FAST
    characteristic_time: float = 0.0
    persistence: float = 0.0
    reversibility: ReversibilityClass = (
        ReversibilityClass.UNKNOWN
    )
    capacity_effect: float = 0.0
    confidence: float = 1.0
    description: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    event_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        try:
            kind = HistoryEventKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise HistoryValidationError(
                f"unsupported history event kind: {self.kind!r}"
            ) from exc

        time = _validate_nonnegative_finite(
            self.time,
            field_name="time",
        )

        if not isinstance(self.target, HistoryTarget):
            raise HistoryValidationError(
                "target must be a HistoryTarget"
            )

        deltas = tuple(self.deltas)

        if not all(
            isinstance(item, StateDelta)
            for item in deltas
        ):
            raise HistoryValidationError(
                "all deltas must be StateDelta objects"
            )

        cause_event_ids = self._normalize_ids(
            self.cause_event_ids,
            field_name="cause_event_ids",
        )
        plane_ids = self._normalize_ids(
            self.plane_ids,
            field_name="plane_ids",
        )
        agent_ids = self._normalize_ids(
            self.agent_ids,
            field_name="agent_ids",
        )

        try:
            time_scale = TimeScale(self.time_scale)
        except (TypeError, ValueError) as exc:
            raise HistoryValidationError(
                f"unsupported time scale: {self.time_scale!r}"
            ) from exc

        characteristic_time = (
            _validate_nonnegative_finite(
                self.characteristic_time,
                field_name="characteristic_time",
            )
        )

        persistence = _validate_unit_interval(
            self.persistence,
            field_name="persistence",
        )

        try:
            reversibility = ReversibilityClass(
                self.reversibility
            )
        except (TypeError, ValueError) as exc:
            raise HistoryValidationError(
                "unsupported reversibility class: "
                f"{self.reversibility!r}"
            ) from exc

        try:
            capacity_effect = float(
                self.capacity_effect
            )
        except (TypeError, ValueError) as exc:
            raise HistoryValidationError(
                "capacity_effect must be a real number"
            ) from exc

        if not isfinite(capacity_effect):
            raise HistoryValidationError(
                "capacity_effect must be finite"
            )

        if not -1.0 <= capacity_effect <= 1.0:
            raise HistoryValidationError(
                "capacity_effect must be in [-1, 1]"
            )

        confidence = _validate_unit_interval(
            self.confidence,
            field_name="confidence",
        )

        description = self.description

        if description is not None:
            description = _require_nonempty_string(
                description,
                field_name="description",
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        event_id = _require_nonempty_string(
            self.event_id,
            field_name="event_id",
        )

        if event_id in cause_event_ids:
            raise HistoryValidationError(
                "an event cannot directly cause itself"
            )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(
            self,
            "cause_event_ids",
            cause_event_ids,
        )
        object.__setattr__(
            self,
            "plane_ids",
            plane_ids,
        )
        object.__setattr__(
            self,
            "agent_ids",
            agent_ids,
        )
        object.__setattr__(
            self,
            "time_scale",
            time_scale,
        )
        object.__setattr__(
            self,
            "characteristic_time",
            characteristic_time,
        )
        object.__setattr__(
            self,
            "persistence",
            persistence,
        )
        object.__setattr__(
            self,
            "reversibility",
            reversibility,
        )
        object.__setattr__(
            self,
            "capacity_effect",
            capacity_effect,
        )
        object.__setattr__(
            self,
            "confidence",
            confidence,
        )
        object.__setattr__(
            self,
            "description",
            description,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )
        object.__setattr__(
            self,
            "event_id",
            event_id,
        )

    @staticmethod
    def _normalize_ids(
        values: tuple[str, ...] | list[str],
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()

        for raw_value in values:
            value = _require_nonempty_string(
                raw_value,
                field_name=field_name,
            )

            if value in seen:
                continue

            seen.add(value)
            normalized.append(value)

        return tuple(normalized)

    @property
    def is_persistent(self) -> bool:
        """Return True when the event leaves a modeled persistent trace."""

        return self.persistence > 0.0

    @property
    def is_irreversible(self) -> bool:
        """Return True for explicitly irreversible events."""

        return (
            self.reversibility
            is ReversibilityClass.IRREVERSIBLE
        )

    @property
    def reduces_capacity(self) -> bool:
        """Return True when the event reduces effective capacity."""

        return self.capacity_effect < 0.0

    @property
    def improves_capacity(self) -> bool:
        """Return True when the event improves effective capacity."""

        return self.capacity_effect > 0.0

    @property
    def changed_quantities(self) -> tuple[str, ...]:
        """Return names of quantities whose values actually changed."""

        return tuple(
            delta.quantity
            for delta in self.deltas
            if delta.changed
        )

    def has_cause(
        self,
        event_id: str,
    ) -> bool:
        """Return whether event_id is a direct predecessor."""

        normalized = _require_nonempty_string(
            event_id,
            field_name="event_id",
        )
        return normalized in self.cause_event_ids

    def involves_plane(
        self,
        plane_id: str,
    ) -> bool:
        """Return whether an influence plane participated."""

        normalized = _require_nonempty_string(
            plane_id,
            field_name="plane_id",
        )
        return normalized in self.plane_ids

    def involves_agent(
        self,
        agent_id: str,
    ) -> bool:
        """Return whether an agent participated."""

        normalized = _require_nonempty_string(
            agent_id,
            field_name="agent_id",
        )
        return normalized in self.agent_ids

    def delta_for(
        self,
        quantity: str,
    ) -> StateDelta | None:
        """Return the first delta for a named quantity."""

        normalized = _require_nonempty_string(
            quantity,
            field_name="quantity",
        )

        for delta in self.deltas:
            if delta.quantity == normalized:
                return delta

        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event into standard Python values."""

        return {
            "version": self.VERSION,
            "event_id": self.event_id,
            "kind": self.kind.value,
            "time": self.time,
            "target": self.target.to_dict(),
            "deltas": [
                delta.to_dict()
                for delta in self.deltas
            ],
            "cause_event_ids": list(
                self.cause_event_ids
            ),
            "plane_ids": list(self.plane_ids),
            "agent_ids": list(self.agent_ids),
            "time_scale": self.time_scale.value,
            "characteristic_time": (
                self.characteristic_time
            ),
            "persistence": self.persistence,
            "reversibility": self.reversibility.value,
            "capacity_effect": self.capacity_effect,
            "confidence": self.confidence,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> HistoryEvent:
        """Deserialize a HistoryEvent."""

        if not isinstance(data, Mapping):
            raise HistoryValidationError(
                "history event data must be a mapping"
            )

        version = data.get("version", cls.VERSION)

        if version != cls.VERSION:
            raise HistoryValidationError(
                "unsupported history event version: "
                f"{version!r}"
            )

        target_data = data.get("target")

        if not isinstance(target_data, Mapping):
            raise HistoryValidationError(
                "history event target is missing or invalid"
            )

        raw_deltas = data.get("deltas", ())

        if not isinstance(raw_deltas, (list, tuple)):
            raise HistoryValidationError(
                "deltas must be a sequence"
            )

        return cls(
            event_id=str(data["event_id"]),
            kind=HistoryEventKind(data["kind"]),
            time=float(data["time"]),
            target=HistoryTarget.from_dict(
                target_data
            ),
            deltas=tuple(
                StateDelta.from_dict(item)
                for item in raw_deltas
            ),
            cause_event_ids=tuple(
                data.get("cause_event_ids", ())
            ),
            plane_ids=tuple(
                data.get("plane_ids", ())
            ),
            agent_ids=tuple(
                data.get("agent_ids", ())
            ),
            time_scale=TimeScale(
                data.get(
                    "time_scale",
                    TimeScale.FAST.value,
                )
            ),
            characteristic_time=float(
                data.get(
                    "characteristic_time",
                    0.0,
                )
            ),
            persistence=float(
                data.get("persistence", 0.0)
            ),
            reversibility=ReversibilityClass(
                data.get(
                    "reversibility",
                    ReversibilityClass.UNKNOWN.value,
                )
            ),
            capacity_effect=float(
                data.get("capacity_effect", 0.0)
            ),
            confidence=float(
                data.get("confidence", 1.0)
            ),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "HistoryEvent",
    "HistoryEventKind",
    "HistoryTarget",
    "HistoryTargetKind",
    "HistoryValidationError",
    "ReversibilityClass",
    "StateDelta",
    "TimeScale",
]