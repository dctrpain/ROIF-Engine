from __future__ import annotations

"""
Persistent structural changes for the ROIF Structural Memory Engine.

HistoryEvent answers:

    What happened?

IrreversibleChange answers:

    What remained in the structure after it happened?

The class is domain-independent and can represent:

- residual deformation;
- residual stress;
- plastic deformation;
- material damage;
- fatigue;
- corrosion;
- biological remodeling;
- fibrosis or scar formation;
- permanent geometry drift;
- reference-length drift;
- anisotropy drift;
- capacity loss or gain;
- topology change;
- persistent controller adaptation.

An IrreversibleChange is immutable. Once recorded, its historical contents
cannot be silently rewritten.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .event import (
    HistoryEvent,
    HistoryTarget,
    HistoryValidationError,
    ReversibilityClass,
    StateDelta,
    TimeScale,
)


class IrreversibleChangeError(
    HistoryValidationError
):
    """Raised when a persistent structural change is invalid."""


class IrreversibleChangeKind(str, Enum):
    """Stable categories of persistent structural memory."""

    DAMAGE = "damage"
    FATIGUE = "fatigue"
    CREEP = "creep"
    PLASTIC_STRAIN = "plastic_strain"
    RESIDUAL_STRAIN = "residual_strain"
    RESIDUAL_STRESS = "residual_stress"
    REFERENCE_LENGTH_DRIFT = "reference_length_drift"
    GEOMETRY_DRIFT = "geometry_drift"
    TOPOLOGY_CHANGE = "topology_change"
    STIFFNESS_CHANGE = "stiffness_change"
    DAMPING_CHANGE = "damping_change"
    CAPACITY_CHANGE = "capacity_change"
    ANISOTROPY_CHANGE = "anisotropy_change"
    CORROSION = "corrosion"
    AGING = "aging"
    REMODELING = "remodeling"
    SCAR_FORMATION = "scar_formation"
    FIBROSIS = "fibrosis"
    HYPERTROPHY = "hypertrophy"
    ATROPHY = "atrophy"
    ADAPTIVE_REORGANIZATION = "adaptive_reorganization"
    CONTROL_PATTERN_CHANGE = "control_pattern_change"
    CUSTOM = "custom"


class TracePersistence(str, Enum):
    """
    Persistence class of a structural trace.

    The numerical retained_fraction remains the primary value. This enum
    supports grouping, filtering, scheduling, and visualization.
    """

    TRANSIENT = "transient"
    LONG_LIVED = "long_lived"
    PERMANENT = "permanent"


class FunctionalEffect(str, Enum):
    """
    Functional meaning of the structural change.

    Irreversibility is not automatically harmful. Remodeling may improve
    function, reduce function, or create a trade-off.
    """

    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"
    MIXED = "mixed"
    UNKNOWN = "unknown"


def _require_nonempty_string(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise IrreversibleChangeError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise IrreversibleChangeError(
            f"{field_name} must not be empty"
        )

    return normalized


def _validate_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise IrreversibleChangeError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise IrreversibleChangeError(
            f"{field_name} must be finite"
        )

    return numeric


def _validate_nonnegative_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _validate_finite(
        value,
        field_name=field_name,
    )

    if numeric < 0.0:
        raise IrreversibleChangeError(
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
        raise IrreversibleChangeError(
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
        raise IrreversibleChangeError(
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


@dataclass(frozen=True, slots=True)
class IrreversibleChange:
    """
    One persistent trace stored in the structure.

    Parameters
    ----------
    kind:
        Semantic class of the persistent structural change.

    source_event_id:
        HistoryEvent that created or revealed the trace.

    target:
        Stable identifier of the affected structural object.

    delta:
        State transition whose retained part forms the trace.

    onset_time:
        Time at which the persistent change began.

    recorded_time:
        Time at which the trace was recorded or recognized.

    retained_fraction:
        Fraction of the original StateDelta retained in the structure.

        0.0 means no persistent trace.
        1.0 means the complete change is retained.

    permanence:
        Human-readable persistence class.

    characteristic_time:
        Characteristic time of formation or consolidation in seconds.

    memory_strength:
        Strength of the trace as a history signature in [0, 1].

    capacity_effect:
        Signed normalized effect on effective Capacity.

        Negative values reduce Capacity.
        Positive values increase Capacity.

    functional_effect:
        Whether the structural change is beneficial, harmful, mixed,
        neutral, or still unknown.

    cause_change_ids:
        Persistent predecessor changes that contributed to this change.

    plane_ids:
        Influence planes involved in producing the trace.

    agent_ids:
        External or internal agents involved in producing the trace.
    """

    kind: IrreversibleChangeKind
    source_event_id: str
    target: HistoryTarget
    delta: StateDelta
    onset_time: float
    recorded_time: float
    retained_fraction: float
    permanence: TracePersistence
    characteristic_time: float = 0.0
    time_scale: TimeScale = TimeScale.SLOW
    memory_strength: float = 1.0
    capacity_effect: float = 0.0
    functional_effect: FunctionalEffect = (
        FunctionalEffect.UNKNOWN
    )
    reversibility: ReversibilityClass = (
        ReversibilityClass.IRREVERSIBLE
    )
    cause_change_ids: tuple[str, ...] = ()
    plane_ids: tuple[str, ...] = ()
    agent_ids: tuple[str, ...] = ()
    description: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    change_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        try:
            kind = IrreversibleChangeKind(
                self.kind
            )
        except (TypeError, ValueError) as exc:
            raise IrreversibleChangeError(
                "unsupported irreversible change kind: "
                f"{self.kind!r}"
            ) from exc

        source_event_id = _require_nonempty_string(
            self.source_event_id,
            field_name="source_event_id",
        )

        if not isinstance(
            self.target,
            HistoryTarget,
        ):
            raise IrreversibleChangeError(
                "target must be a HistoryTarget"
            )

        if not isinstance(
            self.delta,
            StateDelta,
        ):
            raise IrreversibleChangeError(
                "delta must be a StateDelta"
            )

        if not self.delta.changed:
            raise IrreversibleChangeError(
                "delta must describe an actual state change"
            )

        onset_time = _validate_nonnegative_finite(
            self.onset_time,
            field_name="onset_time",
        )

        recorded_time = (
            _validate_nonnegative_finite(
                self.recorded_time,
                field_name="recorded_time",
            )
        )

        if recorded_time < onset_time:
            raise IrreversibleChangeError(
                "recorded_time must not precede onset_time"
            )

        retained_fraction = (
            _validate_unit_interval(
                self.retained_fraction,
                field_name="retained_fraction",
            )
        )

        if retained_fraction <= 0.0:
            raise IrreversibleChangeError(
                "retained_fraction must be greater than zero"
            )

        try:
            permanence = TracePersistence(
                self.permanence
            )
        except (TypeError, ValueError) as exc:
            raise IrreversibleChangeError(
                "unsupported trace persistence: "
                f"{self.permanence!r}"
            ) from exc

        if (
            permanence is TracePersistence.TRANSIENT
            and retained_fraction >= 1.0
        ):
            raise IrreversibleChangeError(
                "a fully retained trace cannot be transient"
            )

        characteristic_time = (
            _validate_nonnegative_finite(
                self.characteristic_time,
                field_name="characteristic_time",
            )
        )

        try:
            time_scale = TimeScale(
                self.time_scale
            )
        except (TypeError, ValueError) as exc:
            raise IrreversibleChangeError(
                f"unsupported time scale: {self.time_scale!r}"
            ) from exc

        memory_strength = _validate_unit_interval(
            self.memory_strength,
            field_name="memory_strength",
        )

        capacity_effect = _validate_finite(
            self.capacity_effect,
            field_name="capacity_effect",
        )

        if not -1.0 <= capacity_effect <= 1.0:
            raise IrreversibleChangeError(
                "capacity_effect must be in [-1, 1]"
            )

        try:
            functional_effect = FunctionalEffect(
                self.functional_effect
            )
        except (TypeError, ValueError) as exc:
            raise IrreversibleChangeError(
                "unsupported functional effect: "
                f"{self.functional_effect!r}"
            ) from exc

        try:
            reversibility = ReversibilityClass(
                self.reversibility
            )
        except (TypeError, ValueError) as exc:
            raise IrreversibleChangeError(
                "unsupported reversibility class: "
                f"{self.reversibility!r}"
            ) from exc

        if (
            reversibility
            is ReversibilityClass.REVERSIBLE
        ):
            raise IrreversibleChangeError(
                "an IrreversibleChange cannot be fully reversible"
            )

        cause_change_ids = _normalize_ids(
            self.cause_change_ids,
            field_name="cause_change_ids",
        )
        plane_ids = _normalize_ids(
            self.plane_ids,
            field_name="plane_ids",
        )
        agent_ids = _normalize_ids(
            self.agent_ids,
            field_name="agent_ids",
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

        change_id = _require_nonempty_string(
            self.change_id,
            field_name="change_id",
        )

        if change_id in cause_change_ids:
            raise IrreversibleChangeError(
                "a change cannot directly cause itself"
            )

        object.__setattr__(
            self,
            "kind",
            kind,
        )
        object.__setattr__(
            self,
            "source_event_id",
            source_event_id,
        )
        object.__setattr__(
            self,
            "onset_time",
            onset_time,
        )
        object.__setattr__(
            self,
            "recorded_time",
            recorded_time,
        )
        object.__setattr__(
            self,
            "retained_fraction",
            retained_fraction,
        )
        object.__setattr__(
            self,
            "permanence",
            permanence,
        )
        object.__setattr__(
            self,
            "characteristic_time",
            characteristic_time,
        )
        object.__setattr__(
            self,
            "time_scale",
            time_scale,
        )
        object.__setattr__(
            self,
            "memory_strength",
            memory_strength,
        )
        object.__setattr__(
            self,
            "capacity_effect",
            capacity_effect,
        )
        object.__setattr__(
            self,
            "functional_effect",
            functional_effect,
        )
        object.__setattr__(
            self,
            "reversibility",
            reversibility,
        )
        object.__setattr__(
            self,
            "cause_change_ids",
            cause_change_ids,
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
            "change_id",
            change_id,
        )

    @property
    def formation_delay(self) -> float:
        """Return recorded_time - onset_time."""

        return self.recorded_time - self.onset_time

    @property
    def is_permanent(self) -> bool:
        """Return whether the trace is classified as permanent."""

        return (
            self.permanence
            is TracePersistence.PERMANENT
        )

    @property
    def is_partially_reversible(self) -> bool:
        """Return whether some reversal remains possible."""

        return (
            self.reversibility
            is ReversibilityClass.PARTIALLY_REVERSIBLE
        )

    @property
    def reduces_capacity(self) -> bool:
        """Return whether the change reduces effective Capacity."""

        return self.capacity_effect < 0.0

    @property
    def improves_capacity(self) -> bool:
        """Return whether the change improves effective Capacity."""

        return self.capacity_effect > 0.0

    @property
    def is_harmful(self) -> bool:
        """Return whether the functional effect is harmful."""

        return (
            self.functional_effect
            is FunctionalEffect.HARMFUL
        )

    @property
    def is_beneficial(self) -> bool:
        """Return whether the functional effect is beneficial."""

        return (
            self.functional_effect
            is FunctionalEffect.BENEFICIAL
        )

    @property
    def numeric_total_change(self) -> float | None:
        """
        Return the complete numeric state change before retention.

        Returns None for non-numeric deltas.
        """

        return self.delta.numeric_change

    @property
    def retained_numeric_change(self) -> float | None:
        """
        Return the persistent numeric part of the StateDelta.

        Example:

            total damage change = 0.20
            retained_fraction   = 0.75
            retained change     = 0.15
        """

        total = self.numeric_total_change

        if total is None:
            return None

        return total * self.retained_fraction

    @property
    def signature_weight(self) -> float:
        """
        Return the normalized contribution to a structural signature.

        The value combines persistence and current memory strength.
        """

        return (
            self.retained_fraction
            * self.memory_strength
        )

    def has_cause_change(
        self,
        change_id: str,
    ) -> bool:
        """Return whether change_id is a direct persistent predecessor."""

        normalized = _require_nonempty_string(
            change_id,
            field_name="change_id",
        )

        return normalized in self.cause_change_ids

    def involves_plane(
        self,
        plane_id: str,
    ) -> bool:
        """Return whether an influence plane contributed."""

        normalized = _require_nonempty_string(
            plane_id,
            field_name="plane_id",
        )

        return normalized in self.plane_ids

    def involves_agent(
        self,
        agent_id: str,
    ) -> bool:
        """Return whether an agent contributed."""

        normalized = _require_nonempty_string(
            agent_id,
            field_name="agent_id",
        )

        return normalized in self.agent_ids

    def to_dict(self) -> dict[str, Any]:
        """Serialize the persistent change."""

        return {
            "version": self.VERSION,
            "change_id": self.change_id,
            "kind": self.kind.value,
            "source_event_id": self.source_event_id,
            "target": self.target.to_dict(),
            "delta": self.delta.to_dict(),
            "onset_time": self.onset_time,
            "recorded_time": self.recorded_time,
            "retained_fraction": self.retained_fraction,
            "permanence": self.permanence.value,
            "characteristic_time": (
                self.characteristic_time
            ),
            "time_scale": self.time_scale.value,
            "memory_strength": self.memory_strength,
            "capacity_effect": self.capacity_effect,
            "functional_effect": (
                self.functional_effect.value
            ),
            "reversibility": self.reversibility.value,
            "cause_change_ids": list(
                self.cause_change_ids
            ),
            "plane_ids": list(self.plane_ids),
            "agent_ids": list(self.agent_ids),
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> IrreversibleChange:
        """Deserialize an IrreversibleChange."""

        if not isinstance(data, Mapping):
            raise IrreversibleChangeError(
                "irreversible change data must be a mapping"
            )

        version = data.get(
            "version",
            cls.VERSION,
        )

        if version != cls.VERSION:
            raise IrreversibleChangeError(
                "unsupported irreversible change version: "
                f"{version!r}"
            )

        target_data = data.get("target")

        if not isinstance(target_data, Mapping):
            raise IrreversibleChangeError(
                "target is missing or invalid"
            )

        delta_data = data.get("delta")

        if not isinstance(delta_data, Mapping):
            raise IrreversibleChangeError(
                "delta is missing or invalid"
            )

        return cls(
            change_id=str(data["change_id"]),
            kind=IrreversibleChangeKind(
                data["kind"]
            ),
            source_event_id=str(
                data["source_event_id"]
            ),
            target=HistoryTarget.from_dict(
                target_data
            ),
            delta=StateDelta.from_dict(
                delta_data
            ),
            onset_time=float(
                data["onset_time"]
            ),
            recorded_time=float(
                data["recorded_time"]
            ),
            retained_fraction=float(
                data["retained_fraction"]
            ),
            permanence=TracePersistence(
                data["permanence"]
            ),
            characteristic_time=float(
                data.get(
                    "characteristic_time",
                    0.0,
                )
            ),
            time_scale=TimeScale(
                data.get(
                    "time_scale",
                    TimeScale.SLOW.value,
                )
            ),
            memory_strength=float(
                data.get(
                    "memory_strength",
                    1.0,
                )
            ),
            capacity_effect=float(
                data.get(
                    "capacity_effect",
                    0.0,
                )
            ),
            functional_effect=FunctionalEffect(
                data.get(
                    "functional_effect",
                    FunctionalEffect.UNKNOWN.value,
                )
            ),
            reversibility=ReversibilityClass(
                data.get(
                    "reversibility",
                    ReversibilityClass.IRREVERSIBLE.value,
                )
            ),
            cause_change_ids=tuple(
                data.get(
                    "cause_change_ids",
                    (),
                )
            ),
            plane_ids=tuple(
                data.get(
                    "plane_ids",
                    (),
                )
            ),
            agent_ids=tuple(
                data.get(
                    "agent_ids",
                    (),
                )
            ),
            description=data.get(
                "description"
            ),
            metadata=data.get(
                "metadata",
                {}
            ),
        )

    @classmethod
    def from_event(
        cls,
        event: HistoryEvent,
        *,
        kind: IrreversibleChangeKind,
        quantity: str,
        retained_fraction: float,
        permanence: TracePersistence,
        recorded_time: float | None = None,
        memory_strength: float | None = None,
        capacity_effect: float | None = None,
        functional_effect: FunctionalEffect = (
            FunctionalEffect.UNKNOWN
        ),
        reversibility: ReversibilityClass | None = None,
        cause_change_ids: tuple[str, ...] = (),
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        change_id: str | None = None,
    ) -> IrreversibleChange:
        """
        Build a persistent structural change from a HistoryEvent.

        The selected quantity must exist in event.deltas and must actually
        change.

        Event persistence, capacity effect, temporal scale, participating
        planes, agents, and target are inherited unless explicitly replaced.
        """

        if not isinstance(event, HistoryEvent):
            raise IrreversibleChangeError(
                "event must be a HistoryEvent"
            )

        normalized_quantity = (
            _require_nonempty_string(
                quantity,
                field_name="quantity",
            )
        )

        delta = event.delta_for(
            normalized_quantity
        )

        if delta is None:
            raise IrreversibleChangeError(
                "event contains no delta for quantity "
                f"{normalized_quantity!r}"
            )

        if not delta.changed:
            raise IrreversibleChangeError(
                "selected event delta contains no state change"
            )

        if recorded_time is None:
            recorded_time = event.time

        if memory_strength is None:
            memory_strength = max(
                event.persistence,
                retained_fraction,
            )

        if capacity_effect is None:
            capacity_effect = (
                event.capacity_effect
                * retained_fraction
            )

        if reversibility is None:
            if (
                event.reversibility
                is ReversibilityClass.REVERSIBLE
            ):
                reversibility = (
                    ReversibilityClass.PARTIALLY_REVERSIBLE
                )
            else:
                reversibility = event.reversibility

        merged_metadata: dict[str, Any] = {
            "source_event_kind": (
                event.kind.value
            ),
            "source_event_confidence": (
                event.confidence
            ),
        }

        if metadata is not None:
            merged_metadata.update(
                dict(metadata)
            )

        constructor_args: dict[str, Any] = {
            "kind": kind,
            "source_event_id": event.event_id,
            "target": event.target,
            "delta": delta,
            "onset_time": event.time,
            "recorded_time": recorded_time,
            "retained_fraction": retained_fraction,
            "permanence": permanence,
            "characteristic_time": (
                event.characteristic_time
            ),
            "time_scale": event.time_scale,
            "memory_strength": memory_strength,
            "capacity_effect": capacity_effect,
            "functional_effect": functional_effect,
            "reversibility": reversibility,
            "cause_change_ids": cause_change_ids,
            "plane_ids": event.plane_ids,
            "agent_ids": event.agent_ids,
            "description": (
                description
                if description is not None
                else event.description
            ),
            "metadata": merged_metadata,
        }

        if change_id is not None:
            constructor_args["change_id"] = (
                change_id
            )

        return cls(**constructor_args)


__all__ = [
    "FunctionalEffect",
    "IrreversibleChange",
    "IrreversibleChangeError",
    "IrreversibleChangeKind",
    "TracePersistence",
]