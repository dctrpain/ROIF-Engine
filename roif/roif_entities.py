"""
Canonical ROIF entities.

This module contains representation-layer entities only. It does not run a
solver and does not make clinical decisions.

Core distinctions:
- D_origin: historical beginning of a cascade.
- D_fast: first channel whose reserve crosses a critical threshold.
- D_root_current: current self-maintaining root; may be a node or a cycle.
- Node*: counterfactual target that best restores unavailable function or
  breaks the sustaining cycle.

Influence planes are transformation contexts. Influence operators define
what happens inside a plane: activation, inhibition, compression, rotation,
mobilization, load transfer, remodeling, and so on.

Effective output is multiplicative inside a channel and vectorial across
channels:

    output_i =
        baseline
        * activation availability
        * reserve factor
        * geometric efficiency
        * history factor

Author: Architect (Dctr Pain)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
import math
import uuid


class ROIFEntityError(ValueError):
    """Raised when a ROIF entity violates an invariant."""


class EntityKind(str, Enum):
    GENERIC = "generic"
    MUSCLE = "muscle"
    MUSCLE_REGION = "muscle_region"
    TENDON = "tendon"
    LIGAMENT = "ligament"
    FASCIA = "fascia"
    NERVE = "nerve"
    VESSEL = "vessel"
    BONE = "bone"
    JOINT = "joint"
    ORGAN = "organ"
    SEGMENT = "segment"
    BEHAVIOR = "behavior"
    INFORMATION = "information"
    ENVIRONMENT = "environment"
    CONTROL = "control"


class FunctionalRole(str, Enum):
    UNKNOWN = "unknown"
    PRIME_MOVER = "prime_mover"
    SYNERGIST = "synergist"
    ANTAGONIST = "antagonist"
    STABILIZER = "stabilizer"
    FIXATOR = "fixator"
    COMPENSATOR = "compensator"
    TRANSMITTER = "transmitter"
    MODULATOR = "modulator"
    SENSOR = "sensor"
    CONTROLLER = "controller"
    CONSTRAINT = "constraint"
    BUFFER = "buffer"


class PlaneKind(str, Enum):
    ARITHMETIC = "arithmetic"
    MECHANICAL = "mechanical"
    GEOMETRIC = "geometric"
    VECTOR = "vector"
    TEMPORAL = "temporal"
    NEURAL = "neural"
    AFFERENT = "afferent"
    MOTOR = "motor"
    VASCULAR = "vascular"
    METABOLIC = "metabolic"
    FASCIAL = "fascial"
    BEHAVIORAL = "behavioral"
    INFORMATIONAL = "informational"
    ENVIRONMENTAL = "environmental"
    METEOROLOGICAL = "meteorological"
    SOCIAL = "social"
    CONTROL = "control"
    HISTORICAL = "historical"
    COUNTERFACTUAL = "counterfactual"


class OperatorKind(str, Enum):
    IDENTITY = "identity"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    ACTIVATE = "activate"
    INHIBIT = "inhibit"
    FACILITATE = "facilitate"
    SUPPRESS = "suppress"
    LOAD = "load"
    UNLOAD = "unload"
    TRANSFER_LOAD = "transfer_load"
    REDISTRIBUTE = "redistribute"
    COMPRESS = "compress"
    DECOMPRESS = "decompress"
    STRETCH = "stretch"
    SHORTEN = "shorten"
    MOBILIZE = "mobilize"
    FIXATE = "fixate"
    RELEASE = "release"
    ROTATE = "rotate"
    TRANSLATE = "translate"
    STABILIZE = "stabilize"
    DESTABILIZE = "destabilize"
    RECRUIT = "recruit"
    DERECRUIT = "derecruit"
    SYNCHRONIZE = "synchronize"
    DESYNCHRONIZE = "desynchronize"
    AMPLIFY = "amplify"
    ATTENUATE = "attenuate"
    FILTER = "filter"
    DELAY = "delay"
    TRIGGER = "trigger"
    GATE = "gate"
    ADAPT = "adapt"
    REMODEL = "remodel"
    HABITUATE = "habituate"
    SENSITIZE = "sensitize"
    LOCK_IN = "lock_in"
    UNLOCK = "unlock"
    DAMAGE = "damage"
    RESTORE = "restore"


class CompositionKind(str, Enum):
    SEQUENTIAL = "sequential"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    CONDITIONAL = "conditional"
    VECTOR_SUM = "vector_sum"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    CUSTOM = "custom"


class TemporalMode(str, Enum):
    STATIC = "static"
    CONTINUOUS = "continuous"
    PHASIC = "phasic"
    PERIODIC = "periodic"
    EVENT_DRIVEN = "event_driven"
    DELAYED = "delayed"
    CUMULATIVE = "cumulative"
    HYSTERETIC = "hysteretic"


class ChannelState(str, Enum):
    AVAILABLE = "available"
    COMPENSATING = "compensating"
    OVERLOADED = "overloaded"
    DEPLETED = "depleted"
    LOCKED = "locked"
    INHIBITED = "inhibited"
    REMODELED = "remodeled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RootRole(str, Enum):
    D_ORIGIN = "d_origin"
    D_FAST = "d_fast"
    D_ROOT_CURRENT = "d_root_current"
    D_ROOT_FUTURE = "d_root_future"
    NODE_STAR = "node_star"
    SYMPTOM = "symptom"
    COMPENSATOR = "compensator"
    CYCLE_ROOT = "cycle_root"


class RootTargetKind(str, Enum):
    ENTITY = "entity"
    CHANNEL = "channel"
    OPERATOR = "operator"
    PLANE = "plane"
    EDGE = "edge"
    CYCLE = "cycle"
    EVENT = "event"
    STATE = "state"


class InterventionKind(str, Enum):
    NONE = "none"
    ACTIVATE = "activate"
    INHIBIT = "inhibit"
    MOBILIZE = "mobilize"
    RELEASE = "release"
    STABILIZE = "stabilize"
    DECOMPRESS = "decompress"
    REDUCE_LOAD = "reduce_load"
    RESTORE_GEOMETRY = "restore_geometry"
    RESTORE_TIMING = "restore_timing"
    BREAK_CYCLE = "break_cycle"
    CUSTOM = "custom"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFEntityError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFEntityError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ROIFEntityError(f"{name} must be finite.")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ROIFEntityError(f"{name} cannot be negative.")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ROIFEntityError(f"{name} must be within [0, 1].")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFEntityError(f"{name} must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFEntityError("metadata must be a mapping.")
    return MappingProxyType(dict(value))


def new_id(prefix: str) -> str:
    return f"{_text(prefix, 'prefix')}_{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite(self.x, "x"))
        object.__setattr__(self, "y", _finite(self.y, "y"))
        object.__setattr__(self, "z", _finite(self.z, "z"))

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vector3":
        magnitude = self.magnitude
        if math.isclose(magnitude, 0.0, abs_tol=1e-15):
            return Vector3()
        return Vector3(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude,
        )

    def scaled(self, factor: float) -> "Vector3":
        factor = _finite(factor, "factor")
        return Vector3(self.x * factor, self.y * factor, self.z * factor)

    def __add__(self, other: "Vector3") -> "Vector3":
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)


@dataclass(frozen=True, slots=True)
class GeometryState:
    position: Vector3 = field(default_factory=Vector3)
    orientation: Vector3 = field(default_factory=Vector3)
    length: float = 1.0
    optimal_length: float = 1.0
    moment_arm: float = 1.0
    optimal_moment_arm: float = 1.0
    mobility: float = 1.0
    stability: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("length", "optimal_length", "moment_arm", "optimal_moment_arm"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ROIFEntityError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "mobility", _unit(self.mobility, "mobility"))
        object.__setattr__(self, "stability", _unit(self.stability, "stability"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def geometric_efficiency(self) -> float:
        length_ratio = self.length / self.optimal_length
        moment_ratio = min(self.moment_arm / self.optimal_moment_arm, 1.0)
        length_factor = math.exp(-abs(math.log(max(length_ratio, 1e-12))))
        return max(
            0.0,
            min(
                1.0,
                length_factor * moment_ratio * self.mobility * self.stability,
            ),
        )


@dataclass(frozen=True, slots=True)
class ActivationAvailability:
    command: float = 1.0
    neural_drive: float = 1.0
    afferent_gate: float = 1.0
    timing: float = 1.0
    coordination: float = 1.0
    geometric_access: float = 1.0
    task_compatibility: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "command",
            "neural_drive",
            "afferent_gate",
            "timing",
            "coordination",
            "geometric_access",
            "task_compatibility",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def effective(self) -> float:
        return (
            self.command
            * self.neural_drive
            * self.afferent_gate
            * self.timing
            * self.coordination
            * self.geometric_access
            * self.task_compatibility
        )


@dataclass(frozen=True, slots=True)
class CapacityState:
    capacity: float
    load: float
    critical_reserve_fraction: float = 0.10

    def __post_init__(self) -> None:
        capacity = _finite(self.capacity, "capacity")
        load = _finite(self.load, "load")
        if capacity <= 0.0 or load < 0.0:
            raise ROIFEntityError("capacity must be positive and load nonnegative.")
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "load", load)
        object.__setattr__(
            self,
            "critical_reserve_fraction",
            _unit(self.critical_reserve_fraction, "critical_reserve_fraction"),
        )

    @property
    def reserve(self) -> float:
        return self.capacity - self.load

    @property
    def reserve_fraction(self) -> float:
        return self.reserve / self.capacity

    @property
    def utilization(self) -> float:
        return self.load / self.capacity

    @property
    def is_critical(self) -> bool:
        return self.reserve_fraction <= self.critical_reserve_fraction


@dataclass(frozen=True, slots=True)
class FunctionalChannel:
    channel_id: str
    entity_id: str
    name: str
    role: FunctionalRole = FunctionalRole.UNKNOWN
    state: ChannelState = ChannelState.AVAILABLE
    direction: Vector3 = field(default_factory=Vector3)
    capacity_state: CapacityState = field(
        default_factory=lambda: CapacityState(1.0, 0.0)
    )
    geometry: GeometryState = field(default_factory=GeometryState)
    activation: ActivationAvailability = field(
        default_factory=ActivationAvailability
    )
    baseline_output: float = 1.0
    history_factor: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("channel_id", "entity_id", "name"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.role, FunctionalRole):
            raise ROIFEntityError("role must be FunctionalRole.")
        if not isinstance(self.state, ChannelState):
            raise ROIFEntityError("state must be ChannelState.")
        object.__setattr__(
            self, "baseline_output", _nonnegative(self.baseline_output, "baseline_output")
        )
        object.__setattr__(
            self, "history_factor", _nonnegative(self.history_factor, "history_factor")
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def effective_scalar_output(self) -> float:
        reserve_factor = max(0.0, min(1.0, self.capacity_state.reserve_fraction))
        return (
            self.baseline_output
            * self.activation.effective
            * reserve_factor
            * self.geometry.geometric_efficiency
            * self.history_factor
        )

    @property
    def effective_vector_output(self) -> Vector3:
        return self.direction.normalized().scaled(self.effective_scalar_output)


@dataclass(frozen=True, slots=True)
class StructuralEntity:
    entity_id: str
    name: str
    kind: EntityKind = EntityKind.GENERIC
    channels: tuple[FunctionalChannel, ...] = ()
    parent_entity_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "name", _text(self.name, "name"))
        channels = tuple(self.channels)
        if len({item.channel_id for item in channels}) != len(channels):
            raise ROIFEntityError("channel_id values must be unique.")
        if any(item.entity_id != self.entity_id for item in channels):
            raise ROIFEntityError("channel.entity_id must match entity_id.")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class InfluencePlane:
    plane_id: str
    name: str
    kind: PlaneKind
    temporal_mode: TemporalMode = TemporalMode.CONTINUOUS
    active: bool = True
    intensity: float = 1.0
    timescale: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane_id", _text(self.plane_id, "plane_id"))
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(self, "intensity", _nonnegative(self.intensity, "intensity"))
        timescale = _finite(self.timescale, "timescale")
        if timescale <= 0.0:
            raise ROIFEntityError("timescale must be positive.")
        object.__setattr__(self, "timescale", timescale)
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class InfluenceOperator:
    operator_id: str
    name: str
    plane_id: str
    kind: OperatorKind
    composition: CompositionKind = CompositionKind.MULTIPLICATIVE
    source_ids: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()
    gain: float = 1.0
    threshold: float | None = None
    delay: float = 0.0
    direction: Vector3 = field(default_factory=Vector3)
    active: bool = True
    condition_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("operator_id", "name", "plane_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "source_ids", tuple(_text(x, "source_id") for x in self.source_ids)
        )
        object.__setattr__(
            self, "target_ids", tuple(_text(x, "target_id") for x in self.target_ids)
        )
        object.__setattr__(self, "gain", _finite(self.gain, "gain"))
        object.__setattr__(self, "delay", _nonnegative(self.delay, "delay"))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class OperatorComposition:
    composition_id: str
    operator_ids: tuple[str, ...]
    kind: CompositionKind
    ordered: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "composition_id", _text(self.composition_id, "composition_id")
        )
        operator_ids = tuple(_text(x, "operator_id") for x in self.operator_ids)
        if not operator_ids:
            raise ROIFEntityError("operator_ids cannot be empty.")
        object.__setattr__(self, "operator_ids", operator_ids)
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PhasicAfferentEvent:
    event_id: str
    channel_id: str
    time: float
    activation_attempt: float
    active_tension: float
    tension_rate: float = 0.0
    ib_drive: float = 0.0
    joint_afferent_drive: float = 0.0
    spindle_drive: float = 0.0
    inhibitory_gain: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "channel_id", _text(self.channel_id, "channel_id"))
        object.__setattr__(self, "time", _nonnegative(self.time, "time"))
        object.__setattr__(
            self, "activation_attempt", _unit(self.activation_attempt, "activation_attempt")
        )
        for name in (
            "active_tension",
            "ib_drive",
            "joint_afferent_drive",
            "spindle_drive",
            "inhibitory_gain",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(self, "tension_rate", _finite(self.tension_rate, "tension_rate"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def effective_gate(self) -> float:
        total = self.ib_drive + self.joint_afferent_drive + self.spindle_drive
        return max(0.0, min(1.0, 1.0 - self.inhibitory_gain * total))


@dataclass(frozen=True, slots=True)
class RepetitionMemory:
    memory_id: str
    pattern_id: str
    repetitions: int = 0
    cumulative_exposure: float = 0.0
    adaptation_level: float = 0.0
    lock_in_level: float = 0.0
    reversibility: float = 1.0
    last_time: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _text(self.memory_id, "memory_id"))
        object.__setattr__(self, "pattern_id", _text(self.pattern_id, "pattern_id"))
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions < 0
        ):
            raise ROIFEntityError("repetitions must be a nonnegative integer.")
        object.__setattr__(
            self,
            "cumulative_exposure",
            _nonnegative(self.cumulative_exposure, "cumulative_exposure"),
        )
        for name in ("adaptation_level", "lock_in_level", "reversibility"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CompensationLink:
    link_id: str
    deficient_channel_id: str
    compensator_channel_id: str
    transferred_fraction: float
    task_id: str | None = None
    active: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("link_id", "deficient_channel_id", "compensator_channel_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.deficient_channel_id == self.compensator_channel_id:
            raise ROIFEntityError("deficient and compensator channels must differ.")
        object.__setattr__(
            self,
            "transferred_fraction",
            _unit(self.transferred_fraction, "transferred_fraction"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CompensatorLockIn:
    lock_id: str
    deficient_channel_id: str
    compensator_channel_ids: tuple[str, ...]
    lock_strength: float = 0.0
    self_reinforcement: float = 0.0
    original_channel_availability: float = 1.0
    active: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lock_id", _text(self.lock_id, "lock_id"))
        object.__setattr__(
            self,
            "deficient_channel_id",
            _text(self.deficient_channel_id, "deficient_channel_id"),
        )
        compensators = tuple(
            _text(x, "compensator_channel_id")
            for x in self.compensator_channel_ids
        )
        if not compensators:
            raise ROIFEntityError("compensator_channel_ids cannot be empty.")
        object.__setattr__(self, "compensator_channel_ids", compensators)
        for name in (
            "lock_strength",
            "self_reinforcement",
            "original_channel_availability",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CascadeEvent:
    event_id: str
    time: float
    operator_id: str
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    magnitude: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "operator_id", _text(self.operator_id, "operator_id"))
        object.__setattr__(self, "time", _nonnegative(self.time, "time"))
        object.__setattr__(
            self, "source_ids", tuple(_text(x, "source_id") for x in self.source_ids)
        )
        object.__setattr__(
            self, "target_ids", tuple(_text(x, "target_id") for x in self.target_ids)
        )
        object.__setattr__(self, "magnitude", _finite(self.magnitude, "magnitude"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CycleRoot:
    cycle_id: str
    member_ids: tuple[str, ...]
    operator_ids: tuple[str, ...]
    gain: float
    persistence: float
    closure: float
    self_sustaining: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_id", _text(self.cycle_id, "cycle_id"))
        members = tuple(_text(x, "member_id") for x in self.member_ids)
        if len(members) < 2:
            raise ROIFEntityError("cycle must contain at least two members.")
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(
            self, "operator_ids", tuple(_text(x, "operator_id") for x in self.operator_ids)
        )
        object.__setattr__(self, "gain", _nonnegative(self.gain, "gain"))
        object.__setattr__(self, "persistence", _unit(self.persistence, "persistence"))
        object.__setattr__(self, "closure", _unit(self.closure, "closure"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RootAssignment:
    assignment_id: str
    role: RootRole
    target_kind: RootTargetKind
    target_id: str
    time: float
    confidence: float = 1.0
    score: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assignment_id", _text(self.assignment_id, "assignment_id")
        )
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(self, "time", _nonnegative(self.time, "time"))
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(
            self, "evidence_ids", tuple(_text(x, "evidence_id") for x in self.evidence_ids)
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RootMigration:
    migration_id: str
    from_assignment_id: str
    to_assignment_id: str
    time: float
    trigger_event_ids: tuple[str, ...] = ()
    mechanism: str = "unspecified"
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "migration_id",
            "from_assignment_id",
            "to_assignment_id",
            "mechanism",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.from_assignment_id == self.to_assignment_id:
            raise ROIFEntityError("root migration must change assignment.")
        object.__setattr__(self, "time", _nonnegative(self.time, "time"))
        object.__setattr__(
            self,
            "trigger_event_ids",
            tuple(_text(x, "trigger_event_id") for x in self.trigger_event_ids),
        )
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CounterfactualIntervention:
    intervention_id: str
    target_kind: RootTargetKind
    target_id: str
    kind: InterventionKind
    magnitude: float
    plane_id: str | None = None
    operator_id: str | None = None
    reversible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "intervention_id", _text(self.intervention_id, "intervention_id")
        )
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(self, "magnitude", _nonnegative(self.magnitude, "magnitude"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    result_id: str
    intervention_id: str
    restored_reserve: float = 0.0
    restored_availability: float = 0.0
    cycle_break_score: float = 0.0
    geometry_restoration: float = 0.0
    global_integrity_gain: float = 0.0
    harm_risk: float = 0.0
    uncertainty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _text(self.result_id, "result_id"))
        object.__setattr__(
            self, "intervention_id", _text(self.intervention_id, "intervention_id")
        )
        for name in (
            "restored_reserve",
            "restored_availability",
            "cycle_break_score",
            "geometry_restoration",
            "global_integrity_gain",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "harm_risk", _unit(self.harm_risk, "harm_risk"))
        object.__setattr__(self, "uncertainty", _unit(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def unlock_score(self) -> float:
        benefit = (
            self.restored_reserve
            + self.restored_availability
            + self.cycle_break_score
            + self.geometry_restoration
            + self.global_integrity_gain
        )
        return benefit * (1.0 - self.harm_risk) * (1.0 - self.uncertainty)


@dataclass(frozen=True, slots=True)
class ROIFSystem:
    system_id: str
    name: str
    entities: tuple[StructuralEntity, ...] = ()
    planes: tuple[InfluencePlane, ...] = ()
    operators: tuple[InfluenceOperator, ...] = ()
    compositions: tuple[OperatorComposition, ...] = ()
    compensation_links: tuple[CompensationLink, ...] = ()
    lock_ins: tuple[CompensatorLockIn, ...] = ()
    cycle_roots: tuple[CycleRoot, ...] = ()
    memories: tuple[RepetitionMemory, ...] = ()
    events: tuple[CascadeEvent, ...] = ()
    afferent_events: tuple[PhasicAfferentEvent, ...] = ()
    root_assignments: tuple[RootAssignment, ...] = ()
    root_migrations: tuple[RootMigration, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_id", _text(self.system_id, "system_id"))
        object.__setattr__(self, "name", _text(self.name, "name"))
        for name in (
            "entities",
            "planes",
            "operators",
            "compositions",
            "compensation_links",
            "lock_ins",
            "cycle_roots",
            "memories",
            "events",
            "afferent_events",
            "root_assignments",
            "root_migrations",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(
            channel.channel_id
            for entity in self.entities
            for channel in entity.channels
        )

    def channel(self, channel_id: str) -> FunctionalChannel:
        for entity in self.entities:
            for channel in entity.channels:
                if channel.channel_id == channel_id:
                    return channel
        raise KeyError(channel_id)

    def assignments_for(self, role: RootRole) -> tuple[RootAssignment, ...]:
        return tuple(item for item in self.root_assignments if item.role is role)


__all__ = [
    "ActivationAvailability",
    "CapacityState",
    "CascadeEvent",
    "ChannelState",
    "CompensationLink",
    "CompensatorLockIn",
    "CompositionKind",
    "CounterfactualIntervention",
    "CounterfactualResult",
    "CycleRoot",
    "EntityKind",
    "FunctionalChannel",
    "FunctionalRole",
    "GeometryState",
    "InfluenceOperator",
    "InfluencePlane",
    "InterventionKind",
    "OperatorComposition",
    "OperatorKind",
    "PhasicAfferentEvent",
    "PlaneKind",
    "ROIFEntityError",
    "ROIFSystem",
    "RepetitionMemory",
    "RootAssignment",
    "RootMigration",
    "RootRole",
    "RootTargetKind",
    "StructuralEntity",
    "TemporalMode",
    "Vector3",
    "new_id",
]
