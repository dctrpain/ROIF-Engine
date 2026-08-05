"""
ROIF Engine
Recursive Cascade Propagation

This module propagates system state through a state-dependent Capacity Tensor.

It is domain-independent. A cascade may represent:

- force redistribution through muscles, ligaments, fascia, nerves, or vessels;
- stress transfer through steel, concrete, timber, composites, cables, or soil;
- propagation through electrical, informational, behavioural, or control
  networks;
- any recursively coupled system represented by ROIF functional channels.

Core ROIF principles implemented here
-------------------------------------

1. A cascade is recursive:
       x(t + 1) = F(x(t), W(t), u(t))

2. The operator may be state-dependent:
       W(t) = CapacityTensor(system_state(t))

3. Compensation is initially useful but may accumulate:
       deficient channel
       -> load transfer
       -> compensator overload
       -> lock-in
       -> root migration

4. Chronic structure and acute events are distinct:
       persistent geometry/history
       -> repeated event-driven response at each activation attempt

5. Saturation, dissipation, feedback, thresholds, and delays determine whether
   influence decays, stabilizes, oscillates, or amplifies.

This module does not identify D_origin, D_fast, D_root_current, or Node*.
It produces the trajectories and event records used by:

- ``roif_root_detector.py``;
- ``roif_counterfactual.py``;
- ``roif_solver.py``.

This is a research-prototype computation layer.

Author:
    Architect (Dctr Pain)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from .roif_entities import (
    CascadeEvent,
    ChannelState,
    FunctionalChannel,
    ROIFSystem,
)
from .roif_influence import InfluenceContext
from .roif_materials import MaterialModel
from .roif_tensor import (
    CapacityTensor,
    TensorBuildConfig,
    build_capacity_tensor,
    spectral_radius,
    state_vector,
)


class ROIFCascadeError(ValueError):
    """Raised when a cascade configuration or state is invalid."""


class CascadeUpdateMode(str, Enum):
    """Discrete update rule used by the cascade."""

    LINEAR = "linear"
    LEAKY = "leaky"
    LOGISTIC = "logistic"
    TANH = "tanh"
    THRESHOLD = "threshold"
    CONSERVATIVE_TRANSFER = "conservative_transfer"


class CascadeDirection(str, Enum):
    """Interpretation of state values."""

    LOAD = "load"
    DEFICIT = "deficit"
    ACTIVATION = "activation"
    DAMAGE = "damage"
    INFORMATION = "information"
    GENERIC = "generic"


class TensorRefreshMode(str, Enum):
    """When the state-dependent tensor is rebuilt."""

    STATIC = "static"
    EVERY_STEP = "every_step"
    ON_STATE_CHANGE = "on_state_change"
    PERIODIC = "periodic"


class CascadeTerminationReason(str, Enum):
    MAX_STEPS = "max_steps"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    FAILED = "failed"
    EMPTY = "empty"


class EventKind(str, Enum):
    """Canonical event categories emitted during propagation."""

    INJECTION = "injection"
    TRANSMISSION = "transmission"
    THRESHOLD_CROSSING = "threshold_crossing"
    RESERVE_DEPLETION = "reserve_depletion"
    OVERLOAD = "overload"
    COMPENSATION = "compensation"
    LOCK_IN = "lock_in"
    RECOVERY = "recovery"
    SATURATION = "saturation"
    FAILURE = "failure"
    CONVERGENCE = "convergence"
    TENSOR_REFRESH = "tensor_refresh"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFCascadeError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFCascadeError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ROIFCascadeError(f"{name} must be finite.")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ROIFCascadeError(f"{name} must be positive.")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ROIFCascadeError(f"{name} cannot be negative.")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ROIFCascadeError(f"{name} must be within [0, 1].")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFCascadeError(f"{name} must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFCascadeError("metadata must be a mapping.")
    return MappingProxyType(dict(value))


def _readonly_vector(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ROIFCascadeError(f"{name} must be numeric.") from exc
    if array.ndim != 1:
        raise ROIFCascadeError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ROIFCascadeError(f"{name} must contain finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ROIFCascadeError(f"{name} must be numeric.") from exc
    if array.ndim != 2:
        raise ROIFCascadeError(f"{name} must be two-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ROIFCascadeError(f"{name} must contain finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    """Configuration for recursive cascade propagation."""

    steps: int = 32
    delta_time: float = 1.0
    update_mode: CascadeUpdateMode = CascadeUpdateMode.LEAKY
    direction: CascadeDirection = CascadeDirection.GENERIC
    tensor_refresh: TensorRefreshMode = TensorRefreshMode.STATIC

    dissipation: float = 0.10
    retention: float = 0.0
    external_gain: float = 1.0
    feedback_gain: float = 1.0

    lower_bound: float = 0.0
    upper_bound: float = 1.0
    threshold: float = 0.5
    logistic_midpoint: float = 0.5
    logistic_steepness: float = 8.0

    convergence_tolerance: float = 1e-8
    convergence_patience: int = 3
    divergence_threshold: float = 1e6
    tensor_refresh_period: int = 1
    tensor_refresh_tolerance: float = 1e-6

    allow_negative_state: bool = False
    clip_state: bool = True
    stop_on_failure: bool = True
    record_transmission_events: bool = True
    record_zero_events: bool = False
    update_system_channels: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps < 1
        ):
            raise ROIFCascadeError("steps must be a positive integer.")
        object.__setattr__(
            self,
            "delta_time",
            _positive(self.delta_time, "delta_time"),
        )

        enum_fields = (
            ("update_mode", self.update_mode, CascadeUpdateMode),
            ("direction", self.direction, CascadeDirection),
            ("tensor_refresh", self.tensor_refresh, TensorRefreshMode),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise ROIFCascadeError(
                    f"{name} must be {enum_type.__name__}."
                )

        object.__setattr__(
            self,
            "dissipation",
            _unit(self.dissipation, "dissipation"),
        )
        object.__setattr__(
            self,
            "retention",
            _nonnegative(self.retention, "retention"),
        )
        object.__setattr__(
            self,
            "external_gain",
            _finite(self.external_gain, "external_gain"),
        )
        object.__setattr__(
            self,
            "feedback_gain",
            _finite(self.feedback_gain, "feedback_gain"),
        )

        for name in (
            "lower_bound",
            "upper_bound",
            "threshold",
            "logistic_midpoint",
            "logistic_steepness",
            "convergence_tolerance",
            "divergence_threshold",
            "tensor_refresh_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )

        if self.lower_bound > self.upper_bound:
            raise ROIFCascadeError(
                "lower_bound cannot exceed upper_bound."
            )
        if self.logistic_steepness <= 0.0:
            raise ROIFCascadeError(
                "logistic_steepness must be positive."
            )
        if self.convergence_tolerance <= 0.0:
            raise ROIFCascadeError(
                "convergence_tolerance must be positive."
            )
        if self.divergence_threshold <= 0.0:
            raise ROIFCascadeError(
                "divergence_threshold must be positive."
            )
        if self.tensor_refresh_tolerance < 0.0:
            raise ROIFCascadeError(
                "tensor_refresh_tolerance cannot be negative."
            )

        if (
            isinstance(self.convergence_patience, bool)
            or not isinstance(self.convergence_patience, int)
            or self.convergence_patience < 1
        ):
            raise ROIFCascadeError(
                "convergence_patience must be a positive integer."
            )
        if (
            isinstance(self.tensor_refresh_period, bool)
            or not isinstance(self.tensor_refresh_period, int)
            or self.tensor_refresh_period < 1
        ):
            raise ROIFCascadeError(
                "tensor_refresh_period must be a positive integer."
            )

        for name in (
            "allow_negative_state",
            "clip_state",
            "stop_on_failure",
            "record_transmission_events",
            "record_zero_events",
            "update_system_channels",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ROIFCascadeError(f"{name} must be bool.")

        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CascadeInjection:
    """External input applied at a specified step."""

    injection_id: str
    step: int
    values: Mapping[str, float]
    gain: float = 1.0
    persistent: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "injection_id",
            _text(self.injection_id, "injection_id"),
        )
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ROIFCascadeError(
                "step must be a nonnegative integer."
            )
        if not isinstance(self.values, Mapping):
            raise ROIFCascadeError("values must be a mapping.")
        object.__setattr__(
            self,
            "values",
            MappingProxyType(
                {
                    _text(str(key), "channel_id"): _finite(
                        value,
                        f"values[{key!r}]",
                    )
                    for key, value in self.values.items()
                }
            ),
        )
        object.__setattr__(self, "gain", _finite(self.gain, "gain"))
        if not isinstance(self.persistent, bool):
            raise ROIFCascadeError("persistent must be bool.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CascadeStep:
    """Complete immutable record of one propagation step."""

    step: int
    time: float
    state_before: np.ndarray
    external_input: np.ndarray
    propagated_input: np.ndarray
    feedback_input: np.ndarray
    state_after: np.ndarray
    state_delta: np.ndarray
    tensor_matrix: np.ndarray
    spectral_radius: float
    maximum_value: float
    minimum_value: float
    l1_norm: float
    l2_norm: float
    converged: bool = False
    failed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ROIFCascadeError(
                "step must be a nonnegative integer."
            )
        object.__setattr__(self, "time", _nonnegative(self.time, "time"))

        vector_fields = (
            "state_before",
            "external_input",
            "propagated_input",
            "feedback_input",
            "state_after",
            "state_delta",
        )
        vectors: dict[str, np.ndarray] = {}
        for name in vector_fields:
            vectors[name] = _readonly_vector(getattr(self, name), name)

        lengths = {vector.shape for vector in vectors.values()}
        if len(lengths) != 1:
            raise ROIFCascadeError(
                "all cascade step vectors must have the same shape."
            )
        for name, vector in vectors.items():
            object.__setattr__(self, name, vector)

        matrix = _readonly_matrix(
            self.tensor_matrix,
            "tensor_matrix",
        )
        node_count = vectors["state_before"].shape[0]
        if matrix.shape != (node_count, node_count):
            raise ROIFCascadeError(
                "tensor_matrix shape must match state size."
            )
        object.__setattr__(self, "tensor_matrix", matrix)

        for name in (
            "spectral_radius",
            "maximum_value",
            "minimum_value",
            "l1_norm",
            "l2_norm",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )

        if not isinstance(self.converged, bool):
            raise ROIFCascadeError("converged must be bool.")
        if not isinstance(self.failed, bool):
            raise ROIFCascadeError("failed must be bool.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CascadeTrajectory:
    """Full result of a cascade simulation."""

    channel_ids: tuple[str, ...]
    initial_state: np.ndarray
    final_state: np.ndarray
    steps: tuple[CascadeStep, ...]
    events: tuple[CascadeEvent, ...]
    tensors: tuple[CapacityTensor, ...]
    termination_reason: CascadeTerminationReason
    converged: bool
    failed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(_text(value, "channel_id") for value in self.channel_ids)
        if len(ids) != len(set(ids)):
            raise ROIFCascadeError("channel_ids must be unique.")
        object.__setattr__(self, "channel_ids", ids)

        initial = _readonly_vector(self.initial_state, "initial_state")
        final = _readonly_vector(self.final_state, "final_state")
        if initial.shape != final.shape:
            raise ROIFCascadeError(
                "initial_state and final_state must have the same shape."
            )
        if initial.shape[0] != len(ids):
            raise ROIFCascadeError(
                "state size must match channel_ids."
            )
        object.__setattr__(self, "initial_state", initial)
        object.__setattr__(self, "final_state", final)
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "tensors", tuple(self.tensors))

        if not isinstance(
            self.termination_reason,
            CascadeTerminationReason,
        ):
            raise ROIFCascadeError(
                "termination_reason must be CascadeTerminationReason."
            )
        if not isinstance(self.converged, bool):
            raise ROIFCascadeError("converged must be bool.")
        if not isinstance(self.failed, bool):
            raise ROIFCascadeError("failed must be bool.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def state_history(self) -> np.ndarray:
        if not self.steps:
            history = np.expand_dims(self.initial_state, axis=0)
        else:
            history = np.vstack(
                [self.initial_state]
                + [step.state_after for step in self.steps]
            )
        history = np.asarray(history, dtype=float)
        history.setflags(write=False)
        return history

    def index(self, channel_id: str) -> int:
        try:
            return self.channel_ids.index(channel_id)
        except ValueError as exc:
            raise KeyError(channel_id) from exc

    def series(self, channel_id: str) -> np.ndarray:
        values = self.state_history[:, self.index(channel_id)]
        result = np.array(values, copy=True)
        result.setflags(write=False)
        return result

    def peak(self, channel_id: str) -> float:
        return float(np.max(self.series(channel_id)))

    def time_to_peak(self, channel_id: str) -> int:
        return int(np.argmax(self.series(channel_id)))

    def total_exposure(self, channel_id: str) -> float:
        return float(np.sum(np.abs(self.series(channel_id))))

    def first_threshold_crossing(
        self,
        channel_id: str,
        threshold: float,
    ) -> int | None:
        threshold = _finite(threshold, "threshold")
        series = self.series(channel_id)
        indices = np.flatnonzero(series >= threshold)
        if indices.size == 0:
            return None
        return int(indices[0])


TensorBuilder = Callable[
    [
        ROIFSystem,
        InfluenceContext,
        Mapping[str, MaterialModel],
        TensorBuildConfig,
    ],
    CapacityTensor,
]


def _default_tensor_builder(
    system: ROIFSystem,
    context: InfluenceContext,
    materials: Mapping[str, MaterialModel],
    config: TensorBuildConfig,
) -> CapacityTensor:
    return build_capacity_tensor(
        system,
        context=context,
        materials=materials,
        config=config,
    )


def _channels(system: ROIFSystem) -> tuple[FunctionalChannel, ...]:
    result = tuple(
        channel
        for entity in system.entities
        for channel in entity.channels
    )
    if not result:
        raise ROIFCascadeError(
            "system must contain at least one FunctionalChannel."
        )
    return result


def _channel_map(
    system: ROIFSystem,
) -> Mapping[str, FunctionalChannel]:
    return MappingProxyType(
        {
            channel.channel_id: channel
            for channel in _channels(system)
        }
    )


def default_initial_state(
    system: ROIFSystem,
    *,
    direction: CascadeDirection = CascadeDirection.GENERIC,
) -> Mapping[str, float]:
    """Build a deterministic initial state from channel properties."""

    values: dict[str, float] = {}
    for channel in _channels(system):
        if direction is CascadeDirection.LOAD:
            value = channel.capacity_state.load
        elif direction is CascadeDirection.DEFICIT:
            value = max(
                0.0,
                1.0 - channel.capacity_state.reserve_fraction,
            )
        elif direction is CascadeDirection.ACTIVATION:
            value = channel.activation.effective
        elif direction is CascadeDirection.DAMAGE:
            value = max(
                0.0,
                1.0 - channel.history_factor,
            )
        elif direction in (
            CascadeDirection.INFORMATION,
            CascadeDirection.GENERIC,
        ):
            value = channel.effective_scalar_output
        else:
            raise ROIFCascadeError(
                "unsupported CascadeDirection."
            )
        values[channel.channel_id] = float(value)
    return MappingProxyType(values)


def injection_vector(
    injection: CascadeInjection,
    channel_ids: Sequence[str],
) -> np.ndarray:
    """Convert one injection to a positional state vector."""

    values = {
        channel_id: injection.gain * value
        for channel_id, value in injection.values.items()
    }
    return state_vector(values, channel_ids)


def active_injection_vector(
    injections: Sequence[CascadeInjection],
    step: int,
    channel_ids: Sequence[str],
) -> np.ndarray:
    """Combine all injections active at one simulation step."""

    result = np.zeros(len(channel_ids), dtype=float)
    for injection in injections:
        active = injection.step == step or (
            injection.persistent and injection.step <= step
        )
        if active:
            result += injection_vector(injection, channel_ids)
    result.setflags(write=False)
    return result


def _linear_update(
    state: np.ndarray,
    propagated: np.ndarray,
    feedback: np.ndarray,
    external: np.ndarray,
    config: CascadeConfig,
) -> np.ndarray:
    return (
        config.retention * state
        + config.feedback_gain * propagated
        + feedback
        + config.external_gain * external
    )


def _leaky_update(
    state: np.ndarray,
    propagated: np.ndarray,
    feedback: np.ndarray,
    external: np.ndarray,
    config: CascadeConfig,
) -> np.ndarray:
    retained = (1.0 - config.dissipation) * state
    incoming = (
        config.feedback_gain * propagated
        + feedback
        + config.external_gain * external
    )
    return config.retention * state + retained + incoming


def _logistic_transform(
    values: np.ndarray,
    config: CascadeConfig,
) -> np.ndarray:
    exponent = -config.logistic_steepness * (
        values - config.logistic_midpoint
    )
    exponent = np.clip(exponent, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(exponent))


def update_state(
    state: np.ndarray,
    propagated: np.ndarray,
    feedback: np.ndarray,
    external: np.ndarray,
    config: CascadeConfig,
) -> np.ndarray:
    """Apply the configured nonlinear cascade update."""

    if config.update_mode is CascadeUpdateMode.LINEAR:
        updated = _linear_update(
            state,
            propagated,
            feedback,
            external,
            config,
        )

    elif config.update_mode is CascadeUpdateMode.LEAKY:
        updated = _leaky_update(
            state,
            propagated,
            feedback,
            external,
            config,
        )

    elif config.update_mode is CascadeUpdateMode.LOGISTIC:
        raw = _leaky_update(
            state,
            propagated,
            feedback,
            external,
            config,
        )
        updated = _logistic_transform(raw, config)

    elif config.update_mode is CascadeUpdateMode.TANH:
        raw = _leaky_update(
            state,
            propagated,
            feedback,
            external,
            config,
        )
        updated = np.tanh(raw)

    elif config.update_mode is CascadeUpdateMode.THRESHOLD:
        raw = _leaky_update(
            state,
            propagated,
            feedback,
            external,
            config,
        )
        updated = np.where(
            raw >= config.threshold,
            raw,
            0.0,
        )

    elif (
        config.update_mode
        is CascadeUpdateMode.CONSERVATIVE_TRANSFER
    ):
        incoming = (
            config.feedback_gain * propagated
            + feedback
            + config.external_gain * external
        )
        outgoing = np.sum(np.abs(propagated))
        total_before = np.sum(state) + np.sum(external)
        updated = (1.0 - config.dissipation) * state + incoming
        total_after = np.sum(updated)
        if (
            abs(total_after) > 1e-15
            and abs(total_before) > 1e-15
            and outgoing > 0.0
        ):
            updated *= total_before / total_after

    else:
        raise ROIFCascadeError(
            "unsupported CascadeUpdateMode."
        )

    if not config.allow_negative_state:
        updated = np.maximum(updated, 0.0)
    if config.clip_state:
        updated = np.clip(
            updated,
            config.lower_bound,
            config.upper_bound,
        )
    return np.asarray(updated, dtype=float)


def _feedback_vector(
    state: np.ndarray,
    context: InfluenceContext,
    channel_ids: Sequence[str],
) -> np.ndarray:
    """
    Resolve optional external feedback.

    Context keys:
        feedback.<channel_id>
    """

    result = np.asarray(
        [
            context.scalar(f"feedback.{channel_id}", 0.0)
            for channel_id in channel_ids
        ],
        dtype=float,
    )
    result.setflags(write=False)
    return result


def _event(
    *,
    event_id: str,
    time: float,
    operator_id: str,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    magnitude: float,
    kind: EventKind,
    metadata: Mapping[str, Any] | None = None,
) -> CascadeEvent:
    return CascadeEvent(
        event_id=event_id,
        time=time,
        operator_id=operator_id,
        source_ids=tuple(source_ids),
        target_ids=tuple(target_ids),
        magnitude=magnitude,
        metadata={
            "event_kind": kind.value,
            **dict(metadata or {}),
        },
    )


def transmission_events(
    tensor: CapacityTensor,
    state: np.ndarray,
    step: int,
    time: float,
    *,
    include_zero: bool = False,
) -> tuple[CascadeEvent, ...]:
    """Create pairwise transmission events from the current tensor."""

    events: list[CascadeEvent] = []
    for target_index, target_id in enumerate(tensor.channel_ids):
        for source_index, source_id in enumerate(tensor.channel_ids):
            coupling = float(
                tensor.matrix[target_index, source_index]
            )
            magnitude = coupling * float(state[source_index])
            if (
                not include_zero
                and math.isclose(magnitude, 0.0, abs_tol=1e-15)
            ):
                continue
            events.append(
                _event(
                    event_id=(
                        f"transmission_{step}_{source_id}_{target_id}"
                    ),
                    time=time,
                    operator_id="capacity_tensor",
                    source_ids=(source_id,),
                    target_ids=(target_id,),
                    magnitude=magnitude,
                    kind=EventKind.TRANSMISSION,
                    metadata={
                        "coupling": coupling,
                        "step": step,
                    },
                )
            )
    return tuple(events)


def threshold_events(
    before: np.ndarray,
    after: np.ndarray,
    channel_ids: Sequence[str],
    step: int,
    time: float,
    config: CascadeConfig,
) -> tuple[CascadeEvent, ...]:
    """Emit threshold, saturation, and recovery events."""

    events: list[CascadeEvent] = []
    for index, channel_id in enumerate(channel_ids):
        left = float(before[index])
        right = float(after[index])

        if left < config.threshold <= right:
            events.append(
                _event(
                    event_id=f"threshold_{step}_{channel_id}",
                    time=time,
                    operator_id="cascade_threshold",
                    source_ids=(channel_id,),
                    target_ids=(channel_id,),
                    magnitude=right,
                    kind=EventKind.THRESHOLD_CROSSING,
                    metadata={"step": step},
                )
            )

        if (
            config.clip_state
            and right >= config.upper_bound
            and left < config.upper_bound
        ):
            events.append(
                _event(
                    event_id=f"saturation_{step}_{channel_id}",
                    time=time,
                    operator_id="cascade_saturation",
                    source_ids=(channel_id,),
                    target_ids=(channel_id,),
                    magnitude=right,
                    kind=EventKind.SATURATION,
                    metadata={"step": step},
                )
            )

        if left >= config.threshold > right:
            events.append(
                _event(
                    event_id=f"recovery_{step}_{channel_id}",
                    time=time,
                    operator_id="cascade_recovery",
                    source_ids=(channel_id,),
                    target_ids=(channel_id,),
                    magnitude=right,
                    kind=EventKind.RECOVERY,
                    metadata={"step": step},
                )
            )

    return tuple(events)


def reserve_events(
    system: ROIFSystem,
    state: np.ndarray,
    channel_ids: Sequence[str],
    step: int,
    time: float,
    direction: CascadeDirection,
) -> tuple[CascadeEvent, ...]:
    """
    Emit reserve depletion and overload events.

    For ``LOAD`` state, the simulated value is compared with channel capacity.
    For normalized deficit/damage states, values >= 1 represent depletion.
    """

    channel_map = _channel_map(system)
    events: list[CascadeEvent] = []

    for index, channel_id in enumerate(channel_ids):
        value = float(state[index])
        channel = channel_map[channel_id]

        if direction is CascadeDirection.LOAD:
            capacity = channel.capacity_state.capacity
            reserve = capacity - value
            if reserve <= 0.0:
                events.append(
                    _event(
                        event_id=f"overload_{step}_{channel_id}",
                        time=time,
                        operator_id="capacity_depletion",
                        source_ids=(channel_id,),
                        target_ids=(channel_id,),
                        magnitude=-reserve,
                        kind=EventKind.OVERLOAD,
                        metadata={
                            "capacity": capacity,
                            "simulated_load": value,
                            "step": step,
                        },
                    )
                )
            elif reserve / capacity <= (
                channel.capacity_state.critical_reserve_fraction
            ):
                events.append(
                    _event(
                        event_id=f"reserve_{step}_{channel_id}",
                        time=time,
                        operator_id="reserve_threshold",
                        source_ids=(channel_id,),
                        target_ids=(channel_id,),
                        magnitude=reserve,
                        kind=EventKind.RESERVE_DEPLETION,
                        metadata={
                            "capacity": capacity,
                            "simulated_load": value,
                            "step": step,
                        },
                    )
                )

        elif direction in (
            CascadeDirection.DEFICIT,
            CascadeDirection.DAMAGE,
        ) and value >= 1.0:
            events.append(
                _event(
                    event_id=f"failure_{step}_{channel_id}",
                    time=time,
                    operator_id="normalized_failure",
                    source_ids=(channel_id,),
                    target_ids=(channel_id,),
                    magnitude=value,
                    kind=EventKind.FAILURE,
                    metadata={"step": step},
                )
            )

    return tuple(events)


def _should_refresh_tensor(
    mode: TensorRefreshMode,
    *,
    step: int,
    current_state: np.ndarray,
    previous_state: np.ndarray,
    config: CascadeConfig,
) -> bool:
    if mode is TensorRefreshMode.STATIC:
        return False
    if mode is TensorRefreshMode.EVERY_STEP:
        return True
    if mode is TensorRefreshMode.PERIODIC:
        return step % config.tensor_refresh_period == 0
    if mode is TensorRefreshMode.ON_STATE_CHANGE:
        delta = float(
            np.max(np.abs(current_state - previous_state))
        )
        return delta >= config.tensor_refresh_tolerance
    raise ROIFCascadeError("unsupported TensorRefreshMode.")


def system_with_simulated_state(
    system: ROIFSystem,
    state: Sequence[float],
    *,
    direction: CascadeDirection,
) -> ROIFSystem:
    """
    Return a new ROIFSystem whose channels reflect the simulated state.

    This adapter enables state-dependent tensor rebuilding. It does not mutate
    the original system.
    """

    values = state_vector(state, system.channel_ids)
    value_by_id = {
        channel_id: float(values[index])
        for index, channel_id in enumerate(system.channel_ids)
    }

    new_entities = []
    for entity in system.entities:
        channels = []
        for channel in entity.channels:
            value = value_by_id[channel.channel_id]

            if direction is CascadeDirection.LOAD:
                new_capacity = replace(
                    channel.capacity_state,
                    load=max(0.0, value),
                )
                updated = replace(
                    channel,
                    capacity_state=new_capacity,
                )

            elif direction is CascadeDirection.ACTIVATION:
                updated = replace(
                    channel,
                    activation=replace(
                        channel.activation,
                        command=max(0.0, min(1.0, value)),
                    ),
                )

            elif direction is CascadeDirection.DEFICIT:
                updated = replace(
                    channel,
                    history_factor=max(0.0, 1.0 - value),
                )

            elif direction is CascadeDirection.DAMAGE:
                updated = replace(
                    channel,
                    history_factor=max(0.0, 1.0 - value),
                    state=(
                        ChannelState.FAILED
                        if value >= 1.0
                        else channel.state
                    ),
                )

            else:
                updated = replace(
                    channel,
                    baseline_output=max(0.0, value),
                )

            channels.append(updated)

        new_entities.append(
            replace(entity, channels=tuple(channels))
        )

    return replace(system, entities=tuple(new_entities))


def run_cascade(
    system: ROIFSystem,
    *,
    initial_state: Sequence[float] | Mapping[str, float] | None = None,
    injections: Sequence[CascadeInjection] = (),
    context: InfluenceContext | None = None,
    materials: Mapping[str, MaterialModel] | None = None,
    tensor_config: TensorBuildConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    tensor_builder: TensorBuilder | None = None,
) -> CascadeTrajectory:
    """Run a deterministic recursive cascade simulation."""

    if not isinstance(system, ROIFSystem):
        raise TypeError("system must be ROIFSystem.")

    cascade_config = cascade_config or CascadeConfig()
    tensor_config = tensor_config or TensorBuildConfig()
    context = context or InfluenceContext(
        delta_time=cascade_config.delta_time
    )
    materials = materials or {}
    tensor_builder = tensor_builder or _default_tensor_builder
    injections = tuple(injections)

    channel_ids = system.channel_ids
    if not channel_ids:
        return CascadeTrajectory(
            channel_ids=(),
            initial_state=np.zeros(0),
            final_state=np.zeros(0),
            steps=(),
            events=(),
            tensors=(),
            termination_reason=CascadeTerminationReason.EMPTY,
            converged=True,
            failed=False,
        )

    if initial_state is None:
        initial_state = default_initial_state(
            system,
            direction=cascade_config.direction,
        )

    state = np.array(
        state_vector(initial_state, channel_ids),
        dtype=float,
        copy=True,
    )
    if not cascade_config.allow_negative_state:
        state = np.maximum(state, 0.0)
    if cascade_config.clip_state:
        state = np.clip(
            state,
            cascade_config.lower_bound,
            cascade_config.upper_bound,
        )
    initial = np.array(state, copy=True)

    working_system = system_with_simulated_state(
        system,
        state,
        direction=cascade_config.direction,
    ) if cascade_config.update_system_channels else system

    tensor = tensor_builder(
        working_system,
        context,
        materials,
        tensor_config,
    )
    tensors: list[CapacityTensor] = [tensor]
    steps: list[CascadeStep] = []
    events: list[CascadeEvent] = []

    # Diagnose the supplied initial state before the first propagation step.
    # An initial overload is recorded as an event, but adaptive systems are
    # still allowed to redistribute the cascade during subsequent steps.
    initial_state_events = reserve_events(
        working_system,
        state,
        channel_ids,
        -1,
        context.time,
        cascade_config.direction,
    )
    events.extend(initial_state_events)

    convergence_count = 0
    converged = False
    failed = False
    termination = CascadeTerminationReason.MAX_STEPS
    previous_for_refresh = np.array(state, copy=True)

    for step_index in range(cascade_config.steps):
        time = context.time + step_index * cascade_config.delta_time

        step_context = InfluenceContext(
            time=time,
            delta_time=cascade_config.delta_time,
            task_id=context.task_id,
            scalar_state=context.scalar_state,
            vector_state=context.vector_state,
            conditions=context.conditions,
            events=context.events,
            metadata={
                **dict(context.metadata),
                "cascade_step": step_index,
            },
        )

        if step_index > 0 and _should_refresh_tensor(
            cascade_config.tensor_refresh,
            step=step_index,
            current_state=state,
            previous_state=previous_for_refresh,
            config=cascade_config,
        ):
            previous_for_refresh = np.array(state, copy=True)
            working_system = (
                system_with_simulated_state(
                    system,
                    state,
                    direction=cascade_config.direction,
                )
                if cascade_config.update_system_channels
                else system
            )
            tensor = tensor_builder(
                working_system,
                step_context,
                materials,
                tensor_config,
            )
            tensors.append(tensor)
            events.append(
                _event(
                    event_id=f"tensor_refresh_{step_index}",
                    time=time,
                    operator_id="tensor_refresh",
                    source_ids=channel_ids,
                    target_ids=channel_ids,
                    magnitude=tensor.spectral_radius(),
                    kind=EventKind.TENSOR_REFRESH,
                    metadata={"step": step_index},
                )
            )

        before = np.array(state, copy=True)
        external = np.array(
            active_injection_vector(
                injections,
                step_index,
                channel_ids,
            ),
            copy=True,
        )
        propagated = np.asarray(
            tensor.matrix @ before,
            dtype=float,
        )
        feedback = np.array(
            _feedback_vector(
                before,
                step_context,
                channel_ids,
            ),
            copy=True,
        )

        after = update_state(
            before,
            propagated,
            feedback,
            external,
            cascade_config,
        )
        delta = after - before

        max_abs_delta = float(np.max(np.abs(delta)))
        if max_abs_delta <= cascade_config.convergence_tolerance:
            convergence_count += 1
        else:
            convergence_count = 0

        step_converged = (
            convergence_count
            >= cascade_config.convergence_patience
        )

        current_failed = bool(
            np.any(~np.isfinite(after))
            or np.max(np.abs(after))
            >= cascade_config.divergence_threshold
        )

        reserve_step_events = reserve_events(
            working_system,
            after,
            channel_ids,
            step_index,
            time,
            cascade_config.direction,
        )
        if any(
            event.metadata.get("event_kind")
            in {
                EventKind.OVERLOAD.value,
                EventKind.FAILURE.value,
            }
            for event in reserve_step_events
        ):
            current_failed = True

        if cascade_config.record_transmission_events:
            events.extend(
                transmission_events(
                    tensor,
                    before,
                    step_index,
                    time,
                    include_zero=cascade_config.record_zero_events,
                )
            )

        for injection in injections:
            active = injection.step == step_index or (
                injection.persistent
                and injection.step <= step_index
            )
            if active:
                events.append(
                    _event(
                        event_id=(
                            f"injection_{step_index}_{injection.injection_id}"
                        ),
                        time=time,
                        operator_id="external_injection",
                        source_ids=(injection.injection_id,),
                        target_ids=tuple(injection.values),
                        magnitude=float(
                            np.sum(
                                np.abs(
                                    injection_vector(
                                        injection,
                                        channel_ids,
                                    )
                                )
                            )
                        ),
                        kind=EventKind.INJECTION,
                        metadata={"step": step_index},
                    )
                )

        events.extend(
            threshold_events(
                before,
                after,
                channel_ids,
                step_index,
                time,
                cascade_config,
            )
        )
        events.extend(reserve_step_events)

        if step_converged:
            events.append(
                _event(
                    event_id=f"convergence_{step_index}",
                    time=time,
                    operator_id="cascade_convergence",
                    source_ids=channel_ids,
                    target_ids=channel_ids,
                    magnitude=max_abs_delta,
                    kind=EventKind.CONVERGENCE,
                    metadata={"step": step_index},
                )
            )

        radius = spectral_radius(tensor.matrix)
        step_record = CascadeStep(
            step=step_index,
            time=time,
            state_before=before,
            external_input=external,
            propagated_input=propagated,
            feedback_input=feedback,
            state_after=after,
            state_delta=delta,
            tensor_matrix=tensor.matrix,
            spectral_radius=radius,
            maximum_value=float(np.max(after)),
            minimum_value=float(np.min(after)),
            l1_norm=float(np.linalg.norm(after, ord=1)),
            l2_norm=float(np.linalg.norm(after, ord=2)),
            converged=step_converged,
            failed=current_failed,
            metadata={
                "tensor_index": len(tensors) - 1,
                "max_abs_delta": max_abs_delta,
            },
        )
        steps.append(step_record)
        state = np.array(after, copy=True)

        if current_failed:
            failed = True
            termination = (
                CascadeTerminationReason.DIVERGED
                if np.max(np.abs(after))
                >= cascade_config.divergence_threshold
                else CascadeTerminationReason.FAILED
            )
            if cascade_config.stop_on_failure:
                break

        if step_converged:
            converged = True
            termination = CascadeTerminationReason.CONVERGED
            break

    return CascadeTrajectory(
        channel_ids=channel_ids,
        initial_state=initial,
        final_state=state,
        steps=tuple(steps),
        events=tuple(events),
        tensors=tuple(tensors),
        termination_reason=termination,
        converged=converged,
        failed=failed,
        metadata={
            "system_id": system.system_id,
            "configured_steps": cascade_config.steps,
            "completed_steps": len(steps),
            "tensor_refresh_mode": (
                cascade_config.tensor_refresh.value
            ),
            "update_mode": cascade_config.update_mode.value,
            "direction": cascade_config.direction.value,
            "final_l1_norm": float(
                np.linalg.norm(state, ord=1)
            ),
            "final_l2_norm": float(
                np.linalg.norm(state, ord=2)
            ),
        },
    )


def trajectory_difference(
    left: CascadeTrajectory,
    right: CascadeTrajectory,
) -> np.ndarray:
    """Return final-state difference for matching channel axes."""

    if left.channel_ids != right.channel_ids:
        raise ROIFCascadeError(
            "trajectories must use the same channel order."
        )
    result = np.asarray(
        left.final_state - right.final_state,
        dtype=float,
    )
    result.setflags(write=False)
    return result


def trajectory_summary(
    trajectory: CascadeTrajectory,
) -> Mapping[str, Any]:
    """Return a compact immutable cascade summary."""

    history = trajectory.state_history
    return MappingProxyType(
        {
            "channel_count": len(trajectory.channel_ids),
            "step_count": trajectory.step_count,
            "event_count": len(trajectory.events),
            "tensor_count": len(trajectory.tensors),
            "termination_reason": (
                trajectory.termination_reason.value
            ),
            "converged": trajectory.converged,
            "failed": trajectory.failed,
            "initial_l1_norm": float(
                np.linalg.norm(
                    trajectory.initial_state,
                    ord=1,
                )
            ),
            "final_l1_norm": float(
                np.linalg.norm(
                    trajectory.final_state,
                    ord=1,
                )
            ),
            "peak_value": (
                float(np.max(history))
                if history.size
                else 0.0
            ),
            "total_exposure": (
                float(np.sum(np.abs(history)))
                if history.size
                else 0.0
            ),
        }
    )


__all__ = [
    "CascadeConfig",
    "CascadeDirection",
    "CascadeInjection",
    "CascadeStep",
    "CascadeTerminationReason",
    "CascadeTrajectory",
    "CascadeUpdateMode",
    "EventKind",
    "ROIFCascadeError",
    "TensorBuilder",
    "TensorRefreshMode",
    "active_injection_vector",
    "default_initial_state",
    "injection_vector",
    "reserve_events",
    "run_cascade",
    "system_with_simulated_state",
    "threshold_events",
    "trajectory_difference",
    "trajectory_summary",
    "transmission_events",
    "update_state",
]

