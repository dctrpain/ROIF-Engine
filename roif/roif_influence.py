"""
ROIF Engine
Influence Evaluation Layer

This module implements the first executable layer above ``roif_entities``.

It answers one question:

    Given an InfluencePlane and one or more InfluenceOperators,
    how is the current system state transformed?

The module does not yet propagate a complete cascade across time. That work
belongs to ``roif_cascade.py``. Here we define the operator algebra used by
that future solver.

Implemented capabilities
------------------------

- scalar and vector influence evaluation;
- additive, subtractive, multiplicative, divisive, gating, threshold,
  activation, inhibition, load transfer, compression, mobilization,
  fixation, rotation, stabilization, adaptation, and restoration operators;
- state-dependent plane intensity;
- static, continuous, phasic, periodic, delayed, cumulative, and
  event-driven temporal modes;
- sequential, additive, multiplicative, conditional, vector-sum,
  minimum, and maximum operator composition;
- immutable state snapshots;
- operator traces suitable for later cascade reconstruction;
- safe handling of inactive planes, inactive operators, delays, thresholds,
  and missing state values;
- direct transformation of FunctionalChannel availability, geometry,
  capacity/load, and history factor.

Important ROIF distinction
--------------------------

An InfluencePlane describes the transformation context.
An InfluenceOperator describes the action inside that context.

For example:

    plane:
        mechanical geometry

    operators:
        shorten hamstrings
        rotate pelvis posteriorly
        reduce gluteal geometric access
        transfer load to lumbar stabilizers

The same structural entities may therefore interact differently in different
planes.

This is a research-prototype computation layer. It does not diagnose or
recommend treatment.

Author:
    Architect (Dctr Pain)

License:
    See project license.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any
import math

from .roif_entities import (
    ActivationAvailability,
    CapacityState,
    CompositionKind,
    FunctionalChannel,
    GeometryState,
    InfluenceOperator,
    InfluencePlane,
    OperatorComposition,
    OperatorKind,
    PlaneKind,
    ROIFEntityError,
    TemporalMode,
    Vector3,
)


class ROIFInfluenceError(ROIFEntityError):
    """Raised when an influence operation cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class InfluenceContext:
    """
    Immutable runtime context for one influence evaluation.

    ``scalar_state`` is a general-purpose state map. Keys may represent
    reserve, load, angle, tension, task demand, weather intensity, or any
    other scalar quantity understood by a domain adapter.

    ``conditions`` controls conditional operators and event-driven planes.

    ``events`` stores event identifiers active at the current step.
    """

    time: float = 0.0
    delta_time: float = 1.0
    task_id: str | None = None
    scalar_state: Mapping[str, float] = field(default_factory=dict)
    vector_state: Mapping[str, Vector3] = field(default_factory=dict)
    conditions: Mapping[str, bool] = field(default_factory=dict)
    events: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _finite(self.time, "time")
        delta_time = _finite(self.delta_time, "delta_time")
        if time < 0.0:
            raise ROIFInfluenceError("time cannot be negative.")
        if delta_time <= 0.0:
            raise ROIFInfluenceError("delta_time must be positive.")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "delta_time", delta_time)

        if self.task_id is not None:
            object.__setattr__(
                self,
                "task_id",
                _nonempty_text(self.task_id, "task_id"),
            )

        object.__setattr__(
            self,
            "scalar_state",
            MappingProxyType(
                {
                    str(key): _finite(value, f"scalar_state[{key!r}]")
                    for key, value in self.scalar_state.items()
                }
            ),
        )

        vector_state: dict[str, Vector3] = {}
        for key, value in self.vector_state.items():
            if not isinstance(value, Vector3):
                raise ROIFInfluenceError(
                    f"vector_state[{key!r}] must be Vector3."
                )
            vector_state[str(key)] = value
        object.__setattr__(
            self,
            "vector_state",
            MappingProxyType(vector_state),
        )

        object.__setattr__(
            self,
            "conditions",
            MappingProxyType(
                {
                    str(key): bool(value)
                    for key, value in self.conditions.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "events",
            frozenset(str(value) for value in self.events),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )

    def scalar(self, key: str, default: float = 0.0) -> float:
        return self.scalar_state.get(key, default)

    def vector(self, key: str, default: Vector3 | None = None) -> Vector3:
        if key in self.vector_state:
            return self.vector_state[key]
        return default if default is not None else Vector3()

    def condition(self, key: str | None) -> bool:
        if key is None:
            return True
        return bool(self.conditions.get(key, False))


@dataclass(frozen=True, slots=True)
class InfluenceValue:
    """
    Scalar-vector pair transformed by influence operators.

    Most operators act on ``scalar``. Vector operators may additionally
    transform ``vector``. The scalar can be interpreted as load, reserve,
    activation availability, intensity, or any other domain-specific value.
    """

    scalar: float = 0.0
    vector: Vector3 = field(default_factory=Vector3)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scalar",
            _finite(self.scalar, "scalar"),
        )
        if not isinstance(self.vector, Vector3):
            raise ROIFInfluenceError("vector must be Vector3.")
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class OperatorApplication:
    """Trace of one operator evaluation."""

    operator_id: str
    plane_id: str
    applied: bool
    reason: str
    input_value: InfluenceValue
    output_value: InfluenceValue
    effective_gain: float
    time: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator_id",
            _nonempty_text(self.operator_id, "operator_id"),
        )
        object.__setattr__(
            self,
            "plane_id",
            _nonempty_text(self.plane_id, "plane_id"),
        )
        object.__setattr__(
            self,
            "reason",
            _nonempty_text(self.reason, "reason"),
        )
        object.__setattr__(
            self,
            "effective_gain",
            _finite(self.effective_gain, "effective_gain"),
        )
        object.__setattr__(
            self,
            "time",
            _nonnegative(self.time, "time"),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class CompositionApplication:
    """Trace of a complete operator composition."""

    composition_id: str
    kind: CompositionKind
    input_value: InfluenceValue
    output_value: InfluenceValue
    operator_traces: tuple[OperatorApplication, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "composition_id",
            _nonempty_text(
                self.composition_id,
                "composition_id",
            ),
        )
        if not isinstance(self.kind, CompositionKind):
            raise ROIFInfluenceError(
                "kind must be CompositionKind."
            )
        object.__setattr__(
            self,
            "operator_traces",
            tuple(self.operator_traces),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ChannelInfluenceResult:
    """Result of applying influence operators to a FunctionalChannel."""

    channel_before: FunctionalChannel
    channel_after: FunctionalChannel
    applications: tuple[OperatorApplication, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.channel_before, FunctionalChannel):
            raise ROIFInfluenceError(
                "channel_before must be FunctionalChannel."
            )
        if not isinstance(self.channel_after, FunctionalChannel):
            raise ROIFInfluenceError(
                "channel_after must be FunctionalChannel."
            )
        if (
            self.channel_before.channel_id
            != self.channel_after.channel_id
        ):
            raise ROIFInfluenceError(
                "channel identity cannot change during influence evaluation."
            )
        object.__setattr__(
            self,
            "applications",
            tuple(self.applications),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFInfluenceError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFInfluenceError(
            f"{name} must be numeric."
        ) from exc
    if not math.isfinite(result):
        raise ROIFInfluenceError(
            f"{name} must be finite."
        )
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ROIFInfluenceError(
            f"{name} cannot be negative."
        )
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ROIFInfluenceError(
            f"{name} must be within [0, 1]."
        )
    return result


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFInfluenceError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFInfluenceError(
            "metadata must be a mapping."
        )
    return MappingProxyType(dict(value))


def plane_activity(
    plane: InfluencePlane,
    context: InfluenceContext,
) -> float:
    """
    Return the effective activity of a plane at the current time.

    The result is nonnegative and may exceed one when plane intensity is
    explicitly configured above one.
    """

    if not isinstance(plane, InfluencePlane):
        raise TypeError("plane must be InfluencePlane.")
    if not isinstance(context, InfluenceContext):
        raise TypeError("context must be InfluenceContext.")

    if not plane.active:
        return 0.0

    base = plane.intensity

    if plane.temporal_mode is TemporalMode.STATIC:
        return base

    if plane.temporal_mode is TemporalMode.CONTINUOUS:
        return base * context.delta_time / plane.timescale

    if plane.temporal_mode is TemporalMode.PHASIC:
        event_key = str(
            plane.metadata.get(
                "event_key",
                plane.plane_id,
            )
        )
        return base if event_key in context.events else 0.0

    if plane.temporal_mode is TemporalMode.PERIODIC:
        period = _finite(
            plane.metadata.get("period", plane.timescale),
            "period",
        )
        duty_cycle = _unit(
            plane.metadata.get("duty_cycle", 0.5),
            "duty_cycle",
        )
        phase = context.time % period
        return base if phase <= period * duty_cycle else 0.0

    if plane.temporal_mode is TemporalMode.EVENT_DRIVEN:
        event_key = str(
            plane.metadata.get(
                "event_key",
                plane.plane_id,
            )
        )
        return base if event_key in context.events else 0.0

    if plane.temporal_mode is TemporalMode.DELAYED:
        start_time = _nonnegative(
            plane.metadata.get("start_time", plane.timescale),
            "start_time",
        )
        return base if context.time >= start_time else 0.0

    if plane.temporal_mode is TemporalMode.CUMULATIVE:
        exposure_key = str(
            plane.metadata.get(
                "exposure_key",
                f"{plane.plane_id}.exposure",
            )
        )
        exposure = max(0.0, context.scalar(exposure_key, 0.0))
        return base * exposure

    if plane.temporal_mode is TemporalMode.HYSTERETIC:
        active_key = str(
            plane.metadata.get(
                "active_key",
                f"{plane.plane_id}.active",
            )
        )
        return base if context.condition(active_key) else 0.0

    return base


def operator_is_applicable(
    operator: InfluenceOperator,
    plane: InfluencePlane,
    context: InfluenceContext,
    value: InfluenceValue,
) -> tuple[bool, str]:
    """Evaluate operator activation, plane activity, delay, condition, and threshold."""

    if operator.plane_id != plane.plane_id:
        return False, "operator-plane mismatch"

    if not operator.active:
        return False, "operator inactive"

    activity = plane_activity(plane, context)
    if activity <= 0.0:
        return False, "plane inactive"

    if context.time < operator.delay:
        return False, "operator delay not reached"

    if not context.condition(operator.condition_key):
        return False, "operator condition false"

    if operator.threshold is not None:
        comparison = str(
            operator.metadata.get(
                "threshold_mode",
                "greater_equal",
            )
        )
        threshold = operator.threshold
        scalar = value.scalar

        if comparison == "greater_equal" and scalar < threshold:
            return False, "threshold not reached"
        if comparison == "greater" and scalar <= threshold:
            return False, "threshold not reached"
        if comparison == "less_equal" and scalar > threshold:
            return False, "threshold not reached"
        if comparison == "less" and scalar >= threshold:
            return False, "threshold not reached"
        if comparison == "absolute_greater_equal" and abs(scalar) < abs(
            threshold
        ):
            return False, "threshold not reached"

    return True, "applied"


def effective_operator_gain(
    operator: InfluenceOperator,
    plane: InfluencePlane,
    context: InfluenceContext,
) -> float:
    """
    Compute state-dependent effective gain.

    Optional metadata:
        gain_state_key:
            multiply gain by context.scalar_state[key];

        gain_condition_key:
            set gain to zero unless condition is true;

        clamp_min / clamp_max:
            clamp the final gain.
    """

    gain = operator.gain * plane_activity(plane, context)

    gain_state_key = operator.metadata.get("gain_state_key")
    if gain_state_key is not None:
        gain *= context.scalar(str(gain_state_key), 1.0)

    gain_condition_key = operator.metadata.get(
        "gain_condition_key"
    )
    if gain_condition_key is not None and not context.condition(
        str(gain_condition_key)
    ):
        gain = 0.0

    clamp_min = operator.metadata.get("clamp_min")
    clamp_max = operator.metadata.get("clamp_max")
    if clamp_min is not None:
        gain = max(gain, _finite(clamp_min, "clamp_min"))
    if clamp_max is not None:
        gain = min(gain, _finite(clamp_max, "clamp_max"))

    return gain


def _safe_divide(
    numerator: float,
    denominator: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    if abs(denominator) <= epsilon:
        raise ROIFInfluenceError(
            "division operator encountered a near-zero denominator."
        )
    return numerator / denominator


def _rotate_vector(
    vector: Vector3,
    direction: Vector3,
    gain: float,
) -> Vector3:
    """
    Minimal vector rotation representation.

    This layer stores the change in orientation as a vector addition.
    Exact rigid-body rotation belongs in the future geometric solver.
    """

    return vector + direction.scaled(gain)


def apply_operator(
    value: InfluenceValue,
    operator: InfluenceOperator,
    plane: InfluencePlane,
    context: InfluenceContext,
) -> OperatorApplication:
    """Apply one influence operator to one scalar-vector value."""

    if not isinstance(value, InfluenceValue):
        raise TypeError("value must be InfluenceValue.")
    if not isinstance(operator, InfluenceOperator):
        raise TypeError("operator must be InfluenceOperator.")
    if not isinstance(plane, InfluencePlane):
        raise TypeError("plane must be InfluencePlane.")
    if not isinstance(context, InfluenceContext):
        raise TypeError("context must be InfluenceContext.")

    applicable, reason = operator_is_applicable(
        operator,
        plane,
        context,
        value,
    )

    if not applicable:
        return OperatorApplication(
            operator_id=operator.operator_id,
            plane_id=plane.plane_id,
            applied=False,
            reason=reason,
            input_value=value,
            output_value=value,
            effective_gain=0.0,
            time=context.time,
        )

    gain = effective_operator_gain(
        operator,
        plane,
        context,
    )
    scalar = value.scalar
    vector = value.vector

    kind = operator.kind

    if kind is OperatorKind.IDENTITY:
        output_scalar = scalar
        output_vector = vector

    elif kind in (
        OperatorKind.ADD,
        OperatorKind.LOAD,
        OperatorKind.AMPLIFY,
        OperatorKind.FACILITATE,
        OperatorKind.ACTIVATE,
        OperatorKind.RECRUIT,
        OperatorKind.SENSITIZE,
        OperatorKind.RESTORE,
    ):
        output_scalar = scalar + gain
        output_vector = vector + operator.direction.scaled(gain)

    elif kind in (
        OperatorKind.SUBTRACT,
        OperatorKind.UNLOAD,
        OperatorKind.ATTENUATE,
        OperatorKind.SUPPRESS,
        OperatorKind.INHIBIT,
        OperatorKind.DERECRUIT,
        OperatorKind.DAMAGE,
    ):
        output_scalar = scalar - gain
        output_vector = vector + operator.direction.scaled(-gain)

    elif kind is OperatorKind.MULTIPLY:
        output_scalar = scalar * gain
        output_vector = vector.scaled(gain)

    elif kind is OperatorKind.DIVIDE:
        output_scalar = _safe_divide(scalar, gain)
        output_vector = vector.scaled(
            _safe_divide(1.0, gain)
        )

    elif kind in (
        OperatorKind.TRANSFER_LOAD,
        OperatorKind.REDISTRIBUTE,
    ):
        transfer_fraction = max(
            0.0,
            min(1.0, abs(gain)),
        )
        output_scalar = scalar * (1.0 - transfer_fraction)
        output_vector = vector + operator.direction.scaled(
            scalar * transfer_fraction
        )

    elif kind in (
        OperatorKind.COMPRESS,
        OperatorKind.SHORTEN,
        OperatorKind.FIXATE,
        OperatorKind.GATE,
        OperatorKind.LOCK_IN,
    ):
        factor = max(0.0, 1.0 - gain)
        output_scalar = scalar * factor
        output_vector = vector.scaled(factor)

    elif kind in (
        OperatorKind.DECOMPRESS,
        OperatorKind.STRETCH,
        OperatorKind.MOBILIZE,
        OperatorKind.RELEASE,
        OperatorKind.UNLOCK,
    ):
        factor = max(0.0, 1.0 + gain)
        output_scalar = scalar * factor
        output_vector = vector.scaled(factor)

    elif kind is OperatorKind.ROTATE:
        output_scalar = scalar
        output_vector = _rotate_vector(
            vector,
            operator.direction,
            gain,
        )

    elif kind is OperatorKind.TRANSLATE:
        output_scalar = scalar
        output_vector = vector + operator.direction.scaled(gain)

    elif kind in (
        OperatorKind.STABILIZE,
        OperatorKind.SYNCHRONIZE,
    ):
        target = _finite(
            operator.metadata.get("target", 1.0),
            "target",
        )
        fraction = max(0.0, min(1.0, abs(gain)))
        output_scalar = scalar + (
            target - scalar
        ) * fraction
        output_vector = vector

    elif kind in (
        OperatorKind.DESTABILIZE,
        OperatorKind.DESYNCHRONIZE,
    ):
        output_scalar = scalar * max(0.0, 1.0 - abs(gain))
        output_vector = vector

    elif kind is OperatorKind.FILTER:
        lower = _finite(
            operator.metadata.get("lower", -math.inf),
            "lower",
        )
        upper = _finite(
            operator.metadata.get("upper", math.inf),
            "upper",
        )
        if lower > upper:
            raise ROIFInfluenceError(
                "filter lower bound cannot exceed upper bound."
            )
        output_scalar = min(max(scalar, lower), upper)
        output_vector = vector

    elif kind is OperatorKind.DELAY:
        output_scalar = scalar
        output_vector = vector

    elif kind is OperatorKind.TRIGGER:
        trigger_value = _finite(
            operator.metadata.get("trigger_value", gain),
            "trigger_value",
        )
        output_scalar = trigger_value
        output_vector = vector

    elif kind in (
        OperatorKind.ADAPT,
        OperatorKind.REMODEL,
        OperatorKind.HABITUATE,
    ):
        target = _finite(
            operator.metadata.get("target", scalar),
            "target",
        )
        fraction = max(0.0, min(1.0, abs(gain)))
        output_scalar = scalar + (
            target - scalar
        ) * fraction
        output_vector = vector

    else:
        raise ROIFInfluenceError(
            f"operator kind {kind.value!r} is not implemented."
        )

    output = InfluenceValue(
        scalar=output_scalar,
        vector=output_vector,
        metadata={
            **dict(value.metadata),
            "last_operator_id": operator.operator_id,
            "last_plane_id": plane.plane_id,
        },
    )

    return OperatorApplication(
        operator_id=operator.operator_id,
        plane_id=plane.plane_id,
        applied=True,
        reason="applied",
        input_value=value,
        output_value=output,
        effective_gain=gain,
        time=context.time,
        metadata={
            "operator_kind": operator.kind.value,
            "plane_kind": plane.kind.value,
            "composition": operator.composition.value,
        },
    )


def compose_operators(
    initial: InfluenceValue,
    composition: OperatorComposition,
    operators: Mapping[str, InfluenceOperator],
    planes: Mapping[str, InfluencePlane],
    context: InfluenceContext,
) -> CompositionApplication:
    """Apply a canonical composition of operators."""

    missing = [
        operator_id
        for operator_id in composition.operator_ids
        if operator_id not in operators
    ]
    if missing:
        raise ROIFInfluenceError(
            f"unknown operators in composition: {missing!r}."
        )

    traces: list[OperatorApplication] = []

    if composition.kind is CompositionKind.SEQUENTIAL:
        current = initial
        for operator_id in composition.operator_ids:
            operator = operators[operator_id]
            if operator.plane_id not in planes:
                raise ROIFInfluenceError(
                    f"unknown plane {operator.plane_id!r}."
                )
            trace = apply_operator(
                current,
                operator,
                planes[operator.plane_id],
                context,
            )
            traces.append(trace)
            current = trace.output_value

        output = current

    else:
        independent: list[OperatorApplication] = []
        for operator_id in composition.operator_ids:
            operator = operators[operator_id]
            if operator.plane_id not in planes:
                raise ROIFInfluenceError(
                    f"unknown plane {operator.plane_id!r}."
                )
            trace = apply_operator(
                initial,
                operator,
                planes[operator.plane_id],
                context,
            )
            independent.append(trace)
        traces.extend(independent)

        applied_values = [
            trace.output_value
            for trace in independent
            if trace.applied
        ]

        if not applied_values:
            output = initial

        elif composition.kind is CompositionKind.ADDITIVE:
            delta = sum(
                item.scalar - initial.scalar
                for item in applied_values
            )
            vector = initial.vector
            for item in applied_values:
                vector = vector + (
                    item.vector + initial.vector.scaled(-1.0)
                )
            output = InfluenceValue(
                scalar=initial.scalar + delta,
                vector=vector,
            )

        elif composition.kind is CompositionKind.MULTIPLICATIVE:
            scalar = initial.scalar
            vector_scale = 1.0
            for trace in independent:
                if not trace.applied:
                    continue
                if math.isclose(
                    trace.input_value.scalar,
                    0.0,
                    abs_tol=1e-15,
                ):
                    factor = trace.effective_gain
                else:
                    factor = (
                        trace.output_value.scalar
                        / trace.input_value.scalar
                    )
                scalar *= factor
                vector_scale *= factor
            output = InfluenceValue(
                scalar=scalar,
                vector=initial.vector.scaled(vector_scale),
            )

        elif composition.kind is CompositionKind.CONDITIONAL:
            chosen = next(
                (
                    trace.output_value
                    for trace in independent
                    if trace.applied
                ),
                initial,
            )
            output = chosen

        elif composition.kind is CompositionKind.VECTOR_SUM:
            scalar = sum(
                item.scalar for item in applied_values
            )
            vector = Vector3()
            for item in applied_values:
                vector = vector + item.vector
            output = InfluenceValue(
                scalar=scalar,
                vector=vector,
            )

        elif composition.kind is CompositionKind.MAXIMUM:
            output = max(
                applied_values,
                key=lambda item: item.scalar,
            )

        elif composition.kind is CompositionKind.MINIMUM:
            output = min(
                applied_values,
                key=lambda item: item.scalar,
            )

        else:
            raise ROIFInfluenceError(
                "CUSTOM composition requires a domain adapter."
            )

    return CompositionApplication(
        composition_id=composition.composition_id,
        kind=composition.kind,
        input_value=initial,
        output_value=output,
        operator_traces=tuple(traces),
    )


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def apply_operator_to_channel(
    channel: FunctionalChannel,
    operator: InfluenceOperator,
    plane: InfluencePlane,
    context: InfluenceContext,
) -> ChannelInfluenceResult:
    """
    Apply one operator directly to a FunctionalChannel.

    The target channel component is chosen through
    ``operator.metadata["channel_field"]``.

    Supported channel fields:
        load
        capacity
        baseline_output
        history_factor
        activation.command
        activation.neural_drive
        activation.afferent_gate
        activation.timing
        activation.coordination
        activation.geometric_access
        activation.task_compatibility
        geometry.length
        geometry.moment_arm
        geometry.mobility
        geometry.stability
        geometry.orientation
        direction
    """

    if not isinstance(channel, FunctionalChannel):
        raise TypeError("channel must be FunctionalChannel.")

    field_name = str(
        operator.metadata.get(
            "channel_field",
            "baseline_output",
        )
    )

    if field_name == "load":
        initial = InfluenceValue(
            scalar=channel.capacity_state.load,
            vector=channel.direction,
        )
    elif field_name == "capacity":
        initial = InfluenceValue(
            scalar=channel.capacity_state.capacity,
            vector=channel.direction,
        )
    elif field_name == "baseline_output":
        initial = InfluenceValue(
            scalar=channel.baseline_output,
            vector=channel.direction,
        )
    elif field_name == "history_factor":
        initial = InfluenceValue(
            scalar=channel.history_factor,
            vector=channel.direction,
        )
    elif field_name.startswith("activation."):
        attr = field_name.split(".", 1)[1]
        if not hasattr(channel.activation, attr):
            raise ROIFInfluenceError(
                f"unknown activation field {attr!r}."
            )
        initial = InfluenceValue(
            scalar=float(getattr(channel.activation, attr)),
            vector=channel.direction,
        )
    elif field_name.startswith("geometry."):
        attr = field_name.split(".", 1)[1]
        if attr == "orientation":
            initial = InfluenceValue(
                scalar=0.0,
                vector=channel.geometry.orientation,
            )
        else:
            if not hasattr(channel.geometry, attr):
                raise ROIFInfluenceError(
                    f"unknown geometry field {attr!r}."
                )
            initial = InfluenceValue(
                scalar=float(getattr(channel.geometry, attr)),
                vector=channel.direction,
            )
    elif field_name == "direction":
        initial = InfluenceValue(
            scalar=channel.effective_scalar_output,
            vector=channel.direction,
        )
    else:
        raise ROIFInfluenceError(
            f"unsupported channel_field {field_name!r}."
        )

    trace = apply_operator(
        initial,
        operator,
        plane,
        context,
    )

    if not trace.applied:
        return ChannelInfluenceResult(
            channel_before=channel,
            channel_after=channel,
            applications=(trace,),
        )

    output = trace.output_value
    updated = channel

    if field_name == "load":
        new_state = CapacityState(
            capacity=channel.capacity_state.capacity,
            load=max(0.0, output.scalar),
            critical_reserve_fraction=(
                channel.capacity_state.critical_reserve_fraction
            ),
        )
        updated = replace(
            channel,
            capacity_state=new_state,
        )

    elif field_name == "capacity":
        new_capacity = max(1e-12, output.scalar)
        new_state = CapacityState(
            capacity=new_capacity,
            load=channel.capacity_state.load,
            critical_reserve_fraction=(
                channel.capacity_state.critical_reserve_fraction
            ),
        )
        updated = replace(
            channel,
            capacity_state=new_state,
        )

    elif field_name == "baseline_output":
        updated = replace(
            channel,
            baseline_output=max(0.0, output.scalar),
        )

    elif field_name == "history_factor":
        updated = replace(
            channel,
            history_factor=max(0.0, output.scalar),
        )

    elif field_name.startswith("activation."):
        attr = field_name.split(".", 1)[1]
        activation = replace(
            channel.activation,
            **{attr: _bounded(output.scalar)},
        )
        updated = replace(
            channel,
            activation=activation,
        )

    elif field_name.startswith("geometry."):
        attr = field_name.split(".", 1)[1]
        if attr == "orientation":
            geometry = replace(
                channel.geometry,
                orientation=output.vector,
            )
        elif attr in ("mobility", "stability"):
            geometry = replace(
                channel.geometry,
                **{attr: _bounded(output.scalar)},
            )
        elif attr in (
            "length",
            "optimal_length",
            "moment_arm",
            "optimal_moment_arm",
        ):
            geometry = replace(
                channel.geometry,
                **{attr: max(1e-12, output.scalar)},
            )
        else:
            raise ROIFInfluenceError(
                f"unsupported geometry field {attr!r}."
            )
        updated = replace(
            channel,
            geometry=geometry,
        )

    elif field_name == "direction":
        updated = replace(
            channel,
            direction=output.vector,
        )

    return ChannelInfluenceResult(
        channel_before=channel,
        channel_after=updated,
        applications=(trace,),
        metadata={
            "channel_field": field_name,
            "operator_kind": operator.kind.value,
            "plane_kind": plane.kind.value,
        },
    )


def apply_operator_sequence_to_channel(
    channel: FunctionalChannel,
    operator_ids: Sequence[str],
    operators: Mapping[str, InfluenceOperator],
    planes: Mapping[str, InfluencePlane],
    context: InfluenceContext,
) -> ChannelInfluenceResult:
    """Apply operators sequentially to one channel."""

    current = channel
    traces: list[OperatorApplication] = []

    for operator_id in operator_ids:
        if operator_id not in operators:
            raise ROIFInfluenceError(
                f"unknown operator {operator_id!r}."
            )
        operator = operators[operator_id]
        if operator.plane_id not in planes:
            raise ROIFInfluenceError(
                f"unknown plane {operator.plane_id!r}."
            )

        result = apply_operator_to_channel(
            current,
            operator,
            planes[operator.plane_id],
            context,
        )
        current = result.channel_after
        traces.extend(result.applications)

    return ChannelInfluenceResult(
        channel_before=channel,
        channel_after=current,
        applications=tuple(traces),
        metadata={
            "operator_count": len(operator_ids),
            "mode": "sequential",
        },
    )


def influence_factor_product(
    factors: Mapping[str, float],
    *,
    floor: float = 0.0,
    ceiling: float | None = None,
) -> float:
    """
    Compute a named multiplicative factor product.

    This helper is the simplest explicit representation of ROIF
    multiplicativity:

        capacity
        * reserve
        * geometry
        * activation
        * history
        * plane factors
    """

    floor = _finite(floor, "floor")
    if ceiling is not None:
        ceiling = _finite(ceiling, "ceiling")
        if ceiling < floor:
            raise ROIFInfluenceError(
                "ceiling cannot be lower than floor."
            )

    result = 1.0
    for name, value in factors.items():
        factor = _finite(value, f"factor[{name!r}]")
        result *= factor

    result = max(floor, result)
    if ceiling is not None:
        result = min(ceiling, result)
    return result


def vector_sum(
    vectors: Sequence[Vector3],
) -> Vector3:
    """Return the vector sum of all supplied channel outputs."""

    result = Vector3()
    for vector in vectors:
        if not isinstance(vector, Vector3):
            raise ROIFInfluenceError(
                "all vector_sum values must be Vector3."
            )
        result = result + vector
    return result


def channels_vector_sum(
    channels: Sequence[FunctionalChannel],
) -> Vector3:
    """Combine effective vector outputs of multiple functional channels."""

    return vector_sum(
        [
            channel.effective_vector_output
            for channel in channels
        ]
    )


def planes_by_id(
    planes: Sequence[InfluencePlane],
) -> Mapping[str, InfluencePlane]:
    """Build an immutable plane lookup with duplicate protection."""

    lookup: dict[str, InfluencePlane] = {}
    for plane in planes:
        if plane.plane_id in lookup:
            raise ROIFInfluenceError(
                f"duplicate plane_id {plane.plane_id!r}."
            )
        lookup[plane.plane_id] = plane
    return MappingProxyType(lookup)


def operators_by_id(
    operators: Sequence[InfluenceOperator],
) -> Mapping[str, InfluenceOperator]:
    """Build an immutable operator lookup with duplicate protection."""

    lookup: dict[str, InfluenceOperator] = {}
    for operator in operators:
        if operator.operator_id in lookup:
            raise ROIFInfluenceError(
                f"duplicate operator_id {operator.operator_id!r}."
            )
        lookup[operator.operator_id] = operator
    return MappingProxyType(lookup)


__all__ = [
    "ChannelInfluenceResult",
    "CompositionApplication",
    "InfluenceContext",
    "InfluenceValue",
    "OperatorApplication",
    "ROIFInfluenceError",
    "apply_operator",
    "apply_operator_sequence_to_channel",
    "apply_operator_to_channel",
    "channels_vector_sum",
    "compose_operators",
    "effective_operator_gain",
    "influence_factor_product",
    "operator_is_applicable",
    "operators_by_id",
    "plane_activity",
    "planes_by_id",
    "vector_sum",
]
