"""
ROIF Engine
Counterfactual Scenario Evaluation

This module evaluates structured "what if?" interventions over a ROIF system.

It extends the reduced-order virtual intervention logic from
``roif_root_detector.py`` into a reusable scenario engine capable of:

- single-node and multi-node interventions;
- incoming, outgoing, reserve, activation, geometry, and history changes;
- adaptive versus strict-linear safety policies;
- comparison against the observed baseline cascade;
- ranking by cascade reduction, spectral gain, cost, collateral effect,
  uncertainty, irreversibility, and safety;
- deterministic scenario batches;
- Pareto-front extraction.

The module is domain-independent. Scenarios may represent biological,
engineering, informational, behavioural, organisational, or control systems.

This is a research-prototype computation layer. It does not authorize
real-world interventions.

Author:
    Architect (Dctr Pain)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from .roif_entities import (
    ActivationAvailability,
    CapacityState,
    FunctionalChannel,
    GeometryState,
    ROIFSystem,
)
from .roif_tensor import (
    CapacityTensor,
    TensorBuildConfig,
    build_capacity_tensor,
    spectral_radius,
)
from .roif_cascade import (
    CascadeConfig,
    CascadeDirection,
    CascadeTrajectory,
    InfluenceContext,
    run_cascade,
)
from .roif_materials import MaterialModel
from .roif_root_detector import (
    InterventionPolicy,
    ROIFRootDetectorError,
    RootDetectorThresholds,
    matrix_affected_channel_count,
    matrix_trajectory_burden,
    simulate_matrix_cascade,
)


class ROIFCounterfactualError(ValueError):
    """Raised when a counterfactual scenario is invalid."""


class CounterfactualAction(str, Enum):
    """Primitive intervention applied to one functional channel."""

    RESTORE_CHANNEL = "restore_channel"
    REMOVE_OUTGOING_INFLUENCE = "remove_outgoing_influence"
    REDUCE_OUTGOING_INFLUENCE = "reduce_outgoing_influence"
    REDUCE_INCOMING_LOAD = "reduce_incoming_load"
    RESTORE_RESERVE = "restore_reserve"
    CHANGE_ACTIVATION = "change_activation"
    CHANGE_GEOMETRY = "change_geometry"
    CHANGE_HISTORY = "change_history"
    SCALE_CHANNEL_STATE = "scale_channel_state"
    CUSTOM = "custom"


class CounterfactualExecutionMode(str, Enum):
    """How a scenario is evaluated."""

    MATRIX_ONLY = "matrix_only"
    REBUILD_TENSOR = "rebuild_tensor"
    FULL_CASCADE = "full_cascade"


class CounterfactualStatus(str, Enum):
    """Final scenario status."""

    VALID = "valid"
    UNSAFE = "unsafe"
    INVALID = "invalid"
    DOMINATED = "dominated"


class CounterfactualObjective(str, Enum):
    """Primary ranking objective."""

    MAXIMUM_REDUCTION = "maximum_reduction"
    MAXIMUM_SPECTRAL_GAIN = "maximum_spectral_gain"
    MINIMUM_COST = "minimum_cost"
    MINIMUM_RISK = "minimum_risk"
    BALANCED_UTILITY = "balanced_utility"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFCounterfactualError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFCounterfactualError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ROIFCounterfactualError(f"{name} must be finite.")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ROIFCounterfactualError(f"{name} cannot be negative.")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ROIFCounterfactualError(f"{name} must be positive.")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ROIFCounterfactualError(f"{name} must be within [0, 1].")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFCounterfactualError(f"{name} must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFCounterfactualError("metadata must be a mapping.")
    return MappingProxyType(dict(value))


def _float_mapping(
    value: Mapping[str, Any] | None,
    name: str,
) -> Mapping[str, float]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFCounterfactualError(f"{name} must be a mapping.")
    return MappingProxyType(
        {
            str(key): _finite(item, f"{name}[{key!r}]")
            for key, item in value.items()
        }
    )


def _readonly_matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ROIFCounterfactualError(f"{name} must be numeric.") from exc
    if array.ndim != 2:
        raise ROIFCounterfactualError(f"{name} must be two-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ROIFCounterfactualError(f"{name} must contain finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_vector(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ROIFCounterfactualError(f"{name} must be numeric.") from exc
    if array.ndim != 1:
        raise ROIFCounterfactualError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ROIFCounterfactualError(f"{name} must contain finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CounterfactualWeights:
    """Weights used to calculate balanced scenario utility."""

    cascade_reduction: float = 1.0
    spectral_gain: float = 0.60
    final_state_reduction: float = 0.50
    peak_reduction: float = 0.40

    cost: float = 0.40
    collateral: float = 0.70
    safety_risk: float = 1.0
    uncertainty: float = 0.60
    irreversibility: float = 1.0
    complexity: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            "cascade_reduction",
            "spectral_gain",
            "final_state_reduction",
            "peak_reduction",
            "cost",
            "collateral",
            "safety_risk",
            "uncertainty",
            "irreversibility",
            "complexity",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class CounterfactualConfig:
    """Global counterfactual evaluation configuration."""

    execution_mode: CounterfactualExecutionMode = (
        CounterfactualExecutionMode.MATRIX_ONLY
    )
    objective: CounterfactualObjective = (
        CounterfactualObjective.BALANCED_UTILITY
    )
    policy: InterventionPolicy = InterventionPolicy.OBSERVATIONAL

    steps: int | None = None
    dissipation: float = 0.0
    retention: float = 0.0

    clip_min: float | None = None
    clip_max: float | None = None

    thresholds: RootDetectorThresholds = field(
        default_factory=RootDetectorThresholds
    )
    weights: CounterfactualWeights = field(
        default_factory=CounterfactualWeights
    )

    reject_unsafe: bool = False
    mark_dominated: bool = True
    include_baseline: bool = True
    deterministic: bool = True

    default_cost: float = 0.0
    default_safety_risk: float = 0.0
    default_uncertainty: float = 0.0
    default_irreversibility: float = 0.0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.execution_mode,
            CounterfactualExecutionMode,
        ):
            raise ROIFCounterfactualError(
                "execution_mode must be CounterfactualExecutionMode."
            )
        if not isinstance(self.objective, CounterfactualObjective):
            raise ROIFCounterfactualError(
                "objective must be CounterfactualObjective."
            )
        if not isinstance(self.policy, InterventionPolicy):
            raise ROIFCounterfactualError(
                "policy must be InterventionPolicy."
            )

        if self.steps is not None:
            if (
                isinstance(self.steps, bool)
                or not isinstance(self.steps, int)
                or self.steps < 1
            ):
                raise ROIFCounterfactualError(
                    "steps must be a positive integer or None."
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

        if self.clip_min is not None:
            object.__setattr__(
                self,
                "clip_min",
                _finite(self.clip_min, "clip_min"),
            )
        if self.clip_max is not None:
            object.__setattr__(
                self,
                "clip_max",
                _finite(self.clip_max, "clip_max"),
            )
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min > self.clip_max
        ):
            raise ROIFCounterfactualError(
                "clip_min cannot exceed clip_max."
            )

        if not isinstance(self.thresholds, RootDetectorThresholds):
            raise ROIFCounterfactualError(
                "thresholds must be RootDetectorThresholds."
            )
        if not isinstance(self.weights, CounterfactualWeights):
            raise ROIFCounterfactualError(
                "weights must be CounterfactualWeights."
            )

        for name in (
            "reject_unsafe",
            "mark_dominated",
            "include_baseline",
            "deterministic",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ROIFCounterfactualError(f"{name} must be bool.")

        for name in (
            "default_cost",
            "default_safety_risk",
            "default_uncertainty",
            "default_irreversibility",
        ):
            object.__setattr__(
                self,
                name,
                _unit(getattr(self, name), name),
            )

        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CounterfactualIntervention:
    """One primitive channel intervention."""

    intervention_id: str
    channel_id: str
    action: CounterfactualAction

    magnitude: float = 1.0
    target_value: float | None = None

    cost: float | None = None
    safety_risk: float | None = None
    uncertainty: float | None = None
    irreversibility: float | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intervention_id",
            _text(self.intervention_id, "intervention_id"),
        )
        object.__setattr__(
            self,
            "channel_id",
            _text(self.channel_id, "channel_id"),
        )
        if not isinstance(self.action, CounterfactualAction):
            raise ROIFCounterfactualError(
                "action must be CounterfactualAction."
            )
        object.__setattr__(
            self,
            "magnitude",
            _unit(self.magnitude, "magnitude"),
        )
        if self.target_value is not None:
            object.__setattr__(
                self,
                "target_value",
                _finite(self.target_value, "target_value"),
            )

        for name in (
            "cost",
            "safety_risk",
            "uncertainty",
            "irreversibility",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _unit(value, name),
                )

        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CounterfactualScenario:
    """Named collection of one or more interventions."""

    scenario_id: str
    name: str
    interventions: tuple[CounterfactualIntervention, ...]

    description: str = ""
    priority: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_id",
            _text(self.scenario_id, "scenario_id"),
        )
        object.__setattr__(
            self,
            "name",
            _text(self.name, "name"),
        )
        interventions = tuple(self.interventions)
        if not interventions:
            raise ROIFCounterfactualError(
                "interventions cannot be empty."
            )
        if not all(
            isinstance(item, CounterfactualIntervention)
            for item in interventions
        ):
            raise ROIFCounterfactualError(
                "all interventions must be CounterfactualIntervention."
            )
        ids = tuple(item.intervention_id for item in interventions)
        if len(ids) != len(set(ids)):
            raise ROIFCounterfactualError(
                "intervention_id values must be unique within a scenario."
            )
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(
            self,
            "description",
            str(self.description),
        )
        object.__setattr__(
            self,
            "priority",
            _finite(self.priority, "priority"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CounterfactualMetrics:
    """Numerical comparison between baseline and one scenario."""

    baseline_burden: float
    scenario_burden: float
    cascade_reduction: float
    relative_reduction: float

    baseline_final_burden: float
    scenario_final_burden: float
    final_state_reduction: float

    baseline_peak_burden: float
    scenario_peak_burden: float
    peak_reduction: float

    baseline_spectral_radius: float
    scenario_spectral_radius: float
    spectral_gain: float

    affected_channel_count: int
    collateral_effect: float

    cost: float
    safety_risk: float
    uncertainty: float
    irreversibility: float
    complexity: float

    utility: float
    safe: bool

    def __post_init__(self) -> None:
        for name in (
            "baseline_burden",
            "scenario_burden",
            "cascade_reduction",
            "relative_reduction",
            "baseline_final_burden",
            "scenario_final_burden",
            "final_state_reduction",
            "baseline_peak_burden",
            "scenario_peak_burden",
            "peak_reduction",
            "baseline_spectral_radius",
            "scenario_spectral_radius",
            "spectral_gain",
            "collateral_effect",
            "cost",
            "safety_risk",
            "uncertainty",
            "irreversibility",
            "complexity",
            "utility",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        if (
            isinstance(self.affected_channel_count, bool)
            or not isinstance(self.affected_channel_count, int)
            or self.affected_channel_count < 0
        ):
            raise ROIFCounterfactualError(
                "affected_channel_count must be a nonnegative integer."
            )
        if not isinstance(self.safe, bool):
            raise ROIFCounterfactualError("safe must be bool.")


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    """Complete result for one scenario."""

    scenario: CounterfactualScenario
    status: CounterfactualStatus
    metrics: CounterfactualMetrics

    baseline_matrix: np.ndarray
    scenario_matrix: np.ndarray
    baseline_history: np.ndarray
    scenario_history: np.ndarray

    channel_ids: tuple[str, ...]
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, CounterfactualScenario):
            raise ROIFCounterfactualError(
                "scenario must be CounterfactualScenario."
            )
        if not isinstance(self.status, CounterfactualStatus):
            raise ROIFCounterfactualError(
                "status must be CounterfactualStatus."
            )
        if not isinstance(self.metrics, CounterfactualMetrics):
            raise ROIFCounterfactualError(
                "metrics must be CounterfactualMetrics."
            )

        baseline_matrix = _readonly_matrix(
            self.baseline_matrix,
            "baseline_matrix",
        )
        scenario_matrix = _readonly_matrix(
            self.scenario_matrix,
            "scenario_matrix",
        )
        baseline_history = _readonly_matrix(
            self.baseline_history,
            "baseline_history",
        )
        scenario_history = _readonly_matrix(
            self.scenario_history,
            "scenario_history",
        )

        if baseline_matrix.shape != scenario_matrix.shape:
            raise ROIFCounterfactualError(
                "baseline_matrix and scenario_matrix must match."
            )
        if baseline_history.shape != scenario_history.shape:
            raise ROIFCounterfactualError(
                "baseline_history and scenario_history must match."
            )

        channel_ids = tuple(
            _text(value, "channel_id")
            for value in self.channel_ids
        )
        if baseline_matrix.shape != (
            len(channel_ids),
            len(channel_ids),
        ):
            raise ROIFCounterfactualError(
                "matrix shape must match channel_ids."
            )
        if baseline_history.shape[1] != len(channel_ids):
            raise ROIFCounterfactualError(
                "history width must match channel_ids."
            )

        object.__setattr__(
            self,
            "baseline_matrix",
            baseline_matrix,
        )
        object.__setattr__(
            self,
            "scenario_matrix",
            scenario_matrix,
        )
        object.__setattr__(
            self,
            "baseline_history",
            baseline_history,
        )
        object.__setattr__(
            self,
            "scenario_history",
            scenario_history,
        )
        object.__setattr__(self, "channel_ids", channel_ids)
        object.__setattr__(
            self,
            "notes",
            tuple(str(note) for note in self.notes),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CounterfactualBatchResult:
    """Ranked collection of counterfactual results."""

    results: tuple[CounterfactualResult, ...]
    objective: CounterfactualObjective
    pareto_front_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not results:
            raise ROIFCounterfactualError("results cannot be empty.")
        if not isinstance(self.objective, CounterfactualObjective):
            raise ROIFCounterfactualError(
                "objective must be CounterfactualObjective."
            )
        object.__setattr__(self, "results", results)
        object.__setattr__(
            self,
            "pareto_front_ids",
            tuple(self.pareto_front_ids),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def winner(self) -> CounterfactualResult:
        return self.results[0]


def validate_counterfactual_inputs(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
) -> None:
    """Validate channel-axis consistency."""

    if not isinstance(system, ROIFSystem):
        raise TypeError("system must be ROIFSystem.")
    if not isinstance(tensor, CapacityTensor):
        raise TypeError("tensor must be CapacityTensor.")
    if not isinstance(trajectory, CascadeTrajectory):
        raise TypeError("trajectory must be CascadeTrajectory.")

    channel_ids = tuple(system.channel_ids)
    if tensor.channel_ids != channel_ids:
        raise ROIFCounterfactualError(
            "tensor channel order must match system.channel_ids."
        )
    if trajectory.channel_ids != channel_ids:
        raise ROIFCounterfactualError(
            "trajectory channel order must match system.channel_ids."
        )


def _resolve_intervention_value(
    explicit: float | None,
    default: float,
    name: str,
) -> float:
    return _unit(
        default if explicit is None else explicit,
        name,
    )


def scenario_costs(
    scenario: CounterfactualScenario,
    config: CounterfactualConfig,
) -> tuple[float, float, float, float, float]:
    """Aggregate scenario cost, risk, uncertainty, irreversibility, complexity."""

    cost = 0.0
    risk = 0.0
    uncertainty = 0.0
    irreversibility = 0.0

    for intervention in scenario.interventions:
        cost += _resolve_intervention_value(
            intervention.cost,
            config.default_cost,
            "cost",
        )
        risk += _resolve_intervention_value(
            intervention.safety_risk,
            config.default_safety_risk,
            "safety_risk",
        )
        uncertainty += _resolve_intervention_value(
            intervention.uncertainty,
            config.default_uncertainty,
            "uncertainty",
        )
        irreversibility += _resolve_intervention_value(
            intervention.irreversibility,
            config.default_irreversibility,
            "irreversibility",
        )

    count = float(len(scenario.interventions))
    return (
        min(1.0, cost / count),
        min(1.0, risk / count),
        min(1.0, uncertainty / count),
        min(1.0, irreversibility / count),
        min(1.0, count / max(1.0, len(scenario.interventions) + 2.0)),
    )


def apply_intervention_to_matrix(
    matrix: np.ndarray,
    channel_ids: Sequence[str],
    intervention: CounterfactualIntervention,
) -> np.ndarray:
    """Apply one intervention to a tensor matrix."""

    result = np.array(matrix, dtype=float, copy=True)
    ids = tuple(channel_ids)

    try:
        index = ids.index(intervention.channel_id)
    except ValueError as exc:
        raise ROIFCounterfactualError(
            f"unknown channel_id {intervention.channel_id!r}."
        ) from exc

    magnitude = intervention.magnitude
    action = intervention.action

    if action is CounterfactualAction.RESTORE_CHANNEL:
        result[:, index] *= 1.0 - magnitude
        result[index, :] *= 1.0 - magnitude

    elif action is CounterfactualAction.REMOVE_OUTGOING_INFLUENCE:
        result[:, index] = 0.0

    elif action is CounterfactualAction.REDUCE_OUTGOING_INFLUENCE:
        result[:, index] *= 1.0 - magnitude

    elif action is CounterfactualAction.REDUCE_INCOMING_LOAD:
        result[index, :] *= 1.0 - magnitude

    elif action is CounterfactualAction.RESTORE_RESERVE:
        result[:, index] *= 1.0 - magnitude
        result[index, :] *= 1.0 - magnitude

    elif action in {
        CounterfactualAction.CHANGE_ACTIVATION,
        CounterfactualAction.CHANGE_GEOMETRY,
        CounterfactualAction.CHANGE_HISTORY,
        CounterfactualAction.SCALE_CHANNEL_STATE,
    }:
        scale = (
            intervention.target_value
            if intervention.target_value is not None
            else 1.0 - magnitude
        )
        result[:, index] *= scale

    elif action is CounterfactualAction.CUSTOM:
        raise ROIFCounterfactualError(
            "CUSTOM action requires a domain adapter."
        )

    else:
        raise ROIFCounterfactualError(
            f"unsupported action {action!r}."
        )

    result.setflags(write=False)
    return result


def apply_scenario_to_matrix(
    tensor: CapacityTensor,
    scenario: CounterfactualScenario,
) -> np.ndarray:
    """Apply all scenario interventions in declared order."""

    matrix = np.array(tensor.matrix, dtype=float, copy=True)

    for intervention in scenario.interventions:
        matrix = np.array(
            apply_intervention_to_matrix(
                matrix,
                tensor.channel_ids,
                intervention,
            ),
            copy=True,
        )

    matrix.setflags(write=False)
    return matrix


def _replace_channel(
    channel: FunctionalChannel,
    intervention: CounterfactualIntervention,
) -> FunctionalChannel:
    """Return a modified FunctionalChannel for tensor rebuild modes."""

    magnitude = intervention.magnitude
    target = intervention.target_value
    action = intervention.action

    if action in {
        CounterfactualAction.RESTORE_CHANNEL,
        CounterfactualAction.RESTORE_RESERVE,
    }:
        capacity = channel.capacity_state.capacity
        restored_load = (
            capacity * (1.0 - magnitude)
            if target is None
            else max(0.0, target)
        )
        return replace(
            channel,
            capacity_state=replace(
                channel.capacity_state,
                load=restored_load,
            ),
            activation=replace(
                channel.activation,
                command=1.0,
            ),
            geometry=replace(
                channel.geometry,
                mobility=1.0,
            ),
            history_factor=1.0,
        )

    if action is CounterfactualAction.CHANGE_ACTIVATION:
        value = (
            1.0 - magnitude
            if target is None
            else target
        )
        return replace(
            channel,
            activation=replace(
                channel.activation,
                command=max(0.0, min(1.0, value)),
            ),
        )

    if action is CounterfactualAction.CHANGE_GEOMETRY:
        value = (
            1.0 - magnitude
            if target is None
            else target
        )
        return replace(
            channel,
            geometry=replace(
                channel.geometry,
                mobility=max(0.0, value),
            ),
        )

    if action is CounterfactualAction.CHANGE_HISTORY:
        value = (
            1.0 - magnitude
            if target is None
            else target
        )
        return replace(
            channel,
            history_factor=max(0.0, value),
        )

    if action is CounterfactualAction.SCALE_CHANNEL_STATE:
        scale = (
            1.0 - magnitude
            if target is None
            else target
        )
        return replace(
            channel,
            baseline_output=max(
                0.0,
                channel.baseline_output * scale,
            ),
        )

    return channel


def apply_scenario_to_system(
    system: ROIFSystem,
    scenario: CounterfactualScenario,
) -> ROIFSystem:
    """Return a new ROIFSystem with channel-state interventions applied."""

    interventions_by_channel: dict[
        str,
        list[CounterfactualIntervention],
    ] = {}

    for intervention in scenario.interventions:
        interventions_by_channel.setdefault(
            intervention.channel_id,
            [],
        ).append(intervention)

    known = set(system.channel_ids)
    unknown = set(interventions_by_channel) - known
    if unknown:
        raise ROIFCounterfactualError(
            f"unknown intervention channel_ids: {sorted(unknown)!r}."
        )

    new_entities = []

    for entity in system.entities:
        new_channels = []

        for channel in entity.channels:
            updated = channel

            for intervention in interventions_by_channel.get(
                channel.channel_id,
                (),
            ):
                updated = _replace_channel(
                    updated,
                    intervention,
                )

            new_channels.append(updated)

        new_entities.append(
            replace(
                entity,
                channels=tuple(new_channels),
            )
        )

    return replace(
        system,
        entities=tuple(new_entities),
    )


def _baseline_matrix_history(
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: CounterfactualConfig,
) -> tuple[np.ndarray, np.ndarray]:
    steps = config.steps or max(1, trajectory.step_count)

    history = simulate_matrix_cascade(
        tensor.matrix,
        trajectory.initial_state,
        steps=steps,
        dissipation=config.dissipation,
        retention=config.retention,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
    )

    return tensor.matrix, history


def _scenario_matrix_history(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    scenario: CounterfactualScenario,
    config: CounterfactualConfig,
    *,
    tensor_config: TensorBuildConfig | None,
    cascade_config: CascadeConfig | None,
    context: InfluenceContext | None,
    materials: Mapping[str, MaterialModel],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate scenario according to configured execution mode."""

    steps = config.steps or max(1, trajectory.step_count)

    if config.execution_mode is CounterfactualExecutionMode.MATRIX_ONLY:
        scenario_matrix = apply_scenario_to_matrix(
            tensor,
            scenario,
        )
        scenario_history = simulate_matrix_cascade(
            scenario_matrix,
            trajectory.initial_state,
            steps=steps,
            dissipation=config.dissipation,
            retention=config.retention,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )
        return scenario_matrix, scenario_history

    modified_system = apply_scenario_to_system(
        system,
        scenario,
    )

    rebuilt_tensor = build_capacity_tensor(
        modified_system,
        context=context,
        materials=materials,
        config=tensor_config,
    )

    if (
        config.execution_mode
        is CounterfactualExecutionMode.REBUILD_TENSOR
    ):
        scenario_history = simulate_matrix_cascade(
            rebuilt_tensor.matrix,
            trajectory.initial_state,
            steps=steps,
            dissipation=config.dissipation,
            retention=config.retention,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )
        return rebuilt_tensor.matrix, scenario_history

    if (
        config.execution_mode
        is CounterfactualExecutionMode.FULL_CASCADE
    ):
        full_config = cascade_config or CascadeConfig(
            steps=steps,
            direction=CascadeDirection.GENERIC,
        )
        scenario_trajectory = run_cascade(
            modified_system,
            initial_state=trajectory.initial_state,
            context=context,
            materials=materials,
            tensor_config=tensor_config,
            cascade_config=full_config,
        )
        return (
            scenario_trajectory.tensors[-1].matrix,
            scenario_trajectory.state_history,
        )

    raise ROIFCounterfactualError(
        "unsupported CounterfactualExecutionMode."
    )


def _global_burden_per_step(history: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(history), axis=1)


def _relative_gain(
    baseline: float,
    scenario: float,
    epsilon: float,
) -> float:
    return (baseline - scenario) / max(abs(baseline), epsilon)


def _collateral_effect(
    baseline_history: np.ndarray,
    scenario_history: np.ndarray,
    intervention_indices: frozenset[int],
    epsilon: float,
) -> float:
    difference = np.abs(
        baseline_history - scenario_history
    )

    baseline_external = np.array(
        baseline_history,
        dtype=float,
        copy=True,
    )

    for index in intervention_indices:
        difference[:, index] = 0.0
        baseline_external[:, index] = 0.0

    denominator = max(
        float(np.sum(np.abs(baseline_external))),
        epsilon,
    )

    return max(
        0.0,
        min(
            1.0,
            float(np.sum(difference)) / denominator,
        ),
    )


def _scenario_safe(
    metrics: Mapping[str, float],
    scenario_history: np.ndarray,
    config: CounterfactualConfig,
) -> bool:
    thresholds = config.thresholds

    if metrics["collateral"] > thresholds.maximum_acceptable_collateral:
        return False
    if metrics["risk"] > thresholds.maximum_acceptable_safety_risk:
        return False
    if metrics["uncertainty"] > thresholds.maximum_acceptable_uncertainty:
        return False
    if metrics["irreversibility"] > thresholds.irreversible_action_threshold:
        return False

    if config.policy is InterventionPolicy.STRICT_LINEAR:
        if np.any(
            np.abs(scenario_history)
            >= thresholds.failure_threshold
        ):
            return False

    return True


def _utility(
    *,
    relative_reduction: float,
    spectral_gain: float,
    final_reduction: float,
    peak_reduction: float,
    cost: float,
    collateral: float,
    risk: float,
    uncertainty: float,
    irreversibility: float,
    complexity: float,
    weights: CounterfactualWeights,
) -> float:
    gains = (
        weights.cascade_reduction * relative_reduction
        + weights.spectral_gain * spectral_gain
        + weights.final_state_reduction * final_reduction
        + weights.peak_reduction * peak_reduction
    )

    penalties = (
        weights.cost * cost
        + weights.collateral * collateral
        + weights.safety_risk * risk
        + weights.uncertainty * uncertainty
        + weights.irreversibility * irreversibility
        + weights.complexity * complexity
    )

    return gains - penalties


def evaluate_counterfactual(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    scenario: CounterfactualScenario,
    config: CounterfactualConfig | None = None,
    *,
    tensor_config: TensorBuildConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    context: InfluenceContext | None = None,
    materials: Mapping[str, MaterialModel] | None = None,
) -> CounterfactualResult:
    """Evaluate one counterfactual scenario against the observed baseline."""

    config = config or CounterfactualConfig()
    materials = materials or {}

    validate_counterfactual_inputs(
        system,
        tensor,
        trajectory,
    )

    baseline_matrix, baseline_history = _baseline_matrix_history(
        tensor,
        trajectory,
        config,
    )

    scenario_matrix, scenario_history = _scenario_matrix_history(
        system,
        tensor,
        trajectory,
        scenario,
        config,
        tensor_config=tensor_config,
        cascade_config=cascade_config,
        context=context,
        materials=materials,
    )

    if baseline_history.shape != scenario_history.shape:
        raise ROIFCounterfactualError(
            "baseline and scenario histories must have the same shape."
        )

    epsilon = config.thresholds.numerical_epsilon

    baseline_burden = matrix_trajectory_burden(
        baseline_history
    )
    scenario_burden = matrix_trajectory_burden(
        scenario_history
    )

    cascade_reduction = baseline_burden - scenario_burden
    relative_reduction = _relative_gain(
        baseline_burden,
        scenario_burden,
        epsilon,
    )

    baseline_step_burden = _global_burden_per_step(
        baseline_history
    )
    scenario_step_burden = _global_burden_per_step(
        scenario_history
    )

    baseline_final = float(baseline_step_burden[-1])
    scenario_final = float(scenario_step_burden[-1])
    final_reduction = _relative_gain(
        baseline_final,
        scenario_final,
        epsilon,
    )

    baseline_peak = float(np.max(baseline_step_burden))
    scenario_peak = float(np.max(scenario_step_burden))
    peak_reduction = _relative_gain(
        baseline_peak,
        scenario_peak,
        epsilon,
    )

    baseline_radius = spectral_radius(
        baseline_matrix
    )
    scenario_radius = spectral_radius(
        scenario_matrix
    )
    spectral_gain = baseline_radius - scenario_radius

    indices = frozenset(
        tensor.index(intervention.channel_id)
        for intervention in scenario.interventions
    )

    collateral = _collateral_effect(
        baseline_history,
        scenario_history,
        indices,
        epsilon,
    )

    affected_count = matrix_affected_channel_count(
        baseline_history,
        scenario_history,
        threshold=config.thresholds.activity_threshold,
    )

    cost, risk, uncertainty, irreversibility, complexity = (
        scenario_costs(scenario, config)
    )

    safety_values = {
        "collateral": collateral,
        "risk": risk,
        "uncertainty": uncertainty,
        "irreversibility": irreversibility,
    }

    safe = _scenario_safe(
        safety_values,
        scenario_history,
        config,
    )

    utility = _utility(
        relative_reduction=relative_reduction,
        spectral_gain=spectral_gain,
        final_reduction=final_reduction,
        peak_reduction=peak_reduction,
        cost=cost,
        collateral=collateral,
        risk=risk,
        uncertainty=uncertainty,
        irreversibility=irreversibility,
        complexity=complexity,
        weights=config.weights,
    )

    if (
        config.policy is InterventionPolicy.STRICT_LINEAR
        and not safe
    ):
        utility = -abs(utility) - 1.0
    elif (
        config.policy is InterventionPolicy.ADAPTIVE
        and not safe
    ):
        utility -= 0.5

    status = (
        CounterfactualStatus.VALID
        if safe
        else CounterfactualStatus.UNSAFE
    )

    if config.reject_unsafe and not safe:
        status = CounterfactualStatus.INVALID

    metrics = CounterfactualMetrics(
        baseline_burden=baseline_burden,
        scenario_burden=scenario_burden,
        cascade_reduction=cascade_reduction,
        relative_reduction=relative_reduction,
        baseline_final_burden=baseline_final,
        scenario_final_burden=scenario_final,
        final_state_reduction=final_reduction,
        baseline_peak_burden=baseline_peak,
        scenario_peak_burden=scenario_peak,
        peak_reduction=peak_reduction,
        baseline_spectral_radius=baseline_radius,
        scenario_spectral_radius=scenario_radius,
        spectral_gain=spectral_gain,
        affected_channel_count=affected_count,
        collateral_effect=collateral,
        cost=cost,
        safety_risk=risk,
        uncertainty=uncertainty,
        irreversibility=irreversibility,
        complexity=complexity,
        utility=utility,
        safe=safe,
    )

    notes: list[str] = []

    if cascade_reduction < 0.0:
        notes.append(
            "Scenario increases total cascade burden."
        )
    if spectral_gain < 0.0:
        notes.append(
            "Scenario increases spectral instability."
        )
    if not safe:
        notes.append(
            "Scenario violates the configured safety policy."
        )

    return CounterfactualResult(
        scenario=scenario,
        status=status,
        metrics=metrics,
        baseline_matrix=baseline_matrix,
        scenario_matrix=scenario_matrix,
        baseline_history=baseline_history,
        scenario_history=scenario_history,
        channel_ids=tensor.channel_ids,
        notes=tuple(notes),
        metadata={
            "system_id": system.system_id,
            "execution_mode": config.execution_mode.value,
            "objective": config.objective.value,
            "policy": config.policy.value,
        },
    )


def _objective_key(
    result: CounterfactualResult,
    objective: CounterfactualObjective,
) -> tuple[float, str]:
    metrics = result.metrics

    if objective is CounterfactualObjective.MAXIMUM_REDUCTION:
        primary = metrics.relative_reduction
    elif objective is CounterfactualObjective.MAXIMUM_SPECTRAL_GAIN:
        primary = metrics.spectral_gain
    elif objective is CounterfactualObjective.MINIMUM_COST:
        primary = -metrics.cost
    elif objective is CounterfactualObjective.MINIMUM_RISK:
        primary = -metrics.safety_risk
    elif objective is CounterfactualObjective.BALANCED_UTILITY:
        primary = metrics.utility
    else:
        raise ROIFCounterfactualError(
            "unsupported CounterfactualObjective."
        )

    return (-primary, result.scenario.scenario_id)


def dominates(
    left: CounterfactualResult,
    right: CounterfactualResult,
) -> bool:
    """Return True when left Pareto-dominates right."""

    left_values = (
        left.metrics.relative_reduction,
        left.metrics.spectral_gain,
        -left.metrics.cost,
        -left.metrics.collateral_effect,
        -left.metrics.safety_risk,
        -left.metrics.uncertainty,
        -left.metrics.irreversibility,
    )
    right_values = (
        right.metrics.relative_reduction,
        right.metrics.spectral_gain,
        -right.metrics.cost,
        -right.metrics.collateral_effect,
        -right.metrics.safety_risk,
        -right.metrics.uncertainty,
        -right.metrics.irreversibility,
    )

    no_worse = all(
        left_value >= right_value
        for left_value, right_value in zip(
            left_values,
            right_values,
        )
    )
    strictly_better = any(
        left_value > right_value
        for left_value, right_value in zip(
            left_values,
            right_values,
        )
    )

    return no_worse and strictly_better


def pareto_front(
    results: Sequence[CounterfactualResult],
) -> tuple[CounterfactualResult, ...]:
    """Return non-dominated counterfactual results."""

    values = tuple(results)
    front = []

    for candidate in values:
        if not any(
            dominates(other, candidate)
            for other in values
            if other is not candidate
        ):
            front.append(candidate)

    return tuple(
        sorted(
            front,
            key=lambda item: item.scenario.scenario_id,
        )
    )


def mark_dominated_results(
    results: Sequence[CounterfactualResult],
) -> tuple[CounterfactualResult, ...]:
    """Return results with Pareto-dominated scenarios marked."""

    values = tuple(results)
    front_ids = {
        item.scenario.scenario_id
        for item in pareto_front(values)
    }

    updated = []

    for result in values:
        if result.scenario.scenario_id in front_ids:
            updated.append(result)
        else:
            updated.append(
                replace(
                    result,
                    status=CounterfactualStatus.DOMINATED,
                )
            )

    return tuple(updated)


def evaluate_counterfactual_batch(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    scenarios: Sequence[CounterfactualScenario],
    config: CounterfactualConfig | None = None,
    *,
    tensor_config: TensorBuildConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    context: InfluenceContext | None = None,
    materials: Mapping[str, MaterialModel] | None = None,
) -> CounterfactualBatchResult:
    """Evaluate and rank a deterministic batch of scenarios."""

    config = config or CounterfactualConfig()
    scenarios = tuple(scenarios)

    if not scenarios:
        raise ROIFCounterfactualError(
            "scenarios cannot be empty."
        )

    ids = tuple(item.scenario_id for item in scenarios)
    if len(ids) != len(set(ids)):
        raise ROIFCounterfactualError(
            "scenario_id values must be unique."
        )

    results = tuple(
        evaluate_counterfactual(
            system,
            tensor,
            trajectory,
            scenario,
            config,
            tensor_config=tensor_config,
            cascade_config=cascade_config,
            context=context,
            materials=materials,
        )
        for scenario in scenarios
    )

    if config.mark_dominated:
        results = mark_dominated_results(results)

    ranked = tuple(
        sorted(
            results,
            key=lambda result: _objective_key(
                result,
                config.objective,
            ),
        )
    )

    front_ids = tuple(
        item.scenario.scenario_id
        for item in pareto_front(results)
    )

    return CounterfactualBatchResult(
        results=ranked,
        objective=config.objective,
        pareto_front_ids=front_ids,
        metadata={
            "scenario_count": len(results),
            "safe_count": sum(
                1 for item in results if item.metrics.safe
            ),
            "execution_mode": config.execution_mode.value,
            "policy": config.policy.value,
        },
    )


def generate_single_channel_scenarios(
    channel_ids: Sequence[str],
    *,
    action: CounterfactualAction = (
        CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
    ),
    magnitude: float = 1.0,
    cost_by_channel: Mapping[str, float] | None = None,
    risk_by_channel: Mapping[str, float] | None = None,
    uncertainty_by_channel: Mapping[str, float] | None = None,
    irreversibility_by_channel: Mapping[str, float] | None = None,
) -> tuple[CounterfactualScenario, ...]:
    """Generate one scenario per channel."""

    magnitude = _unit(magnitude, "magnitude")
    cost_by_channel = cost_by_channel or {}
    risk_by_channel = risk_by_channel or {}
    uncertainty_by_channel = uncertainty_by_channel or {}
    irreversibility_by_channel = irreversibility_by_channel or {}

    scenarios = []

    for channel_id in channel_ids:
        channel_id = _text(channel_id, "channel_id")

        intervention = CounterfactualIntervention(
            intervention_id=f"{action.value}:{channel_id}",
            channel_id=channel_id,
            action=action,
            magnitude=magnitude,
            cost=cost_by_channel.get(channel_id),
            safety_risk=risk_by_channel.get(channel_id),
            uncertainty=uncertainty_by_channel.get(channel_id),
            irreversibility=irreversibility_by_channel.get(
                channel_id
            ),
        )

        scenarios.append(
            CounterfactualScenario(
                scenario_id=f"scenario:{action.value}:{channel_id}",
                name=f"{action.value} on {channel_id}",
                interventions=(intervention,),
            )
        )

    return tuple(scenarios)


def result_by_scenario_id(
    batch: CounterfactualBatchResult,
    scenario_id: str,
) -> CounterfactualResult:
    """Return one batch result by scenario identifier."""

    scenario_id = _text(scenario_id, "scenario_id")

    for result in batch.results:
        if result.scenario.scenario_id == scenario_id:
            return result

    raise KeyError(scenario_id)


def counterfactual_summary(
    result: CounterfactualResult,
) -> Mapping[str, Any]:
    """Return compact immutable scenario summary."""

    return MappingProxyType(
        {
            "scenario_id": result.scenario.scenario_id,
            "status": result.status.value,
            "safe": result.metrics.safe,
            "relative_reduction": (
                result.metrics.relative_reduction
            ),
            "spectral_gain": result.metrics.spectral_gain,
            "final_state_reduction": (
                result.metrics.final_state_reduction
            ),
            "peak_reduction": result.metrics.peak_reduction,
            "cost": result.metrics.cost,
            "collateral_effect": (
                result.metrics.collateral_effect
            ),
            "safety_risk": result.metrics.safety_risk,
            "uncertainty": result.metrics.uncertainty,
            "irreversibility": result.metrics.irreversibility,
            "utility": result.metrics.utility,
        }
    )


def batch_summary(
    batch: CounterfactualBatchResult,
) -> Mapping[str, Any]:
    """Return compact immutable batch summary."""

    return MappingProxyType(
        {
            "scenario_count": len(batch.results),
            "winner": batch.winner.scenario.scenario_id,
            "objective": batch.objective.value,
            "pareto_front_ids": batch.pareto_front_ids,
            "safe_count": sum(
                1
                for result in batch.results
                if result.metrics.safe
            ),
        }
    )


__all__ = [
    "CounterfactualAction",
    "CounterfactualBatchResult",
    "CounterfactualConfig",
    "CounterfactualExecutionMode",
    "CounterfactualIntervention",
    "CounterfactualMetrics",
    "CounterfactualObjective",
    "CounterfactualResult",
    "CounterfactualScenario",
    "CounterfactualStatus",
    "CounterfactualWeights",
    "ROIFCounterfactualError",
    "apply_intervention_to_matrix",
    "apply_scenario_to_matrix",
    "apply_scenario_to_system",
    "batch_summary",
    "counterfactual_summary",
    "dominates",
    "evaluate_counterfactual",
    "evaluate_counterfactual_batch",
    "generate_single_channel_scenarios",
    "mark_dominated_results",
    "pareto_front",
    "result_by_scenario_id",
    "scenario_costs",
    "validate_counterfactual_inputs",
]
