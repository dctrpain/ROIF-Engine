"""
ROIF Memory 2.0
Contextual Trajectory Core

Thirteenth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor
    AttractorDynamics
    AttractorTrajectory
    ExperienceConditioning
    ConditionedPrestress
    ConditionedDynamics
    ConditionedTrajectory
    AssociativeContext
    ContextualDynamics

Purpose
-------
Track a sequence of associative contexts through time:

    state_0
      + context_0
      -> contextual dynamics_0
      -> state_1
      + context_1
      -> contextual dynamics_1
      -> state_2
      ...

The trajectory layer measures:
- continuity of contextual state propagation
- context sequence identity
- baseline dominant-attractor sequence
- contextual dominant-attractor sequence
- contextual switching and returns
- context-induced state divergence per step
- cumulative contextual displacement
- cumulative context-induced divergence
- target-weight gains over time

This module remains descriptive:
- no learning
- no source memory mutation
- no attractor mutation
- no action selection
- no policy modification
- no diagnosis
- no biological contextual-dynamics claim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from roif.history.experience_attractor import ExperienceAttractor
from roif.history.attractor_dynamics import DynamicsState, state_shift_magnitude
from roif.history.associative_context import AssociativeContext
from roif.history.contextual_dynamics import (
    ContextualDynamicsResult,
    context_target_weight_gain,
    contextual_dynamics_is_policy_free,
    run_contextual_dynamics,
)


SCHEMA_VERSION = "contextual_trajectory_v1"


class ContextualTrajectoryError(ValueError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _validate_id(name: str, value: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ContextualTrajectoryError(f"{name} must not be empty")
    return cleaned


def _validate_finite(name: str, value: float) -> float:
    x = float(value)
    if not isfinite(x):
        raise ContextualTrajectoryError(f"{name} must be finite")
    return x


@dataclass(frozen=True, slots=True)
class ContextualTrajectoryStep:
    step_index: int
    context: AssociativeContext
    state_before: DynamicsState
    dynamics: ContextualDynamicsResult

    baseline_dominant_attractor_id: str | None
    contextual_dominant_attractor_id: str | None

    contextual_switched_from_previous: bool
    contextual_returned_to_previous_attractor: bool

    target_weight_gain: float
    context_induced_state_divergence: float
    contextual_state_shift_magnitude: float

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ContextualTrajectoryError("step_index must be non-negative")

        for name in (
            "target_weight_gain",
            "context_induced_state_divergence",
            "contextual_state_shift_magnitude",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(name, getattr(self, name)),
            )

        if self.context_induced_state_divergence < 0.0:
            raise ContextualTrajectoryError(
                "context_induced_state_divergence must be non-negative"
            )

        if self.contextual_state_shift_magnitude < 0.0:
            raise ContextualTrajectoryError(
                "contextual_state_shift_magnitude must be non-negative"
            )

        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ContextualTrajectorySummary:
    step_count: int
    unique_contextual_attractor_count: int

    switch_count: int
    return_count: int

    cumulative_contextual_displacement: float
    cumulative_context_induced_divergence: float

    mean_contextual_displacement: float
    mean_context_induced_divergence: float
    mean_target_weight_gain: float

    continuity_preserved: bool
    all_steps_policy_free: bool

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_count < 1:
            raise ContextualTrajectoryError("step_count must be >= 1")

        for name in (
            "unique_contextual_attractor_count",
            "switch_count",
            "return_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ContextualTrajectoryError(f"{name} must be non-negative")

        for name in (
            "cumulative_contextual_displacement",
            "cumulative_context_induced_divergence",
            "mean_contextual_displacement",
            "mean_context_induced_divergence",
            "mean_target_weight_gain",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(name, getattr(self, name)),
            )

        for name in (
            "cumulative_contextual_displacement",
            "cumulative_context_induced_divergence",
            "mean_contextual_displacement",
            "mean_context_induced_divergence",
        ):
            if getattr(self, name) < 0.0:
                raise ContextualTrajectoryError(f"{name} must be non-negative")

        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ContextualTrajectory:
    trajectory_id: str
    initial_state: DynamicsState

    steps: tuple[ContextualTrajectoryStep, ...]

    final_state: DynamicsState
    summary: ContextualTrajectorySummary

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_id",
            _validate_id("trajectory_id", self.trajectory_id),
        )

        if not self.steps:
            raise ContextualTrajectoryError("steps must not be empty")

        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


def contextual_trajectory_continuity_break_indices(
    steps: Iterable[ContextualTrajectoryStep],
) -> tuple[int, ...]:
    items = tuple(steps)
    breaks = []

    for index in range(len(items) - 1):
        if (
            items[index].dynamics.contextual.state_after
            != items[index + 1].state_before
        ):
            breaks.append(index)

    return tuple(breaks)


def contextual_trajectory_is_continuous(
    steps: Iterable[ContextualTrajectoryStep],
) -> bool:
    return not contextual_trajectory_continuity_break_indices(steps)


def context_id_series(
    trajectory: ContextualTrajectory,
) -> tuple[str, ...]:
    return tuple(step.context.context_id for step in trajectory.steps)


def baseline_dominant_series(
    trajectory: ContextualTrajectory,
) -> tuple[str | None, ...]:
    return tuple(
        step.baseline_dominant_attractor_id
        for step in trajectory.steps
    )


def contextual_dominant_series(
    trajectory: ContextualTrajectory,
) -> tuple[str | None, ...]:
    return tuple(
        step.contextual_dominant_attractor_id
        for step in trajectory.steps
    )


def switch_flags(
    trajectory: ContextualTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.contextual_switched_from_previous
        for step in trajectory.steps
    )


def return_flags(
    trajectory: ContextualTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.contextual_returned_to_previous_attractor
        for step in trajectory.steps
    )


def target_weight_gain_series(
    trajectory: ContextualTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.target_weight_gain
        for step in trajectory.steps
    )


def context_induced_divergence_series(
    trajectory: ContextualTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.context_induced_state_divergence
        for step in trajectory.steps
    )


def contextual_displacement_series(
    trajectory: ContextualTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.contextual_state_shift_magnitude
        for step in trajectory.steps
    )


def state_id_series(
    trajectory: ContextualTrajectory,
) -> tuple[str, ...]:
    return (
        trajectory.initial_state.state_id,
        *tuple(
            step.dynamics.contextual.state_after.state_id
            for step in trajectory.steps
        ),
    )


def summarize_contextual_trajectory(
    steps: Iterable[ContextualTrajectoryStep],
) -> ContextualTrajectorySummary:
    items = tuple(steps)

    if not items:
        raise ContextualTrajectoryError(
            "cannot summarize empty contextual trajectory"
        )

    contextual_ids = tuple(
        step.contextual_dominant_attractor_id
        for step in items
    )

    switch_count = sum(
        1
        for step in items
        if step.contextual_switched_from_previous
    )

    return_count = sum(
        1
        for step in items
        if step.contextual_returned_to_previous_attractor
    )

    displacements = tuple(
        step.contextual_state_shift_magnitude
        for step in items
    )

    divergences = tuple(
        step.context_induced_state_divergence
        for step in items
    )

    gains = tuple(
        step.target_weight_gain
        for step in items
    )

    return ContextualTrajectorySummary(
        step_count=len(items),
        unique_contextual_attractor_count=len(
            {
                attractor_id
                for attractor_id in contextual_ids
                if attractor_id is not None
            }
        ),
        switch_count=switch_count,
        return_count=return_count,
        cumulative_contextual_displacement=sum(displacements),
        cumulative_context_induced_divergence=sum(divergences),
        mean_contextual_displacement=(
            sum(displacements) / len(displacements)
        ),
        mean_context_induced_divergence=(
            sum(divergences) / len(divergences)
        ),
        mean_target_weight_gain=(
            sum(gains) / len(gains)
        ),
        continuity_preserved=contextual_trajectory_is_continuous(items),
        all_steps_policy_free=all(
            contextual_dynamics_is_policy_free(step.dynamics)
            for step in items
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "trajectory_summary_is_descriptive": True,
            "memory_mutated": False,
            "learning_applied": False,
            "biological_context_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def build_contextual_trajectory(
    *,
    trajectory_id: str,
    initial_state: DynamicsState,
    attractors: Iterable[ExperienceAttractor],
    context_sequence: Iterable[AssociativeContext],
    distance_scale: float = 1.0,
    step_size: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> ContextualTrajectory:
    attractor_items = tuple(attractors)

    if not attractor_items:
        raise ContextualTrajectoryError(
            "attractors must not be empty"
        )

    context_items = tuple(context_sequence)

    if not context_items:
        raise ContextualTrajectoryError(
            "context_sequence must not be empty"
        )

    steps = []
    state = initial_state
    seen_contextual: list[str | None] = []

    for index, context in enumerate(context_items):
        dynamics = run_contextual_dynamics(
            result_id=f"{trajectory_id}::step_{index}",
            state=state,
            attractors=attractor_items,
            context=context,
            distance_scale=distance_scale,
            step_size=step_size,
        )

        contextual_dominant = (
            dynamics.contextual.dominant_attractor_id
        )

        baseline_dominant = (
            dynamics.baseline.dominant_attractor_id
        )

        previous_contextual = (
            seen_contextual[-1]
            if seen_contextual
            else None
        )

        switched = (
            index > 0
            and contextual_dominant
            != previous_contextual
        )

        returned = (
            index > 1
            and contextual_dominant
            in seen_contextual[:-1]
            and contextual_dominant
            != previous_contextual
        )

        step = ContextualTrajectoryStep(
            step_index=index,
            context=context,
            state_before=state,
            dynamics=dynamics,
            baseline_dominant_attractor_id=baseline_dominant,
            contextual_dominant_attractor_id=contextual_dominant,
            contextual_switched_from_previous=switched,
            contextual_returned_to_previous_attractor=returned,
            target_weight_gain=context_target_weight_gain(dynamics),
            context_induced_state_divergence=dynamics.state_after_distance,
            contextual_state_shift_magnitude=state_shift_magnitude(
                dynamics.contextual
            ),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "memory_mutated": False,
                "learning_applied": False,
                "action_selected": False,
                "policy_modified": False,
                "biological_context_claimed": False,
                "causal_truth_inferred": False,
            },
        )

        steps.append(step)
        seen_contextual.append(contextual_dominant)
        state = dynamics.contextual.state_after

    step_items = tuple(steps)
    summary = summarize_contextual_trajectory(step_items)

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "contextual_trajectory_mode": "descriptive",
        "memory_mutated": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_context_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(dict(metadata))

    return ContextualTrajectory(
        trajectory_id=trajectory_id,
        initial_state=initial_state,
        steps=step_items,
        final_state=state,
        summary=summary,
        metadata=merged_metadata,
    )


def contextual_trajectory_is_policy_free(
    trajectory: ContextualTrajectory,
) -> bool:
    return (
        trajectory.metadata.get("action_selected") is False
        and trajectory.metadata.get("policy_modified") is False
        and trajectory.metadata.get("memory_mutated") is False
        and trajectory.metadata.get("learning_applied") is False
        and trajectory.summary.all_steps_policy_free
    )


__all__ = [
    "ContextualTrajectory",
    "ContextualTrajectoryError",
    "ContextualTrajectoryStep",
    "ContextualTrajectorySummary",
    "SCHEMA_VERSION",
    "baseline_dominant_series",
    "build_contextual_trajectory",
    "context_id_series",
    "context_induced_divergence_series",
    "contextual_displacement_series",
    "contextual_dominant_series",
    "contextual_trajectory_continuity_break_indices",
    "contextual_trajectory_is_continuous",
    "contextual_trajectory_is_policy_free",
    "return_flags",
    "state_id_series",
    "summarize_contextual_trajectory",
    "switch_flags",
    "target_weight_gain_series",
]
