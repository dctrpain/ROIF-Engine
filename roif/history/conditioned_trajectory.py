"""
ROIF Memory 2.0
Conditioned Trajectory Core

Tenth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor
    AttractorDynamics
    AttractorTrajectory
    ExperienceConditioning
    ConditionedPrestress
    ConditionedDynamics

Purpose
-------
Track a sequence of conditioned cues through time:

    state_0
      + cue_0
      -> conditioned dynamics_0
      -> state_1
      + cue_1
      -> conditioned dynamics_1
      -> state_2
      ...

The trajectory layer measures:
- continuity of state propagation
- cue sequence identity
- conditioned dominant-attractor sequence
- baseline dominant-attractor sequence
- conditioned-vs-baseline divergence per step
- attractor switching and returns
- cumulative conditioned displacement
- cumulative cue-induced divergence
- competition/capture deltas over time

This module remains descriptive:
- no learning
- no source memory mutation
- no attractor mutation
- no action selection
- no policy modification
- no diagnosis
- no biological conditioning claim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from roif.history.experience_attractor import (
    ExperienceAttractor,
)
from roif.history.attractor_dynamics import (
    DynamicsState,
    state_shift_magnitude,
)
from roif.history.experience_conditioning import (
    ConditioningCue,
    ConditioningMemory,
)
from roif.history.conditioned_prestress import (
    ConditionedPrestressConfig,
)
from roif.history.conditioned_dynamics import (
    ConditionedDynamicsResult,
    conditioned_dynamics_is_policy_free,
    conditioned_target_weight_gain,
    run_conditioned_dynamics,
)


SCHEMA_VERSION = "conditioned_trajectory_v1"


class ConditionedTrajectoryError(ValueError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _validate_id(
    name: str,
    value: str,
) -> str:
    cleaned = str(value).strip()

    if not cleaned:
        raise ConditionedTrajectoryError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise ConditionedTrajectoryError(
            f"{name} must be finite"
        )

    return x


@dataclass(frozen=True, slots=True)
class ConditionedTrajectoryStep:
    step_index: int
    cue: ConditioningCue
    state_before: DynamicsState
    dynamics: ConditionedDynamicsResult

    baseline_dominant_attractor_id: str | None
    conditioned_dominant_attractor_id: str | None

    conditioned_switched_from_previous: bool
    conditioned_returned_to_previous_attractor: bool

    target_weight_gain: float
    cue_induced_state_divergence: float
    conditioned_state_shift_magnitude: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ConditionedTrajectoryError(
                "step_index must be non-negative"
            )

        for name in (
            "target_weight_gain",
            "cue_induced_state_divergence",
            "conditioned_state_shift_magnitude",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        if self.cue_induced_state_divergence < 0.0:
            raise ConditionedTrajectoryError(
                "cue_induced_state_divergence must be non-negative"
            )

        if self.conditioned_state_shift_magnitude < 0.0:
            raise ConditionedTrajectoryError(
                "conditioned_state_shift_magnitude must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionedTrajectorySummary:
    step_count: int
    unique_conditioned_attractor_count: int

    switch_count: int
    return_count: int

    cumulative_conditioned_displacement: float
    cumulative_cue_induced_divergence: float

    mean_conditioned_displacement: float
    mean_cue_induced_divergence: float
    mean_target_weight_gain: float

    continuity_preserved: bool
    all_steps_policy_free: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_count < 1:
            raise ConditionedTrajectoryError(
                "step_count must be >= 1"
            )

        for name in (
            "unique_conditioned_attractor_count",
            "switch_count",
            "return_count",
        ):
            if int(
                getattr(
                    self,
                    name,
                )
            ) < 0:
                raise ConditionedTrajectoryError(
                    f"{name} must be non-negative"
                )

        for name in (
            "cumulative_conditioned_displacement",
            "cumulative_cue_induced_divergence",
            "mean_conditioned_displacement",
            "mean_cue_induced_divergence",
            "mean_target_weight_gain",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        for name in (
            "cumulative_conditioned_displacement",
            "cumulative_cue_induced_divergence",
            "mean_conditioned_displacement",
            "mean_cue_induced_divergence",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise ConditionedTrajectoryError(
                    f"{name} must be non-negative"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionedTrajectory:
    trajectory_id: str
    initial_state: DynamicsState

    steps: tuple[
        ConditionedTrajectoryStep,
        ...,
    ]

    final_state: DynamicsState
    summary: ConditionedTrajectorySummary

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_id",
            _validate_id(
                "trajectory_id",
                self.trajectory_id,
            ),
        )

        if not self.steps:
            raise ConditionedTrajectoryError(
                "steps must not be empty"
            )

        object.__setattr__(
            self,
            "steps",
            tuple(
                self.steps
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def conditioned_trajectory_continuity_break_indices(
    steps: Iterable[
        ConditionedTrajectoryStep
    ],
) -> tuple[int, ...]:
    items = tuple(
        steps
    )

    breaks = []

    for index in range(
        len(
            items
        ) - 1
    ):
        if (
            items[
                index
            ].dynamics.conditioned.state_after
            != items[
                index + 1
            ].state_before
        ):
            breaks.append(
                index
            )

    return tuple(
        breaks
    )


def conditioned_trajectory_is_continuous(
    steps: Iterable[
        ConditionedTrajectoryStep
    ],
) -> bool:
    return not conditioned_trajectory_continuity_break_indices(
        steps
    )


def cue_id_series(
    trajectory: ConditionedTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.cue.cue_id
        for step in trajectory.steps
    )


def baseline_dominant_series(
    trajectory: ConditionedTrajectory,
) -> tuple[str | None, ...]:
    return tuple(
        step.baseline_dominant_attractor_id
        for step in trajectory.steps
    )


def conditioned_dominant_series(
    trajectory: ConditionedTrajectory,
) -> tuple[str | None, ...]:
    return tuple(
        step.conditioned_dominant_attractor_id
        for step in trajectory.steps
    )


def switch_flags(
    trajectory: ConditionedTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.conditioned_switched_from_previous
        for step in trajectory.steps
    )


def return_flags(
    trajectory: ConditionedTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.conditioned_returned_to_previous_attractor
        for step in trajectory.steps
    )


def target_weight_gain_series(
    trajectory: ConditionedTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.target_weight_gain
        for step in trajectory.steps
    )


def cue_induced_divergence_series(
    trajectory: ConditionedTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.cue_induced_state_divergence
        for step in trajectory.steps
    )


def conditioned_displacement_series(
    trajectory: ConditionedTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.conditioned_state_shift_magnitude
        for step in trajectory.steps
    )


def state_id_series(
    trajectory: ConditionedTrajectory,
) -> tuple[str, ...]:
    return (
        trajectory.initial_state.state_id,
        *tuple(
            step.dynamics.conditioned.state_after.state_id
            for step in trajectory.steps
        ),
    )


def summarize_conditioned_trajectory(
    steps: Iterable[
        ConditionedTrajectoryStep
    ],
) -> ConditionedTrajectorySummary:
    items = tuple(
        steps
    )

    if not items:
        raise ConditionedTrajectoryError(
            "cannot summarize empty conditioned trajectory"
        )

    conditioned_ids = tuple(
        step.conditioned_dominant_attractor_id
        for step in items
    )

    switch_count = sum(
        1
        for step in items
        if step.conditioned_switched_from_previous
    )

    return_count = sum(
        1
        for step in items
        if step.conditioned_returned_to_previous_attractor
    )

    displacements = tuple(
        step.conditioned_state_shift_magnitude
        for step in items
    )

    divergences = tuple(
        step.cue_induced_state_divergence
        for step in items
    )

    gains = tuple(
        step.target_weight_gain
        for step in items
    )

    return ConditionedTrajectorySummary(
        step_count=len(
            items
        ),
        unique_conditioned_attractor_count=len(
            {
                attractor_id
                for attractor_id in conditioned_ids
                if attractor_id is not None
            }
        ),
        switch_count=switch_count,
        return_count=return_count,
        cumulative_conditioned_displacement=sum(
            displacements
        ),
        cumulative_cue_induced_divergence=sum(
            divergences
        ),
        mean_conditioned_displacement=(
            sum(
                displacements
            )
            / len(
                displacements
            )
        ),
        mean_cue_induced_divergence=(
            sum(
                divergences
            )
            / len(
                divergences
            )
        ),
        mean_target_weight_gain=(
            sum(
                gains
            )
            / len(
                gains
            )
        ),
        continuity_preserved=(
            conditioned_trajectory_is_continuous(
                items
            )
        ),
        all_steps_policy_free=all(
            conditioned_dynamics_is_policy_free(
                step.dynamics
            )
            for step in items
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "trajectory_summary_is_descriptive": True,
            "memory_mutated": False,
            "learning_applied": False,
            "biological_conditioning_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def build_conditioned_trajectory(
    *,
    trajectory_id: str,
    initial_state: DynamicsState,
    attractors: Iterable[
        ExperienceAttractor
    ],
    cue_sequence: Iterable[
        ConditioningCue
    ],
    conditioning_memory: ConditioningMemory,
    prestress_config: ConditionedPrestressConfig | None = None,
    generalization_scale: float = 1.0,
    distance_scale: float = 1.0,
    step_size: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> ConditionedTrajectory:
    attractor_items = tuple(
        attractors
    )

    if not attractor_items:
        raise ConditionedTrajectoryError(
            "attractors must not be empty"
        )

    cue_items = tuple(
        cue_sequence
    )

    if not cue_items:
        raise ConditionedTrajectoryError(
            "cue_sequence must not be empty"
        )

    steps = []
    state = initial_state
    seen_conditioned: list[
        str | None
    ] = []

    for index, cue in enumerate(
        cue_items
    ):
        dynamics = run_conditioned_dynamics(
            result_id=(
                f"{trajectory_id}::step_{index}"
            ),
            state=state,
            attractors=attractor_items,
            query_cue=cue,
            conditioning_memory=conditioning_memory,
            prestress_config=prestress_config,
            generalization_scale=generalization_scale,
            distance_scale=distance_scale,
            step_size=step_size,
        )

        conditioned_dominant = (
            dynamics.conditioned.dominant_attractor_id
        )

        baseline_dominant = (
            dynamics.baseline.dominant_attractor_id
        )

        previous_conditioned = (
            seen_conditioned[
                -1
            ]
            if seen_conditioned
            else None
        )

        switched = (
            index > 0
            and conditioned_dominant
            != previous_conditioned
        )

        returned = (
            index > 1
            and conditioned_dominant
            in seen_conditioned[
                :-1
            ]
            and conditioned_dominant
            != previous_conditioned
        )

        step = ConditionedTrajectoryStep(
            step_index=index,
            cue=cue,
            state_before=state,
            dynamics=dynamics,
            baseline_dominant_attractor_id=(
                baseline_dominant
            ),
            conditioned_dominant_attractor_id=(
                conditioned_dominant
            ),
            conditioned_switched_from_previous=(
                switched
            ),
            conditioned_returned_to_previous_attractor=(
                returned
            ),
            target_weight_gain=(
                conditioned_target_weight_gain(
                    dynamics
                )
            ),
            cue_induced_state_divergence=(
                dynamics.state_after_distance
            ),
            conditioned_state_shift_magnitude=(
                state_shift_magnitude(
                    dynamics.conditioned
                )
            ),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "memory_mutated": False,
                "learning_applied": False,
                "action_selected": False,
                "policy_modified": False,
                "biological_conditioning_claimed": False,
                "causal_truth_inferred": False,
            },
        )

        steps.append(
            step
        )

        seen_conditioned.append(
            conditioned_dominant
        )

        state = dynamics.conditioned.state_after

    step_items = tuple(
        steps
    )

    summary = summarize_conditioned_trajectory(
        step_items
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "conditioned_trajectory_mode": "descriptive",
        "memory_mutated": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_conditioning_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return ConditionedTrajectory(
        trajectory_id=trajectory_id,
        initial_state=initial_state,
        steps=step_items,
        final_state=state,
        summary=summary,
        metadata=merged_metadata,
    )


def conditioned_trajectory_is_policy_free(
    trajectory: ConditionedTrajectory,
) -> bool:
    return (
        trajectory.metadata.get(
            "action_selected"
        )
        is False
        and trajectory.metadata.get(
            "policy_modified"
        )
        is False
        and trajectory.metadata.get(
            "memory_mutated"
        )
        is False
        and trajectory.metadata.get(
            "learning_applied"
        )
        is False
        and trajectory.summary.all_steps_policy_free
    )


__all__ = [
    "ConditionedTrajectory",
    "ConditionedTrajectoryError",
    "ConditionedTrajectoryStep",
    "ConditionedTrajectorySummary",
    "SCHEMA_VERSION",
    "baseline_dominant_series",
    "build_conditioned_trajectory",
    "conditioned_displacement_series",
    "conditioned_dominant_series",
    "conditioned_trajectory_continuity_break_indices",
    "conditioned_trajectory_is_continuous",
    "conditioned_trajectory_is_policy_free",
    "cue_id_series",
    "cue_induced_divergence_series",
    "return_flags",
    "state_id_series",
    "summarize_conditioned_trajectory",
    "switch_flags",
    "target_weight_gain_series",
]
