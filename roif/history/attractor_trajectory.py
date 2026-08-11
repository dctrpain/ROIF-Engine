"""
ROIF Memory 2.0
Attractor Trajectory Core

Sixth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor
    AttractorDynamics

Purpose
-------
Track attractor-state dynamics through time:

    state_0
      -> dynamics_0
      -> state_1
      -> dynamics_1
      -> state_2
      -> ...

The trajectory layer measures:
- continuity of state propagation
- dominant-attractor sequence
- attractor switching
- dwell lengths
- returns to previously dominant attractors
- cumulative state displacement
- competition / uncertainty trajectory
- capture-margin trajectory

This module remains descriptive:
- no memory mutation
- no learning
- no policy modification
- no action selection
- no diagnosis
- no biological attractor claim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_attractor import (
    ExperienceAttractor,
)
from roif.history.attractor_dynamics import (
    AttractorDynamicsResult,
    AttractorPrestress,
    DynamicsState,
    dynamics_is_policy_free,
    run_attractor_dynamics,
    state_shift_magnitude,
)


SCHEMA_VERSION = "attractor_trajectory_v1"


class AttractorTrajectoryError(ValueError):
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
        raise AttractorTrajectoryError(
            f"{name} must not be empty"
        )
    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)
    if not isfinite(x):
        raise AttractorTrajectoryError(
            f"{name} must be finite"
        )
    return x


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    step_index: int
    state_before: DynamicsState
    prestress: AttractorPrestress | None
    dynamics: AttractorDynamicsResult
    dominant_attractor_id: str | None
    switched_from_previous: bool
    returned_to_previous_attractor: bool
    displacement_magnitude: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise AttractorTrajectoryError(
                "step_index must be non-negative"
            )

        object.__setattr__(
            self,
            "displacement_magnitude",
            _validate_finite(
                "displacement_magnitude",
                self.displacement_magnitude,
            ),
        )

        if self.displacement_magnitude < 0.0:
            raise AttractorTrajectoryError(
                "displacement_magnitude must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AttractorTrajectorySummary:
    step_count: int
    unique_dominant_attractor_count: int

    switch_count: int
    return_count: int

    cumulative_displacement: float
    mean_displacement: float

    mean_competition_entropy: float
    mean_capture_margin: float

    longest_dwell_length: int

    continuity_preserved: bool
    all_steps_policy_free: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_count < 1:
            raise AttractorTrajectoryError(
                "step_count must be >= 1"
            )

        for name in (
            "unique_dominant_attractor_count",
            "switch_count",
            "return_count",
            "longest_dwell_length",
        ):
            value = int(
                getattr(
                    self,
                    name,
                )
            )
            if value < 0:
                raise AttractorTrajectoryError(
                    f"{name} must be non-negative"
                )

        for name in (
            "cumulative_displacement",
            "mean_displacement",
            "mean_competition_entropy",
            "mean_capture_margin",
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
            "cumulative_displacement",
            "mean_displacement",
            "mean_competition_entropy",
            "mean_capture_margin",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise AttractorTrajectoryError(
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
class AttractorTrajectory:
    trajectory_id: str
    initial_state: DynamicsState
    steps: tuple[
        TrajectoryStep,
        ...,
    ]
    final_state: DynamicsState
    summary: AttractorTrajectorySummary

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
            raise AttractorTrajectoryError(
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


def trajectory_continuity_break_indices(
    steps: Iterable[
        TrajectoryStep
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
            ].dynamics.state_after
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


def trajectory_is_continuous(
    steps: Iterable[
        TrajectoryStep
    ],
) -> bool:
    return not trajectory_continuity_break_indices(
        steps
    )


def dominant_attractor_series(
    trajectory: AttractorTrajectory,
) -> tuple[str | None, ...]:
    return tuple(
        step.dominant_attractor_id
        for step in trajectory.steps
    )


def switch_flags(
    trajectory: AttractorTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.switched_from_previous
        for step in trajectory.steps
    )


def return_flags(
    trajectory: AttractorTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.returned_to_previous_attractor
        for step in trajectory.steps
    )


def displacement_series(
    trajectory: AttractorTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.displacement_magnitude
        for step in trajectory.steps
    )


def competition_entropy_series(
    trajectory: AttractorTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.dynamics.competition_entropy_proxy
        for step in trajectory.steps
    )


def capture_margin_series(
    trajectory: AttractorTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.dynamics.capture_margin
        for step in trajectory.steps
    )


def _longest_dwell(
    dominant_ids: Sequence[
        str | None
    ],
) -> int:
    if not dominant_ids:
        return 0

    longest = 1
    current = 1

    for previous, current_id in zip(
        dominant_ids,
        dominant_ids[
            1:
        ],
    ):
        if (
            current_id
            == previous
        ):
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 1

    return longest


def summarize_attractor_trajectory(
    steps: Iterable[
        TrajectoryStep
    ],
) -> AttractorTrajectorySummary:
    items = tuple(
        steps
    )

    if not items:
        raise AttractorTrajectoryError(
            "cannot summarize empty trajectory"
        )

    dominant_ids = tuple(
        step.dominant_attractor_id
        for step in items
    )

    switch_count = sum(
        1
        for step in items
        if step.switched_from_previous
    )

    return_count = sum(
        1
        for step in items
        if step.returned_to_previous_attractor
    )

    displacements = tuple(
        step.displacement_magnitude
        for step in items
    )

    entropy_values = tuple(
        step.dynamics.competition_entropy_proxy
        for step in items
    )

    margin_values = tuple(
        step.dynamics.capture_margin
        for step in items
    )

    return AttractorTrajectorySummary(
        step_count=len(
            items
        ),
        unique_dominant_attractor_count=len(
            {
                attractor_id
                for attractor_id in dominant_ids
                if attractor_id is not None
            }
        ),
        switch_count=(
            switch_count
        ),
        return_count=(
            return_count
        ),
        cumulative_displacement=sum(
            displacements
        ),
        mean_displacement=(
            sum(
                displacements
            )
            / len(
                displacements
            )
        ),
        mean_competition_entropy=(
            sum(
                entropy_values
            )
            / len(
                entropy_values
            )
        ),
        mean_capture_margin=(
            sum(
                margin_values
            )
            / len(
                margin_values
            )
        ),
        longest_dwell_length=(
            _longest_dwell(
                dominant_ids
            )
        ),
        continuity_preserved=(
            trajectory_is_continuous(
                items
            )
        ),
        all_steps_policy_free=all(
            dynamics_is_policy_free(
                step.dynamics
            )
            for step in items
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "trajectory_summary_is_descriptive": True,
            "biological_attractor_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def build_attractor_trajectory(
    *,
    trajectory_id: str,
    initial_state: DynamicsState,
    attractors: Iterable[
        ExperienceAttractor
    ],
    prestress_sequence: Iterable[
        AttractorPrestress | None
    ],
    distance_scale: float = 1.0,
    step_size: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> AttractorTrajectory:
    attractor_items = tuple(
        attractors
    )

    if not attractor_items:
        raise AttractorTrajectoryError(
            "attractors must not be empty"
        )

    prestress_items = tuple(
        prestress_sequence
    )

    if not prestress_items:
        raise AttractorTrajectoryError(
            "prestress_sequence must not be empty"
        )

    steps = []
    state = initial_state
    seen_dominants: list[
        str | None
    ] = []

    for index, prestress in enumerate(
        prestress_items
    ):
        dynamics = run_attractor_dynamics(
            state=state,
            attractors=attractor_items,
            prestress=prestress,
            distance_scale=distance_scale,
            step_size=step_size,
        )

        dominant = (
            dynamics.dominant_attractor_id
        )

        previous_dominant = (
            seen_dominants[
                -1
            ]
            if seen_dominants
            else None
        )

        switched = (
            index > 0
            and dominant
            != previous_dominant
        )

        returned = (
            index > 1
            and dominant
            in seen_dominants[
                :-1
            ]
            and dominant
            != previous_dominant
        )

        step = TrajectoryStep(
            step_index=index,
            state_before=state,
            prestress=prestress,
            dynamics=dynamics,
            dominant_attractor_id=dominant,
            switched_from_previous=(
                switched
            ),
            returned_to_previous_attractor=(
                returned
            ),
            displacement_magnitude=(
                state_shift_magnitude(
                    dynamics
                )
            ),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "memory_mutated": False,
                "learning_applied": False,
                "action_selected": False,
                "policy_modified": False,
                "biological_attractor_claimed": False,
                "causal_truth_inferred": False,
            },
        )

        steps.append(
            step
        )

        seen_dominants.append(
            dominant
        )

        state = dynamics.state_after

    step_items = tuple(
        steps
    )

    summary = summarize_attractor_trajectory(
        step_items
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_mode": "descriptive",
        "memory_mutated": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_attractor_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return AttractorTrajectory(
        trajectory_id=trajectory_id,
        initial_state=initial_state,
        steps=step_items,
        final_state=state,
        summary=summary,
        metadata=merged_metadata,
    )


def trajectory_is_policy_free(
    trajectory: AttractorTrajectory,
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


def state_id_series(
    trajectory: AttractorTrajectory,
) -> tuple[str, ...]:
    return (
        trajectory.initial_state.state_id,
        *tuple(
            step.dynamics.state_after.state_id
            for step in trajectory.steps
        ),
    )


__all__ = [
    "AttractorTrajectory",
    "AttractorTrajectoryError",
    "AttractorTrajectorySummary",
    "SCHEMA_VERSION",
    "TrajectoryStep",
    "build_attractor_trajectory",
    "capture_margin_series",
    "competition_entropy_series",
    "displacement_series",
    "dominant_attractor_series",
    "return_flags",
    "state_id_series",
    "summarize_attractor_trajectory",
    "switch_flags",
    "trajectory_continuity_break_indices",
    "trajectory_is_continuous",
    "trajectory_is_policy_free",
]
