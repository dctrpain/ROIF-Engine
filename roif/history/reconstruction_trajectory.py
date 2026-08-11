"""
ROIF Memory 2.0
Reconstruction Trajectory Core

Fifteenth layer above:
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
    ContextualTrajectory
    ContextualReconstruction

Purpose
-------
Track contextual reconstruction through time.

Each step combines:
    cue_t + context_t -> reconstructed meaning_t

The previous reconstructed meaning is carried forward as the trajectory's
semantic state descriptor. The layer measures:
- cue/context sequence identity
- dominant reconstructed-attractor sequence
- meaning switching and returns
- per-step semantic shift from cue
- inter-step reconstruction drift
- cumulative semantic drift
- cumulative cue-to-meaning shift
- ambiguity trajectory
- dominant-weight trajectory

This module remains descriptive:
- no learning
- no source memory mutation
- no attractor mutation
- no action selection
- no policy modification
- no diagnosis
- no biological semantic-reconstruction claim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_attractor import ExperienceAttractor
from roif.history.experience_conditioning import ConditioningCue
from roif.history.associative_context import AssociativeContext
from roif.history.contextual_reconstruction import (
    ContextualReconstructionResult,
    reconstruct_contextual_meaning,
    reconstruction_is_policy_free,
)


SCHEMA_VERSION = "reconstruction_trajectory_v1"


class ReconstructionTrajectoryError(ValueError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _validate_id(name: str, value: str) -> str:
    cleaned = str(value).strip()

    if not cleaned:
        raise ReconstructionTrajectoryError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(name: str, value: float) -> float:
    x = float(value)

    if not isfinite(x):
        raise ReconstructionTrajectoryError(
            f"{name} must be finite"
        )

    return x


def _vector_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ReconstructionTrajectoryError(
            "vector dimensions must match"
        )

    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right)
        )
    )


@dataclass(frozen=True, slots=True)
class ReconstructionFrame:
    cue: ConditioningCue
    context: AssociativeContext

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionTrajectoryStep:
    step_index: int
    frame: ReconstructionFrame
    reconstruction: ContextualReconstructionResult

    previous_reconstructed_vector: tuple[float, ...] | None

    dominant_attractor_id: str
    switched_from_previous: bool
    returned_to_previous_attractor: bool

    semantic_shift_from_cue: float
    inter_step_drift: float
    ambiguity: float
    dominant_weight: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ReconstructionTrajectoryError(
                "step_index must be non-negative"
            )

        if self.previous_reconstructed_vector is not None:
            object.__setattr__(
                self,
                "previous_reconstructed_vector",
                tuple(
                    float(value)
                    for value in self.previous_reconstructed_vector
                ),
            )

        object.__setattr__(
            self,
            "dominant_attractor_id",
            _validate_id(
                "dominant_attractor_id",
                self.dominant_attractor_id,
            ),
        )

        for name in (
            "semantic_shift_from_cue",
            "inter_step_drift",
            "ambiguity",
            "dominant_weight",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(self, name),
                ),
            )

        if self.semantic_shift_from_cue < 0.0:
            raise ReconstructionTrajectoryError(
                "semantic_shift_from_cue must be non-negative"
            )

        if self.inter_step_drift < 0.0:
            raise ReconstructionTrajectoryError(
                "inter_step_drift must be non-negative"
            )

        for name in (
            "ambiguity",
            "dominant_weight",
        ):
            value = getattr(self, name)

            if not 0.0 <= value <= 1.0:
                raise ReconstructionTrajectoryError(
                    f"{name} must be within [0,1]"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionTrajectorySummary:
    step_count: int
    unique_dominant_attractor_count: int

    switch_count: int
    return_count: int

    cumulative_inter_step_drift: float
    cumulative_semantic_shift: float

    mean_inter_step_drift: float
    mean_semantic_shift: float
    mean_ambiguity: float
    mean_dominant_weight: float

    all_steps_policy_free: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_count < 1:
            raise ReconstructionTrajectoryError(
                "step_count must be >= 1"
            )

        for name in (
            "unique_dominant_attractor_count",
            "switch_count",
            "return_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ReconstructionTrajectoryError(
                    f"{name} must be non-negative"
                )

        for name in (
            "cumulative_inter_step_drift",
            "cumulative_semantic_shift",
            "mean_inter_step_drift",
            "mean_semantic_shift",
            "mean_ambiguity",
            "mean_dominant_weight",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(self, name),
                ),
            )

        for name in (
            "cumulative_inter_step_drift",
            "cumulative_semantic_shift",
            "mean_inter_step_drift",
            "mean_semantic_shift",
        ):
            if getattr(self, name) < 0.0:
                raise ReconstructionTrajectoryError(
                    f"{name} must be non-negative"
                )

        for name in (
            "mean_ambiguity",
            "mean_dominant_weight",
        ):
            value = getattr(self, name)

            if not 0.0 <= value <= 1.0:
                raise ReconstructionTrajectoryError(
                    f"{name} must be within [0,1]"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionTrajectory:
    trajectory_id: str

    frames: tuple[
        ReconstructionFrame,
        ...,
    ]

    steps: tuple[
        ReconstructionTrajectoryStep,
        ...,
    ]

    initial_reconstructed_vector: tuple[float, ...] | None
    final_reconstructed_vector: tuple[float, ...]

    summary: ReconstructionTrajectorySummary

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

        if not self.frames:
            raise ReconstructionTrajectoryError(
                "frames must not be empty"
            )

        if not self.steps:
            raise ReconstructionTrajectoryError(
                "steps must not be empty"
            )

        object.__setattr__(
            self,
            "frames",
            tuple(self.frames),
        )

        object.__setattr__(
            self,
            "steps",
            tuple(self.steps),
        )

        if self.initial_reconstructed_vector is not None:
            object.__setattr__(
                self,
                "initial_reconstructed_vector",
                tuple(
                    float(value)
                    for value in self.initial_reconstructed_vector
                ),
            )

        object.__setattr__(
            self,
            "final_reconstructed_vector",
            tuple(
                float(value)
                for value in self.final_reconstructed_vector
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


def frame_cue_id_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.frame.cue.cue_id
        for step in trajectory.steps
    )


def frame_context_id_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.frame.context.context_id
        for step in trajectory.steps
    )


def dominant_attractor_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.dominant_attractor_id
        for step in trajectory.steps
    )


def switch_flags(
    trajectory: ReconstructionTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.switched_from_previous
        for step in trajectory.steps
    )


def return_flags(
    trajectory: ReconstructionTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.returned_to_previous_attractor
        for step in trajectory.steps
    )


def semantic_shift_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.semantic_shift_from_cue
        for step in trajectory.steps
    )


def inter_step_drift_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.inter_step_drift
        for step in trajectory.steps
    )


def ambiguity_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.ambiguity
        for step in trajectory.steps
    )


def dominant_weight_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.dominant_weight
        for step in trajectory.steps
    )


def reconstructed_vector_series(
    trajectory: ReconstructionTrajectory,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        step.reconstruction.reconstructed_vector
        for step in trajectory.steps
    )


def summarize_reconstruction_trajectory(
    steps: Iterable[
        ReconstructionTrajectoryStep
    ],
) -> ReconstructionTrajectorySummary:
    items = tuple(steps)

    if not items:
        raise ReconstructionTrajectoryError(
            "cannot summarize empty reconstruction trajectory"
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

    drift_values = tuple(
        step.inter_step_drift
        for step in items
    )

    semantic_values = tuple(
        step.semantic_shift_from_cue
        for step in items
    )

    ambiguity_values = tuple(
        step.ambiguity
        for step in items
    )

    dominant_weight_values = tuple(
        step.dominant_weight
        for step in items
    )

    return ReconstructionTrajectorySummary(
        step_count=len(items),
        unique_dominant_attractor_count=len(
            set(dominant_ids)
        ),
        switch_count=switch_count,
        return_count=return_count,
        cumulative_inter_step_drift=sum(
            drift_values
        ),
        cumulative_semantic_shift=sum(
            semantic_values
        ),
        mean_inter_step_drift=(
            sum(drift_values)
            / len(drift_values)
        ),
        mean_semantic_shift=(
            sum(semantic_values)
            / len(semantic_values)
        ),
        mean_ambiguity=(
            sum(ambiguity_values)
            / len(ambiguity_values)
        ),
        mean_dominant_weight=(
            sum(dominant_weight_values)
            / len(dominant_weight_values)
        ),
        all_steps_policy_free=all(
            reconstruction_is_policy_free(
                step.reconstruction
            )
            for step in items
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "trajectory_summary_is_descriptive": True,
            "memory_mutated": False,
            "learning_applied": False,
            "biological_reconstruction_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def build_reconstruction_trajectory(
    *,
    trajectory_id: str,
    frames: Iterable[
        ReconstructionFrame
    ],
    attractors: Iterable[
        ExperienceAttractor
    ],
    compatibility_scale: float = 1.0,
    baseline_cue_weight: float = 0.25,
    context_gain: float = 2.0,
    metadata: Mapping[str, Any] | None = None,
) -> ReconstructionTrajectory:
    frame_items = tuple(frames)

    if not frame_items:
        raise ReconstructionTrajectoryError(
            "frames must not be empty"
        )

    attractor_items = tuple(attractors)

    if not attractor_items:
        raise ReconstructionTrajectoryError(
            "attractors must not be empty"
        )

    steps = []
    previous_vector: tuple[float, ...] | None = None
    seen_dominants: list[str] = []

    for index, frame in enumerate(frame_items):
        reconstruction = reconstruct_contextual_meaning(
            reconstruction_id=(
                f"{trajectory_id}::step_{index}"
            ),
            cue=frame.cue,
            context=frame.context,
            attractors=attractor_items,
            compatibility_scale=compatibility_scale,
            baseline_cue_weight=baseline_cue_weight,
            context_gain=context_gain,
        )

        dominant = (
            reconstruction.dominant_attractor_id
        )

        previous_dominant = (
            seen_dominants[-1]
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
            in seen_dominants[:-1]
            and dominant
            != previous_dominant
        )

        inter_step_drift = (
            0.0
            if previous_vector is None
            else _vector_distance(
                previous_vector,
                reconstruction.reconstructed_vector,
            )
        )

        step = ReconstructionTrajectoryStep(
            step_index=index,
            frame=frame,
            reconstruction=reconstruction,
            previous_reconstructed_vector=previous_vector,
            dominant_attractor_id=dominant,
            switched_from_previous=switched,
            returned_to_previous_attractor=returned,
            semantic_shift_from_cue=(
                reconstruction.semantic_shift_from_cue
            ),
            inter_step_drift=inter_step_drift,
            ambiguity=reconstruction.ambiguity,
            dominant_weight=(
                reconstruction.dominant_weight
            ),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "memory_mutated": False,
                "learning_applied": False,
                "action_selected": False,
                "policy_modified": False,
                "biological_reconstruction_claimed": False,
                "causal_truth_inferred": False,
            },
        )

        steps.append(step)
        previous_vector = (
            reconstruction.reconstructed_vector
        )
        seen_dominants.append(dominant)

    step_items = tuple(steps)

    summary = summarize_reconstruction_trajectory(
        step_items
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "reconstruction_trajectory_mode": "descriptive",
        "memory_mutated": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_reconstruction_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(metadata)
        )

    return ReconstructionTrajectory(
        trajectory_id=trajectory_id,
        frames=frame_items,
        steps=step_items,
        initial_reconstructed_vector=None,
        final_reconstructed_vector=(
            step_items[-1].reconstruction.reconstructed_vector
        ),
        summary=summary,
        metadata=merged_metadata,
    )


def reconstruction_trajectory_is_policy_free(
    trajectory: ReconstructionTrajectory,
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
    "ReconstructionFrame",
    "ReconstructionTrajectory",
    "ReconstructionTrajectoryError",
    "ReconstructionTrajectoryStep",
    "ReconstructionTrajectorySummary",
    "SCHEMA_VERSION",
    "ambiguity_series",
    "build_reconstruction_trajectory",
    "dominant_attractor_series",
    "dominant_weight_series",
    "frame_context_id_series",
    "frame_cue_id_series",
    "inter_step_drift_series",
    "reconstructed_vector_series",
    "reconstruction_trajectory_is_policy_free",
    "return_flags",
    "semantic_shift_series",
    "summarize_reconstruction_trajectory",
    "switch_flags",
]
