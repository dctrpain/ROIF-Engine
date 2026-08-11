"""
ROIF Memory 2.0
Feedback Trajectory Core

Seventeenth layer above:
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
    ReconstructionTrajectory
    ReconstructionFeedback

Purpose
-------
Track recursive reconstruction feedback through time.

The recursive loop is:

    cue_0 + context_0
        -> baseline reconstruction_0
        -> seed meaning_0

    cue_1 + context_1 + meaning_0
        -> feedback reconstruction_1
        -> meaning_1

    cue_2 + context_2 + meaning_1
        -> feedback reconstruction_2
        -> meaning_2

The layer measures:
- cue/context sequence identity
- baseline vs feedback dominant-attractor sequence
- recursive switching and returns
- feedback semantic displacement per step
- inter-step recursive drift
- cumulative feedback displacement
- cumulative recursive drift
- ambiguity trajectory
- dominant-weight trajectory
- continuity of recursive feedback propagation

This module remains descriptive:
- no memory mutation
- no learning
- no attractor mutation
- no action selection
- no policy modification
- no diagnosis
- no biological recursive-feedback claim
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
    reconstruct_contextual_meaning,
    reconstruction_is_policy_free,
)
from roif.history.reconstruction_feedback import (
    ReconstructionFeedbackConfig,
    ReconstructionFeedbackResult,
    apply_reconstruction_feedback,
    reconstruction_feedback_is_policy_free,
)


SCHEMA_VERSION = "feedback_trajectory_v1"


class FeedbackTrajectoryError(ValueError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _validate_id(name: str, value: str) -> str:
    cleaned = str(value).strip()

    if not cleaned:
        raise FeedbackTrajectoryError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(name: str, value: float) -> float:
    x = float(value)

    if not isfinite(x):
        raise FeedbackTrajectoryError(
            f"{name} must be finite"
        )

    return x


def _validate_vector(
    name: str,
    values: Sequence[float],
) -> tuple[float, ...]:
    vector = tuple(
        _validate_finite(
            f"{name}[{index}]",
            value,
        )
        for index, value in enumerate(values)
    )

    if not vector:
        raise FeedbackTrajectoryError(
            f"{name} must not be empty"
        )

    return vector


def _distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise FeedbackTrajectoryError(
            "vector dimensions must match"
        )

    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right)
        )
    )


@dataclass(frozen=True, slots=True)
class FeedbackFrame:
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
class FeedbackTrajectoryStep:
    step_index: int
    frame: FeedbackFrame

    previous_reconstructed_vector: tuple[float, ...] | None

    feedback: ReconstructionFeedbackResult | None

    baseline_dominant_attractor_id: str
    feedback_dominant_attractor_id: str

    switched_from_previous: bool
    returned_to_previous_attractor: bool

    feedback_semantic_displacement: float
    inter_step_recursive_drift: float

    baseline_ambiguity: float
    feedback_ambiguity: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise FeedbackTrajectoryError(
                "step_index must be non-negative"
            )

        if self.previous_reconstructed_vector is not None:
            object.__setattr__(
                self,
                "previous_reconstructed_vector",
                _validate_vector(
                    "previous_reconstructed_vector",
                    self.previous_reconstructed_vector,
                ),
            )

        object.__setattr__(
            self,
            "baseline_dominant_attractor_id",
            _validate_id(
                "baseline_dominant_attractor_id",
                self.baseline_dominant_attractor_id,
            ),
        )

        object.__setattr__(
            self,
            "feedback_dominant_attractor_id",
            _validate_id(
                "feedback_dominant_attractor_id",
                self.feedback_dominant_attractor_id,
            ),
        )

        for name in (
            "feedback_semantic_displacement",
            "inter_step_recursive_drift",
            "baseline_ambiguity",
            "feedback_ambiguity",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(self, name),
                ),
            )

        if self.feedback_semantic_displacement < 0.0:
            raise FeedbackTrajectoryError(
                "feedback_semantic_displacement must be non-negative"
            )

        if self.inter_step_recursive_drift < 0.0:
            raise FeedbackTrajectoryError(
                "inter_step_recursive_drift must be non-negative"
            )

        for name in (
            "baseline_ambiguity",
            "feedback_ambiguity",
        ):
            value = getattr(self, name)

            if not 0.0 <= value <= 1.0:
                raise FeedbackTrajectoryError(
                    f"{name} must be within [0,1]"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class FeedbackTrajectorySummary:
    step_count: int
    recursive_step_count: int

    unique_feedback_attractor_count: int
    switch_count: int
    return_count: int

    cumulative_feedback_semantic_displacement: float
    cumulative_recursive_drift: float

    mean_feedback_semantic_displacement: float
    mean_recursive_drift: float

    mean_baseline_ambiguity: float
    mean_feedback_ambiguity: float

    continuity_preserved: bool
    all_steps_policy_free: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_count < 1:
            raise FeedbackTrajectoryError(
                "step_count must be >= 1"
            )

        if self.recursive_step_count < 0:
            raise FeedbackTrajectoryError(
                "recursive_step_count must be non-negative"
            )

        for name in (
            "unique_feedback_attractor_count",
            "switch_count",
            "return_count",
        ):
            if int(getattr(self, name)) < 0:
                raise FeedbackTrajectoryError(
                    f"{name} must be non-negative"
                )

        for name in (
            "cumulative_feedback_semantic_displacement",
            "cumulative_recursive_drift",
            "mean_feedback_semantic_displacement",
            "mean_recursive_drift",
            "mean_baseline_ambiguity",
            "mean_feedback_ambiguity",
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
            "cumulative_feedback_semantic_displacement",
            "cumulative_recursive_drift",
            "mean_feedback_semantic_displacement",
            "mean_recursive_drift",
        ):
            if getattr(self, name) < 0.0:
                raise FeedbackTrajectoryError(
                    f"{name} must be non-negative"
                )

        for name in (
            "mean_baseline_ambiguity",
            "mean_feedback_ambiguity",
        ):
            value = getattr(self, name)

            if not 0.0 <= value <= 1.0:
                raise FeedbackTrajectoryError(
                    f"{name} must be within [0,1]"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class FeedbackTrajectory:
    trajectory_id: str

    frames: tuple[
        FeedbackFrame,
        ...,
    ]

    steps: tuple[
        FeedbackTrajectoryStep,
        ...,
    ]

    initial_reconstructed_vector: tuple[float, ...]
    final_reconstructed_vector: tuple[float, ...]

    summary: FeedbackTrajectorySummary

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
            raise FeedbackTrajectoryError(
                "frames must not be empty"
            )

        if not self.steps:
            raise FeedbackTrajectoryError(
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

        object.__setattr__(
            self,
            "initial_reconstructed_vector",
            _validate_vector(
                "initial_reconstructed_vector",
                self.initial_reconstructed_vector,
            ),
        )

        object.__setattr__(
            self,
            "final_reconstructed_vector",
            _validate_vector(
                "final_reconstructed_vector",
                self.final_reconstructed_vector,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


def frame_cue_id_series(
    trajectory: FeedbackTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.frame.cue.cue_id
        for step in trajectory.steps
    )


def frame_context_id_series(
    trajectory: FeedbackTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.frame.context.context_id
        for step in trajectory.steps
    )


def baseline_dominant_series(
    trajectory: FeedbackTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.baseline_dominant_attractor_id
        for step in trajectory.steps
    )


def feedback_dominant_series(
    trajectory: FeedbackTrajectory,
) -> tuple[str, ...]:
    return tuple(
        step.feedback_dominant_attractor_id
        for step in trajectory.steps
    )


def switch_flags(
    trajectory: FeedbackTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.switched_from_previous
        for step in trajectory.steps
    )


def return_flags(
    trajectory: FeedbackTrajectory,
) -> tuple[bool, ...]:
    return tuple(
        step.returned_to_previous_attractor
        for step in trajectory.steps
    )


def feedback_semantic_displacement_series(
    trajectory: FeedbackTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.feedback_semantic_displacement
        for step in trajectory.steps
    )


def recursive_drift_series(
    trajectory: FeedbackTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.inter_step_recursive_drift
        for step in trajectory.steps
    )


def baseline_ambiguity_series(
    trajectory: FeedbackTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.baseline_ambiguity
        for step in trajectory.steps
    )


def feedback_ambiguity_series(
    trajectory: FeedbackTrajectory,
) -> tuple[float, ...]:
    return tuple(
        step.feedback_ambiguity
        for step in trajectory.steps
    )


def reconstructed_vector_series(
    trajectory: FeedbackTrajectory,
) -> tuple[tuple[float, ...], ...]:
    vectors = []

    for step in trajectory.steps:
        if step.feedback is None:
            vectors.append(
                trajectory.initial_reconstructed_vector
            )
        else:
            vectors.append(
                step.feedback.feedback_reconstructed_vector
            )

    return tuple(vectors)


def feedback_trajectory_continuity_break_indices(
    trajectory: FeedbackTrajectory,
) -> tuple[int, ...]:
    breaks = []
    vectors = reconstructed_vector_series(
        trajectory
    )

    for index in range(
        1,
        len(trajectory.steps),
    ):
        expected_previous = vectors[index - 1]
        actual_previous = trajectory.steps[
            index
        ].previous_reconstructed_vector

        if actual_previous != expected_previous:
            breaks.append(
                index - 1
            )

    return tuple(breaks)


def feedback_trajectory_is_continuous(
    trajectory: FeedbackTrajectory,
) -> bool:
    return not feedback_trajectory_continuity_break_indices(
        trajectory
    )


def summarize_feedback_trajectory(
    trajectory_id: str,
    steps: Iterable[
        FeedbackTrajectoryStep
    ],
    initial_reconstructed_vector: Sequence[float],
) -> FeedbackTrajectorySummary:
    _validate_id(
        "trajectory_id",
        trajectory_id,
    )

    _validate_vector(
        "initial_reconstructed_vector",
        initial_reconstructed_vector,
    )

    items = tuple(
        steps
    )

    if not items:
        raise FeedbackTrajectoryError(
            "cannot summarize empty feedback trajectory"
        )

    dominant_ids = tuple(
        step.feedback_dominant_attractor_id
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

    feedback_displacements = tuple(
        step.feedback_semantic_displacement
        for step in items
    )

    recursive_drifts = tuple(
        step.inter_step_recursive_drift
        for step in items
    )

    baseline_ambiguities = tuple(
        step.baseline_ambiguity
        for step in items
    )

    feedback_ambiguities = tuple(
        step.feedback_ambiguity
        for step in items
    )

    all_policy_free = all(
        (
            step.feedback is None
            or reconstruction_feedback_is_policy_free(
                step.feedback
            )
        )
        for step in items
    )

    recursive_step_count = sum(
        1
        for step in items
        if step.feedback is not None
    )

    continuity_preserved = True

    previous_vector = tuple(
        float(value)
        for value in initial_reconstructed_vector
    )

    for step in items:
        if step.step_index == 0:
            continue

        if step.previous_reconstructed_vector != previous_vector:
            continuity_preserved = False
            break

        if step.feedback is not None:
            previous_vector = (
                step.feedback.feedback_reconstructed_vector
            )

    return FeedbackTrajectorySummary(
        step_count=len(items),
        recursive_step_count=recursive_step_count,
        unique_feedback_attractor_count=len(
            set(dominant_ids)
        ),
        switch_count=switch_count,
        return_count=return_count,
        cumulative_feedback_semantic_displacement=sum(
            feedback_displacements
        ),
        cumulative_recursive_drift=sum(
            recursive_drifts
        ),
        mean_feedback_semantic_displacement=(
            sum(feedback_displacements)
            / len(feedback_displacements)
        ),
        mean_recursive_drift=(
            sum(recursive_drifts)
            / len(recursive_drifts)
        ),
        mean_baseline_ambiguity=(
            sum(baseline_ambiguities)
            / len(baseline_ambiguities)
        ),
        mean_feedback_ambiguity=(
            sum(feedback_ambiguities)
            / len(feedback_ambiguities)
        ),
        continuity_preserved=continuity_preserved,
        all_steps_policy_free=all_policy_free,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "trajectory_summary_is_descriptive": True,
            "memory_mutated": False,
            "learning_applied": False,
            "biological_feedback_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def build_feedback_trajectory(
    *,
    trajectory_id: str,
    frames: Iterable[
        FeedbackFrame
    ],
    attractors: Iterable[
        ExperienceAttractor
    ],
    config: ReconstructionFeedbackConfig | None = None,
    seed_from_first_baseline: bool = True,
    initial_reconstructed_vector: Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeedbackTrajectory:
    frame_items = tuple(
        frames
    )

    if not frame_items:
        raise FeedbackTrajectoryError(
            "frames must not be empty"
        )

    attractor_items = tuple(
        attractors
    )

    if not attractor_items:
        raise FeedbackTrajectoryError(
            "attractors must not be empty"
        )

    if (
        not seed_from_first_baseline
        and initial_reconstructed_vector is None
    ):
        raise FeedbackTrajectoryError(
            "initial_reconstructed_vector is required when seed_from_first_baseline is False"
        )

    actual_config = (
        config
        if config is not None
        else ReconstructionFeedbackConfig()
    )

    first_frame = frame_items[
        0
    ]

    first_baseline = reconstruct_contextual_meaning(
        reconstruction_id=(
            f"{trajectory_id}::seed_baseline"
        ),
        cue=first_frame.cue,
        context=first_frame.context,
        attractors=attractor_items,
        compatibility_scale=actual_config.compatibility_scale,
        baseline_cue_weight=actual_config.baseline_cue_weight,
        context_gain=actual_config.context_gain,
    )

    if not reconstruction_is_policy_free(
        first_baseline
    ):
        raise FeedbackTrajectoryError(
            "seed baseline must be policy-free"
        )

    if seed_from_first_baseline:
        seed_vector = (
            first_baseline.reconstructed_vector
        )
    else:
        seed_vector = _validate_vector(
            "initial_reconstructed_vector",
            initial_reconstructed_vector or (),
        )

    width = len(
        attractor_items[
            0
        ].center
    )

    if len(seed_vector) != width:
        raise FeedbackTrajectoryError(
            "initial reconstructed vector dimension must match attractor space"
        )

    steps = []
    seen_dominants: list[str] = []

    first_step = FeedbackTrajectoryStep(
        step_index=0,
        frame=first_frame,
        previous_reconstructed_vector=None,
        feedback=None,
        baseline_dominant_attractor_id=(
            first_baseline.dominant_attractor_id
        ),
        feedback_dominant_attractor_id=(
            first_baseline.dominant_attractor_id
        ),
        switched_from_previous=False,
        returned_to_previous_attractor=False,
        feedback_semantic_displacement=0.0,
        inter_step_recursive_drift=0.0,
        baseline_ambiguity=first_baseline.ambiguity,
        feedback_ambiguity=first_baseline.ambiguity,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "seed_step": True,
            "memory_mutated": False,
            "learning_applied": False,
            "action_selected": False,
            "policy_modified": False,
            "biological_feedback_claimed": False,
            "causal_truth_inferred": False,
        },
    )

    steps.append(
        first_step
    )

    seen_dominants.append(
        first_baseline.dominant_attractor_id
    )

    previous_vector = seed_vector

    for index, frame in enumerate(
        frame_items[
            1:
        ],
        start=1,
    ):
        feedback = apply_reconstruction_feedback(
            feedback_id=(
                f"{trajectory_id}::step_{index}"
            ),
            cue=frame.cue,
            context=frame.context,
            attractors=attractor_items,
            previous_reconstructed_vector=previous_vector,
            config=actual_config,
        )

        current_dominant = (
            feedback.feedback_dominant_attractor_id
        )

        previous_dominant = (
            seen_dominants[
                -1
            ]
        )

        switched = (
            current_dominant
            != previous_dominant
        )

        returned = (
            index > 1
            and current_dominant
            in seen_dominants[
                :-1
            ]
            and current_dominant
            != previous_dominant
        )

        current_vector = (
            feedback.feedback_reconstructed_vector
        )

        recursive_drift = _distance(
            previous_vector,
            current_vector,
        )

        step = FeedbackTrajectoryStep(
            step_index=index,
            frame=frame,
            previous_reconstructed_vector=(
                previous_vector
            ),
            feedback=feedback,
            baseline_dominant_attractor_id=(
                feedback.baseline_dominant_attractor_id
            ),
            feedback_dominant_attractor_id=(
                current_dominant
            ),
            switched_from_previous=switched,
            returned_to_previous_attractor=returned,
            feedback_semantic_displacement=(
                feedback.feedback_semantic_displacement
            ),
            inter_step_recursive_drift=(
                recursive_drift
            ),
            baseline_ambiguity=(
                feedback.baseline_ambiguity
            ),
            feedback_ambiguity=(
                feedback.feedback_ambiguity
            ),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "seed_step": False,
                "memory_mutated": False,
                "learning_applied": False,
                "action_selected": False,
                "policy_modified": False,
                "biological_feedback_claimed": False,
                "causal_truth_inferred": False,
            },
        )

        steps.append(
            step
        )

        seen_dominants.append(
            current_dominant
        )

        previous_vector = (
            current_vector
        )

    step_items = tuple(
        steps
    )

    summary = summarize_feedback_trajectory(
        trajectory_id,
        step_items,
        seed_vector,
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "feedback_trajectory_mode": "descriptive",
        "seed_from_first_baseline": seed_from_first_baseline,
        "memory_mutated": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_feedback_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(metadata)
        )

    return FeedbackTrajectory(
        trajectory_id=trajectory_id,
        frames=frame_items,
        steps=step_items,
        initial_reconstructed_vector=seed_vector,
        final_reconstructed_vector=(
            previous_vector
        ),
        summary=summary,
        metadata=merged_metadata,
    )


def feedback_trajectory_is_policy_free(
    trajectory: FeedbackTrajectory,
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
    "FeedbackFrame",
    "FeedbackTrajectory",
    "FeedbackTrajectoryError",
    "FeedbackTrajectoryStep",
    "FeedbackTrajectorySummary",
    "SCHEMA_VERSION",
    "baseline_ambiguity_series",
    "baseline_dominant_series",
    "build_feedback_trajectory",
    "feedback_ambiguity_series",
    "feedback_dominant_series",
    "feedback_semantic_displacement_series",
    "feedback_trajectory_continuity_break_indices",
    "feedback_trajectory_is_continuous",
    "feedback_trajectory_is_policy_free",
    "frame_context_id_series",
    "frame_cue_id_series",
    "reconstructed_vector_series",
    "recursive_drift_series",
    "return_flags",
    "summarize_feedback_trajectory",
    "switch_flags",
]
