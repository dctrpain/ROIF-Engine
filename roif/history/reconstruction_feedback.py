"""
ROIF Memory 2.0
Reconstruction Feedback Core

Sixteenth layer above:
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

Purpose
-------
Introduce recursive semantic feedback:

    cue_t
    + context_t
    + reconstructed_meaning_(t-1)
        -> reconstruction_t

The previous reconstructed meaning is not treated as a new memory write.
Instead it acts as a temporary feedback field that biases current
attractor reconstruction.

This module computes:
- baseline contextual reconstruction
- feedback compatibility with each attractor
- feedback gain per attractor
- feedback-weighted reconstruction weights
- reconstructed vector under feedback
- dominant-attractor change
- ambiguity change
- semantic displacement from baseline reconstruction

This module does NOT:
- mutate memory
- learn
- mutate attractors
- select actions
- modify policy
- diagnose
- claim a biological recursive mechanism
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, sqrt
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


SCHEMA_VERSION = "reconstruction_feedback_v1"


class ReconstructionFeedbackError(ValueError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _validate_id(name: str, value: str) -> str:
    cleaned = str(value).strip()

    if not cleaned:
        raise ReconstructionFeedbackError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(name: str, value: float) -> float:
    x = float(value)

    if not isfinite(x):
        raise ReconstructionFeedbackError(
            f"{name} must be finite"
        )

    return x


def _validate_nonnegative(name: str, value: float) -> float:
    x = _validate_finite(name, value)

    if x < 0.0:
        raise ReconstructionFeedbackError(
            f"{name} must be non-negative"
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
        raise ReconstructionFeedbackError(
            f"{name} must not be empty"
        )

    return vector


def _distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ReconstructionFeedbackError(
            "vector dimensions must match"
        )

    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right)
        )
    )


def _weighted_mean(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, ...]:
    if not vectors:
        raise ReconstructionFeedbackError(
            "vectors must not be empty"
        )

    if len(vectors) != len(weights):
        raise ReconstructionFeedbackError(
            "vectors and weights must have equal length"
        )

    width = len(vectors[0])

    if width == 0:
        raise ReconstructionFeedbackError(
            "vectors must not be empty"
        )

    if any(len(vector) != width for vector in vectors):
        raise ReconstructionFeedbackError(
            "vector dimensions must match"
        )

    total = sum(float(weight) for weight in weights)

    if total <= 0.0:
        return tuple(
            0.0
            for _ in range(width)
        )

    return tuple(
        sum(
            float(vector[index]) * float(weight)
            for vector, weight in zip(vectors, weights)
        )
        / total
        for index in range(width)
    )


@dataclass(frozen=True, slots=True)
class ReconstructionFeedbackConfig:
    feedback_gain: float = 1.0
    feedback_scale: float = 1.0
    feedback_contrast_power: float = 4.0
    max_feedback_multiplier: float = 3.0
    baseline_cue_weight: float = 0.25
    context_gain: float = 2.0
    compatibility_scale: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "feedback_gain",
            "feedback_scale",
            "feedback_contrast_power",
            "max_feedback_multiplier",
            "baseline_cue_weight",
            "context_gain",
            "compatibility_scale",
        ):
            object.__setattr__(
                self,
                name,
                _validate_nonnegative(
                    name,
                    getattr(self, name),
                ),
            )

        if self.feedback_scale <= 0.0:
            raise ReconstructionFeedbackError(
                "feedback_scale must be positive"
            )

        if self.feedback_contrast_power <= 0.0:
            raise ReconstructionFeedbackError(
                "feedback_contrast_power must be positive"
            )

        if self.compatibility_scale <= 0.0:
            raise ReconstructionFeedbackError(
                "compatibility_scale must be positive"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class FeedbackContribution:
    attractor_id: str

    baseline_weight: float
    feedback_distance: float
    feedback_compatibility: float
    feedback_multiplier: float

    raw_feedback_weight: float
    normalized_feedback_weight: float
    weight_delta: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attractor_id",
            _validate_id(
                "attractor_id",
                self.attractor_id,
            ),
        )

        for name in (
            "baseline_weight",
            "feedback_distance",
            "feedback_compatibility",
            "feedback_multiplier",
            "raw_feedback_weight",
            "normalized_feedback_weight",
            "weight_delta",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(self, name),
                ),
            )

        if self.feedback_distance < 0.0:
            raise ReconstructionFeedbackError(
                "feedback_distance must be non-negative"
            )

        for name in (
            "baseline_weight",
            "feedback_compatibility",
            "normalized_feedback_weight",
        ):
            value = getattr(self, name)

            if not 0.0 <= value <= 1.0:
                raise ReconstructionFeedbackError(
                    f"{name} must be within [0,1]"
                )

        if self.feedback_multiplier < 0.0:
            raise ReconstructionFeedbackError(
                "feedback_multiplier must be non-negative"
            )

        if self.raw_feedback_weight < 0.0:
            raise ReconstructionFeedbackError(
                "raw_feedback_weight must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionFeedbackResult:
    feedback_id: str

    cue: ConditioningCue
    context: AssociativeContext
    previous_reconstructed_vector: tuple[float, ...]

    baseline: ContextualReconstructionResult

    contributions: tuple[
        FeedbackContribution,
        ...,
    ]

    feedback_reconstructed_vector: tuple[float, ...]

    baseline_dominant_attractor_id: str
    feedback_dominant_attractor_id: str
    dominant_attractor_changed: bool

    baseline_ambiguity: float
    feedback_ambiguity: float
    ambiguity_delta: float

    feedback_semantic_displacement: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feedback_id",
            _validate_id(
                "feedback_id",
                self.feedback_id,
            ),
        )

        object.__setattr__(
            self,
            "previous_reconstructed_vector",
            _validate_vector(
                "previous_reconstructed_vector",
                self.previous_reconstructed_vector,
            ),
        )

        if not self.contributions:
            raise ReconstructionFeedbackError(
                "contributions must not be empty"
            )

        object.__setattr__(
            self,
            "contributions",
            tuple(self.contributions),
        )

        object.__setattr__(
            self,
            "feedback_reconstructed_vector",
            _validate_vector(
                "feedback_reconstructed_vector",
                self.feedback_reconstructed_vector,
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
            "baseline_ambiguity",
            "feedback_ambiguity",
            "ambiguity_delta",
            "feedback_semantic_displacement",
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
            "baseline_ambiguity",
            "feedback_ambiguity",
        ):
            value = getattr(self, name)

            if not 0.0 <= value <= 1.0:
                raise ReconstructionFeedbackError(
                    f"{name} must be within [0,1]"
                )

        if self.feedback_semantic_displacement < 0.0:
            raise ReconstructionFeedbackError(
                "feedback_semantic_displacement must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


def _baseline_weight_map(
    baseline: ContextualReconstructionResult,
) -> dict[str, float]:
    return {
        contribution.attractor_id:
            contribution.normalized_reconstruction_weight
        for contribution in baseline.contributions
    }


def apply_reconstruction_feedback(
    *,
    feedback_id: str,
    cue: ConditioningCue,
    context: AssociativeContext,
    attractors: Iterable[ExperienceAttractor],
    previous_reconstructed_vector: Sequence[float],
    config: ReconstructionFeedbackConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReconstructionFeedbackResult:
    attractor_items = tuple(attractors)

    if not attractor_items:
        raise ReconstructionFeedbackError(
            "attractors must not be empty"
        )

    actual_config = (
        config
        if config is not None
        else ReconstructionFeedbackConfig()
    )

    previous_vector = _validate_vector(
        "previous_reconstructed_vector",
        previous_reconstructed_vector,
    )

    width = len(attractor_items[0].center)

    if len(previous_vector) != width:
        raise ReconstructionFeedbackError(
            "previous reconstructed vector dimension must match attractor space"
        )

    if any(
        len(attractor.center) != width
        for attractor in attractor_items
    ):
        raise ReconstructionFeedbackError(
            "all attractor centers must have equal dimension"
        )

    baseline = reconstruct_contextual_meaning(
        reconstruction_id=f"{feedback_id}::baseline",
        cue=cue,
        context=context,
        attractors=attractor_items,
        compatibility_scale=actual_config.compatibility_scale,
        baseline_cue_weight=actual_config.baseline_cue_weight,
        context_gain=actual_config.context_gain,
    )

    if not reconstruction_is_policy_free(
        baseline
    ):
        raise ReconstructionFeedbackError(
            "baseline reconstruction must be policy-free"
        )

    baseline_weights = _baseline_weight_map(
        baseline
    )

    compatibility_rows = []

    for attractor in attractor_items:
        baseline_weight = baseline_weights[
            attractor.attractor_id
        ]

        feedback_distance = _distance(
            previous_vector,
            attractor.center,
        )

        feedback_compatibility = exp(
            -feedback_distance
            / actual_config.feedback_scale
        )

        compatibility_rows.append(
            (
                attractor,
                baseline_weight,
                feedback_distance,
                feedback_compatibility,
            )
        )

    max_feedback_compatibility = max(
        row[3]
        for row in compatibility_rows
    )

    rows = []

    for (
        attractor,
        baseline_weight,
        feedback_distance,
        feedback_compatibility,
    ) in compatibility_rows:
        relative_compatibility = (
            feedback_compatibility
            / max_feedback_compatibility
            if max_feedback_compatibility > 0.0
            else 0.0
        )

        contrastive_compatibility = (
            relative_compatibility
            ** actual_config.feedback_contrast_power
        )

        feedback_multiplier = min(
            actual_config.max_feedback_multiplier,
            (
                1.0
                + actual_config.feedback_gain
                * contrastive_compatibility
            ),
        )

        raw_feedback_weight = (
            baseline_weight
            * feedback_multiplier
        )

        rows.append(
            (
                attractor,
                baseline_weight,
                feedback_distance,
                feedback_compatibility,
                feedback_multiplier,
                raw_feedback_weight,
            )
        )

    raw_total = sum(
        row[5]
        for row in rows
    )

    if raw_total <= 0.0:
        normalized = tuple(
            1.0 / len(rows)
            for _ in rows
        )
    else:
        normalized = tuple(
            row[5] / raw_total
            for row in rows
        )

    contributions = tuple(
        FeedbackContribution(
            attractor_id=row[0].attractor_id,
            baseline_weight=row[1],
            feedback_distance=row[2],
            feedback_compatibility=row[3],
            feedback_multiplier=row[4],
            raw_feedback_weight=row[5],
            normalized_feedback_weight=weight,
            weight_delta=(
                weight - row[1]
            ),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "memory_mutated": False,
                "learning_applied": False,
                "action_selected": False,
                "policy_modified": False,
                "biological_feedback_claimed": False,
                "causal_truth_inferred": False,
            },
        )
        for row, weight in zip(
            rows,
            normalized,
        )
    )

    feedback_attractor_reconstruction = _weighted_mean(
        tuple(
            attractor.center
            for attractor in attractor_items
        ),
        tuple(
            contribution.normalized_feedback_weight
            for contribution in contributions
        ),
    )

    baseline_attractor_reconstruction = _weighted_mean(
        tuple(
            attractor.center
            for attractor in attractor_items
        ),
        tuple(
            contribution.normalized_reconstruction_weight
            for contribution in baseline.contributions
        ),
    )

    if actual_config.baseline_cue_weight > 0.0:
        cue_anchor = tuple(
            (
                (
                    baseline.reconstructed_vector[index]
                    * (
                        actual_config.baseline_cue_weight
                        + 1.0
                    )
                )
                - baseline_attractor_reconstruction[index]
            )
            / actual_config.baseline_cue_weight
            for index in range(
                len(
                    baseline.reconstructed_vector
                )
            )
        )

        reconstructed_vector = _weighted_mean(
            (
                cue_anchor,
                feedback_attractor_reconstruction,
            ),
            (
                actual_config.baseline_cue_weight,
                1.0,
            ),
        )
    else:
        reconstructed_vector = (
            feedback_attractor_reconstruction
        )

    dominant = sorted(
        contributions,
        key=lambda contribution: (
            -contribution.normalized_feedback_weight,
            contribution.attractor_id,
        ),
    )[0]

    feedback_ambiguity = max(
        0.0,
        min(
            1.0,
            1.0
            - sum(
                contribution.normalized_feedback_weight ** 2
                for contribution in contributions
            ),
        ),
    )

    feedback_displacement = _distance(
        baseline.reconstructed_vector,
        reconstructed_vector,
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "reconstruction_feedback_mode": "descriptive",
        "context_id": context.context_id,
        "feedback_contrast_power": actual_config.feedback_contrast_power,
        "baseline_cue_blend_preserved": True,
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

    return ReconstructionFeedbackResult(
        feedback_id=feedback_id,
        cue=cue,
        context=context,
        previous_reconstructed_vector=previous_vector,
        baseline=baseline,
        contributions=contributions,
        feedback_reconstructed_vector=reconstructed_vector,
        baseline_dominant_attractor_id=(
            baseline.dominant_attractor_id
        ),
        feedback_dominant_attractor_id=(
            dominant.attractor_id
        ),
        dominant_attractor_changed=(
            baseline.dominant_attractor_id
            != dominant.attractor_id
        ),
        baseline_ambiguity=baseline.ambiguity,
        feedback_ambiguity=feedback_ambiguity,
        ambiguity_delta=(
            feedback_ambiguity
            - baseline.ambiguity
        ),
        feedback_semantic_displacement=(
            feedback_displacement
        ),
        metadata=merged_metadata,
    )


def feedback_contribution_by_id(
    result: ReconstructionFeedbackResult,
    attractor_id: str,
) -> FeedbackContribution:
    target = _validate_id(
        "attractor_id",
        attractor_id,
    )

    for contribution in result.contributions:
        if contribution.attractor_id == target:
            return contribution

    raise ReconstructionFeedbackError(
        f"unknown attractor_id: {target}"
    )


def feedback_weight_series(
    result: ReconstructionFeedbackResult,
) -> tuple[float, ...]:
    return tuple(
        contribution.normalized_feedback_weight
        for contribution in result.contributions
    )


def feedback_weight_delta_series(
    result: ReconstructionFeedbackResult,
) -> tuple[float, ...]:
    return tuple(
        contribution.weight_delta
        for contribution in result.contributions
    )


def reconstruction_feedback_is_policy_free(
    result: ReconstructionFeedbackResult,
) -> bool:
    return (
        result.metadata.get(
            "action_selected"
        )
        is False
        and result.metadata.get(
            "policy_modified"
        )
        is False
        and result.metadata.get(
            "memory_mutated"
        )
        is False
        and result.metadata.get(
            "learning_applied"
        )
        is False
        and reconstruction_is_policy_free(
            result.baseline
        )
    )


__all__ = [
    "FeedbackContribution",
    "ReconstructionFeedbackConfig",
    "ReconstructionFeedbackError",
    "ReconstructionFeedbackResult",
    "SCHEMA_VERSION",
    "apply_reconstruction_feedback",
    "feedback_contribution_by_id",
    "feedback_weight_delta_series",
    "feedback_weight_series",
    "reconstruction_feedback_is_policy_free",
]


