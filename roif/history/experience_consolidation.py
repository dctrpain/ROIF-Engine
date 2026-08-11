"""
ROIF Memory 2.0
Experience Consolidation Core

Nineteenth layer above:
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
    FeedbackTrajectory
    FeedbackStability

Purpose
-------
Evaluate whether a stable recursive feedback trajectory provides enough
descriptive evidence to form an experience-consolidation candidate.

Important boundary:
    candidate != committed memory

This module intentionally does not write to ConditioningMemory or any other
persistent memory structure.

It computes:
- repetition evidence
- stability evidence
- ambiguity evidence
- dominant-attractor consistency
- semantic displacement stability
- recursive-drift stability
- consolidation score
- consolidation eligibility
- candidate representation vector
- candidate dominant attractor

This module does NOT:
- mutate memory
- learn
- commit a memory trace
- select actions
- modify policy
- diagnose
- claim biological consolidation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.history.feedback_trajectory import (
    FeedbackTrajectory,
    feedback_ambiguity_series,
    feedback_dominant_series,
    feedback_semantic_displacement_series,
    feedback_trajectory_is_policy_free,
    reconstructed_vector_series,
    recursive_drift_series,
)
from roif.history.feedback_stability import (
    FeedbackStabilityResult,
    feedback_stability_is_policy_free,
)


SCHEMA_VERSION = "experience_consolidation_v1"


class ExperienceConsolidationError(ValueError):
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
        raise ExperienceConsolidationError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise ExperienceConsolidationError(
            f"{name} must be finite"
        )

    return x


def _validate_nonnegative(
    name: str,
    value: float,
) -> float:
    x = _validate_finite(
        name,
        value,
    )

    if x < 0.0:
        raise ExperienceConsolidationError(
            f"{name} must be non-negative"
        )

    return x


def _validate_unit_interval(
    name: str,
    value: float,
) -> float:
    x = _validate_finite(
        name,
        value,
    )

    if not 0.0 <= x <= 1.0:
        raise ExperienceConsolidationError(
            f"{name} must be within [0,1]"
        )

    return x


def _mean(
    values: Sequence[float],
) -> float:
    if not values:
        return 0.0

    return sum(
        float(value)
        for value in values
    ) / len(values)


def _weighted_mean_vectors(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, ...]:
    if not vectors:
        raise ExperienceConsolidationError(
            "vectors must not be empty"
        )

    if len(vectors) != len(weights):
        raise ExperienceConsolidationError(
            "vectors and weights must have equal length"
        )

    width = len(vectors[0])

    if width == 0:
        raise ExperienceConsolidationError(
            "vectors must not be empty"
        )

    if any(
        len(vector) != width
        for vector in vectors
    ):
        raise ExperienceConsolidationError(
            "vector dimensions must match"
        )

    total = sum(
        float(weight)
        for weight in weights
    )

    if total <= 0.0:
        return tuple(
            sum(
                float(vector[index])
                for vector in vectors
            ) / len(vectors)
            for index in range(width)
        )

    return tuple(
        sum(
            float(vector[index])
            * float(weight)
            for vector, weight in zip(
                vectors,
                weights,
            )
        ) / total
        for index in range(width)
    )


@dataclass(frozen=True, slots=True)
class ExperienceConsolidationConfig:
    minimum_recursive_steps: int = 3

    minimum_stability_score: float = 0.65
    minimum_repetition_score: float = 0.60
    minimum_consistency_score: float = 0.60
    minimum_total_score: float = 0.65

    maximum_mean_ambiguity: float = 0.60
    maximum_terminal_drift: float = 0.10

    convergence_weight: float = 0.30
    persistence_weight: float = 0.20
    repetition_weight: float = 0.20
    consistency_weight: float = 0.20
    ambiguity_weight: float = 0.10

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.minimum_recursive_steps < 1:
            raise ExperienceConsolidationError(
                "minimum_recursive_steps must be >= 1"
            )

        for name in (
            "minimum_stability_score",
            "minimum_repetition_score",
            "minimum_consistency_score",
            "minimum_total_score",
            "maximum_mean_ambiguity",
        ):
            object.__setattr__(
                self,
                name,
                _validate_unit_interval(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        object.__setattr__(
            self,
            "maximum_terminal_drift",
            _validate_nonnegative(
                "maximum_terminal_drift",
                self.maximum_terminal_drift,
            ),
        )

        for name in (
            "convergence_weight",
            "persistence_weight",
            "repetition_weight",
            "consistency_weight",
            "ambiguity_weight",
        ):
            object.__setattr__(
                self,
                name,
                _validate_nonnegative(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        total_weight = (
            self.convergence_weight
            + self.persistence_weight
            + self.repetition_weight
            + self.consistency_weight
            + self.ambiguity_weight
        )

        if total_weight <= 0.0:
            raise ExperienceConsolidationError(
                "at least one consolidation weight must be positive"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ConsolidationEvidence:
    recursive_step_count: int

    repetition_score: float
    stability_score: float
    consistency_score: float
    ambiguity_quality: float
    terminal_drift_quality: float

    mean_ambiguity: float
    terminal_recursive_drift: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.recursive_step_count < 0:
            raise ExperienceConsolidationError(
                "recursive_step_count must be non-negative"
            )

        for name in (
            "repetition_score",
            "stability_score",
            "consistency_score",
            "ambiguity_quality",
            "terminal_drift_quality",
            "mean_ambiguity",
        ):
            object.__setattr__(
                self,
                name,
                _validate_unit_interval(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        object.__setattr__(
            self,
            "terminal_recursive_drift",
            _validate_nonnegative(
                "terminal_recursive_drift",
                self.terminal_recursive_drift,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperienceConsolidationCandidate:
    candidate_id: str

    source_trajectory_id: str
    source_stability_analysis_id: str

    dominant_attractor_id: str
    representation_vector: tuple[float, ...]

    evidence: ConsolidationEvidence

    consolidation_score: float
    eligible_for_consolidation: bool

    reason: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _validate_id(
                "candidate_id",
                self.candidate_id,
            ),
        )

        object.__setattr__(
            self,
            "source_trajectory_id",
            _validate_id(
                "source_trajectory_id",
                self.source_trajectory_id,
            ),
        )

        object.__setattr__(
            self,
            "source_stability_analysis_id",
            _validate_id(
                "source_stability_analysis_id",
                self.source_stability_analysis_id,
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

        vector = tuple(
            _validate_finite(
                f"representation_vector[{index}]",
                value,
            )
            for index, value in enumerate(
                self.representation_vector
            )
        )

        if not vector:
            raise ExperienceConsolidationError(
                "representation_vector must not be empty"
            )

        object.__setattr__(
            self,
            "representation_vector",
            vector,
        )

        object.__setattr__(
            self,
            "consolidation_score",
            _validate_unit_interval(
                "consolidation_score",
                self.consolidation_score,
            ),
        )

        object.__setattr__(
            self,
            "reason",
            _validate_id(
                "reason",
                self.reason,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _dominant_consistency_score(
    dominant_series: Sequence[str],
) -> tuple[str, float]:
    if not dominant_series:
        raise ExperienceConsolidationError(
            "dominant series must not be empty"
        )

    counts: dict[str, int] = {}

    for attractor_id in dominant_series:
        counts[attractor_id] = (
            counts.get(
                attractor_id,
                0,
            )
            + 1
        )

    dominant = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[0]

    return (
        dominant[0],
        dominant[1] / len(dominant_series),
    )


def _repetition_score(
    trajectory: FeedbackTrajectory,
) -> float:
    recursive_steps = trajectory.summary.recursive_step_count

    if recursive_steps <= 0:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            recursive_steps / 5.0,
        ),
    )


def _stability_score(
    stability: FeedbackStabilityResult,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            (
                stability.convergence_score
                + stability.persistence_score
                + (
                    1.0
                    - stability.divergence_score
                )
            )
            / 3.0,
        ),
    )


def _ambiguity_quality(
    mean_ambiguity: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            1.0
            - mean_ambiguity,
        ),
    )


def _terminal_drift_quality(
    terminal_drift: float,
    maximum_terminal_drift: float,
) -> float:
    if maximum_terminal_drift <= 0.0:
        return (
            1.0
            if terminal_drift <= 0.0
            else 0.0
        )

    return max(
        0.0,
        min(
            1.0,
            1.0
            - (
                terminal_drift
                / maximum_terminal_drift
            ),
        ),
    )


def _candidate_representation_vector(
    trajectory: FeedbackTrajectory,
    dominant_attractor_id: str,
) -> tuple[float, ...]:
    vectors = reconstructed_vector_series(
        trajectory
    )

    dominants = feedback_dominant_series(
        trajectory
    )

    selected_vectors = tuple(
        vector
        for vector, attractor_id in zip(
            vectors,
            dominants,
        )
        if attractor_id
        == dominant_attractor_id
    )

    if not selected_vectors:
        selected_vectors = vectors

    ambiguities = feedback_ambiguity_series(
        trajectory
    )

    selected_weights = tuple(
        max(
            1e-12,
            1.0 - ambiguity,
        )
        for ambiguity, attractor_id in zip(
            ambiguities,
            dominants,
        )
        if attractor_id
        == dominant_attractor_id
    )

    if len(
        selected_weights
    ) != len(
        selected_vectors
    ):
        selected_weights = tuple(
            1.0
            for _ in selected_vectors
        )

    return _weighted_mean_vectors(
        selected_vectors,
        selected_weights,
    )


def evaluate_experience_consolidation(
    *,
    candidate_id: str,
    trajectory: FeedbackTrajectory,
    stability: FeedbackStabilityResult,
    config: ExperienceConsolidationConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExperienceConsolidationCandidate:
    if (
        trajectory.trajectory_id
        != stability.trajectory_id
    ):
        raise ExperienceConsolidationError(
            "trajectory and stability analysis must reference the same trajectory"
        )

    if not feedback_trajectory_is_policy_free(
        trajectory
    ):
        raise ExperienceConsolidationError(
            "feedback trajectory must be policy-free"
        )

    if not feedback_stability_is_policy_free(
        stability
    ):
        raise ExperienceConsolidationError(
            "feedback stability result must be policy-free"
        )

    actual_config = (
        config
        if config is not None
        else ExperienceConsolidationConfig()
    )

    dominant_series = feedback_dominant_series(
        trajectory
    )

    dominant_attractor_id, consistency_score = (
        _dominant_consistency_score(
            dominant_series
        )
    )

    repetition_score = _repetition_score(
        trajectory
    )

    stability_score = _stability_score(
        stability
    )

    ambiguities = feedback_ambiguity_series(
        trajectory
    )

    mean_ambiguity = _mean(
        ambiguities
    )

    ambiguity_quality = _ambiguity_quality(
        mean_ambiguity
    )

    terminal_drift = (
        stability.terminal_recursive_drift
    )

    terminal_drift_quality = _terminal_drift_quality(
        terminal_drift,
        actual_config.maximum_terminal_drift,
    )

    evidence = ConsolidationEvidence(
        recursive_step_count=(
            trajectory.summary.recursive_step_count
        ),
        repetition_score=repetition_score,
        stability_score=stability_score,
        consistency_score=consistency_score,
        ambiguity_quality=ambiguity_quality,
        terminal_drift_quality=(
            terminal_drift_quality
        ),
        mean_ambiguity=mean_ambiguity,
        terminal_recursive_drift=(
            terminal_drift
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "evidence_is_descriptive": True,
            "memory_mutated": False,
            "learning_applied": False,
            "memory_committed": False,
            "causal_truth_inferred": False,
        },
    )

    weights = (
        actual_config.convergence_weight,
        actual_config.persistence_weight,
        actual_config.repetition_weight,
        actual_config.consistency_weight,
        actual_config.ambiguity_weight,
    )

    weighted_values = (
        stability.convergence_score,
        stability.persistence_score,
        repetition_score,
        consistency_score,
        ambiguity_quality,
    )

    total_weight = sum(
        weights
    )

    consolidation_score = (
        sum(
            weight * value
            for weight, value in zip(
                weights,
                weighted_values,
            )
        )
        / total_weight
    )

    enough_steps = (
        evidence.recursive_step_count
        >= actual_config.minimum_recursive_steps
    )

    stable_enough = (
        evidence.stability_score
        >= actual_config.minimum_stability_score
    )

    repeated_enough = (
        evidence.repetition_score
        >= actual_config.minimum_repetition_score
    )

    consistent_enough = (
        evidence.consistency_score
        >= actual_config.minimum_consistency_score
    )

    ambiguity_ok = (
        evidence.mean_ambiguity
        <= actual_config.maximum_mean_ambiguity
    )

    drift_ok = (
        evidence.terminal_recursive_drift
        <= actual_config.maximum_terminal_drift
    )

    total_ok = (
        consolidation_score
        >= actual_config.minimum_total_score
    )

    eligible = all(
        (
            enough_steps,
            stable_enough,
            repeated_enough,
            consistent_enough,
            ambiguity_ok,
            drift_ok,
            total_ok,
        )
    )

    if eligible:
        reason = "eligible_candidate_only"
    elif not enough_steps:
        reason = "insufficient_recursive_repetition"
    elif not stable_enough:
        reason = "insufficient_stability"
    elif not repeated_enough:
        reason = "insufficient_repetition"
    elif not consistent_enough:
        reason = "insufficient_dominant_consistency"
    elif not ambiguity_ok:
        reason = "ambiguity_too_high"
    elif not drift_ok:
        reason = "terminal_drift_too_high"
    else:
        reason = "consolidation_score_below_threshold"

    representation_vector = (
        _candidate_representation_vector(
            trajectory,
            dominant_attractor_id,
        )
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "experience_consolidation_mode": "candidate_evaluation_only",
        "source_trajectory_id": trajectory.trajectory_id,
        "source_stability_analysis_id": stability.analysis_id,
        "memory_mutated": False,
        "learning_applied": False,
        "memory_committed": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_consolidation_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return ExperienceConsolidationCandidate(
        candidate_id=candidate_id,
        source_trajectory_id=trajectory.trajectory_id,
        source_stability_analysis_id=stability.analysis_id,
        dominant_attractor_id=dominant_attractor_id,
        representation_vector=representation_vector,
        evidence=evidence,
        consolidation_score=consolidation_score,
        eligible_for_consolidation=eligible,
        reason=reason,
        metadata=merged_metadata,
    )


def consolidation_is_policy_free(
    candidate: ExperienceConsolidationCandidate,
) -> bool:
    return (
        candidate.metadata.get(
            "action_selected"
        )
        is False
        and candidate.metadata.get(
            "policy_modified"
        )
        is False
        and candidate.metadata.get(
            "memory_mutated"
        )
        is False
        and candidate.metadata.get(
            "learning_applied"
        )
        is False
        and candidate.metadata.get(
            "memory_committed"
        )
        is False
    )


__all__ = [
    "ConsolidationEvidence",
    "ExperienceConsolidationCandidate",
    "ExperienceConsolidationConfig",
    "ExperienceConsolidationError",
    "SCHEMA_VERSION",
    "consolidation_is_policy_free",
    "evaluate_experience_consolidation",
]
