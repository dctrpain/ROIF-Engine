"""
ROIF Memory 2.0
Feedback Stability Core

Eighteenth layer above:
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

Purpose
-------
Analyze recursive feedback trajectories without mutating them.

The layer classifies descriptive recursive regimes such as:
- convergent
- persistent
- oscillatory
- switching
- divergent
- mixed

It computes:
- switching rate
- return rate
- dominant-attractor persistence
- recursive-drift trend
- feedback-displacement trend
- ambiguity trend
- terminal drift
- convergence score
- divergence score
- oscillation score
- persistence score
- regime label

This module does NOT:
- learn
- mutate source trajectories
- mutate memory
- mutate attractors
- select actions
- modify policy
- diagnose
- claim biological stability mechanisms
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
    recursive_drift_series,
    return_flags,
    switch_flags,
)


SCHEMA_VERSION = "feedback_stability_v1"


class FeedbackStabilityError(ValueError):
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
        raise FeedbackStabilityError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise FeedbackStabilityError(
            f"{name} must be finite"
        )

    return x


def _bounded01(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def _mean(
    values: Sequence[float],
) -> float:
    if not values:
        return 0.0

    return sum(
        float(value)
        for value in values
    ) / len(values)


def _linear_slope(
    values: Sequence[float],
) -> float:
    """
    Deterministic least-squares slope against step index.

    For one value, the slope is zero.
    """
    n = len(values)

    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = _mean(values)

    numerator = 0.0
    denominator = 0.0

    for index, value in enumerate(values):
        dx = index - x_mean
        numerator += dx * (float(value) - y_mean)
        denominator += dx * dx

    if denominator <= 0.0:
        return 0.0

    return numerator / denominator


def _transition_count(
    values: Sequence[str],
) -> int:
    if len(values) < 2:
        return 0

    return sum(
        1
        for index in range(1, len(values))
        if values[index] != values[index - 1]
    )


def _alternation_score(
    values: Sequence[str],
) -> float:
    """
    Score strict A-B-A-B style alternation.

    Returns 0 for fewer than three values.
    """
    if len(values) < 3:
        return 0.0

    eligible = 0
    matches = 0

    for index in range(2, len(values)):
        eligible += 1

        if (
            values[index] == values[index - 2]
            and values[index] != values[index - 1]
        ):
            matches += 1

    if eligible == 0:
        return 0.0

    return matches / eligible


def _longest_run_fraction(
    values: Sequence[str],
) -> float:
    if not values:
        return 0.0

    longest = 1
    current = 1

    for index in range(1, len(values)):
        if values[index] == values[index - 1]:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 1

    return longest / len(values)


@dataclass(frozen=True, slots=True)
class FeedbackStabilityConfig:
    drift_epsilon: float = 1e-9
    convergence_slope_threshold: float = -1e-6
    divergence_slope_threshold: float = 1e-6

    persistent_run_threshold: float = 0.75
    oscillation_threshold: float = 0.75
    high_switch_rate_threshold: float = 0.50

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "drift_epsilon",
            "persistent_run_threshold",
            "oscillation_threshold",
            "high_switch_rate_threshold",
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
            "convergence_slope_threshold",
            "divergence_slope_threshold",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(self, name),
                ),
            )

        if self.drift_epsilon < 0.0:
            raise FeedbackStabilityError(
                "drift_epsilon must be non-negative"
            )

        for name in (
            "persistent_run_threshold",
            "oscillation_threshold",
            "high_switch_rate_threshold",
        ):
            value = getattr(
                self,
                name,
            )

            if not 0.0 <= value <= 1.0:
                raise FeedbackStabilityError(
                    f"{name} must be within [0,1]"
                )

        if (
            self.convergence_slope_threshold
            >= self.divergence_slope_threshold
        ):
            raise FeedbackStabilityError(
                "convergence_slope_threshold must be less than divergence_slope_threshold"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class FeedbackStabilityResult:
    analysis_id: str
    trajectory_id: str

    step_count: int
    recursive_step_count: int

    switch_count: int
    return_count: int
    switch_rate: float
    return_rate: float

    unique_dominant_count: int
    longest_dominant_run_fraction: float
    alternation_score: float

    mean_recursive_drift: float
    terminal_recursive_drift: float
    recursive_drift_slope: float

    mean_feedback_displacement: float
    feedback_displacement_slope: float

    mean_feedback_ambiguity: float
    feedback_ambiguity_slope: float

    convergence_score: float
    divergence_score: float
    oscillation_score: float
    persistence_score: float

    regime: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_id",
            _validate_id(
                "analysis_id",
                self.analysis_id,
            ),
        )

        object.__setattr__(
            self,
            "trajectory_id",
            _validate_id(
                "trajectory_id",
                self.trajectory_id,
            ),
        )

        for name in (
            "step_count",
            "recursive_step_count",
            "switch_count",
            "return_count",
            "unique_dominant_count",
        ):
            if int(
                getattr(
                    self,
                    name,
                )
            ) < 0:
                raise FeedbackStabilityError(
                    f"{name} must be non-negative"
                )

        if self.step_count < 1:
            raise FeedbackStabilityError(
                "step_count must be >= 1"
            )

        for name in (
            "switch_rate",
            "return_rate",
            "longest_dominant_run_fraction",
            "alternation_score",
            "convergence_score",
            "divergence_score",
            "oscillation_score",
            "persistence_score",
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

            value = getattr(
                self,
                name,
            )

            if not 0.0 <= value <= 1.0:
                raise FeedbackStabilityError(
                    f"{name} must be within [0,1]"
                )

        for name in (
            "mean_recursive_drift",
            "terminal_recursive_drift",
            "recursive_drift_slope",
            "mean_feedback_displacement",
            "feedback_displacement_slope",
            "mean_feedback_ambiguity",
            "feedback_ambiguity_slope",
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
            "mean_recursive_drift",
            "terminal_recursive_drift",
            "mean_feedback_displacement",
            "mean_feedback_ambiguity",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise FeedbackStabilityError(
                    f"{name} must be non-negative"
                )

        object.__setattr__(
            self,
            "regime",
            _validate_id(
                "regime",
                self.regime,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def classify_feedback_regime(
    *,
    recursive_drift_slope: float,
    terminal_recursive_drift: float,
    mean_recursive_drift: float = 0.0,
    switch_rate: float,
    alternation_score: float,
    persistence_score: float,
    config: FeedbackStabilityConfig,
) -> str:
    if (
        alternation_score
        >= config.oscillation_threshold
        and switch_rate
        >= config.high_switch_rate_threshold
    ):
        return "oscillatory"

    if (
        recursive_drift_slope
        <= config.convergence_slope_threshold
        and switch_rate
        < config.high_switch_rate_threshold
        and terminal_recursive_drift
        <= (
            mean_recursive_drift
            + config.drift_epsilon
        )
    ):
        return "convergent"

    if (
        recursive_drift_slope
        >= config.divergence_slope_threshold
    ):
        return "divergent"

    if (
        persistence_score
        >= config.persistent_run_threshold
        and switch_rate
        < config.high_switch_rate_threshold
    ):
        return "persistent"

    if (
        switch_rate
        >= config.high_switch_rate_threshold
    ):
        return "switching"

    return "mixed"


def analyze_feedback_stability(
    *,
    analysis_id: str,
    trajectory: FeedbackTrajectory,
    config: FeedbackStabilityConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeedbackStabilityResult:
    if not feedback_trajectory_is_policy_free(
        trajectory
    ):
        raise FeedbackStabilityError(
            "feedback trajectory must be policy-free"
        )

    actual_config = (
        config
        if config is not None
        else FeedbackStabilityConfig()
    )

    dominants = feedback_dominant_series(
        trajectory
    )

    switches = switch_flags(
        trajectory
    )

    returns = return_flags(
        trajectory
    )

    drifts_all = recursive_drift_series(
        trajectory
    )

    displacements_all = feedback_semantic_displacement_series(
        trajectory
    )

    ambiguities = feedback_ambiguity_series(
        trajectory
    )

    # Exclude seed step from recursive stability slopes.
    recursive_drifts = tuple(
        drifts_all[1:]
    )

    recursive_displacements = tuple(
        displacements_all[1:]
    )

    switch_count = sum(
        1
        for flag in switches
        if flag
    )

    return_count = sum(
        1
        for flag in returns
        if flag
    )

    transition_slots = max(
        1,
        len(dominants) - 1,
    )

    switch_rate = (
        switch_count
        / transition_slots
    )

    return_rate = (
        return_count
        / transition_slots
    )

    longest_run_fraction = _longest_run_fraction(
        dominants
    )

    alternation = _alternation_score(
        dominants
    )

    drift_slope = _linear_slope(
        recursive_drifts
    )

    displacement_slope = _linear_slope(
        recursive_displacements
    )

    ambiguity_slope = _linear_slope(
        ambiguities
    )

    mean_drift = _mean(
        recursive_drifts
    )

    terminal_drift = (
        recursive_drifts[-1]
        if recursive_drifts
        else 0.0
    )

    mean_displacement = _mean(
        recursive_displacements
    )

    mean_ambiguity = _mean(
        ambiguities
    )

    persistence_score = _bounded01(
        longest_run_fraction
        * (
            1.0
            - switch_rate
        )
    )

    oscillation_score = _bounded01(
        (
            alternation
            + switch_rate
            + return_rate
        )
        / 3.0
    )

    convergence_trend = (
        1.0
        if drift_slope
        <= actual_config.convergence_slope_threshold
        else 0.0
    )

    terminal_small = (
        1.0
        if terminal_drift
        <= actual_config.drift_epsilon
        else (
            1.0
            / (
                1.0
                + terminal_drift
            )
        )
    )

    convergence_score = _bounded01(
        (
            convergence_trend
            + terminal_small
            + (
                1.0
                - switch_rate
            )
        )
        / 3.0
    )

    divergence_trend = (
        1.0
        if drift_slope
        >= actual_config.divergence_slope_threshold
        else 0.0
    )

    divergence_score = _bounded01(
        (
            divergence_trend
            + (
                1.0
                - (
                    1.0
                    / (
                        1.0
                        + terminal_drift
                    )
                )
            )
            + switch_rate
        )
        / 3.0
    )

    regime = classify_feedback_regime(
        recursive_drift_slope=drift_slope,
        terminal_recursive_drift=terminal_drift,
        mean_recursive_drift=mean_drift,
        switch_rate=switch_rate,
        alternation_score=alternation,
        persistence_score=persistence_score,
        config=actual_config,
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "feedback_stability_mode": "descriptive",
        "source_trajectory_id": trajectory.trajectory_id,
        "experience_semantics": "state_transition_then_settling_allowed",
        "memory_mutated": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_stability_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return FeedbackStabilityResult(
        analysis_id=analysis_id,
        trajectory_id=trajectory.trajectory_id,
        step_count=len(
            trajectory.steps
        ),
        recursive_step_count=trajectory.summary.recursive_step_count,
        switch_count=switch_count,
        return_count=return_count,
        switch_rate=switch_rate,
        return_rate=return_rate,
        unique_dominant_count=len(
            set(
                dominants
            )
        ),
        longest_dominant_run_fraction=(
            longest_run_fraction
        ),
        alternation_score=alternation,
        mean_recursive_drift=mean_drift,
        terminal_recursive_drift=terminal_drift,
        recursive_drift_slope=drift_slope,
        mean_feedback_displacement=mean_displacement,
        feedback_displacement_slope=displacement_slope,
        mean_feedback_ambiguity=mean_ambiguity,
        feedback_ambiguity_slope=ambiguity_slope,
        convergence_score=convergence_score,
        divergence_score=divergence_score,
        oscillation_score=oscillation_score,
        persistence_score=persistence_score,
        regime=regime,
        metadata=merged_metadata,
    )


def feedback_stability_is_policy_free(
    result: FeedbackStabilityResult,
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
    )


__all__ = [
    "FeedbackStabilityConfig",
    "FeedbackStabilityError",
    "FeedbackStabilityResult",
    "SCHEMA_VERSION",
    "analyze_feedback_stability",
    "classify_feedback_regime",
    "feedback_stability_is_policy_free",
]

