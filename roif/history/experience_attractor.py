"""
ROIF Memory 2.0
Experience Attractor Core

Fourth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern

Purpose
-------
Represent a recurrent, sufficiently stable region of experience-pattern space
that the system tends to revisit.

This is intentionally a descriptive computational object.
It does NOT claim a biological attractor has been proven.
It does NOT diagnose.
It does NOT select actions.
It does NOT modify policy.

Key concepts
------------
- attractor center:
    centroid of pattern-level feature vectors

- basin:
    set of source patterns assigned to the attractor

- compactness:
    how tightly patterns cluster around the center

- recurrence:
    how often the same event-family / direction reappears

- return tendency:
    how often later patterns remain close to or return toward the center

- direction:
    sensitization / adaptation / mixed, derived from source patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_pattern import (
    ExperiencePattern,
    pattern_direction,
    pattern_is_policy_free,
)


SCHEMA_VERSION = "experience_attractor_v1"


class ExperienceAttractorError(ValueError):
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
        raise ExperienceAttractorError(
            f"{name} must not be empty"
        )
    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)
    if not isfinite(x):
        raise ExperienceAttractorError(
            f"{name} must be finite"
        )
    return x


def _l2(
    vector: Sequence[float],
) -> float:
    return sqrt(
        sum(
            float(x) ** 2
            for x in vector
        )
    )


def _distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise ExperienceAttractorError(
            "vector dimensions must match"
        )

    return _l2(
        tuple(
            float(x) - float(y)
            for x, y in zip(
                a,
                b,
            )
        )
    )


def _mean_vector(
    vectors: Sequence[
        Sequence[float]
    ],
) -> tuple[float, ...]:
    if not vectors:
        raise ExperienceAttractorError(
            "vectors must not be empty"
        )

    width = len(
        vectors[0]
    )

    if width == 0:
        raise ExperienceAttractorError(
            "vectors must not be empty"
        )

    if any(
        len(vector) != width
        for vector in vectors
    ):
        raise ExperienceAttractorError(
            "vector dimensions must match"
        )

    return tuple(
        mean(
            float(vector[index])
            for vector in vectors
        )
        for index in range(width)
    )


def pattern_feature_vector(
    pattern: ExperiencePattern,
) -> tuple[float, ...]:
    """
    Compact pattern-level representation.

    The vector is deliberately descriptive and made only from already-derived
    pattern metrics.
    """

    metrics = pattern.metrics

    return (
        metrics.response_stability,
        metrics.body_stability,
        metrics.scar_stability,
        metrics.pattern_confidence,
        metrics.mean_reserve_delta,
        metrics.mean_sensitization,
        metrics.mean_adaptation,
        metrics.reserve_decline_fraction,
        metrics.reserve_gain_fraction,
    )


@dataclass(frozen=True, slots=True)
class AttractorMember:
    pattern_id: str
    event_type: str
    direction: str
    feature_vector: tuple[float, ...]
    distance_to_center: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pattern_id",
            _validate_id(
                "pattern_id",
                self.pattern_id,
            ),
        )

        object.__setattr__(
            self,
            "event_type",
            _validate_id(
                "event_type",
                self.event_type,
            ),
        )

        direction = _validate_id(
            "direction",
            self.direction,
        )

        if direction not in {
            "sensitization",
            "adaptation",
            "mixed",
        }:
            raise ExperienceAttractorError(
                "invalid direction"
            )

        object.__setattr__(
            self,
            "direction",
            direction,
        )

        vector = tuple(
            _validate_finite(
                f"feature_vector[{index}]",
                value,
            )
            for index, value
            in enumerate(
                self.feature_vector
            )
        )

        if not vector:
            raise ExperienceAttractorError(
                "feature_vector must not be empty"
            )

        object.__setattr__(
            self,
            "feature_vector",
            vector,
        )

        distance = _validate_finite(
            "distance_to_center",
            self.distance_to_center,
        )

        if distance < 0.0:
            raise ExperienceAttractorError(
                "distance_to_center must be non-negative"
            )

        object.__setattr__(
            self,
            "distance_to_center",
            distance,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperienceAttractorMetrics:
    member_count: int
    event_type_count: int

    mean_distance_to_center: float
    max_distance_to_center: float
    compactness: float

    recurrence_score: float
    return_tendency: float
    attractor_confidence: float

    sensitization_fraction: float
    adaptation_fraction: float
    mixed_fraction: float

    dominant_direction: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.member_count < 1:
            raise ExperienceAttractorError(
                "member_count must be >= 1"
            )

        if self.event_type_count < 1:
            raise ExperienceAttractorError(
                "event_type_count must be >= 1"
            )

        for name in (
            "mean_distance_to_center",
            "max_distance_to_center",
            "compactness",
            "recurrence_score",
            "return_tendency",
            "attractor_confidence",
            "sensitization_fraction",
            "adaptation_fraction",
            "mixed_fraction",
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
            "mean_distance_to_center",
            "max_distance_to_center",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise ExperienceAttractorError(
                    f"{name} must be non-negative"
                )

        for name in (
            "compactness",
            "recurrence_score",
            "return_tendency",
            "attractor_confidence",
            "sensitization_fraction",
            "adaptation_fraction",
            "mixed_fraction",
        ):
            value = getattr(
                self,
                name,
            )
            if not 0.0 <= value <= 1.0:
                raise ExperienceAttractorError(
                    f"{name} must be within [0, 1]"
                )

        direction = _validate_id(
            "dominant_direction",
            self.dominant_direction,
        )

        if direction not in {
            "sensitization",
            "adaptation",
            "mixed",
        }:
            raise ExperienceAttractorError(
                "invalid dominant_direction"
            )

        object.__setattr__(
            self,
            "dominant_direction",
            direction,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperienceAttractor:
    attractor_id: str
    center: tuple[float, ...]
    members: tuple[
        AttractorMember,
        ...,
    ]
    metrics: ExperienceAttractorMetrics

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

        center = tuple(
            _validate_finite(
                f"center[{index}]",
                value,
            )
            for index, value
            in enumerate(
                self.center
            )
        )

        if not center:
            raise ExperienceAttractorError(
                "center must not be empty"
            )

        object.__setattr__(
            self,
            "center",
            center,
        )

        if not self.members:
            raise ExperienceAttractorError(
                "members must not be empty"
            )

        object.__setattr__(
            self,
            "members",
            tuple(
                self.members
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _dominant_direction(
    directions: Sequence[str],
) -> str:
    counts = {
        "sensitization": 0,
        "adaptation": 0,
        "mixed": 0,
    }

    for direction in directions:
        counts[
            direction
        ] += 1

    maximum = max(
        counts.values()
    )

    winners = tuple(
        direction
        for direction, count
        in counts.items()
        if count == maximum
    )

    if len(
        winners
    ) == 1:
        return winners[
            0
        ]

    return "mixed"


def _recurrence_score(
    patterns: Sequence[
        ExperiencePattern
    ],
) -> float:
    """
    Repetition score based on recurrence of event families.

    One observation -> 0.2
    Five or more observations -> 1.0
    """

    return min(
        1.0,
        len(
            patterns
        )
        / 5.0,
    )


def _return_tendency(
    vectors: Sequence[
        Sequence[float]
    ],
    center: Sequence[float],
) -> float:
    """
    Operational return tendency.

    For each consecutive pair:
    if current distance to center <= previous distance, count as return/stay.

    Single-member attractor gets 1.0 by convention because no contrary
    transition is observed.
    """

    if len(
        vectors
    ) <= 1:
        return 1.0

    distances = tuple(
        _distance(
            vector,
            center,
        )
        for vector in vectors
    )

    return mean(
        1.0
        if current <= previous
        else 0.0
        for previous, current
        in zip(
            distances,
            distances[
                1:
            ],
        )
    )


def build_experience_attractor(
    *,
    attractor_id: str,
    patterns: Iterable[
        ExperiencePattern
    ],
    metadata: Mapping[str, Any] | None = None,
) -> ExperienceAttractor:
    items = tuple(
        patterns
    )

    if not items:
        raise ExperienceAttractorError(
            "patterns must not be empty"
        )

    if not all(
        pattern_is_policy_free(
            pattern
        )
        for pattern in items
    ):
        raise ExperienceAttractorError(
            "all source patterns must be policy-free"
        )

    vectors = tuple(
        pattern_feature_vector(
            pattern
        )
        for pattern in items
    )

    center = _mean_vector(
        vectors
    )

    directions = tuple(
        pattern_direction(
            pattern
        )
        for pattern in items
    )

    distances = tuple(
        _distance(
            vector,
            center,
        )
        for vector in vectors
    )

    members = tuple(
        AttractorMember(
            pattern_id=pattern.pattern_id,
            event_type=pattern.event_type,
            direction=direction,
            feature_vector=vector,
            distance_to_center=distance,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "evaluator_only": True,
                "causal_truth_inferred": False,
            },
        )
        for pattern, direction, vector, distance
        in zip(
            items,
            directions,
            vectors,
            distances,
        )
    )

    mean_distance = mean(
        distances
    )

    max_distance = max(
        distances
    )

    compactness = 1.0 / (
        1.0
        + mean_distance
    )

    recurrence_score = _recurrence_score(
        items
    )

    return_tendency = _return_tendency(
        vectors,
        center,
    )

    attractor_confidence = mean(
        (
            compactness,
            recurrence_score,
            return_tendency,
        )
    )

    count = len(
        items
    )

    sensitization_fraction = (
        directions.count(
            "sensitization"
        )
        / count
    )

    adaptation_fraction = (
        directions.count(
            "adaptation"
        )
        / count
    )

    mixed_fraction = (
        directions.count(
            "mixed"
        )
        / count
    )

    metrics = ExperienceAttractorMetrics(
        member_count=count,
        event_type_count=len(
            {
                pattern.event_type
                for pattern in items
            }
        ),
        mean_distance_to_center=(
            mean_distance
        ),
        max_distance_to_center=(
            max_distance
        ),
        compactness=(
            compactness
        ),
        recurrence_score=(
            recurrence_score
        ),
        return_tendency=(
            return_tendency
        ),
        attractor_confidence=(
            attractor_confidence
        ),
        sensitization_fraction=(
            sensitization_fraction
        ),
        adaptation_fraction=(
            adaptation_fraction
        ),
        mixed_fraction=(
            mixed_fraction
        ),
        dominant_direction=(
            _dominant_direction(
                directions
            )
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "attractor_metric_is_descriptive": True,
            "biological_attractor_claimed": False,
            "causal_truth_inferred": False,
        },
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "attractor_detection_mode": "descriptive",
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

    return ExperienceAttractor(
        attractor_id=attractor_id,
        center=center,
        members=members,
        metrics=metrics,
        metadata=merged_metadata,
    )


def attractor_is_policy_free(
    attractor: ExperienceAttractor,
) -> bool:
    return (
        attractor.metadata.get(
            "action_selected"
        )
        is False
        and attractor.metadata.get(
            "policy_modified"
        )
        is False
    )


def attractor_is_recurrent(
    attractor: ExperienceAttractor,
    *,
    minimum_members: int = 2,
) -> bool:
    if minimum_members < 1:
        raise ExperienceAttractorError(
            "minimum_members must be >= 1"
        )

    return (
        attractor.metrics.member_count
        >= minimum_members
    )


def attractor_is_compact(
    attractor: ExperienceAttractor,
    *,
    minimum_compactness: float = 0.75,
) -> bool:
    if not 0.0 <= minimum_compactness <= 1.0:
        raise ExperienceAttractorError(
            "minimum_compactness must be within [0, 1]"
        )

    return (
        attractor.metrics.compactness
        >= minimum_compactness
    )


def attractor_has_return_tendency(
    attractor: ExperienceAttractor,
    *,
    minimum_return_tendency: float = 0.5,
) -> bool:
    if not 0.0 <= minimum_return_tendency <= 1.0:
        raise ExperienceAttractorError(
            "minimum_return_tendency must be within [0, 1]"
        )

    return (
        attractor.metrics.return_tendency
        >= minimum_return_tendency
    )


def attractor_member_ids(
    attractor: ExperienceAttractor,
) -> tuple[str, ...]:
    return tuple(
        member.pattern_id
        for member in attractor.members
    )


def attractor_event_types(
    attractor: ExperienceAttractor,
) -> tuple[str, ...]:
    return tuple(
        member.event_type
        for member in attractor.members
    )


def attractor_distance_series(
    attractor: ExperienceAttractor,
) -> tuple[float, ...]:
    return tuple(
        member.distance_to_center
        for member in attractor.members
    )


def attractor_direction(
    attractor: ExperienceAttractor,
) -> str:
    return attractor.metrics.dominant_direction


__all__ = [
    "AttractorMember",
    "ExperienceAttractor",
    "ExperienceAttractorError",
    "ExperienceAttractorMetrics",
    "SCHEMA_VERSION",
    "attractor_direction",
    "attractor_distance_series",
    "attractor_event_types",
    "attractor_has_return_tendency",
    "attractor_is_compact",
    "attractor_is_policy_free",
    "attractor_is_recurrent",
    "attractor_member_ids",
    "build_experience_attractor",
    "pattern_feature_vector",
]
