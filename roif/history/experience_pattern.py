"""
ROIF Memory 2.0
Experience Pattern Core

Third layer above:
    ExperienceTransformation
    ExperienceSequence

Purpose
-------
Detect repeated transition structure across one or more experience sequences.

A pattern is NOT merely:
    "the same event happened several times"

A pattern is:
    "similar events repeatedly encounter the system in related states and
     produce a sufficiently stable transformation / response structure"

This layer measures:
- event-family repetition
- response stability
- body-response stability
- scar stability
- reserve-direction consistency
- sensitization / adaptation tendency
- pattern confidence
- variability

Architectural boundaries
------------------------
ExperiencePattern:
- does not infer causal truth;
- does not diagnose;
- does not select actions;
- does not modify policy;
- treats pattern detection as descriptive/evaluator logic only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_sequence import (
    ExperienceSequence,
    sequence_is_policy_free,
)
from roif.history.experience_transformation import (
    ExperienceTransformation,
)


SCHEMA_VERSION = "experience_pattern_v1"


class ExperiencePatternError(ValueError):
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
        raise ExperiencePatternError(
            f"{name} must not be empty"
        )
    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)
    if not isfinite(x):
        raise ExperiencePatternError(
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
        raise ExperiencePatternError(
            "vector dimensions must match"
        )
    return _l2(
        tuple(
            float(x) - float(y)
            for x, y in zip(a, b)
        )
    )


def _mean_vector(
    vectors: Sequence[
        Sequence[float]
    ],
) -> tuple[float, ...]:
    if not vectors:
        raise ExperiencePatternError(
            "vectors must not be empty"
        )

    width = len(
        vectors[0]
    )

    if width == 0:
        raise ExperiencePatternError(
            "vectors must not be empty"
        )

    if any(
        len(vector) != width
        for vector in vectors
    ):
        raise ExperiencePatternError(
            "vector dimensions must match"
        )

    return tuple(
        mean(
            float(vector[index])
            for vector in vectors
        )
        for index in range(width)
    )


def _flatten_body_response(
    transformation: ExperienceTransformation,
) -> tuple[float, ...]:
    body = transformation.body_response
    return (
        *body.autonomic,
        *body.endocrine,
        *body.immune,
        *body.motor,
        *body.interoceptive,
    )


def _flatten_system_response(
    transformation: ExperienceTransformation,
) -> tuple[float, ...]:
    response = transformation.response
    return (
        *response.cognitive_response,
        *response.physiological_response,
        *response.behavioral_response,
    )


def _flatten_scar(
    transformation: ExperienceTransformation,
) -> tuple[float, ...]:
    scar = transformation.scar
    return (
        *scar.cognitive_delta,
        *scar.physiological_delta,
        *scar.contextual_delta,
        scar.reserve_delta,
    )


def _mean_distance_from_centroid(
    vectors: Sequence[
        Sequence[float]
    ],
) -> float:
    if not vectors:
        return 0.0

    centroid = _mean_vector(
        vectors
    )

    return mean(
        _distance(
            vector,
            centroid,
        )
        for vector in vectors
    )


@dataclass(frozen=True, slots=True)
class PatternMember:
    sequence_id: str
    transformation_id: str
    event_type: str

    response_vector: tuple[float, ...]
    body_vector: tuple[float, ...]
    scar_vector: tuple[float, ...]

    reserve_before: float
    reserve_after: float
    reserve_delta: float

    sensitization_index: float
    adaptation_index: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_id",
            _validate_id(
                "sequence_id",
                self.sequence_id,
            ),
        )

        object.__setattr__(
            self,
            "transformation_id",
            _validate_id(
                "transformation_id",
                self.transformation_id,
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

        for name in (
            "response_vector",
            "body_vector",
            "scar_vector",
        ):
            vector = tuple(
                _validate_finite(
                    f"{name}[{index}]",
                    value,
                )
                for index, value
                in enumerate(
                    getattr(
                        self,
                        name,
                    )
                )
            )

            if not vector:
                raise ExperiencePatternError(
                    f"{name} must not be empty"
                )

            object.__setattr__(
                self,
                name,
                vector,
            )

        for name in (
            "reserve_before",
            "reserve_after",
            "reserve_delta",
            "sensitization_index",
            "adaptation_index",
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

        if self.sensitization_index < 0.0:
            raise ExperiencePatternError(
                "sensitization_index must be non-negative"
            )

        if self.adaptation_index < 0.0:
            raise ExperiencePatternError(
                "adaptation_index must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperiencePatternMetrics:
    member_count: int
    sequence_count: int

    response_variability: float
    body_variability: float
    scar_variability: float

    mean_reserve_delta: float
    mean_sensitization: float
    mean_adaptation: float

    reserve_decline_fraction: float
    reserve_gain_fraction: float

    response_stability: float
    body_stability: float
    scar_stability: float

    pattern_confidence: float

    sensitization_dominant: bool
    adaptation_dominant: bool
    mixed_direction: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.member_count < 1:
            raise ExperiencePatternError(
                "member_count must be >= 1"
            )

        if self.sequence_count < 1:
            raise ExperiencePatternError(
                "sequence_count must be >= 1"
            )

        for name in (
            "response_variability",
            "body_variability",
            "scar_variability",
            "mean_reserve_delta",
            "mean_sensitization",
            "mean_adaptation",
            "reserve_decline_fraction",
            "reserve_gain_fraction",
            "response_stability",
            "body_stability",
            "scar_stability",
            "pattern_confidence",
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
            "response_variability",
            "body_variability",
            "scar_variability",
            "mean_sensitization",
            "mean_adaptation",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise ExperiencePatternError(
                    f"{name} must be non-negative"
                )

        for name in (
            "reserve_decline_fraction",
            "reserve_gain_fraction",
            "response_stability",
            "body_stability",
            "scar_stability",
            "pattern_confidence",
        ):
            value = getattr(
                self,
                name,
            )
            if not 0.0 <= value <= 1.0:
                raise ExperiencePatternError(
                    f"{name} must be within [0, 1]"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperiencePattern:
    pattern_id: str
    event_type: str

    members: tuple[
        PatternMember,
        ...,
    ]

    metrics: ExperiencePatternMetrics

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

        if not self.members:
            raise ExperiencePatternError(
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


def extract_pattern_members(
    sequences: Iterable[
        ExperienceSequence
    ],
    *,
    event_type: str,
) -> tuple[
    PatternMember,
    ...,
]:
    target = _validate_id(
        "event_type",
        event_type,
    )

    members = []

    for sequence in sequences:
        for transformation in sequence.transformations:
            if (
                transformation.event.event_type
                != target
            ):
                continue

            members.append(
                PatternMember(
                    sequence_id=sequence.sequence_id,
                    transformation_id=(
                        transformation.transformation_id
                    ),
                    event_type=target,
                    response_vector=(
                        _flatten_system_response(
                            transformation
                        )
                    ),
                    body_vector=(
                        _flatten_body_response(
                            transformation
                        )
                    ),
                    scar_vector=(
                        _flatten_scar(
                            transformation
                        )
                    ),
                    reserve_before=(
                        transformation.state_before.reserve
                    ),
                    reserve_after=(
                        transformation.state_after.reserve
                    ),
                    reserve_delta=(
                        transformation.scar.reserve_delta
                    ),
                    sensitization_index=(
                        transformation.scar.sensitization_index
                    ),
                    adaptation_index=(
                        transformation.scar.adaptation_index
                    ),
                    metadata={
                        "schema_version": SCHEMA_VERSION,
                        "evaluator_only": True,
                        "causal_truth_inferred": False,
                    },
                )
            )

    return tuple(
        members
    )


def _stability_from_variability(
    variability: float,
) -> float:
    """
    Smooth bounded transformation:
        variability = 0 -> stability = 1
        variability grows -> stability approaches 0
    """

    return 1.0 / (
        1.0
        + max(
            0.0,
            variability,
        )
    )


def compute_pattern_metrics(
    members: Sequence[
        PatternMember
    ],
) -> ExperiencePatternMetrics:
    items = tuple(
        members
    )

    if not items:
        raise ExperiencePatternError(
            "members must not be empty"
        )

    response_vectors = tuple(
        member.response_vector
        for member in items
    )

    body_vectors = tuple(
        member.body_vector
        for member in items
    )

    scar_vectors = tuple(
        member.scar_vector
        for member in items
    )

    response_variability = (
        _mean_distance_from_centroid(
            response_vectors
        )
    )

    body_variability = (
        _mean_distance_from_centroid(
            body_vectors
        )
    )

    scar_variability = (
        _mean_distance_from_centroid(
            scar_vectors
        )
    )

    mean_reserve_delta = mean(
        member.reserve_delta
        for member in items
    )

    mean_sensitization = mean(
        member.sensitization_index
        for member in items
    )

    mean_adaptation = mean(
        member.adaptation_index
        for member in items
    )

    decline_count = sum(
        1
        for member in items
        if member.reserve_delta < 0.0
    )

    gain_count = sum(
        1
        for member in items
        if member.reserve_delta > 0.0
    )

    count = len(
        items
    )

    reserve_decline_fraction = (
        decline_count
        / count
    )

    reserve_gain_fraction = (
        gain_count
        / count
    )

    response_stability = (
        _stability_from_variability(
            response_variability
        )
    )

    body_stability = (
        _stability_from_variability(
            body_variability
        )
    )

    scar_stability = (
        _stability_from_variability(
            scar_variability
        )
    )

    recurrence_factor = min(
        1.0,
        count
        / 5.0,
    )

    pattern_confidence = (
        recurrence_factor
        * mean(
            (
                response_stability,
                body_stability,
                scar_stability,
            )
        )
    )

    sensitization_dominant = (
        mean_sensitization
        > mean_adaptation
        and reserve_decline_fraction
        > reserve_gain_fraction
    )

    adaptation_dominant = (
        mean_adaptation
        > mean_sensitization
        and reserve_gain_fraction
        > reserve_decline_fraction
    )

    mixed_direction = not (
        sensitization_dominant
        or adaptation_dominant
    )

    return ExperiencePatternMetrics(
        member_count=count,
        sequence_count=len(
            {
                member.sequence_id
                for member in items
            }
        ),
        response_variability=(
            response_variability
        ),
        body_variability=(
            body_variability
        ),
        scar_variability=(
            scar_variability
        ),
        mean_reserve_delta=(
            mean_reserve_delta
        ),
        mean_sensitization=(
            mean_sensitization
        ),
        mean_adaptation=(
            mean_adaptation
        ),
        reserve_decline_fraction=(
            reserve_decline_fraction
        ),
        reserve_gain_fraction=(
            reserve_gain_fraction
        ),
        response_stability=(
            response_stability
        ),
        body_stability=(
            body_stability
        ),
        scar_stability=(
            scar_stability
        ),
        pattern_confidence=(
            pattern_confidence
        ),
        sensitization_dominant=(
            sensitization_dominant
        ),
        adaptation_dominant=(
            adaptation_dominant
        ),
        mixed_direction=(
            mixed_direction
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "pattern_metric_is_descriptive": True,
            "causal_truth_inferred": False,
        },
    )


def build_experience_pattern(
    *,
    pattern_id: str,
    event_type: str,
    sequences: Iterable[
        ExperienceSequence
    ],
    metadata: Mapping[str, Any] | None = None,
) -> ExperiencePattern:
    sequence_items = tuple(
        sequences
    )

    if not sequence_items:
        raise ExperiencePatternError(
            "sequences must not be empty"
        )

    if not all(
        sequence_is_policy_free(
            sequence
        )
        for sequence in sequence_items
    ):
        raise ExperiencePatternError(
            "all source sequences must be policy-free"
        )

    members = extract_pattern_members(
        sequence_items,
        event_type=event_type,
    )

    if not members:
        raise ExperiencePatternError(
            "no matching event_type found in sequences"
        )

    metrics = compute_pattern_metrics(
        members
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_sequence_count": len(
            sequence_items
        ),
        "pattern_detection_mode": "descriptive",
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return ExperiencePattern(
        pattern_id=pattern_id,
        event_type=event_type,
        members=members,
        metrics=metrics,
        metadata=merged_metadata,
    )


def pattern_is_policy_free(
    pattern: ExperiencePattern,
) -> bool:
    return (
        pattern.metadata.get(
            "action_selected"
        )
        is False
        and pattern.metadata.get(
            "policy_modified"
        )
        is False
    )


def pattern_is_recurrent(
    pattern: ExperiencePattern,
    *,
    minimum_members: int = 2,
) -> bool:
    if minimum_members < 1:
        raise ExperiencePatternError(
            "minimum_members must be >= 1"
        )

    return (
        pattern.metrics.member_count
        >= minimum_members
    )


def pattern_has_stable_body_response(
    pattern: ExperiencePattern,
    *,
    minimum_stability: float = 0.75,
) -> bool:
    if not 0.0 <= minimum_stability <= 1.0:
        raise ExperiencePatternError(
            "minimum_stability must be within [0, 1]"
        )

    return (
        pattern.metrics.body_stability
        >= minimum_stability
    )


def pattern_has_stable_scar(
    pattern: ExperiencePattern,
    *,
    minimum_stability: float = 0.75,
) -> bool:
    if not 0.0 <= minimum_stability <= 1.0:
        raise ExperiencePatternError(
            "minimum_stability must be within [0, 1]"
        )

    return (
        pattern.metrics.scar_stability
        >= minimum_stability
    )


def pattern_direction(
    pattern: ExperiencePattern,
) -> str:
    if pattern.metrics.sensitization_dominant:
        return "sensitization"

    if pattern.metrics.adaptation_dominant:
        return "adaptation"

    return "mixed"


def member_reserve_delta_series(
    pattern: ExperiencePattern,
) -> tuple[float, ...]:
    return tuple(
        member.reserve_delta
        for member in pattern.members
    )


def member_sensitization_series(
    pattern: ExperiencePattern,
) -> tuple[float, ...]:
    return tuple(
        member.sensitization_index
        for member in pattern.members
    )


def member_adaptation_series(
    pattern: ExperiencePattern,
) -> tuple[float, ...]:
    return tuple(
        member.adaptation_index
        for member in pattern.members
    )


__all__ = [
    "ExperiencePattern",
    "ExperiencePatternError",
    "ExperiencePatternMetrics",
    "PatternMember",
    "SCHEMA_VERSION",
    "build_experience_pattern",
    "compute_pattern_metrics",
    "extract_pattern_members",
    "member_adaptation_series",
    "member_reserve_delta_series",
    "member_sensitization_series",
    "pattern_direction",
    "pattern_has_stable_body_response",
    "pattern_has_stable_scar",
    "pattern_is_policy_free",
    "pattern_is_recurrent",
]
