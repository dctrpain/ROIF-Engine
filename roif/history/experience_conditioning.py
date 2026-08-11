"""
ROIF Memory 2.0
Experience Conditioning Core

Seventh layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor
    AttractorDynamics
    AttractorTrajectory

Purpose
-------
Represent descriptive associative conditioning over repeated trajectories.

The conditioning layer models how a cue may become associated with a recurrent
trajectory outcome:

    cue
      + repeated co-occurrence with attractor trajectory
      -> association strength
      -> conditioned bias toward an attractor outcome

This module supports descriptive:
- acquisition
- extinction
- generalization
- conditioned response proxy
- association confidence

This module does NOT:
- diagnose
- claim biological conditioning has been proven
- mutate source memory objects
- select actions
- modify policy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.attractor_trajectory import (
    AttractorTrajectory,
    dominant_attractor_series,
    trajectory_is_policy_free,
)


SCHEMA_VERSION = "experience_conditioning_v1"


class ExperienceConditioningError(ValueError):
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
        raise ExperienceConditioningError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise ExperienceConditioningError(
            f"{name} must be finite"
        )

    return x


def _validate_probability(
    name: str,
    value: float,
) -> float:
    x = _validate_finite(
        name,
        value,
    )

    if not 0.0 <= x <= 1.0:
        raise ExperienceConditioningError(
            f"{name} must be within [0,1]"
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
        for index, value
        in enumerate(values)
    )

    if not vector:
        raise ExperienceConditioningError(
            f"{name} must not be empty"
        )

    return vector


def _distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise ExperienceConditioningError(
            "vector dimensions must match"
        )

    return sqrt(
        sum(
            (float(x) - float(y)) ** 2
            for x, y in zip(
                a,
                b,
            )
        )
    )


@dataclass(frozen=True, slots=True)
class ConditioningCue:
    cue_id: str
    cue_type: str
    feature_vector: tuple[float, ...]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cue_id",
            _validate_id(
                "cue_id",
                self.cue_id,
            ),
        )

        object.__setattr__(
            self,
            "cue_type",
            _validate_id(
                "cue_type",
                self.cue_type,
            ),
        )

        object.__setattr__(
            self,
            "feature_vector",
            _validate_vector(
                "feature_vector",
                self.feature_vector,
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
class ConditioningObservation:
    observation_id: str
    cue: ConditioningCue
    trajectory_id: str
    terminal_attractor_id: str | None
    dominant_series: tuple[
        str | None,
        ...,
    ]

    reinforcement_present: bool
    reinforcement_strength: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _validate_id(
                "observation_id",
                self.observation_id,
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

        object.__setattr__(
            self,
            "dominant_series",
            tuple(
                self.dominant_series
            ),
        )

        object.__setattr__(
            self,
            "reinforcement_strength",
            _validate_probability(
                "reinforcement_strength",
                self.reinforcement_strength,
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
class ConditioningAssociation:
    cue_id: str
    attractor_id: str

    exposure_count: int
    reinforced_exposure_count: int
    extinction_exposure_count: int

    acquisition_strength: float
    extinction_strength: float
    effective_association_strength: float

    association_confidence: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cue_id",
            _validate_id(
                "cue_id",
                self.cue_id,
            ),
        )

        object.__setattr__(
            self,
            "attractor_id",
            _validate_id(
                "attractor_id",
                self.attractor_id,
            ),
        )

        for name in (
            "exposure_count",
            "reinforced_exposure_count",
            "extinction_exposure_count",
        ):
            value = int(
                getattr(
                    self,
                    name,
                )
            )

            if value < 0:
                raise ExperienceConditioningError(
                    f"{name} must be non-negative"
                )

        for name in (
            "acquisition_strength",
            "extinction_strength",
            "effective_association_strength",
            "association_confidence",
        ):
            object.__setattr__(
                self,
                name,
                _validate_probability(
                    name,
                    getattr(
                        self,
                        name,
                    ),
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
class ConditioningMemory:
    memory_id: str
    cue: ConditioningCue
    observations: tuple[
        ConditioningObservation,
        ...,
    ]
    associations: tuple[
        ConditioningAssociation,
        ...,
    ]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_id",
            _validate_id(
                "memory_id",
                self.memory_id,
            ),
        )

        if not self.observations:
            raise ExperienceConditioningError(
                "observations must not be empty"
            )

        if not self.associations:
            raise ExperienceConditioningError(
                "associations must not be empty"
            )

        object.__setattr__(
            self,
            "observations",
            tuple(
                self.observations
            ),
        )

        object.__setattr__(
            self,
            "associations",
            tuple(
                self.associations
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
class ConditionedResponse:
    query_cue_id: str
    matched_memory_cue_id: str

    cue_distance: float
    generalization_weight: float

    predicted_attractor_id: str
    predicted_association_strength: float
    conditioned_response_strength: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_cue_id",
            _validate_id(
                "query_cue_id",
                self.query_cue_id,
            ),
        )

        object.__setattr__(
            self,
            "matched_memory_cue_id",
            _validate_id(
                "matched_memory_cue_id",
                self.matched_memory_cue_id,
            ),
        )

        object.__setattr__(
            self,
            "predicted_attractor_id",
            _validate_id(
                "predicted_attractor_id",
                self.predicted_attractor_id,
            ),
        )

        object.__setattr__(
            self,
            "cue_distance",
            _validate_finite(
                "cue_distance",
                self.cue_distance,
            ),
        )

        if self.cue_distance < 0.0:
            raise ExperienceConditioningError(
                "cue_distance must be non-negative"
            )

        for name in (
            "generalization_weight",
            "predicted_association_strength",
            "conditioned_response_strength",
        ):
            object.__setattr__(
                self,
                name,
                _validate_probability(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def observation_from_trajectory(
    *,
    observation_id: str,
    cue: ConditioningCue,
    trajectory: AttractorTrajectory,
    reinforcement_present: bool,
    reinforcement_strength: float = 1.0,
) -> ConditioningObservation:
    if not trajectory_is_policy_free(
        trajectory
    ):
        raise ExperienceConditioningError(
            "trajectory must be policy-free"
        )

    dominant_series = dominant_attractor_series(
        trajectory
    )

    terminal_attractor_id = (
        dominant_series[
            -1
        ]
        if dominant_series
        else None
    )

    strength = (
        _validate_probability(
            "reinforcement_strength",
            reinforcement_strength,
        )
        if reinforcement_present
        else 0.0
    )

    return ConditioningObservation(
        observation_id=observation_id,
        cue=cue,
        trajectory_id=trajectory.trajectory_id,
        terminal_attractor_id=terminal_attractor_id,
        dominant_series=dominant_series,
        reinforcement_present=bool(
            reinforcement_present
        ),
        reinforcement_strength=strength,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "source_trajectory_policy_free": True,
            "memory_mutated": False,
            "action_selected": False,
            "policy_modified": False,
            "biological_conditioning_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def _association_for_target(
    *,
    cue: ConditioningCue,
    target_attractor_id: str,
    observations: Sequence[
        ConditioningObservation
    ],
    acquisition_rate: float,
    extinction_rate: float,
) -> ConditioningAssociation:
    target = _validate_id(
        "target_attractor_id",
        target_attractor_id,
    )

    acquisition_rate = _validate_probability(
        "acquisition_rate",
        acquisition_rate,
    )

    extinction_rate = _validate_probability(
        "extinction_rate",
        extinction_rate,
    )

    relevant = tuple(
        observation
        for observation in observations
        if observation.terminal_attractor_id == target
    )

    if not relevant:
        raise ExperienceConditioningError(
            f"no observations for attractor: {target}"
        )

    acquisition = 0.0
    extinction = 0.0

    reinforced_count = 0
    extinction_count = 0

    for observation in relevant:
        if observation.reinforcement_present:
            reinforced_count += 1

            effective_rate = (
                acquisition_rate
                * observation.reinforcement_strength
            )

            acquisition += (
                1.0
                - acquisition
            ) * effective_rate

            extinction *= (
                1.0
                - effective_rate
            )

        else:
            extinction_count += 1

            extinction += (
                1.0
                - extinction
            ) * extinction_rate

            acquisition *= (
                1.0
                - extinction_rate
            )

    effective_strength = max(
        0.0,
        min(
            1.0,
            acquisition
            * (
                1.0
                - extinction
            ),
        ),
    )

    exposure_count = len(
        relevant
    )

    recurrence_factor = min(
        1.0,
        exposure_count
        / 5.0,
    )

    reinforcement_consistency = (
        reinforced_count
        / exposure_count
        if exposure_count > 0
        else 0.0
    )

    association_confidence = max(
        0.0,
        min(
            1.0,
            0.5
            * recurrence_factor
            + 0.5
            * reinforcement_consistency,
        ),
    )

    return ConditioningAssociation(
        cue_id=cue.cue_id,
        attractor_id=target,
        exposure_count=exposure_count,
        reinforced_exposure_count=(
            reinforced_count
        ),
        extinction_exposure_count=(
            extinction_count
        ),
        acquisition_strength=(
            acquisition
        ),
        extinction_strength=(
            extinction
        ),
        effective_association_strength=(
            effective_strength
        ),
        association_confidence=(
            association_confidence
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "association_mode": "descriptive",
            "memory_mutated": False,
            "biological_conditioning_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def build_conditioning_memory(
    *,
    memory_id: str,
    cue: ConditioningCue,
    observations: Iterable[
        ConditioningObservation
    ],
    acquisition_rate: float = 0.35,
    extinction_rate: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> ConditioningMemory:
    items = tuple(
        observations
    )

    if not items:
        raise ExperienceConditioningError(
            "observations must not be empty"
        )

    if any(
        observation.cue.cue_id
        != cue.cue_id
        for observation in items
    ):
        raise ExperienceConditioningError(
            "all observations must belong to the same cue"
        )

    target_ids = tuple(
        sorted(
            {
                observation.terminal_attractor_id
                for observation in items
                if observation.terminal_attractor_id
                is not None
            }
        )
    )

    if not target_ids:
        raise ExperienceConditioningError(
            "observations contain no terminal attractors"
        )

    associations = tuple(
        _association_for_target(
            cue=cue,
            target_attractor_id=target_id,
            observations=items,
            acquisition_rate=acquisition_rate,
            extinction_rate=extinction_rate,
        )
        for target_id in target_ids
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "conditioning_mode": "descriptive",
        "memory_mutated": False,
        "learning_applied_to_source_memory": False,
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

    return ConditioningMemory(
        memory_id=memory_id,
        cue=cue,
        observations=items,
        associations=associations,
        metadata=merged_metadata,
    )


def association_by_attractor_id(
    memory: ConditioningMemory,
    attractor_id: str,
) -> ConditioningAssociation:
    target = _validate_id(
        "attractor_id",
        attractor_id,
    )

    for association in memory.associations:
        if association.attractor_id == target:
            return association

    raise ExperienceConditioningError(
        f"unknown attractor_id: {target}"
    )


def dominant_conditioned_attractor(
    memory: ConditioningMemory,
) -> str:
    ordered = tuple(
        sorted(
            memory.associations,
            key=lambda association: (
                -association.effective_association_strength,
                -association.association_confidence,
                association.attractor_id,
            ),
        )
    )

    return ordered[
        0
    ].attractor_id


def conditioned_response(
    *,
    query_cue: ConditioningCue,
    memory: ConditioningMemory,
    generalization_scale: float = 1.0,
) -> ConditionedResponse:
    if generalization_scale <= 0.0:
        raise ExperienceConditioningError(
            "generalization_scale must be positive"
        )

    if len(
        query_cue.feature_vector
    ) != len(
        memory.cue.feature_vector
    ):
        raise ExperienceConditioningError(
            "cue vector dimensions must match"
        )

    cue_distance = _distance(
        query_cue.feature_vector,
        memory.cue.feature_vector,
    )

    generalization_weight = exp(
        -cue_distance
        / float(
            generalization_scale
        )
    )

    target_id = dominant_conditioned_attractor(
        memory
    )

    association = association_by_attractor_id(
        memory,
        target_id,
    )

    response_strength = max(
        0.0,
        min(
            1.0,
            association.effective_association_strength
            * association.association_confidence
            * generalization_weight,
        ),
    )

    return ConditionedResponse(
        query_cue_id=query_cue.cue_id,
        matched_memory_cue_id=(
            memory.cue.cue_id
        ),
        cue_distance=cue_distance,
        generalization_weight=(
            generalization_weight
        ),
        predicted_attractor_id=(
            target_id
        ),
        predicted_association_strength=(
            association.effective_association_strength
        ),
        conditioned_response_strength=(
            response_strength
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "response_mode": "conditioned_proxy",
            "source_memory_mutated": False,
            "action_selected": False,
            "policy_modified": False,
            "diagnosis_generated": False,
            "biological_conditioning_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def conditioning_memory_is_policy_free(
    memory: ConditioningMemory,
) -> bool:
    return (
        memory.metadata.get(
            "action_selected"
        )
        is False
        and memory.metadata.get(
            "policy_modified"
        )
        is False
        and memory.metadata.get(
            "memory_mutated"
        )
        is False
    )


def conditioned_response_is_policy_free(
    response: ConditionedResponse,
) -> bool:
    return (
        response.metadata.get(
            "action_selected"
        )
        is False
        and response.metadata.get(
            "policy_modified"
        )
        is False
        and response.metadata.get(
            "source_memory_mutated"
        )
        is False
    )


def association_strength_series(
    memory: ConditioningMemory,
) -> tuple[float, ...]:
    return tuple(
        association.effective_association_strength
        for association in memory.associations
    )


def association_confidence_series(
    memory: ConditioningMemory,
) -> tuple[float, ...]:
    return tuple(
        association.association_confidence
        for association in memory.associations
    )


__all__ = [
    "ConditionedResponse",
    "ConditioningAssociation",
    "ConditioningCue",
    "ConditioningMemory",
    "ConditioningObservation",
    "ExperienceConditioningError",
    "SCHEMA_VERSION",
    "association_by_attractor_id",
    "association_confidence_series",
    "association_strength_series",
    "build_conditioning_memory",
    "conditioned_response",
    "conditioned_response_is_policy_free",
    "conditioning_memory_is_policy_free",
    "dominant_conditioned_attractor",
    "observation_from_trajectory",
]
