"""
ROIF Memory 2.0
Attractor Dynamics Core

Fifth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor

Purpose
-------
Model descriptive state-space dynamics around already-built experience attractors.

Given:
    - a current state feature vector,
    - one or more ExperienceAttractor objects,
    - optional prestress bias,

compute:
    - basin distance,
    - attractor activation,
    - competition,
    - capture strength,
    - weighted state shift.

This module does NOT:
    - learn new attractors,
    - mutate memory,
    - select actions,
    - modify policy,
    - diagnose,
    - claim biological attractor dynamics.

All quantities are computational descriptors only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, sqrt
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_attractor import (
    ExperienceAttractor,
    attractor_is_policy_free,
)


SCHEMA_VERSION = "attractor_dynamics_v1"


class AttractorDynamicsError(ValueError):
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
        raise AttractorDynamicsError(
            f"{name} must not be empty"
        )
    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)
    if not isfinite(x):
        raise AttractorDynamicsError(
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
        for index, value
        in enumerate(values)
    )

    if not vector:
        raise AttractorDynamicsError(
            f"{name} must not be empty"
        )

    return vector


def _distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise AttractorDynamicsError(
            "vector dimensions must match"
        )

    return sqrt(
        sum(
            (float(x) - float(y)) ** 2
            for x, y in zip(a, b)
        )
    )


def _weighted_sum(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, ...]:
    if not vectors:
        raise AttractorDynamicsError(
            "vectors must not be empty"
        )

    if len(vectors) != len(weights):
        raise AttractorDynamicsError(
            "vectors and weights must have equal length"
        )

    width = len(vectors[0])

    if width == 0:
        raise AttractorDynamicsError(
            "vectors must not be empty"
        )

    if any(
        len(vector) != width
        for vector in vectors
    ):
        raise AttractorDynamicsError(
            "vector dimensions must match"
        )

    total_weight = sum(
        float(weight)
        for weight in weights
    )

    if total_weight <= 0.0:
        return tuple(
            0.0
            for _ in range(width)
        )

    return tuple(
        sum(
            float(vector[index]) * float(weight)
            for vector, weight
            in zip(vectors, weights)
        )
        / total_weight
        for index in range(width)
    )


def _subtract(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, ...]:
    if len(a) != len(b):
        raise AttractorDynamicsError(
            "vector dimensions must match"
        )

    return tuple(
        float(x) - float(y)
        for x, y in zip(a, b)
    )


def _add(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, ...]:
    if len(a) != len(b):
        raise AttractorDynamicsError(
            "vector dimensions must match"
        )

    return tuple(
        float(x) + float(y)
        for x, y in zip(a, b)
    )


def _scale(
    vector: Sequence[float],
    factor: float,
) -> tuple[float, ...]:
    return tuple(
        float(value) * float(factor)
        for value in vector
    )


@dataclass(frozen=True, slots=True)
class DynamicsState:
    state_id: str
    feature_vector: tuple[float, ...]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_id",
            _validate_id(
                "state_id",
                self.state_id,
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
class AttractorPrestress:
    """
    Optional contextual bias applied before competition.

    bias_by_attractor_id:
        additive scalar bias to attractor activation.

    Positive bias raises effective activation.
    Negative bias lowers it.

    This is not learning and does not mutate the attractor.
    """

    prestress_id: str
    bias_by_attractor_id: Mapping[str, float]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prestress_id",
            _validate_id(
                "prestress_id",
                self.prestress_id,
            ),
        )

        cleaned = {}

        for key, value in dict(
            self.bias_by_attractor_id
        ).items():
            attractor_id = _validate_id(
                "attractor_id",
                key,
            )

            cleaned[
                attractor_id
            ] = _validate_finite(
                f"bias[{attractor_id}]",
                value,
            )

        object.__setattr__(
            self,
            "bias_by_attractor_id",
            MappingProxyType(
                cleaned
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
class AttractorActivation:
    attractor_id: str

    basin_distance: float
    raw_activation: float
    prestress_bias: float
    biased_activation: float

    competition_weight: float
    capture_strength: float

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
            "basin_distance",
            "raw_activation",
            "prestress_bias",
            "biased_activation",
            "competition_weight",
            "capture_strength",
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

        if self.basin_distance < 0.0:
            raise AttractorDynamicsError(
                "basin_distance must be non-negative"
            )

        for name in (
            "raw_activation",
            "biased_activation",
            "competition_weight",
            "capture_strength",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise AttractorDynamicsError(
                    f"{name} must be non-negative"
                )

        if not 0.0 <= self.competition_weight <= 1.0:
            raise AttractorDynamicsError(
                "competition_weight must be within [0,1]"
            )

        if not 0.0 <= self.capture_strength <= 1.0:
            raise AttractorDynamicsError(
                "capture_strength must be within [0,1]"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AttractorDynamicsResult:
    state_before: DynamicsState

    activations: tuple[
        AttractorActivation,
        ...,
    ]

    dominant_attractor_id: str | None
    dominant_competition_weight: float

    weighted_target: tuple[float, ...]
    state_shift: tuple[float, ...]
    state_after: DynamicsState

    competition_entropy_proxy: float
    capture_margin: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.activations:
            raise AttractorDynamicsError(
                "activations must not be empty"
            )

        object.__setattr__(
            self,
            "activations",
            tuple(
                self.activations
            ),
        )

        object.__setattr__(
            self,
            "weighted_target",
            _validate_vector(
                "weighted_target",
                self.weighted_target,
            ),
        )

        object.__setattr__(
            self,
            "state_shift",
            _validate_vector(
                "state_shift",
                self.state_shift,
            ),
        )

        object.__setattr__(
            self,
            "dominant_competition_weight",
            _validate_finite(
                "dominant_competition_weight",
                self.dominant_competition_weight,
            ),
        )

        object.__setattr__(
            self,
            "competition_entropy_proxy",
            _validate_finite(
                "competition_entropy_proxy",
                self.competition_entropy_proxy,
            ),
        )

        object.__setattr__(
            self,
            "capture_margin",
            _validate_finite(
                "capture_margin",
                self.capture_margin,
            ),
        )

        if not 0.0 <= self.dominant_competition_weight <= 1.0:
            raise AttractorDynamicsError(
                "dominant_competition_weight must be within [0,1]"
            )

        if self.competition_entropy_proxy < 0.0:
            raise AttractorDynamicsError(
                "competition_entropy_proxy must be non-negative"
            )

        if self.capture_margin < 0.0:
            raise AttractorDynamicsError(
                "capture_margin must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _activation_from_distance(
    distance: float,
    *,
    distance_scale: float,
) -> float:
    """
    Exponential basin activation.

        activation = exp(-distance / distance_scale)

    distance=0 -> 1
    distance grows -> activation approaches 0
    """

    if distance_scale <= 0.0:
        raise AttractorDynamicsError(
            "distance_scale must be positive"
        )

    return exp(
        -float(distance)
        / float(distance_scale)
    )


def _soft_competition_weights(
    activations: Sequence[float],
) -> tuple[float, ...]:
    if not activations:
        raise AttractorDynamicsError(
            "activations must not be empty"
        )

    cleaned = tuple(
        max(
            0.0,
            float(value),
        )
        for value in activations
    )

    total = sum(
        cleaned
    )

    if total <= 0.0:
        uniform = 1.0 / len(
            cleaned
        )

        return tuple(
            uniform
            for _ in cleaned
        )

    return tuple(
        value / total
        for value in cleaned
    )


def _capture_strength(
    *,
    competition_weight: float,
    attractor_confidence: float,
    compactness: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            mean(
                (
                    competition_weight,
                    attractor_confidence,
                    compactness,
                )
            ),
        ),
    )


def _entropy_proxy(
    weights: Sequence[float],
) -> float:
    """
    Simple uncertainty proxy:
        1 - sum(w^2)

    0 when one attractor dominates completely.
    Higher when competition is distributed.
    """

    return max(
        0.0,
        1.0
        - sum(
            float(weight) ** 2
            for weight in weights
        ),
    )


def run_attractor_dynamics(
    *,
    state: DynamicsState,
    attractors: Iterable[
        ExperienceAttractor
    ],
    prestress: AttractorPrestress | None = None,
    distance_scale: float = 1.0,
    step_size: float = 0.25,
) -> AttractorDynamicsResult:
    attractor_items = tuple(
        attractors
    )

    if not attractor_items:
        raise AttractorDynamicsError(
            "attractors must not be empty"
        )

    if not all(
        attractor_is_policy_free(
            attractor
        )
        for attractor in attractor_items
    ):
        raise AttractorDynamicsError(
            "all attractors must be policy-free"
        )

    if distance_scale <= 0.0:
        raise AttractorDynamicsError(
            "distance_scale must be positive"
        )

    if not 0.0 <= step_size <= 1.0:
        raise AttractorDynamicsError(
            "step_size must be within [0,1]"
        )

    width = len(
        state.feature_vector
    )

    if any(
        len(
            attractor.center
        ) != width
        for attractor in attractor_items
    ):
        raise AttractorDynamicsError(
            "state and attractor dimensions must match"
        )

    raw_rows = []

    for attractor in attractor_items:
        distance = _distance(
            state.feature_vector,
            attractor.center,
        )

        raw_activation = _activation_from_distance(
            distance,
            distance_scale=distance_scale,
        )

        bias = (
            prestress.bias_by_attractor_id.get(
                attractor.attractor_id,
                0.0,
            )
            if prestress is not None
            else 0.0
        )

        biased_activation = max(
            0.0,
            raw_activation
            + bias,
        )

        raw_rows.append(
            (
                attractor,
                distance,
                raw_activation,
                bias,
                biased_activation,
            )
        )

    competition_weights = _soft_competition_weights(
        tuple(
            row[
                4
            ]
            for row in raw_rows
        )
    )

    activations = []

    for row, competition_weight in zip(
        raw_rows,
        competition_weights,
    ):
        attractor = row[
            0
        ]

        capture = _capture_strength(
            competition_weight=competition_weight,
            attractor_confidence=(
                attractor.metrics.attractor_confidence
            ),
            compactness=(
                attractor.metrics.compactness
            ),
        )

        activations.append(
            AttractorActivation(
                attractor_id=(
                    attractor.attractor_id
                ),
                basin_distance=row[
                    1
                ],
                raw_activation=row[
                    2
                ],
                prestress_bias=row[
                    3
                ],
                biased_activation=row[
                    4
                ],
                competition_weight=(
                    competition_weight
                ),
                capture_strength=(
                    capture
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
        )

    activation_items = tuple(
        activations
    )

    ordered = tuple(
        sorted(
            activation_items,
            key=lambda item: (
                -item.competition_weight,
                item.attractor_id,
            ),
        )
    )

    dominant = ordered[
        0
    ]

    second_weight = (
        ordered[
            1
        ].competition_weight
        if len(
            ordered
        ) > 1
        else 0.0
    )

    capture_margin = max(
        0.0,
        dominant.competition_weight
        - second_weight,
    )

    attractor_by_id = {
        attractor.attractor_id: attractor
        for attractor in attractor_items
    }

    centers = tuple(
        attractor_by_id[
            activation.attractor_id
        ].center
        for activation in activation_items
    )

    shift_weights = tuple(
        activation.capture_strength
        for activation in activation_items
    )

    weighted_target = _weighted_sum(
        centers,
        shift_weights,
    )

    direction = _subtract(
        weighted_target,
        state.feature_vector,
    )

    state_shift = _scale(
        direction,
        step_size,
    )

    state_after_vector = _add(
        state.feature_vector,
        state_shift,
    )

    state_after = DynamicsState(
        state_id=(
            f"{state.state_id}::after"
        ),
        feature_vector=(
            state_after_vector
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "derived_from_attractor_dynamics": True,
            "memory_mutated": False,
            "learning_applied": False,
        },
    )

    return AttractorDynamicsResult(
        state_before=state,
        activations=activation_items,
        dominant_attractor_id=(
            dominant.attractor_id
        ),
        dominant_competition_weight=(
            dominant.competition_weight
        ),
        weighted_target=(
            weighted_target
        ),
        state_shift=(
            state_shift
        ),
        state_after=(
            state_after
        ),
        competition_entropy_proxy=(
            _entropy_proxy(
                competition_weights
            )
        ),
        capture_margin=(
            capture_margin
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "attractor_dynamics_mode": "descriptive",
            "prestress_used": (
                prestress
                is not None
            ),
            "memory_mutated": False,
            "learning_applied": False,
            "action_selected": False,
            "policy_modified": False,
            "diagnosis_generated": False,
            "biological_attractor_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def dynamics_is_policy_free(
    result: AttractorDynamicsResult,
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


def activation_by_id(
    result: AttractorDynamicsResult,
    attractor_id: str,
) -> AttractorActivation:
    target = _validate_id(
        "attractor_id",
        attractor_id,
    )

    for activation in result.activations:
        if activation.attractor_id == target:
            return activation

    raise AttractorDynamicsError(
        f"unknown attractor_id: {target}"
    )


def competition_weight_series(
    result: AttractorDynamicsResult,
) -> tuple[float, ...]:
    return tuple(
        activation.competition_weight
        for activation in result.activations
    )


def capture_strength_series(
    result: AttractorDynamicsResult,
) -> tuple[float, ...]:
    return tuple(
        activation.capture_strength
        for activation in result.activations
    )


def basin_distance_series(
    result: AttractorDynamicsResult,
) -> tuple[float, ...]:
    return tuple(
        activation.basin_distance
        for activation in result.activations
    )


def state_shift_magnitude(
    result: AttractorDynamicsResult,
) -> float:
    return sqrt(
        sum(
            float(value) ** 2
            for value in result.state_shift
        )
    )


__all__ = [
    "AttractorActivation",
    "AttractorDynamicsError",
    "AttractorDynamicsResult",
    "AttractorPrestress",
    "DynamicsState",
    "SCHEMA_VERSION",
    "activation_by_id",
    "basin_distance_series",
    "capture_strength_series",
    "competition_weight_series",
    "dynamics_is_policy_free",
    "run_attractor_dynamics",
    "state_shift_magnitude",
]
