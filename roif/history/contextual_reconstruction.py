"""
ROIF Memory 2.0
Contextual Reconstruction Core

Fourteenth layer above:
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

Purpose
-------
Represent descriptive reconstruction of a cue under an associative context.

The core comparison is:

    same cue
    + different associative context
        -> different attractor weighting
        -> different reconstructed meaning vector

This module computes:
- cue-to-attractor compatibility
- context-weighted attractor activation
- normalized reconstruction weights
- reconstructed meaning vector
- dominant reconstructed attractor
- reconstruction ambiguity
- semantic shift from cue baseline
- context contribution to reconstruction

This module does NOT:
- learn
- mutate memory
- mutate attractors
- select actions
- modify policy
- diagnose
- claim biological semantic reconstruction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_attractor import ExperienceAttractor
from roif.history.experience_conditioning import ConditioningCue
from roif.history.associative_context import (
    AssociativeContext,
    associative_context_is_policy_free,
)


SCHEMA_VERSION = "contextual_reconstruction_v1"


class ContextualReconstructionError(ValueError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _validate_id(name: str, value: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ContextualReconstructionError(
            f"{name} must not be empty"
        )
    return cleaned


def _validate_finite(name: str, value: float) -> float:
    x = float(value)
    if not isfinite(x):
        raise ContextualReconstructionError(
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
        raise ContextualReconstructionError(
            f"{name} must not be empty"
        )

    return vector


def _distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise ContextualReconstructionError(
            "vector dimensions must match"
        )

    return sqrt(
        sum(
            (float(x) - float(y)) ** 2
            for x, y in zip(a, b)
        )
    )


def _weighted_mean(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, ...]:
    if not vectors:
        raise ContextualReconstructionError(
            "vectors must not be empty"
        )

    if len(vectors) != len(weights):
        raise ContextualReconstructionError(
            "vectors and weights must have equal length"
        )

    width = len(vectors[0])

    if width == 0:
        raise ContextualReconstructionError(
            "vectors must not be empty"
        )

    if any(len(vector) != width for vector in vectors):
        raise ContextualReconstructionError(
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
class ReconstructionContribution:
    attractor_id: str

    cue_distance: float
    cue_compatibility: float

    context_bias: float
    raw_reconstruction_weight: float
    normalized_reconstruction_weight: float

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
            "cue_distance",
            "cue_compatibility",
            "context_bias",
            "raw_reconstruction_weight",
            "normalized_reconstruction_weight",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(self, name),
                ),
            )

        if self.cue_distance < 0.0:
            raise ContextualReconstructionError(
                "cue_distance must be non-negative"
            )

        for name in (
            "cue_compatibility",
            "normalized_reconstruction_weight",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ContextualReconstructionError(
                    f"{name} must be within [0,1]"
                )

        if self.context_bias < 0.0:
            raise ContextualReconstructionError(
                "context_bias must be non-negative"
            )

        if self.raw_reconstruction_weight < 0.0:
            raise ContextualReconstructionError(
                "raw_reconstruction_weight must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ContextualReconstructionResult:
    reconstruction_id: str
    cue: ConditioningCue
    context: AssociativeContext

    contributions: tuple[
        ReconstructionContribution,
        ...,
    ]

    reconstructed_vector: tuple[float, ...]

    dominant_attractor_id: str
    dominant_weight: float

    ambiguity: float
    semantic_shift_from_cue: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reconstruction_id",
            _validate_id(
                "reconstruction_id",
                self.reconstruction_id,
            ),
        )

        if not self.contributions:
            raise ContextualReconstructionError(
                "contributions must not be empty"
            )

        object.__setattr__(
            self,
            "contributions",
            tuple(self.contributions),
        )

        object.__setattr__(
            self,
            "reconstructed_vector",
            _validate_vector(
                "reconstructed_vector",
                self.reconstructed_vector,
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

        object.__setattr__(
            self,
            "dominant_weight",
            _validate_finite(
                "dominant_weight",
                self.dominant_weight,
            ),
        )

        object.__setattr__(
            self,
            "ambiguity",
            _validate_finite(
                "ambiguity",
                self.ambiguity,
            ),
        )

        object.__setattr__(
            self,
            "semantic_shift_from_cue",
            _validate_finite(
                "semantic_shift_from_cue",
                self.semantic_shift_from_cue,
            ),
        )

        for name in (
            "dominant_weight",
            "ambiguity",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ContextualReconstructionError(
                    f"{name} must be within [0,1]"
                )

        if self.semantic_shift_from_cue < 0.0:
            raise ContextualReconstructionError(
                "semantic_shift_from_cue must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


def _project_cue_to_attractor_space(
    cue: ConditioningCue,
    target_width: int,
) -> tuple[float, ...]:
    source = cue.feature_vector

    if len(source) == target_width:
        return source

    if len(source) > target_width:
        return tuple(
            source[:target_width]
        )

    if not source:
        raise ContextualReconstructionError(
            "cue feature vector must not be empty"
        )

    expanded = []
    index = 0

    while len(expanded) < target_width:
        expanded.append(
            source[index % len(source)]
        )
        index += 1

    return tuple(expanded)


def reconstruct_contextual_meaning(
    *,
    reconstruction_id: str,
    cue: ConditioningCue,
    context: AssociativeContext,
    attractors: Iterable[ExperienceAttractor],
    compatibility_scale: float = 1.0,
    baseline_cue_weight: float = 0.25,
    context_gain: float = 2.0,
    metadata: Mapping[str, Any] | None = None,
) -> ContextualReconstructionResult:
    attractor_items = tuple(attractors)

    if not attractor_items:
        raise ContextualReconstructionError(
            "attractors must not be empty"
        )

    if not associative_context_is_policy_free(context):
        raise ContextualReconstructionError(
            "associative context must be policy-free"
        )

    if compatibility_scale <= 0.0:
        raise ContextualReconstructionError(
            "compatibility_scale must be positive"
        )

    if baseline_cue_weight < 0.0:
        raise ContextualReconstructionError(
            "baseline_cue_weight must be non-negative"
        )

    context_gain = _validate_finite(
        "context_gain",
        context_gain,
    )

    if context_gain < 0.0:
        raise ContextualReconstructionError(
            "context_gain must be non-negative"
        )

    width = len(attractor_items[0].center)

    if any(
        len(attractor.center) != width
        for attractor in attractor_items
    ):
        raise ContextualReconstructionError(
            "all attractor centers must have equal dimension"
        )

    cue_vector = _project_cue_to_attractor_space(
        cue,
        width,
    )

    context_bias_map = dict(
        context.prestress.bias_by_attractor_id
    )

    rows = []

    for attractor in attractor_items:
        cue_distance = _distance(
            cue_vector,
            attractor.center,
        )

        cue_compatibility = exp(
            -cue_distance
            / float(compatibility_scale)
        )

        context_bias = max(
            0.0,
            float(
                context_bias_map.get(
                    attractor.attractor_id,
                    0.0,
                )
            ),
        )

        raw_weight = (
            cue_compatibility
            * (
                1.0
                + context_gain
                * context_bias
            )
            * attractor.metrics.attractor_confidence
        )

        rows.append(
            (
                attractor,
                cue_distance,
                cue_compatibility,
                context_bias,
                raw_weight,
            )
        )

    raw_total = sum(
        row[4]
        for row in rows
    )

    if raw_total <= 0.0:
        normalized = tuple(
            1.0 / len(rows)
            for _ in rows
        )
    else:
        normalized = tuple(
            row[4] / raw_total
            for row in rows
        )

    contributions = tuple(
        ReconstructionContribution(
            attractor_id=row[0].attractor_id,
            cue_distance=row[1],
            cue_compatibility=row[2],
            context_bias=row[3],
            raw_reconstruction_weight=row[4],
            normalized_reconstruction_weight=weight,
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
        for row, weight in zip(
            rows,
            normalized,
        )
    )

    dominant = sorted(
        contributions,
        key=lambda item: (
            -item.normalized_reconstruction_weight,
            item.attractor_id,
        ),
    )[0]

    attractor_vectors = tuple(
        attractor.center
        for attractor in attractor_items
    )

    attractor_weights = tuple(
        contribution.normalized_reconstruction_weight
        for contribution in contributions
    )

    attractor_reconstruction = _weighted_mean(
        attractor_vectors,
        attractor_weights,
    )

    combined_vectors = (
        cue_vector,
        attractor_reconstruction,
    )

    combined_weights = (
        baseline_cue_weight,
        1.0,
    )

    reconstructed_vector = _weighted_mean(
        combined_vectors,
        combined_weights,
    )

    ambiguity = max(
        0.0,
        min(
            1.0,
            1.0
            - sum(
                weight ** 2
                for weight in attractor_weights
            ),
        ),
    )

    semantic_shift = _distance(
        cue_vector,
        reconstructed_vector,
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "contextual_reconstruction_mode": "descriptive",
        "context_id": context.context_id,
        "context_gain": context_gain,
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

    return ContextualReconstructionResult(
        reconstruction_id=reconstruction_id,
        cue=cue,
        context=context,
        contributions=contributions,
        reconstructed_vector=reconstructed_vector,
        dominant_attractor_id=(
            dominant.attractor_id
        ),
        dominant_weight=(
            dominant.normalized_reconstruction_weight
        ),
        ambiguity=ambiguity,
        semantic_shift_from_cue=semantic_shift,
        metadata=merged_metadata,
    )


def reconstruction_contribution_by_id(
    result: ContextualReconstructionResult,
    attractor_id: str,
) -> ReconstructionContribution:
    target = _validate_id(
        "attractor_id",
        attractor_id,
    )

    for contribution in result.contributions:
        if contribution.attractor_id == target:
            return contribution

    raise ContextualReconstructionError(
        f"unknown attractor_id: {target}"
    )


def reconstruction_weight_series(
    result: ContextualReconstructionResult,
) -> tuple[float, ...]:
    return tuple(
        contribution.normalized_reconstruction_weight
        for contribution in result.contributions
    )


def reconstruction_is_policy_free(
    result: ContextualReconstructionResult,
) -> bool:
    return (
        result.metadata.get("action_selected") is False
        and result.metadata.get("policy_modified") is False
        and result.metadata.get("memory_mutated") is False
        and result.metadata.get("learning_applied") is False
        and associative_context_is_policy_free(result.context)
    )


def compare_contextual_reconstructions(
    left: ContextualReconstructionResult,
    right: ContextualReconstructionResult,
) -> float:
    if left.cue != right.cue:
        raise ContextualReconstructionError(
            "reconstructions must use the same cue"
        )

    return _distance(
        left.reconstructed_vector,
        right.reconstructed_vector,
    )


__all__ = [
    "ContextualReconstructionError",
    "ContextualReconstructionResult",
    "ReconstructionContribution",
    "SCHEMA_VERSION",
    "compare_contextual_reconstructions",
    "reconstruct_contextual_meaning",
    "reconstruction_contribution_by_id",
    "reconstruction_is_policy_free",
    "reconstruction_weight_series",
]

