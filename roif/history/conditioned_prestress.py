"""
ROIF Memory 2.0
Conditioned Prestress Core

Eighth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor
    AttractorDynamics
    AttractorTrajectory
    ExperienceConditioning

Purpose
-------
Convert descriptive conditioned associations into temporary attractor prestress.

The core idea:

    conditioned cue
        -> conditioned association
        -> temporary prestress bias
        -> altered attractor competition

This layer does NOT:
- mutate conditioning memory
- learn
- select actions
- modify policy
- diagnose
- claim biological conditioning/prestress mechanisms

All values are descriptive computational proxies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from roif.history.attractor_dynamics import (
    AttractorPrestress,
)
from roif.history.experience_conditioning import (
    ConditionedResponse,
    ConditioningMemory,
    ConditioningCue,
    conditioned_response,
    conditioning_memory_is_policy_free,
)


SCHEMA_VERSION = "conditioned_prestress_v1"


class ConditionedPrestressError(ValueError):
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
        raise ConditionedPrestressError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise ConditionedPrestressError(
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
        raise ConditionedPrestressError(
            f"{name} must be non-negative"
        )

    return x


@dataclass(frozen=True, slots=True)
class ConditionedPrestressConfig:
    max_bias: float = 1.0
    strength_gain: float = 1.0
    confidence_gain: float = 1.0
    generalization_gain: float = 1.0
    minimum_response_strength: float = 0.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "max_bias",
            "strength_gain",
            "confidence_gain",
            "generalization_gain",
            "minimum_response_strength",
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

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionedPrestressResult:
    query_cue_id: str
    memory_id: str

    conditioned_response: ConditionedResponse
    prestress: AttractorPrestress

    raw_bias: float
    applied_bias: float
    suppressed_by_threshold: bool

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
            "memory_id",
            _validate_id(
                "memory_id",
                self.memory_id,
            ),
        )

        object.__setattr__(
            self,
            "raw_bias",
            _validate_nonnegative(
                "raw_bias",
                self.raw_bias,
            ),
        )

        object.__setattr__(
            self,
            "applied_bias",
            _validate_nonnegative(
                "applied_bias",
                self.applied_bias,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _association_confidence_for_predicted(
    memory: ConditioningMemory,
    attractor_id: str,
) -> float:
    for association in memory.associations:
        if association.attractor_id == attractor_id:
            return association.association_confidence

    raise ConditionedPrestressError(
        f"predicted attractor not found in memory: {attractor_id}"
    )


def _compute_raw_bias(
    *,
    response: ConditionedResponse,
    association_confidence: float,
    config: ConditionedPrestressConfig,
) -> float:
    return (
        response.predicted_association_strength
        * config.strength_gain
        * association_confidence
        * config.confidence_gain
        * response.generalization_weight
        * config.generalization_gain
    )


def build_conditioned_prestress(
    *,
    prestress_id: str,
    query_cue: ConditioningCue,
    memory: ConditioningMemory,
    config: ConditionedPrestressConfig | None = None,
    generalization_scale: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> ConditionedPrestressResult:
    if not conditioning_memory_is_policy_free(
        memory
    ):
        raise ConditionedPrestressError(
            "conditioning memory must be policy-free"
        )

    actual_config = (
        config
        if config is not None
        else ConditionedPrestressConfig()
    )

    response = conditioned_response(
        query_cue=query_cue,
        memory=memory,
        generalization_scale=generalization_scale,
    )

    confidence = _association_confidence_for_predicted(
        memory,
        response.predicted_attractor_id,
    )

    raw_bias = _compute_raw_bias(
        response=response,
        association_confidence=confidence,
        config=actual_config,
    )

    suppressed = (
        response.conditioned_response_strength
        < actual_config.minimum_response_strength
    )

    applied_bias = (
        0.0
        if suppressed
        else min(
            actual_config.max_bias,
            raw_bias,
        )
    )

    prestress = AttractorPrestress(
        prestress_id=prestress_id,
        bias_by_attractor_id={
            response.predicted_attractor_id: applied_bias,
        },
        metadata={
            "schema_version": SCHEMA_VERSION,
            "source": "conditioned_association",
            "query_cue_id": query_cue.cue_id,
            "memory_id": memory.memory_id,
            "memory_mutated": False,
            "learning_applied": False,
            "action_selected": False,
            "policy_modified": False,
            "biological_conditioning_claimed": False,
            "causal_truth_inferred": False,
        },
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "conditioned_prestress_mode": "descriptive",
        "memory_mutated": False,
        "learning_applied": False,
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

    return ConditionedPrestressResult(
        query_cue_id=query_cue.cue_id,
        memory_id=memory.memory_id,
        conditioned_response=response,
        prestress=prestress,
        raw_bias=raw_bias,
        applied_bias=applied_bias,
        suppressed_by_threshold=suppressed,
        metadata=merged_metadata,
    )


def conditioned_prestress_is_policy_free(
    result: ConditionedPrestressResult,
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


def conditioned_prestress_bias(
    result: ConditionedPrestressResult,
) -> float:
    return result.applied_bias


def conditioned_prestress_target_id(
    result: ConditionedPrestressResult,
) -> str:
    keys = tuple(
        result.prestress.bias_by_attractor_id.keys()
    )

    if len(keys) != 1:
        raise ConditionedPrestressError(
            "conditioned prestress must target exactly one attractor"
        )

    return keys[
        0
    ]


__all__ = [
    "ConditionedPrestressConfig",
    "ConditionedPrestressError",
    "ConditionedPrestressResult",
    "SCHEMA_VERSION",
    "build_conditioned_prestress",
    "conditioned_prestress_bias",
    "conditioned_prestress_is_policy_free",
    "conditioned_prestress_target_id",
]
