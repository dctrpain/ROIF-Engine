"""
ROIF Memory 2.0
Associative Context Core

Eleventh layer above:
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

Purpose
-------
Combine multiple conditioned cues into one descriptive associative context.

Instead of:

    one cue -> one temporary prestress

this layer models:

    cue_1
    cue_2
    cue_3
      -> conditioned contributions
      -> associative context
      -> aggregated attractor prestress

This supports the idea that the meaning/effect of a cue can depend on the
currently active ensemble of related cues.

The module computes:
- per-cue conditioned contribution
- cue salience weighting
- target-attractor aggregation
- context competition
- context confidence
- associative-context prestress

This module does NOT:
- mutate source conditioning memories
- learn
- select actions
- modify policy
- diagnose
- claim a biological associative-context mechanism
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.attractor_dynamics import (
    AttractorPrestress,
)
from roif.history.experience_conditioning import (
    ConditioningCue,
    ConditioningMemory,
    conditioning_memory_is_policy_free,
)
from roif.history.conditioned_prestress import (
    ConditionedPrestressConfig,
    ConditionedPrestressResult,
    build_conditioned_prestress,
    conditioned_prestress_is_policy_free,
)


SCHEMA_VERSION = "associative_context_v1"


class AssociativeContextError(ValueError):
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
        raise AssociativeContextError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise AssociativeContextError(
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
        raise AssociativeContextError(
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
        raise AssociativeContextError(
            f"{name} must be within [0,1]"
        )

    return x


@dataclass(frozen=True, slots=True)
class AssociativeCueInput:
    cue: ConditioningCue
    memory: ConditioningMemory
    salience: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "salience",
            _validate_nonnegative(
                "salience",
                self.salience,
            ),
        )

        if not conditioning_memory_is_policy_free(
            self.memory
        ):
            raise AssociativeContextError(
                "conditioning memory must be policy-free"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AssociativeCueContribution:
    cue_id: str
    memory_id: str
    target_attractor_id: str

    salience: float
    conditioned_bias: float
    weighted_bias: float

    conditioned_response_strength: float
    generalization_weight: float
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
            "memory_id",
            _validate_id(
                "memory_id",
                self.memory_id,
            ),
        )

        object.__setattr__(
            self,
            "target_attractor_id",
            _validate_id(
                "target_attractor_id",
                self.target_attractor_id,
            ),
        )

        for name in (
            "salience",
            "conditioned_bias",
            "weighted_bias",
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

        for name in (
            "conditioned_response_strength",
            "generalization_weight",
            "association_confidence",
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
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AttractorContextAggregate:
    attractor_id: str

    contributor_count: int
    total_weighted_bias: float
    normalized_context_weight: float

    mean_response_strength: float
    mean_generalization_weight: float
    mean_association_confidence: float

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

        if self.contributor_count < 1:
            raise AssociativeContextError(
                "contributor_count must be >= 1"
            )

        object.__setattr__(
            self,
            "total_weighted_bias",
            _validate_nonnegative(
                "total_weighted_bias",
                self.total_weighted_bias,
            ),
        )

        object.__setattr__(
            self,
            "normalized_context_weight",
            _validate_unit_interval(
                "normalized_context_weight",
                self.normalized_context_weight,
            ),
        )

        for name in (
            "mean_response_strength",
            "mean_generalization_weight",
            "mean_association_confidence",
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
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AssociativeContext:
    context_id: str

    contributions: tuple[
        AssociativeCueContribution,
        ...,
    ]

    aggregates: tuple[
        AttractorContextAggregate,
        ...,
    ]

    prestress: AttractorPrestress

    dominant_attractor_id: str
    context_confidence: float
    competition_margin: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_id",
            _validate_id(
                "context_id",
                self.context_id,
            ),
        )

        if not self.contributions:
            raise AssociativeContextError(
                "contributions must not be empty"
            )

        if not self.aggregates:
            raise AssociativeContextError(
                "aggregates must not be empty"
            )

        object.__setattr__(
            self,
            "contributions",
            tuple(
                self.contributions
            ),
        )

        object.__setattr__(
            self,
            "aggregates",
            tuple(
                self.aggregates
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
            "context_confidence",
            _validate_unit_interval(
                "context_confidence",
                self.context_confidence,
            ),
        )

        object.__setattr__(
            self,
            "competition_margin",
            _validate_unit_interval(
                "competition_margin",
                self.competition_margin,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _association_confidence(
    result: ConditionedPrestressResult,
    memory: ConditioningMemory,
) -> float:
    target_ids = tuple(
        result.prestress.bias_by_attractor_id.keys()
    )

    if len(
        target_ids
    ) != 1:
        raise AssociativeContextError(
            "conditioned prestress must target exactly one attractor"
        )

    target = target_ids[
        0
    ]

    for association in memory.associations:
        if association.attractor_id == target:
            return association.association_confidence

    raise AssociativeContextError(
        f"target attractor not found in conditioning memory: {target}"
    )


def _build_contribution(
    *,
    cue_input: AssociativeCueInput,
    prestress_config: ConditionedPrestressConfig | None,
    generalization_scale: float,
) -> AssociativeCueContribution:
    result = build_conditioned_prestress(
        prestress_id=(
            f"context::{cue_input.cue.cue_id}"
        ),
        query_cue=cue_input.cue,
        memory=cue_input.memory,
        config=prestress_config,
        generalization_scale=generalization_scale,
    )

    if not conditioned_prestress_is_policy_free(
        result
    ):
        raise AssociativeContextError(
            "conditioned prestress must be policy-free"
        )

    target_ids = tuple(
        result.prestress.bias_by_attractor_id.keys()
    )

    if len(
        target_ids
    ) != 1:
        raise AssociativeContextError(
            "conditioned prestress must target exactly one attractor"
        )

    target_id = target_ids[
        0
    ]

    confidence = _association_confidence(
        result,
        cue_input.memory,
    )

    weighted_bias = (
        result.applied_bias
        * cue_input.salience
    )

    return AssociativeCueContribution(
        cue_id=cue_input.cue.cue_id,
        memory_id=cue_input.memory.memory_id,
        target_attractor_id=target_id,
        salience=cue_input.salience,
        conditioned_bias=result.applied_bias,
        weighted_bias=weighted_bias,
        conditioned_response_strength=(
            result.conditioned_response.conditioned_response_strength
        ),
        generalization_weight=(
            result.conditioned_response.generalization_weight
        ),
        association_confidence=confidence,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "source_conditioned_prestress_policy_free": True,
            "memory_mutated": False,
            "learning_applied": False,
            "action_selected": False,
            "policy_modified": False,
            "biological_context_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def _aggregate_contributions(
    contributions: Sequence[
        AssociativeCueContribution
    ],
) -> tuple[
    AttractorContextAggregate,
    ...,
]:
    grouped: dict[
        str,
        list[AssociativeCueContribution],
    ] = {}

    for contribution in contributions:
        grouped.setdefault(
            contribution.target_attractor_id,
            [],
        ).append(
            contribution
        )

    total_bias = sum(
        contribution.weighted_bias
        for contribution in contributions
    )

    aggregates = []

    for attractor_id in sorted(
        grouped
    ):
        items = tuple(
            grouped[
                attractor_id
            ]
        )

        total_weighted_bias = sum(
            item.weighted_bias
            for item in items
        )

        normalized = (
            total_weighted_bias
            / total_bias
            if total_bias > 0.0
            else 0.0
        )

        count = len(
            items
        )

        aggregates.append(
            AttractorContextAggregate(
                attractor_id=attractor_id,
                contributor_count=count,
                total_weighted_bias=(
                    total_weighted_bias
                ),
                normalized_context_weight=(
                    normalized
                ),
                mean_response_strength=(
                    sum(
                        item.conditioned_response_strength
                        for item in items
                    )
                    / count
                ),
                mean_generalization_weight=(
                    sum(
                        item.generalization_weight
                        for item in items
                    )
                    / count
                ),
                mean_association_confidence=(
                    sum(
                        item.association_confidence
                        for item in items
                    )
                    / count
                ),
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "aggregate_is_descriptive": True,
                    "memory_mutated": False,
                    "learning_applied": False,
                    "biological_context_claimed": False,
                    "causal_truth_inferred": False,
                },
            )
        )

    return tuple(
        aggregates
    )


def _dominant_aggregate(
    aggregates: Sequence[
        AttractorContextAggregate
    ],
) -> AttractorContextAggregate:
    ordered = tuple(
        sorted(
            aggregates,
            key=lambda item: (
                -item.normalized_context_weight,
                -item.total_weighted_bias,
                item.attractor_id,
            ),
        )
    )

    return ordered[
        0
    ]


def _competition_margin(
    aggregates: Sequence[
        AttractorContextAggregate
    ],
) -> float:
    weights = sorted(
        (
            aggregate.normalized_context_weight
            for aggregate in aggregates
        ),
        reverse=True,
    )

    if not weights:
        return 0.0

    if len(
        weights
    ) == 1:
        return weights[
            0
        ]

    return max(
        0.0,
        min(
            1.0,
            weights[
                0
            ]
            - weights[
                1
            ],
        ),
    )


def _context_confidence(
    aggregates: Sequence[
        AttractorContextAggregate
    ],
    competition_margin: float,
) -> float:
    dominant = _dominant_aggregate(
        aggregates
    )

    return max(
        0.0,
        min(
            1.0,
            (
                dominant.normalized_context_weight
                + dominant.mean_association_confidence
                + competition_margin
            )
            / 3.0,
        ),
    )


def build_associative_context(
    *,
    context_id: str,
    cue_inputs: Iterable[
        AssociativeCueInput
    ],
    prestress_config: ConditionedPrestressConfig | None = None,
    generalization_scale: float = 1.0,
    max_total_bias: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> AssociativeContext:
    items = tuple(
        cue_inputs
    )

    if not items:
        raise AssociativeContextError(
            "cue_inputs must not be empty"
        )

    max_total_bias = _validate_nonnegative(
        "max_total_bias",
        max_total_bias,
    )

    contributions = tuple(
        _build_contribution(
            cue_input=item,
            prestress_config=prestress_config,
            generalization_scale=generalization_scale,
        )
        for item in items
    )

    aggregates = _aggregate_contributions(
        contributions
    )

    dominant = _dominant_aggregate(
        aggregates
    )

    margin = _competition_margin(
        aggregates
    )

    confidence = _context_confidence(
        aggregates,
        margin,
    )

    raw_total = sum(
        aggregate.total_weighted_bias
        for aggregate in aggregates
    )

    if raw_total <= 0.0:
        bias_map = {
            aggregate.attractor_id: 0.0
            for aggregate in aggregates
        }
    else:
        scale = min(
            1.0,
            (
                max_total_bias
                / raw_total
                if raw_total > 0.0
                else 1.0
            ),
        )

        bias_map = {
            aggregate.attractor_id: (
                aggregate.total_weighted_bias
                * scale
            )
            for aggregate in aggregates
        }

    prestress = AttractorPrestress(
        prestress_id=(
            f"{context_id}::prestress"
        ),
        bias_by_attractor_id=bias_map,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "source": "associative_context",
            "context_id": context_id,
            "memory_mutated": False,
            "learning_applied": False,
            "action_selected": False,
            "policy_modified": False,
            "biological_context_claimed": False,
            "causal_truth_inferred": False,
        },
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "associative_context_mode": "descriptive",
        "source_cue_count": len(
            items
        ),
        "memory_mutated": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_context_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return AssociativeContext(
        context_id=context_id,
        contributions=contributions,
        aggregates=aggregates,
        prestress=prestress,
        dominant_attractor_id=(
            dominant.attractor_id
        ),
        context_confidence=confidence,
        competition_margin=margin,
        metadata=merged_metadata,
    )


def associative_context_is_policy_free(
    context: AssociativeContext,
) -> bool:
    return (
        context.metadata.get(
            "action_selected"
        )
        is False
        and context.metadata.get(
            "policy_modified"
        )
        is False
        and context.metadata.get(
            "memory_mutated"
        )
        is False
        and context.metadata.get(
            "learning_applied"
        )
        is False
    )


def contribution_by_cue_id(
    context: AssociativeContext,
    cue_id: str,
) -> AssociativeCueContribution:
    target = _validate_id(
        "cue_id",
        cue_id,
    )

    for contribution in context.contributions:
        if contribution.cue_id == target:
            return contribution

    raise AssociativeContextError(
        f"unknown cue_id: {target}"
    )


def aggregate_by_attractor_id(
    context: AssociativeContext,
    attractor_id: str,
) -> AttractorContextAggregate:
    target = _validate_id(
        "attractor_id",
        attractor_id,
    )

    for aggregate in context.aggregates:
        if aggregate.attractor_id == target:
            return aggregate

    raise AssociativeContextError(
        f"unknown attractor_id: {target}"
    )


def context_attractor_ids(
    context: AssociativeContext,
) -> tuple[str, ...]:
    return tuple(
        aggregate.attractor_id
        for aggregate in context.aggregates
    )


def context_weight_series(
    context: AssociativeContext,
) -> tuple[float, ...]:
    return tuple(
        aggregate.normalized_context_weight
        for aggregate in context.aggregates
    )


def contribution_weight_series(
    context: AssociativeContext,
) -> tuple[float, ...]:
    return tuple(
        contribution.weighted_bias
        for contribution in context.contributions
    )


__all__ = [
    "AssociativeContext",
    "AssociativeContextError",
    "AssociativeCueContribution",
    "AssociativeCueInput",
    "AttractorContextAggregate",
    "SCHEMA_VERSION",
    "aggregate_by_attractor_id",
    "associative_context_is_policy_free",
    "build_associative_context",
    "context_attractor_ids",
    "context_weight_series",
    "contribution_by_cue_id",
    "contribution_weight_series",
]
