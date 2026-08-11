"""
ROIF Predictive Preload
=======================

Purpose
-------
Provide the first explicit, auditable bridge from stored controller memory
into a future predictive-control cycle.

Input:
    PredictiveState
    relevant MemoryScar(s) / hierarchical memory context

Output:
    PredictivePreload
    preloaded PredictiveState

Architectural boundaries
------------------------

    Memory Retrieval != Action Selection
    PredictivePreload != Policy Override
    MemoryScar != Direct State Mutation
    Hierarchical Context != Ground Truth

The preload layer may:
- bias the predicted initial expectation using previously observed systematic
  prediction error;
- reduce uncertainty when repeated memory is consistent and confident;
- combine local and parent-level memory with explicit weights;
- expose all applied contributions in immutable audit metadata.

The preload layer must NOT:
- directly choose an action;
- change candidate ranking by itself;
- modify accepted causal graph structure;
- overwrite observed values;
- use evaluator labels as hidden truth;
- perform reinforcement learning.

Matryoshka / nested-memory semantics
------------------------------------
Memory may exist at multiple levels:

    raw trace
        ↓
    local cluster / MemoryScar
        ↓
    parent cluster
        ↓
    higher-order context

More local memory should normally receive greater weight.
Higher-order memory acts as a broader prior, not as a replacement for local
experience.

The current module accepts an ordered sequence of MemoryScar contributions,
each with an explicit hierarchy level and weight. A later retrieval layer may
derive those scars automatically from HierarchicalMemory nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.controller_memory import (
    MemoryScar,
    scar_is_systematic,
)
from roif.predictive_control import (
    PredictiveState,
)


# =============================================================================
# Errors
# =============================================================================


class PredictivePreloadError(RuntimeError):
    """Base predictive-preload error."""


class InvalidPredictivePreloadError(PredictivePreloadError):
    """Raised when preload structure is inconsistent."""


# =============================================================================
# Helpers
# =============================================================================


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {}
        if value is None
        else dict(value)
    )


def _finite(
    value: float,
    *,
    name: str,
) -> float:
    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


def _clamp01(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class PredictivePreloadConfig:
    """
    Conservative memory-to-prediction coupling.

    max_bias_fraction:
        Maximum fraction of a MemoryScar bias vector allowed into preload.

    uncertainty_reduction_scale:
        How strongly confident/systematic memory may reduce uncertainty.

    minimum_scar_confidence:
        Ignore scars below this confidence.

    minimum_scar_consistency:
        Ignore scars below this directional consistency.

    require_systematic_scar:
        If True, only systematic scars can contribute.

    local_level_decay:
        Weight multiplier applied per hierarchy level:
            level 0 -> 1.0
            level 1 -> decay
            level 2 -> decay^2
        so local memory dominates parent memory.
    """

    max_bias_fraction: float = 0.75
    uncertainty_reduction_scale: float = 0.35
    minimum_scar_confidence: float = 0.25
    minimum_scar_consistency: float = 0.80
    require_systematic_scar: bool = True
    local_level_decay: float = 0.50

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_bias_fraction",
            _clamp01(
                self.max_bias_fraction
            ),
        )

        object.__setattr__(
            self,
            "uncertainty_reduction_scale",
            _clamp01(
                self.uncertainty_reduction_scale
            ),
        )

        object.__setattr__(
            self,
            "minimum_scar_confidence",
            _clamp01(
                self.minimum_scar_confidence
            ),
        )

        object.__setattr__(
            self,
            "minimum_scar_consistency",
            _clamp01(
                self.minimum_scar_consistency
            ),
        )

        object.__setattr__(
            self,
            "local_level_decay",
            _clamp01(
                self.local_level_decay
            ),
        )


# =============================================================================
# Nested memory contribution
# =============================================================================


@dataclass(frozen=True, slots=True)
class MemoryPreloadContribution:
    """
    One memory contribution to a future predictive preload.

    hierarchy_level:
        0 = local / most specific memory
        1 = parent cluster
        2 = grandparent cluster
        ...

    explicit_weight:
        Additional caller-controlled weight before config level decay.
    """

    scar: MemoryScar
    hierarchy_level: int = 0
    explicit_weight: float = 1.0
    source_cluster_id: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.hierarchy_level < 0:
            raise InvalidPredictivePreloadError(
                "hierarchy_level must be non-negative"
            )

        weight = _finite(
            self.explicit_weight,
            name="explicit_weight",
        )

        if weight < 0.0:
            raise InvalidPredictivePreloadError(
                "explicit_weight must be non-negative"
            )

        object.__setattr__(
            self,
            "explicit_weight",
            weight,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AppliedPreloadContribution:
    scar_id: str
    hierarchy_level: int
    effective_weight: float
    confidence: float
    consistency: float
    bias_by_variable: Mapping[str, float]
    source_cluster_id: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_weight",
            _finite(
                self.effective_weight,
                name="effective_weight",
            ),
        )

        object.__setattr__(
            self,
            "bias_by_variable",
            MappingProxyType(
                {
                    str(key): float(value)
                    for key, value
                    in self.bias_by_variable.items()
                }
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


# =============================================================================
# Preload result
# =============================================================================


@dataclass(frozen=True, slots=True)
class PredictivePreload:
    original_state: PredictiveState
    preloaded_state: PredictiveState

    applied_bias_by_variable: Mapping[str, float]
    applied_contributions: tuple[
        AppliedPreloadContribution,
        ...,
    ]

    uncertainty_before: float
    uncertainty_after: float

    used_memory: bool
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_bias_by_variable",
            MappingProxyType(
                {
                    str(key): float(value)
                    for key, value
                    in self.applied_bias_by_variable.items()
                }
            ),
        )

        object.__setattr__(
            self,
            "applied_contributions",
            tuple(
                self.applied_contributions
            ),
        )

        object.__setattr__(
            self,
            "uncertainty_before",
            _clamp01(
                self.uncertainty_before
            ),
        )

        object.__setattr__(
            self,
            "uncertainty_after",
            _clamp01(
                self.uncertainty_after
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


# =============================================================================
# Scar eligibility
# =============================================================================


def contribution_is_eligible(
    contribution: MemoryPreloadContribution,
    *,
    config: PredictivePreloadConfig,
) -> bool:
    scar = contribution.scar

    if (
        scar.confidence
        < config.minimum_scar_confidence
    ):
        return False

    if (
        scar.consistency
        < config.minimum_scar_consistency
    ):
        return False

    if (
        config.require_systematic_scar
        and not scar_is_systematic(
            scar,
            min_recurrence=2,
            min_consistency=config.minimum_scar_consistency,
            min_confidence=config.minimum_scar_confidence,
        )
    ):
        return False

    if contribution.explicit_weight <= 0.0:
        return False

    return True


def effective_contribution_weight(
    contribution: MemoryPreloadContribution,
    *,
    config: PredictivePreloadConfig,
) -> float:
    """
    Nested-memory / Matryoshka weighting.

    local level 0:
        explicit_weight * confidence * consistency

    parent level 1:
        above * local_level_decay

    level n:
        above * local_level_decay ** n
    """

    if not contribution_is_eligible(
        contribution,
        config=config,
    ):
        return 0.0

    scar = contribution.scar

    return (
        contribution.explicit_weight
        * scar.confidence
        * scar.consistency
        * (
            config.local_level_decay
            ** contribution.hierarchy_level
        )
    )


# =============================================================================
# Bias aggregation
# =============================================================================


def aggregate_memory_bias(
    contributions: Sequence[
        MemoryPreloadContribution
    ],
    *,
    config: PredictivePreloadConfig,
) -> tuple[
    Mapping[str, float],
    tuple[
        AppliedPreloadContribution,
        ...,
    ],
]:
    """
    Aggregate eligible nested memory contributions.

    Weighted average is used rather than unbounded summation so multiple parent
    levels do not explode preload magnitude.
    """

    weighted_sum: dict[
        str,
        float,
    ] = {}

    weight_sum: dict[
        str,
        float,
    ] = {}

    applied = []

    for contribution in contributions:
        weight = effective_contribution_weight(
            contribution,
            config=config,
        )

        if weight <= 0.0:
            continue

        scar = contribution.scar

        applied.append(
            AppliedPreloadContribution(
                scar_id=scar.scar_id,
                hierarchy_level=contribution.hierarchy_level,
                effective_weight=weight,
                confidence=scar.confidence,
                consistency=scar.consistency,
                bias_by_variable=scar.bias_by_variable,
                source_cluster_id=contribution.source_cluster_id,
                metadata={
                    "memory_preload_contribution": True,
                    "external_expected_label_used": False,
                },
            )
        )

        for key, value in scar.bias_by_variable.items():
            weighted_sum[
                key
            ] = (
                weighted_sum.get(
                    key,
                    0.0,
                )
                + weight
                * float(value)
            )

            weight_sum[
                key
            ] = (
                weight_sum.get(
                    key,
                    0.0,
                )
                + weight
            )

    bias = {}

    for key in sorted(
        weighted_sum
    ):
        total_weight = weight_sum[
            key
        ]

        if total_weight <= 0.0:
            continue

        raw_bias = (
            weighted_sum[
                key
            ]
            / total_weight
        )

        bias[
            key
        ] = (
            raw_bias
            * config.max_bias_fraction
        )

    return (
        MappingProxyType(
            bias
        ),
        tuple(
            applied
        ),
    )


# =============================================================================
# Uncertainty preload
# =============================================================================


def _preloaded_uncertainty(
    original_uncertainty: float,
    applied: Sequence[
        AppliedPreloadContribution
    ],
    *,
    config: PredictivePreloadConfig,
) -> float:
    if not applied:
        return _clamp01(
            original_uncertainty
        )

    total_weight = sum(
        item.effective_weight
        for item in applied
    )

    normalized_support = (
        total_weight
        / (
            1.0
            + total_weight
        )
    )

    reduction = (
        config.uncertainty_reduction_scale
        * normalized_support
    )

    return _clamp01(
        original_uncertainty
        * (
            1.0
            - reduction
        )
    )


# =============================================================================
# Apply preload
# =============================================================================


def apply_predictive_preload(
    state: PredictiveState,
    contributions: Sequence[
        MemoryPreloadContribution
    ],
    *,
    config: PredictivePreloadConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PredictivePreload:
    """
    Apply memory bias to a NEW PredictiveState.

    Semantics:
        remembered prediction error = observed - predicted

    Therefore future expectation is shifted in the same direction as the
    previously observed systematic bias:

        preloaded_value = current_value + remembered_bias

    Only variables already present in the current PredictiveState are changed.
    Memory never invents new state variables.
    """

    config = (
        config
        or PredictivePreloadConfig()
    )

    bias, applied = aggregate_memory_bias(
        contributions,
        config=config,
    )

    values = dict(
        state.values
    )

    applied_bias = {}

    for key, remembered_bias in bias.items():
        if key not in values:
            continue

        updated = (
            float(
                values[
                    key
                ]
            )
            + float(
                remembered_bias
            )
        )

        values[
            key
        ] = updated

        applied_bias[
            key
        ] = float(
            remembered_bias
        )

    uncertainty_after = _preloaded_uncertainty(
        state.uncertainty,
        applied,
        config=config,
    )

    preloaded_state = PredictiveState(
        values=values,
        target_values=dict(
            state.target_values
        ),
        timestamp=state.timestamp,
        uncertainty=uncertainty_after,
        metadata={
            **dict(
                state.metadata
            ),
            "predictive_preload_applied": bool(
                applied_bias
            ),
            "memory_contribution_count": len(
                applied
            ),
            "policy_modified": False,
            "action_selected": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )

    return PredictivePreload(
        original_state=state,
        preloaded_state=preloaded_state,
        applied_bias_by_variable=applied_bias,
        applied_contributions=applied,
        uncertainty_before=state.uncertainty,
        uncertainty_after=uncertainty_after,
        used_memory=bool(
            applied_bias
        ),
        metadata={
            **(
                {}
                if metadata is None
                else dict(
                    metadata
                )
            ),
            "predictive_preload": True,
            "memory_retrieval_applied": bool(
                applied
            ),
            "policy_modified": False,
            "action_selected": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Inspection helpers
# =============================================================================


def preload_changed_variable(
    preload: PredictivePreload,
    variable: str,
) -> bool:
    return (
        variable
        in preload.applied_bias_by_variable
    )


def preload_is_policy_free(
    preload: PredictivePreload,
) -> bool:
    return (
        preload.metadata.get(
            "policy_modified"
        )
        is False
        and preload.metadata.get(
            "action_selected"
        )
        is False
        and preload.metadata.get(
            "graph_mutated"
        )
        is False
        and preload.preloaded_state.metadata.get(
            "policy_modified"
        )
        is False
        and preload.preloaded_state.metadata.get(
            "action_selected"
        )
        is False
    )


def hierarchy_levels_used(
    preload: PredictivePreload,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                item.hierarchy_level
                for item
                in preload.applied_contributions
            }
        )
    )


# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "AppliedPreloadContribution",
    "InvalidPredictivePreloadError",
    "MemoryPreloadContribution",
    "PredictivePreload",
    "PredictivePreloadConfig",
    "PredictivePreloadError",
    "aggregate_memory_bias",
    "apply_predictive_preload",
    "contribution_is_eligible",
    "effective_contribution_weight",
    "hierarchy_levels_used",
    "preload_changed_variable",
    "preload_is_policy_free",
]
