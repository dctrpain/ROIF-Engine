"""
ROIF Memory 2.0
Contextual Dynamics Core

Twelfth layer above:
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

Purpose
-------
Apply an already-built AssociativeContext prestress to attractor dynamics and
compare it against an otherwise identical baseline run.

The core comparison is:

    same state
    same attractors
    different associative context
        -> different prestress
        -> different competition geometry
        -> possibly different dominant attractor
        -> different state_after

This module computes:
- per-attractor competition deltas
- per-attractor capture-strength deltas
- dominant-attractor change
- capture-margin delta
- entropy-proxy delta
- state-after divergence
- context-target weight/capture gain

This module does NOT:
- learn
- mutate source memories
- mutate attractors
- select actions
- modify policy
- diagnose
- claim biological contextual dynamics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_attractor import (
    ExperienceAttractor,
)
from roif.history.attractor_dynamics import (
    AttractorDynamicsResult,
    DynamicsState,
    activation_by_id,
    dynamics_is_policy_free,
    run_attractor_dynamics,
)
from roif.history.associative_context import (
    AssociativeContext,
    associative_context_is_policy_free,
)


SCHEMA_VERSION = "contextual_dynamics_v1"


class ContextualDynamicsError(ValueError):
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
        raise ContextualDynamicsError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise ContextualDynamicsError(
            f"{name} must be finite"
        )

    return x


def _vector_distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise ContextualDynamicsError(
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
class ContextCompetitionDelta:
    attractor_id: str

    baseline_weight: float
    contextual_weight: float
    weight_delta: float

    baseline_capture_strength: float
    contextual_capture_strength: float
    capture_strength_delta: float

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
            "baseline_weight",
            "contextual_weight",
            "weight_delta",
            "baseline_capture_strength",
            "contextual_capture_strength",
            "capture_strength_delta",
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
            "baseline_weight",
            "contextual_weight",
            "baseline_capture_strength",
            "contextual_capture_strength",
        ):
            value = getattr(
                self,
                name,
            )

            if not 0.0 <= value <= 1.0:
                raise ContextualDynamicsError(
                    f"{name} must be within [0,1]"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ContextualDynamicsResult:
    result_id: str
    state_before: DynamicsState
    context: AssociativeContext

    baseline: AttractorDynamicsResult
    contextual: AttractorDynamicsResult

    competition_deltas: tuple[
        ContextCompetitionDelta,
        ...,
    ]

    dominant_attractor_changed: bool

    capture_margin_delta: float
    entropy_proxy_delta: float
    state_after_distance: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_id",
            _validate_id(
                "result_id",
                self.result_id,
            ),
        )

        if not self.competition_deltas:
            raise ContextualDynamicsError(
                "competition_deltas must not be empty"
            )

        object.__setattr__(
            self,
            "competition_deltas",
            tuple(
                self.competition_deltas
            ),
        )

        for name in (
            "capture_margin_delta",
            "entropy_proxy_delta",
            "state_after_distance",
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

        if self.state_after_distance < 0.0:
            raise ContextualDynamicsError(
                "state_after_distance must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _build_competition_deltas(
    *,
    baseline: AttractorDynamicsResult,
    contextual: AttractorDynamicsResult,
) -> tuple[
    ContextCompetitionDelta,
    ...,
]:
    baseline_ids = {
        activation.attractor_id
        for activation in baseline.activations
    }

    contextual_ids = {
        activation.attractor_id
        for activation in contextual.activations
    }

    if baseline_ids != contextual_ids:
        raise ContextualDynamicsError(
            "baseline and contextual attractor sets must match"
        )

    deltas = []

    for attractor_id in sorted(
        baseline_ids
    ):
        baseline_activation = activation_by_id(
            baseline,
            attractor_id,
        )

        contextual_activation = activation_by_id(
            contextual,
            attractor_id,
        )

        deltas.append(
            ContextCompetitionDelta(
                attractor_id=attractor_id,
                baseline_weight=(
                    baseline_activation.competition_weight
                ),
                contextual_weight=(
                    contextual_activation.competition_weight
                ),
                weight_delta=(
                    contextual_activation.competition_weight
                    - baseline_activation.competition_weight
                ),
                baseline_capture_strength=(
                    baseline_activation.capture_strength
                ),
                contextual_capture_strength=(
                    contextual_activation.capture_strength
                ),
                capture_strength_delta=(
                    contextual_activation.capture_strength
                    - baseline_activation.capture_strength
                ),
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "descriptive_comparison": True,
                    "memory_mutated": False,
                    "learning_applied": False,
                    "action_selected": False,
                    "policy_modified": False,
                    "biological_context_claimed": False,
                    "causal_truth_inferred": False,
                },
            )
        )

    return tuple(
        deltas
    )


def run_contextual_dynamics(
    *,
    result_id: str,
    state: DynamicsState,
    attractors: Iterable[
        ExperienceAttractor
    ],
    context: AssociativeContext,
    distance_scale: float = 1.0,
    step_size: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> ContextualDynamicsResult:
    attractor_items = tuple(
        attractors
    )

    if not attractor_items:
        raise ContextualDynamicsError(
            "attractors must not be empty"
        )

    if not associative_context_is_policy_free(
        context
    ):
        raise ContextualDynamicsError(
            "associative context must be policy-free"
        )

    baseline = run_attractor_dynamics(
        state=state,
        attractors=attractor_items,
        prestress=None,
        distance_scale=distance_scale,
        step_size=step_size,
    )

    contextual = run_attractor_dynamics(
        state=state,
        attractors=attractor_items,
        prestress=context.prestress,
        distance_scale=distance_scale,
        step_size=step_size,
    )

    if not dynamics_is_policy_free(
        baseline
    ):
        raise ContextualDynamicsError(
            "baseline dynamics must be policy-free"
        )

    if not dynamics_is_policy_free(
        contextual
    ):
        raise ContextualDynamicsError(
            "contextual dynamics must be policy-free"
        )

    deltas = _build_competition_deltas(
        baseline=baseline,
        contextual=contextual,
    )

    state_after_distance = _vector_distance(
        baseline.state_after.feature_vector,
        contextual.state_after.feature_vector,
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "contextual_dynamics_mode": "descriptive",
        "context_id": context.context_id,
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

    return ContextualDynamicsResult(
        result_id=result_id,
        state_before=state,
        context=context,
        baseline=baseline,
        contextual=contextual,
        competition_deltas=deltas,
        dominant_attractor_changed=(
            baseline.dominant_attractor_id
            != contextual.dominant_attractor_id
        ),
        capture_margin_delta=(
            contextual.capture_margin
            - baseline.capture_margin
        ),
        entropy_proxy_delta=(
            contextual.competition_entropy_proxy
            - baseline.competition_entropy_proxy
        ),
        state_after_distance=(
            state_after_distance
        ),
        metadata=merged_metadata,
    )


def context_competition_delta_by_id(
    result: ContextualDynamicsResult,
    attractor_id: str,
) -> ContextCompetitionDelta:
    target = _validate_id(
        "attractor_id",
        attractor_id,
    )

    for delta in result.competition_deltas:
        if delta.attractor_id == target:
            return delta

    raise ContextualDynamicsError(
        f"unknown attractor_id: {target}"
    )


def context_target_weight_gain(
    result: ContextualDynamicsResult,
) -> float:
    delta = context_competition_delta_by_id(
        result,
        result.context.dominant_attractor_id,
    )

    return delta.weight_delta


def context_target_capture_gain(
    result: ContextualDynamicsResult,
) -> float:
    delta = context_competition_delta_by_id(
        result,
        result.context.dominant_attractor_id,
    )

    return delta.capture_strength_delta


def contextual_dynamics_is_policy_free(
    result: ContextualDynamicsResult,
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
        and associative_context_is_policy_free(
            result.context
        )
        and dynamics_is_policy_free(
            result.baseline
        )
        and dynamics_is_policy_free(
            result.contextual
        )
    )


def context_weight_delta_series(
    result: ContextualDynamicsResult,
) -> tuple[float, ...]:
    return tuple(
        delta.weight_delta
        for delta in result.competition_deltas
    )


def context_capture_delta_series(
    result: ContextualDynamicsResult,
) -> tuple[float, ...]:
    return tuple(
        delta.capture_strength_delta
        for delta in result.competition_deltas
    )


__all__ = [
    "ContextCompetitionDelta",
    "ContextualDynamicsError",
    "ContextualDynamicsResult",
    "SCHEMA_VERSION",
    "context_capture_delta_series",
    "context_competition_delta_by_id",
    "context_target_capture_gain",
    "context_target_weight_gain",
    "context_weight_delta_series",
    "contextual_dynamics_is_policy_free",
    "run_contextual_dynamics",
]
