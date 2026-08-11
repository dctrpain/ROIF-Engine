"""
ROIF Memory 2.0
Conditioned Dynamics Core

Ninth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor
    AttractorDynamics
    AttractorTrajectory
    ExperienceConditioning
    ConditionedPrestress

Purpose
-------
Close the descriptive loop:

    query cue
        -> conditioned response
        -> conditioned prestress
        -> attractor competition
        -> state_after

This module explicitly compares:
- baseline attractor dynamics without conditioned prestress
- conditioned attractor dynamics with conditioned prestress

It measures:
- competition-weight deltas
- dominant-attractor change
- capture-margin change
- entropy-proxy change
- state-after divergence

This module does NOT:
- learn
- mutate source conditioning memory
- mutate source attractors
- select actions
- modify policy
- diagnose
- claim biological conditioned dynamics
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


SCHEMA_VERSION = "conditioned_dynamics_v1"


class ConditionedDynamicsError(ValueError):
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
        raise ConditionedDynamicsError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise ConditionedDynamicsError(
            f"{name} must be finite"
        )

    return x


def _vector_distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise ConditionedDynamicsError(
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
class CompetitionDelta:
    attractor_id: str
    baseline_weight: float
    conditioned_weight: float
    weight_delta: float

    baseline_capture_strength: float
    conditioned_capture_strength: float
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
            "conditioned_weight",
            "weight_delta",
            "baseline_capture_strength",
            "conditioned_capture_strength",
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
            "conditioned_weight",
            "baseline_capture_strength",
            "conditioned_capture_strength",
        ):
            value = getattr(
                self,
                name,
            )

            if not 0.0 <= value <= 1.0:
                raise ConditionedDynamicsError(
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
class ConditionedDynamicsResult:
    result_id: str

    query_cue: ConditioningCue
    state_before: DynamicsState

    conditioned_prestress: ConditionedPrestressResult

    baseline: AttractorDynamicsResult
    conditioned: AttractorDynamicsResult

    competition_deltas: tuple[
        CompetitionDelta,
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
            raise ConditionedDynamicsError(
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
            raise ConditionedDynamicsError(
                "state_after_distance must be non-negative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _competition_deltas(
    *,
    baseline: AttractorDynamicsResult,
    conditioned: AttractorDynamicsResult,
) -> tuple[
    CompetitionDelta,
    ...,
]:
    baseline_ids = {
        activation.attractor_id
        for activation in baseline.activations
    }

    conditioned_ids = {
        activation.attractor_id
        for activation in conditioned.activations
    }

    if baseline_ids != conditioned_ids:
        raise ConditionedDynamicsError(
            "baseline and conditioned attractor sets must match"
        )

    deltas = []

    for attractor_id in sorted(
        baseline_ids
    ):
        baseline_activation = activation_by_id(
            baseline,
            attractor_id,
        )

        conditioned_activation = activation_by_id(
            conditioned,
            attractor_id,
        )

        deltas.append(
            CompetitionDelta(
                attractor_id=attractor_id,
                baseline_weight=(
                    baseline_activation.competition_weight
                ),
                conditioned_weight=(
                    conditioned_activation.competition_weight
                ),
                weight_delta=(
                    conditioned_activation.competition_weight
                    - baseline_activation.competition_weight
                ),
                baseline_capture_strength=(
                    baseline_activation.capture_strength
                ),
                conditioned_capture_strength=(
                    conditioned_activation.capture_strength
                ),
                capture_strength_delta=(
                    conditioned_activation.capture_strength
                    - baseline_activation.capture_strength
                ),
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "descriptive_comparison": True,
                    "memory_mutated": False,
                    "learning_applied": False,
                    "action_selected": False,
                    "policy_modified": False,
                    "biological_conditioning_claimed": False,
                    "causal_truth_inferred": False,
                },
            )
        )

    return tuple(
        deltas
    )


def run_conditioned_dynamics(
    *,
    result_id: str,
    state: DynamicsState,
    attractors: Iterable[
        ExperienceAttractor
    ],
    query_cue: ConditioningCue,
    conditioning_memory: ConditioningMemory,
    prestress_config: ConditionedPrestressConfig | None = None,
    generalization_scale: float = 1.0,
    distance_scale: float = 1.0,
    step_size: float = 0.25,
    metadata: Mapping[str, Any] | None = None,
) -> ConditionedDynamicsResult:
    attractor_items = tuple(
        attractors
    )

    if not attractor_items:
        raise ConditionedDynamicsError(
            "attractors must not be empty"
        )

    if not conditioning_memory_is_policy_free(
        conditioning_memory
    ):
        raise ConditionedDynamicsError(
            "conditioning memory must be policy-free"
        )

    conditioned_prestress = build_conditioned_prestress(
        prestress_id=(
            f"{result_id}::conditioned_prestress"
        ),
        query_cue=query_cue,
        memory=conditioning_memory,
        config=prestress_config,
        generalization_scale=generalization_scale,
    )

    if not conditioned_prestress_is_policy_free(
        conditioned_prestress
    ):
        raise ConditionedDynamicsError(
            "conditioned prestress must be policy-free"
        )

    baseline = run_attractor_dynamics(
        state=state,
        attractors=attractor_items,
        prestress=None,
        distance_scale=distance_scale,
        step_size=step_size,
    )

    conditioned = run_attractor_dynamics(
        state=state,
        attractors=attractor_items,
        prestress=(
            conditioned_prestress.prestress
        ),
        distance_scale=distance_scale,
        step_size=step_size,
    )

    if not dynamics_is_policy_free(
        baseline
    ):
        raise ConditionedDynamicsError(
            "baseline dynamics must be policy-free"
        )

    if not dynamics_is_policy_free(
        conditioned
    ):
        raise ConditionedDynamicsError(
            "conditioned dynamics must be policy-free"
        )

    deltas = _competition_deltas(
        baseline=baseline,
        conditioned=conditioned,
    )

    state_after_distance = _vector_distance(
        baseline.state_after.feature_vector,
        conditioned.state_after.feature_vector,
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "conditioned_dynamics_mode": "descriptive",
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

    return ConditionedDynamicsResult(
        result_id=result_id,
        query_cue=query_cue,
        state_before=state,
        conditioned_prestress=(
            conditioned_prestress
        ),
        baseline=baseline,
        conditioned=conditioned,
        competition_deltas=deltas,
        dominant_attractor_changed=(
            baseline.dominant_attractor_id
            != conditioned.dominant_attractor_id
        ),
        capture_margin_delta=(
            conditioned.capture_margin
            - baseline.capture_margin
        ),
        entropy_proxy_delta=(
            conditioned.competition_entropy_proxy
            - baseline.competition_entropy_proxy
        ),
        state_after_distance=(
            state_after_distance
        ),
        metadata=merged_metadata,
    )


def competition_delta_by_id(
    result: ConditionedDynamicsResult,
    attractor_id: str,
) -> CompetitionDelta:
    target = _validate_id(
        "attractor_id",
        attractor_id,
    )

    for delta in result.competition_deltas:
        if delta.attractor_id == target:
            return delta

    raise ConditionedDynamicsError(
        f"unknown attractor_id: {target}"
    )


def conditioned_target_weight_gain(
    result: ConditionedDynamicsResult,
) -> float:
    target_id = tuple(
        result.conditioned_prestress.prestress.bias_by_attractor_id.keys()
    )

    if len(
        target_id
    ) != 1:
        raise ConditionedDynamicsError(
            "conditioned prestress must target exactly one attractor"
        )

    delta = competition_delta_by_id(
        result,
        target_id[
            0
        ],
    )

    return delta.weight_delta


def conditioned_target_capture_gain(
    result: ConditionedDynamicsResult,
) -> float:
    target_id = tuple(
        result.conditioned_prestress.prestress.bias_by_attractor_id.keys()
    )

    if len(
        target_id
    ) != 1:
        raise ConditionedDynamicsError(
            "conditioned prestress must target exactly one attractor"
        )

    delta = competition_delta_by_id(
        result,
        target_id[
            0
        ],
    )

    return delta.capture_strength_delta


def conditioned_dynamics_is_policy_free(
    result: ConditionedDynamicsResult,
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
        and dynamics_is_policy_free(
            result.baseline
        )
        and dynamics_is_policy_free(
            result.conditioned
        )
        and conditioned_prestress_is_policy_free(
            result.conditioned_prestress
        )
    )


def competition_weight_delta_series(
    result: ConditionedDynamicsResult,
) -> tuple[float, ...]:
    return tuple(
        delta.weight_delta
        for delta in result.competition_deltas
    )


def capture_strength_delta_series(
    result: ConditionedDynamicsResult,
) -> tuple[float, ...]:
    return tuple(
        delta.capture_strength_delta
        for delta in result.competition_deltas
    )


__all__ = [
    "CompetitionDelta",
    "ConditionedDynamicsError",
    "ConditionedDynamicsResult",
    "SCHEMA_VERSION",
    "capture_strength_delta_series",
    "competition_delta_by_id",
    "competition_weight_delta_series",
    "conditioned_dynamics_is_policy_free",
    "conditioned_target_capture_gain",
    "conditioned_target_weight_gain",
    "run_conditioned_dynamics",
]
