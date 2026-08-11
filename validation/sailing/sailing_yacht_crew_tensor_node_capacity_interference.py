"""
ROIF External-Domain Validation
Scenario 04J — Tensor Node Capacity / Interference Benchmark

Purpose
-------
Measure how many structural memory images one shared tensor-like node can
multiplex before response ambiguity and interference rise.

This benchmark extends 04I:

    one physical node
        + many structural image prototypes
        + anisotropic prestressed response
        -> capacity / interference curve

Default image counts:

    2, 4, 8, 16, 32, 64

For each capacity point we measure:

    query_count
    unique_response_profile_count
    unique_profile_fraction
    unique_dominant_image_count
    dominant_coverage_fraction
    mean_response_energy
    mean_top1_activation
    mean_top2_activation
    mean_dominant_margin
    mean_ambiguity
    high_ambiguity_fraction
    mean_interference_ratio
    collision_count
    collision_fraction

Definitions
-----------
dominant margin:
    top1_activation - top2_activation

ambiguity:
    top2_activation / top1_activation
    bounded to [0, 1] when top1 > 0

interference ratio:
    non-dominant activation mass / total activation mass

collision:
    two or more query responses collapse to the same rounded response profile

Architectural boundaries
------------------------

    Capacity Benchmark != Learning
    Capacity Benchmark != Action Selection
    Capacity Benchmark != Policy Override
    Structural Image ID != Retrieval Ground Truth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from validation.sailing.sailing_yacht_crew_tensor_node_multiplex_memory import (
    DEFAULT_HISTORY_PER_IMAGE,
    TensorNodeMultiplexResult,
    dominant_image_ids,
    multiplex_node_is_policy_free,
    response_energies,
    response_profiles,
    run_tensor_node_multiplex_benchmark,
    unique_response_profile_count,
)


SCENARIO_ID = "sailing_04J_tensor_node_capacity_interference"

DEFAULT_IMAGE_COUNTS = (
    2,
    4,
    8,
    16,
    32,
    64,
)

DEFAULT_QUERY_REPETITIONS = 1
DEFAULT_PROFILE_PRECISION = 10
DEFAULT_HIGH_AMBIGUITY_THRESHOLD = 0.80


class TensorNodeCapacityError(RuntimeError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _validate_unit_interval(
    name: str,
    value: float,
) -> None:
    if (
        not isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise TensorNodeCapacityError(
            f"{name} must be finite and within [0, 1]"
        )


def normalize_image_counts(
    image_counts: Iterable[int],
) -> tuple[int, ...]:
    values = tuple(
        int(value)
        for value in image_counts
    )

    if not values:
        raise TensorNodeCapacityError(
            "image_counts must not be empty"
        )

    if any(
        value < 2
        for value in values
    ):
        raise TensorNodeCapacityError(
            "all image counts must be >= 2"
        )

    if any(
        right <= left
        for left, right in zip(
            values,
            values[
                1:
            ],
        )
    ):
        raise TensorNodeCapacityError(
            "image_counts must be strictly increasing"
        )

    return values


@dataclass(frozen=True, slots=True)
class TensorNodeCapacityCheckpoint:
    image_count: int
    history_per_image: int
    query_repetitions: int

    source_record_count: int
    query_count: int

    unique_response_profile_count: int
    unique_profile_fraction: float

    unique_dominant_image_count: int
    dominant_coverage_fraction: float

    mean_response_energy: float
    mean_top1_activation: float
    mean_top2_activation: float
    mean_dominant_margin: float
    mean_ambiguity: float
    high_ambiguity_fraction: float
    mean_interference_ratio: float

    collision_count: int
    collision_fraction: float

    same_physical_node: bool
    policy_free: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.image_count < 2:
            raise TensorNodeCapacityError(
                "image_count must be >= 2"
            )

        if self.history_per_image < 2:
            raise TensorNodeCapacityError(
                "history_per_image must be >= 2"
            )

        if self.query_repetitions < 1:
            raise TensorNodeCapacityError(
                "query_repetitions must be >= 1"
            )

        for name in (
            "source_record_count",
            "query_count",
            "unique_response_profile_count",
            "unique_dominant_image_count",
            "collision_count",
        ):
            if getattr(
                self,
                name,
            ) < 0:
                raise TensorNodeCapacityError(
                    f"{name} must be non-negative"
                )

        for name in (
            "unique_profile_fraction",
            "dominant_coverage_fraction",
            "mean_ambiguity",
            "high_ambiguity_fraction",
            "mean_interference_ratio",
            "collision_fraction",
        ):
            _validate_unit_interval(
                name,
                float(
                    getattr(
                        self,
                        name,
                    )
                ),
            )

        for name in (
            "mean_response_energy",
            "mean_top1_activation",
            "mean_top2_activation",
            "mean_dominant_margin",
        ):
            value = float(
                getattr(
                    self,
                    name,
                )
            )
            if (
                not isfinite(value)
                or value < 0.0
            ):
                raise TensorNodeCapacityError(
                    f"{name} must be finite and non-negative"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class TensorNodeCapacitySummary:
    first_image_count: int
    last_image_count: int

    first_unique_profile_fraction: float
    last_unique_profile_fraction: float

    first_mean_ambiguity: float
    last_mean_ambiguity: float

    first_mean_interference_ratio: float
    last_mean_interference_ratio: float

    first_mean_dominant_margin: float
    last_mean_dominant_margin: float

    first_collision_fraction: float
    last_collision_fraction: float

    ambiguity_increased: bool
    interference_increased: bool
    dominant_margin_decreased: bool
    collisions_increased: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.first_image_count < 2:
            raise TensorNodeCapacityError(
                "first_image_count must be >= 2"
            )

        if self.last_image_count < self.first_image_count:
            raise TensorNodeCapacityError(
                "last_image_count must be >= first_image_count"
            )

        for name in (
            "first_unique_profile_fraction",
            "last_unique_profile_fraction",
            "first_mean_ambiguity",
            "last_mean_ambiguity",
            "first_mean_interference_ratio",
            "last_mean_interference_ratio",
            "first_collision_fraction",
            "last_collision_fraction",
        ):
            _validate_unit_interval(
                name,
                float(
                    getattr(
                        self,
                        name,
                    )
                ),
            )

        for name in (
            "first_mean_dominant_margin",
            "last_mean_dominant_margin",
        ):
            value = float(
                getattr(
                    self,
                    name,
                )
            )
            if (
                not isfinite(value)
                or value < 0.0
            ):
                raise TensorNodeCapacityError(
                    f"{name} must be finite and non-negative"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class TensorNodeCapacityResult:
    scenario_id: str

    checkpoints: tuple[
        TensorNodeCapacityCheckpoint,
        ...,
    ]

    summary: TensorNodeCapacitySummary

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        checkpoints = tuple(
            self.checkpoints
        )

        if not checkpoints:
            raise TensorNodeCapacityError(
                "at least one checkpoint is required"
            )

        object.__setattr__(
            self,
            "checkpoints",
            checkpoints,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _activation_statistics(
    result: TensorNodeMultiplexResult,
    *,
    high_ambiguity_threshold: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    top1_values = []
    top2_values = []
    margins = []
    ambiguities = []
    interference_ratios = []

    for response in result.responses:
        activations = tuple(
            float(value)
            for _, value
            in response.image_response_profile
        )

        if not activations:
            top1 = 0.0
            top2 = 0.0
            total = 0.0
        else:
            ordered = sorted(
                activations,
                reverse=True,
            )
            top1 = ordered[
                0
            ]
            top2 = (
                ordered[
                    1
                ]
                if len(
                    ordered
                ) > 1
                else 0.0
            )
            total = sum(
                ordered
            )

        margin = max(
            0.0,
            top1 - top2,
        )

        ambiguity = (
            min(
                1.0,
                _safe_ratio(
                    top2,
                    top1,
                ),
            )
            if top1 > 0.0
            else 0.0
        )

        interference = (
            min(
                1.0,
                _safe_ratio(
                    max(
                        0.0,
                        total - top1,
                    ),
                    total,
                ),
            )
            if total > 0.0
            else 0.0
        )

        top1_values.append(
            top1
        )
        top2_values.append(
            top2
        )
        margins.append(
            margin
        )
        ambiguities.append(
            ambiguity
        )
        interference_ratios.append(
            interference
        )

    high_ambiguity_fraction = _safe_ratio(
        sum(
            1
            for value
            in ambiguities
            if value >= high_ambiguity_threshold
        ),
        len(
            ambiguities
        ),
    )

    return (
        mean(
            top1_values
        )
        if top1_values
        else 0.0,
        mean(
            top2_values
        )
        if top2_values
        else 0.0,
        mean(
            margins
        )
        if margins
        else 0.0,
        mean(
            ambiguities
        )
        if ambiguities
        else 0.0,
        high_ambiguity_fraction,
        mean(
            interference_ratios
        )
        if interference_ratios
        else 0.0,
    )


def build_capacity_checkpoint(
    *,
    image_count: int,
    history_per_image: int = DEFAULT_HISTORY_PER_IMAGE,
    query_repetitions: int = DEFAULT_QUERY_REPETITIONS,
    profile_precision: int = DEFAULT_PROFILE_PRECISION,
    high_ambiguity_threshold: float = DEFAULT_HIGH_AMBIGUITY_THRESHOLD,
) -> TensorNodeCapacityCheckpoint:
    if image_count < 2:
        raise TensorNodeCapacityError(
            "image_count must be >= 2"
        )

    if history_per_image < 2:
        raise TensorNodeCapacityError(
            "history_per_image must be >= 2"
        )

    if query_repetitions < 1:
        raise TensorNodeCapacityError(
            "query_repetitions must be >= 1"
        )

    if profile_precision < 0:
        raise TensorNodeCapacityError(
            "profile_precision must be non-negative"
        )

    _validate_unit_interval(
        "high_ambiguity_threshold",
        float(
            high_ambiguity_threshold
        ),
    )

    result = run_tensor_node_multiplex_benchmark(
        history_per_image=history_per_image,
        image_count=image_count,
        query_repetitions=query_repetitions,
    )

    unique_profiles = unique_response_profile_count(
        result,
        precision=profile_precision,
    )

    query_count = len(
        result.responses
    )

    unique_dominants = len(
        {
            image_id
            for image_id
            in dominant_image_ids(
                result
            )
            if image_id is not None
        }
    )

    (
        mean_top1,
        mean_top2,
        mean_margin,
        mean_ambiguity,
        high_ambiguity_fraction,
        mean_interference,
    ) = _activation_statistics(
        result,
        high_ambiguity_threshold=high_ambiguity_threshold,
    )

    collision_count = max(
        0,
        query_count
        - unique_profiles,
    )

    return TensorNodeCapacityCheckpoint(
        image_count=image_count,
        history_per_image=history_per_image,
        query_repetitions=query_repetitions,
        source_record_count=(
            result.node.source_record_count
        ),
        query_count=query_count,
        unique_response_profile_count=(
            unique_profiles
        ),
        unique_profile_fraction=_safe_ratio(
            unique_profiles,
            query_count,
        ),
        unique_dominant_image_count=(
            unique_dominants
        ),
        dominant_coverage_fraction=_safe_ratio(
            unique_dominants,
            image_count,
        ),
        mean_response_energy=(
            mean(
                response_energies(
                    result
                )
            )
            if result.responses
            else 0.0
        ),
        mean_top1_activation=mean_top1,
        mean_top2_activation=mean_top2,
        mean_dominant_margin=mean_margin,
        mean_ambiguity=mean_ambiguity,
        high_ambiguity_fraction=(
            high_ambiguity_fraction
        ),
        mean_interference_ratio=(
            mean_interference
        ),
        collision_count=collision_count,
        collision_fraction=_safe_ratio(
            collision_count,
            query_count,
        ),
        same_physical_node=bool(
            result.metadata[
                "all_queries_use_same_physical_node"
            ]
        ),
        policy_free=multiplex_node_is_policy_free(
            result
        ),
        metadata={
            "scenario_id": SCENARIO_ID,
            "source_scenario_id": result.scenario_id,
            "shared_node_id": result.node.node_id,
            "profile_precision": profile_precision,
            "high_ambiguity_threshold": high_ambiguity_threshold,
            "response_derived_from_structural_geometry": True,
            "memory_selects_action": False,
            "policy_override_enabled": False,
            "external_expected_label_used": False,
        },
    )


def summarize_capacity(
    checkpoints: Iterable[
        TensorNodeCapacityCheckpoint
    ],
) -> TensorNodeCapacitySummary:
    values = tuple(
        checkpoints
    )

    if not values:
        raise TensorNodeCapacityError(
            "cannot summarize zero checkpoints"
        )

    first = values[
        0
    ]

    last = values[
        -1
    ]

    return TensorNodeCapacitySummary(
        first_image_count=(
            first.image_count
        ),
        last_image_count=(
            last.image_count
        ),
        first_unique_profile_fraction=(
            first.unique_profile_fraction
        ),
        last_unique_profile_fraction=(
            last.unique_profile_fraction
        ),
        first_mean_ambiguity=(
            first.mean_ambiguity
        ),
        last_mean_ambiguity=(
            last.mean_ambiguity
        ),
        first_mean_interference_ratio=(
            first.mean_interference_ratio
        ),
        last_mean_interference_ratio=(
            last.mean_interference_ratio
        ),
        first_mean_dominant_margin=(
            first.mean_dominant_margin
        ),
        last_mean_dominant_margin=(
            last.mean_dominant_margin
        ),
        first_collision_fraction=(
            first.collision_fraction
        ),
        last_collision_fraction=(
            last.collision_fraction
        ),
        ambiguity_increased=(
            last.mean_ambiguity
            > first.mean_ambiguity
        ),
        interference_increased=(
            last.mean_interference_ratio
            > first.mean_interference_ratio
        ),
        dominant_margin_decreased=(
            last.mean_dominant_margin
            < first.mean_dominant_margin
        ),
        collisions_increased=(
            last.collision_fraction
            > first.collision_fraction
        ),
        metadata={
            "summary_method": "first_last_capacity_comparison",
            "checkpoint_count": len(
                values
            ),
            "external_expected_label_used": False,
        },
    )


def run_tensor_node_capacity_benchmark(
    *,
    image_counts: Iterable[int] = DEFAULT_IMAGE_COUNTS,
    history_per_image: int = DEFAULT_HISTORY_PER_IMAGE,
    query_repetitions: int = DEFAULT_QUERY_REPETITIONS,
    profile_precision: int = DEFAULT_PROFILE_PRECISION,
    high_ambiguity_threshold: float = DEFAULT_HIGH_AMBIGUITY_THRESHOLD,
) -> TensorNodeCapacityResult:
    counts = normalize_image_counts(
        image_counts
    )

    checkpoints = tuple(
        build_capacity_checkpoint(
            image_count=image_count,
            history_per_image=history_per_image,
            query_repetitions=query_repetitions,
            profile_precision=profile_precision,
            high_ambiguity_threshold=high_ambiguity_threshold,
        )
        for image_count
        in counts
    )

    summary = summarize_capacity(
        checkpoints
    )

    return TensorNodeCapacityResult(
        scenario_id=SCENARIO_ID,
        checkpoints=checkpoints,
        summary=summary,
        metadata={
            "benchmark_type": "tensor_node_capacity_interference",
            "checkpoint_count": len(
                checkpoints
            ),
            "single_shared_physical_node_per_checkpoint": True,
            "response_derived_from_structural_geometry": True,
            "memory_selects_action": False,
            "policy_override_enabled": False,
            "external_expected_label_used": False,
        },
    )


def image_count_series(
    result: TensorNodeCapacityResult,
) -> tuple[int, ...]:
    return tuple(
        item.image_count
        for item in result.checkpoints
    )


def unique_profile_fraction_series(
    result: TensorNodeCapacityResult,
) -> tuple[float, ...]:
    return tuple(
        item.unique_profile_fraction
        for item in result.checkpoints
    )


def dominant_coverage_series(
    result: TensorNodeCapacityResult,
) -> tuple[float, ...]:
    return tuple(
        item.dominant_coverage_fraction
        for item in result.checkpoints
    )


def mean_ambiguity_series(
    result: TensorNodeCapacityResult,
) -> tuple[float, ...]:
    return tuple(
        item.mean_ambiguity
        for item in result.checkpoints
    )


def high_ambiguity_fraction_series(
    result: TensorNodeCapacityResult,
) -> tuple[float, ...]:
    return tuple(
        item.high_ambiguity_fraction
        for item in result.checkpoints
    )


def mean_interference_series(
    result: TensorNodeCapacityResult,
) -> tuple[float, ...]:
    return tuple(
        item.mean_interference_ratio
        for item in result.checkpoints
    )


def dominant_margin_series(
    result: TensorNodeCapacityResult,
) -> tuple[float, ...]:
    return tuple(
        item.mean_dominant_margin
        for item in result.checkpoints
    )


def collision_fraction_series(
    result: TensorNodeCapacityResult,
) -> tuple[float, ...]:
    return tuple(
        item.collision_fraction
        for item in result.checkpoints
    )


__all__ = [
    "DEFAULT_HIGH_AMBIGUITY_THRESHOLD",
    "DEFAULT_IMAGE_COUNTS",
    "DEFAULT_PROFILE_PRECISION",
    "DEFAULT_QUERY_REPETITIONS",
    "SCENARIO_ID",
    "TensorNodeCapacityCheckpoint",
    "TensorNodeCapacityError",
    "TensorNodeCapacityResult",
    "TensorNodeCapacitySummary",
    "build_capacity_checkpoint",
    "collision_fraction_series",
    "dominant_coverage_series",
    "dominant_margin_series",
    "high_ambiguity_fraction_series",
    "image_count_series",
    "mean_ambiguity_series",
    "mean_interference_series",
    "normalize_image_counts",
    "run_tensor_node_capacity_benchmark",
    "summarize_capacity",
    "unique_profile_fraction_series",
]
