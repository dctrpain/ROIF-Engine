"""
ROIF Engine
Path Dependence Core

Purpose
-------
Formalize historical order effects in a stateful adaptive connection.

Core question
-------------
Do the same exposures produce the same final state when applied in a different
order?

    A o B ?= B o A

If final states differ, the connection is path-dependent under the chosen
update rule.

This module is intentionally narrow:
- reuses AdaptiveConnectionState / ConnectionExposure
- applies explicit exposure sequences
- compares final states
- reports component-wise and aggregate path difference
- does not mutate source state
- does not learn
- does not change topology
- does not infer biological truth

Path dependence here is a computational property of the configured update law.
It is not itself a diagnosis or causal biological claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.adaptive_connection import (
    AdaptiveConnectionConfig,
    AdaptiveConnectionResult,
    AdaptiveConnectionState,
    ConnectionExposure,
    adaptive_connection_signature,
    update_adaptive_connection,
)


SCHEMA_VERSION = "path_dependence_v1"


class PathDependenceError(ValueError):
    """Raised when path-dependence inputs are invalid."""


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
        raise PathDependenceError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    number = float(value)

    if not isfinite(number):
        raise PathDependenceError(
            f"{name} must be finite"
        )

    return number


@dataclass(frozen=True, slots=True)
class ExposureSequence:
    """
    Immutable ordered exposure path.
    """

    sequence_id: str
    exposures: tuple[ConnectionExposure, ...]
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

        exposures = tuple(
            self.exposures
        )

        if not exposures:
            raise PathDependenceError(
                "exposures must not be empty"
            )

        object.__setattr__(
            self,
            "exposures",
            exposures,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class PathStep:
    """
    One explicit state transition inside a sequence.
    """

    sequence_id: str
    step_index: int
    exposure_id: str
    source_state: AdaptiveConnectionState
    target_state: AdaptiveConnectionState
    update_result: AdaptiveConnectionResult
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if int(
            self.step_index
        ) < 1:
            raise PathDependenceError(
                "step_index must be >= 1"
            )

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
            "step_index",
            int(
                self.step_index
            ),
        )

        object.__setattr__(
            self,
            "exposure_id",
            _validate_id(
                "exposure_id",
                self.exposure_id,
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
class PathExecutionResult:
    path_id: str
    source_state: AdaptiveConnectionState
    sequence: ExposureSequence
    steps: tuple[PathStep, ...]
    final_state: AdaptiveConnectionState
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path_id",
            _validate_id(
                "path_id",
                self.path_id,
            ),
        )

        object.__setattr__(
            self,
            "steps",
            tuple(
                self.steps
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
class PathDifference:
    """
    Component-wise final-state difference between two paths.

    right - left
    """

    stiffness_delta: float
    contractile_capacity_delta: float
    reflex_gain_delta: float
    fatigue_delta: float
    remodeling_bias_delta: float

    euclidean_distance: float
    max_abs_component_delta: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "stiffness_delta",
            "contractile_capacity_delta",
            "reflex_gain_delta",
            "fatigue_delta",
            "remodeling_bias_delta",
            "euclidean_distance",
            "max_abs_component_delta",
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

        if self.euclidean_distance < 0.0:
            raise PathDependenceError(
                "euclidean_distance must be nonnegative"
            )

        if self.max_abs_component_delta < 0.0:
            raise PathDependenceError(
                "max_abs_component_delta must be nonnegative"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class PathDependenceConfig:
    """
    Comparison configuration.

    tolerance
        Final-state distance <= tolerance is treated as effectively commutative.
    """

    tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        tolerance = _validate_finite(
            "tolerance",
            self.tolerance,
        )

        if tolerance < 0.0:
            raise PathDependenceError(
                "tolerance must be nonnegative"
            )

        object.__setattr__(
            self,
            "tolerance",
            tolerance,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class PathDependenceResult:
    comparison_id: str
    left_path: PathExecutionResult
    right_path: PathExecutionResult
    difference: PathDifference
    path_dependent: bool
    commutative_within_tolerance: bool
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "comparison_id",
            _validate_id(
                "comparison_id",
                self.comparison_id,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def execute_exposure_sequence(
    *,
    path_id: str,
    source_state: AdaptiveConnectionState,
    sequence: ExposureSequence,
    config: AdaptiveConnectionConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PathExecutionResult:
    """
    Execute one ordered exposure sequence from the same source state.
    """
    path_id = _validate_id(
        "path_id",
        path_id,
    )

    for exposure in sequence.exposures:
        if (
            exposure.connection_id
            != source_state.connection_id
        ):
            raise PathDependenceError(
                "all exposures must target source_state.connection_id"
            )

    current = source_state
    steps = []

    for index, exposure in enumerate(
        sequence.exposures,
        start=1,
    ):
        update_result = (
            update_adaptive_connection(
                update_id=(
                    f"{path_id}::step_{index}::{exposure.exposure_id}"
                ),
                state=current,
                exposure=exposure,
                config=config,
            )
        )

        step = PathStep(
            sequence_id=sequence.sequence_id,
            step_index=index,
            exposure_id=exposure.exposure_id,
            source_state=current,
            target_state=update_result.target_state,
            update_result=update_result,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "memory_mutated": False,
                "learning_applied": False,
                "topology_modified": False,
            },
        )

        steps.append(
            step
        )

        current = (
            update_result.target_state
        )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "path_execution_mode": "ordered_deterministic_state_updates",
        "memory_mutated": False,
        "learning_applied": False,
        "topology_modified": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_truth_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return PathExecutionResult(
        path_id=path_id,
        source_state=source_state,
        sequence=sequence,
        steps=tuple(
            steps
        ),
        final_state=current,
        metadata=merged_metadata,
    )


def _state_difference(
    left: AdaptiveConnectionState,
    right: AdaptiveConnectionState,
) -> PathDifference:
    stiffness_delta = (
        right.stiffness
        - left.stiffness
    )

    contractile_capacity_delta = (
        right.contractile_capacity
        - left.contractile_capacity
    )

    reflex_gain_delta = (
        right.reflex_gain
        - left.reflex_gain
    )

    fatigue_delta = (
        right.fatigue
        - left.fatigue
    )

    remodeling_bias_delta = (
        right.remodeling_bias
        - left.remodeling_bias
    )

    components = (
        stiffness_delta,
        contractile_capacity_delta,
        reflex_gain_delta,
        fatigue_delta,
        remodeling_bias_delta,
    )

    euclidean_distance = sqrt(
        sum(
            value * value
            for value in components
        )
    )

    max_abs_component_delta = max(
        abs(
            value
        )
        for value in components
    )

    return PathDifference(
        stiffness_delta=stiffness_delta,
        contractile_capacity_delta=contractile_capacity_delta,
        reflex_gain_delta=reflex_gain_delta,
        fatigue_delta=fatigue_delta,
        remodeling_bias_delta=remodeling_bias_delta,
        euclidean_distance=euclidean_distance,
        max_abs_component_delta=max_abs_component_delta,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "difference_direction": "right_minus_left",
        },
    )


def compare_exposure_paths(
    *,
    comparison_id: str,
    source_state: AdaptiveConnectionState,
    left_sequence: ExposureSequence,
    right_sequence: ExposureSequence,
    adaptive_config: AdaptiveConnectionConfig | None = None,
    config: PathDependenceConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PathDependenceResult:
    """
    Compare two ordered exposure paths from the exact same source state.
    """
    comparison_id = _validate_id(
        "comparison_id",
        comparison_id,
    )

    actual_config = (
        config
        if config is not None
        else PathDependenceConfig()
    )

    left_path = execute_exposure_sequence(
        path_id=f"{comparison_id}::left",
        source_state=source_state,
        sequence=left_sequence,
        config=adaptive_config,
    )

    right_path = execute_exposure_sequence(
        path_id=f"{comparison_id}::right",
        source_state=source_state,
        sequence=right_sequence,
        config=adaptive_config,
    )

    difference = _state_difference(
        left_path.final_state,
        right_path.final_state,
    )

    commutative = (
        difference.euclidean_distance
        <= actual_config.tolerance
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": "same_source_different_order",
        "left_sequence_id": left_sequence.sequence_id,
        "right_sequence_id": right_sequence.sequence_id,
        "memory_mutated": False,
        "learning_applied": False,
        "topology_modified": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_truth_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return PathDependenceResult(
        comparison_id=comparison_id,
        left_path=left_path,
        right_path=right_path,
        difference=difference,
        path_dependent=not commutative,
        commutative_within_tolerance=commutative,
        metadata=merged_metadata,
    )


def reverse_sequence(
    sequence: ExposureSequence,
    *,
    sequence_id: str | None = None,
) -> ExposureSequence:
    """
    Return a reversed immutable sequence.
    """
    reversed_id = (
        sequence_id
        if sequence_id is not None
        else f"{sequence.sequence_id}::reversed"
    )

    return ExposureSequence(
        sequence_id=reversed_id,
        exposures=tuple(
            reversed(
                sequence.exposures
            )
        ),
        metadata=dict(
            sequence.metadata
        ),
    )


def compare_sequence_with_reverse(
    *,
    comparison_id: str,
    source_state: AdaptiveConnectionState,
    sequence: ExposureSequence,
    adaptive_config: AdaptiveConnectionConfig | None = None,
    config: PathDependenceConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PathDependenceResult:
    """
    Convenience helper for A,B,C versus C,B,A.
    """
    reversed_path = reverse_sequence(
        sequence
    )

    return compare_exposure_paths(
        comparison_id=comparison_id,
        source_state=source_state,
        left_sequence=sequence,
        right_sequence=reversed_path,
        adaptive_config=adaptive_config,
        config=config,
        metadata=metadata,
    )


def path_dependence_is_policy_free(
    result: PathDependenceResult,
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
            "learning_applied"
        )
        is False
        and result.metadata.get(
            "topology_modified"
        )
        is False
    )


def path_signature(
    path: PathExecutionResult,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[
        str,
        str,
        float,
        float,
        float,
        float,
        float,
    ],
]:
    """
    Deterministic reporting signature.
    """
    dummy_result = AdaptiveConnectionResult(
        update_id=f"{path.path_id}::final_signature",
        source_state=path.source_state,
        exposure=path.sequence.exposures[-1],
        target_state=path.final_state,
        delta=path.steps[-1].update_result.delta,
        metadata={
            "schema_version": SCHEMA_VERSION,
        },
    )

    return (
        path.path_id,
        tuple(
            exposure.exposure_id
            for exposure in path.sequence.exposures
        ),
        adaptive_connection_signature(
            dummy_result
        ),
    )


def path_difference_vector(
    result: PathDependenceResult,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    difference = (
        result.difference
    )

    return (
        difference.stiffness_delta,
        difference.contractile_capacity_delta,
        difference.reflex_gain_delta,
        difference.fatigue_delta,
        difference.remodeling_bias_delta,
    )


__all__ = [
    "ExposureSequence",
    "PathDependenceConfig",
    "PathDependenceError",
    "PathDependenceResult",
    "PathDifference",
    "PathExecutionResult",
    "PathStep",
    "SCHEMA_VERSION",
    "compare_exposure_paths",
    "compare_sequence_with_reverse",
    "execute_exposure_sequence",
    "path_dependence_is_policy_free",
    "path_difference_vector",
    "path_signature",
    "reverse_sequence",
]
