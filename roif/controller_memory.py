"""
ROIF Controller Memory
======================

Purpose
-------
Provide a persistent, auditable controller-memory layer built from immutable
ExperienceTrace records.

This module introduces two concepts:

1. ControllerMemory
   A bounded collection of completed ExperienceTrace objects.

2. MemoryScar
   A derived, immutable summary of repeated systematic prediction error for a
   recognizable context/pattern.

Architectural boundaries
------------------------

    ExperienceTrace != Learning
    ControllerMemory != PredictivePreload
    MemoryScar != PredictiveState Mutation

This module DOES NOT:
- directly modify PredictiveState;
- alter candidate ranking;
- alter control policy;
- apply reinforcement learning;
- mutate graphs;
- rewrite past ExperienceTrace objects;
- infer biological memory.

It only stores and summarizes prior experience.

A later layer may choose whether and how an authorized MemoryScar influences
PredictivePreload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from roif.experience_trace import (
    ExperienceTrace,
    StabilizationOutcome,
    trace_is_learning_free,
)


# =============================================================================
# Errors
# =============================================================================


class ControllerMemoryError(RuntimeError):
    """Base controller-memory error."""


class InvalidControllerMemoryError(ControllerMemoryError):
    """Raised when controller memory is structurally invalid."""


class DuplicateExperienceTraceError(ControllerMemoryError):
    """Raised when the same trace_id is inserted twice."""


class MemoryScarError(ControllerMemoryError):
    """Raised when MemoryScar derivation fails."""


# =============================================================================
# Helpers
# =============================================================================


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {}
        if value is None
        else dict(value)
    )


def _finite_float(
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


def _finite_non_negative(
    value: float,
    *,
    name: str,
) -> float:
    value = _finite_float(
        value,
        name=name,
    )

    if value < 0.0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return value


def _mean(
    values: Sequence[float],
) -> float:
    if not values:
        return 0.0

    return sum(
        float(value)
        for value
        in values
    ) / len(values)


def _l2_norm(
    values: Iterable[float],
) -> float:
    return sqrt(
        sum(
            float(value) ** 2
            for value
            in values
        )
    )


# =============================================================================
# Pattern identity
# =============================================================================


@dataclass(frozen=True, slots=True)
class MemoryPattern:
    """
    Domain-neutral identity for a recognizable experience family.

    Examples:
    - "sailing:pattern_A"
    - "mechanical:reverse_response"
    - "clinical:foot_knee_pattern_1"

    features are optional descriptors for later similarity logic.
    """

    pattern_id: str
    features: Mapping[str, float] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.pattern_id:
            raise InvalidControllerMemoryError(
                "pattern_id must not be empty"
            )

        object.__setattr__(
            self,
            "features",
            MappingProxyType(
                {
                    str(key): float(value)
                    for key, value
                    in self.features.items()
                }
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# Memory scar
# =============================================================================


@dataclass(frozen=True, slots=True)
class MemoryScar:
    """
    Immutable summary of repeated systematic prediction error.

    bias_by_variable:
        Mean observed-minus-predicted error for each shared state variable.

    error_norm_mean:
        Mean L2 prediction error across included traces.

    error_norm_last:
        Most recent L2 prediction error.

    recurrence_count:
        Number of traces included.

    consistency:
        [0, 1], high when repeated errors point in a stable direction.

    confidence:
        [0, 1], derived from recurrence and consistency.

    Important:
    MemoryScar is descriptive. It does not modify PredictiveState.
    """

    scar_id: str
    pattern: MemoryPattern
    trace_ids: tuple[str, ...]
    bias_by_variable: Mapping[str, float]
    error_norm_mean: float
    error_norm_last: float
    recurrence_count: int
    consistency: float
    confidence: float
    improved_count: int
    unchanged_count: int
    worsened_count: int
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.scar_id:
            raise MemoryScarError(
                "scar_id must not be empty"
            )

        if self.recurrence_count <= 0:
            raise MemoryScarError(
                "recurrence_count must be positive"
            )

        if len(
            self.trace_ids
        ) != self.recurrence_count:
            raise MemoryScarError(
                "trace_ids length must equal recurrence_count"
            )

        object.__setattr__(
            self,
            "trace_ids",
            tuple(
                str(value)
                for value
                in self.trace_ids
            ),
        )

        object.__setattr__(
            self,
            "bias_by_variable",
            MappingProxyType(
                {
                    str(key): _finite_float(
                        value,
                        name=f"bias_by_variable[{key}]",
                    )
                    for key, value
                    in self.bias_by_variable.items()
                }
            ),
        )

        object.__setattr__(
            self,
            "error_norm_mean",
            _finite_non_negative(
                self.error_norm_mean,
                name="error_norm_mean",
            ),
        )

        object.__setattr__(
            self,
            "error_norm_last",
            _finite_non_negative(
                self.error_norm_last,
                name="error_norm_last",
            ),
        )

        object.__setattr__(
            self,
            "consistency",
            max(
                0.0,
                min(
                    1.0,
                    float(
                        self.consistency
                    ),
                ),
            ),
        )

        object.__setattr__(
            self,
            "confidence",
            max(
                0.0,
                min(
                    1.0,
                    float(
                        self.confidence
                    ),
                ),
            ),
        )

        for name in (
            "improved_count",
            "unchanged_count",
            "worsened_count",
        ):
            value = int(
                getattr(
                    self,
                    name,
                )
            )

            if value < 0:
                raise MemoryScarError(
                    f"{name} must be non-negative"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        if (
            self.improved_count
            + self.unchanged_count
            + self.worsened_count
            != self.recurrence_count
        ):
            raise MemoryScarError(
                "stabilization counts must sum to recurrence_count"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# Controller memory
# =============================================================================


@dataclass(frozen=True, slots=True)
class ControllerMemory:
    """
    Immutable controller-memory snapshot.

    Adding a trace returns a NEW ControllerMemory object.
    Existing memory snapshots are never mutated.
    """

    traces: tuple[ExperienceTrace, ...] = ()
    max_traces: int | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        traces = tuple(
            self.traces
        )

        trace_ids = tuple(
            trace.identity.trace_id
            for trace
            in traces
        )

        if len(
            set(
                trace_ids
            )
        ) != len(
            trace_ids
        ):
            raise DuplicateExperienceTraceError(
                "ControllerMemory contains duplicate trace_id values"
            )

        if self.max_traces is not None:
            if self.max_traces <= 0:
                raise InvalidControllerMemoryError(
                    "max_traces must be positive when provided"
                )

            if len(
                traces
            ) > self.max_traces:
                raise InvalidControllerMemoryError(
                    "traces exceed max_traces"
                )

        if not all(
            trace_is_learning_free(
                trace
            )
            for trace
            in traces
        ):
            raise InvalidControllerMemoryError(
                "ControllerMemory only accepts learning-free ExperienceTrace records"
            )

        object.__setattr__(
            self,
            "traces",
            traces,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )

    @property
    def size(self) -> int:
        return len(
            self.traces
        )

    @property
    def trace_ids(self) -> tuple[str, ...]:
        return tuple(
            trace.identity.trace_id
            for trace
            in self.traces
        )


# =============================================================================
# Memory mutation-as-copy
# =============================================================================


def append_experience_trace(
    memory: ControllerMemory,
    trace: ExperienceTrace,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ControllerMemory:
    """
    Return a NEW memory snapshot containing trace.

    No in-place mutation is performed.
    """

    if not trace_is_learning_free(
        trace
    ):
        raise InvalidControllerMemoryError(
            "cannot store a trace that already declares learning/mutation"
        )

    if (
        trace.identity.trace_id
        in memory.trace_ids
    ):
        raise DuplicateExperienceTraceError(
            f"duplicate trace_id: {trace.identity.trace_id}"
        )

    updated = (
        memory.traces
        + (
            trace,
        )
    )

    if (
        memory.max_traces
        is not None
        and len(
            updated
        )
        > memory.max_traces
    ):
        updated = updated[
            -memory.max_traces:
        ]

    return ControllerMemory(
        traces=updated,
        max_traces=memory.max_traces,
        metadata={
            **dict(
                memory.metadata
            ),
            **(
                {}
                if metadata is None
                else dict(
                    metadata
                )
            ),
            "controller_memory_snapshot": True,
            "predictive_state_mutated": False,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def append_experience_traces(
    memory: ControllerMemory,
    traces: Sequence[ExperienceTrace],
) -> ControllerMemory:
    result = memory

    for trace in traces:
        result = append_experience_trace(
            result,
            trace,
        )

    return result


# =============================================================================
# Query helpers
# =============================================================================


TracePredicate = Callable[
    [ExperienceTrace],
    bool,
]


def filter_traces(
    memory: ControllerMemory,
    predicate: TracePredicate,
) -> tuple[ExperienceTrace, ...]:
    return tuple(
        trace
        for trace
        in memory.traces
        if predicate(
            trace
        )
    )


def traces_for_pattern(
    memory: ControllerMemory,
    pattern_id: str,
    *,
    metadata_key: str = "pattern_family",
) -> tuple[ExperienceTrace, ...]:
    """
    Retrieve traces whose event/trace metadata declares a matching pattern.

    04B stores pattern_family on the disturbance event metadata, while the
    ExperienceTrace itself stores event/scenario provenance. Since generic
    ExperienceTrace does not know domain-specific event classes, this helper
    supports either:
      - trace.metadata[metadata_key]
      - trace.identity.metadata[metadata_key]
      - trace.disturbance.metadata[metadata_key]

    Later adapters may supply richer pattern indexing.
    """

    matches = []

    for trace in memory.traces:
        values = (
            trace.metadata.get(
                metadata_key
            ),
            trace.identity.metadata.get(
                metadata_key
            ),
            trace.disturbance.metadata.get(
                metadata_key
            ),
        )

        if pattern_id in values:
            matches.append(
                trace
            )

    return tuple(
        matches
    )


def most_recent_trace(
    memory: ControllerMemory,
) -> ExperienceTrace | None:
    if not memory.traces:
        return None

    return memory.traces[
        -1
    ]


# =============================================================================
# Scar derivation
# =============================================================================


def _trace_error_vector(
    trace: ExperienceTrace,
) -> Mapping[str, float]:
    if trace.prediction_error is None:
        return MappingProxyType(
            {}
        )

    return trace.prediction_error.variable_errors


def _mean_bias_by_variable(
    traces: Sequence[ExperienceTrace],
) -> Mapping[str, float]:
    keys = sorted(
        {
            key
            for trace
            in traces
            for key
            in _trace_error_vector(
                trace
            )
        }
    )

    result = {}

    for key in keys:
        values = tuple(
            _trace_error_vector(
                trace
            )[
                key
            ]
            for trace
            in traces
            if (
                key
                in _trace_error_vector(
                    trace
                )
            )
        )

        if values:
            result[
                key
            ] = _mean(
                values
            )

    return MappingProxyType(
        result
    )


def _direction_consistency(
    traces: Sequence[ExperienceTrace],
    mean_bias: Mapping[str, float],
) -> float:
    """
    Measure whether error vectors point in the same broad direction.

    1.0  -> all nonzero vectors aligned with mean bias
    0.0  -> maximally inconsistent / no directional structure
    """

    if not traces:
        return 0.0

    mean_norm = _l2_norm(
        mean_bias.values()
    )

    if mean_norm <= 1e-12:
        return 0.0

    cosines = []

    keys = tuple(
        mean_bias.keys()
    )

    for trace in traces:
        vector = _trace_error_vector(
            trace
        )

        components = tuple(
            float(
                vector.get(
                    key,
                    0.0,
                )
            )
            for key
            in keys
        )

        norm = _l2_norm(
            components
        )

        if norm <= 1e-12:
            continue

        dot = sum(
            components[index]
            * mean_bias[
                key
            ]
            for index, key
            in enumerate(
                keys
            )
        )

        cosine = (
            dot
            / (
                norm
                * mean_norm
            )
        )

        cosines.append(
            max(
                -1.0,
                min(
                    1.0,
                    cosine,
                ),
            )
        )

    if not cosines:
        return 0.0

    # Map [-1, 1] -> [0, 1]
    return _mean(
        tuple(
            (
                cosine
                + 1.0
            )
            / 2.0
            for cosine
            in cosines
        )
    )


def _recurrence_confidence(
    recurrence_count: int,
    consistency: float,
) -> float:
    """
    Conservative confidence:
    recurrence rises asymptotically, then is gated by consistency.
    """

    recurrence_factor = (
        recurrence_count
        / (
            recurrence_count
            + 2.0
        )
    )

    return max(
        0.0,
        min(
            1.0,
            recurrence_factor
            * consistency,
        ),
    )


def derive_memory_scar(
    *,
    pattern: MemoryPattern,
    traces: Sequence[ExperienceTrace],
    scar_id: str | None = None,
    require_prediction_error: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> MemoryScar:
    """
    Derive a descriptive MemoryScar from repeated traces.

    No predictive state or policy is modified.
    """

    trace_tuple = tuple(
        traces
    )

    if not trace_tuple:
        raise MemoryScarError(
            "cannot derive MemoryScar from zero traces"
        )

    if not all(
        trace_is_learning_free(
            trace
        )
        for trace
        in trace_tuple
    ):
        raise MemoryScarError(
            "MemoryScar requires learning-free source traces"
        )

    if require_prediction_error:
        missing = tuple(
            trace.identity.trace_id
            for trace
            in trace_tuple
            if trace.prediction_error is None
        )

        if missing:
            raise MemoryScarError(
                "prediction_error required for traces: "
                f"{missing}"
            )

    usable = tuple(
        trace
        for trace
        in trace_tuple
        if trace.prediction_error is not None
    )

    if not usable:
        raise MemoryScarError(
            "no traces with prediction error available"
        )

    mean_bias = _mean_bias_by_variable(
        usable
    )

    error_norms = tuple(
        trace.prediction_error.l2_error
        for trace
        in usable
        if trace.prediction_error is not None
    )

    consistency = _direction_consistency(
        usable,
        mean_bias,
    )

    confidence = _recurrence_confidence(
        len(
            usable
        ),
        consistency,
    )

    improved = sum(
        1
        for trace
        in usable
        if (
            trace.stabilization.outcome
            is StabilizationOutcome.IMPROVED
        )
    )

    unchanged = sum(
        1
        for trace
        in usable
        if (
            trace.stabilization.outcome
            is StabilizationOutcome.UNCHANGED
        )
    )

    worsened = sum(
        1
        for trace
        in usable
        if (
            trace.stabilization.outcome
            is StabilizationOutcome.WORSENED
        )
    )

    resolved_scar_id = (
        scar_id
        or (
            "memory_scar:"
            + pattern.pattern_id
        )
    )

    return MemoryScar(
        scar_id=resolved_scar_id,
        pattern=pattern,
        trace_ids=tuple(
            trace.identity.trace_id
            for trace
            in usable
        ),
        bias_by_variable=mean_bias,
        error_norm_mean=_mean(
            error_norms
        ),
        error_norm_last=error_norms[
            -1
        ],
        recurrence_count=len(
            usable
        ),
        consistency=consistency,
        confidence=confidence,
        improved_count=improved,
        unchanged_count=unchanged,
        worsened_count=worsened,
        metadata={
            **(
                {}
                if metadata is None
                else dict(
                    metadata
                )
            ),
            "memory_scar_generated": True,
            "source_trace_count": len(
                usable
            ),
            "predictive_state_mutated": False,
            "predictive_preload_applied": False,
            "policy_modified": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def derive_memory_scar_from_memory(
    memory: ControllerMemory,
    *,
    pattern: MemoryPattern,
    metadata_key: str = "pattern_family",
    scar_id: str | None = None,
) -> MemoryScar:
    traces = traces_for_pattern(
        memory,
        pattern.pattern_id,
        metadata_key=metadata_key,
    )

    return derive_memory_scar(
        pattern=pattern,
        traces=traces,
        scar_id=scar_id,
    )


# =============================================================================
# Scar inspection
# =============================================================================


def scar_bias_norm(
    scar: MemoryScar,
) -> float:
    return _l2_norm(
        scar.bias_by_variable.values()
    )


def scar_is_systematic(
    scar: MemoryScar,
    *,
    min_recurrence: int = 2,
    min_consistency: float = 0.80,
    min_confidence: float = 0.25,
) -> bool:
    return (
        scar.recurrence_count
        >= int(
            min_recurrence
        )
        and scar.consistency
        >= float(
            min_consistency
        )
        and scar.confidence
        >= float(
            min_confidence
        )
        and scar_bias_norm(
            scar
        )
        > 0.0
    )


def memory_is_preload_free(
    memory: ControllerMemory,
) -> bool:
    """
    Audit helper:
    ControllerMemory stores experience but does not apply PredictivePreload.
    """

    return (
        memory.metadata.get(
            "predictive_preload_applied",
            False,
        )
        is False
        and memory.metadata.get(
            "predictive_state_mutated",
            False,
        )
        is False
        and memory.metadata.get(
            "graph_mutated",
            False,
        )
        is False
    )


def scar_is_preload_free(
    scar: MemoryScar,
) -> bool:
    return (
        scar.metadata.get(
            "predictive_preload_applied"
        )
        is False
        and scar.metadata.get(
            "predictive_state_mutated"
        )
        is False
        and scar.metadata.get(
            "policy_modified"
        )
        is False
        and scar.metadata.get(
            "graph_mutated"
        )
        is False
    )


# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "ControllerMemory",
    "ControllerMemoryError",
    "DuplicateExperienceTraceError",
    "InvalidControllerMemoryError",
    "MemoryPattern",
    "MemoryScar",
    "MemoryScarError",
    "TracePredicate",
    "append_experience_trace",
    "append_experience_traces",
    "derive_memory_scar",
    "derive_memory_scar_from_memory",
    "filter_traces",
    "memory_is_preload_free",
    "most_recent_trace",
    "scar_bias_norm",
    "scar_is_preload_free",
    "scar_is_systematic",
    "traces_for_pattern",
]
