"""
ROIF External-Domain Validation
Scenario 04K — Contextual Prestress / Associative Transition Benchmark

Purpose
-------
Test the stronger associative-memory hypothesis:

    SAME physical tensor node
    SAME stored memory
    SAME query input
    SAME policy
    NO learning between queries

        +

    DIFFERENT contextual prestress

        -> different transformed response geometry
        -> different dominant association
        -> different next-state attractor

This benchmark extends 04I/04J.

Critical architectural rule
---------------------------
Context is NOT a label used to choose the answer.

We do NOT implement:

    if context == "home":
        return "home"

Instead, context is encoded only as a prestress vector in the same structural
space as the stored tensor node. That prestress perturbs the node's effective
anisotropic transformation before response projection.

The dominant association and next-state attractor are derived from the
resulting structural geometry.

Interpretation
--------------
The benchmark is an operational computational abstraction. It does not claim
that biological memory literally implements this exact tensor operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from validation.sailing.sailing_yacht_crew_tensor_node_multiplex_memory import (
    EPSILON,
    SHARED_NODE_ID,
    TensorMemoryNode,
    TensorNodeMemoryError,
    build_tensor_memory_node,
    build_tensor_query_record,
)


SCENARIO_ID = "sailing_04K_contextual_prestress_memory"

DEFAULT_HISTORY_PER_IMAGE = 16
DEFAULT_IMAGE_COUNT = 4
DEFAULT_QUERY_INDEX = 65

DEFAULT_PRESTRESS_GAIN = 1.25
DEFAULT_CONTEXT_ALIGNMENT_GAIN = 1.00

CONTEXT_IDS = (
    "office",
    "home",
    "son_room",
    "basement",
)


class ContextualPrestressError(RuntimeError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _dot(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise ContextualPrestressError(
            "vector dimensions must match"
        )

    return sum(
        float(x) * float(y)
        for x, y in zip(
            a,
            b,
        )
    )


def _norm(
    vector: Sequence[float],
) -> float:
    return sqrt(
        sum(
            float(value) * float(value)
            for value in vector
        )
    )


def _normalize(
    vector: Sequence[float],
) -> tuple[float, ...]:
    magnitude = _norm(
        vector
    )

    if magnitude <= EPSILON:
        return tuple(
            0.0
            for _ in vector
        )

    return tuple(
        float(value)
        / magnitude
        for value in vector
    )


def _subtract(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, ...]:
    if len(a) != len(b):
        raise ContextualPrestressError(
            "vector dimensions must match"
        )

    return tuple(
        float(x) - float(y)
        for x, y in zip(
            a,
            b,
        )
    )


def _add(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, ...]:
    if len(a) != len(b):
        raise ContextualPrestressError(
            "vector dimensions must match"
        )

    return tuple(
        float(x) + float(y)
        for x, y in zip(
            a,
            b,
        )
    )


def _scale(
    vector: Sequence[float],
    factor: float,
) -> tuple[float, ...]:
    return tuple(
        float(value) * float(factor)
        for value in vector
    )


def _matrix_vector(
    matrix: Sequence[
        Sequence[float]
    ],
    vector: Sequence[float],
) -> tuple[float, ...]:
    if len(matrix) != len(vector):
        raise ContextualPrestressError(
            "matrix/vector dimensions must match"
        )

    if any(
        len(row) != len(vector)
        for row in matrix
    ):
        raise ContextualPrestressError(
            "matrix must be square"
        )

    return tuple(
        _dot(
            row,
            vector,
        )
        for row in matrix
    )


def _record_vector(
    record,
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in record.signature.compact_vector()
    )


@dataclass(frozen=True, slots=True)
class ContextPrestressState:
    context_id: str
    prestress_vector: tuple[float, ...]
    target_prototype_id: str
    prestress_gain: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ContextualPrestressError(
                "context_id must not be empty"
            )

        if not self.prestress_vector:
            raise ContextualPrestressError(
                "prestress_vector must not be empty"
            )

        if not self.target_prototype_id:
            raise ContextualPrestressError(
                "target_prototype_id must not be empty"
            )

        if (
            self.prestress_gain < 0.0
            or not isfinite(
                self.prestress_gain
            )
        ):
            raise ContextualPrestressError(
                "prestress_gain must be finite and non-negative"
            )

        object.__setattr__(
            self,
            "prestress_vector",
            tuple(
                self.prestress_vector
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
class ContextualAssociationResponse:
    query_record_id: str
    node_id: str
    context_id: str

    raw_query_direction: tuple[float, ...]
    prestressed_query_direction: tuple[float, ...]
    transformed_direction: tuple[float, ...]

    response_profile: tuple[
        tuple[str, float],
        ...,
    ]

    dominant_association_id: str | None
    dominant_activation: float
    second_activation: float
    dominant_margin: float
    ambiguity: float
    response_energy: float

    next_state_attractor_id: str | None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.query_record_id:
            raise ContextualPrestressError(
                "query_record_id must not be empty"
            )

        if not self.node_id:
            raise ContextualPrestressError(
                "node_id must not be empty"
            )

        if not self.context_id:
            raise ContextualPrestressError(
                "context_id must not be empty"
            )

        for name in (
            "dominant_activation",
            "second_activation",
            "dominant_margin",
            "ambiguity",
            "response_energy",
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
                raise ContextualPrestressError(
                    f"{name} must be finite and non-negative"
                )

        if self.ambiguity > 1.0:
            raise ContextualPrestressError(
                "ambiguity must be <= 1"
            )

        object.__setattr__(
            self,
            "raw_query_direction",
            tuple(
                self.raw_query_direction
            ),
        )

        object.__setattr__(
            self,
            "prestressed_query_direction",
            tuple(
                self.prestressed_query_direction
            ),
        )

        object.__setattr__(
            self,
            "transformed_direction",
            tuple(
                self.transformed_direction
            ),
        )

        object.__setattr__(
            self,
            "response_profile",
            tuple(
                self.response_profile
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
class ContextualPrestressBenchmarkResult:
    scenario_id: str

    node: TensorMemoryNode
    query_record: Any

    contexts: tuple[
        ContextPrestressState,
        ...,
    ]

    responses: tuple[
        ContextualAssociationResponse,
        ...,
    ]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ContextualPrestressError(
                "scenario_id must not be empty"
            )

        if not self.contexts:
            raise ContextualPrestressError(
                "contexts must not be empty"
            )

        if len(
            self.contexts
        ) != len(
            self.responses
        ):
            raise ContextualPrestressError(
                "contexts and responses must have equal length"
            )

        object.__setattr__(
            self,
            "contexts",
            tuple(
                self.contexts
            ),
        )

        object.__setattr__(
            self,
            "responses",
            tuple(
                self.responses
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
# Context construction
# =============================================================================


def build_context_prestress_states(
    node: TensorMemoryNode,
    *,
    prestress_gain: float = DEFAULT_PRESTRESS_GAIN,
) -> tuple[
    ContextPrestressState,
    ...,
]:
    """
    Build context prestress vectors from the node's own prototype directions.

    This intentionally avoids a direct context -> answer lookup during
    response calculation.  Each context only supplies a structural vector.
    """

    if (
        prestress_gain < 0.0
        or not isfinite(
            prestress_gain
        )
    ):
        raise ContextualPrestressError(
            "prestress_gain must be finite and non-negative"
        )

    if len(
        node.image_prototypes
    ) < len(
        CONTEXT_IDS
    ):
        raise ContextualPrestressError(
            "node does not contain enough prototypes for default contexts"
        )

    states = []

    for context_id, prototype in zip(
        CONTEXT_IDS,
        node.image_prototypes,
    ):
        prestress_vector = _normalize(
            prototype.direction
        )

        states.append(
            ContextPrestressState(
                context_id=context_id,
                prestress_vector=prestress_vector,
                target_prototype_id=prototype.image_id,
                prestress_gain=prestress_gain,
                metadata={
                    "scenario_id": SCENARIO_ID,
                    "context_encoded_as_structural_vector": True,
                    "context_label_used_for_response_selection": False,
                    "external_expected_label_used": False,
                },
            )
        )

    return tuple(
        states
    )


# =============================================================================
# Contextual response
# =============================================================================


def respond_with_contextual_prestress(
    node: TensorMemoryNode,
    query_record,
    context: ContextPrestressState,
    *,
    context_alignment_gain: float = DEFAULT_CONTEXT_ALIGNMENT_GAIN,
) -> ContextualAssociationResponse:
    """
    Apply structural prestress to the SAME query before tensor-node response.

    No stored memory is changed.
    No prototype is added.
    No policy is modified.
    """

    if (
        context_alignment_gain < 0.0
        or not isfinite(
            context_alignment_gain
        )
    ):
        raise ContextualPrestressError(
            "context_alignment_gain must be finite and non-negative"
        )

    query_vector = _record_vector(
        query_record
    )

    if len(
        query_vector
    ) != len(
        node.centroid
    ):
        raise ContextualPrestressError(
            "query/node vector dimensions must match"
        )

    if len(
        context.prestress_vector
    ) != len(
        node.centroid
    ):
        raise ContextualPrestressError(
            "context/node vector dimensions must match"
        )

    raw_query_direction = _normalize(
        _subtract(
            query_vector,
            node.centroid,
        )
    )

    prestress_term = _scale(
        context.prestress_vector,
        (
            context.prestress_gain
            * context_alignment_gain
        ),
    )

    prestressed_query_direction = _normalize(
        _add(
            raw_query_direction,
            prestress_term,
        )
    )

    transformed = _matrix_vector(
        node.second_moment,
        prestressed_query_direction,
    )

    transformed_direction = _normalize(
        transformed
    )

    responses = []

    for prototype in node.image_prototypes:
        directional_alignment = _dot(
            transformed_direction,
            prototype.direction,
        )

        activation = max(
            0.0,
            directional_alignment,
        ) * prototype.prestress_weight

        responses.append(
            (
                prototype.image_id,
                activation,
            )
        )

    responses.sort(
        key=lambda item: (
            -item[
                1
            ],
            item[
                0
            ],
        )
    )

    if responses:
        dominant_id = (
            responses[
                0
            ][
                0
            ]
        )

        top1 = float(
            responses[
                0
            ][
                1
            ]
        )

        top2 = (
            float(
                responses[
                    1
                ][
                    1
                ]
            )
            if len(
                responses
            ) > 1
            else 0.0
        )
    else:
        dominant_id = None
        top1 = 0.0
        top2 = 0.0

    dominant_margin = max(
        0.0,
        top1 - top2,
    )

    ambiguity = (
        min(
            1.0,
            top2 / top1,
        )
        if top1 > EPSILON
        else 0.0
    )

    response_energy = sum(
        value * value
        for _image_id, value
        in responses
    )

    next_state_attractor_id = (
        f"attractor::{dominant_id}"
        if (
            dominant_id is not None
            and top1 > EPSILON
        )
        else None
    )

    return ContextualAssociationResponse(
        query_record_id=query_record.trace_id,
        node_id=node.node_id,
        context_id=context.context_id,
        raw_query_direction=raw_query_direction,
        prestressed_query_direction=prestressed_query_direction,
        transformed_direction=transformed_direction,
        response_profile=tuple(
            responses
        ),
        dominant_association_id=dominant_id,
        dominant_activation=top1,
        second_activation=top2,
        dominant_margin=dominant_margin,
        ambiguity=ambiguity,
        response_energy=response_energy,
        next_state_attractor_id=next_state_attractor_id,
        metadata={
            "scenario_id": SCENARIO_ID,
            "same_physical_node": True,
            "same_stored_memory": True,
            "same_query_input": True,
            "prestress_only_difference": True,
            "response_derived_from_structural_geometry": True,
            "learning_applied": False,
            "memory_mutated": False,
            "action_selected": False,
            "policy_modified": False,
            "context_label_used_for_response_selection": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Benchmark
# =============================================================================


def run_contextual_prestress_benchmark(
    *,
    history_per_image: int = DEFAULT_HISTORY_PER_IMAGE,
    image_count: int = DEFAULT_IMAGE_COUNT,
    query_index: int = DEFAULT_QUERY_INDEX,
    prestress_gain: float = DEFAULT_PRESTRESS_GAIN,
    context_alignment_gain: float = DEFAULT_CONTEXT_ALIGNMENT_GAIN,
) -> ContextualPrestressBenchmarkResult:
    node = build_tensor_memory_node(
        history_per_image=history_per_image,
        image_count=image_count,
    )

    query_record = build_tensor_query_record(
        query_index=query_index,
        image_index=0,
    )

    contexts = build_context_prestress_states(
        node,
        prestress_gain=prestress_gain,
    )

    responses = tuple(
        respond_with_contextual_prestress(
            node,
            query_record,
            context,
            context_alignment_gain=context_alignment_gain,
        )
        for context
        in contexts
    )

    return ContextualPrestressBenchmarkResult(
        scenario_id=SCENARIO_ID,
        node=node,
        query_record=query_record,
        contexts=contexts,
        responses=responses,
        metadata={
            "benchmark_type": "contextual_prestress_associative_transition",
            "shared_node_id": node.node_id,
            "same_physical_node": all(
                response.node_id
                == node.node_id
                for response
                in responses
            ),
            "same_query_record": all(
                response.query_record_id
                == query_record.trace_id
                for response
                in responses
            ),
            "same_stored_memory": True,
            "learning_between_queries": False,
            "memory_mutated_between_queries": False,
            "context_label_used_for_response_selection": False,
            "memory_selects_action": False,
            "policy_override_enabled": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Reporting helpers
# =============================================================================


def context_ids(
    result: ContextualPrestressBenchmarkResult,
) -> tuple[str, ...]:
    return tuple(
        context.context_id
        for context
        in result.contexts
    )


def dominant_association_ids(
    result: ContextualPrestressBenchmarkResult,
) -> tuple[
    str | None,
    ...,
]:
    return tuple(
        response.dominant_association_id
        for response
        in result.responses
    )


def next_state_attractor_ids(
    result: ContextualPrestressBenchmarkResult,
) -> tuple[
    str | None,
    ...,
]:
    return tuple(
        response.next_state_attractor_id
        for response
        in result.responses
    )


def response_margins(
    result: ContextualPrestressBenchmarkResult,
) -> tuple[
    float,
    ...,
]:
    return tuple(
        response.dominant_margin
        for response
        in result.responses
    )


def response_ambiguities(
    result: ContextualPrestressBenchmarkResult,
) -> tuple[
    float,
    ...,
]:
    return tuple(
        response.ambiguity
        for response
        in result.responses
    )


def response_energies(
    result: ContextualPrestressBenchmarkResult,
) -> tuple[
    float,
    ...,
]:
    return tuple(
        response.response_energy
        for response
        in result.responses
    )


def unique_dominant_association_count(
    result: ContextualPrestressBenchmarkResult,
) -> int:
    return len(
        {
            item
            for item in dominant_association_ids(
                result
            )
            if item is not None
        }
    )


def unique_attractor_count(
    result: ContextualPrestressBenchmarkResult,
) -> int:
    return len(
        {
            item
            for item in next_state_attractor_ids(
                result
            )
            if item is not None
        }
    )


def contextual_prestress_is_policy_free(
    result: ContextualPrestressBenchmarkResult,
) -> bool:
    return (
        result.metadata.get(
            "memory_selects_action"
        )
        is False
        and result.metadata.get(
            "policy_override_enabled"
        )
        is False
        and result.metadata.get(
            "learning_between_queries"
        )
        is False
        and result.metadata.get(
            "memory_mutated_between_queries"
        )
        is False
        and all(
            response.metadata.get(
                "action_selected"
            )
            is False
            and response.metadata.get(
                "policy_modified"
            )
            is False
            and response.metadata.get(
                "learning_applied"
            )
            is False
            and response.metadata.get(
                "memory_mutated"
            )
            is False
            for response
            in result.responses
        )
    )


__all__ = [
    "CONTEXT_IDS",
    "ContextPrestressState",
    "ContextualAssociationResponse",
    "ContextualPrestressBenchmarkResult",
    "ContextualPrestressError",
    "DEFAULT_CONTEXT_ALIGNMENT_GAIN",
    "DEFAULT_HISTORY_PER_IMAGE",
    "DEFAULT_IMAGE_COUNT",
    "DEFAULT_PRESTRESS_GAIN",
    "DEFAULT_QUERY_INDEX",
    "SCENARIO_ID",
    "build_context_prestress_states",
    "context_ids",
    "contextual_prestress_is_policy_free",
    "dominant_association_ids",
    "next_state_attractor_ids",
    "respond_with_contextual_prestress",
    "response_ambiguities",
    "response_energies",
    "response_margins",
    "run_contextual_prestress_benchmark",
    "unique_attractor_count",
    "unique_dominant_association_count",
]
