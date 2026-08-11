"""
ROIF External-Domain Validation
Scenario 04I — Tensor Node / Multiplex Memory

Purpose
-------
Test the stronger recursive-memory hypothesis:

    one physical recursive memory node
        can participate in multiple structural memory images
        and produce different responses to different query directions.

This is NOT implemented as:
    one image -> one dedicated node

Instead:
    many structural images -> one shared tensor-like node
    query direction + prestressed internal profile -> response profile

The implementation remains deliberately conservative:
- no action selection by memory;
- no policy override;
- no evaluator label is used for retrieval;
- response is derived only from structural signature geometry;
- "tensor node" is an operational computational abstraction, not a
  neurobiological claim.

Scenario
--------
Four structural image families are projected into one shared multiplex node.
Each family contributes a direction in structural-signature space.

The node stores:
    centroid
    covariance-like second moment
    family-free structural directions
    prestress weights

A query is projected through the shared node.  Different query directions
should produce different response profiles while preserving the SAME node_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from validation.sailing.sailing_yacht_crew_deep_recursive_memory import (
    build_deep_memory_event,
)
from validation.sailing.sailing_yacht_crew_long_memory_compression import (
    event_to_controller_history_record,
)
from roif.history.controller_history_adapter import (
    ControllerHistoryRecord,
    adapt_experience_trace,
)


SCENARIO_ID = "sailing_04I_tensor_node_multiplex_memory"
SHARED_NODE_ID = "tensor_node_shared_0001"

DEFAULT_HISTORY_PER_IMAGE = 16
DEFAULT_IMAGE_COUNT = 4
DEFAULT_QUERY_REPETITIONS = 2
EPSILON = 1.0e-12


class TensorNodeMemoryError(RuntimeError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise TensorNodeMemoryError("vector dimensions must match")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return sqrt(sum(float(x) * float(x) for x in a))


def _normalize(a: Sequence[float]) -> tuple[float, ...]:
    n = _norm(a)
    if n <= EPSILON:
        return tuple(0.0 for _ in a)
    return tuple(float(x) / n for x in a)


def _subtract(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, ...]:
    if len(a) != len(b):
        raise TensorNodeMemoryError("vector dimensions must match")
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _mean_vector(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    if not vectors:
        raise TensorNodeMemoryError("cannot average zero vectors")

    width = len(vectors[0])
    if width == 0:
        raise TensorNodeMemoryError("vectors must not be empty")

    if any(len(v) != width for v in vectors):
        raise TensorNodeMemoryError("all vectors must have equal length")

    return tuple(
        sum(float(v[i]) for v in vectors) / len(vectors)
        for i in range(width)
    )


def _outer_second_moment(
    centered_vectors: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not centered_vectors:
        raise TensorNodeMemoryError("cannot build moment from zero vectors")

    width = len(centered_vectors[0])
    count = float(len(centered_vectors))

    return tuple(
        tuple(
            sum(float(v[i]) * float(v[j]) for v in centered_vectors) / count
            for j in range(width)
        )
        for i in range(width)
    )


def _matrix_vector(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, ...]:
    if len(matrix) != len(vector):
        raise TensorNodeMemoryError("matrix/vector dimensions must match")

    if any(len(row) != len(vector) for row in matrix):
        raise TensorNodeMemoryError("matrix must be square")

    return tuple(_dot(row, vector) for row in matrix)


@dataclass(frozen=True, slots=True)
class MultiplexImagePrototype:
    image_id: str
    source_record_ids: tuple[str, ...]
    centroid: tuple[float, ...]
    direction: tuple[float, ...]
    prestress_weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_id:
            raise TensorNodeMemoryError("image_id must not be empty")
        if not self.source_record_ids:
            raise TensorNodeMemoryError("source_record_ids must not be empty")
        if not self.centroid or not self.direction:
            raise TensorNodeMemoryError("prototype vectors must not be empty")
        if len(self.centroid) != len(self.direction):
            raise TensorNodeMemoryError("prototype vector dimensions must match")
        if self.prestress_weight < 0.0 or not isfinite(self.prestress_weight):
            raise TensorNodeMemoryError(
                "prestress_weight must be finite and non-negative"
            )
        object.__setattr__(self, "source_record_ids", tuple(self.source_record_ids))
        object.__setattr__(self, "centroid", tuple(self.centroid))
        object.__setattr__(self, "direction", tuple(self.direction))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class TensorMemoryNode:
    node_id: str
    source_record_count: int
    image_prototypes: tuple[MultiplexImagePrototype, ...]
    centroid: tuple[float, ...]
    second_moment: tuple[tuple[float, ...], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise TensorNodeMemoryError("node_id must not be empty")
        if self.source_record_count <= 0:
            raise TensorNodeMemoryError("source_record_count must be positive")
        if len(self.image_prototypes) < 2:
            raise TensorNodeMemoryError(
                "tensor node requires at least two structural images"
            )
        if not self.centroid:
            raise TensorNodeMemoryError("centroid must not be empty")

        width = len(self.centroid)
        if len(self.second_moment) != width:
            raise TensorNodeMemoryError("second_moment dimension mismatch")
        if any(len(row) != width for row in self.second_moment):
            raise TensorNodeMemoryError("second_moment must be square")

        object.__setattr__(
            self, "image_prototypes", tuple(self.image_prototypes)
        )
        object.__setattr__(self, "centroid", tuple(self.centroid))
        object.__setattr__(
            self,
            "second_moment",
            tuple(tuple(row) for row in self.second_moment),
        )
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class TensorNodeResponse:
    query_record_id: str
    node_id: str
    query_direction: tuple[float, ...]
    transformed_direction: tuple[float, ...]
    image_response_profile: tuple[tuple[str, float], ...]
    dominant_image_id: str | None
    response_energy: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_record_id:
            raise TensorNodeMemoryError("query_record_id must not be empty")
        if not self.node_id:
            raise TensorNodeMemoryError("node_id must not be empty")
        if self.response_energy < 0.0 or not isfinite(self.response_energy):
            raise TensorNodeMemoryError(
                "response_energy must be finite and non-negative"
            )
        object.__setattr__(self, "query_direction", tuple(self.query_direction))
        object.__setattr__(
            self, "transformed_direction", tuple(self.transformed_direction)
        )
        object.__setattr__(
            self, "image_response_profile", tuple(self.image_response_profile)
        )
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class TensorNodeMultiplexResult:
    scenario_id: str
    node: TensorMemoryNode
    query_records: tuple[ControllerHistoryRecord, ...]
    responses: tuple[TensorNodeResponse, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise TensorNodeMemoryError("scenario_id must not be empty")
        if not self.query_records:
            raise TensorNodeMemoryError("query_records must not be empty")
        if len(self.query_records) != len(self.responses):
            raise TensorNodeMemoryError(
                "query_records and responses must have equal length"
            )
        object.__setattr__(self, "query_records", tuple(self.query_records))
        object.__setattr__(self, "responses", tuple(self.responses))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


def _record_vector(record: ControllerHistoryRecord) -> tuple[float, ...]:
    return tuple(float(x) for x in record.signature.compact_vector())


def build_multiplex_history_record(
    *,
    event_index: int,
    image_index: int,
) -> ControllerHistoryRecord:
    """
    Build one deterministic structural experience for the shared tensor node.

    The current 04G generator accepts a single integer index. We encode the
    desired multiplex image into that index so the existing 16 x 16 x phase
    lattice produces distinct structural directions without changing 04G.
    """
    if event_index <= 0:
        raise TensorNodeMemoryError("event_index must be positive")
    if image_index < 0:
        raise TensorNodeMemoryError("image_index must be non-negative")

    sample = (event_index - 1) % 16

    lattice_x = (sample + image_index * 3) % 16
    lattice_y = (image_index * 5 + sample // 4) % 16
    lattice_phase = image_index % 8

    deep_index = (
        1
        + lattice_x
        + 16 * lattice_y
        + 256 * lattice_phase
    )

    event = build_deep_memory_event(
        deep_index
    )

    return event_to_controller_history_record(
        event=event,
        episode_index=event_index,
        sequence_id="04I_tensor_node_multiplex",
    )

def build_tensor_memory_node(
    *,
    history_per_image: int = DEFAULT_HISTORY_PER_IMAGE,
    image_count: int = DEFAULT_IMAGE_COUNT,
) -> TensorMemoryNode:
    if history_per_image < 2:
        raise TensorNodeMemoryError("history_per_image must be >= 2")
    if image_count < 2:
        raise TensorNodeMemoryError("image_count must be >= 2")

    grouped_records: list[tuple[ControllerHistoryRecord, ...]] = []
    all_records: list[ControllerHistoryRecord] = []

    event_index = 1

    for image_index in range(image_count):
        group: list[ControllerHistoryRecord] = []

        for _ in range(history_per_image):
            record = build_multiplex_history_record(
                event_index=event_index,
                image_index=image_index,
            )
            group.append(record)
            all_records.append(record)
            event_index += 1

        grouped_records.append(tuple(group))

    all_vectors = tuple(_record_vector(record) for record in all_records)
    node_centroid = _mean_vector(all_vectors)

    centered = tuple(
        _subtract(vector, node_centroid)
        for vector in all_vectors
    )

    second_moment = _outer_second_moment(centered)

    prototypes: list[MultiplexImagePrototype] = []

    for image_index, records in enumerate(grouped_records):
        vectors = tuple(_record_vector(record) for record in records)
        centroid = _mean_vector(vectors)
        direction = _normalize(_subtract(centroid, node_centroid))

        spread = sum(
            _norm(_subtract(vector, centroid))
            for vector in vectors
        ) / len(vectors)

        # Higher structural coherence -> higher prestress weight.
        prestress_weight = 1.0 / (1.0 + spread)

        prototypes.append(
            MultiplexImagePrototype(
                image_id=f"image_{image_index:02d}",
                source_record_ids=tuple(record.trace_id for record in records),
                centroid=centroid,
                direction=direction,
                prestress_weight=prestress_weight,
                metadata={
                    "structural_image_index": image_index,
                    "evaluator_only_image_identity": True,
                    "external_expected_label_used": False,
                },
            )
        )

    return TensorMemoryNode(
        node_id=SHARED_NODE_ID,
        source_record_count=len(all_records),
        image_prototypes=tuple(prototypes),
        centroid=node_centroid,
        second_moment=second_moment,
        metadata={
            "scenario_id": SCENARIO_ID,
            "shared_physical_node": True,
            "multiplex_structural_images": True,
            "tensor_like_second_moment": True,
            "prestressed_response_enabled": True,
            "action_selected": False,
            "policy_modified": False,
            "external_expected_label_used": False,
        },
    )


def build_tensor_query_record(
    *,
    query_index: int,
    image_index: int,
) -> ControllerHistoryRecord:
    return build_multiplex_history_record(
        event_index=query_index,
        image_index=image_index,
    )


def respond_tensor_memory_node(
    node: TensorMemoryNode,
    query_record: ControllerHistoryRecord,
) -> TensorNodeResponse:
    query_vector = _record_vector(query_record)

    if len(query_vector) != len(node.centroid):
        raise TensorNodeMemoryError("query/node vector dimensions must match")

    query_direction = _normalize(
        _subtract(query_vector, node.centroid)
    )

    # Prestressed anisotropic transformation:
    # the same incoming direction is transformed by the node's stored
    # second-moment structure.
    transformed = _matrix_vector(
        node.second_moment,
        query_direction,
    )

    transformed_direction = _normalize(transformed)

    responses: list[tuple[str, float]] = []

    for prototype in node.image_prototypes:
        directional_alignment = _dot(
            transformed_direction,
            prototype.direction,
        )

        # Rectification keeps response magnitude interpretable as activation.
        activation = max(0.0, directional_alignment) * prototype.prestress_weight

        responses.append(
            (
                prototype.image_id,
                activation,
            )
        )

    responses.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    dominant_image_id = (
        responses[0][0]
        if responses and responses[0][1] > EPSILON
        else None
    )

    response_energy = sum(
        value * value
        for _, value in responses
    )

    return TensorNodeResponse(
        query_record_id=query_record.trace_id,
        node_id=node.node_id,
        query_direction=query_direction,
        transformed_direction=transformed_direction,
        image_response_profile=tuple(responses),
        dominant_image_id=dominant_image_id,
        response_energy=response_energy,
        metadata={
            "same_physical_node": True,
            "response_derived_from_structural_geometry": True,
            "action_selected": False,
            "policy_modified": False,
            "external_expected_label_used": False,
        },
    )


def run_tensor_node_multiplex_benchmark(
    *,
    history_per_image: int = DEFAULT_HISTORY_PER_IMAGE,
    image_count: int = DEFAULT_IMAGE_COUNT,
    query_repetitions: int = DEFAULT_QUERY_REPETITIONS,
) -> TensorNodeMultiplexResult:
    if query_repetitions < 1:
        raise TensorNodeMemoryError("query_repetitions must be >= 1")

    node = build_tensor_memory_node(
        history_per_image=history_per_image,
        image_count=image_count,
    )

    queries: list[ControllerHistoryRecord] = []
    responses: list[TensorNodeResponse] = []

    query_index = (
        history_per_image * image_count
    ) + 1

    for repetition in range(query_repetitions):
        for image_index in range(image_count):
            query = build_tensor_query_record(
                query_index=query_index,
                image_index=image_index,
            )
            response = respond_tensor_memory_node(
                node,
                query,
            )

            queries.append(query)
            responses.append(response)
            query_index += 1

    return TensorNodeMultiplexResult(
        scenario_id=SCENARIO_ID,
        node=node,
        query_records=tuple(queries),
        responses=tuple(responses),
        metadata={
            "benchmark_type": "tensor_node_multiplex_memory",
            "shared_node_id": node.node_id,
            "image_count": image_count,
            "query_repetitions": query_repetitions,
            "all_queries_use_same_physical_node": all(
                response.node_id == node.node_id
                for response in responses
            ),
            "memory_selects_action": False,
            "policy_override_enabled": False,
            "external_expected_label_used": False,
        },
    )


def response_node_ids(
    result: TensorNodeMultiplexResult,
) -> tuple[str, ...]:
    return tuple(response.node_id for response in result.responses)


def dominant_image_ids(
    result: TensorNodeMultiplexResult,
) -> tuple[str | None, ...]:
    return tuple(
        response.dominant_image_id
        for response in result.responses
    )


def response_profiles(
    result: TensorNodeMultiplexResult,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    return tuple(
        response.image_response_profile
        for response in result.responses
    )


def response_energies(
    result: TensorNodeMultiplexResult,
) -> tuple[float, ...]:
    return tuple(
        response.response_energy
        for response in result.responses
    )


def unique_response_profile_count(
    result: TensorNodeMultiplexResult,
    *,
    precision: int = 12,
) -> int:
    normalized = {
        tuple(
            (
                image_id,
                round(value, precision),
            )
            for image_id, value in response.image_response_profile
        )
        for response in result.responses
    }
    return len(normalized)


def multiplex_node_is_policy_free(
    result: TensorNodeMultiplexResult,
) -> bool:
    return (
        result.metadata.get("memory_selects_action") is False
        and result.metadata.get("policy_override_enabled") is False
        and result.node.metadata.get("action_selected") is False
        and result.node.metadata.get("policy_modified") is False
        and all(
            response.metadata.get("action_selected") is False
            and response.metadata.get("policy_modified") is False
            for response in result.responses
        )
    )


__all__ = [
    "DEFAULT_HISTORY_PER_IMAGE",
    "DEFAULT_IMAGE_COUNT",
    "DEFAULT_QUERY_REPETITIONS",
    "EPSILON",
    "MultiplexImagePrototype",
    "SCENARIO_ID",
    "SHARED_NODE_ID",
    "TensorMemoryNode",
    "TensorNodeMemoryError",
    "TensorNodeMultiplexResult",
    "TensorNodeResponse",
    "build_multiplex_history_record",
    "build_tensor_memory_node",
    "build_tensor_query_record",
    "dominant_image_ids",
    "multiplex_node_is_policy_free",
    "respond_tensor_memory_node",
    "response_energies",
    "response_node_ids",
    "response_profiles",
    "run_tensor_node_multiplex_benchmark",
    "unique_response_profile_count",
]
