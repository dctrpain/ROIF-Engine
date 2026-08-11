"""
ROIF External-Domain Validation
Scenario 04L — Context Gating / Interference Suppression
Projector-Based Contextual Prestress (v3)

Goal
----
Use the SAME physical tensor node, SAME stored memory, SAME query, and NO learning,
while context changes only the effective operator.

Projector prestress:
    c = normalized context direction
    P = c c^T
    Q = I - P

    T_context =
        g_parallel * P T P
        + g_cross * (P T Q + Q T P)
        + g_perp * Q T Q

The stored tensor T is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from validation.sailing.sailing_yacht_crew_tensor_node_multiplex_memory import (
    EPSILON,
    TensorMemoryNode,
    build_tensor_memory_node,
    build_tensor_query_record,
)

SCENARIO_ID = "sailing_04L_context_gating_interference_suppression"

DEFAULT_IMAGE_COUNTS = (4, 8, 16)
DEFAULT_HISTORY_PER_IMAGE = 16
DEFAULT_QUERY_INDEX = 65

DEFAULT_PARALLEL_GAIN = 3.0
DEFAULT_CROSS_GAIN = 0.85
DEFAULT_ORTHOGONAL_GAIN = 0.25


class ContextGatingError(RuntimeError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ContextGatingError("vector dimensions must match")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(v: Sequence[float]) -> float:
    return sqrt(sum(float(x) * float(x) for x in v))


def _normalize(v: Sequence[float]) -> tuple[float, ...]:
    n = _norm(v)
    if n <= EPSILON:
        return tuple(0.0 for _ in v)
    return tuple(float(x) / n for x in v)


def _subtract(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    if len(a) != len(b):
        raise ContextGatingError("vector dimensions must match")
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _matrix_vector(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, ...]:
    if len(matrix) != len(vector):
        raise ContextGatingError("matrix/vector dimensions must match")
    if any(len(row) != len(vector) for row in matrix):
        raise ContextGatingError("matrix must be square")
    return tuple(_dot(row, vector) for row in matrix)


def _matmul(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not a or not b:
        raise ContextGatingError("matrices must not be empty")
    n = len(a)
    if any(len(row) != n for row in a):
        raise ContextGatingError("left matrix must be square")
    if len(b) != n or any(len(row) != n for row in b):
        raise ContextGatingError("right matrix must match dimensions")

    return tuple(
        tuple(
            sum(float(a[i][k]) * float(b[k][j]) for k in range(n))
            for j in range(n)
        )
        for i in range(n)
    )


def _matrix_add(
    *matrices: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not matrices:
        raise ContextGatingError("at least one matrix is required")
    n = len(matrices[0])
    if any(len(m) != n for m in matrices):
        raise ContextGatingError("matrix dimensions must match")
    if any(any(len(row) != n for row in m) for m in matrices):
        raise ContextGatingError("all matrices must be square")
    return tuple(
        tuple(sum(float(m[i][j]) for m in matrices) for j in range(n))
        for i in range(n)
    )


def _matrix_scale(
    matrix: Sequence[Sequence[float]],
    factor: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) * float(factor) for value in row)
        for row in matrix
    )


def _identity(n: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(n))
        for i in range(n)
    )


def _outer(v: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(v[i]) * float(v[j]) for j in range(len(v)))
        for i in range(len(v))
    )


def _matrix_subtract(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    n = len(a)
    if len(b) != n:
        raise ContextGatingError("matrix dimensions must match")
    return tuple(
        tuple(float(a[i][j]) - float(b[i][j]) for j in range(n))
        for i in range(n)
    )


def _record_vector(record) -> tuple[float, ...]:
    return tuple(float(x) for x in record.signature.compact_vector())


def normalize_image_counts(image_counts: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(x) for x in image_counts)
    if not values:
        raise ContextGatingError("image_counts must not be empty")
    if any(x < 2 for x in values):
        raise ContextGatingError("every image_count must be >= 2")
    if len(set(values)) != len(values):
        raise ContextGatingError("image_counts must be unique")
    if tuple(sorted(values)) != values:
        raise ContextGatingError("image_counts must be increasing")
    return values


@dataclass(frozen=True, slots=True)
class ProjectorPrestress:
    context_id: str
    context_vector: tuple[float, ...]
    target_prototype_id: str
    parallel_gain: float
    cross_gain: float
    orthogonal_gain: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ContextGatingError("context_id must not be empty")
        if not self.context_vector:
            raise ContextGatingError("context_vector must not be empty")
        if not self.target_prototype_id:
            raise ContextGatingError("target_prototype_id must not be empty")

        for name in ("parallel_gain", "cross_gain", "orthogonal_gain"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ContextGatingError(f"{name} must be finite and non-negative")

        object.__setattr__(self, "context_vector", tuple(self.context_vector))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ResponseMetrics:
    dominant_association_id: str | None
    dominant_activation: float
    second_activation: float
    dominant_margin: float
    ambiguity: float
    response_energy: float
    interference_ratio: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ProjectorGatedResponse:
    query_record_id: str
    node_id: str
    context_id: str
    raw_query_direction: tuple[float, ...]
    transformed_direction: tuple[float, ...]
    response_profile: tuple[tuple[str, float], ...]
    metrics: ResponseMetrics
    target_prototype_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_query_direction", tuple(self.raw_query_direction))
        object.__setattr__(self, "transformed_direction", tuple(self.transformed_direction))
        object.__setattr__(self, "response_profile", tuple(self.response_profile))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ContextGatingCheckpoint:
    image_count: int
    source_record_count: int
    node_id: str
    query_record_id: str
    target_association_id: str
    baseline: ResponseMetrics
    gated: ResponseMetrics
    projector_prestress: ProjectorPrestress
    gated_response: ProjectorGatedResponse
    ambiguity_reduction: float
    interference_reduction: float
    margin_gain: float
    target_preserved: bool
    memory_unchanged: bool
    query_unchanged: bool
    policy_free: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ContextGatingSummary:
    checkpoint_count: int
    all_targets_preserved: bool
    all_memory_unchanged: bool
    all_queries_unchanged: bool
    all_policy_free: bool
    ambiguity_suppressed_all: bool
    interference_suppressed_all: bool
    margin_improved_all: bool
    mean_ambiguity_reduction: float
    mean_interference_reduction: float
    mean_margin_gain: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ContextGatingBenchmarkResult:
    scenario_id: str
    checkpoints: tuple[ContextGatingCheckpoint, ...]
    summary: ContextGatingSummary
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


def _metrics_from_profile(profile: Sequence[tuple[str, float]]) -> ResponseMetrics:
    ordered = tuple(sorted(
        ((str(k), max(0.0, float(v))) for k, v in profile),
        key=lambda item: (-item[1], item[0]),
    ))

    if ordered:
        dominant = ordered[0][0]
        top1 = ordered[0][1]
        top2 = ordered[1][1] if len(ordered) > 1 else 0.0
    else:
        dominant, top1, top2 = None, 0.0, 0.0

    margin = max(0.0, top1 - top2)
    ambiguity = min(1.0, top2 / top1) if top1 > EPSILON else 0.0
    energy = sum(v * v for _, v in ordered)
    competitor_energy = sum(v * v for k, v in ordered if k != dominant)
    interference = competitor_energy / energy if energy > EPSILON else 0.0

    return ResponseMetrics(
        dominant_association_id=dominant,
        dominant_activation=top1,
        second_activation=top2,
        dominant_margin=margin,
        ambiguity=ambiguity,
        response_energy=energy,
        interference_ratio=interference,
        metadata={"external_expected_label_used": False},
    )


def _baseline_response(node: TensorMemoryNode, query_record):
    q = _normalize(_subtract(_record_vector(query_record), node.centroid))
    transformed = _normalize(_matrix_vector(node.second_moment, q))
    profile = tuple(
        (
            p.image_id,
            max(0.0, _dot(transformed, p.direction)) * p.prestress_weight,
        )
        for p in node.image_prototypes
    )
    return q, transformed, profile, _metrics_from_profile(profile)


def build_projector_prestress(
    node: TensorMemoryNode,
    *,
    target_prototype_id: str,
    parallel_gain: float = DEFAULT_PARALLEL_GAIN,
    cross_gain: float = DEFAULT_CROSS_GAIN,
    orthogonal_gain: float = DEFAULT_ORTHOGONAL_GAIN,
) -> ProjectorPrestress:
    prototype = next(
        (p for p in node.image_prototypes if p.image_id == target_prototype_id),
        None,
    )
    if prototype is None:
        raise ContextGatingError(f"unknown prototype: {target_prototype_id}")

    for name, value in (
        ("parallel_gain", parallel_gain),
        ("cross_gain", cross_gain),
        ("orthogonal_gain", orthogonal_gain),
    ):
        if not isfinite(float(value)) or float(value) < 0.0:
            raise ContextGatingError(f"{name} must be finite and non-negative")

    return ProjectorPrestress(
        context_id=f"projector_gate::{target_prototype_id}",
        context_vector=_normalize(prototype.direction),
        target_prototype_id=target_prototype_id,
        parallel_gain=parallel_gain,
        cross_gain=cross_gain,
        orthogonal_gain=orthogonal_gain,
        metadata={
            "projector_level_prestress": True,
            "query_modified": False,
            "stored_memory_modified": False,
            "target_used_for_response_selection": False,
            "external_expected_label_used": False,
        },
    )


def build_effective_operator(
    node: TensorMemoryNode,
    prestress: ProjectorPrestress,
) -> tuple[tuple[float, ...], ...]:
    c = _normalize(prestress.context_vector)
    n = len(c)
    if n != len(node.centroid):
        raise ContextGatingError("context/node dimensions must match")

    P = _outer(c)
    I = _identity(n)
    Q = _matrix_subtract(I, P)
    T = node.second_moment

    PTP = _matmul(_matmul(P, T), P)
    PTQ = _matmul(_matmul(P, T), Q)
    QTP = _matmul(_matmul(Q, T), P)
    QTQ = _matmul(_matmul(Q, T), Q)

    return _matrix_add(
        _matrix_scale(PTP, prestress.parallel_gain),
        _matrix_scale(_matrix_add(PTQ, QTP), prestress.cross_gain),
        _matrix_scale(QTQ, prestress.orthogonal_gain),
    )


def respond_with_projector_prestress(
    node: TensorMemoryNode,
    query_record,
    prestress: ProjectorPrestress,
) -> ProjectorGatedResponse:
    q = _normalize(_subtract(_record_vector(query_record), node.centroid))
    effective = build_effective_operator(node, prestress)
    transformed = _normalize(_matrix_vector(effective, q))

    profile = tuple(
        (
            p.image_id,
            max(0.0, _dot(transformed, p.direction)) * p.prestress_weight,
        )
        for p in node.image_prototypes
    )
    metrics = _metrics_from_profile(profile)

    return ProjectorGatedResponse(
        query_record_id=query_record.trace_id,
        node_id=node.node_id,
        context_id=prestress.context_id,
        raw_query_direction=q,
        transformed_direction=transformed,
        response_profile=tuple(sorted(profile, key=lambda item: (-item[1], item[0]))),
        metrics=metrics,
        target_prototype_id=prestress.target_prototype_id,
        metadata={
            "same_physical_node": True,
            "same_query_input": True,
            "same_stored_memory": True,
            "projector_level_prestress": True,
            "query_modified": False,
            "learning_applied": False,
            "memory_mutated": False,
            "action_selected": False,
            "policy_modified": False,
            "target_used_for_response_selection": False,
            "external_expected_label_used": False,
        },
    )


def _node_signature(node: TensorMemoryNode):
    return (
        node.node_id,
        node.source_record_count,
        node.centroid,
        node.second_moment,
        tuple(
            (
                p.image_id,
                p.source_record_ids,
                p.centroid,
                p.direction,
                p.prestress_weight,
            )
            for p in node.image_prototypes
        ),
    )


def _query_signature(record):
    return (
        record.trace_id,
        record.sequence_id,
        record.pattern_id,
        record.signature.compact_vector(),
    )


def build_context_gating_checkpoint(
    *,
    image_count: int,
    history_per_image: int = DEFAULT_HISTORY_PER_IMAGE,
    query_index: int = DEFAULT_QUERY_INDEX,
    parallel_gain: float = DEFAULT_PARALLEL_GAIN,
    cross_gain: float = DEFAULT_CROSS_GAIN,
    orthogonal_gain: float = DEFAULT_ORTHOGONAL_GAIN,
) -> ContextGatingCheckpoint:
    node = build_tensor_memory_node(
        history_per_image=history_per_image,
        image_count=image_count,
    )
    query = build_tensor_query_record(
        query_index=query_index,
        image_index=0,
    )

    node_before = _node_signature(node)
    query_before = _query_signature(query)

    _, _, _, baseline = _baseline_response(node, query)
    if baseline.dominant_association_id is None:
        raise ContextGatingError("baseline produced no dominant association")

    target = baseline.dominant_association_id

    prestress = build_projector_prestress(
        node,
        target_prototype_id=target,
        parallel_gain=parallel_gain,
        cross_gain=cross_gain,
        orthogonal_gain=orthogonal_gain,
    )

    gated_response = respond_with_projector_prestress(
        node,
        query,
        prestress,
    )
    gated = gated_response.metrics

    return ContextGatingCheckpoint(
        image_count=image_count,
        source_record_count=node.source_record_count,
        node_id=node.node_id,
        query_record_id=query.trace_id,
        target_association_id=target,
        baseline=baseline,
        gated=gated,
        projector_prestress=prestress,
        gated_response=gated_response,
        ambiguity_reduction=baseline.ambiguity - gated.ambiguity,
        interference_reduction=baseline.interference_ratio - gated.interference_ratio,
        margin_gain=gated.dominant_margin - baseline.dominant_margin,
        target_preserved=(gated.dominant_association_id == target),
        memory_unchanged=(_node_signature(node) == node_before),
        query_unchanged=(_query_signature(query) == query_before),
        policy_free=(
            gated_response.metadata["action_selected"] is False
            and gated_response.metadata["policy_modified"] is False
            and gated_response.metadata["learning_applied"] is False
            and gated_response.metadata["memory_mutated"] is False
        ),
        metadata={
            "projector_level_prestress": True,
            "query_modified": False,
            "target_used_for_response_selection": False,
            "external_expected_label_used": False,
        },
    )


def summarize_context_gating(
    checkpoints: Sequence[ContextGatingCheckpoint],
) -> ContextGatingSummary:
    items = tuple(checkpoints)
    if not items:
        raise ContextGatingError("checkpoints must not be empty")
    n = len(items)

    return ContextGatingSummary(
        checkpoint_count=n,
        all_targets_preserved=all(x.target_preserved for x in items),
        all_memory_unchanged=all(x.memory_unchanged for x in items),
        all_queries_unchanged=all(x.query_unchanged for x in items),
        all_policy_free=all(x.policy_free for x in items),
        ambiguity_suppressed_all=all(x.ambiguity_reduction > 0.0 for x in items),
        interference_suppressed_all=all(x.interference_reduction > 0.0 for x in items),
        margin_improved_all=all(x.margin_gain > 0.0 for x in items),
        mean_ambiguity_reduction=sum(x.ambiguity_reduction for x in items) / n,
        mean_interference_reduction=sum(x.interference_reduction for x in items) / n,
        mean_margin_gain=sum(x.margin_gain for x in items) / n,
        metadata={
            "projector_level_prestress": True,
            "external_expected_label_used": False,
        },
    )


def run_context_gating_benchmark(
    *,
    image_counts: Sequence[int] = DEFAULT_IMAGE_COUNTS,
    history_per_image: int = DEFAULT_HISTORY_PER_IMAGE,
    query_index: int = DEFAULT_QUERY_INDEX,
    parallel_gain: float = DEFAULT_PARALLEL_GAIN,
    cross_gain: float = DEFAULT_CROSS_GAIN,
    orthogonal_gain: float = DEFAULT_ORTHOGONAL_GAIN,
) -> ContextGatingBenchmarkResult:
    counts = normalize_image_counts(image_counts)

    checkpoints = tuple(
        build_context_gating_checkpoint(
            image_count=n,
            history_per_image=history_per_image,
            query_index=query_index,
            parallel_gain=parallel_gain,
            cross_gain=cross_gain,
            orthogonal_gain=orthogonal_gain,
        )
        for n in counts
    )

    summary = summarize_context_gating(checkpoints)

    return ContextGatingBenchmarkResult(
        scenario_id=SCENARIO_ID,
        checkpoints=checkpoints,
        summary=summary,
        metadata={
            "benchmark_type": "projector_context_gating_interference_suppression",
            "projector_level_prestress": True,
            "same_node_within_each_checkpoint": True,
            "same_query_within_each_checkpoint": True,
            "same_memory_within_each_checkpoint": True,
            "query_modified_by_context": False,
            "learning_between_baseline_and_gate": False,
            "memory_mutated_between_baseline_and_gate": False,
            "memory_selects_action": False,
            "policy_override_enabled": False,
            "target_used_for_response_selection": False,
            "external_expected_label_used": False,
        },
    )


def image_count_series(result):
    return tuple(x.image_count for x in result.checkpoints)


def baseline_ambiguity_series(result):
    return tuple(x.baseline.ambiguity for x in result.checkpoints)


def gated_ambiguity_series(result):
    return tuple(x.gated.ambiguity for x in result.checkpoints)


def ambiguity_reduction_series(result):
    return tuple(x.ambiguity_reduction for x in result.checkpoints)


def baseline_interference_series(result):
    return tuple(x.baseline.interference_ratio for x in result.checkpoints)


def gated_interference_series(result):
    return tuple(x.gated.interference_ratio for x in result.checkpoints)


def interference_reduction_series(result):
    return tuple(x.interference_reduction for x in result.checkpoints)


def baseline_margin_series(result):
    return tuple(x.baseline.dominant_margin for x in result.checkpoints)


def gated_margin_series(result):
    return tuple(x.gated.dominant_margin for x in result.checkpoints)


def margin_gain_series(result):
    return tuple(x.margin_gain for x in result.checkpoints)


def target_preserved_series(result):
    return tuple(x.target_preserved for x in result.checkpoints)


def query_unchanged_series(result):
    return tuple(x.query_unchanged for x in result.checkpoints)


def context_gating_is_policy_free(result) -> bool:
    return (
        result.metadata.get("memory_selects_action") is False
        and result.metadata.get("policy_override_enabled") is False
        and result.metadata.get("learning_between_baseline_and_gate") is False
        and result.metadata.get("memory_mutated_between_baseline_and_gate") is False
        and result.metadata.get("query_modified_by_context") is False
        and all(x.policy_free for x in result.checkpoints)
    )


__all__ = [
    "ContextGatingBenchmarkResult",
    "ContextGatingCheckpoint",
    "ContextGatingError",
    "ContextGatingSummary",
    "DEFAULT_CROSS_GAIN",
    "DEFAULT_HISTORY_PER_IMAGE",
    "DEFAULT_IMAGE_COUNTS",
    "DEFAULT_ORTHOGONAL_GAIN",
    "DEFAULT_PARALLEL_GAIN",
    "DEFAULT_QUERY_INDEX",
    "ProjectorGatedResponse",
    "ProjectorPrestress",
    "ResponseMetrics",
    "SCENARIO_ID",
    "ambiguity_reduction_series",
    "baseline_ambiguity_series",
    "baseline_interference_series",
    "baseline_margin_series",
    "build_context_gating_checkpoint",
    "build_effective_operator",
    "build_projector_prestress",
    "context_gating_is_policy_free",
    "gated_ambiguity_series",
    "gated_interference_series",
    "gated_margin_series",
    "image_count_series",
    "interference_reduction_series",
    "margin_gain_series",
    "normalize_image_counts",
    "query_unchanged_series",
    "respond_with_projector_prestress",
    "run_context_gating_benchmark",
    "summarize_context_gating",
    "target_preserved_series",
]
