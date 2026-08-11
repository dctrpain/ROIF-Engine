"""
ROIF Memory 2.0
Memory Integration Core

Twenty-first layer above:
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
    ContextualDynamics
    ContextualTrajectory
    ContextualReconstruction
    ReconstructionTrajectory
    ReconstructionFeedback
    FeedbackTrajectory
    FeedbackStability
    ExperienceConsolidation
    MemoryCommitment

Purpose
-------
Analyze how a newly committed immutable memory trace relates to an existing
set of committed traces.

The integration layer is descriptive only. It computes:
- vector similarity
- vector distance
- dominant-attractor agreement
- reinforcement evidence
- conflict evidence
- novelty evidence
- nearest existing trace
- integration relation labels
- cluster hint

Important boundary:
    integration analysis != memory rewrite

This module does NOT:
- merge traces
- delete traces
- mutate traces
- rewrite existing memory
- automatically learn
- select actions
- modify policy
- diagnose
- claim biological integration mechanisms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.memory_commitment import (
    CommittedMemoryTrace,
    memory_commitment_is_policy_free,
)


SCHEMA_VERSION = "memory_integration_v1"


class MemoryIntegrationError(ValueError):
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
        raise MemoryIntegrationError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise MemoryIntegrationError(
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
        raise MemoryIntegrationError(
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
        raise MemoryIntegrationError(
            f"{name} must be within [0,1]"
        )

    return x


def _vector(
    name: str,
    values: Sequence[float],
) -> tuple[float, ...]:
    result = tuple(
        _validate_finite(
            f"{name}[{index}]",
            value,
        )
        for index, value in enumerate(values)
    )

    if not result:
        raise MemoryIntegrationError(
            f"{name} must not be empty"
        )

    return result


def _distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise MemoryIntegrationError(
            "trace representation dimensions must match"
        )

    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(
                left,
                right,
            )
        )
    )


def _norm(
    vector: Sequence[float],
) -> float:
    return sqrt(
        sum(
            float(value) ** 2
            for value in vector
        )
    )


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise MemoryIntegrationError(
            "trace representation dimensions must match"
        )

    left_norm = _norm(left)
    right_norm = _norm(right)

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    dot = sum(
        float(a) * float(b)
        for a, b in zip(
            left,
            right,
        )
    )

    cosine = dot / (
        left_norm
        * right_norm
    )

    return max(
        -1.0,
        min(
            1.0,
            cosine,
        ),
    )


def _similarity_01(
    cosine_similarity: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            (
                cosine_similarity
                + 1.0
            )
            / 2.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class MemoryIntegrationConfig:
    reinforcement_similarity_threshold: float = 0.85
    related_similarity_threshold: float = 0.65
    novelty_similarity_threshold: float = 0.45

    conflict_similarity_threshold: float = 0.70
    require_attractor_mismatch_for_conflict: bool = True

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "reinforcement_similarity_threshold",
            "related_similarity_threshold",
            "novelty_similarity_threshold",
            "conflict_similarity_threshold",
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

        if not (
            self.novelty_similarity_threshold
            <= self.related_similarity_threshold
            <= self.reinforcement_similarity_threshold
        ):
            raise MemoryIntegrationError(
                "similarity thresholds must satisfy novelty <= related <= reinforcement"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class MemoryTraceRelation:
    existing_trace_id: str

    vector_distance: float
    cosine_similarity: float
    similarity_score: float

    same_dominant_attractor: bool

    reinforcement_score: float
    conflict_score: float
    novelty_score: float

    relation: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "existing_trace_id",
            _validate_id(
                "existing_trace_id",
                self.existing_trace_id,
            ),
        )

        object.__setattr__(
            self,
            "vector_distance",
            _validate_nonnegative(
                "vector_distance",
                self.vector_distance,
            ),
        )

        object.__setattr__(
            self,
            "cosine_similarity",
            _validate_finite(
                "cosine_similarity",
                self.cosine_similarity,
            ),
        )

        if not -1.0 <= self.cosine_similarity <= 1.0:
            raise MemoryIntegrationError(
                "cosine_similarity must be within [-1,1]"
            )

        for name in (
            "similarity_score",
            "reinforcement_score",
            "conflict_score",
            "novelty_score",
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
            "relation",
            _validate_id(
                "relation",
                self.relation,
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
class MemoryIntegrationResult:
    integration_id: str
    new_trace_id: str

    existing_trace_count: int
    relations: tuple[
        MemoryTraceRelation,
        ...,
    ]

    nearest_trace_id: str | None
    nearest_similarity_score: float

    reinforcement_count: int
    conflict_count: int
    related_count: int
    novel_count: int

    integration_relation: str
    cluster_hint: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integration_id",
            _validate_id(
                "integration_id",
                self.integration_id,
            ),
        )

        object.__setattr__(
            self,
            "new_trace_id",
            _validate_id(
                "new_trace_id",
                self.new_trace_id,
            ),
        )

        if self.existing_trace_count < 0:
            raise MemoryIntegrationError(
                "existing_trace_count must be non-negative"
            )

        object.__setattr__(
            self,
            "relations",
            tuple(
                self.relations
            ),
        )

        if len(
            self.relations
        ) != self.existing_trace_count:
            raise MemoryIntegrationError(
                "relations count must match existing_trace_count"
            )

        if self.nearest_trace_id is not None:
            object.__setattr__(
                self,
                "nearest_trace_id",
                _validate_id(
                    "nearest_trace_id",
                    self.nearest_trace_id,
                ),
            )

        object.__setattr__(
            self,
            "nearest_similarity_score",
            _validate_unit_interval(
                "nearest_similarity_score",
                self.nearest_similarity_score,
            ),
        )

        for name in (
            "reinforcement_count",
            "conflict_count",
            "related_count",
            "novel_count",
        ):
            if int(
                getattr(
                    self,
                    name,
                )
            ) < 0:
                raise MemoryIntegrationError(
                    f"{name} must be non-negative"
                )

        object.__setattr__(
            self,
            "integration_relation",
            _validate_id(
                "integration_relation",
                self.integration_relation,
            ),
        )

        object.__setattr__(
            self,
            "cluster_hint",
            _validate_id(
                "cluster_hint",
                self.cluster_hint,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def classify_memory_relation(
    *,
    similarity_score: float,
    same_dominant_attractor: bool,
    config: MemoryIntegrationConfig,
) -> str:
    similarity_score = _validate_unit_interval(
        "similarity_score",
        similarity_score,
    )

    conflict_condition = (
        similarity_score
        >= config.conflict_similarity_threshold
        and (
            not same_dominant_attractor
            if config.require_attractor_mismatch_for_conflict
            else True
        )
    )

    if conflict_condition:
        return "conflict"

    if (
        same_dominant_attractor
        and similarity_score
        >= config.reinforcement_similarity_threshold
    ):
        return "reinforcement"

    if (
        similarity_score
        >= config.related_similarity_threshold
    ):
        return "related"

    if (
        similarity_score
        <= config.novelty_similarity_threshold
    ):
        return "novel"

    return "weakly_related"


def compare_memory_traces(
    *,
    new_trace: CommittedMemoryTrace,
    existing_trace: CommittedMemoryTrace,
    config: MemoryIntegrationConfig | None = None,
) -> MemoryTraceRelation:
    if not memory_commitment_is_policy_free(
        new_trace
    ):
        raise MemoryIntegrationError(
            "new committed trace must be policy-free"
        )

    if not memory_commitment_is_policy_free(
        existing_trace
    ):
        raise MemoryIntegrationError(
            "existing committed trace must be policy-free"
        )

    actual_config = (
        config
        if config is not None
        else MemoryIntegrationConfig()
    )

    new_vector = _vector(
        "new_trace.representation_vector",
        new_trace.representation_vector,
    )

    existing_vector = _vector(
        "existing_trace.representation_vector",
        existing_trace.representation_vector,
    )

    distance = _distance(
        new_vector,
        existing_vector,
    )

    cosine = _cosine_similarity(
        new_vector,
        existing_vector,
    )

    similarity = _similarity_01(
        cosine
    )

    same_attractor = (
        new_trace.dominant_attractor_id
        == existing_trace.dominant_attractor_id
    )

    reinforcement_score = (
        similarity
        if same_attractor
        else 0.0
    )

    conflict_score = (
        similarity
        if not same_attractor
        else 0.0
    )

    novelty_score = max(
        0.0,
        min(
            1.0,
            1.0
            - similarity,
        ),
    )

    relation = classify_memory_relation(
        similarity_score=similarity,
        same_dominant_attractor=same_attractor,
        config=actual_config,
    )

    return MemoryTraceRelation(
        existing_trace_id=(
            existing_trace.trace_id
        ),
        vector_distance=distance,
        cosine_similarity=cosine,
        similarity_score=similarity,
        same_dominant_attractor=(
            same_attractor
        ),
        reinforcement_score=(
            reinforcement_score
        ),
        conflict_score=(
            conflict_score
        ),
        novelty_score=(
            novelty_score
        ),
        relation=relation,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "integration_relation_mode": "descriptive",
            "source_memory_mutated": False,
            "existing_memory_rewritten": False,
            "trace_merged": False,
            "trace_deleted": False,
            "learning_applied": False,
            "causal_truth_inferred": False,
        },
    )


def analyze_memory_integration(
    *,
    integration_id: str,
    new_trace: CommittedMemoryTrace,
    existing_traces: Iterable[
        CommittedMemoryTrace
    ],
    config: MemoryIntegrationConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MemoryIntegrationResult:
    if not memory_commitment_is_policy_free(
        new_trace
    ):
        raise MemoryIntegrationError(
            "new committed trace must be policy-free"
        )

    existing_items = tuple(
        existing_traces
    )

    if any(
        trace.trace_id
        == new_trace.trace_id
        for trace in existing_items
    ):
        raise MemoryIntegrationError(
            "new trace must not already exist in existing_traces"
        )

    if len(
        {
            trace.trace_id
            for trace in existing_items
        }
    ) != len(
        existing_items
    ):
        raise MemoryIntegrationError(
            "existing trace IDs must be unique"
        )

    actual_config = (
        config
        if config is not None
        else MemoryIntegrationConfig()
    )

    relations = tuple(
        compare_memory_traces(
            new_trace=new_trace,
            existing_trace=existing_trace,
            config=actual_config,
        )
        for existing_trace in existing_items
    )

    if relations:
        nearest = sorted(
            relations,
            key=lambda item: (
                -item.similarity_score,
                item.vector_distance,
                item.existing_trace_id,
            ),
        )[0]

        nearest_trace_id = (
            nearest.existing_trace_id
        )

        nearest_similarity = (
            nearest.similarity_score
        )
    else:
        nearest_trace_id = None
        nearest_similarity = 0.0

    reinforcement_count = sum(
        1
        for relation in relations
        if relation.relation
        == "reinforcement"
    )

    conflict_count = sum(
        1
        for relation in relations
        if relation.relation
        == "conflict"
    )

    related_count = sum(
        1
        for relation in relations
        if relation.relation
        in {
            "related",
            "weakly_related",
        }
    )

    novel_count = sum(
        1
        for relation in relations
        if relation.relation
        == "novel"
    )

    if not relations:
        integration_relation = (
            "first_trace"
        )
        cluster_hint = (
            "create_new_cluster_candidate"
        )
    elif conflict_count > 0:
        integration_relation = (
            "conflict_present"
        )
        cluster_hint = (
            "preserve_separate_conflicting_trace"
        )
    elif reinforcement_count > 0:
        integration_relation = (
            "reinforcement_present"
        )
        cluster_hint = (
            "same_cluster_candidate"
        )
    elif novel_count == len(
        relations
    ):
        integration_relation = (
            "novel_trace"
        )
        cluster_hint = (
            "create_new_cluster_candidate"
        )
    else:
        integration_relation = (
            "related_trace"
        )
        cluster_hint = (
            "related_cluster_candidate"
        )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "memory_integration_mode": "analysis_only",
        "new_trace_id": new_trace.trace_id,
        "source_memory_mutated": False,
        "existing_memory_rewritten": False,
        "trace_merged": False,
        "trace_deleted": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_integration_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return MemoryIntegrationResult(
        integration_id=integration_id,
        new_trace_id=new_trace.trace_id,
        existing_trace_count=len(
            existing_items
        ),
        relations=relations,
        nearest_trace_id=nearest_trace_id,
        nearest_similarity_score=(
            nearest_similarity
        ),
        reinforcement_count=(
            reinforcement_count
        ),
        conflict_count=conflict_count,
        related_count=related_count,
        novel_count=novel_count,
        integration_relation=(
            integration_relation
        ),
        cluster_hint=cluster_hint,
        metadata=merged_metadata,
    )


def memory_integration_is_policy_free(
    result: MemoryIntegrationResult,
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
            "source_memory_mutated"
        )
        is False
        and result.metadata.get(
            "existing_memory_rewritten"
        )
        is False
        and result.metadata.get(
            "trace_merged"
        )
        is False
        and result.metadata.get(
            "trace_deleted"
        )
        is False
    )


__all__ = [
    "MemoryIntegrationConfig",
    "MemoryIntegrationError",
    "MemoryIntegrationResult",
    "MemoryTraceRelation",
    "SCHEMA_VERSION",
    "analyze_memory_integration",
    "classify_memory_relation",
    "compare_memory_traces",
    "memory_integration_is_policy_free",
]
