"""
Tests for ROIF Memory 2.0 — Memory Integration Core.

Boundary under test:

    new committed trace + existing committed traces
        -> descriptive integration analysis

without:
    merge
    rewrite
    delete
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.history.memory_commitment import CommittedMemoryTrace
from roif.history.memory_integration import (
    MemoryIntegrationConfig,
    MemoryIntegrationError,
    MemoryIntegrationResult,
    MemoryTraceRelation,
    SCHEMA_VERSION,
    analyze_memory_integration,
    classify_memory_relation,
    compare_memory_traces,
    memory_integration_is_policy_free,
)


def make_trace(
    trace_id: str,
    *,
    attractor_id: str,
    vector,
    score: float = 0.8,
    confidence: float = 0.8,
):
    return CommittedMemoryTrace(
        trace_id=trace_id,
        source_candidate_id=f"{trace_id}_candidate",
        source_trajectory_id=f"{trace_id}_trajectory",
        source_stability_analysis_id=f"{trace_id}_stability",
        dominant_attractor_id=attractor_id,
        representation_vector=tuple(vector),
        consolidation_score=score,
        commitment_confidence=confidence,
        commitment_applied=True,
        source_memory_mutated=False,
        existing_memory_rewritten=False,
        metadata={
            "action_selected": False,
            "policy_modified": False,
            "source_memory_mutated": False,
            "existing_memory_rewritten": False,
        },
    )


@pytest.fixture
def new_trace():
    return make_trace(
        "new",
        attractor_id="sens",
        vector=(1.0, 0.0, 0.0),
    )


@pytest.fixture
def reinforce_trace():
    return make_trace(
        "reinforce",
        attractor_id="sens",
        vector=(0.99, 0.01, 0.0),
    )


@pytest.fixture
def conflict_trace():
    return make_trace(
        "conflict",
        attractor_id="adapt",
        vector=(0.99, 0.01, 0.0),
    )


@pytest.fixture
def related_trace():
    return make_trace(
        "related",
        attractor_id="sens",
        vector=(0.7, 0.7, 0.0),
    )


@pytest.fixture
def novel_trace():
    return make_trace(
        "novel",
        attractor_id="adapt",
        vector=(-1.0, 0.0, 0.0),
    )


def test_schema_version():
    assert SCHEMA_VERSION == "memory_integration_v1"


def test_default_config_constructs():
    assert isinstance(
        MemoryIntegrationConfig(),
        MemoryIntegrationConfig,
    )


def test_config_rejects_invalid_threshold_order():
    with pytest.raises(MemoryIntegrationError):
        MemoryIntegrationConfig(
            novelty_similarity_threshold=0.9,
            related_similarity_threshold=0.6,
            reinforcement_similarity_threshold=0.8,
        )


def test_config_rejects_negative_threshold():
    with pytest.raises(MemoryIntegrationError):
        MemoryIntegrationConfig(
            novelty_similarity_threshold=-0.1,
        )


def test_config_rejects_threshold_above_one():
    with pytest.raises(MemoryIntegrationError):
        MemoryIntegrationConfig(
            conflict_similarity_threshold=1.1,
        )


def test_compare_returns_relation(new_trace, reinforce_trace):
    result = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert isinstance(
        result,
        MemoryTraceRelation,
    )


def test_reinforcement_relation(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert relation.relation == "reinforcement"


def test_reinforcement_same_attractor_true(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert relation.same_dominant_attractor is True


def test_reinforcement_score_positive(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert relation.reinforcement_score > 0.0


def test_reinforcement_conflict_score_zero(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert relation.conflict_score == pytest.approx(0.0)


def test_conflict_relation(new_trace, conflict_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=conflict_trace,
    )

    assert relation.relation == "conflict"


def test_conflict_same_attractor_false(new_trace, conflict_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=conflict_trace,
    )

    assert relation.same_dominant_attractor is False


def test_conflict_score_positive(new_trace, conflict_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=conflict_trace,
    )

    assert relation.conflict_score > 0.0


def test_conflict_reinforcement_score_zero(new_trace, conflict_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=conflict_trace,
    )

    assert relation.reinforcement_score == pytest.approx(0.0)


def test_related_relation(new_trace, related_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=related_trace,
    )

    assert relation.relation in {
        "related",
        "reinforcement",
    }


def test_novel_relation(new_trace, novel_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=novel_trace,
    )

    assert relation.relation == "novel"


def test_novelty_score_high_for_opposite_vector(new_trace, novel_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=novel_trace,
    )

    assert relation.novelty_score == pytest.approx(1.0)


def test_similarity_score_bounded(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert 0.0 <= relation.similarity_score <= 1.0


def test_cosine_similarity_bounded(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert -1.0 <= relation.cosine_similarity <= 1.0


def test_vector_distance_nonnegative(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert relation.vector_distance >= 0.0


def test_identical_vector_similarity_is_one(new_trace):
    same_vector_trace = make_trace(
        "same_vector",
        attractor_id="sens",
        vector=(1.0, 0.0, 0.0),
    )

    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=same_vector_trace,
    )

    assert relation.similarity_score == pytest.approx(1.0)


def test_opposite_vector_similarity_is_zero(new_trace, novel_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=novel_trace,
    )

    assert relation.similarity_score == pytest.approx(0.0)


def test_classify_reinforcement():
    config = MemoryIntegrationConfig()

    assert classify_memory_relation(
        similarity_score=0.95,
        same_dominant_attractor=True,
        config=config,
    ) == "reinforcement"


def test_classify_conflict():
    config = MemoryIntegrationConfig()

    assert classify_memory_relation(
        similarity_score=0.90,
        same_dominant_attractor=False,
        config=config,
    ) == "conflict"


def test_classify_related():
    config = MemoryIntegrationConfig()

    assert classify_memory_relation(
        similarity_score=0.75,
        same_dominant_attractor=True,
        config=config,
    ) == "related"


def test_classify_novel():
    config = MemoryIntegrationConfig()

    assert classify_memory_relation(
        similarity_score=0.20,
        same_dominant_attractor=True,
        config=config,
    ) == "novel"


def test_classify_weakly_related():
    config = MemoryIntegrationConfig()

    assert classify_memory_relation(
        similarity_score=0.55,
        same_dominant_attractor=True,
        config=config,
    ) == "weakly_related"


def test_conflict_requires_mismatch_by_default():
    config = MemoryIntegrationConfig()

    assert classify_memory_relation(
        similarity_score=0.95,
        same_dominant_attractor=True,
        config=config,
    ) != "conflict"


def test_conflict_mismatch_requirement_can_be_disabled():
    config = MemoryIntegrationConfig(
        require_attractor_mismatch_for_conflict=False,
    )

    assert classify_memory_relation(
        similarity_score=0.90,
        same_dominant_attractor=True,
        config=config,
    ) == "conflict"


def test_first_trace_analysis(new_trace):
    result = analyze_memory_integration(
        integration_id="first",
        new_trace=new_trace,
        existing_traces=(),
    )

    assert isinstance(
        result,
        MemoryIntegrationResult,
    )
    assert result.integration_relation == "first_trace"
    assert result.cluster_hint == "create_new_cluster_candidate"


def test_first_trace_has_no_nearest_trace(new_trace):
    result = analyze_memory_integration(
        integration_id="first",
        new_trace=new_trace,
        existing_traces=(),
    )

    assert result.nearest_trace_id is None
    assert result.nearest_similarity_score == pytest.approx(0.0)


def test_reinforcement_analysis(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="reinforcement",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.integration_relation == "reinforcement_present"
    assert result.cluster_hint == "same_cluster_candidate"
    assert result.reinforcement_count == 1


def test_conflict_analysis(new_trace, conflict_trace):
    result = analyze_memory_integration(
        integration_id="conflict",
        new_trace=new_trace,
        existing_traces=(conflict_trace,),
    )

    assert result.integration_relation == "conflict_present"
    assert result.cluster_hint == "preserve_separate_conflicting_trace"
    assert result.conflict_count == 1


def test_novel_analysis(new_trace, novel_trace):
    result = analyze_memory_integration(
        integration_id="novel",
        new_trace=new_trace,
        existing_traces=(novel_trace,),
    )

    assert result.integration_relation == "novel_trace"
    assert result.cluster_hint == "create_new_cluster_candidate"
    assert result.novel_count == 1


def test_related_analysis(new_trace, related_trace):
    result = analyze_memory_integration(
        integration_id="related",
        new_trace=new_trace,
        existing_traces=(related_trace,),
    )

    assert result.integration_relation in {
        "reinforcement_present",
        "related_trace",
    }


def test_nearest_trace_is_most_similar(
    new_trace,
    reinforce_trace,
    related_trace,
    novel_trace,
):
    result = analyze_memory_integration(
        integration_id="nearest",
        new_trace=new_trace,
        existing_traces=(
            novel_trace,
            related_trace,
            reinforce_trace,
        ),
    )

    assert result.nearest_trace_id == "reinforce"


def test_existing_trace_count_matches(
    new_trace,
    reinforce_trace,
    related_trace,
    novel_trace,
):
    result = analyze_memory_integration(
        integration_id="count",
        new_trace=new_trace,
        existing_traces=(
            reinforce_trace,
            related_trace,
            novel_trace,
        ),
    )

    assert result.existing_trace_count == 3
    assert len(result.relations) == 3


def test_relation_counts_partition_existing_traces(
    new_trace,
    reinforce_trace,
    conflict_trace,
    related_trace,
    novel_trace,
):
    result = analyze_memory_integration(
        integration_id="partition",
        new_trace=new_trace,
        existing_traces=(
            reinforce_trace,
            conflict_trace,
            related_trace,
            novel_trace,
        ),
    )

    total = (
        result.reinforcement_count
        + result.conflict_count
        + result.related_count
        + result.novel_count
    )

    assert total == result.existing_trace_count


def test_conflict_has_priority_over_reinforcement(
    new_trace,
    reinforce_trace,
    conflict_trace,
):
    result = analyze_memory_integration(
        integration_id="priority",
        new_trace=new_trace,
        existing_traces=(
            reinforce_trace,
            conflict_trace,
        ),
    )

    assert result.integration_relation == "conflict_present"


def test_self_integration_rejected(new_trace):
    with pytest.raises(MemoryIntegrationError):
        analyze_memory_integration(
            integration_id="self",
            new_trace=new_trace,
            existing_traces=(new_trace,),
        )


def test_duplicate_existing_trace_ids_rejected(new_trace, reinforce_trace):
    duplicate = make_trace(
        "reinforce",
        attractor_id="adapt",
        vector=(0.0, 1.0, 0.0),
    )

    with pytest.raises(MemoryIntegrationError):
        analyze_memory_integration(
            integration_id="duplicates",
            new_trace=new_trace,
            existing_traces=(
                reinforce_trace,
                duplicate,
            ),
        )


def test_dimension_mismatch_rejected(new_trace):
    mismatch = make_trace(
        "mismatch",
        attractor_id="sens",
        vector=(1.0, 0.0),
    )

    with pytest.raises(MemoryIntegrationError):
        compare_memory_traces(
            new_trace=new_trace,
            existing_trace=mismatch,
        )


def test_relation_metadata_declares_no_rewrite(
    new_trace,
    reinforce_trace,
):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert relation.metadata["existing_memory_rewritten"] is False
    assert relation.metadata["trace_merged"] is False
    assert relation.metadata["trace_deleted"] is False


def test_result_metadata_declares_analysis_only(
    new_trace,
    reinforce_trace,
):
    result = analyze_memory_integration(
        integration_id="metadata",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.metadata["memory_integration_mode"] == "analysis_only"
    assert result.metadata["existing_memory_rewritten"] is False
    assert result.metadata["trace_merged"] is False
    assert result.metadata["trace_deleted"] is False


def test_result_declares_no_learning(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="metadata",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.metadata["learning_applied"] is False


def test_result_declares_no_action(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="metadata",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.metadata["action_selected"] is False


def test_result_declares_no_policy_change(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="metadata",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.metadata["policy_modified"] is False


def test_result_declares_no_diagnosis(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="metadata",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.metadata["diagnosis_generated"] is False


def test_result_declares_no_biological_claim(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="metadata",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.metadata["biological_integration_claimed"] is False


def test_result_declares_no_causal_truth(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="metadata",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert result.metadata["causal_truth_inferred"] is False


def test_memory_integration_is_policy_free(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="policy",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert memory_integration_is_policy_free(result) is True


def test_compare_does_not_mutate_new_trace(new_trace, reinforce_trace):
    before = new_trace.representation_vector

    compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert new_trace.representation_vector == before


def test_compare_does_not_mutate_existing_trace(new_trace, reinforce_trace):
    before = reinforce_trace.representation_vector

    compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert reinforce_trace.representation_vector == before


def test_analysis_does_not_mutate_existing_collection(
    new_trace,
    reinforce_trace,
    related_trace,
):
    existing = (
        reinforce_trace,
        related_trace,
    )
    before = tuple(existing)

    analyze_memory_integration(
        integration_id="immutability",
        new_trace=new_trace,
        existing_traces=existing,
    )

    assert existing == before


def test_config_is_frozen():
    config = MemoryIntegrationConfig()

    with pytest.raises(FrozenInstanceError):
        config.related_similarity_threshold = 0.0  # type: ignore[misc]


def test_relation_is_frozen(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    with pytest.raises(FrozenInstanceError):
        relation.relation = "changed"  # type: ignore[misc]


def test_result_is_frozen(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="frozen",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    with pytest.raises(FrozenInstanceError):
        result.integration_relation = "changed"  # type: ignore[misc]


def test_config_metadata_is_read_only():
    config = MemoryIntegrationConfig(
        metadata={"x": 1},
    )

    assert isinstance(
        config.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        config.metadata["x"] = 2  # type: ignore[index]


def test_relation_metadata_is_read_only(new_trace, reinforce_trace):
    relation = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert isinstance(
        relation.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        relation.metadata["x"] = 1  # type: ignore[index]


def test_result_metadata_is_read_only(new_trace, reinforce_trace):
    result = analyze_memory_integration(
        integration_id="readonly",
        new_trace=new_trace,
        existing_traces=(reinforce_trace,),
    )

    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_compare_is_deterministic(new_trace, reinforce_trace):
    left = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    right = compare_memory_traces(
        new_trace=new_trace,
        existing_trace=reinforce_trace,
    )

    assert left == right


def test_analysis_is_deterministic(
    new_trace,
    reinforce_trace,
    conflict_trace,
    related_trace,
    novel_trace,
):
    existing = (
        reinforce_trace,
        conflict_trace,
        related_trace,
        novel_trace,
    )

    left = analyze_memory_integration(
        integration_id="same",
        new_trace=new_trace,
        existing_traces=existing,
    )

    right = analyze_memory_integration(
        integration_id="same",
        new_trace=new_trace,
        existing_traces=existing,
    )

    assert left == right


def test_classification_is_deterministic():
    config = MemoryIntegrationConfig()

    args = dict(
        similarity_score=0.75,
        same_dominant_attractor=True,
        config=config,
    )

    assert classify_memory_relation(
        **args
    ) == classify_memory_relation(
        **args
    )


def test_nearest_selection_is_deterministic(
    new_trace,
    reinforce_trace,
    related_trace,
):
    existing = (
        related_trace,
        reinforce_trace,
    )

    left = analyze_memory_integration(
        integration_id="nearest_same",
        new_trace=new_trace,
        existing_traces=existing,
    )

    right = analyze_memory_integration(
        integration_id="nearest_same",
        new_trace=new_trace,
        existing_traces=existing,
    )

    assert left.nearest_trace_id == right.nearest_trace_id
    assert (
        left.nearest_similarity_score
        == right.nearest_similarity_score
    )
