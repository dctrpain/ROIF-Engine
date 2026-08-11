"""
Tests for ROIF Memory 2.0 — Memory State Core.

Boundary under test:

    M_t + committed trace -> M_{t+1}

with:
    append-only semantics
    immutable snapshots
    no rewrite / delete / merge
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.history.memory_commitment import CommittedMemoryTrace
from roif.history.memory_integration import analyze_memory_integration
from roif.history.memory_state import (
    MemoryState,
    MemoryStateError,
    MemoryStateTransition,
    SCHEMA_VERSION,
    append_committed_trace,
    build_memory_state,
    contains_memory_trace,
    empty_memory_state,
    get_memory_trace,
    memory_state_is_policy_free,
    memory_state_signature,
    memory_state_transition_is_policy_free,
)


def make_trace(
    trace_id: str,
    *,
    attractor_id: str = "sens",
    vector=(1.0, 0.0, 0.0),
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
def trace_a():
    return make_trace(
        "trace_a",
        attractor_id="sens",
        vector=(1.0, 0.0, 0.0),
    )


@pytest.fixture
def trace_b():
    return make_trace(
        "trace_b",
        attractor_id="sens",
        vector=(0.99, 0.01, 0.0),
    )


@pytest.fixture
def trace_c():
    return make_trace(
        "trace_c",
        attractor_id="adapt",
        vector=(-1.0, 0.0, 0.0),
    )


@pytest.fixture
def empty_state():
    return empty_memory_state(
        state_id="m0",
    )


@pytest.fixture
def state_one(trace_a):
    return build_memory_state(
        state_id="m1",
        traces=(trace_a,),
        revision=1,
        parent_state_id="m0",
        appended_trace_id="trace_a",
    )


def test_schema_version():
    assert SCHEMA_VERSION == "memory_state_v1"


def test_empty_state_constructs(empty_state):
    assert isinstance(
        empty_state,
        MemoryState,
    )


def test_empty_state_revision_zero(empty_state):
    assert empty_state.revision == 0


def test_empty_state_has_no_traces(empty_state):
    assert empty_state.trace_count == 0
    assert empty_state.trace_ids == ()


def test_build_state_with_trace(trace_a):
    state = build_memory_state(
        state_id="x",
        traces=(trace_a,),
        revision=0,
    )

    assert state.trace_count == 1


def test_build_rejects_negative_revision():
    with pytest.raises(MemoryStateError):
        build_memory_state(
            state_id="bad",
            revision=-1,
        )


def test_build_rejects_empty_state_id():
    with pytest.raises(MemoryStateError):
        build_memory_state(
            state_id="",
        )


def test_build_rejects_duplicate_trace_ids(trace_a):
    with pytest.raises(MemoryStateError):
        build_memory_state(
            state_id="dup",
            traces=(trace_a, trace_a),
        )


def test_appended_trace_id_must_exist(trace_a):
    with pytest.raises(MemoryStateError):
        build_memory_state(
            state_id="bad_append",
            traces=(trace_a,),
            appended_trace_id="missing",
        )


def test_trace_count_property(state_one):
    assert state_one.trace_count == 1


def test_trace_ids_property(state_one):
    assert state_one.trace_ids == ("trace_a",)


def test_get_existing_trace(state_one, trace_a):
    assert get_memory_trace(
        state_one,
        "trace_a",
    ) == trace_a


def test_get_missing_trace_returns_none(state_one):
    assert get_memory_trace(
        state_one,
        "missing",
    ) is None


def test_contains_existing_trace(state_one):
    assert contains_memory_trace(
        state_one,
        "trace_a",
    ) is True


def test_contains_missing_trace(state_one):
    assert contains_memory_trace(
        state_one,
        "missing",
    ) is False


def test_append_returns_state_and_transition(empty_state, trace_a):
    target, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert isinstance(
        target,
        MemoryState,
    )
    assert isinstance(
        transition,
        MemoryStateTransition,
    )


def test_append_increments_revision(empty_state, trace_a):
    target, _ = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert target.revision == empty_state.revision + 1


def test_append_increments_trace_count(empty_state, trace_a):
    target, _ = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert target.trace_count == empty_state.trace_count + 1


def test_append_sets_parent_state_id(empty_state, trace_a):
    target, _ = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert target.parent_state_id == empty_state.state_id


def test_append_sets_appended_trace_id(empty_state, trace_a):
    target, _ = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert target.appended_trace_id == trace_a.trace_id


def test_append_preserves_old_trace_order(state_one, trace_b):
    target, _ = append_committed_trace(
        transition_id="t1_2",
        target_state_id="m2",
        state=state_one,
        trace=trace_b,
    )

    assert target.trace_ids == (
        "trace_a",
        "trace_b",
    )


def test_append_does_not_mutate_source_state(state_one, trace_b):
    before_signature = memory_state_signature(
        state_one
    )
    before_traces = state_one.traces

    append_committed_trace(
        transition_id="t1_2",
        target_state_id="m2",
        state=state_one,
        trace=trace_b,
    )

    assert memory_state_signature(
        state_one
    ) == before_signature
    assert state_one.traces == before_traces


def test_append_preserves_existing_trace_identity(state_one, trace_b):
    target, _ = append_committed_trace(
        transition_id="t1_2",
        target_state_id="m2",
        state=state_one,
        trace=trace_b,
    )

    assert target.traces[0] is state_one.traces[0]


def test_append_rejects_existing_trace_id(state_one, trace_a):
    with pytest.raises(MemoryStateError):
        append_committed_trace(
            transition_id="dup",
            target_state_id="m2",
            state=state_one,
            trace=trace_a,
        )


def test_transition_source_target_ids(empty_state, trace_a):
    target, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.source_state_id == "m0"
    assert transition.target_state_id == target.state_id


def test_transition_revision_chain(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.source_revision == 0
    assert transition.target_revision == 1


def test_transition_trace_count_chain(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.source_trace_count == 0
    assert transition.target_trace_count == 1


def test_transition_appended_trace_id(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.appended_trace_id == "trace_a"


def test_transition_declares_no_source_mutation(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.source_state_mutated is False


def test_transition_declares_no_rewrite(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.existing_trace_rewritten is False


def test_transition_declares_no_delete(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.trace_deleted is False


def test_transition_declares_no_merge(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert transition.trace_merged is False


def test_integration_can_be_attached(state_one, trace_b):
    integration = analyze_memory_integration(
        integration_id="integration_b",
        new_trace=trace_b,
        existing_traces=state_one.traces,
    )

    target, transition = append_committed_trace(
        transition_id="t1_2",
        target_state_id="m2",
        state=state_one,
        trace=trace_b,
        integration=integration,
    )

    assert (
        transition.integration_relation
        == integration.integration_relation
    )
    assert transition.cluster_hint == integration.cluster_hint
    assert (
        target.metadata["integration_relation"]
        == integration.integration_relation
    )


def test_integration_new_trace_must_match(state_one, trace_b, trace_c):
    integration = analyze_memory_integration(
        integration_id="integration_c",
        new_trace=trace_c,
        existing_traces=state_one.traces,
    )

    with pytest.raises(MemoryStateError):
        append_committed_trace(
            transition_id="bad_integration",
            target_state_id="m2",
            state=state_one,
            trace=trace_b,
            integration=integration,
        )


def test_integration_existing_count_must_match(empty_state, trace_a, trace_b):
    integration = analyze_memory_integration(
        integration_id="integration_b",
        new_trace=trace_b,
        existing_traces=(trace_a,),
    )

    with pytest.raises(MemoryStateError):
        append_committed_trace(
            transition_id="bad_count",
            target_state_id="m1",
            state=empty_state,
            trace=trace_b,
            integration=integration,
        )


def test_first_trace_integration(empty_state, trace_a):
    integration = analyze_memory_integration(
        integration_id="first",
        new_trace=trace_a,
        existing_traces=(),
    )

    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
        integration=integration,
    )

    assert transition.integration_relation == "first_trace"
    assert (
        transition.cluster_hint
        == "create_new_cluster_candidate"
    )


def test_reinforcement_integration(state_one, trace_b):
    integration = analyze_memory_integration(
        integration_id="reinforcement",
        new_trace=trace_b,
        existing_traces=state_one.traces,
    )

    _, transition = append_committed_trace(
        transition_id="t1_2",
        target_state_id="m2",
        state=state_one,
        trace=trace_b,
        integration=integration,
    )

    assert transition.integration_relation == "reinforcement_present"
    assert transition.cluster_hint == "same_cluster_candidate"


def test_state_metadata_declares_immutable_snapshot(state_one):
    assert state_one.metadata["memory_state_mode"] == "immutable_snapshot"


def test_state_metadata_declares_no_memory_mutation(state_one):
    assert state_one.metadata["memory_mutated"] is False


def test_state_metadata_declares_no_rewrite(state_one):
    assert state_one.metadata["existing_trace_rewritten"] is False


def test_state_metadata_declares_no_delete(state_one):
    assert state_one.metadata["trace_deleted"] is False


def test_state_metadata_declares_no_merge(state_one):
    assert state_one.metadata["trace_merged"] is False


def test_state_declares_no_action(state_one):
    assert state_one.metadata["action_selected"] is False


def test_state_declares_no_policy_change(state_one):
    assert state_one.metadata["policy_modified"] is False


def test_state_declares_no_diagnosis(state_one):
    assert state_one.metadata["diagnosis_generated"] is False


def test_state_declares_no_biological_claim(state_one):
    assert state_one.metadata["biological_memory_claimed"] is False


def test_state_declares_no_causal_truth(state_one):
    assert state_one.metadata["causal_truth_inferred"] is False


def test_memory_state_is_policy_free(state_one):
    assert memory_state_is_policy_free(
        state_one
    ) is True


def test_transition_is_policy_free(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert memory_state_transition_is_policy_free(
        transition
    ) is True


def test_signature_regression(state_one):
    assert memory_state_signature(
        state_one
    ) == (
        "m1",
        1,
        ("trace_a",),
    )


def test_state_is_frozen(state_one):
    with pytest.raises(FrozenInstanceError):
        state_one.revision = 9  # type: ignore[misc]


def test_transition_is_frozen(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    with pytest.raises(FrozenInstanceError):
        transition.target_revision = 9  # type: ignore[misc]


def test_state_traces_are_tuple(state_one):
    assert isinstance(
        state_one.traces,
        tuple,
    )


def test_state_metadata_is_read_only(state_one):
    assert isinstance(
        state_one.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        state_one.metadata["x"] = 1  # type: ignore[index]


def test_transition_metadata_is_read_only(empty_state, trace_a):
    _, transition = append_committed_trace(
        transition_id="t0_1",
        target_state_id="m1",
        state=empty_state,
        trace=trace_a,
    )

    assert isinstance(
        transition.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        transition.metadata["x"] = 1  # type: ignore[index]


def test_empty_state_is_deterministic():
    left = empty_memory_state(
        state_id="same",
    )
    right = empty_memory_state(
        state_id="same",
    )

    assert left == right


def test_build_state_is_deterministic(trace_a):
    left = build_memory_state(
        state_id="same",
        traces=(trace_a,),
        revision=1,
    )
    right = build_memory_state(
        state_id="same",
        traces=(trace_a,),
        revision=1,
    )

    assert left == right


def test_append_is_deterministic(state_one, trace_b):
    left = append_committed_trace(
        transition_id="same_transition",
        target_state_id="same_target",
        state=state_one,
        trace=trace_b,
    )
    right = append_committed_trace(
        transition_id="same_transition",
        target_state_id="same_target",
        state=state_one,
        trace=trace_b,
    )

    assert left == right


def test_append_with_integration_is_deterministic(state_one, trace_b):
    integration = analyze_memory_integration(
        integration_id="same_integration",
        new_trace=trace_b,
        existing_traces=state_one.traces,
    )

    left = append_committed_trace(
        transition_id="same_transition",
        target_state_id="same_target",
        state=state_one,
        trace=trace_b,
        integration=integration,
    )
    right = append_committed_trace(
        transition_id="same_transition",
        target_state_id="same_target",
        state=state_one,
        trace=trace_b,
        integration=integration,
    )

    assert left == right


def test_lookup_is_deterministic(state_one):
    assert get_memory_trace(
        state_one,
        "trace_a",
    ) == get_memory_trace(
        state_one,
        "trace_a",
    )


def test_signature_is_deterministic(state_one):
    assert memory_state_signature(
        state_one
    ) == memory_state_signature(
        state_one
    )
