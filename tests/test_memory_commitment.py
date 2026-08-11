"""
Tests for ROIF Memory 2.0 — Memory Commitment Core.

Boundary under test:

    eligible consolidation candidate
    + explicit commitment request
    + score gate
        -> new immutable committed trace

while:

    existing memory is not rewritten
    source objects are not mutated
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.history.experience_transformation import (
    BodyResponse,
    ExperienceEvent,
    SystemResponse,
    SystemState,
    build_experience_transformation,
)
from roif.history.experience_sequence import build_experience_sequence
from roif.history.experience_pattern import build_experience_pattern
from roif.history.experience_attractor import build_experience_attractor
from roif.history.attractor_dynamics import AttractorPrestress, DynamicsState
from roif.history.attractor_trajectory import build_attractor_trajectory
from roif.history.experience_conditioning import (
    ConditioningCue,
    build_conditioning_memory,
    observation_from_trajectory,
)
from roif.history.associative_context import (
    AssociativeCueInput,
    build_associative_context,
)
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    build_feedback_trajectory,
)
from roif.history.feedback_stability import (
    analyze_feedback_stability,
)
from roif.history.experience_consolidation import (
    ExperienceConsolidationConfig,
    evaluate_experience_consolidation,
)
from roif.history.memory_commitment import (
    CommittedMemoryTrace,
    MemoryCommitmentConfig,
    MemoryCommitmentError,
    MemoryCommitmentGate,
    SCHEMA_VERSION,
    commit_memory_trace,
    evaluate_memory_commitment_gate,
    memory_commitment_is_policy_free,
)


def make_state(state_id: str, *, cognitive, physiological, reserve: float):
    return SystemState(
        state_id=state_id,
        cognitive=tuple(cognitive),
        physiological=tuple(physiological),
        contextual=(0.5, 0.2),
        reserve=reserve,
    )


def make_response(response_id: str, magnitude: float) -> SystemResponse:
    return SystemResponse(
        response_id=response_id,
        cognitive_response=(magnitude, magnitude * 0.9),
        physiological_response=(magnitude * 0.8, magnitude * 0.7),
        behavioral_response=(magnitude * 0.6, magnitude * 0.5),
    )


def make_body(magnitude: float) -> BodyResponse:
    return BodyResponse(
        autonomic=(magnitude, magnitude * 0.9),
        endocrine=(magnitude * 0.8, magnitude * 0.7),
        immune=(magnitude * 0.4, magnitude * 0.3),
        motor=(magnitude * 0.75, magnitude * 0.65),
        interoceptive=(magnitude * 0.95, magnitude * 0.85),
    )


def build_sequence_from_states(sequence_id, event, states, magnitudes):
    transformations = []

    for index in range(len(states) - 1):
        transformations.append(
            build_experience_transformation(
                transformation_id=f"{sequence_id}_tx{index}",
                state_before=states[index],
                event=event,
                response=make_response(
                    f"{sequence_id}_r{index}",
                    magnitudes[index],
                ),
                body_response=make_body(magnitudes[index]),
                state_after=states[index + 1],
            )
        )

    return build_experience_sequence(
        sequence_id=sequence_id,
        transformations=tuple(transformations),
    )


def build_sensitizing_pattern(pattern_id, sequence_id, event, offset):
    states = (
        make_state(
            f"{sequence_id}_s0",
            cognitive=(0.10 + offset, 0.10 + offset, 0.10 + offset),
            physiological=(0.10 + offset, 0.08 + offset, 0.06 + offset),
            reserve=0.90 - offset,
        ),
        make_state(
            f"{sequence_id}_s1",
            cognitive=(0.20 + offset, 0.25 + offset, 0.18 + offset),
            physiological=(0.20 + offset, 0.18 + offset, 0.16 + offset),
            reserve=0.80 - offset,
        ),
        make_state(
            f"{sequence_id}_s2",
            cognitive=(0.38 + offset, 0.45 + offset, 0.36 + offset),
            physiological=(0.40 + offset, 0.36 + offset, 0.32 + offset),
            reserve=0.65 - offset,
        ),
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(
            build_sequence_from_states(
                sequence_id,
                event,
                states,
                magnitudes=(0.20 + offset, 0.45 + offset),
            ),
        ),
    )


def build_adaptation_pattern(pattern_id, sequence_id, event):
    states = (
        make_state(
            f"{sequence_id}_s0",
            cognitive=(0.50, 0.50, 0.50),
            physiological=(0.50, 0.50, 0.50),
            reserve=0.40,
        ),
        make_state(
            f"{sequence_id}_s1",
            cognitive=(0.30, 0.30, 0.30),
            physiological=(0.28, 0.28, 0.28),
            reserve=0.65,
        ),
        make_state(
            f"{sequence_id}_s2",
            cognitive=(0.18, 0.18, 0.18),
            physiological=(0.16, 0.16, 0.16),
            reserve=0.85,
        ),
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(
            build_sequence_from_states(
                sequence_id,
                event,
                states,
                magnitudes=(0.40, 0.30),
            ),
        ),
    )


@pytest.fixture
def event():
    return ExperienceEvent(
        event_id="event_repeat",
        event_type="repeated_cue",
        structural_vector=(1.0, 0.25, -0.1, 0.4),
    )


@pytest.fixture
def attractors(event):
    return (
        build_experience_attractor(
            attractor_id="sensitization_attractor",
            patterns=(
                build_sensitizing_pattern("sens_1", "sens_seq_1", event, 0.00),
                build_sensitizing_pattern("sens_2", "sens_seq_2", event, 0.01),
            ),
        ),
        build_experience_attractor(
            attractor_id="adaptation_attractor",
            patterns=(
                build_adaptation_pattern("adapt_1", "adapt_seq_1", event),
                build_adaptation_pattern("adapt_2", "adapt_seq_2", event),
            ),
        ),
    )


@pytest.fixture
def initial_state(attractors):
    return DynamicsState(
        state_id="midpoint",
        feature_vector=tuple(
            (a + b) / 2.0
            for a, b in zip(attractors[0].center, attractors[1].center)
        ),
    )


@pytest.fixture
def sens_bias():
    return AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={"sensitization_attractor": 0.75},
    )


@pytest.fixture
def sens_training(attractors, initial_state, sens_bias):
    return build_attractor_trajectory(
        trajectory_id="sens_training",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(sens_bias, sens_bias, sens_bias),
        step_size=0.10,
    )


@pytest.fixture
def cue_door():
    return ConditioningCue(
        cue_id="cue_door",
        cue_type="visual_symbol",
        feature_vector=(1.0, 0.2, 0.1, 0.0),
    )


@pytest.fixture
def sens_memory(cue_door, sens_training):
    return build_conditioning_memory(
        memory_id="sens_memory",
        cue=cue_door,
        observations=tuple(
            observation_from_trajectory(
                observation_id=f"sens_obs_{i}",
                cue=cue_door,
                trajectory=sens_training,
                reinforcement_present=True,
                reinforcement_strength=1.0,
            )
            for i in range(5)
        ),
    )


@pytest.fixture
def sens_context(cue_door, sens_memory):
    return build_associative_context(
        context_id="sens_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_door,
                memory=sens_memory,
                salience=1.0,
            ),
        ),
    )


@pytest.fixture
def stable_trajectory(cue_door, sens_context, attractors):
    return build_feedback_trajectory(
        trajectory_id="stable_sens",
        frames=tuple(
            FeedbackFrame(
                cue=cue_door,
                context=sens_context,
            )
            for _ in range(6)
        ),
        attractors=attractors,
    )


@pytest.fixture
def stable_stability(stable_trajectory):
    return analyze_feedback_stability(
        analysis_id="stable_analysis",
        trajectory=stable_trajectory,
    )


@pytest.fixture
def eligible_candidate(stable_trajectory, stable_stability):
    return evaluate_experience_consolidation(
        candidate_id="eligible_candidate",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=3,
            minimum_stability_score=0.45,
            minimum_repetition_score=0.60,
            minimum_consistency_score=0.50,
            minimum_total_score=0.50,
            maximum_mean_ambiguity=0.60,
            maximum_terminal_drift=0.10,
        ),
    )


@pytest.fixture
def committed_trace(eligible_candidate):
    return commit_memory_trace(
        trace_id="trace_001",
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )


def test_schema_version():
    assert SCHEMA_VERSION == "memory_commitment_v1"


def test_default_config_constructs():
    assert isinstance(
        MemoryCommitmentConfig(),
        MemoryCommitmentConfig,
    )


def test_config_rejects_negative_score():
    with pytest.raises(MemoryCommitmentError):
        MemoryCommitmentConfig(
            minimum_candidate_score=-0.1,
        )


def test_config_rejects_score_above_one():
    with pytest.raises(MemoryCommitmentError):
        MemoryCommitmentConfig(
            minimum_candidate_score=1.1,
        )


def test_gate_returns_expected_type(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert isinstance(
        gate,
        MemoryCommitmentGate,
    )


def test_gate_preserves_candidate_id(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert gate.candidate_id == eligible_candidate.candidate_id


def test_eligible_candidate_passes_eligibility_gate(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert gate.eligibility_gate_passed is True


def test_eligible_candidate_passes_score_gate(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert gate.score_gate_passed is True


def test_explicit_request_passes_explicit_gate(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert gate.explicit_commit_gate_passed is True


def test_all_gates_allow_commitment(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert gate.commitment_allowed is True
    assert gate.reason == "commitment_allowed"


def test_missing_explicit_request_blocks_commitment(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=False,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert gate.commitment_allowed is False
    assert gate.reason == "explicit_commit_not_requested"


def test_score_threshold_can_block_commitment(eligible_candidate):
    threshold = min(
        1.0,
        eligible_candidate.consolidation_score + 0.10,
    )

    if threshold > eligible_candidate.consolidation_score:
        gate = evaluate_memory_commitment_gate(
            candidate=eligible_candidate,
            explicit_commit_requested=True,
            config=MemoryCommitmentConfig(
                minimum_candidate_score=threshold,
            ),
        )

        assert gate.commitment_allowed is False
        assert gate.reason == "candidate_score_below_threshold"


def test_explicit_gate_can_be_disabled(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=False,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
            require_explicit_commit=False,
        ),
    )

    assert gate.explicit_commit_gate_passed is True
    assert gate.commitment_allowed is True


def test_commit_returns_trace(committed_trace):
    assert isinstance(
        committed_trace,
        CommittedMemoryTrace,
    )


def test_trace_id_preserved(committed_trace):
    assert committed_trace.trace_id == "trace_001"


def test_source_candidate_provenance_preserved(
    committed_trace,
    eligible_candidate,
):
    assert (
        committed_trace.source_candidate_id
        == eligible_candidate.candidate_id
    )


def test_source_trajectory_provenance_preserved(
    committed_trace,
    eligible_candidate,
):
    assert (
        committed_trace.source_trajectory_id
        == eligible_candidate.source_trajectory_id
    )


def test_source_stability_provenance_preserved(
    committed_trace,
    eligible_candidate,
):
    assert (
        committed_trace.source_stability_analysis_id
        == eligible_candidate.source_stability_analysis_id
    )


def test_dominant_attractor_preserved(
    committed_trace,
    eligible_candidate,
):
    assert (
        committed_trace.dominant_attractor_id
        == eligible_candidate.dominant_attractor_id
    )


def test_representation_vector_preserved(
    committed_trace,
    eligible_candidate,
):
    assert (
        committed_trace.representation_vector
        == eligible_candidate.representation_vector
    )


def test_consolidation_score_preserved(
    committed_trace,
    eligible_candidate,
):
    assert (
        committed_trace.consolidation_score
        == eligible_candidate.consolidation_score
    )


def test_commitment_confidence_bounded(committed_trace):
    assert 0.0 <= committed_trace.commitment_confidence <= 1.0


def test_trace_declares_commitment_applied(committed_trace):
    assert committed_trace.commitment_applied is True


def test_trace_does_not_mutate_source_memory(committed_trace):
    assert committed_trace.source_memory_mutated is False


def test_trace_does_not_rewrite_existing_memory(committed_trace):
    assert committed_trace.existing_memory_rewritten is False


def test_metadata_declares_commitment_applied(committed_trace):
    assert committed_trace.metadata["commitment_applied"] is True


def test_metadata_declares_no_source_memory_mutation(committed_trace):
    assert committed_trace.metadata["source_memory_mutated"] is False


def test_metadata_declares_no_existing_memory_rewrite(committed_trace):
    assert committed_trace.metadata["existing_memory_rewritten"] is False


def test_metadata_declares_no_learning(committed_trace):
    assert committed_trace.metadata["learning_applied"] is False


def test_metadata_declares_no_action(committed_trace):
    assert committed_trace.metadata["action_selected"] is False


def test_metadata_declares_no_policy_change(committed_trace):
    assert committed_trace.metadata["policy_modified"] is False


def test_metadata_declares_no_diagnosis(committed_trace):
    assert committed_trace.metadata["diagnosis_generated"] is False


def test_metadata_declares_no_biological_claim(committed_trace):
    assert committed_trace.metadata["biological_memory_claimed"] is False


def test_metadata_declares_no_causal_truth(committed_trace):
    assert committed_trace.metadata["causal_truth_inferred"] is False


def test_memory_commitment_is_policy_free(committed_trace):
    assert memory_commitment_is_policy_free(
        committed_trace
    ) is True


def test_commit_without_explicit_request_raises(eligible_candidate):
    with pytest.raises(MemoryCommitmentError):
        commit_memory_trace(
            trace_id="denied",
            candidate=eligible_candidate,
            explicit_commit_requested=False,
            config=MemoryCommitmentConfig(
                minimum_candidate_score=0.50,
            ),
        )


def test_commit_with_excessive_score_threshold_raises(eligible_candidate):
    threshold = min(
        1.0,
        eligible_candidate.consolidation_score + 0.10,
    )

    if threshold > eligible_candidate.consolidation_score:
        with pytest.raises(MemoryCommitmentError):
            commit_memory_trace(
                trace_id="denied_score",
                candidate=eligible_candidate,
                explicit_commit_requested=True,
                config=MemoryCommitmentConfig(
                    minimum_candidate_score=threshold,
                ),
            )


def test_empty_trace_id_rejected(eligible_candidate):
    with pytest.raises(MemoryCommitmentError):
        commit_memory_trace(
            trace_id="",
            candidate=eligible_candidate,
            explicit_commit_requested=True,
            config=MemoryCommitmentConfig(
                minimum_candidate_score=0.50,
            ),
        )


def test_commit_does_not_mutate_candidate(eligible_candidate):
    before_vector = eligible_candidate.representation_vector
    before_score = eligible_candidate.consolidation_score
    before_metadata = dict(eligible_candidate.metadata)

    commit_memory_trace(
        trace_id="immutability",
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert eligible_candidate.representation_vector == before_vector
    assert eligible_candidate.consolidation_score == before_score
    assert dict(eligible_candidate.metadata) == before_metadata


def test_commit_does_not_mutate_candidate_evidence(eligible_candidate):
    before = eligible_candidate.evidence

    commit_memory_trace(
        trace_id="immutability_evidence",
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert eligible_candidate.evidence == before


def test_gate_is_frozen(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        gate.commitment_allowed = False  # type: ignore[misc]


def test_trace_is_frozen(committed_trace):
    with pytest.raises(FrozenInstanceError):
        committed_trace.commitment_confidence = 0.0  # type: ignore[misc]


def test_config_is_frozen():
    config = MemoryCommitmentConfig()

    with pytest.raises(FrozenInstanceError):
        config.minimum_candidate_score = 0.0  # type: ignore[misc]


def test_gate_metadata_is_read_only(eligible_candidate):
    gate = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
    )

    assert isinstance(
        gate.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        gate.metadata["x"] = 1  # type: ignore[index]


def test_trace_metadata_is_read_only(committed_trace):
    assert isinstance(
        committed_trace.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        committed_trace.metadata["x"] = 1  # type: ignore[index]


def test_config_metadata_is_read_only():
    config = MemoryCommitmentConfig(
        metadata={"x": 1},
    )

    assert isinstance(
        config.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        config.metadata["x"] = 2  # type: ignore[index]


def test_gate_evaluation_is_deterministic(eligible_candidate):
    config = MemoryCommitmentConfig(
        minimum_candidate_score=0.50,
    )

    left = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=config,
    )

    right = evaluate_memory_commitment_gate(
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=config,
    )

    assert left == right


def test_commit_is_deterministic(eligible_candidate):
    config = MemoryCommitmentConfig(
        minimum_candidate_score=0.50,
    )

    left = commit_memory_trace(
        trace_id="same_trace",
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=config,
    )

    right = commit_memory_trace(
        trace_id="same_trace",
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=config,
    )

    assert left == right


def test_commitment_confidence_is_deterministic(eligible_candidate):
    config = MemoryCommitmentConfig(
        minimum_candidate_score=0.50,
    )

    left = commit_memory_trace(
        trace_id="same_conf",
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=config,
    )

    right = commit_memory_trace(
        trace_id="same_conf",
        candidate=eligible_candidate,
        explicit_commit_requested=True,
        config=config,
    )

    assert (
        left.commitment_confidence
        == right.commitment_confidence
    )
