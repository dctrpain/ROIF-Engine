"""
Tests for ROIF Memory 2.0 — Experience Cycle Core.

Final architectural orchestration boundary:

    M_t
      -> feedback trajectory
      -> feedback stability
      -> consolidation candidate
      -> explicit memory commitment
      -> memory integration
      -> M_(t+1)

The suite validates that the orchestrator composes already-tested layers
without hidden learning, silent commitment, rewrite, delete, or merge.
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
from roif.history.feedback_trajectory import FeedbackFrame
from roif.history.feedback_stability import FeedbackStabilityConfig
from roif.history.experience_consolidation import ExperienceConsolidationConfig
from roif.history.memory_commitment import MemoryCommitmentConfig
from roif.history.memory_integration import MemoryIntegrationConfig
from roif.history.memory_state import empty_memory_state
from roif.history.reconstruction_feedback import ReconstructionFeedbackConfig
from roif.history.experience_cycle import (
    ExperienceCycleConfig,
    ExperienceCycleError,
    ExperienceCycleResult,
    SCHEMA_VERSION,
    experience_cycle_is_policy_free,
    experience_cycle_signature,
    run_experience_cycle,
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
def frames(cue_door, sens_context):
    return tuple(
        FeedbackFrame(
            cue=cue_door,
            context=sens_context,
        )
        for _ in range(6)
    )


@pytest.fixture
def source_memory():
    return empty_memory_state(
        state_id="M0",
    )


@pytest.fixture
def permissive_config():
    return ExperienceCycleConfig(
        feedback=ReconstructionFeedbackConfig(),
        stability=FeedbackStabilityConfig(),
        consolidation=ExperienceConsolidationConfig(
            minimum_recursive_steps=3,
            minimum_stability_score=0.45,
            minimum_repetition_score=0.60,
            minimum_consistency_score=0.50,
            minimum_total_score=0.50,
            maximum_mean_ambiguity=0.60,
            maximum_terminal_drift=0.10,
        ),
        commitment=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
        integration=MemoryIntegrationConfig(),
    )


@pytest.fixture
def cycle_result(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    return run_experience_cycle(
        cycle_id="cycle_001",
        source_memory_state=source_memory,
        target_memory_state_id="M1",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="trace_001",
        candidate_id="candidate_001",
        config=permissive_config,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "experience_cycle_v1"


def test_default_config_constructs():
    assert isinstance(
        ExperienceCycleConfig(),
        ExperienceCycleConfig,
    )


def test_run_returns_expected_type(cycle_result):
    assert isinstance(
        cycle_result,
        ExperienceCycleResult,
    )


def test_cycle_id_preserved(cycle_result):
    assert cycle_result.cycle_id == "cycle_001"


def test_source_state_preserved(cycle_result, source_memory):
    assert cycle_result.source_memory_state == source_memory


def test_target_state_id(cycle_result):
    assert cycle_result.target_memory_state.state_id == "M1"


def test_target_revision_incremented(cycle_result):
    assert (
        cycle_result.target_memory_state.revision
        == cycle_result.source_memory_state.revision + 1
    )


def test_target_trace_count_incremented(cycle_result):
    assert (
        cycle_result.target_memory_state.trace_count
        == cycle_result.source_memory_state.trace_count + 1
    )


def test_source_state_not_mutated(cycle_result, source_memory):
    assert source_memory.revision == 0
    assert source_memory.trace_count == 0


def test_feedback_trajectory_created(cycle_result):
    assert (
        cycle_result.feedback_trajectory.trajectory_id
        == "cycle_001::feedback_trajectory"
    )


def test_stability_references_feedback_trajectory(cycle_result):
    assert (
        cycle_result.stability.trajectory_id
        == cycle_result.feedback_trajectory.trajectory_id
    )


def test_candidate_id_preserved(cycle_result):
    assert (
        cycle_result.consolidation_candidate.candidate_id
        == "candidate_001"
    )


def test_candidate_is_eligible(cycle_result):
    assert (
        cycle_result.consolidation_candidate.eligible_for_consolidation
        is True
    )


def test_trace_id_preserved(cycle_result):
    assert cycle_result.committed_trace.trace_id == "trace_001"


def test_trace_provenance_candidate(cycle_result):
    assert (
        cycle_result.committed_trace.source_candidate_id
        == cycle_result.consolidation_candidate.candidate_id
    )


def test_integration_references_trace(cycle_result):
    assert (
        cycle_result.integration.new_trace_id
        == cycle_result.committed_trace.trace_id
    )


def test_first_cycle_integration_is_first_trace(cycle_result):
    assert cycle_result.integration.integration_relation == "first_trace"


def test_memory_transition_appends_trace(cycle_result):
    assert (
        cycle_result.memory_transition.appended_trace_id
        == cycle_result.committed_trace.trace_id
    )


def test_target_contains_committed_trace(cycle_result):
    assert cycle_result.target_memory_state.trace_ids == (
        "trace_001",
    )


def test_memory_transition_source_matches(cycle_result):
    assert (
        cycle_result.memory_transition.source_state_id
        == cycle_result.source_memory_state.state_id
    )


def test_memory_transition_target_matches(cycle_result):
    assert (
        cycle_result.memory_transition.target_state_id
        == cycle_result.target_memory_state.state_id
    )


def test_explicit_commit_recorded(cycle_result):
    assert cycle_result.explicit_commit_requested is True


def test_missing_explicit_commit_raises(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    with pytest.raises(Exception):
        run_experience_cycle(
            cycle_id="cycle_denied",
            source_memory_state=source_memory,
            target_memory_state_id="M1",
            frames=frames,
            attractors=attractors,
            explicit_commit_requested=False,
            trace_id="trace_denied",
            candidate_id="candidate_denied",
            config=permissive_config,
        )


def test_empty_frames_rejected(
    source_memory,
    attractors,
    permissive_config,
):
    with pytest.raises(ExperienceCycleError):
        run_experience_cycle(
            cycle_id="bad_frames",
            source_memory_state=source_memory,
            target_memory_state_id="M1",
            frames=(),
            attractors=attractors,
            explicit_commit_requested=True,
            trace_id="trace_x",
            candidate_id="candidate_x",
            config=permissive_config,
        )


def test_empty_attractors_rejected(
    source_memory,
    frames,
    permissive_config,
):
    with pytest.raises(ExperienceCycleError):
        run_experience_cycle(
            cycle_id="bad_attractors",
            source_memory_state=source_memory,
            target_memory_state_id="M1",
            frames=frames,
            attractors=(),
            explicit_commit_requested=True,
            trace_id="trace_x",
            candidate_id="candidate_x",
            config=permissive_config,
        )


def test_empty_cycle_id_rejected(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    with pytest.raises(ExperienceCycleError):
        run_experience_cycle(
            cycle_id="",
            source_memory_state=source_memory,
            target_memory_state_id="M1",
            frames=frames,
            attractors=attractors,
            explicit_commit_requested=True,
            trace_id="trace_x",
            candidate_id="candidate_x",
            config=permissive_config,
        )


def test_empty_target_state_id_rejected(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    with pytest.raises(ExperienceCycleError):
        run_experience_cycle(
            cycle_id="bad_target",
            source_memory_state=source_memory,
            target_memory_state_id="",
            frames=frames,
            attractors=attractors,
            explicit_commit_requested=True,
            trace_id="trace_x",
            candidate_id="candidate_x",
            config=permissive_config,
        )


def test_empty_trace_id_rejected(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    with pytest.raises(ExperienceCycleError):
        run_experience_cycle(
            cycle_id="bad_trace",
            source_memory_state=source_memory,
            target_memory_state_id="M1",
            frames=frames,
            attractors=attractors,
            explicit_commit_requested=True,
            trace_id="",
            candidate_id="candidate_x",
            config=permissive_config,
        )


def test_empty_candidate_id_rejected(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    with pytest.raises(ExperienceCycleError):
        run_experience_cycle(
            cycle_id="bad_candidate",
            source_memory_state=source_memory,
            target_memory_state_id="M1",
            frames=frames,
            attractors=attractors,
            explicit_commit_requested=True,
            trace_id="trace_x",
            candidate_id="",
            config=permissive_config,
        )


def test_cycle_metadata_mode(cycle_result):
    assert (
        cycle_result.metadata["experience_cycle_mode"]
        == "deterministic_orchestration"
    )


def test_cycle_metadata_source_target(cycle_result):
    assert (
        cycle_result.metadata["source_memory_state_id"]
        == "M0"
    )
    assert (
        cycle_result.metadata["target_memory_state_id"]
        == "M1"
    )


def test_cycle_metadata_revision_chain(cycle_result):
    assert cycle_result.metadata["source_revision"] == 0
    assert cycle_result.metadata["target_revision"] == 1


def test_cycle_declares_no_in_place_mutation(cycle_result):
    assert cycle_result.metadata["memory_mutated_in_place"] is False


def test_cycle_declares_no_rewrite(cycle_result):
    assert cycle_result.metadata["existing_trace_rewritten"] is False


def test_cycle_declares_no_delete(cycle_result):
    assert cycle_result.metadata["trace_deleted"] is False


def test_cycle_declares_no_merge(cycle_result):
    assert cycle_result.metadata["trace_merged"] is False


def test_cycle_declares_no_action(cycle_result):
    assert cycle_result.metadata["action_selected"] is False


def test_cycle_declares_no_policy_change(cycle_result):
    assert cycle_result.metadata["policy_modified"] is False


def test_cycle_declares_no_diagnosis(cycle_result):
    assert cycle_result.metadata["diagnosis_generated"] is False


def test_cycle_declares_no_biological_claim(cycle_result):
    assert cycle_result.metadata["biological_cycle_claimed"] is False


def test_cycle_declares_no_causal_truth(cycle_result):
    assert cycle_result.metadata["causal_truth_inferred"] is False


def test_cycle_is_policy_free(cycle_result):
    assert experience_cycle_is_policy_free(
        cycle_result
    ) is True


def test_signature_regression(cycle_result):
    assert experience_cycle_signature(
        cycle_result
    ) == (
        "cycle_001",
        "M0",
        0,
        "M1",
        1,
        "trace_001",
    )


def test_second_cycle_appends_second_trace(
    cycle_result,
    frames,
    attractors,
    permissive_config,
):
    second = run_experience_cycle(
        cycle_id="cycle_002",
        source_memory_state=cycle_result.target_memory_state,
        target_memory_state_id="M2",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="trace_002",
        candidate_id="candidate_002",
        config=permissive_config,
    )

    assert second.source_memory_state.trace_ids == (
        "trace_001",
    )
    assert second.target_memory_state.trace_ids == (
        "trace_001",
        "trace_002",
    )


def test_second_cycle_revision_chain(
    cycle_result,
    frames,
    attractors,
    permissive_config,
):
    second = run_experience_cycle(
        cycle_id="cycle_002",
        source_memory_state=cycle_result.target_memory_state,
        target_memory_state_id="M2",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="trace_002",
        candidate_id="candidate_002",
        config=permissive_config,
    )

    assert second.source_memory_state.revision == 1
    assert second.target_memory_state.revision == 2


def test_second_cycle_does_not_rewrite_first_trace(
    cycle_result,
    frames,
    attractors,
    permissive_config,
):
    original_trace = cycle_result.target_memory_state.traces[0]

    second = run_experience_cycle(
        cycle_id="cycle_002",
        source_memory_state=cycle_result.target_memory_state,
        target_memory_state_id="M2",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="trace_002",
        candidate_id="candidate_002",
        config=permissive_config,
    )

    assert second.target_memory_state.traces[0] is original_trace


def test_second_cycle_integration_has_existing_trace(
    cycle_result,
    frames,
    attractors,
    permissive_config,
):
    second = run_experience_cycle(
        cycle_id="cycle_002",
        source_memory_state=cycle_result.target_memory_state,
        target_memory_state_id="M2",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="trace_002",
        candidate_id="candidate_002",
        config=permissive_config,
    )

    assert second.integration.existing_trace_count == 1


def test_result_is_frozen(cycle_result):
    with pytest.raises(FrozenInstanceError):
        cycle_result.cycle_id = "changed"  # type: ignore[misc]


def test_config_is_frozen(permissive_config):
    with pytest.raises(FrozenInstanceError):
        permissive_config.feedback = ReconstructionFeedbackConfig()  # type: ignore[misc]


def test_result_metadata_is_read_only(cycle_result):
    assert isinstance(
        cycle_result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        cycle_result.metadata["x"] = 1  # type: ignore[index]


def test_config_metadata_is_read_only():
    config = ExperienceCycleConfig(
        metadata={"x": 1},
    )

    assert isinstance(
        config.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        config.metadata["x"] = 2  # type: ignore[index]


def test_run_is_deterministic(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    left = run_experience_cycle(
        cycle_id="same_cycle",
        source_memory_state=source_memory,
        target_memory_state_id="same_target",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="same_trace",
        candidate_id="same_candidate",
        config=permissive_config,
    )

    right = run_experience_cycle(
        cycle_id="same_cycle",
        source_memory_state=source_memory,
        target_memory_state_id="same_target",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="same_trace",
        candidate_id="same_candidate",
        config=permissive_config,
    )

    assert left == right


def test_signature_is_deterministic(cycle_result):
    assert experience_cycle_signature(
        cycle_result
    ) == experience_cycle_signature(
        cycle_result
    )


def test_two_cycle_chain_is_deterministic(
    source_memory,
    frames,
    attractors,
    permissive_config,
):
    first_left = run_experience_cycle(
        cycle_id="c1",
        source_memory_state=source_memory,
        target_memory_state_id="M1",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="t1",
        candidate_id="k1",
        config=permissive_config,
    )

    second_left = run_experience_cycle(
        cycle_id="c2",
        source_memory_state=first_left.target_memory_state,
        target_memory_state_id="M2",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="t2",
        candidate_id="k2",
        config=permissive_config,
    )

    first_right = run_experience_cycle(
        cycle_id="c1",
        source_memory_state=source_memory,
        target_memory_state_id="M1",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="t1",
        candidate_id="k1",
        config=permissive_config,
    )

    second_right = run_experience_cycle(
        cycle_id="c2",
        source_memory_state=first_right.target_memory_state,
        target_memory_state_id="M2",
        frames=frames,
        attractors=attractors,
        explicit_commit_requested=True,
        trace_id="t2",
        candidate_id="k2",
        config=permissive_config,
    )

    assert first_left == first_right
    assert second_left == second_right
