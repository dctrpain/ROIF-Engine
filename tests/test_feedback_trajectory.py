"""
Tests for ROIF Memory 2.0 — Feedback Trajectory Core.

Layer stack:

    ExperienceTransformation
        -> ExperienceSequence
        -> ExperiencePattern
        -> ExperienceAttractor
        -> AttractorDynamics
        -> AttractorTrajectory
        -> ExperienceConditioning
        -> ConditionedPrestress
        -> ConditionedDynamics
        -> ConditionedTrajectory
        -> AssociativeContext
        -> ContextualDynamics
        -> ContextualTrajectory
        -> ContextualReconstruction
        -> ReconstructionTrajectory
        -> ReconstructionFeedback
        -> FeedbackTrajectory

The suite validates recursive semantic propagation through time.
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
from roif.history.reconstruction_feedback import (
    ReconstructionFeedbackConfig,
)
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    FeedbackTrajectory,
    FeedbackTrajectoryError,
    FeedbackTrajectoryStep,
    FeedbackTrajectorySummary,
    SCHEMA_VERSION,
    baseline_ambiguity_series,
    baseline_dominant_series,
    build_feedback_trajectory,
    feedback_ambiguity_series,
    feedback_dominant_series,
    feedback_semantic_displacement_series,
    feedback_trajectory_continuity_break_indices,
    feedback_trajectory_is_continuous,
    feedback_trajectory_is_policy_free,
    frame_context_id_series,
    frame_cue_id_series,
    reconstructed_vector_series,
    recursive_drift_series,
    return_flags,
    summarize_feedback_trajectory,
    switch_flags,
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

    sequence = build_sequence_from_states(
        sequence_id,
        event,
        states,
        magnitudes=(0.20 + offset, 0.45 + offset),
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(sequence,),
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

    sequence = build_sequence_from_states(
        sequence_id,
        event,
        states,
        magnitudes=(0.40, 0.30),
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(sequence,),
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
    sens_1 = build_sensitizing_pattern("sens_1", "sens_seq_1", event, 0.00)
    sens_2 = build_sensitizing_pattern("sens_2", "sens_seq_2", event, 0.01)
    adapt_1 = build_adaptation_pattern("adapt_1", "adapt_seq_1", event)
    adapt_2 = build_adaptation_pattern("adapt_2", "adapt_seq_2", event)

    return (
        build_experience_attractor(
            attractor_id="sensitization_attractor",
            patterns=(sens_1, sens_2),
        ),
        build_experience_attractor(
            attractor_id="adaptation_attractor",
            patterns=(adapt_1, adapt_2),
        ),
    )


@pytest.fixture
def initial_state(attractors):
    vector = tuple(
        (a + b) / 2.0
        for a, b in zip(attractors[0].center, attractors[1].center)
    )

    return DynamicsState(
        state_id="midpoint",
        feature_vector=vector,
    )


@pytest.fixture
def sens_bias():
    return AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={"sensitization_attractor": 0.75},
    )


@pytest.fixture
def adapt_bias():
    return AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={"adaptation_attractor": 0.75},
    )


@pytest.fixture
def sens_trajectory(attractors, initial_state, sens_bias):
    return build_attractor_trajectory(
        trajectory_id="sens_trajectory",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(sens_bias, sens_bias, sens_bias),
        step_size=0.10,
    )


@pytest.fixture
def adapt_trajectory(attractors, initial_state, adapt_bias):
    return build_attractor_trajectory(
        trajectory_id="adapt_trajectory",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(adapt_bias, adapt_bias, adapt_bias),
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
def cue_safe():
    return ConditioningCue(
        cue_id="cue_safe",
        cue_type="visual_symbol",
        feature_vector=(0.1, 0.9, 0.8, 0.7),
    )


@pytest.fixture
def sens_memory(cue_door, sens_trajectory):
    observations = tuple(
        observation_from_trajectory(
            observation_id=f"sens_obs_{i}",
            cue=cue_door,
            trajectory=sens_trajectory,
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for i in range(5)
    )

    return build_conditioning_memory(
        memory_id="sens_memory",
        cue=cue_door,
        observations=observations,
    )


@pytest.fixture
def adapt_memory(cue_safe, adapt_trajectory):
    observations = tuple(
        observation_from_trajectory(
            observation_id=f"adapt_obs_{i}",
            cue=cue_safe,
            trajectory=adapt_trajectory,
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for i in range(5)
    )

    return build_conditioning_memory(
        memory_id="adapt_memory",
        cue=cue_safe,
        observations=observations,
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
def adapt_context(cue_safe, adapt_memory):
    return build_associative_context(
        context_id="adapt_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_safe,
                memory=adapt_memory,
                salience=1.0,
            ),
        ),
    )


@pytest.fixture
def sas_frames(cue_door, sens_context, adapt_context):
    return (
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=adapt_context),
        FeedbackFrame(cue=cue_door, context=sens_context),
    )


@pytest.fixture
def sss_frames(cue_door, sens_context):
    return (
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=sens_context),
    )


@pytest.fixture
def default_config():
    return ReconstructionFeedbackConfig()


@pytest.fixture
def strong_config():
    return ReconstructionFeedbackConfig(
        feedback_gain=20.0,
        max_feedback_multiplier=100.0,
    )


@pytest.fixture
def sas_trajectory(sas_frames, attractors):
    return build_feedback_trajectory(
        trajectory_id="sas",
        frames=sas_frames,
        attractors=attractors,
    )


@pytest.fixture
def sss_trajectory(sss_frames, attractors):
    return build_feedback_trajectory(
        trajectory_id="sss",
        frames=sss_frames,
        attractors=attractors,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "feedback_trajectory_v1"


def test_frame_constructs(cue_door, sens_context):
    frame = FeedbackFrame(
        cue=cue_door,
        context=sens_context,
    )
    assert isinstance(frame, FeedbackFrame)


def test_build_returns_feedback_trajectory(sas_trajectory):
    assert isinstance(sas_trajectory, FeedbackTrajectory)


def test_summary_type(sas_trajectory):
    assert isinstance(
        sas_trajectory.summary,
        FeedbackTrajectorySummary,
    )


def test_steps_are_expected_type(sas_trajectory):
    assert all(
        isinstance(step, FeedbackTrajectoryStep)
        for step in sas_trajectory.steps
    )


def test_build_rejects_empty_frames(attractors):
    with pytest.raises(FeedbackTrajectoryError):
        build_feedback_trajectory(
            trajectory_id="x",
            frames=(),
            attractors=attractors,
        )


def test_build_rejects_empty_attractors(sas_frames):
    with pytest.raises(FeedbackTrajectoryError):
        build_feedback_trajectory(
            trajectory_id="x",
            frames=sas_frames,
            attractors=(),
        )


def test_build_rejects_empty_trajectory_id(sas_frames, attractors):
    with pytest.raises(FeedbackTrajectoryError):
        build_feedback_trajectory(
            trajectory_id="",
            frames=sas_frames,
            attractors=attractors,
        )


def test_custom_seed_required_when_baseline_seed_disabled(sas_frames, attractors):
    with pytest.raises(FeedbackTrajectoryError):
        build_feedback_trajectory(
            trajectory_id="x",
            frames=sas_frames,
            attractors=attractors,
            seed_from_first_baseline=False,
        )


def test_wrong_custom_seed_dimension_rejected(sas_frames, attractors):
    with pytest.raises(FeedbackTrajectoryError):
        build_feedback_trajectory(
            trajectory_id="x",
            frames=sas_frames,
            attractors=attractors,
            seed_from_first_baseline=False,
            initial_reconstructed_vector=(1.0, 2.0),
        )


def test_frames_are_tuple(sas_trajectory):
    assert isinstance(sas_trajectory.frames, tuple)


def test_steps_are_tuple(sas_trajectory):
    assert isinstance(sas_trajectory.steps, tuple)


def test_three_frames_produce_three_steps(sas_trajectory):
    assert len(sas_trajectory.steps) == 3


def test_summary_step_count_is_three(sas_trajectory):
    assert sas_trajectory.summary.step_count == 3


def test_summary_recursive_step_count_is_two(sas_trajectory):
    assert sas_trajectory.summary.recursive_step_count == 2


def test_first_step_is_seed(sas_trajectory):
    assert sas_trajectory.steps[0].feedback is None


def test_second_step_has_feedback(sas_trajectory):
    assert sas_trajectory.steps[1].feedback is not None


def test_third_step_has_feedback(sas_trajectory):
    assert sas_trajectory.steps[2].feedback is not None


def test_frame_cue_id_series_regression(sas_trajectory):
    assert frame_cue_id_series(sas_trajectory) == (
        "cue_door",
        "cue_door",
        "cue_door",
    )


def test_frame_context_id_series_regression(sas_trajectory):
    assert frame_context_id_series(sas_trajectory) == (
        "sens_context",
        "adapt_context",
        "sens_context",
    )


def test_baseline_dominant_series_has_three_values(sas_trajectory):
    assert len(
        baseline_dominant_series(sas_trajectory)
    ) == 3


def test_feedback_dominant_series_has_three_values(sas_trajectory):
    assert len(
        feedback_dominant_series(sas_trajectory)
    ) == 3


def test_first_feedback_dominant_equals_seed_baseline(sas_trajectory):
    assert (
        sas_trajectory.steps[0].feedback_dominant_attractor_id
        == sas_trajectory.steps[0].baseline_dominant_attractor_id
    )


def test_first_previous_vector_is_none(sas_trajectory):
    assert sas_trajectory.steps[0].previous_reconstructed_vector is None


def test_second_previous_vector_matches_initial_seed(sas_trajectory):
    assert (
        sas_trajectory.steps[1].previous_reconstructed_vector
        == sas_trajectory.initial_reconstructed_vector
    )


def test_third_previous_vector_matches_second_feedback_vector(sas_trajectory):
    assert (
        sas_trajectory.steps[2].previous_reconstructed_vector
        == sas_trajectory.steps[1].feedback.feedback_reconstructed_vector
    )


def test_final_vector_matches_last_feedback_vector(sas_trajectory):
    assert (
        sas_trajectory.final_reconstructed_vector
        == sas_trajectory.steps[-1].feedback.feedback_reconstructed_vector
    )


def test_reconstructed_vector_series_has_three_vectors(sas_trajectory):
    assert len(
        reconstructed_vector_series(sas_trajectory)
    ) == 3


def test_reconstructed_vector_series_first_is_initial_seed(sas_trajectory):
    assert (
        reconstructed_vector_series(sas_trajectory)[0]
        == sas_trajectory.initial_reconstructed_vector
    )


def test_continuity_break_indices_empty(sas_trajectory):
    assert feedback_trajectory_continuity_break_indices(
        sas_trajectory
    ) == ()


def test_feedback_trajectory_is_continuous(sas_trajectory):
    assert feedback_trajectory_is_continuous(
        sas_trajectory
    ) is True


def test_summary_declares_continuity(sas_trajectory):
    assert sas_trajectory.summary.continuity_preserved is True


def test_switch_flags_length_three(sas_trajectory):
    assert len(switch_flags(sas_trajectory)) == 3


def test_return_flags_length_three(sas_trajectory):
    assert len(return_flags(sas_trajectory)) == 3


def test_first_step_never_switches(sas_trajectory):
    assert switch_flags(sas_trajectory)[0] is False


def test_first_step_never_returns(sas_trajectory):
    assert return_flags(sas_trajectory)[0] is False


def test_summary_switch_count_matches_flags(sas_trajectory):
    assert sas_trajectory.summary.switch_count == sum(
        switch_flags(sas_trajectory)
    )


def test_summary_return_count_matches_flags(sas_trajectory):
    assert sas_trajectory.summary.return_count == sum(
        return_flags(sas_trajectory)
    )


def test_unique_feedback_count_matches_series(sas_trajectory):
    expected = len(set(feedback_dominant_series(sas_trajectory)))
    assert (
        sas_trajectory.summary.unique_feedback_attractor_count
        == expected
    )


def test_feedback_semantic_displacement_series_has_three_values(sas_trajectory):
    assert len(
        feedback_semantic_displacement_series(sas_trajectory)
    ) == 3


def test_seed_feedback_displacement_is_zero(sas_trajectory):
    assert feedback_semantic_displacement_series(
        sas_trajectory
    )[0] == pytest.approx(0.0)


def test_recursive_feedback_displacements_nonnegative(sas_trajectory):
    assert all(
        value >= 0.0
        for value in feedback_semantic_displacement_series(
            sas_trajectory
        )
    )


def test_recursive_drift_series_has_three_values(sas_trajectory):
    assert len(
        recursive_drift_series(sas_trajectory)
    ) == 3


def test_seed_recursive_drift_is_zero(sas_trajectory):
    assert recursive_drift_series(
        sas_trajectory
    )[0] == pytest.approx(0.0)


def test_recursive_drifts_nonnegative(sas_trajectory):
    assert all(
        value >= 0.0
        for value in recursive_drift_series(
            sas_trajectory
        )
    )


def test_baseline_ambiguity_series_has_three_values(sas_trajectory):
    assert len(
        baseline_ambiguity_series(sas_trajectory)
    ) == 3


def test_feedback_ambiguity_series_has_three_values(sas_trajectory):
    assert len(
        feedback_ambiguity_series(sas_trajectory)
    ) == 3


def test_all_baseline_ambiguities_bounded(sas_trajectory):
    assert all(
        0.0 <= value <= 1.0
        for value in baseline_ambiguity_series(
            sas_trajectory
        )
    )


def test_all_feedback_ambiguities_bounded(sas_trajectory):
    assert all(
        0.0 <= value <= 1.0
        for value in feedback_ambiguity_series(
            sas_trajectory
        )
    )


def test_cumulative_feedback_displacement_matches_sum(sas_trajectory):
    assert (
        sas_trajectory.summary.cumulative_feedback_semantic_displacement
        == pytest.approx(
            sum(
                feedback_semantic_displacement_series(
                    sas_trajectory
                )
            )
        )
    )


def test_cumulative_recursive_drift_matches_sum(sas_trajectory):
    assert (
        sas_trajectory.summary.cumulative_recursive_drift
        == pytest.approx(
            sum(
                recursive_drift_series(
                    sas_trajectory
                )
            )
        )
    )


def test_mean_feedback_displacement_matches_series(sas_trajectory):
    values = feedback_semantic_displacement_series(
        sas_trajectory
    )
    assert (
        sas_trajectory.summary.mean_feedback_semantic_displacement
        == pytest.approx(
            sum(values) / len(values)
        )
    )


def test_mean_recursive_drift_matches_series(sas_trajectory):
    values = recursive_drift_series(
        sas_trajectory
    )
    assert sas_trajectory.summary.mean_recursive_drift == pytest.approx(
        sum(values) / len(values)
    )


def test_mean_baseline_ambiguity_matches_series(sas_trajectory):
    values = baseline_ambiguity_series(
        sas_trajectory
    )
    assert sas_trajectory.summary.mean_baseline_ambiguity == pytest.approx(
        sum(values) / len(values)
    )


def test_mean_feedback_ambiguity_matches_series(sas_trajectory):
    values = feedback_ambiguity_series(
        sas_trajectory
    )
    assert sas_trajectory.summary.mean_feedback_ambiguity == pytest.approx(
        sum(values) / len(values)
    )


def test_sss_context_sequence_is_constant(sss_trajectory):
    assert frame_context_id_series(sss_trajectory) == (
        "sens_context",
        "sens_context",
        "sens_context",
    )


def test_zero_feedback_gain_keeps_recursive_dominance_baseline(
    sas_frames,
    attractors,
):
    trajectory = build_feedback_trajectory(
        trajectory_id="zero_gain",
        frames=sas_frames,
        attractors=attractors,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )

    assert feedback_dominant_series(
        trajectory
    ) == baseline_dominant_series(
        trajectory
    )


def test_zero_feedback_gain_zeroes_feedback_displacement(
    sas_frames,
    attractors,
):
    trajectory = build_feedback_trajectory(
        trajectory_id="zero_gain",
        frames=sas_frames,
        attractors=attractors,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )

    assert feedback_semantic_displacement_series(
        trajectory
    ) == pytest.approx(
        (0.0, 0.0, 0.0),
        abs=1e-12,
    )


def test_custom_sens_seed_is_preserved(
    sas_frames,
    attractors,
):
    trajectory = build_feedback_trajectory(
        trajectory_id="custom_seed",
        frames=sas_frames,
        attractors=attractors,
        seed_from_first_baseline=False,
        initial_reconstructed_vector=attractors[0].center,
    )

    assert trajectory.initial_reconstructed_vector == attractors[0].center


def test_custom_adapt_seed_is_preserved(
    sas_frames,
    attractors,
):
    trajectory = build_feedback_trajectory(
        trajectory_id="custom_seed",
        frames=sas_frames,
        attractors=attractors,
        seed_from_first_baseline=False,
        initial_reconstructed_vector=attractors[1].center,
    )

    assert trajectory.initial_reconstructed_vector == attractors[1].center


def test_strong_custom_sens_seed_can_hold_sens_against_adapt_context(
    cue_door,
    adapt_context,
    sens_context,
    attractors,
    strong_config,
):
    frames = (
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=adapt_context),
    )

    trajectory = build_feedback_trajectory(
        trajectory_id="strong_seed",
        frames=frames,
        attractors=attractors,
        config=strong_config,
        seed_from_first_baseline=False,
        initial_reconstructed_vector=attractors[0].center,
    )

    assert (
        trajectory.steps[1].feedback_dominant_attractor_id
        == "sensitization_attractor"
    )


def test_summary_recomputation_matches(sas_trajectory):
    recomputed = summarize_feedback_trajectory(
        sas_trajectory.trajectory_id,
        sas_trajectory.steps,
        sas_trajectory.initial_reconstructed_vector,
    )

    assert recomputed == sas_trajectory.summary


def test_summary_rejects_empty_steps(sas_trajectory):
    with pytest.raises(FeedbackTrajectoryError):
        summarize_feedback_trajectory(
            sas_trajectory.trajectory_id,
            (),
            sas_trajectory.initial_reconstructed_vector,
        )


def test_feedback_trajectory_is_policy_free(sas_trajectory):
    assert feedback_trajectory_is_policy_free(
        sas_trajectory
    ) is True


def test_summary_declares_all_steps_policy_free(sas_trajectory):
    assert sas_trajectory.summary.all_steps_policy_free is True


def test_trajectory_declares_no_memory_mutation(sas_trajectory):
    assert sas_trajectory.metadata["memory_mutated"] is False


def test_trajectory_declares_no_learning(sas_trajectory):
    assert sas_trajectory.metadata["learning_applied"] is False


def test_trajectory_declares_no_action_selected(sas_trajectory):
    assert sas_trajectory.metadata["action_selected"] is False


def test_trajectory_declares_no_policy_modified(sas_trajectory):
    assert sas_trajectory.metadata["policy_modified"] is False


def test_trajectory_declares_no_diagnosis(sas_trajectory):
    assert sas_trajectory.metadata["diagnosis_generated"] is False


def test_trajectory_declares_no_biological_claim(sas_trajectory):
    assert sas_trajectory.metadata["biological_feedback_claimed"] is False


def test_trajectory_declares_no_causal_truth(sas_trajectory):
    assert sas_trajectory.metadata["causal_truth_inferred"] is False


def test_every_step_declares_no_memory_mutation(sas_trajectory):
    assert all(
        step.metadata["memory_mutated"] is False
        for step in sas_trajectory.steps
    )


def test_every_step_declares_no_learning(sas_trajectory):
    assert all(
        step.metadata["learning_applied"] is False
        for step in sas_trajectory.steps
    )


def test_frame_is_frozen(cue_door, sens_context):
    frame = FeedbackFrame(
        cue=cue_door,
        context=sens_context,
    )

    with pytest.raises(FrozenInstanceError):
        frame.cue = cue_door  # type: ignore[misc]


def test_trajectory_is_frozen(sas_trajectory):
    with pytest.raises(FrozenInstanceError):
        sas_trajectory.trajectory_id = "changed"  # type: ignore[misc]


def test_summary_is_frozen(sas_trajectory):
    with pytest.raises(FrozenInstanceError):
        sas_trajectory.summary.step_count = 99  # type: ignore[misc]


def test_step_is_frozen(sas_trajectory):
    with pytest.raises(FrozenInstanceError):
        sas_trajectory.steps[0].step_index = 99  # type: ignore[misc]


def test_trajectory_metadata_is_read_only(sas_trajectory):
    assert isinstance(
        sas_trajectory.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        sas_trajectory.metadata["x"] = 1  # type: ignore[index]


def test_summary_metadata_is_read_only(sas_trajectory):
    assert isinstance(
        sas_trajectory.summary.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        sas_trajectory.summary.metadata["x"] = 1  # type: ignore[index]


def test_step_metadata_is_read_only(sas_trajectory):
    assert isinstance(
        sas_trajectory.steps[0].metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        sas_trajectory.steps[0].metadata["x"] = 1  # type: ignore[index]


def test_frame_metadata_is_read_only(cue_door, sens_context):
    frame = FeedbackFrame(
        cue=cue_door,
        context=sens_context,
        metadata={"x": 1},
    )

    assert isinstance(
        frame.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        frame.metadata["x"] = 2  # type: ignore[index]


def test_sas_build_is_deterministic(sas_frames, attractors):
    left = build_feedback_trajectory(
        trajectory_id="same",
        frames=sas_frames,
        attractors=attractors,
    )

    right = build_feedback_trajectory(
        trajectory_id="same",
        frames=sas_frames,
        attractors=attractors,
    )

    assert left == right


def test_sss_build_is_deterministic(sss_frames, attractors):
    left = build_feedback_trajectory(
        trajectory_id="same_sss",
        frames=sss_frames,
        attractors=attractors,
    )

    right = build_feedback_trajectory(
        trajectory_id="same_sss",
        frames=sss_frames,
        attractors=attractors,
    )

    assert left == right


def test_custom_seed_build_is_deterministic(sas_frames, attractors):
    left = build_feedback_trajectory(
        trajectory_id="custom_same",
        frames=sas_frames,
        attractors=attractors,
        seed_from_first_baseline=False,
        initial_reconstructed_vector=attractors[0].center,
    )

    right = build_feedback_trajectory(
        trajectory_id="custom_same",
        frames=sas_frames,
        attractors=attractors,
        seed_from_first_baseline=False,
        initial_reconstructed_vector=attractors[0].center,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(sas_trajectory):
    assert frame_cue_id_series(
        sas_trajectory
    ) == frame_cue_id_series(
        sas_trajectory
    )

    assert frame_context_id_series(
        sas_trajectory
    ) == frame_context_id_series(
        sas_trajectory
    )

    assert baseline_dominant_series(
        sas_trajectory
    ) == baseline_dominant_series(
        sas_trajectory
    )

    assert feedback_dominant_series(
        sas_trajectory
    ) == feedback_dominant_series(
        sas_trajectory
    )

    assert switch_flags(
        sas_trajectory
    ) == switch_flags(
        sas_trajectory
    )

    assert return_flags(
        sas_trajectory
    ) == return_flags(
        sas_trajectory
    )

    assert feedback_semantic_displacement_series(
        sas_trajectory
    ) == feedback_semantic_displacement_series(
        sas_trajectory
    )

    assert recursive_drift_series(
        sas_trajectory
    ) == recursive_drift_series(
        sas_trajectory
    )

    assert baseline_ambiguity_series(
        sas_trajectory
    ) == baseline_ambiguity_series(
        sas_trajectory
    )

    assert feedback_ambiguity_series(
        sas_trajectory
    ) == feedback_ambiguity_series(
        sas_trajectory
    )

    assert reconstructed_vector_series(
        sas_trajectory
    ) == reconstructed_vector_series(
        sas_trajectory
    )
