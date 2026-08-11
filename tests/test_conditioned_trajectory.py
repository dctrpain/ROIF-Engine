"""
Tests for ROIF Memory 2.0 — Conditioned Trajectory Core.

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

The suite validates temporal conditioned dynamics across a sequence of cues.
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
from roif.history.conditioned_prestress import ConditionedPrestressConfig
from roif.history.conditioned_trajectory import (
    ConditionedTrajectory,
    ConditionedTrajectoryError,
    ConditionedTrajectoryStep,
    ConditionedTrajectorySummary,
    SCHEMA_VERSION,
    baseline_dominant_series,
    build_conditioned_trajectory,
    conditioned_displacement_series,
    conditioned_dominant_series,
    conditioned_trajectory_continuity_break_indices,
    conditioned_trajectory_is_continuous,
    conditioned_trajectory_is_policy_free,
    cue_id_series,
    cue_induced_divergence_series,
    return_flags,
    state_id_series,
    summarize_conditioned_trajectory,
    switch_flags,
    target_weight_gain_series,
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
                response=make_response(f"{sequence_id}_r{index}", magnitudes[index]),
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
    return DynamicsState(state_id="midpoint", feature_vector=vector)


@pytest.fixture
def sens_bias():
    return AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={"sensitization_attractor": 0.75},
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
def cue():
    return ConditioningCue(
        cue_id="cue_door",
        cue_type="visual_symbol",
        feature_vector=(1.0, 0.2, 0.1, 0.0),
    )


@pytest.fixture
def similar_cue():
    return ConditioningCue(
        cue_id="cue_door_similar",
        cue_type="visual_symbol",
        feature_vector=(0.95, 0.22, 0.12, 0.02),
    )


@pytest.fixture
def distant_cue():
    return ConditioningCue(
        cue_id="cue_far",
        cue_type="visual_symbol",
        feature_vector=(0.0, 1.0, 1.0, 1.0),
    )


@pytest.fixture
def memory(cue, sens_trajectory):
    observations = tuple(
        observation_from_trajectory(
            observation_id=f"obs_{i}",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for i in range(5)
    )
    return build_conditioning_memory(
        memory_id="memory_door",
        cue=cue,
        observations=observations,
    )


@pytest.fixture
def mixed_cue_trajectory(initial_state, attractors, cue, similar_cue, distant_cue, memory):
    return build_conditioned_trajectory(
        trajectory_id="mixed_cues",
        initial_state=initial_state,
        attractors=attractors,
        cue_sequence=(cue, similar_cue, distant_cue),
        conditioning_memory=memory,
        step_size=0.20,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "conditioned_trajectory_v1"


def test_build_returns_conditioned_trajectory(mixed_cue_trajectory):
    assert isinstance(mixed_cue_trajectory, ConditionedTrajectory)


def test_summary_type(mixed_cue_trajectory):
    assert isinstance(mixed_cue_trajectory.summary, ConditionedTrajectorySummary)


def test_steps_are_conditioned_trajectory_steps(mixed_cue_trajectory):
    assert all(isinstance(step, ConditionedTrajectoryStep) for step in mixed_cue_trajectory.steps)


def test_build_rejects_empty_attractors(initial_state, cue, memory):
    with pytest.raises(ConditionedTrajectoryError):
        build_conditioned_trajectory(
            trajectory_id="x",
            initial_state=initial_state,
            attractors=(),
            cue_sequence=(cue,),
            conditioning_memory=memory,
        )


def test_build_rejects_empty_cue_sequence(initial_state, attractors, memory):
    with pytest.raises(ConditionedTrajectoryError):
        build_conditioned_trajectory(
            trajectory_id="x",
            initial_state=initial_state,
            attractors=attractors,
            cue_sequence=(),
            conditioning_memory=memory,
        )


def test_build_rejects_empty_trajectory_id(initial_state, attractors, cue, memory):
    with pytest.raises(ConditionedTrajectoryError):
        build_conditioned_trajectory(
            trajectory_id="",
            initial_state=initial_state,
            attractors=attractors,
            cue_sequence=(cue,),
            conditioning_memory=memory,
        )


def test_three_cues_produce_three_steps(mixed_cue_trajectory):
    assert len(mixed_cue_trajectory.steps) == 3


def test_summary_step_count_is_three(mixed_cue_trajectory):
    assert mixed_cue_trajectory.summary.step_count == 3


def test_cue_id_series_regression(mixed_cue_trajectory):
    assert cue_id_series(mixed_cue_trajectory) == (
        "cue_door",
        "cue_door_similar",
        "cue_far",
    )


def test_state_id_series_has_four_states(mixed_cue_trajectory):
    assert len(state_id_series(mixed_cue_trajectory)) == 4


def test_final_state_matches_last_conditioned_state(mixed_cue_trajectory):
    assert mixed_cue_trajectory.final_state == mixed_cue_trajectory.steps[-1].dynamics.conditioned.state_after


def test_continuity_break_indices_empty(mixed_cue_trajectory):
    assert conditioned_trajectory_continuity_break_indices(mixed_cue_trajectory.steps) == ()


def test_conditioned_trajectory_is_continuous(mixed_cue_trajectory):
    assert conditioned_trajectory_is_continuous(mixed_cue_trajectory.steps) is True


def test_summary_declares_continuity(mixed_cue_trajectory):
    assert mixed_cue_trajectory.summary.continuity_preserved is True


def test_first_after_equals_second_before(mixed_cue_trajectory):
    assert (
        mixed_cue_trajectory.steps[0].dynamics.conditioned.state_after
        == mixed_cue_trajectory.steps[1].state_before
    )


def test_second_after_equals_third_before(mixed_cue_trajectory):
    assert (
        mixed_cue_trajectory.steps[1].dynamics.conditioned.state_after
        == mixed_cue_trajectory.steps[2].state_before
    )


def test_baseline_dominant_series_has_three_values(mixed_cue_trajectory):
    assert len(baseline_dominant_series(mixed_cue_trajectory)) == 3


def test_conditioned_dominant_series_has_three_values(mixed_cue_trajectory):
    assert len(conditioned_dominant_series(mixed_cue_trajectory)) == 3


def test_conditioned_first_step_targets_sensitization(mixed_cue_trajectory):
    assert mixed_cue_trajectory.steps[0].conditioned_dominant_attractor_id == "sensitization_attractor"


def test_first_target_weight_gain_positive(mixed_cue_trajectory):
    assert mixed_cue_trajectory.steps[0].target_weight_gain > 0.0


def test_similar_target_weight_gain_positive(mixed_cue_trajectory):
    assert mixed_cue_trajectory.steps[1].target_weight_gain > 0.0


def test_distant_target_weight_gain_is_smallest(mixed_cue_trajectory):
    gains = target_weight_gain_series(mixed_cue_trajectory)
    assert gains[2] < gains[1] < gains[0]


def test_cue_induced_divergence_series_has_three_values(mixed_cue_trajectory):
    assert len(cue_induced_divergence_series(mixed_cue_trajectory)) == 3


def test_first_cue_induced_divergence_positive(mixed_cue_trajectory):
    assert cue_induced_divergence_series(mixed_cue_trajectory)[0] > 0.0


def test_distant_cue_divergence_is_smallest(mixed_cue_trajectory):
    values = cue_induced_divergence_series(mixed_cue_trajectory)
    assert values[2] < values[1] < values[0]


def test_conditioned_displacement_series_has_three_values(mixed_cue_trajectory):
    assert len(conditioned_displacement_series(mixed_cue_trajectory)) == 3


def test_conditioned_displacements_nonnegative(mixed_cue_trajectory):
    assert all(value >= 0.0 for value in conditioned_displacement_series(mixed_cue_trajectory))


def test_cumulative_displacement_matches_sum(mixed_cue_trajectory):
    assert mixed_cue_trajectory.summary.cumulative_conditioned_displacement == pytest.approx(
        sum(conditioned_displacement_series(mixed_cue_trajectory))
    )


def test_cumulative_divergence_matches_sum(mixed_cue_trajectory):
    assert mixed_cue_trajectory.summary.cumulative_cue_induced_divergence == pytest.approx(
        sum(cue_induced_divergence_series(mixed_cue_trajectory))
    )


def test_mean_displacement_matches_series(mixed_cue_trajectory):
    values = conditioned_displacement_series(mixed_cue_trajectory)
    assert mixed_cue_trajectory.summary.mean_conditioned_displacement == pytest.approx(
        sum(values) / len(values)
    )


def test_mean_divergence_matches_series(mixed_cue_trajectory):
    values = cue_induced_divergence_series(mixed_cue_trajectory)
    assert mixed_cue_trajectory.summary.mean_cue_induced_divergence == pytest.approx(
        sum(values) / len(values)
    )


def test_mean_target_weight_gain_matches_series(mixed_cue_trajectory):
    values = target_weight_gain_series(mixed_cue_trajectory)
    assert mixed_cue_trajectory.summary.mean_target_weight_gain == pytest.approx(
        sum(values) / len(values)
    )


def test_switch_flags_length(mixed_cue_trajectory):
    assert len(switch_flags(mixed_cue_trajectory)) == 3


def test_return_flags_length(mixed_cue_trajectory):
    assert len(return_flags(mixed_cue_trajectory)) == 3


def test_first_step_never_switches(mixed_cue_trajectory):
    assert switch_flags(mixed_cue_trajectory)[0] is False


def test_first_step_never_returns(mixed_cue_trajectory):
    assert return_flags(mixed_cue_trajectory)[0] is False


def test_summary_switch_count_matches_flags(mixed_cue_trajectory):
    assert mixed_cue_trajectory.summary.switch_count == sum(switch_flags(mixed_cue_trajectory))


def test_summary_return_count_matches_flags(mixed_cue_trajectory):
    assert mixed_cue_trajectory.summary.return_count == sum(return_flags(mixed_cue_trajectory))


def test_unique_conditioned_count_matches_series(mixed_cue_trajectory):
    expected = len({x for x in conditioned_dominant_series(mixed_cue_trajectory) if x is not None})
    assert mixed_cue_trajectory.summary.unique_conditioned_attractor_count == expected


def test_threshold_suppression_zeroes_divergence(initial_state, attractors, cue, memory):
    trajectory = build_conditioned_trajectory(
        trajectory_id="suppressed",
        initial_state=initial_state,
        attractors=attractors,
        cue_sequence=(cue, cue, cue),
        conditioning_memory=memory,
        prestress_config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
        step_size=0.20,
    )

    assert cue_induced_divergence_series(trajectory) == pytest.approx((0.0, 0.0, 0.0))


def test_threshold_suppression_zeroes_target_weight_gain(initial_state, attractors, cue, memory):
    trajectory = build_conditioned_trajectory(
        trajectory_id="suppressed",
        initial_state=initial_state,
        attractors=attractors,
        cue_sequence=(cue, cue),
        conditioning_memory=memory,
        prestress_config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert target_weight_gain_series(trajectory) == pytest.approx((0.0, 0.0))


def test_summary_recomputation_matches(mixed_cue_trajectory):
    recomputed = summarize_conditioned_trajectory(mixed_cue_trajectory.steps)
    assert recomputed == mixed_cue_trajectory.summary


def test_summary_rejects_empty_steps():
    with pytest.raises(ConditionedTrajectoryError):
        summarize_conditioned_trajectory(())


def test_conditioned_trajectory_is_policy_free(mixed_cue_trajectory):
    assert conditioned_trajectory_is_policy_free(mixed_cue_trajectory) is True


def test_summary_declares_all_steps_policy_free(mixed_cue_trajectory):
    assert mixed_cue_trajectory.summary.all_steps_policy_free is True


def test_trajectory_declares_no_memory_mutation(mixed_cue_trajectory):
    assert mixed_cue_trajectory.metadata["memory_mutated"] is False


def test_trajectory_declares_no_learning(mixed_cue_trajectory):
    assert mixed_cue_trajectory.metadata["learning_applied"] is False


def test_trajectory_declares_no_action_selected(mixed_cue_trajectory):
    assert mixed_cue_trajectory.metadata["action_selected"] is False


def test_trajectory_declares_no_policy_modified(mixed_cue_trajectory):
    assert mixed_cue_trajectory.metadata["policy_modified"] is False


def test_trajectory_declares_no_diagnosis(mixed_cue_trajectory):
    assert mixed_cue_trajectory.metadata["diagnosis_generated"] is False


def test_trajectory_declares_no_biological_claim(mixed_cue_trajectory):
    assert mixed_cue_trajectory.metadata["biological_conditioning_claimed"] is False


def test_trajectory_declares_no_causal_truth(mixed_cue_trajectory):
    assert mixed_cue_trajectory.metadata["causal_truth_inferred"] is False


def test_every_step_declares_no_memory_mutation(mixed_cue_trajectory):
    assert all(step.metadata["memory_mutated"] is False for step in mixed_cue_trajectory.steps)


def test_every_step_declares_no_learning(mixed_cue_trajectory):
    assert all(step.metadata["learning_applied"] is False for step in mixed_cue_trajectory.steps)


def test_steps_are_tuple(mixed_cue_trajectory):
    assert isinstance(mixed_cue_trajectory.steps, tuple)


def test_trajectory_is_frozen(mixed_cue_trajectory):
    with pytest.raises(FrozenInstanceError):
        mixed_cue_trajectory.trajectory_id = "changed"  # type: ignore[misc]


def test_summary_is_frozen(mixed_cue_trajectory):
    with pytest.raises(FrozenInstanceError):
        mixed_cue_trajectory.summary.step_count = 99  # type: ignore[misc]


def test_step_is_frozen(mixed_cue_trajectory):
    with pytest.raises(FrozenInstanceError):
        mixed_cue_trajectory.steps[0].step_index = 99  # type: ignore[misc]


def test_trajectory_metadata_is_read_only(mixed_cue_trajectory):
    assert isinstance(mixed_cue_trajectory.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        mixed_cue_trajectory.metadata["x"] = 1  # type: ignore[index]


def test_summary_metadata_is_read_only(mixed_cue_trajectory):
    assert isinstance(mixed_cue_trajectory.summary.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        mixed_cue_trajectory.summary.metadata["x"] = 1  # type: ignore[index]


def test_step_metadata_is_read_only(mixed_cue_trajectory):
    assert isinstance(mixed_cue_trajectory.steps[0].metadata, MappingProxyType)
    with pytest.raises(TypeError):
        mixed_cue_trajectory.steps[0].metadata["x"] = 1  # type: ignore[index]


def test_build_is_deterministic(initial_state, attractors, cue, similar_cue, distant_cue, memory):
    left = build_conditioned_trajectory(
        trajectory_id="same",
        initial_state=initial_state,
        attractors=attractors,
        cue_sequence=(cue, similar_cue, distant_cue),
        conditioning_memory=memory,
        step_size=0.20,
    )
    right = build_conditioned_trajectory(
        trajectory_id="same",
        initial_state=initial_state,
        attractors=attractors,
        cue_sequence=(cue, similar_cue, distant_cue),
        conditioning_memory=memory,
        step_size=0.20,
    )
    assert left == right


def test_reporting_helpers_are_deterministic(mixed_cue_trajectory):
    assert cue_id_series(mixed_cue_trajectory) == cue_id_series(mixed_cue_trajectory)
    assert baseline_dominant_series(mixed_cue_trajectory) == baseline_dominant_series(mixed_cue_trajectory)
    assert conditioned_dominant_series(mixed_cue_trajectory) == conditioned_dominant_series(mixed_cue_trajectory)
    assert switch_flags(mixed_cue_trajectory) == switch_flags(mixed_cue_trajectory)
    assert return_flags(mixed_cue_trajectory) == return_flags(mixed_cue_trajectory)
    assert target_weight_gain_series(mixed_cue_trajectory) == target_weight_gain_series(mixed_cue_trajectory)
    assert cue_induced_divergence_series(mixed_cue_trajectory) == cue_induced_divergence_series(mixed_cue_trajectory)
    assert conditioned_displacement_series(mixed_cue_trajectory) == conditioned_displacement_series(mixed_cue_trajectory)
