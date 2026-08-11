"""
Tests for ROIF Memory 2.0 — Contextual Dynamics Core.

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

The suite validates:
same state + same attractors + different associative context
-> different competition geometry -> different state_after.
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
from roif.history.contextual_dynamics import (
    ContextCompetitionDelta,
    ContextualDynamicsError,
    ContextualDynamicsResult,
    SCHEMA_VERSION,
    context_capture_delta_series,
    context_competition_delta_by_id,
    context_target_capture_gain,
    context_target_weight_gain,
    context_weight_delta_series,
    contextual_dynamics_is_policy_free,
    run_contextual_dynamics,
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
def cue_room():
    return ConditioningCue(
        cue_id="cue_room",
        cue_type="visual_symbol",
        feature_vector=(0.9, 0.3, 0.15, 0.05),
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
def sens_context(cue_door, cue_room, sens_memory):
    return build_associative_context(
        context_id="sens_context",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_room, memory=sens_memory, salience=0.8),
        ),
    )


@pytest.fixture
def adapt_context(cue_safe, adapt_memory):
    return build_associative_context(
        context_id="adapt_context",
        cue_inputs=(
            AssociativeCueInput(cue=cue_safe, memory=adapt_memory, salience=1.0),
        ),
    )


@pytest.fixture
def zero_context(cue_door, sens_memory):
    return build_associative_context(
        context_id="zero_context",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=0.0),
        ),
    )


@pytest.fixture
def sens_result(initial_state, attractors, sens_context):
    return run_contextual_dynamics(
        result_id="sens_result",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
        step_size=0.25,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "contextual_dynamics_v1"


def test_run_returns_contextual_dynamics_result(sens_result):
    assert isinstance(sens_result, ContextualDynamicsResult)


def test_competition_deltas_are_tuple(sens_result):
    assert isinstance(sens_result.competition_deltas, tuple)


def test_two_competition_deltas(sens_result):
    assert len(sens_result.competition_deltas) == 2


def test_delta_objects_have_expected_type(sens_result):
    assert all(isinstance(x, ContextCompetitionDelta) for x in sens_result.competition_deltas)


def test_run_rejects_empty_attractors(initial_state, sens_context):
    with pytest.raises(ContextualDynamicsError):
        run_contextual_dynamics(
            result_id="x",
            state=initial_state,
            attractors=(),
            context=sens_context,
        )


def test_run_rejects_empty_result_id(initial_state, attractors, sens_context):
    with pytest.raises(ContextualDynamicsError):
        run_contextual_dynamics(
            result_id="",
            state=initial_state,
            attractors=attractors,
            context=sens_context,
        )


def test_baseline_and_contextual_share_state_before(sens_result):
    assert sens_result.baseline.state_before == sens_result.contextual.state_before == sens_result.state_before


def test_context_preserved(sens_result, sens_context):
    assert sens_result.context == sens_context


def test_sens_context_target_weight_gain_positive(sens_result):
    assert context_target_weight_gain(sens_result) > 0.0


def test_sens_context_target_capture_gain_positive(sens_result):
    assert context_target_capture_gain(sens_result) > 0.0


def test_sensitization_weight_delta_positive(sens_result):
    delta = context_competition_delta_by_id(
        sens_result,
        "sensitization_attractor",
    )
    assert delta.weight_delta > 0.0


def test_adaptation_weight_delta_negative(sens_result):
    delta = context_competition_delta_by_id(
        sens_result,
        "adaptation_attractor",
    )
    assert delta.weight_delta < 0.0


def test_weight_deltas_sum_to_zero(sens_result):
    assert sum(context_weight_delta_series(sens_result)) == pytest.approx(0.0, abs=1e-12)


def test_sens_contextual_dominant_is_sensitization(sens_result):
    assert sens_result.contextual.dominant_attractor_id == "sensitization_attractor"


def test_context_capture_margin_increases(sens_result):
    assert sens_result.capture_margin_delta > 0.0


def test_context_entropy_decreases(sens_result):
    assert sens_result.entropy_proxy_delta < 0.0


def test_context_changes_state_after(sens_result):
    assert sens_result.state_after_distance > 0.0


def test_baseline_and_contextual_state_after_differ(sens_result):
    assert (
        sens_result.baseline.state_after.feature_vector
        != sens_result.contextual.state_after.feature_vector
    )


def test_delta_lookup_rejects_unknown(sens_result):
    with pytest.raises(ContextualDynamicsError):
        context_competition_delta_by_id(sens_result, "missing")


def test_adapt_context_changes_dominant_to_adaptation(initial_state, attractors, adapt_context):
    result = run_contextual_dynamics(
        result_id="adapt_result",
        state=initial_state,
        attractors=attractors,
        context=adapt_context,
        step_size=0.25,
    )
    assert result.contextual.dominant_attractor_id == "adaptation_attractor"


def test_adapt_context_target_weight_gain_positive(initial_state, attractors, adapt_context):
    result = run_contextual_dynamics(
        result_id="adapt_result",
        state=initial_state,
        attractors=attractors,
        context=adapt_context,
    )
    assert context_target_weight_gain(result) > 0.0


def test_different_contexts_produce_different_state_after(initial_state, attractors, sens_context, adapt_context):
    sens = run_contextual_dynamics(
        result_id="sens",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
    )
    adapt = run_contextual_dynamics(
        result_id="adapt",
        state=initial_state,
        attractors=attractors,
        context=adapt_context,
    )

    assert (
        sens.contextual.state_after.feature_vector
        != adapt.contextual.state_after.feature_vector
    )


def test_different_contexts_produce_different_dominant_attractors(initial_state, attractors, sens_context, adapt_context):
    sens = run_contextual_dynamics(
        result_id="sens",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
    )
    adapt = run_contextual_dynamics(
        result_id="adapt",
        state=initial_state,
        attractors=attractors,
        context=adapt_context,
    )

    assert sens.contextual.dominant_attractor_id != adapt.contextual.dominant_attractor_id


def test_zero_context_weight_gain_zero(initial_state, attractors, zero_context):
    result = run_contextual_dynamics(
        result_id="zero",
        state=initial_state,
        attractors=attractors,
        context=zero_context,
    )
    assert context_target_weight_gain(result) == pytest.approx(0.0, abs=1e-12)


def test_zero_context_state_after_matches_baseline(initial_state, attractors, zero_context):
    result = run_contextual_dynamics(
        result_id="zero",
        state=initial_state,
        attractors=attractors,
        context=zero_context,
    )
    assert result.state_after_distance == pytest.approx(0.0, abs=1e-12)


def test_zero_context_capture_margin_delta_zero(initial_state, attractors, zero_context):
    result = run_contextual_dynamics(
        result_id="zero",
        state=initial_state,
        attractors=attractors,
        context=zero_context,
    )
    assert result.capture_margin_delta == pytest.approx(0.0, abs=1e-12)


def test_zero_context_entropy_delta_zero(initial_state, attractors, zero_context):
    result = run_contextual_dynamics(
        result_id="zero",
        state=initial_state,
        attractors=attractors,
        context=zero_context,
    )
    assert result.entropy_proxy_delta == pytest.approx(0.0, abs=1e-12)


def test_step_size_zero_keeps_state_after_distance_zero(initial_state, attractors, sens_context):
    result = run_contextual_dynamics(
        result_id="zero_step",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
        step_size=0.0,
    )
    assert result.state_after_distance == pytest.approx(0.0, abs=1e-12)


def test_step_size_zero_still_changes_competition(initial_state, attractors, sens_context):
    result = run_contextual_dynamics(
        result_id="zero_step",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
        step_size=0.0,
    )
    assert context_target_weight_gain(result) > 0.0


def test_weight_delta_series_is_tuple(sens_result):
    assert isinstance(context_weight_delta_series(sens_result), tuple)


def test_capture_delta_series_is_tuple(sens_result):
    assert isinstance(context_capture_delta_series(sens_result), tuple)


def test_weight_delta_series_length_two(sens_result):
    assert len(context_weight_delta_series(sens_result)) == 2


def test_capture_delta_series_length_two(sens_result):
    assert len(context_capture_delta_series(sens_result)) == 2


def test_contextual_dynamics_is_policy_free(sens_result):
    assert contextual_dynamics_is_policy_free(sens_result) is True


def test_result_declares_no_memory_mutation(sens_result):
    assert sens_result.metadata["memory_mutated"] is False


def test_result_declares_no_learning(sens_result):
    assert sens_result.metadata["learning_applied"] is False


def test_result_declares_no_action_selected(sens_result):
    assert sens_result.metadata["action_selected"] is False


def test_result_declares_no_policy_modified(sens_result):
    assert sens_result.metadata["policy_modified"] is False


def test_result_declares_no_diagnosis(sens_result):
    assert sens_result.metadata["diagnosis_generated"] is False


def test_result_declares_no_biological_claim(sens_result):
    assert sens_result.metadata["biological_context_claimed"] is False


def test_result_declares_no_causal_truth(sens_result):
    assert sens_result.metadata["causal_truth_inferred"] is False


def test_every_delta_declares_no_memory_mutation(sens_result):
    assert all(x.metadata["memory_mutated"] is False for x in sens_result.competition_deltas)


def test_every_delta_declares_no_policy_modified(sens_result):
    assert all(x.metadata["policy_modified"] is False for x in sens_result.competition_deltas)


def test_run_does_not_mutate_context(initial_state, attractors, sens_context):
    before_contributions = sens_context.contributions
    before_aggregates = sens_context.aggregates
    before_bias = dict(sens_context.prestress.bias_by_attractor_id)

    run_contextual_dynamics(
        result_id="immutability",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
    )

    assert sens_context.contributions == before_contributions
    assert sens_context.aggregates == before_aggregates
    assert dict(sens_context.prestress.bias_by_attractor_id) == before_bias


def test_run_does_not_mutate_attractor_centers(initial_state, attractors, sens_context):
    before = tuple(attractor.center for attractor in attractors)

    run_contextual_dynamics(
        result_id="immutability",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
    )

    after = tuple(attractor.center for attractor in attractors)
    assert after == before


def test_result_is_frozen(sens_result):
    with pytest.raises(FrozenInstanceError):
        sens_result.state_after_distance = 0.0  # type: ignore[misc]


def test_delta_is_frozen(sens_result):
    with pytest.raises(FrozenInstanceError):
        sens_result.competition_deltas[0].weight_delta = 0.0  # type: ignore[misc]


def test_result_metadata_is_read_only(sens_result):
    assert isinstance(sens_result.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        sens_result.metadata["x"] = 1  # type: ignore[index]


def test_delta_metadata_is_read_only(sens_result):
    assert isinstance(sens_result.competition_deltas[0].metadata, MappingProxyType)
    with pytest.raises(TypeError):
        sens_result.competition_deltas[0].metadata["x"] = 1  # type: ignore[index]


def test_sens_context_run_is_deterministic(initial_state, attractors, sens_context):
    left = run_contextual_dynamics(
        result_id="same",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
    )

    right = run_contextual_dynamics(
        result_id="same",
        state=initial_state,
        attractors=attractors,
        context=sens_context,
    )

    assert left == right


def test_adapt_context_run_is_deterministic(initial_state, attractors, adapt_context):
    left = run_contextual_dynamics(
        result_id="same_adapt",
        state=initial_state,
        attractors=attractors,
        context=adapt_context,
    )

    right = run_contextual_dynamics(
        result_id="same_adapt",
        state=initial_state,
        attractors=attractors,
        context=adapt_context,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(sens_result):
    assert context_weight_delta_series(sens_result) == context_weight_delta_series(sens_result)
    assert context_capture_delta_series(sens_result) == context_capture_delta_series(sens_result)
    assert context_target_weight_gain(sens_result) == context_target_weight_gain(sens_result)
    assert context_target_capture_gain(sens_result) == context_target_capture_gain(sens_result)
    assert context_competition_delta_by_id(
        sens_result,
        "sensitization_attractor",
    ) == context_competition_delta_by_id(
        sens_result,
        "sensitization_attractor",
    )
