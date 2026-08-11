"""
Tests for ROIF Memory 2.0 — Associative Context Core.

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

The suite validates multi-cue aggregation into a shared associative context.
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
    AssociativeContext,
    AssociativeContextError,
    AssociativeCueContribution,
    AssociativeCueInput,
    AttractorContextAggregate,
    SCHEMA_VERSION,
    aggregate_by_attractor_id,
    associative_context_is_policy_free,
    build_associative_context,
    context_attractor_ids,
    context_weight_series,
    contribution_by_cue_id,
    contribution_weight_series,
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
def two_sens_context(cue_door, cue_room, sens_memory):
    return build_associative_context(
        context_id="two_sens",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_room, memory=sens_memory, salience=0.8),
        ),
    )


@pytest.fixture
def mixed_context(cue_door, cue_safe, sens_memory, adapt_memory):
    return build_associative_context(
        context_id="mixed_context",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_safe, memory=adapt_memory, salience=0.5),
        ),
    )


def test_schema_version():
    assert SCHEMA_VERSION == "associative_context_v1"


def test_associative_cue_input_constructs(cue_door, sens_memory):
    item = AssociativeCueInput(cue=cue_door, memory=sens_memory)
    assert isinstance(item, AssociativeCueInput)


def test_associative_cue_input_rejects_negative_salience(cue_door, sens_memory):
    with pytest.raises(AssociativeContextError):
        AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=-0.1)


def test_build_returns_associative_context(two_sens_context):
    assert isinstance(two_sens_context, AssociativeContext)


def test_contributions_are_tuple(two_sens_context):
    assert isinstance(two_sens_context.contributions, tuple)


def test_aggregates_are_tuple(two_sens_context):
    assert isinstance(two_sens_context.aggregates, tuple)


def test_contributions_are_expected_type(two_sens_context):
    assert all(isinstance(x, AssociativeCueContribution) for x in two_sens_context.contributions)


def test_aggregates_are_expected_type(two_sens_context):
    assert all(isinstance(x, AttractorContextAggregate) for x in two_sens_context.aggregates)


def test_build_rejects_empty_inputs():
    with pytest.raises(AssociativeContextError):
        build_associative_context(
            context_id="x",
            cue_inputs=(),
        )


def test_build_rejects_empty_context_id(cue_door, sens_memory):
    with pytest.raises(AssociativeContextError):
        build_associative_context(
            context_id="",
            cue_inputs=(AssociativeCueInput(cue=cue_door, memory=sens_memory),),
        )


def test_two_sens_context_has_two_contributions(two_sens_context):
    assert len(two_sens_context.contributions) == 2


def test_two_sens_context_has_one_aggregate(two_sens_context):
    assert len(two_sens_context.aggregates) == 1


def test_two_sens_context_targets_sensitization(two_sens_context):
    assert two_sens_context.dominant_attractor_id == "sensitization_attractor"


def test_two_sens_context_attractor_ids(two_sens_context):
    assert context_attractor_ids(two_sens_context) == ("sensitization_attractor",)


def test_single_aggregate_weight_is_one(two_sens_context):
    assert context_weight_series(two_sens_context) == pytest.approx((1.0,))


def test_two_sens_context_prestress_targets_sensitization(two_sens_context):
    assert tuple(two_sens_context.prestress.bias_by_attractor_id.keys()) == (
        "sensitization_attractor",
    )


def test_door_contribution_positive(two_sens_context):
    item = contribution_by_cue_id(two_sens_context, "cue_door")
    assert item.weighted_bias > 0.0


def test_room_contribution_positive(two_sens_context):
    item = contribution_by_cue_id(two_sens_context, "cue_room")
    assert item.weighted_bias > 0.0


def test_lower_salience_reduces_weighted_bias(two_sens_context):
    door = contribution_by_cue_id(two_sens_context, "cue_door")
    room = contribution_by_cue_id(two_sens_context, "cue_room")
    assert room.weighted_bias < door.weighted_bias


def test_contribution_lookup_rejects_unknown(two_sens_context):
    with pytest.raises(AssociativeContextError):
        contribution_by_cue_id(two_sens_context, "missing")


def test_aggregate_lookup_returns_expected(two_sens_context):
    aggregate = aggregate_by_attractor_id(two_sens_context, "sensitization_attractor")
    assert aggregate.attractor_id == "sensitization_attractor"


def test_aggregate_lookup_rejects_unknown(two_sens_context):
    with pytest.raises(AssociativeContextError):
        aggregate_by_attractor_id(two_sens_context, "missing")


def test_two_sens_context_confidence_bounded(two_sens_context):
    assert 0.0 <= two_sens_context.context_confidence <= 1.0


def test_two_sens_context_margin_is_one(two_sens_context):
    assert two_sens_context.competition_margin == pytest.approx(1.0)


def test_mixed_context_has_two_aggregates(mixed_context):
    assert len(mixed_context.aggregates) == 2


def test_mixed_context_has_two_attractor_ids(mixed_context):
    assert set(context_attractor_ids(mixed_context)) == {
        "sensitization_attractor",
        "adaptation_attractor",
    }


def test_mixed_context_weights_sum_to_one(mixed_context):
    assert sum(context_weight_series(mixed_context)) == pytest.approx(1.0)


def test_mixed_context_dominant_is_sensitization(mixed_context):
    assert mixed_context.dominant_attractor_id == "sensitization_attractor"


def test_mixed_context_sens_weight_exceeds_adapt(mixed_context):
    sens = aggregate_by_attractor_id(mixed_context, "sensitization_attractor")
    adapt = aggregate_by_attractor_id(mixed_context, "adaptation_attractor")
    assert sens.normalized_context_weight > adapt.normalized_context_weight


def test_mixed_context_margin_positive(mixed_context):
    assert mixed_context.competition_margin > 0.0


def test_mixed_context_confidence_bounded(mixed_context):
    assert 0.0 <= mixed_context.context_confidence <= 1.0


def test_equal_salience_can_reduce_margin(cue_door, cue_safe, sens_memory, adapt_memory):
    context = build_associative_context(
        context_id="equal",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_safe, memory=adapt_memory, salience=1.0),
        ),
    )
    assert context.competition_margin <= 1.0


def test_zero_salience_yields_zero_weighted_bias(cue_door, sens_memory):
    context = build_associative_context(
        context_id="zero",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=0.0),
        ),
    )
    assert contribution_weight_series(context) == pytest.approx((0.0,))


def test_zero_salience_context_weight_is_zero(cue_door, sens_memory):
    context = build_associative_context(
        context_id="zero",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=0.0),
        ),
    )
    assert context_weight_series(context) == pytest.approx((0.0,))


def test_zero_salience_prestress_bias_is_zero(cue_door, sens_memory):
    context = build_associative_context(
        context_id="zero",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=0.0),
        ),
    )
    assert context.prestress.bias_by_attractor_id["sensitization_attractor"] == pytest.approx(0.0)


def test_max_total_bias_caps_total(two_sens_context, cue_door, cue_room, sens_memory):
    capped = build_associative_context(
        context_id="capped",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_room, memory=sens_memory, salience=1.0),
        ),
        max_total_bias=0.10,
    )
    assert sum(capped.prestress.bias_by_attractor_id.values()) <= 0.10 + 1e-12


def test_negative_max_total_bias_rejected(cue_door, sens_memory):
    with pytest.raises(AssociativeContextError):
        build_associative_context(
            context_id="bad",
            cue_inputs=(AssociativeCueInput(cue=cue_door, memory=sens_memory),),
            max_total_bias=-0.1,
        )


def test_prestress_metadata_source(two_sens_context):
    assert two_sens_context.prestress.metadata["source"] == "associative_context"


def test_prestress_metadata_preserves_context_id(two_sens_context):
    assert two_sens_context.prestress.metadata["context_id"] == "two_sens"


def test_context_is_policy_free(two_sens_context):
    assert associative_context_is_policy_free(two_sens_context) is True


def test_context_declares_no_memory_mutation(two_sens_context):
    assert two_sens_context.metadata["memory_mutated"] is False


def test_context_declares_no_learning(two_sens_context):
    assert two_sens_context.metadata["learning_applied"] is False


def test_context_declares_no_action_selected(two_sens_context):
    assert two_sens_context.metadata["action_selected"] is False


def test_context_declares_no_policy_modified(two_sens_context):
    assert two_sens_context.metadata["policy_modified"] is False


def test_context_declares_no_diagnosis(two_sens_context):
    assert two_sens_context.metadata["diagnosis_generated"] is False


def test_context_declares_no_biological_claim(two_sens_context):
    assert two_sens_context.metadata["biological_context_claimed"] is False


def test_context_declares_no_causal_truth(two_sens_context):
    assert two_sens_context.metadata["causal_truth_inferred"] is False


def test_every_contribution_declares_no_mutation(two_sens_context):
    assert all(x.metadata["memory_mutated"] is False for x in two_sens_context.contributions)


def test_every_aggregate_declares_no_mutation(two_sens_context):
    assert all(x.metadata["memory_mutated"] is False for x in two_sens_context.aggregates)


def test_build_does_not_mutate_source_memory(cue_door, cue_room, sens_memory):
    before_associations = sens_memory.associations
    before_observations = sens_memory.observations

    build_associative_context(
        context_id="immutability",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory),
            AssociativeCueInput(cue=cue_room, memory=sens_memory),
        ),
    )

    assert sens_memory.associations == before_associations
    assert sens_memory.observations == before_observations


def test_context_is_frozen(two_sens_context):
    with pytest.raises(FrozenInstanceError):
        two_sens_context.context_id = "changed"  # type: ignore[misc]


def test_contribution_is_frozen(two_sens_context):
    with pytest.raises(FrozenInstanceError):
        two_sens_context.contributions[0].weighted_bias = 0.0  # type: ignore[misc]


def test_aggregate_is_frozen(two_sens_context):
    with pytest.raises(FrozenInstanceError):
        two_sens_context.aggregates[0].contributor_count = 99  # type: ignore[misc]


def test_cue_input_is_frozen(cue_door, sens_memory):
    item = AssociativeCueInput(cue=cue_door, memory=sens_memory)
    with pytest.raises(FrozenInstanceError):
        item.salience = 2.0  # type: ignore[misc]


def test_context_metadata_is_read_only(two_sens_context):
    assert isinstance(two_sens_context.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        two_sens_context.metadata["x"] = 1  # type: ignore[index]


def test_contribution_metadata_is_read_only(two_sens_context):
    assert isinstance(two_sens_context.contributions[0].metadata, MappingProxyType)
    with pytest.raises(TypeError):
        two_sens_context.contributions[0].metadata["x"] = 1  # type: ignore[index]


def test_aggregate_metadata_is_read_only(two_sens_context):
    assert isinstance(two_sens_context.aggregates[0].metadata, MappingProxyType)
    with pytest.raises(TypeError):
        two_sens_context.aggregates[0].metadata["x"] = 1  # type: ignore[index]


def test_build_is_deterministic(cue_door, cue_room, sens_memory):
    left = build_associative_context(
        context_id="same",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_room, memory=sens_memory, salience=0.8),
        ),
    )

    right = build_associative_context(
        context_id="same",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_room, memory=sens_memory, salience=0.8),
        ),
    )

    assert left == right


def test_mixed_context_is_deterministic(cue_door, cue_safe, sens_memory, adapt_memory):
    left = build_associative_context(
        context_id="mixed_same",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_safe, memory=adapt_memory, salience=0.5),
        ),
    )

    right = build_associative_context(
        context_id="mixed_same",
        cue_inputs=(
            AssociativeCueInput(cue=cue_door, memory=sens_memory, salience=1.0),
            AssociativeCueInput(cue=cue_safe, memory=adapt_memory, salience=0.5),
        ),
    )

    assert left == right


def test_reporting_helpers_are_deterministic(two_sens_context):
    assert context_attractor_ids(two_sens_context) == context_attractor_ids(two_sens_context)
    assert context_weight_series(two_sens_context) == context_weight_series(two_sens_context)
    assert contribution_weight_series(two_sens_context) == contribution_weight_series(two_sens_context)
    assert contribution_by_cue_id(two_sens_context, "cue_door") == contribution_by_cue_id(two_sens_context, "cue_door")
    assert aggregate_by_attractor_id(
        two_sens_context,
        "sensitization_attractor",
    ) == aggregate_by_attractor_id(
        two_sens_context,
        "sensitization_attractor",
    )
