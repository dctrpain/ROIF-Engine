"""
Tests for ROIF Memory 2.0 — Experience Conditioning Core.

Layer stack:

    ExperienceTransformation
        -> ExperienceSequence
        -> ExperiencePattern
        -> ExperienceAttractor
        -> AttractorDynamics
        -> AttractorTrajectory
        -> ExperienceConditioning

The suite validates descriptive acquisition, extinction, generalization,
conditioned-response proxy behavior, policy boundaries, immutability,
and determinism.
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
from roif.history.experience_sequence import (
    build_experience_sequence,
)
from roif.history.experience_pattern import (
    build_experience_pattern,
)
from roif.history.experience_attractor import (
    build_experience_attractor,
)
from roif.history.attractor_dynamics import (
    AttractorPrestress,
    DynamicsState,
)
from roif.history.attractor_trajectory import (
    build_attractor_trajectory,
)
from roif.history.experience_conditioning import (
    ConditionedResponse,
    ConditioningAssociation,
    ConditioningCue,
    ConditioningMemory,
    ConditioningObservation,
    ExperienceConditioningError,
    SCHEMA_VERSION,
    association_by_attractor_id,
    association_confidence_series,
    association_strength_series,
    build_conditioning_memory,
    conditioned_response,
    conditioned_response_is_policy_free,
    conditioning_memory_is_policy_free,
    dominant_conditioned_attractor,
    observation_from_trajectory,
)


# =============================================================================
# Helpers
# =============================================================================


def make_state(
    state_id: str,
    *,
    cognitive,
    physiological,
    reserve: float,
):
    return SystemState(
        state_id=state_id,
        cognitive=tuple(cognitive),
        physiological=tuple(physiological),
        contextual=(0.5, 0.2),
        reserve=reserve,
    )


def make_response(
    response_id: str,
    magnitude: float,
) -> SystemResponse:
    return SystemResponse(
        response_id=response_id,
        cognitive_response=(magnitude, magnitude * 0.9),
        physiological_response=(magnitude * 0.8, magnitude * 0.7),
        behavioral_response=(magnitude * 0.6, magnitude * 0.5),
    )


def make_body(
    magnitude: float,
) -> BodyResponse:
    return BodyResponse(
        autonomic=(magnitude, magnitude * 0.9),
        endocrine=(magnitude * 0.8, magnitude * 0.7),
        immune=(magnitude * 0.4, magnitude * 0.3),
        motor=(magnitude * 0.75, magnitude * 0.65),
        interoceptive=(magnitude * 0.95, magnitude * 0.85),
    )


def build_sequence_from_states(
    sequence_id: str,
    event: ExperienceEvent,
    states,
    magnitudes,
):
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
                body_response=make_body(
                    magnitudes[index]
                ),
                state_after=states[index + 1],
            )
        )

    return build_experience_sequence(
        sequence_id=sequence_id,
        transformations=tuple(transformations),
    )


def build_sensitizing_pattern(
    pattern_id: str,
    sequence_id: str,
    event: ExperienceEvent,
    offset: float,
):
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
        magnitudes=(
            0.20 + offset,
            0.45 + offset,
        ),
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(sequence,),
    )


def build_adaptation_pattern(
    pattern_id: str,
    sequence_id: str,
    event: ExperienceEvent,
):
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


def make_attractors(event: ExperienceEvent):
    sens_1 = build_sensitizing_pattern(
        "sens_1",
        "sens_seq_1",
        event,
        0.00,
    )
    sens_2 = build_sensitizing_pattern(
        "sens_2",
        "sens_seq_2",
        event,
        0.01,
    )

    adapt_1 = build_adaptation_pattern(
        "adapt_1",
        "adapt_seq_1",
        event,
    )
    adapt_2 = build_adaptation_pattern(
        "adapt_2",
        "adapt_seq_2",
        event,
    )

    sensitization_attractor = build_experience_attractor(
        attractor_id="sensitization_attractor",
        patterns=(sens_1, sens_2),
    )

    adaptation_attractor = build_experience_attractor(
        attractor_id="adaptation_attractor",
        patterns=(adapt_1, adapt_2),
    )

    return (
        sensitization_attractor,
        adaptation_attractor,
    )


def midpoint_state(
    left,
    right,
) -> DynamicsState:
    vector = tuple(
        (a + b) / 2.0
        for a, b in zip(
            left.center,
            right.center,
        )
    )

    return DynamicsState(
        state_id="midpoint",
        feature_vector=vector,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event() -> ExperienceEvent:
    return ExperienceEvent(
        event_id="event_repeat",
        event_type="repeated_cue",
        structural_vector=(1.0, 0.25, -0.1, 0.4),
    )


@pytest.fixture
def attractors(event):
    return make_attractors(event)


@pytest.fixture
def initial_state(attractors):
    return midpoint_state(
        attractors[0],
        attractors[1],
    )


@pytest.fixture
def sens_bias() -> AttractorPrestress:
    return AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={
            "sensitization_attractor": 0.75,
        },
    )


@pytest.fixture
def adapt_bias() -> AttractorPrestress:
    return AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.75,
        },
    )


@pytest.fixture
def sens_trajectory(
    attractors,
    initial_state,
    sens_bias,
):
    return build_attractor_trajectory(
        trajectory_id="sens_trajectory",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sens_bias,
            sens_bias,
            sens_bias,
        ),
        step_size=0.10,
    )


@pytest.fixture
def adapt_trajectory(
    attractors,
    initial_state,
    adapt_bias,
):
    return build_attractor_trajectory(
        trajectory_id="adapt_trajectory",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            adapt_bias,
            adapt_bias,
            adapt_bias,
        ),
        step_size=0.10,
    )


@pytest.fixture
def cue() -> ConditioningCue:
    return ConditioningCue(
        cue_id="cue_door",
        cue_type="visual_symbol",
        feature_vector=(1.0, 0.2, 0.1, 0.0),
    )


@pytest.fixture
def similar_cue() -> ConditioningCue:
    return ConditioningCue(
        cue_id="cue_door_similar",
        cue_type="visual_symbol",
        feature_vector=(0.95, 0.22, 0.12, 0.02),
    )


@pytest.fixture
def distant_cue() -> ConditioningCue:
    return ConditioningCue(
        cue_id="cue_far",
        cue_type="visual_symbol",
        feature_vector=(0.0, 1.0, 1.0, 1.0),
    )


@pytest.fixture
def reinforced_observations(
    cue,
    sens_trajectory,
):
    return tuple(
        observation_from_trajectory(
            observation_id=f"obs_{index}",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for index in range(5)
    )


@pytest.fixture
def conditioned_memory(
    cue,
    reinforced_observations,
) -> ConditioningMemory:
    return build_conditioning_memory(
        memory_id="memory_door",
        cue=cue,
        observations=reinforced_observations,
    )


# =============================================================================
# Schema / cue
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "experience_conditioning_v1"


def test_cue_constructed(
    cue,
) -> None:
    assert isinstance(
        cue,
        ConditioningCue,
    )


def test_cue_rejects_empty_id() -> None:
    with pytest.raises(
        ExperienceConditioningError
    ):
        ConditioningCue(
            cue_id="",
            cue_type="visual",
            feature_vector=(1.0,),
        )


def test_cue_rejects_empty_type() -> None:
    with pytest.raises(
        ExperienceConditioningError
    ):
        ConditioningCue(
            cue_id="cue",
            cue_type="",
            feature_vector=(1.0,),
        )


def test_cue_rejects_empty_vector() -> None:
    with pytest.raises(
        ExperienceConditioningError
    ):
        ConditioningCue(
            cue_id="cue",
            cue_type="visual",
            feature_vector=(),
        )


# =============================================================================
# Observation from trajectory
# =============================================================================


def test_observation_type(
    cue,
    sens_trajectory,
) -> None:
    observation = observation_from_trajectory(
        observation_id="obs",
        cue=cue,
        trajectory=sens_trajectory,
        reinforcement_present=True,
    )

    assert isinstance(
        observation,
        ConditioningObservation,
    )


def test_sens_trajectory_terminal_attractor(
    cue,
    sens_trajectory,
) -> None:
    observation = observation_from_trajectory(
        observation_id="obs",
        cue=cue,
        trajectory=sens_trajectory,
        reinforcement_present=True,
    )

    assert (
        observation.terminal_attractor_id
        == "sensitization_attractor"
    )


def test_adapt_trajectory_terminal_attractor(
    cue,
    adapt_trajectory,
) -> None:
    observation = observation_from_trajectory(
        observation_id="obs",
        cue=cue,
        trajectory=adapt_trajectory,
        reinforcement_present=True,
    )

    assert (
        observation.terminal_attractor_id
        == "adaptation_attractor"
    )


def test_unreinforced_observation_strength_is_zero(
    cue,
    sens_trajectory,
) -> None:
    observation = observation_from_trajectory(
        observation_id="obs",
        cue=cue,
        trajectory=sens_trajectory,
        reinforcement_present=False,
        reinforcement_strength=1.0,
    )

    assert observation.reinforcement_strength == pytest.approx(
        0.0
    )


def test_observation_preserves_dominant_series(
    cue,
    sens_trajectory,
) -> None:
    observation = observation_from_trajectory(
        observation_id="obs",
        cue=cue,
        trajectory=sens_trajectory,
        reinforcement_present=True,
    )

    assert len(
        observation.dominant_series
    ) == 3


# =============================================================================
# Acquisition
# =============================================================================


def test_conditioning_memory_constructed(
    conditioned_memory,
) -> None:
    assert isinstance(
        conditioned_memory,
        ConditioningMemory,
    )


def test_conditioning_memory_has_five_observations(
    conditioned_memory,
) -> None:
    assert len(
        conditioned_memory.observations
    ) == 5


def test_conditioning_memory_has_one_association(
    conditioned_memory,
) -> None:
    assert len(
        conditioned_memory.associations
    ) == 1


def test_association_type(
    conditioned_memory,
) -> None:
    assert isinstance(
        conditioned_memory.associations[0],
        ConditioningAssociation,
    )


def test_acquisition_strength_positive(
    conditioned_memory,
) -> None:
    association = conditioned_memory.associations[0]
    assert association.acquisition_strength > 0.0


def test_effective_association_strength_positive(
    conditioned_memory,
) -> None:
    association = conditioned_memory.associations[0]
    assert association.effective_association_strength > 0.0


def test_reinforced_count_is_five(
    conditioned_memory,
) -> None:
    association = conditioned_memory.associations[0]
    assert association.reinforced_exposure_count == 5


def test_extinction_count_is_zero_initially(
    conditioned_memory,
) -> None:
    association = conditioned_memory.associations[0]
    assert association.extinction_exposure_count == 0


def test_association_confidence_positive(
    conditioned_memory,
) -> None:
    association = conditioned_memory.associations[0]
    assert association.association_confidence > 0.0


def test_dominant_conditioned_attractor_is_sensitization(
    conditioned_memory,
) -> None:
    assert dominant_conditioned_attractor(
        conditioned_memory
    ) == "sensitization_attractor"


# =============================================================================
# Extinction
# =============================================================================


def test_extinction_reduces_effective_strength(
    cue,
    sens_trajectory,
    conditioned_memory,
) -> None:
    extinction_observations = tuple(
        observation_from_trajectory(
            observation_id=f"ext_{index}",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=False,
        )
        for index in range(5)
    )

    extended = build_conditioning_memory(
        memory_id="memory_ext",
        cue=cue,
        observations=(
            *conditioned_memory.observations,
            *extinction_observations,
        ),
    )

    before = association_by_attractor_id(
        conditioned_memory,
        "sensitization_attractor",
    )

    after = association_by_attractor_id(
        extended,
        "sensitization_attractor",
    )

    assert (
        after.effective_association_strength
        < before.effective_association_strength
    )


def test_extinction_strength_positive_after_unreinforced_exposures(
    cue,
    sens_trajectory,
    conditioned_memory,
) -> None:
    extinction_observations = tuple(
        observation_from_trajectory(
            observation_id=f"ext_{index}",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=False,
        )
        for index in range(3)
    )

    extended = build_conditioning_memory(
        memory_id="memory_ext",
        cue=cue,
        observations=(
            *conditioned_memory.observations,
            *extinction_observations,
        ),
    )

    association = association_by_attractor_id(
        extended,
        "sensitization_attractor",
    )

    assert association.extinction_strength > 0.0


def test_extinction_count_matches_unreinforced_exposures(
    cue,
    sens_trajectory,
    conditioned_memory,
) -> None:
    extinction_observations = tuple(
        observation_from_trajectory(
            observation_id=f"ext_{index}",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=False,
        )
        for index in range(3)
    )

    extended = build_conditioning_memory(
        memory_id="memory_ext",
        cue=cue,
        observations=(
            *conditioned_memory.observations,
            *extinction_observations,
        ),
    )

    association = association_by_attractor_id(
        extended,
        "sensitization_attractor",
    )

    assert association.extinction_exposure_count == 3


# =============================================================================
# Multiple outcomes
# =============================================================================


def test_memory_can_hold_two_attractor_associations(
    cue,
    sens_trajectory,
    adapt_trajectory,
) -> None:
    observations = (
        observation_from_trajectory(
            observation_id="s1",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
        ),
        observation_from_trajectory(
            observation_id="s2",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
        ),
        observation_from_trajectory(
            observation_id="a1",
            cue=cue,
            trajectory=adapt_trajectory,
            reinforcement_present=True,
        ),
    )

    memory = build_conditioning_memory(
        memory_id="multi",
        cue=cue,
        observations=observations,
    )

    assert len(
        memory.associations
    ) == 2


def test_stronger_recurrence_can_dominate(
    cue,
    sens_trajectory,
    adapt_trajectory,
) -> None:
    observations = (
        observation_from_trajectory(
            observation_id="s1",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
        ),
        observation_from_trajectory(
            observation_id="s2",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
        ),
        observation_from_trajectory(
            observation_id="s3",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
        ),
        observation_from_trajectory(
            observation_id="a1",
            cue=cue,
            trajectory=adapt_trajectory,
            reinforcement_present=True,
        ),
    )

    memory = build_conditioning_memory(
        memory_id="multi",
        cue=cue,
        observations=observations,
    )

    assert dominant_conditioned_attractor(
        memory
    ) == "sensitization_attractor"


# =============================================================================
# Generalization
# =============================================================================


def test_identical_cue_generalization_weight_is_one(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert response.generalization_weight == pytest.approx(
        1.0
    )


def test_similar_cue_generalizes(
    similar_cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=similar_cue,
        memory=conditioned_memory,
    )

    assert 0.0 < response.generalization_weight < 1.0


def test_distant_cue_has_lower_generalization_than_similar(
    similar_cue,
    distant_cue,
    conditioned_memory,
) -> None:
    similar = conditioned_response(
        query_cue=similar_cue,
        memory=conditioned_memory,
    )

    distant = conditioned_response(
        query_cue=distant_cue,
        memory=conditioned_memory,
    )

    assert (
        distant.generalization_weight
        < similar.generalization_weight
    )


def test_distant_cue_has_weaker_conditioned_response(
    similar_cue,
    distant_cue,
    conditioned_memory,
) -> None:
    similar = conditioned_response(
        query_cue=similar_cue,
        memory=conditioned_memory,
    )

    distant = conditioned_response(
        query_cue=distant_cue,
        memory=conditioned_memory,
    )

    assert (
        distant.conditioned_response_strength
        < similar.conditioned_response_strength
    )


def test_conditioned_response_type(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert isinstance(
        response,
        ConditionedResponse,
    )


def test_conditioned_response_predicts_sensitization(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert (
        response.predicted_attractor_id
        == "sensitization_attractor"
    )


def test_conditioned_response_strength_positive(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert response.conditioned_response_strength > 0.0


def test_invalid_generalization_scale_rejected(
    cue,
    conditioned_memory,
) -> None:
    with pytest.raises(
        ExperienceConditioningError
    ):
        conditioned_response(
            query_cue=cue,
            memory=conditioned_memory,
            generalization_scale=0.0,
        )


def test_mismatched_cue_dimensions_rejected(
    conditioned_memory,
) -> None:
    other = ConditioningCue(
        cue_id="other",
        cue_type="visual_symbol",
        feature_vector=(1.0, 0.2),
    )

    with pytest.raises(
        ExperienceConditioningError
    ):
        conditioned_response(
            query_cue=other,
            memory=conditioned_memory,
        )


# =============================================================================
# Association helpers
# =============================================================================


def test_association_by_id_returns_expected(
    conditioned_memory,
) -> None:
    association = association_by_attractor_id(
        conditioned_memory,
        "sensitization_attractor",
    )

    assert (
        association.attractor_id
        == "sensitization_attractor"
    )


def test_association_by_id_rejects_unknown(
    conditioned_memory,
) -> None:
    with pytest.raises(
        ExperienceConditioningError
    ):
        association_by_attractor_id(
            conditioned_memory,
            "missing",
        )


def test_strength_series_is_tuple(
    conditioned_memory,
) -> None:
    assert isinstance(
        association_strength_series(
            conditioned_memory
        ),
        tuple,
    )


def test_confidence_series_is_tuple(
    conditioned_memory,
) -> None:
    assert isinstance(
        association_confidence_series(
            conditioned_memory
        ),
        tuple,
    )


def test_strength_series_has_one_value(
    conditioned_memory,
) -> None:
    assert len(
        association_strength_series(
            conditioned_memory
        )
    ) == 1


def test_confidence_series_has_one_value(
    conditioned_memory,
) -> None:
    assert len(
        association_confidence_series(
            conditioned_memory
        )
    ) == 1


# =============================================================================
# Validation / boundaries
# =============================================================================


def test_build_memory_rejects_empty_observations(
    cue,
) -> None:
    with pytest.raises(
        ExperienceConditioningError
    ):
        build_conditioning_memory(
            memory_id="x",
            cue=cue,
            observations=(),
        )


def test_build_memory_rejects_mixed_cues(
    cue,
    similar_cue,
    sens_trajectory,
) -> None:
    observations = (
        observation_from_trajectory(
            observation_id="a",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
        ),
        observation_from_trajectory(
            observation_id="b",
            cue=similar_cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
        ),
    )

    with pytest.raises(
        ExperienceConditioningError
    ):
        build_conditioning_memory(
            memory_id="x",
            cue=cue,
            observations=observations,
        )


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_conditioning_memory_is_policy_free(
    conditioned_memory,
) -> None:
    assert conditioning_memory_is_policy_free(
        conditioned_memory
    ) is True


def test_conditioned_response_is_policy_free(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert conditioned_response_is_policy_free(
        response
    ) is True


def test_memory_declares_no_source_memory_mutation(
    conditioned_memory,
) -> None:
    assert conditioned_memory.metadata[
        "memory_mutated"
    ] is False


def test_memory_declares_no_policy_modified(
    conditioned_memory,
) -> None:
    assert conditioned_memory.metadata[
        "policy_modified"
    ] is False


def test_memory_declares_no_action_selected(
    conditioned_memory,
) -> None:
    assert conditioned_memory.metadata[
        "action_selected"
    ] is False


def test_memory_declares_no_diagnosis(
    conditioned_memory,
) -> None:
    assert conditioned_memory.metadata[
        "diagnosis_generated"
    ] is False


def test_memory_declares_no_biological_claim(
    conditioned_memory,
) -> None:
    assert conditioned_memory.metadata[
        "biological_conditioning_claimed"
    ] is False


def test_memory_declares_no_causal_truth(
    conditioned_memory,
) -> None:
    assert conditioned_memory.metadata[
        "causal_truth_inferred"
    ] is False


def test_response_declares_no_biological_claim(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert response.metadata[
        "biological_conditioning_claimed"
    ] is False


def test_every_observation_declares_no_memory_mutation(
    conditioned_memory,
) -> None:
    assert all(
        observation.metadata[
            "memory_mutated"
        ] is False
        for observation
        in conditioned_memory.observations
    )


# =============================================================================
# Immutability
# =============================================================================


def test_memory_observations_are_tuple(
    conditioned_memory,
) -> None:
    assert isinstance(
        conditioned_memory.observations,
        tuple,
    )


def test_memory_associations_are_tuple(
    conditioned_memory,
) -> None:
    assert isinstance(
        conditioned_memory.associations,
        tuple,
    )


def test_cue_is_frozen(
    cue,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        cue.cue_id = "changed"  # type: ignore[misc]


def test_observation_is_frozen(
    reinforced_observations,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        reinforced_observations[
            0
        ].observation_id = "changed"  # type: ignore[misc]


def test_association_is_frozen(
    conditioned_memory,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        conditioned_memory.associations[
            0
        ].exposure_count = 99  # type: ignore[misc]


def test_memory_is_frozen(
    conditioned_memory,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        conditioned_memory.memory_id = "changed"  # type: ignore[misc]


def test_response_is_frozen(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        response.cue_distance = 0.0  # type: ignore[misc]


def test_memory_metadata_is_read_only(
    conditioned_memory,
) -> None:
    assert isinstance(
        conditioned_memory.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        conditioned_memory.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_association_metadata_is_read_only(
    conditioned_memory,
) -> None:
    assert isinstance(
        conditioned_memory.associations[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        conditioned_memory.associations[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_response_metadata_is_read_only(
    cue,
    conditioned_memory,
) -> None:
    response = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert isinstance(
        response.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        response.metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_observation_from_trajectory_is_deterministic(
    cue,
    sens_trajectory,
) -> None:
    left = observation_from_trajectory(
        observation_id="same",
        cue=cue,
        trajectory=sens_trajectory,
        reinforcement_present=True,
    )

    right = observation_from_trajectory(
        observation_id="same",
        cue=cue,
        trajectory=sens_trajectory,
        reinforcement_present=True,
    )

    assert left == right


def test_memory_build_is_deterministic(
    cue,
    reinforced_observations,
) -> None:
    left = build_conditioning_memory(
        memory_id="same",
        cue=cue,
        observations=reinforced_observations,
    )

    right = build_conditioning_memory(
        memory_id="same",
        cue=cue,
        observations=reinforced_observations,
    )

    assert left == right


def test_conditioned_response_is_deterministic(
    cue,
    conditioned_memory,
) -> None:
    left = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    right = conditioned_response(
        query_cue=cue,
        memory=conditioned_memory,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(
    conditioned_memory,
) -> None:
    assert association_strength_series(
        conditioned_memory
    ) == association_strength_series(
        conditioned_memory
    )

    assert association_confidence_series(
        conditioned_memory
    ) == association_confidence_series(
        conditioned_memory
    )

    assert dominant_conditioned_attractor(
        conditioned_memory
    ) == dominant_conditioned_attractor(
        conditioned_memory
    )
