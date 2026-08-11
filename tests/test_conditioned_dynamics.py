"""
Tests for ROIF Memory 2.0 — Conditioned Dynamics Core.

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

The suite validates the closed descriptive loop:
cue -> conditioned prestress -> altered attractor competition -> state divergence.
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
    ConditioningCue,
    build_conditioning_memory,
    observation_from_trajectory,
)
from roif.history.conditioned_prestress import (
    ConditionedPrestressConfig,
)
from roif.history.conditioned_dynamics import (
    CompetitionDelta,
    ConditionedDynamicsError,
    ConditionedDynamicsResult,
    SCHEMA_VERSION,
    capture_strength_delta_series,
    competition_delta_by_id,
    competition_weight_delta_series,
    conditioned_dynamics_is_policy_free,
    conditioned_target_capture_gain,
    conditioned_target_weight_gain,
    run_conditioned_dynamics,
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
        magnitudes=(0.20 + offset, 0.45 + offset),
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
def conditioning_memory(
    cue,
    sens_trajectory,
):
    observations = tuple(
        observation_from_trajectory(
            observation_id=f"obs_{index}",
            cue=cue,
            trajectory=sens_trajectory,
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for index in range(5)
    )

    return build_conditioning_memory(
        memory_id="memory_door",
        cue=cue,
        observations=observations,
    )


@pytest.fixture
def default_result(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> ConditionedDynamicsResult:
    return run_conditioned_dynamics(
        result_id="conditioned_run",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        step_size=0.25,
    )


# =============================================================================
# Schema / construction
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "conditioned_dynamics_v1"


def test_run_returns_conditioned_dynamics_result(
    default_result,
) -> None:
    assert isinstance(
        default_result,
        ConditionedDynamicsResult,
    )


def test_competition_deltas_are_tuple(
    default_result,
) -> None:
    assert isinstance(
        default_result.competition_deltas,
        tuple,
    )


def test_competition_deltas_have_two_members(
    default_result,
) -> None:
    assert len(
        default_result.competition_deltas
    ) == 2


def test_competition_deltas_are_competition_delta_objects(
    default_result,
) -> None:
    assert all(
        isinstance(
            delta,
            CompetitionDelta,
        )
        for delta in default_result.competition_deltas
    )


def test_run_rejects_empty_attractors(
    initial_state,
    cue,
    conditioning_memory,
) -> None:
    with pytest.raises(
        ConditionedDynamicsError
    ):
        run_conditioned_dynamics(
            result_id="x",
            state=initial_state,
            attractors=(),
            query_cue=cue,
            conditioning_memory=conditioning_memory,
        )


def test_run_rejects_empty_result_id(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    with pytest.raises(
        ConditionedDynamicsError
    ):
        run_conditioned_dynamics(
            result_id="",
            state=initial_state,
            attractors=attractors,
            query_cue=cue,
            conditioning_memory=conditioning_memory,
        )


# =============================================================================
# Shared initial conditions
# =============================================================================


def test_baseline_and_conditioned_share_same_state_before(
    default_result,
) -> None:
    assert (
        default_result.baseline.state_before
        == default_result.conditioned.state_before
        == default_result.state_before
    )


def test_query_cue_preserved(
    default_result,
    cue,
) -> None:
    assert default_result.query_cue == cue


def test_conditioned_prestress_targets_sensitization(
    default_result,
) -> None:
    target_ids = tuple(
        default_result.conditioned_prestress.prestress.bias_by_attractor_id.keys()
    )

    assert target_ids == (
        "sensitization_attractor",
    )


# =============================================================================
# Competition effect
# =============================================================================


def test_conditioned_target_weight_gain_positive(
    default_result,
) -> None:
    assert conditioned_target_weight_gain(
        default_result
    ) > 0.0


def test_conditioned_target_capture_gain_positive(
    default_result,
) -> None:
    assert conditioned_target_capture_gain(
        default_result
    ) > 0.0


def test_sensitization_weight_delta_positive(
    default_result,
) -> None:
    delta = competition_delta_by_id(
        default_result,
        "sensitization_attractor",
    )

    assert delta.weight_delta > 0.0


def test_adaptation_weight_delta_negative(
    default_result,
) -> None:
    delta = competition_delta_by_id(
        default_result,
        "adaptation_attractor",
    )

    assert delta.weight_delta < 0.0


def test_weight_deltas_sum_to_zero(
    default_result,
) -> None:
    assert sum(
        competition_weight_delta_series(
            default_result
        )
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_conditioned_sensitization_capture_exceeds_baseline(
    default_result,
) -> None:
    delta = competition_delta_by_id(
        default_result,
        "sensitization_attractor",
    )

    assert (
        delta.conditioned_capture_strength
        > delta.baseline_capture_strength
    )


def test_competition_delta_lookup_rejects_unknown(
    default_result,
) -> None:
    with pytest.raises(
        ConditionedDynamicsError
    ):
        competition_delta_by_id(
            default_result,
            "missing",
        )


# =============================================================================
# Dominance / competition geometry
# =============================================================================


def test_conditioned_run_dominant_is_sensitization(
    default_result,
) -> None:
    assert (
        default_result.conditioned.dominant_attractor_id
        == "sensitization_attractor"
    )


def test_conditioned_capture_margin_exceeds_baseline(
    default_result,
) -> None:
    assert default_result.capture_margin_delta > 0.0


def test_conditioned_entropy_is_lower_than_baseline(
    default_result,
) -> None:
    assert default_result.entropy_proxy_delta < 0.0


def test_state_after_diverges_from_baseline(
    default_result,
) -> None:
    assert default_result.state_after_distance > 0.0


def test_baseline_and_conditioned_state_after_differ(
    default_result,
) -> None:
    assert (
        default_result.baseline.state_after.feature_vector
        != default_result.conditioned.state_after.feature_vector
    )


# =============================================================================
# Generalization effect
# =============================================================================


def test_similar_cue_has_positive_target_weight_gain(
    initial_state,
    attractors,
    similar_cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="similar",
        state=initial_state,
        attractors=attractors,
        query_cue=similar_cue,
        conditioning_memory=conditioning_memory,
    )

    assert conditioned_target_weight_gain(
        result
    ) > 0.0


def test_similar_cue_causes_less_state_divergence_than_identical(
    initial_state,
    attractors,
    cue,
    similar_cue,
    conditioning_memory,
) -> None:
    identical = run_conditioned_dynamics(
        result_id="identical",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
    )

    similar = run_conditioned_dynamics(
        result_id="similar",
        state=initial_state,
        attractors=attractors,
        query_cue=similar_cue,
        conditioning_memory=conditioning_memory,
    )

    assert (
        similar.state_after_distance
        < identical.state_after_distance
    )


def test_distant_cue_causes_less_state_divergence_than_similar(
    initial_state,
    attractors,
    similar_cue,
    distant_cue,
    conditioning_memory,
) -> None:
    similar = run_conditioned_dynamics(
        result_id="similar",
        state=initial_state,
        attractors=attractors,
        query_cue=similar_cue,
        conditioning_memory=conditioning_memory,
    )

    distant = run_conditioned_dynamics(
        result_id="distant",
        state=initial_state,
        attractors=attractors,
        query_cue=distant_cue,
        conditioning_memory=conditioning_memory,
    )

    assert (
        distant.state_after_distance
        < similar.state_after_distance
    )


def test_distant_cue_target_weight_gain_lower_than_identical(
    initial_state,
    attractors,
    cue,
    distant_cue,
    conditioning_memory,
) -> None:
    identical = run_conditioned_dynamics(
        result_id="identical",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
    )

    distant = run_conditioned_dynamics(
        result_id="distant",
        state=initial_state,
        attractors=attractors,
        query_cue=distant_cue,
        conditioning_memory=conditioning_memory,
    )

    assert (
        conditioned_target_weight_gain(
            distant
        )
        < conditioned_target_weight_gain(
            identical
        )
    )


# =============================================================================
# Threshold control
# =============================================================================


def test_threshold_can_suppress_conditioned_prestress(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="suppressed",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        prestress_config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert (
        result.conditioned_prestress.suppressed_by_threshold
        is True
    )


def test_threshold_suppression_zeroes_target_weight_gain(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="suppressed",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        prestress_config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert conditioned_target_weight_gain(
        result
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_threshold_suppression_makes_state_after_match_baseline(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="suppressed",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        prestress_config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert result.state_after_distance == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_threshold_suppression_zeroes_capture_margin_delta(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="suppressed",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        prestress_config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert result.capture_margin_delta == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_threshold_suppression_zeroes_entropy_delta(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="suppressed",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        prestress_config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert result.entropy_proxy_delta == pytest.approx(
        0.0,
        abs=1e-12,
    )


# =============================================================================
# Step-size controls
# =============================================================================


def test_step_size_zero_keeps_state_after_distance_zero(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="zero_step",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        step_size=0.0,
    )

    assert result.state_after_distance == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_step_size_zero_can_still_change_competition(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    result = run_conditioned_dynamics(
        result_id="zero_step",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
        step_size=0.0,
    )

    assert conditioned_target_weight_gain(
        result
    ) > 0.0


# =============================================================================
# Delta series helpers
# =============================================================================


def test_weight_delta_series_is_tuple(
    default_result,
) -> None:
    assert isinstance(
        competition_weight_delta_series(
            default_result
        ),
        tuple,
    )


def test_capture_delta_series_is_tuple(
    default_result,
) -> None:
    assert isinstance(
        capture_strength_delta_series(
            default_result
        ),
        tuple,
    )


def test_weight_delta_series_has_two_values(
    default_result,
) -> None:
    assert len(
        competition_weight_delta_series(
            default_result
        )
    ) == 2


def test_capture_delta_series_has_two_values(
    default_result,
) -> None:
    assert len(
        capture_strength_delta_series(
            default_result
        )
    ) == 2


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_conditioned_dynamics_is_policy_free(
    default_result,
) -> None:
    assert conditioned_dynamics_is_policy_free(
        default_result
    ) is True


def test_result_declares_no_memory_mutation(
    default_result,
) -> None:
    assert default_result.metadata[
        "memory_mutated"
    ] is False


def test_result_declares_no_learning(
    default_result,
) -> None:
    assert default_result.metadata[
        "learning_applied"
    ] is False


def test_result_declares_no_action_selected(
    default_result,
) -> None:
    assert default_result.metadata[
        "action_selected"
    ] is False


def test_result_declares_no_policy_modified(
    default_result,
) -> None:
    assert default_result.metadata[
        "policy_modified"
    ] is False


def test_result_declares_no_diagnosis(
    default_result,
) -> None:
    assert default_result.metadata[
        "diagnosis_generated"
    ] is False


def test_result_declares_no_biological_claim(
    default_result,
) -> None:
    assert default_result.metadata[
        "biological_conditioning_claimed"
    ] is False


def test_result_declares_no_causal_truth(
    default_result,
) -> None:
    assert default_result.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_delta_declares_no_memory_mutation(
    default_result,
) -> None:
    assert all(
        delta.metadata[
            "memory_mutated"
        ] is False
        for delta in default_result.competition_deltas
    )


def test_every_delta_declares_no_policy_modification(
    default_result,
) -> None:
    assert all(
        delta.metadata[
            "policy_modified"
        ] is False
        for delta in default_result.competition_deltas
    )


# =============================================================================
# Source immutability
# =============================================================================


def test_run_does_not_mutate_conditioning_memory(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    before_associations = conditioning_memory.associations
    before_observations = conditioning_memory.observations

    run_conditioned_dynamics(
        result_id="immutability",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
    )

    assert conditioning_memory.associations == before_associations
    assert conditioning_memory.observations == before_observations


def test_run_does_not_mutate_attractor_centers(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    before = tuple(
        attractor.center
        for attractor in attractors
    )

    run_conditioned_dynamics(
        result_id="immutability",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
    )

    after = tuple(
        attractor.center
        for attractor in attractors
    )

    assert after == before


# =============================================================================
# Immutability
# =============================================================================


def test_result_is_frozen(
    default_result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        default_result.state_after_distance = 0.0  # type: ignore[misc]


def test_competition_delta_is_frozen(
    default_result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        default_result.competition_deltas[
            0
        ].weight_delta = 0.0  # type: ignore[misc]


def test_result_metadata_is_read_only(
    default_result,
) -> None:
    assert isinstance(
        default_result.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        default_result.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_delta_metadata_is_read_only(
    default_result,
) -> None:
    assert isinstance(
        default_result.competition_deltas[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        default_result.competition_deltas[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_default_run_is_deterministic(
    initial_state,
    attractors,
    cue,
    conditioning_memory,
) -> None:
    left = run_conditioned_dynamics(
        result_id="same",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
    )

    right = run_conditioned_dynamics(
        result_id="same",
        state=initial_state,
        attractors=attractors,
        query_cue=cue,
        conditioning_memory=conditioning_memory,
    )

    assert left == right


def test_similar_cue_run_is_deterministic(
    initial_state,
    attractors,
    similar_cue,
    conditioning_memory,
) -> None:
    left = run_conditioned_dynamics(
        result_id="same_similar",
        state=initial_state,
        attractors=attractors,
        query_cue=similar_cue,
        conditioning_memory=conditioning_memory,
    )

    right = run_conditioned_dynamics(
        result_id="same_similar",
        state=initial_state,
        attractors=attractors,
        query_cue=similar_cue,
        conditioning_memory=conditioning_memory,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(
    default_result,
) -> None:
    assert competition_weight_delta_series(
        default_result
    ) == competition_weight_delta_series(
        default_result
    )

    assert capture_strength_delta_series(
        default_result
    ) == capture_strength_delta_series(
        default_result
    )

    assert conditioned_target_weight_gain(
        default_result
    ) == conditioned_target_weight_gain(
        default_result
    )

    assert conditioned_target_capture_gain(
        default_result
    ) == conditioned_target_capture_gain(
        default_result
    )
