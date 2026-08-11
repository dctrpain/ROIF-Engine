"""
Tests for ROIF Memory 2.0 — Reconstruction Feedback Core.

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

The suite validates recursive semantic feedback:
previous reconstructed meaning -> temporary attractor bias -> current reconstruction.
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
from roif.history.contextual_reconstruction import (
    reconstruct_contextual_meaning,
)
from roif.history.reconstruction_feedback import (
    FeedbackContribution,
    ReconstructionFeedbackConfig,
    ReconstructionFeedbackError,
    ReconstructionFeedbackResult,
    SCHEMA_VERSION,
    apply_reconstruction_feedback,
    feedback_contribution_by_id,
    feedback_weight_delta_series,
    feedback_weight_series,
    reconstruction_feedback_is_policy_free,
)


# =============================================================================
# Helpers
# =============================================================================


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


# =============================================================================
# Fixtures
# =============================================================================


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
def sens_baseline(cue_door, sens_context, attractors):
    return reconstruct_contextual_meaning(
        reconstruction_id="sens_baseline",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
    )


@pytest.fixture
def default_feedback(cue_door, sens_context, attractors, sens_baseline):
    return apply_reconstruction_feedback(
        feedback_id="default_feedback",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
    )


# =============================================================================
# Schema / config
# =============================================================================


def test_schema_version():
    assert SCHEMA_VERSION == "reconstruction_feedback_v1"


def test_default_config_constructs():
    config = ReconstructionFeedbackConfig()
    assert isinstance(config, ReconstructionFeedbackConfig)


def test_config_rejects_negative_feedback_gain():
    with pytest.raises(ReconstructionFeedbackError):
        ReconstructionFeedbackConfig(feedback_gain=-0.1)


def test_config_rejects_zero_feedback_scale():
    with pytest.raises(ReconstructionFeedbackError):
        ReconstructionFeedbackConfig(feedback_scale=0.0)


def test_config_rejects_negative_feedback_scale():
    with pytest.raises(ReconstructionFeedbackError):
        ReconstructionFeedbackConfig(feedback_scale=-1.0)


def test_config_rejects_negative_max_feedback_multiplier():
    with pytest.raises(ReconstructionFeedbackError):
        ReconstructionFeedbackConfig(max_feedback_multiplier=-1.0)


def test_config_rejects_zero_compatibility_scale():
    with pytest.raises(ReconstructionFeedbackError):
        ReconstructionFeedbackConfig(compatibility_scale=0.0)


def test_config_rejects_negative_context_gain():
    with pytest.raises(ReconstructionFeedbackError):
        ReconstructionFeedbackConfig(context_gain=-0.1)


# =============================================================================
# Construction / baseline
# =============================================================================


def test_apply_returns_expected_type(default_feedback):
    assert isinstance(
        default_feedback,
        ReconstructionFeedbackResult,
    )


def test_contributions_are_tuple(default_feedback):
    assert isinstance(
        default_feedback.contributions,
        tuple,
    )


def test_contributions_have_expected_type(default_feedback):
    assert all(
        isinstance(item, FeedbackContribution)
        for item in default_feedback.contributions
    )


def test_two_attractors_produce_two_contributions(default_feedback):
    assert len(default_feedback.contributions) == 2


def test_apply_rejects_empty_attractors(cue_door, sens_context, sens_baseline):
    with pytest.raises(ReconstructionFeedbackError):
        apply_reconstruction_feedback(
            feedback_id="x",
            cue=cue_door,
            context=sens_context,
            attractors=(),
            previous_reconstructed_vector=sens_baseline.reconstructed_vector,
        )


def test_apply_rejects_empty_feedback_id(cue_door, sens_context, attractors, sens_baseline):
    with pytest.raises(ReconstructionFeedbackError):
        apply_reconstruction_feedback(
            feedback_id="",
            cue=cue_door,
            context=sens_context,
            attractors=attractors,
            previous_reconstructed_vector=sens_baseline.reconstructed_vector,
        )


def test_apply_rejects_empty_previous_vector(cue_door, sens_context, attractors):
    with pytest.raises(ReconstructionFeedbackError):
        apply_reconstruction_feedback(
            feedback_id="x",
            cue=cue_door,
            context=sens_context,
            attractors=attractors,
            previous_reconstructed_vector=(),
        )


def test_apply_rejects_wrong_previous_vector_dimension(cue_door, sens_context, attractors):
    with pytest.raises(ReconstructionFeedbackError):
        apply_reconstruction_feedback(
            feedback_id="x",
            cue=cue_door,
            context=sens_context,
            attractors=attractors,
            previous_reconstructed_vector=(1.0, 2.0),
        )


def test_baseline_is_preserved(default_feedback):
    assert (
        default_feedback.baseline_dominant_attractor_id
        == default_feedback.baseline.dominant_attractor_id
    )


# =============================================================================
# Feedback weight geometry
# =============================================================================


def test_feedback_weight_series_is_tuple(default_feedback):
    assert isinstance(
        feedback_weight_series(default_feedback),
        tuple,
    )


def test_feedback_weights_sum_to_one(default_feedback):
    assert sum(
        feedback_weight_series(default_feedback)
    ) == pytest.approx(1.0)


def test_feedback_weights_are_bounded(default_feedback):
    assert all(
        0.0 <= value <= 1.0
        for value in feedback_weight_series(default_feedback)
    )


def test_weight_delta_series_is_tuple(default_feedback):
    assert isinstance(
        feedback_weight_delta_series(default_feedback),
        tuple,
    )


def test_weight_deltas_sum_to_zero(default_feedback):
    assert sum(
        feedback_weight_delta_series(default_feedback)
    ) == pytest.approx(0.0, abs=1e-12)


def test_feedback_multiplier_is_at_least_one(default_feedback):
    assert all(
        contribution.feedback_multiplier >= 1.0
        for contribution in default_feedback.contributions
    )


def test_feedback_compatibility_is_bounded(default_feedback):
    assert all(
        0.0 <= contribution.feedback_compatibility <= 1.0
        for contribution in default_feedback.contributions
    )


def test_feedback_distances_are_nonnegative(default_feedback):
    assert all(
        contribution.feedback_distance >= 0.0
        for contribution in default_feedback.contributions
    )


def test_lookup_returns_expected(default_feedback):
    item = feedback_contribution_by_id(
        default_feedback,
        "sensitization_attractor",
    )

    assert item.attractor_id == "sensitization_attractor"


def test_lookup_rejects_unknown(default_feedback):
    with pytest.raises(ReconstructionFeedbackError):
        feedback_contribution_by_id(
            default_feedback,
            "missing",
        )


# =============================================================================
# Zero-feedback control
# =============================================================================


def test_zero_feedback_gain_preserves_baseline_weights(
    cue_door,
    sens_context,
    attractors,
    sens_baseline,
):
    result = apply_reconstruction_feedback(
        feedback_id="zero_gain",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )

    baseline_weights = tuple(
        contribution.normalized_reconstruction_weight
        for contribution in result.baseline.contributions
    )

    assert feedback_weight_series(result) == pytest.approx(
        baseline_weights
    )


def test_zero_feedback_gain_zeroes_weight_deltas(
    cue_door,
    sens_context,
    attractors,
    sens_baseline,
):
    result = apply_reconstruction_feedback(
        feedback_id="zero_gain",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )

    assert feedback_weight_delta_series(result) == pytest.approx(
        (0.0, 0.0),
        abs=1e-12,
    )


def test_zero_feedback_gain_preserves_dominant(
    cue_door,
    sens_context,
    attractors,
    sens_baseline,
):
    result = apply_reconstruction_feedback(
        feedback_id="zero_gain",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )

    assert (
        result.feedback_dominant_attractor_id
        == result.baseline_dominant_attractor_id
    )


def test_zero_feedback_gain_reports_no_dominant_change(
    cue_door,
    sens_context,
    attractors,
    sens_baseline,
):
    result = apply_reconstruction_feedback(
        feedback_id="zero_gain",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )

    assert result.dominant_attractor_changed is False


# =============================================================================
# Strong feedback toward attractor centers
# =============================================================================


def test_strong_sens_feedback_makes_sens_dominant(
    cue_door,
    adapt_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="strong_sens",
        cue=cue_door,
        context=adapt_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
        config=ReconstructionFeedbackConfig(
            feedback_gain=20.0,
            max_feedback_multiplier=100.0,
        ),
    )

    assert (
        result.feedback_dominant_attractor_id
        == "sensitization_attractor"
    )


def test_strong_adapt_feedback_makes_adapt_dominant(
    cue_door,
    sens_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="strong_adapt",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[1].center,
        config=ReconstructionFeedbackConfig(
            feedback_gain=20.0,
            max_feedback_multiplier=100.0,
        ),
    )

    assert (
        result.feedback_dominant_attractor_id
        == "adaptation_attractor"
    )


def test_strong_sens_feedback_increases_sens_weight(
    cue_door,
    adapt_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="strong_sens",
        cue=cue_door,
        context=adapt_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
        config=ReconstructionFeedbackConfig(
            feedback_gain=20.0,
            max_feedback_multiplier=100.0,
        ),
    )

    item = feedback_contribution_by_id(
        result,
        "sensitization_attractor",
    )

    assert item.weight_delta > 0.0


def test_strong_adapt_feedback_increases_adapt_weight(
    cue_door,
    sens_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="strong_adapt",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[1].center,
        config=ReconstructionFeedbackConfig(
            feedback_gain=20.0,
            max_feedback_multiplier=100.0,
        ),
    )

    item = feedback_contribution_by_id(
        result,
        "adaptation_attractor",
    )

    assert item.weight_delta > 0.0


def test_feedback_at_exact_center_has_compatibility_one(
    cue_door,
    sens_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="center",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
    )

    item = feedback_contribution_by_id(
        result,
        "sensitization_attractor",
    )

    assert item.feedback_compatibility == pytest.approx(1.0)


# =============================================================================
# Feedback cap
# =============================================================================


def test_max_feedback_multiplier_caps_multiplier(
    cue_door,
    sens_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="capped",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
        config=ReconstructionFeedbackConfig(
            feedback_gain=100.0,
            max_feedback_multiplier=1.5,
        ),
    )

    assert all(
        contribution.feedback_multiplier <= 1.5 + 1e-12
        for contribution in result.contributions
    )


def test_exact_center_hits_cap_when_gain_is_large(
    cue_door,
    sens_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="capped",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
        config=ReconstructionFeedbackConfig(
            feedback_gain=100.0,
            max_feedback_multiplier=1.5,
        ),
    )

    item = feedback_contribution_by_id(
        result,
        "sensitization_attractor",
    )

    assert item.feedback_multiplier == pytest.approx(1.5)


# =============================================================================
# Reconstruction / ambiguity
# =============================================================================


def test_feedback_reconstructed_vector_is_tuple(default_feedback):
    assert isinstance(
        default_feedback.feedback_reconstructed_vector,
        tuple,
    )


def test_feedback_vector_dimension_matches_attractor_space(
    default_feedback,
    attractors,
):
    assert len(
        default_feedback.feedback_reconstructed_vector
    ) == len(
        attractors[0].center
    )


def test_feedback_semantic_displacement_nonnegative(default_feedback):
    assert default_feedback.feedback_semantic_displacement >= 0.0


def test_feedback_ambiguity_bounded(default_feedback):
    assert 0.0 <= default_feedback.feedback_ambiguity <= 1.0


def test_baseline_ambiguity_bounded(default_feedback):
    assert 0.0 <= default_feedback.baseline_ambiguity <= 1.0


def test_ambiguity_delta_matches_difference(default_feedback):
    assert default_feedback.ambiguity_delta == pytest.approx(
        default_feedback.feedback_ambiguity
        - default_feedback.baseline_ambiguity
    )


def test_strong_center_feedback_reduces_ambiguity(
    cue_door,
    sens_context,
    attractors,
):
    result = apply_reconstruction_feedback(
        feedback_id="strong_center",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
        config=ReconstructionFeedbackConfig(
            feedback_gain=20.0,
            max_feedback_multiplier=100.0,
        ),
    )

    assert result.feedback_ambiguity < result.baseline_ambiguity


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_feedback_is_policy_free(default_feedback):
    assert reconstruction_feedback_is_policy_free(
        default_feedback
    ) is True


def test_result_declares_no_memory_mutation(default_feedback):
    assert default_feedback.metadata["memory_mutated"] is False


def test_result_declares_no_learning(default_feedback):
    assert default_feedback.metadata["learning_applied"] is False


def test_result_declares_no_action_selected(default_feedback):
    assert default_feedback.metadata["action_selected"] is False


def test_result_declares_no_policy_modified(default_feedback):
    assert default_feedback.metadata["policy_modified"] is False


def test_result_declares_no_diagnosis(default_feedback):
    assert default_feedback.metadata["diagnosis_generated"] is False


def test_result_declares_no_biological_claim(default_feedback):
    assert default_feedback.metadata["biological_feedback_claimed"] is False


def test_result_declares_no_causal_truth(default_feedback):
    assert default_feedback.metadata["causal_truth_inferred"] is False


def test_every_contribution_declares_no_memory_mutation(default_feedback):
    assert all(
        item.metadata["memory_mutated"] is False
        for item in default_feedback.contributions
    )


def test_every_contribution_declares_no_learning(default_feedback):
    assert all(
        item.metadata["learning_applied"] is False
        for item in default_feedback.contributions
    )


# =============================================================================
# Source immutability
# =============================================================================


def test_feedback_does_not_mutate_context(
    cue_door,
    sens_context,
    attractors,
    sens_baseline,
):
    before_contributions = sens_context.contributions
    before_aggregates = sens_context.aggregates
    before_bias = dict(
        sens_context.prestress.bias_by_attractor_id
    )

    apply_reconstruction_feedback(
        feedback_id="immutability",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
    )

    assert sens_context.contributions == before_contributions
    assert sens_context.aggregates == before_aggregates
    assert dict(
        sens_context.prestress.bias_by_attractor_id
    ) == before_bias


def test_feedback_does_not_mutate_attractors(
    cue_door,
    sens_context,
    attractors,
    sens_baseline,
):
    before = tuple(
        attractor.center
        for attractor in attractors
    )

    apply_reconstruction_feedback(
        feedback_id="immutability",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
    )

    after = tuple(
        attractor.center
        for attractor in attractors
    )

    assert after == before


# =============================================================================
# Immutability
# =============================================================================


def test_config_is_frozen():
    config = ReconstructionFeedbackConfig()

    with pytest.raises(FrozenInstanceError):
        config.feedback_gain = 2.0  # type: ignore[misc]


def test_result_is_frozen(default_feedback):
    with pytest.raises(FrozenInstanceError):
        default_feedback.feedback_ambiguity = 0.0  # type: ignore[misc]


def test_contribution_is_frozen(default_feedback):
    with pytest.raises(FrozenInstanceError):
        default_feedback.contributions[0].weight_delta = 0.0  # type: ignore[misc]


def test_config_metadata_is_read_only():
    config = ReconstructionFeedbackConfig(
        metadata={"x": 1},
    )

    assert isinstance(
        config.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        config.metadata["x"] = 2  # type: ignore[index]


def test_result_metadata_is_read_only(default_feedback):
    assert isinstance(
        default_feedback.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        default_feedback.metadata["x"] = 1  # type: ignore[index]


def test_contribution_metadata_is_read_only(default_feedback):
    assert isinstance(
        default_feedback.contributions[0].metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        default_feedback.contributions[0].metadata["x"] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_default_feedback_is_deterministic(
    cue_door,
    sens_context,
    attractors,
    sens_baseline,
):
    left = apply_reconstruction_feedback(
        feedback_id="same",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
    )

    right = apply_reconstruction_feedback(
        feedback_id="same",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        previous_reconstructed_vector=sens_baseline.reconstructed_vector,
    )

    assert left == right


def test_strong_sens_feedback_is_deterministic(
    cue_door,
    adapt_context,
    attractors,
):
    config = ReconstructionFeedbackConfig(
        feedback_gain=20.0,
        max_feedback_multiplier=100.0,
    )

    left = apply_reconstruction_feedback(
        feedback_id="same_strong",
        cue=cue_door,
        context=adapt_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
        config=config,
    )

    right = apply_reconstruction_feedback(
        feedback_id="same_strong",
        cue=cue_door,
        context=adapt_context,
        attractors=attractors,
        previous_reconstructed_vector=attractors[0].center,
        config=config,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(default_feedback):
    assert feedback_weight_series(
        default_feedback
    ) == feedback_weight_series(
        default_feedback
    )

    assert feedback_weight_delta_series(
        default_feedback
    ) == feedback_weight_delta_series(
        default_feedback
    )

    assert feedback_contribution_by_id(
        default_feedback,
        "sensitization_attractor",
    ) == feedback_contribution_by_id(
        default_feedback,
        "sensitization_attractor",
    )
