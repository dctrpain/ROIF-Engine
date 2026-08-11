"""
Tests for ROIF Memory 2.0 — Contextual Reconstruction Core.

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

The suite validates:
same cue + different associative context
-> different reconstruction weights
-> different dominant meaning
-> different reconstructed vector.
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
    ContextualReconstructionError,
    ContextualReconstructionResult,
    ReconstructionContribution,
    SCHEMA_VERSION,
    compare_contextual_reconstructions,
    reconstruct_contextual_meaning,
    reconstruction_contribution_by_id,
    reconstruction_is_policy_free,
    reconstruction_weight_series,
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
        for a, b in zip(
            attractors[0].center,
            attractors[1].center,
        )
    )

    return DynamicsState(
        state_id="midpoint",
        feature_vector=vector,
    )


@pytest.fixture
def sens_bias():
    return AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={
            "sensitization_attractor": 0.75,
        },
    )


@pytest.fixture
def adapt_bias():
    return AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.75,
        },
    )


@pytest.fixture
def sens_trajectory(attractors, initial_state, sens_bias):
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
def adapt_trajectory(attractors, initial_state, adapt_bias):
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
def zero_context(cue_door, sens_memory):
    return build_associative_context(
        context_id="zero_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_door,
                memory=sens_memory,
                salience=0.0,
            ),
        ),
    )


@pytest.fixture
def sens_reconstruction(cue_door, sens_context, attractors):
    return reconstruct_contextual_meaning(
        reconstruction_id="sens_reconstruction",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
    )


@pytest.fixture
def adapt_reconstruction(cue_door, adapt_context, attractors):
    return reconstruct_contextual_meaning(
        reconstruction_id="adapt_reconstruction",
        cue=cue_door,
        context=adapt_context,
        attractors=attractors,
    )


# =============================================================================
# Schema / construction
# =============================================================================


def test_schema_version():
    assert SCHEMA_VERSION == "contextual_reconstruction_v1"


def test_reconstruction_returns_expected_type(sens_reconstruction):
    assert isinstance(
        sens_reconstruction,
        ContextualReconstructionResult,
    )


def test_contributions_are_tuple(sens_reconstruction):
    assert isinstance(
        sens_reconstruction.contributions,
        tuple,
    )


def test_contributions_are_expected_type(sens_reconstruction):
    assert all(
        isinstance(item, ReconstructionContribution)
        for item in sens_reconstruction.contributions
    )


def test_two_attractors_produce_two_contributions(sens_reconstruction):
    assert len(sens_reconstruction.contributions) == 2


def test_rejects_empty_attractors(cue_door, sens_context):
    with pytest.raises(ContextualReconstructionError):
        reconstruct_contextual_meaning(
            reconstruction_id="x",
            cue=cue_door,
            context=sens_context,
            attractors=(),
        )


def test_rejects_empty_reconstruction_id(cue_door, sens_context, attractors):
    with pytest.raises(ContextualReconstructionError):
        reconstruct_contextual_meaning(
            reconstruction_id="",
            cue=cue_door,
            context=sens_context,
            attractors=attractors,
        )


def test_rejects_nonpositive_compatibility_scale(cue_door, sens_context, attractors):
    with pytest.raises(ContextualReconstructionError):
        reconstruct_contextual_meaning(
            reconstruction_id="x",
            cue=cue_door,
            context=sens_context,
            attractors=attractors,
            compatibility_scale=0.0,
        )


def test_rejects_negative_baseline_cue_weight(cue_door, sens_context, attractors):
    with pytest.raises(ContextualReconstructionError):
        reconstruct_contextual_meaning(
            reconstruction_id="x",
            cue=cue_door,
            context=sens_context,
            attractors=attractors,
            baseline_cue_weight=-0.1,
        )


# =============================================================================
# Weight geometry
# =============================================================================


def test_weight_series_is_tuple(sens_reconstruction):
    assert isinstance(
        reconstruction_weight_series(
            sens_reconstruction
        ),
        tuple,
    )


def test_weights_sum_to_one(sens_reconstruction):
    assert sum(
        reconstruction_weight_series(
            sens_reconstruction
        )
    ) == pytest.approx(1.0)


def test_all_weights_are_bounded(sens_reconstruction):
    assert all(
        0.0 <= value <= 1.0
        for value in reconstruction_weight_series(
            sens_reconstruction
        )
    )


def test_sens_context_gives_sens_positive_context_bias(sens_reconstruction):
    contribution = reconstruction_contribution_by_id(
        sens_reconstruction,
        "sensitization_attractor",
    )

    assert contribution.context_bias > 0.0


def test_sens_context_gives_adapt_zero_context_bias(sens_reconstruction):
    contribution = reconstruction_contribution_by_id(
        sens_reconstruction,
        "adaptation_attractor",
    )

    assert contribution.context_bias == pytest.approx(0.0)


def test_adapt_context_gives_adapt_positive_context_bias(adapt_reconstruction):
    contribution = reconstruction_contribution_by_id(
        adapt_reconstruction,
        "adaptation_attractor",
    )

    assert contribution.context_bias > 0.0


def test_adapt_context_gives_sens_zero_context_bias(adapt_reconstruction):
    contribution = reconstruction_contribution_by_id(
        adapt_reconstruction,
        "sensitization_attractor",
    )

    assert contribution.context_bias == pytest.approx(0.0)


def test_lookup_rejects_unknown(sens_reconstruction):
    with pytest.raises(ContextualReconstructionError):
        reconstruction_contribution_by_id(
            sens_reconstruction,
            "missing",
        )


# =============================================================================
# Same cue, different context
# =============================================================================


def test_same_cue_is_preserved_across_contexts(
    sens_reconstruction,
    adapt_reconstruction,
):
    assert sens_reconstruction.cue == adapt_reconstruction.cue


def test_contexts_are_different(
    sens_reconstruction,
    adapt_reconstruction,
):
    assert (
        sens_reconstruction.context.context_id
        != adapt_reconstruction.context.context_id
    )


def test_different_contexts_produce_different_weight_series(
    sens_reconstruction,
    adapt_reconstruction,
):
    assert reconstruction_weight_series(
        sens_reconstruction
    ) != reconstruction_weight_series(
        adapt_reconstruction
    )


def test_different_contexts_produce_different_reconstructed_vectors(
    sens_reconstruction,
    adapt_reconstruction,
):
    assert (
        sens_reconstruction.reconstructed_vector
        != adapt_reconstruction.reconstructed_vector
    )


def test_contextual_reconstruction_distance_positive(
    sens_reconstruction,
    adapt_reconstruction,
):
    assert compare_contextual_reconstructions(
        sens_reconstruction,
        adapt_reconstruction,
    ) > 0.0


def test_sens_context_reconstruction_dominant_sensitization(
    sens_reconstruction,
):
    assert (
        sens_reconstruction.dominant_attractor_id
        == "sensitization_attractor"
    )


def test_adapt_context_reconstruction_dominant_adaptation(
    adapt_reconstruction,
):
    assert (
        adapt_reconstruction.dominant_attractor_id
        == "adaptation_attractor"
    )


def test_different_contexts_change_dominant_meaning(
    sens_reconstruction,
    adapt_reconstruction,
):
    assert (
        sens_reconstruction.dominant_attractor_id
        != adapt_reconstruction.dominant_attractor_id
    )


# =============================================================================
# Reconstruction vector / semantic shift
# =============================================================================


def test_reconstructed_vector_is_tuple(sens_reconstruction):
    assert isinstance(
        sens_reconstruction.reconstructed_vector,
        tuple,
    )


def test_reconstructed_vector_dimension_matches_attractor_space(
    sens_reconstruction,
    attractors,
):
    assert len(
        sens_reconstruction.reconstructed_vector
    ) == len(
        attractors[0].center
    )


def test_semantic_shift_is_nonnegative(sens_reconstruction):
    assert sens_reconstruction.semantic_shift_from_cue >= 0.0


def test_semantic_shift_is_positive_for_sens_context(sens_reconstruction):
    assert sens_reconstruction.semantic_shift_from_cue > 0.0


def test_semantic_shift_is_positive_for_adapt_context(adapt_reconstruction):
    assert adapt_reconstruction.semantic_shift_from_cue > 0.0


def test_dominant_weight_bounded(sens_reconstruction):
    assert 0.0 <= sens_reconstruction.dominant_weight <= 1.0


def test_ambiguity_bounded(sens_reconstruction):
    assert 0.0 <= sens_reconstruction.ambiguity <= 1.0


def test_two_attractor_ambiguity_nonnegative(sens_reconstruction):
    assert sens_reconstruction.ambiguity >= 0.0


# =============================================================================
# Baseline cue weight
# =============================================================================


def test_larger_baseline_cue_weight_reduces_semantic_shift(
    cue_door,
    sens_context,
    attractors,
):
    low = reconstruct_contextual_meaning(
        reconstruction_id="low",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        baseline_cue_weight=0.0,
    )

    high = reconstruct_contextual_meaning(
        reconstruction_id="high",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        baseline_cue_weight=10.0,
    )

    assert (
        high.semantic_shift_from_cue
        < low.semantic_shift_from_cue
    )


def test_zero_baseline_cue_weight_is_supported(
    cue_door,
    sens_context,
    attractors,
):
    result = reconstruct_contextual_meaning(
        reconstruction_id="zero_baseline",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
        baseline_cue_weight=0.0,
    )

    assert result.semantic_shift_from_cue > 0.0


# =============================================================================
# Zero-context control
# =============================================================================


def test_zero_context_has_zero_context_bias_for_target(
    cue_door,
    zero_context,
    attractors,
):
    result = reconstruct_contextual_meaning(
        reconstruction_id="zero_context",
        cue=cue_door,
        context=zero_context,
        attractors=attractors,
    )

    contribution = reconstruction_contribution_by_id(
        result,
        "sensitization_attractor",
    )

    assert contribution.context_bias == pytest.approx(0.0)


def test_zero_context_reconstruction_is_deterministic(
    cue_door,
    zero_context,
    attractors,
):
    left = reconstruct_contextual_meaning(
        reconstruction_id="zero",
        cue=cue_door,
        context=zero_context,
        attractors=attractors,
    )

    right = reconstruct_contextual_meaning(
        reconstruction_id="zero",
        cue=cue_door,
        context=zero_context,
        attractors=attractors,
    )

    assert left == right


# =============================================================================
# Same cue comparison guard
# =============================================================================


def test_compare_rejects_different_cues(
    sens_reconstruction,
    cue_safe,
    sens_context,
    attractors,
):
    other = reconstruct_contextual_meaning(
        reconstruction_id="other",
        cue=cue_safe,
        context=sens_context,
        attractors=attractors,
    )

    with pytest.raises(ContextualReconstructionError):
        compare_contextual_reconstructions(
            sens_reconstruction,
            other,
        )


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_reconstruction_is_policy_free(sens_reconstruction):
    assert reconstruction_is_policy_free(
        sens_reconstruction
    ) is True


def test_result_declares_no_memory_mutation(sens_reconstruction):
    assert sens_reconstruction.metadata[
        "memory_mutated"
    ] is False


def test_result_declares_no_learning(sens_reconstruction):
    assert sens_reconstruction.metadata[
        "learning_applied"
    ] is False


def test_result_declares_no_action_selected(sens_reconstruction):
    assert sens_reconstruction.metadata[
        "action_selected"
    ] is False


def test_result_declares_no_policy_modified(sens_reconstruction):
    assert sens_reconstruction.metadata[
        "policy_modified"
    ] is False


def test_result_declares_no_diagnosis(sens_reconstruction):
    assert sens_reconstruction.metadata[
        "diagnosis_generated"
    ] is False


def test_result_declares_no_biological_claim(sens_reconstruction):
    assert sens_reconstruction.metadata[
        "biological_reconstruction_claimed"
    ] is False


def test_result_declares_no_causal_truth(sens_reconstruction):
    assert sens_reconstruction.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_contribution_declares_no_memory_mutation(
    sens_reconstruction,
):
    assert all(
        contribution.metadata[
            "memory_mutated"
        ] is False
        for contribution in sens_reconstruction.contributions
    )


def test_every_contribution_declares_no_learning(
    sens_reconstruction,
):
    assert all(
        contribution.metadata[
            "learning_applied"
        ] is False
        for contribution in sens_reconstruction.contributions
    )


# =============================================================================
# Source immutability
# =============================================================================


def test_reconstruction_does_not_mutate_context(
    cue_door,
    sens_context,
    attractors,
):
    before_contributions = sens_context.contributions
    before_aggregates = sens_context.aggregates
    before_bias = dict(
        sens_context.prestress.bias_by_attractor_id
    )

    reconstruct_contextual_meaning(
        reconstruction_id="immutability",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
    )

    assert sens_context.contributions == before_contributions
    assert sens_context.aggregates == before_aggregates
    assert dict(
        sens_context.prestress.bias_by_attractor_id
    ) == before_bias


def test_reconstruction_does_not_mutate_attractors(
    cue_door,
    sens_context,
    attractors,
):
    before = tuple(
        attractor.center
        for attractor in attractors
    )

    reconstruct_contextual_meaning(
        reconstruction_id="immutability",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
    )

    after = tuple(
        attractor.center
        for attractor in attractors
    )

    assert after == before


# =============================================================================
# Immutability
# =============================================================================


def test_result_is_frozen(sens_reconstruction):
    with pytest.raises(FrozenInstanceError):
        sens_reconstruction.dominant_weight = 0.0  # type: ignore[misc]


def test_contribution_is_frozen(sens_reconstruction):
    with pytest.raises(FrozenInstanceError):
        sens_reconstruction.contributions[
            0
        ].context_bias = 0.0  # type: ignore[misc]


def test_result_metadata_is_read_only(sens_reconstruction):
    assert isinstance(
        sens_reconstruction.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        sens_reconstruction.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_contribution_metadata_is_read_only(sens_reconstruction):
    assert isinstance(
        sens_reconstruction.contributions[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        sens_reconstruction.contributions[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_sens_reconstruction_is_deterministic(
    cue_door,
    sens_context,
    attractors,
):
    left = reconstruct_contextual_meaning(
        reconstruction_id="same",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
    )

    right = reconstruct_contextual_meaning(
        reconstruction_id="same",
        cue=cue_door,
        context=sens_context,
        attractors=attractors,
    )

    assert left == right


def test_adapt_reconstruction_is_deterministic(
    cue_door,
    adapt_context,
    attractors,
):
    left = reconstruct_contextual_meaning(
        reconstruction_id="same_adapt",
        cue=cue_door,
        context=adapt_context,
        attractors=attractors,
    )

    right = reconstruct_contextual_meaning(
        reconstruction_id="same_adapt",
        cue=cue_door,
        context=adapt_context,
        attractors=attractors,
    )

    assert left == right


def test_comparison_is_deterministic(
    sens_reconstruction,
    adapt_reconstruction,
):
    assert compare_contextual_reconstructions(
        sens_reconstruction,
        adapt_reconstruction,
    ) == compare_contextual_reconstructions(
        sens_reconstruction,
        adapt_reconstruction,
    )


def test_reporting_helpers_are_deterministic(
    sens_reconstruction,
):
    assert reconstruction_weight_series(
        sens_reconstruction
    ) == reconstruction_weight_series(
        sens_reconstruction
    )

    assert reconstruction_contribution_by_id(
        sens_reconstruction,
        "sensitization_attractor",
    ) == reconstruction_contribution_by_id(
        sens_reconstruction,
        "sensitization_attractor",
    )
