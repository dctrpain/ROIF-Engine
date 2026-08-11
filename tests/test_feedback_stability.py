"""
Tests for ROIF Memory 2.0 — Feedback Stability Core.

Validates descriptive stability analysis over FeedbackTrajectory.
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
from roif.history.reconstruction_feedback import ReconstructionFeedbackConfig
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    build_feedback_trajectory,
)
from roif.history.feedback_stability import (
    FeedbackStabilityConfig,
    FeedbackStabilityError,
    FeedbackStabilityResult,
    SCHEMA_VERSION,
    analyze_feedback_stability,
    classify_feedback_regime,
    feedback_stability_is_policy_free,
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
    return build_conditioning_memory(
        memory_id="sens_memory",
        cue=cue_door,
        observations=tuple(
            observation_from_trajectory(
                observation_id=f"sens_obs_{i}",
                cue=cue_door,
                trajectory=sens_trajectory,
                reinforcement_present=True,
                reinforcement_strength=1.0,
            )
            for i in range(5)
        ),
    )


@pytest.fixture
def adapt_memory(cue_safe, adapt_trajectory):
    return build_conditioning_memory(
        memory_id="adapt_memory",
        cue=cue_safe,
        observations=tuple(
            observation_from_trajectory(
                observation_id=f"adapt_obs_{i}",
                cue=cue_safe,
                trajectory=adapt_trajectory,
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
def sss_trajectory(cue_door, sens_context, attractors):
    frames = tuple(
        FeedbackFrame(cue=cue_door, context=sens_context)
        for _ in range(5)
    )
    return build_feedback_trajectory(
        trajectory_id="sss",
        frames=frames,
        attractors=attractors,
    )


@pytest.fixture
def sasas_trajectory(cue_door, sens_context, adapt_context, attractors):
    frames = (
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=adapt_context),
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=adapt_context),
        FeedbackFrame(cue=cue_door, context=sens_context),
    )
    return build_feedback_trajectory(
        trajectory_id="sasas",
        frames=frames,
        attractors=attractors,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )


@pytest.fixture
def sss_stability(sss_trajectory):
    return analyze_feedback_stability(
        analysis_id="sss_analysis",
        trajectory=sss_trajectory,
    )


@pytest.fixture
def sasas_stability(sasas_trajectory):
    return analyze_feedback_stability(
        analysis_id="sasas_analysis",
        trajectory=sasas_trajectory,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "feedback_stability_v1"


def test_default_config_constructs():
    assert isinstance(FeedbackStabilityConfig(), FeedbackStabilityConfig)


def test_config_rejects_negative_drift_epsilon():
    with pytest.raises(FeedbackStabilityError):
        FeedbackStabilityConfig(drift_epsilon=-1.0)


def test_config_rejects_persistent_threshold_above_one():
    with pytest.raises(FeedbackStabilityError):
        FeedbackStabilityConfig(persistent_run_threshold=1.1)


def test_config_rejects_negative_oscillation_threshold():
    with pytest.raises(FeedbackStabilityError):
        FeedbackStabilityConfig(oscillation_threshold=-0.1)


def test_config_rejects_switch_threshold_above_one():
    with pytest.raises(FeedbackStabilityError):
        FeedbackStabilityConfig(high_switch_rate_threshold=1.1)


def test_config_rejects_invalid_slope_order():
    with pytest.raises(FeedbackStabilityError):
        FeedbackStabilityConfig(
            convergence_slope_threshold=1.0,
            divergence_slope_threshold=0.0,
        )


def test_analysis_returns_expected_type(sss_stability):
    assert isinstance(sss_stability, FeedbackStabilityResult)


def test_analysis_preserves_trajectory_id(sss_stability):
    assert sss_stability.trajectory_id == "sss"


def test_analysis_step_count_matches_trajectory(sss_stability, sss_trajectory):
    assert sss_stability.step_count == len(sss_trajectory.steps)


def test_analysis_recursive_step_count_matches_summary(sss_stability, sss_trajectory):
    assert (
        sss_stability.recursive_step_count
        == sss_trajectory.summary.recursive_step_count
    )


def test_switch_rate_bounded(sss_stability):
    assert 0.0 <= sss_stability.switch_rate <= 1.0


def test_return_rate_bounded(sss_stability):
    assert 0.0 <= sss_stability.return_rate <= 1.0


def test_longest_run_fraction_bounded(sss_stability):
    assert 0.0 <= sss_stability.longest_dominant_run_fraction <= 1.0


def test_alternation_score_bounded(sss_stability):
    assert 0.0 <= sss_stability.alternation_score <= 1.0


def test_convergence_score_bounded(sss_stability):
    assert 0.0 <= sss_stability.convergence_score <= 1.0


def test_divergence_score_bounded(sss_stability):
    assert 0.0 <= sss_stability.divergence_score <= 1.0


def test_oscillation_score_bounded(sss_stability):
    assert 0.0 <= sss_stability.oscillation_score <= 1.0


def test_persistence_score_bounded(sss_stability):
    assert 0.0 <= sss_stability.persistence_score <= 1.0


def test_sss_allows_at_most_one_internal_switch(sss_stability):
    assert sss_stability.switch_count <= 1


def test_sss_has_zero_returns(sss_stability):
    assert sss_stability.return_count == 0


def test_sss_uses_at_most_two_unique_dominants(sss_stability):
    assert sss_stability.unique_dominant_count <= 2


def test_sss_longest_run_fraction_is_majority(sss_stability):
    assert sss_stability.longest_dominant_run_fraction >= 0.60


def test_sss_alternation_is_zero(sss_stability):
    assert sss_stability.alternation_score == pytest.approx(0.0)


def test_sss_has_positive_persistence(sss_stability):
    assert sss_stability.persistence_score > 0.0


def test_sss_recursive_drift_is_decreasing(sss_stability):
    assert sss_stability.recursive_drift_slope < 0.0


def test_sss_convergence_exceeds_divergence(sss_stability):
    assert (
        sss_stability.convergence_score
        > sss_stability.divergence_score
    )


def test_sss_is_classified_convergent(sss_stability):
    assert sss_stability.regime == "convergent"


def test_sasas_has_high_switch_rate(sasas_stability):
    assert sasas_stability.switch_rate >= 0.5


def test_sasas_has_positive_return_rate(sasas_stability):
    assert sasas_stability.return_rate > 0.0


def test_sasas_alternation_score_is_one(sasas_stability):
    assert sasas_stability.alternation_score == pytest.approx(1.0)


def test_sasas_classified_oscillatory(sasas_stability):
    assert sasas_stability.regime == "oscillatory"


def test_classify_oscillatory():
    config = FeedbackStabilityConfig()
    assert classify_feedback_regime(
        recursive_drift_slope=0.0,
        terminal_recursive_drift=0.2,
        switch_rate=1.0,
        alternation_score=1.0,
        persistence_score=0.0,
        config=config,
    ) == "oscillatory"


def test_classify_convergent():
    config = FeedbackStabilityConfig(drift_epsilon=0.01)
    assert classify_feedback_regime(
        recursive_drift_slope=-0.1,
        terminal_recursive_drift=0.001,
        mean_recursive_drift=0.05,
        switch_rate=0.0,
        alternation_score=0.0,
        persistence_score=0.5,
        config=config,
    ) == "convergent"


def test_classify_divergent():
    config = FeedbackStabilityConfig()
    assert classify_feedback_regime(
        recursive_drift_slope=0.1,
        terminal_recursive_drift=0.5,
        switch_rate=0.1,
        alternation_score=0.0,
        persistence_score=0.5,
        config=config,
    ) == "divergent"


def test_classify_persistent():
    config = FeedbackStabilityConfig()
    assert classify_feedback_regime(
        recursive_drift_slope=0.0,
        terminal_recursive_drift=0.2,
        switch_rate=0.0,
        alternation_score=0.0,
        persistence_score=0.9,
        config=config,
    ) == "persistent"


def test_classify_switching():
    config = FeedbackStabilityConfig()
    assert classify_feedback_regime(
        recursive_drift_slope=0.0,
        terminal_recursive_drift=0.2,
        switch_rate=0.75,
        alternation_score=0.2,
        persistence_score=0.1,
        config=config,
    ) == "switching"


def test_classify_mixed():
    config = FeedbackStabilityConfig()
    assert classify_feedback_regime(
        recursive_drift_slope=0.0,
        terminal_recursive_drift=0.2,
        switch_rate=0.25,
        alternation_score=0.0,
        persistence_score=0.4,
        config=config,
    ) == "mixed"


def test_mean_recursive_drift_nonnegative(sss_stability):
    assert sss_stability.mean_recursive_drift >= 0.0


def test_terminal_recursive_drift_nonnegative(sss_stability):
    assert sss_stability.terminal_recursive_drift >= 0.0


def test_mean_feedback_displacement_nonnegative(sss_stability):
    assert sss_stability.mean_feedback_displacement >= 0.0


def test_mean_feedback_ambiguity_bounded(sss_stability):
    assert 0.0 <= sss_stability.mean_feedback_ambiguity <= 1.0


def test_analysis_metadata_source_id(sss_stability):
    assert sss_stability.metadata["source_trajectory_id"] == "sss"


def test_analysis_declares_no_memory_mutation(sss_stability):
    assert sss_stability.metadata["memory_mutated"] is False


def test_analysis_declares_no_learning(sss_stability):
    assert sss_stability.metadata["learning_applied"] is False


def test_analysis_declares_no_action_selected(sss_stability):
    assert sss_stability.metadata["action_selected"] is False


def test_analysis_declares_no_policy_modified(sss_stability):
    assert sss_stability.metadata["policy_modified"] is False


def test_analysis_declares_no_diagnosis(sss_stability):
    assert sss_stability.metadata["diagnosis_generated"] is False


def test_analysis_declares_no_biological_claim(sss_stability):
    assert sss_stability.metadata["biological_stability_claimed"] is False


def test_analysis_declares_no_causal_truth(sss_stability):
    assert sss_stability.metadata["causal_truth_inferred"] is False


def test_feedback_stability_is_policy_free(sss_stability):
    assert feedback_stability_is_policy_free(sss_stability) is True


def test_analysis_does_not_mutate_trajectory(sss_trajectory):
    before_steps = sss_trajectory.steps
    before_final = sss_trajectory.final_reconstructed_vector

    analyze_feedback_stability(
        analysis_id="immutability",
        trajectory=sss_trajectory,
    )

    assert sss_trajectory.steps == before_steps
    assert sss_trajectory.final_reconstructed_vector == before_final


def test_config_is_frozen():
    config = FeedbackStabilityConfig()
    with pytest.raises(FrozenInstanceError):
        config.drift_epsilon = 1.0  # type: ignore[misc]


def test_result_is_frozen(sss_stability):
    with pytest.raises(FrozenInstanceError):
        sss_stability.regime = "changed"  # type: ignore[misc]


def test_config_metadata_is_read_only():
    config = FeedbackStabilityConfig(metadata={"x": 1})
    assert isinstance(config.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        config.metadata["x"] = 2  # type: ignore[index]


def test_result_metadata_is_read_only(sss_stability):
    assert isinstance(sss_stability.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        sss_stability.metadata["x"] = 1  # type: ignore[index]


def test_default_analysis_is_deterministic(sss_trajectory):
    left = analyze_feedback_stability(
        analysis_id="same",
        trajectory=sss_trajectory,
    )
    right = analyze_feedback_stability(
        analysis_id="same",
        trajectory=sss_trajectory,
    )
    assert left == right


def test_oscillatory_analysis_is_deterministic(sasas_trajectory):
    left = analyze_feedback_stability(
        analysis_id="same_osc",
        trajectory=sasas_trajectory,
    )
    right = analyze_feedback_stability(
        analysis_id="same_osc",
        trajectory=sasas_trajectory,
    )
    assert left == right


def test_custom_config_analysis_is_deterministic(sss_trajectory):
    config = FeedbackStabilityConfig(
        drift_epsilon=0.05,
        persistent_run_threshold=0.60,
    )
    left = analyze_feedback_stability(
        analysis_id="custom",
        trajectory=sss_trajectory,
        config=config,
    )
    right = analyze_feedback_stability(
        analysis_id="custom",
        trajectory=sss_trajectory,
        config=config,
    )
    assert left == right


def test_regime_classification_is_deterministic():
    config = FeedbackStabilityConfig()
    args = dict(
        recursive_drift_slope=0.0,
        terminal_recursive_drift=0.2,
        switch_rate=0.25,
        alternation_score=0.0,
        persistence_score=0.4,
        config=config,
    )
    assert classify_feedback_regime(**args) == classify_feedback_regime(**args)

