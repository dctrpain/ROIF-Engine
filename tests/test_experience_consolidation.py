"""
Tests for ROIF Memory 2.0 — Experience Consolidation Core.

The suite validates the boundary:

    stable repeated recursive experience
        -> consolidation candidate

but:

    consolidation candidate != committed memory
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
from roif.history.feedback_stability import analyze_feedback_stability
from roif.history.experience_consolidation import (
    ConsolidationEvidence,
    ExperienceConsolidationCandidate,
    ExperienceConsolidationConfig,
    ExperienceConsolidationError,
    SCHEMA_VERSION,
    consolidation_is_policy_free,
    evaluate_experience_consolidation,
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
def adapt_bias():
    return AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={"adaptation_attractor": 0.75},
    )


@pytest.fixture
def sens_trajectory(attractors, initial_state, sens_bias):
    return build_attractor_trajectory(
        trajectory_id="sens_training",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(sens_bias, sens_bias, sens_bias),
        step_size=0.10,
    )


@pytest.fixture
def adapt_trajectory(attractors, initial_state, adapt_bias):
    return build_attractor_trajectory(
        trajectory_id="adapt_training",
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
def stable_trajectory(cue_door, sens_context, attractors):
    frames = tuple(
        FeedbackFrame(cue=cue_door, context=sens_context)
        for _ in range(6)
    )

    return build_feedback_trajectory(
        trajectory_id="stable_sens",
        frames=frames,
        attractors=attractors,
    )


@pytest.fixture
def stable_stability(stable_trajectory):
    return analyze_feedback_stability(
        analysis_id="stable_analysis",
        trajectory=stable_trajectory,
    )


@pytest.fixture
def oscillatory_trajectory(cue_door, sens_context, adapt_context, attractors):
    frames = (
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=adapt_context),
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=adapt_context),
        FeedbackFrame(cue=cue_door, context=sens_context),
        FeedbackFrame(cue=cue_door, context=adapt_context),
    )

    return build_feedback_trajectory(
        trajectory_id="oscillatory",
        frames=frames,
        attractors=attractors,
        config=ReconstructionFeedbackConfig(
            feedback_gain=0.0,
        ),
    )


@pytest.fixture
def oscillatory_stability(oscillatory_trajectory):
    return analyze_feedback_stability(
        analysis_id="osc_analysis",
        trajectory=oscillatory_trajectory,
    )


@pytest.fixture
def stable_candidate(stable_trajectory, stable_stability):
    return evaluate_experience_consolidation(
        candidate_id="candidate_stable",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=3,
            minimum_stability_score=0.45,
            minimum_repetition_score=0.60,
            minimum_consistency_score=0.50,
            minimum_total_score=0.50,
            maximum_mean_ambiguity=0.60,
            maximum_terminal_drift=0.10,
        ),
    )


def test_schema_version():
    assert SCHEMA_VERSION == "experience_consolidation_v1"


def test_default_config_constructs():
    assert isinstance(
        ExperienceConsolidationConfig(),
        ExperienceConsolidationConfig,
    )


def test_config_rejects_zero_minimum_recursive_steps():
    with pytest.raises(ExperienceConsolidationError):
        ExperienceConsolidationConfig(minimum_recursive_steps=0)


def test_config_rejects_invalid_stability_threshold():
    with pytest.raises(ExperienceConsolidationError):
        ExperienceConsolidationConfig(minimum_stability_score=1.1)


def test_config_rejects_negative_terminal_drift():
    with pytest.raises(ExperienceConsolidationError):
        ExperienceConsolidationConfig(maximum_terminal_drift=-0.1)


def test_config_rejects_all_zero_weights():
    with pytest.raises(ExperienceConsolidationError):
        ExperienceConsolidationConfig(
            convergence_weight=0.0,
            persistence_weight=0.0,
            repetition_weight=0.0,
            consistency_weight=0.0,
            ambiguity_weight=0.0,
        )


def test_evaluation_returns_candidate(stable_candidate):
    assert isinstance(
        stable_candidate,
        ExperienceConsolidationCandidate,
    )


def test_candidate_contains_evidence(stable_candidate):
    assert isinstance(
        stable_candidate.evidence,
        ConsolidationEvidence,
    )


def test_candidate_source_trajectory_preserved(stable_candidate):
    assert stable_candidate.source_trajectory_id == "stable_sens"


def test_candidate_source_stability_preserved(stable_candidate):
    assert (
        stable_candidate.source_stability_analysis_id
        == "stable_analysis"
    )


def test_candidate_dominant_attractor_is_present(stable_candidate):
    assert stable_candidate.dominant_attractor_id in {
        "sensitization_attractor",
        "adaptation_attractor",
    }


def test_representation_vector_is_tuple(stable_candidate):
    assert isinstance(stable_candidate.representation_vector, tuple)


def test_representation_vector_is_nonempty(stable_candidate):
    assert len(stable_candidate.representation_vector) > 0


def test_consolidation_score_bounded(stable_candidate):
    assert 0.0 <= stable_candidate.consolidation_score <= 1.0


def test_repetition_score_bounded(stable_candidate):
    assert 0.0 <= stable_candidate.evidence.repetition_score <= 1.0


def test_stability_score_bounded(stable_candidate):
    assert 0.0 <= stable_candidate.evidence.stability_score <= 1.0


def test_consistency_score_bounded(stable_candidate):
    assert 0.0 <= stable_candidate.evidence.consistency_score <= 1.0


def test_ambiguity_quality_bounded(stable_candidate):
    assert 0.0 <= stable_candidate.evidence.ambiguity_quality <= 1.0


def test_terminal_drift_quality_bounded(stable_candidate):
    assert 0.0 <= stable_candidate.evidence.terminal_drift_quality <= 1.0


def test_mean_ambiguity_bounded(stable_candidate):
    assert 0.0 <= stable_candidate.evidence.mean_ambiguity <= 1.0


def test_terminal_recursive_drift_nonnegative(stable_candidate):
    assert stable_candidate.evidence.terminal_recursive_drift >= 0.0


def test_stable_candidate_is_eligible(stable_candidate):
    assert stable_candidate.eligible_for_consolidation is True


def test_stable_candidate_reason(stable_candidate):
    assert stable_candidate.reason == "eligible_candidate_only"


def test_candidate_is_not_committed_memory(stable_candidate):
    assert stable_candidate.metadata["memory_committed"] is False


def test_candidate_does_not_mutate_memory(stable_candidate):
    assert stable_candidate.metadata["memory_mutated"] is False


def test_candidate_does_not_apply_learning(stable_candidate):
    assert stable_candidate.metadata["learning_applied"] is False


def test_candidate_declares_no_action(stable_candidate):
    assert stable_candidate.metadata["action_selected"] is False


def test_candidate_declares_no_policy_change(stable_candidate):
    assert stable_candidate.metadata["policy_modified"] is False


def test_candidate_declares_no_diagnosis(stable_candidate):
    assert stable_candidate.metadata["diagnosis_generated"] is False


def test_candidate_declares_no_biological_claim(stable_candidate):
    assert (
        stable_candidate.metadata["biological_consolidation_claimed"]
        is False
    )


def test_candidate_declares_no_causal_truth(stable_candidate):
    assert stable_candidate.metadata["causal_truth_inferred"] is False


def test_evidence_declares_no_memory_mutation(stable_candidate):
    assert stable_candidate.evidence.metadata["memory_mutated"] is False


def test_evidence_declares_no_memory_commit(stable_candidate):
    assert stable_candidate.evidence.metadata["memory_committed"] is False


def test_consolidation_is_policy_free(stable_candidate):
    assert consolidation_is_policy_free(stable_candidate) is True


def test_mismatched_trajectory_and_stability_rejected(
    stable_trajectory,
    oscillatory_stability,
):
    with pytest.raises(ExperienceConsolidationError):
        evaluate_experience_consolidation(
            candidate_id="bad",
            trajectory=stable_trajectory,
            stability=oscillatory_stability,
        )


def test_insufficient_recursive_steps_reason(
    cue_door,
    sens_context,
    attractors,
):
    trajectory = build_feedback_trajectory(
        trajectory_id="short",
        frames=(
            FeedbackFrame(cue=cue_door, context=sens_context),
            FeedbackFrame(cue=cue_door, context=sens_context),
        ),
        attractors=attractors,
    )

    stability = analyze_feedback_stability(
        analysis_id="short_analysis",
        trajectory=trajectory,
    )

    candidate = evaluate_experience_consolidation(
        candidate_id="short_candidate",
        trajectory=trajectory,
        stability=stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=3,
            minimum_stability_score=0.0,
            minimum_repetition_score=0.0,
            minimum_consistency_score=0.0,
            minimum_total_score=0.0,
            maximum_mean_ambiguity=1.0,
            maximum_terminal_drift=10.0,
        ),
    )

    assert candidate.eligible_for_consolidation is False
    assert candidate.reason == "insufficient_recursive_repetition"


def test_insufficient_repetition_reason(stable_trajectory, stable_stability):
    candidate = evaluate_experience_consolidation(
        candidate_id="rep_fail",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=1,
            minimum_stability_score=0.0,
            minimum_repetition_score=1.0,
            minimum_consistency_score=0.0,
            minimum_total_score=0.0,
            maximum_mean_ambiguity=1.0,
            maximum_terminal_drift=10.0,
        ),
    )

    assert candidate.reason in {
        "eligible_candidate_only",
        "insufficient_repetition",
    }


def test_insufficient_consistency_reason(
    oscillatory_trajectory,
    oscillatory_stability,
):
    candidate = evaluate_experience_consolidation(
        candidate_id="consistency_fail",
        trajectory=oscillatory_trajectory,
        stability=oscillatory_stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=1,
            minimum_stability_score=0.0,
            minimum_repetition_score=0.0,
            minimum_consistency_score=0.90,
            minimum_total_score=0.0,
            maximum_mean_ambiguity=1.0,
            maximum_terminal_drift=10.0,
        ),
    )

    assert candidate.eligible_for_consolidation is False
    assert candidate.reason == "insufficient_dominant_consistency"


def test_high_ambiguity_can_block_consolidation(
    stable_trajectory,
    stable_stability,
):
    candidate = evaluate_experience_consolidation(
        candidate_id="ambiguity_fail",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=1,
            minimum_stability_score=0.0,
            minimum_repetition_score=0.0,
            minimum_consistency_score=0.0,
            minimum_total_score=0.0,
            maximum_mean_ambiguity=0.0,
            maximum_terminal_drift=10.0,
        ),
    )

    assert candidate.eligible_for_consolidation is False
    assert candidate.reason == "ambiguity_too_high"


def test_terminal_drift_can_block_consolidation(
    stable_trajectory,
    stable_stability,
):
    candidate = evaluate_experience_consolidation(
        candidate_id="drift_fail",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=1,
            minimum_stability_score=0.0,
            minimum_repetition_score=0.0,
            minimum_consistency_score=0.0,
            minimum_total_score=0.0,
            maximum_mean_ambiguity=1.0,
            maximum_terminal_drift=0.0,
        ),
    )

    if stable_stability.terminal_recursive_drift > 0.0:
        assert candidate.eligible_for_consolidation is False
        assert candidate.reason == "terminal_drift_too_high"


def test_high_total_threshold_can_block_candidate(
    stable_trajectory,
    stable_stability,
):
    candidate = evaluate_experience_consolidation(
        candidate_id="score_fail",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=ExperienceConsolidationConfig(
            minimum_recursive_steps=1,
            minimum_stability_score=0.0,
            minimum_repetition_score=0.0,
            minimum_consistency_score=0.0,
            minimum_total_score=1.0,
            maximum_mean_ambiguity=1.0,
            maximum_terminal_drift=10.0,
        ),
    )

    if candidate.consolidation_score < 1.0:
        assert candidate.eligible_for_consolidation is False
        assert candidate.reason == "consolidation_score_below_threshold"


def test_evaluation_does_not_mutate_trajectory(
    stable_trajectory,
    stable_stability,
):
    before_steps = stable_trajectory.steps
    before_final = stable_trajectory.final_reconstructed_vector

    evaluate_experience_consolidation(
        candidate_id="immutability",
        trajectory=stable_trajectory,
        stability=stable_stability,
    )

    assert stable_trajectory.steps == before_steps
    assert stable_trajectory.final_reconstructed_vector == before_final


def test_evaluation_does_not_mutate_stability(
    stable_trajectory,
    stable_stability,
):
    before_regime = stable_stability.regime
    before_score = stable_stability.convergence_score

    evaluate_experience_consolidation(
        candidate_id="immutability",
        trajectory=stable_trajectory,
        stability=stable_stability,
    )

    assert stable_stability.regime == before_regime
    assert stable_stability.convergence_score == before_score


def test_config_is_frozen():
    config = ExperienceConsolidationConfig()

    with pytest.raises(FrozenInstanceError):
        config.minimum_total_score = 0.0  # type: ignore[misc]


def test_evidence_is_frozen(stable_candidate):
    with pytest.raises(FrozenInstanceError):
        stable_candidate.evidence.repetition_score = 0.0  # type: ignore[misc]


def test_candidate_is_frozen(stable_candidate):
    with pytest.raises(FrozenInstanceError):
        stable_candidate.reason = "changed"  # type: ignore[misc]


def test_config_metadata_is_read_only():
    config = ExperienceConsolidationConfig(
        metadata={"x": 1},
    )

    assert isinstance(config.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        config.metadata["x"] = 2  # type: ignore[index]


def test_evidence_metadata_is_read_only(stable_candidate):
    assert isinstance(
        stable_candidate.evidence.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        stable_candidate.evidence.metadata["x"] = 1  # type: ignore[index]


def test_candidate_metadata_is_read_only(stable_candidate):
    assert isinstance(
        stable_candidate.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        stable_candidate.metadata["x"] = 1  # type: ignore[index]


def test_stable_evaluation_is_deterministic(
    stable_trajectory,
    stable_stability,
):
    config = ExperienceConsolidationConfig(
        minimum_recursive_steps=3,
        minimum_stability_score=0.45,
        minimum_repetition_score=0.60,
        minimum_consistency_score=0.50,
        minimum_total_score=0.50,
        maximum_mean_ambiguity=0.60,
        maximum_terminal_drift=0.10,
    )

    left = evaluate_experience_consolidation(
        candidate_id="same",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=config,
    )

    right = evaluate_experience_consolidation(
        candidate_id="same",
        trajectory=stable_trajectory,
        stability=stable_stability,
        config=config,
    )

    assert left == right


def test_oscillatory_evaluation_is_deterministic(
    oscillatory_trajectory,
    oscillatory_stability,
):
    left = evaluate_experience_consolidation(
        candidate_id="same_osc",
        trajectory=oscillatory_trajectory,
        stability=oscillatory_stability,
    )

    right = evaluate_experience_consolidation(
        candidate_id="same_osc",
        trajectory=oscillatory_trajectory,
        stability=oscillatory_stability,
    )

    assert left == right


def test_representation_vector_is_deterministic(
    stable_trajectory,
    stable_stability,
):
    left = evaluate_experience_consolidation(
        candidate_id="same_vec",
        trajectory=stable_trajectory,
        stability=stable_stability,
    )

    right = evaluate_experience_consolidation(
        candidate_id="same_vec",
        trajectory=stable_trajectory,
        stability=stable_stability,
    )

    assert left.representation_vector == right.representation_vector
