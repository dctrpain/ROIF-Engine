"""
Tests for ROIF Memory 2.0 — Conditioned Prestress Core.

Layer stack:

    ExperienceTransformation
        -> ExperienceSequence
        -> ExperiencePattern
        -> ExperienceAttractor
        -> AttractorDynamics
        -> AttractorTrajectory
        -> ExperienceConditioning
        -> ConditionedPrestress

The suite validates the bridge from conditioned association to temporary
attractor prestress, including strength/confidence/generalization scaling,
threshold suppression, bias capping, immutability, and policy-free boundaries.
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
    ConditionedPrestressError,
    ConditionedPrestressResult,
    SCHEMA_VERSION,
    build_conditioned_prestress,
    conditioned_prestress_bias,
    conditioned_prestress_is_policy_free,
    conditioned_prestress_target_id,
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
    cue,
    conditioning_memory,
) -> ConditionedPrestressResult:
    return build_conditioned_prestress(
        prestress_id="conditioned_prestress",
        query_cue=cue,
        memory=conditioning_memory,
    )


# =============================================================================
# Schema / config
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "conditioned_prestress_v1"


def test_default_config_constructs() -> None:
    config = ConditionedPrestressConfig()
    assert isinstance(
        config,
        ConditionedPrestressConfig,
    )


def test_config_rejects_negative_max_bias() -> None:
    with pytest.raises(
        ConditionedPrestressError
    ):
        ConditionedPrestressConfig(
            max_bias=-0.1,
        )


def test_config_rejects_negative_strength_gain() -> None:
    with pytest.raises(
        ConditionedPrestressError
    ):
        ConditionedPrestressConfig(
            strength_gain=-0.1,
        )


def test_config_rejects_negative_confidence_gain() -> None:
    with pytest.raises(
        ConditionedPrestressError
    ):
        ConditionedPrestressConfig(
            confidence_gain=-0.1,
        )


def test_config_rejects_negative_generalization_gain() -> None:
    with pytest.raises(
        ConditionedPrestressError
    ):
        ConditionedPrestressConfig(
            generalization_gain=-0.1,
        )


def test_config_rejects_negative_minimum_response_strength() -> None:
    with pytest.raises(
        ConditionedPrestressError
    ):
        ConditionedPrestressConfig(
            minimum_response_strength=-0.1,
        )


# =============================================================================
# Default conditioned prestress
# =============================================================================


def test_build_returns_conditioned_prestress_result(
    default_result,
) -> None:
    assert isinstance(
        default_result,
        ConditionedPrestressResult,
    )


def test_result_contains_attractor_prestress(
    default_result,
) -> None:
    assert isinstance(
        default_result.prestress,
        AttractorPrestress,
    )


def test_identical_cue_targets_sensitization_attractor(
    default_result,
) -> None:
    assert conditioned_prestress_target_id(
        default_result
    ) == "sensitization_attractor"


def test_identical_cue_bias_positive(
    default_result,
) -> None:
    assert conditioned_prestress_bias(
        default_result
    ) > 0.0


def test_raw_bias_positive(
    default_result,
) -> None:
    assert default_result.raw_bias > 0.0


def test_applied_bias_positive(
    default_result,
) -> None:
    assert default_result.applied_bias > 0.0


def test_default_result_not_suppressed(
    default_result,
) -> None:
    assert default_result.suppressed_by_threshold is False


def test_applied_bias_matches_prestress_mapping(
    default_result,
) -> None:
    target = conditioned_prestress_target_id(
        default_result
    )

    assert default_result.prestress.bias_by_attractor_id[
        target
    ] == pytest.approx(
        default_result.applied_bias
    )


def test_default_bias_does_not_exceed_one(
    default_result,
) -> None:
    assert default_result.applied_bias <= 1.0


# =============================================================================
# Generalization scaling
# =============================================================================


def test_similar_cue_has_positive_bias(
    similar_cue,
    conditioning_memory,
) -> None:
    result = build_conditioned_prestress(
        prestress_id="similar",
        query_cue=similar_cue,
        memory=conditioning_memory,
    )

    assert result.applied_bias > 0.0


def test_similar_cue_bias_lower_than_identical(
    cue,
    similar_cue,
    conditioning_memory,
) -> None:
    identical = build_conditioned_prestress(
        prestress_id="identical",
        query_cue=cue,
        memory=conditioning_memory,
    )

    similar = build_conditioned_prestress(
        prestress_id="similar",
        query_cue=similar_cue,
        memory=conditioning_memory,
    )

    assert similar.applied_bias < identical.applied_bias


def test_distant_cue_bias_lower_than_similar(
    similar_cue,
    distant_cue,
    conditioning_memory,
) -> None:
    similar = build_conditioned_prestress(
        prestress_id="similar",
        query_cue=similar_cue,
        memory=conditioning_memory,
    )

    distant = build_conditioned_prestress(
        prestress_id="distant",
        query_cue=distant_cue,
        memory=conditioning_memory,
    )

    assert distant.applied_bias < similar.applied_bias


def test_smaller_generalization_scale_reduces_similar_cue_bias(
    similar_cue,
    conditioning_memory,
) -> None:
    broad = build_conditioned_prestress(
        prestress_id="broad",
        query_cue=similar_cue,
        memory=conditioning_memory,
        generalization_scale=1.0,
    )

    narrow = build_conditioned_prestress(
        prestress_id="narrow",
        query_cue=similar_cue,
        memory=conditioning_memory,
        generalization_scale=0.1,
    )

    assert narrow.applied_bias < broad.applied_bias


# =============================================================================
# Gains / cap
# =============================================================================


def test_strength_gain_increases_raw_bias(
    cue,
    conditioning_memory,
) -> None:
    baseline = build_conditioned_prestress(
        prestress_id="baseline",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=10.0,
            strength_gain=1.0,
        ),
    )

    amplified = build_conditioned_prestress(
        prestress_id="amplified",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=10.0,
            strength_gain=2.0,
        ),
    )

    assert amplified.raw_bias > baseline.raw_bias


def test_confidence_gain_increases_raw_bias(
    cue,
    conditioning_memory,
) -> None:
    baseline = build_conditioned_prestress(
        prestress_id="baseline",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=10.0,
            confidence_gain=1.0,
        ),
    )

    amplified = build_conditioned_prestress(
        prestress_id="amplified",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=10.0,
            confidence_gain=2.0,
        ),
    )

    assert amplified.raw_bias > baseline.raw_bias


def test_generalization_gain_increases_raw_bias(
    cue,
    conditioning_memory,
) -> None:
    baseline = build_conditioned_prestress(
        prestress_id="baseline",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=10.0,
            generalization_gain=1.0,
        ),
    )

    amplified = build_conditioned_prestress(
        prestress_id="amplified",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=10.0,
            generalization_gain=2.0,
        ),
    )

    assert amplified.raw_bias > baseline.raw_bias


def test_max_bias_caps_applied_bias(
    cue,
    conditioning_memory,
) -> None:
    result = build_conditioned_prestress(
        prestress_id="capped",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=0.05,
            strength_gain=10.0,
            confidence_gain=10.0,
            generalization_gain=10.0,
        ),
    )

    assert result.applied_bias == pytest.approx(
        0.05
    )


def test_raw_bias_can_exceed_applied_bias_when_capped(
    cue,
    conditioning_memory,
) -> None:
    result = build_conditioned_prestress(
        prestress_id="capped",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            max_bias=0.05,
            strength_gain=10.0,
            confidence_gain=10.0,
            generalization_gain=10.0,
        ),
    )

    assert result.raw_bias > result.applied_bias


# =============================================================================
# Threshold suppression
# =============================================================================


def test_high_threshold_suppresses_bias(
    cue,
    conditioning_memory,
) -> None:
    result = build_conditioned_prestress(
        prestress_id="threshold",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert result.suppressed_by_threshold is True


def test_suppressed_bias_is_zero(
    cue,
    conditioning_memory,
) -> None:
    result = build_conditioned_prestress(
        prestress_id="threshold",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert result.applied_bias == pytest.approx(
        0.0
    )


def test_suppressed_prestress_still_targets_predicted_attractor(
    cue,
    conditioning_memory,
) -> None:
    result = build_conditioned_prestress(
        prestress_id="threshold",
        query_cue=cue,
        memory=conditioning_memory,
        config=ConditionedPrestressConfig(
            minimum_response_strength=1.0,
        ),
    )

    assert conditioned_prestress_target_id(
        result
    ) == "sensitization_attractor"


# =============================================================================
# Metadata / source identity
# =============================================================================


def test_query_cue_id_preserved(
    default_result,
) -> None:
    assert default_result.query_cue_id == "cue_door"


def test_memory_id_preserved(
    default_result,
) -> None:
    assert default_result.memory_id == "memory_door"


def test_prestress_metadata_source(
    default_result,
) -> None:
    assert default_result.prestress.metadata[
        "source"
    ] == "conditioned_association"


def test_prestress_metadata_preserves_query_cue(
    default_result,
) -> None:
    assert default_result.prestress.metadata[
        "query_cue_id"
    ] == "cue_door"


def test_prestress_metadata_preserves_memory_id(
    default_result,
) -> None:
    assert default_result.prestress.metadata[
        "memory_id"
    ] == "memory_door"


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_conditioned_prestress_is_policy_free(
    default_result,
) -> None:
    assert conditioned_prestress_is_policy_free(
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


def test_prestress_declares_no_memory_mutation(
    default_result,
) -> None:
    assert default_result.prestress.metadata[
        "memory_mutated"
    ] is False


def test_prestress_declares_no_learning(
    default_result,
) -> None:
    assert default_result.prestress.metadata[
        "learning_applied"
    ] is False


def test_prestress_declares_no_policy_modified(
    default_result,
) -> None:
    assert default_result.prestress.metadata[
        "policy_modified"
    ] is False


# =============================================================================
# Source memory immutability
# =============================================================================


def test_build_does_not_change_memory_associations(
    cue,
    conditioning_memory,
) -> None:
    before = conditioning_memory.associations

    build_conditioned_prestress(
        prestress_id="x",
        query_cue=cue,
        memory=conditioning_memory,
    )

    assert conditioning_memory.associations == before


def test_build_does_not_change_memory_observations(
    cue,
    conditioning_memory,
) -> None:
    before = conditioning_memory.observations

    build_conditioned_prestress(
        prestress_id="x",
        query_cue=cue,
        memory=conditioning_memory,
    )

    assert conditioning_memory.observations == before


# =============================================================================
# Immutability
# =============================================================================


def test_config_is_frozen() -> None:
    config = ConditionedPrestressConfig()

    with pytest.raises(
        FrozenInstanceError
    ):
        config.max_bias = 2.0  # type: ignore[misc]


def test_result_is_frozen(
    default_result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        default_result.applied_bias = 0.0  # type: ignore[misc]


def test_config_metadata_is_read_only() -> None:
    config = ConditionedPrestressConfig(
        metadata={
            "x": 1,
        }
    )

    assert isinstance(
        config.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        config.metadata[
            "x"
        ] = 2  # type: ignore[index]


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


# =============================================================================
# Determinism
# =============================================================================


def test_default_build_is_deterministic(
    cue,
    conditioning_memory,
) -> None:
    left = build_conditioned_prestress(
        prestress_id="same",
        query_cue=cue,
        memory=conditioning_memory,
    )

    right = build_conditioned_prestress(
        prestress_id="same",
        query_cue=cue,
        memory=conditioning_memory,
    )

    assert left == right


def test_similar_cue_build_is_deterministic(
    similar_cue,
    conditioning_memory,
) -> None:
    left = build_conditioned_prestress(
        prestress_id="same",
        query_cue=similar_cue,
        memory=conditioning_memory,
    )

    right = build_conditioned_prestress(
        prestress_id="same",
        query_cue=similar_cue,
        memory=conditioning_memory,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(
    default_result,
) -> None:
    assert conditioned_prestress_bias(
        default_result
    ) == conditioned_prestress_bias(
        default_result
    )

    assert conditioned_prestress_target_id(
        default_result
    ) == conditioned_prestress_target_id(
        default_result
    )
