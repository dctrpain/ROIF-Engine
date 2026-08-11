"""
Tests for ROIF Memory 2.0 — Experience Attractor Core.

Layer stack:

    ExperienceTransformation
        -> ExperienceSequence
        -> ExperiencePattern
        -> ExperienceAttractor

The suite validates descriptive attractor behavior without making a biological
attractor claim.
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
    AttractorMember,
    ExperienceAttractor,
    ExperienceAttractorError,
    ExperienceAttractorMetrics,
    SCHEMA_VERSION,
    attractor_direction,
    attractor_distance_series,
    attractor_event_types,
    attractor_has_return_tendency,
    attractor_is_compact,
    attractor_is_policy_free,
    attractor_is_recurrent,
    attractor_member_ids,
    build_experience_attractor,
    pattern_feature_vector,
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def repeated_event() -> ExperienceEvent:
    return ExperienceEvent(
        event_id="event_repeat",
        event_type="repeated_cue",
        structural_vector=(1.0, 0.25, -0.1, 0.4),
    )


@pytest.fixture
def pattern_a(
    repeated_event,
):
    return build_sensitizing_pattern(
        "pattern_a",
        "seq_a",
        repeated_event,
        0.00,
    )


@pytest.fixture
def pattern_b(
    repeated_event,
):
    return build_sensitizing_pattern(
        "pattern_b",
        "seq_b",
        repeated_event,
        0.01,
    )


@pytest.fixture
def pattern_c(
    repeated_event,
):
    return build_sensitizing_pattern(
        "pattern_c",
        "seq_c",
        repeated_event,
        0.02,
    )


@pytest.fixture
def attractor(
    pattern_a,
    pattern_b,
    pattern_c,
) -> ExperienceAttractor:
    return build_experience_attractor(
        attractor_id="attractor_repeat",
        patterns=(
            pattern_a,
            pattern_b,
            pattern_c,
        ),
    )


# =============================================================================
# Schema / construction
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "experience_attractor_v1"


def test_build_returns_experience_attractor(
    attractor,
) -> None:
    assert isinstance(
        attractor,
        ExperienceAttractor,
    )


def test_metrics_type(
    attractor,
) -> None:
    assert isinstance(
        attractor.metrics,
        ExperienceAttractorMetrics,
    )


def test_members_are_attractor_members(
    attractor,
) -> None:
    assert all(
        isinstance(
            member,
            AttractorMember,
        )
        for member
        in attractor.members
    )


def test_build_rejects_empty_patterns() -> None:
    with pytest.raises(
        ExperienceAttractorError
    ):
        build_experience_attractor(
            attractor_id="x",
            patterns=(),
        )


def test_build_rejects_empty_attractor_id(
    pattern_a,
) -> None:
    with pytest.raises(
        ExperienceAttractorError
    ):
        build_experience_attractor(
            attractor_id="",
            patterns=(pattern_a,),
        )


# =============================================================================
# Pattern feature vector
# =============================================================================


def test_pattern_feature_vector_is_tuple(
    pattern_a,
) -> None:
    vector = pattern_feature_vector(
        pattern_a
    )
    assert isinstance(
        vector,
        tuple,
    )


def test_pattern_feature_vector_length_regression(
    pattern_a,
) -> None:
    assert len(
        pattern_feature_vector(
            pattern_a
        )
    ) == 9


def test_pattern_feature_vector_is_deterministic(
    pattern_a,
) -> None:
    assert pattern_feature_vector(
        pattern_a
    ) == pattern_feature_vector(
        pattern_a
    )


# =============================================================================
# Center / distances
# =============================================================================


def test_attractor_has_three_members(
    attractor,
) -> None:
    assert attractor.metrics.member_count == 3


def test_center_has_nine_dimensions(
    attractor,
) -> None:
    assert len(
        attractor.center
    ) == 9


def test_member_ids_regression(
    attractor,
) -> None:
    assert attractor_member_ids(
        attractor
    ) == (
        "pattern_a",
        "pattern_b",
        "pattern_c",
    )


def test_event_types_regression(
    attractor,
) -> None:
    assert attractor_event_types(
        attractor
    ) == (
        "repeated_cue",
        "repeated_cue",
        "repeated_cue",
    )


def test_event_type_count_is_one(
    attractor,
) -> None:
    assert attractor.metrics.event_type_count == 1


def test_distance_series_has_three_values(
    attractor,
) -> None:
    assert len(
        attractor_distance_series(
            attractor
        )
    ) == 3


def test_distances_are_non_negative(
    attractor,
) -> None:
    assert all(
        value >= 0.0
        for value
        in attractor_distance_series(
            attractor
        )
    )


def test_mean_distance_matches_series(
    attractor,
) -> None:
    values = attractor_distance_series(
        attractor
    )
    assert attractor.metrics.mean_distance_to_center == pytest.approx(
        sum(values) / len(values)
    )


def test_max_distance_matches_series(
    attractor,
) -> None:
    values = attractor_distance_series(
        attractor
    )
    assert attractor.metrics.max_distance_to_center == pytest.approx(
        max(values)
    )


# =============================================================================
# Compactness / recurrence / return
# =============================================================================


def test_compactness_is_bounded(
    attractor,
) -> None:
    assert 0.0 <= attractor.metrics.compactness <= 1.0


def test_recurrence_score_regression(
    attractor,
) -> None:
    assert attractor.metrics.recurrence_score == pytest.approx(
        0.6
    )


def test_return_tendency_is_bounded(
    attractor,
) -> None:
    assert 0.0 <= attractor.metrics.return_tendency <= 1.0


def test_attractor_confidence_is_bounded(
    attractor,
) -> None:
    assert 0.0 <= attractor.metrics.attractor_confidence <= 1.0


def test_attractor_confidence_matches_component_mean(
    attractor,
) -> None:
    expected = (
        attractor.metrics.compactness
        + attractor.metrics.recurrence_score
        + attractor.metrics.return_tendency
    ) / 3.0

    assert attractor.metrics.attractor_confidence == pytest.approx(
        expected
    )


def test_attractor_is_recurrent_default(
    attractor,
) -> None:
    assert attractor_is_recurrent(
        attractor
    ) is True


def test_attractor_recurrent_with_three_threshold(
    attractor,
) -> None:
    assert attractor_is_recurrent(
        attractor,
        minimum_members=3,
    ) is True


def test_attractor_not_recurrent_with_four_threshold(
    attractor,
) -> None:
    assert attractor_is_recurrent(
        attractor,
        minimum_members=4,
    ) is False


def test_attractor_is_recurrent_rejects_zero_threshold(
    attractor,
) -> None:
    with pytest.raises(
        ExperienceAttractorError
    ):
        attractor_is_recurrent(
            attractor,
            minimum_members=0,
        )


def test_compactness_helper_matches_metric(
    attractor,
) -> None:
    threshold = attractor.metrics.compactness
    assert attractor_is_compact(
        attractor,
        minimum_compactness=threshold,
    ) is True


def test_compactness_helper_rejects_invalid_threshold(
    attractor,
) -> None:
    with pytest.raises(
        ExperienceAttractorError
    ):
        attractor_is_compact(
            attractor,
            minimum_compactness=1.1,
        )


def test_return_tendency_helper_matches_metric(
    attractor,
) -> None:
    threshold = attractor.metrics.return_tendency
    assert attractor_has_return_tendency(
        attractor,
        minimum_return_tendency=threshold,
    ) is True


def test_return_tendency_helper_rejects_invalid_threshold(
    attractor,
) -> None:
    with pytest.raises(
        ExperienceAttractorError
    ):
        attractor_has_return_tendency(
            attractor,
            minimum_return_tendency=-0.1,
        )


# =============================================================================
# Direction
# =============================================================================


def test_all_members_are_sensitization_direction(
    attractor,
) -> None:
    assert all(
        member.direction == "sensitization"
        for member
        in attractor.members
    )


def test_sensitization_fraction_is_one(
    attractor,
) -> None:
    assert attractor.metrics.sensitization_fraction == pytest.approx(
        1.0
    )


def test_adaptation_fraction_is_zero(
    attractor,
) -> None:
    assert attractor.metrics.adaptation_fraction == pytest.approx(
        0.0
    )


def test_mixed_fraction_is_zero(
    attractor,
) -> None:
    assert attractor.metrics.mixed_fraction == pytest.approx(
        0.0
    )


def test_dominant_direction_is_sensitization(
    attractor,
) -> None:
    assert attractor.metrics.dominant_direction == "sensitization"


def test_direction_helper_returns_sensitization(
    attractor,
) -> None:
    assert attractor_direction(
        attractor
    ) == "sensitization"


# =============================================================================
# Adaptation control
# =============================================================================


def test_adaptation_attractor_direction(
    repeated_event,
) -> None:
    p1 = build_adaptation_pattern(
        "adapt_1",
        "adapt_seq_1",
        repeated_event,
    )
    p2 = build_adaptation_pattern(
        "adapt_2",
        "adapt_seq_2",
        repeated_event,
    )

    result = build_experience_attractor(
        attractor_id="adaptation_attractor",
        patterns=(p1, p2),
    )

    assert result.metrics.adaptation_fraction == pytest.approx(
        1.0
    )
    assert result.metrics.dominant_direction == "adaptation"
    assert attractor_direction(result) == "adaptation"


# =============================================================================
# Mixed control
# =============================================================================


def test_mixed_attractor_direction(
    repeated_event,
    pattern_a,
) -> None:
    adaptation_pattern = build_adaptation_pattern(
        "adapt_mixed",
        "adapt_seq_mixed",
        repeated_event,
    )

    result = build_experience_attractor(
        attractor_id="mixed_attractor",
        patterns=(
            pattern_a,
            adaptation_pattern,
        ),
    )

    assert result.metrics.sensitization_fraction == pytest.approx(
        0.5
    )
    assert result.metrics.adaptation_fraction == pytest.approx(
        0.5
    )
    assert result.metrics.dominant_direction == "mixed"
    assert attractor_direction(result) == "mixed"


# =============================================================================
# Single member control
# =============================================================================


def test_single_member_return_tendency_is_one(
    pattern_a,
) -> None:
    result = build_experience_attractor(
        attractor_id="single",
        patterns=(pattern_a,),
    )

    assert result.metrics.return_tendency == pytest.approx(
        1.0
    )


def test_single_member_recurrence_score_is_point_two(
    pattern_a,
) -> None:
    result = build_experience_attractor(
        attractor_id="single",
        patterns=(pattern_a,),
    )

    assert result.metrics.recurrence_score == pytest.approx(
        0.2
    )


def test_single_member_distance_is_zero(
    pattern_a,
) -> None:
    result = build_experience_attractor(
        attractor_id="single",
        patterns=(pattern_a,),
    )

    assert attractor_distance_series(
        result
    ) == pytest.approx(
        (0.0,)
    )


def test_single_member_compactness_is_one(
    pattern_a,
) -> None:
    result = build_experience_attractor(
        attractor_id="single",
        patterns=(pattern_a,),
    )

    assert result.metrics.compactness == pytest.approx(
        1.0
    )


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_attractor_is_policy_free_returns_true(
    attractor,
) -> None:
    assert attractor_is_policy_free(
        attractor
    ) is True


def test_attractor_declares_no_action_selected(
    attractor,
) -> None:
    assert attractor.metadata[
        "action_selected"
    ] is False


def test_attractor_declares_no_policy_modified(
    attractor,
) -> None:
    assert attractor.metadata[
        "policy_modified"
    ] is False


def test_attractor_declares_no_diagnosis_generated(
    attractor,
) -> None:
    assert attractor.metadata[
        "diagnosis_generated"
    ] is False


def test_attractor_declares_no_biological_claim(
    attractor,
) -> None:
    assert attractor.metadata[
        "biological_attractor_claimed"
    ] is False


def test_metrics_declares_no_biological_claim(
    attractor,
) -> None:
    assert attractor.metrics.metadata[
        "biological_attractor_claimed"
    ] is False


def test_attractor_declares_no_causal_truth(
    attractor,
) -> None:
    assert attractor.metadata[
        "causal_truth_inferred"
    ] is False


def test_metrics_declares_no_causal_truth(
    attractor,
) -> None:
    assert attractor.metrics.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_member_declares_no_causal_truth(
    attractor,
) -> None:
    assert all(
        member.metadata[
            "causal_truth_inferred"
        ] is False
        for member
        in attractor.members
    )


# =============================================================================
# Immutability
# =============================================================================


def test_members_are_tuple(
    attractor,
) -> None:
    assert isinstance(
        attractor.members,
        tuple,
    )


def test_center_is_tuple(
    attractor,
) -> None:
    assert isinstance(
        attractor.center,
        tuple,
    )


def test_attractor_is_frozen(
    attractor,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        attractor.attractor_id = "changed"  # type: ignore[misc]


def test_metrics_are_frozen(
    attractor,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        attractor.metrics.member_count = 99  # type: ignore[misc]


def test_member_is_frozen(
    attractor,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        attractor.members[
            0
        ].pattern_id = "changed"  # type: ignore[misc]


def test_attractor_metadata_is_read_only(
    attractor,
) -> None:
    assert isinstance(
        attractor.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        attractor.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_metrics_metadata_is_read_only(
    attractor,
) -> None:
    assert isinstance(
        attractor.metrics.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        attractor.metrics.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_member_metadata_is_read_only(
    attractor,
) -> None:
    assert isinstance(
        attractor.members[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        attractor.members[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_complete_attractor_build_is_deterministic(
    pattern_a,
    pattern_b,
    pattern_c,
) -> None:
    left = build_experience_attractor(
        attractor_id="same",
        patterns=(
            pattern_a,
            pattern_b,
            pattern_c,
        ),
    )

    right = build_experience_attractor(
        attractor_id="same",
        patterns=(
            pattern_a,
            pattern_b,
            pattern_c,
        ),
    )

    assert left == right


def test_center_is_deterministic(
    attractor,
) -> None:
    assert attractor.center == attractor.center


def test_distance_series_is_deterministic(
    attractor,
) -> None:
    assert attractor_distance_series(
        attractor
    ) == attractor_distance_series(
        attractor
    )


def test_direction_helper_is_deterministic(
    attractor,
) -> None:
    assert attractor_direction(
        attractor
    ) == attractor_direction(
        attractor
    )
