"""
Tests for ROIF Memory 2.0 — Experience Pattern Core.

The suite validates the third memory layer:

    ExperienceTransformation
        -> ExperienceSequence
        -> ExperiencePattern

A pattern is not merely repeated event identity.
It is repeated transition structure with measurable stability / variability.
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
    ExperiencePattern,
    ExperiencePatternError,
    ExperiencePatternMetrics,
    PatternMember,
    SCHEMA_VERSION,
    build_experience_pattern,
    compute_pattern_metrics,
    extract_pattern_members,
    member_adaptation_series,
    member_reserve_delta_series,
    member_sensitization_series,
    pattern_direction,
    pattern_has_stable_body_response,
    pattern_has_stable_scar,
    pattern_is_policy_free,
    pattern_is_recurrent,
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
def other_event() -> ExperienceEvent:
    return ExperienceEvent(
        event_id="event_other",
        event_type="other_cue",
        structural_vector=(0.2, 0.8, 0.1, -0.2),
    )


@pytest.fixture
def sensitizing_sequence_a(
    repeated_event,
):
    states = (
        make_state(
            "a_s0",
            cognitive=(0.10, 0.10, 0.10),
            physiological=(0.10, 0.08, 0.06),
            reserve=0.90,
        ),
        make_state(
            "a_s1",
            cognitive=(0.20, 0.25, 0.18),
            physiological=(0.20, 0.18, 0.16),
            reserve=0.80,
        ),
        make_state(
            "a_s2",
            cognitive=(0.38, 0.45, 0.36),
            physiological=(0.40, 0.36, 0.32),
            reserve=0.65,
        ),
    )

    return build_sequence_from_states(
        "seq_a",
        repeated_event,
        states,
        magnitudes=(0.20, 0.45),
    )


@pytest.fixture
def sensitizing_sequence_b(
    repeated_event,
):
    states = (
        make_state(
            "b_s0",
            cognitive=(0.12, 0.11, 0.10),
            physiological=(0.11, 0.09, 0.07),
            reserve=0.88,
        ),
        make_state(
            "b_s1",
            cognitive=(0.22, 0.26, 0.19),
            physiological=(0.21, 0.19, 0.17),
            reserve=0.77,
        ),
        make_state(
            "b_s2",
            cognitive=(0.39, 0.46, 0.37),
            physiological=(0.41, 0.37, 0.33),
            reserve=0.62,
        ),
    )

    return build_sequence_from_states(
        "seq_b",
        repeated_event,
        states,
        magnitudes=(0.22, 0.47),
    )


@pytest.fixture
def sensitization_pattern(
    sensitizing_sequence_a,
    sensitizing_sequence_b,
) -> ExperiencePattern:
    return build_experience_pattern(
        pattern_id="pattern_repeat",
        event_type="repeated_cue",
        sequences=(
            sensitizing_sequence_a,
            sensitizing_sequence_b,
        ),
    )


# =============================================================================
# Schema / extraction
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "experience_pattern_v1"


def test_extract_pattern_members_returns_tuple(
    sensitizing_sequence_a,
) -> None:
    members = extract_pattern_members(
        (sensitizing_sequence_a,),
        event_type="repeated_cue",
    )
    assert isinstance(
        members,
        tuple,
    )


def test_extract_pattern_members_count_from_one_sequence(
    sensitizing_sequence_a,
) -> None:
    members = extract_pattern_members(
        (sensitizing_sequence_a,),
        event_type="repeated_cue",
    )
    assert len(members) == 2


def test_extract_pattern_members_count_from_two_sequences(
    sensitizing_sequence_a,
    sensitizing_sequence_b,
) -> None:
    members = extract_pattern_members(
        (
            sensitizing_sequence_a,
            sensitizing_sequence_b,
        ),
        event_type="repeated_cue",
    )
    assert len(members) == 4


def test_extract_pattern_members_are_pattern_members(
    sensitizing_sequence_a,
) -> None:
    members = extract_pattern_members(
        (sensitizing_sequence_a,),
        event_type="repeated_cue",
    )
    assert all(
        isinstance(member, PatternMember)
        for member in members
    )


def test_extract_members_preserves_sequence_ids(
    sensitizing_sequence_a,
    sensitizing_sequence_b,
) -> None:
    members = extract_pattern_members(
        (
            sensitizing_sequence_a,
            sensitizing_sequence_b,
        ),
        event_type="repeated_cue",
    )

    assert {
        member.sequence_id
        for member in members
    } == {
        "seq_a",
        "seq_b",
    }


def test_extract_members_preserves_event_type(
    sensitizing_sequence_a,
) -> None:
    members = extract_pattern_members(
        (sensitizing_sequence_a,),
        event_type="repeated_cue",
    )

    assert all(
        member.event_type == "repeated_cue"
        for member in members
    )


def test_extract_unknown_event_returns_empty_tuple(
    sensitizing_sequence_a,
) -> None:
    members = extract_pattern_members(
        (sensitizing_sequence_a,),
        event_type="unknown",
    )

    assert members == ()


# =============================================================================
# Pattern construction
# =============================================================================


def test_build_returns_experience_pattern(
    sensitization_pattern,
) -> None:
    assert isinstance(
        sensitization_pattern,
        ExperiencePattern,
    )


def test_pattern_id_regression(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.pattern_id == "pattern_repeat"


def test_pattern_event_type_regression(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.event_type == "repeated_cue"


def test_pattern_contains_four_members(
    sensitization_pattern,
) -> None:
    assert len(
        sensitization_pattern.members
    ) == 4


def test_pattern_metrics_type(
    sensitization_pattern,
) -> None:
    assert isinstance(
        sensitization_pattern.metrics,
        ExperiencePatternMetrics,
    )


def test_build_rejects_empty_sequences() -> None:
    with pytest.raises(
        ExperiencePatternError
    ):
        build_experience_pattern(
            pattern_id="x",
            event_type="repeated_cue",
            sequences=(),
        )


def test_build_rejects_event_type_not_found(
    sensitizing_sequence_a,
) -> None:
    with pytest.raises(
        ExperiencePatternError
    ):
        build_experience_pattern(
            pattern_id="x",
            event_type="missing",
            sequences=(
                sensitizing_sequence_a,
            ),
        )


def test_build_rejects_empty_pattern_id(
    sensitizing_sequence_a,
) -> None:
    with pytest.raises(
        ExperiencePatternError
    ):
        build_experience_pattern(
            pattern_id="",
            event_type="repeated_cue",
            sequences=(
                sensitizing_sequence_a,
            ),
        )


# =============================================================================
# Metrics
# =============================================================================


def test_metrics_member_count_regression(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.member_count == 4


def test_metrics_sequence_count_regression(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.sequence_count == 2


def test_response_variability_is_positive(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.response_variability > 0.0


def test_body_variability_is_positive(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.body_variability > 0.0


def test_scar_variability_is_positive(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.scar_variability > 0.0


def test_response_stability_is_bounded(
    sensitization_pattern,
) -> None:
    value = sensitization_pattern.metrics.response_stability
    assert 0.0 <= value <= 1.0


def test_body_stability_is_bounded(
    sensitization_pattern,
) -> None:
    value = sensitization_pattern.metrics.body_stability
    assert 0.0 <= value <= 1.0


def test_scar_stability_is_bounded(
    sensitization_pattern,
) -> None:
    value = sensitization_pattern.metrics.scar_stability
    assert 0.0 <= value <= 1.0


def test_pattern_confidence_is_bounded(
    sensitization_pattern,
) -> None:
    value = sensitization_pattern.metrics.pattern_confidence
    assert 0.0 <= value <= 1.0


def test_pattern_confidence_positive(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.pattern_confidence > 0.0


def test_mean_reserve_delta_is_negative(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.mean_reserve_delta < 0.0


def test_reserve_decline_fraction_is_one(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.reserve_decline_fraction == pytest.approx(
        1.0
    )


def test_reserve_gain_fraction_is_zero(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.reserve_gain_fraction == pytest.approx(
        0.0
    )


def test_mean_sensitization_positive(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.mean_sensitization > 0.0


def test_mean_adaptation_zero(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.mean_adaptation == pytest.approx(
        0.0
    )


# =============================================================================
# Pattern direction
# =============================================================================


def test_sensitization_is_dominant(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.sensitization_dominant is True


def test_adaptation_is_not_dominant(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.adaptation_dominant is False


def test_pattern_not_mixed(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.mixed_direction is False


def test_pattern_direction_returns_sensitization(
    sensitization_pattern,
) -> None:
    assert pattern_direction(
        sensitization_pattern
    ) == "sensitization"


# =============================================================================
# Recurrence / stability helpers
# =============================================================================


def test_pattern_is_recurrent_default(
    sensitization_pattern,
) -> None:
    assert pattern_is_recurrent(
        sensitization_pattern
    ) is True


def test_pattern_is_recurrent_with_four_member_threshold(
    sensitization_pattern,
) -> None:
    assert pattern_is_recurrent(
        sensitization_pattern,
        minimum_members=4,
    ) is True


def test_pattern_not_recurrent_with_five_member_threshold(
    sensitization_pattern,
) -> None:
    assert pattern_is_recurrent(
        sensitization_pattern,
        minimum_members=5,
    ) is False


def test_pattern_is_recurrent_rejects_zero_threshold(
    sensitization_pattern,
) -> None:
    with pytest.raises(
        ExperiencePatternError
    ):
        pattern_is_recurrent(
            sensitization_pattern,
            minimum_members=0,
        )


def test_body_stability_helper_matches_metric(
    sensitization_pattern,
) -> None:
    threshold = sensitization_pattern.metrics.body_stability

    assert pattern_has_stable_body_response(
        sensitization_pattern,
        minimum_stability=threshold,
    ) is True


def test_scar_stability_helper_matches_metric(
    sensitization_pattern,
) -> None:
    threshold = sensitization_pattern.metrics.scar_stability

    assert pattern_has_stable_scar(
        sensitization_pattern,
        minimum_stability=threshold,
    ) is True


def test_body_stability_helper_rejects_invalid_threshold(
    sensitization_pattern,
) -> None:
    with pytest.raises(
        ExperiencePatternError
    ):
        pattern_has_stable_body_response(
            sensitization_pattern,
            minimum_stability=1.1,
        )


def test_scar_stability_helper_rejects_invalid_threshold(
    sensitization_pattern,
) -> None:
    with pytest.raises(
        ExperiencePatternError
    ):
        pattern_has_stable_scar(
            sensitization_pattern,
            minimum_stability=-0.1,
        )


# =============================================================================
# Member series
# =============================================================================


def test_member_reserve_delta_series_has_four_values(
    sensitization_pattern,
) -> None:
    assert len(
        member_reserve_delta_series(
            sensitization_pattern
        )
    ) == 4


def test_member_reserve_delta_series_all_negative(
    sensitization_pattern,
) -> None:
    assert all(
        value < 0.0
        for value
        in member_reserve_delta_series(
            sensitization_pattern
        )
    )


def test_member_sensitization_series_has_four_values(
    sensitization_pattern,
) -> None:
    assert len(
        member_sensitization_series(
            sensitization_pattern
        )
    ) == 4


def test_member_sensitization_series_all_positive(
    sensitization_pattern,
) -> None:
    assert all(
        value > 0.0
        for value
        in member_sensitization_series(
            sensitization_pattern
        )
    )


def test_member_adaptation_series_is_zero(
    sensitization_pattern,
) -> None:
    assert member_adaptation_series(
        sensitization_pattern
    ) == pytest.approx(
        (
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )


# =============================================================================
# Direct metric computation
# =============================================================================


def test_compute_pattern_metrics_matches_built_pattern(
    sensitization_pattern,
) -> None:
    recomputed = compute_pattern_metrics(
        sensitization_pattern.members
    )

    assert recomputed == sensitization_pattern.metrics


def test_compute_pattern_metrics_rejects_empty_members() -> None:
    with pytest.raises(
        ExperiencePatternError
    ):
        compute_pattern_metrics(
            ()
        )


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_pattern_is_policy_free_returns_true(
    sensitization_pattern,
) -> None:
    assert pattern_is_policy_free(
        sensitization_pattern
    ) is True


def test_pattern_declares_no_action_selected(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metadata[
        "action_selected"
    ] is False


def test_pattern_declares_no_policy_modified(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metadata[
        "policy_modified"
    ] is False


def test_pattern_declares_no_diagnosis_generated(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metadata[
        "diagnosis_generated"
    ] is False


def test_pattern_declares_no_causal_truth(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metadata[
        "causal_truth_inferred"
    ] is False


def test_metrics_declares_descriptive_mode(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.metadata[
        "pattern_metric_is_descriptive"
    ] is True


def test_metrics_declares_no_causal_truth(
    sensitization_pattern,
) -> None:
    assert sensitization_pattern.metrics.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_member_declares_no_causal_truth(
    sensitization_pattern,
) -> None:
    assert all(
        member.metadata[
            "causal_truth_inferred"
        ] is False
        for member
        in sensitization_pattern.members
    )


# =============================================================================
# Immutability
# =============================================================================


def test_pattern_members_are_tuple(
    sensitization_pattern,
) -> None:
    assert isinstance(
        sensitization_pattern.members,
        tuple,
    )


def test_pattern_is_frozen(
    sensitization_pattern,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        sensitization_pattern.pattern_id = "changed"  # type: ignore[misc]


def test_metrics_are_frozen(
    sensitization_pattern,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        sensitization_pattern.metrics.member_count = 99  # type: ignore[misc]


def test_member_is_frozen(
    sensitization_pattern,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        sensitization_pattern.members[
            0
        ].sequence_id = "changed"  # type: ignore[misc]


def test_pattern_metadata_is_read_only(
    sensitization_pattern,
) -> None:
    assert isinstance(
        sensitization_pattern.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        sensitization_pattern.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_metrics_metadata_is_read_only(
    sensitization_pattern,
) -> None:
    assert isinstance(
        sensitization_pattern.metrics.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        sensitization_pattern.metrics.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_member_metadata_is_read_only(
    sensitization_pattern,
) -> None:
    assert isinstance(
        sensitization_pattern.members[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        sensitization_pattern.members[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_member_extraction_is_deterministic(
    sensitizing_sequence_a,
    sensitizing_sequence_b,
) -> None:
    left = extract_pattern_members(
        (
            sensitizing_sequence_a,
            sensitizing_sequence_b,
        ),
        event_type="repeated_cue",
    )

    right = extract_pattern_members(
        (
            sensitizing_sequence_a,
            sensitizing_sequence_b,
        ),
        event_type="repeated_cue",
    )

    assert left == right


def test_metric_computation_is_deterministic(
    sensitization_pattern,
) -> None:
    left = compute_pattern_metrics(
        sensitization_pattern.members
    )

    right = compute_pattern_metrics(
        sensitization_pattern.members
    )

    assert left == right


def test_complete_pattern_build_is_deterministic(
    sensitizing_sequence_a,
    sensitizing_sequence_b,
) -> None:
    left = build_experience_pattern(
        pattern_id="same",
        event_type="repeated_cue",
        sequences=(
            sensitizing_sequence_a,
            sensitizing_sequence_b,
        ),
    )

    right = build_experience_pattern(
        pattern_id="same",
        event_type="repeated_cue",
        sequences=(
            sensitizing_sequence_a,
            sensitizing_sequence_b,
        ),
    )

    assert left == right


def test_direction_helper_is_deterministic(
    sensitization_pattern,
) -> None:
    assert pattern_direction(
        sensitization_pattern
    ) == pattern_direction(
        sensitization_pattern
    )


# =============================================================================
# Adaptation control pattern
# =============================================================================


def test_adaptation_pattern_direction(
    repeated_event,
) -> None:
    states = (
        make_state(
            "c_s0",
            cognitive=(0.50, 0.50, 0.50),
            physiological=(0.50, 0.50, 0.50),
            reserve=0.40,
        ),
        make_state(
            "c_s1",
            cognitive=(0.30, 0.30, 0.30),
            physiological=(0.28, 0.28, 0.28),
            reserve=0.65,
        ),
        make_state(
            "c_s2",
            cognitive=(0.18, 0.18, 0.18),
            physiological=(0.16, 0.16, 0.16),
            reserve=0.85,
        ),
    )

    sequence = build_sequence_from_states(
        "seq_adapt",
        repeated_event,
        states,
        magnitudes=(0.40, 0.30),
    )

    pattern = build_experience_pattern(
        pattern_id="adapt",
        event_type="repeated_cue",
        sequences=(sequence,),
    )

    assert pattern.metrics.adaptation_dominant is True
    assert pattern.metrics.sensitization_dominant is False
    assert pattern_direction(pattern) == "adaptation"


# =============================================================================
# Mixed control pattern
# =============================================================================


def test_mixed_pattern_direction(
    repeated_event,
) -> None:
    s0 = make_state(
        "m_s0",
        cognitive=(0.10, 0.10, 0.10),
        physiological=(0.10, 0.10, 0.10),
        reserve=0.60,
    )
    s1 = make_state(
        "m_s1",
        cognitive=(0.20, 0.20, 0.20),
        physiological=(0.20, 0.20, 0.20),
        reserve=0.50,
    )
    s2 = make_state(
        "m_s2",
        cognitive=(0.15, 0.15, 0.15),
        physiological=(0.15, 0.15, 0.15),
        reserve=0.65,
    )

    sequence = build_sequence_from_states(
        "seq_mix",
        repeated_event,
        (s0, s1, s2),
        magnitudes=(0.30, 0.25),
    )

    pattern = build_experience_pattern(
        pattern_id="mixed",
        event_type="repeated_cue",
        sequences=(sequence,),
    )

    assert pattern.metrics.mixed_direction is True
    assert pattern_direction(pattern) == "mixed"

