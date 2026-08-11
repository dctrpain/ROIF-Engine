"""
Tests for ROIF Memory 2.0 — Experience Sequence Core.

Core chain invariant:

    S0 --E--> S1 --E--> S2 --E--> S3

with:
    state_after[n] == state_before[n+1]

The suite validates:
- continuity
- repeated-event dynamics
- reserve trajectory
- cumulative scar
- sensitization / adaptation
- policy-free boundaries
- immutability
- determinism
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
    ExperienceSequence,
    ExperienceSequenceError,
    ExperienceSequenceSummary,
    SCHEMA_VERSION,
    SequenceStepMetrics,
    adaptation_series,
    build_experience_sequence,
    cognitive_scar_series,
    continuity_break_indices,
    physiological_scar_series,
    repeated_event_flags,
    reserve_trajectory,
    response_change_flags,
    sensitization_series,
    sequence_is_continuous,
    sequence_is_policy_free,
    summarize_experience_sequence,
    total_scar_series,
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
def s0() -> SystemState:
    return SystemState(
        state_id="S0",
        cognitive=(0.10, 0.10, 0.10),
        physiological=(0.10, 0.08, 0.06),
        contextual=(0.50, 0.20),
        reserve=0.90,
    )


@pytest.fixture
def s1() -> SystemState:
    return SystemState(
        state_id="S1",
        cognitive=(0.20, 0.25, 0.18),
        physiological=(0.20, 0.18, 0.16),
        contextual=(0.50, 0.20),
        reserve=0.80,
    )


@pytest.fixture
def s2() -> SystemState:
    return SystemState(
        state_id="S2",
        cognitive=(0.38, 0.45, 0.36),
        physiological=(0.40, 0.36, 0.32),
        contextual=(0.50, 0.20),
        reserve=0.65,
    )


@pytest.fixture
def s3() -> SystemState:
    return SystemState(
        state_id="S3",
        cognitive=(0.55, 0.62, 0.50),
        physiological=(0.60, 0.56, 0.52),
        contextual=(0.50, 0.20),
        reserve=0.45,
    )


def _response(
    response_id: str,
    magnitude: float,
) -> SystemResponse:
    return SystemResponse(
        response_id=response_id,
        cognitive_response=(magnitude, magnitude * 0.9),
        physiological_response=(magnitude * 0.8, magnitude * 0.7),
        behavioral_response=(magnitude * 0.6, magnitude * 0.5),
    )


def _body(
    magnitude: float,
) -> BodyResponse:
    return BodyResponse(
        autonomic=(magnitude, magnitude * 0.9),
        endocrine=(magnitude * 0.8, magnitude * 0.7),
        immune=(magnitude * 0.4, magnitude * 0.3),
        motor=(magnitude * 0.75, magnitude * 0.65),
        interoceptive=(magnitude * 0.95, magnitude * 0.85),
    )


@pytest.fixture
def tx0(
    s0,
    s1,
    repeated_event,
):
    return build_experience_transformation(
        transformation_id="tx0",
        state_before=s0,
        event=repeated_event,
        response=_response("r0", 0.20),
        body_response=_body(0.20),
        state_after=s1,
    )


@pytest.fixture
def tx1(
    s1,
    s2,
    repeated_event,
):
    return build_experience_transformation(
        transformation_id="tx1",
        state_before=s1,
        event=repeated_event,
        response=_response("r1", 0.45),
        body_response=_body(0.45),
        state_after=s2,
    )


@pytest.fixture
def tx2(
    s2,
    s3,
    repeated_event,
):
    return build_experience_transformation(
        transformation_id="tx2",
        state_before=s2,
        event=repeated_event,
        response=_response("r2", 0.75),
        body_response=_body(0.75),
        state_after=s3,
    )


@pytest.fixture
def continuous_transformations(
    tx0,
    tx1,
    tx2,
):
    return (
        tx0,
        tx1,
        tx2,
    )


@pytest.fixture
def sequence(
    continuous_transformations,
) -> ExperienceSequence:
    return build_experience_sequence(
        sequence_id="seq_repeat",
        transformations=continuous_transformations,
    )


# =============================================================================
# Schema / construction
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "experience_sequence_v1"


def test_build_returns_experience_sequence(
    sequence,
) -> None:
    assert isinstance(
        sequence,
        ExperienceSequence,
    )


def test_sequence_has_three_transformations(
    sequence,
) -> None:
    assert len(
        sequence.transformations
    ) == 3


def test_sequence_has_three_step_metrics(
    sequence,
) -> None:
    assert len(
        sequence.step_metrics
    ) == 3


def test_sequence_summary_type(
    sequence,
) -> None:
    assert isinstance(
        sequence.summary,
        ExperienceSequenceSummary,
    )


def test_step_metric_type(
    sequence,
) -> None:
    assert all(
        isinstance(
            step,
            SequenceStepMetrics,
        )
        for step in sequence.step_metrics
    )


def test_build_rejects_empty_transformations() -> None:
    with pytest.raises(
        ExperienceSequenceError
    ):
        build_experience_sequence(
            sequence_id="empty",
            transformations=(),
        )


def test_build_rejects_empty_sequence_id(
    continuous_transformations,
) -> None:
    with pytest.raises(
        ExperienceSequenceError
    ):
        build_experience_sequence(
            sequence_id="",
            transformations=continuous_transformations,
        )


# =============================================================================
# Continuity
# =============================================================================


def test_continuous_sequence_has_no_breaks(
    continuous_transformations,
) -> None:
    assert continuity_break_indices(
        continuous_transformations
    ) == ()


def test_sequence_is_continuous_returns_true(
    continuous_transformations,
) -> None:
    assert sequence_is_continuous(
        continuous_transformations
    ) is True


def test_state_after_of_first_equals_before_of_second(
    tx0,
    tx1,
) -> None:
    assert tx0.state_after == tx1.state_before


def test_state_after_of_second_equals_before_of_third(
    tx1,
    tx2,
) -> None:
    assert tx1.state_after == tx2.state_before


def test_summary_declares_continuity_preserved(
    sequence,
) -> None:
    assert sequence.summary.continuity_preserved is True


def test_continuity_break_detected(
    tx0,
    tx2,
) -> None:
    assert continuity_break_indices(
        (
            tx0,
            tx2,
        )
    ) == (
        0,
    )


def test_sequence_is_continuous_returns_false_for_break(
    tx0,
    tx2,
) -> None:
    assert sequence_is_continuous(
        (
            tx0,
            tx2,
        )
    ) is False


def test_build_rejects_broken_sequence_by_default(
    tx0,
    tx2,
) -> None:
    with pytest.raises(
        ExperienceSequenceError
    ):
        build_experience_sequence(
            sequence_id="broken",
            transformations=(
                tx0,
                tx2,
            ),
        )


def test_build_can_preserve_broken_sequence_when_explicitly_allowed(
    tx0,
    tx2,
) -> None:
    result = build_experience_sequence(
        sequence_id="broken_allowed",
        transformations=(
            tx0,
            tx2,
        ),
        require_continuity=False,
    )

    assert result.summary.continuity_preserved is False


# =============================================================================
# Reserve trajectory
# =============================================================================


def test_reserve_trajectory_regression(
    sequence,
) -> None:
    assert reserve_trajectory(
        sequence
    ) == pytest.approx(
        (
            0.90,
            0.80,
            0.65,
            0.45,
        )
    )


def test_initial_reserve_regression(
    sequence,
) -> None:
    assert sequence.summary.initial_reserve == pytest.approx(
        0.90
    )


def test_final_reserve_regression(
    sequence,
) -> None:
    assert sequence.summary.final_reserve == pytest.approx(
        0.45
    )


def test_reserve_change_regression(
    sequence,
) -> None:
    assert sequence.summary.reserve_change == pytest.approx(
        -0.45
    )


def test_reserve_strictly_decreases(
    sequence,
) -> None:
    trajectory = reserve_trajectory(
        sequence
    )

    assert all(
        left > right
        for left, right
        in zip(
            trajectory,
            trajectory[
                1:
            ],
        )
    )


# =============================================================================
# Repeated-event dynamics
# =============================================================================


def test_repeated_event_flags_regression(
    sequence,
) -> None:
    assert repeated_event_flags(
        sequence
    ) == (
        False,
        True,
        True,
    )


def test_response_change_flags_regression(
    sequence,
) -> None:
    assert response_change_flags(
        sequence
    ) == (
        False,
        True,
        True,
    )


def test_summary_counts_two_repeated_event_pairs(
    sequence,
) -> None:
    assert sequence.summary.repeated_event_pair_count == 2


def test_summary_counts_two_changed_repeated_responses(
    sequence,
) -> None:
    assert (
        sequence.summary.repeated_event_response_change_count
        == 2
    )


def test_same_event_is_structurally_preserved(
    sequence,
) -> None:
    vectors = tuple(
        transformation.event.structural_vector
        for transformation
        in sequence.transformations
    )

    assert len(
        set(
            vectors
        )
    ) == 1


def test_same_event_type_is_preserved(
    sequence,
) -> None:
    event_types = tuple(
        transformation.event.event_type
        for transformation
        in sequence.transformations
    )

    assert len(
        set(
            event_types
        )
    ) == 1


def test_repeated_event_meets_different_before_states(
    sequence,
) -> None:
    before_states = tuple(
        transformation.state_before
        for transformation
        in sequence.transformations
    )

    assert len(
        set(
            state.state_id
            for state
            in before_states
        )
    ) == 3


def test_repeated_event_body_response_changes_over_sequence(
    sequence,
) -> None:
    responses = tuple(
        transformation.body_response
        for transformation
        in sequence.transformations
    )

    assert len(
        set(
            response.autonomic
            for response
            in responses
        )
    ) == 3


# =============================================================================
# Scar accumulation
# =============================================================================


def test_total_scar_series_has_three_values(
    sequence,
) -> None:
    assert len(
        total_scar_series(
            sequence
        )
    ) == 3


def test_physiological_scar_series_has_three_values(
    sequence,
) -> None:
    assert len(
        physiological_scar_series(
            sequence
        )
    ) == 3


def test_cognitive_scar_series_has_three_values(
    sequence,
) -> None:
    assert len(
        cognitive_scar_series(
            sequence
        )
    ) == 3


def test_total_scar_values_are_positive(
    sequence,
) -> None:
    assert all(
        value > 0.0
        for value
        in total_scar_series(
            sequence
        )
    )


def test_physiological_scar_values_are_positive(
    sequence,
) -> None:
    assert all(
        value > 0.0
        for value
        in physiological_scar_series(
            sequence
        )
    )


def test_cognitive_scar_values_are_positive(
    sequence,
) -> None:
    assert all(
        value > 0.0
        for value
        in cognitive_scar_series(
            sequence
        )
    )


def test_cumulative_total_scar_matches_series_sum(
    sequence,
) -> None:
    assert sequence.summary.cumulative_total_scar == pytest.approx(
        sum(
            total_scar_series(
                sequence
            )
        )
    )


def test_cumulative_physiological_scar_matches_series_sum(
    sequence,
) -> None:
    assert sequence.summary.cumulative_physiological_scar == pytest.approx(
        sum(
            physiological_scar_series(
                sequence
            )
        )
    )


def test_cumulative_cognitive_scar_matches_series_sum(
    sequence,
) -> None:
    assert sequence.summary.cumulative_cognitive_scar == pytest.approx(
        sum(
            cognitive_scar_series(
                sequence
            )
        )
    )


# =============================================================================
# Sensitization / adaptation
# =============================================================================


def test_sensitization_series_has_three_values(
    sequence,
) -> None:
    assert len(
        sensitization_series(
            sequence
        )
    ) == 3


def test_adaptation_series_has_three_values(
    sequence,
) -> None:
    assert len(
        adaptation_series(
            sequence
        )
    ) == 3


def test_sensitization_values_are_positive(
    sequence,
) -> None:
    assert all(
        value > 0.0
        for value
        in sensitization_series(
            sequence
        )
    )


def test_adaptation_is_zero_when_reserve_declines(
    sequence,
) -> None:
    assert adaptation_series(
        sequence
    ) == pytest.approx(
        (
            0.0,
            0.0,
            0.0,
        )
    )


def test_cumulative_sensitization_matches_series_sum(
    sequence,
) -> None:
    assert sequence.summary.cumulative_sensitization == pytest.approx(
        sum(
            sensitization_series(
                sequence
            )
        )
    )


def test_cumulative_adaptation_is_zero(
    sequence,
) -> None:
    assert sequence.summary.cumulative_adaptation == pytest.approx(
        0.0
    )


# =============================================================================
# Step metrics
# =============================================================================


def test_step_indices_regression(
    sequence,
) -> None:
    assert tuple(
        step.step_index
        for step in sequence.step_metrics
    ) == (
        0,
        1,
        2,
    )


def test_step_transformation_ids_regression(
    sequence,
) -> None:
    assert tuple(
        step.transformation_id
        for step in sequence.step_metrics
    ) == (
        "tx0",
        "tx1",
        "tx2",
    )


def test_step_reserve_before_after_chain(
    sequence,
) -> None:
    actual = tuple(
        (
            step.reserve_before,
            step.reserve_after,
        )
        for step
        in sequence.step_metrics
    )

    expected = (
        (0.90, 0.80),
        (0.80, 0.65),
        (0.65, 0.45),
    )

    for actual_pair, expected_pair in zip(
        actual,
        expected,
    ):
        assert actual_pair == pytest.approx(
            expected_pair
        )


def test_first_step_is_not_repeated_from_previous(
    sequence,
) -> None:
    assert (
        sequence.step_metrics[
            0
        ].repeated_event_from_previous
        is False
    )


def test_later_steps_are_repeated_from_previous(
    sequence,
) -> None:
    assert all(
        step.repeated_event_from_previous
        for step
        in sequence.step_metrics[
            1:
        ]
    )


def test_later_steps_report_response_change(
    sequence,
) -> None:
    assert all(
        step.response_changed_from_previous
        for step
        in sequence.step_metrics[
            1:
        ]
    )


# =============================================================================
# Policy / epistemic boundary
# =============================================================================


def test_sequence_is_policy_free_returns_true(
    sequence,
) -> None:
    assert sequence_is_policy_free(
        sequence
    ) is True


def test_summary_declares_all_transitions_policy_free(
    sequence,
) -> None:
    assert sequence.summary.all_transitions_policy_free is True


def test_sequence_declares_no_action_selected(
    sequence,
) -> None:
    assert sequence.metadata[
        "action_selected"
    ] is False


def test_sequence_declares_no_policy_modified(
    sequence,
) -> None:
    assert sequence.metadata[
        "policy_modified"
    ] is False


def test_sequence_declares_no_diagnosis_generated(
    sequence,
) -> None:
    assert sequence.metadata[
        "diagnosis_generated"
    ] is False


def test_sequence_declares_no_causal_truth_inferred(
    sequence,
) -> None:
    assert sequence.metadata[
        "causal_truth_inferred"
    ] is False


def test_summary_declares_no_causal_truth(
    sequence,
) -> None:
    assert sequence.summary.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_step_declares_no_causal_truth(
    sequence,
) -> None:
    assert all(
        step.metadata[
            "causal_truth_inferred"
        ]
        is False
        for step in sequence.step_metrics
    )


# =============================================================================
# Immutability
# =============================================================================


def test_transformations_are_tuple(
    sequence,
) -> None:
    assert isinstance(
        sequence.transformations,
        tuple,
    )


def test_step_metrics_are_tuple(
    sequence,
) -> None:
    assert isinstance(
        sequence.step_metrics,
        tuple,
    )


def test_sequence_is_frozen(
    sequence,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        sequence.sequence_id = "changed"  # type: ignore[misc]


def test_summary_is_frozen(
    sequence,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        sequence.summary.final_reserve = 1.0  # type: ignore[misc]


def test_step_metric_is_frozen(
    sequence,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        sequence.step_metrics[
            0
        ].step_index = 99  # type: ignore[misc]


def test_sequence_metadata_is_read_only(
    sequence,
) -> None:
    assert isinstance(
        sequence.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        sequence.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_summary_metadata_is_read_only(
    sequence,
) -> None:
    assert isinstance(
        sequence.summary.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        sequence.summary.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_step_metadata_is_read_only(
    sequence,
) -> None:
    assert isinstance(
        sequence.step_metrics[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        sequence.step_metrics[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_continuity_check_is_deterministic(
    continuous_transformations,
) -> None:
    assert continuity_break_indices(
        continuous_transformations
    ) == continuity_break_indices(
        continuous_transformations
    )


def test_summary_is_deterministic(
    continuous_transformations,
) -> None:
    left = summarize_experience_sequence(
        continuous_transformations
    )

    right = summarize_experience_sequence(
        continuous_transformations
    )

    assert left == right


def test_complete_sequence_build_is_deterministic(
    continuous_transformations,
) -> None:
    left = build_experience_sequence(
        sequence_id="same",
        transformations=continuous_transformations,
    )

    right = build_experience_sequence(
        sequence_id="same",
        transformations=continuous_transformations,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(
    sequence,
) -> None:
    assert reserve_trajectory(
        sequence
    ) == reserve_trajectory(
        sequence
    )

    assert total_scar_series(
        sequence
    ) == total_scar_series(
        sequence
    )

    assert sensitization_series(
        sequence
    ) == sensitization_series(
        sequence
    )

    assert response_change_flags(
        sequence
    ) == response_change_flags(
        sequence
    )

