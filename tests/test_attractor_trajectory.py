"""
Tests for ROIF Memory 2.0 — Attractor Trajectory Core.

Layer stack:

    ExperienceTransformation
        -> ExperienceSequence
        -> ExperiencePattern
        -> ExperienceAttractor
        -> AttractorDynamics
        -> AttractorTrajectory

The suite validates temporal continuity, switching, return behavior,
dwell metrics, displacement, uncertainty trajectories, policy-free
boundaries, immutability, and determinism.
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
    AttractorTrajectory,
    AttractorTrajectoryError,
    AttractorTrajectorySummary,
    SCHEMA_VERSION,
    TrajectoryStep,
    build_attractor_trajectory,
    capture_margin_series,
    competition_entropy_series,
    displacement_series,
    dominant_attractor_series,
    return_flags,
    state_id_series,
    summarize_attractor_trajectory,
    switch_flags,
    trajectory_continuity_break_indices,
    trajectory_is_continuous,
    trajectory_is_policy_free,
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
def aba_trajectory(
    attractors,
    initial_state,
    sens_bias,
    adapt_bias,
) -> AttractorTrajectory:
    return build_attractor_trajectory(
        trajectory_id="trajectory_aba",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sens_bias,
            adapt_bias,
            sens_bias,
        ),
        step_size=0.10,
    )


@pytest.fixture
def aaa_trajectory(
    attractors,
    initial_state,
    sens_bias,
) -> AttractorTrajectory:
    return build_attractor_trajectory(
        trajectory_id="trajectory_aaa",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sens_bias,
            sens_bias,
            sens_bias,
        ),
        step_size=0.10,
    )


# =============================================================================
# Schema / construction
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "attractor_trajectory_v1"


def test_build_returns_trajectory(
    aba_trajectory,
) -> None:
    assert isinstance(
        aba_trajectory,
        AttractorTrajectory,
    )


def test_summary_type(
    aba_trajectory,
) -> None:
    assert isinstance(
        aba_trajectory.summary,
        AttractorTrajectorySummary,
    )


def test_steps_are_trajectory_steps(
    aba_trajectory,
) -> None:
    assert all(
        isinstance(
            step,
            TrajectoryStep,
        )
        for step in aba_trajectory.steps
    )


def test_build_rejects_empty_attractors(
    initial_state,
    sens_bias,
) -> None:
    with pytest.raises(
        AttractorTrajectoryError
    ):
        build_attractor_trajectory(
            trajectory_id="x",
            initial_state=initial_state,
            attractors=(),
            prestress_sequence=(sens_bias,),
        )


def test_build_rejects_empty_prestress_sequence(
    attractors,
    initial_state,
) -> None:
    with pytest.raises(
        AttractorTrajectoryError
    ):
        build_attractor_trajectory(
            trajectory_id="x",
            initial_state=initial_state,
            attractors=attractors,
            prestress_sequence=(),
        )


def test_build_rejects_empty_trajectory_id(
    attractors,
    initial_state,
    sens_bias,
) -> None:
    with pytest.raises(
        AttractorTrajectoryError
    ):
        build_attractor_trajectory(
            trajectory_id="",
            initial_state=initial_state,
            attractors=attractors,
            prestress_sequence=(sens_bias,),
        )


# =============================================================================
# Basic trajectory structure
# =============================================================================


def test_aba_trajectory_has_three_steps(
    aba_trajectory,
) -> None:
    assert len(
        aba_trajectory.steps
    ) == 3


def test_summary_step_count_is_three(
    aba_trajectory,
) -> None:
    assert aba_trajectory.summary.step_count == 3


def test_initial_state_preserved(
    aba_trajectory,
    initial_state,
) -> None:
    assert aba_trajectory.initial_state == initial_state


def test_final_state_matches_last_dynamics_after(
    aba_trajectory,
) -> None:
    assert (
        aba_trajectory.final_state
        == aba_trajectory.steps[-1].dynamics.state_after
    )


def test_state_id_series_has_four_states(
    aba_trajectory,
) -> None:
    assert len(
        state_id_series(
            aba_trajectory
        )
    ) == 4


def test_step_indices_regression(
    aba_trajectory,
) -> None:
    assert tuple(
        step.step_index
        for step in aba_trajectory.steps
    ) == (
        0,
        1,
        2,
    )


# =============================================================================
# Continuity
# =============================================================================


def test_trajectory_has_no_continuity_breaks(
    aba_trajectory,
) -> None:
    assert trajectory_continuity_break_indices(
        aba_trajectory.steps
    ) == ()


def test_trajectory_is_continuous_returns_true(
    aba_trajectory,
) -> None:
    assert trajectory_is_continuous(
        aba_trajectory.steps
    ) is True


def test_summary_declares_continuity_preserved(
    aba_trajectory,
) -> None:
    assert aba_trajectory.summary.continuity_preserved is True


def test_first_after_equals_second_before(
    aba_trajectory,
) -> None:
    assert (
        aba_trajectory.steps[0].dynamics.state_after
        == aba_trajectory.steps[1].state_before
    )


def test_second_after_equals_third_before(
    aba_trajectory,
) -> None:
    assert (
        aba_trajectory.steps[1].dynamics.state_after
        == aba_trajectory.steps[2].state_before
    )


# =============================================================================
# Dominant attractor trajectory
# =============================================================================


def test_aba_dominant_series_regression(
    aba_trajectory,
) -> None:
    assert dominant_attractor_series(
        aba_trajectory
    ) == (
        "sensitization_attractor",
        "adaptation_attractor",
        "sensitization_attractor",
    )


def test_aaa_dominant_series_regression(
    aaa_trajectory,
) -> None:
    assert dominant_attractor_series(
        aaa_trajectory
    ) == (
        "sensitization_attractor",
        "sensitization_attractor",
        "sensitization_attractor",
    )


def test_aba_uses_two_unique_dominant_attractors(
    aba_trajectory,
) -> None:
    assert (
        aba_trajectory.summary.unique_dominant_attractor_count
        == 2
    )


def test_aaa_uses_one_unique_dominant_attractor(
    aaa_trajectory,
) -> None:
    assert (
        aaa_trajectory.summary.unique_dominant_attractor_count
        == 1
    )


# =============================================================================
# Switches / returns
# =============================================================================


def test_aba_switch_flags_regression(
    aba_trajectory,
) -> None:
    assert switch_flags(
        aba_trajectory
    ) == (
        False,
        True,
        True,
    )


def test_aba_return_flags_regression(
    aba_trajectory,
) -> None:
    assert return_flags(
        aba_trajectory
    ) == (
        False,
        False,
        True,
    )


def test_aba_switch_count_is_two(
    aba_trajectory,
) -> None:
    assert aba_trajectory.summary.switch_count == 2


def test_aba_return_count_is_one(
    aba_trajectory,
) -> None:
    assert aba_trajectory.summary.return_count == 1


def test_aaa_has_no_switches(
    aaa_trajectory,
) -> None:
    assert aaa_trajectory.summary.switch_count == 0


def test_aaa_has_no_returns(
    aaa_trajectory,
) -> None:
    assert aaa_trajectory.summary.return_count == 0


# =============================================================================
# Dwell
# =============================================================================


def test_aba_longest_dwell_is_one(
    aba_trajectory,
) -> None:
    assert aba_trajectory.summary.longest_dwell_length == 1


def test_aaa_longest_dwell_is_three(
    aaa_trajectory,
) -> None:
    assert aaa_trajectory.summary.longest_dwell_length == 3


# =============================================================================
# Displacement
# =============================================================================


def test_displacement_series_has_three_values(
    aba_trajectory,
) -> None:
    assert len(
        displacement_series(
            aba_trajectory
        )
    ) == 3


def test_all_displacements_are_nonnegative(
    aba_trajectory,
) -> None:
    assert all(
        value >= 0.0
        for value in displacement_series(
            aba_trajectory
        )
    )


def test_cumulative_displacement_matches_series_sum(
    aba_trajectory,
) -> None:
    assert aba_trajectory.summary.cumulative_displacement == pytest.approx(
        sum(
            displacement_series(
                aba_trajectory
            )
        )
    )


def test_mean_displacement_matches_series_mean(
    aba_trajectory,
) -> None:
    values = displacement_series(
        aba_trajectory
    )

    assert aba_trajectory.summary.mean_displacement == pytest.approx(
        sum(values) / len(values)
    )


# =============================================================================
# Competition / capture trajectories
# =============================================================================


def test_entropy_series_has_three_values(
    aba_trajectory,
) -> None:
    assert len(
        competition_entropy_series(
            aba_trajectory
        )
    ) == 3


def test_capture_margin_series_has_three_values(
    aba_trajectory,
) -> None:
    assert len(
        capture_margin_series(
            aba_trajectory
        )
    ) == 3


def test_entropy_values_are_nonnegative(
    aba_trajectory,
) -> None:
    assert all(
        value >= 0.0
        for value in competition_entropy_series(
            aba_trajectory
        )
    )


def test_capture_margin_values_are_nonnegative(
    aba_trajectory,
) -> None:
    assert all(
        value >= 0.0
        for value in capture_margin_series(
            aba_trajectory
        )
    )


def test_mean_entropy_matches_series_mean(
    aba_trajectory,
) -> None:
    values = competition_entropy_series(
        aba_trajectory
    )

    assert aba_trajectory.summary.mean_competition_entropy == pytest.approx(
        sum(values) / len(values)
    )


def test_mean_capture_margin_matches_series_mean(
    aba_trajectory,
) -> None:
    values = capture_margin_series(
        aba_trajectory
    )

    assert aba_trajectory.summary.mean_capture_margin == pytest.approx(
        sum(values) / len(values)
    )


# =============================================================================
# Prestress propagation
# =============================================================================


def test_first_step_uses_sens_bias(
    aba_trajectory,
) -> None:
    assert (
        aba_trajectory.steps[0].prestress.prestress_id
        == "sens_bias"
    )


def test_second_step_uses_adapt_bias(
    aba_trajectory,
) -> None:
    assert (
        aba_trajectory.steps[1].prestress.prestress_id
        == "adapt_bias"
    )


def test_third_step_uses_sens_bias(
    aba_trajectory,
) -> None:
    assert (
        aba_trajectory.steps[2].prestress.prestress_id
        == "sens_bias"
    )


def test_none_prestress_is_supported(
    attractors,
    initial_state,
) -> None:
    trajectory = build_attractor_trajectory(
        trajectory_id="none_prestress",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            None,
            None,
        ),
        step_size=0.10,
    )

    assert trajectory.steps[0].prestress is None
    assert trajectory.steps[1].prestress is None


# =============================================================================
# Summary recomputation
# =============================================================================


def test_recomputed_summary_matches_result(
    aba_trajectory,
) -> None:
    recomputed = summarize_attractor_trajectory(
        aba_trajectory.steps
    )

    assert recomputed == aba_trajectory.summary


def test_summary_rejects_empty_steps() -> None:
    with pytest.raises(
        AttractorTrajectoryError
    ):
        summarize_attractor_trajectory(
            ()
        )


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_trajectory_is_policy_free_returns_true(
    aba_trajectory,
) -> None:
    assert trajectory_is_policy_free(
        aba_trajectory
    ) is True


def test_summary_declares_all_steps_policy_free(
    aba_trajectory,
) -> None:
    assert aba_trajectory.summary.all_steps_policy_free is True


def test_trajectory_declares_no_memory_mutation(
    aba_trajectory,
) -> None:
    assert aba_trajectory.metadata[
        "memory_mutated"
    ] is False


def test_trajectory_declares_no_learning(
    aba_trajectory,
) -> None:
    assert aba_trajectory.metadata[
        "learning_applied"
    ] is False


def test_trajectory_declares_no_action_selected(
    aba_trajectory,
) -> None:
    assert aba_trajectory.metadata[
        "action_selected"
    ] is False


def test_trajectory_declares_no_policy_modified(
    aba_trajectory,
) -> None:
    assert aba_trajectory.metadata[
        "policy_modified"
    ] is False


def test_trajectory_declares_no_diagnosis(
    aba_trajectory,
) -> None:
    assert aba_trajectory.metadata[
        "diagnosis_generated"
    ] is False


def test_trajectory_declares_no_biological_claim(
    aba_trajectory,
) -> None:
    assert aba_trajectory.metadata[
        "biological_attractor_claimed"
    ] is False


def test_trajectory_declares_no_causal_truth(
    aba_trajectory,
) -> None:
    assert aba_trajectory.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_step_declares_no_memory_mutation(
    aba_trajectory,
) -> None:
    assert all(
        step.metadata[
            "memory_mutated"
        ] is False
        for step in aba_trajectory.steps
    )


def test_every_step_declares_no_learning(
    aba_trajectory,
) -> None:
    assert all(
        step.metadata[
            "learning_applied"
        ] is False
        for step in aba_trajectory.steps
    )


def test_every_step_declares_no_policy_modification(
    aba_trajectory,
) -> None:
    assert all(
        step.metadata[
            "policy_modified"
        ] is False
        for step in aba_trajectory.steps
    )


# =============================================================================
# Immutability
# =============================================================================


def test_steps_are_tuple(
    aba_trajectory,
) -> None:
    assert isinstance(
        aba_trajectory.steps,
        tuple,
    )


def test_trajectory_is_frozen(
    aba_trajectory,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        aba_trajectory.trajectory_id = "changed"  # type: ignore[misc]


def test_summary_is_frozen(
    aba_trajectory,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        aba_trajectory.summary.switch_count = 99  # type: ignore[misc]


def test_step_is_frozen(
    aba_trajectory,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        aba_trajectory.steps[
            0
        ].step_index = 99  # type: ignore[misc]


def test_trajectory_metadata_is_read_only(
    aba_trajectory,
) -> None:
    assert isinstance(
        aba_trajectory.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        aba_trajectory.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_summary_metadata_is_read_only(
    aba_trajectory,
) -> None:
    assert isinstance(
        aba_trajectory.summary.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        aba_trajectory.summary.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_step_metadata_is_read_only(
    aba_trajectory,
) -> None:
    assert isinstance(
        aba_trajectory.steps[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        aba_trajectory.steps[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_aba_trajectory_is_deterministic(
    attractors,
    initial_state,
    sens_bias,
    adapt_bias,
) -> None:
    left = build_attractor_trajectory(
        trajectory_id="same",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sens_bias,
            adapt_bias,
            sens_bias,
        ),
        step_size=0.10,
    )

    right = build_attractor_trajectory(
        trajectory_id="same",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sens_bias,
            adapt_bias,
            sens_bias,
        ),
        step_size=0.10,
    )

    assert left == right


def test_aaa_trajectory_is_deterministic(
    attractors,
    initial_state,
    sens_bias,
) -> None:
    left = build_attractor_trajectory(
        trajectory_id="same_aaa",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sens_bias,
            sens_bias,
            sens_bias,
        ),
        step_size=0.10,
    )

    right = build_attractor_trajectory(
        trajectory_id="same_aaa",
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sens_bias,
            sens_bias,
            sens_bias,
        ),
        step_size=0.10,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(
    aba_trajectory,
) -> None:
    assert dominant_attractor_series(
        aba_trajectory
    ) == dominant_attractor_series(
        aba_trajectory
    )

    assert switch_flags(
        aba_trajectory
    ) == switch_flags(
        aba_trajectory
    )

    assert return_flags(
        aba_trajectory
    ) == return_flags(
        aba_trajectory
    )

    assert displacement_series(
        aba_trajectory
    ) == displacement_series(
        aba_trajectory
    )

    assert competition_entropy_series(
        aba_trajectory
    ) == competition_entropy_series(
        aba_trajectory
    )

    assert capture_margin_series(
        aba_trajectory
    ) == capture_margin_series(
        aba_trajectory
    )
