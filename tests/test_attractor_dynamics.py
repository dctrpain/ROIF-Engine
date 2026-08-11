"""
Tests for ROIF Memory 2.0 — Attractor Dynamics Core.

Layer stack:

    ExperienceTransformation
        -> ExperienceSequence
        -> ExperiencePattern
        -> ExperienceAttractor
        -> AttractorDynamics

The suite validates descriptive attractor-state dynamics without making a
biological attractor claim.
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
    AttractorActivation,
    AttractorDynamicsError,
    AttractorDynamicsResult,
    AttractorPrestress,
    DynamicsState,
    SCHEMA_VERSION,
    activation_by_id,
    basin_distance_series,
    capture_strength_series,
    competition_weight_series,
    dynamics_is_policy_free,
    run_attractor_dynamics,
    state_shift_magnitude,
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
    s1 = build_sensitizing_pattern(
        "sens_1",
        "sens_seq_1",
        event,
        0.00,
    )
    s2 = build_sensitizing_pattern(
        "sens_2",
        "sens_seq_2",
        event,
        0.01,
    )

    a1 = build_adaptation_pattern(
        "adapt_1",
        "adapt_seq_1",
        event,
    )
    a2 = build_adaptation_pattern(
        "adapt_2",
        "adapt_seq_2",
        event,
    )

    sensitization_attractor = build_experience_attractor(
        attractor_id="sensitization_attractor",
        patterns=(s1, s2),
    )

    adaptation_attractor = build_experience_attractor(
        attractor_id="adaptation_attractor",
        patterns=(a1, a2),
    )

    return (
        sensitization_attractor,
        adaptation_attractor,
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
def sensitization_attractor(attractors):
    return attractors[0]


@pytest.fixture
def adaptation_attractor(attractors):
    return attractors[1]


@pytest.fixture
def midpoint_state(
    sensitization_attractor,
    adaptation_attractor,
) -> DynamicsState:
    midpoint = tuple(
        (left + right) / 2.0
        for left, right in zip(
            sensitization_attractor.center,
            adaptation_attractor.center,
        )
    )

    return DynamicsState(
        state_id="midpoint",
        feature_vector=midpoint,
    )


@pytest.fixture
def near_sensitization_state(
    sensitization_attractor,
    adaptation_attractor,
) -> DynamicsState:
    vector = tuple(
        0.90 * sens
        + 0.10 * adapt
        for sens, adapt in zip(
            sensitization_attractor.center,
            adaptation_attractor.center,
        )
    )

    return DynamicsState(
        state_id="near_sens",
        feature_vector=vector,
    )


# =============================================================================
# Schema / object construction
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "attractor_dynamics_v1"


def test_dynamics_state_is_constructed(
    midpoint_state,
) -> None:
    assert isinstance(
        midpoint_state,
        DynamicsState,
    )


def test_dynamics_state_rejects_empty_id() -> None:
    with pytest.raises(
        AttractorDynamicsError
    ):
        DynamicsState(
            state_id="",
            feature_vector=(0.1,),
        )


def test_dynamics_state_rejects_empty_vector() -> None:
    with pytest.raises(
        AttractorDynamicsError
    ):
        DynamicsState(
            state_id="x",
            feature_vector=(),
        )


def test_prestress_is_constructed() -> None:
    prestress = AttractorPrestress(
        prestress_id="p",
        bias_by_attractor_id={
            "a": 0.1,
        },
    )

    assert isinstance(
        prestress,
        AttractorPrestress,
    )


def test_prestress_rejects_empty_id() -> None:
    with pytest.raises(
        AttractorDynamicsError
    ):
        AttractorPrestress(
            prestress_id="",
            bias_by_attractor_id={},
        )


# =============================================================================
# Baseline dynamics
# =============================================================================


def test_run_returns_dynamics_result(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert isinstance(
        result,
        AttractorDynamicsResult,
    )


def test_result_contains_two_activations(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert len(
        result.activations
    ) == 2


def test_activations_are_activation_objects(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert all(
        isinstance(
            item,
            AttractorActivation,
        )
        for item in result.activations
    )


def test_near_sensitization_state_has_lower_sens_distance(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    sens = activation_by_id(
        result,
        "sensitization_attractor",
    )

    adapt = activation_by_id(
        result,
        "adaptation_attractor",
    )

    assert sens.basin_distance < adapt.basin_distance


def test_near_sensitization_state_has_higher_sens_raw_activation(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    sens = activation_by_id(
        result,
        "sensitization_attractor",
    )

    adapt = activation_by_id(
        result,
        "adaptation_attractor",
    )

    assert sens.raw_activation > adapt.raw_activation


def test_near_sensitization_state_has_higher_sens_competition_weight(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    sens = activation_by_id(
        result,
        "sensitization_attractor",
    )

    adapt = activation_by_id(
        result,
        "adaptation_attractor",
    )

    assert sens.competition_weight > adapt.competition_weight


def test_near_sensitization_state_selects_sens_as_dominant_descriptor(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.dominant_attractor_id == "sensitization_attractor"


def test_competition_weights_sum_to_one(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert sum(
        competition_weight_series(
            result
        )
    ) == pytest.approx(
        1.0
    )


def test_capture_strengths_are_bounded(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert all(
        0.0 <= value <= 1.0
        for value in capture_strength_series(
            result
        )
    )


def test_basin_distances_are_nonnegative(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert all(
        value >= 0.0
        for value in basin_distance_series(
            result
        )
    )


# =============================================================================
# State shift
# =============================================================================


def test_state_after_has_same_dimension(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert len(
        result.state_after.feature_vector
    ) == len(
        near_sensitization_state.feature_vector
    )


def test_state_shift_has_same_dimension(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert len(
        result.state_shift
    ) == len(
        near_sensitization_state.feature_vector
    )


def test_state_shift_magnitude_positive(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert state_shift_magnitude(
        result
    ) > 0.0


def test_step_size_zero_produces_no_state_shift(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
        step_size=0.0,
    )

    assert result.state_shift == pytest.approx(
        tuple(
            0.0
            for _ in result.state_shift
        )
    )


def test_step_size_zero_preserves_state_vector(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
        step_size=0.0,
    )

    assert result.state_after.feature_vector == pytest.approx(
        near_sensitization_state.feature_vector
    )


def test_step_size_one_moves_state_to_weighted_target(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
        step_size=1.0,
    )

    assert result.state_after.feature_vector == pytest.approx(
        result.weighted_target
    )


def test_invalid_step_size_rejected(
    near_sensitization_state,
    attractors,
) -> None:
    with pytest.raises(
        AttractorDynamicsError
    ):
        run_attractor_dynamics(
            state=near_sensitization_state,
            attractors=attractors,
            step_size=1.1,
        )


def test_invalid_distance_scale_rejected(
    near_sensitization_state,
    attractors,
) -> None:
    with pytest.raises(
        AttractorDynamicsError
    ):
        run_attractor_dynamics(
            state=near_sensitization_state,
            attractors=attractors,
            distance_scale=0.0,
        )


# =============================================================================
# Prestress
# =============================================================================


def test_prestress_can_bias_adaptation_from_midpoint(
    midpoint_state,
    attractors,
) -> None:
    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    assert result.dominant_attractor_id == "adaptation_attractor"


def test_prestress_bias_recorded(
    midpoint_state,
    attractors,
) -> None:
    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    adaptation = activation_by_id(
        result,
        "adaptation_attractor",
    )

    assert adaptation.prestress_bias == pytest.approx(
        0.5
    )


def test_unlisted_attractor_gets_zero_bias(
    midpoint_state,
    attractors,
) -> None:
    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    sensitization = activation_by_id(
        result,
        "sensitization_attractor",
    )

    assert sensitization.prestress_bias == pytest.approx(
        0.0
    )


def test_prestress_does_not_mutate_attractor_center(
    midpoint_state,
    attractors,
) -> None:
    before = tuple(
        attractor.center
        for attractor in attractors
    )

    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    after = tuple(
        attractor.center
        for attractor in attractors
    )

    assert after == before


def test_result_metadata_declares_prestress_used(
    midpoint_state,
    attractors,
) -> None:
    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    assert result.metadata[
        "prestress_used"
    ] is True


def test_baseline_metadata_declares_no_prestress(
    midpoint_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
    )

    assert result.metadata[
        "prestress_used"
    ] is False


# =============================================================================
# Competition / uncertainty
# =============================================================================


def test_entropy_proxy_nonnegative(
    midpoint_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
    )

    assert result.competition_entropy_proxy >= 0.0


def test_capture_margin_nonnegative(
    midpoint_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
    )

    assert result.capture_margin >= 0.0


def test_midpoint_has_small_capture_margin(
    midpoint_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
    )

    assert result.capture_margin == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_midpoint_has_high_entropy_proxy(
    midpoint_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
    )

    assert result.competition_entropy_proxy == pytest.approx(
        0.5,
        abs=1e-12,
    )


def test_bias_increases_capture_margin(
    midpoint_state,
    attractors,
) -> None:
    baseline = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
    )

    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    biased = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    assert biased.capture_margin > baseline.capture_margin


def test_bias_reduces_entropy_proxy(
    midpoint_state,
    attractors,
) -> None:
    baseline = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
    )

    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    biased = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    assert (
        biased.competition_entropy_proxy
        < baseline.competition_entropy_proxy
    )


# =============================================================================
# Helpers / lookup
# =============================================================================


def test_activation_by_id_returns_expected_activation(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    activation = activation_by_id(
        result,
        "sensitization_attractor",
    )

    assert activation.attractor_id == "sensitization_attractor"


def test_activation_by_id_rejects_unknown(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    with pytest.raises(
        AttractorDynamicsError
    ):
        activation_by_id(
            result,
            "missing",
        )


# =============================================================================
# Policy / epistemic boundaries
# =============================================================================


def test_dynamics_is_policy_free_returns_true(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert dynamics_is_policy_free(
        result
    ) is True


def test_result_declares_no_memory_mutation(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.metadata[
        "memory_mutated"
    ] is False


def test_result_declares_no_learning(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.metadata[
        "learning_applied"
    ] is False


def test_result_declares_no_action_selected(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.metadata[
        "action_selected"
    ] is False


def test_result_declares_no_policy_modified(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.metadata[
        "policy_modified"
    ] is False


def test_result_declares_no_diagnosis(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.metadata[
        "diagnosis_generated"
    ] is False


def test_result_declares_no_biological_claim(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.metadata[
        "biological_attractor_claimed"
    ] is False


def test_result_declares_no_causal_truth(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert result.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_activation_declares_no_memory_mutation(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert all(
        activation.metadata[
            "memory_mutated"
        ] is False
        for activation in result.activations
    )


def test_every_activation_declares_no_policy_modification(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert all(
        activation.metadata[
            "policy_modified"
        ] is False
        for activation in result.activations
    )


# =============================================================================
# Immutability
# =============================================================================


def test_result_activations_are_tuple(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert isinstance(
        result.activations,
        tuple,
    )


def test_result_is_frozen(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        result.capture_margin = 1.0  # type: ignore[misc]


def test_activation_is_frozen(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        result.activations[
            0
        ].basin_distance = 0.0  # type: ignore[misc]


def test_state_is_frozen(
    midpoint_state,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        midpoint_state.state_id = "changed"  # type: ignore[misc]


def test_prestress_is_frozen() -> None:
    prestress = AttractorPrestress(
        prestress_id="p",
        bias_by_attractor_id={
            "a": 0.1,
        },
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        prestress.prestress_id = "changed"  # type: ignore[misc]


def test_prestress_bias_mapping_is_read_only() -> None:
    prestress = AttractorPrestress(
        prestress_id="p",
        bias_by_attractor_id={
            "a": 0.1,
        },
    )

    assert isinstance(
        prestress.bias_by_attractor_id,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        prestress.bias_by_attractor_id[
            "a"
        ] = 0.2  # type: ignore[index]


def test_result_metadata_is_read_only(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        result.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_activation_metadata_is_read_only(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert isinstance(
        result.activations[
            0
        ].metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        result.activations[
            0
        ].metadata[
            "x"
        ] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_baseline_dynamics_is_deterministic(
    near_sensitization_state,
    attractors,
) -> None:
    left = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    right = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert left == right


def test_prestressed_dynamics_is_deterministic(
    midpoint_state,
    attractors,
) -> None:
    prestress = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.5,
        },
    )

    left = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    right = run_attractor_dynamics(
        state=midpoint_state,
        attractors=attractors,
        prestress=prestress,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(
    near_sensitization_state,
    attractors,
) -> None:
    result = run_attractor_dynamics(
        state=near_sensitization_state,
        attractors=attractors,
    )

    assert competition_weight_series(
        result
    ) == competition_weight_series(
        result
    )

    assert capture_strength_series(
        result
    ) == capture_strength_series(
        result
    )

    assert basin_distance_series(
        result
    ) == basin_distance_series(
        result
    )

    assert state_shift_magnitude(
        result
    ) == state_shift_magnitude(
        result
    )
