"""
Tests for ROIF Memory 2.0 — Experience Transformation Core.

Core invariant:

    same structural event
    + different system state
    -> potentially different cognitive/body response
    -> different state_after
    -> different scar

The tests deliberately distinguish:
- event identity
- system state
- body response
- resulting scar
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.history.experience_transformation import (
    BodyResponse,
    ExperienceEvent,
    ExperienceTransformation,
    ExperienceTransformationError,
    SCHEMA_VERSION,
    ScarSignature,
    SystemResponse,
    SystemState,
    build_experience_transformation,
    cognitive_scar_magnitude,
    derive_scar_signature,
    physiological_scar_magnitude,
    repeated_event_response_changed,
    same_event_different_state,
    transition_magnitude,
    transition_vector,
    transformation_is_policy_free,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event() -> ExperienceEvent:
    return ExperienceEvent(
        event_id="event_door_001",
        event_type="door_cue",
        structural_vector=(1.0, 0.2, -0.1, 0.4),
        metadata={
            "external_expected_label_used": False,
        },
    )


@pytest.fixture
def calm_before() -> SystemState:
    return SystemState(
        state_id="state_calm_before",
        cognitive=(0.10, 0.20, 0.10),
        physiological=(0.10, 0.10, 0.05),
        contextual=(0.50, 0.20),
        reserve=0.90,
        metadata={
            "condition": "calm",
        },
    )


@pytest.fixture
def scarred_before() -> SystemState:
    return SystemState(
        state_id="state_scarred_before",
        cognitive=(0.35, 0.60, 0.40),
        physiological=(0.55, 0.40, 0.35),
        contextual=(0.50, 0.20),
        reserve=0.55,
        metadata={
            "condition": "sensitized",
        },
    )


@pytest.fixture
def calm_response() -> SystemResponse:
    return SystemResponse(
        response_id="response_calm",
        cognitive_response=(0.15, 0.20),
        physiological_response=(0.10, 0.05),
        behavioral_response=(0.05, 0.10),
    )


@pytest.fixture
def scarred_response() -> SystemResponse:
    return SystemResponse(
        response_id="response_sensitized",
        cognitive_response=(0.75, 0.80),
        physiological_response=(0.85, 0.70),
        behavioral_response=(0.65, 0.55),
    )


@pytest.fixture
def calm_body() -> BodyResponse:
    return BodyResponse(
        autonomic=(0.10, 0.05),
        endocrine=(0.05, 0.02),
        immune=(0.02, 0.01),
        motor=(0.10, 0.05),
        interoceptive=(0.08, 0.04),
    )


@pytest.fixture
def scarred_body() -> BodyResponse:
    return BodyResponse(
        autonomic=(0.85, 0.70),
        endocrine=(0.75, 0.60),
        immune=(0.35, 0.25),
        motor=(0.80, 0.65),
        interoceptive=(0.90, 0.80),
    )


@pytest.fixture
def calm_after() -> SystemState:
    return SystemState(
        state_id="state_calm_after",
        cognitive=(0.12, 0.22, 0.12),
        physiological=(0.12, 0.11, 0.06),
        contextual=(0.50, 0.20),
        reserve=0.88,
    )


@pytest.fixture
def scarred_after() -> SystemState:
    return SystemState(
        state_id="state_scarred_after",
        cognitive=(0.70, 0.90, 0.75),
        physiological=(0.90, 0.85, 0.80),
        contextual=(0.50, 0.20),
        reserve=0.35,
    )


@pytest.fixture
def calm_transformation(
    calm_before,
    event,
    calm_response,
    calm_body,
    calm_after,
) -> ExperienceTransformation:
    return build_experience_transformation(
        transformation_id="tx_calm",
        state_before=calm_before,
        event=event,
        response=calm_response,
        body_response=calm_body,
        state_after=calm_after,
    )


@pytest.fixture
def scarred_transformation(
    scarred_before,
    event,
    scarred_response,
    scarred_body,
    scarred_after,
) -> ExperienceTransformation:
    return build_experience_transformation(
        transformation_id="tx_scarred",
        state_before=scarred_before,
        event=event,
        response=scarred_response,
        body_response=scarred_body,
        state_after=scarred_after,
    )


# =============================================================================
# Schema / identity
# =============================================================================


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "experience_transformation_v1"


def test_system_state_requires_nonempty_id() -> None:
    with pytest.raises(ExperienceTransformationError):
        SystemState(
            state_id="",
            cognitive=(0.1,),
            physiological=(0.1,),
            contextual=(0.1,),
            reserve=0.5,
        )


def test_system_state_rejects_empty_cognitive_vector() -> None:
    with pytest.raises(ExperienceTransformationError):
        SystemState(
            state_id="x",
            cognitive=(),
            physiological=(0.1,),
            contextual=(0.1,),
            reserve=0.5,
        )


def test_system_state_rejects_empty_physiological_vector() -> None:
    with pytest.raises(ExperienceTransformationError):
        SystemState(
            state_id="x",
            cognitive=(0.1,),
            physiological=(),
            contextual=(0.1,),
            reserve=0.5,
        )


def test_system_state_rejects_empty_contextual_vector() -> None:
    with pytest.raises(ExperienceTransformationError):
        SystemState(
            state_id="x",
            cognitive=(0.1,),
            physiological=(0.1,),
            contextual=(),
            reserve=0.5,
        )


def test_system_state_rejects_reserve_below_zero() -> None:
    with pytest.raises(ExperienceTransformationError):
        SystemState(
            state_id="x",
            cognitive=(0.1,),
            physiological=(0.1,),
            contextual=(0.1,),
            reserve=-0.1,
        )


def test_system_state_rejects_reserve_above_one() -> None:
    with pytest.raises(ExperienceTransformationError):
        SystemState(
            state_id="x",
            cognitive=(0.1,),
            physiological=(0.1,),
            contextual=(0.1,),
            reserve=1.1,
        )


def test_experience_event_requires_nonempty_event_id() -> None:
    with pytest.raises(ExperienceTransformationError):
        ExperienceEvent(
            event_id="",
            event_type="door",
            structural_vector=(1.0,),
        )


def test_experience_event_requires_nonempty_event_type() -> None:
    with pytest.raises(ExperienceTransformationError):
        ExperienceEvent(
            event_id="e1",
            event_type="",
            structural_vector=(1.0,),
        )


def test_experience_event_rejects_empty_structural_vector() -> None:
    with pytest.raises(ExperienceTransformationError):
        ExperienceEvent(
            event_id="e1",
            event_type="door",
            structural_vector=(),
        )


def test_system_response_requires_nonempty_id() -> None:
    with pytest.raises(ExperienceTransformationError):
        SystemResponse(
            response_id="",
            cognitive_response=(0.1,),
            physiological_response=(0.1,),
            behavioral_response=(0.1,),
        )


def test_body_response_rejects_empty_autonomic_vector() -> None:
    with pytest.raises(ExperienceTransformationError):
        BodyResponse(
            autonomic=(),
            endocrine=(0.1,),
            immune=(0.1,),
            motor=(0.1,),
            interoceptive=(0.1,),
        )


# =============================================================================
# Scar derivation
# =============================================================================


def test_derive_scar_signature_returns_scar(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert isinstance(scar, ScarSignature)


def test_cognitive_delta_regression(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.cognitive_delta == pytest.approx(
        (0.02, 0.02, 0.02)
    )


def test_physiological_delta_regression(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.physiological_delta == pytest.approx(
        (0.02, 0.01, 0.01)
    )


def test_contextual_delta_can_be_zero(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.contextual_delta == pytest.approx(
        (0.0, 0.0)
    )


def test_reserve_delta_regression(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.reserve_delta == pytest.approx(-0.02)


def test_scar_total_magnitude_positive(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.total_magnitude > 0.0


def test_scar_declares_transition_descriptor(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.metadata["scar_is_transition_descriptor"] is True


def test_scar_does_not_claim_damage(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.metadata["scar_is_damage_claim"] is False


def test_scar_does_not_claim_causal_truth(
    calm_before,
    calm_after,
) -> None:
    scar = derive_scar_signature(
        calm_before,
        calm_after,
    )
    assert scar.metadata["causal_truth_inferred"] is False


def test_scar_rejects_dimension_mismatch() -> None:
    before = SystemState(
        state_id="before",
        cognitive=(0.1, 0.2),
        physiological=(0.1,),
        contextual=(0.1,),
        reserve=0.5,
    )
    after = SystemState(
        state_id="after",
        cognitive=(0.1,),
        physiological=(0.1,),
        contextual=(0.1,),
        reserve=0.5,
    )

    with pytest.raises(ExperienceTransformationError):
        derive_scar_signature(
            before,
            after,
        )


# =============================================================================
# Transformation construction
# =============================================================================


def test_build_returns_experience_transformation(
    calm_transformation,
) -> None:
    assert isinstance(
        calm_transformation,
        ExperienceTransformation,
    )


def test_transformation_preserves_before_state(
    calm_transformation,
    calm_before,
) -> None:
    assert calm_transformation.state_before == calm_before


def test_transformation_preserves_event(
    calm_transformation,
    event,
) -> None:
    assert calm_transformation.event == event


def test_transformation_preserves_response(
    calm_transformation,
    calm_response,
) -> None:
    assert calm_transformation.response == calm_response


def test_transformation_preserves_body_response(
    calm_transformation,
    calm_body,
) -> None:
    assert calm_transformation.body_response == calm_body


def test_transformation_preserves_after_state(
    calm_transformation,
    calm_after,
) -> None:
    assert calm_transformation.state_after == calm_after


def test_transformation_builds_scar_automatically(
    calm_transformation,
) -> None:
    assert isinstance(
        calm_transformation.scar,
        ScarSignature,
    )


def test_transformation_declares_policy_free(
    calm_transformation,
) -> None:
    assert transformation_is_policy_free(
        calm_transformation
    ) is True


def test_transformation_declares_no_diagnosis(
    calm_transformation,
) -> None:
    assert calm_transformation.metadata[
        "diagnosis_generated"
    ] is False


def test_transformation_declares_no_causal_truth(
    calm_transformation,
) -> None:
    assert calm_transformation.metadata[
        "causal_truth_inferred"
    ] is False


# =============================================================================
# Core Memory 2.0 principle
# =============================================================================


def test_same_structural_event_is_preserved(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert (
        calm_transformation.event.event_type
        == scarred_transformation.event.event_type
    )
    assert (
        calm_transformation.event.structural_vector
        == scarred_transformation.event.structural_vector
    )


def test_same_event_different_state_detected(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert same_event_different_state(
        calm_transformation,
        scarred_transformation,
    ) is True


def test_same_event_can_have_different_cognitive_response(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert (
        calm_transformation.response.cognitive_response
        != scarred_transformation.response.cognitive_response
    )


def test_same_event_can_have_different_physiological_response(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert (
        calm_transformation.response.physiological_response
        != scarred_transformation.response.physiological_response
    )


def test_same_event_can_have_different_body_response(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert (
        calm_transformation.body_response
        != scarred_transformation.body_response
    )


def test_same_event_can_have_different_interoceptive_return(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert (
        calm_transformation.body_response.interoceptive
        != scarred_transformation.body_response.interoceptive
    )


def test_same_event_can_have_different_state_after(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert (
        calm_transformation.state_after
        != scarred_transformation.state_after
    )


def test_same_event_can_generate_different_scar(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert (
        calm_transformation.scar
        != scarred_transformation.scar
    )


def test_repeated_event_response_changed_detected(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert repeated_event_response_changed(
        calm_transformation,
        scarred_transformation,
    ) is True


def test_sensitized_case_has_larger_physiological_scar(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert physiological_scar_magnitude(
        scarred_transformation
    ) > physiological_scar_magnitude(
        calm_transformation
    )


def test_sensitized_case_has_larger_cognitive_scar(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert cognitive_scar_magnitude(
        scarred_transformation
    ) > cognitive_scar_magnitude(
        calm_transformation
    )


def test_sensitized_case_has_larger_total_transition(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert transition_magnitude(
        scarred_transformation
    ) > transition_magnitude(
        calm_transformation
    )


def test_transition_vectors_differ_between_states(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert transition_vector(
        calm_transformation
    ) != transition_vector(
        scarred_transformation
    )


# =============================================================================
# Negative controls
# =============================================================================


def test_same_event_same_state_returns_false_for_different_state_helper(
    calm_transformation,
) -> None:
    duplicate = build_experience_transformation(
        transformation_id="duplicate",
        state_before=calm_transformation.state_before,
        event=calm_transformation.event,
        response=calm_transformation.response,
        body_response=calm_transformation.body_response,
        state_after=calm_transformation.state_after,
    )

    assert same_event_different_state(
        calm_transformation,
        duplicate,
    ) is False


def test_different_event_returns_false_for_same_event_helper(
    calm_transformation,
) -> None:
    other_event = ExperienceEvent(
        event_id="event_other",
        event_type="window_cue",
        structural_vector=(0.5, 0.1, 0.3, 0.2),
    )

    other = build_experience_transformation(
        transformation_id="other",
        state_before=SystemState(
            state_id="other_before",
            cognitive=(0.9, 0.9, 0.9),
            physiological=(0.9, 0.9, 0.9),
            contextual=(0.5, 0.2),
            reserve=0.3,
        ),
        event=other_event,
        response=calm_transformation.response,
        body_response=calm_transformation.body_response,
        state_after=calm_transformation.state_after,
    )

    assert same_event_different_state(
        calm_transformation,
        other,
    ) is False


def test_repeated_event_response_changed_false_when_not_same_event(
    calm_transformation,
) -> None:
    other_event = ExperienceEvent(
        event_id="event_other",
        event_type="window_cue",
        structural_vector=(0.5, 0.1, 0.3, 0.2),
    )

    other = build_experience_transformation(
        transformation_id="other",
        state_before=calm_transformation.state_before,
        event=other_event,
        response=calm_transformation.response,
        body_response=calm_transformation.body_response,
        state_after=calm_transformation.state_after,
    )

    assert repeated_event_response_changed(
        calm_transformation,
        other,
    ) is False


# =============================================================================
# Immutability
# =============================================================================


def test_system_state_is_frozen(
    calm_before,
) -> None:
    with pytest.raises(FrozenInstanceError):
        calm_before.reserve = 0.1  # type: ignore[misc]


def test_event_is_frozen(
    event,
) -> None:
    with pytest.raises(FrozenInstanceError):
        event.event_type = "changed"  # type: ignore[misc]


def test_body_response_is_frozen(
    calm_body,
) -> None:
    with pytest.raises(FrozenInstanceError):
        calm_body.autonomic = (1.0,)  # type: ignore[misc]


def test_scar_is_frozen(
    calm_transformation,
) -> None:
    with pytest.raises(FrozenInstanceError):
        calm_transformation.scar.reserve_delta = 0.5  # type: ignore[misc]


def test_transformation_is_frozen(
    calm_transformation,
) -> None:
    with pytest.raises(FrozenInstanceError):
        calm_transformation.transformation_id = "changed"  # type: ignore[misc]


def test_state_metadata_is_read_only(
    calm_before,
) -> None:
    assert isinstance(
        calm_before.metadata,
        MappingProxyType,
    )
    with pytest.raises(TypeError):
        calm_before.metadata["x"] = 1  # type: ignore[index]


def test_event_metadata_is_read_only(
    event,
) -> None:
    assert isinstance(
        event.metadata,
        MappingProxyType,
    )
    with pytest.raises(TypeError):
        event.metadata["x"] = 1  # type: ignore[index]


def test_scar_metadata_is_read_only(
    calm_transformation,
) -> None:
    assert isinstance(
        calm_transformation.scar.metadata,
        MappingProxyType,
    )
    with pytest.raises(TypeError):
        calm_transformation.scar.metadata["x"] = 1  # type: ignore[index]


def test_transformation_metadata_is_read_only(
    calm_transformation,
) -> None:
    assert isinstance(
        calm_transformation.metadata,
        MappingProxyType,
    )
    with pytest.raises(TypeError):
        calm_transformation.metadata["x"] = 1  # type: ignore[index]


# =============================================================================
# Determinism
# =============================================================================


def test_scar_derivation_is_deterministic(
    calm_before,
    calm_after,
) -> None:
    left = derive_scar_signature(
        calm_before,
        calm_after,
    )
    right = derive_scar_signature(
        calm_before,
        calm_after,
    )

    assert left == right


def test_transformation_build_is_deterministic(
    calm_before,
    event,
    calm_response,
    calm_body,
    calm_after,
) -> None:
    left = build_experience_transformation(
        transformation_id="same",
        state_before=calm_before,
        event=event,
        response=calm_response,
        body_response=calm_body,
        state_after=calm_after,
    )

    right = build_experience_transformation(
        transformation_id="same",
        state_before=calm_before,
        event=event,
        response=calm_response,
        body_response=calm_body,
        state_after=calm_after,
    )

    assert left == right


def test_transition_vector_is_deterministic(
    calm_transformation,
) -> None:
    assert transition_vector(
        calm_transformation
    ) == transition_vector(
        calm_transformation
    )


def test_complete_same_event_different_state_semantics_are_deterministic(
    calm_transformation,
    scarred_transformation,
) -> None:
    assert same_event_different_state(
        calm_transformation,
        scarred_transformation,
    ) is True

    assert repeated_event_response_changed(
        calm_transformation,
        scarred_transformation,
    ) is True
