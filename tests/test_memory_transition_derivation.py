from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.history.associative_context import (
    AssociativeCueInput,
    build_associative_context,
)

from roif.history.attractor_dynamics import (
    AttractorPrestress,
    DynamicsState,
)

from roif.history.attractor_trajectory import (
    build_attractor_trajectory,
)

from roif.history.experience_attractor import (
    build_experience_attractor,
)

from roif.history.experience_conditioning import (
    ConditioningCue,
    build_conditioning_memory,
    observation_from_trajectory,
)

from roif.history.experience_pattern import (
    build_experience_pattern,
)

from roif.history.experience_sequence import (
    build_experience_sequence,
)

from roif.history.experience_transformation import (
    BodyResponse,
    ExperienceEvent,
    SystemResponse,
    SystemState,
    build_experience_transformation,
)

from roif.history.memory_transition_derivation import (
    DEFAULT_MAX_ABS_MODIFIER,
    MemoryTransitionDerivationError,
    MemoryTransitionDerivationResult,
    SCHEMA_VERSION,
    TransitionTargetBinding,
    derivation_is_policy_free,
    derivation_preserves_explicit_binding,
    derive_conditioning_transition_modifiers,
    derive_context_transition_modifiers,
    memory_transition_derivation_signature,
    transition_target_binding_signature,
)

from roif.history.transition_modifiers import (
    TransitionChannel,
)


# =============================================================================
# HELPERS
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
        physiological=tuple(
            physiological
        ),
        contextual=(0.5, 0.2),
        reserve=reserve,
    )


def make_response(
    response_id: str,
    magnitude: float,
):
    return SystemResponse(
        response_id=response_id,
        cognitive_response=(
            magnitude,
            magnitude * 0.9,
        ),
        physiological_response=(
            magnitude * 0.8,
            magnitude * 0.7,
        ),
        behavioral_response=(
            magnitude * 0.6,
            magnitude * 0.5,
        ),
    )


def make_body(
    magnitude: float,
):
    return BodyResponse(
        autonomic=(
            magnitude,
            magnitude * 0.9,
        ),
        endocrine=(
            magnitude * 0.8,
            magnitude * 0.7,
        ),
        immune=(
            magnitude * 0.4,
            magnitude * 0.3,
        ),
        motor=(
            magnitude * 0.75,
            magnitude * 0.65,
        ),
        interoceptive=(
            magnitude * 0.95,
            magnitude * 0.85,
        ),
    )


def build_sequence_from_states(
    sequence_id: str,
    event: ExperienceEvent,
    states,
    magnitudes,
):
    transformations = []

    for index in range(
        len(states) - 1
    ):
        transformations.append(
            build_experience_transformation(
                transformation_id=(
                    f"{sequence_id}_tx{index}"
                ),
                state_before=(
                    states[index]
                ),
                event=event,
                response=make_response(
                    f"{sequence_id}_r{index}",
                    magnitudes[index],
                ),
                body_response=make_body(
                    magnitudes[index]
                ),
                state_after=(
                    states[index + 1]
                ),
            )
        )

    return build_experience_sequence(
        sequence_id=sequence_id,
        transformations=tuple(
            transformations
        ),
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
            cognitive=(
                0.10 + offset,
                0.10 + offset,
                0.10 + offset,
            ),
            physiological=(
                0.10 + offset,
                0.08 + offset,
                0.06 + offset,
            ),
            reserve=(
                0.90 - offset
            ),
        ),
        make_state(
            f"{sequence_id}_s1",
            cognitive=(
                0.20 + offset,
                0.25 + offset,
                0.18 + offset,
            ),
            physiological=(
                0.20 + offset,
                0.18 + offset,
                0.16 + offset,
            ),
            reserve=(
                0.80 - offset
            ),
        ),
        make_state(
            f"{sequence_id}_s2",
            cognitive=(
                0.38 + offset,
                0.45 + offset,
                0.36 + offset,
            ),
            physiological=(
                0.40 + offset,
                0.36 + offset,
                0.32 + offset,
            ),
            reserve=(
                0.65 - offset
            ),
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


def build_attractor(
    event: ExperienceEvent,
):
    p1 = build_sensitizing_pattern(
        "sens_1",
        "sens_seq_1",
        event,
        0.00,
    )

    p2 = build_sensitizing_pattern(
        "sens_2",
        "sens_seq_2",
        event,
        0.01,
    )

    return build_experience_attractor(
        attractor_id=(
            "sensitization_attractor"
        ),
        patterns=(
            p1,
            p2,
        ),
    )


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def event():
    return ExperienceEvent(
        event_id="event_repeat",
        event_type="repeated_cue",
        structural_vector=(
            1.0,
            0.25,
            -0.1,
            0.4,
        ),
    )


@pytest.fixture
def attractor(
    event,
):
    return build_attractor(
        event
    )


@pytest.fixture
def initial_state(
    attractor,
):
    return DynamicsState(
        state_id="initial",
        feature_vector=(
            attractor.center
        ),
    )


@pytest.fixture
def prestress():
    return AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={
            "sensitization_attractor":
            0.75,
        },
    )


@pytest.fixture
def trajectory(
    attractor,
    initial_state,
    prestress,
):
    return build_attractor_trajectory(
        trajectory_id="sens_trajectory",
        initial_state=initial_state,
        attractors=(
            attractor,
        ),
        prestress_sequence=(
            prestress,
            prestress,
            prestress,
        ),
        step_size=0.10,
    )


@pytest.fixture
def cue_door():
    return ConditioningCue(
        cue_id="cue_door",
        cue_type="visual_symbol",
        feature_vector=(
            1.0,
            0.2,
            0.1,
            0.0,
        ),
    )


@pytest.fixture
def cue_room():
    return ConditioningCue(
        cue_id="cue_room",
        cue_type="visual_symbol",
        feature_vector=(
            0.9,
            0.3,
            0.15,
            0.05,
        ),
    )


@pytest.fixture
def memory(
    cue_door,
    trajectory,
):
    observations = tuple(
        observation_from_trajectory(
            observation_id=(
                f"obs_{index}"
            ),
            cue=cue_door,
            trajectory=trajectory,
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for index in range(5)
    )

    return build_conditioning_memory(
        memory_id="sens_memory",
        cue=cue_door,
        observations=observations,
    )


@pytest.fixture
def context(
    cue_door,
    cue_room,
    memory,
):
    return build_associative_context(
        context_id="sens_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_door,
                memory=memory,
                salience=1.0,
            ),
            AssociativeCueInput(
                cue=cue_room,
                memory=memory,
                salience=0.8,
            ),
        ),
    )


@pytest.fixture
def positive_binding():
    return TransitionTargetBinding(
        binding_id="bind_sens_to_AB",
        semantic_target_id=(
            "sensitization_attractor"
        ),
        physical_target_id="A_B",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        polarity=1.0,
        scale=0.5,
    )


@pytest.fixture
def negative_binding():
    return TransitionTargetBinding(
        binding_id="bind_sens_to_AB_negative",
        semantic_target_id=(
            "sensitization_attractor"
        ),
        physical_target_id="A_B",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        polarity=-1.0,
        scale=0.5,
    )


# =============================================================================
# CONTRACT
# =============================================================================


def test_schema_version():
    assert SCHEMA_VERSION == (
        "memory_transition_derivation_v1"
    )


def test_default_max_abs_modifier():
    assert DEFAULT_MAX_ABS_MODIFIER == pytest.approx(
        1.0
    )


def test_binding_constructs(
    positive_binding,
):
    assert (
        positive_binding.semantic_target_id
        ==
        "sensitization_attractor"
    )

    assert (
        positive_binding.physical_target_id
        ==
        "A_B"
    )


def test_binding_is_frozen(
    positive_binding,
):
    with pytest.raises(
        FrozenInstanceError
    ):
        positive_binding.scale = (
            0.25
        )  # type: ignore[misc]


def test_binding_metadata_read_only():
    binding = TransitionTargetBinding(
        binding_id="b",
        semantic_target_id="semantic",
        physical_target_id="A_B",
        metadata={
            "x": 1,
        },
    )

    assert isinstance(
        binding.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        binding.metadata[
            "x"
        ] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    "polarity",
    [
        -0.5,
        0.0,
        0.5,
        2.0,
    ],
)
def test_invalid_binding_polarity_rejected(
    polarity,
):
    with pytest.raises(
        MemoryTransitionDerivationError
    ):
        TransitionTargetBinding(
            binding_id="bad",
            semantic_target_id="semantic",
            physical_target_id="A_B",
            polarity=polarity,
        )


@pytest.mark.parametrize(
    "scale",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_binding_scale_rejected(
    scale,
):
    with pytest.raises(
        MemoryTransitionDerivationError
    ):
        TransitionTargetBinding(
            binding_id="bad",
            semantic_target_id="semantic",
            physical_target_id="A_B",
            scale=scale,
        )


# =============================================================================
# CONDITIONING DERIVATION
# =============================================================================


def test_conditioning_derivation_returns_result(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert isinstance(
        result,
        MemoryTransitionDerivationResult,
    )


def test_conditioning_derivation_mode(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert result.derivation_mode == (
        "conditioning_memory"
    )


def test_conditioning_derivation_generates_one_modifier(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert len(
        result.modifier_set.modifiers
    ) == 1


def test_conditioning_derivation_preserves_memory_id(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        result.modifier_set
        .source_memory_ids
        ==
        (
            "sens_memory",
        )
    )


def test_conditioning_derivation_targets_bound_physical_connection(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    modifier = (
        result.modifier_set
        .modifiers[0]
    )

    assert (
        modifier.target_id
        ==
        "A_B"
    )


def test_conditioning_positive_polarity_produces_positive_modifier(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        result.modifier_set
        .modifiers[0]
        .value
        > 0.0
    )


def test_conditioning_negative_polarity_produces_negative_modifier(
    cue_door,
    memory,
    negative_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                negative_binding,
            ),
        )
    )

    assert (
        result.modifier_set
        .modifiers[0]
        .value
        < 0.0
    )


def test_binding_scale_reduces_modifier(
    cue_door,
    memory,
):
    full = TransitionTargetBinding(
        binding_id="full",
        semantic_target_id=(
            "sensitization_attractor"
        ),
        physical_target_id="A_B",
        scale=1.0,
    )

    half = TransitionTargetBinding(
        binding_id="half",
        semantic_target_id=(
            "sensitization_attractor"
        ),
        physical_target_id="A_B",
        scale=0.5,
    )

    full_result = (
        derive_conditioning_transition_modifiers(
            derivation_id="full",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                full,
            ),
        )
    )

    half_result = (
        derive_conditioning_transition_modifiers(
            derivation_id="half",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                half,
            ),
        )
    )

    full_value = (
        full_result.modifier_set
        .modifiers[0]
        .value
    )

    half_value = (
        half_result.modifier_set
        .modifiers[0]
        .value
    )

    assert half_value == pytest.approx(
        full_value * 0.5
    )


def test_conditioning_requires_explicit_binding(
    cue_door,
    memory,
):
    wrong = TransitionTargetBinding(
        binding_id="wrong",
        semantic_target_id=(
            "other_attractor"
        ),
        physical_target_id="A_B",
    )

    with pytest.raises(
        MemoryTransitionDerivationError,
        match=(
            "no explicit physical transition binding"
        ),
    ):
        derive_conditioning_transition_modifiers(
            derivation_id="missing",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                wrong,
            ),
        )


def test_conditioning_preserves_explicit_binding(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        derivation_preserves_explicit_binding(
            result
        )
        is True
    )


def test_conditioning_derivation_is_policy_free(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert derivation_is_policy_free(
        result
    ) is True


# =============================================================================
# ASSOCIATIVE CONTEXT DERIVATION
# =============================================================================


def test_context_derivation_returns_result(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert isinstance(
        result,
        MemoryTransitionDerivationResult,
    )


def test_context_derivation_mode(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert result.derivation_mode == (
        "associative_context"
    )


def test_context_derivation_preserves_context_id(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        result.modifier_set
        .source_context_ids
        ==
        (
            "sens_context",
        )
    )


def test_context_derivation_preserves_source_memory_ids(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        result.modifier_set
        .source_memory_ids
        ==
        (
            "sens_memory",
        )
    )


def test_context_derivation_targets_bound_connection(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    modifier = (
        result.modifier_set
        .modifiers[0]
    )

    assert (
        modifier.target_id
        ==
        "A_B"
    )


def test_context_positive_polarity_positive_modifier(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        result.modifier_set
        .modifiers[0]
        .value
        > 0.0
    )


def test_context_negative_polarity_negative_modifier(
    context,
    negative_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                negative_binding,
            ),
        )
    )

    assert (
        result.modifier_set
        .modifiers[0]
        .value
        < 0.0
    )


def test_context_requires_binding_for_every_semantic_target(
    context,
):
    wrong = TransitionTargetBinding(
        binding_id="wrong",
        semantic_target_id=(
            "other_attractor"
        ),
        physical_target_id="A_B",
    )

    with pytest.raises(
        MemoryTransitionDerivationError,
        match=(
            "missing explicit physical transition bindings"
        ),
    ):
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                wrong,
            ),
        )


def test_context_derivation_preserves_semantic_signal_map(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        "sensitization_attractor"
        in result.semantic_signal_by_target
    )

    assert (
        result.semantic_signal_by_target[
            "sensitization_attractor"
        ]
        > 0.0
    )


def test_context_derivation_is_policy_free(
    context,
    positive_binding,
):
    result = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert derivation_is_policy_free(
        result
    ) is True


# =============================================================================
# ZERO CONTEXT SIGNAL
# =============================================================================


def test_zero_context_signal_can_produce_zero_modifier(
    cue_door,
    memory,
    positive_binding,
):
    zero_context = (
        build_associative_context(
            context_id="zero_context",
            cue_inputs=(
                AssociativeCueInput(
                    cue=cue_door,
                    memory=memory,
                    salience=0.0,
                ),
            ),
        )
    )

    result = (
        derive_context_transition_modifiers(
            derivation_id="zero",
            context=zero_context,
            bindings=(
                positive_binding,
            ),
            include_zero_signals=True,
        )
    )

    assert len(
        result.modifier_set.modifiers
    ) == 1

    assert (
        result.modifier_set
        .modifiers[0]
        .is_zero
        is True
    )


def test_zero_context_signal_skipped_by_default(
    cue_door,
    memory,
    positive_binding,
):
    zero_context = (
        build_associative_context(
            context_id="zero_context",
            cue_inputs=(
                AssociativeCueInput(
                    cue=cue_door,
                    memory=memory,
                    salience=0.0,
                ),
            ),
        )
    )

    result = (
        derive_context_transition_modifiers(
            derivation_id="zero",
            context=zero_context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        result.modifier_set.modifiers
        ==
        ()
    )

    assert (
        result.modifier_set.is_zero
        is True
    )


# =============================================================================
# ANTI-REDUCTION / SAFETY
# =============================================================================


def test_duplicate_physical_transition_target_rejected(
    context,
):
    left = TransitionTargetBinding(
        binding_id="left",
        semantic_target_id=(
            "sensitization_attractor"
        ),
        physical_target_id="A_B",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
    )

    right = TransitionTargetBinding(
        binding_id="right",
        semantic_target_id="other_attractor",
        physical_target_id="A_B",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
    )

    with pytest.raises(
        MemoryTransitionDerivationError,
        match=(
            "multiple bindings may not resolve "
            "to the same"
        ),
    ):
        derive_context_transition_modifiers(
            derivation_id="dup",
            context=context,
            bindings=(
                left,
                right,
            ),
        )

def test_binding_signature_is_deterministic(
    positive_binding,
):
    assert (
        transition_target_binding_signature(
            positive_binding
        )
        ==
        transition_target_binding_signature(
            positive_binding
        )
    )


def test_derivation_signature_is_deterministic(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        memory_transition_derivation_signature(
            result
        )
        ==
        memory_transition_derivation_signature(
            result
        )
    )


def test_same_memory_signal_different_binding_polarity_changes_modifier_sign(
    cue_door,
    memory,
    positive_binding,
    negative_binding,
):
    positive = (
        derive_conditioning_transition_modifiers(
            derivation_id="positive",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    negative = (
        derive_conditioning_transition_modifiers(
            derivation_id="negative",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                negative_binding,
            ),
        )
    )

    positive_value = (
        positive.modifier_set
        .modifiers[0]
        .value
    )

    negative_value = (
        negative.modifier_set
        .modifiers[0]
        .value
    )

    assert positive_value > 0.0
    assert negative_value < 0.0

    assert abs(
        positive_value
    ) == pytest.approx(
        abs(
            negative_value
        )
    )


def test_derivation_does_not_execute_transition(
    cue_door,
    memory,
    positive_binding,
):
    result = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert result.metadata[
        "transition_executed"
    ] is False

    assert result.modifier_set.metadata[
        "transition_executed"
    ] is False


def test_derivation_does_not_mutate_memory(
    cue_door,
    memory,
    positive_binding,
):
    before_observations = (
        memory.observations
    )

    before_associations = (
        memory.associations
    )

    derive_conditioning_transition_modifiers(
        derivation_id="conditioning",
        query_cue=cue_door,
        memory=memory,
        bindings=(
            positive_binding,
        ),
    )

    assert (
        memory.observations
        ==
        before_observations
    )

    assert (
        memory.associations
        ==
        before_associations
    )


# =============================================================================
# CENTRAL REGRESSION GUARD
# =============================================================================


def test_memory_transition_derivation_holds_together(
    cue_door,
    memory,
    context,
    positive_binding,
):
    """
    Central derivation-layer guard.

    Structured memory remains semantic.

    Physical target identity remains explicit.

    No topology is inferred.

    No action or policy is selected.

    Derivation only produces a bounded TransitionModifierSet.
    """

    conditioned = (
        derive_conditioning_transition_modifiers(
            derivation_id="conditioning",
            query_cue=cue_door,
            memory=memory,
            bindings=(
                positive_binding,
            ),
        )
    )

    contextual = (
        derive_context_transition_modifiers(
            derivation_id="context",
            context=context,
            bindings=(
                positive_binding,
            ),
        )
    )

    assert (
        conditioned.modifier_set
        .modifiers
    )

    assert (
        contextual.modifier_set
        .modifiers
    )

    assert (
        conditioned.modifier_set
        .modifiers[0]
        .target_id
        ==
        "A_B"
    )

    assert (
        contextual.modifier_set
        .modifiers[0]
        .target_id
        ==
        "A_B"
    )

    assert derivation_preserves_explicit_binding(
        conditioned
    ) is True

    assert derivation_preserves_explicit_binding(
        contextual
    ) is True

    assert derivation_is_policy_free(
        conditioned
    ) is True

    assert derivation_is_policy_free(
        contextual
    ) is True
