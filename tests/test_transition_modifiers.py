from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.history.transition_modifiers import (
    MAX_MODIFIER_VALUE,
    MIN_MODIFIER_VALUE,
    MemoryEvidenceRef,
    MemorySourceKind,
    SCHEMA_VERSION,
    TransitionChannel,
    TransitionModifier,
    TransitionModifierError,
    TransitionModifierSet,
    ZERO_TOLERANCE,
    build_transition_modifier_set,
    memory_evidence_signature,
    modifier_set_preserves_structured_provenance,
    transition_modifier_set_is_policy_free,
    transition_modifier_set_signature,
    transition_modifier_signature,
    zero_transition_modifier_set,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def evidence_trace():
    return MemoryEvidenceRef(
        source_kind=MemorySourceKind.HISTORICAL_TRACE,
        source_id="trace_E1",
        order_index=0,
        relation="reinforcement",
        metadata={
            "source": "test",
        },
    )


@pytest.fixture
def evidence_conditioning():
    return MemoryEvidenceRef(
        source_kind=MemorySourceKind.CONDITIONING_MEMORY,
        source_id="memory_door",
        order_index=1,
        cue_id="cue_door",
        attractor_id="sensitization_attractor",
        relation="conditioned_match",
    )


@pytest.fixture
def evidence_context():
    return MemoryEvidenceRef(
        source_kind=MemorySourceKind.ASSOCIATIVE_CONTEXT,
        source_id="context_room",
        order_index=2,
        cue_id="cue_room",
        attractor_id="sensitization_attractor",
        context_id="room_context",
        relation="associative_context",
    )


@pytest.fixture
def modifier(
    evidence_conditioning,
):
    return TransitionModifier(
        modifier_id="m_stiffness_AB",
        channel=(
            TransitionChannel.CONNECTION_STIFFNESS_UPDATE
        ),
        target_id="A_B",
        value=0.25,
        evidence=(
            evidence_conditioning,
        ),
        metadata={
            "source": "unit_test",
        },
    )


@pytest.fixture
def second_modifier(
    evidence_context,
):
    return TransitionModifier(
        modifier_id="m_transfer_AB",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=-0.15,
        evidence=(
            evidence_context,
        ),
    )


@pytest.fixture
def modifier_set(
    modifier,
    second_modifier,
):
    return TransitionModifierSet(
        modifier_set_id="modifier_set_1",
        modifiers=(
            modifier,
            second_modifier,
        ),
        source_memory_ids=(
            "memory_door",
        ),
        source_context_ids=(
            "room_context",
        ),
        metadata={
            "policy_modified": False,
        },
    )


# =============================================================================
# SCHEMA / CONSTANTS
# =============================================================================


def test_schema_version():
    assert SCHEMA_VERSION == (
        "transition_modifiers_v1"
    )


def test_modifier_bounds_are_symmetric():
    assert MIN_MODIFIER_VALUE == pytest.approx(
        -1.0
    )

    assert MAX_MODIFIER_VALUE == pytest.approx(
        1.0
    )


def test_zero_tolerance_positive():
    assert ZERO_TOLERANCE > 0.0


# =============================================================================
# ENUMS
# =============================================================================


def test_transition_channels_are_strings():
    assert (
        TransitionChannel.PRESTRESS_TRANSFER.value
        ==
        "prestress_transfer"
    )


def test_memory_source_kinds_are_strings():
    assert (
        MemorySourceKind.ASSOCIATIVE_CONTEXT.value
        ==
        "associative_context"
    )


# =============================================================================
# MEMORY EVIDENCE
# =============================================================================


def test_memory_evidence_constructs(
    evidence_trace,
):
    assert (
        evidence_trace.source_id
        ==
        "trace_E1"
    )

    assert (
        evidence_trace.order_index
        ==
        0
    )


def test_memory_evidence_metadata_is_read_only(
    evidence_trace,
):
    assert isinstance(
        evidence_trace.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        evidence_trace.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_memory_evidence_is_frozen(
    evidence_trace,
):
    with pytest.raises(
        FrozenInstanceError
    ):
        evidence_trace.source_id = (
            "changed"
        )  # type: ignore[misc]


@pytest.mark.parametrize(
    "source_id",
    [
        "",
        "   ",
    ],
)
def test_empty_memory_source_id_rejected(
    source_id,
):
    with pytest.raises(
        TransitionModifierError
    ):
        MemoryEvidenceRef(
            source_kind=(
                MemorySourceKind.HISTORICAL_TRACE
            ),
            source_id=source_id,
        )


def test_negative_order_index_rejected():
    with pytest.raises(
        TransitionModifierError
    ):
        MemoryEvidenceRef(
            source_kind=(
                MemorySourceKind.HISTORICAL_TRACE
            ),
            source_id="trace",
            order_index=-1,
        )


def test_non_integer_order_index_rejected():
    with pytest.raises(
        TransitionModifierError
    ):
        MemoryEvidenceRef(
            source_kind=(
                MemorySourceKind.HISTORICAL_TRACE
            ),
            source_id="trace",
            order_index=1.5,  # type: ignore[arg-type]
        )


def test_invalid_source_kind_rejected():
    with pytest.raises(
        TransitionModifierError
    ):
        MemoryEvidenceRef(
            source_kind="history",  # type: ignore[arg-type]
            source_id="trace",
        )


# =============================================================================
# TRANSITION MODIFIER
# =============================================================================


def test_transition_modifier_constructs(
    modifier,
):
    assert (
        modifier.modifier_id
        ==
        "m_stiffness_AB"
    )

    assert (
        modifier.target_id
        ==
        "A_B"
    )

    assert modifier.value == pytest.approx(
        0.25
    )


def test_transition_modifier_is_frozen(
    modifier,
):
    with pytest.raises(
        FrozenInstanceError
    ):
        modifier.value = (
            0.0
        )  # type: ignore[misc]


def test_transition_modifier_metadata_is_read_only(
    modifier,
):
    assert isinstance(
        modifier.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        modifier.metadata[
            "x"
        ] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "value",
    [
        -1.000001,
        1.000001,
    ],
)
def test_modifier_outside_bounds_rejected(
    value,
    evidence_trace,
):
    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifier(
            modifier_id="bad",
            channel=(
                TransitionChannel.PRESTRESS_TRANSFER
            ),
            target_id="A_B",
            value=value,
            evidence=(
                evidence_trace,
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_non_finite_modifier_value_rejected(
    value,
    evidence_trace,
):
    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifier(
            modifier_id="bad",
            channel=(
                TransitionChannel.PRESTRESS_TRANSFER
            ),
            target_id="A_B",
            value=value,
            evidence=(
                evidence_trace,
            ),
        )


def test_nonzero_modifier_requires_evidence():
    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifier(
            modifier_id="anonymous_gain",
            channel=(
                TransitionChannel.PRESTRESS_TRANSFER
            ),
            target_id="A_B",
            value=0.5,
            evidence=(),
        )


def test_zero_modifier_may_have_no_evidence():
    modifier = TransitionModifier(
        modifier_id="zero",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=0.0,
        evidence=(),
    )

    assert modifier.is_zero is True


def test_zero_tolerance_counts_as_zero():
    modifier = TransitionModifier(
        modifier_id="near_zero",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=(
            ZERO_TOLERANCE
            / 2.0
        ),
        evidence=(),
    )

    assert modifier.is_zero is True


def test_active_modifier_not_zero(
    modifier,
):
    assert modifier.is_zero is False


def test_modifier_key(
    modifier,
):
    assert modifier.key == (
        TransitionChannel.CONNECTION_STIFFNESS_UPDATE,
        "A_B",
    )


def test_invalid_modifier_channel_rejected(
    evidence_trace,
):
    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifier(
            modifier_id="bad_channel",
            channel="x",  # type: ignore[arg-type]
            target_id="A_B",
            value=0.1,
            evidence=(
                evidence_trace,
            ),
        )


@pytest.mark.parametrize(
    "target_id",
    [
        "",
        "   ",
    ],
)
def test_empty_modifier_target_rejected(
    target_id,
    evidence_trace,
):
    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifier(
            modifier_id="bad_target",
            channel=(
                TransitionChannel.PRESTRESS_TRANSFER
            ),
            target_id=target_id,
            value=0.1,
            evidence=(
                evidence_trace,
            ),
        )


# =============================================================================
# MODIFIER SET
# =============================================================================


def test_modifier_set_constructs(
    modifier_set,
):
    assert (
        modifier_set.modifier_set_id
        ==
        "modifier_set_1"
    )

    assert len(
        modifier_set.modifiers
    ) == 2


def test_modifier_set_is_frozen(
    modifier_set,
):
    with pytest.raises(
        FrozenInstanceError
    ):
        modifier_set.modifier_set_id = (
            "changed"
        )  # type: ignore[misc]


def test_modifier_set_metadata_read_only(
    modifier_set,
):
    assert isinstance(
        modifier_set.metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError
    ):
        modifier_set.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_modifier_set_has_effect(
    modifier_set,
):
    assert (
        modifier_set.has_effect
        is True
    )

    assert (
        modifier_set.is_zero
        is False
    )


def test_active_modifiers_returns_only_nonzero(
    evidence_trace,
):
    active = TransitionModifier(
        modifier_id="active",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=0.3,
        evidence=(
            evidence_trace,
        ),
    )

    zero = TransitionModifier(
        modifier_id="zero",
        channel=(
            TransitionChannel.RESERVE_SENSITIVITY
        ),
        target_id="A",
        value=0.0,
    )

    result = TransitionModifierSet(
        modifier_set_id="mixed",
        modifiers=(
            zero,
            active,
        ),
    )

    assert result.active_modifiers == (
        active,
    )


def test_channels_preserve_first_seen_order(
    modifier,
    second_modifier,
    evidence_trace,
):
    third = TransitionModifier(
        modifier_id="m_reflex_BC",
        channel=(
            TransitionChannel.CONNECTION_STIFFNESS_UPDATE
        ),
        target_id="B_C",
        value=0.1,
        evidence=(
            evidence_trace,
        ),
    )

    result = TransitionModifierSet(
        modifier_set_id="channels",
        modifiers=(
            modifier,
            second_modifier,
            third,
        ),
    )

    assert result.channels == (
        TransitionChannel.CONNECTION_STIFFNESS_UPDATE,
        TransitionChannel.PRESTRESS_TRANSFER,
    )


def test_duplicate_channel_target_rejected(
    modifier,
    evidence_trace,
):
    duplicate = TransitionModifier(
        modifier_id="duplicate",
        channel=(
            modifier.channel
        ),
        target_id=(
            modifier.target_id
        ),
        value=-0.1,
        evidence=(
            evidence_trace,
        ),
    )

    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifierSet(
            modifier_set_id="duplicates",
            modifiers=(
                modifier,
                duplicate,
            ),
        )


def test_duplicate_modifier_ids_rejected(
    modifier,
    evidence_trace,
):
    duplicate_id = TransitionModifier(
        modifier_id=(
            modifier.modifier_id
        ),
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="B_C",
        value=0.1,
        evidence=(
            evidence_trace,
        ),
    )

    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifierSet(
            modifier_set_id="duplicate_ids",
            modifiers=(
                modifier,
                duplicate_id,
            ),
        )


def test_duplicate_source_memory_ids_rejected():
    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifierSet(
            modifier_set_id="bad_memory_ids",
            source_memory_ids=(
                "memory_1",
                "memory_1",
            ),
        )


def test_duplicate_source_context_ids_rejected():
    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifierSet(
            modifier_set_id="bad_context_ids",
            source_context_ids=(
                "context_1",
                "context_1",
            ),
        )


# =============================================================================
# LOOKUP HELPERS
# =============================================================================


def test_modifier_for_returns_match(
    modifier_set,
    modifier,
):
    result = (
        modifier_set.modifier_for(
            channel=modifier.channel,
            target_id=(
                modifier.target_id
            ),
        )
    )

    assert result == modifier


def test_modifier_for_returns_none_when_missing(
    modifier_set,
):
    result = (
        modifier_set.modifier_for(
            channel=(
                TransitionChannel.RESERVE_SENSITIVITY
            ),
            target_id="missing",
        )
    )

    assert result is None


def test_modifiers_for_channel(
    modifier_set,
    modifier,
):
    result = (
        modifier_set.modifiers_for_channel(
            TransitionChannel.CONNECTION_STIFFNESS_UPDATE
        )
    )

    assert result == (
        modifier,
    )


def test_modifiers_for_target(
    modifier_set,
):
    result = (
        modifier_set.modifiers_for_target(
            "A_B"
        )
    )

    assert len(
        result
    ) == 2


# =============================================================================
# ZERO MODIFIER SET
# =============================================================================


def test_zero_modifier_set_constructs():
    result = (
        zero_transition_modifier_set()
    )

    assert result.modifiers == ()
    assert result.is_zero is True
    assert result.has_effect is False
    assert result.active_modifiers == ()


def test_zero_modifier_set_has_no_sources():
    result = (
        zero_transition_modifier_set()
    )

    assert (
        result.source_memory_ids
        ==
        ()
    )

    assert (
        result.source_context_ids
        ==
        ()
    )


def test_zero_modifier_set_is_policy_free():
    result = (
        zero_transition_modifier_set()
    )

    assert transition_modifier_set_is_policy_free(
        result
    ) is True


def test_zero_modifier_set_preserves_provenance_vacuously():
    result = (
        zero_transition_modifier_set()
    )

    assert modifier_set_preserves_structured_provenance(
        result
    ) is True


# =============================================================================
# BUILDER
# =============================================================================


def test_build_transition_modifier_set(
    modifier,
    second_modifier,
):
    result = (
        build_transition_modifier_set(
            modifier_set_id="built",
            modifiers=(
                modifier,
                second_modifier,
            ),
            source_memory_ids=(
                "memory_1",
            ),
            source_context_ids=(
                "context_1",
            ),
        )
    )

    assert result.modifiers == (
        modifier,
        second_modifier,
    )

    assert (
        result.source_memory_ids
        ==
        (
            "memory_1",
        )
    )


# =============================================================================
# SIGNATURES
# =============================================================================


def test_memory_evidence_signature_is_deterministic(
    evidence_context,
):
    left = (
        memory_evidence_signature(
            evidence_context
        )
    )

    right = (
        memory_evidence_signature(
            evidence_context
        )
    )

    assert left == right


def test_memory_evidence_signature_preserves_order_index(
    evidence_context,
):
    signature = (
        memory_evidence_signature(
            evidence_context
        )
    )

    assert signature[
        2
    ] == evidence_context.order_index


def test_transition_modifier_signature_is_deterministic(
    modifier,
):
    assert (
        transition_modifier_signature(
            modifier
        )
        ==
        transition_modifier_signature(
            modifier
        )
    )


def test_transition_modifier_signature_contains_structured_evidence(
    modifier,
):
    signature = (
        transition_modifier_signature(
            modifier
        )
    )

    evidence_signature = (
        signature[-1]
    )

    assert evidence_signature
    assert (
        evidence_signature[
            0
        ][
            0
        ]
        ==
        MemorySourceKind.CONDITIONING_MEMORY.value
    )


def test_modifier_set_signature_is_deterministic(
    modifier_set,
):
    assert (
        transition_modifier_set_signature(
            modifier_set
        )
        ==
        transition_modifier_set_signature(
            modifier_set
        )
    )


def test_modifier_set_signature_preserves_source_identity(
    modifier_set,
):
    signature = (
        transition_modifier_set_signature(
            modifier_set
        )
    )

    assert signature[
        2
    ] == (
        "memory_door",
    )

    assert signature[
        3
    ] == (
        "room_context",
    )


# =============================================================================
# ARCHITECTURAL GUARDS
# =============================================================================


def test_nonzero_modifier_set_preserves_structured_provenance(
    modifier_set,
):
    assert modifier_set_preserves_structured_provenance(
        modifier_set
    ) is True


def test_policy_free_modifier_set(
    modifier_set,
):
    assert transition_modifier_set_is_policy_free(
        modifier_set
    ) is True


@pytest.mark.parametrize(
    "metadata_key",
    [
        "action_selected",
        "policy_modified",
        "memory_mutated",
        "transition_executed",
    ],
)
def test_policy_free_guard_rejects_execution_flags(
    modifier,
    metadata_key,
):
    result = TransitionModifierSet(
        modifier_set_id="not_policy_free",
        modifiers=(
            modifier,
        ),
        metadata={
            metadata_key: True,
        },
    )

    assert transition_modifier_set_is_policy_free(
        result
    ) is False


# =============================================================================
# ANTI-REDUCTION TESTS
# =============================================================================


def test_ordered_evidence_produces_different_signature(
    evidence_trace,
):
    first = MemoryEvidenceRef(
        source_kind=(
            evidence_trace.source_kind
        ),
        source_id=(
            evidence_trace.source_id
        ),
        order_index=0,
        relation=(
            evidence_trace.relation
        ),
    )

    second = MemoryEvidenceRef(
        source_kind=(
            evidence_trace.source_kind
        ),
        source_id=(
            evidence_trace.source_id
        ),
        order_index=1,
        relation=(
            evidence_trace.relation
        ),
    )

    assert memory_evidence_signature(
        first
    ) != memory_evidence_signature(
        second
    )


def test_different_context_identity_produces_different_evidence_signature():
    left = MemoryEvidenceRef(
        source_kind=(
            MemorySourceKind.ASSOCIATIVE_CONTEXT
        ),
        source_id="context_source",
        context_id="context_A",
    )

    right = MemoryEvidenceRef(
        source_kind=(
            MemorySourceKind.ASSOCIATIVE_CONTEXT
        ),
        source_id="context_source",
        context_id="context_B",
    )

    assert (
        memory_evidence_signature(
            left
        )
        !=
        memory_evidence_signature(
            right
        )
    )


def test_different_attractor_identity_produces_different_evidence_signature():
    left = MemoryEvidenceRef(
        source_kind=(
            MemorySourceKind.EXPERIENCE_ATTRACTOR
        ),
        source_id="attractor_source",
        attractor_id="A",
    )

    right = MemoryEvidenceRef(
        source_kind=(
            MemorySourceKind.EXPERIENCE_ATTRACTOR
        ),
        source_id="attractor_source",
        attractor_id="B",
    )

    assert (
        memory_evidence_signature(
            left
        )
        !=
        memory_evidence_signature(
            right
        )
    )


def test_same_numeric_modifier_with_different_memory_provenance_is_not_same_signature():
    left_evidence = MemoryEvidenceRef(
        source_kind=(
            MemorySourceKind.CONDITIONING_MEMORY
        ),
        source_id="memory_A",
        cue_id="cue",
    )

    right_evidence = MemoryEvidenceRef(
        source_kind=(
            MemorySourceKind.CONDITIONING_MEMORY
        ),
        source_id="memory_B",
        cue_id="cue",
    )

    left = TransitionModifier(
        modifier_id="left",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=0.25,
        evidence=(
            left_evidence,
        ),
    )

    right = TransitionModifier(
        modifier_id="right",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=0.25,
        evidence=(
            right_evidence,
        ),
    )

    assert (
        transition_modifier_signature(
            left
        )
        !=
        transition_modifier_signature(
            right
        )
    )


def test_modifier_set_does_not_implicitly_combine_conflicting_same_target_evidence(
    evidence_trace,
):
    positive = TransitionModifier(
        modifier_id="positive",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=0.5,
        evidence=(
            evidence_trace,
        ),
    )

    negative = TransitionModifier(
        modifier_id="negative",
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id="A_B",
        value=-0.5,
        evidence=(
            evidence_trace,
        ),
    )

    with pytest.raises(
        TransitionModifierError
    ):
        TransitionModifierSet(
            modifier_set_id="conflict",
            modifiers=(
                positive,
                negative,
            ),
        )


# =============================================================================
# CENTRAL CONTRACT REGRESSION GUARD
# =============================================================================


def test_transition_modifier_contract_holds_together(
    modifier_set,
):
    """
    Central regression guard for the Memory-to-Transition contract.

    The contract must preserve:

        explicit channels,
        explicit targets,
        bounded values,
        structured provenance,
        immutability,
        deterministic signatures,
        no implicit history aggregation,
        policy-free semantics,
        canonical zero state.
    """

    assert (
        modifier_set.has_effect
        is True
    )

    assert (
        modifier_set.is_zero
        is False
    )

    assert modifier_set_preserves_structured_provenance(
        modifier_set
    ) is True

    assert transition_modifier_set_is_policy_free(
        modifier_set
    ) is True

    assert (
        transition_modifier_set_signature(
            modifier_set
        )
        ==
        transition_modifier_set_signature(
            modifier_set
        )
    )

    zero = (
        zero_transition_modifier_set()
    )

    assert zero.is_zero is True
    assert zero.has_effect is False
    assert zero.modifiers == ()
