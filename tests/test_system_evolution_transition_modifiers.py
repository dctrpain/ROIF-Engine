"""
Integration tests for Memory-to-Transition modifiers in SystemEvolution.

These tests protect the first executable transition channel:

    TransitionChannel.PRESTRESS_TRANSFER

Critical invariants
-------------------

1. Legacy evolution == explicit None == canonical zero modifier set.
2. Zero modifier sets do not alter result metadata.
3. PRESTRESS_TRANSFER modifies only transfer through existing topology.
4. Adaptive connection evolution is unchanged by transfer modifiers.
5. Positive and negative modifiers change downstream redistribution
   directionally.
6. Unsupported channels and unknown targets fail explicitly.
7. The same modifier set can propagate through evolve_sequence().
"""

from __future__ import annotations

import pytest

from roif.history.adaptive_connection import (
    AdaptiveConnectionConfig,
    AdaptiveConnectionState,
    ConnectionExposure,
)
from roif.history.prestress_redistribution import (
    PrestressNodeState,
    PrestressPerturbation,
    PrestressRedistributionConfig,
)
from roif.history.system_evolution import (
    SystemEvent,
    SystemEvolutionConfig,
    SystemEvolutionError,
    evolve_sequence,
    evolve_system,
)
from roif.history.system_image import (
    HistoricalTrace,
    SystemImageContext,
    SystemMeasure,
    build_system_image,
    system_image_signature,
)
from roif.history.transition_modifiers import (
    MemoryEvidenceRef,
    MemorySourceKind,
    TransitionChannel,
    TransitionModifier,
    TransitionModifierSet,
    zero_transition_modifier_set,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def source_image():
    return build_system_image(
        image_id="image_0",
        revision=0,
        measures=(
            SystemMeasure(
                measure_id="mass",
                measure_type="mass",
                value=10.0,
                unit="kg",
                domain="physical",
                normalized_value=0.5,
            ),
        ),
        prestress_nodes=(
            PrestressNodeState(
                node_id="A",
                prestress=0.20,
                reserve=1.0,
            ),
            PrestressNodeState(
                node_id="B",
                prestress=0.10,
                reserve=1.0,
            ),
            PrestressNodeState(
                node_id="C",
                prestress=0.00,
                reserve=1.0,
            ),
        ),
        adaptive_connections=(
            AdaptiveConnectionState(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                stiffness=1.0,
                contractile_capacity=1.0,
                reflex_gain=1.0,
                fatigue=0.0,
                remodeling_bias=0.0,
            ),
            AdaptiveConnectionState(
                connection_id="B_C",
                source_node_id="B",
                target_node_id="C",
                stiffness=1.0,
                contractile_capacity=1.0,
                reflex_gain=1.0,
                fatigue=0.0,
                remodeling_bias=0.0,
            ),
        ),
        historical_traces=(
            HistoricalTrace(
                trace_id="old_trace",
                source_event_id="old_event",
                trace_type="history",
                magnitude=0.1,
                persistence=1.0,
                relation_count=1,
            ),
        ),
        context=SystemImageContext(
            context_id="ctx_0",
            timestamp_label="t0",
        ),
    )


@pytest.fixture
def event():
    return SystemEvent(
        event_id="event_1",
        event_type="combined_exposure",
        connection_exposures=(
            ConnectionExposure(
                exposure_id="exp_A_B",
                connection_id="A_B",
                load=0.8,
                strain=0.2,
                activation=0.7,
                damage=0.1,
                recovery=0.0,
            ),
        ),
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id="perturb_A",
                node_id="A",
                delta=-0.30,
            ),
        ),
        target_context=SystemImageContext(
            context_id="ctx_1",
            timestamp_label="t1",
        ),
    )


@pytest.fixture
def config():
    return SystemEvolutionConfig(
        adaptive_config=(
            AdaptiveConnectionConfig()
        ),
        prestress_config=(
            PrestressRedistributionConfig(
                propagation_steps=2,
                propagation_decay=0.5,
                reserve_modulation=False,
            )
        ),
        trace_type="evolution_trace",
        trace_persistence=0.9,
    )


@pytest.fixture
def memory_evidence():
    return MemoryEvidenceRef(
        source_kind=(
            MemorySourceKind.CONDITIONING_MEMORY
        ),
        source_id="memory_A",
        order_index=2,
        cue_id="cue_A",
        attractor_id="attractor_A",
        relation="conditioned_match",
    )


def transfer_modifier_set(
    *,
    value: float,
    evidence: MemoryEvidenceRef,
    target_id: str = "A_B",
    modifier_set_id: str = "transfer_modifiers",
):
    modifier = TransitionModifier(
        modifier_id=(
            f"{modifier_set_id}::modifier"
        ),
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        target_id=target_id,
        value=value,
        evidence=(
            evidence,
        ),
    )

    return TransitionModifierSet(
        modifier_set_id=modifier_set_id,
        modifiers=(
            modifier,
        ),
        source_memory_ids=(
            "memory_A",
        ),
    )


def unsupported_modifier_set(
    *,
    evidence: MemoryEvidenceRef,
):
    modifier = TransitionModifier(
        modifier_id="unsupported",
        channel=(
            TransitionChannel.CONNECTION_STIFFNESS_UPDATE
        ),
        target_id="A_B",
        value=0.25,
        evidence=(
            evidence,
        ),
    )

    return TransitionModifierSet(
        modifier_set_id="unsupported_set",
        modifiers=(
            modifier,
        ),
    )


def run(
    *,
    source_image,
    event,
    config,
    transition_modifiers=None,
    evolution_id="integration",
    target_image_id="target",
):
    return evolve_system(
        evolution_id=evolution_id,
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transition_modifiers
        ),
        target_image_id=(
            target_image_id
        ),
    )


# =============================================================================
# STRICT BACKWARD COMPATIBILITY
# =============================================================================


def test_legacy_and_explicit_none_are_identical(
    source_image,
    event,
    config,
):
    legacy = evolve_system(
        evolution_id="same",
        source_image=source_image,
        event=event,
        config=config,
        target_image_id="same_target",
    )

    explicit_none = evolve_system(
        evolution_id="same",
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=None,
        target_image_id="same_target",
    )

    assert explicit_none == legacy


def test_legacy_and_zero_modifier_set_are_identical(
    source_image,
    event,
    config,
):
    legacy = evolve_system(
        evolution_id="same",
        source_image=source_image,
        event=event,
        config=config,
        target_image_id="same_target",
    )

    zero = evolve_system(
        evolution_id="same",
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            zero_transition_modifier_set()
        ),
        target_image_id="same_target",
    )

    assert zero == legacy


def test_none_zero_and_legacy_all_equal(
    source_image,
    event,
    config,
):
    legacy = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    explicit_none = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=None,
    )

    zero = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            zero_transition_modifier_set()
        ),
    )

    assert (
        legacy
        ==
        explicit_none
        ==
        zero
    )


def test_zero_modifier_does_not_add_modifier_metadata(
    source_image,
    event,
    config,
):
    result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            zero_transition_modifier_set()
        ),
    )

    assert (
        "transition_modifiers_applied"
        not in result.metadata
    )

    assert (
        "transition_modifier_set_id"
        not in result.metadata
    )


def test_zero_modifier_preserves_full_system_image_signature(
    source_image,
    event,
    config,
):
    legacy = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    zero = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            zero_transition_modifier_set()
        ),
    )

    assert system_image_signature(
        zero.target_image
    ) == system_image_signature(
        legacy.target_image
    )


def test_zero_modifier_preserves_generated_trace(
    source_image,
    event,
    config,
):
    legacy = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    zero = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            zero_transition_modifier_set()
        ),
    )

    assert (
        zero.generated_trace
        ==
        legacy.generated_trace
    )


# =============================================================================
# ACTIVE PRESTRESS TRANSFER CHANNEL
# =============================================================================


def test_positive_transfer_modifier_changes_result(
    source_image,
    event,
    config,
    memory_evidence,
):
    modifiers = transfer_modifier_set(
        value=0.5,
        evidence=memory_evidence,
    )

    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=modifiers,
    )

    assert (
        modified.prestress_result
        !=
        baseline.prestress_result
    )


def test_negative_transfer_modifier_changes_result(
    source_image,
    event,
    config,
    memory_evidence,
):
    modifiers = transfer_modifier_set(
        value=-0.5,
        evidence=memory_evidence,
    )

    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=modifiers,
    )

    assert (
        modified.prestress_result
        !=
        baseline.prestress_result
    )


def test_positive_modifier_increases_downstream_B_response(
    source_image,
    event,
    config,
    memory_evidence,
):
    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    baseline_delta = abs(
        baseline.prestress_result
        .node_delta_by_id["B"]
    )

    modified_delta = abs(
        modified.prestress_result
        .node_delta_by_id["B"]
    )

    assert (
        modified_delta
        >
        baseline_delta
    )


def test_negative_modifier_reduces_downstream_B_response(
    source_image,
    event,
    config,
    memory_evidence,
):
    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=-0.5,
                evidence=memory_evidence,
            )
        ),
    )

    baseline_delta = abs(
        baseline.prestress_result
        .node_delta_by_id["B"]
    )

    modified_delta = abs(
        modified.prestress_result
        .node_delta_by_id["B"]
    )

    assert (
        modified_delta
        <
        baseline_delta
    )


def test_full_negative_modifier_blocks_A_B_transfer(
    source_image,
    event,
    config,
    memory_evidence,
):
    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=-1.0,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        modified.prestress_result
        .node_delta_by_id["B"]
        ==
        pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_full_negative_modifier_blocks_downstream_C_transfer_from_A(
    source_image,
    event,
    config,
    memory_evidence,
):
    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=-1.0,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        modified.prestress_result
        .node_delta_by_id["C"]
        ==
        pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


# =============================================================================
# PHYSICAL STATE / TOPOLOGY MUST REMAIN UNCHANGED
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        -0.75,
        0.50,
        1.00,
    ],
)
def test_transfer_modifier_does_not_change_adaptive_connection_states(
    source_image,
    event,
    config,
    memory_evidence,
    value,
):
    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=value,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        modified.target_image
        .adaptive_connections
        ==
        baseline.target_image
        .adaptive_connections
    )


@pytest.mark.parametrize(
    "value",
    [
        -0.75,
        0.50,
        1.00,
    ],
)
def test_transfer_modifier_does_not_change_connection_records(
    source_image,
    event,
    config,
    memory_evidence,
    value,
):
    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=value,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        modified.connection_records
        ==
        baseline.connection_records
    )


def test_transfer_modifier_does_not_change_connection_topology(
    source_image,
    event,
    config,
    memory_evidence,
):
    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    topology_before = tuple(
        (
            item.connection_id,
            item.source_node_id,
            item.target_node_id,
        )
        for item
        in source_image.adaptive_connections
    )

    topology_after = tuple(
        (
            item.connection_id,
            item.source_node_id,
            item.target_node_id,
        )
        for item
        in modified.target_image.adaptive_connections
    )

    assert (
        topology_after
        ==
        topology_before
    )


def test_transfer_modifier_does_not_mutate_source_image(
    source_image,
    event,
    config,
    memory_evidence,
):
    before = system_image_signature(
        source_image
    )

    run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    after = system_image_signature(
        source_image
    )

    assert after == before


# =============================================================================
# ACTIVE MODIFIER METADATA / PROVENANCE
# =============================================================================


def test_active_modifier_marks_result_metadata(
    source_image,
    event,
    config,
    memory_evidence,
):
    result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    assert result.metadata[
        "transition_modifiers_applied"
    ] is True


def test_active_modifier_records_modifier_set_id(
    source_image,
    event,
    config,
    memory_evidence,
):
    result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
                modifier_set_id=(
                    "memory_transition_A"
                ),
            )
        ),
    )

    assert result.metadata[
        "transition_modifier_set_id"
    ] == "memory_transition_A"


def test_active_modifier_records_channel(
    source_image,
    event,
    config,
    memory_evidence,
):
    result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    assert result.metadata[
        "transition_modifier_channels"
    ] == (
        "prestress_transfer",
    )


def test_active_modifier_records_memory_source(
    source_image,
    event,
    config,
    memory_evidence,
):
    result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    assert result.metadata[
        "transition_modifier_source_memory_ids"
    ] == (
        "memory_A",
    )


def test_modifier_application_is_not_reported_as_learning(
    source_image,
    event,
    config,
    memory_evidence,
):
    result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    assert result.metadata[
        "learning_applied"
    ] is False

    assert result.metadata[
        "memory_mutated"
    ] is False


def test_modifier_application_is_not_policy_selection(
    source_image,
    event,
    config,
    memory_evidence,
):
    result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    assert result.metadata[
        "action_selected"
    ] is False

    assert result.metadata[
        "policy_modified"
    ] is False


# =============================================================================
# INVALID INTEGRATION REQUESTS
# =============================================================================


def test_unknown_transfer_target_rejected(
    source_image,
    event,
    config,
    memory_evidence,
):
    modifiers = transfer_modifier_set(
        value=0.5,
        evidence=memory_evidence,
        target_id="UNKNOWN_CONNECTION",
    )

    with pytest.raises(
        SystemEvolutionError,
        match="unknown PRESTRESS_TRANSFER modifier target",
    ):
        run(
            source_image=source_image,
            event=event,
            config=config,
            transition_modifiers=(
                modifiers
            ),
        )


def test_unsupported_active_channel_rejected(
    source_image,
    event,
    config,
    memory_evidence,
):
    modifiers = (
        unsupported_modifier_set(
            evidence=memory_evidence
        )
    )

    with pytest.raises(
        SystemEvolutionError,
        match=(
            "active transition modifier channels "
            "are not yet supported"
        ),
    ):
        run(
            source_image=source_image,
            event=event,
            config=config,
            transition_modifiers=(
                modifiers
            ),
        )


def test_zero_unsupported_channel_does_not_change_legacy_path(
    source_image,
    event,
    config,
):
    zero_unsupported = TransitionModifier(
        modifier_id="zero_unsupported",
        channel=(
            TransitionChannel.CONNECTION_STIFFNESS_UPDATE
        ),
        target_id="A_B",
        value=0.0,
        evidence=(),
    )

    modifier_set = TransitionModifierSet(
        modifier_set_id=(
            "zero_unsupported_set"
        ),
        modifiers=(
            zero_unsupported,
        ),
    )

    legacy = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    zero_result = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            modifier_set
        ),
    )

    assert zero_result == legacy


# =============================================================================
# DETERMINISM
# =============================================================================


def test_active_modifier_path_is_deterministic(
    source_image,
    event,
    config,
    memory_evidence,
):
    modifiers = transfer_modifier_set(
        value=0.5,
        evidence=memory_evidence,
    )

    left = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            modifiers
        ),
        evolution_id="same",
        target_image_id="same_target",
    )

    right = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            modifiers
        ),
        evolution_id="same",
        target_image_id="same_target",
    )

    assert left == right


# =============================================================================
# SEQUENCE INTEGRATION
# =============================================================================


def test_sequence_accepts_transition_modifier_set(
    source_image,
    event,
    config,
    memory_evidence,
):
    modifiers = transfer_modifier_set(
        value=0.25,
        evidence=memory_evidence,
    )

    results = evolve_sequence(
        sequence_id="modified_sequence",
        source_image=source_image,
        events=(
            event,
            event,
        ),
        config=config,
        transition_modifiers=modifiers,
    )

    assert len(results) == 2


def test_sequence_applies_modifier_set_to_every_step(
    source_image,
    event,
    config,
    memory_evidence,
):
    modifiers = transfer_modifier_set(
        value=0.25,
        evidence=memory_evidence,
    )

    results = evolve_sequence(
        sequence_id="modified_sequence",
        source_image=source_image,
        events=(
            event,
            event,
        ),
        config=config,
        transition_modifiers=modifiers,
    )

    assert all(
        result.metadata[
            "transition_modifiers_applied"
        ]
        is True
        for result in results
    )


def test_zero_modifier_sequence_equals_legacy_sequence(
    source_image,
    event,
    config,
):
    legacy = evolve_sequence(
        sequence_id="same_sequence",
        source_image=source_image,
        events=(
            event,
            event,
        ),
        config=config,
    )

    zero = evolve_sequence(
        sequence_id="same_sequence",
        source_image=source_image,
        events=(
            event,
            event,
        ),
        config=config,
        transition_modifiers=(
            zero_transition_modifier_set()
        ),
    )

    assert zero == legacy


def test_modified_sequence_preserves_revision_chain(
    source_image,
    event,
    config,
    memory_evidence,
):
    results = evolve_sequence(
        sequence_id="modified_sequence",
        source_image=source_image,
        events=(
            event,
            event,
        ),
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.25,
                evidence=memory_evidence,
            )
        ),
    )

    assert tuple(
        result.target_image.revision
        for result in results
    ) == (
        1,
        2,
    )


def test_modified_sequence_remains_recursive(
    source_image,
    event,
    config,
    memory_evidence,
):
    results = evolve_sequence(
        sequence_id="modified_sequence",
        source_image=source_image,
        events=(
            event,
            event,
        ),
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.25,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        results[1].source_image
        ==
        results[0].target_image
    )


# =============================================================================
# ANTI-REDUCTION / ARCHITECTURAL BOUNDARIES
# =============================================================================


def test_transfer_modifier_changes_prestress_but_not_connection_evolution(
    source_image,
    event,
    config,
    memory_evidence,
):
    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.75,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        modified.connection_records
        ==
        baseline.connection_records
    )

    assert (
        modified.prestress_result
        !=
        baseline.prestress_result
    )


def test_transfer_modifier_does_not_replace_event_perturbation(
    source_image,
    event,
    config,
    memory_evidence,
):
    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.75,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        modified.prestress_result
        .node_delta_by_id["A"]
        ==
        pytest.approx(
            baseline.prestress_result
            .node_delta_by_id["A"]
        )
    )


def test_modifier_changes_generated_history_only_via_changed_transition(
    source_image,
    event,
    config,
    memory_evidence,
):
    baseline = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    modified = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.75,
                evidence=memory_evidence,
            )
        ),
    )

    assert (
        modified.generated_trace.source_event_id
        ==
        baseline.generated_trace.source_event_id
        ==
        event.event_id
    )

    assert (
        modified.generated_trace.magnitude
        !=
        baseline.generated_trace.magnitude
    )


# =============================================================================
# CENTRAL INTEGRATION REGRESSION GUARD
# =============================================================================


def test_system_evolution_transition_modifier_integration_holds_together(
    source_image,
    event,
    config,
    memory_evidence,
):
    """
    Central regression guard for first Memory-to-Transition integration.

    Established semantics:

        Gamma = 0
            -> exact legacy transition

        PRESTRESS_TRANSFER > 0
            -> stronger transmission through existing targeted path

        PRESTRESS_TRANSFER < 0
            -> weaker transmission through existing targeted path

    while:

        adaptive connection state remains unchanged relative to baseline;
        topology remains unchanged;
        source image remains immutable;
        memory is not mutated;
        no policy or action is selected.

    This is a local PRESTRESS_TRANSFER integration rule only.
    It is not a general implementation of Phi_H.
    """

    legacy = run(
        source_image=source_image,
        event=event,
        config=config,
    )

    zero = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            zero_transition_modifier_set()
        ),
    )

    positive = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=0.5,
                evidence=memory_evidence,
            )
        ),
    )

    negative = run(
        source_image=source_image,
        event=event,
        config=config,
        transition_modifiers=(
            transfer_modifier_set(
                value=-0.5,
                evidence=memory_evidence,
            )
        ),
    )

    assert zero == legacy

    assert (
        positive.target_image
        .adaptive_connections
        ==
        legacy.target_image
        .adaptive_connections
        ==
        negative.target_image
        .adaptive_connections
    )

    baseline_B = abs(
        legacy.prestress_result
        .node_delta_by_id["B"]
    )

    positive_B = abs(
        positive.prestress_result
        .node_delta_by_id["B"]
    )

    negative_B = abs(
        negative.prestress_result
        .node_delta_by_id["B"]
    )

    assert (
        negative_B
        <
        baseline_B
        <
        positive_B
    )

    assert (
        positive.metadata[
            "memory_mutated"
        ]
        is False
    )

    assert (
        positive.metadata[
            "learning_applied"
        ]
        is False
    )

    assert (
        positive.metadata[
            "action_selected"
        ]
        is False
    )

    assert (
        positive.metadata[
            "policy_modified"
        ]
        is False
    )
