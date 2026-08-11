"""
Tests for roif.history.system_evolution
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

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
    ConnectionEvolutionRecord,
    SCHEMA_VERSION,
    SystemEvent,
    SystemEvolutionConfig,
    SystemEvolutionError,
    SystemEvolutionResult,
    SystemEvolutionSummary,
    evolve_sequence,
    evolve_system,
    final_system_image,
    system_evolution_is_policy_free,
    system_evolution_signature,
)
from roif.history.system_image import (
    HistoricalTrace,
    SystemImageContext,
    SystemMeasure,
    build_system_image,
    system_image_signature,
)


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
        adaptive_config=AdaptiveConnectionConfig(),
        prestress_config=PrestressRedistributionConfig(
            propagation_steps=2,
            propagation_decay=0.5,
            reserve_modulation=False,
        ),
        trace_type="evolution_trace",
        trace_persistence=0.9,
    )


@pytest.fixture
def result(source_image, event, config):
    return evolve_system(
        evolution_id="evo_1",
        source_image=source_image,
        event=event,
        config=config,
        target_image_id="image_1",
    )


def test_schema_version():
    assert SCHEMA_VERSION == "system_evolution_v1"


def test_event_constructs(event):
    assert event.event_id == "event_1"


def test_default_config_constructs():
    config = SystemEvolutionConfig()
    assert config.trace_persistence == pytest.approx(1.0)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_event_id_rejected(value):
    with pytest.raises(SystemEvolutionError):
        SystemEvent(
            event_id=value,
            event_type="x",
        )


def test_invalid_trace_persistence_rejected():
    with pytest.raises(SystemEvolutionError):
        SystemEvolutionConfig(
            trace_persistence=1.1,
        )


def test_result_type(result):
    assert isinstance(
        result,
        SystemEvolutionResult,
    )


def test_summary_type(result):
    assert isinstance(
        result.summary,
        SystemEvolutionSummary,
    )


def test_target_revision_incremented(result):
    assert result.target_image.revision == 1


def test_source_revision_preserved(result):
    assert result.source_image.revision == 0


def test_target_image_id_preserved(result):
    assert result.target_image.image_id == "image_1"


def test_target_context_updated(result):
    assert result.target_image.context.context_id == "ctx_1"


def test_connection_record_created(result):
    assert len(result.connection_records) == 1
    assert isinstance(
        result.connection_records[0],
        ConnectionEvolutionRecord,
    )


def test_exposed_connection_changes(result):
    assert (
        result.connection_records[0].transfer_gain_delta
        != pytest.approx(0.0)
    )


def test_unexposed_connection_preserved(source_image, result):
    source = {
        c.connection_id: c
        for c in source_image.adaptive_connections
    }
    target = {
        c.connection_id: c
        for c in result.target_image.adaptive_connections
    }

    assert target["B_C"] == source["B_C"]


def test_prestress_result_created(result):
    assert (
        result.prestress_result.redistribution_id
        == "evo_1::prestress"
    )


def test_direct_prestress_node_changes(result):
    delta = result.prestress_result.node_delta_by_id["A"]
    assert delta < 0.0


def test_downstream_prestress_can_change(result):
    assert (
        result.prestress_result.node_delta_by_id["B"]
        != pytest.approx(0.0)
    )


def test_generated_trace_created(result):
    assert result.generated_trace.trace_id == "evo_1::trace"


def test_generated_trace_references_event(result):
    assert result.generated_trace.source_event_id == "event_1"


def test_generated_trace_type_from_config(result):
    assert result.generated_trace.trace_type == "evolution_trace"


def test_generated_trace_persistence_from_config(result):
    assert result.generated_trace.persistence == pytest.approx(0.9)


def test_target_image_contains_old_and_new_trace(result):
    ids = tuple(
        trace.trace_id
        for trace in result.target_image.historical_traces
    )
    assert ids == (
        "old_trace",
        "evo_1::trace",
    )


def test_trace_magnitude_matches_summary(result):
    assert result.generated_trace.magnitude == pytest.approx(
        result.summary.trace_magnitude
    )


def test_connection_change_norm_nonnegative(result):
    assert result.summary.connection_change_norm >= 0.0


def test_prestress_change_norm_positive(result):
    assert result.summary.prestress_change_norm > 0.0


def test_changed_connection_count(result):
    assert result.summary.changed_connection_count >= 1


def test_changed_prestress_node_count(result):
    assert result.summary.changed_prestress_node_count >= 1


def test_adaptive_update_count(result):
    assert result.summary.adaptive_update_count == 1


def test_replacement_measures_supported(source_image, config):
    event = SystemEvent(
        event_id="measure_event",
        event_type="measurement_update",
        replacement_measures=(
            SystemMeasure(
                measure_id="mass",
                measure_type="mass",
                value=9.5,
                unit="kg",
                normalized_value=0.45,
            ),
        ),
    )

    result = evolve_system(
        evolution_id="measure_evo",
        source_image=source_image,
        event=event,
        config=config,
    )

    assert result.target_image.measures[0].value == pytest.approx(9.5)


def test_measures_preserved_when_not_replaced(source_image, event, config):
    result = evolve_system(
        evolution_id="preserve_measures",
        source_image=source_image,
        event=event,
        config=config,
    )

    assert result.target_image.measures == source_image.measures


def test_source_image_not_mutated(source_image, event, config):
    before = system_image_signature(
        source_image
    )

    evolve_system(
        evolution_id="immutability",
        source_image=source_image,
        event=event,
        config=config,
    )

    after = system_image_signature(
        source_image
    )

    assert after == before


def test_unknown_connection_exposure_rejected(source_image):
    event = SystemEvent(
        event_id="bad_event",
        event_type="bad",
        connection_exposures=(
            ConnectionExposure(
                exposure_id="bad",
                connection_id="missing",
            ),
        ),
    )

    with pytest.raises(SystemEvolutionError):
        evolve_system(
            evolution_id="bad_evo",
            source_image=source_image,
            event=event,
        )


def test_sequence_requires_events(source_image):
    with pytest.raises(SystemEvolutionError):
        evolve_sequence(
            sequence_id="empty",
            source_image=source_image,
            events=(),
        )


def test_sequence_returns_tuple(source_image, event, config):
    results = evolve_sequence(
        sequence_id="seq",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    assert isinstance(results, tuple)
    assert len(results) == 2


def test_sequence_revision_chain(source_image, event, config):
    results = evolve_sequence(
        sequence_id="seq",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    assert tuple(
        r.target_image.revision
        for r in results
    ) == (1, 2)


def test_sequence_second_source_is_first_target(
    source_image,
    event,
    config,
):
    results = evolve_sequence(
        sequence_id="seq",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    assert (
        results[1].source_image
        == results[0].target_image
    )


def test_sequence_accumulates_traces(source_image, event, config):
    results = evolve_sequence(
        sequence_id="seq",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    assert len(
        results[-1].target_image.historical_traces
    ) == 3


def test_final_system_image_helper(source_image, event, config):
    results = evolve_sequence(
        sequence_id="seq",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    assert final_system_image(
        results
    ) == results[-1].target_image


def test_final_system_image_rejects_empty():
    with pytest.raises(SystemEvolutionError):
        final_system_image(())


def test_result_metadata_schema(result):
    assert result.metadata[
        "schema_version"
    ] == SCHEMA_VERSION


def test_result_declares_no_memory_mutation(result):
    assert result.metadata[
        "memory_mutated"
    ] is False


def test_result_declares_no_learning(result):
    assert result.metadata[
        "learning_applied"
    ] is False


def test_result_declares_no_topology_mutation(result):
    assert result.metadata[
        "topology_modified"
    ] is False


def test_result_declares_no_action(result):
    assert result.metadata[
        "action_selected"
    ] is False


def test_result_declares_no_policy_modified(result):
    assert result.metadata[
        "policy_modified"
    ] is False


def test_result_declares_no_diagnosis(result):
    assert result.metadata[
        "diagnosis_generated"
    ] is False


def test_result_declares_no_biological_truth(result):
    assert result.metadata[
        "biological_truth_claimed"
    ] is False


def test_result_declares_no_causal_truth(result):
    assert result.metadata[
        "causal_truth_inferred"
    ] is False


def test_policy_free_helper(result):
    assert system_evolution_is_policy_free(
        result
    ) is True


def test_connection_record_metadata_safe(result):
    record = result.connection_records[0]

    assert record.metadata[
        "learning_applied"
    ] is False

    assert record.metadata[
        "topology_modified"
    ] is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SystemEvent(
            event_id="e",
            event_type="x",
        ),
        lambda: SystemEvolutionConfig(),
        lambda: SystemEvolutionSummary(
            adaptive_update_count=0,
            prestress_contribution_count=0,
            changed_connection_count=0,
            changed_prestress_node_count=0,
            connection_change_norm=0.0,
            prestress_change_norm=0.0,
            trace_magnitude=0.0,
        ),
    ],
)
def test_primary_objects_are_frozen(factory):
    obj = factory()

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        obj.metadata = {}


def test_result_is_frozen(result):
    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        result.evolution_id = "changed"


def test_connection_record_is_frozen(result):
    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        result.connection_records[
            0
        ].transfer_gain_delta = 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SystemEvent(
            event_id="e",
            event_type="x",
            metadata={"x": 1},
        ),
        lambda: SystemEvolutionConfig(
            metadata={"x": 1},
        ),
        lambda: SystemEvolutionSummary(
            adaptive_update_count=0,
            prestress_contribution_count=0,
            changed_connection_count=0,
            changed_prestress_node_count=0,
            connection_change_norm=0.0,
            prestress_change_norm=0.0,
            trace_magnitude=0.0,
            metadata={"x": 1},
        ),
    ],
)
def test_primary_metadata_is_read_only(factory):
    obj = factory()

    with pytest.raises(TypeError):
        obj.metadata["x"] = 2


def test_result_metadata_is_read_only(result):
    with pytest.raises(TypeError):
        result.metadata[
            "schema_version"
        ] = "changed"


def test_summary_metadata_is_read_only(result):
    with pytest.raises(TypeError):
        result.summary.metadata[
            "summary_mode"
        ] = "changed"


def test_record_metadata_is_read_only(result):
    with pytest.raises(TypeError):
        result.connection_records[
            0
        ].metadata[
            "learning_applied"
        ] = True


def test_evolution_is_deterministic(source_image, event, config):
    left = evolve_system(
        evolution_id="same",
        source_image=source_image,
        event=event,
        config=config,
        target_image_id="next",
    )

    right = evolve_system(
        evolution_id="same",
        source_image=source_image,
        event=event,
        config=config,
        target_image_id="next",
    )

    assert left == right


def test_sequence_is_deterministic(source_image, event, config):
    left = evolve_sequence(
        sequence_id="seq_same",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    right = evolve_sequence(
        sequence_id="seq_same",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    assert left == right


def test_signature_is_deterministic(result):
    assert system_evolution_signature(
        result
    ) == system_evolution_signature(
        result
    )


def test_reporting_helpers_are_deterministic(
    source_image,
    event,
    config,
):
    results = evolve_sequence(
        sequence_id="reporting",
        source_image=source_image,
        events=(event, event),
        config=config,
    )

    assert final_system_image(
        results
    ) == final_system_image(
        results
    )

    assert system_evolution_signature(
        results[0]
    ) == system_evolution_signature(
        results[0]
    )
