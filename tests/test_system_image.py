"""
Tests for roif.history.system_image
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.adaptive_connection import (
    AdaptiveConnectionState,
)
from roif.history.prestress_redistribution import (
    PrestressNodeState,
)
from roif.history.system_image import (
    HistoricalTrace,
    SCHEMA_VERSION,
    SystemImage,
    SystemImageContext,
    SystemImageError,
    SystemImageSummary,
    SystemMeasure,
    adaptive_connection_by_id,
    build_system_image,
    measure_by_id,
    normalized_measure_vector,
    prestress_node_by_id,
    revise_system_image,
    summarize_system_image_layers,
    system_image_is_policy_free,
    system_image_signature,
    trace_by_id,
)


@pytest.fixture
def measures():
    return (
        SystemMeasure(
            measure_id="mass",
            measure_type="mass",
            value=90.0,
            unit="kg",
            domain="physical",
            normalized_value=0.60,
        ),
        SystemMeasure(
            measure_id="volume",
            measure_type="volume",
            value=0.085,
            unit="m3",
            domain="physical",
            normalized_value=0.40,
        ),
        SystemMeasure(
            measure_id="current",
            measure_type="electric_current",
            value=0.002,
            unit="A",
            domain="electrical",
            normalized_value=None,
        ),
    )


@pytest.fixture
def prestress_nodes():
    return (
        PrestressNodeState(
            node_id="A",
            prestress=0.20,
            reserve=1.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=-0.10,
            reserve=0.8,
        ),
    )


@pytest.fixture
def adaptive_connections():
    return (
        AdaptiveConnectionState(
            connection_id="A_B",
            source_node_id="A",
            target_node_id="B",
            stiffness=2.0,
            contractile_capacity=3.0,
            reflex_gain=1.5,
            fatigue=0.10,
            remodeling_bias=0.0,
        ),
    )


@pytest.fixture
def traces():
    return (
        HistoricalTrace(
            trace_id="trace_1",
            source_event_id="event_1",
            trace_type="structural",
            magnitude=0.4,
            persistence=0.75,
            relation_count=2,
        ),
        HistoricalTrace(
            trace_id="trace_2",
            source_event_id="event_2",
            trace_type="adaptive",
            magnitude=0.2,
            persistence=0.50,
            relation_count=1,
        ),
    )


@pytest.fixture
def context():
    return SystemImageContext(
        context_id="ctx_0",
        timestamp_label="t0",
    )


@pytest.fixture
def image(
    measures,
    prestress_nodes,
    adaptive_connections,
    traces,
    context,
):
    return build_system_image(
        image_id="image_0",
        revision=0,
        measures=measures,
        prestress_nodes=prestress_nodes,
        adaptive_connections=adaptive_connections,
        historical_traces=traces,
        context=context,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "system_image_v1"


def test_measure_constructs(measures):
    assert measures[0].measure_id == "mass"
    assert measures[0].unit == "kg"


def test_context_constructs(context):
    assert context.context_id == "ctx_0"


def test_trace_constructs(traces):
    assert traces[0].trace_id == "trace_1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("measure_id", ""),
        ("measure_type", ""),
        ("unit", ""),
        ("domain", ""),
    ],
)
def test_empty_measure_fields_rejected(field, value):
    kwargs = dict(
        measure_id="m",
        measure_type="mass",
        value=1.0,
        unit="kg",
        domain="physical",
    )
    kwargs[field] = value

    with pytest.raises(SystemImageError):
        SystemMeasure(**kwargs)


def test_negative_trace_magnitude_rejected():
    with pytest.raises(SystemImageError):
        HistoricalTrace(
            trace_id="t",
            source_event_id="e",
            trace_type="x",
            magnitude=-0.1,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_trace_persistence_rejected(value):
    with pytest.raises(SystemImageError):
        HistoricalTrace(
            trace_id="t",
            source_event_id="e",
            trace_type="x",
            magnitude=0.1,
            persistence=value,
        )


def test_negative_trace_relation_count_rejected():
    with pytest.raises(SystemImageError):
        HistoricalTrace(
            trace_id="t",
            source_event_id="e",
            trace_type="x",
            magnitude=0.1,
            relation_count=-1,
        )


def test_negative_revision_rejected(
    measures,
    prestress_nodes,
    adaptive_connections,
    traces,
    context,
):
    with pytest.raises(SystemImageError):
        build_system_image(
            image_id="bad",
            revision=-1,
            measures=measures,
            prestress_nodes=prestress_nodes,
            adaptive_connections=adaptive_connections,
            historical_traces=traces,
            context=context,
        )


def test_duplicate_measure_id_rejected(
    prestress_nodes,
    adaptive_connections,
    traces,
    context,
):
    measures = (
        SystemMeasure(
            measure_id="x",
            measure_type="mass",
            value=1.0,
            unit="kg",
        ),
        SystemMeasure(
            measure_id="x",
            measure_type="volume",
            value=2.0,
            unit="m3",
        ),
    )

    with pytest.raises(SystemImageError):
        build_system_image(
            image_id="bad",
            revision=0,
            measures=measures,
            prestress_nodes=prestress_nodes,
            adaptive_connections=adaptive_connections,
            historical_traces=traces,
            context=context,
        )


def test_duplicate_prestress_node_id_rejected(
    measures,
    adaptive_connections,
    traces,
    context,
):
    nodes = (
        PrestressNodeState(
            node_id="A",
            prestress=0.0,
        ),
        PrestressNodeState(
            node_id="A",
            prestress=0.1,
        ),
    )

    with pytest.raises(SystemImageError):
        build_system_image(
            image_id="bad",
            revision=0,
            measures=measures,
            prestress_nodes=nodes,
            adaptive_connections=adaptive_connections,
            historical_traces=traces,
            context=context,
        )


def test_duplicate_connection_id_rejected(
    measures,
    prestress_nodes,
    traces,
    context,
):
    connections = (
        AdaptiveConnectionState(
            connection_id="c",
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
        ),
        AdaptiveConnectionState(
            connection_id="c",
            source_node_id="B",
            target_node_id="A",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
        ),
    )

    with pytest.raises(SystemImageError):
        build_system_image(
            image_id="bad",
            revision=0,
            measures=measures,
            prestress_nodes=prestress_nodes,
            adaptive_connections=connections,
            historical_traces=traces,
            context=context,
        )


def test_duplicate_trace_id_rejected(
    measures,
    prestress_nodes,
    adaptive_connections,
    context,
):
    traces = (
        HistoricalTrace(
            trace_id="t",
            source_event_id="e1",
            trace_type="x",
            magnitude=0.1,
        ),
        HistoricalTrace(
            trace_id="t",
            source_event_id="e2",
            trace_type="y",
            magnitude=0.2,
        ),
    )

    with pytest.raises(SystemImageError):
        build_system_image(
            image_id="bad",
            revision=0,
            measures=measures,
            prestress_nodes=prestress_nodes,
            adaptive_connections=adaptive_connections,
            historical_traces=traces,
            context=context,
        )


def test_build_returns_expected_type(image):
    assert isinstance(
        image,
        SystemImage,
    )


def test_build_preserves_revision(image):
    assert image.revision == 0


def test_measures_are_tuple(image):
    assert isinstance(
        image.measures,
        tuple,
    )


def test_prestress_nodes_are_tuple(image):
    assert isinstance(
        image.prestress_nodes,
        tuple,
    )


def test_adaptive_connections_are_tuple(image):
    assert isinstance(
        image.adaptive_connections,
        tuple,
    )


def test_historical_traces_are_tuple(image):
    assert isinstance(
        image.historical_traces,
        tuple,
    )


def test_summary_type(image):
    assert isinstance(
        image.summary,
        SystemImageSummary,
    )


def test_summary_measure_count(image):
    assert image.summary.measure_count == 3


def test_summary_prestress_node_count(image):
    assert image.summary.prestress_node_count == 2


def test_summary_connection_count(image):
    assert image.summary.adaptive_connection_count == 1


def test_summary_trace_count(image):
    assert image.summary.trace_count == 2


def test_normalized_measure_rms_uses_only_explicit_normalized_values(image):
    expected = (
        ((0.60 ** 2 + 0.40 ** 2) / 2.0) ** 0.5
    )

    assert image.summary.normalized_measure_rms == pytest.approx(
        expected
    )


def test_prestress_rms_matches_nodes(image):
    expected = (
        ((0.20 ** 2 + (-0.10) ** 2) / 2.0) ** 0.5
    )

    assert image.summary.prestress_rms == pytest.approx(
        expected
    )


def test_trace_load_matches_persistence_weighting(image):
    expected = (
        0.4 * 0.75
        + 0.2 * 0.50
    )

    assert image.summary.trace_load == pytest.approx(
        expected
    )


def test_trace_relation_total_matches(image):
    assert image.summary.trace_relation_total == 3


def test_mean_effective_transfer_gain_present(image):
    assert (
        image.summary.mean_effective_transfer_gain
        is not None
    )


def test_heterogeneous_raw_measure_values_not_collapsed(image):
    assert image.metadata[
        "heterogeneous_measures_collapsed"
    ] is False


def test_normalized_measure_vector_contains_only_normalized(image):
    vector = normalized_measure_vector(
        image
    )

    assert vector == (
        ("mass", 0.60),
        ("volume", 0.40),
    )


def test_measure_lookup_found(image):
    measure = measure_by_id(
        image,
        "mass",
    )

    assert measure is not None
    assert measure.value == pytest.approx(90.0)


def test_measure_lookup_missing(image):
    assert measure_by_id(
        image,
        "missing",
    ) is None


def test_prestress_lookup_found(image):
    node = prestress_node_by_id(
        image,
        "A",
    )

    assert node is not None
    assert node.prestress == pytest.approx(0.20)


def test_prestress_lookup_missing(image):
    assert prestress_node_by_id(
        image,
        "missing",
    ) is None


def test_connection_lookup_found(image):
    connection = adaptive_connection_by_id(
        image,
        "A_B",
    )

    assert connection is not None
    assert connection.stiffness == pytest.approx(2.0)


def test_connection_lookup_missing(image):
    assert adaptive_connection_by_id(
        image,
        "missing",
    ) is None


def test_trace_lookup_found(image):
    trace = trace_by_id(
        image,
        "trace_1",
    )

    assert trace is not None
    assert trace.magnitude == pytest.approx(0.4)


def test_trace_lookup_missing(image):
    assert trace_by_id(
        image,
        "missing",
    ) is None


def test_revise_increments_revision(image):
    revised = revise_system_image(
        source=image,
        target_image_id="image_1",
    )

    assert revised.revision == 1


def test_revise_preserves_omitted_layers(image):
    revised = revise_system_image(
        source=image,
        target_image_id="image_1",
    )

    assert revised.measures == image.measures
    assert revised.prestress_nodes == image.prestress_nodes
    assert revised.adaptive_connections == image.adaptive_connections
    assert revised.historical_traces == image.historical_traces


def test_revise_can_replace_measure_layer(image):
    new_measures = (
        SystemMeasure(
            measure_id="mass",
            measure_type="mass",
            value=88.0,
            unit="kg",
            domain="physical",
            normalized_value=0.55,
        ),
    )

    revised = revise_system_image(
        source=image,
        target_image_id="image_1",
        measures=new_measures,
    )

    assert len(revised.measures) == 1
    assert revised.measures[0].value == pytest.approx(88.0)


def test_revise_can_append_trace_by_replacing_trace_layer(image):
    new_trace = HistoricalTrace(
        trace_id="trace_3",
        source_event_id="event_3",
        trace_type="structural",
        magnitude=0.3,
        persistence=1.0,
        relation_count=2,
    )

    revised = revise_system_image(
        source=image,
        target_image_id="image_1",
        historical_traces=(
            *image.historical_traces,
            new_trace,
        ),
    )

    assert revised.summary.trace_count == 3


def test_revise_does_not_mutate_source(image):
    before = system_image_signature(
        image
    )

    revise_system_image(
        source=image,
        target_image_id="image_1",
    )

    after = system_image_signature(
        image
    )

    assert after == before


def test_summary_helper_matches_image_summary(
    measures,
    prestress_nodes,
    adaptive_connections,
    traces,
    image,
):
    summary = summarize_system_image_layers(
        measures=measures,
        prestress_nodes=prestress_nodes,
        adaptive_connections=adaptive_connections,
        historical_traces=traces,
    )

    assert summary == image.summary


def test_empty_layers_supported(context):
    image = build_system_image(
        image_id="empty_layers",
        revision=0,
        measures=(),
        prestress_nodes=(),
        adaptive_connections=(),
        historical_traces=(),
        context=context,
    )

    assert image.summary.measure_count == 0
    assert image.summary.prestress_node_count == 0
    assert image.summary.adaptive_connection_count == 0
    assert image.summary.trace_count == 0


def test_empty_normalized_measure_rms_is_none(context):
    image = build_system_image(
        image_id="empty_norm",
        revision=0,
        measures=(
            SystemMeasure(
                measure_id="mass",
                measure_type="mass",
                value=1.0,
                unit="kg",
            ),
        ),
        prestress_nodes=(),
        adaptive_connections=(),
        historical_traces=(),
        context=context,
    )

    assert image.summary.normalized_measure_rms is None


def test_empty_transfer_gain_mean_is_none(context):
    image = build_system_image(
        image_id="empty_connections",
        revision=0,
        measures=(),
        prestress_nodes=(),
        adaptive_connections=(),
        historical_traces=(),
        context=context,
    )

    assert image.summary.mean_effective_transfer_gain is None


def test_result_metadata_schema(image):
    assert image.metadata[
        "schema_version"
    ] == SCHEMA_VERSION


def test_result_declares_no_memory_mutation(image):
    assert image.metadata[
        "memory_mutated"
    ] is False


def test_result_declares_no_learning(image):
    assert image.metadata[
        "learning_applied"
    ] is False


def test_result_declares_no_topology_mutation(image):
    assert image.metadata[
        "topology_modified"
    ] is False


def test_result_declares_no_action(image):
    assert image.metadata[
        "action_selected"
    ] is False


def test_result_declares_no_policy_modified(image):
    assert image.metadata[
        "policy_modified"
    ] is False


def test_result_declares_no_diagnosis(image):
    assert image.metadata[
        "diagnosis_generated"
    ] is False


def test_result_declares_no_biological_truth(image):
    assert image.metadata[
        "biological_truth_claimed"
    ] is False


def test_result_declares_no_causal_truth(image):
    assert image.metadata[
        "causal_truth_inferred"
    ] is False


def test_policy_free_helper(image):
    assert system_image_is_policy_free(
        image
    ) is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SystemMeasure(
            measure_id="m",
            measure_type="mass",
            value=1.0,
            unit="kg",
        ),
        lambda: HistoricalTrace(
            trace_id="t",
            source_event_id="e",
            trace_type="x",
            magnitude=0.1,
        ),
        lambda: SystemImageContext(
            context_id="c",
            timestamp_label="t0",
        ),
        lambda: SystemImageSummary(
            measure_count=0,
            prestress_node_count=0,
            adaptive_connection_count=0,
            trace_count=0,
            normalized_measure_rms=None,
            prestress_rms=0.0,
            mean_effective_transfer_gain=None,
            trace_load=0.0,
            trace_relation_total=0,
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


def test_image_is_frozen(image):
    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        image.image_id = "changed"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SystemMeasure(
            measure_id="m",
            measure_type="mass",
            value=1.0,
            unit="kg",
            metadata={"x": 1},
        ),
        lambda: HistoricalTrace(
            trace_id="t",
            source_event_id="e",
            trace_type="x",
            magnitude=0.1,
            metadata={"x": 1},
        ),
        lambda: SystemImageContext(
            context_id="c",
            timestamp_label="t0",
            metadata={"x": 1},
        ),
    ],
)
def test_primary_metadata_is_read_only(factory):
    obj = factory()

    with pytest.raises(TypeError):
        obj.metadata["x"] = 2


def test_image_metadata_is_read_only(image):
    with pytest.raises(TypeError):
        image.metadata[
            "schema_version"
        ] = "changed"


def test_summary_metadata_is_read_only(image):
    with pytest.raises(TypeError):
        image.summary.metadata[
            "summary_mode"
        ] = "changed"


def test_build_is_deterministic(
    measures,
    prestress_nodes,
    adaptive_connections,
    traces,
    context,
):
    left = build_system_image(
        image_id="same",
        revision=0,
        measures=measures,
        prestress_nodes=prestress_nodes,
        adaptive_connections=adaptive_connections,
        historical_traces=traces,
        context=context,
    )

    right = build_system_image(
        image_id="same",
        revision=0,
        measures=measures,
        prestress_nodes=prestress_nodes,
        adaptive_connections=adaptive_connections,
        historical_traces=traces,
        context=context,
    )

    assert left == right


def test_revision_is_deterministic(image):
    left = revise_system_image(
        source=image,
        target_image_id="next",
    )

    right = revise_system_image(
        source=image,
        target_image_id="next",
    )

    assert left == right


def test_signature_is_deterministic(image):
    assert system_image_signature(
        image
    ) == system_image_signature(
        image
    )


def test_normalized_vector_is_deterministic(image):
    assert normalized_measure_vector(
        image
    ) == normalized_measure_vector(
        image
    )


def test_reporting_helpers_are_deterministic(image):
    assert measure_by_id(
        image,
        "mass",
    ) == measure_by_id(
        image,
        "mass",
    )

    assert prestress_node_by_id(
        image,
        "A",
    ) == prestress_node_by_id(
        image,
        "A",
    )

    assert adaptive_connection_by_id(
        image,
        "A_B",
    ) == adaptive_connection_by_id(
        image,
        "A_B",
    )

    assert trace_by_id(
        image,
        "trace_1",
    ) == trace_by_id(
        image,
        "trace_1",
    )
