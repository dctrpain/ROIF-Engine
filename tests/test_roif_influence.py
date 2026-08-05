"""
Tests for roif.roif_influence.

The suite fixes the executable influence-layer contract:

- immutable runtime context and values;
- temporal activity of influence planes;
- applicability rules, delays, conditions, and thresholds;
- scalar and vector operator algebra;
- state-dependent gains;
- operator composition;
- direct transformation of FunctionalChannel fields;
- multiplicative factor products;
- vector aggregation;
- immutable lookup helpers.

The module under test is a research-prototype computation layer. These tests
do not assume a clinical diagnosis or treatment recommendation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
import math

import pytest

from roif.roif_entities import (
    ActivationAvailability,
    CapacityState,
    CompositionKind,
    FunctionalChannel,
    FunctionalRole,
    GeometryState,
    InfluenceOperator,
    InfluencePlane,
    OperatorComposition,
    OperatorKind,
    PlaneKind,
    TemporalMode,
    Vector3,
)
from roif.roif_influence import (
    ChannelInfluenceResult,
    CompositionApplication,
    InfluenceContext,
    InfluenceValue,
    OperatorApplication,
    ROIFInfluenceError,
    apply_operator,
    apply_operator_sequence_to_channel,
    apply_operator_to_channel,
    channels_vector_sum,
    compose_operators,
    effective_operator_gain,
    influence_factor_product,
    operator_is_applicable,
    operators_by_id,
    plane_activity,
    planes_by_id,
    vector_sum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plane(
    *,
    plane_id: str = "mechanical",
    temporal_mode: TemporalMode = TemporalMode.STATIC,
    active: bool = True,
    intensity: float = 1.0,
    timescale: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> InfluencePlane:
    return InfluencePlane(
        plane_id=plane_id,
        name=f"{plane_id} plane",
        kind=PlaneKind.MECHANICAL,
        temporal_mode=temporal_mode,
        active=active,
        intensity=intensity,
        timescale=timescale,
        metadata=metadata or {},
    )


def make_operator(
    *,
    operator_id: str = "operator",
    plane_id: str = "mechanical",
    kind: OperatorKind = OperatorKind.ADD,
    composition: CompositionKind = CompositionKind.MULTIPLICATIVE,
    gain: float = 1.0,
    threshold: float | None = None,
    delay: float = 0.0,
    direction: Vector3 | None = None,
    active: bool = True,
    condition_key: str | None = None,
    metadata: dict[str, object] | None = None,
) -> InfluenceOperator:
    return InfluenceOperator(
        operator_id=operator_id,
        name=operator_id,
        plane_id=plane_id,
        kind=kind,
        composition=composition,
        gain=gain,
        threshold=threshold,
        delay=delay,
        direction=direction or Vector3(),
        active=active,
        condition_key=condition_key,
        metadata=metadata or {},
    )


def make_context(
    *,
    time: float = 0.0,
    delta_time: float = 1.0,
    scalar_state: dict[str, float] | None = None,
    conditions: dict[str, bool] | None = None,
    events: set[str] | None = None,
) -> InfluenceContext:
    return InfluenceContext(
        time=time,
        delta_time=delta_time,
        scalar_state=scalar_state or {},
        conditions=conditions or {},
        events=frozenset(events or set()),
    )


def make_channel() -> FunctionalChannel:
    return FunctionalChannel(
        channel_id="glute_extension",
        entity_id="glute_max",
        name="Gluteus maximus extension",
        role=FunctionalRole.PRIME_MOVER,
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(
            capacity=10.0,
            load=2.0,
        ),
        geometry=GeometryState(),
        activation=ActivationAvailability(),
        baseline_output=1.0,
        history_factor=1.0,
    )


# ---------------------------------------------------------------------------
# InfluenceContext and InfluenceValue
# ---------------------------------------------------------------------------


def test_context_creation_and_accessors() -> None:
    context = InfluenceContext(
        time=2.0,
        delta_time=0.5,
        task_id="gait",
        scalar_state={"load": 3.0},
        vector_state={"force": Vector3(1.0, 2.0, 3.0)},
        conditions={"active": True},
        events=frozenset({"heel_strike"}),
        metadata={"phase": "stance"},
    )

    assert context.scalar("load") == pytest.approx(3.0)
    assert context.scalar("missing", 7.0) == pytest.approx(7.0)
    assert context.vector("force") == Vector3(1.0, 2.0, 3.0)
    assert context.vector("missing") == Vector3()
    assert context.condition("active") is True
    assert context.condition("missing") is False
    assert isinstance(context.scalar_state, MappingProxyType)
    assert isinstance(context.vector_state, MappingProxyType)
    assert isinstance(context.conditions, MappingProxyType)
    assert isinstance(context.metadata, MappingProxyType)


def test_context_is_immutable() -> None:
    context = InfluenceContext()

    with pytest.raises(FrozenInstanceError):
        context.time = 10.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("time", -0.1),
        ("delta_time", 0.0),
        ("delta_time", -1.0),
    ],
)
def test_context_rejects_invalid_time_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ROIFInfluenceError):
        InfluenceContext(**{field_name: value})


def test_context_rejects_non_vector_state_value() -> None:
    with pytest.raises(ROIFInfluenceError):
        InfluenceContext(
            vector_state={"force": 1.0},  # type: ignore[arg-type]
        )


def test_influence_value_creation() -> None:
    value = InfluenceValue(
        scalar=2.0,
        vector=Vector3(1.0, 0.0, 0.0),
        metadata={"unit": "relative"},
    )

    assert value.scalar == pytest.approx(2.0)
    assert isinstance(value.metadata, MappingProxyType)


def test_influence_value_rejects_non_vector() -> None:
    with pytest.raises(ROIFInfluenceError):
        InfluenceValue(
            scalar=1.0,
            vector=1.0,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Plane activity
# ---------------------------------------------------------------------------


def test_inactive_plane_has_zero_activity() -> None:
    plane = make_plane(active=False)

    assert plane_activity(plane, make_context()) == pytest.approx(0.0)


def test_static_plane_activity_equals_intensity() -> None:
    plane = make_plane(
        temporal_mode=TemporalMode.STATIC,
        intensity=2.0,
    )

    assert plane_activity(plane, make_context()) == pytest.approx(2.0)


def test_continuous_plane_scales_with_delta_time_and_timescale() -> None:
    plane = make_plane(
        temporal_mode=TemporalMode.CONTINUOUS,
        intensity=2.0,
        timescale=4.0,
    )
    context = make_context(delta_time=0.5)

    assert plane_activity(plane, context) == pytest.approx(0.25)


def test_phasic_plane_requires_event() -> None:
    plane = make_plane(
        temporal_mode=TemporalMode.PHASIC,
        metadata={"event_key": "activation_attempt"},
    )

    inactive = plane_activity(
        plane,
        make_context(events=set()),
    )
    active = plane_activity(
        plane,
        make_context(events={"activation_attempt"}),
    )

    assert inactive == pytest.approx(0.0)
    assert active == pytest.approx(1.0)


def test_periodic_plane_respects_duty_cycle() -> None:
    plane = make_plane(
        temporal_mode=TemporalMode.PERIODIC,
        metadata={
            "period": 10.0,
            "duty_cycle": 0.25,
        },
    )

    assert plane_activity(
        plane,
        make_context(time=2.0),
    ) == pytest.approx(1.0)
    assert plane_activity(
        plane,
        make_context(time=5.0),
    ) == pytest.approx(0.0)


def test_event_driven_plane_requires_event() -> None:
    plane = make_plane(
        temporal_mode=TemporalMode.EVENT_DRIVEN,
        metadata={"event_key": "wind_gust"},
    )

    assert plane_activity(
        plane,
        make_context(events={"wind_gust"}),
    ) == pytest.approx(1.0)


def test_delayed_plane_activates_after_start_time() -> None:
    plane = make_plane(
        temporal_mode=TemporalMode.DELAYED,
        metadata={"start_time": 5.0},
    )

    assert plane_activity(
        plane,
        make_context(time=4.9),
    ) == pytest.approx(0.0)
    assert plane_activity(
        plane,
        make_context(time=5.0),
    ) == pytest.approx(1.0)


def test_cumulative_plane_uses_exposure_state() -> None:
    plane = make_plane(
        plane_id="history",
        temporal_mode=TemporalMode.CUMULATIVE,
        intensity=0.5,
        metadata={"exposure_key": "sitting_hours"},
    )
    context = make_context(
        scalar_state={"sitting_hours": 8.0},
    )

    assert plane_activity(plane, context) == pytest.approx(4.0)


def test_hysteretic_plane_uses_condition() -> None:
    plane = make_plane(
        plane_id="lock",
        temporal_mode=TemporalMode.HYSTERETIC,
        metadata={"active_key": "locked"},
    )

    assert plane_activity(
        plane,
        make_context(conditions={"locked": True}),
    ) == pytest.approx(1.0)
    assert plane_activity(
        plane,
        make_context(conditions={"locked": False}),
    ) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Applicability and gain
# ---------------------------------------------------------------------------


def test_operator_plane_mismatch_is_not_applicable() -> None:
    operator = make_operator(plane_id="neural")
    plane = make_plane(plane_id="mechanical")

    applicable, reason = operator_is_applicable(
        operator,
        plane,
        make_context(),
        InfluenceValue(1.0),
    )

    assert applicable is False
    assert "mismatch" in reason


def test_inactive_operator_is_not_applicable() -> None:
    operator = make_operator(active=False)

    applicable, reason = operator_is_applicable(
        operator,
        make_plane(),
        make_context(),
        InfluenceValue(1.0),
    )

    assert applicable is False
    assert reason == "operator inactive"


def test_operator_delay_blocks_early_application() -> None:
    operator = make_operator(delay=5.0)

    applicable, reason = operator_is_applicable(
        operator,
        make_plane(),
        make_context(time=4.0),
        InfluenceValue(1.0),
    )

    assert applicable is False
    assert "delay" in reason


def test_operator_condition_controls_application() -> None:
    operator = make_operator(condition_key="task_ready")

    false_result = operator_is_applicable(
        operator,
        make_plane(),
        make_context(conditions={"task_ready": False}),
        InfluenceValue(1.0),
    )
    true_result = operator_is_applicable(
        operator,
        make_plane(),
        make_context(conditions={"task_ready": True}),
        InfluenceValue(1.0),
    )

    assert false_result[0] is False
    assert true_result[0] is True


@pytest.mark.parametrize(
    ("mode", "scalar", "threshold", "expected"),
    [
        ("greater_equal", 2.0, 2.0, True),
        ("greater_equal", 1.0, 2.0, False),
        ("greater", 2.0, 2.0, False),
        ("less_equal", 2.0, 2.0, True),
        ("less", 2.0, 2.0, False),
        ("absolute_greater_equal", -3.0, 2.0, True),
    ],
)
def test_threshold_modes(
    mode: str,
    scalar: float,
    threshold: float,
    expected: bool,
) -> None:
    operator = make_operator(
        threshold=threshold,
        metadata={"threshold_mode": mode},
    )

    applicable, _ = operator_is_applicable(
        operator,
        make_plane(),
        make_context(),
        InfluenceValue(scalar),
    )

    assert applicable is expected


def test_effective_gain_uses_plane_activity_and_state() -> None:
    operator = make_operator(
        gain=2.0,
        metadata={"gain_state_key": "reserve_factor"},
    )
    plane = make_plane(
        temporal_mode=TemporalMode.STATIC,
        intensity=0.5,
    )
    context = make_context(
        scalar_state={"reserve_factor": 0.25},
    )

    assert effective_operator_gain(
        operator,
        plane,
        context,
    ) == pytest.approx(0.25)


def test_effective_gain_condition_can_zero_gain() -> None:
    operator = make_operator(
        gain=2.0,
        metadata={"gain_condition_key": "enabled"},
    )

    assert effective_operator_gain(
        operator,
        make_plane(),
        make_context(conditions={"enabled": False}),
    ) == pytest.approx(0.0)


def test_effective_gain_clamps() -> None:
    operator = make_operator(
        gain=10.0,
        metadata={
            "clamp_min": 1.0,
            "clamp_max": 3.0,
        },
    )

    assert effective_operator_gain(
        operator,
        make_plane(),
        make_context(),
    ) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Scalar and vector operator algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "input_scalar", "gain", "expected"),
    [
        (OperatorKind.IDENTITY, 5.0, 2.0, 5.0),
        (OperatorKind.ADD, 5.0, 2.0, 7.0),
        (OperatorKind.LOAD, 5.0, 2.0, 7.0),
        (OperatorKind.ACTIVATE, 5.0, 2.0, 7.0),
        (OperatorKind.SUBTRACT, 5.0, 2.0, 3.0),
        (OperatorKind.UNLOAD, 5.0, 2.0, 3.0),
        (OperatorKind.INHIBIT, 5.0, 2.0, 3.0),
        (OperatorKind.MULTIPLY, 5.0, 2.0, 10.0),
        (OperatorKind.DIVIDE, 6.0, 2.0, 3.0),
        (OperatorKind.COMPRESS, 5.0, 0.2, 4.0),
        (OperatorKind.SHORTEN, 5.0, 0.2, 4.0),
        (OperatorKind.MOBILIZE, 5.0, 0.2, 6.0),
        (OperatorKind.RELEASE, 5.0, 0.2, 6.0),
        (OperatorKind.DESTABILIZE, 5.0, 0.2, 4.0),
    ],
)
def test_scalar_operators(
    kind: OperatorKind,
    input_scalar: float,
    gain: float,
    expected: float,
) -> None:
    trace = apply_operator(
        InfluenceValue(input_scalar),
        make_operator(
            kind=kind,
            gain=gain,
        ),
        make_plane(),
        make_context(),
    )

    assert trace.applied is True
    assert trace.output_value.scalar == pytest.approx(expected)


def test_add_operator_changes_vector_by_direction() -> None:
    trace = apply_operator(
        InfluenceValue(
            scalar=1.0,
            vector=Vector3(1.0, 0.0, 0.0),
        ),
        make_operator(
            kind=OperatorKind.ADD,
            gain=2.0,
            direction=Vector3(0.0, 1.0, 0.0),
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.vector == Vector3(1.0, 2.0, 0.0)


def test_multiply_scales_vector() -> None:
    trace = apply_operator(
        InfluenceValue(
            scalar=2.0,
            vector=Vector3(1.0, 2.0, 3.0),
        ),
        make_operator(
            kind=OperatorKind.MULTIPLY,
            gain=3.0,
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.vector == Vector3(3.0, 6.0, 9.0)


def test_division_by_zero_is_rejected() -> None:
    with pytest.raises(ROIFInfluenceError):
        apply_operator(
            InfluenceValue(1.0),
            make_operator(
                kind=OperatorKind.DIVIDE,
                gain=0.0,
            ),
            make_plane(),
            make_context(),
        )


def test_transfer_load_reduces_scalar_and_adds_vector() -> None:
    trace = apply_operator(
        InfluenceValue(
            scalar=10.0,
            vector=Vector3(),
        ),
        make_operator(
            kind=OperatorKind.TRANSFER_LOAD,
            gain=0.3,
            direction=Vector3(1.0, 0.0, 0.0),
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.scalar == pytest.approx(7.0)
    assert trace.output_value.vector == Vector3(3.0, 0.0, 0.0)


def test_rotation_updates_vector_orientation() -> None:
    trace = apply_operator(
        InfluenceValue(
            scalar=1.0,
            vector=Vector3(1.0, 0.0, 0.0),
        ),
        make_operator(
            kind=OperatorKind.ROTATE,
            gain=0.5,
            direction=Vector3(0.0, 1.0, 0.0),
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.scalar == pytest.approx(1.0)
    assert trace.output_value.vector == Vector3(1.0, 0.5, 0.0)


def test_stabilize_moves_scalar_toward_target() -> None:
    trace = apply_operator(
        InfluenceValue(0.2),
        make_operator(
            kind=OperatorKind.STABILIZE,
            gain=0.5,
            metadata={"target": 1.0},
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.scalar == pytest.approx(0.6)


def test_filter_clamps_scalar() -> None:
    trace = apply_operator(
        InfluenceValue(10.0),
        make_operator(
            kind=OperatorKind.FILTER,
            metadata={
                "lower": 0.0,
                "upper": 5.0,
            },
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.scalar == pytest.approx(5.0)


def test_filter_rejects_invalid_bounds() -> None:
    with pytest.raises(ROIFInfluenceError):
        apply_operator(
            InfluenceValue(1.0),
            make_operator(
                kind=OperatorKind.FILTER,
                metadata={
                    "lower": 2.0,
                    "upper": 1.0,
                },
            ),
            make_plane(),
            make_context(),
        )


def test_trigger_sets_configured_value() -> None:
    trace = apply_operator(
        InfluenceValue(1.0),
        make_operator(
            kind=OperatorKind.TRIGGER,
            gain=2.0,
            metadata={"trigger_value": 9.0},
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.scalar == pytest.approx(9.0)


def test_adapt_moves_toward_target() -> None:
    trace = apply_operator(
        InfluenceValue(0.0),
        make_operator(
            kind=OperatorKind.ADAPT,
            gain=0.25,
            metadata={"target": 1.0},
        ),
        make_plane(),
        make_context(),
    )

    assert trace.output_value.scalar == pytest.approx(0.25)


def test_inapplicable_operator_returns_identity_trace() -> None:
    value = InfluenceValue(3.0)
    trace = apply_operator(
        value,
        make_operator(active=False),
        make_plane(),
        make_context(),
    )

    assert trace.applied is False
    assert trace.output_value == value
    assert trace.effective_gain == pytest.approx(0.0)


def test_operator_application_metadata_is_read_only() -> None:
    trace = apply_operator(
        InfluenceValue(1.0),
        make_operator(),
        make_plane(),
        make_context(),
    )

    assert isinstance(trace, OperatorApplication)
    assert isinstance(trace.metadata, MappingProxyType)


# ---------------------------------------------------------------------------
# Operator composition
# ---------------------------------------------------------------------------


def test_sequential_composition() -> None:
    operators = {
        "add": make_operator(
            operator_id="add",
            kind=OperatorKind.ADD,
            gain=2.0,
        ),
        "multiply": make_operator(
            operator_id="multiply",
            kind=OperatorKind.MULTIPLY,
            gain=3.0,
        ),
    }
    composition = OperatorComposition(
        composition_id="sequential",
        operator_ids=("add", "multiply"),
        kind=CompositionKind.SEQUENTIAL,
    )

    result = compose_operators(
        InfluenceValue(1.0),
        composition,
        operators,
        {"mechanical": make_plane()},
        make_context(),
    )

    assert result.output_value.scalar == pytest.approx(9.0)
    assert len(result.operator_traces) == 2


def test_additive_composition_combines_deltas() -> None:
    operators = {
        "add2": make_operator(
            operator_id="add2",
            kind=OperatorKind.ADD,
            gain=2.0,
        ),
        "add3": make_operator(
            operator_id="add3",
            kind=OperatorKind.ADD,
            gain=3.0,
        ),
    }
    composition = OperatorComposition(
        composition_id="additive",
        operator_ids=("add2", "add3"),
        kind=CompositionKind.ADDITIVE,
    )

    result = compose_operators(
        InfluenceValue(10.0),
        composition,
        operators,
        {"mechanical": make_plane()},
        make_context(),
    )

    assert result.output_value.scalar == pytest.approx(15.0)


def test_multiplicative_composition_combines_factors() -> None:
    operators = {
        "x2": make_operator(
            operator_id="x2",
            kind=OperatorKind.MULTIPLY,
            gain=2.0,
        ),
        "x3": make_operator(
            operator_id="x3",
            kind=OperatorKind.MULTIPLY,
            gain=3.0,
        ),
    }
    composition = OperatorComposition(
        composition_id="multiplicative",
        operator_ids=("x2", "x3"),
        kind=CompositionKind.MULTIPLICATIVE,
    )

    result = compose_operators(
        InfluenceValue(5.0),
        composition,
        operators,
        {"mechanical": make_plane()},
        make_context(),
    )

    assert result.output_value.scalar == pytest.approx(30.0)


def test_conditional_composition_chooses_first_applied_operator() -> None:
    operators = {
        "inactive": make_operator(
            operator_id="inactive",
            active=False,
            gain=100.0,
        ),
        "active": make_operator(
            operator_id="active",
            kind=OperatorKind.ADD,
            gain=2.0,
        ),
    }
    composition = OperatorComposition(
        composition_id="conditional",
        operator_ids=("inactive", "active"),
        kind=CompositionKind.CONDITIONAL,
    )

    result = compose_operators(
        InfluenceValue(1.0),
        composition,
        operators,
        {"mechanical": make_plane()},
        make_context(),
    )

    assert result.output_value.scalar == pytest.approx(3.0)


def test_vector_sum_composition() -> None:
    operators = {
        "x": make_operator(
            operator_id="x",
            kind=OperatorKind.ADD,
            gain=1.0,
            direction=Vector3(1.0, 0.0, 0.0),
        ),
        "y": make_operator(
            operator_id="y",
            kind=OperatorKind.ADD,
            gain=1.0,
            direction=Vector3(0.0, 1.0, 0.0),
        ),
    }
    composition = OperatorComposition(
        composition_id="vector_sum",
        operator_ids=("x", "y"),
        kind=CompositionKind.VECTOR_SUM,
    )

    result = compose_operators(
        InfluenceValue(0.0),
        composition,
        operators,
        {"mechanical": make_plane()},
        make_context(),
    )

    assert result.output_value.scalar == pytest.approx(2.0)
    assert result.output_value.vector == Vector3(1.0, 1.0, 0.0)


def test_maximum_and_minimum_compositions() -> None:
    operators = {
        "small": make_operator(
            operator_id="small",
            kind=OperatorKind.ADD,
            gain=1.0,
        ),
        "large": make_operator(
            operator_id="large",
            kind=OperatorKind.ADD,
            gain=5.0,
        ),
    }
    planes = {"mechanical": make_plane()}
    context = make_context()

    maximum = compose_operators(
        InfluenceValue(0.0),
        OperatorComposition(
            composition_id="max",
            operator_ids=("small", "large"),
            kind=CompositionKind.MAXIMUM,
        ),
        operators,
        planes,
        context,
    )
    minimum = compose_operators(
        InfluenceValue(0.0),
        OperatorComposition(
            composition_id="min",
            operator_ids=("small", "large"),
            kind=CompositionKind.MINIMUM,
        ),
        operators,
        planes,
        context,
    )

    assert maximum.output_value.scalar == pytest.approx(5.0)
    assert minimum.output_value.scalar == pytest.approx(1.0)


def test_composition_rejects_unknown_operator() -> None:
    with pytest.raises(ROIFInfluenceError):
        compose_operators(
            InfluenceValue(1.0),
            OperatorComposition(
                composition_id="bad",
                operator_ids=("missing",),
                kind=CompositionKind.SEQUENTIAL,
            ),
            {},
            {"mechanical": make_plane()},
            make_context(),
        )


def test_composition_rejects_unknown_plane() -> None:
    operator = make_operator(plane_id="missing")

    with pytest.raises(ROIFInfluenceError):
        compose_operators(
            InfluenceValue(1.0),
            OperatorComposition(
                composition_id="bad",
                operator_ids=("operator",),
                kind=CompositionKind.SEQUENTIAL,
            ),
            {"operator": operator},
            {},
            make_context(),
        )


def test_custom_composition_requires_domain_adapter() -> None:
    with pytest.raises(ROIFInfluenceError):
        compose_operators(
            InfluenceValue(1.0),
            OperatorComposition(
                composition_id="custom",
                operator_ids=("operator",),
                kind=CompositionKind.CUSTOM,
            ),
            {"operator": make_operator()},
            {"mechanical": make_plane()},
            make_context(),
        )


def test_composition_application_is_created() -> None:
    result = compose_operators(
        InfluenceValue(1.0),
        OperatorComposition(
            composition_id="single",
            operator_ids=("operator",),
            kind=CompositionKind.SEQUENTIAL,
        ),
        {"operator": make_operator()},
        {"mechanical": make_plane()},
        make_context(),
    )

    assert isinstance(result, CompositionApplication)
    assert isinstance(result.metadata, MappingProxyType)


# ---------------------------------------------------------------------------
# Direct channel transformation
# ---------------------------------------------------------------------------


def test_apply_operator_to_channel_load() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.LOAD,
            gain=3.0,
            metadata={"channel_field": "load"},
        ),
        make_plane(),
        make_context(),
    )

    assert isinstance(result, ChannelInfluenceResult)
    assert result.channel_before.capacity_state.load == pytest.approx(2.0)
    assert result.channel_after.capacity_state.load == pytest.approx(5.0)


def test_apply_operator_to_channel_capacity() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.SUBTRACT,
            gain=2.0,
            metadata={"channel_field": "capacity"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.capacity_state.capacity == pytest.approx(8.0)


def test_apply_operator_to_baseline_output() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.MULTIPLY,
            gain=2.0,
            metadata={"channel_field": "baseline_output"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.baseline_output == pytest.approx(2.0)


def test_apply_operator_to_history_factor() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.MULTIPLY,
            gain=0.5,
            metadata={"channel_field": "history_factor"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.history_factor == pytest.approx(0.5)


def test_apply_operator_to_activation_field() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.COMPRESS,
            gain=0.4,
            metadata={"channel_field": "activation.afferent_gate"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.activation.afferent_gate == pytest.approx(0.6)


def test_activation_field_is_clamped_to_unit_interval() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.ADD,
            gain=10.0,
            metadata={"channel_field": "activation.command"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.activation.command == pytest.approx(1.0)


def test_apply_operator_to_geometry_length() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.SHORTEN,
            gain=0.2,
            metadata={"channel_field": "geometry.length"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.geometry.length == pytest.approx(0.8)


def test_apply_operator_to_geometry_mobility() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.COMPRESS,
            gain=0.25,
            metadata={"channel_field": "geometry.mobility"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.geometry.mobility == pytest.approx(0.75)


def test_apply_rotation_to_geometry_orientation() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.ROTATE,
            gain=0.5,
            direction=Vector3(0.0, 1.0, 0.0),
            metadata={"channel_field": "geometry.orientation"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.geometry.orientation == Vector3(
        0.0,
        0.5,
        0.0,
    )


def test_apply_operator_to_direction() -> None:
    result = apply_operator_to_channel(
        make_channel(),
        make_operator(
            kind=OperatorKind.ROTATE,
            gain=0.5,
            direction=Vector3(0.0, 1.0, 0.0),
            metadata={"channel_field": "direction"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after.direction == Vector3(1.0, 0.5, 0.0)


def test_inactive_operator_preserves_channel() -> None:
    channel = make_channel()
    result = apply_operator_to_channel(
        channel,
        make_operator(
            active=False,
            metadata={"channel_field": "load"},
        ),
        make_plane(),
        make_context(),
    )

    assert result.channel_after == channel
    assert result.applications[0].applied is False


def test_unknown_channel_field_is_rejected() -> None:
    with pytest.raises(ROIFInfluenceError):
        apply_operator_to_channel(
            make_channel(),
            make_operator(
                metadata={"channel_field": "unknown"},
            ),
            make_plane(),
            make_context(),
        )


def test_unknown_activation_field_is_rejected() -> None:
    with pytest.raises(ROIFInfluenceError):
        apply_operator_to_channel(
            make_channel(),
            make_operator(
                metadata={"channel_field": "activation.unknown"},
            ),
            make_plane(),
            make_context(),
        )


def test_operator_sequence_updates_channel_stepwise() -> None:
    operators = {
        "load": make_operator(
            operator_id="load",
            kind=OperatorKind.LOAD,
            gain=2.0,
            metadata={"channel_field": "load"},
        ),
        "inhibit": make_operator(
            operator_id="inhibit",
            kind=OperatorKind.COMPRESS,
            gain=0.5,
            metadata={"channel_field": "activation.afferent_gate"},
        ),
    }

    result = apply_operator_sequence_to_channel(
        make_channel(),
        ("load", "inhibit"),
        operators,
        {"mechanical": make_plane()},
        make_context(),
    )

    assert result.channel_after.capacity_state.load == pytest.approx(4.0)
    assert result.channel_after.activation.afferent_gate == pytest.approx(0.5)
    assert len(result.applications) == 2


def test_operator_sequence_rejects_unknown_operator() -> None:
    with pytest.raises(ROIFInfluenceError):
        apply_operator_sequence_to_channel(
            make_channel(),
            ("missing",),
            {},
            {"mechanical": make_plane()},
            make_context(),
        )


# ---------------------------------------------------------------------------
# Multiplicativity and vector aggregation
# ---------------------------------------------------------------------------


def test_influence_factor_product() -> None:
    result = influence_factor_product(
        {
            "reserve": 0.8,
            "geometry": 0.5,
            "activation": 0.25,
        }
    )

    assert result == pytest.approx(0.1)


def test_factor_product_respects_floor_and_ceiling() -> None:
    low = influence_factor_product(
        {"a": 0.1, "b": 0.1},
        floor=0.5,
    )
    high = influence_factor_product(
        {"a": 10.0, "b": 10.0},
        ceiling=5.0,
    )

    assert low == pytest.approx(0.5)
    assert high == pytest.approx(5.0)


def test_factor_product_rejects_inverted_bounds() -> None:
    with pytest.raises(ROIFInfluenceError):
        influence_factor_product(
            {"a": 1.0},
            floor=2.0,
            ceiling=1.0,
        )


def test_vector_sum() -> None:
    result = vector_sum(
        (
            Vector3(1.0, 0.0, 0.0),
            Vector3(0.0, 2.0, 0.0),
            Vector3(0.0, 0.0, 3.0),
        )
    )

    assert result == Vector3(1.0, 2.0, 3.0)


def test_vector_sum_rejects_non_vector() -> None:
    with pytest.raises(ROIFInfluenceError):
        vector_sum(
            (
                Vector3(),
                1.0,  # type: ignore[arg-type]
            )
        )


def test_channels_vector_sum_combines_effective_outputs() -> None:
    left = FunctionalChannel(
        channel_id="left",
        entity_id="entity",
        name="Left",
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(1.0, 0.0),
    )
    right = FunctionalChannel(
        channel_id="right",
        entity_id="entity",
        name="Right",
        direction=Vector3(0.0, 1.0, 0.0),
        capacity_state=CapacityState(1.0, 0.0),
    )

    result = channels_vector_sum((left, right))

    assert result == Vector3(1.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def test_planes_by_id_returns_read_only_lookup() -> None:
    lookup = planes_by_id(
        (
            make_plane(plane_id="mechanical"),
            make_plane(plane_id="neural"),
        )
    )

    assert isinstance(lookup, MappingProxyType)
    assert set(lookup) == {"mechanical", "neural"}


def test_planes_by_id_rejects_duplicates() -> None:
    plane = make_plane()

    with pytest.raises(ROIFInfluenceError):
        planes_by_id((plane, plane))


def test_operators_by_id_returns_read_only_lookup() -> None:
    lookup = operators_by_id(
        (
            make_operator(operator_id="a"),
            make_operator(operator_id="b"),
        )
    )

    assert isinstance(lookup, MappingProxyType)
    assert set(lookup) == {"a", "b"}


def test_operators_by_id_rejects_duplicates() -> None:
    operator = make_operator()

    with pytest.raises(ROIFInfluenceError):
        operators_by_id((operator, operator))
