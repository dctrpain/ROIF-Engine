"""
Tests for roif.roif_cascade.

The suite verifies the dynamic ROIF cascade contract:

- immutable configuration, injections, steps, and trajectories;
- linear, leaky, logistic, tanh, threshold, and conservative updates;
- external injections and feedback;
- transmission, threshold, reserve, overload, recovery, and convergence events;
- deterministic recursive propagation;
- static and refreshed Capacity Tensors;
- state-dependent channel updates;
- convergence, failure, trajectory inspection, comparison, and summaries.

The tests are domain-independent and apply to biological, engineering,
informational, behavioural, and control systems represented by ROIF channels.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from roif.roif_entities import (
    ActivationAvailability,
    CapacityState,
    EntityKind,
    FunctionalChannel,
    FunctionalRole,
    GeometryState,
    InfluenceOperator,
    InfluencePlane,
    OperatorKind,
    PlaneKind,
    ROIFSystem,
    StructuralEntity,
    TemporalMode,
    Vector3,
)
from roif.roif_influence import InfluenceContext
from roif.roif_tensor import (
    SelfCouplingMode,
    TensorBuildConfig,
)
from roif.roif_cascade import (
    CascadeConfig,
    CascadeDirection,
    CascadeInjection,
    CascadeStep,
    CascadeTerminationReason,
    CascadeTrajectory,
    CascadeUpdateMode,
    EventKind,
    ROIFCascadeError,
    TensorRefreshMode,
    active_injection_vector,
    default_initial_state,
    injection_vector,
    reserve_events,
    run_cascade,
    system_with_simulated_state,
    threshold_events,
    trajectory_difference,
    trajectory_summary,
    transmission_events,
    update_state,
)


def make_channel(
    channel_id: str,
    *,
    entity_id: str | None = None,
    capacity: float = 10.0,
    load: float = 0.0,
    direction: Vector3 = Vector3(1.0, 0.0, 0.0),
    baseline_output: float = 0.0,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id or f"entity_{channel_id}",
        name=channel_id,
        role=FunctionalRole.TRANSMITTER,
        direction=direction,
        capacity_state=CapacityState(capacity=capacity, load=load),
        activation=ActivationAvailability(),
        geometry=GeometryState(),
        baseline_output=baseline_output,
    )


def make_system(
    *,
    gain: float = 0.5,
    channels: tuple[FunctionalChannel, ...] | None = None,
    self_edge: bool = False,
) -> ROIFSystem:
    channels = channels or (
        make_channel("a", entity_id="entity_a"),
        make_channel("b", entity_id="entity_b"),
    )
    entities = tuple(
        StructuralEntity(
            entity_id=channel.entity_id,
            name=channel.entity_id,
            kind=EntityKind.GENERIC,
            channels=(channel,),
        )
        for channel in channels
    )
    operators = [
        InfluenceOperator(
            operator_id="a_to_b",
            name="a_to_b",
            plane_id="mechanical",
            kind=OperatorKind.TRANSFER_LOAD,
            source_ids=("a",),
            target_ids=("b",),
            gain=gain,
        )
    ]
    if self_edge:
        operators.append(
            InfluenceOperator(
                operator_id="b_to_b",
                name="b_to_b",
                plane_id="mechanical",
                kind=OperatorKind.TRANSFER_LOAD,
                source_ids=("b",),
                target_ids=("b",),
                gain=gain,
            )
        )
    return ROIFSystem(
        system_id="cascade_system",
        name="Cascade system",
        entities=entities,
        planes=(
            InfluencePlane(
                plane_id="mechanical",
                name="Mechanical",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.STATIC,
            ),
        ),
        operators=tuple(operators),
    )


def neutral_tensor_config() -> TensorBuildConfig:
    return TensorBuildConfig(
        tensor_ceiling=100.0,
        source_reserve_exponent=0.0,
        target_reserve_exponent=0.0,
        availability_exponent=0.0,
        geometry_exponent=0.0,
        material_exponent=0.0,
        history_exponent=0.0,
        state_dependent=False,
        self_coupling_mode=SelfCouplingMode.NONE,
    )


def simple_config(**overrides: object) -> CascadeConfig:
    values: dict[str, object] = {
        "steps": 3,
        "update_mode": CascadeUpdateMode.LINEAR,
        "retention": 0.0,
        "dissipation": 0.0,
        "lower_bound": 0.0,
        "upper_bound": 100.0,
        "clip_state": False,
        "convergence_tolerance": 1e-12,
        "convergence_patience": 10,
        "record_transmission_events": False,
    }
    values.update(overrides)
    return CascadeConfig(**values)


# ---------------------------------------------------------------------------
# Configuration and value objects
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_immutable() -> None:
    config = CascadeConfig()

    assert config.steps == 32
    assert config.update_mode is CascadeUpdateMode.LEAKY
    assert isinstance(config.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        config.steps = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("steps", 0),
        ("delta_time", 0.0),
        ("dissipation", -0.1),
        ("dissipation", 1.1),
        ("logistic_steepness", 0.0),
        ("convergence_tolerance", 0.0),
        ("convergence_patience", 0),
        ("divergence_threshold", 0.0),
        ("tensor_refresh_period", 0),
        ("tensor_refresh_tolerance", -0.1),
    ],
)
def test_config_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ROIFCascadeError):
        CascadeConfig(**{field_name: value})


def test_config_rejects_inverted_bounds() -> None:
    with pytest.raises(ROIFCascadeError):
        CascadeConfig(lower_bound=2.0, upper_bound=1.0)


def test_injection_creation_and_freezing() -> None:
    injection = CascadeInjection(
        injection_id="pulse",
        step=2,
        values={"a": 1.0},
        gain=0.5,
        persistent=True,
    )

    assert injection.values["a"] == pytest.approx(1.0)
    assert isinstance(injection.values, MappingProxyType)
    assert isinstance(injection.metadata, MappingProxyType)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("injection_id", ""),
        ("step", -1),
    ],
)
def test_injection_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "injection_id": "pulse",
        "step": 0,
        "values": {"a": 1.0},
    }
    kwargs[field_name] = value
    with pytest.raises(ROIFCascadeError):
        CascadeInjection(**kwargs)


def test_cascade_step_arrays_are_read_only() -> None:
    step = CascadeStep(
        step=0,
        time=0.0,
        state_before=np.array([1.0, 0.0]),
        external_input=np.zeros(2),
        propagated_input=np.array([0.0, 0.5]),
        feedback_input=np.zeros(2),
        state_after=np.array([0.0, 0.5]),
        state_delta=np.array([-1.0, 0.5]),
        tensor_matrix=np.array([[0.0, 0.0], [0.5, 0.0]]),
        spectral_radius=0.0,
        maximum_value=0.5,
        minimum_value=0.0,
        l1_norm=0.5,
        l2_norm=0.5,
    )

    assert step.state_after.flags.writeable is False
    assert step.tensor_matrix.flags.writeable is False


def test_cascade_step_rejects_mismatched_shapes() -> None:
    with pytest.raises(ROIFCascadeError):
        CascadeStep(
            step=0,
            time=0.0,
            state_before=np.zeros(2),
            external_input=np.zeros(1),
            propagated_input=np.zeros(2),
            feedback_input=np.zeros(2),
            state_after=np.zeros(2),
            state_delta=np.zeros(2),
            tensor_matrix=np.zeros((2, 2)),
            spectral_radius=0.0,
            maximum_value=0.0,
            minimum_value=0.0,
            l1_norm=0.0,
            l2_norm=0.0,
        )


# ---------------------------------------------------------------------------
# Initial state and injections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (CascadeDirection.LOAD, {"a": 2.0, "b": 4.0}),
        (CascadeDirection.DEFICIT, {"a": 0.2, "b": 0.4}),
        (CascadeDirection.ACTIVATION, {"a": 1.0, "b": 1.0}),
        (CascadeDirection.DAMAGE, {"a": 0.0, "b": 0.0}),
    ],
)
def test_default_initial_state_modes(
    direction: CascadeDirection,
    expected: dict[str, float],
) -> None:
    system = make_system(
        channels=(
            make_channel("a", load=2.0),
            make_channel("b", load=4.0),
        )
    )

    state = default_initial_state(system, direction=direction)

    assert dict(state) == pytest.approx(expected)


def test_injection_vector_uses_channel_order_and_gain() -> None:
    injection = CascadeInjection(
        injection_id="pulse",
        step=0,
        values={"b": 2.0, "a": 1.0},
        gain=0.5,
    )

    vector = injection_vector(injection, ("a", "b"))

    assert np.array_equal(vector, np.array([0.5, 1.0]))
    assert vector.flags.writeable is False


def test_active_injection_vector_combines_active_inputs() -> None:
    injections = (
        CascadeInjection("one", 0, {"a": 1.0}),
        CascadeInjection("two", 0, {"a": 2.0, "b": 1.0}),
    )

    vector = active_injection_vector(injections, 0, ("a", "b"))

    assert np.array_equal(vector, np.array([3.0, 1.0]))


def test_persistent_injection_remains_active() -> None:
    injection = CascadeInjection(
        "persistent",
        1,
        {"a": 2.0},
        persistent=True,
    )

    assert np.array_equal(
        active_injection_vector((injection,), 0, ("a",)),
        np.array([0.0]),
    )
    assert np.array_equal(
        active_injection_vector((injection,), 3, ("a",)),
        np.array([2.0]),
    )


# ---------------------------------------------------------------------------
# Update rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (CascadeUpdateMode.LINEAR, np.array([0.5, 1.0])),
        (CascadeUpdateMode.LEAKY, np.array([1.4, 1.9])),
    ],
)
def test_basic_update_modes(
    mode: CascadeUpdateMode,
    expected: np.ndarray,
) -> None:
    config = CascadeConfig(
        update_mode=mode,
        retention=0.0,
        dissipation=0.1,
        clip_state=False,
    )

    result = update_state(
        np.array([1.0, 1.0]),
        np.array([0.5, 1.0]),
        np.zeros(2),
        np.zeros(2),
        config,
    )

    assert np.allclose(result, expected)


def test_linear_update_includes_retention_feedback_and_external() -> None:
    config = CascadeConfig(
        update_mode=CascadeUpdateMode.LINEAR,
        retention=0.5,
        feedback_gain=2.0,
        external_gain=3.0,
        clip_state=False,
    )

    result = update_state(
        np.array([2.0]),
        np.array([1.0]),
        np.array([4.0]),
        np.array([5.0]),
        config,
    )

    assert result[0] == pytest.approx(22.0)


def test_logistic_update_is_bounded() -> None:
    result = update_state(
        np.array([0.0, 0.0]),
        np.array([-100.0, 100.0]),
        np.zeros(2),
        np.zeros(2),
        CascadeConfig(
            update_mode=CascadeUpdateMode.LOGISTIC,
            allow_negative_state=True,
            clip_state=False,
        ),
    )

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_tanh_update_is_bounded() -> None:
    result = update_state(
        np.zeros(2),
        np.array([-100.0, 100.0]),
        np.zeros(2),
        np.zeros(2),
        CascadeConfig(
            update_mode=CascadeUpdateMode.TANH,
            allow_negative_state=True,
            clip_state=False,
        ),
    )

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_threshold_update_suppresses_subthreshold_values() -> None:
    result = update_state(
        np.zeros(2),
        np.array([0.4, 0.6]),
        np.zeros(2),
        np.zeros(2),
        CascadeConfig(
            update_mode=CascadeUpdateMode.THRESHOLD,
            threshold=0.5,
            clip_state=False,
        ),
    )

    assert np.array_equal(result, np.array([0.0, 0.6]))


def test_negative_state_is_clamped_when_disallowed() -> None:
    result = update_state(
        np.zeros(1),
        np.array([-1.0]),
        np.zeros(1),
        np.zeros(1),
        CascadeConfig(
            update_mode=CascadeUpdateMode.LINEAR,
            allow_negative_state=False,
            clip_state=False,
        ),
    )

    assert result[0] == pytest.approx(0.0)


def test_state_clipping() -> None:
    result = update_state(
        np.zeros(1),
        np.array([10.0]),
        np.zeros(1),
        np.zeros(1),
        CascadeConfig(
            update_mode=CascadeUpdateMode.LINEAR,
            lower_bound=0.0,
            upper_bound=1.0,
            clip_state=True,
        ),
    )

    assert result[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def test_threshold_crossing_event() -> None:
    events = threshold_events(
        np.array([0.4]),
        np.array([0.6]),
        ("a",),
        0,
        0.0,
        CascadeConfig(threshold=0.5),
    )

    assert len(events) == 1
    assert events[0].metadata["event_kind"] == (
        EventKind.THRESHOLD_CROSSING.value
    )


def test_recovery_event() -> None:
    events = threshold_events(
        np.array([0.6]),
        np.array([0.4]),
        ("a",),
        0,
        0.0,
        CascadeConfig(threshold=0.5),
    )

    assert any(
        event.metadata["event_kind"] == EventKind.RECOVERY.value
        for event in events
    )


def test_saturation_event() -> None:
    events = threshold_events(
        np.array([0.5]),
        np.array([1.0]),
        ("a",),
        0,
        0.0,
        CascadeConfig(upper_bound=1.0, clip_state=True),
    )

    assert any(
        event.metadata["event_kind"] == EventKind.SATURATION.value
        for event in events
    )


def test_load_overload_event() -> None:
    system = make_system(
        channels=(
            make_channel("a", capacity=1.0),
            make_channel("b", capacity=1.0),
        )
    )
    events = reserve_events(
        system,
        np.array([2.0, 0.0]),
        ("a", "b"),
        0,
        0.0,
        CascadeDirection.LOAD,
    )

    assert events[0].metadata["event_kind"] == EventKind.OVERLOAD.value


def test_normalized_failure_event() -> None:
    events = reserve_events(
        make_system(),
        np.array([1.0, 0.0]),
        ("a", "b"),
        0,
        0.0,
        CascadeDirection.DAMAGE,
    )

    assert events[0].metadata["event_kind"] == EventKind.FAILURE.value


# ---------------------------------------------------------------------------
# System-state adapter
# ---------------------------------------------------------------------------


def test_system_with_load_state_updates_capacity_load() -> None:
    updated = system_with_simulated_state(
        make_system(),
        (3.0, 4.0),
        direction=CascadeDirection.LOAD,
    )

    assert updated.channel("a").capacity_state.load == pytest.approx(3.0)
    assert updated.channel("b").capacity_state.load == pytest.approx(4.0)


def test_system_with_activation_state_updates_command() -> None:
    updated = system_with_simulated_state(
        make_system(),
        (0.2, 0.8),
        direction=CascadeDirection.ACTIVATION,
    )

    assert updated.channel("a").activation.command == pytest.approx(0.2)
    assert updated.channel("b").activation.command == pytest.approx(0.8)


def test_system_with_damage_state_updates_history_factor() -> None:
    updated = system_with_simulated_state(
        make_system(),
        (0.2, 0.8),
        direction=CascadeDirection.DAMAGE,
    )

    assert updated.channel("a").history_factor == pytest.approx(0.8)
    assert updated.channel("b").history_factor == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Complete cascade execution
# ---------------------------------------------------------------------------


def test_one_step_cascade_transmits_a_to_b() -> None:
    trajectory = run_cascade(
        make_system(gain=0.5),
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=1),
    )

    assert np.array_equal(
        trajectory.final_state,
        np.array([0.0, 1.0]),
    )


def test_multi_step_chain_is_recursive() -> None:
    trajectory = run_cascade(
        make_system(gain=0.5, self_edge=True),
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=3),
    )

    assert trajectory.step_count == 3
    assert trajectory.steps[0].state_after[1] == pytest.approx(1.0)
    assert trajectory.steps[1].state_after[1] == pytest.approx(0.5)
    assert trajectory.steps[2].state_after[1] == pytest.approx(0.25)


def test_external_injection_enters_at_requested_step() -> None:
    trajectory = run_cascade(
        make_system(gain=0.0),
        initial_state={"a": 0.0, "b": 0.0},
        injections=(
            CascadeInjection("pulse", 1, {"a": 2.0}),
        ),
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=2),
    )

    assert trajectory.steps[0].external_input[0] == pytest.approx(0.0)
    assert trajectory.steps[1].external_input[0] == pytest.approx(2.0)


def test_context_feedback_is_applied() -> None:
    trajectory = run_cascade(
        make_system(gain=0.0),
        initial_state={"a": 0.0, "b": 0.0},
        context=InfluenceContext(
            scalar_state={"feedback.a": 0.5},
        ),
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=1),
    )

    assert trajectory.final_state[0] == pytest.approx(0.5)


def test_cascade_is_deterministic() -> None:
    kwargs = dict(
        system=make_system(),
        initial_state={"a": 1.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(),
    )

    left = run_cascade(**kwargs)
    right = run_cascade(**kwargs)

    assert np.array_equal(left.final_state, right.final_state)
    assert np.array_equal(left.state_history, right.state_history)
    assert left.events == right.events


def test_zero_system_converges() -> None:
    trajectory = run_cascade(
        make_system(gain=0.0),
        initial_state={"a": 0.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(
            steps=10,
            convergence_patience=2,
        ),
    )

    assert trajectory.converged is True
    assert trajectory.termination_reason is (
        CascadeTerminationReason.CONVERGED
    )


def test_overload_can_stop_cascade() -> None:
    system = make_system(
        gain=0.0,
        channels=(
            make_channel("a", capacity=1.0),
            make_channel("b", capacity=1.0),
        ),
    )
    trajectory = run_cascade(
        system,
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(
            direction=CascadeDirection.LOAD,
            steps=3,
            stop_on_failure=True,
        ),
    )

    assert any(
        event.metadata["event_kind"] == EventKind.OVERLOAD.value
        for event in trajectory.events
    )

    assert trajectory.failed is False


def test_transmission_events_can_be_recorded() -> None:
    trajectory = run_cascade(
        make_system(),
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(
            steps=1,
            record_transmission_events=True,
        ),
    )

    assert any(
        event.metadata["event_kind"]
        == EventKind.TRANSMISSION.value
        for event in trajectory.events
    )


def test_every_step_tensor_refresh_records_tensors_and_events() -> None:
    trajectory = run_cascade(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(
            steps=3,
            tensor_refresh=TensorRefreshMode.EVERY_STEP,
        ),
    )

    assert len(trajectory.tensors) == 3
    assert sum(
        event.metadata["event_kind"]
        == EventKind.TENSOR_REFRESH.value
        for event in trajectory.events
    ) == 2


def test_update_system_channels_with_refresh_runs() -> None:
    trajectory = run_cascade(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0},
        tensor_config=TensorBuildConfig(
            tensor_ceiling=100.0,
            self_coupling_mode=SelfCouplingMode.NONE,
        ),
        cascade_config=simple_config(
            steps=2,
            tensor_refresh=TensorRefreshMode.EVERY_STEP,
            update_system_channels=True,
        ),
    )

    assert trajectory.step_count == 2
    assert len(trajectory.tensors) == 2


# ---------------------------------------------------------------------------
# Trajectory inspection and comparison
# ---------------------------------------------------------------------------


def test_trajectory_history_series_peak_and_exposure() -> None:
    trajectory = run_cascade(
        make_system(gain=0.5, self_edge=True),
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=3),
    )

    assert trajectory.state_history.shape == (4, 2)
    assert trajectory.series("b").flags.writeable is False
    assert trajectory.peak("b") == pytest.approx(1.0)
    assert trajectory.time_to_peak("b") == 1
    assert trajectory.total_exposure("b") == pytest.approx(1.75)


def test_first_threshold_crossing() -> None:
    trajectory = run_cascade(
        make_system(),
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=1),
    )

    assert trajectory.first_threshold_crossing("b", 0.5) == 1
    assert trajectory.first_threshold_crossing("b", 2.0) is None


def test_trajectory_difference() -> None:
    left = run_cascade(
        make_system(gain=0.5),
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=1),
    )
    right = run_cascade(
        make_system(gain=0.25),
        initial_state={"a": 2.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=1),
    )

    difference = trajectory_difference(left, right)

    assert difference[1] == pytest.approx(0.5)
    assert difference.flags.writeable is False


def test_trajectory_summary_is_read_only() -> None:
    trajectory = run_cascade(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=1),
    )
    summary = trajectory_summary(trajectory)

    assert isinstance(summary, MappingProxyType)
    assert summary["channel_count"] == 2
    assert summary["step_count"] == 1
    assert summary["tensor_count"] == 1


def test_unknown_trajectory_channel_raises_key_error() -> None:
    trajectory = run_cascade(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0},
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_config(steps=1),
    )

    with pytest.raises(KeyError):
        trajectory.index("missing")

