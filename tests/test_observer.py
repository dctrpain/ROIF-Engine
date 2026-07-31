"""Tests for the ROIF simulation observer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from roif.cascade_event import CascadeEventBatch
from roif.history import CascadeHistory
from roif.metrics import SimulationMetrics, SnapshotMetrics
from roif.observer import (
    AlertKind,
    AlertLevel,
    Observation,
    ObservationSeries,
    ObserverAlert,
    ObserverConfig,
    ObserverError,
    SimulationObserver,
    observe_snapshots,
)
from roif.plane_kernel import PlaneKernelResult
from roif.state_snapshot import StateSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_snapshot(
    *,
    snapshot_id: str = "snapshot-1",
    time: float = 1.0,
    step_index: int = 1,
    plane_ids: tuple[str, ...] = ("A", "B"),
    activation_before: tuple[float, ...] = (0.1, 0.2),
    activation_after: tuple[float, ...] = (0.2, 0.1),
) -> StateSnapshot:
    count = len(plane_ids)
    before = np.asarray(activation_before, dtype=float)
    after = np.asarray(activation_after, dtype=float)

    if before.size != count or after.size != count:
        raise ValueError("activation vectors must match plane_ids")

    result = PlaneKernelResult(
        plane_ids=plane_ids,
        time=time,
        dt=1.0,
        activation_before=tuple(before),
        interaction=tuple(np.zeros(count)),
        rate=tuple(after - before),
        activation_after=tuple(after),
        clipped=tuple(False for _ in range(count)),
        retained_by_hysteresis=tuple(False for _ in range(count)),
    )

    return StateSnapshot(
        kernel_result=result,
        event_batch=CascadeEventBatch(time=time),
        history=CascadeHistory(),
        step_index=step_index,
        snapshot_id=snapshot_id,
    )


def make_observation(
    *,
    snapshot_id: str = "snapshot-1",
    time: float = 1.0,
    reserve: tuple[float, ...] = (0.2, 0.8),
    eta: float | None = None,
    alerts: tuple[ObserverAlert, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Observation:
    snapshot = make_snapshot(
        snapshot_id=snapshot_id,
        time=time,
        step_index=int(time),
    )
    observer = SimulationObserver()
    metrics = observer.observe(snapshot, reserve=reserve).metrics

    return Observation(
        snapshot=snapshot,
        metrics=metrics,
        eta=eta,
        alerts=alerts,
        metadata={} if metadata is None else metadata,
    )


def make_alert(
    *,
    kind: AlertKind = AlertKind.ETA_WARNING,
    level: AlertLevel = AlertLevel.WARNING,
    snapshot_id: str = "snapshot-1",
    time: float = 1.0,
    value: float | str | None = 0.2,
    threshold: float | None = 0.25,
    metadata: dict[str, Any] | None = None,
) -> ObserverAlert:
    return ObserverAlert(
        kind=kind,
        level=level,
        snapshot_id=snapshot_id,
        time=time,
        message=" Test alert ",
        value=value,
        threshold=threshold,
        metadata={} if metadata is None else metadata,
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_alert_level_values_are_stable() -> None:
    assert AlertLevel.INFO.value == "info"
    assert AlertLevel.WARNING.value == "warning"
    assert AlertLevel.CRITICAL.value == "critical"


def test_alert_kind_values_are_stable() -> None:
    assert AlertKind.ETA_WARNING.value == "eta_warning"
    assert AlertKind.ETA_CRITICAL.value == "eta_critical"
    assert AlertKind.EVENT_BURDEN.value == "event_burden"
    assert AlertKind.MAXIMUM_DELTA.value == "maximum_delta"
    assert AlertKind.D_FAST_CHANGED.value == "d_fast_changed"


# ---------------------------------------------------------------------------
# ObserverAlert
# ---------------------------------------------------------------------------


def test_observer_alert_normalizes_values() -> None:
    alert = ObserverAlert(
        kind=AlertKind.ETA_WARNING,
        level=AlertLevel.WARNING,
        snapshot_id="  s1  ",
        time="1.5",
        message="  warning  ",
        value="  A  ",
        threshold="0.25",
        metadata={" source ": "test"},
    )

    assert alert.snapshot_id == "s1"
    assert alert.time == pytest.approx(1.5)
    assert alert.message == "warning"
    assert alert.value == "A"
    assert alert.threshold == pytest.approx(0.25)
    assert alert.metadata == {"source": "test"}


def test_observer_alert_numeric_value_is_float() -> None:
    alert = make_alert(value="0.2", threshold=0.25)
    assert alert.value == "0.2"

    numeric = make_alert(value=2)
    assert numeric.value == pytest.approx(2.0)


def test_observer_alert_metadata_is_frozen_and_copied() -> None:
    source = {"source": "unit"}
    alert = make_alert(metadata=source)
    source["source"] = "changed"

    assert isinstance(alert.metadata, MappingProxyType)
    assert alert.metadata["source"] == "unit"

    with pytest.raises(TypeError):
        alert.metadata["new"] = "value"  # type: ignore[index]


def test_observer_alert_is_immutable() -> None:
    alert = make_alert()

    with pytest.raises(FrozenInstanceError):
        alert.time = 2.0  # type: ignore[misc]


def test_observer_alert_as_dict() -> None:
    alert = make_alert(metadata={"source": "test"})
    payload = alert.as_dict()

    assert payload == {
        "kind": "eta_warning",
        "level": "warning",
        "snapshot_id": "snapshot-1",
        "time": 1.0,
        "message": "Test alert",
        "value": 0.2,
        "threshold": 0.25,
        "metadata": {"source": "test"},
    }


@pytest.mark.parametrize("kind", ["eta_warning", None, object()])
def test_observer_alert_rejects_invalid_kind(kind) -> None:
    with pytest.raises(ObserverError):
        ObserverAlert(
            kind=kind,  # type: ignore[arg-type]
            level=AlertLevel.WARNING,
            snapshot_id="s1",
            time=1.0,
            message="message",
        )


@pytest.mark.parametrize("level", ["warning", None, object()])
def test_observer_alert_rejects_invalid_level(level) -> None:
    with pytest.raises(ObserverError):
        ObserverAlert(
            kind=AlertKind.ETA_WARNING,
            level=level,  # type: ignore[arg-type]
            snapshot_id="s1",
            time=1.0,
            message="message",
        )


@pytest.mark.parametrize("snapshot_id", ["", "   ", 1, None])
def test_observer_alert_rejects_invalid_snapshot_id(snapshot_id) -> None:
    with pytest.raises(ObserverError):
        ObserverAlert(
            kind=AlertKind.ETA_WARNING,
            level=AlertLevel.WARNING,
            snapshot_id=snapshot_id,  # type: ignore[arg-type]
            time=1.0,
            message="message",
        )


@pytest.mark.parametrize("time", [-1.0, np.nan, np.inf, -np.inf, True, "bad"])
def test_observer_alert_rejects_invalid_time(time) -> None:
    with pytest.raises(ObserverError):
        ObserverAlert(
            kind=AlertKind.ETA_WARNING,
            level=AlertLevel.WARNING,
            snapshot_id="s1",
            time=time,
            message="message",
        )


@pytest.mark.parametrize("message", ["", "   ", 1, None])
def test_observer_alert_rejects_invalid_message(message) -> None:
    with pytest.raises(ObserverError):
        ObserverAlert(
            kind=AlertKind.ETA_WARNING,
            level=AlertLevel.WARNING,
            snapshot_id="s1",
            time=1.0,
            message=message,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, np.nan, np.inf, -np.inf, "", "   "])
def test_observer_alert_rejects_invalid_value(value) -> None:
    with pytest.raises(ObserverError):
        make_alert(value=value)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf, True, "bad"])
def test_observer_alert_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ObserverError):
        make_alert(threshold=threshold)


@pytest.mark.parametrize(
    "metadata",
    [[], "bad", {1: "x"}, {"": "x"}, {"   ": "x"}],
)
def test_observer_alert_rejects_invalid_metadata(metadata) -> None:
    with pytest.raises(ObserverError):
        make_alert(metadata=metadata)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ObserverConfig
# ---------------------------------------------------------------------------


def test_observer_config_defaults() -> None:
    config = ObserverConfig()

    assert config.eta_warning_threshold == pytest.approx(0.25)
    assert config.eta_critical_threshold == pytest.approx(0.0)
    assert config.event_burden_threshold is None
    assert config.maximum_delta_threshold is None
    assert config.emit_d_fast_change is True
    assert config.emit_repeated_threshold_alerts is False


def test_observer_config_normalizes_numbers() -> None:
    config = ObserverConfig(
        eta_warning_threshold="0.3",
        eta_critical_threshold="-0.1",
        event_burden_threshold="2.0",
        maximum_delta_threshold="0.4",
    )

    assert config.eta_warning_threshold == pytest.approx(0.3)
    assert config.eta_critical_threshold == pytest.approx(-0.1)
    assert config.event_burden_threshold == pytest.approx(2.0)
    assert config.maximum_delta_threshold == pytest.approx(0.4)


def test_observer_config_is_immutable() -> None:
    config = ObserverConfig()

    with pytest.raises(FrozenInstanceError):
        config.eta_warning_threshold = 1.0  # type: ignore[misc]


def test_observer_config_rejects_critical_above_warning() -> None:
    with pytest.raises(ObserverError):
        ObserverConfig(
            eta_warning_threshold=0.1,
            eta_critical_threshold=0.2,
        )


@pytest.mark.parametrize(
    "field_name",
    ["eta_warning_threshold", "eta_critical_threshold"],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "bad"])
def test_observer_config_rejects_invalid_eta_threshold(
    field_name,
    value,
) -> None:
    kwargs = {
        "eta_warning_threshold": 0.25,
        "eta_critical_threshold": 0.0,
    }
    kwargs[field_name] = value

    with pytest.raises(ObserverError):
        ObserverConfig(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    ["event_burden_threshold", "maximum_delta_threshold"],
)
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf, True, "bad"])
def test_observer_config_rejects_invalid_optional_threshold(
    field_name,
    value,
) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ObserverError):
        ObserverConfig(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    ["emit_d_fast_change", "emit_repeated_threshold_alerts"],
)
@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_observer_config_requires_boolean_flags(
    field_name,
    value,
) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ObserverError):
        ObserverConfig(**kwargs)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def test_observation_properties() -> None:
    alert = make_alert()
    observation = make_observation(
        eta=0.2,
        alerts=(alert,),
        metadata={"source": "test"},
    )

    assert observation.snapshot_id == "snapshot-1"
    assert observation.time == pytest.approx(1.0)
    assert observation.d_fast_id == "A"
    assert observation.has_alerts is True
    assert observation.highest_alert_level is AlertLevel.WARNING
    assert observation.metadata == {"source": "test"}


def test_observation_without_alerts() -> None:
    observation = make_observation()

    assert observation.has_alerts is False
    assert observation.highest_alert_level is None


def test_observation_highest_alert_level() -> None:
    alerts = (
        make_alert(level=AlertLevel.INFO),
        make_alert(
            kind=AlertKind.ETA_CRITICAL,
            level=AlertLevel.CRITICAL,
        ),
        make_alert(level=AlertLevel.WARNING),
    )
    observation = make_observation(alerts=alerts)

    assert observation.highest_alert_level is AlertLevel.CRITICAL


def test_observation_converts_alert_list_to_tuple() -> None:
    observation = make_observation()
    converted = Observation(
        snapshot=observation.snapshot,
        metrics=observation.metrics,
        alerts=[make_alert()],  # type: ignore[arg-type]
    )

    assert isinstance(converted.alerts, tuple)


def test_observation_metadata_is_frozen() -> None:
    observation = make_observation(metadata={"source": "test"})

    assert isinstance(observation.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        observation.metadata["x"] = 1  # type: ignore[index]


def test_observation_as_dict() -> None:
    observation = make_observation(eta=0.2)
    payload = observation.as_dict()

    assert payload["snapshot_id"] == "snapshot-1"
    assert payload["time"] == pytest.approx(1.0)
    assert payload["eta"] == pytest.approx(0.2)
    assert payload["d_fast_id"] == "A"
    assert isinstance(payload["metrics"], dict)
    assert payload["alerts"] == []


def test_observation_is_immutable() -> None:
    observation = make_observation()

    with pytest.raises(FrozenInstanceError):
        observation.eta = 0.5  # type: ignore[misc]


@pytest.mark.parametrize("snapshot", [None, object(), {}])
def test_observation_rejects_invalid_snapshot(snapshot) -> None:
    valid = make_observation()

    with pytest.raises(ObserverError):
        Observation(
            snapshot=snapshot,  # type: ignore[arg-type]
            metrics=valid.metrics,
        )


@pytest.mark.parametrize("metrics", [None, object(), {}])
def test_observation_rejects_invalid_metrics(metrics) -> None:
    valid = make_observation()

    with pytest.raises(ObserverError):
        Observation(
            snapshot=valid.snapshot,
            metrics=metrics,  # type: ignore[arg-type]
        )


def test_observation_rejects_mismatched_snapshot_id() -> None:
    first = make_observation(snapshot_id="s1")
    second = make_observation(snapshot_id="s2")

    with pytest.raises(ObserverError):
        Observation(
            snapshot=first.snapshot,
            metrics=second.metrics,
        )


@pytest.mark.parametrize("eta", [np.nan, np.inf, -np.inf, True, "bad"])
def test_observation_rejects_invalid_eta(eta) -> None:
    valid = make_observation()

    with pytest.raises(ObserverError):
        Observation(
            snapshot=valid.snapshot,
            metrics=valid.metrics,
            eta=eta,
        )


def test_observation_rejects_invalid_alert_member() -> None:
    valid = make_observation()

    with pytest.raises(ObserverError):
        Observation(
            snapshot=valid.snapshot,
            metrics=valid.metrics,
            alerts=("bad",),  # type: ignore[arg-type]
        )


def test_observation_rejects_alert_for_other_snapshot() -> None:
    valid = make_observation(snapshot_id="s1")
    alert = make_alert(snapshot_id="s2")

    with pytest.raises(ObserverError):
        Observation(
            snapshot=valid.snapshot,
            metrics=valid.metrics,
            alerts=(alert,),
        )


# ---------------------------------------------------------------------------
# ObservationSeries
# ---------------------------------------------------------------------------


def test_empty_observation_series() -> None:
    series = ObservationSeries(tuple())

    assert len(series) == 0
    assert series.first is None
    assert series.final is None
    assert series.alerts == ()
    assert series.eta_series == ()
    assert series.d_fast_series == ()
    assert isinstance(series.metrics, SimulationMetrics)


def test_observation_series_accepts_list() -> None:
    series = ObservationSeries(
        [
            make_observation(snapshot_id="s1", time=1.0),
            make_observation(snapshot_id="s2", time=2.0),
        ]
    )

    assert isinstance(series.observations, tuple)
    assert len(series) == 2


def test_observation_series_properties() -> None:
    first = make_observation(
        snapshot_id="s1",
        time=1.0,
        eta=0.4,
    )
    second_alert = make_alert(
        snapshot_id="s2",
        time=2.0,
    )
    second = make_observation(
        snapshot_id="s2",
        time=2.0,
        reserve=(0.8, 0.2),
        eta=0.1,
        alerts=(second_alert,),
    )
    series = ObservationSeries((first, second))

    assert tuple(series) == (first, second)
    assert series.first is first
    assert series.final is second
    assert series.alerts == (second_alert,)
    assert series.eta_series == ((1.0, 0.4), (2.0, 0.1))
    assert series.d_fast_series == ((1.0, "A"), (2.0, "B"))
    assert len(series.metrics) == 2


def test_observation_series_omits_missing_eta() -> None:
    series = ObservationSeries(
        (
            make_observation(snapshot_id="s1", time=1.0),
            make_observation(snapshot_id="s2", time=2.0, eta=0.2),
        )
    )

    assert series.eta_series == ((2.0, 0.2),)


def test_observation_series_as_dict() -> None:
    series = ObservationSeries(
        (make_observation(snapshot_id="s1", time=1.0),)
    )
    payload = series.as_dict()

    assert payload["observation_count"] == 1
    assert payload["alert_count"] == 0
    assert len(payload["observations"]) == 1
    assert isinstance(payload["metrics"], dict)


def test_observation_series_rejects_invalid_member() -> None:
    with pytest.raises(ObserverError):
        ObservationSeries(("bad",))  # type: ignore[arg-type]


def test_observation_series_rejects_non_chronological_order() -> None:
    with pytest.raises(ObserverError):
        ObservationSeries(
            (
                make_observation(snapshot_id="s2", time=2.0),
                make_observation(snapshot_id="s1", time=1.0),
            )
        )


def test_observation_series_allows_equal_times() -> None:
    series = ObservationSeries(
        (
            make_observation(snapshot_id="s1", time=1.0),
            make_observation(snapshot_id="s2", time=1.0),
        )
    )
    assert len(series) == 2


def test_observation_series_allows_float_tolerance() -> None:
    series = ObservationSeries(
        (
            make_observation(snapshot_id="s1", time=1.0),
            make_observation(
                snapshot_id="s2",
                time=1.0 - 5e-13,
            ),
        )
    )
    assert len(series) == 2


def test_observation_series_rejects_duplicate_snapshot_ids() -> None:
    with pytest.raises(ObserverError):
        ObservationSeries(
            (
                make_observation(snapshot_id="same", time=1.0),
                make_observation(snapshot_id="same", time=2.0),
            )
        )


# ---------------------------------------------------------------------------
# SimulationObserver construction and state
# ---------------------------------------------------------------------------


def test_simulation_observer_defaults() -> None:
    observer = SimulationObserver()

    assert isinstance(observer.config, ObserverConfig)
    assert observer.observations == ()
    assert observer.latest is None
    assert len(observer) == 0
    assert len(observer.series) == 0


def test_simulation_observer_accepts_config_and_hooks() -> None:
    config = ObserverConfig(eta_warning_threshold=0.3)
    hook = lambda observation: None
    observer = SimulationObserver(config, hooks=(hook,))

    assert observer.config is config


@pytest.mark.parametrize("config", [object(), {}, "bad"])
def test_simulation_observer_rejects_invalid_config(config) -> None:
    with pytest.raises(ObserverError):
        SimulationObserver(config)  # type: ignore[arg-type]


@pytest.mark.parametrize("hooks", ["bad", b"bad", None, 1])
def test_simulation_observer_rejects_invalid_hooks_container(hooks) -> None:
    with pytest.raises(ObserverError):
        SimulationObserver(hooks=hooks)  # type: ignore[arg-type]


def test_simulation_observer_rejects_non_callable_hook() -> None:
    with pytest.raises(ObserverError):
        SimulationObserver(hooks=(object(),))


def test_simulation_observer_reset() -> None:
    observer = SimulationObserver()
    observer.observe(make_snapshot())
    assert len(observer) == 1

    observer.reset()

    assert len(observer) == 0
    assert observer.latest is None
    assert observer.observations == ()


# ---------------------------------------------------------------------------
# observe
# ---------------------------------------------------------------------------


def test_observe_calculates_metrics_and_records_observation() -> None:
    observer = SimulationObserver()
    snapshot = make_snapshot()

    observation = observer.observe(
        snapshot,
        reserve=(0.2, 0.8),
        metadata={"source": "test"},
    )

    assert observation.snapshot is snapshot
    assert isinstance(observation.metrics, SnapshotMetrics)
    assert observation.metrics.d_fast.winner_id == "A"
    assert observation.metadata == {"source": "test"}
    assert observer.latest is observation
    assert observer.observations == (observation,)
    assert len(observer) == 1


def test_observe_default_reserve_uses_activation_after() -> None:
    observer = SimulationObserver()
    snapshot = make_snapshot(
        activation_after=(0.9, 0.1),
    )

    observation = observer.observe(snapshot)

    # Default reserve = 1 - abs(activation_after).
    assert observation.d_fast_id == "A"


def test_observe_calculates_eta() -> None:
    observer = SimulationObserver()
    observation = observer.observe(
        make_snapshot(),
        operator=[[0.5, 0.0], [0.0, 0.25]],
        lambda_critical=1.0,
    )

    assert observation.eta == pytest.approx(0.5)


def test_observe_stable_eta_has_no_eta_alert() -> None:
    observer = SimulationObserver()
    observation = observer.observe(
        make_snapshot(),
        operator=[[0.5, 0.0], [0.0, 0.25]],
        lambda_critical=1.0,
    )

    assert observation.alerts == ()


def test_observe_warning_eta_emits_warning() -> None:
    observer = SimulationObserver()
    observation = observer.observe(
        make_snapshot(),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert observation.eta == pytest.approx(0.2)
    assert len(observation.alerts) == 1
    alert = observation.alerts[0]
    assert alert.kind is AlertKind.ETA_WARNING
    assert alert.level is AlertLevel.WARNING
    assert alert.threshold == pytest.approx(0.25)


def test_observe_critical_eta_emits_critical() -> None:
    observer = SimulationObserver()
    observation = observer.observe(
        make_snapshot(),
        operator=[[1.0, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert observation.eta == pytest.approx(0.0)
    assert observation.alerts[0].kind is AlertKind.ETA_CRITICAL
    assert observation.alerts[0].level is AlertLevel.CRITICAL


def test_observe_negative_eta_is_critical() -> None:
    observer = SimulationObserver()
    observation = observer.observe(
        make_snapshot(),
        operator=[[1.5, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert observation.eta == pytest.approx(-0.5)
    assert observation.alerts[0].kind is AlertKind.ETA_CRITICAL


def test_eta_warning_not_repeated_in_same_zone_by_default() -> None:
    observer = SimulationObserver()
    first = observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        operator=[[0.85, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert len(first.alerts) == 1
    assert second.alerts == ()


def test_eta_zone_transition_emits_new_alert() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        operator=[[1.0, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert len(second.alerts) == 1
    assert second.alerts[0].kind is AlertKind.ETA_CRITICAL


def test_eta_alert_repeats_when_configured() -> None:
    observer = SimulationObserver(
        ObserverConfig(emit_repeated_threshold_alerts=True)
    )
    first = observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        operator=[[0.85, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert len(first.alerts) == 1
    assert len(second.alerts) == 1


def test_eta_alert_can_reemit_after_return_to_stable() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )
    stable = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        operator=[[0.5, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )
    warning = observer.observe(
        make_snapshot(snapshot_id="s3", time=3.0, step_index=3),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert stable.alerts == ()
    assert len(warning.alerts) == 1
    assert warning.alerts[0].kind is AlertKind.ETA_WARNING


def test_event_burden_threshold_zero_emits_alert() -> None:
    observer = SimulationObserver(
        ObserverConfig(event_burden_threshold=0.0)
    )
    observation = observer.observe(make_snapshot())

    assert any(
        alert.kind is AlertKind.EVENT_BURDEN
        for alert in observation.alerts
    )


def test_event_burden_alert_not_repeated_by_default() -> None:
    observer = SimulationObserver(
        ObserverConfig(event_burden_threshold=0.0)
    )
    first = observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0)
    )
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2)
    )

    assert any(
        alert.kind is AlertKind.EVENT_BURDEN
        for alert in first.alerts
    )
    assert not any(
        alert.kind is AlertKind.EVENT_BURDEN
        for alert in second.alerts
    )


def test_maximum_delta_threshold_emits_alert() -> None:
    observer = SimulationObserver(
        ObserverConfig(maximum_delta_threshold=0.5)
    )
    observation = observer.observe(
        make_snapshot(
            activation_before=(0.0, 0.0),
            activation_after=(0.5, 0.1),
        )
    )

    alert = next(
        item
        for item in observation.alerts
        if item.kind is AlertKind.MAXIMUM_DELTA
    )
    assert alert.value == pytest.approx(0.5)
    assert alert.threshold == pytest.approx(0.5)


def test_maximum_delta_below_threshold_has_no_alert() -> None:
    observer = SimulationObserver(
        ObserverConfig(maximum_delta_threshold=0.6)
    )
    observation = observer.observe(
        make_snapshot(
            activation_before=(0.0, 0.0),
            activation_after=(0.5, 0.1),
        )
    )

    assert not any(
        alert.kind is AlertKind.MAXIMUM_DELTA
        for alert in observation.alerts
    )


def test_d_fast_change_emits_info_alert() -> None:
    observer = SimulationObserver()
    first = observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        reserve=(0.1, 0.9),
    )
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        reserve=(0.9, 0.1),
    )

    assert first.d_fast_id == "A"
    assert second.d_fast_id == "B"

    alert = next(
        item
        for item in second.alerts
        if item.kind is AlertKind.D_FAST_CHANGED
    )
    assert alert.level is AlertLevel.INFO
    assert alert.value == "B"
    assert alert.metadata["previous_plane_id"] == "A"
    assert alert.metadata["current_plane_id"] == "B"


def test_d_fast_change_can_be_disabled() -> None:
    observer = SimulationObserver(
        ObserverConfig(emit_d_fast_change=False)
    )
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        reserve=(0.1, 0.9),
    )
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        reserve=(0.9, 0.1),
    )

    assert not any(
        alert.kind is AlertKind.D_FAST_CHANGED
        for alert in second.alerts
    )


def test_d_fast_same_winner_has_no_change_alert() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        reserve=(0.1, 0.9),
    )
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        reserve=(0.2, 0.8),
    )

    assert not any(
        alert.kind is AlertKind.D_FAST_CHANGED
        for alert in second.alerts
    )


def test_observe_calls_hooks_after_recording() -> None:
    calls: list[tuple[int, str]] = []

    def hook(observation: Observation) -> None:
        calls.append((len(observer), observation.snapshot_id))

    observer = SimulationObserver(hooks=(hook,))
    observer.observe(make_snapshot())

    assert calls == [(1, "snapshot-1")]


def test_observe_calls_hooks_in_order() -> None:
    calls: list[str] = []

    def first(_: Observation) -> None:
        calls.append("first")

    def second(_: Observation) -> None:
        calls.append("second")

    observer = SimulationObserver(hooks=(first, second))
    observer.observe(make_snapshot())

    assert calls == ["first", "second"]


def test_hook_exception_propagates_but_observation_is_recorded() -> None:
    def failing(_: Observation) -> None:
        raise RuntimeError("hook failed")

    observer = SimulationObserver(hooks=(failing,))

    with pytest.raises(RuntimeError, match="hook failed"):
        observer.observe(make_snapshot())

    assert len(observer) == 1


@pytest.mark.parametrize("snapshot", [None, object(), {}, []])
def test_observe_rejects_invalid_snapshot(snapshot) -> None:
    observer = SimulationObserver()

    with pytest.raises(ObserverError):
        observer.observe(snapshot)  # type: ignore[arg-type]


def test_observe_rejects_lambda_without_operator() -> None:
    observer = SimulationObserver()

    with pytest.raises(ObserverError):
        observer.observe(
            make_snapshot(),
            lambda_critical=1.0,
        )


def test_observe_rejects_operator_without_lambda() -> None:
    observer = SimulationObserver()

    with pytest.raises(ObserverError):
        observer.observe(
            make_snapshot(),
            operator=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_observe_wraps_invalid_reserve_error() -> None:
    observer = SimulationObserver()

    with pytest.raises(
        ObserverError,
        match="Cannot calculate snapshot metrics",
    ):
        observer.observe(
            make_snapshot(),
            reserve=(0.1,),
        )


def test_observe_wraps_invalid_operator_error() -> None:
    observer = SimulationObserver()

    with pytest.raises(
        ObserverError,
        match="Cannot calculate spectral coherence",
    ):
        observer.observe(
            make_snapshot(),
            operator=[[1.0, 2.0, 3.0]],
            lambda_critical=1.0,
        )


def test_observe_rejects_duplicate_latest_snapshot_id() -> None:
    observer = SimulationObserver()
    snapshot = make_snapshot(snapshot_id="same")
    observer.observe(snapshot)

    with pytest.raises(ObserverError):
        observer.observe(snapshot)


def test_observe_rejects_duplicate_earlier_snapshot_id() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0)
    )
    observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2)
    )

    with pytest.raises(ObserverError):
        observer.observe(
            make_snapshot(snapshot_id="s1", time=3.0, step_index=3)
        )


def test_observe_rejects_non_chronological_snapshot() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="later", time=2.0, step_index=2)
    )

    with pytest.raises(ObserverError):
        observer.observe(
            make_snapshot(snapshot_id="earlier", time=1.0)
        )


def test_observe_allows_equal_times() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0)
    )
    observer.observe(
        make_snapshot(snapshot_id="s2", time=1.0, step_index=2)
    )

    assert len(observer) == 2


def test_observe_allows_tiny_time_tolerance() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0)
    )
    observer.observe(
        make_snapshot(
            snapshot_id="s2",
            time=1.0 - 5e-13,
            step_index=2,
        )
    )

    assert len(observer) == 2


def test_reset_clears_eta_zone_state() -> None:
    observer = SimulationObserver()
    first = observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )
    observer.reset()
    second = observer.observe(
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        operator=[[0.8, 0.0], [0.0, 0.1]],
        lambda_critical=1.0,
    )

    assert len(first.alerts) == 1
    assert len(second.alerts) == 1


# ---------------------------------------------------------------------------
# observe_many
# ---------------------------------------------------------------------------


def test_observe_many_records_all_snapshots() -> None:
    snapshots = (
        make_snapshot(snapshot_id="s1", time=1.0),
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
    )
    observer = SimulationObserver()
    series = observer.observe_many(snapshots)

    assert len(series) == 2
    assert len(observer) == 2
    assert series.final is observer.latest


def test_observe_many_applies_reserves() -> None:
    snapshots = (
        make_snapshot(snapshot_id="s1", time=1.0),
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
    )
    series = SimulationObserver().observe_many(
        snapshots,
        reserves=((0.1, 0.9), (0.9, 0.1)),
    )

    assert series.d_fast_series == ((1.0, "A"), (2.0, "B"))


def test_observe_many_applies_operators() -> None:
    snapshots = (
        make_snapshot(snapshot_id="s1", time=1.0),
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
    )
    series = SimulationObserver().observe_many(
        snapshots,
        operators=(
            [[0.5, 0.0], [0.0, 0.1]],
            [[0.8, 0.0], [0.0, 0.1]],
        ),
        lambda_critical=1.0,
    )

    assert len(series.eta_series) == 2
    assert series.eta_series[0][0] == pytest.approx(1.0)
    assert series.eta_series[0][1] == pytest.approx(0.5)
    assert series.eta_series[1][0] == pytest.approx(2.0)
    assert series.eta_series[1][1] == pytest.approx(0.2)


def test_observe_many_accepts_empty_sequence() -> None:
    observer = SimulationObserver()
    series = observer.observe_many([])

    assert len(series) == 0
    assert len(observer) == 0


@pytest.mark.parametrize("snapshots", ["bad", b"bad", None, 1])
def test_observe_many_rejects_invalid_snapshots_container(
    snapshots,
) -> None:
    observer = SimulationObserver()

    with pytest.raises(ObserverError):
        observer.observe_many(snapshots)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["reserves", "operators"])
def test_observe_many_rejects_mismatched_optional_sequence(name) -> None:
    snapshots = (
        make_snapshot(snapshot_id="s1", time=1.0),
        make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
    )
    kwargs = {name: (None,)}

    with pytest.raises(ObserverError):
        SimulationObserver().observe_many(
            snapshots,
            **kwargs,
        )


@pytest.mark.parametrize("name", ["reserves", "operators"])
@pytest.mark.parametrize("value", ["bad", b"bad", 1])
def test_observe_many_rejects_invalid_optional_container(
    name,
    value,
) -> None:
    kwargs = {name: value}

    with pytest.raises(ObserverError):
        SimulationObserver().observe_many(
            [make_snapshot()],
            **kwargs,
        )


def test_observe_many_requires_lambda_for_operators() -> None:
    with pytest.raises(ObserverError):
        SimulationObserver().observe_many(
            [make_snapshot()],
            operators=(
                [[1.0, 0.0], [0.0, 1.0]],
            ),
        )


def test_observe_many_requires_lambda_when_operator_sequence_present() -> None:
    with pytest.raises(ObserverError):
        SimulationObserver().observe_many(
            [make_snapshot()],
            operators=(None,),
        )


def test_observe_many_appends_to_existing_observer() -> None:
    observer = SimulationObserver()
    observer.observe(
        make_snapshot(snapshot_id="s1", time=1.0)
    )
    series = observer.observe_many(
        (
            make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
            make_snapshot(snapshot_id="s3", time=3.0, step_index=3),
        )
    )

    assert len(series) == 3
    assert series.first.snapshot_id == "s1"
    assert series.final.snapshot_id == "s3"


# ---------------------------------------------------------------------------
# observe_snapshots convenience function
# ---------------------------------------------------------------------------


def test_observe_snapshots_returns_series() -> None:
    series = observe_snapshots(
        (
            make_snapshot(snapshot_id="s1", time=1.0),
            make_snapshot(snapshot_id="s2", time=2.0, step_index=2),
        )
    )

    assert isinstance(series, ObservationSeries)
    assert len(series) == 2


def test_observe_snapshots_forwards_config() -> None:
    series = observe_snapshots(
        [make_snapshot()],
        config=ObserverConfig(maximum_delta_threshold=0.1),
    )

    assert any(
        alert.kind is AlertKind.MAXIMUM_DELTA
        for alert in series.final.alerts
    )


def test_observe_snapshots_forwards_reserves() -> None:
    series = observe_snapshots(
        [make_snapshot()],
        reserves=((0.9, 0.1),),
    )

    assert series.final.d_fast_id == "B"


def test_observe_snapshots_forwards_operators() -> None:
    series = observe_snapshots(
        [make_snapshot()],
        operators=(
            [[0.8, 0.0], [0.0, 0.1]],
        ),
        lambda_critical=1.0,
    )

    assert series.final.eta == pytest.approx(0.2)
    assert series.final.alerts[0].kind is AlertKind.ETA_WARNING

