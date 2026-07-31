"""Tests for ROIF predictor."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from roif.analyzer import CascadePhase
from roif.cascade_event import CascadeEventBatch
from roif.history import CascadeHistory
from roif.observer import Observation, ObservationSeries, SimulationObserver
from roif.plane_kernel import PlaneKernelResult
from roif.predictor import (
    CascadePrediction,
    CascadePredictor,
    CrossingKind,
    DFastForecast,
    ForecastInterval,
    PredictionStatus,
    PredictorConfig,
    PredictorError,
    StabilityForecast,
    ThresholdCrossing,
    predict_observations,
)
from roif.state_snapshot import StateSnapshot


def make_snapshot(
    *,
    snapshot_id: str,
    time: float,
    step_index: int,
    delta: float = 0.1,
) -> StateSnapshot:
    result = PlaneKernelResult(
        plane_ids=("A", "B"),
        time=time,
        dt=1.0,
        activation_before=(0.0, 0.0),
        interaction=(0.0, 0.0),
        rate=(delta, 0.0),
        activation_after=(delta, 0.0),
        clipped=(False, False),
        retained_by_hysteresis=(False, False),
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
    snapshot_id: str,
    time: float,
    step_index: int,
    eta: float | None,
    reserve: tuple[float, float] = (0.1, 0.9),
) -> Observation:
    snapshot = make_snapshot(
        snapshot_id=snapshot_id,
        time=time,
        step_index=step_index,
    )
    metrics = SimulationObserver().observe(
        snapshot,
        reserve=reserve,
    ).metrics
    return Observation(
        snapshot=snapshot,
        metrics=metrics,
        eta=eta,
    )


def make_series(
    etas: tuple[float | None, ...],
    *,
    times: tuple[float, ...] | None = None,
    reserves: tuple[tuple[float, float], ...] | None = None,
) -> ObservationSeries:
    if times is None:
        times = tuple(float(index) for index in range(1, len(etas) + 1))
    if reserves is None:
        reserves = tuple((0.1, 0.9) for _ in etas)

    return ObservationSeries(
        tuple(
            make_observation(
                snapshot_id=f"s{index}",
                time=times[index - 1],
                step_index=index,
                eta=eta,
                reserve=reserves[index - 1],
            )
            for index, eta in enumerate(etas, start=1)
        )
    )


def make_prediction(**overrides: Any) -> CascadePrediction:
    values = {
        "status": PredictionStatus.AVAILABLE,
        "stability": StabilityForecast.STABLE,
        "source_observation_count": 2,
        "source_point_count": 2,
        "last_time": 2.0,
        "forecast_time": 3.0,
        "eta": ForecastInterval(0.5, 0.4, 0.6, 0.05),
        "eta_slope": 0.0,
        "critical_probability": 0.1,
        "first_crossing": None,
        "d_fast": DFastForecast("A", 1.0, 0.0, 2),
        "source_phase": CascadePhase.QUIESCENT,
        "summary": " Forecast summary. ",
        "metadata": {"source": "test"},
    }
    values.update(overrides)
    return CascadePrediction(**values)


def test_prediction_status_values_are_stable() -> None:
    assert tuple(item.value for item in PredictionStatus) == (
        "available",
        "insufficient_data",
        "eta_unavailable",
    )


def test_stability_forecast_values_are_stable() -> None:
    assert tuple(item.value for item in StabilityForecast) == (
        "stable",
        "warning",
        "critical",
        "unknown",
    )


def test_crossing_kind_values_are_stable() -> None:
    assert tuple(item.value for item in CrossingKind) == (
        "warning",
        "critical",
    )


def test_predictor_config_defaults() -> None:
    config = PredictorConfig()
    assert config.horizon == pytest.approx(1.0)
    assert config.confidence_z == pytest.approx(1.96)
    assert config.minimum_points == 2
    assert config.maximum_points is None
    assert config.eta_warning_threshold == pytest.approx(0.25)
    assert config.eta_critical_threshold == pytest.approx(0.0)


def test_predictor_config_normalizes_numbers() -> None:
    config = PredictorConfig(
        horizon="2",
        confidence_z="1",
        eta_warning_threshold="0.4",
        eta_critical_threshold="-0.1",
        slope_tolerance="0.01",
        probability_scale="0.2",
    )
    assert config.horizon == pytest.approx(2.0)
    assert config.confidence_z == pytest.approx(1.0)
    assert config.probability_scale == pytest.approx(0.2)


def test_predictor_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        PredictorConfig().horizon = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("field", ("horizon", "probability_scale"))
@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), True, "bad"))
def test_config_rejects_nonpositive_fields(field: str, value: Any) -> None:
    with pytest.raises(PredictorError):
        PredictorConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "confidence_z",
        "eta_warning_threshold",
        "eta_critical_threshold",
        "slope_tolerance",
    ),
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "bad"))
def test_config_rejects_invalid_numeric_fields(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(PredictorError):
        PredictorConfig(**{field: value})


@pytest.mark.parametrize("field", ("confidence_z", "slope_tolerance"))
def test_config_rejects_negative_nonnegative_fields(field: str) -> None:
    with pytest.raises(PredictorError):
        PredictorConfig(**{field: -0.1})


def test_config_rejects_critical_above_warning() -> None:
    with pytest.raises(PredictorError):
        PredictorConfig(
            eta_warning_threshold=0.0,
            eta_critical_threshold=0.1,
        )


@pytest.mark.parametrize("value", (True, 1, 0, -1, 1.5, "2"))
def test_config_rejects_invalid_minimum_points(value: Any) -> None:
    with pytest.raises(PredictorError):
        PredictorConfig(minimum_points=value)


@pytest.mark.parametrize("value", (True, 1, 1.5, "3"))
def test_config_rejects_invalid_maximum_points(value: Any) -> None:
    with pytest.raises(PredictorError):
        PredictorConfig(minimum_points=2, maximum_points=value)


def test_forecast_interval_properties() -> None:
    interval = ForecastInterval("0.5", "0.4", "0.6", "0.1")
    assert interval.value == pytest.approx(0.5)
    assert interval.lower == pytest.approx(0.4)
    assert interval.upper == pytest.approx(0.6)
    assert interval.standard_error == pytest.approx(0.1)


def test_forecast_interval_as_dict() -> None:
    assert ForecastInterval(1, 0, 2, 0.5).as_dict() == {
        "value": 1.0,
        "lower": 0.0,
        "upper": 2.0,
        "standard_error": 0.5,
    }


def test_forecast_interval_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        ForecastInterval(1, 0, 2, 0.5).value = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "args",
    (
        (0.5, 0.6, 0.7, 0.1),
        (0.5, 0.3, 0.4, 0.1),
        (0.5, 0.4, 0.6, -0.1),
    ),
)
def test_forecast_interval_rejects_invalid_order(args: tuple[Any, ...]) -> None:
    with pytest.raises(PredictorError):
        ForecastInterval(*args)


@pytest.mark.parametrize("field", ("value", "lower", "upper", "standard_error"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "bad"))
def test_forecast_interval_rejects_invalid_numbers(
    field: str,
    value: Any,
) -> None:
    values = {
        "value": 0.5,
        "lower": 0.4,
        "upper": 0.6,
        "standard_error": 0.1,
    }
    values[field] = value
    with pytest.raises(PredictorError):
        ForecastInterval(**values)


def test_threshold_crossing_properties() -> None:
    crossing = ThresholdCrossing(
        CrossingKind.WARNING,
        0.25,
        3.0,
        1.0,
    )
    assert crossing.kind is CrossingKind.WARNING
    assert crossing.threshold == pytest.approx(0.25)
    assert crossing.time_from_last == pytest.approx(1.0)


def test_threshold_crossing_as_dict() -> None:
    payload = ThresholdCrossing(
        CrossingKind.CRITICAL,
        0.0,
        4.0,
        2.0,
    ).as_dict()
    assert payload["kind"] == "critical"
    assert payload["time"] == pytest.approx(4.0)


@pytest.mark.parametrize("kind", ("warning", None, 1))
def test_threshold_crossing_rejects_invalid_kind(kind: Any) -> None:
    with pytest.raises(PredictorError):
        ThresholdCrossing(kind, 0.25, 3.0, 1.0)


@pytest.mark.parametrize("value", (-1.0, float("nan"), True, "bad"))
def test_threshold_crossing_rejects_invalid_delta(value: Any) -> None:
    with pytest.raises(PredictorError):
        ThresholdCrossing(
            CrossingKind.WARNING,
            0.25,
            3.0,
            value,
        )


def test_d_fast_forecast_properties() -> None:
    forecast = DFastForecast(" A ", 0.75, 0.25, 4)
    assert forecast.plane_id == "A"
    assert forecast.confidence == pytest.approx(0.75)
    assert forecast.switch_rate == pytest.approx(0.25)


def test_d_fast_forecast_as_dict() -> None:
    assert DFastForecast(None, 0, 0, 0).as_dict() == {
        "plane_id": None,
        "confidence": 0.0,
        "switch_rate": 0.0,
        "observed_count": 0,
    }


@pytest.mark.parametrize("plane_id", ("", "   ", 1))
def test_d_fast_forecast_rejects_invalid_plane_id(plane_id: Any) -> None:
    with pytest.raises(PredictorError):
        DFastForecast(plane_id, 1.0, 0.0, 1)


@pytest.mark.parametrize("field", ("confidence", "switch_rate"))
@pytest.mark.parametrize(
    "value",
    (-0.1, 1.1, float("nan"), True, "bad"),
)
def test_d_fast_forecast_rejects_invalid_rates(
    field: str,
    value: Any,
) -> None:
    values = {
        "plane_id": "A",
        "confidence": 1.0,
        "switch_rate": 0.0,
        "observed_count": 1,
    }
    values[field] = value
    with pytest.raises(PredictorError):
        DFastForecast(**values)


@pytest.mark.parametrize("value", (-1, True, 1.5, "1"))
def test_d_fast_forecast_rejects_invalid_count(value: Any) -> None:
    with pytest.raises(PredictorError):
        DFastForecast("A", 1.0, 0.0, value)


def test_cascade_prediction_properties() -> None:
    prediction = make_prediction()
    assert prediction.summary == "Forecast summary."
    assert prediction.is_available
    assert not prediction.predicts_critical
    assert isinstance(prediction.metadata, MappingProxyType)


def test_cascade_prediction_critical_property() -> None:
    prediction = make_prediction(
        stability=StabilityForecast.CRITICAL
    )
    assert prediction.predicts_critical


def test_cascade_prediction_as_dict() -> None:
    payload = make_prediction().as_dict()
    assert payload["status"] == "available"
    assert payload["stability"] == "stable"
    assert payload["eta"]["value"] == pytest.approx(0.5)
    assert payload["d_fast"]["plane_id"] == "A"
    assert payload["source_phase"] == "quiescent"


def test_prediction_metadata_is_copied_and_frozen() -> None:
    source = {"experiment": "E1"}
    prediction = make_prediction(metadata=source)
    source["experiment"] = "changed"
    assert prediction.metadata["experiment"] == "E1"
    with pytest.raises(TypeError):
        prediction.metadata["x"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "available"),
        ("stability", "stable"),
        ("source_phase", "quiescent"),
        ("eta", "bad"),
        ("first_crossing", "bad"),
        ("d_fast", "bad"),
    ),
)
def test_prediction_rejects_invalid_typed_fields(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(PredictorError):
        make_prediction(**{field: value})


@pytest.mark.parametrize(
    "field",
    ("source_observation_count", "source_point_count"),
)
@pytest.mark.parametrize("value", (-1, True, 1.5, "1"))
def test_prediction_rejects_invalid_counts(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(PredictorError):
        make_prediction(**{field: value})


@pytest.mark.parametrize("value", (-0.1, 1.1, float("nan"), True, "bad"))
def test_prediction_rejects_invalid_probability(value: Any) -> None:
    with pytest.raises(PredictorError):
        make_prediction(critical_probability=value)


@pytest.mark.parametrize("summary", ("", "   ", None, 1))
def test_prediction_rejects_invalid_summary(summary: Any) -> None:
    with pytest.raises(PredictorError):
        make_prediction(summary=summary)


@pytest.mark.parametrize("metadata", ("bad", {1: "bad"}, {" ": 1}))
def test_prediction_rejects_invalid_metadata(metadata: Any) -> None:
    with pytest.raises(PredictorError):
        make_prediction(metadata=metadata)


def test_predictor_defaults() -> None:
    predictor = CascadePredictor()
    assert isinstance(predictor.config, PredictorConfig)


def test_predictor_accepts_config() -> None:
    config = PredictorConfig(horizon=2.0)
    assert CascadePredictor(config).config is config


@pytest.mark.parametrize("config", ({}, "bad", 1))
def test_predictor_rejects_invalid_config(config: Any) -> None:
    with pytest.raises(PredictorError):
        CascadePredictor(config)


@pytest.mark.parametrize("series", (None, (), "bad"))
def test_predict_rejects_invalid_series(series: Any) -> None:
    with pytest.raises(PredictorError):
        CascadePredictor().predict(series)


def test_predict_empty_series() -> None:
    prediction = CascadePredictor().predict(ObservationSeries(()))
    assert prediction.status is PredictionStatus.INSUFFICIENT_DATA
    assert prediction.stability is StabilityForecast.UNKNOWN
    assert prediction.source_observation_count == 0
    assert prediction.eta is None
    assert prediction.last_time is None


def test_predict_eta_unavailable() -> None:
    prediction = CascadePredictor().predict(
        make_series((None, None))
    )
    assert prediction.status is PredictionStatus.ETA_UNAVAILABLE
    assert prediction.source_point_count == 0
    assert prediction.eta is None


def test_predict_one_eta_is_insufficient() -> None:
    prediction = CascadePredictor().predict(
        make_series((None, 0.5))
    )
    assert prediction.status is PredictionStatus.INSUFFICIENT_DATA
    assert prediction.source_point_count == 1


def test_constant_eta_forecast() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.5, 0.5))
    )
    assert prediction.status is PredictionStatus.AVAILABLE
    assert prediction.eta.value == pytest.approx(0.5)
    assert prediction.eta_slope == pytest.approx(0.0)
    assert prediction.stability is StabilityForecast.STABLE


def test_decreasing_eta_forecast() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.4, 0.3))
    )
    assert prediction.eta.value == pytest.approx(0.2)
    assert prediction.eta_slope == pytest.approx(-0.1)
    assert prediction.stability is StabilityForecast.WARNING


def test_increasing_eta_forecast() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.2, 0.3, 0.4))
    )
    assert prediction.eta.value == pytest.approx(0.5)
    assert prediction.eta_slope == pytest.approx(0.1)
    assert prediction.first_crossing is None


def test_critical_forecast() -> None:
    prediction = CascadePredictor(
        PredictorConfig(horizon=2.0)
    ).predict(make_series((0.2, 0.1)))
    assert prediction.eta.value == pytest.approx(-0.1)
    assert prediction.stability is StabilityForecast.CRITICAL
    assert prediction.predicts_critical


def test_warning_crossing_is_projected_first() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.4))
    )
    assert prediction.first_crossing is not None
    assert prediction.first_crossing.kind is CrossingKind.WARNING
    assert prediction.first_crossing.time_from_last == pytest.approx(1.5)


def test_no_crossing_when_slope_is_positive() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.4, 0.5))
    )
    assert prediction.first_crossing is None


def test_critical_crossing_when_already_below_warning() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.2, 0.1))
    )
    assert prediction.first_crossing is not None
    assert prediction.first_crossing.kind is CrossingKind.CRITICAL
    assert prediction.first_crossing.time_from_last == pytest.approx(1.0)


def test_probability_is_between_zero_and_one() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.2, 0.1))
    )
    assert 0.0 <= prediction.critical_probability <= 1.0


def test_lower_forecast_has_higher_critical_probability() -> None:
    high = CascadePredictor().predict(
        make_series((0.6, 0.5))
    )
    low = CascadePredictor().predict(
        make_series((0.2, 0.1))
    )
    assert low.critical_probability > high.critical_probability


def test_confidence_interval_zero_for_perfect_line() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.4, 0.3))
    )
    assert prediction.eta.standard_error == pytest.approx(0.0)
    assert prediction.eta.lower == pytest.approx(prediction.eta.value)
    assert prediction.eta.upper == pytest.approx(prediction.eta.value)


def test_noisy_line_has_nonzero_interval() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.35, 0.4, 0.2))
    )
    assert prediction.eta.standard_error > 0.0
    assert prediction.eta.lower < prediction.eta.value
    assert prediction.eta.upper > prediction.eta.value


def test_equal_times_produce_constant_mean_forecast() -> None:
    prediction = CascadePredictor().predict(
        make_series(
            (0.4, 0.6),
            times=(1.0, 1.0),
        )
    )
    assert prediction.eta.value == pytest.approx(0.5)
    assert prediction.eta_slope == pytest.approx(0.0)


def test_horizon_changes_forecast_time_and_value() -> None:
    prediction = CascadePredictor(
        PredictorConfig(horizon=2.0)
    ).predict(make_series((0.5, 0.4)))
    assert prediction.forecast_time == pytest.approx(4.0)
    assert prediction.eta.value == pytest.approx(0.2)


def test_maximum_points_uses_recent_window() -> None:
    prediction = CascadePredictor(
        PredictorConfig(maximum_points=2)
    ).predict(make_series((0.9, 0.8, 0.4, 0.3)))
    assert prediction.source_point_count == 2
    assert prediction.eta.value == pytest.approx(0.2)


def test_none_eta_values_are_omitted() -> None:
    prediction = CascadePredictor().predict(
        make_series((None, 0.5, None, 0.3))
    )
    assert prediction.source_point_count == 2
    assert prediction.eta_slope == pytest.approx(-0.1)


def test_d_fast_forecast_uses_last_plane() -> None:
    prediction = CascadePredictor().predict(
        make_series(
            (0.5, 0.5, 0.5),
            reserves=((0.1, 0.9), (0.9, 0.1), (0.9, 0.1)),
        )
    )
    assert prediction.d_fast.plane_id == "B"
    assert prediction.d_fast.confidence == pytest.approx(2 / 3)


def test_d_fast_switch_rate() -> None:
    prediction = CascadePredictor().predict(
        make_series(
            (0.5, 0.5, 0.5),
            reserves=((0.1, 0.9), (0.9, 0.1), (0.1, 0.9)),
        )
    )
    assert prediction.d_fast.switch_rate == pytest.approx(1.0)


def test_prediction_contains_source_phase() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.4))
    )
    assert prediction.source_phase is CascadePhase.DESTABILIZING


def test_prediction_summary_contains_findings() -> None:
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.4))
    )
    assert "Predicted stability:" in prediction.summary
    assert "Forecast eta:" in prediction.summary
    assert "Persistent D_fast candidate:" in prediction.summary


def test_metadata_is_forwarded() -> None:
    metadata = {"experiment": "E1"}
    prediction = CascadePredictor().predict(
        make_series((0.5, 0.4)),
        metadata=metadata,
    )
    metadata["experiment"] = "changed"
    assert prediction.metadata["experiment"] == "E1"


def test_predict_observations_wrapper() -> None:
    prediction = predict_observations(
        make_series((0.5, 0.4)),
        metadata={"wrapper": True},
    )
    assert isinstance(prediction, CascadePrediction)
    assert prediction.metadata["wrapper"] is True


def test_predict_observations_forwards_config() -> None:
    prediction = predict_observations(
        make_series((0.5, 0.4)),
        config=PredictorConfig(horizon=2.0),
    )
    assert prediction.forecast_time == pytest.approx(4.0)
