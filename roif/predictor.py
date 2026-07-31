"""
Deterministic short-horizon forecasting for ROIF observation series.

The predictor extrapolates spectral coherence (eta) with ordinary least
squares, estimates uncertainty from regression residuals, projects threshold
crossings, and summarizes the persistence of D_fast. It is intentionally a
transparent engineering forecast rather than a clinical diagnosis or a
learned probabilistic model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from .analyzer import AnalyzerConfig, CascadeAnalyzer, CascadePhase
from .observer import ObservationSeries


class PredictorError(ValueError):
    """Raised when predictor input or configuration is invalid."""


_EPSILON = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise PredictorError(f"{name} must be a real number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictorError(f"{name} must be a real number.") from exc

    if not math.isfinite(result):
        raise PredictorError(f"{name} must be finite.")

    if strictly_positive and result <= 0.0:
        raise PredictorError(f"{name} must be greater than zero.")

    if minimum is not None and result < minimum:
        raise PredictorError(
            f"{name} must be greater than or equal to {minimum}."
        )

    return result


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise PredictorError("metadata must be a mapping.")

    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise PredictorError("metadata keys must be strings.")

        normalized = key.strip()
        if not normalized:
            raise PredictorError("metadata keys cannot be empty.")

        copied[normalized] = value

    return MappingProxyType(copied)


class PredictionStatus(str, Enum):
    """Availability and interpretation status of a forecast."""

    AVAILABLE = "available"
    INSUFFICIENT_DATA = "insufficient_data"
    ETA_UNAVAILABLE = "eta_unavailable"


class StabilityForecast(str, Enum):
    """Predicted stability condition at the forecast horizon."""

    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CrossingKind(str, Enum):
    """Threshold expected to be crossed first."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Configuration for deterministic ROIF forecasting."""

    horizon: float = 1.0
    confidence_z: float = 1.96
    minimum_points: int = 2
    maximum_points: int | None = None
    eta_warning_threshold: float = 0.25
    eta_critical_threshold: float = 0.0
    slope_tolerance: float = 1e-9
    probability_scale: float = 0.1

    def __post_init__(self) -> None:
        horizon = _finite_float(
            self.horizon,
            name="horizon",
            strictly_positive=True,
        )
        confidence_z = _finite_float(
            self.confidence_z,
            name="confidence_z",
            minimum=0.0,
        )
        warning = _finite_float(
            self.eta_warning_threshold,
            name="eta_warning_threshold",
        )
        critical = _finite_float(
            self.eta_critical_threshold,
            name="eta_critical_threshold",
        )
        slope_tolerance = _finite_float(
            self.slope_tolerance,
            name="slope_tolerance",
            minimum=0.0,
        )
        probability_scale = _finite_float(
            self.probability_scale,
            name="probability_scale",
            strictly_positive=True,
        )

        if critical > warning:
            raise PredictorError(
                "eta_critical_threshold cannot exceed "
                "eta_warning_threshold."
            )

        if (
            isinstance(self.minimum_points, bool)
            or not isinstance(self.minimum_points, int)
            or self.minimum_points < 2
        ):
            raise PredictorError(
                "minimum_points must be an integer greater than or equal to 2."
            )

        maximum_points = self.maximum_points
        if maximum_points is not None:
            if (
                isinstance(maximum_points, bool)
                or not isinstance(maximum_points, int)
                or maximum_points < self.minimum_points
            ):
                raise PredictorError(
                    "maximum_points must be None or an integer greater than "
                    "or equal to minimum_points."
                )

        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "confidence_z", confidence_z)
        object.__setattr__(self, "eta_warning_threshold", warning)
        object.__setattr__(self, "eta_critical_threshold", critical)
        object.__setattr__(self, "slope_tolerance", slope_tolerance)
        object.__setattr__(self, "probability_scale", probability_scale)


@dataclass(frozen=True, slots=True)
class ForecastInterval:
    """Point forecast and symmetric uncertainty interval."""

    value: float
    lower: float
    upper: float
    standard_error: float

    def __post_init__(self) -> None:
        value = _finite_float(self.value, name="value")
        lower = _finite_float(self.lower, name="lower")
        upper = _finite_float(self.upper, name="upper")
        standard_error = _finite_float(
            self.standard_error,
            name="standard_error",
            minimum=0.0,
        )

        if lower > value + _EPSILON or value > upper + _EPSILON:
            raise PredictorError(
                "Forecast interval must satisfy lower <= value <= upper."
            )

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "standard_error", standard_error)

    def as_dict(self) -> dict[str, float]:
        return {
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "standard_error": self.standard_error,
        }


@dataclass(frozen=True, slots=True)
class ThresholdCrossing:
    """Projected crossing of an eta threshold."""

    kind: CrossingKind
    threshold: float
    time: float
    time_from_last: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CrossingKind):
            raise PredictorError("kind must be a CrossingKind.")

        threshold = _finite_float(self.threshold, name="threshold")
        time = _finite_float(self.time, name="time")
        time_from_last = _finite_float(
            self.time_from_last,
            name="time_from_last",
            minimum=0.0,
        )

        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "time_from_last", time_from_last)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "threshold": self.threshold,
            "time": self.time,
            "time_from_last": self.time_from_last,
        }


@dataclass(frozen=True, slots=True)
class DFastForecast:
    """Persistence forecast for the fast-failure plane."""

    plane_id: str | None
    confidence: float
    switch_rate: float
    observed_count: int

    def __post_init__(self) -> None:
        if self.plane_id is not None:
            if not isinstance(self.plane_id, str):
                raise PredictorError("plane_id must be a string or None.")
            normalized = self.plane_id.strip()
            if not normalized:
                raise PredictorError("plane_id cannot be empty.")
            object.__setattr__(self, "plane_id", normalized)

        confidence = _finite_float(
            self.confidence,
            name="confidence",
            minimum=0.0,
        )
        switch_rate = _finite_float(
            self.switch_rate,
            name="switch_rate",
            minimum=0.0,
        )

        if confidence > 1.0 + _EPSILON:
            raise PredictorError("confidence cannot exceed 1.0.")
        if switch_rate > 1.0 + _EPSILON:
            raise PredictorError("switch_rate cannot exceed 1.0.")
        if (
            isinstance(self.observed_count, bool)
            or not isinstance(self.observed_count, int)
            or self.observed_count < 0
        ):
            raise PredictorError(
                "observed_count must be a non-negative integer."
            )

        object.__setattr__(self, "confidence", min(1.0, confidence))
        object.__setattr__(self, "switch_rate", min(1.0, switch_rate))

    def as_dict(self) -> dict[str, Any]:
        return {
            "plane_id": self.plane_id,
            "confidence": self.confidence,
            "switch_rate": self.switch_rate,
            "observed_count": self.observed_count,
        }


@dataclass(frozen=True, slots=True)
class CascadePrediction:
    """Immutable predictor output."""

    status: PredictionStatus
    stability: StabilityForecast
    source_observation_count: int
    source_point_count: int
    last_time: float | None
    forecast_time: float | None
    eta: ForecastInterval | None
    eta_slope: float | None
    critical_probability: float | None
    first_crossing: ThresholdCrossing | None
    d_fast: DFastForecast
    source_phase: CascadePhase
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PredictionStatus):
            raise PredictorError("status must be a PredictionStatus.")
        if not isinstance(self.stability, StabilityForecast):
            raise PredictorError("stability must be a StabilityForecast.")
        if not isinstance(self.source_phase, CascadePhase):
            raise PredictorError("source_phase must be a CascadePhase.")

        for name in ("source_observation_count", "source_point_count"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise PredictorError(
                    f"{name} must be a non-negative integer."
                )

        for name in ("last_time", "forecast_time", "eta_slope"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_float(value, name=name),
                )

        if self.eta is not None and not isinstance(
            self.eta, ForecastInterval
        ):
            raise PredictorError("eta must be a ForecastInterval or None.")

        probability = self.critical_probability
        if probability is not None:
            probability = _finite_float(
                probability,
                name="critical_probability",
                minimum=0.0,
            )
            if probability > 1.0 + _EPSILON:
                raise PredictorError(
                    "critical_probability cannot exceed 1.0."
                )
            object.__setattr__(
                self,
                "critical_probability",
                min(1.0, probability),
            )

        if (
            self.first_crossing is not None
            and not isinstance(self.first_crossing, ThresholdCrossing)
        ):
            raise PredictorError(
                "first_crossing must be a ThresholdCrossing or None."
            )

        if not isinstance(self.d_fast, DFastForecast):
            raise PredictorError("d_fast must be a DFastForecast.")

        if not isinstance(self.summary, str):
            raise PredictorError("summary must be a string.")
        summary = self.summary.strip()
        if not summary:
            raise PredictorError("summary cannot be empty.")

        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def is_available(self) -> bool:
        return self.status is PredictionStatus.AVAILABLE

    @property
    def predicts_critical(self) -> bool:
        return self.stability is StabilityForecast.CRITICAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "stability": self.stability.value,
            "source_observation_count": self.source_observation_count,
            "source_point_count": self.source_point_count,
            "last_time": self.last_time,
            "forecast_time": self.forecast_time,
            "eta": None if self.eta is None else self.eta.as_dict(),
            "eta_slope": self.eta_slope,
            "critical_probability": self.critical_probability,
            "first_crossing": (
                None
                if self.first_crossing is None
                else self.first_crossing.as_dict()
            ),
            "d_fast": self.d_fast.as_dict(),
            "source_phase": self.source_phase.value,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


def _fit_linear_forecast(
    points: Sequence[tuple[float, float]],
    *,
    forecast_time: float,
    confidence_z: float,
) -> tuple[ForecastInterval, float]:
    times = np.asarray([item[0] for item in points], dtype=float)
    values = np.asarray([item[1] for item in points], dtype=float)

    center = float(np.mean(times))
    x = times - center
    denominator = float(np.dot(x, x))

    if denominator <= _EPSILON:
        slope = 0.0
        intercept = float(np.mean(values))
    else:
        slope = float(np.dot(x, values - np.mean(values)) / denominator)
        intercept = float(np.mean(values))

    delta = forecast_time - center
    predicted = intercept + slope * delta

    fitted = intercept + slope * x
    residuals = values - fitted
    degrees_of_freedom = max(1, len(points) - 2)
    residual_variance = float(
        np.dot(residuals, residuals) / degrees_of_freedom
    )

    leverage = 1.0 / len(points)
    if denominator > _EPSILON:
        leverage += (delta * delta) / denominator

    standard_error = math.sqrt(max(0.0, residual_variance * leverage))
    margin = confidence_z * standard_error

    return (
        ForecastInterval(
            value=predicted,
            lower=predicted - margin,
            upper=predicted + margin,
            standard_error=standard_error,
        ),
        slope,
    )


def _critical_probability(
    *,
    predicted_eta: float,
    critical_threshold: float,
    scale: float,
) -> float:
    z = (predicted_eta - critical_threshold) / scale
    if z >= 700.0:
        return 0.0
    if z <= -700.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _stability(
    eta: float,
    *,
    warning: float,
    critical: float,
) -> StabilityForecast:
    if eta <= critical:
        return StabilityForecast.CRITICAL
    if eta <= warning:
        return StabilityForecast.WARNING
    return StabilityForecast.STABLE


def _first_crossing(
    *,
    last_time: float,
    last_eta: float,
    slope: float,
    warning: float,
    critical: float,
) -> ThresholdCrossing | None:
    if slope >= -_EPSILON:
        return None

    candidates: list[ThresholdCrossing] = []
    for kind, threshold in (
        (CrossingKind.WARNING, warning),
        (CrossingKind.CRITICAL, critical),
    ):
        if last_eta <= threshold:
            continue

        delta = (threshold - last_eta) / slope
        if delta >= 0.0 and math.isfinite(delta):
            candidates.append(
                ThresholdCrossing(
                    kind=kind,
                    threshold=threshold,
                    time=last_time + delta,
                    time_from_last=delta,
                )
            )

    if not candidates:
        return None

    return min(candidates, key=lambda item: item.time_from_last)


def _d_fast_forecast(series: ObservationSeries) -> DFastForecast:
    ids = tuple(
        observation.d_fast_id
        for observation in series.observations
        if observation.d_fast_id is not None
    )

    if not ids:
        return DFastForecast(
            plane_id=None,
            confidence=0.0,
            switch_rate=0.0,
            observed_count=0,
        )

    counts = Counter(ids)
    last = ids[-1]
    confidence = counts[last] / len(ids)
    switches = sum(
        current != previous
        for previous, current in zip(ids, ids[1:])
    )
    switch_rate = switches / max(1, len(ids) - 1)

    return DFastForecast(
        plane_id=last,
        confidence=confidence,
        switch_rate=switch_rate,
        observed_count=len(ids),
    )


def _summary(
    *,
    status: PredictionStatus,
    stability: StabilityForecast,
    eta: ForecastInterval | None,
    slope: float | None,
    crossing: ThresholdCrossing | None,
    d_fast: DFastForecast,
) -> str:
    if status is not PredictionStatus.AVAILABLE:
        return (
            f"Prediction status: {status.value}. "
            "A numerical eta forecast is unavailable."
        )

    parts = [
        f"Predicted stability: {stability.value}.",
        f"Forecast eta: {eta.value:.6g}.",
        f"Eta slope: {slope:.6g}.",
    ]

    if crossing is None:
        parts.append("No future downward threshold crossing was projected.")
    else:
        parts.append(
            f"First projected crossing: {crossing.kind.value} "
            f"in {crossing.time_from_last:.6g} time units."
        )

    if d_fast.plane_id is None:
        parts.append("D_fast forecast is unavailable.")
    else:
        parts.append(
            f"Persistent D_fast candidate: {d_fast.plane_id} "
            f"(confidence {d_fast.confidence:.3f})."
        )

    return " ".join(parts)


class CascadePredictor:
    """Transparent short-horizon predictor for ROIF observations."""

    def __init__(
        self,
        config: PredictorConfig | None = None,
    ) -> None:
        if config is None:
            config = PredictorConfig()

        if not isinstance(config, PredictorConfig):
            raise PredictorError("config must be a PredictorConfig.")

        self._config = config
        self._analyzer = CascadeAnalyzer(
            AnalyzerConfig(
                eta_warning_threshold=config.eta_warning_threshold,
                eta_critical_threshold=config.eta_critical_threshold,
                stable_slope_tolerance=config.slope_tolerance,
            )
        )

    @property
    def config(self) -> PredictorConfig:
        return self._config

    def predict(
        self,
        series: ObservationSeries,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CascadePrediction:
        if not isinstance(series, ObservationSeries):
            raise PredictorError(
                "series must be an ObservationSeries."
            )

        analysis = self._analyzer.analyze(series)
        d_fast = _d_fast_forecast(series)
        points = [
            (observation.time, observation.eta)
            for observation in series.observations
            if observation.eta is not None
        ]

        if self._config.maximum_points is not None:
            points = points[-self._config.maximum_points :]

        count = len(points)
        last_time = (
            series.final.time if series.final is not None else None
        )

        if not series.observations:
            status = PredictionStatus.INSUFFICIENT_DATA
            return CascadePrediction(
                status=status,
                stability=StabilityForecast.UNKNOWN,
                source_observation_count=0,
                source_point_count=0,
                last_time=None,
                forecast_time=None,
                eta=None,
                eta_slope=None,
                critical_probability=None,
                first_crossing=None,
                d_fast=d_fast,
                source_phase=analysis.phase,
                summary=_summary(
                    status=status,
                    stability=StabilityForecast.UNKNOWN,
                    eta=None,
                    slope=None,
                    crossing=None,
                    d_fast=d_fast,
                ),
                metadata=metadata,
            )

        if count == 0:
            status = PredictionStatus.ETA_UNAVAILABLE
            return CascadePrediction(
                status=status,
                stability=StabilityForecast.UNKNOWN,
                source_observation_count=len(series),
                source_point_count=0,
                last_time=last_time,
                forecast_time=None,
                eta=None,
                eta_slope=None,
                critical_probability=None,
                first_crossing=None,
                d_fast=d_fast,
                source_phase=analysis.phase,
                summary=_summary(
                    status=status,
                    stability=StabilityForecast.UNKNOWN,
                    eta=None,
                    slope=None,
                    crossing=None,
                    d_fast=d_fast,
                ),
                metadata=metadata,
            )

        if count < self._config.minimum_points:
            status = PredictionStatus.INSUFFICIENT_DATA
            return CascadePrediction(
                status=status,
                stability=StabilityForecast.UNKNOWN,
                source_observation_count=len(series),
                source_point_count=count,
                last_time=last_time,
                forecast_time=None,
                eta=None,
                eta_slope=None,
                critical_probability=None,
                first_crossing=None,
                d_fast=d_fast,
                source_phase=analysis.phase,
                summary=_summary(
                    status=status,
                    stability=StabilityForecast.UNKNOWN,
                    eta=None,
                    slope=None,
                    crossing=None,
                    d_fast=d_fast,
                ),
                metadata=metadata,
            )

        forecast_time = points[-1][0] + self._config.horizon
        eta_interval, slope = _fit_linear_forecast(
            points,
            forecast_time=forecast_time,
            confidence_z=self._config.confidence_z,
        )

        if abs(slope) <= self._config.slope_tolerance:
            slope = 0.0

        stability = _stability(
            eta_interval.value,
            warning=self._config.eta_warning_threshold,
            critical=self._config.eta_critical_threshold,
        )
        crossing = _first_crossing(
            last_time=points[-1][0],
            last_eta=points[-1][1],
            slope=slope,
            warning=self._config.eta_warning_threshold,
            critical=self._config.eta_critical_threshold,
        )
        probability = _critical_probability(
            predicted_eta=eta_interval.value,
            critical_threshold=self._config.eta_critical_threshold,
            scale=self._config.probability_scale,
        )

        status = PredictionStatus.AVAILABLE
        return CascadePrediction(
            status=status,
            stability=stability,
            source_observation_count=len(series),
            source_point_count=count,
            last_time=last_time,
            forecast_time=forecast_time,
            eta=eta_interval,
            eta_slope=slope,
            critical_probability=probability,
            first_crossing=crossing,
            d_fast=d_fast,
            source_phase=analysis.phase,
            summary=_summary(
                status=status,
                stability=stability,
                eta=eta_interval,
                slope=slope,
                crossing=crossing,
                d_fast=d_fast,
            ),
            metadata=metadata,
        )


def predict_observations(
    series: ObservationSeries,
    *,
    config: PredictorConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CascadePrediction:
    """Convenience wrapper around ``CascadePredictor.predict``."""

    return CascadePredictor(config).predict(
        series,
        metadata=metadata,
    )


__all__ = [
    "CascadePrediction",
    "CascadePredictor",
    "CrossingKind",
    "DFastForecast",
    "ForecastInterval",
    "PredictionStatus",
    "PredictorConfig",
    "PredictorError",
    "StabilityForecast",
    "ThresholdCrossing",
    "predict_observations",
]
