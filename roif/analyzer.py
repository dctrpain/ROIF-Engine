"""
ROIF cascade analyzer.

The analyzer is a deterministic interpretation layer above
``ObservationSeries``. It does not mutate observations and does not perform
clinical diagnosis. It summarizes temporal direction, coherence state,
dominant fast-failure planes, alert burden, and cascade phase into immutable
analysis objects suitable for downstream reporting and decision support.
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

from .observer import (
    AlertLevel,
    Observation,
    ObservationSeries,
    ObserverAlert,
)


class AnalyzerError(ValueError):
    """Raised when analyzer input or configuration is invalid."""


_FLOAT_TOLERANCE = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise AnalyzerError(f"{name} must be a real number.")

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalyzerError(f"{name} must be a real number.") from exc

    if not math.isfinite(normalized):
        raise AnalyzerError(f"{name} must be finite.")

    if minimum is not None and normalized < minimum:
        raise AnalyzerError(
            f"{name} must be greater than or equal to {minimum}."
        )

    return normalized


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise AnalyzerError("metadata must be a mapping.")

    copied: dict[str, Any] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise AnalyzerError("metadata keys must be strings.")

        normalized = key.strip()
        if not normalized:
            raise AnalyzerError("metadata keys cannot be empty.")

        copied[normalized] = value

    return MappingProxyType(copied)


class TrendDirection(str, Enum):
    """Direction of a scalar time series."""

    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    UNKNOWN = "unknown"


class CoherenceState(str, Enum):
    """Final spectral-coherence state."""

    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CascadePhase(str, Enum):
    """High-level cascade phase inferred from the observation sequence."""

    QUIESCENT = "quiescent"
    COMPENSATING = "compensating"
    DESTABILIZING = "destabilizing"
    SUPERCRITICAL = "supercritical"
    RECOVERING = "recovering"
    INDETERMINATE = "indeterminate"


class AnalysisRisk(str, Enum):
    """Overall analysis risk level."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    """Thresholds used by ``CascadeAnalyzer``."""

    eta_warning_threshold: float = 0.25
    eta_critical_threshold: float = 0.0
    stable_slope_tolerance: float = 1e-9
    burden_growth_tolerance: float = 1e-9
    minimum_trend_points: int = 2

    def __post_init__(self) -> None:
        warning = _finite_float(
            self.eta_warning_threshold,
            name="eta_warning_threshold",
        )
        critical = _finite_float(
            self.eta_critical_threshold,
            name="eta_critical_threshold",
        )
        slope_tolerance = _finite_float(
            self.stable_slope_tolerance,
            name="stable_slope_tolerance",
            minimum=0.0,
        )
        burden_tolerance = _finite_float(
            self.burden_growth_tolerance,
            name="burden_growth_tolerance",
            minimum=0.0,
        )

        if critical > warning:
            raise AnalyzerError(
                "eta_critical_threshold cannot be greater than "
                "eta_warning_threshold."
            )

        if (
            isinstance(self.minimum_trend_points, bool)
            or not isinstance(self.minimum_trend_points, int)
            or self.minimum_trend_points < 2
        ):
            raise AnalyzerError(
                "minimum_trend_points must be an integer "
                "greater than or equal to 2."
            )

        object.__setattr__(self, "eta_warning_threshold", warning)
        object.__setattr__(self, "eta_critical_threshold", critical)
        object.__setattr__(
            self,
            "stable_slope_tolerance",
            slope_tolerance,
        )
        object.__setattr__(
            self,
            "burden_growth_tolerance",
            burden_tolerance,
        )


@dataclass(frozen=True, slots=True)
class ScalarTrend:
    """Trend summary for one scalar series."""

    direction: TrendDirection
    first_value: float | None
    final_value: float | None
    absolute_change: float | None
    slope: float | None
    point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.direction, TrendDirection):
            raise AnalyzerError(
                "direction must be a TrendDirection."
            )

        if (
            isinstance(self.point_count, bool)
            or not isinstance(self.point_count, int)
            or self.point_count < 0
        ):
            raise AnalyzerError(
                "point_count must be a non-negative integer."
            )

        for name in (
            "first_value",
            "final_value",
            "absolute_change",
            "slope",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_float(value, name=name),
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "first_value": self.first_value,
            "final_value": self.final_value,
            "absolute_change": self.absolute_change,
            "slope": self.slope,
            "point_count": self.point_count,
        }


@dataclass(frozen=True, slots=True)
class PlaneDominance:
    """Frequency summary for a D_fast plane."""

    plane_id: str
    count: int
    fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.plane_id, str):
            raise AnalyzerError("plane_id must be a string.")

        plane_id = self.plane_id.strip()
        if not plane_id:
            raise AnalyzerError("plane_id cannot be empty.")

        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise AnalyzerError(
                "count must be an integer greater than or equal to 1."
            )

        fraction = _finite_float(
            self.fraction,
            name="fraction",
            minimum=0.0,
        )
        if fraction > 1.0 + _FLOAT_TOLERANCE:
            raise AnalyzerError("fraction cannot exceed 1.0.")

        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "fraction", min(1.0, fraction))

    def as_dict(self) -> dict[str, Any]:
        return {
            "plane_id": self.plane_id,
            "count": self.count,
            "fraction": self.fraction,
        }


@dataclass(frozen=True, slots=True)
class CascadeAnalysis:
    """Immutable analysis of an ``ObservationSeries``."""

    observation_count: int
    phase: CascadePhase
    risk: AnalysisRisk
    coherence_state: CoherenceState
    eta_trend: ScalarTrend
    burden_trend: ScalarTrend
    delta_trend: ScalarTrend
    dominant_d_fast: PlaneDominance | None
    d_fast_switch_count: int
    alert_count: int
    critical_alert_count: int
    warning_alert_count: int
    first_snapshot_id: str | None
    final_snapshot_id: str | None
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "observation_count",
            "d_fast_switch_count",
            "alert_count",
            "critical_alert_count",
            "warning_alert_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise AnalyzerError(
                    f"{name} must be a non-negative integer."
                )

        if not isinstance(self.phase, CascadePhase):
            raise AnalyzerError("phase must be a CascadePhase.")

        if not isinstance(self.risk, AnalysisRisk):
            raise AnalyzerError("risk must be an AnalysisRisk.")

        if not isinstance(self.coherence_state, CoherenceState):
            raise AnalyzerError(
                "coherence_state must be a CoherenceState."
            )

        for name in ("eta_trend", "burden_trend", "delta_trend"):
            if not isinstance(getattr(self, name), ScalarTrend):
                raise AnalyzerError(
                    f"{name} must be a ScalarTrend."
                )

        if (
            self.dominant_d_fast is not None
            and not isinstance(self.dominant_d_fast, PlaneDominance)
        ):
            raise AnalyzerError(
                "dominant_d_fast must be a PlaneDominance or None."
            )

        for name in ("first_snapshot_id", "final_snapshot_id"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str):
                    raise AnalyzerError(f"{name} must be a string or None.")
                normalized = value.strip()
                if not normalized:
                    raise AnalyzerError(f"{name} cannot be empty.")
                object.__setattr__(self, name, normalized)

        if not isinstance(self.summary, str):
            raise AnalyzerError("summary must be a string.")

        summary = self.summary.strip()
        if not summary:
            raise AnalyzerError("summary cannot be empty.")

        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def has_critical_alerts(self) -> bool:
        return self.critical_alert_count > 0

    @property
    def is_supercritical(self) -> bool:
        return self.phase is CascadePhase.SUPERCRITICAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "phase": self.phase.value,
            "risk": self.risk.value,
            "coherence_state": self.coherence_state.value,
            "eta_trend": self.eta_trend.as_dict(),
            "burden_trend": self.burden_trend.as_dict(),
            "delta_trend": self.delta_trend.as_dict(),
            "dominant_d_fast": (
                None
                if self.dominant_d_fast is None
                else self.dominant_d_fast.as_dict()
            ),
            "d_fast_switch_count": self.d_fast_switch_count,
            "alert_count": self.alert_count,
            "critical_alert_count": self.critical_alert_count,
            "warning_alert_count": self.warning_alert_count,
            "first_snapshot_id": self.first_snapshot_id,
            "final_snapshot_id": self.final_snapshot_id,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


def _linear_slope(
    points: Sequence[tuple[float, float]],
) -> float:
    if len(points) < 2:
        raise AnalyzerError(
            "At least two points are required to calculate slope."
        )

    times = np.asarray([item[0] for item in points], dtype=float)
    values = np.asarray([item[1] for item in points], dtype=float)

    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise AnalyzerError("trend points must contain finite values.")

    centered_time = times - float(np.mean(times))
    denominator = float(np.dot(centered_time, centered_time))

    if denominator <= _FLOAT_TOLERANCE:
        return 0.0

    centered_values = values - float(np.mean(values))
    return float(
        np.dot(centered_time, centered_values) / denominator
    )


def calculate_scalar_trend(
    points: Sequence[tuple[float, float]],
    *,
    improving_when_increasing: bool,
    stable_tolerance: float = 1e-9,
    minimum_points: int = 2,
) -> ScalarTrend:
    """Calculate a deterministic linear trend from ``(time, value)`` pairs."""

    if isinstance(points, (str, bytes)):
        raise AnalyzerError(
            "points must be a sequence of (time, value) pairs."
        )

    try:
        normalized_points = tuple(points)
    except TypeError as exc:
        raise AnalyzerError(
            "points must be a sequence of (time, value) pairs."
        ) from exc

    if not isinstance(improving_when_increasing, bool):
        raise AnalyzerError(
            "improving_when_increasing must be a bool."
        )

    tolerance = _finite_float(
        stable_tolerance,
        name="stable_tolerance",
        minimum=0.0,
    )

    if (
        isinstance(minimum_points, bool)
        or not isinstance(minimum_points, int)
        or minimum_points < 2
    ):
        raise AnalyzerError(
            "minimum_points must be an integer greater than or equal to 2."
        )

    parsed: list[tuple[float, float]] = []
    previous_time: float | None = None

    for item in normalized_points:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise AnalyzerError(
                "Every trend point must contain exactly time and value."
            )

        time = _finite_float(item[0], name="point time")
        value = _finite_float(item[1], name="point value")

        if (
            previous_time is not None
            and time < previous_time - _FLOAT_TOLERANCE
        ):
            raise AnalyzerError(
                "trend points must be chronological."
            )

        parsed.append((time, value))
        previous_time = time

    if not parsed:
        return ScalarTrend(
            direction=TrendDirection.UNKNOWN,
            first_value=None,
            final_value=None,
            absolute_change=None,
            slope=None,
            point_count=0,
        )

    first_value = parsed[0][1]
    final_value = parsed[-1][1]
    change = final_value - first_value

    if len(parsed) < minimum_points:
        return ScalarTrend(
            direction=TrendDirection.UNKNOWN,
            first_value=first_value,
            final_value=final_value,
            absolute_change=change,
            slope=None,
            point_count=len(parsed),
        )

    slope = _linear_slope(parsed)

    if abs(slope) <= tolerance:
        direction = TrendDirection.STABLE
    else:
        increasing = slope > 0.0
        improving = (
            increasing
            if improving_when_increasing
            else not increasing
        )
        direction = (
            TrendDirection.IMPROVING
            if improving
            else TrendDirection.WORSENING
        )

    return ScalarTrend(
        direction=direction,
        first_value=first_value,
        final_value=final_value,
        absolute_change=change,
        slope=slope,
        point_count=len(parsed),
    )


def _coherence_state(
    final_eta: float | None,
    *,
    config: AnalyzerConfig,
) -> CoherenceState:
    if final_eta is None:
        return CoherenceState.UNKNOWN

    if final_eta <= config.eta_critical_threshold:
        return CoherenceState.CRITICAL

    if final_eta <= config.eta_warning_threshold:
        return CoherenceState.WARNING

    return CoherenceState.STABLE


def _dominant_d_fast(
    observations: Sequence[Observation],
) -> PlaneDominance | None:
    plane_ids = tuple(
        item.d_fast_id
        for item in observations
        if item.d_fast_id is not None
    )

    if not plane_ids:
        return None

    counts = Counter(plane_ids)
    first_index = {
        plane_id: plane_ids.index(plane_id)
        for plane_id in counts
    }
    winner_id, winner_count = min(
        counts.items(),
        key=lambda item: (
            -item[1],
            first_index[item[0]],
            item[0],
        ),
    )

    return PlaneDominance(
        plane_id=winner_id,
        count=winner_count,
        fraction=winner_count / len(plane_ids),
    )


def _d_fast_switch_count(
    observations: Sequence[Observation],
) -> int:
    plane_ids = tuple(item.d_fast_id for item in observations)
    return sum(
        current != previous
        for previous, current in zip(plane_ids, plane_ids[1:])
    )


def _alert_counts(
    alerts: Sequence[ObserverAlert],
) -> tuple[int, int]:
    critical = sum(
        alert.level is AlertLevel.CRITICAL
        for alert in alerts
    )
    warning = sum(
        alert.level is AlertLevel.WARNING
        for alert in alerts
    )
    return critical, warning


def _phase(
    *,
    coherence_state: CoherenceState,
    eta_trend: ScalarTrend,
    burden_trend: ScalarTrend,
    alert_count: int,
    observation_count: int,
) -> CascadePhase:
    if observation_count == 0:
        return CascadePhase.INDETERMINATE

    if coherence_state is CoherenceState.CRITICAL:
        return CascadePhase.SUPERCRITICAL

    if (
        eta_trend.direction is TrendDirection.IMPROVING
        and burden_trend.direction is not TrendDirection.WORSENING
    ):
        return CascadePhase.RECOVERING

    if (
        coherence_state is CoherenceState.WARNING
        or eta_trend.direction is TrendDirection.WORSENING
        or burden_trend.direction is TrendDirection.WORSENING
    ):
        return CascadePhase.DESTABILIZING

    if alert_count > 0:
        return CascadePhase.COMPENSATING

    if coherence_state in (
        CoherenceState.STABLE,
        CoherenceState.UNKNOWN,
    ):
        return CascadePhase.QUIESCENT

    return CascadePhase.INDETERMINATE


def _risk(
    *,
    phase: CascadePhase,
    coherence_state: CoherenceState,
    critical_alert_count: int,
    warning_alert_count: int,
) -> AnalysisRisk:
    if (
        phase is CascadePhase.SUPERCRITICAL
        or coherence_state is CoherenceState.CRITICAL
        or critical_alert_count > 0
    ):
        return AnalysisRisk.CRITICAL

    if (
        phase is CascadePhase.DESTABILIZING
        or coherence_state is CoherenceState.WARNING
    ):
        return AnalysisRisk.HIGH

    if (
        phase is CascadePhase.COMPENSATING
        or warning_alert_count > 0
    ):
        return AnalysisRisk.MODERATE

    return AnalysisRisk.LOW


def _summary(
    *,
    phase: CascadePhase,
    risk: AnalysisRisk,
    coherence_state: CoherenceState,
    dominant_d_fast: PlaneDominance | None,
    eta_trend: ScalarTrend,
) -> str:
    parts = [
        f"Cascade phase: {phase.value}.",
        f"Overall risk: {risk.value}.",
        f"Coherence state: {coherence_state.value}.",
        f"Eta trend: {eta_trend.direction.value}.",
    ]

    if dominant_d_fast is not None:
        parts.append(
            "Dominant D_fast: "
            f"{dominant_d_fast.plane_id} "
            f"({dominant_d_fast.count} observations, "
            f"{dominant_d_fast.fraction:.3f} of classified states)."
        )
    else:
        parts.append("Dominant D_fast is unavailable.")

    return " ".join(parts)


class CascadeAnalyzer:
    """Analyze immutable observer output."""

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
    ) -> None:
        if config is None:
            config = AnalyzerConfig()

        if not isinstance(config, AnalyzerConfig):
            raise AnalyzerError(
                "config must be an AnalyzerConfig."
            )

        self._config = config

    @property
    def config(self) -> AnalyzerConfig:
        return self._config

    def analyze(
        self,
        series: ObservationSeries,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CascadeAnalysis:
        if not isinstance(series, ObservationSeries):
            raise AnalyzerError(
                "series must be an ObservationSeries."
            )

        observations = series.observations

        eta_points = tuple(
            (item.time, item.eta)
            for item in observations
            if item.eta is not None
        )
        burden_points = tuple(
            (item.time, item.metrics.event_burden)
            for item in observations
        )
        delta_points = tuple(
            (item.time, item.metrics.maximum_absolute_delta)
            for item in observations
        )

        eta_trend = calculate_scalar_trend(
            eta_points,
            improving_when_increasing=True,
            stable_tolerance=self._config.stable_slope_tolerance,
            minimum_points=self._config.minimum_trend_points,
        )
        burden_trend = calculate_scalar_trend(
            burden_points,
            improving_when_increasing=False,
            stable_tolerance=self._config.burden_growth_tolerance,
            minimum_points=self._config.minimum_trend_points,
        )
        delta_trend = calculate_scalar_trend(
            delta_points,
            improving_when_increasing=False,
            stable_tolerance=self._config.burden_growth_tolerance,
            minimum_points=self._config.minimum_trend_points,
        )

        final_eta = (
            series.final.eta
            if series.final is not None
            else None
        )
        coherence_state = _coherence_state(
            final_eta,
            config=self._config,
        )

        dominant = _dominant_d_fast(observations)
        switch_count = _d_fast_switch_count(observations)
        critical_count, warning_count = _alert_counts(series.alerts)

        phase = _phase(
            coherence_state=coherence_state,
            eta_trend=eta_trend,
            burden_trend=burden_trend,
            alert_count=len(series.alerts),
            observation_count=len(series),
        )
        risk = _risk(
            phase=phase,
            coherence_state=coherence_state,
            critical_alert_count=critical_count,
            warning_alert_count=warning_count,
        )

        return CascadeAnalysis(
            observation_count=len(series),
            phase=phase,
            risk=risk,
            coherence_state=coherence_state,
            eta_trend=eta_trend,
            burden_trend=burden_trend,
            delta_trend=delta_trend,
            dominant_d_fast=dominant,
            d_fast_switch_count=switch_count,
            alert_count=len(series.alerts),
            critical_alert_count=critical_count,
            warning_alert_count=warning_count,
            first_snapshot_id=(
                series.first.snapshot_id
                if series.first is not None
                else None
            ),
            final_snapshot_id=(
                series.final.snapshot_id
                if series.final is not None
                else None
            ),
            summary=_summary(
                phase=phase,
                risk=risk,
                coherence_state=coherence_state,
                dominant_d_fast=dominant,
                eta_trend=eta_trend,
            ),
            metadata=metadata,
        )


def analyze_observations(
    series: ObservationSeries,
    *,
    config: AnalyzerConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CascadeAnalysis:
    """Convenience wrapper around ``CascadeAnalyzer.analyze``."""

    return CascadeAnalyzer(config).analyze(
        series,
        metadata=metadata,
    )


__all__ = [
    "AnalysisRisk",
    "AnalyzerConfig",
    "AnalyzerError",
    "CascadeAnalysis",
    "CascadeAnalyzer",
    "CascadePhase",
    "CoherenceState",
    "PlaneDominance",
    "ScalarTrend",
    "TrendDirection",
    "analyze_observations",
    "calculate_scalar_trend",
]
