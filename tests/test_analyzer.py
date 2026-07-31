"""Tests for the ROIF cascade analyzer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from roif.analyzer import (
    AnalysisRisk,
    AnalyzerConfig,
    AnalyzerError,
    CascadeAnalysis,
    CascadeAnalyzer,
    CascadePhase,
    CoherenceState,
    PlaneDominance,
    ScalarTrend,
    TrendDirection,
    analyze_observations,
    calculate_scalar_trend,
)
from roif.cascade_event import CascadeEventBatch
from roif.history import CascadeHistory
from roif.observer import (
    AlertKind,
    AlertLevel,
    Observation,
    ObservationSeries,
    ObserverAlert,
    SimulationObserver,
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


def make_alert(
    *,
    snapshot_id: str = "snapshot-1",
    time: float = 1.0,
    kind: AlertKind = AlertKind.ETA_WARNING,
    level: AlertLevel = AlertLevel.WARNING,
) -> ObserverAlert:
    return ObserverAlert(
        kind=kind,
        level=level,
        snapshot_id=snapshot_id,
        time=time,
        message="Analyzer test alert",
        value=0.2,
        threshold=0.25,
    )


def make_observation(
    *,
    snapshot_id: str = "snapshot-1",
    time: float = 1.0,
    step_index: int = 1,
    reserve: tuple[float, ...] = (0.2, 0.8),
    eta: float | None = None,
    alerts: tuple[ObserverAlert, ...] = (),
    activation_before: tuple[float, ...] = (0.1, 0.2),
    activation_after: tuple[float, ...] = (0.2, 0.1),
) -> Observation:
    snapshot = make_snapshot(
        snapshot_id=snapshot_id,
        time=time,
        step_index=step_index,
        activation_before=activation_before,
        activation_after=activation_after,
    )
    metrics = SimulationObserver().observe(
        snapshot,
        reserve=reserve,
    ).metrics

    return Observation(
        snapshot=snapshot,
        metrics=metrics,
        eta=eta,
        alerts=alerts,
    )


def make_series(
    etas: tuple[float | None, ...] = (0.5, 0.4),
    *,
    reserves: tuple[tuple[float, ...], ...] | None = None,
    alerts: tuple[tuple[ObserverAlert, ...], ...] | None = None,
    deltas: tuple[float, ...] | None = None,
) -> ObservationSeries:
    count = len(etas)

    if reserves is None:
        reserves = tuple((0.2, 0.8) for _ in range(count))
    if alerts is None:
        alerts = tuple(() for _ in range(count))
    if deltas is None:
        deltas = tuple(0.1 for _ in range(count))

    observations = []
    for index, eta in enumerate(etas, start=1):
        delta = deltas[index - 1]
        observations.append(
            make_observation(
                snapshot_id=f"s{index}",
                time=float(index),
                step_index=index,
                reserve=reserves[index - 1],
                eta=eta,
                alerts=alerts[index - 1],
                activation_before=(0.0, 0.0),
                activation_after=(delta, 0.0),
            )
        )

    return ObservationSeries(tuple(observations))


def make_trend(
    direction: TrendDirection = TrendDirection.STABLE,
) -> ScalarTrend:
    return ScalarTrend(
        direction=direction,
        first_value=1.0,
        final_value=1.0,
        absolute_change=0.0,
        slope=0.0,
        point_count=2,
    )


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


def test_trend_direction_values_are_stable() -> None:
    assert tuple(item.value for item in TrendDirection) == (
        "improving",
        "stable",
        "worsening",
        "unknown",
    )


def test_coherence_state_values_are_stable() -> None:
    assert tuple(item.value for item in CoherenceState) == (
        "stable",
        "warning",
        "critical",
        "unknown",
    )


def test_cascade_phase_values_are_stable() -> None:
    assert tuple(item.value for item in CascadePhase) == (
        "quiescent",
        "compensating",
        "destabilizing",
        "supercritical",
        "recovering",
        "indeterminate",
    )


def test_analysis_risk_values_are_stable() -> None:
    assert tuple(item.value for item in AnalysisRisk) == (
        "low",
        "moderate",
        "high",
        "critical",
    )


# ---------------------------------------------------------------------------
# AnalyzerConfig
# ---------------------------------------------------------------------------


def test_analyzer_config_defaults() -> None:
    config = AnalyzerConfig()

    assert config.eta_warning_threshold == pytest.approx(0.25)
    assert config.eta_critical_threshold == pytest.approx(0.0)
    assert config.stable_slope_tolerance == pytest.approx(1e-9)
    assert config.burden_growth_tolerance == pytest.approx(1e-9)
    assert config.minimum_trend_points == 2


def test_analyzer_config_normalizes_numbers() -> None:
    config = AnalyzerConfig(
        eta_warning_threshold="0.3",
        eta_critical_threshold="-0.2",
        stable_slope_tolerance="0.01",
        burden_growth_tolerance=1,
    )

    assert config.eta_warning_threshold == pytest.approx(0.3)
    assert config.eta_critical_threshold == pytest.approx(-0.2)
    assert config.stable_slope_tolerance == pytest.approx(0.01)
    assert config.burden_growth_tolerance == pytest.approx(1.0)


def test_analyzer_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        AnalyzerConfig().eta_warning_threshold = 0.5  # type: ignore[misc]


def test_analyzer_config_rejects_critical_above_warning() -> None:
    with pytest.raises(AnalyzerError):
        AnalyzerConfig(
            eta_warning_threshold=0.0,
            eta_critical_threshold=0.1,
        )


@pytest.mark.parametrize(
    "field",
    (
        "eta_warning_threshold",
        "eta_critical_threshold",
        "stable_slope_tolerance",
        "burden_growth_tolerance",
    ),
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "bad"))
def test_analyzer_config_rejects_invalid_numeric_values(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(AnalyzerError):
        AnalyzerConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    ("stable_slope_tolerance", "burden_growth_tolerance"),
)
def test_analyzer_config_rejects_negative_tolerances(field: str) -> None:
    with pytest.raises(AnalyzerError):
        AnalyzerConfig(**{field: -0.1})


@pytest.mark.parametrize("value", (True, 1.5, 1, 0, -1, "2"))
def test_analyzer_config_rejects_invalid_minimum_points(value: Any) -> None:
    with pytest.raises(AnalyzerError):
        AnalyzerConfig(minimum_trend_points=value)


# ---------------------------------------------------------------------------
# ScalarTrend
# ---------------------------------------------------------------------------


def test_scalar_trend_properties() -> None:
    trend = ScalarTrend(
        direction=TrendDirection.IMPROVING,
        first_value=1,
        final_value=2,
        absolute_change=1,
        slope=0.5,
        point_count=3,
    )

    assert trend.direction is TrendDirection.IMPROVING
    assert trend.first_value == pytest.approx(1.0)
    assert trend.final_value == pytest.approx(2.0)
    assert trend.absolute_change == pytest.approx(1.0)
    assert trend.slope == pytest.approx(0.5)
    assert trend.point_count == 3


def test_scalar_trend_as_dict() -> None:
    payload = make_trend().as_dict()

    assert payload == {
        "direction": "stable",
        "first_value": 1.0,
        "final_value": 1.0,
        "absolute_change": 0.0,
        "slope": 0.0,
        "point_count": 2,
    }


def test_scalar_trend_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_trend().point_count = 3  # type: ignore[misc]


@pytest.mark.parametrize("direction", ("stable", None, 1))
def test_scalar_trend_rejects_invalid_direction(direction: Any) -> None:
    with pytest.raises(AnalyzerError):
        ScalarTrend(
            direction=direction,
            first_value=None,
            final_value=None,
            absolute_change=None,
            slope=None,
            point_count=0,
        )


@pytest.mark.parametrize("point_count", (True, -1, 1.5, "2"))
def test_scalar_trend_rejects_invalid_point_count(point_count: Any) -> None:
    with pytest.raises(AnalyzerError):
        ScalarTrend(
            direction=TrendDirection.UNKNOWN,
            first_value=None,
            final_value=None,
            absolute_change=None,
            slope=None,
            point_count=point_count,
        )


@pytest.mark.parametrize(
    "field",
    ("first_value", "final_value", "absolute_change", "slope"),
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "bad"))
def test_scalar_trend_rejects_invalid_optional_numbers(
    field: str,
    value: Any,
) -> None:
    values = {
        "direction": TrendDirection.STABLE,
        "first_value": 1.0,
        "final_value": 1.0,
        "absolute_change": 0.0,
        "slope": 0.0,
        "point_count": 2,
    }
    values[field] = value

    with pytest.raises(AnalyzerError):
        ScalarTrend(**values)


# ---------------------------------------------------------------------------
# calculate_scalar_trend
# ---------------------------------------------------------------------------


def test_calculate_scalar_trend_empty_is_unknown() -> None:
    trend = calculate_scalar_trend(
        (),
        improving_when_increasing=True,
    )

    assert trend.direction is TrendDirection.UNKNOWN
    assert trend.point_count == 0
    assert trend.first_value is None
    assert trend.final_value is None


def test_calculate_scalar_trend_one_point_is_unknown() -> None:
    trend = calculate_scalar_trend(
        ((1.0, 4.0),),
        improving_when_increasing=True,
    )

    assert trend.direction is TrendDirection.UNKNOWN
    assert trend.first_value == pytest.approx(4.0)
    assert trend.final_value == pytest.approx(4.0)
    assert trend.absolute_change == pytest.approx(0.0)
    assert trend.slope is None


def test_increasing_series_is_improving_when_requested() -> None:
    trend = calculate_scalar_trend(
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
        improving_when_increasing=True,
    )

    assert trend.direction is TrendDirection.IMPROVING
    assert trend.slope == pytest.approx(1.0)


def test_increasing_series_is_worsening_when_lower_is_better() -> None:
    trend = calculate_scalar_trend(
        ((1.0, 1.0), (2.0, 2.0)),
        improving_when_increasing=False,
    )

    assert trend.direction is TrendDirection.WORSENING


def test_decreasing_series_is_worsening_when_higher_is_better() -> None:
    trend = calculate_scalar_trend(
        ((1.0, 3.0), (2.0, 2.0), (3.0, 1.0)),
        improving_when_increasing=True,
    )

    assert trend.direction is TrendDirection.WORSENING
    assert trend.slope == pytest.approx(-1.0)


def test_decreasing_series_is_improving_when_lower_is_better() -> None:
    trend = calculate_scalar_trend(
        ((1.0, 3.0), (2.0, 2.0)),
        improving_when_increasing=False,
    )

    assert trend.direction is TrendDirection.IMPROVING


def test_flat_series_is_stable() -> None:
    trend = calculate_scalar_trend(
        ((1.0, 2.0), (2.0, 2.0), (3.0, 2.0)),
        improving_when_increasing=True,
    )

    assert trend.direction is TrendDirection.STABLE
    assert trend.slope == pytest.approx(0.0)


def test_tolerance_can_classify_small_slope_as_stable() -> None:
    trend = calculate_scalar_trend(
        ((0.0, 0.0), (1.0, 0.001)),
        improving_when_increasing=True,
        stable_tolerance=0.01,
    )

    assert trend.direction is TrendDirection.STABLE


def test_equal_times_produce_zero_slope() -> None:
    trend = calculate_scalar_trend(
        ((1.0, 1.0), (1.0, 2.0)),
        improving_when_increasing=True,
    )

    assert trend.direction is TrendDirection.STABLE
    assert trend.slope == pytest.approx(0.0)


@pytest.mark.parametrize("points", ("bad", 1, None))
def test_calculate_scalar_trend_rejects_invalid_container(points: Any) -> None:
    with pytest.raises(AnalyzerError):
        calculate_scalar_trend(
            points,
            improving_when_increasing=True,
        )


@pytest.mark.parametrize(
    "item",
    (
        (1.0,),
        (1.0, 2.0, 3.0),
        "ab",
        1,
    ),
)
def test_calculate_scalar_trend_rejects_invalid_point(item: Any) -> None:
    with pytest.raises(AnalyzerError):
        calculate_scalar_trend(
            (item,),
            improving_when_increasing=True,
        )


def test_calculate_scalar_trend_rejects_non_chronological_points() -> None:
    with pytest.raises(AnalyzerError):
        calculate_scalar_trend(
            ((2.0, 1.0), (1.0, 2.0)),
            improving_when_increasing=True,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "bad"))
def test_calculate_scalar_trend_rejects_invalid_time(value: Any) -> None:
    with pytest.raises(AnalyzerError):
        calculate_scalar_trend(
            ((value, 1.0), (2.0, 2.0)),
            improving_when_increasing=True,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "bad"))
def test_calculate_scalar_trend_rejects_invalid_value(value: Any) -> None:
    with pytest.raises(AnalyzerError):
        calculate_scalar_trend(
            ((1.0, value), (2.0, 2.0)),
            improving_when_increasing=True,
        )


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_calculate_scalar_trend_requires_boolean_direction(value: Any) -> None:
    with pytest.raises(AnalyzerError):
        calculate_scalar_trend(
            ((1.0, 1.0), (2.0, 2.0)),
            improving_when_increasing=value,
        )


@pytest.mark.parametrize("value", (-1.0, float("nan"), True, "bad"))
def test_calculate_scalar_trend_rejects_invalid_tolerance(value: Any) -> None:
    with pytest.raises(AnalyzerError):
        calculate_scalar_trend(
            ((1.0, 1.0), (2.0, 2.0)),
            improving_when_increasing=True,
            stable_tolerance=value,
        )


# ---------------------------------------------------------------------------
# PlaneDominance
# ---------------------------------------------------------------------------


def test_plane_dominance_properties() -> None:
    dominance = PlaneDominance(" A ", 3, 0.75)

    assert dominance.plane_id == "A"
    assert dominance.count == 3
    assert dominance.fraction == pytest.approx(0.75)


def test_plane_dominance_as_dict() -> None:
    assert PlaneDominance("A", 2, 0.5).as_dict() == {
        "plane_id": "A",
        "count": 2,
        "fraction": 0.5,
    }


def test_plane_dominance_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        PlaneDominance("A", 1, 1.0).count = 2  # type: ignore[misc]


@pytest.mark.parametrize("plane_id", ("", "   ", None, 1))
def test_plane_dominance_rejects_invalid_plane_id(plane_id: Any) -> None:
    with pytest.raises(AnalyzerError):
        PlaneDominance(plane_id, 1, 1.0)


@pytest.mark.parametrize("count", (0, -1, True, 1.5, "1"))
def test_plane_dominance_rejects_invalid_count(count: Any) -> None:
    with pytest.raises(AnalyzerError):
        PlaneDominance("A", count, 1.0)


@pytest.mark.parametrize(
    "fraction",
    (-0.1, 1.1, float("nan"), float("inf"), True, "bad"),
)
def test_plane_dominance_rejects_invalid_fraction(fraction: Any) -> None:
    with pytest.raises(AnalyzerError):
        PlaneDominance("A", 1, fraction)


# ---------------------------------------------------------------------------
# CascadeAnalysis
# ---------------------------------------------------------------------------


def make_analysis(**overrides: Any) -> CascadeAnalysis:
    values = {
        "observation_count": 2,
        "phase": CascadePhase.QUIESCENT,
        "risk": AnalysisRisk.LOW,
        "coherence_state": CoherenceState.STABLE,
        "eta_trend": make_trend(),
        "burden_trend": make_trend(),
        "delta_trend": make_trend(),
        "dominant_d_fast": PlaneDominance("A", 2, 1.0),
        "d_fast_switch_count": 0,
        "alert_count": 0,
        "critical_alert_count": 0,
        "warning_alert_count": 0,
        "first_snapshot_id": "s1",
        "final_snapshot_id": "s2",
        "summary": " Stable summary. ",
        "metadata": {"source": "test"},
    }
    values.update(overrides)
    return CascadeAnalysis(**values)


def test_cascade_analysis_properties() -> None:
    analysis = make_analysis()

    assert analysis.summary == "Stable summary."
    assert analysis.observation_count == 2
    assert analysis.has_critical_alerts is False
    assert analysis.is_supercritical is False
    assert isinstance(analysis.metadata, MappingProxyType)


def test_cascade_analysis_metadata_is_copied_and_frozen() -> None:
    source = {"source": "test"}
    analysis = make_analysis(metadata=source)
    source["source"] = "changed"

    assert analysis.metadata["source"] == "test"
    with pytest.raises(TypeError):
        analysis.metadata["new"] = 1  # type: ignore[index]


def test_cascade_analysis_supercritical_properties() -> None:
    analysis = make_analysis(
        phase=CascadePhase.SUPERCRITICAL,
        risk=AnalysisRisk.CRITICAL,
        critical_alert_count=1,
    )

    assert analysis.has_critical_alerts
    assert analysis.is_supercritical


def test_cascade_analysis_as_dict() -> None:
    payload = make_analysis().as_dict()

    assert payload["phase"] == "quiescent"
    assert payload["risk"] == "low"
    assert payload["coherence_state"] == "stable"
    assert payload["dominant_d_fast"]["plane_id"] == "A"
    assert payload["metadata"] == {"source": "test"}


@pytest.mark.parametrize(
    "field",
    (
        "observation_count",
        "d_fast_switch_count",
        "alert_count",
        "critical_alert_count",
        "warning_alert_count",
    ),
)
@pytest.mark.parametrize("value", (-1, True, 1.5, "1"))
def test_cascade_analysis_rejects_invalid_counts(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(AnalyzerError):
        make_analysis(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("phase", "quiescent"),
        ("risk", "low"),
        ("coherence_state", "stable"),
        ("eta_trend", None),
        ("burden_trend", None),
        ("delta_trend", None),
        ("dominant_d_fast", "A"),
    ),
)
def test_cascade_analysis_rejects_invalid_typed_fields(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(AnalyzerError):
        make_analysis(**{field: value})


@pytest.mark.parametrize("summary", ("", "   ", None, 1))
def test_cascade_analysis_rejects_invalid_summary(summary: Any) -> None:
    with pytest.raises(AnalyzerError):
        make_analysis(summary=summary)


@pytest.mark.parametrize("snapshot_id", ("", "   ", 1))
@pytest.mark.parametrize(
    "field",
    ("first_snapshot_id", "final_snapshot_id"),
)
def test_cascade_analysis_rejects_invalid_snapshot_ids(
    field: str,
    snapshot_id: Any,
) -> None:
    with pytest.raises(AnalyzerError):
        make_analysis(**{field: snapshot_id})


@pytest.mark.parametrize(
    "metadata",
    ("bad", {1: "bad"}, {" ": "bad"}),
)
def test_cascade_analysis_rejects_invalid_metadata(metadata: Any) -> None:
    with pytest.raises(AnalyzerError):
        make_analysis(metadata=metadata)


# ---------------------------------------------------------------------------
# CascadeAnalyzer
# ---------------------------------------------------------------------------


def test_cascade_analyzer_defaults() -> None:
    analyzer = CascadeAnalyzer()

    assert isinstance(analyzer.config, AnalyzerConfig)


def test_cascade_analyzer_accepts_config() -> None:
    config = AnalyzerConfig(eta_warning_threshold=0.4)
    analyzer = CascadeAnalyzer(config)

    assert analyzer.config is config


@pytest.mark.parametrize("config", ({}, "bad", 1))
def test_cascade_analyzer_rejects_invalid_config(config: Any) -> None:
    with pytest.raises(AnalyzerError):
        CascadeAnalyzer(config)


@pytest.mark.parametrize("series", (None, (), "bad"))
def test_analyze_rejects_invalid_series(series: Any) -> None:
    with pytest.raises(AnalyzerError):
        CascadeAnalyzer().analyze(series)


def test_analyze_empty_series() -> None:
    analysis = CascadeAnalyzer().analyze(ObservationSeries(()))

    assert analysis.observation_count == 0
    assert analysis.phase is CascadePhase.INDETERMINATE
    assert analysis.risk is AnalysisRisk.LOW
    assert analysis.coherence_state is CoherenceState.UNKNOWN
    assert analysis.first_snapshot_id is None
    assert analysis.final_snapshot_id is None
    assert analysis.dominant_d_fast is None


def test_analyze_stable_series_is_quiescent_low_risk() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((0.5, 0.5))
    )

    assert analysis.phase is CascadePhase.QUIESCENT
    assert analysis.risk is AnalysisRisk.LOW
    assert analysis.coherence_state is CoherenceState.STABLE
    assert analysis.eta_trend.direction is TrendDirection.STABLE


def test_analyze_warning_eta_is_destabilizing_high_risk() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((0.5, 0.2))
    )

    assert analysis.coherence_state is CoherenceState.WARNING
    assert analysis.phase is CascadePhase.DESTABILIZING
    assert analysis.risk is AnalysisRisk.HIGH


def test_analyze_critical_eta_is_supercritical() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((0.2, 0.0))
    )

    assert analysis.coherence_state is CoherenceState.CRITICAL
    assert analysis.phase is CascadePhase.SUPERCRITICAL
    assert analysis.risk is AnalysisRisk.CRITICAL
    assert analysis.is_supercritical


def test_analyze_negative_eta_is_critical() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((0.1, -0.1))
    )

    assert analysis.coherence_state is CoherenceState.CRITICAL
    assert analysis.risk is AnalysisRisk.CRITICAL


def test_analyze_improving_eta_is_recovering() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((0.3, 0.6))
    )

    assert analysis.eta_trend.direction is TrendDirection.IMPROVING
    assert analysis.phase is CascadePhase.RECOVERING
    assert analysis.risk is AnalysisRisk.LOW


def test_analyze_alert_in_stable_state_is_compensating() -> None:
    alert = make_alert(snapshot_id="s2", time=2.0)
    analysis = CascadeAnalyzer().analyze(
        make_series(
            (0.5, 0.5),
            alerts=((), (alert,)),
        )
    )

    assert analysis.phase is CascadePhase.COMPENSATING
    assert analysis.risk is AnalysisRisk.MODERATE
    assert analysis.alert_count == 1
    assert analysis.warning_alert_count == 1


def test_analyze_critical_alert_forces_critical_risk() -> None:
    alert = make_alert(
        snapshot_id="s2",
        time=2.0,
        kind=AlertKind.ETA_CRITICAL,
        level=AlertLevel.CRITICAL,
    )
    analysis = CascadeAnalyzer().analyze(
        make_series(
            (0.5, 0.5),
            alerts=((), (alert,)),
        )
    )

    assert analysis.critical_alert_count == 1
    assert analysis.risk is AnalysisRisk.CRITICAL
    assert analysis.has_critical_alerts


def test_analyze_omits_none_eta_values_from_trend() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((None, 0.4, None, 0.6))
    )

    assert analysis.eta_trend.point_count == 2
    assert analysis.eta_trend.first_value == pytest.approx(0.4)
    assert analysis.eta_trend.final_value == pytest.approx(0.6)


def test_analyze_all_none_eta_is_unknown() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((None, None))
    )

    assert analysis.coherence_state is CoherenceState.UNKNOWN
    assert analysis.eta_trend.direction is TrendDirection.UNKNOWN


def test_analyze_detects_dominant_d_fast() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series(
            (0.5, 0.5, 0.5),
            reserves=(
                (0.1, 0.9),
                (0.2, 0.8),
                (0.9, 0.1),
            ),
        )
    )

    assert analysis.dominant_d_fast is not None
    assert analysis.dominant_d_fast.plane_id == "A"
    assert analysis.dominant_d_fast.count == 2
    assert analysis.dominant_d_fast.fraction == pytest.approx(2 / 3)


def test_dominance_tie_uses_first_appearance() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series(
            (0.5, 0.5),
            reserves=((0.9, 0.1), (0.1, 0.9)),
        )
    )

    assert analysis.dominant_d_fast is not None
    assert analysis.dominant_d_fast.plane_id == "B"


def test_analyze_counts_d_fast_switches() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series(
            (0.5, 0.5, 0.5, 0.5),
            reserves=(
                (0.1, 0.9),
                (0.9, 0.1),
                (0.9, 0.1),
                (0.1, 0.9),
            ),
        )
    )

    assert analysis.d_fast_switch_count == 2


def test_analyze_detects_worsening_delta() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series(
            (0.5, 0.5, 0.5),
            deltas=(0.1, 0.2, 0.3),
        )
    )

    assert analysis.delta_trend.direction is TrendDirection.WORSENING


def test_analyze_detects_improving_delta() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series(
            (0.5, 0.5, 0.5),
            deltas=(0.3, 0.2, 0.1),
        )
    )

    assert analysis.delta_trend.direction is TrendDirection.IMPROVING


def test_analyze_records_snapshot_ids() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((0.5, 0.4, 0.3))
    )

    assert analysis.first_snapshot_id == "s1"
    assert analysis.final_snapshot_id == "s3"


def test_analyze_copies_metadata() -> None:
    metadata = {"experiment": "E1"}
    analysis = CascadeAnalyzer().analyze(
        make_series((0.5, 0.5)),
        metadata=metadata,
    )
    metadata["experiment"] = "changed"

    assert analysis.metadata["experiment"] == "E1"


def test_analyze_summary_contains_core_findings() -> None:
    analysis = CascadeAnalyzer().analyze(
        make_series((0.5, 0.2))
    )

    assert "Cascade phase: destabilizing." in analysis.summary
    assert "Overall risk: high." in analysis.summary
    assert "Coherence state: warning." in analysis.summary
    assert "Dominant D_fast:" in analysis.summary


def test_custom_thresholds_change_coherence_state() -> None:
    analyzer = CascadeAnalyzer(
        AnalyzerConfig(
            eta_warning_threshold=0.6,
            eta_critical_threshold=0.3,
        )
    )
    analysis = analyzer.analyze(make_series((0.7, 0.4)))

    assert analysis.coherence_state is CoherenceState.WARNING


def test_custom_minimum_points_can_leave_trend_unknown() -> None:
    analyzer = CascadeAnalyzer(
        AnalyzerConfig(minimum_trend_points=3)
    )
    analysis = analyzer.analyze(make_series((0.4, 0.5)))

    assert analysis.eta_trend.direction is TrendDirection.UNKNOWN


def test_analyze_observations_wrapper() -> None:
    analysis = analyze_observations(
        make_series((0.5, 0.5)),
        metadata={"wrapper": True},
    )

    assert isinstance(analysis, CascadeAnalysis)
    assert analysis.metadata["wrapper"] is True


def test_analyze_observations_forwards_config() -> None:
    analysis = analyze_observations(
        make_series((0.5, 0.4)),
        config=AnalyzerConfig(eta_warning_threshold=0.45),
    )

    assert analysis.coherence_state is CoherenceState.WARNING
