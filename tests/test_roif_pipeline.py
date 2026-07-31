"""End-to-end integration tests for the complete ROIF pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from roif.analyzer import (
    AnalysisRisk,
    CascadeAnalyzer,
    CascadePhase,
    CoherenceState,
)
from roif.cascade_event import CascadeEventBatch
from roif.executor import (
    ActionOutcome,
    CascadeExecutor,
    ExecutionMode,
    ExecutionStatus,
    ExecutorConfig,
    FeedbackState,
)
from roif.history import CascadeHistory
from roif.observer import AlertKind, ObservationSeries, ObserverConfig, SimulationObserver
from roif.plane_kernel import PlaneKernelResult
from roif.planner import (
    CascadePlanner,
    InterventionCandidate,
    PlanStatus,
    PlannerConfig,
    TargetRole,
)
from roif.predictor import (
    CascadePredictor,
    PredictionStatus,
    PredictorConfig,
    StabilityForecast,
)
from roif.state_snapshot import StateSnapshot


class DeterministicClock:
    def __init__(self, start: float = 100.0, step: float = 0.1) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def make_snapshot(
    *,
    snapshot_id: str,
    time: float,
    step_index: int,
    activation_before: tuple[float, float],
    activation_after: tuple[float, float],
) -> StateSnapshot:
    before = np.asarray(activation_before, dtype=float)
    after = np.asarray(activation_after, dtype=float)

    result = PlaneKernelResult(
        plane_ids=("A", "B"),
        time=time,
        dt=1.0,
        activation_before=tuple(before),
        interaction=(0.0, 0.0),
        rate=tuple(after - before),
        activation_after=tuple(after),
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


def build_series(
    eta_values: tuple[float, ...],
    reserves: tuple[tuple[float, float], ...],
    *,
    activation_deltas: tuple[float, ...] | None = None,
    observer_config: ObserverConfig | None = None,
) -> ObservationSeries:
    if activation_deltas is None:
        activation_deltas = tuple(0.1 for _ in eta_values)

    observer = SimulationObserver(observer_config)

    for index, (eta, reserve, delta) in enumerate(
        zip(eta_values, reserves, activation_deltas),
        start=1,
    ):
        snapshot = make_snapshot(
            snapshot_id=f"s{index}",
            time=float(index),
            step_index=index,
            activation_before=(0.0, 0.0),
            activation_after=(delta, 0.0),
        )
        operator_value = 1.0 - eta
        observer.observe(
            snapshot,
            reserve=reserve,
            operator=((operator_value, 0.0), (0.0, 0.1)),
            lambda_critical=1.0,
            metadata={"pipeline_step": index},
        )

    return observer.series


def candidate(
    candidate_id: str,
    *,
    target_id: str,
    target_role: TargetRole,
    expected_eta_gain: float,
    risk: float = 0.1,
    evidence_strength: float = 0.9,
    reversibility: float = 0.9,
    cascade_risk: float = 0.1,
    external_system_risk: float = 0.0,
) -> InterventionCandidate:
    return InterventionCandidate(
        candidate_id=candidate_id,
        target_id=target_id,
        target_role=target_role,
        expected_eta_gain=expected_eta_gain,
        cost=1.0,
        risk=risk,
        delay=1.0,
        evidence_strength=evidence_strength,
        reversibility=reversibility,
        cascade_risk=cascade_risk,
        external_system_risk=external_system_risk,
        rationale="Pipeline candidate.",
        tags=("pipeline",),
        metadata={"source": "integration"},
    )


def execute_measured(plan, eta_before: float, eta_after: float):
    executor = CascadeExecutor(
        clock=DeterministicClock(),
        id_factory=lambda: "pipeline-execution",
    )
    return executor.execute(
        plan,
        lambda _: ActionOutcome(
            applied=True,
            eta_before=eta_before,
            eta_after=eta_after,
            message="Synthetic intervention applied.",
        ),
        operator_confirmed=True,
    )


def test_pipeline_stable_recovering_system() -> None:
    series = build_series(
        (0.45, 0.50, 0.55),
        ((0.15, 0.85), (0.20, 0.80), (0.25, 0.75)),
        activation_deltas=(0.15, 0.10, 0.05),
    )
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "node-star-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.20,
            ),
            candidate(
                "support-B",
                target_id="B",
                target_role=TargetRole.SUPPORT,
                expected_eta_gain=0.10,
            ),
        ),
        prediction=prediction,
    )
    result = execute_measured(plan, 0.55, 0.75)

    assert analysis.coherence_state is CoherenceState.STABLE
    assert analysis.phase is CascadePhase.RECOVERING
    assert analysis.risk is AnalysisRisk.LOW
    assert prediction.status is PredictionStatus.AVAILABLE
    assert prediction.stability is StabilityForecast.STABLE
    assert plan.status is PlanStatus.AVAILABLE
    assert plan.node_star == "A"
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.feedback.state is FeedbackState.MATCHED


def test_pipeline_destabilizing_system() -> None:
    series = build_series(
        (0.50, 0.38, 0.28),
        ((0.10, 0.90), (0.08, 0.92), (0.05, 0.95)),
        activation_deltas=(0.10, 0.20, 0.30),
    )
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor(PredictorConfig(horizon=1.0)).predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "restore-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.25,
            ),
            candidate(
                "fast-A",
                target_id="A",
                target_role=TargetRole.D_FAST,
                expected_eta_gain=0.15,
            ),
        ),
        prediction=prediction,
    )
    result = execute_measured(plan, 0.28, 0.53)

    assert analysis.phase is CascadePhase.DESTABILIZING
    assert analysis.risk is AnalysisRisk.HIGH
    assert prediction.eta_slope < 0.0
    assert plan.recommended.candidate.candidate_id == "restore-A"
    assert result.status is ExecutionStatus.SUCCEEDED


def test_pipeline_supercritical_context_sets_full_urgency() -> None:
    series = build_series(
        (0.15, 0.05, -0.05),
        ((0.08, 0.92), (0.05, 0.95), (0.02, 0.98)),
    )
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "critical-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.35,
            ),
        ),
        prediction=prediction,
    )

    assert analysis.phase is CascadePhase.SUPERCRITICAL
    assert analysis.risk is AnalysisRisk.CRITICAL
    assert prediction.stability is StabilityForecast.CRITICAL
    assert plan.recommended.score.urgency == pytest.approx(1.0)


def test_pipeline_non_fonit_execution_gate_blocks_action() -> None:
    series = build_series(
        (0.45, 0.35, 0.25),
        ((0.10, 0.90),) * 3,
    )
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)

    plan = CascadePlanner(
        PlannerConfig(non_fonit_gate_enabled=False)
    ).plan(
        analysis,
        (
            candidate(
                "unsafe-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.40,
                cascade_risk=0.95,
                external_system_risk=0.80,
            ),
        ),
        prediction=prediction,
    )

    called = False

    def action(_: InterventionCandidate) -> ActionOutcome:
        nonlocal called
        called = True
        return ActionOutcome(eta_before=0.25, eta_after=0.65)

    result = CascadeExecutor(
        clock=DeterministicClock(),
        id_factory=lambda: "blocked",
    ).execute(
        plan,
        action,
        operator_confirmed=True,
    )

    assert plan.status is PlanStatus.AVAILABLE
    assert result.status is ExecutionStatus.BLOCKED
    assert called is False


def test_pipeline_requires_operator_confirmation() -> None:
    series = build_series((0.50, 0.42, 0.34), ((0.10, 0.90),) * 3)
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "candidate-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.20,
            ),
        ),
        prediction=prediction,
    )

    result = CascadeExecutor(
        clock=DeterministicClock(),
        id_factory=lambda: "confirmation",
    ).execute(plan, lambda _: None, operator_confirmed=False)

    assert result.status is ExecutionStatus.BLOCKED


def test_pipeline_dry_run_does_not_apply_action() -> None:
    series = build_series((0.40, 0.35, 0.30), ((0.10, 0.90),) * 3)
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "dry-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.20,
            ),
        ),
        prediction=prediction,
    )

    called = False

    def action(_: InterventionCandidate) -> None:
        nonlocal called
        called = True

    result = CascadeExecutor(
        ExecutorConfig(mode=ExecutionMode.DRY_RUN),
        clock=DeterministicClock(),
        id_factory=lambda: "dry",
    ).execute(plan, action, operator_confirmed=True)

    assert result.status is ExecutionStatus.DRY_RUN
    assert called is False


def test_pipeline_partial_effect() -> None:
    series = build_series((0.48, 0.40, 0.32), ((0.10, 0.90),) * 3)
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "partial-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.20,
            ),
        ),
        prediction=prediction,
    )
    result = execute_measured(plan, 0.32, 0.44)

    assert result.feedback.state is FeedbackState.UNDERPERFORMED
    assert result.feedback.effect_ratio == pytest.approx(0.60)
    assert result.status is ExecutionStatus.PARTIAL


def test_pipeline_harmful_effect() -> None:
    series = build_series((0.45, 0.37, 0.29), ((0.10, 0.90),) * 3)
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "harmful-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.20,
            ),
        ),
        prediction=prediction,
    )
    result = execute_measured(plan, 0.29, 0.20)

    assert result.status is ExecutionStatus.FAILED
    assert result.feedback.state is FeedbackState.HARMFUL
    assert result.error_type == "HarmfulEffect"


def test_pipeline_preserves_d_fast_and_node_star_separation() -> None:
    series = build_series(
        (0.50, 0.43, 0.36),
        ((0.05, 0.95), (0.06, 0.94), (0.07, 0.93)),
    )
    analysis = CascadeAnalyzer().analyze(series)
    prediction = CascadePredictor().predict(series)
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "node-star-B",
                target_id="B",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.30,
            ),
            candidate(
                "d-fast-A",
                target_id="A",
                target_role=TargetRole.D_FAST,
                expected_eta_gain=0.10,
            ),
        ),
        prediction=prediction,
    )

    assert analysis.dominant_d_fast.plane_id == "A"
    assert plan.node_star == "B"


def test_pipeline_alerts_flow_into_analysis() -> None:
    series = build_series(
        (0.50, 0.20, -0.05),
        ((0.10, 0.90),) * 3,
        observer_config=ObserverConfig(
            eta_warning_threshold=0.25,
            eta_critical_threshold=0.0,
        ),
    )
    analysis = CascadeAnalyzer().analyze(series)

    assert any(a.kind is AlertKind.ETA_WARNING for a in series.alerts)
    assert any(a.kind is AlertKind.ETA_CRITICAL for a in series.alerts)
    assert analysis.critical_alert_count >= 1
    assert analysis.risk is AnalysisRisk.CRITICAL


def test_pipeline_serialization_across_layers() -> None:
    series = build_series((0.50, 0.42, 0.34), ((0.10, 0.90),) * 3)
    analysis = CascadeAnalyzer().analyze(series, metadata={"layer": "analysis"})
    prediction = CascadePredictor().predict(
        series,
        metadata={"layer": "prediction"},
    )
    plan = CascadePlanner().plan(
        analysis,
        (
            candidate(
                "serialize-A",
                target_id="A",
                target_role=TargetRole.NODE_STAR,
                expected_eta_gain=0.20,
            ),
        ),
        prediction=prediction,
        metadata={"layer": "planner"},
    )
    result = execute_measured(plan, 0.34, 0.54)

    assert series.as_dict()["observation_count"] == 3
    assert analysis.as_dict()["metadata"] == {"layer": "analysis"}
    assert prediction.as_dict()["metadata"] == {"layer": "prediction"}
    assert plan.as_dict()["metadata"] == {"layer": "planner"}
    assert result.as_dict()["feedback"]["state"] == "matched"


@pytest.mark.parametrize(
    ("etas", "phase"),
    (
        ((0.50, 0.50, 0.50), CascadePhase.QUIESCENT),
        ((0.30, 0.45, 0.60), CascadePhase.RECOVERING),
        ((0.50, 0.38, 0.28), CascadePhase.DESTABILIZING),
        ((0.10, 0.02, -0.05), CascadePhase.SUPERCRITICAL),
    ),
)
def test_pipeline_phase_regression(
    etas: tuple[float, float, float],
    phase: CascadePhase,
) -> None:
    series = build_series(etas, ((0.10, 0.90),) * 3)
    assert CascadeAnalyzer().analyze(series).phase is phase
