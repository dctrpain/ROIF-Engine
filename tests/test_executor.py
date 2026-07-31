"""Tests for the ROIF controlled intervention executor."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest

from roif.analyzer import AnalysisRisk, CascadePhase
from roif.executor import (
    ActionOutcome,
    CascadeExecutor,
    ExecutionFeedback,
    ExecutionGateReason,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    ExecutionStep,
    ExecutorConfig,
    ExecutorError,
    FeedbackState,
    execute_plan,
)
from roif.planner import (
    CandidateDecision,
    InterventionCandidate,
    InterventionPlan,
    PlanStatus,
    RankedIntervention,
    ScoreBreakdown,
    TargetRole,
)
from roif.predictor import StabilityForecast


class IncrementingClock:
    """Deterministic monotonic clock for execution tests."""

    def __init__(self, start: float = 10.0, step: float = 0.25) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def make_candidate(
    candidate_id: str = "c1",
    *,
    target_id: str = "A",
    target_role: TargetRole = TargetRole.NODE_STAR,
    expected_eta_gain: float = 0.2,
    cost: float = 1.0,
    risk: float = 0.1,
    delay: float = 1.0,
    evidence_strength: float = 0.8,
    reversibility: float = 0.9,
    cascade_risk: float = 0.1,
    external_system_risk: float = 0.0,
) -> InterventionCandidate:
    return InterventionCandidate(
        candidate_id=candidate_id,
        target_id=target_id,
        target_role=target_role,
        expected_eta_gain=expected_eta_gain,
        cost=cost,
        risk=risk,
        delay=delay,
        evidence_strength=evidence_strength,
        reversibility=reversibility,
        cascade_risk=cascade_risk,
        external_system_risk=external_system_risk,
        rationale="Test candidate.",
        tags=("test",),
        metadata={"source": "unit"},
    )


def make_score(utility: float = 0.5) -> ScoreBreakdown:
    return ScoreBreakdown(
        benefit=1.0,
        urgency=0.5,
        target_alignment=1.0,
        evidence=0.8,
        reversibility=0.9,
        cost_penalty=0.2,
        risk_penalty=0.1,
        delay_penalty=0.2,
        utility=utility,
    )


def make_ranked(
    candidate: InterventionCandidate | None = None,
    *,
    decision: CandidateDecision = CandidateDecision.RECOMMENDED,
    rank: int = 1,
) -> RankedIntervention:
    return RankedIntervention(
        rank=rank,
        decision=decision,
        candidate=candidate or make_candidate(),
        score=make_score(),
        explanation="Planner explanation.",
    )


def make_plan(
    candidate: InterventionCandidate | None = None,
    *,
    status: PlanStatus = PlanStatus.AVAILABLE,
    recommended: RankedIntervention | None | object = ...,
    alternatives: tuple[RankedIntervention, ...] = (),
) -> InterventionPlan:
    if recommended is ...:
        recommended = (
            make_ranked(candidate)
            if status is PlanStatus.AVAILABLE
            else None
        )

    return InterventionPlan(
        status=status,
        analysis_phase=CascadePhase.QUIESCENT,
        analysis_risk=AnalysisRisk.LOW,
        prediction_stability=StabilityForecast.STABLE,
        recommended=recommended,  # type: ignore[arg-type]
        alternatives=alternatives,
        rejected=(),
        vetoed=(),
        candidate_count=(
            int(recommended is not None) + len(alternatives)
        ),
        summary="Plan summary.",
        metadata={"plan": "test"},
    )


def make_feedback(
    state: FeedbackState = FeedbackState.MATCHED,
    *,
    expected_eta_gain: float = 0.2,
    actual_eta_gain: float | None = 0.2,
    effect_error: float | None = 0.0,
    effect_ratio: float | None = 1.0,
    eta_before: float | None = 0.4,
    eta_after: float | None = 0.6,
) -> ExecutionFeedback:
    return ExecutionFeedback(
        state=state,
        expected_eta_gain=expected_eta_gain,
        actual_eta_gain=actual_eta_gain,
        effect_error=effect_error,
        effect_ratio=effect_ratio,
        eta_before=eta_before,
        eta_after=eta_after,
    )


def make_outcome(
    *,
    applied: bool = True,
    eta_before: float | None = 0.4,
    eta_after: float | None = 0.6,
) -> ActionOutcome:
    return ActionOutcome(
        applied=applied,
        eta_before=eta_before,
        eta_after=eta_after,
        actual_cost=1.0,
        actual_delay=0.5,
        actual_risk=0.1,
        message="Applied.",
        warnings=("observe",),
        metadata={"adapter": "test"},
    )


def make_steps() -> tuple[ExecutionStep, ...]:
    return (
        ExecutionStep(1, "select", True, "Selected.", 10.0),
        ExecutionStep(2, "gate", True, "Passed.", 10.1),
    )


def make_result(
    *,
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
    gate_reasons: tuple[ExecutionGateReason, ...] = (),
    error_type: str | None = None,
    error_message: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        execution_id="exec-1",
        status=status,
        candidate=make_candidate(),
        feedback=make_feedback(),
        outcome=make_outcome(),
        gate_reasons=gate_reasons,
        steps=make_steps(),
        started_at=10.0,
        finished_at=11.0,
        duration=1.0,
        summary="Execution summary.",
        error_type=error_type,
        error_message=error_message,
        metadata={"source": "unit"},
    )


def make_executor(
    config: ExecutorConfig | None = None,
) -> CascadeExecutor:
    return CascadeExecutor(
        config,
        clock=IncrementingClock(),
        id_factory=lambda: "exec-test",
    )


def test_execution_status_values_are_stable() -> None:
    assert tuple(item.value for item in ExecutionStatus) == (
        "succeeded",
        "partial",
        "failed",
        "blocked",
        "skipped",
        "dry_run",
    )


def test_execution_mode_values_are_stable() -> None:
    assert tuple(item.value for item in ExecutionMode) == (
        "apply",
        "dry_run",
    )


def test_feedback_state_values_are_stable() -> None:
    assert tuple(item.value for item in FeedbackState) == (
        "exceeded",
        "matched",
        "underperformed",
        "harmful",
        "unmeasured",
    )


def test_gate_reason_values_are_stable() -> None:
    assert tuple(item.value for item in ExecutionGateReason) == (
        "plan_unavailable",
        "no_recommendation",
        "invalid_recommendation",
        "cascade_risk",
        "external_system_risk",
        "irreversible_high_risk",
        "low_evidence",
        "operator_confirmation_required",
    )


def test_executor_config_defaults() -> None:
    config = ExecutorConfig()
    assert config.mode is ExecutionMode.APPLY
    assert config.require_operator_confirmation is True
    assert config.non_fonit_gate_enabled is True
    assert config.maximum_cascade_risk == pytest.approx(0.70)
    assert config.effect_tolerance == pytest.approx(0.02)
    assert config.partial_effect_ratio == pytest.approx(0.50)


def test_executor_config_normalizes_numbers() -> None:
    config = ExecutorConfig(
        maximum_cascade_risk="0.5",
        effect_tolerance="0.1",
        partial_effect_ratio="0.75",
    )
    assert config.maximum_cascade_risk == pytest.approx(0.5)
    assert config.effect_tolerance == pytest.approx(0.1)
    assert config.partial_effect_ratio == pytest.approx(0.75)


def test_executor_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        ExecutorConfig().effect_tolerance = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("value", ("apply", None, 1))
def test_config_rejects_invalid_mode(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ExecutorConfig(mode=value)


@pytest.mark.parametrize(
    "field",
    (
        "require_operator_confirmation",
        "non_fonit_gate_enabled",
        "allow_alternative_fallback",
        "capture_exceptions",
        "generate_execution_id",
    ),
)
@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_config_rejects_non_boolean_flags(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ExecutorError):
        ExecutorConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "maximum_cascade_risk",
        "maximum_external_system_risk",
        "irreversible_risk_threshold",
        "minimum_evidence_strength",
        "partial_effect_ratio",
    ),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, 1.1, float("nan"), float("inf"), True, "bad"),
)
def test_config_rejects_invalid_unit_interval_values(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ExecutorError):
        ExecutorConfig(**{field: value})


@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), float("inf"), True, "bad"),
)
def test_config_rejects_invalid_effect_tolerance(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ExecutorConfig(effect_tolerance=value)


def test_action_outcome_properties() -> None:
    outcome = make_outcome()
    assert outcome.applied
    assert outcome.actual_eta_gain == pytest.approx(0.2)
    assert outcome.warnings == ("observe",)
    assert isinstance(outcome.metadata, MappingProxyType)


def test_action_outcome_as_dict() -> None:
    payload = make_outcome().as_dict()
    assert payload["applied"] is True
    assert payload["actual_eta_gain"] == pytest.approx(0.2)
    assert payload["warnings"] == ["observe"]
    assert payload["metadata"] == {"adapter": "test"}


def test_action_outcome_metadata_is_copied_and_frozen() -> None:
    metadata = {"x": 1}
    outcome = ActionOutcome(metadata=metadata)
    metadata["x"] = 2
    assert outcome.metadata["x"] == 1
    with pytest.raises(TypeError):
        outcome.metadata["y"] = 2  # type: ignore[index]


def test_action_outcome_uses_metadata_eta_before() -> None:
    outcome = ActionOutcome(
        eta_after=0.7,
        metadata={"eta_before": 0.5},
    )
    assert outcome.eta_before == pytest.approx(0.5)
    assert outcome.actual_eta_gain == pytest.approx(0.2)


def test_action_outcome_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_outcome().applied = False  # type: ignore[misc]


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_outcome_rejects_invalid_applied(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ActionOutcome(applied=value)


@pytest.mark.parametrize(
    "field",
    ("eta_before", "eta_after"),
)
@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_outcome_rejects_invalid_eta_values(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ExecutorError):
        ActionOutcome(**{field: value})


@pytest.mark.parametrize(
    "field",
    ("actual_cost", "actual_delay"),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), True, "bad"),
)
def test_outcome_rejects_invalid_nonnegative_values(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ExecutorError):
        ActionOutcome(**{field: value})


@pytest.mark.parametrize(
    "value",
    (-0.1, 1.1, float("nan"), True, "bad"),
)
def test_outcome_rejects_invalid_actual_risk(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ActionOutcome(actual_risk=value)


@pytest.mark.parametrize("value", ("", "   ", 1))
def test_outcome_rejects_invalid_message(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ActionOutcome(message=value)


@pytest.mark.parametrize("value", ("warning", (1,), ("",)))
def test_outcome_rejects_invalid_warnings(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ActionOutcome(warnings=value)


def test_unmeasured_feedback_properties() -> None:
    feedback = make_feedback(
        FeedbackState.UNMEASURED,
        actual_eta_gain=None,
        effect_error=None,
        effect_ratio=None,
        eta_after=None,
    )
    assert feedback.is_beneficial is None
    assert feedback.state is FeedbackState.UNMEASURED


def test_measured_feedback_properties() -> None:
    feedback = make_feedback()
    assert feedback.is_beneficial is True
    assert feedback.effect_ratio == pytest.approx(1.0)


def test_feedback_as_dict() -> None:
    payload = make_feedback().as_dict()
    assert payload["state"] == "matched"
    assert payload["is_beneficial"] is True


@pytest.mark.parametrize("value", ("matched", None, 1))
def test_feedback_rejects_invalid_state(value: Any) -> None:
    with pytest.raises(ExecutorError):
        make_feedback(state=value)


@pytest.mark.parametrize(
    "field",
    (
        "expected_eta_gain",
        "actual_eta_gain",
        "effect_error",
        "effect_ratio",
        "eta_before",
        "eta_after",
    ),
)
@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_feedback_rejects_invalid_numbers(
    field: str,
    value: Any,
) -> None:
    values = {
        "state": FeedbackState.MATCHED,
        "expected_eta_gain": 0.2,
        "actual_eta_gain": 0.2,
        "effect_error": 0.0,
        "effect_ratio": 1.0,
        "eta_before": 0.4,
        "eta_after": 0.6,
    }
    values[field] = value
    with pytest.raises(ExecutorError):
        ExecutionFeedback(**values)


def test_unmeasured_feedback_rejects_measured_values() -> None:
    with pytest.raises(ExecutorError):
        make_feedback(
            FeedbackState.UNMEASURED,
            actual_eta_gain=0.1,
            effect_error=0.0,
            effect_ratio=1.0,
        )


@pytest.mark.parametrize(
    "field",
    ("actual_eta_gain", "effect_error"),
)
def test_measured_feedback_requires_core_values(field: str) -> None:
    values = {
        "state": FeedbackState.MATCHED,
        "expected_eta_gain": 0.2,
        "actual_eta_gain": 0.2,
        "effect_error": 0.0,
        "effect_ratio": 1.0,
        "eta_before": 0.4,
        "eta_after": 0.6,
    }
    values[field] = None
    with pytest.raises(ExecutorError):
        ExecutionFeedback(**values)


def test_execution_step_properties() -> None:
    step = ExecutionStep(
        sequence=1,
        name=" select ",
        succeeded=True,
        message=" selected ",
        timestamp="10.5",
        metadata={"x": 1},
    )
    assert step.name == "select"
    assert step.message == "selected"
    assert step.timestamp == pytest.approx(10.5)
    assert isinstance(step.metadata, MappingProxyType)


def test_execution_step_as_dict() -> None:
    payload = make_steps()[0].as_dict()
    assert payload["sequence"] == 1
    assert payload["name"] == "select"
    assert payload["succeeded"] is True


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "1"))
def test_step_rejects_invalid_sequence(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ExecutionStep(value, "step", True, "message", 1.0)


@pytest.mark.parametrize("field", ("name", "message"))
@pytest.mark.parametrize("value", ("", "   ", None, 1))
def test_step_rejects_invalid_text(
    field: str,
    value: Any,
) -> None:
    values = {
        "sequence": 1,
        "name": "step",
        "succeeded": True,
        "message": "message",
        "timestamp": 1.0,
    }
    values[field] = value
    with pytest.raises(ExecutorError):
        ExecutionStep(**values)


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_step_rejects_invalid_succeeded(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ExecutionStep(1, "step", value, "message", 1.0)


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_step_rejects_invalid_timestamp(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ExecutionStep(1, "step", True, "message", value)


def test_execution_result_properties() -> None:
    result = make_result()
    assert result.succeeded
    assert result.was_applied
    assert result.node_star == "A"
    assert isinstance(result.metadata, MappingProxyType)


def test_execution_result_as_dict() -> None:
    payload = make_result().as_dict()
    assert payload["execution_id"] == "exec-1"
    assert payload["status"] == "succeeded"
    assert payload["candidate"]["candidate_id"] == "c1"
    assert payload["feedback"]["state"] == "matched"


def test_blocked_result_is_valid_with_reason() -> None:
    result = ExecutionResult(
        execution_id="exec-blocked",
        status=ExecutionStatus.BLOCKED,
        candidate=make_candidate(),
        feedback=None,
        outcome=None,
        gate_reasons=(
            ExecutionGateReason.OPERATOR_CONFIRMATION_REQUIRED,
        ),
        steps=make_steps(),
        started_at=10.0,
        finished_at=10.5,
        duration=0.5,
        summary="Blocked.",
    )
    assert not result.succeeded
    assert not result.was_applied


def test_failed_result_requires_error_details() -> None:
    result = make_result(
        status=ExecutionStatus.FAILED,
        error_type="Failure",
        error_message="Failed.",
    )
    assert result.status is ExecutionStatus.FAILED


@pytest.mark.parametrize("value", ("", "   ", None, 1))
def test_result_rejects_invalid_execution_id(value: Any) -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            execution_id=value,
            status=ExecutionStatus.SUCCEEDED,
            candidate=None,
            feedback=None,
            outcome=None,
            gate_reasons=(),
            steps=(),
            started_at=1.0,
            finished_at=1.0,
            duration=0.0,
            summary="Done.",
        )


@pytest.mark.parametrize("value", ("succeeded", None, 1))
def test_result_rejects_invalid_status(value: Any) -> None:
    with pytest.raises(ExecutorError):
        make_result(status=value)


def test_result_rejects_invalid_candidate() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.SUCCEEDED, "bad", None, None, (),
            (), 1.0, 1.0, 0.0, "Done.",
        )


def test_result_rejects_invalid_feedback() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.SUCCEEDED, None, "bad", None, (),
            (), 1.0, 1.0, 0.0, "Done.",
        )


def test_result_rejects_invalid_outcome() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.SUCCEEDED, None, None, "bad", (),
            (), 1.0, 1.0, 0.0, "Done.",
        )


def test_result_rejects_invalid_gate_reason_member() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.BLOCKED, None, None, None, ("bad",),
            (), 1.0, 1.0, 0.0, "Blocked.",
        )


def test_result_rejects_non_contiguous_steps() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e",
            ExecutionStatus.SUCCEEDED,
            None,
            None,
            None,
            (),
            (ExecutionStep(2, "step", True, "Done.", 1.0),),
            1.0,
            1.0,
            0.0,
            "Done.",
        )


def test_result_rejects_finished_before_started() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.SUCCEEDED, None, None, None, (),
            (), 2.0, 1.0, 0.0, "Done.",
        )


def test_result_rejects_duration_mismatch() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.SUCCEEDED, None, None, None, (),
            (), 1.0, 2.0, 0.5, "Done.",
        )


def test_blocked_result_requires_reason() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.BLOCKED, None, None, None, (),
            (), 1.0, 1.0, 0.0, "Blocked.",
        )


def test_non_blocked_result_disallows_reasons() -> None:
    with pytest.raises(ExecutorError):
        ExecutionResult(
            "e", ExecutionStatus.SUCCEEDED, None, None, None,
            (ExecutionGateReason.CASCADE_RISK,),
            (), 1.0, 1.0, 0.0, "Done.",
        )


def test_failed_result_requires_error_type_and_message() -> None:
    with pytest.raises(ExecutorError):
        make_result(status=ExecutionStatus.FAILED)


def test_non_failed_result_disallows_error_details() -> None:
    with pytest.raises(ExecutorError):
        make_result(
            error_type="Error",
            error_message="Unexpected.",
        )


def test_executor_defaults() -> None:
    executor = CascadeExecutor()
    assert isinstance(executor.config, ExecutorConfig)


def test_executor_accepts_dependencies() -> None:
    config = ExecutorConfig()
    executor = CascadeExecutor(
        config,
        clock=IncrementingClock(),
        id_factory=lambda: "fixed",
    )
    assert executor.config is config


@pytest.mark.parametrize("config", ({}, "bad", 1))
def test_executor_rejects_invalid_config(config: Any) -> None:
    with pytest.raises(ExecutorError):
        CascadeExecutor(config)


@pytest.mark.parametrize("clock", (1, "clock", None))
def test_executor_rejects_invalid_explicit_clock(clock: Any) -> None:
    if clock is None:
        return
    with pytest.raises(ExecutorError):
        CascadeExecutor(clock=clock)


@pytest.mark.parametrize("factory", (1, "factory"))
def test_executor_rejects_invalid_id_factory(factory: Any) -> None:
    with pytest.raises(ExecutorError):
        CascadeExecutor(id_factory=factory)


@pytest.mark.parametrize("plan", (None, "bad", 1))
def test_execute_rejects_invalid_plan(plan: Any) -> None:
    with pytest.raises(ExecutorError):
        make_executor().execute(plan, None)


@pytest.mark.parametrize("action", ("bad", 1, object()))
def test_execute_rejects_invalid_action(action: Any) -> None:
    with pytest.raises(ExecutorError):
        make_executor().execute(make_plan(), action)


@pytest.mark.parametrize(
    "eta_before",
    (float("nan"), float("inf"), True, "bad"),
)
def test_execute_rejects_invalid_eta_before(eta_before: Any) -> None:
    with pytest.raises(ExecutorError):
        make_executor().execute(
            make_plan(),
            lambda _: None,
            operator_confirmed=True,
            eta_before=eta_before,
        )


def test_missing_operator_confirmation_blocks_execution() -> None:
    called = False

    def action(_: InterventionCandidate) -> None:
        nonlocal called
        called = True

    result = make_executor().execute(make_plan(), action)
    assert result.status is ExecutionStatus.BLOCKED
    assert ExecutionGateReason.OPERATOR_CONFIRMATION_REQUIRED in (
        result.gate_reasons
    )
    assert called is False


def test_operator_confirmation_can_be_disabled() -> None:
    executor = make_executor(
        ExecutorConfig(require_operator_confirmation=False)
    )
    result = executor.execute(
        make_plan(),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.6),
    )
    assert result.status is ExecutionStatus.SUCCEEDED


def test_unavailable_plan_is_blocked() -> None:
    plan = make_plan(
        status=PlanStatus.NO_CANDIDATES,
        recommended=None,
    )
    result = make_executor().execute(
        plan,
        None,
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.BLOCKED
    assert ExecutionGateReason.PLAN_UNAVAILABLE in result.gate_reasons
    assert ExecutionGateReason.NO_RECOMMENDATION in result.gate_reasons


def test_non_fonit_cascade_risk_blocks_execution() -> None:
    plan = make_plan(make_candidate(cascade_risk=0.9))
    result = make_executor().execute(
        plan,
        None,
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.BLOCKED
    assert ExecutionGateReason.CASCADE_RISK in result.gate_reasons


def test_external_system_risk_blocks_execution() -> None:
    plan = make_plan(make_candidate(external_system_risk=0.9))
    result = make_executor().execute(
        plan,
        None,
        operator_confirmed=True,
    )
    assert ExecutionGateReason.EXTERNAL_SYSTEM_RISK in result.gate_reasons


def test_irreversible_high_risk_blocks_execution() -> None:
    plan = make_plan(
        make_candidate(risk=0.9, reversibility=0.1)
    )
    result = make_executor().execute(
        plan,
        None,
        operator_confirmed=True,
    )
    assert ExecutionGateReason.IRREVERSIBLE_HIGH_RISK in (
        result.gate_reasons
    )


def test_low_evidence_blocks_execution() -> None:
    config = ExecutorConfig(minimum_evidence_strength=0.9)
    plan = make_plan(make_candidate(evidence_strength=0.2))
    result = make_executor(config).execute(
        plan,
        None,
        operator_confirmed=True,
    )
    assert ExecutionGateReason.LOW_EVIDENCE in result.gate_reasons


def test_non_fonit_gate_can_be_disabled() -> None:
    config = ExecutorConfig(
        non_fonit_gate_enabled=False,
        require_operator_confirmation=False,
    )
    plan = make_plan(
        make_candidate(
            cascade_risk=1.0,
            external_system_risk=1.0,
            risk=1.0,
            reversibility=0.0,
        )
    )
    result = make_executor(config).execute(
        plan,
        lambda _: None,
    )
    assert result.status is ExecutionStatus.SUCCEEDED


def test_dry_run_does_not_call_action() -> None:
    called = False

    def action(_: InterventionCandidate) -> None:
        nonlocal called
        called = True

    config = ExecutorConfig(mode=ExecutionMode.DRY_RUN)
    result = make_executor(config).execute(
        make_plan(),
        action,
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.DRY_RUN
    assert result.outcome is None
    assert result.feedback is None
    assert called is False
    assert result.steps[-1].name == "dry_run"


def test_apply_mode_requires_action_after_gate_passes() -> None:
    with pytest.raises(ExecutorError):
        make_executor().execute(
            make_plan(),
            None,
            operator_confirmed=True,
        )


def test_action_receives_selected_candidate() -> None:
    received: list[InterventionCandidate] = []

    def action(item: InterventionCandidate) -> ActionOutcome:
        received.append(item)
        return ActionOutcome(eta_before=0.4, eta_after=0.6)

    plan = make_plan(make_candidate("selected"))
    result = make_executor().execute(
        plan,
        action,
        operator_confirmed=True,
    )
    assert received == [plan.recommended.candidate]
    assert result.candidate.candidate_id == "selected"


def test_action_outcome_return_is_used() -> None:
    outcome = ActionOutcome(
        eta_before=0.4,
        eta_after=0.65,
        message="Measured.",
    )
    result = make_executor().execute(
        make_plan(),
        lambda _: outcome,
        operator_confirmed=True,
    )
    assert result.outcome is outcome
    assert result.feedback.actual_eta_gain == pytest.approx(0.25)


def test_mapping_outcome_is_coerced() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: {
            "applied": True,
            "eta_before": 0.4,
            "eta_after": 0.6,
            "message": "Mapped.",
        },
        operator_confirmed=True,
    )
    assert isinstance(result.outcome, ActionOutcome)
    assert result.outcome.message == "Mapped."
    assert result.status is ExecutionStatus.SUCCEEDED


def test_numeric_outcome_is_eta_after() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: 0.65,
        operator_confirmed=True,
        eta_before=0.4,
    )
    assert result.outcome.eta_after == pytest.approx(0.65)
    assert result.feedback.actual_eta_gain == pytest.approx(0.25)


def test_none_outcome_is_successful_unmeasured_application() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: None,
        operator_confirmed=True,
        eta_before=0.4,
    )
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.feedback.state is FeedbackState.UNMEASURED
    assert result.outcome.applied is True


def test_eta_before_argument_fills_outcome() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: ActionOutcome(eta_after=0.6),
        operator_confirmed=True,
        eta_before=0.4,
    )
    assert result.outcome.eta_before == pytest.approx(0.4)
    assert result.feedback.actual_eta_gain == pytest.approx(0.2)


def test_bool_adapter_result_is_captured_as_failure() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: True,
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "ExecutorError"


def test_mapping_with_unknown_field_is_failure() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: {"eta_after": 0.6, "unknown": 1},
        operator_confirmed=True,
        eta_before=0.4,
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "ExecutorError"


def test_unsupported_adapter_result_is_failure() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: object(),
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "ExecutorError"


def test_exception_is_captured() -> None:
    def action(_: InterventionCandidate) -> None:
        raise RuntimeError("boom")

    result = make_executor().execute(
        make_plan(),
        action,
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "RuntimeError"
    assert result.error_message == "boom"
    assert result.steps[-1].name == "execution_error"


def test_exception_can_propagate() -> None:
    def action(_: InterventionCandidate) -> None:
        raise RuntimeError("boom")

    executor = make_executor(
        ExecutorConfig(capture_exceptions=False)
    )
    with pytest.raises(RuntimeError, match="boom"):
        executor.execute(
            make_plan(),
            action,
            operator_confirmed=True,
        )


def test_matching_feedback_succeeds() -> None:
    result = make_executor().execute(
        make_plan(make_candidate(expected_eta_gain=0.2)),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.6),
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.feedback.state is FeedbackState.MATCHED


def test_exceeded_feedback_succeeds() -> None:
    result = make_executor().execute(
        make_plan(make_candidate(expected_eta_gain=0.2)),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.7),
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.feedback.state is FeedbackState.EXCEEDED


def test_underperformed_feedback_can_be_partial() -> None:
    result = make_executor().execute(
        make_plan(make_candidate(expected_eta_gain=0.2)),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.52),
        operator_confirmed=True,
    )
    assert result.feedback.state is FeedbackState.UNDERPERFORMED
    assert result.status is ExecutionStatus.PARTIAL


def test_underperformed_feedback_can_fail() -> None:
    result = make_executor().execute(
        make_plan(make_candidate(expected_eta_gain=0.2)),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.45),
        operator_confirmed=True,
    )
    assert result.feedback.state is FeedbackState.UNDERPERFORMED
    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "InsufficientEffect"


def test_harmful_feedback_fails() -> None:
    result = make_executor().execute(
        make_plan(make_candidate(expected_eta_gain=0.2)),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.3),
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.feedback.state is FeedbackState.HARMFUL
    assert result.error_type == "HarmfulEffect"


def test_not_applied_outcome_fails() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: ActionOutcome(
            applied=False,
            eta_before=0.4,
            eta_after=0.4,
            message="Rejected by adapter.",
        ),
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "ActionNotApplied"
    assert result.error_message == "Rejected by adapter."


def test_effect_tolerance_controls_match() -> None:
    config = ExecutorConfig(effect_tolerance=0.05)
    result = make_executor(config).execute(
        make_plan(make_candidate(expected_eta_gain=0.2)),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.57),
        operator_confirmed=True,
    )
    assert result.feedback.state is FeedbackState.MATCHED


def test_partial_effect_ratio_controls_status() -> None:
    config = ExecutorConfig(partial_effect_ratio=0.75)
    result = make_executor(config).execute(
        make_plan(make_candidate(expected_eta_gain=0.2)),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.52),
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.FAILED


def test_execution_audit_steps_are_contiguous() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.6),
        operator_confirmed=True,
    )
    assert tuple(step.sequence for step in result.steps) == (
        1, 2, 3, 4, 5
    )
    assert tuple(step.name for step in result.steps) == (
        "select_intervention",
        "execution_gate",
        "apply_intervention",
        "normalize_outcome",
        "evaluate_feedback",
    )


def test_execution_id_comes_from_factory() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: None,
        operator_confirmed=True,
    )
    assert result.execution_id == "exec-test"


def test_execution_id_can_be_fixed() -> None:
    config = ExecutorConfig(generate_execution_id=False)
    result = make_executor(config).execute(
        make_plan(),
        lambda _: None,
        operator_confirmed=True,
    )
    assert result.execution_id == "execution"


def test_invalid_generated_execution_id_is_rejected() -> None:
    executor = CascadeExecutor(
        clock=IncrementingClock(),
        id_factory=lambda: "   ",
    )
    with pytest.raises(ExecutorError):
        executor.execute(
            make_plan(),
            lambda _: None,
            operator_confirmed=True,
        )


def test_metadata_is_forwarded_and_frozen() -> None:
    metadata = {"experiment": "E1"}
    result = make_executor().execute(
        make_plan(),
        lambda _: None,
        operator_confirmed=True,
        metadata=metadata,
    )
    metadata["experiment"] = "changed"
    assert result.metadata["experiment"] == "E1"
    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_result_duration_uses_clock() -> None:
    clock = IncrementingClock(start=1.0, step=0.5)
    executor = CascadeExecutor(
        clock=clock,
        id_factory=lambda: "e",
    )
    result = executor.execute(
        make_plan(),
        lambda _: None,
        operator_confirmed=True,
    )
    assert result.finished_at > result.started_at
    assert result.duration == pytest.approx(
        result.finished_at - result.started_at
    )


def test_execution_summary_contains_feedback() -> None:
    result = make_executor().execute(
        make_plan(),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.6),
        operator_confirmed=True,
    )
    assert "expected eta gain" in result.summary
    assert "actual eta gain" in result.summary
    assert "matched" in result.summary


def test_blocked_summary_contains_gate_reason() -> None:
    result = make_executor().execute(
        make_plan(),
        None,
        operator_confirmed=False,
    )
    assert "operator_confirmation_required" in result.summary


def test_dry_run_summary_mentions_no_action() -> None:
    result = make_executor(
        ExecutorConfig(mode=ExecutionMode.DRY_RUN)
    ).execute(
        make_plan(),
        None,
        operator_confirmed=True,
    )
    assert "no action was applied" in result.summary


def test_execute_plan_wrapper() -> None:
    result = execute_plan(
        make_plan(),
        lambda _: ActionOutcome(eta_before=0.4, eta_after=0.6),
        config=ExecutorConfig(
            require_operator_confirmation=False
        ),
        metadata={"wrapper": True},
    )
    assert isinstance(result, ExecutionResult)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.metadata["wrapper"] is True


def test_execute_plan_wrapper_forwards_confirmation() -> None:
    result = execute_plan(
        make_plan(),
        lambda _: None,
        operator_confirmed=True,
    )
    assert result.status is ExecutionStatus.SUCCEEDED
