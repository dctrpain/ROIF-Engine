"""
Controlled intervention execution and feedback capture for the ROIF engine.

The executor is the boundary between an intervention plan and an external
system. It deliberately does not know how to mutate a specific simulation,
network, patient model, or device. Instead, the caller supplies an action
adapter implementing one of these contracts:

    action(candidate) -> ActionOutcome
    action(candidate) -> Mapping[str, Any]
    action(candidate) -> float
    action(candidate) -> None

A float is interpreted as the measured eta after execution. A mapping may
contain fields accepted by ``ActionOutcome``. ``None`` means that the action
was applied but no measurement was returned.

This design keeps the ROIF core deterministic, testable, and independent of
domain-specific side effects. The Non-Fonit execution gate remains active
even when a plan was already screened by the planner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
import math
import time
import uuid

from .planner import (
    CandidateDecision,
    InterventionCandidate,
    InterventionPlan,
    PlanStatus,
    RankedIntervention,
)


class ExecutorError(ValueError):
    """Raised when execution input or configuration is invalid."""


_EPSILON = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ExecutorError(f"{name} must be a real number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutorError(f"{name} must be a real number.") from exc

    if not math.isfinite(result):
        raise ExecutorError(f"{name} must be finite.")

    if strictly_positive and result <= 0.0:
        raise ExecutorError(f"{name} must be greater than zero.")

    if minimum is not None and result < minimum:
        raise ExecutorError(
            f"{name} must be greater than or equal to {minimum}."
        )

    if maximum is not None and result > maximum:
        raise ExecutorError(
            f"{name} must be less than or equal to {maximum}."
        )

    return result


def _optional_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite_float(
        value,
        name=name,
        minimum=minimum,
        maximum=maximum,
    )


def _normalized_text(
    value: Any,
    *,
    name: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None

    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise ExecutorError(f"{name} must be a string{suffix}.")

    result = value.strip()
    if not result:
        raise ExecutorError(f"{name} cannot be empty.")

    return result


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise ExecutorError("metadata must be a mapping.")

    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized = _normalized_text(key, name="metadata key")
        copied[normalized] = value

    return MappingProxyType(copied)


def _normalize_messages(
    values: Sequence[str] | None,
    *,
    name: str,
) -> tuple[str, ...]:
    if values is None:
        return ()

    if isinstance(values, (str, bytes)):
        raise ExecutorError(f"{name} must be a sequence of strings.")

    try:
        source = tuple(values)
    except TypeError as exc:
        raise ExecutorError(
            f"{name} must be a sequence of strings."
        ) from exc

    normalized: list[str] = []
    for item in source:
        text = _normalized_text(item, name=f"{name} item")
        if text not in normalized:
            normalized.append(text)

    return tuple(normalized)


class ExecutionStatus(str, Enum):
    """Final state of an execution attempt."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


class ExecutionMode(str, Enum):
    """How an intervention is executed."""

    APPLY = "apply"
    DRY_RUN = "dry_run"


class FeedbackState(str, Enum):
    """Interpretation of measured effect relative to expectation."""

    EXCEEDED = "exceeded"
    MATCHED = "matched"
    UNDERPERFORMED = "underperformed"
    HARMFUL = "harmful"
    UNMEASURED = "unmeasured"


class ExecutionGateReason(str, Enum):
    """Reason why execution was blocked before calling the adapter."""

    PLAN_UNAVAILABLE = "plan_unavailable"
    NO_RECOMMENDATION = "no_recommendation"
    INVALID_RECOMMENDATION = "invalid_recommendation"
    CASCADE_RISK = "cascade_risk"
    EXTERNAL_SYSTEM_RISK = "external_system_risk"
    IRREVERSIBLE_HIGH_RISK = "irreversible_high_risk"
    LOW_EVIDENCE = "low_evidence"
    OPERATOR_CONFIRMATION_REQUIRED = "operator_confirmation_required"


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Execution rules and feedback tolerances."""

    mode: ExecutionMode = ExecutionMode.APPLY
    require_operator_confirmation: bool = True
    non_fonit_gate_enabled: bool = True
    maximum_cascade_risk: float = 0.70
    maximum_external_system_risk: float = 0.20
    irreversible_risk_threshold: float = 0.35
    minimum_evidence_strength: float = 0.0
    effect_tolerance: float = 0.02
    partial_effect_ratio: float = 0.50
    allow_alternative_fallback: bool = False
    capture_exceptions: bool = True
    generate_execution_id: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExecutionMode):
            raise ExecutorError("mode must be an ExecutionMode.")

        for name in (
            "require_operator_confirmation",
            "non_fonit_gate_enabled",
            "allow_alternative_fallback",
            "capture_exceptions",
            "generate_execution_id",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ExecutorError(f"{name} must be a bool.")

        for name in (
            "maximum_cascade_risk",
            "maximum_external_system_risk",
            "irreversible_risk_threshold",
            "minimum_evidence_strength",
            "partial_effect_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

        object.__setattr__(
            self,
            "effect_tolerance",
            _finite_float(
                self.effect_tolerance,
                name="effect_tolerance",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """Raw result returned by a domain-specific action adapter."""

    applied: bool = True
    eta_before: float | None = None
    eta_after: float | None = None
    actual_cost: float | None = None
    actual_delay: float | None = None
    actual_risk: float | None = None
    message: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise ExecutorError("applied must be a bool.")

        object.__setattr__(
            self,
            "eta_before",
            _optional_float(self.eta_before, name="eta_before"),
        )
        object.__setattr__(
            self,
            "eta_after",
            _optional_float(self.eta_after, name="eta_after"),
        )
        object.__setattr__(
            self,
            "actual_cost",
            _optional_float(
                self.actual_cost,
                name="actual_cost",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "actual_delay",
            _optional_float(
                self.actual_delay,
                name="actual_delay",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "actual_risk",
            _optional_float(
                self.actual_risk,
                name="actual_risk",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "message",
            _normalized_text(
                self.message,
                name="message",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "warnings",
            _normalize_messages(self.warnings, name="warnings"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

        if (
            self.eta_before is None
            and self.eta_after is not None
            and "eta_before" in self.metadata
        ):
            object.__setattr__(
                self,
                "eta_before",
                _finite_float(
                    self.metadata["eta_before"],
                    name="metadata eta_before",
                ),
            )

    @property
    def actual_eta_gain(self) -> float | None:
        if self.eta_before is None or self.eta_after is None:
            return None
        return self.eta_after - self.eta_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "eta_before": self.eta_before,
            "eta_after": self.eta_after,
            "actual_eta_gain": self.actual_eta_gain,
            "actual_cost": self.actual_cost,
            "actual_delay": self.actual_delay,
            "actual_risk": self.actual_risk,
            "message": self.message,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExecutionFeedback:
    """Comparison between expected and measured intervention effect."""

    state: FeedbackState
    expected_eta_gain: float
    actual_eta_gain: float | None
    effect_error: float | None
    effect_ratio: float | None
    eta_before: float | None
    eta_after: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, FeedbackState):
            raise ExecutorError("state must be a FeedbackState.")

        object.__setattr__(
            self,
            "expected_eta_gain",
            _finite_float(
                self.expected_eta_gain,
                name="expected_eta_gain",
            ),
        )

        for name in (
            "actual_eta_gain",
            "effect_error",
            "effect_ratio",
            "eta_before",
            "eta_after",
        ):
            object.__setattr__(
                self,
                name,
                _optional_float(getattr(self, name), name=name),
            )

        if self.state is FeedbackState.UNMEASURED:
            if (
                self.actual_eta_gain is not None
                or self.effect_error is not None
                or self.effect_ratio is not None
            ):
                raise ExecutorError(
                    "Unmeasured feedback cannot contain measured effect values."
                )
        elif self.actual_eta_gain is None or self.effect_error is None:
            raise ExecutorError(
                "Measured feedback requires actual_eta_gain and effect_error."
            )

    @property
    def is_beneficial(self) -> bool | None:
        if self.actual_eta_gain is None:
            return None
        return self.actual_eta_gain > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "expected_eta_gain": self.expected_eta_gain,
            "actual_eta_gain": self.actual_eta_gain,
            "effect_error": self.effect_error,
            "effect_ratio": self.effect_ratio,
            "eta_before": self.eta_before,
            "eta_after": self.eta_after,
            "is_beneficial": self.is_beneficial,
        }


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    """One immutable event in the execution audit trail."""

    sequence: int
    name: str
    succeeded: bool
    message: str
    timestamp: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ExecutorError(
                "sequence must be an integer greater than zero."
            )

        object.__setattr__(self, "name", _normalized_text(self.name, name="name"))

        if not isinstance(self.succeeded, bool):
            raise ExecutorError("succeeded must be a bool.")

        object.__setattr__(
            self,
            "message",
            _normalized_text(self.message, name="message"),
        )
        object.__setattr__(
            self,
            "timestamp",
            _finite_float(self.timestamp, name="timestamp"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "name": self.name,
            "succeeded": self.succeeded,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Final immutable result of an execution attempt."""

    execution_id: str
    status: ExecutionStatus
    candidate: InterventionCandidate | None
    feedback: ExecutionFeedback | None
    outcome: ActionOutcome | None
    gate_reasons: tuple[ExecutionGateReason, ...]
    steps: tuple[ExecutionStep, ...]
    started_at: float
    finished_at: float
    duration: float
    summary: str
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            _normalized_text(self.execution_id, name="execution_id"),
        )

        if not isinstance(self.status, ExecutionStatus):
            raise ExecutorError("status must be an ExecutionStatus.")

        if self.candidate is not None and not isinstance(
            self.candidate,
            InterventionCandidate,
        ):
            raise ExecutorError(
                "candidate must be an InterventionCandidate or None."
            )

        if self.feedback is not None and not isinstance(
            self.feedback,
            ExecutionFeedback,
        ):
            raise ExecutorError(
                "feedback must be an ExecutionFeedback or None."
            )

        if self.outcome is not None and not isinstance(
            self.outcome,
            ActionOutcome,
        ):
            raise ExecutorError("outcome must be an ActionOutcome or None.")

        reasons = tuple(self.gate_reasons)
        if any(
            not isinstance(item, ExecutionGateReason)
            for item in reasons
        ):
            raise ExecutorError(
                "gate_reasons must contain ExecutionGateReason values."
            )
        object.__setattr__(self, "gate_reasons", reasons)

        steps = tuple(self.steps)
        if any(not isinstance(item, ExecutionStep) for item in steps):
            raise ExecutorError(
                "steps must contain ExecutionStep values."
            )
        if tuple(item.sequence for item in steps) != tuple(
            range(1, len(steps) + 1)
        ):
            raise ExecutorError(
                "step sequence values must be contiguous and start at one."
            )
        object.__setattr__(self, "steps", steps)

        started_at = _finite_float(self.started_at, name="started_at")
        finished_at = _finite_float(self.finished_at, name="finished_at")
        duration = _finite_float(
            self.duration,
            name="duration",
            minimum=0.0,
        )

        if finished_at + _EPSILON < started_at:
            raise ExecutorError(
                "finished_at cannot be earlier than started_at."
            )

        if abs(duration - (finished_at - started_at)) > 1e-6:
            raise ExecutorError(
                "duration must equal finished_at - started_at."
            )

        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "finished_at", finished_at)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(
            self,
            "summary",
            _normalized_text(self.summary, name="summary"),
        )
        object.__setattr__(
            self,
            "error_type",
            _normalized_text(
                self.error_type,
                name="error_type",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "error_message",
            _normalized_text(
                self.error_message,
                name="error_message",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

        if self.status is ExecutionStatus.BLOCKED and not reasons:
            raise ExecutorError(
                "Blocked execution requires at least one gate reason."
            )

        if self.status is not ExecutionStatus.BLOCKED and reasons:
            raise ExecutorError(
                "Only blocked execution may contain gate reasons."
            )

        if self.status is ExecutionStatus.FAILED:
            if self.error_type is None or self.error_message is None:
                raise ExecutorError(
                    "Failed execution requires error_type and error_message."
                )
        elif self.error_type is not None or self.error_message is not None:
            raise ExecutorError(
                "Only failed execution may contain error details."
            )

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.SUCCEEDED

    @property
    def was_applied(self) -> bool:
        return self.outcome is not None and self.outcome.applied

    @property
    def node_star(self) -> str | None:
        return None if self.candidate is None else self.candidate.target_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "candidate": (
                None if self.candidate is None else self.candidate.as_dict()
            ),
            "feedback": (
                None if self.feedback is None else self.feedback.as_dict()
            ),
            "outcome": (
                None if self.outcome is None else self.outcome.as_dict()
            ),
            "gate_reasons": [item.value for item in self.gate_reasons],
            "steps": [item.as_dict() for item in self.steps],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "summary": self.summary,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class InterventionAction(Protocol):
    """Callable contract for domain-specific intervention adapters."""

    def __call__(
        self,
        candidate: InterventionCandidate,
    ) -> ActionOutcome | Mapping[str, Any] | float | None:
        ...


def _gate_reasons(
    plan: InterventionPlan,
    ranked: RankedIntervention | None,
    config: ExecutorConfig,
    *,
    operator_confirmed: bool,
) -> tuple[ExecutionGateReason, ...]:
    reasons: list[ExecutionGateReason] = []

    if plan.status is not PlanStatus.AVAILABLE:
        reasons.append(ExecutionGateReason.PLAN_UNAVAILABLE)

    if ranked is None:
        reasons.append(ExecutionGateReason.NO_RECOMMENDATION)
    elif ranked.decision not in (
        CandidateDecision.RECOMMENDED,
        CandidateDecision.ALTERNATIVE,
    ):
        reasons.append(ExecutionGateReason.INVALID_RECOMMENDATION)

    if config.require_operator_confirmation and not operator_confirmed:
        reasons.append(
            ExecutionGateReason.OPERATOR_CONFIRMATION_REQUIRED
        )

    if ranked is not None and config.non_fonit_gate_enabled:
        candidate = ranked.candidate

        if candidate.cascade_risk > config.maximum_cascade_risk:
            reasons.append(ExecutionGateReason.CASCADE_RISK)

        if (
            candidate.external_system_risk
            > config.maximum_external_system_risk
        ):
            reasons.append(ExecutionGateReason.EXTERNAL_SYSTEM_RISK)

        if (
            candidate.reversibility < 0.5
            and candidate.risk
            > config.irreversible_risk_threshold
        ):
            reasons.append(
                ExecutionGateReason.IRREVERSIBLE_HIGH_RISK
            )

        if (
            candidate.evidence_strength
            < config.minimum_evidence_strength
        ):
            reasons.append(ExecutionGateReason.LOW_EVIDENCE)

    unique: list[ExecutionGateReason] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)

    return tuple(unique)


def _select_ranked_intervention(
    plan: InterventionPlan,
    config: ExecutorConfig,
) -> RankedIntervention | None:
    if plan.recommended is not None:
        return plan.recommended

    if config.allow_alternative_fallback and plan.alternatives:
        return plan.alternatives[0]

    return None


def _coerce_outcome(
    value: ActionOutcome | Mapping[str, Any] | float | None,
    *,
    eta_before: float | None,
) -> ActionOutcome:
    if isinstance(value, ActionOutcome):
        if value.eta_before is None and eta_before is not None:
            return ActionOutcome(
                applied=value.applied,
                eta_before=eta_before,
                eta_after=value.eta_after,
                actual_cost=value.actual_cost,
                actual_delay=value.actual_delay,
                actual_risk=value.actual_risk,
                message=value.message,
                warnings=value.warnings,
                metadata=value.metadata,
            )
        return value

    if value is None:
        return ActionOutcome(
            applied=True,
            eta_before=eta_before,
            message="Action adapter returned no measurement.",
        )

    if isinstance(value, bool):
        raise ExecutorError(
            "Action adapter cannot return bool; return ActionOutcome instead."
        )

    if isinstance(value, (int, float)):
        return ActionOutcome(
            applied=True,
            eta_before=eta_before,
            eta_after=_finite_float(value, name="adapter eta_after"),
        )

    if isinstance(value, Mapping):
        allowed = {
            "applied",
            "eta_before",
            "eta_after",
            "actual_cost",
            "actual_delay",
            "actual_risk",
            "message",
            "warnings",
            "metadata",
        }
        unknown = set(value) - allowed
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ExecutorError(
                f"Action outcome mapping contains unknown fields: {names}."
            )

        payload = dict(value)
        if payload.get("eta_before") is None and eta_before is not None:
            payload["eta_before"] = eta_before

        try:
            return ActionOutcome(**payload)
        except TypeError as exc:
            raise ExecutorError(
                "Action outcome mapping has invalid fields."
            ) from exc

    raise ExecutorError(
        "Action adapter must return ActionOutcome, mapping, float, or None."
    )


def _feedback(
    candidate: InterventionCandidate,
    outcome: ActionOutcome,
    config: ExecutorConfig,
) -> ExecutionFeedback:
    expected = candidate.expected_eta_gain
    actual = outcome.actual_eta_gain

    if actual is None:
        return ExecutionFeedback(
            state=FeedbackState.UNMEASURED,
            expected_eta_gain=expected,
            actual_eta_gain=None,
            effect_error=None,
            effect_ratio=None,
            eta_before=outcome.eta_before,
            eta_after=outcome.eta_after,
        )

    error = actual - expected
    ratio = None
    if abs(expected) > _EPSILON:
        ratio = actual / expected

    if actual < -config.effect_tolerance:
        state = FeedbackState.HARMFUL
    elif abs(error) <= config.effect_tolerance:
        state = FeedbackState.MATCHED
    elif actual > expected + config.effect_tolerance:
        state = FeedbackState.EXCEEDED
    else:
        state = FeedbackState.UNDERPERFORMED

    return ExecutionFeedback(
        state=state,
        expected_eta_gain=expected,
        actual_eta_gain=actual,
        effect_error=error,
        effect_ratio=ratio,
        eta_before=outcome.eta_before,
        eta_after=outcome.eta_after,
    )


def _status_from_outcome(
    outcome: ActionOutcome,
    feedback: ExecutionFeedback,
    config: ExecutorConfig,
) -> ExecutionStatus:
    if not outcome.applied:
        return ExecutionStatus.FAILED

    if feedback.state is FeedbackState.HARMFUL:
        return ExecutionStatus.FAILED

    if feedback.state is FeedbackState.UNMEASURED:
        return ExecutionStatus.SUCCEEDED

    if feedback.state in (
        FeedbackState.MATCHED,
        FeedbackState.EXCEEDED,
    ):
        return ExecutionStatus.SUCCEEDED

    expected = feedback.expected_eta_gain
    actual = feedback.actual_eta_gain

    if (
        actual is not None
        and expected > _EPSILON
        and actual / expected >= config.partial_effect_ratio
    ):
        return ExecutionStatus.PARTIAL

    return ExecutionStatus.FAILED


def _summary(
    status: ExecutionStatus,
    candidate: InterventionCandidate | None,
    feedback: ExecutionFeedback | None,
    reasons: Sequence[ExecutionGateReason] = (),
) -> str:
    if status is ExecutionStatus.BLOCKED:
        joined = ", ".join(item.value for item in reasons)
        return f"Execution was blocked by the safety gate: {joined}."

    if status is ExecutionStatus.SKIPPED:
        return "Execution was skipped because no executable intervention was available."

    if candidate is None:
        return f"Execution finished with status {status.value}."

    if status is ExecutionStatus.DRY_RUN:
        return (
            f"Dry run completed for candidate {candidate.candidate_id} "
            f"targeting {candidate.target_id}; no action was applied."
        )

    if feedback is None:
        return (
            f"Candidate {candidate.candidate_id} finished with status "
            f"{status.value}."
        )

    if feedback.actual_eta_gain is None:
        return (
            f"Candidate {candidate.candidate_id} was applied; "
            "eta feedback was not measured."
        )

    return (
        f"Candidate {candidate.candidate_id} finished with status "
        f"{status.value}; expected eta gain "
        f"{feedback.expected_eta_gain:.6g}, actual eta gain "
        f"{feedback.actual_eta_gain:.6g}, feedback "
        f"{feedback.state.value}."
    )


class CascadeExecutor:
    """Execute one selected intervention through a caller-supplied adapter."""

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if config is None:
            config = ExecutorConfig()

        if not isinstance(config, ExecutorConfig):
            raise ExecutorError("config must be an ExecutorConfig.")

        if clock is None:
            clock = time.monotonic
        if not callable(clock):
            raise ExecutorError("clock must be callable.")

        if id_factory is None:
            id_factory = lambda: uuid.uuid4().hex
        if not callable(id_factory):
            raise ExecutorError("id_factory must be callable.")

        self._config = config
        self._clock = clock
        self._id_factory = id_factory

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    def execute(
        self,
        plan: InterventionPlan,
        action: InterventionAction | None,
        *,
        operator_confirmed: bool = False,
        eta_before: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        if not isinstance(plan, InterventionPlan):
            raise ExecutorError("plan must be an InterventionPlan.")

        if action is not None and not callable(action):
            raise ExecutorError("action must be callable or None.")

        eta_before = _optional_float(
            eta_before,
            name="eta_before",
        )
        frozen_metadata = _freeze_metadata(metadata)

        started_at = _finite_float(
            self._clock(),
            name="clock started_at",
        )
        execution_id = (
            _normalized_text(
                self._id_factory(),
                name="generated execution_id",
            )
            if self.config.generate_execution_id
            else "execution"
        )
        steps: list[ExecutionStep] = []

        def add_step(
            name: str,
            succeeded: bool,
            message: str,
            step_metadata: Mapping[str, Any] | None = None,
        ) -> None:
            steps.append(
                ExecutionStep(
                    sequence=len(steps) + 1,
                    name=name,
                    succeeded=succeeded,
                    message=message,
                    timestamp=_finite_float(
                        self._clock(),
                        name="step timestamp",
                    ),
                    metadata=step_metadata,
                )
            )

        ranked = _select_ranked_intervention(plan, self.config)
        add_step(
            "select_intervention",
            ranked is not None,
            (
                "Selected planned intervention."
                if ranked is not None
                else "No executable intervention was selected."
            ),
            (
                None
                if ranked is None
                else {
                    "candidate_id": ranked.candidate.candidate_id,
                    "target_id": ranked.candidate.target_id,
                    "decision": ranked.decision.value,
                }
            ),
        )

        reasons = _gate_reasons(
            plan,
            ranked,
            self.config,
            operator_confirmed=operator_confirmed,
        )
        add_step(
            "execution_gate",
            not reasons,
            (
                "Execution gate passed."
                if not reasons
                else "Execution gate blocked the intervention."
            ),
            {"reasons": tuple(item.value for item in reasons)},
        )

        if reasons:
            finished_at = _finite_float(
                self._clock(),
                name="clock finished_at",
            )
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.BLOCKED,
                candidate=(
                    None if ranked is None else ranked.candidate
                ),
                feedback=None,
                outcome=None,
                gate_reasons=reasons,
                steps=tuple(steps),
                started_at=started_at,
                finished_at=finished_at,
                duration=finished_at - started_at,
                summary=_summary(
                    ExecutionStatus.BLOCKED,
                    None if ranked is None else ranked.candidate,
                    None,
                    reasons,
                ),
                metadata=frozen_metadata,
            )

        if ranked is None:
            finished_at = _finite_float(
                self._clock(),
                name="clock finished_at",
            )
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.SKIPPED,
                candidate=None,
                feedback=None,
                outcome=None,
                gate_reasons=(),
                steps=tuple(steps),
                started_at=started_at,
                finished_at=finished_at,
                duration=finished_at - started_at,
                summary=_summary(
                    ExecutionStatus.SKIPPED,
                    None,
                    None,
                ),
                metadata=frozen_metadata,
            )

        candidate = ranked.candidate

        if self.config.mode is ExecutionMode.DRY_RUN:
            add_step(
                "dry_run",
                True,
                "Dry run completed; the action adapter was not called.",
                {
                    "candidate_id": candidate.candidate_id,
                    "expected_eta_gain": candidate.expected_eta_gain,
                },
            )
            finished_at = _finite_float(
                self._clock(),
                name="clock finished_at",
            )
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.DRY_RUN,
                candidate=candidate,
                feedback=None,
                outcome=None,
                gate_reasons=(),
                steps=tuple(steps),
                started_at=started_at,
                finished_at=finished_at,
                duration=finished_at - started_at,
                summary=_summary(
                    ExecutionStatus.DRY_RUN,
                    candidate,
                    None,
                ),
                metadata=frozen_metadata,
            )

        if action is None:
            raise ExecutorError(
                "action is required when execution mode is APPLY."
            )

        try:
            raw_outcome = action(candidate)
            add_step(
                "apply_intervention",
                True,
                "Action adapter returned successfully.",
                {"candidate_id": candidate.candidate_id},
            )

            outcome = _coerce_outcome(
                raw_outcome,
                eta_before=eta_before,
            )
            add_step(
                "normalize_outcome",
                True,
                "Action outcome was normalized.",
                {
                    "applied": outcome.applied,
                    "measured": outcome.actual_eta_gain is not None,
                },
            )

            feedback = _feedback(candidate, outcome, self.config)
            status = _status_from_outcome(
                outcome,
                feedback,
                self.config,
            )
            add_step(
                "evaluate_feedback",
                status in (
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.PARTIAL,
                ),
                (
                    f"Feedback evaluated as {feedback.state.value}; "
                    f"execution status is {status.value}."
                ),
                feedback.as_dict(),
            )

            if not outcome.applied and status is ExecutionStatus.FAILED:
                error_type = "ActionNotApplied"
                error_message = (
                    outcome.message
                    or "Action adapter reported that the intervention was not applied."
                )
            elif (
                feedback.state is FeedbackState.HARMFUL
                and status is ExecutionStatus.FAILED
            ):
                error_type = "HarmfulEffect"
                error_message = (
                    "Measured eta change was harmful relative to baseline."
                )
            elif status is ExecutionStatus.FAILED:
                error_type = "InsufficientEffect"
                error_message = (
                    "Measured effect did not reach the configured partial-effect threshold."
                )
            else:
                error_type = None
                error_message = None

            finished_at = _finite_float(
                self._clock(),
                name="clock finished_at",
            )
            return ExecutionResult(
                execution_id=execution_id,
                status=status,
                candidate=candidate,
                feedback=feedback,
                outcome=outcome,
                gate_reasons=(),
                steps=tuple(steps),
                started_at=started_at,
                finished_at=finished_at,
                duration=finished_at - started_at,
                summary=_summary(
                    status,
                    candidate,
                    feedback,
                ),
                error_type=error_type,
                error_message=error_message,
                metadata=frozen_metadata,
            )

        except Exception as exc:
            if not self.config.capture_exceptions:
                raise

            add_step(
                "execution_error",
                False,
                f"Execution failed: {type(exc).__name__}: {exc}",
                {"error_type": type(exc).__name__},
            )
            finished_at = _finite_float(
                self._clock(),
                name="clock finished_at",
            )
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                candidate=candidate,
                feedback=None,
                outcome=None,
                gate_reasons=(),
                steps=tuple(steps),
                started_at=started_at,
                finished_at=finished_at,
                duration=finished_at - started_at,
                summary=(
                    f"Candidate {candidate.candidate_id} failed during "
                    f"execution: {type(exc).__name__}."
                ),
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
                metadata=frozen_metadata,
            )


def execute_plan(
    plan: InterventionPlan,
    action: InterventionAction | None,
    *,
    config: ExecutorConfig | None = None,
    operator_confirmed: bool = False,
    eta_before: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Convenience wrapper around ``CascadeExecutor.execute``."""

    return CascadeExecutor(config).execute(
        plan,
        action,
        operator_confirmed=operator_confirmed,
        eta_before=eta_before,
        metadata=metadata,
    )


__all__ = [
    "ActionOutcome",
    "CascadeExecutor",
    "ExecutionFeedback",
    "ExecutionGateReason",
    "ExecutionMode",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionStep",
    "ExecutorConfig",
    "ExecutorError",
    "FeedbackState",
    "InterventionAction",
    "execute_plan",
]
