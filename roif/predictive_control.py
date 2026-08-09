"""
ROIF Predictive Control Layer
=============================

Purpose
-------
Domain-agnostic predictive stabilization layer for ROIF.

Core loop:
    current state + disturbance + control reserve
        -> candidate control actions
        -> predicted next state
        -> predicted residual error
        -> ACTION / PROBE / HOLD
        -> observed next state
        -> prediction error

Boundaries:
- no memory / learning;
- no policy remodeling across episodes;
- no hidden evaluator labels;
- does not replace Active Probe or Active Cascade;
- does not assume every control relation is spatial/mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


class PredictiveControlError(RuntimeError):
    """Base predictive-control error."""


class InvalidPredictiveStateError(PredictiveControlError):
    """Invalid predictive state."""


class InvalidControlCandidateError(PredictiveControlError):
    """Invalid control candidate."""


class NoViableControlCandidateError(PredictiveControlError):
    """No viable control candidate."""


class PredictiveControlMode(str, Enum):
    STABILIZE = "stabilize"
    PROBE_IF_UNCERTAIN = "probe_if_uncertain"
    OBSERVE_ONLY = "observe_only"


class PredictiveControlDecisionKind(str, Enum):
    ACTION = "action"
    PROBE = "probe"
    HOLD = "hold"
    NO_SAFE_ACTION = "no_safe_action"


class ControlCandidateKind(str, Enum):
    CORRECTIVE = "corrective"
    PROBE = "probe"
    HOLD = "hold"


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _finite_non_negative(value: float, *, name: str) -> float:
    value = float(value)
    if value != value:
        raise ValueError(f"{name} must not be NaN")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _euclidean_norm(values: Iterable[float]) -> float:
    return sqrt(sum(float(value) ** 2 for value in values))


@dataclass(frozen=True, slots=True)
class PredictiveState:
    values: Mapping[str, float]
    target_values: Mapping[str, float]
    timestamp: float | None = None
    uncertainty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.values:
            raise InvalidPredictiveStateError(
                "PredictiveState.values must not be empty."
            )

        values = {str(k): float(v) for k, v in self.values.items()}
        targets = {str(k): float(v) for k, v in self.target_values.items()}

        missing_targets = set(targets) - set(values)
        if missing_targets:
            raise InvalidPredictiveStateError(
                "target_values contain unknown state variables: "
                f"{sorted(missing_targets)}"
            )

        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "target_values", MappingProxyType(targets))
        object.__setattr__(self, "uncertainty", _clamp01(self.uncertainty))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    def residuals(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                key: self.values[key] - target
                for key, target in self.target_values.items()
            }
        )

    def residual_norm(self) -> float:
        return _euclidean_norm(self.residuals().values())


@dataclass(frozen=True, slots=True)
class DisturbanceEstimate:
    components: Mapping[str, float]
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            MappingProxyType({str(k): float(v) for k, v in self.components.items()}),
        )
        object.__setattr__(self, "confidence", _clamp01(self.confidence))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    @property
    def magnitude(self) -> float:
        return _euclidean_norm(self.components.values())


@dataclass(frozen=True, slots=True)
class ControlReserve:
    capacity: float
    committed: float
    recoverable_fraction: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        capacity = _finite_non_negative(self.capacity, name="capacity")
        committed = _finite_non_negative(self.committed, name="committed")
        if committed > capacity:
            raise ValueError("committed control resource cannot exceed capacity")

        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "committed", committed)
        object.__setattr__(
            self,
            "recoverable_fraction",
            _clamp01(self.recoverable_fraction),
        )
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    @property
    def available(self) -> float:
        recoverable = self.committed * self.recoverable_fraction
        return max(0.0, self.capacity - self.committed + recoverable)

    @property
    def utilization(self) -> float:
        if self.capacity <= 0.0:
            return 1.0
        return _clamp01(self.committed / self.capacity)


@dataclass(frozen=True, slots=True)
class StabilizationDemand:
    state_error: float
    disturbance_load: float
    uncertainty_load: float = 0.0
    urgency: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_error",
            _finite_non_negative(self.state_error, name="state_error"),
        )
        object.__setattr__(
            self,
            "disturbance_load",
            _finite_non_negative(self.disturbance_load, name="disturbance_load"),
        )
        object.__setattr__(
            self,
            "uncertainty_load",
            _finite_non_negative(self.uncertainty_load, name="uncertainty_load"),
        )
        object.__setattr__(
            self,
            "urgency",
            _finite_non_negative(self.urgency, name="urgency"),
        )
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    @property
    def total(self) -> float:
        return (
            self.state_error
            + self.disturbance_load
            + self.uncertainty_load
        ) * self.urgency

    def reserve_ratio(self, reserve: ControlReserve) -> float:
        if self.total <= 0.0:
            return float("inf")
        return reserve.available / self.total


@dataclass(frozen=True, slots=True)
class ControlCandidate:
    candidate_id: str
    kind: ControlCandidateKind
    control_delta: Mapping[str, float]
    estimated_cost: float
    reversibility: float = 1.0
    safety_risk: float = 0.0
    uncertainty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise InvalidControlCandidateError("candidate_id must not be empty")

        object.__setattr__(self, "kind", ControlCandidateKind(self.kind))
        object.__setattr__(
            self,
            "control_delta",
            MappingProxyType({str(k): float(v) for k, v in self.control_delta.items()}),
        )
        object.__setattr__(
            self,
            "estimated_cost",
            _finite_non_negative(self.estimated_cost, name="estimated_cost"),
        )
        object.__setattr__(self, "reversibility", _clamp01(self.reversibility))
        object.__setattr__(self, "safety_risk", _clamp01(self.safety_risk))
        object.__setattr__(self, "uncertainty", _clamp01(self.uncertainty))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PredictedOutcome:
    candidate_id: str
    predicted_state: PredictiveState
    predicted_residual_error: float
    predicted_control_cost: float
    predicted_uncertainty: float
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_residual_error",
            _finite_non_negative(
                self.predicted_residual_error,
                name="predicted_residual_error",
            ),
        )
        object.__setattr__(
            self,
            "predicted_control_cost",
            _finite_non_negative(
                self.predicted_control_cost,
                name="predicted_control_cost",
            ),
        )
        object.__setattr__(
            self,
            "predicted_uncertainty",
            _clamp01(self.predicted_uncertainty),
        )
        object.__setattr__(self, "confidence", _clamp01(self.confidence))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: ControlCandidate
    outcome: PredictedOutcome
    reserve_after_action: float
    stabilization_margin: float
    score: float
    safe: bool
    viable: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reserve_after_action", float(self.reserve_after_action))
        object.__setattr__(self, "stabilization_margin", float(self.stabilization_margin))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PredictionError:
    candidate_id: str
    variable_errors: Mapping[str, float]
    l2_error: float
    normalized_error: float
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variable_errors",
            MappingProxyType(
                {str(k): float(v) for k, v in self.variable_errors.items()}
            ),
        )
        object.__setattr__(
            self,
            "l2_error",
            _finite_non_negative(self.l2_error, name="l2_error"),
        )
        object.__setattr__(
            self,
            "normalized_error",
            _finite_non_negative(
                self.normalized_error,
                name="normalized_error",
            ),
        )
        object.__setattr__(self, "confidence", _clamp01(self.confidence))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PredictiveControlConfig:
    mode: PredictiveControlMode = PredictiveControlMode.PROBE_IF_UNCERTAIN
    uncertainty_probe_threshold: float = 0.35
    max_safety_risk: float = 0.30
    min_reversibility_for_probe: float = 0.70
    residual_weight: float = 1.0
    cost_weight: float = 0.25
    uncertainty_weight: float = 0.50
    reserve_weight: float = 0.25
    safety_weight: float = 1.0
    min_stabilization_margin: float = 0.0
    require_positive_reserve: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PredictiveControlMode(self.mode))

        for name in (
            "uncertainty_probe_threshold",
            "max_safety_risk",
            "min_reversibility_for_probe",
        ):
            object.__setattr__(self, name, _clamp01(getattr(self, name)))

        for name in (
            "residual_weight",
            "cost_weight",
            "uncertainty_weight",
            "reserve_weight",
            "safety_weight",
        ):
            object.__setattr__(
                self,
                name,
                _finite_non_negative(getattr(self, name), name=name),
            )

        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PredictiveControlDecision:
    kind: PredictiveControlDecisionKind
    selected_candidate_id: str | None
    selected_evaluation: CandidateEvaluation | None
    ranked_evaluations: tuple[CandidateEvaluation, ...]
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PredictiveControlDecisionKind(self.kind))
        object.__setattr__(
            self,
            "ranked_evaluations",
            tuple(self.ranked_evaluations),
        )
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


PredictionModel = Callable[
    [PredictiveState, DisturbanceEstimate, ControlCandidate],
    PredictiveState,
]


def estimate_stabilization_demand(
    state: PredictiveState,
    disturbance: DisturbanceEstimate,
    *,
    uncertainty_weight: float = 1.0,
    urgency: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> StabilizationDemand:
    return StabilizationDemand(
        state_error=state.residual_norm(),
        disturbance_load=disturbance.magnitude * disturbance.confidence,
        uncertainty_load=state.uncertainty * max(0.0, float(uncertainty_weight)),
        urgency=max(0.0, float(urgency)),
        metadata=metadata,
    )


def predict_outcome(
    state: PredictiveState,
    disturbance: DisturbanceEstimate,
    candidate: ControlCandidate,
    *,
    model: PredictionModel,
    metadata: Mapping[str, Any] | None = None,
) -> PredictedOutcome:
    predicted_state = model(state, disturbance, candidate)

    if not isinstance(predicted_state, PredictiveState):
        raise PredictiveControlError(
            "prediction model must return PredictiveState"
        )

    return PredictedOutcome(
        candidate_id=candidate.candidate_id,
        predicted_state=predicted_state,
        predicted_residual_error=predicted_state.residual_norm(),
        predicted_control_cost=candidate.estimated_cost,
        predicted_uncertainty=predicted_state.uncertainty,
        confidence=1.0 - predicted_state.uncertainty,
        metadata=metadata,
    )


def evaluate_candidate(
    candidate: ControlCandidate,
    outcome: PredictedOutcome,
    reserve: ControlReserve,
    demand: StabilizationDemand,
    *,
    config: PredictiveControlConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateEvaluation:
    config = config or PredictiveControlConfig()

    reserve_after = reserve.available - candidate.estimated_cost
    stabilization_margin = reserve_after - demand.total

    safe = candidate.safety_risk <= config.max_safety_risk

    reserve_ok = (
        reserve_after > 0.0
        if config.require_positive_reserve
        else reserve_after >= 0.0
    )

    margin_ok = stabilization_margin >= config.min_stabilization_margin

    viable = safe and reserve_ok and margin_ok

    score = (
        -config.residual_weight * outcome.predicted_residual_error
        -config.cost_weight * outcome.predicted_control_cost
        -config.uncertainty_weight * outcome.predicted_uncertainty
        +config.reserve_weight * reserve_after
        -config.safety_weight * candidate.safety_risk
    )

    return CandidateEvaluation(
        candidate=candidate,
        outcome=outcome,
        reserve_after_action=reserve_after,
        stabilization_margin=stabilization_margin,
        score=score,
        safe=safe,
        viable=viable,
        metadata=metadata,
    )


def rank_candidate_evaluations(
    evaluations: Sequence[CandidateEvaluation],
) -> tuple[CandidateEvaluation, ...]:
    return tuple(
        sorted(
            evaluations,
            key=lambda item: (
                0 if item.viable else 1,
                -item.score,
                item.outcome.predicted_residual_error,
                item.outcome.predicted_uncertainty,
                item.candidate.candidate_id,
            ),
        )
    )


def _best_probe_candidate(
    ranked: Sequence[CandidateEvaluation],
    *,
    config: PredictiveControlConfig,
) -> CandidateEvaluation | None:
    for evaluation in ranked:
        candidate = evaluation.candidate
        if (
            candidate.kind is ControlCandidateKind.PROBE
            and candidate.reversibility >= config.min_reversibility_for_probe
            and candidate.safety_risk <= config.max_safety_risk
        ):
            return evaluation
    return None


def select_control_action(
    evaluations: Sequence[CandidateEvaluation],
    *,
    config: PredictiveControlConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PredictiveControlDecision:
    config = config or PredictiveControlConfig()
    ranked = rank_candidate_evaluations(evaluations)

    if not ranked:
        return PredictiveControlDecision(
            kind=PredictiveControlDecisionKind.HOLD,
            selected_candidate_id=None,
            selected_evaluation=None,
            ranked_evaluations=(),
            reason="no_candidates",
            metadata=metadata,
        )

    if config.mode is PredictiveControlMode.OBSERVE_ONLY:
        return PredictiveControlDecision(
            kind=PredictiveControlDecisionKind.HOLD,
            selected_candidate_id=None,
            selected_evaluation=None,
            ranked_evaluations=ranked,
            reason="observe_only_mode",
            metadata=metadata,
        )

    viable = tuple(item for item in ranked if item.viable)

    if not viable:
        probe = _best_probe_candidate(ranked, config=config)

        if (
            config.mode is PredictiveControlMode.PROBE_IF_UNCERTAIN
            and probe is not None
        ):
            return PredictiveControlDecision(
                kind=PredictiveControlDecisionKind.PROBE,
                selected_candidate_id=probe.candidate.candidate_id,
                selected_evaluation=probe,
                ranked_evaluations=ranked,
                reason="no_viable_corrective_action_probe_available",
                metadata=metadata,
            )

        return PredictiveControlDecision(
            kind=PredictiveControlDecisionKind.NO_SAFE_ACTION,
            selected_candidate_id=None,
            selected_evaluation=None,
            ranked_evaluations=ranked,
            reason="no_viable_candidate",
            metadata=metadata,
        )

    aggregate_uncertainty = max(
        item.outcome.predicted_uncertainty
        for item in viable
    )

    if (
        config.mode is PredictiveControlMode.PROBE_IF_UNCERTAIN
        and aggregate_uncertainty >= config.uncertainty_probe_threshold
    ):
        probe = _best_probe_candidate(ranked, config=config)
        if probe is not None:
            return PredictiveControlDecision(
                kind=PredictiveControlDecisionKind.PROBE,
                selected_candidate_id=probe.candidate.candidate_id,
                selected_evaluation=probe,
                ranked_evaluations=ranked,
                reason="uncertainty_requires_probe",
                metadata=metadata,
            )

    best = viable[0]

    if best.candidate.kind is ControlCandidateKind.HOLD:
        kind = PredictiveControlDecisionKind.HOLD
    elif best.candidate.kind is ControlCandidateKind.PROBE:
        kind = PredictiveControlDecisionKind.PROBE
    else:
        kind = PredictiveControlDecisionKind.ACTION

    return PredictiveControlDecision(
        kind=kind,
        selected_candidate_id=best.candidate.candidate_id,
        selected_evaluation=best,
        ranked_evaluations=ranked,
        reason="best_viable_candidate",
        metadata=metadata,
    )


def compare_prediction(
    outcome: PredictedOutcome,
    observed_state: PredictiveState,
    *,
    normalization_floor: float = 1e-9,
    metadata: Mapping[str, Any] | None = None,
) -> PredictionError:
    shared = set(outcome.predicted_state.values) & set(observed_state.values)

    if not shared:
        raise PredictiveControlError(
            "predicted and observed state share no variables"
        )

    errors = {
        key: observed_state.values[key] - outcome.predicted_state.values[key]
        for key in sorted(shared)
    }

    l2_error = _euclidean_norm(errors.values())

    scale = max(
        normalization_floor,
        outcome.predicted_state.residual_norm(),
        observed_state.residual_norm(),
        1.0,
    )

    normalized_error = l2_error / scale
    confidence = 1.0 - _clamp01(normalized_error)

    return PredictionError(
        candidate_id=outcome.candidate_id,
        variable_errors=errors,
        l2_error=l2_error,
        normalized_error=normalized_error,
        confidence=confidence,
        metadata=metadata,
    )


def evaluate_control_candidates(
    state: PredictiveState,
    disturbance: DisturbanceEstimate,
    candidates: Sequence[ControlCandidate],
    reserve: ControlReserve,
    *,
    model: PredictionModel,
    config: PredictiveControlConfig | None = None,
    demand: StabilizationDemand | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[
    StabilizationDemand,
    tuple[CandidateEvaluation, ...],
]:
    config = config or PredictiveControlConfig()

    demand = demand or estimate_stabilization_demand(
        state,
        disturbance,
    )

    evaluations = []

    for candidate in candidates:
        outcome = predict_outcome(
            state,
            disturbance,
            candidate,
            model=model,
            metadata={
                "candidate_id": candidate.candidate_id,
                "external_expected_label_used": False,
            },
        )

        evaluations.append(
            evaluate_candidate(
                candidate,
                outcome,
                reserve,
                demand,
                config=config,
                metadata={
                    "candidate_id": candidate.candidate_id,
                    "external_expected_label_used": False,
                },
            )
        )

    return demand, rank_candidate_evaluations(evaluations)


__all__ = [
    "CandidateEvaluation",
    "ControlCandidate",
    "ControlCandidateKind",
    "ControlReserve",
    "DisturbanceEstimate",
    "InvalidControlCandidateError",
    "InvalidPredictiveStateError",
    "NoViableControlCandidateError",
    "PredictionError",
    "PredictionModel",
    "PredictedOutcome",
    "PredictiveControlConfig",
    "PredictiveControlDecision",
    "PredictiveControlDecisionKind",
    "PredictiveControlError",
    "PredictiveControlMode",
    "PredictiveState",
    "StabilizationDemand",
    "compare_prediction",
    "estimate_stabilization_demand",
    "evaluate_candidate",
    "evaluate_control_candidates",
    "predict_outcome",
    "rank_candidate_evaluations",
    "select_control_action",
]
