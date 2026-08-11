"""
ROIF Experience Trace
=====================

Purpose
-------
Represent one completed predictive-control episode as an immutable,
auditable record.

The Experience Trace layer records:

    State(t)
        +
    Disturbance
        +
    Stabilization Demand
        +
    Control Reserve
        +
    Candidate Actions
        ↓
    Predictive Decision
        ↓
    Selected Predicted Outcome
        ↓
    Observed State(t+1)
        ↓
    Prediction Error
        ↓
    ExperienceTrace

Architectural invariant
-----------------------

    ExperienceTrace != Learning

This module DOES NOT:
- update controller memory;
- modify future priors;
- alter predictive preload;
- mutate graphs;
- authorize new relations;
- change candidate policies;
- perform reinforcement learning;
- infer biological memory.

It only records a completed episode.

Persistent controller memory belongs to a later layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.predictive_control import (
    CandidateEvaluation,
    ControlCandidate,
    ControlReserve,
    DisturbanceEstimate,
    PredictionError,
    PredictedOutcome,
    PredictiveControlDecision,
    PredictiveControlDecisionKind,
    PredictiveState,
    StabilizationDemand,
    compare_prediction,
)


# =============================================================================
# Errors
# =============================================================================


class ExperienceTraceError(RuntimeError):
    """Base error for Experience Trace failures."""


class InvalidExperienceTraceError(ExperienceTraceError):
    """Raised when an Experience Trace is structurally invalid."""


class IncompleteExperienceEpisodeError(ExperienceTraceError):
    """Raised when a complete predictive-control episode cannot be recorded."""


# =============================================================================
# Enums
# =============================================================================


class ExperienceActionKind(str, Enum):
    ACTION = "action"
    PROBE = "probe"
    HOLD = "hold"
    NO_SAFE_ACTION = "no_safe_action"


class StabilizationOutcome(str, Enum):
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    WORSENED = "worsened"
    UNKNOWN = "unknown"


# =============================================================================
# Helpers
# =============================================================================


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {}
        if value is None
        else dict(value)
    )


def _finite_float(
    value: float,
    *,
    name: str,
) -> float:
    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


def _finite_non_negative(
    value: float,
    *,
    name: str,
) -> float:
    value = _finite_float(
        value,
        name=name,
    )

    if value < 0.0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return value


def _decision_to_action_kind(
    decision_kind: PredictiveControlDecisionKind,
) -> ExperienceActionKind:
    mapping = {
        PredictiveControlDecisionKind.ACTION:
            ExperienceActionKind.ACTION,
        PredictiveControlDecisionKind.PROBE:
            ExperienceActionKind.PROBE,
        PredictiveControlDecisionKind.HOLD:
            ExperienceActionKind.HOLD,
        PredictiveControlDecisionKind.NO_SAFE_ACTION:
            ExperienceActionKind.NO_SAFE_ACTION,
    }

    return mapping[
        PredictiveControlDecisionKind(
            decision_kind
        )
    ]


# =============================================================================
# Episode identity
# =============================================================================


@dataclass(frozen=True, slots=True)
class ExperienceTraceIdentity:
    """
    Stable identity for one completed predictive-control episode.
    """

    trace_id: str
    episode_index: int
    sequence_id: str | None = None
    parent_trace_id: str | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        if not self.trace_id:
            raise InvalidExperienceTraceError(
                "trace_id must not be empty"
            )

        if self.episode_index < 0:
            raise InvalidExperienceTraceError(
                "episode_index must be non-negative"
            )

        if (
            self.timestamp_start is not None
            and self.timestamp_end is not None
            and self.timestamp_end
            < self.timestamp_start
        ):
            raise InvalidExperienceTraceError(
                "timestamp_end must not precede timestamp_start"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# Outcome summary
# =============================================================================


@dataclass(frozen=True, slots=True)
class StabilizationSummary:
    """
    Auditable before/after stabilization summary.

    This is descriptive only.
    It does not reward, reinforce, or learn from the episode.
    """

    residual_before: float
    residual_predicted: float | None
    residual_observed: float
    residual_change: float
    outcome: StabilizationOutcome

    demand_total: float
    reserve_available_before: float
    selected_action_cost: float
    reserve_after_selected_action: float | None
    stabilization_margin: float | None

    uncertainty_before: float
    uncertainty_predicted: float | None
    uncertainty_observed: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "residual_before",
            _finite_non_negative(
                self.residual_before,
                name="residual_before",
            ),
        )

        if self.residual_predicted is not None:
            object.__setattr__(
                self,
                "residual_predicted",
                _finite_non_negative(
                    self.residual_predicted,
                    name="residual_predicted",
                ),
            )

        object.__setattr__(
            self,
            "residual_observed",
            _finite_non_negative(
                self.residual_observed,
                name="residual_observed",
            ),
        )

        object.__setattr__(
            self,
            "residual_change",
            _finite_float(
                self.residual_change,
                name="residual_change",
            ),
        )

        object.__setattr__(
            self,
            "outcome",
            StabilizationOutcome(
                self.outcome
            ),
        )

        object.__setattr__(
            self,
            "demand_total",
            _finite_non_negative(
                self.demand_total,
                name="demand_total",
            ),
        )

        object.__setattr__(
            self,
            "reserve_available_before",
            _finite_non_negative(
                self.reserve_available_before,
                name="reserve_available_before",
            ),
        )

        object.__setattr__(
            self,
            "selected_action_cost",
            _finite_non_negative(
                self.selected_action_cost,
                name="selected_action_cost",
            ),
        )

        if self.reserve_after_selected_action is not None:
            object.__setattr__(
                self,
                "reserve_after_selected_action",
                _finite_float(
                    self.reserve_after_selected_action,
                    name="reserve_after_selected_action",
                ),
            )

        if self.stabilization_margin is not None:
            object.__setattr__(
                self,
                "stabilization_margin",
                _finite_float(
                    self.stabilization_margin,
                    name="stabilization_margin",
                ),
            )

        object.__setattr__(
            self,
            "uncertainty_before",
            max(
                0.0,
                min(
                    1.0,
                    float(
                        self.uncertainty_before
                    ),
                ),
            ),
        )

        if self.uncertainty_predicted is not None:
            object.__setattr__(
                self,
                "uncertainty_predicted",
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self.uncertainty_predicted
                        ),
                    ),
                ),
            )

        object.__setattr__(
            self,
            "uncertainty_observed",
            max(
                0.0,
                min(
                    1.0,
                    float(
                        self.uncertainty_observed
                    ),
                ),
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# Main trace
# =============================================================================


@dataclass(frozen=True, slots=True)
class ExperienceTrace:
    """
    Immutable record of one completed predictive-control episode.

    Important:
    storing this object must not modify controller state.
    """

    identity: ExperienceTraceIdentity

    initial_state: PredictiveState
    disturbance: DisturbanceEstimate
    demand: StabilizationDemand
    reserve: ControlReserve

    candidates: tuple[
        ControlCandidate,
        ...,
    ]

    evaluations: tuple[
        CandidateEvaluation,
        ...,
    ]

    decision: PredictiveControlDecision

    selected_candidate: ControlCandidate | None
    selected_prediction: PredictedOutcome | None

    observed_state: PredictiveState
    prediction_error: PredictionError | None

    action_kind: ExperienceActionKind
    stabilization: StabilizationSummary

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "candidates",
            tuple(
                self.candidates
            ),
        )

        object.__setattr__(
            self,
            "evaluations",
            tuple(
                self.evaluations
            ),
        )

        expected_kind = _decision_to_action_kind(
            self.decision.kind
        )

        if (
            self.action_kind
            is not expected_kind
        ):
            raise InvalidExperienceTraceError(
                "action_kind does not match decision.kind"
            )

        if (
            self.decision.selected_candidate_id
            is None
        ):
            if self.selected_candidate is not None:
                raise InvalidExperienceTraceError(
                    "selected_candidate must be None when decision has no selection"
                )

            if self.selected_prediction is not None:
                raise InvalidExperienceTraceError(
                    "selected_prediction must be None when decision has no selection"
                )

            if self.prediction_error is not None:
                raise InvalidExperienceTraceError(
                    "prediction_error must be None when no candidate was selected"
                )

        else:
            if self.selected_candidate is None:
                raise InvalidExperienceTraceError(
                    "selected_candidate is required for selected decisions"
                )

            if (
                self.selected_candidate.candidate_id
                != self.decision.selected_candidate_id
            ):
                raise InvalidExperienceTraceError(
                    "selected_candidate does not match decision"
                )

            if self.selected_prediction is None:
                raise InvalidExperienceTraceError(
                    "selected_prediction is required for selected decisions"
                )

            if (
                self.selected_prediction.candidate_id
                != self.selected_candidate.candidate_id
            ):
                raise InvalidExperienceTraceError(
                    "selected_prediction does not match selected_candidate"
                )

            if self.prediction_error is None:
                raise InvalidExperienceTraceError(
                    "prediction_error is required when a candidate was selected"
                )

            if (
                self.prediction_error.candidate_id
                != self.selected_candidate.candidate_id
            ):
                raise InvalidExperienceTraceError(
                    "prediction_error does not match selected_candidate"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# Trace construction helpers
# =============================================================================


def find_selected_evaluation(
    decision: PredictiveControlDecision,
    evaluations: Sequence[
        CandidateEvaluation
    ],
) -> CandidateEvaluation | None:
    """
    Resolve the selected CandidateEvaluation from a decision and evaluation set.
    """

    if (
        decision.selected_candidate_id
        is None
    ):
        return None

    if (
        decision.selected_evaluation
        is not None
    ):
        if (
            decision.selected_evaluation.candidate.candidate_id
            != decision.selected_candidate_id
        ):
            raise IncompleteExperienceEpisodeError(
                "decision.selected_evaluation does not match selected_candidate_id"
            )

        return decision.selected_evaluation

    matches = tuple(
        item
        for item
        in evaluations
        if (
            item.candidate.candidate_id
            == decision.selected_candidate_id
        )
    )

    if len(
        matches
    ) != 1:
        raise IncompleteExperienceEpisodeError(
            "selected candidate must resolve to exactly one evaluation"
        )

    return matches[
        0
    ]


def classify_stabilization_outcome(
    initial_state: PredictiveState,
    observed_state: PredictiveState,
    *,
    tolerance: float = 1e-12,
) -> StabilizationOutcome:
    """
    Classify observed residual change.

    lower observed residual -> IMPROVED
    equal within tolerance -> UNCHANGED
    higher residual -> WORSENED
    """

    before = initial_state.residual_norm()
    after = observed_state.residual_norm()

    difference = (
        after
        - before
    )

    if abs(
        difference
    ) <= abs(
        float(
            tolerance
        )
    ):
        return StabilizationOutcome.UNCHANGED

    if after < before:
        return StabilizationOutcome.IMPROVED

    return StabilizationOutcome.WORSENED


def build_stabilization_summary(
    *,
    initial_state: PredictiveState,
    observed_state: PredictiveState,
    demand: StabilizationDemand,
    reserve: ControlReserve,
    selected_evaluation: CandidateEvaluation | None,
    metadata: Mapping[str, Any] | None = None,
) -> StabilizationSummary:
    """
    Build descriptive before/predicted/observed stabilization audit.
    """

    residual_before = (
        initial_state.residual_norm()
    )

    residual_observed = (
        observed_state.residual_norm()
    )

    residual_predicted = (
        None
        if selected_evaluation is None
        else selected_evaluation.outcome.predicted_residual_error
    )

    selected_action_cost = (
        0.0
        if selected_evaluation is None
        else selected_evaluation.candidate.estimated_cost
    )

    reserve_after = (
        None
        if selected_evaluation is None
        else selected_evaluation.reserve_after_action
    )

    stabilization_margin = (
        None
        if selected_evaluation is None
        else selected_evaluation.stabilization_margin
    )

    uncertainty_predicted = (
        None
        if selected_evaluation is None
        else selected_evaluation.outcome.predicted_uncertainty
    )

    return StabilizationSummary(
        residual_before=residual_before,
        residual_predicted=residual_predicted,
        residual_observed=residual_observed,
        residual_change=(
            residual_observed
            - residual_before
        ),
        outcome=classify_stabilization_outcome(
            initial_state,
            observed_state,
        ),
        demand_total=demand.total,
        reserve_available_before=reserve.available,
        selected_action_cost=selected_action_cost,
        reserve_after_selected_action=reserve_after,
        stabilization_margin=stabilization_margin,
        uncertainty_before=initial_state.uncertainty,
        uncertainty_predicted=uncertainty_predicted,
        uncertainty_observed=observed_state.uncertainty,
        metadata=metadata,
    )


def build_experience_trace(
    *,
    identity: ExperienceTraceIdentity,
    initial_state: PredictiveState,
    disturbance: DisturbanceEstimate,
    demand: StabilizationDemand,
    reserve: ControlReserve,
    candidates: Sequence[
        ControlCandidate
    ],
    evaluations: Sequence[
        CandidateEvaluation
    ],
    decision: PredictiveControlDecision,
    observed_state: PredictiveState,
    metadata: Mapping[str, Any] | None = None,
) -> ExperienceTrace:
    """
    Build one immutable completed episode.

    If the decision selected an action/probe/hold candidate, prediction error is
    computed against its predicted outcome.

    If decision == NO_SAFE_ACTION with no selected candidate, the trace records
    the episode without a prediction error.
    """

    candidates_tuple = tuple(
        candidates
    )

    evaluations_tuple = tuple(
        evaluations
    )

    selected_evaluation = (
        find_selected_evaluation(
            decision,
            evaluations_tuple,
        )
    )

    if selected_evaluation is None:
        selected_candidate = None
        selected_prediction = None
        prediction_error = None

    else:
        selected_candidate = (
            selected_evaluation.candidate
        )

        selected_prediction = (
            selected_evaluation.outcome
        )

        prediction_error = compare_prediction(
            selected_prediction,
            observed_state,
            metadata={
                "trace_id": identity.trace_id,
                "experience_trace_generated": True,
                "learning_applied": False,
                "controller_memory_mutated": False,
                "external_expected_label_used": False,
            },
        )

    stabilization = build_stabilization_summary(
        initial_state=initial_state,
        observed_state=observed_state,
        demand=demand,
        reserve=reserve,
        selected_evaluation=selected_evaluation,
        metadata={
            "trace_id": identity.trace_id,
            "learning_applied": False,
            "controller_memory_mutated": False,
        },
    )

    action_kind = _decision_to_action_kind(
        decision.kind
    )

    return ExperienceTrace(
        identity=identity,
        initial_state=initial_state,
        disturbance=disturbance,
        demand=demand,
        reserve=reserve,
        candidates=candidates_tuple,
        evaluations=evaluations_tuple,
        decision=decision,
        selected_candidate=selected_candidate,
        selected_prediction=selected_prediction,
        observed_state=observed_state,
        prediction_error=prediction_error,
        action_kind=action_kind,
        stabilization=stabilization,
        metadata={
            **(
                {}
                if metadata is None
                else dict(
                    metadata
                )
            ),
            "experience_trace_generated": True,
            "learning_applied": False,
            "controller_memory_mutated": False,
            "predictive_preload_modified": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Trace inspection helpers
# =============================================================================


def selected_candidate_id(
    trace: ExperienceTrace,
) -> str | None:
    if trace.selected_candidate is None:
        return None

    return trace.selected_candidate.candidate_id


def prediction_error_norm(
    trace: ExperienceTrace,
) -> float | None:
    if trace.prediction_error is None:
        return None

    return trace.prediction_error.l2_error


def improved_stabilization(
    trace: ExperienceTrace,
) -> bool:
    return (
        trace.stabilization.outcome
        is StabilizationOutcome.IMPROVED
    )


def trace_is_learning_free(
    trace: ExperienceTrace,
) -> bool:
    """
    Explicit audit helper for the core architectural invariant.

    ExperienceTrace must remain a passive record.
    """

    return (
        trace.metadata.get(
            "learning_applied"
        )
        is False
        and trace.metadata.get(
            "controller_memory_mutated"
        )
        is False
        and trace.metadata.get(
            "predictive_preload_modified"
        )
        is False
        and trace.metadata.get(
            "graph_mutated"
        )
        is False
    )


# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "ExperienceActionKind",
    "ExperienceTrace",
    "ExperienceTraceError",
    "ExperienceTraceIdentity",
    "IncompleteExperienceEpisodeError",
    "InvalidExperienceTraceError",
    "StabilizationOutcome",
    "StabilizationSummary",
    "build_experience_trace",
    "build_stabilization_summary",
    "classify_stabilization_outcome",
    "find_selected_evaluation",
    "improved_stabilization",
    "prediction_error_norm",
    "selected_candidate_id",
    "trace_is_learning_free",
]
