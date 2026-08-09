"""
ROIF External-Domain Validation
Scenario 04A — Sailing Yacht + Crew Predictive Control Bridge

Purpose
-------
Connect the domain-agnostic predictive-control layer to the published-topology
sailing yacht + crew benchmark.

This module does NOT introduce memory or learning yet.

It implements the first closed predictive stabilization loop:

    wind / wave disturbance
        -> yacht state
        -> stabilization demand
        -> crew control reserve
        -> candidate actions
        -> predicted yacht state
        -> ACTION / PROBE / HOLD
        -> prediction error after observation

Architectural boundary
----------------------
- Physical yacht topology stays in sailing_yacht_crew_coupled_case.py
- Predictive-control mathematics stays in roif.predictive_control.py
- This file is only the bridge between those two layers.
- External evaluator labels are not used for candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from roif.predictive_control import (
    CandidateEvaluation,
    ControlCandidate,
    ControlCandidateKind,
    ControlReserve,
    DisturbanceEstimate,
    PredictionError,
    PredictiveControlConfig,
    PredictiveControlDecision,
    PredictiveControlMode,
    PredictiveState,
    StabilizationDemand,
    compare_prediction,
    estimate_stabilization_demand,
    evaluate_control_candidates,
    select_control_action,
)

from validation.sailing.sailing_yacht_crew_coupled_case import (
    COURSE_ERROR_CHANNEL,
    HEEL_YAW_CHANNEL,
    HELM_CONTROL_DEMAND_CHANNEL,
    RUDDER_ACTION_CHANNEL,
    SAIL_LOAD_CHANNEL,
    SAIL_TRIM_ACTION_CHANNEL,
    TRIM_CONTROL_DEMAND_CHANNEL,
    WIND_DISTURBANCE_CHANNEL,
    SailingYachtCrewValidationCase,
    build_sailing_yacht_crew_coupled_case,
)


# =============================================================================
# Candidate identifiers
# =============================================================================


HELM_CORRECTION_ID = "helm_corrective_action"
TRIM_CORRECTION_ID = "sail_trim_corrective_action"
COUPLED_CORRECTION_ID = "coupled_helm_trim_action"
INFORMATION_PROBE_ID = "small_information_probe"
HOLD_ID = "hold_current_control"


# =============================================================================
# Bridge configuration
# =============================================================================


DEFAULT_HELM_AUTHORITY = 0.70
DEFAULT_TRIM_AUTHORITY = 0.55
DEFAULT_COUPLED_AUTHORITY = 0.85

DEFAULT_PROBE_MAGNITUDE = 0.05

DEFAULT_CREW_RESERVE_CAPACITY = 2.0

DEFAULT_WIND_DISTURBANCE_SCALE = 0.35
DEFAULT_WAVE_DISTURBANCE_SCALE = 0.25


class SailingPredictiveControlError(RuntimeError):
    """Raised when the sailing predictive-control bridge is invalid."""


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {}
        if value is None
        else dict(value)
    )


# =============================================================================
# Immutable result
# =============================================================================


@dataclass(frozen=True, slots=True)
class SailingPredictiveControlResult:
    case_id: str

    predictive_state: PredictiveState
    disturbance: DisturbanceEstimate
    reserve: ControlReserve
    demand: StabilizationDemand

    candidates: tuple[
        ControlCandidate,
        ...,
    ]

    evaluations: tuple[
        CandidateEvaluation,
        ...,
    ]

    decision: PredictiveControlDecision

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

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# Case / channel helpers
# =============================================================================


def build_case(
) -> SailingYachtCrewValidationCase:
    return build_sailing_yacht_crew_coupled_case()


def channel_lookup(
    case: SailingYachtCrewValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            channel.channel_id: channel
            for entity
            in case.pathological_system.entities
            for channel
            in entity.channels
        }
    )


def channel_load(
    case: SailingYachtCrewValidationCase,
    channel_id: str,
) -> float:
    return float(
        channel_lookup(
            case
        )[
            channel_id
        ].capacity_state.load
    )


def channel_capacity(
    case: SailingYachtCrewValidationCase,
    channel_id: str,
) -> float:
    return float(
        channel_lookup(
            case
        )[
            channel_id
        ].capacity_state.capacity
    )


# =============================================================================
# Predictive state bridge
# =============================================================================


def build_predictive_state(
    case: SailingYachtCrewValidationCase,
) -> PredictiveState:
    """
    Convert the current stressed yacht + crew condition into controller state.

    The controller can observe:
    - sail load;
    - heel/yaw response;
    - course error;
    - helm demand;
    - trimmer demand;
    - rudder action;
    - sail-trim action.

    The regulated targets are:
    - course error -> 0
    - heel/yaw response -> 0

    Control-demand channels are retained in the state but are not treated as
    regulated outputs.
    """

    values = {
        SAIL_LOAD_CHANNEL: channel_load(
            case,
            SAIL_LOAD_CHANNEL,
        ),
        HEEL_YAW_CHANNEL: channel_load(
            case,
            HEEL_YAW_CHANNEL,
        ),
        COURSE_ERROR_CHANNEL: channel_load(
            case,
            COURSE_ERROR_CHANNEL,
        ),
        HELM_CONTROL_DEMAND_CHANNEL: channel_load(
            case,
            HELM_CONTROL_DEMAND_CHANNEL,
        ),
        TRIM_CONTROL_DEMAND_CHANNEL: channel_load(
            case,
            TRIM_CONTROL_DEMAND_CHANNEL,
        ),
        RUDDER_ACTION_CHANNEL: channel_load(
            case,
            RUDDER_ACTION_CHANNEL,
        ),
        SAIL_TRIM_ACTION_CHANNEL: channel_load(
            case,
            SAIL_TRIM_ACTION_CHANNEL,
        ),
    }

    # Aggregate uncertainty is intentionally benchmark-normalized.
    # It is not read from evaluator labels.
    uncertainty = 0.20

    return PredictiveState(
        values=values,
        target_values={
            COURSE_ERROR_CHANNEL: 0.0,
            HEEL_YAW_CHANNEL: 0.0,
        },
        timestamp=0.0,
        uncertainty=uncertainty,
        metadata={
            "validation_case": "sailing_04A",
            "bridge": "yacht_crew_predictive_control",
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Disturbance bridge
# =============================================================================


def build_disturbance_estimate(
    case: SailingYachtCrewValidationCase,
    *,
    wind_scale: float = DEFAULT_WIND_DISTURBANCE_SCALE,
    wave_component: float = DEFAULT_WAVE_DISTURBANCE_SCALE,
    confidence: float = 0.90,
) -> DisturbanceEstimate:
    """
    Convert the current environmental load to predictive-control disturbance.

    Wind derives from the benchmark case.
    Wave is currently a normalized external component because 04A topology
    does not yet contain a dedicated wave channel.
    """

    wind_load = channel_load(
        case,
        WIND_DISTURBANCE_CHANNEL,
    )

    return DisturbanceEstimate(
        components={
            "wind": (
                wind_load
                * float(
                    wind_scale
                )
            ),
            "wave": float(
                wave_component
            ),
        },
        confidence=confidence,
        metadata={
            "validation_case": "sailing_04A",
            "wave_component_is_benchmark_normalized": True,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Crew reserve
# =============================================================================


def build_crew_control_reserve(
    case: SailingYachtCrewValidationCase,
    *,
    reserve_capacity: float = DEFAULT_CREW_RESERVE_CAPACITY,
) -> ControlReserve:
    """
    Estimate remaining crew stabilization capacity.

    Current committed effort is based on normalized helm + trim demand.
    This is a benchmark-level bridge, not a physiological claim.
    """

    helm_load = channel_load(
        case,
        HELM_CONTROL_DEMAND_CHANNEL,
    )

    helm_capacity = channel_capacity(
        case,
        HELM_CONTROL_DEMAND_CHANNEL,
    )

    trim_load = channel_load(
        case,
        TRIM_CONTROL_DEMAND_CHANNEL,
    )

    trim_capacity = channel_capacity(
        case,
        TRIM_CONTROL_DEMAND_CHANNEL,
    )

    helm_utilization = (
        0.0
        if helm_capacity <= 0.0
        else helm_load / helm_capacity
    )

    trim_utilization = (
        0.0
        if trim_capacity <= 0.0
        else trim_load / trim_capacity
    )

    mean_utilization = max(
        0.0,
        min(
            1.0,
            (
                helm_utilization
                + trim_utilization
            )
            / 2.0,
        ),
    )

    capacity = float(
        reserve_capacity
    )

    committed = (
        capacity
        * mean_utilization
    )

    return ControlReserve(
        capacity=capacity,
        committed=committed,
        recoverable_fraction=0.15,
        metadata={
            "validation_case": "sailing_04A",
            "helm_utilization": helm_utilization,
            "trim_utilization": trim_utilization,
            "mean_utilization": mean_utilization,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Candidate actions
# =============================================================================


def build_control_candidates(
) -> tuple[
    ControlCandidate,
    ...,
]:
    """
    Candidate set intentionally includes:
    - helm-only correction;
    - trim-only correction;
    - coupled correction;
    - reversible Probe;
    - hold.

    No candidate is marked as externally "correct".
    """

    return (
        ControlCandidate(
            candidate_id=HELM_CORRECTION_ID,
            kind=ControlCandidateKind.CORRECTIVE,
            control_delta={
                RUDDER_ACTION_CHANNEL: -0.30,
            },
            estimated_cost=0.22,
            reversibility=0.90,
            safety_risk=0.05,
            uncertainty=0.15,
            metadata={
                "validation_case": "sailing_04A",
                "control_mode": "helm_only",
                "external_expected_label_used": False,
            },
        ),

        ControlCandidate(
            candidate_id=TRIM_CORRECTION_ID,
            kind=ControlCandidateKind.CORRECTIVE,
            control_delta={
                SAIL_TRIM_ACTION_CHANNEL: -0.28,
            },
            estimated_cost=0.20,
            reversibility=0.92,
            safety_risk=0.05,
            uncertainty=0.18,
            metadata={
                "validation_case": "sailing_04A",
                "control_mode": "trim_only",
                "external_expected_label_used": False,
            },
        ),

        ControlCandidate(
            candidate_id=COUPLED_CORRECTION_ID,
            kind=ControlCandidateKind.CORRECTIVE,
            control_delta={
                RUDDER_ACTION_CHANNEL: -0.24,
                SAIL_TRIM_ACTION_CHANNEL: -0.22,
            },
            estimated_cost=0.34,
            reversibility=0.88,
            safety_risk=0.07,
            uncertainty=0.12,
            metadata={
                "validation_case": "sailing_04A",
                "control_mode": "coupled",
                "external_expected_label_used": False,
            },
        ),

        ControlCandidate(
            candidate_id=INFORMATION_PROBE_ID,
            kind=ControlCandidateKind.PROBE,
            control_delta={
                RUDDER_ACTION_CHANNEL: -DEFAULT_PROBE_MAGNITUDE,
                SAIL_TRIM_ACTION_CHANNEL: -DEFAULT_PROBE_MAGNITUDE,
            },
            estimated_cost=0.06,
            reversibility=1.0,
            safety_risk=0.01,
            uncertainty=0.05,
            metadata={
                "validation_case": "sailing_04A",
                "control_mode": "probe",
                "external_expected_label_used": False,
            },
        ),

        ControlCandidate(
            candidate_id=HOLD_ID,
            kind=ControlCandidateKind.HOLD,
            control_delta={},
            estimated_cost=0.0,
            reversibility=1.0,
            safety_risk=0.0,
            uncertainty=0.20,
            metadata={
                "validation_case": "sailing_04A",
                "control_mode": "hold",
                "external_expected_label_used": False,
            },
        ),
    )


# =============================================================================
# Domain prediction model
# =============================================================================


def sailing_prediction_model(
    state: PredictiveState,
    disturbance: DisturbanceEstimate,
    candidate: ControlCandidate,
) -> PredictiveState:
    """
    Minimal normalized yacht-control predictor.

    This predictor is intentionally simple and transparent.

    It captures five ideas:
    1. wind/wave disturbance pushes heel/yaw and course error upward;
    2. helm action reduces course error strongly;
    3. sail-trim action reduces heel/yaw strongly and course error secondarily;
    4. coupled action can reduce both outputs;
    5. Probe reduces uncertainty more than ordinary corrective action.

    This is NOT claimed as a hydrodynamic model.
    It is the first predictive-control bridge used to test the ROIF control
    architecture.
    """

    values = dict(
        state.values
    )

    wind = float(
        disturbance.components.get(
            "wind",
            0.0,
        )
    )

    wave = float(
        disturbance.components.get(
            "wave",
            0.0,
        )
    )

    disturbance_push = (
        wind
        + wave
    )

    current_course = float(
        state.values[
            COURSE_ERROR_CHANNEL
        ]
    )

    current_heel = float(
        state.values[
            HEEL_YAW_CHANNEL
        ]
    )

    helm_delta = float(
        candidate.control_delta.get(
            RUDDER_ACTION_CHANNEL,
            0.0,
        )
    )

    trim_delta = float(
        candidate.control_delta.get(
            SAIL_TRIM_ACTION_CHANNEL,
            0.0,
        )
    )

    # Disturbance propagation.
    predicted_heel = (
        current_heel
        + 0.34
        * disturbance_push
    )

    predicted_course = (
        current_course
        + 0.22
        * disturbance_push
    )

    # Corrective authority.
    predicted_course += (
        DEFAULT_HELM_AUTHORITY
        * helm_delta
    )

    predicted_heel += (
        DEFAULT_TRIM_AUTHORITY
        * trim_delta
    )

    # Sail trim also changes yaw/course tendency.
    predicted_course += (
        0.25
        * trim_delta
    )

    # Small cross-coupling from helm to heel.
    predicted_heel += (
        0.08
        * helm_delta
    )

    if (
        candidate.candidate_id
        == COUPLED_CORRECTION_ID
    ):
        predicted_course -= 0.05 * DEFAULT_COUPLED_AUTHORITY
        predicted_heel -= 0.05 * DEFAULT_COUPLED_AUTHORITY

    values[
        COURSE_ERROR_CHANNEL
    ] = predicted_course

    values[
        HEEL_YAW_CHANNEL
    ] = predicted_heel

    values[
        RUDDER_ACTION_CHANNEL
    ] = (
        state.values[
            RUDDER_ACTION_CHANNEL
        ]
        + helm_delta
    )

    values[
        SAIL_TRIM_ACTION_CHANNEL
    ] = (
        state.values[
            SAIL_TRIM_ACTION_CHANNEL
        ]
        + trim_delta
    )

    if (
        candidate.kind
        is ControlCandidateKind.PROBE
    ):
        next_uncertainty = max(
            0.0,
            state.uncertainty
            - 0.30,
        )

    elif (
        candidate.kind
        is ControlCandidateKind.CORRECTIVE
    ):
        next_uncertainty = max(
            0.0,
            state.uncertainty
            - 0.05,
        )

    else:
        next_uncertainty = min(
            1.0,
            state.uncertainty
            + 0.02,
        )

    return PredictiveState(
        values=values,
        target_values=dict(
            state.target_values
        ),
        timestamp=(
            None
            if state.timestamp is None
            else state.timestamp + 1.0
        ),
        uncertainty=next_uncertainty,
        metadata={
            "validation_case": "sailing_04A",
            "model": "normalized_yacht_crew_predictor",
            "hydrodynamic_claim": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Config
# =============================================================================


def build_predictive_control_config(
) -> PredictiveControlConfig:
    return PredictiveControlConfig(
        mode=PredictiveControlMode.PROBE_IF_UNCERTAIN,
        uncertainty_probe_threshold=0.35,
        max_safety_risk=0.25,
        min_reversibility_for_probe=0.80,
        residual_weight=2.5,
        cost_weight=0.30,
        uncertainty_weight=0.80,
        reserve_weight=0.35,
        safety_weight=1.0,
        min_stabilization_margin=-2.0,
        require_positive_reserve=True,
        metadata={
            "validation_case": "sailing_04A",
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Complete predictive run
# =============================================================================


def run_sailing_yacht_crew_predictive_control(
    case: SailingYachtCrewValidationCase | None = None,
    *,
    config: PredictiveControlConfig | None = None,
) -> SailingPredictiveControlResult:
    case = (
        case
        or build_case()
    )

    config = (
        config
        or build_predictive_control_config()
    )

    state = build_predictive_state(
        case
    )

    disturbance = build_disturbance_estimate(
        case
    )

    reserve = build_crew_control_reserve(
        case
    )

    demand = estimate_stabilization_demand(
        state,
        disturbance,
        uncertainty_weight=0.50,
        urgency=1.0,
        metadata={
            "validation_case": "sailing_04A",
            "external_expected_label_used": False,
        },
    )

    candidates = build_control_candidates()

    computed_demand, evaluations = evaluate_control_candidates(
        state,
        disturbance,
        candidates,
        reserve,
        model=sailing_prediction_model,
        config=config,
        demand=demand,
        metadata={
            "validation_case": "sailing_04A",
            "external_expected_label_used": False,
        },
    )

    decision = select_control_action(
        evaluations,
        config=config,
        metadata={
            "validation_case": "sailing_04A",
            "external_expected_label_used": False,
        },
    )

    return SailingPredictiveControlResult(
        case_id=case.case_id,
        predictive_state=state,
        disturbance=disturbance,
        reserve=reserve,
        demand=computed_demand,
        candidates=candidates,
        evaluations=evaluations,
        decision=decision,
        metadata={
            "validation_case": "sailing_04A",
            "control_question": (
                "can_predictive_control_estimate_stabilization_demand_"
                "and_select_safe_crew_action"
            ),
            "memory_enabled": False,
            "learning_enabled": False,
            "published_topology_basis": True,
            "normalized_predictive_parameters": True,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Observation / prediction-error helper
# =============================================================================


def compare_selected_prediction_with_observation(
    result: SailingPredictiveControlResult,
    observed_state: PredictiveState,
) -> PredictionError:
    """
    Compare the selected predicted outcome with an observed yacht state.
    """

    selected = result.decision.selected_evaluation

    if selected is None:
        raise SailingPredictiveControlError(
            "predictive-control decision has no selected evaluation"
        )

    return compare_prediction(
        selected.outcome,
        observed_state,
        metadata={
            "validation_case": "sailing_04A",
            "external_expected_label_used": False,
        },
    )


__all__ = [
    "COUPLED_CORRECTION_ID",
    "DEFAULT_COUPLED_AUTHORITY",
    "DEFAULT_CREW_RESERVE_CAPACITY",
    "DEFAULT_HELM_AUTHORITY",
    "DEFAULT_PROBE_MAGNITUDE",
    "DEFAULT_TRIM_AUTHORITY",
    "DEFAULT_WAVE_DISTURBANCE_SCALE",
    "DEFAULT_WIND_DISTURBANCE_SCALE",
    "HELM_CORRECTION_ID",
    "HOLD_ID",
    "INFORMATION_PROBE_ID",
    "SAIL_TRIM_ACTION_CHANNEL",
    "SailingPredictiveControlError",
    "SailingPredictiveControlResult",
    "TRIM_CORRECTION_ID",
    "build_case",
    "build_control_candidates",
    "build_crew_control_reserve",
    "build_disturbance_estimate",
    "build_predictive_control_config",
    "build_predictive_state",
    "channel_capacity",
    "channel_load",
    "channel_lookup",
    "compare_selected_prediction_with_observation",
    "run_sailing_yacht_crew_predictive_control",
    "sailing_prediction_model",
]
