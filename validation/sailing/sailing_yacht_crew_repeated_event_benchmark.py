"""
ROIF External-Domain Validation
Scenario 04B вЂ” Repeated-Event YachtвЂ“Crew Benchmark

Purpose
-------
Create a repeated-event sailing benchmark on top of:

- 04A yachtвЂ“crew coupled system;
- ROIF Predictive Control;
- ROIF Experience Trace.

04B is intentionally a NO-MEMORY baseline.

Each episode:

    disturbance
        ->
    predictive-control evaluation
        ->
    selected action / probe / hold
        ->
    synthetic observed state
        ->
    prediction error
        ->
    ExperienceTrace

Critical architectural invariant:

    Previous ExperienceTrace
            !=
    Input to next Predictive-Control decision

This file therefore provides the control condition required before introducing
persistent controller memory, MemoryScar, or PredictivePreload.

No learning is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.experience_trace import (
    ExperienceTrace,
    ExperienceTraceIdentity,
    StabilizationOutcome,
    build_experience_trace,
    trace_is_learning_free,
)
from roif.predictive_control import (
    CandidateEvaluation,
    ControlCandidate,
    ControlReserve,
    DisturbanceEstimate,
    PredictiveControlConfig,
    PredictiveControlDecision,
    PredictiveState,
    StabilizationDemand,
    estimate_stabilization_demand,
    evaluate_control_candidates,
    select_control_action,
)

from validation.sailing.sailing_yacht_crew_coupled_case import (
    COURSE_ERROR_CHANNEL,
    HEEL_YAW_CHANNEL,
    RUDDER_ACTION_CHANNEL,
    SAIL_TRIM_ACTION_CHANNEL,
    SailingYachtCrewValidationCase,
)

from validation.sailing.sailing_yacht_crew_predictive_control import (
    build_case,
    build_control_candidates,
    build_crew_control_reserve,
    build_predictive_control_config,
    build_predictive_state,
    sailing_prediction_model,
)


# =============================================================================
# Constants
# =============================================================================


SCENARIO_ID = "sailing_04B_repeated_event_no_memory"

DEFAULT_SEQUENCE_ID = "04B_default_sequence"

DEFAULT_OBSERVATION_MODEL_GAIN = 1.0

DEFAULT_OBSERVATION_NOISE = 0.0


# =============================================================================
# Errors
# =============================================================================


class RepeatedEventBenchmarkError(RuntimeError):
    """Base 04B repeated-event benchmark error."""


class InvalidRepeatedEventScenarioError(RepeatedEventBenchmarkError):
    """Raised when an episode scenario is invalid."""


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


# =============================================================================
# Event specification
# =============================================================================


@dataclass(frozen=True, slots=True)
class SailingDisturbanceEvent:
    """
    One external disturbance episode.

    wind and wave are normalized benchmark inputs.
    """

    event_id: str
    wind: float
    wave: float
    confidence: float = 0.90
    observation_bias_course: float = 0.0
    observation_bias_heel: float = 0.0
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        if not self.event_id:
            raise InvalidRepeatedEventScenarioError(
                "event_id must not be empty"
            )

        if self.wind < 0.0:
            raise InvalidRepeatedEventScenarioError(
                "wind must be non-negative"
            )

        if self.wave < 0.0:
            raise InvalidRepeatedEventScenarioError(
                "wave must be non-negative"
            )

        object.__setattr__(
            self,
            "confidence",
            max(
                0.0,
                min(
                    1.0,
                    float(
                        self.confidence
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


@dataclass(frozen=True, slots=True)
class RepeatedEventEpisodeResult:
    episode_index: int
    event: SailingDisturbanceEvent

    initial_state: PredictiveState
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

    observed_state: PredictiveState
    trace: ExperienceTrace

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


@dataclass(frozen=True, slots=True)
class RepeatedEventBenchmarkResult:
    scenario_id: str
    sequence_id: str
    episodes: tuple[
        RepeatedEventEpisodeResult,
        ...,
    ]

    mean_prediction_error: float
    mean_observed_residual: float
    improved_episode_count: int
    unchanged_episode_count: int
    worsened_episode_count: int

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "episodes",
            tuple(
                self.episodes
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
# Default disturbance sequence
# =============================================================================


def build_default_disturbance_sequence(
) -> tuple[
    SailingDisturbanceEvent,
    ...,
]:
    """
    Deterministic repeated disturbances.

    Pattern:
    moderate -> stronger -> similar -> stronger -> moderate

    The sequence deliberately contains repeated / similar disturbances so that
    a later memory-enabled controller can be compared against this no-memory
    baseline.
    """

    return (
        SailingDisturbanceEvent(
            event_id="event_01_moderate",
            wind=0.28,
            wave=0.18,
            observation_bias_course=0.08,
            observation_bias_heel=0.04,
            metadata={
                "pattern_family": "A",
                "external_expected_label_used": False,
            },
        ),
        SailingDisturbanceEvent(
            event_id="event_02_stronger",
            wind=0.42,
            wave=0.26,
            observation_bias_course=-0.03,
            observation_bias_heel=0.09,
            metadata={
                "pattern_family": "B",
                "external_expected_label_used": False,
            },
        ),
        SailingDisturbanceEvent(
            event_id="event_03_repeat_moderate",
            wind=0.29,
            wave=0.18,
            observation_bias_course=0.08,
            observation_bias_heel=0.04,
            metadata={
                "pattern_family": "A",
                "external_expected_label_used": False,
            },
        ),
        SailingDisturbanceEvent(
            event_id="event_04_strong",
            wind=0.48,
            wave=0.31,
            observation_bias_course=0.12,
            observation_bias_heel=0.07,
            metadata={
                "pattern_family": "C",
                "external_expected_label_used": False,
            },
        ),
        SailingDisturbanceEvent(
            event_id="event_05_repeat_moderate",
            wind=0.28,
            wave=0.19,
            observation_bias_course=0.08,
            observation_bias_heel=0.04,
            metadata={
                "pattern_family": "A",
                "external_expected_label_used": False,
            },
        ),
    )


# =============================================================================
# Episode construction
# =============================================================================


def build_event_disturbance_estimate(
    event: SailingDisturbanceEvent,
) -> DisturbanceEstimate:
    return DisturbanceEstimate(
        components={
            "wind": float(
                event.wind
            ),
            "wave": float(
                event.wave
            ),
        },
        confidence=event.confidence,
        metadata={
            "scenario_id": SCENARIO_ID,
            "event_id": event.event_id,
            "external_expected_label_used": False,
        },
    )


def build_episode_initial_state(
    case: SailingYachtCrewValidationCase,
    *,
    episode_index: int,
) -> PredictiveState:
    """
    Build the episode state from the same 04A benchmark source each time.

    This is intentional.

    04B is the NO-MEMORY baseline:
    previous traces do not modify the next episode's predictive state.
    """

    state = build_predictive_state(
        case
    )

    return PredictiveState(
        values=dict(
            state.values
        ),
        target_values=dict(
            state.target_values
        ),
        timestamp=float(
            episode_index
        ),
        uncertainty=state.uncertainty,
        metadata={
            **dict(
                state.metadata
            ),
            "scenario_id": SCENARIO_ID,
            "episode_index": episode_index,
            "memory_input_used": False,
            "previous_trace_used": False,
            "predictive_preload_used": False,
            "external_expected_label_used": False,
        },
    )


def synthesize_observed_state(
    *,
    initial_state: PredictiveState,
    selected_evaluation: CandidateEvaluation | None,
    event: SailingDisturbanceEvent,
) -> PredictiveState:
    """
    Build deterministic evaluator-side observation.

    This is not learning.

    If a candidate was selected, the observed state is based on the predicted
    state plus small event-specific deterministic bias.

    If no candidate was selected, the state is propagated with disturbance only.

    The function is intentionally transparent and deterministic so later
    memory-enabled benchmarks can be compared against exactly the same
    observation model.
    """

    if selected_evaluation is None:
        values = dict(
            initial_state.values
        )

        values[
            COURSE_ERROR_CHANNEL
        ] = (
            values[
                COURSE_ERROR_CHANNEL
            ]
            + 0.22
            * (
                event.wind
                + event.wave
            )
        )

        values[
            HEEL_YAW_CHANNEL
        ] = (
            values[
                HEEL_YAW_CHANNEL
            ]
            + 0.34
            * (
                event.wind
                + event.wave
            )
        )

        uncertainty = min(
            1.0,
            initial_state.uncertainty
            + 0.05,
        )

    else:
        predicted = (
            selected_evaluation.outcome.predicted_state
        )

        values = dict(
            predicted.values
        )

        values[
            COURSE_ERROR_CHANNEL
        ] = (
            values[
                COURSE_ERROR_CHANNEL
            ]
            + float(
                event.observation_bias_course
            )
        )

        values[
            HEEL_YAW_CHANNEL
        ] = (
            values[
                HEEL_YAW_CHANNEL
            ]
            + float(
                event.observation_bias_heel
            )
        )

        uncertainty = (
            predicted.uncertainty
        )

    return PredictiveState(
        values=values,
        target_values=dict(
            initial_state.target_values
        ),
        timestamp=(
            None
            if initial_state.timestamp is None
            else initial_state.timestamp + 1.0
        ),
        uncertainty=uncertainty,
        metadata={
            "scenario_id": SCENARIO_ID,
            "event_id": event.event_id,
            "observation_model": "deterministic_04B_evaluator",
            "learning_applied": False,
            "controller_memory_mutated": False,
            "external_expected_label_used": False,
        },
    )


def run_repeated_event_episode(
    *,
    case: SailingYachtCrewValidationCase,
    event: SailingDisturbanceEvent,
    episode_index: int,
    sequence_id: str,
    config: PredictiveControlConfig | None = None,
) -> RepeatedEventEpisodeResult:
    """
    Run one complete 04B episode.

    Previous ExperienceTrace objects are deliberately absent from the input.
    """

    config = (
        config
        or build_predictive_control_config()
    )

    initial_state = build_episode_initial_state(
        case,
        episode_index=episode_index,
    )

    event_disturbance = (
        build_event_disturbance_estimate(
            event
        )
    )

    episode_reserve = (
        build_crew_control_reserve(
            case
        )
    )

    episode_demand = (
        estimate_stabilization_demand(
            initial_state,
            event_disturbance,
            uncertainty_weight=0.50,
            urgency=1.0,
            metadata={
                "scenario_id": SCENARIO_ID,
                "event_id": event.event_id,
                "memory_input_used": False,
                "external_expected_label_used": False,
            },
        )
    )

    candidates = (
        build_control_candidates()
    )

    computed_demand, evaluations = (
        evaluate_control_candidates(
            initial_state,
            event_disturbance,
            candidates,
            episode_reserve,
            model=sailing_prediction_model,
            config=config,
            demand=episode_demand,
            metadata={
                "scenario_id": SCENARIO_ID,
                "event_id": event.event_id,
                "memory_input_used": False,
                "external_expected_label_used": False,
            },
        )
    )

    decision = select_control_action(
        evaluations,
        config=config,
        metadata={
            "scenario_id": SCENARIO_ID,
            "event_id": event.event_id,
            "memory_input_used": False,
            "external_expected_label_used": False,
        },
    )

    selected_evaluation = (
        decision.selected_evaluation
    )

    observed = synthesize_observed_state(
        initial_state=initial_state,
        selected_evaluation=selected_evaluation,
        event=event,
    )

    trace = build_experience_trace(
        identity=ExperienceTraceIdentity(
            trace_id=(
                f"{sequence_id}_"
                f"{episode_index:03d}_"
                f"{event.event_id}"
            ),
            episode_index=episode_index,
            sequence_id=sequence_id,
            timestamp_start=float(
                episode_index
            ),
            timestamp_end=float(
                episode_index + 1
            ),
            metadata={
                "scenario_id": SCENARIO_ID,
                "event_id": event.event_id,
                "memory_input_used": False,
                "external_expected_label_used": False,
            },
        ),
        initial_state=initial_state,
        disturbance=event_disturbance,
        demand=computed_demand,
        reserve=episode_reserve,
        candidates=candidates,
        evaluations=evaluations,
        decision=decision,
        observed_state=observed,
        metadata={
            "scenario_id": SCENARIO_ID,
            "event_id": event.event_id,
            "no_memory_baseline": True,
            "previous_trace_used": False,
            "learning_applied": False,
            "controller_memory_mutated": False,
            "predictive_preload_modified": False,
            "external_expected_label_used": False,
        },
    )

    return RepeatedEventEpisodeResult(
        episode_index=episode_index,
        event=event,
        initial_state=initial_state,
        disturbance=event_disturbance,
        reserve=episode_reserve,
        demand=computed_demand,
        candidates=candidates,
        evaluations=evaluations,
        decision=decision,
        observed_state=observed,
        trace=trace,
        metadata={
            "scenario_id": SCENARIO_ID,
            "sequence_id": sequence_id,
            "no_memory_baseline": True,
            "previous_trace_used": False,
            "learning_applied": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Benchmark aggregation
# =============================================================================


def _mean(
    values: Sequence[
        float
    ],
) -> float:
    if not values:
        return 0.0

    return (
        sum(
            float(
                value
            )
            for value
            in values
        )
        / len(
            values
        )
    )


def run_repeated_event_benchmark(
    *,
    case: SailingYachtCrewValidationCase | None = None,
    events: Sequence[
        SailingDisturbanceEvent
    ] | None = None,
    sequence_id: str = DEFAULT_SEQUENCE_ID,
    config: PredictiveControlConfig | None = None,
) -> RepeatedEventBenchmarkResult:
    """
    Run the complete 04B no-memory baseline.

    No episode receives prior ExperienceTrace objects.
    """

    case = (
        case
        or build_case()
    )

    event_sequence = tuple(
        events
        if events is not None
        else build_default_disturbance_sequence()
    )

    if not event_sequence:
        raise InvalidRepeatedEventScenarioError(
            "04B requires at least one disturbance event"
        )

    episodes = tuple(
        run_repeated_event_episode(
            case=case,
            event=event,
            episode_index=index,
            sequence_id=sequence_id,
            config=config,
        )
        for index, event
        in enumerate(
            event_sequence,
            start=1,
        )
    )

    prediction_errors = tuple(
        episode.trace.prediction_error.l2_error
        for episode
        in episodes
        if (
            episode.trace.prediction_error
            is not None
        )
    )

    observed_residuals = tuple(
        episode.observed_state.residual_norm()
        for episode
        in episodes
    )

    improved = sum(
        1
        for episode
        in episodes
        if (
            episode.trace.stabilization.outcome
            is StabilizationOutcome.IMPROVED
        )
    )

    unchanged = sum(
        1
        for episode
        in episodes
        if (
            episode.trace.stabilization.outcome
            is StabilizationOutcome.UNCHANGED
        )
    )

    worsened = sum(
        1
        for episode
        in episodes
        if (
            episode.trace.stabilization.outcome
            is StabilizationOutcome.WORSENED
        )
    )

    result = RepeatedEventBenchmarkResult(
        scenario_id=SCENARIO_ID,
        sequence_id=sequence_id,
        episodes=episodes,
        mean_prediction_error=_mean(
            prediction_errors
        ),
        mean_observed_residual=_mean(
            observed_residuals
        ),
        improved_episode_count=improved,
        unchanged_episode_count=unchanged,
        worsened_episode_count=worsened,
        metadata={
            "benchmark_type": "repeated_event_no_memory_baseline",
            "memory_enabled": False,
            "learning_enabled": False,
            "previous_trace_used_for_next_decision": False,
            "predictive_preload_enabled": False,
            "external_expected_label_used": False,
        },
    )

    if not all(
        trace_is_learning_free(
            episode.trace
        )
        for episode
        in result.episodes
    ):
        raise RepeatedEventBenchmarkError(
            "04B no-memory baseline unexpectedly applied learning."
        )

    return result


# =============================================================================
# Inspection helpers
# =============================================================================


def selected_action_ids(
    result: RepeatedEventBenchmarkResult,
) -> tuple[
    str | None,
    ...,
]:
    return tuple(
        episode.decision.selected_candidate_id
        for episode
        in result.episodes
    )


def prediction_error_series(
    result: RepeatedEventBenchmarkResult,
) -> tuple[
    float | None,
    ...,
]:
    return tuple(
        (
            None
            if episode.trace.prediction_error
            is None
            else episode.trace.prediction_error.l2_error
        )
        for episode
        in result.episodes
    )


def observed_residual_series(
    result: RepeatedEventBenchmarkResult,
) -> tuple[
    float,
    ...,
]:
    return tuple(
        episode.observed_state.residual_norm()
        for episode
        in result.episodes
    )


def all_traces_learning_free(
    result: RepeatedEventBenchmarkResult,
) -> bool:
    return all(
        trace_is_learning_free(
            episode.trace
        )
        for episode
        in result.episodes
    )


# =============================================================================
# Public exports
# =============================================================================


__all__ = [
    "DEFAULT_OBSERVATION_MODEL_GAIN",
    "DEFAULT_OBSERVATION_NOISE",
    "DEFAULT_SEQUENCE_ID",
    "InvalidRepeatedEventScenarioError",
    "RepeatedEventBenchmarkError",
    "RepeatedEventBenchmarkResult",
    "RepeatedEventEpisodeResult",
    "SCENARIO_ID",
    "SailingDisturbanceEvent",
    "all_traces_learning_free",
    "build_default_disturbance_sequence",
    "build_episode_initial_state",
    "build_event_disturbance_estimate",
    "observed_residual_series",
    "prediction_error_series",
    "run_repeated_event_benchmark",
    "run_repeated_event_episode",
    "selected_action_ids",
    "synthesize_observed_state",
]

