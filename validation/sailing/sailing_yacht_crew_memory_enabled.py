"""
ROIF External-Domain Validation
Scenario 04D — Memory-Enabled Sailing Benchmark

Purpose
-------
Demonstrate the first complete memory-enabled predictive-control loop.

Contrast with 04B:
    04B:
        repeated Pattern A
        -> same systematic prediction error
        -> no memory effect

    04D:
        repeated Pattern A
        -> ExperienceTrace
        -> ControllerMemory
        -> MemoryScar
        -> PredictivePreload
        -> reduced future prediction error

Architectural boundaries
------------------------

    Memory Retrieval != Action Selection
    PredictivePreload != Policy Override
    ExperienceTrace != Learning
    MemoryScar != Ground Truth
    Benchmark Pattern Labels != Controller Input

The controller does not receive evaluator pattern-family labels as action labels.
Pattern-family metadata is used only by the benchmark to identify which completed
traces belong to the repeated control condition.

The actual preload comes from derived prediction-error memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.controller_memory import (
    ControllerMemory,
    MemoryPattern,
    MemoryScar,
    append_experience_trace,
    derive_memory_scar,
)
from roif.experience_trace import (
    ExperienceTrace,
    ExperienceTraceIdentity,
    build_experience_trace,
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
from roif.predictive_preload import (
    MemoryPreloadContribution,
    PredictivePreload,
    PredictivePreloadConfig,
    apply_predictive_preload,
)

from validation.sailing.sailing_yacht_crew_predictive_control import (
    build_case,
    build_control_candidates,
    build_crew_control_reserve,
    build_predictive_control_config,
    build_predictive_state,
    sailing_prediction_model,
)

from validation.sailing.sailing_yacht_crew_repeated_event_benchmark import (
    SCENARIO_ID as BASELINE_SCENARIO_ID,
    SailingDisturbanceEvent,
    build_default_disturbance_sequence,
    build_event_disturbance_estimate,
    synthesize_observed_state,
)


SCENARIO_ID = "sailing_04D_memory_enabled"
DEFAULT_SEQUENCE_ID = "04D_memory_enabled_sequence"
MEMORY_PATTERN_ID = "A"


class MemoryEnabledSailingBenchmarkError(RuntimeError):
    """Base 04D error."""


@dataclass(frozen=True, slots=True)
class MemoryEnabledEpisodeResult:
    episode_index: int
    event: SailingDisturbanceEvent

    raw_initial_state: PredictiveState
    preload: PredictivePreload | None
    control_state: PredictiveState

    disturbance: DisturbanceEstimate
    reserve: ControlReserve
    demand: StabilizationDemand

    candidates: tuple[ControlCandidate, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    decision: PredictiveControlDecision

    observed_state: PredictiveState
    trace: ExperienceTrace

    memory_before: ControllerMemory
    memory_after: ControllerMemory

    scar_before: MemoryScar | None
    scar_after: MemoryScar | None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class MemoryEnabledSailingBenchmarkResult:
    scenario_id: str
    baseline_scenario_id: str
    sequence_id: str
    episodes: tuple[MemoryEnabledEpisodeResult, ...]
    final_memory: ControllerMemory
    pattern_a_error_series: tuple[float, ...]
    pattern_a_preload_series: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", tuple(self.episodes))
        object.__setattr__(
            self,
            "pattern_a_error_series",
            tuple(float(x) for x in self.pattern_a_error_series),
        )
        object.__setattr__(
            self,
            "pattern_a_preload_series",
            tuple(float(x) for x in self.pattern_a_preload_series),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def _pattern_family(event: SailingDisturbanceEvent) -> str:
    return str(event.metadata["pattern_family"])


def _build_raw_initial_state(
    *,
    episode_index: int,
) -> PredictiveState:
    case = build_case()
    state = build_predictive_state(case)

    return PredictiveState(
        values=dict(state.values),
        target_values=dict(state.target_values),
        timestamp=float(episode_index),
        uncertainty=state.uncertainty,
        metadata={
            **dict(state.metadata),
            "scenario_id": SCENARIO_ID,
            "episode_index": episode_index,
            "memory_preload_applied": False,
            "external_expected_label_used": False,
        },
    )


def _pattern_a_traces(
    memory: ControllerMemory,
) -> tuple[ExperienceTrace, ...]:
    traces = []

    for trace in memory.traces:
        family = (
            trace.metadata.get("pattern_family")
            or trace.identity.metadata.get("pattern_family")
            or trace.disturbance.metadata.get("pattern_family")
        )

        if family == MEMORY_PATTERN_ID:
            traces.append(trace)

    return tuple(traces)


def derive_pattern_a_scar(
    memory: ControllerMemory,
) -> MemoryScar | None:
    traces = _pattern_a_traces(memory)

    if len(traces) < 2:
        return None

    return derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id=MEMORY_PATTERN_ID,
            metadata={
                "scenario_id": SCENARIO_ID,
                "benchmark_pattern": True,
            },
        ),
        traces=traces,
        scar_id="04D_memory_scar_A",
        metadata={
            "scenario_id": SCENARIO_ID,
            "pattern_family": MEMORY_PATTERN_ID,
            "external_expected_label_used": False,
        },
    )


def _apply_memory_preload_if_available(
    state: PredictiveState,
    scar: MemoryScar | None,
) -> PredictivePreload | None:
    if scar is None:
        return None

    return apply_predictive_preload(
        state,
        (
            MemoryPreloadContribution(
                scar=scar,
                hierarchy_level=0,
                explicit_weight=1.0,
                source_cluster_id="04D_pattern_A_local",
                metadata={
                    "scenario_id": SCENARIO_ID,
                    "external_expected_label_used": False,
                },
            ),
        ),
        config=PredictivePreloadConfig(
            max_bias_fraction=0.75,
            uncertainty_reduction_scale=0.35,
            minimum_scar_confidence=0.25,
            minimum_scar_consistency=0.80,
            require_systematic_scar=True,
            local_level_decay=0.50,
        ),
        metadata={
            "scenario_id": SCENARIO_ID,
            "memory_pattern_id": MEMORY_PATTERN_ID,
            "external_expected_label_used": False,
        },
    )


def _prediction_error_l2(trace: ExperienceTrace) -> float:
    if trace.prediction_error is None:
        return 0.0
    return float(trace.prediction_error.l2_error)


def _preload_norm(preload: PredictivePreload | None) -> float:
    if preload is None:
        return 0.0

    return (
        sum(
            float(value) ** 2
            for value in preload.applied_bias_by_variable.values()
        )
        ** 0.5
    )


def run_memory_enabled_episode(
    *,
    memory: ControllerMemory,
    event: SailingDisturbanceEvent,
    episode_index: int,
    sequence_id: str = DEFAULT_SEQUENCE_ID,
    config: PredictiveControlConfig | None = None,
) -> MemoryEnabledEpisodeResult:
    case = build_case()
    config = config or build_predictive_control_config()

    raw_state = _build_raw_initial_state(
        episode_index=episode_index
    )

    scar_before = (
        derive_pattern_a_scar(memory)
        if _pattern_family(event) == MEMORY_PATTERN_ID
        else None
    )

    preload = _apply_memory_preload_if_available(
        raw_state,
        scar_before,
    )

    control_state = (
        raw_state
        if preload is None
        else preload.preloaded_state
    )

    disturbance = build_event_disturbance_estimate(
        event
    )

    reserve = build_crew_control_reserve(
        case
    )

    demand = estimate_stabilization_demand(
        control_state,
        disturbance,
        uncertainty_weight=0.50,
        urgency=1.0,
        metadata={
            "scenario_id": SCENARIO_ID,
            "pattern_family": _pattern_family(event),
            "memory_preload_used": preload is not None,
            "external_expected_label_used": False,
        },
    )

    candidates = build_control_candidates()

    computed_demand, evaluations = evaluate_control_candidates(
        control_state,
        disturbance,
        candidates,
        reserve,
        model=sailing_prediction_model,
        config=config,
        demand=demand,
        metadata={
            "scenario_id": SCENARIO_ID,
            "pattern_family": _pattern_family(event),
            "memory_preload_used": preload is not None,
            "external_expected_label_used": False,
        },
    )

    decision = select_control_action(
        evaluations,
        config=config,
        metadata={
            "scenario_id": SCENARIO_ID,
            "pattern_family": _pattern_family(event),
            "memory_preload_used": preload is not None,
            "external_expected_label_used": False,
        },
    )

    selected = decision.selected_evaluation

    # ---------------------------------------------------------------------
    # Evaluator-side plant truth
    # ---------------------------------------------------------------------
    # Critical benchmark rule:
    #
    #   observed plant state must NOT be generated from the memory-preloaded
    #   prediction itself.
    #
    # Otherwise:
    #       observation = preloaded_prediction + fixed_bias
    # and memory could never reduce prediction error.
    #
    # Instead we independently evaluate the SAME selected action from the raw,
    # non-preloaded state. That raw prediction represents the controller's
    # baseline plant model. The deterministic hidden event bias is then added
    # by the existing 04B observation model.
    #
    # The memory-enabled prediction is compared against this independent
    # evaluator-side observation.
    baseline_demand = estimate_stabilization_demand(
        raw_state,
        disturbance,
        uncertainty_weight=0.50,
        urgency=1.0,
        metadata={
            "scenario_id": SCENARIO_ID,
            "evaluator_side_baseline": True,
            "external_expected_label_used": False,
        },
    )

    _, baseline_evaluations = evaluate_control_candidates(
        raw_state,
        disturbance,
        candidates,
        reserve,
        model=sailing_prediction_model,
        config=config,
        demand=baseline_demand,
        metadata={
            "scenario_id": SCENARIO_ID,
            "evaluator_side_baseline": True,
            "external_expected_label_used": False,
        },
    )

    baseline_selected = None

    if selected is not None:
        baseline_selected = next(
            (
                evaluation
                for evaluation in baseline_evaluations
                if (
                    evaluation.candidate.candidate_id
                    == selected.candidate.candidate_id
                )
            ),
            None,
        )

    observed = synthesize_observed_state(
        initial_state=raw_state,
        selected_evaluation=baseline_selected,
        event=event,
    )

    # Preserve benchmark provenance explicitly.
    observed = PredictiveState(
        values=dict(observed.values),
        target_values=dict(observed.target_values),
        timestamp=observed.timestamp,
        uncertainty=observed.uncertainty,
        metadata={
            **dict(observed.metadata),
            "scenario_id": SCENARIO_ID,
            "observation_source": "independent_raw_state_plant_evaluator",
            "memory_preload_used_to_generate_observation": False,
            "external_expected_label_used": False,
        },
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
            timestamp_start=float(episode_index),
            timestamp_end=float(episode_index + 1),
            metadata={
                "scenario_id": SCENARIO_ID,
                "pattern_family": _pattern_family(event),
                "memory_preload_used": preload is not None,
                "external_expected_label_used": False,
            },
        ),
        initial_state=control_state,
        disturbance=disturbance,
        demand=computed_demand,
        reserve=reserve,
        candidates=candidates,
        evaluations=evaluations,
        decision=decision,
        observed_state=observed,
        metadata={
            "scenario_id": SCENARIO_ID,
            "pattern_family": _pattern_family(event),
            "memory_enabled": True,
            "memory_preload_used": preload is not None,
            "policy_modified_by_memory": False,
            "external_expected_label_used": False,
        },
    )

    memory_after = append_experience_trace(
        memory,
        trace,
        metadata={
            "scenario_id": SCENARIO_ID,
            "memory_enabled_benchmark": True,
            "external_expected_label_used": False,
        },
    )

    scar_after = (
        derive_pattern_a_scar(memory_after)
        if _pattern_family(event) == MEMORY_PATTERN_ID
        else None
    )

    return MemoryEnabledEpisodeResult(
        episode_index=episode_index,
        event=event,
        raw_initial_state=raw_state,
        preload=preload,
        control_state=control_state,
        disturbance=disturbance,
        reserve=reserve,
        demand=computed_demand,
        candidates=candidates,
        evaluations=evaluations,
        decision=decision,
        observed_state=observed,
        trace=trace,
        memory_before=memory,
        memory_after=memory_after,
        scar_before=scar_before,
        scar_after=scar_after,
        metadata={
            "scenario_id": SCENARIO_ID,
            "pattern_family": _pattern_family(event),
            "memory_preload_used": preload is not None,
            "memory_retrieval_used_for_action_selection": False,
            "policy_modified_by_memory": False,
            "external_expected_label_used": False,
        },
    )


def run_memory_enabled_sailing_benchmark(
    *,
    events: Sequence[SailingDisturbanceEvent] | None = None,
    sequence_id: str = DEFAULT_SEQUENCE_ID,
) -> MemoryEnabledSailingBenchmarkResult:
    event_sequence = tuple(
        build_default_disturbance_sequence()
        if events is None
        else events
    )

    if not event_sequence:
        raise MemoryEnabledSailingBenchmarkError(
            "04D requires at least one event"
        )

    memory = ControllerMemory(
        metadata={
            "scenario_id": SCENARIO_ID,
            "memory_enabled_benchmark": True,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )

    episodes = []

    for index, event in enumerate(
        event_sequence,
        start=1,
    ):
        episode = run_memory_enabled_episode(
            memory=memory,
            event=event,
            episode_index=index,
            sequence_id=sequence_id,
        )

        episodes.append(
            episode
        )

        memory = episode.memory_after

    pattern_a_episodes = tuple(
        episode
        for episode
        in episodes
        if _pattern_family(
            episode.event
        ) == MEMORY_PATTERN_ID
    )

    error_series = tuple(
        _prediction_error_l2(
            episode.trace
        )
        for episode
        in pattern_a_episodes
    )

    preload_series = tuple(
        _preload_norm(
            episode.preload
        )
        for episode
        in pattern_a_episodes
    )

    return MemoryEnabledSailingBenchmarkResult(
        scenario_id=SCENARIO_ID,
        baseline_scenario_id=BASELINE_SCENARIO_ID,
        sequence_id=sequence_id,
        episodes=tuple(
            episodes
        ),
        final_memory=memory,
        pattern_a_error_series=error_series,
        pattern_a_preload_series=preload_series,
        metadata={
            "benchmark_type": "memory_enabled_predictive_control",
            "memory_enabled": True,
            "predictive_preload_enabled": True,
            "memory_retrieval_used_for_action_selection": False,
            "policy_override_enabled": False,
            "pattern_labels_evaluator_only": True,
            "external_expected_label_used": False,
        },
    )


def selected_action_ids(
    result: MemoryEnabledSailingBenchmarkResult,
) -> tuple[str | None, ...]:
    return tuple(
        episode.decision.selected_candidate_id
        for episode
        in result.episodes
    )


def pattern_a_memory_usage(
    result: MemoryEnabledSailingBenchmarkResult,
) -> tuple[bool, ...]:
    return tuple(
        episode.preload is not None
        for episode
        in result.episodes
        if _pattern_family(
            episode.event
        ) == MEMORY_PATTERN_ID
    )


def pattern_a_scar_confidence_series(
    result: MemoryEnabledSailingBenchmarkResult,
) -> tuple[float | None, ...]:
    return tuple(
        (
            None
            if episode.scar_after is None
            else episode.scar_after.confidence
        )
        for episode
        in result.episodes
        if _pattern_family(
            episode.event
        ) == MEMORY_PATTERN_ID
    )


__all__ = [
    "DEFAULT_SEQUENCE_ID",
    "MEMORY_PATTERN_ID",
    "MemoryEnabledEpisodeResult",
    "MemoryEnabledSailingBenchmarkError",
    "MemoryEnabledSailingBenchmarkResult",
    "SCENARIO_ID",
    "derive_pattern_a_scar",
    "pattern_a_memory_usage",
    "pattern_a_scar_confidence_series",
    "run_memory_enabled_episode",
    "run_memory_enabled_sailing_benchmark",
    "selected_action_ids",
]
