"""
ROIF External-Domain Validation
Scenario 04F — Autonomous Memory Retrieval E2E Benchmark

Purpose
-------
Validate the first complete autonomous memory-enabled control loop using the
explicit RecursiveMemoryTree:

    past episodes
        -> ExperienceTrace
        -> ControllerHistoryAdapter
        -> HierarchicalMemory
        -> RecursiveMemoryTree
        -> automatic nearest-cluster retrieval
        -> real ancestry: local -> parent -> grandparent -> ...
        -> PredictivePreload
        -> Predictive Control
        -> independent observation
        -> prediction-error comparison

Critical distinction
--------------------
04D manually selected Pattern-A memory.
Early 04F synthesized parent/grandparent context during retrieval.
Current 04F does neither.

The query is matched to HierarchicalMemory by StructuralSignature geometry.
The recursive tree supplies the real hierarchy topology.

Architectural boundaries
------------------------

    Retrieval != Action Selection
    PredictivePreload != Policy Override
    Cluster Match != Evaluator Label
    Recursive Tree != Ground Truth
    Observation != Preloaded Prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from roif.experience_trace import (
    ExperienceTrace,
    ExperienceTraceIdentity,
    build_experience_trace,
)
from roif.hierarchical_memory import (
    HierarchicalMemory,
    HierarchicalMemoryConfig,
    append_controller_history_record,
)
from roif.hierarchical_memory_retrieval import (
    HierarchicalMemoryRetrieval,
    RetrievalConfig,
    retrieve_hierarchical_memory,
)
from roif.history.controller_history_adapter import (
    ControllerHistoryRecord,
    adapt_experience_trace,
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
    PredictivePreload,
    PredictivePreloadConfig,
    apply_predictive_preload,
)
from roif.recursive_hierarchical_memory import (
    RecursiveMemoryTree,
    build_recursive_memory_tree,
)

from validation.sailing.sailing_yacht_crew_long_memory_compression import (
    build_long_memory_event,
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
    SailingDisturbanceEvent,
    build_event_disturbance_estimate,
    synthesize_observed_state,
)


SCENARIO_ID = "sailing_04F_memory_e2e"
DEFAULT_HISTORY_EVENT_COUNT = 100
DEFAULT_QUERY_INDEX = 101


class SailingMemoryE2EError(RuntimeError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


@dataclass(frozen=True, slots=True)
class SailingMemoryE2EHistory:
    event_count: int
    hierarchical_memory: HierarchicalMemory
    recursive_tree: RecursiveMemoryTree
    records: tuple[ControllerHistoryRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(self.records),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class SailingMemoryE2EQueryResult:
    event: SailingDisturbanceEvent

    raw_state: PredictiveState
    query_record: ControllerHistoryRecord
    retrieval: HierarchicalMemoryRetrieval
    preload: PredictivePreload

    disturbance: DisturbanceEstimate
    reserve: ControlReserve
    demand: StabilizationDemand

    candidates: tuple[ControlCandidate, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    decision: PredictiveControlDecision

    observed_state: PredictiveState
    memory_trace: ExperienceTrace
    baseline_trace: ExperienceTrace

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates",
            tuple(self.candidates),
        )
        object.__setattr__(
            self,
            "evaluations",
            tuple(self.evaluations),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )

    @property
    def memory_error(self) -> float:
        if self.memory_trace.prediction_error is None:
            return 0.0
        return float(
            self.memory_trace.prediction_error.l2_error
        )

    @property
    def baseline_error(self) -> float:
        if self.baseline_trace.prediction_error is None:
            return 0.0
        return float(
            self.baseline_trace.prediction_error.l2_error
        )

    @property
    def improvement_fraction(self) -> float:
        if self.baseline_error <= 1e-12:
            return 0.0

        return (
            self.baseline_error
            - self.memory_error
        ) / self.baseline_error


@dataclass(frozen=True, slots=True)
class SailingMemoryE2EResult:
    scenario_id: str
    history: SailingMemoryE2EHistory
    query: SailingMemoryE2EQueryResult
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


# =============================================================================
# Shared episode helpers
# =============================================================================


def _raw_state(
    *,
    episode_index: int,
) -> PredictiveState:
    case = build_case()
    base = build_predictive_state(case)

    return PredictiveState(
        values=dict(base.values),
        target_values=dict(base.target_values),
        timestamp=float(episode_index),
        uncertainty=base.uncertainty,
        metadata={
            **dict(base.metadata),
            "scenario_id": SCENARIO_ID,
            "episode_index": episode_index,
            "predictive_preload_applied": False,
            "external_expected_label_used": False,
        },
    )


def _run_control_cycle(
    *,
    state: PredictiveState,
    event: SailingDisturbanceEvent,
    episode_index: int,
    sequence_id: str,
    config: PredictiveControlConfig,
    trace_metadata: Mapping[str, Any],
) -> tuple[
    DisturbanceEstimate,
    ControlReserve,
    StabilizationDemand,
    tuple[ControlCandidate, ...],
    tuple[CandidateEvaluation, ...],
    PredictiveControlDecision,
    PredictiveState,
    ExperienceTrace,
]:
    case = build_case()

    disturbance = build_event_disturbance_estimate(event)
    reserve = build_crew_control_reserve(case)

    demand = estimate_stabilization_demand(
        state,
        disturbance,
        uncertainty_weight=0.50,
        urgency=1.0,
        metadata={
            "scenario_id": SCENARIO_ID,
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
            "scenario_id": SCENARIO_ID,
            "external_expected_label_used": False,
        },
    )

    decision = select_control_action(
        evaluations,
        config=config,
        metadata={
            "scenario_id": SCENARIO_ID,
            "external_expected_label_used": False,
        },
    )

    observed = synthesize_observed_state(
        initial_state=state,
        selected_evaluation=decision.selected_evaluation,
        event=event,
    )

    trace = build_experience_trace(
        identity=ExperienceTraceIdentity(
            trace_id=(
                f"{sequence_id}_"
                f"{episode_index:05d}_"
                f"{event.event_id}"
            ),
            episode_index=episode_index,
            sequence_id=sequence_id,
            timestamp_start=float(episode_index),
            timestamp_end=float(episode_index + 1),
            metadata={
                "scenario_id": SCENARIO_ID,
                **dict(trace_metadata),
                "external_expected_label_used": False,
            },
        ),
        initial_state=state,
        disturbance=disturbance,
        demand=computed_demand,
        reserve=reserve,
        candidates=candidates,
        evaluations=evaluations,
        decision=decision,
        observed_state=observed,
        metadata={
            "scenario_id": SCENARIO_ID,
            **dict(trace_metadata),
            "external_expected_label_used": False,
        },
    )

    return (
        disturbance,
        reserve,
        computed_demand,
        candidates,
        evaluations,
        decision,
        observed,
        trace,
    )


# =============================================================================
# History building
# =============================================================================


def build_e2e_history(
    *,
    event_count: int = DEFAULT_HISTORY_EVENT_COUNT,
) -> SailingMemoryE2EHistory:
    if event_count < 2:
        raise SailingMemoryE2EError(
            "history requires at least two events"
        )

    config = build_predictive_control_config()

    memory = HierarchicalMemory(
        config=HierarchicalMemoryConfig(
            assignment_radius=0.10,
            anomaly_radius=0.50,
            max_representatives=5,
            representative_refresh_interval=10,
        ),
        metadata={
            "scenario_id": SCENARIO_ID,
            "history_role": True,
            "predictive_preload_applied": False,
            "policy_modified": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )

    records = []

    for index in range(
        1,
        event_count + 1,
    ):
        event = build_long_memory_event(index)
        state = _raw_state(
            episode_index=index
        )

        (
            _disturbance,
            _reserve,
            _demand,
            _candidates,
            _evaluations,
            _decision,
            _observed,
            trace,
        ) = _run_control_cycle(
            state=state,
            event=event,
            episode_index=index,
            sequence_id="04F_history",
            config=config,
            trace_metadata={
                "history_episode": True,
                "pattern_family": event.metadata[
                    "pattern_family"
                ],
            },
        )

        record = adapt_experience_trace(
            trace,
            pattern_id=None,
        )

        records.append(record)

        memory = append_controller_history_record(
            memory,
            record,
        )

    recursive_tree = build_recursive_memory_tree(
        memory
    )

    return SailingMemoryE2EHistory(
        event_count=event_count,
        hierarchical_memory=memory,
        recursive_tree=recursive_tree,
        records=tuple(records),
        metadata={
            "scenario_id": SCENARIO_ID,
            "history_built_without_predictive_preload": True,
            "recursive_tree_built": True,
            "recursive_tree_node_count": recursive_tree.node_count,
            "recursive_tree_max_level": recursive_tree.max_level,
            "cluster_assignment_uses_external_pattern_label": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Query construction
# =============================================================================


def _build_query_probe_record(
    event: SailingDisturbanceEvent,
    *,
    episode_index: int,
    config: PredictiveControlConfig,
) -> ControllerHistoryRecord:
    state = _raw_state(
        episode_index=episode_index
    )

    (
        _disturbance,
        _reserve,
        _demand,
        _candidates,
        _evaluations,
        _decision,
        _observed,
        trace,
    ) = _run_control_cycle(
        state=state,
        event=event,
        episode_index=episode_index,
        sequence_id="04F_query_probe",
        config=config,
        trace_metadata={
            "query_probe": True,
            "pattern_family": event.metadata[
                "pattern_family"
            ],
            "used_for_retrieval_only": True,
        },
    )

    return adapt_experience_trace(
        trace,
        pattern_id=None,
    )


# =============================================================================
# Independent evaluator-side observation
# =============================================================================


def _independent_observation_for_selected_action(
    *,
    raw_state: PredictiveState,
    event: SailingDisturbanceEvent,
    selected_evaluation: CandidateEvaluation | None,
    config: PredictiveControlConfig,
) -> PredictiveState:
    case = build_case()
    disturbance = build_event_disturbance_estimate(event)
    reserve = build_crew_control_reserve(case)

    demand = estimate_stabilization_demand(
        raw_state,
        disturbance,
        uncertainty_weight=0.50,
        urgency=1.0,
        metadata={
            "scenario_id": SCENARIO_ID,
            "evaluator_side_raw_state": True,
            "external_expected_label_used": False,
        },
    )

    candidates = build_control_candidates()

    _, baseline_evaluations = evaluate_control_candidates(
        raw_state,
        disturbance,
        candidates,
        reserve,
        model=sailing_prediction_model,
        config=config,
        demand=demand,
        metadata={
            "scenario_id": SCENARIO_ID,
            "evaluator_side_raw_state": True,
            "external_expected_label_used": False,
        },
    )

    baseline_selected = None

    if selected_evaluation is not None:
        baseline_selected = next(
            (
                evaluation
                for evaluation in baseline_evaluations
                if (
                    evaluation.candidate.candidate_id
                    == selected_evaluation.candidate.candidate_id
                )
            ),
            None,
        )

    observed = synthesize_observed_state(
        initial_state=raw_state,
        selected_evaluation=baseline_selected,
        event=event,
    )

    return PredictiveState(
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


# =============================================================================
# E2E query
# =============================================================================


def run_e2e_query(
    history: SailingMemoryE2EHistory,
    *,
    query_index: int = DEFAULT_QUERY_INDEX,
) -> SailingMemoryE2EQueryResult:
    event = build_long_memory_event(
        query_index
    )

    config = build_predictive_control_config()

    raw_state = _raw_state(
        episode_index=query_index
    )

    query_record = _build_query_probe_record(
        event,
        episode_index=query_index,
        config=config,
    )

    retrieval = retrieve_hierarchical_memory(
        history.hierarchical_memory,
        history.recursive_tree,
        query_record,
        config=RetrievalConfig(
            max_hierarchy_levels=3,
            local_weight=1.0,
            parent_weight=1.0,
            grandparent_weight=1.0,
            minimum_cluster_members_for_scar=2,
        ),
    )

    preload = apply_predictive_preload(
        raw_state,
        retrieval.contributions,
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
            "automatic_retrieval": True,
            "recursive_tree_retrieval": True,
            "external_expected_label_used": False,
        },
    )

    control_state = preload.preloaded_state

    case = build_case()
    disturbance = build_event_disturbance_estimate(event)
    reserve = build_crew_control_reserve(case)

    demand = estimate_stabilization_demand(
        control_state,
        disturbance,
        uncertainty_weight=0.50,
        urgency=1.0,
        metadata={
            "scenario_id": SCENARIO_ID,
            "automatic_retrieval": True,
            "recursive_tree_retrieval": True,
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
            "automatic_retrieval": True,
            "recursive_tree_retrieval": True,
            "external_expected_label_used": False,
        },
    )

    decision = select_control_action(
        evaluations,
        config=config,
        metadata={
            "scenario_id": SCENARIO_ID,
            "memory_retrieval_used_for_action_selection": False,
            "recursive_tree_retrieval": True,
            "external_expected_label_used": False,
        },
    )

    observed = _independent_observation_for_selected_action(
        raw_state=raw_state,
        event=event,
        selected_evaluation=decision.selected_evaluation,
        config=config,
    )

    memory_trace = build_experience_trace(
        identity=ExperienceTraceIdentity(
            trace_id=(
                f"04F_memory_query_"
                f"{query_index:05d}_"
                f"{event.event_id}"
            ),
            episode_index=query_index,
            sequence_id="04F_memory_query",
            timestamp_start=float(query_index),
            timestamp_end=float(query_index + 1),
            metadata={
                "scenario_id": SCENARIO_ID,
                "automatic_retrieval": True,
                "recursive_tree_retrieval": True,
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
            "automatic_retrieval": True,
            "recursive_tree_retrieval": True,
            "policy_modified_by_memory": False,
            "external_expected_label_used": False,
        },
    )

    (
        _bd,
        _br,
        _bde,
        _bc,
        _be,
        _bdecision,
        _bobserved,
        baseline_trace,
    ) = _run_control_cycle(
        state=raw_state,
        event=event,
        episode_index=query_index,
        sequence_id="04F_baseline_query",
        config=config,
        trace_metadata={
            "scenario_id": SCENARIO_ID,
            "baseline_comparison": True,
            "external_expected_label_used": False,
        },
    )

    return SailingMemoryE2EQueryResult(
        event=event,
        raw_state=raw_state,
        query_record=query_record,
        retrieval=retrieval,
        preload=preload,
        disturbance=disturbance,
        reserve=reserve,
        demand=computed_demand,
        candidates=candidates,
        evaluations=evaluations,
        decision=decision,
        observed_state=observed,
        memory_trace=memory_trace,
        baseline_trace=baseline_trace,
        metadata={
            "scenario_id": SCENARIO_ID,
            "query_pattern_family_for_reporting_only": event.metadata[
                "pattern_family"
            ],
            "cluster_assignment_uses_external_pattern_label": False,
            "automatic_retrieval": True,
            "recursive_tree_retrieval": True,
            "retrieval_tree_levels": tuple(
                level.tree_level
                for level in retrieval.levels
            ),
            "retrieval_tree_node_ids": tuple(
                level.tree_node_id
                for level in retrieval.levels
            ),
            "memory_retrieval_used_for_action_selection": False,
            "policy_modified_by_memory": False,
            "external_expected_label_used": False,
        },
    )


def run_sailing_memory_e2e_benchmark(
    *,
    history_event_count: int = DEFAULT_HISTORY_EVENT_COUNT,
    query_index: int = DEFAULT_QUERY_INDEX,
) -> SailingMemoryE2EResult:
    history = build_e2e_history(
        event_count=history_event_count
    )

    query = run_e2e_query(
        history,
        query_index=query_index,
    )

    return SailingMemoryE2EResult(
        scenario_id=SCENARIO_ID,
        history=history,
        query=query,
        metadata={
            "benchmark_type": "autonomous_recursive_memory_retrieval_e2e",
            "automatic_retrieval": True,
            "recursive_memory_tree_enabled": True,
            "nested_memory_enabled": True,
            "predictive_preload_enabled": True,
            "policy_override_enabled": False,
            "action_selected_by_memory": False,
            "cluster_assignment_uses_external_pattern_label": False,
            "external_expected_label_used": False,
        },
    )


def retrieval_level_ids(
    result: SailingMemoryE2EResult,
) -> tuple[int, ...]:
    return tuple(
        level.level
        for level in result.query.retrieval.levels
    )


def retrieval_tree_levels(
    result: SailingMemoryE2EResult,
) -> tuple[int, ...]:
    return tuple(
        level.tree_level
        for level in result.query.retrieval.levels
    )


def retrieval_tree_node_ids(
    result: SailingMemoryE2EResult,
) -> tuple[str, ...]:
    return tuple(
        level.tree_node_id
        for level in result.query.retrieval.levels
    )


def selected_action_id(
    result: SailingMemoryE2EResult,
) -> str | None:
    return result.query.decision.selected_candidate_id


__all__ = [
    "DEFAULT_HISTORY_EVENT_COUNT",
    "DEFAULT_QUERY_INDEX",
    "SCENARIO_ID",
    "SailingMemoryE2EError",
    "SailingMemoryE2EHistory",
    "SailingMemoryE2EQueryResult",
    "SailingMemoryE2EResult",
    "build_e2e_history",
    "retrieval_level_ids",
    "retrieval_tree_levels",
    "retrieval_tree_node_ids",
    "run_e2e_query",
    "run_sailing_memory_e2e_benchmark",
    "selected_action_id",
]
