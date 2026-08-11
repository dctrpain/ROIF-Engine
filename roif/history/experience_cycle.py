"""
ROIF Memory 2.0
Experience Cycle Core

Twenty-third layer and final architectural orchestration layer before
end-to-end benchmarking.

Purpose
-------
Execute one complete deterministic experience-to-memory-state cycle:

    M_t
      -> feedback trajectory
      -> feedback stability
      -> consolidation candidate
      -> explicit memory commitment
      -> memory integration
      -> M_(t+1)

This module is an orchestrator. It intentionally delegates all substantive
logic to already-tested lower layers and only coordinates provenance,
gates, and immutable state transition.

Important boundaries:
- no hidden learning
- no silent commitment
- no rewrite/delete/merge of existing traces
- source MemoryState remains immutable
- commitment requires explicit caller intent
- integration remains analysis-only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.experience_attractor import ExperienceAttractor
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    FeedbackTrajectory,
    build_feedback_trajectory,
    feedback_trajectory_is_policy_free,
)
from roif.history.feedback_stability import (
    FeedbackStabilityConfig,
    FeedbackStabilityResult,
    analyze_feedback_stability,
    feedback_stability_is_policy_free,
)
from roif.history.experience_consolidation import (
    ExperienceConsolidationCandidate,
    ExperienceConsolidationConfig,
    consolidation_is_policy_free,
    evaluate_experience_consolidation,
)
from roif.history.memory_commitment import (
    CommittedMemoryTrace,
    MemoryCommitmentConfig,
    commit_memory_trace,
    memory_commitment_is_policy_free,
)
from roif.history.memory_integration import (
    MemoryIntegrationConfig,
    MemoryIntegrationResult,
    analyze_memory_integration,
    memory_integration_is_policy_free,
)
from roif.history.memory_state import (
    MemoryState,
    MemoryStateTransition,
    append_committed_trace,
    memory_state_is_policy_free,
    memory_state_transition_is_policy_free,
)
from roif.history.reconstruction_feedback import (
    ReconstructionFeedbackConfig,
)


SCHEMA_VERSION = "experience_cycle_v1"


class ExperienceCycleError(ValueError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _validate_id(
    name: str,
    value: str,
) -> str:
    cleaned = str(value).strip()

    if not cleaned:
        raise ExperienceCycleError(
            f"{name} must not be empty"
        )

    return cleaned


@dataclass(frozen=True, slots=True)
class ExperienceCycleConfig:
    feedback: ReconstructionFeedbackConfig = field(
        default_factory=ReconstructionFeedbackConfig
    )
    stability: FeedbackStabilityConfig = field(
        default_factory=FeedbackStabilityConfig
    )
    consolidation: ExperienceConsolidationConfig = field(
        default_factory=ExperienceConsolidationConfig
    )
    commitment: MemoryCommitmentConfig = field(
        default_factory=MemoryCommitmentConfig
    )
    integration: MemoryIntegrationConfig = field(
        default_factory=MemoryIntegrationConfig
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperienceCycleResult:
    cycle_id: str

    source_memory_state: MemoryState
    target_memory_state: MemoryState

    feedback_trajectory: FeedbackTrajectory
    stability: FeedbackStabilityResult
    consolidation_candidate: ExperienceConsolidationCandidate
    committed_trace: CommittedMemoryTrace
    integration: MemoryIntegrationResult
    memory_transition: MemoryStateTransition

    explicit_commit_requested: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cycle_id",
            _validate_id(
                "cycle_id",
                self.cycle_id,
            ),
        )

        if (
            self.memory_transition.source_state_id
            != self.source_memory_state.state_id
        ):
            raise ExperienceCycleError(
                "memory transition source state must match source_memory_state"
            )

        if (
            self.memory_transition.target_state_id
            != self.target_memory_state.state_id
        ):
            raise ExperienceCycleError(
                "memory transition target state must match target_memory_state"
            )

        if (
            self.committed_trace.trace_id
            != self.memory_transition.appended_trace_id
        ):
            raise ExperienceCycleError(
                "committed trace must match appended memory trace"
            )

        if (
            self.integration.new_trace_id
            != self.committed_trace.trace_id
        ):
            raise ExperienceCycleError(
                "integration must reference committed trace"
            )

        if (
            self.consolidation_candidate.candidate_id
            != self.committed_trace.source_candidate_id
        ):
            raise ExperienceCycleError(
                "committed trace provenance must match consolidation candidate"
            )

        if (
            self.stability.trajectory_id
            != self.feedback_trajectory.trajectory_id
        ):
            raise ExperienceCycleError(
                "stability must reference feedback trajectory"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def run_experience_cycle(
    *,
    cycle_id: str,
    source_memory_state: MemoryState,
    target_memory_state_id: str,
    frames: Iterable[FeedbackFrame],
    attractors: Iterable[ExperienceAttractor],
    explicit_commit_requested: bool,
    trace_id: str,
    candidate_id: str,
    config: ExperienceCycleConfig | None = None,
    seed_from_first_baseline: bool = True,
    initial_reconstructed_vector: Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExperienceCycleResult:
    cycle_id = _validate_id(
        "cycle_id",
        cycle_id,
    )

    target_memory_state_id = _validate_id(
        "target_memory_state_id",
        target_memory_state_id,
    )

    trace_id = _validate_id(
        "trace_id",
        trace_id,
    )

    candidate_id = _validate_id(
        "candidate_id",
        candidate_id,
    )

    if not memory_state_is_policy_free(
        source_memory_state
    ):
        raise ExperienceCycleError(
            "source memory state must be policy-free"
        )

    actual_config = (
        config
        if config is not None
        else ExperienceCycleConfig()
    )

    frame_items = tuple(
        frames
    )

    if not frame_items:
        raise ExperienceCycleError(
            "frames must not be empty"
        )

    attractor_items = tuple(
        attractors
    )

    if not attractor_items:
        raise ExperienceCycleError(
            "attractors must not be empty"
        )

    feedback_trajectory = build_feedback_trajectory(
        trajectory_id=f"{cycle_id}::feedback_trajectory",
        frames=frame_items,
        attractors=attractor_items,
        config=actual_config.feedback,
        seed_from_first_baseline=seed_from_first_baseline,
        initial_reconstructed_vector=initial_reconstructed_vector,
        metadata={
            "source_memory_state_id": source_memory_state.state_id,
            "cycle_id": cycle_id,
        },
    )

    if not feedback_trajectory_is_policy_free(
        feedback_trajectory
    ):
        raise ExperienceCycleError(
            "feedback trajectory must remain policy-free"
        )

    stability = analyze_feedback_stability(
        analysis_id=f"{cycle_id}::stability",
        trajectory=feedback_trajectory,
        config=actual_config.stability,
        metadata={
            "cycle_id": cycle_id,
        },
    )

    if not feedback_stability_is_policy_free(
        stability
    ):
        raise ExperienceCycleError(
            "stability analysis must remain policy-free"
        )

    consolidation_candidate = evaluate_experience_consolidation(
        candidate_id=candidate_id,
        trajectory=feedback_trajectory,
        stability=stability,
        config=actual_config.consolidation,
        metadata={
            "cycle_id": cycle_id,
        },
    )

    if not consolidation_is_policy_free(
        consolidation_candidate
    ):
        raise ExperienceCycleError(
            "consolidation candidate must remain policy-free"
        )

    committed_trace = commit_memory_trace(
        trace_id=trace_id,
        candidate=consolidation_candidate,
        explicit_commit_requested=explicit_commit_requested,
        config=actual_config.commitment,
        metadata={
            "cycle_id": cycle_id,
        },
    )

    if not memory_commitment_is_policy_free(
        committed_trace
    ):
        raise ExperienceCycleError(
            "committed trace must remain policy-free"
        )

    integration = analyze_memory_integration(
        integration_id=f"{cycle_id}::integration",
        new_trace=committed_trace,
        existing_traces=source_memory_state.traces,
        config=actual_config.integration,
        metadata={
            "cycle_id": cycle_id,
            "source_memory_state_id": source_memory_state.state_id,
        },
    )

    if not memory_integration_is_policy_free(
        integration
    ):
        raise ExperienceCycleError(
            "memory integration must remain policy-free"
        )

    target_memory_state, memory_transition = append_committed_trace(
        transition_id=f"{cycle_id}::memory_transition",
        target_state_id=target_memory_state_id,
        state=source_memory_state,
        trace=committed_trace,
        integration=integration,
        metadata={
            "cycle_id": cycle_id,
        },
    )

    if not memory_state_is_policy_free(
        target_memory_state
    ):
        raise ExperienceCycleError(
            "target memory state must remain policy-free"
        )

    if not memory_state_transition_is_policy_free(
        memory_transition
    ):
        raise ExperienceCycleError(
            "memory state transition must remain policy-free"
        )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "experience_cycle_mode": "deterministic_orchestration",
        "source_memory_state_id": source_memory_state.state_id,
        "target_memory_state_id": target_memory_state.state_id,
        "source_revision": source_memory_state.revision,
        "target_revision": target_memory_state.revision,
        "explicit_commit_requested": bool(
            explicit_commit_requested
        ),
        "memory_mutated_in_place": False,
        "existing_trace_rewritten": False,
        "trace_deleted": False,
        "trace_merged": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_cycle_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return ExperienceCycleResult(
        cycle_id=cycle_id,
        source_memory_state=source_memory_state,
        target_memory_state=target_memory_state,
        feedback_trajectory=feedback_trajectory,
        stability=stability,
        consolidation_candidate=consolidation_candidate,
        committed_trace=committed_trace,
        integration=integration,
        memory_transition=memory_transition,
        explicit_commit_requested=bool(
            explicit_commit_requested
        ),
        metadata=merged_metadata,
    )


def experience_cycle_is_policy_free(
    result: ExperienceCycleResult,
) -> bool:
    return (
        result.metadata.get(
            "action_selected"
        )
        is False
        and result.metadata.get(
            "policy_modified"
        )
        is False
        and result.metadata.get(
            "memory_mutated_in_place"
        )
        is False
        and result.metadata.get(
            "existing_trace_rewritten"
        )
        is False
        and result.metadata.get(
            "trace_deleted"
        )
        is False
        and result.metadata.get(
            "trace_merged"
        )
        is False
        and memory_state_is_policy_free(
            result.source_memory_state
        )
        and memory_state_is_policy_free(
            result.target_memory_state
        )
        and feedback_trajectory_is_policy_free(
            result.feedback_trajectory
        )
        and feedback_stability_is_policy_free(
            result.stability
        )
        and consolidation_is_policy_free(
            result.consolidation_candidate
        )
        and memory_commitment_is_policy_free(
            result.committed_trace
        )
        and memory_integration_is_policy_free(
            result.integration
        )
        and memory_state_transition_is_policy_free(
            result.memory_transition
        )
    )


def experience_cycle_signature(
    result: ExperienceCycleResult,
) -> tuple[
    str,
    str,
    int,
    str,
    int,
    str,
]:
    return (
        result.cycle_id,
        result.source_memory_state.state_id,
        result.source_memory_state.revision,
        result.target_memory_state.state_id,
        result.target_memory_state.revision,
        result.committed_trace.trace_id,
    )


__all__ = [
    "ExperienceCycleConfig",
    "ExperienceCycleError",
    "ExperienceCycleResult",
    "SCHEMA_VERSION",
    "experience_cycle_is_policy_free",
    "experience_cycle_signature",
    "run_experience_cycle",
]
