"""
ROIF Memory 2.0
Memory Commitment Core

Twentieth layer above:
    ExperienceTransformation
    ExperienceSequence
    ExperiencePattern
    ExperienceAttractor
    AttractorDynamics
    AttractorTrajectory
    ExperienceConditioning
    ConditionedPrestress
    ConditionedDynamics
    ConditionedTrajectory
    AssociativeContext
    ContextualDynamics
    ContextualTrajectory
    ContextualReconstruction
    ReconstructionTrajectory
    ReconstructionFeedback
    FeedbackTrajectory
    FeedbackStability
    ExperienceConsolidation

Purpose
-------
Convert an eligible ExperienceConsolidationCandidate into a new immutable
CommittedMemoryTrace through an explicit commitment gate.

Important boundary:
    candidate evaluation != memory commitment

The commitment operation:
- requires an eligible candidate
- requires explicit caller intent
- preserves provenance
- creates a new immutable trace
- does not mutate source trajectory, stability analysis, or candidate
- does not mutate existing ConditioningMemory objects

This module does NOT:
- silently learn
- rewrite existing memory
- mutate source objects
- select actions
- modify policy
- diagnose
- claim biological memory consolidation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.history.experience_consolidation import (
    ExperienceConsolidationCandidate,
    consolidation_is_policy_free,
)


SCHEMA_VERSION = "memory_commitment_v1"


class MemoryCommitmentError(ValueError):
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
        raise MemoryCommitmentError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise MemoryCommitmentError(
            f"{name} must be finite"
        )

    return x


def _validate_unit_interval(
    name: str,
    value: float,
) -> float:
    x = _validate_finite(
        name,
        value,
    )

    if not 0.0 <= x <= 1.0:
        raise MemoryCommitmentError(
            f"{name} must be within [0,1]"
        )

    return x


def _validate_vector(
    name: str,
    values: Sequence[float],
) -> tuple[float, ...]:
    vector = tuple(
        _validate_finite(
            f"{name}[{index}]",
            value,
        )
        for index, value in enumerate(values)
    )

    if not vector:
        raise MemoryCommitmentError(
            f"{name} must not be empty"
        )

    return vector


@dataclass(frozen=True, slots=True)
class MemoryCommitmentConfig:
    minimum_candidate_score: float = 0.65
    require_candidate_eligibility: bool = True
    require_explicit_commit: bool = True

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_candidate_score",
            _validate_unit_interval(
                "minimum_candidate_score",
                self.minimum_candidate_score,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class MemoryCommitmentGate:
    candidate_id: str

    candidate_eligible: bool
    candidate_score: float
    minimum_candidate_score: float

    explicit_commit_requested: bool

    eligibility_gate_passed: bool
    score_gate_passed: bool
    explicit_commit_gate_passed: bool

    commitment_allowed: bool
    reason: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _validate_id(
                "candidate_id",
                self.candidate_id,
            ),
        )

        for name in (
            "candidate_score",
            "minimum_candidate_score",
        ):
            object.__setattr__(
                self,
                name,
                _validate_unit_interval(
                    name,
                    getattr(self, name),
                ),
            )

        object.__setattr__(
            self,
            "reason",
            _validate_id(
                "reason",
                self.reason,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class CommittedMemoryTrace:
    trace_id: str

    source_candidate_id: str
    source_trajectory_id: str
    source_stability_analysis_id: str

    dominant_attractor_id: str
    representation_vector: tuple[float, ...]

    consolidation_score: float
    commitment_confidence: float

    commitment_applied: bool
    source_memory_mutated: bool
    existing_memory_rewritten: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "trace_id",
            "source_candidate_id",
            "source_trajectory_id",
            "source_stability_analysis_id",
            "dominant_attractor_id",
        ):
            object.__setattr__(
                self,
                name,
                _validate_id(
                    name,
                    getattr(self, name),
                ),
            )

        object.__setattr__(
            self,
            "representation_vector",
            _validate_vector(
                "representation_vector",
                self.representation_vector,
            ),
        )

        for name in (
            "consolidation_score",
            "commitment_confidence",
        ):
            object.__setattr__(
                self,
                name,
                _validate_unit_interval(
                    name,
                    getattr(self, name),
                ),
            )

        if self.commitment_applied is not True:
            raise MemoryCommitmentError(
                "committed trace must declare commitment_applied=True"
            )

        if self.source_memory_mutated is not False:
            raise MemoryCommitmentError(
                "committed trace must not mutate source memory"
            )

        if self.existing_memory_rewritten is not False:
            raise MemoryCommitmentError(
                "committed trace must not rewrite existing memory"
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def evaluate_memory_commitment_gate(
    *,
    candidate: ExperienceConsolidationCandidate,
    explicit_commit_requested: bool,
    config: MemoryCommitmentConfig | None = None,
) -> MemoryCommitmentGate:
    if not consolidation_is_policy_free(
        candidate
    ):
        raise MemoryCommitmentError(
            "consolidation candidate must be policy-free"
        )

    actual_config = (
        config
        if config is not None
        else MemoryCommitmentConfig()
    )

    eligibility_gate_passed = (
        candidate.eligible_for_consolidation
        if actual_config.require_candidate_eligibility
        else True
    )

    score_gate_passed = (
        candidate.consolidation_score
        >= actual_config.minimum_candidate_score
    )

    explicit_commit_gate_passed = (
        explicit_commit_requested
        if actual_config.require_explicit_commit
        else True
    )

    commitment_allowed = all(
        (
            eligibility_gate_passed,
            score_gate_passed,
            explicit_commit_gate_passed,
        )
    )

    if commitment_allowed:
        reason = "commitment_allowed"
    elif not eligibility_gate_passed:
        reason = "candidate_not_eligible"
    elif not score_gate_passed:
        reason = "candidate_score_below_threshold"
    else:
        reason = "explicit_commit_not_requested"

    return MemoryCommitmentGate(
        candidate_id=candidate.candidate_id,
        candidate_eligible=(
            candidate.eligible_for_consolidation
        ),
        candidate_score=(
            candidate.consolidation_score
        ),
        minimum_candidate_score=(
            actual_config.minimum_candidate_score
        ),
        explicit_commit_requested=(
            bool(
                explicit_commit_requested
            )
        ),
        eligibility_gate_passed=(
            eligibility_gate_passed
        ),
        score_gate_passed=(
            score_gate_passed
        ),
        explicit_commit_gate_passed=(
            explicit_commit_gate_passed
        ),
        commitment_allowed=(
            commitment_allowed
        ),
        reason=reason,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "commitment_gate_mode": "explicit",
            "memory_mutated": False,
            "learning_applied": False,
            "existing_memory_rewritten": False,
            "action_selected": False,
            "policy_modified": False,
            "diagnosis_generated": False,
            "biological_memory_claimed": False,
            "causal_truth_inferred": False,
        },
    )


def commit_memory_trace(
    *,
    trace_id: str,
    candidate: ExperienceConsolidationCandidate,
    explicit_commit_requested: bool,
    config: MemoryCommitmentConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CommittedMemoryTrace:
    gate = evaluate_memory_commitment_gate(
        candidate=candidate,
        explicit_commit_requested=explicit_commit_requested,
        config=config,
    )

    if not gate.commitment_allowed:
        raise MemoryCommitmentError(
            f"memory commitment denied: {gate.reason}"
        )

    commitment_confidence = min(
        1.0,
        max(
            0.0,
            (
                candidate.consolidation_score
                + candidate.evidence.stability_score
                + candidate.evidence.consistency_score
                + candidate.evidence.repetition_score
            )
            / 4.0,
        ),
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "memory_commitment_mode": "new_immutable_trace",
        "commitment_gate_reason": gate.reason,
        "source_candidate_id": candidate.candidate_id,
        "source_trajectory_id": candidate.source_trajectory_id,
        "source_stability_analysis_id": (
            candidate.source_stability_analysis_id
        ),
        "commitment_applied": True,
        "source_memory_mutated": False,
        "existing_memory_rewritten": False,
        "learning_applied": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_memory_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return CommittedMemoryTrace(
        trace_id=trace_id,
        source_candidate_id=candidate.candidate_id,
        source_trajectory_id=(
            candidate.source_trajectory_id
        ),
        source_stability_analysis_id=(
            candidate.source_stability_analysis_id
        ),
        dominant_attractor_id=(
            candidate.dominant_attractor_id
        ),
        representation_vector=(
            candidate.representation_vector
        ),
        consolidation_score=(
            candidate.consolidation_score
        ),
        commitment_confidence=(
            commitment_confidence
        ),
        commitment_applied=True,
        source_memory_mutated=False,
        existing_memory_rewritten=False,
        metadata=merged_metadata,
    )


def memory_commitment_is_policy_free(
    trace: CommittedMemoryTrace,
) -> bool:
    return (
        trace.metadata.get(
            "action_selected"
        )
        is False
        and trace.metadata.get(
            "policy_modified"
        )
        is False
        and trace.metadata.get(
            "source_memory_mutated"
        )
        is False
        and trace.metadata.get(
            "existing_memory_rewritten"
        )
        is False
    )


__all__ = [
    "CommittedMemoryTrace",
    "MemoryCommitmentConfig",
    "MemoryCommitmentError",
    "MemoryCommitmentGate",
    "SCHEMA_VERSION",
    "commit_memory_trace",
    "evaluate_memory_commitment_gate",
    "memory_commitment_is_policy_free",
]
