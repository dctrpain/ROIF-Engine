"""
ROIF Active Cascade Explorer

Purpose
-------
This module turns Active Probe Engine (APE) into part of cascade construction.

Canonical rule
--------------
For an incomplete or uncertain graph:

    Node_i
        ↓
    candidate outgoing relations
        ↓
    confidence sufficient?
        ├── yes -> ConfirmedEdge -> Node_i+1
        └── no  -> REQUIRES_PROBE
                        ↓
                      APE
                        ↓
                     Probe*
                        ↓
                    response
                        ↓
                GraphUpdateProposal
                        ↓
                  confirmed/rejected edge
                        ↓
                     Node_i+1

Invariant
---------
No unresolved cascade transition is silently promoted to a confirmed edge.

The module does not mutate ROIFSystem directly, execute probes, fabricate
observations, use hidden validation labels, or authorize graph updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence


class ActiveCascadeError(ValueError):
    """Raised when active-cascade state is inconsistent."""


class TransitionStatus(str, Enum):
    CONFIRMED = "confirmed"
    REQUIRES_PROBE = "requires_probe"
    REJECTED = "rejected"
    TERMINAL = "terminal"
    BLOCKED = "blocked"


class EdgeEvidenceState(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CandidateTransition:
    source_id: str
    target_id: str
    relation_id: str
    confidence: float
    uncertainty: float
    score: float = 0.0
    evidence_state: EdgeEvidenceState = EdgeEvidenceState.CANDIDATE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_id", "target_id", "relation_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ActiveCascadeError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

        for name in ("confidence", "uncertainty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ActiveCascadeError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)

        score = float(self.score)
        if not math.isfinite(score):
            raise ActiveCascadeError("score must be finite.")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "evidence_state", EdgeEvidenceState(self.evidence_state))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def edge(self) -> tuple[str, str]:
        return self.source_id, self.target_id


@dataclass(frozen=True, slots=True)
class TransitionPolicy:
    min_confidence: float = 0.90
    max_uncertainty: float = 0.10
    min_margin: float = 0.15
    require_unique_best: bool = True
    probe_when_multiple_viable: bool = True

    def __post_init__(self) -> None:
        for name in ("min_confidence", "max_uncertainty", "min_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ActiveCascadeError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    current_node_id: str
    status: TransitionStatus
    selected: CandidateTransition | None
    candidates: tuple[CandidateTransition, ...]
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        current = str(self.current_node_id).strip()
        if not current:
            raise ActiveCascadeError("current_node_id must not be empty.")
        object.__setattr__(self, "current_node_id", current)
        object.__setattr__(self, "status", TransitionStatus(self.status))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ConfirmedCascadeEdge:
    source_id: str
    target_id: str
    relation_id: str
    confirmation_source: str
    confidence: float
    probe_identifier: str | None = None
    graph_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_id", "target_id", "relation_id", "confirmation_source"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ActiveCascadeError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ActiveCascadeError("confidence must be finite and in [0, 1].")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ActiveCascadeState:
    start_node_id: str
    current_node_id: str
    confirmed_edges: tuple[ConfirmedCascadeEdge, ...] = ()
    rejected_relations: tuple[str, ...] = ()
    step_index: int = 0
    graph_version: str | None = None
    complete: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("start_node_id", "current_node_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ActiveCascadeError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ActiveCascadeError("step_index must be a nonnegative integer.")

        object.__setattr__(self, "confirmed_edges", tuple(self.confirmed_edges))
        object.__setattr__(self, "rejected_relations", tuple(self.rejected_relations))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def node_path(self) -> tuple[str, ...]:
        nodes = [self.start_node_id]
        for edge in self.confirmed_edges:
            if nodes[-1] != edge.source_id:
                raise ActiveCascadeError("confirmed_edges do not form a continuous cascade path.")
            nodes.append(edge.target_id)
        return tuple(nodes)


class CandidateProvider(Protocol):
    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> Sequence[CandidateTransition]: ...


class ProbePlannerBridge(Protocol):
    def plan_probe(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> Any: ...


class GraphUpdateResolver(Protocol):
    def resolve_update(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_update: Any,
        state: ActiveCascadeState,
    ) -> tuple[CandidateTransition | None, tuple[str, ...]]: ...


def rank_candidates(
    candidates: Iterable[CandidateTransition],
) -> tuple[CandidateTransition, ...]:
    """Deterministic candidate ranking."""
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                0 if item.evidence_state is EdgeEvidenceState.CONFIRMED else 1,
                -item.confidence,
                item.uncertainty,
                -item.score,
                item.relation_id,
            ),
        )
    )


def resolve_transition(
    current_node_id: str,
    candidates: Iterable[CandidateTransition],
    *,
    policy: TransitionPolicy | None = None,
) -> TransitionDecision:
    """Resolve one local cascade step or explicitly require Probe."""
    policy = policy or TransitionPolicy()
    ranked = rank_candidates(candidates)

    if not ranked:
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.TERMINAL,
            selected=None,
            candidates=(),
            reason="no_outgoing_candidates",
        )

    confirmed = tuple(
        candidate
        for candidate in ranked
        if candidate.evidence_state is EdgeEvidenceState.CONFIRMED
    )

    if len(confirmed) == 1:
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.CONFIRMED,
            selected=confirmed[0],
            candidates=ranked,
            reason="previously_confirmed_relation",
        )

    if len(confirmed) > 1:
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.REQUIRES_PROBE,
            selected=None,
            candidates=ranked,
            reason="multiple_confirmed_relations_require_disambiguation",
        )

    viable = tuple(
        candidate
        for candidate in ranked
        if candidate.evidence_state is not EdgeEvidenceState.REJECTED
    )

    if not viable:
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.BLOCKED,
            selected=None,
            candidates=ranked,
            reason="all_candidate_relations_rejected",
        )

    best = viable[0]
    second = viable[1] if len(viable) > 1 else None
    confidence_ok = best.confidence >= policy.min_confidence
    uncertainty_ok = best.uncertainty <= policy.max_uncertainty

    if second is None:
        if confidence_ok and uncertainty_ok:
            return TransitionDecision(
                current_node_id=current_node_id,
                status=TransitionStatus.CONFIRMED,
                selected=best,
                candidates=ranked,
                reason="single_candidate_with_sufficient_evidence",
            )
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.REQUIRES_PROBE,
            selected=None,
            candidates=ranked,
            reason="single_candidate_evidence_insufficient",
        )

    margin = best.confidence - second.confidence
    exact_tie = math.isclose(best.confidence, second.confidence, rel_tol=0.0, abs_tol=1e-12)

    if policy.require_unique_best and exact_tie:
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.REQUIRES_PROBE,
            selected=None,
            candidates=ranked,
            reason="best_candidates_tied",
            metadata={"confidence_margin": margin},
        )

    if confidence_ok and uncertainty_ok and margin >= policy.min_margin:
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.CONFIRMED,
            selected=best,
            candidates=ranked,
            reason="dominant_candidate_with_sufficient_evidence",
            metadata={"confidence_margin": margin},
        )

    if policy.probe_when_multiple_viable:
        return TransitionDecision(
            current_node_id=current_node_id,
            status=TransitionStatus.REQUIRES_PROBE,
            selected=None,
            candidates=ranked,
            reason="multiple_viable_candidates",
            metadata={"confidence_margin": margin},
        )

    return TransitionDecision(
        current_node_id=current_node_id,
        status=TransitionStatus.CONFIRMED,
        selected=best,
        candidates=ranked,
        reason="best_candidate_selected_by_policy",
        metadata={"confidence_margin": margin},
    )


class ActiveCascadeExplorer:
    """Orchestrates cascade reconstruction with Probe gating."""

    def __init__(
        self,
        *,
        candidate_provider: CandidateProvider,
        probe_planner: ProbePlannerBridge,
        graph_update_resolver: GraphUpdateResolver,
        policy: TransitionPolicy | None = None,
    ) -> None:
        self._candidate_provider = candidate_provider
        self._probe_planner = probe_planner
        self._graph_update_resolver = graph_update_resolver
        self._policy = policy or TransitionPolicy()

    @property
    def policy(self) -> TransitionPolicy:
        return self._policy

    def inspect_transition(
        self,
        state: ActiveCascadeState,
        *,
        graph_snapshot: Any,
    ) -> TransitionDecision:
        candidates = tuple(
            self._candidate_provider.candidates_for(
                state.current_node_id,
                graph_snapshot=graph_snapshot,
                state=state,
            )
        )
        rejected = set(state.rejected_relations)
        filtered = tuple(
            candidate
            for candidate in candidates
            if candidate.relation_id not in rejected
        )
        return resolve_transition(
            state.current_node_id,
            filtered,
            policy=self._policy,
        )

    def plan_required_probe(
        self,
        state: ActiveCascadeState,
        *,
        graph_snapshot: Any,
        decision: TransitionDecision | None = None,
    ) -> Any:
        decision = decision or self.inspect_transition(
            state,
            graph_snapshot=graph_snapshot,
        )
        if decision.status is not TransitionStatus.REQUIRES_PROBE:
            raise ActiveCascadeError(
                "Probe requested for a transition that does not require Probe."
            )
        return self._probe_planner.plan_probe(
            current_node_id=state.current_node_id,
            candidates=decision.candidates,
            graph_snapshot=graph_snapshot,
            state=state,
        )

    def accept_confirmed_transition(
        self,
        state: ActiveCascadeState,
        transition: CandidateTransition,
        *,
        confirmation_source: str,
        probe_identifier: str | None = None,
        graph_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActiveCascadeState:
        if transition.source_id != state.current_node_id:
            raise ActiveCascadeError(
                "transition source does not match current cascade node."
            )
        if transition.relation_id in state.rejected_relations:
            raise ActiveCascadeError(
                "cannot confirm a relation already rejected by evidence."
            )

        edge = ConfirmedCascadeEdge(
            source_id=transition.source_id,
            target_id=transition.target_id,
            relation_id=transition.relation_id,
            confirmation_source=confirmation_source,
            confidence=transition.confidence,
            probe_identifier=probe_identifier,
            graph_version=graph_version,
            metadata=metadata or {},
        )

        return ActiveCascadeState(
            start_node_id=state.start_node_id,
            current_node_id=transition.target_id,
            confirmed_edges=(*state.confirmed_edges, edge),
            rejected_relations=state.rejected_relations,
            step_index=state.step_index + 1,
            graph_version=graph_version if graph_version is not None else state.graph_version,
            complete=False,
            metadata=state.metadata,
        )

    def advance_if_confirmed(
        self,
        state: ActiveCascadeState,
        *,
        graph_snapshot: Any,
    ) -> tuple[ActiveCascadeState, TransitionDecision]:
        decision = self.inspect_transition(state, graph_snapshot=graph_snapshot)

        if decision.status is TransitionStatus.CONFIRMED:
            if decision.selected is None:
                raise ActiveCascadeError("confirmed decision has no selected transition.")
            next_state = self.accept_confirmed_transition(
                state,
                decision.selected,
                confirmation_source="passive_evidence",
                graph_version=state.graph_version,
                metadata={"decision_reason": decision.reason},
            )
            return next_state, decision

        if decision.status is TransitionStatus.TERMINAL:
            return (
                ActiveCascadeState(
                    start_node_id=state.start_node_id,
                    current_node_id=state.current_node_id,
                    confirmed_edges=state.confirmed_edges,
                    rejected_relations=state.rejected_relations,
                    step_index=state.step_index,
                    graph_version=state.graph_version,
                    complete=True,
                    metadata=state.metadata,
                ),
                decision,
            )

        return state, decision

    def apply_authorized_probe_update(
        self,
        state: ActiveCascadeState,
        *,
        graph_snapshot: Any,
        graph_update: Any,
        probe_identifier: str,
        graph_version: str | None = None,
    ) -> ActiveCascadeState:
        """Continue only after an externally authorized Probe update."""
        decision = self.inspect_transition(state, graph_snapshot=graph_snapshot)

        if decision.status is not TransitionStatus.REQUIRES_PROBE:
            raise ActiveCascadeError(
                "authorized Probe update supplied when transition does not require Probe."
            )

        confirmed, rejected = self._graph_update_resolver.resolve_update(
            current_node_id=state.current_node_id,
            candidates=decision.candidates,
            graph_update=graph_update,
            state=state,
        )

        rejected_relations = tuple(
            dict.fromkeys((*state.rejected_relations, *rejected))
        )

        intermediate = ActiveCascadeState(
            start_node_id=state.start_node_id,
            current_node_id=state.current_node_id,
            confirmed_edges=state.confirmed_edges,
            rejected_relations=rejected_relations,
            step_index=state.step_index,
            graph_version=graph_version if graph_version is not None else state.graph_version,
            complete=False,
            metadata=state.metadata,
        )

        if confirmed is None:
            return intermediate

        return self.accept_confirmed_transition(
            intermediate,
            confirmed,
            confirmation_source="active_probe",
            probe_identifier=probe_identifier,
            graph_version=graph_version,
            metadata={"probe_resolved_transition": True},
        )


def start_active_cascade(
    start_node_id: str,
    *,
    graph_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ActiveCascadeState:
    return ActiveCascadeState(
        start_node_id=start_node_id,
        current_node_id=start_node_id,
        confirmed_edges=(),
        rejected_relations=(),
        step_index=0,
        graph_version=graph_version,
        complete=False,
        metadata=metadata or {},
    )


__all__ = [
    "ActiveCascadeError",
    "ActiveCascadeExplorer",
    "ActiveCascadeState",
    "CandidateProvider",
    "CandidateTransition",
    "ConfirmedCascadeEdge",
    "EdgeEvidenceState",
    "GraphUpdateResolver",
    "ProbePlannerBridge",
    "TransitionDecision",
    "TransitionPolicy",
    "TransitionStatus",
    "rank_candidates",
    "resolve_transition",
    "start_active_cascade",
]
