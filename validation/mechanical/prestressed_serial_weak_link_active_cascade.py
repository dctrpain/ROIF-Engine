"""
ROIF Mechanical Validation
Scenario 03A — Pre-Stressed Serial Weak-Link Active Cascade Control

Purpose
-------
Scenario 03A is the negative/control counterpart to Scenario 02A.

02A contains a genuine branching ambiguity and therefore requires Active Probe
to decide which outgoing relation may enter the confirmed cascade history.

03A is deliberately serial:

    preload_disturbance
            |
            v
    proximal_transmission
            |
            v
      serial_weak_link
            |
            v
    distal_transmission
            |
            v
    terminal_displacement

Every non-terminal node has exactly one declared outgoing relation.

Control hypothesis
------------------
APE capability must not imply Probe use.

If the sole local relation already carries sufficient public structural
confidence, ActiveCascadeExplorer should advance it through passive evidence.
A Probe request in this scenario is treated as a validation failure.

This module does not:
- read evaluator-side expected D_origin / D_fast / D_root / Node* labels;
- fabricate a second candidate;
- execute a Probe;
- authorize a graph update;
- mutate the ROIF graph;
- infer Node*.

It validates only active-cascade traversal of an unambiguous serial graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from roif.active_cascade import (
    ActiveCascadeError,
    ActiveCascadeExplorer,
    ActiveCascadeState,
    CandidateTransition,
    EdgeEvidenceState,
    TransitionDecision,
    TransitionPolicy,
    TransitionStatus,
    start_active_cascade,
)

from validation.mechanical.prestressed_serial_weak_link_case import (
    DISTAL_CHANNEL,
    DISTAL_TO_TERMINAL,
    PRELOAD_CHANNEL,
    PRELOAD_TO_PROXIMAL,
    PROXIMAL_CHANNEL,
    PROXIMAL_TO_WEAK_LINK,
    TERMINAL_CHANNEL,
    WEAK_LINK_CHANNEL,
    WEAK_LINK_TO_DISTAL,
    PrestressedSerialWeakLinkValidationCase,
    build_prestressed_serial_weak_link_case,
)


# =============================================================================
# Configuration
# =============================================================================


SERIAL_EDGE_CONFIDENCE = 0.98
SERIAL_EDGE_UNCERTAINTY = 0.02

SERIAL_POLICY = TransitionPolicy(
    min_confidence=0.90,
    max_uncertainty=0.10,
    min_margin=0.15,
    require_unique_best=True,
    probe_when_multiple_viable=True,
)


class SerialActiveCascadeValidationError(ActiveCascadeError):
    """Raised when Scenario 03A violates its serial-control contract."""


# =============================================================================
# Immutable audit objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class SerialGraphSnapshot:
    """
    Minimal public graph snapshot for the known serial-control case.

    No hidden expected role labels are included. The snapshot records only
    graph identity/version and the fact that this is a known serial topology.
    """

    graph_id: str = "mechanical_03A_serial_active_cascade"
    version: str = "v0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class SerialCascadeStepAudit:
    step_index: int
    source_id: str
    target_id: str | None
    relation_id: str | None
    status: TransitionStatus
    reason: str
    confirmation_source: str | None
    probe_used: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            TransitionStatus(self.status),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class SerialActiveCascadeResult:
    state: ActiveCascadeState
    decisions: tuple[TransitionDecision, ...]
    audit_steps: tuple[SerialCascadeStepAudit, ...]
    probe_request_count: int
    graph_update_resolution_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decisions",
            tuple(self.decisions),
        )
        object.__setattr__(
            self,
            "audit_steps",
            tuple(self.audit_steps),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def node_path(self) -> tuple[str, ...]:
        return self.state.node_path

    @property
    def relation_path(self) -> tuple[str, ...]:
        return tuple(
            edge.relation_id
            for edge in self.state.confirmed_edges
        )

    @property
    def probe_used(self) -> bool:
        return self.probe_request_count > 0


# =============================================================================
# Public graph helpers
# =============================================================================


def build_case() -> PrestressedSerialWeakLinkValidationCase:
    return build_prestressed_serial_weak_link_case()


def build_serial_snapshot() -> SerialGraphSnapshot:
    return SerialGraphSnapshot(
        metadata={
            "validation_case": "mechanical_03A",
            "topology": "serial_weak_link",
            "branching": False,
            "feedback": False,
            "known_graph": True,
            "expected_role_labels_included": False,
        },
    )


def operator_lookup(
    case: PrestressedSerialWeakLinkValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            operator.operator_id: operator
            for operator in case.pathological_system.operators
        }
    )


def outgoing_operators(
    case: PrestressedSerialWeakLinkValidationCase,
    source_id: str,
) -> tuple[Any, ...]:
    return tuple(
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (source_id,)
    )


def candidate_from_operator(
    operator: Any,
) -> CandidateTransition:
    """
    Convert one known public mechanical relation into a cascade candidate.

    Confidence is structural confidence in the declared serial graph, not
    confidence imported from evaluator-side causal-role labels.
    """

    if len(operator.source_ids) != 1:
        raise SerialActiveCascadeValidationError(
            "03A serial operator must have exactly one source."
        )

    if len(operator.target_ids) != 1:
        raise SerialActiveCascadeValidationError(
            "03A serial operator must have exactly one target."
        )

    return CandidateTransition(
        source_id=operator.source_ids[0],
        target_id=operator.target_ids[0],
        relation_id=operator.operator_id,
        confidence=SERIAL_EDGE_CONFIDENCE,
        uncertainty=SERIAL_EDGE_UNCERTAINTY,
        score=abs(float(operator.gain)),
        evidence_state=EdgeEvidenceState.CANDIDATE,
        metadata={
            "validation_case": "mechanical_03A",
            "evidence_source": "declared_serial_mechanical_graph",
            "branching": False,
            "expected_role_label_used": False,
        },
    )


def serial_candidates(
    case: PrestressedSerialWeakLinkValidationCase,
    source_id: str,
) -> tuple[CandidateTransition, ...]:
    candidates = tuple(
        candidate_from_operator(operator)
        for operator in outgoing_operators(case, source_id)
    )

    if len(candidates) > 1:
        raise SerialActiveCascadeValidationError(
            "Scenario 03A unexpectedly contains branching."
        )

    return candidates


class Scenario03ASerialCandidateProvider:
    """Candidate provider backed only by the public 03A mechanical graph."""

    def __init__(
        self,
        case: PrestressedSerialWeakLinkValidationCase,
    ) -> None:
        self._case = case
        self.calls: list[str] = []

    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> tuple[CandidateTransition, ...]:
        self.calls.append(current_node_id)

        if state.current_node_id != current_node_id:
            raise SerialActiveCascadeValidationError(
                "candidate request does not match active-cascade state."
            )

        return serial_candidates(
            self._case,
            current_node_id,
        )


# =============================================================================
# Probe veto components
# =============================================================================


class ProbeMustNotRunPlanner:
    """
    ProbePlannerBridge used as a tripwire.

    Any invocation means the serial-control hypothesis failed.
    """

    def __init__(self) -> None:
        self.calls = 0

    def plan_probe(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> Any:
        self.calls += 1
        raise SerialActiveCascadeValidationError(
            "Probe was requested for a sufficiently known single-candidate "
            f"serial transition at {current_node_id!r}."
        )


class GraphUpdateMustNotRunResolver:
    """
    GraphUpdateResolver tripwire.

    No active graph-update resolution is expected in Scenario 03A.
    """

    def __init__(self) -> None:
        self.calls = 0

    def resolve_update(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_update: Any,
        state: ActiveCascadeState,
    ) -> tuple[CandidateTransition | None, tuple[str, ...]]:
        self.calls += 1
        raise SerialActiveCascadeValidationError(
            "Graph-update resolution was invoked in the passive serial "
            "control path."
        )


# =============================================================================
# Explorer construction
# =============================================================================


@dataclass(slots=True)
class SerialExplorerBundle:
    explorer: ActiveCascadeExplorer
    candidate_provider: Scenario03ASerialCandidateProvider
    probe_planner: ProbeMustNotRunPlanner
    graph_update_resolver: GraphUpdateMustNotRunResolver


def build_serial_explorer(
    case: PrestressedSerialWeakLinkValidationCase,
) -> SerialExplorerBundle:
    provider = Scenario03ASerialCandidateProvider(
        case
    )
    planner = ProbeMustNotRunPlanner()
    resolver = GraphUpdateMustNotRunResolver()

    explorer = ActiveCascadeExplorer(
        candidate_provider=provider,
        probe_planner=planner,
        graph_update_resolver=resolver,
        policy=SERIAL_POLICY,
    )

    return SerialExplorerBundle(
        explorer=explorer,
        candidate_provider=provider,
        probe_planner=planner,
        graph_update_resolver=resolver,
    )


# =============================================================================
# One-step and complete traversal
# =============================================================================


def inspect_serial_transition(
    case: PrestressedSerialWeakLinkValidationCase,
    current_node_id: str,
) -> TransitionDecision:
    bundle = build_serial_explorer(case)
    snapshot = build_serial_snapshot()

    state = start_active_cascade(
        current_node_id,
        graph_version=snapshot.version,
        metadata={
            "validation_case": "mechanical_03A",
            "expected_role_labels_used": False,
        },
    )

    return bundle.explorer.inspect_transition(
        state,
        graph_snapshot=snapshot,
    )


def _audit_decision(
    state_before: ActiveCascadeState,
    state_after: ActiveCascadeState,
    decision: TransitionDecision,
) -> SerialCascadeStepAudit:
    if decision.status is TransitionStatus.CONFIRMED:
        selected = decision.selected
        if selected is None:
            raise SerialActiveCascadeValidationError(
                "confirmed serial decision has no selected transition."
            )

        edge = state_after.confirmed_edges[-1]

        return SerialCascadeStepAudit(
            step_index=state_before.step_index,
            source_id=state_before.current_node_id,
            target_id=selected.target_id,
            relation_id=selected.relation_id,
            status=decision.status,
            reason=decision.reason,
            confirmation_source=edge.confirmation_source,
            probe_used=edge.probe_identifier is not None,
            metadata={
                "candidate_count": len(decision.candidates),
                "confidence": selected.confidence,
                "uncertainty": selected.uncertainty,
            },
        )

    if decision.status is TransitionStatus.TERMINAL:
        return SerialCascadeStepAudit(
            step_index=state_before.step_index,
            source_id=state_before.current_node_id,
            target_id=None,
            relation_id=None,
            status=decision.status,
            reason=decision.reason,
            confirmation_source=None,
            probe_used=False,
            metadata={
                "candidate_count": 0,
            },
        )

    return SerialCascadeStepAudit(
        step_index=state_before.step_index,
        source_id=state_before.current_node_id,
        target_id=None,
        relation_id=None,
        status=decision.status,
        reason=decision.reason,
        confirmation_source=None,
        probe_used=False,
        metadata={
            "candidate_count": len(decision.candidates),
        },
    )


def run_serial_active_cascade(
    case: PrestressedSerialWeakLinkValidationCase | None = None,
) -> SerialActiveCascadeResult:
    """
    Traverse the complete 03A serial chain using ActiveCascadeExplorer.

    A valid run contains four passive confirmations followed by one terminal
    decision. Any REQUIRES_PROBE, BLOCKED, or REJECTED status is a control-case
    failure.
    """

    case = case or build_case()
    snapshot = build_serial_snapshot()
    bundle = build_serial_explorer(case)

    state = start_active_cascade(
        PRELOAD_CHANNEL,
        graph_version=snapshot.version,
        metadata={
            "validation_case": "mechanical_03A",
            "mode": "serial_active_cascade_control",
            "expected_role_labels_used": False,
        },
    )

    decisions: list[TransitionDecision] = []
    audit_steps: list[SerialCascadeStepAudit] = []

    # Four serial edges + one terminal inspection. The explicit cap prevents
    # accidental cycles from being hidden by an infinite traversal.
    for _ in range(len(case.channel_ids) + 1):
        before = state

        state, decision = bundle.explorer.advance_if_confirmed(
            state,
            graph_snapshot=snapshot,
        )

        decisions.append(decision)
        audit_steps.append(
            _audit_decision(
                before,
                state,
                decision,
            )
        )

        if decision.status is TransitionStatus.TERMINAL:
            if not state.complete:
                raise SerialActiveCascadeValidationError(
                    "terminal serial state was not marked complete."
                )
            break

        if decision.status is TransitionStatus.REQUIRES_PROBE:
            raise SerialActiveCascadeValidationError(
                "03A unexpectedly requires Probe despite a sufficiently "
                "known single outgoing relation."
            )

        if decision.status in {
            TransitionStatus.BLOCKED,
            TransitionStatus.REJECTED,
        }:
            raise SerialActiveCascadeValidationError(
                f"03A serial traversal stopped with {decision.status.value!r}."
            )

        if decision.status is not TransitionStatus.CONFIRMED:
            raise SerialActiveCascadeValidationError(
                f"unsupported 03A transition status: {decision.status.value!r}."
            )
    else:
        raise SerialActiveCascadeValidationError(
            "03A serial traversal exceeded the expected path length."
        )

    expected_node_path = (
        PRELOAD_CHANNEL,
        PROXIMAL_CHANNEL,
        WEAK_LINK_CHANNEL,
        DISTAL_CHANNEL,
        TERMINAL_CHANNEL,
    )

    expected_relation_path = (
        PRELOAD_TO_PROXIMAL,
        PROXIMAL_TO_WEAK_LINK,
        WEAK_LINK_TO_DISTAL,
        DISTAL_TO_TERMINAL,
    )

    if state.node_path != expected_node_path:
        raise SerialActiveCascadeValidationError(
            "03A reconstructed node path differs from declared serial graph."
        )

    actual_relations = tuple(
        edge.relation_id
        for edge in state.confirmed_edges
    )

    if actual_relations != expected_relation_path:
        raise SerialActiveCascadeValidationError(
            "03A reconstructed relation path differs from declared serial graph."
        )

    if bundle.probe_planner.calls != 0:
        raise SerialActiveCascadeValidationError(
            "Probe planner was invoked in Scenario 03A."
        )

    if bundle.graph_update_resolver.calls != 0:
        raise SerialActiveCascadeValidationError(
            "Graph-update resolver was invoked in Scenario 03A."
        )

    return SerialActiveCascadeResult(
        state=state,
        decisions=tuple(decisions),
        audit_steps=tuple(audit_steps),
        probe_request_count=bundle.probe_planner.calls,
        graph_update_resolution_count=bundle.graph_update_resolver.calls,
        metadata={
            "validation_case": "mechanical_03A",
            "control_question": (
                "does_not_probe_sufficiently_known_single_candidate_edges"
            ),
            "expected_role_labels_used": False,
            "branching": False,
            "feedback": False,
            "passive_confirmed_edge_count": len(state.confirmed_edges),
        },
    )


__all__ = [
    "GraphUpdateMustNotRunResolver",
    "ProbeMustNotRunPlanner",
    "SERIAL_EDGE_CONFIDENCE",
    "SERIAL_EDGE_UNCERTAINTY",
    "SERIAL_POLICY",
    "Scenario03ASerialCandidateProvider",
    "SerialActiveCascadeResult",
    "SerialActiveCascadeValidationError",
    "SerialCascadeStepAudit",
    "SerialExplorerBundle",
    "SerialGraphSnapshot",
    "build_case",
    "build_serial_explorer",
    "build_serial_snapshot",
    "candidate_from_operator",
    "inspect_serial_transition",
    "operator_lookup",
    "outgoing_operators",
    "run_serial_active_cascade",
    "serial_candidates",
]
