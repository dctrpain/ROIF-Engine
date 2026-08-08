"""
Tests for roif.active_cascade

The suite validates the orchestration rule:

    No unresolved cascade transition is silently promoted
    to a confirmed cascade edge.

These are unit tests for active cascade construction. They do not execute
real probes and they do not mutate a ROIF graph. Minimal test doubles are
used for candidate discovery, Probe planning, and authorized graph-update
resolution.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_cascade import (
    ActiveCascadeError,
    ActiveCascadeExplorer,
    ActiveCascadeState,
    CandidateTransition,
    ConfirmedCascadeEdge,
    EdgeEvidenceState,
    TransitionDecision,
    TransitionPolicy,
    TransitionStatus,
    rank_candidates,
    resolve_transition,
    start_active_cascade,
)


# =============================================================================
# Test doubles
# =============================================================================


class StaticCandidateProvider:
    def __init__(
        self,
        by_node: dict[str, tuple[CandidateTransition, ...]],
    ) -> None:
        self.by_node = by_node
        self.calls: list[str] = []

    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot,
        state: ActiveCascadeState,
    ):
        self.calls.append(current_node_id)
        return self.by_node.get(current_node_id, ())


class RecordingProbePlanner:
    def __init__(self) -> None:
        self.calls = []

    def plan_probe(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_snapshot,
        state: ActiveCascadeState,
    ):
        payload = {
            "probe_id": f"probe:{current_node_id}",
            "candidate_ids": tuple(
                candidate.relation_id
                for candidate in candidates
            ),
        }
        self.calls.append(payload)
        return payload


class StaticGraphUpdateResolver:
    def __init__(
        self,
        *,
        confirmed: CandidateTransition | None = None,
        rejected: tuple[str, ...] = (),
    ) -> None:
        self.confirmed = confirmed
        self.rejected = rejected
        self.calls = 0

    def resolve_update(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_update,
        state: ActiveCascadeState,
    ):
        self.calls += 1
        return self.confirmed, self.rejected


def candidate(
    target: str,
    *,
    relation: str | None = None,
    confidence: float = 0.50,
    uncertainty: float = 0.50,
    score: float = 0.0,
    evidence: EdgeEvidenceState = EdgeEvidenceState.CANDIDATE,
    source: str = "A",
) -> CandidateTransition:
    return CandidateTransition(
        source_id=source,
        target_id=target,
        relation_id=relation or f"{source}_to_{target}",
        confidence=confidence,
        uncertainty=uncertainty,
        score=score,
        evidence_state=evidence,
    )


def explorer_for(
    candidates: tuple[CandidateTransition, ...],
    *,
    resolver: StaticGraphUpdateResolver | None = None,
    policy: TransitionPolicy | None = None,
):
    provider = StaticCandidateProvider(
        {"A": candidates}
    )
    planner = RecordingProbePlanner()
    resolver = resolver or StaticGraphUpdateResolver()

    explorer = ActiveCascadeExplorer(
        candidate_provider=provider,
        probe_planner=planner,
        graph_update_resolver=resolver,
        policy=policy,
    )

    return explorer, provider, planner, resolver


# =============================================================================
# CandidateTransition
# =============================================================================


def test_candidate_transition_contract() -> None:
    item = candidate(
        "B",
        confidence=0.8,
        uncertainty=0.2,
        score=3.0,
    )

    assert item.source_id == "A"
    assert item.target_id == "B"
    assert item.relation_id == "A_to_B"
    assert item.edge == ("A", "B")
    assert item.confidence == pytest.approx(0.8)
    assert item.uncertainty == pytest.approx(0.2)
    assert item.score == pytest.approx(3.0)


def test_candidate_transition_is_frozen() -> None:
    item = candidate("B")

    with pytest.raises(FrozenInstanceError):
        item.score = 99.0  # type: ignore[misc]


def test_candidate_metadata_is_read_only() -> None:
    item = CandidateTransition(
        source_id="A",
        target_id="B",
        relation_id="A_to_B",
        confidence=0.5,
        uncertainty=0.5,
        metadata={"x": 1},
    )

    assert isinstance(item.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        item.metadata["x"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", ""),
        ("target_id", " "),
        ("relation_id", ""),
    ],
)
def test_candidate_rejects_empty_identifiers(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "source_id": "A",
        "target_id": "B",
        "relation_id": "A_to_B",
        "confidence": 0.5,
        "uncertainty": 0.5,
    }
    kwargs[field] = value

    with pytest.raises(ActiveCascadeError):
        CandidateTransition(**kwargs)


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01],
)
def test_candidate_rejects_invalid_confidence(
    value: float,
) -> None:
    with pytest.raises(ActiveCascadeError):
        candidate("B", confidence=value)


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01],
)
def test_candidate_rejects_invalid_uncertainty(
    value: float,
) -> None:
    with pytest.raises(ActiveCascadeError):
        candidate("B", uncertainty=value)


# =============================================================================
# TransitionPolicy
# =============================================================================


def test_default_policy_is_conservative() -> None:
    policy = TransitionPolicy()

    assert policy.min_confidence == pytest.approx(0.90)
    assert policy.max_uncertainty == pytest.approx(0.10)
    assert policy.min_margin == pytest.approx(0.15)
    assert policy.require_unique_best is True
    assert policy.probe_when_multiple_viable is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_confidence", -0.1),
        ("min_confidence", 1.1),
        ("max_uncertainty", -0.1),
        ("max_uncertainty", 1.1),
        ("min_margin", -0.1),
        ("min_margin", 1.1),
    ],
)
def test_policy_rejects_values_outside_unit_interval(
    field: str,
    value: float,
) -> None:
    kwargs = {
        "min_confidence": 0.9,
        "max_uncertainty": 0.1,
        "min_margin": 0.15,
    }
    kwargs[field] = value

    with pytest.raises(ActiveCascadeError):
        TransitionPolicy(**kwargs)


# =============================================================================
# rank_candidates
# =============================================================================


def test_rank_candidates_prefers_confirmed_evidence() -> None:
    uncertain = candidate(
        "B",
        confidence=0.99,
        uncertainty=0.01,
        score=100.0,
    )
    confirmed = candidate(
        "C",
        confidence=0.20,
        uncertainty=0.80,
        score=-10.0,
        evidence=EdgeEvidenceState.CONFIRMED,
    )

    ranked = rank_candidates(
        (uncertain, confirmed)
    )

    assert ranked[0] is confirmed


def test_rank_candidates_prefers_higher_confidence() -> None:
    lower = candidate(
        "B",
        confidence=0.7,
        uncertainty=0.1,
    )
    higher = candidate(
        "C",
        confidence=0.8,
        uncertainty=0.5,
    )

    ranked = rank_candidates(
        (lower, higher)
    )

    assert ranked[0] is higher


def test_rank_candidates_uses_lower_uncertainty_after_confidence_tie() -> None:
    uncertain = candidate(
        "B",
        confidence=0.8,
        uncertainty=0.4,
    )
    clearer = candidate(
        "C",
        confidence=0.8,
        uncertainty=0.2,
    )

    assert rank_candidates(
        (uncertain, clearer)
    )[0] is clearer


def test_rank_candidates_uses_score_after_confidence_and_uncertainty_tie() -> None:
    lower_score = candidate(
        "B",
        confidence=0.8,
        uncertainty=0.2,
        score=1.0,
    )
    higher_score = candidate(
        "C",
        confidence=0.8,
        uncertainty=0.2,
        score=9.0,
    )

    assert rank_candidates(
        (lower_score, higher_score)
    )[0] is higher_score


def test_rank_candidates_is_deterministic() -> None:
    items = (
        candidate("C", confidence=0.7),
        candidate("B", confidence=0.7),
    )

    assert (
        rank_candidates(items)
        == rank_candidates(items)
    )


# =============================================================================
# resolve_transition — canonical gating
# =============================================================================


def test_no_candidates_means_terminal() -> None:
    decision = resolve_transition(
        "A",
        (),
    )

    assert decision.status is TransitionStatus.TERMINAL
    assert decision.selected is None
    assert decision.reason == "no_outgoing_candidates"


def test_one_previously_confirmed_edge_is_accepted() -> None:
    edge = candidate(
        "B",
        evidence=EdgeEvidenceState.CONFIRMED,
        confidence=0.5,
        uncertainty=0.5,
    )

    decision = resolve_transition(
        "A",
        (edge,),
    )

    assert decision.status is TransitionStatus.CONFIRMED
    assert decision.selected is edge
    assert decision.reason == "previously_confirmed_relation"


def test_multiple_confirmed_edges_require_disambiguation() -> None:
    first = candidate(
        "B",
        relation="r1",
        evidence=EdgeEvidenceState.CONFIRMED,
    )
    second = candidate(
        "C",
        relation="r2",
        evidence=EdgeEvidenceState.CONFIRMED,
    )

    decision = resolve_transition(
        "A",
        (first, second),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert decision.selected is None


def test_all_rejected_candidates_block_transition() -> None:
    decision = resolve_transition(
        "A",
        (
            candidate(
                "B",
                evidence=EdgeEvidenceState.REJECTED,
            ),
            candidate(
                "C",
                evidence=EdgeEvidenceState.REJECTED,
            ),
        ),
    )

    assert decision.status is TransitionStatus.BLOCKED
    assert decision.selected is None


def test_single_candidate_with_strong_evidence_is_confirmed() -> None:
    edge = candidate(
        "B",
        confidence=0.95,
        uncertainty=0.05,
    )

    decision = resolve_transition(
        "A",
        (edge,),
    )

    assert decision.status is TransitionStatus.CONFIRMED
    assert decision.selected is edge


def test_single_candidate_with_low_confidence_requires_probe() -> None:
    edge = candidate(
        "B",
        confidence=0.60,
        uncertainty=0.05,
        score=999999.0,
    )

    decision = resolve_transition(
        "A",
        (edge,),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert decision.selected is None


def test_single_candidate_with_high_uncertainty_requires_probe() -> None:
    edge = candidate(
        "B",
        confidence=0.99,
        uncertainty=0.40,
        score=999999.0,
    )

    decision = resolve_transition(
        "A",
        (edge,),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE


def test_two_tied_candidates_require_probe_even_if_scores_differ() -> None:
    """
    Core invariant:
    passive score is not allowed to resolve causal ambiguity.
    """

    high_score = candidate(
        "B",
        confidence=0.95,
        uncertainty=0.05,
        score=1_000_000.0,
    )
    low_score = candidate(
        "C",
        confidence=0.95,
        uncertainty=0.05,
        score=0.0,
    )

    decision = resolve_transition(
        "A",
        (high_score, low_score),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert decision.selected is None
    assert decision.reason == "best_candidates_tied"


def test_two_close_candidates_require_probe() -> None:
    first = candidate(
        "B",
        confidence=0.95,
        uncertainty=0.05,
        score=100.0,
    )
    second = candidate(
        "C",
        confidence=0.88,
        uncertainty=0.05,
        score=1.0,
    )

    decision = resolve_transition(
        "A",
        (first, second),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert decision.selected is None


def test_dominant_candidate_can_be_confirmed_without_probe() -> None:
    first = candidate(
        "B",
        confidence=0.97,
        uncertainty=0.03,
    )
    second = candidate(
        "C",
        confidence=0.70,
        uncertainty=0.10,
    )

    decision = resolve_transition(
        "A",
        (first, second),
    )

    assert decision.status is TransitionStatus.CONFIRMED
    assert decision.selected is first
    assert decision.reason == "dominant_candidate_with_sufficient_evidence"


def test_high_score_cannot_compensate_for_insufficient_confidence_margin() -> None:
    first = candidate(
        "B",
        confidence=0.91,
        uncertainty=0.05,
        score=1e9,
    )
    second = candidate(
        "C",
        confidence=0.90,
        uncertainty=0.05,
        score=-1e9,
    )

    decision = resolve_transition(
        "A",
        (first, second),
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE


def test_rejected_candidate_does_not_compete_with_viable_candidate() -> None:
    viable = candidate(
        "B",
        confidence=0.95,
        uncertainty=0.05,
    )
    rejected = candidate(
        "C",
        confidence=1.0,
        uncertainty=0.0,
        evidence=EdgeEvidenceState.REJECTED,
    )

    decision = resolve_transition(
        "A",
        (rejected, viable),
    )

    assert decision.status is TransitionStatus.CONFIRMED
    assert decision.selected is viable


# =============================================================================
# ActiveCascadeState
# =============================================================================


def test_start_active_cascade_contract() -> None:
    state = start_active_cascade(
        "A",
        graph_version="v0",
        metadata={"case": "test"},
    )

    assert state.start_node_id == "A"
    assert state.current_node_id == "A"
    assert state.confirmed_edges == ()
    assert state.rejected_relations == ()
    assert state.step_index == 0
    assert state.graph_version == "v0"
    assert state.complete is False
    assert state.node_path == ("A",)


def test_state_metadata_is_read_only() -> None:
    state = start_active_cascade(
        "A",
        metadata={"case": "test"},
    )

    assert isinstance(state.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        state.metadata["case"] = "changed"  # type: ignore[index]


def test_state_is_frozen() -> None:
    state = start_active_cascade("A")

    with pytest.raises(FrozenInstanceError):
        state.current_node_id = "B"  # type: ignore[misc]


def test_node_path_reconstructs_confirmed_chain() -> None:
    edge1 = ConfirmedCascadeEdge(
        source_id="A",
        target_id="B",
        relation_id="A_to_B",
        confirmation_source="probe",
        confidence=1.0,
    )
    edge2 = ConfirmedCascadeEdge(
        source_id="B",
        target_id="C",
        relation_id="B_to_C",
        confirmation_source="probe",
        confidence=1.0,
    )

    state = ActiveCascadeState(
        start_node_id="A",
        current_node_id="C",
        confirmed_edges=(edge1, edge2),
        step_index=2,
    )

    assert state.node_path == (
        "A",
        "B",
        "C",
    )


def test_node_path_rejects_discontinuous_history() -> None:
    edge = ConfirmedCascadeEdge(
        source_id="X",
        target_id="B",
        relation_id="X_to_B",
        confirmation_source="probe",
        confidence=1.0,
    )

    state = ActiveCascadeState(
        start_node_id="A",
        current_node_id="B",
        confirmed_edges=(edge,),
        step_index=1,
    )

    with pytest.raises(ActiveCascadeError):
        _ = state.node_path


# =============================================================================
# ActiveCascadeExplorer inspection / planning
# =============================================================================


def test_explorer_inspection_uses_candidate_provider() -> None:
    edge = candidate(
        "B",
        confidence=0.95,
        uncertainty=0.05,
    )
    explorer, provider, _, _ = explorer_for(
        (edge,)
    )

    decision = explorer.inspect_transition(
        start_active_cascade("A"),
        graph_snapshot={"version": 0},
    )

    assert provider.calls == ["A"]
    assert decision.status is TransitionStatus.CONFIRMED


def test_explorer_filters_relations_rejected_by_prior_state() -> None:
    first = candidate(
        "B",
        relation="r1",
        confidence=0.99,
        uncertainty=0.01,
    )
    second = candidate(
        "C",
        relation="r2",
        confidence=0.95,
        uncertainty=0.05,
    )

    explorer, _, _, _ = explorer_for(
        (first, second)
    )

    state = ActiveCascadeState(
        start_node_id="A",
        current_node_id="A",
        rejected_relations=("r1",),
    )

    decision = explorer.inspect_transition(
        state,
        graph_snapshot={},
    )

    assert decision.status is TransitionStatus.CONFIRMED
    assert decision.selected is second


def test_plan_required_probe_calls_probe_planner() -> None:
    first = candidate(
        "B",
        confidence=0.8,
        uncertainty=0.2,
    )
    second = candidate(
        "C",
        confidence=0.8,
        uncertainty=0.2,
    )

    explorer, _, planner, _ = explorer_for(
        (first, second)
    )

    state = start_active_cascade("A")

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot={},
    )

    assert plan["probe_id"] == "probe:A"
    assert set(plan["candidate_ids"]) == {
        "A_to_B",
        "A_to_C",
    }
    assert len(planner.calls) == 1


def test_probe_cannot_be_planned_for_confirmed_transition() -> None:
    edge = candidate(
        "B",
        confidence=0.99,
        uncertainty=0.01,
    )

    explorer, _, _, _ = explorer_for(
        (edge,)
    )

    with pytest.raises(
        ActiveCascadeError,
        match="does not require Probe",
    ):
        explorer.plan_required_probe(
            start_active_cascade("A"),
            graph_snapshot={},
        )


# =============================================================================
# Passive advancement
# =============================================================================


def test_advance_if_confirmed_moves_to_next_node() -> None:
    edge = candidate(
        "B",
        confidence=0.99,
        uncertainty=0.01,
    )

    explorer, _, _, _ = explorer_for(
        (edge,)
    )

    next_state, decision = explorer.advance_if_confirmed(
        start_active_cascade("A"),
        graph_snapshot={},
    )

    assert decision.status is TransitionStatus.CONFIRMED
    assert next_state.current_node_id == "B"
    assert next_state.step_index == 1
    assert next_state.node_path == ("A", "B")
    assert len(next_state.confirmed_edges) == 1
    assert (
        next_state.confirmed_edges[0].confirmation_source
        == "passive_evidence"
    )


def test_unresolved_transition_does_not_advance_state() -> None:
    """
    Main active-cascade invariant.
    """

    first = candidate(
        "B",
        confidence=0.80,
        uncertainty=0.20,
        score=10_000.0,
    )
    second = candidate(
        "C",
        confidence=0.79,
        uncertainty=0.20,
        score=-10_000.0,
    )

    explorer, _, _, _ = explorer_for(
        (first, second)
    )

    state = start_active_cascade("A")

    next_state, decision = explorer.advance_if_confirmed(
        state,
        graph_snapshot={},
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert next_state == state
    assert next_state.current_node_id == "A"
    assert next_state.confirmed_edges == ()
    assert next_state.step_index == 0


def test_terminal_transition_marks_state_complete() -> None:
    explorer, _, _, _ = explorer_for(())

    state, decision = explorer.advance_if_confirmed(
        start_active_cascade("A"),
        graph_snapshot={},
    )

    assert decision.status is TransitionStatus.TERMINAL
    assert state.complete is True
    assert state.current_node_id == "A"


def test_accept_confirmed_transition_rejects_wrong_source() -> None:
    explorer, _, _, _ = explorer_for(())

    wrong = candidate(
        "C",
        source="B",
        relation="B_to_C",
    )

    with pytest.raises(
        ActiveCascadeError,
        match="source does not match",
    ):
        explorer.accept_confirmed_transition(
            start_active_cascade("A"),
            wrong,
            confirmation_source="test",
        )


def test_accept_confirmed_transition_cannot_revive_rejected_relation() -> None:
    explorer, _, _, _ = explorer_for(())

    edge = candidate(
        "B",
        relation="r1",
    )

    state = ActiveCascadeState(
        start_node_id="A",
        current_node_id="A",
        rejected_relations=("r1",),
    )

    with pytest.raises(
        ActiveCascadeError,
        match="already rejected",
    ):
        explorer.accept_confirmed_transition(
            state,
            edge,
            confirmation_source="test",
        )


# =============================================================================
# Authorized Probe update
# =============================================================================


def test_authorized_probe_update_can_confirm_next_edge() -> None:
    first = candidate(
        "B",
        relation="r1",
        confidence=0.7,
        uncertainty=0.3,
    )
    second = candidate(
        "C",
        relation="r2",
        confidence=0.7,
        uncertainty=0.3,
    )

    resolver = StaticGraphUpdateResolver(
        confirmed=first,
        rejected=("r2",),
    )

    explorer, _, _, resolver = explorer_for(
        (first, second),
        resolver=resolver,
    )

    state = explorer.apply_authorized_probe_update(
        start_active_cascade("A"),
        graph_snapshot={},
        graph_update={"authorized": True},
        probe_identifier="probe_001",
        graph_version="v1",
    )

    assert resolver.calls == 1
    assert state.current_node_id == "B"
    assert state.step_index == 1
    assert state.rejected_relations == ("r2",)
    assert state.graph_version == "v1"

    edge = state.confirmed_edges[0]

    assert edge.relation_id == "r1"
    assert edge.confirmation_source == "active_probe"
    assert edge.probe_identifier == "probe_001"


def test_probe_update_may_only_reject_and_remain_on_same_node() -> None:
    first = candidate(
        "B",
        relation="r1",
        confidence=0.6,
        uncertainty=0.4,
    )
    second = candidate(
        "C",
        relation="r2",
        confidence=0.6,
        uncertainty=0.4,
    )

    resolver = StaticGraphUpdateResolver(
        confirmed=None,
        rejected=("r1",),
    )

    explorer, _, _, _ = explorer_for(
        (first, second),
        resolver=resolver,
    )

    state = explorer.apply_authorized_probe_update(
        start_active_cascade("A"),
        graph_snapshot={},
        graph_update={"authorized": True},
        probe_identifier="probe_001",
        graph_version="v1",
    )

    assert state.current_node_id == "A"
    assert state.step_index == 0
    assert state.confirmed_edges == ()
    assert state.rejected_relations == ("r1",)
    assert state.graph_version == "v1"


def test_probe_update_rejections_are_deduplicated() -> None:
    first = candidate(
        "B",
        relation="r1",
        confidence=0.6,
        uncertainty=0.4,
    )
    second = candidate(
        "C",
        relation="r2",
        confidence=0.6,
        uncertainty=0.4,
    )

    resolver = StaticGraphUpdateResolver(
        confirmed=None,
        rejected=("r1", "r1", "r2"),
    )

    explorer, _, _, _ = explorer_for(
        (first, second),
        resolver=resolver,
    )

    state = ActiveCascadeState(
        start_node_id="A",
        current_node_id="A",
        rejected_relations=("r1",),
    )

    # r1 is already filtered before inspection, leaving r2 as a single
    # low-confidence candidate, so it still requires Probe.
    updated = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot={},
        graph_update={"authorized": True},
        probe_identifier="probe_001",
    )

    assert updated.rejected_relations == (
        "r1",
        "r2",
    )


def test_authorized_update_is_rejected_when_probe_not_required() -> None:
    edge = candidate(
        "B",
        confidence=0.99,
        uncertainty=0.01,
    )

    explorer, _, _, _ = explorer_for(
        (edge,)
    )

    with pytest.raises(
        ActiveCascadeError,
        match="does not require Probe",
    ):
        explorer.apply_authorized_probe_update(
            start_active_cascade("A"),
            graph_snapshot={},
            graph_update={"authorized": True},
            probe_identifier="probe_001",
        )


# =============================================================================
# Multi-step reconstruction
# =============================================================================


def test_two_step_cascade_can_mix_passive_and_active_confirmation() -> None:
    a_to_b = candidate(
        "B",
        relation="A_to_B",
        confidence=0.99,
        uncertainty=0.01,
        source="A",
    )

    b_to_c = candidate(
        "C",
        relation="B_to_C",
        confidence=0.70,
        uncertainty=0.30,
        source="B",
    )
    b_to_d = candidate(
        "D",
        relation="B_to_D",
        confidence=0.69,
        uncertainty=0.31,
        source="B",
    )

    provider = StaticCandidateProvider(
        {
            "A": (a_to_b,),
            "B": (b_to_c, b_to_d),
            "C": (),
        }
    )
    planner = RecordingProbePlanner()
    resolver = StaticGraphUpdateResolver(
        confirmed=b_to_c,
        rejected=("B_to_D",),
    )

    explorer = ActiveCascadeExplorer(
        candidate_provider=provider,
        probe_planner=planner,
        graph_update_resolver=resolver,
    )

    state = start_active_cascade(
        "A",
        graph_version="v0",
    )

    state, first_decision = explorer.advance_if_confirmed(
        state,
        graph_snapshot={},
    )

    assert first_decision.status is TransitionStatus.CONFIRMED
    assert state.current_node_id == "B"

    second_decision = explorer.inspect_transition(
        state,
        graph_snapshot={},
    )

    assert second_decision.status is TransitionStatus.REQUIRES_PROBE

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot={},
        decision=second_decision,
    )

    assert plan["probe_id"] == "probe:B"

    state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot={},
        graph_update={"authorized": True},
        probe_identifier="probe:B",
        graph_version="v1",
    )

    assert state.current_node_id == "C"
    assert state.step_index == 2
    assert state.node_path == (
        "A",
        "B",
        "C",
    )
    assert tuple(
        edge.confirmation_source
        for edge in state.confirmed_edges
    ) == (
        "passive_evidence",
        "active_probe",
    )

    state, terminal = explorer.advance_if_confirmed(
        state,
        graph_snapshot={},
    )

    assert terminal.status is TransitionStatus.TERMINAL
    assert state.complete is True


def test_unresolved_high_score_branch_never_enters_confirmed_history() -> None:
    """
    Regression guard for the architectural rule that motivated this module.
    """

    tempting = candidate(
        "B",
        confidence=0.60,
        uncertainty=0.40,
        score=1e12,
    )
    alternative = candidate(
        "C",
        confidence=0.59,
        uncertainty=0.41,
        score=0.0,
    )

    explorer, _, _, _ = explorer_for(
        (tempting, alternative)
    )

    state = start_active_cascade("A")

    next_state, decision = explorer.advance_if_confirmed(
        state,
        graph_snapshot={},
    )

    assert decision.status is TransitionStatus.REQUIRES_PROBE
    assert next_state.confirmed_edges == ()
    assert "A_to_B" not in {
        edge.relation_id
        for edge in next_state.confirmed_edges
    }


def test_active_cascade_unit_pipeline_is_deterministic() -> None:
    first = candidate(
        "B",
        confidence=0.75,
        uncertainty=0.25,
        score=2.0,
    )
    second = candidate(
        "C",
        confidence=0.74,
        uncertainty=0.26,
        score=1.0,
    )

    left = resolve_transition(
        "A",
        (first, second),
    )
    right = resolve_transition(
        "A",
        (first, second),
    )

    assert left == right
    assert left.status is TransitionStatus.REQUIRES_PROBE
