"""
ROIF Mechanical Validation
Scenario 03A — Serial Weak-Link Active Cascade Control Tests

These tests validate the control property established by Scenario 03A:

    APE capability does not imply Probe use.

For a sufficiently known serial graph with one outgoing candidate per
non-terminal node, ActiveCascadeExplorer should advance by passive evidence.

Expected path:

    preload_disturbance
        -> proximal_transmission
        -> serial_weak_link
        -> distal_transmission
        -> terminal_displacement
        -> TERMINAL

No evaluator-side D_origin / D_fast / D_root / Node* label is required for
this traversal.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_cascade import (
    ActiveCascadeError,
    EdgeEvidenceState,
    TransitionStatus,
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
)

from validation.mechanical.prestressed_serial_weak_link_active_cascade import (
    SERIAL_EDGE_CONFIDENCE,
    SERIAL_EDGE_UNCERTAINTY,
    SERIAL_POLICY,
    GraphUpdateMustNotRunResolver,
    ProbeMustNotRunPlanner,
    Scenario03ASerialCandidateProvider,
    SerialActiveCascadeResult,
    SerialActiveCascadeValidationError,
    SerialCascadeStepAudit,
    SerialGraphSnapshot,
    build_case,
    build_serial_explorer,
    build_serial_snapshot,
    candidate_from_operator,
    inspect_serial_transition,
    operator_lookup,
    outgoing_operators,
    run_serial_active_cascade,
    serial_candidates,
)


@pytest.fixture(scope="module")
def case() -> PrestressedSerialWeakLinkValidationCase:
    return build_case()


@pytest.fixture(scope="module")
def result(
    case: PrestressedSerialWeakLinkValidationCase,
) -> SerialActiveCascadeResult:
    return run_serial_active_cascade(case)


# =============================================================================
# Configuration contract
# =============================================================================


def test_serial_edge_confidence_exceeds_policy_threshold() -> None:
    assert SERIAL_EDGE_CONFIDENCE >= SERIAL_POLICY.min_confidence


def test_serial_edge_uncertainty_is_below_policy_threshold() -> None:
    assert SERIAL_EDGE_UNCERTAINTY <= SERIAL_POLICY.max_uncertainty


def test_serial_policy_still_allows_probe_for_real_ambiguity() -> None:
    assert SERIAL_POLICY.require_unique_best is True
    assert SERIAL_POLICY.probe_when_multiple_viable is True


# =============================================================================
# Snapshot / hidden-label boundary
# =============================================================================


def test_serial_snapshot_contract() -> None:
    snapshot = build_serial_snapshot()

    assert isinstance(snapshot, SerialGraphSnapshot)
    assert snapshot.graph_id == "mechanical_03A_serial_active_cascade"
    assert snapshot.version == "v0"


def test_serial_snapshot_declares_no_expected_role_labels() -> None:
    snapshot = build_serial_snapshot()

    assert snapshot.metadata[
        "expected_role_labels_included"
    ] is False


def test_serial_snapshot_metadata_is_read_only() -> None:
    snapshot = build_serial_snapshot()

    assert isinstance(snapshot.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        snapshot.metadata["x"] = 1  # type: ignore[index]


def test_serial_snapshot_is_frozen() -> None:
    snapshot = build_serial_snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.version = "v1"  # type: ignore[misc]


# =============================================================================
# Candidate extraction
# =============================================================================


def test_operator_lookup_contains_exact_serial_relations(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    lookup = operator_lookup(case)

    assert set(lookup) == {
        PRELOAD_TO_PROXIMAL,
        PROXIMAL_TO_WEAK_LINK,
        WEAK_LINK_TO_DISTAL,
        DISTAL_TO_TERMINAL,
    }


def test_each_nonterminal_node_has_one_outgoing_operator(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    for channel_id in (
        PRELOAD_CHANNEL,
        PROXIMAL_CHANNEL,
        WEAK_LINK_CHANNEL,
        DISTAL_CHANNEL,
    ):
        assert len(
            outgoing_operators(
                case,
                channel_id,
            )
        ) == 1


def test_terminal_has_no_outgoing_operator(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert outgoing_operators(
        case,
        TERMINAL_CHANNEL,
    ) == ()


def test_candidate_from_operator_uses_structural_confidence(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    operator = operator_lookup(case)[
        PRELOAD_TO_PROXIMAL
    ]

    candidate = candidate_from_operator(
        operator
    )

    assert candidate.source_id == PRELOAD_CHANNEL
    assert candidate.target_id == PROXIMAL_CHANNEL
    assert candidate.relation_id == PRELOAD_TO_PROXIMAL
    assert candidate.confidence == pytest.approx(
        SERIAL_EDGE_CONFIDENCE
    )
    assert candidate.uncertainty == pytest.approx(
        SERIAL_EDGE_UNCERTAINTY
    )
    assert candidate.evidence_state is EdgeEvidenceState.CANDIDATE


def test_candidate_metadata_uses_no_expected_role_label(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    operator = operator_lookup(case)[
        PRELOAD_TO_PROXIMAL
    ]

    candidate = candidate_from_operator(
        operator
    )

    assert candidate.metadata[
        "expected_role_label_used"
    ] is False


def test_serial_candidates_never_invent_second_branch(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    for channel_id in (
        PRELOAD_CHANNEL,
        PROXIMAL_CHANNEL,
        WEAK_LINK_CHANNEL,
        DISTAL_CHANNEL,
    ):
        candidates = serial_candidates(
            case,
            channel_id,
        )

        assert len(candidates) == 1


def test_terminal_serial_candidates_are_empty(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert serial_candidates(
        case,
        TERMINAL_CHANNEL,
    ) == ()


# =============================================================================
# Local transition decisions
# =============================================================================


@pytest.mark.parametrize(
    ("source_id", "target_id", "relation_id"),
    (
        (
            PRELOAD_CHANNEL,
            PROXIMAL_CHANNEL,
            PRELOAD_TO_PROXIMAL,
        ),
        (
            PROXIMAL_CHANNEL,
            WEAK_LINK_CHANNEL,
            PROXIMAL_TO_WEAK_LINK,
        ),
        (
            WEAK_LINK_CHANNEL,
            DISTAL_CHANNEL,
            WEAK_LINK_TO_DISTAL,
        ),
        (
            DISTAL_CHANNEL,
            TERMINAL_CHANNEL,
            DISTAL_TO_TERMINAL,
        ),
    ),
)
def test_each_serial_transition_is_passively_confirmable(
    case: PrestressedSerialWeakLinkValidationCase,
    source_id: str,
    target_id: str,
    relation_id: str,
) -> None:
    decision = inspect_serial_transition(
        case,
        source_id,
    )

    assert decision.status is TransitionStatus.CONFIRMED
    assert decision.selected is not None
    assert decision.selected.target_id == target_id
    assert decision.selected.relation_id == relation_id
    assert (
        decision.reason
        == "single_candidate_with_sufficient_evidence"
    )


def test_terminal_transition_is_terminal(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    decision = inspect_serial_transition(
        case,
        TERMINAL_CHANNEL,
    )

    assert decision.status is TransitionStatus.TERMINAL
    assert decision.selected is None
    assert decision.reason == "no_outgoing_candidates"


# =============================================================================
# Probe / graph-update tripwires
# =============================================================================


def test_probe_planner_tripwire_raises_if_called() -> None:
    planner = ProbeMustNotRunPlanner()

    with pytest.raises(
        SerialActiveCascadeValidationError
    ):
        planner.plan_probe(
            current_node_id=PRELOAD_CHANNEL,
            candidates=(),
            graph_snapshot=build_serial_snapshot(),
            state=run_serial_active_cascade().state,
        )

    assert planner.calls == 1


def test_graph_update_resolver_tripwire_raises_if_called() -> None:
    resolver = GraphUpdateMustNotRunResolver()

    with pytest.raises(
        SerialActiveCascadeValidationError
    ):
        resolver.resolve_update(
            current_node_id=PRELOAD_CHANNEL,
            candidates=(),
            graph_update={},
            state=run_serial_active_cascade().state,
        )

    assert resolver.calls == 1


def test_explorer_probe_request_is_rejected_for_confirmed_transition(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    bundle = build_serial_explorer(case)
    snapshot = build_serial_snapshot()

    from roif.active_cascade import start_active_cascade

    state = start_active_cascade(
        PRELOAD_CHANNEL,
        graph_version=snapshot.version,
    )

    decision = bundle.explorer.inspect_transition(
        state,
        graph_snapshot=snapshot,
    )

    assert decision.status is TransitionStatus.CONFIRMED

    with pytest.raises(ActiveCascadeError):
        bundle.explorer.plan_required_probe(
            state,
            graph_snapshot=snapshot,
            decision=decision,
        )

    assert bundle.probe_planner.calls == 0


# =============================================================================
# Complete traversal
# =============================================================================


def test_complete_serial_active_cascade_runs(
    result: SerialActiveCascadeResult,
) -> None:
    assert isinstance(
        result,
        SerialActiveCascadeResult,
    )


def test_complete_serial_active_cascade_is_complete(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.state.complete is True
    assert result.state.current_node_id == TERMINAL_CHANNEL


def test_complete_node_path_is_exact(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.node_path == (
        PRELOAD_CHANNEL,
        PROXIMAL_CHANNEL,
        WEAK_LINK_CHANNEL,
        DISTAL_CHANNEL,
        TERMINAL_CHANNEL,
    )


def test_complete_relation_path_is_exact(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.relation_path == (
        PRELOAD_TO_PROXIMAL,
        PROXIMAL_TO_WEAK_LINK,
        WEAK_LINK_TO_DISTAL,
        DISTAL_TO_TERMINAL,
    )


def test_complete_path_contains_four_confirmed_edges(
    result: SerialActiveCascadeResult,
) -> None:
    assert len(
        result.state.confirmed_edges
    ) == 4


def test_all_confirmed_edges_use_passive_evidence(
    result: SerialActiveCascadeResult,
) -> None:
    assert {
        edge.confirmation_source
        for edge in result.state.confirmed_edges
    } == {
        "passive_evidence"
    }


def test_no_confirmed_edge_contains_probe_identifier(
    result: SerialActiveCascadeResult,
) -> None:
    assert all(
        edge.probe_identifier is None
        for edge in result.state.confirmed_edges
    )


def test_complete_pipeline_never_requests_probe(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.probe_request_count == 0
    assert result.probe_used is False


def test_complete_pipeline_never_resolves_graph_update(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.graph_update_resolution_count == 0


def test_complete_pipeline_records_four_confirmations_and_terminal(
    result: SerialActiveCascadeResult,
) -> None:
    assert len(result.decisions) == 5

    assert tuple(
        decision.status
        for decision in result.decisions
    ) == (
        TransitionStatus.CONFIRMED,
        TransitionStatus.CONFIRMED,
        TransitionStatus.CONFIRMED,
        TransitionStatus.CONFIRMED,
        TransitionStatus.TERMINAL,
    )


def test_audit_steps_preserve_same_status_sequence(
    result: SerialActiveCascadeResult,
) -> None:
    assert tuple(
        step.status
        for step in result.audit_steps
    ) == (
        TransitionStatus.CONFIRMED,
        TransitionStatus.CONFIRMED,
        TransitionStatus.CONFIRMED,
        TransitionStatus.CONFIRMED,
        TransitionStatus.TERMINAL,
    )


def test_confirmed_audit_steps_use_no_probe(
    result: SerialActiveCascadeResult,
) -> None:
    confirmed = tuple(
        step
        for step in result.audit_steps
        if step.status is TransitionStatus.CONFIRMED
    )

    assert len(confirmed) == 4

    assert all(
        step.confirmation_source == "passive_evidence"
        for step in confirmed
    )

    assert all(
        step.probe_used is False
        for step in confirmed
    )


def test_terminal_audit_step_has_no_relation(
    result: SerialActiveCascadeResult,
) -> None:
    terminal = result.audit_steps[-1]

    assert terminal.status is TransitionStatus.TERMINAL
    assert terminal.source_id == TERMINAL_CHANNEL
    assert terminal.target_id is None
    assert terminal.relation_id is None
    assert terminal.confirmation_source is None
    assert terminal.probe_used is False


# =============================================================================
# Audit boundary
# =============================================================================


def test_result_metadata_declares_no_expected_labels(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.metadata[
        "expected_role_labels_used"
    ] is False


def test_result_metadata_preserves_control_question(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.metadata[
        "control_question"
    ] == (
        "does_not_probe_sufficiently_known_single_candidate_edges"
    )


def test_result_metadata_records_four_passive_edges(
    result: SerialActiveCascadeResult,
) -> None:
    assert result.metadata[
        "passive_confirmed_edge_count"
    ] == 4


# =============================================================================
# Immutability
# =============================================================================


def test_result_metadata_is_read_only(
    result: SerialActiveCascadeResult,
) -> None:
    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_audit_step_metadata_is_read_only(
    result: SerialActiveCascadeResult,
) -> None:
    step = result.audit_steps[0]

    assert isinstance(
        step.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        step.metadata["x"] = 1  # type: ignore[index]


def test_audit_step_is_frozen(
    result: SerialActiveCascadeResult,
) -> None:
    step: SerialCascadeStepAudit = result.audit_steps[0]

    with pytest.raises(FrozenInstanceError):
        step.source_id = "x"  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_candidate_provider_is_deterministic(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    left = Scenario03ASerialCandidateProvider(
        case
    )
    right = Scenario03ASerialCandidateProvider(
        case
    )

    from roif.active_cascade import start_active_cascade

    snapshot = build_serial_snapshot()

    state = start_active_cascade(
        PRELOAD_CHANNEL,
        graph_version=snapshot.version,
    )

    assert (
        left.candidates_for(
            PRELOAD_CHANNEL,
            graph_snapshot=snapshot,
            state=state,
        )
        ==
        right.candidates_for(
            PRELOAD_CHANNEL,
            graph_snapshot=snapshot,
            state=state,
        )
    )


def test_complete_serial_active_cascade_is_deterministic(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    left = run_serial_active_cascade(
        case
    )
    right = run_serial_active_cascade(
        case
    )

    assert left.state == right.state
    assert left.decisions == right.decisions
    assert left.audit_steps == right.audit_steps
    assert left.probe_request_count == right.probe_request_count
    assert (
        left.graph_update_resolution_count
        == right.graph_update_resolution_count
    )
    assert left.metadata == right.metadata
