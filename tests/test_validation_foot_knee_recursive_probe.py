"""
ROIF Validation Suite
Clinical Scenario 01C:
Recursive Active Probe Reconstruction

Scientific purpose
------------------
Scenario 01A:
    known graph -> causal-role inference

Scenario 01B:
    incomplete graph -> one blind Probe* -> GraphUpdateProposal

Scenario 01C:
    incomplete graph
        -> Probe* #1
        -> observation
        -> explicit authorization
        -> updated graph G1
        -> reduced uncertainty
        -> Probe* #2
        -> observation
        -> explicit authorization
        -> reconstructed graph G2
        -> ROIF Solver
        -> D_origin / D_fast / D_root / Node*
        -> comparison with Scenario 01A external labels

Hidden ground truth is never used during Probe estimation or ranking.
It is queried only after Probe* selection to simulate the resulting
observation in this controlled computational validation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_probe_engine import ActiveProbeEngine
from roif.probe_entities import (
    GraphUpdate,
    Observation,
    ObservationChannel,
    Perturbation,
    PerturbationType,
    ProbeDefinition,
    ProbeMethod,
    ProbePhase,
    ProbePurpose,
    ProbeRegime,
)
from roif.probe_graph_adapter import (
    GraphUpdateProposal,
    ProbeGraphAdapter,
    ProbeGraphLink,
    ProbeGraphTarget,
)
from roif.probe_policy import (
    ProbeAssessment,
    ProbePolicyContext,
)
from roif.probe_registry import ProbeRegistry
from roif.roif_solver import (
    ROIFSolution,
    solve_roif,
)

from validation.clinical.foot_knee_case import (
    FootKneeValidationCase,
    build_foot_knee_case,
)
from validation.clinical.foot_knee_probe_case import (
    FootKneeProbeValidationCase,
    build_foot_knee_probe_case,
    evaluate_edge_hypothesis,
)
from validation.clinical.foot_knee_recursive_probe_case import (
    AuthorizedGraphUpdate,
    FootKneeRecursiveProbeValidationError,
    RecursiveProbeState,
    assert_complete_reconstruction,
    authorize_relation_update,
    build_recursive_probe_initial_state,
    build_recursive_snapshot,
    reconstruction_matches_reference_topology,
    reference_operator_ids,
)


FIRST_PROBE_ID = "probe_forefoot_compliance_response"
SECOND_PROBE_ID = "probe_tibial_rotation_response"

RELATION_BY_PROBE = {
    FIRST_PROBE_ID: "fibrosis_to_compliance",
    SECOND_PROBE_ID: "compliance_to_tibial_rotation",
}

PROBE_BY_RELATION = {
    relation_id: probe_id
    for probe_id, relation_id in RELATION_BY_PROBE.items()
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def probe_case() -> FootKneeProbeValidationCase:
    return build_foot_knee_probe_case()


@pytest.fixture()
def known_case() -> FootKneeValidationCase:
    return build_foot_knee_case()


@pytest.fixture()
def initial_state() -> RecursiveProbeState:
    return build_recursive_probe_initial_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_identifier(candidate) -> str:
    definition = getattr(candidate, "definition", None)

    if definition is not None:
        identifier = getattr(definition, "identifier", None)
        if identifier is not None:
            return identifier

    for name in ("probe_identifier", "identifier"):
        identifier = getattr(candidate, name, None)
        if identifier is not None:
            return identifier

    raise AssertionError(
        "Planner candidate exposes no Probe identifier."
    )


def _probe_definition_for_relation(
    relation_id: str,
) -> ProbeDefinition:
    if relation_id == "fibrosis_to_compliance":
        return ProbeDefinition(
            identifier=FIRST_PROBE_ID,
            name="Forefoot compliance response probe",
            method=ProbeMethod.SIMULATION,
            regime=ProbeRegime.DYNAMIC,
            purpose=ProbePurpose.REDUCE_UNCERTAINTY,
            perturbation=Perturbation(
                kind=PerturbationType.SIMULATED_INTERVENTION,
                magnitude=1.0,
                magnitude_units="normalized",
                duration_seconds=1.0,
                parameters={
                    "validation_stage": "01C",
                },
            ),
            description=(
                "Synthetic reversible Probe of the public uncertainty "
                "between forefoot fibrosis and forefoot compliance."
            ),
            tags=(
                "validation",
                "clinical_01C",
                "synthetic",
            ),
        )

    if relation_id == "compliance_to_tibial_rotation":
        return ProbeDefinition(
            identifier=SECOND_PROBE_ID,
            name="Tibial rotation response probe",
            method=ProbeMethod.SIMULATION,
            regime=ProbeRegime.DYNAMIC,
            purpose=ProbePurpose.REDUCE_UNCERTAINTY,
            perturbation=Perturbation(
                kind=PerturbationType.SIMULATED_INTERVENTION,
                magnitude=1.0,
                magnitude_units="normalized",
                duration_seconds=1.0,
                parameters={
                    "validation_stage": "01C",
                },
            ),
            description=(
                "Synthetic reversible Probe of the public uncertainty "
                "between forefoot compliance and tibial rotation control."
            ),
            tags=(
                "validation",
                "clinical_01C",
                "synthetic",
            ),
        )

    raise KeyError(relation_id)


def _probe_link_for_relation(
    relation_id: str,
) -> ProbeGraphLink:
    if relation_id == "fibrosis_to_compliance":
        return ProbeGraphLink(
            probe_identifier=FIRST_PROBE_ID,
            targets=(
                ProbeGraphTarget(
                    target_identifier=relation_id,
                    sensitivity=1.0,
                    discrimination=1.0,
                    expected_uncertainty_reduction=0.80,
                    metadata={
                        "ground_truth_included": False,
                    },
                ),
            ),
            novelty=0.90,
            feasibility=0.95,
            estimate_confidence=0.90,
            metadata={
                "validation_stage": "01C",
                "ground_truth_included": False,
            },
        )

    if relation_id == "compliance_to_tibial_rotation":
        return ProbeGraphLink(
            probe_identifier=SECOND_PROBE_ID,
            targets=(
                ProbeGraphTarget(
                    target_identifier=relation_id,
                    sensitivity=0.85,
                    discrimination=0.90,
                    expected_uncertainty_reduction=0.55,
                    metadata={
                        "ground_truth_included": False,
                    },
                ),
            ),
            novelty=0.50,
            feasibility=0.95,
            estimate_confidence=0.85,
            metadata={
                "validation_stage": "01C",
                "ground_truth_included": False,
            },
        )

    raise KeyError(relation_id)


def _safe_assessment() -> ProbeAssessment:
    return ProbeAssessment(
        expected_cost=0.05,
        expected_duration_seconds=1.0,
        cascade_risk=0.0,
        uncertainty=0.05,
        affects_external_systems=False,
        large_scale_effect_possible=False,
        uncontrolled_propagation_possible=False,
        requires_human_authorization=False,
        metadata={
            "validation_stage": "01C",
        },
    )


def _safe_context() -> ProbePolicyContext:
    return ProbePolicyContext(
        human_authorized=False,
        maximum_cost=1.0,
        maximum_duration_seconds=60.0,
        maximum_cascade_risk=0.25,
        maximum_uncertainty=1.0,
        non_fonit_gate_enabled=True,
        metadata={
            "validation_case": "clinical_01C_foot_knee_recursive_probe",
        },
    )


def _build_active_stack(
    state: RecursiveProbeState,
):
    """
    Build APE objects only for currently unresolved relations.

    A resolved relation therefore cannot remain a candidate in the
    following recursive step.
    """

    definitions = tuple(
        _probe_definition_for_relation(relation_id)
        for relation_id in state.unresolved_relation_ids
    )

    links = tuple(
        _probe_link_for_relation(relation_id)
        for relation_id in state.unresolved_relation_ids
    )

    registry = ProbeRegistry(definitions)
    adapter = ProbeGraphAdapter(links)
    snapshot = build_recursive_snapshot(state)

    estimates = adapter.estimate_all(
        registry,
        snapshot,
    )

    assessments = {
        definition.identifier: _safe_assessment()
        for definition in definitions
    }

    engine = ActiveProbeEngine(registry)

    plan = engine.plan(
        estimates,
        assessments=assessments,
        context=_safe_context(),
    )

    return (
        registry,
        adapter,
        snapshot,
        estimates,
        engine,
        plan,
    )


def _observation_pair(
    probe_case: FootKneeProbeValidationCase,
    probe_identifier: str,
) -> tuple[Observation, Observation]:
    """
    Query hidden truth only after Probe* selection.
    """

    relation_id = RELATION_BY_PROBE[probe_identifier]
    relation = probe_case.uncertain_relation(relation_id)

    edge_present = evaluate_edge_hypothesis(
        probe_case,
        source_id=relation.source_id,
        target_id=relation.target_id,
    )

    baseline = Observation(
        phase=ProbePhase.BASELINE,
        channel=ObservationChannel.SIGNAL,
        target=relation_id,
        value=0.0,
        units="normalized",
        confidence=0.95,
        metadata={
            "validation_stage": "01C",
            "hidden_truth_exposed_to_planner": False,
        },
    )

    reassessment = Observation(
        phase=ProbePhase.REASSESSMENT,
        channel=ObservationChannel.SIGNAL,
        target=relation_id,
        value=1.0 if edge_present else 0.0,
        units="normalized",
        confidence=0.95,
        metadata={
            "validation_stage": "01C",
            "synthetic_oracle_observation": True,
            "hidden_truth_exposed_to_planner": False,
        },
    )

    return baseline, reassessment


def _execute_one_recursive_probe(
    state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
):
    """
    Perform one complete recursive Probe step.

    Hidden truth is accessed only after Planner has selected Probe*.
    """

    (
        registry,
        adapter,
        snapshot,
        estimates,
        engine,
        plan,
    ) = _build_active_stack(state)

    assert plan.selected is not None

    selected_id = _candidate_identifier(
        plan.selected
    )

    relation_id = RELATION_BY_PROBE[
        selected_id
    ]

    relation = probe_case.uncertain_relation(
        relation_id
    )

    run = engine.start_selected(
        plan,
        metadata={
            "validation_stage": "01C",
            "graph_version": state.graph_version,
        },
    )

    baseline, reassessment = _observation_pair(
        probe_case,
        selected_id,
    )

    engine.add_observation(baseline)
    engine.add_observation(reassessment)

    result = engine.complete(
        graph_update=GraphUpdate(
            affected_nodes=(
                relation.source_id,
                relation.target_id,
            ),
            affected_edges=(
                (
                    relation.source_id,
                    relation.target_id,
                ),
            ),
            confidence=0.95,
            metadata={
                "validation_stage": "01C",
                "proposal_only": True,
                "automatic_mutation": False,
            },
        ),
        notes=(
            "Clinical Scenario 01C recursive synthetic observation."
        ),
        metadata={
            "validation_stage": "01C",
        },
    )

    proposal = adapter.propose_graph_update(
        result,
        snapshot,
        metadata={
            "validation_stage": "01C",
            "authorization_required": True,
        },
    )

    updated_state = authorize_relation_update(
        state,
        probe_identifier=selected_id,
        relation_id=relation_id,
        observation_confidence=0.95,
        authorized=True,
    )

    return (
        registry,
        adapter,
        snapshot,
        estimates,
        engine,
        plan,
        run,
        result,
        proposal,
        relation,
        updated_state,
    )


def _run_complete_reconstruction(
    initial: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
):
    first = _execute_one_recursive_probe(
        initial,
        probe_case,
    )

    state_1 = first[-1]

    second = _execute_one_recursive_probe(
        state_1,
        probe_case,
    )

    state_2 = second[-1]

    return first, second, state_2


def _inferred_roles(
    solution: ROIFSolution,
) -> dict[str, str]:
    return {
        "d_origin": solution.root_result.d_origin,
        "d_fast": solution.root_result.d_fast,
        "d_root": solution.root_result.d_root,
        "node_star": solution.root_result.node_star,
    }


# ---------------------------------------------------------------------------
# Initial 01C contract
# ---------------------------------------------------------------------------


def test_recursive_case_starts_from_01b_incomplete_graph(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    assert (
        initial_state.system.channel_ids
        == probe_case.public_system.channel_ids
    )

    assert set(initial_state.operator_ids) == set(
        probe_case.public_operator_ids
    )

    assert initial_state.unresolved_relation_ids == (
        "fibrosis_to_compliance",
        "compliance_to_tibial_rotation",
    )


def test_recursive_case_starts_at_version_zero(
    initial_state: RecursiveProbeState,
) -> None:
    assert (
        initial_state.graph_version
        == "clinical-01C-v0"
    )

    assert initial_state.probe_count == 0
    assert initial_state.resolved_relation_ids == ()
    assert initial_state.is_fully_reconstructed is False


def test_recursive_state_metadata_preserves_safety_boundary(
    initial_state: RecursiveProbeState,
) -> None:
    assert isinstance(
        initial_state.metadata,
        MappingProxyType,
    )

    assert (
        initial_state.metadata[
            "hidden_graph_exposed_to_planner"
        ]
        is False
    )

    assert (
        initial_state.metadata[
            "hidden_labels_exposed_to_planner"
        ]
        is False
    )

    assert (
        initial_state.metadata[
            "automatic_graph_mutation"
        ]
        is False
    )

    assert (
        initial_state.metadata[
            "explicit_authorization_required"
        ]
        is True
    )


def test_recursive_state_is_immutable(
    initial_state: RecursiveProbeState,
) -> None:
    with pytest.raises(FrozenInstanceError):
        initial_state.graph_version = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Recursive snapshot
# ---------------------------------------------------------------------------


def test_initial_snapshot_contains_two_uncertainties(
    initial_state: RecursiveProbeState,
) -> None:
    snapshot = build_recursive_snapshot(
        initial_state
    )

    assert len(snapshot.uncertainties) == 2

    assert {
        item.target.identifier
        for item in snapshot.uncertainties
    } == {
        "fibrosis_to_compliance",
        "compliance_to_tibial_rotation",
    }


def test_initial_snapshot_contains_no_hidden_truth(
    initial_state: RecursiveProbeState,
) -> None:
    snapshot = build_recursive_snapshot(
        initial_state
    )

    assert (
        snapshot.metadata[
            "hidden_graph_exposed"
        ]
        is False
    )

    assert (
        snapshot.metadata[
            "hidden_labels_exposed"
        ]
        is False
    )

    for uncertainty in snapshot.uncertainties:
        assert not hasattr(
            uncertainty,
            "expected_present",
        )

        assert not hasattr(
            uncertainty,
            "gain",
        )


# ---------------------------------------------------------------------------
# Explicit authorization
# ---------------------------------------------------------------------------


def test_graph_update_cannot_be_applied_without_authorization(
    initial_state: RecursiveProbeState,
) -> None:
    with pytest.raises(
        FootKneeRecursiveProbeValidationError
    ):
        authorize_relation_update(
            initial_state,
            probe_identifier=FIRST_PROBE_ID,
            relation_id="fibrosis_to_compliance",
            observation_confidence=0.95,
            authorized=False,
        )


def test_authorized_update_record_requires_true_authorization() -> None:
    with pytest.raises(
        FootKneeRecursiveProbeValidationError
    ):
        AuthorizedGraphUpdate(
            proposal_identifier=FIRST_PROBE_ID,
            relation_id="fibrosis_to_compliance",
            operator_id="fibrosis_to_compliance",
            graph_version_before="v0",
            graph_version_after="v1",
            authorization=False,
        )


# ---------------------------------------------------------------------------
# First recursive Probe
# ---------------------------------------------------------------------------


def test_first_recursive_step_builds_real_blind_plan(
    initial_state: RecursiveProbeState,
) -> None:
    *_, plan = _build_active_stack(
        initial_state
    )

    assert plan.selected is not None
    assert len(plan.ranked_candidates) == 2

    assert _candidate_identifier(
        plan.selected
    ) in {
        FIRST_PROBE_ID,
        SECOND_PROBE_ID,
    }


def test_first_probe_selection_is_deterministic(
    initial_state: RecursiveProbeState,
) -> None:
    *_, left = _build_active_stack(
        initial_state
    )

    *_, right = _build_active_stack(
        initial_state
    )

    assert left.selected is not None
    assert right.selected is not None

    assert (
        _candidate_identifier(left.selected)
        == _candidate_identifier(right.selected)
    )


def test_first_recursive_step_generates_graph_update_proposal(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    *_, proposal, relation, updated = (
        _execute_one_recursive_probe(
            initial_state,
            probe_case,
        )
    )

    assert isinstance(
        proposal,
        GraphUpdateProposal,
    )

    assert (
        (
            relation.source_id,
            relation.target_id,
        )
        in proposal.graph_update.affected_edges
    )

    assert updated.probe_count == 1


def test_first_authorized_update_advances_graph_version(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    *_, state_1 = _execute_one_recursive_probe(
        initial_state,
        probe_case,
    )

    assert (
        state_1.graph_version
        == "clinical-01C-v1"
    )

    assert state_1.probe_count == 1
    assert len(state_1.resolved_relation_ids) == 1
    assert len(state_1.unresolved_relation_ids) == 1


def test_first_authorized_update_adds_exactly_one_operator(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    *_, state_1 = _execute_one_recursive_probe(
        initial_state,
        probe_case,
    )

    assert (
        len(state_1.system.operators)
        == len(initial_state.system.operators) + 1
    )

    added = (
        set(state_1.operator_ids)
        - set(initial_state.operator_ids)
    )

    assert added == set(
        state_1.resolved_relation_ids
    )


def test_first_update_removes_only_resolved_uncertainty(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    *_, state_1 = _execute_one_recursive_probe(
        initial_state,
        probe_case,
    )

    assert len(state_1.unresolved_relation_ids) == 1

    assert set(
        state_1.unresolved_relation_ids
    ).isdisjoint(
        state_1.resolved_relation_ids
    )


# ---------------------------------------------------------------------------
# Second recursive Probe
# ---------------------------------------------------------------------------


def test_second_snapshot_contains_only_remaining_uncertainty(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    *_, state_1 = _execute_one_recursive_probe(
        initial_state,
        probe_case,
    )

    snapshot = build_recursive_snapshot(
        state_1
    )

    assert len(snapshot.uncertainties) == 1

    assert (
        snapshot.uncertainties[0].target.identifier
        == state_1.unresolved_relation_ids[0]
    )


def test_second_plan_contains_only_remaining_probe(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    *_, state_1 = _execute_one_recursive_probe(
        initial_state,
        probe_case,
    )

    *_, plan = _build_active_stack(
        state_1
    )

    assert plan.selected is not None
    assert len(plan.ranked_candidates) == 1

    selected_id = _candidate_identifier(
        plan.selected
    )

    assert (
        RELATION_BY_PROBE[selected_id]
        == state_1.unresolved_relation_ids[0]
    )


def test_resolved_probe_cannot_be_selected_again(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    first = _execute_one_recursive_probe(
        initial_state,
        probe_case,
    )

    first_plan = first[5]
    state_1 = first[-1]

    assert first_plan.selected is not None

    first_selected = _candidate_identifier(
        first_plan.selected
    )

    *_, second_plan = _build_active_stack(
        state_1
    )

    assert second_plan.selected is not None

    second_selected = _candidate_identifier(
        second_plan.selected
    )

    assert second_selected != first_selected


def test_second_authorized_update_completes_reconstruction(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    _, _, state_2 = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    assert (
        state_2.graph_version
        == "clinical-01C-v2"
    )

    assert state_2.probe_count == 2
    assert state_2.unresolved_relation_ids == ()
    assert state_2.is_fully_reconstructed is True


# ---------------------------------------------------------------------------
# Topology reconstruction
# ---------------------------------------------------------------------------


def test_two_probes_resolve_both_hidden_relations(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    _, _, state_2 = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    assert set(
        state_2.resolved_relation_ids
    ) == {
        "fibrosis_to_compliance",
        "compliance_to_tibial_rotation",
    }


def test_reconstructed_graph_has_five_reference_operators(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    _, _, state_2 = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    assert len(state_2.system.operators) == 5

    assert set(
        state_2.operator_ids
    ) == set(
        reference_operator_ids()
    )


def test_reconstructed_topology_matches_01a(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    _, _, state_2 = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    assert (
        reconstruction_matches_reference_topology(
            state_2
        )
        is True
    )

    assert_complete_reconstruction(
        state_2
    )


def test_reconstruction_does_not_replace_entities_or_channels(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    _, _, state_2 = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    assert (
        state_2.system.channel_ids
        == initial_state.system.channel_ids
    )

    assert (
        state_2.system.entities
        == initial_state.system.entities
    )

    assert (
        state_2.system.planes
        == initial_state.system.planes
    )


def test_reconstruction_history_records_two_authorized_updates(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    _, _, state_2 = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    assert len(
        state_2.authorized_updates
    ) == 2

    assert all(
        update.authorization is True
        for update in state_2.authorized_updates
    )

    assert {
        update.relation_id
        for update in state_2.authorized_updates
    } == {
        "fibrosis_to_compliance",
        "compliance_to_tibial_rotation",
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_complete_recursive_reconstruction_is_deterministic(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
) -> None:
    left_first, left_second, left_state = (
        _run_complete_reconstruction(
            initial_state,
            probe_case,
        )
    )

    right_first, right_second, right_state = (
        _run_complete_reconstruction(
            initial_state,
            probe_case,
        )
    )

    assert (
        _candidate_identifier(
            left_first[5].selected
        )
        == _candidate_identifier(
            right_first[5].selected
        )
    )

    assert (
        _candidate_identifier(
            left_second[5].selected
        )
        == _candidate_identifier(
            right_second[5].selected
        )
    )

    assert (
        left_state.resolved_relation_ids
        == right_state.resolved_relation_ids
    )

    assert (
        set(left_state.operator_ids)
        == set(right_state.operator_ids)
    )


# ---------------------------------------------------------------------------
# Return reconstructed graph to normal ROIF Solver
# ---------------------------------------------------------------------------


def test_reconstructed_graph_can_be_solved_by_normal_roif_pipeline(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
    known_case: FootKneeValidationCase,
) -> None:
    _, _, reconstructed = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    solution = solve_roif(
        reconstructed.system,
        initial_state=known_case.initial_state,
        scenarios=known_case.scenarios,
        config=known_case.solver_config,
    )

    assert isinstance(
        solution,
        ROIFSolution,
    )

    assert solution.root_result is not None


def test_solver_receives_no_external_role_labels(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
    known_case: FootKneeValidationCase,
) -> None:
    _, _, reconstructed = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    # Expected labels are retained only for post-inference comparison.
    expected = dict(
        known_case.expected.as_mapping()
    )

    solution = solve_roif(
        reconstructed.system,
        initial_state=known_case.initial_state,
        scenarios=known_case.scenarios,
        config=known_case.solver_config,
    )

    assert _inferred_roles(solution)
    assert expected


def test_reconstructed_graph_recovers_01a_causal_roles(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
    known_case: FootKneeValidationCase,
) -> None:
    """
    Final scientific assertion for Scenario 01C.

    Expected labels are compared only after the full recursive
    reconstruction and normal Solver inference have completed.
    """

    _, _, reconstructed = _run_complete_reconstruction(
        initial_state,
        probe_case,
    )

    solution = solve_roif(
        reconstructed.system,
        initial_state=known_case.initial_state,
        scenarios=known_case.scenarios,
        config=known_case.solver_config,
    )

    actual = _inferred_roles(
        solution
    )

    expected = dict(
        known_case.expected.as_mapping()
    )

    assert actual == expected, {
        "expected": expected,
        "inferred": actual,
        "resolved_relation_ids": (
            reconstructed.resolved_relation_ids
        ),
        "operator_ids": (
            reconstructed.operator_ids
        ),
    }


def test_recursive_01c_end_to_end(
    initial_state: RecursiveProbeState,
    probe_case: FootKneeProbeValidationCase,
    known_case: FootKneeValidationCase,
) -> None:
    """
    Complete Scenario 01C validation invariant.
    """

    first, second, reconstructed = (
        _run_complete_reconstruction(
            initial_state,
            probe_case,
        )
    )

    first_plan = first[5]
    second_plan = second[5]

    assert first_plan.selected is not None
    assert second_plan.selected is not None

    first_probe = _candidate_identifier(
        first_plan.selected
    )

    second_probe = _candidate_identifier(
        second_plan.selected
    )

    # Recursive exploration must not repeat the same resolved Probe.
    assert first_probe != second_probe

    # Both hidden relations must now be authorized into the graph.
    assert reconstructed.probe_count == 2
    assert reconstructed.is_fully_reconstructed is True

    assert_complete_reconstruction(
        reconstructed
    )

    # The final graph returns to the ordinary ROIF inference path.
    solution = solve_roif(
        reconstructed.system,
        initial_state=known_case.initial_state,
        scenarios=known_case.scenarios,
        config=known_case.solver_config,
    )

    assert _inferred_roles(
        solution
    ) == dict(
        known_case.expected.as_mapping()
    )
