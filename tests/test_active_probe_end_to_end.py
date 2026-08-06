"""
End-to-end tests for the universal Active Probe Engine pipeline.

These tests verify the complete path:

    incomplete graph uncertainty
        ->
    Probe information estimates
        ->
    policy evaluation
        ->
    Probe* selection
        ->
    controlled Probe lifecycle
        ->
    ObservationDelta
        ->
    GraphUpdateProposal

The pipeline must not automatically mutate the graph or invoke the ROIF Solver.
"""

from __future__ import annotations

import pytest

from roif.active_probe_engine import (
    ActiveProbeEngine,
    ActiveProbeEngineState,
    ProbeExecutionRejectedError,
)
from roif.probe_entities import (
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
    GraphTarget,
    GraphTargetKind,
    GraphUncertainty,
    GraphUpdateProposal,
    IncompleteGraphSnapshot,
    ProbeGraphAdapter,
    ProbeGraphLink,
    ProbeGraphTarget,
)
from roif.probe_planner import (
    ProbeSelectionStatus,
)
from roif.probe_policy import (
    PolicyReasonCode,
    ProbeAssessment,
    ProbePolicy,
    ProbePolicyContext,
    ProbeRiskLevel,
)
from roif.probe_registry import ProbeRegistry


# ============================================================================
# Fixtures and helpers
# ============================================================================


def make_probe(
    identifier: str,
    *,
    name: str,
    method: ProbeMethod,
    regime: ProbeRegime,
    purpose: ProbePurpose,
    perturbation: PerturbationType,
    tags: tuple[str, ...],
) -> ProbeDefinition:
    """Create one universal Probe definition."""

    return ProbeDefinition(
        identifier=identifier,
        name=name,
        method=method,
        regime=regime,
        purpose=purpose,
        perturbation=Perturbation(
            kind=perturbation,
            magnitude=1.0,
            magnitude_units="relative",
            duration_seconds=5.0,
        ),
        description=f"Universal Probe: {name}.",
        tags=tags,
    )


@pytest.fixture
def observation_probe() -> ProbeDefinition:
    return make_probe(
        "system.passive_observation",
        name="Passive observation",
        method=ProbeMethod.OBSERVATION,
        regime=ProbeRegime.REST,
        purpose=ProbePurpose.OBSERVE,
        perturbation=PerturbationType.NONE,
        tags=("passive", "universal"),
    )


@pytest.fixture
def load_probe() -> ProbeDefinition:
    return make_probe(
        "system.controlled_load",
        name="Controlled load experiment",
        method=ProbeMethod.INSTRUMENTAL,
        regime=ProbeRegime.STATIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=PerturbationType.LOAD_CHANGE,
        tags=("controlled", "load", "universal"),
    )


@pytest.fixture
def simulation_probe() -> ProbeDefinition:
    return make_probe(
        "system.virtual_release",
        name="Virtual release simulation",
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.TRANSITION,
        purpose=ProbePurpose.REJECT_HYPOTHESIS,
        perturbation=PerturbationType.SIMULATED_INTERVENTION,
        tags=("simulation", "counterfactual", "universal"),
    )


@pytest.fixture
def registry(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
    simulation_probe: ProbeDefinition,
) -> ProbeRegistry:
    return ProbeRegistry(
        (
            observation_probe,
            load_probe,
            simulation_probe,
        )
    )


@pytest.fixture
def incomplete_graph() -> IncompleteGraphSnapshot:
    """
    Return a graph uncertainty view.

    The graph contains:

        node_source
            ->
        edge_source_mediator
            ->
        node_mediator
            ->
        edge_mediator_output
            ->
        node_output
    """

    return IncompleteGraphSnapshot(
        graph_id="universal-cascade-graph",
        version="v1",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.NODE,
                    identifier="node_source",
                ),
                uncertainty=0.25,
                importance=0.80,
                confidence=0.95,
                hypothesis_count=2,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier="edge_source_mediator",
                    source="node_source",
                    target="node_mediator",
                ),
                uncertainty=0.85,
                importance=1.00,
                confidence=0.90,
                hypothesis_count=4,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.NODE,
                    identifier="node_mediator",
                ),
                uncertainty=0.90,
                importance=1.00,
                confidence=0.92,
                hypothesis_count=4,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier="edge_mediator_output",
                    source="node_mediator",
                    target="node_output",
                ),
                uncertainty=0.70,
                importance=0.75,
                confidence=0.88,
                hypothesis_count=3,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.NODE,
                    identifier="node_output",
                ),
                uncertainty=0.30,
                importance=0.50,
                confidence=0.95,
                hypothesis_count=2,
            ),
        ),
        metadata={
            "graph_status": "incomplete",
            "automatic_mutation_allowed": False,
        },
    )


@pytest.fixture
def graph_adapter(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
    simulation_probe: ProbeDefinition,
) -> ProbeGraphAdapter:
    """
    Return graph mappings designed so that the controlled load Probe provides
    the greatest useful information.
    """

    return ProbeGraphAdapter(
        (
            ProbeGraphLink(
                probe_identifier=observation_probe.identifier,
                targets=(
                    ProbeGraphTarget(
                        target_identifier="node_output",
                        sensitivity=0.40,
                        discrimination=0.20,
                        expected_uncertainty_reduction=0.20,
                    ),
                ),
                novelty=0.15,
                feasibility=1.00,
                estimate_confidence=0.95,
            ),
            ProbeGraphLink(
                probe_identifier=load_probe.identifier,
                targets=(
                    ProbeGraphTarget(
                        target_identifier="edge_source_mediator",
                        sensitivity=0.95,
                        discrimination=0.95,
                        expected_uncertainty_reduction=0.90,
                    ),
                    ProbeGraphTarget(
                        target_identifier="node_mediator",
                        sensitivity=0.95,
                        discrimination=0.90,
                        expected_uncertainty_reduction=0.90,
                    ),
                    ProbeGraphTarget(
                        target_identifier="edge_mediator_output",
                        sensitivity=0.80,
                        discrimination=0.75,
                        expected_uncertainty_reduction=0.75,
                    ),
                ),
                novelty=0.90,
                feasibility=0.95,
                estimate_confidence=0.95,
            ),
            ProbeGraphLink(
                probe_identifier=simulation_probe.identifier,
                targets=(
                    ProbeGraphTarget(
                        target_identifier="node_source",
                        sensitivity=0.70,
                        discrimination=0.65,
                        expected_uncertainty_reduction=0.60,
                    ),
                    ProbeGraphTarget(
                        target_identifier="node_mediator",
                        sensitivity=0.70,
                        discrimination=0.65,
                        expected_uncertainty_reduction=0.60,
                    ),
                ),
                novelty=0.65,
                feasibility=1.00,
                estimate_confidence=0.90,
            ),
        )
    )


@pytest.fixture
def assessments(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
    simulation_probe: ProbeDefinition,
) -> dict[str, ProbeAssessment]:
    return {
        observation_probe.identifier: ProbeAssessment(
            expected_cost=0.0,
            expected_duration_seconds=10.0,
            risk_level=ProbeRiskLevel.NEGLIGIBLE,
            cascade_risk=0.0,
            uncertainty=0.05,
        ),
        load_probe.identifier: ProbeAssessment(
            expected_cost=0.10,
            expected_duration_seconds=5.0,
            risk_level=ProbeRiskLevel.LOW,
            cascade_risk=0.05,
            uncertainty=0.10,
            required_resources=frozenset(
                {"load_device", "sensor"}
            ),
        ),
        simulation_probe.identifier: ProbeAssessment(
            expected_cost=0.05,
            expected_duration_seconds=20.0,
            risk_level=ProbeRiskLevel.NEGLIGIBLE,
            cascade_risk=0.0,
            uncertainty=0.15,
            required_resources=frozenset({"simulator"}),
        ),
    }


@pytest.fixture
def policy_context() -> ProbePolicyContext:
    return ProbePolicyContext(
        available_resources=frozenset(
            {
                "load_device",
                "sensor",
                "simulator",
            }
        ),
        maximum_cost=1.0,
        maximum_duration_seconds=60.0,
        maximum_risk_level=ProbeRiskLevel.LOW,
        maximum_cascade_risk=0.20,
        maximum_uncertainty=0.50,
    )


def make_observation(
    phase: ProbePhase,
    *,
    value: float,
    target: str = "node_mediator",
) -> Observation:
    return Observation(
        phase=phase,
        channel=ObservationChannel.DISPLACEMENT,
        target=target,
        value=value,
        units="mm",
        confidence=0.95,
    )


# ============================================================================
# Estimate stage
# ============================================================================


def test_incomplete_graph_generates_estimates_for_all_registered_probes(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
    incomplete_graph: IncompleteGraphSnapshot,
) -> None:
    estimates = graph_adapter.estimate_all(
        registry,
        incomplete_graph,
    )

    assert len(estimates) == len(registry)

    assert tuple(
        estimate.probe_identifier
        for estimate in estimates
    ) == registry.identifiers()

    assert all(
        estimate.metadata["graph_id"]
        == incomplete_graph.graph_id
        for estimate in estimates
    )


def test_controlled_load_has_greater_information_value(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
    incomplete_graph: IncompleteGraphSnapshot,
    load_probe: ProbeDefinition,
    observation_probe: ProbeDefinition,
) -> None:
    estimates = {
        estimate.probe_identifier: estimate
        for estimate in graph_adapter.estimate_all(
            registry,
            incomplete_graph,
        )
    }

    assert (
        estimates[load_probe.identifier].information_gain
        > estimates[observation_probe.identifier].information_gain
    )

    assert (
        estimates[load_probe.identifier].uncertainty_reduction
        > estimates[
            observation_probe.identifier
        ].uncertainty_reduction
    )


# ============================================================================
# Planning stage
# ============================================================================


def test_planner_selects_controlled_load_as_probe_star(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
    incomplete_graph: IncompleteGraphSnapshot,
    assessments: dict[str, ProbeAssessment],
    policy_context: ProbePolicyContext,
    load_probe: ProbeDefinition,
) -> None:
    estimates = graph_adapter.estimate_all(
        registry,
        incomplete_graph,
    )

    engine = ActiveProbeEngine(
        registry,
        policy=ProbePolicy(),
    )

    plan = engine.plan(
        estimates,
        assessments=assessments,
        context=policy_context,
    )

    assert plan.status is ProbeSelectionStatus.SELECTED
    assert plan.probe_star is load_probe
    assert plan.selected is not None
    assert plan.selected.rank == 1


def test_policy_veto_can_override_best_information_probe(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
    incomplete_graph: IncompleteGraphSnapshot,
    assessments: dict[str, ProbeAssessment],
    policy_context: ProbePolicyContext,
    load_probe: ProbeDefinition,
) -> None:
    estimates = graph_adapter.estimate_all(
        registry,
        incomplete_graph,
    )

    unsafe_assessments = dict(assessments)
    unsafe_assessments[load_probe.identifier] = ProbeAssessment(
        risk_level=ProbeRiskLevel.CRITICAL,
        cascade_risk=0.95,
        uncontrolled_propagation_possible=True,
    )

    engine = ActiveProbeEngine(
        registry,
        policy=ProbePolicy(),
    )

    plan = engine.plan(
        estimates,
        assessments=unsafe_assessments,
        context=policy_context,
    )

    assert plan.probe_star is not load_probe

    rejected = tuple(
        candidate
        for candidate in plan.rejected_candidates
        if candidate.definition is load_probe
    )

    assert len(rejected) == 1

    reason_codes = {
        reason.code
        for reason in rejected[0].policy_result.reasons
    }

    assert PolicyReasonCode.NON_FONIT_VETO in reason_codes


# ============================================================================
# Complete Active Probe lifecycle
# ============================================================================


def test_complete_active_probe_pipeline(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
    incomplete_graph: IncompleteGraphSnapshot,
    assessments: dict[str, ProbeAssessment],
    policy_context: ProbePolicyContext,
    load_probe: ProbeDefinition,
) -> None:
    """
    Verify the complete Active Probe cycle without automatic graph mutation.
    """

    registry_before = registry.definitions()
    graph_before = incomplete_graph

    estimates = graph_adapter.estimate_all(
        registry,
        incomplete_graph,
    )

    engine = ActiveProbeEngine(
        registry,
        policy=ProbePolicy(),
    )

    plan = engine.plan(
        estimates,
        assessments=assessments,
        context=policy_context,
    )

    assert plan.status is ProbeSelectionStatus.SELECTED
    assert plan.probe_star is load_probe
    assert engine.state is ActiveProbeEngineState.PLANNED

    run = engine.start_selected(plan)

    assert run.definition is load_probe
    assert engine.state is ActiveProbeEngineState.RUNNING

    baseline = make_observation(
        ProbePhase.BASELINE,
        value=1.0,
    )

    perturbation = make_observation(
        ProbePhase.PERTURBATION,
        value=4.0,
    )

    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value=1.6,
    )

    engine.add_observations(
        (
            baseline,
            perturbation,
            reassessment,
        )
    )

    result = engine.complete(
        notes="Universal controlled experiment completed.",
    )

    assert engine.state is ActiveProbeEngineState.COMPLETED

    assert result.execution.definition is load_probe

    assert result.execution.baseline_observations == [
        baseline
    ]
    assert result.execution.perturbation_observations == [
        perturbation
    ]
    assert result.execution.reassessment_observations == [
        reassessment
    ]

    assert len(result.deltas) == 1
    assert result.deltas[0].delta == pytest.approx(0.6)

    proposal = graph_adapter.propose_graph_update(
        result,
        incomplete_graph,
        graph_version="v1",
    )

    assert isinstance(proposal, GraphUpdateProposal)

    assert proposal.graph_id == incomplete_graph.graph_id
    assert proposal.base_version == incomplete_graph.version
    assert proposal.probe_identifier == load_probe.identifier

    assert proposal.graph_update.affected_nodes == (
        "node_mediator",
    )

    assert proposal.graph_update.affected_edges == (
        ("node_source", "node_mediator"),
        ("node_mediator", "node_output"),
    )

    assert proposal.graph_update.evidence == tuple(
        result.deltas
    )

    assert (
        proposal.graph_update.metadata[
            "application_status"
        ]
        == "proposal_only"
    )

    assert proposal.metadata["automatic_application"] is False

    # The Registry remains unchanged.
    assert registry.definitions() == registry_before

    # The immutable graph uncertainty snapshot remains unchanged.
    assert incomplete_graph is graph_before
    assert incomplete_graph.version == "v1"
    assert (
        incomplete_graph.metadata[
            "automatic_mutation_allowed"
        ]
        is False
    )

    # Neither the engine nor adapter applies the proposal.
    assert not hasattr(engine, "apply_graph_update")
    assert not hasattr(graph_adapter, "apply_graph_update")


def test_engine_result_contains_no_automatic_graph_update(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(registry)

    engine.start_probe(load_probe.identifier)

    engine.add_observations(
        (
            make_observation(
                ProbePhase.BASELINE,
                value=1.0,
            ),
            make_observation(
                ProbePhase.REASSESSMENT,
                value=1.5,
            ),
        )
    )

    result = engine.complete()

    assert result.graph_update is None


def test_proposal_generation_does_not_change_snapshot_burden(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
    incomplete_graph: IncompleteGraphSnapshot,
    load_probe: ProbeDefinition,
) -> None:
    burden_before = (
        incomplete_graph.total_uncertainty_burden
    )
    uncertainties_before = incomplete_graph.uncertainties

    engine = ActiveProbeEngine(registry)
    engine.start_probe(load_probe.identifier)

    engine.add_observations(
        (
            make_observation(
                ProbePhase.BASELINE,
                value=1.0,
            ),
            make_observation(
                ProbePhase.REASSESSMENT,
                value=2.0,
            ),
        )
    )

    result = engine.complete()

    graph_adapter.propose_graph_update(
        result,
        incomplete_graph,
    )

    assert (
        incomplete_graph.total_uncertainty_burden
        == pytest.approx(burden_before)
    )
    assert incomplete_graph.uncertainties == uncertainties_before


# ============================================================================
# Direct execution safety
# ============================================================================


def test_direct_execution_still_cannot_bypass_non_fonit_gate(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        policy=ProbePolicy(),
    )

    with pytest.raises(ProbeExecutionRejectedError):
        engine.start_probe(
            load_probe.identifier,
            assessment=ProbeAssessment(
                risk_level=ProbeRiskLevel.CRITICAL,
                cascade_risk=1.0,
                affects_external_systems=True,
                large_scale_effect_possible=True,
                uncontrolled_propagation_possible=True,
            ),
            context=ProbePolicyContext(
                maximum_risk_level=ProbeRiskLevel.CRITICAL,
                maximum_cascade_risk=1.0,
                human_authorized=True,
            ),
        )


# ============================================================================
# Reproducibility and architectural boundaries
# ============================================================================


def test_full_pipeline_is_deterministic(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
    incomplete_graph: IncompleteGraphSnapshot,
    assessments: dict[str, ProbeAssessment],
    policy_context: ProbePolicyContext,
) -> None:
    estimates_first = graph_adapter.estimate_all(
        registry,
        incomplete_graph,
    )
    estimates_second = graph_adapter.estimate_all(
        registry,
        incomplete_graph,
    )

    assert estimates_first == estimates_second

    first_engine = ActiveProbeEngine(registry)
    second_engine = ActiveProbeEngine(registry)

    first_plan = first_engine.plan(
        estimates_first,
        assessments=assessments,
        context=policy_context,
    )
    second_plan = second_engine.plan(
        estimates_second,
        assessments=assessments,
        context=policy_context,
    )

    assert first_plan == second_plan
    assert first_plan.probe_star is second_plan.probe_star


def test_end_to_end_pipeline_has_no_solver_or_role_calls(
    registry: ProbeRegistry,
    graph_adapter: ProbeGraphAdapter,
) -> None:
    engine = ActiveProbeEngine(registry)

    forbidden_engine_methods = (
        "solve",
        "run_solver",
        "calculate_d_origin",
        "calculate_d_fast",
        "calculate_d_root",
        "calculate_node_star",
        "apply_graph_update",
    )

    forbidden_adapter_methods = (
        "solve",
        "run_solver",
        "calculate_d_origin",
        "calculate_d_fast",
        "calculate_d_root",
        "calculate_node_star",
        "apply_graph_update",
    )

    for method_name in forbidden_engine_methods:
        assert not hasattr(engine, method_name)

    for method_name in forbidden_adapter_methods:
        assert not hasattr(graph_adapter, method_name)
