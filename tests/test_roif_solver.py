"""
Tests for roif.roif_solver.

The suite verifies the complete ROIF decision pipeline:

- solver configuration and immutable value objects;
- tensor and cascade construction;
- scenario generation;
- counterfactual selection policies;
- Non-Fonit and cascade-risk vetoes;
- analysis-only, recommendation, and strict-safety modes;
- human authorization;
- complete deterministic solve_roif execution;
- summaries and actionability.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.roif_entities import (
    CapacityState,
    EntityKind,
    FunctionalChannel,
    FunctionalRole,
    InfluenceOperator,
    InfluencePlane,
    OperatorKind,
    PlaneKind,
    ROIFSystem,
    StructuralEntity,
    TemporalMode,
    Vector3,
)
from roif.roif_tensor import (
    SelfCouplingMode,
    TensorBuildConfig,
    build_capacity_tensor,
)
from roif.roif_cascade import (
    CascadeConfig,
    CascadeDirection,
    CascadeUpdateMode,
    run_cascade,
)
from roif.roif_root_detector import (
    InterventionPolicy,
    RootDetectorConfig,
)
from roif.roif_counterfactual import (
    CounterfactualAction,
    CounterfactualConfig,
    CounterfactualExecutionMode,
    CounterfactualIntervention,
    CounterfactualObjective,
    CounterfactualScenario,
    evaluate_counterfactual_batch,
    generate_single_channel_scenarios,
)
from roif.roif_solver import (
    NonFonitGate,
    ROIFSolution,
    ROIFSolverError,
    ScenarioGenerationMode,
    SelectionPolicy,
    SolverConfig,
    SolverDecision,
    SolverEvidence,
    SolverMode,
    SolverStatus,
    SolverThresholds,
    VetoReason,
    build_solver_evidence,
    build_solver_tensor,
    build_solver_trajectory,
    decision_from_result,
    empty_solver_decision,
    evaluate_non_fonit_gate,
    generate_solver_scenarios,
    safe_candidates,
    scenario_passes_thresholds,
    scenario_veto_reason,
    select_solver_result,
    solution_is_actionable,
    solution_summary,
    solve_analysis_only,
    solve_roif,
    validate_solver_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_channel(
    channel_id: str,
    *,
    capacity: float = 10.0,
    load: float = 0.0,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=f"entity_{channel_id}",
        name=channel_id,
        role=FunctionalRole.TRANSMITTER,
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(
            capacity=capacity,
            load=load,
        ),
    )


def make_operator(
    operator_id: str,
    source: str,
    target: str,
    gain: float,
) -> InfluenceOperator:
    return InfluenceOperator(
        operator_id=operator_id,
        name=operator_id,
        plane_id="mechanical",
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=(source,),
        target_ids=(target,),
        gain=gain,
    )


def make_system() -> ROIFSystem:
    channels = (
        make_channel("a"),
        make_channel("b"),
        make_channel("c"),
    )

    entities = tuple(
        StructuralEntity(
            entity_id=channel.entity_id,
            name=channel.entity_id,
            kind=EntityKind.GENERIC,
            channels=(channel,),
        )
        for channel in channels
    )

    return ROIFSystem(
        system_id="solver_test",
        name="Solver test",
        entities=entities,
        planes=(
            InfluencePlane(
                plane_id="mechanical",
                name="Mechanical",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.STATIC,
            ),
        ),
        operators=(
            make_operator("a_to_b", "a", "b", 0.8),
            make_operator("b_to_c", "b", "c", 0.6),
        ),
    )


def neutral_tensor_config() -> TensorBuildConfig:
    return TensorBuildConfig(
        tensor_ceiling=100.0,
        source_reserve_exponent=0.0,
        target_reserve_exponent=0.0,
        availability_exponent=0.0,
        geometry_exponent=0.0,
        material_exponent=0.0,
        history_exponent=0.0,
        state_dependent=False,
        self_coupling_mode=SelfCouplingMode.NONE,
    )


def simple_cascade_config(steps: int = 4) -> CascadeConfig:
    return CascadeConfig(
        steps=steps,
        update_mode=CascadeUpdateMode.LINEAR,
        direction=CascadeDirection.GENERIC,
        retention=0.0,
        dissipation=0.0,
        lower_bound=0.0,
        upper_bound=100.0,
        clip_state=False,
        convergence_tolerance=1e-12,
        convergence_patience=steps + 2,
        record_transmission_events=False,
    )


def solver_config(**overrides: object) -> SolverConfig:
    values: dict[str, object] = {
        "tensor_config": neutral_tensor_config(),
        "cascade_config": simple_cascade_config(),
        "root_config": RootDetectorConfig(),
        "counterfactual_config": CounterfactualConfig(
            execution_mode=CounterfactualExecutionMode.MATRIX_ONLY,
            objective=CounterfactualObjective.BALANCED_UTILITY,
            mark_dominated=False,
        ),
        "non_fonit_gate": NonFonitGate(
            enabled=True,
            human_authorization=False,
            bounded_scope=True,
            reversible=True,
        ),
    }
    values.update(overrides)
    return SolverConfig(**values)


def build_fixture():
    system = make_system()
    config = solver_config()

    tensor = build_capacity_tensor(
        system,
        config=config.tensor_config,
    )
    trajectory = run_cascade(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        tensor_config=config.tensor_config,
        cascade_config=config.cascade_config,
    )
    return system, tensor, trajectory, config


def make_scenario(
    scenario_id: str = "provided",
    *,
    channel_id: str = "a",
    cost: float | None = None,
    risk: float | None = None,
    uncertainty: float | None = None,
    irreversibility: float | None = None,
) -> CounterfactualScenario:
    return CounterfactualScenario(
        scenario_id=scenario_id,
        name=scenario_id,
        interventions=(
            CounterfactualIntervention(
                intervention_id=f"intervention:{scenario_id}",
                channel_id=channel_id,
                action=(
                    CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
                ),
                cost=cost,
                safety_risk=risk,
                uncertainty=uncertainty,
                irreversibility=irreversibility,
            ),
        ),
    )


def build_batch(
    *,
    scenarios: tuple[CounterfactualScenario, ...] | None = None,
    counterfactual_config: CounterfactualConfig | None = None,
):
    system, tensor, trajectory, _ = build_fixture()
    scenarios = scenarios or generate_single_channel_scenarios(
        system.channel_ids
    )
    batch = evaluate_counterfactual_batch(
        system,
        tensor,
        trajectory,
        scenarios,
        counterfactual_config
        or CounterfactualConfig(mark_dominated=False),
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_cascade_config(),
    )
    return system, tensor, trajectory, batch


# ---------------------------------------------------------------------------
# Config and immutable objects
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_immutable() -> None:
    config = SolverConfig()

    assert config.mode is SolverMode.ANALYSIS_ONLY
    assert config.scenario_generation is (
        ScenarioGenerationMode.ALL_CHANNELS
    )
    assert isinstance(config.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        config.mode = SolverMode.RECOMMENDATION  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("generated_magnitude", -0.1),
        ("generated_magnitude", 1.1),
    ],
)
def test_config_rejects_invalid_generated_magnitude(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ROIFSolverError):
        SolverConfig(**{field_name: value})


def test_thresholds_reject_invalid_values() -> None:
    with pytest.raises(ROIFSolverError):
        SolverThresholds(maximum_scope_risk=1.1)


def test_non_fonit_gate_rejects_invalid_risk() -> None:
    with pytest.raises(ROIFSolverError):
        NonFonitGate(scope_risk=-0.1)


def test_solver_evidence_is_immutable() -> None:
    evidence = SolverEvidence(
        evidence_id="x",
        category="safety",
        value=0.1,
        threshold=0.2,
        passed=True,
        description="Evidence.",
    )

    assert isinstance(evidence.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        evidence.value = 0.2  # type: ignore[misc]


def test_empty_solver_decision() -> None:
    decision = empty_solver_decision(
        VetoReason.NO_SAFE_SCENARIO
    )

    assert decision.scenario_id is None
    assert decision.safe is False
    assert decision.executable is False
    assert decision.veto_reason is VetoReason.NO_SAFE_SCENARIO


# ---------------------------------------------------------------------------
# Input validation and baseline builders
# ---------------------------------------------------------------------------


def test_validate_solver_inputs_accepts_matching_data() -> None:
    system, tensor, trajectory, _ = build_fixture()

    validate_solver_inputs(system, tensor, trajectory)


def test_build_solver_tensor() -> None:
    system = make_system()
    config = solver_config()

    tensor = build_solver_tensor(
        system,
        config,
        context=None,
        materials={},
    )

    assert tensor.channel_ids == system.channel_ids


def test_build_solver_trajectory() -> None:
    system = make_system()
    config = solver_config()

    trajectory = build_solver_trajectory(
        system,
        config,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        context=None,
        materials={},
    )

    assert trajectory.channel_ids == system.channel_ids
    assert trajectory.step_count == 4


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


def test_provided_only_requires_scenarios() -> None:
    system, tensor, trajectory, config = build_fixture()
    root_result = solve_roif(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=config,
        tensor=tensor,
        trajectory=trajectory,
    ).root_result

    with pytest.raises(ROIFSolverError):
        generate_solver_scenarios(
            system,
            root_result,
            (),
            solver_config(
                scenario_generation=(
                    ScenarioGenerationMode.PROVIDED_ONLY
                )
            ),
        )


def test_provided_only_returns_provided_scenarios() -> None:
    system, _, _, config = build_fixture()
    solution = solve_roif(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=config,
    )
    provided = (make_scenario("one"),)

    scenarios = generate_solver_scenarios(
        system,
        solution.root_result,
        provided,
        solver_config(
            scenario_generation=(
                ScenarioGenerationMode.PROVIDED_ONLY
            )
        ),
    )

    assert scenarios == provided


def test_all_channels_generates_one_scenario_per_channel() -> None:
    system, _, _, config = build_fixture()
    root_result = solve_roif(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=config,
    ).root_result

    scenarios = generate_solver_scenarios(
        system,
        root_result,
        (),
        config,
    )

    assert len(scenarios) == 3


def test_node_star_only_generates_one_scenario() -> None:
    system, _, _, config = build_fixture()
    root_result = solve_roif(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=config,
    ).root_result

    scenarios = generate_solver_scenarios(
        system,
        root_result,
        (),
        solver_config(
            scenario_generation=(
                ScenarioGenerationMode.NODE_STAR_ONLY
            )
        ),
    )

    assert len(scenarios) == 1
    assert (
        scenarios[0].interventions[0].channel_id
        == root_result.node_star
    )


def test_root_and_node_star_deduplicates_same_channel() -> None:
    system, _, _, config = build_fixture()
    root_result = solve_roif(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=config,
    ).root_result

    scenarios = generate_solver_scenarios(
        system,
        root_result,
        (),
        solver_config(
            scenario_generation=(
                ScenarioGenerationMode.ROOT_AND_NODE_STAR
            )
        ),
    )

    expected = len(
        {
            root_result.d_root,
            root_result.node_star,
        }
    )
    assert len(scenarios) == expected


# ---------------------------------------------------------------------------
# Thresholds and selection
# ---------------------------------------------------------------------------


def test_scenario_passes_default_thresholds() -> None:
    _, _, _, batch = build_batch()
    result = batch.results[0]

    assert scenario_passes_thresholds(
        result,
        SolverThresholds(),
        require_positive_reduction=False,
    ) is result.metrics.safe


def test_positive_reduction_requirement_filters_zero_gain() -> None:
    _, _, _, batch = build_batch()
    result = min(
        batch.results,
        key=lambda item: item.metrics.relative_reduction,
    )

    accepted = scenario_passes_thresholds(
        result,
        SolverThresholds(),
        require_positive_reduction=True,
    )

    assert accepted is (
        result.metrics.safe
        and result.metrics.relative_reduction > 0.0
    )


def test_safe_candidates_returns_only_accepted_results() -> None:
    _, _, _, batch = build_batch()
    config = solver_config(
        require_positive_reduction=True,
    )

    accepted = safe_candidates(batch, config)

    assert all(
        item.metrics.safe
        and item.metrics.relative_reduction > 0.0
        for item in accepted
    )


@pytest.mark.parametrize(
    "policy",
    [
        SelectionPolicy.BEST_RANKED,
        SelectionPolicy.BEST_SAFE,
        SelectionPolicy.PARETO_SAFE,
        SelectionPolicy.MAXIMUM_REDUCTION,
        SelectionPolicy.MINIMUM_RISK,
    ],
)
def test_selection_policies_return_valid_result_or_none(
    policy: SelectionPolicy,
) -> None:
    _, _, _, batch = build_batch()

    selected = select_solver_result(
        batch,
        solver_config(selection_policy=policy),
    )

    assert selected is None or selected in batch.results


def test_maximum_reduction_selects_largest_reduction() -> None:
    _, _, _, batch = build_batch()

    selected = select_solver_result(
        batch,
        solver_config(
            selection_policy=SelectionPolicy.MAXIMUM_REDUCTION,
        ),
    )

    accepted = safe_candidates(
        batch,
        solver_config(
            selection_policy=SelectionPolicy.MAXIMUM_REDUCTION,
        ),
    )
    if accepted:
        assert selected is not None
        assert selected.metrics.relative_reduction == pytest.approx(
            max(
                item.metrics.relative_reduction
                for item in accepted
            )
        )


# ---------------------------------------------------------------------------
# Non-Fonit gate and vetoes
# ---------------------------------------------------------------------------


def test_disabled_non_fonit_gate_passes() -> None:
    passed, reason, evidence = evaluate_non_fonit_gate(
        NonFonitGate(enabled=False),
        SolverThresholds(),
    )

    assert passed is True
    assert reason is VetoReason.NONE
    assert evidence == ()


def test_scope_risk_triggers_non_fonit_veto() -> None:
    passed, reason, evidence = evaluate_non_fonit_gate(
        NonFonitGate(scope_risk=0.5),
        SolverThresholds(maximum_scope_risk=0.1),
    )

    assert passed is False
    assert reason is VetoReason.NON_FONIT_SCOPE
    assert evidence


def test_unbounded_scope_triggers_veto() -> None:
    passed, reason, _ = evaluate_non_fonit_gate(
        NonFonitGate(bounded_scope=False),
        SolverThresholds(),
    )

    assert passed is False
    assert reason is VetoReason.NON_FONIT_SCOPE


def test_irreversible_without_authorization_triggers_veto() -> None:
    passed, reason, _ = evaluate_non_fonit_gate(
        NonFonitGate(
            reversible=False,
            human_authorization=False,
        ),
        SolverThresholds(),
    )

    assert passed is False
    assert reason is VetoReason.IRREVERSIBILITY


def test_scenario_veto_reason_is_none_for_safe_result() -> None:
    _, _, _, batch = build_batch()
    safe_result = next(
        item
        for item in batch.results
        if item.metrics.safe
    )

    assert scenario_veto_reason(
        safe_result,
        solver_config(),
    ) is VetoReason.NONE


# ---------------------------------------------------------------------------
# Decision and evidence
# ---------------------------------------------------------------------------


def test_decision_from_result_analysis_only_is_not_executable() -> None:
    _, _, _, batch = build_batch()
    result = next(
        item
        for item in batch.results
        if item.metrics.safe
    )

    decision = decision_from_result(
        result,
        solver_config(mode=SolverMode.ANALYSIS_ONLY),
        gate_passed=True,
        gate_reason=VetoReason.NONE,
    )

    assert decision.safe is True
    assert decision.executable is False


def test_decision_requires_human_authorization_for_execution() -> None:
    _, _, _, batch = build_batch()
    result = next(
        item
        for item in batch.results
        if item.metrics.safe
    )

    unauthorized = decision_from_result(
        result,
        solver_config(
            mode=SolverMode.RECOMMENDATION,
            non_fonit_gate=NonFonitGate(
                human_authorization=False,
            ),
        ),
        gate_passed=True,
        gate_reason=VetoReason.NONE,
    )
    authorized = decision_from_result(
        result,
        solver_config(
            mode=SolverMode.RECOMMENDATION,
            non_fonit_gate=NonFonitGate(
                human_authorization=True,
            ),
        ),
        gate_passed=True,
        gate_reason=VetoReason.NONE,
    )

    assert unauthorized.executable is False
    assert authorized.executable is True


def test_build_solver_evidence_without_selection() -> None:
    evidence = build_solver_evidence(
        None,
        solver_config(),
        (),
    )

    assert len(evidence) == 1
    assert evidence[0].passed is False


# ---------------------------------------------------------------------------
# Complete solver pipeline
# ---------------------------------------------------------------------------


def test_solve_roif_returns_complete_solution() -> None:
    system = make_system()
    solution = solve_roif(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(),
    )

    assert isinstance(solution, ROIFSolution)
    assert solution.tensor.channel_ids == system.channel_ids
    assert solution.trajectory.channel_ids == system.channel_ids
    assert solution.root_result.d_root in system.channel_ids
    assert solution.counterfactual_batch.results
    assert solution.alternatives
    assert solution.evidence


def test_analysis_only_status_is_observation_only() -> None:
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(
            mode=SolverMode.ANALYSIS_ONLY,
        ),
    )

    assert solution.status is SolverStatus.OBSERVATION_ONLY
    assert solution.decision.executable is False


def test_recommendation_with_authorization_can_be_actionable() -> None:
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(
            mode=SolverMode.RECOMMENDATION,
            non_fonit_gate=NonFonitGate(
                human_authorization=True,
                bounded_scope=True,
                reversible=True,
            ),
        ),
    )

    assert solution.status in {
        SolverStatus.SOLVED,
        SolverStatus.NO_SAFE_SOLUTION,
        SolverStatus.VETOED,
    }
    if solution.status is SolverStatus.SOLVED:
        assert solution_is_actionable(solution) is True


def test_non_fonit_veto_blocks_solution() -> None:
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(
            mode=SolverMode.RECOMMENDATION,
            non_fonit_gate=NonFonitGate(
                scope_risk=1.0,
                human_authorization=True,
            ),
        ),
    )

    assert solution.status is SolverStatus.VETOED
    assert solution.decision.executable is False
    assert (
        solution.decision.veto_reason
        is VetoReason.NON_FONIT_SCOPE
    )


def test_no_safe_solution_when_threshold_is_impossible() -> None:
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(
            thresholds=SolverThresholds(
                minimum_relative_reduction=2.0,
            ),
        ),
    )

    assert solution.status is SolverStatus.NO_SAFE_SOLUTION
    assert solution.decision.scenario_id is None
    assert (
        solution.decision.veto_reason
        is VetoReason.NO_SAFE_SCENARIO
    )


def test_provided_scenario_pipeline() -> None:
    scenario = make_scenario("manual", channel_id="a")
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        scenarios=(scenario,),
        config=solver_config(
            scenario_generation=(
                ScenarioGenerationMode.PROVIDED_ONLY
            ),
        ),
    )

    assert len(solution.counterfactual_batch.results) == 1
    assert (
        solution.counterfactual_batch.results[0].scenario.scenario_id
        == "manual"
    )


def test_precomputed_tensor_and_trajectory_are_reused() -> None:
    system, tensor, trajectory, config = build_fixture()

    solution = solve_roif(
        system,
        tensor=tensor,
        trajectory=trajectory,
        config=config,
    )

    assert solution.tensor is tensor
    assert solution.trajectory is trajectory


def test_solver_is_deterministic() -> None:
    system = make_system()
    config = solver_config()
    kwargs = {
        "initial_state": {
            "a": 1.0,
            "b": 0.0,
            "c": 0.0,
        },
        "config": config,
    }

    left = solve_roif(system, **kwargs)
    right = solve_roif(system, **kwargs)

    assert left.status is right.status
    assert left.decision == right.decision
    assert left.root_result == right.root_result
    assert tuple(
        item.scenario.scenario_id
        for item in left.counterfactual_batch.results
    ) == tuple(
        item.scenario.scenario_id
        for item in right.counterfactual_batch.results
    )


def test_solve_analysis_only_forces_analysis_mode() -> None:
    solution = solve_analysis_only(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(
            mode=SolverMode.RECOMMENDATION,
            non_fonit_gate=NonFonitGate(
                human_authorization=True,
            ),
        ),
    )

    assert solution.status is SolverStatus.OBSERVATION_ONLY
    assert solution.decision.executable is False


def test_solution_summary_is_read_only() -> None:
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(),
    )
    summary = solution_summary(solution)

    assert isinstance(summary, MappingProxyType)
    assert summary["status"] == solution.status.value
    assert summary["d_root"] == solution.root_result.d_root
    assert summary["node_star"] == solution.root_result.node_star


def test_analysis_solution_is_not_actionable() -> None:
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(
            mode=SolverMode.ANALYSIS_ONLY,
        ),
    )

    assert solution_is_actionable(solution) is False


def test_solver_does_not_require_hidden_truth_labels() -> None:
    solution = solve_roif(
        make_system(),
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        config=solver_config(),
    )

    assert solution.root_result.d_origin in solution.tensor.channel_ids
    assert solution.root_result.d_fast in solution.tensor.channel_ids
    assert solution.root_result.d_root in solution.tensor.channel_ids
    assert solution.root_result.node_star in solution.tensor.channel_ids
