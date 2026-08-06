"""
End-to-end integration tests for the complete ROIF Engine pipeline.
"""

from __future__ import annotations

from types import MappingProxyType

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
from roif.roif_tensor import SelfCouplingMode, TensorBuildConfig, build_capacity_tensor
from roif.roif_cascade import (
    CascadeConfig,
    CascadeDirection,
    CascadeTrajectory,
    CascadeUpdateMode,
    run_cascade,
)
from roif.roif_root_detector import ROIFRootResult, RootDetectorConfig
from roif.roif_counterfactual import (
    CounterfactualAction,
    CounterfactualConfig,
    CounterfactualExecutionMode,
    CounterfactualObjective,
)
from roif.roif_solver import (
    NonFonitGate,
    ROIFSolution,
    ScenarioGenerationMode,
    SelectionPolicy,
    SolverConfig,
    SolverMode,
    SolverStatus,
    VetoReason,
    solution_is_actionable,
    solution_summary,
    solve_roif,
)


def make_channel(channel_id: str, *, capacity: float, load: float) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=f"entity_{channel_id}",
        name=channel_id,
        role=FunctionalRole.TRANSMITTER,
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(capacity=capacity, load=load),
    )


def make_operator(
    operator_id: str,
    *,
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


def make_end_to_end_system() -> ROIFSystem:
    channels = (
        make_channel("source", capacity=10.0, load=1.0),
        make_channel("compensator", capacity=5.0, load=2.0),
        make_channel("terminal", capacity=3.0, load=2.5),
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
        system_id="roif_end_to_end",
        name="ROIF end-to-end integration system",
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
            make_operator(
                "source_to_compensator",
                source="source",
                target="compensator",
                gain=0.8,
            ),
            make_operator(
                "compensator_to_terminal",
                source="compensator",
                target="terminal",
                gain=0.7,
            ),
        ),
    )


def tensor_config() -> TensorBuildConfig:
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


def cascade_config() -> CascadeConfig:
    return CascadeConfig(
        steps=5,
        update_mode=CascadeUpdateMode.LINEAR,
        direction=CascadeDirection.GENERIC,
        retention=0.0,
        dissipation=0.0,
        lower_bound=0.0,
        upper_bound=100.0,
        clip_state=False,
        convergence_tolerance=1e-12,
        convergence_patience=8,
        record_transmission_events=True,
        stop_on_failure=False,
    )


def solver_config(
    *,
    mode: SolverMode = SolverMode.ANALYSIS_ONLY,
    human_authorization: bool = False,
    scope_risk: float = 0.0,
) -> SolverConfig:
    return SolverConfig(
        mode=mode,
        scenario_generation=ScenarioGenerationMode.ALL_CHANNELS,
        selection_policy=SelectionPolicy.BEST_SAFE,
        generated_action=CounterfactualAction.REMOVE_OUTGOING_INFLUENCE,
        generated_magnitude=1.0,
        tensor_config=tensor_config(),
        cascade_config=cascade_config(),
        root_config=RootDetectorConfig(),
        counterfactual_config=CounterfactualConfig(
            execution_mode=CounterfactualExecutionMode.MATRIX_ONLY,
            objective=CounterfactualObjective.BALANCED_UTILITY,
            mark_dominated=False,
        ),
        non_fonit_gate=NonFonitGate(
            enabled=True,
            scope_risk=scope_risk,
            externality_risk=0.0,
            propagation_uncertainty=0.0,
            human_authorization=human_authorization,
            bounded_scope=True,
            reversible=True,
        ),
        require_safe_solution=True,
        require_positive_reduction=False,
        include_all_ranked_alternatives=True,
    )


def initial_state() -> dict[str, float]:
    return {
        "source": 1.0,
        "compensator": 0.0,
        "terminal": 0.0,
    }


def test_complete_roif_pipeline_end_to_end() -> None:
    system = make_end_to_end_system()

    solution = solve_roif(
        system,
        initial_state=initial_state(),
        config=solver_config(),
    )

    assert isinstance(solution, ROIFSolution)
    assert isinstance(solution.trajectory, CascadeTrajectory)
    assert isinstance(solution.root_result, ROIFRootResult)

    assert solution.tensor.channel_ids == system.channel_ids
    assert solution.trajectory.channel_ids == system.channel_ids

    assert solution.root_result.d_origin in system.channel_ids
    assert solution.root_result.d_fast in system.channel_ids
    assert solution.root_result.d_root in system.channel_ids
    assert solution.root_result.node_star in system.channel_ids

    assert len(solution.counterfactual_batch.results) == len(system.channel_ids)
    assert solution.decision.scenario_id is not None
    assert solution.decision.node_ids
    assert solution.decision.action_ids

    assert solution.status is SolverStatus.OBSERVATION_ONLY
    assert solution.decision.safe is True
    assert solution.decision.executable is False
    assert solution.decision.veto_reason is VetoReason.NONE

    assert solution.alternatives
    assert solution.evidence


def test_end_to_end_pipeline_reuses_precomputed_objects() -> None:
    system = make_end_to_end_system()
    config = solver_config()

    tensor = build_capacity_tensor(system, config=config.tensor_config)
    trajectory = run_cascade(
        system,
        initial_state=initial_state(),
        tensor_config=config.tensor_config,
        cascade_config=config.cascade_config,
    )

    solution = solve_roif(
        system,
        tensor=tensor,
        trajectory=trajectory,
        config=config,
    )

    assert solution.tensor is tensor
    assert solution.trajectory is trajectory


def test_end_to_end_pipeline_is_deterministic() -> None:
    system = make_end_to_end_system()
    config = solver_config()

    left = solve_roif(system, initial_state=initial_state(), config=config)
    right = solve_roif(system, initial_state=initial_state(), config=config)

    assert left.status is right.status
    assert left.decision == right.decision
    assert left.root_result == right.root_result

    assert tuple(
        result.scenario.scenario_id
        for result in left.counterfactual_batch.results
    ) == tuple(
        result.scenario.scenario_id
        for result in right.counterfactual_batch.results
    )


def test_authorized_recommendation_can_be_actionable() -> None:
    solution = solve_roif(
        make_end_to_end_system(),
        initial_state=initial_state(),
        config=solver_config(
            mode=SolverMode.RECOMMENDATION,
            human_authorization=True,
        ),
    )

    assert solution.status is SolverStatus.SOLVED
    assert solution.decision.safe is True
    assert solution.decision.executable is True
    assert solution.decision.veto_reason is VetoReason.NONE
    assert solution_is_actionable(solution) is True


def test_non_fonit_gate_vetoes_complete_pipeline() -> None:
    solution = solve_roif(
        make_end_to_end_system(),
        initial_state=initial_state(),
        config=solver_config(
            mode=SolverMode.RECOMMENDATION,
            human_authorization=True,
            scope_risk=1.0,
        ),
    )

    assert solution.status is SolverStatus.VETOED
    assert solution.decision.safe is False
    assert solution.decision.executable is False
    assert solution.decision.veto_reason is VetoReason.NON_FONIT_SCOPE
    assert solution_is_actionable(solution) is False


def test_end_to_end_solution_summary_is_consistent() -> None:
    solution = solve_roif(
        make_end_to_end_system(),
        initial_state=initial_state(),
        config=solver_config(),
    )

    summary = solution_summary(solution)

    assert isinstance(summary, MappingProxyType)
    assert summary["status"] == solution.status.value
    assert summary["scenario_id"] == solution.decision.scenario_id
    assert summary["d_origin"] == solution.root_result.d_origin
    assert summary["d_fast"] == solution.root_result.d_fast
    assert summary["d_root"] == solution.root_result.d_root
    assert summary["node_star"] == solution.root_result.node_star
    assert summary["alternative_count"] == len(solution.alternatives)
    assert summary["evidence_count"] == len(solution.evidence)


def test_end_to_end_pipeline_uses_no_hidden_truth_labels() -> None:
    solution = solve_roif(
        make_end_to_end_system(),
        initial_state=initial_state(),
        config=solver_config(),
    )

    derived_roles = {
        solution.root_result.d_origin,
        solution.root_result.d_fast,
        solution.root_result.d_root,
        solution.root_result.node_star,
    }

    assert derived_roles
    assert derived_roles.issubset(set(solution.tensor.channel_ids))
