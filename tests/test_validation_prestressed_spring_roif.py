"""
Mechanical Validation — Scenario 02R
Physics-to-ROIF Validation for the Pre-Stressed Parallel Spring Benchmark

This suite compares ROIF inference against an independently validated
mechanical benchmark.

Strict physics-grounded labels:
    D_origin = primary_branch_stiffness_loss
    D_fast   = bypass_branch

Exploratory ROIF-specific roles:
    D_root
    Node*

Important:
    D_root and Node* are not forced to match preselected values because
    they are ROIF-specific structural/counterfactual roles rather than
    standard quantities from elementary spring mechanics.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from roif.roif_cascade import (
    CascadeTrajectory,
    run_cascade,
)
from roif.roif_solver import (
    ROIFSolution,
    SolverStatus,
    solution_summary,
    solve_roif,
)
from roif.roif_tensor import (
    CapacityTensor,
    build_capacity_tensor,
)

from validation.mechanical.prestressed_spring_ground_truth import (
    MechanicalGroundTruth,
    assert_ground_truth_consistency,
    build_physics_ground_truth,
)
from validation.mechanical.prestressed_spring_roif_case import (
    PhysicsDerivedLabels,
    PrestressedSpringROIFCase,
    build_cascade_config,
    build_expected_labels,
    build_initial_state,
    build_prestressed_spring_roif_case,
    build_roif_system_from_physics,
    build_solver_config,
    build_tensor_config,
    build_validation_scenarios,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def physics() -> MechanicalGroundTruth:
    gt = build_physics_ground_truth()
    assert_ground_truth_consistency(gt)
    return gt


@pytest.fixture(scope="module")
def case() -> PrestressedSpringROIFCase:
    return build_prestressed_spring_roif_case()


@pytest.fixture(scope="module")
def tensor(
    case: PrestressedSpringROIFCase,
) -> CapacityTensor:
    return build_capacity_tensor(
        case.system,
        config=case.tensor_config,
    )


@pytest.fixture(scope="module")
def trajectory(
    case: PrestressedSpringROIFCase,
) -> CascadeTrajectory:
    return run_cascade(
        case.system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )


@pytest.fixture(scope="module")
def solution(
    case: PrestressedSpringROIFCase,
) -> ROIFSolution:
    return solve_roif(
        case.system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channels(
    case: PrestressedSpringROIFCase,
):
    return {
        channel.channel_id: channel
        for entity in case.system.entities
        for channel in entity.channels
    }


def _operators(
    case: PrestressedSpringROIFCase,
):
    return {
        operator.operator_id: operator
        for operator in case.system.operators
    }


def _roles(
    solution: ROIFSolution,
) -> dict[str, str]:
    root = solution.root_result

    return {
        "d_origin": root.d_origin,
        "d_fast": root.d_fast,
        "d_root": root.d_root,
        "node_star": root.node_star,
    }


# ---------------------------------------------------------------------------
# Case contract
# ---------------------------------------------------------------------------


def test_case_contract_is_complete(
    case: PrestressedSpringROIFCase,
) -> None:
    assert (
        case.case_id
        == "mechanical_02R_prestressed_spring_roif"
    )

    assert case.physics is not None
    assert case.system is not None
    assert case.initial_state
    assert case.scenarios
    assert case.expected is not None


def test_case_is_explicitly_physics_grounded(
    case: PrestressedSpringROIFCase,
) -> None:
    assert case.metadata["domain"] == "mechanical"

    assert (
        case.metadata[
            "roif_used_to_generate_ground_truth"
        ]
        is False
    )

    assert (
        case.metadata["biological_semantics"]
        is False
    )


def test_case_links_to_02p_ground_truth(
    case: PrestressedSpringROIFCase,
) -> None:
    assert (
        case.metadata[
            "physics_ground_truth_case"
        ]
        == "mechanical_02P_parallel_spring_ground_truth"
    )


def test_strict_and_exploratory_role_boundary_is_explicit(
    case: PrestressedSpringROIFCase,
) -> None:
    assert case.metadata[
        "strict_physics_labels"
    ] == (
        "d_origin",
        "d_fast",
    )

    assert case.metadata[
        "exploratory_roif_labels"
    ] == (
        "d_root",
        "node_star",
    )


# ---------------------------------------------------------------------------
# Physics -> ROIF mapping
# ---------------------------------------------------------------------------


def test_roif_channel_set_is_expected(
    case: PrestressedSpringROIFCase,
) -> None:
    assert case.channel_ids == (
        "primary_branch_stiffness_loss",
        "primary_branch",
        "bypass_branch",
        "load_sharing_junction",
        "downstream_member",
    )


def test_primary_capacity_comes_from_physical_allowable_force(
    case: PrestressedSpringROIFCase,
    physics: MechanicalGroundTruth,
) -> None:
    channel = _channels(case)[
        "primary_branch"
    ]

    assert (
        channel.capacity_state.capacity
        == pytest.approx(
            physics.defective_primary.allowable_force_n
        )
    )


def test_primary_load_comes_from_physical_force(
    case: PrestressedSpringROIFCase,
    physics: MechanicalGroundTruth,
) -> None:
    channel = _channels(case)[
        "primary_branch"
    ]

    assert (
        channel.capacity_state.load
        == pytest.approx(
            physics.defective_state.primary_force_n
        )
    )


def test_bypass_capacity_comes_from_physical_allowable_force(
    case: PrestressedSpringROIFCase,
    physics: MechanicalGroundTruth,
) -> None:
    channel = _channels(case)[
        "bypass_branch"
    ]

    assert (
        channel.capacity_state.capacity
        == pytest.approx(
            physics.bypass.allowable_force_n
        )
    )


def test_bypass_load_comes_from_physical_force(
    case: PrestressedSpringROIFCase,
    physics: MechanicalGroundTruth,
) -> None:
    channel = _channels(case)[
        "bypass_branch"
    ]

    assert (
        channel.capacity_state.load
        == pytest.approx(
            physics.defective_state.bypass_force_n
        )
    )


def test_downstream_capacity_and_load_come_from_physics(
    case: PrestressedSpringROIFCase,
    physics: MechanicalGroundTruth,
) -> None:
    channel = _channels(case)[
        "downstream_member"
    ]

    assert (
        channel.capacity_state.capacity
        == pytest.approx(
            physics.downstream.allowable_force_n
        )
    )

    assert (
        channel.capacity_state.load
        == pytest.approx(
            physics.defective_state.downstream_force_n
        )
    )


def test_bypass_is_physically_over_capacity_in_roif_input(
    case: PrestressedSpringROIFCase,
) -> None:
    bypass = _channels(case)[
        "bypass_branch"
    ].capacity_state

    assert bypass.load > bypass.capacity


def test_primary_is_physically_below_capacity_in_roif_input(
    case: PrestressedSpringROIFCase,
) -> None:
    primary = _channels(case)[
        "primary_branch"
    ].capacity_state

    assert primary.load < primary.capacity


def test_downstream_is_physically_below_capacity_in_roif_input(
    case: PrestressedSpringROIFCase,
) -> None:
    downstream = _channels(case)[
        "downstream_member"
    ].capacity_state

    assert downstream.load < downstream.capacity


# ---------------------------------------------------------------------------
# Stiffness-derived topology
# ---------------------------------------------------------------------------


def test_parallel_force_share_gains_are_derived_from_stiffness(
    case: PrestressedSpringROIFCase,
    physics: MechanicalGroundTruth,
) -> None:
    operators = _operators(case)

    primary_gain = operators[
        "primary_to_junction"
    ].gain

    bypass_gain = operators[
        "bypass_to_junction"
    ].gain

    k_primary = (
        physics.defective_primary.axial_stiffness_n_per_m
    )
    k_bypass = (
        physics.bypass.axial_stiffness_n_per_m
    )

    expected_primary = (
        k_primary / (k_primary + k_bypass)
    )

    expected_bypass = (
        k_bypass / (k_primary + k_bypass)
    )

    assert primary_gain == pytest.approx(
        expected_primary
    )

    assert bypass_gain == pytest.approx(
        expected_bypass
    )


def test_parallel_force_share_gains_sum_to_one(
    case: PrestressedSpringROIFCase,
) -> None:
    operators = _operators(case)

    total = (
        operators["primary_to_junction"].gain
        + operators["bypass_to_junction"].gain
    )

    assert total == pytest.approx(1.0)


def test_defect_targets_primary_branch_only(
    case: PrestressedSpringROIFCase,
) -> None:
    operator = _operators(case)[
        "defect_to_primary"
    ]

    assert operator.source_ids == (
        "primary_branch_stiffness_loss",
    )

    assert operator.target_ids == (
        "primary_branch",
    )


def test_downstream_series_transfer_has_unit_gain(
    case: PrestressedSpringROIFCase,
) -> None:
    operator = _operators(case)[
        "junction_to_downstream"
    ]

    assert operator.gain == pytest.approx(
        1.0
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_state_seeds_only_stiffness_defect(
    case: PrestressedSpringROIFCase,
) -> None:
    assert (
        case.initial_state[
            "primary_branch_stiffness_loss"
        ]
        == pytest.approx(1.0)
    )

    assert all(
        value == pytest.approx(0.0)
        for key, value in case.initial_state.items()
        if key != "primary_branch_stiffness_loss"
    )


def test_initial_state_is_read_only(
    case: PrestressedSpringROIFCase,
) -> None:
    assert isinstance(
        case.initial_state,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        case.initial_state[
            "primary_branch"
        ] = 1.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# Factory determinism
# ---------------------------------------------------------------------------


def test_roif_system_factory_is_deterministic(
    physics: MechanicalGroundTruth,
) -> None:
    left = build_roif_system_from_physics(
        physics
    )

    right = build_roif_system_from_physics(
        physics
    )

    assert left.channel_ids == right.channel_ids

    assert tuple(
        operator.operator_id
        for operator in left.operators
    ) == tuple(
        operator.operator_id
        for operator in right.operators
    )


def test_configuration_factories_are_deterministic() -> None:
    assert (
        build_tensor_config()
        == build_tensor_config()
    )

    assert (
        build_cascade_config()
        == build_cascade_config()
    )

    assert (
        build_solver_config()
        == build_solver_config()
    )


def test_scenario_factory_is_deterministic() -> None:
    assert (
        build_validation_scenarios()
        == build_validation_scenarios()
    )


def test_expected_label_factory_is_deterministic() -> None:
    assert (
        build_expected_labels()
        == build_expected_labels()
    )


def test_complete_case_factory_is_deterministic() -> None:
    left = build_prestressed_spring_roif_case()
    right = build_prestressed_spring_roif_case()

    assert left.case_id == right.case_id
    assert left.channel_ids == right.channel_ids
    assert left.initial_state == right.initial_state
    assert left.scenarios == right.scenarios
    assert left.expected == right.expected


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


def test_capacity_tensor_builds(
    case: PrestressedSpringROIFCase,
    tensor: CapacityTensor,
) -> None:
    assert isinstance(
        tensor,
        CapacityTensor,
    )

    assert tensor.channel_ids == case.channel_ids

    assert tensor.matrix.shape == (
        len(case.channel_ids),
        len(case.channel_ids),
    )


def test_tensor_is_deterministic(
    case: PrestressedSpringROIFCase,
) -> None:
    left = build_capacity_tensor(
        case.system,
        config=case.tensor_config,
    )

    right = build_capacity_tensor(
        case.system,
        config=case.tensor_config,
    )

    assert left.matrix.shape == right.matrix.shape
    assert (left.matrix == right.matrix).all()


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


def test_cascade_runs(
    trajectory: CascadeTrajectory,
) -> None:
    assert isinstance(
        trajectory,
        CascadeTrajectory,
    )

    assert trajectory.step_count > 0


def test_cascade_reaches_primary_branch(
    trajectory: CascadeTrajectory,
) -> None:
    assert (
        trajectory.peak("primary_branch")
        > 0.0
    )


def test_cascade_reaches_junction(
    trajectory: CascadeTrajectory,
) -> None:
    assert (
        trajectory.peak(
            "load_sharing_junction"
        )
        > 0.0
    )


def test_cascade_reaches_downstream_member(
    trajectory: CascadeTrajectory,
) -> None:
    assert (
        trajectory.peak(
            "downstream_member"
        )
        > 0.0
    )


def test_cascade_is_deterministic(
    case: PrestressedSpringROIFCase,
) -> None:
    left = run_cascade(
        case.system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )

    right = run_cascade(
        case.system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )

    assert (
        left.state_history.shape
        == right.state_history.shape
    )

    assert (
        left.state_history
        == right.state_history
    ).all()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def test_full_roif_pipeline_runs(
    solution: ROIFSolution,
) -> None:
    assert isinstance(
        solution,
        ROIFSolution,
    )

    assert solution.status in {
        SolverStatus.OBSERVATION_ONLY,
        SolverStatus.SOLVED,
        SolverStatus.NO_SAFE_SOLUTION,
        SolverStatus.VETOED,
    }

    assert solution.root_result is not None


def test_solver_produces_four_roles(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    roles = _roles(solution)

    assert set(roles) == {
        "d_origin",
        "d_fast",
        "d_root",
        "node_star",
    }

    assert all(
        value in case.channel_ids
        for value in roles.values()
    )


# ---------------------------------------------------------------------------
# Strict physics-grounded validation
# ---------------------------------------------------------------------------


def test_physics_ground_truth_declares_bypass_first_failure(
    physics: MechanicalGroundTruth,
) -> None:
    assert (
        physics.failure_sequence[
            0
        ].failed_member_id
        == "bypass_branch"
    )


def test_expected_d_fast_comes_from_physics(
    case: PrestressedSpringROIFCase,
) -> None:
    assert (
        case.expected.d_fast
        == "bypass_branch"
    )


def test_expected_d_origin_is_imposed_stiffness_defect(
    case: PrestressedSpringROIFCase,
) -> None:
    assert (
        case.expected.d_origin
        == "primary_branch_stiffness_loss"
    )


def test_roif_recovers_physics_grounded_d_origin(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    assert (
        solution.root_result.d_origin
        == case.expected.d_origin
    ), {
        "expected_d_origin": case.expected.d_origin,
        "inferred_d_origin": (
            solution.root_result.d_origin
        ),
        "all_roles": _roles(solution),
    }


def test_roif_recovers_physics_grounded_d_fast(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    """
    Main strict physics-to-ROIF assertion.

    The independently solved mechanical benchmark says that the bypass
    branch is the first physical overload after primary stiffness loss.
    """
    assert (
        solution.root_result.d_fast
        == case.expected.d_fast
    ), {
        "expected_d_fast": case.expected.d_fast,
        "inferred_d_fast": solution.root_result.d_fast,
        "all_roles": _roles(solution),
        "physics_primary_force_n": (
            case.physics.defective_state.primary_force_n
        ),
        "physics_primary_allowable_n": (
            case.physics.defective_primary.allowable_force_n
        ),
        "physics_bypass_force_n": (
            case.physics.defective_state.bypass_force_n
        ),
        "physics_bypass_allowable_n": (
            case.physics.bypass.allowable_force_n
        ),
    }


def test_strict_physics_roles_match_together(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    assert {
        "d_origin": solution.root_result.d_origin,
        "d_fast": solution.root_result.d_fast,
    } == {
        "d_origin": case.expected.d_origin,
        "d_fast": case.expected.d_fast,
    }


# ---------------------------------------------------------------------------
# Exploratory ROIF-specific roles
# ---------------------------------------------------------------------------


def test_d_root_is_reported_but_not_preloaded_as_physics_truth(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    assert (
        case.expected.d_root_hypothesis
        is None
    )

    assert (
        solution.root_result.d_root
        in case.channel_ids
    )


def test_node_star_is_reported_but_not_preloaded_as_physics_truth(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    assert (
        case.expected.node_star_hypothesis
        is None
    )

    assert (
        solution.root_result.node_star
        in case.channel_ids
    )


def test_root_result_exposes_rankings_for_audit(
    solution: ROIFSolution,
) -> None:
    root = solution.root_result

    assert root.origin_ranking is not None
    assert root.fast_ranking is not None
    assert root.root_ranking is not None
    assert root.node_star_ranking is not None


# ---------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------


def test_solver_evaluates_all_physical_counterfactuals(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    expected_ids = {
        scenario.scenario_id
        for scenario in case.scenarios
    }

    actual_ids = {
        result.scenario.scenario_id
        for result
        in solution.counterfactual_batch.results
    }

    assert actual_ids == expected_ids


def test_counterfactual_set_contains_defect_primary_and_bypass_actions(
    case: PrestressedSpringROIFCase,
) -> None:
    ids = {
        scenario.scenario_id
        for scenario in case.scenarios
    }

    assert ids == {
        "remove_stiffness_defect",
        "restore_primary_branch",
        "restore_bypass_branch",
    }


# ---------------------------------------------------------------------------
# Summary / determinism
# ---------------------------------------------------------------------------


def test_solution_summary_matches_root_result(
    solution: ROIFSolution,
) -> None:
    summary = solution_summary(
        solution
    )

    assert (
        summary["d_origin"]
        == solution.root_result.d_origin
    )

    assert (
        summary["d_fast"]
        == solution.root_result.d_fast
    )

    assert (
        summary["d_root"]
        == solution.root_result.d_root
    )

    assert (
        summary["node_star"]
        == solution.root_result.node_star
    )


def test_full_solver_result_is_deterministic(
    case: PrestressedSpringROIFCase,
    solution: ROIFSolution,
) -> None:
    repeated = solve_roif(
        case.system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    assert (
        repeated.root_result
        == solution.root_result
    )

    assert (
        repeated.decision
        == solution.decision
    )


def test_scenario_02r_end_to_end_physics_to_roif(
    case: PrestressedSpringROIFCase,
    physics: MechanicalGroundTruth,
    solution: ROIFSolution,
) -> None:
    """
    Final Scenario 02R invariant.

    1. Physics ground truth is independently valid.
    2. ROIF input is derived from that physics.
    3. Expected labels are not passed to Solver.
    4. ROIF must recover the strict physics-grounded origin and first failure.
    5. D_root and Node* remain exploratory outputs.
    """

    assert_ground_truth_consistency(
        physics
    )

    assert (
        case.metadata[
            "roif_used_to_generate_ground_truth"
        ]
        is False
    )

    assert (
        physics.failure_sequence[
            0
        ].failed_member_id
        == "bypass_branch"
    )

    assert (
        solution.root_result.d_origin
        == "primary_branch_stiffness_loss"
    )

    assert (
        solution.root_result.d_fast
        == "bypass_branch"
    )

    assert (
        solution.root_result.d_root
        in case.channel_ids
    )

    assert (
        solution.root_result.node_star
        in case.channel_ids
    )
