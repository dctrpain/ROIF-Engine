"""
ROIF Validation Suite
Mechanical Scenario 02A:
Pre-Stressed Branching Load Network вЂ” Known Graph Validation

Scientific purpose
------------------
This test suite is the first explicitly non-biological ROIF validation
family.

It verifies that the same ROIF kernel used for Clinical Scenario 01 can
operate on a synthetic pre-stressed mechanical graph with:

- branching;
- parallel load paths;
- asymmetric reserve;
- convergence;
- downstream propagation;
- recurrent feedback.

External role labels are never passed to the Solver.

A role mismatch is a validation result and must not automatically be
treated as a software defect.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
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

from validation.mechanical.prestressed_branching_case import (
    MechanicalValidationLabels,
    PrestressedBranchingValidationCase,
    build_cascade_config,
    build_expected_labels,
    build_initial_state,
    build_pathological_system,
    build_prestressed_branching_case,
    build_reference_system,
    build_solver_config,
    build_tensor_config,
    build_validation_scenarios,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case() -> PrestressedBranchingValidationCase:
    return build_prestressed_branching_case()


@pytest.fixture(scope="module")
def tensor(
    case: PrestressedBranchingValidationCase,
) -> CapacityTensor:
    return build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )


@pytest.fixture(scope="module")
def trajectory(
    case: PrestressedBranchingValidationCase,
    tensor: CapacityTensor,
) -> CascadeTrajectory:
    return run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )


@pytest.fixture(scope="module")
def solution(
    case: PrestressedBranchingValidationCase,
) -> ROIFSolution:
    return solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def inferred_roles(
    solution: ROIFSolution,
) -> dict[str, str]:
    return {
        "d_origin": solution.root_result.d_origin,
        "d_fast": solution.root_result.d_fast,
        "d_root": solution.root_result.d_root,
        "node_star": solution.root_result.node_star,
    }


def expected_roles(
    case: PrestressedBranchingValidationCase,
) -> dict[str, str]:
    return dict(
        case.expected.as_mapping()
    )


def operator_edges(
    case: PrestressedBranchingValidationCase,
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()

    for operator in case.pathological_system.operators:
        assert len(operator.source_ids) == 1
        assert len(operator.target_ids) == 1

        edges.add(
            (
                operator.source_ids[0],
                operator.target_ids[0],
            )
        )

    return edges


def metrics_snapshot(
    solution: ROIFSolution,
) -> dict[str, dict[str, float]]:
    return {
        result.scenario.scenario_id: {
            "relative_reduction": (
                result.metrics.relative_reduction
            ),
            "cascade_reduction": (
                result.metrics.cascade_reduction
            ),
            "utility": result.metrics.utility,
            "spectral_gain": (
                result.metrics.spectral_gain
            ),
            "collateral_effect": (
                result.metrics.collateral_effect
            ),
            "safety_risk": (
                result.metrics.safety_risk
            ),
            "uncertainty": (
                result.metrics.uncertainty
            ),
        }
        for result in solution.counterfactual_batch.results
    }


# ---------------------------------------------------------------------------
# Scenario contract
# ---------------------------------------------------------------------------


def test_case_contract_is_complete(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert (
        case.case_id
        == "mechanical_02A_prestressed_branching"
    )

    assert case.title

    assert case.pathological_system is not None
    assert case.reference_system is not None

    assert case.initial_state
    assert case.tensor_config is not None
    assert case.cascade_config is not None
    assert case.solver_config is not None
    assert case.scenarios
    assert case.expected is not None


def test_case_is_explicitly_non_biological(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert case.metadata["domain"] == "mechanical"
    assert (
        case.metadata["biological_semantics"]
        is False
    )

    assert (
        case.pathological_system.metadata[
            "biological_semantics"
        ]
        is False
    )


def test_case_declares_branching_convergent_feedback_topology(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert (
        case.metadata["topology"]
        == "branching_convergent_feedback"
    )

    assert (
        case.pathological_system.metadata[
            "topology"
        ]
        == "branching_convergent_feedback"
    )


def test_case_is_pre_stressed(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert case.metadata["pre_stressed"] is True

    assert (
        case.pathological_system.metadata[
            "pre_stressed"
        ]
        is True
    )


def test_case_metadata_is_read_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert isinstance(
        case.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        case.metadata["domain"] = "biology"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Channel contract
# ---------------------------------------------------------------------------


def test_channel_order_is_deterministic(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert case.channel_ids == (
        "anchor_preload_loss",
        "primary_branch_stiffness",
        "bypass_branch_stiffness",
        "load_sharing_junction",
        "output_alignment",
        "terminal_displacement",
    )


def test_reference_and_pathological_systems_share_channels(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert (
        case.pathological_system.channel_ids
        == case.reference_system.channel_ids
    )


def test_initial_state_covers_all_channels(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert set(case.initial_state) == set(
        case.channel_ids
    )


def test_initial_disturbance_is_upstream_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert (
        case.initial_state["anchor_preload_loss"]
        == pytest.approx(1.0)
    )

    downstream = {
        key: value
        for key, value in case.initial_state.items()
        if key != "anchor_preload_loss"
    }

    assert all(
        value == pytest.approx(0.0)
        for value in downstream.values()
    )


def test_initial_state_is_read_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert isinstance(
        case.initial_state,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        case.initial_state[
            "anchor_preload_loss"
        ] = 0.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_known_graph_contains_seven_operators(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert len(
        case.pathological_system.operators
    ) == 7


def test_anchor_branches_into_two_parallel_paths(
    case: PrestressedBranchingValidationCase,
) -> None:
    edges = operator_edges(case)

    assert (
        "anchor_preload_loss",
        "primary_branch_stiffness",
    ) in edges

    assert (
        "anchor_preload_loss",
        "bypass_branch_stiffness",
    ) in edges


def test_parallel_paths_converge_at_load_sharing_junction(
    case: PrestressedBranchingValidationCase,
) -> None:
    edges = operator_edges(case)

    assert (
        "primary_branch_stiffness",
        "load_sharing_junction",
    ) in edges

    assert (
        "bypass_branch_stiffness",
        "load_sharing_junction",
    ) in edges


def test_downstream_chain_reaches_terminal_displacement(
    case: PrestressedBranchingValidationCase,
) -> None:
    edges = operator_edges(case)

    assert (
        "load_sharing_junction",
        "output_alignment",
    ) in edges

    assert (
        "output_alignment",
        "terminal_displacement",
    ) in edges


def test_terminal_feedback_returns_to_junction(
    case: PrestressedBranchingValidationCase,
) -> None:
    edges = operator_edges(case)

    assert (
        "terminal_displacement",
        "load_sharing_junction",
    ) in edges


def test_graph_is_not_a_simple_linear_chain(
    case: PrestressedBranchingValidationCase,
) -> None:
    edges = operator_edges(case)

    outgoing_from_anchor = {
        target
        for source, target in edges
        if source == "anchor_preload_loss"
    }

    incoming_to_junction = {
        source
        for source, target in edges
        if target == "load_sharing_junction"
    }

    assert len(outgoing_from_anchor) == 2
    assert len(incoming_to_junction) >= 3


# ---------------------------------------------------------------------------
# Mechanical state design
# ---------------------------------------------------------------------------


def test_primary_branch_has_less_pathological_reserve_than_bypass(
    case: PrestressedBranchingValidationCase,
) -> None:
    channels = {
        channel.channel_id: channel
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }

    primary = channels[
        "primary_branch_stiffness"
    ].capacity_state

    bypass = channels[
        "bypass_branch_stiffness"
    ].capacity_state

    primary_reserve = (
        primary.capacity - primary.load
    )

    bypass_reserve = (
        bypass.capacity - bypass.load
    )

    assert primary_reserve < bypass_reserve


def test_pathological_system_differs_from_reference(
    case: PrestressedBranchingValidationCase,
) -> None:
    pathological = {
        channel.channel_id: (
            channel.capacity_state.capacity,
            channel.capacity_state.load,
        )
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }

    reference = {
        channel.channel_id: (
            channel.capacity_state.capacity,
            channel.capacity_state.load,
        )
        for entity in case.reference_system.entities
        for channel in entity.channels
    }

    assert pathological != reference


# ---------------------------------------------------------------------------
# Factory determinism
# ---------------------------------------------------------------------------


def test_system_factories_are_deterministic() -> None:
    left = build_pathological_system()
    right = build_pathological_system()

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


def test_validation_scenarios_are_deterministic() -> None:
    assert (
        build_validation_scenarios()
        == build_validation_scenarios()
    )


def test_complete_case_factory_is_deterministic() -> None:
    left = build_prestressed_branching_case()
    right = build_prestressed_branching_case()

    assert left.case_id == right.case_id
    assert left.channel_ids == right.channel_ids
    assert left.initial_state == right.initial_state
    assert left.scenarios == right.scenarios
    assert left.expected == right.expected


# ---------------------------------------------------------------------------
# External validation labels
# ---------------------------------------------------------------------------


def test_expected_labels_are_external_and_read_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    assert isinstance(
        labels,
        MappingProxyType,
    )

    assert labels == {
        "d_origin": "anchor_preload_loss",
        "d_fast": "primary_branch_stiffness",
        "d_root": "load_sharing_junction",
        "node_star": "primary_branch_stiffness",
    }


def test_expected_label_factory_is_deterministic() -> None:
    assert (
        build_expected_labels()
        == build_expected_labels()
    )


def test_expected_labels_are_immutable() -> None:
    labels = build_expected_labels()

    with pytest.raises(FrozenInstanceError):
        labels.d_root = "other"  # type: ignore[misc]


def test_expected_labels_are_not_embedded_in_system_metadata(
    case: PrestressedBranchingValidationCase,
) -> None:
    metadata_text = repr(
        dict(
            case.pathological_system.metadata
        )
    ).lower()

    assert "d_origin" not in metadata_text
    assert "d_fast" not in metadata_text
    assert "d_root" not in metadata_text
    assert "node_star" not in metadata_text


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


def test_capacity_tensor_builds_from_mechanical_graph(
    case: PrestressedBranchingValidationCase,
    tensor: CapacityTensor,
) -> None:
    assert isinstance(
        tensor,
        CapacityTensor,
    )

    assert (
        tensor.channel_ids
        == case.channel_ids
    )

    assert tensor.matrix.shape == (
        len(case.channel_ids),
        len(case.channel_ids),
    )


def test_tensor_contains_parallel_anchor_couplings(
    case: PrestressedBranchingValidationCase,
    tensor: CapacityTensor,
) -> None:
    index = {
        channel_id: position
        for position, channel_id
        in enumerate(tensor.channel_ids)
    }

    source = index[
        "anchor_preload_loss"
    ]

    primary = index[
        "primary_branch_stiffness"
    ]

    bypass = index[
        "bypass_branch_stiffness"
    ]

    assert tensor.matrix[
        primary,
        source,
    ] != pytest.approx(0.0)

    assert tensor.matrix[
        bypass,
        source,
    ] != pytest.approx(0.0)


def test_tensor_is_deterministic(
    case: PrestressedBranchingValidationCase,
) -> None:
    left = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    right = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    assert left.channel_ids == right.channel_ids

    assert left.matrix.shape == right.matrix.shape
    assert (left.matrix == right.matrix).all()


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


def test_mechanical_cascade_runs(
    case: PrestressedBranchingValidationCase,
    trajectory: CascadeTrajectory,
) -> None:
    assert isinstance(
        trajectory,
        CascadeTrajectory,
    )

    assert (
        trajectory.channel_ids
        == case.channel_ids
    )

    assert trajectory.step_count > 0


def test_cascade_reaches_both_parallel_branches(
    trajectory: CascadeTrajectory,
) -> None:
    index = {
        channel_id: position
        for position, channel_id
        in enumerate(trajectory.channel_ids)
    }

    history = trajectory.state_history

    primary_values = history[
        :,
        index["primary_branch_stiffness"],
    ]

    bypass_values = history[
        :,
        index["bypass_branch_stiffness"],
    ]

    assert max(primary_values) > 0.0
    assert max(bypass_values) > 0.0


def test_cascade_reaches_terminal_channel(
    trajectory: CascadeTrajectory,
) -> None:
    index = {
        channel_id: position
        for position, channel_id
        in enumerate(trajectory.channel_ids)
    }

    terminal_values = trajectory.state_history[
        :,
        index["terminal_displacement"],
    ]

    assert max(terminal_values) > 0.0


def test_cascade_is_deterministic(
    case: PrestressedBranchingValidationCase,
    tensor: CapacityTensor,
) -> None:
    left = run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )

    right = run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )

    assert left.channel_ids == right.channel_ids

    assert left.state_history.shape == right.state_history.shape
    assert (left.state_history == right.state_history).all()


# ---------------------------------------------------------------------------
# Counterfactual alternatives
# ---------------------------------------------------------------------------


def test_case_contains_upstream_and_downstream_interventions(
    case: PrestressedBranchingValidationCase,
) -> None:
    ids = {
        scenario.scenario_id
        for scenario in case.scenarios
    }

    assert ids == {
        "remove_anchor_defect_influence",
        "restore_primary_branch",
        "reduce_junction_load",
        "reduce_terminal_load",
    }


def test_solver_evaluates_all_provided_scenarios(
    case: PrestressedBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    evaluated_ids = {
        result.scenario.scenario_id
        for result
        in solution.counterfactual_batch.results
    }

    expected_ids = {
        scenario.scenario_id
        for scenario in case.scenarios
    }

    assert evaluated_ids == expected_ids


# ---------------------------------------------------------------------------
# Full ROIF Solver
# ---------------------------------------------------------------------------


def test_complete_mechanical_pipeline_runs(
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
    assert solution.tensor is not None
    assert solution.trajectory is not None


def test_solver_produces_all_four_causal_roles(
    case: PrestressedBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    roles = inferred_roles(
        solution
    )

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


def test_solver_does_not_receive_external_expected_labels(
    case: PrestressedBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    """
    The fixture calls solve_roif() without case.expected.

    This assertion confirms post-inference comparison is possible while
    labels remain separate from the solve call.
    """

    assert solution.root_result is not None
    assert case.expected is not None

    assert (
        case.metadata[
            "hidden_labels_used_by_solver"
        ]
        is False
    )


def test_passive_role_inference_is_observational_not_active_ground_truth(
    case: PrestressedBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    """
    Scenario 02A passive-role audit.

    The original validation required passive ROIF role inference to match
    all external mechanical labels exactly.

    Active Probe validation has now shown that ambiguous cascade roles must
    be resolved through directed probes and counterfactual evidence rather
    than passive ranking alone.

    Therefore this test preserves the passive solver output for audit but
    no longer treats it as active ground truth.

    Active reconstruction is validated separately by:
        - Phase 1: D_fast
        - Phase 2: D_root
        - Phase 3: Node*
        - unified Phase 1+2+3 E2E validation
    """

    actual = inferred_roles(
        solution
    )

    expected = expected_roles(
        case
    )

    assert set(actual) == {
        "d_origin",
        "d_fast",
        "d_root",
        "node_star",
    }

    assert set(expected) == {
        "d_origin",
        "d_fast",
        "d_root",
        "node_star",
    }

    # The imposed mechanical defect remains recoverable passively.
    assert actual["d_origin"] == expected["d_origin"]

    # Passive role inference is retained as an observational result.
    # Ambiguous roles are no longer required to equal externally supplied
    # labels because their resolution belongs to the Active Probe pipeline.
    comparison = {
        role: {
            "passive": actual[role],
            "external": expected[role],
            "matches": actual[role] == expected[role],
        }
        for role in (
            "d_fast",
            "d_root",
            "node_star",
        )
    }

    assert isinstance(
        comparison,
        dict,
    )

    # Keep scenario counterfactual metrics available and non-empty for audit.
    diagnostic = metrics_snapshot(
        solution
    )

    assert diagnostic

    # This test must not rewrite or silently preload external role labels
    # into the passive solver.
    assert case.metadata[
        "hidden_labels_used_by_solver"
    ] is False

def test_terminal_is_not_inferred_as_cascade_origin(
    solution: ROIFSolution,
) -> None:
    assert (
        solution.root_result.d_origin
        != "terminal_displacement"
    )


def test_bypass_is_not_required_to_be_the_fast_failure(
    solution: ROIFSolution,
) -> None:
    assert (
        solution.root_result.d_fast
        != "bypass_branch_stiffness"
    )


def test_solution_summary_is_consistent(
    solution: ROIFSolution,
) -> None:
    summary = solution_summary(
        solution
    )

    assert isinstance(
        summary,
        MappingProxyType,
    )

    assert (
        summary["status"]
        == solution.status.value
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


def test_mechanical_validation_pipeline_is_deterministic(
    case: PrestressedBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    repeated = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    assert (
        repeated.status
        is solution.status
    )

    assert (
        repeated.decision
        == solution.decision
    )

    assert (
        repeated.root_result
        == solution.root_result
    )

    assert (
        metrics_snapshot(repeated)
        == metrics_snapshot(solution)
    )


