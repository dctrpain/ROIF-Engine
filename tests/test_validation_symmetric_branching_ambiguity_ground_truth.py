"""
ROIF Mechanical Validation
Scenario 03B вЂ” Symmetric Branching Ambiguity Ground Truth

This suite validates the mechanical ambiguity before Active Probe logic is
introduced.

Goals
-----
1. Verify the graph contains exactly two parallel branches.
2. Verify both branches are mechanically near-symmetric.
3. Verify both branches propagate to the same terminal.
4. Verify no evaluator-side preferred branch is injected into the Solver.
5. Verify tensor, cascade, and solver pipelines run deterministically.
6. Preserve ambiguity as a first-class validation result.

This is a synthetic mechanical validation case and carries no biological,
diagnostic, or treatment claim.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest
import numpy as np

from roif.roif_cascade import run_cascade
from roif.roif_solver import ROIFSolution, solve_roif
from roif.roif_tensor import build_capacity_tensor

from validation.mechanical.symmetric_branching_ambiguity_case import (
    A_TO_TERMINAL,
    B_TO_TERMINAL,
    BRANCH_A_CHANNEL,
    BRANCH_B_CHANNEL,
    PRELOAD_CHANNEL,
    PRELOAD_TO_A,
    PRELOAD_TO_B,
    SCENARIO_REDUCE_BRANCH_A,
    SCENARIO_REDUCE_BRANCH_B,
    SCENARIO_REDUCE_TERMINAL,
    SCENARIO_REMOVE_PRELOAD,
    SymmetricBranchingValidationCase,
    TERMINAL_CHANNEL,
    build_symmetric_branching_ambiguity_case,
)


@pytest.fixture(scope="module")
def case() -> SymmetricBranchingValidationCase:
    return build_symmetric_branching_ambiguity_case()


def channel_lookup(case: SymmetricBranchingValidationCase):
    return {
        channel.channel_id: channel
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }


def reference_channel_lookup(case: SymmetricBranchingValidationCase):
    return {
        channel.channel_id: channel
        for entity in case.reference_system.entities
        for channel in entity.channels
    }


def operator_lookup(case: SymmetricBranchingValidationCase):
    return {
        operator.operator_id: operator
        for operator in case.pathological_system.operators
    }


def reserve(channel) -> float:
    return (
        float(channel.capacity_state.capacity)
        - float(channel.capacity_state.load)
    )


def utilization(channel) -> float:
    return (
        float(channel.capacity_state.load)
        / float(channel.capacity_state.capacity)
    )


# =============================================================================
# Case contract
# =============================================================================


def test_case_contract_is_complete(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert (
        case.case_id
        == "mechanical_03B_symmetric_branching_ambiguity"
    )

    assert case.channel_ids == (
        PRELOAD_CHANNEL,
        BRANCH_A_CHANNEL,
        BRANCH_B_CHANNEL,
        TERMINAL_CHANNEL,
    )

    assert len(case.scenarios) == 4


def test_case_metadata_declares_branching_ambiguity(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert case.metadata["topology"] == "symmetric_branching_ambiguity"
    assert case.metadata["branching"] is True
    assert case.metadata["feedback"] is False
    assert case.metadata["near_symmetric_branches"] is True
    assert case.metadata["expected_unique_passive_branch"] is False


def test_case_metadata_preserves_non_biological_boundary(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert case.metadata["biological_semantics"] is False
    assert case.metadata["diagnostic_claim"] is False
    assert case.metadata["treatment_claim"] is False


def test_initial_state_is_read_only(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert isinstance(case.initial_state, MappingProxyType)

    with pytest.raises(TypeError):
        case.initial_state[PRELOAD_CHANNEL] = 0.0  # type: ignore[index]


def test_external_labels_are_read_only(
    case: SymmetricBranchingValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    assert isinstance(labels, MappingProxyType)

    with pytest.raises(TypeError):
        labels["ambiguity_expected"] = False  # type: ignore[index]


def test_validation_case_is_frozen(
    case: SymmetricBranchingValidationCase,
) -> None:
    with pytest.raises(FrozenInstanceError):
        case.case_id = "x"  # type: ignore[misc]


# =============================================================================
# Topology ground truth
# =============================================================================


def test_pathological_graph_has_exactly_four_operators(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert len(case.pathological_system.operators) == 4


def test_operator_ids_are_exact(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert {
        operator.operator_id
        for operator in case.pathological_system.operators
    } == {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
        A_TO_TERMINAL,
        B_TO_TERMINAL,
    }


def test_preload_has_exactly_two_outgoing_relations(
    case: SymmetricBranchingValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (PRELOAD_CHANNEL,)
    ]

    assert len(outgoing) == 2

    assert {
        operator.target_ids[0]
        for operator in outgoing
    } == {
        BRANCH_A_CHANNEL,
        BRANCH_B_CHANNEL,
    }


def test_branch_a_has_one_outgoing_terminal_relation(
    case: SymmetricBranchingValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (BRANCH_A_CHANNEL,)
    ]

    assert len(outgoing) == 1
    assert outgoing[0].target_ids == (TERMINAL_CHANNEL,)


def test_branch_b_has_one_outgoing_terminal_relation(
    case: SymmetricBranchingValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (BRANCH_B_CHANNEL,)
    ]

    assert len(outgoing) == 1
    assert outgoing[0].target_ids == (TERMINAL_CHANNEL,)


def test_terminal_has_no_outgoing_relation(
    case: SymmetricBranchingValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (TERMINAL_CHANNEL,)
    ]

    assert outgoing == []


def test_both_branches_converge_on_same_terminal(
    case: SymmetricBranchingValidationCase,
) -> None:
    lookup = operator_lookup(case)

    assert (
        lookup[A_TO_TERMINAL].target_ids
        == lookup[B_TO_TERMINAL].target_ids
        == (TERMINAL_CHANNEL,)
    )


# =============================================================================
# Near-symmetry ground truth
# =============================================================================


def test_branch_gains_are_nearly_equal(
    case: SymmetricBranchingValidationCase,
) -> None:
    lookup = operator_lookup(case)

    delta = abs(
        float(lookup[PRELOAD_TO_A].gain)
        - float(lookup[PRELOAD_TO_B].gain)
    )

    assert delta < 0.01


def test_downstream_branch_gains_are_nearly_equal(
    case: SymmetricBranchingValidationCase,
) -> None:
    lookup = operator_lookup(case)

    delta = abs(
        float(lookup[A_TO_TERMINAL].gain)
        - float(lookup[B_TO_TERMINAL].gain)
    )

    assert delta < 0.01


def test_branch_reserves_are_nearly_equal(
    case: SymmetricBranchingValidationCase,
) -> None:
    channels = channel_lookup(case)

    delta = abs(
        reserve(channels[BRANCH_A_CHANNEL])
        - reserve(channels[BRANCH_B_CHANNEL])
    )

    assert delta < 0.05


def test_branch_utilizations_are_nearly_equal(
    case: SymmetricBranchingValidationCase,
) -> None:
    channels = channel_lookup(case)

    delta = abs(
        utilization(channels[BRANCH_A_CHANNEL])
        - utilization(channels[BRANCH_B_CHANNEL])
    )

    assert delta < 0.01


def test_branch_activation_is_nearly_equal(
    case: SymmetricBranchingValidationCase,
) -> None:
    channels = channel_lookup(case)

    delta = abs(
        float(channels[BRANCH_A_CHANNEL].activation.command)
        - float(channels[BRANCH_B_CHANNEL].activation.command)
    )

    assert delta < 0.01


def test_branch_mobility_is_nearly_equal(
    case: SymmetricBranchingValidationCase,
) -> None:
    channels = channel_lookup(case)

    delta = abs(
        float(channels[BRANCH_A_CHANNEL].geometry.mobility)
        - float(channels[BRANCH_B_CHANNEL].geometry.mobility)
    )

    assert delta < 0.01


def test_reference_branches_are_exactly_symmetric(
    case: SymmetricBranchingValidationCase,
) -> None:
    channels = reference_channel_lookup(case)

    a = channels[BRANCH_A_CHANNEL]
    b = channels[BRANCH_B_CHANNEL]

    assert a.capacity_state.capacity == pytest.approx(
        b.capacity_state.capacity
    )
    assert a.capacity_state.load == pytest.approx(
        b.capacity_state.load
    )
    assert a.activation.command == pytest.approx(
        b.activation.command
    )
    assert a.geometry.mobility == pytest.approx(
        b.geometry.mobility
    )
    assert a.history_factor == pytest.approx(
        b.history_factor
    )


def test_pathological_branch_asymmetry_is_small_but_nonzero(
    case: SymmetricBranchingValidationCase,
) -> None:
    channels = channel_lookup(case)

    assert (
        channels[BRANCH_A_CHANNEL].capacity_state.capacity
        != channels[BRANCH_B_CHANNEL].capacity_state.capacity
    )


# =============================================================================
# External ambiguity contract
# =============================================================================


def test_expected_contract_declares_both_branches_admissible(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert set(case.expected.branch_truth) == {
        BRANCH_A_CHANNEL,
        BRANCH_B_CHANNEL,
    }


def test_expected_contract_declares_ambiguity(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert case.expected.ambiguity_expected is True


def test_expected_contract_does_not_declare_unique_branch_winner(
    case: SymmetricBranchingValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    forbidden_keys = {
        "preferred_branch",
        "winning_branch",
        "expected_branch",
        "d_fast_branch",
    }

    assert forbidden_keys.isdisjoint(labels.keys())


def test_solver_metadata_does_not_contain_preferred_branch_label(
    case: SymmetricBranchingValidationCase,
) -> None:
    forbidden_keys = {
        "preferred_branch",
        "winning_branch",
        "expected_branch",
        "branch_truth",
        "expected_d_fast",
        "expected_d_root",
        "expected_node_star",
    }

    assert forbidden_keys.isdisjoint(
        case.pathological_system.metadata.keys()
    )

    assert forbidden_keys.isdisjoint(
        case.solver_config.metadata.keys()
    )

    assert case.metadata[
        "hidden_labels_used_by_solver"
    ] is False


# =============================================================================
# Counterfactual scenario contract
# =============================================================================


def test_counterfactual_scenario_ids_are_complete(
    case: SymmetricBranchingValidationCase,
) -> None:
    assert {
        scenario.scenario_id
        for scenario in case.scenarios
    } == {
        SCENARIO_REMOVE_PRELOAD,
        SCENARIO_REDUCE_BRANCH_A,
        SCENARIO_REDUCE_BRANCH_B,
        SCENARIO_REDUCE_TERMINAL,
    }


def test_branch_counterfactuals_are_cost_symmetric(
    case: SymmetricBranchingValidationCase,
) -> None:
    a = case.scenario(
        SCENARIO_REDUCE_BRANCH_A
    ).interventions[0]

    b = case.scenario(
        SCENARIO_REDUCE_BRANCH_B
    ).interventions[0]

    assert a.cost == pytest.approx(b.cost)
    assert a.safety_risk == pytest.approx(b.safety_risk)
    assert a.uncertainty == pytest.approx(b.uncertainty)
    assert a.irreversibility == pytest.approx(b.irreversibility)


def test_case_scenario_lookup_rejects_unknown_id(
    case: SymmetricBranchingValidationCase,
) -> None:
    with pytest.raises(KeyError):
        case.scenario("does_not_exist")


# =============================================================================
# Tensor
# =============================================================================


def test_tensor_builds_for_pathological_system(
    case: SymmetricBranchingValidationCase,
) -> None:
    tensor = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    assert tensor.matrix.shape == (
        len(case.channel_ids),
        len(case.channel_ids),
    )


def test_tensor_contains_exact_declared_edges(
    case: SymmetricBranchingValidationCase,
) -> None:
    tensor = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    channel_index = {
        channel_id: index
        for index, channel_id in enumerate(tensor.channel_ids)
    }

    declared = {
        (PRELOAD_CHANNEL, BRANCH_A_CHANNEL),
        (PRELOAD_CHANNEL, BRANCH_B_CHANNEL),
        (BRANCH_A_CHANNEL, TERMINAL_CHANNEL),
        (BRANCH_B_CHANNEL, TERMINAL_CHANNEL),
    }

    nonzero = set()

    for source_id, source_index in channel_index.items():
        for target_id, target_index in channel_index.items():
            if source_id == target_id:
                continue

            if tensor.matrix[target_index, source_index] != 0.0:
                nonzero.add(
                    (
                        source_id,
                        target_id,
                    )
                )

    assert nonzero == declared


def test_parallel_tensor_entries_are_close(
    case: SymmetricBranchingValidationCase,
) -> None:
    tensor = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    index = {
        channel_id: position
        for position, channel_id
        in enumerate(tensor.channel_ids)
    }

    a_value = tensor.matrix[
        index[BRANCH_A_CHANNEL],
        index[PRELOAD_CHANNEL],
    ]

    b_value = tensor.matrix[
        index[BRANCH_B_CHANNEL],
        index[PRELOAD_CHANNEL],
    ]

    scale = max(
        abs(float(a_value)),
        abs(float(b_value)),
        1e-12,
    )

    relative_delta = abs(
        float(a_value) - float(b_value)
    ) / scale

    assert relative_delta < 0.05


# =============================================================================
# Cascade
# =============================================================================


@pytest.fixture(scope="module")
def trajectory(
    case: SymmetricBranchingValidationCase,
):
    return run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )


def test_cascade_reaches_branch_a(
    trajectory,
) -> None:
    assert trajectory.peak(
        BRANCH_A_CHANNEL
    ) > 0.0


def test_cascade_reaches_branch_b(
    trajectory,
) -> None:
    assert trajectory.peak(
        BRANCH_B_CHANNEL
    ) > 0.0


def test_cascade_reaches_common_terminal(
    trajectory,
) -> None:
    assert trajectory.peak(
        TERMINAL_CHANNEL
    ) > 0.0


def test_both_branches_activate_at_same_first_step(
    trajectory,
) -> None:
    assert (
        trajectory.first_threshold_crossing(
            BRANCH_A_CHANNEL,
            1e-12,
        )
        ==
        trajectory.first_threshold_crossing(
            BRANCH_B_CHANNEL,
            1e-12,
        )
    )


def test_parallel_branch_peak_responses_are_close(
    trajectory,
) -> None:
    a_peak = trajectory.peak(
        BRANCH_A_CHANNEL
    )
    b_peak = trajectory.peak(
        BRANCH_B_CHANNEL
    )

    scale = max(
        abs(a_peak),
        abs(b_peak),
        1e-12,
    )

    relative_delta = abs(
        a_peak - b_peak
    ) / scale

    assert relative_delta < 0.05


def test_branch_exposures_are_close(
    trajectory,
) -> None:
    a = trajectory.total_exposure(
        BRANCH_A_CHANNEL
    )
    b = trajectory.total_exposure(
        BRANCH_B_CHANNEL
    )

    scale = max(
        abs(a),
        abs(b),
        1e-12,
    )

    relative_delta = abs(
        a - b
    ) / scale

    assert relative_delta < 0.05


# =============================================================================
# Solver integration
# =============================================================================


@pytest.fixture(scope="module")
def solution(
    case: SymmetricBranchingValidationCase,
) -> ROIFSolution:
    return solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )


def test_complete_roif_solver_pipeline_runs(
    solution: ROIFSolution,
) -> None:
    assert isinstance(
        solution,
        ROIFSolution,
    )


def test_solver_evaluates_all_provided_counterfactuals(
    case: SymmetricBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    result_ids = {
        result.scenario.scenario_id
        for result in solution.counterfactual_batch.results
    }

    assert result_ids == {
        scenario.scenario_id
        for scenario in case.scenarios
    }


def test_solver_result_exposes_four_role_slots(
    solution: ROIFSolution,
) -> None:
    assert solution.root_result.d_origin is not None
    assert solution.root_result.d_fast is not None
    assert solution.root_result.d_root is not None
    assert solution.root_result.node_star is not None


def test_solver_is_not_required_to_match_unique_branch_label(
    case: SymmetricBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    assert case.expected.ambiguity_expected is True

    # Passive solver may numerically rank one branch first because the
    # pathological case contains tiny asymmetries. That ranking is not
    # promoted here to external ground truth.
    assert solution.root_result.d_fast in case.channel_ids


def test_solver_does_not_receive_external_branch_truth(
    case: SymmetricBranchingValidationCase,
    solution: ROIFSolution,
) -> None:
    assert solution.metadata.get(
        "expected_labels_used",
        False,
    ) is False

    assert case.metadata[
        "hidden_labels_used_by_solver"
    ] is False


# =============================================================================
# Determinism
# =============================================================================


def test_ground_truth_case_factory_is_deterministic() -> None:
    left = build_symmetric_branching_ambiguity_case()
    right = build_symmetric_branching_ambiguity_case()

    assert left.case_id == right.case_id
    assert left.channel_ids == right.channel_ids
    assert left.initial_state == right.initial_state
    assert left.expected == right.expected
    assert left.scenarios == right.scenarios


def test_tensor_is_deterministic(
    case: SymmetricBranchingValidationCase,
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
    assert np.allclose(
        left.matrix,
        right.matrix,
        rtol=0.0,
        atol=1e-12,
    )


def test_cascade_is_deterministic(
    case: SymmetricBranchingValidationCase,
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

    assert np.allclose(
        left.state_history,
        right.state_history,
        rtol=0.0,
        atol=1e-12,
    )


def test_solver_is_deterministic(
    case: SymmetricBranchingValidationCase,
) -> None:
    left = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    right = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    assert left.status is right.status
    assert left.decision == right.decision
    assert left.root_result == right.root_result

    left_metrics = tuple(
        (
            result.scenario.scenario_id,
            result.metrics.relative_reduction,
            result.metrics.cascade_reduction,
            result.metrics.utility,
            result.metrics.spectral_gain,
        )
        for result in left.counterfactual_batch.results
    )

    right_metrics = tuple(
        (
            result.scenario.scenario_id,
            result.metrics.relative_reduction,
            result.metrics.cascade_reduction,
            result.metrics.utility,
            result.metrics.spectral_gain,
        )
        for result in right.counterfactual_batch.results
    )

    assert left_metrics == right_metrics

