"""
Ground-truth validation tests for Mechanical Scenario 03A:
Pre-Stressed Serial Weak-Link Chain.

This file validates the mechanical case before any Active Probe policy is added.

Goals
-----
1. Verify the graph is strictly serial and unambiguous.
2. Verify serial_weak_link is the designed lowest-reserve working component.
3. Verify the pathological system differs materially from the reference.
4. Verify the standard ROIF tensor/cascade/solver pipeline runs.
5. Verify external expected labels are evaluator-side only.
6. Preserve determinism and auditability.

This is a synthetic mechanical validation case, not a biological model.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.roif_cascade import run_cascade
from roif.roif_solver import ROIFSolution, solve_roif
from roif.roif_tensor import build_capacity_tensor

from validation.mechanical.prestressed_serial_weak_link_case import (
    DISTAL_CHANNEL,
    DISTAL_TO_TERMINAL,
    PRELOAD_CHANNEL,
    PRELOAD_TO_PROXIMAL,
    PROXIMAL_CHANNEL,
    PROXIMAL_TO_WEAK_LINK,
    PrestressedSerialWeakLinkValidationCase,
    SCENARIO_REDUCE_DISTAL,
    SCENARIO_REDUCE_TERMINAL,
    SCENARIO_REDUCE_WEAK_LINK,
    SCENARIO_REMOVE_PRELOAD,
    TERMINAL_CHANNEL,
    WEAK_LINK_CHANNEL,
    WEAK_LINK_TO_DISTAL,
    build_prestressed_serial_weak_link_case,
)


@pytest.fixture(scope="module")
def case() -> PrestressedSerialWeakLinkValidationCase:
    return build_prestressed_serial_weak_link_case()


def channel_lookup(case: PrestressedSerialWeakLinkValidationCase):
    return {
        channel.channel_id: channel
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }


def reference_channel_lookup(case: PrestressedSerialWeakLinkValidationCase):
    return {
        channel.channel_id: channel
        for entity in case.reference_system.entities
        for channel in entity.channels
    }


def operator_lookup(case: PrestressedSerialWeakLinkValidationCase):
    return {
        operator.operator_id: operator
        for operator in case.pathological_system.operators
    }


def reserve(channel) -> float:
    return (
        float(channel.capacity_state.capacity)
        - float(channel.capacity_state.load)
    )


# =============================================================================
# Case contract
# =============================================================================


def test_case_contract_is_complete(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert case.case_id == "mechanical_03A_prestressed_serial_weak_link"

    assert case.channel_ids == (
        PRELOAD_CHANNEL,
        PROXIMAL_CHANNEL,
        WEAK_LINK_CHANNEL,
        DISTAL_CHANNEL,
        TERMINAL_CHANNEL,
    )

    assert len(case.scenarios) == 4


def test_case_metadata_declares_serial_control_topology(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert case.metadata["topology"] == "serial_weak_link"
    assert case.metadata["pre_stressed"] is True
    assert case.metadata["branching"] is False
    assert case.metadata["feedback"] is False
    assert case.metadata["active_probe_control_case"] is True
    assert case.metadata["expected_branch_ambiguity"] is False


def test_case_metadata_preserves_non_biological_boundary(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert case.metadata["biological_semantics"] is False
    assert case.metadata["diagnostic_claim"] is False
    assert case.metadata["treatment_claim"] is False


def test_initial_state_is_read_only(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert isinstance(case.initial_state, MappingProxyType)

    with pytest.raises(TypeError):
        case.initial_state[PRELOAD_CHANNEL] = 0.0  # type: ignore[index]


def test_external_labels_are_read_only(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    assert isinstance(labels, MappingProxyType)

    with pytest.raises(TypeError):
        labels["d_fast"] = "x"  # type: ignore[index]


def test_validation_case_is_frozen(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    with pytest.raises(FrozenInstanceError):
        case.case_id = "x"  # type: ignore[misc]


# =============================================================================
# Serial topology ground truth
# =============================================================================


def test_pathological_graph_has_exactly_four_operators(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert len(case.pathological_system.operators) == 4


def test_serial_operator_ids_are_exact(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert tuple(
        operator.operator_id
        for operator in case.pathological_system.operators
    ) == (
        PRELOAD_TO_PROXIMAL,
        PROXIMAL_TO_WEAK_LINK,
        WEAK_LINK_TO_DISTAL,
        DISTAL_TO_TERMINAL,
    )


def test_preload_has_only_one_outgoing_relation(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if PRELOAD_CHANNEL in operator.source_ids
    ]

    assert len(outgoing) == 1
    assert outgoing[0].target_ids == (PROXIMAL_CHANNEL,)


def test_proximal_has_only_one_outgoing_relation(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if PROXIMAL_CHANNEL in operator.source_ids
    ]

    assert len(outgoing) == 1
    assert outgoing[0].target_ids == (WEAK_LINK_CHANNEL,)


def test_weak_link_has_only_one_outgoing_relation(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if WEAK_LINK_CHANNEL in operator.source_ids
    ]

    assert len(outgoing) == 1
    assert outgoing[0].target_ids == (DISTAL_CHANNEL,)


def test_distal_has_only_one_outgoing_relation(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if DISTAL_CHANNEL in operator.source_ids
    ]

    assert len(outgoing) == 1
    assert outgoing[0].target_ids == (TERMINAL_CHANNEL,)


def test_terminal_has_no_outgoing_relation(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if TERMINAL_CHANNEL in operator.source_ids
    ]

    assert outgoing == []


def test_no_operator_points_upstream(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    expected_edges = {
        PRELOAD_CHANNEL: PROXIMAL_CHANNEL,
        PROXIMAL_CHANNEL: WEAK_LINK_CHANNEL,
        WEAK_LINK_CHANNEL: DISTAL_CHANNEL,
        DISTAL_CHANNEL: TERMINAL_CHANNEL,
    }

    for operator in case.pathological_system.operators:
        source = operator.source_ids[0]
        target = operator.target_ids[0]

        assert expected_edges[source] == target


# =============================================================================
# Mechanical reserve ground truth
# =============================================================================


def test_weak_link_has_smallest_working_reserve(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    channels = channel_lookup(case)

    working_ids = (
        PROXIMAL_CHANNEL,
        WEAK_LINK_CHANNEL,
        DISTAL_CHANNEL,
        TERMINAL_CHANNEL,
    )

    reserves = {
        channel_id: reserve(channels[channel_id])
        for channel_id in working_ids
    }

    assert min(
        reserves,
        key=reserves.get,
    ) == WEAK_LINK_CHANNEL


def test_weak_link_reserve_is_positive_but_small(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    channel = channel_lookup(case)[WEAK_LINK_CHANNEL]

    value = reserve(channel)

    assert value > 0.0
    assert value < 0.20


def test_weak_link_has_highest_working_utilization(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    channels = channel_lookup(case)

    working_ids = (
        PROXIMAL_CHANNEL,
        WEAK_LINK_CHANNEL,
        DISTAL_CHANNEL,
        TERMINAL_CHANNEL,
    )

    utilization = {
        channel_id: (
            channels[channel_id].capacity_state.load
            / channels[channel_id].capacity_state.capacity
        )
        for channel_id in working_ids
    }

    assert max(
        utilization,
        key=utilization.get,
    ) == WEAK_LINK_CHANNEL


def test_weak_link_is_more_degraded_than_reference(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    pathological = channel_lookup(case)[WEAK_LINK_CHANNEL]
    reference = reference_channel_lookup(case)[WEAK_LINK_CHANNEL]

    assert (
        pathological.capacity_state.capacity
        < reference.capacity_state.capacity
    )
    assert pathological.activation.command < reference.activation.command
    assert pathological.geometry.mobility < reference.geometry.mobility
    assert pathological.history_factor < reference.history_factor


def test_pathological_chain_has_lower_total_reserve_than_reference(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    pathological = channel_lookup(case)
    reference = reference_channel_lookup(case)

    pathological_total = sum(
        reserve(pathological[channel_id])
        for channel_id in case.channel_ids
    )
    reference_total = sum(
        reserve(reference[channel_id])
        for channel_id in case.channel_ids
    )

    assert pathological_total < reference_total


# =============================================================================
# External role hypothesis boundary
# =============================================================================


def test_expected_roles_are_declared_externally(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert case.expected.d_origin == PRELOAD_CHANNEL
    assert case.expected.d_fast == WEAK_LINK_CHANNEL
    assert case.expected.d_root == WEAK_LINK_CHANNEL
    assert case.expected.node_star == PRELOAD_CHANNEL


def test_solver_metadata_does_not_contain_expected_role_labels(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert case.metadata["hidden_labels_used_by_solver"] is False
    assert (
        case.pathological_system.metadata[
            "hidden_labels_used_by_solver"
        ]
        is False
    )

    forbidden_keys = {
        "expected_d_origin",
        "expected_d_fast",
        "expected_d_root",
        "expected_node_star",
        "d_origin_label",
        "d_fast_label",
        "d_root_label",
        "node_star_label",
    }

    assert forbidden_keys.isdisjoint(
        case.pathological_system.metadata.keys()
    )
    assert forbidden_keys.isdisjoint(
        case.solver_config.metadata.keys()
    )


# =============================================================================
# Counterfactual scenario contract
# =============================================================================


def test_counterfactual_scenario_ids_are_complete(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    assert {
        scenario.scenario_id
        for scenario in case.scenarios
    } == {
        SCENARIO_REMOVE_PRELOAD,
        SCENARIO_REDUCE_WEAK_LINK,
        SCENARIO_REDUCE_DISTAL,
        SCENARIO_REDUCE_TERMINAL,
    }


def test_case_scenario_lookup_returns_requested_scenario(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    scenario = case.scenario(
        SCENARIO_REMOVE_PRELOAD
    )

    assert scenario.scenario_id == SCENARIO_REMOVE_PRELOAD


def test_case_scenario_lookup_rejects_unknown_id(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    with pytest.raises(KeyError):
        case.scenario("does_not_exist")


# =============================================================================
# Tensor and cascade
# =============================================================================


def test_tensor_builds_for_pathological_system(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    tensor = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    assert tensor.matrix.shape == (
        len(case.channel_ids),
        len(case.channel_ids),
    )


def test_serial_tensor_contains_only_declared_off_diagonal_edges(
    case: PrestressedSerialWeakLinkValidationCase,
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
        (PRELOAD_CHANNEL, PROXIMAL_CHANNEL),
        (PROXIMAL_CHANNEL, WEAK_LINK_CHANNEL),
        (WEAK_LINK_CHANNEL, DISTAL_CHANNEL),
        (DISTAL_CHANNEL, TERMINAL_CHANNEL),
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


def test_cascade_reaches_terminal_in_serial_order(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    tensor = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    trajectory = run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )

    indices = {
        channel_id: tensor.channel_ids.index(channel_id)
        for channel_id in case.channel_ids
    }

    def first_positive(channel_id: str) -> int:
        index = indices[channel_id]

        for time_index, state in enumerate(trajectory.state_history):
            if state[index] > 0.0:
                return time_index

        raise AssertionError(
            f"{channel_id} never became active"
        )

    assert first_positive(PRELOAD_CHANNEL) < first_positive(PROXIMAL_CHANNEL)
    assert first_positive(PROXIMAL_CHANNEL) < first_positive(WEAK_LINK_CHANNEL)
    assert first_positive(WEAK_LINK_CHANNEL) < first_positive(DISTAL_CHANNEL)
    assert first_positive(DISTAL_CHANNEL) < first_positive(TERMINAL_CHANNEL)


def test_terminal_eventually_becomes_active(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    tensor = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    trajectory = run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )

    terminal_index = tensor.channel_ids.index(
        TERMINAL_CHANNEL
    )

    assert any(
        state[terminal_index] > 0.0
        for state in trajectory.state_history
    )


# =============================================================================
# Standard solver integration
# =============================================================================


def test_complete_roif_solver_pipeline_runs(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    solution = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    assert isinstance(
        solution,
        ROIFSolution,
    )


def test_solver_evaluates_all_provided_counterfactuals(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    solution = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    result_ids = {
        result.scenario.scenario_id
        for result in solution.counterfactual_batch.results
    }

    assert result_ids == {
        scenario.scenario_id
        for scenario in case.scenarios
    }


def test_solver_root_result_exposes_all_four_roles(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    solution = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    assert solution.root_result.d_origin is not None
    assert solution.root_result.d_fast is not None
    assert solution.root_result.d_root is not None
    assert solution.root_result.node_star is not None


def test_solver_does_not_receive_expected_labels(
    case: PrestressedSerialWeakLinkValidationCase,
) -> None:
    solution = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

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
    left = build_prestressed_serial_weak_link_case()
    right = build_prestressed_serial_weak_link_case()

    assert left.case_id == right.case_id
    assert left.channel_ids == right.channel_ids
    assert left.initial_state == right.initial_state
    assert left.expected == right.expected


def test_standard_solver_result_is_deterministic(
    case: PrestressedSerialWeakLinkValidationCase,
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



