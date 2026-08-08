"""
ROIF Mechanical Validation
Scenario 03C вЂ” Misleading High-Amplitude Branch Ground Truth

This suite validates the scalar-vs-vector conflict before Active Probe logic
is introduced.

Goals
-----
1. Verify the graph contains two competing branches.
2. Verify the loud branch has the larger scalar transfer gain/response.
3. Verify the loud branch direction is orthogonal to the candidate edge.
4. Verify the quieter branch is aligned with the candidate edge.
5. Verify both branches reach the same terminal.
6. Verify evaluator-side winners do not leak into Solver inputs.
7. Verify tensor, cascade, and solver behavior is deterministic.

This is a synthetic mechanical validation case and carries no biological,
diagnostic, or treatment claim.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from roif.roif_cascade import run_cascade
from roif.roif_solver import ROIFSolution, solve_roif
from roif.roif_tensor import build_capacity_tensor

from validation.mechanical.misleading_high_amplitude_branch_case import (
    ALIGNED_BRANCH_CHANNEL,
    ALIGNED_GAIN,
    ALIGNED_RESPONSE_DIRECTION,
    ALIGNED_TO_TERMINAL,
    CANDIDATE_EDGE_DIRECTION,
    LOUD_BRANCH_CHANNEL,
    LOUD_GAIN,
    LOUD_RESPONSE_DIRECTION,
    LOUD_TO_TERMINAL,
    MisleadingHighAmplitudeValidationCase,
    PROBE_NODE_CHANNEL,
    PROBE_TO_ALIGNED,
    PROBE_TO_LOUD,
    SCENARIO_REDUCE_ALIGNED,
    SCENARIO_REDUCE_LOUD,
    SCENARIO_REDUCE_TERMINAL,
    SCENARIO_REMOVE_PROBE_NODE,
    TERMINAL_CHANNEL,
    build_misleading_high_amplitude_case,
)


@pytest.fixture(scope="module")
def case() -> MisleadingHighAmplitudeValidationCase:
    return build_misleading_high_amplitude_case()


def channel_lookup(case: MisleadingHighAmplitudeValidationCase):
    return {
        channel.channel_id: channel
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }


def reference_channel_lookup(case: MisleadingHighAmplitudeValidationCase):
    return {
        channel.channel_id: channel
        for entity in case.reference_system.entities
        for channel in entity.channels
    }


def operator_lookup(case: MisleadingHighAmplitudeValidationCase):
    return {
        operator.operator_id: operator
        for operator in case.pathological_system.operators
    }


def vector_tuple(vector) -> tuple[float, float, float]:
    return (
        float(vector.x),
        float(vector.y),
        float(vector.z),
    )


def norm(vector) -> float:
    return float(
        np.linalg.norm(
            np.asarray(
                vector_tuple(vector),
                dtype=float,
            )
        )
    )


def normalized(vector) -> np.ndarray:
    arr = np.asarray(
        vector_tuple(vector),
        dtype=float,
    )
    magnitude = np.linalg.norm(arr)

    if magnitude <= 0.0:
        raise AssertionError(
            "validation direction must be non-zero"
        )

    return arr / magnitude


def alignment(left, right) -> float:
    return float(
        np.dot(
            normalized(left),
            normalized(right),
        )
    )


# =============================================================================
# Case contract
# =============================================================================


def test_case_contract_is_complete(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert (
        case.case_id
        == "mechanical_03C_misleading_high_amplitude_branch"
    )

    assert case.channel_ids == (
        PROBE_NODE_CHANNEL,
        LOUD_BRANCH_CHANNEL,
        ALIGNED_BRANCH_CHANNEL,
        TERMINAL_CHANNEL,
    )

    assert len(case.scenarios) == 4


def test_case_metadata_declares_scalar_vector_conflict(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert (
        case.metadata["topology"]
        == "misleading_high_amplitude_branch"
    )
    assert case.metadata["branching"] is True
    assert case.metadata["feedback"] is False
    assert case.metadata["pre_stressed"] is True
    assert case.metadata["expected_scalar_vector_conflict"] is True


def test_case_metadata_preserves_non_biological_boundary(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert case.metadata["biological_semantics"] is False
    assert case.metadata["diagnostic_claim"] is False
    assert case.metadata["treatment_claim"] is False


def test_initial_state_is_read_only(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert isinstance(case.initial_state, MappingProxyType)

    with pytest.raises(TypeError):
        case.initial_state[PROBE_NODE_CHANNEL] = 0.0  # type: ignore[index]


def test_external_labels_are_read_only(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    assert isinstance(labels, MappingProxyType)

    with pytest.raises(TypeError):
        labels["amplitude_winner"] = ALIGNED_BRANCH_CHANNEL  # type: ignore[index]


def test_validation_case_is_frozen(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    with pytest.raises(FrozenInstanceError):
        case.case_id = "x"  # type: ignore[misc]


# =============================================================================
# Topology ground truth
# =============================================================================


def test_pathological_graph_has_exactly_four_operators(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert len(case.pathological_system.operators) == 4


def test_operator_ids_are_exact(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert {
        operator.operator_id
        for operator in case.pathological_system.operators
    } == {
        PROBE_TO_LOUD,
        PROBE_TO_ALIGNED,
        LOUD_TO_TERMINAL,
        ALIGNED_TO_TERMINAL,
    }


def test_probe_node_has_exactly_two_outgoing_relations(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (PROBE_NODE_CHANNEL,)
    ]

    assert len(outgoing) == 2

    assert {
        operator.target_ids[0]
        for operator in outgoing
    } == {
        LOUD_BRANCH_CHANNEL,
        ALIGNED_BRANCH_CHANNEL,
    }


def test_loud_branch_reaches_terminal(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    lookup = operator_lookup(case)

    assert lookup[
        LOUD_TO_TERMINAL
    ].target_ids == (
        TERMINAL_CHANNEL,
    )


def test_aligned_branch_reaches_terminal(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    lookup = operator_lookup(case)

    assert lookup[
        ALIGNED_TO_TERMINAL
    ].target_ids == (
        TERMINAL_CHANNEL,
    )


def test_terminal_has_no_outgoing_relation(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    outgoing = [
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (TERMINAL_CHANNEL,)
    ]

    assert outgoing == []


# =============================================================================
# Scalar amplitude conflict ground truth
# =============================================================================


def test_declared_loud_gain_exceeds_aligned_gain() -> None:
    assert LOUD_GAIN > ALIGNED_GAIN


def test_pathological_loud_operator_gain_exceeds_aligned(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    lookup = operator_lookup(case)

    assert (
        float(lookup[PROBE_TO_LOUD].gain)
        >
        float(lookup[PROBE_TO_ALIGNED].gain)
    )


def test_scalar_gain_margin_is_material(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    lookup = operator_lookup(case)

    loud = float(
        lookup[PROBE_TO_LOUD].gain
    )
    aligned_gain = float(
        lookup[PROBE_TO_ALIGNED].gain
    )

    assert loud / aligned_gain > 1.50


def test_reference_probe_branch_gains_are_equal(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    lookup = {
        operator.operator_id: operator
        for operator in case.reference_system.operators
    }

    assert float(
        lookup[PROBE_TO_LOUD].gain
    ) == pytest.approx(
        float(
            lookup[PROBE_TO_ALIGNED].gain
        )
    )


# =============================================================================
# Directional ground truth
# =============================================================================


def test_candidate_edge_direction_is_x_axis() -> None:
    assert vector_tuple(
        CANDIDATE_EDGE_DIRECTION
    ) == pytest.approx(
        (1.0, 0.0, 0.0)
    )


def test_loud_response_direction_is_y_axis() -> None:
    assert vector_tuple(
        LOUD_RESPONSE_DIRECTION
    ) == pytest.approx(
        (0.0, 1.0, 0.0)
    )


def test_aligned_response_direction_is_x_axis() -> None:
    assert vector_tuple(
        ALIGNED_RESPONSE_DIRECTION
    ) == pytest.approx(
        (1.0, 0.0, 0.0)
    )


def test_loud_response_is_orthogonal_to_candidate_edge() -> None:
    assert alignment(
        LOUD_RESPONSE_DIRECTION,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_aligned_response_is_parallel_to_candidate_edge() -> None:
    assert alignment(
        ALIGNED_RESPONSE_DIRECTION,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_transport_and_probe_response_directions_are_separated(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    """
    CapacityTensor transport directions are aligned for both branches,
    while Probe-response directions preserve the scalar-vector conflict.
    """

    channels = channel_lookup(case)

    # Passive transport must allow both branches to propagate.
    assert alignment(
        channels[LOUD_BRANCH_CHANNEL].direction,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert alignment(
        channels[ALIGNED_BRANCH_CHANNEL].direction,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    # Probe-response evidence remains discriminative.
    assert alignment(
        LOUD_RESPONSE_DIRECTION,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert alignment(
        ALIGNED_RESPONSE_DIRECTION,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


# =============================================================================
# External validation contract
# =============================================================================


def test_expected_contract_declares_scalar_winner(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert (
        case.expected.amplitude_winner
        == LOUD_BRANCH_CHANNEL
    )


def test_expected_contract_declares_directional_winner(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert (
        case.expected.directional_winner
        == ALIGNED_BRANCH_CHANNEL
    )


def test_expected_scalar_and_directional_winners_are_distinct(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert (
        case.expected.amplitude_winner
        != case.expected.directional_winner
    )


def test_expected_contract_requires_amplitude_not_to_decide_direction(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert (
        case.expected.amplitude_must_not_decide_direction
        is True
    )


def test_solver_metadata_does_not_contain_evaluator_winners(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    forbidden_keys = {
        "amplitude_winner",
        "directional_winner",
        "expected_amplitude_winner",
        "expected_directional_winner",
        "preferred_branch",
        "winning_branch",
    }

    assert forbidden_keys.isdisjoint(
        case.pathological_system.metadata.keys()
    )

    assert forbidden_keys.isdisjoint(
        case.solver_config.metadata.keys()
    )

    assert (
        case.pathological_system.metadata[
            "expected_directional_winner_used_by_solver"
        ]
        is False
    )

    assert (
        case.metadata[
            "hidden_labels_used_by_solver"
        ]
        is False
    )


# =============================================================================
# Counterfactual scenario contract
# =============================================================================


def test_counterfactual_scenario_ids_are_complete(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    assert {
        scenario.scenario_id
        for scenario in case.scenarios
    } == {
        SCENARIO_REMOVE_PROBE_NODE,
        SCENARIO_REDUCE_LOUD,
        SCENARIO_REDUCE_ALIGNED,
        SCENARIO_REDUCE_TERMINAL,
    }


def test_branch_counterfactuals_are_cost_symmetric(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    loud = case.scenario(
        SCENARIO_REDUCE_LOUD
    ).interventions[0]

    aligned_branch = case.scenario(
        SCENARIO_REDUCE_ALIGNED
    ).interventions[0]

    assert loud.cost == pytest.approx(
        aligned_branch.cost
    )
    assert loud.safety_risk == pytest.approx(
        aligned_branch.safety_risk
    )
    assert loud.uncertainty == pytest.approx(
        aligned_branch.uncertainty
    )
    assert loud.irreversibility == pytest.approx(
        aligned_branch.irreversibility
    )


def test_case_scenario_lookup_rejects_unknown_id(
    case: MisleadingHighAmplitudeValidationCase,
) -> None:
    with pytest.raises(KeyError):
        case.scenario("does_not_exist")


# =============================================================================
# Tensor
# =============================================================================


@pytest.fixture(scope="module")
def tensor(
    case: MisleadingHighAmplitudeValidationCase,
):
    return build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )


def test_tensor_builds_for_pathological_system(
    case: MisleadingHighAmplitudeValidationCase,
    tensor,
) -> None:
    assert tensor.matrix.shape == (
        len(case.channel_ids),
        len(case.channel_ids),
    )


def test_tensor_contains_exact_declared_edges(
    tensor,
) -> None:
    channel_index = {
        channel_id: index
        for index, channel_id
        in enumerate(tensor.channel_ids)
    }

    declared = {
        (PROBE_NODE_CHANNEL, LOUD_BRANCH_CHANNEL),
        (PROBE_NODE_CHANNEL, ALIGNED_BRANCH_CHANNEL),
        (LOUD_BRANCH_CHANNEL, TERMINAL_CHANNEL),
        (ALIGNED_BRANCH_CHANNEL, TERMINAL_CHANNEL),
    }

    nonzero = set()

    for source_id, source_index in channel_index.items():
        for target_id, target_index in channel_index.items():
            if source_id == target_id:
                continue

            if tensor.matrix[
                target_index,
                source_index,
            ] != 0.0:
                nonzero.add(
                    (
                        source_id,
                        target_id,
                    )
                )

    assert nonzero == declared


def test_loud_tensor_entry_exceeds_aligned_tensor_entry(
    tensor,
) -> None:
    index = {
        channel_id: position
        for position, channel_id
        in enumerate(tensor.channel_ids)
    }

    loud = float(
        tensor.matrix[
            index[LOUD_BRANCH_CHANNEL],
            index[PROBE_NODE_CHANNEL],
        ]
    )

    aligned_value = float(
        tensor.matrix[
            index[ALIGNED_BRANCH_CHANNEL],
            index[PROBE_NODE_CHANNEL],
        ]
    )

    assert loud > aligned_value


# =============================================================================
# Cascade scalar response
# =============================================================================


@pytest.fixture(scope="module")
def trajectory(
    case: MisleadingHighAmplitudeValidationCase,
):
    return run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )


def test_cascade_reaches_loud_branch(
    trajectory,
) -> None:
    assert trajectory.peak(
        LOUD_BRANCH_CHANNEL
    ) > 0.0


def test_cascade_reaches_aligned_branch(
    trajectory,
) -> None:
    assert trajectory.peak(
        ALIGNED_BRANCH_CHANNEL
    ) > 0.0


def test_cascade_reaches_common_terminal(
    trajectory,
) -> None:
    assert trajectory.peak(
        TERMINAL_CHANNEL
    ) > 0.0


def test_loud_branch_has_larger_peak_scalar_response(
    trajectory,
) -> None:
    assert (
        trajectory.peak(
            LOUD_BRANCH_CHANNEL
        )
        >
        trajectory.peak(
            ALIGNED_BRANCH_CHANNEL
        )
    )


def test_loud_branch_has_larger_total_exposure(
    trajectory,
) -> None:
    assert (
        trajectory.total_exposure(
            LOUD_BRANCH_CHANNEL
        )
        >
        trajectory.total_exposure(
            ALIGNED_BRANCH_CHANNEL
        )
    )


def test_both_branches_activate_at_same_first_step(
    trajectory,
) -> None:
    assert (
        trajectory.first_threshold_crossing(
            LOUD_BRANCH_CHANNEL,
            1e-12,
        )
        ==
        trajectory.first_threshold_crossing(
            ALIGNED_BRANCH_CHANNEL,
            1e-12,
        )
    )


# =============================================================================
# Solver integration
# =============================================================================


@pytest.fixture(scope="module")
def solution(
    case: MisleadingHighAmplitudeValidationCase,
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
    case: MisleadingHighAmplitudeValidationCase,
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


def test_solver_is_not_used_as_directional_ground_truth(
    case: MisleadingHighAmplitudeValidationCase,
    solution: ROIFSolution,
) -> None:
    # The passive solver may react strongly to scalar gain. 03C does not
    # promote that passive ranking to directional ground truth.
    assert (
        case.expected.directional_winner
        == ALIGNED_BRANCH_CHANNEL
    )

    assert solution.root_result.d_fast in case.channel_ids


def test_solver_does_not_receive_external_winner_labels(
    case: MisleadingHighAmplitudeValidationCase,
    solution: ROIFSolution,
) -> None:
    assert solution.metadata.get(
        "expected_labels_used",
        False,
    ) is False

    assert (
        case.metadata[
            "hidden_labels_used_by_solver"
        ]
        is False
    )


# =============================================================================
# Determinism
# =============================================================================


def test_ground_truth_case_factory_is_deterministic() -> None:
    left = build_misleading_high_amplitude_case()
    right = build_misleading_high_amplitude_case()

    assert left.case_id == right.case_id
    assert left.channel_ids == right.channel_ids
    assert left.initial_state == right.initial_state
    assert left.expected == right.expected
    assert left.scenarios == right.scenarios


def test_tensor_is_deterministic(
    case: MisleadingHighAmplitudeValidationCase,
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
    case: MisleadingHighAmplitudeValidationCase,
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
    case: MisleadingHighAmplitudeValidationCase,
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

