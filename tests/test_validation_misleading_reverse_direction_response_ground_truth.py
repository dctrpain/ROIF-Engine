"""
ROIF Mechanical Validation
Scenario 03D — Misleading Reverse-Direction Response Ground Truth

This suite validates the mechanical scalar-vs-vector conflict before
Active Probe inference is introduced.

Goals
-----
1. Verify the graph contains two competing branches.
2. Verify the reverse branch has the larger scalar transfer gain/response.
3. Verify both branch channel transport directions remain forward (+X).
4. Verify the reverse Probe-response direction is strictly opposite (-X).
5. Verify the aligned Probe-response direction is forward (+X).
6. Verify both branches converge on the same terminal.
7. Verify evaluator-side labels do not leak into Solver inputs.
8. Verify tensor, cascade, and solver behavior is deterministic.

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

from validation.mechanical.misleading_reverse_direction_response_case import (
    ALIGNED_BRANCH_CHANNEL,
    ALIGNED_GAIN,
    ALIGNED_RESPONSE_DIRECTION,
    ALIGNED_TO_TERMINAL,
    ALIGNED_TRANSPORT_DIRECTION,
    CANDIDATE_EDGE_DIRECTION,
    MisleadingReverseDirectionValidationCase,
    PROBE_NODE_CHANNEL,
    PROBE_TO_ALIGNED,
    PROBE_TO_REVERSE,
    REVERSE_BRANCH_CHANNEL,
    REVERSE_GAIN,
    REVERSE_RESPONSE_DIRECTION,
    REVERSE_TO_TERMINAL,
    REVERSE_TRANSPORT_DIRECTION,
    SCENARIO_REDUCE_ALIGNED,
    SCENARIO_REDUCE_REVERSE,
    SCENARIO_REDUCE_TERMINAL,
    SCENARIO_REMOVE_PROBE_NODE,
    TERMINAL_CHANNEL,
    build_misleading_reverse_direction_case,
    probe_response_direction,
    transport_direction,
)


@pytest.fixture(scope="module")
def case() -> MisleadingReverseDirectionValidationCase:
    return build_misleading_reverse_direction_case()


def channel_lookup(
    case: MisleadingReverseDirectionValidationCase,
):
    return {
        channel.channel_id: channel
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }


def operator_lookup(
    case: MisleadingReverseDirectionValidationCase,
):
    return {
        operator.operator_id: operator
        for operator in case.pathological_system.operators
    }


def vector_tuple(
    vector,
) -> tuple[
    float,
    float,
    float,
]:
    return (
        float(vector.x),
        float(vector.y),
        float(vector.z),
    )


def normalized(
    vector,
) -> np.ndarray:
    arr = np.asarray(
        vector_tuple(vector),
        dtype=float,
    )

    magnitude = np.linalg.norm(
        arr
    )

    if magnitude <= 0.0:
        raise AssertionError(
            "validation direction must be non-zero"
        )

    return arr / magnitude


def alignment(
    left,
    right,
) -> float:
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
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        case.case_id
        == "mechanical_03D_misleading_reverse_direction_response"
    )

    assert case.channel_ids == (
        PROBE_NODE_CHANNEL,
        REVERSE_BRANCH_CHANNEL,
        ALIGNED_BRANCH_CHANNEL,
        TERMINAL_CHANNEL,
    )

    assert len(case.scenarios) == 4


def test_case_metadata_declares_reverse_direction_conflict(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        case.metadata["topology"]
        == "misleading_reverse_direction_response"
    )

    assert (
        case.metadata[
            "expected_reverse_response_contradiction"
        ]
        is True
    )

    assert case.metadata["branching"] is True
    assert case.metadata["feedback"] is False
    assert case.metadata["pre_stressed"] is True


def test_case_metadata_preserves_non_biological_boundary(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert case.metadata["biological_semantics"] is False
    assert case.metadata["diagnostic_claim"] is False
    assert case.metadata["treatment_claim"] is False


def test_initial_state_is_read_only(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert isinstance(
        case.initial_state,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        case.initial_state[
            PROBE_NODE_CHANNEL
        ] = 0.0  # type: ignore[index]


def test_external_labels_are_read_only(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    assert isinstance(
        labels,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        labels[
            "directional_winner"
        ] = REVERSE_BRANCH_CHANNEL  # type: ignore[index]


def test_validation_case_is_frozen(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        case.case_id = "x"  # type: ignore[misc]


# =============================================================================
# Topology ground truth
# =============================================================================


def test_pathological_graph_has_exactly_four_operators(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert len(
        case.pathological_system.operators
    ) == 4


def test_operator_ids_are_exact(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert {
        operator.operator_id
        for operator
        in case.pathological_system.operators
    } == {
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
        REVERSE_TO_TERMINAL,
        ALIGNED_TO_TERMINAL,
    }


def test_probe_node_has_exactly_two_outgoing_relations(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    outgoing = [
        operator
        for operator
        in case.pathological_system.operators
        if operator.source_ids
        == (
            PROBE_NODE_CHANNEL,
        )
    ]

    assert len(
        outgoing
    ) == 2

    assert {
        operator.target_ids[
            0
        ]
        for operator
        in outgoing
    } == {
        REVERSE_BRANCH_CHANNEL,
        ALIGNED_BRANCH_CHANNEL,
    }


def test_reverse_branch_reaches_terminal(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    lookup = operator_lookup(
        case
    )

    assert lookup[
        REVERSE_TO_TERMINAL
    ].target_ids == (
        TERMINAL_CHANNEL,
    )


def test_aligned_branch_reaches_terminal(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    lookup = operator_lookup(
        case
    )

    assert lookup[
        ALIGNED_TO_TERMINAL
    ].target_ids == (
        TERMINAL_CHANNEL,
    )


def test_terminal_has_no_outgoing_relation(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    outgoing = [
        operator
        for operator
        in case.pathological_system.operators
        if operator.source_ids
        == (
            TERMINAL_CHANNEL,
        )
    ]

    assert outgoing == []


# =============================================================================
# Scalar dominance ground truth
# =============================================================================


def test_declared_reverse_gain_exceeds_aligned_gain() -> None:
    assert REVERSE_GAIN > ALIGNED_GAIN


def test_pathological_reverse_operator_gain_exceeds_aligned(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    lookup = operator_lookup(
        case
    )

    assert (
        float(
            lookup[
                PROBE_TO_REVERSE
            ].gain
        )
        >
        float(
            lookup[
                PROBE_TO_ALIGNED
            ].gain
        )
    )


def test_scalar_gain_margin_is_material(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    lookup = operator_lookup(
        case
    )

    reverse = float(
        lookup[
            PROBE_TO_REVERSE
        ].gain
    )

    aligned = float(
        lookup[
            PROBE_TO_ALIGNED
        ].gain
    )

    assert (
        reverse
        / aligned
        > 1.50
    )


def test_reference_probe_branch_gains_are_equal(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    lookup = {
        operator.operator_id: operator
        for operator
        in case.reference_system.operators
    }

    assert float(
        lookup[
            PROBE_TO_REVERSE
        ].gain
    ) == pytest.approx(
        float(
            lookup[
                PROBE_TO_ALIGNED
            ].gain
        )
    )


# =============================================================================
# Transport direction ground truth
# =============================================================================


def test_candidate_edge_direction_is_forward_x() -> None:
    assert vector_tuple(
        CANDIDATE_EDGE_DIRECTION
    ) == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
        )
    )


def test_reverse_transport_direction_is_forward_x() -> None:
    assert vector_tuple(
        REVERSE_TRANSPORT_DIRECTION
    ) == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
        )
    )


def test_aligned_transport_direction_is_forward_x() -> None:
    assert vector_tuple(
        ALIGNED_TRANSPORT_DIRECTION
    ) == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
        )
    )


def test_both_branch_channel_transport_directions_are_forward(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    channels = channel_lookup(
        case
    )

    assert alignment(
        channels[
            REVERSE_BRANCH_CHANNEL
        ].direction,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert alignment(
        channels[
            ALIGNED_BRANCH_CHANNEL
        ].direction,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_transport_direction_helper_preserves_forward_transport() -> None:
    assert alignment(
        transport_direction(
            REVERSE_BRANCH_CHANNEL
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert alignment(
        transport_direction(
            ALIGNED_BRANCH_CHANNEL
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


# =============================================================================
# Probe-response direction ground truth
# =============================================================================


def test_reverse_response_direction_is_negative_x() -> None:
    assert vector_tuple(
        REVERSE_RESPONSE_DIRECTION
    ) == pytest.approx(
        (
            -1.0,
            0.0,
            0.0,
        )
    )


def test_aligned_response_direction_is_positive_x() -> None:
    assert vector_tuple(
        ALIGNED_RESPONSE_DIRECTION
    ) == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
        )
    )


def test_reverse_response_is_strictly_opposed_to_candidate_edge() -> None:
    assert alignment(
        REVERSE_RESPONSE_DIRECTION,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        -1.0,
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


def test_probe_response_helper_preserves_reverse_conflict() -> None:
    assert alignment(
        probe_response_direction(
            PROBE_TO_REVERSE
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        -1.0,
        abs=1e-12,
    )

    assert alignment(
        probe_response_direction(
            PROBE_TO_ALIGNED
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_transport_and_probe_response_are_separated_for_reverse_branch() -> None:
    assert alignment(
        transport_direction(
            REVERSE_BRANCH_CHANNEL
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert alignment(
        probe_response_direction(
            PROBE_TO_REVERSE
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        -1.0,
        abs=1e-12,
    )


# =============================================================================
# External validation contract
# =============================================================================


def test_expected_contract_declares_scalar_winner(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        case.expected.amplitude_winner
        == REVERSE_BRANCH_CHANNEL
    )


def test_expected_contract_declares_contradiction_branch(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        case.expected.contradiction_branch
        == REVERSE_BRANCH_CHANNEL
    )


def test_expected_contract_declares_directional_winner(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        case.expected.directional_winner
        == ALIGNED_BRANCH_CHANNEL
    )


def test_expected_scalar_and_directional_winners_are_distinct(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        case.expected.amplitude_winner
        != case.expected.directional_winner
    )


def test_expected_contract_requires_reverse_rejection(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert (
        case.expected.reverse_must_be_rejected
        is True
    )


def test_solver_metadata_does_not_contain_evaluator_winners(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    forbidden_keys = {
        "amplitude_winner",
        "contradiction_branch",
        "directional_winner",
        "expected_amplitude_winner",
        "expected_contradiction_branch",
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
            "external_contradiction_label_used_by_solver"
        ]
        is False
    )

    assert (
        case.pathological_system.metadata[
            "external_directional_winner_used_by_solver"
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
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    assert {
        scenario.scenario_id
        for scenario
        in case.scenarios
    } == {
        SCENARIO_REMOVE_PROBE_NODE,
        SCENARIO_REDUCE_REVERSE,
        SCENARIO_REDUCE_ALIGNED,
        SCENARIO_REDUCE_TERMINAL,
    }


def test_branch_counterfactuals_are_cost_symmetric(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    reverse = case.scenario(
        SCENARIO_REDUCE_REVERSE
    ).interventions[
        0
    ]

    aligned = case.scenario(
        SCENARIO_REDUCE_ALIGNED
    ).interventions[
        0
    ]

    assert reverse.cost == pytest.approx(
        aligned.cost
    )

    assert reverse.safety_risk == pytest.approx(
        aligned.safety_risk
    )

    assert reverse.uncertainty == pytest.approx(
        aligned.uncertainty
    )

    assert reverse.irreversibility == pytest.approx(
        aligned.irreversibility
    )


def test_case_scenario_lookup_rejects_unknown_id(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    with pytest.raises(
        KeyError
    ):
        case.scenario(
            "does_not_exist"
        )


# =============================================================================
# Tensor
# =============================================================================


@pytest.fixture(scope="module")
def tensor(
    case: MisleadingReverseDirectionValidationCase,
):
    return build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )


def test_tensor_builds_for_pathological_system(
    case: MisleadingReverseDirectionValidationCase,
    tensor,
) -> None:
    assert tensor.matrix.shape == (
        len(
            case.channel_ids
        ),
        len(
            case.channel_ids
        ),
    )


def test_tensor_contains_exact_declared_edges(
    tensor,
) -> None:
    channel_index = {
        channel_id: index
        for index, channel_id
        in enumerate(
            tensor.channel_ids
        )
    }

    declared = {
        (
            PROBE_NODE_CHANNEL,
            REVERSE_BRANCH_CHANNEL,
        ),
        (
            PROBE_NODE_CHANNEL,
            ALIGNED_BRANCH_CHANNEL,
        ),
        (
            REVERSE_BRANCH_CHANNEL,
            TERMINAL_CHANNEL,
        ),
        (
            ALIGNED_BRANCH_CHANNEL,
            TERMINAL_CHANNEL,
        ),
    }

    nonzero = set()

    for source_id, source_index in channel_index.items():
        for target_id, target_index in channel_index.items():
            if source_id == target_id:
                continue

            if (
                tensor.matrix[
                    target_index,
                    source_index,
                ]
                != 0.0
            ):
                nonzero.add(
                    (
                        source_id,
                        target_id,
                    )
                )

    assert nonzero == declared


def test_reverse_tensor_entry_exceeds_aligned_tensor_entry(
    tensor,
) -> None:
    index = {
        channel_id: position
        for position, channel_id
        in enumerate(
            tensor.channel_ids
        )
    }

    reverse = float(
        tensor.matrix[
            index[
                REVERSE_BRANCH_CHANNEL
            ],
            index[
                PROBE_NODE_CHANNEL
            ],
        ]
    )

    aligned = float(
        tensor.matrix[
            index[
                ALIGNED_BRANCH_CHANNEL
            ],
            index[
                PROBE_NODE_CHANNEL
            ],
        ]
    )

    assert reverse > aligned


# =============================================================================
# Cascade scalar response
# =============================================================================


@pytest.fixture(scope="module")
def trajectory(
    case: MisleadingReverseDirectionValidationCase,
):
    return run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )


def test_cascade_reaches_reverse_branch(
    trajectory,
) -> None:
    assert trajectory.peak(
        REVERSE_BRANCH_CHANNEL
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


def test_reverse_branch_has_larger_peak_scalar_response(
    trajectory,
) -> None:
    assert (
        trajectory.peak(
            REVERSE_BRANCH_CHANNEL
        )
        >
        trajectory.peak(
            ALIGNED_BRANCH_CHANNEL
        )
    )


def test_reverse_branch_has_larger_total_exposure(
    trajectory,
) -> None:
    assert (
        trajectory.total_exposure(
            REVERSE_BRANCH_CHANNEL
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
            REVERSE_BRANCH_CHANNEL,
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
    case: MisleadingReverseDirectionValidationCase,
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
    case: MisleadingReverseDirectionValidationCase,
    solution: ROIFSolution,
) -> None:
    result_ids = {
        result.scenario.scenario_id
        for result
        in solution.counterfactual_batch.results
    }

    assert result_ids == {
        scenario.scenario_id
        for scenario
        in case.scenarios
    }


def test_solver_result_exposes_four_role_slots(
    solution: ROIFSolution,
) -> None:
    assert (
        solution.root_result.d_origin
        is not None
    )
    assert (
        solution.root_result.d_fast
        is not None
    )
    assert (
        solution.root_result.d_root
        is not None
    )
    assert (
        solution.root_result.node_star
        is not None
    )


def test_passive_solver_is_not_used_as_vector_contradiction_ground_truth(
    case: MisleadingReverseDirectionValidationCase,
    solution: ROIFSolution,
) -> None:
    assert (
        case.expected.contradiction_branch
        == REVERSE_BRANCH_CHANNEL
    )

    assert (
        solution.root_result.d_fast
        in case.channel_ids
    )


def test_solver_does_not_receive_external_winner_labels(
    case: MisleadingReverseDirectionValidationCase,
    solution: ROIFSolution,
) -> None:
    assert (
        solution.metadata.get(
            "expected_labels_used",
            False,
        )
        is False
    )

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
    left = build_misleading_reverse_direction_case()
    right = build_misleading_reverse_direction_case()

    assert (
        left.case_id
        == right.case_id
    )
    assert (
        left.channel_ids
        == right.channel_ids
    )
    assert (
        left.initial_state
        == right.initial_state
    )
    assert (
        left.expected
        == right.expected
    )
    assert (
        left.scenarios
        == right.scenarios
    )


def test_tensor_is_deterministic(
    case: MisleadingReverseDirectionValidationCase,
) -> None:
    left = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    right = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )

    assert (
        left.channel_ids
        == right.channel_ids
    )

    assert np.allclose(
        left.matrix,
        right.matrix,
        rtol=0.0,
        atol=1e-12,
    )


def test_cascade_is_deterministic(
    case: MisleadingReverseDirectionValidationCase,
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

    assert (
        left.channel_ids
        == right.channel_ids
    )

    assert np.allclose(
        left.state_history,
        right.state_history,
        rtol=0.0,
        atol=1e-12,
    )


def test_solver_is_deterministic(
    case: MisleadingReverseDirectionValidationCase,
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

    assert (
        left.status
        is right.status
    )

    assert (
        left.decision
        == right.decision
    )

    assert (
        left.root_result
        == right.root_result
    )

    left_metrics = tuple(
        (
            result.scenario.scenario_id,
            result.metrics.relative_reduction,
            result.metrics.cascade_reduction,
            result.metrics.utility,
            result.metrics.spectral_gain,
        )
        for result
        in left.counterfactual_batch.results
    )

    right_metrics = tuple(
        (
            result.scenario.scenario_id,
            result.metrics.relative_reduction,
            result.metrics.cascade_reduction,
            result.metrics.utility,
            result.metrics.spectral_gain,
        )
        for result
        in right.counterfactual_batch.results
    )

    assert (
        left_metrics
        == right_metrics
    )
