"""
ROIF Mechanical Validation
Scenario 03E вЂ” Delayed Misleading Response Ground Truth

This suite validates the temporal-precedence conflict before Active Probe
inference is introduced.

Goals
-----
1. Verify the graph contains two competing branches.
2. Verify the fast branch becomes visible inside the early observation window.
3. Verify the delayed branch is absent from the early observation window.
4. Verify both branches are visible in the full observation window.
5. Verify first response and evaluator-side causal winner are distinct.
6. Verify both branches remain valid passive mechanical paths to one terminal.
7. Verify evaluator-side temporal labels do not leak into Solver inputs.
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

from validation.mechanical.delayed_misleading_response_case import (
    CANDIDATE_EDGE_DIRECTION,
    DELAYED_BRANCH_CHANNEL,
    DELAYED_GAIN,
    DELAYED_RESPONSE_DELAY_SECONDS,
    DELAYED_RESPONSE_DIRECTION,
    DELAYED_TO_TERMINAL,
    DELAYED_TRANSPORT_DIRECTION,
    DelayedMisleadingResponseValidationCase,
    EARLY_OBSERVATION_WINDOW_SECONDS,
    FAST_BRANCH_CHANNEL,
    FAST_GAIN,
    FAST_RESPONSE_DELAY_SECONDS,
    FAST_RESPONSE_DIRECTION,
    FAST_TO_TERMINAL,
    FAST_TRANSPORT_DIRECTION,
    FULL_OBSERVATION_WINDOW_SECONDS,
    PROBE_NODE_CHANNEL,
    PROBE_TO_DELAYED,
    PROBE_TO_FAST,
    SCENARIO_REDUCE_DELAYED,
    SCENARIO_REDUCE_FAST,
    SCENARIO_REDUCE_TERMINAL,
    SCENARIO_REMOVE_PROBE_NODE,
    TERMINAL_CHANNEL,
    build_delayed_misleading_response_case,
    is_visible_in_window,
    probe_response_direction,
    response_delay_seconds,
)


@pytest.fixture(scope="module")
def case() -> DelayedMisleadingResponseValidationCase:
    return build_delayed_misleading_response_case()


def channel_lookup(
    case: DelayedMisleadingResponseValidationCase,
):
    return {
        channel.channel_id: channel
        for entity in case.pathological_system.entities
        for channel in entity.channels
    }


def operator_lookup(
    case: DelayedMisleadingResponseValidationCase,
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
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert (
        case.case_id
        == "mechanical_03E_delayed_misleading_response"
    )

    assert case.channel_ids == (
        PROBE_NODE_CHANNEL,
        FAST_BRANCH_CHANNEL,
        DELAYED_BRANCH_CHANNEL,
        TERMINAL_CHANNEL,
    )

    assert len(
        case.scenarios
    ) == 4


def test_case_metadata_declares_temporal_conflict(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert (
        case.metadata[
            "topology"
        ]
        == "delayed_misleading_response"
    )

    assert (
        case.metadata[
            "temporal_conflict"
        ]
        is True
    )

    assert (
        case.metadata[
            "expected_first_response_not_final_winner"
        ]
        is True
    )


def test_case_metadata_preserves_non_biological_boundary(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert (
        case.metadata[
            "biological_semantics"
        ]
        is False
    )

    assert (
        case.metadata[
            "diagnostic_claim"
        ]
        is False
    )

    assert (
        case.metadata[
            "treatment_claim"
        ]
        is False
    )


def test_initial_state_is_read_only(
    case: DelayedMisleadingResponseValidationCase,
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
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    assert isinstance(
        labels,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        labels[
            "directional_winner"
        ] = FAST_BRANCH_CHANNEL  # type: ignore[index]


def test_validation_case_is_frozen(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        case.case_id = "x"  # type: ignore[misc]


# =============================================================================
# Temporal ground truth
# =============================================================================


def test_fast_response_delay_is_positive() -> None:
    assert (
        FAST_RESPONSE_DELAY_SECONDS
        > 0.0
    )


def test_delayed_response_delay_is_positive() -> None:
    assert (
        DELAYED_RESPONSE_DELAY_SECONDS
        > 0.0
    )


def test_fast_response_precedes_delayed_response() -> None:
    assert (
        FAST_RESPONSE_DELAY_SECONDS
        < DELAYED_RESPONSE_DELAY_SECONDS
    )


def test_early_window_contains_fast_response() -> None:
    assert is_visible_in_window(
        PROBE_TO_FAST,
        window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    ) is True


def test_early_window_excludes_delayed_response() -> None:
    assert is_visible_in_window(
        PROBE_TO_DELAYED,
        window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    ) is False


def test_full_window_contains_fast_response() -> None:
    assert is_visible_in_window(
        PROBE_TO_FAST,
        window_seconds=(
            FULL_OBSERVATION_WINDOW_SECONDS
        ),
    ) is True


def test_full_window_contains_delayed_response() -> None:
    assert is_visible_in_window(
        PROBE_TO_DELAYED,
        window_seconds=(
            FULL_OBSERVATION_WINDOW_SECONDS
        ),
    ) is True


def test_early_window_is_shorter_than_delayed_response_delay() -> None:
    assert (
        EARLY_OBSERVATION_WINDOW_SECONDS
        < DELAYED_RESPONSE_DELAY_SECONDS
    )


def test_full_window_exceeds_delayed_response_delay() -> None:
    assert (
        FULL_OBSERVATION_WINDOW_SECONDS
        >= DELAYED_RESPONSE_DELAY_SECONDS
    )


def test_response_delay_helper_is_exact() -> None:
    assert response_delay_seconds(
        PROBE_TO_FAST
    ) == pytest.approx(
        FAST_RESPONSE_DELAY_SECONDS
    )

    assert response_delay_seconds(
        PROBE_TO_DELAYED
    ) == pytest.approx(
        DELAYED_RESPONSE_DELAY_SECONDS
    )


def test_response_delay_helper_rejects_unknown_relation() -> None:
    with pytest.raises(
        KeyError
    ):
        response_delay_seconds(
            "unknown_relation"
        )


# =============================================================================
# Topology ground truth
# =============================================================================


def test_pathological_graph_has_exactly_four_operators(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert len(
        case.pathological_system.operators
    ) == 4


def test_operator_ids_are_exact(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert {
        operator.operator_id
        for operator
        in case.pathological_system.operators
    } == {
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
        FAST_TO_TERMINAL,
        DELAYED_TO_TERMINAL,
    }


def test_probe_node_has_exactly_two_outgoing_relations(
    case: DelayedMisleadingResponseValidationCase,
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
        FAST_BRANCH_CHANNEL,
        DELAYED_BRANCH_CHANNEL,
    }


def test_fast_branch_reaches_terminal(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    lookup = operator_lookup(
        case
    )

    assert lookup[
        FAST_TO_TERMINAL
    ].target_ids == (
        TERMINAL_CHANNEL,
    )


def test_delayed_branch_reaches_terminal(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    lookup = operator_lookup(
        case
    )

    assert lookup[
        DELAYED_TO_TERMINAL
    ].target_ids == (
        TERMINAL_CHANNEL,
    )


def test_terminal_has_no_outgoing_relation(
    case: DelayedMisleadingResponseValidationCase,
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
# Passive scalar mechanics
# =============================================================================


def test_declared_probe_gains_are_close_but_not_equal() -> None:
    assert (
        FAST_GAIN
        != DELAYED_GAIN
    )

    assert abs(
        FAST_GAIN
        - DELAYED_GAIN
    ) < 0.10


def test_pathological_fast_probe_gain_slightly_exceeds_delayed(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    lookup = operator_lookup(
        case
    )

    assert (
        float(
            lookup[
                PROBE_TO_FAST
            ].gain
        )
        >
        float(
            lookup[
                PROBE_TO_DELAYED
            ].gain
        )
    )


def test_reference_probe_branch_gains_are_equal(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    lookup = {
        operator.operator_id: operator
        for operator
        in case.reference_system.operators
    }

    assert float(
        lookup[
            PROBE_TO_FAST
        ].gain
    ) == pytest.approx(
        float(
            lookup[
                PROBE_TO_DELAYED
            ].gain
        )
    )


# =============================================================================
# Direction ground truth
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


def test_fast_transport_direction_is_forward_x() -> None:
    assert vector_tuple(
        FAST_TRANSPORT_DIRECTION
    ) == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
        )
    )


def test_delayed_transport_direction_is_forward_x() -> None:
    assert vector_tuple(
        DELAYED_TRANSPORT_DIRECTION
    ) == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
        )
    )


def test_fast_response_direction_is_forward() -> None:
    assert alignment(
        FAST_RESPONSE_DIRECTION,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_delayed_response_direction_is_forward() -> None:
    assert alignment(
        DELAYED_RESPONSE_DIRECTION,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_probe_response_direction_helper_is_forward_for_both() -> None:
    assert alignment(
        probe_response_direction(
            PROBE_TO_FAST
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert alignment(
        probe_response_direction(
            PROBE_TO_DELAYED
        ),
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_branch_channel_transport_directions_are_forward(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    channels = channel_lookup(
        case
    )

    assert alignment(
        channels[
            FAST_BRANCH_CHANNEL
        ].direction,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert alignment(
        channels[
            DELAYED_BRANCH_CHANNEL
        ].direction,
        CANDIDATE_EDGE_DIRECTION,
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )


# =============================================================================
# External validation contract
# =============================================================================


def test_expected_contract_declares_first_response_branch(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert (
        case.expected.first_response_branch
        == FAST_BRANCH_CHANNEL
    )


def test_expected_contract_declares_directional_winner(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert (
        case.expected.directional_winner
        == DELAYED_BRANCH_CHANNEL
    )


def test_first_response_and_directional_winner_are_distinct(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert (
        case.expected.first_response_branch
        != case.expected.directional_winner
    )


def test_expected_contract_forbids_first_response_as_final_decision(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert (
        case.expected.first_response_must_not_decide
        is True
    )


def test_solver_metadata_does_not_contain_evaluator_temporal_labels(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    forbidden_keys = {
        "first_response_branch",
        "directional_winner",
        "expected_first_response_branch",
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
            "external_first_response_label_used_by_solver"
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
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    assert {
        scenario.scenario_id
        for scenario
        in case.scenarios
    } == {
        SCENARIO_REMOVE_PROBE_NODE,
        SCENARIO_REDUCE_FAST,
        SCENARIO_REDUCE_DELAYED,
        SCENARIO_REDUCE_TERMINAL,
    }


def test_branch_counterfactuals_are_cost_symmetric(
    case: DelayedMisleadingResponseValidationCase,
) -> None:
    fast = case.scenario(
        SCENARIO_REDUCE_FAST
    ).interventions[
        0
    ]

    delayed = case.scenario(
        SCENARIO_REDUCE_DELAYED
    ).interventions[
        0
    ]

    assert fast.cost == pytest.approx(
        delayed.cost
    )

    assert fast.safety_risk == pytest.approx(
        delayed.safety_risk
    )

    assert fast.uncertainty == pytest.approx(
        delayed.uncertainty
    )

    assert fast.irreversibility == pytest.approx(
        delayed.irreversibility
    )


def test_case_scenario_lookup_rejects_unknown_id(
    case: DelayedMisleadingResponseValidationCase,
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
    case: DelayedMisleadingResponseValidationCase,
):
    return build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )


def test_tensor_builds_for_pathological_system(
    case: DelayedMisleadingResponseValidationCase,
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
            FAST_BRANCH_CHANNEL,
        ),
        (
            PROBE_NODE_CHANNEL,
            DELAYED_BRANCH_CHANNEL,
        ),
        (
            FAST_BRANCH_CHANNEL,
            TERMINAL_CHANNEL,
        ),
        (
            DELAYED_BRANCH_CHANNEL,
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


def test_both_probe_branch_tensor_entries_are_positive(
    tensor,
) -> None:
    index = {
        channel_id: position
        for position, channel_id
        in enumerate(
            tensor.channel_ids
        )
    }

    fast = float(
        tensor.matrix[
            index[
                FAST_BRANCH_CHANNEL
            ],
            index[
                PROBE_NODE_CHANNEL
            ],
        ]
    )

    delayed = float(
        tensor.matrix[
            index[
                DELAYED_BRANCH_CHANNEL
            ],
            index[
                PROBE_NODE_CHANNEL
            ],
        ]
    )

    assert fast > 0.0
    assert delayed > 0.0


# =============================================================================
# Passive cascade
# =============================================================================


@pytest.fixture(scope="module")
def trajectory(
    case: DelayedMisleadingResponseValidationCase,
):
    return run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )


def test_cascade_reaches_fast_branch(
    trajectory,
) -> None:
    assert trajectory.peak(
        FAST_BRANCH_CHANNEL
    ) > 0.0


def test_cascade_reaches_delayed_branch(
    trajectory,
) -> None:
    assert trajectory.peak(
        DELAYED_BRANCH_CHANNEL
    ) > 0.0


def test_cascade_reaches_common_terminal(
    trajectory,
) -> None:
    assert trajectory.peak(
        TERMINAL_CHANNEL
    ) > 0.0


def test_passive_cascade_does_not_encode_probe_response_delay(
    trajectory,
) -> None:
    """
    The current passive cascade has no sub-step Probe-response timing model.
    Temporal precedence is therefore kept in the validation/Probe layer.
    """

    assert (
        trajectory.first_threshold_crossing(
            FAST_BRANCH_CHANNEL,
            1e-12,
        )
        ==
        trajectory.first_threshold_crossing(
            DELAYED_BRANCH_CHANNEL,
            1e-12,
        )
    )


def test_passive_cascade_does_not_decide_temporal_winner(
    case: DelayedMisleadingResponseValidationCase,
    trajectory,
) -> None:
    assert (
        case.expected.first_response_branch
        == FAST_BRANCH_CHANNEL
    )

    assert (
        case.expected.directional_winner
        == DELAYED_BRANCH_CHANNEL
    )

    assert trajectory.peak(
        FAST_BRANCH_CHANNEL
    ) > 0.0

    assert trajectory.peak(
        DELAYED_BRANCH_CHANNEL
    ) > 0.0


# =============================================================================
# Solver integration
# =============================================================================


@pytest.fixture(scope="module")
def solution(
    case: DelayedMisleadingResponseValidationCase,
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
    case: DelayedMisleadingResponseValidationCase,
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


def test_passive_solver_is_not_used_as_temporal_ground_truth(
    case: DelayedMisleadingResponseValidationCase,
    solution: ROIFSolution,
) -> None:
    assert (
        case.expected.first_response_branch
        == FAST_BRANCH_CHANNEL
    )

    assert (
        case.expected.directional_winner
        == DELAYED_BRANCH_CHANNEL
    )

    assert (
        solution.root_result.d_fast
        in case.channel_ids
    )


def test_solver_does_not_receive_external_temporal_labels(
    case: DelayedMisleadingResponseValidationCase,
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
    left = build_delayed_misleading_response_case()
    right = build_delayed_misleading_response_case()

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


def test_temporal_visibility_is_deterministic() -> None:
    for relation_id in (
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
    ):
        for window in (
            EARLY_OBSERVATION_WINDOW_SECONDS,
            FULL_OBSERVATION_WINDOW_SECONDS,
        ):
            assert is_visible_in_window(
                relation_id,
                window_seconds=window,
            ) is is_visible_in_window(
                relation_id,
                window_seconds=window,
            )


def test_tensor_is_deterministic(
    case: DelayedMisleadingResponseValidationCase,
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
    case: DelayedMisleadingResponseValidationCase,
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
    case: DelayedMisleadingResponseValidationCase,
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

