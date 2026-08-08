"""
Tests for roif.utilization

These tests validate the physical utilization layer independently from
D_fast / D_root / Node* logic.

Core invariant:

    utilization = |demand| / effective_capacity

Functional failure is defined by utilization crossing the configured
threshold.

The suite also contains a small regression bridge to the independently
validated pre-stressed spring benchmark.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from types import MappingProxyType

import numpy as np
import pytest

from roif.utilization import (
    FailureCriterion,
    FailureEvent,
    UtilizationError,
    UtilizationState,
    UtilizationTrajectory,
    build_utilization_trajectory,
    compute_reserve,
    compute_reserve_margin,
    compute_utilization,
    earliest_failure_event,
    evaluate_utilization_state,
    first_capacity_exceedance,
    utilization_matrix,
)

from validation.mechanical.prestressed_spring_ground_truth import (
    MechanicalGroundTruth,
    build_physics_ground_truth,
)


# ---------------------------------------------------------------------------
# compute_utilization
# ---------------------------------------------------------------------------


def test_compute_utilization_basic_ratio() -> None:
    assert compute_utilization(
        50.0,
        100.0,
    ) == pytest.approx(0.5)


def test_compute_utilization_uses_demand_magnitude() -> None:
    assert compute_utilization(
        -50.0,
        100.0,
    ) == pytest.approx(0.5)


def test_compute_utilization_at_capacity_is_one() -> None:
    assert compute_utilization(
        100.0,
        100.0,
    ) == pytest.approx(1.0)


def test_compute_utilization_over_capacity_exceeds_one() -> None:
    assert compute_utilization(
        150.0,
        100.0,
    ) == pytest.approx(1.5)


def test_compute_utilization_zero_demand_zero_capacity_is_zero() -> None:
    assert compute_utilization(
        0.0,
        0.0,
    ) == pytest.approx(0.0)


def test_compute_utilization_nonzero_demand_zero_capacity_is_infinite() -> None:
    result = compute_utilization(
        1.0,
        0.0,
    )

    assert math.isinf(result)
    assert result > 0.0


def test_compute_utilization_rejects_negative_capacity() -> None:
    with pytest.raises(
        UtilizationError
    ):
        compute_utilization(
            1.0,
            -1.0,
        )


def test_compute_utilization_rejects_nan_demand() -> None:
    with pytest.raises(
        UtilizationError
    ):
        compute_utilization(
            math.nan,
            1.0,
        )


def test_compute_utilization_rejects_infinite_capacity() -> None:
    with pytest.raises(
        UtilizationError
    ):
        compute_utilization(
            1.0,
            math.inf,
        )


def test_compute_utilization_rejects_negative_epsilon() -> None:
    with pytest.raises(
        UtilizationError
    ):
        compute_utilization(
            1.0,
            1.0,
            epsilon=-1.0,
        )


# ---------------------------------------------------------------------------
# reserve
# ---------------------------------------------------------------------------


def test_compute_reserve_below_capacity_is_positive() -> None:
    assert compute_reserve(
        0.75
    ) == pytest.approx(0.25)


def test_compute_reserve_at_capacity_is_zero() -> None:
    assert compute_reserve(
        1.0
    ) == pytest.approx(0.0)


def test_compute_reserve_overload_is_negative() -> None:
    assert compute_reserve(
        1.25
    ) == pytest.approx(-0.25)


def test_compute_reserve_margin_clips_negative_reserve() -> None:
    assert compute_reserve_margin(
        1.25
    ) == pytest.approx(0.0)


def test_compute_reserve_margin_preserves_positive_reserve() -> None:
    assert compute_reserve_margin(
        0.25
    ) == pytest.approx(0.75)


def test_compute_reserve_rejects_negative_utilization() -> None:
    with pytest.raises(
        UtilizationError
    ):
        compute_reserve(
            -0.1
        )


# ---------------------------------------------------------------------------
# scalar utilization state
# ---------------------------------------------------------------------------


def test_evaluate_utilization_state_below_threshold() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=80.0,
        effective_capacity=100.0,
    )

    assert isinstance(
        state,
        UtilizationState,
    )
    assert state.utilization == pytest.approx(0.8)
    assert state.reserve == pytest.approx(0.2)
    assert state.reserve_margin == pytest.approx(0.2)
    assert state.overloaded is False


def test_evaluate_utilization_state_at_threshold_fails_by_default() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=100.0,
        effective_capacity=100.0,
    )

    assert state.utilization == pytest.approx(1.0)
    assert state.overloaded is True


def test_gt_threshold_does_not_fail_at_exact_threshold() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=100.0,
        effective_capacity=100.0,
        criterion=FailureCriterion.GT_THRESHOLD,
    )

    assert state.utilization == pytest.approx(1.0)
    assert state.overloaded is False


def test_gt_threshold_fails_above_threshold() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=101.0,
        effective_capacity=100.0,
        criterion=FailureCriterion.GT_THRESHOLD,
    )

    assert state.overloaded is True


def test_evaluate_state_zero_capacity_nonzero_demand_is_overload() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=1.0,
        effective_capacity=0.0,
    )

    assert math.isinf(
        state.utilization
    )
    assert state.overloaded is True
    assert state.reserve == -math.inf
    assert state.reserve_margin == pytest.approx(0.0)


def test_utilization_state_metadata_is_read_only() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=1.0,
        effective_capacity=2.0,
        metadata={
            "source": "test",
        },
    )

    assert isinstance(
        state.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        state.metadata["source"] = "changed"  # type: ignore[index]


def test_utilization_state_is_frozen() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=1.0,
        effective_capacity=2.0,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        state.demand = 3.0  # type: ignore[misc]


def test_evaluate_state_preserves_time_index() -> None:
    state = evaluate_utilization_state(
        "member",
        demand=1.0,
        effective_capacity=2.0,
        time_index=7,
    )

    assert state.time_index == 7


def test_evaluate_state_rejects_negative_time_index() -> None:
    with pytest.raises(
        UtilizationError
    ):
        evaluate_utilization_state(
            "member",
            demand=1.0,
            effective_capacity=2.0,
            time_index=-1,
        )


def test_evaluate_state_rejects_nonpositive_threshold() -> None:
    with pytest.raises(
        UtilizationError
    ):
        evaluate_utilization_state(
            "member",
            demand=1.0,
            effective_capacity=2.0,
            threshold=0.0,
        )


# ---------------------------------------------------------------------------
# utilization trajectory
# ---------------------------------------------------------------------------


def test_build_trajectory_with_scalar_capacity() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            0.0,
            25.0,
            50.0,
            100.0,
            125.0,
        ),
        capacities=100.0,
    )

    assert isinstance(
        trajectory,
        UtilizationTrajectory,
    )

    assert trajectory.utilization.tolist() == pytest.approx(
        [
            0.0,
            0.25,
            0.5,
            1.0,
            1.25,
        ]
    )


def test_trajectory_signed_reserve_history() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            50.0,
            100.0,
            150.0,
        ),
        capacities=100.0,
    )

    assert trajectory.reserve.tolist() == pytest.approx(
        [
            0.5,
            0.0,
            -0.5,
        ]
    )


def test_trajectory_failure_flags() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            50.0,
            99.0,
            100.0,
            101.0,
        ),
        capacities=100.0,
    )

    assert trajectory.overloaded.tolist() == [
        False,
        False,
        True,
        True,
    ]


def test_trajectory_first_failure_index() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            50.0,
            99.0,
            100.0,
            101.0,
        ),
        capacities=100.0,
    )

    assert trajectory.first_failure_index == 2


def test_trajectory_failed_property() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            50.0,
            100.0,
        ),
        capacities=100.0,
    )

    assert trajectory.failed is True


def test_trajectory_without_failure_returns_none() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            10.0,
            20.0,
            30.0,
        ),
        capacities=100.0,
    )

    assert trajectory.first_failure_index is None
    assert trajectory.failed is False
    assert trajectory.first_failure_event() is None


def test_trajectory_peak_utilization() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            10.0,
            150.0,
            50.0,
        ),
        capacities=100.0,
    )

    assert trajectory.peak_utilization == pytest.approx(1.5)


def test_trajectory_minimum_reserve() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            10.0,
            150.0,
            50.0,
        ),
        capacities=100.0,
    )

    assert trajectory.minimum_reserve == pytest.approx(-0.5)


def test_trajectory_state_at_returns_correct_state() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            50.0,
            100.0,
            150.0,
        ),
        capacities=100.0,
    )

    state = trajectory.state_at(
        1
    )

    assert state.time_index == 1
    assert state.demand == pytest.approx(100.0)
    assert state.effective_capacity == pytest.approx(100.0)
    assert state.utilization == pytest.approx(1.0)
    assert state.overloaded is True


def test_trajectory_state_at_rejects_out_of_range_index() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            1.0,
            2.0,
        ),
        capacities=10.0,
    )

    with pytest.raises(
        UtilizationError
    ):
        trajectory.state_at(
            2
        )


def test_trajectory_arrays_are_read_only() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            1.0,
            2.0,
        ),
        capacities=10.0,
    )

    assert trajectory.demand.flags.writeable is False
    assert trajectory.effective_capacity.flags.writeable is False
    assert trajectory.utilization.flags.writeable is False
    assert trajectory.reserve.flags.writeable is False
    assert trajectory.overloaded.flags.writeable is False

    with pytest.raises(ValueError):
        trajectory.utilization[0] = 10.0


def test_trajectory_supports_time_varying_capacity() -> None:
    trajectory = build_utilization_trajectory(
        "member",
        demands=(
            50.0,
            50.0,
            50.0,
        ),
        capacities=(
            100.0,
            60.0,
            40.0,
        ),
    )

    assert trajectory.utilization.tolist() == pytest.approx(
        [
            0.5,
            50.0 / 60.0,
            1.25,
        ]
    )

    assert trajectory.first_failure_index == 2


def test_trajectory_rejects_mismatched_lengths() -> None:
    with pytest.raises(
        UtilizationError
    ):
        build_utilization_trajectory(
            "member",
            demands=(
                1.0,
                2.0,
            ),
            capacities=(
                1.0,
            ),
        )


def test_trajectory_rejects_negative_capacity() -> None:
    with pytest.raises(
        UtilizationError
    ):
        build_utilization_trajectory(
            "member",
            demands=(
                1.0,
            ),
            capacities=(
                -1.0,
            ),
        )


# ---------------------------------------------------------------------------
# failure event
# ---------------------------------------------------------------------------


def test_first_capacity_exceedance_returns_event() -> None:
    event = first_capacity_exceedance(
        "member",
        demands=(
            20.0,
            50.0,
            100.0,
            120.0,
        ),
        capacities=100.0,
    )

    assert isinstance(
        event,
        FailureEvent,
    )

    assert event.time_index == 2
    assert event.channel_id == "member"
    assert event.demand == pytest.approx(100.0)
    assert event.effective_capacity == pytest.approx(100.0)
    assert event.utilization == pytest.approx(1.0)
    assert event.reserve == pytest.approx(0.0)


def test_first_capacity_exceedance_returns_none_without_failure() -> None:
    event = first_capacity_exceedance(
        "member",
        demands=(
            20.0,
            50.0,
            99.0,
        ),
        capacities=100.0,
    )

    assert event is None


def test_failure_event_metadata_is_read_only() -> None:
    event = first_capacity_exceedance(
        "member",
        demands=(
            0.0,
            2.0,
        ),
        capacities=1.0,
        metadata={
            "source": "test",
        },
    )

    assert event is not None

    assert isinstance(
        event.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        event.metadata["source"] = "changed"  # type: ignore[index]


# ---------------------------------------------------------------------------
# earliest event across channels
# ---------------------------------------------------------------------------


def test_earliest_failure_event_selects_smallest_time_index() -> None:
    early = build_utilization_trajectory(
        "early",
        demands=(
            0.0,
            100.0,
            120.0,
        ),
        capacities=100.0,
    )

    late = build_utilization_trajectory(
        "late",
        demands=(
            0.0,
            50.0,
            100.0,
        ),
        capacities=100.0,
    )

    event = earliest_failure_event(
        (
            late,
            early,
        )
    )

    assert event is not None
    assert event.channel_id == "early"
    assert event.time_index == 1


def test_earliest_failure_event_prefers_larger_utilization_on_tie() -> None:
    weaker = build_utilization_trajectory(
        "weaker",
        demands=(
            0.0,
            100.0,
        ),
        capacities=100.0,
    )

    stronger = build_utilization_trajectory(
        "stronger",
        demands=(
            0.0,
            150.0,
        ),
        capacities=100.0,
    )

    event = earliest_failure_event(
        (
            weaker,
            stronger,
        )
    )

    assert event is not None
    assert event.time_index == 1
    assert event.channel_id == "stronger"
    assert event.utilization == pytest.approx(1.5)


def test_earliest_failure_event_uses_channel_id_for_exact_tie() -> None:
    zeta = build_utilization_trajectory(
        "zeta",
        demands=(
            0.0,
            100.0,
        ),
        capacities=100.0,
    )

    alpha = build_utilization_trajectory(
        "alpha",
        demands=(
            0.0,
            100.0,
        ),
        capacities=100.0,
    )

    event = earliest_failure_event(
        (
            zeta,
            alpha,
        )
    )

    assert event is not None
    assert event.channel_id == "alpha"


def test_earliest_failure_event_returns_none_when_no_channel_fails() -> None:
    a = build_utilization_trajectory(
        "a",
        demands=(
            0.0,
            50.0,
        ),
        capacities=100.0,
    )

    b = build_utilization_trajectory(
        "b",
        demands=(
            0.0,
            75.0,
        ),
        capacities=100.0,
    )

    assert earliest_failure_event(
        (
            a,
            b,
        )
    ) is None


# ---------------------------------------------------------------------------
# utilization matrix
# ---------------------------------------------------------------------------


def test_utilization_matrix_basic() -> None:
    demands = np.asarray(
        [
            [10.0, 20.0],
            [50.0, 100.0],
        ],
        dtype=np.float64,
    )

    capacities = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 200.0],
        ],
        dtype=np.float64,
    )

    result = utilization_matrix(
        demands,
        capacities,
    )

    expected = np.asarray(
        [
            [0.1, 0.2],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )

    assert result.shape == expected.shape
    assert np.allclose(
        result,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_utilization_matrix_handles_zero_capacity() -> None:
    result = utilization_matrix(
        [
            [0.0, 1.0],
        ],
        [
            [0.0, 0.0],
        ],
    )

    assert result[0, 0] == pytest.approx(0.0)
    assert math.isinf(
        float(
            result[0, 1]
        )
    )


def test_utilization_matrix_rejects_shape_mismatch() -> None:
    with pytest.raises(
        UtilizationError
    ):
        utilization_matrix(
            [
                [1.0, 2.0],
            ],
            [
                [1.0],
            ],
        )


# ---------------------------------------------------------------------------
# spring ground-truth bridge
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spring_gt() -> MechanicalGroundTruth:
    return build_physics_ground_truth()


def test_spring_defective_primary_utilization_matches_physics(
    spring_gt: MechanicalGroundTruth,
) -> None:
    result = compute_utilization(
        spring_gt.defective_state.primary_force_n,
        spring_gt.defective_primary.allowable_force_n,
    )

    assert result == pytest.approx(
        80.0 / 190.0
    )

    assert result < 1.0


def test_spring_bypass_utilization_matches_physics(
    spring_gt: MechanicalGroundTruth,
) -> None:
    result = compute_utilization(
        spring_gt.defective_state.bypass_force_n,
        spring_gt.bypass.allowable_force_n,
    )

    assert result == pytest.approx(
        160.0 / 130.0
    )

    assert result > 1.0


def test_spring_downstream_utilization_matches_physics(
    spring_gt: MechanicalGroundTruth,
) -> None:
    result = compute_utilization(
        spring_gt.defective_state.downstream_force_n,
        spring_gt.downstream.allowable_force_n,
    )

    assert result == pytest.approx(
        240.0 / 300.0
    )

    assert result < 1.0


def test_spring_defective_snapshot_identifies_only_bypass_overload(
    spring_gt: MechanicalGroundTruth,
) -> None:
    primary = evaluate_utilization_state(
        "primary_branch",
        demand=(
            spring_gt.defective_state.primary_force_n
        ),
        effective_capacity=(
            spring_gt.defective_primary.allowable_force_n
        ),
    )

    bypass = evaluate_utilization_state(
        "bypass_branch",
        demand=(
            spring_gt.defective_state.bypass_force_n
        ),
        effective_capacity=(
            spring_gt.bypass.allowable_force_n
        ),
    )

    downstream = evaluate_utilization_state(
        "downstream_member",
        demand=(
            spring_gt.defective_state.downstream_force_n
        ),
        effective_capacity=(
            spring_gt.downstream.allowable_force_n
        ),
    )

    assert primary.overloaded is False
    assert bypass.overloaded is True
    assert downstream.overloaded is False


def test_spring_first_failure_event_is_bypass_branch(
    spring_gt: MechanicalGroundTruth,
) -> None:
    """
    Regression bridge to the independent mechanical benchmark.

    We encode three physical stages:

    t0: healthy
    t1: primary stiffness defect redistributes force
    t2: bypass lost, primary receives full force

    The first physical capacity crossing must therefore occur at t1 in
    bypass_branch.
    """

    primary = build_utilization_trajectory(
        "primary_branch",
        demands=(
            spring_gt.healthy_state.primary_force_n,
            spring_gt.defective_state.primary_force_n,
            spring_gt.load_case.total_force_n,
        ),
        capacities=(
            spring_gt.healthy_primary.allowable_force_n,
            spring_gt.defective_primary.allowable_force_n,
            spring_gt.defective_primary.allowable_force_n,
        ),
    )

    bypass = build_utilization_trajectory(
        "bypass_branch",
        demands=(
            spring_gt.healthy_state.bypass_force_n,
            spring_gt.defective_state.bypass_force_n,
            0.0,
        ),
        capacities=(
            spring_gt.bypass.allowable_force_n,
            spring_gt.bypass.allowable_force_n,
            0.0,
        ),
    )

    downstream = build_utilization_trajectory(
        "downstream_member",
        demands=(
            spring_gt.healthy_state.downstream_force_n,
            spring_gt.defective_state.downstream_force_n,
            spring_gt.load_case.total_force_n,
        ),
        capacities=(
            spring_gt.downstream.allowable_force_n,
            spring_gt.downstream.allowable_force_n,
            spring_gt.downstream.allowable_force_n,
        ),
    )

    event = earliest_failure_event(
        (
            primary,
            bypass,
            downstream,
        )
    )

    assert event is not None

    assert event.channel_id == "bypass_branch"
    assert event.time_index == 1
    assert event.demand == pytest.approx(160.0)
    assert event.effective_capacity == pytest.approx(130.0)
    assert event.utilization == pytest.approx(
        160.0 / 130.0
    )


def test_spring_primary_fails_second_after_bypass_loss(
    spring_gt: MechanicalGroundTruth,
) -> None:
    primary = build_utilization_trajectory(
        "primary_branch",
        demands=(
            spring_gt.healthy_state.primary_force_n,
            spring_gt.defective_state.primary_force_n,
            spring_gt.load_case.total_force_n,
        ),
        capacities=(
            spring_gt.healthy_primary.allowable_force_n,
            spring_gt.defective_primary.allowable_force_n,
            spring_gt.defective_primary.allowable_force_n,
        ),
    )

    event = primary.first_failure_event()

    assert event is not None
    assert event.channel_id == "primary_branch"
    assert event.time_index == 2
    assert event.demand == pytest.approx(240.0)
    assert event.effective_capacity == pytest.approx(190.0)
    assert event.utilization == pytest.approx(
        240.0 / 190.0
    )


def test_utilization_layer_reproduces_spring_failure_order(
    spring_gt: MechanicalGroundTruth,
) -> None:
    """
    Final physics invariant for the utilization layer.
    """

    primary = build_utilization_trajectory(
        "primary_branch",
        demands=(
            160.0,
            80.0,
            240.0,
        ),
        capacities=(
            190.0,
            190.0,
            190.0,
        ),
    )

    bypass = build_utilization_trajectory(
        "bypass_branch",
        demands=(
            80.0,
            160.0,
            0.0,
        ),
        capacities=(
            130.0,
            130.0,
            0.0,
        ),
    )

    assert bypass.first_failure_index == 1
    assert primary.first_failure_index == 2

    first = earliest_failure_event(
        (
            primary,
            bypass,
        )
    )

    assert first is not None
    assert first.channel_id == "bypass_branch"

    assert tuple(
        step.failed_member_id
        for step in spring_gt.failure_sequence
    ) == (
        "bypass_branch",
        "primary_branch",
    )

