"""
Mechanical Validation — Scenario 02P
Physics Ground Truth Tests for a Pre-Stressed Parallel Spring Network

This suite validates the analytical benchmark itself.

Important:
    ROIF is intentionally NOT imported here.

The benchmark must be mechanically self-consistent before any comparison
with ROIF is attempted.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from validation.mechanical.prestressed_spring_ground_truth import (
    AxialMember,
    MechanicalGroundTruth,
    MechanicalGroundTruthError,
    assert_ground_truth_consistency,
    build_physics_ground_truth,
    expected_numeric_results,
    force_after_single_branch_loss,
    parallel_force_state,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ground_truth() -> MechanicalGroundTruth:
    return build_physics_ground_truth()


# ---------------------------------------------------------------------------
# Independence contract
# ---------------------------------------------------------------------------


def test_ground_truth_case_id(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.case_id
        == "mechanical_02P_parallel_spring_ground_truth"
    )


def test_ground_truth_is_explicitly_mechanical(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert ground_truth.metadata["domain"] == "mechanical"
    assert (
        ground_truth.metadata["model"]
        == "linear_axial_springs"
    )
    assert (
        ground_truth.metadata["biological_semantics"]
        is False
    )


def test_ground_truth_declares_roif_independence(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.metadata[
            "roif_used_to_generate_ground_truth"
        ]
        is False
    )


def test_ground_truth_uses_si_units(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert ground_truth.metadata["units"] == "SI"


def test_ground_truth_metadata_is_read_only(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert isinstance(
        ground_truth.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        ground_truth.metadata["domain"] = "other"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Axial stiffness: k = E A / L
# ---------------------------------------------------------------------------


def test_healthy_primary_stiffness_is_1000_n_per_m(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_primary.axial_stiffness_n_per_m
        == pytest.approx(1000.0)
    )


def test_defective_primary_stiffness_is_250_n_per_m(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_primary.axial_stiffness_n_per_m
        == pytest.approx(250.0)
    )


def test_bypass_stiffness_is_500_n_per_m(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.bypass.axial_stiffness_n_per_m
        == pytest.approx(500.0)
    )


def test_downstream_stiffness_is_800_n_per_m(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.downstream.axial_stiffness_n_per_m
        == pytest.approx(800.0)
    )


def test_primary_defect_reduces_stiffness_by_factor_four(
    ground_truth: MechanicalGroundTruth,
) -> None:
    healthy = (
        ground_truth.healthy_primary.axial_stiffness_n_per_m
    )

    defective = (
        ground_truth.defective_primary.axial_stiffness_n_per_m
    )

    assert defective == pytest.approx(
        healthy / 4.0
    )


# ---------------------------------------------------------------------------
# Applied load / prestress
# ---------------------------------------------------------------------------


def test_preload_is_120_n(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.load_case.preload_n
        == pytest.approx(120.0)
    )


def test_external_increment_is_120_n(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.load_case.external_increment_n
        == pytest.approx(120.0)
    )


def test_total_force_is_240_n(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.load_case.total_force_n
        == pytest.approx(240.0)
    )


# ---------------------------------------------------------------------------
# Healthy parallel network
# ---------------------------------------------------------------------------


def test_healthy_parallel_elongation_is_016_m(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.elongation_m
        == pytest.approx(0.16)
    )


def test_healthy_primary_carries_160_n(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.primary_force_n
        == pytest.approx(160.0)
    )


def test_healthy_bypass_carries_80_n(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.bypass_force_n
        == pytest.approx(80.0)
    )


def test_healthy_parallel_forces_sum_to_total_force(
    ground_truth: MechanicalGroundTruth,
) -> None:
    total = (
        ground_truth.healthy_state.primary_force_n
        + ground_truth.healthy_state.bypass_force_n
    )

    assert total == pytest.approx(
        ground_truth.load_case.total_force_n
    )


def test_healthy_downstream_member_carries_total_force(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.downstream_force_n
        == pytest.approx(
            ground_truth.load_case.total_force_n
        )
    )


def test_healthy_primary_is_below_allowable_force(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.primary_overloaded
        is False
    )

    assert (
        ground_truth.healthy_state.primary_utilization
        < 1.0
    )


def test_healthy_bypass_is_below_allowable_force(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.bypass_overloaded
        is False
    )

    assert (
        ground_truth.healthy_state.bypass_utilization
        < 1.0
    )


def test_healthy_downstream_is_below_allowable_force(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.downstream_overloaded
        is False
    )

    assert (
        ground_truth.healthy_state.downstream_utilization
        < 1.0
    )


def test_healthy_network_has_no_overload(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.healthy_state.any_overload
        is False
    )


# ---------------------------------------------------------------------------
# Defective primary: load redistribution
# ---------------------------------------------------------------------------


def test_defective_parallel_elongation_doubles(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_state.elongation_m
        == pytest.approx(0.32)
    )

    assert (
        ground_truth.defective_state.elongation_m
        == pytest.approx(
            ground_truth.healthy_state.elongation_m * 2.0
        )
    )


def test_primary_force_drops_to_80_n_after_stiffness_loss(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_state.primary_force_n
        == pytest.approx(80.0)
    )


def test_bypass_force_rises_to_160_n_after_stiffness_loss(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_state.bypass_force_n
        == pytest.approx(160.0)
    )


def test_defective_parallel_forces_still_sum_to_total_force(
    ground_truth: MechanicalGroundTruth,
) -> None:
    total = (
        ground_truth.defective_state.primary_force_n
        + ground_truth.defective_state.bypass_force_n
    )

    assert total == pytest.approx(
        ground_truth.load_case.total_force_n
    )


def test_primary_remains_below_force_limit_immediately_after_defect(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_state.primary_overloaded
        is False
    )

    assert (
        ground_truth.defective_state.primary_utilization
        < 1.0
    )


def test_bypass_exceeds_force_limit_after_primary_stiffness_loss(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_state.bypass_overloaded
        is True
    )

    assert (
        ground_truth.defective_state.bypass_utilization
        > 1.0
    )


def test_downstream_remains_below_force_limit_after_primary_defect(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_state.downstream_overloaded
        is False
    )


def test_defective_network_contains_overload(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert (
        ground_truth.defective_state.any_overload
        is True
    )


# ---------------------------------------------------------------------------
# Physical cascade sequence
# ---------------------------------------------------------------------------


def test_bypass_is_first_physical_failure(
    ground_truth: MechanicalGroundTruth,
) -> None:
    first = ground_truth.failure_sequence[0]

    assert first.step_index == 1
    assert (
        first.failed_member_id
        == "bypass_branch"
    )

    assert first.force_at_failure_n == pytest.approx(
        160.0
    )

    assert first.utilization > 1.0


def test_primary_is_second_physical_failure(
    ground_truth: MechanicalGroundTruth,
) -> None:
    second = ground_truth.failure_sequence[1]

    assert second.step_index == 2
    assert (
        second.failed_member_id
        == "primary_branch"
    )

    assert second.force_at_failure_n == pytest.approx(
        240.0
    )

    assert second.utilization > 1.0


def test_primary_carries_full_load_after_bypass_loss(
    ground_truth: MechanicalGroundTruth,
) -> None:
    force_n, utilization, overloaded = (
        force_after_single_branch_loss(
            ground_truth.defective_primary,
            total_force_n=(
                ground_truth.load_case.total_force_n
            ),
        )
    )

    assert force_n == pytest.approx(240.0)
    assert utilization == pytest.approx(
        240.0
        / ground_truth.defective_primary.allowable_force_n
    )
    assert overloaded is True


def test_failure_sequence_is_ordered_and_unique(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert tuple(
        step.step_index
        for step in ground_truth.failure_sequence
    ) == (1, 2)

    assert tuple(
        step.failed_member_id
        for step in ground_truth.failure_sequence
    ) == (
        "bypass_branch",
        "primary_branch",
    )


# ---------------------------------------------------------------------------
# Analytical equations and utility functions
# ---------------------------------------------------------------------------


def test_parallel_force_solver_reproduces_healthy_state(
    ground_truth: MechanicalGroundTruth,
) -> None:
    solved = parallel_force_state(
        ground_truth.healthy_primary,
        ground_truth.bypass,
        ground_truth.downstream,
        total_force_n=(
            ground_truth.load_case.total_force_n
        ),
    )

    assert solved == ground_truth.healthy_state


def test_parallel_force_solver_reproduces_defective_state(
    ground_truth: MechanicalGroundTruth,
) -> None:
    solved = parallel_force_state(
        ground_truth.defective_primary,
        ground_truth.bypass,
        ground_truth.downstream,
        total_force_n=(
            ground_truth.load_case.total_force_n
        ),
    )

    assert solved == ground_truth.defective_state


def test_parallel_force_solver_rejects_compression_for_tension_only_case(
    ground_truth: MechanicalGroundTruth,
) -> None:
    with pytest.raises(
        MechanicalGroundTruthError
    ):
        parallel_force_state(
            ground_truth.defective_primary,
            ground_truth.bypass,
            ground_truth.downstream,
            total_force_n=-1.0,
        )


def test_axial_member_rejects_nonpositive_length() -> None:
    with pytest.raises(
        MechanicalGroundTruthError
    ):
        AxialMember(
            member_id="bad",
            length_m=0.0,
            area_m2=1.0e-4,
            young_modulus_pa=1.0e6,
            allowable_force_n=100.0,
        )


def test_axial_member_rejects_nonpositive_area() -> None:
    with pytest.raises(
        MechanicalGroundTruthError
    ):
        AxialMember(
            member_id="bad",
            length_m=1.0,
            area_m2=0.0,
            young_modulus_pa=1.0e6,
            allowable_force_n=100.0,
        )


def test_axial_member_rejects_nonpositive_modulus() -> None:
    with pytest.raises(
        MechanicalGroundTruthError
    ):
        AxialMember(
            member_id="bad",
            length_m=1.0,
            area_m2=1.0e-4,
            young_modulus_pa=0.0,
            allowable_force_n=100.0,
        )


def test_axial_member_rejects_nonpositive_allowable_force() -> None:
    with pytest.raises(
        MechanicalGroundTruthError
    ):
        AxialMember(
            member_id="bad",
            length_m=1.0,
            area_m2=1.0e-4,
            young_modulus_pa=1.0e6,
            allowable_force_n=0.0,
        )


# ---------------------------------------------------------------------------
# Ground-truth values
# ---------------------------------------------------------------------------


def test_expected_numeric_results_are_read_only() -> None:
    expected = expected_numeric_results()

    assert isinstance(
        expected,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        expected["total_force_n"] = 1.0  # type: ignore[index]


def test_expected_numeric_results_match_case(
    ground_truth: MechanicalGroundTruth,
) -> None:
    expected = expected_numeric_results()

    assert (
        ground_truth.healthy_primary.axial_stiffness_n_per_m
        == pytest.approx(
            expected[
                "healthy_primary_stiffness_n_per_m"
            ]
        )
    )

    assert (
        ground_truth.defective_primary.axial_stiffness_n_per_m
        == pytest.approx(
            expected[
                "defective_primary_stiffness_n_per_m"
            ]
        )
    )

    assert (
        ground_truth.defective_state.bypass_force_n
        == pytest.approx(
            expected[
                "defective_bypass_force_n"
            ]
        )
    )


def test_ground_truth_consistency_guard_passes(
    ground_truth: MechanicalGroundTruth,
) -> None:
    assert_ground_truth_consistency(
        ground_truth
    )


# ---------------------------------------------------------------------------
# Immutability and determinism
# ---------------------------------------------------------------------------


def test_axial_member_is_immutable(
    ground_truth: MechanicalGroundTruth,
) -> None:
    with pytest.raises(FrozenInstanceError):
        ground_truth.bypass.area_m2 = 1.0  # type: ignore[misc]


def test_complete_ground_truth_is_deterministic() -> None:
    left = build_physics_ground_truth()
    right = build_physics_ground_truth()

    assert left == right


def test_ground_truth_consistency_is_deterministic() -> None:
    left = build_physics_ground_truth()
    right = build_physics_ground_truth()

    assert_ground_truth_consistency(left)
    assert_ground_truth_consistency(right)

    assert left.failure_sequence == right.failure_sequence


def test_scenario_02p_end_to_end_physics_invariant(
    ground_truth: MechanicalGroundTruth,
) -> None:
    """
    Final independent physics invariant.

    A primary stiffness defect must redistribute load into the bypass,
    overload the bypass first, and then overload the defective primary
    after bypass loss.
    """

    assert (
        ground_truth.metadata[
            "roif_used_to_generate_ground_truth"
        ]
        is False
    )

    assert (
        ground_truth.healthy_state.any_overload
        is False
    )

    assert (
        ground_truth.defective_state.primary_overloaded
        is False
    )

    assert (
        ground_truth.defective_state.bypass_overloaded
        is True
    )

    assert tuple(
        step.failed_member_id
        for step in ground_truth.failure_sequence
    ) == (
        "bypass_branch",
        "primary_branch",
    )

    assert_ground_truth_consistency(
        ground_truth
    )
