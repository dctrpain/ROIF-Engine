"""
Tests for roif.fast_layer.

The fast layer solves

    A T = f

and evaluates element utilization

    u_i = |T_i| / κ_i.

The most utilized element is

    D_fast = argmax(u).
"""

from __future__ import annotations

import numpy as np
import pytest

from roif.capacity_tensor import CapacityTensor
from roif.direction import Direction
from roif.fast_layer import (
    FastLayer,
    FastLayerError,
    FastLayerResult,
    SolveMethod,
    solve_fast_layer,
)


def make_capacity(
    base_capacity: float,
    local_axis: list[float],
) -> CapacityTensor:
    """Create a fully available CapacityTensor for fast-layer tests."""

    return CapacityTensor(
        base_capacity=base_capacity,
        availability=1.0,
        local_axis=local_axis,
    )


def test_direct_solver_solves_identity_system() -> None:
    result = FastLayer(method=SolveMethod.DIRECT).solve(
        equilibrium_matrix=np.eye(2),
        load=np.array([300.0, 400.0]),
        capacities=[600.0, 500.0],
    )
    assert np.allclose(result.efforts, [300.0, 400.0])
    assert np.allclose(result.reconstructed_load, [300.0, 400.0])
    assert result.solve_method is SolveMethod.DIRECT
    assert result.is_exact_equilibrium


def test_direct_solver_solves_non_identity_system() -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    expected_efforts = np.array([4.0, 2.0])
    result = FastLayer(method=SolveMethod.DIRECT).solve(
        equilibrium_matrix=matrix,
        load=matrix @ expected_efforts,
        capacities=[10.0, 10.0],
    )
    assert np.allclose(result.efforts, expected_efforts)
    assert result.residual_norm == pytest.approx(0.0)
    assert result.is_exact_equilibrium


def test_direction_load_uses_complete_force_vector() -> None:
    load = Direction(vector=[3.0, 4.0], magnitude=10.0, name="external_load")
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=load,
        capacities=[10.0, 10.0],
    )
    assert np.allclose(load.force_vector, [6.0, 8.0])
    assert np.allclose(result.efforts, [6.0, 8.0])


def test_utilization_is_absolute_effort_over_capacity() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[-300.0, 400.0],
        capacities=[600.0, 500.0],
    )
    assert np.allclose(result.utilization, [0.5, 0.8])


def test_signed_utilization_preserves_effort_sign() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[-300.0, 400.0],
        capacities=[600.0, 500.0],
    )
    assert np.allclose(result.signed_utilization, [-0.5, 0.8])


def test_d_fast_is_most_utilized_element() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(3),
        load=[20.0, 80.0, 30.0],
        capacities=[100.0, 100.0, 100.0],
    )
    assert result.d_fast_index == 1
    assert result.max_utilization == pytest.approx(0.8)


def test_d_fast_uses_first_index_when_maximum_is_tied() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(3),
        load=[80.0, -80.0, 20.0],
        capacities=[100.0, 100.0, 100.0],
    )
    assert result.d_fast_index == 0


def test_zero_effort_and_zero_capacity_give_zero_utilization() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[0.0, 50.0],
        capacities=[0.0, 100.0],
    )
    assert result.utilization[0] == pytest.approx(0.0)
    assert not result.has_unavailable_loaded_element


def test_nonzero_effort_and_zero_capacity_give_infinite_utilization() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[10.0, 50.0],
        capacities=[0.0, 100.0],
    )
    assert np.isinf(result.utilization[0])
    assert result.has_unavailable_loaded_element
    assert result.has_overload


def test_negative_effort_and_zero_capacity_give_negative_signed_infinity() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(1),
        load=[-10.0],
        capacities=[0.0],
    )
    assert np.isneginf(result.signed_utilization[0])


def test_capacity_epsilon_treats_small_capacity_as_unavailable() -> None:
    result = FastLayer(capacity_epsilon=1e-6).solve(
        equilibrium_matrix=np.eye(1),
        load=[1.0],
        capacities=[1e-8],
    )
    assert np.isinf(result.utilization[0])


def test_overloaded_indices_include_only_values_above_one() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(4),
        load=[50.0, 100.0, 120.0, 250.0],
        capacities=[100.0, 100.0, 100.0, 200.0],
    )
    assert result.overloaded_indices == (2, 3)


def test_critical_indices_include_values_equal_to_one() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(4),
        load=[50.0, 100.0, 120.0, 250.0],
        capacities=[100.0, 100.0, 100.0, 200.0],
    )
    assert result.critical_indices == (1, 2, 3)


def test_no_overload_when_all_utilizations_are_below_one() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[20.0, 30.0],
        capacities=[100.0, 100.0],
    )
    assert not result.has_overload
    assert result.overloaded_indices == ()


def test_auto_uses_direct_for_square_full_rank_matrix() -> None:
    result = FastLayer(method=SolveMethod.AUTO).solve(
        equilibrium_matrix=np.eye(3),
        load=[1.0, 2.0, 3.0],
        capacities=[10.0, 10.0, 10.0],
    )
    assert result.solve_method is SolveMethod.DIRECT


def test_auto_uses_least_squares_for_rectangular_matrix() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = np.array([2.0, 3.0])
    result = FastLayer(method=SolveMethod.AUTO).solve(
        equilibrium_matrix=matrix,
        load=matrix @ expected,
        capacities=[10.0, 10.0],
    )
    assert result.solve_method is SolveMethod.LEAST_SQUARES
    assert np.allclose(result.efforts, expected)


def test_auto_uses_least_squares_for_singular_square_matrix() -> None:
    matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    result = FastLayer(method=SolveMethod.AUTO).solve(
        equilibrium_matrix=matrix,
        load=[4.0, 8.0],
        capacities=[10.0, 10.0],
    )
    assert result.solve_method is SolveMethod.LEAST_SQUARES
    assert result.is_exact_equilibrium


def test_explicit_least_squares_reports_inconsistent_residual() -> None:
    result = FastLayer(
        method=SolveMethod.LEAST_SQUARES,
        residual_tolerance=1e-12,
        relative_residual_tolerance=1e-12,
    ).solve(
        equilibrium_matrix=np.array([[1.0], [1.0]]),
        load=[1.0, 3.0],
        capacities=[10.0],
    )
    assert result.efforts[0] == pytest.approx(2.0)
    assert not result.is_exact_equilibrium


def test_pseudoinverse_solves_singular_consistent_system() -> None:
    matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    result = FastLayer(method=SolveMethod.PSEUDOINVERSE).solve(
        equilibrium_matrix=matrix,
        load=[4.0, 8.0],
        capacities=[10.0, 10.0],
    )
    assert np.allclose(result.efforts, [2.0, 2.0])


def test_direct_rejects_rectangular_matrix() -> None:
    with pytest.raises(FastLayerError, match="square"):
        FastLayer(method=SolveMethod.DIRECT).solve(
            equilibrium_matrix=np.ones((3, 2)),
            load=[1.0, 1.0, 2.0],
            capacities=[10.0, 10.0],
        )


def test_direct_rejects_singular_matrix() -> None:
    with pytest.raises(FastLayerError, match="singular|unstable"):
        FastLayer(method=SolveMethod.DIRECT).solve(
            equilibrium_matrix=np.array([[1.0, 1.0], [2.0, 2.0]]),
            load=[4.0, 8.0],
            capacities=[10.0, 10.0],
        )


def test_result_reports_full_rank() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(3),
        load=[1.0, 2.0, 3.0],
        capacities=[10.0, 10.0, 10.0],
    )
    assert result.rank == 3


def test_result_reports_reduced_rank() -> None:
    matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    result = FastLayer().solve(
        equilibrium_matrix=matrix,
        load=[3.0, 6.0],
        capacities=[10.0, 10.0],
    )
    assert result.rank == 1


def test_identity_condition_number_is_one() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(3),
        load=[1.0, 2.0, 3.0],
        capacities=[10.0, 10.0, 10.0],
    )
    assert result.condition_number == pytest.approx(1.0)


def test_singular_matrix_condition_number_is_infinite_or_very_large() -> None:
    matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    result = FastLayer().solve(
        equilibrium_matrix=matrix,
        load=[3.0, 6.0],
        capacities=[10.0, 10.0],
    )
    assert np.isinf(result.condition_number) or result.condition_number > 1e12


def test_capacity_tensors_are_evaluated_from_matrix_columns() -> None:
    capacities = [
        make_capacity(100.0, [1.0, 0.0]),
        make_capacity(200.0, [0.0, 1.0]),
    ]
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[50.0, 100.0],
        capacities=capacities,
    )
    assert np.allclose(result.capacities, [100.0, 200.0])
    assert np.allclose(result.utilization, [0.5, 0.5])


def test_explicit_service_directions_are_used_for_capacity() -> None:
    capacities = [
        make_capacity(100.0, [1.0, 0.0]),
        make_capacity(200.0, [1.0, 0.0]),
    ]
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[50.0, 100.0],
        capacities=capacities,
        service_directions=[[1.0, 0.0], [1.0, 0.0]],
    )
    assert np.allclose(result.capacities, [100.0, 200.0])


def test_direction_objects_can_be_service_directions() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0])
    direction = Direction(vector=[10.0, 0.0], magnitude=999.0)
    result = FastLayer().solve(
        equilibrium_matrix=np.array([[1.0], [0.0]]),
        load=[50.0, 0.0],
        capacities=[tensor],
        service_directions=[direction],
    )
    assert result.capacities[0] == pytest.approx(100.0)
    assert result.utilization[0] == pytest.approx(0.5)


def test_capacity_tensor_configuration_rotates_axis() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0])
    rotation_90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = FastLayer().solve(
        equilibrium_matrix=np.array([[0.0], [1.0]]),
        load=[0.0, 50.0],
        capacities=[tensor],
        service_directions=[[0.0, 1.0]],
        configurations=[rotation_90],
    )
    assert result.capacities[0] == pytest.approx(100.0)
    assert result.utilization[0] == pytest.approx(0.5)


def test_orthogonal_tensor_capacity_produces_infinite_utilization() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0])
    result = FastLayer().solve(
        equilibrium_matrix=np.array([[0.0], [1.0]]),
        load=[0.0, 50.0],
        capacities=[tensor],
        service_directions=[[0.0, 1.0]],
    )
    assert result.capacities[0] == pytest.approx(0.0)
    assert np.isinf(result.utilization[0])
    assert result.has_unavailable_loaded_element


def test_result_reports_equation_and_element_counts() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = FastLayer().solve(
        equilibrium_matrix=matrix,
        load=[1.0, 2.0, 3.0],
        capacities=[10.0, 10.0],
    )
    assert result.equation_count == 3
    assert result.element_count == 2


def test_result_index_accessors() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[20.0, -30.0],
        capacities=[100.0, 60.0],
    )
    assert result.effort_of(0) == pytest.approx(20.0)
    assert result.capacity_of(1) == pytest.approx(60.0)
    assert result.utilization_of(1) == pytest.approx(0.5)


@pytest.mark.parametrize("index", [-1, 2, 100])
def test_result_accessors_reject_out_of_range_index(index: int) -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[1.0, 2.0],
        capacities=[10.0, 10.0],
    )
    with pytest.raises(FastLayerError):
        result.effort_of(index)


def test_result_accessors_reject_non_integer_index() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(1),
        load=[1.0],
        capacities=[10.0],
    )
    with pytest.raises(FastLayerError):
        result.effort_of(0.5)  # type: ignore[arg-type]


def test_result_arrays_are_read_only() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[1.0, 2.0],
        capacities=[10.0, 20.0],
    )
    for array in (
        result.equilibrium_matrix,
        result.load_vector,
        result.efforts,
        result.capacities,
        result.utilization,
        result.signed_utilization,
        result.residual_vector,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 999.0


def test_input_arrays_are_copied() -> None:
    matrix = np.eye(2)
    load = np.array([10.0, 20.0])
    capacities = np.array([100.0, 200.0])
    result = FastLayer().solve(matrix, load, capacities)
    matrix[0, 0] = 999.0
    load[0] = 999.0
    capacities[0] = 999.0
    assert np.allclose(result.equilibrium_matrix, np.eye(2))
    assert np.allclose(result.load_vector, [10.0, 20.0])
    assert np.allclose(result.capacities, [100.0, 200.0])


def test_result_as_dict_contains_main_values() -> None:
    result = FastLayer().solve(
        equilibrium_matrix=np.eye(2),
        load=[20.0, 40.0],
        capacities=[100.0, 50.0],
    )
    data = result.as_dict()
    assert data["solve_method"] == "direct"
    assert data["d_fast_index"] == 1


def test_solve_fast_layer_function_matches_class_interface() -> None:
    result = solve_fast_layer(
        equilibrium_matrix=np.array([[2.0, 0.0], [0.0, 4.0]]),
        load=[10.0, 20.0],
        capacities=[10.0, 10.0],
    )
    assert isinstance(result, FastLayerResult)
    assert np.allclose(result.efforts, [5.0, 5.0])


def test_numeric_capacity_convenience_method() -> None:
    result = FastLayer().solve_with_numeric_capacities(
        equilibrium_matrix=np.eye(2),
        load=[10.0, 20.0],
        capacities=[100.0, 100.0],
    )
    assert np.allclose(result.utilization, [0.1, 0.2])


@pytest.mark.parametrize("method", ["invalid", "", "DIRECT", 123])
def test_invalid_solve_method_is_rejected(method: object) -> None:
    with pytest.raises(FastLayerError):
        FastLayer(method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf, np.nan])
def test_invalid_residual_tolerance_is_rejected(value: float) -> None:
    with pytest.raises(FastLayerError):
        FastLayer(residual_tolerance=value)


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf, np.nan])
def test_invalid_relative_residual_tolerance_is_rejected(value: float) -> None:
    with pytest.raises(FastLayerError):
        FastLayer(relative_residual_tolerance=value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_rank_tolerance_is_rejected(value: float) -> None:
    with pytest.raises(FastLayerError):
        FastLayer(rank_tolerance=value)


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf, np.nan])
def test_invalid_capacity_epsilon_is_rejected(value: float) -> None:
    with pytest.raises(FastLayerError):
        FastLayer(capacity_epsilon=value)


def test_matrix_must_be_two_dimensional() -> None:
    with pytest.raises(FastLayerError, match="two-dimensional"):
        FastLayer().solve([1.0, 2.0], [1.0], [10.0, 10.0])


def test_matrix_must_contain_at_least_one_equation() -> None:
    with pytest.raises(FastLayerError, match="equation"):
        FastLayer().solve(np.empty((0, 2)), [], [10.0, 10.0])


def test_matrix_must_contain_at_least_one_element() -> None:
    with pytest.raises(FastLayerError, match="element"):
        FastLayer().solve(np.empty((2, 0)), [1.0, 2.0], [])


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_matrix_rejects_non_finite_values(invalid_value: float) -> None:
    matrix = np.eye(2)
    matrix[0, 0] = invalid_value
    with pytest.raises(FastLayerError, match="non-finite"):
        FastLayer().solve(matrix, [1.0, 2.0], [10.0, 10.0])


def test_load_must_be_one_dimensional() -> None:
    with pytest.raises(FastLayerError, match="one-dimensional"):
        FastLayer().solve(np.eye(2), [[1.0, 2.0]], [10.0, 10.0])


def test_load_dimension_must_equal_equation_count() -> None:
    with pytest.raises(FastLayerError, match="dimension"):
        FastLayer().solve(np.eye(3), [1.0, 2.0], [10.0, 10.0, 10.0])


@pytest.mark.parametrize("load", [[np.nan, 0.0], [np.inf, 0.0], [-np.inf, 0.0]])
def test_load_rejects_non_finite_values(load: list[float]) -> None:
    with pytest.raises(FastLayerError, match="non-finite"):
        FastLayer().solve(np.eye(2), load, [10.0, 10.0])


def test_direction_load_dimension_must_match_matrix() -> None:
    direction = Direction(vector=[1.0, 0.0], magnitude=10.0)
    with pytest.raises(FastLayerError, match="dimension"):
        FastLayer().solve(np.eye(3), direction, [10.0, 10.0, 10.0])


def test_capacity_count_must_equal_element_count() -> None:
    with pytest.raises(FastLayerError, match="Capacity count"):
        FastLayer().solve(np.eye(3), [1.0, 2.0, 3.0], [10.0, 10.0])


def test_numeric_capacities_must_be_non_negative() -> None:
    with pytest.raises(FastLayerError, match="non-negative"):
        FastLayer().solve(np.eye(2), [1.0, 2.0], [10.0, -1.0])


@pytest.mark.parametrize("invalid_capacity", [np.nan, np.inf, -np.inf])
def test_numeric_capacities_must_be_finite(invalid_capacity: float) -> None:
    with pytest.raises(FastLayerError, match="non-finite"):
        FastLayer().solve(np.eye(2), [1.0, 2.0], [10.0, invalid_capacity])


def test_numeric_and_tensor_capacities_cannot_be_mixed() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0])
    with pytest.raises(FastLayerError, match="either only"):
        FastLayer().solve(np.eye(2), [10.0, 20.0], [tensor, 100.0])


def test_numeric_capacities_reject_service_directions() -> None:
    with pytest.raises(FastLayerError, match="service_directions"):
        FastLayer().solve(
            np.eye(2),
            [10.0, 20.0],
            [100.0, 100.0],
            service_directions=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_numeric_capacities_reject_configurations() -> None:
    with pytest.raises(FastLayerError, match="configurations"):
        FastLayer().solve(
            np.eye(2),
            [10.0, 20.0],
            [100.0, 100.0],
            configurations=[np.eye(2), np.eye(2)],
        )


def test_service_direction_count_must_equal_element_count() -> None:
    tensors = [
        make_capacity(100.0, [1.0, 0.0]),
        make_capacity(100.0, [0.0, 1.0]),
    ]
    with pytest.raises(FastLayerError, match="service_directions count"):
        FastLayer().solve(
            np.eye(2),
            [10.0, 20.0],
            tensors,
            service_directions=[[1.0, 0.0]],
        )


def test_service_direction_dimension_must_match_equations() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0, 0.0])
    with pytest.raises(FastLayerError, match="dimension"):
        FastLayer().solve(
            np.eye(2),
            [10.0, 20.0],
            [tensor, tensor],
            service_directions=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )


def test_service_direction_cannot_be_zero() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0])
    with pytest.raises(FastLayerError, match="cannot be zero"):
        FastLayer().solve(
            np.array([[1.0], [0.0]]),
            [10.0, 0.0],
            [tensor],
            service_directions=[[0.0, 0.0]],
        )


def test_service_direction_must_be_one_dimensional() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0])
    with pytest.raises(FastLayerError, match="one-dimensional"):
        FastLayer().solve(
            np.eye(1),
            [10.0],
            [tensor],
            service_directions=[[[1.0, 0.0]]],
        )


def test_service_direction_must_be_finite() -> None:
    tensor = make_capacity(100.0, [1.0, 0.0])
    with pytest.raises(FastLayerError, match="non-finite"):
        FastLayer().solve(
            np.array([[1.0], [0.0]]),
            [10.0, 0.0],
            [tensor],
            service_directions=[[np.nan, 0.0]],
        )


def test_zero_matrix_column_requires_explicit_service_direction() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
    tensors = [
        make_capacity(100.0, [1.0, 0.0]),
        make_capacity(100.0, [0.0, 1.0]),
    ]
    with pytest.raises(FastLayerError, match="Cannot infer service direction"):
        FastLayer().solve(matrix, [10.0, 0.0], tensors)


def test_configuration_count_must_equal_element_count() -> None:
    tensors = [
        make_capacity(100.0, [1.0, 0.0]),
        make_capacity(100.0, [0.0, 1.0]),
    ]
    with pytest.raises(FastLayerError, match="configurations count"):
        FastLayer().solve(
            np.eye(2),
            [10.0, 20.0],
            tensors,
            configurations=[np.eye(2)],
        )
