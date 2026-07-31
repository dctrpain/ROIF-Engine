"""
Tests for roif.capacity_tensor.

The suite fixes the core invariants of the ROIF directed-capacity model:

    κ(d, x) = c · α · β · response(<d, a(x)>) · w_A(d)

For the minimal passive isotropic tension-only case:

    κ(d, x) = c · α · <d, a(x)>_+
"""

from __future__ import annotations

import numpy as np
import pytest

from roif.capacity_tensor import (
    CapacityTensor,
    CapacityTensorError,
    ElementActivity,
    ResponseMode,
    evaluate_capacity_batch,
)


# ---------------------------------------------------------------------------
# Basic article-consistent behavior
# ---------------------------------------------------------------------------


def test_parallel_direction_returns_available_capacity() -> None:
    """Parallel loading must give κ = c · α."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
    )

    result = capacity.kappa([1.0, 0.0, 0.0])

    assert result == pytest.approx(800.0)


def test_direction_is_normalized_automatically() -> None:
    """Direction magnitude must not change κ."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
    )

    unit_result = capacity.kappa([1.0, 0.0, 0.0])
    scaled_result = capacity.kappa([25.0, 0.0, 0.0])

    assert scaled_result == pytest.approx(unit_result)


def test_local_axis_is_normalized_automatically() -> None:
    """Axis magnitude must not change the directional projection."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[10.0, 0.0, 0.0],
    )

    assert capacity.kappa([1.0, 0.0, 0.0]) == pytest.approx(800.0)
    assert np.allclose(capacity.axis, [1.0, 0.0, 0.0])


def test_orthogonal_direction_has_zero_capacity() -> None:
    """A tension-only element cannot serve an orthogonal direction."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
    )

    result = capacity.kappa([0.0, 1.0, 0.0])

    assert result == pytest.approx(0.0)


def test_opposite_direction_has_zero_tension_capacity() -> None:
    """Positive-part projection must reject the opposite direction."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
        response_mode=ResponseMode.TENSION,
    )

    result = capacity.kappa([-1.0, 0.0, 0.0])

    assert result == pytest.approx(0.0)


def test_diagonal_direction_scales_capacity_by_projection() -> None:
    """A 45-degree load must scale κ by cos(45°)."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
    )

    result = capacity.kappa([1.0, 1.0, 0.0])

    expected = 800.0 / np.sqrt(2.0)

    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Response modes
# ---------------------------------------------------------------------------


def test_compression_mode_serves_opposite_direction() -> None:
    """Compression mode must serve loading opposite to the local axis."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
        response_mode=ResponseMode.COMPRESSION,
    )

    result = capacity.kappa([-1.0, 0.0, 0.0])

    assert result == pytest.approx(800.0)


def test_compression_mode_rejects_forward_direction() -> None:
    """Compression-only element must reject the tension direction."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
        response_mode=ResponseMode.COMPRESSION,
    )

    result = capacity.kappa([1.0, 0.0, 0.0])

    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ([1.0, 0.0, 0.0], 800.0),
        ([-1.0, 0.0, 0.0], 800.0),
        ([0.0, 1.0, 0.0], 0.0),
    ],
)
def test_bidirectional_mode_serves_both_axis_signs(
    direction: list[float],
    expected: float,
) -> None:
    """Bidirectional mode must use the absolute axial projection."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
        response_mode=ResponseMode.BIDIRECTIONAL,
    )

    assert capacity.kappa(direction) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Configuration x
# ---------------------------------------------------------------------------


def test_configuration_rotates_service_axis() -> None:
    """A 90-degree rotation must change the direction served by the element."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
    )

    rotation_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    reference_capacity = capacity.kappa([0.0, 1.0, 0.0])
    rotated_capacity = capacity.kappa(
        [0.0, 1.0, 0.0],
        configuration=rotation_z_90,
    )

    assert reference_capacity == pytest.approx(0.0)
    assert rotated_capacity == pytest.approx(800.0)


def test_configured_axis_is_normalized_after_transformation() -> None:
    """
    A slightly scaled configuration matrix must not change axis magnitude.

    The current implementation intentionally allows matrices that are not
    perfectly orthogonal because numerical solvers may introduce small drift.
    """

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
    )

    configuration = np.array(
        [
            [0.0, -2.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    axis = capacity.configured_axis(configuration)

    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert np.allclose(axis, [0.0, 1.0, 0.0])


def test_identity_configuration_preserves_capacity() -> None:
    """Identity x must reproduce the reference configuration."""

    capacity = CapacityTensor(
        base_capacity=500.0,
        availability=0.75,
        local_axis=[1.0, 0.0, 0.0],
    )

    without_configuration = capacity.kappa([1.0, 0.0, 0.0])
    with_identity = capacity.kappa(
        [1.0, 0.0, 0.0],
        configuration=np.eye(3),
    )

    assert with_identity == pytest.approx(without_configuration)


# ---------------------------------------------------------------------------
# Active and passive elements
# ---------------------------------------------------------------------------


def test_passive_element_ignores_activation() -> None:
    """Passive capacity must remain c · α regardless of activation value."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        activation=0.2,
        activity=ElementActivity.PASSIVE,
        local_axis=[1.0, 0.0, 0.0],
    )

    assert capacity.effective_activation == pytest.approx(1.0)
    assert capacity.kappa([1.0, 0.0, 0.0]) == pytest.approx(800.0)


def test_active_element_is_scaled_by_activation() -> None:
    """Active capacity must include β."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        activation=0.25,
        activity=ElementActivity.ACTIVE,
        local_axis=[1.0, 0.0, 0.0],
    )

    result = capacity.kappa([1.0, 0.0, 0.0])

    assert capacity.effective_activation == pytest.approx(0.25)
    assert result == pytest.approx(200.0)


def test_zero_activation_disables_active_element() -> None:
    """An inactive active element must have zero directed capacity."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        activation=0.0,
        activity=ElementActivity.ACTIVE,
        local_axis=[1.0, 0.0, 0.0],
    )

    assert capacity.kappa([1.0, 0.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Anisotropy
# ---------------------------------------------------------------------------


def test_identity_anisotropy_has_unit_weight() -> None:
    """Identity anisotropy must not modify the minimal ROIF equation."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
        anisotropy=np.eye(3),
    )

    evaluation = capacity.evaluate([1.0, 0.0, 0.0])

    assert evaluation.anisotropic_weight == pytest.approx(1.0)
    assert evaluation.effective_capacity == pytest.approx(800.0)


def test_anisotropy_scales_capacity_in_preferred_direction() -> None:
    """A larger anisotropy eigenvalue must increase the directional weight."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
        anisotropy=np.diag([4.0, 1.0, 1.0]),
    )

    evaluation = capacity.evaluate([1.0, 0.0, 0.0])

    assert evaluation.anisotropic_weight == pytest.approx(2.0)
    assert evaluation.effective_capacity == pytest.approx(2000.0)


def test_anisotropy_does_not_restore_orthogonal_projection() -> None:
    """
    Anisotropy may scale a usable direction but cannot replace geometry.

    If <d, a>_+ is zero, κ must remain zero.
    """

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
        anisotropy=np.diag([1.0, 9.0, 1.0]),
    )

    result = capacity.kappa([0.0, 1.0, 0.0])

    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Detailed evaluation result
# ---------------------------------------------------------------------------


def test_evaluate_returns_all_capacity_components() -> None:
    """The diagnostic result must expose every factor used in κ."""

    capacity = CapacityTensor(
        element_id="cable_1",
        base_capacity=1000.0,
        availability=0.8,
        activation=0.5,
        activity=ElementActivity.ACTIVE,
        local_axis=[1.0, 0.0, 0.0],
    )

    evaluation = capacity.evaluate([1.0, 1.0, 0.0])

    expected_projection = 1.0 / np.sqrt(2.0)
    expected_capacity = 1000.0 * 0.8 * 0.5 * expected_projection

    assert evaluation.base_capacity == pytest.approx(1000.0)
    assert evaluation.availability == pytest.approx(0.8)
    assert evaluation.activation == pytest.approx(0.5)
    assert evaluation.projection == pytest.approx(expected_projection)
    assert evaluation.usable_projection == pytest.approx(expected_projection)
    assert evaluation.anisotropic_weight == pytest.approx(1.0)
    assert evaluation.effective_capacity == pytest.approx(expected_capacity)
    assert evaluation.utilization_denominator == pytest.approx(
        expected_capacity
    )
    assert np.linalg.norm(evaluation.direction) == pytest.approx(1.0)
    assert np.linalg.norm(evaluation.service_axis) == pytest.approx(1.0)


def test_evaluation_arrays_do_not_mutate_tensor_state() -> None:
    """Arrays returned by evaluate must be independent defensive copies."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
    )

    evaluation = capacity.evaluate([1.0, 0.0, 0.0])

    evaluation.service_axis[0] = 0.0
    evaluation.direction[0] = 0.0

    assert np.allclose(capacity.axis, [1.0, 0.0, 0.0])
    assert capacity.kappa([1.0, 0.0, 0.0]) == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# State updates and counterfactual copies
# ---------------------------------------------------------------------------


def test_set_base_capacity_updates_kappa() -> None:
    """Slow-layer changes of c must immediately affect κ."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
    )

    capacity.set_base_capacity(500.0)

    assert capacity.kappa([1.0, 0.0, 0.0]) == pytest.approx(400.0)


def test_set_availability_updates_kappa() -> None:
    """Availability updates must immediately affect κ."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=0.8,
        local_axis=[1.0, 0.0, 0.0],
    )

    capacity.set_availability(0.4)

    assert capacity.kappa([1.0, 0.0, 0.0]) == pytest.approx(400.0)


def test_set_activation_updates_active_capacity() -> None:
    """Activation updates must affect active elements."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        activation=0.2,
        activity=ElementActivity.ACTIVE,
        local_axis=[1.0, 0.0, 0.0],
    )

    capacity.set_activation(0.75)

    assert capacity.kappa([1.0, 0.0, 0.0]) == pytest.approx(750.0)


def test_copy_with_creates_independent_counterfactual() -> None:
    """Counterfactual perturbation must not mutate the original tensor."""

    original = CapacityTensor(
        element_id="element_1",
        base_capacity=1000.0,
        availability=0.8,
        activation=0.5,
        activity=ElementActivity.ACTIVE,
        local_axis=[1.0, 0.0, 0.0],
        anisotropy=np.diag([4.0, 1.0, 1.0]),
    )

    counterfactual = original.copy_with(
        base_capacity=1500.0,
        availability=1.0,
    )

    assert original.base_capacity == pytest.approx(1000.0)
    assert original.availability == pytest.approx(0.8)

    assert counterfactual.base_capacity == pytest.approx(1500.0)
    assert counterfactual.availability == pytest.approx(1.0)
    assert counterfactual.element_id == original.element_id
    assert counterfactual.response_mode is original.response_mode
    assert counterfactual.activity is original.activity
    assert np.allclose(
        counterfactual.anisotropy_matrix,
        original.anisotropy_matrix,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def test_evaluate_capacity_batch_returns_one_value_per_element() -> None:
    """Batch evaluation must preserve element order."""

    capacities = [
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
        ),
        CapacityTensor(
            base_capacity=500.0,
            availability=0.5,
            local_axis=[1.0, 0.0, 0.0],
        ),
        CapacityTensor(
            base_capacity=250.0,
            availability=1.0,
            local_axis=[0.0, 1.0, 0.0],
        ),
    ]

    result = evaluate_capacity_batch(
        capacities,
        direction=[1.0, 0.0, 0.0],
    )

    assert result.dtype == np.float64
    assert result.shape == (3,)
    assert np.allclose(result, [1000.0, 250.0, 0.0])


def test_batch_supports_individual_configurations() -> None:
    """Each element may receive its own configuration x."""

    capacities = [
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
        ),
        CapacityTensor(
            base_capacity=500.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
        ),
    ]

    rotation_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = evaluate_capacity_batch(
        capacities,
        direction=[0.0, 1.0, 0.0],
        configurations=[
            None,
            rotation_z_90,
        ],
    )

    assert np.allclose(result, [0.0, 500.0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_capacity",
    [-1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_base_capacity_is_rejected(
    base_capacity: float,
) -> None:
    """Structural capacity must be finite and non-negative."""

    with pytest.raises(CapacityTensorError):
        CapacityTensor(
            base_capacity=base_capacity,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
        )


@pytest.mark.parametrize(
    "availability",
    [-0.1, 1.1, np.inf, np.nan],
)
def test_invalid_availability_is_rejected(
    availability: float,
) -> None:
    """Availability must remain inside [0, 1]."""

    with pytest.raises(CapacityTensorError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=availability,
            local_axis=[1.0, 0.0, 0.0],
        )


@pytest.mark.parametrize(
    "activation",
    [-0.1, 1.1, np.inf, np.nan],
)
def test_invalid_activation_is_rejected(
    activation: float,
) -> None:
    """Activation must remain inside [0, 1]."""

    with pytest.raises(CapacityTensorError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            activation=activation,
            local_axis=[1.0, 0.0, 0.0],
        )


def test_zero_local_axis_is_rejected() -> None:
    """The service axis must have a defined direction."""

    with pytest.raises(CapacityTensorError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[0.0, 0.0, 0.0],
        )


def test_zero_direction_is_rejected() -> None:
    """κ cannot be evaluated for an undefined load direction."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
    )

    with pytest.raises(CapacityTensorError):
        capacity.kappa([0.0, 0.0, 0.0])


def test_direction_dimension_mismatch_is_rejected() -> None:
    """Direction and tensor dimensions must agree."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
    )

    with pytest.raises(CapacityTensorError):
        capacity.kappa([1.0, 0.0])


def test_invalid_configuration_shape_is_rejected() -> None:
    """Configuration must be an n × n matrix."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
    )

    with pytest.raises(CapacityTensorError):
        capacity.kappa(
            [1.0, 0.0, 0.0],
            configuration=np.eye(2),
        )


def test_configuration_collapsing_axis_is_rejected() -> None:
    """A configuration cannot transform the service axis into zero."""

    capacity = CapacityTensor(
        base_capacity=1000.0,
        availability=1.0,
        local_axis=[1.0, 0.0, 0.0],
    )

    with pytest.raises(CapacityTensorError):
        capacity.kappa(
            [1.0, 0.0, 0.0],
            configuration=np.zeros((3, 3)),
        )


def test_non_symmetric_anisotropy_is_rejected() -> None:
    """Anisotropy matrix must be symmetric."""

    anisotropy = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(CapacityTensorError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
            anisotropy=anisotropy,
        )


def test_negative_anisotropy_eigenvalue_is_rejected() -> None:
    """Anisotropy matrix must be positive semidefinite."""

    anisotropy = np.diag([1.0, -1.0, 1.0])

    with pytest.raises(CapacityTensorError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
            anisotropy=anisotropy,
        )


def test_anisotropy_shape_mismatch_is_rejected() -> None:
    """Anisotropy dimension must match the service-axis dimension."""

    with pytest.raises(CapacityTensorError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
            anisotropy=np.eye(2),
        )


def test_batch_configuration_count_mismatch_is_rejected() -> None:
    """Batch mode must receive exactly one configuration per tensor."""

    capacities = [
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
        ),
        CapacityTensor(
            base_capacity=500.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
        ),
    ]

    with pytest.raises(CapacityTensorError):
        evaluate_capacity_batch(
            capacities,
            direction=[1.0, 0.0, 0.0],
            configurations=[None],
        )


def test_invalid_string_enum_is_rejected() -> None:
    """Unknown response and activity modes must fail during construction."""

    with pytest.raises(ValueError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
            response_mode="unknown",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        CapacityTensor(
            base_capacity=1000.0,
            availability=1.0,
            local_axis=[1.0, 0.0, 0.0],
            activity="unknown",  # type: ignore[arg-type]
        )