"""Tests for the ROIF plane interaction kernel."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from roif.operator import Operator, OperatorError, OperatorStage
from roif.plane_kernel import (
    PlaneKernel,
    PlaneKernelError,
    PlaneKernelResult,
    solve_plane_kernel,
)
from roif.state import PlaneState, PlaneStatus, SystemState


def make_plane(
    plane_id: str,
    *,
    activation: float = 0.0,
    characteristic_time: float = 1.0,
    hysteresis: float = 0.0,
    status: PlaneStatus | str = PlaneStatus.INACTIVE,
    last_update_time: float = 0.0,
) -> PlaneState:
    """Create a valid plane snapshot for kernel tests."""

    return PlaneState(
        plane_id=plane_id,
        activation=activation,
        characteristic_time=characteristic_time,
        hysteresis=hysteresis,
        status=status,
        last_update_time=last_update_time,
    )


def make_state(
    *planes: PlaneState,
    time: float = 0.0,
) -> SystemState:
    """Create a system containing the supplied planes."""

    return SystemState(
        time=time,
        planes={plane.plane_id: plane for plane in planes},
    )


def make_two_plane_state(
    *,
    a: float = 0.0,
    b: float = 0.0,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    hysteresis_a: float = 0.0,
    hysteresis_b: float = 0.0,
    status_a: PlaneStatus | str = PlaneStatus.INACTIVE,
    status_b: PlaneStatus | str = PlaneStatus.INACTIVE,
    time: float = 0.0,
) -> SystemState:
    return make_state(
        make_plane(
            "a",
            activation=a,
            characteristic_time=tau_a,
            hysteresis=hysteresis_a,
            status=status_a,
            last_update_time=time,
        ),
        make_plane(
            "b",
            activation=b,
            characteristic_time=tau_b,
            hysteresis=hysteresis_b,
            status=status_b,
            last_update_time=time,
        ),
        time=time,
    )


def make_kernel(
    *,
    plane_ids: tuple[str, ...] = ("a", "b"),
    coupling_matrix: Any = None,
    dt: float = 1.0,
    external_drive: Any = None,
    characteristic_times: Any = None,
    minimum_activation: float = 0.0,
    maximum_activation: float = 1.0,
    use_plane_hysteresis: bool = True,
    preserve_disabled_planes: bool = True,
    kernel_metadata: dict[str, Any] | None = None,
) -> PlaneKernel:
    size = len(plane_ids)

    if coupling_matrix is None:
        coupling_matrix = np.zeros((size, size))

    return PlaneKernel(
        plane_ids=plane_ids,
        coupling_matrix=coupling_matrix,
        dt=dt,
        external_drive=external_drive,
        characteristic_times=characteristic_times,
        minimum_activation=minimum_activation,
        maximum_activation=maximum_activation,
        use_plane_hysteresis=use_plane_hysteresis,
        preserve_disabled_planes=preserve_disabled_planes,
        kernel_metadata={} if kernel_metadata is None else kernel_metadata,
    )


# ---------------------------------------------------------------------------
# Construction and architectural integration
# ---------------------------------------------------------------------------


def test_plane_kernel_is_operator() -> None:
    assert isinstance(make_kernel(), Operator)


def test_plane_kernel_default_descriptors() -> None:
    kernel = make_kernel()

    assert kernel.operator_id == "plane_kernel"
    assert kernel.name == "Plane interaction kernel"
    assert kernel.stage is OperatorStage.P
    assert kernel.enabled is True


def test_plane_kernel_plane_count() -> None:
    kernel = make_kernel(plane_ids=("a", "b", "c"))

    assert kernel.plane_count == 3


def test_plane_ids_are_trimmed() -> None:
    kernel = make_kernel(plane_ids=("  a  ", " b "))

    assert kernel.plane_ids == ("a", "b")


def test_default_external_drive_is_zero_vector() -> None:
    kernel = make_kernel()

    np.testing.assert_allclose(kernel.external_drive, [0.0, 0.0])


def test_default_characteristic_times_are_read_from_state() -> None:
    kernel = make_kernel()

    assert kernel.characteristic_times is None


def test_explicit_characteristic_times_are_stored() -> None:
    kernel = make_kernel(characteristic_times=[2.0, 3.0])

    np.testing.assert_allclose(kernel.characteristic_times, [2.0, 3.0])


def test_kernel_arrays_are_read_only() -> None:
    kernel = make_kernel(
        coupling_matrix=[[0.0, 1.0], [2.0, 0.0]],
        external_drive=[0.1, 0.2],
        characteristic_times=[1.0, 2.0],
    )

    for array in (
        kernel.coupling_matrix,
        kernel.external_drive,
        kernel.characteristic_times,
    ):
        assert array is not None
        assert array.flags.writeable is False


def test_kernel_input_arrays_are_copied() -> None:
    matrix = np.zeros((2, 2))
    drive = np.zeros(2)
    times = np.ones(2)

    kernel = make_kernel(
        coupling_matrix=matrix,
        external_drive=drive,
        characteristic_times=times,
    )

    matrix[0, 0] = 10.0
    drive[0] = 10.0
    times[0] = 10.0

    assert kernel.coupling_matrix[0, 0] == pytest.approx(0.0)
    assert kernel.external_drive[0] == pytest.approx(0.0)
    assert kernel.characteristic_times is not None
    assert kernel.characteristic_times[0] == pytest.approx(1.0)


def test_kernel_is_frozen() -> None:
    kernel = make_kernel()

    with pytest.raises(FrozenInstanceError):
        kernel.dt = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Numerical evaluation
# ---------------------------------------------------------------------------


def test_zero_state_zero_drive_remains_zero() -> None:
    state = make_two_plane_state()
    result = make_kernel().evaluate(state)

    np.testing.assert_allclose(result.activation_before, [0.0, 0.0])
    np.testing.assert_allclose(result.interaction, [0.0, 0.0])
    np.testing.assert_allclose(result.rate, [0.0, 0.0])
    np.testing.assert_allclose(result.activation_after, [0.0, 0.0])


def test_external_drive_activates_plane() -> None:
    state = make_two_plane_state()
    kernel = make_kernel(
        external_drive=[0.5, 0.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    np.testing.assert_allclose(result.activation_after, [0.5, 0.0])


def test_existing_activation_decays_without_drive() -> None:
    state = make_two_plane_state(a=1.0)
    kernel = make_kernel(dt=0.25)

    result = kernel.evaluate(state)

    np.testing.assert_allclose(result.activation_after, [0.75, 0.0])


def test_characteristic_time_slows_response() -> None:
    state = make_two_plane_state(a=0.0, tau_a=2.0)
    kernel = make_kernel(
        external_drive=[1.0, 0.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.5)


def test_explicit_characteristic_time_overrides_plane_value() -> None:
    state = make_two_plane_state(a=0.0, tau_a=100.0)
    kernel = make_kernel(
        external_drive=[1.0, 0.0],
        characteristic_times=[2.0, 1.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.5)


def test_coupling_uses_source_column_target_row_convention() -> None:
    state = make_two_plane_state(a=1.0, b=0.0)
    kernel = make_kernel(
        coupling_matrix=[
            [0.0, 0.0],
            [0.5, 0.0],
        ],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.interaction[0] == pytest.approx(0.0)
    assert result.interaction[1] == pytest.approx(0.5)
    assert result.activation_after[1] == pytest.approx(0.5)


def test_reverse_coupling_is_not_implicitly_added() -> None:
    state = make_two_plane_state(a=1.0, b=0.0)
    kernel = make_kernel(
        coupling_matrix=[
            [0.0, 0.5],
            [0.0, 0.0],
        ],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.interaction[1] == pytest.approx(0.0)
    assert result.activation_after[1] == pytest.approx(0.0)


def test_self_coupling_can_preserve_activation() -> None:
    state = make_two_plane_state(a=0.6)
    kernel = make_kernel(
        coupling_matrix=[
            [1.0, 0.0],
            [0.0, 0.0],
        ],
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.6)


def test_negative_coupling_inhibits_target_plane() -> None:
    state = make_two_plane_state(a=1.0, b=0.8)
    kernel = make_kernel(
        coupling_matrix=[
            [0.0, 0.0],
            [-0.5, 0.0],
        ],
        dt=0.5,
    )

    result = kernel.evaluate(state)

    assert result.interaction[1] == pytest.approx(-0.5)
    assert result.activation_after[1] == pytest.approx(0.15)


def test_euler_step_uses_dt() -> None:
    state = make_two_plane_state()
    kernel = make_kernel(
        external_drive=[1.0, 0.0],
        dt=0.2,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.2)


def test_result_time_advances_by_dt() -> None:
    state = make_two_plane_state(time=3.0)
    result = make_kernel(dt=0.25).evaluate(state)

    assert result.time == pytest.approx(3.25)
    assert result.dt == pytest.approx(0.25)


def test_activation_is_clipped_at_upper_bound() -> None:
    state = make_two_plane_state()
    kernel = make_kernel(
        external_drive=[10.0, 0.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(1.0)
    assert bool(result.clipped[0]) is True


def test_activation_is_clipped_at_lower_bound() -> None:
    state = make_two_plane_state(a=0.5)
    kernel = make_kernel(
        external_drive=[-10.0, 0.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.0)
    assert bool(result.clipped[0]) is True


def test_custom_activation_bounds_are_used() -> None:
    state = make_two_plane_state(a=0.5)
    kernel = make_kernel(
        external_drive=[10.0, 0.0],
        minimum_activation=0.2,
        maximum_activation=0.8,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.8)


def test_unclipped_plane_is_reported_as_not_clipped() -> None:
    state = make_two_plane_state()
    result = make_kernel(
        external_drive=[0.5, 0.0],
    ).evaluate(state)

    assert bool(result.clipped[0]) is False


# ---------------------------------------------------------------------------
# Hysteresis and disabled planes
# ---------------------------------------------------------------------------


def test_hysteresis_retains_small_change() -> None:
    state = make_two_plane_state(
        a=0.4,
        hysteresis_a=0.15,
    )
    kernel = make_kernel(
        external_drive=[0.5, 0.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.4)
    assert bool(result.retained_by_hysteresis[0]) is True


def test_hysteresis_allows_change_above_threshold() -> None:
    state = make_two_plane_state(
        a=0.4,
        hysteresis_a=0.05,
    )
    kernel = make_kernel(
        external_drive=[0.5, 0.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.5)
    assert bool(result.retained_by_hysteresis[0]) is False


def test_hysteresis_is_inclusive_at_threshold() -> None:
    state = make_two_plane_state(
        a=0.4,
        hysteresis_a=0.1,
    )
    kernel = make_kernel(
        external_drive=[0.5, 0.0],
        dt=1.0,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.4)
    assert bool(result.retained_by_hysteresis[0]) is True


def test_hysteresis_can_be_disabled() -> None:
    state = make_two_plane_state(
        a=0.4,
        hysteresis_a=1.0,
    )
    kernel = make_kernel(
        external_drive=[0.5, 0.0],
        use_plane_hysteresis=False,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.5)
    assert bool(result.retained_by_hysteresis[0]) is False


def test_disabled_plane_is_preserved_by_default() -> None:
    state = make_two_plane_state(
        a=0.2,
        status_a=PlaneStatus.DISABLED,
    )
    kernel = make_kernel(
        external_drive=[1.0, 0.0],
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(0.2)
    assert bool(result.retained_by_hysteresis[0]) is True


def test_disabled_plane_can_be_evolved_when_preservation_is_off() -> None:
    state = make_two_plane_state(
        a=0.2,
        status_a=PlaneStatus.DISABLED,
    )
    kernel = make_kernel(
        external_drive=[1.0, 0.0],
        preserve_disabled_planes=False,
    )

    result = kernel.evaluate(state)

    assert result.activation_after[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Applying the operator to SystemState
# ---------------------------------------------------------------------------


def test_apply_returns_new_system_state() -> None:
    state = make_two_plane_state()
    kernel = make_kernel(external_drive=[0.5, 0.0])

    updated = kernel.apply(state)

    assert isinstance(updated, SystemState)
    assert updated is not state


def test_apply_advances_system_time() -> None:
    state = make_two_plane_state(time=2.0)
    updated = make_kernel(dt=0.5).apply(state)

    assert updated.time == pytest.approx(2.5)


def test_apply_updates_plane_activation() -> None:
    state = make_two_plane_state()
    updated = make_kernel(
        external_drive=[0.5, 0.25],
    ).apply(state)

    assert updated.plane("a").activation == pytest.approx(0.5)
    assert updated.plane("b").activation == pytest.approx(0.25)


def test_apply_updates_plane_last_update_time() -> None:
    state = make_two_plane_state(time=2.0)
    updated = make_kernel(dt=0.5).apply(state)

    assert updated.plane("a").last_update_time == pytest.approx(2.5)
    assert updated.plane("b").last_update_time == pytest.approx(2.5)


def test_apply_does_not_mutate_original_state() -> None:
    state = make_two_plane_state(a=0.1)
    updated = make_kernel(
        external_drive=[1.0, 0.0],
    ).apply(state)

    assert state.time == pytest.approx(0.0)
    assert state.plane("a").activation == pytest.approx(0.1)
    assert updated.plane("a").activation == pytest.approx(1.0)


def test_disabled_kernel_returns_original_state() -> None:
    state = make_two_plane_state()
    kernel = PlaneKernel(
        enabled=False,
        plane_ids=("a", "b"),
        coupling_matrix=np.zeros((2, 2)),
        external_drive=[1.0, 1.0],
    )

    assert kernel.apply(state) is state


def test_kernel_can_be_used_in_operator_pipeline() -> None:
    first = make_kernel(
        external_drive=[0.5, 0.0],
        dt=1.0,
    )
    second = make_kernel(
        external_drive=[1.0, 0.0],
        dt=1.0,
    )
    state = make_two_plane_state()

    updated = (first >> second)(state)

    assert updated.time == pytest.approx(2.0)
    assert updated.plane("a").activation == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PlaneKernelResult
# ---------------------------------------------------------------------------


def test_result_plane_count() -> None:
    result = make_kernel().evaluate(make_two_plane_state())

    assert result.plane_count == 2


def test_result_changed_false_for_unchanged_activation() -> None:
    result = make_kernel().evaluate(make_two_plane_state())

    assert result.changed is False
    assert result.changed_plane_ids == ()


def test_result_changed_plane_ids() -> None:
    result = make_kernel(
        external_drive=[0.5, 0.0],
    ).evaluate(make_two_plane_state())

    assert result.changed is True
    assert result.changed_plane_ids == ("a",)


def test_result_activation_for_known_plane() -> None:
    result = make_kernel(
        external_drive=[0.25, 0.75],
    ).evaluate(make_two_plane_state())

    assert result.activation_for("a") == pytest.approx(0.25)
    assert result.activation_for("b") == pytest.approx(0.75)


def test_result_activation_for_unknown_plane_raises() -> None:
    result = make_kernel().evaluate(make_two_plane_state())

    with pytest.raises(PlaneKernelError, match="Unknown plane_id"):
        result.activation_for("missing")


def test_result_arrays_are_read_only() -> None:
    result = make_kernel().evaluate(make_two_plane_state())

    for array in (
        result.activation_before,
        result.interaction,
        result.rate,
        result.activation_after,
        result.clipped,
        result.retained_by_hysteresis,
    ):
        assert array.flags.writeable is False


def test_result_input_arrays_are_copied() -> None:
    before = np.array([0.0, 0.0])
    interaction = np.array([0.0, 0.0])
    rate = np.array([0.0, 0.0])
    after = np.array([0.0, 0.0])
    clipped = np.array([False, False])
    retained = np.array([False, False])

    result = PlaneKernelResult(
        plane_ids=("a", "b"),
        time=1.0,
        dt=1.0,
        activation_before=before,
        interaction=interaction,
        rate=rate,
        activation_after=after,
        clipped=clipped,
        retained_by_hysteresis=retained,
    )

    before[0] = 1.0
    after[0] = 1.0
    clipped[0] = True

    assert result.activation_before[0] == pytest.approx(0.0)
    assert result.activation_after[0] == pytest.approx(0.0)
    assert bool(result.clipped[0]) is False


def test_result_is_frozen() -> None:
    result = make_kernel().evaluate(make_two_plane_state())

    with pytest.raises(FrozenInstanceError):
        result.dt = 2.0  # type: ignore[misc]


def test_result_as_dict() -> None:
    result = make_kernel(
        external_drive=[0.5, 0.0],
    ).evaluate(make_two_plane_state())

    data = result.as_dict()

    assert data["plane_ids"] == ["a", "b"]
    assert data["time"] == pytest.approx(1.0)
    assert data["dt"] == pytest.approx(1.0)
    assert data["activation_before"] == [0.0, 0.0]
    assert data["activation_after"] == [0.5, 0.0]
    assert data["changed"] is True
    assert data["changed_plane_ids"] == ["a"]


# ---------------------------------------------------------------------------
# Counterfactual helpers and serialization
# ---------------------------------------------------------------------------


def test_with_external_drive_returns_independent_kernel() -> None:
    original = make_kernel()
    changed = original.with_external_drive([0.5, 0.25])

    np.testing.assert_allclose(original.external_drive, [0.0, 0.0])
    np.testing.assert_allclose(changed.external_drive, [0.5, 0.25])
    assert changed is not original


def test_with_coupling_matrix_returns_independent_kernel() -> None:
    original = make_kernel()
    changed = original.with_coupling_matrix(
        [[0.0, 1.0], [0.0, 0.0]]
    )

    np.testing.assert_allclose(original.coupling_matrix, np.zeros((2, 2)))
    np.testing.assert_allclose(
        changed.coupling_matrix,
        [[0.0, 1.0], [0.0, 0.0]],
    )


def test_kernel_metadata_is_trimmed_and_read_only() -> None:
    kernel = make_kernel(
        kernel_metadata={"  source  ": "test"},
    )

    assert dict(kernel.kernel_metadata) == {"source": "test"}

    with pytest.raises(TypeError):
        kernel.kernel_metadata["source"] = "changed"  # type: ignore[index]


def test_kernel_metadata_input_is_copied() -> None:
    metadata = {"source": "original"}
    kernel = make_kernel(kernel_metadata=metadata)

    metadata["source"] = "changed"

    assert kernel.kernel_metadata["source"] == "original"


def test_kernel_as_dict() -> None:
    kernel = make_kernel(
        coupling_matrix=[[0.0, 0.5], [0.25, 0.0]],
        external_drive=[0.1, 0.2],
        characteristic_times=[2.0, 3.0],
        dt=0.5,
        kernel_metadata={"source": "test"},
    )

    data = kernel.as_dict()

    assert data["type"] == "PlaneKernel"
    assert data["stage"] == "P"
    assert data["plane_ids"] == ["a", "b"]
    assert data["plane_count"] == 2
    assert data["coupling_matrix"] == [[0.0, 0.5], [0.25, 0.0]]
    assert data["external_drive"] == [0.1, 0.2]
    assert data["characteristic_times"] == [2.0, 3.0]
    assert data["dt"] == pytest.approx(0.5)
    assert data["kernel_metadata"] == {"source": "test"}


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_solve_plane_kernel_returns_state_and_result() -> None:
    state = make_two_plane_state()

    updated, result = solve_plane_kernel(
        state,
        plane_ids=("a", "b"),
        coupling_matrix=np.zeros((2, 2)),
        external_drive=[0.5, 0.0],
    )

    assert isinstance(updated, SystemState)
    assert isinstance(result, PlaneKernelResult)
    assert updated.time == pytest.approx(1.0)
    assert updated.plane("a").activation == pytest.approx(0.5)
    assert result.activation_after[0] == pytest.approx(0.5)


def test_solve_plane_kernel_does_not_mutate_input() -> None:
    state = make_two_plane_state()

    updated, _ = solve_plane_kernel(
        state,
        plane_ids=("a", "b"),
        coupling_matrix=np.zeros((2, 2)),
        external_drive=[1.0, 0.0],
    )

    assert state.time == pytest.approx(0.0)
    assert state.plane("a").activation == pytest.approx(0.0)
    assert updated.plane("a").activation == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Validation: plane IDs and state compatibility
# ---------------------------------------------------------------------------


def test_plane_ids_cannot_be_string() -> None:
    with pytest.raises(
        PlaneKernelError,
        match="plane_ids must be a sequence",
    ):
        make_kernel(plane_ids="ab")  # type: ignore[arg-type]


def test_plane_ids_cannot_be_empty() -> None:
    with pytest.raises(PlaneKernelError, match="plane_ids cannot be empty"):
        make_kernel(plane_ids=())


def test_plane_ids_must_be_unique() -> None:
    with pytest.raises(PlaneKernelError, match="plane_ids must be unique"):
        make_kernel(plane_ids=("a", "a"))


@pytest.mark.parametrize("plane_id", ["", " ", "\t", "\n"])
def test_plane_id_cannot_be_empty(plane_id: str) -> None:
    with pytest.raises(PlaneKernelError, match="cannot be empty"):
        make_kernel(plane_ids=("a", plane_id))


def test_plane_id_must_be_string() -> None:
    with pytest.raises(PlaneKernelError, match="must be a string"):
        make_kernel(plane_ids=("a", 1))  # type: ignore[arg-type]


def test_evaluate_rejects_missing_plane() -> None:
    state = make_state(make_plane("a"))

    with pytest.raises(
        PlaneKernelError,
        match="does not contain plane 'b'",
    ):
        make_kernel().evaluate(state)


def test_evaluate_rejects_non_system_state() -> None:
    with pytest.raises(OperatorError, match="requires a SystemState"):
        make_kernel().evaluate(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Validation: matrix and vectors
# ---------------------------------------------------------------------------


def test_coupling_matrix_must_be_two_dimensional() -> None:
    with pytest.raises(PlaneKernelError, match="must be 2-dimensional"):
        make_kernel(coupling_matrix=[0.0, 0.0])


def test_coupling_matrix_shape_must_match_plane_count() -> None:
    with pytest.raises(PlaneKernelError, match="must have shape"):
        make_kernel(coupling_matrix=np.zeros((3, 3)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_coupling_matrix_must_be_finite(value: float) -> None:
    matrix = np.zeros((2, 2))
    matrix[0, 0] = value

    with pytest.raises(PlaneKernelError, match="finite values"):
        make_kernel(coupling_matrix=matrix)


def test_external_drive_must_be_one_dimensional() -> None:
    with pytest.raises(PlaneKernelError, match="must be 1-dimensional"):
        make_kernel(external_drive=[[0.0, 0.0]])


def test_external_drive_shape_must_match_plane_count() -> None:
    with pytest.raises(PlaneKernelError, match="must have shape"):
        make_kernel(external_drive=[0.0, 0.0, 0.0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_external_drive_must_be_finite(value: float) -> None:
    with pytest.raises(PlaneKernelError, match="finite values"):
        make_kernel(external_drive=[value, 0.0])


def test_characteristic_times_must_be_one_dimensional() -> None:
    with pytest.raises(PlaneKernelError, match="must be 1-dimensional"):
        make_kernel(characteristic_times=[[1.0, 1.0]])


def test_characteristic_times_shape_must_match_plane_count() -> None:
    with pytest.raises(PlaneKernelError, match="must have shape"):
        make_kernel(characteristic_times=[1.0])


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_characteristic_times_must_be_positive(value: float) -> None:
    with pytest.raises(PlaneKernelError, match="strictly positive"):
        make_kernel(characteristic_times=[value, 1.0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_characteristic_times_must_be_finite(value: float) -> None:
    with pytest.raises(PlaneKernelError, match="finite values"):
        make_kernel(characteristic_times=[value, 1.0])


# ---------------------------------------------------------------------------
# Validation: scalar settings and metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_dt_must_be_finite_and_positive(dt: float) -> None:
    with pytest.raises(PlaneKernelError, match="strictly positive"):
        make_kernel(dt=dt)


def test_minimum_activation_cannot_be_negative() -> None:
    with pytest.raises(PlaneKernelError, match="cannot be negative"):
        make_kernel(minimum_activation=-0.1)


def test_maximum_activation_cannot_exceed_one() -> None:
    with pytest.raises(PlaneKernelError, match="cannot exceed 1"):
        make_kernel(maximum_activation=1.1)


def test_minimum_activation_cannot_exceed_maximum() -> None:
    with pytest.raises(PlaneKernelError, match="cannot exceed"):
        make_kernel(
            minimum_activation=0.8,
            maximum_activation=0.2,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_minimum_activation_must_be_finite(value: float) -> None:
    with pytest.raises(PlaneKernelError, match="minimum_activation must be finite"):
        make_kernel(minimum_activation=value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_maximum_activation_must_be_finite(value: float) -> None:
    with pytest.raises(PlaneKernelError, match="maximum_activation must be finite"):
        make_kernel(maximum_activation=value)


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_use_plane_hysteresis_must_be_bool(value: object) -> None:
    with pytest.raises(PlaneKernelError, match="must be a bool"):
        make_kernel(use_plane_hysteresis=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_preserve_disabled_planes_must_be_bool(value: object) -> None:
    with pytest.raises(PlaneKernelError, match="must be a bool"):
        make_kernel(preserve_disabled_planes=value)  # type: ignore[arg-type]


def test_kernel_metadata_must_be_mapping() -> None:
    with pytest.raises(PlaneKernelError, match="must be a mapping"):
        make_kernel(kernel_metadata=["invalid"])  # type: ignore[arg-type]


def test_kernel_metadata_key_must_be_string() -> None:
    with pytest.raises(PlaneKernelError, match="keys must be strings"):
        make_kernel(kernel_metadata={1: "invalid"})  # type: ignore[dict-item]


def test_kernel_metadata_key_cannot_be_empty() -> None:
    with pytest.raises(PlaneKernelError, match="keys cannot be empty"):
        make_kernel(kernel_metadata={" ": "invalid"})


# ---------------------------------------------------------------------------
# PlaneKernelResult validation
# ---------------------------------------------------------------------------


def valid_result_kwargs() -> dict[str, Any]:
    return {
        "plane_ids": ("a", "b"),
        "time": 1.0,
        "dt": 1.0,
        "activation_before": [0.0, 0.0],
        "interaction": [0.0, 0.0],
        "rate": [0.0, 0.0],
        "activation_after": [0.0, 0.0],
        "clipped": [False, False],
        "retained_by_hysteresis": [False, False],
    }


@pytest.mark.parametrize("time", [-1.0, np.nan, np.inf, -np.inf])
def test_result_time_must_be_finite_and_non_negative(time: float) -> None:
    kwargs = valid_result_kwargs()
    kwargs["time"] = time

    with pytest.raises(PlaneKernelError, match="time must be finite"):
        PlaneKernelResult(**kwargs)


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_result_dt_must_be_finite_and_positive(dt: float) -> None:
    kwargs = valid_result_kwargs()
    kwargs["dt"] = dt

    with pytest.raises(PlaneKernelError, match="strictly positive"):
        PlaneKernelResult(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "activation_before",
        "interaction",
        "rate",
        "activation_after",
    ],
)
def test_result_numeric_vector_shape_must_match_planes(
    field_name: str,
) -> None:
    kwargs = valid_result_kwargs()
    kwargs[field_name] = [0.0]

    with pytest.raises(PlaneKernelError, match="must have shape"):
        PlaneKernelResult(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "clipped",
        "retained_by_hysteresis",
    ],
)
def test_result_boolean_vector_shape_must_match_planes(
    field_name: str,
) -> None:
    kwargs = valid_result_kwargs()
    kwargs[field_name] = [False]

    with pytest.raises(PlaneKernelError, match="must have shape"):
        PlaneKernelResult(**kwargs)
