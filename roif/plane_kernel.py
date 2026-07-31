"""
Plane interaction kernel for the ROIF evolution layer.

The plane kernel evolves activation across an ordered set of planes. It is a
numerical operator over ``SystemState`` and therefore fits directly into the
operator pipeline:

    state_next = plane_kernel(state)

The kernel uses a bounded first-order interaction model:

    da/dt = (u + W a - a) / tau

where:

- ``a`` is the current plane-activation vector;
- ``u`` is an external drive vector;
- ``W`` is the directed plane-coupling matrix;
- ``tau`` is the characteristic-time vector.

One explicit Euler step gives:

    a_next = clip(a + dt * (u + W a - a) / tau, 0, 1)

This module intentionally does not create cascade events or persistent
history records. Those responsibilities belong to future modules such as
``cascade_event.py`` and ``cascade_history.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .operator import Operator, OperatorError, OperatorStage
from .state import PlaneState, SystemState


FloatArray = NDArray[np.float64]


class PlaneKernelError(OperatorError):
    """Raised when a plane kernel or plane-kernel evaluation is invalid."""


def _readonly_array(
    value: ArrayLike,
    *,
    ndim: int,
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=float)

    if array.ndim != ndim:
        raise PlaneKernelError(
            f"{name} must be {ndim}-dimensional, received shape {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise PlaneKernelError(f"{name} must contain only finite values.")

    copied = np.array(array, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _normalize_plane_ids(plane_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(plane_ids, (str, bytes)):
        raise PlaneKernelError("plane_ids must be a sequence of strings.")

    normalized: list[str] = []

    for index, plane_id in enumerate(plane_ids):
        if not isinstance(plane_id, str):
            raise PlaneKernelError(
                f"plane_ids[{index}] must be a string."
            )

        trimmed = plane_id.strip()

        if not trimmed:
            raise PlaneKernelError(
                f"plane_ids[{index}] cannot be empty."
            )

        normalized.append(trimmed)

    if not normalized:
        raise PlaneKernelError("plane_ids cannot be empty.")

    if len(set(normalized)) != len(normalized):
        raise PlaneKernelError("plane_ids must be unique.")

    return tuple(normalized)


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise PlaneKernelError("kernel_metadata must be a mapping.")

    copied: dict[str, Any] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise PlaneKernelError(
                "kernel_metadata keys must be strings."
            )

        trimmed = key.strip()

        if not trimmed:
            raise PlaneKernelError(
                "kernel_metadata keys cannot be empty."
            )

        copied[trimmed] = value

    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class PlaneKernelResult:
    """Immutable numerical result of one plane-kernel evaluation."""

    plane_ids: tuple[str, ...]
    time: float
    dt: float
    activation_before: FloatArray
    interaction: FloatArray
    rate: FloatArray
    activation_after: FloatArray
    clipped: NDArray[np.bool_]
    retained_by_hysteresis: NDArray[np.bool_]

    def __post_init__(self) -> None:
        plane_ids = _normalize_plane_ids(self.plane_ids)
        object.__setattr__(self, "plane_ids", plane_ids)

        time = float(self.time)
        dt = float(self.dt)

        if not np.isfinite(time) or time < 0.0:
            raise PlaneKernelError(
                "time must be finite and non-negative."
            )

        if not np.isfinite(dt) or dt <= 0.0:
            raise PlaneKernelError(
                "dt must be finite and strictly positive."
            )

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "dt", dt)

        size = len(plane_ids)

        for field_name in (
            "activation_before",
            "interaction",
            "rate",
            "activation_after",
        ):
            array = _readonly_array(
                getattr(self, field_name),
                ndim=1,
                name=field_name,
            )

            if array.shape != (size,):
                raise PlaneKernelError(
                    f"{field_name} must have shape ({size},), "
                    f"received {array.shape}."
                )

            object.__setattr__(self, field_name, array)

        for field_name in (
            "clipped",
            "retained_by_hysteresis",
        ):
            array = np.asarray(getattr(self, field_name), dtype=bool)

            if array.ndim != 1 or array.shape != (size,):
                raise PlaneKernelError(
                    f"{field_name} must have shape ({size},), "
                    f"received {array.shape}."
                )

            copied = np.array(array, dtype=bool, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, field_name, copied)

    @property
    def plane_count(self) -> int:
        return len(self.plane_ids)

    @property
    def changed(self) -> bool:
        return not np.array_equal(
            self.activation_before,
            self.activation_after,
        )

    @property
    def changed_plane_ids(self) -> tuple[str, ...]:
        changed = ~np.isclose(
            self.activation_before,
            self.activation_after,
            rtol=0.0,
            atol=0.0,
        )
        return tuple(
            plane_id
            for plane_id, is_changed in zip(
                self.plane_ids,
                changed,
                strict=True,
            )
            if bool(is_changed)
        )

    def activation_for(self, plane_id: str) -> float:
        """Return the resulting activation of one plane."""

        try:
            index = self.plane_ids.index(plane_id)
        except ValueError as exc:
            raise PlaneKernelError(
                f"Unknown plane_id {plane_id!r}."
            ) from exc

        return float(self.activation_after[index])

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return {
            "plane_ids": list(self.plane_ids),
            "time": self.time,
            "dt": self.dt,
            "activation_before": self.activation_before.tolist(),
            "interaction": self.interaction.tolist(),
            "rate": self.rate.tolist(),
            "activation_after": self.activation_after.tolist(),
            "clipped": self.clipped.tolist(),
            "retained_by_hysteresis": (
                self.retained_by_hysteresis.tolist()
            ),
            "changed": self.changed,
            "changed_plane_ids": list(self.changed_plane_ids),
        }


@dataclass(frozen=True, slots=True)
class PlaneKernel(Operator):
    """
    Directed interaction operator over ``PlaneState.activation``.

    Matrix convention
    -----------------
    ``coupling_matrix[i, j]`` is the influence of source plane ``j`` on
    target plane ``i``. Therefore the total interaction is:

        interaction = external_drive + coupling_matrix @ activation
    """

    operator_id: str = "plane_kernel"
    name: str | None = "Plane interaction kernel"
    stage: OperatorStage = OperatorStage.P
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    plane_ids: tuple[str, ...] = field(default_factory=tuple)
    coupling_matrix: FloatArray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=float)
    )
    dt: float = 1.0
    external_drive: FloatArray | None = None
    characteristic_times: FloatArray | None = None
    minimum_activation: float = 0.0
    maximum_activation: float = 1.0
    use_plane_hysteresis: bool = True
    preserve_disabled_planes: bool = True
    kernel_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super(PlaneKernel, self).__post_init__()

        plane_ids = _normalize_plane_ids(self.plane_ids)
        object.__setattr__(self, "plane_ids", plane_ids)

        size = len(plane_ids)

        matrix = _readonly_array(
            self.coupling_matrix,
            ndim=2,
            name="coupling_matrix",
        )

        if matrix.shape != (size, size):
            raise PlaneKernelError(
                "coupling_matrix must have shape "
                f"({size}, {size}), received {matrix.shape}."
            )

        object.__setattr__(self, "coupling_matrix", matrix)

        dt = float(self.dt)

        if not np.isfinite(dt) or dt <= 0.0:
            raise PlaneKernelError(
                "dt must be finite and strictly positive."
            )

        object.__setattr__(self, "dt", dt)

        minimum = float(self.minimum_activation)
        maximum = float(self.maximum_activation)

        if not np.isfinite(minimum):
            raise PlaneKernelError(
                "minimum_activation must be finite."
            )

        if not np.isfinite(maximum):
            raise PlaneKernelError(
                "maximum_activation must be finite."
            )

        if minimum < 0.0:
            raise PlaneKernelError(
                "minimum_activation cannot be negative."
            )

        if maximum > 1.0:
            raise PlaneKernelError(
                "maximum_activation cannot exceed 1."
            )

        if minimum > maximum:
            raise PlaneKernelError(
                "minimum_activation cannot exceed maximum_activation."
            )

        object.__setattr__(self, "minimum_activation", minimum)
        object.__setattr__(self, "maximum_activation", maximum)

        if not isinstance(self.use_plane_hysteresis, bool):
            raise PlaneKernelError(
                "use_plane_hysteresis must be a bool."
            )

        if not isinstance(self.preserve_disabled_planes, bool):
            raise PlaneKernelError(
                "preserve_disabled_planes must be a bool."
            )

        if self.external_drive is None:
            drive = np.zeros(size, dtype=float)
            drive.setflags(write=False)
        else:
            drive = _readonly_array(
                self.external_drive,
                ndim=1,
                name="external_drive",
            )

            if drive.shape != (size,):
                raise PlaneKernelError(
                    "external_drive must have shape "
                    f"({size},), received {drive.shape}."
                )

        object.__setattr__(self, "external_drive", drive)

        if self.characteristic_times is None:
            times = None
        else:
            times = _readonly_array(
                self.characteristic_times,
                ndim=1,
                name="characteristic_times",
            )

            if times.shape != (size,):
                raise PlaneKernelError(
                    "characteristic_times must have shape "
                    f"({size},), received {times.shape}."
                )

            if np.any(times <= 0.0):
                raise PlaneKernelError(
                    "characteristic_times must be strictly positive."
                )

        object.__setattr__(self, "characteristic_times", times)
        object.__setattr__(
            self,
            "kernel_metadata",
            _freeze_metadata(self.kernel_metadata),
        )

    @property
    def plane_count(self) -> int:
        return len(self.plane_ids)

    def _planes_from_state(
        self,
        state: SystemState,
    ) -> tuple[PlaneState, ...]:
        planes: list[PlaneState] = []

        for plane_id in self.plane_ids:
            try:
                plane = state.plane(plane_id)
            except Exception as exc:
                raise PlaneKernelError(
                    f"SystemState does not contain plane {plane_id!r}."
                ) from exc

            planes.append(plane)

        return tuple(planes)

    def _characteristic_time_vector(
        self,
        planes: tuple[PlaneState, ...],
    ) -> FloatArray:
        if self.characteristic_times is not None:
            return self.characteristic_times

        values = np.asarray(
            [
                float(plane.characteristic_time)
                for plane in planes
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise PlaneKernelError(
                "Plane characteristic times must be finite."
            )

        if np.any(values <= 0.0):
            raise PlaneKernelError(
                "Plane characteristic times must be strictly positive."
            )

        values.setflags(write=False)
        return values

    @staticmethod
    def _is_disabled(plane: PlaneState) -> bool:
        status = getattr(plane, "status", None)
        value = getattr(status, "value", status)
        return value == "disabled"

    def evaluate(self, state: SystemState) -> PlaneKernelResult:
        """Evaluate one numerical step without constructing a new state."""

        self.validate_input(state)
        planes = self._planes_from_state(state)

        activation_before = np.asarray(
            [float(plane.activation) for plane in planes],
            dtype=float,
        )

        if not np.all(np.isfinite(activation_before)):
            raise PlaneKernelError(
                "Plane activations must be finite."
            )

        times = self._characteristic_time_vector(planes)
        interaction = (
            np.asarray(self.external_drive, dtype=float)
            + self.coupling_matrix @ activation_before
        )
        rate = (interaction - activation_before) / times
        raw_after = activation_before + self.dt * rate

        clipped_mask = (
            (raw_after < self.minimum_activation)
            | (raw_after > self.maximum_activation)
        )

        activation_after = np.clip(
            raw_after,
            self.minimum_activation,
            self.maximum_activation,
        )

        retained = np.zeros(self.plane_count, dtype=bool)

        if self.use_plane_hysteresis:
            for index, plane in enumerate(planes):
                hysteresis = float(plane.hysteresis)

                if not np.isfinite(hysteresis):
                    raise PlaneKernelError(
                        f"Plane {plane.plane_id!r} has non-finite hysteresis."
                    )

                if abs(
                    activation_after[index]
                    - activation_before[index]
                ) <= hysteresis:
                    activation_after[index] = activation_before[index]
                    retained[index] = True

        if self.preserve_disabled_planes:
            for index, plane in enumerate(planes):
                if self._is_disabled(plane):
                    activation_after[index] = activation_before[index]
                    retained[index] = True

        target_time = state.time + self.dt

        return PlaneKernelResult(
            plane_ids=self.plane_ids,
            time=target_time,
            dt=self.dt,
            activation_before=activation_before,
            interaction=interaction,
            rate=rate,
            activation_after=activation_after,
            clipped=clipped_mask,
            retained_by_hysteresis=retained,
        )

    def _apply(self, state: SystemState) -> SystemState:
        result = self.evaluate(state)
        updated_state = state.at_time(result.time)

        for plane_id, activation in zip(
            result.plane_ids,
            result.activation_after,
            strict=True,
        ):
            plane = updated_state.plane(plane_id)

            updated_plane = replace(
                plane,
                activation=float(activation),
                last_update_time=result.time,
            )

            updated_state = updated_state.with_plane(updated_plane)

        return updated_state

    def with_external_drive(
        self,
        external_drive: ArrayLike,
    ) -> PlaneKernel:
        """Return an independent kernel with a replaced drive vector."""

        return replace(
            self,
            external_drive=np.asarray(
                external_drive,
                dtype=float,
            ),
        )

    def with_coupling_matrix(
        self,
        coupling_matrix: ArrayLike,
    ) -> PlaneKernel:
        """Return an independent kernel with a replaced coupling matrix."""

        return replace(
            self,
            coupling_matrix=np.asarray(
                coupling_matrix,
                dtype=float,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        data = super(PlaneKernel, self).as_dict()
        data.update(
            {
                "plane_ids": list(self.plane_ids),
                "plane_count": self.plane_count,
                "coupling_matrix": self.coupling_matrix.tolist(),
                "dt": self.dt,
                "external_drive": self.external_drive.tolist(),
                "characteristic_times": (
                    None
                    if self.characteristic_times is None
                    else self.characteristic_times.tolist()
                ),
                "minimum_activation": self.minimum_activation,
                "maximum_activation": self.maximum_activation,
                "use_plane_hysteresis": self.use_plane_hysteresis,
                "preserve_disabled_planes": (
                    self.preserve_disabled_planes
                ),
                "kernel_metadata": dict(self.kernel_metadata),
            }
        )
        return data


def solve_plane_kernel(
    state: SystemState,
    *,
    plane_ids: Sequence[str],
    coupling_matrix: ArrayLike,
    dt: float = 1.0,
    external_drive: ArrayLike | None = None,
    characteristic_times: ArrayLike | None = None,
    minimum_activation: float = 0.0,
    maximum_activation: float = 1.0,
    use_plane_hysteresis: bool = True,
    preserve_disabled_planes: bool = True,
) -> tuple[SystemState, PlaneKernelResult]:
    """
    Convenience interface for one plane-kernel step.

    Returns both the updated immutable state and the numerical result.
    """

    kernel = PlaneKernel(
        plane_ids=tuple(plane_ids),
        coupling_matrix=np.asarray(coupling_matrix, dtype=float),
        dt=dt,
        external_drive=(
            None
            if external_drive is None
            else np.asarray(external_drive, dtype=float)
        ),
        characteristic_times=(
            None
            if characteristic_times is None
            else np.asarray(characteristic_times, dtype=float)
        ),
        minimum_activation=minimum_activation,
        maximum_activation=maximum_activation,
        use_plane_hysteresis=use_plane_hysteresis,
        preserve_disabled_planes=preserve_disabled_planes,
    )

    result = kernel.evaluate(state)
    updated_state = kernel.apply(state)
    return updated_state, result


__all__ = [
    "PlaneKernel",
    "PlaneKernelError",
    "PlaneKernelResult",
    "solve_plane_kernel",
]
