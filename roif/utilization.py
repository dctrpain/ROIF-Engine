"""
ROIF Utilization Layer

Purpose
-------
This module provides a small, domain-independent physical layer between:

    demand / transmitted load T_i(t)

and

    effective capacity κ_i(t)

The central quantity is utilization:

    u_i(t) = |T_i(t)| / κ_i(t)

for magnitude-based loading.

From utilization we derive normalized reserve:

    R_i(t) = 1 - u_i(t)

and functional failure:

    u_i(t) >= failure_threshold

This module deliberately does NOT decide D_fast, D_root, D_origin, or Node*.

Its responsibility is only to expose physically auditable utilization,
reserve, and first capacity-exceedance events. Higher ROIF layers may then
use those events for causal-role inference.

Design principles
-----------------
- domain-independent;
- deterministic;
- immutable public result objects;
- explicit handling of zero capacity;
- no hidden normalization;
- no clipping of overload by default;
- suitable for scalar states and time trajectories;
- independent from the ROIF root detector.

Zero-capacity convention
------------------------
If κ == 0:

    T == 0  -> utilization = 0
    T != 0  -> utilization = +inf

This expresses the physical fact that a channel with no capacity can carry
no nonzero demand.

The signed reserve is kept intentionally:

    reserve = 1 - utilization

Therefore overloaded states have negative reserve.

A nonnegative reserve margin is also exposed:

    reserve_margin = max(0, reserve)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

_EPSILON = 1e-12


class UtilizationError(ValueError):
    """Raised when utilization analysis receives invalid data."""


class FailureCriterion(str, Enum):
    """
    Criterion used to classify a utilization state.

    GE_THRESHOLD
        Failure when utilization >= threshold.

    GT_THRESHOLD
        Failure only when utilization > threshold.
    """

    GE_THRESHOLD = "ge_threshold"
    GT_THRESHOLD = "gt_threshold"


@dataclass(frozen=True, slots=True)
class UtilizationState:
    """
    Physical utilization state for one channel at one instant.

    Parameters
    ----------
    channel_id:
        Stable channel identifier.

    demand:
        Current load / effort / transmitted demand.

    effective_capacity:
        Current effective capacity κ.

    utilization:
        |demand| / effective_capacity using the module's zero-capacity
        convention.

    reserve:
        Signed normalized reserve 1 - utilization.

    reserve_margin:
        Nonnegative reserve max(0, 1 - utilization).

    overloaded:
        True when the configured failure criterion is met.

    threshold:
        Utilization threshold defining functional failure.

    time_index:
        Optional discrete trajectory index.

    metadata:
        Read-only auxiliary information.
    """

    channel_id: str
    demand: float
    effective_capacity: float
    utilization: float
    reserve: float
    reserve_margin: float
    overloaded: bool
    threshold: float = 1.0
    time_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channel_id = str(self.channel_id).strip()
        if not channel_id:
            raise UtilizationError(
                "channel_id must not be empty."
            )

        object.__setattr__(
            self,
            "channel_id",
            channel_id,
        )

        for name in (
            "demand",
            "effective_capacity",
            "threshold",
        ):
            value = float(
                getattr(self, name)
            )

            if not math.isfinite(value):
                raise UtilizationError(
                    f"{name} must be finite."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        if self.effective_capacity < 0.0:
            raise UtilizationError(
                "effective_capacity must be non-negative."
            )

        if self.threshold <= 0.0:
            raise UtilizationError(
                "threshold must be positive."
            )

        utilization = float(
            self.utilization
        )

        if math.isnan(utilization):
            raise UtilizationError(
                "utilization must not be NaN."
            )

        if utilization < 0.0:
            raise UtilizationError(
                "utilization must be non-negative."
            )

        object.__setattr__(
            self,
            "utilization",
            utilization,
        )

        reserve = float(self.reserve)
        reserve_margin = float(
            self.reserve_margin
        )

        if math.isnan(reserve):
            raise UtilizationError(
                "reserve must not be NaN."
            )

        if (
            not math.isfinite(reserve_margin)
            or reserve_margin < 0.0
        ):
            raise UtilizationError(
                "reserve_margin must be finite and non-negative."
            )

        object.__setattr__(
            self,
            "reserve",
            reserve,
        )

        object.__setattr__(
            self,
            "reserve_margin",
            reserve_margin,
        )

        if not isinstance(
            self.overloaded,
            bool,
        ):
            raise UtilizationError(
                "overloaded must be bool."
            )

        if self.time_index is not None:
            if (
                isinstance(self.time_index, bool)
                or not isinstance(
                    self.time_index,
                    int,
                )
                or self.time_index < 0
            ):
                raise UtilizationError(
                    "time_index must be a nonnegative integer or None."
                )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """
    First physical capacity-exceedance event for one channel.
    """

    channel_id: str
    time_index: int
    demand: float
    effective_capacity: float
    utilization: float
    reserve: float
    threshold: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channel_id = str(self.channel_id).strip()
        if not channel_id:
            raise UtilizationError(
                "channel_id must not be empty."
            )

        if (
            isinstance(self.time_index, bool)
            or not isinstance(
                self.time_index,
                int,
            )
            or self.time_index < 0
        ):
            raise UtilizationError(
                "time_index must be a nonnegative integer."
            )

        object.__setattr__(
            self,
            "channel_id",
            channel_id,
        )

        for name in (
            "demand",
            "effective_capacity",
            "threshold",
        ):
            value = float(
                getattr(self, name)
            )
            if not math.isfinite(value):
                raise UtilizationError(
                    f"{name} must be finite."
                )
            object.__setattr__(
                self,
                name,
                value,
            )

        utilization = float(
            self.utilization
        )
        reserve = float(self.reserve)

        if math.isnan(utilization):
            raise UtilizationError(
                "utilization must not be NaN."
            )

        if math.isnan(reserve):
            raise UtilizationError(
                "reserve must not be NaN."
            )

        if utilization < 0.0:
            raise UtilizationError(
                "utilization must be non-negative."
            )

        if self.effective_capacity < 0.0:
            raise UtilizationError(
                "effective_capacity must be non-negative."
            )

        if self.threshold <= 0.0:
            raise UtilizationError(
                "threshold must be positive."
            )

        object.__setattr__(
            self,
            "utilization",
            utilization,
        )

        object.__setattr__(
            self,
            "reserve",
            reserve,
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


@dataclass(frozen=True, slots=True)
class UtilizationTrajectory:
    """
    Time history of utilization for one channel.

    All arrays are copied and made read-only.
    """

    channel_id: str
    demand: FloatArray
    effective_capacity: FloatArray
    utilization: FloatArray
    reserve: FloatArray
    overloaded: NDArray[np.bool_]
    threshold: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channel_id = str(self.channel_id).strip()

        if not channel_id:
            raise UtilizationError(
                "channel_id must not be empty."
            )

        object.__setattr__(
            self,
            "channel_id",
            channel_id,
        )

        arrays = {}

        for name in (
            "demand",
            "effective_capacity",
            "utilization",
            "reserve",
        ):
            array = np.asarray(
                getattr(self, name),
                dtype=np.float64,
            ).copy()

            if array.ndim != 1:
                raise UtilizationError(
                    f"{name} must be one-dimensional."
                )

            if name in {
                "demand",
                "effective_capacity",
            }:
                if not np.all(
                    np.isfinite(array)
                ):
                    raise UtilizationError(
                        f"{name} must contain only finite values."
                    )

            if name == "effective_capacity":
                if np.any(array < 0.0):
                    raise UtilizationError(
                        "effective_capacity must be non-negative."
                    )

            if name == "utilization":
                if np.any(np.isnan(array)):
                    raise UtilizationError(
                        "utilization must not contain NaN."
                    )

                if np.any(array < 0.0):
                    raise UtilizationError(
                        "utilization must be non-negative."
                    )

            if name == "reserve":
                if np.any(np.isnan(array)):
                    raise UtilizationError(
                        "reserve must not contain NaN."
                    )

            array.setflags(
                write=False
            )
            arrays[name] = array

        overloaded = np.asarray(
            self.overloaded,
            dtype=np.bool_,
        ).copy()

        if overloaded.ndim != 1:
            raise UtilizationError(
                "overloaded must be one-dimensional."
            )

        overloaded.setflags(
            write=False
        )

        lengths = {
            arrays["demand"].size,
            arrays["effective_capacity"].size,
            arrays["utilization"].size,
            arrays["reserve"].size,
            overloaded.size,
        }

        if len(lengths) != 1:
            raise UtilizationError(
                "trajectory arrays must have identical length."
            )

        threshold = float(
            self.threshold
        )

        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
        ):
            raise UtilizationError(
                "threshold must be finite and positive."
            )

        object.__setattr__(
            self,
            "demand",
            arrays["demand"],
        )
        object.__setattr__(
            self,
            "effective_capacity",
            arrays[
                "effective_capacity"
            ],
        )
        object.__setattr__(
            self,
            "utilization",
            arrays["utilization"],
        )
        object.__setattr__(
            self,
            "reserve",
            arrays["reserve"],
        )
        object.__setattr__(
            self,
            "overloaded",
            overloaded,
        )
        object.__setattr__(
            self,
            "threshold",
            threshold,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )

    @property
    def step_count(self) -> int:
        return int(
            self.utilization.size
        )

    @property
    def peak_utilization(self) -> float:
        if self.utilization.size == 0:
            return 0.0

        return float(
            np.max(
                self.utilization
            )
        )

    @property
    def minimum_reserve(self) -> float:
        if self.reserve.size == 0:
            return 1.0

        return float(
            np.min(
                self.reserve
            )
        )

    @property
    def first_failure_index(self) -> int | None:
        indices = np.flatnonzero(
            self.overloaded
        )

        if indices.size == 0:
            return None

        return int(
            indices[0]
        )

    @property
    def failed(self) -> bool:
        return (
            self.first_failure_index
            is not None
        )

    def state_at(
        self,
        time_index: int,
    ) -> UtilizationState:
        if (
            isinstance(time_index, bool)
            or not isinstance(
                time_index,
                int,
            )
            or time_index < 0
            or time_index >= self.step_count
        ):
            raise UtilizationError(
                "time_index is out of range."
            )

        return UtilizationState(
            channel_id=self.channel_id,
            demand=float(
                self.demand[
                    time_index
                ]
            ),
            effective_capacity=float(
                self.effective_capacity[
                    time_index
                ]
            ),
            utilization=float(
                self.utilization[
                    time_index
                ]
            ),
            reserve=float(
                self.reserve[
                    time_index
                ]
            ),
            reserve_margin=max(
                0.0,
                float(
                    self.reserve[
                        time_index
                    ]
                ),
            ),
            overloaded=bool(
                self.overloaded[
                    time_index
                ]
            ),
            threshold=self.threshold,
            time_index=time_index,
            metadata={
                "source": (
                    "UtilizationTrajectory"
                ),
            },
        )

    def first_failure_event(
        self,
    ) -> FailureEvent | None:
        index = self.first_failure_index

        if index is None:
            return None

        return FailureEvent(
            channel_id=self.channel_id,
            time_index=index,
            demand=float(
                self.demand[index]
            ),
            effective_capacity=float(
                self.effective_capacity[
                    index
                ]
            ),
            utilization=float(
                self.utilization[
                    index
                ]
            ),
            reserve=float(
                self.reserve[index]
            ),
            threshold=self.threshold,
            metadata=self.metadata,
        )


def _validate_threshold(
    threshold: float,
) -> float:
    value = float(threshold)

    if (
        not math.isfinite(value)
        or value <= 0.0
    ):
        raise UtilizationError(
            "threshold must be finite and positive."
        )

    return value


def _is_overloaded(
    utilization: float,
    threshold: float,
    criterion: FailureCriterion,
) -> bool:
    if criterion is FailureCriterion.GE_THRESHOLD:
        return utilization >= threshold

    if criterion is FailureCriterion.GT_THRESHOLD:
        return utilization > threshold

    raise UtilizationError(
        "unsupported FailureCriterion."
    )


def compute_utilization(
    demand: float,
    effective_capacity: float,
    *,
    epsilon: float = _EPSILON,
) -> float:
    """
    Compute magnitude-based physical utilization |T| / κ.

    Zero-capacity convention:
        κ <= epsilon and |T| <= epsilon -> 0
        κ <= epsilon and |T| >  epsilon -> +inf
    """

    demand_value = float(
        demand
    )
    capacity_value = float(
        effective_capacity
    )
    epsilon_value = float(
        epsilon
    )

    if not math.isfinite(
        demand_value
    ):
        raise UtilizationError(
            "demand must be finite."
        )

    if not math.isfinite(
        capacity_value
    ):
        raise UtilizationError(
            "effective_capacity must be finite."
        )

    if capacity_value < 0.0:
        raise UtilizationError(
            "effective_capacity must be non-negative."
        )

    if (
        not math.isfinite(
            epsilon_value
        )
        or epsilon_value < 0.0
    ):
        raise UtilizationError(
            "epsilon must be finite and non-negative."
        )

    magnitude = abs(
        demand_value
    )

    if capacity_value <= epsilon_value:
        if magnitude <= epsilon_value:
            return 0.0

        return math.inf

    return magnitude / capacity_value


def compute_reserve(
    utilization: float,
) -> float:
    """
    Signed normalized reserve.

        reserve = 1 - utilization

    Negative values explicitly represent overload.
    """

    value = float(
        utilization
    )

    if math.isnan(value):
        raise UtilizationError(
            "utilization must not be NaN."
        )

    if value < 0.0:
        raise UtilizationError(
            "utilization must be non-negative."
        )

    return 1.0 - value


def compute_reserve_margin(
    utilization: float,
) -> float:
    """
    Nonnegative reserve margin.

        max(0, 1 - utilization)
    """

    return max(
        0.0,
        compute_reserve(
            utilization
        ),
    )


def evaluate_utilization_state(
    channel_id: str,
    *,
    demand: float,
    effective_capacity: float,
    threshold: float = 1.0,
    criterion: FailureCriterion = FailureCriterion.GE_THRESHOLD,
    time_index: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> UtilizationState:
    """
    Evaluate one channel at one instant.
    """

    criterion = FailureCriterion(
        criterion
    )

    threshold_value = _validate_threshold(
        threshold
    )

    utilization = compute_utilization(
        demand,
        effective_capacity,
    )

    reserve = compute_reserve(
        utilization
    )

    return UtilizationState(
        channel_id=channel_id,
        demand=float(demand),
        effective_capacity=float(
            effective_capacity
        ),
        utilization=utilization,
        reserve=reserve,
        reserve_margin=max(
            0.0,
            reserve,
        ),
        overloaded=_is_overloaded(
            utilization,
            threshold_value,
            criterion,
        ),
        threshold=threshold_value,
        time_index=time_index,
        metadata=_readonly_mapping(
            metadata
        ),
    )


def build_utilization_trajectory(
    channel_id: str,
    demands: Sequence[float] | Iterable[float],
    capacities: Sequence[float] | Iterable[float] | float,
    *,
    threshold: float = 1.0,
    criterion: FailureCriterion = FailureCriterion.GE_THRESHOLD,
    metadata: Mapping[str, Any] | None = None,
) -> UtilizationTrajectory:
    """
    Build utilization and reserve histories for one channel.

    capacities may be:
        - one scalar capacity applied to all steps;
        - one capacity value per demand sample.
    """

    criterion = FailureCriterion(
        criterion
    )

    threshold_value = _validate_threshold(
        threshold
    )

    demand_array = np.asarray(
        tuple(demands),
        dtype=np.float64,
    )

    if demand_array.ndim != 1:
        raise UtilizationError(
            "demands must be one-dimensional."
        )

    if not np.all(
        np.isfinite(
            demand_array
        )
    ):
        raise UtilizationError(
            "demands must contain only finite values."
        )

    if np.isscalar(
        capacities
    ):
        capacity_value = float(
            capacities
        )

        if not math.isfinite(
            capacity_value
        ):
            raise UtilizationError(
                "capacity must be finite."
            )

        capacity_array = np.full(
            demand_array.shape,
            capacity_value,
            dtype=np.float64,
        )
    else:
        capacity_array = np.asarray(
            tuple(capacities),
            dtype=np.float64,
        )

    if capacity_array.ndim != 1:
        raise UtilizationError(
            "capacities must be one-dimensional."
        )

    if (
        capacity_array.size
        != demand_array.size
    ):
        raise UtilizationError(
            "demands and capacities must have identical length."
        )

    if not np.all(
        np.isfinite(
            capacity_array
        )
    ):
        raise UtilizationError(
            "capacities must contain only finite values."
        )

    if np.any(
        capacity_array < 0.0
    ):
        raise UtilizationError(
            "capacities must be non-negative."
        )

    utilization = np.asarray(
        [
            compute_utilization(
                demand,
                capacity,
            )
            for demand, capacity in zip(
                demand_array,
                capacity_array,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    reserve = np.asarray(
        [
            compute_reserve(
                value
            )
            for value in utilization
        ],
        dtype=np.float64,
    )

    overloaded = np.asarray(
        [
            _is_overloaded(
                float(value),
                threshold_value,
                criterion,
            )
            for value in utilization
        ],
        dtype=np.bool_,
    )

    return UtilizationTrajectory(
        channel_id=channel_id,
        demand=demand_array,
        effective_capacity=capacity_array,
        utilization=utilization,
        reserve=reserve,
        overloaded=overloaded,
        threshold=threshold_value,
        metadata=_readonly_mapping(
            metadata
        ),
    )


def first_capacity_exceedance(
    channel_id: str,
    demands: Sequence[float] | Iterable[float],
    capacities: Sequence[float] | Iterable[float] | float,
    *,
    threshold: float = 1.0,
    criterion: FailureCriterion = FailureCriterion.GE_THRESHOLD,
    metadata: Mapping[str, Any] | None = None,
) -> FailureEvent | None:
    """
    Return the first functional-capacity failure event for one channel.
    """

    trajectory = build_utilization_trajectory(
        channel_id,
        demands,
        capacities,
        threshold=threshold,
        criterion=criterion,
        metadata=metadata,
    )

    return trajectory.first_failure_event()


def earliest_failure_event(
    trajectories: Iterable[UtilizationTrajectory],
) -> FailureEvent | None:
    """
    Return the earliest capacity-exceedance event across channels.

    Tie-breaking is deterministic:
        1. smaller time_index;
        2. larger utilization;
        3. lexicographically smaller channel_id.

    The second rule makes simultaneous failure favor the stronger physical
    exceedance while remaining fully deterministic.
    """

    events = tuple(
        event
        for trajectory in trajectories
        if (
            event
            := trajectory.first_failure_event()
        )
        is not None
    )

    if not events:
        return None

    return min(
        events,
        key=lambda event: (
            event.time_index,
            -event.utilization,
            event.channel_id,
        ),
    )


def utilization_matrix(
    demands: Sequence[
        Sequence[float]
    ]
    | FloatArray,
    capacities: Sequence[
        Sequence[float]
    ]
    | FloatArray,
) -> FloatArray:
    """
    Compute elementwise utilization for equally shaped 2-D arrays.

    Intended for future multi-channel trajectory integration.
    Rows are time steps; columns are channels.
    """

    demand_matrix = np.asarray(
        demands,
        dtype=np.float64,
    )

    capacity_matrix = np.asarray(
        capacities,
        dtype=np.float64,
    )

    if demand_matrix.ndim != 2:
        raise UtilizationError(
            "demands must be two-dimensional."
        )

    if capacity_matrix.ndim != 2:
        raise UtilizationError(
            "capacities must be two-dimensional."
        )

    if (
        demand_matrix.shape
        != capacity_matrix.shape
    ):
        raise UtilizationError(
            "demands and capacities must have identical shape."
        )

    if not np.all(
        np.isfinite(
            demand_matrix
        )
    ):
        raise UtilizationError(
            "demands must contain only finite values."
        )

    if not np.all(
        np.isfinite(
            capacity_matrix
        )
    ):
        raise UtilizationError(
            "capacities must contain only finite values."
        )

    if np.any(
        capacity_matrix < 0.0
    ):
        raise UtilizationError(
            "capacities must be non-negative."
        )

    result = np.empty_like(
        demand_matrix,
        dtype=np.float64,
    )

    for row in range(
        demand_matrix.shape[0]
    ):
        for column in range(
            demand_matrix.shape[1]
        ):
            result[
                row,
                column,
            ] = compute_utilization(
                demand_matrix[
                    row,
                    column,
                ],
                capacity_matrix[
                    row,
                    column,
                ],
            )

    return result


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        dict(
            value or {}
        )
    )


__all__ = [
    "FailureCriterion",
    "FailureEvent",
    "UtilizationError",
    "UtilizationState",
    "UtilizationTrajectory",
    "build_utilization_trajectory",
    "compute_reserve",
    "compute_reserve_margin",
    "compute_utilization",
    "earliest_failure_event",
    "evaluate_utilization_state",
    "first_capacity_exceedance",
    "utilization_matrix",
]
