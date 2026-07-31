"""
ROIF fast mechanical layer.

The fast layer solves the quasi-static equilibrium problem

    A(x) T = f

and evaluates element utilization

    u_i = |T_i| / κ_i(d_i, x).

Here:

    A(x)  — equilibrium / geometry matrix;
    T     — signed internal element efforts;
    f     — external load vector;
    κ_i   — available directional capacity of element i;
    u_i   — non-negative utilization ratio.

The element with the greatest instantaneous utilization is

    D_fast = argmax_i(u_i).

Important
---------
D_fast is an instantaneous fast-layer object. It is not automatically the
cascade root D_root and not automatically the optimal intervention Node*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .capacity_tensor import CapacityTensor
from .direction import Direction


FloatArray: TypeAlias = NDArray[np.float64]

_EPSILON = 1e-12


class FastLayerError(ValueError):
    """Raised when the fast-layer problem is invalid or cannot be solved."""


class SolveMethod(str, Enum):
    """Numerical method used to solve A T = f."""

    AUTO = "auto"
    DIRECT = "direct"
    LEAST_SQUARES = "least_squares"
    PSEUDOINVERSE = "pseudoinverse"


@dataclass(frozen=True, slots=True)
class FastLayerResult:
    """Immutable result of a fast-layer equilibrium calculation."""

    equilibrium_matrix: FloatArray
    load_vector: FloatArray
    efforts: FloatArray
    capacities: FloatArray
    utilization: FloatArray
    signed_utilization: FloatArray
    residual_vector: FloatArray
    residual_norm: float
    rank: int
    condition_number: float
    solve_method: SolveMethod
    d_fast_index: int | None
    is_exact_equilibrium: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "equilibrium_matrix", self._readonly_copy(self.equilibrium_matrix))
        object.__setattr__(self, "load_vector", self._readonly_copy(self.load_vector))
        object.__setattr__(self, "efforts", self._readonly_copy(self.efforts))
        object.__setattr__(self, "capacities", self._readonly_copy(self.capacities))
        object.__setattr__(self, "utilization", self._readonly_copy(self.utilization))
        object.__setattr__(self, "signed_utilization", self._readonly_copy(self.signed_utilization))
        object.__setattr__(self, "residual_vector", self._readonly_copy(self.residual_vector))

    @property
    def element_count(self) -> int:
        return int(self.efforts.size)

    @property
    def equation_count(self) -> int:
        return int(self.load_vector.size)

    @property
    def max_utilization(self) -> float:
        if self.utilization.size == 0:
            return 0.0
        return float(np.max(self.utilization))

    @property
    def d_fast(self) -> int | None:
        return self.d_fast_index

    @property
    def overloaded_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.utilization > 1.0))

    @property
    def critical_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.utilization >= 1.0))

    @property
    def has_overload(self) -> bool:
        return bool(np.any(self.utilization > 1.0))

    @property
    def has_unavailable_loaded_element(self) -> bool:
        return bool(np.any(np.isinf(self.utilization)))

    @property
    def reconstructed_load(self) -> FloatArray:
        return self.equilibrium_matrix @ self.efforts

    def utilization_of(self, element_index: int) -> float:
        self._validate_element_index(element_index)
        return float(self.utilization[element_index])

    def effort_of(self, element_index: int) -> float:
        self._validate_element_index(element_index)
        return float(self.efforts[element_index])

    def capacity_of(self, element_index: int) -> float:
        self._validate_element_index(element_index)
        return float(self.capacities[element_index])

    def as_dict(self) -> dict[str, object]:
        return {
            "equilibrium_matrix": self.equilibrium_matrix.tolist(),
            "load_vector": self.load_vector.tolist(),
            "efforts": self.efforts.tolist(),
            "capacities": self.capacities.tolist(),
            "utilization": self.utilization.tolist(),
            "signed_utilization": self.signed_utilization.tolist(),
            "residual_vector": self.residual_vector.tolist(),
            "residual_norm": self.residual_norm,
            "rank": self.rank,
            "condition_number": self.condition_number,
            "solve_method": self.solve_method.value,
            "d_fast_index": self.d_fast_index,
            "is_exact_equilibrium": self.is_exact_equilibrium,
            "overloaded_indices": list(self.overloaded_indices),
        }

    def _validate_element_index(self, element_index: int) -> None:
        if not isinstance(element_index, int):
            raise FastLayerError("element_index must be an integer.")
        if element_index < 0 or element_index >= self.element_count:
            raise FastLayerError(
                f"element_index must be in range [0, {self.element_count - 1}], "
                f"received {element_index}."
            )

    @staticmethod
    def _readonly_copy(value: FloatArray) -> FloatArray:
        result = np.asarray(value, dtype=np.float64).copy()
        result.setflags(write=False)
        return result


class FastLayer:
    """Solver for the ROIF fast mechanical layer."""

    def __init__(
        self,
        *,
        method: SolveMethod | str = SolveMethod.AUTO,
        residual_tolerance: float = 1e-9,
        relative_residual_tolerance: float = 1e-9,
        rank_tolerance: float | None = None,
        capacity_epsilon: float = _EPSILON,
    ) -> None:
        self.method = self._coerce_method(method)
        self.residual_tolerance = self._validate_non_negative_scalar(
            residual_tolerance, name="residual_tolerance"
        )
        self.relative_residual_tolerance = self._validate_non_negative_scalar(
            relative_residual_tolerance, name="relative_residual_tolerance"
        )

        if rank_tolerance is None:
            self.rank_tolerance = None
        else:
            self.rank_tolerance = self._validate_positive_scalar(
                rank_tolerance, name="rank_tolerance"
            )

        self.capacity_epsilon = self._validate_non_negative_scalar(
            capacity_epsilon, name="capacity_epsilon"
        )

    def solve(
        self,
        equilibrium_matrix: Iterable[Iterable[float]] | FloatArray,
        load: Direction | Iterable[float] | FloatArray,
        capacities: Sequence[CapacityTensor] | Iterable[float] | FloatArray,
        *,
        service_directions: Sequence[Direction | Iterable[float] | FloatArray] | None = None,
        configurations: Sequence[Iterable[Iterable[float]] | FloatArray | None] | None = None,
    ) -> FastLayerResult:
        matrix = self._validate_matrix(equilibrium_matrix)
        load_vector = self._coerce_load_vector(load, expected_dimension=matrix.shape[0])

        efforts, actual_method = self._solve_efforts(matrix, load_vector)

        evaluated_capacities = self._evaluate_capacities(
            matrix=matrix,
            capacities=capacities,
            service_directions=service_directions,
            configurations=configurations,
        )

        utilization, signed_utilization = self._compute_utilization(
            efforts, evaluated_capacities
        )

        residual_vector = matrix @ efforts - load_vector
        residual_norm = float(np.linalg.norm(residual_vector))
        load_norm = float(np.linalg.norm(load_vector))
        accepted_residual = (
            self.residual_tolerance
            + self.relative_residual_tolerance * load_norm
        )

        rank = int(np.linalg.matrix_rank(matrix, tol=self.rank_tolerance))
        condition_number = self._condition_number(matrix)
        d_fast_index = self._locate_d_fast(utilization)

        return FastLayerResult(
            equilibrium_matrix=matrix,
            load_vector=load_vector,
            efforts=efforts,
            capacities=evaluated_capacities,
            utilization=utilization,
            signed_utilization=signed_utilization,
            residual_vector=residual_vector,
            residual_norm=residual_norm,
            rank=rank,
            condition_number=condition_number,
            solve_method=actual_method,
            d_fast_index=d_fast_index,
            is_exact_equilibrium=residual_norm <= accepted_residual,
        )

    def solve_with_numeric_capacities(
        self,
        equilibrium_matrix: Iterable[Iterable[float]] | FloatArray,
        load: Direction | Iterable[float] | FloatArray,
        capacities: Iterable[float] | FloatArray,
    ) -> FastLayerResult:
        return self.solve(
            equilibrium_matrix=equilibrium_matrix,
            load=load,
            capacities=capacities,
        )

    def _solve_efforts(
        self,
        matrix: FloatArray,
        load_vector: FloatArray,
    ) -> tuple[FloatArray, SolveMethod]:
        method = self.method

        if method is SolveMethod.AUTO:
            is_square = matrix.shape[0] == matrix.shape[1]
            rank = int(np.linalg.matrix_rank(matrix, tol=self.rank_tolerance))
            if is_square and rank == matrix.shape[0]:
                method = SolveMethod.DIRECT
            else:
                method = SolveMethod.LEAST_SQUARES

        if method is SolveMethod.DIRECT:
            if matrix.shape[0] != matrix.shape[1]:
                raise FastLayerError("DIRECT solving requires a square equilibrium matrix.")
            try:
                efforts = np.linalg.solve(matrix, load_vector)
            except np.linalg.LinAlgError as error:
                raise FastLayerError(
                    "Direct equilibrium solve failed because the matrix "
                    "is singular or numerically unstable."
                ) from error

        elif method is SolveMethod.LEAST_SQUARES:
            efforts, _, _, _ = np.linalg.lstsq(
                matrix, load_vector, rcond=self.rank_tolerance
            )

        elif method is SolveMethod.PSEUDOINVERSE:
            efforts = np.linalg.pinv(
                matrix,
                rcond=(
                    self.rank_tolerance
                    if self.rank_tolerance is not None
                    else 1e-15
                ),
            ) @ load_vector

        else:
            raise FastLayerError(f"Unsupported solve method: {method!r}.")

        efforts = np.asarray(efforts, dtype=np.float64)

        if efforts.ndim != 1:
            raise FastLayerError("Internal solver returned a non-vector effort result.")
        if efforts.size != matrix.shape[1]:
            raise FastLayerError(
                "Internal effort count does not match the number of matrix columns."
            )
        if not np.all(np.isfinite(efforts)):
            raise FastLayerError("Equilibrium solution contains non-finite efforts.")

        return efforts, method

    def _evaluate_capacities(
        self,
        *,
        matrix: FloatArray,
        capacities: Sequence[CapacityTensor] | Iterable[float] | FloatArray,
        service_directions: Sequence[Direction | Iterable[float] | FloatArray] | None,
        configurations: Sequence[Iterable[Iterable[float]] | FloatArray | None] | None,
    ) -> FloatArray:
        element_count = matrix.shape[1]
        capacity_items = list(capacities)

        if len(capacity_items) != element_count:
            raise FastLayerError(
                "Capacity count must equal the number of elements: "
                f"expected {element_count}, received {len(capacity_items)}."
            )

        if not capacity_items:
            return np.empty(0, dtype=np.float64)

        contains_tensors = all(isinstance(item, CapacityTensor) for item in capacity_items)
        contains_numbers = all(not isinstance(item, CapacityTensor) for item in capacity_items)

        if not contains_tensors and not contains_numbers:
            raise FastLayerError(
                "capacities must contain either only CapacityTensor objects "
                "or only numeric capacity values."
            )

        if contains_numbers:
            if service_directions is not None:
                raise FastLayerError(
                    "service_directions cannot be used when capacities are already numeric."
                )
            if configurations is not None:
                raise FastLayerError(
                    "configurations cannot be used when capacities are already numeric."
                )
            return self._validate_numeric_capacities(capacity_items)

        tensors = [item for item in capacity_items if isinstance(item, CapacityTensor)]

        directions = self._resolve_service_directions(
            matrix=matrix,
            service_directions=service_directions,
        )
        resolved_configurations = self._resolve_configurations(
            configurations=configurations,
            element_count=element_count,
        )

        values = np.empty(element_count, dtype=np.float64)

        for index, tensor in enumerate(tensors):
            direction = directions[index]
            direction_vector = (
                direction.unit_vector
                if isinstance(direction, Direction)
                else np.asarray(direction, dtype=np.float64)
            )
            configuration = resolved_configurations[index]

            try:
                values[index] = tensor.kappa(direction_vector, configuration)
            except TypeError:
                values[index] = tensor.kappa(
                    direction_vector,
                    configuration=configuration,
                )
            except ValueError as error:
                raise FastLayerError(
                    f"Capacity evaluation failed for element {index}: {error}"
                ) from error

        return self._validate_numeric_capacities(values)

    def _resolve_service_directions(
        self,
        *,
        matrix: FloatArray,
        service_directions: Sequence[Direction | Iterable[float] | FloatArray] | None,
    ) -> list[Direction | FloatArray]:
        element_count = matrix.shape[1]

        if service_directions is not None:
            directions = list(service_directions)
            if len(directions) != element_count:
                raise FastLayerError(
                    "service_directions count must equal the number of elements: "
                    f"expected {element_count}, received {len(directions)}."
                )

            return [
                self._validate_service_direction(
                    direction,
                    expected_dimension=matrix.shape[0],
                    element_index=index,
                )
                for index, direction in enumerate(directions)
            ]

        inferred_directions: list[FloatArray] = []

        for index in range(element_count):
            column = matrix[:, index]
            norm = float(np.linalg.norm(column))

            if norm <= _EPSILON:
                raise FastLayerError(
                    f"Cannot infer service direction for element {index}: "
                    "the corresponding equilibrium-matrix column is zero. "
                    "Provide service_directions explicitly."
                )

            inferred_directions.append(column / norm)

        return inferred_directions

    @staticmethod
    def _resolve_configurations(
        *,
        configurations: Sequence[Iterable[Iterable[float]] | FloatArray | None] | None,
        element_count: int,
    ) -> list[Iterable[Iterable[float]] | FloatArray | None]:
        if configurations is None:
            return [None] * element_count

        resolved = list(configurations)

        if len(resolved) != element_count:
            raise FastLayerError(
                "configurations count must equal the number of elements: "
                f"expected {element_count}, received {len(resolved)}."
            )

        return resolved

    def _compute_utilization(
        self,
        efforts: FloatArray,
        capacities: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        utilization = np.empty_like(efforts)
        signed_utilization = np.empty_like(efforts)

        usable_capacity = capacities > self.capacity_epsilon
        zero_effort = np.abs(efforts) <= _EPSILON

        utilization[usable_capacity] = (
            np.abs(efforts[usable_capacity]) / capacities[usable_capacity]
        )
        signed_utilization[usable_capacity] = (
            efforts[usable_capacity] / capacities[usable_capacity]
        )

        unavailable = ~usable_capacity

        utilization[unavailable & zero_effort] = 0.0
        signed_utilization[unavailable & zero_effort] = 0.0

        utilization[unavailable & ~zero_effort] = np.inf
        signed_utilization[unavailable & ~zero_effort] = np.copysign(
            np.inf,
            efforts[unavailable & ~zero_effort],
        )

        return utilization, signed_utilization

    @staticmethod
    def _locate_d_fast(utilization: FloatArray) -> int | None:
        if utilization.size == 0:
            return None
        return int(np.argmax(utilization))

    @staticmethod
    def _validate_matrix(
        matrix: Iterable[Iterable[float]] | FloatArray,
    ) -> FloatArray:
        result = np.asarray(matrix, dtype=np.float64)

        if result.ndim != 2:
            raise FastLayerError("equilibrium_matrix must be two-dimensional.")
        if result.shape[0] == 0:
            raise FastLayerError("equilibrium_matrix must contain at least one equation.")
        if result.shape[1] == 0:
            raise FastLayerError("equilibrium_matrix must contain at least one element.")
        if not np.all(np.isfinite(result)):
            raise FastLayerError("equilibrium_matrix contains non-finite values.")

        return result.copy()

    @staticmethod
    def _coerce_load_vector(
        load: Direction | Iterable[float] | FloatArray,
        *,
        expected_dimension: int,
    ) -> FloatArray:
        if isinstance(load, Direction):
            result = load.force_vector
        else:
            result = np.asarray(load, dtype=np.float64)

        if result.ndim != 1:
            raise FastLayerError("load must be a one-dimensional vector.")
        if result.size != expected_dimension:
            raise FastLayerError(
                f"load must have dimension {expected_dimension}, received {result.size}."
            )
        if not np.all(np.isfinite(result)):
            raise FastLayerError("load contains non-finite values.")

        return result.copy()

    @staticmethod
    def _validate_service_direction(
        direction: Direction | Iterable[float] | FloatArray,
        *,
        expected_dimension: int,
        element_index: int,
    ) -> Direction | FloatArray:
        if isinstance(direction, Direction):
            if direction.dimension != expected_dimension:
                raise FastLayerError(
                    f"Service direction for element {element_index} "
                    f"must have dimension {expected_dimension}, "
                    f"received {direction.dimension}."
                )
            return direction

        result = np.asarray(direction, dtype=np.float64)

        if result.ndim != 1:
            raise FastLayerError(
                f"Service direction for element {element_index} must be one-dimensional."
            )
        if result.size != expected_dimension:
            raise FastLayerError(
                f"Service direction for element {element_index} "
                f"must have dimension {expected_dimension}, received {result.size}."
            )
        if not np.all(np.isfinite(result)):
            raise FastLayerError(
                f"Service direction for element {element_index} "
                "contains non-finite values."
            )

        norm = float(np.linalg.norm(result))

        if norm <= _EPSILON:
            raise FastLayerError(
                f"Service direction for element {element_index} cannot be zero."
            )

        return result / norm

    @staticmethod
    def _validate_numeric_capacities(
        capacities: Iterable[float] | FloatArray,
    ) -> FloatArray:
        result = np.asarray(list(capacities), dtype=np.float64)

        if result.ndim != 1:
            raise FastLayerError(
                "Numeric capacities must form a one-dimensional vector."
            )
        if not np.all(np.isfinite(result)):
            raise FastLayerError("Numeric capacities contain non-finite values.")
        if np.any(result < 0.0):
            raise FastLayerError("Numeric capacities must be non-negative.")

        return result.copy()

    @staticmethod
    def _condition_number(matrix: FloatArray) -> float:
        try:
            value = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError:
            return float("inf")

        if np.isnan(value):
            return float("inf")

        return value

    @staticmethod
    def _coerce_method(method: SolveMethod | str) -> SolveMethod:
        if isinstance(method, SolveMethod):
            return method

        try:
            return SolveMethod(method)
        except (TypeError, ValueError) as error:
            valid = ", ".join(item.value for item in SolveMethod)
            raise FastLayerError(
                f"Unknown solve method {method!r}. Expected one of: {valid}."
            ) from error

    @staticmethod
    def _validate_non_negative_scalar(value: float, *, name: str) -> float:
        result = float(value)

        if not np.isfinite(result):
            raise FastLayerError(f"{name} must be finite.")
        if result < 0.0:
            raise FastLayerError(
                f"{name} must be non-negative, received {result}."
            )

        return result

    @staticmethod
    def _validate_positive_scalar(value: float, *, name: str) -> float:
        result = float(value)

        if not np.isfinite(result):
            raise FastLayerError(f"{name} must be finite.")
        if result <= 0.0:
            raise FastLayerError(f"{name} must be positive, received {result}.")

        return result


def solve_fast_layer(
    equilibrium_matrix: Iterable[Iterable[float]] | FloatArray,
    load: Direction | Iterable[float] | FloatArray,
    capacities: Sequence[CapacityTensor] | Iterable[float] | FloatArray,
    *,
    service_directions: Sequence[Direction | Iterable[float] | FloatArray] | None = None,
    configurations: Sequence[Iterable[Iterable[float]] | FloatArray | None] | None = None,
    method: SolveMethod | str = SolveMethod.AUTO,
    residual_tolerance: float = 1e-9,
    relative_residual_tolerance: float = 1e-9,
) -> FastLayerResult:
    solver = FastLayer(
        method=method,
        residual_tolerance=residual_tolerance,
        relative_residual_tolerance=relative_residual_tolerance,
    )

    return solver.solve(
        equilibrium_matrix=equilibrium_matrix,
        load=load,
        capacities=capacities,
        service_directions=service_directions,
        configurations=configurations,
    )
