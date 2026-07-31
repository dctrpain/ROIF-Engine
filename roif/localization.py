"""
Inverse cascade localization for ROIF Engine v1.

This module implements the three distinct localization objects defined by the
ROIF mathematical framework:

    D_fast  вЂ” earliest reserve-threshold crossing,
    D_root  вЂ” hypothetical origin minimizing localization discrepancy,
    Node*   вЂ” intentionally not computed here; it belongs to optimization.py.

The module is domain-independent. It does not assume a particular graph,
solver, state class, or physical interpretation. A caller supplies:

1. an observed state vector;
2. candidate origin identifiers;
3. a forward simulator:
       simulate(candidate_id) -> predicted state or LocalizationSimulation;
4. optionally, reserve trajectories for D_fast estimation.

The reference discrepancy is

    J_i = ||S_i_pred - S_obs||,

with configurable L1, L2, L-infinity, weighted L2, or custom metrics.

The API is immutable, deterministic, serializable, and suitable for seeded
validation experiments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Hashable, Protocol, TypeAlias, runtime_checkable
import math

import numpy as np


class LocalizationError(ValueError):
    """Raised when localization input or configuration is invalid."""


NodeId: TypeAlias = Hashable
StateLike: TypeAlias = Sequence[float] | np.ndarray
DistanceFunction: TypeAlias = Callable[[np.ndarray, np.ndarray], float]


_EPSILON = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise LocalizationError(f"{name} must be a real number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LocalizationError(f"{name} must be a real number.") from exc

    if not math.isfinite(result):
        raise LocalizationError(f"{name} must be finite.")

    if minimum is not None and result < minimum:
        raise LocalizationError(
            f"{name} must be greater than or equal to {minimum}."
        )

    if maximum is not None and result > maximum:
        raise LocalizationError(
            f"{name} must be less than or equal to {maximum}."
        )

    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LocalizationError(
            f"{name} must be an integer greater than or equal to 1."
        )
    return value


def _normalize_node_id(value: Any, *, name: str = "node_id") -> NodeId:
    if value is None:
        raise LocalizationError(f"{name} cannot be None.")

    try:
        hash(value)
    except TypeError as exc:
        raise LocalizationError(f"{name} must be hashable.") from exc

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise LocalizationError(f"{name} cannot be empty.")
        return normalized

    return value


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise LocalizationError("metadata must be a mapping.")

    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise LocalizationError("metadata keys must be strings.")

        normalized = key.strip()
        if not normalized:
            raise LocalizationError("metadata keys cannot be empty.")

        copied[normalized] = value

    return MappingProxyType(copied)


def _state_vector(
    value: StateLike,
    *,
    name: str,
    expected_size: int | None = None,
) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise LocalizationError(
            f"{name} must be a one-dimensional numeric state vector."
        )

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LocalizationError(
            f"{name} must be a one-dimensional numeric state vector."
        ) from exc

    if array.ndim != 1:
        raise LocalizationError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise LocalizationError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(array)):
        raise LocalizationError(f"{name} must contain only finite values.")

    if expected_size is not None and array.size != expected_size:
        raise LocalizationError(
            f"{name} has size {array.size}; expected {expected_size}."
        )

    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _optional_state_vector(
    value: StateLike | None,
    *,
    name: str,
    expected_size: int | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    return _state_vector(
        value,
        name=name,
        expected_size=expected_size,
    )


def _normalize_candidates(
    candidates: Iterable[NodeId],
) -> tuple[NodeId, ...]:
    if isinstance(candidates, (str, bytes)):
        raise LocalizationError(
            "candidates must be an iterable of node identifiers."
        )

    try:
        values = tuple(candidates)
    except TypeError as exc:
        raise LocalizationError(
            "candidates must be an iterable of node identifiers."
        ) from exc

    if not values:
        raise LocalizationError("candidates cannot be empty.")

    normalized: list[NodeId] = []
    seen: set[NodeId] = set()

    for value in values:
        node_id = _normalize_node_id(value, name="candidate node_id")
        if node_id in seen:
            raise LocalizationError(
                f"Duplicate candidate node_id: {node_id!r}."
            )
        seen.add(node_id)
        normalized.append(node_id)

    return tuple(normalized)


class LocalizationMetric(str, Enum):
    """Built-in discrepancy metrics for the localization functional."""

    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    WEIGHTED_L2 = "weighted_l2"
    CUSTOM = "custom"


class LocalizationStatus(str, Enum):
    """Outcome of a root-localization pass."""

    LOCALIZED = "localized"
    TIED = "tied"
    PARTIAL = "partial"
    FAILED = "failed"


class SimulationStatus(str, Enum):
    """Outcome of one hypothetical-origin forward simulation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FastFailureStatus(str, Enum):
    """Availability state of D_fast estimation."""

    IDENTIFIED = "identified"
    TIED = "tied"
    NOT_REACHED = "not_reached"


@dataclass(frozen=True, slots=True)
class LocalizationConfig:
    """Configuration for inverse cascade localization."""

    metric: LocalizationMetric = LocalizationMetric.L2
    weights: tuple[float, ...] | None = None
    tie_tolerance: float = 1e-9
    top_k: int = 5
    critical_reserve: float = 0.0
    allow_partial_results: bool = True
    capture_simulation_errors: bool = True
    normalize_by_dimension: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.metric, LocalizationMetric):
            raise LocalizationError(
                "metric must be a LocalizationMetric."
            )

        object.__setattr__(
            self,
            "tie_tolerance",
            _finite_float(
                self.tie_tolerance,
                name="tie_tolerance",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "critical_reserve",
            _finite_float(
                self.critical_reserve,
                name="critical_reserve",
            ),
        )
        object.__setattr__(
            self,
            "top_k",
            _positive_int(self.top_k, name="top_k"),
        )

        for name in (
            "allow_partial_results",
            "capture_simulation_errors",
            "normalize_by_dimension",
        ):
            if not isinstance(getattr(self, name), bool):
                raise LocalizationError(f"{name} must be a bool.")

        if self.weights is not None:
            weights = _state_vector(
                self.weights,
                name="weights",
            )
            if np.any(weights < 0.0):
                raise LocalizationError(
                    "weights cannot contain negative values."
                )
            if not np.any(weights > 0.0):
                raise LocalizationError(
                    "weights must contain at least one positive value."
                )
            object.__setattr__(
                self,
                "weights",
                tuple(float(item) for item in weights),
            )

        if (
            self.metric is LocalizationMetric.WEIGHTED_L2
            and self.weights is None
        ):
            raise LocalizationError(
                "weights are required for WEIGHTED_L2."
            )

        if (
            self.metric is not LocalizationMetric.WEIGHTED_L2
            and self.weights is not None
        ):
            raise LocalizationError(
                "weights are only valid for WEIGHTED_L2."
            )


@dataclass(frozen=True, slots=True)
class LocalizationSimulation:
    """Forward-simulation result for one hypothetical origin."""

    candidate_id: NodeId
    predicted_state: np.ndarray
    converged: bool = True
    iterations: int | None = None
    propagation_time: float | None = None
    reserve_history: Mapping[NodeId, Sequence[tuple[float, float]]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _normalize_node_id(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "predicted_state",
            _state_vector(
                self.predicted_state,
                name="predicted_state",
            ),
        )

        if not isinstance(self.converged, bool):
            raise LocalizationError("converged must be a bool.")

        if self.iterations is not None:
            if (
                isinstance(self.iterations, bool)
                or not isinstance(self.iterations, int)
                or self.iterations < 0
            ):
                raise LocalizationError(
                    "iterations must be a non-negative integer or None."
                )

        object.__setattr__(
            self,
            "propagation_time",
            (
                None
                if self.propagation_time is None
                else _finite_float(
                    self.propagation_time,
                    name="propagation_time",
                    minimum=0.0,
                )
            ),
        )
        object.__setattr__(
            self,
            "reserve_history",
            _normalize_reserve_history(self.reserve_history),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "predicted_state": self.predicted_state.tolist(),
            "converged": self.converged,
            "iterations": self.iterations,
            "propagation_time": self.propagation_time,
            "reserve_history": (
                None
                if self.reserve_history is None
                else {
                    node_id: [list(item) for item in series]
                    for node_id, series in self.reserve_history.items()
                }
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LocalizationScore:
    """Localization functional value for one candidate origin."""

    candidate_id: NodeId
    discrepancy: float
    rank: int
    tied_for_best: bool
    simulation_status: SimulationStatus
    converged: bool | None
    predicted_state: np.ndarray | None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _normalize_node_id(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "discrepancy",
            _finite_float(
                self.discrepancy,
                name="discrepancy",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "rank",
            _positive_int(self.rank, name="rank"),
        )

        if not isinstance(self.tied_for_best, bool):
            raise LocalizationError("tied_for_best must be a bool.")

        if not isinstance(self.simulation_status, SimulationStatus):
            raise LocalizationError(
                "simulation_status must be a SimulationStatus."
            )

        if self.converged is not None and not isinstance(
            self.converged,
            bool,
        ):
            raise LocalizationError("converged must be a bool or None.")

        object.__setattr__(
            self,
            "predicted_state",
            _optional_state_vector(
                self.predicted_state,
                name="predicted_state",
            ),
        )

        for name in ("error_type", "error_message"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str):
                    raise LocalizationError(
                        f"{name} must be a string or None."
                    )
                normalized = value.strip()
                if not normalized:
                    raise LocalizationError(f"{name} cannot be empty.")
                object.__setattr__(self, name, normalized)

        if self.simulation_status is SimulationStatus.FAILED:
            if self.error_type is None or self.error_message is None:
                raise LocalizationError(
                    "Failed simulation score requires error details."
                )
        elif self.error_type is not None or self.error_message is not None:
            raise LocalizationError(
                "Successful simulation score cannot contain error details."
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "discrepancy": self.discrepancy,
            "rank": self.rank,
            "tied_for_best": self.tied_for_best,
            "simulation_status": self.simulation_status.value,
            "converged": self.converged,
            "predicted_state": (
                None
                if self.predicted_state is None
                else self.predicted_state.tolist()
            ),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FastFailureResult:
    """Result of D_fast estimation from reserve trajectories."""

    status: FastFailureStatus
    node_id: NodeId | None
    crossing_time: float | None
    tied_node_ids: tuple[NodeId, ...]
    critical_reserve: float
    crossing_times: Mapping[NodeId, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, FastFailureStatus):
            raise LocalizationError(
                "status must be a FastFailureStatus."
            )

        if self.node_id is not None:
            object.__setattr__(
                self,
                "node_id",
                _normalize_node_id(self.node_id, name="node_id"),
            )

        object.__setattr__(
            self,
            "crossing_time",
            (
                None
                if self.crossing_time is None
                else _finite_float(
                    self.crossing_time,
                    name="crossing_time",
                    minimum=0.0,
                )
            ),
        )

        tied = tuple(
            _normalize_node_id(item, name="tied_node_id")
            for item in self.tied_node_ids
        )
        if len(tied) != len(set(tied)):
            raise LocalizationError(
                "tied_node_ids cannot contain duplicates."
            )
        object.__setattr__(self, "tied_node_ids", tied)

        object.__setattr__(
            self,
            "critical_reserve",
            _finite_float(
                self.critical_reserve,
                name="critical_reserve",
            ),
        )

        normalized_times: dict[NodeId, float] = {}
        for key, value in self.crossing_times.items():
            node_id = _normalize_node_id(key, name="crossing node_id")
            normalized_times[node_id] = _finite_float(
                value,
                name="crossing time",
                minimum=0.0,
            )
        object.__setattr__(
            self,
            "crossing_times",
            MappingProxyType(normalized_times),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

        if self.status is FastFailureStatus.NOT_REACHED:
            if (
                self.node_id is not None
                or self.crossing_time is not None
                or self.tied_node_ids
                or self.crossing_times
            ):
                raise LocalizationError(
                    "NOT_REACHED cannot contain a D_fast result."
                )
        else:
            if self.node_id is None or self.crossing_time is None:
                raise LocalizationError(
                    "Identified D_fast requires node_id and crossing_time."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "node_id": self.node_id,
            "crossing_time": self.crossing_time,
            "tied_node_ids": list(self.tied_node_ids),
            "critical_reserve": self.critical_reserve,
            "crossing_times": dict(self.crossing_times),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RootLocalizationResult:
    """Immutable D_root localization output."""

    status: LocalizationStatus
    root_id: NodeId | None
    best_discrepancy: float | None
    tied_root_ids: tuple[NodeId, ...]
    scores: tuple[LocalizationScore, ...]
    observed_state: np.ndarray
    metric: LocalizationMetric
    successful_candidate_count: int
    failed_candidate_count: int
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, LocalizationStatus):
            raise LocalizationError(
                "status must be a LocalizationStatus."
            )

        if self.root_id is not None:
            object.__setattr__(
                self,
                "root_id",
                _normalize_node_id(self.root_id, name="root_id"),
            )

        object.__setattr__(
            self,
            "best_discrepancy",
            (
                None
                if self.best_discrepancy is None
                else _finite_float(
                    self.best_discrepancy,
                    name="best_discrepancy",
                    minimum=0.0,
                )
            ),
        )

        tied = tuple(
            _normalize_node_id(item, name="tied_root_id")
            for item in self.tied_root_ids
        )
        if len(tied) != len(set(tied)):
            raise LocalizationError(
                "tied_root_ids cannot contain duplicates."
            )
        object.__setattr__(self, "tied_root_ids", tied)

        scores = tuple(self.scores)
        if any(not isinstance(item, LocalizationScore) for item in scores):
            raise LocalizationError(
                "scores must contain LocalizationScore values."
            )
        object.__setattr__(self, "scores", scores)

        object.__setattr__(
            self,
            "observed_state",
            _state_vector(
                self.observed_state,
                name="observed_state",
            ),
        )

        if not isinstance(self.metric, LocalizationMetric):
            raise LocalizationError(
                "metric must be a LocalizationMetric."
            )

        for name in (
            "successful_candidate_count",
            "failed_candidate_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise LocalizationError(
                    f"{name} must be a non-negative integer."
                )

        if (
            self.successful_candidate_count
            + self.failed_candidate_count
            != len(scores)
        ):
            raise LocalizationError(
                "candidate counts must equal the number of scores."
            )

        if not isinstance(self.summary, str):
            raise LocalizationError("summary must be a string.")

        summary = self.summary.strip()
        if not summary:
            raise LocalizationError("summary cannot be empty.")

        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

        if self.status is LocalizationStatus.FAILED:
            if self.root_id is not None or self.best_discrepancy is not None:
                raise LocalizationError(
                    "FAILED localization cannot contain a root."
                )
        else:
            if self.root_id is None or self.best_discrepancy is None:
                raise LocalizationError(
                    "Successful localization requires a root."
                )

    @property
    def d_root(self) -> NodeId | None:
        return self.root_id

    @property
    def is_tied(self) -> bool:
        return self.status is LocalizationStatus.TIED

    @property
    def top_scores(self) -> tuple[LocalizationScore, ...]:
        return self.scores

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "root_id": self.root_id,
            "d_root": self.root_id,
            "best_discrepancy": self.best_discrepancy,
            "tied_root_ids": list(self.tied_root_ids),
            "scores": [item.as_dict() for item in self.scores],
            "observed_state": self.observed_state.tolist(),
            "metric": self.metric.value,
            "successful_candidate_count": self.successful_candidate_count,
            "failed_candidate_count": self.failed_candidate_count,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ForwardSimulator(Protocol):
    """Callable contract for hypothetical-origin forward simulation."""

    def __call__(
        self,
        candidate_id: NodeId,
    ) -> LocalizationSimulation | StateLike:
        ...


def _normalize_reserve_history(
    history: Mapping[
        NodeId,
        Sequence[tuple[float, float]],
    ]
    | None,
) -> Mapping[NodeId, tuple[tuple[float, float], ...]] | None:
    if history is None:
        return None

    if not isinstance(history, Mapping):
        raise LocalizationError(
            "reserve_history must be a mapping or None."
        )

    normalized: dict[NodeId, tuple[tuple[float, float], ...]] = {}

    for raw_node_id, raw_series in history.items():
        node_id = _normalize_node_id(
            raw_node_id,
            name="reserve_history node_id",
        )

        if isinstance(raw_series, (str, bytes)):
            raise LocalizationError(
                "reserve history series must contain (time, reserve) pairs."
            )

        try:
            series = tuple(raw_series)
        except TypeError as exc:
            raise LocalizationError(
                "reserve history series must be iterable."
            ) from exc

        parsed: list[tuple[float, float]] = []
        previous_time: float | None = None

        for item in series:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                raise LocalizationError(
                    "reserve history points must contain time and reserve."
                )

            time = _finite_float(
                item[0],
                name="reserve history time",
                minimum=0.0,
            )
            reserve = _finite_float(
                item[1],
                name="reserve history reserve",
            )

            if (
                previous_time is not None
                and time < previous_time - _EPSILON
            ):
                raise LocalizationError(
                    "reserve history must be chronological."
                )

            parsed.append((time, reserve))
            previous_time = time

        normalized[node_id] = tuple(parsed)

    return MappingProxyType(normalized)


def _built_in_distance(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    config: LocalizationConfig,
) -> float:
    difference = predicted - observed

    if config.metric is LocalizationMetric.L1:
        value = float(np.linalg.norm(difference, ord=1))
    elif config.metric is LocalizationMetric.L2:
        value = float(np.linalg.norm(difference, ord=2))
    elif config.metric is LocalizationMetric.LINF:
        value = float(np.linalg.norm(difference, ord=np.inf))
    elif config.metric is LocalizationMetric.WEIGHTED_L2:
        weights = np.asarray(config.weights, dtype=float)
        if weights.size != difference.size:
            raise LocalizationError(
                f"weights have size {weights.size}; "
                f"expected {difference.size}."
            )
        value = float(
            math.sqrt(
                float(np.dot(weights, difference * difference))
            )
        )
    else:
        raise LocalizationError(
            "CUSTOM metric requires a custom distance function."
        )

    if config.normalize_by_dimension:
        value /= math.sqrt(difference.size)

    return value


def localization_discrepancy(
    observed_state: StateLike,
    predicted_state: StateLike,
    *,
    config: LocalizationConfig | None = None,
    distance: DistanceFunction | None = None,
) -> float:
    """Compute the localization functional J_i."""

    if config is None:
        config = LocalizationConfig()

    if not isinstance(config, LocalizationConfig):
        raise LocalizationError(
            "config must be a LocalizationConfig."
        )

    observed = _state_vector(
        observed_state,
        name="observed_state",
    )
    predicted = _state_vector(
        predicted_state,
        name="predicted_state",
        expected_size=observed.size,
    )

    if config.metric is LocalizationMetric.CUSTOM:
        if distance is None or not callable(distance):
            raise LocalizationError(
                "A callable distance is required for CUSTOM metric."
            )
        value = _finite_float(
            distance(observed, predicted),
            name="custom discrepancy",
            minimum=0.0,
        )
        if config.normalize_by_dimension:
            value /= math.sqrt(observed.size)
        return value

    if distance is not None:
        raise LocalizationError(
            "distance can only be supplied with CUSTOM metric."
        )

    return _built_in_distance(
        observed,
        predicted,
        config=config,
    )


def estimate_d_fast(
    reserve_history: Mapping[
        NodeId,
        Sequence[tuple[float, float]],
    ],
    *,
    critical_reserve: float = 0.0,
    tie_tolerance: float = 1e-9,
    metadata: Mapping[str, Any] | None = None,
) -> FastFailureResult:
    """Estimate D_fast from first reserve-threshold crossing times."""

    normalized_history = _normalize_reserve_history(reserve_history)
    assert normalized_history is not None

    threshold = _finite_float(
        critical_reserve,
        name="critical_reserve",
    )
    tolerance = _finite_float(
        tie_tolerance,
        name="tie_tolerance",
        minimum=0.0,
    )

    crossing_times: dict[NodeId, float] = {}

    for node_id, series in normalized_history.items():
        for time, reserve in series:
            if reserve <= threshold:
                crossing_times[node_id] = time
                break

    if not crossing_times:
        return FastFailureResult(
            status=FastFailureStatus.NOT_REACHED,
            node_id=None,
            crossing_time=None,
            tied_node_ids=(),
            critical_reserve=threshold,
            crossing_times={},
            metadata=metadata,
        )

    earliest = min(crossing_times.values())
    tied = tuple(
        node_id
        for node_id, crossing_time in crossing_times.items()
        if abs(crossing_time - earliest) <= tolerance
    )

    status = (
        FastFailureStatus.TIED
        if len(tied) > 1
        else FastFailureStatus.IDENTIFIED
    )

    return FastFailureResult(
        status=status,
        node_id=tied[0],
        crossing_time=earliest,
        tied_node_ids=tied,
        critical_reserve=threshold,
        crossing_times=crossing_times,
        metadata=metadata,
    )


class RootLocalizationEngine:
    """Enumerate candidate origins and minimize J_i."""

    def __init__(
        self,
        config: LocalizationConfig | None = None,
        *,
        distance: DistanceFunction | None = None,
    ) -> None:
        if config is None:
            config = LocalizationConfig()

        if not isinstance(config, LocalizationConfig):
            raise LocalizationError(
                "config must be a LocalizationConfig."
            )

        if distance is not None and not callable(distance):
            raise LocalizationError(
                "distance must be callable or None."
            )

        if (
            config.metric is LocalizationMetric.CUSTOM
            and distance is None
        ):
            raise LocalizationError(
                "CUSTOM metric requires a distance function."
            )

        if (
            config.metric is not LocalizationMetric.CUSTOM
            and distance is not None
        ):
            raise LocalizationError(
                "distance is only valid with CUSTOM metric."
            )

        self._config = config
        self._distance = distance

    @property
    def config(self) -> LocalizationConfig:
        return self._config

    def localize(
        self,
        observed_state: StateLike,
        candidates: Iterable[NodeId],
        simulate: ForwardSimulator,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> RootLocalizationResult:
        if not callable(simulate):
            raise LocalizationError("simulate must be callable.")

        observed = _state_vector(
            observed_state,
            name="observed_state",
        )
        candidate_ids = _normalize_candidates(candidates)

        successful: list[
            tuple[NodeId, float, LocalizationSimulation]
        ] = []
        failures: list[
            tuple[NodeId, str, str]
        ] = []

        for candidate_id in candidate_ids:
            try:
                raw_result = simulate(candidate_id)

                if isinstance(raw_result, LocalizationSimulation):
                    simulation = raw_result
                    if simulation.candidate_id != candidate_id:
                        raise LocalizationError(
                            "Simulation candidate_id does not match "
                            "the requested candidate."
                        )
                else:
                    simulation = LocalizationSimulation(
                        candidate_id=candidate_id,
                        predicted_state=_state_vector(
                            raw_result,
                            name="simulated predicted_state",
                            expected_size=observed.size,
                        ),
                    )

                predicted = _state_vector(
                    simulation.predicted_state,
                    name="predicted_state",
                    expected_size=observed.size,
                )
                discrepancy = localization_discrepancy(
                    observed,
                    predicted,
                    config=self.config,
                    distance=self._distance,
                )
                successful.append(
                    (candidate_id, discrepancy, simulation)
                )

            except Exception as exc:
                if not self.config.capture_simulation_errors:
                    raise

                failures.append(
                    (
                        candidate_id,
                        type(exc).__name__,
                        str(exc) or type(exc).__name__,
                    )
                )

        if not successful:
            failed_scores = tuple(
                LocalizationScore(
                    candidate_id=candidate_id,
                    discrepancy=float(np.finfo(float).max),
                    rank=index,
                    tied_for_best=False,
                    simulation_status=SimulationStatus.FAILED,
                    converged=None,
                    predicted_state=None,
                    error_type=error_type,
                    error_message=error_message,
                )
                for index, (
                    candidate_id,
                    error_type,
                    error_message,
                ) in enumerate(failures, start=1)
            )

            # LocalizationScore requires finite discrepancy. Failed candidates
            # therefore cannot use infinity in the public score object.
            failed_scores = tuple(
                LocalizationScore(
                    candidate_id=item.candidate_id,
                    discrepancy=float(np.finfo(float).max),
                    rank=item.rank,
                    tied_for_best=False,
                    simulation_status=SimulationStatus.FAILED,
                    converged=None,
                    predicted_state=None,
                    error_type=item.error_type,
                    error_message=item.error_message,
                )
                for item in failed_scores
            )

            return RootLocalizationResult(
                status=LocalizationStatus.FAILED,
                root_id=None,
                best_discrepancy=None,
                tied_root_ids=(),
                scores=failed_scores,
                observed_state=observed,
                metric=self.config.metric,
                successful_candidate_count=0,
                failed_candidate_count=len(failures),
                summary=(
                    "Root localization failed because no candidate "
                    "simulation completed successfully."
                ),
                metadata=metadata,
            )

        successful.sort(
            key=lambda item: (
                item[1],
                str(item[0]),
            )
        )

        best = successful[0][1]
        tied_ids = tuple(
            candidate_id
            for candidate_id, discrepancy, _ in successful
            if abs(discrepancy - best) <= self.config.tie_tolerance
        )

        successful_scores: list[LocalizationScore] = []
        for rank, (
            candidate_id,
            discrepancy,
            simulation,
        ) in enumerate(successful, start=1):
            successful_scores.append(
                LocalizationScore(
                    candidate_id=candidate_id,
                    discrepancy=discrepancy,
                    rank=rank,
                    tied_for_best=candidate_id in tied_ids,
                    simulation_status=SimulationStatus.SUCCEEDED,
                    converged=simulation.converged,
                    predicted_state=simulation.predicted_state,
                    metadata={
                        "iterations": simulation.iterations,
                        "propagation_time": simulation.propagation_time,
                        **dict(simulation.metadata),
                    },
                )
            )

        failed_scores = tuple(
            LocalizationScore(
                candidate_id=candidate_id,
                discrepancy=float(np.finfo(float).max),
                rank=len(successful_scores) + index,
                tied_for_best=False,
                simulation_status=SimulationStatus.FAILED,
                converged=None,
                predicted_state=None,
                error_type=error_type,
                error_message=error_message,
            )
            for index, (
                candidate_id,
                error_type,
                error_message,
            ) in enumerate(failures, start=1)
        )

        all_scores = (
            tuple(successful_scores)
            + failed_scores
        )
        top_scores = all_scores[: self.config.top_k]

        if len(tied_ids) > 1:
            status = LocalizationStatus.TIED
        elif failures:
            status = LocalizationStatus.PARTIAL
        else:
            status = LocalizationStatus.LOCALIZED

        if failures and not self.config.allow_partial_results:
            return RootLocalizationResult(
                status=LocalizationStatus.FAILED,
                root_id=None,
                best_discrepancy=None,
                tied_root_ids=(),
                scores=all_scores,
                observed_state=observed,
                metric=self.config.metric,
                successful_candidate_count=len(successful),
                failed_candidate_count=len(failures),
                summary=(
                    "Root localization was rejected because at least one "
                    "candidate simulation failed and partial results are disabled."
                ),
                metadata=metadata,
            )

        root_id = tied_ids[0]

        return RootLocalizationResult(
            status=status,
            root_id=root_id,
            best_discrepancy=best,
            tied_root_ids=tied_ids,
            scores=all_scores,
            observed_state=observed,
            metric=self.config.metric,
            successful_candidate_count=len(successful),
            failed_candidate_count=len(failures),
            summary=_localization_summary(
                status=status,
                root_id=root_id,
                best_discrepancy=best,
                tied_ids=tied_ids,
                successful_count=len(successful),
                failed_count=len(failures),
            ),
            metadata=metadata,
        )


def _localization_summary(
    *,
    status: LocalizationStatus,
    root_id: NodeId,
    best_discrepancy: float,
    tied_ids: Sequence[NodeId],
    successful_count: int,
    failed_count: int,
) -> str:
    if status is LocalizationStatus.TIED:
        return (
            f"D_root is tied among {len(tied_ids)} candidates: "
            f"{', '.join(str(item) for item in tied_ids)}. "
            f"Best discrepancy J = {best_discrepancy:.6g}. "
            f"{successful_count} simulations succeeded; "
            f"{failed_count} failed."
        )

    return (
        f"D_root candidate: {root_id}. "
        f"Minimum localization discrepancy J = "
        f"{best_discrepancy:.6g}. "
        f"{successful_count} simulations succeeded; "
        f"{failed_count} failed."
    )


def localize_root(
    observed_state: StateLike,
    candidates: Iterable[NodeId],
    simulate: ForwardSimulator,
    *,
    config: LocalizationConfig | None = None,
    distance: DistanceFunction | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RootLocalizationResult:
    """Convenience wrapper around ``RootLocalizationEngine.localize``."""

    return RootLocalizationEngine(
        config,
        distance=distance,
    ).localize(
        observed_state,
        candidates,
        simulate,
        metadata=metadata,
    )


__all__ = [
    "DistanceFunction",
    "FastFailureResult",
    "FastFailureStatus",
    "ForwardSimulator",
    "LocalizationConfig",
    "LocalizationError",
    "LocalizationMetric",
    "LocalizationScore",
    "LocalizationSimulation",
    "LocalizationStatus",
    "NodeId",
    "RootLocalizationEngine",
    "RootLocalizationResult",
    "SimulationStatus",
    "StateLike",
    "estimate_d_fast",
    "localization_discrepancy",
    "localize_root",
]
