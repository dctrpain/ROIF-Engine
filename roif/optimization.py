"""
Counterfactual intervention optimization for ROIF Engine v1.

This module computes the operational intervention node Node* by evaluating
candidate interventions through caller-supplied counterfactual simulations.

For each candidate v_i, the optimizer estimates a recovery functional Γ(v_i).
The default recovery measure is the increase in global reserve:

    Γ(v_i) = R_global_after(v_i) - R_global_before

The candidate with maximal Γ is selected:

    Node* = argmax_i Γ(v_i)

The implementation is domain-independent. It does not mutate a network or
simulation directly. A caller supplies:

    simulate(candidate_id) -> InterventionSimulation | state vector | mapping

The optimizer supports:

- scalar or vector reserve states;
- sum, mean, minimum, weighted-sum, and custom global reserve aggregators;
- intervention cost, risk, delay, evidence, and reversibility penalties;
- deterministic ranking;
- ties;
- top-k reporting;
- partial simulation failures;
- optional Non-Fonit safety vetoes;
- immutable, serializable results.

This module is mathematically upstream of planner.py:

    optimization.py
        computes counterfactual recovery and Node*

    planner.py
        ranks practical intervention candidates using cost, risk,
        evidence, reversibility, urgency, and safety constraints.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Hashable, Protocol, TypeAlias, runtime_checkable
import math

import numpy as np


class OptimizationError(ValueError):
    """Raised when optimization input or configuration is invalid."""


NodeId: TypeAlias = Hashable
ReserveLike: TypeAlias = float | Sequence[float] | np.ndarray
ReserveAggregator: TypeAlias = Callable[[np.ndarray], float]
InterventionUtility: TypeAlias = Callable[
    ["InterventionSimulation", float, "OptimizationConfig"],
    float,
]

_EPSILON = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise OptimizationError(f"{name} must be a real number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OptimizationError(f"{name} must be a real number.") from exc

    if not math.isfinite(result):
        raise OptimizationError(f"{name} must be finite.")

    if minimum is not None and result < minimum:
        raise OptimizationError(
            f"{name} must be greater than or equal to {minimum}."
        )

    if maximum is not None and result > maximum:
        raise OptimizationError(
            f"{name} must be less than or equal to {maximum}."
        )

    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OptimizationError(
            f"{name} must be an integer greater than or equal to 1."
        )
    return value


def _normalize_node_id(value: Any, *, name: str = "node_id") -> NodeId:
    if value is None:
        raise OptimizationError(f"{name} cannot be None.")

    try:
        hash(value)
    except TypeError as exc:
        raise OptimizationError(f"{name} must be hashable.") from exc

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise OptimizationError(f"{name} cannot be empty.")
        return normalized

    return value


def _normalize_candidates(
    candidates: Iterable[NodeId],
) -> tuple[NodeId, ...]:
    if isinstance(candidates, (str, bytes)):
        raise OptimizationError(
            "candidates must be an iterable of node identifiers."
        )

    try:
        values = tuple(candidates)
    except TypeError as exc:
        raise OptimizationError(
            "candidates must be an iterable of node identifiers."
        ) from exc

    if not values:
        raise OptimizationError("candidates cannot be empty.")

    normalized: list[NodeId] = []
    seen: set[NodeId] = set()

    for value in values:
        node_id = _normalize_node_id(value, name="candidate node_id")
        if node_id in seen:
            raise OptimizationError(
                f"Duplicate candidate node_id: {node_id!r}."
            )
        seen.add(node_id)
        normalized.append(node_id)

    return tuple(normalized)


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise OptimizationError("metadata must be a mapping.")

    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise OptimizationError("metadata keys must be strings.")

        normalized = key.strip()
        if not normalized:
            raise OptimizationError("metadata keys cannot be empty.")

        copied[normalized] = value

    return MappingProxyType(copied)


def _reserve_vector(
    value: ReserveLike,
    *,
    name: str,
    expected_size: int | None = None,
) -> np.ndarray:
    if isinstance(value, bool):
        raise OptimizationError(
            f"{name} must be a scalar or one-dimensional reserve vector."
        )

    if isinstance(value, (int, float, np.number)):
        array = np.asarray((value,), dtype=float)
    else:
        if isinstance(value, (str, bytes)):
            raise OptimizationError(
                f"{name} must be a scalar or one-dimensional reserve vector."
            )
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise OptimizationError(
                f"{name} must be a scalar or one-dimensional reserve vector."
            ) from exc

    if array.ndim != 1:
        raise OptimizationError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise OptimizationError(f"{name} cannot be empty.")

    if expected_size is not None and array.size != expected_size:
        raise OptimizationError(
            f"{name} has size {array.size}; expected {expected_size}."
        )

    if not np.all(np.isfinite(array)):
        raise OptimizationError(
            f"{name} must contain only finite values."
        )

    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


class ReserveAggregation(str, Enum):
    """Built-in aggregation rules for global reserve."""

    SUM = "sum"
    MEAN = "mean"
    MINIMUM = "minimum"
    WEIGHTED_SUM = "weighted_sum"
    CUSTOM = "custom"


class OptimizationStatus(str, Enum):
    """Outcome of a counterfactual optimization pass."""

    OPTIMIZED = "optimized"
    TIED = "tied"
    PARTIAL = "partial"
    FAILED = "failed"
    ALL_VETOED = "all_vetoed"


class InterventionSimulationStatus(str, Enum):
    """Outcome of one counterfactual intervention simulation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    VETOED = "vetoed"


class OptimizationVetoReason(str, Enum):
    """Reason why a candidate was excluded before simulation."""

    CASCADE_RISK = "cascade_risk"
    EXTERNAL_SYSTEM_RISK = "external_system_risk"
    IRREVERSIBLE_HIGH_RISK = "irreversible_high_risk"
    LOW_EVIDENCE = "low_evidence"


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Configuration for Node* counterfactual optimization."""

    aggregation: ReserveAggregation = ReserveAggregation.SUM
    weights: tuple[float, ...] | None = None

    gain_weight: float = 1.0
    cost_weight: float = 0.0
    risk_weight: float = 0.0
    delay_weight: float = 0.0
    evidence_weight: float = 0.0
    reversibility_weight: float = 0.0

    tie_tolerance: float = 1e-9
    top_k: int = 5
    minimum_recovery_gain: float = 0.0

    allow_partial_results: bool = True
    capture_simulation_errors: bool = True

    non_fonit_gate_enabled: bool = True
    maximum_cascade_risk: float = 0.70
    maximum_external_system_risk: float = 0.20
    irreversible_risk_threshold: float = 0.35
    minimum_evidence_strength: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.aggregation, ReserveAggregation):
            raise OptimizationError(
                "aggregation must be a ReserveAggregation."
            )

        for name in (
            "gain_weight",
            "cost_weight",
            "risk_weight",
            "delay_weight",
            "evidence_weight",
            "reversibility_weight",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                ),
            )

        if self.weight_sum <= _EPSILON:
            raise OptimizationError(
                "At least one optimization weight must be positive."
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
            "top_k",
            _positive_int(self.top_k, name="top_k"),
        )
        object.__setattr__(
            self,
            "minimum_recovery_gain",
            _finite_float(
                self.minimum_recovery_gain,
                name="minimum_recovery_gain",
            ),
        )

        for name in (
            "allow_partial_results",
            "capture_simulation_errors",
            "non_fonit_gate_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise OptimizationError(f"{name} must be a bool.")

        for name in (
            "maximum_cascade_risk",
            "maximum_external_system_risk",
            "irreversible_risk_threshold",
            "minimum_evidence_strength",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

        if self.weights is not None:
            weights = _reserve_vector(
                self.weights,
                name="weights",
            )
            if np.any(weights < 0.0):
                raise OptimizationError(
                    "weights cannot contain negative values."
                )
            if not np.any(weights > 0.0):
                raise OptimizationError(
                    "weights must contain at least one positive value."
                )
            object.__setattr__(
                self,
                "weights",
                tuple(float(item) for item in weights),
            )

        if (
            self.aggregation is ReserveAggregation.WEIGHTED_SUM
            and self.weights is None
        ):
            raise OptimizationError(
                "weights are required for WEIGHTED_SUM."
            )

        if (
            self.aggregation is not ReserveAggregation.WEIGHTED_SUM
            and self.weights is not None
        ):
            raise OptimizationError(
                "weights are only valid for WEIGHTED_SUM."
            )

    @property
    def weight_sum(self) -> float:
        return (
            self.gain_weight
            + self.cost_weight
            + self.risk_weight
            + self.delay_weight
            + self.evidence_weight
            + self.reversibility_weight
        )


@dataclass(frozen=True, slots=True)
class InterventionSimulation:
    """Counterfactual state after intervening on one candidate node."""

    candidate_id: NodeId
    reserve_after: np.ndarray
    cost: float = 0.0
    risk: float = 0.0
    delay: float = 0.0
    evidence_strength: float = 0.0
    reversibility: float = 1.0
    cascade_risk: float = 0.0
    external_system_risk: float = 0.0
    converged: bool = True
    iterations: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _normalize_node_id(
                self.candidate_id,
                name="candidate_id",
            ),
        )
        object.__setattr__(
            self,
            "reserve_after",
            _reserve_vector(
                self.reserve_after,
                name="reserve_after",
            ),
        )

        for name in ("cost", "delay"):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                ),
            )

        for name in (
            "risk",
            "evidence_strength",
            "reversibility",
            "cascade_risk",
            "external_system_risk",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

        if not isinstance(self.converged, bool):
            raise OptimizationError("converged must be a bool.")

        if self.iterations is not None:
            if (
                isinstance(self.iterations, bool)
                or not isinstance(self.iterations, int)
                or self.iterations < 0
            ):
                raise OptimizationError(
                    "iterations must be a non-negative integer or None."
                )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reserve_after": self.reserve_after.tolist(),
            "cost": self.cost,
            "risk": self.risk,
            "delay": self.delay,
            "evidence_strength": self.evidence_strength,
            "reversibility": self.reversibility,
            "cascade_risk": self.cascade_risk,
            "external_system_risk": self.external_system_risk,
            "converged": self.converged,
            "iterations": self.iterations,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class InterventionScore:
    """Counterfactual recovery score for one candidate."""

    candidate_id: NodeId
    rank: int
    status: InterventionSimulationStatus

    global_reserve_before: float
    global_reserve_after: float | None
    recovery_gain: float | None
    utility: float | None

    tied_for_best: bool
    converged: bool | None

    cost: float | None
    risk: float | None
    delay: float | None
    evidence_strength: float | None
    reversibility: float | None

    veto_reasons: tuple[OptimizationVetoReason, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _normalize_node_id(
                self.candidate_id,
                name="candidate_id",
            ),
        )
        object.__setattr__(
            self,
            "rank",
            _positive_int(self.rank, name="rank"),
        )

        if not isinstance(self.status, InterventionSimulationStatus):
            raise OptimizationError(
                "status must be an InterventionSimulationStatus."
            )

        object.__setattr__(
            self,
            "global_reserve_before",
            _finite_float(
                self.global_reserve_before,
                name="global_reserve_before",
            ),
        )

        for name in (
            "global_reserve_after",
            "recovery_gain",
            "utility",
            "cost",
            "risk",
            "delay",
            "evidence_strength",
            "reversibility",
        ):
            value = getattr(self, name)
            if value is not None:
                kwargs: dict[str, Any] = {}
                if name in ("cost", "delay"):
                    kwargs["minimum"] = 0.0
                if name in (
                    "risk",
                    "evidence_strength",
                    "reversibility",
                ):
                    kwargs["minimum"] = 0.0
                    kwargs["maximum"] = 1.0
                object.__setattr__(
                    self,
                    name,
                    _finite_float(
                        value,
                        name=name,
                        **kwargs,
                    ),
                )

        if not isinstance(self.tied_for_best, bool):
            raise OptimizationError(
                "tied_for_best must be a bool."
            )

        if self.converged is not None and not isinstance(
            self.converged,
            bool,
        ):
            raise OptimizationError(
                "converged must be a bool or None."
            )

        reasons = tuple(self.veto_reasons)
        if any(
            not isinstance(item, OptimizationVetoReason)
            for item in reasons
        ):
            raise OptimizationError(
                "veto_reasons must contain OptimizationVetoReason values."
            )
        if len(reasons) != len(set(reasons)):
            raise OptimizationError(
                "veto_reasons cannot contain duplicates."
            )
        object.__setattr__(self, "veto_reasons", reasons)

        for name in ("error_type", "error_message"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str):
                    raise OptimizationError(
                        f"{name} must be a string or None."
                    )
                normalized = value.strip()
                if not normalized:
                    raise OptimizationError(
                        f"{name} cannot be empty."
                    )
                object.__setattr__(self, name, normalized)

        if self.status is InterventionSimulationStatus.SUCCEEDED:
            required = (
                self.global_reserve_after,
                self.recovery_gain,
                self.utility,
                self.cost,
                self.risk,
                self.delay,
                self.evidence_strength,
                self.reversibility,
            )
            if any(item is None for item in required):
                raise OptimizationError(
                    "Successful intervention score requires all numeric fields."
                )
            if reasons:
                raise OptimizationError(
                    "Successful intervention score cannot contain veto reasons."
                )
            if self.error_type is not None or self.error_message is not None:
                raise OptimizationError(
                    "Successful intervention score cannot contain errors."
                )

        elif self.status is InterventionSimulationStatus.VETOED:
            if not reasons:
                raise OptimizationError(
                    "Vetoed intervention score requires veto reasons."
                )
            if self.utility is not None:
                raise OptimizationError(
                    "Vetoed intervention score cannot contain utility."
                )
            if self.error_type is not None or self.error_message is not None:
                raise OptimizationError(
                    "Vetoed intervention score cannot contain errors."
                )

        elif self.status is InterventionSimulationStatus.FAILED:
            if self.error_type is None or self.error_message is None:
                raise OptimizationError(
                    "Failed intervention score requires error details."
                )
            if reasons:
                raise OptimizationError(
                    "Failed intervention score cannot contain veto reasons."
                )
            if self.utility is not None:
                raise OptimizationError(
                    "Failed intervention score cannot contain utility."
                )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "status": self.status.value,
            "global_reserve_before": self.global_reserve_before,
            "global_reserve_after": self.global_reserve_after,
            "recovery_gain": self.recovery_gain,
            "utility": self.utility,
            "tied_for_best": self.tied_for_best,
            "converged": self.converged,
            "cost": self.cost,
            "risk": self.risk,
            "delay": self.delay,
            "evidence_strength": self.evidence_strength,
            "reversibility": self.reversibility,
            "veto_reasons": [
                item.value for item in self.veto_reasons
            ],
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Immutable Node* optimization result."""

    status: OptimizationStatus
    node_star: NodeId | None
    best_utility: float | None
    best_recovery_gain: float | None
    tied_node_ids: tuple[NodeId, ...]

    scores: tuple[InterventionScore, ...]
    aggregation: ReserveAggregation
    global_reserve_before: float

    successful_candidate_count: int
    failed_candidate_count: int
    vetoed_candidate_count: int

    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, OptimizationStatus):
            raise OptimizationError(
                "status must be an OptimizationStatus."
            )

        if self.node_star is not None:
            object.__setattr__(
                self,
                "node_star",
                _normalize_node_id(
                    self.node_star,
                    name="node_star",
                ),
            )

        for name in ("best_utility", "best_recovery_gain"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_float(value, name=name),
                )

        tied = tuple(
            _normalize_node_id(item, name="tied_node_id")
            for item in self.tied_node_ids
        )
        if len(tied) != len(set(tied)):
            raise OptimizationError(
                "tied_node_ids cannot contain duplicates."
            )
        object.__setattr__(self, "tied_node_ids", tied)

        scores = tuple(self.scores)
        if any(
            not isinstance(item, InterventionScore)
            for item in scores
        ):
            raise OptimizationError(
                "scores must contain InterventionScore values."
            )
        object.__setattr__(self, "scores", scores)

        if not isinstance(self.aggregation, ReserveAggregation):
            raise OptimizationError(
                "aggregation must be a ReserveAggregation."
            )

        object.__setattr__(
            self,
            "global_reserve_before",
            _finite_float(
                self.global_reserve_before,
                name="global_reserve_before",
            ),
        )

        for name in (
            "successful_candidate_count",
            "failed_candidate_count",
            "vetoed_candidate_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise OptimizationError(
                    f"{name} must be a non-negative integer."
                )

        if (
            self.successful_candidate_count
            + self.failed_candidate_count
            + self.vetoed_candidate_count
            != len(scores)
        ):
            raise OptimizationError(
                "candidate counts must equal the number of scores."
            )

        if not isinstance(self.summary, str):
            raise OptimizationError("summary must be a string.")

        summary = self.summary.strip()
        if not summary:
            raise OptimizationError("summary cannot be empty.")
        object.__setattr__(self, "summary", summary)

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

        if self.status in (
            OptimizationStatus.FAILED,
            OptimizationStatus.ALL_VETOED,
        ):
            if (
                self.node_star is not None
                or self.best_utility is not None
                or self.best_recovery_gain is not None
            ):
                raise OptimizationError(
                    "Failed optimization cannot contain Node*."
                )
        else:
            if (
                self.node_star is None
                or self.best_utility is None
                or self.best_recovery_gain is None
            ):
                raise OptimizationError(
                    "Successful optimization requires Node* and best values."
                )

    @property
    def is_tied(self) -> bool:
        return self.status is OptimizationStatus.TIED

    @property
    def top_scores(self) -> tuple[InterventionScore, ...]:
        return self.scores

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "node_star": self.node_star,
            "best_utility": self.best_utility,
            "best_recovery_gain": self.best_recovery_gain,
            "tied_node_ids": list(self.tied_node_ids),
            "scores": [item.as_dict() for item in self.scores],
            "aggregation": self.aggregation.value,
            "global_reserve_before": self.global_reserve_before,
            "successful_candidate_count": self.successful_candidate_count,
            "failed_candidate_count": self.failed_candidate_count,
            "vetoed_candidate_count": self.vetoed_candidate_count,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class InterventionSimulator(Protocol):
    """Callable contract for counterfactual intervention simulation."""

    def __call__(
        self,
        candidate_id: NodeId,
    ) -> InterventionSimulation | ReserveLike | Mapping[str, Any]:
        ...


def aggregate_reserve(
    reserve: ReserveLike,
    *,
    config: OptimizationConfig | None = None,
    aggregator: ReserveAggregator | None = None,
) -> float:
    """Aggregate a reserve vector into one global reserve value."""

    if config is None:
        config = OptimizationConfig()

    if not isinstance(config, OptimizationConfig):
        raise OptimizationError(
            "config must be an OptimizationConfig."
        )

    vector = _reserve_vector(
        reserve,
        name="reserve",
    )

    if config.aggregation is ReserveAggregation.CUSTOM:
        if aggregator is None or not callable(aggregator):
            raise OptimizationError(
                "A callable aggregator is required for CUSTOM aggregation."
            )
        return _finite_float(
            aggregator(vector),
            name="custom global reserve",
        )

    if aggregator is not None:
        raise OptimizationError(
            "aggregator can only be supplied with CUSTOM aggregation."
        )

    if config.aggregation is ReserveAggregation.SUM:
        return float(np.sum(vector))

    if config.aggregation is ReserveAggregation.MEAN:
        return float(np.mean(vector))

    if config.aggregation is ReserveAggregation.MINIMUM:
        return float(np.min(vector))

    if config.aggregation is ReserveAggregation.WEIGHTED_SUM:
        weights = np.asarray(config.weights, dtype=float)
        if weights.size != vector.size:
            raise OptimizationError(
                f"weights have size {weights.size}; "
                f"expected {vector.size}."
            )
        return float(np.dot(weights, vector))

    raise OptimizationError("Unsupported reserve aggregation.")


def _coerce_simulation(
    candidate_id: NodeId,
    value: InterventionSimulation | ReserveLike | Mapping[str, Any],
    *,
    expected_size: int,
) -> InterventionSimulation:
    if isinstance(value, InterventionSimulation):
        if value.candidate_id != candidate_id:
            raise OptimizationError(
                "Simulation candidate_id does not match requested candidate."
            )
        if value.reserve_after.size != expected_size:
            raise OptimizationError(
                f"reserve_after has size {value.reserve_after.size}; "
                f"expected {expected_size}."
            )
        return value

    if isinstance(value, Mapping):
        allowed = {
            "reserve_after",
            "cost",
            "risk",
            "delay",
            "evidence_strength",
            "reversibility",
            "cascade_risk",
            "external_system_risk",
            "converged",
            "iterations",
            "metadata",
        }
        unknown = set(value) - allowed
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise OptimizationError(
                f"Intervention mapping contains unknown fields: {names}."
            )
        if "reserve_after" not in value:
            raise OptimizationError(
                "Intervention mapping requires reserve_after."
            )
        payload = dict(value)
        return InterventionSimulation(
            candidate_id=candidate_id,
            **payload,
        )

    simulation = InterventionSimulation(
        candidate_id=candidate_id,
        reserve_after=_reserve_vector(
            value,
            name="simulated reserve_after",
            expected_size=expected_size,
        ),
    )
    return simulation


def _veto_reasons(
    simulation: InterventionSimulation,
    config: OptimizationConfig,
) -> tuple[OptimizationVetoReason, ...]:
    if not config.non_fonit_gate_enabled:
        return ()

    reasons: list[OptimizationVetoReason] = []

    if simulation.cascade_risk > config.maximum_cascade_risk:
        reasons.append(OptimizationVetoReason.CASCADE_RISK)

    if (
        simulation.external_system_risk
        > config.maximum_external_system_risk
    ):
        reasons.append(
            OptimizationVetoReason.EXTERNAL_SYSTEM_RISK
        )

    if (
        simulation.reversibility < 0.5
        and simulation.risk
        > config.irreversible_risk_threshold
    ):
        reasons.append(
            OptimizationVetoReason.IRREVERSIBLE_HIGH_RISK
        )

    if (
        simulation.evidence_strength
        < config.minimum_evidence_strength
    ):
        reasons.append(OptimizationVetoReason.LOW_EVIDENCE)

    return tuple(reasons)


def _normalize_penalties(
    simulations: Sequence[
        tuple[NodeId, InterventionSimulation, float]
    ],
) -> dict[NodeId, dict[str, float]]:
    max_cost = max((item.cost for _, item, _ in simulations), default=0.0)
    max_delay = max((item.delay for _, item, _ in simulations), default=0.0)

    return {
        candidate_id: {
            "cost": (
                simulation.cost / max_cost
                if max_cost > _EPSILON
                else 0.0
            ),
            "delay": (
                simulation.delay / max_delay
                if max_delay > _EPSILON
                else 0.0
            ),
        }
        for candidate_id, simulation, _ in simulations
    }


def _default_utility(
    simulation: InterventionSimulation,
    recovery_gain: float,
    config: OptimizationConfig,
    *,
    normalized_cost: float,
    normalized_delay: float,
) -> float:
    return (
        config.gain_weight * recovery_gain
        - config.cost_weight * normalized_cost
        - config.risk_weight * simulation.risk
        - config.delay_weight * normalized_delay
        + config.evidence_weight * simulation.evidence_strength
        + config.reversibility_weight * simulation.reversibility
    ) / config.weight_sum


class InterventionOptimizer:
    """Evaluate candidate interventions and select Node*."""

    def __init__(
        self,
        config: OptimizationConfig | None = None,
        *,
        aggregator: ReserveAggregator | None = None,
        utility: InterventionUtility | None = None,
    ) -> None:
        if config is None:
            config = OptimizationConfig()

        if not isinstance(config, OptimizationConfig):
            raise OptimizationError(
                "config must be an OptimizationConfig."
            )

        if aggregator is not None and not callable(aggregator):
            raise OptimizationError(
                "aggregator must be callable or None."
            )

        if utility is not None and not callable(utility):
            raise OptimizationError(
                "utility must be callable or None."
            )

        if (
            config.aggregation is ReserveAggregation.CUSTOM
            and aggregator is None
        ):
            raise OptimizationError(
                "CUSTOM aggregation requires an aggregator."
            )

        if (
            config.aggregation is not ReserveAggregation.CUSTOM
            and aggregator is not None
        ):
            raise OptimizationError(
                "aggregator is only valid with CUSTOM aggregation."
            )

        self._config = config
        self._aggregator = aggregator
        self._utility = utility

    @property
    def config(self) -> OptimizationConfig:
        return self._config

    def optimize(
        self,
        reserve_before: ReserveLike,
        candidates: Iterable[NodeId],
        simulate: InterventionSimulator,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        if not callable(simulate):
            raise OptimizationError("simulate must be callable.")

        reserve_vector = _reserve_vector(
            reserve_before,
            name="reserve_before",
        )
        candidate_ids = _normalize_candidates(candidates)

        global_before = aggregate_reserve(
            reserve_vector,
            config=self.config,
            aggregator=self._aggregator,
        )

        successful_raw: list[
            tuple[NodeId, InterventionSimulation, float]
        ] = []
        failures: list[tuple[NodeId, str, str]] = []
        vetoed_raw: list[
            tuple[
                NodeId,
                InterventionSimulation,
                float,
                tuple[OptimizationVetoReason, ...],
            ]
        ] = []

        for candidate_id in candidate_ids:
            try:
                raw = simulate(candidate_id)
                simulation = _coerce_simulation(
                    candidate_id,
                    raw,
                    expected_size=reserve_vector.size,
                )

                global_after = aggregate_reserve(
                    simulation.reserve_after,
                    config=self.config,
                    aggregator=self._aggregator,
                )
                gain = global_after - global_before
                reasons = _veto_reasons(
                    simulation,
                    self.config,
                )

                if reasons:
                    vetoed_raw.append(
                        (
                            candidate_id,
                            simulation,
                            global_after,
                            reasons,
                        )
                    )
                else:
                    successful_raw.append(
                        (
                            candidate_id,
                            simulation,
                            global_after,
                        )
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

        if not successful_raw:
            status = (
                OptimizationStatus.ALL_VETOED
                if vetoed_raw and not failures
                else OptimizationStatus.FAILED
            )

            scores = self._build_non_success_scores(
                global_before=global_before,
                failures=failures,
                vetoed=vetoed_raw,
            )

            return OptimizationResult(
                status=status,
                node_star=None,
                best_utility=None,
                best_recovery_gain=None,
                tied_node_ids=(),
                scores=scores,
                aggregation=self.config.aggregation,
                global_reserve_before=global_before,
                successful_candidate_count=0,
                failed_candidate_count=len(failures),
                vetoed_candidate_count=len(vetoed_raw),
                summary=(
                    "Intervention optimization failed because no candidate "
                    "passed simulation and safety evaluation."
                    if status is OptimizationStatus.FAILED
                    else "All intervention candidates were vetoed by safety rules."
                ),
                metadata=metadata,
            )

        penalties = _normalize_penalties(successful_raw)
        scored_successes: list[
            tuple[
                NodeId,
                InterventionSimulation,
                float,
                float,
                float,
            ]
        ] = []

        for candidate_id, simulation, global_after in successful_raw:
            gain = global_after - global_before

            if self._utility is None:
                utility = _default_utility(
                    simulation,
                    gain,
                    self.config,
                    normalized_cost=penalties[candidate_id]["cost"],
                    normalized_delay=penalties[candidate_id]["delay"],
                )
            else:
                utility = _finite_float(
                    self._utility(
                        simulation,
                        gain,
                        self.config,
                    ),
                    name="custom intervention utility",
                )

            scored_successes.append(
                (
                    candidate_id,
                    simulation,
                    global_after,
                    gain,
                    utility,
                )
            )

        scored_successes.sort(
            key=lambda item: (
                -item[4],
                -item[3],
                item[1].risk,
                item[1].cost,
                str(item[0]),
            )
        )

        best_utility = scored_successes[0][4]
        tied_ids = tuple(
            candidate_id
            for candidate_id, _, _, _, utility in scored_successes
            if abs(utility - best_utility) <= self.config.tie_tolerance
        )

        successful_scores: list[InterventionScore] = []
        for rank, (
            candidate_id,
            simulation,
            global_after,
            gain,
            utility,
        ) in enumerate(scored_successes, start=1):
            successful_scores.append(
                InterventionScore(
                    candidate_id=candidate_id,
                    rank=rank,
                    status=InterventionSimulationStatus.SUCCEEDED,
                    global_reserve_before=global_before,
                    global_reserve_after=global_after,
                    recovery_gain=gain,
                    utility=utility,
                    tied_for_best=candidate_id in tied_ids,
                    converged=simulation.converged,
                    cost=simulation.cost,
                    risk=simulation.risk,
                    delay=simulation.delay,
                    evidence_strength=simulation.evidence_strength,
                    reversibility=simulation.reversibility,
                    metadata={
                        "iterations": simulation.iterations,
                        **dict(simulation.metadata),
                    },
                )
            )

        non_success_scores = self._build_non_success_scores(
            global_before=global_before,
            failures=failures,
            vetoed=vetoed_raw,
            rank_offset=len(successful_scores),
        )

        all_scores = (
            tuple(successful_scores)
            + non_success_scores
        )

        if failures and not self.config.allow_partial_results:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                node_star=None,
                best_utility=None,
                best_recovery_gain=None,
                tied_node_ids=(),
                scores=all_scores,
                aggregation=self.config.aggregation,
                global_reserve_before=global_before,
                successful_candidate_count=len(successful_scores),
                failed_candidate_count=len(failures),
                vetoed_candidate_count=len(vetoed_raw),
                summary=(
                    "Intervention optimization was rejected because at least "
                    "one candidate simulation failed and partial results are disabled."
                ),
                metadata=metadata,
            )

        best_candidate_id, _, _, best_gain, _ = scored_successes[0]

        if best_gain < self.config.minimum_recovery_gain:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                node_star=None,
                best_utility=None,
                best_recovery_gain=None,
                tied_node_ids=(),
                scores=all_scores,
                aggregation=self.config.aggregation,
                global_reserve_before=global_before,
                successful_candidate_count=len(successful_scores),
                failed_candidate_count=len(failures),
                vetoed_candidate_count=len(vetoed_raw),
                summary=(
                    "No candidate reached the configured minimum recovery gain."
                ),
                metadata=metadata,
            )

        if len(tied_ids) > 1:
            status = OptimizationStatus.TIED
        elif failures or vetoed_raw:
            status = OptimizationStatus.PARTIAL
        else:
            status = OptimizationStatus.OPTIMIZED

        return OptimizationResult(
            status=status,
            node_star=best_candidate_id,
            best_utility=best_utility,
            best_recovery_gain=best_gain,
            tied_node_ids=tied_ids,
            scores=all_scores,
            aggregation=self.config.aggregation,
            global_reserve_before=global_before,
            successful_candidate_count=len(successful_scores),
            failed_candidate_count=len(failures),
            vetoed_candidate_count=len(vetoed_raw),
            summary=_optimization_summary(
                status=status,
                node_star=best_candidate_id,
                best_utility=best_utility,
                best_gain=best_gain,
                tied_ids=tied_ids,
                successful_count=len(successful_scores),
                failed_count=len(failures),
                vetoed_count=len(vetoed_raw),
            ),
            metadata=metadata,
        )

    def _build_non_success_scores(
        self,
        *,
        global_before: float,
        failures: Sequence[tuple[NodeId, str, str]],
        vetoed: Sequence[
            tuple[
                NodeId,
                InterventionSimulation,
                float,
                tuple[OptimizationVetoReason, ...],
            ]
        ],
        rank_offset: int = 0,
    ) -> tuple[InterventionScore, ...]:
        result: list[InterventionScore] = []
        rank = rank_offset

        for (
            candidate_id,
            simulation,
            global_after,
            reasons,
        ) in vetoed:
            rank += 1
            result.append(
                InterventionScore(
                    candidate_id=candidate_id,
                    rank=rank,
                    status=InterventionSimulationStatus.VETOED,
                    global_reserve_before=global_before,
                    global_reserve_after=global_after,
                    recovery_gain=global_after - global_before,
                    utility=None,
                    tied_for_best=False,
                    converged=simulation.converged,
                    cost=simulation.cost,
                    risk=simulation.risk,
                    delay=simulation.delay,
                    evidence_strength=simulation.evidence_strength,
                    reversibility=simulation.reversibility,
                    veto_reasons=reasons,
                    metadata={
                        "iterations": simulation.iterations,
                        **dict(simulation.metadata),
                    },
                )
            )

        for candidate_id, error_type, error_message in failures:
            rank += 1
            result.append(
                InterventionScore(
                    candidate_id=candidate_id,
                    rank=rank,
                    status=InterventionSimulationStatus.FAILED,
                    global_reserve_before=global_before,
                    global_reserve_after=None,
                    recovery_gain=None,
                    utility=None,
                    tied_for_best=False,
                    converged=None,
                    cost=None,
                    risk=None,
                    delay=None,
                    evidence_strength=None,
                    reversibility=None,
                    error_type=error_type,
                    error_message=error_message,
                )
            )

        return tuple(result)


def _optimization_summary(
    *,
    status: OptimizationStatus,
    node_star: NodeId,
    best_utility: float,
    best_gain: float,
    tied_ids: Sequence[NodeId],
    successful_count: int,
    failed_count: int,
    vetoed_count: int,
) -> str:
    if status is OptimizationStatus.TIED:
        return (
            f"Node* is tied among {len(tied_ids)} candidates: "
            f"{', '.join(str(item) for item in tied_ids)}. "
            f"Best utility = {best_utility:.6g}; "
            f"recovery gain Γ = {best_gain:.6g}. "
            f"{successful_count} candidates succeeded, "
            f"{failed_count} failed, and {vetoed_count} were vetoed."
        )

    return (
        f"Node*: {node_star}. "
        f"Best utility = {best_utility:.6g}; "
        f"recovery gain Γ = {best_gain:.6g}. "
        f"{successful_count} candidates succeeded, "
        f"{failed_count} failed, and {vetoed_count} were vetoed."
    )


def optimize_interventions(
    reserve_before: ReserveLike,
    candidates: Iterable[NodeId],
    simulate: InterventionSimulator,
    *,
    config: OptimizationConfig | None = None,
    aggregator: ReserveAggregator | None = None,
    utility: InterventionUtility | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizationResult:
    """Convenience wrapper around ``InterventionOptimizer.optimize``."""

    return InterventionOptimizer(
        config,
        aggregator=aggregator,
        utility=utility,
    ).optimize(
        reserve_before,
        candidates,
        simulate,
        metadata=metadata,
    )


__all__ = [
    "InterventionOptimizer",
    "InterventionScore",
    "InterventionSimulation",
    "InterventionSimulationStatus",
    "InterventionSimulator",
    "InterventionUtility",
    "NodeId",
    "OptimizationConfig",
    "OptimizationError",
    "OptimizationResult",
    "OptimizationStatus",
    "OptimizationVetoReason",
    "ReserveAggregation",
    "ReserveAggregator",
    "ReserveLike",
    "aggregate_reserve",
    "optimize_interventions",
]
