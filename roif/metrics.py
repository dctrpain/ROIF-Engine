"""
ROIF metrics and cascade diagnostics.

This module provides immutable metric objects and deterministic helpers for
computing:

* spectral coherence ``eta``;
* ``D_fast`` вЂ” the plane with the smallest remaining reserve;
* ``D_root`` вЂ” the plane with the largest destabilizing sensitivity;
* ``Node*`` вЂ” the best counterfactual intervention candidate;
* aggregate metrics for one ``StateSnapshot``;
* aggregate metrics for a complete ``SimulationResult``.

The implementation deliberately separates raw mathematical inputs from ROIF
interpretation. Functions accept explicit arrays or immutable snapshots and
perform strict validation before calculating any metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from .cascade_event import CascadeEventSeverity
from .history import CascadeHistory
from .state_snapshot import StateSnapshot


class MetricsError(ValueError):
    """Raised when metric inputs are invalid or internally inconsistent."""


_FLOAT_TOLERANCE = 1e-12


def _as_finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MetricsError(f"{name} must contain real numbers.") from exc

    if vector.ndim != 1:
        raise MetricsError(f"{name} must be one-dimensional.")

    if not allow_empty and vector.size == 0:
        raise MetricsError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(vector)):
        raise MetricsError(f"{name} must contain only finite values.")

    return vector.copy()


def _as_square_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MetricsError(f"{name} must contain real numbers.") from exc

    if matrix.ndim != 2:
        raise MetricsError(f"{name} must be two-dimensional.")

    rows, columns = matrix.shape

    if rows != columns:
        raise MetricsError(f"{name} must be square.")

    if not allow_empty and rows == 0:
        raise MetricsError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(matrix)):
        raise MetricsError(f"{name} must contain only finite values.")

    return matrix.copy()


def _normalize_plane_ids(
    plane_ids: Sequence[str],
    *,
    expected_size: int,
) -> tuple[str, ...]:
    if isinstance(plane_ids, (str, bytes)):
        raise MetricsError("plane_ids must be a sequence of strings.")

    try:
        normalized = tuple(
            value.strip() if isinstance(value, str) else value
            for value in plane_ids
        )
    except TypeError as exc:
        raise MetricsError(
            "plane_ids must be a sequence of strings."
        ) from exc

    if len(normalized) != expected_size:
        raise MetricsError(
            "plane_ids length must match the metric dimension."
        )

    if any(not isinstance(value, str) for value in normalized):
        raise MetricsError("Every plane_id must be a string.")

    if any(not value for value in normalized):
        raise MetricsError("plane_id cannot be empty.")

    if len(set(normalized)) != len(normalized):
        raise MetricsError("plane_ids must be unique.")

    return normalized


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise MetricsError("metadata must be a mapping.")

    copied: dict[str, Any] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise MetricsError("metadata key must be a string.")

        normalized = key.strip()

        if not normalized:
            raise MetricsError("metadata key cannot be empty.")

        copied[normalized] = value

    return MappingProxyType(copied)


def spectral_radius(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """
    Return the spectral radius of a real square matrix.

    Complex eigenvalues are supported; their absolute magnitudes are used.
    """

    array = _as_square_matrix(matrix, name="matrix")

    try:
        eigenvalues = np.linalg.eigvals(array)
    except np.linalg.LinAlgError as exc:
        raise MetricsError(
            "Cannot compute eigenvalues for matrix."
        ) from exc

    radius = float(np.max(np.abs(eigenvalues)))

    if not math.isfinite(radius):
        raise MetricsError("Computed spectral radius is not finite.")

    return radius


def spectral_coherence(
    operator: Sequence[Sequence[float]] | np.ndarray,
    *,
    lambda_critical: float,
    clamp: bool = False,
) -> float:
    """
    Compute ROIF spectral coherence.

    ``eta = 1 - spectral_radius(operator) / lambda_critical``

    Positive values indicate reserve below the critical spectral boundary.
    Zero marks the boundary. Negative values indicate supercritical dynamics.
    """

    if isinstance(lambda_critical, bool):
        raise MetricsError(
            "lambda_critical must be a positive real number."
        )

    try:
        critical = float(lambda_critical)
    except (TypeError, ValueError) as exc:
        raise MetricsError(
            "lambda_critical must be a positive real number."
        ) from exc

    if not math.isfinite(critical) or critical <= 0.0:
        raise MetricsError(
            "lambda_critical must be finite and greater than 0.0."
        )

    if not isinstance(clamp, bool):
        raise MetricsError("clamp must be a bool.")

    eta = 1.0 - spectral_radius(operator) / critical

    if clamp:
        return min(1.0, max(-1.0, eta))

    return eta


@dataclass(frozen=True, slots=True)
class RankedPlane:
    """One ranked plane and its scalar score."""

    plane_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.plane_id, str):
            raise MetricsError("plane_id must be a string.")

        plane_id = self.plane_id.strip()

        if not plane_id:
            raise MetricsError("plane_id cannot be empty.")

        try:
            score = float(self.score)
        except (TypeError, ValueError) as exc:
            raise MetricsError("score must be a real number.") from exc

        if not math.isfinite(score):
            raise MetricsError("score must be finite.")

        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise MetricsError(
                "rank must be an integer greater than or equal to 1."
            )

        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "score", score)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plane_id": self.plane_id,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class PlaneRanking:
    """Immutable ordered plane ranking."""

    metric_name: str
    entries: tuple[RankedPlane, ...]
    higher_is_more_important: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str):
            raise MetricsError("metric_name must be a string.")

        metric_name = self.metric_name.strip()

        if not metric_name:
            raise MetricsError("metric_name cannot be empty.")

        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))

        if any(
            not isinstance(entry, RankedPlane)
            for entry in self.entries
        ):
            raise MetricsError(
                "entries must contain only RankedPlane objects."
            )

        ids = tuple(entry.plane_id for entry in self.entries)

        if len(set(ids)) != len(ids):
            raise MetricsError(
                "entries cannot contain duplicate plane IDs."
            )

        expected_ranks = tuple(range(1, len(self.entries) + 1))
        actual_ranks = tuple(entry.rank for entry in self.entries)

        if actual_ranks != expected_ranks:
            raise MetricsError(
                "entry ranks must be consecutive and start at 1."
            )

        if not isinstance(self.higher_is_more_important, bool):
            raise MetricsError(
                "higher_is_more_important must be a bool."
            )

        object.__setattr__(self, "metric_name", metric_name)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def winner(self) -> RankedPlane | None:
        return self.entries[0] if self.entries else None

    @property
    def winner_id(self) -> str | None:
        winner = self.winner
        return None if winner is None else winner.plane_id

    def score_for(self, plane_id: str) -> float:
        if not isinstance(plane_id, str):
            raise MetricsError("plane_id must be a string.")

        normalized = plane_id.strip()

        if not normalized:
            raise MetricsError("plane_id cannot be empty.")

        for entry in self.entries:
            if entry.plane_id == normalized:
                return entry.score

        raise MetricsError(f"Unknown plane_id: {normalized!r}.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "higher_is_more_important": (
                self.higher_is_more_important
            ),
            "entries": [
                entry.as_dict() for entry in self.entries
            ],
        }


def _build_ranking(
    *,
    metric_name: str,
    plane_ids: tuple[str, ...],
    scores: np.ndarray,
    higher_is_more_important: bool,
) -> PlaneRanking:
    indexed = list(enumerate(scores.tolist()))

    indexed.sort(
        key=lambda item: (
            -item[1] if higher_is_more_important else item[1],
            item[0],
        )
    )

    entries = tuple(
        RankedPlane(
            plane_id=plane_ids[index],
            score=float(score),
            rank=rank,
        )
        for rank, (index, score) in enumerate(indexed, start=1)
    )

    return PlaneRanking(
        metric_name=metric_name,
        entries=entries,
        higher_is_more_important=higher_is_more_important,
    )


def rank_d_fast(
    plane_ids: Sequence[str],
    reserve: Sequence[float] | np.ndarray,
) -> PlaneRanking:
    """
    Rank planes by loss of functional reserve.

    The smallest reserve is ranked first. This operationalizes ``D_fast`` as
    the first element that loses support rather than the largest activation.
    """

    reserve_vector = _as_finite_vector(reserve, name="reserve")
    ids = _normalize_plane_ids(
        plane_ids,
        expected_size=reserve_vector.size,
    )

    return _build_ranking(
        metric_name="D_fast",
        plane_ids=ids,
        scores=reserve_vector,
        higher_is_more_important=False,
    )


def rank_d_root(
    plane_ids: Sequence[str],
    sensitivity: Sequence[Sequence[float]] | np.ndarray,
) -> PlaneRanking:
    """
    Rank planes by destabilizing sensitivity.

    The score is the absolute column sum of the supplied Jacobian or local
    sensitivity matrix. A high score means perturbing the plane influences
    many downstream dimensions.
    """

    matrix = _as_square_matrix(
        sensitivity,
        name="sensitivity",
    )
    ids = _normalize_plane_ids(
        plane_ids,
        expected_size=matrix.shape[0],
    )

    scores = np.sum(np.abs(matrix), axis=0)

    return _build_ranking(
        metric_name="D_root",
        plane_ids=ids,
        scores=scores,
        higher_is_more_important=True,
    )


def rank_node_star(
    plane_ids: Sequence[str],
    benefit: Sequence[float] | np.ndarray,
    *,
    cost: Sequence[float] | np.ndarray | None = None,
    risk: Sequence[float] | np.ndarray | None = None,
    risk_weight: float = 1.0,
) -> PlaneRanking:
    """
    Rank candidate intervention nodes.

    ``score = benefit / cost - risk_weight * risk``

    Cost defaults to one and risk defaults to zero.
    """

    benefit_vector = _as_finite_vector(benefit, name="benefit")
    size = benefit_vector.size
    ids = _normalize_plane_ids(
        plane_ids,
        expected_size=size,
    )

    if cost is None:
        cost_vector = np.ones(size, dtype=float)
    else:
        cost_vector = _as_finite_vector(cost, name="cost")

        if cost_vector.size != size:
            raise MetricsError(
                "cost length must match benefit length."
            )

        if np.any(cost_vector <= 0.0):
            raise MetricsError(
                "cost values must be greater than 0.0."
            )

    if risk is None:
        risk_vector = np.zeros(size, dtype=float)
    else:
        risk_vector = _as_finite_vector(risk, name="risk")

        if risk_vector.size != size:
            raise MetricsError(
                "risk length must match benefit length."
            )

    if isinstance(risk_weight, bool):
        raise MetricsError("risk_weight must be a real number.")

    try:
        normalized_risk_weight = float(risk_weight)
    except (TypeError, ValueError) as exc:
        raise MetricsError(
            "risk_weight must be a real number."
        ) from exc

    if not math.isfinite(normalized_risk_weight):
        raise MetricsError("risk_weight must be finite.")

    if normalized_risk_weight < 0.0:
        raise MetricsError(
            "risk_weight must be greater than or equal to 0.0."
        )

    scores = (
        benefit_vector / cost_vector
        - normalized_risk_weight * risk_vector
    )

    return _build_ranking(
        metric_name="Node*",
        plane_ids=ids,
        scores=scores,
        higher_is_more_important=True,
    )


_SEVERITY_WEIGHT = {
    CascadeEventSeverity.TRACE: 0.25,
    CascadeEventSeverity.LOW: 0.5,
    CascadeEventSeverity.MODERATE: 1.0,
    CascadeEventSeverity.HIGH: 2.0,
    CascadeEventSeverity.CRITICAL: 4.0,
}


def event_burden(history: CascadeHistory) -> float:
    """Return severity-weighted absolute event magnitude."""

    if not isinstance(history, CascadeHistory):
        raise MetricsError(
            "history must be a CascadeHistory."
        )

    burden = 0.0

    for event in history.events:
        weight = _SEVERITY_WEIGHT[event.severity]
        burden += weight * abs(event.delta)

    return float(burden)


@dataclass(frozen=True, slots=True)
class SnapshotMetrics:
    """Aggregate diagnostics for one StateSnapshot."""

    snapshot_id: str
    time: float
    plane_count: int
    event_count: int
    history_event_count: int
    changed_plane_count: int
    maximum_absolute_delta: float
    mean_absolute_delta: float
    event_burden: float
    d_fast: PlaneRanking
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str):
            raise MetricsError("snapshot_id must be a string.")

        snapshot_id = self.snapshot_id.strip()

        if not snapshot_id:
            raise MetricsError("snapshot_id cannot be empty.")

        for name in (
            "plane_count",
            "event_count",
            "history_event_count",
            "changed_plane_count",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise MetricsError(
                    f"{name} must be a non-negative integer."
                )

        for name in (
            "time",
            "maximum_absolute_delta",
            "mean_absolute_delta",
            "event_burden",
        ):
            value = float(getattr(self, name))

            if not math.isfinite(value):
                raise MetricsError(f"{name} must be finite.")

            if value < 0.0:
                raise MetricsError(
                    f"{name} must be non-negative."
                )

            object.__setattr__(self, name, value)

        if not isinstance(self.d_fast, PlaneRanking):
            raise MetricsError(
                "d_fast must be a PlaneRanking."
            )

        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "time": self.time,
            "plane_count": self.plane_count,
            "event_count": self.event_count,
            "history_event_count": self.history_event_count,
            "changed_plane_count": self.changed_plane_count,
            "maximum_absolute_delta": self.maximum_absolute_delta,
            "mean_absolute_delta": self.mean_absolute_delta,
            "event_burden": self.event_burden,
            "d_fast": self.d_fast.as_dict(),
            "metadata": dict(self.metadata),
        }


def calculate_snapshot_metrics(
    snapshot: StateSnapshot,
    *,
    reserve: Sequence[float] | np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SnapshotMetrics:
    """
    Calculate deterministic aggregate metrics for one snapshot.

    When explicit reserve values are omitted, reserve is approximated as
    ``1 - abs(activation_after)``. This keeps the default operational and
    bounded for normalized activation states while still allowing callers to
    supply domain-specific capacity reserve.
    """

    if not isinstance(snapshot, StateSnapshot):
        raise MetricsError(
            "snapshot must be a StateSnapshot."
        )

    activation_before = np.asarray(
        snapshot.kernel_result.activation_before,
        dtype=float,
    )
    activation_after = np.asarray(
        snapshot.kernel_result.activation_after,
        dtype=float,
    )
    absolute_delta = np.abs(
        activation_after - activation_before
    )

    if reserve is None:
        reserve_vector = 1.0 - np.abs(activation_after)
    else:
        reserve_vector = _as_finite_vector(
            reserve,
            name="reserve",
        )

        if reserve_vector.size != snapshot.plane_count:
            raise MetricsError(
                "reserve length must match snapshot plane count."
            )

    d_fast = rank_d_fast(
        snapshot.plane_ids,
        reserve_vector,
    )

    return SnapshotMetrics(
        snapshot_id=snapshot.snapshot_id,
        time=snapshot.time,
        plane_count=snapshot.plane_count,
        event_count=snapshot.event_count,
        history_event_count=snapshot.history_event_count,
        changed_plane_count=len(snapshot.changed_plane_ids),
        maximum_absolute_delta=(
            float(np.max(absolute_delta))
            if absolute_delta.size
            else 0.0
        ),
        mean_absolute_delta=(
            float(np.mean(absolute_delta))
            if absolute_delta.size
            else 0.0
        ),
        event_burden=event_burden(snapshot.history),
        d_fast=d_fast,
        metadata={} if metadata is None else metadata,
    )


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    """Aggregate metrics across a snapshot sequence."""

    snapshots: tuple[SnapshotMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshots, tuple):
            object.__setattr__(
                self,
                "snapshots",
                tuple(self.snapshots),
            )

        if any(
            not isinstance(item, SnapshotMetrics)
            for item in self.snapshots
        ):
            raise MetricsError(
                "snapshots must contain only SnapshotMetrics."
            )

        times = tuple(item.time for item in self.snapshots)

        if any(
            current < previous - _FLOAT_TOLERANCE
            for previous, current in zip(
                times,
                times[1:],
            )
        ):
            raise MetricsError(
                "snapshot metrics must be chronological."
            )

    def __iter__(self):
        return iter(self.snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def final(self) -> SnapshotMetrics | None:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def peak_event_burden(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(item.event_burden for item in self.snapshots)

    @property
    def maximum_absolute_delta(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(
            item.maximum_absolute_delta
            for item in self.snapshots
        )

    @property
    def total_current_events(self) -> int:
        return sum(item.event_count for item in self.snapshots)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_count": len(self.snapshots),
            "peak_event_burden": self.peak_event_burden,
            "maximum_absolute_delta": (
                self.maximum_absolute_delta
            ),
            "total_current_events": self.total_current_events,
            "snapshots": [
                item.as_dict() for item in self.snapshots
            ],
        }


def calculate_simulation_metrics(
    snapshots: Sequence[StateSnapshot],
) -> SimulationMetrics:
    """Calculate metrics for a chronological snapshot sequence."""

    if isinstance(snapshots, (str, bytes)):
        raise MetricsError(
            "snapshots must be a sequence of StateSnapshot."
        )

    try:
        snapshot_tuple = tuple(snapshots)
    except TypeError as exc:
        raise MetricsError(
            "snapshots must be a sequence of StateSnapshot."
        ) from exc

    if any(
        not isinstance(snapshot, StateSnapshot)
        for snapshot in snapshot_tuple
    ):
        raise MetricsError(
            "snapshots must contain only StateSnapshot objects."
        )

    metrics = tuple(
        calculate_snapshot_metrics(snapshot)
        for snapshot in snapshot_tuple
    )

    return SimulationMetrics(metrics)


__all__ = [
    "MetricsError",
    "PlaneRanking",
    "RankedPlane",
    "SimulationMetrics",
    "SnapshotMetrics",
    "calculate_simulation_metrics",
    "calculate_snapshot_metrics",
    "event_burden",
    "rank_d_fast",
    "rank_d_root",
    "rank_node_star",
    "spectral_coherence",
    "spectral_radius",
]
