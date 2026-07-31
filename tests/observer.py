"""
ROIF simulation observer.

The observer is a stateful, deterministic layer between immutable
``StateSnapshot`` objects and downstream analytics. It records chronological
observations, calculates ``SnapshotMetrics``, optionally calculates spectral
coherence ``eta``, detects threshold transitions, and emits immutable alerts.

The module does not mutate snapshots, simulation state, or metric objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from .metrics import (
    MetricsError,
    SimulationMetrics,
    SnapshotMetrics,
    calculate_snapshot_metrics,
    spectral_coherence,
)
from .state_snapshot import StateSnapshot


class ObserverError(ValueError):
    """Raised when observer input or configuration is invalid."""


_FLOAT_TOLERANCE = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ObserverError(f"{name} must be a real number.")

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ObserverError(
            f"{name} must be a real number."
        ) from exc

    if not math.isfinite(normalized):
        raise ObserverError(f"{name} must be finite.")

    if strictly_positive and normalized <= 0.0:
        raise ObserverError(
            f"{name} must be greater than 0.0."
        )

    if minimum is not None and normalized < minimum:
        raise ObserverError(
            f"{name} must be greater than or equal to {minimum}."
        )

    return normalized


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise ObserverError("metadata must be a mapping.")

    copied: dict[str, Any] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ObserverError("metadata keys must be strings.")

        normalized = key.strip()

        if not normalized:
            raise ObserverError("metadata keys cannot be empty.")

        copied[normalized] = value

    return MappingProxyType(copied)


class AlertLevel(str, Enum):
    """Observer alert severity."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertKind(str, Enum):
    """Stable machine-readable alert categories."""

    ETA_WARNING = "eta_warning"
    ETA_CRITICAL = "eta_critical"
    EVENT_BURDEN = "event_burden"
    MAXIMUM_DELTA = "maximum_delta"
    D_FAST_CHANGED = "d_fast_changed"


@dataclass(frozen=True, slots=True)
class ObserverAlert:
    """One immutable alert produced from an observation."""

    kind: AlertKind
    level: AlertLevel
    snapshot_id: str
    time: float
    message: str
    value: float | str | None = None
    threshold: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AlertKind):
            raise ObserverError("kind must be an AlertKind.")

        if not isinstance(self.level, AlertLevel):
            raise ObserverError("level must be an AlertLevel.")

        if not isinstance(self.snapshot_id, str):
            raise ObserverError("snapshot_id must be a string.")

        snapshot_id = self.snapshot_id.strip()

        if not snapshot_id:
            raise ObserverError("snapshot_id cannot be empty.")

        time = _finite_float(
            self.time,
            name="time",
            minimum=0.0,
        )

        if not isinstance(self.message, str):
            raise ObserverError("message must be a string.")

        message = self.message.strip()

        if not message:
            raise ObserverError("message cannot be empty.")

        value = self.value

        if isinstance(value, bool):
            raise ObserverError(
                "value must be a finite number, string, or None."
            )

        if value is not None and not isinstance(value, str):
            value = _finite_float(value, name="value")

        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ObserverError(
                    "string value cannot be empty."
                )

        threshold = self.threshold

        if threshold is not None:
            threshold = _finite_float(
                threshold,
                name="threshold",
            )

        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "level": self.level.value,
            "snapshot_id": self.snapshot_id,
            "time": self.time,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ObserverConfig:
    """Thresholds and behavior for ``SimulationObserver``."""

    eta_warning_threshold: float = 0.25
    eta_critical_threshold: float = 0.0
    event_burden_threshold: float | None = None
    maximum_delta_threshold: float | None = None
    emit_d_fast_change: bool = True
    emit_repeated_threshold_alerts: bool = False

    def __post_init__(self) -> None:
        warning = _finite_float(
            self.eta_warning_threshold,
            name="eta_warning_threshold",
        )
        critical = _finite_float(
            self.eta_critical_threshold,
            name="eta_critical_threshold",
        )

        if critical > warning:
            raise ObserverError(
                "eta_critical_threshold cannot be greater than "
                "eta_warning_threshold."
            )

        burden = self.event_burden_threshold
        if burden is not None:
            burden = _finite_float(
                burden,
                name="event_burden_threshold",
                minimum=0.0,
            )

        delta = self.maximum_delta_threshold
        if delta is not None:
            delta = _finite_float(
                delta,
                name="maximum_delta_threshold",
                minimum=0.0,
            )

        if not isinstance(self.emit_d_fast_change, bool):
            raise ObserverError(
                "emit_d_fast_change must be a bool."
            )

        if not isinstance(
            self.emit_repeated_threshold_alerts,
            bool,
        ):
            raise ObserverError(
                "emit_repeated_threshold_alerts must be a bool."
            )

        object.__setattr__(
            self,
            "eta_warning_threshold",
            warning,
        )
        object.__setattr__(
            self,
            "eta_critical_threshold",
            critical,
        )
        object.__setattr__(
            self,
            "event_burden_threshold",
            burden,
        )
        object.__setattr__(
            self,
            "maximum_delta_threshold",
            delta,
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """Metrics and alerts generated from one snapshot."""

    snapshot: StateSnapshot
    metrics: SnapshotMetrics
    eta: float | None = None
    alerts: tuple[ObserverAlert, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, StateSnapshot):
            raise ObserverError(
                "snapshot must be a StateSnapshot."
            )

        if not isinstance(self.metrics, SnapshotMetrics):
            raise ObserverError(
                "metrics must be a SnapshotMetrics."
            )

        if self.metrics.snapshot_id != self.snapshot.snapshot_id:
            raise ObserverError(
                "metrics and snapshot IDs must match."
            )

        eta = self.eta
        if eta is not None:
            eta = _finite_float(eta, name="eta")

        if not isinstance(self.alerts, tuple):
            object.__setattr__(
                self,
                "alerts",
                tuple(self.alerts),
            )

        if any(
            not isinstance(alert, ObserverAlert)
            for alert in self.alerts
        ):
            raise ObserverError(
                "alerts must contain only ObserverAlert objects."
            )

        if any(
            alert.snapshot_id != self.snapshot.snapshot_id
            for alert in self.alerts
        ):
            raise ObserverError(
                "every alert must reference the observation snapshot."
            )

        object.__setattr__(self, "eta", eta)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def snapshot_id(self) -> str:
        return self.snapshot.snapshot_id

    @property
    def time(self) -> float:
        return self.snapshot.time

    @property
    def d_fast_id(self) -> str | None:
        return self.metrics.d_fast.winner_id

    @property
    def has_alerts(self) -> bool:
        return bool(self.alerts)

    @property
    def highest_alert_level(self) -> AlertLevel | None:
        if not self.alerts:
            return None

        priority = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.CRITICAL: 2,
        }
        return max(
            (alert.level for alert in self.alerts),
            key=priority.__getitem__,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "time": self.time,
            "eta": self.eta,
            "d_fast_id": self.d_fast_id,
            "metrics": self.metrics.as_dict(),
            "alerts": [
                alert.as_dict() for alert in self.alerts
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ObservationSeries:
    """Immutable chronological collection of observations."""

    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observations, tuple):
            object.__setattr__(
                self,
                "observations",
                tuple(self.observations),
            )

        if any(
            not isinstance(item, Observation)
            for item in self.observations
        ):
            raise ObserverError(
                "observations must contain only Observation objects."
            )

        times = tuple(
            item.time for item in self.observations
        )

        if any(
            current < previous - _FLOAT_TOLERANCE
            for previous, current in zip(times, times[1:])
        ):
            raise ObserverError(
                "observations must be chronological."
            )

        snapshot_ids = tuple(
            item.snapshot_id for item in self.observations
        )

        if len(set(snapshot_ids)) != len(snapshot_ids):
            raise ObserverError(
                "observations cannot contain duplicate snapshot IDs."
            )

    def __iter__(self):
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def first(self) -> Observation | None:
        return (
            self.observations[0]
            if self.observations
            else None
        )

    @property
    def final(self) -> Observation | None:
        return (
            self.observations[-1]
            if self.observations
            else None
        )

    @property
    def metrics(self) -> SimulationMetrics:
        return SimulationMetrics(
            tuple(
                item.metrics
                for item in self.observations
            )
        )

    @property
    def alerts(self) -> tuple[ObserverAlert, ...]:
        return tuple(
            alert
            for observation in self.observations
            for alert in observation.alerts
        )

    @property
    def eta_series(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (item.time, item.eta)
            for item in self.observations
            if item.eta is not None
        )

    @property
    def d_fast_series(
        self,
    ) -> tuple[tuple[float, str | None], ...]:
        return tuple(
            (item.time, item.d_fast_id)
            for item in self.observations
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_count": len(self.observations),
            "alert_count": len(self.alerts),
            "eta_series": [
                {"time": time, "eta": eta}
                for time, eta in self.eta_series
            ],
            "d_fast_series": [
                {"time": time, "plane_id": plane_id}
                for time, plane_id in self.d_fast_series
            ],
            "metrics": self.metrics.as_dict(),
            "observations": [
                item.as_dict()
                for item in self.observations
            ],
        }


ObservationHook = Callable[[Observation], None]


class SimulationObserver:
    """
    Stateful observer for chronological ``StateSnapshot`` objects.

    ``observe`` accepts optional domain-specific reserve values and an optional
    operator matrix. When an operator is supplied, ``lambda_critical`` is
    required and spectral coherence is calculated for that observation.
    """

    def __init__(
        self,
        config: ObserverConfig | None = None,
        *,
        hooks: Sequence[ObservationHook] = (),
    ) -> None:
        if config is None:
            config = ObserverConfig()

        if not isinstance(config, ObserverConfig):
            raise ObserverError(
                "config must be an ObserverConfig."
            )

        if isinstance(hooks, (str, bytes)):
            raise ObserverError(
                "hooks must be a sequence of callables."
            )

        try:
            normalized_hooks = tuple(hooks)
        except TypeError as exc:
            raise ObserverError(
                "hooks must be a sequence of callables."
            ) from exc

        if any(not callable(hook) for hook in normalized_hooks):
            raise ObserverError(
                "every hook must be callable."
            )

        self._config = config
        self._hooks = normalized_hooks
        self._observations: list[Observation] = []
        self._last_eta_zone: str | None = None
        self._burden_above_threshold = False
        self._delta_above_threshold = False

    @property
    def config(self) -> ObserverConfig:
        return self._config

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    @property
    def series(self) -> ObservationSeries:
        return ObservationSeries(self.observations)

    @property
    def latest(self) -> Observation | None:
        return (
            self._observations[-1]
            if self._observations
            else None
        )

    def __len__(self) -> int:
        return len(self._observations)

    def reset(self) -> None:
        self._observations.clear()
        self._last_eta_zone = None
        self._burden_above_threshold = False
        self._delta_above_threshold = False

    def observe(
        self,
        snapshot: StateSnapshot,
        *,
        reserve: Sequence[float] | np.ndarray | None = None,
        operator: Sequence[Sequence[float]] | np.ndarray | None = None,
        lambda_critical: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Observation:
        if not isinstance(snapshot, StateSnapshot):
            raise ObserverError(
                "snapshot must be a StateSnapshot."
            )

        self._validate_chronology(snapshot)

        if operator is None and lambda_critical is not None:
            raise ObserverError(
                "lambda_critical requires operator."
            )

        if operator is not None and lambda_critical is None:
            raise ObserverError(
                "operator requires lambda_critical."
            )

        try:
            metrics = calculate_snapshot_metrics(
                snapshot,
                reserve=reserve,
                metadata=metadata,
            )
        except MetricsError as exc:
            raise ObserverError(
                f"Cannot calculate snapshot metrics: {exc}"
            ) from exc

        eta: float | None = None

        if operator is not None:
            try:
                eta = spectral_coherence(
                    operator,
                    lambda_critical=lambda_critical,
                )
            except MetricsError as exc:
                raise ObserverError(
                    f"Cannot calculate spectral coherence: {exc}"
                ) from exc

        alerts = self._build_alerts(
            snapshot=snapshot,
            metrics=metrics,
            eta=eta,
        )

        observation = Observation(
            snapshot=snapshot,
            metrics=metrics,
            eta=eta,
            alerts=alerts,
            metadata=metadata,
        )

        self._observations.append(observation)
        self._update_threshold_state(
            metrics=metrics,
            eta=eta,
        )

        for hook in self._hooks:
            hook(observation)

        return observation

    def observe_many(
        self,
        snapshots: Sequence[StateSnapshot],
        *,
        reserves: Sequence[
            Sequence[float] | np.ndarray | None
        ] | None = None,
        operators: Sequence[
            Sequence[Sequence[float]] | np.ndarray | None
        ] | None = None,
        lambda_critical: float | None = None,
    ) -> ObservationSeries:
        if isinstance(snapshots, (str, bytes)):
            raise ObserverError(
                "snapshots must be a sequence of StateSnapshot."
            )

        try:
            snapshot_tuple = tuple(snapshots)
        except TypeError as exc:
            raise ObserverError(
                "snapshots must be a sequence of StateSnapshot."
            ) from exc

        reserve_tuple = self._normalize_optional_sequence(
            reserves,
            expected_size=len(snapshot_tuple),
            name="reserves",
        )
        operator_tuple = self._normalize_optional_sequence(
            operators,
            expected_size=len(snapshot_tuple),
            name="operators",
        )

        if operators is not None and lambda_critical is None:
            raise ObserverError(
                "operators require lambda_critical."
            )

        for index, snapshot in enumerate(snapshot_tuple):
            self.observe(
                snapshot,
                reserve=reserve_tuple[index],
                operator=operator_tuple[index],
                lambda_critical=(
                    lambda_critical
                    if operator_tuple[index] is not None
                    else None
                ),
            )

        return self.series

    def _validate_chronology(
        self,
        snapshot: StateSnapshot,
    ) -> None:
        if not self._observations:
            return

        latest = self._observations[-1]

        if snapshot.snapshot_id == latest.snapshot_id:
            raise ObserverError(
                "duplicate snapshot_id cannot be observed."
            )

        if snapshot.time < latest.time - _FLOAT_TOLERANCE:
            raise ObserverError(
                "snapshots must be observed chronologically."
            )

        if any(
            item.snapshot_id == snapshot.snapshot_id
            for item in self._observations
        ):
            raise ObserverError(
                "duplicate snapshot_id cannot be observed."
            )

    @staticmethod
    def _normalize_optional_sequence(
        values: Sequence[Any] | None,
        *,
        expected_size: int,
        name: str,
    ) -> tuple[Any, ...]:
        if values is None:
            return (None,) * expected_size

        if isinstance(values, (str, bytes)):
            raise ObserverError(
                f"{name} must be a sequence."
            )

        try:
            normalized = tuple(values)
        except TypeError as exc:
            raise ObserverError(
                f"{name} must be a sequence."
            ) from exc

        if len(normalized) != expected_size:
            raise ObserverError(
                f"{name} length must match snapshots length."
            )

        return normalized

    def _eta_zone(self, eta: float) -> str:
        if eta <= self._config.eta_critical_threshold:
            return "critical"

        if eta <= self._config.eta_warning_threshold:
            return "warning"

        return "stable"

    def _should_emit_zone_alert(
        self,
        current_zone: str,
    ) -> bool:
        if current_zone == "stable":
            return False

        if self._config.emit_repeated_threshold_alerts:
            return True

        return current_zone != self._last_eta_zone

    def _should_emit_threshold_alert(
        self,
        *,
        currently_above: bool,
        previously_above: bool,
    ) -> bool:
        if not currently_above:
            return False

        if self._config.emit_repeated_threshold_alerts:
            return True

        return not previously_above

    def _build_alerts(
        self,
        *,
        snapshot: StateSnapshot,
        metrics: SnapshotMetrics,
        eta: float | None,
    ) -> tuple[ObserverAlert, ...]:
        alerts: list[ObserverAlert] = []

        if eta is not None:
            zone = self._eta_zone(eta)

            if self._should_emit_zone_alert(zone):
                if zone == "critical":
                    alerts.append(
                        ObserverAlert(
                            kind=AlertKind.ETA_CRITICAL,
                            level=AlertLevel.CRITICAL,
                            snapshot_id=snapshot.snapshot_id,
                            time=snapshot.time,
                            message=(
                                "Spectral coherence reached the "
                                "critical region."
                            ),
                            value=eta,
                            threshold=(
                                self._config
                                .eta_critical_threshold
                            ),
                        )
                    )
                elif zone == "warning":
                    alerts.append(
                        ObserverAlert(
                            kind=AlertKind.ETA_WARNING,
                            level=AlertLevel.WARNING,
                            snapshot_id=snapshot.snapshot_id,
                            time=snapshot.time,
                            message=(
                                "Spectral coherence entered the "
                                "warning region."
                            ),
                            value=eta,
                            threshold=(
                                self._config
                                .eta_warning_threshold
                            ),
                        )
                    )

        burden_threshold = (
            self._config.event_burden_threshold
        )

        if burden_threshold is not None:
            above = (
                metrics.event_burden
                >= burden_threshold
            )

            if self._should_emit_threshold_alert(
                currently_above=above,
                previously_above=(
                    self._burden_above_threshold
                ),
            ):
                alerts.append(
                    ObserverAlert(
                        kind=AlertKind.EVENT_BURDEN,
                        level=AlertLevel.WARNING,
                        snapshot_id=snapshot.snapshot_id,
                        time=snapshot.time,
                        message=(
                            "Event burden reached the configured "
                            "threshold."
                        ),
                        value=metrics.event_burden,
                        threshold=burden_threshold,
                    )
                )

        delta_threshold = (
            self._config.maximum_delta_threshold
        )

        if delta_threshold is not None:
            above = (
                metrics.maximum_absolute_delta
                >= delta_threshold
            )

            if self._should_emit_threshold_alert(
                currently_above=above,
                previously_above=(
                    self._delta_above_threshold
                ),
            ):
                alerts.append(
                    ObserverAlert(
                        kind=AlertKind.MAXIMUM_DELTA,
                        level=AlertLevel.WARNING,
                        snapshot_id=snapshot.snapshot_id,
                        time=snapshot.time,
                        message=(
                            "Maximum activation delta reached the "
                            "configured threshold."
                        ),
                        value=metrics.maximum_absolute_delta,
                        threshold=delta_threshold,
                    )
                )

        previous_d_fast = (
            self.latest.d_fast_id
            if self.latest is not None
            else None
        )
        current_d_fast = metrics.d_fast.winner_id

        if (
            self._config.emit_d_fast_change
            and previous_d_fast is not None
            and current_d_fast is not None
            and current_d_fast != previous_d_fast
        ):
            alerts.append(
                ObserverAlert(
                    kind=AlertKind.D_FAST_CHANGED,
                    level=AlertLevel.INFO,
                    snapshot_id=snapshot.snapshot_id,
                    time=snapshot.time,
                    message=(
                        "D_fast changed to a different plane."
                    ),
                    value=current_d_fast,
                    metadata={
                        "previous_plane_id": previous_d_fast,
                        "current_plane_id": current_d_fast,
                    },
                )
            )

        return tuple(alerts)

    def _update_threshold_state(
        self,
        *,
        metrics: SnapshotMetrics,
        eta: float | None,
    ) -> None:
        if eta is not None:
            self._last_eta_zone = self._eta_zone(eta)

        burden_threshold = (
            self._config.event_burden_threshold
        )
        if burden_threshold is not None:
            self._burden_above_threshold = (
                metrics.event_burden
                >= burden_threshold
            )

        delta_threshold = (
            self._config.maximum_delta_threshold
        )
        if delta_threshold is not None:
            self._delta_above_threshold = (
                metrics.maximum_absolute_delta
                >= delta_threshold
            )


def observe_snapshots(
    snapshots: Sequence[StateSnapshot],
    *,
    config: ObserverConfig | None = None,
    reserves: Sequence[
        Sequence[float] | np.ndarray | None
    ] | None = None,
    operators: Sequence[
        Sequence[Sequence[float]] | np.ndarray | None
    ] | None = None,
    lambda_critical: float | None = None,
) -> ObservationSeries:
    """Convenience wrapper for one-shot chronological observation."""

    observer = SimulationObserver(config)
    return observer.observe_many(
        snapshots,
        reserves=reserves,
        operators=operators,
        lambda_critical=lambda_critical,
    )


__all__ = [
    "AlertKind",
    "AlertLevel",
    "Observation",
    "ObservationHook",
    "ObservationSeries",
    "ObserverAlert",
    "ObserverConfig",
    "ObserverError",
    "SimulationObserver",
    "observe_snapshots",
]
