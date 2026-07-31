"""
ROIF simulation engine.

Coordinates repeated PlaneKernel execution while accumulating cascade
history and producing immutable StateSnapshot objects.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .cascade_event import (
    CascadeEventBatch,
    CascadeEventPolicy,
    detect_cascade_events,
)
from .history import CascadeHistory
from .plane_kernel import PlaneKernel, PlaneKernelResult
from .state_snapshot import StateSnapshot, create_state_snapshot


class SimulationError(ValueError):
    """Raised when simulation configuration is invalid."""


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable simulator configuration."""

    event_policy: CascadeEventPolicy = field(
        default_factory=CascadeEventPolicy
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_policy, CascadeEventPolicy):
            raise SimulationError(
                "event_policy must be a CascadeEventPolicy."
            )
        if not isinstance(self.metadata, Mapping):
            raise SimulationError("metadata must be a mapping.")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Immutable collection of simulation snapshots."""

    snapshots: tuple[StateSnapshot, ...]

    def __iter__(self) -> Iterator[StateSnapshot]:
        return iter(self.snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def final_snapshot(self) -> StateSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def history(self) -> CascadeHistory:
        if not self.snapshots:
            return CascadeHistory()
        return self.snapshots[-1].history


def run_simulation(
    kernel: PlaneKernel,
    initial_activation: Iterable[float],
    *,
    steps: int,
    dt: float,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """
    Execute a deterministic ROIF simulation.

    Returns a sequence of immutable StateSnapshot objects.
    """

    if not isinstance(kernel, PlaneKernel):
        raise SimulationError("kernel must be a PlaneKernel.")

    if not isinstance(steps, int) or isinstance(steps, bool):
        raise SimulationError("steps must be an integer.")

    if steps < 0:
        raise SimulationError("steps must be non-negative.")

    if config is None:
        config = SimulationConfig()

    if not isinstance(config, SimulationConfig):
        raise SimulationError(
            "config must be a SimulationConfig."
        )

    history = CascadeHistory()
    activation = tuple(float(x) for x in initial_activation)
    snapshots: list[StateSnapshot] = []

    for step in range(steps):
        result: PlaneKernelResult = kernel.step(
            activation,
            dt=dt,
        )

        batch: CascadeEventBatch = detect_cascade_events(
            result,
            policy=config.event_policy,
        )

        snapshot = create_state_snapshot(
            result,
            batch,
            history,
            step_index=step,
            metadata=config.metadata,
        )

        history = snapshot.history
        activation = tuple(result.activation_after)
        snapshots.append(snapshot)

    return SimulationResult(tuple(snapshots))


__all__ = [
    "SimulationConfig",
    "SimulationError",
    "SimulationResult",
    "run_simulation",
]
