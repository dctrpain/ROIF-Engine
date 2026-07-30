from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .network import Network, NetworkState, NetworkStepStats
from .node import Node


ExternalForceMap = Mapping[Node, Any]
ExternalForceProvider = Callable[
    ["Simulation"],
    ExternalForceMap | None,
]
StepHook = Callable[
    ["Simulation"],
    None,
]


@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable execution settings for Simulation.
    """

    dt: float = 0.01
    update_materials: bool = True
    include_active: bool = True
    solve_constraints: bool = True
    record_network_history: bool = False
    record_simulation_history: bool = True

    def __post_init__(self) -> None:
        dt = float(self.dt)

        if not np.isfinite(dt):
            raise ValueError("dt must be finite")

        if dt <= 0.0:
            raise ValueError("dt must be positive")

        object.__setattr__(self, "dt", dt)


@dataclass
class SimulationFrame:
    """
    Complete state captured after one physical step.
    """

    frame_index: int
    time: float
    step_index: int
    dt: float
    state: NetworkState
    statistics: NetworkStepStats

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "time": float(self.time),
            "step_index": int(self.step_index),
            "dt": float(self.dt),
            "statistics": self.statistics.as_dict(),
        }


@dataclass
class SimulationRun:
    """
    Result returned by run_steps() and run().
    """

    statistics: list[NetworkStepStats] = field(
        default_factory=list
    )

    @property
    def steps(self) -> int:
        return len(self.statistics)

    @property
    def final_statistics(
        self,
    ) -> NetworkStepStats | None:
        if not self.statistics:
            return None

        return self.statistics[-1]


class Simulation:
    """
    High-level physical simulation controller.

    Simulation owns execution policy. Network owns physical state
    and performs one physical step.

    Execution path
    --------------
        Simulation.step()
            -> resolve external forces
            -> Network.step(...)
            -> capture SimulationFrame
            -> run hooks

    The class does not duplicate force laws, integration, material
    evolution, constraints, or diagnostics already implemented by
    Network.
    """

    VERSION = "2.1"

    def __init__(
        self,
        network: Network,
        *,
        dt: float = 0.01,
        external_forces: (
            ExternalForceMap
            | ExternalForceProvider
            | None
        ) = None,
        update_materials: bool = True,
        include_active: bool = True,
        solve_constraints: bool = True,
        record_network_history: bool = False,
        record_simulation_history: bool = True,
    ) -> None:
        if not isinstance(network, Network):
            raise TypeError(
                "network must be an instance of Network"
            )

        self.network = network
        self.config = SimulationConfig(
            dt=dt,
            update_materials=update_materials,
            include_active=include_active,
            solve_constraints=solve_constraints,
            record_network_history=(
                record_network_history
            ),
            record_simulation_history=(
                record_simulation_history
            ),
        )

        self.external_forces = external_forces

        self.before_step_hooks: list[StepHook] = []
        self.after_step_hooks: list[StepHook] = []

        self.frames: list[SimulationFrame] = []
        self.statistics: list[NetworkStepStats] = []

        self._initial_network = network.clone()
        self._frame_index = 0

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def time(self) -> float:
        return float(self.network.time)

    @property
    def step_index(self) -> int:
        return int(self.network.step_index)

    @property
    def frame_index(self) -> int:
        return int(self._frame_index)

    @property
    def latest_frame(
        self,
    ) -> SimulationFrame | None:
        if not self.frames:
            return None

        return self.frames[-1]

    def add_before_step_hook(
        self,
        hook: StepHook,
    ) -> None:
        if not callable(hook):
            raise TypeError("hook must be callable")

        self.before_step_hooks.append(hook)

    def add_after_step_hook(
        self,
        hook: StepHook,
    ) -> None:
        if not callable(hook):
            raise TypeError("hook must be callable")

        self.after_step_hooks.append(hook)

    def _run_hooks(
        self,
        hooks: list[StepHook],
    ) -> None:
        for hook in tuple(hooks):
            hook(self)

    def _resolve_external_forces(
        self,
    ) -> ExternalForceMap | None:
        source = self.external_forces

        if source is None:
            return None

        if callable(source):
            resolved = source(self)
        else:
            resolved = source

        if resolved is None:
            return None

        if not isinstance(resolved, Mapping):
            raise TypeError(
                "external forces must resolve to a mapping "
                "{node: force}"
            )

        return resolved

    def _capture_frame(
        self,
        stats: NetworkStepStats,
    ) -> SimulationFrame:
        frame = SimulationFrame(
            frame_index=self._frame_index,
            time=self.time,
            step_index=self.step_index,
            dt=float(stats.dt),
            state=self.network.snapshot(),
            statistics=deepcopy(stats),
        )

        self._frame_index += 1

        if self.config.record_simulation_history:
            self.frames.append(frame)

        return frame

    def step(
        self,
        *,
        dt: float | None = None,
    ) -> NetworkStepStats:
        """
        Advance exactly one physical step.

        A temporary dt may be supplied for the final partial step of
        run(duration). Normal solver-driven animation should call
        step() without overriding dt.
        """

        step_dt = (
            self.dt
            if dt is None
            else Network._validate_dt(dt)
        )

        self._run_hooks(self.before_step_hooks)

        current_external_forces = (
            self._resolve_external_forces()
        )

        stats = self.network.step(
            dt=step_dt,
            external_forces=current_external_forces,
            update_materials=(
                self.config.update_materials
            ),
            include_active=self.config.include_active,
            solve_constraints=(
                self.config.solve_constraints
            ),
            record=(
                self.config.record_network_history
            ),
        )

        self.statistics.append(
            deepcopy(stats)
        )

        self._capture_frame(stats)
        self._run_hooks(self.after_step_hooks)

        return stats

    def run_steps(
        self,
        steps: int,
    ) -> SimulationRun:
        """
        Advance a fixed integer number of physical steps.
        """

        steps = int(steps)

        if steps < 0:
            raise ValueError("steps cannot be negative")

        result = SimulationRun()

        for _ in range(steps):
            result.statistics.append(
                self.step()
            )

        return result

    def run(
        self,
        duration: float,
    ) -> SimulationRun:
        """
        Advance for the requested physical duration.

        Full steps use config.dt. If duration is not divisible by dt,
        the last step is shortened so elapsed time is exact.
        """

        duration = float(duration)

        if not np.isfinite(duration):
            raise ValueError("duration must be finite")

        if duration < 0.0:
            raise ValueError(
                "duration cannot be negative"
            )

        result = SimulationRun()
        remaining = duration
        tolerance = np.finfo(float).eps * max(
            1.0,
            abs(duration),
        )

        while remaining > tolerance:
            step_dt = min(
                self.dt,
                remaining,
            )

            result.statistics.append(
                self.step(dt=step_dt)
            )

            remaining -= step_dt

        return result

    def reset(
        self,
        *,
        clear_history: bool = True,
    ) -> None:
        """
        Restore the complete network state captured at construction.
        """

        self.network.restore(
            self._initial_network
        )

        self._frame_index = 0
        self.statistics.clear()

        if clear_history:
            self.frames.clear()
            self.network.clear_history()

    def set_initial_state(self) -> None:
        """
        Replace the reset checkpoint with the current network state.
        """

        self._initial_network = (
            self.network.clone()
        )

    def clear_history(self) -> None:
        self.frames.clear()
        self.statistics.clear()
        self.network.clear_history()

    def snapshot(self) -> NetworkState:
        return self.network.snapshot()

    def advance_frame(
        self,
        frame_index: int | None = None,
    ) -> NetworkStepStats:
        """
        Animator-compatible callback.

        frame_index is accepted intentionally but physical evolution
        always advances by one Simulation step.
        """

        del frame_index
        return self.step()
