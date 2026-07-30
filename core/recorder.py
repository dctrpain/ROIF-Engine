from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from .simulation import Simulation


@dataclass(frozen=True, slots=True)
class RecordingFrame:
    """
    Immutable frame captured from a completed simulation step.

    A frame owns a deep-copied Network instance. Playback therefore
    never advances or mutates the original Simulation.
    """

    index: int
    step_index: int
    time: float
    network: Any
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class SimulationRecording(Sequence[RecordingFrame]):
    """
    Read-only sequence of recorded simulation frames.
    """

    VERSION = "2.3"

    def __init__(
        self,
        frames: Sequence[RecordingFrame],
        *,
        simulation_dt: float,
        sample_every: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not frames:
            raise ValueError(
                "recording must contain at least one frame"
            )

        simulation_dt = float(simulation_dt)
        if not np.isfinite(simulation_dt):
            raise ValueError(
                "simulation_dt must be finite"
            )
        if simulation_dt <= 0.0:
            raise ValueError(
                "simulation_dt must be positive"
            )

        sample_every = int(sample_every)
        if sample_every <= 0:
            raise ValueError(
                "sample_every must be positive"
            )

        self._frames = tuple(frames)
        self.simulation_dt = simulation_dt
        self.sample_every = sample_every
        self.metadata = dict(metadata or {})

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(
        self,
        index: int | slice,
    ) -> RecordingFrame | tuple[RecordingFrame, ...]:
        return self._frames[index]

    def __iter__(self) -> Iterator[RecordingFrame]:
        return iter(self._frames)

    @property
    def first_frame(self) -> RecordingFrame:
        return self._frames[0]

    @property
    def last_frame(self) -> RecordingFrame:
        return self._frames[-1]

    @property
    def start_time(self) -> float:
        return float(self.first_frame.time)

    @property
    def end_time(self) -> float:
        return float(self.last_frame.time)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def physical_steps(self) -> int:
        return (
            self.last_frame.step_index
            - self.first_frame.step_index
        )

    @property
    def frame_times(self) -> np.ndarray:
        return np.asarray(
            [frame.time for frame in self._frames],
            dtype=float,
        )

    def frame(self, index: int) -> RecordingFrame:
        return self._frames[int(index)]

    def nearest_frame(
        self,
        time: float,
    ) -> RecordingFrame:
        time = float(time)

        if not np.isfinite(time):
            raise ValueError("time must be finite")

        times = self.frame_times
        index = int(
            np.argmin(np.abs(times - time))
        )
        return self._frames[index]


class SimulationRecorder:
    """
    Offline recorder for Simulation Engine.

    The recorder advances Simulation first and stores independent,
    deep-copied Network states. Recorded data can then be replayed,
    inspected, or exported without running physics again.
    """

    VERSION = "2.3"

    def __init__(
        self,
        simulation: Simulation,
    ) -> None:
        if not isinstance(simulation, Simulation):
            raise TypeError(
                "simulation must be an instance of Simulation"
            )

        self.simulation = simulation

    @staticmethod
    def _validate_steps(steps: int) -> int:
        steps = int(steps)

        if steps < 0:
            raise ValueError(
                "steps cannot be negative"
            )

        return steps

    @staticmethod
    def _validate_sample_every(
        sample_every: int,
    ) -> int:
        sample_every = int(sample_every)

        if sample_every <= 0:
            raise ValueError(
                "sample_every must be positive"
            )

        return sample_every

    def _capture(
        self,
        index: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RecordingFrame:
        return RecordingFrame(
            index=int(index),
            step_index=int(
                self.simulation.step_index
            ),
            time=float(self.simulation.time),
            network=deepcopy(
                self.simulation.network
            ),
            metadata=dict(metadata or {}),
        )

    def run_steps(
        self,
        steps: int,
        *,
        sample_every: int = 1,
        include_initial: bool = True,
        include_final: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationRecording:
        """
        Run a fixed number of physical steps and record snapshots.

        sample_every controls how many physical steps occur between
        stored frames. The final state can be forced into the recording
        even when steps is not divisible by sample_every.
        """

        steps = self._validate_steps(steps)
        sample_every = self._validate_sample_every(
            sample_every
        )

        frames: list[RecordingFrame] = []

        if include_initial:
            frames.append(
                self._capture(len(frames))
            )

        completed = 0

        while completed < steps:
            block = min(
                sample_every,
                steps - completed,
            )

            self.simulation.run_steps(block)
            completed += block

            should_capture = (
                completed % sample_every == 0
                or (
                    include_final
                    and completed == steps
                )
            )

            if should_capture:
                frames.append(
                    self._capture(len(frames))
                )

        if not frames:
            frames.append(
                self._capture(0)
            )

        recording_metadata = dict(metadata or {})
        recording_metadata.update(
            {
                "recorder_version": self.VERSION,
                "requested_steps": steps,
                "sample_every": sample_every,
                "include_initial": bool(
                    include_initial
                ),
                "include_final": bool(
                    include_final
                ),
            }
        )

        return SimulationRecording(
            frames,
            simulation_dt=self.simulation.dt,
            sample_every=sample_every,
            metadata=recording_metadata,
        )

    def run(
        self,
        duration: float,
        *,
        sample_every: int = 1,
        include_initial: bool = True,
        include_final: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationRecording:
        """
        Run for a physical duration.

        Duration must correspond to a whole number of simulation steps
        within floating-point tolerance.
        """

        duration = float(duration)

        if not np.isfinite(duration):
            raise ValueError(
                "duration must be finite"
            )
        if duration < 0.0:
            raise ValueError(
                "duration cannot be negative"
            )

        raw_steps = duration / self.simulation.dt
        steps = int(round(raw_steps))

        if not np.isclose(
            raw_steps,
            steps,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "duration must be an integer multiple "
                "of simulation.dt"
            )

        return self.run_steps(
            steps,
            sample_every=sample_every,
            include_initial=include_initial,
            include_final=include_final,
            metadata=metadata,
        )
