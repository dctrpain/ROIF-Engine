from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError as exc:
    raise ImportError(
        "NetworkAnimator requires matplotlib. "
        "Install it with: python -m pip install matplotlib"
    ) from exc

from core.simulation import Simulation
from .plotter import NetworkPlotter


FrameCallback = Callable[[int], None]


class NetworkAnimator:
    """
    Animated visualization for a ROIF Engine network.

    NetworkAnimator v2.2.2 provides deterministic simulation timing.

    Matplotlib is allowed to redraw, duplicate, or skip visual frames.
    The animator guarantees:

    - the same visual frame never advances physics twice;
    - skipped frame indices are caught up deterministically;
    - show(..., complete_on_close=True) completes the requested
      physical interval after the window closes.

    This last rule is important because GUI backends may stop one or
    more frames before the nominal end when the user closes a window,
    even when the animation appears visually complete.
    """

    VERSION = "2.2.2"

    def __init__(
        self,
        network: Any | None = None,
        *,
        simulation: Simulation | None = None,
        update_frame: FrameCallback | None = None,
        simulation_steps_per_frame: int = 1,
    ) -> None:
        if simulation is not None:
            if not isinstance(simulation, Simulation):
                raise TypeError(
                    "simulation must be an instance of Simulation"
                )

            if network is None:
                network = simulation.network
            elif network is not simulation.network:
                raise ValueError(
                    "network must be simulation.network"
                )

        if network is None:
            raise ValueError(
                "network or simulation must be provided"
            )

        if simulation is not None and update_frame is not None:
            raise ValueError(
                "provide either simulation or update_frame, "
                "not both"
            )

        if update_frame is not None and not callable(update_frame):
            raise TypeError(
                "update_frame must be callable"
            )

        simulation_steps_per_frame = int(
            simulation_steps_per_frame
        )

        if simulation_steps_per_frame <= 0:
            raise ValueError(
                "simulation_steps_per_frame must be positive"
            )

        self.network = network
        self.simulation = simulation
        self.update_frame = update_frame
        self.simulation_steps_per_frame = (
            simulation_steps_per_frame
        )

        self.plotter = NetworkPlotter(network)

        self._animation: FuncAnimation | None = None
        self._figure: Any | None = None
        self._axes: Any | None = None

        self._last_advanced_frame_index = -1
        self._requested_frame_count = 0

    @classmethod
    def from_simulation(
        cls,
        simulation: Simulation,
        *,
        simulation_steps_per_frame: int = 1,
    ) -> "NetworkAnimator":
        return cls(
            simulation=simulation,
            simulation_steps_per_frame=(
                simulation_steps_per_frame
            ),
        )

    @property
    def is_simulation_driven(self) -> bool:
        return self.simulation is not None

    @property
    def last_advanced_frame_index(self) -> int:
        return int(self._last_advanced_frame_index)

    @property
    def requested_frame_count(self) -> int:
        return int(self._requested_frame_count)

    @property
    def expected_physical_steps(self) -> int:
        return (
            self._requested_frame_count
            * self.simulation_steps_per_frame
        )

    @staticmethod
    def _validate_frames(frames: int) -> int:
        frames = int(frames)

        if frames <= 0:
            raise ValueError("frames must be positive")

        return frames

    @staticmethod
    def _validate_interval(
        interval_ms: float,
    ) -> float:
        interval_ms = float(interval_ms)

        if not np.isfinite(interval_ms):
            raise ValueError(
                "interval_ms must be finite"
            )

        if interval_ms <= 0.0:
            raise ValueError(
                "interval_ms must be positive"
            )

        return interval_ms

    @staticmethod
    def _validate_frame_index(
        frame_index: int,
    ) -> int:
        frame_index = int(frame_index)

        if frame_index < 0:
            raise ValueError(
                "frame_index cannot be negative"
            )

        return frame_index

    def set_reference_geometry(self) -> None:
        self.plotter.set_reference_geometry()

    def reset_frame_tracking(self) -> None:
        self._last_advanced_frame_index = -1

    def _advance_simulation_to_frame(
        self,
        frame_index: int,
    ) -> None:
        if self.simulation is None:
            return

        frame_index = self._validate_frame_index(
            frame_index
        )

        if frame_index <= self._last_advanced_frame_index:
            return

        visual_intervals = (
            frame_index
            - self._last_advanced_frame_index
        )
        physical_steps = (
            visual_intervals
            * self.simulation_steps_per_frame
        )

        self.simulation.run_steps(
            physical_steps
        )

        self._last_advanced_frame_index = frame_index

    def complete_requested_frames(self) -> None:
        """
        Complete the physical interval represented by create()/show().

        This is idempotent. Calling it more than once does not advance
        the simulation again.
        """

        if self.simulation is None:
            return

        if self._requested_frame_count <= 0:
            return

        final_frame_index = (
            self._requested_frame_count - 1
        )

        self._advance_simulation_to_frame(
            final_frame_index
        )

    def _advance_source(
        self,
        frame_index: int,
    ) -> None:
        frame_index = self._validate_frame_index(
            frame_index
        )

        if self.simulation is not None:
            self._advance_simulation_to_frame(
                frame_index
            )
            return

        if self.update_frame is not None:
            self.update_frame(frame_index)

    def _frame_title(
        self,
        frame_index: int,
        title: str | None,
    ) -> str:
        prefix = (
            title
            if title
            else (
                "ROIF Engine — "
                f"NetworkAnimator v{self.VERSION}"
            )
        )

        if self.simulation is not None:
            return (
                f"{prefix} — frame {frame_index} "
                f"— t={self.simulation.time:.4f} s "
                f"— step {self.simulation.step_index}"
            )

        return f"{prefix} — frame {frame_index}"

    def _draw_frame(
        self,
        frame_index: int,
        *,
        mode: str,
        show_reference: bool,
        show_labels: bool,
        show_colorbar: bool,
        title: str | None,
    ) -> tuple[Any, ...]:
        frame_index = self._validate_frame_index(
            frame_index
        )

        self._advance_source(frame_index)

        if self._axes is None:
            raise RuntimeError(
                "animation axes have not been initialized"
            )

        self._axes.clear()

        self.plotter.plot(
            mode=mode,
            show_reference=show_reference,
            show_labels=show_labels,
            show_colorbar=False,
            title=self._frame_title(
                frame_index,
                title,
            ),
            axes=self._axes,
        )

        return (
            tuple(self._axes.lines)
            + tuple(self._axes.collections)
        )

    def create(
        self,
        *,
        frames: int,
        interval_ms: float = 50.0,
        mode: str = "geometry",
        show_reference: bool = True,
        show_labels: bool = True,
        show_colorbar: bool = False,
        title: str | None = None,
        repeat: bool = False,
    ) -> FuncAnimation:
        frames = self._validate_frames(frames)
        interval_ms = self._validate_interval(
            interval_ms
        )

        self.plotter._validate_mode(mode)

        self.reset_frame_tracking()
        self._requested_frame_count = frames

        self._figure, self._axes = (
            self.plotter._create_axes()
        )

        self._animation = FuncAnimation(
            self._figure,
            lambda frame_index: self._draw_frame(
                frame_index,
                mode=mode,
                show_reference=show_reference,
                show_labels=show_labels,
                show_colorbar=show_colorbar,
                title=title,
            ),
            frames=range(frames),
            interval=interval_ms,
            repeat=bool(repeat),
            blit=False,
            cache_frame_data=False,
        )

        return self._animation

    def show(
        self,
        *,
        frames: int,
        interval_ms: float = 50.0,
        mode: str = "geometry",
        show_reference: bool = True,
        show_labels: bool = True,
        title: str | None = None,
        repeat: bool = False,
        complete_on_close: bool = True,
    ) -> FuncAnimation:
        """
        Open an interactive animation window.

        When complete_on_close is True, the requested physical interval
        is completed after the GUI window returns. Therefore a nominal
        300-frame run with 5 physical steps per frame always finishes
        at exactly 1500 physical steps, even if the GUI backend rendered
        only 298 or 299 unique frames before closing.
        """

        animation = self.create(
            frames=frames,
            interval_ms=interval_ms,
            mode=mode,
            show_reference=show_reference,
            show_labels=show_labels,
            show_colorbar=False,
            title=title,
            repeat=repeat,
        )

        plt.show()

        if complete_on_close:
            self.complete_requested_frames()

        return animation

    def save_gif(
        self,
        path: str | Path,
        *,
        frames: int,
        interval_ms: float = 50.0,
        mode: str = "geometry",
        show_reference: bool = True,
        show_labels: bool = True,
        title: str | None = None,
        repeat: bool = False,
        dpi: int = 120,
    ) -> Path:
        output_path = Path(path)

        if output_path.suffix.lower() != ".gif":
            raise ValueError(
                "GIF output path must end with .gif"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        animation = self.create(
            frames=frames,
            interval_ms=interval_ms,
            mode=mode,
            show_reference=show_reference,
            show_labels=show_labels,
            show_colorbar=False,
            title=title,
            repeat=repeat,
        )

        fps = max(
            1.0,
            1000.0 / float(interval_ms),
        )

        animation.save(
            output_path,
            writer="pillow",
            fps=fps,
            dpi=int(dpi),
        )

        self.complete_requested_frames()
        plt.close(self._figure)

        return output_path.resolve()
