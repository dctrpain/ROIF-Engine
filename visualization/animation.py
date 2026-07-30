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

    NetworkAnimator v2.2 supports two execution sources:

    1. Simulation-driven mode
       A Simulation advances physical time and updates the network.

    2. Callback mode
       A legacy update_frame(frame_index) callback updates the network.

    The animator never implements mechanics. It only advances the
    configured source and redraws the resulting network state through
    NetworkPlotter.
    """

    VERSION = "2.2"

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

        if (
            simulation is not None
            and update_frame is not None
        ):
            raise ValueError(
                "provide either simulation or update_frame, "
                "not both"
            )

        if (
            update_frame is not None
            and not callable(update_frame)
        ):
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

    @classmethod
    def from_simulation(
        cls,
        simulation: Simulation,
        *,
        simulation_steps_per_frame: int = 1,
    ) -> "NetworkAnimator":
        """
        Construct an animator directly from Simulation Engine.
        """

        return cls(
            simulation=simulation,
            simulation_steps_per_frame=(
                simulation_steps_per_frame
            ),
        )

    @property
    def is_simulation_driven(self) -> bool:
        return self.simulation is not None

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

    def set_reference_geometry(self) -> None:
        """
        Capture current geometry as visual reference.
        """

        self.plotter.set_reference_geometry()

    def _advance_source(
        self,
        frame_index: int,
    ) -> None:
        if self.simulation is not None:
            for _ in range(
                self.simulation_steps_per_frame
            ):
                self.simulation.advance_frame(
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
        """
        Build and return a Matplotlib FuncAnimation.

        Dynamic colorbars remain disabled because per-frame colorbar
        recreation causes duplicated axes and memory growth.
        """

        frames = self._validate_frames(frames)
        interval_ms = self._validate_interval(
            interval_ms
        )

        self.plotter._validate_mode(mode)

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
            frames=frames,
            interval=interval_ms,
            repeat=bool(repeat),
            blit=False,
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
    ) -> FuncAnimation:
        """
        Create the animation and open the Matplotlib window.
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
        """
        Save animation as GIF using Matplotlib's Pillow writer.
        """

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

        plt.close(self._figure)

        return output_path.resolve()
