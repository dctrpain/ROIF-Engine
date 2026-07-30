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

from .plotter import NetworkPlotter

FrameCallback = Callable[[int], None]


class NetworkAnimator:
    """Animated visualization layer for a ROIF Engine Network."""

    VERSION = "2.0"

    def __init__(
        self,
        network: Any,
        *,
        update_frame: FrameCallback | None = None,
    ) -> None:
        self.network = network
        self.update_frame = update_frame
        self.plotter = NetworkPlotter(network)
        self._animation: FuncAnimation | None = None
        self._figure: Any | None = None
        self._axes: Any | None = None

    @staticmethod
    def _validate_frames(frames: int) -> int:
        frames = int(frames)
        if frames <= 0:
            raise ValueError("frames must be positive")
        return frames

    @staticmethod
    def _validate_interval(interval_ms: float) -> float:
        interval_ms = float(interval_ms)
        if not np.isfinite(interval_ms):
            raise ValueError("interval_ms must be finite")
        if interval_ms <= 0.0:
            raise ValueError("interval_ms must be positive")
        return interval_ms

    def set_reference_geometry(self) -> None:
        self.plotter.set_reference_geometry()

    def _draw_frame(
        self,
        frame_index: int,
        *,
        mode: str,
        show_reference: bool,
        show_labels: bool,
        title: str | None,
    ) -> tuple[Any, ...]:
        if self.update_frame is not None:
            self.update_frame(frame_index)

        if self._axes is None:
            raise RuntimeError("animation axes have not been initialized")

        self._axes.clear()
        frame_title = (
            f"{title} — frame {frame_index}"
            if title
            else f"ROIF Engine — NetworkAnimator v{self.VERSION} — frame {frame_index}"
        )

        self.plotter.plot(
            mode=mode,
            show_reference=show_reference,
            show_labels=show_labels,
            show_colorbar=False,
            title=frame_title,
            axes=self._axes,
        )

        return tuple(self._axes.lines) + tuple(self._axes.collections)

    def create(
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
        frames = self._validate_frames(frames)
        interval_ms = self._validate_interval(interval_ms)
        self.plotter._validate_mode(mode)

        self._figure, self._axes = self.plotter._create_axes()
        self._animation = FuncAnimation(
            self._figure,
            lambda frame_index: self._draw_frame(
                frame_index,
                mode=mode,
                show_reference=show_reference,
                show_labels=show_labels,
                title=title,
            ),
            frames=frames,
            interval=interval_ms,
            repeat=bool(repeat),
            blit=False,
        )
        return self._animation

    def show(self, **kwargs: Any) -> FuncAnimation:
        animation = self.create(**kwargs)
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
        output_path = Path(path)
        if output_path.suffix.lower() != ".gif":
            raise ValueError("GIF output path must end with .gif")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        animation = self.create(
            frames=frames,
            interval_ms=interval_ms,
            mode=mode,
            show_reference=show_reference,
            show_labels=show_labels,
            title=title,
            repeat=repeat,
        )
        animation.save(
            output_path,
            writer="pillow",
            fps=max(1.0, 1000.0 / float(interval_ms)),
            dpi=int(dpi),
        )
        plt.close(self._figure)
        return output_path.resolve()
