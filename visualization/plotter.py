from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
except ImportError as exc:
    raise ImportError(
        "NetworkPlotter requires matplotlib. Install it with: "
        "python -m pip install matplotlib"
    ) from exc

from .styles import (
    CURRENT_ELEMENT_STYLE,
    FAILED_ELEMENT_STYLE,
    FIXED_NODE_STYLE,
    FORCE_COLORBAR_LABEL,
    FORCE_COLORMAP,
    FORCE_MAX_LINEWIDTH,
    FORCE_MIN_LINEWIDTH,
    FREE_NODE_STYLE,
    GRID_STYLE,
    MATERIAL_MODE_STYLES,
    NODE_LABEL_STYLE,
    REFERENCE_ELEMENT_STYLE,
    element_role_label,
    element_role_style,
)

from .viewer import ViewerState

PlotMode = Literal[
    "geometry", "force", "damage", "fatigue", "integrity", "remodeling"
]


class NetworkPlotter:
    """Static, read-only visualizer for a ROIF Engine Network."""

    VERSION = "1.3"
    SUPPORTED_MODES = (
        "geometry", "force", "damage", "fatigue", "integrity", "remodeling"
    )
    MATERIAL_MODES = ("damage", "fatigue", "integrity", "remodeling")

    def __init__(self, network: Any) -> None:
        self.network = network
        self._validate_network()
        self._reference_positions = self._capture_positions()

    def _validate_network(self) -> None:
        if not hasattr(self.network, "nodes"):
            raise TypeError("network must provide a nodes collection")
        if not hasattr(self.network, "elements"):
            raise TypeError("network must provide an elements collection")
        if not self.network.nodes:
            raise ValueError("network must contain at least one node")

        dimensions: set[int] = set()
        for node in self.network.nodes:
            position = np.asarray(node.position, dtype=float)
            if position.ndim != 1:
                raise ValueError("node positions must be one-dimensional vectors")
            if position.size not in (1, 2, 3):
                raise ValueError("NetworkPlotter supports only 1D, 2D, and 3D networks")
            if not np.all(np.isfinite(position)):
                raise ValueError("node positions must contain finite values")
            dimensions.add(int(position.size))

        if len(dimensions) != 1:
            raise ValueError("all network nodes must have the same dimension")

        for element in self.network.elements:
            if element.node_a not in self.network.nodes:
                raise ValueError("element.node_a is not part of the network")
            if element.node_b not in self.network.nodes:
                raise ValueError("element.node_b is not part of the network")
            if not hasattr(element, "material"):
                raise TypeError("every element must provide material")

    @staticmethod
    def _validate_mode(mode: str) -> PlotMode:
        if mode not in NetworkPlotter.SUPPORTED_MODES:
            supported = ", ".join(NetworkPlotter.SUPPORTED_MODES)
            raise ValueError(
                f"unsupported plot mode {mode!r}; expected one of: {supported}"
            )
        return mode  # type: ignore[return-value]

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.network.nodes[0].position, dtype=float).size)

    def _capture_positions(self) -> dict[int, np.ndarray]:
        return {
            id(node): np.asarray(node.position, dtype=float).copy()
            for node in self.network.nodes
        }

    def set_reference_geometry(self) -> None:
        self._validate_network()
        self._reference_positions = self._capture_positions()

    def reference_position(self, node: Any) -> np.ndarray:
        try:
            return self._reference_positions[id(node)].copy()
        except KeyError as exc:
            raise ValueError(
                "node was added after reference geometry was captured; "
                "call set_reference_geometry()"
            ) from exc

    @staticmethod
    def _node_label(node: Any, fallback_index: int) -> str:
        return str(getattr(node, "id", fallback_index))

    @staticmethod
    def _element_label(element: Any, fallback_index: int) -> str:
        return str(getattr(element, "id", fallback_index))

    @staticmethod
    def _is_failed(element: Any) -> bool:
        state = getattr(getattr(element, "material", None), "state", None)
        return bool(getattr(state, "failed", False))

    @staticmethod
    def element_force(element: Any) -> float:
        last_force = getattr(element, "last_force", None)
        total = getattr(last_force, "total", None)
        if total is None:
            state = getattr(getattr(element, "material", None), "state", None)
            total = getattr(state, "total_force_component", 0.0)
        force = float(total)
        if not np.isfinite(force):
            raise ValueError("element force must be finite")
        return force

    @classmethod
    def element_force_stimulus(cls, element: Any) -> float:
        material = getattr(element, "material", None)
        method = getattr(material, "normalized_force_stimulus", None)
        if method is None:
            raise TypeError("element material must provide normalized_force_stimulus()")
        stimulus = float(method(force=cls.element_force(element)))
        if not np.isfinite(stimulus):
            raise ValueError("element force stimulus must be finite")
        return max(0.0, stimulus)

    @staticmethod
    def element_material_value(element: Any, mode: str) -> float:
        material = getattr(element, "material", None)
        state = getattr(material, "state", None)
        if material is None or state is None:
            raise TypeError("element must provide material state")

        if mode == "integrity":
            method = getattr(material, "integrity", None)
            if method is None:
                raise TypeError("element material must provide integrity()")
            value = float(method())
        elif mode in ("damage", "fatigue", "remodeling"):
            value = float(getattr(state, mode))
        else:
            raise ValueError(f"unsupported material mode: {mode!r}")

        if not np.isfinite(value):
            raise ValueError(f"material {mode} must be finite")
        return value

    def _create_axes(self) -> tuple[Any, Any]:
        figure = plt.figure()
        axes = (
            figure.add_subplot(111, projection="3d")
            if self.dimension == 3
            else figure.add_subplot(111)
        )
        return figure, axes

    def _prepare_axes(self, axes: Any, title: str | None, mode: PlotMode) -> None:
        axes.set_title(
            title or f"ROIF Engine Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ NetworkPlotter v{self.VERSION} ({mode} mode)"
        )
        axes.set_xlabel("X")
        if self.dimension >= 2:
            axes.set_ylabel("Y")
        if self.dimension == 3:
            axes.set_zlabel("Z")
        axes.grid(True, **GRID_STYLE)
        if self.dimension != 3:
            axes.set_aspect("equal", adjustable="datalim")

    def _plot_segment(self, axes: Any, point_a: np.ndarray, point_b: np.ndarray, **kwargs: Any) -> Any:
        if self.dimension == 1:
            lines = axes.plot([point_a[0], point_b[0]], [0.0, 0.0], **kwargs)
        elif self.dimension == 2:
            lines = axes.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], **kwargs)
        else:
            lines = axes.plot(
                [point_a[0], point_b[0]],
                [point_a[1], point_b[1]],
                [point_a[2], point_b[2]],
                **kwargs,
            )
        return lines[0]

    def _scatter_nodes(self, axes: Any, nodes: list[Any], *, style: dict[str, Any], label: str) -> None:
        if not nodes:
            return
        positions = [np.asarray(node.position, dtype=float) for node in nodes]
        if self.dimension == 1:
            axes.scatter([p[0] for p in positions], [0.0 for _ in positions], label=label, **style)
        elif self.dimension == 2:
            axes.scatter([p[0] for p in positions], [p[1] for p in positions], label=label, **style)
        else:
            axes.scatter(
                [p[0] for p in positions],
                [p[1] for p in positions],
                [p[2] for p in positions],
                label=label,
                depthshade=False,
                **style,
            )

    def _annotate_node(self, axes: Any, node: Any, fallback_index: int) -> None:
        position = np.asarray(node.position, dtype=float)
        label = self._node_label(node, fallback_index)
        if self.dimension == 1:
            axes.annotate(label, (position[0], 0.0), xytext=(6, 8), textcoords="offset points", **NODE_LABEL_STYLE)
        elif self.dimension == 2:
            axes.annotate(label, (position[0], position[1]), xytext=(6, 6), textcoords="offset points", **NODE_LABEL_STYLE)
        else:
            axes.text(position[0], position[1], position[2], f"  {label}", **NODE_LABEL_STYLE)

    @staticmethod
    def _scaled_force_linewidth(force_magnitude: float, maximum_force: float) -> float:
        if maximum_force <= 0.0:
            return float(FORCE_MIN_LINEWIDTH)
        fraction = float(np.clip(force_magnitude / maximum_force, 0.0, 1.0))
        return float(
            FORCE_MIN_LINEWIDTH
            + fraction * (FORCE_MAX_LINEWIDTH - FORCE_MIN_LINEWIDTH)
        )

    def mechanical_snapshot(self) -> dict[int, dict[str, float]]:
        snapshot: dict[int, dict[str, float]] = {}
        for element in self.network.elements:
            force = self.element_force(element)
            snapshot[id(element)] = {
                "force": force,
                "force_magnitude": abs(force),
                "force_stimulus": self.element_force_stimulus(element),
            }
        return snapshot

    def material_snapshot(self, mode: str) -> dict[int, float]:
        if mode not in self.MATERIAL_MODES:
            raise ValueError(f"unsupported material mode: {mode!r}")
        return {
            id(element): self.element_material_value(element, mode)
            for element in self.network.elements
        }

    @staticmethod
    def _add_colorbar(figure: Any, axes: Any, *, normalization: Normalize, colormap: Any, label: str) -> None:
        scalar_mappable = ScalarMappable(norm=normalization, cmap=colormap)
        scalar_mappable.set_array([])
        colorbar = figure.colorbar(scalar_mappable, ax=axes, pad=0.02)
        colorbar.set_label(label)

    def plot(
        self,
        *,
        mode: PlotMode = "geometry",
        show_reference: bool = True,
        show_labels: bool = True,
        show_colorbar: bool = True,
        title: str | None = None,
        axes: Any | None = None,
    ) -> tuple[Any, Any]:
        self._validate_network()
        mode = self._validate_mode(mode)

        if axes is None:
            figure, axes = self._create_axes()
        else:
            figure = axes.figure

        self._prepare_axes(axes, title, mode)

        mechanical = self.mechanical_snapshot() if mode == "force" else {}
        material_values = self.material_snapshot(mode) if mode in self.MATERIAL_MODES else {}

        maximum_force = 0.0
        normalization = None
        colormap = None
        colorbar_label = ""

        if mode == "force":
            maximum_force = max((v["force_magnitude"] for v in mechanical.values()), default=0.0)
            maximum_stimulus = max((v["force_stimulus"] for v in mechanical.values()), default=0.0)
            normalization = Normalize(vmin=0.0, vmax=max(1.0, maximum_stimulus), clip=True)
            colormap = colormaps.get_cmap(FORCE_COLORMAP)
            colorbar_label = FORCE_COLORBAR_LABEL
        elif mode in self.MATERIAL_MODES:
            style = MATERIAL_MODE_STYLES[mode]
            normalization = Normalize(vmin=float(style["vmin"]), vmax=float(style["vmax"]), clip=True)
            colormap = colormaps.get_cmap(str(style["cmap"]))
            colorbar_label = str(style["label"])

        reference_label_added = False
        current_label_added = False
        failed_label_added = False

        for index, element in enumerate(self.network.elements):
            element_label = self._element_label(element, index)

            if show_reference:
                reference_style = dict(REFERENCE_ELEMENT_STYLE)
                reference_style["label"] = "Reference geometry" if not reference_label_added else None
                line = self._plot_segment(
                    axes,
                    self.reference_position(element.node_a),
                    self.reference_position(element.node_b),
                    **reference_style,
                )
                line.set_gid(f"reference:{element_label}")
                reference_label_added = True

            current_a = np.asarray(element.node_a.position, dtype=float)
            current_b = np.asarray(element.node_b.position, dtype=float)

            if self._is_failed(element):
                failed_style = dict(FAILED_ELEMENT_STYLE)
                failed_style["label"] = "Failed element" if not failed_label_added else None
                line = self._plot_segment(axes, current_a, current_b, **failed_style)
                line.set_gid(f"failed:{element_label}")
                failed_label_added = True
                continue

            role = ViewerState.element_role(element)

            if mode == "geometry":
                current_style = element_role_style(role)
            else:
                current_style = dict(CURRENT_ELEMENT_STYLE)

            if mode == "force":
                values = mechanical[id(element)]
                current_style["color"] = colormap(normalization(values["force_stimulus"]))
                current_style["linewidth"] = self._scaled_force_linewidth(
                    values["force_magnitude"], maximum_force
                )
            elif mode in self.MATERIAL_MODES:
                current_style["color"] = colormap(normalization(material_values[id(element)]))

            if mode == "geometry":
                current_style["label"] = element_role_label(role)
            else:
                current_style["label"] = (
                    "Current geometry"
                    if not current_label_added
                    else None
                )
            line = self._plot_segment(axes, current_a, current_b, **current_style)
            line.set_gid(f"current:{element_label}")
            current_label_added = True

        fixed_nodes = [node for node in self.network.nodes if bool(getattr(node, "fixed", False))]
        free_nodes = [node for node in self.network.nodes if not bool(getattr(node, "fixed", False))]
        self._scatter_nodes(axes, fixed_nodes, style=FIXED_NODE_STYLE, label="Fixed nodes")
        self._scatter_nodes(axes, free_nodes, style=FREE_NODE_STYLE, label="Free nodes")

        if show_labels:
            for index, node in enumerate(self.network.nodes):
                self._annotate_node(axes, node, index)

        handles, labels = axes.get_legend_handles_labels()
        if handles:
            unique: dict[str, Any] = {}
            for handle, label in zip(handles, labels):
                if label and label not in unique:
                    unique[label] = handle
            axes.legend(unique.values(), unique.keys())

        if mode != "geometry" and show_colorbar and normalization is not None and colormap is not None:
            self._add_colorbar(
                figure,
                axes,
                normalization=normalization,
                colormap=colormap,
                label=colorbar_label,
            )

        figure.tight_layout()
        return figure, axes

    def show(self, **kwargs: Any) -> tuple[Any, Any]:
        figure, axes = self.plot(**kwargs)
        plt.show()
        return figure, axes

    def save(self, path: str | Path, *, dpi: int = 150, **kwargs: Any) -> Path:
        output_path = Path(path)
        if output_path.suffix == "":
            raise ValueError("output path must include a file extension")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, _ = self.plot(**kwargs)
        figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        return output_path.resolve()
