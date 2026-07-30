from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "NetworkPlotter requires matplotlib. "
        "Install it with: python -m pip install matplotlib"
    ) from exc


class NetworkPlotter:
    """
    Static, read-only geometry visualizer for a ROIF Engine Network.

    Version 1.0 draws:
    - elements;
    - fixed and free nodes;
    - node identifiers;
    - stored reference geometry;
    - current geometry.

    One-, two-, and three-dimensional networks are supported.
    Reference geometry is captured when the plotter is created.
    """

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
            if not hasattr(node, "position"):
                raise TypeError("every node must provide position")

            position = np.asarray(node.position, dtype=float)

            if position.ndim != 1:
                raise ValueError("node positions must be one-dimensional vectors")
            if position.size not in (1, 2, 3):
                raise ValueError(
                    "NetworkPlotter supports only 1D, 2D, and 3D networks"
                )
            if not np.all(np.isfinite(position)):
                raise ValueError("node positions must contain finite values")

            dimensions.add(int(position.size))

        if len(dimensions) != 1:
            raise ValueError("all network nodes must have the same dimension")

        for element in self.network.elements:
            if not hasattr(element, "node_a") or not hasattr(element, "node_b"):
                raise TypeError("every element must provide node_a and node_b")
            if element.node_a not in self.network.nodes:
                raise ValueError("element.node_a is not part of the network")
            if element.node_b not in self.network.nodes:
                raise ValueError("element.node_b is not part of the network")

    @property
    def dimension(self) -> int:
        return int(
            np.asarray(self.network.nodes[0].position, dtype=float).size
        )

    def _capture_positions(self) -> dict[int, np.ndarray]:
        return {
            id(node): np.asarray(node.position, dtype=float).copy()
            for node in self.network.nodes
        }

    def set_reference_geometry(self) -> None:
        """Capture current node positions as the new reference geometry."""
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
    def _is_failed(element: Any) -> bool:
        material = getattr(element, "material", None)
        state = getattr(material, "state", None)
        return bool(getattr(state, "failed", False))

    def _create_axes(self) -> tuple[Any, Any]:
        figure = plt.figure()
        if self.dimension == 3:
            axes = figure.add_subplot(111, projection="3d")
        else:
            axes = figure.add_subplot(111)
        return figure, axes

    def _prepare_axes(self, axes: Any, title: str | None) -> None:
        axes.set_title(title or "ROIF Engine — NetworkPlotter v1.0")
        axes.set_xlabel("X")

        if self.dimension >= 2:
            axes.set_ylabel("Y")
        if self.dimension == 3:
            axes.set_zlabel("Z")

        axes.grid(True)

        if self.dimension != 3:
            axes.set_aspect("equal", adjustable="datalim")

    def _plot_segment(
        self,
        axes: Any,
        point_a: np.ndarray,
        point_b: np.ndarray,
        **kwargs: Any,
    ) -> None:
        if self.dimension == 1:
            axes.plot(
                [point_a[0], point_b[0]],
                [0.0, 0.0],
                **kwargs,
            )
        elif self.dimension == 2:
            axes.plot(
                [point_a[0], point_b[0]],
                [point_a[1], point_b[1]],
                **kwargs,
            )
        else:
            axes.plot(
                [point_a[0], point_b[0]],
                [point_a[1], point_b[1]],
                [point_a[2], point_b[2]],
                **kwargs,
            )

    def _scatter_nodes(
        self,
        axes: Any,
        nodes: list[Any],
        *,
        marker: str,
        label: str,
    ) -> None:
        if not nodes:
            return

        positions = [
            np.asarray(node.position, dtype=float)
            for node in nodes
        ]

        if self.dimension == 1:
            axes.scatter(
                [position[0] for position in positions],
                [0.0 for _ in positions],
                marker=marker,
                s=70,
                label=label,
                zorder=4,
            )
        elif self.dimension == 2:
            axes.scatter(
                [position[0] for position in positions],
                [position[1] for position in positions],
                marker=marker,
                s=70,
                label=label,
                zorder=4,
            )
        else:
            axes.scatter(
                [position[0] for position in positions],
                [position[1] for position in positions],
                [position[2] for position in positions],
                marker=marker,
                s=70,
                label=label,
                depthshade=False,
            )

    def _annotate_node(
        self,
        axes: Any,
        node: Any,
        fallback_index: int,
    ) -> None:
        position = np.asarray(node.position, dtype=float)
        label = self._node_label(node, fallback_index)

        if self.dimension == 1:
            axes.annotate(
                label,
                (position[0], 0.0),
                xytext=(6, 8),
                textcoords="offset points",
            )
        elif self.dimension == 2:
            axes.annotate(
                label,
                (position[0], position[1]),
                xytext=(6, 6),
                textcoords="offset points",
            )
        else:
            axes.text(
                position[0],
                position[1],
                position[2],
                f"  {label}",
            )

    def plot(
        self,
        *,
        show_reference: bool = True,
        show_labels: bool = True,
        title: str | None = None,
        axes: Any | None = None,
    ) -> tuple[Any, Any]:
        """Draw the network and return ``(figure, axes)``."""
        self._validate_network()

        if axes is None:
            figure, axes = self._create_axes()
        else:
            figure = axes.figure

        self._prepare_axes(axes, title)

        reference_label_added = False
        current_label_added = False
        failed_label_added = False

        for element in self.network.elements:
            if show_reference:
                self._plot_segment(
                    axes,
                    self.reference_position(element.node_a),
                    self.reference_position(element.node_b),
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.55,
                    label=(
                        "Reference geometry"
                        if not reference_label_added
                        else None
                    ),
                )
                reference_label_added = True

            current_a = np.asarray(element.node_a.position, dtype=float)
            current_b = np.asarray(element.node_b.position, dtype=float)

            if self._is_failed(element):
                self._plot_segment(
                    axes,
                    current_a,
                    current_b,
                    linestyle=":",
                    linewidth=2.0,
                    label=(
                        "Failed element"
                        if not failed_label_added
                        else None
                    ),
                )
                failed_label_added = True
            else:
                self._plot_segment(
                    axes,
                    current_a,
                    current_b,
                    linestyle="-",
                    linewidth=2.0,
                    label=(
                        "Current geometry"
                        if not current_label_added
                        else None
                    ),
                )
                current_label_added = True

        fixed_nodes = [
            node
            for node in self.network.nodes
            if bool(getattr(node, "fixed", False))
        ]
        free_nodes = [
            node
            for node in self.network.nodes
            if not bool(getattr(node, "fixed", False))
        ]

        self._scatter_nodes(
            axes,
            fixed_nodes,
            marker="s",
            label="Fixed nodes",
        )
        self._scatter_nodes(
            axes,
            free_nodes,
            marker="o",
            label="Free nodes",
        )

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

        figure.tight_layout()
        return figure, axes

    def show(
        self,
        *,
        show_reference: bool = True,
        show_labels: bool = True,
        title: str | None = None,
    ) -> tuple[Any, Any]:
        """Draw the network and open the Matplotlib window."""
        figure, axes = self.plot(
            show_reference=show_reference,
            show_labels=show_labels,
            title=title,
        )
        plt.show()
        return figure, axes

    def save(
        self,
        path: str | Path,
        *,
        show_reference: bool = True,
        show_labels: bool = True,
        title: str | None = None,
        dpi: int = 150,
    ) -> Path:
        """Save a static image and return its absolute path."""
        output_path = Path(path)

        if output_path.suffix == "":
            raise ValueError("output path must include a file extension")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, _ = self.plot(
            show_reference=show_reference,
            show_labels=show_labels,
            title=title,
        )
        figure.savefig(
            output_path,
            dpi=int(dpi),
            bbox_inches="tight",
        )
        plt.close(figure)

        return output_path.resolve()
