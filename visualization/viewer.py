from __future__ import annotations

"""
State and object-model adapter for ROIF Viewer.

The viewer does not run physics and does not modify a Network. It receives
either a live Network or an immutable RecordingFrame and exposes a stable,
read-only interface for visualization layers.

This module intentionally contains no Matplotlib code. Rendering remains the
responsibility of NetworkPlotter, NetworkAnimator, and future overlay modules.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

import numpy as np

from core.recorder import RecordingFrame


class ViewerError(RuntimeError):
    """Base error raised by the ROIF Viewer state layer."""


class ViewerMode(str, Enum):
    """Primary data layer displayed by the viewer."""

    GEOMETRY = "geometry"
    FORCE = "force"
    STRAIN = "strain"
    STRESS = "stress"
    DAMAGE = "damage"
    FATIGUE = "fatigue"
    INTEGRITY = "integrity"
    REMODELING = "remodeling"
    CAPACITY = "capacity"
    RESERVE = "reserve"
    CASCADE = "cascade"


class ElementVisualRole(str, Enum):
    """
    Mechanical role used only for visualization.

    This role is derived from the real Element object. It does not alter the
    element, its material, or solver behavior.
    """

    RIGID = "rigid"
    TENSION = "tension"
    ACTIVE_TENSION = "active_tension"
    COMPRESSION = "compression"
    GENERIC = "generic"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ViewerSelection:
    """Currently selected object in the viewer."""

    node_id: Any | None = None
    element_id: Any | None = None

    def __post_init__(self) -> None:
        if self.node_id is not None and self.element_id is not None:
            raise ValueError(
                "only one node or element may be selected at a time"
            )

    @property
    def empty(self) -> bool:
        return self.node_id is None and self.element_id is None


@dataclass(frozen=True, slots=True)
class ViewerLayers:
    """Visibility switches for independent viewer overlays."""

    geometry: bool = True
    reference_geometry: bool = True
    labels: bool = True
    nodes: bool = True
    elements: bool = True
    forces: bool = False
    velocities: bool = False
    material_state: bool = False
    capacity: bool = False
    reserve: bool = False
    cascade: bool = False


@dataclass(frozen=True, slots=True)
class ViewerState:
    """
    Read-only state consumed by ROIF visualization components.

    Exactly one source is active:

    - frame: playback of a recorded simulation state;
    - network: direct inspection of a live or static Network.

    The underlying object model remains the source of truth.
    """

    frame: RecordingFrame | None = None
    network: Any | None = None
    mode: ViewerMode = ViewerMode.GEOMETRY
    selection: ViewerSelection = field(
        default_factory=ViewerSelection
    )
    layers: ViewerLayers = field(
        default_factory=ViewerLayers
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    VERSION = "3.0.0"

    def __post_init__(self) -> None:
        if self.frame is None and self.network is None:
            raise ValueError(
                "ViewerState requires a RecordingFrame or Network"
            )

        if self.frame is not None and self.network is not None:
            if self.network is not self.frame.network:
                raise ValueError(
                    "network must be frame.network when both are provided"
                )

        self._validate_network(self.source_network)

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @classmethod
    def from_frame(
        cls,
        frame: RecordingFrame,
        *,
        mode: ViewerMode | str = ViewerMode.GEOMETRY,
        metadata: dict[str, Any] | None = None,
    ) -> ViewerState:
        if not isinstance(frame, RecordingFrame):
            raise TypeError(
                "frame must be an instance of RecordingFrame"
            )

        merged_metadata = dict(frame.metadata)
        merged_metadata.update(metadata or {})

        return cls(
            frame=frame,
            network=None,
            mode=cls._coerce_mode(mode),
            metadata=merged_metadata,
        )

    @classmethod
    def from_network(
        cls,
        network: Any,
        *,
        mode: ViewerMode | str = ViewerMode.GEOMETRY,
        metadata: dict[str, Any] | None = None,
    ) -> ViewerState:
        return cls(
            frame=None,
            network=network,
            mode=cls._coerce_mode(mode),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _coerce_mode(
        mode: ViewerMode | str,
    ) -> ViewerMode:
        if isinstance(mode, ViewerMode):
            return mode

        try:
            return ViewerMode(str(mode))
        except ValueError as exc:
            supported = ", ".join(
                item.value for item in ViewerMode
            )
            raise ValueError(
                f"unsupported viewer mode {mode!r}; "
                f"expected one of: {supported}"
            ) from exc

    @staticmethod
    def _validate_network(network: Any) -> None:
        if network is None:
            raise TypeError("network cannot be None")

        if not hasattr(network, "nodes"):
            raise TypeError(
                "network must provide a nodes collection"
            )

        if not hasattr(network, "elements"):
            raise TypeError(
                "network must provide an elements collection"
            )

        nodes = list(network.nodes)
        elements = list(network.elements)

        for node in nodes:
            if not hasattr(node, "position"):
                raise TypeError(
                    "every node must provide position"
                )

            position = np.asarray(
                node.position,
                dtype=float,
            )

            if position.ndim != 1:
                raise ValueError(
                    "node position must be a one-dimensional vector"
                )

            if position.size not in (1, 2, 3):
                raise ValueError(
                    "ROIF Viewer supports only 1D, 2D, and 3D networks"
                )

            if not np.all(np.isfinite(position)):
                raise ValueError(
                    "node position must contain finite values"
                )

        for element in elements:
            if not hasattr(element, "node_a"):
                raise TypeError(
                    "every element must provide node_a"
                )

            if not hasattr(element, "node_b"):
                raise TypeError(
                    "every element must provide node_b"
                )

            if element.node_a not in nodes:
                raise ValueError(
                    "element.node_a is not part of the network"
                )

            if element.node_b not in nodes:
                raise ValueError(
                    "element.node_b is not part of the network"
                )

    @property
    def source_network(self) -> Any:
        if self.frame is not None:
            return self.frame.network

        return self.network

    @property
    def is_recorded(self) -> bool:
        return self.frame is not None

    @property
    def frame_index(self) -> int | None:
        if self.frame is None:
            return None
        return int(self.frame.index)

    @property
    def step_index(self) -> int | None:
        if self.frame is not None:
            return int(self.frame.step_index)

        value = getattr(
            self.source_network,
            "step_index",
            None,
        )
        return None if value is None else int(value)

    @property
    def time(self) -> float | None:
        if self.frame is not None:
            return float(self.frame.time)

        value = getattr(
            self.source_network,
            "time",
            None,
        )
        return None if value is None else float(value)

    @property
    def nodes(self) -> tuple[Any, ...]:
        return tuple(self.source_network.nodes)

    @property
    def elements(self) -> tuple[Any, ...]:
        return tuple(self.source_network.elements)

    @property
    def dimension(self) -> int:
        if not self.nodes:
            raise ViewerError(
                "network must contain at least one node"
            )

        return int(
            np.asarray(
                self.nodes[0].position,
                dtype=float,
            ).size
        )

    def with_mode(
        self,
        mode: ViewerMode | str,
    ) -> ViewerState:
        return replace(
            self,
            mode=self._coerce_mode(mode),
        )

    def with_layers(
        self,
        **changes: bool,
    ) -> ViewerState:
        valid_names = set(
            ViewerLayers.__dataclass_fields__
        )

        unknown = set(changes) - valid_names
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"unknown viewer layer fields: {names}"
            )

        for name, value in changes.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"layer {name!r} must be bool"
                )

        return replace(
            self,
            layers=replace(
                self.layers,
                **changes,
            ),
        )

    def clear_selection(self) -> ViewerState:
        return replace(
            self,
            selection=ViewerSelection(),
        )

    def select_node(
        self,
        node_id: Any,
    ) -> ViewerState:
        self.node_by_id(node_id)

        return replace(
            self,
            selection=ViewerSelection(
                node_id=node_id,
            ),
        )

    def select_element(
        self,
        element_id: Any,
    ) -> ViewerState:
        self.element_by_id(element_id)

        return replace(
            self,
            selection=ViewerSelection(
                element_id=element_id,
            ),
        )

    def node_by_id(
        self,
        node_id: Any,
    ) -> Any:
        for index, node in enumerate(self.nodes):
            candidate = getattr(
                node,
                "id",
                index,
            )

            if candidate == node_id:
                return node

        raise KeyError(
            f"unknown node id: {node_id!r}"
        )

    def element_by_id(
        self,
        element_id: Any,
    ) -> Any:
        for index, element in enumerate(self.elements):
            candidate = getattr(
                element,
                "id",
                index,
            )

            if candidate == element_id:
                return element

        raise KeyError(
            f"unknown element id: {element_id!r}"
        )

    @staticmethod
    def element_role(
        element: Any,
    ) -> ElementVisualRole:
        """
        Derive a visual role from the actual Element and Material objects.

        Priority:

        1. failed;
        2. disabled;
        3. active muscle-like material;
        4. tension-only;
        5. compression-only;
        6. explicitly rigid material or element;
        7. generic axial element.
        """

        material = getattr(
            element,
            "material",
            None,
        )
        material_state = getattr(
            material,
            "state",
            None,
        )

        if bool(
            getattr(
                material_state,
                "failed",
                False,
            )
        ):
            return ElementVisualRole.FAILED

        if not bool(
            getattr(
                element,
                "enabled",
                True,
            )
        ):
            return ElementVisualRole.DISABLED

        material_name = (
            type(material).__name__.lower()
            if material is not None
            else ""
        )

        material_label = str(
            getattr(
                material,
                "name",
                "",
            )
        ).lower()

        type_label = (
            material_name
            + " "
            + material_label
        )

        active_force = float(
            getattr(
                getattr(
                    element,
                    "force_components",
                    None,
                ),
                "active",
                0.0,
            )
            or 0.0
        )

        activation = float(
            getattr(
                material_state,
                "activation",
                0.0,
            )
            or 0.0
        )

        if (
            "muscle" in type_label
            or active_force != 0.0
            or activation > 0.0
        ):
            return ElementVisualRole.ACTIVE_TENSION

        if bool(
            getattr(
                element,
                "tension_only",
                False,
            )
        ):
            return ElementVisualRole.TENSION

        if bool(
            getattr(
                element,
                "compression_only",
                False,
            )
        ):
            return ElementVisualRole.COMPRESSION

        explicit_role = str(
            getattr(
                element,
                "mechanical_role",
                "",
            )
        ).lower()

        if explicit_role in {
            "rigid",
            "strut",
            "bar",
            "rod",
        }:
            return ElementVisualRole.RIGID

        if any(
            marker in type_label
            for marker in (
                "rigid",
                "strut",
                "rod",
                "bone",
            )
        ):
            return ElementVisualRole.RIGID

        return ElementVisualRole.GENERIC

    def node_snapshot(
        self,
        node: Any,
    ) -> dict[str, Any]:
        position = np.asarray(
            node.position,
            dtype=float,
        )

        velocity = np.asarray(
            getattr(
                node,
                "velocity",
                np.zeros_like(position),
            ),
            dtype=float,
        )

        force = np.asarray(
            getattr(
                node,
                "force",
                np.zeros_like(position),
            ),
            dtype=float,
        )

        return {
            "id": getattr(node, "id", None),
            "position": position.copy(),
            "velocity": velocity.copy(),
            "force": force.copy(),
            "mass": float(
                getattr(node, "mass", 0.0)
            ),
            "fixed": bool(
                getattr(node, "fixed", False)
            ),
        }

    def element_snapshot(
        self,
        element: Any,
    ) -> dict[str, Any]:
        material = getattr(
            element,
            "material",
            None,
        )
        material_state = getattr(
            material,
            "state",
            None,
        )
        parameters = getattr(
            material,
            "parameters",
            None,
        )

        current_length = self._safe_call(
            element,
            "current_length",
            default=0.0,
        )

        strain = self._safe_call(
            element,
            "strain",
            default=0.0,
        )

        force_value = self._element_force(element)

        reference_force = float(
            getattr(
                parameters,
                "reference_force",
                1.0,
            )
            or 1.0
        )

        load_ratio = (
            abs(force_value) / reference_force
            if reference_force > 0.0
            else 0.0
        )

        capacity = max(
            0.0,
            1.0 - float(
                getattr(
                    material_state,
                    "damage",
                    0.0,
                )
                or 0.0
            ),
        )

        reserve = capacity - load_ratio

        return {
            "id": getattr(element, "id", None),
            "name": getattr(element, "name", None),
            "node_a_id": getattr(
                element.node_a,
                "id",
                None,
            ),
            "node_b_id": getattr(
                element.node_b,
                "id",
                None,
            ),
            "role": self.element_role(element).value,
            "enabled": bool(
                getattr(element, "enabled", True)
            ),
            "tension_only": bool(
                getattr(
                    element,
                    "tension_only",
                    False,
                )
            ),
            "compression_only": bool(
                getattr(
                    element,
                    "compression_only",
                    False,
                )
            ),
            "rest_length": float(
                getattr(element, "rest_length", 0.0)
            ),
            "current_length": float(current_length),
            "strain": float(strain),
            "force": float(force_value),
            "load_ratio": float(load_ratio),
            "capacity": float(capacity),
            "reserve": float(reserve),
            "damage": float(
                getattr(
                    material_state,
                    "damage",
                    0.0,
                )
                or 0.0
            ),
            "fatigue": float(
                getattr(
                    material_state,
                    "fatigue",
                    0.0,
                )
                or 0.0
            ),
            "integrity": float(
                getattr(
                    material_state,
                    "integrity",
                    capacity,
                )
                or 0.0
            ),
            "failed": bool(
                getattr(
                    material_state,
                    "failed",
                    False,
                )
            ),
            "material_type": (
                type(material).__name__
                if material is not None
                else None
            ),
            "material_name": getattr(
                material,
                "name",
                None,
            ),
        }

    def network_snapshot(
        self,
    ) -> dict[str, Any]:
        return {
            "viewer_version": self.VERSION,
            "recorded": self.is_recorded,
            "frame_index": self.frame_index,
            "step_index": self.step_index,
            "time": self.time,
            "mode": self.mode.value,
            "dimension": self.dimension,
            "nodes": [
                self.node_snapshot(node)
                for node in self.nodes
            ],
            "elements": [
                self.element_snapshot(element)
                for element in self.elements
            ],
            "selection": {
                "node_id": self.selection.node_id,
                "element_id": self.selection.element_id,
            },
            "layers": {
                name: getattr(self.layers, name)
                for name in ViewerLayers.__dataclass_fields__
            },
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _safe_call(
        obj: Any,
        method_name: str,
        *,
        default: float,
    ) -> float:
        method = getattr(
            obj,
            method_name,
            None,
        )

        if not callable(method):
            return float(default)

        try:
            value = float(method())
        except (TypeError, ValueError, AttributeError):
            return float(default)

        if not np.isfinite(value):
            return float(default)

        return value

    @staticmethod
    def _element_force(
        element: Any,
    ) -> float:
        for name in (
            "last_force",
            "force",
            "total_force",
            "axial_force",
        ):
            value = getattr(
                element,
                name,
                None,
            )

            if value is None or callable(value):
                continue

            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(scalar):
                return scalar

        components = getattr(
            element,
            "force_components",
            None,
        )

        if components is not None:
            value = getattr(
                components,
                "total",
                None,
            )

            if value is not None:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    scalar = 0.0

                if np.isfinite(scalar):
                    return scalar

        compute_force = getattr(
            element,
            "compute_force",
            None,
        )

        if callable(compute_force):
            try:
                value = compute_force(
                    update_material=False,
                    include_active=True,
                )
            except TypeError:
                try:
                    value = compute_force()
                except Exception:
                    return 0.0
            except Exception:
                return 0.0

            try:
                scalar = float(value)
            except (TypeError, ValueError):
                return 0.0

            if np.isfinite(scalar):
                return scalar

        return 0.0


def viewer_states_from_frames(
    frames: Iterable[RecordingFrame],
    *,
    mode: ViewerMode | str = ViewerMode.GEOMETRY,
) -> tuple[ViewerState, ...]:
    """Convert recording frames into immutable viewer states."""

    resolved_mode = ViewerState._coerce_mode(mode)

    states: list[ViewerState] = []

    for frame in frames:
        states.append(
            ViewerState.from_frame(
                frame,
                mode=resolved_mode,
            )
        )

    return tuple(states)


__all__ = [
    "ElementVisualRole",
    "ViewerError",
    "ViewerLayers",
    "ViewerMode",
    "ViewerSelection",
    "ViewerState",
    "viewer_states_from_frames",
]