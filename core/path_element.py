from __future__ import annotations

from typing import Any

import numpy as np

from .element import Element
from .material import Material
from .path_geometry import PathGeometry


class PathElement(Element):
    """
    Axial material element acting along a multipoint path.

    The constitutive/material lifecycle is inherited from Element.
    Geometry and force transmission are provided by PathGeometry.

    A PathElement owns one scalar material state and therefore one
    scalar path tension, regardless of the number of path segments.
    """

    def __init__(
        self,
        path: PathGeometry,
        *,
        material: Material | None = None,
        rest_length: float | None = None,
        element_id: int | str | None = None,
        name: str | None = None,
        stiffness: float = 1.0,
        damping: float = 0.0,
        pretension: float = 0.0,
        tension_only: bool = False,
        compression_only: bool = False,
        enabled: bool = True,
        record_history: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        if not isinstance(path, PathGeometry):
            raise TypeError(
                "path must be a PathGeometry"
            )

        self.path = path

        self._initialize_common(
            material=material,
            rest_length=rest_length,
            element_id=element_id,
            name=name,
            stiffness=stiffness,
            damping=damping,
            pretension=pretension,
            tension_only=tension_only,
            compression_only=compression_only,
            enabled=enabled,
            record_history=record_history,
            epsilon=epsilon,
        )

        self.validate()

    @property
    def dimension(self) -> int:
        return self.path.dimension

    def connected_nodes(
        self,
    ) -> tuple[Any, ...]:
        return self.path.connected_nodes()

    def current_length(
        self,
    ) -> float:
        return self.path.current_length()

    def length_velocity(
        self,
    ) -> float:
        return self.path.length_velocity()

    def displacement_vector(
        self,
    ) -> np.ndarray:
        raise NotImplementedError(
            "PathElement has no single displacement vector"
        )

    def direction(
        self,
    ) -> np.ndarray:
        raise NotImplementedError(
            "PathElement has no single axial direction"
        )

    def relative_velocity_vector(
        self,
    ) -> np.ndarray:
        raise NotImplementedError(
            "PathElement has no single relative velocity vector"
        )

    def validate(self) -> None:
        if not isinstance(
            self.path,
            PathGeometry,
        ):
            raise TypeError(
                "path must be a PathGeometry"
            )

        if len(self.path.attachments) < 2:
            raise ValueError(
                "PathElement path requires at least "
                "two attachments"
            )

        self.rest_length = self._validate_length(
            self.rest_length,
            name="rest_length",
        )

        if not isinstance(
            self.material,
            Material,
        ):
            raise TypeError(
                "material must inherit from Material"
            )

        material_validate = getattr(
            self.material,
            "validate",
            None,
        )

        if callable(material_validate):
            material_validate()

        self.enabled = bool(
            self.enabled
        )

        self.tension_only = bool(
            self.tension_only
        )

        self.compression_only = bool(
            self.compression_only
        )

        if (
            self.tension_only
            and self.compression_only
        ):
            raise ValueError(
                "element cannot be both tension-only "
                "and compression-only"
            )

        self.last_force.validate()

    def force_vector(
        self,
        axial_force: float | None = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "PathElement has multiple force vectors; "
            "use path.attachment_forces()"
        )

    def apply_forces(
        self,
        *,
        dt: float | None = None,
        update_material: bool = False,
        include_active: bool = True,
    ) -> float:
        axial_force = self.compute_axial_force(
            dt=dt,
            update_material=update_material,
            include_active=include_active,
        )

        self.path.apply_tension(
            axial_force
        )

        return axial_force

    def snapshot(
        self,
    ) -> dict[str, Any]:
        snapshot = {
            "id": self.id,
            "name": self.name,
            "type": type(self).__name__,
            "connected_nodes": [
                getattr(node, "id", None)
                for node in self.connected_nodes()
            ],
            "path": self.path.snapshot(),
            "enabled": bool(self.enabled),
            "dimension": self.dimension,
            "tension_only": bool(
                self.tension_only
            ),
            "compression_only": bool(
                self.compression_only
            ),
            "rest_length": float(
                self.rest_length
            ),
            "material_reference_length": float(
                self.material_reference_length()
            ),
            "current_length": float(
                self.current_length()
            ),
            "extension": float(
                self.extension()
            ),
            "strain": float(
                self.strain()
            ),
            "length_velocity": float(
                self.length_velocity()
            ),
            "force": self.last_force.as_dict(),
            "tension": self.tension(),
            "compression": self.compression(),
            "slack": self.is_slack(),
            "failed": self.is_failed(),
            "elastic_energy": (
                self.elastic_energy()
            ),
            "dissipated_energy": (
                self.dissipated_energy()
            ),
        }

        material_snapshot = getattr(
            self.material,
            "snapshot",
            None,
        )

        if callable(material_snapshot):
            snapshot["material"] = (
                material_snapshot()
            )
        else:
            snapshot["material"] = {
                "type": type(
                    self.material
                ).__name__,
            }

        return snapshot

    def __repr__(
        self,
    ) -> str:
        return (
            f"PathElement("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"attachments="
            f"{len(self.path.attachments)}, "
            f"material="
            f"{type(self.material).__name__}, "
            f"rest_length="
            f"{self.rest_length:.6f}, "
            f"current_length="
            f"{self.current_length():.6f}, "
            f"force="
            f"{self.last_force.total:.6f}, "
            f"enabled={self.enabled})"
        )
