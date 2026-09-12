from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.surface_mesh import SurfaceMesh


@dataclass(frozen=True)
class SurfaceContactCandidate:
    """
    Pure geometric description of the closest relation
    between two surface meshes.

    This object contains no contact force, stiffness,
    friction, penetration law, or joint semantics.
    """

    point_a: np.ndarray
    point_b: np.ndarray
    triangle_a: int
    triangle_b: int
    distance: float
    normal_a: np.ndarray | None
    normal_b: np.ndarray | None
    separation_direction: np.ndarray | None
    contact_normal: np.ndarray | None
    mutually_facing: bool

    @classmethod
    def from_meshes(
        cls,
        mesh_a: SurfaceMesh,
        mesh_b: SurfaceMesh,
    ) -> SurfaceContactCandidate:
        (
            point_a,
            point_b,
            triangle_a,
            triangle_b,
            distance,
        ) = mesh_a.closest_points_to_mesh(mesh_b)

        normal_a = mesh_a.triangle_normal(
            triangle_a
        )
        normal_b = mesh_b.triangle_normal(
            triangle_b
        )

        delta = point_b - point_a
        delta_norm = float(
            np.linalg.norm(delta)
        )

        if delta_norm > 0.0:
            separation_direction = (
                delta / delta_norm
            )
        else:
            separation_direction = None

        if separation_direction is not None:
            contact_normal = separation_direction.copy()
        elif normal_a is not None and normal_b is not None:
            normal_delta = normal_a - normal_b
            normal_delta_norm = float(
                np.linalg.norm(normal_delta)
            )

            if normal_delta_norm > 0.0:
                contact_normal = (
                    normal_delta / normal_delta_norm
                )
            else:
                contact_normal = normal_a.copy()
        elif normal_a is not None:
            contact_normal = normal_a.copy()
        elif normal_b is not None:
            contact_normal = -normal_b.copy()
        else:
            contact_normal = None

        mutually_facing = False

        if (
            contact_normal is not None
            and normal_a is not None
            and normal_b is not None
        ):
            facing_a = float(
                np.dot(
                    normal_a,
                    contact_normal,
                )
            )

            facing_b = float(
                np.dot(
                    normal_b,
                    -contact_normal,
                )
            )

            mutually_facing = (
                facing_a > 0.0
                and facing_b > 0.0
            )

        return cls(
            point_a=point_a.copy(),
            point_b=point_b.copy(),
            triangle_a=int(triangle_a),
            triangle_b=int(triangle_b),
            distance=float(distance),
            normal_a=(
                None
                if normal_a is None
                else normal_a.copy()
            ),
            normal_b=(
                None
                if normal_b is None
                else normal_b.copy()
            ),
            separation_direction=(
                None
                if separation_direction is None
                else separation_direction.copy()
            ),
            contact_normal=(
                None
                if contact_normal is None
                else contact_normal.copy()
            ),
            mutually_facing=mutually_facing,
        )


    @classmethod
    def from_facing_nearby_meshes(
        cls,
        mesh_a: SurfaceMesh,
        mesh_b: SurfaceMesh,
        max_distance: float,
    ) -> list["SurfaceContactCandidate"]:
        """Return nearby candidates whose local surfaces face each other."""
        return [
            candidate
            for candidate in cls.from_nearby_meshes(
                mesh_a,
                mesh_b,
                max_distance,
            )
            if candidate.mutually_facing
        ]
    @classmethod
    def from_nearby_meshes(
        cls,
        mesh_a: SurfaceMesh,
        mesh_b: SurfaceMesh,
        max_distance: float,
    ) -> list[SurfaceContactCandidate]:
        """
        Return geometric contact candidates for all triangle
        pairs within max_distance.

        No force, penetration, stiffness, friction, or joint
        semantics are introduced here.
        """
        pairs = mesh_a.near_triangle_pairs(
            mesh_b,
            max_distance=max_distance,
        )

        candidates: list[
            SurfaceContactCandidate
        ] = []

        for (
            triangle_a,
            triangle_b,
            point_a,
            point_b,
            distance,
        ) in pairs:
            normal_a = mesh_a.triangle_normal(
                triangle_a
            )
            normal_b = mesh_b.triangle_normal(
                triangle_b
            )

            delta = point_b - point_a
            delta_norm = float(
                np.linalg.norm(delta)
            )

            if delta_norm > 0.0:
                separation_direction = (
                    delta / delta_norm
                )
            else:
                separation_direction = None

            if separation_direction is not None:
                contact_normal = (
                    separation_direction.copy()
                )
            elif (
                normal_a is not None
                and normal_b is not None
            ):
                normal_delta = normal_a - normal_b
                normal_delta_norm = float(
                    np.linalg.norm(normal_delta)
                )

                if normal_delta_norm > 0.0:
                    contact_normal = (
                        normal_delta
                        / normal_delta_norm
                    )
                else:
                    contact_normal = normal_a.copy()
            elif normal_a is not None:
                contact_normal = normal_a.copy()
            elif normal_b is not None:
                contact_normal = -normal_b.copy()
            else:
                contact_normal = None

            mutually_facing = False

            if (
                contact_normal is not None
                and normal_a is not None
                and normal_b is not None
            ):
                facing_a = float(
                    np.dot(
                        normal_a,
                        contact_normal,
                    )
                )

                facing_b = float(
                    np.dot(
                        normal_b,
                        -contact_normal,
                    )
                )

                mutually_facing = (
                    facing_a > 0.0
                    and facing_b > 0.0
                )

            candidates.append(
                cls(
                    point_a=point_a.copy(),
                    point_b=point_b.copy(),
                    triangle_a=int(triangle_a),
                    triangle_b=int(triangle_b),
                    distance=float(distance),
                    normal_a=(
                        None
                        if normal_a is None
                        else normal_a.copy()
                    ),
                    normal_b=(
                        None
                        if normal_b is None
                        else normal_b.copy()
                    ),
                    separation_direction=(
                        None
                        if separation_direction is None
                        else separation_direction.copy()
                    ),
                    contact_normal=(
                        None
                        if contact_normal is None
                        else contact_normal.copy()
                    ),
                    mutually_facing=mutually_facing,
                )
            )

        return candidates




def normal_interval_gap(
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Signed gap between triangle projection intervals along a normal.

    Positive: separated projection intervals.
    Zero: touching projection intervals.
    Negative: overlapping projection intervals.

    This is a geometric directional measure only. A negative value
    does not by itself prove physical surface penetration.
    """
    a = np.asarray(triangle_a, dtype=float)
    b = np.asarray(triangle_b, dtype=float)
    n = np.asarray(normal, dtype=float)

    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError(
            "triangles must have shape (3, 3)"
        )

    if n.shape != (3,):
        raise ValueError(
            "normal must have shape (3,)"
        )

    if (
        not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
        or not np.all(np.isfinite(n))
    ):
        raise ValueError(
            "triangle coordinates and normal must be finite"
        )

    norm = float(np.linalg.norm(n))

    if norm <= 1e-12:
        raise ValueError(
            "normal must have non-zero length"
        )

    unit_normal = n / norm

    projection_a = a @ unit_normal
    projection_b = b @ unit_normal

    return float(
        np.min(projection_b)
        - np.max(projection_a)
    )


@dataclass(frozen=True)
class SurfaceContactState:
    gap: float
    relative_normal_velocity: float
    active_closing: bool


def make_surface_contact_state(
    gap: float,
    relative_normal_velocity: float,
) -> SurfaceContactState:
    gap = float(gap)
    relative_normal_velocity = float(
        relative_normal_velocity
    )

    if not np.isfinite(gap):
        raise ValueError("gap must be finite")

    if not np.isfinite(relative_normal_velocity):
        raise ValueError(
            "relative_normal_velocity must be finite"
        )

    return SurfaceContactState(
        gap=gap,
        relative_normal_velocity=relative_normal_velocity,
        active_closing=(
            gap <= 0.0
            and relative_normal_velocity < 0.0
        ),
    )


def unilateral_normal_reaction(
    compression: float,
    relative_normal_velocity: float,
    stiffness: float,
    damping: float,
) -> float:
    compression = float(compression)
    relative_normal_velocity = float(
        relative_normal_velocity
    )
    stiffness = float(stiffness)
    damping = float(damping)

    values = (
        compression,
        relative_normal_velocity,
        stiffness,
        damping,
    )

    if not all(np.isfinite(value) for value in values):
        raise ValueError(
            "contact reaction inputs must be finite"
        )

    if compression < 0.0:
        raise ValueError(
            "compression cannot be negative"
        )

    if stiffness <= 0.0:
        raise ValueError(
            "stiffness must be > 0"
        )

    if damping < 0.0:
        raise ValueError(
            "damping cannot be negative"
        )

    reaction = (
        stiffness * compression
        - damping * relative_normal_velocity
    )

    return max(0.0, float(reaction))


def compliant_layer_compression(
    surface_distance: float,
    reference_thickness: float,
) -> float:
    surface_distance = float(surface_distance)
    reference_thickness = float(reference_thickness)

    if not np.isfinite(surface_distance):
        raise ValueError(
            "surface_distance must be finite"
        )

    if not np.isfinite(reference_thickness):
        raise ValueError(
            "reference_thickness must be finite"
        )

    if surface_distance < 0.0:
        raise ValueError(
            "surface_distance cannot be negative"
        )

    if reference_thickness < 0.0:
        raise ValueError(
            "reference_thickness cannot be negative"
        )

    return max(
        0.0,
        reference_thickness - surface_distance,
    )
