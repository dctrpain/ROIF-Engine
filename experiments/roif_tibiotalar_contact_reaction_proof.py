from __future__ import annotations

from pathlib import Path

import numpy as np

from core.attached_surface_mesh import AttachedSurfaceMesh
from core.node import Node
from core.surface_contact import (
    SurfaceContactCandidate,
    normal_interval_gap,
)
from core.surface_mesh import SurfaceMesh


MESH_DIR = Path(
    r"C:\Users\DELL\OneDrive\Документы\roif-dev\body_parts_3d_api\meshes"
)

TIBIA_PATH = (
    MESH_DIR
    / "FJ3387_BP23960_FMA24477_Right tibia.obj"
)

TALUS_PATH = (
    MESH_DIR
    / "FJ3385_BP23520_FMA24482_Right talus.obj"
)


def make_reference_nodes(
    mesh: SurfaceMesh,
    prefix: str,
) -> list[Node]:
    minimum = np.min(mesh.vertices, axis=0)
    maximum = np.max(mesh.vertices, axis=0)
    extent = maximum - minimum

    positions = (
        minimum,
        minimum + np.array([extent[0], 0.0, 0.0]),
        minimum + np.array([0.0, extent[1], 0.0]),
        minimum + np.array([0.0, 0.0, extent[2]]),
    )

    return [
        Node(
            position=np.asarray(position, dtype=float),
            mass=1.0,
            node_id=f"{prefix}_{index}",
        )
        for index, position in enumerate(positions)
    ]


def main() -> None:
    tibia_mesh = SurfaceMesh.from_obj(TIBIA_PATH)
    talus_mesh = SurfaceMesh.from_obj(TALUS_PATH)

    tibia_nodes = make_reference_nodes(tibia_mesh, "tibia")
    talus_nodes = make_reference_nodes(talus_mesh, "talus")

    tibia_attached = AttachedSurfaceMesh(
        tibia_mesh,
        tibia_nodes,
    )
    talus_attached = AttachedSurfaceMesh(
        talus_mesh,
        talus_nodes,
    )

    for node in talus_nodes:
        node.position = node.position + np.array(
            [0.0, 0.0, 2.0]
        )

    current_tibia = tibia_attached.current_mesh()
    current_talus = talus_attached.current_mesh()

    candidates = SurfaceContactCandidate.from_facing_nearby_meshes(
        current_tibia,
        current_talus,
        5.0,
    )

    negative: list[
        tuple[float, SurfaceContactCandidate]
    ] = []

    for candidate in candidates:
        if candidate.contact_normal is None:
            continue

        triangle_a = np.vstack(
            current_tibia.triangle_vertices(
                candidate.triangle_a
            )
        )
        triangle_b = np.vstack(
            current_talus.triangle_vertices(
                candidate.triangle_b
            )
        )

        gap = normal_interval_gap(
            triangle_a,
            triangle_b,
            candidate.contact_normal,
        )

        if gap < 0.0:
            negative.append((gap, candidate))

    if not negative:
        raise RuntimeError("no negative contact candidate found")

    gap, candidate = min(
        negative,
        key=lambda item: item[0],
    )

    normal = candidate.contact_normal
    assert normal is not None

    reaction_magnitude = 1.0

    force_on_tibia = (
        -reaction_magnitude * normal
    )
    force_on_talus = (
        reaction_magnitude * normal
    )

    tibia_node_forces = (
        tibia_attached.node_forces_for_triangle_point(
            candidate.triangle_a,
            candidate.point_a,
            force_on_tibia,
        )
    )

    talus_node_forces = (
        talus_attached.node_forces_for_triangle_point(
            candidate.triangle_b,
            candidate.point_b,
            force_on_talus,
        )
    )

    total_force = (
        np.sum(tibia_node_forces, axis=0)
        + np.sum(talus_node_forces, axis=0)
    )

    tibia_positions = np.vstack(
        [node.position for node in tibia_nodes]
    )
    talus_positions = np.vstack(
        [node.position for node in talus_nodes]
    )

    distributed_moment = (
        np.sum(
            np.cross(
                tibia_positions,
                tibia_node_forces,
            ),
            axis=0,
        )
        + np.sum(
            np.cross(
                talus_positions,
                talus_node_forces,
            ),
            axis=0,
        )
    )

    expected_moment = (
        np.cross(
            candidate.point_a,
            force_on_tibia,
        )
        + np.cross(
            candidate.point_b,
            force_on_talus,
        )
    )

    force_error = float(
        np.linalg.norm(total_force)
    )

    moment_error = float(
        np.linalg.norm(
            distributed_moment
            - expected_moment
        )
    )

    print("selected_gap_mm:", gap)
    print(
        "selected_triangle_tibia:",
        candidate.triangle_a,
    )
    print(
        "selected_triangle_talus:",
        candidate.triangle_b,
    )
    print(
        "contact_normal:",
        normal.tolist(),
    )
    print(
        "total_force:",
        total_force.tolist(),
    )
    print(
        "force_error:",
        force_error,
    )
    print(
        "distributed_moment:",
        distributed_moment.tolist(),
    )
    print(
        "expected_moment:",
        expected_moment.tolist(),
    )
    print(
        "moment_error:",
        moment_error,
    )

    proof_passed = (
        force_error <= 1e-12
        and moment_error <= 1e-10
    )

    print("proof_passed:", proof_passed)

    if not proof_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
