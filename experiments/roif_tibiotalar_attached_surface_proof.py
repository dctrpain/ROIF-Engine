from __future__ import annotations

from pathlib import Path

import numpy as np

from core.attached_surface_mesh import AttachedSurfaceMesh
from core.node import Node
from core.surface_contact import SurfaceContactCandidate
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

    tibia_nodes = make_reference_nodes(
        tibia_mesh,
        "TIBIA",
    )
    talus_nodes = make_reference_nodes(
        talus_mesh,
        "TALUS",
    )

    tibia_attached = AttachedSurfaceMesh(
        tibia_mesh,
        tibia_nodes,
    )
    talus_attached = AttachedSurfaceMesh(
        talus_mesh,
        talus_nodes,
    )

    initial_candidate = (
        SurfaceContactCandidate.from_meshes(
            tibia_attached.current_mesh(),
            talus_attached.current_mesh(),
        )
    )

    translation = np.array(
        [0.0, 0.0, 2.0],
        dtype=float,
    )

    for node in talus_nodes:
        node.position = (
            node.position
            + translation
        )

    moved_candidate = (
        SurfaceContactCandidate.from_meshes(
            tibia_attached.current_mesh(),
            talus_attached.current_mesh(),
        )
    )

    print(
        f"initial_distance_mm: "
        f"{initial_candidate.distance:.9f}"
    )
    print(
        f"moved_distance_mm: "
        f"{moved_candidate.distance:.9f}"
    )
    print(
        f"distance_change_mm: "
        f"{moved_candidate.distance - initial_candidate.distance:.9f}"
    )
    print(
        "initial_contact_normal:",
        None
        if initial_candidate.contact_normal is None
        else initial_candidate.contact_normal.tolist(),
    )
    print(
        "moved_contact_normal:",
        None
        if moved_candidate.contact_normal is None
        else moved_candidate.contact_normal.tolist(),
    )

    initial_geometry_reproduced = bool(
        np.isclose(
            initial_candidate.distance,
            0.332542573,
            rtol=0.0,
            atol=1e-9,
        )
    )

    contact_geometry_changed = bool(
        not np.isclose(
            moved_candidate.distance,
            initial_candidate.distance,
            rtol=0.0,
            atol=1e-9,
        )
    )

    moved_surface_approached = bool(
        moved_candidate.distance
        < initial_candidate.distance
    )

    proof_passed = bool(
        initial_geometry_reproduced
        and contact_geometry_changed
        and moved_surface_approached
    )

    print("initial_geometry_reproduced:", initial_geometry_reproduced)
    print("contact_geometry_changed:", contact_geometry_changed)
    print("moved_surface_approached:", moved_surface_approached)
    print("proof_passed:", proof_passed)

    if not proof_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
