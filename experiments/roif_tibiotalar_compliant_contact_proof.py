from __future__ import annotations

from pathlib import Path

import numpy as np

from core.attached_surface_mesh import AttachedSurfaceMesh
from core.node import Node
from core.network import Network
from core.surface_contact import (
    SurfaceContactCandidate,
    compliant_layer_compression,
    normal_interval_gap,
    unilateral_normal_reaction,
)
from core.surface_mesh import SurfaceMesh

MM_TO_M = 1e-3


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
    tibia_raw = SurfaceMesh.from_obj(TIBIA_PATH)
    talus_raw = SurfaceMesh.from_obj(TALUS_PATH)

    tibia_mesh = SurfaceMesh(
        tibia_raw.vertices * MM_TO_M,
        tibia_raw.triangles,
    )
    talus_mesh = SurfaceMesh(
        talus_raw.vertices * MM_TO_M,
        talus_raw.triangles,
    )

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
            [0.0, 0.0, 0.002]
        )

    current_tibia = tibia_attached.current_mesh()
    current_talus = talus_attached.current_mesh()

    candidates = SurfaceContactCandidate.from_facing_nearby_meshes(
        current_tibia,
        current_talus,
        0.005,
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

    velocity_tibia = (
        tibia_attached.velocity_for_triangle_point(
            candidate.triangle_a,
            candidate.point_a,
        )
    )

    velocity_talus = (
        talus_attached.velocity_for_triangle_point(
            candidate.triangle_b,
            candidate.point_b,
        )
    )

    relative_velocity = (
        velocity_talus - velocity_tibia
    )

    relative_normal_velocity = float(
        np.dot(
            relative_velocity,
            normal,
        )
    )

    print(
        "relative_normal_velocity:",
        relative_normal_velocity,
    )
    closing_speed = 1.0

    for node in talus_nodes:
        node.velocity = (
            -closing_speed * normal
        )

    dynamic_velocity_tibia = (
        tibia_attached.velocity_for_triangle_point(
            candidate.triangle_a,
            candidate.point_a,
        )
    )

    dynamic_velocity_talus = (
        talus_attached.velocity_for_triangle_point(
            candidate.triangle_b,
            candidate.point_b,
        )
    )

    dynamic_relative_normal_velocity = float(
        np.dot(
            dynamic_velocity_talus
            - dynamic_velocity_tibia,
            normal,
        )
    )

    print(
        "dynamic_relative_normal_velocity:",
        dynamic_relative_normal_velocity,
    )

    if dynamic_relative_normal_velocity >= 0.0:
        raise RuntimeError(
            "closing contact must have negative "
            "relative normal velocity"
        )

    # Synthetic SI parameters for mechanics proof only.
    # These are not physiological tibiotalar cartilage values.
    synthetic_reference_thickness = 0.002
    synthetic_stiffness = 5000.0
    synthetic_damping = 50.0

    surface_distance = float(candidate.distance)

    compression = compliant_layer_compression(
        surface_distance=surface_distance,
        reference_thickness=synthetic_reference_thickness,
    )

    reaction_magnitude = unilateral_normal_reaction(
        compression=compression,
        relative_normal_velocity=dynamic_relative_normal_velocity,
        stiffness=synthetic_stiffness,
        damping=synthetic_damping,
    )

    print("surface_distance_m:", surface_distance)
    print("synthetic_reference_thickness_m:", synthetic_reference_thickness)
    print("compression_m:", compression)
    print("reaction_magnitude_N:", reaction_magnitude)

    for node in talus_nodes:
        node.velocity = np.zeros(3, dtype=float)

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

    tibia_force_map = (
        tibia_attached.nodal_force_map_for_triangle_point(
            candidate.triangle_a,
            candidate.point_a,
            force_on_tibia,
        )
    )

    talus_force_map = (
        talus_attached.nodal_force_map_for_triangle_point(
            candidate.triangle_b,
            candidate.point_b,
            force_on_talus,
        )
    )

    external_forces: dict[Node, np.ndarray] = {}

    for force_map in (
        tibia_force_map,
        talus_force_map,
    ):
        for node, node_force in force_map.items():
            if node in external_forces:
                external_forces[node] = (
                    external_forces[node] + node_force
                )
            else:
                external_forces[node] = node_force.copy()

    network = Network(record_history=False)

    for node in tibia_nodes + talus_nodes:
        network.add_node(node)

    dt = 0.01

    network.step(
        dt=dt,
        external_forces=external_forces,
        update_materials=False,
        include_active=False,
        solve_constraints=False,
        record=False,
    )

    tibia_momentum = np.sum(
        np.vstack(
            [
                node.mass * node.velocity
                for node in tibia_nodes
            ]
        ),
        axis=0,
    )

    talus_momentum = np.sum(
        np.vstack(
            [
                node.mass * node.velocity
                for node in talus_nodes
            ]
        ),
        axis=0,
    )

    tibia_impulse_error = float(
        np.linalg.norm(
            tibia_momentum - force_on_tibia * dt
        )
    )

    talus_impulse_error = float(
        np.linalg.norm(
            talus_momentum - force_on_talus * dt
        )
    )

    print(
        "tibia_impulse_error:",
        tibia_impulse_error,
    )
    print(
        "talus_impulse_error:",
        talus_impulse_error,
    )
    print("selected_gap_m:", gap)
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






