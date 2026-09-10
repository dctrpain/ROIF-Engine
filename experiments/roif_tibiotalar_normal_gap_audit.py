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


def gaps_for_state(
    mesh_a: SurfaceMesh,
    mesh_b: SurfaceMesh,
    max_distance: float = 5.0,
) -> tuple[np.ndarray, list[SurfaceContactCandidate]]:
    candidates = SurfaceContactCandidate.from_facing_nearby_meshes(
        mesh_a,
        mesh_b,
        max_distance,
    )

    gaps: list[float] = []
    negative: list[SurfaceContactCandidate] = []

    for candidate in candidates:
        if candidate.contact_normal is None:
            continue

        triangle_a = np.vstack(
            mesh_a.triangle_vertices(candidate.triangle_a)
        )
        triangle_b = np.vstack(
            mesh_b.triangle_vertices(candidate.triangle_b)
        )

        gap = normal_interval_gap(
            triangle_a,
            triangle_b,
            candidate.contact_normal,
        )

        gaps.append(gap)

        if gap < 0.0:
            negative.append(candidate)

    return np.asarray(gaps, dtype=float), negative

def report(
    label: str,
    gaps: np.ndarray,
) -> None:
    print(f"{label}_count: {gaps.size}")

    if gaps.size == 0:
        return

    print(
        f"{label}_min_mm: "
        f"{float(np.min(gaps)):.9f}"
    )
    print(
        f"{label}_median_mm: "
        f"{float(np.median(gaps)):.9f}"
    )
    print(
        f"{label}_max_mm: "
        f"{float(np.max(gaps)):.9f}"
    )
    print(
        f"{label}_negative_count: "
        f"{int(np.sum(gaps < 0.0))}"
    )
    print(
        f"{label}_zero_count: "
        f"{int(np.sum(np.isclose(gaps, 0.0, atol=1e-12)))}"
    )
    print(
        f"{label}_positive_count: "
        f"{int(np.sum(gaps > 0.0))}"
    )



def triangle_component_sizes(
    mesh: SurfaceMesh,
    triangle_indices: set[int],
) -> list[int]:
    remaining = set(triangle_indices)
    components: list[int] = []

    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]

        while frontier:
            current = frontier.pop()
            current_vertices = set(
                int(vertex)
                for vertex in mesh.triangles[current]
            )

            connected = {
                candidate
                for candidate in remaining
                if current_vertices.intersection(
                    int(vertex)
                    for vertex in mesh.triangles[candidate]
                )
            }

            remaining.difference_update(connected)
            component.update(connected)
            frontier.extend(connected)

        components.append(len(component))

    return sorted(components, reverse=True)

def pair_graph_component_sizes(
    candidates: list[SurfaceContactCandidate],
) -> list[int]:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}

    for candidate in candidates:
        node_a = ("tibia", candidate.triangle_a)
        node_b = ("talus", candidate.triangle_b)

        adjacency.setdefault(node_a, set()).add(node_b)
        adjacency.setdefault(node_b, set()).add(node_a)

    remaining = set(adjacency)
    component_edge_counts: list[int] = []

    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]

        while frontier:
            current = frontier.pop()

            for neighbour in adjacency[current]:
                if neighbour not in component:
                    component.add(neighbour)
                    remaining.discard(neighbour)
                    frontier.append(neighbour)

        edge_count = sum(
            len(adjacency[node])
            for node in component
            if node[0] == "tibia"
        )
        component_edge_counts.append(edge_count)

    return sorted(component_edge_counts, reverse=True)

def pair_graph_components(
    candidates: list[SurfaceContactCandidate],
) -> list[list[SurfaceContactCandidate]]:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    edge_candidates: dict[frozenset[tuple[str, int]], list[SurfaceContactCandidate]] = {}

    for candidate in candidates:
        node_a = ("tibia", candidate.triangle_a)
        node_b = ("talus", candidate.triangle_b)

        adjacency.setdefault(node_a, set()).add(node_b)
        adjacency.setdefault(node_b, set()).add(node_a)

        key = frozenset((node_a, node_b))
        edge_candidates.setdefault(key, []).append(candidate)

    remaining = set(adjacency)
    components: list[list[SurfaceContactCandidate]] = []

    while remaining:
        seed = remaining.pop()
        component_nodes = {seed}
        frontier = [seed]

        while frontier:
            current = frontier.pop()

            for neighbour in adjacency[current]:
                if neighbour not in component_nodes:
                    component_nodes.add(neighbour)
                    remaining.discard(neighbour)
                    frontier.append(neighbour)

        component_candidates: list[SurfaceContactCandidate] = []

        for key, candidate_list in edge_candidates.items():
            if key.issubset(component_nodes):
                component_candidates.extend(candidate_list)

        components.append(component_candidates)

    return sorted(components, key=len, reverse=True)


def component_pair_distances(
    components: list[list[SurfaceContactCandidate]],
) -> list[tuple[int, int, float]]:
    points: list[np.ndarray] = []

    for component in components:
        component_points = []

        for candidate in component:
            component_points.append(candidate.point_a)
            component_points.append(candidate.point_b)

        points.append(np.vstack(component_points))

    distances: list[tuple[int, int, float]] = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            delta = points[i][:, None, :] - points[j][None, :, :]
            distance = float(
                np.min(np.linalg.norm(delta, axis=2))
            )
            distances.append((i + 1, j + 1, distance))

    return distances

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

    initial_gaps, _ = gaps_for_state(
        tibia_attached.current_mesh(),
        talus_attached.current_mesh(),
    )

    for node in talus_nodes:
        node.position = (
            node.position
            + np.array([0.0, 0.0, 2.0])
        )

    moved_gaps, moved_negative = gaps_for_state(
        tibia_attached.current_mesh(),
        talus_attached.current_mesh(),
    )

    report("initial", initial_gaps)
    report("moved", moved_gaps)

    print(f"negative_region_count: {len(moved_negative)}")

    if moved_negative:
        points_a = np.vstack(
            [candidate.point_a for candidate in moved_negative]
        )
        points_b = np.vstack(
            [candidate.point_b for candidate in moved_negative]
        )

        print(
            "negative_unique_tibia_triangles:",
            len({c.triangle_a for c in moved_negative}),
        )
        print(
            "negative_unique_talus_triangles:",
            len({c.triangle_b for c in moved_negative}),
        )
        print(
            "negative_tibia_bbox_min:",
            np.min(points_a, axis=0).tolist(),
        )
        print(
            "negative_tibia_bbox_max:",
            np.max(points_a, axis=0).tolist(),
        )
        print(
            "negative_talus_bbox_min:",
            np.min(points_b, axis=0).tolist(),
        )
        print(
            "negative_talus_bbox_max:",
            np.max(points_b, axis=0).tolist(),
        )

        tibia_triangles = {
            c.triangle_a for c in moved_negative
        }
        talus_triangles = {
            c.triangle_b for c in moved_negative
        }

        print(
            "negative_tibia_component_sizes:",
            triangle_component_sizes(
                tibia_attached.current_mesh(),
                tibia_triangles,
            ),
        )
        print(
            "negative_talus_component_sizes:",
            triangle_component_sizes(
                talus_attached.current_mesh(),
                talus_triangles,
            ),
        )
        print(
            "negative_pair_graph_component_sizes:",
            pair_graph_component_sizes(moved_negative),
        )

        components = pair_graph_components(moved_negative)

        print(
            "negative_pair_graph_component_distances_mm:",
            component_pair_distances(components),
        )


if __name__ == "__main__":
    main()




