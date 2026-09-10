from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .attachment import affine_weights_for_point
from .node import Node
from .surface_mesh import SurfaceMesh


class AttachedSurfaceMesh:
    """
    Surface geometry affinely attached to four non-coplanar ROIF nodes.

    This class adds no mass, integration, constraints, contact law,
    or rigid-body solver. It only maps a reference SurfaceMesh into
    the current configuration of an existing ROIF bone-node cluster.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        nodes: Sequence[Node],
    ) -> None:
        self.mesh = mesh
        self.nodes = tuple(nodes)

        if len(self.nodes) != 4:
            raise ValueError(
                "attached surface requires exactly four reference nodes"
            )

        self.weights = np.vstack(
            [
                affine_weights_for_point(
                    self.nodes,
                    vertex,
                )
                for vertex in self.mesh.vertices
            ]
        )

    def current_vertices(self) -> np.ndarray:
        positions = np.vstack(
            [
                node.position
                for node in self.nodes
            ]
        )

        return self.weights @ positions

    def current_mesh(self) -> SurfaceMesh:
        return SurfaceMesh(
            self.current_vertices(),
            self.mesh.triangles,
        )


    def node_weights_for_triangle_point(
        self,
        triangle_index: int,
        point: np.ndarray,
    ) -> np.ndarray:
        if triangle_index < 0 or triangle_index >= self.mesh.triangle_count:
            raise IndexError("triangle_index out of range")

        point_vector = np.asarray(point, dtype=float)

        if point_vector.shape != (3,):
            raise ValueError("point must have shape (3,)")
        if not np.all(np.isfinite(point_vector)):
            raise ValueError("point must be finite")

        vertex_indices = self.mesh.triangles[triangle_index]
        current_vertices = self.current_vertices()

        a, b, c = current_vertices[vertex_indices]

        v0 = b - a
        v1 = c - a
        v2 = point_vector - a

        d00 = float(np.dot(v0, v0))
        d01 = float(np.dot(v0, v1))
        d11 = float(np.dot(v1, v1))
        d20 = float(np.dot(v2, v0))
        d21 = float(np.dot(v2, v1))

        denominator = d00 * d11 - d01 * d01

        if abs(denominator) <= 1e-12:
            raise ValueError("triangle is degenerate")

        bary_b = (d11 * d20 - d01 * d21) / denominator
        bary_c = (d00 * d21 - d01 * d20) / denominator
        bary_a = 1.0 - bary_b - bary_c

        barycentric = np.array(
            [bary_a, bary_b, bary_c],
            dtype=float,
        )

        vertex_node_weights = self.weights[vertex_indices]

        return barycentric @ vertex_node_weights

    def node_forces_for_triangle_point(
        self,
        triangle_index: int,
        point: np.ndarray,
        force: np.ndarray,
    ) -> np.ndarray:
        force_vector = np.asarray(force, dtype=float)

        if force_vector.shape != (3,):
            raise ValueError("force must have shape (3,)")
        if not np.all(np.isfinite(force_vector)):
            raise ValueError("force must be finite")

        weights = self.node_weights_for_triangle_point(
            triangle_index,
            point,
        )

        return weights[:, None] * force_vector[None, :]
