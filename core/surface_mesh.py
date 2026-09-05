from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class SurfaceMesh:
    """
    Triangulated surface geometry.

    This class owns surface geometry only.

    It has no material state, contact force law, mass,
    integration, joint semantics, or anatomy-specific behavior.
    """

    epsilon = 1e-12

    def __init__(
        self,
        vertices: Sequence[Sequence[float]] | np.ndarray,
        triangles: Sequence[Sequence[int]] | np.ndarray,
    ) -> None:
        self.vertices = np.asarray(
            vertices,
            dtype=float,
        ).copy()

        self.triangles = np.asarray(
            triangles,
            dtype=int,
        ).copy()

        self.validate()

    def validate(self) -> None:
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1] != 3
        ):
            raise ValueError(
                "surface vertices must have shape (n, 3)"
            )

        if (
            self.triangles.ndim != 2
            or self.triangles.shape[1] != 3
        ):
            raise ValueError(
                "surface triangles must have shape (m, 3)"
            )

        if len(self.vertices) == 0:
            raise ValueError(
                "surface mesh requires at least one vertex"
            )

        if len(self.triangles) == 0:
            raise ValueError(
                "surface mesh requires at least one triangle"
            )

        if not np.all(
            np.isfinite(self.vertices)
        ):
            raise ValueError(
                "surface vertices must be finite"
            )

        if np.any(self.triangles < 0):
            raise ValueError(
                "triangle indices must be non-negative"
            )

        if np.any(
            self.triangles >= len(self.vertices)
        ):
            raise ValueError(
                "triangle index exceeds vertex count"
            )

    @property
    def vertex_count(self) -> int:
        return int(
            len(self.vertices)
        )

    @property
    def triangle_count(self) -> int:
        return int(
            len(self.triangles)
        )

    def triangle_vertices(
        self,
        triangle_index: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        indices = self.triangles[
            triangle_index
        ]

        return (
            self.vertices[indices[0]].copy(),
            self.vertices[indices[1]].copy(),
            self.vertices[indices[2]].copy(),
        )

    def triangle_centroid(
        self,
        triangle_index: int,
    ) -> np.ndarray:
        a, b, c = self.triangle_vertices(
            triangle_index
        )

        return (
            a + b + c
        ) / 3.0

    def triangle_normal(
        self,
        triangle_index: int,
    ) -> np.ndarray | None:
        a, b, c = self.triangle_vertices(
            triangle_index
        )

        normal = np.cross(
            b - a,
            c - a,
        )

        magnitude = float(
            np.linalg.norm(normal)
        )

        if magnitude <= self.epsilon:
            return None

        return normal / magnitude


    @staticmethod
    def closest_point_on_triangle(
        point: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """
        Return the closest point on triangle ABC to a 3D point.

        Uses barycentric region tests and returns a point on
        a vertex, edge, or the triangle interior.
        """
        point = np.asarray(
            point,
            dtype=float,
        )

        a = np.asarray(
            a,
            dtype=float,
        )

        b = np.asarray(
            b,
            dtype=float,
        )

        c = np.asarray(
            c,
            dtype=float,
        )

        ab = b - a
        ac = c - a
        ap = point - a

        d1 = float(np.dot(ab, ap))
        d2 = float(np.dot(ac, ap))

        if d1 <= 0.0 and d2 <= 0.0:
            return a.copy()

        bp = point - b

        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))

        if d3 >= 0.0 and d4 <= d3:
            return b.copy()

        vc = d1 * d4 - d3 * d2

        if (
            vc <= 0.0
            and d1 >= 0.0
            and d3 <= 0.0
        ):
            denominator = d1 - d3

            if abs(denominator) <= 1e-12:
                return a.copy()

            v = d1 / denominator

            return a + v * ab

        cp = point - c

        d5 = float(np.dot(ab, cp))
        d6 = float(np.dot(ac, cp))

        if d6 >= 0.0 and d5 <= d6:
            return c.copy()

        vb = d5 * d2 - d1 * d6

        if (
            vb <= 0.0
            and d2 >= 0.0
            and d6 <= 0.0
        ):
            denominator = d2 - d6

            if abs(denominator) <= 1e-12:
                return a.copy()

            w = d2 / denominator

            return a + w * ac

        va = d3 * d6 - d5 * d4

        if (
            va <= 0.0
            and (d4 - d3) >= 0.0
            and (d5 - d6) >= 0.0
        ):
            bc = c - b

            denominator = (
                d4 - d3
            ) + (
                d5 - d6
            )

            if abs(denominator) <= 1e-12:
                return b.copy()

            w = (
                d4 - d3
            ) / denominator

            return b + w * bc

        denominator = (
            va + vb + vc
        )

        if abs(denominator) <= 1e-12:
            distances = (
                (
                    float(
                        np.linalg.norm(
                            point - a
                        )
                    ),
                    a,
                ),
                (
                    float(
                        np.linalg.norm(
                            point - b
                        )
                    ),
                    b,
                ),
                (
                    float(
                        np.linalg.norm(
                            point - c
                        )
                    ),
                    c,
                ),
            )

            return min(
                distances,
                key=lambda item: item[0],
            )[1].copy()

        inverse = 1.0 / denominator

        v = vb * inverse
        w = vc * inverse

        return (
            a
            + ab * v
            + ac * w
        )


    @staticmethod
    def closest_points_on_segments(
        p1: np.ndarray,
        q1: np.ndarray,
        p2: np.ndarray,
        q2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the closest points between two 3D line segments.
        """
        p1 = np.asarray(p1, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        q2 = np.asarray(q2, dtype=float)

        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2

        a = float(np.dot(d1, d1))
        e = float(np.dot(d2, d2))
        f = float(np.dot(d2, r))

        epsilon = 1e-12

        if a <= epsilon and e <= epsilon:
            return p1.copy(), p2.copy()

        if a <= epsilon:
            s = 0.0
            t = np.clip(
                f / e,
                0.0,
                1.0,
            )

        else:
            c = float(np.dot(d1, r))

            if e <= epsilon:
                t = 0.0
                s = np.clip(
                    -c / a,
                    0.0,
                    1.0,
                )

            else:
                b = float(np.dot(d1, d2))

                denominator = (
                    a * e
                    - b * b
                )

                if abs(denominator) > epsilon:
                    s = np.clip(
                        (
                            b * f
                            - c * e
                        )
                        / denominator,
                        0.0,
                        1.0,
                    )
                else:
                    s = 0.0

                t = (
                    b * s
                    + f
                ) / e

                if t < 0.0:
                    t = 0.0
                    s = np.clip(
                        -c / a,
                        0.0,
                        1.0,
                    )

                elif t > 1.0:
                    t = 1.0
                    s = np.clip(
                        (
                            b - c
                        )
                        / a,
                        0.0,
                        1.0,
                    )

        closest_1 = (
            p1 + d1 * s
        )

        closest_2 = (
            p2 + d2 * t
        )

        return (
            closest_1,
            closest_2,
        )
    def closest_point(
        self,
        point: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """
        Return the closest point on the surface mesh.

        Returns:
            closest_position,
            triangle_index,
            distance.
        """
        query = np.asarray(
            point,
            dtype=float,
        )

        if query.shape != (3,):
            raise ValueError(
                "query point must have shape (3,)"
            )

        if not np.all(
            np.isfinite(query)
        ):
            raise ValueError(
                "query point must be finite"
            )

        best_position: np.ndarray | None = None
        best_triangle_index = -1
        best_distance_squared = float("inf")

        for triangle_index in range(
            self.triangle_count
        ):
            a, b, c = self.triangle_vertices(
                triangle_index
            )

            candidate = (
                self.closest_point_on_triangle(
                    query,
                    a,
                    b,
                    c,
                )
            )

            delta = query - candidate

            distance_squared = float(
                np.dot(
                    delta,
                    delta,
                )
            )

            if (
                distance_squared
                < best_distance_squared
            ):
                best_position = candidate
                best_triangle_index = (
                    triangle_index
                )
                best_distance_squared = (
                    distance_squared
                )

        if best_position is None:
            raise RuntimeError(
                "surface mesh contains no queryable triangles"
            )

        return (
            best_position.copy(),
            best_triangle_index,
            float(
                np.sqrt(
                    best_distance_squared
                )
            ),
        )
    @classmethod
    def from_obj(
        cls,
        path: str | Path,
    ) -> SurfaceMesh:
        path = Path(path)

        vertices: list[
            tuple[float, float, float]
        ] = []

        triangles: list[
            tuple[int, int, int]
        ] = []

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            for line in file:
                if line.startswith("v "):
                    parts = line.split()

                    vertices.append(
                        (
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
                    )

                elif line.startswith("f "):
                    indices: list[int] = []

                    for token in line.split()[1:]:
                        raw_index = token.split(
                            "/"
                        )[0]

                        index = int(
                            raw_index
                        )

                        if index < 0:
                            index = (
                                len(vertices)
                                + index
                            )
                        else:
                            index -= 1

                        indices.append(
                            index
                        )

                    for offset in range(
                        1,
                        len(indices) - 1,
                    ):
                        triangles.append(
                            (
                                indices[0],
                                indices[offset],
                                indices[offset + 1],
                            )
                        )

        return cls(
            vertices=vertices,
            triangles=triangles,
        )