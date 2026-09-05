import numpy as np
import pytest

from core.surface_mesh import SurfaceMesh


def make_triangle() -> SurfaceMesh:
    return SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )


def test_surface_mesh_counts() -> None:
    mesh = make_triangle()

    assert mesh.vertex_count == 3
    assert mesh.triangle_count == 1


def test_triangle_vertices() -> None:
    mesh = make_triangle()

    a, b, c = mesh.triangle_vertices(0)

    assert np.allclose(a, (0.0, 0.0, 0.0))
    assert np.allclose(b, (1.0, 0.0, 0.0))
    assert np.allclose(c, (0.0, 1.0, 0.0))


def test_triangle_centroid() -> None:
    mesh = make_triangle()

    centroid = mesh.triangle_centroid(0)

    assert np.allclose(
        centroid,
        (1.0 / 3.0, 1.0 / 3.0, 0.0),
    )


def test_triangle_normal() -> None:
    mesh = make_triangle()

    normal = mesh.triangle_normal(0)

    assert normal is not None
    assert np.allclose(
        normal,
        (0.0, 0.0, 1.0),
    )


def test_degenerate_triangle_has_no_normal() -> None:
    mesh = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    assert mesh.triangle_normal(0) is None


def test_rejects_out_of_range_triangle_index() -> None:
    with pytest.raises(
        ValueError,
        match="triangle index exceeds vertex count",
    ):
        SurfaceMesh(
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            triangles=[
                (0, 1, 3),
            ],
        )

def test_closest_point_projects_to_triangle_interior() -> None:
    mesh = make_triangle()

    point = np.array(
        [0.25, 0.25, 2.0],
        dtype=float,
    )

    a, b, c = mesh.triangle_vertices(0)

    closest = mesh.closest_point_on_triangle(
        point,
        a,
        b,
        c,
    )

    assert np.allclose(
        closest,
        (0.25, 0.25, 0.0),
    )


def test_closest_point_projects_to_edge() -> None:
    mesh = make_triangle()

    point = np.array(
        [0.75, -1.0, 0.0],
        dtype=float,
    )

    a, b, c = mesh.triangle_vertices(0)

    closest = mesh.closest_point_on_triangle(
        point,
        a,
        b,
        c,
    )

    assert np.allclose(
        closest,
        (0.75, 0.0, 0.0),
    )


def test_closest_point_projects_to_vertex() -> None:
    mesh = make_triangle()

    point = np.array(
        [-1.0, -1.0, 0.0],
        dtype=float,
    )

    a, b, c = mesh.triangle_vertices(0)

    closest = mesh.closest_point_on_triangle(
        point,
        a,
        b,
        c,
    )

    assert np.allclose(
        closest,
        (0.0, 0.0, 0.0),
    )


def test_closest_point_handles_point_above_edge() -> None:
    mesh = make_triangle()

    point = np.array(
        [0.5, -0.2, 1.0],
        dtype=float,
    )

    a, b, c = mesh.triangle_vertices(0)

    closest = mesh.closest_point_on_triangle(
        point,
        a,
        b,
        c,
    )

    assert np.allclose(
        closest,
        (0.5, 0.0, 0.0),
    )


def make_two_triangle_surface() -> SurfaceMesh:
    return SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (10.0, 0.0, 0.0),
            (11.0, 0.0, 0.0),
            (10.0, 1.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
            (3, 4, 5),
        ],
    )


def test_surface_closest_point_selects_nearest_triangle() -> None:
    mesh = make_two_triangle_surface()

    closest, triangle_index, distance = mesh.closest_point(
        (10.25, 0.25, 2.0)
    )

    assert triangle_index == 1

    assert np.allclose(
        closest,
        (10.25, 0.25, 0.0),
    )

    assert distance == pytest.approx(
        2.0
    )


def test_surface_closest_point_selects_first_triangle() -> None:
    mesh = make_two_triangle_surface()

    closest, triangle_index, distance = mesh.closest_point(
        (0.25, 0.25, 3.0)
    )

    assert triangle_index == 0

    assert np.allclose(
        closest,
        (0.25, 0.25, 0.0),
    )

    assert distance == pytest.approx(
        3.0
    )


def test_surface_closest_point_rejects_wrong_shape() -> None:
    mesh = make_triangle()

    with pytest.raises(
        ValueError,
        match="query point must have shape",
    ):
        mesh.closest_point(
            (1.0, 2.0)
        )


def test_surface_closest_point_rejects_non_finite_point() -> None:
    mesh = make_triangle()

    with pytest.raises(
        ValueError,
        match="query point must be finite",
    ):
        mesh.closest_point(
            (0.0, np.nan, 0.0)
        )


def test_closest_points_on_segments_intersecting() -> None:
    closest_1, closest_2 = SurfaceMesh.closest_points_on_segments(
        np.array((0.0, 0.0, 0.0)),
        np.array((1.0, 1.0, 0.0)),
        np.array((0.0, 1.0, 0.0)),
        np.array((1.0, 0.0, 0.0)),
    )

    expected = np.array((0.5, 0.5, 0.0))

    assert np.allclose(
        closest_1,
        expected,
    )

    assert np.allclose(
        closest_2,
        expected,
    )


def test_closest_points_on_segments_parallel() -> None:
    closest_1, closest_2 = SurfaceMesh.closest_points_on_segments(
        np.array((0.0, 0.0, 0.0)),
        np.array((1.0, 0.0, 0.0)),
        np.array((0.0, 2.0, 0.0)),
        np.array((1.0, 2.0, 0.0)),
    )

    assert np.allclose(
        closest_1,
        (0.0, 0.0, 0.0),
    )

    assert np.allclose(
        closest_2,
        (0.0, 2.0, 0.0),
    )


def test_closest_points_on_segments_handles_degenerate_segment() -> None:
    closest_1, closest_2 = SurfaceMesh.closest_points_on_segments(
        np.array((0.5, 1.0, 0.0)),
        np.array((0.5, 1.0, 0.0)),
        np.array((0.0, 0.0, 0.0)),
        np.array((1.0, 0.0, 0.0)),
    )

    assert np.allclose(
        closest_1,
        (0.5, 1.0, 0.0),
    )

    assert np.allclose(
        closest_2,
        (0.5, 0.0, 0.0),
    )


def test_closest_points_on_triangles_parallel_separated() -> None:
    point_a, point_b, distance = SurfaceMesh.closest_points_on_triangles(
        np.array((0.0, 0.0, 0.0)),
        np.array((1.0, 0.0, 0.0)),
        np.array((0.0, 1.0, 0.0)),
        np.array((0.0, 0.0, 2.0)),
        np.array((1.0, 0.0, 2.0)),
        np.array((0.0, 1.0, 2.0)),
    )

    assert distance == pytest.approx(2.0)

    assert point_a[2] == pytest.approx(0.0)
    assert point_b[2] == pytest.approx(2.0)

    assert np.allclose(
        point_a[:2],
        point_b[:2],
    )


def test_closest_points_on_triangles_edge_edge_minimum() -> None:
    point_a, point_b, distance = SurfaceMesh.closest_points_on_triangles(
        np.array((-1.0, 0.0, 0.0)),
        np.array((1.0, 0.0, 0.0)),
        np.array((0.72940745, -2.86544745, -0.66946019)),
        np.array((0.0, -1.0, 1.0)),
        np.array((0.0, 1.0, 1.0)),
        np.array((0.96092953, -1.29637640, 3.53023627)),
    )

    assert distance == pytest.approx(1.0)

    assert np.allclose(
        point_a,
        (0.0, 0.0, 0.0),
        atol=1e-12,
    )

    assert np.allclose(
        point_b,
        (0.0, 0.0, 1.0),
        atol=1e-12,
    )


def test_closest_points_on_triangles_intersecting() -> None:
    point_a, point_b, distance = SurfaceMesh.closest_points_on_triangles(
        np.array((0.0, 0.0, 0.0)),
        np.array((2.0, 0.0, 0.0)),
        np.array((0.0, 2.0, 0.0)),
        np.array((0.5, 0.5, -1.0)),
        np.array((0.5, 0.5, 1.0)),
        np.array((1.5, 0.5, 0.0)),
    )

    assert distance == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert np.allclose(
        point_a,
        point_b,
        atol=1e-12,
    )


def test_closest_points_to_mesh_finds_global_minimum() -> None:
    mesh_a = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    mesh_b = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
            (0.0, 1.0, 3.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    (
        point_a,
        point_b,
        triangle_a,
        triangle_b,
        distance,
    ) = mesh_a.closest_points_to_mesh(mesh_b)

    assert triangle_a == 0
    assert triangle_b == 0
    assert distance == pytest.approx(3.0)

    assert np.allclose(
        point_a[:2],
        point_b[:2],
    )


def test_closest_points_to_mesh_selects_correct_triangle_pair() -> None:
    mesh_a = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (10.0, 0.0, 0.0),
            (11.0, 0.0, 0.0),
            (10.0, 1.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
            (3, 4, 5),
        ],
    )

    mesh_b = SurfaceMesh(
        vertices=[
            (10.0, 0.0, 2.0),
            (11.0, 0.0, 2.0),
            (10.0, 1.0, 2.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    (
        _,
        _,
        triangle_a,
        triangle_b,
        distance,
    ) = mesh_a.closest_points_to_mesh(mesh_b)

    assert triangle_a == 1
    assert triangle_b == 0
    assert distance == pytest.approx(2.0)


def test_closest_points_to_mesh_intersection_has_zero_distance() -> None:
    mesh_a = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    mesh_b = SurfaceMesh(
        vertices=[
            (0.5, 0.5, -1.0),
            (0.5, 0.5, 1.0),
            (1.5, 0.5, 0.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    (
        point_a,
        point_b,
        triangle_a,
        triangle_b,
        distance,
    ) = mesh_a.closest_points_to_mesh(mesh_b)

    assert triangle_a == 0
    assert triangle_b == 0

    assert distance == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert np.allclose(
        point_a,
        point_b,
        atol=1e-12,
    )
