import numpy as np
import pytest

from core.surface_contact import SurfaceContactCandidate, make_surface_contact_state
from core.surface_mesh import SurfaceMesh


def test_contact_candidate_preserves_closest_geometry() -> None:
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
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert candidate.triangle_a == 0
    assert candidate.triangle_b == 0
    assert candidate.distance == pytest.approx(2.0)

    assert np.allclose(
        candidate.point_a[:2],
        candidate.point_b[:2],
    )

    assert candidate.point_a[2] == pytest.approx(0.0)
    assert candidate.point_b[2] == pytest.approx(2.0)


def test_contact_candidate_reports_surface_normals() -> None:
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
            (0.0, 1.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.0, 0.0, 2.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert candidate.normal_a is not None
    assert candidate.normal_b is not None

    assert np.allclose(
        candidate.normal_a,
        (0.0, 0.0, 1.0),
    )

    assert np.allclose(
        candidate.normal_b,
        (0.0, 0.0, -1.0),
    )


def test_contact_candidate_reports_separation_direction() -> None:
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

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert candidate.separation_direction is not None

    assert np.allclose(
        candidate.separation_direction,
        (0.0, 0.0, 1.0),
    )


def test_contact_candidate_has_no_separation_direction_at_intersection() -> None:
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

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert candidate.distance == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert np.allclose(
        candidate.point_a,
        candidate.point_b,
        atol=1e-12,
    )

    assert candidate.separation_direction is None


def test_contact_normal_follows_a_to_b_when_separated() -> None:
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
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert candidate.contact_normal is not None

    assert np.allclose(
        candidate.contact_normal,
        (0.0, 0.0, 1.0),
    )

    assert np.allclose(
        candidate.contact_normal,
        candidate.separation_direction,
    )


def test_contact_normal_uses_opposing_surface_normals_at_zero_distance() -> None:
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
            (0.0, 2.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert candidate.distance == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert candidate.separation_direction is None
    assert candidate.contact_normal is not None

    assert np.allclose(
        candidate.normal_a,
        (0.0, 0.0, 1.0),
    )

    assert np.allclose(
        candidate.normal_b,
        (0.0, 0.0, -1.0),
    )

    assert np.allclose(
        candidate.contact_normal,
        (0.0, 0.0, 1.0),
    )


def test_contact_normal_is_unit_length() -> None:
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
            (1.0, 2.0, 3.0),
            (2.0, 2.0, 3.0),
            (1.0, 3.0, 3.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert candidate.contact_normal is not None

    assert np.linalg.norm(
        candidate.contact_normal
    ) == pytest.approx(1.0)


def test_from_nearby_meshes_returns_all_candidates_within_distance() -> None:
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
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (10.0, 1.0, 2.0),
            (11.0, 0.0, 2.0),
            (10.0, 0.0, 2.0),
        ],
        triangles=[
            (0, 1, 2),
            (3, 4, 5),
        ],
    )

    candidates = SurfaceContactCandidate.from_nearby_meshes(
        mesh_a,
        mesh_b,
        max_distance=2.0,
    )

    assert len(candidates) == 2

    pairs = {
        (
            candidate.triangle_a,
            candidate.triangle_b,
        )
        for candidate in candidates
    }

    assert pairs == {
        (0, 0),
        (1, 1),
    }


def test_from_nearby_meshes_preserves_local_distances() -> None:
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
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (10.0, 0.0, 2.0),
            (11.0, 0.0, 2.0),
            (10.0, 1.0, 2.0),
        ],
        triangles=[
            (0, 1, 2),
            (3, 4, 5),
        ],
    )

    candidates = SurfaceContactCandidate.from_nearby_meshes(
        mesh_a,
        mesh_b,
        max_distance=2.0,
    )

    distances = {
        (
            candidate.triangle_a,
            candidate.triangle_b,
        ): candidate.distance
        for candidate in candidates
    }

    assert distances[(0, 0)] == pytest.approx(1.0)
    assert distances[(1, 1)] == pytest.approx(2.0)


def test_from_nearby_meshes_builds_unit_contact_normals() -> None:
    mesh_a = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (5.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
            (5.0, 1.0, 0.0),
        ],
        triangles=[
            (0, 1, 2),
            (3, 4, 5),
        ],
    )

    mesh_b = SurfaceMesh(
        vertices=[
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (5.0, 1.0, 1.5),
            (6.0, 0.0, 1.5),
            (5.0, 0.0, 1.5),
        ],
        triangles=[
            (0, 1, 2),
            (3, 4, 5),
        ],
    )

    candidates = SurfaceContactCandidate.from_nearby_meshes(
        mesh_a,
        mesh_b,
        max_distance=2.0,
    )

    assert len(candidates) == 2

    for candidate in candidates:
        assert candidate.contact_normal is not None

        assert np.linalg.norm(
            candidate.contact_normal
        ) == pytest.approx(1.0)


def test_mutually_facing_true_for_opposing_surfaces() -> None:
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
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert np.allclose(
        candidate.normal_a,
        (0.0, 0.0, 1.0),
    )

    assert np.allclose(
        candidate.normal_b,
        (0.0, 0.0, -1.0),
    )

    assert candidate.mutually_facing is True


def test_mutually_facing_false_for_same_facing_surfaces() -> None:
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
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ],
        triangles=[
            (0, 1, 2),
        ],
    )

    candidate = SurfaceContactCandidate.from_meshes(
        mesh_a,
        mesh_b,
    )

    assert np.allclose(
        candidate.normal_a,
        (0.0, 0.0, 1.0),
    )

    assert np.allclose(
        candidate.normal_b,
        (0.0, 0.0, 1.0),
    )

    assert candidate.mutually_facing is False


def test_from_facing_nearby_meshes_filters_non_facing_pair() -> None:
    mesh_a = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        triangles=[(0, 1, 2)],
    )

    mesh_b = SurfaceMesh(
        vertices=[
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ],
        triangles=[(0, 1, 2)],
    )

    candidates = SurfaceContactCandidate.from_facing_nearby_meshes(
        mesh_a,
        mesh_b,
        max_distance=2.0,
    )

    assert candidates == []


def test_normal_interval_gap_separated() -> None:
    from core.surface_contact import normal_interval_gap

    triangle_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle_b = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )

    assert normal_interval_gap(
        triangle_a,
        triangle_b,
        np.array([0.0, 0.0, 1.0]),
    ) == 1.0


def test_normal_interval_gap_touching() -> None:
    from core.surface_contact import normal_interval_gap

    triangle_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle_b = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert normal_interval_gap(
        triangle_a,
        triangle_b,
        np.array([0.0, 0.0, 1.0]),
    ) == 0.0


def test_normal_interval_gap_overlapping() -> None:
    from core.surface_contact import normal_interval_gap

    triangle_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle_b = np.array(
        [
            [0.0, 0.0, -0.2],
            [0.0, 1.0, 0.2],
            [1.0, 0.0, 0.2],
        ]
    )

    assert np.isclose(
        normal_interval_gap(
            triangle_a,
            triangle_b,
            np.array([0.0, 0.0, 1.0]),
        ),
        -0.2,
    )



def test_surface_contact_state_classifies_normal_motion() -> None:
    separated = make_surface_contact_state(
        gap=1.0,
        relative_normal_velocity=-1.0,
    )
    assert separated.active_closing is False

    touching_static = make_surface_contact_state(
        gap=0.0,
        relative_normal_velocity=0.0,
    )
    assert touching_static.active_closing is False

    overlapping_closing = make_surface_contact_state(
        gap=-1.0,
        relative_normal_velocity=-1.0,
    )
    assert overlapping_closing.active_closing is True

    overlapping_separating = make_surface_contact_state(
        gap=-1.0,
        relative_normal_velocity=1.0,
    )
    assert overlapping_separating.active_closing is False
