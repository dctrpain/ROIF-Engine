import numpy as np

from core.attachment import Attachment
from core.node import Node
from core.path_geometry import PathGeometry


def make_node(
    position,
    velocity=None,
) -> Node:
    return Node(
        position=np.asarray(position, dtype=float),
        velocity=(
            np.zeros(3, dtype=float)
            if velocity is None
            else np.asarray(velocity, dtype=float)
        ),
        mass=1.0,
    )


def make_attachment(
    node: Node,
) -> Attachment:
    return Attachment(
        (node,),
        (1.0,),
    )


def test_path_geometry_sums_segment_lengths() -> None:
    node_a = make_node([0.0, 0.0, 0.0])
    node_b = make_node([1.0, 0.0, 0.0])
    node_c = make_node([1.0, 1.0, 0.0])

    path = PathGeometry(
        (
            make_attachment(node_a),
            make_attachment(node_b),
            make_attachment(node_c),
        )
    )

    assert np.allclose(
        path.segment_lengths(),
        (1.0, 1.0),
    )

    assert np.isclose(
        path.current_length(),
        2.0,
    )


def test_path_geometry_sums_segment_length_velocities() -> None:
    node_a = make_node(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    node_b = make_node(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )
    node_c = make_node(
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
    )

    path = PathGeometry(
        (
            make_attachment(node_a),
            make_attachment(node_b),
            make_attachment(node_c),
        )
    )

    assert np.isclose(
        path.length_velocity(),
        3.0,
    )


def test_path_geometry_reports_unique_connected_nodes() -> None:
    node_a = make_node([0.0, 0.0, 0.0])
    node_b = make_node([1.0, 0.0, 0.0])
    node_c = make_node([1.0, 1.0, 0.0])

    shared = Attachment(
        (node_a, node_b),
        (0.5, 0.5),
    )

    path = PathGeometry(
        (
            shared,
            Attachment((node_b,), (1.0,)),
            Attachment((node_c,), (1.0,)),
        )
    )

    assert path.connected_nodes() == (
        node_a,
        node_b,
        node_c,
    )


def test_path_geometry_zero_length_segment_is_finite() -> None:
    node_a = make_node([0.0, 0.0, 0.0])
    node_b = make_node([0.0, 0.0, 0.0])
    node_c = make_node([1.0, 0.0, 0.0])

    path = PathGeometry(
        (
            make_attachment(node_a),
            make_attachment(node_b),
            make_attachment(node_c),
        )
    )

    directions = path.segment_directions()

    assert np.all(
        np.isfinite(
            directions[0]
        )
    )

    assert np.allclose(
        directions[0],
        np.zeros(3),
    )

    assert np.isclose(
        path.current_length(),
        1.0,
    )


def test_path_tension_preserves_resultant_and_moment() -> None:
    node_a = make_node([0.0, 0.0, 0.0])
    node_b = make_node([1.0, 0.0, 0.0])
    node_c = make_node([1.0, 1.0, 0.0])

    path = PathGeometry(
        (
            make_attachment(node_a),
            make_attachment(node_b),
            make_attachment(node_c),
        )
    )

    tension = 7.5
    forces = path.attachment_forces(tension)
    positions = path.positions()

    resultant = sum(
        forces,
        np.zeros(3),
    )

    moment = sum(
        (
            np.cross(position, force)
            for position, force in zip(
                positions,
                forces,
                strict=True,
            )
        ),
        np.zeros(3),
    )

    assert np.linalg.norm(
        forces[1]
    ) > 0.0

    assert np.allclose(
        resultant,
        np.zeros(3),
    )

    assert np.allclose(
        moment,
        np.zeros(3),
    )


def test_path_tension_satisfies_virtual_work_identity() -> None:
    node_a = make_node(
        [0.0, 0.0, 0.0],
        [0.2, -0.1, 0.0],
    )
    node_b = make_node(
        [1.0, 0.0, 0.0],
        [0.4, 0.3, 0.0],
    )
    node_c = make_node(
        [1.0, 1.0, 0.0],
        [-0.2, 0.5, 0.0],
    )

    path = PathGeometry(
        (
            make_attachment(node_a),
            make_attachment(node_b),
            make_attachment(node_c),
        )
    )

    tension = 7.5
    forces = path.attachment_forces(tension)
    velocities = path.velocities()

    power = sum(
        float(
            np.dot(
                force,
                velocity,
            )
        )
        for force, velocity in zip(
            forces,
            velocities,
            strict=True,
        )
    )

    expected_power = (
        -tension
        * path.length_velocity()
    )

    assert np.isclose(
        power,
        expected_power,
    )
