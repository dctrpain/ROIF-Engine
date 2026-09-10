import numpy as np

from core.attached_surface_mesh import AttachedSurfaceMesh
from core.node import Node
from core.surface_mesh import SurfaceMesh


def test_attached_surface_follows_rigid_node_motion() -> None:
    reference_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    nodes = [
        Node(position=position.copy(), mass=1.0)
        for position in reference_nodes
    ]

    vertices = np.array(
        [
            [1.2, -0.3, 0.5],
            [0.2, 0.4, -0.1],
            [-0.5, 0.7, 0.8],
        ]
    )

    mesh = SurfaceMesh(
        vertices,
        np.array([[0, 1, 2]]),
    )

    attached = AttachedSurfaceMesh(
        mesh,
        nodes,
    )

    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    translation = np.array(
        [2.0, 3.0, 4.0]
    )

    for node, reference_position in zip(
        nodes,
        reference_nodes,
        strict=True,
    ):
        node.position = (
            rotation @ reference_position
            + translation
        )

    expected = (
        vertices @ rotation.T
        + translation
    )

    np.testing.assert_allclose(
        attached.current_vertices(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_triangle_point_maps_back_to_roif_nodes() -> None:
    reference_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    nodes = [
        Node(position=position.copy(), mass=1.0)
        for position in reference_nodes
    ]

    vertices = np.array(
        [
            [1.2, -0.3, 0.5],
            [0.2, 0.4, -0.1],
            [-0.5, 0.7, 0.8],
        ]
    )

    mesh = SurfaceMesh(
        vertices,
        np.array([[0, 1, 2]]),
    )

    attached = AttachedSurfaceMesh(
        mesh,
        nodes,
    )

    point = (
        0.2 * vertices[0]
        + 0.3 * vertices[1]
        + 0.5 * vertices[2]
    )

    weights = attached.node_weights_for_triangle_point(
        0,
        point,
    )

    np.testing.assert_allclose(
        np.sum(weights),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    current_node_positions = np.vstack(
        [node.position for node in nodes]
    )

    reconstructed = weights @ current_node_positions

    np.testing.assert_allclose(
        reconstructed,
        point,
        rtol=0.0,
        atol=1e-12,
    )


def test_triangle_point_force_preserves_resultant_and_moment() -> None:
    reference_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    nodes = [
        Node(position=position.copy(), mass=1.0)
        for position in reference_nodes
    ]

    vertices = np.array(
        [
            [1.2, -0.3, 0.5],
            [0.2, 0.4, -0.1],
            [-0.5, 0.7, 0.8],
        ]
    )

    mesh = SurfaceMesh(
        vertices,
        np.array([[0, 1, 2]]),
    )

    attached = AttachedSurfaceMesh(mesh, nodes)

    point = (
        0.2 * vertices[0]
        + 0.3 * vertices[1]
        + 0.5 * vertices[2]
    )

    force = np.array([3.0, -2.0, 5.0])

    node_forces = attached.node_forces_for_triangle_point(
        0,
        point,
        force,
    )

    np.testing.assert_allclose(
        np.sum(node_forces, axis=0),
        force,
        rtol=0.0,
        atol=1e-12,
    )

    node_positions = np.vstack(
        [node.position for node in nodes]
    )

    distributed_moment = np.sum(
        np.cross(node_positions, node_forces),
        axis=0,
    )

    contact_moment = np.cross(point, force)

    np.testing.assert_allclose(
        distributed_moment,
        contact_moment,
        rtol=0.0,
        atol=1e-12,
    )


def test_surface_force_map_enters_network_step() -> None:
    from core.network import Network

    reference_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    nodes = [
        Node(position=position.copy(), mass=1.0)
        for position in reference_nodes
    ]

    mesh = SurfaceMesh(
        np.array(
            [
                [0.2, 0.2, 0.0],
                [0.7, 0.2, 0.0],
                [0.2, 0.7, 0.0],
            ]
        ),
        np.array([[0, 1, 2]]),
    )

    attached = AttachedSurfaceMesh(
        mesh,
        nodes,
    )

    point = np.array(
        [0.3, 0.3, 0.0]
    )
    force = np.array(
        [0.0, 0.0, 4.0]
    )

    external_forces = (
        attached.nodal_force_map_for_triangle_point(
            0,
            point,
            force,
        )
    )

    network = Network(
        record_history=False,
    )

    for node in nodes:
        network.add_node(node)

    network.step(
        dt=0.01,
        external_forces=external_forces,
        update_materials=False,
        include_active=False,
        solve_constraints=False,
        record=False,
    )

    total_nodal_force = np.sum(
        np.vstack([node.force for node in nodes]),
        axis=0,
    )

    total_momentum = np.sum(
        np.vstack(
            [
                node.mass * node.velocity
                for node in nodes
            ]
        ),
        axis=0,
    )

    np.testing.assert_allclose(
        total_nodal_force,
        force,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        total_momentum,
        force * 0.01,
        rtol=0.0,
        atol=1e-12,
    )


def test_triangle_point_velocity_matches_node_motion() -> None:
    reference_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    nodes = [
        Node(position=position.copy(), mass=1.0)
        for position in reference_nodes
    ]

    node_velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )

    for node, velocity in zip(
        nodes,
        node_velocities,
        strict=True,
    ):
        node.velocity = velocity.copy()

    mesh = SurfaceMesh(
        np.array(
            [
                [0.2, 0.2, 0.0],
                [0.7, 0.2, 0.0],
                [0.2, 0.7, 0.0],
            ]
        ),
        np.array([[0, 1, 2]]),
    )

    attached = AttachedSurfaceMesh(
        mesh,
        nodes,
    )

    point = np.array(
        [0.3, 0.3, 0.0]
    )

    weights = attached.node_weights_for_triangle_point(
        0,
        point,
    )

    expected_velocity = (
        weights @ node_velocities
    )

    actual_velocity = (
        attached.velocity_for_triangle_point(
            0,
            point,
        )
    )

    np.testing.assert_allclose(
        actual_velocity,
        expected_velocity,
        rtol=0.0,
        atol=1e-12,
    )
