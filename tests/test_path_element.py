import numpy as np

from core.attachment import Attachment
from core.node import Node
from core.path_element import PathElement
from core.path_geometry import PathGeometry


def test_path_element_maps_one_scalar_force_to_multipoint_path() -> None:
    node_a = Node(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        mass=1.0,
    )
    node_b = Node(
        position=np.array([1.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        mass=1.0,
    )
    node_c = Node(
        position=np.array([1.0, 1.0, 0.0]),
        velocity=np.zeros(3),
        mass=1.0,
    )

    path = PathGeometry(
        (
            Attachment((node_a,), (1.0,)),
            Attachment((node_b,), (1.0,)),
            Attachment((node_c,), (1.0,)),
        )
    )

    element = PathElement(
        path,
        stiffness=100.0,
        rest_length=1.5,
        tension_only=True,
    )

    axial_force = element.apply_forces()

    expected_force = 50.0

    assert np.isclose(
        axial_force,
        expected_force,
    )

    assert np.allclose(
        node_a.force,
        np.array([50.0, 0.0, 0.0]),
    )

    assert np.allclose(
        node_b.force,
        np.array([-50.0, 50.0, 0.0]),
    )

    assert np.allclose(
        node_c.force,
        np.array([0.0, -50.0, 0.0]),
    )

    assert np.isclose(
        element.last_force.total,
        expected_force,
    )

    assert len(
        element.connected_nodes()
    ) == 3
