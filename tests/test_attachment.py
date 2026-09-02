import numpy as np
import pytest

from core.attachment import Attachment
from core.node import Node


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


def test_attachment_position_and_velocity_are_affine() -> None:
    node_a = make_node(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )
    node_b = make_node(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )
    node_c = make_node(
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )

    attachment = Attachment(
        (node_a, node_b, node_c),
        (0.2, 0.3, 0.5),
    )

    assert np.allclose(
        attachment.position(),
        np.array([0.3, 0.5, 0.0]),
    )

    assert np.allclose(
        attachment.velocity(),
        np.array([0.2, 0.3, 0.5]),
    )


def test_attachment_force_preserves_resultant_and_moment() -> None:
    node_a = make_node([0.0, 0.0, 0.0])
    node_b = make_node([1.0, 0.0, 0.0])
    node_c = make_node([0.0, 1.0, 0.0])

    attachment = Attachment(
        (node_a, node_b, node_c),
        (0.2, 0.3, 0.5),
    )

    force = np.array(
        [2.0, -1.0, 3.0],
        dtype=float,
    )

    position = attachment.position()

    attachment.apply_force(force)

    resultant = (
        node_a.force
        + node_b.force
        + node_c.force
    )

    moment = (
        np.cross(node_a.position, node_a.force)
        + np.cross(node_b.position, node_b.force)
        + np.cross(node_c.position, node_c.force)
    )

    expected_moment = np.cross(
        position,
        force,
    )

    assert np.allclose(
        resultant,
        force,
    )

    assert np.allclose(
        moment,
        expected_moment,
    )


def test_attachment_rejects_weights_that_do_not_sum_to_one() -> None:
    node_a = make_node([0.0, 0.0, 0.0])
    node_b = make_node([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        Attachment(
            (node_a, node_b),
            (0.4, 0.4),
        )


def test_attachment_reports_connected_nodes() -> None:
    node_a = make_node([0.0, 0.0, 0.0])
    node_b = make_node([1.0, 0.0, 0.0])

    attachment = Attachment(
        (node_a, node_b),
        (0.25, 0.75),
    )

    assert attachment.connected_nodes() == (
        node_a,
        node_b,
    )
