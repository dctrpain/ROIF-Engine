from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node
from visualization import NetworkAnimator


def make_network() -> Network:
    fixed = Node(position=[0.0, 0.0], fixed=True, node_id="A")
    free = Node(position=[1.0, 0.0], node_id="B")
    material = Material(
        name="animation_test_material",
        parameters=MaterialParameters(
            stiffness=10.0,
            damping=0.0,
            failure_threshold=100.0,
            reference_force=1.0,
            reference_strain=1.0,
        ),
        reference_length=1.0,
    )
    element = Element(
        node_a=fixed,
        node_b=free,
        material=material,
        rest_length=1.0,
        element_id="AB",
    )
    network = Network()
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)
    return network


def test_network_animator_calls_update_callback_and_draws_frame():
    network = make_network()
    free = network.nodes[1]
    calls: list[int] = []

    def update_frame(frame_index: int) -> None:
        calls.append(frame_index)
        free.set_position([1.0 + 0.1 * frame_index, 0.0])

    animator = NetworkAnimator(network, update_frame=update_frame)
    animator.create(frames=3, interval_ms=20.0, mode="geometry")
    artists = animator._draw_frame(
        2,
        mode="geometry",
        show_reference=True,
        show_labels=True,
        title="Test",
    )

    assert calls == [2]
    assert free.position[0] == pytest.approx(1.2)
    assert len(artists) > 0


def test_network_animator_rejects_invalid_frames():
    with pytest.raises(ValueError, match="frames must be positive"):
        NetworkAnimator(make_network()).create(frames=0)


def test_network_animator_rejects_invalid_interval():
    with pytest.raises(ValueError, match="interval_ms must be positive"):
        NetworkAnimator(make_network()).create(frames=2, interval_ms=0.0)


def test_network_animator_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported plot mode"):
        NetworkAnimator(make_network()).create(frames=2, mode="energy")
