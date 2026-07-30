from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from core.network import Network
from core.node import Node
from core.simulation import Simulation
from visualization import NetworkAnimator


def make_network() -> tuple[Network, Node]:
    network = Network(
        gravity=[0.0, -10.0],
        record_history=False,
    )

    node = Node(
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
        mass=1.0,
        node_id="A",
    )

    network.add_node(node)

    return network, node


def close_test_animation(
    animator: NetworkAnimator,
) -> None:
    if animator._animation is not None:
        animator._animation._draw_was_started = True

    if animator._figure is not None:
        plt.close(animator._figure)


def draw_one_frame(
    animator: NetworkAnimator,
    frame_index: int = 0,
):
    return animator._draw_frame(
        frame_index,
        mode="geometry",
        show_reference=True,
        show_labels=True,
        show_colorbar=False,
        title="Test",
    )


def test_from_simulation_uses_simulation_network():
    network, _ = make_network()
    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
    )

    animator = NetworkAnimator.from_simulation(
        simulation
    )

    assert animator.network is network
    assert animator.simulation is simulation
    assert animator.is_simulation_driven is True


def test_simulation_driven_frame_advances_physics():
    network, node = make_network()
    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
    )
    animator = NetworkAnimator.from_simulation(
        simulation
    )

    animator.create(
        frames=2,
        interval_ms=20.0,
        mode="geometry",
    )

    artists = draw_one_frame(animator)

    assert simulation.step_index == 1
    assert simulation.time == pytest.approx(0.1)
    assert node.velocity == pytest.approx(
        [0.0, -1.0]
    )
    assert node.position == pytest.approx(
        [0.0, -0.1]
    )
    assert len(artists) > 0

    close_test_animation(animator)


def test_simulation_steps_per_frame_are_applied():
    network, _ = make_network()
    simulation = Simulation(
        network,
        dt=0.01,
        update_materials=False,
    )
    animator = NetworkAnimator.from_simulation(
        simulation,
        simulation_steps_per_frame=4,
    )

    animator.create(
        frames=2,
        mode="geometry",
    )
    draw_one_frame(animator, frame_index=1)

    assert simulation.step_index == 4
    assert simulation.time == pytest.approx(0.04)
    assert len(simulation.frames) == 4

    close_test_animation(animator)


def test_callback_mode_remains_supported():
    network, node = make_network()
    calls: list[int] = []

    def update_frame(frame_index: int) -> None:
        calls.append(frame_index)
        node.set_position(
            [float(frame_index), 0.0]
        )

    animator = NetworkAnimator(
        network,
        update_frame=update_frame,
    )

    animator.create(
        frames=3,
        mode="geometry",
    )
    draw_one_frame(animator, frame_index=2)

    assert calls == [2]
    assert node.position[0] == pytest.approx(2.0)

    close_test_animation(animator)


def test_rejects_simulation_and_callback_together():
    network, _ = make_network()
    simulation = Simulation(network)

    with pytest.raises(
        ValueError,
        match="either simulation or update_frame",
    ):
        NetworkAnimator(
            simulation=simulation,
            update_frame=lambda frame_index: None,
        )


def test_rejects_different_network_from_simulation():
    network, _ = make_network()
    other_network, _ = make_network()
    simulation = Simulation(network)

    with pytest.raises(
        ValueError,
        match="network must be simulation.network",
    ):
        NetworkAnimator(
            other_network,
            simulation=simulation,
        )


def test_rejects_invalid_steps_per_frame():
    network, _ = make_network()
    simulation = Simulation(network)

    with pytest.raises(
        ValueError,
        match=(
            "simulation_steps_per_frame "
            "must be positive"
        ),
    ):
        NetworkAnimator.from_simulation(
            simulation,
            simulation_steps_per_frame=0,
        )


def test_create_rejects_invalid_frames():
    network, _ = make_network()
    animator = NetworkAnimator(network)

    with pytest.raises(
        ValueError,
        match="frames must be positive",
    ):
        animator.create(frames=0)


def test_create_rejects_invalid_interval():
    network, _ = make_network()
    animator = NetworkAnimator(network)

    with pytest.raises(
        ValueError,
        match="interval_ms must be positive",
    ):
        animator.create(
            frames=2,
            interval_ms=0.0,
        )


def test_create_rejects_unknown_mode():
    network, _ = make_network()
    animator = NetworkAnimator(network)

    with pytest.raises(
        ValueError,
        match="unsupported plot mode",
    ):
        animator.create(
            frames=2,
            mode="energy",
        )
