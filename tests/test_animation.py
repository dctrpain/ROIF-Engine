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


def test_duplicate_visual_frame_does_not_advance_twice():
    network, _ = make_network()
    simulation = Simulation(
        network,
        dt=0.01,
        update_materials=False,
    )
    animator = NetworkAnimator.from_simulation(
        simulation,
        simulation_steps_per_frame=5,
    )

    animator.create(
        frames=3,
        mode="geometry",
    )

    draw_one_frame(animator, 0)
    draw_one_frame(animator, 0)

    assert simulation.step_index == 5
    assert simulation.time == pytest.approx(0.05)

    close_test_animation(animator)


def test_skipped_frames_are_caught_up():
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
        frames=5,
        mode="geometry",
    )

    draw_one_frame(animator, 0)
    draw_one_frame(animator, 3)

    assert simulation.step_index == 16
    assert simulation.time == pytest.approx(0.16)

    close_test_animation(animator)


def test_complete_requested_frames_fills_missing_tail():
    network, _ = make_network()
    simulation = Simulation(
        network,
        dt=0.01,
        update_materials=False,
    )
    animator = NetworkAnimator.from_simulation(
        simulation,
        simulation_steps_per_frame=5,
    )

    animator.create(
        frames=300,
        mode="geometry",
    )

    # Simulate a GUI backend that rendered only frames 0..297.
    draw_one_frame(animator, 297)

    assert simulation.step_index == 1490

    animator.complete_requested_frames()

    assert simulation.step_index == 1500
    assert simulation.time == pytest.approx(15.0)
    assert len(simulation.frames) == 1500
    assert animator.last_advanced_frame_index == 299

    # Completion is idempotent.
    animator.complete_requested_frames()
    assert simulation.step_index == 1500

    close_test_animation(animator)


def test_requested_step_count_property():
    network, _ = make_network()
    simulation = Simulation(network)
    animator = NetworkAnimator.from_simulation(
        simulation,
        simulation_steps_per_frame=5,
    )

    animator.create(
        frames=300,
        mode="geometry",
    )

    assert animator.requested_frame_count == 300
    assert animator.expected_physical_steps == 1500

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
    draw_one_frame(animator, 2)

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


def test_rejects_invalid_steps_per_frame():
    network, _ = make_network()
    simulation = Simulation(network)

    with pytest.raises(
        ValueError,
        match="must be positive",
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


def test_negative_frame_index_is_rejected():
    network, _ = make_network()
    simulation = Simulation(network)
    animator = NetworkAnimator.from_simulation(
        simulation
    )

    animator.create(
        frames=2,
        mode="geometry",
    )

    with pytest.raises(
        ValueError,
        match="frame_index cannot be negative",
    ):
        draw_one_frame(animator, -1)

    close_test_animation(animator)
