from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from core.network import Network
from core.node import Node
from core.recorder import SimulationRecorder
from core.simulation import Simulation
from visualization import NetworkAnimator


def make_recording():
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

    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
        record_simulation_history=False,
    )

    recording = SimulationRecorder(
        simulation
    ).run_steps(
        3,
        sample_every=1,
    )

    return simulation, recording


def close_animation(animator):
    if animator._animation is not None:
        animator._animation._draw_was_started = True
    if animator._figure is not None:
        plt.close(animator._figure)


def test_animator_from_recording():
    simulation, recording = make_recording()
    animator = NetworkAnimator.from_recording(
        recording
    )

    assert animator.is_recording_playback is True
    assert animator.recording is recording
    assert simulation.step_index == 3


def test_recording_create_uses_full_length_by_default():
    _, recording = make_recording()
    animator = NetworkAnimator.from_recording(
        recording
    )

    animator.create(mode="geometry")

    assert animator._requested_frame_count == len(
        recording
    )

    close_animation(animator)


def test_recording_frame_selection_does_not_run_physics():
    simulation, recording = make_recording()
    animator = NetworkAnimator.from_recording(
        recording
    )

    animator.create(mode="geometry")

    step_before = simulation.step_index
    time_before = simulation.time

    animator._draw_frame(
        2,
        mode="geometry",
        show_reference=True,
        show_labels=True,
        show_colorbar=False,
        title="Test",
    )

    assert simulation.step_index == step_before
    assert simulation.time == time_before
    assert animator.network is recording[2].network

    close_animation(animator)


def test_frames_cannot_exceed_recording():
    _, recording = make_recording()
    animator = NetworkAnimator.from_recording(
        recording
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed recording length",
    ):
        animator.create(
            frames=len(recording) + 1,
            mode="geometry",
        )
