from __future__ import annotations

from core.network import Network
from core.node import Node
from core.recorder import SimulationRecorder
from core.simulation import Simulation
from visualization import NetworkAnimator


def make_simulation() -> Simulation:
    network = Network(
        gravity=[0.0, -10.0],
        record_history=False,
    )
    network.add_node(
        Node(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            mass=1.0,
            node_id="A",
        )
    )
    return Simulation(
        network,
        dt=0.01,
        update_materials=False,
    )


def test_v231_preserves_v222_public_properties():
    simulation = make_simulation()
    animator = NetworkAnimator.from_simulation(
        simulation,
        simulation_steps_per_frame=5,
    )

    assert animator.last_advanced_frame_index == -1
    assert animator.requested_frame_count == 0
    assert animator.expected_physical_steps == 0


def test_recording_mode_does_not_report_simulation_driven():
    simulation = make_simulation()
    recording = SimulationRecorder(
        simulation
    ).run_steps(2)

    animator = NetworkAnimator.from_recording(
        recording
    )

    assert animator.is_recording_playback is True
    assert animator.is_simulation_driven is False
