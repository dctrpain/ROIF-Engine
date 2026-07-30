from __future__ import annotations

import pytest

from core.network import Network
from core.node import Node
from core.recorder import SimulationRecorder
from core.simulation import Simulation


def make_simulation(
    dt: float = 0.1,
) -> tuple[Simulation, Node]:
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
        dt=dt,
        update_materials=False,
        record_simulation_history=False,
    )

    return simulation, node


def test_run_steps_records_initial_and_final():
    simulation, _ = make_simulation()
    recorder = SimulationRecorder(simulation)

    recording = recorder.run_steps(
        4,
        sample_every=2,
    )

    assert simulation.step_index == 4
    assert len(recording) == 3
    assert recording[0].step_index == 0
    assert recording[1].step_index == 2
    assert recording[2].step_index == 4
    assert recording.end_time == pytest.approx(0.4)


def test_recorded_networks_are_independent():
    simulation, node = make_simulation()
    recorder = SimulationRecorder(simulation)

    recording = recorder.run_steps(
        2,
        sample_every=1,
    )

    first_position = (
        recording.first_frame.network.nodes[0]
        .position.copy()
    )

    node.set_position([100.0, 100.0])

    assert (
        recording.first_frame.network.nodes[0]
        .position
        == pytest.approx(first_position)
    )


def test_playback_data_does_not_advance_simulation():
    simulation, _ = make_simulation()
    recorder = SimulationRecorder(simulation)

    recording = recorder.run_steps(
        3,
        sample_every=1,
    )

    step_before = simulation.step_index
    time_before = simulation.time

    _ = recording.frame(0)
    _ = recording.frame(1)
    _ = recording.last_frame

    assert simulation.step_index == step_before
    assert simulation.time == time_before


def test_nearest_frame():
    simulation, _ = make_simulation()
    recorder = SimulationRecorder(simulation)

    recording = recorder.run_steps(
        4,
        sample_every=1,
    )

    frame = recording.nearest_frame(0.26)

    assert frame.time == pytest.approx(0.3)


def test_run_duration():
    simulation, _ = make_simulation(dt=0.05)
    recorder = SimulationRecorder(simulation)

    recording = recorder.run(
        0.20,
        sample_every=2,
    )

    assert simulation.step_index == 4
    assert recording.last_frame.time == pytest.approx(
        0.20
    )


def test_rejects_non_integral_duration():
    simulation, _ = make_simulation(dt=0.1)
    recorder = SimulationRecorder(simulation)

    with pytest.raises(
        ValueError,
        match="integer multiple",
    ):
        recorder.run(0.25)


def test_forces_final_frame_for_partial_sample_block():
    simulation, _ = make_simulation()
    recorder = SimulationRecorder(simulation)

    recording = recorder.run_steps(
        5,
        sample_every=2,
        include_final=True,
    )

    assert [
        frame.step_index
        for frame in recording
    ] == [0, 2, 4, 5]


def test_invalid_sample_every():
    simulation, _ = make_simulation()
    recorder = SimulationRecorder(simulation)

    with pytest.raises(
        ValueError,
        match="sample_every must be positive",
    ):
        recorder.run_steps(
            10,
            sample_every=0,
        )
