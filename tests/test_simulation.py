from __future__ import annotations

import numpy as np
import pytest

from core.network import Network
from core.node import Node
from core.simulation import (
    Simulation,
    SimulationConfig,
)


def make_free_fall_network() -> tuple[Network, Node]:
    network = Network(
        gravity=[0.0, -10.0],
        record_history=False,
    )

    node = Node(
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
        mass=2.0,
        node_id="A",
    )

    network.add_node(node)

    return network, node


def test_simulation_step_advances_network_physics():
    network, node = make_free_fall_network()

    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
    )

    stats = simulation.step()

    assert simulation.time == pytest.approx(0.1)
    assert simulation.step_index == 1
    assert node.acceleration == pytest.approx(
        [0.0, -10.0]
    )
    assert node.velocity == pytest.approx(
        [0.0, -1.0]
    )
    assert node.position == pytest.approx(
        [0.0, -0.1]
    )
    assert stats.dt == pytest.approx(0.1)


def test_simulation_resolves_dynamic_external_forces():
    network = Network(record_history=False)
    node = Node(
        position=[0.0, 0.0],
        mass=1.0,
        node_id="A",
    )
    network.add_node(node)

    calls: list[int] = []

    def external_forces(simulation: Simulation):
        calls.append(simulation.step_index)
        return {
            node: np.array([2.0, 0.0]),
        }

    simulation = Simulation(
        network,
        dt=0.5,
        external_forces=external_forces,
        update_materials=False,
    )

    simulation.step()
    simulation.step()

    assert calls == [0, 1]
    assert node.velocity[0] == pytest.approx(2.0)
    assert node.position[0] == pytest.approx(1.5)


def test_simulation_run_steps_records_frames():
    network, _ = make_free_fall_network()

    simulation = Simulation(
        network,
        dt=0.01,
        update_materials=False,
        record_simulation_history=True,
    )

    result = simulation.run_steps(5)

    assert result.steps == 5
    assert simulation.step_index == 5
    assert len(simulation.statistics) == 5
    assert len(simulation.frames) == 5
    assert simulation.latest_frame is simulation.frames[-1]
    assert simulation.frames[-1].time == pytest.approx(
        0.05
    )


def test_simulation_run_uses_exact_final_partial_step():
    network, _ = make_free_fall_network()

    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
    )

    result = simulation.run(0.25)

    assert result.steps == 3
    assert simulation.time == pytest.approx(0.25)
    assert result.statistics[-1].dt == pytest.approx(
        0.05
    )


def test_simulation_reset_restores_initial_state():
    network, node = make_free_fall_network()

    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
    )

    simulation.run_steps(3)
    assert node.position[1] < 0.0

    simulation.reset()

    restored_node = simulation.network.nodes[0]

    assert simulation.time == pytest.approx(0.0)
    assert simulation.step_index == 0
    assert simulation.frame_index == 0
    assert len(simulation.frames) == 0
    assert len(simulation.statistics) == 0
    assert restored_node.position == pytest.approx(
        [0.0, 0.0]
    )
    assert restored_node.velocity == pytest.approx(
        [0.0, 0.0]
    )


def test_simulation_hooks_execute_in_order():
    network, _ = make_free_fall_network()
    events: list[str] = []

    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
    )

    simulation.add_before_step_hook(
        lambda current: events.append(
            f"before:{current.step_index}"
        )
    )
    simulation.add_after_step_hook(
        lambda current: events.append(
            f"after:{current.step_index}"
        )
    )

    simulation.step()

    assert events == [
        "before:0",
        "after:1",
    ]


def test_simulation_config_rejects_invalid_dt():
    with pytest.raises(
        ValueError,
        match="dt must be positive",
    ):
        SimulationConfig(dt=0.0)


def test_simulation_rejects_negative_step_count():
    network, _ = make_free_fall_network()
    simulation = Simulation(network)

    with pytest.raises(
        ValueError,
        match="steps cannot be negative",
    ):
        simulation.run_steps(-1)
