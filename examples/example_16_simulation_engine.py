r"""
ROIF Engine Example 16

Simulation Engine v2.1.

A free node attached to a fixed node by an axial element evolves under
gravity. Node motion is calculated by Network.step(); no position is
prescribed by the example.

Run from the project root:

    python examples\example_16_simulation_engine.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node
from core.simulation import Simulation


def build_network() -> tuple[Network, Node]:
    network = Network(
        gravity=[0.0, -9.81],
        global_damping=1.5,
        record_history=False,
    )

    fixed = Node(
        position=[0.0, 1.0],
        fixed=True,
        mass=1.0,
        node_id="A",
    )
    free = Node(
        position=[0.35, 0.0],
        mass=1.0,
        node_id="B",
    )

    material = Material(
        name="suspension_material",
        parameters=MaterialParameters(
            stiffness=80.0,
            damping=2.0,
            failure_threshold=100.0,
            reference_force=20.0,
            reference_strain=1.0,
        ),
        reference_length=0.8,
    )

    element = Element(
        node_a=fixed,
        node_b=free,
        material=material,
        rest_length=0.8,
        element_id="AB",
        name="Suspension_AB",
    )

    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)

    return network, free


def main() -> None:
    print("=" * 88)
    print("ROIF Engine - Example 16")
    print("Simulation Engine v2.1")
    print("=" * 88)

    network, free = build_network()

    simulation = Simulation(
        network,
        dt=0.002,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=True,
    )

    print()
    print(
        f"{'step':>8}"
        f"{'time [s]':>14}"
        f"{'x [m]':>14}"
        f"{'y [m]':>14}"
        f"{'speed [m/s]':>18}"
        f"{'energy [J]':>16}"
    )
    print("-" * 84)

    for _ in range(1000):
        stats = simulation.step()

        if simulation.step_index % 100 == 0:
            speed = (
                free.velocity[0] ** 2
                + free.velocity[1] ** 2
            ) ** 0.5

            print(
                f"{simulation.step_index:>8d}"
                f"{simulation.time:>14.6f}"
                f"{free.position[0]:>14.6f}"
                f"{free.position[1]:>14.6f}"
                f"{speed:>18.6f}"
                f"{stats.total_energy:>16.6f}"
            )

    print()
    print("Simulation summary")
    print("-" * 84)
    print(f"Physical steps : {simulation.step_index}")
    print(f"Physical time  : {simulation.time:.6f} s")
    print(f"Stored frames  : {len(simulation.frames)}")
    print(
        "Final position: "
        f"[{free.position[0]:.6f}, "
        f"{free.position[1]:.6f}]"
    )


if __name__ == "__main__":
    main()
