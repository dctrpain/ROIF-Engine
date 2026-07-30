r"""
ROIF Engine Example 17

Solver-driven animation with Simulation Engine v2.1 and
NetworkAnimator v2.2.

No node position is prescribed. Every visual frame advances the
physical simulation and displays the resulting network state.

Run from the project root:

    python examples\example_17_solver_driven_animation.py
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
from visualization import NetworkAnimator


FRAME_COUNT = 300
INTERVAL_MS = 30.0
SIMULATION_STEPS_PER_FRAME = 5


def build_network() -> Network:
    network = Network(
        gravity=[0.0, -9.81],
        global_damping=1.2,
        record_history=False,
    )

    fixed = Node(
        position=[0.0, 1.0],
        fixed=True,
        mass=1.0,
        node_id="A",
    )
    free = Node(
        position=[0.55, 0.05],
        velocity=[0.0, 0.0],
        mass=1.0,
        node_id="B",
    )

    material = Material(
        name="solver_driven_material",
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

    return network


def main() -> None:
    print("=" * 88)
    print("ROIF Engine - Example 17")
    print("Solver-Driven Animation")
    print("=" * 88)

    network = build_network()

    simulation = Simulation(
        network,
        dt=0.002,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=True,
    )

    animator = NetworkAnimator.from_simulation(
        simulation,
        simulation_steps_per_frame=(
            SIMULATION_STEPS_PER_FRAME
        ),
    )

    print()
    print(
        "Each visual frame advances "
        f"{SIMULATION_STEPS_PER_FRAME} physical steps."
    )
    print(
        "Physical time per visual frame: "
        f"{simulation.dt * SIMULATION_STEPS_PER_FRAME:.4f} s"
    )
    print("Close the Matplotlib window to finish.")

    animator.show(
        frames=FRAME_COUNT,
        interval_ms=INTERVAL_MS,
        mode="force",
        show_reference=True,
        show_labels=True,
        title="ROIF — Solver-Driven Motion",
        repeat=False,
    )

    print()
    print("Final simulation state")
    print("-" * 88)
    print(f"Physical steps : {simulation.step_index}")
    print(f"Physical time  : {simulation.time:.6f} s")
    print(f"Stored frames  : {len(simulation.frames)}")


if __name__ == "__main__":
    main()
