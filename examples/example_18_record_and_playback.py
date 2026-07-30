r"""
ROIF Engine Example 18

Offline simulation recording and playback.

Run:

    python examples\example_18_record_and_playback.py
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
from core.recorder import SimulationRecorder
from core.simulation import Simulation
from visualization import NetworkAnimator


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
        name="recording_material",
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
    print("ROIF Engine - Example 18")
    print("Simulation Recorder v2.3")
    print("=" * 88)

    network = build_network()

    simulation = Simulation(
        network,
        dt=0.002,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=False,
    )

    recorder = SimulationRecorder(simulation)

    recording = recorder.run_steps(
        1500,
        sample_every=5,
        include_initial=True,
        include_final=True,
        metadata={
            "name": "damped suspension",
        },
    )

    print()
    print("Recording summary")
    print("-" * 88)
    print(f"Physical steps : {simulation.step_index}")
    print(f"Physical time  : {simulation.time:.6f} s")
    print(f"Stored frames  : {len(recording)}")
    print(f"First frame    : t={recording.start_time:.6f} s")
    print(f"Last frame     : t={recording.end_time:.6f} s")
    print(f"Duration       : {recording.duration:.6f} s")

    middle = recording.frame(
        len(recording) // 2
    )

    print()
    print("Random-access inspection")
    print("-" * 88)
    print(f"Frame index    : {middle.index}")
    print(f"Physical step  : {middle.step_index}")
    print(f"Physical time  : {middle.time:.6f} s")

    final_step_before_playback = (
        simulation.step_index
    )
    final_time_before_playback = simulation.time

    animator = NetworkAnimator.from_recording(
        recording
    )

    animator.show(
        interval_ms=30.0,
        mode="force",
        show_reference=True,
        show_labels=True,
        title="ROIF — Recorded Playback",
        repeat=False,
    )

    print()
    print("Playback verification")
    print("-" * 88)
    print(
        "Simulation steps unchanged: "
        f"{simulation.step_index == final_step_before_playback}"
    )
    print(
        "Simulation time unchanged : "
        f"{simulation.time == final_time_before_playback}"
    )


if __name__ == "__main__":
    main()
