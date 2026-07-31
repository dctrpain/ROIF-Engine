r"""
ROIF Engine Example 19

Save a SimulationRecording to disk, inspect it, load it back, and play
the loaded recording without rerunning physics.

Run:

    python examples\example_19_recording_storage.py
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
from core.storage import RecordingStorage
from visualization import NetworkAnimator


OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "example_19_recording.roifrec"
)


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
        name="storage_material",
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
    print("ROIF Engine - Example 19")
    print("Recording Storage v2.4")
    print("=" * 88)

    simulation = Simulation(
        build_network(),
        dt=0.002,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=False,
    )

    recording = SimulationRecorder(
        simulation
    ).run_steps(
        1500,
        sample_every=5,
        metadata={
            "name": "stored damped suspension",
            "purpose": "offline playback",
        },
    )

    saved_path = RecordingStorage.save(
        recording,
        OUTPUT_PATH,
        compress=True,
        overwrite=True,
    )

    print()
    print("Saved recording")
    print("-" * 88)
    print(f"Path           : {saved_path}")
    print(
        f"File size      : "
        f"{saved_path.stat().st_size:,} bytes"
    )

    info = RecordingStorage.inspect(
        saved_path,
        verify_checksum=True,
    )

    print()
    print("Manifest inspection")
    print("-" * 88)
    print(f"Format         : {info.format_name}")
    print(f"Format version : {info.format_version}")
    print(f"Storage version: {info.storage_version}")
    print(f"Compressed     : {info.compressed}")
    print(f"Frames         : {info.frame_count}")
    print(f"Physical steps : {info.physical_steps}")
    print(f"Duration       : {info.duration:.6f} s")
    print(f"SHA-256        : {info.payload_sha256}")

    loaded = RecordingStorage.load(
        saved_path,
        verify_checksum=True,
    )

    print()
    print("Loaded recording")
    print("-" * 88)
    print(f"Frames         : {len(loaded)}")
    print(f"Start time     : {loaded.start_time:.6f} s")
    print(f"End time       : {loaded.end_time:.6f} s")
    print(
        "Last frame step: "
        f"{loaded.last_frame.step_index}"
    )

    step_before_playback = simulation.step_index
    time_before_playback = simulation.time

    NetworkAnimator.from_recording(
        loaded
    ).show(
        interval_ms=30.0,
        mode="force",
        show_reference=True,
        show_labels=True,
        title="ROIF — Loaded Recording",
        repeat=False,
    )

    print()
    print("Playback verification")
    print("-" * 88)
    print(
        "Simulation steps unchanged: "
        f"{simulation.step_index == step_before_playback}"
    )
    print(
        "Simulation time unchanged : "
        f"{simulation.time == time_before_playback}"
    )


if __name__ == "__main__":
    main()
