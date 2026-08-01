r"""
ROIF Engine Example 21

Dynamic tensegrity-like recording driven by the real mechanical solver.

The model contains:

- two high-stiffness compression-capable struts;
- four tension-only cables;
- initial cable pretension;
- two fixed support nodes;
- two free dynamic nodes;
- gravity, damping, recording, storage, and offline playback.

Important
---------
The elements marked as ``mechanical_role = "rigid"`` are not exact rigid
bodies. They are high-stiffness axial elements. The role currently controls
visual classification, while the mechanical solver still treats them as
deformable axial elements.

Run:

    python examples\example_21_tensegrity_recording.py

Then replay:

    python play.py output\example_21_tensegrity_recording.roifrec

Alternative views:

    python play.py output\example_21_tensegrity_recording.roifrec --mode geometry
    python play.py output\example_21_tensegrity_recording.roifrec --mode force
    python play.py output\example_21_tensegrity_recording.roifrec --mode strain
    python play.py output\example_21_tensegrity_recording.roifrec --mode stress
"""

from __future__ import annotations

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
from visualization.viewer import ViewerState


OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "example_21_tensegrity_recording.roifrec"
)


def make_material(
    name: str,
    *,
    stiffness: float,
    damping: float,
    pretension: float = 0.0,
    reference_force: float = 100.0,
) -> Material:
    material = Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=stiffness,
            damping=damping,
            failure_threshold=1000.0,
            overload_threshold=10.0,
            reference_force=reference_force,
            reference_strain=1.0,
        ),
    )

    if hasattr(material.state, "pretension"):
        material.state.pretension = float(pretension)

    return material


def add_element(
    network: Network,
    node_a: Node,
    node_b: Node,
    *,
    element_id: str,
    material: Material,
    rest_length: float | None = None,
    tension_only: bool = False,
    compression_only: bool = False,
    mechanical_role: str | None = None,
) -> Element:
    element = Element(
        node_a=node_a,
        node_b=node_b,
        material=material,
        rest_length=rest_length,
        element_id=element_id,
        name=element_id,
        tension_only=tension_only,
        compression_only=compression_only,
        record_history=False,
    )

    if mechanical_role is not None:
        element.mechanical_role = mechanical_role

    network.add_element(element)

    return element


def build_network() -> Network:
    network = Network(
        gravity=[0.0, -2.0],
        global_damping=1.8,
        record_history=False,
    )

    # Fixed supports.
    node_a = Node(
        position=[-1.2, -0.65],
        velocity=[0.0, 0.0],
        fixed=True,
        mass=1.0,
        node_id="A",
    )
    node_d = Node(
        position=[1.2, 0.65],
        velocity=[0.0, 0.0],
        fixed=True,
        mass=1.0,
        node_id="D",
    )

    # Free dynamic nodes.
    node_b = Node(
        position=[-0.45, 0.82],
        velocity=[0.0, 0.0],
        fixed=False,
        mass=0.8,
        node_id="B",
    )
    node_c = Node(
        position=[0.45, -0.82],
        velocity=[0.0, 0.0],
        fixed=False,
        mass=0.8,
        node_id="C",
    )

    for node in (
        node_a,
        node_b,
        node_c,
        node_d,
    ):
        network.add_node(node)

    strut_material = make_material(
        "high_stiffness_strut",
        stiffness=1800.0,
        damping=10.0,
        reference_force=500.0,
    )

    cable_material = make_material(
        "pretensioned_tension_cable",
        stiffness=220.0,
        damping=4.0,
        pretension=12.0,
        reference_force=100.0,
    )

    # High-stiffness struts.
    add_element(
        network,
        node_a,
        node_b,
        element_id="STRUT_AB",
        material=strut_material,
        mechanical_role="rigid",
    )

    add_element(
        network,
        node_c,
        node_d,
        element_id="STRUT_CD",
        material=strut_material.clone(),
        mechanical_role="rigid",
    )

    # Tension-only cable network.
    add_element(
        network,
        node_a,
        node_c,
        element_id="CABLE_AC",
        material=cable_material.clone(),
        tension_only=True,
    )

    add_element(
        network,
        node_a,
        node_d,
        element_id="CABLE_AD",
        material=cable_material.clone(),
        tension_only=True,
    )

    add_element(
        network,
        node_b,
        node_c,
        element_id="CABLE_BC",
        material=cable_material.clone(),
        tension_only=True,
    )

    add_element(
        network,
        node_b,
        node_d,
        element_id="CABLE_BD",
        material=cable_material.clone(),
        tension_only=True,
    )

    return network


def print_element_roles(
    network: Network,
) -> None:
    print()
    print("Element roles")
    print("-" * 88)

    for element in network.elements:
        role = ViewerState.element_role(element)

        print(
            f"{str(element.id):16s} "
            f"role={role.value:12s} "
            f"tension_only={str(element.tension_only):5s} "
            f"compression_only={str(element.compression_only):5s}"
        )


def main() -> None:
    print("=" * 88)
    print("ROIF Engine - Example 21")
    print("Dynamic tensegrity-like recording")
    print("=" * 88)

    network = build_network()

    print_element_roles(network)

    simulation = Simulation(
        network,
        dt=0.0005,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=False,
    )

    print()
    print("Running physical simulation...")
    print("-" * 88)

    recording = SimulationRecorder(
        simulation
    ).run_steps(
        4000,
        sample_every=10,
        include_initial=True,
        include_final=True,
        metadata={
            "name": "dynamic tensegrity-like network",
            "purpose": (
                "real solver-driven tensegrity visualization"
            ),
            "strut_model": (
                "high-stiffness deformable axial element"
            ),
            "cable_model": "tension-only axial element",
            "pretension": 12.0,
        },
    )

    output_path = RecordingStorage.save(
        recording,
        OUTPUT_PATH,
        compress=True,
        overwrite=True,
    )

    info = RecordingStorage.inspect(
        output_path,
        verify_checksum=True,
    )

    print(f"Physical steps : {info.physical_steps}")
    print(f"Frames         : {info.frame_count}")
    print(f"Duration       : {info.duration:.6f} s")
    print(f"File           : {info.path}")
    print(f"File size      : {info.file_size_bytes:,} bytes")
    print(f"SHA-256        : {info.payload_sha256}")

    print()
    print("Final node positions")
    print("-" * 88)

    for node in simulation.network.nodes:
        coordinates = ", ".join(
            f"{value:+.6f}"
            for value in node.position
        )

        print(
            f"{str(node.id):4s} "
            f"fixed={str(node.fixed):5s} "
            f"position=[{coordinates}]"
        )

    print()
    print("Recording created successfully.")
    print()
    print("Replay:")
    print(
        "python play.py "
        "output\\example_21_tensegrity_recording.roifrec"
    )
    print()
    print("Role view:")
    print(
        "python play.py "
        "output\\example_21_tensegrity_recording.roifrec "
        "--mode geometry"
    )
    print()
    print("Force view:")
    print(
        "python play.py "
        "output\\example_21_tensegrity_recording.roifrec "
        "--mode force"
    )


if __name__ == "__main__":
    main()