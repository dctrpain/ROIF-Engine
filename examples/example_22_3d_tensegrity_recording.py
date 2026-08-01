r"""
ROIF Engine Example 22

Animated 3D tensegrity-prism recording driven by the real mechanical solver.

The model contains:

- six nodes in 3D;
- three high-stiffness compression-capable struts;
- nine tension-only cables;
- initial cable pretension;
- fixed lower support nodes;
- free upper nodes;
- gravity;
- a lateral initial impulse;
- damping;
- recording to .roifrec;
- offline 3D playback through play.py.

Important
---------
The elements marked as ``mechanical_role = "rigid"`` are still deformable
high-stiffness axial elements. The role currently controls visualization.
The solver does not yet treat them as exact rigid bodies.

Run:

    python examples\example_22_3d_tensegrity_recording.py

Then replay:

    python play.py output\example_22_3d_tensegrity_recording.roifrec --mode geometry

Force view:

    python play.py output\example_22_3d_tensegrity_recording.roifrec --mode force
"""

from __future__ import annotations

from math import cos, pi, sin
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
    / "example_22_3d_tensegrity_recording.roifrec"
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
        material.state.pretension = float(
            pretension
        )

    return material


def triangle_points(
    *,
    radius: float,
    z: float,
    rotation: float,
) -> list[list[float]]:
    points: list[list[float]] = []

    for index in range(3):
        angle = rotation + index * 2.0 * pi / 3.0

        points.append(
            [
                radius * cos(angle),
                radius * sin(angle),
                z,
            ]
        )

    return points


def add_element(
    network: Network,
    node_a: Node,
    node_b: Node,
    *,
    element_id: str,
    material: Material,
    tension_only: bool = False,
    compression_only: bool = False,
    mechanical_role: str | None = None,
) -> Element:
    element = Element(
        node_a=node_a,
        node_b=node_b,
        material=material,
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
        gravity=[0.0, 0.0, -1.5],
        global_damping=1.6,
        record_history=False,
    )

    lower_positions = triangle_points(
        radius=1.0,
        z=-0.9,
        rotation=0.0,
    )

    upper_positions = triangle_points(
        radius=0.95,
        z=0.9,
        rotation=pi / 3.0,
    )

    lower_nodes = [
        Node(
            position=position,
            velocity=[0.0, 0.0, 0.0],
            fixed=True,
            mass=1.0,
            node_id=f"L{index + 1}",
        )
        for index, position in enumerate(
            lower_positions
        )
    ]

    upper_nodes = [
        Node(
            position=position,
            velocity=[
                0.35 if index == 0 else -0.12,
                0.10 if index == 1 else -0.05,
                0.0,
            ],
            fixed=False,
            mass=0.7,
            node_id=f"U{index + 1}",
        )
        for index, position in enumerate(
            upper_positions
        )
    ]

    for node in (
        *lower_nodes,
        *upper_nodes,
    ):
        network.add_node(node)

    strut_material = make_material(
        "high_stiffness_3d_strut",
        stiffness=2200.0,
        damping=12.0,
        reference_force=600.0,
    )

    cable_material = make_material(
        "pretensioned_3d_cable",
        stiffness=260.0,
        damping=4.5,
        pretension=10.0,
        reference_force=120.0,
    )

    # Three crossing struts.
    strut_pairs = (
        (lower_nodes[0], upper_nodes[1]),
        (lower_nodes[1], upper_nodes[2]),
        (lower_nodes[2], upper_nodes[0]),
    )

    for index, (
        node_a,
        node_b,
    ) in enumerate(
        strut_pairs,
        start=1,
    ):
        add_element(
            network,
            node_a,
            node_b,
            element_id=f"STRUT_{index}",
            material=strut_material.clone(),
            mechanical_role="rigid",
        )

    # Lower triangle cables.
    lower_cables = (
        (lower_nodes[0], lower_nodes[1]),
        (lower_nodes[1], lower_nodes[2]),
        (lower_nodes[2], lower_nodes[0]),
    )

    for index, (
        node_a,
        node_b,
    ) in enumerate(
        lower_cables,
        start=1,
    ):
        add_element(
            network,
            node_a,
            node_b,
            element_id=f"LOWER_CABLE_{index}",
            material=cable_material.clone(),
            tension_only=True,
        )

    # Upper triangle cables.
    upper_cables = (
        (upper_nodes[0], upper_nodes[1]),
        (upper_nodes[1], upper_nodes[2]),
        (upper_nodes[2], upper_nodes[0]),
    )

    for index, (
        node_a,
        node_b,
    ) in enumerate(
        upper_cables,
        start=1,
    ):
        add_element(
            network,
            node_a,
            node_b,
            element_id=f"UPPER_CABLE_{index}",
            material=cable_material.clone(),
            tension_only=True,
        )

    # Vertical/diagonal side cables.
    side_cables = (
        (lower_nodes[0], upper_nodes[0]),
        (lower_nodes[1], upper_nodes[1]),
        (lower_nodes[2], upper_nodes[2]),
    )

    for index, (
        node_a,
        node_b,
    ) in enumerate(
        side_cables,
        start=1,
    ):
        add_element(
            network,
            node_a,
            node_b,
            element_id=f"SIDE_CABLE_{index}",
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
        role = ViewerState.element_role(
            element
        )

        print(
            f"{str(element.id):20s} "
            f"role={role.value:12s} "
            f"tension_only={str(element.tension_only):5s}"
        )


def main() -> None:
    print("=" * 88)
    print("ROIF Engine - Example 22")
    print("Animated 3D tensegrity prism")
    print("=" * 88)

    network = build_network()

    print_element_roles(network)

    simulation = Simulation(
        network,
        dt=0.0004,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=False,
    )

    print()
    print("Running 3D physical simulation...")
    print("-" * 88)

    recording = SimulationRecorder(
        simulation
    ).run_steps(
        6000,
        sample_every=12,
        include_initial=True,
        include_final=True,
        metadata={
            "name": "animated 3D tensegrity prism",
            "purpose": (
                "three-dimensional solver-driven "
                "tensegrity visualization"
            ),
            "strut_count": 3,
            "cable_count": 9,
            "strut_model": (
                "high-stiffness deformable axial element"
            ),
            "cable_model": "tension-only axial element",
            "pretension": 10.0,
            "initial_excitation": (
                "lateral velocity applied to upper nodes"
            ),
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
    print("Geometry playback:")
    print(
        "python play.py "
        "output\\example_22_3d_tensegrity_recording.roifrec "
        "--mode geometry --interval 20"
    )
    print()
    print("Force playback:")
    print(
        "python play.py "
        "output\\example_22_3d_tensegrity_recording.roifrec "
        "--mode force --interval 20"
    )
    print()
    print(
        "The Matplotlib 3D scene can be rotated "
        "with the mouse during playback."
    )


if __name__ == "__main__":
    main()