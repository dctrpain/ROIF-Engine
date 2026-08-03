r"""
ROIF Engine Example 23

Dynamic 3D tensegrity-inspired upper-limb model.

This is a simplified mechanical analogy of an upper limb, not an
anatomically exact model.

The model contains:

- a fixed shoulder-girdle frame;
- a high-stiffness humeral frame;
- separate radius-like and ulna-like segments;
- a rigid palm frame;
- three simplified finger rays;
- passive tension-only soft tissues;
- shoulder antagonistic cable geometry;
- elbow stabilizers;
- an interosseous membrane analogue;
- wrist stabilizers;
- flexor/extensor tendon analogues;
- initial prestress from shortened cable rest lengths;
- a clearly visible impulse applied to the palm and fingers;
- real solver-driven 3D dynamics;
- recording to .roifrec.

Important
---------
Rigid elements are high-stiffness deformable axial elements, not exact
mathematical rigid bodies.

Soft tissues are passive tension-only elements. Active muscle control will
be introduced in a later example.

Run:

    python examples\example_23_3d_arm_tensegrity.py

Geometry playback:

    python play.py output\example_23_3d_arm_tensegrity.roifrec \
        --mode geometry --interval 12 --repeat

Force playback:

    python play.py output\example_23_3d_arm_tensegrity.roifrec \
        --mode force --interval 12 --repeat
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys

import numpy as np


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
    / "example_23_3d_arm_tensegrity.roifrec"
)


def distance(
    node_a: Node,
    node_b: Node,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(node_b.position, dtype=float)
            - np.asarray(node_a.position, dtype=float)
        )
    )


def make_material(
    name: str,
    *,
    stiffness: float,
    damping: float,
    reference_force: float,
    failure_threshold: float = 1000.0,
) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=stiffness,
            damping=damping,
            failure_threshold=failure_threshold,
            overload_threshold=10.0,
            reference_force=reference_force,
            reference_strain=1.0,
        ),
    )


def add_nodes(
    network: Network,
    nodes: Iterable[Node],
) -> None:
    for node in nodes:
        network.add_node(node)


def add_rigid(
    network: Network,
    node_a: Node,
    node_b: Node,
    *,
    element_id: str,
    material: Material,
) -> Element:
    element = Element(
        node_a=node_a,
        node_b=node_b,
        material=material.clone(),
        rest_length=distance(node_a, node_b),
        element_id=element_id,
        name=element_id,
        tension_only=False,
        compression_only=False,
        record_history=False,
    )

    element.mechanical_role = "rigid"
    network.add_element(element)

    return element


def add_tissue(
    network: Network,
    node_a: Node,
    node_b: Node,
    *,
    element_id: str,
    material: Material,
    rest_scale: float,
) -> Element:
    if not 0.0 < rest_scale <= 1.0:
        raise ValueError(
            "rest_scale must satisfy 0 < rest_scale <= 1"
        )

    element = Element(
        node_a=node_a,
        node_b=node_b,
        material=material.clone(),
        rest_length=(
            distance(node_a, node_b)
            * float(rest_scale)
        ),
        element_id=element_id,
        name=element_id,
        tension_only=True,
        compression_only=False,
        record_history=False,
    )

    network.add_element(element)

    return element


def add_rigid_paths(
    network: Network,
    paths: Iterable[tuple[Node, Node, str]],
    material: Material,
) -> None:
    for node_a, node_b, element_id in paths:
        add_rigid(
            network,
            node_a,
            node_b,
            element_id=element_id,
            material=material,
        )


def add_tissue_paths(
    network: Network,
    paths: Iterable[tuple[Node, Node, str]],
    material: Material,
    *,
    rest_scale: float,
) -> None:
    for node_a, node_b, element_id in paths:
        add_tissue(
            network,
            node_a,
            node_b,
            element_id=element_id,
            material=material,
            rest_scale=rest_scale,
        )


def build_network() -> Network:
    network = Network(
        gravity=[0.0, 0.0, -2.2],
        global_damping=0.65,
        record_history=False,
    )

    # ================================================================
    # Fixed shoulder-girdle / torso frame
    # ================================================================

    base_upper_front = Node(
        position=[0.0, -0.65, 0.80],
        fixed=True,
        mass=2.0,
        node_id="BASE_UF",
    )
    base_upper_back = Node(
        position=[0.0, 0.65, 0.80],
        fixed=True,
        mass=2.0,
        node_id="BASE_UB",
    )
    base_lower_front = Node(
        position=[0.0, -0.65, -0.80],
        fixed=True,
        mass=2.0,
        node_id="BASE_LF",
    )
    base_lower_back = Node(
        position=[0.0, 0.65, -0.80],
        fixed=True,
        mass=2.0,
        node_id="BASE_LB",
    )

    # ================================================================
    # Humerus spatial frame
    # ================================================================

    humerus_prox_front = Node(
        position=[0.55, -0.18, 0.22],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.75,
        node_id="H_PF",
    )
    humerus_prox_back = Node(
        position=[0.55, 0.18, 0.22],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.75,
        node_id="H_PB",
    )
    humerus_dist_front = Node(
        position=[1.85, -0.16, 0.02],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.70,
        node_id="H_DF",
    )
    humerus_dist_back = Node(
        position=[1.85, 0.16, 0.02],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.70,
        node_id="H_DB",
    )

    # ================================================================
    # Radius-like and ulna-like forearm chains
    # ================================================================

    radius_prox = Node(
        position=[2.10, -0.22, -0.02],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.45,
        node_id="R_P",
    )
    radius_mid = Node(
        position=[2.72, -0.28, -0.05],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.40,
        node_id="R_M",
    )
    radius_dist = Node(
        position=[3.34, -0.24, -0.08],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.35,
        node_id="R_D",
    )

    ulna_prox = Node(
        position=[2.10, 0.22, -0.02],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.45,
        node_id="U_P",
    )
    ulna_mid = Node(
        position=[2.72, 0.28, -0.05],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.40,
        node_id="U_M",
    )
    ulna_dist = Node(
        position=[3.34, 0.24, -0.08],
        velocity=[0.0, 0.0, 0.0],
        fixed=False,
        mass=0.35,
        node_id="U_D",
    )

    # ================================================================
    # Palm
    #
    # The palm receives a coordinated initial impulse. This produces
    # visible wrist rotation and transmits motion proximally.
    # ================================================================

    palm_prox_front = Node(
        position=[3.55, -0.32, -0.10],
        velocity=[0.0, 1.25, 0.55],
        fixed=False,
        mass=0.30,
        node_id="P_PF",
    )
    palm_prox_back = Node(
        position=[3.55, 0.32, -0.10],
        velocity=[0.0, 1.00, 0.40],
        fixed=False,
        mass=0.30,
        node_id="P_PB",
    )
    palm_dist_front = Node(
        position=[4.15, -0.38, -0.12],
        velocity=[0.0, 2.10, 0.95],
        fixed=False,
        mass=0.25,
        node_id="P_DF",
    )
    palm_dist_back = Node(
        position=[4.15, 0.38, -0.12],
        velocity=[0.0, 1.65, 0.70],
        fixed=False,
        mass=0.25,
        node_id="P_DB",
    )

    # ================================================================
    # Simplified finger rays
    #
    # Different velocities create visible bending and torsion instead
    # of a pure rigid translation of the entire hand.
    # ================================================================

    finger_front = Node(
        position=[4.75, -0.38, -0.15],
        velocity=[0.0, 3.20, 1.40],
        fixed=False,
        mass=0.16,
        node_id="F_FRONT",
    )
    finger_middle = Node(
        position=[4.82, 0.00, -0.12],
        velocity=[0.0, 3.80, 1.70],
        fixed=False,
        mass=0.17,
        node_id="F_MIDDLE",
    )
    finger_back = Node(
        position=[4.75, 0.38, -0.15],
        velocity=[0.0, 3.00, 1.30],
        fixed=False,
        mass=0.16,
        node_id="F_BACK",
    )

    all_nodes = (
        base_upper_front,
        base_upper_back,
        base_lower_front,
        base_lower_back,
        humerus_prox_front,
        humerus_prox_back,
        humerus_dist_front,
        humerus_dist_back,
        radius_prox,
        radius_mid,
        radius_dist,
        ulna_prox,
        ulna_mid,
        ulna_dist,
        palm_prox_front,
        palm_prox_back,
        palm_dist_front,
        palm_dist_back,
        finger_front,
        finger_middle,
        finger_back,
    )

    add_nodes(network, all_nodes)

    # ================================================================
    # Materials
    # ================================================================

    base_material = make_material(
        "fixed_support_frame",
        stiffness=9000.0,
        damping=12.0,
        reference_force=1000.0,
    )

    bone_material = make_material(
        "high_stiffness_bone_analogue",
        stiffness=6500.0,
        damping=9.0,
        reference_force=900.0,
    )

    ligament_material = make_material(
        "passive_ligament_analogue",
        stiffness=380.0,
        damping=3.5,
        reference_force=180.0,
    )

    # The name intentionally avoids the word "muscle".
    # These elements are passive in Example 23.
    contractile_line_material = make_material(
        "passive_contractile_line_analogue",
        stiffness=250.0,
        damping=3.0,
        reference_force=150.0,
    )

    membrane_material = make_material(
        "interosseous_membrane_analogue",
        stiffness=180.0,
        damping=2.5,
        reference_force=110.0,
    )

    tendon_material = make_material(
        "passive_tendon_analogue",
        stiffness=320.0,
        damping=3.2,
        reference_force=160.0,
    )

    # ================================================================
    # Fixed support frame
    # ================================================================

    add_rigid_paths(
        network,
        (
            (
                base_upper_front,
                base_upper_back,
                "BASE_TOP",
            ),
            (
                base_lower_front,
                base_lower_back,
                "BASE_BOTTOM",
            ),
            (
                base_upper_front,
                base_lower_front,
                "BASE_FRONT",
            ),
            (
                base_upper_back,
                base_lower_back,
                "BASE_BACK",
            ),
            (
                base_upper_front,
                base_lower_back,
                "BASE_DIAGONAL_1",
            ),
            (
                base_upper_back,
                base_lower_front,
                "BASE_DIAGONAL_2",
            ),
        ),
        base_material,
    )

    # ================================================================
    # Humerus rigid frame
    # ================================================================

    add_rigid_paths(
        network,
        (
            (
                humerus_prox_front,
                humerus_prox_back,
                "HUMERUS_PROXIMAL_BAR",
            ),
            (
                humerus_dist_front,
                humerus_dist_back,
                "HUMERUS_DISTAL_BAR",
            ),
            (
                humerus_prox_front,
                humerus_dist_front,
                "HUMERUS_FRONT_SHAFT",
            ),
            (
                humerus_prox_back,
                humerus_dist_back,
                "HUMERUS_BACK_SHAFT",
            ),
            (
                humerus_prox_front,
                humerus_dist_back,
                "HUMERUS_DIAGONAL_1",
            ),
            (
                humerus_prox_back,
                humerus_dist_front,
                "HUMERUS_DIAGONAL_2",
            ),
        ),
        bone_material,
    )

    # ================================================================
    # Shoulder antagonistic and stabilizing paths
    # ================================================================

    add_tissue_paths(
        network,
        (
            (
                base_upper_front,
                humerus_prox_front,
                "SHOULDER_SUPERIOR_FRONT",
            ),
            (
                base_upper_back,
                humerus_prox_back,
                "SHOULDER_SUPERIOR_BACK",
            ),
            (
                base_lower_front,
                humerus_prox_front,
                "SHOULDER_INFERIOR_FRONT",
            ),
            (
                base_lower_back,
                humerus_prox_back,
                "SHOULDER_INFERIOR_BACK",
            ),
            (
                base_upper_front,
                humerus_prox_back,
                "SHOULDER_EXTERNAL_ROTATOR_1",
            ),
            (
                base_lower_back,
                humerus_prox_front,
                "SHOULDER_EXTERNAL_ROTATOR_2",
            ),
            (
                base_upper_back,
                humerus_prox_front,
                "SHOULDER_INTERNAL_ROTATOR_1",
            ),
            (
                base_lower_front,
                humerus_prox_back,
                "SHOULDER_INTERNAL_ROTATOR_2",
            ),
            (
                base_upper_front,
                humerus_dist_back,
                "SHOULDER_LONG_DIAGONAL_1",
            ),
            (
                base_upper_back,
                humerus_dist_front,
                "SHOULDER_LONG_DIAGONAL_2",
            ),
            (
                base_lower_front,
                humerus_dist_back,
                "SHOULDER_LONG_DIAGONAL_3",
            ),
            (
                base_lower_back,
                humerus_dist_front,
                "SHOULDER_LONG_DIAGONAL_4",
            ),
        ),
        contractile_line_material,
        rest_scale=0.965,
    )

    # ================================================================
    # Radius and ulna rigid shafts
    # ================================================================

    add_rigid_paths(
        network,
        (
            (
                radius_prox,
                radius_mid,
                "RADIUS_PROXIMAL_SHAFT",
            ),
            (
                radius_mid,
                radius_dist,
                "RADIUS_DISTAL_SHAFT",
            ),
            (
                ulna_prox,
                ulna_mid,
                "ULNA_PROXIMAL_SHAFT",
            ),
            (
                ulna_mid,
                ulna_dist,
                "ULNA_DISTAL_SHAFT",
            ),
        ),
        bone_material,
    )

    # ================================================================
    # Elbow soft-tissue network
    # ================================================================

    add_tissue_paths(
        network,
        (
            (
                humerus_dist_front,
                radius_prox,
                "ELBOW_RADIAL_COLLATERAL",
            ),
            (
                humerus_dist_back,
                ulna_prox,
                "ELBOW_ULNAR_COLLATERAL",
            ),
            (
                humerus_dist_front,
                ulna_prox,
                "ELBOW_CROSSED_1",
            ),
            (
                humerus_dist_back,
                radius_prox,
                "ELBOW_CROSSED_2",
            ),
            (
                humerus_prox_front,
                radius_prox,
                "ELBOW_FLEXOR_LINE_1",
            ),
            (
                humerus_prox_back,
                ulna_prox,
                "ELBOW_EXTENSOR_LINE_1",
            ),
            (
                humerus_dist_front,
                ulna_mid,
                "ELBOW_ROTATIONAL_STABILIZER_1",
            ),
            (
                humerus_dist_back,
                radius_mid,
                "ELBOW_ROTATIONAL_STABILIZER_2",
            ),
        ),
        ligament_material,
        rest_scale=0.97,
    )

    # ================================================================
    # Interosseous membrane and rotational paths
    # ================================================================

    add_tissue_paths(
        network,
        (
            (
                radius_prox,
                ulna_mid,
                "MEMBRANE_PROXIMAL_DIAGONAL_1",
            ),
            (
                ulna_prox,
                radius_mid,
                "MEMBRANE_PROXIMAL_DIAGONAL_2",
            ),
            (
                radius_mid,
                ulna_dist,
                "MEMBRANE_DISTAL_DIAGONAL_1",
            ),
            (
                ulna_mid,
                radius_dist,
                "MEMBRANE_DISTAL_DIAGONAL_2",
            ),
            (
                radius_prox,
                ulna_prox,
                "MEMBRANE_PROXIMAL_TRANSVERSE",
            ),
            (
                radius_mid,
                ulna_mid,
                "MEMBRANE_MIDDLE_TRANSVERSE",
            ),
            (
                radius_dist,
                ulna_dist,
                "MEMBRANE_DISTAL_TRANSVERSE",
            ),
            (
                humerus_dist_front,
                radius_dist,
                "PRONATOR_LONG_PATH",
            ),
            (
                humerus_dist_back,
                ulna_dist,
                "SUPINATOR_LONG_PATH",
            ),
            (
                humerus_dist_front,
                ulna_dist,
                "PRONATOR_CROSSED_PATH",
            ),
            (
                humerus_dist_back,
                radius_dist,
                "SUPINATOR_CROSSED_PATH",
            ),
        ),
        membrane_material,
        rest_scale=0.975,
    )

    # ================================================================
    # Palm rigid frame
    # ================================================================

    add_rigid_paths(
        network,
        (
            (
                palm_prox_front,
                palm_prox_back,
                "PALM_PROXIMAL_BAR",
            ),
            (
                palm_dist_front,
                palm_dist_back,
                "PALM_DISTAL_BAR",
            ),
            (
                palm_prox_front,
                palm_dist_front,
                "PALM_FRONT_RAY",
            ),
            (
                palm_prox_back,
                palm_dist_back,
                "PALM_BACK_RAY",
            ),
            (
                palm_prox_front,
                palm_dist_back,
                "PALM_DIAGONAL_1",
            ),
            (
                palm_prox_back,
                palm_dist_front,
                "PALM_DIAGONAL_2",
            ),
        ),
        bone_material,
    )

    # ================================================================
    # Wrist stabilizers
    # ================================================================

    add_tissue_paths(
        network,
        (
            (
                radius_dist,
                palm_prox_front,
                "WRIST_RADIAL_FRONT",
            ),
            (
                radius_dist,
                palm_prox_back,
                "WRIST_RADIAL_CROSSED",
            ),
            (
                ulna_dist,
                palm_prox_back,
                "WRIST_ULNAR_BACK",
            ),
            (
                ulna_dist,
                palm_prox_front,
                "WRIST_ULNAR_CROSSED",
            ),
            (
                radius_mid,
                palm_dist_front,
                "WRIST_FLEXOR_LONG_1",
            ),
            (
                ulna_mid,
                palm_dist_back,
                "WRIST_EXTENSOR_LONG_1",
            ),
            (
                radius_mid,
                palm_dist_back,
                "WRIST_DEVIATION_DIAGONAL_1",
            ),
            (
                ulna_mid,
                palm_dist_front,
                "WRIST_DEVIATION_DIAGONAL_2",
            ),
        ),
        ligament_material,
        rest_scale=0.97,
    )

    # ================================================================
    # Simplified finger rays
    # ================================================================

    add_rigid_paths(
        network,
        (
            (
                palm_dist_front,
                finger_front,
                "FINGER_FRONT_RAY",
            ),
            (
                palm_dist_front,
                finger_middle,
                "FINGER_MIDDLE_FRONT_RAY",
            ),
            (
                palm_dist_back,
                finger_middle,
                "FINGER_MIDDLE_BACK_RAY",
            ),
            (
                palm_dist_back,
                finger_back,
                "FINGER_BACK_RAY",
            ),
        ),
        bone_material,
    )

    # ================================================================
    # Flexor, extensor and web tendon analogues
    # ================================================================

    add_tissue_paths(
        network,
        (
            (
                radius_dist,
                finger_front,
                "FLEXOR_TENDON_FRONT",
            ),
            (
                radius_dist,
                finger_middle,
                "FLEXOR_TENDON_MIDDLE",
            ),
            (
                radius_dist,
                finger_back,
                "FLEXOR_TENDON_BACK",
            ),
            (
                ulna_dist,
                finger_front,
                "EXTENSOR_TENDON_FRONT",
            ),
            (
                ulna_dist,
                finger_middle,
                "EXTENSOR_TENDON_MIDDLE",
            ),
            (
                ulna_dist,
                finger_back,
                "EXTENSOR_TENDON_BACK",
            ),
            (
                palm_prox_front,
                finger_back,
                "PALMAR_CROSSED_TENDON_1",
            ),
            (
                palm_prox_back,
                finger_front,
                "PALMAR_CROSSED_TENDON_2",
            ),
            (
                finger_front,
                finger_middle,
                "FINGER_WEB_FRONT",
            ),
            (
                finger_middle,
                finger_back,
                "FINGER_WEB_BACK",
            ),
        ),
        tendon_material,
        rest_scale=0.98,
    )

    return network


def print_summary(
    network: Network,
) -> None:
    role_counts: dict[str, int] = {}

    for element in network.elements:
        role = ViewerState.element_role(element).value
        role_counts[role] = role_counts.get(role, 0) + 1

    print()
    print("Model summary")
    print("-" * 96)
    print(f"Nodes          : {len(network.nodes)}")
    print(f"Elements       : {len(network.elements)}")

    for role, count in sorted(role_counts.items()):
        print(f"{role:14s}: {count}")

    print()
    print("Interpretation")
    print("-" * 96)
    print("BASE_*       : fixed shoulder-girdle frame")
    print("H_*          : humeral spatial frame")
    print("R_* / U_*    : radius-like and ulna-like chains")
    print("P_* / F_*    : palm and simplified finger rays")
    print("Rigid lines  : high-stiffness bone analogues")
    print("Soft lines   : passive tension-only tissues")
    print("Excitation   : coordinated palm and finger impulse")
    print("Duration     : 3.0 seconds of physical time")


def main() -> None:
    print("=" * 96)
    print("ROIF Engine - Example 23")
    print("Dynamic 3D tensegrity-inspired upper-limb model")
    print("=" * 96)

    network = build_network()
    print_summary(network)

    simulation = Simulation(
        network,
        dt=0.0002,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=False,
    )

    print()
    print("Running physical simulation...")
    print("-" * 96)

    recording = SimulationRecorder(
        simulation
    ).run_steps(
        15000,
        sample_every=20,
        include_initial=True,
        include_final=True,
        metadata={
            "name": (
                "dynamic 3D tensegrity-inspired upper limb"
            ),
            "purpose": (
                "visible passive cascade after distal impulse"
            ),
            "model_scope": (
                "simplified shoulder, arm, forearm, palm, fingers"
            ),
            "anatomical_accuracy": False,
            "bone_model": (
                "high-stiffness deformable axial frame"
            ),
            "soft_tissue_model": (
                "passive tension-only prestressed elements"
            ),
            "active_muscles": False,
            "external_excitation": (
                "coordinated initial velocity of palm and fingers"
            ),
            "physical_duration_seconds": 3.0,
            "requested_steps": 15000,
            "sample_every": 20,
        },
    )

    saved_path = RecordingStorage.save(
        recording,
        OUTPUT_PATH,
        compress=True,
        overwrite=True,
    )

    info = RecordingStorage.inspect(
        saved_path,
        verify_checksum=True,
    )

    print(f"Physical steps : {info.physical_steps}")
    print(f"Frames         : {info.frame_count}")
    print(f"Duration       : {info.duration:.6f} s")
    print(f"File           : {saved_path}")
    print(
        f"File size      : "
        f"{saved_path.stat().st_size:,} bytes"
    )
    print(f"SHA-256        : {info.payload_sha256}")

    print()
    print("Final key-node positions")
    print("-" * 96)

    key_ids = {
        "H_PF",
        "H_DF",
        "R_D",
        "U_D",
        "P_DF",
        "P_DB",
        "F_FRONT",
        "F_MIDDLE",
        "F_BACK",
    }

    for node in simulation.network.nodes:
        if str(node.id) not in key_ids:
            continue

        coordinates = ", ".join(
            f"{float(value):+.6f}"
            for value in node.position
        )

        print(
            f"{str(node.id):10s} "
            f"position=[{coordinates}]"
        )

    print()
    print("Recording created successfully.")
    print()
    print("Geometry playback:")
    print(
        "python play.py "
        "output\\example_23_3d_arm_tensegrity.roifrec "
        "--mode geometry --interval 12 --repeat"
    )
    print()
    print("Force playback:")
    print(
        "python play.py "
        "output\\example_23_3d_arm_tensegrity.roifrec "
        "--mode force --interval 12 --repeat"
    )
    print()
    print(
        "The distal impulse is intentionally strong so the "
        "motion is visible without numerical interpretation."
    )


if __name__ == "__main__":
    main()