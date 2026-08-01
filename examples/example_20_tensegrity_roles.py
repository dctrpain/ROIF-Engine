r"""
ROIF Engine Example 20

Visual verification of tensegrity-aware element roles.

This example does not run physics. It builds a static 2D network and verifies
that NetworkPlotter distinguishes real Element properties:

- rigid elements via ``mechanical_role = "rigid"``;
- tension-only elements via ``tension_only=True``;
- compression-only elements via ``compression_only=True``;
- generic axial elements with no special role.

Run:

    python examples\example_20_tensegrity_roles.py
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
from visualization import NetworkPlotter
from visualization.viewer import ViewerState


def make_material(
    name: str,
    *,
    stiffness: float,
    damping: float = 0.0,
) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=stiffness,
            damping=damping,
            failure_threshold=100.0,
            reference_force=20.0,
            reference_strain=1.0,
        ),
    )


def build_network() -> Network:
    network = Network(
        gravity=[0.0, 0.0],
        global_damping=0.0,
        record_history=False,
    )

    # Two separated rigid struts.
    node_a = Node(
        position=[-1.2, -0.7],
        fixed=True,
        mass=1.0,
        node_id="A",
    )
    node_b = Node(
        position=[-0.4, 0.8],
        fixed=True,
        mass=1.0,
        node_id="B",
    )
    node_c = Node(
        position=[0.4, -0.8],
        fixed=True,
        mass=1.0,
        node_id="C",
    )
    node_d = Node(
        position=[1.2, 0.7],
        fixed=True,
        mass=1.0,
        node_id="D",
    )

    for node in (node_a, node_b, node_c, node_d):
        network.add_node(node)

    rigid_material = make_material(
        "rigid_strut_material",
        stiffness=5000.0,
    )
    cable_material = make_material(
        "tension_cable_material",
        stiffness=180.0,
        damping=1.0,
    )
    compression_material = make_material(
        "compression_material",
        stiffness=350.0,
    )
    generic_material = make_material(
        "generic_axial_material",
        stiffness=120.0,
    )

    strut_ab = Element(
        node_a=node_a,
        node_b=node_b,
        material=rigid_material,
        element_id="STRUT_AB",
        name="Rigid_Strut_AB",
        record_history=False,
    )
    strut_ab.mechanical_role = "rigid"

    strut_cd = Element(
        node_a=node_c,
        node_b=node_d,
        material=rigid_material.clone(),
        element_id="STRUT_CD",
        name="Rigid_Strut_CD",
        record_history=False,
    )
    strut_cd.mechanical_role = "rigid"

    cable_ac = Element(
        node_a=node_a,
        node_b=node_c,
        material=cable_material,
        element_id="CABLE_AC",
        name="Tension_Cable_AC",
        tension_only=True,
        record_history=False,
    )

    cable_ad = Element(
        node_a=node_a,
        node_b=node_d,
        material=cable_material.clone(),
        element_id="CABLE_AD",
        name="Tension_Cable_AD",
        tension_only=True,
        record_history=False,
    )

    cable_bc = Element(
        node_a=node_b,
        node_b=node_c,
        material=cable_material.clone(),
        element_id="CABLE_BC",
        name="Tension_Cable_BC",
        tension_only=True,
        record_history=False,
    )

    cable_bd = Element(
        node_a=node_b,
        node_b=node_d,
        material=cable_material.clone(),
        element_id="CABLE_BD",
        name="Tension_Cable_BD",
        tension_only=True,
        record_history=False,
    )

    compression_bc = Element(
        node_a=node_b,
        node_b=node_c,
        material=compression_material,
        element_id="COMPRESSION_BC",
        name="Compression_Only_BC",
        compression_only=True,
        record_history=False,
    )

    generic_ab = Element(
        node_a=node_a,
        node_b=node_d,
        material=generic_material,
        element_id="GENERIC_AD",
        name="Generic_Axial_AD",
        record_history=False,
    )

    for element in (
        strut_ab,
        strut_cd,
        cable_ac,
        cable_ad,
        cable_bc,
        cable_bd,
        compression_bc,
        generic_ab,
    ):
        network.add_element(element)

    return network


def print_roles(network: Network) -> None:
    print("=" * 88)
    print("ROIF Engine - Example 20")
    print("Tensegrity-aware element roles")
    print("=" * 88)
    print()

    for element in network.elements:
        role = ViewerState.element_role(element)

        print(
            f"{str(element.id):18s} "
            f"role={role.value:16s} "
            f"tension_only={element.tension_only!s:5s} "
            f"compression_only={element.compression_only!s:5s}"
        )


def main() -> None:
    network = build_network()

    print_roles(network)

    print()
    print("Opening geometry viewer...")
    print("Expected visual roles:")
    print("  rigid            : thick dark struts")
    print("  tension          : thin blue cables")
    print("  compression      : thicker purple element")
    print("  generic          : standard axial element")
    print()
    print("Close the Matplotlib window to return to PowerShell.")

    NetworkPlotter(
        network
    ).show(
        mode="geometry",
        show_reference=False,
        show_labels=True,
        title="ROIF — Tensegrity Element Roles",
    )


if __name__ == "__main__":
    main()