r"""
ROIF Engine Example 13

NetworkPlotter v1.2 mechanical-state visualization.

Line width:
    absolute transmitted axial force.

Line color:
    normalized force stimulus.

Run from the project root:

    python examples\example_13_visualization.py
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


def make_material(
    name: str,
    stiffness: float,
) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=stiffness,
            damping=0.0,
            failure_threshold=100.0,
            reference_force=20.0,
            reference_strain=1.0,
        ),
        reference_length=1.0,
    )


def add_element(
    network: Network,
    node_a: Node,
    node_b: Node,
    name: str,
    stiffness: float,
) -> None:
    network.add_element(
        Element(
            node_a=node_a,
            node_b=node_b,
            material=make_material(
                f"{name}_material",
                stiffness,
            ),
            element_id=name,
            name=name,
        )
    )


def build_square_network() -> Network:
    network = Network()

    lower_left = Node(
        position=[0.0, 0.0],
        fixed=True,
        node_id="A",
    )
    lower_right = Node(
        position=[1.0, 0.0],
        fixed=True,
        node_id="B",
    )
    upper_right = Node(
        position=[1.0, 1.0],
        node_id="C",
    )
    upper_left = Node(
        position=[0.0, 1.0],
        node_id="D",
    )

    for node in (
        lower_left,
        lower_right,
        upper_right,
        upper_left,
    ):
        network.add_node(node)

    add_element(network, lower_left, lower_right, "AB", 80.0)
    add_element(network, lower_right, upper_right, "BC", 100.0)
    add_element(network, upper_right, upper_left, "CD", 120.0)
    add_element(network, upper_left, lower_left, "DA", 90.0)
    add_element(network, lower_left, upper_right, "AC", 160.0)
    add_element(network, lower_right, upper_left, "BD", 60.0)

    return network


def evaluate_forces(network: Network) -> None:
    network.clear_forces()
    network.assemble_element_forces(include_active=False)


def main() -> None:
    print("=" * 88)
    print("ROIF Engine - Example 13")
    print("NetworkPlotter v1.2 — Mechanical State")
    print("=" * 88)

    network = build_square_network()

    # The undeformed square becomes the stored reference geometry.
    plotter = NetworkPlotter(network)

    # Prescribed displacement creates different element forces.
    network.nodes[2].set_position([1.25, 0.82])
    network.nodes[3].set_position([0.18, 1.15])

    # The visualizer does not calculate physics. The engine does.
    evaluate_forces(network)

    print()
    print("Element mechanical state")
    print("-" * 88)
    print(
        f"{'element':<12}"
        f"{'force [N]':>16}"
        f"{'force stimulus':>20}"
    )

    for element in network.elements:
        force = plotter.element_force(element)
        stimulus = plotter.element_force_stimulus(element)
        print(
            f"{str(element.id):<12}"
            f"{force:>16.6f}"
            f"{stimulus:>20.6f}"
        )

    print()
    print("Line width represents absolute force.")
    print("Line color represents normalized force stimulus.")
    print("Opening visualization window...")

    plotter.show(
        mode="force",
        show_reference=True,
        show_labels=True,
        show_colorbar=True,
        title="ROIF Engine — Mechanical State",
    )


if __name__ == "__main__":
    main()
