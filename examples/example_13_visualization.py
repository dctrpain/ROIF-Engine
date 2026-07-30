r"""
ROIF Engine Example 13

NetworkPlotter v1.0:
- reference geometry;
- current geometry;
- fixed and free nodes;
- node labels.

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


def make_material(name: str) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=100.0,
            damping=0.0,
            failure_threshold=100.0,
        ),
        reference_length=1.0,
    )


def add_element(
    network: Network,
    node_a: Node,
    node_b: Node,
    name: str,
) -> None:
    network.add_element(
        Element(
            node_a=node_a,
            node_b=node_b,
            material=make_material(f"{name}_material"),
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

    add_element(network, lower_left, lower_right, "AB")
    add_element(network, lower_right, upper_right, "BC")
    add_element(network, upper_right, upper_left, "CD")
    add_element(network, upper_left, lower_left, "DA")
    add_element(network, lower_left, upper_right, "AC")
    add_element(network, lower_right, upper_left, "BD")

    return network


def main() -> None:
    print("=" * 88)
    print("ROIF Engine - Example 13")
    print("NetworkPlotter v1.0")
    print("=" * 88)

    network = build_square_network()

    # The undeformed square becomes the stored reference geometry.
    plotter = NetworkPlotter(network)

    # Prescribed displacement creates a visible current geometry.
    network.nodes[2].set_position([1.25, 0.82])
    network.nodes[3].set_position([0.18, 1.15])

    print("Reference geometry captured.")
    print("Current geometry prescribed.")
    print("Opening visualization window...")

    plotter.show(
        show_reference=True,
        show_labels=True,
        title="ROIF Engine — Reference and Current Geometry",
    )


if __name__ == "__main__":
    main()
