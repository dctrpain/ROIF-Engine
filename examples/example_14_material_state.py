r"""
ROIF Engine Example 14

NetworkPlotter v1.3 material-state modes:
    damage
    fatigue
    integrity
    remodeling

Run from the project root:

    python examples\example_14_material_state.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.element import Element
from core.material import (
    Material,
    MaterialParameters,
    MaterialState,
)
from core.network import Network
from core.node import Node
from visualization import NetworkPlotter


def make_material(
    name: str,
    *,
    damage: float,
    fatigue: float,
    remodeling: float,
) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=100.0,
            damping=0.0,
            failure_threshold=100.0,
            reference_force=20.0,
            reference_strain=1.0,
        ),
        state=MaterialState(
            damage=damage,
            fatigue=fatigue,
            remodeling=remodeling,
        ),
        reference_length=1.0,
    )


def add_element(
    network: Network,
    node_a: Node,
    node_b: Node,
    name: str,
    *,
    damage: float,
    fatigue: float,
    remodeling: float,
) -> None:
    network.add_element(
        Element(
            node_a=node_a,
            node_b=node_b,
            material=make_material(
                f"{name}_material",
                damage=damage,
                fatigue=fatigue,
                remodeling=remodeling,
            ),
            element_id=name,
            name=name,
        )
    )


def build_network() -> Network:
    network = Network()

    node_a = Node(position=[0.0, 0.0], fixed=True, node_id="A")
    node_b = Node(position=[1.0, 0.0], fixed=True, node_id="B")
    node_c = Node(position=[1.25, 0.82], node_id="C")
    node_d = Node(position=[0.18, 1.15], node_id="D")

    for node in (node_a, node_b, node_c, node_d):
        network.add_node(node)

    add_element(
        network, node_a, node_b, "AB",
        damage=0.05, fatigue=0.10, remodeling=0.90,
    )
    add_element(
        network, node_b, node_c, "BC",
        damage=0.20, fatigue=0.35, remodeling=0.70,
    )
    add_element(
        network, node_c, node_d, "CD",
        damage=0.45, fatigue=0.25, remodeling=0.55,
    )
    add_element(
        network, node_d, node_a, "DA",
        damage=0.70, fatigue=0.60, remodeling=0.40,
    )
    add_element(
        network, node_a, node_c, "AC",
        damage=0.30, fatigue=0.80, remodeling=0.95,
    )
    add_element(
        network, node_b, node_d, "BD",
        damage=0.55, fatigue=0.45, remodeling=0.65,
    )

    return network


def print_material_state(
    plotter: NetworkPlotter,
    network: Network,
) -> None:
    print()
    print("Element material state")
    print("-" * 104)
    print(
        f"{'element':<12}"
        f"{'damage':>14}"
        f"{'fatigue':>14}"
        f"{'integrity':>14}"
        f"{'remodeling':>16}"
    )

    for element in network.elements:
        print(
            f"{str(element.id):<12}"
            f"{plotter.element_material_value(element, 'damage'):>14.6f}"
            f"{plotter.element_material_value(element, 'fatigue'):>14.6f}"
            f"{plotter.element_material_value(element, 'integrity'):>14.6f}"
            f"{plotter.element_material_value(element, 'remodeling'):>16.6f}"
        )


def main() -> None:
    print("=" * 104)
    print("ROIF Engine - Example 14")
    print("NetworkPlotter v1.3 — Material State")
    print("=" * 104)

    network = build_network()
    plotter = NetworkPlotter(network)

    print_material_state(plotter, network)

    mode = "damage"

    print()
    print(f"Opening material-state visualization: {mode}")

    plotter.show(
        mode=mode,
        show_reference=False,
        show_labels=True,
        show_colorbar=True,
        title=f"ROIF Engine — Material State: {mode}",
    )


if __name__ == "__main__":
    main()
