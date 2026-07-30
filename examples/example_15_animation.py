r"""ROIF Engine Example 15 — NetworkAnimator v2.0."""

from math import pi, sin
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node
from visualization import NetworkAnimator

FRAME_COUNT = 120
INTERVAL_MS = 40.0


def make_material(name: str) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=100.0,
            damping=0.0,
            failure_threshold=100.0,
            reference_force=20.0,
            reference_strain=1.0,
        ),
        reference_length=1.0,
    )


def add_element(network: Network, node_a: Node, node_b: Node, name: str) -> None:
    network.add_element(
        Element(
            node_a=node_a,
            node_b=node_b,
            material=make_material(f"{name}_material"),
            element_id=name,
            name=name,
        )
    )


def build_network() -> Network:
    network = Network()
    nodes = [
        Node(position=[0.0, 0.0], fixed=True, node_id="A"),
        Node(position=[1.0, 0.0], fixed=True, node_id="B"),
        Node(position=[1.0, 1.0], fixed=True, node_id="C"),
        Node(position=[0.0, 1.0], fixed=True, node_id="D"),
    ]
    for node in nodes:
        network.add_node(node)
    add_element(network, nodes[0], nodes[1], "AB")
    add_element(network, nodes[1], nodes[2], "BC")
    add_element(network, nodes[2], nodes[3], "CD")
    add_element(network, nodes[3], nodes[0], "DA")
    add_element(network, nodes[0], nodes[2], "AC")
    add_element(network, nodes[1], nodes[3], "BD")
    return network


def main() -> None:
    print("=" * 88)
    print("ROIF Engine - Example 15")
    print("NetworkAnimator v2.0")
    print("=" * 88)

    network = build_network()
    node_c = network.nodes[2]
    node_d = network.nodes[3]

    def update_frame(frame_index: int) -> None:
        phase = 2.0 * pi * frame_index / FRAME_COUNT
        node_c.set_position([
            1.0 + 0.25 * sin(phase),
            1.0 - 0.15 * sin(phase),
        ])
        node_d.set_position([
            0.18 * sin(phase + pi / 2.0),
            1.0 + 0.20 * sin(phase),
        ])
        network.clear_forces()
        network.assemble_element_forces(include_active=False)

    animator = NetworkAnimator(network, update_frame=update_frame)
    print("Opening animated mechanical-state visualization...")
    print("Close the Matplotlib window to finish.")
    animator.show(
        frames=FRAME_COUNT,
        interval_ms=INTERVAL_MS,
        mode="force",
        show_reference=True,
        show_labels=True,
        title="ROIF Engine — Animated Mechanical State",
        repeat=True,
    )


if __name__ == "__main__":
    main()
