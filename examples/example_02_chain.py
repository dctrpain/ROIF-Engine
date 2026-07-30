"""
ROIF Engine Example 02

Chain of Elastic Elements

This example demonstrates how multiple elements
can be connected to form a simple mechanical chain.

Structure:

Fixed ---- Node ---- Node

The external node is displaced.

The simulation demonstrates how forces propagate
through multiple connected elements.

New concepts introduced:

- Multiple elements
- Force transmission
- Network behaviour
"""

from roif_engine.core.node import Node
from roif_engine.core.element import Element
from roif_engine.core.material import Material
from roif_engine.core.network import Network


def print_state(network, elements):

    print("-" * 65)

    for i, node in enumerate(network.nodes):
        print(
            f"Node {i}: "
            f"({node.position[0]:.4f}, "
            f"{node.position[1]:.4f})"
        )

    print()

    for i, element in enumerate(elements):
        print(
            f"Element {i}: "
            f"Length={element.current_length():.4f}   "
            f"Extension={element.extension():.4f}"
        )

    print()


def main():

    print("=" * 65)
    print("ROIF Engine Example 02")
    print("Elastic Chain")
    print("=" * 65)

    material = Material(
        stiffness=100.0,
        damping=1.0,
    )

    n0 = Node(
        position=[0.0, 0.0],
        mass=1.0,
        fixed=True,
    )

    n1 = Node(
        position=[1.0, 0.0],
        mass=1.0,
    )

    # Stretch last node
    n2 = Node(
        position=[2.2, 0.0],
        mass=1.0,
    )

    e0 = Element(
        node_a=n0,
        node_b=n1,
        rest_length=1.0,
        material=material,
    )

    e1 = Element(
        node_a=n1,
        node_b=n2,
        rest_length=1.0,
        material=material,
    )

    network = Network()

    network.add_node(n0)
    network.add_node(n1)
    network.add_node(n2)

    network.add_element(e0)
    network.add_element(e1)

    print("\nInitial configuration\n")
    print_state(network, [e0, e1])

    dt = 0.01
    steps = 50

    for _ in range(steps):
        network.step(dt)

    print("\nConfiguration after simulation\n")
    print_state(network, [e0, e1])

    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()