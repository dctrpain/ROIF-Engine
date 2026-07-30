"""
ROIF Engine Example 01

Single Elastic Element

This example demonstrates the smallest possible biomechanical
simulation using ROIF Engine.

The example creates:

    Node ----- Element ----- Node

One node is fixed.
The second node is displaced.
The resulting elastic force is computed.

This example introduces:

- Node
- Material
- Element
- Network
- Simulation step
"""

from roif_engine.core.node import Node
from roif_engine.core.element import Element
from roif_engine.core.material import Material
from roif_engine.core.network import Network


def main():

    print("=" * 60)
    print("ROIF Engine - Example 01")
    print("Single Elastic Element")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create two nodes
    # ---------------------------------------------------------

    node_a = Node(
        position=[0.0, 0.0],
        mass=1.0,
        fixed=True,
    )

    node_b = Node(
        position=[1.10, 0.0],
        mass=1.0,
    )

    # ---------------------------------------------------------
    # Create elastic material
    # ---------------------------------------------------------

    material = Material(
        stiffness=100.0,
        damping=0.0,
    )

    # ---------------------------------------------------------
    # Connect nodes
    # ---------------------------------------------------------

    element = Element(
        node_a=node_a,
        node_b=node_b,
        rest_length=1.0,
        material=material,
    )

    # ---------------------------------------------------------
    # Create network
    # ---------------------------------------------------------

    network = Network()

    network.add_node(node_a)
    network.add_node(node_b)

    network.add_element(element)

    # ---------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------

    print("\nInitial State")
    print("-" * 60)

    print(f"Rest length    : {element.rest_length:.3f}")
    print(f"Current length : {element.current_length():.3f}")
    print(f"Extension      : {element.extension():.3f}")

    # ---------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------

    dt = 0.01

    for step in range(10):
        network.step(dt)

    # ---------------------------------------------------------
    # Final state
    # ---------------------------------------------------------

    print("\nFinal State")
    print("-" * 60)

    print(f"Current length : {element.current_length():.6f}")
    print(f"Extension      : {element.extension():.6f}")

    force = material.force(
        extension=element.extension(),
        velocity=0.0,
    )

    print(f"Elastic force  : {force:.6f} N")

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()