"""
ROIF Engine Example 03

Triangular Elastic Network

This example creates a closed triangular network:

            Node 2
           /      \
          /        \
     Node 0 ------ Node 1

Node 0 and Node 1 are fixed.

Node 2 starts above its equilibrium position. The two diagonal
elements pull it toward the reference configuration.

The example demonstrates:

- closed network topology;
- load redistribution;
- geometric stability;
- multiple force-transmission paths.
"""

from pathlib import Path
import sys


# Add the project root directory to Python's module search path.
#
# This allows the example to be launched from the project root:
#
#     python examples\example_03_triangle.py
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node


def create_material(name: str) -> Material:
    """
    Create an independent elastic material.

    A separate material instance is created for every element.
    This is important because materials may contain their own
    fatigue, damage, remodeling, and pretension state.
    """

    parameters = MaterialParameters(
        stiffness=100.0,
        damping=2.0,
    )

    return Material(
        name=name,
        parameters=parameters,
    )


def print_state(
    title: str,
    network: Network,
    elements: list[Element],
) -> None:
    """Print node positions and element diagnostics."""

    print()
    print(title)
    print("-" * 72)

    for index, node in enumerate(network.nodes):
        print(
            f"Node {index}: "
            f"x={node.position[0]:.6f}, "
            f"y={node.position[1]:.6f}, "
            f"vx={node.velocity[0]:.6f}, "
            f"vy={node.velocity[1]:.6f}, "
            f"fixed={node.fixed}"
        )

    print()

    for index, element in enumerate(elements):
        force = element.material.force(
            current_length=element.current_length(),
            length_velocity=element.length_velocity(),
            reference_length=element.material_reference_length(),
            include_active=False,
            store_state=False,
        )

        print(
            f"Element {index}: "
            f"length={element.current_length():.6f}, "
            f"extension={element.extension():.6f}, "
            f"force={force:.6f} N"
        )


def main() -> None:
    """Run the triangular-network simulation."""

    print("=" * 72)
    print("ROIF Engine - Example 03")
    print("Triangular Elastic Network")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Create nodes
    # ------------------------------------------------------------------
    #
    # The lower nodes form a fixed base.
    #
    # The reference triangle has its upper node at:
    #
    #     x = 1.0
    #     y = 1.0
    #
    # The actual upper node begins at y = 1.2, so both diagonal
    # elements are initially stretched.
    #

    node_0 = Node(
        position=[0.0, 0.0],
        mass=1.0,
        fixed=True,
        node_id="node_0",
    )

    node_1 = Node(
        position=[2.0, 0.0],
        mass=1.0,
        fixed=True,
        node_id="node_1",
    )

    node_2 = Node(
        position=[1.0, 1.2],
        mass=1.0,
        fixed=False,
        node_id="node_2",
    )

    # ------------------------------------------------------------------
    # Define reference geometry
    # ------------------------------------------------------------------
    #
    # The base length is 2.0.
    #
    # Each diagonal has horizontal distance 1.0 and vertical distance
    # 1.0 in the reference configuration:
    #
    #     diagonal length = sqrt(1^2 + 1^2) = sqrt(2)
    #

    diagonal_rest_length = 2.0 ** 0.5

    # ------------------------------------------------------------------
    # Create elements
    # ------------------------------------------------------------------

    base_element = Element(
        node_a=node_0,
        node_b=node_1,
        rest_length=2.0,
        material=create_material("base_material"),
        element_id="base",
        name="Base element",
    )

    left_element = Element(
        node_a=node_0,
        node_b=node_2,
        rest_length=diagonal_rest_length,
        material=create_material("left_material"),
        element_id="left",
        name="Left diagonal",
    )

    right_element = Element(
        node_a=node_1,
        node_b=node_2,
        rest_length=diagonal_rest_length,
        material=create_material("right_material"),
        element_id="right",
        name="Right diagonal",
    )

    elements = [
        base_element,
        left_element,
        right_element,
    ]

    # ------------------------------------------------------------------
    # Create network
    # ------------------------------------------------------------------

    network = Network(
        gravity=None,
        global_damping=0.2,
        record_history=True,
    )

    # Nodes must be added before their elements.
    network.add_node(node_0)
    network.add_node(node_1)
    network.add_node(node_2)

    network.add_element(base_element)
    network.add_element(left_element)
    network.add_element(right_element)

    # ------------------------------------------------------------------
    # Print initial state
    # ------------------------------------------------------------------

    print_state(
        title="Initial configuration",
        network=network,
        elements=elements,
    )

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------

    time_step = 0.001
    number_of_steps = 2000

    for _ in range(number_of_steps):
        network.step(
            time_step,
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=True,
        )

    # ------------------------------------------------------------------
    # Print final state
    # ------------------------------------------------------------------

    print_state(
        title="Configuration after simulation",
        network=network,
        elements=elements,
    )

    # ------------------------------------------------------------------
    # Print network diagnostics
    # ------------------------------------------------------------------

    stats = network.last_step_stats

    print()
    print("Final diagnostics")
    print("-" * 72)
    print(f"Simulation time : {network.time:.6f} s")
    print(f"Step count      : {network.step_index}")
    print(f"Kinetic energy  : {stats.kinetic_energy:.9f}")
    print(f"Elastic energy  : {stats.elastic_energy:.9f}")
    print(f"Total energy    : {stats.total_energy:.9f}")
    print(f"Maximum force   : {stats.maximum_force:.9f} N")
    print(f"Maximum velocity: {stats.maximum_velocity:.9f}")

    print()
    print("Interpretation")
    print("-" * 72)
    print(
        "The upper node begins above the reference configuration."
    )
    print(
        "Both diagonal elements are stretched and generate restoring forces."
    )
    print(
        "Because the geometry is symmetric, the horizontal force components"
    )
    print(
        "cancel while the vertical components pull the node downward."
    )
    print(
        "The closed triangular topology provides two mechanical pathways"
    )
    print(
        "for transmitting the load to the fixed base."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()