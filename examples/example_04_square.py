"""
ROIF Engine Example 04

Square Network Without Diagonal Bracing

This example creates a four-node network:

    Node 3 -------- Node 2
      |                |
      |                |
    Node 0 -------- Node 1

Node 0 and Node 1 are fixed.

The upper pair of nodes begins displaced to the right and slightly
above the reference configuration.

The example demonstrates:

- a four-node closed network;
- force redistribution through four edge elements;
- deformation of a quadrilateral;
- why a square without diagonal bracing is less geometrically
  stable than a triangle.

A square made only from edge elements can change shape toward a
parallelogram while preserving approximately the same edge lengths.
"""

from pathlib import Path
import sys


# ----------------------------------------------------------------------
# Make the project root visible to Python
# ----------------------------------------------------------------------
#
# This allows the example to be launched from the project root:
#
#     python examples\example_04_square.py
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
    Create one independent elastic material.

    Each element receives its own material instance because material
    objects can store fatigue, damage, remodeling, pretension, and
    other internal state.
    """

    parameters = MaterialParameters(
        stiffness=100.0,
        damping=3.0,
    )

    return Material(
        name=name,
        parameters=parameters,
    )


def evaluate_element_forces(network: Network) -> None:
    """
    Evaluate the current element forces without advancing time.

    This updates each element's last_force diagnostic value.
    """

    network.clear_forces()

    network.assemble_element_forces(
        include_active=False,
    )


def print_state(
    title: str,
    network: Network,
    elements: list[Element],
) -> None:
    """Print node positions and element diagnostics."""

    evaluate_element_forces(network)

    print()
    print(title)
    print("-" * 76)

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
        print(
            f"Element {index}: "
            f"name={element.name}, "
            f"length={element.current_length():.6f}, "
            f"extension={element.extension():.6f}, "
            f"force={element.last_force.total:.6f} N"
        )


def main() -> None:
    """Run the square-network simulation."""

    print("=" * 76)
    print("ROIF Engine - Example 04")
    print("Square Network Without Diagonal Bracing")
    print("=" * 76)

    # ------------------------------------------------------------------
    # Create nodes
    # ------------------------------------------------------------------
    #
    # Reference configuration:
    #
    #     node_3 = [0.0, 1.0]
    #     node_2 = [1.0, 1.0]
    #
    # Initial configuration:
    #
    #     node_3 = [0.2, 1.1]
    #     node_2 = [1.2, 1.1]
    #
    # The upper edge remains horizontal, but the whole upper part is
    # shifted to the right. Both side elements are initially stretched.
    #

    node_0 = Node(
        position=[0.0, 0.0],
        mass=1.0,
        fixed=True,
        node_id="node_0",
    )

    node_1 = Node(
        position=[1.0, 0.0],
        mass=1.0,
        fixed=True,
        node_id="node_1",
    )

    node_2 = Node(
        position=[1.2, 1.1],
        mass=1.0,
        fixed=False,
        node_id="node_2",
    )

    node_3 = Node(
        position=[0.2, 1.1],
        mass=1.0,
        fixed=False,
        node_id="node_3",
    )

    # ------------------------------------------------------------------
    # Create square edge elements
    # ------------------------------------------------------------------
    #
    # All four reference edge lengths are 1.0.
    #
    # No diagonal elements are added in this example.
    #

    bottom_element = Element(
        node_a=node_0,
        node_b=node_1,
        rest_length=1.0,
        material=create_material("bottom_material"),
        element_id="bottom",
        name="Bottom edge",
    )

    right_element = Element(
        node_a=node_1,
        node_b=node_2,
        rest_length=1.0,
        material=create_material("right_material"),
        element_id="right",
        name="Right edge",
    )

    top_element = Element(
        node_a=node_2,
        node_b=node_3,
        rest_length=1.0,
        material=create_material("top_material"),
        element_id="top",
        name="Top edge",
    )

    left_element = Element(
        node_a=node_3,
        node_b=node_0,
        rest_length=1.0,
        material=create_material("left_material"),
        element_id="left",
        name="Left edge",
    )

    elements = [
        bottom_element,
        right_element,
        top_element,
        left_element,
    ]

    # ------------------------------------------------------------------
    # Create network
    # ------------------------------------------------------------------

    network = Network(
        gravity=None,
        global_damping=0.4,
        record_history=True,
    )

    # Nodes must be added before elements.
    network.add_node(node_0)
    network.add_node(node_1)
    network.add_node(node_2)
    network.add_node(node_3)

    for element in elements:
        network.add_element(element)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    print_state(
        title="Initial configuration",
        network=network,
        elements=elements,
    )

    initial_top_center_x = (
        node_2.position[0]
        + node_3.position[0]
    ) / 2.0

    initial_top_center_y = (
        node_2.position[1]
        + node_3.position[1]
    ) / 2.0

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------

    time_step = 0.001
    number_of_steps = 3000

    for _ in range(number_of_steps):
        network.step(
            time_step,
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=True,
        )

    # ------------------------------------------------------------------
    # Final state
    # ------------------------------------------------------------------

    print_state(
        title="Configuration after simulation",
        network=network,
        elements=elements,
    )

    final_top_center_x = (
        node_2.position[0]
        + node_3.position[0]
    ) / 2.0

    final_top_center_y = (
        node_2.position[1]
        + node_3.position[1]
    ) / 2.0

    # ------------------------------------------------------------------
    # Network diagnostics
    # ------------------------------------------------------------------

    stats = network.last_step_stats

    print()
    print("Final diagnostics")
    print("-" * 76)
    print(f"Simulation time      : {network.time:.6f} s")
    print(f"Step count           : {network.step_index}")
    print(f"Kinetic energy       : {stats.kinetic_energy:.9f}")
    print(f"Elastic energy       : {stats.elastic_energy:.9f}")
    print(f"Total energy         : {stats.total_energy:.9f}")
    print(f"Maximum force        : {stats.maximum_force:.9f} N")
    print(f"Maximum velocity     : {stats.maximum_velocity:.9f}")
    print(
        f"Initial top center   : "
        f"({initial_top_center_x:.6f}, "
        f"{initial_top_center_y:.6f})"
    )
    print(
        f"Final top center     : "
        f"({final_top_center_x:.6f}, "
        f"{final_top_center_y:.6f})"
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print()
    print("Interpretation")
    print("-" * 76)
    print(
        "The two side elements begin in extension and generate"
    )
    print(
        "restoring forces that pull the upper nodes toward the base."
    )
    print(
        "The top element couples the motion of the two upper nodes."
    )
    print(
        "However, the network contains no diagonal connection."
    )
    print(
        "Therefore, the quadrilateral can shear and approach a"
    )
    print(
        "parallelogram while keeping its edge lengths near their"
    )
    print(
        "reference values."
    )
    print(
        "The next example will add diagonal bracing and compare the"
    )
    print(
        "resulting increase in geometric stability."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()