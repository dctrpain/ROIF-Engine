r"""
ROIF Engine Example 05

Cross-Braced Square Network

This example extends Example 04 by adding two diagonal elements
to the square network.

Structure:

    Node 3 +----------+ Node 2
           |\        /|
           | \      / |
           |  \    /  |
           |   \  /   |
           |    XX    |
           |   /  \   |
           |  /    \  |
           | /      \ |
           |/        \|
    Node 0 +----------+ Node 1

Node 0 and Node 1 are fixed.

The upper pair of nodes begins displaced to the right and slightly
above the reference configuration.

The example demonstrates:

- diagonal bracing;
- increased geometric rigidity;
- load redistribution through multiple pathways;
- resistance to shear deformation;
- comparison with the unbraced square from Example 04.
"""

from pathlib import Path
import sys


# ----------------------------------------------------------------------
# Make the project root visible to Python
# ----------------------------------------------------------------------
#
# This allows the example to be launched from the project root:
#
#     python examples\example_05_cross_bracing.py
#

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node


def create_material(
    name: str,
    stiffness: float = 100.0,
    damping: float = 3.0,
) -> Material:
    """
    Create one independent elastic material.

    Every element receives its own material instance because a material
    may store fatigue, damage, remodeling, pretension, and other state.
    """

    parameters = MaterialParameters(
        stiffness=stiffness,
        damping=damping,
    )

    return Material(
        name=name,
        parameters=parameters,
    )


def evaluate_element_forces(network: Network) -> None:
    """
    Evaluate current mechanical forces without advancing simulation time.

    The force accumulators are cleared first so diagnostic evaluation
    does not add forces to values remaining from an earlier calculation.
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
    """Print node positions and element-force diagnostics."""

    evaluate_element_forces(network)

    print()
    print(title)
    print("-" * 82)

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


def compute_top_center(
    node_2: Node,
    node_3: Node,
) -> tuple[float, float]:
    """Return the center point between the two upper nodes."""

    center_x = (
        node_2.position[0]
        + node_3.position[0]
    ) / 2.0

    center_y = (
        node_2.position[1]
        + node_3.position[1]
    ) / 2.0

    return (
        float(center_x),
        float(center_y),
    )


def compute_shear_measure(
    node_0: Node,
    node_1: Node,
    node_2: Node,
    node_3: Node,
) -> float:
    """
    Measure horizontal shear of the upper edge relative to the base.

    A value near zero means that the center of the upper edge is
    vertically aligned with the center of the lower edge.

    A positive value means displacement to the right.
    A negative value means displacement to the left.
    """

    bottom_center_x = (
        node_0.position[0]
        + node_1.position[0]
    ) / 2.0

    top_center_x = (
        node_2.position[0]
        + node_3.position[0]
    ) / 2.0

    return float(
        top_center_x
        - bottom_center_x
    )


def compute_shear_reduction(
    initial_shear: float,
    final_shear: float,
) -> tuple[float, float]:
    """
    Return absolute and percentage reduction in shear magnitude.

    The sign of shear represents direction, so reduction is calculated
    using absolute shear magnitudes.
    """

    initial_magnitude = abs(initial_shear)
    final_magnitude = abs(final_shear)

    absolute_reduction = (
        initial_magnitude
        - final_magnitude
    )

    if initial_magnitude == 0.0:
        percentage_reduction = 0.0
    else:
        percentage_reduction = (
            absolute_reduction
            / initial_magnitude
            * 100.0
        )

    return (
        float(absolute_reduction),
        float(percentage_reduction),
    )


def force_mode(force: float, tolerance: float = 1e-9) -> str:
    """
    Describe the axial-force mode.

    Positive force represents tension.
    Negative force represents compression.
    """

    if force > tolerance:
        return "tension"

    if force < -tolerance:
        return "compression"

    return "neutral"


def main() -> None:
    """Run the cross-braced square simulation."""

    print("=" * 82)
    print("ROIF Engine - Example 05")
    print("Cross-Braced Square Network")
    print("=" * 82)

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
    # This matches Example 04 and allows direct comparison.
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
    # Reference lengths
    # ------------------------------------------------------------------

    edge_rest_length = 1.0
    diagonal_rest_length = 2.0 ** 0.5

    # ------------------------------------------------------------------
    # Create square edge elements
    # ------------------------------------------------------------------

    bottom_element = Element(
        node_a=node_0,
        node_b=node_1,
        rest_length=edge_rest_length,
        material=create_material("bottom_material"),
        element_id="bottom",
        name="Bottom edge",
    )

    right_element = Element(
        node_a=node_1,
        node_b=node_2,
        rest_length=edge_rest_length,
        material=create_material("right_material"),
        element_id="right",
        name="Right edge",
    )

    top_element = Element(
        node_a=node_2,
        node_b=node_3,
        rest_length=edge_rest_length,
        material=create_material("top_material"),
        element_id="top",
        name="Top edge",
    )

    left_element = Element(
        node_a=node_3,
        node_b=node_0,
        rest_length=edge_rest_length,
        material=create_material("left_material"),
        element_id="left",
        name="Left edge",
    )

    # ------------------------------------------------------------------
    # Create diagonal bracing
    # ------------------------------------------------------------------
    #
    # The two diagonals form an X:
    #
    #     node_0 -> node_2
    #     node_1 -> node_3
    #

    diagonal_0 = Element(
        node_a=node_0,
        node_b=node_2,
        rest_length=diagonal_rest_length,
        material=create_material(
            name="diagonal_0_material",
            stiffness=120.0,
            damping=4.0,
        ),
        element_id="diagonal_0",
        name="Diagonal 0-2",
    )

    diagonal_1 = Element(
        node_a=node_1,
        node_b=node_3,
        rest_length=diagonal_rest_length,
        material=create_material(
            name="diagonal_1_material",
            stiffness=120.0,
            damping=4.0,
        ),
        element_id="diagonal_1",
        name="Diagonal 1-3",
    )

    elements = [
        bottom_element,
        right_element,
        top_element,
        left_element,
        diagonal_0,
        diagonal_1,
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
    # Initial measurements
    # ------------------------------------------------------------------

    print_state(
        title="Initial configuration",
        network=network,
        elements=elements,
    )

    initial_top_center = compute_top_center(
        node_2,
        node_3,
    )

    initial_shear = compute_shear_measure(
        node_0,
        node_1,
        node_2,
        node_3,
    )

    initial_diagonal_0_force = (
        diagonal_0.last_force.total
    )

    initial_diagonal_1_force = (
        diagonal_1.last_force.total
    )

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    #
    # Biological updating is disabled because this example examines
    # pure mechanical behavior rather than fatigue, failure, damage,
    # recovery, or remodeling.
    #

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

    # Preserve the statistics generated by the final physical step.
    stats = network.last_step_stats

    # ------------------------------------------------------------------
    # Final measurements
    # ------------------------------------------------------------------

    print_state(
        title="Configuration after simulation",
        network=network,
        elements=elements,
    )

    final_top_center = compute_top_center(
        node_2,
        node_3,
    )

    final_shear = compute_shear_measure(
        node_0,
        node_1,
        node_2,
        node_3,
    )

    shear_reduction, shear_reduction_percent = (
        compute_shear_reduction(
            initial_shear,
            final_shear,
        )
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print()
    print("Final diagnostics")
    print("-" * 82)
    print(f"Simulation time        : {network.time:.6f} s")
    print(f"Step count             : {network.step_index}")
    print(f"Kinetic energy         : {stats.kinetic_energy:.9f}")
    print(f"Elastic energy         : {stats.elastic_energy:.9f}")
    print(f"Total energy           : {stats.total_energy:.9f}")
    print(f"Maximum force          : {stats.maximum_force:.9f} N")
    print(f"Maximum velocity       : {stats.maximum_velocity:.9f}")

    print(
        f"Initial top center     : "
        f"({initial_top_center[0]:.6f}, "
        f"{initial_top_center[1]:.6f})"
    )

    print(
        f"Final top center       : "
        f"({final_top_center[0]:.6f}, "
        f"{final_top_center[1]:.6f})"
    )

    print(f"Initial shear measure  : {initial_shear:.6f}")
    print(f"Final shear measure    : {final_shear:.6f}")
    print(f"Shear reduction        : {shear_reduction:.6f}")
    print(
        f"Shear reduction percent: "
        f"{shear_reduction_percent:.2f}%"
    )

    # ------------------------------------------------------------------
    # Initial diagonal loading
    # ------------------------------------------------------------------

    print()
    print("Initial diagonal loading")
    print("-" * 82)

    print(
        f"Diagonal 0-2: "
        f"{initial_diagonal_0_force:.6f} N "
        f"({force_mode(initial_diagonal_0_force)})"
    )

    print(
        f"Diagonal 1-3: "
        f"{initial_diagonal_1_force:.6f} N "
        f"({force_mode(initial_diagonal_1_force)})"
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print()
    print("Interpretation")
    print("-" * 82)
    print(
        "The diagonal elements create two additional mechanical"
    )
    print(
        "pathways between the upper nodes and the fixed base."
    )
    print(
        "The initial horizontal displacement stretches one diagonal"
    )
    print(
        "and compresses the other."
    )
    print(
        "Their opposing horizontal force components resist shear."
    )
    print(
        "Unlike the unbraced square, the cross-braced structure"
    )
    print(
        "cannot deform into a parallelogram without changing the"
    )
    print(
        "lengths of its diagonal elements."
    )

    print()
    print("Key observation")
    print("-" * 82)
    print(
        "Diagonal bracing converts the flexible quadrilateral into"
    )
    print(
        "a system of mechanically coupled triangles."
    )
    print(
        "The upper edge returns close to alignment with the base,"
    )
    print(
        f"reducing the shear magnitude by "
        f"{shear_reduction_percent:.2f}%."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()