r"""
ROIF Engine Example 06

Pretensioned Mechanical Network

This example compares two mechanically identical V-shaped networks:

1. a passive network without pretension;
2. a prestressed network with equal pretension in both elements.

Structure:

    Fixed node A                 Fixed node B
          +---------------------------+
           \                         /
            \                       /
             \                     /
              \                   /
               + Free center node

The free center node begins slightly above the reference line.

Both networks use the same:

- geometry;
- masses;
- elastic stiffness;
- damping;
- timestep;
- simulation duration.

The only difference is the initial material pretension.

The example demonstrates:

- internal force without geometric extension;
- balanced prestress in a symmetric configuration;
- increased restoring response after perturbation;
- separation of elastic and pretension force components.
"""

from pathlib import Path
import sys


# ----------------------------------------------------------------------
# Make the project root visible to Python
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node


# ----------------------------------------------------------------------
# Experiment settings
# ----------------------------------------------------------------------

STIFFNESS = 100.0
ELEMENT_DAMPING = 2.0
GLOBAL_DAMPING = 0.6

PRETENSION_FORCE = 5.0

INITIAL_VERTICAL_DISPLACEMENT = 0.20

TIME_STEP = 0.001
NUMBER_OF_STEPS = 2000


def create_material(
    name: str,
    pretension: float,
) -> Material:
    """
    Create an elastic material with a specified axial pretension.

    Pretension belongs to the mutable material state rather than to
    the geometric Element object.
    """

    parameters = MaterialParameters(
        stiffness=STIFFNESS,
        damping=ELEMENT_DAMPING,
    )

    material = Material(
        name=name,
        parameters=parameters,
    )

    material.state.pretension = float(
        pretension
    )

    material.validate()

    return material


def create_network(
    name: str,
    pretension: float,
) -> tuple[
    Network,
    Node,
    list[Element],
]:
    """
    Create one symmetric two-element network.

    Reference configuration:

        anchor_a = [-1.0, 0.0]
        center   = [ 0.0, 0.0]
        anchor_b = [ 1.0, 0.0]

    Initial perturbed configuration:

        center   = [0.0, 0.2]
    """

    anchor_a = Node(
        position=[-1.0, 0.0],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_anchor_a",
    )

    center = Node(
        position=[
            0.0,
            INITIAL_VERTICAL_DISPLACEMENT,
        ],
        mass=1.0,
        fixed=False,
        node_id=f"{name}_center",
    )

    anchor_b = Node(
        position=[1.0, 0.0],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_anchor_b",
    )

    left_element = Element(
        node_a=anchor_a,
        node_b=center,
        rest_length=1.0,
        material=create_material(
            name=f"{name}_left_material",
            pretension=pretension,
        ),
        element_id=f"{name}_left_element",
        name="Left element",
    )

    right_element = Element(
        node_a=center,
        node_b=anchor_b,
        rest_length=1.0,
        material=create_material(
            name=f"{name}_right_material",
            pretension=pretension,
        ),
        element_id=f"{name}_right_element",
        name="Right element",
    )

    elements = [
        left_element,
        right_element,
    ]

    network = Network(
        gravity=None,
        global_damping=GLOBAL_DAMPING,
        record_history=True,
    )

    network.add_node(anchor_a)
    network.add_node(center)
    network.add_node(anchor_b)

    for element in elements:
        network.add_element(element)

    return (
        network,
        center,
        elements,
    )


def evaluate_forces(
    network: Network,
) -> None:
    """
    Evaluate current element forces without advancing simulation time.
    """

    network.clear_forces()

    network.assemble_element_forces(
        include_active=False,
    )


def print_element_state(
    elements: list[Element],
) -> None:
    """Print total and decomposed axial forces."""

    for index, element in enumerate(elements):
        force = element.last_force

        print(
            f"Element {index}: "
            f"name={element.name}, "
            f"length={element.current_length():.6f}, "
            f"extension={element.extension():.6f}"
        )

        print(
            f"    elastic   = "
            f"{force.elastic:.6f} N"
        )

        print(
            f"    damping   = "
            f"{force.damping:.6f} N"
        )

        print(
            f"    pretension= "
            f"{force.pretension:.6f} N"
        )

        print(
            f"    total     = "
            f"{force.total:.6f} N"
        )


def print_network_state(
    title: str,
    network: Network,
    center: Node,
    elements: list[Element],
) -> None:
    """Print the current state of one network."""

    evaluate_forces(network)

    print()
    print(title)
    print("-" * 82)

    print(
        f"Center position : "
        f"x={center.position[0]:.6f}, "
        f"y={center.position[1]:.6f}"
    )

    print(
        f"Center velocity : "
        f"vx={center.velocity[0]:.6f}, "
        f"vy={center.velocity[1]:.6f}"
    )

    print(
        f"Center force    : "
        f"Fx={center.force[0]:.6f}, "
        f"Fy={center.force[1]:.6f}"
    )

    print()

    print_element_state(elements)


def simulate_network(
    network: Network,
) -> None:
    """
    Advance one network using pure mechanical behavior.

    Biological updating is disabled so pretension remains constant
    throughout this experiment.
    """

    for _ in range(NUMBER_OF_STEPS):
        network.step(
            TIME_STEP,
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=True,
        )


def percentage_difference(
    baseline_value: float,
    comparison_value: float,
) -> float:
    """
    Return percentage reduction in absolute magnitude.

    Positive output means the comparison value is closer to zero.
    """

    baseline_magnitude = abs(
        baseline_value
    )

    comparison_magnitude = abs(
        comparison_value
    )

    if baseline_magnitude == 0.0:
        return 0.0

    return float(
        (
            baseline_magnitude
            - comparison_magnitude
        )
        / baseline_magnitude
        * 100.0
    )


def main() -> None:
    """Run the pretension comparison experiment."""

    print("=" * 82)
    print("ROIF Engine - Example 06")
    print("Pretensioned Mechanical Network")
    print("=" * 82)

    # ------------------------------------------------------------------
    # Create baseline network
    # ------------------------------------------------------------------

    (
        passive_network,
        passive_center,
        passive_elements,
    ) = create_network(
        name="passive",
        pretension=0.0,
    )

    # ------------------------------------------------------------------
    # Create prestressed network
    # ------------------------------------------------------------------

    (
        prestressed_network,
        prestressed_center,
        prestressed_elements,
    ) = create_network(
        name="prestressed",
        pretension=PRETENSION_FORCE,
    )

    # ------------------------------------------------------------------
    # Initial states
    # ------------------------------------------------------------------

    print_network_state(
        title="Initial passive network",
        network=passive_network,
        center=passive_center,
        elements=passive_elements,
    )

    initial_passive_vertical_force = float(
        passive_center.force[1]
    )

    print_network_state(
        title="Initial prestressed network",
        network=prestressed_network,
        center=prestressed_center,
        elements=prestressed_elements,
    )

    initial_prestressed_vertical_force = float(
        prestressed_center.force[1]
    )

    # ------------------------------------------------------------------
    # Run both simulations
    # ------------------------------------------------------------------

    simulate_network(
        passive_network
    )

    simulate_network(
        prestressed_network
    )

    passive_stats = (
        passive_network.last_step_stats
    )

    prestressed_stats = (
        prestressed_network.last_step_stats
    )

    # ------------------------------------------------------------------
    # Final states
    # ------------------------------------------------------------------

    print_network_state(
        title="Final passive network",
        network=passive_network,
        center=passive_center,
        elements=passive_elements,
    )

    print_network_state(
        title="Final prestressed network",
        network=prestressed_network,
        center=prestressed_center,
        elements=prestressed_elements,
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    passive_final_y = float(
        passive_center.position[1]
    )

    prestressed_final_y = float(
        prestressed_center.position[1]
    )

    displacement_reduction_percent = (
        percentage_difference(
            baseline_value=passive_final_y,
            comparison_value=prestressed_final_y,
        )
    )

    print()
    print("Comparison")
    print("-" * 82)

    print(
        f"Initial displacement          : "
        f"{INITIAL_VERTICAL_DISPLACEMENT:.6f}"
    )

    print(
        f"Configured pretension         : "
        f"{PRETENSION_FORCE:.6f} N per element"
    )

    print(
        f"Passive initial vertical force: "
        f"{initial_passive_vertical_force:.6f} N"
    )

    print(
        f"Prestressed initial vert force: "
        f"{initial_prestressed_vertical_force:.6f} N"
    )

    print(
        f"Passive final center y        : "
        f"{passive_final_y:.6f}"
    )

    print(
        f"Prestressed final center y    : "
        f"{prestressed_final_y:.6f}"
    )

    print(
        f"Final displacement reduction  : "
        f"{displacement_reduction_percent:.2f}%"
    )

    print()
    print("Passive final diagnostics")
    print("-" * 82)

    print(
        f"Kinetic energy : "
        f"{passive_stats.kinetic_energy:.9f}"
    )

    print(
        f"Elastic energy : "
        f"{passive_stats.elastic_energy:.9f}"
    )

    print(
        f"Maximum force  : "
        f"{passive_stats.maximum_force:.9f} N"
    )

    print()
    print("Prestressed final diagnostics")
    print("-" * 82)

    print(
        f"Kinetic energy : "
        f"{prestressed_stats.kinetic_energy:.9f}"
    )

    print(
        f"Elastic energy : "
        f"{prestressed_stats.elastic_energy:.9f}"
    )

    print(
        f"Maximum force  : "
        f"{prestressed_stats.maximum_force:.9f} N"
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print()
    print("Interpretation")
    print("-" * 82)

    print(
        "Both networks have identical geometry, stiffness, damping,"
    )

    print(
        "mass, timestep, and initial displacement."
    )

    print(
        "The prestressed network contains an additional tensile"
    )

    print(
        "force component even before further elastic deformation."
    )

    print(
        "In the symmetric horizontal reference configuration, the"
    )

    print(
        "left and right pretension forces balance each other."
    )

    print(
        "After the center node is displaced upward, both elements"
    )

    print(
        "develop downward force components."
    )

    print(
        "Pretension increases this restoring action without requiring"
    )

    print(
        "a corresponding increase in elastic extension."
    )

    print()
    print("Key observation")
    print("-" * 82)

    print(
        "A prestressed structure can contain nonzero internal forces"
    )

    print(
        "even when its elements remain at their reference lengths."
    )

    print(
        "Perturbation changes the directions of those internal forces,"
    )

    print(
        "producing an immediate system-level restoring response."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()