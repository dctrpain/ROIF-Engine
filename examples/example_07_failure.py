r"""
ROIF Engine Example 07

Element Failure and Load Redistribution

This example creates a two-path mechanical network:

    Fixed anchor A              Fixed anchor B
           +---------------------------+
            \                         /
             \                       /
              \                     /
               \                   /
                + Free loaded node

The free node is connected to two fixed anchors.

A constant downward external force is applied to the free node.

The experiment has two phases:

1. both elements are intact;
2. the left element is marked as failed.

After failure, the left element loses its effective stiffness and
no longer carries mechanical load. The right element becomes the
only remaining load path.

The example demonstrates:

- material failure;
- loss of effective stiffness;
- disappearance of force in the failed element;
- load-path removal;
- redistribution of force through the surviving structure;
- system-level displacement after local failure.
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

ELEMENT_STIFFNESS = 120.0
ELEMENT_DAMPING = 4.0
GLOBAL_DAMPING = 1.0

EXTERNAL_LOAD = [0.0, -5.0]

TIME_STEP = 0.001

INTACT_PHASE_STEPS = 3000
FAILED_PHASE_STEPS = 3000


def create_material(
    name: str,
) -> Material:
    """
    Create an independent elastic material.

    Biological updating is disabled during this experiment. Failure is
    introduced explicitly so the mechanical consequence can be studied
    independently of fatigue and damage accumulation.
    """

    parameters = MaterialParameters(
        stiffness=ELEMENT_STIFFNESS,
        damping=ELEMENT_DAMPING,
    )

    return Material(
        name=name,
        parameters=parameters,
    )


def evaluate_forces(
    network: Network,
    loaded_node: Node,
) -> None:
    """
    Evaluate the current internal and external forces without advancing
    simulation time.
    """

    network.clear_forces()

    network.assemble_element_forces(
        include_active=False,
    )

    network.apply_external_force(
        node=loaded_node,
        force=EXTERNAL_LOAD,
    )


def print_element_state(
    index: int,
    element: Element,
) -> None:
    """Print the current state of one element."""

    force = element.last_force
    material = element.material

    print(
        f"Element {index}: "
        f"name={element.name}"
    )

    print(
        f"    failed              = "
        f"{material.state.failed}"
    )

    print(
        f"    length              = "
        f"{element.current_length():.6f}"
    )

    print(
        f"    reference length    = "
        f"{element.material_reference_length():.6f}"
    )

    print(
        f"    extension           = "
        f"{element.extension():.6f}"
    )

    print(
        f"    effective stiffness = "
        f"{material.effective_stiffness():.6f} N/m"
    )

    print(
        f"    elastic force       = "
        f"{force.elastic:.6f} N"
    )

    print(
        f"    damping force       = "
        f"{force.damping:.6f} N"
    )

    print(
        f"    total force         = "
        f"{force.total:.6f} N"
    )


def print_network_state(
    title: str,
    network: Network,
    loaded_node: Node,
    elements: list[Element],
) -> None:
    """Print network geometry, forces, and failure state."""

    evaluate_forces(
        network=network,
        loaded_node=loaded_node,
    )

    print()
    print(title)
    print("-" * 84)

    print(
        f"Loaded node position : "
        f"x={loaded_node.position[0]:.6f}, "
        f"y={loaded_node.position[1]:.6f}"
    )

    print(
        f"Loaded node velocity : "
        f"vx={loaded_node.velocity[0]:.6f}, "
        f"vy={loaded_node.velocity[1]:.6f}"
    )

    print(
        f"Net nodal force      : "
        f"Fx={loaded_node.force[0]:.6f}, "
        f"Fy={loaded_node.force[1]:.6f}"
    )

    print(
        f"Applied external load: "
        f"Fx={EXTERNAL_LOAD[0]:.6f}, "
        f"Fy={EXTERNAL_LOAD[1]:.6f}"
    )

    print()

    for index, element in enumerate(elements):
        print_element_state(
            index=index,
            element=element,
        )

        print()


def simulate_phase(
    network: Network,
    loaded_node: Node,
    number_of_steps: int,
) -> None:
    """
    Advance the network under a constant external load.

    Biological updating is disabled so no fatigue, recovery, remodeling,
    or automatic failure is introduced during this demonstration.
    """

    external_forces = {
        loaded_node: EXTERNAL_LOAD,
    }

    for _ in range(number_of_steps):
        network.step(
            dt=TIME_STEP,
            external_forces=external_forces,
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=True,
        )


def distance_between_positions(
    position_a,
    position_b,
) -> float:
    """Return Euclidean distance between two stored positions."""

    difference_x = (
        float(position_b[0])
        - float(position_a[0])
    )

    difference_y = (
        float(position_b[1])
        - float(position_a[1])
    )

    return float(
        (
            difference_x * difference_x
            + difference_y * difference_y
        )
        ** 0.5
    )


def main() -> None:
    """Run the element-failure experiment."""

    print("=" * 84)
    print("ROIF Engine - Example 07")
    print("Element Failure and Load Redistribution")
    print("=" * 84)

    # ------------------------------------------------------------------
    # Create nodes
    # ------------------------------------------------------------------
    #
    # Both anchors are fixed.
    #
    # The loaded node begins below and between them.
    #

    anchor_left = Node(
        position=[-1.0, 1.0],
        mass=1.0,
        fixed=True,
        node_id="anchor_left",
    )

    loaded_node = Node(
        position=[0.0, 0.0],
        mass=1.0,
        fixed=False,
        node_id="loaded_node",
    )

    anchor_right = Node(
        position=[1.0, 1.0],
        mass=1.0,
        fixed=True,
        node_id="anchor_right",
    )

    # ------------------------------------------------------------------
    # Create elements
    # ------------------------------------------------------------------
    #
    # Both elements begin at their reference length:
    #
    #     sqrt(1^2 + 1^2) = sqrt(2)
    #

    reference_length = 2.0 ** 0.5

    left_element = Element(
        node_a=anchor_left,
        node_b=loaded_node,
        rest_length=reference_length,
        material=create_material(
            name="left_material",
        ),
        element_id="left_element",
        name="Left load path",
    )

    right_element = Element(
        node_a=loaded_node,
        node_b=anchor_right,
        rest_length=reference_length,
        material=create_material(
            name="right_material",
        ),
        element_id="right_element",
        name="Right load path",
    )

    elements = [
        left_element,
        right_element,
    ]

    # ------------------------------------------------------------------
    # Create network
    # ------------------------------------------------------------------

    network = Network(
        gravity=None,
        global_damping=GLOBAL_DAMPING,
        record_history=True,
    )

    network.add_node(anchor_left)
    network.add_node(loaded_node)
    network.add_node(anchor_right)

    network.add_element(left_element)
    network.add_element(right_element)

    # ------------------------------------------------------------------
    # Initial unloaded geometry
    # ------------------------------------------------------------------

    print_network_state(
        title="Initial network before mechanical settling",
        network=network,
        loaded_node=loaded_node,
        elements=elements,
    )

    # ------------------------------------------------------------------
    # Phase 1: intact structure
    # ------------------------------------------------------------------

    print()
    print("Phase 1")
    print("-" * 84)
    print(
        "Both elements are intact. The external load is distributed"
    )
    print(
        "through the left and right mechanical pathways."
    )

    simulate_phase(
        network=network,
        loaded_node=loaded_node,
        number_of_steps=INTACT_PHASE_STEPS,
    )

    intact_position = (
        loaded_node.position.copy()
    )

    intact_left_force = float(
        left_element.last_force.total
    )

    intact_right_force = float(
        right_element.last_force.total
    )

    intact_stats = network.last_step_stats

    print_network_state(
        title="Settled intact network",
        network=network,
        loaded_node=loaded_node,
        elements=elements,
    )

    # ------------------------------------------------------------------
    # Introduce failure
    # ------------------------------------------------------------------

    print()
    print("Failure event")
    print("-" * 84)
    print(
        "The left material is explicitly marked as failed."
    )

    left_element.material.set_failed(
        True
    )

    # Re-evaluate immediately, before any additional movement.
    evaluate_forces(
        network=network,
        loaded_node=loaded_node,
    )

    force_immediately_after_failure = float(
        left_element.last_force.total
    )

    stiffness_immediately_after_failure = float(
        left_element.material.effective_stiffness()
    )

    print(
        f"Left force immediately after failure    : "
        f"{force_immediately_after_failure:.6f} N"
    )

    print(
        f"Left stiffness immediately after failure: "
        f"{stiffness_immediately_after_failure:.6f} N/m"
    )

    # ------------------------------------------------------------------
    # Phase 2: failed structure
    # ------------------------------------------------------------------

    print()
    print("Phase 2")
    print("-" * 84)
    print(
        "The left load path is unavailable. The structure evolves"
    )
    print(
        "under the same external load using only the right element."
    )

    simulate_phase(
        network=network,
        loaded_node=loaded_node,
        number_of_steps=FAILED_PHASE_STEPS,
    )

    failed_position = (
        loaded_node.position.copy()
    )

    failed_left_force = float(
        left_element.last_force.total
    )

    failed_right_force = float(
        right_element.last_force.total
    )

    failed_stats = network.last_step_stats

    print_network_state(
        title="Network after left-element failure",
        network=network,
        loaded_node=loaded_node,
        elements=elements,
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    position_change = distance_between_positions(
        intact_position,
        failed_position,
    )

    right_force_change = (
        abs(failed_right_force)
        - abs(intact_right_force)
    )

    print()
    print("Comparison")
    print("-" * 84)

    print(
        f"Intact equilibrium position : "
        f"({intact_position[0]:.6f}, "
        f"{intact_position[1]:.6f})"
    )

    print(
        f"Post-failure position       : "
        f"({failed_position[0]:.6f}, "
        f"{failed_position[1]:.6f})"
    )

    print(
        f"Position change after failure: "
        f"{position_change:.6f}"
    )

    print()
    print(
        f"Intact left force           : "
        f"{intact_left_force:.6f} N"
    )

    print(
        f"Failed left force           : "
        f"{failed_left_force:.6f} N"
    )

    print(
        f"Intact right force          : "
        f"{intact_right_force:.6f} N"
    )

    print(
        f"Post-failure right force    : "
        f"{failed_right_force:.6f} N"
    )

    print(
        f"Change in surviving-path load: "
        f"{right_force_change:.6f} N"
    )

    print()
    print(
        f"Reported failed elements    : "
        f"{failed_stats.failed_elements}"
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print()
    print("Intact final-step diagnostics")
    print("-" * 84)

    print(
        f"Kinetic energy : "
        f"{intact_stats.kinetic_energy:.9f}"
    )

    print(
        f"Elastic energy : "
        f"{intact_stats.elastic_energy:.9f}"
    )

    print(
        f"Maximum force  : "
        f"{intact_stats.maximum_force:.9f} N"
    )

    print()
    print("Failed final-step diagnostics")
    print("-" * 84)

    print(
        f"Kinetic energy : "
        f"{failed_stats.kinetic_energy:.9f}"
    )

    print(
        f"Elastic energy : "
        f"{failed_stats.elastic_energy:.9f}"
    )

    print(
        f"Maximum force  : "
        f"{failed_stats.maximum_force:.9f} N"
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print()
    print("Interpretation")
    print("-" * 84)

    print(
        "Before failure, the external load is transmitted through"
    )

    print(
        "two mechanically connected pathways."
    )

    print(
        "Failure sets the effective stiffness and force capacity of"
    )

    print(
        "the left material to zero."
    )

    print(
        "The failed element remains part of the network topology,"
    )

    print(
        "but it no longer contributes mechanical resistance."
    )

    print(
        "The surviving right element becomes the only active load"
    )

    print(
        "path, causing force redistribution and geometric movement."
    )

    print()
    print("Key observation")
    print("-" * 84)

    print(
        "A local material failure can produce a global change in"
    )

    print(
        "network geometry and load distribution."
    )

    print(
        "The failed element may become mechanically silent while"
    )

    print(
        "the surviving compensating element carries the observable"
    )

    print(
        "increase in demand."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()