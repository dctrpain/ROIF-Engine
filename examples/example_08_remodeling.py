r"""
ROIF Engine Example 08

Load-Dependent Material Remodeling

This example compares two mechanically identical elements:

1. a control element with remodeling disabled;
2. an adaptive element with remodeling enabled.

Structure:

    Control element:

    Fixed node A +----------------+ Fixed node B


    Adaptive element:

    Fixed node C +----------------+ Fixed node D

All four nodes are fixed, so geometry remains constant throughout
the experiment.

Both elements have:

- the same initial length;
- the same reference length;
- the same initial stiffness;
- the same damping;
- the same mechanical extension.

The adaptive material experiences a stimulus below its reference
demand. Its remodeling state therefore gradually decreases.

Because effective stiffness depends on remodeling, the adaptive
element becomes mechanically less stiff while the control element
remains unchanged.

The example demonstrates:

- biological updating through the Material lifecycle;
- load-dependent remodeling;
- separation of geometry from material adaptation;
- change in effective stiffness without geometric movement;
- progressive reduction of force capacity under low demand.
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

BASE_STIFFNESS = 100.0
ELEMENT_DAMPING = 0.0

REFERENCE_LENGTH = 1.0
CURRENT_LENGTH = 1.1

CONTROL_REMODELING_RATE = 0.0
ADAPTIVE_REMODELING_RATE = 0.5

REFERENCE_FORCE = 100.0
REFERENCE_STRAIN = 1.0

OVERLOAD_THRESHOLD = 10.0
FAILURE_THRESHOLD = 100.0

TIME_STEP = 0.01
NUMBER_OF_STEPS = 1000

REPORT_INTERVAL = 100


def create_material(
    name: str,
    remodeling_rate: float,
) -> Material:
    """
    Create a material with a selected remodeling rate.

    Large overload and failure thresholds are used because this
    experiment studies remodeling independently of damage and failure.
    """

    parameters = MaterialParameters(
        stiffness=BASE_STIFFNESS,
        damping=ELEMENT_DAMPING,
        remodeling_rate=remodeling_rate,
        recovery_rate=0.0,
        fatigue_rate=0.0,
        production_rate=0.0,
        damage_rate=0.0,
        pretension_rate=0.0,
        energy_decay_rate=0.0,
        overload_threshold=OVERLOAD_THRESHOLD,
        failure_threshold=FAILURE_THRESHOLD,
        reference_force=REFERENCE_FORCE,
        reference_strain=REFERENCE_STRAIN,
    )

    return Material(
        name=name,
        parameters=parameters,
    )


def create_fixed_element(
    name: str,
    start_x: float,
    remodeling_rate: float,
) -> tuple[
    Node,
    Node,
    Element,
]:
    """
    Create one stretched element between two fixed nodes.

    The nodes are separated by 1.1 m while the element reference
    length is 1.0 m. The extension therefore remains exactly 0.1 m.
    """

    node_a = Node(
        position=[start_x, 0.0],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_node_a",
    )

    node_b = Node(
        position=[
            start_x + CURRENT_LENGTH,
            0.0,
        ],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_node_b",
    )

    element = Element(
        node_a=node_a,
        node_b=node_b,
        rest_length=REFERENCE_LENGTH,
        material=create_material(
            name=f"{name}_material",
            remodeling_rate=remodeling_rate,
        ),
        element_id=f"{name}_element",
        name=name,
    )

    return (
        node_a,
        node_b,
        element,
    )


def evaluate_forces(
    network: Network,
) -> None:
    """
    Evaluate mechanical forces without advancing biological time.
    """

    network.clear_forces()

    network.assemble_element_forces(
        include_active=False,
    )


def remodeling_target(
    stimulus: float,
) -> float:
    """
    Reproduce the current base Material remodeling-target equation.

    target = clip(
        1 + 0.25 * (stimulus - 1),
        0,
        1,
    )
    """

    raw_target = (
        1.0
        + 0.25
        * (
            stimulus
            - 1.0
        )
    )

    return float(
        min(
            1.0,
            max(
                0.0,
                raw_target,
            ),
        )
    )


def print_element_state(
    title: str,
    element: Element,
) -> None:
    """Print mechanical and biological state of one element."""

    material = element.material
    state = material.state

    print()
    print(title)
    print("-" * 84)

    print(
        f"Element name          : "
        f"{element.name}"
    )

    print(
        f"Current length        : "
        f"{element.current_length():.6f} m"
    )

    print(
        f"Reference length      : "
        f"{element.material_reference_length():.6f} m"
    )

    print(
        f"Extension             : "
        f"{element.extension():.6f} m"
    )

    print(
        f"Base stiffness        : "
        f"{material.parameters.stiffness:.6f} N/m"
    )

    print(
        f"Remodeling state      : "
        f"{state.remodeling:.6f}"
    )

    print(
        f"Effective stiffness   : "
        f"{material.effective_stiffness():.6f} N/m"
    )

    print(
        f"Normalized stimulus   : "
        f"{state.normalized_stimulus:.6f}"
    )

    print(
        f"Remodeling target     : "
        f"{remodeling_target(state.normalized_stimulus):.6f}"
    )

    print(
        f"Total axial force     : "
        f"{element.last_force.total:.6f} N"
    )

    print(
        f"Damage                : "
        f"{state.damage:.6f}"
    )

    print(
        f"Fatigue               : "
        f"{state.fatigue:.6f}"
    )

    print(
        f"Energy                : "
        f"{state.energy:.6f}"
    )

    print(
        f"Failed                : "
        f"{state.failed}"
    )

    print(
        f"Biological updates    : "
        f"{state.update_count}"
    )


def percent_change(
    initial_value: float,
    final_value: float,
) -> float:
    """Return signed percentage change."""

    if initial_value == 0.0:
        return 0.0

    return float(
        (
            final_value
            - initial_value
        )
        / initial_value
        * 100.0
    )


def main() -> None:
    """Run the remodeling comparison experiment."""

    print("=" * 84)
    print("ROIF Engine - Example 08")
    print("Load-Dependent Material Remodeling")
    print("=" * 84)

    # ------------------------------------------------------------------
    # Create control element
    # ------------------------------------------------------------------

    (
        control_node_a,
        control_node_b,
        control_element,
    ) = create_fixed_element(
        name="Control element",
        start_x=0.0,
        remodeling_rate=(
            CONTROL_REMODELING_RATE
        ),
    )

    # ------------------------------------------------------------------
    # Create adaptive element
    # ------------------------------------------------------------------

    (
        adaptive_node_a,
        adaptive_node_b,
        adaptive_element,
    ) = create_fixed_element(
        name="Adaptive element",
        start_x=2.0,
        remodeling_rate=(
            ADAPTIVE_REMODELING_RATE
        ),
    )

    elements = [
        control_element,
        adaptive_element,
    ]

    # ------------------------------------------------------------------
    # Create network
    # ------------------------------------------------------------------

    network = Network(
        gravity=None,
        global_damping=0.0,
        record_history=True,
    )

    network.add_node(control_node_a)
    network.add_node(control_node_b)
    network.add_node(adaptive_node_a)
    network.add_node(adaptive_node_b)

    network.add_element(control_element)
    network.add_element(adaptive_element)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    evaluate_forces(network)

    initial_control_remodeling = float(
        control_element.material.state.remodeling
    )

    initial_adaptive_remodeling = float(
        adaptive_element.material.state.remodeling
    )

    initial_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    initial_adaptive_stiffness = float(
        adaptive_element.material.effective_stiffness()
    )

    initial_control_force = float(
        control_element.last_force.total
    )

    initial_adaptive_force = float(
        adaptive_element.last_force.total
    )

    print_element_state(
        title="Initial control element",
        element=control_element,
    )

    print_element_state(
        title="Initial adaptive element",
        element=adaptive_element,
    )

    # ------------------------------------------------------------------
    # Run biological remodeling
    # ------------------------------------------------------------------
    #
    # All nodes are fixed, so geometry remains constant.
    #
    # update_materials=True is essential here because remodeling belongs
    # to the biological Material lifecycle.
    #

    print()
    print("Remodeling history")
    print("-" * 84)

    print(
        "time       "
        "control_R   "
        "adaptive_R  "
        "control_k   "
        "adaptive_k  "
        "adaptive_F"
    )

    print(
        f"{network.time:8.3f}   "
        f"{control_element.material.state.remodeling:9.6f}   "
        f"{adaptive_element.material.state.remodeling:10.6f}   "
        f"{control_element.material.effective_stiffness():9.4f}   "
        f"{adaptive_element.material.effective_stiffness():10.4f}   "
        f"{adaptive_element.last_force.total:10.6f}"
    )

    for step in range(
        1,
        NUMBER_OF_STEPS + 1,
    ):
        network.step(
            dt=TIME_STEP,
            update_materials=True,
            include_active=False,
            solve_constraints=True,
            record=True,
        )

        if (
            step % REPORT_INTERVAL == 0
            or step == NUMBER_OF_STEPS
        ):
            evaluate_forces(network)

            print(
                f"{network.time:8.3f}   "
                f"{control_element.material.state.remodeling:9.6f}   "
                f"{adaptive_element.material.state.remodeling:10.6f}   "
                f"{control_element.material.effective_stiffness():9.4f}   "
                f"{adaptive_element.material.effective_stiffness():10.4f}   "
                f"{adaptive_element.last_force.total:10.6f}"
            )

    # ------------------------------------------------------------------
    # Final state
    # ------------------------------------------------------------------

    evaluate_forces(network)

    final_control_remodeling = float(
        control_element.material.state.remodeling
    )

    final_adaptive_remodeling = float(
        adaptive_element.material.state.remodeling
    )

    final_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    final_adaptive_stiffness = float(
        adaptive_element.material.effective_stiffness()
    )

    final_control_force = float(
        control_element.last_force.total
    )

    final_adaptive_force = float(
        adaptive_element.last_force.total
    )

    print_element_state(
        title="Final control element",
        element=control_element,
    )

    print_element_state(
        title="Final adaptive element",
        element=adaptive_element,
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    adaptive_remodeling_change = (
        percent_change(
            initial_adaptive_remodeling,
            final_adaptive_remodeling,
        )
    )

    adaptive_stiffness_change = (
        percent_change(
            initial_adaptive_stiffness,
            final_adaptive_stiffness,
        )
    )

    adaptive_force_change = (
        percent_change(
            initial_adaptive_force,
            final_adaptive_force,
        )
    )

    control_stiffness_change = (
        percent_change(
            initial_control_stiffness,
            final_control_stiffness,
        )
    )

    print()
    print("Comparison")
    print("-" * 84)

    print(
        f"Simulation duration          : "
        f"{network.time:.6f} s"
    )

    print(
        f"Physical step count          : "
        f"{network.step_index}"
    )

    print(
        f"Control remodeling rate      : "
        f"{CONTROL_REMODELING_RATE:.6f}"
    )

    print(
        f"Adaptive remodeling rate     : "
        f"{ADAPTIVE_REMODELING_RATE:.6f}"
    )

    print()

    print(
        f"Control remodeling           : "
        f"{initial_control_remodeling:.6f} "
        f"-> {final_control_remodeling:.6f}"
    )

    print(
        f"Adaptive remodeling          : "
        f"{initial_adaptive_remodeling:.6f} "
        f"-> {final_adaptive_remodeling:.6f}"
    )

    print(
        f"Adaptive remodeling change   : "
        f"{adaptive_remodeling_change:.2f}%"
    )

    print()

    print(
        f"Control effective stiffness  : "
        f"{initial_control_stiffness:.6f} "
        f"-> {final_control_stiffness:.6f} N/m"
    )

    print(
        f"Control stiffness change     : "
        f"{control_stiffness_change:.2f}%"
    )

    print(
        f"Adaptive effective stiffness : "
        f"{initial_adaptive_stiffness:.6f} "
        f"-> {final_adaptive_stiffness:.6f} N/m"
    )

    print(
        f"Adaptive stiffness change    : "
        f"{adaptive_stiffness_change:.2f}%"
    )

    print()

    print(
        f"Control axial force          : "
        f"{initial_control_force:.6f} "
        f"-> {final_control_force:.6f} N"
    )

    print(
        f"Adaptive axial force         : "
        f"{initial_adaptive_force:.6f} "
        f"-> {final_adaptive_force:.6f} N"
    )

    print(
        f"Adaptive force change        : "
        f"{adaptive_force_change:.2f}%"
    )

    print()

    print(
        f"Control update count         : "
        f"{control_element.material.state.update_count}"
    )

    print(
        f"Adaptive update count        : "
        f"{adaptive_element.material.state.update_count}"
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print()
    print("Interpretation")
    print("-" * 84)

    print(
        "Both elements remain at the same constant geometric extension."
    )

    print(
        "The control material has remodeling disabled, so its"
    )

    print(
        "remodeling state, effective stiffness, and force remain stable."
    )

    print(
        "The adaptive material receives a demand below its configured"
    )

    print(
        "reference stimulus."
    )

    print(
        "Its remodeling state gradually moves toward a lower target."
    )

    print(
        "Because effective stiffness is multiplied by remodeling,"
    )

    print(
        "the adaptive element progressively carries less force despite"
    )

    print(
        "unchanged geometry."
    )

    print()
    print("Key observation")
    print("-" * 84)

    print(
        "Identical geometry does not guarantee identical mechanical"
    )

    print(
        "behavior when material state evolves over time."
    )

    print(
        "A persistent change in mechanical demand can alter future"
    )

    print(
        "force transmission by changing the material itself."
    )

    print()
    print("Model limitation")
    print("-" * 84)

    print(
        "The current base remodeling law constrains remodeling to"
    )

    print(
        "the interval from 0 to 1."
    )

    print(
        "It therefore demonstrates down-regulation under low demand,"
    )

    print(
        "but does not yet represent strengthening above the original"
    )

    print(
        "baseline capacity."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()