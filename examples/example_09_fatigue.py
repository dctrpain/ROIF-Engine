r"""
ROIF Engine Example 09

Fatigue Under Sustained Mechanical Loading

This example compares two mechanically identical elements:

1. a control element with fatigue disabled;
2. a fatigue-sensitive element under sustained extension.

Structure:

    Control element:

    Fixed node A +----------------+ Fixed node B


    Fatigue-sensitive element:

    Fixed node C +----------------+ Fixed node D

All nodes are fixed. Therefore, both elements remain at the same
constant geometric extension during the entire experiment.

The only difference is the fatigue accumulation rate.

The example demonstrates:

- fatigue accumulation under sustained loading;
- reduction of material integrity;
- reduction of effective stiffness;
- progressive loss of force capacity;
- unchanged geometry with changing mechanical behavior;
- separation of fatigue from damage and failure.
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

CONTROL_FATIGUE_RATE = 0.0
ACTIVE_FATIGUE_RATE = 0.4

# Recovery is deliberately disabled in this example.
RECOVERY_RATE = 0.0

# Remodeling and damage are also disabled so fatigue can be studied
# independently.
REMODELING_RATE = 0.0
DAMAGE_RATE = 0.0

# Initial force is approximately 10 N.
# A reference force of 20 N produces an initial normalized force
# stimulus of approximately 0.5.
REFERENCE_FORCE = 20.0
REFERENCE_STRAIN = 1.0

# High thresholds prevent overload damage and automatic failure.
OVERLOAD_THRESHOLD = 10.0
FAILURE_THRESHOLD = 100.0

TIME_STEP = 0.01
NUMBER_OF_STEPS = 1000

REPORT_INTERVAL = 100


def create_material(
    name: str,
    fatigue_rate: float,
) -> Material:
    """
    Create a material with a selected fatigue rate.

    Damage, remodeling, recovery, and production are disabled so that
    the experiment isolates fatigue accumulation.
    """

    parameters = MaterialParameters(
        stiffness=BASE_STIFFNESS,
        damping=ELEMENT_DAMPING,
        recovery_rate=RECOVERY_RATE,
        fatigue_rate=fatigue_rate,
        remodeling_rate=REMODELING_RATE,
        production_rate=0.0,
        damage_rate=DAMAGE_RATE,
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
    fatigue_rate: float,
) -> tuple[
    Node,
    Node,
    Element,
]:
    """
    Create one stretched element between two fixed nodes.

    Node separation is 1.1 m, while reference length is 1.0 m.
    The geometric extension therefore remains 0.1 m.
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
            fatigue_rate=fatigue_rate,
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


def current_stimulus(
    element: Element,
) -> float:
    """
    Calculate current normalized mechanical stimulus.

    This calculates the value directly from the current force and
    strain, so it is meaningful even before the first biological update.
    """

    material = element.material

    reference_length = (
        element.material_reference_length()
    )

    extension = element.extension()

    if reference_length == 0.0:
        strain = element.current_length()
    else:
        strain = (
            extension
            / reference_length
        )

    return float(
        material.normalized_stimulus(
            force=element.last_force.total,
            strain=strain,
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
        f"Fatigue               : "
        f"{state.fatigue:.6f}"
    )

    print(
        f"Damage                : "
        f"{state.damage:.6f}"
    )

    print(
        f"Integrity              : "
        f"{material.integrity():.6f}"
    )

    print(
        f"Effective stiffness   : "
        f"{material.effective_stiffness():.6f} N/m"
    )

    print(
        f"Current stimulus      : "
        f"{current_stimulus(element):.6f}"
    )

    print(
        f"Stored stimulus       : "
        f"{state.normalized_stimulus:.6f}"
    )

    print(
        f"Total axial force     : "
        f"{element.last_force.total:.6f} N"
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
    """Run the sustained-loading fatigue experiment."""

    print("=" * 84)
    print("ROIF Engine - Example 09")
    print("Fatigue Under Sustained Mechanical Loading")
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
        fatigue_rate=CONTROL_FATIGUE_RATE,
    )

    # ------------------------------------------------------------------
    # Create fatigue-sensitive element
    # ------------------------------------------------------------------

    (
        fatigue_node_a,
        fatigue_node_b,
        fatigue_element,
    ) = create_fixed_element(
        name="Fatigue-sensitive element",
        start_x=2.0,
        fatigue_rate=ACTIVE_FATIGUE_RATE,
    )

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
    network.add_node(fatigue_node_a)
    network.add_node(fatigue_node_b)

    network.add_element(control_element)
    network.add_element(fatigue_element)

    # ------------------------------------------------------------------
    # Initial force evaluation
    # ------------------------------------------------------------------

    evaluate_forces(network)

    initial_control_fatigue = float(
        control_element.material.state.fatigue
    )

    initial_fatigue_value = float(
        fatigue_element.material.state.fatigue
    )

    initial_control_integrity = float(
        control_element.material.integrity()
    )

    initial_fatigue_integrity = float(
        fatigue_element.material.integrity()
    )

    initial_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    initial_fatigue_stiffness = float(
        fatigue_element.material.effective_stiffness()
    )

    initial_control_force = float(
        control_element.last_force.total
    )

    initial_fatigue_force = float(
        fatigue_element.last_force.total
    )

    initial_fatigue_stimulus = (
        current_stimulus(
            fatigue_element
        )
    )

    print_element_state(
        title="Initial control element",
        element=control_element,
    )

    print_element_state(
        title="Initial fatigue-sensitive element",
        element=fatigue_element,
    )

    # ------------------------------------------------------------------
    # Run sustained loading
    # ------------------------------------------------------------------
    #
    # update_materials=True is required because fatigue is part of the
    # biological Material lifecycle.
    #
    # All nodes remain fixed, so geometry does not change.
    #

    print()
    print("Fatigue history")
    print("-" * 84)

    print(
        "time       "
        "control_f   "
        "fatigue_f   "
        "integrity   "
        "effective_k   "
        "force       "
        "stimulus"
    )

    print(
        f"{network.time:8.3f}   "
        f"{control_element.material.state.fatigue:9.6f}   "
        f"{fatigue_element.material.state.fatigue:9.6f}   "
        f"{fatigue_element.material.integrity():9.6f}   "
        f"{fatigue_element.material.effective_stiffness():11.4f}   "
        f"{fatigue_element.last_force.total:9.6f}   "
        f"{current_stimulus(fatigue_element):8.6f}"
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
                f"{control_element.material.state.fatigue:9.6f}   "
                f"{fatigue_element.material.state.fatigue:9.6f}   "
                f"{fatigue_element.material.integrity():9.6f}   "
                f"{fatigue_element.material.effective_stiffness():11.4f}   "
                f"{fatigue_element.last_force.total:9.6f}   "
                f"{current_stimulus(fatigue_element):8.6f}"
            )

    # ------------------------------------------------------------------
    # Final state
    # ------------------------------------------------------------------

    evaluate_forces(network)

    final_control_fatigue = float(
        control_element.material.state.fatigue
    )

    final_fatigue_value = float(
        fatigue_element.material.state.fatigue
    )

    final_control_integrity = float(
        control_element.material.integrity()
    )

    final_fatigue_integrity = float(
        fatigue_element.material.integrity()
    )

    final_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    final_fatigue_stiffness = float(
        fatigue_element.material.effective_stiffness()
    )

    final_control_force = float(
        control_element.last_force.total
    )

    final_fatigue_force = float(
        fatigue_element.last_force.total
    )

    final_fatigue_stimulus = (
        current_stimulus(
            fatigue_element
        )
    )

    print_element_state(
        title="Final control element",
        element=control_element,
    )

    print_element_state(
        title="Final fatigue-sensitive element",
        element=fatigue_element,
    )

    # ------------------------------------------------------------------
    # Calculate changes
    # ------------------------------------------------------------------

    fatigue_change = (
        final_fatigue_value
        - initial_fatigue_value
    )

    integrity_change_percent = (
        percent_change(
            initial_fatigue_integrity,
            final_fatigue_integrity,
        )
    )

    stiffness_change_percent = (
        percent_change(
            initial_fatigue_stiffness,
            final_fatigue_stiffness,
        )
    )

    force_change_percent = (
        percent_change(
            initial_fatigue_force,
            final_fatigue_force,
        )
    )

    stimulus_change_percent = (
        percent_change(
            initial_fatigue_stimulus,
            final_fatigue_stimulus,
        )
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    print()
    print("Comparison")
    print("-" * 84)

    print(
        f"Simulation duration           : "
        f"{network.time:.6f} s"
    )

    print(
        f"Physical step count           : "
        f"{network.step_index}"
    )

    print(
        f"Control fatigue rate          : "
        f"{CONTROL_FATIGUE_RATE:.6f}"
    )

    print(
        f"Active fatigue rate           : "
        f"{ACTIVE_FATIGUE_RATE:.6f}"
    )

    print(
        f"Recovery rate                 : "
        f"{RECOVERY_RATE:.6f}"
    )

    print()

    print(
        f"Control fatigue               : "
        f"{initial_control_fatigue:.6f} "
        f"-> {final_control_fatigue:.6f}"
    )

    print(
        f"Fatigue-sensitive fatigue     : "
        f"{initial_fatigue_value:.6f} "
        f"-> {final_fatigue_value:.6f}"
    )

    print(
        f"Absolute fatigue accumulation : "
        f"{fatigue_change:.6f}"
    )

    print()

    print(
        f"Control integrity             : "
        f"{initial_control_integrity:.6f} "
        f"-> {final_control_integrity:.6f}"
    )

    print(
        f"Fatigue-sensitive integrity   : "
        f"{initial_fatigue_integrity:.6f} "
        f"-> {final_fatigue_integrity:.6f}"
    )

    print(
        f"Integrity change              : "
        f"{integrity_change_percent:.2f}%"
    )

    print()

    print(
        f"Control effective stiffness   : "
        f"{initial_control_stiffness:.6f} "
        f"-> {final_control_stiffness:.6f} N/m"
    )

    print(
        f"Fatigue effective stiffness   : "
        f"{initial_fatigue_stiffness:.6f} "
        f"-> {final_fatigue_stiffness:.6f} N/m"
    )

    print(
        f"Effective stiffness change    : "
        f"{stiffness_change_percent:.2f}%"
    )

    print()

    print(
        f"Control axial force           : "
        f"{initial_control_force:.6f} "
        f"-> {final_control_force:.6f} N"
    )

    print(
        f"Fatigue axial force           : "
        f"{initial_fatigue_force:.6f} "
        f"-> {final_fatigue_force:.6f} N"
    )

    print(
        f"Axial-force change            : "
        f"{force_change_percent:.2f}%"
    )

    print()

    print(
        f"Fatigue stimulus              : "
        f"{initial_fatigue_stimulus:.6f} "
        f"-> {final_fatigue_stimulus:.6f}"
    )

    print(
        f"Stimulus change               : "
        f"{stimulus_change_percent:.2f}%"
    )

    print()

    print(
        f"Control damage                : "
        f"{control_element.material.state.damage:.6f}"
    )

    print(
        f"Fatigue-sensitive damage      : "
        f"{fatigue_element.material.state.damage:.6f}"
    )

    print(
        f"Control failed                : "
        f"{control_element.material.state.failed}"
    )

    print(
        f"Fatigue-sensitive failed      : "
        f"{fatigue_element.material.state.failed}"
    )

    print(
        f"Control update count          : "
        f"{control_element.material.state.update_count}"
    )

    print(
        f"Fatigue update count          : "
        f"{fatigue_element.material.state.update_count}"
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
        "The control element has fatigue accumulation disabled and"
    )

    print(
        "therefore maintains its original integrity and stiffness."
    )

    print(
        "The fatigue-sensitive material accumulates fatigue during"
    )

    print(
        "sustained mechanical loading."
    )

    print(
        "Fatigue reduces structural integrity, which reduces effective"
    )

    print(
        "stiffness and the force carried at the same extension."
    )

    print(
        "As force capacity decreases, normalized mechanical stimulus"
    )

    print(
        "also decreases, slowing further fatigue accumulation."
    )

    print()
    print("Key observation")
    print("-" * 84)

    print(
        "A material can carry progressively less force even though its"
    )

    print(
        "geometry and imposed extension remain unchanged."
    )

    print(
        "The visible reduction in force is produced by an evolving"
    )

    print(
        "internal material state rather than by geometric unloading."
    )

    print()
    print("Model separation")
    print("-" * 84)

    print(
        "Damage, remodeling, recovery, and automatic failure are"
    )

    print(
        "disabled in this experiment."
    )

    print(
        "The observed mechanical decline is therefore attributable"
    )

    print(
        "specifically to fatigue."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()