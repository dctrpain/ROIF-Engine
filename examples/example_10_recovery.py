r"""
ROIF Engine Example 10

Fatigue Recovery After Mechanical Unloading

This example compares two initially identical materials:

1. a non-recovering control material;
2. a material whose recovery mechanism is activated after unloading.

Structure:

    Control element:

    Fixed node A +----------------+ Fixed node B


    Recovering element:

    Fixed node C +----------------+ Fixed node D

The experiment has two phases.

Phase 1 — Loading
-----------------

Both elements are stretched from 1.0 m to 1.1 m.

Both materials:

- have the same fatigue rate;
- have recovery disabled;
- accumulate approximately equal fatigue;
- progressively lose integrity, stiffness, and force capacity.

Phase 2 — Recovery
------------------

Both elements are returned to their reference length of 1.0 m.

Mechanical stimulus therefore falls to zero.

Recovery remains disabled in the control material but is activated
in the recovering material.

The example demonstrates:

- fatigue accumulation under sustained loading;
- mechanical unloading;
- fatigue recovery during rest;
- restoration of structural integrity;
- restoration of effective stiffness;
- separation of recovery from geometric loading.
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
LOADED_LENGTH = 1.1
UNLOADED_LENGTH = 1.0

FATIGUE_RATE = 0.4

INITIAL_RECOVERY_RATE = 0.0
CONTROL_RECOVERY_RATE = 0.0
ACTIVE_RECOVERY_RATE = 0.5

REMODELING_RATE = 0.0
DAMAGE_RATE = 0.0
ENERGY_DECAY_RATE = 0.0

REFERENCE_FORCE = 20.0
REFERENCE_STRAIN = 1.0

OVERLOAD_THRESHOLD = 10.0
FAILURE_THRESHOLD = 100.0

TIME_STEP = 0.01

LOADING_STEPS = 500
RECOVERY_STEPS = 1000

REPORT_INTERVAL = 100


def create_material(
    name: str,
) -> Material:
    """
    Create a fatigue-sensitive material.

    Recovery begins disabled. It will be selectively enabled after
    the loading phase.
    """

    parameters = MaterialParameters(
        stiffness=BASE_STIFFNESS,
        damping=ELEMENT_DAMPING,
        recovery_rate=INITIAL_RECOVERY_RATE,
        fatigue_rate=FATIGUE_RATE,
        remodeling_rate=REMODELING_RATE,
        production_rate=0.0,
        damage_rate=DAMAGE_RATE,
        pretension_rate=0.0,
        energy_decay_rate=ENERGY_DECAY_RATE,
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
) -> tuple[
    Node,
    Node,
    Element,
]:
    """
    Create one initially stretched element between two fixed nodes.
    """

    node_a = Node(
        position=[start_x, 0.0],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_node_a",
    )

    node_b = Node(
        position=[
            start_x + LOADED_LENGTH,
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
    Evaluate current forces without advancing biological time.
    """

    network.clear_forces()

    network.assemble_element_forces(
        include_active=False,
    )


def calculate_current_stimulus(
    element: Element,
) -> float:
    """
    Calculate normalized stimulus from current force and strain.
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


def unload_element(
    node_a: Node,
    node_b: Node,
) -> None:
    """
    Return one fixed element to its reference length.

    The element remains horizontal. Only the second node is moved.
    """

    node_b.set_position(
        [
            node_a.position[0]
            + UNLOADED_LENGTH,
            node_a.position[1],
        ]
    )

    node_b.set_velocity(
        [0.0, 0.0]
    )


def print_element_state(
    title: str,
    element: Element,
) -> None:
    """Print current mechanical and biological state."""

    material = element.material
    state = material.state

    print()
    print(title)
    print("-" * 86)

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
        f"Fatigue rate          : "
        f"{material.parameters.fatigue_rate:.6f}"
    )

    print(
        f"Recovery rate         : "
        f"{material.parameters.recovery_rate:.6f}"
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
        f"{calculate_current_stimulus(element):.6f}"
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


def recovery_fraction(
    fatigued_value: float,
    recovered_value: float,
) -> float:
    """
    Return percentage of accumulated fatigue that was recovered.
    """

    if fatigued_value <= 0.0:
        return 0.0

    recovered_amount = (
        fatigued_value
        - recovered_value
    )

    return float(
        recovered_amount
        / fatigued_value
        * 100.0
    )


def run_steps(
    network: Network,
    number_of_steps: int,
) -> None:
    """
    Advance biological time while geometry remains fixed.
    """

    for _ in range(number_of_steps):
        network.step(
            dt=TIME_STEP,
            update_materials=True,
            include_active=False,
            solve_constraints=True,
            record=True,
        )


def main() -> None:
    """Run the fatigue-loading and recovery experiment."""

    print("=" * 86)
    print("ROIF Engine - Example 10")
    print("Fatigue Recovery After Mechanical Unloading")
    print("=" * 86)

    # ------------------------------------------------------------------
    # Create control element
    # ------------------------------------------------------------------

    (
        control_node_a,
        control_node_b,
        control_element,
    ) = create_fixed_element(
        name="Non-recovering control",
        start_x=0.0,
    )

    # ------------------------------------------------------------------
    # Create recovering element
    # ------------------------------------------------------------------

    (
        recovery_node_a,
        recovery_node_b,
        recovery_element,
    ) = create_fixed_element(
        name="Recovering material",
        start_x=2.0,
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
    network.add_node(recovery_node_a)
    network.add_node(recovery_node_b)

    network.add_element(control_element)
    network.add_element(recovery_element)

    # ------------------------------------------------------------------
    # Initial loaded state
    # ------------------------------------------------------------------

    evaluate_forces(network)

    print_element_state(
        title="Initial control element",
        element=control_element,
    )

    print_element_state(
        title="Initial recovering element",
        element=recovery_element,
    )

    # ------------------------------------------------------------------
    # Phase 1 — fatigue loading
    # ------------------------------------------------------------------

    print()
    print("Phase 1 — sustained loading")
    print("-" * 86)

    print(
        "Both elements are held at 0.1 m extension."
    )

    print(
        "Recovery is disabled in both materials."
    )

    print()
    print(
        "time       "
        "control_f   "
        "recovery_f  "
        "control_k   "
        "recovery_k  "
        "force"
    )

    print(
        f"{network.time:8.3f}   "
        f"{control_element.material.state.fatigue:9.6f}   "
        f"{recovery_element.material.state.fatigue:10.6f}   "
        f"{control_element.material.effective_stiffness():9.4f}   "
        f"{recovery_element.material.effective_stiffness():10.4f}   "
        f"{recovery_element.last_force.total:9.6f}"
    )

    for step in range(
        1,
        LOADING_STEPS + 1,
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
            or step == LOADING_STEPS
        ):
            evaluate_forces(network)

            print(
                f"{network.time:8.3f}   "
                f"{control_element.material.state.fatigue:9.6f}   "
                f"{recovery_element.material.state.fatigue:10.6f}   "
                f"{control_element.material.effective_stiffness():9.4f}   "
                f"{recovery_element.material.effective_stiffness():10.4f}   "
                f"{recovery_element.last_force.total:9.6f}"
            )

    evaluate_forces(network)

    control_fatigue_after_loading = float(
        control_element.material.state.fatigue
    )

    recovery_fatigue_after_loading = float(
        recovery_element.material.state.fatigue
    )

    control_stiffness_after_loading = float(
        control_element.material.effective_stiffness()
    )

    recovery_stiffness_after_loading = float(
        recovery_element.material.effective_stiffness()
    )

    print_element_state(
        title="Control element after loading",
        element=control_element,
    )

    print_element_state(
        title="Recovering element after loading",
        element=recovery_element,
    )

    # ------------------------------------------------------------------
    # Transition — unload both elements
    # ------------------------------------------------------------------

    print()
    print("Unloading event")
    print("-" * 86)

    unload_element(
        node_a=control_node_a,
        node_b=control_node_b,
    )

    unload_element(
        node_a=recovery_node_a,
        node_b=recovery_node_b,
    )

    # Recovery remains disabled for the control material.
    control_element.material.parameters.recovery_rate = (
        CONTROL_RECOVERY_RATE
    )

    # Recovery is activated only for the second material.
    recovery_element.material.parameters.recovery_rate = (
        ACTIVE_RECOVERY_RATE
    )

    control_element.material.parameters.validate()
    recovery_element.material.parameters.validate()

    evaluate_forces(network)

    print(
        f"Control extension after unloading   : "
        f"{control_element.extension():.6f} m"
    )

    print(
        f"Recovery extension after unloading  : "
        f"{recovery_element.extension():.6f} m"
    )

    print(
        f"Control force after unloading       : "
        f"{control_element.last_force.total:.6f} N"
    )

    print(
        f"Recovery force after unloading      : "
        f"{recovery_element.last_force.total:.6f} N"
    )

    print(
        f"Control recovery rate               : "
        f"{control_element.material.parameters.recovery_rate:.6f}"
    )

    print(
        f"Active recovery rate                : "
        f"{recovery_element.material.parameters.recovery_rate:.6f}"
    )

    # ------------------------------------------------------------------
    # Phase 2 — recovery at zero mechanical load
    # ------------------------------------------------------------------

    print()
    print("Phase 2 — unloaded recovery")
    print("-" * 86)

    print(
        "Both elements remain at their reference length."
    )

    print(
        "Only the recovering material has a nonzero recovery rate."
    )

    print()
    print(
        "time       "
        "control_f   "
        "recovery_f  "
        "control_i   "
        "recovery_i  "
        "recovery_k"
    )

    print(
        f"{network.time:8.3f}   "
        f"{control_element.material.state.fatigue:9.6f}   "
        f"{recovery_element.material.state.fatigue:10.6f}   "
        f"{control_element.material.integrity():9.6f}   "
        f"{recovery_element.material.integrity():10.6f}   "
        f"{recovery_element.material.effective_stiffness():10.4f}"
    )

    for step in range(
        1,
        RECOVERY_STEPS + 1,
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
            or step == RECOVERY_STEPS
        ):
            evaluate_forces(network)

            print(
                f"{network.time:8.3f}   "
                f"{control_element.material.state.fatigue:9.6f}   "
                f"{recovery_element.material.state.fatigue:10.6f}   "
                f"{control_element.material.integrity():9.6f}   "
                f"{recovery_element.material.integrity():10.6f}   "
                f"{recovery_element.material.effective_stiffness():10.4f}"
            )

    # ------------------------------------------------------------------
    # Final states
    # ------------------------------------------------------------------

    evaluate_forces(network)

    final_control_fatigue = float(
        control_element.material.state.fatigue
    )

    final_recovery_fatigue = float(
        recovery_element.material.state.fatigue
    )

    final_control_integrity = float(
        control_element.material.integrity()
    )

    final_recovery_integrity = float(
        recovery_element.material.integrity()
    )

    final_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    final_recovery_stiffness = float(
        recovery_element.material.effective_stiffness()
    )

    print_element_state(
        title="Final non-recovering control",
        element=control_element,
    )

    print_element_state(
        title="Final recovering material",
        element=recovery_element,
    )

    # ------------------------------------------------------------------
    # Calculate recovery metrics
    # ------------------------------------------------------------------

    fatigue_recovery_percent = (
        recovery_fraction(
            fatigued_value=(
                recovery_fatigue_after_loading
            ),
            recovered_value=(
                final_recovery_fatigue
            ),
        )
    )

    stiffness_restoration_percent = (
        percent_change(
            initial_value=(
                recovery_stiffness_after_loading
            ),
            final_value=(
                final_recovery_stiffness
            ),
        )
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    print()
    print("Comparison")
    print("-" * 86)

    print(
        f"Loading duration               : "
        f"{LOADING_STEPS * TIME_STEP:.6f} s"
    )

    print(
        f"Recovery duration              : "
        f"{RECOVERY_STEPS * TIME_STEP:.6f} s"
    )

    print(
        f"Total physical steps           : "
        f"{network.step_index}"
    )

    print()

    print(
        f"Control fatigue after loading  : "
        f"{control_fatigue_after_loading:.6f}"
    )

    print(
        f"Recovery fatigue after loading : "
        f"{recovery_fatigue_after_loading:.6f}"
    )

    print(
        f"Final control fatigue          : "
        f"{final_control_fatigue:.6f}"
    )

    print(
        f"Final recovering fatigue       : "
        f"{final_recovery_fatigue:.6f}"
    )

    print(
        f"Recovered fatigue fraction     : "
        f"{fatigue_recovery_percent:.2f}%"
    )

    print()

    print(
        f"Control stiffness after loading: "
        f"{control_stiffness_after_loading:.6f} N/m"
    )

    print(
        f"Recovery stiffness after load  : "
        f"{recovery_stiffness_after_loading:.6f} N/m"
    )

    print(
        f"Final control stiffness        : "
        f"{final_control_stiffness:.6f} N/m"
    )

    print(
        f"Final recovering stiffness     : "
        f"{final_recovery_stiffness:.6f} N/m"
    )

    print(
        f"Stiffness increase during rest : "
        f"{stiffness_restoration_percent:.2f}%"
    )

    print()

    print(
        f"Final control integrity        : "
        f"{final_control_integrity:.6f}"
    )

    print(
        f"Final recovering integrity     : "
        f"{final_recovery_integrity:.6f}"
    )

    print(
        f"Final control force            : "
        f"{control_element.last_force.total:.6f} N"
    )

    print(
        f"Final recovering force         : "
        f"{recovery_element.last_force.total:.6f} N"
    )

    print(
        f"Control damage                 : "
        f"{control_element.material.state.damage:.6f}"
    )

    print(
        f"Recovering damage              : "
        f"{recovery_element.material.state.damage:.6f}"
    )

    print(
        f"Control failed                 : "
        f"{control_element.material.state.failed}"
    )

    print(
        f"Recovering failed              : "
        f"{recovery_element.material.state.failed}"
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print()
    print("Interpretation")
    print("-" * 86)

    print(
        "During the loading phase, both materials experience the same"
    )

    print(
        "extension, force stimulus, and fatigue accumulation law."
    )

    print(
        "They therefore reach approximately the same fatigued state."
    )

    print(
        "Unloading removes elastic force and reduces mechanical"
    )

    print(
        "stimulus to zero."
    )

    print(
        "The control material retains its accumulated fatigue because"
    )

    print(
        "its recovery rate remains zero."
    )

    print(
        "The recovering material progressively reduces fatigue during"
    )

    print(
        "rest, restoring integrity and effective stiffness."
    )

    print()
    print("Key observation")
    print("-" * 86)

    print(
        "Removal of mechanical load stops additional fatigue, but"
    )

    print(
        "unloading alone does not restore the material."
    )

    print(
        "Recovery requires a separate biological recovery mechanism"
    )

    print(
        "and sufficient available energy."
    )

    print()
    print("Model separation")
    print("-" * 86)

    print(
        "Damage, remodeling, production, and automatic failure are"
    )

    print(
        "disabled in this experiment."
    )

    print(
        "The observed restoration of stiffness is therefore caused"
    )

    print(
        "specifically by fatigue recovery."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()