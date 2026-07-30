r"""
ROIF Engine Example 11

Damage Accumulation Under Mechanical Overload

This example compares two materials:

1. a control material operating below the overload threshold;
2. an overloaded material operating above the overload threshold.

Both elements have:

- the same base stiffness;
- the same reference length;
- the same damage law;
- fatigue disabled;
- recovery disabled;
- remodeling disabled;
- automatic failure prevented by a high failure threshold.

The only experimental difference is geometric extension.

Control element:

    reference length = 1.0 m
    current length   = 1.05 m

Overloaded element:

    reference length = 1.0 m
    current length   = 1.20 m

The experiment has two phases.

Phase 1 — Sustained loading
---------------------------

The control element remains below the overload threshold and should
not accumulate damage.

The overloaded element exceeds the threshold and progressively
accumulates structural damage.

As damage increases:

- integrity decreases;
- effective stiffness decreases;
- carried force decreases;
- normalized stimulus decreases;
- further damage accumulation slows.

Phase 2 — Unloading
-------------------

Both elements are returned to their reference length.

Mechanical force falls to zero, but previously accumulated damage
remains because recovery is disabled.

This separates:

- current mechanical loading;
- persistent internal structural damage.
"""

from pathlib import Path
import sys


# ----------------------------------------------------------------------
# Make project root visible to Python
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

CONTROL_LENGTH = 1.05
OVERLOADED_LENGTH = 1.20
UNLOADED_LENGTH = 1.0

DAMAGE_RATE = 0.8

# Fatigue is disabled so that the observed decline is caused only
# by structural damage.
FATIGUE_RATE = 0.0

RECOVERY_RATE = 0.0
REMODELING_RATE = 0.0
PRODUCTION_RATE = 0.0
ENERGY_DECAY_RATE = 0.0

REFERENCE_FORCE = 20.0
REFERENCE_STRAIN = 1.0

# With the chosen geometry:
#
# control initial stimulus    ≈ 0.25
# overloaded initial stimulus ≈ 1.00
#
# Therefore only the overloaded element exceeds this threshold.
OVERLOAD_THRESHOLD = 0.40

# Keep automatic failure outside this experiment.
FAILURE_THRESHOLD = 100.0

TIME_STEP = 0.01
LOADING_STEPS = 1000
UNLOADED_OBSERVATION_STEPS = 200

REPORT_INTERVAL = 100


def create_material(
    name: str,
) -> Material:
    """Create a damage-sensitive material."""

    parameters = MaterialParameters(
        stiffness=BASE_STIFFNESS,
        damping=ELEMENT_DAMPING,
        recovery_rate=RECOVERY_RATE,
        fatigue_rate=FATIGUE_RATE,
        remodeling_rate=REMODELING_RATE,
        production_rate=PRODUCTION_RATE,
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
    *,
    name: str,
    start_x: float,
    current_length: float,
) -> tuple[
    Node,
    Node,
    Element,
]:
    """
    Create one horizontal element between two fixed nodes.
    """

    node_a = Node(
        position=[start_x, 0.0],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_node_a",
    )

    node_b = Node(
        position=[
            start_x + current_length,
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
    Recalculate current mechanical forces without advancing time.
    """

    network.clear_forces()

    network.assemble_element_forces(
        include_active=False,
    )


def calculate_current_stimulus(
    element: Element,
) -> float:
    """
    Calculate normalized mechanical stimulus from current force
    and strain.
    """

    reference_length = (
        element.material_reference_length()
    )

    if reference_length == 0.0:
        strain = element.current_length()
    else:
        strain = (
            element.extension()
            / reference_length
        )

    return float(
        element.material.normalized_stimulus(
            force=element.last_force.total,
            strain=strain,
        )
    )


def calculate_overload(
    element: Element,
) -> float:
    """
    Return the part of stimulus above the overload threshold.
    """

    stimulus = calculate_current_stimulus(
        element
    )

    threshold = (
        element.material.parameters.overload_threshold
    )

    return max(
        0.0,
        stimulus - threshold,
    )


def unload_element(
    node_a: Node,
    node_b: Node,
) -> None:
    """
    Return an element to its reference length.
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
    print("-" * 92)

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
        f"Damage rate           : "
        f"{material.parameters.damage_rate:.6f}"
    )

    print(
        f"Overload threshold    : "
        f"{material.parameters.overload_threshold:.6f}"
    )

    print(
        f"Current stimulus      : "
        f"{calculate_current_stimulus(element):.6f}"
    )

    print(
        f"Current overload      : "
        f"{calculate_overload(element):.6f}"
    )

    print(
        f"Stored stimulus       : "
        f"{state.normalized_stimulus:.6f}"
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


def damage_ratio(
    control_damage: float,
    overloaded_damage: float,
) -> float:
    """
    Return the overloaded-to-control damage ratio.

    If control damage is zero, infinity is returned.
    """

    if control_damage == 0.0:
        if overloaded_damage > 0.0:
            return float("inf")

        return 1.0

    return float(
        overloaded_damage
        / control_damage
    )


def main() -> None:
    """Run the structural-damage experiment."""

    print("=" * 92)
    print("ROIF Engine - Example 11")
    print("Damage Accumulation Under Mechanical Overload")
    print("=" * 92)

    # ------------------------------------------------------------------
    # Create the control element
    # ------------------------------------------------------------------

    (
        control_node_a,
        control_node_b,
        control_element,
    ) = create_fixed_element(
        name="Sub-threshold control",
        start_x=0.0,
        current_length=CONTROL_LENGTH,
    )

    # ------------------------------------------------------------------
    # Create the overloaded element
    # ------------------------------------------------------------------

    (
        overloaded_node_a,
        overloaded_node_b,
        overloaded_element,
    ) = create_fixed_element(
        name="Overloaded element",
        start_x=2.0,
        current_length=OVERLOADED_LENGTH,
    )

    # ------------------------------------------------------------------
    # Create the network
    # ------------------------------------------------------------------

    network = Network(
        gravity=None,
        global_damping=0.0,
        record_history=True,
    )

    network.add_node(control_node_a)
    network.add_node(control_node_b)

    network.add_node(overloaded_node_a)
    network.add_node(overloaded_node_b)

    network.add_element(control_element)
    network.add_element(overloaded_element)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    evaluate_forces(network)

    initial_control_damage = float(
        control_element.material.state.damage
    )

    initial_overloaded_damage = float(
        overloaded_element.material.state.damage
    )

    initial_control_integrity = float(
        control_element.material.integrity()
    )

    initial_overloaded_integrity = float(
        overloaded_element.material.integrity()
    )

    initial_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    initial_overloaded_stiffness = float(
        overloaded_element.material.effective_stiffness()
    )

    initial_control_force = float(
        control_element.last_force.total
    )

    initial_overloaded_force = float(
        overloaded_element.last_force.total
    )

    print_element_state(
        title="Initial control element",
        element=control_element,
    )

    print_element_state(
        title="Initial overloaded element",
        element=overloaded_element,
    )

    # ------------------------------------------------------------------
    # Phase 1 — sustained mechanical loading
    # ------------------------------------------------------------------

    print()
    print("Phase 1 — sustained mechanical loading")
    print("-" * 92)

    print(
        "The control element remains below the overload threshold."
    )

    print(
        "The overloaded element starts above the overload threshold."
    )

    print(
        "Fatigue, recovery, remodeling, and failure are disabled."
    )

    print()
    print(
        "time       "
        "control_d   "
        "overload_d  "
        "integrity   "
        "effective_k   "
        "force        "
        "stimulus     "
        "overload"
    )

    print(
        f"{network.time:8.3f}   "
        f"{control_element.material.state.damage:9.6f}   "
        f"{overloaded_element.material.state.damage:10.6f}   "
        f"{overloaded_element.material.integrity():9.6f}   "
        f"{overloaded_element.material.effective_stiffness():11.4f}   "
        f"{overloaded_element.last_force.total:10.6f}   "
        f"{calculate_current_stimulus(overloaded_element):10.6f}   "
        f"{calculate_overload(overloaded_element):8.6f}"
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
                f"{control_element.material.state.damage:9.6f}   "
                f"{overloaded_element.material.state.damage:10.6f}   "
                f"{overloaded_element.material.integrity():9.6f}   "
                f"{overloaded_element.material.effective_stiffness():11.4f}   "
                f"{overloaded_element.last_force.total:10.6f}   "
                f"{calculate_current_stimulus(overloaded_element):10.6f}   "
                f"{calculate_overload(overloaded_element):8.6f}"
            )

    evaluate_forces(network)

    loaded_control_damage = float(
        control_element.material.state.damage
    )

    loaded_overloaded_damage = float(
        overloaded_element.material.state.damage
    )

    loaded_control_integrity = float(
        control_element.material.integrity()
    )

    loaded_overloaded_integrity = float(
        overloaded_element.material.integrity()
    )

    loaded_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    loaded_overloaded_stiffness = float(
        overloaded_element.material.effective_stiffness()
    )

    loaded_control_force = float(
        control_element.last_force.total
    )

    loaded_overloaded_force = float(
        overloaded_element.last_force.total
    )

    print_element_state(
        title="Control element after sustained loading",
        element=control_element,
    )

    print_element_state(
        title="Overloaded element after sustained loading",
        element=overloaded_element,
    )

    # ------------------------------------------------------------------
    # Phase 2 — unloading
    # ------------------------------------------------------------------

    print()
    print("Phase 2 — unloading")
    print("-" * 92)

    unload_element(
        node_a=control_node_a,
        node_b=control_node_b,
    )

    unload_element(
        node_a=overloaded_node_a,
        node_b=overloaded_node_b,
    )

    evaluate_forces(network)

    damage_before_unloaded_observation = float(
        overloaded_element.material.state.damage
    )

    print(
        f"Control extension after unloading      : "
        f"{control_element.extension():.6f} m"
    )

    print(
        f"Overloaded extension after unloading   : "
        f"{overloaded_element.extension():.6f} m"
    )

    print(
        f"Control force after unloading          : "
        f"{control_element.last_force.total:.6f} N"
    )

    print(
        f"Overloaded force after unloading       : "
        f"{overloaded_element.last_force.total:.6f} N"
    )

    print(
        f"Accumulated overloaded damage          : "
        f"{damage_before_unloaded_observation:.6f}"
    )

    print()
    print(
        "The unloaded elements are now advanced biologically while"
    )

    print(
        "recovery remains disabled."
    )

    print()
    print(
        "time       "
        "control_d   "
        "overload_d  "
        "control_F   "
        "overload_F"
    )

    print(
        f"{network.time:8.3f}   "
        f"{control_element.material.state.damage:9.6f}   "
        f"{overloaded_element.material.state.damage:10.6f}   "
        f"{control_element.last_force.total:9.6f}   "
        f"{overloaded_element.last_force.total:10.6f}"
    )

    for step in range(
        1,
        UNLOADED_OBSERVATION_STEPS + 1,
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
            or step == UNLOADED_OBSERVATION_STEPS
        ):
            evaluate_forces(network)

            print(
                f"{network.time:8.3f}   "
                f"{control_element.material.state.damage:9.6f}   "
                f"{overloaded_element.material.state.damage:10.6f}   "
                f"{control_element.last_force.total:9.6f}   "
                f"{overloaded_element.last_force.total:10.6f}"
            )

    # ------------------------------------------------------------------
    # Final states
    # ------------------------------------------------------------------

    evaluate_forces(network)

    final_control_damage = float(
        control_element.material.state.damage
    )

    final_overloaded_damage = float(
        overloaded_element.material.state.damage
    )

    final_control_integrity = float(
        control_element.material.integrity()
    )

    final_overloaded_integrity = float(
        overloaded_element.material.integrity()
    )

    final_control_stiffness = float(
        control_element.material.effective_stiffness()
    )

    final_overloaded_stiffness = float(
        overloaded_element.material.effective_stiffness()
    )

    final_control_force = float(
        control_element.last_force.total
    )

    final_overloaded_force = float(
        overloaded_element.last_force.total
    )

    print_element_state(
        title="Final unloaded control element",
        element=control_element,
    )

    print_element_state(
        title="Final unloaded damaged element",
        element=overloaded_element,
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    ratio = damage_ratio(
        control_damage=final_control_damage,
        overloaded_damage=final_overloaded_damage,
    )

    overloaded_stiffness_change = percent_change(
        initial_value=initial_overloaded_stiffness,
        final_value=loaded_overloaded_stiffness,
    )

    overloaded_force_change = percent_change(
        initial_value=initial_overloaded_force,
        final_value=loaded_overloaded_force,
    )

    print()
    print("Comparison")
    print("-" * 92)

    print(
        f"Loading duration                     : "
        f"{LOADING_STEPS * TIME_STEP:.6f} s"
    )

    print(
        f"Unloaded observation duration        : "
        f"{UNLOADED_OBSERVATION_STEPS * TIME_STEP:.6f} s"
    )

    print(
        f"Total physical steps                 : "
        f"{network.step_index}"
    )

    print()

    print(
        f"Control initial stimulus             : "
        f"{initial_control_force / REFERENCE_FORCE:.6f}"
    )

    print(
        f"Overloaded initial stimulus          : "
        f"{initial_overloaded_force / REFERENCE_FORCE:.6f}"
    )

    print(
        f"Overload threshold                   : "
        f"{OVERLOAD_THRESHOLD:.6f}"
    )

    print()

    print(
        f"Control damage                       : "
        f"{initial_control_damage:.6f} -> "
        f"{loaded_control_damage:.6f}"
    )

    print(
        f"Overloaded damage                    : "
        f"{initial_overloaded_damage:.6f} -> "
        f"{loaded_overloaded_damage:.6f}"
    )

    print(
        f"Final unloaded control damage        : "
        f"{final_control_damage:.6f}"
    )

    print(
        f"Final unloaded overloaded damage     : "
        f"{final_overloaded_damage:.6f}"
    )

    if ratio == float("inf"):
        print(
            "Overloaded/control damage ratio      : "
            "infinite (control damage is zero)"
        )
    else:
        print(
            f"Overloaded/control damage ratio      : "
            f"{ratio:.6f}"
        )

    print()

    print(
        f"Control integrity after loading      : "
        f"{initial_control_integrity:.6f} -> "
        f"{loaded_control_integrity:.6f}"
    )

    print(
        f"Overloaded integrity after loading   : "
        f"{initial_overloaded_integrity:.6f} -> "
        f"{loaded_overloaded_integrity:.6f}"
    )

    print(
        f"Final control integrity              : "
        f"{final_control_integrity:.6f}"
    )

    print(
        f"Final overloaded integrity           : "
        f"{final_overloaded_integrity:.6f}"
    )

    print()

    print(
        f"Control stiffness after loading      : "
        f"{initial_control_stiffness:.6f} -> "
        f"{loaded_control_stiffness:.6f} N/m"
    )

    print(
        f"Overloaded stiffness after loading   : "
        f"{initial_overloaded_stiffness:.6f} -> "
        f"{loaded_overloaded_stiffness:.6f} N/m"
    )

    print(
        f"Overloaded stiffness change          : "
        f"{overloaded_stiffness_change:.2f}%"
    )

    print(
        f"Final unloaded control stiffness     : "
        f"{final_control_stiffness:.6f} N/m"
    )

    print(
        f"Final unloaded damaged stiffness     : "
        f"{final_overloaded_stiffness:.6f} N/m"
    )

    print()

    print(
        f"Control force during loading         : "
        f"{initial_control_force:.6f} -> "
        f"{loaded_control_force:.6f} N"
    )

    print(
        f"Overloaded force during loading      : "
        f"{initial_overloaded_force:.6f} -> "
        f"{loaded_overloaded_force:.6f} N"
    )

    print(
        f"Overloaded force change              : "
        f"{overloaded_force_change:.2f}%"
    )

    print(
        f"Final unloaded control force         : "
        f"{final_control_force:.6f} N"
    )

    print(
        f"Final unloaded damaged force         : "
        f"{final_overloaded_force:.6f} N"
    )

    print()

    print(
        f"Control fatigue                      : "
        f"{control_element.material.state.fatigue:.6f}"
    )

    print(
        f"Overloaded fatigue                   : "
        f"{overloaded_element.material.state.fatigue:.6f}"
    )

    print(
        f"Control failed                       : "
        f"{control_element.material.state.failed}"
    )

    print(
        f"Overloaded failed                    : "
        f"{overloaded_element.material.state.failed}"
    )

    print(
        f"Control update count                 : "
        f"{control_element.material.state.update_count}"
    )

    print(
        f"Overloaded update count              : "
        f"{overloaded_element.material.state.update_count}"
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print()
    print("Interpretation")
    print("-" * 92)

    print(
        "The control element remains below the overload threshold and"
    )

    print(
        "therefore does not accumulate structural damage."
    )

    print(
        "The overloaded element begins above the threshold and"
    )

    print(
        "progressively accumulates damage."
    )

    print(
        "Damage reduces structural integrity, effective stiffness,"
    )

    print(
        "and the force carried at the same imposed extension."
    )

    print(
        "As the damaged material carries less force, its normalized"
    )

    print(
        "stimulus falls toward the overload threshold."
    )

    print(
        "Damage accumulation therefore slows and may stop even while"
    )

    print(
        "the original geometric extension remains imposed."
    )

    print()
    print("Key observation")
    print("-" * 92)

    print(
        "After unloading, elastic force and mechanical stimulus fall"
    )

    print(
        "to zero, but accumulated structural damage remains."
    )

    print(
        "Zero force does not mean that the material has returned to"
    )

    print(
        "its original structural state."
    )

    print()
    print("Model separation")
    print("-" * 92)

    print(
        "Fatigue, recovery, remodeling, production, and automatic"
    )

    print(
        "failure are disabled in this experiment."
    )

    print(
        "The observed loss of integrity and stiffness is therefore"
    )

    print(
        "attributable specifically to accumulated damage."
    )

    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()