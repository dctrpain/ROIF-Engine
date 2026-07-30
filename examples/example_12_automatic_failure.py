r"""
ROIF Engine Example 12

Automatic Failure Threshold with separated force, strain, and combined
stimulus reporting.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node

BASE_STIFFNESS = 100.0
REFERENCE_LENGTH = 1.0
REFERENCE_FORCE = 20.0
REFERENCE_STRAIN = 1.0

INITIAL_EXTENSION = 0.0
FINAL_EXTENSION = 0.30

FAILURE_ELEMENT_THRESHOLD = 0.80
CONTROL_FAILURE_THRESHOLD = 100.0
OVERLOAD_THRESHOLD = 100.0

TIME_STEP = 0.01
LOADING_STEPS = 300
REPORT_INTERVAL = 25
POST_FAILURE_STEPS = 100


def create_material(name: str, failure_threshold: float) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=BASE_STIFFNESS,
            damping=0.0,
            fatigue_rate=0.0,
            damage_rate=0.0,
            recovery_rate=0.0,
            remodeling_rate=0.0,
            production_rate=0.0,
            pretension_rate=0.0,
            energy_decay_rate=0.0,
            overload_threshold=OVERLOAD_THRESHOLD,
            failure_threshold=failure_threshold,
            reference_force=REFERENCE_FORCE,
            reference_strain=REFERENCE_STRAIN,
        ),
    )


def create_element(
    name: str,
    start_x: float,
    failure_threshold: float,
) -> tuple[Node, Node, Element]:
    node_a = Node(
        position=[start_x, 0.0],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_node_a",
    )
    node_b = Node(
        position=[start_x + REFERENCE_LENGTH, 0.0],
        mass=1.0,
        fixed=True,
        node_id=f"{name}_node_b",
    )
    element = Element(
        node_a=node_a,
        node_b=node_b,
        rest_length=REFERENCE_LENGTH,
        material=create_material(
            f"{name}_material",
            failure_threshold,
        ),
        element_id=f"{name}_element",
        name=name,
    )
    return node_a, node_b, element


def set_extension(node_a: Node, node_b: Node, extension: float) -> None:
    node_b.set_position(
        [
            node_a.position[0] + REFERENCE_LENGTH + extension,
            node_a.position[1],
        ]
    )
    node_b.set_velocity([0.0, 0.0])


def evaluate_forces(network: Network) -> None:
    network.clear_forces()
    network.assemble_element_forces(include_active=False)


def strain(element: Element) -> float:
    reference_length = element.material_reference_length()
    if reference_length == 0.0:
        return 0.0
    return float(element.extension() / reference_length)


def force_stimulus(element: Element) -> float:
    return element.material.normalized_force_stimulus(
        force=element.last_force.total,
    )


def strain_stimulus(element: Element) -> float:
    return element.material.normalized_strain_stimulus(
        strain=strain(element),
    )


def combined_stimulus(element: Element) -> float:
    return element.material.normalized_stimulus(
        force=element.last_force.total,
        strain=strain(element),
    )


def failure_ratio(element: Element) -> float:
    threshold = element.material.parameters.failure_threshold
    return float(combined_stimulus(element) / threshold)


def print_state(title: str, element: Element) -> None:
    material = element.material
    state = material.state

    print()
    print(title)
    print("-" * 104)
    print(f"Element name          : {element.name}")
    print(f"Current length        : {element.current_length():.6f} m")
    print(f"Reference length      : {element.material_reference_length():.6f} m")
    print(f"Extension             : {element.extension():.6f} m")
    print(f"Strain                : {strain(element):.6f}")
    print(f"Failure threshold     : {material.parameters.failure_threshold:.6f}")
    print(f"Force stimulus        : {force_stimulus(element):.6f}")
    print(f"Strain stimulus       : {strain_stimulus(element):.6f}")
    print(f"Combined stimulus     : {combined_stimulus(element):.6f}")
    print(f"Stored stimulus       : {state.normalized_stimulus:.6f}")
    print(f"Failure ratio         : {failure_ratio(element):.6f}")
    print(f"Damage                : {state.damage:.6f}")
    print(f"Fatigue               : {state.fatigue:.6f}")
    print(f"Integrity             : {material.integrity():.6f}")
    print(f"Effective stiffness   : {material.effective_stiffness():.6f} N/m")
    print(f"Total axial force     : {element.last_force.total:.6f} N")
    print(f"Failed                : {state.failed}")
    print(f"Biological updates    : {state.update_count}")


def main() -> None:
    print("=" * 104)
    print("ROIF Engine - Example 12")
    print("Automatic Failure Threshold")
    print("=" * 104)

    control_a, control_b, control = create_element(
        "High-threshold control",
        0.0,
        CONTROL_FAILURE_THRESHOLD,
    )
    failure_a, failure_b, failure = create_element(
        "Automatic-failure element",
        2.0,
        FAILURE_ELEMENT_THRESHOLD,
    )

    network = Network(
        gravity=None,
        global_damping=0.0,
        record_history=True,
    )
    for node in (control_a, control_b, failure_a, failure_b):
        network.add_node(node)
    network.add_element(control)
    network.add_element(failure)

    evaluate_forces(network)
    print_state("Initial control element", control)
    print_state("Initial automatic-failure element", failure)

    print()
    print("Phase 1 — progressive prescribed loading")
    print("-" * 104)
    print(
        "time       extension   control_F   failure_F   "
        "force_stim   strain_stim  combined     ratio      failed"
    )

    failure_data = None

    for step in range(1, LOADING_STEPS + 1):
        extension = (
            INITIAL_EXTENSION
            + (FINAL_EXTENSION - INITIAL_EXTENSION)
            * step
            / LOADING_STEPS
        )

        set_extension(control_a, control_b, extension)
        set_extension(failure_a, failure_b, extension)
        evaluate_forces(network)

        pre_force = float(failure.last_force.total)
        pre_force_stimulus = float(force_stimulus(failure))
        pre_strain_stimulus = float(strain_stimulus(failure))
        pre_combined = float(combined_stimulus(failure))
        pre_ratio = float(pre_combined / FAILURE_ELEMENT_THRESHOLD)
        was_failed = failure.material.state.failed

        network.step(
            dt=TIME_STEP,
            update_materials=True,
            include_active=False,
            solve_constraints=True,
            record=True,
        )
        evaluate_forces(network)

        is_failed = failure.material.state.failed
        failure_now = not was_failed and is_failed

        if failure_now:
            failure_data = {
                "step": step,
                "time": network.time,
                "extension": extension,
                "force": pre_force,
                "force_stimulus": pre_force_stimulus,
                "strain_stimulus": pre_strain_stimulus,
                "combined": pre_combined,
                "ratio": pre_ratio,
            }

        if failure_now or step % REPORT_INTERVAL == 0:
            marker = "  <-- FAILURE" if failure_now else ""
            print(
                f"{network.time:8.3f}   "
                f"{extension:9.6f}   "
                f"{control.last_force.total:9.6f}   "
                f"{failure.last_force.total:9.6f}   "
                f"{pre_force_stimulus:10.6f}   "
                f"{pre_strain_stimulus:11.6f}   "
                f"{pre_combined:10.6f}   "
                f"{pre_ratio:8.6f}   "
                f"{is_failed}{marker}"
            )

        if is_failed:
            break

    print_state("Control element at failure event", control)
    print_state("Automatic-failure element after transition", failure)

    print()
    print("Phase 2 — post-failure loading")
    print("-" * 104)

    if failure_data is None:
        print("Failure was not reached. Increase FINAL_EXTENSION or lower the threshold.")
    else:
        print(
            "time       extension   control_F   failed_F    "
            "force_stim   strain_stim  combined     failed"
        )

        start_extension = float(failure_data["extension"])

        for step in range(1, POST_FAILURE_STEPS + 1):
            extension = (
                start_extension
                + (FINAL_EXTENSION - start_extension)
                * step
                / POST_FAILURE_STEPS
            )

            set_extension(control_a, control_b, extension)
            set_extension(failure_a, failure_b, extension)

            network.step(
                dt=TIME_STEP,
                update_materials=True,
                include_active=False,
                solve_constraints=True,
                record=True,
            )
            evaluate_forces(network)

            if step % 20 == 0 or step == POST_FAILURE_STEPS:
                print(
                    f"{network.time:8.3f}   "
                    f"{extension:9.6f}   "
                    f"{control.last_force.total:9.6f}   "
                    f"{failure.last_force.total:8.6f}   "
                    f"{force_stimulus(failure):10.6f}   "
                    f"{strain_stimulus(failure):11.6f}   "
                    f"{combined_stimulus(failure):10.6f}   "
                    f"{failure.material.state.failed}"
                )

    print_state("Final control element", control)
    print_state("Final failed element", failure)

    print()
    print("Comparison")
    print("-" * 104)
    print(f"Failure detected                    : {failure_data is not None}")

    if failure_data is not None:
        print(f"Failure step                        : {failure_data['step']}")
        print(f"Failure time                        : {failure_data['time']:.6f} s")
        print(f"Extension at failure                : {failure_data['extension']:.6f} m")
        print(f"Force immediately before failure    : {failure_data['force']:.6f} N")
        print(f"Force stimulus before failure       : {failure_data['force_stimulus']:.6f}")
        print(f"Strain stimulus before failure      : {failure_data['strain_stimulus']:.6f}")
        print(f"Combined stimulus before failure    : {failure_data['combined']:.6f}")
        print(f"Failure ratio before transition     : {failure_data['ratio']:.6f}")

    print(f"Final control force                 : {control.last_force.total:.6f} N")
    print(f"Final failed-element force          : {failure.last_force.total:.6f} N")
    print(f"Final failed force stimulus         : {force_stimulus(failure):.6f}")
    print(f"Final failed strain stimulus        : {strain_stimulus(failure):.6f}")
    print(f"Final failed combined stimulus      : {combined_stimulus(failure):.6f}")
    print(f"Final failed-element stiffness      : {failure.material.effective_stiffness():.6f} N/m")
    print(f"Final failed-element integrity      : {failure.material.integrity():.6f}")
    print(f"Experimental element failed         : {failure.material.state.failed}")

    print()
    print("Interpretation")
    print("-" * 104)
    print("Force stimulus describes transmitted mechanical load.")
    print("Strain stimulus describes geometric deformation.")
    print("Combined stimulus remains the maximum of the two values.")
    print("After failure, force stimulus is zero while strain stimulus may remain nonzero.")
    print()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()
