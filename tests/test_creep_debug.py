"""
ROIF Engine
Diagnostic script for constant-load extension history.

This is not a pytest assertion test.
It prints time, extension, velocity, internal force, and net force so we can
distinguish true creep from an ordinary damped transient.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python tests\test_creep_debug.py
"""

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


REST_LENGTH = 1.0
STIFFNESS = 10.0
DAMPING = 2.0
MASS = 1.0

EXTERNAL_FORCE = 1.0
DT = 0.001
TOTAL_STEPS = 12000
PRINT_EVERY = 1000


def main() -> None:
    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(REST_LENGTH, 0.0, 0.0),
        mass=MASS,
    )

    material = Material(
        name="creep_debug_material",
        parameters=MaterialParameters(
            stiffness=STIFFNESS,
            damping=DAMPING,
            recovery_rate=0.0,
            fatigue_rate=0.0,
            remodeling_rate=0.0,
            production_rate=0.0,
            damage_rate=0.0,
            pretension_rate=0.0,
            energy_decay_rate=0.0,
            overload_threshold=100.0,
            failure_threshold=1000.0,
            reference_force=1.0,
            reference_strain=1.0,
        ),
        reference_length=REST_LENGTH,
    )

    element = Element(
        fixed,
        free,
        material=material,
        rest_length=REST_LENGTH,
    )

    network = Network()
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)

    constant_load = {
        free: (EXTERNAL_FORCE, 0.0, 0.0),
    }

    print(
        "step,time,extension,velocity_x,"
        "internal_force,external_force,net_force"
    )

    for step in range(TOTAL_STEPS + 1):
        if step % PRINT_EVERY == 0:
            extension = free.position[0] - REST_LENGTH
            velocity_x = free.velocity[0]

            internal_force = element.compute_axial_force()
            net_force = EXTERNAL_FORCE - internal_force

            print(
                f"{step},"
                f"{network.time:.6f},"
                f"{extension:.12f},"
                f"{velocity_x:.12f},"
                f"{internal_force:.12f},"
                f"{EXTERNAL_FORCE:.12f},"
                f"{net_force:.12f}"
            )

        if step < TOTAL_STEPS:
            network.step(
                dt=DT,
                external_forces=constant_load,
                update_materials=False,
            )

    final_extension = free.position[0] - REST_LENGTH
    theoretical_static_extension = EXTERNAL_FORCE / STIFFNESS

    print()
    print("Summary")
    print("-------")
    print(f"Final extension:              {final_extension:.12f}")
    print(
        "Elastic static extension F/k: "
        f"{theoretical_static_extension:.12f}"
    )
    print(
        "Difference:                   "
        f"{final_extension - theoretical_static_extension:.12f}"
    )


if __name__ == "__main__":
    main()
