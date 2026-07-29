"""
ROIF Engine
Mechanical integration test: bounded energy in an undamped spring system.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_energy.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


MASS = 1.0
STIFFNESS = 10.0
REST_LENGTH = 1.0
INITIAL_LENGTH = 1.1
DT = 0.001
STEPS = 500


def mechanical_energy(node: Node) -> float:
    """
    Total mechanical energy of the one-dimensional mass-spring system:

        E = kinetic energy + elastic potential energy
    """
    velocity_squared = sum(value * value for value in node.velocity)
    kinetic = 0.5 * MASS * velocity_squared

    extension = node.position[0] - REST_LENGTH
    elastic = 0.5 * STIFFNESS * extension * extension

    return kinetic + elastic


def test_undamped_system_energy_remains_bounded():
    """
    Undamped spring system:

        fixed -------- free mass

    The element starts stretched by 0.1.

    With zero damping and no active or biological rates:
    - elastic energy must transform into kinetic energy and back;
    - total mechanical energy must remain finite;
    - numerical integration must not create runaway energy;
    - energy drift must remain small for a short simulation.
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(INITIAL_LENGTH, 0.0, 0.0),
        mass=MASS,
    )

    material = Material(
        name="undamped_energy_material",
        parameters=MaterialParameters(
            stiffness=STIFFNESS,
            damping=0.0,
            recovery_rate=0.0,
            fatigue_rate=0.0,
            remodeling_rate=0.0,
            production_rate=0.0,
            damage_rate=0.0,
            pretension_rate=0.0,
            energy_decay_rate=0.0,
            failure_threshold=100.0,
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

    initial_energy = mechanical_energy(free)
    energies = [initial_energy]

    assert initial_energy > 0.0
    assert math.isclose(initial_energy, 0.05, rel_tol=1e-12, abs_tol=1e-12)

    for _ in range(STEPS):
        network.step(dt=DT)
        energy = mechanical_energy(free)

        assert math.isfinite(energy)
        assert energy >= 0.0

        energies.append(energy)

    final_energy = energies[-1]
    maximum_energy = max(energies)
    minimum_energy = min(energies)

    # The system must actually oscillate: the node must acquire kinetic energy.
    assert any(abs(value) > 0.0 for value in free.velocity)

    # No runaway numerical energy growth is allowed.
    assert maximum_energy < initial_energy * 1.05

    # Total energy must remain close to its initial value.
    relative_final_drift = abs(final_energy - initial_energy) / initial_energy
    assert relative_final_drift < 0.02

    # The total energy range must remain narrow over the simulated interval.
    relative_energy_range = (maximum_energy - minimum_energy) / initial_energy
    assert relative_energy_range < 0.03

    # The fixed node must remain immobile.
    assert tuple(fixed.position) == (0.0, 0.0, 0.0)

    # Simulation time must advance by the requested duration.
    assert math.isclose(
        network.time,
        STEPS * DT,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
