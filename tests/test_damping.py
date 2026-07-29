"""
ROIF Engine
Mechanical integration test: damping dissipates mechanical energy.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_damping.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


MASS = 1.0
STIFFNESS = 10.0
DAMPING = 1.0
REST_LENGTH = 1.0
INITIAL_LENGTH = 1.1
DT = 0.001
STEPS = 2000


def mechanical_energy(node: Node) -> float:
    """Return kinetic plus elastic potential energy."""
    velocity_squared = sum(value * value for value in node.velocity)
    kinetic = 0.5 * MASS * velocity_squared

    extension = node.position[0] - REST_LENGTH
    elastic = 0.5 * STIFFNESS * extension * extension

    return kinetic + elastic


def test_damping_reduces_mechanical_energy():
    """
    Damped spring system:

        fixed -------- free mass

    The element starts stretched by 0.1.

    Expected behaviour:
    - the free node begins moving toward equilibrium;
    - mechanical energy remains finite and non-negative;
    - damping prevents runaway energy growth;
    - final energy is lower than initial energy;
    - oscillation amplitude decreases.
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
        name="damped_test_material",
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

    initial_fixed_position = tuple(fixed.position)
    initial_energy = mechanical_energy(free)
    initial_amplitude = abs(free.position[0] - REST_LENGTH)

    energies = [initial_energy]
    amplitudes = [initial_amplitude]

    for _ in range(STEPS):
        network.step(dt=DT)

        energy = mechanical_energy(free)
        amplitude = abs(free.position[0] - REST_LENGTH)

        assert math.isfinite(energy)
        assert energy >= 0.0
        assert math.isfinite(amplitude)

        energies.append(energy)
        amplitudes.append(amplitude)

    final_energy = energies[-1]
    maximum_energy = max(energies)

    # The spring must start pulling the mass toward equilibrium.
    assert min(node_x for node_x in [free.position[0], INITIAL_LENGTH]) <= INITIAL_LENGTH

    # Damping must dissipate a meaningful part of the initial energy.
    assert final_energy < initial_energy * 0.5

    # Numerical integration must not generate runaway energy.
    assert maximum_energy < initial_energy * 1.05

    # Compare peak amplitudes from early and late simulation windows.
    early_peak = max(amplitudes[:500])
    late_peak = max(amplitudes[-500:])
    assert late_peak < early_peak * 0.75

    # The final state should be closer to equilibrium than the initial state.
    final_amplitude = amplitudes[-1]
    assert final_amplitude < initial_amplitude

    # The fixed node must remain immobile.
    assert tuple(fixed.position) == initial_fixed_position

    # Simulation time must advance by the requested duration.
    assert math.isclose(
        network.time,
        STEPS * DT,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
