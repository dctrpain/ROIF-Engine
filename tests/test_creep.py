from __future__ import annotations

"""
Regression test for sustained-load creep in the ROIF material model.

This test verifies that an explicitly enabled Standard Linear Solid (SLS)
material continues deforming beyond the instantaneous elastic equilibrium
under a constant tensile load.

For the configured system:

    F = 1
    K0 = 10

the instantaneous elastic equilibrium is:

    F / K0 = 0.1

With relaxed stiffness:

    K_inf = 5

the long-time SLS equilibrium is approximately:

    F / K_inf = 0.2

The test requires only that the final extension exceeds 0.12, so it checks
true delayed creep without depending on an exact integration trajectory.
"""

import math

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node


REST_LENGTH = 1.0
MASS = 1.0

STIFFNESS = 10.0
RELAXED_STIFFNESS = 5.0
DAMPING = 2.0
CREEP_TIME_CONSTANT = 0.5

EXTERNAL_FORCE = 1.0

DT = 0.002
STEPS = 7000

ELASTIC_EQUILIBRIUM_EXTENSION = (
    EXTERNAL_FORCE / STIFFNESS
)
REQUIRED_CREEP_EXTENSION = (
    ELASTIC_EQUILIBRIUM_EXTENSION * 1.20
)


def test_constant_load_produces_extension_beyond_elastic_equilibrium() -> None:
    """
    A creep-capable material must deform beyond F / K0 under sustained load.

    Rheology is enabled explicitly, and material evolution is enabled in the
    network step so that the internal SLS state advances once per timestep.
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(REST_LENGTH, 0.0, 0.0),
        mass=MASS,
    )

    material = Material(
        name="sls_creep_material",
        parameters=MaterialParameters(
            stiffness=STIFFNESS,
            relaxed_stiffness=RELAXED_STIFFNESS,
            damping=DAMPING,
            creep_time_constant=CREEP_TIME_CONSTANT,
            rheology_enabled=True,
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

    network = Network(
        global_damping=0.0,
        record_history=False,
    )
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)

    constant_load = {
        free: (
            EXTERNAL_FORCE,
            0.0,
            0.0,
        ),
    }

    for _ in range(STEPS):
        network.step(
            dt=DT,
            external_forces=constant_load,
            update_materials=True,
        )

    final_extension = float(
        free.position[0] - REST_LENGTH
    )

    assert math.isfinite(final_extension)
    assert math.isfinite(
        float(material.state.creep_strain)
    )
    assert material.state.rheology_time > 0.0
    assert material.state.creep_strain > 0.0

    assert (
        final_extension
        > REQUIRED_CREEP_EXTENSION
    )
