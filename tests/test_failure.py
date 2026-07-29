"""
ROIF Engine
Mechanical integration test: overloaded material fails and stops carrying force.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_failure.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


def test_overloaded_element_fails_and_stops_carrying_force():
    """
    Failure scenario:

        fixed -------- free

    Current length = 1.2
    Rest length    = 1.0
    Stiffness      = 10.0

    Initial elastic force is approximately 2.0.

    The material has:
        reference_force    = 1.0
        failure_threshold  = 0.5

    Therefore the normalized load is above the failure threshold.

    Expected behaviour:
    - the material becomes failed during a biological/material update;
    - failed material has zero integrity;
    - the element subsequently carries zero axial force;
    - all numerical values remain finite.
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(1.2, 0.0, 0.0),
        mass=1.0,
    )

    material = Material(
        name="failure_test_material",
        parameters=MaterialParameters(
            stiffness=10.0,
            damping=0.0,
            reference_force=1.0,
            overload_threshold=0.25,
            failure_threshold=0.5,
            recovery_rate=0.0,
            fatigue_rate=0.0,
            remodeling_rate=0.0,
            production_rate=0.0,
            damage_rate=0.0,
            pretension_rate=0.0,
            energy_decay_rate=0.0,
        ),
        reference_length=1.0,
    )

    element = Element(
        fixed,
        free,
        material=material,
        rest_length=1.0,
    )

    network = Network()
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)

    initial_force = element.compute_axial_force()

    assert math.isfinite(initial_force)
    assert initial_force > 0.0
    assert material.state.failed is False

    # This step must evaluate the load and update material failure state.
    network.step(
        dt=0.01,
        update_materials=True,
    )

    assert material.state.failed is True
    assert math.isclose(material.integrity(), 0.0, abs_tol=1e-12)

    # A failed material must no longer carry axial force.
    force_after_failure = element.compute_axial_force()

    assert math.isfinite(force_after_failure)
    assert math.isclose(force_after_failure, 0.0, abs_tol=1e-12)

    # Fixed support remains immobile and all dynamic values remain finite.
    assert tuple(fixed.position) == (0.0, 0.0, 0.0)
    assert all(math.isfinite(value) for value in free.position)
    assert all(math.isfinite(value) for value in free.velocity)

    assert network.time > 0.0
