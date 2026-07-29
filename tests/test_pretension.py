"""
ROIF Engine
Mechanical integration test: pretension without geometric extension.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_pretension.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import (
    Material,
    MaterialParameters,
    MaterialState,
)
from core.network import Network


def make_pretensioned_material(name: str, pretension: float) -> Material:
    """Create an elastic material with pretension stored in MaterialState."""
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=10.0,
            failure_threshold=100.0,
        ),
        state=MaterialState(
            pretension=pretension,
        ),
        reference_length=1.0,
    )


def test_pretension_moves_node_without_initial_extension():
    """
    Pretensioned element:

        fixed -------- free

    Current length = rest length = 1.0.

    Therefore ordinary elastic extension is zero.
    The only initial tensile force comes from material.state.pretension.
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(1.0, 0.0, 0.0),
        mass=1.0,
    )

    material = make_pretensioned_material(
        name="pretension_material",
        pretension=2.0,
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

    initial_fixed_position = tuple(fixed.position)
    initial_free_position = tuple(free.position)

    # No geometric extension exists before the step.
    initial_length = math.dist(fixed.position, free.position)
    assert math.isclose(initial_length, 1.0, abs_tol=1e-12)
    assert math.isclose(element.rest_length, 1.0, abs_tol=1e-12)

    # Confirm that pretension is present before integration.
    assert math.isclose(material.state.pretension, 2.0, abs_tol=1e-12)
    assert math.isclose(material.pretension_force(), 2.0, abs_tol=1e-12)

    network.step(dt=0.01)

    # Pretension alone must pull the free node toward the fixed node.
    assert free.position[0] < initial_free_position[0]
    assert free.velocity[0] < 0.0

    # Motion must remain on the element axis.
    assert math.isclose(free.position[1], 0.0, abs_tol=1e-12)
    assert math.isclose(free.position[2], 0.0, abs_tol=1e-12)
    assert math.isclose(free.velocity[1], 0.0, abs_tol=1e-12)
    assert math.isclose(free.velocity[2], 0.0, abs_tol=1e-12)

    # The fixed node must remain immobile.
    assert tuple(fixed.position) == initial_fixed_position

    # Numerical state must remain finite.
    assert all(math.isfinite(value) for value in free.position)
    assert all(math.isfinite(value) for value in free.velocity)

    # Simulation time must advance.
    assert network.time > 0.0
