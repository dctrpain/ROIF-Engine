"""
ROIF Engine
Minimal integration test for one elastic element.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_single_element.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


def test_single_element():
    """A stretched elastic element must pull the free node toward the fixed node."""

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(1.2, 0.0, 0.0),
        mass=1.0,
    )

    parameters = MaterialParameters(
        stiffness=10.0,
        failure_threshold=10.0,
    )

    material = Material(
        name="test_elastic_material",
        parameters=parameters,
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

    initial_fixed_position = tuple(fixed.position)
    initial_free_x = free.position[0]

    network.step(dt=0.01)

    assert free.position[0] < initial_free_x
    assert tuple(fixed.position) == initial_fixed_position
    assert all(math.isfinite(value) for value in free.position)
    assert all(math.isfinite(value) for value in free.velocity)
    assert network.time > 0.0

