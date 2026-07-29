"""
ROIF Engine
Integration test for a two-element chain.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_chain.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


def make_test_material(name: str) -> Material:
    """Create an independent elastic material for one element."""
    parameters = MaterialParameters(
        stiffness=10.0,
        failure_threshold=10.0,
    )

    return Material(
        name=name,
        parameters=parameters,
        reference_length=1.0,
    )


def test_two_element_chain():
    """
    Chain geometry:

        fixed ---- middle ---- free

    Initial lengths:
        fixed-middle = 1.1
        middle-free  = 1.2

    Both elements have rest length 1.0.

    Therefore:
    - the free node must move left;
    - the middle node must move right because the second element
      is stretched more strongly than the first;
    - the fixed node must remain immobile.
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    middle = Node(
        position=(1.1, 0.0, 0.0),
        mass=1.0,
    )

    free = Node(
        position=(2.3, 0.0, 0.0),
        mass=1.0,
    )

    first_element = Element(
        fixed,
        middle,
        material=make_test_material("chain_material_1"),
        rest_length=1.0,
    )

    second_element = Element(
        middle,
        free,
        material=make_test_material("chain_material_2"),
        rest_length=1.0,
    )

    network = Network()

    network.add_node(fixed)
    network.add_node(middle)
    network.add_node(free)

    network.add_element(first_element)
    network.add_element(second_element)

    initial_fixed_position = tuple(fixed.position)
    initial_middle_x = middle.position[0]
    initial_free_x = free.position[0]

    network.step(dt=0.01)

    # The more strongly stretched second element pulls the middle node right.
    assert middle.position[0] > initial_middle_x

    # The free node is pulled toward the chain.
    assert free.position[0] < initial_free_x

    # Velocity directions must agree with the displacement directions.
    assert middle.velocity[0] > 0.0
    assert free.velocity[0] < 0.0

    # The fixed node must remain immobile.
    assert tuple(fixed.position) == initial_fixed_position

    # All dynamic values must remain finite.
    for node in (middle, free):
        assert all(math.isfinite(value) for value in node.position)
        assert all(math.isfinite(value) for value in node.velocity)

    # Simulation time must advance.
    assert network.time > 0.0
