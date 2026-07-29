"""
ROIF Engine
Mechanical integration test: symmetric triangular contour.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_triangle.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


def make_test_material(name: str) -> Material:
    """Create an independent elastic material for one triangle edge."""
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=10.0,
            failure_threshold=10.0,
        ),
        reference_length=1.0,
    )


def test_symmetric_triangle():
    """
    Symmetric triangular contour:

              apex
              /  \
             /    \
        fixed------fixed

    The two lower nodes are fixed.

    Initial geometry:
        left  = (-0.5, 0.0, 0.0)
        right = ( 0.5, 0.0, 0.0)
        apex  = ( 0.0, 1.0, 0.0)

    The base has current and rest length 1.0.

    Each diagonal has current length sqrt(1.25), but rest length 1.0.
    Therefore both diagonals are stretched and pull the apex downward.
    Their horizontal force components must cancel by symmetry.
    """

    left = Node(
        position=(-0.5, 0.0, 0.0),
        fixed=True,
    )

    right = Node(
        position=(0.5, 0.0, 0.0),
        fixed=True,
    )

    apex = Node(
        position=(0.0, 1.0, 0.0),
        mass=1.0,
    )

    base = Element(
        left,
        right,
        material=make_test_material("triangle_base"),
        rest_length=1.0,
    )

    left_side = Element(
        left,
        apex,
        material=make_test_material("triangle_left"),
        rest_length=1.0,
    )

    right_side = Element(
        right,
        apex,
        material=make_test_material("triangle_right"),
        rest_length=1.0,
    )

    network = Network()

    for node in (left, right, apex):
        network.add_node(node)

    for element in (base, left_side, right_side):
        network.add_element(element)

    initial_left_position = tuple(left.position)
    initial_right_position = tuple(right.position)
    initial_apex_x = apex.position[0]
    initial_apex_y = apex.position[1]

    network.step(dt=0.01)

    # Both stretched diagonal elements must pull the apex downward.
    assert apex.position[1] < initial_apex_y
    assert apex.velocity[1] < 0.0

    # Symmetry must cancel horizontal motion.
    assert math.isclose(apex.position[0], initial_apex_x, abs_tol=1e-12)
    assert math.isclose(apex.velocity[0], 0.0, abs_tol=1e-12)

    # Fixed base nodes must remain immobile.
    assert tuple(left.position) == initial_left_position
    assert tuple(right.position) == initial_right_position

    # Numerical state must remain finite.
    assert all(math.isfinite(value) for value in apex.position)
    assert all(math.isfinite(value) for value in apex.velocity)

    # Simulation time must advance.
    assert network.time > 0.0
