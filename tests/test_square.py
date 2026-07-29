"""
ROIF Engine
Mechanical integration test: unbraced square frame.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_square.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


def make_test_material(name: str) -> Material:
    """Create an independent elastic material for one frame edge."""
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=10.0,
            failure_threshold=10.0,
        ),
        reference_length=1.0,
    )


def test_unbraced_square_frame():
    """
    Unbraced square frame:

        top_left -------- top_right
            |                |
            |                |
        fixed_left ------ fixed_right

    Bottom nodes are fixed.

    Initial geometry:
        fixed_left  = (0.0, 0.0, 0.0)
        fixed_right = (1.0, 0.0, 0.0)
        top_left    = (0.0, 1.1, 0.0)
        top_right   = (1.0, 1.1, 0.0)

    Rest lengths:
        bottom = 1.0
        top    = 1.0
        sides  = 1.0

    Both vertical sides are stretched by 0.1, so the two upper nodes
    must move downward. Because the geometry and loading are symmetric,
    there must be no horizontal drift.
    """

    fixed_left = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    fixed_right = Node(
        position=(1.0, 0.0, 0.0),
        fixed=True,
    )

    top_left = Node(
        position=(0.0, 1.1, 0.0),
        mass=1.0,
    )

    top_right = Node(
        position=(1.0, 1.1, 0.0),
        mass=1.0,
    )

    bottom = Element(
        fixed_left,
        fixed_right,
        material=make_test_material("square_bottom"),
        rest_length=1.0,
    )

    left_side = Element(
        fixed_left,
        top_left,
        material=make_test_material("square_left"),
        rest_length=1.0,
    )

    top = Element(
        top_left,
        top_right,
        material=make_test_material("square_top"),
        rest_length=1.0,
    )

    right_side = Element(
        fixed_right,
        top_right,
        material=make_test_material("square_right"),
        rest_length=1.0,
    )

    network = Network()

    for node in (fixed_left, fixed_right, top_left, top_right):
        network.add_node(node)

    for element in (bottom, left_side, top, right_side):
        network.add_element(element)

    initial_fixed_left = tuple(fixed_left.position)
    initial_fixed_right = tuple(fixed_right.position)

    initial_top_left_x = top_left.position[0]
    initial_top_left_y = top_left.position[1]

    initial_top_right_x = top_right.position[0]
    initial_top_right_y = top_right.position[1]

    network.step(dt=0.01)

    # Both stretched vertical sides must pull the top nodes downward.
    assert top_left.position[1] < initial_top_left_y
    assert top_right.position[1] < initial_top_right_y

    assert top_left.velocity[1] < 0.0
    assert top_right.velocity[1] < 0.0

    # Symmetry must prevent horizontal drift.
    assert math.isclose(
        top_left.position[0],
        initial_top_left_x,
        abs_tol=1e-12,
    )
    assert math.isclose(
        top_right.position[0],
        initial_top_right_x,
        abs_tol=1e-12,
    )

    assert math.isclose(top_left.velocity[0], 0.0, abs_tol=1e-12)
    assert math.isclose(top_right.velocity[0], 0.0, abs_tol=1e-12)

    # The upper edge must remain horizontal after one symmetric step.
    assert math.isclose(
        top_left.position[1],
        top_right.position[1],
        abs_tol=1e-12,
    )

    # Fixed nodes must remain immobile.
    assert tuple(fixed_left.position) == initial_fixed_left
    assert tuple(fixed_right.position) == initial_fixed_right

    # Numerical state must remain finite.
    for node in (top_left, top_right):
        assert all(math.isfinite(value) for value in node.position)
        assert all(math.isfinite(value) for value in node.velocity)

    # Simulation time must advance.
    assert network.time > 0.0
