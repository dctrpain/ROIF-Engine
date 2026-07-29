"""
ROIF Engine
Mechanical integration test: square frame with cross bracing.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_cross_bracing.py -v
"""

import math

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


def make_test_material(name: str) -> Material:
    """Create an independent elastic material for one structural element."""
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=10.0,
            failure_threshold=10.0,
        ),
        reference_length=1.0,
    )


def test_cross_braced_square_frame():
    r"""
    Cross-braced square frame:

        top_left -------- top_right
            | \          / |
            |   \      /   |
            |     \  /     |
            |      X       |
            |     /  \     |
            |   /      \   |
            | /          \ |
        fixed_left ------ fixed_right

    Bottom nodes are fixed.

    Initial geometry:
        fixed_left  = (0.0, 0.0, 0.0)
        fixed_right = (1.0, 0.0, 0.0)
        top_left    = (0.0, 1.1, 0.0)
        top_right   = (1.0, 1.1, 0.0)

    Rest lengths:
        horizontal edges = 1.0
        vertical edges   = 1.0
        diagonals        = sqrt(2.0)

    Because the frame height is initially 1.1:
    - both vertical elements are stretched;
    - both diagonal elements are stretched;
    - both upper nodes must move downward;
    - the diagonal braces must pull the upper nodes inward;
    - mirror symmetry must be preserved.
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
        material=make_test_material("cross_bottom"),
        rest_length=1.0,
    )

    left_side = Element(
        fixed_left,
        top_left,
        material=make_test_material("cross_left_side"),
        rest_length=1.0,
    )

    top = Element(
        top_left,
        top_right,
        material=make_test_material("cross_top"),
        rest_length=1.0,
    )

    right_side = Element(
        fixed_right,
        top_right,
        material=make_test_material("cross_right_side"),
        rest_length=1.0,
    )

    diagonal_left_to_right = Element(
        fixed_left,
        top_right,
        material=make_test_material("cross_diagonal_left_to_right"),
        rest_length=math.sqrt(2.0),
    )

    diagonal_right_to_left = Element(
        fixed_right,
        top_left,
        material=make_test_material("cross_diagonal_right_to_left"),
        rest_length=math.sqrt(2.0),
    )

    network = Network()

    for node in (fixed_left, fixed_right, top_left, top_right):
        network.add_node(node)

    for element in (
        bottom,
        left_side,
        top,
        right_side,
        diagonal_left_to_right,
        diagonal_right_to_left,
    ):
        network.add_element(element)

    initial_fixed_left = tuple(fixed_left.position)
    initial_fixed_right = tuple(fixed_right.position)

    initial_top_left_x = top_left.position[0]
    initial_top_left_y = top_left.position[1]

    initial_top_right_x = top_right.position[0]
    initial_top_right_y = top_right.position[1]

    initial_top_width = initial_top_right_x - initial_top_left_x

    network.step(dt=0.01)

    # Vertical and diagonal tension must pull both upper nodes downward.
    assert top_left.position[1] < initial_top_left_y
    assert top_right.position[1] < initial_top_right_y
    assert top_left.velocity[1] < 0.0
    assert top_right.velocity[1] < 0.0

    # Cross braces must pull the upper nodes inward.
    assert top_left.position[0] > initial_top_left_x
    assert top_right.position[0] < initial_top_right_x
    assert top_left.velocity[0] > 0.0
    assert top_right.velocity[0] < 0.0

    # The upper span must become narrower.
    final_top_width = top_right.position[0] - top_left.position[0]
    assert final_top_width < initial_top_width

    # Mirror symmetry must be preserved.
    assert math.isclose(
        top_left.position[0] + top_right.position[0],
        1.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        top_left.position[1],
        top_right.position[1],
        abs_tol=1e-12,
    )
    assert math.isclose(
        top_left.velocity[0],
        -top_right.velocity[0],
        abs_tol=1e-12,
    )
    assert math.isclose(
        top_left.velocity[1],
        top_right.velocity[1],
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
