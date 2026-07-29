"""
ROIF Engine
Mechanical integration test: velocity-dependent viscoelastic force.

The current base Material implements a Kelvin-Voigt-type axial response:

    total force = elastic force + viscous damping force

For the same extension:
- zero length velocity gives the purely elastic force;
- positive length velocity increases tensile force;
- negative length velocity reduces tensile force.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_viscoelastic.py -v
"""

import math

import pytest

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters


STIFFNESS = 10.0
DAMPING = 2.0
REST_LENGTH = 1.0
CURRENT_LENGTH = 1.1
EXTENSION = CURRENT_LENGTH - REST_LENGTH
LENGTH_SPEED = 0.5


def make_viscoelastic_element() -> tuple[Node, Node, Material, Element]:
    """Create one stretched axial viscoelastic element."""

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    moving = Node(
        position=(CURRENT_LENGTH, 0.0, 0.0),
        mass=1.0,
    )

    material = Material(
        name="viscoelastic_test_material",
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
            overload_threshold=10.0,
            failure_threshold=100.0,
        ),
        reference_length=REST_LENGTH,
    )

    element = Element(
        fixed,
        moving,
        material=material,
        rest_length=REST_LENGTH,
    )

    return fixed, moving, material, element


def test_viscoelastic_force_depends_on_length_velocity():
    """
    Verify the implemented elastic-plus-viscous constitutive law.

    With:
        stiffness = 10
        extension = 0.1
        damping = 2
        speed = 0.5

    Expected components:
        elastic force = 10 * 0.1 = 1.0
        damping force = 2 * 0.5 = 1.0

    Therefore:
        static force     = 1.0
        extending force  = 2.0
        shortening force = 0.0
    """

    _, moving, material, element = make_viscoelastic_element()

    # Element length velocity is obtained from relative nodal velocity.
    moving.velocity[:] = 0.0
    static_force = element.compute_axial_force()

    moving.velocity[:] = (LENGTH_SPEED, 0.0, 0.0)
    extending_force = element.compute_axial_force()

    moving.velocity[:] = (-LENGTH_SPEED, 0.0, 0.0)
    shortening_force = element.compute_axial_force()

    expected_elastic = STIFFNESS * EXTENSION
    expected_damping = DAMPING * LENGTH_SPEED

    assert static_force == pytest.approx(
        expected_elastic,
        abs=1e-12,
    )

    assert extending_force == pytest.approx(
        expected_elastic + expected_damping,
        abs=1e-12,
    )

    assert shortening_force == pytest.approx(
        expected_elastic - expected_damping,
        abs=1e-12,
    )

    # At identical geometry, force must vary with deformation rate.
    assert extending_force > static_force
    assert shortening_force < static_force

    # Symmetry around the purely elastic response.
    assert extending_force - static_force == pytest.approx(
        static_force - shortening_force,
        abs=1e-12,
    )

    # Diagnostic force components must match the last calculation.
    assert element.last_force.elastic == pytest.approx(
        expected_elastic,
        abs=1e-12,
    )
    assert element.last_force.damping == pytest.approx(
        -expected_damping,
        abs=1e-12,
    )
    assert element.last_force.total == pytest.approx(
        shortening_force,
        abs=1e-12,
    )

    # Material state and all calculated values must remain finite.
    assert material.state.failed is False

    for value in (
        static_force,
        extending_force,
        shortening_force,
        element.last_force.elastic,
        element.last_force.damping,
        element.last_force.total,
    ):
        assert math.isfinite(value)
