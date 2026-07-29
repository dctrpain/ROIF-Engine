"""
ROIF Engine
Biological integration test: remodeling responds to sustained underloading.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_remodeling.py -v
"""

import math

import pytest

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


REST_LENGTH = 1.0
REMODELING_RATE = 1.0
DT = 0.1


def test_underloaded_material_remodeling_decreases():
    """
    Unloaded element:

        fixed -------- free

    Current length equals rest length, so:

        extension = 0
        force = 0
        normalized stimulus = 0

    The current Material law defines the remodeling target as:

        target = clip(1 + 0.25 * (stimulus - 1), 0, 1)

    Therefore, at stimulus = 0:

        target = 0.75

    Starting from remodeling = 1.0, one biological update with
    remodeling_rate = 1.0 and dt = 0.1 must produce:

        remodeling_new
        = 1.0 + 1.0 * (0.75 - 1.0) * 0.1
        = 0.975
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(REST_LENGTH, 0.0, 0.0),
        mass=1.0,
    )

    material = Material(
        name="remodeling_test_material",
        parameters=MaterialParameters(
            stiffness=10.0,
            damping=0.0,
            remodeling_rate=REMODELING_RATE,
            recovery_rate=0.0,
            fatigue_rate=0.0,
            production_rate=0.0,
            damage_rate=0.0,
            pretension_rate=0.0,
            energy_decay_rate=0.0,
            overload_threshold=10.0,
            failure_threshold=100.0,
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

    network = Network()
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)

    initial_fixed_position = tuple(fixed.position)
    initial_free_position = tuple(free.position)
    initial_remodeling = material.state.remodeling
    initial_update_count = material.state.update_count

    assert initial_remodeling == pytest.approx(1.0, abs=1e-12)
    assert material.state.failed is False

    network.step(
        dt=DT,
        update_materials=True,
    )

    expected_remodeling = 0.975

    # Biological state must advance exactly once.
    assert material.state.update_count == initial_update_count + 1

    # Zero force and zero strain produce zero normalized stimulus.
    assert material.state.normalized_stimulus == pytest.approx(
        0.0,
        abs=1e-12,
    )

    # Remodeling must move from 1.0 toward the underload target 0.75.
    assert material.state.remodeling < initial_remodeling
    assert material.state.remodeling == pytest.approx(
        expected_remodeling,
        abs=1e-12,
    )

    # No mechanical motion should occur because the element is at rest length.
    assert tuple(fixed.position) == initial_fixed_position
    assert tuple(free.position) == initial_free_position
    assert all(
        math.isclose(value, 0.0, abs_tol=1e-12)
        for value in free.velocity
    )

    # The material must remain valid and unfailed.
    assert material.state.failed is False
    assert math.isfinite(material.state.remodeling)
    assert network.time == pytest.approx(DT, abs=1e-12)
