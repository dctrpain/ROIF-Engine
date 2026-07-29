"""
ROIF Engine
Biological integration test: fatigue accumulates under sustained loading.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_fatigue.py -v
"""

import math

import pytest

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


STIFFNESS = 10.0
REST_LENGTH = 1.0
CURRENT_LENGTH = 1.2
FATIGUE_RATE = 0.5
DT = 0.1


def test_loaded_material_accumulates_fatigue():
    """
    Loaded elastic element:

        fixed -------- free

    Current length = 1.2
    Rest length    = 1.0
    Extension      = 0.2
    Stiffness      = 10.0

    Initial force:

        force = stiffness * extension = 2.0

    With reference_force = 1.0, normalized force stimulus is 2.0.
    Strain stimulus is only 0.2, so the total normalized stimulus is 2.0.

    The current fatigue law is:

        fatigue_gain
        = fatigue_rate
        * stimulus
        * (1 - fatigue)
        * dt

    Starting from fatigue = 0, with fatigue_rate = 0.5 and dt = 0.1:

        fatigue_new = 0.5 * 2.0 * 1.0 * 0.1 = 0.1
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(CURRENT_LENGTH, 0.0, 0.0),
        mass=1.0,
    )

    material = Material(
        name="fatigue_test_material",
        parameters=MaterialParameters(
            stiffness=STIFFNESS,
            damping=0.0,
            fatigue_rate=FATIGUE_RATE,
            recovery_rate=0.0,
            remodeling_rate=0.0,
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

    initial_fatigue = material.state.fatigue
    initial_update_count = material.state.update_count
    initial_fixed_position = tuple(fixed.position)

    assert initial_fatigue == pytest.approx(0.0, abs=1e-12)
    assert material.state.failed is False

    network.step(
        dt=DT,
        update_materials=True,
    )

    expected_stimulus = 2.0
    expected_fatigue = 0.1

    # Biological state must advance exactly once.
    assert material.state.update_count == initial_update_count + 1

    # The load must be measured correctly.
    assert material.state.normalized_stimulus == pytest.approx(
        expected_stimulus,
        abs=1e-12,
    )

    # Fatigue must accumulate according to the implemented law.
    assert material.state.fatigue > initial_fatigue
    assert material.state.fatigue == pytest.approx(
        expected_fatigue,
        abs=1e-12,
    )

    # No damage or failure is expected in this isolated fatigue test.
    assert material.state.damage == pytest.approx(0.0, abs=1e-12)
    assert material.state.failed is False

    # Fatigue must reduce integrity from 1.0 to 0.9.
    assert material.integrity() == pytest.approx(
        0.9,
        abs=1e-12,
    )

    # Fixed support remains immobile; all dynamic values remain finite.
    assert tuple(fixed.position) == initial_fixed_position
    assert all(math.isfinite(value) for value in free.position)
    assert all(math.isfinite(value) for value in free.velocity)
    assert math.isfinite(material.state.fatigue)

    assert network.time == pytest.approx(DT, abs=1e-12)
