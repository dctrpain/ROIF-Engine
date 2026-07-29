"""
ROIF Engine
Biological integration test: fatigue and damage recover without load.

Run from the project folder that contains:
    core/
    tests/
    main.py

Command:
    python -m pytest tests\test_recovery.py -v
"""

import math

import pytest

from core.node import Node
from core.element import Element
from core.material import (
    Material,
    MaterialParameters,
    MaterialState,
)
from core.network import Network


REST_LENGTH = 1.0
RECOVERY_RATE = 0.5
INITIAL_FATIGUE = 0.4
INITIAL_DAMAGE = 0.3
DT = 0.1


def test_unloaded_material_recovers_fatigue_and_damage():
    """
    Unloaded material starts with fatigue and damage:

        fatigue = 0.4
        damage  = 0.3
        energy  = 1.0

    Current length equals rest length, therefore:

        force = 0
        strain = 0
        stimulus = 0

    Current fatigue recovery law:

        fatigue_recovery
        = recovery_rate * energy * fatigue * dt

        = 0.5 * 1.0 * 0.4 * 0.1
        = 0.02

        fatigue_new = 0.4 - 0.02 = 0.38

    Current damage repair law at zero overload:

        damage_repair
        = recovery_rate * energy * damage * dt

        = 0.5 * 1.0 * 0.3 * 0.1
        = 0.015

        damage_new = 0.3 - 0.015 = 0.285
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
        name="recovery_test_material",
        parameters=MaterialParameters(
            stiffness=10.0,
            damping=0.0,
            recovery_rate=RECOVERY_RATE,
            fatigue_rate=0.0,
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
        state=MaterialState(
            fatigue=INITIAL_FATIGUE,
            damage=INITIAL_DAMAGE,
            energy=1.0,
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

    initial_integrity = material.integrity()
    initial_update_count = material.state.update_count
    initial_fixed_position = tuple(fixed.position)
    initial_free_position = tuple(free.position)

    assert material.state.fatigue == pytest.approx(
        INITIAL_FATIGUE,
        abs=1e-12,
    )
    assert material.state.damage == pytest.approx(
        INITIAL_DAMAGE,
        abs=1e-12,
    )
    assert material.state.failed is False

    network.step(
        dt=DT,
        update_materials=True,
    )

    expected_fatigue = 0.38
    expected_damage = 0.285
    expected_integrity = (
        (1.0 - expected_damage)
        * (1.0 - expected_fatigue)
    )

    # Biological state must advance exactly once.
    assert material.state.update_count == initial_update_count + 1

    # At rest length, the material receives no mechanical stimulus.
    assert material.state.normalized_stimulus == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert material.state.overloaded is False

    # Fatigue and damage must both decrease.
    assert material.state.fatigue < INITIAL_FATIGUE
    assert material.state.damage < INITIAL_DAMAGE

    assert material.state.fatigue == pytest.approx(
        expected_fatigue,
        abs=1e-12,
    )
    assert material.state.damage == pytest.approx(
        expected_damage,
        abs=1e-12,
    )

    # Recovery must increase remaining structural integrity.
    assert material.integrity() > initial_integrity
    assert material.integrity() == pytest.approx(
        expected_integrity,
        abs=1e-12,
    )

    # No motion or failure is expected.
    assert tuple(fixed.position) == initial_fixed_position
    assert tuple(free.position) == initial_free_position
    assert all(
        math.isclose(value, 0.0, abs_tol=1e-12)
        for value in free.velocity
    )
    assert material.state.failed is False

    # All relevant values must remain finite.
    for value in (
        material.state.fatigue,
        material.state.damage,
        material.integrity(),
        network.time,
    ):
        assert math.isfinite(value)

    assert network.time == pytest.approx(DT, abs=1e-12)
