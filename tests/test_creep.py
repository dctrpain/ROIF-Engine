"""
ROIF Engine
Future specification test: true creep under constant external load.

The current base Material behaves as an elastic spring with viscous damping.
It approaches the ordinary static extension F / k and does not contain an
internal creep strain, Maxwell branch, SLS branch, or another long-term
rheological state.

This test is intentionally marked xfail. It must remain XFAIL until true
creep is implemented.

Run:
    python -m pytest tests\test_creep.py -v

Expected current result:
    XFAIL
"""

import math

import pytest

from core.node import Node
from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network


REST_LENGTH = 1.0
STIFFNESS = 10.0
DAMPING = 2.0
MASS = 1.0

EXTERNAL_FORCE = 1.0
DT = 0.002
STEPS = 6000

ELASTIC_EQUILIBRIUM_EXTENSION = EXTERNAL_FORCE / STIFFNESS
REQUIRED_CREEP_EXTENSION = ELASTIC_EQUILIBRIUM_EXTENSION * 1.20


@pytest.mark.xfail(
    reason=(
        "True creep is not implemented yet: Material has no internal "
        "creep strain or Maxwell/SLS rheological branch."
    ),
    strict=True,
)
def test_constant_load_produces_extension_beyond_elastic_equilibrium():
    """
    Future creep requirement:

        fixed -------- free mass  -> constant external force

    A purely elastic-damped material settles near:

        extension = F / k = 0.1

    A true creep-capable material must continue deforming beyond that static
    elastic equilibrium under the same sustained load.

    This specification requires at least 20% additional extension:

        final extension > 0.12
    """

    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(REST_LENGTH, 0.0, 0.0),
        mass=MASS,
    )

    material = Material(
        name="future_creep_material",
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
            overload_threshold=100.0,
            failure_threshold=1000.0,
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

    constant_load = {
        free: (EXTERNAL_FORCE, 0.0, 0.0),
    }

    for _ in range(STEPS):
        network.step(
            dt=DT,
            external_forces=constant_load,
            update_materials=False,
        )

    final_extension = free.position[0] - REST_LENGTH

    assert math.isfinite(final_extension)

    # This is the future capability requirement.
    # The present elastic-damped model settles near 0.1 and therefore XFAILs.
    assert final_extension > REQUIRED_CREEP_EXTENSION
