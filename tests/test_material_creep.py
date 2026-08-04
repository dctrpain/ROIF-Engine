from __future__ import annotations

"""
ROIF Engine
Material creep and viscoelastic-memory specification.

Purpose
-------
This module defines the expected physical behaviour of the future ROIF
viscoelastic material model.

The target constitutive family is a finite-creep Standard Linear Solid (SLS)
or an equivalent internal-variable formulation with:

- instantaneous elastic response;
- delayed creep under sustained load;
- finite relaxed equilibrium;
- stress relaxation under fixed extension;
- partial recovery after unloading;
- serializable internal rheological state;
- stable time integration;
- backward compatibility when rheology is disabled.

Current status
--------------
The ordinary elastic reference tests should pass now.

The rheology tests are active acceptance tests for the implemented SLS/internal-
variable material model.

Proposed public parameters
--------------------------
MaterialParameters.relaxed_stiffness:
    Long-time equilibrium stiffness K_inf.

MaterialParameters.creep_time_constant:
    Characteristic rheological time tau.

MaterialParameters.rheology_enabled:
    Explicit switch for internal viscoelastic evolution.

Proposed public state
---------------------
MaterialState.creep_strain:
    Delayed internal strain.

MaterialState.rheology_time:
    Accumulated rheological integration time.

The exact internal numerical method may change. The observable physical
requirements below should remain stable.
"""

import math
from collections.abc import Iterable

import numpy as np
import pytest

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REST_LENGTH = 1.0
INSTANTANEOUS_STIFFNESS = 10.0
RELAXED_STIFFNESS = 5.0
DAMPING = 2.0
MASS = 1.0

EXTERNAL_FORCE = 1.0

DEFAULT_DT = 0.002
FAST_TIME_CONSTANT = 0.25
SLOW_TIME_CONSTANT = 1.00

ELASTIC_EXTENSION = (
    EXTERNAL_FORCE / INSTANTANEOUS_STIFFNESS
)
RELAXED_EXTENSION = (
    EXTERNAL_FORCE / RELAXED_STIFFNESS
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_parameters(
    *,
    stiffness: float = INSTANTANEOUS_STIFFNESS,
    damping: float = DAMPING,
    relaxed_stiffness: float | None = None,
    creep_time_constant: float | None = None,
    rheology_enabled: bool | None = None,
) -> MaterialParameters:
    """
    Build parameters while keeping biological evolution disabled.

    Future rheology fields are passed only when explicitly requested. This
    allows the elastic reference tests to run against the current API.
    """

    values: dict[str, object] = {
        "stiffness": stiffness,
        "damping": damping,
        "recovery_rate": 0.0,
        "fatigue_rate": 0.0,
        "remodeling_rate": 0.0,
        "production_rate": 0.0,
        "damage_rate": 0.0,
        "pretension_rate": 0.0,
        "energy_decay_rate": 0.0,
        "overload_threshold": 100.0,
        "failure_threshold": 1000.0,
        "reference_force": 1.0,
        "reference_strain": 1.0,
    }

    if relaxed_stiffness is not None:
        values["relaxed_stiffness"] = (
            relaxed_stiffness
        )

    if creep_time_constant is not None:
        values["creep_time_constant"] = (
            creep_time_constant
        )

    if rheology_enabled is not None:
        values["rheology_enabled"] = (
            rheology_enabled
        )

    return MaterialParameters(**values)


def make_material(
    *,
    rheology_enabled: bool = False,
    relaxed_stiffness: float = RELAXED_STIFFNESS,
    creep_time_constant: float = SLOW_TIME_CONSTANT,
    stiffness: float = INSTANTANEOUS_STIFFNESS,
    damping: float = DAMPING,
) -> Material:
    parameters = make_parameters(
        stiffness=stiffness,
        damping=damping,
        relaxed_stiffness=(
            relaxed_stiffness
            if rheology_enabled
            else None
        ),
        creep_time_constant=(
            creep_time_constant
            if rheology_enabled
            else None
        ),
        rheology_enabled=(
            True
            if rheology_enabled
            else None
        ),
    )

    return Material(
        name=(
            "sls_material"
            if rheology_enabled
            else "elastic_reference_material"
        ),
        parameters=parameters,
        reference_length=REST_LENGTH,
    )


def make_loaded_network(
    *,
    material: Material,
    initial_length: float = REST_LENGTH,
    fixed_free_node: bool = False,
) -> tuple[Network, Node, Node, Element]:
    fixed = Node(
        position=(0.0, 0.0, 0.0),
        fixed=True,
    )

    free = Node(
        position=(initial_length, 0.0, 0.0),
        mass=MASS,
        fixed=fixed_free_node,
    )

    element = Element(
        fixed,
        free,
        material=material,
        rest_length=REST_LENGTH,
    )

    network = Network(
        global_damping=0.0,
        record_history=False,
    )
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)

    return network, fixed, free, element


def run_steps(
    network: Network,
    *,
    steps: int,
    dt: float,
    external_forces: dict[Node, tuple[float, ...]]
    | None = None,
    update_materials: bool = True,
) -> None:
    for _ in range(steps):
        network.step(
            dt=dt,
            external_forces=external_forces,
            update_materials=update_materials,
        )


def extension(node: Node) -> float:
    return float(node.position[0] - REST_LENGTH)


def constant_load_for(
    node: Node,
    *,
    force: float = EXTERNAL_FORCE,
) -> dict[Node, tuple[float, float, float]]:
    return {
        node: (force, 0.0, 0.0),
    }


def sample_extensions(
    network: Network,
    node: Node,
    *,
    total_steps: int,
    sample_every: int,
    dt: float,
    external_forces: dict[
        Node,
        tuple[float, float, float],
    ],
) -> list[float]:
    values: list[float] = []

    for index in range(total_steps):
        network.step(
            dt=dt,
            external_forces=external_forces,
            update_materials=True,
        )

        if (index + 1) % sample_every == 0:
            values.append(extension(node))

    return values


def relative_difference(
    left: float,
    right: float,
) -> float:
    scale = max(
        abs(left),
        abs(right),
        1e-12,
    )
    return abs(left - right) / scale


def assert_all_finite(
    values: Iterable[float],
) -> None:
    assert all(
        math.isfinite(float(value))
        for value in values
    )


# ---------------------------------------------------------------------------
# Elastic reference and backward compatibility
# ---------------------------------------------------------------------------


def test_elastic_reference_reaches_hookean_equilibrium() -> None:
    """
    The current elastic-damped material must settle near F / K.

    This is the baseline against which true creep is distinguished.
    """

    material = make_material(
        rheology_enabled=False
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    run_steps(
        network,
        steps=8000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
        update_materials=False,
    )

    final_extension = extension(free)

    assert math.isfinite(final_extension)
    assert final_extension == pytest.approx(
        ELASTIC_EXTENSION,
        rel=0.03,
        abs=0.003,
    )


def test_disabled_rheology_preserves_elastic_model() -> None:
    """
    Adding rheology support must not change legacy behaviour when disabled.
    """

    material = make_material(
        rheology_enabled=False
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    run_steps(
        network,
        steps=6000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
        update_materials=True,
    )

    assert extension(free) == pytest.approx(
        ELASTIC_EXTENSION,
        rel=0.04,
        abs=0.004,
    )


# ---------------------------------------------------------------------------
# Future rheology specification
# ---------------------------------------------------------------------------


def test_constant_load_produces_true_creep() -> None:
    """
    Sustained load must produce delayed extension beyond F / K0.
    """

    material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.50,
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    run_steps(
        network,
        steps=7000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
    )

    final_extension = extension(free)

    assert math.isfinite(final_extension)
    assert final_extension > ELASTIC_EXTENSION * 1.20


def test_creep_rate_depends_on_time_constant() -> None:
    """
    Smaller tau must approach the relaxed state faster.
    """

    fast_material = make_material(
        rheology_enabled=True,
        creep_time_constant=FAST_TIME_CONSTANT,
    )
    slow_material = make_material(
        rheology_enabled=True,
        creep_time_constant=SLOW_TIME_CONSTANT,
    )

    fast_network, _, fast_node, _ = (
        make_loaded_network(
            material=fast_material
        )
    )
    slow_network, _, slow_node, _ = (
        make_loaded_network(
            material=slow_material
        )
    )

    run_steps(
        fast_network,
        steps=1000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(
            fast_node
        ),
    )
    run_steps(
        slow_network,
        steps=1000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(
            slow_node
        ),
    )

    fast_extension = extension(fast_node)
    slow_extension = extension(slow_node)

    assert fast_extension > slow_extension
    assert fast_extension > ELASTIC_EXTENSION


def test_sls_creep_reaches_finite_relaxed_limit() -> None:
    """
    SLS creep must converge to F / K_inf rather than diverge.
    """

    material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.35,
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    samples = sample_extensions(
        network,
        free,
        total_steps=12000,
        sample_every=1000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
    )

    assert_all_finite(samples)

    final_extension = samples[-1]
    previous_extension = samples[-2]

    assert final_extension == pytest.approx(
        RELAXED_EXTENSION,
        rel=0.08,
        abs=0.01,
    )
    assert abs(
        final_extension - previous_extension
    ) < 0.01
    assert final_extension < RELAXED_EXTENSION * 1.15


def test_stress_relaxes_under_fixed_extension() -> None:
    """
    Under fixed strain, force must decay from K0*e toward K_inf*e.
    """

    imposed_extension = 0.10

    material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.40,
    )
    network, _, _, element = make_loaded_network(
        material=material,
        initial_length=(
            REST_LENGTH + imposed_extension
        ),
        fixed_free_node=True,
    )

    forces: list[float] = []

    for index in range(5000):
        network.step(
            dt=DEFAULT_DT,
            external_forces=None,
            update_materials=True,
        )

        if index in {
            0,
            100,
            1000,
            4999,
        }:
            forces.append(
                abs(float(element.last_force.total))
            )

    assert_all_finite(forces)

    initial_force = forces[0]
    final_force = forces[-1]

    assert final_force < initial_force
    assert initial_force == pytest.approx(
        INSTANTANEOUS_STIFFNESS
        * imposed_extension,
        rel=0.12,
    )
    assert final_force == pytest.approx(
        RELAXED_STIFFNESS
        * imposed_extension,
        rel=0.15,
    )


def test_creep_partially_recovers_after_unloading() -> None:
    """
    Removing load must recover delayed reversible strain.
    """

    material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.40,
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    run_steps(
        network,
        steps=5000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
    )
    loaded_extension = extension(free)

    run_steps(
        network,
        steps=5000,
        dt=DEFAULT_DT,
        external_forces=None,
    )
    recovered_extension = extension(free)

    assert loaded_extension > ELASTIC_EXTENSION
    assert recovered_extension < loaded_extension
    assert abs(recovered_extension) < (
        loaded_extension * 0.25
    )


def test_creep_state_is_explicit_and_finite() -> None:
    """
    The delayed strain must be represented in MaterialState.
    """

    material = make_material(
        rheology_enabled=True
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    run_steps(
        network,
        steps=2000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
    )

    state = material.state

    assert hasattr(state, "creep_strain")
    assert hasattr(state, "rheology_time")

    assert math.isfinite(
        float(state.creep_strain)
    )
    assert math.isfinite(
        float(state.rheology_time)
    )

    assert state.creep_strain > 0.0
    assert state.rheology_time > 0.0


def test_creep_state_is_in_material_snapshot() -> None:
    """
    Rheological memory must survive serialization boundaries.
    """

    material = make_material(
        rheology_enabled=True
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    run_steps(
        network,
        steps=1500,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
    )

    snapshot = material.snapshot()

    assert "state" in snapshot
    assert "parameters" in snapshot

    state_snapshot = snapshot["state"]
    parameter_snapshot = snapshot["parameters"]

    assert "creep_strain" in state_snapshot
    assert "rheology_time" in state_snapshot
    assert "relaxed_stiffness" in parameter_snapshot
    assert "creep_time_constant" in parameter_snapshot

    assert state_snapshot["creep_strain"] > 0.0


def test_creep_integration_is_time_step_stable() -> None:
    """
    Halving dt should not materially change the long-time result.
    """

    duration = 6.0

    coarse_dt = 0.002
    fine_dt = 0.001

    coarse_material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.50,
    )
    fine_material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.50,
    )

    coarse_network, _, coarse_node, _ = (
        make_loaded_network(
            material=coarse_material
        )
    )
    fine_network, _, fine_node, _ = (
        make_loaded_network(
            material=fine_material
        )
    )

    run_steps(
        coarse_network,
        steps=round(duration / coarse_dt),
        dt=coarse_dt,
        external_forces=constant_load_for(
            coarse_node
        ),
    )
    run_steps(
        fine_network,
        steps=round(duration / fine_dt),
        dt=fine_dt,
        external_forces=constant_load_for(
            fine_node
        ),
    )

    coarse_extension = extension(coarse_node)
    fine_extension = extension(fine_node)

    assert math.isfinite(coarse_extension)
    assert math.isfinite(fine_extension)

    assert relative_difference(
        coarse_extension,
        fine_extension,
    ) < 0.04


def test_creep_history_is_monotonic_after_transient() -> None:
    """
    After the inertial transient, sustained tensile load should not produce
    backward rheological evolution.
    """

    material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.50,
        damping=4.0,
    )
    network, _, free, _ = make_loaded_network(
        material=material
    )

    samples = sample_extensions(
        network,
        free,
        total_steps=7000,
        sample_every=250,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
    )

    assert_all_finite(samples)

    late_samples = samples[8:]

    assert len(late_samples) >= 5

    differences = np.diff(
        np.asarray(late_samples)
    )

    assert np.all(differences >= -1e-4)


def test_rheology_does_not_produce_nan_or_infinite_energy() -> None:
    """
    Rheological evolution must remain numerically finite.
    """

    material = make_material(
        rheology_enabled=True,
        creep_time_constant=0.20,
    )
    network, _, free, element = (
        make_loaded_network(
            material=material
        )
    )

    run_steps(
        network,
        steps=10000,
        dt=DEFAULT_DT,
        external_forces=constant_load_for(free),
    )

    values = (
        extension(free),
        float(element.last_force.elastic),
        float(element.last_force.damping),
        float(element.last_force.total),
        float(network.last_step_stats.kinetic_energy),
        float(network.last_step_stats.elastic_energy),
        float(network.last_step_stats.total_energy),
        float(material.state.creep_strain),
    )

    assert_all_finite(values)


# ---------------------------------------------------------------------------
# Parameter validation specification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_relaxed_stiffness",
    (
        -1.0,
        0.0,
        INSTANTANEOUS_STIFFNESS + 1.0,
        float("inf"),
        float("nan"),
    ),
)
def test_relaxed_stiffness_validation(
    bad_relaxed_stiffness: float,
) -> None:
    """
    SLS requires 0 < K_inf <= K0.
    """

    with pytest.raises(ValueError):
        make_parameters(
            relaxed_stiffness=(
                bad_relaxed_stiffness
            ),
            creep_time_constant=1.0,
            rheology_enabled=True,
        )


@pytest.mark.parametrize(
    "bad_time_constant",
    (
        -1.0,
        0.0,
        float("inf"),
        float("nan"),
    ),
)
def test_creep_time_constant_validation(
    bad_time_constant: float,
) -> None:
    """
    Enabled rheology requires a finite positive time constant.
    """

    with pytest.raises(ValueError):
        make_parameters(
            relaxed_stiffness=RELAXED_STIFFNESS,
            creep_time_constant=bad_time_constant,
            rheology_enabled=True,
        )
