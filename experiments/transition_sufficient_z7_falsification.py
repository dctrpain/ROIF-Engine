"""
Transition-Sufficient Z7 Falsification Benchmark
================================================

Purpose
-------
Test whether the registered seven-dimensional interface representation

    Z7 = (
        P_A,
        P_B,
        P_C,
        g_AB,
        c_AB,
        g_BC,
        c_BC,
    )

is transition-sufficient for the registered production dynamics.

Here:

    P_*   = node prestress
    g_*   = effective_transfer_gain(connection)
    c_*   = connection.contractile_capacity

The full registered dynamic state contains:

    3 prestress coordinates
    +
    2 * 5 adaptive-connection coordinates

for a total of 13 scalar dynamic coordinates.

The Z7 representation intentionally preserves the quantities directly
exported by the adaptive-connection layer into the prestress-transfer
layer, while discarding the internal decomposition of effective transfer.

Scientific question
-------------------
Can two distinct full states that collide under Z7 remain dynamically
equivalent under the registered event family?

Null hypothesis
---------------
For the registered domain, if

    Z7(I_A) == Z7(I_B),

then identical admissible future event sequences produce equivalent
registered outputs within tolerance.

Falsification logic
-------------------
1. Construct two distinct full states with identical Z7.
2. Verify the collision explicitly.
3. Apply an identical direct prestress probe without adaptive exposure.
   This is the positive reduction control: the current interface state
   should be sufficient for the immediate prestress redistribution.
4. Apply an identical adaptive WRITE event.
5. Test whether the two systems still have identical Z7.
6. Apply an identical prestress READ probe.
7. Test whether externally visible prestress behavior diverges.

Interpretation boundary
-----------------------
Failure of Z7 does NOT prove:
- that the full 13-dimensional state is globally minimal;
- that no other reduced representation exists;
- that history itself must be stored explicitly;
- biological or clinical validity;
- empirical validity outside the registered computational system.

Success of Z7 on the direct prestress control and failure after adaptive
conditioning would establish only that the current prestress-transfer
interface is sufficient for the restricted direct-probe domain but is not
closed under the registered adaptive transition.

Production modules are imported but not modified.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import Iterable, Sequence

from roif.history.adaptive_connection import (
    AdaptiveConnectionConfig,
    AdaptiveConnectionState,
    ConnectionExposure,
    effective_transfer_gain,
)
from roif.history.prestress_redistribution import (
    PrestressNodeState,
    PrestressPerturbation,
)
from roif.history.system_evolution import (
    SystemEvent,
    SystemEvolutionConfig,
    evolve_system,
)
from roif.history.system_image import (
    SystemImage,
    SystemImageContext,
    build_system_image,
)


BENCHMARK_VERSION = "transition_sufficient_z7_falsification_v1"

TOLERANCE = 1e-12

NODE_IDS = ("A", "B", "C")
CONNECTION_IDS = ("A_B", "B_C")

RESULTS_DIR = Path("benchmark_results")
RESULT_PATH = RESULTS_DIR / "transition_sufficient_z7_falsification.json"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _euclidean(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "vectors must have the same length"
        )

    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right)
        )
    )


def _max_abs_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "vectors must have the same length"
        )

    if not left:
        return 0.0

    return max(
        abs(float(a) - float(b))
        for a, b in zip(left, right)
    )


def _close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = TOLERANCE,
) -> bool:
    return (
        _max_abs_difference(
            left,
            right,
        )
        <= tolerance
    )


# ---------------------------------------------------------------------------
# Registered state extraction
# ---------------------------------------------------------------------------


def _prestress_vector(
    image: SystemImage,
) -> tuple[float, ...]:
    by_id = {
        node.node_id: node
        for node in image.prestress_nodes
    }

    return tuple(
        float(
            by_id[node_id].prestress
        )
        for node_id in NODE_IDS
    )


def _adaptive_full_vector(
    image: SystemImage,
) -> tuple[float, ...]:
    by_id = {
        connection.connection_id: connection
        for connection in image.adaptive_connections
    }

    values: list[float] = []

    for connection_id in CONNECTION_IDS:
        connection = by_id[connection_id]

        values.extend(
            (
                float(connection.stiffness),
                float(connection.contractile_capacity),
                float(connection.reflex_gain),
                float(connection.fatigue),
                float(connection.remodeling_bias),
            )
        )

    return tuple(values)


def full_state_vector(
    image: SystemImage,
) -> tuple[float, ...]:
    """
    Registered 13-dimensional dynamic state.

    I = (
        P_A, P_B, P_C,
        s_AB, c_AB, r_AB, f_AB, b_AB,
        s_BC, c_BC, r_BC, f_BC, b_BC,
    )
    """
    return (
        *_prestress_vector(image),
        *_adaptive_full_vector(image),
    )


def z7_vector(
    image: SystemImage,
) -> tuple[float, ...]:
    """
    Registered seven-dimensional interface reduction.

    Z7 = (
        P_A,
        P_B,
        P_C,
        g_AB,
        c_AB,
        g_BC,
        c_BC,
    )
    """
    by_id = {
        connection.connection_id: connection
        for connection in image.adaptive_connections
    }

    ab = by_id["A_B"]
    bc = by_id["B_C"]

    return (
        *_prestress_vector(image),
        float(effective_transfer_gain(ab)),
        float(ab.contractile_capacity),
        float(effective_transfer_gain(bc)),
        float(bc.contractile_capacity),
    )


def _connection_state_dict(
    image: SystemImage,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for connection in image.adaptive_connections:
        if connection.connection_id not in CONNECTION_IDS:
            continue

        result[connection.connection_id] = {
            "stiffness": float(connection.stiffness),
            "contractile_capacity": float(
                connection.contractile_capacity
            ),
            "reflex_gain": float(connection.reflex_gain),
            "fatigue": float(connection.fatigue),
            "remodeling_bias": float(
                connection.remodeling_bias
            ),
            "effective_transfer_gain": float(
                effective_transfer_gain(connection)
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Registered source states
# ---------------------------------------------------------------------------


def _prestress_nodes() -> tuple[
    PrestressNodeState,
    ...,
]:
    return (
        PrestressNodeState(
            node_id="A",
            prestress=0.20,
            min_prestress=-1.0,
            max_prestress=1.0,
            reserve=1.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=0.10,
            min_prestress=-1.0,
            max_prestress=1.0,
            reserve=1.0,
        ),
        PrestressNodeState(
            node_id="C",
            prestress=0.00,
            min_prestress=-1.0,
            max_prestress=1.0,
            reserve=1.0,
        ),
    )


def _connection_ab_variant_a() -> AdaptiveConnectionState:
    """
    Variant A for A_B.

    g_eff = 1 * 1 * 1 * (1 - 0) = 1
    capacity = 1
    """
    return AdaptiveConnectionState(
        connection_id="A_B",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=1.0,
        reflex_gain=1.0,
        fatigue=0.0,
        remodeling_bias=0.0,
        min_stiffness=0.0,
        max_stiffness=10.0,
        min_contractile_capacity=0.0,
        max_contractile_capacity=10.0,
        min_reflex_gain=0.0,
        max_reflex_gain=10.0,
    )


def _connection_ab_variant_b() -> AdaptiveConnectionState:
    """
    Variant B for A_B.

    Different full state, but:

        g_eff = 1.25 * 1 * 0.8 * (1 - 0) = 1
        capacity = 1

    Therefore A and B collide under the registered Z7 projection.
    """
    return AdaptiveConnectionState(
        connection_id="A_B",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.25,
        contractile_capacity=1.0,
        reflex_gain=0.8,
        fatigue=0.0,
        remodeling_bias=0.0,
        min_stiffness=0.0,
        max_stiffness=10.0,
        min_contractile_capacity=0.0,
        max_contractile_capacity=10.0,
        min_reflex_gain=0.0,
        max_reflex_gain=10.0,
    )


def _connection_bc_common() -> AdaptiveConnectionState:
    return AdaptiveConnectionState(
        connection_id="B_C",
        source_node_id="B",
        target_node_id="C",
        stiffness=1.0,
        contractile_capacity=1.0,
        reflex_gain=1.0,
        fatigue=0.0,
        remodeling_bias=0.0,
        min_stiffness=0.0,
        max_stiffness=10.0,
        min_contractile_capacity=0.0,
        max_contractile_capacity=10.0,
        min_reflex_gain=0.0,
        max_reflex_gain=10.0,
    )


def build_source_image(
    *,
    variant: str,
) -> SystemImage:
    if variant == "A":
        ab = _connection_ab_variant_a()
    elif variant == "B":
        ab = _connection_ab_variant_b()
    else:
        raise ValueError(
            f"unknown variant: {variant!r}"
        )

    return build_system_image(
        image_id=f"z7_source_{variant}",
        revision=0,
        measures=(),
        prestress_nodes=_prestress_nodes(),
        adaptive_connections=(
            ab,
            _connection_bc_common(),
        ),
        historical_traces=(),
        context=SystemImageContext(
            context_id=f"z7_context_{variant}",
            timestamp_label="t0",
        ),
        metadata={
            "benchmark_version": BENCHMARK_VERSION,
            "variant": variant,
            "purpose": (
                "registered_z7_collision_source"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Registered production configuration
# ---------------------------------------------------------------------------


def _evolution_config() -> SystemEvolutionConfig:
    """
    Use the production SystemEvolutionConfig defaults.

    AdaptiveConnectionConfig is made explicit here so the benchmark records
    the coefficients governing the WRITE transition.
    """
    return SystemEvolutionConfig(
        adaptive_config=AdaptiveConnectionConfig(),
    )


# ---------------------------------------------------------------------------
# Registered events
# ---------------------------------------------------------------------------


def direct_probe_event(
    *,
    event_id: str,
) -> SystemEvent:
    """
    Direct mechanical probe.

    No adaptive exposure is present.

    This branch tests whether the current Z7 interface representation is
    sufficient for immediate prestress redistribution.
    """
    return SystemEvent(
        event_id=event_id,
        event_type="z7_direct_prestress_probe",
        connection_exposures=(),
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id=f"{event_id}::P_A",
                node_id="A",
                delta=-0.25,
            ),
        ),
        metadata={
            "benchmark_version": BENCHMARK_VERSION,
            "role": "direct_positive_control",
        },
    )


def write_event(
    *,
    event_id: str,
) -> SystemEvent:
    """
    Adaptive WRITE.

    The exposure is identical for both collided source states.

    It acts on A_B and is chosen from the already registered production
    event family used by the physical-history benchmark.
    """
    return SystemEvent(
        event_id=event_id,
        event_type="z7_adaptive_write",
        connection_exposures=(
            ConnectionExposure(
                exposure_id=f"{event_id}::A_B",
                connection_id="A_B",
                load=0.7,
                strain=0.3,
                activation=0.8,
                damage=0.45,
                recovery=0.0,
            ),
        ),
        prestress_perturbations=(),
        metadata={
            "benchmark_version": BENCHMARK_VERSION,
            "role": "adaptive_write",
        },
    )


def read_event(
    *,
    event_id: str,
) -> SystemEvent:
    """
    Mechanical READ after adaptive conditioning.

    No adaptive exposure is applied during READ.
    """
    return SystemEvent(
        event_id=event_id,
        event_type="z7_future_prestress_read",
        connection_exposures=(),
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id=f"{event_id}::P_A",
                node_id="A",
                delta=-0.25,
            ),
        ),
        metadata={
            "benchmark_version": BENCHMARK_VERSION,
            "role": "future_read",
        },
    )


# ---------------------------------------------------------------------------
# Production transition wrapper
# ---------------------------------------------------------------------------


def _evolve(
    *,
    source: SystemImage,
    event: SystemEvent,
    target_image_id: str,
) -> SystemImage:
    result = evolve_system(
        evolution_id=(
            f"evolution::{target_image_id}"
        ),
        source_image=source,
        event=event,
        target_image_id=target_image_id,
        config=_evolution_config(),
    )

    return result.target_image


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, object]:
    source_a = build_source_image(
        variant="A",
    )
    source_b = build_source_image(
        variant="B",
    )

    full_a_t0 = full_state_vector(source_a)
    full_b_t0 = full_state_vector(source_b)

    z7_a_t0 = z7_vector(source_a)
    z7_b_t0 = z7_vector(source_b)

    initial_full_state_distance = _euclidean(
        full_a_t0,
        full_b_t0,
    )

    initial_z7_distance = _euclidean(
        z7_a_t0,
        z7_b_t0,
    )

    initial_full_states_distinct = (
        initial_full_state_distance
        > TOLERANCE
    )

    initial_z7_collision = (
        initial_z7_distance
        <= TOLERANCE
    )

    # ---------------------------------------------------------------
    # Positive reduction control:
    # same Z7 -> same immediate direct prestress response
    # ---------------------------------------------------------------

    direct_a = _evolve(
        source=source_a,
        event=direct_probe_event(
            event_id="direct_probe_A",
        ),
        target_image_id="z7_direct_A",
    )

    direct_b = _evolve(
        source=source_b,
        event=direct_probe_event(
            event_id="direct_probe_B",
        ),
        target_image_id="z7_direct_B",
    )

    direct_prestress_a = _prestress_vector(
        direct_a
    )
    direct_prestress_b = _prestress_vector(
        direct_b
    )

    direct_prestress_distance = _euclidean(
        direct_prestress_a,
        direct_prestress_b,
    )

    direct_control_equivalent = (
        direct_prestress_distance
        <= TOLERANCE
    )

    # ---------------------------------------------------------------
    # Adaptive WRITE:
    # same event applied to collided Z7 states
    # ---------------------------------------------------------------

    write_a = _evolve(
        source=source_a,
        event=write_event(
            event_id="write_A",
        ),
        target_image_id="z7_write_A",
    )

    write_b = _evolve(
        source=source_b,
        event=write_event(
            event_id="write_B",
        ),
        target_image_id="z7_write_B",
    )

    write_prestress_a = _prestress_vector(
        write_a
    )
    write_prestress_b = _prestress_vector(
        write_b
    )

    write_prestress_distance = _euclidean(
        write_prestress_a,
        write_prestress_b,
    )

    z7_a_after_write = z7_vector(write_a)
    z7_b_after_write = z7_vector(write_b)

    z7_after_write_distance = _euclidean(
        z7_a_after_write,
        z7_b_after_write,
    )

    z7_closed_under_write = (
        z7_after_write_distance
        <= TOLERANCE
    )

    # ---------------------------------------------------------------
    # Future READ:
    # identical mechanical probe after identical adaptive WRITE
    # ---------------------------------------------------------------

    read_a = _evolve(
        source=write_a,
        event=read_event(
            event_id="read_A",
        ),
        target_image_id="z7_read_A",
    )

    read_b = _evolve(
        source=write_b,
        event=read_event(
            event_id="read_B",
        ),
        target_image_id="z7_read_B",
    )

    read_prestress_a = _prestress_vector(
        read_a
    )
    read_prestress_b = _prestress_vector(
        read_b
    )

    future_read_distance = _euclidean(
        read_prestress_a,
        read_prestress_b,
    )

    future_read_discriminates = (
        future_read_distance
        > TOLERANCE
    )

    # ---------------------------------------------------------------
    # Negative control:
    # each state compared with an independent repeat of itself
    # ---------------------------------------------------------------

    control_write_a = _evolve(
        source=source_a,
        event=write_event(
            event_id="control_write_A",
        ),
        target_image_id="z7_control_write_A",
    )

    control_read_a = _evolve(
        source=control_write_a,
        event=read_event(
            event_id="control_read_A",
        ),
        target_image_id="z7_control_read_A",
    )

    negative_control_a_distance = _euclidean(
        read_prestress_a,
        _prestress_vector(control_read_a),
    )

    control_write_b = _evolve(
        source=source_b,
        event=write_event(
            event_id="control_write_B",
        ),
        target_image_id="z7_control_write_B",
    )

    control_read_b = _evolve(
        source=control_write_b,
        event=read_event(
            event_id="control_read_B",
        ),
        target_image_id="z7_control_read_B",
    )

    negative_control_b_distance = _euclidean(
        read_prestress_b,
        _prestress_vector(control_read_b),
    )

    negative_controls_valid = (
        negative_control_a_distance
        <= TOLERANCE
        and negative_control_b_distance
        <= TOLERANCE
    )

    # ---------------------------------------------------------------
    # Registered decision
    # ---------------------------------------------------------------

    benchmark_valid = (
        initial_full_states_distinct
        and initial_z7_collision
        and direct_control_equivalent
        and negative_controls_valid
    )

    z7_falsified = (
        benchmark_valid
        and not z7_closed_under_write
        and future_read_discriminates
    )

    if not initial_full_states_distinct:
        decision = (
            "INVALID_SOURCE_STATES_NOT_DISTINCT"
        )
    elif not initial_z7_collision:
        decision = (
            "INVALID_NO_INITIAL_Z7_COLLISION"
        )
    elif not direct_control_equivalent:
        decision = (
            "INVALID_DIRECT_INTERFACE_CONTROL_FAILED"
        )
    elif not negative_controls_valid:
        decision = (
            "INVALID_NEGATIVE_CONTROL_FAILED"
        )
    elif z7_falsified:
        decision = (
            "Z7_NOT_TRANSITION_SUFFICIENT_OVER_REGISTERED_DOMAIN"
        )
    elif z7_closed_under_write and not future_read_discriminates:
        decision = (
            "Z7_SURVIVES_REGISTERED_TEST"
        )
    elif not z7_closed_under_write:
        decision = (
            "Z7_NOT_CLOSED_UNDER_WRITE_BUT_EXTERNAL_READOUT_NOT_DISCRIMINATING"
        )
    else:
        decision = (
            "INCONCLUSIVE_REGISTERED_RESULT"
        )

    result: dict[str, object] = {
        "benchmark": (
            "transition_sufficient_z7_falsification"
        ),
        "benchmark_version": BENCHMARK_VERSION,
        "tolerance": TOLERANCE,
        "registered_state_dimension": 13,
        "registered_reduction_dimension": 7,
        "registered_reduction": [
            "P_A",
            "P_B",
            "P_C",
            "g_AB",
            "c_AB",
            "g_BC",
            "c_BC",
        ],
        "initial_state": {
            "full_state_A": list(full_a_t0),
            "full_state_B": list(full_b_t0),
            "full_state_distance": (
                initial_full_state_distance
            ),
            "full_states_distinct": (
                initial_full_states_distinct
            ),
            "z7_A": list(z7_a_t0),
            "z7_B": list(z7_b_t0),
            "z7_distance": initial_z7_distance,
            "z7_collision": initial_z7_collision,
            "connections_A": _connection_state_dict(
                source_a
            ),
            "connections_B": _connection_state_dict(
                source_b
            ),
        },
        "direct_positive_control": {
            "prestress_A": list(
                direct_prestress_a
            ),
            "prestress_B": list(
                direct_prestress_b
            ),
            "distance": (
                direct_prestress_distance
            ),
            "equivalent": (
                direct_control_equivalent
            ),
        },
        "after_identical_write": {
            "prestress_A": list(
                write_prestress_a
            ),
            "prestress_B": list(
                write_prestress_b
            ),
            "prestress_distance": (
                write_prestress_distance
            ),
            "z7_A": list(
                z7_a_after_write
            ),
            "z7_B": list(
                z7_b_after_write
            ),
            "z7_distance": (
                z7_after_write_distance
            ),
            "z7_closed_under_write": (
                z7_closed_under_write
            ),
            "connections_A": (
                _connection_state_dict(
                    write_a
                )
            ),
            "connections_B": (
                _connection_state_dict(
                    write_b
                )
            ),
        },
        "future_identical_read": {
            "prestress_A": list(
                read_prestress_a
            ),
            "prestress_B": list(
                read_prestress_b
            ),
            "distance": (
                future_read_distance
            ),
            "discriminates": (
                future_read_discriminates
            ),
        },
        "negative_controls": {
            "A_repeat_distance": (
                negative_control_a_distance
            ),
            "B_repeat_distance": (
                negative_control_b_distance
            ),
            "valid": negative_controls_valid,
        },
        "registered_decision": {
            "benchmark_valid": benchmark_valid,
            "z7_falsified": z7_falsified,
            "decision": decision,
        },
        "claim_scope": {
            "full_state_globally_minimal_proved": False,
            "all_possible_reductions_falsified": False,
            "explicit_history_required_proved": False,
            "biological_validity_tested": False,
            "clinical_validity_tested": False,
            "registered_claim": (
                "The benchmark tests only whether the "
                "registered Z7 interface reduction is "
                "transition-sufficient over the registered "
                "production event sequence."
            ),
        },
    }

    return result


def save_result(
    result: dict[str, object],
) -> Path:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return RESULT_PATH


def main() -> None:
    result = run_benchmark()
    path = save_result(result)

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        f"Saved: {path}"
    )


if __name__ == "__main__":
    main()
