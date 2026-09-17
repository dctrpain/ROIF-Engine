"""
Transition Order Mechanism Falsification
========================================

Purpose
-------
Separate three possible levels of order dependence under the same
initial state and the same event pair A/B:

1. ADDITIVE
   Pure additive reference. No propagation, no capacity clipping,
   no reserve modulation, no adaptive reconfiguration.

2. FIXED_NETWORK
   Production prestress redistribution on a fixed transfer network.
   Adaptive connection state is not allowed to reconfigure the
   transfer operator between events.

3. RECONFIGURING_NETWORK
   Full production SystemEvolution path:
       event
       -> adaptive connection update
       -> updated transfer network
       -> prestress redistribution
       -> next SystemImage

This experiment does not assume in advance that the fixed-network
system is commutative.

Pre-registered interpretation
-----------------------------
- ADDITIVE must be order-independent within tolerance. Otherwise the
  experimental procedure is invalid.

- FIXED_NETWORK may or may not be order-dependent.

- If RECONFIGURING_NETWORK shows more order dependence than
  FIXED_NETWORK, adaptive operator reconfiguration contributes
  additional trajectory separation in this registered system.

- If RECONFIGURING_NETWORK is approximately equal to FIXED_NETWORK,
  the hypothesis of a substantial additional reconfiguration
  contribution is not supported for this registered experiment.

- If RECONFIGURING_NETWORK shows less order dependence than
  FIXED_NETWORK, reconfiguration does not amplify order dependence
  in this registered experiment.

Scope
-----
Computational model only.
No biological validation.
No clinical validation.
No causal truth claim.
No claim that noncommutativity is unique to this architecture.
No claim that the full state is globally minimal.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

from experiments.roif_physical_history_integration_benchmark import (
    benchmark_config,
    build_source_image,
    event_a,
    event_b,
)

from roif.history.prestress_redistribution import (
    PrestressConnection,
    PrestressNodeState,
    redistribute_prestress,
)

from roif.history.system_evolution import (
    evolve_sequence,
)


BENCHMARK_VERSION = (
    "transition_order_mechanism_falsification_v1"
)

TOLERANCE = 1e-12


def _euclidean(left, right) -> float:
    left = tuple(float(value) for value in left)
    right = tuple(float(value) for value in right)

    if len(left) != len(right):
        raise ValueError("vector lengths differ")

    return sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(left, right)
        )
    )


def _prestress_vector(nodes) -> tuple[float, ...]:
    return tuple(
        float(node.prestress)
        for node in nodes
    )


def _connection_state_vector(connections) -> tuple[float, ...]:
    from roif.history.adaptive_connection import (
        effective_transfer_gain,
    )

    values = []

    for connection in connections:
        values.extend(
            (
                float(
                    effective_transfer_gain(
                        connection
                    )
                ),
                float(connection.contractile_capacity),
                float(connection.reflex_gain),
                float(connection.fatigue),
                float(connection.remodeling_bias),
            )
        )

    return tuple(values)


def _full_registered_state_vector(image) -> tuple[float, ...]:
    return (
        *_prestress_vector(image.prestress_nodes),
        *_connection_state_vector(
            image.adaptive_connections
        ),
    )


def _node_map(nodes):
    return {
        node.node_id: node
        for node in nodes
    }


def _perturbation_delta_by_node(event):
    result = {}

    for perturbation in event.prestress_perturbations:
        result[perturbation.node_id] = (
            result.get(
                perturbation.node_id,
                0.0,
            )
            + float(perturbation.delta)
        )

    return result


def _apply_additive_event(
    nodes,
    event,
) -> tuple[PrestressNodeState, ...]:
    """
    Pure additive reference.

    Deliberately excludes:
    - graph propagation,
    - connection gains,
    - capacity clipping,
    - reserve modulation,
    - adaptive connection updates.

    Node bounds are not invoked here because this is the algebraic
    additive control, not the production prestress solver.
    """

    delta_by_node = _perturbation_delta_by_node(
        event
    )

    return tuple(
        PrestressNodeState(
            node_id=node.node_id,
            prestress=(
                float(node.prestress)
                + delta_by_node.get(
                    node.node_id,
                    0.0,
                )
            ),
            min_prestress=node.min_prestress,
            max_prestress=node.max_prestress,
            reserve=node.reserve,
            metadata=dict(node.metadata),
        )
        for node in nodes
    )


def _run_additive_sequence(events):
    source = build_source_image()
    current = tuple(source.prestress_nodes)

    for event in events:
        current = _apply_additive_event(
            current,
            event,
        )

    return current


def _fixed_prestress_connections(
    source_image,
) -> tuple[PrestressConnection, ...]:
    """
    Freeze the transfer network at its initial state.

    This reproduces the production bridge from adaptive connection
    state to PrestressConnection, but it is constructed once from X0
    and reused for every subsequent event.
    """

    from roif.history.adaptive_connection import (
        effective_transfer_gain,
    )

    return tuple(
        PrestressConnection(
            connection_id=(
                f"fixed::{connection.connection_id}"
            ),
            source_node_id=(
                connection.source_node_id
            ),
            target_node_id=(
                connection.target_node_id
            ),
            transfer_gain=(
                effective_transfer_gain(
                    connection
                )
            ),
            attenuation=1.0,
            capacity=max(
                0.0,
                connection.contractile_capacity,
            ),
            enabled=True,
            metadata={
                "mode": "fixed_initial_network",
                "source_connection_id": (
                    connection.connection_id
                ),
            },
        )
        for connection
        in source_image.adaptive_connections
    )


def _run_fixed_network_sequence(events):
    """
    Run the production prestress redistribution algorithm while
    holding the transfer network fixed at its initial configuration.

    Prestress itself remains recursive:
        P_(t+1) receives P_t.

    Only adaptive operator reconfiguration is removed.
    """

    source = build_source_image()
    config = benchmark_config()

    fixed_connections = (
        _fixed_prestress_connections(source)
    )

    current_nodes = tuple(
        source.prestress_nodes
    )

    results = []

    for index, event in enumerate(
        events,
        start=1,
    ):
        result = redistribute_prestress(
            redistribution_id=(
                f"fixed::step_{index}"
                f"::{event.event_id}"
            ),
            nodes=current_nodes,
            connections=fixed_connections,
            perturbations=(
                event.prestress_perturbations
            ),
            config=config.prestress_config,
            metadata={
                "experimental_mode": (
                    "fixed_initial_network"
                ),
                "adaptive_reconfiguration": False,
            },
        )

        results.append(result)

        current_nodes = tuple(
            result.target_states
        )

    return tuple(results)


def _run_reconfiguring_sequence(events):
    source = build_source_image()

    return evolve_sequence(
        sequence_id="reconfiguring",
        source_image=source,
        events=events,
        config=benchmark_config(),
    )


def _node_values(nodes):
    return {
        node.node_id: float(node.prestress)
        for node in nodes
    }


def _node_differences(left_nodes, right_nodes):
    left = _node_values(left_nodes)
    right = _node_values(right_nodes)

    ids = sorted(
        set(left)
        | set(right)
    )

    return {
        node_id: (
            left.get(node_id, 0.0)
            - right.get(node_id, 0.0)
        )
        for node_id in ids
    }


def run_additive():
    ab = _run_additive_sequence(
        (
            event_a(),
            event_b(),
        )
    )

    ba = _run_additive_sequence(
        (
            event_b(),
            event_a(),
        )
    )

    distance = _euclidean(
        _prestress_vector(ab),
        _prestress_vector(ba),
    )

    return {
        "ab_final_prestress": _node_values(ab),
        "ba_final_prestress": _node_values(ba),
        "ab_minus_ba_by_node": (
            _node_differences(ab, ba)
        ),
        "prestress_distance": distance,
        "order_dependent": (
            distance > TOLERANCE
        ),
    }


def run_fixed_network():
    ab_results = (
        _run_fixed_network_sequence(
            (
                event_a(),
                event_b(),
            )
        )
    )

    ba_results = (
        _run_fixed_network_sequence(
            (
                event_b(),
                event_a(),
            )
        )
    )

    ab_final = (
        ab_results[-1].target_states
    )

    ba_final = (
        ba_results[-1].target_states
    )

    distance = _euclidean(
        _prestress_vector(ab_final),
        _prestress_vector(ba_final),
    )

    return {
        "ab_final_prestress": (
            _node_values(ab_final)
        ),
        "ba_final_prestress": (
            _node_values(ba_final)
        ),
        "ab_minus_ba_by_node": (
            _node_differences(
                ab_final,
                ba_final,
            )
        ),
        "prestress_distance": distance,
        "order_dependent": (
            distance > TOLERANCE
        ),
    }


def run_reconfiguring_network():
    ab_results = (
        _run_reconfiguring_sequence(
            (
                event_a(),
                event_b(),
            )
        )
    )

    ba_results = (
        _run_reconfiguring_sequence(
            (
                event_b(),
                event_a(),
            )
        )
    )

    ab_final = (
        ab_results[-1].target_image
    )

    ba_final = (
        ba_results[-1].target_image
    )

    prestress_distance = _euclidean(
        _prestress_vector(
            ab_final.prestress_nodes
        ),
        _prestress_vector(
            ba_final.prestress_nodes
        ),
    )

    connection_distance = _euclidean(
        _connection_state_vector(
            ab_final.adaptive_connections
        ),
        _connection_state_vector(
            ba_final.adaptive_connections
        ),
    )

    full_distance = _euclidean(
        _full_registered_state_vector(
            ab_final
        ),
        _full_registered_state_vector(
            ba_final
        ),
    )

    return {
        "ab_final_prestress": (
            _node_values(
                ab_final.prestress_nodes
            )
        ),
        "ba_final_prestress": (
            _node_values(
                ba_final.prestress_nodes
            )
        ),
        "ab_minus_ba_by_node": (
            _node_differences(
                ab_final.prestress_nodes,
                ba_final.prestress_nodes,
            )
        ),
        "prestress_distance": (
            prestress_distance
        ),
        "connection_state_distance": (
            connection_distance
        ),
        "registered_state_distance": (
            full_distance
        ),
        "order_dependent_prestress": (
            prestress_distance > TOLERANCE
        ),
        "order_dependent_connections": (
            connection_distance > TOLERANCE
        ),
        "order_dependent_registered_state": (
            full_distance > TOLERANCE
        ),
    }


def _classify(
    additive,
    fixed,
    reconfiguring,
):
    additive_distance = (
        additive["prestress_distance"]
    )

    fixed_distance = (
        fixed["prestress_distance"]
    )

    reconfiguring_distance = (
        reconfiguring["prestress_distance"]
    )

    additive_control_valid = (
        additive_distance <= TOLERANCE
    )

    if not additive_control_valid:
        conclusion = (
            "INVALID_EXPERIMENT:"
            "_ADDITIVE_CONTROL_ORDER_DEPENDENT"
        )

    elif (
        fixed_distance <= TOLERANCE
        and reconfiguring_distance > TOLERANCE
    ):
        conclusion = (
            "RECONFIGURATION_INTRODUCES_"
            "REGISTERED_ORDER_DEPENDENCE"
        )

    elif (
        fixed_distance > TOLERANCE
        and reconfiguring_distance
        > fixed_distance + TOLERANCE
    ):
        conclusion = (
            "FIXED_REDISTRIBUTION_IS_ORDER_DEPENDENT_"
            "AND_RECONFIGURATION_INCREASES_SEPARATION"
        )

    elif (
        fixed_distance > TOLERANCE
        and abs(
            reconfiguring_distance
            - fixed_distance
        ) <= TOLERANCE
    ):
        conclusion = (
            "NO_DETECTABLE_ADDITIONAL_"
            "RECONFIGURATION_CONTRIBUTION"
        )

    elif (
        fixed_distance > TOLERANCE
        and reconfiguring_distance
        < fixed_distance - TOLERANCE
    ):
        conclusion = (
            "RECONFIGURATION_REDUCES_"
            "REGISTERED_ORDER_SEPARATION"
        )

    elif (
        fixed_distance <= TOLERANCE
        and reconfiguring_distance <= TOLERANCE
    ):
        conclusion = (
            "NO_REGISTERED_PRESTRESS_"
            "ORDER_DEPENDENCE"
        )

    else:
        conclusion = (
            "ORDER_DEPENDENCE_PRESENT_"
            "WITHOUT_SIMPLE_AMPLIFICATION_CLASSIFICATION"
        )

    return {
        "tolerance": TOLERANCE,
        "additive_control_valid": (
            additive_control_valid
        ),
        "fixed_network_order_dependent": (
            fixed_distance > TOLERANCE
        ),
        "reconfiguring_network_order_dependent": (
            reconfiguring_distance > TOLERANCE
        ),
        "fixed_prestress_distance": (
            fixed_distance
        ),
        "reconfiguring_prestress_distance": (
            reconfiguring_distance
        ),
        "reconfiguration_minus_fixed_distance": (
            reconfiguring_distance
            - fixed_distance
        ),
        "registered_conclusion": conclusion,
    }


def run_benchmark():
    additive = run_additive()
    fixed = run_fixed_network()
    reconfiguring = (
        run_reconfiguring_network()
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": (
            "computational_registered_system_only"
        ),
        "preregistered": {
            "initial_state": (
                "build_source_image()"
            ),
            "events": [
                "A",
                "B",
            ],
            "sequences": [
                "AB",
                "BA",
            ],
            "tolerance": TOLERANCE,
            "primary_readout": (
                "euclidean_distance_final_prestress"
            ),
            "secondary_readouts": [
                "nodewise_final_prestress_difference",
                "adaptive_connection_state_distance",
                "registered_state_distance",
            ],
            "defeat_conditions": {
                "procedure_invalid_if": (
                    "additive_AB_BA_distance"
                    "_exceeds_tolerance"
                ),
                "reconfiguration_amplification_"
                "hypothesis_defeated_if": (
                    "reconfiguring_prestress_distance"
                    "_is_not_greater_than_"
                    "fixed_network_prestress_distance"
                    "_beyond_tolerance"
                ),
            },
        },
        "additive_reference": additive,
        "fixed_network": fixed,
        "reconfiguring_network": reconfiguring,
        "decision": _classify(
            additive,
            fixed,
            reconfiguring,
        ),
        "claims_not_tested": [
            "biological_validity",
            "clinical_validity",
            "causal_truth_in_nature",
            "global_minimality_of_full_state",
            "uniqueness_of_noncommutativity",
            "chaos",
            "calibrated_probability",
        ],
    }


def main():
    data = run_benchmark()

    output_dir = Path(
        "benchmark_results"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "transition_order_mechanism_falsification.json"
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "\nSaved:",
        output_path,
    )


if __name__ == "__main__":
    main()
