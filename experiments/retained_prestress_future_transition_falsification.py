"""
Retained Prestress Future-Transition Falsification
==================================================

Purpose
-------
Test whether distributed prestress retained after two different event
orders remains relevant to a subsequent identical physical transition
when the registered adaptive connection states are equivalent.

This benchmark is governed by the preregistration:

    experiments/
    retained_prestress_future_transition_preregistration.json

No probe selection or tolerance modification is permitted after observing
the result.

Registered construction
-----------------------
Starting from the same source image:

    AB = event_a -> event_b
    BA = event_b -> event_a

using the full production reconfiguring SystemEvolution path.

Required terminal condition:

    C_AB == C_BA       within tolerance
    P_AB != P_BA       beyond tolerance

Both terminal SystemImages then receive the same previously locked READ:

    node A
    delta = +0.30
    no connection exposure

Primary test:

    distance(P_ABR, P_BAR) > tolerance

Scope
-----
Computational registered system only.

This experiment does not establish:
- global minimality,
- biological validity,
- clinical validity,
- necessity of prestress in all dynamical systems,
- novelty of mechanical memory.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

from experiments.transition_order_mechanism_falsification import (
    TOLERANCE,
    _connection_state_vector,
    _prestress_vector,
    benchmark_config,
    build_source_image,
    event_a,
    event_b,
)


from roif.history.system_evolution import (
    SystemEvent,
    evolve_sequence,
    evolve_system,
)
from roif.history.prestress_redistribution import (
    PrestressPerturbation,
)

from roif.history.system_image import (
    SystemImageContext,
)


BENCHMARK_VERSION = (
    "retained_prestress_future_transition_falsification_v1"
)

PREREGISTRATION_PATH = Path(
    "experiments/"
    "retained_prestress_future_transition_preregistration.json"
)

OUTPUT_PATH = Path(
    "benchmark_results/"
    "retained_prestress_future_transition_falsification.json"
)


def read_event(event_id: str):
    return SystemEvent(
        event_id=event_id,
        event_type="shared_physical_probe",
        connection_exposures=(),
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id=f"{event_id}_perturbation",
                node_id="A",
                delta=0.30,
            ),
        ),
        target_context=SystemImageContext(
            context_id=f"{event_id}_ctx",
            timestamp_label="read",
        ),
    )

def _euclidean(left, right) -> float:
    if len(left) != len(right):
        raise ValueError(
            "Cannot compare vectors with different dimensions."
        )

    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right)
        )
    )


def _node_values(nodes):
    return {
        node.node_id: float(node.prestress)
        for node in nodes
    }


def _connection_values(connections):
    return {
        connection.connection_id: {
            "stiffness": float(connection.stiffness),
            "contractile_capacity": float(
                connection.contractile_capacity
            ),
            "reflex_gain": float(connection.reflex_gain),
            "fatigue": float(connection.fatigue),
            "remodeling_bias": float(
                connection.remodeling_bias
            ),
        }
        for connection in connections
    }


def _run_registered_sequence(
    sequence_id: str,
    events,
):
    return evolve_sequence(
        sequence_id=sequence_id,
        source_image=build_source_image(),
        events=events,
        config=benchmark_config(),
    )


def _build_terminal_states():
    ab_results = _run_registered_sequence(
        sequence_id="AB",
        events=(
            event_a(),
            event_b(),
        ),
    )

    ba_results = _run_registered_sequence(
        sequence_id="BA",
        events=(
            event_b(),
            event_a(),
        ),
    )

    if not ab_results or not ba_results:
        raise RuntimeError(
            "Registered AB/BA sequence returned no results."
        )

    return (
        ab_results[-1].target_image,
        ba_results[-1].target_image,
    )


def _validate_locked_read(event) -> dict:
    perturbations = tuple(
        event.prestress_perturbations
    )

    exposures = tuple(
        event.connection_exposures
    )

    exact_registered_read = (
        event.event_type == "shared_physical_probe"
        and len(exposures) == 0
        and len(perturbations) == 1
        and perturbations[0].node_id == "A"
        and abs(
            float(perturbations[0].delta) - 0.30
        ) <= TOLERANCE
    )

    return {
        "event_type": event.event_type,
        "connection_exposure_count": len(exposures),
        "prestress_perturbation_count": len(
            perturbations
        ),
        "perturbations": [
            {
                "node_id": perturbation.node_id,
                "delta": float(perturbation.delta),
            }
            for perturbation in perturbations
        ],
        "matches_preregistered_read": (
            exact_registered_read
        ),
    }


def run_benchmark() -> dict:
    if not PREREGISTRATION_PATH.exists():
        raise RuntimeError(
            "Preregistration file is missing: "
            f"{PREREGISTRATION_PATH}"
        )

    preregistration = json.loads(
        PREREGISTRATION_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    ab_final, ba_final = _build_terminal_states()

    ab_connections = _connection_state_vector(
        ab_final.adaptive_connections
    )
    ba_connections = _connection_state_vector(
        ba_final.adaptive_connections
    )

    ab_prestress = _prestress_vector(
        ab_final.prestress_nodes
    )
    ba_prestress = _prestress_vector(
        ba_final.prestress_nodes
    )

    terminal_connection_distance = _euclidean(
        ab_connections,
        ba_connections,
    )

    terminal_prestress_distance = _euclidean(
        ab_prestress,
        ba_prestress,
    )

    registered_read = read_event(
        "retained_prestress_registered_read"
    )

    read_validation = _validate_locked_read(
        registered_read
    )

    connection_equivalent = (
        terminal_connection_distance
        <= TOLERANCE
    )

    prestress_distinguishable = (
        terminal_prestress_distance
        > TOLERANCE
    )

    read_valid = bool(
        read_validation[
            "matches_preregistered_read"
        ]
    )

    prerequisites_valid = (
        connection_equivalent
        and prestress_distinguishable
        and read_valid
    )

    if prerequisites_valid:
        abr_result = evolve_system(
            evolution_id="retained_prestress_ABR",
            source_image=ab_final,
            event=registered_read,
            config=benchmark_config(),
        )

        bar_result = evolve_system(
            evolution_id="retained_prestress_BAR",
            source_image=ba_final,
            event=registered_read,
            config=benchmark_config(),
        )

        abr_final = abr_result.target_image
        bar_final = bar_result.target_image

        post_read_prestress_distance = _euclidean(
            _prestress_vector(
                abr_final.prestress_nodes
            ),
            _prestress_vector(
                bar_final.prestress_nodes
            ),
        )

        post_read_connection_distance = _euclidean(
            _connection_state_vector(
                abr_final.adaptive_connections
            ),
            _connection_state_vector(
                bar_final.adaptive_connections
            ),
        )

        retained_prestress_transition_relevant = (
            post_read_prestress_distance
            > TOLERANCE
        )

        if retained_prestress_transition_relevant:
            decision = (
                "RETAINED_PRESTRESS_IS_"
                "TRANSITION_RELEVANT_FOR_REGISTERED_READ"
            )
        else:
            decision = (
                "NULL_NOT_REJECTED_FOR_REGISTERED_READ"
            )

        post_read = {
            "ABR_prestress": _node_values(
                abr_final.prestress_nodes
            ),
            "BAR_prestress": _node_values(
                bar_final.prestress_nodes
            ),
            "prestress_distance": (
                post_read_prestress_distance
            ),
            "connection_state_distance": (
                post_read_connection_distance
            ),
            "retained_prestress_transition_relevant": (
                retained_prestress_transition_relevant
            ),
        }

    else:
        decision = (
            "INVALID_EXPERIMENT:"
            "_PREREGISTERED_TERMINAL_OR_READ_"
            "CONDITIONS_NOT_SATISFIED"
        )

        post_read = None

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "preregistration": preregistration,
        "tolerance": TOLERANCE,

        "terminal_state": {
            "AB_prestress": _node_values(
                ab_final.prestress_nodes
            ),
            "BA_prestress": _node_values(
                ba_final.prestress_nodes
            ),
            "prestress_distance": (
                terminal_prestress_distance
            ),

            "AB_connections": _connection_values(
                ab_final.adaptive_connections
            ),
            "BA_connections": _connection_values(
                ba_final.adaptive_connections
            ),
            "connection_state_distance": (
                terminal_connection_distance
            ),

            "connection_equivalent": (
                connection_equivalent
            ),
            "prestress_distinguishable": (
                prestress_distinguishable
            ),
        },

        "registered_read_validation": (
            read_validation
        ),

        "prerequisites_valid": prerequisites_valid,

        "post_read": post_read,

        "decision": decision,

        "claims_not_tested": [
            "global_minimality",
            "biological_validity",
            "clinical_validity",
            "necessity_of_prestress_in_all_dynamical_systems",
            "uniqueness_of_path_dependence",
            "novelty_of_mechanical_memory",
        ],
    }


def main():
    data = run_benchmark()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "RETAINED PRESTRESS FUTURE-TRANSITION "
        "FALSIFICATION"
    )
    print("=" * 60)

    terminal = data["terminal_state"]

    print(
        "Terminal connection distance:",
        terminal["connection_state_distance"],
    )
    print(
        "Terminal prestress distance:",
        terminal["prestress_distance"],
    )
    print(
        "Connection states equivalent:",
        terminal["connection_equivalent"],
    )
    print(
        "Prestress distinguishable:",
        terminal["prestress_distinguishable"],
    )
    print(
        "Locked READ valid:",
        data["registered_read_validation"][
            "matches_preregistered_read"
        ],
    )
    print(
        "Prerequisites valid:",
        data["prerequisites_valid"],
    )

    if data["post_read"] is not None:
        print(
            "Post-READ prestress distance:",
            data["post_read"]["prestress_distance"],
        )
        print(
            "Post-READ connection distance:",
            data["post_read"][
                "connection_state_distance"
            ],
        )

    print()
    print("DECISION:", data["decision"])
    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()


