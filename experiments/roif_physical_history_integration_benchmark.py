"""
ROIF Engine
Physical History Integration Benchmark

Purpose
-------
End-to-end proof-of-concept for the new physical-history branch.

Questions
---------
Q1. Order dependence:
    Does A -> B differ from B -> A?

Q2. Global redistribution:
    Can a local perturbation alter prestress beyond the directly perturbed node?

Q3. Repeated-event state dependence:
    Does the same event produce a different effect after the system has already
    been changed by a previous event?

Scope
-----
Computational model only.
No clinical validation.
No biological truth claim.
No causal truth claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path

from roif.history.adaptive_connection import (
    AdaptiveConnectionState,
    ConnectionExposure,
)
from roif.history.prestress_redistribution import (
    PrestressNodeState,
    PrestressPerturbation,
)
from roif.history.system_evolution import (
    SystemEvent,
    SystemEvolutionConfig,
    evolve_sequence,
    evolve_system,
)
from roif.history.system_image import (
    HistoricalTrace,
    SystemImageContext,
    SystemMeasure,
    build_system_image,
    system_image_signature,
)


BENCHMARK_VERSION = "roif_physical_history_integration_v1"


def build_source_image():
    return build_system_image(
        image_id="physical_source",
        revision=0,
        measures=(
            SystemMeasure(
                measure_id="global_load",
                measure_type="normalized_load",
                value=1.0,
                unit="arb",
                normalized_value=0.5,
            ),
        ),
        prestress_nodes=(
            PrestressNodeState(
                node_id="A",
                prestress=0.20,
                reserve=1.0,
            ),
            PrestressNodeState(
                node_id="B",
                prestress=0.10,
                reserve=1.0,
            ),
            PrestressNodeState(
                node_id="C",
                prestress=0.00,
                reserve=1.0,
            ),
        ),
        adaptive_connections=(
            AdaptiveConnectionState(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                stiffness=1.0,
                contractile_capacity=1.0,
                reflex_gain=1.0,
                fatigue=0.0,
                remodeling_bias=0.0,
            ),
            AdaptiveConnectionState(
                connection_id="B_C",
                source_node_id="B",
                target_node_id="C",
                stiffness=1.0,
                contractile_capacity=1.0,
                reflex_gain=1.0,
                fatigue=0.0,
                remodeling_bias=0.0,
            ),
        ),
        historical_traces=(
            HistoricalTrace(
                trace_id="baseline_trace",
                source_event_id="baseline",
                trace_type="baseline",
                magnitude=0.0,
                persistence=1.0,
                relation_count=0,
            ),
        ),
        context=SystemImageContext(
            context_id="ctx_0",
            timestamp_label="t0",
        ),
    )


def event_a():
    return SystemEvent(
        event_id="A",
        event_type="local_damage_A_B",
        connection_exposures=(
            ConnectionExposure(
                exposure_id="A_exp",
                connection_id="A_B",
                load=0.7,
                strain=0.3,
                activation=0.8,
                damage=0.45,
                recovery=0.0,
            ),
        ),
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id="A_perturb",
                node_id="A",
                delta=-0.25,
            ),
        ),
        target_context=SystemImageContext(
            context_id="ctx_A",
            timestamp_label="after_A",
        ),
    )


def event_b():
    return SystemEvent(
        event_id="B",
        event_type="secondary_loading",
        connection_exposures=(
            ConnectionExposure(
                exposure_id="B_exp",
                connection_id="B_C",
                load=0.6,
                strain=0.5,
                activation=0.4,
                damage=0.10,
                recovery=0.0,
            ),
        ),
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id="B_perturb",
                node_id="B",
                delta=0.20,
            ),
        ),
        target_context=SystemImageContext(
            context_id="ctx_B",
            timestamp_label="after_B",
        ),
    )


def benchmark_config():
    return SystemEvolutionConfig()


def _vector_distance(left, right):
    if len(left) != len(right):
        raise ValueError("vector lengths differ")

    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right)
        )
    )


def _image_state_vector(image):
    prestress = tuple(
        node.prestress
        for node in sorted(
            image.prestress_nodes,
            key=lambda x: x.node_id,
        )
    )

    adaptive = []

    for connection in sorted(
        image.adaptive_connections,
        key=lambda x: x.connection_id,
    ):
        adaptive.extend(
            (
                connection.stiffness,
                connection.contractile_capacity,
                connection.reflex_gain,
                connection.fatigue,
                connection.remodeling_bias,
            )
        )

    return prestress + tuple(adaptive)


def run_order_dependence():
    source = build_source_image()
    config = benchmark_config()

    ab = evolve_sequence(
        sequence_id="AB",
        source_image=source,
        events=(
            event_a(),
            event_b(),
        ),
        config=config,
    )

    ba = evolve_sequence(
        sequence_id="BA",
        source_image=source,
        events=(
            event_b(),
            event_a(),
        ),
        config=config,
    )

    ab_final = ab[-1].target_image
    ba_final = ba[-1].target_image

    distance = _vector_distance(
        _image_state_vector(ab_final),
        _image_state_vector(ba_final),
    )

    return {
        "ab_final_revision": ab_final.revision,
        "ba_final_revision": ba_final.revision,
        "final_state_distance": distance,
        "path_dependent": distance > 1e-12,
        "ab_signature": system_image_signature(
            ab_final
        ),
        "ba_signature": system_image_signature(
            ba_final
        ),
    }


def run_global_redistribution():
    source = build_source_image()

    result = evolve_system(
        evolution_id="global_redistribution",
        source_image=source,
        event=event_a(),
        config=benchmark_config(),
    )

    deltas = dict(
        result.prestress_result.node_delta_by_id
    )

    directly_perturbed = {
        perturbation.node_id
        for perturbation
        in result.event.prestress_perturbations
    }

    downstream_changed = {
        node_id: delta
        for node_id, delta in deltas.items()
        if (
            node_id not in directly_perturbed
            and abs(delta) > 1e-12
        )
    }

    return {
        "directly_perturbed_nodes": sorted(
            directly_perturbed
        ),
        "all_node_deltas": deltas,
        "downstream_changed_nodes": downstream_changed,
        "global_redistribution_detected": bool(
            downstream_changed
        ),
        "prestress_change_norm": (
            result.summary.prestress_change_norm
        ),
    }


def run_repeated_event_dependence():
    source = build_source_image()
    config = benchmark_config()
    event = event_a()

    first = evolve_system(
        evolution_id="repeat_1",
        source_image=source,
        event=event,
        config=config,
        target_image_id="repeat_image_1",
    )

    second = evolve_system(
        evolution_id="repeat_2",
        source_image=first.target_image,
        event=event,
        config=config,
        target_image_id="repeat_image_2",
    )

    first_response = (
        first.summary.connection_change_norm,
        first.summary.prestress_change_norm,
        first.summary.trace_magnitude,
    )

    second_response = (
        second.summary.connection_change_norm,
        second.summary.prestress_change_norm,
        second.summary.trace_magnitude,
    )

    distance = _vector_distance(
        first_response,
        second_response,
    )

    return {
        "first_response": {
            "connection_change_norm": first_response[0],
            "prestress_change_norm": first_response[1],
            "trace_magnitude": first_response[2],
        },
        "second_response": {
            "connection_change_norm": second_response[0],
            "prestress_change_norm": second_response[1],
            "trace_magnitude": second_response[2],
        },
        "response_distance": distance,
        "same_event_different_response": (
            distance > 1e-12
        ),
        "final_revision": (
            second.target_image.revision
        ),
        "final_trace_count": len(
            second.target_image.historical_traces
        ),
    }


def run_benchmark():
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": "computational_model_only",
        "clinical_validation_claimed": False,
        "biological_truth_claimed": False,
        "causal_truth_claimed": False,
        "order_dependence": run_order_dependence(),
        "global_redistribution": run_global_redistribution(),
        "repeated_event_state_dependence": (
            run_repeated_event_dependence()
        ),
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
        / "physical_history_integration_v1.json"
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print(
        "\nSaved physical-history benchmark to: "
        + str(
            output_path.resolve()
        )
    )


if __name__ == "__main__":
    main()
