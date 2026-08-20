
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from roif.history.adaptive_connection import AdaptiveConnectionState, ConnectionExposure
from roif.history.prestress_redistribution import PrestressNodeState, PrestressPerturbation
from roif.history.system_evolution import SystemEvent, SystemEvolutionConfig, evolve_system
from roif.history.system_image import (
    HistoricalTrace,
    SystemImage,
    SystemImageContext,
    SystemMeasure,
    build_system_image,
)

BENCHMARK_VERSION = "roif_temporal_image_reconstruction_v1"
CLAIM_SCOPE = "computational_model_only"


class TemporalReconstructionError(RuntimeError):
    pass


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise TemporalReconstructionError("vector lengths differ")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _rmse(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise TemporalReconstructionError("vector lengths differ")
    if not a:
        return 0.0
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / len(a))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_source_image() -> SystemImage:
    return build_system_image(
        image_id="temporal_source",
        revision=0,
        measures=(
            SystemMeasure(
                measure_id="temperature",
                measure_type="temperature",
                value=36.8,
                unit="degC",
                domain="thermal",
                normalized_value=0.48,
            ),
            SystemMeasure(
                measure_id="mass_proxy",
                measure_type="mass",
                value=80.0,
                unit="kg",
                domain="physical",
                normalized_value=0.55,
            ),
            SystemMeasure(
                measure_id="electrical_proxy",
                measure_type="electrical_activity",
                value=1.0,
                unit="arb",
                domain="electrical",
                normalized_value=0.40,
            ),
            SystemMeasure(
                measure_id="chemical_proxy",
                measure_type="chemical_state",
                value=7.2,
                unit="pH",
                domain="chemical",
                normalized_value=0.60,
            ),
        ),
        prestress_nodes=(
            PrestressNodeState(node_id="A", prestress=0.20, reserve=1.00),
            PrestressNodeState(node_id="B", prestress=0.10, reserve=0.90),
            PrestressNodeState(node_id="C", prestress=0.05, reserve=0.85),
            PrestressNodeState(node_id="D", prestress=0.00, reserve=0.80),
        ),
        adaptive_connections=(
            AdaptiveConnectionState(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                stiffness=1.00,
                contractile_capacity=1.00,
                reflex_gain=1.00,
                fatigue=0.00,
                remodeling_bias=0.00,
            ),
            AdaptiveConnectionState(
                connection_id="B_C",
                source_node_id="B",
                target_node_id="C",
                stiffness=0.95,
                contractile_capacity=0.95,
                reflex_gain=0.90,
                fatigue=0.00,
                remodeling_bias=0.00,
            ),
            AdaptiveConnectionState(
                connection_id="C_D",
                source_node_id="C",
                target_node_id="D",
                stiffness=0.90,
                contractile_capacity=0.90,
                reflex_gain=0.85,
                fatigue=0.00,
                remodeling_bias=0.00,
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
        context=SystemImageContext(context_id="temporal_ctx_0", timestamp_label="t0"),
    )


def build_events() -> tuple[SystemEvent, ...]:
    return (
        SystemEvent(
            event_id="E1",
            event_type="local_damage_A_B",
            connection_exposures=(
                ConnectionExposure(
                    exposure_id="E1_A_B",
                    connection_id="A_B",
                    load=0.70,
                    strain=0.25,
                    activation=0.80,
                    damage=0.30,
                    recovery=0.00,
                ),
            ),
            prestress_perturbations=(
                PrestressPerturbation(
                    perturbation_id="E1_A",
                    node_id="A",
                    delta=-0.20,
                ),
            ),
            target_context=SystemImageContext(context_id="temporal_ctx_1", timestamp_label="t1"),
        ),
        SystemEvent(
            event_id="E2",
            event_type="secondary_loading_B_C",
            connection_exposures=(
                ConnectionExposure(
                    exposure_id="E2_B_C",
                    connection_id="B_C",
                    load=0.60,
                    strain=0.40,
                    activation=0.50,
                    damage=0.10,
                    recovery=0.00,
                ),
            ),
            prestress_perturbations=(
                PrestressPerturbation(
                    perturbation_id="E2_B",
                    node_id="B",
                    delta=0.16,
                ),
            ),
            target_context=SystemImageContext(context_id="temporal_ctx_2", timestamp_label="t2"),
        ),
        SystemEvent(
            event_id="E3",
            event_type="partial_recovery_A_B",
            connection_exposures=(
                ConnectionExposure(
                    exposure_id="E3_A_B",
                    connection_id="A_B",
                    load=0.35,
                    strain=0.15,
                    activation=0.45,
                    damage=0.00,
                    recovery=0.30,
                ),
            ),
            prestress_perturbations=(
                PrestressPerturbation(
                    perturbation_id="E3_C",
                    node_id="C",
                    delta=0.12,
                ),
            ),
            target_context=SystemImageContext(context_id="temporal_ctx_3", timestamp_label="t3"),
        ),
        SystemEvent(
            event_id="E4",
            event_type="distal_loading_C_D",
            connection_exposures=(
                ConnectionExposure(
                    exposure_id="E4_C_D",
                    connection_id="C_D",
                    load=0.80,
                    strain=0.30,
                    activation=0.65,
                    damage=0.20,
                    recovery=0.00,
                ),
            ),
            prestress_perturbations=(
                PrestressPerturbation(
                    perturbation_id="E4_D",
                    node_id="D",
                    delta=-0.14,
                ),
            ),
            target_context=SystemImageContext(context_id="temporal_ctx_4", timestamp_label="t4"),
        ),
        SystemEvent(
            event_id="E5",
            event_type="global_rechallenge",
            connection_exposures=(
                ConnectionExposure(
                    exposure_id="E5_A_B",
                    connection_id="A_B",
                    load=0.55,
                    strain=0.20,
                    activation=0.55,
                    damage=0.05,
                    recovery=0.00,
                ),
                ConnectionExposure(
                    exposure_id="E5_B_C",
                    connection_id="B_C",
                    load=0.55,
                    strain=0.25,
                    activation=0.55,
                    damage=0.05,
                    recovery=0.00,
                ),
            ),
            prestress_perturbations=(
                PrestressPerturbation(
                    perturbation_id="E5_A",
                    node_id="A",
                    delta=0.10,
                ),
            ),
            target_context=SystemImageContext(context_id="temporal_ctx_5", timestamp_label="t5"),
        ),
    )


def benchmark_config() -> SystemEvolutionConfig:
    return SystemEvolutionConfig(
        trace_type="temporal_reconstruction",
        trace_persistence=1.0,
    )


def image_vector(image: SystemImage) -> tuple[float, ...]:
    measures = tuple(
        float(m.normalized_value)
        for m in sorted(image.measures, key=lambda x: x.measure_id)
    )
    prestress = tuple(
        float(n.prestress)
        for n in sorted(image.prestress_nodes, key=lambda x: x.node_id)
    )
    adaptive: list[float] = []
    for c in sorted(image.adaptive_connections, key=lambda x: x.connection_id):
        adaptive.extend(
            (
                float(c.stiffness),
                float(c.contractile_capacity),
                float(c.reflex_gain),
                float(c.fatigue),
                float(c.remodeling_bias),
            )
        )
    trace_values = tuple(float(t.magnitude) for t in image.historical_traces)
    return measures + prestress + tuple(adaptive) + (
        float(len(trace_values)),
        float(sum(trace_values)),
        float(math.sqrt(sum(v * v for v in trace_values))),
    )


def generate_truth_trajectory() -> tuple[SystemImage, ...]:
    current = build_source_image()
    images = [current]
    for index, event in enumerate(build_events(), start=1):
        result = evolve_system(
            evolution_id=f"truth::{index}::{event.event_id}",
            source_image=current,
            event=event,
            config=benchmark_config(),
            target_image_id=f"truth_image_{index}",
        )
        current = result.target_image
        images.append(current)
    return tuple(images)


def linear_interpolate_vector(
    left: Sequence[float],
    right: Sequence[float],
    fraction: float,
) -> tuple[float, ...]:
    if len(left) != len(right):
        raise TemporalReconstructionError("vector lengths differ")
    if not 0.0 <= fraction <= 1.0:
        raise TemporalReconstructionError("fraction must lie in [0,1]")
    return tuple(
        float(a) + fraction * (float(b) - float(a))
        for a, b in zip(left, right)
    )


def linear_interpolation_reconstruction(
    truth: Sequence[SystemImage],
) -> dict[int, tuple[float, ...]]:
    vectors = [image_vector(x) for x in truth]
    return {
        1: linear_interpolate_vector(vectors[0], vectors[2], 0.5),
        3: linear_interpolate_vector(vectors[2], vectors[4], 0.5),
        5: tuple(
            float(v4) + 0.5 * (float(v4) - float(v2))
            for v2, v4 in zip(vectors[2], vectors[4])
        ),
    }


def memory_free_reconstruction(
    source: SystemImage,
    events: Sequence[SystemEvent],
) -> dict[int, tuple[float, ...]]:
    output: dict[int, tuple[float, ...]] = {}
    for index, event in enumerate(events, start=1):
        result = evolve_system(
            evolution_id=f"memory_free::{index}::{event.event_id}",
            source_image=source,
            event=event,
            config=benchmark_config(),
            target_image_id=f"memory_free_image_{index}",
        )
        if index in {1, 3, 5}:
            output[index] = image_vector(result.target_image)
    return output


def _restore_baseline_connections(
    image: SystemImage,
    baseline: SystemImage,
    image_id: str,
) -> SystemImage:
    return build_system_image(
        image_id=image_id,
        revision=image.revision,
        measures=image.measures,
        prestress_nodes=image.prestress_nodes,
        adaptive_connections=baseline.adaptive_connections,
        historical_traces=image.historical_traces,
        context=image.context,
        metadata={
            "control": "history_aware_fixed_coupling",
            "adaptive_connections_restored_to_baseline": True,
        },
    )


def fixed_coupling_reconstruction(
    source: SystemImage,
    events: Sequence[SystemEvent],
) -> dict[int, tuple[float, ...]]:
    output: dict[int, tuple[float, ...]] = {}
    current = source
    for index, event in enumerate(events, start=1):
        prepared = _restore_baseline_connections(
            current,
            source,
            image_id=f"fixed_prepared_{index}",
        )
        result = evolve_system(
            evolution_id=f"fixed::{index}::{event.event_id}",
            source_image=prepared,
            event=event,
            config=benchmark_config(),
            target_image_id=f"fixed_image_{index}",
        )
        current = result.target_image
        if index in {1, 3, 5}:
            output[index] = image_vector(current)
    return output


def full_roif_reconstruction(
    source: SystemImage,
    events: Sequence[SystemEvent],
) -> dict[int, tuple[float, ...]]:
    output: dict[int, tuple[float, ...]] = {}
    current = source
    for index, event in enumerate(events, start=1):
        result = evolve_system(
            evolution_id=f"full::{index}::{event.event_id}",
            source_image=current,
            event=event,
            config=benchmark_config(),
            target_image_id=f"full_image_{index}",
        )
        current = result.target_image
        if index in {1, 3, 5}:
            output[index] = image_vector(current)
    return output


def method_errors(
    truth: Sequence[SystemImage],
    reconstructed: dict[int, tuple[float, ...]],
) -> dict[str, Any]:
    per_slice: dict[str, dict[str, float]] = {}
    hidden_rmse: list[float] = []
    future_rmse: list[float] = []
    all_rmse: list[float] = []
    for index in (1, 3, 5):
        actual = image_vector(truth[index])
        estimate = reconstructed[index]
        distance = _euclidean(actual, estimate)
        rmse = _rmse(actual, estimate)
        per_slice[str(index)] = {
            "euclidean_distance": distance,
            "rmse": rmse,
        }
        all_rmse.append(rmse)
        (hidden_rmse if index in {1, 3} else future_rmse).append(rmse)
    return {
        "per_slice": per_slice,
        "mean_hidden_slice_rmse": _mean(hidden_rmse),
        "future_slice_rmse": _mean(future_rmse),
        "mean_all_target_rmse": _mean(all_rmse),
    }


def run_temporal_reconstruction() -> dict[str, Any]:
    truth = generate_truth_trajectory()
    source = truth[0]
    events = build_events()
    reconstructions = {
        "linear_interpolation": linear_interpolation_reconstruction(truth),
        "memory_free": memory_free_reconstruction(source, events),
        "history_aware_fixed_coupling": fixed_coupling_reconstruction(source, events),
        "full_roif": full_roif_reconstruction(source, events),
    }
    errors = {
        name: method_errors(truth, reconstruction)
        for name, reconstruction in reconstructions.items()
    }
    ranking = sorted(
        ((name, float(block["mean_all_target_rmse"])) for name, block in errors.items()),
        key=lambda x: (x[1], x[0]),
    )
    other_best = min(
        block["mean_all_target_rmse"]
        for name, block in errors.items()
        if name != "full_roif"
    )
    return {
        "observed_slice_indices": [0, 2, 4],
        "hidden_slice_indices": [1, 3],
        "future_slice_indices": [5],
        "methods": errors,
        "ranking_by_mean_all_target_rmse": [
            {"rank": i, "method": name, "mean_all_target_rmse": score}
            for i, (name, score) in enumerate(ranking, start=1)
        ],
        "full_roif_best_or_tied": (
            errors["full_roif"]["mean_all_target_rmse"] <= other_best
        ),
        "note": (
            "The synthetic truth and full replay share the same deterministic "
            "transition model; this tests internal reconstruction consistency, "
            "not external predictive validity."
        ),
    }


def _event_copy(event: SystemEvent, event_id: str) -> SystemEvent:
    return SystemEvent(
        event_id=event_id,
        event_type=event.event_type,
        connection_exposures=event.connection_exposures,
        prestress_perturbations=event.prestress_perturbations,
        replacement_measures=event.replacement_measures,
        target_context=event.target_context,
        metadata=event.metadata,
    )


def _trajectory_from_events(
    sequence_id: str,
    events: Sequence[SystemEvent],
) -> tuple[SystemImage, ...]:
    current = build_source_image()
    images = [current]
    for index, event in enumerate(events, start=1):
        result = evolve_system(
            evolution_id=f"{sequence_id}::{index}::{event.event_id}",
            source_image=current,
            event=event,
            config=benchmark_config(),
            target_image_id=f"{sequence_id}_image_{index}",
        )
        current = result.target_image
        images.append(current)
    return tuple(images)


def run_operator_evolution() -> dict[str, Any]:
    events = build_events()
    a = _event_copy(events[0], "A")
    b = _event_copy(events[1], "B")
    c = _event_copy(events[4], "C")

    abc = _trajectory_from_events("ABC", (a, b, c))
    bac = _trajectory_from_events("BAC", (b, a, c))

    pre_c = _euclidean(image_vector(abc[2]), image_vector(bac[2]))
    post_c = _euclidean(image_vector(abc[3]), image_vector(bac[3]))

    abc_response = tuple(
        after - before
        for before, after in zip(image_vector(abc[2]), image_vector(abc[3]))
    )
    bac_response = tuple(
        after - before
        for before, after in zip(image_vector(bac[2]), image_vector(bac[3]))
    )
    c_response_distance = _euclidean(abc_response, bac_response)

    trajectory_distance = math.sqrt(
        sum(
            _euclidean(image_vector(x), image_vector(y)) ** 2
            for x, y in zip(abc, bac)
        )
    )

    return {
        "same_event_c_used": True,
        "pre_c_state_distance": pre_c,
        "post_c_state_distance": post_c,
        "c_response_distance": c_response_distance,
        "trajectory_distance": trajectory_distance,
        "history_changes_response_to_c": c_response_distance > 1e-12,
        "abc_final_revision": abc[-1].revision,
        "bac_final_revision": bac[-1].revision,
        "interpretation": (
            "A non-zero C-response distance means that the same event C acts "
            "differently after AB versus BA history in this computational model."
        ),
    }


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def run_benchmark() -> dict[str, Any]:
    truth = generate_truth_trajectory()
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "clinical_validation_claimed": False,
        "biological_truth_claimed": False,
        "causal_truth_claimed": False,
        "external_predictive_validity_claimed": False,
        "ground_truth_source": "synthetic_roif_generated_trajectory",
        "temporal_reconstruction": run_temporal_reconstruction(),
        "operator_evolution": run_operator_evolution(),
        "trajectory": {
            "slice_count": len(truth),
            "observed_slice_indices": [0, 2, 4],
            "withheld_intermediate_slice_indices": [1, 3],
            "withheld_future_slice_indices": [5],
            "vector_dimension": len(image_vector(truth[0])),
        },
        "interpretation_note": (
            "This benchmark evaluates controlled synthetic reconstruction and "
            "operator/history dependence. It is not empirical forecasting "
            "validation."
        ),
    }


def main() -> None:
    data = run_benchmark()
    truth = generate_truth_trajectory()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "temporal_image_reconstruction_v1.json").write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _write_csv(
        output_dir / "temporal_image_reconstruction_slices_v1.csv",
        (
            "slice_index",
            "image_id",
            "revision",
            "vector_dimension",
            "vector_json",
            "observed",
            "withheld_intermediate",
            "withheld_future",
        ),
        (
            (
                index,
                image.image_id,
                image.revision,
                len(image_vector(image)),
                json.dumps(image_vector(image)),
                index in {0, 2, 4},
                index in {1, 3},
                index == 5,
            )
            for index, image in enumerate(truth)
        ),
    )

    _write_csv(
        output_dir / "temporal_image_reconstruction_methods_v1.csv",
        (
            "method",
            "mean_hidden_slice_rmse",
            "future_slice_rmse",
            "mean_all_target_rmse",
        ),
        (
            (
                method,
                block["mean_hidden_slice_rmse"],
                block["future_slice_rmse"],
                block["mean_all_target_rmse"],
            )
            for method, block
            in data["temporal_reconstruction"]["methods"].items()
        ),
    )

    _write_csv(
        output_dir / "operator_evolution_v1.csv",
        ("metric", "value"),
        (
            ("pre_c_state_distance", data["operator_evolution"]["pre_c_state_distance"]),
            ("post_c_state_distance", data["operator_evolution"]["post_c_state_distance"]),
            ("c_response_distance", data["operator_evolution"]["c_response_distance"]),
            ("trajectory_distance", data["operator_evolution"]["trajectory_distance"]),
        ),
    )

    print(json.dumps(data, indent=2, sort_keys=True))
    print("\nSaved temporal reconstruction outputs to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
