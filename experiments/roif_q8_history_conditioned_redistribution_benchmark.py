from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from roif.history.adaptive_connection import (
    AdaptiveConnectionState,
    ConnectionExposure,
    effective_transfer_gain,
)

from roif.history.prestress_redistribution import (
    PrestressNodeState,
    PrestressPerturbation,
    PrestressRedistributionConfig,
)

from roif.history.system_evolution import (
    SystemEvent,
    SystemEvolutionConfig,
    SystemEvolutionResult,
    evolve_system,
)

from roif.history.system_image import (
    HistoricalTrace,
    SystemImage,
    SystemImageContext,
    SystemMeasure,
    build_system_image,
)


# =============================================================================
# Q8 METADATA
# =============================================================================

BENCHMARK_VERSION = (
    "roif_q8_history_conditioned_redistribution_capacity_v1"
)

CLAIM_SCOPE = "computational_model_only"

CONDITIONING_CYCLE_COUNT = 5

# Standard matched challenge applied once to:
#   1. baseline structure
#   2. history-conditioned structure
STANDARD_CHALLENGE_DELTA = -0.40


# =============================================================================
# ERRORS
# =============================================================================


class Q8BenchmarkError(RuntimeError):
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================


def benchmark_config() -> SystemEvolutionConfig:
    """
    Use the existing ROIF Engine state-update rules.

    Prestress propagation is explicit across three links:
        A -> B -> C -> D

    No topology change, learning algorithm, prediction policy, or biological
    interpretation is introduced.
    """

    return SystemEvolutionConfig(
        prestress_config=PrestressRedistributionConfig(
            propagation_steps=3,
            propagation_decay=0.75,
            reserve_modulation=False,
        ),
        trace_type="q8_history_conditioned_redistribution",
        trace_persistence=1.0,
    )


# =============================================================================
# BASELINE SYSTEM
# =============================================================================


def build_source_image() -> SystemImage:
    """
    Minimal four-node prestressed network.

        A -> B -> C -> D

    All three adaptive connections begin from the same state.

    The network is intentionally minimal so that historical modification of
    transmission capacity remains directly interpretable.
    """

    return build_system_image(
        image_id="q8_source",
        revision=0,
        measures=(
            SystemMeasure(
                measure_id="external_load_proxy",
                measure_type="normalized_load",
                value=0.0,
                unit="arb",
                domain="mechanical",
                normalized_value=0.0,
            ),
        ),
        prestress_nodes=(
            PrestressNodeState(
                node_id="A",
                prestress=0.20,
                min_prestress=-2.0,
                max_prestress=2.0,
                reserve=1.0,
            ),
            PrestressNodeState(
                node_id="B",
                prestress=0.10,
                min_prestress=-2.0,
                max_prestress=2.0,
                reserve=1.0,
            ),
            PrestressNodeState(
                node_id="C",
                prestress=0.05,
                min_prestress=-2.0,
                max_prestress=2.0,
                reserve=1.0,
            ),
            PrestressNodeState(
                node_id="D",
                prestress=0.00,
                min_prestress=-2.0,
                max_prestress=2.0,
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
                min_stiffness=0.25,
                max_stiffness=5.0,
                min_contractile_capacity=0.25,
                max_contractile_capacity=5.0,
                min_reflex_gain=0.25,
                max_reflex_gain=5.0,
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
                min_stiffness=0.25,
                max_stiffness=5.0,
                min_contractile_capacity=0.25,
                max_contractile_capacity=5.0,
                min_reflex_gain=0.25,
                max_reflex_gain=5.0,
            ),
            AdaptiveConnectionState(
                connection_id="C_D",
                source_node_id="C",
                target_node_id="D",
                stiffness=1.0,
                contractile_capacity=1.0,
                reflex_gain=1.0,
                fatigue=0.0,
                remodeling_bias=0.0,
                min_stiffness=0.25,
                max_stiffness=5.0,
                min_contractile_capacity=0.25,
                max_contractile_capacity=5.0,
                min_reflex_gain=0.25,
                max_reflex_gain=5.0,
            ),
        ),
        historical_traces=(
            HistoricalTrace(
                trace_id="q8_baseline_trace",
                source_event_id="baseline",
                trace_type="baseline",
                magnitude=0.0,
                persistence=1.0,
                relation_count=0,
            ),
        ),
        context=SystemImageContext(
            context_id="q8_t0",
            timestamp_label="t0",
        ),
        metadata={
            "benchmark": BENCHMARK_VERSION,
            "topology_modified": False,
            "learning_applied": False,
            "biological_truth_claimed": False,
        },
    )


# =============================================================================
# ADMISSIBLE CONDITIONING HISTORY
# =============================================================================


def build_conditioning_event(
    cycle_index: int,
) -> SystemEvent:
    """
    One aggregated admissible load/recovery cycle.

    This is deliberately NOT called exercise, tissue training, biological
    adaptation, or development.

    It is a domain-agnostic computational exposure designed to produce:

        load + activation
            -> state-dependent connection update
            -> retained historical revision
            -> altered future transfer structure

    Exposure values are chosen so that the cycle contains a strengthening
    signal while avoiding damage.

    Recovery is included in the same aggregated cycle to prevent the benchmark
    from degenerating into a pure fatigue accumulation experiment.
    """

    exposures = tuple(
        ConnectionExposure(
            exposure_id=f"conditioning_{cycle_index}_{connection_id}",
            connection_id=connection_id,
            load=0.40,
            strain=0.10,
            activation=0.50,
            damage=0.00,
            recovery=0.50,
        )
        for connection_id in (
            "A_B",
            "B_C",
            "C_D",
        )
    )

    return SystemEvent(
        event_id=f"conditioning_cycle_{cycle_index}",
        event_type="admissible_repeated_loading",
        connection_exposures=exposures,

        # A modest structural perturbation accompanies each history cycle.
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id=(
                    f"conditioning_perturbation_{cycle_index}"
                ),
                node_id="A",
                delta=-0.05,
            ),
        ),
        target_context=SystemImageContext(
            context_id=f"q8_conditioning_{cycle_index}",
            timestamp_label=f"conditioning_{cycle_index}",
        ),
        metadata={
            "conditioning_cycle": cycle_index,
            "damage_intentionally_zero": True,
            "biological_truth_claimed": False,
        },
    )


def build_conditioning_history() -> tuple[SystemEvent, ...]:
    return tuple(
        build_conditioning_event(index)
        for index in range(
            1,
            CONDITIONING_CYCLE_COUNT + 1,
        )
    )


# =============================================================================
# MATCHED CHALLENGE
# =============================================================================


def build_standard_challenge(
    event_id: str,
) -> SystemEvent:
    """
    Matched external perturbation.

    IMPORTANT:
    The challenge contains no connection exposure.

    Therefore it does not strengthen or weaken connections immediately before
    redistribution. The exact same perturbation is allowed to propagate through
    whichever structure already exists at challenge time.

    This isolates:

        baseline transfer structure
            versus
        history-conditioned transfer structure.
    """

    return SystemEvent(
        event_id=event_id,
        event_type="matched_external_challenge",
        connection_exposures=(),
        prestress_perturbations=(
            PrestressPerturbation(
                perturbation_id=f"{event_id}_A",
                node_id="A",
                delta=STANDARD_CHALLENGE_DELTA,
            ),
        ),
        target_context=SystemImageContext(
            context_id=f"{event_id}_context",
            timestamp_label=event_id,
        ),
        metadata={
            "matched_challenge": True,
            "connection_exposure_during_challenge": False,
            "biological_truth_claimed": False,
        },
    )


# =============================================================================
# STRUCTURAL METRICS
# =============================================================================


def connection_metrics(
    image: SystemImage,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}

    gains: list[float] = []
    stiffness_values: list[float] = []
    capacity_values: list[float] = []
    reflex_values: list[float] = []
    fatigue_values: list[float] = []
    remodeling_values: list[float] = []

    for connection in sorted(
        image.adaptive_connections,
        key=lambda item: item.connection_id,
    ):
        gain = float(
            effective_transfer_gain(connection)
        )

        gains.append(gain)
        stiffness_values.append(
            float(connection.stiffness)
        )
        capacity_values.append(
            float(connection.contractile_capacity)
        )
        reflex_values.append(
            float(connection.reflex_gain)
        )
        fatigue_values.append(
            float(connection.fatigue)
        )
        remodeling_values.append(
            float(connection.remodeling_bias)
        )

        rows[connection.connection_id] = {
            "stiffness": float(
                connection.stiffness
            ),
            "contractile_capacity": float(
                connection.contractile_capacity
            ),
            "reflex_gain": float(
                connection.reflex_gain
            ),
            "fatigue": float(
                connection.fatigue
            ),
            "remodeling_bias": float(
                connection.remodeling_bias
            ),
            "effective_transfer_gain": gain,
        }

    return {
        "per_connection": rows,
        "mean_effective_transfer_gain": (
            sum(gains) / len(gains)
            if gains
            else 0.0
        ),
        "minimum_effective_transfer_gain": (
            min(gains)
            if gains
            else 0.0
        ),
        "mean_stiffness": (
            sum(stiffness_values)
            / len(stiffness_values)
            if stiffness_values
            else 0.0
        ),
        "mean_contractile_capacity": (
            sum(capacity_values)
            / len(capacity_values)
            if capacity_values
            else 0.0
        ),
        "mean_reflex_gain": (
            sum(reflex_values)
            / len(reflex_values)
            if reflex_values
            else 0.0
        ),
        "mean_fatigue": (
            sum(fatigue_values)
            / len(fatigue_values)
            if fatigue_values
            else 0.0
        ),
        "mean_remodeling_bias": (
            sum(remodeling_values)
            / len(remodeling_values)
            if remodeling_values
            else 0.0
        ),
    }


# =============================================================================
# REDISTRIBUTION METRICS
# =============================================================================


def _normalized_entropy(
    values: Sequence[float],
) -> float:
    """
    Normalized Shannon entropy of absolute node response distribution.

    0:
        response concentrated at one node.

    1:
        response distributed equally across all represented nodes.

    This is a distribution metric only.
    It is NOT thermodynamic entropy.
    """

    magnitudes = [
        abs(float(value))
        for value in values
    ]

    total = sum(magnitudes)

    if total <= 0.0:
        return 0.0

    probabilities = [
        value / total
        for value in magnitudes
        if value > 0.0
    ]

    entropy = -sum(
        p * math.log(p)
        for p in probabilities
    )

    maximum_entropy = math.log(
        len(magnitudes)
    )

    if maximum_entropy <= 0.0:
        return 0.0

    return entropy / maximum_entropy


def redistribution_metrics(
    result: SystemEvolutionResult,
) -> dict[str, Any]:
    deltas = {
        str(node_id): float(delta)
        for node_id, delta
        in result.prestress_result.node_delta_by_id.items()
    }

    ordered_ids = (
        "A",
        "B",
        "C",
        "D",
    )

    ordered = tuple(
        float(deltas.get(node_id, 0.0))
        for node_id in ordered_ids
    )

    magnitudes = tuple(
        abs(value)
        for value in ordered
    )

    total_abs = sum(magnitudes)

    source_abs = magnitudes[0]

    downstream_abs = sum(
        magnitudes[1:]
    )

    peak_abs = max(
        magnitudes,
        default=0.0,
    )

    source_fraction = (
        source_abs / total_abs
        if total_abs > 0.0
        else 0.0
    )

    downstream_fraction = (
        downstream_abs / total_abs
        if total_abs > 0.0
        else 0.0
    )

    downstream_to_source_ratio = (
        downstream_abs / source_abs
        if source_abs > 0.0
        else 0.0
    )

    peak_fraction = (
        peak_abs / total_abs
        if total_abs > 0.0
        else 0.0
    )

    reached_node_count = sum(
        1
        for value in magnitudes
        if value > 1e-12
    )

    return {
        "node_deltas": {
            node_id: ordered[index]
            for index, node_id
            in enumerate(ordered_ids)
        },
        "total_absolute_response": total_abs,
        "source_absolute_response": source_abs,
        "downstream_absolute_response": downstream_abs,
        "source_fraction_of_response": source_fraction,
        "downstream_fraction_of_response": downstream_fraction,
        "downstream_to_source_ratio": (
            downstream_to_source_ratio
        ),
        "peak_absolute_node_response": peak_abs,
        "peak_fraction_of_response": peak_fraction,
        "distribution_entropy_normalized": (
            _normalized_entropy(ordered)
        ),
        "reached_node_count": reached_node_count,
        "prestress_change_norm": float(
            result.summary.prestress_change_norm
        ),
    }


# =============================================================================
# HISTORY EXECUTION
# =============================================================================


def evolve_conditioning_history(
    source: SystemImage,
) -> tuple[
    SystemImage,
    tuple[dict[str, Any], ...],
]:
    """
    Apply repeated admissible conditioning cycles recursively.

    Each event acts on the SystemImage produced by the previous event.
    """

    current = source
    trajectory: list[dict[str, Any]] = []

    for index, event in enumerate(
        build_conditioning_history(),
        start=1,
    ):
        before_metrics = connection_metrics(
            current
        )

        result = evolve_system(
            evolution_id=(
                f"q8_conditioning::{index}"
            ),
            source_image=current,
            event=event,
            config=benchmark_config(),
            target_image_id=(
                f"q8_conditioned_image_{index}"
            ),
        )

        current = result.target_image

        after_metrics = connection_metrics(
            current
        )

        trajectory.append(
            {
                "cycle": index,
                "source_revision": (
                    result.source_image.revision
                ),
                "target_revision": (
                    result.target_image.revision
                ),
                "mean_transfer_gain_before": (
                    before_metrics[
                        "mean_effective_transfer_gain"
                    ]
                ),
                "mean_transfer_gain_after": (
                    after_metrics[
                        "mean_effective_transfer_gain"
                    ]
                ),
                "mean_stiffness_after": (
                    after_metrics[
                        "mean_stiffness"
                    ]
                ),
                "mean_contractile_capacity_after": (
                    after_metrics[
                        "mean_contractile_capacity"
                    ]
                ),
                "mean_reflex_gain_after": (
                    after_metrics[
                        "mean_reflex_gain"
                    ]
                ),
                "mean_fatigue_after": (
                    after_metrics[
                        "mean_fatigue"
                    ]
                ),
                "mean_remodeling_bias_after": (
                    after_metrics[
                        "mean_remodeling_bias"
                    ]
                ),
                "connection_change_norm": float(
                    result.summary.connection_change_norm
                ),
                "prestress_change_norm": float(
                    result.summary.prestress_change_norm
                ),
                "trace_magnitude": float(
                    result.summary.trace_magnitude
                ),
            }
        )

    return current, tuple(trajectory)


# =============================================================================
# MATCHED-CHALLENGE COMPARISON
# =============================================================================


def run_matched_challenge(
    source: SystemImage,
    *,
    run_id: str,
) -> SystemEvolutionResult:
    return evolve_system(
        evolution_id=run_id,
        source_image=source,
        event=build_standard_challenge(
            event_id=f"{run_id}_challenge"
        ),
        config=benchmark_config(),
        target_image_id=f"{run_id}_post",
    )


# =============================================================================
# FULL Q8 BENCHMARK
# =============================================================================


def run_benchmark() -> dict[str, Any]:
    """
    Q8-v1:
    History-conditioned strengthening of perturbation redistribution capacity.

    Experimental logic
    ------------------

    BASELINE:
        I0
        -> matched challenge C
        -> R_C(I0)

    CONDITIONED:
        I0
        -> H = E1 -> E2 -> ... -> En
        -> I_H
        -> same matched challenge C
        -> R_C(I_H)

    The challenge itself contains no adaptive-connection exposure.

    Therefore any difference in redistribution under C follows from the
    structure accumulated before C.

    The benchmark asks:

    1. Did repeated admissible history alter/strengthen connection state?
    2. Did effective transfer capacity increase?
    3. Did the same local perturbation become less concentrated at its source?
    4. Did a larger fraction of the response become distributed downstream?

    It does NOT test:
    - energy dissipation,
    - literal mechanical absorption,
    - injury/failure thresholds,
    - biological adaptation,
    - universal tensegrity behavior,
    - predictive preconfiguration.
    """

    baseline = build_source_image()

    baseline_structure = connection_metrics(
        baseline
    )

    baseline_challenge = run_matched_challenge(
        baseline,
        run_id="q8_baseline",
    )

    baseline_response = redistribution_metrics(
        baseline_challenge
    )

    conditioned, history_trajectory = (
        evolve_conditioning_history(
            baseline
        )
    )

    conditioned_structure = connection_metrics(
        conditioned
    )

    conditioned_challenge = run_matched_challenge(
        conditioned,
        run_id="q8_conditioned",
    )

    conditioned_response = redistribution_metrics(
        conditioned_challenge
    )

    baseline_gain = float(
        baseline_structure[
            "mean_effective_transfer_gain"
        ]
    )

    conditioned_gain = float(
        conditioned_structure[
            "mean_effective_transfer_gain"
        ]
    )

    gain_change = (
        conditioned_gain - baseline_gain
    )

    gain_relative_change = (
        gain_change / baseline_gain
        if baseline_gain > 0.0
        else 0.0
    )

    baseline_source_fraction = float(
        baseline_response[
            "source_fraction_of_response"
        ]
    )

    conditioned_source_fraction = float(
        conditioned_response[
            "source_fraction_of_response"
        ]
    )

    baseline_downstream_fraction = float(
        baseline_response[
            "downstream_fraction_of_response"
        ]
    )

    conditioned_downstream_fraction = float(
        conditioned_response[
            "downstream_fraction_of_response"
        ]
    )

    baseline_entropy = float(
        baseline_response[
            "distribution_entropy_normalized"
        ]
    )

    conditioned_entropy = float(
        conditioned_response[
            "distribution_entropy_normalized"
        ]
    )

    transfer_gain_increased = (
        conditioned_gain > baseline_gain
    )

    source_concentration_reduced = (
        conditioned_source_fraction
        < baseline_source_fraction
    )

    downstream_fraction_increased = (
        conditioned_downstream_fraction
        > baseline_downstream_fraction
    )

    distribution_entropy_increased = (
        conditioned_entropy
        > baseline_entropy
    )

    cycle_gains = tuple(
        float(
            item["mean_transfer_gain_after"]
        )
        for item in history_trajectory
    )

    progressive_gain_increase = all(
        right > left
        for left, right
        in zip(
            (baseline_gain, *cycle_gains[:-1]),
            cycle_gains,
        )
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": CLAIM_SCOPE,

        "mechanism_under_test": (
            "history_conditioned_redistribution_capacity"
        ),

        "conditioning_cycle_count": (
            CONDITIONING_CYCLE_COUNT
        ),

        "matched_challenge_delta": (
            STANDARD_CHALLENGE_DELTA
        ),

        "baseline_structure": (
            baseline_structure
        ),

        "conditioned_structure": (
            conditioned_structure
        ),

        "conditioning_history_trajectory": list(
            history_trajectory
        ),

        "baseline_matched_challenge": (
            baseline_response
        ),

        "conditioned_matched_challenge": (
            conditioned_response
        ),

        "central_results": {
            "mean_effective_transfer_gain_change": (
                gain_change
            ),
            "mean_effective_transfer_gain_relative_change": (
                gain_relative_change
            ),
            "transfer_gain_increased_after_history": (
                transfer_gain_increased
            ),
            "progressive_transfer_gain_increase_across_cycles": (
                progressive_gain_increase
            ),
            "source_response_concentration_reduced": (
                source_concentration_reduced
            ),
            "downstream_response_fraction_increased": (
                downstream_fraction_increased
            ),
            "distribution_entropy_increased": (
                distribution_entropy_increased
            ),
        },

        # ---------------------------------------------------------------------
        # EXPLICIT CLAIM BOUNDARIES
        # ---------------------------------------------------------------------

        "history_conditioned_connection_change_tested": True,
        "matched_event_redistribution_change_tested": True,

        "energy_dissipation_tested": False,
        "literal_absorption_tested": False,
        "injury_or_failure_threshold_tested": False,
        "biological_adaptation_claimed": False,
        "human_foot_model_claimed": False,
        "predictive_preconfiguration_tested": False,
        "whole_system_stability_claimed": False,
        "universal_tensegrity_claimed": False,

        "interpretation": (
            "This controlled computational benchmark tests whether repeated "
            "admissible history changes the adaptive connection state of a "
            "prestressed ROIF network and thereby changes redistribution of "
            "the same subsequent local perturbation. The implemented engine "
            "updates stiffness, contractile capacity, reflex gain, fatigue, "
            "and remodeling bias, and derives effective transfer gain from "
            "the resulting connection state. Q8-v1 therefore tests "
            "history-conditioned strengthening of redistribution capacity "
            "and matched-event redistribution geometry. Reduced source "
            "concentration and increased downstream distribution, if "
            "observed, indicate broader redistribution within this "
            "computational network. This benchmark does not yet model "
            "physical energy dissipation, literal absorption, injury or "
            "failure thresholds, biological tissue adaptation, human-foot "
            "biomechanics, predictive preconfiguration, or universal "
            "tensegrity stability."
        ),
    }


# =============================================================================
# OUTPUT
# =============================================================================


def _write_json(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    data = run_benchmark()

    root = Path(__file__).resolve().parents[1]

    output_path = (
        root
        / "benchmark_results"
        / "q8_history_conditioned_redistribution_capacity_v1.json"
    )

    _write_json(
        output_path,
        data,
    )

    print(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("Saved Q8 benchmark:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()