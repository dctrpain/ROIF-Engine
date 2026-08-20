from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.roif_temporal_image_reconstruction_benchmark import (
    benchmark_config,
    build_events,
    build_source_image,
)

from roif.history.system_evolution import (
    SystemEvent,
    evolve_system,
)

from roif.history.system_image import (
    SystemImage,
    revise_system_image,
)


# =============================================================================
# BENCHMARK METADATA
# =============================================================================

BENCHMARK_VERSION = "roif_predictive_stabilization_v1"
CLAIM_SCOPE = "computational_model_only"

HISTORY_EVENT_COUNT = 4

CONTROLLER_GAIN = 0.80
MAX_ABS_ACTION_PER_NODE = 0.12

CHALLENGE_FACTORS = (
    0.80,
    1.00,
    1.20,
)


# =============================================================================
# ERRORS
# =============================================================================


class PredictiveStabilizationBenchmarkError(RuntimeError):
    """Raised when the stabilization benchmark cannot be evaluated."""


# =============================================================================
# BASIC NUMERIC HELPERS
# =============================================================================


def _euclidean(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise PredictiveStabilizationBenchmarkError(
            "vector lengths differ"
        )

    return math.sqrt(
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
        raise PredictiveStabilizationBenchmarkError(
            "vector lengths differ"
        )

    if not left:
        return 0.0

    return max(
        abs(float(a) - float(b))
        for a, b in zip(left, right)
    )


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(
        sum(float(value) ** 2 for value in values)
    )


# =============================================================================
# SYSTEMIMAGE REPRESENTATION USED BY THIS BENCHMARK
# =============================================================================


def prestress_vector(
    image: SystemImage,
) -> tuple[float, ...]:
    """
    Extract the prestress component of the current SystemImage.

    The benchmark evaluates stabilization of the prestress configuration,
    while the full SystemImage remains the state object evolved by ROIF.
    """

    return tuple(
        float(node.prestress)
        for node in sorted(
            image.prestress_nodes,
            key=lambda item: item.node_id,
        )
    )


def prestress_by_node(
    image: SystemImage,
) -> dict[str, float]:
    return {
        node.node_id: float(node.prestress)
        for node in image.prestress_nodes
    }


# =============================================================================
# BUILD HISTORY-CONDITIONED PRE-EVENT STATE
# =============================================================================


def build_history_conditioned_pre_event_image() -> SystemImage:
    """
    Generate the pre-challenge SystemImage after E1 -> E2 -> E3 -> E4.

    The challenge E5 is withheld.

    This state therefore contains the consequences of previous ROIF evolution:
    - prestress redistribution,
    - adaptive connection modification,
    - structural traces,
    - accumulated historical consequences.
    """

    current = build_source_image()

    history_events = build_events()[:HISTORY_EVENT_COUNT]

    for index, event in enumerate(
        history_events,
        start=1,
    ):
        result = evolve_system(
            evolution_id=(
                f"predictive_stabilization_history::"
                f"{index}::{event.event_id}"
            ),
            source_image=current,
            event=event,
            config=benchmark_config(),
            target_image_id=(
                f"predictive_stabilization_history_image_{index}"
            ),
        )

        current = result.target_image

    return current


# =============================================================================
# CHALLENGE SCALING
# =============================================================================


def _bounded_unit_interval(
    value: float,
) -> float:
    return max(
        0.0,
        min(1.0, float(value)),
    )


def scale_challenge(
    event: SystemEvent,
    factor: float,
    *,
    event_id: str,
) -> SystemEvent:
    """
    Create a controlled intensity variant of the same nominal challenge.

    The event topology and affected components remain unchanged.

    Scaled quantities:
    - load
    - strain
    - activation
    - damage
    - prestress perturbation

    Recovery is preserved because it represents a distinct process rather than
    challenge magnitude.
    """

    factor = float(factor)

    if factor <= 0.0:
        raise PredictiveStabilizationBenchmarkError(
            "challenge factor must be positive"
        )

    scaled_exposures = tuple(
        replace(
            exposure,
            exposure_id=(
                f"{exposure.exposure_id}"
                f"__scale_{factor:.2f}"
            ),
            load=float(exposure.load) * factor,
            strain=float(exposure.strain) * factor,
            activation=_bounded_unit_interval(
                float(exposure.activation) * factor
            ),
            damage=_bounded_unit_interval(
                float(exposure.damage) * factor
            ),
        )
        for exposure in event.connection_exposures
    )

    scaled_perturbations = tuple(
        replace(
            perturbation,
            perturbation_id=(
                f"{perturbation.perturbation_id}"
                f"__scale_{factor:.2f}"
            ),
            delta=float(perturbation.delta) * factor,
        )
        for perturbation in event.prestress_perturbations
    )

    return replace(
        event,
        event_id=event_id,
        connection_exposures=scaled_exposures,
        prestress_perturbations=scaled_perturbations,
    )


# =============================================================================
# ROIF EVENT RESPONSE
# =============================================================================


def apply_event(
    source_image: SystemImage,
    event: SystemEvent,
    *,
    run_id: str,
) -> SystemImage:
    """
    Apply one real ROIF evolution step.
    """

    result = evolve_system(
        evolution_id=f"{run_id}::{event.event_id}",
        source_image=source_image,
        event=event,
        config=benchmark_config(),
        target_image_id=f"{run_id}::post_event",
    )

    return result.target_image


def predict_prestress_response(
    source_image: SystemImage,
    nominal_event: SystemEvent,
    *,
    run_id: str,
) -> tuple[float, ...]:
    """
    Model-conditioned prediction of the prestress component of the future
    SystemImage under the nominal expected event.

    This is not external forecasting.

    It is an internal model-based estimate:

        I_t + expected event
            -> predicted future SystemImage

    from which the prestress displacement is extracted.
    """

    predicted_post = apply_event(
        source_image,
        nominal_event,
        run_id=run_id,
    )

    before = prestress_vector(source_image)
    after = prestress_vector(predicted_post)

    return tuple(
        float(post) - float(pre)
        for pre, post in zip(before, after)
    )


# =============================================================================
# PREDICTIVE PRECONFIGURATION
# =============================================================================


def bounded_predictive_action(
    predicted_response: Sequence[float],
    *,
    gain: float = CONTROLLER_GAIN,
    max_abs_action: float = MAX_ABS_ACTION_PER_NODE,
) -> tuple[float, ...]:
    """
    Construct bounded anticipatory preconfiguration:

        P_t = -gain * predicted displacement

    The controller therefore prepares the system against the modeled expected
    displacement before the event occurs.
    """

    if gain < 0.0:
        raise PredictiveStabilizationBenchmarkError(
            "controller gain must be non-negative"
        )

    if max_abs_action < 0.0:
        raise PredictiveStabilizationBenchmarkError(
            "maximum action magnitude must be non-negative"
        )

    action = []

    for predicted_delta in predicted_response:
        requested = (
            -float(gain)
            * float(predicted_delta)
        )

        bounded = max(
            -float(max_abs_action),
            min(
                float(max_abs_action),
                requested,
            ),
        )

        action.append(bounded)

    return tuple(action)


def rescale_action_to_norm(
    action: Sequence[float],
    target_norm: float,
    *,
    max_abs_action: float = MAX_ABS_ACTION_PER_NODE,
) -> tuple[float, ...]:
    """
    Rescale a control action to the requested L2 action budget while preserving
    its directional pattern as far as possible.

    This is used to make the stale/history-blind control comparable with the
    history-conditioned action under the same total intervention budget.
    """

    values = tuple(float(value) for value in action)

    current_norm = _norm(values)

    if current_norm == 0.0:
        return tuple(0.0 for _ in values)

    if target_norm < 0.0:
        raise PredictiveStabilizationBenchmarkError(
            "target action norm must be non-negative"
        )

    scale = float(target_norm) / current_norm

    scaled = tuple(
        float(value) * scale
        for value in values
    )

    peak = max(
        (abs(value) for value in scaled),
        default=0.0,
    )

    if peak > max_abs_action:
        cap_scale = (
            float(max_abs_action) / peak
        )

        scaled = tuple(
            value * cap_scale
            for value in scaled
        )

    return scaled


def apply_action_to_system_image(
    image: SystemImage,
    action: Sequence[float],
    *,
    image_id: str,
    policy_name: str,
) -> SystemImage:
    """
    Apply P_t to the prestress layer of the current SystemImage before the
    challenge occurs.

    No topology is changed.
    No historical trace is fabricated.
    No future observation is injected.
    """

    ordered_nodes = sorted(
        image.prestress_nodes,
        key=lambda item: item.node_id,
    )

    if len(ordered_nodes) != len(action):
        raise PredictiveStabilizationBenchmarkError(
            "action dimension does not match prestress dimension"
        )

    action_by_node = {
        node.node_id: float(delta)
        for node, delta in zip(
            ordered_nodes,
            action,
        )
    }

    revised_nodes = []

    for node in image.prestress_nodes:
        requested = (
            float(node.prestress)
            + action_by_node[node.node_id]
        )

        bounded = max(
            float(node.min_prestress),
            min(
                float(node.max_prestress),
                requested,
            ),
        )

        revised_nodes.append(
            replace(
                node,
                prestress=bounded,
            )
        )

    return revise_system_image(
        source=image,
        target_image_id=image_id,
        prestress_nodes=tuple(revised_nodes),
        metadata={
            "predictive_preconfiguration_applied": True,
            "preconfiguration_policy": policy_name,
            "action_selected": True,
            "policy_modified": True,
            "learning_applied": False,
            "topology_modified": False,
            "biological_truth_claimed": False,
            "causal_truth_inferred": False,
        },
    )


# =============================================================================
# METRICS
# =============================================================================


def stabilization_metrics(
    reference: SystemImage,
    observed: SystemImage,
) -> dict[str, float]:
    """
    Compare the post-event prestress configuration with the history-conditioned
    pre-event reference configuration.

    Lower values indicate less event-induced displacement from the pre-event
    operating configuration.
    """

    reference_vector = prestress_vector(reference)
    observed_vector = prestress_vector(observed)

    return {
        "euclidean_deviation": _euclidean(
            reference_vector,
            observed_vector,
        ),
        "peak_abs_node_deviation": _max_abs_difference(
            reference_vector,
            observed_vector,
        ),
    }


def action_metrics(
    action: Sequence[float],
) -> dict[str, float]:
    return {
        "action_l2_norm": _norm(action),
        "action_peak_abs": (
            max(abs(float(value)) for value in action)
            if action
            else 0.0
        ),
    }


# =============================================================================
# SINGLE CHALLENGE CONDITION
# =============================================================================


def run_challenge_condition(
    *,
    pre_event_image: SystemImage,
    realised_event: SystemEvent,
    history_conditioned_action: Sequence[float],
    stale_action: Sequence[float],
    opposite_action: Sequence[float],
    condition_name: str,
) -> dict[str, Any]:
    """
    Compare four conditions under the identical realised event:

    1. no preconfiguration
    2. history-conditioned predictive preconfiguration
    3. stale/history-blind preconfiguration
    4. opposite same-budget sham preconfiguration
    """

    # -------------------------------------------------------------------------
    # 1. NO PRECONFIGURATION
    # -------------------------------------------------------------------------

    uncontrolled_post = apply_event(
        pre_event_image,
        realised_event,
        run_id=f"{condition_name}::no_preconfiguration",
    )

    uncontrolled_metrics = stabilization_metrics(
        pre_event_image,
        uncontrolled_post,
    )

    # -------------------------------------------------------------------------
    # 2. HISTORY-CONDITIONED P_t
    # -------------------------------------------------------------------------

    history_prepared = apply_action_to_system_image(
        pre_event_image,
        history_conditioned_action,
        image_id=(
            f"{condition_name}::"
            "history_conditioned_prepared"
        ),
        policy_name="history_conditioned_predictive",
    )

    history_post = apply_event(
        history_prepared,
        realised_event,
        run_id=(
            f"{condition_name}::"
            "history_conditioned"
        ),
    )

    history_metrics = stabilization_metrics(
        pre_event_image,
        history_post,
    )

    # -------------------------------------------------------------------------
    # 3. STALE / HISTORY-BLIND CONTROL
    # -------------------------------------------------------------------------

    stale_prepared = apply_action_to_system_image(
        pre_event_image,
        stale_action,
        image_id=(
            f"{condition_name}::"
            "stale_prepared"
        ),
        policy_name="stale_history_blind",
    )

    stale_post = apply_event(
        stale_prepared,
        realised_event,
        run_id=f"{condition_name}::stale",
    )

    stale_metrics = stabilization_metrics(
        pre_event_image,
        stale_post,
    )

    # -------------------------------------------------------------------------
    # 4. OPPOSITE SAME-BUDGET SHAM
    # -------------------------------------------------------------------------

    opposite_prepared = apply_action_to_system_image(
        pre_event_image,
        opposite_action,
        image_id=(
            f"{condition_name}::"
            "opposite_sham_prepared"
        ),
        policy_name="opposite_same_budget_sham",
    )

    opposite_post = apply_event(
        opposite_prepared,
        realised_event,
        run_id=f"{condition_name}::opposite_sham",
    )

    opposite_metrics = stabilization_metrics(
        pre_event_image,
        opposite_post,
    )

    baseline = float(
        uncontrolled_metrics["euclidean_deviation"]
    )

    history_error = float(
        history_metrics["euclidean_deviation"]
    )

    stale_error = float(
        stale_metrics["euclidean_deviation"]
    )

    opposite_error = float(
        opposite_metrics["euclidean_deviation"]
    )

    absolute_improvement = (
        baseline - history_error
    )

    relative_improvement = (
        absolute_improvement / baseline
        if baseline > 0.0
        else 0.0
    )

    return {
        "condition": condition_name,
        "realised_event_id": realised_event.event_id,

        "reference_prestress": prestress_by_node(
            pre_event_image
        ),

        "no_preconfiguration": {
            **uncontrolled_metrics,
            "post_prestress": prestress_by_node(
                uncontrolled_post
            ),
        },

        "history_conditioned_predictive": {
            **history_metrics,
            **action_metrics(
                history_conditioned_action
            ),
            "action": list(
                history_conditioned_action
            ),
            "post_prestress": prestress_by_node(
                history_post
            ),
        },

        "stale_history_blind": {
            **stale_metrics,
            **action_metrics(stale_action),
            "action": list(stale_action),
            "post_prestress": prestress_by_node(
                stale_post
            ),
        },

        "opposite_same_budget_sham": {
            **opposite_metrics,
            **action_metrics(opposite_action),
            "action": list(opposite_action),
            "post_prestress": prestress_by_node(
                opposite_post
            ),
        },

        "absolute_improvement_vs_no_preconfiguration": (
            absolute_improvement
        ),

        "relative_improvement_vs_no_preconfiguration": (
            relative_improvement
        ),

        "history_conditioned_beats_no_preconfiguration": (
            history_error < baseline
        ),

        "history_conditioned_beats_stale_control": (
            history_error < stale_error
        ),

        "history_conditioned_beats_opposite_sham": (
            history_error < opposite_error
        ),
    }


# =============================================================================
# FULL BENCHMARK
# =============================================================================


def run_benchmark() -> dict[str, Any]:
    """
    Predictive stabilization benchmark.

    Experimental logic
    ------------------

    Historical trajectory:

        E1 -> E2 -> E3 -> E4

    produces the current history-conditioned SystemImage I_t.

    The nominal expected challenge E5 is then used internally to construct a
    possible event-conditioned future response.

    From that predicted response we construct:

        P_t

    before the realised E5-like event occurs.

    Primary comparison:

        same history
        same realised event
        no P_t
            versus
        history-conditioned P_t

    Additional controls:

        stale/history-blind P_t
        opposite same-budget sham P_t

    Challenge intensity is varied around the nominal expected value to ensure
    that the controller is not evaluated only under exact event matching.
    """

    baseline_source = build_source_image()

    pre_event_image = (
        build_history_conditioned_pre_event_image()
    )

    nominal_event = build_events()[4]

    # -------------------------------------------------------------------------
    # HISTORY-CONDITIONED PREDICTION
    # -------------------------------------------------------------------------

    history_predicted_response = (
        predict_prestress_response(
            pre_event_image,
            nominal_event,
            run_id=(
                "predictive_stabilization::"
                "history_conditioned_prediction"
            ),
        )
    )

    history_action = bounded_predictive_action(
        history_predicted_response
    )

    # -------------------------------------------------------------------------
    # STALE / HISTORY-BLIND CONTROL
    #
    # Predict the same nominal event from the original baseline SystemImage
    # rather than from the history-conditioned pre-event SystemImage.
    # The resulting action is then applied to the actual current system.
    # -------------------------------------------------------------------------

    stale_predicted_response = (
        predict_prestress_response(
            baseline_source,
            nominal_event,
            run_id=(
                "predictive_stabilization::"
                "stale_prediction"
            ),
        )
    )

    raw_stale_action = bounded_predictive_action(
        stale_predicted_response
    )

    history_action_norm = _norm(
        history_action
    )

    stale_action = rescale_action_to_norm(
        raw_stale_action,
        history_action_norm,
    )

    # -------------------------------------------------------------------------
    # SAME-BUDGET DIRECTIONAL SHAM
    # -------------------------------------------------------------------------

    opposite_action = tuple(
        -float(value)
        for value in history_action
    )

    # -------------------------------------------------------------------------
    # REALISED CHALLENGES
    # -------------------------------------------------------------------------

    conditions: dict[str, Any] = {}

    for factor in CHALLENGE_FACTORS:
        name = f"challenge_{int(round(factor * 100)):03d}pct"

        realised_event = scale_challenge(
            nominal_event,
            factor,
            event_id=(
                f"{nominal_event.event_id}"
                f"__realised_{factor:.2f}"
            ),
        )

        conditions[name] = run_challenge_condition(
            pre_event_image=pre_event_image,
            realised_event=realised_event,
            history_conditioned_action=history_action,
            stale_action=stale_action,
            opposite_action=opposite_action,
            condition_name=name,
        )

    # -------------------------------------------------------------------------
    # CENTRAL RESULT FLAGS
    # -------------------------------------------------------------------------

    beats_no_control_all = all(
        block[
            "history_conditioned_beats_no_preconfiguration"
        ]
        for block in conditions.values()
    )

    beats_stale_all = all(
        block[
            "history_conditioned_beats_stale_control"
        ]
        for block in conditions.values()
    )

    beats_opposite_all = all(
        block[
            "history_conditioned_beats_opposite_sham"
        ]
        for block in conditions.values()
    )

    improvements = tuple(
        float(
            block[
                "relative_improvement_vs_no_preconfiguration"
            ]
        )
        for block in conditions.values()
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": CLAIM_SCOPE,

        "clinical_validation_claimed": False,
        "biological_truth_claimed": False,
        "external_predictive_validity_claimed": False,
        "topology_independent_robustness_claimed": False,
        "universal_stability_claimed": False,

        "prestress_layer_one_step_stabilization_tested": True,
        "whole_system_image_stabilization_claimed": False,
        "multi_step_temporal_image_stabilization_claimed": False,
        "general_optimal_stabilization_policy_claimed": False,

        "mechanism_under_test": (
            "history_conditioned_one_step_prestress_preconfiguration"
        ),

        "history_event_count": HISTORY_EVENT_COUNT,

        "history_sequence": [
            event.event_id
            for event in build_events()[
                :HISTORY_EVENT_COUNT
            ]
        ],

        "nominal_expected_event": (
            nominal_event.event_id
        ),

        "challenge_factors": list(
            CHALLENGE_FACTORS
        ),

        "controller": {
            "gain": CONTROLLER_GAIN,
            "max_abs_action_per_node": (
                MAX_ABS_ACTION_PER_NODE
            ),
            "history_conditioned_action": list(
                history_action
            ),
            "stale_history_blind_action": list(
                stale_action
            ),
            "opposite_same_budget_action": list(
                opposite_action
            ),
            "history_conditioned_action_norm": _norm(
                history_action
            ),
            "stale_history_blind_action_norm": _norm(
                stale_action
            ),
        },

        "history_conditioned_prediction": list(
            history_predicted_response
        ),

        "stale_prediction": list(
            stale_predicted_response
        ),

        "pre_event_prestress": prestress_by_node(
            pre_event_image
        ),

        "conditions": conditions,

        "central_results": {
            "history_conditioned_beats_no_control_all_conditions": (
                beats_no_control_all
            ),
            "history_conditioned_beats_stale_all_conditions": (
                beats_stale_all
            ),
            "history_conditioned_beats_opposite_sham_all_conditions": (
                beats_opposite_all
            ),
            "mean_relative_improvement_vs_no_control": (
                sum(improvements) / len(improvements)
                if improvements
                else 0.0
            ),
            "minimum_relative_improvement_vs_no_control": (
                min(improvements)
                if improvements
                else 0.0
            ),
            "maximum_relative_improvement_vs_no_control": (
                max(improvements)
                if improvements
                else 0.0
            ),
        },

        "interpretation": (
            "This controlled computational benchmark evaluates a one-step "
            "history-conditioned predictive preconfiguration mechanism restricted "
            "to the prestress layer of the implemented ROIF model. The current "
            "history-conditioned SystemImage is evolved under a nominal expected "
            "event to obtain a model-internal future SystemImage, but the control "
            "action is derived only from the predicted prestress displacement and "
            "is applied only to the prestress layer. The outcome metric measures "
            "post-event prestress deviation from the pre-event prestress "
            "configuration. The benchmark therefore demonstrates only a bounded "
            "one-step prestress-layer stabilization effect within this controlled "
            "computational implementation. It does not test whole-SystemImage "
            "predictive stabilization, multi-step Temporal-Image stabilization, "
            "or a general optimal stabilization policy. It also does not establish "
            "external predictive validity, biological or clinical validity, "
            "universal stability, or topology-independent robustness."
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

    output_dir = (
        root
        / "benchmark_results"
    )

    output_path = (
        output_dir
        / "predictive_stabilization_v1.json"
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
    print("Saved predictive stabilization benchmark:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
