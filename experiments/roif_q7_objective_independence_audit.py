from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import roif_predictive_stabilization_benchmark as q7


BENCHMARK_VERSION = "roif_q7_objective_independence_audit_v1"
CLAIM_SCOPE = "computational_model_only"

ZERO_TOL = 1e-12


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def vector_add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        float(a) + float(b)
        for a, b in zip(left, right)
    )


def vector_subtract(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        float(a) - float(b)
        for a, b in zip(left, right)
    )


def vector_norm(
    values: Sequence[float],
) -> float:
    return math.sqrt(
        sum(float(value) ** 2 for value in values)
    )


def max_abs(
    values: Sequence[float],
) -> float:
    return max(
        (abs(float(value)) for value in values),
        default=0.0,
    )


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    return (
        sum(
            float(a) * float(b)
            for a, b in zip(left, right)
        )
        / (left_norm * right_norm)
    )


def prestress_displacement(
    reference,
    observed,
) -> tuple[float, ...]:
    before = q7.prestress_vector(reference)
    after = q7.prestress_vector(observed)

    return vector_subtract(
        after,
        before,
    )


# =============================================================================
# ACTION APPLICATION
# =============================================================================


def evaluate_action(
    *,
    pre_event_image,
    realised_event,
    action: Sequence[float],
    label: str,
) -> dict[str, Any]:
    prepared = q7.apply_action_to_system_image(
        pre_event_image,
        action,
        image_id=f"{label}::prepared",
        policy_name=f"audit::{label}",
    )

    post = q7.apply_event(
        prepared,
        realised_event,
        run_id=f"audit::{label}",
    )

    displacement = prestress_displacement(
        pre_event_image,
        post,
    )

    return {
        "action": tuple(float(x) for x in action),
        "post_displacement": displacement,
        "euclidean_deviation": vector_norm(displacement),
        "peak_abs_deviation": max_abs(displacement),
    }


# =============================================================================
# EQUAL-BUDGET ORACLE
# =============================================================================


def equal_budget_inverse_action(
    realised_no_control_displacement: Sequence[float],
    *,
    target_norm: float,
) -> tuple[float, ...]:
    """
    Construct an ex-post equal-budget inverse action using the realised
    no-control displacement itself.

    This is NOT a valid predictive controller.

    It is an oracle used only to audit whether the Q7 objective is directly
    minimized by cancelling the same vector that the metric later measures.
    """

    inverse = tuple(
        -float(value)
        for value in realised_no_control_displacement
    )

    return q7.rescale_action_to_norm(
        inverse,
        target_norm,
    )


# =============================================================================
# DETERMINISTIC DIRECTION SEARCH
# =============================================================================


def candidate_equal_budget_actions(
    *,
    dimension: int,
    target_norm: float,
) -> tuple[tuple[float, ...], ...]:
    """
    Deterministic equal-budget direction set.

    Directions are generated from {-1, 0, +1}^n excluding the zero vector.
    Every candidate is rescaled to the same L2 intervention budget.

    This is an audit search, not an optimizer used by ROIF.
    """

    candidates = []
    seen = set()

    for raw in itertools.product(
        (-1.0, 0.0, 1.0),
        repeat=dimension,
    ):
        if all(value == 0.0 for value in raw):
            continue

        action = q7.rescale_action_to_norm(
            raw,
            target_norm,
        )

        key = tuple(
            round(float(value), 15)
            for value in action
        )

        if key in seen:
            continue

        seen.add(key)
        candidates.append(action)

    return tuple(candidates)


# =============================================================================
# SINGLE CONDITION AUDIT
# =============================================================================


def audit_condition(
    *,
    pre_event_image,
    nominal_event,
    realised_event,
    history_action: Sequence[float],
    condition_name: str,
) -> dict[str, Any]:
    # -------------------------------------------------------------------------
    # NO CONTROL
    # -------------------------------------------------------------------------

    no_control_post = q7.apply_event(
        pre_event_image,
        realised_event,
        run_id=f"audit::{condition_name}::no_control",
    )

    event_displacement = prestress_displacement(
        pre_event_image,
        no_control_post,
    )

    # -------------------------------------------------------------------------
    # Q7 HISTORY-CONDITIONED ACTION
    # -------------------------------------------------------------------------

    history_result = evaluate_action(
        pre_event_image=pre_event_image,
        realised_event=realised_event,
        action=history_action,
        label=f"{condition_name}::history_action",
    )

    # -------------------------------------------------------------------------
    # SUPERPOSITION / OBJECTIVE-COUPLING AUDIT
    # -------------------------------------------------------------------------

    algebraic_prediction = vector_add(
        event_displacement,
        history_action,
    )

    superposition_residual = vector_subtract(
        history_result["post_displacement"],
        algebraic_prediction,
    )

    # -------------------------------------------------------------------------
    # EX-POST ORACLE
    # -------------------------------------------------------------------------

    action_budget = vector_norm(
        history_action
    )

    oracle_action = equal_budget_inverse_action(
        event_displacement,
        target_norm=action_budget,
    )

    oracle_result = evaluate_action(
        pre_event_image=pre_event_image,
        realised_event=realised_event,
        action=oracle_action,
        label=f"{condition_name}::oracle",
    )

    # -------------------------------------------------------------------------
    # DETERMINISTIC EQUAL-BUDGET DIRECTION SEARCH
    # -------------------------------------------------------------------------

    candidates = candidate_equal_budget_actions(
        dimension=len(history_action),
        target_norm=action_budget,
    )

    candidate_results = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        evaluated = evaluate_action(
            pre_event_image=pre_event_image,
            realised_event=realised_event,
            action=candidate,
            label=(
                f"{condition_name}::candidate_{index:03d}"
            ),
        )

        candidate_results.append(
            {
                "index": index,
                "action": list(candidate),
                "euclidean_deviation": (
                    evaluated["euclidean_deviation"]
                ),
                "peak_abs_deviation": (
                    evaluated["peak_abs_deviation"]
                ),
            }
        )

    ranked = sorted(
        candidate_results,
        key=lambda item: (
            item["euclidean_deviation"],
            item["index"],
        ),
    )

    best_search = ranked[0]

    history_euclidean = float(
        history_result["euclidean_deviation"]
    )

    oracle_euclidean = float(
        oracle_result["euclidean_deviation"]
    )

    best_search_euclidean = float(
        best_search["euclidean_deviation"]
    )

    inverse_realised_direction = tuple(
        -float(value)
        for value in event_displacement
    )

    history_alignment_with_inverse_realised = (
        cosine_similarity(
            history_action,
            inverse_realised_direction,
        )
    )

    nominal_prediction = q7.predict_prestress_response(
        pre_event_image,
        nominal_event,
        run_id=(
            f"audit::{condition_name}::nominal_prediction"
        ),
    )

    realised_vs_nominal_cosine = cosine_similarity(
        nominal_prediction,
        event_displacement,
    )

    return {
        "condition": condition_name,

        "event_displacement_no_control": list(
            event_displacement
        ),

        "history_action": list(
            history_action
        ),

        "history_action_budget": action_budget,

        "history_post_displacement": list(
            history_result["post_displacement"]
        ),

        "algebraic_event_plus_action_prediction": list(
            algebraic_prediction
        ),

        "superposition_residual": list(
            superposition_residual
        ),

        "superposition_residual_norm": vector_norm(
            superposition_residual
        ),

        "exact_additive_superposition_within_tolerance": (
            vector_norm(superposition_residual)
            <= ZERO_TOL
        ),

        "history_action_alignment_with_inverse_realised_response": (
            history_alignment_with_inverse_realised
        ),

        "nominal_prediction_alignment_with_realised_response": (
            realised_vs_nominal_cosine
        ),

        "no_control_euclidean_deviation": vector_norm(
            event_displacement
        ),

        "history_action_euclidean_deviation": (
            history_euclidean
        ),

        "history_action_peak_abs_deviation": float(
            history_result["peak_abs_deviation"]
        ),

        "oracle_action": list(
            oracle_action
        ),

        "oracle_euclidean_deviation": (
            oracle_euclidean
        ),

        "oracle_peak_abs_deviation": float(
            oracle_result["peak_abs_deviation"]
        ),

        "history_minus_oracle_euclidean": (
            history_euclidean
            - oracle_euclidean
        ),

        "history_matches_oracle_within_tolerance": (
            abs(
                history_euclidean
                - oracle_euclidean
            )
            <= ZERO_TOL
        ),

        "equal_budget_direction_search": {
            "candidate_count": len(
                candidate_results
            ),
            "best_candidate": best_search,
            "history_minus_best_search_euclidean": (
                history_euclidean
                - best_search_euclidean
            ),
            "history_beats_or_matches_discrete_search": (
                history_euclidean
                <= best_search_euclidean
                + ZERO_TOL
            ),
        },
    }


# =============================================================================
# FULL AUDIT
# =============================================================================


def run_audit() -> dict[str, Any]:
    """
    Audit Q7 for objective/controller coupling.

    This audit does NOT ask whether the Q7 software is correct.
    It asks whether the observed prestress-layer improvement is partly or
    wholly expected from the mathematical construction:

        action = -g * predicted prestress displacement

    followed by evaluation with:

        ||post prestress - pre-event prestress||_2

    Critical questions
    ------------------

    1. Does the controlled post-event displacement equal:
           event displacement + applied action
       to numerical precision?

    2. Is the Q7 history action strongly aligned with the inverse realised
       displacement?

    3. How close is Q7 to an ex-post equal-budget oracle that directly uses
       the realised displacement?

    4. Does Q7 outperform a broad deterministic set of equal-budget directions?

    A positive superposition result would mean that reduction in the same
    prestress-distance objective is algebraically coupled to action direction.
    That would narrow the scientific interpretation of Q7 rather than invalidate
    its software result.
    """

    pre_event_image = (
        q7.build_history_conditioned_pre_event_image()
    )

    nominal_event = q7.build_events()[4]

    predicted_response = q7.predict_prestress_response(
        pre_event_image,
        nominal_event,
        run_id="q7_audit::history_prediction",
    )

    history_action = q7.bounded_predictive_action(
        predicted_response
    )

    conditions: dict[str, Any] = {}

    for factor in q7.CHALLENGE_FACTORS:
        condition_name = (
            f"challenge_{int(round(factor * 100)):03d}pct"
        )

        realised_event = q7.scale_challenge(
            nominal_event,
            factor,
            event_id=(
                f"{nominal_event.event_id}"
                f"__audit_realised_{factor:.2f}"
            ),
        )

        conditions[condition_name] = audit_condition(
            pre_event_image=pre_event_image,
            nominal_event=nominal_event,
            realised_event=realised_event,
            history_action=history_action,
            condition_name=condition_name,
        )

    all_superposition_exact = all(
        block[
            "exact_additive_superposition_within_tolerance"
        ]
        for block in conditions.values()
    )

    alignments = tuple(
        float(
            block[
                "history_action_alignment_with_inverse_realised_response"
            ]
        )
        for block in conditions.values()
    )

    oracle_gaps = tuple(
        float(
            block[
                "history_minus_oracle_euclidean"
            ]
        )
        for block in conditions.values()
    )

    search_flags = tuple(
        bool(
            block[
                "equal_budget_direction_search"
            ][
                "history_beats_or_matches_discrete_search"
            ]
        )
        for block in conditions.values()
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": CLAIM_SCOPE,

        "audit_target": (
            "q7_one_step_prestress_preconfiguration"
        ),

        "questions_under_audit": {
            "additive_superposition": True,
            "objective_controller_coupling": True,
            "equal_budget_oracle_gap": True,
            "equal_budget_direction_search": True,
        },

        "conditions": conditions,

        "central_results": {
            "additive_superposition_exact_all_conditions": (
                all_superposition_exact
            ),

            "minimum_alignment_with_inverse_realised_response": (
                min(alignments)
                if alignments
                else 0.0
            ),

            "mean_alignment_with_inverse_realised_response": (
                sum(alignments) / len(alignments)
                if alignments
                else 0.0
            ),

            "maximum_alignment_with_inverse_realised_response": (
                max(alignments)
                if alignments
                else 0.0
            ),

            "minimum_history_minus_oracle_euclidean": (
                min(oracle_gaps)
                if oracle_gaps
                else 0.0
            ),

            "maximum_history_minus_oracle_euclidean": (
                max(oracle_gaps)
                if oracle_gaps
                else 0.0
            ),

            "history_beats_or_matches_discrete_search_all_conditions": (
                all(search_flags)
            ),
        },

        # ---------------------------------------------------------------------
        # CLAIM BOUNDARIES
        # ---------------------------------------------------------------------

        "whole_system_stabilization_tested": False,
        "multi_step_temporal_image_tested": False,
        "objective_independent_stabilization_tested": False,
        "external_predictive_validity_tested": False,
        "biological_or_clinical_validity_tested": False,

        "interpretation": (
            "This audit examines whether the Q7 prestress-layer result is "
            "mathematically coupled to its controller and objective. Q7 builds "
            "a bounded action opposite to a predicted prestress displacement "
            "and evaluates post-event Euclidean deviation in the same prestress "
            "space. Exact additive superposition, strong alignment with the "
            "inverse realised response, or near-oracle performance would show "
            "that part of the observed Q7 improvement follows directly from "
            "this construction. Such a result would not invalidate Q7 as a "
            "controlled one-step prestress preconfiguration experiment, but it "
            "would prevent interpreting Q7 as objective-independent evidence "
            "for general predictive stabilization."
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
    data = run_audit()

    root = Path(__file__).resolve().parents[1]

    output_path = (
        root
        / "benchmark_results"
        / "q7_objective_independence_audit_v1.json"
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
    print("Saved Q7 objective-independence audit:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
