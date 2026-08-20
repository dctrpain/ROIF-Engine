from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import roif_predictive_stabilization_benchmark as q7


# =============================================================================
# METADATA
# =============================================================================

BENCHMARK_VERSION = (
    "roif_q7_off_nominal_predictive_action_transferability_audit_v2"
)

CLAIM_SCOPE = "computational_model_only"

ZERO_TOL = 1e-12


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def vector_add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")

    return tuple(
        float(a) + float(b)
        for a, b in zip(left, right)
    )


def vector_subtract(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")

    return tuple(
        float(a) - float(b)
        for a, b in zip(left, right)
    )


def vector_norm(
    values: Sequence[float],
) -> float:
    return math.sqrt(
        sum(
            float(value) ** 2
            for value in values
        )
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
    return vector_subtract(
        q7.prestress_vector(observed),
        q7.prestress_vector(reference),
    )


# =============================================================================
# OFF-NOMINAL FUTURE SCENARIOS
# =============================================================================


def build_future_scenarios() -> dict[
    str,
    tuple[Any, ...],
]:
    """
    Construct realised futures that differ from nominal E5 in different ways.

    IMPORTANT
    ---------

    The predictive action is ALWAYS constructed from nominal E5.

    The realised future is then changed independently.

    Scenario classes:
        1. exact nominal event
        2. intensity mismatch
        3. spatial mismatch
        4. directional/sign mismatch
        5. distributed/split perturbation
        6. altered adaptive-connection exposure
        7. multi-step future sequence
        8. reversed multi-step future sequence

    This remains a prestress-layer audit.

    It does NOT implement the full ROIF future Temporal-Image architecture.
    """

    nominal = q7.build_events()[4]

    if len(nominal.prestress_perturbations) != 1:
        raise RuntimeError(
            "Q7 audit expects nominal E5 to contain exactly "
            "one prestress perturbation"
        )

    nominal_perturbation = (
        nominal.prestress_perturbations[0]
    )

    # -------------------------------------------------------------------------
    # 1. EXACT NOMINAL
    # -------------------------------------------------------------------------

    exact_nominal = replace(
        nominal,
        event_id="audit_exact_nominal_E5",
    )

    # -------------------------------------------------------------------------
    # 2. INTENSITY MISMATCH
    # -------------------------------------------------------------------------

    intensity_low = q7.scale_challenge(
        nominal,
        0.80,
        event_id="audit_intensity_080",
    )

    intensity_high = q7.scale_challenge(
        nominal,
        1.20,
        event_id="audit_intensity_120",
    )

    # -------------------------------------------------------------------------
    # 3. SPATIAL MISMATCH
    #
    # Same perturbation magnitude, different node.
    # -------------------------------------------------------------------------

    spatial_shift_D = replace(
        nominal,
        event_id="audit_spatial_shift_D",
        prestress_perturbations=(
            replace(
                nominal_perturbation,
                perturbation_id=(
                    "audit_spatial_shift_D_perturbation"
                ),
                node_id="D",
                delta=float(
                    nominal_perturbation.delta
                ),
            ),
        ),
    )

    # -------------------------------------------------------------------------
    # 4. SIGN / DIRECTION MISMATCH
    # -------------------------------------------------------------------------

    opposite_sign_A = replace(
        nominal,
        event_id="audit_opposite_sign_A",
        prestress_perturbations=(
            replace(
                nominal_perturbation,
                perturbation_id=(
                    "audit_opposite_sign_A_perturbation"
                ),
                node_id="A",
                delta=(
                    -float(
                        nominal_perturbation.delta
                    )
                ),
            ),
        ),
    )

    # -------------------------------------------------------------------------
    # 5. DISTRIBUTED / SPLIT PERTURBATION
    #
    # Same total signed direct perturbation magnitude:
    #     +0.05 at A
    #     +0.05 at D
    #
    # instead of:
    #     +0.10 at A
    # -------------------------------------------------------------------------

    half_delta = (
        0.5
        * float(
            nominal_perturbation.delta
        )
    )

    split_A_D = replace(
        nominal,
        event_id="audit_split_A_D",
        prestress_perturbations=(
            replace(
                nominal_perturbation,
                perturbation_id=(
                    "audit_split_A"
                ),
                node_id="A",
                delta=half_delta,
            ),
            replace(
                nominal_perturbation,
                perturbation_id=(
                    "audit_split_D"
                ),
                node_id="D",
                delta=half_delta,
            ),
        ),
    )

    # -------------------------------------------------------------------------
    # 6. ALTERED CONNECTION-EXPOSURE PATTERN
    #
    # Direct prestress perturbation remains nominal at A, while the event changes
    # the adaptive connections differently before redistribution.
    #
    # evolve_system() updates adaptive connections before converting them into
    # current prestress-transfer paths.
    # -------------------------------------------------------------------------

    altered_exposures = []

    for index, exposure in enumerate(
        nominal.connection_exposures,
    ):
        if index == 0:
            altered_exposures.append(
                replace(
                    exposure,
                    exposure_id=(
                        "audit_altered_exposure_1"
                    ),
                    load=0.20,
                    strain=0.08,
                    activation=0.20,
                    damage=0.00,
                    recovery=0.00,
                )
            )
        else:
            altered_exposures.append(
                replace(
                    exposure,
                    exposure_id=(
                        "audit_altered_exposure_2"
                    ),
                    load=0.85,
                    strain=0.35,
                    activation=0.85,
                    damage=0.10,
                    recovery=0.00,
                )
            )

    altered_connection_exposure = replace(
        nominal,
        event_id=(
            "audit_altered_connection_exposure"
        ),
        connection_exposures=tuple(
            altered_exposures
        ),
    )

    # -------------------------------------------------------------------------
    # 7/8. MULTI-STEP FUTURES
    #
    # P_t is applied once before the sequence.
    #
    # This is NOT yet a predicted Temporal Image.
    # It only audits one-step nominal P_t against a realised multi-event future.
    # -------------------------------------------------------------------------

    second_D = replace(
        nominal,
        event_id="audit_second_event_D",
        connection_exposures=(),
        prestress_perturbations=(
            replace(
                nominal_perturbation,
                perturbation_id=(
                    "audit_second_event_D_perturbation"
                ),
                node_id="D",
                delta=float(
                    nominal_perturbation.delta
                ),
            ),
        ),
    )

    first_D = replace(
        second_D,
        event_id="audit_first_event_D",
    )

    second_A = replace(
        nominal,
        event_id="audit_second_event_A",
    )

    return {
        "exact_nominal": (
            exact_nominal,
        ),

        "intensity_080": (
            intensity_low,
        ),

        "intensity_120": (
            intensity_high,
        ),

        "spatial_shift_A_to_D": (
            spatial_shift_D,
        ),

        "opposite_sign_at_A": (
            opposite_sign_A,
        ),

        "split_A_D": (
            split_A_D,
        ),

        "altered_connection_exposure": (
            altered_connection_exposure,
        ),

        "sequence_A_then_D": (
            exact_nominal,
            second_D,
        ),

        "sequence_D_then_A": (
            first_D,
            second_A,
        ),
    }


# =============================================================================
# SEQUENCE EXECUTION
# =============================================================================


def evaluate_future(
    *,
    pre_event_image,
    events: Sequence[Any],
    action: Sequence[float] | None,
    label: str,
) -> dict[str, Any]:
    """
    Apply an optional P_t once before the realised future.

    Then evolve all realised events recursively.

    Report both:
        - final prestress deviation
        - trajectory-integrated prestress deviation

    The second metric prevents a two-step scenario from being reduced only to
    its final endpoint.
    """

    reference = pre_event_image

    if action is None:
        current = pre_event_image
    else:
        current = (
            q7.apply_action_to_system_image(
                pre_event_image,
                action,
                image_id=(
                    f"{label}::prepared"
                ),
                policy_name=(
                    f"audit_v2::{label}"
                ),
            )
        )

    trajectory = []

    for index, event in enumerate(
        events,
        start=1,
    ):
        current = q7.apply_event(
            current,
            event,
            run_id=(
                f"{label}::step_{index}"
            ),
        )

        displacement = (
            prestress_displacement(
                reference,
                current,
            )
        )

        trajectory.append(
            {
                "step": index,
                "event_id": event.event_id,
                "displacement": list(
                    displacement
                ),
                "euclidean_deviation": (
                    vector_norm(displacement)
                ),
            }
        )

    final_displacement = (
        prestress_displacement(
            reference,
            current,
        )
    )

    step_norms = tuple(
        float(
            item["euclidean_deviation"]
        )
        for item in trajectory
    )

    trajectory_integrated_deviation = (
        sum(step_norms)
    )

    trajectory_rms_deviation = (
        math.sqrt(
            sum(
                value * value
                for value in step_norms
            )
            / len(step_norms)
        )
        if step_norms
        else 0.0
    )

    return {
        "final_displacement": (
            final_displacement
        ),

        "final_euclidean_deviation": (
            vector_norm(
                final_displacement
            )
        ),

        "trajectory_integrated_deviation": (
            trajectory_integrated_deviation
        ),

        "trajectory_rms_deviation": (
            trajectory_rms_deviation
        ),

        "trajectory": trajectory,
    }


# =============================================================================
# EX-POST FINAL-STATE ORACLE
# =============================================================================


def build_equal_budget_final_inverse_oracle(
    realised_final_displacement: Sequence[float],
    *,
    action_budget: float,
) -> tuple[float, ...]:
    """
    Ex-post inverse of realised final prestress displacement.

    This is NOT a predictive policy.

    It is only a diagnostic reference for final-state objective coupling.
    """

    inverse = tuple(
        -float(value)
        for value
        in realised_final_displacement
    )

    return q7.rescale_action_to_norm(
        inverse,
        action_budget,
    )


# =============================================================================
# SINGLE SCENARIO AUDIT
# =============================================================================


def audit_scenario(
    *,
    pre_event_image,
    events: Sequence[Any],
    nominal_prediction: Sequence[float],
    history_action: Sequence[float],
    scenario_name: str,
) -> dict[str, Any]:

    no_control = evaluate_future(
        pre_event_image=pre_event_image,
        events=events,
        action=None,
        label=(
            f"{scenario_name}::no_control"
        ),
    )

    controlled = evaluate_future(
        pre_event_image=pre_event_image,
        events=events,
        action=history_action,
        label=(
            f"{scenario_name}::history_action"
        ),
    )

    realised_final = tuple(
        float(value)
        for value
        in no_control[
            "final_displacement"
        ]
    )

    inverse_realised = tuple(
        -float(value)
        for value in realised_final
    )

    nominal_realised_alignment = (
        cosine_similarity(
            nominal_prediction,
            realised_final,
        )
    )

    action_inverse_realised_alignment = (
        cosine_similarity(
            history_action,
            inverse_realised,
        )
    )

    no_final = float(
        no_control[
            "final_euclidean_deviation"
        ]
    )

    controlled_final = float(
        controlled[
            "final_euclidean_deviation"
        ]
    )

    no_integrated = float(
        no_control[
            "trajectory_integrated_deviation"
        ]
    )

    controlled_integrated = float(
        controlled[
            "trajectory_integrated_deviation"
        ]
    )

    final_improvement = (
        no_final
        - controlled_final
    )

    integrated_improvement = (
        no_integrated
        - controlled_integrated
    )

    action_budget = vector_norm(
        history_action
    )

    oracle_action = (
        build_equal_budget_final_inverse_oracle(
            realised_final,
            action_budget=action_budget,
        )
    )

    oracle = evaluate_future(
        pre_event_image=pre_event_image,
        events=events,
        action=oracle_action,
        label=(
            f"{scenario_name}::oracle"
        ),
    )

    oracle_final = float(
        oracle[
            "final_euclidean_deviation"
        ]
    )

    # -------------------------------------------------------------------------
    # ADDITIVE COUPLING DIAGNOSTIC
    #
    # We do not assume this must hold for off-nominal or multi-step futures.
    # We measure it.
    # -------------------------------------------------------------------------

    algebraic_prediction = vector_add(
        realised_final,
        history_action,
    )

    controlled_final_vector = tuple(
        float(value)
        for value
        in controlled[
            "final_displacement"
        ]
    )

    superposition_residual = (
        vector_subtract(
            controlled_final_vector,
            algebraic_prediction,
        )
    )

    return {
        "scenario": scenario_name,

        "event_ids": [
            event.event_id
            for event in events
        ],

        "event_count": len(events),

        "no_control": {
            **no_control,
            "final_displacement": list(
                no_control[
                    "final_displacement"
                ]
            ),
        },

        "history_action_control": {
            **controlled,
            "final_displacement": list(
                controlled[
                    "final_displacement"
                ]
            ),
        },

        "nominal_prediction_alignment_with_realised_final_response": (
            nominal_realised_alignment
        ),

        "history_action_alignment_with_inverse_realised_final_response": (
            action_inverse_realised_alignment
        ),

        "final_euclidean_improvement": (
            final_improvement
        ),

        "final_euclidean_improved": (
            controlled_final
            < no_final
        ),

        "trajectory_integrated_improvement": (
            integrated_improvement
        ),

        "trajectory_integrated_improved": (
            controlled_integrated
            < no_integrated
        ),

        "relative_final_improvement": (
            final_improvement / no_final
            if no_final > 0.0
            else 0.0
        ),

        "relative_integrated_improvement": (
            integrated_improvement
            / no_integrated
            if no_integrated > 0.0
            else 0.0
        ),

        "equal_budget_final_inverse_oracle": {
            "action": list(
                oracle_action
            ),
            "final_euclidean_deviation": (
                oracle_final
            ),
            "history_minus_oracle_final_euclidean": (
                controlled_final
                - oracle_final
            ),
        },

        "additive_coupling_diagnostic": {
            "algebraic_realised_plus_action": list(
                algebraic_prediction
            ),

            "controlled_final_displacement": list(
                controlled_final_vector
            ),

            "residual": list(
                superposition_residual
            ),

            "residual_norm": (
                vector_norm(
                    superposition_residual
                )
            ),

            "exact_within_tolerance": (
                vector_norm(
                    superposition_residual
                )
                <= ZERO_TOL
            ),
        },
    }


# =============================================================================
# FULL AUDIT
# =============================================================================


def run_audit() -> dict[str, Any]:
    """
    Q7 off-nominal transferability audit.

    The action P_t is constructed ONCE from:

        current history-conditioned SystemImage
        + nominal expected E5

    That same P_t is then evaluated against several realised futures that may
    differ from E5 in:

        magnitude,
        spatial location,
        sign,
        perturbation distribution,
        connection-exposure pattern,
        event number,
        event order.

    This audit therefore asks a narrower and more useful question than v1:

        How scenario-specific is the observed benefit of Q7's one-step
        prestress preconfiguration?

    It still does NOT test the full ROIF predictive architecture because:

        - the predicted object is still a one-step prestress projection;
        - P_t still modifies only prestress;
        - the objective is still prestress-based;
        - no future Temporal Image is reconstructed by Q itself;
        - no general preconfiguration operator Pi is implemented.
    """

    pre_event_image = (
        q7.build_history_conditioned_pre_event_image()
    )

    nominal_event = q7.build_events()[4]

    nominal_prediction = (
        q7.predict_prestress_response(
            pre_event_image,
            nominal_event,
            run_id=(
                "q7_off_nominal_v2::"
                "nominal_prediction"
            ),
        )
    )

    history_action = (
        q7.bounded_predictive_action(
            nominal_prediction
        )
    )

    scenarios = build_future_scenarios()

    results: dict[str, Any] = {}

    for name, events in scenarios.items():
        results[name] = audit_scenario(
            pre_event_image=pre_event_image,
            events=events,
            nominal_prediction=nominal_prediction,
            history_action=history_action,
            scenario_name=name,
        )

    final_successes = [
        name
        for name, block
        in results.items()
        if block[
            "final_euclidean_improved"
        ]
    ]

    final_failures = [
        name
        for name, block
        in results.items()
        if not block[
            "final_euclidean_improved"
        ]
    ]

    integrated_successes = [
        name
        for name, block
        in results.items()
        if block[
            "trajectory_integrated_improved"
        ]
    ]

    integrated_failures = [
        name
        for name, block
        in results.items()
        if not block[
            "trajectory_integrated_improved"
        ]
    ]

    alignments = tuple(
        float(
            block[
                "nominal_prediction_alignment_with_realised_final_response"
            ]
        )
        for block in results.values()
    )

    relative_final_improvements = tuple(
        float(
            block[
                "relative_final_improvement"
            ]
        )
        for block in results.values()
    )

    additive_flags = tuple(
        bool(
            block[
                "additive_coupling_diagnostic"
            ][
                "exact_within_tolerance"
            ]
        )
        for block in results.values()
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": CLAIM_SCOPE,

        "audit_target": (
            "q7_one_step_nominal_E5_prestress_preconfiguration"
        ),

        "nominal_expected_event_id": (
            nominal_event.event_id
        ),

        "nominal_prediction": list(
            nominal_prediction
        ),

        "history_action": list(
            history_action
        ),

        "history_action_budget": (
            vector_norm(
                history_action
            )
        ),

        "scenario_count": len(
            results
        ),

        "scenarios": results,

        "central_results": {
            "final_objective_success_count": (
                len(final_successes)
            ),

            "final_objective_failure_count": (
                len(final_failures)
            ),

            "final_objective_success_scenarios": (
                final_successes
            ),

            "final_objective_failure_scenarios": (
                final_failures
            ),

            "trajectory_objective_success_count": (
                len(integrated_successes)
            ),

            "trajectory_objective_failure_count": (
                len(integrated_failures)
            ),

            "trajectory_objective_success_scenarios": (
                integrated_successes
            ),

            "trajectory_objective_failure_scenarios": (
                integrated_failures
            ),

            "minimum_nominal_realised_alignment": (
                min(alignments)
                if alignments
                else 0.0
            ),

            "maximum_nominal_realised_alignment": (
                max(alignments)
                if alignments
                else 0.0
            ),

            "minimum_relative_final_improvement": (
                min(
                    relative_final_improvements
                )
                if relative_final_improvements
                else 0.0
            ),

            "maximum_relative_final_improvement": (
                max(
                    relative_final_improvements
                )
                if relative_final_improvements
                else 0.0
            ),

            "additive_coupling_exact_all_scenarios": (
                all(additive_flags)
            ),
        },

        # ---------------------------------------------------------------------
        # EXPLICIT CLAIM BOUNDARIES
        # ---------------------------------------------------------------------

        "off_nominal_transferability_tested": True,

        "full_predictive_preconfiguration_architecture_tested": False,
        "future_temporal_image_reconstruction_tested": False,
        "whole_system_image_stabilization_tested": False,
        "objective_independent_stabilization_tested": False,
        "external_predictive_validity_tested": False,
        "universal_stability_tested": False,
        "biological_or_clinical_validity_tested": False,

        "interpretation": (
            "This audit evaluates the scenario transferability of a single "
            "Q7 prestress-layer action constructed from the nominal expected "
            "E5 event. The same action is applied to realised futures that "
            "differ in intensity, spatial location, sign, perturbation "
            "distribution, adaptive-connection exposure, event number, and "
            "event order. Improvement or deterioration is reported rather "
            "than assumed. The audit remains restricted to the Q7 one-step "
            "prestress projection and does not constitute a test of ROIF's "
            "full future-Temporal-Image reconstruction or general predictive "
            "preconfiguration architecture."
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
        / "q7_off_nominal_transferability_audit_v2.json"
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
    print("Saved Q7 off-nominal transferability audit:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
