from __future__ import annotations

import pytest

from experiments import roif_predictive_stabilization_benchmark as q7
from experiments import roif_q7_objective_independence_audit as q7_objective
from experiments import roif_q7_off_nominal_transferability_audit as q7_transfer
from experiments import roif_q8_history_conditioned_redistribution_benchmark as q8_v1
from experiments import roif_q8_history_conditioned_redistribution_matched_state_benchmark as q8_v2
from experiments import roif_multilayer_temporal_image_trajectory_benchmark as temporal
from experiments import roif_matched_state_history_operator_identifiability_benchmark as matched_operator


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def q7_result():
    return q7.run_benchmark()


@pytest.fixture(scope="module")
def q7_objective_result():
    return q7_objective.run_audit()


@pytest.fixture(scope="module")
def q7_transfer_result():
    return q7_transfer.run_audit()


@pytest.fixture(scope="module")
def q8_v1_result():
    return q8_v1.run_benchmark()


@pytest.fixture(scope="module")
def q8_v2_result():
    return q8_v2.run_benchmark()


@pytest.fixture(scope="module")
def temporal_result():
    return temporal.run_benchmark()


@pytest.fixture(scope="module")
def matched_operator_result():
    return matched_operator.run_benchmark()


# =============================================================================
# Q7 — ONE-STEP PRESTRESS PRECONFIGURATION ONLY
# =============================================================================


def test_q7_tests_only_one_step_prestress_preconfiguration(
    q7_result,
):
    assert q7_result[
        "prestress_layer_one_step_stabilization_tested"
    ] is True


def test_q7_does_not_claim_whole_system_image_stabilization(
    q7_result,
):
    assert q7_result[
        "whole_system_image_stabilization_claimed"
    ] is False


def test_q7_does_not_claim_multi_step_temporal_image_stabilization(
    q7_result,
):
    assert q7_result[
        "multi_step_temporal_image_stabilization_claimed"
    ] is False


def test_q7_does_not_claim_general_optimal_policy(
    q7_result,
):
    assert q7_result[
        "general_optimal_stabilization_policy_claimed"
    ] is False


def test_q7_mechanism_is_explicitly_prestress_restricted(
    q7_result,
):
    assert q7_result[
        "mechanism_under_test"
    ] == (
        "history_conditioned_one_step_prestress_preconfiguration"
    )


# =============================================================================
# Q7 OBJECTIVE-COUPLING AUDIT
# =============================================================================


def test_q7_objective_audit_detects_additive_prestress_channel(
    q7_objective_result,
):
    assert q7_objective_result[
        "central_results"
    ][
        "additive_superposition_exact_all_conditions"
    ] is True


def test_q7_objective_audit_does_not_claim_objective_independent_stabilization(
    q7_objective_result,
):
    assert q7_objective_result[
        "objective_independent_stabilization_tested"
    ] is False


def test_q7_objective_audit_does_not_claim_whole_system_stabilization(
    q7_objective_result,
):
    assert q7_objective_result[
        "whole_system_stabilization_tested"
    ] is False


def test_q7_objective_audit_does_not_claim_temporal_image_prediction(
    q7_objective_result,
):
    assert q7_objective_result[
        "multi_step_temporal_image_tested"
    ] is False


def test_q7_objective_audit_interpretation_is_limited_to_q7(
    q7_objective_result,
):
    text = q7_objective_result[
        "interpretation"
    ].lower()

    assert "q7" in text
    assert "prestress-layer" in text
    assert "general predictive stabilization" in text


# =============================================================================
# Q7 OFF-NOMINAL TRANSFERABILITY
# =============================================================================


def test_q7_transferability_is_explicitly_off_nominal(
    q7_transfer_result,
):
    assert q7_transfer_result[
        "off_nominal_transferability_tested"
    ] is True


def test_q7_transferability_does_not_claim_full_predictive_architecture(
    q7_transfer_result,
):
    assert q7_transfer_result[
        "full_predictive_preconfiguration_architecture_tested"
    ] is False


def test_q7_transferability_does_not_claim_future_temporal_image_reconstruction(
    q7_transfer_result,
):
    assert q7_transfer_result[
        "future_temporal_image_reconstruction_tested"
    ] is False


def test_q7_transferability_does_not_claim_whole_system_stabilization(
    q7_transfer_result,
):
    assert q7_transfer_result[
        "whole_system_image_stabilization_tested"
    ] is False


def test_q7_transferability_contains_both_success_and_failure_cases(
    q7_transfer_result,
):
    central = q7_transfer_result[
        "central_results"
    ]

    assert central[
        "final_objective_success_count"
    ] > 0

    assert central[
        "final_objective_failure_count"
    ] > 0


def test_q7_transferability_is_scenario_sensitive_not_universal(
    q7_transfer_result,
):
    central = q7_transfer_result[
        "central_results"
    ]

    assert central[
        "final_objective_success_count"
    ] < q7_transfer_result[
        "scenario_count"
    ]


# =============================================================================
# Q8-v1 — HISTORICAL RECONFIGURATION, NOT STABILIZATION
# =============================================================================


def test_q8_v1_tests_history_conditioned_connection_change(
    q8_v1_result,
):
    assert q8_v1_result[
        "history_conditioned_connection_change_tested"
    ] is True


def test_q8_v1_tests_matched_event_redistribution_change(
    q8_v1_result,
):
    assert q8_v1_result[
        "matched_event_redistribution_change_tested"
    ] is True


def test_q8_v1_does_not_claim_whole_system_stability(
    q8_v1_result,
):
    assert q8_v1_result[
        "whole_system_stability_claimed"
    ] is False


def test_q8_v1_does_not_claim_energy_dissipation(
    q8_v1_result,
):
    assert q8_v1_result[
        "energy_dissipation_tested"
    ] is False


def test_q8_v1_does_not_claim_literal_absorption(
    q8_v1_result,
):
    assert q8_v1_result[
        "literal_absorption_tested"
    ] is False


def test_q8_v1_does_not_claim_failure_threshold(
    q8_v1_result,
):
    assert q8_v1_result[
        "injury_or_failure_threshold_tested"
    ] is False


def test_q8_v1_does_not_claim_predictive_preconfiguration(
    q8_v1_result,
):
    assert q8_v1_result[
        "predictive_preconfiguration_tested"
    ] is False


# =============================================================================
# Q8-v2 — MATCHED-STATE CONNECTION EFFECT
# =============================================================================


def test_q8_v2_establishes_connection_effect_at_matched_prestress(
    q8_v2_result,
):
    assert q8_v2_result[
        "central_results"
    ][
        "connection_state_changes_response_at_matched_prestress"
    ] is True


def test_q8_v2_explicitly_matches_prestress(
    q8_v2_result,
):
    assert q8_v2_result[
        "state_control_audit"
    ][
        "baseline_prestress_matches_connection_only_control"
    ] is True


def test_q8_v2_does_not_reinterpret_redistribution_as_stabilization(
    q8_v2_result,
):
    assert q8_v2_result[
        "stabilization_tested"
    ] is False


def test_q8_v2_does_not_claim_absorption(
    q8_v2_result,
):
    assert q8_v2_result[
        "literal_absorption_tested"
    ] is False


def test_q8_v2_does_not_claim_predictive_preconfiguration(
    q8_v2_result,
):
    assert q8_v2_result[
        "predictive_preconfiguration_tested"
    ] is False


# =============================================================================
# TEMPORAL MULTILAYER BENCHMARK
# =============================================================================


def test_temporal_benchmark_evaluates_realised_multilayer_sequences(
    temporal_result,
):
    assert temporal_result[
        "realised_multilayer_systemimage_sequence_evaluated"
    ] is True


def test_temporal_benchmark_does_not_collapse_layers_to_one_metric(
    temporal_result,
):
    assert temporal_result[
        "heterogeneous_layers_collapsed_to_single_metric"
    ] is False


def test_temporal_benchmark_does_not_claim_predicted_temporal_image(
    temporal_result,
):
    assert temporal_result[
        "predicted_temporal_image_reconstructed"
    ] is False


def test_temporal_benchmark_does_not_claim_temporal_image_prediction_validity(
    temporal_result,
):
    assert temporal_result[
        "future_temporal_image_prediction_validated"
    ] is False


def test_temporal_benchmark_does_not_claim_general_preconfiguration_operator(
    temporal_result,
):
    assert temporal_result[
        "general_preconfiguration_operator_tested"
    ] is False


def test_temporal_benchmark_can_show_same_endpoint_but_different_trajectory(
    temporal_result,
):
    central = temporal_result[
        "central_results"
    ]

    assert central[
        "no_control_same_final_prestress_within_tolerance"
    ] is True

    assert central[
        "no_control_multilayer_trajectory_differs_by_order"
    ] is True

    assert central[
        "no_control_final_prestress_endpoint_insufficient"
    ] is True


# =============================================================================
# MATCHED-CURRENT-STATE HISTORY / OPERATOR IDENTIFIABILITY
# =============================================================================


def test_matched_operator_control_establishes_same_current_state(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "matched_current_state_established"
    ] is True


def test_matched_operator_control_establishes_different_history(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "different_ordered_history_established"
    ] is True


def test_matched_operator_control_establishes_different_canonical_order(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "different_canonical_event_order_established"
    ] is True


def test_matched_operator_control_applies_same_probe(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "same_probe_applied"
    ] is True


def test_matched_operator_control_detects_no_history_specific_transition_effect(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "history_specific_transition_effect_detected"
    ] is False


def test_matched_operator_transition_distance_is_zero(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "transition_response_distance"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_matched_operator_post_probe_current_state_remains_equal(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "post_probe_current_state_equal"
    ] is True


def test_matched_operator_histories_remain_different_after_probe(
    matched_operator_result,
):
    assert matched_operator_result[
        "central_results"
    ][
        "post_probe_histories_remain_different"
    ] is True


def test_matched_operator_does_not_claim_absence_of_architectural_history_effect(
    matched_operator_result,
):
    assert matched_operator_result[
        "absence_of_architectural_history_effect_claimed"
    ] is False


def test_matched_operator_does_not_claim_history_conditioned_operator_effect(
    matched_operator_result,
):
    assert matched_operator_result[
        "history_conditioned_operator_effect_claimed"
    ] is False


# =============================================================================
# ANTI-REDUCTION CLAIM MAP
# =============================================================================


def test_history_reconfiguration_is_demonstrated(
    q8_v1_result,
    q8_v2_result,
):
    assert q8_v1_result[
        "history_conditioned_connection_change_tested"
    ] is True

    assert q8_v2_result[
        "central_results"
    ][
        "connection_state_changes_response_at_matched_prestress"
    ] is True


def test_independent_history_operator_effect_is_not_demonstrated(
    matched_operator_result,
):
    assert matched_operator_result[
        "matched_state_operator_identifiability_test_performed"
    ] is True

    assert matched_operator_result[
        "central_results"
    ][
        "history_specific_transition_effect_detected"
    ] is False


def test_one_step_prestress_preconfiguration_is_demonstrated(
    q7_result,
):
    assert q7_result[
        "prestress_layer_one_step_stabilization_tested"
    ] is True


def test_whole_system_image_stabilization_is_not_demonstrated(
    q7_result,
    q7_transfer_result,
):
    assert q7_result[
        "whole_system_image_stabilization_claimed"
    ] is False

    assert q7_transfer_result[
        "whole_system_image_stabilization_tested"
    ] is False


def test_temporal_image_prediction_is_not_demonstrated(
    q7_result,
    q7_transfer_result,
    temporal_result,
):
    assert q7_result[
        "multi_step_temporal_image_stabilization_claimed"
    ] is False

    assert q7_transfer_result[
        "future_temporal_image_reconstruction_tested"
    ] is False

    assert temporal_result[
        "predicted_temporal_image_reconstructed"
    ] is False


def test_q7_objective_coupling_is_not_generalized_to_roif(
    q7_objective_result,
):
    assert q7_objective_result[
        "objective_independent_stabilization_tested"
    ] is False

    assert q7_objective_result[
        "whole_system_stabilization_tested"
    ] is False


def test_q8_redistribution_is_not_generalized_to_stability(
    q8_v1_result,
    q8_v2_result,
):
    assert q8_v1_result[
        "whole_system_stability_claimed"
    ] is False

    assert q8_v2_result[
        "stabilization_tested"
    ] is False


# =============================================================================
# CENTRAL ARCHITECTURE-BOUNDARY REGRESSION GUARD
# =============================================================================


def test_roif_architecture_claim_boundaries_hold_together(
    q7_result,
    q7_objective_result,
    q7_transfer_result,
    q8_v1_result,
    q8_v2_result,
    temporal_result,
    matched_operator_result,
):
    """
    Central anti-reduction regression guard.

    This test intentionally separates:

        demonstrated mechanisms

    from:

        broader ROIF architectural objects not yet demonstrated by these
        specific benchmarks.

    If this test fails in the future, the failure should be reviewed
    scientifically before changing the assertion.
    """

    # -------------------------------------------------------------------------
    # DEMONSTRATED
    # -------------------------------------------------------------------------

    assert q8_v1_result[
        "history_conditioned_connection_change_tested"
    ] is True

    assert q8_v2_result[
        "central_results"
    ][
        "connection_state_changes_response_at_matched_prestress"
    ] is True

    assert q7_result[
        "prestress_layer_one_step_stabilization_tested"
    ] is True

    assert temporal_result[
        "realised_multilayer_systemimage_sequence_evaluated"
    ] is True

    # -------------------------------------------------------------------------
    # NOT DEMONSTRATED / NOT CLAIMED
    # -------------------------------------------------------------------------

    assert matched_operator_result[
        "central_results"
    ][
        "history_specific_transition_effect_detected"
    ] is False

    assert matched_operator_result[
        "absence_of_architectural_history_effect_claimed"
    ] is False

    assert q7_result[
        "whole_system_image_stabilization_claimed"
    ] is False

    assert q7_result[
        "multi_step_temporal_image_stabilization_claimed"
    ] is False

    assert q7_transfer_result[
        "future_temporal_image_reconstruction_tested"
    ] is False

    assert temporal_result[
        "predicted_temporal_image_reconstructed"
    ] is False

    assert temporal_result[
        "general_preconfiguration_operator_tested"
    ] is False

    assert q7_objective_result[
        "objective_independent_stabilization_tested"
    ] is False

    assert q8_v1_result[
        "whole_system_stability_claimed"
    ] is False

    assert q8_v2_result[
        "stabilization_tested"
    ] is False

    assert q8_v1_result[
        "literal_absorption_tested"
    ] is False

    assert q8_v1_result[
        "energy_dissipation_tested"
    ] is False
