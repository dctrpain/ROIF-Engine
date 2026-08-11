"""
Regression tests for:
    experiments.roif_physical_history_integration_benchmark

These tests lock the qualitative and deterministic behavior of the
physical-history integration benchmark without overfitting to every
floating-point value.
"""

from __future__ import annotations

import math

import pytest

from experiments.roif_physical_history_integration_benchmark import (
    BENCHMARK_VERSION,
    build_source_image,
    event_a,
    event_b,
    run_benchmark,
    run_global_redistribution,
    run_order_dependence,
    run_repeated_event_dependence,
)


@pytest.fixture(scope="module")
def benchmark():
    return run_benchmark()


@pytest.fixture(scope="module")
def order_result(benchmark):
    return benchmark["order_dependence"]


@pytest.fixture(scope="module")
def redistribution_result(benchmark):
    return benchmark["global_redistribution"]


@pytest.fixture(scope="module")
def repeated_result(benchmark):
    return benchmark["repeated_event_state_dependence"]


def test_benchmark_version():
    assert BENCHMARK_VERSION == "roif_physical_history_integration_v1"


def test_benchmark_reports_version(benchmark):
    assert benchmark["benchmark_version"] == BENCHMARK_VERSION


def test_claim_scope_is_computational_only(benchmark):
    assert benchmark["claim_scope"] == "computational_model_only"


def test_no_clinical_validation_claim(benchmark):
    assert benchmark["clinical_validation_claimed"] is False


def test_no_biological_truth_claim(benchmark):
    assert benchmark["biological_truth_claimed"] is False


def test_no_causal_truth_claim(benchmark):
    assert benchmark["causal_truth_claimed"] is False


def test_source_image_starts_at_revision_zero():
    assert build_source_image().revision == 0


def test_source_image_has_three_prestress_nodes():
    assert len(build_source_image().prestress_nodes) == 3


def test_source_image_has_two_adaptive_connections():
    assert len(build_source_image().adaptive_connections) == 2


def test_event_a_targets_connection_a_b():
    event = event_a()
    assert event.connection_exposures[0].connection_id == "A_B"


def test_event_a_directly_perturbs_a():
    event = event_a()
    assert event.prestress_perturbations[0].node_id == "A"


def test_event_b_targets_connection_b_c():
    event = event_b()
    assert event.connection_exposures[0].connection_id == "B_C"


def test_event_b_directly_perturbs_b():
    event = event_b()
    assert event.prestress_perturbations[0].node_id == "B"


# ---------------------------------------------------------------------------
# Q1: ORDER / PATH DEPENDENCE
# ---------------------------------------------------------------------------

def test_order_dependence_detected(order_result):
    assert order_result["path_dependent"] is True


def test_order_dependence_distance_positive(order_result):
    assert order_result["final_state_distance"] > 0.0


def test_order_dependence_distance_is_finite(order_result):
    assert math.isfinite(order_result["final_state_distance"])


def test_ab_final_revision_is_two(order_result):
    assert order_result["ab_final_revision"] == 2


def test_ba_final_revision_is_two(order_result):
    assert order_result["ba_final_revision"] == 2


def test_ab_and_ba_signatures_differ(order_result):
    assert order_result["ab_signature"] != order_result["ba_signature"]


def test_ab_and_ba_same_measure_layer(order_result):
    assert (
        order_result["ab_signature"][2]
        == order_result["ba_signature"][2]
    )


def test_ab_and_ba_same_final_adaptive_connection_layer(order_result):
    assert (
        order_result["ab_signature"][4]
        == order_result["ba_signature"][4]
    )


def test_ab_and_ba_differ_in_final_prestress_layer(order_result):
    assert (
        order_result["ab_signature"][3]
        != order_result["ba_signature"][3]
    )


def test_order_dependence_is_not_only_trace_id_difference(order_result):
    """
    Regression guard:
    path dependence must remain visible in the physical state itself,
    not merely because trace IDs contain AB/BA labels.
    """
    ab_prestress = order_result["ab_signature"][3]
    ba_prestress = order_result["ba_signature"][3]

    assert ab_prestress != ba_prestress


def test_order_dependence_distance_materially_nonzero(order_result):
    assert order_result["final_state_distance"] > 1e-6


# ---------------------------------------------------------------------------
# Q2: LOCAL EVENT -> GLOBAL PRESTRESS REDISTRIBUTION
# ---------------------------------------------------------------------------

def test_global_redistribution_detected(redistribution_result):
    assert (
        redistribution_result["global_redistribution_detected"]
        is True
    )


def test_only_a_is_directly_perturbed(redistribution_result):
    assert redistribution_result["directly_perturbed_nodes"] == ["A"]


def test_direct_node_a_delta_is_negative(redistribution_result):
    assert redistribution_result["all_node_deltas"]["A"] < 0.0


def test_downstream_node_b_changes(redistribution_result):
    assert (
        abs(
            redistribution_result[
                "downstream_changed_nodes"
            ]["B"]
        )
        > 0.0
    )


def test_downstream_node_c_changes(redistribution_result):
    assert (
        abs(
            redistribution_result[
                "downstream_changed_nodes"
            ]["C"]
        )
        > 0.0
    )


def test_downstream_nodes_are_not_directly_perturbed(redistribution_result):
    direct = set(
        redistribution_result["directly_perturbed_nodes"]
    )
    downstream = set(
        redistribution_result["downstream_changed_nodes"]
    )

    assert direct.isdisjoint(downstream)


def test_redistribution_reaches_two_downstream_nodes(redistribution_result):
    assert set(
        redistribution_result["downstream_changed_nodes"]
    ) == {"B", "C"}


def test_prestress_change_norm_positive(redistribution_result):
    assert redistribution_result["prestress_change_norm"] > 0.0


def test_prestress_change_norm_finite(redistribution_result):
    assert math.isfinite(
        redistribution_result["prestress_change_norm"]
    )


def test_all_expected_nodes_report_deltas(redistribution_result):
    assert set(
        redistribution_result["all_node_deltas"]
    ) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Q3: SAME EVENT, DIFFERENT STATE -> DIFFERENT RESPONSE
# ---------------------------------------------------------------------------

def test_repeated_event_state_dependence_detected(repeated_result):
    assert repeated_result["same_event_different_response"] is True


def test_repeated_event_response_distance_positive(repeated_result):
    assert repeated_result["response_distance"] > 0.0


def test_repeated_event_response_distance_finite(repeated_result):
    assert math.isfinite(repeated_result["response_distance"])


def test_second_event_connection_response_differs(repeated_result):
    assert (
        repeated_result["first_response"]["connection_change_norm"]
        != pytest.approx(
            repeated_result[
                "second_response"
            ]["connection_change_norm"]
        )
    )


def test_second_event_prestress_response_differs(repeated_result):
    assert (
        repeated_result["first_response"]["prestress_change_norm"]
        != pytest.approx(
            repeated_result[
                "second_response"
            ]["prestress_change_norm"]
        )
    )


def test_second_event_trace_magnitude_differs(repeated_result):
    assert (
        repeated_result["first_response"]["trace_magnitude"]
        != pytest.approx(
            repeated_result[
                "second_response"
            ]["trace_magnitude"]
        )
    )


def test_second_response_not_simple_copy_of_first(repeated_result):
    first = repeated_result["first_response"]
    second = repeated_result["second_response"]

    assert first != second


def test_repeated_event_final_revision_is_two(repeated_result):
    assert repeated_result["final_revision"] == 2


def test_repeated_event_accumulates_three_traces(repeated_result):
    assert repeated_result["final_trace_count"] == 3


def test_connection_change_can_increase_on_second_exposure(repeated_result):
    assert (
        repeated_result["second_response"]["connection_change_norm"]
        >
        repeated_result["first_response"]["connection_change_norm"]
    )


def test_prestress_change_can_decrease_on_second_exposure(repeated_result):
    assert (
        repeated_result["second_response"]["prestress_change_norm"]
        <
        repeated_result["first_response"]["prestress_change_norm"]
    )


def test_repeated_response_is_not_uniform_scalar_amplification(repeated_result):
    """
    One response component increases while another decreases.
    This guards against reducing state dependence to a single scalar multiplier.
    """
    connection_ratio = (
        repeated_result["second_response"]["connection_change_norm"]
        / repeated_result["first_response"]["connection_change_norm"]
    )

    prestress_ratio = (
        repeated_result["second_response"]["prestress_change_norm"]
        / repeated_result["first_response"]["prestress_change_norm"]
    )

    assert connection_ratio > 1.0
    assert prestress_ratio < 1.0
    assert connection_ratio != pytest.approx(prestress_ratio)


# ---------------------------------------------------------------------------
# DETERMINISM / REGRESSION
# ---------------------------------------------------------------------------

def test_order_dependence_is_deterministic():
    assert run_order_dependence() == run_order_dependence()


def test_global_redistribution_is_deterministic():
    assert run_global_redistribution() == run_global_redistribution()


def test_repeated_event_dependence_is_deterministic():
    assert (
        run_repeated_event_dependence()
        == run_repeated_event_dependence()
    )


def test_full_benchmark_is_deterministic():
    assert run_benchmark() == run_benchmark()


def test_regression_order_distance(order_result):
    assert order_result["final_state_distance"] == pytest.approx(
        0.011814192037381752,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_redistribution_norm(redistribution_result):
    assert redistribution_result["prestress_change_norm"] == pytest.approx(
        0.30046937477559266,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_repeat_response_distance(repeated_result):
    assert repeated_result["response_distance"] == pytest.approx(
        0.06723725281827213,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_a_delta(redistribution_result):
    assert redistribution_result["all_node_deltas"]["A"] == pytest.approx(
        -0.25,
        rel=1e-12,
        abs=1e-12,
    )


def test_regression_b_delta(redistribution_result):
    assert redistribution_result["all_node_deltas"]["B"] == pytest.approx(
        -0.1333430947366333,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_c_delta(redistribution_result):
    assert redistribution_result["all_node_deltas"]["C"] == pytest.approx(
        -0.10000732105247498,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_first_trace_magnitude(repeated_result):
    assert repeated_result["first_response"]["trace_magnitude"] == pytest.approx(
        0.4167835870428668,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_second_trace_magnitude(repeated_result):
    assert repeated_result["second_response"]["trace_magnitude"] == pytest.approx(
        0.4332537396233776,
        rel=1e-9,
        abs=1e-12,
    )


def test_all_three_core_findings_hold_together(
    order_result,
    redistribution_result,
    repeated_result,
):
    assert (
        order_result["path_dependent"]
        and redistribution_result["global_redistribution_detected"]
        and repeated_result["same_event_different_response"]
    )
