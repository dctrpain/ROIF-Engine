from __future__ import annotations

import math

import pytest

from experiments import (
    roif_q8_history_conditioned_redistribution_matched_state_benchmark
    as bench
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def result():
    return bench.run_benchmark()


@pytest.fixture(scope="module")
def central(result):
    return result["central_results"]


@pytest.fixture(scope="module")
def audit(result):
    return result["state_control_audit"]


@pytest.fixture(scope="module")
def factorial(result):
    return result["factorial_response_analysis"]


@pytest.fixture(scope="module")
def responses(result):
    return result["matched_challenge_responses"]


# =============================================================================
# BASIC BENCHMARK IDENTITY
# =============================================================================


def test_benchmark_version(result):
    assert result["benchmark_version"] == (
        "roif_q8_history_conditioned_redistribution_matched_state_v2"
    )


def test_claim_scope_is_computational_only(result):
    assert result["claim_scope"] == "computational_model_only"


def test_mechanism_under_test(result):
    assert result["mechanism_under_test"] == (
        "history_conditioned_redistribution_reconfiguration"
    )


def test_experimental_design(result):
    assert result["experimental_design"] == (
        "2x2_matched_current_state_layer_decomposition"
    )


def test_parent_q8_v1_is_recorded(result):
    assert result["q8_v1_parent_version"] == (
        "roif_q8_history_conditioned_redistribution_capacity_v1"
    )


# =============================================================================
# FACTORIAL DESIGN
# =============================================================================


def test_factorial_prestress_levels(result):
    assert result["factors"]["prestress"] == [
        "baseline_p0",
        "conditioned_pH",
    ]


def test_factorial_connection_levels(result):
    assert result["factors"]["adaptive_connections"] == [
        "baseline_theta0",
        "conditioned_thetaH",
    ]


def test_four_factorial_states_present(result):
    assert set(result["matched_challenge_responses"]) == {
        "B00_baseline_p0_theta0",
        "C01_baseline_prestress_conditioned_connections",
        "C10_conditioned_prestress_baseline_connections",
        "H11_conditioned_prestress_conditioned_connections",
    }


# =============================================================================
# STATE CONTROL AUDIT
# =============================================================================


def test_baseline_prestress_matches_connection_only_control(audit):
    assert audit[
        "baseline_prestress_matches_connection_only_control"
    ] is True


def test_conditioned_prestress_matches_full_conditioned(audit):
    assert audit[
        "conditioned_prestress_matches_full_conditioned"
    ] is True


def test_baseline_connections_match_prestress_only_control(audit):
    assert audit[
        "baseline_connections_match_prestress_only_control"
    ] is True


def test_conditioned_connections_match_full_conditioned(audit):
    assert audit[
        "conditioned_connections_match_full_conditioned"
    ] is True


def test_baseline_transfer_gains_match_prestress_only_control(audit):
    assert audit[
        "baseline_transfer_gains_match_prestress_only_control"
    ] is True


def test_conditioned_transfer_gains_match_full_conditioned(audit):
    assert audit[
        "conditioned_transfer_gains_match_full_conditioned"
    ] is True


def test_baseline_and_conditioned_prestress_differ(audit):
    assert audit[
        "baseline_vs_conditioned_prestress_differ"
    ] is True


def test_baseline_and_conditioned_connections_differ(audit):
    assert audit[
        "baseline_vs_conditioned_connections_differ"
    ] is True


# =============================================================================
# MATCHED-PRESTRESS CONNECTION EFFECT
# =============================================================================


def test_connection_state_changes_response_at_matched_prestress(
    central,
):
    assert central[
        "connection_state_changes_response_at_matched_prestress"
    ] is True


def test_connection_state_effect_norm_is_positive_at_matched_prestress(
    central,
):
    value = central[
        "connection_state_effect_norm_at_matched_prestress"
    ]

    assert value > 0.0
    assert math.isfinite(value)


def test_factorial_helper_detects_connection_effect_at_matched_prestress(
    factorial,
):
    assert factorial[
        "connection_effect_detected_at_matched_prestress"
    ] is True


def test_connection_effect_vector_changes_downstream_nodes(
    factorial,
):
    vector = factorial[
        "connection_state_effect_at_matched_baseline_prestress"
    ]["vector"]

    assert vector["A"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert abs(vector["B"]) > 0.0
    assert abs(vector["C"]) > 0.0
    assert abs(vector["D"]) > 0.0


def test_connection_effect_norm_matches_central_result(
    central,
    factorial,
):
    expected = factorial[
        "connection_state_effect_at_matched_baseline_prestress"
    ]["norm"]

    observed = central[
        "connection_state_effect_norm_at_matched_prestress"
    ]

    assert observed == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


# =============================================================================
# MATCHED-CONNECTION PRESTRESS EFFECT
# =============================================================================


def test_prestress_state_does_not_change_response_at_matched_connections(
    central,
):
    assert central[
        "prestress_state_changes_response_at_matched_connections"
    ] is False


def test_prestress_effect_norm_is_numerically_zero_at_matched_connections(
    central,
):
    value = central[
        "prestress_state_effect_norm_at_matched_connections"
    ]

    assert abs(value) <= 1e-12


def test_factorial_helper_reports_no_prestress_effect_at_matched_connections(
    factorial,
):
    assert factorial[
        "prestress_effect_detected_at_matched_connections"
    ] is False


def test_prestress_effect_vector_is_numerically_zero(
    factorial,
):
    vector = factorial[
        "prestress_state_effect_at_matched_baseline_connections"
    ]["vector"]

    assert all(
        abs(float(value)) <= 1e-12
        for value in vector.values()
    )


# =============================================================================
# RESPONSE VECTOR MATCHING
# =============================================================================


def test_r00_and_r10_are_equal_with_baseline_connections(
    factorial,
):
    r00 = factorial["response_vectors"]["R00"]
    r10 = factorial["response_vectors"]["R10"]

    assert r10 == pytest.approx(
        r00,
        rel=1e-12,
        abs=1e-12,
    )


def test_r01_and_r11_are_equal_with_conditioned_connections(
    factorial,
):
    r01 = factorial["response_vectors"]["R01"]
    r11 = factorial["response_vectors"]["R11"]

    assert r11 == pytest.approx(
        r01,
        rel=1e-12,
        abs=1e-12,
    )


def test_r01_differs_from_r00(
    factorial,
):
    r00 = factorial["response_vectors"]["R00"]
    r01 = factorial["response_vectors"]["R01"]

    difference = math.sqrt(
        sum(
            (
                float(r01[node_id])
                - float(r00[node_id])
            ) ** 2
            for node_id in bench.NODE_ORDER
        )
    )

    assert difference > 0.0


# =============================================================================
# FACTORIAL INTERACTION
# =============================================================================


def test_factorial_interaction_not_detected(
    central,
):
    assert central[
        "factorial_interaction_detected"
    ] is False


def test_factorial_interaction_norm_is_numerically_zero(
    central,
):
    value = central[
        "factorial_interaction_norm"
    ]

    assert abs(value) <= 1e-12


def test_factorial_helper_interaction_flag_is_false(
    factorial,
):
    assert factorial[
        "factorial_interaction_detected"
    ] is False


def test_factorial_interaction_vector_is_numerically_zero(
    factorial,
):
    vector = factorial[
        "factorial_interaction_residual"
    ]["vector"]

    assert all(
        abs(float(value)) <= 1e-12
        for value in vector.values()
    )


# =============================================================================
# CONNECTION STRUCTURE HISTORY
# =============================================================================


def test_conditioning_history_has_expected_cycle_count(result):
    trajectory = result["conditioning_history_trajectory"]

    assert len(trajectory) == 5


def test_transfer_gain_increases_across_conditioning_cycles(result):
    trajectory = result["conditioning_history_trajectory"]

    gains = [
        float(item["mean_transfer_gain_after"])
        for item in trajectory
    ]

    assert all(
        right > left
        for left, right in zip(
            gains[:-1],
            gains[1:],
        )
    )


def test_conditioned_transfer_gain_exceeds_baseline(result):
    structures = result["connection_structures"]

    baseline = structures[
        "B00_baseline_p0_theta0"
    ]["mean_effective_transfer_gain"]

    conditioned = structures[
        "C01_baseline_prestress_conditioned_connections"
    ]["mean_effective_transfer_gain"]

    assert conditioned > baseline


def test_conditioned_connection_state_differs_from_baseline(result):
    structures = result["connection_structures"]

    baseline = structures[
        "B00_baseline_p0_theta0"
    ]

    conditioned = structures[
        "C01_baseline_prestress_conditioned_connections"
    ]

    assert (
        conditioned["mean_stiffness"]
        > baseline["mean_stiffness"]
    )

    assert (
        conditioned["mean_contractile_capacity"]
        > baseline["mean_contractile_capacity"]
    )

    assert (
        conditioned["mean_reflex_gain"]
        > baseline["mean_reflex_gain"]
    )


# =============================================================================
# RESPONSE-GEOMETRY OBSERVATIONS
# =============================================================================


def test_conditioned_connections_increase_downstream_response_fraction(
    responses,
):
    baseline = responses[
        "B00_baseline_p0_theta0"
    ]

    conditioned_connections = responses[
        "C01_baseline_prestress_conditioned_connections"
    ]

    assert (
        conditioned_connections[
            "downstream_fraction_of_response"
        ]
        >
        baseline[
            "downstream_fraction_of_response"
        ]
    )


def test_conditioned_connections_increase_total_absolute_response(
    responses,
):
    baseline = responses[
        "B00_baseline_p0_theta0"
    ]

    conditioned_connections = responses[
        "C01_baseline_prestress_conditioned_connections"
    ]

    assert (
        conditioned_connections[
            "total_absolute_response"
        ]
        >
        baseline[
            "total_absolute_response"
        ]
    )


def test_conditioned_connections_increase_peak_downstream_response(
    responses,
):
    baseline = responses[
        "B00_baseline_p0_theta0"
    ]

    conditioned_connections = responses[
        "C01_baseline_prestress_conditioned_connections"
    ]

    assert (
        conditioned_connections[
            "peak_absolute_node_response"
        ]
        >
        baseline[
            "peak_absolute_node_response"
        ]
    )


def test_distribution_entropy_is_not_used_as_stability_success_metric(
    result,
):
    text = result["interpretation"].lower()

    assert "stabilization" in text
    assert "does not interpret" in text


# =============================================================================
# CLAIM BOUNDARY GUARDS
# =============================================================================


def test_matched_prestress_connection_effect_is_explicitly_tested(
    result,
):
    assert result[
        "matched_prestress_connection_effect_tested"
    ] is True


def test_matched_connection_prestress_effect_is_explicitly_tested(
    result,
):
    assert result[
        "matched_connection_prestress_effect_tested"
    ] is True


def test_stabilization_not_tested(result):
    assert result["stabilization_tested"] is False


def test_energy_dissipation_not_tested(result):
    assert result["energy_dissipation_tested"] is False


def test_literal_absorption_not_tested(result):
    assert result["literal_absorption_tested"] is False


def test_injury_failure_threshold_not_tested(result):
    assert result[
        "injury_or_failure_threshold_tested"
    ] is False


def test_predictive_preconfiguration_not_tested(result):
    assert result[
        "predictive_preconfiguration_tested"
    ] is False


def test_no_biological_adaptation_claim(result):
    assert result[
        "biological_adaptation_claimed"
    ] is False


def test_no_human_foot_model_claim(result):
    assert result[
        "human_foot_model_claimed"
    ] is False


def test_no_universal_tensegrity_claim(result):
    assert result[
        "universal_tensegrity_claimed"
    ] is False


# =============================================================================
# INTERPRETATION GUARDS
# =============================================================================


def test_interpretation_mentions_matched_prestress(result):
    text = result["interpretation"].lower()

    assert "matched prestress" in text


def test_interpretation_mentions_connection_state_contribution(result):
    text = result["interpretation"].lower()

    assert "connection-state contribution" in text


def test_interpretation_mentions_prestress_state_contribution(result):
    text = result["interpretation"].lower()

    assert "prestress-state contribution" in text


def test_interpretation_rejects_stabilization_inference(result):
    text = result["interpretation"].lower()

    assert "improved stabilization" in text


def test_interpretation_rejects_absorption_inference(result):
    text = result["interpretation"].lower()

    assert "absorption" in text


def test_interpretation_rejects_dissipation_inference(result):
    text = result["interpretation"].lower()

    assert "dissipation" in text


# =============================================================================
# DETERMINISM
# =============================================================================


def test_benchmark_is_deterministic():
    first = bench.run_benchmark()
    second = bench.run_benchmark()

    assert first == second


def test_factorial_states_are_deterministic():
    baseline_a = bench.build_source_image()
    conditioned_a, _ = bench.evolve_conditioning_history(
        baseline_a
    )

    states_a = bench.build_factorial_states(
        baseline_a,
        conditioned_a,
    )

    baseline_b = bench.build_source_image()
    conditioned_b, _ = bench.evolve_conditioning_history(
        baseline_b
    )

    states_b = bench.build_factorial_states(
        baseline_b,
        conditioned_b,
    )

    assert set(states_a) == set(states_b)

    for key in states_a:
        prestress_a = bench.prestress_signature(
            states_a[key]
        )
        prestress_b = bench.prestress_signature(
            states_b[key]
        )

        assert prestress_a == prestress_b

        connections_a = bench.connection_state_signature(
            states_a[key]
        )
        connections_b = bench.connection_state_signature(
            states_b[key]
        )

        assert connections_a == connections_b

# =============================================================================
# CENTRAL SCIENTIFIC INVARIANT
# =============================================================================


def test_matched_state_identifiability_result_holds_together(
    result,
    central,
    audit,
    factorial,
):
    """
    Central Q8-v2 regression guard.

    The scientific claim is narrow:

    history-conditioned adaptive connection state changes matched-event
    redistribution under explicitly matched pre-challenge prestress.

    No stabilization claim is made.
    """

    assert audit[
        "baseline_prestress_matches_connection_only_control"
    ] is True

    assert audit[
        "baseline_connections_match_prestress_only_control"
    ] is True

    assert central[
        "connection_state_changes_response_at_matched_prestress"
    ] is True

    assert central[
        "connection_state_effect_norm_at_matched_prestress"
    ] > 0.0

    assert central[
        "prestress_state_changes_response_at_matched_connections"
    ] is False

    assert abs(
        central[
            "prestress_state_effect_norm_at_matched_connections"
        ]
    ) <= 1e-12

    assert factorial[
        "factorial_interaction_detected"
    ] is False

    assert result[
        "stabilization_tested"
    ] is False

    assert result[
        "literal_absorption_tested"
    ] is False

    assert result[
        "predictive_preconfiguration_tested"
    ] is False
