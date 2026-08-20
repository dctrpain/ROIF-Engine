from __future__ import annotations

import math

import pytest

from experiments import roif_predictive_stabilization_benchmark as bench


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def result():
    return bench.run_benchmark()


@pytest.fixture(scope="module")
def conditions(result):
    return result["conditions"]


@pytest.fixture(scope="module")
def central(result):
    return result["central_results"]


# =============================================================================
# BASIC BENCHMARK STRUCTURE
# =============================================================================


def test_benchmark_version(result):
    assert result["benchmark_version"] == "roif_predictive_stabilization_v1"


def test_claim_scope_is_computational_only(result):
    assert result["claim_scope"] == "computational_model_only"


def test_no_external_predictive_validity_claim(result):
    assert result["external_predictive_validity_claimed"] is False


def test_no_biological_truth_claim(result):
    assert result["biological_truth_claimed"] is False


def test_no_clinical_validation_claim(result):
    assert result["clinical_validation_claimed"] is False


def test_no_topology_independent_robustness_claim(result):
    assert result["topology_independent_robustness_claimed"] is False


def test_mechanism_under_test_is_prestress_layer_one_step(result):
    assert result["mechanism_under_test"] == (
        "history_conditioned_one_step_prestress_preconfiguration"
    )


def test_prestress_layer_one_step_stabilization_is_tested(result):
    assert result[
        "prestress_layer_one_step_stabilization_tested"
    ] is True


def test_no_whole_system_image_stabilization_claim(result):
    assert result[
        "whole_system_image_stabilization_claimed"
    ] is False


def test_no_multi_step_temporal_image_stabilization_claim(result):
    assert result[
        "multi_step_temporal_image_stabilization_claimed"
    ] is False


def test_no_general_optimal_stabilization_policy_claim(result):
    assert result[
        "general_optimal_stabilization_policy_claimed"
    ] is False


def test_no_universal_stability_claim(result):
    assert result["universal_stability_claimed"] is False


# =============================================================================
# HISTORY / EVENT STRUCTURE
# =============================================================================


def test_history_event_count(result):
    assert result["history_event_count"] == 4


def test_history_sequence(result):
    assert result["history_sequence"] == [
        "E1",
        "E2",
        "E3",
        "E4",
    ]


def test_nominal_expected_event(result):
    assert result["nominal_expected_event"] == "E5"


def test_three_challenge_factors_present(result):
    assert result["challenge_factors"] == [
        0.8,
        1.0,
        1.2,
    ]


def test_expected_challenge_conditions_present(conditions):
    assert set(conditions) == {
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    }


# =============================================================================
# CONTROLLER STRUCTURE
# =============================================================================


def test_controller_gain(result):
    assert result["controller"]["gain"] == pytest.approx(
        bench.CONTROLLER_GAIN
    )


def test_controller_action_bound(result):
    assert result["controller"]["max_abs_action_per_node"] == pytest.approx(
        bench.MAX_ABS_ACTION_PER_NODE
    )


def test_history_conditioned_action_dimension_matches_prestress(result):
    action = result["controller"]["history_conditioned_action"]
    prestress = result["pre_event_prestress"]

    assert len(action) == len(prestress)


def test_stale_action_dimension_matches_history_action(result):
    history_action = result["controller"]["history_conditioned_action"]
    stale_action = result["controller"]["stale_history_blind_action"]

    assert len(history_action) == len(stale_action)


def test_opposite_action_dimension_matches_history_action(result):
    history_action = result["controller"]["history_conditioned_action"]
    opposite_action = result["controller"]["opposite_same_budget_action"]

    assert len(history_action) == len(opposite_action)


def test_history_conditioned_action_is_bounded(result):
    action = result["controller"]["history_conditioned_action"]

    assert all(
        abs(float(value))
        <= bench.MAX_ABS_ACTION_PER_NODE + 1e-12
        for value in action
    )


def test_stale_action_is_bounded(result):
    action = result["controller"]["stale_history_blind_action"]

    assert all(
        abs(float(value))
        <= bench.MAX_ABS_ACTION_PER_NODE + 1e-12
        for value in action
    )


def test_opposite_action_is_bounded(result):
    action = result["controller"]["opposite_same_budget_action"]

    assert all(
        abs(float(value))
        <= bench.MAX_ABS_ACTION_PER_NODE + 1e-12
        for value in action
    )


def test_opposite_action_is_exact_sign_reversal(result):
    history_action = result["controller"]["history_conditioned_action"]
    opposite_action = result["controller"]["opposite_same_budget_action"]

    assert opposite_action == pytest.approx(
        [-float(value) for value in history_action]
    )


# =============================================================================
# HISTORY CONDITIONING
# =============================================================================


def test_history_conditioned_prediction_differs_from_stale_prediction(result):
    history_prediction = result["history_conditioned_prediction"]
    stale_prediction = result["stale_prediction"]

    assert history_prediction != pytest.approx(
        stale_prediction,
        abs=1e-15,
    )


def test_history_conditioned_action_differs_from_stale_action(result):
    history_action = result["controller"]["history_conditioned_action"]
    stale_action = result["controller"]["stale_history_blind_action"]

    difference = math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(
                history_action,
                stale_action,
            )
        )
    )

    assert difference > 0.0


# =============================================================================
# CONDITION-LEVEL METRIC SANITY
# =============================================================================


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_no_control_deviation_is_positive(
    conditions,
    condition_name,
):
    value = conditions[condition_name][
        "no_preconfiguration"
    ]["euclidean_deviation"]

    assert value > 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_history_conditioned_deviation_is_finite(
    conditions,
    condition_name,
):
    value = conditions[condition_name][
        "history_conditioned_predictive"
    ]["euclidean_deviation"]

    assert value >= 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_stale_deviation_is_finite(
    conditions,
    condition_name,
):
    value = conditions[condition_name][
        "stale_history_blind"
    ]["euclidean_deviation"]

    assert value >= 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_opposite_sham_deviation_is_finite(
    conditions,
    condition_name,
):
    value = conditions[condition_name][
        "opposite_same_budget_sham"
    ]["euclidean_deviation"]

    assert value >= 0.0
    assert math.isfinite(value)


# =============================================================================
# PRIMARY STABILIZATION CLAIM
# =============================================================================


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_history_conditioned_preconfiguration_improves_vs_no_control(
    conditions,
    condition_name,
):
    block = conditions[condition_name]

    controlled = block[
        "history_conditioned_predictive"
    ]["euclidean_deviation"]

    uncontrolled = block[
        "no_preconfiguration"
    ]["euclidean_deviation"]

    assert controlled < uncontrolled

    assert block[
        "history_conditioned_beats_no_preconfiguration"
    ] is True


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_absolute_improvement_is_positive(
    conditions,
    condition_name,
):
    improvement = conditions[condition_name][
        "absolute_improvement_vs_no_preconfiguration"
    ]

    assert improvement > 0.0
    assert math.isfinite(improvement)


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_relative_improvement_is_positive(
    conditions,
    condition_name,
):
    improvement = conditions[condition_name][
        "relative_improvement_vs_no_preconfiguration"
    ]

    assert improvement > 0.0
    assert math.isfinite(improvement)


# =============================================================================
# HISTORY-CONDITIONING CONTROL
# =============================================================================


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_history_conditioned_beats_stale_history_blind_control(
    conditions,
    condition_name,
):
    block = conditions[condition_name]

    history_conditioned = block[
        "history_conditioned_predictive"
    ]["euclidean_deviation"]

    stale = block[
        "stale_history_blind"
    ]["euclidean_deviation"]

    assert history_conditioned < stale

    assert block[
        "history_conditioned_beats_stale_control"
    ] is True


# =============================================================================
# DIRECTIONAL / SAME-BUDGET SHAM CONTROL
# =============================================================================


@pytest.mark.parametrize(
    "condition_name",
    [
        "challenge_080pct",
        "challenge_100pct",
        "challenge_120pct",
    ],
)
def test_history_conditioned_beats_opposite_same_budget_sham(
    conditions,
    condition_name,
):
    block = conditions[condition_name]

    history_conditioned = block[
        "history_conditioned_predictive"
    ]["euclidean_deviation"]

    opposite = block[
        "opposite_same_budget_sham"
    ]["euclidean_deviation"]

    assert history_conditioned < opposite

    assert block[
        "history_conditioned_beats_opposite_sham"
    ] is True


def test_opposite_sham_has_same_action_norm_as_history_conditioned(
    conditions,
):
    for block in conditions.values():
        history_norm = block[
            "history_conditioned_predictive"
        ]["action_l2_norm"]

        opposite_norm = block[
            "opposite_same_budget_sham"
        ]["action_l2_norm"]

        assert opposite_norm == pytest.approx(
            history_norm,
            rel=1e-12,
            abs=1e-12,
        )


# =============================================================================
# CENTRAL BENCHMARK FLAGS
# =============================================================================


def test_history_conditioned_beats_no_control_all_conditions(central):
    assert central[
        "history_conditioned_beats_no_control_all_conditions"
    ] is True


def test_history_conditioned_beats_stale_all_conditions(central):
    assert central[
        "history_conditioned_beats_stale_all_conditions"
    ] is True


def test_history_conditioned_beats_opposite_sham_all_conditions(central):
    assert central[
        "history_conditioned_beats_opposite_sham_all_conditions"
    ] is True


def test_mean_relative_improvement_positive(central):
    assert central[
        "mean_relative_improvement_vs_no_control"
    ] > 0.0


def test_minimum_relative_improvement_positive(central):
    assert central[
        "minimum_relative_improvement_vs_no_control"
    ] > 0.0


def test_maximum_relative_improvement_positive(central):
    assert central[
        "maximum_relative_improvement_vs_no_control"
    ] > 0.0


def test_relative_improvement_order_is_consistent(central):
    minimum = central[
        "minimum_relative_improvement_vs_no_control"
    ]

    mean = central[
        "mean_relative_improvement_vs_no_control"
    ]

    maximum = central[
        "maximum_relative_improvement_vs_no_control"
    ]

    assert minimum <= mean <= maximum


# =============================================================================
# DETERMINISM
# =============================================================================


def test_benchmark_is_deterministic():
    first = bench.run_benchmark()
    second = bench.run_benchmark()

    assert first == second


def test_history_conditioned_pre_event_image_is_deterministic():
    first = bench.build_history_conditioned_pre_event_image()
    second = bench.build_history_conditioned_pre_event_image()

    assert bench.prestress_vector(first) == pytest.approx(
        bench.prestress_vector(second),
        rel=0.0,
        abs=0.0,
    )


def test_history_conditioned_prediction_is_deterministic():
    pre_event = bench.build_history_conditioned_pre_event_image()
    nominal_event = bench.build_events()[4]

    first = bench.predict_prestress_response(
        pre_event,
        nominal_event,
        run_id="determinism_first",
    )

    second = bench.predict_prestress_response(
        pre_event,
        nominal_event,
        run_id="determinism_second",
    )

    assert first == pytest.approx(
        second,
        rel=0.0,
        abs=0.0,
    )


# =============================================================================
# CLAIM-BOUNDARY GUARDS
# =============================================================================


def test_interpretation_is_computationally_bounded(result):
    text = result["interpretation"].lower()

    assert "computational" in text


def test_interpretation_identifies_prestress_layer_scope(result):
    text = result["interpretation"].lower()

    assert "prestress layer" in text


def test_interpretation_rejects_whole_system_image_claim(result):
    text = result["interpretation"].lower()

    assert "whole-systemimage" in text


def test_interpretation_rejects_multi_step_temporal_image_claim(result):
    text = result["interpretation"].lower()

    assert "multi-step temporal-image" in text


def test_interpretation_rejects_general_optimal_policy_claim(result):
    text = result["interpretation"].lower()

    assert "general optimal stabilization policy" in text


def test_interpretation_rejects_external_predictive_claim(result):
    text = result["interpretation"].lower()

    assert "external predictive validity" in text


def test_interpretation_rejects_biological_claim(result):
    text = result["interpretation"].lower()

    assert "biological" in text


def test_interpretation_rejects_clinical_claim(result):
    text = result["interpretation"].lower()

    assert "clinical" in text


def test_interpretation_rejects_universal_stability_claim(result):
    text = result["interpretation"].lower()

    assert "universal stability" in text


def test_interpretation_rejects_topology_independent_claim(result):
    text = result["interpretation"].lower()

    assert "topology-independent robustness" in text


# =============================================================================
# CENTRAL SCIENTIFIC INVARIANT
# =============================================================================


def test_stabilization_result_requires_history_specific_direction(
    conditions,
):
    """
    The important scientific guard.

    Improvement must not merely come from spending an equal amount of
    preconfiguration effort.

    The history-conditioned action must outperform the same-budget action in
    the opposite direction for every tested challenge.
    """

    for block in conditions.values():
        history_error = block[
            "history_conditioned_predictive"
        ]["euclidean_deviation"]

        opposite_error = block[
            "opposite_same_budget_sham"
        ]["euclidean_deviation"]

        assert history_error < opposite_error


def test_all_primary_stabilization_findings_hold_together(
    conditions,
    central,
):
    """
    Single regression guard for the complete Q7 stabilization result.
    """

    assert central[
        "history_conditioned_beats_no_control_all_conditions"
    ] is True

    assert central[
        "history_conditioned_beats_stale_all_conditions"
    ] is True

    assert central[
        "history_conditioned_beats_opposite_sham_all_conditions"
    ] is True

    for block in conditions.values():
        history_error = block[
            "history_conditioned_predictive"
        ]["euclidean_deviation"]

        no_control_error = block[
            "no_preconfiguration"
        ]["euclidean_deviation"]

        stale_error = block[
            "stale_history_blind"
        ]["euclidean_deviation"]

        opposite_error = block[
            "opposite_same_budget_sham"
        ]["euclidean_deviation"]

        assert history_error < no_control_error
        assert history_error < stale_error
        assert history_error < opposite_error
