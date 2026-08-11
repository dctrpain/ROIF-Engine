"""
Regression tests for ROIF Engine v1.0 benchmarks Q1-Q5.

These tests freeze the computational claims used by the first algorithmic
paper. They do not constitute clinical validation.

Q1 History dependence
Q2 Recursive stabilization
Q3 Memory-state transition
Q4 Matched additive / nonlinear-additive / multiplicative control
Q5 Separability / recursive non-separability
"""

from __future__ import annotations

import pytest

from experiments.roif_end_to_end_benchmark import (
    BENCHMARK_VERSION,
    benchmark_summary,
    build_world,
    history_pair_distances,
    run_additive_multiplicative_control,
    run_history_dependence,
    run_memory_cycles,
    run_recursive_stabilization,
)
from experiments.roif_additive_multiplicative_control_v2 import (
    CONTROL_VERSION,
    pairwise_history_distance,
    run_matched_synthetic_control,
    run_roif_history_control,
    summarize_control,
)
from experiments.roif_separability_benchmark import (
    BENCHMARK_VERSION as SEPARABILITY_BENCHMARK_VERSION,
    history_pair_distances as q5_history_pair_distances,
    run_recursive_history_branch,
    run_recursive_scalar_demo,
    run_separability_grid,
    scalar_final_distance,
    summarize_separability,
)


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture(scope="module")
def q1_rows(world):
    return run_history_dependence(
        world,
        feedback_gains=(0.0, 1.0, 5.0, 20.0),
    )


@pytest.fixture(scope="module")
def q2_rows(world):
    return run_recursive_stabilization(
        world,
        exposure_counts=(2, 3, 4, 6, 8),
        feedback_gain=1.0,
    )


@pytest.fixture(scope="module")
def q3_rows(world):
    return run_memory_cycles(
        world,
        cycle_count=3,
        feedback_gain=1.0,
    )


@pytest.fixture(scope="module")
def q4_v1_rows():
    return run_additive_multiplicative_control(
        couplings=(0.02, 0.05, 0.10, 0.20),
        steps=12,
    )


@pytest.fixture(scope="module")
def q4_v2_synthetic():
    return run_matched_synthetic_control()


@pytest.fixture(scope="module")
def q4_v2_history():
    return run_roif_history_control()


@pytest.fixture(scope="module")
def q5_sep_rows():
    return run_separability_grid()


@pytest.fixture(scope="module")
def q5_history_rows():
    return run_recursive_history_branch()


@pytest.fixture(scope="module")
def q5_scalar_rows():
    return run_recursive_scalar_demo()


# ---------------------------------------------------------------------------
# General / versions
# ---------------------------------------------------------------------------

def test_q1_q3_benchmark_version():
    assert BENCHMARK_VERSION == "roif_e2e_benchmark_v1"


def test_q4_control_version():
    assert CONTROL_VERSION == "roif_additive_multiplicative_control_v2"


def test_q5_benchmark_version():
    assert SEPARABILITY_BENCHMARK_VERSION == "roif_separability_benchmark_v1"


# ---------------------------------------------------------------------------
# Q1 — History dependence
# ---------------------------------------------------------------------------

def test_q1_zero_feedback_removes_history_effect(q1_rows):
    distances = history_pair_distances(q1_rows)
    assert distances["0.0"] == pytest.approx(0.0, abs=1e-15)


def test_q1_positive_feedback_creates_nonzero_history_effect(q1_rows):
    distances = history_pair_distances(q1_rows)

    assert distances["1.0"] > 0.0
    assert distances["5.0"] > 0.0
    assert distances["20.0"] > 0.0


def test_q1_history_effect_increases_across_selected_gains(q1_rows):
    distances = history_pair_distances(q1_rows)

    assert (
        distances["0.0"]
        < distances["1.0"]
        < distances["5.0"]
        < distances["20.0"]
    )


def test_q1_nonzero_history_effect_count_is_three(q1_rows):
    distances = history_pair_distances(q1_rows)

    assert sum(
        1
        for value in distances.values()
        if value > 1e-12
    ) == 3


def test_q1_high_gain_effect_is_materially_larger_than_low_gain(q1_rows):
    distances = history_pair_distances(q1_rows)

    assert distances["20.0"] > 1000.0 * distances["1.0"]


# ---------------------------------------------------------------------------
# Q2 — Recursive stabilization
# ---------------------------------------------------------------------------

def test_q2_exposure_counts_preserved(q2_rows):
    assert tuple(
        row.exposure_count
        for row in q2_rows
    ) == (2, 3, 4, 6, 8)


def test_q2_terminal_drift_decreases_monotonically(q2_rows):
    drifts = tuple(
        row.terminal_drift
        for row in q2_rows
    )

    assert all(
        later < earlier
        for earlier, later in zip(
            drifts,
            drifts[1:],
        )
    )


def test_q2_eight_exposures_have_lower_drift_than_two(q2_rows):
    assert q2_rows[-1].terminal_drift < q2_rows[0].terminal_drift


def test_q2_eight_exposure_drift_reduction_is_large(q2_rows):
    ratio = (
        q2_rows[0].terminal_drift
        / q2_rows[-1].terminal_drift
    )

    assert ratio > 20.0


def test_q2_two_exposures_persistent(q2_rows):
    assert q2_rows[0].regime == "persistent"


def test_q2_three_or_more_selected_exposures_convergent(q2_rows):
    assert all(
        row.regime == "convergent"
        for row in q2_rows[1:]
    )


def test_q2_final_selected_convergence_score_high(q2_rows):
    assert q2_rows[-1].convergence_score > 0.90


# ---------------------------------------------------------------------------
# Q3 — Memory state transition
# ---------------------------------------------------------------------------

def test_q3_three_cycles_created(q3_rows):
    assert len(q3_rows) == 3


def test_q3_revision_chain_is_sequential(q3_rows):
    assert tuple(
        (
            row.source_revision,
            row.target_revision,
        )
        for row in q3_rows
    ) == (
        (0, 1),
        (1, 2),
        (2, 3),
    )


def test_q3_trace_count_chain_is_sequential(q3_rows):
    assert tuple(
        (
            row.source_trace_count,
            row.target_trace_count,
        )
        for row in q3_rows
    ) == (
        (0, 1),
        (1, 2),
        (2, 3),
    )


def test_q3_first_cycle_is_first_trace(q3_rows):
    assert q3_rows[0].integration_relation == "first_trace"


def test_q3_later_cycles_reinforce_existing_memory(q3_rows):
    assert tuple(
        row.integration_relation
        for row in q3_rows[1:]
    ) == (
        "reinforcement_present",
        "reinforcement_present",
    )


def test_q3_final_revision_three(q3_rows):
    assert q3_rows[-1].target_revision == 3


def test_q3_final_trace_count_three(q3_rows):
    assert q3_rows[-1].target_trace_count == 3


def test_q3_commitment_confidence_bounded(q3_rows):
    assert all(
        0.0 <= row.commitment_confidence <= 1.0
        for row in q3_rows
    )


# ---------------------------------------------------------------------------
# Q4 v1 — retained only as historical baseline
# ---------------------------------------------------------------------------

def test_q4_v1_contains_additive_and_multiplicative(q4_v1_rows):
    assert {
        row.model
        for row in q4_v1_rows
    } == {
        "additive",
        "multiplicative",
    }


def test_q4_v1_is_not_used_as_biological_proof(q1_rows, q2_rows, q3_rows, q4_v1_rows):
    summary = benchmark_summary(
        q1_rows,
        q2_rows,
        q3_rows,
        q4_v1_rows,
    )

    assert summary["clinical_validation_claimed"] is False
    assert summary["causal_truth_claimed"] is False


# ---------------------------------------------------------------------------
# Q4 v2 — matched controls
# ---------------------------------------------------------------------------

def test_q4_v2_contains_three_synthetic_models(q4_v2_synthetic):
    assert {
        row.model
        for row in q4_v2_synthetic
    } == {
        "linear_additive",
        "nonlinear_additive_threshold",
        "multiplicative",
    }


def test_q4_v2_each_model_has_same_gain_grid(q4_v2_synthetic):
    expected = {0.02, 0.05, 0.10, 0.20}

    grouped = {}

    for row in q4_v2_synthetic:
        grouped.setdefault(
            row.model,
            set(),
        ).add(row.gain)

    assert all(
        gains == expected
        for gains in grouped.values()
    )


def test_q4_v2_threshold_crossing_not_unique_to_multiplicative(q4_v2_synthetic):
    counts = {}

    for row in q4_v2_synthetic:
        counts.setdefault(
            row.model,
            0,
        )

        if row.threshold_crossed:
            counts[row.model] += 1

    assert counts["linear_additive"] >= 1
    assert counts["nonlinear_additive_threshold"] >= 1
    assert counts["multiplicative"] >= 1


def test_q4_v2_nonlinear_additive_can_match_or_exceed_multiplicative_mean(q4_v2_synthetic):
    means = {}

    for model in {
        row.model
        for row in q4_v2_synthetic
    }:
        subset = [
            row.final_total
            for row in q4_v2_synthetic
            if row.model == model
        ]
        means[model] = sum(subset) / len(subset)

    assert (
        means["nonlinear_additive_threshold"]
        >= means["multiplicative"]
    )


def test_q4_v2_summary_refuses_biological_multiplicativity_claim(
    q4_v2_synthetic,
    q4_v2_history,
):
    summary = summarize_control(
        q4_v2_synthetic,
        q4_v2_history,
    )

    assert summary["biological_multiplicativity_claimed"] is False
    assert summary["clinical_validation_claimed"] is False


def test_q4_v2_roif_history_zero_at_zero_feedback(q4_v2_history):
    distances = pairwise_history_distance(
        q4_v2_history
    )

    assert distances["0.0"] == pytest.approx(
        0.0,
        abs=1e-15,
    )


def test_q4_v2_roif_history_nonzero_with_feedback(q4_v2_history):
    distances = pairwise_history_distance(
        q4_v2_history
    )

    assert all(
        distances[key] > 0.0
        for key in (
            "1.0",
            "5.0",
            "20.0",
        )
    )


# ---------------------------------------------------------------------------
# Q5 — Separability / non-separability
# ---------------------------------------------------------------------------

def test_q5_separable_additive_has_zero_interaction(q5_sep_rows):
    subset = [
        row
        for row in q5_sep_rows
        if row.model == "separable_additive"
    ]

    assert all(
        abs(row.interaction_residual) <= 1e-12
        for row in subset
    )


def test_q5_nonlinear_but_separable_has_zero_interaction(q5_sep_rows):
    subset = [
        row
        for row in q5_sep_rows
        if row.model == "nonlinear_but_separable"
    ]

    assert all(
        abs(row.interaction_residual) <= 1e-12
        for row in subset
    )


def test_q5_multiplicative_interaction_is_nonseparable(q5_sep_rows):
    subset = [
        row
        for row in q5_sep_rows
        if row.model == "multiplicative_interaction"
    ]

    assert all(
        row.separable_within_tolerance is False
        for row in subset
    )


def test_q5_state_modulated_interaction_is_nonseparable(q5_sep_rows):
    subset = [
        row
        for row in q5_sep_rows
        if row.model == "state_modulated_interaction"
    ]

    assert all(
        row.separable_within_tolerance is False
        for row in subset
    )


def test_q5_state_modulated_has_larger_mean_interaction_than_simple_multiplicative(
    q5_sep_rows,
):
    summary = summarize_separability(
        q5_sep_rows
    )

    assert (
        summary["state_modulated_interaction"][
            "mean_abs_interaction_residual"
        ]
        >
        summary["multiplicative_interaction"][
            "mean_abs_interaction_residual"
        ]
    )


def test_q5_separable_fraction_one_for_separable_models(q5_sep_rows):
    summary = summarize_separability(
        q5_sep_rows
    )

    assert (
        summary["separable_additive"]["separable_fraction"]
        == pytest.approx(1.0)
    )
    assert (
        summary["nonlinear_but_separable"]["separable_fraction"]
        == pytest.approx(1.0)
    )


def test_q5_separable_fraction_zero_for_interaction_models(q5_sep_rows):
    summary = summarize_separability(
        q5_sep_rows
    )

    assert (
        summary["multiplicative_interaction"]["separable_fraction"]
        == pytest.approx(0.0)
    )
    assert (
        summary["state_modulated_interaction"]["separable_fraction"]
        == pytest.approx(0.0)
    )


def test_q5_roif_zero_history_effect_at_zero_feedback(q5_history_rows):
    distances = q5_history_pair_distances(
        q5_history_rows
    )

    assert distances["0.0"] == pytest.approx(
        0.0,
        abs=1e-15,
    )


def test_q5_roif_history_effect_positive_with_recursive_feedback(q5_history_rows):
    distances = q5_history_pair_distances(
        q5_history_rows
    )

    assert all(
        distances[key] > 0.0
        for key in (
            "1.0",
            "5.0",
            "20.0",
        )
    )


def test_q5_recursive_scalar_same_input_retains_history_difference(q5_scalar_rows):
    assert scalar_final_distance(
        q5_scalar_rows
    ) > 0.0


def test_q5_recursive_scalar_effect_is_material(q5_scalar_rows):
    assert scalar_final_distance(
        q5_scalar_rows
    ) > 0.5


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_q1_is_deterministic(world):
    left = run_history_dependence(
        world,
        feedback_gains=(0.0, 1.0, 5.0, 20.0),
    )
    right = run_history_dependence(
        world,
        feedback_gains=(0.0, 1.0, 5.0, 20.0),
    )

    assert left == right


def test_q2_is_deterministic(world):
    left = run_recursive_stabilization(
        world,
        exposure_counts=(2, 3, 4, 6, 8),
        feedback_gain=1.0,
    )
    right = run_recursive_stabilization(
        world,
        exposure_counts=(2, 3, 4, 6, 8),
        feedback_gain=1.0,
    )

    assert left == right


def test_q3_is_deterministic(world):
    left = run_memory_cycles(
        world,
        cycle_count=3,
        feedback_gain=1.0,
    )
    right = run_memory_cycles(
        world,
        cycle_count=3,
        feedback_gain=1.0,
    )

    assert left == right


def test_q4_v2_is_deterministic():
    assert (
        run_matched_synthetic_control()
        == run_matched_synthetic_control()
    )


def test_q5_is_deterministic():
    assert (
        run_separability_grid()
        == run_separability_grid()
    )
