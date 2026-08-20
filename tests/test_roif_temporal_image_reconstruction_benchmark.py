"""
Regression tests for:
    experiments.roif_temporal_image_reconstruction_benchmark

The suite locks the qualitative and selected quantitative behavior of the
temporal-image reconstruction benchmark.

Important:
- zero-error full ROIF replay is treated as an internal consistency result,
  because truth and full replay use the same deterministic transition model;
- these tests do not convert synthetic reconstruction into external predictive
  validity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import experiments.roif_temporal_image_reconstruction_benchmark as bench


@pytest.fixture(scope="module")
def benchmark():
    return bench.run_benchmark()


@pytest.fixture(scope="module")
def reconstruction(benchmark):
    return benchmark["temporal_reconstruction"]


@pytest.fixture(scope="module")
def operator(benchmark):
    return benchmark["operator_evolution"]


# ---------------------------------------------------------------------------
# METADATA / CLAIM BOUNDARIES
# ---------------------------------------------------------------------------

def test_benchmark_version():
    assert bench.BENCHMARK_VERSION == "roif_temporal_image_reconstruction_v1"


def test_claim_scope_constant():
    assert bench.CLAIM_SCOPE == "computational_model_only"


def test_benchmark_reports_version(benchmark):
    assert benchmark["benchmark_version"] == bench.BENCHMARK_VERSION


def test_benchmark_reports_computational_scope(benchmark):
    assert benchmark["claim_scope"] == "computational_model_only"


def test_no_clinical_validation_claim(benchmark):
    assert benchmark["clinical_validation_claimed"] is False


def test_no_biological_truth_claim(benchmark):
    assert benchmark["biological_truth_claimed"] is False


def test_no_causal_truth_claim(benchmark):
    assert benchmark["causal_truth_claimed"] is False


def test_no_external_predictive_validity_claim(benchmark):
    assert benchmark["external_predictive_validity_claimed"] is False


def test_ground_truth_is_synthetic_roif_generated(benchmark):
    assert (
        benchmark["ground_truth_source"]
        == "synthetic_roif_generated_trajectory"
    )


def test_interpretation_note_mentions_no_empirical_forecasting(benchmark):
    text = benchmark["interpretation_note"].lower()
    assert "not empirical forecasting" in text


# ---------------------------------------------------------------------------
# SOURCE IMAGE / TEMPORAL GEOMETRY
# ---------------------------------------------------------------------------

def test_source_revision_zero():
    assert bench.build_source_image().revision == 0


def test_source_has_four_heterogeneous_measures():
    image = bench.build_source_image()
    assert len(image.measures) == 4


def test_source_measure_domains_are_heterogeneous():
    domains = {measure.domain for measure in bench.build_source_image().measures}
    assert domains == {"thermal", "physical", "electrical", "chemical"}


def test_source_has_four_prestress_nodes():
    assert len(bench.build_source_image().prestress_nodes) == 4


def test_source_has_three_adaptive_connections():
    assert len(bench.build_source_image().adaptive_connections) == 3


def test_five_events_defined():
    assert len(bench.build_events()) == 5


def test_truth_trajectory_has_six_slices():
    truth = bench.generate_truth_trajectory()
    assert len(truth) == 6


def test_truth_revisions_are_monotonic():
    truth = bench.generate_truth_trajectory()
    assert [image.revision for image in truth] == [0, 1, 2, 3, 4, 5]


def test_vector_dimension_is_constant():
    truth = bench.generate_truth_trajectory()
    dimensions = {len(bench.image_vector(image)) for image in truth}
    assert len(dimensions) == 1


def test_vector_dimension_regression(benchmark):
    assert benchmark["trajectory"]["vector_dimension"] == 26


def test_observed_slice_indices(benchmark):
    assert benchmark["trajectory"]["observed_slice_indices"] == [0, 2, 4]


def test_hidden_slice_indices(benchmark):
    assert (
        benchmark["trajectory"]["withheld_intermediate_slice_indices"]
        == [1, 3]
    )


def test_future_slice_index(benchmark):
    assert benchmark["trajectory"]["withheld_future_slice_indices"] == [5]


# ---------------------------------------------------------------------------
# LINEAR INTERPOLATION HELPERS
# ---------------------------------------------------------------------------

def test_linear_interpolation_midpoint():
    result = bench.linear_interpolate_vector((0.0, 2.0), (2.0, 6.0), 0.5)
    assert result == pytest.approx((1.0, 4.0))


def test_linear_interpolation_rejects_mismatched_lengths():
    with pytest.raises(bench.TemporalReconstructionError):
        bench.linear_interpolate_vector((1.0,), (1.0, 2.0), 0.5)


def test_linear_interpolation_rejects_fraction_below_zero():
    with pytest.raises(bench.TemporalReconstructionError):
        bench.linear_interpolate_vector((0.0,), (1.0,), -0.1)


def test_linear_interpolation_rejects_fraction_above_one():
    with pytest.raises(bench.TemporalReconstructionError):
        bench.linear_interpolate_vector((0.0,), (1.0,), 1.1)


def test_euclidean_rejects_mismatched_lengths():
    with pytest.raises(bench.TemporalReconstructionError):
        bench._euclidean((1.0,), (1.0, 2.0))


def test_rmse_rejects_mismatched_lengths():
    with pytest.raises(bench.TemporalReconstructionError):
        bench._rmse((1.0,), (1.0, 2.0))


# ---------------------------------------------------------------------------
# TEMPORAL RECONSTRUCTION STRUCTURE
# ---------------------------------------------------------------------------

def test_four_methods_present(reconstruction):
    assert set(reconstruction["methods"]) == {
        "linear_interpolation",
        "memory_free",
        "history_aware_fixed_coupling",
        "full_roif",
    }


def test_reconstruction_observed_indices(reconstruction):
    assert reconstruction["observed_slice_indices"] == [0, 2, 4]


def test_reconstruction_hidden_indices(reconstruction):
    assert reconstruction["hidden_slice_indices"] == [1, 3]


def test_reconstruction_future_indices(reconstruction):
    assert reconstruction["future_slice_indices"] == [5]


def test_full_roif_best_or_tied(reconstruction):
    assert reconstruction["full_roif_best_or_tied"] is True


def test_full_roif_ranked_first(reconstruction):
    first = reconstruction["ranking_by_mean_all_target_rmse"][0]
    assert first["method"] == "full_roif"
    assert first["rank"] == 1


def test_expected_method_ranking(reconstruction):
    ranking = [
        row["method"]
        for row in reconstruction["ranking_by_mean_all_target_rmse"]
    ]
    assert ranking == [
        "full_roif",
        "linear_interpolation",
        "history_aware_fixed_coupling",
        "memory_free",
    ]


# ---------------------------------------------------------------------------
# FULL ROIF INTERNAL CONSISTENCY
# ---------------------------------------------------------------------------

def test_full_roif_hidden_rmse_zero(reconstruction):
    assert reconstruction["methods"]["full_roif"]["mean_hidden_slice_rmse"] == 0.0


def test_full_roif_future_rmse_zero(reconstruction):
    assert reconstruction["methods"]["full_roif"]["future_slice_rmse"] == 0.0


def test_full_roif_all_target_rmse_zero(reconstruction):
    assert reconstruction["methods"]["full_roif"]["mean_all_target_rmse"] == 0.0


@pytest.mark.parametrize("slice_index", ["1", "3", "5"])
def test_full_roif_each_target_slice_exact(reconstruction, slice_index):
    block = reconstruction["methods"]["full_roif"]["per_slice"][slice_index]
    assert block["euclidean_distance"] == 0.0
    assert block["rmse"] == 0.0


def test_reconstruction_note_declares_internal_consistency(reconstruction):
    text = reconstruction["note"].lower()
    assert "internal reconstruction consistency" in text
    assert "not external predictive validity" in text


# ---------------------------------------------------------------------------
# CONTROL BEHAVIOR
# ---------------------------------------------------------------------------

def test_linear_interpolation_nonzero_error(reconstruction):
    assert (
        reconstruction["methods"]["linear_interpolation"]["mean_all_target_rmse"]
        > 0.0
    )


def test_fixed_coupling_nonzero_error(reconstruction):
    assert (
        reconstruction["methods"]["history_aware_fixed_coupling"][
            "mean_all_target_rmse"
        ]
        > 0.0
    )


def test_memory_free_nonzero_error(reconstruction):
    assert (
        reconstruction["methods"]["memory_free"]["mean_all_target_rmse"]
        > 0.0
    )


def test_memory_free_is_worst_control(reconstruction):
    memory_free = reconstruction["methods"]["memory_free"]["mean_all_target_rmse"]
    linear = reconstruction["methods"]["linear_interpolation"]["mean_all_target_rmse"]
    fixed = reconstruction["methods"]["history_aware_fixed_coupling"][
        "mean_all_target_rmse"
    ]
    assert memory_free > linear
    assert memory_free > fixed


def test_memory_free_future_error_larger_than_fixed_coupling(reconstruction):
    memory_free = reconstruction["methods"]["memory_free"]["future_slice_rmse"]
    fixed = reconstruction["methods"]["history_aware_fixed_coupling"][
        "future_slice_rmse"
    ]
    assert memory_free > fixed


def test_memory_free_future_error_larger_than_linear(reconstruction):
    memory_free = reconstruction["methods"]["memory_free"]["future_slice_rmse"]
    linear = reconstruction["methods"]["linear_interpolation"]["future_slice_rmse"]
    assert memory_free > linear


def test_fixed_coupling_differs_from_full_roif_by_t3(reconstruction):
    fixed_t3 = reconstruction["methods"]["history_aware_fixed_coupling"][
        "per_slice"
    ]["3"]["rmse"]
    assert fixed_t3 > 0.0


def test_fixed_coupling_differs_from_full_roif_by_t5(reconstruction):
    fixed_t5 = reconstruction["methods"]["history_aware_fixed_coupling"][
        "per_slice"
    ]["5"]["rmse"]
    assert fixed_t5 > 0.0


def test_memory_free_matches_first_event_before_history_accumulates(reconstruction):
    t1 = reconstruction["methods"]["memory_free"]["per_slice"]["1"]
    assert t1["euclidean_distance"] == 0.0
    assert t1["rmse"] == 0.0


def test_fixed_coupling_matches_first_event_before_connection_history_accumulates(
    reconstruction,
):
    t1 = reconstruction["methods"]["history_aware_fixed_coupling"]["per_slice"]["1"]
    assert t1["euclidean_distance"] == 0.0
    assert t1["rmse"] == 0.0


# ---------------------------------------------------------------------------
# NUMERIC REGRESSION VALUES
# ---------------------------------------------------------------------------

def test_regression_linear_mean_rmse(reconstruction):
    assert reconstruction["methods"]["linear_interpolation"][
        "mean_all_target_rmse"
    ] == pytest.approx(
        0.054229841117239,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_fixed_coupling_mean_rmse(reconstruction):
    assert reconstruction["methods"]["history_aware_fixed_coupling"][
        "mean_all_target_rmse"
    ] == pytest.approx(
        0.062329718533164,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_memory_free_mean_rmse(reconstruction):
    assert reconstruction["methods"]["memory_free"][
        "mean_all_target_rmse"
    ] == pytest.approx(
        0.4098266249510687,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_linear_future_rmse(reconstruction):
    assert reconstruction["methods"]["linear_interpolation"][
        "future_slice_rmse"
    ] == pytest.approx(
        0.0679866511763248,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_fixed_future_rmse(reconstruction):
    assert reconstruction["methods"]["history_aware_fixed_coupling"][
        "future_slice_rmse"
    ] == pytest.approx(
        0.10595295262400659,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_memory_free_future_rmse(reconstruction):
    assert reconstruction["methods"]["memory_free"][
        "future_slice_rmse"
    ] == pytest.approx(
        0.8137153737057264,
        rel=1e-9,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# OPERATOR EVOLUTION: AB->C VS BA->C
# ---------------------------------------------------------------------------

def test_same_event_c_used(operator):
    assert operator["same_event_c_used"] is True


def test_history_changes_response_to_c(operator):
    assert operator["history_changes_response_to_c"] is True


def test_c_response_distance_positive(operator):
    assert operator["c_response_distance"] > 0.0


def test_c_response_distance_finite(operator):
    assert math.isfinite(operator["c_response_distance"])


def test_pre_c_state_distance_positive(operator):
    assert operator["pre_c_state_distance"] > 0.0


def test_post_c_state_distance_positive(operator):
    assert operator["post_c_state_distance"] > 0.0


def test_trajectory_distance_positive(operator):
    assert operator["trajectory_distance"] > 0.0


def test_operator_final_revisions_equal(operator):
    assert operator["abc_final_revision"] == 3
    assert operator["bac_final_revision"] == 3


def test_operator_interpretation_is_computationally_bounded(operator):
    text = operator["interpretation"].lower()
    assert "computational model" in text


def test_regression_c_response_distance(operator):
    assert operator["c_response_distance"] == pytest.approx(
        9.896623991806353e-05,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_pre_c_state_distance(operator):
    assert operator["pre_c_state_distance"] == pytest.approx(
        0.005851093109146412,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_post_c_state_distance(operator):
    assert operator["post_c_state_distance"] == pytest.approx(
        0.005836506871609373,
        rel=1e-9,
        abs=1e-12,
    )


def test_regression_trajectory_distance(operator):
    assert operator["trajectory_distance"] == pytest.approx(
        0.5556140263347483,
        rel=1e-9,
        abs=1e-12,
    )


def test_same_c_changes_response_not_merely_state_label(operator):
    """
    The core operator-evolution guard:
    the response vector to C itself must differ after AB vs BA history.
    """
    assert operator["c_response_distance"] > 1e-12


# ---------------------------------------------------------------------------
# DETERMINISM
# ---------------------------------------------------------------------------

def test_truth_trajectory_is_deterministic():
    first = tuple(bench.image_vector(x) for x in bench.generate_truth_trajectory())
    second = tuple(bench.image_vector(x) for x in bench.generate_truth_trajectory())
    assert first == second


def test_temporal_reconstruction_is_deterministic():
    assert bench.run_temporal_reconstruction() == bench.run_temporal_reconstruction()


def test_operator_evolution_is_deterministic():
    assert bench.run_operator_evolution() == bench.run_operator_evolution()


def test_full_benchmark_is_deterministic():
    assert bench.run_benchmark() == bench.run_benchmark()


# ---------------------------------------------------------------------------
# OUTPUT FILES
# ---------------------------------------------------------------------------

def test_expected_output_filenames():
    expected = {
        "temporal_image_reconstruction_v1.json",
        "temporal_image_reconstruction_slices_v1.csv",
        "temporal_image_reconstruction_methods_v1.csv",
        "operator_evolution_v1.csv",
    }
    assert len(expected) == 4


def test_saved_json_exists_after_benchmark_run():
    root = Path(bench.__file__).resolve().parents[1]
    path = root / "benchmark_results" / "temporal_image_reconstruction_v1.json"
    assert path.exists()
    assert path.stat().st_size > 0


def test_saved_json_is_valid():
    root = Path(bench.__file__).resolve().parents[1]
    path = root / "benchmark_results" / "temporal_image_reconstruction_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["benchmark_version"] == bench.BENCHMARK_VERSION


@pytest.mark.parametrize(
    "filename",
    [
        "temporal_image_reconstruction_slices_v1.csv",
        "temporal_image_reconstruction_methods_v1.csv",
        "operator_evolution_v1.csv",
    ],
)
def test_saved_csv_exists_and_nonempty(filename):
    root = Path(bench.__file__).resolve().parents[1]
    path = root / "benchmark_results" / filename
    assert path.exists()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# CENTRAL INVARIANTS
# ---------------------------------------------------------------------------

def test_all_core_temporal_findings_hold_together(reconstruction, operator):
    assert reconstruction["full_roif_best_or_tied"] is True
    assert reconstruction["methods"]["full_roif"]["mean_all_target_rmse"] == 0.0
    assert (
        reconstruction["methods"]["memory_free"]["mean_all_target_rmse"]
        >
        reconstruction["methods"]["history_aware_fixed_coupling"][
            "mean_all_target_rmse"
        ]
    )
    assert operator["history_changes_response_to_c"] is True
    assert operator["c_response_distance"] > 0.0
    assert operator["trajectory_distance"] > 0.0
