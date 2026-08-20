"""
Regression tests for experiments.build_entropy_figures
Publication asset generator v2.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import experiments.build_entropy_figures as gen


@pytest.fixture(scope="module")
def generated():
    return gen.run()


# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------

def test_generator_version():
    assert gen.GENERATOR_VERSION == "roif_entropy_assets_v2"


def test_claim_scope(generated):
    assert generated["claim_scope"] == "computational_model_only"


def test_source_files_declared(generated):
    assert generated["source_files"] == [
        "benchmark_results/physical_history_integration_v1.json",
        "benchmark_results/separability_benchmark_v1.json",
        "benchmark_results/q6_lyapunov_like_v1.json",
        "benchmark_results/temporal_image_reconstruction_v1.json",
    ]


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------

def test_seven_figures_declared(generated):
    assert set(generated["figures"]) == {
        "global_redistribution",
        "path_dependence",
        "repeated_event",
        "separability",
        "q6",
        "temporal_reconstruction",
        "operator_evolution",
    }


@pytest.mark.parametrize(
    "figure_key",
    [
        "global_redistribution",
        "path_dependence",
        "repeated_event",
        "separability",
        "q6",
        "temporal_reconstruction",
        "operator_evolution",
    ],
)
def test_each_figure_has_pdf_and_png(generated, figure_key):
    assert set(generated["figures"][figure_key]) == {"pdf", "png"}


@pytest.mark.parametrize(
    "figure_key",
    [
        "global_redistribution",
        "path_dependence",
        "repeated_event",
        "separability",
        "q6",
        "temporal_reconstruction",
        "operator_evolution",
    ],
)
def test_each_figure_file_exists(generated, figure_key):
    for rel in generated["figures"][figure_key].values():
        path = gen.ROOT / rel
        assert path.exists()
        assert path.stat().st_size > 0


def test_temporal_figure_paths(generated):
    block = generated["figures"]["temporal_reconstruction"]
    assert block["pdf"].endswith("figure_7_temporal_image_reconstruction.pdf")
    assert block["png"].endswith("figure_7_temporal_image_reconstruction.png")


def test_operator_figure_paths(generated):
    block = generated["figures"]["operator_evolution"]
    assert block["pdf"].endswith("figure_8_operator_evolution.pdf")
    assert block["png"].endswith("figure_8_operator_evolution.png")


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------

def test_five_tables_declared(generated):
    assert set(generated["tables"]) == {
        "physical_history",
        "temporal_reconstruction",
        "operator_evolution",
        "claim_boundaries",
        "benchmark_summary",
    }


@pytest.mark.parametrize(
    "table_key",
    [
        "physical_history",
        "temporal_reconstruction",
        "operator_evolution",
        "claim_boundaries",
        "benchmark_summary",
    ],
)
def test_each_table_has_csv_and_tex(generated, table_key):
    assert set(generated["tables"][table_key]) == {"csv", "tex"}


@pytest.mark.parametrize(
    "table_key",
    [
        "physical_history",
        "temporal_reconstruction",
        "operator_evolution",
        "claim_boundaries",
        "benchmark_summary",
    ],
)
def test_each_table_file_exists(generated, table_key):
    for rel in generated["tables"][table_key].values():
        path = gen.ROOT / rel
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

def test_manifest_exists(generated):
    path = gen.ROOT / generated["manifest"]
    assert path.exists()
    assert path.stat().st_size > 0


def test_manifest_is_valid_json(generated):
    path = gen.ROOT / generated["manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["generator_version"] == gen.GENERATOR_VERSION


def test_manifest_contains_temporal_source(generated):
    path = gen.ROOT / generated["manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        "benchmark_results/temporal_image_reconstruction_v1.json"
        in payload["source_files"]
    )


# ---------------------------------------------------------------------------
# PHYSICAL HISTORY VALUES
# ---------------------------------------------------------------------------

def test_physical_history_table_preserves_order_distance(generated):
    path = gen.ROOT / generated["tables"]["physical_history"]["csv"]

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    row = next(
        item
        for item in rows
        if item["Experiment"] == "Order dependence: A→B vs B→A"
    )

    assert float(row["Value"]) == pytest.approx(
        0.011814192037381752,
        rel=1e-8,
        abs=1e-12,
    )


def test_physical_history_table_preserves_repeat_distance(generated):
    path = gen.ROOT / generated["tables"]["physical_history"]["csv"]

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    row = next(
        item
        for item in rows
        if item["Experiment"] == "Repeated identical event"
    )

    assert float(row["Value"]) == pytest.approx(
        0.06723725281827213,
        rel=1e-8,
        abs=1e-12,
    )


def test_physical_history_table_preserves_redistribution_norm(generated):
    path = gen.ROOT / generated["tables"]["physical_history"]["csv"]

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    row = next(
        item
        for item in rows
        if item["Experiment"] == "Local-to-global redistribution"
    )

    assert float(row["Value"]) == pytest.approx(
        0.30046937477559266,
        rel=1e-8,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# TEMPORAL RECONSTRUCTION TABLE
# ---------------------------------------------------------------------------

def _temporal_rows(generated):
    path = gen.ROOT / generated["tables"]["temporal_reconstruction"]["csv"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_temporal_table_has_four_methods(generated):
    rows = _temporal_rows(generated)
    assert [row["Method"] for row in rows] == [
        "full_roif",
        "linear_interpolation",
        "history_aware_fixed_coupling",
        "memory_free",
    ]


def test_temporal_table_full_roif_zero(generated):
    rows = _temporal_rows(generated)
    row = next(item for item in rows if item["Method"] == "full_roif")
    assert float(row["Hidden-slice RMSE"]) == 0.0
    assert float(row["Future-slice RMSE"]) == 0.0
    assert float(row["All-target RMSE"]) == 0.0


def test_temporal_table_linear_regression(generated):
    rows = _temporal_rows(generated)
    row = next(item for item in rows if item["Method"] == "linear_interpolation")
    assert float(row["All-target RMSE"]) == pytest.approx(
        0.054229841117239,
        rel=1e-8,
        abs=1e-12,
    )


def test_temporal_table_fixed_coupling_regression(generated):
    rows = _temporal_rows(generated)
    row = next(
        item
        for item in rows
        if item["Method"] == "history_aware_fixed_coupling"
    )
    assert float(row["All-target RMSE"]) == pytest.approx(
        0.062329718533164,
        rel=1e-8,
        abs=1e-12,
    )


def test_temporal_table_memory_free_regression(generated):
    rows = _temporal_rows(generated)
    row = next(item for item in rows if item["Method"] == "memory_free")
    assert float(row["All-target RMSE"]) == pytest.approx(
        0.4098266249510687,
        rel=1e-8,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# OPERATOR EVOLUTION TABLE
# ---------------------------------------------------------------------------

def _operator_rows(generated):
    path = gen.ROOT / generated["tables"]["operator_evolution"]["csv"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_operator_table_preserves_pre_c_distance(generated):
    rows = _operator_rows(generated)
    row = next(item for item in rows if item["Metric"] == "Pre-C state distance")
    assert float(row["Value"]) == pytest.approx(
        0.005851093109146412,
        rel=1e-8,
        abs=1e-12,
    )


def test_operator_table_preserves_post_c_distance(generated):
    rows = _operator_rows(generated)
    row = next(item for item in rows if item["Metric"] == "Post-C state distance")
    assert float(row["Value"]) == pytest.approx(
        0.005836506871609373,
        rel=1e-8,
        abs=1e-12,
    )


def test_operator_table_preserves_c_response_distance(generated):
    rows = _operator_rows(generated)
    row = next(item for item in rows if item["Metric"] == "C-response distance")
    assert float(row["Value"]) == pytest.approx(
        9.896623991806353e-05,
        rel=1e-8,
        abs=1e-12,
    )


def test_operator_table_preserves_trajectory_distance(generated):
    rows = _operator_rows(generated)
    row = next(
        item for item in rows if item["Metric"] == "Whole-trajectory distance"
    )
    assert float(row["Value"]) == pytest.approx(
        0.5556140263347483,
        rel=1e-8,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# CLAIM BOUNDARIES
# ---------------------------------------------------------------------------

def test_claim_boundaries_include_no_universal_multiplicativity(generated):
    path = gen.ROOT / generated["tables"]["claim_boundaries"]["csv"]
    text = path.read_text(encoding="utf-8")
    assert "Universal multiplicativity" in text


def test_claim_boundaries_include_no_clinical_causality(generated):
    path = gen.ROOT / generated["tables"]["claim_boundaries"]["csv"]
    text = path.read_text(encoding="utf-8")
    assert "Clinical causality" in text


def test_claim_boundaries_include_no_butterfly_claim(generated):
    path = gen.ROOT / generated["tables"]["claim_boundaries"]["csv"]
    text = path.read_text(encoding="utf-8")
    assert "Deterministic chaos or butterfly effect" in text


def test_claim_boundaries_include_no_external_forecasting_claim(generated):
    path = gen.ROOT / generated["tables"]["claim_boundaries"]["csv"]
    text = path.read_text(encoding="utf-8")
    assert "External forecasting accuracy" in text


def test_claim_boundaries_include_no_universal_causal_law(generated):
    path = gen.ROOT / generated["tables"]["claim_boundaries"]["csv"]
    text = path.read_text(encoding="utf-8")
    assert "Universal causal law" in text


# ---------------------------------------------------------------------------
# BENCHMARK SUMMARY
# ---------------------------------------------------------------------------

def _benchmark_summary_text(generated):
    path = gen.ROOT / generated["tables"]["benchmark_summary"]["csv"]
    return path.read_text(encoding="utf-8")


def test_benchmark_summary_contains_q4_q5_q6(generated):
    text = _benchmark_summary_text(generated)
    assert "Q4" in text
    assert "Q5" in text
    assert "Q6" in text


def test_benchmark_summary_contains_q7_q8_q9(generated):
    text = _benchmark_summary_text(generated)
    assert "Q7" in text
    assert "Q8" in text
    assert "Q9" in text


def test_benchmark_summary_contains_physical_1_2_3(generated):
    text = _benchmark_summary_text(generated)
    assert "Physical 1" in text
    assert "Physical 2" in text
    assert "Physical 3" in text


def test_benchmark_summary_q9_mentions_history_conditioned_response(generated):
    text = _benchmark_summary_text(generated)
    assert "History-conditioned response to the same subsequent event" in text
    assert "History-conditioned response operator" not in text


# ---------------------------------------------------------------------------
# Q5 FINDER
# ---------------------------------------------------------------------------

def test_separability_finder_accepts_direct_payload():
    payload = {
        "separable_additive": {},
        "nonlinear_but_separable": {},
        "multiplicative_interaction": {},
        "state_modulated_interaction": {},
    }
    assert gen._find_separability_models(payload) is payload


def test_separability_finder_accepts_nested_payload():
    models = {
        "separable_additive": {},
        "nonlinear_but_separable": {},
        "multiplicative_interaction": {},
        "state_modulated_interaction": {},
    }
    payload = {"outer": {"inner": models}}
    assert gen._find_separability_models(payload) == models


def test_separability_finder_rejects_missing_models():
    with pytest.raises(gen.PublicationAssetError):
        gen._find_separability_models({"separable_additive": {}})


# ---------------------------------------------------------------------------
# Q6 FINDER
# ---------------------------------------------------------------------------

def _q6_gain_block():
    return {
        "0.0": {
            "all_bounded_under_limit": True,
            "max_peak_amplification_ratio": 1.0,
            "positive_max_lambda_count": 0,
        },
        "1.0": {
            "all_bounded_under_limit": True,
            "max_peak_amplification_ratio": 1.0,
            "positive_max_lambda_count": 0,
        },
    }


def test_q6_finder_accepts_direct_gain_block():
    payload = _q6_gain_block()
    assert gen._find_q6_by_feedback_gain(payload) is payload


def test_q6_finder_accepts_nested_gain_block():
    block = _q6_gain_block()
    payload = {"outer": {"summary": block}}
    assert gen._find_q6_by_feedback_gain(payload) == block


def test_q6_finder_rejects_invalid_payload():
    with pytest.raises(gen.PublicationAssetError):
        gen._find_q6_by_feedback_gain({"0.0": {"something_else": 1}})


# ---------------------------------------------------------------------------
# REQUIRED SOURCES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        "physical_history_integration_v1.json",
        "separability_benchmark_v1.json",
        "q6_lyapunov_like_v1.json",
        "temporal_image_reconstruction_v1.json",
    ],
)
def test_required_benchmark_files_exist(filename):
    path = gen.RESULTS / filename
    assert path.exists()


@pytest.mark.parametrize(
    "filename",
    [
        "physical_history_integration_v1.json",
        "separability_benchmark_v1.json",
        "q6_lyapunov_like_v1.json",
        "temporal_image_reconstruction_v1.json",
    ],
)
def test_required_benchmark_files_are_valid_json(filename):
    payload = gen.load_json(filename)
    assert isinstance(payload, dict)
    assert payload


# ---------------------------------------------------------------------------
# DETERMINISM
# ---------------------------------------------------------------------------

def test_run_is_repeatable():
    first = gen.run()
    second = gen.run()
    assert first == second


def test_manifest_is_stable_across_runs():
    gen.run()

    path = gen.PAPER / "publication_asset_manifest.json"
    first = path.read_text(encoding="utf-8")

    gen.run()

    second = path.read_text(encoding="utf-8")
    assert first == second


def test_all_generated_paths_remain_inside_paper(generated):
    all_paths = []

    for group in generated["figures"].values():
        all_paths.extend(group.values())

    for group in generated["tables"].values():
        all_paths.extend(group.values())

    all_paths.append(generated["manifest"])

    for rel in all_paths:
        normalized = Path(rel)
        assert normalized.parts[0] == "paper"


def test_no_generated_asset_is_empty(generated):
    paths = []

    for group in generated["figures"].values():
        paths.extend(group.values())

    for group in generated["tables"].values():
        paths.extend(group.values())

    paths.append(generated["manifest"])

    for rel in paths:
        path = gen.ROOT / rel
        assert path.stat().st_size > 0
