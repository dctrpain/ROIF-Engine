"""Regression tests for experiments.run_entropy_reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import experiments.run_entropy_reproducibility as repro


def test_runner_version_is_frozen():
    assert repro.RUNNER_VERSION == "roif_entropy_reproducibility_v1"


def test_claim_scope_is_computational_only():
    assert repro.CLAIM_SCOPE == "computational_model_only"


def test_runner_declares_manuscript_benchmarks():
    ids = [step_id for step_id, _ in repro.BENCHMARK_MODULES]

    assert "matched_state_physical_history_identifiability" in ids
    assert "structured_memory_conditioned_matched_state_transition" in ids
    assert "restricted_predictive_preconfiguration" in ids
    assert "q7_objective_independence_audit" in ids
    assert "q7_off_nominal_transferability_audit" in ids
    assert "q8_history_conditioned_redistribution_capacity" in ids
    assert "q8_history_conditioned_redistribution_matched_state" in ids
    assert "temporal_image_reconstruction" in ids
    assert "multilayer_temporal_image_trajectory" in ids


def test_runner_declares_expected_artifacts():
    expected = set(repro.EXPECTED_ARTIFACTS)

    assert (
        "benchmark_results/"
        "matched_state_history_operator_identifiability_v1.json"
        in expected
    )
    assert (
        "benchmark_results/"
        "memory_conditioned_matched_state_transition_v1.json"
        in expected
    )
    assert (
        "benchmark_results/"
        "q7_objective_independence_audit_v1.json"
        in expected
    )
    assert (
        "benchmark_results/"
        "q7_off_nominal_transferability_audit_v2.json"
        in expected
    )
    assert (
        "benchmark_results/"
        "q8_history_conditioned_redistribution_matched_state_v2.json"
        in expected
    )
    assert (
        "benchmark_results/"
        "temporal_image_reconstruction_v1.json"
        in expected
    )


def test_required_runner_inputs_exist():
    assert repro.validate_inputs() == []


def test_expected_artifacts_are_currently_valid():
    ok, records = repro.validate_artifacts()

    assert ok is True
    assert records
    assert all(record["exists"] for record in records)
    assert all(record["valid"] for record in records)
    assert all(record["sha256"] for record in records)


def test_environment_has_reproducibility_metadata():
    env = repro.environment()

    assert env["runner_version"] == repro.RUNNER_VERSION
    assert env["claim_scope"] == repro.CLAIM_SCOPE
    assert env["python"]["version"]
    assert "repository" in env
    assert "commit" in env["repository"]


def test_markdown_report_preserves_claim_boundaries():
    env = {
        "generated_at_utc": "2026-08-20T00:00:00+00:00",
        "repository": {
            "commit": "abc123",
            "describe": "v1.4.1",
        },
    }

    report = repro.markdown_report(
        env=env,
        steps=[],
        artifacts_ok=True,
        artifacts=[],
        full_tests=False,
    )

    assert "biological or clinical validity" in report
    assert "general history-conditioned operator" in report
    assert "complete future Temporal-Image prediction" in report
    assert "objective-independent whole-system predictive stabilization" in report
    assert "restricted tested prestress-preconfiguration" in report


def test_write_json_is_valid_utf8_json(tmp_path: Path):
    path = tmp_path / "report.json"
    repro.write_json(path, {"runner": repro.RUNNER_VERSION})

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {"runner": repro.RUNNER_VERSION}
