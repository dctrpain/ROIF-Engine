"""Regression tests for experiments.run_entropy_reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import experiments.run_entropy_reproducibility as repro


def test_runner_version_is_frozen():
    assert repro.RUNNER_VERSION == "roif_entropy_reproducibility_v2"


def test_claim_scope_is_computational_only():
    assert repro.CLAIM_SCOPE == "computational_model_only"


def test_required_runner_inputs_exist():
    assert repro.validate_inputs() == []


def test_snapshot_contains_all_expected_artifacts():
    snap = repro.snapshot_artifacts()

    assert set(snap) == set(repro.EXPECTED_ARTIFACTS)


def test_expected_artifacts_are_currently_valid():
    ok, records = repro.validate_artifact_content()

    assert ok is True
    assert records
    assert all(record["exists"] for record in records)
    assert all(record["valid"] for record in records)
    assert all(record["sha256"] for record in records)


def test_identical_snapshots_pass_exact_reproduction():
    snap = {
        "benchmark_results/example.json": {
            "path": "benchmark_results/example.json",
            "exists": True,
            "tracked": True,
            "size_bytes": 10,
            "sha256": "abc",
        }
    }

    original = repro.EXPECTED_ARTIFACTS
    try:
        repro.EXPECTED_ARTIFACTS = ("benchmark_results/example.json",)
        ok, records = repro.compare_artifact_snapshots(snap, snap)
    finally:
        repro.EXPECTED_ARTIFACTS = original

    assert ok is True
    assert records[0]["status"] == "REPRODUCED_EXACTLY"


def test_changed_tracked_artifact_fails_exact_reproduction():
    before = {
        "benchmark_results/example.json": {
            "path": "benchmark_results/example.json",
            "exists": True,
            "tracked": True,
            "size_bytes": 10,
            "sha256": "abc",
        }
    }
    after = {
        "benchmark_results/example.json": {
            "path": "benchmark_results/example.json",
            "exists": True,
            "tracked": True,
            "size_bytes": 11,
            "sha256": "def",
        }
    }

    original = repro.EXPECTED_ARTIFACTS
    try:
        repro.EXPECTED_ARTIFACTS = ("benchmark_results/example.json",)
        ok, records = repro.compare_artifact_snapshots(before, after)
    finally:
        repro.EXPECTED_ARTIFACTS = original

    assert ok is False
    assert records[0]["status"] == "CHANGED"


def test_new_untracked_artifact_does_not_define_tracked_equality_failure():
    before = {
        "benchmark_results/example.json": {
            "path": "benchmark_results/example.json",
            "exists": False,
            "tracked": False,
        }
    }
    after = {
        "benchmark_results/example.json": {
            "path": "benchmark_results/example.json",
            "exists": True,
            "tracked": False,
            "size_bytes": 11,
            "sha256": "def",
        }
    }

    original = repro.EXPECTED_ARTIFACTS
    try:
        repro.EXPECTED_ARTIFACTS = ("benchmark_results/example.json",)
        ok, records = repro.compare_artifact_snapshots(before, after)
    finally:
        repro.EXPECTED_ARTIFACTS = original

    assert ok is True
    assert records[0]["status"] == "NEW"


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
        artifacts_valid=True,
        artifact_content=[],
        artifacts_exact=True,
        artifact_comparison=[],
        full_tests=False,
    )

    assert "byte-identical" in report
    assert "biological or clinical validity" in report
    assert "general history-conditioned operator" in report
    assert "complete future Temporal-Image prediction" in report
    assert "objective-independent whole-system predictive stabilization" in report


def test_write_json_is_valid_utf8_json(tmp_path: Path):
    path = tmp_path / "report.json"
    repro.write_json(path, {"runner": repro.RUNNER_VERSION})

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {"runner": repro.RUNNER_VERSION}
