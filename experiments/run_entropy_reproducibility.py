"""ROIF Engine manuscript reproducibility runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

RUNNER_VERSION = "roif_entropy_reproducibility_v2"
CLAIM_SCOPE = "computational_model_only"

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility_results"
LOGS_DIR = RESULTS_DIR / "logs"

BENCHMARK_MODULES = (
    ("q1_q3_end_to_end", "experiments.roif_end_to_end_benchmark"),
    ("q4_matched_additive_multiplicative_control", "experiments.roif_additive_multiplicative_control_v2"),
    ("q5_separability", "experiments.roif_separability_benchmark"),
    ("q6_perturbation_stability", "experiments.roif_q6_lyapunov_benchmark"),
    ("physical_history_integration", "experiments.roif_physical_history_integration_benchmark"),
    ("matched_state_physical_history_identifiability", "experiments.roif_matched_state_history_operator_identifiability_benchmark"),
    ("structured_memory_conditioned_matched_state_transition", "experiments.roif_memory_conditioned_matched_state_transition_benchmark"),
    ("restricted_predictive_preconfiguration", "experiments.roif_predictive_stabilization_benchmark"),
    ("q7_objective_independence_audit", "experiments.roif_q7_objective_independence_audit"),
    ("q7_off_nominal_transferability_audit", "experiments.roif_q7_off_nominal_transferability_audit"),
    ("q8_history_conditioned_redistribution_capacity", "experiments.roif_q8_history_conditioned_redistribution_benchmark"),
    ("q8_history_conditioned_redistribution_matched_state", "experiments.roif_q8_history_conditioned_redistribution_matched_state_benchmark"),
    ("temporal_image_reconstruction", "experiments.roif_temporal_image_reconstruction_benchmark"),
    ("multilayer_temporal_image_trajectory", "experiments.roif_multilayer_temporal_image_trajectory_benchmark"),
)

EXPECTED_ARTIFACTS = (
    "benchmark_results/physical_history_integration_v1.json",
    "benchmark_results/separability_benchmark_v1.json",
    "benchmark_results/q6_lyapunov_like_v1.json",
    "benchmark_results/matched_state_history_operator_identifiability_v1.json",
    "benchmark_results/memory_conditioned_matched_state_transition_v1.json",
    "benchmark_results/predictive_stabilization_v1.json",
    "benchmark_results/q7_objective_independence_audit_v1.json",
    "benchmark_results/q7_off_nominal_transferability_audit_v2.json",
    "benchmark_results/q8_history_conditioned_redistribution_capacity_v1.json",
    "benchmark_results/q8_history_conditioned_redistribution_matched_state_v2.json",
    "benchmark_results/temporal_image_reconstruction_v1.json",
    "benchmark_results/temporal_image_reconstruction_methods_v1.csv",
    "benchmark_results/temporal_image_reconstruction_slices_v1.csv",
    "benchmark_results/multilayer_temporal_image_trajectory_v1.json",
)

FOCUSED_TEST_FILES = (
    "tests/test_roif_benchmarks_q1_q5.py",
    "tests/test_build_entropy_figures.py",
    "tests/test_memory_conditioned_matched_state_transition_benchmark.py",
    "tests/test_memory_transition_derivation.py",
    "tests/test_predictive_stabilization_benchmark.py",
    "tests/test_q8_history_conditioned_redistribution_matched_state_benchmark.py",
    "tests/test_roif_architecture_claim_boundaries.py",
    "tests/test_roif_memory_transition_architecture.py",
    "tests/test_roif_temporal_image_reconstruction_benchmark.py",
    "tests/test_system_evolution_transition_modifiers.py",
    "tests/test_transition_modifiers.py",
)


@dataclass(frozen=True)
class StepResult:
    step_id: str
    kind: str
    command: list[str]
    return_code: int
    duration_seconds: float
    status: str
    stdout_log: str
    stderr_log: str


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def git_value(*args: str) -> str | None:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None

    if p.returncode != 0:
        return None

    value = p.stdout.strip()
    return value or None


def is_git_tracked(path: Path) -> bool:
    try:
        p = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return p.returncode == 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict[str, object]:
    return {
        "runner_version": RUNNER_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "describe": git_value("describe", "--tags", "--always", "--dirty"),
        },
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def module_path(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def validate_inputs() -> list[str]:
    missing = [
        rel(module_path(module))
        for _, module in BENCHMARK_MODULES
        if not module_path(module).is_file()
    ]

    builder = ROOT / "experiments" / "build_entropy_figures.py"
    if not builder.is_file():
        missing.append(rel(builder))

    for item in FOCUSED_TEST_FILES:
        if not (ROOT / item).is_file():
            missing.append(item)

    return missing


def run_step(step_id: str, kind: str, command: Sequence[str]) -> StepResult:
    out = LOGS_DIR / f"{step_id}.stdout.txt"
    err = LOGS_DIR / f"{step_id}.stderr.txt"

    started = time.perf_counter()
    p = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    duration = time.perf_counter() - started

    out.write_text(p.stdout, encoding="utf-8")
    err.write_text(p.stderr, encoding="utf-8")

    return StepResult(
        step_id=step_id,
        kind=kind,
        command=list(command),
        return_code=p.returncode,
        duration_seconds=duration,
        status="passed" if p.returncode == 0 else "failed",
        stdout_log=rel(out),
        stderr_log=rel(err),
    )


def snapshot_artifacts() -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}

    for name in EXPECTED_ARTIFACTS:
        path = ROOT / name
        exists = path.is_file()

        record: dict[str, object] = {
            "path": name,
            "exists": exists,
            "tracked": is_git_tracked(path),
        }

        if exists:
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)

        snapshot[name] = record

    return snapshot


def compare_artifact_snapshots(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> tuple[bool, list[dict[str, object]]]:
    all_ok = True
    records: list[dict[str, object]] = []

    for name in EXPECTED_ARTIFACTS:
        b = before[name]
        a = after[name]

        before_exists = bool(b.get("exists"))
        after_exists = bool(a.get("exists"))
        tracked = bool(b.get("tracked") or a.get("tracked"))

        if before_exists and after_exists:
            same_hash = b.get("sha256") == a.get("sha256")
            status = "REPRODUCED_EXACTLY" if same_hash else "CHANGED"
        elif before_exists and not after_exists:
            same_hash = False
            status = "MISSING"
        elif not before_exists and after_exists:
            same_hash = False
            status = "NEW"
        else:
            same_hash = False
            status = "MISSING"

        # Reviewer-grade reproducibility requires tracked expected artifacts to
        # remain byte-identical after rerunning the benchmark chain.
        if tracked and status != "REPRODUCED_EXACTLY":
            all_ok = False

        records.append(
            {
                "path": name,
                "tracked": tracked,
                "status": status,
                "before_sha256": b.get("sha256"),
                "after_sha256": a.get("sha256"),
                "before_size_bytes": b.get("size_bytes"),
                "after_size_bytes": a.get("size_bytes"),
            }
        )

    return all_ok, records


def validate_artifact_content() -> tuple[bool, list[dict[str, object]]]:
    all_ok = True
    records: list[dict[str, object]] = []

    for name in EXPECTED_ARTIFACTS:
        path = ROOT / name
        rec: dict[str, object] = {"path": name, "exists": path.is_file()}

        if not path.is_file():
            rec["valid"] = False
            rec["error"] = "missing"
            all_ok = False
        else:
            rec["size_bytes"] = path.stat().st_size
            rec["sha256"] = sha256_file(path)
            try:
                if path.stat().st_size == 0:
                    raise ValueError("empty file")
                if path.suffix.lower() == ".json":
                    json.loads(path.read_text(encoding="utf-8-sig"))
                rec["valid"] = True
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                rec["valid"] = False
                rec["error"] = f"{type(exc).__name__}: {exc}"
                all_ok = False

        records.append(rec)

    return all_ok, records


def markdown_report(
    env: dict[str, object],
    steps: list[StepResult],
    artifacts_valid: bool,
    artifact_content: list[dict[str, object]],
    artifacts_exact: bool,
    artifact_comparison: list[dict[str, object]],
    full_tests: bool,
) -> str:
    passed = sum(x.status == "passed" for x in steps)

    lines = [
        "# ROIF Manuscript Reproducibility Report",
        "",
        f"- Runner: `{RUNNER_VERSION}`",
        f"- Claim scope: `{CLAIM_SCOPE}`",
        f"- Generated: `{env['generated_at_utc']}`",
        f"- Commit: `{env['repository']['commit']}`",
        f"- Git state: `{env['repository']['describe']}`",
        "",
        "## Summary",
        "",
        f"- Executed steps: **{len(steps)}**",
        f"- Passed steps: **{passed}**",
        f"- Failed steps: **{len(steps) - passed}**",
        f"- Expected artifact content valid: **{artifacts_valid}**",
        f"- Tracked artifact equality before/after run: **{artifacts_exact}**",
        f"- Full repository tests requested: **{full_tests}**",
        "",
        "| Step | Kind | Status | Duration, s |",
        "|---|---|---|---:|",
    ]

    for x in steps:
        lines.append(
            f"| `{x.step_id}` | {x.kind} | "
            f"**{x.status.upper()}** | {x.duration_seconds:.3f} |"
        )

    lines += [
        "",
        "## Artifact equality",
        "",
        "| Artifact | Tracked | Status | Before SHA-256 | After SHA-256 |",
        "|---|---|---|---|---|",
    ]

    for a in artifact_comparison:
        lines.append(
            f"| `{a['path']}` | {a['tracked']} | **{a['status']}** | "
            f"`{a.get('before_sha256') or ''}` | "
            f"`{a.get('after_sha256') or ''}` |"
        )

    lines += [
        "",
        "## Artifact content validation",
        "",
        "| Artifact | Exists | Valid | SHA-256 |",
        "|---|---|---|---|",
    ]

    for a in artifact_content:
        lines.append(
            f"| `{a['path']}` | {a.get('exists', False)} | "
            f"{a.get('valid', False)} | `{a.get('sha256', '')}` |"
        )

    lines += [
        "",
        "## Scientific interpretation boundary",
        "",
        "A successful run reproduces the computational benchmark paths and "
        "publication assets implemented in this repository and leaves the "
        "tracked expected benchmark artifacts byte-identical.",
        "",
        "It does **not** by itself establish:",
        "",
        "- biological or clinical validity;",
        "- a general history-conditioned operator over the complete `SystemImage`;",
        "- complete future Temporal-Image prediction;",
        "- objective-independent whole-system predictive stabilization;",
        "- biological learning;",
        "- universal stability.",
        "",
        "The predictive benchmark reproduced here is the restricted tested "
        "prestress-preconfiguration mechanism. The Temporal Image benchmark "
        "reproduces trajectory/reconstruction behavior rather than ordinary "
        "whole-system future forecasting.",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-tests",
        action="store_true",
        help="Also run the complete repository pytest suite.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop benchmark execution after the first failed step.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("ROIF Engine — Manuscript Reproducibility Runner")
    print(f"Runner version: {RUNNER_VERSION}")
    print("=" * 88)

    missing = validate_inputs()
    if missing:
        print("ERROR: required reproducibility files are missing:")
        for item in missing:
            print(f"  - {item}")
        return 2

    env = environment()
    write_json(RESULTS_DIR / "environment.json", env)

    before = snapshot_artifacts()
    write_json(RESULTS_DIR / "artifact_snapshot_before.json", before)

    steps: list[StepResult] = []

    def execute(step_id: str, kind: str, command: Sequence[str]) -> bool:
        print(f"[RUN]  {step_id}")
        result = run_step(step_id, kind, command)
        steps.append(result)

        print(
            f"[{'PASS' if result.status == 'passed' else 'FAIL'}] "
            f"{step_id} ({result.duration_seconds:.3f} s)"
        )

        if result.status == "failed":
            print(f"       stdout: {result.stdout_log}")
            print(f"       stderr: {result.stderr_log}")
            return False

        return True

    for step_id, module in BENCHMARK_MODULES:
        ok = execute(step_id, "benchmark", [sys.executable, "-m", module])
        if not ok and args.stop_on_failure:
            break

    if not any(x.status == "failed" for x in steps):
        execute(
            "build_entropy_figures",
            "publication_assets",
            [sys.executable, "-m", "experiments.build_entropy_figures"],
        )

    if not any(x.status == "failed" for x in steps):
        execute(
            "focused_manuscript_tests",
            "tests",
            [sys.executable, "-m", "pytest", "-q", *FOCUSED_TEST_FILES],
        )

    if args.full_tests and not any(x.status == "failed" for x in steps):
        execute(
            "full_repository_tests",
            "tests",
            [sys.executable, "-m", "pytest", "-q"],
        )

    after = snapshot_artifacts()
    write_json(RESULTS_DIR / "artifact_snapshot_after.json", after)

    artifacts_exact, artifact_comparison = compare_artifact_snapshots(before, after)
    artifacts_valid, artifact_content = validate_artifact_content()

    report_payload = {
        "runner_version": RUNNER_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "environment": env,
        "steps": [asdict(x) for x in steps],
        "artifacts_valid": artifacts_valid,
        "artifacts_exact": artifacts_exact,
        "artifact_content": artifact_content,
        "artifact_comparison": artifact_comparison,
        "full_tests_requested": bool(args.full_tests),
    }

    write_json(RESULTS_DIR / "reproduction_report.json", report_payload)

    (RESULTS_DIR / "reproduction_report.md").write_text(
        markdown_report(
            env,
            steps,
            artifacts_valid,
            artifact_content,
            artifacts_exact,
            artifact_comparison,
            bool(args.full_tests),
        ) + "\n",
        encoding="utf-8",
    )

    failed_steps = any(x.status == "failed" for x in steps)
    success = (not failed_steps) and artifacts_valid and artifacts_exact

    print()
    print("=" * 88)
    print(f"ARTIFACT EQUALITY: {'PASS' if artifacts_exact else 'FAIL'}")
    print(f"REPRODUCIBILITY RESULT: {'PASS' if success else 'FAIL'}")
    print("Report: reproducibility_results/reproduction_report.md")
    print("JSON  : reproducibility_results/reproduction_report.json")
    print("=" * 88)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
