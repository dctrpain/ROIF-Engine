from __future__ import annotations

"""
ROIF Engine experimental package runner.

This script executes the currently published research-prototype experiments,
captures their outputs, records the software environment and Git revision, and
writes a reproducible machine-readable summary.

Project stage
-------------
Research prototype.

Evidence scope
--------------
- L0: unit and regression verification;
- L1: controlled synthetic pipeline consistency;
- L2: controlled robustness under predefined perturbations.

This runner does not convert synthetic results into clinical evidence.
"""

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_VERSION = "1.0.0"

EXPERIMENTS = (
    {
        "id": "example_25",
        "title": "Inverse structural-history reconstruction",
        "level": "L1",
        "data_type": "synthetic",
        "command": (
            sys.executable,
            "examples/example_25_inverse_history_reconstruction.py",
        ),
    },
    {
        "id": "example_26",
        "title": "Robustness under noise and partial observability",
        "level": "L2",
        "data_type": "synthetic",
        "command": (
            sys.executable,
            "examples/example_26_inverse_history_robustness.py",
        ),
    },
    {
        "id": "example_27",
        "title": "Unified history-to-action pipeline",
        "level": "L1",
        "data_type": "synthetic",
        "command": (
            sys.executable,
            "examples/example_27_history_pipeline.py",
        ),
    },
    {
        "id": "sls_material",
        "title": "SLS material creep and rheological memory",
        "level": "L0",
        "data_type": "computational",
        "command": (
            sys.executable,
            "-m",
            "pytest",
            "tests/test_creep.py",
            "tests/test_material_creep.py",
            "-q",
        ),
    },
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    experiment_id: str
    title: str
    level: str
    data_type: str
    command: tuple[str, ...]
    status: str
    return_code: int
    duration_seconds: float
    started_at_utc: str
    finished_at_utc: str
    stdout_file: str
    stderr_file: str
    stdout_tail: str
    stderr_tail: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_results_tree(root: Path) -> tuple[Path, Path]:
    results_dir = root / "experiments" / "results"
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, logs_dir


def run_capture(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> tuple[int, float, str, str]:
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr

    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        stderr += (
            f"\nExperiment exceeded timeout of "
            f"{timeout_seconds} seconds.\n"
        )

    duration = time.perf_counter() - started
    return return_code, duration, stdout, stderr


def tail(text: str, *, lines: int = 30) -> str:
    parts = text.rstrip().splitlines()
    return "\n".join(parts[-lines:])


def git_value(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    value = completed.stdout.strip()
    return value or None


def collect_environment(root: Path) -> dict[str, Any]:
    return {
        "package_version": PACKAGE_VERSION,
        "project_stage": "research_prototype",
        "evidence_scope": ["L0", "L1", "L2"],
        "generated_at_utc": utc_now(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "repository": {
            "root": str(root),
            "commit": git_value(root, "rev-parse", "HEAD"),
            "short_commit": git_value(
                root,
                "rev-parse",
                "--short",
                "HEAD",
            ),
            "branch": git_value(
                root,
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ),
            "status_porcelain": git_value(
                root,
                "status",
                "--porcelain",
            ),
        },
    }


def validate_required_files(root: Path) -> list[str]:
    missing: list[str] = []

    required = (
        root / "experiments" / "manifest.json",
        root / "examples" / "example_25_inverse_history_reconstruction.py",
        root / "examples" / "example_26_inverse_history_robustness.py",
        root / "examples" / "example_27_history_pipeline.py",
        root / "tests" / "test_creep.py",
        root / "tests" / "test_material_creep.py",
    )

    for path in required:
        if not path.exists():
            missing.append(str(path.relative_to(root)))

    return missing


def execute_experiment(
    spec: dict[str, Any],
    *,
    root: Path,
    logs_dir: Path,
    timeout_seconds: int,
) -> CommandResult:
    started_at = utc_now()

    return_code, duration, stdout, stderr = run_capture(
        tuple(spec["command"]),
        cwd=root,
        timeout_seconds=timeout_seconds,
    )

    finished_at = utc_now()
    status = "passed" if return_code == 0 else "failed"

    stdout_path = logs_dir / f"{spec['id']}.stdout.txt"
    stderr_path = logs_dir / f"{spec['id']}.stderr.txt"

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    return CommandResult(
        experiment_id=str(spec["id"]),
        title=str(spec["title"]),
        level=str(spec["level"]),
        data_type=str(spec["data_type"]),
        command=tuple(str(item) for item in spec["command"]),
        status=status,
        return_code=return_code,
        duration_seconds=round(duration, 6),
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        stdout_file=str(
            stdout_path.relative_to(root)
        ).replace("\\", "/"),
        stderr_file=str(
            stderr_path.relative_to(root)
        ).replace("\\", "/"),
        stdout_tail=tail(stdout),
        stderr_tail=tail(stderr),
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_markdown_report(
    *,
    environment: dict[str, Any],
    results: list[CommandResult],
) -> str:
    passed = sum(
        result.status == "passed"
        for result in results
    )
    failed = len(results) - passed

    lines = [
        "# ROIF Experimental Package Report",
        "",
        "## Release status",
        "",
        "- **Project stage:** research prototype",
        "- **Validation scope:** controlled synthetic and computational validation",
        "- **Clinical validation:** not performed",
        f"- **Commit:** `{environment['repository']['commit'] or 'unavailable'}`",
        f"- **Generated:** `{environment['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        f"- Experiments: **{len(results)}**",
        f"- Passed: **{passed}**",
        f"- Failed: **{failed}**",
        "",
        "| ID | Level | Data | Status | Duration, s |",
        "|---|---:|---|---|---:|",
    ]

    for result in results:
        lines.append(
            f"| `{result.experiment_id}` "
            f"| {result.level} "
            f"| {result.data_type} "
            f"| **{result.status.upper()}** "
            f"| {result.duration_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "These results demonstrate computational behaviour at the "
                "research-prototype stage. They do not establish unique "
                "real-world causal identification, clinical efficacy, or "
                "generalization to arbitrary biological systems."
            ),
            "",
            "## Experiment details",
            "",
        ]
    )

    for result in results:
        lines.extend(
            [
                f"### {result.experiment_id} — {result.title}",
                "",
                f"- Level: **{result.level}**",
                f"- Data type: **{result.data_type}**",
                f"- Status: **{result.status}**",
                f"- Return code: `{result.return_code}`",
                f"- Duration: `{result.duration_seconds:.6f} s`",
                f"- Command: `{' '.join(result.command)}`",
                f"- Stdout: `{result.stdout_file}`",
                f"- Stderr: `{result.stderr_file}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Required publication label",
            "",
            (
                "> Controlled synthetic validation of a research prototype. "
                "Randomized comparative, external, and clinical validation "
                "have not yet been completed."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    root = project_root()
    results_dir, logs_dir = ensure_results_tree(root)

    print("=" * 96)
    print("ROIF Engine Experimental Package")
    print("Research-prototype reproducibility runner")
    print("=" * 96)
    print(f"Project root : {root}")
    print(f"Results dir  : {results_dir}")
    print()

    missing = validate_required_files(root)

    if missing:
        print("Cannot run package. Missing required files:")

        for item in missing:
            print(f"  - {item}")

        return 2

    environment = collect_environment(root)
    write_json(
        results_dir / "environment.json",
        environment,
    )

    results: list[CommandResult] = []

    for index, spec in enumerate(
        EXPERIMENTS,
        start=1,
    ):
        print(
            f"[{index}/{len(EXPERIMENTS)}] "
            f"{spec['id']}: {spec['title']}"
        )

        result = execute_experiment(
            spec,
            root=root,
            logs_dir=logs_dir,
            timeout_seconds=600,
        )
        results.append(result)

        print(
            f"    status={result.status} "
            f"return_code={result.return_code} "
            f"duration={result.duration_seconds:.3f}s"
        )

    passed = sum(
        result.status == "passed"
        for result in results
    )
    failed = len(results) - passed

    summary = {
        "package_version": PACKAGE_VERSION,
        "project_stage": "research_prototype",
        "validation_scope": (
            "controlled_synthetic_and_computational"
        ),
        "generated_at_utc": utc_now(),
        "commit": environment["repository"]["commit"],
        "branch": environment["repository"]["branch"],
        "experiment_count": len(results),
        "passed": passed,
        "failed": failed,
        "overall_status": (
            "passed"
            if failed == 0
            else "failed"
        ),
        "required_publication_label": (
            "Controlled synthetic validation of a research prototype."
        ),
        "results": [
            asdict(result)
            for result in results
        ],
    }

    write_json(
        results_dir / "summary.json",
        summary,
    )

    report = build_markdown_report(
        environment=environment,
        results=results,
    )
    (
        results_dir / "report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("-" * 96)
    print(f"Passed        : {passed}")
    print(f"Failed        : {failed}")
    print(
        "Summary       : "
        "experiments/results/summary.json"
    )
    print(
        "Environment   : "
        "experiments/results/environment.json"
    )
    print(
        "Report        : "
        "experiments/results/report.md"
    )
    print("-" * 96)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
