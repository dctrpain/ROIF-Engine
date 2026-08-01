"""Tests for roif.cli."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Any
import json

import pytest

from roif.cli import (
    CLIContext,
    CLIError,
    ExitCode,
    OutputFormat,
    build_parser,
    run,
)
from roif.datasets import (
    Dataset,
    DatasetKind,
    DatasetRecord,
    DatasetSchema,
    load_dataset,
    save_dataset,
)
from roif.reporting import (
    ROIFReportBuilder,
    ReportSectionKind,
    render_json,
)
from roif.serialization import (
    save_json,
    save_npz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dataset(
    *,
    count: int = 6,
    name: str = "demo",
) -> Dataset:
    records = tuple(
        DatasetRecord(
            record_id=f"r{index}",
            features={
                "x": float(index),
                "y": index + 1,
            },
            target=index % 2,
            group="A" if index % 2 == 0 else "B",
            tags=("even",) if index % 2 == 0 else ("odd",),
            metadata={"index": index},
        )
        for index in range(count)
    )

    return Dataset(
        name=name,
        records=records,
        kind=DatasetKind.SYNTHETIC,
        version="1.0",
        description="CLI test dataset.",
        schema=DatasetSchema(
            required_features=("x",),
            optional_features=("y",),
            allow_extra_features=False,
            require_target=True,
        ),
        metadata={"source": "test"},
    )


def make_report_mapping() -> dict[str, Any]:
    report = (
        ROIFReportBuilder(
            "report-1",
            title="CLI Report",
            created_at="2026-08-01T10:00:00Z",
        )
        .set_summary("CLI report summary.")
        .add_result(
            "analysis",
            {"eta": 0.5},
            title="Analysis",
            kind=ReportSectionKind.METRICS,
        )
        .build()
    )

    return report.as_dict()


def invoke(
    argv: list[str],
) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()

    code = run(
        argv,
        stdout=stdout,
        stderr=stderr,
    )

    return code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


def test_exit_code_values_are_stable() -> None:
    assert tuple(item.value for item in ExitCode) == (
        0,
        2,
        3,
        4,
        70,
    )


def test_output_format_values_are_stable() -> None:
    assert tuple(item.value for item in OutputFormat) == (
        "json",
        "yaml",
        "csv",
        "markdown",
        "text",
    )


# ---------------------------------------------------------------------------
# CLIContext
# ---------------------------------------------------------------------------


def test_context_write_and_error() -> None:
    stdout = StringIO()
    stderr = StringIO()
    context = CLIContext(
        stdout=stdout,
        stderr=stderr,
    )

    context.write("hello")
    context.error("problem")

    assert stdout.getvalue() == "hello\n"
    assert stderr.getvalue() == "problem\n"


def test_context_write_preserves_existing_newline() -> None:
    stdout = StringIO()
    context = CLIContext(
        stdout=stdout,
        stderr=StringIO(),
    )

    context.write("hello\n")

    assert stdout.getvalue() == "hello\n"


def test_context_quiet_suppresses_normal_output() -> None:
    stdout = StringIO()
    context = CLIContext(
        stdout=stdout,
        stderr=StringIO(),
        quiet=True,
    )

    context.write("hidden")

    assert stdout.getvalue() == ""


def test_context_debug_requires_verbose() -> None:
    stderr = StringIO()

    CLIContext(
        stdout=StringIO(),
        stderr=stderr,
        verbose=False,
    ).debug("hidden")

    assert stderr.getvalue() == ""

    CLIContext(
        stdout=StringIO(),
        stderr=stderr,
        verbose=True,
    ).debug("visible")

    assert stderr.getvalue() == "visible\n"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_build_parser() -> None:
    parser = build_parser()

    assert parser.prog == "roif"


@pytest.mark.parametrize(
    "argv",
    (
        ["version"],
        ["doctor"],
        ["inspect", "result.json"],
        ["convert", "a.json", "b.yaml"],
        ["report", "report.json"],
        ["dataset", "info", "dataset.json"],
        ["dataset", "validate", "dataset.json"],
        [
            "dataset",
            "split",
            "dataset.json",
            "--output-dir",
            "splits",
        ],
        [
            "dataset",
            "filter",
            "dataset.json",
            "--output",
            "filtered.json",
        ],
    ),
)
def test_parser_accepts_supported_commands(
    argv: list[str],
) -> None:
    namespace = build_parser().parse_args(argv)

    assert callable(namespace.handler)


def test_parser_requires_command() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])

    assert exc.value.code == ExitCode.USAGE


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_text() -> None:
    code, stdout, stderr = invoke(["version"])

    assert code == ExitCode.OK
    assert stdout.startswith("ROIF Engine ")
    assert stderr == ""


def test_version_json() -> None:
    code, stdout, stderr = invoke(
        ["version", "--format", "json"]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["name"] == "ROIF Engine"
    assert "version" in payload
    assert "python" in payload
    assert stderr == ""


def test_version_yaml() -> None:
    pytest.importorskip("yaml")

    code, stdout, stderr = invoke(
        ["version", "--format", "yaml"]
    )

    assert code == ExitCode.OK
    assert "name: ROIF Engine" in stdout
    assert stderr == ""


def test_quiet_suppresses_version_output() -> None:
    code, stdout, stderr = invoke(
        ["--quiet", "version"]
    )

    assert code == ExitCode.OK
    assert stdout == ""
    assert stderr == ""


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_text() -> None:
    code, stdout, stderr = invoke(["doctor"])

    assert code in (ExitCode.OK, ExitCode.DATA_ERROR)
    assert "[OK] python:" in stdout
    assert "[OK] numpy:" in stdout
    assert "ROIF environment" in stdout
    assert stderr == ""


def test_doctor_json() -> None:
    code, stdout, stderr = invoke(
        ["doctor", "--format", "json"]
    )

    payload = json.loads(stdout)

    assert code in (ExitCode.OK, ExitCode.DATA_ERROR)
    assert payload["status"] in ("ok", "degraded")
    assert isinstance(payload["checks"], list)
    assert any(
        item["name"] == "python"
        for item in payload["checks"]
    )
    assert stderr == ""


# ---------------------------------------------------------------------------
# dataset info and validation
# ---------------------------------------------------------------------------


def test_dataset_info_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    save_dataset(make_dataset(), path)

    code, stdout, stderr = invoke(
        ["dataset", "info", str(path)]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["name"] == "demo"
    assert payload["kind"] == "synthetic"
    assert payload["summary"]["record_count"] == 6
    assert payload["ids"] is None
    assert stderr == ""


def test_dataset_info_can_include_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.json"
    save_dataset(make_dataset(count=3), path)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "info",
            str(path),
            "--include-ids",
        ]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["ids"] == ["r0", "r1", "r2"]
    assert stderr == ""


def test_dataset_info_text(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    save_dataset(make_dataset(count=2), path)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "info",
            str(path),
            "--format",
            "text",
        ]
    )

    assert code == ExitCode.OK
    assert '"record_count": 2' in stdout
    assert stderr == ""


def test_dataset_validate(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    save_dataset(make_dataset(count=4), path)

    code, stdout, stderr = invoke(
        ["dataset", "validate", str(path)]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload == {
        "valid": True,
        "name": "demo",
        "record_count": 4,
        "schema_present": True,
    }
    assert stderr == ""


def test_dataset_command_reports_missing_file(
    tmp_path: Path,
) -> None:
    code, stdout, stderr = invoke(
        [
            "dataset",
            "info",
            str(tmp_path / "missing.json"),
        ]
    )

    assert code == ExitCode.DATA_ERROR
    assert stdout == ""
    assert "error:" in stderr
    assert "does not exist" in stderr


# ---------------------------------------------------------------------------
# dataset split
# ---------------------------------------------------------------------------


def test_dataset_split_creates_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.json"
    output_dir = tmp_path / "splits"
    save_dataset(make_dataset(count=10), source)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "split",
            str(source),
            "--output-dir",
            str(output_dir),
            "--train",
            "0.6",
            "--validation",
            "0.2",
            "--test",
            "0.2",
            "--seed",
            "42",
        ]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["counts"] == {
        "train": 6,
        "validation": 2,
        "test": 2,
        "total": 10,
    }
    assert (output_dir / "train.json").is_file()
    assert (output_dir / "validation.json").is_file()
    assert (output_dir / "test.json").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert len(load_dataset(output_dir / "train.json")) == 6
    assert stderr == ""


def test_dataset_split_stratifies_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.json"
    output_dir = tmp_path / "splits"
    save_dataset(make_dataset(count=20), source)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "split",
            str(source),
            "--output-dir",
            str(output_dir),
            "--train",
            "0.5",
            "--validation",
            "0.25",
            "--test",
            "0.25",
            "--stratify-target",
        ]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["stratified"] is True
    assert {
        item.target
        for item in load_dataset(
            output_dir / "train.json"
        )
    } == {0, 1}
    assert stderr == ""


def test_dataset_split_rejects_bad_ratios(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.json"
    save_dataset(make_dataset(), source)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "split",
            str(source),
            "--output-dir",
            str(tmp_path / "splits"),
            "--train",
            "0.8",
            "--validation",
            "0.2",
            "--test",
            "0.2",
        ]
    )

    assert code == ExitCode.DATA_ERROR
    assert stdout == ""
    assert "Split ratios must sum to 1" in stderr


# ---------------------------------------------------------------------------
# dataset filter
# ---------------------------------------------------------------------------


def test_dataset_filter_by_group(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.json"
    output = tmp_path / "filtered.json"
    save_dataset(make_dataset(count=6), source)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "filter",
            str(source),
            "--output",
            str(output),
            "--group",
            "A",
        ]
    )

    payload = json.loads(stdout)
    filtered = load_dataset(output)

    assert code == ExitCode.OK
    assert payload["record_count"] == 3
    assert all(item.group == "A" for item in filtered)
    assert stderr == ""


def test_dataset_filter_by_tag(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.json"
    output = tmp_path / "filtered.json"
    save_dataset(make_dataset(count=6), source)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "filter",
            str(source),
            "--output",
            str(output),
            "--tag",
            "odd",
        ]
    )

    filtered = load_dataset(output)

    assert code == ExitCode.OK
    assert len(filtered) == 3
    assert all("odd" in item.tags for item in filtered)
    assert stderr == ""


def test_dataset_filter_by_json_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.json"
    output = tmp_path / "filtered.json"
    save_dataset(make_dataset(count=6), source)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "filter",
            str(source),
            "--output",
            str(output),
            "--target",
            "1",
        ]
    )

    filtered = load_dataset(output)

    assert code == ExitCode.OK
    assert len(filtered) == 3
    assert all(item.target == 1 for item in filtered)
    assert stderr == ""


def test_dataset_filter_custom_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.json"
    output = tmp_path / "filtered.json"
    save_dataset(make_dataset(count=2), source)

    code, stdout, stderr = invoke(
        [
            "dataset",
            "filter",
            str(source),
            "--output",
            str(output),
            "--name",
            "selected",
        ]
    )

    assert code == ExitCode.OK
    assert load_dataset(output).name == "selected"
    assert stderr == ""


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_json_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    save_json(
        {
            "eta": 0.5,
            "nodes": ["A", "B"],
        },
        path,
    )

    code, stdout, stderr = invoke(
        ["inspect", str(path)]
    )

    assert code == ExitCode.OK
    assert json.loads(stdout) == {
        "eta": 0.5,
        "nodes": ["A", "B"],
    }
    assert stderr == ""


def test_inspect_csv_file(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text(
        "id,name\n1,A\n2,B\n",
        encoding="utf-8",
    )

    code, stdout, stderr = invoke(
        [
            "inspect",
            str(path),
            "--format",
            "json",
        ]
    )

    assert code == ExitCode.OK
    assert json.loads(stdout) == [
        {"id": "1", "name": "A"},
        {"id": "2", "name": "B"},
    ]
    assert stderr == ""


def test_inspect_npz_file(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    save_npz(
        {"state": [1.0, 2.0]},
        path,
        metadata={"experiment": "E1"},
    )

    code, stdout, stderr = invoke(
        ["inspect", str(path)]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["arrays"]["state"] == [1.0, 2.0]
    assert payload["metadata"]["experiment"] == "E1"
    assert stderr == ""


def test_inspect_missing_file(tmp_path: Path) -> None:
    code, stdout, stderr = invoke(
        ["inspect", str(tmp_path / "missing.json")]
    )

    assert code == ExitCode.DATA_ERROR
    assert stdout == ""
    assert "does not exist" in stderr


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def test_convert_json_to_yaml(
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")

    source = tmp_path / "source.json"
    output = tmp_path / "output.yaml"
    save_json({"a": 1, "b": [2, 3]}, source)

    code, stdout, stderr = invoke(
        [
            "convert",
            str(source),
            str(output),
            "--format",
            "json",
        ]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["input_format"] == "json"
    assert payload["output_format"] == "yaml"
    assert output.is_file()
    assert "a: 1" in output.read_text(encoding="utf-8")
    assert stderr == ""


def test_convert_yaml_to_json(
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")

    source = tmp_path / "source.yaml"
    output = tmp_path / "output.json"
    source.write_text(
        "a: 1\nb:\n  - 2\n  - 3\n",
        encoding="utf-8",
    )

    code, stdout, stderr = invoke(
        [
            "convert",
            str(source),
            str(output),
        ]
    )

    assert code == ExitCode.OK
    assert json.loads(
        output.read_text(encoding="utf-8")
    ) == {
        "a": 1,
        "b": [2, 3],
    }
    assert stderr == ""


def test_convert_json_to_csv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.csv"
    save_json(
        [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ],
        source,
    )

    code, stdout, stderr = invoke(
        [
            "convert",
            str(source),
            str(output),
        ]
    )

    assert code == ExitCode.OK
    assert output.read_text(
        encoding="utf-8"
    ).splitlines()[0] == "id,name"
    assert stderr == ""


def test_convert_rejects_scalar_to_csv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.csv"
    save_json(5, source)

    code, stdout, stderr = invoke(
        [
            "convert",
            str(source),
            str(output),
        ]
    )

    assert code == ExitCode.DATA_ERROR
    assert stdout == ""
    assert "CSV output requires" in stderr


def test_convert_uses_explicit_output_format(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.data"
    save_json({"a": 1}, source)

    code, stdout, stderr = invoke(
        [
            "convert",
            str(source),
            str(output),
            "--output-format",
            "json",
        ]
    )

    payload = json.loads(
        stdout
        if stdout.lstrip().startswith("{")
        else "{}"
    )

    assert code == ExitCode.OK
    assert output.is_file()
    assert json.loads(
        output.read_text(encoding="utf-8")
    ) == {"a": 1}
    assert stderr == ""


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_markdown_to_stdout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    save_json(make_report_mapping(), path)

    code, stdout, stderr = invoke(
        ["report", str(path)]
    )

    assert code == ExitCode.OK
    assert "# CLI Report" in stdout
    assert "## Analysis" in stdout
    assert "eta" in stdout
    assert stderr == ""


def test_report_json_to_stdout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    save_json(make_report_mapping(), path)

    code, stdout, stderr = invoke(
        [
            "report",
            str(path),
            "--format",
            "json",
        ]
    )

    payload = json.loads(stdout)

    assert code == ExitCode.OK
    assert payload["report_id"] == "report-1"
    assert payload["status"] == "completed"
    assert stderr == ""


def test_report_text_to_stdout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    save_json(make_report_mapping(), path)

    code, stdout, stderr = invoke(
        [
            "report",
            str(path),
            "--format",
            "text",
        ]
    )

    assert code == ExitCode.OK
    assert "CLI Report" in stdout
    assert "Status: completed" in stdout
    assert "Sections: 1" in stdout
    assert stderr == ""


def test_report_writes_output_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    output = tmp_path / "report.md"
    save_json(make_report_mapping(), path)

    code, stdout, stderr = invoke(
        [
            "report",
            str(path),
            "--output",
            str(output),
        ]
    )

    assert code == ExitCode.OK
    assert stdout.strip() == str(output)
    assert "# CLI Report" in output.read_text(
        encoding="utf-8"
    )
    assert stderr == ""


def test_report_extracts_nested_engine_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "engine-result.json"
    save_json(
        {
            "status": "completed",
            "report": make_report_mapping(),
        },
        path,
    )

    code, stdout, stderr = invoke(
        ["report", str(path)]
    )

    assert code == ExitCode.OK
    assert "# CLI Report" in stdout
    assert stderr == ""


def test_report_rejects_missing_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.json"
    save_json({"value": 1}, path)

    code, stdout, stderr = invoke(
        ["report", str(path)]
    )

    assert code == ExitCode.DATA_ERROR
    assert stdout == ""
    assert "No ROIF report was found" in stderr


# ---------------------------------------------------------------------------
# Error handling and verbosity
# ---------------------------------------------------------------------------


def test_verbose_includes_traceback(
    tmp_path: Path,
) -> None:
    code, stdout, stderr = invoke(
        [
            "--verbose",
            "inspect",
            str(tmp_path / "missing.json"),
        ]
    )

    assert code == ExitCode.DATA_ERROR
    assert stdout == ""
    assert "Traceback" in stderr


def test_unknown_extension_returns_data_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.unknown"
    path.write_text("data", encoding="utf-8")

    code, stdout, stderr = invoke(
        ["inspect", str(path)]
    )

    assert code == ExitCode.DATA_ERROR
    assert stdout == ""
    assert "Unsupported file extension" in stderr
