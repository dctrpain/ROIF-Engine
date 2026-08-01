"""
Command-line interface for ROIF Engine v1.

The CLI exposes the infrastructure already implemented in the package:

- dataset inspection and splitting;
- serialization conversion;
- validation and benchmark result inspection;
- report rendering;
- engine result inspection;
- package diagnostics and version output.

The module is deliberately conservative: it does not invent domain-specific
analysis callbacks. Commands that require a configured scientific pipeline
operate on serialized results or on datasets unless an application registers
custom handlers around ``main()``.

Example usage:

    python -m roif.cli version
    python -m roif.cli doctor
    python -m roif.cli dataset info data.json
    python -m roif.cli dataset split data.json --output-dir splits
    python -m roif.cli convert result.json result.yaml
    python -m roif.cli report result.json --format markdown
"""

from __future__ import annotations

from argparse import (
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TextIO
import json
import os
import platform
import sys
import traceback

from .datasets import (
    Dataset,
    DatasetError,
    DatasetKind,
    dataset_from_mapping,
    load_dataset,
    save_dataset,
    split_dataset,
)
from .reporting import (
    ROIFReport,
    ReportArtifact,
    ReportSection,
    ReportSectionKind,
    ReportStatus,
    ReportWarning,
    ReproducibilityInfo,
    WarningSeverity,
    render_json,
    render_markdown,
)
from .serialization import (
    SerializationEnvelope,
    SerializationError,
    SerializationFormat,
    dumps_csv,
    dumps_json,
    dumps_yaml,
    envelope_from_mapping,
    infer_format,
    load_csv,
    load_json,
    load_npz,
    load_yaml,
    save_csv,
    save_json,
    save_npz,
    save_yaml,
    to_serializable,
)


class CLIError(RuntimeError):
    """Raised for user-facing command-line failures."""


class ExitCode(int, Enum):
    """Stable process exit codes."""

    OK = 0
    USAGE = 2
    DATA_ERROR = 3
    EXECUTION_ERROR = 4
    INTERNAL_ERROR = 70


class OutputFormat(str, Enum):
    """Supported CLI output formats."""

    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class CLIContext:
    """Execution context for a CLI command."""

    stdout: TextIO
    stderr: TextIO
    verbose: bool = False
    quiet: bool = False

    def write(self, text: str = "") -> None:
        if not self.quiet:
            self.stdout.write(text)
            if not text.endswith("\n"):
                self.stdout.write("\n")

    def error(self, text: str) -> None:
        self.stderr.write(text)
        if not text.endswith("\n"):
            self.stderr.write("\n")

    def debug(self, text: str) -> None:
        if self.verbose:
            self.error(text)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("roif-engine")
    except Exception:
        return "1.0.0"


def _path(value: str) -> Path:
    path = Path(value)
    if not str(path).strip():
        raise CLIError("Path cannot be empty.")
    return path


def _read_serialized(path: Path) -> Any:
    try:
        format_ = infer_format(path)
    except SerializationError as exc:
        raise CLIError(str(exc)) from exc

    try:
        if format_ is SerializationFormat.JSON:
            return load_json(path)
        if format_ is SerializationFormat.YAML:
            return load_yaml(path)
        if format_ is SerializationFormat.CSV:
            return load_csv(path)
        if format_ is SerializationFormat.NPZ:
            return load_npz(path)
    except SerializationError as exc:
        raise CLIError(str(exc)) from exc

    raise CLIError(f"Unsupported input format: {format_.value}.")


def _write_serialized(
    value: Any,
    path: Path,
    *,
    format_: SerializationFormat | None = None,
) -> Path:
    selected = format_ or infer_format(path)

    try:
        if selected is SerializationFormat.JSON:
            return save_json(value, path)
        if selected is SerializationFormat.YAML:
            return save_yaml(value, path)
        if selected is SerializationFormat.CSV:
            if isinstance(value, Mapping):
                rows = (value,)
            elif isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes),
            ):
                rows = value
            else:
                raise CLIError(
                    "CSV output requires a mapping or a sequence of mappings."
                )
            return save_csv(rows, path)
        if selected is SerializationFormat.NPZ:
            if hasattr(value, "arrays"):
                return save_npz(value, path)
            if not isinstance(value, Mapping):
                raise CLIError("NPZ output requires a mapping of arrays.")
            return save_npz(value, path)
    except SerializationError as exc:
        raise CLIError(str(exc)) from exc

    raise CLIError(f"Unsupported output format: {selected.value}.")


def _render_value(
    value: Any,
    *,
    format_: OutputFormat,
) -> str:
    serializable = to_serializable(value)

    if format_ is OutputFormat.JSON:
        return dumps_json(serializable)
    if format_ is OutputFormat.YAML:
        return dumps_yaml(serializable)
    if format_ is OutputFormat.CSV:
        if isinstance(serializable, Mapping):
            return dumps_csv((serializable,))
        if isinstance(serializable, list) and all(
            isinstance(item, Mapping) for item in serializable
        ):
            return dumps_csv(serializable)
        raise CLIError(
            "CSV rendering requires a mapping or a list of mappings."
        )
    if format_ is OutputFormat.TEXT:
        if isinstance(serializable, (dict, list)):
            return json.dumps(
                serializable,
                ensure_ascii=False,
                indent=2,
            )
        return str(serializable)

    raise CLIError(
        f"Output format {format_.value!r} is not valid for this command."
    )


def _mapping_to_report(value: Mapping[str, Any]) -> ROIFReport:
    try:
        sections = tuple(
            ReportSection(
                section_id=item["section_id"],
                title=item["title"],
                kind=ReportSectionKind(item["kind"]),
                content=item["content"],
                order=item.get("order", 0),
                summary=item.get("summary"),
                metadata=item.get("metadata", {}),
            )
            for item in value.get("sections", ())
        )
        warnings = tuple(
            ReportWarning(
                code=item["code"],
                message=item["message"],
                severity=WarningSeverity(item.get("severity", "warning")),
                source=item.get("source"),
                metadata=item.get("metadata", {}),
            )
            for item in value.get("warnings", ())
        )
        artifacts = tuple(
            ReportArtifact(
                name=item["name"],
                kind=item["kind"],
                location=item["location"],
                description=item.get("description"),
                checksum=item.get("checksum"),
                metadata=item.get("metadata", {}),
            )
            for item in value.get("artifacts", ())
        )

        reproducibility_payload = value.get("reproducibility")
        reproducibility = (
            None
            if reproducibility_payload is None
            else ReproducibilityInfo(
                engine_version=reproducibility_payload["engine_version"],
                schema_version=reproducibility_payload.get(
                    "schema_version",
                    "1.0",
                ),
                commit=reproducibility_payload.get("commit"),
                seed=reproducibility_payload.get("seed"),
                command=reproducibility_payload.get("command"),
                environment=reproducibility_payload.get(
                    "environment",
                    {},
                ),
            )
        )

        return ROIFReport(
            report_id=value["report_id"],
            title=value["title"],
            status=ReportStatus(value["status"]),
            created_at=value["created_at"],
            summary=value["summary"],
            sections=sections,
            warnings=warnings,
            artifacts=artifacts,
            reproducibility=reproducibility,
            duration_seconds=value.get("duration_seconds"),
            metadata=value.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CLIError("Input does not contain a valid ROIF report.") from exc


def _extract_report(value: Any) -> ROIFReport:
    if isinstance(value, ROIFReport):
        return value

    if isinstance(value, Mapping):
        if "report_id" in value and "sections" in value:
            return _mapping_to_report(value)

        nested = value.get("report")
        if isinstance(nested, Mapping):
            return _mapping_to_report(nested)

        if "payload" in value and "object_type" in value:
            try:
                envelope = envelope_from_mapping(value)
            except SerializationError as exc:
                raise CLIError(str(exc)) from exc
            if isinstance(envelope.payload, Mapping):
                return _extract_report(envelope.payload)

    raise CLIError("No ROIF report was found in the input.")


def _command_version(args: Namespace, context: CLIContext) -> int:
    payload = {
        "name": "ROIF Engine",
        "version": _package_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    format_ = OutputFormat(args.format)
    if format_ is OutputFormat.TEXT:
        context.write(f"ROIF Engine {_package_version()}")
    else:
        context.write(_render_value(payload, format_=format_))
    return ExitCode.OK


def _command_doctor(args: Namespace, context: CLIContext) -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({
            "name": name,
            "ok": ok,
            "detail": detail,
        })

    add(
        "python",
        sys.version_info >= (3, 11),
        platform.python_version(),
    )

    try:
        import numpy

        add("numpy", True, numpy.__version__)
    except Exception as exc:
        add("numpy", False, str(exc))

    try:
        import yaml

        add("pyyaml", True, getattr(yaml, "__version__", "available"))
    except Exception as exc:
        add("pyyaml", False, str(exc))

    add(
        "filesystem",
        os.access(Path.cwd(), os.R_OK | os.W_OK),
        str(Path.cwd()),
    )

    status = all(item["ok"] for item in checks)
    payload = {
        "status": "ok" if status else "degraded",
        "checks": checks,
    }

    format_ = OutputFormat(args.format)
    if format_ is OutputFormat.TEXT:
        for item in checks:
            marker = "OK" if item["ok"] else "FAIL"
            context.write(
                f"[{marker}] {item['name']}: {item['detail']}"
            )
        context.write(
            "ROIF environment is ready."
            if status
            else "ROIF environment has failed checks."
        )
    else:
        context.write(_render_value(payload, format_=format_))

    return ExitCode.OK if status else ExitCode.DATA_ERROR


def _command_dataset_info(
    args: Namespace,
    context: CLIContext,
) -> int:
    try:
        dataset = load_dataset(_path(args.path))
    except DatasetError as exc:
        raise CLIError(str(exc)) from exc

    payload = {
        "name": dataset.name,
        "kind": dataset.kind.value,
        "version": dataset.version,
        "description": dataset.description,
        "ids": list(dataset.ids) if args.include_ids else None,
        "summary": dataset.summary().as_dict(),
        "schema": (
            None
            if dataset.schema is None
            else dataset.schema.as_dict()
        ),
        "metadata": dict(dataset.metadata),
    }

    context.write(
        _render_value(
            payload,
            format_=OutputFormat(args.format),
        )
    )
    return ExitCode.OK


def _command_dataset_validate(
    args: Namespace,
    context: CLIContext,
) -> int:
    try:
        dataset = load_dataset(_path(args.path))
    except DatasetError as exc:
        raise CLIError(str(exc)) from exc

    payload = {
        "valid": True,
        "name": dataset.name,
        "record_count": len(dataset),
        "schema_present": dataset.schema is not None,
    }
    context.write(
        _render_value(
            payload,
            format_=OutputFormat(args.format),
        )
    )
    return ExitCode.OK


def _command_dataset_split(
    args: Namespace,
    context: CLIContext,
) -> int:
    try:
        dataset = load_dataset(_path(args.path))
        split = split_dataset(
            dataset,
            train_ratio=args.train,
            validation_ratio=args.validation,
            test_ratio=args.test,
            seed=args.seed,
            stratify_by=(
                (lambda item: item.target)
                if args.stratify_target
                else None
            ),
        )
    except DatasetError as exc:
        raise CLIError(str(exc)) from exc

    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": save_dataset(
            split.train,
            output_dir / "train.json",
        ),
        "validation": save_dataset(
            split.validation,
            output_dir / "validation.json",
        ),
        "test": save_dataset(
            split.test,
            output_dir / "test.json",
        ),
    }

    manifest = {
        "source": str(_path(args.path)),
        "seed": split.seed,
        "stratified": split.stratified,
        "counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "total": split.total_count,
        },
        "files": {
            key: str(value)
            for key, value in paths.items()
        },
    }
    save_json(
        manifest,
        output_dir / "manifest.json",
    )

    context.write(
        _render_value(
            manifest,
            format_=OutputFormat(args.format),
        )
    )
    return ExitCode.OK


def _command_dataset_filter(
    args: Namespace,
    context: CLIContext,
) -> int:
    try:
        dataset = load_dataset(_path(args.path))
    except DatasetError as exc:
        raise CLIError(str(exc)) from exc

    selected = dataset.records

    if args.group is not None:
        selected = tuple(
            item for item in selected
            if item.group == args.group
        )

    if args.tag is not None:
        selected = tuple(
            item for item in selected
            if args.tag in item.tags
        )

    if args.target is not None:
        target_value: Any
        try:
            target_value = json.loads(args.target)
        except json.JSONDecodeError:
            target_value = args.target
        selected = tuple(
            item for item in selected
            if item.target == target_value
        )

    filtered = Dataset(
        name=args.name or f"{dataset.name}-filtered",
        records=selected,
        kind=dataset.kind,
        version=dataset.version,
        description=dataset.description,
        schema=dataset.schema,
        metadata={
            **dict(dataset.metadata),
            "filtered_from": dataset.name,
        },
    )

    try:
        output = save_dataset(filtered, _path(args.output))
    except DatasetError as exc:
        raise CLIError(str(exc)) from exc

    payload = {
        "output": str(output),
        "record_count": len(filtered),
        "ids": list(filtered.ids),
    }
    context.write(
        _render_value(
            payload,
            format_=OutputFormat(args.format),
        )
    )
    return ExitCode.OK


def _command_convert(args: Namespace, context: CLIContext) -> int:
    source = _path(args.input)
    target = _path(args.output)

    value = _read_serialized(source)

    if hasattr(value, "as_dict"):
        value = value.as_dict()

    output_format = (
        None
        if args.output_format is None
        else SerializationFormat(args.output_format)
    )
    output = _write_serialized(
        value,
        target,
        format_=output_format,
    )

    payload = {
        "input": str(source),
        "output": str(output),
        "input_format": infer_format(source).value,
        "output_format": (
            output_format or infer_format(target)
        ).value,
    }
    context.write(
        _render_value(
            payload,
            format_=OutputFormat(args.format),
        )
    )
    return ExitCode.OK


def _command_report(args: Namespace, context: CLIContext) -> int:
    value = _read_serialized(_path(args.path))
    report = _extract_report(value)

    format_ = OutputFormat(args.format)

    if format_ is OutputFormat.MARKDOWN:
        text = render_markdown(report)
    elif format_ is OutputFormat.JSON:
        text = render_json(report)
    elif format_ is OutputFormat.YAML:
        text = dumps_yaml(report.as_dict())
    elif format_ is OutputFormat.TEXT:
        text = (
            f"{report.title}\n"
            f"Status: {report.status.value}\n"
            f"Summary: {report.summary}\n"
            f"Sections: {len(report.sections)}\n"
            f"Warnings: {len(report.warnings)}"
        )
    else:
        raise CLIError(
            "Report output supports text, json, yaml, or markdown."
        )

    if args.output is None:
        context.write(text)
    else:
        output = _path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        context.write(str(output))

    return ExitCode.OK


def _command_inspect(args: Namespace, context: CLIContext) -> int:
    value = _read_serialized(_path(args.path))

    if isinstance(value, SerializationEnvelope):
        payload = value.as_dict()
    elif hasattr(value, "as_dict"):
        payload = value.as_dict()
    else:
        payload = value

    context.write(
        _render_value(
            payload,
            format_=OutputFormat(args.format),
        )
    )
    return ExitCode.OK


def build_parser() -> ArgumentParser:
    """Create the top-level argument parser."""

    parser = ArgumentParser(
        prog="roif",
        description="ROIF Engine v1 command-line interface.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show diagnostic details.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress normal output.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    version_parser = subparsers.add_parser(
        "version",
        help="Show ROIF Engine version.",
    )
    version_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="text",
    )
    version_parser.set_defaults(handler=_command_version)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the local ROIF environment.",
    )
    doctor_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="text",
    )
    doctor_parser.set_defaults(handler=_command_doctor)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a serialized ROIF file.",
    )
    inspect_parser.add_argument("path")
    inspect_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml", "csv"),
        default="json",
    )
    inspect_parser.set_defaults(handler=_command_inspect)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert between supported serialization formats.",
    )
    convert_parser.add_argument("input")
    convert_parser.add_argument("output")
    convert_parser.add_argument(
        "--output-format",
        choices=("json", "yaml", "csv", "npz"),
    )
    convert_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="text",
        help="Format of the conversion summary.",
    )
    convert_parser.set_defaults(handler=_command_convert)

    report_parser = subparsers.add_parser(
        "report",
        help="Render a serialized ROIF report.",
    )
    report_parser.add_argument("path")
    report_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml", "markdown"),
        default="markdown",
    )
    report_parser.add_argument(
        "--output",
        help="Write rendered output to a file.",
    )
    report_parser.set_defaults(handler=_command_report)

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Dataset operations.",
    )
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_command",
        required=True,
    )

    info_parser = dataset_subparsers.add_parser(
        "info",
        help="Show dataset metadata and statistics.",
    )
    info_parser.add_argument("path")
    info_parser.add_argument(
        "--include-ids",
        action=BooleanOptionalAction,
        default=False,
    )
    info_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="json",
    )
    info_parser.set_defaults(handler=_command_dataset_info)

    validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="Validate a dataset file.",
    )
    validate_parser.add_argument("path")
    validate_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="json",
    )
    validate_parser.set_defaults(
        handler=_command_dataset_validate
    )

    split_parser = dataset_subparsers.add_parser(
        "split",
        help="Create train/validation/test datasets.",
    )
    split_parser.add_argument("path")
    split_parser.add_argument(
        "--output-dir",
        required=True,
    )
    split_parser.add_argument(
        "--train",
        type=float,
        default=0.7,
    )
    split_parser.add_argument(
        "--validation",
        type=float,
        default=0.15,
    )
    split_parser.add_argument(
        "--test",
        type=float,
        default=0.15,
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    split_parser.add_argument(
        "--stratify-target",
        action=BooleanOptionalAction,
        default=False,
    )
    split_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="json",
    )
    split_parser.set_defaults(handler=_command_dataset_split)

    filter_parser = dataset_subparsers.add_parser(
        "filter",
        help="Filter dataset records.",
    )
    filter_parser.add_argument("path")
    filter_parser.add_argument(
        "--output",
        required=True,
    )
    filter_parser.add_argument("--name")
    filter_parser.add_argument("--group")
    filter_parser.add_argument("--tag")
    filter_parser.add_argument(
        "--target",
        help="Target value; JSON syntax is accepted.",
    )
    filter_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="json",
    )
    filter_parser.set_defaults(handler=_command_dataset_filter)

    return parser


def run(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    context = CLIContext(
        stdout=sys.stdout if stdout is None else stdout,
        stderr=sys.stderr if stderr is None else stderr,
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
    )

    handler: Callable[[Namespace, CLIContext], int] = args.handler

    try:
        return int(handler(args, context))
    except (
        CLIError,
        DatasetError,
        SerializationError,
    ) as exc:
        context.error(f"error: {exc}")
        if context.verbose:
            context.error(traceback.format_exc())
        return int(ExitCode.DATA_ERROR)
    except KeyboardInterrupt:
        context.error("error: interrupted")
        return 130
    except Exception as exc:
        context.error(f"internal error: {exc}")
        if context.verbose:
            context.error(traceback.format_exc())
        return int(ExitCode.INTERNAL_ERROR)


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point."""
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "CLIContext",
    "CLIError",
    "ExitCode",
    "OutputFormat",
    "build_parser",
    "main",
    "run",
]
