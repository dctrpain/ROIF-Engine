"""
Reporting layer for ROIF Engine v1.

This module assembles analytical, localization, optimization, uncertainty,
validation, and benchmark outputs into one immutable, serializable report.

The reporting layer does not perform new scientific calculations. It preserves
already computed results, adds provenance, warnings, reproducibility metadata,
and renders them into JSON-compatible dictionaries, Markdown, and flat tables.

Typical use:

    report = ROIFReportBuilder("run-001") \
        .set_summary("Stable recovery after intervention.") \
        .add_result("localization", localization_result) \
        .add_result("optimization", optimization_result) \
        .add_warning(...) \
        .build()

    markdown = render_markdown(report)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable
import json
import math


class ReportingError(ValueError):
    """Raised when report input or configuration is invalid."""


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = (
    JSONScalar
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)


def _normalize_text(
    value: Any,
    *,
    name: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise ReportingError(f"{name} must be a string{suffix}.")
    normalized = value.strip()
    if not normalized:
        raise ReportingError(f"{name} cannot be empty.")
    return normalized


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ReportingError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportingError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise ReportingError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ReportingError(
            f"{name} must be greater than or equal to {minimum}."
        )
    if maximum is not None and result > maximum:
        raise ReportingError(
            f"{name} must be less than or equal to {maximum}."
        )
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportingError(f"{name} must be a non-negative integer.")
    return value


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ReportingError(f"{name} must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        copied[_normalize_text(key, name=f"{name} key")] = item
    return MappingProxyType(copied)


def _timestamp(value: Any, *, name: str) -> str:
    text = _normalize_text(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReportingError(
            f"{name} must be an ISO-8601 datetime."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ReportStatus(str, Enum):
    """Overall report status."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ReportSectionKind(str, Enum):
    """Semantic kind of report section."""

    SUMMARY = "summary"
    STATE = "state"
    METRICS = "metrics"
    LOCALIZATION = "localization"
    OPTIMIZATION = "optimization"
    UNCERTAINTY = "uncertainty"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"
    EXECUTION = "execution"
    CUSTOM = "custom"


class WarningSeverity(str, Enum):
    """Severity of a report warning."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ArtifactKind(str, Enum):
    """Kind of external artifact referenced by a report."""

    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    IMAGE = "image"
    LOG = "log"
    DATASET = "dataset"
    OTHER = "other"


@runtime_checkable
class SerializableResult(Protocol):
    """Protocol for result objects accepted by the report builder."""

    def as_dict(self) -> Mapping[str, Any]:
        ...


def to_json_compatible(value: Any) -> JSONValue:
    """Convert supported Python/result objects to JSON-compatible data."""

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportingError(
                "Non-finite floats cannot be serialized."
            )
        return value

    if isinstance(value, Enum):
        return to_json_compatible(value.value)

    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            normalized = _normalize_text(
                key,
                name="serialized mapping key",
            )
            result[normalized] = to_json_compatible(item)
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, SerializableResult):
        return to_json_compatible(value.as_dict())

    if is_dataclass(value):
        return to_json_compatible(asdict(value))

    if hasattr(value, "tolist") and callable(value.tolist):
        return to_json_compatible(value.tolist())

    raise ReportingError(
        f"Unsupported report value type: {type(value).__name__}."
    )


@dataclass(frozen=True, slots=True)
class ReportWarning:
    """One warning attached to the report."""

    code: str
    message: str
    severity: WarningSeverity = WarningSeverity.WARNING
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code", _normalize_text(self.code, name="code")
        )
        object.__setattr__(
            self,
            "message",
            _normalize_text(self.message, name="message"),
        )
        if not isinstance(self.severity, WarningSeverity):
            raise ReportingError(
                "severity must be a WarningSeverity."
            )
        if self.source is not None:
            object.__setattr__(
                self,
                "source",
                _normalize_text(
                    self.source,
                    name="source",
                    optional=True,
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "metadata": to_json_compatible(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    """Reference to an artifact generated outside the report."""

    name: str
    kind: ArtifactKind
    location: str
    description: str | None = None
    checksum: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _normalize_text(self.name, name="name")
        )
        if not isinstance(self.kind, ArtifactKind):
            raise ReportingError("kind must be an ArtifactKind.")
        object.__setattr__(
            self,
            "location",
            _normalize_text(self.location, name="location"),
        )
        for field_name in ("description", "checksum"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _normalize_text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "location": self.location,
            "description": self.description,
            "checksum": self.checksum,
            "metadata": to_json_compatible(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ReproducibilityInfo:
    """Versioning and seed information for reproducing a run."""

    engine_version: str
    schema_version: str = "1.0"
    commit: str | None = None
    seed: int | None = None
    command: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "engine_version",
            _normalize_text(
                self.engine_version,
                name="engine_version",
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            _normalize_text(
                self.schema_version,
                name="schema_version",
            ),
        )
        for field_name in ("commit", "command"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _normalize_text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ReportingError("seed must be an integer or None.")

        object.__setattr__(
            self,
            "environment",
            _freeze_mapping(
                self.environment,
                name="environment",
            ),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "commit": self.commit,
            "seed": self.seed,
            "command": self.command,
            "environment": to_json_compatible(self.environment),
        }


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One named section of a ROIF report."""

    section_id: str
    title: str
    kind: ReportSectionKind
    content: JSONValue
    order: int = 0
    summary: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "section_id",
            _normalize_text(
                self.section_id,
                name="section_id",
            ),
        )
        object.__setattr__(
            self,
            "title",
            _normalize_text(self.title, name="title"),
        )
        if not isinstance(self.kind, ReportSectionKind):
            raise ReportingError(
                "kind must be a ReportSectionKind."
            )
        object.__setattr__(
            self,
            "content",
            to_json_compatible(self.content),
        )
        object.__setattr__(
            self,
            "order",
            _nonnegative_int(self.order, name="order"),
        )
        if self.summary is not None:
            object.__setattr__(
                self,
                "summary",
                _normalize_text(
                    self.summary,
                    name="summary",
                    optional=True,
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "kind": self.kind.value,
            "content": self.content,
            "order": self.order,
            "summary": self.summary,
            "metadata": to_json_compatible(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ROIFReport:
    """Complete immutable ROIF report."""

    report_id: str
    title: str
    status: ReportStatus
    created_at: str
    summary: str
    sections: tuple[ReportSection, ...]
    warnings: tuple[ReportWarning, ...] = ()
    artifacts: tuple[ReportArtifact, ...] = ()
    reproducibility: ReproducibilityInfo | None = None
    duration_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_id",
            _normalize_text(self.report_id, name="report_id"),
        )
        object.__setattr__(
            self,
            "title",
            _normalize_text(self.title, name="title"),
        )
        if not isinstance(self.status, ReportStatus):
            raise ReportingError(
                "status must be a ReportStatus."
            )
        object.__setattr__(
            self,
            "created_at",
            _timestamp(self.created_at, name="created_at"),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_text(self.summary, name="summary"),
        )

        sections = tuple(self.sections)
        if any(not isinstance(item, ReportSection) for item in sections):
            raise ReportingError(
                "sections must contain ReportSection values."
            )
        ids = [item.section_id for item in sections]
        if len(ids) != len(set(ids)):
            raise ReportingError(
                "section_id values must be unique."
            )
        object.__setattr__(
            self,
            "sections",
            tuple(sorted(sections, key=lambda x: (x.order, x.section_id))),
        )

        warnings = tuple(self.warnings)
        if any(not isinstance(item, ReportWarning) for item in warnings):
            raise ReportingError(
                "warnings must contain ReportWarning values."
            )
        object.__setattr__(self, "warnings", warnings)

        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ReportArtifact) for item in artifacts):
            raise ReportingError(
                "artifacts must contain ReportArtifact values."
            )
        object.__setattr__(self, "artifacts", artifacts)

        if self.reproducibility is not None and not isinstance(
            self.reproducibility,
            ReproducibilityInfo,
        ):
            raise ReportingError(
                "reproducibility must be ReproducibilityInfo or None."
            )

        if self.duration_seconds is not None:
            object.__setattr__(
                self,
                "duration_seconds",
                _finite_float(
                    self.duration_seconds,
                    name="duration_seconds",
                    minimum=0.0,
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    @property
    def critical_warning_count(self) -> int:
        return sum(
            item.severity is WarningSeverity.CRITICAL
            for item in self.warnings
        )

    def section(self, section_id: str) -> ReportSection:
        """Return one section by id."""
        normalized = _normalize_text(section_id, name="section_id")
        for item in self.sections:
            if item.section_id == normalized:
                return item
        raise KeyError(normalized)

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at,
            "summary": self.summary,
            "sections": [item.as_dict() for item in self.sections],
            "warnings": [item.as_dict() for item in self.warnings],
            "artifacts": [item.as_dict() for item in self.artifacts],
            "reproducibility": (
                None
                if self.reproducibility is None
                else self.reproducibility.as_dict()
            ),
            "duration_seconds": self.duration_seconds,
            "critical_warning_count": self.critical_warning_count,
            "metadata": to_json_compatible(self.metadata),
        }


class ROIFReportBuilder:
    """Mutable builder that produces an immutable ``ROIFReport``."""

    def __init__(
        self,
        report_id: str,
        *,
        title: str = "ROIF Engine Report",
        created_at: str | None = None,
    ) -> None:
        self._report_id = _normalize_text(
            report_id,
            name="report_id",
        )
        self._title = _normalize_text(title, name="title")
        self._created_at = (
            utc_now_iso()
            if created_at is None
            else _timestamp(created_at, name="created_at")
        )
        self._status = ReportStatus.COMPLETED
        self._summary: str | None = None
        self._sections: list[ReportSection] = []
        self._warnings: list[ReportWarning] = []
        self._artifacts: list[ReportArtifact] = []
        self._reproducibility: ReproducibilityInfo | None = None
        self._duration_seconds: float | None = None
        self._metadata: dict[str, Any] = {}

    def set_status(self, status: ReportStatus) -> "ROIFReportBuilder":
        if not isinstance(status, ReportStatus):
            raise ReportingError(
                "status must be a ReportStatus."
            )
        self._status = status
        return self

    def set_summary(self, summary: str) -> "ROIFReportBuilder":
        self._summary = _normalize_text(summary, name="summary")
        return self

    def set_duration(
        self,
        duration_seconds: float | None,
    ) -> "ROIFReportBuilder":
        self._duration_seconds = (
            None
            if duration_seconds is None
            else _finite_float(
                duration_seconds,
                name="duration_seconds",
                minimum=0.0,
            )
        )
        return self

    def set_reproducibility(
        self,
        value: ReproducibilityInfo | None,
    ) -> "ROIFReportBuilder":
        if value is not None and not isinstance(
            value,
            ReproducibilityInfo,
        ):
            raise ReportingError(
                "value must be ReproducibilityInfo or None."
            )
        self._reproducibility = value
        return self

    def update_metadata(
        self,
        metadata: Mapping[str, Any],
    ) -> "ROIFReportBuilder":
        frozen = _freeze_mapping(metadata, name="metadata")
        self._metadata.update(dict(frozen))
        return self

    def add_section(
        self,
        section_id: str,
        title: str,
        kind: ReportSectionKind,
        content: Any,
        *,
        order: int | None = None,
        summary: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ROIFReportBuilder":
        normalized_id = _normalize_text(
            section_id,
            name="section_id",
        )
        if any(item.section_id == normalized_id for item in self._sections):
            raise ReportingError(
                f"Duplicate section_id: {normalized_id!r}."
            )
        section_order = (
            len(self._sections)
            if order is None
            else _nonnegative_int(order, name="order")
        )
        self._sections.append(
            ReportSection(
                section_id=normalized_id,
                title=title,
                kind=kind,
                content=to_json_compatible(content),
                order=section_order,
                summary=summary,
                metadata={} if metadata is None else metadata,
            )
        )
        return self

    def add_result(
        self,
        section_id: str,
        result: Any,
        *,
        title: str | None = None,
        kind: ReportSectionKind = ReportSectionKind.CUSTOM,
        order: int | None = None,
        summary: str | None = None,
    ) -> "ROIFReportBuilder":
        return self.add_section(
            section_id,
            title or section_id.replace("_", " ").title(),
            kind,
            result,
            order=order,
            summary=summary,
        )

    def add_warning(
        self,
        warning: ReportWarning,
    ) -> "ROIFReportBuilder":
        if not isinstance(warning, ReportWarning):
            raise ReportingError(
                "warning must be a ReportWarning."
            )
        self._warnings.append(warning)
        return self

    def add_artifact(
        self,
        artifact: ReportArtifact,
    ) -> "ROIFReportBuilder":
        if not isinstance(artifact, ReportArtifact):
            raise ReportingError(
                "artifact must be a ReportArtifact."
            )
        self._artifacts.append(artifact)
        return self

    def build(self) -> ROIFReport:
        """Build the immutable report."""
        if self._summary is None:
            raise ReportingError(
                "Report summary must be set before build()."
            )
        return ROIFReport(
            report_id=self._report_id,
            title=self._title,
            status=self._status,
            created_at=self._created_at,
            summary=self._summary,
            sections=tuple(self._sections),
            warnings=tuple(self._warnings),
            artifacts=tuple(self._artifacts),
            reproducibility=self._reproducibility,
            duration_seconds=self._duration_seconds,
            metadata=self._metadata,
        )


def render_json(
    report: ROIFReport,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> str:
    """Render a report as JSON."""
    if not isinstance(report, ROIFReport):
        raise ReportingError("report must be a ROIFReport.")
    if indent is not None:
        indent = _nonnegative_int(indent, name="indent")
    if not isinstance(sort_keys, bool):
        raise ReportingError("sort_keys must be a bool.")
    return json.dumps(
        report.as_dict(),
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
    )


def _markdown_value(value: JSONValue, *, level: int = 0) -> list[str]:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{'  ' * level}- **{key}:**")
                lines.extend(
                    _markdown_value(item, level=level + 1)
                )
            else:
                rendered = "null" if item is None else str(item)
                lines.append(
                    f"{'  ' * level}- **{key}:** {rendered}"
                )
        return lines

    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{'  ' * level}-")
                lines.extend(
                    _markdown_value(item, level=level + 1)
                )
            else:
                rendered = "null" if item is None else str(item)
                lines.append(f"{'  ' * level}- {rendered}")
        return lines

    return [f"{'  ' * level}{value}"]


def render_markdown(report: ROIFReport) -> str:
    """Render a human-readable Markdown report."""
    if not isinstance(report, ROIFReport):
        raise ReportingError("report must be a ROIFReport.")

    lines = [
        f"# {report.title}",
        "",
        f"**Report ID:** `{report.report_id}`  ",
        f"**Status:** `{report.status.value}`  ",
        f"**Created:** `{report.created_at}`",
        "",
        "## Summary",
        "",
        report.summary,
    ]

    if report.duration_seconds is not None:
        lines.extend([
            "",
            f"**Duration:** {report.duration_seconds:.6g} s",
        ])

    for section in report.sections:
        lines.extend([
            "",
            f"## {section.title}",
            "",
        ])
        if section.summary is not None:
            lines.extend([section.summary, ""])
        lines.extend(_markdown_value(section.content))

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in report.warnings:
            source = (
                ""
                if warning.source is None
                else f" ({warning.source})"
            )
            lines.append(
                f"- **{warning.severity.value.upper()} "
                f"[{warning.code}]**{source}: {warning.message}"
            )

    if report.artifacts:
        lines.extend(["", "## Artifacts", ""])
        for artifact in report.artifacts:
            lines.append(
                f"- **{artifact.name}** "
                f"(`{artifact.kind.value}`): `{artifact.location}`"
            )

    if report.reproducibility is not None:
        lines.extend(["", "## Reproducibility", ""])
        lines.extend(
            _markdown_value(
                report.reproducibility.as_dict()
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def flatten_report(
    report: ROIFReport,
) -> tuple[Mapping[str, JSONScalar], ...]:
    """
    Flatten report sections into tabular key/value rows.

    Nested paths use dot notation; list indices use numeric components.
    """

    if not isinstance(report, ROIFReport):
        raise ReportingError("report must be a ROIFReport.")

    rows: list[Mapping[str, JSONScalar]] = []

    def visit(
        value: JSONValue,
        *,
        section_id: str,
        path: str,
    ) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = key if not path else f"{path}.{key}"
                visit(item, section_id=section_id, path=child)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                child = str(index) if not path else f"{path}.{index}"
                visit(item, section_id=section_id, path=child)
            return
        rows.append(
            MappingProxyType({
                "report_id": report.report_id,
                "section_id": section_id,
                "path": path,
                "value": value,
            })
        )

    for section in report.sections:
        visit(
            section.content,
            section_id=section.section_id,
            path="",
        )

    return tuple(rows)


__all__ = [
    "ArtifactKind",
    "JSONScalar",
    "JSONValue",
    "ROIFReport",
    "ROIFReportBuilder",
    "ReportArtifact",
    "ReportSection",
    "ReportSectionKind",
    "ReportStatus",
    "ReportWarning",
    "ReportingError",
    "ReproducibilityInfo",
    "SerializableResult",
    "WarningSeverity",
    "flatten_report",
    "render_json",
    "render_markdown",
    "to_json_compatible",
    "utc_now_iso",
]
