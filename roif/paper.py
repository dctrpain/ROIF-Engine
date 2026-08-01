"""
Scientific paper assembly utilities for ROIF Engine v1.

The module converts already computed ROIF results into a reproducible
publication-oriented document model. It does not perform scientific analysis
itself. Instead, it organizes supplied datasets, validation outputs, benchmark
outputs, engine results, tables, figures, references, and disclosure statements
into a stable paper structure.

Supported outputs:

- immutable ``PaperDocument`` objects;
- JSON-compatible dictionaries;
- Markdown;
- LaTeX;
- flat table/figure manifests;
- reproducibility metadata;
- structured references and citations;
- appendices and supplementary material.

Typical usage:

    paper = (
        PaperBuilder(
            title="ROIF: Cascading Dynamics in Pre-Stressed Systems",
            paper_id="roif-2026-01",
        )
        .add_author(...)
        .set_abstract(...)
        .add_section(...)
        .add_result_section("validation", validation_result)
        .add_table(...)
        .add_figure(...)
        .add_reference(...)
        .build()
    )

    markdown = render_markdown(paper)
    latex = render_latex(paper)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias
import json
import math
import re

from .serialization import to_serializable


class PaperError(ValueError):
    """Raised when paper content or metadata is invalid."""


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = (
    JSONScalar
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)


def _text(
    value: Any,
    *,
    name: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise PaperError(f"{name} must be a string{suffix}.")
    normalized = value.strip()
    if not normalized:
        raise PaperError(f"{name} cannot be empty.")
    return normalized


def _identifier(value: Any, *, name: str) -> str:
    normalized = _text(value, name=name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", normalized):
        raise PaperError(
            f"{name} contains unsupported characters."
        )
    return normalized


def _nonnegative_int(value: Any, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise PaperError(
            f"{name} must be a non-negative integer."
        )
    return value


def _finite(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise PaperError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperError(
            f"{name} must be a real number."
        ) from exc
    if not math.isfinite(result):
        raise PaperError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise PaperError(
            f"{name} must be greater than or equal to {minimum}."
        )
    return result


def _freeze(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise PaperError(f"{name} must be a mapping.")
    parsed: dict[str, Any] = {}
    for key, item in value.items():
        parsed[_text(key, name=f"{name} key")] = item
    return MappingProxyType(parsed)


def _timestamp(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise PaperError(
            f"{name} must be an ISO-8601 datetime."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        parsed.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _json_value(value: Any) -> JSONValue:
    try:
        return to_serializable(value)
    except Exception as exc:
        raise PaperError(
            f"Value is not serializable: {type(value).__name__}."
        ) from exc


def _unique_texts(
    values: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PaperError(f"{name} must be a sequence of strings.")
    parsed = tuple(
        _text(item, name=f"{name} item")
        for item in values
    )
    if len(parsed) != len(set(parsed)):
        raise PaperError(f"{name} cannot contain duplicates.")
    return parsed


class PaperStatus(str, Enum):
    """Publication lifecycle state."""

    DRAFT = "draft"
    PREPRINT = "preprint"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PUBLISHED = "published"


class SectionKind(str, Enum):
    """Canonical scientific paper sections."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    DATASET = "dataset"
    RESULTS = "results"
    DISCUSSION = "discussion"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"
    ACKNOWLEDGEMENTS = "acknowledgements"
    FUNDING = "funding"
    CONFLICTS = "conflicts"
    DATA_AVAILABILITY = "data_availability"
    CODE_AVAILABILITY = "code_availability"
    ETHICS = "ethics"
    REFERENCES = "references"
    APPENDIX = "appendix"
    SUPPLEMENT = "supplement"
    CUSTOM = "custom"


class TableAlignment(str, Enum):
    """Table cell alignment."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class FigureKind(str, Enum):
    """Publication figure category."""

    PLOT = "plot"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    IMAGE = "image"
    HEATMAP = "heatmap"
    NETWORK = "network"
    OTHER = "other"


class ReferenceType(str, Enum):
    """Bibliographic reference type."""

    ARTICLE = "article"
    BOOK = "book"
    CHAPTER = "chapter"
    CONFERENCE = "conference"
    PREPRINT = "preprint"
    DATASET = "dataset"
    SOFTWARE = "software"
    WEB = "web"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Author:
    """One paper author."""

    author_id: str
    given_name: str
    family_name: str
    affiliations: tuple[str, ...] = ()
    email: str | None = None
    orcid: str | None = None
    corresponding: bool = False
    contribution: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "author_id",
            _identifier(self.author_id, name="author_id"),
        )
        object.__setattr__(
            self,
            "given_name",
            _text(self.given_name, name="given_name"),
        )
        object.__setattr__(
            self,
            "family_name",
            _text(self.family_name, name="family_name"),
        )
        object.__setattr__(
            self,
            "affiliations",
            _unique_texts(
                self.affiliations,
                name="affiliations",
            ),
        )
        for field_name in ("email", "orcid", "contribution"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )
        if not isinstance(self.corresponding, bool):
            raise PaperError("corresponding must be a bool.")

    @property
    def display_name(self) -> str:
        return f"{self.given_name} {self.family_name}"

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "author_id": self.author_id,
            "given_name": self.given_name,
            "family_name": self.family_name,
            "display_name": self.display_name,
            "affiliations": list(self.affiliations),
            "email": self.email,
            "orcid": self.orcid,
            "corresponding": self.corresponding,
            "contribution": self.contribution,
        }


@dataclass(frozen=True, slots=True)
class Affiliation:
    """One institutional affiliation."""

    affiliation_id: str
    institution: str
    department: str | None = None
    city: str | None = None
    country: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "affiliation_id",
            _identifier(
                self.affiliation_id,
                name="affiliation_id",
            ),
        )
        object.__setattr__(
            self,
            "institution",
            _text(self.institution, name="institution"),
        )
        for field_name in ("department", "city", "country"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "affiliation_id": self.affiliation_id,
            "institution": self.institution,
            "department": self.department,
            "city": self.city,
            "country": self.country,
        }


@dataclass(frozen=True, slots=True)
class Citation:
    """Inline citation reference."""

    reference_ids: tuple[str, ...]
    locator: str | None = None
    prefix: str | None = None
    suffix: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.reference_ids, (str, bytes)):
            raise PaperError(
                "reference_ids must be a sequence."
            )
        parsed = tuple(
            _identifier(item, name="reference_id")
            for item in self.reference_ids
        )
        if not parsed:
            raise PaperError(
                "reference_ids cannot be empty."
            )
        if len(parsed) != len(set(parsed)):
            raise PaperError(
                "reference_ids cannot contain duplicates."
            )
        object.__setattr__(self, "reference_ids", parsed)
        for field_name in ("locator", "prefix", "suffix"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "reference_ids": list(self.reference_ids),
            "locator": self.locator,
            "prefix": self.prefix,
            "suffix": self.suffix,
        }


@dataclass(frozen=True, slots=True)
class Paragraph:
    """Paragraph with optional citations."""

    text: str
    citations: tuple[Citation, ...] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _text(self.text, name="text"),
        )
        citations = tuple(self.citations)
        if any(not isinstance(item, Citation) for item in citations):
            raise PaperError(
                "citations must contain Citation values."
            )
        object.__setattr__(self, "citations", citations)
        if self.label is not None:
            object.__setattr__(
                self,
                "label",
                _identifier(self.label, name="label"),
            )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "text": self.text,
            "citations": [
                item.as_dict() for item in self.citations
            ],
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class PaperSection:
    """One ordered scientific paper section."""

    section_id: str
    title: str
    kind: SectionKind
    paragraphs: tuple[Paragraph, ...] = ()
    data: JSONValue | None = None
    order: int = 0
    level: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "section_id",
            _identifier(self.section_id, name="section_id"),
        )
        object.__setattr__(
            self,
            "title",
            _text(self.title, name="title"),
        )
        if not isinstance(self.kind, SectionKind):
            raise PaperError("kind must be a SectionKind.")

        paragraphs = tuple(self.paragraphs)
        if any(not isinstance(item, Paragraph) for item in paragraphs):
            raise PaperError(
                "paragraphs must contain Paragraph values."
            )
        object.__setattr__(self, "paragraphs", paragraphs)
        object.__setattr__(
            self,
            "data",
            None if self.data is None else _json_value(self.data),
        )
        object.__setattr__(
            self,
            "order",
            _nonnegative_int(self.order, name="order"),
        )
        level = _nonnegative_int(self.level, name="level")
        if level < 1 or level > 6:
            raise PaperError("level must be between 1 and 6.")
        object.__setattr__(self, "level", level)
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, name="metadata"),
        )

        if not self.paragraphs and self.data is None:
            raise PaperError(
                "Section must contain paragraphs or data."
            )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "kind": self.kind.value,
            "paragraphs": [
                item.as_dict() for item in self.paragraphs
            ],
            "data": self.data,
            "order": self.order,
            "level": self.level,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PaperTable:
    """Publication-ready table."""

    table_id: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[JSONScalar, ...], ...]
    caption: str | None = None
    notes: tuple[str, ...] = ()
    alignments: tuple[TableAlignment, ...] = ()
    source: str | None = None
    order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "table_id",
            _identifier(self.table_id, name="table_id"),
        )
        object.__setattr__(
            self,
            "title",
            _text(self.title, name="title"),
        )
        columns = _unique_texts(
            self.columns,
            name="columns",
        )
        if not columns:
            raise PaperError("columns cannot be empty.")
        object.__setattr__(self, "columns", columns)

        parsed_rows: list[tuple[JSONScalar, ...]] = []
        for row in self.rows:
            if isinstance(row, (str, bytes)):
                raise PaperError(
                    "table rows must be sequences."
                )
            parsed = tuple(row)
            if len(parsed) != len(columns):
                raise PaperError(
                    "Every table row must match column count."
                )
            scalar_row: list[JSONScalar] = []
            for value in parsed:
                serialized = _json_value(value)
                if isinstance(serialized, (dict, list)):
                    raise PaperError(
                        "Table cells must be scalar values."
                    )
                scalar_row.append(serialized)
            parsed_rows.append(tuple(scalar_row))
        object.__setattr__(
            self,
            "rows",
            tuple(parsed_rows),
        )

        for field_name in ("caption", "source"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )

        object.__setattr__(
            self,
            "notes",
            _unique_texts(self.notes, name="notes"),
        )

        alignments = tuple(self.alignments)
        if not alignments:
            alignments = tuple(
                TableAlignment.LEFT for _ in columns
            )
        if len(alignments) != len(columns):
            raise PaperError(
                "alignments must match column count."
            )
        if any(
            not isinstance(item, TableAlignment)
            for item in alignments
        ):
            raise PaperError(
                "alignments must contain TableAlignment values."
            )
        object.__setattr__(
            self,
            "alignments",
            alignments,
        )
        object.__setattr__(
            self,
            "order",
            _nonnegative_int(self.order, name="order"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, name="metadata"),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "table_id": self.table_id,
            "title": self.title,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "caption": self.caption,
            "notes": list(self.notes),
            "alignments": [
                item.value for item in self.alignments
            ],
            "source": self.source,
            "order": self.order,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PaperFigure:
    """Figure metadata and external asset reference."""

    figure_id: str
    title: str
    kind: FigureKind
    location: str
    caption: str
    alt_text: str | None = None
    width_fraction: float = 1.0
    source: str | None = None
    order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "figure_id",
            _identifier(self.figure_id, name="figure_id"),
        )
        object.__setattr__(
            self,
            "title",
            _text(self.title, name="title"),
        )
        if not isinstance(self.kind, FigureKind):
            raise PaperError("kind must be a FigureKind.")
        object.__setattr__(
            self,
            "location",
            _text(self.location, name="location"),
        )
        object.__setattr__(
            self,
            "caption",
            _text(self.caption, name="caption"),
        )
        for field_name in ("alt_text", "source"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )
        width = _finite(
            self.width_fraction,
            name="width_fraction",
            minimum=0.0,
        )
        if width <= 0.0 or width > 1.0:
            raise PaperError(
                "width_fraction must be greater than 0 and at most 1."
            )
        object.__setattr__(
            self,
            "width_fraction",
            width,
        )
        object.__setattr__(
            self,
            "order",
            _nonnegative_int(self.order, name="order"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, name="metadata"),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "figure_id": self.figure_id,
            "title": self.title,
            "kind": self.kind.value,
            "location": self.location,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "width_fraction": self.width_fraction,
            "source": self.source,
            "order": self.order,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Reference:
    """Structured bibliographic reference."""

    reference_id: str
    reference_type: ReferenceType
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    container_title: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    url: str | None = None
    publisher: str | None = None
    accessed_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            _identifier(
                self.reference_id,
                name="reference_id",
            ),
        )
        if not isinstance(self.reference_type, ReferenceType):
            raise PaperError(
                "reference_type must be a ReferenceType."
            )
        object.__setattr__(
            self,
            "title",
            _text(self.title, name="title"),
        )
        object.__setattr__(
            self,
            "authors",
            _unique_texts(self.authors, name="authors"),
        )
        if self.year is not None:
            if (
                isinstance(self.year, bool)
                or not isinstance(self.year, int)
                or self.year < 1000
                or self.year > 9999
            ):
                raise PaperError(
                    "year must be a four-digit integer or None."
                )

        for field_name in (
            "container_title",
            "volume",
            "issue",
            "pages",
            "doi",
            "url",
            "publisher",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )

        if self.accessed_at is not None:
            object.__setattr__(
                self,
                "accessed_at",
                _timestamp(
                    self.accessed_at,
                    name="accessed_at",
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, name="metadata"),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "reference_id": self.reference_id,
            "reference_type": self.reference_type.value,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "container_title": self.container_title,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "url": self.url,
            "publisher": self.publisher,
            "accessed_at": self.accessed_at,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    """Machine-readable reproducibility information."""

    engine_version: str
    schema_version: str = "1.0"
    commit: str | None = None
    seed: int | None = None
    command: str | None = None
    dataset_ids: tuple[str, ...] = ()
    artifact_checksums: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "engine_version",
            _text(
                self.engine_version,
                name="engine_version",
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            _text(
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
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise PaperError(
                "seed must be an integer or None."
            )
        object.__setattr__(
            self,
            "dataset_ids",
            _unique_texts(
                self.dataset_ids,
                name="dataset_ids",
            ),
        )

        checksums = _freeze(
            self.artifact_checksums,
            name="artifact_checksums",
        )
        for key, value in checksums.items():
            _text(value, name=f"checksum for {key}")
        object.__setattr__(
            self,
            "artifact_checksums",
            checksums,
        )
        object.__setattr__(
            self,
            "environment",
            _freeze(self.environment, name="environment"),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "commit": self.commit,
            "seed": self.seed,
            "command": self.command,
            "dataset_ids": list(self.dataset_ids),
            "artifact_checksums": dict(
                self.artifact_checksums
            ),
            "environment": _json_value(self.environment),
        }


@dataclass(frozen=True, slots=True)
class PaperDocument:
    """Complete immutable scientific paper document."""

    paper_id: str
    title: str
    status: PaperStatus
    created_at: str
    authors: tuple[Author, ...]
    affiliations: tuple[Affiliation, ...]
    keywords: tuple[str, ...]
    abstract: str
    sections: tuple[PaperSection, ...]
    tables: tuple[PaperTable, ...] = ()
    figures: tuple[PaperFigure, ...] = ()
    references: tuple[Reference, ...] = ()
    reproducibility: ReproducibilityManifest | None = None
    journal: str | None = None
    doi: str | None = None
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "paper_id",
            _identifier(self.paper_id, name="paper_id"),
        )
        object.__setattr__(
            self,
            "title",
            _text(self.title, name="title"),
        )
        if not isinstance(self.status, PaperStatus):
            raise PaperError(
                "status must be a PaperStatus."
            )
        object.__setattr__(
            self,
            "created_at",
            _timestamp(self.created_at, name="created_at"),
        )

        authors = tuple(self.authors)
        if any(not isinstance(item, Author) for item in authors):
            raise PaperError(
                "authors must contain Author values."
            )
        author_ids = [item.author_id for item in authors]
        if len(author_ids) != len(set(author_ids)):
            raise PaperError(
                "author_id values must be unique."
            )
        object.__setattr__(self, "authors", authors)

        affiliations = tuple(self.affiliations)
        if any(
            not isinstance(item, Affiliation)
            for item in affiliations
        ):
            raise PaperError(
                "affiliations must contain Affiliation values."
            )
        affiliation_ids = [
            item.affiliation_id for item in affiliations
        ]
        if len(affiliation_ids) != len(set(affiliation_ids)):
            raise PaperError(
                "affiliation_id values must be unique."
            )
        object.__setattr__(
            self,
            "affiliations",
            affiliations,
        )

        affiliation_set = set(affiliation_ids)
        for author in authors:
            unknown = set(author.affiliations) - affiliation_set
            if unknown:
                raise PaperError(
                    f"Author {author.author_id} references "
                    "unknown affiliations."
                )

        object.__setattr__(
            self,
            "keywords",
            _unique_texts(self.keywords, name="keywords"),
        )
        object.__setattr__(
            self,
            "abstract",
            _text(self.abstract, name="abstract"),
        )

        sections = tuple(self.sections)
        if any(
            not isinstance(item, PaperSection)
            for item in sections
        ):
            raise PaperError(
                "sections must contain PaperSection values."
            )
        section_ids = [item.section_id for item in sections]
        if len(section_ids) != len(set(section_ids)):
            raise PaperError(
                "section_id values must be unique."
            )
        object.__setattr__(
            self,
            "sections",
            tuple(
                sorted(
                    sections,
                    key=lambda item: (
                        item.order,
                        item.section_id,
                    ),
                )
            ),
        )

        tables = tuple(self.tables)
        if any(not isinstance(item, PaperTable) for item in tables):
            raise PaperError(
                "tables must contain PaperTable values."
            )
        table_ids = [item.table_id for item in tables]
        if len(table_ids) != len(set(table_ids)):
            raise PaperError(
                "table_id values must be unique."
            )
        object.__setattr__(
            self,
            "tables",
            tuple(
                sorted(
                    tables,
                    key=lambda item: (
                        item.order,
                        item.table_id,
                    ),
                )
            ),
        )

        figures = tuple(self.figures)
        if any(
            not isinstance(item, PaperFigure)
            for item in figures
        ):
            raise PaperError(
                "figures must contain PaperFigure values."
            )
        figure_ids = [item.figure_id for item in figures]
        if len(figure_ids) != len(set(figure_ids)):
            raise PaperError(
                "figure_id values must be unique."
            )
        object.__setattr__(
            self,
            "figures",
            tuple(
                sorted(
                    figures,
                    key=lambda item: (
                        item.order,
                        item.figure_id,
                    ),
                )
            ),
        )

        references = tuple(self.references)
        if any(
            not isinstance(item, Reference)
            for item in references
        ):
            raise PaperError(
                "references must contain Reference values."
            )
        reference_ids = [
            item.reference_id for item in references
        ]
        if len(reference_ids) != len(set(reference_ids)):
            raise PaperError(
                "reference_id values must be unique."
            )
        object.__setattr__(
            self,
            "references",
            references,
        )

        known_references = set(reference_ids)
        for section in sections:
            for paragraph in section.paragraphs:
                for citation in paragraph.citations:
                    unknown = (
                        set(citation.reference_ids)
                        - known_references
                    )
                    if unknown:
                        raise PaperError(
                            f"Section {section.section_id} "
                            "contains unknown citations."
                        )

        if self.reproducibility is not None and not isinstance(
            self.reproducibility,
            ReproducibilityManifest,
        ):
            raise PaperError(
                "reproducibility must be "
                "ReproducibilityManifest or None."
            )

        for field_name in ("journal", "doi"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _text(
                        value,
                        name=field_name,
                        optional=True,
                    ),
                )
        object.__setattr__(
            self,
            "language",
            _text(self.language, name="language"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, name="metadata"),
        )

    @property
    def word_count(self) -> int:
        texts = [self.abstract]
        for section in self.sections:
            texts.extend(
                paragraph.text
                for paragraph in section.paragraphs
            )
        return sum(
            len(re.findall(r"\b[\w'-]+\b", text))
            for text in texts
        )

    def section(self, section_id: str) -> PaperSection:
        normalized = _identifier(
            section_id,
            name="section_id",
        )
        for section in self.sections:
            if section.section_id == normalized:
                return section
        raise KeyError(normalized)

    def reference(self, reference_id: str) -> Reference:
        normalized = _identifier(
            reference_id,
            name="reference_id",
        )
        for reference in self.references:
            if reference.reference_id == normalized:
                return reference
        raise KeyError(normalized)

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at,
            "authors": [
                item.as_dict() for item in self.authors
            ],
            "affiliations": [
                item.as_dict() for item in self.affiliations
            ],
            "keywords": list(self.keywords),
            "abstract": self.abstract,
            "sections": [
                item.as_dict() for item in self.sections
            ],
            "tables": [
                item.as_dict() for item in self.tables
            ],
            "figures": [
                item.as_dict() for item in self.figures
            ],
            "references": [
                item.as_dict() for item in self.references
            ],
            "reproducibility": (
                None
                if self.reproducibility is None
                else self.reproducibility.as_dict()
            ),
            "journal": self.journal,
            "doi": self.doi,
            "language": self.language,
            "word_count": self.word_count,
            "metadata": _json_value(self.metadata),
        }

    def to_json(
        self,
        *,
        indent: int | None = 2,
        sort_keys: bool = False,
    ) -> str:
        if indent is not None and (
            isinstance(indent, bool)
            or not isinstance(indent, int)
            or indent < 0
        ):
            raise PaperError(
                "indent must be a non-negative integer or None."
            )
        if not isinstance(sort_keys, bool):
            raise PaperError(
                "sort_keys must be a bool."
            )
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
        )

    def to_markdown(self) -> str:
        return render_markdown(self)

    def to_latex(self) -> str:
        return render_latex(self)


class PaperBuilder:
    """Mutable builder for an immutable ``PaperDocument``."""

    def __init__(
        self,
        title: str,
        paper_id: str,
        *,
        created_at: str | None = None,
        status: PaperStatus = PaperStatus.DRAFT,
        language: str = "en",
    ) -> None:
        self._title = _text(title, name="title")
        self._paper_id = _identifier(
            paper_id,
            name="paper_id",
        )
        self._created_at = (
            utc_now_iso()
            if created_at is None
            else _timestamp(created_at, name="created_at")
        )
        if not isinstance(status, PaperStatus):
            raise PaperError(
                "status must be a PaperStatus."
            )
        self._status = status
        self._language = _text(language, name="language")
        self._authors: list[Author] = []
        self._affiliations: list[Affiliation] = []
        self._keywords: list[str] = []
        self._abstract: str | None = None
        self._sections: list[PaperSection] = []
        self._tables: list[PaperTable] = []
        self._figures: list[PaperFigure] = []
        self._references: list[Reference] = []
        self._reproducibility: ReproducibilityManifest | None = None
        self._journal: str | None = None
        self._doi: str | None = None
        self._metadata: dict[str, Any] = {}

    def set_status(
        self,
        status: PaperStatus,
    ) -> "PaperBuilder":
        if not isinstance(status, PaperStatus):
            raise PaperError(
                "status must be a PaperStatus."
            )
        self._status = status
        return self

    def set_abstract(
        self,
        abstract: str,
    ) -> "PaperBuilder":
        self._abstract = _text(
            abstract,
            name="abstract",
        )
        return self

    def set_keywords(
        self,
        keywords: Sequence[str],
    ) -> "PaperBuilder":
        self._keywords = list(
            _unique_texts(keywords, name="keywords")
        )
        return self

    def set_journal(
        self,
        journal: str | None,
    ) -> "PaperBuilder":
        self._journal = (
            None
            if journal is None
            else _text(journal, name="journal")
        )
        return self

    def set_doi(
        self,
        doi: str | None,
    ) -> "PaperBuilder":
        self._doi = (
            None
            if doi is None
            else _text(doi, name="doi")
        )
        return self

    def set_reproducibility(
        self,
        manifest: ReproducibilityManifest | None,
    ) -> "PaperBuilder":
        if manifest is not None and not isinstance(
            manifest,
            ReproducibilityManifest,
        ):
            raise PaperError(
                "manifest must be ReproducibilityManifest or None."
            )
        self._reproducibility = manifest
        return self

    def update_metadata(
        self,
        metadata: Mapping[str, Any],
    ) -> "PaperBuilder":
        parsed = _freeze(metadata, name="metadata")
        self._metadata.update(dict(parsed))
        return self

    def add_author(
        self,
        author: Author,
    ) -> "PaperBuilder":
        if not isinstance(author, Author):
            raise PaperError("author must be an Author.")
        if any(
            item.author_id == author.author_id
            for item in self._authors
        ):
            raise PaperError(
                f"Duplicate author_id: {author.author_id}."
            )
        self._authors.append(author)
        return self

    def add_affiliation(
        self,
        affiliation: Affiliation,
    ) -> "PaperBuilder":
        if not isinstance(affiliation, Affiliation):
            raise PaperError(
                "affiliation must be an Affiliation."
            )
        if any(
            item.affiliation_id == affiliation.affiliation_id
            for item in self._affiliations
        ):
            raise PaperError(
                "Duplicate affiliation_id: "
                f"{affiliation.affiliation_id}."
            )
        self._affiliations.append(affiliation)
        return self

    def add_section(
        self,
        section: PaperSection,
    ) -> "PaperBuilder":
        if not isinstance(section, PaperSection):
            raise PaperError(
                "section must be a PaperSection."
            )
        if any(
            item.section_id == section.section_id
            for item in self._sections
        ):
            raise PaperError(
                f"Duplicate section_id: {section.section_id}."
            )
        self._sections.append(section)
        return self

    def add_text_section(
        self,
        section_id: str,
        title: str,
        kind: SectionKind,
        text: str,
        *,
        order: int | None = None,
        citations: Sequence[Citation] = (),
        level: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PaperBuilder":
        return self.add_section(
            PaperSection(
                section_id=section_id,
                title=title,
                kind=kind,
                paragraphs=(
                    Paragraph(
                        text=text,
                        citations=tuple(citations),
                    ),
                ),
                order=(
                    len(self._sections)
                    if order is None
                    else order
                ),
                level=level,
                metadata=(
                    {}
                    if metadata is None
                    else metadata
                ),
            )
        )

    def add_result_section(
        self,
        section_id: str,
        result: Any,
        *,
        title: str | None = None,
        kind: SectionKind = SectionKind.RESULTS,
        summary: str | None = None,
        order: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PaperBuilder":
        paragraphs = (
            ()
            if summary is None
            else (Paragraph(summary),)
        )
        return self.add_section(
            PaperSection(
                section_id=section_id,
                title=(
                    title
                    or section_id.replace("_", " ").title()
                ),
                kind=kind,
                paragraphs=paragraphs,
                data=_json_value(result),
                order=(
                    len(self._sections)
                    if order is None
                    else order
                ),
                metadata=(
                    {}
                    if metadata is None
                    else metadata
                ),
            )
        )

    def add_table(
        self,
        table: PaperTable,
    ) -> "PaperBuilder":
        if not isinstance(table, PaperTable):
            raise PaperError(
                "table must be a PaperTable."
            )
        if any(
            item.table_id == table.table_id
            for item in self._tables
        ):
            raise PaperError(
                f"Duplicate table_id: {table.table_id}."
            )
        self._tables.append(table)
        return self

    def add_figure(
        self,
        figure: PaperFigure,
    ) -> "PaperBuilder":
        if not isinstance(figure, PaperFigure):
            raise PaperError(
                "figure must be a PaperFigure."
            )
        if any(
            item.figure_id == figure.figure_id
            for item in self._figures
        ):
            raise PaperError(
                f"Duplicate figure_id: {figure.figure_id}."
            )
        self._figures.append(figure)
        return self

    def add_reference(
        self,
        reference: Reference,
    ) -> "PaperBuilder":
        if not isinstance(reference, Reference):
            raise PaperError(
                "reference must be a Reference."
            )
        if any(
            item.reference_id == reference.reference_id
            for item in self._references
        ):
            raise PaperError(
                "Duplicate reference_id: "
                f"{reference.reference_id}."
            )
        self._references.append(reference)
        return self

    def build(self) -> PaperDocument:
        """Build the immutable paper document."""
        if self._abstract is None:
            raise PaperError(
                "Paper abstract must be set before build()."
            )
        if not self._sections:
            raise PaperError(
                "Paper must contain at least one section."
            )

        return PaperDocument(
            paper_id=self._paper_id,
            title=self._title,
            status=self._status,
            created_at=self._created_at,
            authors=tuple(self._authors),
            affiliations=tuple(self._affiliations),
            keywords=tuple(self._keywords),
            abstract=self._abstract,
            sections=tuple(self._sections),
            tables=tuple(self._tables),
            figures=tuple(self._figures),
            references=tuple(self._references),
            reproducibility=self._reproducibility,
            journal=self._journal,
            doi=self._doi,
            language=self._language,
            metadata=self._metadata,
        )


def _citation_markdown(citation: Citation) -> str:
    joined = "; ".join(
        f"@{reference_id}"
        for reference_id in citation.reference_ids
    )
    prefix = (
        ""
        if citation.prefix is None
        else f"{citation.prefix} "
    )
    locator = (
        ""
        if citation.locator is None
        else f", {citation.locator}"
    )
    suffix = (
        ""
        if citation.suffix is None
        else f" {citation.suffix}"
    )
    return f"[{prefix}{joined}{locator}{suffix}]"


def _reference_markdown(reference: Reference) -> str:
    authors = (
        ", ".join(reference.authors)
        if reference.authors
        else "Unknown author"
    )
    year = (
        "n.d."
        if reference.year is None
        else str(reference.year)
    )
    parts = [
        f"**{reference.reference_id}.** "
        f"{authors} ({year}). "
        f"*{reference.title}*."
    ]
    if reference.container_title is not None:
        parts.append(f" {reference.container_title}.")
    if reference.volume is not None:
        parts.append(f" {reference.volume}")
        if reference.issue is not None:
            parts.append(f"({reference.issue})")
        parts.append(".")
    if reference.pages is not None:
        parts.append(f" {reference.pages}.")
    if reference.doi is not None:
        parts.append(f" DOI: `{reference.doi}`.")
    if reference.url is not None:
        parts.append(f" {reference.url}")
    return "".join(parts)


def render_markdown(paper: PaperDocument) -> str:
    """Render a paper document as Markdown."""
    if not isinstance(paper, PaperDocument):
        raise PaperError(
            "paper must be a PaperDocument."
        )

    lines = [
        f"# {paper.title}",
        "",
    ]

    if paper.authors:
        lines.append(
            ", ".join(author.display_name for author in paper.authors)
        )
        lines.append("")

    if paper.journal is not None:
        lines.extend([
            f"**Journal:** {paper.journal}",
            "",
        ])

    lines.extend([
        f"**Status:** `{paper.status.value}`  ",
        f"**Paper ID:** `{paper.paper_id}`  ",
        f"**Created:** `{paper.created_at}`",
        "",
        "## Abstract",
        "",
        paper.abstract,
    ])

    if paper.keywords:
        lines.extend([
            "",
            "**Keywords:** " + ", ".join(paper.keywords),
        ])

    for section in paper.sections:
        lines.extend([
            "",
            f"{'#' * (section.level + 1)} {section.title}",
            "",
        ])

        for paragraph in section.paragraphs:
            text = paragraph.text
            if paragraph.citations:
                text += " " + " ".join(
                    _citation_markdown(item)
                    for item in paragraph.citations
                )
            lines.extend([text, ""])

        if section.data is not None:
            lines.extend([
                "```json",
                json.dumps(
                    section.data,
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ])

    for index, table in enumerate(paper.tables, start=1):
        lines.extend([
            "",
            f"## Table {index}. {table.title}",
            "",
        ])
        if table.caption is not None:
            lines.extend([table.caption, ""])

        lines.append(
            "| " + " | ".join(table.columns) + " |"
        )
        alignment_tokens = {
            TableAlignment.LEFT: ":---",
            TableAlignment.CENTER: ":---:",
            TableAlignment.RIGHT: "---:",
        }
        lines.append(
            "| "
            + " | ".join(
                alignment_tokens[item]
                for item in table.alignments
            )
            + " |"
        )
        for row in table.rows:
            lines.append(
                "| "
                + " | ".join(
                    "" if value is None else str(value)
                    for value in row
                )
                + " |"
            )
        for note in table.notes:
            lines.append(f"\n*Note:* {note}")

    for index, figure in enumerate(paper.figures, start=1):
        lines.extend([
            "",
            f"## Figure {index}. {figure.title}",
            "",
            f"![{figure.alt_text or figure.title}]"
            f"({figure.location})",
            "",
            figure.caption,
        ])

    if paper.references:
        lines.extend(["", "## References", ""])
        lines.extend(
            f"{index}. {_reference_markdown(reference)}"
            for index, reference in enumerate(
                paper.references,
                start=1,
            )
        )

    if paper.reproducibility is not None:
        lines.extend([
            "",
            "## Reproducibility",
            "",
            "```json",
            json.dumps(
                paper.reproducibility.as_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            "```",
        ])

    return "\n".join(lines).rstrip() + "\n"


_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(value: Any) -> str:
    text = str(value)
    return "".join(
        _LATEX_ESCAPE.get(character, character)
        for character in text
    )


def render_latex(paper: PaperDocument) -> str:
    """Render a standalone LaTeX article."""
    if not isinstance(paper, PaperDocument):
        raise PaperError(
            "paper must be a PaperDocument."
        )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage[margin=1in]{geometry}",
        "",
        rf"\title{{{_latex_escape(paper.title)}}}",
        rf"\author{{{_latex_escape(', '.join(author.display_name for author in paper.authors))}}}",
        rf"\date{{{_latex_escape(paper.created_at[:10])}}}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        _latex_escape(paper.abstract),
        r"\end{abstract}",
    ]

    if paper.keywords:
        lines.extend([
            "",
            r"\noindent\textbf{Keywords:} "
            + _latex_escape(", ".join(paper.keywords)),
        ])

    section_commands = {
        1: r"\section",
        2: r"\subsection",
        3: r"\subsubsection",
        4: r"\paragraph",
        5: r"\subparagraph",
        6: r"\subparagraph",
    }

    for section in paper.sections:
        lines.extend([
            "",
            section_commands[section.level]
            + "{"
            + _latex_escape(section.title)
            + "}",
        ])
        for paragraph in section.paragraphs:
            text = _latex_escape(paragraph.text)
            if paragraph.citations:
                keys = []
                for citation in paragraph.citations:
                    keys.extend(citation.reference_ids)
                text += r" \cite{" + ",".join(keys) + "}"
            lines.extend(["", text])

        if section.data is not None:
            lines.extend([
                "",
                r"\begin{verbatim}",
                json.dumps(
                    section.data,
                    ensure_ascii=False,
                    indent=2,
                ),
                r"\end{verbatim}",
            ])

    for table in paper.tables:
        column_spec = "".join(
            {
                TableAlignment.LEFT: "l",
                TableAlignment.CENTER: "c",
                TableAlignment.RIGHT: "r",
            }[alignment]
            for alignment in table.alignments
        )
        lines.extend([
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{_latex_escape(table.caption or table.title)}}}",
            rf"\label{{tab:{_latex_escape(table.table_id)}}}",
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            " & ".join(
                _latex_escape(column)
                for column in table.columns
            )
            + r" \\",
            r"\midrule",
        ])
        for row in table.rows:
            lines.append(
                " & ".join(
                    _latex_escape(
                        "" if value is None else value
                    )
                    for value in row
                )
                + r" \\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    for figure in paper.figures:
        lines.extend([
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width={figure.width_fraction:.3f}\linewidth]{{{_latex_escape(figure.location)}}}",
            rf"\caption{{{_latex_escape(figure.caption)}}}",
            rf"\label{{fig:{_latex_escape(figure.figure_id)}}}",
            r"\end{figure}",
        ])

    if paper.references:
        lines.extend([
            "",
            r"\begin{thebibliography}{99}",
        ])
        for reference in paper.references:
            authors = (
                ", ".join(reference.authors)
                if reference.authors
                else "Unknown author"
            )
            year = (
                "n.d."
                if reference.year is None
                else str(reference.year)
            )
            entry = (
                f"{authors}. {reference.title}. "
                f"{reference.container_title or ''} "
                f"({year})."
            )
            if reference.doi is not None:
                entry += f" DOI: {reference.doi}."
            lines.append(
                rf"\bibitem{{{_latex_escape(reference.reference_id)}}}"
                + _latex_escape(entry)
            )
        lines.append(r"\end{thebibliography}")

    lines.extend(["", r"\end{document}", ""])
    return "\n".join(lines)


def paper_from_mapping(
    value: Mapping[str, Any],
) -> PaperDocument:
    """Reconstruct a ``PaperDocument`` from serialized data."""
    if not isinstance(value, Mapping):
        raise PaperError("value must be a mapping.")

    try:
        authors = tuple(
            Author(
                author_id=item["author_id"],
                given_name=item["given_name"],
                family_name=item["family_name"],
                affiliations=tuple(
                    item.get("affiliations", ())
                ),
                email=item.get("email"),
                orcid=item.get("orcid"),
                corresponding=item.get(
                    "corresponding",
                    False,
                ),
                contribution=item.get("contribution"),
            )
            for item in value.get("authors", ())
        )
        affiliations = tuple(
            Affiliation(
                affiliation_id=item["affiliation_id"],
                institution=item["institution"],
                department=item.get("department"),
                city=item.get("city"),
                country=item.get("country"),
            )
            for item in value.get("affiliations", ())
        )
        references = tuple(
            Reference(
                reference_id=item["reference_id"],
                reference_type=ReferenceType(
                    item["reference_type"]
                ),
                title=item["title"],
                authors=tuple(item.get("authors", ())),
                year=item.get("year"),
                container_title=item.get(
                    "container_title"
                ),
                volume=item.get("volume"),
                issue=item.get("issue"),
                pages=item.get("pages"),
                doi=item.get("doi"),
                url=item.get("url"),
                publisher=item.get("publisher"),
                accessed_at=item.get("accessed_at"),
                metadata=item.get("metadata", {}),
            )
            for item in value.get("references", ())
        )

        sections: list[PaperSection] = []
        for item in value["sections"]:
            paragraphs = tuple(
                Paragraph(
                    text=paragraph["text"],
                    citations=tuple(
                        Citation(
                            reference_ids=tuple(
                                citation["reference_ids"]
                            ),
                            locator=citation.get("locator"),
                            prefix=citation.get("prefix"),
                            suffix=citation.get("suffix"),
                        )
                        for citation in paragraph.get(
                            "citations",
                            (),
                        )
                    ),
                    label=paragraph.get("label"),
                )
                for paragraph in item.get(
                    "paragraphs",
                    (),
                )
            )
            sections.append(
                PaperSection(
                    section_id=item["section_id"],
                    title=item["title"],
                    kind=SectionKind(item["kind"]),
                    paragraphs=paragraphs,
                    data=item.get("data"),
                    order=item.get("order", 0),
                    level=item.get("level", 1),
                    metadata=item.get("metadata", {}),
                )
            )

        tables = tuple(
            PaperTable(
                table_id=item["table_id"],
                title=item["title"],
                columns=tuple(item["columns"]),
                rows=tuple(
                    tuple(row)
                    for row in item.get("rows", ())
                ),
                caption=item.get("caption"),
                notes=tuple(item.get("notes", ())),
                alignments=tuple(
                    TableAlignment(alignment)
                    for alignment in item.get(
                        "alignments",
                        (),
                    )
                ),
                source=item.get("source"),
                order=item.get("order", 0),
                metadata=item.get("metadata", {}),
            )
            for item in value.get("tables", ())
        )

        figures = tuple(
            PaperFigure(
                figure_id=item["figure_id"],
                title=item["title"],
                kind=FigureKind(item["kind"]),
                location=item["location"],
                caption=item["caption"],
                alt_text=item.get("alt_text"),
                width_fraction=item.get(
                    "width_fraction",
                    1.0,
                ),
                source=item.get("source"),
                order=item.get("order", 0),
                metadata=item.get("metadata", {}),
            )
            for item in value.get("figures", ())
        )

        reproducibility_data = value.get(
            "reproducibility"
        )
        reproducibility = (
            None
            if reproducibility_data is None
            else ReproducibilityManifest(
                engine_version=reproducibility_data[
                    "engine_version"
                ],
                schema_version=reproducibility_data.get(
                    "schema_version",
                    "1.0",
                ),
                commit=reproducibility_data.get(
                    "commit"
                ),
                seed=reproducibility_data.get("seed"),
                command=reproducibility_data.get(
                    "command"
                ),
                dataset_ids=tuple(
                    reproducibility_data.get(
                        "dataset_ids",
                        (),
                    )
                ),
                artifact_checksums=reproducibility_data.get(
                    "artifact_checksums",
                    {},
                ),
                environment=reproducibility_data.get(
                    "environment",
                    {},
                ),
            )
        )

        return PaperDocument(
            paper_id=value["paper_id"],
            title=value["title"],
            status=PaperStatus(value["status"]),
            created_at=value["created_at"],
            authors=authors,
            affiliations=affiliations,
            keywords=tuple(value.get("keywords", ())),
            abstract=value["abstract"],
            sections=tuple(sections),
            tables=tables,
            figures=figures,
            references=references,
            reproducibility=reproducibility,
            journal=value.get("journal"),
            doi=value.get("doi"),
            language=value.get("language", "en"),
            metadata=value.get("metadata", {}),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        PaperError,
    ) as exc:
        if isinstance(exc, PaperError):
            raise
        raise PaperError(
            "Invalid serialized paper mapping."
        ) from exc


__all__ = [
    "Affiliation",
    "Author",
    "Citation",
    "FigureKind",
    "JSONScalar",
    "JSONValue",
    "PaperBuilder",
    "PaperDocument",
    "PaperError",
    "PaperFigure",
    "PaperSection",
    "PaperStatus",
    "PaperTable",
    "Paragraph",
    "Reference",
    "ReferenceType",
    "ReproducibilityManifest",
    "SectionKind",
    "TableAlignment",
    "paper_from_mapping",
    "render_latex",
    "render_markdown",
    "utc_now_iso",
]

