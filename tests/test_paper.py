"""
Tests for roif.paper.

Part 1A
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
import json
import re

import pytest

from roif.paper import (
    Affiliation,
    Author,
    Citation,
    FigureKind,
    PaperError,
    Paragraph,
    PaperStatus,
    Reference,
    ReferenceType,
    ReproducibilityManifest,
    SectionKind,
    TableAlignment,
    utc_now_iso,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_author() -> Author:
    return Author(
        author_id="author-1",
        given_name="Theo",
        family_name="Shapoval",
        affiliations=("aff-1",),
        email="theo@example.com",
        orcid="0000-0000-0000-0001",
        corresponding=True,
        contribution="Conceptualization",
    )


def make_affiliation() -> Affiliation:
    return Affiliation(
        affiliation_id="aff-1",
        institution="ROIF Research Center",
        department="AI Laboratory",
        city="Kyiv",
        country="Ukraine",
    )


def make_reference() -> Reference:
    return Reference(
        reference_id="ref-1",
        reference_type=ReferenceType.ARTICLE,
        title="Recursive Organic Integration Framework",
        authors=("Theo Shapoval",),
        year=2026,
        container_title="Journal of Complex Systems",
        volume="10",
        issue="2",
        pages="100-120",
        doi="10.1000/example",
    )


def make_citation() -> Citation:
    return Citation(
        reference_ids=("ref-1",),
    )


# ---------------------------------------------------------------------
# utc_now_iso
# ---------------------------------------------------------------------


def test_utc_now_iso_returns_iso_timestamp() -> None:
    value = utc_now_iso()

    assert value.endswith("Z")

    parsed = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    assert parsed.tzinfo is not None


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------


def test_paper_status_values() -> None:
    assert tuple(item.value for item in PaperStatus) == (
        "draft",
        "preprint",
        "submitted",
        "accepted",
        "published",
    )


def test_section_kind_contains_expected_values() -> None:
    values = {item.value for item in SectionKind}

    assert "abstract" in values
    assert "results" in values
    assert "discussion" in values
    assert "appendix" in values
    assert "custom" in values


def test_table_alignment_values() -> None:
    assert tuple(item.value for item in TableAlignment) == (
        "left",
        "center",
        "right",
    )


def test_reference_type_contains_article() -> None:
    assert ReferenceType.ARTICLE.value == "article"


def test_figure_kind_contains_plot() -> None:
    assert FigureKind.PLOT.value == "plot"


# ---------------------------------------------------------------------
# Author
# ---------------------------------------------------------------------


def test_author_properties() -> None:
    author = make_author()

    assert author.author_id == "author-1"
    assert author.display_name == "Theo Shapoval"
    assert author.corresponding is True
    assert author.affiliations == ("aff-1",)


def test_author_is_frozen() -> None:
    author = make_author()

    with pytest.raises(FrozenInstanceError):
        author.family_name = "Other"


@pytest.mark.parametrize(
    "value",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_author_rejects_bad_identifier(value) -> None:
    with pytest.raises(PaperError):
        Author(
            author_id=value,
            given_name="Theo",
            family_name="Shapoval",
        )


@pytest.mark.parametrize(
    "field",
    (
        "given_name",
        "family_name",
    ),
)
def test_author_rejects_empty_names(field) -> None:
    kwargs = {
        "author_id": "author",
        "given_name": "Theo",
        "family_name": "Shapoval",
    }

    kwargs[field] = " "

    with pytest.raises(PaperError):
        Author(**kwargs)


def test_author_requires_boolean_corresponding() -> None:
    with pytest.raises(PaperError):
        Author(
            author_id="author",
            given_name="Theo",
            family_name="Shapoval",
            corresponding="yes",
        )


def test_author_as_dict() -> None:
    author = make_author()

    payload = author.as_dict()

    assert payload["display_name"] == "Theo Shapoval"
    assert payload["corresponding"] is True
    assert payload["author_id"] == "author-1"


# ---------------------------------------------------------------------
# Affiliation
# ---------------------------------------------------------------------


def test_affiliation_properties() -> None:
    affiliation = make_affiliation()

    assert affiliation.affiliation_id == "aff-1"
    assert affiliation.institution == "ROIF Research Center"
    assert affiliation.department == "AI Laboratory"


def test_affiliation_is_frozen() -> None:
    affiliation = make_affiliation()

    with pytest.raises(FrozenInstanceError):
        affiliation.city = "Lviv"


@pytest.mark.parametrize(
    "value",
    (
        "",
        " ",
        None,
        123,
    ),
)
def test_affiliation_rejects_bad_identifier(value) -> None:
    with pytest.raises(PaperError):
        Affiliation(
            affiliation_id=value,
            institution="Institute",
        )


def test_affiliation_requires_institution() -> None:
    with pytest.raises(PaperError):
        Affiliation(
            affiliation_id="aff",
            institution=" ",
        )


def test_affiliation_as_dict() -> None:
    payload = make_affiliation().as_dict()

    assert payload["country"] == "Ukraine"
    assert payload["city"] == "Kyiv"


# ---------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------


def test_citation_properties() -> None:
    citation = Citation(
        reference_ids=("ref-1", "ref-2"),
        locator="p.15",
        prefix="see",
        suffix="for details",
    )

    assert citation.reference_ids == (
        "ref-1",
        "ref-2",
    )
    assert citation.locator == "p.15"


def test_citation_requires_reference_ids() -> None:
    with pytest.raises(PaperError):
        Citation(reference_ids=())


def test_citation_rejects_duplicates() -> None:
    with pytest.raises(PaperError):
        Citation(
            reference_ids=(
                "ref",
                "ref",
            )
        )


def test_citation_as_dict() -> None:
    payload = make_citation().as_dict()

    assert payload["reference_ids"] == ["ref-1"]


# ---------------------------------------------------------------------
# Paragraph
# ---------------------------------------------------------------------


def test_paragraph_properties() -> None:
    paragraph = Paragraph(
        text="Example paragraph.",
        citations=(make_citation(),),
        label="intro",
    )

    assert paragraph.text == "Example paragraph."
    assert paragraph.label == "intro"
    assert len(paragraph.citations) == 1


def test_paragraph_requires_text() -> None:
    with pytest.raises(PaperError):
        Paragraph(text=" ")


def test_paragraph_rejects_invalid_label() -> None:
    with pytest.raises(PaperError):
        Paragraph(
            text="Example",
            label="bad label",
        )


def test_paragraph_requires_citation_objects() -> None:
    with pytest.raises(PaperError):
        Paragraph(
            text="Example",
            citations=("bad",),
        )


def test_paragraph_as_dict() -> None:
    payload = Paragraph(
        text="Paragraph",
        citations=(make_citation(),),
    ).as_dict()

    assert payload["text"] == "Paragraph"
    assert len(payload["citations"]) == 1


# ---------------------------------------------------------------------
# Serialization sanity
# ---------------------------------------------------------------------


def test_author_json_serializable() -> None:
    json.dumps(make_author().as_dict())


def test_affiliation_json_serializable() -> None:
    json.dumps(make_affiliation().as_dict())


def test_citation_json_serializable() -> None:
    json.dumps(make_citation().as_dict())


def test_paragraph_json_serializable() -> None:
    json.dumps(
        Paragraph(
            text="Paragraph",
            citations=(make_citation(),),
        ).as_dict()
    )
# ---------------------------------------------------------------------
# PaperSection
# ---------------------------------------------------------------------


def make_section() -> PaperSection:
    return PaperSection(
        section_id="methods",
        title="Methods",
        kind=SectionKind.METHODS,
        paragraphs=(
            Paragraph(
                text="The ROIF method was evaluated.",
                citations=(make_citation(),),
            ),
        ),
        order=1,
        level=1,
        metadata={"source": "test"},
    )


def test_section_properties() -> None:
    section = make_section()

    assert section.section_id == "methods"
    assert section.title == "Methods"
    assert section.kind is SectionKind.METHODS
    assert section.order == 1
    assert section.level == 1
    assert len(section.paragraphs) == 1
    assert isinstance(section.metadata, MappingProxyType)


def test_section_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        make_section().title = "Changed"


@pytest.mark.parametrize(
    "value",
    (
        "",
        " ",
        None,
        "bad section",
    ),
)
def test_section_rejects_bad_identifier(value) -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id=value,
            title="Section",
            kind=SectionKind.CUSTOM,
            paragraphs=(Paragraph("Text"),),
        )


def test_section_requires_content() -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id="empty",
            title="Empty",
            kind=SectionKind.CUSTOM,
        )


def test_section_accepts_data_only() -> None:
    section = PaperSection(
        section_id="results-data",
        title="Results Data",
        kind=SectionKind.RESULTS,
        data={"accuracy": 0.9},
    )

    assert section.paragraphs == ()
    assert section.data == {"accuracy": 0.9}


def test_section_accepts_paragraphs_and_data() -> None:
    section = PaperSection(
        section_id="combined",
        title="Combined",
        kind=SectionKind.RESULTS,
        paragraphs=(Paragraph("Summary."),),
        data={"value": 1},
    )

    assert section.paragraphs[0].text == "Summary."
    assert section.data["value"] == 1


@pytest.mark.parametrize(
    "level",
    (
        0,
        7,
        -1,
        True,
        1.5,
        "1",
    ),
)
def test_section_rejects_bad_level(level) -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id="section",
            title="Section",
            kind=SectionKind.CUSTOM,
            paragraphs=(Paragraph("Text"),),
            level=level,
        )


@pytest.mark.parametrize(
    "order",
    (
        -1,
        True,
        1.5,
        "0",
    ),
)
def test_section_rejects_bad_order(order) -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id="section",
            title="Section",
            kind=SectionKind.CUSTOM,
            paragraphs=(Paragraph("Text"),),
            order=order,
        )


def test_section_requires_paragraph_objects() -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id="section",
            title="Section",
            kind=SectionKind.CUSTOM,
            paragraphs=("bad",),
        )


def test_section_as_dict() -> None:
    payload = make_section().as_dict()

    assert payload["section_id"] == "methods"
    assert payload["kind"] == "methods"
    assert payload["paragraphs"][0]["text"].startswith(
        "The ROIF method"
    )


# ---------------------------------------------------------------------
# PaperTable
# ---------------------------------------------------------------------


def make_table() -> PaperTable:
    return PaperTable(
        table_id="table-1",
        title="Validation Results",
        columns=("Method", "Accuracy", "Runtime"),
        rows=(
            ("ROIF", 0.95, 1.2),
            ("Baseline", 0.75, 0.4),
        ),
        caption="Comparison of ROIF and baseline methods.",
        notes=("Higher accuracy is better.",),
        alignments=(
            TableAlignment.LEFT,
            TableAlignment.RIGHT,
            TableAlignment.RIGHT,
        ),
        source="benchmark",
        order=1,
        metadata={"source": "test"},
    )


def test_table_properties() -> None:
    table = make_table()

    assert table.table_id == "table-1"
    assert table.columns == (
        "Method",
        "Accuracy",
        "Runtime",
    )
    assert len(table.rows) == 2
    assert table.rows[0][1] == pytest.approx(0.95)
    assert table.alignments[0] is TableAlignment.LEFT
    assert isinstance(table.metadata, MappingProxyType)


def test_table_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        make_table().title = "Changed"


def test_table_uses_default_alignments() -> None:
    table = PaperTable(
        table_id="table",
        title="Table",
        columns=("A", "B"),
        rows=((1, 2),),
    )

    assert table.alignments == (
        TableAlignment.LEFT,
        TableAlignment.LEFT,
    )


def test_table_requires_columns() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=(),
            rows=(),
        )


def test_table_rejects_duplicate_columns() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A", "A"),
            rows=((1, 2),),
        )


def test_table_rejects_wrong_row_width() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A", "B"),
            rows=((1,),),
        )


def test_table_rejects_nested_cell_value() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A",),
            rows=(({"x": 1},),),
        )


def test_table_rejects_bad_alignment_count() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A", "B"),
            rows=((1, 2),),
            alignments=(TableAlignment.LEFT,),
        )


def test_table_rejects_bad_alignment_value() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A",),
            rows=((1,),),
            alignments=("left",),
        )


def test_table_rejects_duplicate_notes() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A",),
            rows=((1,),),
            notes=("Note", "Note"),
        )


def test_table_allows_empty_rows() -> None:
    table = PaperTable(
        table_id="empty-table",
        title="Empty Table",
        columns=("A", "B"),
        rows=(),
    )

    assert table.rows == ()


def test_table_as_dict() -> None:
    payload = make_table().as_dict()

    assert payload["table_id"] == "table-1"
    assert payload["rows"][0][0] == "ROIF"
    assert payload["alignments"] == [
        "left",
        "right",
        "right",
    ]


# ---------------------------------------------------------------------
# PaperFigure
# ---------------------------------------------------------------------


def make_figure() -> PaperFigure:
    return PaperFigure(
        figure_id="figure-1",
        title="Cascade Dynamics",
        kind=FigureKind.PLOT,
        location="figures/cascade.png",
        caption="Cascade coherence over time.",
        alt_text="Line plot of cascade coherence.",
        width_fraction=0.8,
        source="validation",
        order=1,
        metadata={"source": "test"},
    )


def test_figure_properties() -> None:
    figure = make_figure()

    assert figure.figure_id == "figure-1"
    assert figure.kind is FigureKind.PLOT
    assert figure.location == "figures/cascade.png"
    assert figure.width_fraction == pytest.approx(0.8)
    assert isinstance(figure.metadata, MappingProxyType)


def test_figure_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        make_figure().caption = "Changed"


@pytest.mark.parametrize(
    "value",
    (
        0.0,
        -0.1,
        1.1,
        float("nan"),
        True,
        "0.5",
    ),
)
def test_figure_rejects_bad_width(value) -> None:
    with pytest.raises(PaperError):
        PaperFigure(
            figure_id="figure",
            title="Figure",
            kind=FigureKind.PLOT,
            location="figure.png",
            caption="Caption.",
            width_fraction=value,
        )


def test_figure_requires_figure_kind() -> None:
    with pytest.raises(PaperError):
        PaperFigure(
            figure_id="figure",
            title="Figure",
            kind="plot",
            location="figure.png",
            caption="Caption.",
        )


def test_figure_requires_location() -> None:
    with pytest.raises(PaperError):
        PaperFigure(
            figure_id="figure",
            title="Figure",
            kind=FigureKind.PLOT,
            location=" ",
            caption="Caption.",
        )


def test_figure_as_dict() -> None:
    payload = make_figure().as_dict()

    assert payload["figure_id"] == "figure-1"
    assert payload["kind"] == "plot"
    assert payload["width_fraction"] == pytest.approx(0.8)


# ---------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------


def test_reference_properties() -> None:
    reference = make_reference()

    assert reference.reference_id == "ref-1"
    assert reference.reference_type is ReferenceType.ARTICLE
    assert reference.year == 2026
    assert reference.doi == "10.1000/example"


def test_reference_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        make_reference().title = "Changed"


@pytest.mark.parametrize(
    "year",
    (
        999,
        10000,
        True,
        2026.5,
        "2026",
    ),
)
def test_reference_rejects_bad_year(year) -> None:
    with pytest.raises(PaperError):
        Reference(
            reference_id="ref",
            reference_type=ReferenceType.ARTICLE,
            title="Title",
            year=year,
        )


def test_reference_allows_missing_year() -> None:
    reference = Reference(
        reference_id="ref",
        reference_type=ReferenceType.WEB,
        title="Web Resource",
        url="https://example.com",
    )

    assert reference.year is None


def test_reference_rejects_duplicate_authors() -> None:
    with pytest.raises(PaperError):
        Reference(
            reference_id="ref",
            reference_type=ReferenceType.ARTICLE,
            title="Title",
            authors=("A", "A"),
        )


def test_reference_normalizes_accessed_at() -> None:
    reference = Reference(
        reference_id="ref",
        reference_type=ReferenceType.WEB,
        title="Web Resource",
        url="https://example.com",
        accessed_at="2026-08-01T12:00:00+03:00",
    )

    assert reference.accessed_at == "2026-08-01T09:00:00Z"


def test_reference_rejects_bad_accessed_at() -> None:
    with pytest.raises(PaperError):
        Reference(
            reference_id="ref",
            reference_type=ReferenceType.WEB,
            title="Web Resource",
            accessed_at="bad-date",
        )


def test_reference_as_dict() -> None:
    payload = make_reference().as_dict()

    assert payload["reference_id"] == "ref-1"
    assert payload["reference_type"] == "article"
    assert payload["authors"] == ["Theo Shapoval"]


# ---------------------------------------------------------------------
# ReproducibilityManifest
# ---------------------------------------------------------------------


def make_manifest() -> ReproducibilityManifest:
    return ReproducibilityManifest(
        engine_version="1.0.0",
        schema_version="1.0",
        commit="abc123",
        seed=42,
        command="python run_experiment.py",
        dataset_ids=("dataset-1", "dataset-2"),
        artifact_checksums={
            "results.json": "abc",
            "figure.png": "def",
        },
        environment={
            "python": "3.13",
            "platform": "win32",
        },
    )


def test_manifest_properties() -> None:
    manifest = make_manifest()

    assert manifest.engine_version == "1.0.0"
    assert manifest.seed == 42
    assert manifest.dataset_ids == (
        "dataset-1",
        "dataset-2",
    )
    assert isinstance(
        manifest.artifact_checksums,
        MappingProxyType,
    )
    assert isinstance(
        manifest.environment,
        MappingProxyType,
    )


def test_manifest_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        make_manifest().seed = 1


@pytest.mark.parametrize(
    "seed",
    (
        True,
        1.5,
        "42",
    ),
)
def test_manifest_rejects_bad_seed(seed) -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version="1.0",
            seed=seed,
        )


def test_manifest_rejects_duplicate_dataset_ids() -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version="1.0",
            dataset_ids=("dataset", "dataset"),
        )


def test_manifest_rejects_empty_checksum() -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version="1.0",
            artifact_checksums={"result.json": " "},
        )


def test_manifest_as_dict() -> None:
    payload = make_manifest().as_dict()

    assert payload["engine_version"] == "1.0.0"
    assert payload["seed"] == 42
    assert payload["dataset_ids"] == [
        "dataset-1",
        "dataset-2",
    ]


# ---------------------------------------------------------------------
# Part 1B serialization sanity
# ---------------------------------------------------------------------


def test_section_json_serializable() -> None:
    json.dumps(make_section().as_dict())


def test_table_json_serializable() -> None:
    json.dumps(make_table().as_dict())


def test_figure_json_serializable() -> None:
    json.dumps(make_figure().as_dict())


def test_reference_json_serializable() -> None:
    json.dumps(make_reference().as_dict())


def test_manifest_json_serializable() -> None:
    json.dumps(make_manifest().as_dict())
# ---------------------------------------------------------------------
# PaperDocument helpers
# ---------------------------------------------------------------------


def make_document() -> PaperDocument:
    return PaperDocument(
        paper_id="paper-1",
        title="ROIF Validation Study",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(make_author(),),
        affiliations=(make_affiliation(),),
        keywords=("ROIF", "cascade", "validation"),
        abstract=(
            "This study evaluates the ROIF framework "
            "for cascade localization and intervention."
        ),
        sections=(
            PaperSection(
                section_id="introduction",
                title="Introduction",
                kind=SectionKind.INTRODUCTION,
                paragraphs=(
                    Paragraph(
                        text="ROIF models cascading dynamics.",
                        citations=(make_citation(),),
                    ),
                ),
                order=0,
            ),
            make_section(),
        ),
        tables=(make_table(),),
        figures=(make_figure(),),
        references=(make_reference(),),
        reproducibility=make_manifest(),
        journal="Journal of Complex Systems",
        doi="10.1000/roif-paper",
        language="en",
        metadata={"experiment": "E1"},
    )


# ---------------------------------------------------------------------
# PaperDocument
# ---------------------------------------------------------------------


def test_document_properties() -> None:
    document = make_document()

    assert document.paper_id == "paper-1"
    assert document.title == "ROIF Validation Study"
    assert document.status is PaperStatus.DRAFT
    assert document.created_at == "2026-08-01T09:00:00Z"
    assert document.authors[0].display_name == "Theo Shapoval"
    assert document.affiliations[0].institution == (
        "ROIF Research Center"
    )
    assert document.keywords == (
        "ROIF",
        "cascade",
        "validation",
    )
    assert document.journal == "Journal of Complex Systems"
    assert document.doi == "10.1000/roif-paper"
    assert document.language == "en"
    assert isinstance(document.metadata, MappingProxyType)


def test_document_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        make_document().title = "Changed"


def test_document_sorts_sections() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="second",
                title="Second",
                kind=SectionKind.RESULTS,
                paragraphs=(Paragraph("Second."),),
                order=2,
            ),
            PaperSection(
                section_id="first",
                title="First",
                kind=SectionKind.METHODS,
                paragraphs=(Paragraph("First."),),
                order=1,
            ),
        ),
    )

    assert tuple(
        section.section_id
        for section in document.sections
    ) == (
        "first",
        "second",
    )


def test_document_sorts_tables() -> None:
    first = PaperTable(
        table_id="first",
        title="First",
        columns=("A",),
        rows=((1,),),
        order=1,
    )
    second = PaperTable(
        table_id="second",
        title="Second",
        columns=("A",),
        rows=((2,),),
        order=2,
    )

    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(make_section(),),
        tables=(second, first),
        references=(make_reference(),),
    )

    assert tuple(
        table.table_id
        for table in document.tables
    ) == (
        "first",
        "second",
    )


def test_document_sorts_figures() -> None:
    first = PaperFigure(
        figure_id="first",
        title="First",
        kind=FigureKind.PLOT,
        location="first.png",
        caption="First.",
        order=1,
    )
    second = PaperFigure(
        figure_id="second",
        title="Second",
        kind=FigureKind.PLOT,
        location="second.png",
        caption="Second.",
        order=2,
    )

    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(make_section(),),
        figures=(second, first),
        references=(make_reference(),),
    )

    assert tuple(
        figure.figure_id
        for figure in document.figures
    ) == (
        "first",
        "second",
    )


def test_document_rejects_duplicate_author_ids() -> None:
    author = make_author()

    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(author, author),
            affiliations=(make_affiliation(),),
            keywords=(),
            abstract="Abstract.",
            sections=(make_section(),),
            references=(make_reference(),),
        )


def test_document_rejects_duplicate_affiliation_ids() -> None:
    affiliation = make_affiliation()

    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(affiliation, affiliation),
            keywords=(),
            abstract="Abstract.",
            sections=(make_section(),),
            references=(make_reference(),),
        )


def test_document_rejects_unknown_author_affiliation() -> None:
    author = Author(
        author_id="author",
        given_name="Theo",
        family_name="Shapoval",
        affiliations=("missing",),
    )

    with pytest.raises(
        PaperError,
        match="unknown affiliations",
    ):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(author,),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(make_section(),),
            references=(make_reference(),),
        )


def test_document_rejects_duplicate_section_ids() -> None:
    section = make_section()

    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(section, section),
            references=(make_reference(),),
        )


def test_document_rejects_duplicate_table_ids() -> None:
    table = make_table()

    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(make_section(),),
            tables=(table, table),
            references=(make_reference(),),
        )


def test_document_rejects_duplicate_figure_ids() -> None:
    figure = make_figure()

    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(make_section(),),
            figures=(figure, figure),
            references=(make_reference(),),
        )


def test_document_rejects_duplicate_reference_ids() -> None:
    reference = make_reference()

    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(make_section(),),
            references=(reference, reference),
        )


def test_document_rejects_unknown_citation() -> None:
    section = PaperSection(
        section_id="section",
        title="Section",
        kind=SectionKind.INTRODUCTION,
        paragraphs=(
            Paragraph(
                text="Text.",
                citations=(
                    Citation(
                        reference_ids=("missing-ref",)
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        PaperError,
        match="unknown citations",
    ):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(section,),
            references=(make_reference(),),
        )


def test_document_rejects_duplicate_keywords() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=("ROIF", "ROIF"),
            abstract="Abstract.",
            sections=(make_section(),),
            references=(make_reference(),),
        )


def test_document_word_count() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="One two three.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(
                    Paragraph("Four five."),
                    Paragraph("Six seven eight."),
                ),
            ),
        ),
    )

    assert document.word_count == 8


def test_document_section_lookup() -> None:
    document = make_document()

    assert document.section("methods").title == "Methods"


def test_document_section_lookup_failure() -> None:
    with pytest.raises(KeyError):
        make_document().section("missing")


def test_document_reference_lookup() -> None:
    document = make_document()

    assert document.reference("ref-1").year == 2026


def test_document_reference_lookup_failure() -> None:
    with pytest.raises(KeyError):
        make_document().reference("missing")


def test_document_as_dict() -> None:
    payload = make_document().as_dict()

    assert payload["paper_id"] == "paper-1"
    assert payload["status"] == "draft"
    assert payload["authors"][0]["author_id"] == "author-1"
    assert payload["sections"][0]["section_id"] == (
        "introduction"
    )
    assert payload["tables"][0]["table_id"] == "table-1"
    assert payload["figures"][0]["figure_id"] == (
        "figure-1"
    )
    assert payload["references"][0]["reference_id"] == (
        "ref-1"
    )
    assert payload["reproducibility"]["seed"] == 42
    assert payload["word_count"] == make_document().word_count


def test_document_to_json() -> None:
    text = make_document().to_json(
        sort_keys=True,
    )
    payload = json.loads(text)

    assert payload["paper_id"] == "paper-1"
    assert payload["title"] == "ROIF Validation Study"


@pytest.mark.parametrize(
    "indent",
    (
        -1,
        True,
        1.5,
        "2",
    ),
)
def test_document_to_json_rejects_bad_indent(indent) -> None:
    with pytest.raises(PaperError):
        make_document().to_json(indent=indent)


def test_document_to_json_rejects_bad_sort_keys() -> None:
    with pytest.raises(PaperError):
        make_document().to_json(
            sort_keys="yes",
        )


# ---------------------------------------------------------------------
# PaperBuilder
# ---------------------------------------------------------------------


def make_builder() -> PaperBuilder:
    return PaperBuilder(
        title="ROIF Study",
        paper_id="paper-builder",
        created_at="2026-08-01T09:00:00Z",
        status=PaperStatus.DRAFT,
        language="en",
    )


def test_builder_builds_document() -> None:
    document = (
        make_builder()
        .add_affiliation(make_affiliation())
        .add_author(make_author())
        .set_abstract("Abstract text.")
        .set_keywords(("ROIF", "validation"))
        .set_journal("Journal")
        .set_doi("10.1000/example")
        .set_reproducibility(make_manifest())
        .update_metadata({"experiment": "E2"})
        .add_reference(make_reference())
        .add_section(make_section())
        .add_table(make_table())
        .add_figure(make_figure())
        .build()
    )

    assert isinstance(document, PaperDocument)
    assert document.title == "ROIF Study"
    assert document.journal == "Journal"
    assert document.doi == "10.1000/example"
    assert document.keywords == (
        "ROIF",
        "validation",
    )
    assert document.metadata["experiment"] == "E2"


def test_builder_requires_abstract() -> None:
    with pytest.raises(
        PaperError,
        match="abstract must be set",
    ):
        (
            make_builder()
            .add_reference(make_reference())
            .add_section(make_section())
            .build()
        )


def test_builder_requires_section() -> None:
    with pytest.raises(
        PaperError,
        match="at least one section",
    ):
        (
            make_builder()
            .set_abstract("Abstract.")
            .build()
        )


def test_builder_rejects_duplicate_author() -> None:
    builder = make_builder()
    author = make_author()
    builder.add_author(author)

    with pytest.raises(PaperError):
        builder.add_author(author)


def test_builder_rejects_duplicate_affiliation() -> None:
    builder = make_builder()
    affiliation = make_affiliation()
    builder.add_affiliation(affiliation)

    with pytest.raises(PaperError):
        builder.add_affiliation(affiliation)


def test_builder_rejects_duplicate_section() -> None:
    builder = make_builder()
    section = make_section()
    builder.add_section(section)

    with pytest.raises(PaperError):
        builder.add_section(section)


def test_builder_rejects_duplicate_table() -> None:
    builder = make_builder()
    table = make_table()
    builder.add_table(table)

    with pytest.raises(PaperError):
        builder.add_table(table)


def test_builder_rejects_duplicate_figure() -> None:
    builder = make_builder()
    figure = make_figure()
    builder.add_figure(figure)

    with pytest.raises(PaperError):
        builder.add_figure(figure)


def test_builder_rejects_duplicate_reference() -> None:
    builder = make_builder()
    reference = make_reference()
    builder.add_reference(reference)

    with pytest.raises(PaperError):
        builder.add_reference(reference)


def test_builder_add_text_section() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_reference(make_reference())
        .add_text_section(
            "introduction",
            "Introduction",
            SectionKind.INTRODUCTION,
            "Introductory text.",
            citations=(make_citation(),),
        )
        .build()
    )

    section = document.section("introduction")

    assert section.paragraphs[0].text == (
        "Introductory text."
    )
    assert section.kind is SectionKind.INTRODUCTION


def test_builder_add_result_section() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_result_section(
            "validation",
            {
                "accuracy": 0.95,
                "runtime": 1.2,
            },
            title="Validation",
            summary="Validation summary.",
        )
        .build()
    )

    section = document.section("validation")

    assert section.kind is SectionKind.RESULTS
    assert section.data["accuracy"] == pytest.approx(0.95)
    assert section.paragraphs[0].text == (
        "Validation summary."
    )


def test_builder_add_result_section_without_summary() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_result_section(
            "benchmark",
            {"score": 1.0},
        )
        .build()
    )

    section = document.section("benchmark")

    assert section.paragraphs == ()
    assert section.data["score"] == pytest.approx(1.0)


def test_builder_set_status() -> None:
    document = (
        make_builder()
        .set_status(PaperStatus.SUBMITTED)
        .set_abstract("Abstract.")
        .add_result_section(
            "results",
            {"value": 1},
        )
        .build()
    )

    assert document.status is PaperStatus.SUBMITTED


def test_builder_set_journal_none() -> None:
    builder = make_builder()
    assert builder.set_journal(None) is builder


def test_builder_set_doi_none() -> None:
    builder = make_builder()
    assert builder.set_doi(None) is builder


def test_builder_set_reproducibility_none() -> None:
    builder = make_builder()
    assert builder.set_reproducibility(None) is builder


def test_builder_rejects_bad_status() -> None:
    with pytest.raises(PaperError):
        make_builder().set_status("draft")


def test_builder_rejects_bad_author() -> None:
    with pytest.raises(PaperError):
        make_builder().add_author("bad")


def test_builder_rejects_bad_section() -> None:
    with pytest.raises(PaperError):
        make_builder().add_section("bad")


def test_builder_rejects_bad_table() -> None:
    with pytest.raises(PaperError):
        make_builder().add_table("bad")


def test_builder_rejects_bad_figure() -> None:
    with pytest.raises(PaperError):
        make_builder().add_figure("bad")


def test_builder_rejects_bad_reference() -> None:
    with pytest.raises(PaperError):
        make_builder().add_reference("bad")


def test_builder_rejects_bad_manifest() -> None:
    with pytest.raises(PaperError):
        make_builder().set_reproducibility("bad")
# ---------------------------------------------------------------------
# Additional imports required by the following test blocks
# ---------------------------------------------------------------------

from types import MappingProxyType

from roif.paper import (
    PaperBuilder,
    PaperDocument,
    PaperFigure,
    PaperSection,
    PaperTable,
    paper_from_mapping,
    render_latex,
    render_markdown,
)


# ---------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------


def test_render_markdown_requires_document() -> None:
    with pytest.raises(PaperError):
        render_markdown("bad")


def test_render_markdown_contains_title() -> None:
    text = render_markdown(make_document())

    assert text.startswith("# ROIF Validation Study")
    assert "Theo Shapoval" in text


def test_render_markdown_contains_metadata() -> None:
    text = render_markdown(make_document())

    assert "**Journal:** Journal of Complex Systems" in text
    assert "**Status:** `draft`" in text
    assert "**Paper ID:** `paper-1`" in text
    assert "**Created:** `2026-08-01T09:00:00Z`" in text


def test_render_markdown_contains_abstract() -> None:
    text = render_markdown(make_document())

    assert "## Abstract" in text
    assert (
        "This study evaluates the ROIF framework"
        in text
    )


def test_render_markdown_contains_keywords() -> None:
    text = render_markdown(make_document())

    assert (
        "**Keywords:** ROIF, cascade, validation"
        in text
    )


def test_render_markdown_contains_sections() -> None:
    text = render_markdown(make_document())

    assert "## Introduction" in text
    assert "## Methods" in text
    assert "ROIF models cascading dynamics." in text


def test_render_markdown_contains_citation() -> None:
    text = render_markdown(make_document())

    assert "[@ref-1]" in text


def test_render_markdown_formats_extended_citation() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="introduction",
                title="Introduction",
                kind=SectionKind.INTRODUCTION,
                paragraphs=(
                    Paragraph(
                        text="Text.",
                        citations=(
                            Citation(
                                reference_ids=(
                                    "ref-1",
                                    "ref-2",
                                ),
                                locator="pp. 10-12",
                                prefix="see",
                                suffix="for comparison",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        references=(
            make_reference(),
            Reference(
                reference_id="ref-2",
                reference_type=ReferenceType.BOOK,
                title="Second Reference",
                authors=("Second Author",),
                year=2025,
            ),
        ),
    )

    text = render_markdown(document)

    assert (
        "[see @ref-1; @ref-2, pp. 10-12 "
        "for comparison]"
        in text
    )


def test_render_markdown_contains_result_data() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_result_section(
            "validation",
            {
                "accuracy": 0.95,
                "converged": True,
            },
        )
        .build()
    )

    text = render_markdown(document)

    assert "```json" in text
    assert '"accuracy": 0.95' in text
    assert '"converged": true' in text


def test_render_markdown_contains_table() -> None:
    text = render_markdown(make_document())

    assert "## Table 1. Validation Results" in text
    assert "| Method | Accuracy | Runtime |" in text
    assert "| :--- | ---: | ---: |" in text
    assert "| ROIF | 0.95 | 1.2 |" in text
    assert "*Note:* Higher accuracy is better." in text


def test_render_markdown_contains_figure() -> None:
    text = render_markdown(make_document())

    assert "## Figure 1. Cascade Dynamics" in text
    assert (
        "![Line plot of cascade coherence.]"
        "(figures/cascade.png)"
        in text
    )
    assert "Cascade coherence over time." in text


def test_render_markdown_uses_figure_title_as_alt_text() -> None:
    figure = PaperFigure(
        figure_id="figure",
        title="Network",
        kind=FigureKind.NETWORK,
        location="network.png",
        caption="Network figure.",
    )

    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(make_section(),),
        figures=(figure,),
        references=(make_reference(),),
    )

    text = render_markdown(document)

    assert "![Network](network.png)" in text


def test_render_markdown_contains_references() -> None:
    text = render_markdown(make_document())

    assert "## References" in text
    assert "**ref-1.**" in text
    assert "Theo Shapoval (2026)" in text
    assert (
        "*Recursive Organic Integration Framework*"
        in text
    )
    assert "DOI: `10.1000/example`" in text


def test_render_markdown_reference_without_authors_or_year() -> None:
    reference = Reference(
        reference_id="web",
        reference_type=ReferenceType.WEB,
        title="Online Resource",
        url="https://example.com",
    )

    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
        references=(reference,),
    )

    text = render_markdown(document)

    assert "Unknown author (n.d.)" in text
    assert "https://example.com" in text


def test_render_markdown_contains_reproducibility() -> None:
    text = render_markdown(make_document())

    assert "## Reproducibility" in text
    assert '"engine_version": "1.0.0"' in text
    assert '"seed": 42' in text


def test_document_to_markdown_matches_renderer() -> None:
    document = make_document()

    assert document.to_markdown() == render_markdown(
        document
    )


def test_render_markdown_ends_with_newline() -> None:
    assert render_markdown(make_document()).endswith("\n")


# ---------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------


def test_render_latex_requires_document() -> None:
    with pytest.raises(PaperError):
        render_latex("bad")


def test_render_latex_is_standalone_document() -> None:
    text = render_latex(make_document())

    assert text.startswith(
        r"\documentclass[11pt]{article}"
    )
    assert r"\begin{document}" in text
    assert text.rstrip().endswith(r"\end{document}")


def test_render_latex_contains_title_and_author() -> None:
    text = render_latex(make_document())

    assert (
        r"\title{ROIF Validation Study}"
        in text
    )
    assert r"\author{Theo Shapoval}" in text
    assert r"\date{2026-08-01}" in text


def test_render_latex_contains_abstract() -> None:
    text = render_latex(make_document())

    assert r"\begin{abstract}" in text
    assert (
        "This study evaluates the ROIF framework"
        in text
    )
    assert r"\end{abstract}" in text


def test_render_latex_contains_keywords() -> None:
    text = render_latex(make_document())

    assert (
        r"\noindent\textbf{Keywords:} "
        "ROIF, cascade, validation"
        in text
    )


def test_render_latex_contains_sections() -> None:
    text = render_latex(make_document())

    assert r"\section{Introduction}" in text
    assert r"\section{Methods}" in text


def test_render_latex_uses_subsection_levels() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="level-1",
                title="Level One",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("One."),),
                level=1,
            ),
            PaperSection(
                section_id="level-2",
                title="Level Two",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Two."),),
                level=2,
            ),
            PaperSection(
                section_id="level-3",
                title="Level Three",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Three."),),
                level=3,
            ),
        ),
    )

    text = render_latex(document)

    assert r"\section{Level One}" in text
    assert r"\subsection{Level Two}" in text
    assert r"\subsubsection{Level Three}" in text


def test_render_latex_contains_citation() -> None:
    text = render_latex(make_document())

    assert r"\cite{ref-1}" in text


def test_render_latex_combines_citation_keys() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(
                    Paragraph(
                        text="Text.",
                        citations=(
                            Citation(
                                reference_ids=(
                                    "ref-1",
                                    "ref-2",
                                )
                            ),
                        ),
                    ),
                ),
            ),
        ),
        references=(
            make_reference(),
            Reference(
                reference_id="ref-2",
                reference_type=ReferenceType.BOOK,
                title="Book",
            ),
        ),
    )

    text = render_latex(document)

    assert r"\cite{ref-1,ref-2}" in text


def test_render_latex_contains_result_data() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_result_section(
            "results",
            {"accuracy": 0.95},
        )
        .build()
    )

    text = render_latex(document)

    assert r"\begin{verbatim}" in text
    assert '"accuracy": 0.95' in text
    assert r"\end{verbatim}" in text


def test_render_latex_contains_table() -> None:
    text = render_latex(make_document())

    assert r"\begin{table}[htbp]" in text
    assert r"\begin{tabular}{lrr}" in text
    assert (
        r"Method & Accuracy & Runtime \\"
        in text
    )
    assert r"ROIF & 0.95 & 1.2 \\" in text
    assert r"\label{tab:table-1}" in text


def test_render_latex_contains_figure() -> None:
    text = render_latex(make_document())

    assert r"\begin{figure}[htbp]" in text
    assert (
        r"\includegraphics[width=0.800\linewidth]"
        r"{figures/cascade.png}"
        in text
    )
    assert (
        r"\caption{Cascade coherence over time.}"
        in text
    )
    assert r"\label{fig:figure-1}" in text


def test_render_latex_contains_bibliography() -> None:
    text = render_latex(make_document())

    assert r"\begin{thebibliography}{99}" in text
    assert r"\bibitem{ref-1}" in text
    assert (
        "Recursive Organic Integration Framework"
        in text
    )
    assert r"\end{thebibliography}" in text


def test_render_latex_escapes_special_characters() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="ROIF & Validation_Study",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=("A&B",),
        abstract="Accuracy = 95% for A_B.",
        sections=(
            PaperSection(
                section_id="section",
                title="Results & Discussion",
                kind=SectionKind.RESULTS,
                paragraphs=(
                    Paragraph("Value #1 costs $5."),
                ),
            ),
        ),
    )

    text = render_latex(document)

    assert r"ROIF \& Validation\_Study" in text
    assert r"95\%" in text
    assert r"A\_B" in text
    assert r"Results \& Discussion" in text
    assert r"Value \#1 costs \$5." in text


def test_document_to_latex_matches_renderer() -> None:
    document = make_document()

    assert document.to_latex() == render_latex(
        document
    )


# ---------------------------------------------------------------------
# paper_from_mapping
# ---------------------------------------------------------------------


def test_paper_from_mapping_round_trip() -> None:
    original = make_document()

    restored = paper_from_mapping(
        original.as_dict()
    )

    assert restored.as_dict() == original.as_dict()


def test_paper_from_mapping_preserves_author() -> None:
    restored = paper_from_mapping(
        make_document().as_dict()
    )

    assert restored.authors[0].author_id == "author-1"
    assert restored.authors[0].corresponding is True
    assert restored.authors[0].orcid == (
        "0000-0000-0000-0001"
    )


def test_paper_from_mapping_preserves_affiliation() -> None:
    restored = paper_from_mapping(
        make_document().as_dict()
    )

    assert (
        restored.affiliations[0].institution
        == "ROIF Research Center"
    )
    assert restored.affiliations[0].country == "Ukraine"


def test_paper_from_mapping_preserves_citations() -> None:
    restored = paper_from_mapping(
        make_document().as_dict()
    )

    citation = (
        restored
        .section("introduction")
        .paragraphs[0]
        .citations[0]
    )

    assert citation.reference_ids == ("ref-1",)


def test_paper_from_mapping_preserves_tables() -> None:
    restored = paper_from_mapping(
        make_document().as_dict()
    )

    assert restored.tables[0].columns == (
        "Method",
        "Accuracy",
        "Runtime",
    )
    assert restored.tables[0].alignments == (
        TableAlignment.LEFT,
        TableAlignment.RIGHT,
        TableAlignment.RIGHT,
    )


def test_paper_from_mapping_preserves_figures() -> None:
    restored = paper_from_mapping(
        make_document().as_dict()
    )

    assert restored.figures[0].kind is FigureKind.PLOT
    assert restored.figures[0].width_fraction == pytest.approx(
        0.8
    )


def test_paper_from_mapping_preserves_references() -> None:
    restored = paper_from_mapping(
        make_document().as_dict()
    )

    reference = restored.reference("ref-1")

    assert reference.reference_type is ReferenceType.ARTICLE
    assert reference.year == 2026
    assert reference.doi == "10.1000/example"


def test_paper_from_mapping_preserves_manifest() -> None:
    restored = paper_from_mapping(
        make_document().as_dict()
    )

    assert restored.reproducibility is not None
    assert restored.reproducibility.seed == 42
    assert restored.reproducibility.dataset_ids == (
        "dataset-1",
        "dataset-2",
    )


def test_paper_from_mapping_requires_mapping() -> None:
    with pytest.raises(PaperError):
        paper_from_mapping("bad")


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"paper_id": "paper"},
        {
            "paper_id": "paper",
            "title": "Paper",
            "status": "unknown",
            "created_at": "2026-08-01T09:00:00Z",
            "abstract": "Abstract.",
            "sections": [],
        },
    ),
)
def test_paper_from_mapping_rejects_invalid_payload(
    payload,
) -> None:
    with pytest.raises(PaperError):
        paper_from_mapping(payload)


def test_paper_from_mapping_rejects_unknown_reference_type() -> None:
    payload = make_document().as_dict()
    payload["references"][0]["reference_type"] = "unknown"

    with pytest.raises(PaperError):
        paper_from_mapping(payload)


def test_paper_from_mapping_rejects_unknown_section_kind() -> None:
    payload = make_document().as_dict()
    payload["sections"][0]["kind"] = "unknown"

    with pytest.raises(PaperError):
        paper_from_mapping(payload)


def test_paper_from_mapping_rejects_unknown_figure_kind() -> None:
    payload = make_document().as_dict()
    payload["figures"][0]["kind"] = "unknown"

    with pytest.raises(PaperError):
        paper_from_mapping(payload)


def test_paper_from_mapping_rejects_unknown_alignment() -> None:
    payload = make_document().as_dict()
    payload["tables"][0]["alignments"][0] = "unknown"

    with pytest.raises(PaperError):
        paper_from_mapping(payload)


# ---------------------------------------------------------------------
# Minimal and edge-case documents
# ---------------------------------------------------------------------


def test_minimal_document() -> None:
    document = PaperDocument(
        paper_id="minimal",
        title="Minimal Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="results",
                title="Results",
                kind=SectionKind.RESULTS,
                data={"value": 1},
            ),
        ),
    )

    assert document.authors == ()
    assert document.references == ()
    assert document.tables == ()
    assert document.figures == ()
    assert document.reproducibility is None


def test_document_normalizes_timezone() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T12:00:00+03:00",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(make_section(),),
        references=(make_reference(),),
    )

    assert document.created_at == "2026-08-01T09:00:00Z"


def test_document_accepts_empty_author_list() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(make_section(),),
        references=(make_reference(),),
    )

    assert document.authors == ()


def test_document_accepts_no_references_without_citations() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
        references=(),
    )

    assert document.references == ()


def test_document_accepts_all_paper_status_values() -> None:
    for status in PaperStatus:
        document = PaperDocument(
            paper_id=f"paper-{status.value}",
            title="Paper",
            status=status,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )

        assert document.status is status


# ---------------------------------------------------------------------
# Complete builder integration
# ---------------------------------------------------------------------


def test_complete_builder_workflow() -> None:
    document = (
        PaperBuilder(
            title=(
                "ROIF: Cascading Dynamics in "
                "Pre-Stressed Systems"
            ),
            paper_id="roif-study-2026",
            created_at="2026-08-01T09:00:00Z",
            status=PaperStatus.PREPRINT,
            language="en",
        )
        .add_affiliation(make_affiliation())
        .add_author(make_author())
        .set_abstract(
            "We evaluate cascade localization, "
            "counterfactual optimization, and coherence."
        )
        .set_keywords(
            (
                "active inference",
                "cascade dynamics",
                "pre-stress",
                "ROIF",
            )
        )
        .set_journal("Complex Systems Preprints")
        .set_reproducibility(make_manifest())
        .add_reference(make_reference())
        .add_text_section(
            "introduction",
            "Introduction",
            SectionKind.INTRODUCTION,
            "Cascades propagate through coupled systems.",
            citations=(make_citation(),),
            order=0,
        )
        .add_text_section(
            "methods",
            "Methods",
            SectionKind.METHODS,
            "The ROIF Engine was used for analysis.",
            order=1,
        )
        .add_result_section(
            "validation",
            {
                "root_accuracy": 0.95,
                "node_star_accuracy": 0.90,
            },
            title="Validation Results",
            summary=(
                "ROIF recovered the generating root "
                "with high accuracy."
            ),
            order=2,
        )
        .add_table(make_table())
        .add_figure(make_figure())
        .update_metadata(
            {
                "experiment": "cascade-001",
                "license": "research",
            }
        )
        .build()
    )

    assert document.status is PaperStatus.PREPRINT
    assert len(document.sections) == 3
    assert len(document.tables) == 1
    assert len(document.figures) == 1
    assert len(document.references) == 1
    assert document.metadata["experiment"] == "cascade-001"

    markdown = document.to_markdown()
    latex = document.to_latex()
    json_payload = json.loads(document.to_json())

    assert "# ROIF: Cascading Dynamics" in markdown
    assert r"\begin{document}" in latex
    assert json_payload["status"] == "preprint"


def test_complete_document_round_trip() -> None:
    original = make_document()
    serialized = original.to_json()
    restored = paper_from_mapping(
        json.loads(serialized)
    )

    assert restored.as_dict() == original.as_dict()
    assert restored.to_markdown() == original.to_markdown()
    assert restored.to_latex() == original.to_latex()


# ---------------------------------------------------------------------
# Final JSON serialization sanity
# ---------------------------------------------------------------------


def test_document_dictionary_is_json_serializable() -> None:
    json.dumps(
        make_document().as_dict(),
        ensure_ascii=False,
    )


def test_complete_markdown_is_nonempty() -> None:
    assert len(make_document().to_markdown()) > 100


def test_complete_latex_is_nonempty() -> None:
    assert len(make_document().to_latex()) > 100


def test_complete_json_is_nonempty() -> None:
    assert len(make_document().to_json()) > 100
# ---------------------------------------------------------------------
# Additional validation coverage
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "paper_id",
    (
        "",
        " ",
        None,
        "bad paper id",
        "/paper",
    ),
)
def test_document_rejects_bad_paper_id(paper_id) -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id=paper_id,
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )


@pytest.mark.parametrize(
    "title",
    (
        "",
        " ",
        None,
        42,
    ),
)
def test_document_rejects_bad_title(title) -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title=title,
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )


@pytest.mark.parametrize(
    "created_at",
    (
        "",
        "bad-date",
        1,
        None,
    ),
)
def test_document_rejects_bad_created_at(
    created_at,
) -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at=created_at,
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )


def test_document_rejects_bad_status() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status="draft",
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )


@pytest.mark.parametrize(
    "abstract",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_document_rejects_bad_abstract(
    abstract,
) -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract=abstract,
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )


@pytest.mark.parametrize(
    "language",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_document_rejects_bad_language(
    language,
) -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
            language=language,
        )


def test_document_rejects_bad_reproducibility() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
            reproducibility="bad",
        )


def test_document_rejects_non_author_values() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=("bad",),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )


def test_document_rejects_non_affiliation_values() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=("bad",),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
        )


def test_document_rejects_non_section_values() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=("bad",),
        )


def test_document_rejects_non_table_values() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
            tables=("bad",),
        )


def test_document_rejects_non_figure_values() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
            figures=("bad",),
        )


def test_document_rejects_non_reference_values() -> None:
    with pytest.raises(PaperError):
        PaperDocument(
            paper_id="paper",
            title="Paper",
            status=PaperStatus.DRAFT,
            created_at="2026-08-01T09:00:00Z",
            authors=(),
            affiliations=(),
            keywords=(),
            abstract="Abstract.",
            sections=(
                PaperSection(
                    section_id="section",
                    title="Section",
                    kind=SectionKind.CUSTOM,
                    paragraphs=(Paragraph("Text."),),
                ),
            ),
            references=("bad",),
        )


# ---------------------------------------------------------------------
# Additional Author validation
# ---------------------------------------------------------------------


def test_author_rejects_duplicate_affiliations() -> None:
    with pytest.raises(PaperError):
        Author(
            author_id="author",
            given_name="Theo",
            family_name="Shapoval",
            affiliations=("aff-1", "aff-1"),
        )


def test_author_rejects_string_affiliations() -> None:
    with pytest.raises(PaperError):
        Author(
            author_id="author",
            given_name="Theo",
            family_name="Shapoval",
            affiliations="aff-1",
        )


@pytest.mark.parametrize(
    "field",
    (
        "email",
        "orcid",
        "contribution",
    ),
)
def test_author_rejects_empty_optional_text(
    field,
) -> None:
    kwargs = {
        "author_id": "author",
        "given_name": "Theo",
        "family_name": "Shapoval",
        field: " ",
    }

    with pytest.raises(PaperError):
        Author(**kwargs)


# ---------------------------------------------------------------------
# Additional Affiliation validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    (
        "department",
        "city",
        "country",
    ),
)
def test_affiliation_rejects_empty_optional_text(
    field,
) -> None:
    kwargs = {
        "affiliation_id": "aff",
        "institution": "Institute",
        field: " ",
    }

    with pytest.raises(PaperError):
        Affiliation(**kwargs)


# ---------------------------------------------------------------------
# Additional Citation validation
# ---------------------------------------------------------------------


def test_citation_rejects_string_reference_ids() -> None:
    with pytest.raises(PaperError):
        Citation(reference_ids="ref-1")


@pytest.mark.parametrize(
    "reference_id",
    (
        "",
        " ",
        "bad reference",
        None,
    ),
)
def test_citation_rejects_bad_reference_identifier(
    reference_id,
) -> None:
    with pytest.raises(PaperError):
        Citation(reference_ids=(reference_id,))


@pytest.mark.parametrize(
    "field",
    (
        "locator",
        "prefix",
        "suffix",
    ),
)
def test_citation_rejects_empty_optional_text(
    field,
) -> None:
    kwargs = {
        "reference_ids": ("ref-1",),
        field: " ",
    }

    with pytest.raises(PaperError):
        Citation(**kwargs)


# ---------------------------------------------------------------------
# Additional Paragraph validation
# ---------------------------------------------------------------------


def test_paragraph_is_frozen() -> None:
    paragraph = Paragraph("Text.")

    with pytest.raises(FrozenInstanceError):
        paragraph.text = "Changed"


def test_paragraph_allows_no_citations() -> None:
    paragraph = Paragraph("Text.")

    assert paragraph.citations == ()
    assert paragraph.label is None


# ---------------------------------------------------------------------
# Additional PaperSection validation
# ---------------------------------------------------------------------


def test_section_rejects_bad_kind() -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id="section",
            title="Section",
            kind="custom",
            paragraphs=(Paragraph("Text."),),
        )


@pytest.mark.parametrize(
    "title",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_section_rejects_bad_title(title) -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id="section",
            title=title,
            kind=SectionKind.CUSTOM,
            paragraphs=(Paragraph("Text."),),
        )


def test_section_serializes_enum_data() -> None:
    section = PaperSection(
        section_id="data",
        title="Data",
        kind=SectionKind.RESULTS,
        data={
            "status": PaperStatus.DRAFT,
        },
    )

    assert section.data == {
        "status": "draft",
    }


def test_section_rejects_nonserializable_data() -> None:
    with pytest.raises(PaperError):
        PaperSection(
            section_id="data",
            title="Data",
            kind=SectionKind.RESULTS,
            data=object(),
        )


# ---------------------------------------------------------------------
# Additional PaperTable validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "table_id",
    (
        "",
        " ",
        "bad table",
        None,
    ),
)
def test_table_rejects_bad_identifier(
    table_id,
) -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id=table_id,
            title="Table",
            columns=("A",),
            rows=((1,),),
        )


@pytest.mark.parametrize(
    "title",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_table_rejects_bad_title(title) -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title=title,
            columns=("A",),
            rows=((1,),),
        )


def test_table_accepts_none_cells() -> None:
    table = PaperTable(
        table_id="table",
        title="Table",
        columns=("A", "B"),
        rows=((None, 1),),
    )

    assert table.rows == ((None, 1),)


def test_table_accepts_boolean_cells() -> None:
    table = PaperTable(
        table_id="table",
        title="Table",
        columns=("A",),
        rows=((True,),),
    )

    assert table.rows == ((True,),)


def test_table_rejects_string_row() -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A",),
            rows=("bad",),
        )


@pytest.mark.parametrize(
    "order",
    (
        -1,
        True,
        1.5,
        "0",
    ),
)
def test_table_rejects_bad_order(order) -> None:
    with pytest.raises(PaperError):
        PaperTable(
            table_id="table",
            title="Table",
            columns=("A",),
            rows=((1,),),
            order=order,
        )


# ---------------------------------------------------------------------
# Additional PaperFigure validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "figure_id",
    (
        "",
        " ",
        "bad figure",
        None,
    ),
)
def test_figure_rejects_bad_identifier(
    figure_id,
) -> None:
    with pytest.raises(PaperError):
        PaperFigure(
            figure_id=figure_id,
            title="Figure",
            kind=FigureKind.PLOT,
            location="figure.png",
            caption="Caption.",
        )


@pytest.mark.parametrize(
    "caption",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_figure_rejects_bad_caption(
    caption,
) -> None:
    with pytest.raises(PaperError):
        PaperFigure(
            figure_id="figure",
            title="Figure",
            kind=FigureKind.PLOT,
            location="figure.png",
            caption=caption,
        )


@pytest.mark.parametrize(
    "order",
    (
        -1,
        True,
        1.5,
        "0",
    ),
)
def test_figure_rejects_bad_order(order) -> None:
    with pytest.raises(PaperError):
        PaperFigure(
            figure_id="figure",
            title="Figure",
            kind=FigureKind.PLOT,
            location="figure.png",
            caption="Caption.",
            order=order,
        )


# ---------------------------------------------------------------------
# Additional Reference validation
# ---------------------------------------------------------------------


def test_reference_rejects_bad_type() -> None:
    with pytest.raises(PaperError):
        Reference(
            reference_id="ref",
            reference_type="article",
            title="Title",
        )


@pytest.mark.parametrize(
    "reference_id",
    (
        "",
        " ",
        "bad reference",
        None,
    ),
)
def test_reference_rejects_bad_identifier(
    reference_id,
) -> None:
    with pytest.raises(PaperError):
        Reference(
            reference_id=reference_id,
            reference_type=ReferenceType.ARTICLE,
            title="Title",
        )


@pytest.mark.parametrize(
    "title",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_reference_rejects_bad_title(title) -> None:
    with pytest.raises(PaperError):
        Reference(
            reference_id="ref",
            reference_type=ReferenceType.ARTICLE,
            title=title,
        )


@pytest.mark.parametrize(
    "field",
    (
        "container_title",
        "volume",
        "issue",
        "pages",
        "doi",
        "url",
        "publisher",
    ),
)
def test_reference_rejects_empty_optional_text(
    field,
) -> None:
    kwargs = {
        "reference_id": "ref",
        "reference_type": ReferenceType.ARTICLE,
        "title": "Title",
        field: " ",
    }

    with pytest.raises(PaperError):
        Reference(**kwargs)


# ---------------------------------------------------------------------
# Additional ReproducibilityManifest validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "version",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_manifest_rejects_bad_engine_version(
    version,
) -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version=version,
        )


@pytest.mark.parametrize(
    "version",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_manifest_rejects_bad_schema_version(
    version,
) -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version="1.0",
            schema_version=version,
        )


def test_manifest_rejects_string_dataset_ids() -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version="1.0",
            dataset_ids="dataset",
        )


def test_manifest_rejects_bad_checksums_mapping() -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version="1.0",
            artifact_checksums="bad",
        )


def test_manifest_rejects_bad_environment_mapping() -> None:
    with pytest.raises(PaperError):
        ReproducibilityManifest(
            engine_version="1.0",
            environment="bad",
        )


# ---------------------------------------------------------------------
# Builder metadata and ordering
# ---------------------------------------------------------------------


def test_builder_preserves_explicit_section_order() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_text_section(
            "second",
            "Second",
            SectionKind.RESULTS,
            "Second.",
            order=2,
        )
        .add_text_section(
            "first",
            "First",
            SectionKind.METHODS,
            "First.",
            order=1,
        )
        .build()
    )

    assert tuple(
        item.section_id
        for item in document.sections
    ) == (
        "first",
        "second",
    )


def test_builder_uses_automatic_section_order() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_text_section(
            "first",
            "First",
            SectionKind.INTRODUCTION,
            "First.",
        )
        .add_text_section(
            "second",
            "Second",
            SectionKind.RESULTS,
            "Second.",
        )
        .build()
    )

    assert document.section("first").order == 0
    assert document.section("second").order == 1


def test_builder_result_section_serializes_object() -> None:
    class ResultObject:
        def as_dict(self):
            return {
                "value": 5,
                "valid": True,
            }

    document = (
        make_builder()
        .set_abstract("Abstract.")
        .add_result_section(
            "result",
            ResultObject(),
        )
        .build()
    )

    assert document.section("result").data == {
        "value": 5,
        "valid": True,
    }


def test_builder_result_section_rejects_bad_result() -> None:
    with pytest.raises(PaperError):
        (
            make_builder()
            .set_abstract("Abstract.")
            .add_result_section(
                "result",
                object(),
            )
        )


def test_builder_update_metadata_accumulates_values() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .update_metadata({"a": 1})
        .update_metadata({"b": 2})
        .add_result_section(
            "results",
            {"value": 1},
        )
        .build()
    )

    assert document.metadata == {
        "a": 1,
        "b": 2,
    }


def test_builder_update_metadata_overwrites_existing_key() -> None:
    document = (
        make_builder()
        .set_abstract("Abstract.")
        .update_metadata({"value": 1})
        .update_metadata({"value": 2})
        .add_result_section(
            "results",
            {"value": 1},
        )
        .build()
    )

    assert document.metadata["value"] == 2


# ---------------------------------------------------------------------
# Final regression checks
# ---------------------------------------------------------------------


def test_markdown_render_is_deterministic() -> None:
    first = render_markdown(make_document())
    second = render_markdown(make_document())

    assert first == second


def test_latex_render_is_deterministic() -> None:
    first = render_latex(make_document())
    second = render_latex(make_document())

    assert first == second


def test_json_render_is_deterministic() -> None:
    first = make_document().to_json(sort_keys=True)
    second = make_document().to_json(sort_keys=True)

    assert first == second


def test_round_trip_preserves_word_count() -> None:
    original = make_document()
    restored = paper_from_mapping(
        original.as_dict()
    )

    assert restored.word_count == original.word_count


def test_round_trip_preserves_markdown() -> None:
    original = make_document()
    restored = paper_from_mapping(
        original.as_dict()
    )

    assert restored.to_markdown() == original.to_markdown()


def test_round_trip_preserves_latex() -> None:
    original = make_document()
    restored = paper_from_mapping(
        original.as_dict()
    )

    assert restored.to_latex() == original.to_latex()
# ---------------------------------------------------------------------
# Complete enum coverage
# ---------------------------------------------------------------------


def test_all_section_kind_values_are_stable() -> None:
    assert tuple(item.value for item in SectionKind) == (
        "abstract",
        "introduction",
        "methods",
        "dataset",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "acknowledgements",
        "funding",
        "conflicts",
        "data_availability",
        "code_availability",
        "ethics",
        "references",
        "appendix",
        "supplement",
        "custom",
    )


def test_all_figure_kind_values_are_stable() -> None:
    assert tuple(item.value for item in FigureKind) == (
        "plot",
        "diagram",
        "flowchart",
        "image",
        "heatmap",
        "network",
        "other",
    )


def test_all_reference_type_values_are_stable() -> None:
    assert tuple(item.value for item in ReferenceType) == (
        "article",
        "book",
        "chapter",
        "conference",
        "preprint",
        "dataset",
        "software",
        "web",
        "other",
    )


# ---------------------------------------------------------------------
# Metadata immutability
# ---------------------------------------------------------------------


def test_section_metadata_is_immutable() -> None:
    section = make_section()

    with pytest.raises(TypeError):
        section.metadata["new"] = 1


def test_table_metadata_is_immutable() -> None:
    table = make_table()

    with pytest.raises(TypeError):
        table.metadata["new"] = 1


def test_figure_metadata_is_immutable() -> None:
    figure = make_figure()

    with pytest.raises(TypeError):
        figure.metadata["new"] = 1


def test_reference_metadata_is_immutable() -> None:
    reference = make_reference()

    with pytest.raises(TypeError):
        reference.metadata["new"] = 1


def test_manifest_environment_is_immutable() -> None:
    manifest = make_manifest()

    with pytest.raises(TypeError):
        manifest.environment["python"] = "changed"


def test_manifest_checksums_are_immutable() -> None:
    manifest = make_manifest()

    with pytest.raises(TypeError):
        manifest.artifact_checksums["new"] = "checksum"


def test_document_metadata_is_immutable() -> None:
    document = make_document()

    with pytest.raises(TypeError):
        document.metadata["new"] = 1


# ---------------------------------------------------------------------
# Defensive copying
# ---------------------------------------------------------------------


def test_section_copies_metadata_mapping() -> None:
    metadata = {"value": 1}

    section = PaperSection(
        section_id="section",
        title="Section",
        kind=SectionKind.CUSTOM,
        paragraphs=(Paragraph("Text."),),
        metadata=metadata,
    )

    metadata["value"] = 2

    assert section.metadata["value"] == 1


def test_table_copies_metadata_mapping() -> None:
    metadata = {"value": 1}

    table = PaperTable(
        table_id="table",
        title="Table",
        columns=("A",),
        rows=((1,),),
        metadata=metadata,
    )

    metadata["value"] = 2

    assert table.metadata["value"] == 1


def test_figure_copies_metadata_mapping() -> None:
    metadata = {"value": 1}

    figure = PaperFigure(
        figure_id="figure",
        title="Figure",
        kind=FigureKind.PLOT,
        location="figure.png",
        caption="Caption.",
        metadata=metadata,
    )

    metadata["value"] = 2

    assert figure.metadata["value"] == 1


def test_reference_copies_metadata_mapping() -> None:
    metadata = {"value": 1}

    reference = Reference(
        reference_id="reference",
        reference_type=ReferenceType.ARTICLE,
        title="Reference",
        metadata=metadata,
    )

    metadata["value"] = 2

    assert reference.metadata["value"] == 1


def test_document_copies_metadata_mapping() -> None:
    metadata = {"value": 1}

    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
        metadata=metadata,
    )

    metadata["value"] = 2

    assert document.metadata["value"] == 1


# ---------------------------------------------------------------------
# Optional field behavior
# ---------------------------------------------------------------------


def test_author_optional_fields_default_to_none() -> None:
    author = Author(
        author_id="author",
        given_name="Theo",
        family_name="Shapoval",
    )

    assert author.affiliations == ()
    assert author.email is None
    assert author.orcid is None
    assert author.contribution is None
    assert author.corresponding is False


def test_affiliation_optional_fields_default_to_none() -> None:
    affiliation = Affiliation(
        affiliation_id="affiliation",
        institution="Institute",
    )

    assert affiliation.department is None
    assert affiliation.city is None
    assert affiliation.country is None


def test_reference_optional_fields_default_to_none() -> None:
    reference = Reference(
        reference_id="reference",
        reference_type=ReferenceType.OTHER,
        title="Reference",
    )

    assert reference.authors == ()
    assert reference.year is None
    assert reference.container_title is None
    assert reference.volume is None
    assert reference.issue is None
    assert reference.pages is None
    assert reference.doi is None
    assert reference.url is None
    assert reference.publisher is None
    assert reference.accessed_at is None


def test_manifest_optional_fields_default_to_none() -> None:
    manifest = ReproducibilityManifest(
        engine_version="1.0",
    )

    assert manifest.commit is None
    assert manifest.seed is None
    assert manifest.command is None
    assert manifest.dataset_ids == ()
    assert manifest.artifact_checksums == {}
    assert manifest.environment == {}


def test_document_optional_fields_default_to_none() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert document.tables == ()
    assert document.figures == ()
    assert document.references == ()
    assert document.reproducibility is None
    assert document.journal is None
    assert document.doi is None
    assert document.language == "en"


# ---------------------------------------------------------------------
# Builder fluent API
# ---------------------------------------------------------------------


def test_builder_set_status_returns_self() -> None:
    builder = make_builder()

    assert (
        builder.set_status(PaperStatus.PREPRINT)
        is builder
    )


def test_builder_set_abstract_returns_self() -> None:
    builder = make_builder()

    assert builder.set_abstract("Abstract.") is builder


def test_builder_set_keywords_returns_self() -> None:
    builder = make_builder()

    assert builder.set_keywords(("ROIF",)) is builder


def test_builder_set_journal_returns_self() -> None:
    builder = make_builder()

    assert builder.set_journal("Journal") is builder


def test_builder_set_doi_returns_self() -> None:
    builder = make_builder()

    assert builder.set_doi("10.1000/example") is builder


def test_builder_set_manifest_returns_self() -> None:
    builder = make_builder()

    assert (
        builder.set_reproducibility(make_manifest())
        is builder
    )


def test_builder_update_metadata_returns_self() -> None:
    builder = make_builder()

    assert builder.update_metadata({"value": 1}) is builder


def test_builder_add_author_returns_self() -> None:
    builder = make_builder()

    assert builder.add_author(make_author()) is builder


def test_builder_add_affiliation_returns_self() -> None:
    builder = make_builder()

    assert (
        builder.add_affiliation(make_affiliation())
        is builder
    )


def test_builder_add_section_returns_self() -> None:
    builder = make_builder()

    assert builder.add_section(make_section()) is builder


def test_builder_add_table_returns_self() -> None:
    builder = make_builder()

    assert builder.add_table(make_table()) is builder


def test_builder_add_figure_returns_self() -> None:
    builder = make_builder()

    assert builder.add_figure(make_figure()) is builder


def test_builder_add_reference_returns_self() -> None:
    builder = make_builder()

    assert builder.add_reference(make_reference()) is builder


# ---------------------------------------------------------------------
# Builder validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "keywords",
    (
        "ROIF",
        ("ROIF", "ROIF"),
        ("",),
        (1,),
    ),
)
def test_builder_rejects_bad_keywords(
    keywords,
) -> None:
    with pytest.raises(PaperError):
        make_builder().set_keywords(keywords)


@pytest.mark.parametrize(
    "abstract",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_builder_rejects_bad_abstract(
    abstract,
) -> None:
    with pytest.raises(PaperError):
        make_builder().set_abstract(abstract)


@pytest.mark.parametrize(
    "journal",
    (
        "",
        " ",
        1,
    ),
)
def test_builder_rejects_bad_journal(
    journal,
) -> None:
    with pytest.raises(PaperError):
        make_builder().set_journal(journal)


@pytest.mark.parametrize(
    "doi",
    (
        "",
        " ",
        1,
    ),
)
def test_builder_rejects_bad_doi(
    doi,
) -> None:
    with pytest.raises(PaperError):
        make_builder().set_doi(doi)


def test_builder_rejects_bad_metadata() -> None:
    with pytest.raises(PaperError):
        make_builder().update_metadata("bad")


@pytest.mark.parametrize(
    "title",
    (
        "",
        " ",
        None,
        1,
    ),
)
def test_builder_rejects_bad_title_at_construction(
    title,
) -> None:
    with pytest.raises(PaperError):
        PaperBuilder(
            title=title,
            paper_id="paper",
        )


@pytest.mark.parametrize(
    "paper_id",
    (
        "",
        " ",
        None,
        "bad paper id",
    ),
)
def test_builder_rejects_bad_paper_id_at_construction(
    paper_id,
) -> None:
    with pytest.raises(PaperError):
        PaperBuilder(
            title="Paper",
            paper_id=paper_id,
        )


def test_builder_rejects_bad_created_at() -> None:
    with pytest.raises(PaperError):
        PaperBuilder(
            title="Paper",
            paper_id="paper",
            created_at="bad-date",
        )


def test_builder_rejects_bad_language() -> None:
    with pytest.raises(PaperError):
        PaperBuilder(
            title="Paper",
            paper_id="paper",
            language=" ",
        )


def test_builder_rejects_bad_initial_status() -> None:
    with pytest.raises(PaperError):
        PaperBuilder(
            title="Paper",
            paper_id="paper",
            status="draft",
        )


# ---------------------------------------------------------------------
# Markdown without optional components
# ---------------------------------------------------------------------


def test_markdown_without_authors() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    text = render_markdown(document)

    assert "# Paper" in text
    assert "Theo Shapoval" not in text


def test_markdown_without_journal() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert "**Journal:**" not in render_markdown(
        document
    )


def test_markdown_without_keywords() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert "**Keywords:**" not in render_markdown(
        document
    )


def test_markdown_without_references() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert "## References" not in render_markdown(
        document
    )


def test_markdown_without_reproducibility() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert "## Reproducibility" not in render_markdown(
        document
    )


# ---------------------------------------------------------------------
# LaTeX without optional components
# ---------------------------------------------------------------------


def test_latex_without_authors() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert r"\author{}" in render_latex(document)


def test_latex_without_keywords() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert r"\textbf{Keywords:}" not in render_latex(
        document
    )


def test_latex_without_bibliography() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Abstract.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Text."),),
            ),
        ),
    )

    assert (
        r"\begin{thebibliography}"
        not in render_latex(document)
    )


# ---------------------------------------------------------------------
# Mapping reconstruction edge cases
# ---------------------------------------------------------------------


def test_paper_from_mapping_accepts_minimal_document() -> None:
    payload = {
        "paper_id": "minimal",
        "title": "Minimal",
        "status": "draft",
        "created_at": "2026-08-01T09:00:00Z",
        "authors": [],
        "affiliations": [],
        "keywords": [],
        "abstract": "Abstract.",
        "sections": [
            {
                "section_id": "results",
                "title": "Results",
                "kind": "results",
                "paragraphs": [],
                "data": {"value": 1},
                "order": 0,
                "level": 1,
                "metadata": {},
            }
        ],
        "tables": [],
        "figures": [],
        "references": [],
        "reproducibility": None,
        "journal": None,
        "doi": None,
        "language": "en",
        "metadata": {},
    }

    document = paper_from_mapping(payload)

    assert document.paper_id == "minimal"
    assert document.section("results").data == {
        "value": 1
    }


def test_paper_from_mapping_defaults_optional_collections() -> None:
    payload = {
        "paper_id": "paper",
        "title": "Paper",
        "status": "draft",
        "created_at": "2026-08-01T09:00:00Z",
        "abstract": "Abstract.",
        "sections": [
            {
                "section_id": "section",
                "title": "Section",
                "kind": "custom",
                "paragraphs": [
                    {
                        "text": "Text.",
                    }
                ],
            }
        ],
    }

    document = paper_from_mapping(payload)

    assert document.authors == ()
    assert document.affiliations == ()
    assert document.keywords == ()
    assert document.tables == ()
    assert document.figures == ()
    assert document.references == ()


def test_paper_from_mapping_defaults_language() -> None:
    payload = make_document().as_dict()
    del payload["language"]

    document = paper_from_mapping(payload)

    assert document.language == "en"


def test_paper_from_mapping_defaults_section_order_and_level() -> None:
    payload = make_document().as_dict()
    del payload["sections"][0]["order"]
    del payload["sections"][0]["level"]

    document = paper_from_mapping(payload)
    section = document.section("introduction")

    assert section.order == 0
    assert section.level == 1


def test_paper_from_mapping_rejects_unknown_author_affiliation() -> None:
    payload = make_document().as_dict()
    payload["authors"][0]["affiliations"] = [
        "missing"
    ]

    with pytest.raises(PaperError):
        paper_from_mapping(payload)


def test_paper_from_mapping_rejects_unknown_citation() -> None:
    payload = make_document().as_dict()
    payload["sections"][0]["paragraphs"][0][
        "citations"
    ][0]["reference_ids"] = ["missing"]

    with pytest.raises(PaperError):
        paper_from_mapping(payload)


# ---------------------------------------------------------------------
# Word count regression
# ---------------------------------------------------------------------


def test_word_count_ignores_section_titles() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Very Long Paper Title",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="One.",
        sections=(
            PaperSection(
                section_id="section",
                title="Very Long Section Title",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Two three."),),
            ),
        ),
    )

    assert document.word_count == 3


def test_word_count_ignores_table_cells() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="One.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Two."),),
            ),
        ),
        tables=(
            PaperTable(
                table_id="table",
                title="Table",
                columns=("A",),
                rows=(("Many words in cell",),),
            ),
        ),
    )

    assert document.word_count == 2


def test_word_count_counts_hyphenated_expression() -> None:
    document = PaperDocument(
        paper_id="paper",
        title="Paper",
        status=PaperStatus.DRAFT,
        created_at="2026-08-01T09:00:00Z",
        authors=(),
        affiliations=(),
        keywords=(),
        abstract="Pre-stressed systems.",
        sections=(
            PaperSection(
                section_id="section",
                title="Section",
                kind=SectionKind.CUSTOM,
                paragraphs=(Paragraph("Counter-factual control."),),
            ),
        ),
    )

    assert document.word_count == 4


# ---------------------------------------------------------------------
# Final public API smoke tests
# ---------------------------------------------------------------------


def test_public_objects_have_as_dict() -> None:
    objects = (
        make_author(),
        make_affiliation(),
        make_citation(),
        Paragraph("Text."),
        make_section(),
        make_table(),
        make_figure(),
        make_reference(),
        make_manifest(),
        make_document(),
    )

    for item in objects:
        assert callable(item.as_dict)
        assert isinstance(item.as_dict(), dict)


def test_document_exports_all_supported_formats() -> None:
    document = make_document()

    assert isinstance(document.to_json(), str)
    assert isinstance(document.to_markdown(), str)
    assert isinstance(document.to_latex(), str)


def test_renderers_do_not_mutate_document() -> None:
    document = make_document()
    before = document.as_dict()

    render_markdown(document)
    render_latex(document)
    document.to_json()

    assert document.as_dict() == before


def test_paper_module_complete_regression() -> None:
    document = make_document()

    restored = paper_from_mapping(
        json.loads(document.to_json())
    )

    assert restored.paper_id == document.paper_id
    assert restored.word_count == document.word_count
    assert restored.as_dict() == document.as_dict()
    assert restored.to_markdown() == document.to_markdown()
    assert restored.to_latex() == document.to_latex()

# End of tests/test_paper.py



