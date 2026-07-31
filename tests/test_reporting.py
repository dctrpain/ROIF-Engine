"""Tests for roif.reporting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType
from typing import Any
import json

import numpy as np
import pytest

from roif.reporting import (
    ArtifactKind,
    ROIFReport,
    ROIFReportBuilder,
    ReportArtifact,
    ReportSection,
    ReportSectionKind,
    ReportStatus,
    ReportWarning,
    ReportingError,
    ReproducibilityInfo,
    WarningSeverity,
    flatten_report,
    render_json,
    render_markdown,
    to_json_compatible,
    utc_now_iso,
)


@dataclass
class PlainResult:
    value: float


class DictResult:
    def as_dict(self):
        return {"value": 2.0, "ok": True}


def section(
    section_id: str = "metrics",
    *,
    order: int = 0,
) -> ReportSection:
    return ReportSection(
        section_id=section_id,
        title="Metrics",
        kind=ReportSectionKind.METRICS,
        content={"eta": 0.5},
        order=order,
        summary="Metric summary.",
        metadata={"source": "test"},
    )


def warning() -> ReportWarning:
    return ReportWarning(
        code="LOW_CONFIDENCE",
        message="Confidence is limited.",
        severity=WarningSeverity.WARNING,
        source="uncertainty",
    )


def artifact() -> ReportArtifact:
    return ReportArtifact(
        name="results",
        kind=ArtifactKind.JSON,
        location="results.json",
        description="Machine-readable results.",
        checksum="abc123",
    )


def reproducibility() -> ReproducibilityInfo:
    return ReproducibilityInfo(
        engine_version="0.9.0",
        schema_version="1.0",
        commit="abc123",
        seed=42,
        command="python run.py",
        environment={"python": "3.13"},
    )


def report() -> ROIFReport:
    return ROIFReport(
        report_id="run-1",
        title="ROIF Report",
        status=ReportStatus.COMPLETED,
        created_at="2026-07-31T20:00:00Z",
        summary="Run completed successfully.",
        sections=(section(),),
        warnings=(warning(),),
        artifacts=(artifact(),),
        reproducibility=reproducibility(),
        duration_seconds=1.25,
        metadata={"experiment": "E1"},
    )


def test_enum_values() -> None:
    assert tuple(x.value for x in ReportStatus) == (
        "completed", "partial", "failed"
    )
    assert tuple(x.value for x in WarningSeverity) == (
        "info", "warning", "critical"
    )
    assert tuple(x.value for x in ArtifactKind) == (
        "json", "csv", "markdown", "image",
        "log", "dataset", "other",
    )


def test_to_json_compatible() -> None:
    result = to_json_compatible({
        "enum": ReportStatus.COMPLETED,
        "tuple": (1, 2),
        "array": np.asarray([3.0, 4.0]),
        "dataclass": PlainResult(5.0),
        "custom": DictResult(),
    })

    assert result == {
        "enum": "completed",
        "tuple": [1, 2],
        "array": [3.0, 4.0],
        "dataclass": {"value": 5.0},
        "custom": {"value": 2.0, "ok": True},
    }


@pytest.mark.parametrize(
    "value", (float("nan"), float("inf"), object())
)
def test_to_json_compatible_rejects_bad_values(value: Any) -> None:
    with pytest.raises(ReportingError):
        to_json_compatible(value)


def test_warning_properties() -> None:
    item = warning()
    assert item.code == "LOW_CONFIDENCE"
    assert item.as_dict()["severity"] == "warning"
    assert isinstance(item.metadata, MappingProxyType)


@pytest.mark.parametrize("value", ("", "   ", None, 1))
def test_warning_rejects_bad_code(value: Any) -> None:
    with pytest.raises(ReportingError):
        ReportWarning(value, "message")


def test_artifact_properties() -> None:
    item = artifact()
    assert item.kind is ArtifactKind.JSON
    assert item.as_dict()["location"] == "results.json"


def test_reproducibility_properties() -> None:
    item = reproducibility()
    assert item.seed == 42
    assert item.as_dict()["environment"]["python"] == "3.13"


@pytest.mark.parametrize("value", (True, 1.5, "42"))
def test_reproducibility_rejects_bad_seed(value: Any) -> None:
    with pytest.raises(ReportingError):
        ReproducibilityInfo("0.9", seed=value)


def test_section_properties() -> None:
    item = section()
    assert item.content == {"eta": 0.5}
    assert item.as_dict()["kind"] == "metrics"
    assert isinstance(item.metadata, MappingProxyType)


def test_section_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        section().title = "X"  # type: ignore[misc]


@pytest.mark.parametrize("value", (-1, True, 1.5, "0"))
def test_section_rejects_bad_order(value: Any) -> None:
    with pytest.raises(ReportingError):
        section(order=value)


def test_report_properties() -> None:
    item = report()
    assert item.created_at == "2026-07-31T20:00:00Z"
    assert item.critical_warning_count == 0
    assert item.section("metrics").title == "Metrics"
    assert item.as_dict()["status"] == "completed"
    assert isinstance(item.metadata, MappingProxyType)


def test_report_sorts_sections() -> None:
    item = ROIFReport(
        report_id="r",
        title="R",
        status=ReportStatus.COMPLETED,
        created_at="2026-01-01T00:00:00Z",
        summary="Done.",
        sections=(
            section("b", order=2),
            section("a", order=1),
        ),
    )
    assert tuple(x.section_id for x in item.sections) == ("a", "b")


def test_report_rejects_duplicate_section_ids() -> None:
    with pytest.raises(ReportingError):
        ROIFReport(
            report_id="r",
            title="R",
            status=ReportStatus.COMPLETED,
            created_at="2026-01-01T00:00:00Z",
            summary="Done.",
            sections=(section("x"), section("x")),
        )


def test_report_section_lookup_failure() -> None:
    with pytest.raises(KeyError):
        report().section("missing")


@pytest.mark.parametrize(
    "value", (-0.1, float("nan"), True, "bad")
)
def test_report_rejects_bad_duration(value: Any) -> None:
    with pytest.raises(ReportingError):
        ROIFReport(
            report_id="r",
            title="R",
            status=ReportStatus.COMPLETED,
            created_at="2026-01-01T00:00:00Z",
            summary="Done.",
            sections=(),
            duration_seconds=value,
        )


def test_builder_builds_report() -> None:
    built = (
        ROIFReportBuilder(
            "run-2",
            title="ROIF",
            created_at="2026-07-31T20:00:00Z",
        )
        .set_summary("Complete.")
        .set_status(ReportStatus.PARTIAL)
        .set_duration(2.5)
        .set_reproducibility(reproducibility())
        .update_metadata({"experiment": "E2"})
        .add_result(
            "localization",
            DictResult(),
            kind=ReportSectionKind.LOCALIZATION,
        )
        .add_warning(warning())
        .add_artifact(artifact())
        .build()
    )

    assert built.status is ReportStatus.PARTIAL
    assert built.duration_seconds == pytest.approx(2.5)
    assert built.section("localization").content["value"] == 2.0
    assert built.metadata["experiment"] == "E2"


def test_builder_requires_summary() -> None:
    with pytest.raises(ReportingError):
        ROIFReportBuilder("r").build()


def test_builder_rejects_duplicate_section() -> None:
    builder = ROIFReportBuilder("r").set_summary("Done.")
    builder.add_section(
        "x",
        "X",
        ReportSectionKind.CUSTOM,
        {"a": 1},
    )
    with pytest.raises(ReportingError):
        builder.add_section(
            "x",
            "X2",
            ReportSectionKind.CUSTOM,
            {"a": 2},
        )


def test_render_json() -> None:
    text = render_json(report(), sort_keys=True)
    payload = json.loads(text)

    assert payload["report_id"] == "run-1"
    assert payload["sections"][0]["content"]["eta"] == 0.5


@pytest.mark.parametrize("value", (-1, True, 1.5, "2"))
def test_render_json_rejects_bad_indent(value: Any) -> None:
    with pytest.raises(ReportingError):
        render_json(report(), indent=value)


def test_render_markdown() -> None:
    text = render_markdown(report())

    assert "# ROIF Report" in text
    assert "## Summary" in text
    assert "## Metrics" in text
    assert "LOW_CONFIDENCE" in text
    assert "## Reproducibility" in text


def test_flatten_report() -> None:
    item = ROIFReport(
        report_id="r",
        title="R",
        status=ReportStatus.COMPLETED,
        created_at="2026-01-01T00:00:00Z",
        summary="Done.",
        sections=(
            ReportSection(
                "nested",
                "Nested",
                ReportSectionKind.CUSTOM,
                {"a": {"b": [1, 2]}},
            ),
        ),
    )

    rows = flatten_report(item)
    assert len(rows) == 2
    assert rows[0]["path"] == "a.b.0"
    assert rows[1]["value"] == 2
    assert isinstance(rows[0], MappingProxyType)


def test_utc_now_iso() -> None:
    value = utc_now_iso()
    assert value.endswith("Z")
    assert "T" in value
