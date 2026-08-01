"""Tests for the public roif package interface."""

from __future__ import annotations

import inspect

import roif
from roif import (
    APIError,
    Dataset,
    DatasetRecord,
    EngineRunStatus,
    EngineStage,
    PaperBuilder,
    PaperDocument,
    PaperStatus,
    ROIF,
    ROIFConfig,
    ROIFEngine,
    SerializationEnvelope,
    SerializationFormat,
    create_roif,
    run_roif,
    version_info,
)


def test_version_metadata() -> None:
    assert roif.__title__ == "ROIF Engine"
    assert roif.__version__ == "1.0.0"
    assert "cascade analysis" in roif.__description__


def test_version_info() -> None:
    assert version_info() == {
        "title": "ROIF Engine",
        "version": "1.0.0",
        "description": (
            "Recursive Organic Integration Framework engine "
            "for cascade analysis."
        ),
    }


def test_expected_public_symbols_are_available() -> None:
    expected = {
        "ROIF",
        "ROIFConfig",
        "ROIFEngine",
        "EngineStage",
        "EngineRunStatus",
        "Dataset",
        "DatasetRecord",
        "PaperBuilder",
        "PaperDocument",
        "PaperStatus",
        "SerializationEnvelope",
        "SerializationFormat",
        "create_roif",
        "run_roif",
    }
    assert expected <= set(roif.__all__)


def test_all_symbols_exist() -> None:
    assert [
        name for name in roif.__all__
        if not hasattr(roif, name)
    ] == []


def test_all_contains_no_duplicates() -> None:
    assert len(roif.__all__) == len(set(roif.__all__))


def test_public_classes_have_expected_modules() -> None:
    assert ROIF.__module__ == "roif.api"
    assert ROIFConfig.__module__ == "roif.api"
    assert ROIFEngine.__module__ == "roif.roif_engine"
    assert Dataset.__module__ == "roif.datasets"
    assert PaperDocument.__module__ == "roif.paper"


def test_public_functions_are_callable() -> None:
    assert callable(create_roif)
    assert callable(run_roif)
    assert callable(version_info)


def test_public_exceptions_are_exception_types() -> None:
    assert issubclass(APIError, RuntimeError)
    assert issubclass(roif.DatasetError, ValueError)
    assert issubclass(roif.PaperError, ValueError)
    assert issubclass(roif.SerializationError, ValueError)


def test_public_enum_values() -> None:
    assert EngineStage.ANALYSIS.value == "analysis"
    assert EngineRunStatus.COMPLETED.value == "completed"
    assert PaperStatus.DRAFT.value == "draft"
    assert SerializationFormat.JSON.value == "json"


def test_create_roif_from_root_import() -> None:
    api = create_roif(
        config=ROIFConfig(
            stage_order=(
                EngineStage.ANALYSIS,
                EngineStage.LOCALIZATION,
                EngineStage.OPTIMIZATION,
            ),
            required_stages=(
                EngineStage.ANALYSIS,
                EngineStage.LOCALIZATION,
                EngineStage.OPTIMIZATION,
            ),
        ),
        stages={
            EngineStage.ANALYSIS: lambda ctx: {"ok": True},
            EngineStage.LOCALIZATION: lambda ctx: {"d_root": "A"},
            EngineStage.OPTIMIZATION: lambda ctx: {"node_star": "B"},
        },
    )
    assert isinstance(api, ROIF)


def test_dataset_types_from_root_import() -> None:
    item = Dataset(
        name="demo",
        records=(
            DatasetRecord(
                record_id="r1",
                features={"x": 1},
            ),
        ),
    )
    assert len(item) == 1
    assert item[0].record_id == "r1"


def test_paper_types_from_root_import() -> None:
    paper = (
        PaperBuilder(
            title="Paper",
            paper_id="paper-1",
            created_at="2026-08-01T09:00:00Z",
        )
        .set_abstract("Abstract.")
        .add_result_section(
            "results",
            {"value": 1},
        )
        .build()
    )
    assert isinstance(paper, PaperDocument)


def test_version_info_has_no_parameters() -> None:
    assert len(inspect.signature(version_info).parameters) == 0
