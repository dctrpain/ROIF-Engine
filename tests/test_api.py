"""Tests for roif.api."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any
import json

import pytest

from roif.api import (
    APIError,
    EngineRunStatus,
    EngineStage,
    EngineState,
    ROIF,
    ROIFConfig,
    create_roif,
    run_roif,
)
from roif.datasets import (
    Dataset,
    DatasetKind,
    DatasetRecord,
)
from roif.paper import (
    PaperBuilder,
    SectionKind,
)
from roif.roif_engine import ROIFEngine
from roif.serialization import (
    envelope_from_mapping,
    verify_envelope,
)


def stages() -> dict[EngineStage, Any]:
    return {
        EngineStage.ANALYSIS: (
            lambda ctx: {
                "score": 1.0,
                "input": ctx.input_data,
            }
        ),
        EngineStage.LOCALIZATION: (
            lambda ctx: {
                "d_root": "A",
                "analysis": ctx.require(
                    EngineStage.ANALYSIS
                ),
            }
        ),
        EngineStage.OPTIMIZATION: (
            lambda ctx: {
                "node_star": "B",
            }
        ),
        EngineStage.EXECUTION: (
            lambda ctx: {
                "executed": True,
            }
        ),
    }


def config(**kwargs: Any) -> ROIFConfig:
    defaults = {
        "stage_order": (
            EngineStage.ANALYSIS,
            EngineStage.LOCALIZATION,
            EngineStage.OPTIMIZATION,
            EngineStage.EXECUTION,
            EngineStage.REPORTING,
        ),
        "required_stages": (
            EngineStage.ANALYSIS,
            EngineStage.LOCALIZATION,
            EngineStage.OPTIMIZATION,
        ),
    }
    defaults.update(kwargs)
    return ROIFConfig(**defaults)


def dataset() -> Dataset:
    return Dataset(
        name="demo",
        kind=DatasetKind.SYNTHETIC,
        records=(
            DatasetRecord(
                record_id="r1",
                features={"x": 1},
                target=0,
            ),
            DatasetRecord(
                record_id="r2",
                features={"x": 2},
                target=1,
            ),
        ),
    )


def paper():
    return (
        PaperBuilder(
            title="Paper",
            paper_id="paper-1",
            created_at="2026-08-01T09:00:00Z",
        )
        .set_abstract("Abstract.")
        .add_text_section(
            "results",
            "Results",
            SectionKind.RESULTS,
            "Results text.",
        )
        .build()
    )


def test_config_defaults() -> None:
    item = ROIFConfig()
    assert item.seed == 0
    assert item.engine_version == "1.0.0"
    assert item.build_report is True
    assert isinstance(item.environment, MappingProxyType)


def test_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        ROIFConfig().seed = 1  # type: ignore[misc]


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_config_rejects_bad_seed(seed: Any) -> None:
    with pytest.raises(APIError):
        ROIFConfig(seed=seed)


def test_config_rejects_duplicate_stage_order() -> None:
    with pytest.raises(APIError):
        ROIFConfig(
            stage_order=(
                EngineStage.ANALYSIS,
                EngineStage.ANALYSIS,
            )
        )


def test_config_rejects_missing_required_stage() -> None:
    with pytest.raises(APIError):
        ROIFConfig(
            stage_order=(EngineStage.ANALYSIS,),
            required_stages=(EngineStage.LOCALIZATION,),
        )


def test_config_to_engine_config() -> None:
    item = config(seed=42)
    internal = item.to_engine_config()

    assert internal.seed == 42
    assert internal.stage_order == item.stage_order
    assert internal.report_title == item.report_title


def test_config_as_dict() -> None:
    payload = config().as_dict()

    assert payload["required_stages"] == [
        "analysis",
        "localization",
        "optimization",
    ]
    assert payload["stage_order"][-1] == "reporting"


def test_roif_properties() -> None:
    api = ROIF(config(), stages=stages())

    assert api.config.seed == 0
    assert isinstance(api.engine, ROIFEngine)
    assert api.state is EngineState.READY
    assert api.last_result is None


def test_roif_rejects_bad_config() -> None:
    with pytest.raises(APIError):
        ROIF(config="bad")  # type: ignore[arg-type]


def test_roif_rejects_bad_engine() -> None:
    with pytest.raises(APIError):
        ROIF(engine="bad")  # type: ignore[arg-type]


def test_roif_rejects_stages_with_existing_engine() -> None:
    with pytest.raises(APIError):
        ROIF(
            config(),
            engine=ROIFEngine(config().to_engine_config()),
            stages=stages(),
        )


def test_register_unregister() -> None:
    api = ROIF(config())
    callback = lambda ctx: {}

    assert api.register(
        EngineStage.ANALYSIS,
        callback,
    ) is api
    assert api.engine.has_stage(EngineStage.ANALYSIS)

    assert api.unregister(EngineStage.ANALYSIS) is api
    assert not api.engine.has_stage(EngineStage.ANALYSIS)


def test_register_rejects_bad_stage() -> None:
    with pytest.raises(APIError):
        ROIF(config()).register(
            "analysis",  # type: ignore[arg-type]
            lambda ctx: {},
        )


def test_register_rejects_bad_callback() -> None:
    with pytest.raises(APIError):
        ROIF(config()).register(
            EngineStage.ANALYSIS,
            None,  # type: ignore[arg-type]
        )


def test_run() -> None:
    api = ROIF(config(), stages=stages())

    result = api.run(
        {"value": 1},
        run_id="run-1",
        metadata={"experiment": "E1"},
    )

    assert result.status is EngineRunStatus.COMPLETED
    assert result.results["localization"]["d_root"] == "A"
    assert result.results["optimization"]["node_star"] == "B"
    assert result.metadata["experiment"] == "E1"
    assert api.last_result is result


def test_run_dry_run() -> None:
    result = ROIF(
        config(),
        stages=stages(),
    ).run(
        None,
        dry_run=True,
    )

    assert result.status is EngineRunStatus.DRY_RUN
    assert "execution" not in result.results


def test_run_returns_failed_result_when_required_stages_are_missing() -> None:
    api = ROIF(config())

    result = api.run(None)

    assert result.status is EngineRunStatus.FAILED


def test_run_dataset() -> None:
    api = ROIF(config(), stages=stages())

    results = api.run_dataset(
        dataset(),
        run_id_prefix="case",
    )

    assert len(results) == 2
    assert results[0].run_id == "case-0"
    assert results[1].run_id == "case-1"
    assert results[0].metadata["record_id"] == "r1"
    assert results[1].metadata["record_id"] == "r2"


def test_run_dataset_passes_record_payload() -> None:
    api = ROIF(config(), stages=stages())

    result = api.run_dataset(dataset())[0]
    payload = result.results["analysis"]["input"]

    assert payload["record_id"] == "r1"
    assert payload["features"] == {"x": 1}
    assert payload["target"] == 0


def test_run_dataset_rejects_bad_dataset() -> None:
    with pytest.raises(APIError):
        ROIF(config(), stages=stages()).run_dataset(
            "bad"  # type: ignore[arg-type]
        )


def test_reset() -> None:
    api = ROIF(config(), stages=stages())
    api.run(None)

    assert api.reset() is api
    assert api.state is EngineState.READY
    assert api.last_result is None


def test_result_json() -> None:
    api = ROIF(config(), stages=stages())
    result = api.run(None)

    payload = json.loads(api.result_json(result))

    assert payload["status"] == "completed"
    assert payload["results"]["optimization"][
        "node_star"
    ] == "B"


def test_result_json_uses_last_result() -> None:
    api = ROIF(config(), stages=stages())
    api.run(None)

    assert json.loads(api.result_json())[
        "status"
    ] == "completed"


def test_result_json_requires_result() -> None:
    with pytest.raises(APIError):
        ROIF(config()).result_json()


def test_result_markdown() -> None:
    api = ROIF(config(), stages=stages())
    api.run(None)

    text = api.result_markdown()

    assert "# ROIF Engine Report" in text
    assert "## Localization" in text


def test_save_result(tmp_path: Path) -> None:
    api = ROIF(config(), stages=stages())
    result = api.run(None)
    path = tmp_path / "result.json"

    saved = api.save_result(result, path)

    assert saved == path
    assert json.loads(
        path.read_text(encoding="utf-8")
    )["status"] == "completed"


def test_save_result_envelope(tmp_path: Path) -> None:
    api = ROIF(config(), stages=stages())
    result = api.run(None)
    path = tmp_path / "result.json"

    api.save_result(
        result,
        path,
        envelope=True,
    )

    envelope = envelope_from_mapping(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert envelope.object_type == "EngineResult"
    assert verify_envelope(envelope)


def test_save_last_result(tmp_path: Path) -> None:
    api = ROIF(config(), stages=stages())
    api.run(None)

    path = api.save_last_result(
        tmp_path / "last.json"
    )

    assert path.is_file()


def test_save_last_result_requires_run(tmp_path: Path) -> None:
    with pytest.raises(APIError):
        ROIF(config()).save_last_result(
            tmp_path / "missing.json"
        )


def test_dataset_helpers(tmp_path: Path) -> None:
    original = dataset()
    path = tmp_path / "dataset.json"

    assert ROIF.save_dataset(original, path) == path
    restored = ROIF.load_dataset(path)

    assert restored.as_dict() == original.as_dict()


def test_split_dataset_helper() -> None:
    split = ROIF.split_dataset(
        dataset(),
        train_ratio=0.5,
        validation_ratio=0.0,
        test_ratio=0.5,
        seed=1,
    )

    assert split.total_count == 2
    assert len(split.train) == 1
    assert len(split.test) == 1


def test_paper_helpers(tmp_path: Path) -> None:
    original = paper()
    path = tmp_path / "paper.json"

    assert ROIF.save_paper(original, path) == path
    restored = ROIF.load_paper(path)

    assert restored.as_dict() == original.as_dict()
    assert "# Paper" in ROIF.paper_markdown(restored)
    assert r"\begin{document}" in ROIF.paper_latex(
        restored
    )


def test_paper_helpers_reject_bad_paper() -> None:
    with pytest.raises(APIError):
        ROIF.paper_markdown("bad")  # type: ignore[arg-type]

    with pytest.raises(APIError):
        ROIF.paper_latex("bad")  # type: ignore[arg-type]


def test_create_roif() -> None:
    api = create_roif(
        config=config(),
        stages=stages(),
    )

    assert isinstance(api, ROIF)
    assert api.engine.has_stage(EngineStage.ANALYSIS)


def test_run_roif() -> None:
    result = run_roif(
        {"value": 1},
        config=config(),
        stages=stages(),
        run_id="single",
    )

    assert result.run_id == "single"
    assert result.status is EngineRunStatus.COMPLETED


def test_existing_engine_is_preserved() -> None:
    engine = ROIFEngine(config().to_engine_config())
    api = ROIF(config(), engine=engine)

    assert api.engine is engine

