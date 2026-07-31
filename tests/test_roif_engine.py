"""Tests for roif.roif_engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any
import json

import pytest

from roif.roif_engine import (
    DEFAULT_STAGE_ORDER,
    EngineConfig,
    EngineConfigurationError,
    EngineContext,
    EngineError,
    EngineResult,
    EngineRunStatus,
    EngineStage,
    EngineState,
    ROIFEngine,
    StageResult,
    StageStatus,
)
from roif.reporting import ROIFReport


class Clock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        self.current += 0.1
        return self.current


def config(**kwargs: Any) -> EngineConfig:
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
    return EngineConfig(**defaults)


def configured_engine(
    *,
    engine_config: EngineConfig | None = None,
) -> ROIFEngine:
    engine = ROIFEngine(
        engine_config or config(),
        clock=Clock(),
    )
    engine.register_stage(
        EngineStage.ANALYSIS,
        lambda ctx: {"score": 1.0, "seed": ctx.config.seed},
    )
    engine.register_stage(
        EngineStage.LOCALIZATION,
        lambda ctx: {
            "d_root": "A",
            "analysis": ctx.require(EngineStage.ANALYSIS),
        },
    )
    engine.register_stage(
        EngineStage.OPTIMIZATION,
        lambda ctx: {
            "node_star": "B",
            "root": ctx.require(EngineStage.LOCALIZATION)["d_root"],
        },
    )
    engine.register_stage(
        EngineStage.EXECUTION,
        lambda ctx: {"executed": True},
    )
    return engine


def test_enum_values() -> None:
    assert tuple(item.value for item in EngineState) == (
        "initialized", "ready", "running", "finished", "failed"
    )
    assert tuple(item.value for item in EngineStage) == (
        "analysis",
        "localization",
        "optimization",
        "prediction",
        "planning",
        "execution",
        "uncertainty",
        "validation",
        "benchmark",
        "reporting",
    )
    assert tuple(item.value for item in StageStatus) == (
        "succeeded", "skipped", "failed"
    )
    assert tuple(item.value for item in EngineRunStatus) == (
        "completed", "partial", "failed", "dry_run"
    )
    assert DEFAULT_STAGE_ORDER[-1] is EngineStage.REPORTING


def test_config_defaults() -> None:
    item = EngineConfig()
    assert item.engine_version == "1.0.0"
    assert item.seed == 0
    assert item.build_report is True
    assert isinstance(item.environment, MappingProxyType)


def test_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        EngineConfig().seed = 1  # type: ignore[misc]


@pytest.mark.parametrize("value", (True, 1.5, "0"))
def test_config_rejects_bad_seed(value: Any) -> None:
    with pytest.raises(EngineConfigurationError):
        EngineConfig(seed=value)


def test_config_rejects_duplicate_stages() -> None:
    with pytest.raises(EngineConfigurationError):
        EngineConfig(
            stage_order=(
                EngineStage.ANALYSIS,
                EngineStage.ANALYSIS,
            )
        )


def test_config_rejects_missing_required_stage() -> None:
    with pytest.raises(EngineConfigurationError):
        EngineConfig(
            stage_order=(EngineStage.ANALYSIS,),
            required_stages=(EngineStage.LOCALIZATION,),
        )


@pytest.mark.parametrize(
    "field",
    (
        "capture_stage_errors",
        "continue_after_optional_failure",
        "measure_runtime",
        "build_report",
    ),
)
@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_config_rejects_bad_flags(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(EngineConfigurationError):
        EngineConfig(**{field: value})


def test_context_accessors() -> None:
    ctx = EngineContext(
        run_id="run",
        input_data={"x": 1},
        config=config(),
    )
    ctx.set_result(EngineStage.ANALYSIS, {"score": 1})
    assert ctx.get(EngineStage.ANALYSIS)["score"] == 1
    assert ctx.require("analysis")["score"] == 1

    with pytest.raises(EngineError):
        ctx.require(EngineStage.LOCALIZATION)


def test_context_warning() -> None:
    ctx = EngineContext(
        run_id="run",
        input_data=None,
        config=config(),
    )
    ctx.add_warning("TEST", "Warning.")
    assert ctx.warnings[0].code == "TEST"


def test_stage_result_success() -> None:
    item = StageResult(
        stage=EngineStage.ANALYSIS,
        status=StageStatus.SUCCEEDED,
        value={"x": 1},
        duration_seconds=0.1,
    )
    assert item.as_dict()["stage"] == "analysis"


def test_failed_stage_requires_error() -> None:
    with pytest.raises(EngineConfigurationError):
        StageResult(
            stage=EngineStage.ANALYSIS,
            status=StageStatus.FAILED,
            value=None,
            duration_seconds=None,
        )


def test_skipped_stage_disallows_value() -> None:
    with pytest.raises(EngineConfigurationError):
        StageResult(
            stage=EngineStage.EXECUTION,
            status=StageStatus.SKIPPED,
            value={"x": 1},
            duration_seconds=None,
        )


def test_engine_starts_ready() -> None:
    engine = ROIFEngine(config())
    assert engine.state is EngineState.READY
    assert engine.last_result is None


def test_register_unregister_stage() -> None:
    engine = ROIFEngine(config())
    engine.register_stage(EngineStage.ANALYSIS, lambda ctx: 1)
    assert engine.has_stage(EngineStage.ANALYSIS)
    engine.unregister_stage(EngineStage.ANALYSIS)
    assert not engine.has_stage(EngineStage.ANALYSIS)


def test_register_rejects_bad_callback() -> None:
    with pytest.raises(EngineConfigurationError):
        ROIFEngine(config()).register_stage(
            EngineStage.ANALYSIS,
            None,  # type: ignore[arg-type]
        )


def test_full_run() -> None:
    engine = configured_engine()

    result = engine.run(
        {"network": "demo"},
        run_id="run-1",
        metadata={"experiment": "E1"},
    )

    assert result.status is EngineRunStatus.COMPLETED
    assert result.state is EngineState.FINISHED
    assert result.results["localization"]["d_root"] == "A"
    assert result.results["optimization"]["node_star"] == "B"
    assert isinstance(result.report, ROIFReport)
    assert result.report.section("localization").content["d_root"] == "A"
    assert result.metadata["experiment"] == "E1"
    assert engine.last_result is result


def test_run_is_deterministic_for_seed() -> None:
    first = configured_engine(
        engine_config=config(seed=42)
    ).run(None)
    second = configured_engine(
        engine_config=config(seed=42)
    ).run(None)

    assert first.results["analysis"]["seed"] == 42
    assert second.results["analysis"]["seed"] == 42
    assert first.results["analysis"] == second.results["analysis"]


def test_dry_run_skips_execution() -> None:
    result = configured_engine().run(
        None,
        dry_run=True,
    )

    assert result.status is EngineRunStatus.DRY_RUN
    execution = result.stage(EngineStage.EXECUTION)
    assert execution.status is StageStatus.SKIPPED
    assert "execution" not in result.results


def test_optional_stage_is_skipped_when_unregistered() -> None:
    item = config(
        stage_order=(
            EngineStage.ANALYSIS,
            EngineStage.LOCALIZATION,
            EngineStage.OPTIMIZATION,
            EngineStage.PREDICTION,
        )
    )
    result = configured_engine(
        engine_config=item
    ).run(None)

    assert result.stage(
        EngineStage.PREDICTION
    ).status is StageStatus.SKIPPED
    assert result.status is EngineRunStatus.COMPLETED


def test_required_missing_stage_fails_run() -> None:
    engine = ROIFEngine(config(), clock=Clock())
    engine.register_stage(
        EngineStage.ANALYSIS,
        lambda ctx: {},
    )

    result = engine.run(None)

    assert result.status is EngineRunStatus.FAILED
    assert result.state is EngineState.FAILED
    assert result.stage(
        EngineStage.LOCALIZATION
    ).status is StageStatus.FAILED


def test_optional_failure_makes_partial_result() -> None:
    item = config(
        stage_order=(
            EngineStage.ANALYSIS,
            EngineStage.LOCALIZATION,
            EngineStage.OPTIMIZATION,
            EngineStage.PREDICTION,
            EngineStage.REPORTING,
        )
    )
    engine = configured_engine(engine_config=item)
    engine.register_stage(
        EngineStage.PREDICTION,
        lambda ctx: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    result = engine.run(None)

    assert result.status is EngineRunStatus.PARTIAL
    assert result.failed_stage_count == 1
    assert result.report is not None
    assert result.report.warnings[0].code == "STAGE_FAILED"


def test_required_failure_stops_later_stages() -> None:
    engine = configured_engine()
    engine.register_stage(
        EngineStage.LOCALIZATION,
        lambda ctx: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    result = engine.run(None)

    assert result.status is EngineRunStatus.FAILED
    assert result.stage(
        EngineStage.LOCALIZATION
    ).status is StageStatus.FAILED
    assert result.stage(
        EngineStage.OPTIMIZATION
    ).status is StageStatus.SKIPPED


def test_error_can_be_propagated() -> None:
    item = config(capture_stage_errors=False)
    engine = configured_engine(engine_config=item)
    engine.register_stage(
        EngineStage.ANALYSIS,
        lambda ctx: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        engine.run(None)

    assert engine.state is EngineState.FAILED


def test_hooks_execute_in_order() -> None:
    events: list[str] = []
    engine = configured_engine()

    engine.add_before_run_hook(
        lambda ctx: events.append("before_run")
    )
    engine.add_before_stage_hook(
        lambda ctx, stage: events.append(
            f"before:{stage.value}"
        )
    )
    engine.add_after_stage_hook(
        lambda ctx, stage: events.append(
            f"after:{stage.value}"
        )
    )
    engine.add_after_run_hook(
        lambda ctx: events.append("after_run")
    )

    engine.run(
        None,
        stages=(EngineStage.ANALYSIS,),
    )

    assert events == [
        "before_run",
        "before:analysis",
        "after:analysis",
        "after_run",
    ]


def test_timing_is_collected() -> None:
    result = configured_engine().run(
        None,
        stages=(EngineStage.ANALYSIS,),
    )

    assert result.total_duration_seconds is not None
    assert result.stage(
        EngineStage.ANALYSIS
    ).duration_seconds == pytest.approx(0.1)


def test_build_report_can_be_disabled() -> None:
    item = config(build_report=False)
    result = configured_engine(
        engine_config=item
    ).run(None)

    assert result.report is None
    assert "reporting" not in result.results


def test_result_serialization() -> None:
    result = configured_engine().run(None)
    payload = json.loads(result.to_json())

    assert payload["status"] == "completed"
    assert payload["results"]["optimization"]["node_star"] == "B"
    assert payload["report"]["status"] == "completed"


def test_result_markdown() -> None:
    text = configured_engine().run(None).to_markdown()

    assert "# ROIF Engine Report" in text
    assert "## Localization" in text
    assert "d_root" in text


def test_engine_exports_last_result() -> None:
    engine = configured_engine()
    engine.run(None)

    assert json.loads(engine.to_json())["status"] == "completed"
    assert "# ROIF Engine Report" in engine.to_markdown()


def test_export_before_run_fails() -> None:
    engine = configured_engine()

    with pytest.raises(EngineError):
        engine.to_json()
    with pytest.raises(EngineError):
        engine.to_markdown()


def test_reset() -> None:
    engine = configured_engine()
    engine.run(None)
    engine.reset()

    assert engine.state is EngineState.READY
    assert engine.last_result is None


def test_selected_stage_validation() -> None:
    engine = configured_engine()

    with pytest.raises(EngineConfigurationError):
        engine.run(None, stages=())

    with pytest.raises(EngineConfigurationError):
        engine.run(
            None,
            stages=(
                EngineStage.ANALYSIS,
                EngineStage.ANALYSIS,
            ),
        )


def test_engine_result_stage_lookup_failure() -> None:
    result = configured_engine().run(
        None,
        stages=(EngineStage.ANALYSIS,),
    )

    with pytest.raises(KeyError):
        result.stage(EngineStage.BENCHMARK)


def test_engine_result_mapping_is_immutable() -> None:
    result = configured_engine().run(
        None,
        stages=(EngineStage.ANALYSIS,),
    )

    assert isinstance(result.results, MappingProxyType)
    with pytest.raises(TypeError):
        result.results["x"] = 1  # type: ignore[index]
