"""
Public Python API for ROIF Engine v1.

This module provides a stable, compact facade over the lower-level ROIF
components. It is intended to be the primary import surface for notebooks,
applications, experiments, and future web services.

The API deliberately separates three concerns:

1. session configuration;
2. execution through a configured ``ROIFEngine``;
3. persistence and publication helpers.

Typical usage:

    from roif.api import ROIF, ROIFConfig

    roif = ROIF(
        ROIFConfig(seed=42),
        stages={
            EngineStage.ANALYSIS: analyze,
            EngineStage.LOCALIZATION: localize,
            EngineStage.OPTIMIZATION: optimize,
        },
    )

    result = roif.run(input_data, run_id="experiment-1")
    roif.save_result(result, "result.json")

The facade does not invent scientific stage implementations. Callers register
their existing analyzer, localization, optimization, prediction, planning,
execution, uncertainty, validation, benchmark, or reporting callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias
import json

from .datasets import (
    Dataset,
    DatasetSplit,
    load_dataset,
    save_dataset,
    split_dataset,
)
from .paper import (
    PaperDocument,
    paper_from_mapping,
    render_latex,
    render_markdown as render_paper_markdown,
)
from .reporting import (
    ROIFReport,
    render_json as render_report_json,
    render_markdown as render_report_markdown,
)
from .roif_engine import (
    EngineConfig,
    EngineContext,
    EngineResult,
    EngineRunStatus,
    EngineStage,
    EngineState,
    ROIFEngine,
)
from .serialization import (
    ChecksumAlgorithm,
    SerializationEnvelope,
    SerializationError,
    load_json,
    make_envelope,
    save_json,
    to_serializable,
)


class APIError(RuntimeError):
    """Raised when the public ROIF API cannot complete an operation."""


StageCallable: TypeAlias = Callable[[EngineContext], Any]


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
        raise APIError(f"{name} must be a string{suffix}.")
    normalized = value.strip()
    if not normalized:
        raise APIError(f"{name} cannot be empty.")
    return normalized


def _freeze(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise APIError(f"{name} must be a mapping.")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ROIFConfig:
    """Stable public configuration for a ROIF session."""

    seed: int = 0
    engine_version: str = "1.0.0"
    schema_version: str = "1.0"
    required_stages: tuple[EngineStage, ...] = (
        EngineStage.ANALYSIS,
        EngineStage.LOCALIZATION,
        EngineStage.OPTIMIZATION,
    )
    stage_order: tuple[EngineStage, ...] = (
        EngineStage.ANALYSIS,
        EngineStage.LOCALIZATION,
        EngineStage.OPTIMIZATION,
        EngineStage.PREDICTION,
        EngineStage.PLANNING,
        EngineStage.EXECUTION,
        EngineStage.UNCERTAINTY,
        EngineStage.VALIDATION,
        EngineStage.BENCHMARK,
        EngineStage.REPORTING,
    )
    capture_stage_errors: bool = True
    continue_after_optional_failure: bool = True
    measure_runtime: bool = True
    build_report: bool = True
    report_title: str = "ROIF Engine Report"
    commit: str | None = None
    command: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise APIError("seed must be an integer.")

        for name in ("engine_version", "schema_version", "report_title"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )

        for name in ("commit", "command"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _text(value, name=name, optional=True),
                )

        stage_order = tuple(self.stage_order)
        required_stages = tuple(self.required_stages)

        if not stage_order:
            raise APIError("stage_order cannot be empty.")
        if any(not isinstance(item, EngineStage) for item in stage_order):
            raise APIError("stage_order must contain EngineStage values.")
        if any(not isinstance(item, EngineStage) for item in required_stages):
            raise APIError(
                "required_stages must contain EngineStage values."
            )
        if len(stage_order) != len(set(stage_order)):
            raise APIError("stage_order cannot contain duplicates.")
        if len(required_stages) != len(set(required_stages)):
            raise APIError("required_stages cannot contain duplicates.")
        if set(required_stages) - set(stage_order):
            raise APIError(
                "required_stages must be present in stage_order."
            )

        object.__setattr__(self, "stage_order", stage_order)
        object.__setattr__(self, "required_stages", required_stages)

        for name in (
            "capture_stage_errors",
            "continue_after_optional_failure",
            "measure_runtime",
            "build_report",
        ):
            if not isinstance(getattr(self, name), bool):
                raise APIError(f"{name} must be a bool.")

        object.__setattr__(
            self,
            "environment",
            _freeze(self.environment, name="environment"),
        )

    def to_engine_config(self) -> EngineConfig:
        """Convert public configuration to the internal engine config."""
        return EngineConfig(
            engine_version=self.engine_version,
            schema_version=self.schema_version,
            seed=self.seed,
            stage_order=self.stage_order,
            required_stages=self.required_stages,
            capture_stage_errors=self.capture_stage_errors,
            continue_after_optional_failure=(
                self.continue_after_optional_failure
            ),
            measure_runtime=self.measure_runtime,
            build_report=self.build_report,
            report_title=self.report_title,
            commit=self.commit,
            command=self.command,
            environment=self.environment,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "required_stages": [
                item.value for item in self.required_stages
            ],
            "stage_order": [
                item.value for item in self.stage_order
            ],
            "capture_stage_errors": self.capture_stage_errors,
            "continue_after_optional_failure": (
                self.continue_after_optional_failure
            ),
            "measure_runtime": self.measure_runtime,
            "build_report": self.build_report,
            "report_title": self.report_title,
            "commit": self.commit,
            "command": self.command,
            "environment": to_serializable(self.environment),
        }


class ROIF:
    """Primary public Python facade for ROIF Engine v1."""

    def __init__(
        self,
        config: ROIFConfig | None = None,
        *,
        stages: Mapping[EngineStage, StageCallable] | None = None,
        engine: ROIFEngine | None = None,
    ) -> None:
        if config is None:
            config = ROIFConfig()
        if not isinstance(config, ROIFConfig):
            raise APIError("config must be a ROIFConfig.")

        if engine is not None and not isinstance(engine, ROIFEngine):
            raise APIError("engine must be a ROIFEngine or None.")

        if engine is not None and stages:
            raise APIError(
                "stages cannot be supplied with an existing engine."
            )

        self._config = config
        self._engine = (
            engine
            if engine is not None
            else ROIFEngine(config.to_engine_config())
        )

        if stages is not None:
            if not isinstance(stages, Mapping):
                raise APIError("stages must be a mapping.")
            for stage, callback in stages.items():
                self.register(stage, callback)

    @property
    def config(self) -> ROIFConfig:
        return self._config

    @property
    def engine(self) -> ROIFEngine:
        return self._engine

    @property
    def state(self) -> EngineState:
        return self._engine.state

    @property
    def last_result(self) -> EngineResult | None:
        return self._engine.last_result

    def register(
        self,
        stage: EngineStage,
        callback: StageCallable,
    ) -> "ROIF":
        """Register or replace one stage callback."""
        if not isinstance(stage, EngineStage):
            raise APIError("stage must be an EngineStage.")
        if not callable(callback):
            raise APIError("callback must be callable.")
        self._engine.register_stage(stage, callback)
        return self

    def unregister(self, stage: EngineStage) -> "ROIF":
        """Remove one stage callback."""
        if not isinstance(stage, EngineStage):
            raise APIError("stage must be an EngineStage.")
        self._engine.unregister_stage(stage)
        return self

    def run(
        self,
        input_data: Any,
        *,
        run_id: str = "roif-run",
        dry_run: bool = False,
        stages: Sequence[EngineStage] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EngineResult:
        """Run the configured ROIF pipeline."""
        try:
            return self._engine.run(
                input_data,
                run_id=run_id,
                dry_run=dry_run,
                stages=stages,
                metadata=metadata,
            )
        except Exception as exc:
            if isinstance(exc, APIError):
                raise
            raise APIError(str(exc) or type(exc).__name__) from exc

    def run_dataset(
        self,
        dataset: Dataset,
        *,
        run_id_prefix: str = "dataset",
        dry_run: bool = False,
        stages: Sequence[EngineStage] | None = None,
        stop_on_failure: bool = False,
    ) -> tuple[EngineResult, ...]:
        """Run each dataset record through the configured engine."""
        if not isinstance(dataset, Dataset):
            raise APIError("dataset must be a Dataset.")
        prefix = _text(run_id_prefix, name="run_id_prefix")
        if not isinstance(stop_on_failure, bool):
            raise APIError("stop_on_failure must be a bool.")

        results: list[EngineResult] = []
        for index, record in enumerate(dataset):
            result = self.run(
                {
                    "record_id": record.record_id,
                    "features": dict(record.features),
                    "target": record.target,
                    "group": record.group,
                    "tags": record.tags,
                    "metadata": dict(record.metadata),
                },
                run_id=f"{prefix}-{index}",
                dry_run=dry_run,
                stages=stages,
                metadata={
                    "dataset": dataset.name,
                    "record_id": record.record_id,
                },
            )
            results.append(result)

            if (
                stop_on_failure
                and result.status is EngineRunStatus.FAILED
            ):
                break

        return tuple(results)

    def reset(self) -> "ROIF":
        self._engine.reset()
        return self

    def save_result(
        self,
        result: EngineResult,
        path: str | Path,
        *,
        envelope: bool = False,
        object_type: str = "EngineResult",
        schema_version: str | None = None,
        checksum_algorithm: ChecksumAlgorithm = (
            ChecksumAlgorithm.SHA256
        ),
    ) -> Path:
        """Persist an engine result as JSON."""
        if not isinstance(result, EngineResult):
            raise APIError("result must be an EngineResult.")

        payload: Any = result
        if envelope:
            payload = make_envelope(
                result,
                object_type=object_type,
                schema_version=(
                    self.config.schema_version
                    if schema_version is None
                    else schema_version
                ),
                producer="ROIF Engine",
                producer_version=self.config.engine_version,
                checksum_algorithm=checksum_algorithm,
            )

        try:
            return save_json(payload, path)
        except SerializationError as exc:
            raise APIError(str(exc)) from exc

    def save_last_result(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Persist the most recent engine result."""
        if self.last_result is None:
            raise APIError("No engine result is available.")
        return self.save_result(
            self.last_result,
            path,
            **kwargs,
        )

    def result_json(
        self,
        result: EngineResult | None = None,
        *,
        indent: int | None = 2,
        sort_keys: bool = False,
    ) -> str:
        """Return one engine result as JSON."""
        selected = self.last_result if result is None else result
        if selected is None:
            raise APIError("No engine result is available.")
        if not isinstance(selected, EngineResult):
            raise APIError("result must be an EngineResult.")
        return selected.to_json(
            indent=indent,
            sort_keys=sort_keys,
        )

    def result_markdown(
        self,
        result: EngineResult | None = None,
    ) -> str:
        """Return one generated report as Markdown."""
        selected = self.last_result if result is None else result
        if selected is None:
            raise APIError("No engine result is available.")
        if selected.report is None:
            raise APIError(
                "The selected result does not contain a report."
            )
        return render_report_markdown(selected.report)

    @staticmethod
    def load_dataset(path: str | Path) -> Dataset:
        try:
            return load_dataset(path)
        except Exception as exc:
            raise APIError(str(exc)) from exc

    @staticmethod
    def save_dataset(
        dataset: Dataset,
        path: str | Path,
    ) -> Path:
        try:
            return save_dataset(dataset, path)
        except Exception as exc:
            raise APIError(str(exc)) from exc

    @staticmethod
    def split_dataset(
        dataset: Dataset,
        **kwargs: Any,
    ) -> DatasetSplit:
        try:
            return split_dataset(dataset, **kwargs)
        except Exception as exc:
            raise APIError(str(exc)) from exc

    @staticmethod
    def load_paper(path: str | Path) -> PaperDocument:
        try:
            value = load_json(path)
        except SerializationError as exc:
            raise APIError(str(exc)) from exc
        if not isinstance(value, Mapping):
            raise APIError("Paper JSON must contain an object.")
        try:
            return paper_from_mapping(value)
        except Exception as exc:
            raise APIError(str(exc)) from exc

    @staticmethod
    def save_paper(
        paper: PaperDocument,
        path: str | Path,
    ) -> Path:
        if not isinstance(paper, PaperDocument):
            raise APIError("paper must be a PaperDocument.")
        try:
            return save_json(paper, path)
        except SerializationError as exc:
            raise APIError(str(exc)) from exc

    @staticmethod
    def paper_markdown(paper: PaperDocument) -> str:
        if not isinstance(paper, PaperDocument):
            raise APIError("paper must be a PaperDocument.")
        return render_paper_markdown(paper)

    @staticmethod
    def paper_latex(paper: PaperDocument) -> str:
        if not isinstance(paper, PaperDocument):
            raise APIError("paper must be a PaperDocument.")
        return render_latex(paper)


def create_roif(
    *,
    config: ROIFConfig | None = None,
    stages: Mapping[EngineStage, StageCallable] | None = None,
) -> ROIF:
    """Convenience constructor for the public facade."""
    return ROIF(config=config, stages=stages)


def run_roif(
    input_data: Any,
    *,
    stages: Mapping[EngineStage, StageCallable],
    config: ROIFConfig | None = None,
    run_id: str = "roif-run",
    dry_run: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> EngineResult:
    """Configure and execute one ROIF run in a single call."""
    return create_roif(
        config=config,
        stages=stages,
    ).run(
        input_data,
        run_id=run_id,
        dry_run=dry_run,
        metadata=metadata,
    )


__all__ = [
    "APIError",
    "EngineContext",
    "EngineResult",
    "EngineRunStatus",
    "EngineStage",
    "EngineState",
    "ROIF",
    "ROIFConfig",
    "StageCallable",
    "create_roif",
    "run_roif",
]
