"""
Top-level facade for ROIF Engine v1.

The facade coordinates already implemented ROIF subsystems without duplicating
their mathematics. Each stage is supplied as a callback, which keeps the engine
compatible with the current analyzer, localization, optimization, prediction,
planning, execution, uncertainty, validation, benchmark, and reporting APIs.

The official execution order is configurable but deterministic:

    analysis
    localization
    optimization
    prediction
    planning
    execution
    uncertainty
    validation
    benchmark
    reporting

A stage receives ``EngineContext`` and returns any serializable result. The
returned value is stored in the context under the stage name and is available
to all later stages.

The facade supports:

- immutable configuration and results;
- lifecycle states;
- stage selection;
- required and optional stages;
- dry-run execution;
- before/after run hooks;
- before/after stage hooks;
- stage timing;
- captured or propagated failures;
- deterministic seed propagation;
- report generation through ``roif.reporting``;
- JSON and Markdown export.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable
import json
import math
import time

from .reporting import (
    ROIFReport,
    ROIFReportBuilder,
    ReportSectionKind,
    ReportStatus,
    ReportWarning,
    ReportingError,
    ReproducibilityInfo,
    WarningSeverity,
    render_json,
    render_markdown,
    to_json_compatible,
)


class EngineError(RuntimeError):
    """Raised when ROIF Engine cannot complete a requested operation."""


class EngineConfigurationError(ValueError):
    """Raised when engine configuration is invalid."""


StageCallable: TypeAlias = Callable[["EngineContext"], Any]
RunHook: TypeAlias = Callable[["EngineContext"], None]
StageHook: TypeAlias = Callable[["EngineContext", "EngineStage"], None]


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
        raise EngineConfigurationError(
            f"{name} must be a string{suffix}."
        )
    normalized = value.strip()
    if not normalized:
        raise EngineConfigurationError(f"{name} cannot be empty.")
    return normalized


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise EngineConfigurationError(
            f"{name} must be a real number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EngineConfigurationError(
            f"{name} must be a real number."
        ) from exc
    if not math.isfinite(result):
        raise EngineConfigurationError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise EngineConfigurationError(
            f"{name} must be greater than or equal to {minimum}."
        )
    return result


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise EngineConfigurationError(f"{name} must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        normalized = _normalize_text(key, name=f"{name} key")
        copied[normalized] = item
    return MappingProxyType(copied)


class EngineState(str, Enum):
    """Lifecycle state of the engine."""

    INITIALIZED = "initialized"
    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class EngineStage(str, Enum):
    """Canonical ROIF Engine v1 stages."""

    ANALYSIS = "analysis"
    LOCALIZATION = "localization"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    PLANNING = "planning"
    EXECUTION = "execution"
    UNCERTAINTY = "uncertainty"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"
    REPORTING = "reporting"


class StageStatus(str, Enum):
    """Outcome of one engine stage."""

    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


class EngineRunStatus(str, Enum):
    """Outcome of a complete engine run."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    DRY_RUN = "dry_run"


DEFAULT_STAGE_ORDER: tuple[EngineStage, ...] = (
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


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Configuration for the top-level ROIF facade."""

    engine_version: str = "1.0.0"
    schema_version: str = "1.0"
    seed: int = 0
    stage_order: tuple[EngineStage, ...] = DEFAULT_STAGE_ORDER
    required_stages: tuple[EngineStage, ...] = (
        EngineStage.ANALYSIS,
        EngineStage.LOCALIZATION,
        EngineStage.OPTIMIZATION,
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

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EngineConfigurationError(
                "seed must be an integer."
            )

        for field_name in ("stage_order", "required_stages"):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(item, EngineStage) for item in values):
                raise EngineConfigurationError(
                    f"{field_name} must contain EngineStage values."
                )
            if len(values) != len(set(values)):
                raise EngineConfigurationError(
                    f"{field_name} cannot contain duplicates."
                )
            object.__setattr__(self, field_name, values)

        if not self.stage_order:
            raise EngineConfigurationError(
                "stage_order cannot be empty."
            )

        missing = set(self.required_stages) - set(self.stage_order)
        if missing:
            names = ", ".join(
                sorted(item.value for item in missing)
            )
            raise EngineConfigurationError(
                f"required_stages are absent from stage_order: {names}."
            )

        for name in (
            "capture_stage_errors",
            "continue_after_optional_failure",
            "measure_runtime",
            "build_report",
        ):
            if not isinstance(getattr(self, name), bool):
                raise EngineConfigurationError(
                    f"{name} must be a bool."
                )

        object.__setattr__(
            self,
            "report_title",
            _normalize_text(
                self.report_title,
                name="report_title",
            ),
        )

        for name in ("commit", "command"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _normalize_text(
                        value,
                        name=name,
                        optional=True,
                    ),
                )

        object.__setattr__(
            self,
            "environment",
            _freeze_mapping(
                self.environment,
                name="environment",
            ),
        )


@dataclass(slots=True)
class EngineContext:
    """Mutable context passed through all engine stages."""

    run_id: str
    input_data: Any
    config: EngineConfig
    dry_run: bool = False
    results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[ReportWarning] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_id = _normalize_text(
            self.run_id,
            name="run_id",
        )
        if not isinstance(self.config, EngineConfig):
            raise EngineConfigurationError(
                "config must be an EngineConfig."
            )
        if not isinstance(self.dry_run, bool):
            raise EngineConfigurationError(
                "dry_run must be a bool."
            )
        if not isinstance(self.results, dict):
            raise EngineConfigurationError(
                "results must be a dict."
            )
        if not isinstance(self.metadata, dict):
            raise EngineConfigurationError(
                "metadata must be a dict."
            )
        if not isinstance(self.warnings, list):
            raise EngineConfigurationError(
                "warnings must be a list."
            )

    def get(self, stage: EngineStage | str, default: Any = None) -> Any:
        key = stage.value if isinstance(stage, EngineStage) else stage
        return self.results.get(key, default)

    def require(self, stage: EngineStage | str) -> Any:
        key = stage.value if isinstance(stage, EngineStage) else stage
        if key not in self.results:
            raise EngineError(
                f"Required stage result is missing: {key}."
            )
        return self.results[key]

    def set_result(self, stage: EngineStage, value: Any) -> None:
        if not isinstance(stage, EngineStage):
            raise EngineConfigurationError(
                "stage must be an EngineStage."
            )
        self.results[stage.value] = value

    def add_warning(
        self,
        code: str,
        message: str,
        *,
        severity: WarningSeverity = WarningSeverity.WARNING,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.warnings.append(
            ReportWarning(
                code=code,
                message=message,
                severity=severity,
                source=source,
                metadata={} if metadata is None else metadata,
            )
        )


@dataclass(frozen=True, slots=True)
class StageResult:
    """Immutable record of one engine stage."""

    stage: EngineStage
    status: StageStatus
    value: Any
    duration_seconds: float | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, EngineStage):
            raise EngineConfigurationError(
                "stage must be an EngineStage."
            )
        if not isinstance(self.status, StageStatus):
            raise EngineConfigurationError(
                "status must be a StageStatus."
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

        for name in ("error_type", "error_message"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _normalize_text(
                        value,
                        name=name,
                        optional=True,
                    ),
                )

        if self.status is StageStatus.SUCCEEDED:
            if self.error_type is not None or self.error_message is not None:
                raise EngineConfigurationError(
                    "Successful stage cannot contain errors."
                )
        elif self.status is StageStatus.FAILED:
            if self.error_type is None or self.error_message is None:
                raise EngineConfigurationError(
                    "Failed stage requires error details."
                )
        elif self.status is StageStatus.SKIPPED:
            if self.value is not None:
                raise EngineConfigurationError(
                    "Skipped stage cannot contain a value."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "value": (
                None
                if self.value is None
                else to_json_compatible(self.value)
            ),
            "duration_seconds": self.duration_seconds,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Immutable result of one full ROIF Engine run."""

    run_id: str
    status: EngineRunStatus
    state: EngineState
    dry_run: bool
    seed: int
    stages: tuple[StageResult, ...]
    results: Mapping[str, Any]
    report: ROIFReport | None
    total_duration_seconds: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _normalize_text(self.run_id, name="run_id"),
        )
        if not isinstance(self.status, EngineRunStatus):
            raise EngineConfigurationError(
                "status must be an EngineRunStatus."
            )
        if not isinstance(self.state, EngineState):
            raise EngineConfigurationError(
                "state must be an EngineState."
            )
        if not isinstance(self.dry_run, bool):
            raise EngineConfigurationError(
                "dry_run must be a bool."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EngineConfigurationError(
                "seed must be an integer."
            )

        stages = tuple(self.stages)
        if any(not isinstance(item, StageResult) for item in stages):
            raise EngineConfigurationError(
                "stages must contain StageResult values."
            )
        object.__setattr__(self, "stages", stages)

        object.__setattr__(
            self,
            "results",
            _freeze_mapping(self.results, name="results"),
        )

        if self.report is not None and not isinstance(
            self.report,
            ROIFReport,
        ):
            raise EngineConfigurationError(
                "report must be a ROIFReport or None."
            )

        if self.total_duration_seconds is not None:
            object.__setattr__(
                self,
                "total_duration_seconds",
                _finite_float(
                    self.total_duration_seconds,
                    name="total_duration_seconds",
                    minimum=0.0,
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    @property
    def failed_stage_count(self) -> int:
        return sum(
            item.status is StageStatus.FAILED
            for item in self.stages
        )

    @property
    def succeeded_stage_count(self) -> int:
        return sum(
            item.status is StageStatus.SUCCEEDED
            for item in self.stages
        )

    def stage(self, stage: EngineStage) -> StageResult:
        if not isinstance(stage, EngineStage):
            raise EngineConfigurationError(
                "stage must be an EngineStage."
            )
        for item in self.stages:
            if item.stage is stage:
                return item
        raise KeyError(stage.value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "state": self.state.value,
            "dry_run": self.dry_run,
            "seed": self.seed,
            "stages": [item.as_dict() for item in self.stages],
            "results": to_json_compatible(self.results),
            "report": (
                None if self.report is None else self.report.as_dict()
            ),
            "total_duration_seconds": self.total_duration_seconds,
            "failed_stage_count": self.failed_stage_count,
            "succeeded_stage_count": self.succeeded_stage_count,
            "metadata": to_json_compatible(self.metadata),
        }

    def to_json(
        self,
        *,
        indent: int | None = 2,
        sort_keys: bool = False,
    ) -> str:
        if indent is not None:
            if (
                isinstance(indent, bool)
                or not isinstance(indent, int)
                or indent < 0
            ):
                raise EngineConfigurationError(
                    "indent must be a non-negative integer or None."
                )
        if not isinstance(sort_keys, bool):
            raise EngineConfigurationError(
                "sort_keys must be a bool."
            )
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
        )

    def to_markdown(self) -> str:
        if self.report is None:
            raise EngineError(
                "Markdown export requires a generated report."
            )
        return render_markdown(self.report)


@runtime_checkable
class EngineStageCallable(Protocol):
    def __call__(self, context: EngineContext) -> Any:
        ...


class ROIFEngine:
    """Official top-level API for ROIF Engine v1."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config is None:
            config = EngineConfig()
        if not isinstance(config, EngineConfig):
            raise EngineConfigurationError(
                "config must be an EngineConfig."
            )
        if not callable(clock):
            raise EngineConfigurationError(
                "clock must be callable."
            )

        self._config = config
        self._clock = clock
        self._state = EngineState.INITIALIZED
        self._stages: dict[EngineStage, StageCallable] = {}
        self._before_run_hooks: list[RunHook] = []
        self._after_run_hooks: list[RunHook] = []
        self._before_stage_hooks: list[StageHook] = []
        self._after_stage_hooks: list[StageHook] = []
        self._last_result: EngineResult | None = None
        self._state = EngineState.READY

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def last_result(self) -> EngineResult | None:
        return self._last_result

    def register_stage(
        self,
        stage: EngineStage,
        callback: StageCallable,
    ) -> "ROIFEngine":
        if not isinstance(stage, EngineStage):
            raise EngineConfigurationError(
                "stage must be an EngineStage."
            )
        if not callable(callback):
            raise EngineConfigurationError(
                "callback must be callable."
            )
        self._stages[stage] = callback
        return self

    def unregister_stage(
        self,
        stage: EngineStage,
    ) -> "ROIFEngine":
        if not isinstance(stage, EngineStage):
            raise EngineConfigurationError(
                "stage must be an EngineStage."
            )
        self._stages.pop(stage, None)
        return self

    def has_stage(self, stage: EngineStage) -> bool:
        return stage in self._stages

    def add_before_run_hook(self, hook: RunHook) -> "ROIFEngine":
        if not callable(hook):
            raise EngineConfigurationError(
                "hook must be callable."
            )
        self._before_run_hooks.append(hook)
        return self

    def add_after_run_hook(self, hook: RunHook) -> "ROIFEngine":
        if not callable(hook):
            raise EngineConfigurationError(
                "hook must be callable."
            )
        self._after_run_hooks.append(hook)
        return self

    def add_before_stage_hook(
        self,
        hook: StageHook,
    ) -> "ROIFEngine":
        if not callable(hook):
            raise EngineConfigurationError(
                "hook must be callable."
            )
        self._before_stage_hooks.append(hook)
        return self

    def add_after_stage_hook(
        self,
        hook: StageHook,
    ) -> "ROIFEngine":
        if not callable(hook):
            raise EngineConfigurationError(
                "hook must be callable."
            )
        self._after_stage_hooks.append(hook)
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
        if self._state is EngineState.RUNNING:
            raise EngineError(
                "The engine is already running."
            )
        if not isinstance(dry_run, bool):
            raise EngineConfigurationError(
                "dry_run must be a bool."
            )

        selected = (
            self.config.stage_order
            if stages is None
            else tuple(stages)
        )
        if not selected:
            raise EngineConfigurationError(
                "At least one stage must be selected."
            )
        if any(not isinstance(item, EngineStage) for item in selected):
            raise EngineConfigurationError(
                "stages must contain EngineStage values."
            )
        if len(selected) != len(set(selected)):
            raise EngineConfigurationError(
                "stages cannot contain duplicates."
            )

        context = EngineContext(
            run_id=run_id,
            input_data=input_data,
            config=self.config,
            dry_run=dry_run,
            metadata={} if metadata is None else dict(metadata),
        )
        context.metadata.setdefault("seed", self.config.seed)

        stage_records: list[StageResult] = []
        self._state = EngineState.RUNNING
        total_started = (
            self._clock()
            if self.config.measure_runtime
            else None
        )

        try:
            for hook in self._before_run_hooks:
                hook(context)

            stop_pipeline = False

            for stage in selected:
                if stop_pipeline:
                    stage_records.append(
                        StageResult(
                            stage=stage,
                            status=StageStatus.SKIPPED,
                            value=None,
                            duration_seconds=None,
                        )
                    )
                    continue

                if dry_run and stage is EngineStage.EXECUTION:
                    stage_records.append(
                        StageResult(
                            stage=stage,
                            status=StageStatus.SKIPPED,
                            value=None,
                            duration_seconds=None,
                        )
                    )
                    continue

                callback = self._stages.get(stage)

                if callback is None:
                    if stage is EngineStage.REPORTING:
                        continue

                    if stage in self.config.required_stages:
                        error = EngineError(
                            f"Required stage is not registered: "
                            f"{stage.value}."
                        )
                        if not self.config.capture_stage_errors:
                            raise error

                        stage_records.append(
                            StageResult(
                                stage=stage,
                                status=StageStatus.FAILED,
                                value=None,
                                duration_seconds=None,
                                error_type=type(error).__name__,
                                error_message=str(error),
                            )
                        )
                        stop_pipeline = True
                    else:
                        stage_records.append(
                            StageResult(
                                stage=stage,
                                status=StageStatus.SKIPPED,
                                value=None,
                                duration_seconds=None,
                            )
                        )
                    continue

                try:
                    for hook in self._before_stage_hooks:
                        hook(context, stage)

                    started = (
                        self._clock()
                        if self.config.measure_runtime
                        else None
                    )
                    value = callback(context)
                    finished = (
                        self._clock()
                        if self.config.measure_runtime
                        else None
                    )
                    duration = (
                        None
                        if started is None or finished is None
                        else _finite_float(
                            finished - started,
                            name="stage duration",
                            minimum=0.0,
                        )
                    )

                    context.set_result(stage, value)
                    stage_records.append(
                        StageResult(
                            stage=stage,
                            status=StageStatus.SUCCEEDED,
                            value=value,
                            duration_seconds=duration,
                        )
                    )

                    for hook in self._after_stage_hooks:
                        hook(context, stage)

                except Exception as exc:
                    if not self.config.capture_stage_errors:
                        raise

                    stage_records.append(
                        StageResult(
                            stage=stage,
                            status=StageStatus.FAILED,
                            value=None,
                            duration_seconds=None,
                            error_type=type(exc).__name__,
                            error_message=(
                                str(exc) or type(exc).__name__
                            ),
                        )
                    )
                    context.add_warning(
                        code="STAGE_FAILED",
                        message=(
                            f"Stage {stage.value} failed: "
                            f"{str(exc) or type(exc).__name__}"
                        ),
                        severity=(
                            WarningSeverity.CRITICAL
                            if stage in self.config.required_stages
                            else WarningSeverity.WARNING
                        ),
                        source=stage.value,
                    )

                    if stage in self.config.required_stages:
                        stop_pipeline = True
                    elif not self.config.continue_after_optional_failure:
                        stop_pipeline = True

            for hook in self._after_run_hooks:
                hook(context)

            total_finished = (
                self._clock()
                if self.config.measure_runtime
                else None
            )
            total_duration = (
                None
                if total_started is None or total_finished is None
                else _finite_float(
                    total_finished - total_started,
                    name="total duration",
                    minimum=0.0,
                )
            )

            failed_required = any(
                item.status is StageStatus.FAILED
                and item.stage in self.config.required_stages
                for item in stage_records
            )
            failed_optional = any(
                item.status is StageStatus.FAILED
                and item.stage not in self.config.required_stages
                for item in stage_records
            )

            if failed_required:
                run_status = EngineRunStatus.FAILED
                self._state = EngineState.FAILED
            elif dry_run:
                run_status = EngineRunStatus.DRY_RUN
                self._state = EngineState.FINISHED
            elif failed_optional:
                run_status = EngineRunStatus.PARTIAL
                self._state = EngineState.FINISHED
            else:
                run_status = EngineRunStatus.COMPLETED
                self._state = EngineState.FINISHED

            report = None
            if self.config.build_report:
                report = self.build_report(
                    context,
                    stage_records=stage_records,
                    run_status=run_status,
                    total_duration=total_duration,
                )
                context.results[EngineStage.REPORTING.value] = report

                if EngineStage.REPORTING in selected:
                    stage_records.append(
                        StageResult(
                            stage=EngineStage.REPORTING,
                            status=StageStatus.SUCCEEDED,
                            value=report,
                            duration_seconds=None,
                        )
                    )

            result = EngineResult(
                run_id=context.run_id,
                status=run_status,
                state=self._state,
                dry_run=dry_run,
                seed=self.config.seed,
                stages=tuple(stage_records),
                results=context.results,
                report=report,
                total_duration_seconds=total_duration,
                metadata=context.metadata,
            )
            self._last_result = result
            return result

        except Exception:
            self._state = EngineState.FAILED
            raise

    def build_report(
        self,
        context: EngineContext,
        *,
        stage_records: Sequence[StageResult],
        run_status: EngineRunStatus,
        total_duration: float | None,
    ) -> ROIFReport:
        """Build the official ROIF report for one engine run."""

        status_map = {
            EngineRunStatus.COMPLETED: ReportStatus.COMPLETED,
            EngineRunStatus.DRY_RUN: ReportStatus.COMPLETED,
            EngineRunStatus.PARTIAL: ReportStatus.PARTIAL,
            EngineRunStatus.FAILED: ReportStatus.FAILED,
        }

        builder = (
            ROIFReportBuilder(
                context.run_id,
                title=self.config.report_title,
            )
            .set_status(status_map[run_status])
            .set_summary(
                f"ROIF Engine run {run_status.value}. "
                f"{sum(item.status is StageStatus.SUCCEEDED for item in stage_records)} "
                f"stages succeeded, "
                f"{sum(item.status is StageStatus.FAILED for item in stage_records)} "
                f"failed."
            )
            .set_duration(total_duration)
            .set_reproducibility(
                ReproducibilityInfo(
                    engine_version=self.config.engine_version,
                    schema_version=self.config.schema_version,
                    commit=self.config.commit,
                    seed=self.config.seed,
                    command=self.config.command,
                    environment=self.config.environment,
                )
            )
            .update_metadata(context.metadata)
        )

        kind_map = {
            EngineStage.ANALYSIS: ReportSectionKind.METRICS,
            EngineStage.LOCALIZATION: ReportSectionKind.LOCALIZATION,
            EngineStage.OPTIMIZATION: ReportSectionKind.OPTIMIZATION,
            EngineStage.PREDICTION: ReportSectionKind.CUSTOM,
            EngineStage.PLANNING: ReportSectionKind.CUSTOM,
            EngineStage.EXECUTION: ReportSectionKind.EXECUTION,
            EngineStage.UNCERTAINTY: ReportSectionKind.UNCERTAINTY,
            EngineStage.VALIDATION: ReportSectionKind.VALIDATION,
            EngineStage.BENCHMARK: ReportSectionKind.BENCHMARK,
        }

        order = 0
        for record in stage_records:
            if (
                record.status is not StageStatus.SUCCEEDED
                or record.stage is EngineStage.REPORTING
            ):
                continue
            builder.add_result(
                record.stage.value,
                record.value,
                title=record.stage.value.replace("_", " ").title(),
                kind=kind_map.get(
                    record.stage,
                    ReportSectionKind.CUSTOM,
                ),
                order=order,
            )
            order += 1

        for item in context.warnings:
            builder.add_warning(item)

        return builder.build()

    def to_json(self, **kwargs: Any) -> str:
        """Export the most recent result as JSON."""
        if self._last_result is None:
            raise EngineError(
                "No engine result is available."
            )
        return self._last_result.to_json(**kwargs)

    def to_markdown(self) -> str:
        """Export the most recent report as Markdown."""
        if self._last_result is None:
            raise EngineError(
                "No engine result is available."
            )
        return self._last_result.to_markdown()

    def reset(self) -> None:
        """Reset runtime state while preserving registered stages and hooks."""
        if self._state is EngineState.RUNNING:
            raise EngineError(
                "Cannot reset while the engine is running."
            )
        self._last_result = None
        self._state = EngineState.READY


__all__ = [
    "DEFAULT_STAGE_ORDER",
    "EngineConfig",
    "EngineConfigurationError",
    "EngineContext",
    "EngineError",
    "EngineResult",
    "EngineRunStatus",
    "EngineStage",
    "EngineStageCallable",
    "EngineState",
    "ROIFEngine",
    "RunHook",
    "StageCallable",
    "StageHook",
    "StageResult",
    "StageStatus",
]
