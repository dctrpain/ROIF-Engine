"""
Benchmarking utilities for ROIF Engine v1.

This module compares ROIF outputs with caller-supplied baseline methods over
seeded validation trials. It is intentionally backend-independent: graph
construction, ROIF execution, and baseline execution are provided as callbacks.

Supported benchmark outputs include:

- root-localization accuracy and top-k accuracy;
- Node* accuracy and top-k accuracy;
- localization distance;
- gain regret;
- runtime and runtime-per-node;
- convergence and failure rates;
- paired deltas against a designated reference method;
- grouped summaries by graph family and graph size;
- deterministic ordering and immutable serialization.

The module integrates naturally with ``roif.validation`` but does not require
networkx or a specific graph implementation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable
import math
import time

import numpy as np

from .validation import (
    DistanceFunction,
    GraphSpec,
    MetricName,
    MetricSummary,
    TrialMetrics,
    ValidationConfig,
    ValidationError,
    ValidationOutcome,
    ValidationTrial,
    compute_trial_metrics,
    derive_seed,
    summarize_metric,
)


class BenchmarkError(ValueError):
    """Raised when benchmark input or configuration is invalid."""


MethodExecutor: TypeAlias = Callable[
    [ValidationTrial, Any],
    ValidationOutcome | Mapping[str, Any],
]
FixtureFactory: TypeAlias = Callable[[ValidationTrial], Any]


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise BenchmarkError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise BenchmarkError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise BenchmarkError(
            f"{name} must be greater than or equal to {minimum}."
        )
    if maximum is not None and result > maximum:
        raise BenchmarkError(
            f"{name} must be less than or equal to {maximum}."
        )
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkError(
            f"{name} must be an integer greater than or equal to 1."
        )
    return value


def _normalize_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise BenchmarkError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise BenchmarkError(f"{name} cannot be empty.")
    return normalized


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise BenchmarkError("metadata must be a mapping.")
    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        copied[_normalize_text(key, name="metadata key")] = value
    return MappingProxyType(copied)


def _coerce_outcome(
    value: ValidationOutcome | Mapping[str, Any],
) -> ValidationOutcome:
    if isinstance(value, ValidationOutcome):
        return value
    if not isinstance(value, Mapping):
        raise BenchmarkError(
            "method executor must return ValidationOutcome or mapping."
        )
    try:
        return ValidationOutcome(**dict(value))
    except (TypeError, ValidationError) as exc:
        raise BenchmarkError(str(exc)) from exc


class BenchmarkMethodKind(str, Enum):
    """Kind of benchmark method."""

    ROIF = "roif"
    BASELINE = "baseline"


class BenchmarkRunStatus(str, Enum):
    """Status of one method execution."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BenchmarkStatus(str, Enum):
    """Status of a complete benchmark."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BenchmarkMethod:
    """Named benchmark method and execution callback."""

    method_id: str
    label: str
    kind: BenchmarkMethodKind
    execute: MethodExecutor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method_id",
            _normalize_text(self.method_id, name="method_id"),
        )
        object.__setattr__(
            self,
            "label",
            _normalize_text(self.label, name="label"),
        )
        if not isinstance(self.kind, BenchmarkMethodKind):
            raise BenchmarkError(
                "kind must be a BenchmarkMethodKind."
            )
        if not callable(self.execute):
            raise BenchmarkError("execute must be callable.")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "label": self.label,
            "kind": self.kind.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for a reproducible benchmark."""

    repetitions: int = 10
    base_seed: int = 0
    top_k: tuple[int, ...] = (1, 3, 5)
    reference_method_id: str = "roif"
    capture_method_errors: bool = True
    measure_runtime: bool = True
    allow_partial_results: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repetitions",
            _positive_int(self.repetitions, name="repetitions"),
        )
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
        ):
            raise BenchmarkError("base_seed must be an integer.")

        if isinstance(self.top_k, (str, bytes)):
            raise BenchmarkError(
                "top_k must be a sequence of positive integers."
            )
        try:
            values = tuple(self.top_k)
        except TypeError as exc:
            raise BenchmarkError(
                "top_k must be a sequence of positive integers."
            ) from exc
        if not values:
            raise BenchmarkError("top_k cannot be empty.")

        normalized: list[int] = []
        seen: set[int] = set()
        for value in values:
            k = _positive_int(value, name="top_k value")
            if k in seen:
                raise BenchmarkError(
                    "top_k cannot contain duplicate values."
                )
            seen.add(k)
            normalized.append(k)
        object.__setattr__(self, "top_k", tuple(sorted(normalized)))

        object.__setattr__(
            self,
            "reference_method_id",
            _normalize_text(
                self.reference_method_id,
                name="reference_method_id",
            ),
        )

        for name in (
            "capture_method_errors",
            "measure_runtime",
            "allow_partial_results",
        ):
            if not isinstance(getattr(self, name), bool):
                raise BenchmarkError(f"{name} must be a bool.")


@dataclass(frozen=True, slots=True)
class BenchmarkTrial:
    """Trial specification shared by all methods."""

    trial: ValidationTrial
    fixture: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trial, ValidationTrial):
            raise BenchmarkError(
                "trial must be a ValidationTrial."
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class MethodRunResult:
    """Result of one method on one trial."""

    method_id: str
    trial_id: str
    status: BenchmarkRunStatus
    outcome: ValidationOutcome | None
    metrics: TrialMetrics | None
    runtime_seconds: float | None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method_id",
            _normalize_text(self.method_id, name="method_id"),
        )
        object.__setattr__(
            self,
            "trial_id",
            _normalize_text(self.trial_id, name="trial_id"),
        )
        if not isinstance(self.status, BenchmarkRunStatus):
            raise BenchmarkError(
                "status must be a BenchmarkRunStatus."
            )
        if self.outcome is not None and not isinstance(
            self.outcome, ValidationOutcome
        ):
            raise BenchmarkError(
                "outcome must be ValidationOutcome or None."
            )
        if self.metrics is not None and not isinstance(
            self.metrics, TrialMetrics
        ):
            raise BenchmarkError(
                "metrics must be TrialMetrics or None."
            )
        if self.runtime_seconds is not None:
            object.__setattr__(
                self,
                "runtime_seconds",
                _finite_float(
                    self.runtime_seconds,
                    name="runtime_seconds",
                    minimum=0.0,
                ),
            )

        for name in ("error_type", "error_message"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _normalize_text(value, name=name),
                )

        if self.status is BenchmarkRunStatus.SUCCEEDED:
            if self.outcome is None or self.metrics is None:
                raise BenchmarkError(
                    "Successful method run requires outcome and metrics."
                )
            if self.error_type is not None or self.error_message is not None:
                raise BenchmarkError(
                    "Successful method run cannot contain errors."
                )
        else:
            if self.outcome is not None or self.metrics is not None:
                raise BenchmarkError(
                    "Failed method run cannot contain outputs."
                )
            if self.error_type is None or self.error_message is None:
                raise BenchmarkError(
                    "Failed method run requires error details."
                )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "trial_id": self.trial_id,
            "status": self.status.value,
            "outcome": (
                None if self.outcome is None else self.outcome.as_dict()
            ),
            "metrics": (
                None if self.metrics is None else self.metrics.as_dict()
            ),
            "runtime_seconds": self.runtime_seconds,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """Paired metric delta against the reference method."""

    method_id: str
    reference_method_id: str
    metric: str
    count: int
    mean_delta: float | None
    median_delta: float | None
    win_rate: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method_id",
            _normalize_text(self.method_id, name="method_id"),
        )
        object.__setattr__(
            self,
            "reference_method_id",
            _normalize_text(
                self.reference_method_id,
                name="reference_method_id",
            ),
        )
        object.__setattr__(
            self,
            "metric",
            _normalize_text(self.metric, name="metric"),
        )
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise BenchmarkError("count must be an integer.")
        if self.count < 0:
            raise BenchmarkError("count cannot be negative.")

        for name in ("mean_delta", "median_delta", "win_rate"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_float(
                        value,
                        name=name,
                        minimum=0.0 if name == "win_rate" else None,
                        maximum=1.0 if name == "win_rate" else None,
                    ),
                )

        if self.count == 0:
            if any(
                getattr(self, name) is not None
                for name in (
                    "mean_delta",
                    "median_delta",
                    "win_rate",
                )
            ):
                raise BenchmarkError(
                    "Empty paired delta cannot contain statistics."
                )
        else:
            if any(
                getattr(self, name) is None
                for name in (
                    "mean_delta",
                    "median_delta",
                    "win_rate",
                )
            ):
                raise BenchmarkError(
                    "Non-empty paired delta requires statistics."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "reference_method_id": self.reference_method_id,
            "metric": self.metric,
            "count": self.count,
            "mean_delta": self.mean_delta,
            "median_delta": self.median_delta,
            "win_rate": self.win_rate,
        }


@dataclass(frozen=True, slots=True)
class MethodSummary:
    """Aggregate summary for one benchmark method."""

    method: BenchmarkMethod
    trial_count: int
    successful_count: int
    failed_count: int
    metrics: Mapping[str, MetricSummary]

    def __post_init__(self) -> None:
        if not isinstance(self.method, BenchmarkMethod):
            raise BenchmarkError(
                "method must be a BenchmarkMethod."
            )
        for name in (
            "trial_count",
            "successful_count",
            "failed_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise BenchmarkError(f"{name} must be an integer.")
            if value < 0:
                raise BenchmarkError(f"{name} cannot be negative.")

        if self.successful_count + self.failed_count != self.trial_count:
            raise BenchmarkError(
                "method counts must equal trial_count."
            )

        normalized: dict[str, MetricSummary] = {}
        for key, value in self.metrics.items():
            if not isinstance(value, MetricSummary):
                raise BenchmarkError(
                    "metrics values must be MetricSummary."
                )
            normalized[_normalize_text(key, name="metric key")] = value
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(normalized),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.as_dict(),
            "trial_count": self.trial_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "metrics": {
                key: value.as_dict()
                for key, value in self.metrics.items()
            },
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete benchmark result."""

    status: BenchmarkStatus
    config: BenchmarkConfig
    graph_specs: tuple[GraphSpec, ...]
    methods: tuple[BenchmarkMethod, ...]
    runs: tuple[MethodRunResult, ...]
    method_summaries: tuple[MethodSummary, ...]
    paired_deltas: tuple[PairedDelta, ...]
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, BenchmarkStatus):
            raise BenchmarkError(
                "status must be a BenchmarkStatus."
            )
        if not isinstance(self.config, BenchmarkConfig):
            raise BenchmarkError(
                "config must be a BenchmarkConfig."
            )

        specs = tuple(self.graph_specs)
        if any(not isinstance(item, GraphSpec) for item in specs):
            raise BenchmarkError(
                "graph_specs must contain GraphSpec values."
            )
        object.__setattr__(self, "graph_specs", specs)

        methods = tuple(self.methods)
        if any(not isinstance(item, BenchmarkMethod) for item in methods):
            raise BenchmarkError(
                "methods must contain BenchmarkMethod values."
            )
        object.__setattr__(self, "methods", methods)

        runs = tuple(self.runs)
        if any(not isinstance(item, MethodRunResult) for item in runs):
            raise BenchmarkError(
                "runs must contain MethodRunResult values."
            )
        object.__setattr__(self, "runs", runs)

        summaries = tuple(self.method_summaries)
        if any(not isinstance(item, MethodSummary) for item in summaries):
            raise BenchmarkError(
                "method_summaries must contain MethodSummary values."
            )
        object.__setattr__(self, "method_summaries", summaries)

        deltas = tuple(self.paired_deltas)
        if any(not isinstance(item, PairedDelta) for item in deltas):
            raise BenchmarkError(
                "paired_deltas must contain PairedDelta values."
            )
        object.__setattr__(self, "paired_deltas", deltas)

        object.__setattr__(
            self,
            "summary",
            _normalize_text(self.summary, name="summary"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def successful_run_count(self) -> int:
        return sum(
            item.status is BenchmarkRunStatus.SUCCEEDED
            for item in self.runs
        )

    @property
    def failed_run_count(self) -> int:
        return sum(
            item.status is BenchmarkRunStatus.FAILED
            for item in self.runs
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "config": {
                "repetitions": self.config.repetitions,
                "base_seed": self.config.base_seed,
                "top_k": list(self.config.top_k),
                "reference_method_id": self.config.reference_method_id,
                "capture_method_errors": (
                    self.config.capture_method_errors
                ),
                "measure_runtime": self.config.measure_runtime,
                "allow_partial_results": (
                    self.config.allow_partial_results
                ),
            },
            "graph_specs": [
                item.as_dict() for item in self.graph_specs
            ],
            "methods": [item.as_dict() for item in self.methods],
            "runs": [item.as_dict() for item in self.runs],
            "method_summaries": [
                item.as_dict() for item in self.method_summaries
            ],
            "paired_deltas": [
                item.as_dict() for item in self.paired_deltas
            ],
            "successful_run_count": self.successful_run_count,
            "failed_run_count": self.failed_run_count,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class BenchmarkFixtureFactory(Protocol):
    def __call__(self, trial: ValidationTrial) -> Any:
        ...


@runtime_checkable
class BenchmarkMethodExecutor(Protocol):
    def __call__(
        self,
        trial: ValidationTrial,
        fixture: Any,
    ) -> ValidationOutcome | Mapping[str, Any]:
        ...


def _metric_values(
    runs: Sequence[MethodRunResult],
    *,
    top_k: Sequence[int],
) -> Mapping[str, MetricSummary]:
    successful = [
        item for item in runs
        if (
            item.status is BenchmarkRunStatus.SUCCEEDED
            and item.metrics is not None
        )
    ]

    values: dict[str, tuple[MetricName, list[float]]] = {
        "root_accuracy": (MetricName.ROOT_ACCURACY, []),
        "root_localization_distance": (
            MetricName.ROOT_LOCALIZATION_DISTANCE,
            [],
        ),
        "node_star_accuracy": (
            MetricName.NODE_STAR_ACCURACY,
            [],
        ),
        "node_star_gain_regret": (
            MetricName.NODE_STAR_GAIN_REGRET,
            [],
        ),
        "convergence_rate": (
            MetricName.CONVERGENCE_RATE,
            [],
        ),
        "runtime_seconds": (
            MetricName.RUNTIME_SECONDS,
            [],
        ),
        "runtime_per_node": (
            MetricName.RUNTIME_PER_NODE,
            [],
        ),
    }
    for k in top_k:
        values[f"root_top_{k}_accuracy"] = (
            MetricName.ROOT_TOP_K_ACCURACY,
            [],
        )
        values[f"node_star_top_{k}_accuracy"] = (
            MetricName.NODE_STAR_TOP_K_ACCURACY,
            [],
        )

    for run in successful:
        metrics = run.metrics
        assert metrics is not None

        if metrics.root_correct is not None:
            values["root_accuracy"][1].append(
                float(metrics.root_correct)
            )
        if metrics.root_localization_distance is not None:
            values["root_localization_distance"][1].append(
                metrics.root_localization_distance
            )
        for k, correct in metrics.root_top_k.items():
            key = f"root_top_{k}_accuracy"
            if key in values:
                values[key][1].append(float(correct))

        if metrics.node_star_correct is not None:
            values["node_star_accuracy"][1].append(
                float(metrics.node_star_correct)
            )
        for k, correct in metrics.node_star_top_k.items():
            key = f"node_star_top_{k}_accuracy"
            if key in values:
                values[key][1].append(float(correct))

        if metrics.node_star_gain_regret is not None:
            values["node_star_gain_regret"][1].append(
                metrics.node_star_gain_regret
            )

        values["convergence_rate"][1].append(
            float(metrics.converged)
        )
        if metrics.runtime_seconds is not None:
            values["runtime_seconds"][1].append(
                metrics.runtime_seconds
            )
        if metrics.runtime_per_node is not None:
            values["runtime_per_node"][1].append(
                metrics.runtime_per_node
            )

    success_rate = (
        len(successful) / len(runs)
        if runs else 0.0
    )
    values["success_rate"] = (
        MetricName.SUCCESS_RATE,
        [success_rate],
    )

    return MappingProxyType({
        key: summarize_metric(metric_name, metric_values)
        for key, (metric_name, metric_values) in values.items()
    })


def summarize_methods(
    methods: Sequence[BenchmarkMethod],
    runs: Sequence[MethodRunResult],
    *,
    top_k: Sequence[int],
) -> tuple[MethodSummary, ...]:
    """Build one aggregate summary per method."""

    grouped: dict[str, list[MethodRunResult]] = defaultdict(list)
    for run in runs:
        grouped[run.method_id].append(run)

    summaries: list[MethodSummary] = []
    for method in methods:
        method_runs = grouped.get(method.method_id, [])
        successful = sum(
            item.status is BenchmarkRunStatus.SUCCEEDED
            for item in method_runs
        )
        summaries.append(
            MethodSummary(
                method=method,
                trial_count=len(method_runs),
                successful_count=successful,
                failed_count=len(method_runs) - successful,
                metrics=_metric_values(
                    method_runs,
                    top_k=top_k,
                ),
            )
        )
    return tuple(summaries)


def _paired_metric_value(
    run: MethodRunResult,
    metric: str,
) -> float | None:
    if (
        run.status is not BenchmarkRunStatus.SUCCEEDED
        or run.metrics is None
    ):
        return None

    metrics = run.metrics
    if metric == "root_accuracy":
        return (
            None if metrics.root_correct is None
            else float(metrics.root_correct)
        )
    if metric == "node_star_accuracy":
        return (
            None if metrics.node_star_correct is None
            else float(metrics.node_star_correct)
        )
    if metric == "root_localization_distance":
        return metrics.root_localization_distance
    if metric == "node_star_gain_regret":
        return metrics.node_star_gain_regret
    if metric == "runtime_seconds":
        return metrics.runtime_seconds
    raise BenchmarkError(f"Unsupported paired metric: {metric}.")


def paired_deltas(
    runs: Sequence[MethodRunResult],
    *,
    reference_method_id: str,
    method_ids: Sequence[str],
    metrics: Sequence[str] = (
        "root_accuracy",
        "node_star_accuracy",
        "root_localization_distance",
        "node_star_gain_regret",
        "runtime_seconds",
    ),
) -> tuple[PairedDelta, ...]:
    """Compute paired deltas against the reference method."""

    reference_id = _normalize_text(
        reference_method_id,
        name="reference_method_id",
    )

    by_key: dict[tuple[str, str], MethodRunResult] = {
        (run.method_id, run.trial_id): run
        for run in runs
    }
    trial_ids = {
        run.trial_id
        for run in runs
        if run.method_id == reference_id
    }

    lower_is_better = {
        "root_localization_distance",
        "node_star_gain_regret",
        "runtime_seconds",
    }

    results: list[PairedDelta] = []
    for method_id in method_ids:
        method_id = _normalize_text(method_id, name="method_id")
        if method_id == reference_id:
            continue

        for metric in metrics:
            differences: list[float] = []
            wins = 0

            for trial_id in trial_ids:
                reference = by_key.get((reference_id, trial_id))
                method = by_key.get((method_id, trial_id))
                if reference is None or method is None:
                    continue

                reference_value = _paired_metric_value(
                    reference, metric
                )
                method_value = _paired_metric_value(method, metric)
                if reference_value is None or method_value is None:
                    continue

                delta = method_value - reference_value
                differences.append(delta)

                if metric in lower_is_better:
                    wins += method_value < reference_value
                else:
                    wins += method_value > reference_value

            if not differences:
                results.append(
                    PairedDelta(
                        method_id=method_id,
                        reference_method_id=reference_id,
                        metric=metric,
                        count=0,
                        mean_delta=None,
                        median_delta=None,
                        win_rate=None,
                    )
                )
            else:
                results.append(
                    PairedDelta(
                        method_id=method_id,
                        reference_method_id=reference_id,
                        metric=metric,
                        count=len(differences),
                        mean_delta=float(np.mean(differences)),
                        median_delta=float(np.median(differences)),
                        win_rate=wins / len(differences),
                    )
                )

    return tuple(results)


class BenchmarkRunner:
    """Run ROIF and baseline methods on identical seeded trials."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config is None:
            config = BenchmarkConfig()
        if not isinstance(config, BenchmarkConfig):
            raise BenchmarkError(
                "config must be a BenchmarkConfig."
            )
        if not callable(clock):
            raise BenchmarkError("clock must be callable.")

        self._config = config
        self._clock = clock

    @property
    def config(self) -> BenchmarkConfig:
        return self._config

    def run(
        self,
        graph_specs: Sequence[GraphSpec],
        methods: Sequence[BenchmarkMethod],
        *,
        fixture_factory: FixtureFactory,
        true_root_factory: Callable[
            [GraphSpec, int, int], Any
        ] | None = None,
        true_node_star_factory: Callable[
            [GraphSpec, int, int], Any
        ] | None = None,
        distance: DistanceFunction | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        specs = tuple(graph_specs)
        if not specs:
            raise BenchmarkError("graph_specs cannot be empty.")
        if any(not isinstance(item, GraphSpec) for item in specs):
            raise BenchmarkError(
                "graph_specs must contain GraphSpec values."
            )

        method_items = tuple(methods)
        if not method_items:
            raise BenchmarkError("methods cannot be empty.")
        if any(
            not isinstance(item, BenchmarkMethod)
            for item in method_items
        ):
            raise BenchmarkError(
                "methods must contain BenchmarkMethod values."
            )

        ids = [item.method_id for item in method_items]
        if len(ids) != len(set(ids)):
            raise BenchmarkError(
                "method_id values must be unique."
            )
        if self.config.reference_method_id not in ids:
            raise BenchmarkError(
                "reference_method_id must match a benchmark method."
            )
        if not callable(fixture_factory):
            raise BenchmarkError(
                "fixture_factory must be callable."
            )
        if distance is not None and not callable(distance):
            raise BenchmarkError(
                "distance must be callable or None."
            )

        trials: list[BenchmarkTrial] = []
        for graph_index, spec in enumerate(specs):
            for repetition in range(self.config.repetitions):
                seed = derive_seed(
                    self.config.base_seed,
                    graph_index,
                    repetition,
                )
                validation_trial = ValidationTrial(
                    trial_id=(
                        f"{graph_index:03d}-"
                        f"{spec.family.value}-"
                        f"n{spec.node_count}-"
                        f"r{repetition:04d}"
                    ),
                    repetition=repetition,
                    seed=seed,
                    graph=spec,
                    true_root=(
                        None
                        if true_root_factory is None
                        else true_root_factory(
                            spec, repetition, seed
                        )
                    ),
                    true_node_star=(
                        None
                        if true_node_star_factory is None
                        else true_node_star_factory(
                            spec, repetition, seed
                        )
                    ),
                )
                fixture = fixture_factory(validation_trial)
                trials.append(
                    BenchmarkTrial(
                        trial=validation_trial,
                        fixture=fixture,
                    )
                )

        runs: list[MethodRunResult] = []

        for benchmark_trial in trials:
            validation_trial = benchmark_trial.trial
            fixture = benchmark_trial.fixture

            for method in method_items:
                try:
                    started = (
                        self._clock()
                        if self.config.measure_runtime
                        else None
                    )
                    raw = method.execute(
                        validation_trial,
                        fixture,
                    )
                    finished = (
                        self._clock()
                        if self.config.measure_runtime
                        else None
                    )

                    result = _coerce_outcome(raw)
                    measured_runtime = (
                        None
                        if started is None or finished is None
                        else _finite_float(
                            finished - started,
                            name="measured runtime",
                            minimum=0.0,
                        )
                    )
                    runtime = (
                        result.runtime_seconds
                        if result.runtime_seconds is not None
                        else measured_runtime
                    )

                    if runtime != result.runtime_seconds:
                        result = ValidationOutcome(
                            predicted_root=result.predicted_root,
                            root_ranking=result.root_ranking,
                            root_localization_distance=(
                                result.root_localization_distance
                            ),
                            predicted_node_star=(
                                result.predicted_node_star
                            ),
                            node_star_ranking=(
                                result.node_star_ranking
                            ),
                            selected_gain=result.selected_gain,
                            optimal_gain=result.optimal_gain,
                            converged=result.converged,
                            runtime_seconds=runtime,
                            metadata=result.metadata,
                        )

                    trial_metrics = compute_trial_metrics(
                        validation_trial,
                        result,
                        top_k=self.config.top_k,
                        distance=distance,
                        fixture=fixture,
                    )

                    runs.append(
                        MethodRunResult(
                            method_id=method.method_id,
                            trial_id=validation_trial.trial_id,
                            status=BenchmarkRunStatus.SUCCEEDED,
                            outcome=result,
                            metrics=trial_metrics,
                            runtime_seconds=runtime,
                            metadata=method.metadata,
                        )
                    )

                except Exception as exc:
                    if not self.config.capture_method_errors:
                        raise

                    runs.append(
                        MethodRunResult(
                            method_id=method.method_id,
                            trial_id=validation_trial.trial_id,
                            status=BenchmarkRunStatus.FAILED,
                            outcome=None,
                            metrics=None,
                            runtime_seconds=None,
                            error_type=type(exc).__name__,
                            error_message=(
                                str(exc) or type(exc).__name__
                            ),
                            metadata=method.metadata,
                        )
                    )

        summaries = summarize_methods(
            method_items,
            runs,
            top_k=self.config.top_k,
        )
        deltas = paired_deltas(
            runs,
            reference_method_id=self.config.reference_method_id,
            method_ids=ids,
        )

        successful = sum(
            item.status is BenchmarkRunStatus.SUCCEEDED
            for item in runs
        )
        if successful == 0:
            status = BenchmarkStatus.FAILED
        elif successful < len(runs):
            status = (
                BenchmarkStatus.PARTIAL
                if self.config.allow_partial_results
                else BenchmarkStatus.FAILED
            )
        else:
            status = BenchmarkStatus.COMPLETED

        return BenchmarkResult(
            status=status,
            config=self.config,
            graph_specs=specs,
            methods=method_items,
            runs=tuple(runs),
            method_summaries=summaries,
            paired_deltas=deltas,
            summary=(
                f"Benchmark {status.value}: "
                f"{successful}/{len(runs)} method runs succeeded."
            ),
            metadata=metadata,
        )


def run_benchmark(
    graph_specs: Sequence[GraphSpec],
    methods: Sequence[BenchmarkMethod],
    *,
    fixture_factory: FixtureFactory,
    config: BenchmarkConfig | None = None,
    true_root_factory: Callable[
        [GraphSpec, int, int], Any
    ] | None = None,
    true_node_star_factory: Callable[
        [GraphSpec, int, int], Any
    ] | None = None,
    distance: DistanceFunction | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Convenience wrapper around ``BenchmarkRunner.run``."""

    return BenchmarkRunner(config).run(
        graph_specs,
        methods,
        fixture_factory=fixture_factory,
        true_root_factory=true_root_factory,
        true_node_star_factory=true_node_star_factory,
        distance=distance,
        metadata=metadata,
    )


__all__ = [
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkFixtureFactory",
    "BenchmarkMethod",
    "BenchmarkMethodExecutor",
    "BenchmarkMethodKind",
    "BenchmarkResult",
    "BenchmarkRunStatus",
    "BenchmarkRunner",
    "BenchmarkStatus",
    "BenchmarkTrial",
    "FixtureFactory",
    "MethodExecutor",
    "MethodRunResult",
    "MethodSummary",
    "PairedDelta",
    "paired_deltas",
    "run_benchmark",
    "summarize_methods",
]
