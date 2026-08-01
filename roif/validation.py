"""
Seeded validation experiments for ROIF Engine v1.

This module provides a reproducible validation layer for the reference
implementation described in the ROIF journal architecture.

It is intentionally independent from any single graph, solver, localization
backend, or intervention optimizer. Callers provide experiment callbacks while
this module manages:

- deterministic seed generation;
- graph-family and trial specifications;
- repeated synthetic trials;
- root-localization metrics;
- Node* optimization metrics;
- top-k accuracy;
- localization distance;
- runtime and scalability measurements;
- convergence and failure accounting;
- grouped summaries;
- immutable serializable results.

Typical workflow:

    config = ValidationConfig(...)
    runner = ValidationRunner(config)

    result = runner.run(
        trial_factory=...,
        execute_trial=...,
    )

A trial callback receives a fully specified ValidationTrial and returns either
a ValidationOutcome or a mapping compatible with that dataclass.

The module does not generate graph objects itself. Instead, graph-family
parameters are represented by GraphSpec so callers may use networkx, the ROIF
Network class, or another backend without changing the validation API.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, median
from types import MappingProxyType
from typing import Any, Hashable, Protocol, TypeAlias, runtime_checkable
import math
import time

import numpy as np


class ValidationError(ValueError):
    """Raised when validation input or configuration is invalid."""


NodeId: TypeAlias = Hashable
TrialFactory: TypeAlias = Callable[["ValidationTrial"], Any]
TrialExecutor: TypeAlias = Callable[
    ["ValidationTrial", Any],
    "ValidationOutcome | Mapping[str, Any]",
]
DistanceFunction: TypeAlias = Callable[[NodeId, NodeId, Any], float]


_EPSILON = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a real number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a real number.") from exc

    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite.")

    if minimum is not None and result < minimum:
        raise ValidationError(
            f"{name} must be greater than or equal to {minimum}."
        )

    if maximum is not None and result > maximum:
        raise ValidationError(
            f"{name} must be less than or equal to {maximum}."
        )

    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(
            f"{name} must be a non-negative integer."
        )
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(
            f"{name} must be an integer greater than or equal to 1."
        )
    return value


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
        raise ValidationError(f"{name} must be a string{suffix}.")

    normalized = value.strip()
    if not normalized:
        raise ValidationError(f"{name} cannot be empty.")

    return normalized


def _normalize_node_id(
    value: Any,
    *,
    name: str = "node_id",
    optional: bool = False,
) -> NodeId | None:
    if value is None and optional:
        return None
    if value is None:
        raise ValidationError(f"{name} cannot be None.")

    try:
        hash(value)
    except TypeError as exc:
        raise ValidationError(f"{name} must be hashable.") from exc

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValidationError(f"{name} cannot be empty.")
        return normalized

    return value


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise ValidationError("metadata must be a mapping.")

    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized = _normalize_text(key, name="metadata key")
        copied[normalized] = value

    return MappingProxyType(copied)


def _normalize_ranking(
    ranking: Sequence[NodeId] | None,
    *,
    name: str,
) -> tuple[NodeId, ...]:
    if ranking is None:
        return ()

    if isinstance(ranking, (str, bytes)):
        raise ValidationError(
            f"{name} must be a sequence of node identifiers."
        )

    try:
        values = tuple(ranking)
    except TypeError as exc:
        raise ValidationError(
            f"{name} must be a sequence of node identifiers."
        ) from exc

    normalized: list[NodeId] = []
    seen: set[NodeId] = set()

    for value in values:
        node_id = _normalize_node_id(value, name=f"{name} node_id")
        assert node_id is not None
        if node_id in seen:
            raise ValidationError(
                f"{name} cannot contain duplicate node identifiers."
            )
        seen.add(node_id)
        normalized.append(node_id)

    return tuple(normalized)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(mean(values))


def _median_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(median(values))


def _std_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


class GraphFamily(str, Enum):
    """Synthetic graph family used in validation."""

    LINEAR_CHAIN = "linear_chain"
    TREE = "tree"
    LATTICE = "lattice"
    ERDOS_RENYI = "erdos_renyi"
    WATTS_STROGATZ = "watts_strogatz"
    BARABASI_ALBERT = "barabasi_albert"
    CUSTOM = "custom"


class TrialStatus(str, Enum):
    """Outcome of one validation trial."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationStatus(str, Enum):
    """Outcome of a validation run."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class MetricName(str, Enum):
    """Canonical validation metric identifiers."""

    ROOT_ACCURACY = "root_accuracy"
    ROOT_TOP_K_ACCURACY = "root_top_k_accuracy"
    ROOT_LOCALIZATION_DISTANCE = "root_localization_distance"
    NODE_STAR_ACCURACY = "node_star_accuracy"
    NODE_STAR_TOP_K_ACCURACY = "node_star_top_k_accuracy"
    NODE_STAR_GAIN_REGRET = "node_star_gain_regret"
    CONVERGENCE_RATE = "convergence_rate"
    SUCCESS_RATE = "success_rate"
    RUNTIME_SECONDS = "runtime_seconds"
    RUNTIME_PER_NODE = "runtime_per_node"


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """Serializable specification of one graph family and size."""

    family: GraphFamily
    node_count: int
    edge_count: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, GraphFamily):
            raise ValidationError(
                "family must be a GraphFamily."
            )

        object.__setattr__(
            self,
            "node_count",
            _positive_int(self.node_count, name="node_count"),
        )

        if self.edge_count is not None:
            object.__setattr__(
                self,
                "edge_count",
                _nonnegative_int(
                    self.edge_count,
                    name="edge_count",
                ),
            )

        object.__setattr__(
            self,
            "parameters",
            _freeze_metadata(self.parameters),
        )
        object.__setattr__(
            self,
            "name",
            _normalize_text(
                self.name,
                name="name",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def label(self) -> str:
        if self.name is not None:
            return self.name
        return f"{self.family.value}:{self.node_count}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "parameters": dict(self.parameters),
            "name": self.name,
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for a seeded validation run."""

    repetitions: int = 10
    base_seed: int = 0
    top_k: tuple[int, ...] = (1, 3, 5)
    capture_trial_errors: bool = True
    allow_partial_results: bool = True
    measure_runtime: bool = True
    minimum_success_rate: float = 0.0
    minimum_convergence_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repetitions",
            _positive_int(
                self.repetitions,
                name="repetitions",
            ),
        )

        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
        ):
            raise ValidationError("base_seed must be an integer.")

        if isinstance(self.top_k, (str, bytes)):
            raise ValidationError(
                "top_k must be a sequence of positive integers."
            )

        try:
            values = tuple(self.top_k)
        except TypeError as exc:
            raise ValidationError(
                "top_k must be a sequence of positive integers."
            ) from exc

        if not values:
            raise ValidationError("top_k cannot be empty.")

        normalized: list[int] = []
        seen: set[int] = set()

        for value in values:
            k = _positive_int(value, name="top_k value")
            if k in seen:
                raise ValidationError(
                    "top_k cannot contain duplicate values."
                )
            seen.add(k)
            normalized.append(k)

        object.__setattr__(
            self,
            "top_k",
            tuple(sorted(normalized)),
        )

        for name in (
            "capture_trial_errors",
            "allow_partial_results",
            "measure_runtime",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValidationError(f"{name} must be a bool.")

        for name in (
            "minimum_success_rate",
            "minimum_convergence_rate",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )


@dataclass(frozen=True, slots=True)
class ValidationTrial:
    """Deterministic specification of one validation trial."""

    trial_id: str
    repetition: int
    seed: int
    graph: GraphSpec
    true_root: NodeId | None = None
    true_node_star: NodeId | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trial_id",
            _normalize_text(self.trial_id, name="trial_id"),
        )
        object.__setattr__(
            self,
            "repetition",
            _nonnegative_int(
                self.repetition,
                name="repetition",
            ),
        )

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValidationError("seed must be an integer.")

        if not isinstance(self.graph, GraphSpec):
            raise ValidationError("graph must be a GraphSpec.")

        object.__setattr__(
            self,
            "true_root",
            _normalize_node_id(
                self.true_root,
                name="true_root",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "true_node_star",
            _normalize_node_id(
                self.true_node_star,
                name="true_node_star",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "repetition": self.repetition,
            "seed": self.seed,
            "graph": self.graph.as_dict(),
            "true_root": self.true_root,
            "true_node_star": self.true_node_star,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Observed output from one localization/optimization trial."""

    predicted_root: NodeId | None = None
    root_ranking: tuple[NodeId, ...] = ()
    root_localization_distance: float | None = None

    predicted_node_star: NodeId | None = None
    node_star_ranking: tuple[NodeId, ...] = ()
    selected_gain: float | None = None
    optimal_gain: float | None = None

    converged: bool = True
    runtime_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_root",
            _normalize_node_id(
                self.predicted_root,
                name="predicted_root",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "root_ranking",
            _normalize_ranking(
                self.root_ranking,
                name="root_ranking",
            ),
        )

        if self.root_localization_distance is not None:
            object.__setattr__(
                self,
                "root_localization_distance",
                _finite_float(
                    self.root_localization_distance,
                    name="root_localization_distance",
                    minimum=0.0,
                ),
            )

        object.__setattr__(
            self,
            "predicted_node_star",
            _normalize_node_id(
                self.predicted_node_star,
                name="predicted_node_star",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "node_star_ranking",
            _normalize_ranking(
                self.node_star_ranking,
                name="node_star_ranking",
            ),
        )

        for name in ("selected_gain", "optimal_gain"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_float(value, name=name),
                )

        if not isinstance(self.converged, bool):
            raise ValidationError("converged must be a bool.")

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

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def gain_regret(self) -> float | None:
        if self.selected_gain is None or self.optimal_gain is None:
            return None
        return self.optimal_gain - self.selected_gain

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_root": self.predicted_root,
            "root_ranking": list(self.root_ranking),
            "root_localization_distance": self.root_localization_distance,
            "predicted_node_star": self.predicted_node_star,
            "node_star_ranking": list(self.node_star_ranking),
            "selected_gain": self.selected_gain,
            "optimal_gain": self.optimal_gain,
            "gain_regret": self.gain_regret,
            "converged": self.converged,
            "runtime_seconds": self.runtime_seconds,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    """Per-trial validation metrics."""

    root_correct: bool | None
    root_top_k: Mapping[int, bool]
    root_localization_distance: float | None

    node_star_correct: bool | None
    node_star_top_k: Mapping[int, bool]
    node_star_gain_regret: float | None

    converged: bool
    runtime_seconds: float | None
    runtime_per_node: float | None

    def __post_init__(self) -> None:
        for name in ("root_correct", "node_star_correct"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValidationError(
                    f"{name} must be a bool or None."
                )

        for name in ("root_top_k", "node_star_top_k"):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping):
                raise ValidationError(f"{name} must be a mapping.")

            normalized: dict[int, bool] = {}
            for key, value in raw.items():
                k = _positive_int(key, name=f"{name} key")
                if not isinstance(value, bool):
                    raise ValidationError(
                        f"{name} values must be bool."
                    )
                normalized[k] = value

            object.__setattr__(
                self,
                name,
                MappingProxyType(normalized),
            )

        for name in (
            "root_localization_distance",
            "node_star_gain_regret",
            "runtime_seconds",
            "runtime_per_node",
        ):
            value = getattr(self, name)
            if value is not None:
                minimum = 0.0
                object.__setattr__(
                    self,
                    name,
                    _finite_float(
                        value,
                        name=name,
                        minimum=minimum,
                    ),
                )

        if not isinstance(self.converged, bool):
            raise ValidationError("converged must be a bool.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_correct": self.root_correct,
            "root_top_k": dict(self.root_top_k),
            "root_localization_distance": (
                self.root_localization_distance
            ),
            "node_star_correct": self.node_star_correct,
            "node_star_top_k": dict(self.node_star_top_k),
            "node_star_gain_regret": self.node_star_gain_regret,
            "converged": self.converged,
            "runtime_seconds": self.runtime_seconds,
            "runtime_per_node": self.runtime_per_node,
        }


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Complete result of one validation trial."""

    trial: ValidationTrial
    status: TrialStatus
    outcome: ValidationOutcome | None
    metrics: TrialMetrics | None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trial, ValidationTrial):
            raise ValidationError(
                "trial must be a ValidationTrial."
            )
        if not isinstance(self.status, TrialStatus):
            raise ValidationError(
                "status must be a TrialStatus."
            )

        if self.outcome is not None and not isinstance(
            self.outcome,
            ValidationOutcome,
        ):
            raise ValidationError(
                "outcome must be a ValidationOutcome or None."
            )

        if self.metrics is not None and not isinstance(
            self.metrics,
            TrialMetrics,
        ):
            raise ValidationError(
                "metrics must be TrialMetrics or None."
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

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

        if self.status is TrialStatus.SUCCEEDED:
            if self.outcome is None or self.metrics is None:
                raise ValidationError(
                    "Successful trial requires outcome and metrics."
                )
            if self.error_type is not None or self.error_message is not None:
                raise ValidationError(
                    "Successful trial cannot contain error details."
                )
        else:
            if self.outcome is not None or self.metrics is not None:
                raise ValidationError(
                    "Failed or skipped trial cannot contain outputs."
                )
            if (
                self.status is TrialStatus.FAILED
                and (
                    self.error_type is None
                    or self.error_message is None
                )
            ):
                raise ValidationError(
                    "Failed trial requires error details."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial.as_dict(),
            "status": self.status.value,
            "outcome": (
                None if self.outcome is None else self.outcome.as_dict()
            ),
            "metrics": (
                None if self.metrics is None else self.metrics.as_dict()
            ),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate statistics for one validation metric."""

    name: MetricName
    count: int
    mean: float | None
    median: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, MetricName):
            raise ValidationError("name must be a MetricName.")

        object.__setattr__(
            self,
            "count",
            _nonnegative_int(self.count, name="count"),
        )

        for field_name in (
            "mean",
            "median",
            "standard_deviation",
            "minimum",
            "maximum",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _finite_float(
                        value,
                        name=field_name,
                        minimum=(
                            0.0
                            if field_name == "standard_deviation"
                            else None
                        ),
                    ),
                )

        if self.count == 0:
            if any(
                getattr(self, field_name) is not None
                for field_name in (
                    "mean",
                    "median",
                    "standard_deviation",
                    "minimum",
                    "maximum",
                )
            ):
                raise ValidationError(
                    "Empty metric summary cannot contain statistics."
                )
        else:
            if any(
                getattr(self, field_name) is None
                for field_name in (
                    "mean",
                    "median",
                    "standard_deviation",
                    "minimum",
                    "maximum",
                )
            ):
                raise ValidationError(
                    "Non-empty metric summary requires all statistics."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class GroupSummary:
    """Summary for one graph family and node count."""

    graph_family: GraphFamily
    node_count: int
    trial_count: int
    successful_count: int
    failed_count: int
    metrics: Mapping[str, MetricSummary]

    def __post_init__(self) -> None:
        if not isinstance(self.graph_family, GraphFamily):
            raise ValidationError(
                "graph_family must be a GraphFamily."
            )

        object.__setattr__(
            self,
            "node_count",
            _positive_int(self.node_count, name="node_count"),
        )
        object.__setattr__(
            self,
            "trial_count",
            _nonnegative_int(self.trial_count, name="trial_count"),
        )
        object.__setattr__(
            self,
            "successful_count",
            _nonnegative_int(
                self.successful_count,
                name="successful_count",
            ),
        )
        object.__setattr__(
            self,
            "failed_count",
            _nonnegative_int(
                self.failed_count,
                name="failed_count",
            ),
        )

        if (
            self.successful_count + self.failed_count
            != self.trial_count
        ):
            raise ValidationError(
                "group counts must equal trial_count."
            )

        if not isinstance(self.metrics, Mapping):
            raise ValidationError("metrics must be a mapping.")

        normalized: dict[str, MetricSummary] = {}
        for key, value in self.metrics.items():
            metric_name = _normalize_text(
                key,
                name="metric key",
            )
            if not isinstance(value, MetricSummary):
                raise ValidationError(
                    "metrics values must be MetricSummary."
                )
            normalized[metric_name] = value

        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(normalized),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_family": self.graph_family.value,
            "node_count": self.node_count,
            "trial_count": self.trial_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "metrics": {
                key: value.as_dict()
                for key, value in self.metrics.items()
            },
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Complete reproducible validation run."""

    status: ValidationStatus
    config: ValidationConfig
    graph_specs: tuple[GraphSpec, ...]
    trials: tuple[TrialResult, ...]
    metrics: Mapping[str, MetricSummary]
    groups: tuple[GroupSummary, ...]
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ValidationStatus):
            raise ValidationError(
                "status must be a ValidationStatus."
            )
        if not isinstance(self.config, ValidationConfig):
            raise ValidationError(
                "config must be a ValidationConfig."
            )

        graph_specs = tuple(self.graph_specs)
        if any(not isinstance(item, GraphSpec) for item in graph_specs):
            raise ValidationError(
                "graph_specs must contain GraphSpec values."
            )
        object.__setattr__(self, "graph_specs", graph_specs)

        trials = tuple(self.trials)
        if any(not isinstance(item, TrialResult) for item in trials):
            raise ValidationError(
                "trials must contain TrialResult values."
            )
        object.__setattr__(self, "trials", trials)

        normalized_metrics: dict[str, MetricSummary] = {}
        for key, value in self.metrics.items():
            metric_name = _normalize_text(
                key,
                name="metric key",
            )
            if not isinstance(value, MetricSummary):
                raise ValidationError(
                    "metrics values must be MetricSummary."
                )
            normalized_metrics[metric_name] = value

        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(normalized_metrics),
        )

        groups = tuple(self.groups)
        if any(not isinstance(item, GroupSummary) for item in groups):
            raise ValidationError(
                "groups must contain GroupSummary values."
            )
        object.__setattr__(self, "groups", groups)

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
    def successful_count(self) -> int:
        return sum(
            item.status is TrialStatus.SUCCEEDED
            for item in self.trials
        )

    @property
    def failed_count(self) -> int:
        return sum(
            item.status is TrialStatus.FAILED
            for item in self.trials
        )

    @property
    def skipped_count(self) -> int:
        return sum(
            item.status is TrialStatus.SKIPPED
            for item in self.trials
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "config": {
                "repetitions": self.config.repetitions,
                "base_seed": self.config.base_seed,
                "top_k": list(self.config.top_k),
                "capture_trial_errors": (
                    self.config.capture_trial_errors
                ),
                "allow_partial_results": (
                    self.config.allow_partial_results
                ),
                "measure_runtime": self.config.measure_runtime,
                "minimum_success_rate": (
                    self.config.minimum_success_rate
                ),
                "minimum_convergence_rate": (
                    self.config.minimum_convergence_rate
                ),
            },
            "graph_specs": [
                item.as_dict() for item in self.graph_specs
            ],
            "trials": [item.as_dict() for item in self.trials],
            "metrics": {
                key: value.as_dict()
                for key, value in self.metrics.items()
            },
            "groups": [item.as_dict() for item in self.groups],
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ValidationTrialFactory(Protocol):
    """Factory contract for building a trial fixture."""

    def __call__(self, trial: ValidationTrial) -> Any:
        ...


@runtime_checkable
class ValidationTrialExecutor(Protocol):
    """Executor contract for one validation trial."""

    def __call__(
        self,
        trial: ValidationTrial,
        fixture: Any,
    ) -> ValidationOutcome | Mapping[str, Any]:
        ...


def derive_seed(
    base_seed: int,
    graph_index: int,
    repetition: int,
) -> int:
    """Derive a deterministic 32-bit seed for one trial."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValidationError("base_seed must be an integer.")

    graph_position = _nonnegative_int(
        graph_index,
        name="graph_index",
    )
    repeat = _nonnegative_int(
        repetition,
        name="repetition",
    )

    sequence = np.random.SeedSequence(
        [base_seed, graph_position, repeat]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def make_trials(
    graph_specs: Sequence[GraphSpec],
    *,
    config: ValidationConfig | None = None,
    true_root_factory: Callable[[GraphSpec, int, int], NodeId | None]
    | None = None,
    true_node_star_factory: Callable[
        [GraphSpec, int, int],
        NodeId | None,
    ]
    | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ValidationTrial, ...]:
    """Build deterministic trial specifications."""

    if config is None:
        config = ValidationConfig()

    if not isinstance(config, ValidationConfig):
        raise ValidationError(
            "config must be a ValidationConfig."
        )

    specs = tuple(graph_specs)
    if not specs:
        raise ValidationError("graph_specs cannot be empty.")
    if any(not isinstance(item, GraphSpec) for item in specs):
        raise ValidationError(
            "graph_specs must contain GraphSpec values."
        )

    for name, callback in (
        ("true_root_factory", true_root_factory),
        ("true_node_star_factory", true_node_star_factory),
    ):
        if callback is not None and not callable(callback):
            raise ValidationError(
                f"{name} must be callable or None."
            )

    trials: list[ValidationTrial] = []

    for graph_index, spec in enumerate(specs):
        for repetition in range(config.repetitions):
            seed = derive_seed(
                config.base_seed,
                graph_index,
                repetition,
            )
            true_root = (
                None
                if true_root_factory is None
                else true_root_factory(spec, repetition, seed)
            )
            true_node_star = (
                None
                if true_node_star_factory is None
                else true_node_star_factory(
                    spec,
                    repetition,
                    seed,
                )
            )

            trial_id = (
                f"{graph_index:03d}-"
                f"{spec.family.value}-"
                f"n{spec.node_count}-"
                f"r{repetition:04d}"
            )

            trials.append(
                ValidationTrial(
                    trial_id=trial_id,
                    repetition=repetition,
                    seed=seed,
                    graph=spec,
                    true_root=true_root,
                    true_node_star=true_node_star,
                    metadata=metadata,
                )
            )

    return tuple(trials)


def _coerce_outcome(
    value: ValidationOutcome | Mapping[str, Any],
) -> ValidationOutcome:
    if isinstance(value, ValidationOutcome):
        return value

    if not isinstance(value, Mapping):
        raise ValidationError(
            "trial executor must return ValidationOutcome or mapping."
        )

    allowed = {
        "predicted_root",
        "root_ranking",
        "root_localization_distance",
        "predicted_node_star",
        "node_star_ranking",
        "selected_gain",
        "optimal_gain",
        "converged",
        "runtime_seconds",
        "metadata",
    }
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise ValidationError(
            f"Outcome mapping contains unknown fields: {names}."
        )

    return ValidationOutcome(**dict(value))


def compute_trial_metrics(
    trial: ValidationTrial,
    outcome: ValidationOutcome,
    *,
    top_k: Sequence[int] = (1, 3, 5),
    distance: DistanceFunction | None = None,
    fixture: Any = None,
) -> TrialMetrics:
    """Compute canonical metrics for one successful trial."""

    if not isinstance(trial, ValidationTrial):
        raise ValidationError(
            "trial must be a ValidationTrial."
        )
    if not isinstance(outcome, ValidationOutcome):
        raise ValidationError(
            "outcome must be a ValidationOutcome."
        )

    ks = tuple(_positive_int(k, name="top_k value") for k in top_k)

    root_correct: bool | None = None
    root_top: dict[int, bool] = {}
    root_distance = outcome.root_localization_distance

    if trial.true_root is not None:
        root_correct = outcome.predicted_root == trial.true_root
        ranking = outcome.root_ranking
        if not ranking and outcome.predicted_root is not None:
            ranking = (outcome.predicted_root,)

        for k in ks:
            root_top[k] = trial.true_root in ranking[:k]

        if root_distance is None and outcome.predicted_root is not None:
            if distance is None:
                root_distance = (
                    0.0
                    if outcome.predicted_root == trial.true_root
                    else 1.0
                )
            else:
                root_distance = _finite_float(
                    distance(
                        trial.true_root,
                        outcome.predicted_root,
                        fixture,
                    ),
                    name="localization distance",
                    minimum=0.0,
                )

    node_correct: bool | None = None
    node_top: dict[int, bool] = {}

    if trial.true_node_star is not None:
        node_correct = (
            outcome.predicted_node_star
            == trial.true_node_star
        )
        ranking = outcome.node_star_ranking
        if (
            not ranking
            and outcome.predicted_node_star is not None
        ):
            ranking = (outcome.predicted_node_star,)

        for k in ks:
            node_top[k] = trial.true_node_star in ranking[:k]

    runtime_per_node = (
        None
        if outcome.runtime_seconds is None
        else outcome.runtime_seconds / trial.graph.node_count
    )

    regret = outcome.gain_regret
    if regret is not None and regret < 0.0 and abs(regret) <= 1e-9:
        regret = 0.0
    if regret is not None and regret < 0.0:
        raise ValidationError(
            "selected_gain cannot exceed optimal_gain."
        )

    return TrialMetrics(
        root_correct=root_correct,
        root_top_k=root_top,
        root_localization_distance=root_distance,
        node_star_correct=node_correct,
        node_star_top_k=node_top,
        node_star_gain_regret=regret,
        converged=outcome.converged,
        runtime_seconds=outcome.runtime_seconds,
        runtime_per_node=runtime_per_node,
    )


def summarize_metric(
    name: MetricName,
    values: Iterable[float],
) -> MetricSummary:
    """Summarize one numeric validation metric."""

    if not isinstance(name, MetricName):
        raise ValidationError("name must be a MetricName.")

    parsed = tuple(
        _finite_float(value, name=f"{name.value} value")
        for value in values
    )

    if not parsed:
        return MetricSummary(
            name=name,
            count=0,
            mean=None,
            median=None,
            standard_deviation=None,
            minimum=None,
            maximum=None,
        )

    return MetricSummary(
        name=name,
        count=len(parsed),
        mean=_mean_or_none(parsed),
        median=_median_or_none(parsed),
        standard_deviation=_std_or_none(parsed),
        minimum=min(parsed),
        maximum=max(parsed),
    )


def _collect_metric_values(
    trials: Sequence[TrialResult],
    *,
    top_k: Sequence[int],
) -> dict[str, tuple[MetricName, list[float]]]:
    values: dict[str, tuple[MetricName, list[float]]] = {
        MetricName.ROOT_ACCURACY.value: (
            MetricName.ROOT_ACCURACY,
            [],
        ),
        MetricName.ROOT_LOCALIZATION_DISTANCE.value: (
            MetricName.ROOT_LOCALIZATION_DISTANCE,
            [],
        ),
        MetricName.NODE_STAR_ACCURACY.value: (
            MetricName.NODE_STAR_ACCURACY,
            [],
        ),
        MetricName.NODE_STAR_GAIN_REGRET.value: (
            MetricName.NODE_STAR_GAIN_REGRET,
            [],
        ),
        MetricName.CONVERGENCE_RATE.value: (
            MetricName.CONVERGENCE_RATE,
            [],
        ),
        MetricName.RUNTIME_SECONDS.value: (
            MetricName.RUNTIME_SECONDS,
            [],
        ),
        MetricName.RUNTIME_PER_NODE.value: (
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

    for result in trials:
        if (
            result.status is not TrialStatus.SUCCEEDED
            or result.metrics is None
        ):
            continue

        metrics = result.metrics

        if metrics.root_correct is not None:
            values[MetricName.ROOT_ACCURACY.value][1].append(
                float(metrics.root_correct)
            )

        if metrics.root_localization_distance is not None:
            values[
                MetricName.ROOT_LOCALIZATION_DISTANCE.value
            ][1].append(metrics.root_localization_distance)

        for k, correct in metrics.root_top_k.items():
            key = f"root_top_{k}_accuracy"
            if key in values:
                values[key][1].append(float(correct))

        if metrics.node_star_correct is not None:
            values[
                MetricName.NODE_STAR_ACCURACY.value
            ][1].append(float(metrics.node_star_correct))

        for k, correct in metrics.node_star_top_k.items():
            key = f"node_star_top_{k}_accuracy"
            if key in values:
                values[key][1].append(float(correct))

        if metrics.node_star_gain_regret is not None:
            values[
                MetricName.NODE_STAR_GAIN_REGRET.value
            ][1].append(metrics.node_star_gain_regret)

        values[MetricName.CONVERGENCE_RATE.value][1].append(
            float(metrics.converged)
        )

        if metrics.runtime_seconds is not None:
            values[MetricName.RUNTIME_SECONDS.value][1].append(
                metrics.runtime_seconds
            )

        if metrics.runtime_per_node is not None:
            values[MetricName.RUNTIME_PER_NODE.value][1].append(
                metrics.runtime_per_node
            )

    success_rate = (
        sum(
            result.status is TrialStatus.SUCCEEDED
            for result in trials
        )
        / len(trials)
        if trials
        else 0.0
    )
    values[MetricName.SUCCESS_RATE.value] = (
        MetricName.SUCCESS_RATE,
        [success_rate],
    )

    return values


def summarize_trials(
    trials: Sequence[TrialResult],
    *,
    top_k: Sequence[int] = (1, 3, 5),
) -> Mapping[str, MetricSummary]:
    """Build aggregate summaries for trial results."""

    parsed = tuple(trials)
    if any(not isinstance(item, TrialResult) for item in parsed):
        raise ValidationError(
            "trials must contain TrialResult values."
        )

    values = _collect_metric_values(parsed, top_k=top_k)

    return MappingProxyType(
        {
            key: summarize_metric(metric_name, metric_values)
            for key, (
                metric_name,
                metric_values,
            ) in values.items()
        }
    )


def group_trial_summaries(
    trials: Sequence[TrialResult],
    *,
    top_k: Sequence[int] = (1, 3, 5),
) -> tuple[GroupSummary, ...]:
    """Group validation results by graph family and node count."""

    grouped: dict[
        tuple[GraphFamily, int],
        list[TrialResult],
    ] = defaultdict(list)

    for result in trials:
        if not isinstance(result, TrialResult):
            raise ValidationError(
                "trials must contain TrialResult values."
            )
        grouped[
            (
                result.trial.graph.family,
                result.trial.graph.node_count,
            )
        ].append(result)

    summaries: list[GroupSummary] = []

    for (family, node_count), items in sorted(
        grouped.items(),
        key=lambda pair: (
            pair[0][0].value,
            pair[0][1],
        ),
    ):
        successful = sum(
            item.status is TrialStatus.SUCCEEDED
            for item in items
        )
        failed = len(items) - successful

        summaries.append(
            GroupSummary(
                graph_family=family,
                node_count=node_count,
                trial_count=len(items),
                successful_count=successful,
                failed_count=failed,
                metrics=summarize_trials(
                    items,
                    top_k=top_k,
                ),
            )
        )

    return tuple(summaries)


class ValidationRunner:
    """Execute reproducible seeded validation experiments."""

    def __init__(
        self,
        config: ValidationConfig | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config is None:
            config = ValidationConfig()

        if not isinstance(config, ValidationConfig):
            raise ValidationError(
                "config must be a ValidationConfig."
            )
        if not callable(clock):
            raise ValidationError("clock must be callable.")

        self._config = config
        self._clock = clock

    @property
    def config(self) -> ValidationConfig:
        return self._config

    def run(
        self,
        graph_specs: Sequence[GraphSpec],
        *,
        trial_factory: TrialFactory,
        execute_trial: TrialExecutor,
        true_root_factory: Callable[
            [GraphSpec, int, int],
            NodeId | None,
        ]
        | None = None,
        true_node_star_factory: Callable[
            [GraphSpec, int, int],
            NodeId | None,
        ]
        | None = None,
        distance: DistanceFunction | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        if not callable(trial_factory):
            raise ValidationError(
                "trial_factory must be callable."
            )
        if not callable(execute_trial):
            raise ValidationError(
                "execute_trial must be callable."
            )
        if distance is not None and not callable(distance):
            raise ValidationError(
                "distance must be callable or None."
            )

        specs = tuple(graph_specs)
        trials = make_trials(
            specs,
            config=self.config,
            true_root_factory=true_root_factory,
            true_node_star_factory=true_node_star_factory,
            metadata=metadata,
        )

        results: list[TrialResult] = []

        for trial in trials:
            try:
                fixture = trial_factory(trial)

                started = (
                    self._clock()
                    if self.config.measure_runtime
                    else None
                )
                raw_outcome = execute_trial(trial, fixture)
                finished = (
                    self._clock()
                    if self.config.measure_runtime
                    else None
                )

                outcome = _coerce_outcome(raw_outcome)

                if (
                    self.config.measure_runtime
                    and outcome.runtime_seconds is None
                ):
                    assert started is not None
                    assert finished is not None
                    measured = _finite_float(
                        finished - started,
                        name="measured runtime",
                        minimum=0.0,
                    )
                    outcome = ValidationOutcome(
                        predicted_root=outcome.predicted_root,
                        root_ranking=outcome.root_ranking,
                        root_localization_distance=(
                            outcome.root_localization_distance
                        ),
                        predicted_node_star=(
                            outcome.predicted_node_star
                        ),
                        node_star_ranking=(
                            outcome.node_star_ranking
                        ),
                        selected_gain=outcome.selected_gain,
                        optimal_gain=outcome.optimal_gain,
                        converged=outcome.converged,
                        runtime_seconds=measured,
                        metadata=outcome.metadata,
                    )

                metrics = compute_trial_metrics(
                    trial,
                    outcome,
                    top_k=self.config.top_k,
                    distance=distance,
                    fixture=fixture,
                )

                results.append(
                    TrialResult(
                        trial=trial,
                        status=TrialStatus.SUCCEEDED,
                        outcome=outcome,
                        metrics=metrics,
                    )
                )

            except Exception as exc:
                if not self.config.capture_trial_errors:
                    raise

                results.append(
                    TrialResult(
                        trial=trial,
                        status=TrialStatus.FAILED,
                        outcome=None,
                        metrics=None,
                        error_type=type(exc).__name__,
                        error_message=(
                            str(exc) or type(exc).__name__
                        ),
                    )
                )

        aggregate = summarize_trials(
            results,
            top_k=self.config.top_k,
        )
        groups = group_trial_summaries(
            results,
            top_k=self.config.top_k,
        )

        successful = sum(
            item.status is TrialStatus.SUCCEEDED
            for item in results
        )
        total = len(results)
        success_rate = successful / total if total else 0.0

        convergence_values = [
            float(item.metrics.converged)
            for item in results
            if (
                item.status is TrialStatus.SUCCEEDED
                and item.metrics is not None
            )
        ]
        convergence_rate = (
            float(mean(convergence_values))
            if convergence_values
            else 0.0
        )

        if successful == 0:
            status = ValidationStatus.FAILED
        elif (
            success_rate < self.config.minimum_success_rate
            or convergence_rate
            < self.config.minimum_convergence_rate
        ):
            status = ValidationStatus.FAILED
        elif successful < total:
            status = (
                ValidationStatus.PARTIAL
                if self.config.allow_partial_results
                else ValidationStatus.FAILED
            )
        else:
            status = ValidationStatus.COMPLETED

        return ValidationResult(
            status=status,
            config=self.config,
            graph_specs=specs,
            trials=tuple(results),
            metrics=aggregate,
            groups=groups,
            summary=_validation_summary(
                status=status,
                total=total,
                successful=successful,
                failed=total - successful,
                success_rate=success_rate,
                convergence_rate=convergence_rate,
            ),
            metadata=metadata,
        )


def _validation_summary(
    *,
    status: ValidationStatus,
    total: int,
    successful: int,
    failed: int,
    success_rate: float,
    convergence_rate: float,
) -> str:
    return (
        f"Validation {status.value}: "
        f"{successful}/{total} trials succeeded, "
        f"{failed} failed; "
        f"success rate = {success_rate:.3f}; "
        f"convergence rate = {convergence_rate:.3f}."
    )


def run_validation(
    graph_specs: Sequence[GraphSpec],
    *,
    trial_factory: TrialFactory,
    execute_trial: TrialExecutor,
    config: ValidationConfig | None = None,
    true_root_factory: Callable[
        [GraphSpec, int, int],
        NodeId | None,
    ]
    | None = None,
    true_node_star_factory: Callable[
        [GraphSpec, int, int],
        NodeId | None,
    ]
    | None = None,
    distance: DistanceFunction | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ValidationResult:
    """Convenience wrapper around ``ValidationRunner.run``."""

    return ValidationRunner(config).run(
        graph_specs,
        trial_factory=trial_factory,
        execute_trial=execute_trial,
        true_root_factory=true_root_factory,
        true_node_star_factory=true_node_star_factory,
        distance=distance,
        metadata=metadata,
    )


__all__ = [
    "DistanceFunction",
    "GraphFamily",
    "GraphSpec",
    "GroupSummary",
    "MetricName",
    "MetricSummary",
    "NodeId",
    "TrialExecutor",
    "TrialFactory",
    "TrialMetrics",
    "TrialResult",
    "TrialStatus",
    "ValidationConfig",
    "ValidationError",
    "ValidationOutcome",
    "ValidationResult",
    "ValidationRunner",
    "ValidationStatus",
    "ValidationTrial",
    "ValidationTrialExecutor",
    "ValidationTrialFactory",
    "compute_trial_metrics",
    "derive_seed",
    "group_trial_summaries",
    "make_trials",
    "run_validation",
    "summarize_metric",
    "summarize_trials",
]
