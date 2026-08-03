from __future__ import annotations

"""
Counterfactual intervention analysis for the ROIF Structural Memory Engine.

HistoryForecaster answers:

    Which future structural transition is most consistent with the current
    signature and the supplied influence planes?

CounterfactualEngine answers:

    Which admissible intervention produces the greatest predicted improvement
    relative to the no-intervention forecast?

The engine does not invent interventions and does not modify a real system.
Every intervention scenario must provide a forward-calculated
HistoryForecastResult produced by an explicit model.

Scientific and safety boundary
------------------------------
This module compares model outputs. It does not prove that an intervention
will work in a real biological, mechanical, social, environmental, or
infrastructure system.

The Non-Fonit Gate is explicit. Interventions marked as unsafe, system-wide,
poorly bounded, or cascade-amplifying can be vetoed before ranking.

The first implementation supports:

- explicit intervention candidates;
- node-level intervention targets;
- baseline-versus-intervention forecast comparison;
- Capacity-loss reduction;
- Capacity-gain increase;
- net-Capacity improvement;
- forecast confidence and validation;
- intervention cost, risk, and complexity penalties;
- deterministic ranking;
- Node* selection;
- ambiguity and confidence indicators;
- immutable serialization-ready data structures.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import exp, isfinite
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .history_forecast import (
    HistoryForecastResult,
)


class CounterfactualError(ValueError):
    """Raised when counterfactual inputs or results are invalid."""


class InterventionStatus(str, Enum):
    """Computational or governance status of an intervention."""

    READY = "ready"
    FORWARD_VALIDATED = "forward_validated"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    FAILED = "failed"
    VETOED = "vetoed"


class CounterfactualStatus(str, Enum):
    """Overall status of one intervention-ranking run."""

    COMPLETED = "completed"
    AMBIGUOUS = "ambiguous"
    NO_SCENARIOS = "no_scenarios"
    NO_VALID_SCENARIOS = "no_valid_scenarios"


class InterventionScope(str, Enum):
    """Declared physical or logical scope of an intervention."""

    NODE = "node"
    EDGE = "edge"
    SUBSYSTEM = "subsystem"
    SYSTEM = "system"
    EXTERNAL_PLANE = "external_plane"


def _require_text(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise CounterfactualError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise CounterfactualError(
            f"{field_name} must not be empty"
        )

    return normalized


def _finite(
    value: float,
    *,
    field_name: str,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CounterfactualError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise CounterfactualError(
            f"{field_name} must be finite"
        )

    return numeric


def _nonnegative(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _finite(
        value,
        field_name=field_name,
    )

    if numeric < 0.0:
        raise CounterfactualError(
            f"{field_name} must be non-negative"
        )

    return numeric


def _positive(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _finite(
        value,
        field_name=field_name,
    )

    if numeric <= 0.0:
        raise CounterfactualError(
            f"{field_name} must be greater than zero"
        )

    return numeric


def _unit(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _nonnegative(
        value,
        field_name=field_name,
    )

    if numeric > 1.0:
        raise CounterfactualError(
            f"{field_name} must be in [0, 1]"
        )

    return numeric


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise CounterfactualError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, Any] = {}

    for raw_key, item in value.items():
        key = _require_text(
            raw_key,
            field_name=f"{field_name} key",
        )
        normalized[key] = item

    return MappingProxyType(
        dict(sorted(normalized.items()))
    )


def _normalize_ids(
    values: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_value in values:
        value = _require_text(
            raw_value,
            field_name=field_name,
        )

        if value not in seen:
            seen.add(value)
            normalized.append(value)

    return tuple(normalized)


def _relative_change(
    before: float,
    after: float,
) -> float:
    """
    Return a signed relative change clipped to [-1, 1].

    Positive values mean improvement when the caller supplies the variables
    in the appropriate orientation.
    """

    denominator = max(
        abs(before),
        abs(after),
        1e-12,
    )

    value = (
        after - before
    ) / denominator

    return max(
        min(value, 1.0),
        -1.0,
    )


def _benefit_from_signed_change(
    signed_change: float,
) -> float:
    """Map a signed improvement in [-1, 1] to [0, 1]."""

    return (
        max(
            min(signed_change, 1.0),
            -1.0,
        )
        + 1.0
    ) / 2.0


@dataclass(frozen=True, slots=True)
class Intervention:
    """
    One explicitly defined candidate intervention.

    non_fonit_allowed:
        Explicit safety gate. False means the intervention is vetoed.

    cascade_risk:
        Normalized estimate of the risk that the intervention amplifies or
        propagates an undesirable cascade.

    reversibility:
        Normalized reversibility estimate. One means fully reversible within
        the modeled scope; zero means effectively irreversible.
    """

    name: str
    target_id: str
    scope: InterventionScope = InterventionScope.NODE

    description: str | None = None
    affected_plane_ids: tuple[str, ...] = ()
    affected_target_ids: tuple[str, ...] = ()

    magnitude: float = 1.0
    cost: float = 0.0
    operational_risk: float = 0.0
    cascade_risk: float = 0.0
    reversibility: float = 1.0
    complexity: float = 0.0

    non_fonit_allowed: bool = True
    status: InterventionStatus = InterventionStatus.READY

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    intervention_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        name = _require_text(
            self.name,
            field_name="name",
        )
        target_id = _require_text(
            self.target_id,
            field_name="target_id",
        )
        intervention_id = _require_text(
            self.intervention_id,
            field_name="intervention_id",
        )

        try:
            scope = InterventionScope(
                self.scope
            )
        except (TypeError, ValueError) as exc:
            raise CounterfactualError(
                f"unsupported intervention scope: {self.scope!r}"
            ) from exc

        try:
            status = InterventionStatus(
                self.status
            )
        except (TypeError, ValueError) as exc:
            raise CounterfactualError(
                f"unsupported intervention status: {self.status!r}"
            ) from exc

        description = self.description

        if description is not None:
            description = _require_text(
                description,
                field_name="description",
            )

        affected_plane_ids = _normalize_ids(
            self.affected_plane_ids,
            field_name="affected_plane_ids",
        )
        affected_target_ids = _normalize_ids(
            self.affected_target_ids,
            field_name="affected_target_ids",
        )

        magnitude = _nonnegative(
            self.magnitude,
            field_name="magnitude",
        )
        cost = _nonnegative(
            self.cost,
            field_name="cost",
        )
        operational_risk = _unit(
            self.operational_risk,
            field_name="operational_risk",
        )
        cascade_risk = _unit(
            self.cascade_risk,
            field_name="cascade_risk",
        )
        reversibility = _unit(
            self.reversibility,
            field_name="reversibility",
        )
        complexity = _nonnegative(
            self.complexity,
            field_name="complexity",
        )

        if not isinstance(
            self.non_fonit_allowed,
            bool,
        ):
            raise CounterfactualError(
                "non_fonit_allowed must be a boolean"
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(
            self,
            "intervention_id",
            intervention_id,
        )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "description",
            description,
        )
        object.__setattr__(
            self,
            "affected_plane_ids",
            affected_plane_ids,
        )
        object.__setattr__(
            self,
            "affected_target_ids",
            affected_target_ids,
        )
        object.__setattr__(
            self,
            "magnitude",
            magnitude,
        )
        object.__setattr__(self, "cost", cost)
        object.__setattr__(
            self,
            "operational_risk",
            operational_risk,
        )
        object.__setattr__(
            self,
            "cascade_risk",
            cascade_risk,
        )
        object.__setattr__(
            self,
            "reversibility",
            reversibility,
        )
        object.__setattr__(
            self,
            "complexity",
            complexity,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @property
    def is_eligible(self) -> bool:
        return (
            self.non_fonit_allowed
            and self.status
            in {
                InterventionStatus.READY,
                InterventionStatus.FORWARD_VALIDATED,
                InterventionStatus.INCOMPLETE,
            }
        )

    @property
    def safety_penalty(self) -> float:
        """
        Combined normalized safety penalty.

        Irreversibility contributes through 1 - reversibility.
        """

        return min(
            (
                self.operational_risk
                + self.cascade_risk
                + (1.0 - self.reversibility)
            )
            / 3.0,
            1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "intervention_id": self.intervention_id,
            "name": self.name,
            "target_id": self.target_id,
            "scope": self.scope.value,
            "description": self.description,
            "affected_plane_ids": list(
                self.affected_plane_ids
            ),
            "affected_target_ids": list(
                self.affected_target_ids
            ),
            "magnitude": self.magnitude,
            "cost": self.cost,
            "operational_risk": self.operational_risk,
            "cascade_risk": self.cascade_risk,
            "reversibility": self.reversibility,
            "complexity": self.complexity,
            "non_fonit_allowed": (
                self.non_fonit_allowed
            ),
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> Intervention:
        if not isinstance(data, Mapping):
            raise CounterfactualError(
                "intervention data must be a mapping"
            )

        version = data.get(
            "version",
            cls.VERSION,
        )

        if version != cls.VERSION:
            raise CounterfactualError(
                "unsupported intervention version: "
                f"{version!r}"
            )

        return cls(
            intervention_id=str(
                data["intervention_id"]
            ),
            name=str(data["name"]),
            target_id=str(data["target_id"]),
            scope=InterventionScope(
                data.get(
                    "scope",
                    InterventionScope.NODE.value,
                )
            ),
            description=data.get("description"),
            affected_plane_ids=tuple(
                data.get(
                    "affected_plane_ids",
                    (),
                )
            ),
            affected_target_ids=tuple(
                data.get(
                    "affected_target_ids",
                    (),
                )
            ),
            magnitude=float(
                data.get("magnitude", 1.0)
            ),
            cost=float(
                data.get("cost", 0.0)
            ),
            operational_risk=float(
                data.get(
                    "operational_risk",
                    0.0,
                )
            ),
            cascade_risk=float(
                data.get(
                    "cascade_risk",
                    0.0,
                )
            ),
            reversibility=float(
                data.get(
                    "reversibility",
                    1.0,
                )
            ),
            complexity=float(
                data.get(
                    "complexity",
                    0.0,
                )
            ),
            non_fonit_allowed=bool(
                data.get(
                    "non_fonit_allowed",
                    True,
                )
            ),
            status=InterventionStatus(
                data.get(
                    "status",
                    InterventionStatus.READY.value,
                )
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualScenario:
    """
    One intervention plus its forward-calculated forecast result.
    """

    intervention: Intervention
    forecast_result: HistoryForecastResult

    validation_score: float = 1.0
    model_penalty: float = 0.0
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    scenario_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(
            self.intervention,
            Intervention,
        ):
            raise CounterfactualError(
                "intervention must be an Intervention"
            )

        if not isinstance(
            self.forecast_result,
            HistoryForecastResult,
        ):
            raise CounterfactualError(
                "forecast_result must be a HistoryForecastResult"
            )

        scenario_id = _require_text(
            self.scenario_id,
            field_name="scenario_id",
        )
        validation_score = _unit(
            self.validation_score,
            field_name="validation_score",
        )
        model_penalty = _unit(
            self.model_penalty,
            field_name="model_penalty",
        )
        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(
            self,
            "scenario_id",
            scenario_id,
        )
        object.__setattr__(
            self,
            "validation_score",
            validation_score,
        )
        object.__setattr__(
            self,
            "model_penalty",
            model_penalty,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @property
    def is_eligible(self) -> bool:
        return (
            self.intervention.is_eligible
            and self.forecast_result.best is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "scenario_id": self.scenario_id,
            "intervention": (
                self.intervention.to_dict()
            ),
            "forecast_result": (
                self.forecast_result.to_dict()
            ),
            "validation_score": (
                self.validation_score
            ),
            "model_penalty": self.model_penalty,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualWeights:
    """
    Weights used to rank intervention outcomes.
    """

    capacity_benefit: float = 1.0
    loss_reduction: float = 0.75
    gain_increase: float = 0.50
    forecast_quality: float = 0.50
    validation: float = 0.50

    cost: float = 0.30
    safety: float = 0.75
    complexity: float = 0.20
    model_penalty: float = 0.30

    def __post_init__(self) -> None:
        for field_name in (
            "capacity_benefit",
            "loss_reduction",
            "gain_increase",
            "forecast_quality",
            "validation",
            "cost",
            "safety",
            "complexity",
            "model_penalty",
        ):
            value = _nonnegative(
                getattr(self, field_name),
                field_name=field_name,
            )
            object.__setattr__(
                self,
                field_name,
                value,
            )

        if self.positive_weight <= 0.0:
            raise CounterfactualError(
                "at least one positive benefit weight is required"
            )

    @property
    def positive_weight(self) -> float:
        return (
            self.capacity_benefit
            + self.loss_reduction
            + self.gain_increase
            + self.forecast_quality
            + self.validation
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "capacity_benefit": (
                self.capacity_benefit
            ),
            "loss_reduction": (
                self.loss_reduction
            ),
            "gain_increase": (
                self.gain_increase
            ),
            "forecast_quality": (
                self.forecast_quality
            ),
            "validation": self.validation,
            "cost": self.cost,
            "safety": self.safety,
            "complexity": self.complexity,
            "model_penalty": (
                self.model_penalty
            ),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualScore:
    """
    Scored and ranked intervention outcome.
    """

    scenario: CounterfactualScenario

    baseline_net_capacity: float
    intervention_net_capacity: float

    net_capacity_delta: float
    capacity_benefit_score: float
    loss_reduction_score: float
    gain_increase_score: float
    forecast_quality_score: float
    validation_score: float

    cost_penalty: float
    safety_penalty: float
    complexity_penalty: float
    model_penalty: float

    raw_score: float
    normalized_score: float
    rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(
            self.scenario,
            CounterfactualScenario,
        ):
            raise CounterfactualError(
                "scenario must be a CounterfactualScenario"
            )

        object.__setattr__(
            self,
            "baseline_net_capacity",
            _finite(
                self.baseline_net_capacity,
                field_name="baseline_net_capacity",
            ),
        )
        object.__setattr__(
            self,
            "intervention_net_capacity",
            _finite(
                self.intervention_net_capacity,
                field_name="intervention_net_capacity",
            ),
        )
        object.__setattr__(
            self,
            "net_capacity_delta",
            _finite(
                self.net_capacity_delta,
                field_name="net_capacity_delta",
            ),
        )

        for field_name in (
            "capacity_benefit_score",
            "loss_reduction_score",
            "gain_increase_score",
            "forecast_quality_score",
            "validation_score",
            "cost_penalty",
            "safety_penalty",
            "complexity_penalty",
            "model_penalty",
            "normalized_score",
        ):
            value = _unit(
                getattr(self, field_name),
                field_name=field_name,
            )
            object.__setattr__(
                self,
                field_name,
                value,
            )

        object.__setattr__(
            self,
            "raw_score",
            _finite(
                self.raw_score,
                field_name="raw_score",
            ),
        )

        if (
            not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise CounterfactualError(
                "rank must be a non-negative integer"
            )

    @property
    def intervention_id(self) -> str:
        return (
            self.scenario.intervention.intervention_id
        )

    @property
    def intervention_name(self) -> str:
        return self.scenario.intervention.name

    @property
    def target_id(self) -> str:
        return self.scenario.intervention.target_id

    @property
    def is_beneficial(self) -> bool:
        return self.net_capacity_delta > 0.0

    def with_rank(
        self,
        rank: int,
    ) -> CounterfactualScore:
        if (
            not isinstance(rank, int)
            or rank <= 0
        ):
            raise CounterfactualError(
                "rank must be a positive integer"
            )

        return CounterfactualScore(
            scenario=self.scenario,
            baseline_net_capacity=(
                self.baseline_net_capacity
            ),
            intervention_net_capacity=(
                self.intervention_net_capacity
            ),
            net_capacity_delta=(
                self.net_capacity_delta
            ),
            capacity_benefit_score=(
                self.capacity_benefit_score
            ),
            loss_reduction_score=(
                self.loss_reduction_score
            ),
            gain_increase_score=(
                self.gain_increase_score
            ),
            forecast_quality_score=(
                self.forecast_quality_score
            ),
            validation_score=(
                self.validation_score
            ),
            cost_penalty=self.cost_penalty,
            safety_penalty=self.safety_penalty,
            complexity_penalty=(
                self.complexity_penalty
            ),
            model_penalty=self.model_penalty,
            raw_score=self.raw_score,
            normalized_score=(
                self.normalized_score
            ),
            rank=rank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "intervention_name": (
                self.intervention_name
            ),
            "target_id": self.target_id,
            "rank": self.rank,
            "baseline_net_capacity": (
                self.baseline_net_capacity
            ),
            "intervention_net_capacity": (
                self.intervention_net_capacity
            ),
            "net_capacity_delta": (
                self.net_capacity_delta
            ),
            "capacity_benefit_score": (
                self.capacity_benefit_score
            ),
            "loss_reduction_score": (
                self.loss_reduction_score
            ),
            "gain_increase_score": (
                self.gain_increase_score
            ),
            "forecast_quality_score": (
                self.forecast_quality_score
            ),
            "validation_score": (
                self.validation_score
            ),
            "cost_penalty": self.cost_penalty,
            "safety_penalty": self.safety_penalty,
            "complexity_penalty": (
                self.complexity_penalty
            ),
            "model_penalty": self.model_penalty,
            "raw_score": self.raw_score,
            "normalized_score": (
                self.normalized_score
            ),
            "is_beneficial": self.is_beneficial,
        }


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    """
    Immutable result of one intervention search.
    """

    baseline_forecast_id: str
    status: CounterfactualStatus
    rankings: tuple[CounterfactualScore, ...] = ()

    confidence: float = 0.0
    ambiguity: float = 1.0
    selection_margin: float = 0.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    result_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        baseline_forecast_id = _require_text(
            self.baseline_forecast_id,
            field_name="baseline_forecast_id",
        )
        result_id = _require_text(
            self.result_id,
            field_name="result_id",
        )

        try:
            status = CounterfactualStatus(
                self.status
            )
        except (TypeError, ValueError) as exc:
            raise CounterfactualError(
                f"unsupported counterfactual status: {self.status!r}"
            ) from exc

        rankings = tuple(self.rankings)

        if not all(
            isinstance(
                item,
                CounterfactualScore,
            )
            for item in rankings
        ):
            raise CounterfactualError(
                "all rankings must be CounterfactualScore objects"
            )

        expected = tuple(
            range(
                1,
                len(rankings) + 1,
            )
        )
        actual = tuple(
            item.rank
            for item in rankings
        )

        if rankings and actual != expected:
            raise CounterfactualError(
                "ranking positions must be consecutive from one"
            )

        object.__setattr__(
            self,
            "baseline_forecast_id",
            baseline_forecast_id,
        )
        object.__setattr__(
            self,
            "result_id",
            result_id,
        )
        object.__setattr__(
            self,
            "status",
            status,
        )
        object.__setattr__(
            self,
            "rankings",
            rankings,
        )
        object.__setattr__(
            self,
            "confidence",
            _unit(
                self.confidence,
                field_name="confidence",
            ),
        )
        object.__setattr__(
            self,
            "ambiguity",
            _unit(
                self.ambiguity,
                field_name="ambiguity",
            ),
        )
        object.__setattr__(
            self,
            "selection_margin",
            _unit(
                self.selection_margin,
                field_name="selection_margin",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(
                self.metadata,
                field_name="metadata",
            ),
        )

    @property
    def best(
        self,
    ) -> CounterfactualScore | None:
        return (
            self.rankings[0]
            if self.rankings
            else None
        )

    @property
    def node_star(self) -> str | None:
        """
        Return the target ID of the best admissible intervention.
        """

        return (
            self.best.target_id
            if self.best is not None
            else None
        )

    @property
    def alternatives(
        self,
    ) -> tuple[CounterfactualScore, ...]:
        return self.rankings[1:]

    @property
    def is_ambiguous(self) -> bool:
        return (
            self.status
            is CounterfactualStatus.AMBIGUOUS
        )

    def top(
        self,
        count: int,
    ) -> tuple[CounterfactualScore, ...]:
        if not isinstance(count, int):
            raise CounterfactualError(
                "count must be an integer"
            )

        if count < 0:
            raise CounterfactualError(
                "count must be non-negative"
            )

        return self.rankings[:count]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "result_id": self.result_id,
            "baseline_forecast_id": (
                self.baseline_forecast_id
            ),
            "status": self.status.value,
            "node_star": self.node_star,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "selection_margin": (
                self.selection_margin
            ),
            "rankings": [
                item.to_dict()
                for item in self.rankings
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualEngine:
    """
    Rank forward-calculated intervention scenarios.

    cost_scale and complexity_scale convert unbounded non-negative values to
    normalized penalties:

        penalty = value / (value + scale)
    """

    weights: CounterfactualWeights = field(
        default_factory=CounterfactualWeights
    )
    ambiguity_threshold: float = 0.05
    cost_scale: float = 1.0
    complexity_scale: float = 5.0
    minimum_score: float = 0.0
    require_benefit: bool = False

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(
            self.weights,
            CounterfactualWeights,
        ):
            raise CounterfactualError(
                "weights must be CounterfactualWeights"
            )

        ambiguity_threshold = _unit(
            self.ambiguity_threshold,
            field_name="ambiguity_threshold",
        )
        cost_scale = _positive(
            self.cost_scale,
            field_name="cost_scale",
        )
        complexity_scale = _positive(
            self.complexity_scale,
            field_name="complexity_scale",
        )
        minimum_score = _unit(
            self.minimum_score,
            field_name="minimum_score",
        )

        if not isinstance(
            self.require_benefit,
            bool,
        ):
            raise CounterfactualError(
                "require_benefit must be a boolean"
            )

        object.__setattr__(
            self,
            "ambiguity_threshold",
            ambiguity_threshold,
        )
        object.__setattr__(
            self,
            "cost_scale",
            cost_scale,
        )
        object.__setattr__(
            self,
            "complexity_scale",
            complexity_scale,
        )
        object.__setattr__(
            self,
            "minimum_score",
            minimum_score,
        )

    def _bounded_penalty(
        self,
        value: float,
        scale: float,
    ) -> float:
        return value / (
            value + scale
        )

    def score_scenario(
        self,
        baseline: HistoryForecastResult,
        scenario: CounterfactualScenario,
    ) -> CounterfactualScore:
        """
        Score one intervention scenario without assigning rank.
        """

        if not isinstance(
            baseline,
            HistoryForecastResult,
        ):
            raise CounterfactualError(
                "baseline must be a HistoryForecastResult"
            )

        if not isinstance(
            scenario,
            CounterfactualScenario,
        ):
            raise CounterfactualError(
                "scenario must be a CounterfactualScenario"
            )

        if baseline.best is None:
            raise CounterfactualError(
                "baseline forecast must contain a best candidate"
            )

        if not scenario.is_eligible:
            raise CounterfactualError(
                "ineligible intervention scenarios cannot be scored"
            )

        baseline_signature = (
            baseline.best.candidate.predicted_signature
        )
        intervention_signature = (
            scenario.forecast_result
            .best
            .candidate
            .predicted_signature
        )

        baseline_net = (
            baseline_signature.net_capacity_effect
        )
        intervention_net = (
            intervention_signature.net_capacity_effect
        )
        net_delta = (
            intervention_net
            - baseline_net
        )

        capacity_signed = _relative_change(
            baseline_net,
            intervention_net,
        )
        capacity_benefit_score = (
            _benefit_from_signed_change(
                capacity_signed
            )
        )

        loss_reduction_signed = _relative_change(
            intervention_signature.total_capacity_loss,
            baseline_signature.total_capacity_loss,
        )
        loss_reduction_score = (
            _benefit_from_signed_change(
                loss_reduction_signed
            )
        )

        gain_increase_signed = _relative_change(
            baseline_signature.total_capacity_gain,
            intervention_signature.total_capacity_gain,
        )
        gain_increase_score = (
            _benefit_from_signed_change(
                gain_increase_signed
            )
        )

        forecast_quality_score = (
            scenario.forecast_result.confidence
        )
        validation_score = (
            scenario.validation_score
        )

        intervention = scenario.intervention

        cost_penalty = (
            self._bounded_penalty(
                intervention.cost,
                self.cost_scale,
            )
        )
        safety_penalty = (
            intervention.safety_penalty
        )
        complexity_penalty = (
            self._bounded_penalty(
                intervention.complexity,
                self.complexity_scale,
            )
        )
        model_penalty = (
            scenario.model_penalty
        )

        positive_sum = (
            capacity_benefit_score
            * self.weights.capacity_benefit
            + loss_reduction_score
            * self.weights.loss_reduction
            + gain_increase_score
            * self.weights.gain_increase
            + forecast_quality_score
            * self.weights.forecast_quality
            + validation_score
            * self.weights.validation
        )

        normalized_positive = (
            positive_sum
            / self.weights.positive_weight
        )

        total_penalty = (
            cost_penalty
            * self.weights.cost
            + safety_penalty
            * self.weights.safety
            + complexity_penalty
            * self.weights.complexity
            + model_penalty
            * self.weights.model_penalty
        )

        raw_score = (
            normalized_positive
            - total_penalty
        )
        normalized_score = max(
            min(raw_score, 1.0),
            0.0,
        )

        return CounterfactualScore(
            scenario=scenario,
            baseline_net_capacity=baseline_net,
            intervention_net_capacity=intervention_net,
            net_capacity_delta=net_delta,
            capacity_benefit_score=(
                capacity_benefit_score
            ),
            loss_reduction_score=(
                loss_reduction_score
            ),
            gain_increase_score=(
                gain_increase_score
            ),
            forecast_quality_score=(
                forecast_quality_score
            ),
            validation_score=(
                validation_score
            ),
            cost_penalty=cost_penalty,
            safety_penalty=safety_penalty,
            complexity_penalty=(
                complexity_penalty
            ),
            model_penalty=model_penalty,
            raw_score=raw_score,
            normalized_score=(
                normalized_score
            ),
        )

    def evaluate(
        self,
        baseline: HistoryForecastResult,
        scenarios: Iterable[
            CounterfactualScenario
        ],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CounterfactualResult:
        """
        Rank all admissible intervention scenarios.
        """

        if not isinstance(
            baseline,
            HistoryForecastResult,
        ):
            raise CounterfactualError(
                "baseline must be a HistoryForecastResult"
            )

        if baseline.best is None:
            raise CounterfactualError(
                "baseline forecast must contain a best candidate"
            )

        scenario_tuple = tuple(scenarios)

        if not all(
            isinstance(
                item,
                CounterfactualScenario,
            )
            for item in scenario_tuple
        ):
            raise CounterfactualError(
                "all scenarios must be CounterfactualScenario objects"
            )

        scenario_ids = [
            item.scenario_id
            for item in scenario_tuple
        ]

        if len(
            scenario_ids
        ) != len(
            set(scenario_ids)
        ):
            raise CounterfactualError(
                "scenario_id values must be unique"
            )

        if not scenario_tuple:
            return CounterfactualResult(
                baseline_forecast_id=(
                    baseline.result_id
                ),
                status=(
                    CounterfactualStatus.NO_SCENARIOS
                ),
                metadata=metadata or {},
            )

        eligible = tuple(
            item
            for item in scenario_tuple
            if item.is_eligible
        )

        if not eligible:
            return CounterfactualResult(
                baseline_forecast_id=(
                    baseline.result_id
                ),
                status=(
                    CounterfactualStatus
                    .NO_VALID_SCENARIOS
                ),
                metadata=metadata or {},
            )

        scored = [
            self.score_scenario(
                baseline,
                scenario,
            )
            for scenario in eligible
        ]

        if self.require_benefit:
            scored = [
                item
                for item in scored
                if item.net_capacity_delta > 0.0
            ]

        scored.sort(
            key=lambda item: (
                -item.normalized_score,
                -item.net_capacity_delta,
                item.safety_penalty,
                item.cost_penalty,
                item.intervention_id,
            )
        )

        ranked = tuple(
            score.with_rank(index)
            for index, score in enumerate(
                scored,
                start=1,
            )
            if score.normalized_score
            >= self.minimum_score
        )

        if not ranked:
            return CounterfactualResult(
                baseline_forecast_id=(
                    baseline.result_id
                ),
                status=(
                    CounterfactualStatus
                    .NO_VALID_SCENARIOS
                ),
                metadata=metadata or {},
            )

        best_score = (
            ranked[0].normalized_score
        )

        if len(ranked) == 1:
            margin = best_score
        else:
            margin = max(
                best_score
                - ranked[1].normalized_score,
                0.0,
            )

        ambiguity = 1.0 - margin

        confidence = min(
            max(
                best_score
                * (
                    0.5
                    + 0.5 * margin
                )
                * (
                    0.5
                    + 0.5
                    * ranked[0].validation_score
                ),
                0.0,
            ),
            1.0,
        )

        status = (
            CounterfactualStatus.AMBIGUOUS
            if (
                len(ranked) > 1
                and margin
                < self.ambiguity_threshold
            )
            else CounterfactualStatus.COMPLETED
        )

        vetoed_count = sum(
            not item.intervention.non_fonit_allowed
            or item.intervention.status
            is InterventionStatus.VETOED
            for item in scenario_tuple
        )

        merged_metadata = {
            "engine_version": self.VERSION,
            "scenario_count": len(
                scenario_tuple
            ),
            "eligible_count": len(
                eligible
            ),
            "ranked_count": len(
                ranked
            ),
            "vetoed_count": vetoed_count,
            "require_benefit": (
                self.require_benefit
            ),
        }

        if metadata is not None:
            merged_metadata.update(
                dict(metadata)
            )

        return CounterfactualResult(
            baseline_forecast_id=(
                baseline.result_id
            ),
            status=status,
            rankings=ranked,
            confidence=confidence,
            ambiguity=ambiguity,
            selection_margin=margin,
            metadata=merged_metadata,
        )

    def softmax_probabilities(
        self,
        result: CounterfactualResult,
        *,
        temperature: float = 1.0,
    ) -> Mapping[str, float]:
        """
        Convert ranked raw scores into model-relative probabilities.

        These values are not calibrated probabilities of real-world success.
        """

        if not isinstance(
            result,
            CounterfactualResult,
        ):
            raise CounterfactualError(
                "result must be a CounterfactualResult"
            )

        temperature = _positive(
            temperature,
            field_name="temperature",
        )

        if not result.rankings:
            return MappingProxyType({})

        maximum = max(
            item.raw_score
            for item in result.rankings
        )

        exponentials = {
            item.intervention_id: exp(
                (
                    item.raw_score
                    - maximum
                )
                / temperature
            )
            for item in result.rankings
        }

        total = sum(
            exponentials.values()
        )

        probabilities = {
            intervention_id: value / total
            for intervention_id, value
            in exponentials.items()
        }

        return MappingProxyType(
            dict(
                sorted(
                    probabilities.items()
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "weights": self.weights.to_dict(),
            "ambiguity_threshold": (
                self.ambiguity_threshold
            ),
            "cost_scale": self.cost_scale,
            "complexity_scale": (
                self.complexity_scale
            ),
            "minimum_score": (
                self.minimum_score
            ),
            "require_benefit": (
                self.require_benefit
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> CounterfactualEngine:
        if not isinstance(data, Mapping):
            raise CounterfactualError(
                "counterfactual-engine data must be a mapping"
            )

        version = data.get(
            "version",
            cls.VERSION,
        )

        if version != cls.VERSION:
            raise CounterfactualError(
                "unsupported counterfactual engine version: "
                f"{version!r}"
            )

        weights_data = data.get(
            "weights",
            {},
        )

        if not isinstance(
            weights_data,
            Mapping,
        ):
            raise CounterfactualError(
                "weights must be a mapping"
            )

        return cls(
            weights=CounterfactualWeights(
                capacity_benefit=float(
                    weights_data.get(
                        "capacity_benefit",
                        1.0,
                    )
                ),
                loss_reduction=float(
                    weights_data.get(
                        "loss_reduction",
                        0.75,
                    )
                ),
                gain_increase=float(
                    weights_data.get(
                        "gain_increase",
                        0.50,
                    )
                ),
                forecast_quality=float(
                    weights_data.get(
                        "forecast_quality",
                        0.50,
                    )
                ),
                validation=float(
                    weights_data.get(
                        "validation",
                        0.50,
                    )
                ),
                cost=float(
                    weights_data.get(
                        "cost",
                        0.30,
                    )
                ),
                safety=float(
                    weights_data.get(
                        "safety",
                        0.75,
                    )
                ),
                complexity=float(
                    weights_data.get(
                        "complexity",
                        0.20,
                    )
                ),
                model_penalty=float(
                    weights_data.get(
                        "model_penalty",
                        0.30,
                    )
                ),
            ),
            ambiguity_threshold=float(
                data.get(
                    "ambiguity_threshold",
                    0.05,
                )
            ),
            cost_scale=float(
                data.get(
                    "cost_scale",
                    1.0,
                )
            ),
            complexity_scale=float(
                data.get(
                    "complexity_scale",
                    5.0,
                )
            ),
            minimum_score=float(
                data.get(
                    "minimum_score",
                    0.0,
                )
            ),
            require_benefit=bool(
                data.get(
                    "require_benefit",
                    False,
                )
            ),
        )


__all__ = [
    "CounterfactualEngine",
    "CounterfactualError",
    "CounterfactualResult",
    "CounterfactualScenario",
    "CounterfactualScore",
    "CounterfactualStatus",
    "CounterfactualWeights",
    "Intervention",
    "InterventionScope",
    "InterventionStatus",
]
