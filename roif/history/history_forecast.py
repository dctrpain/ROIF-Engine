from __future__ import annotations

"""
History-aware forecasting for the ROIF Structural Memory Engine.

The module ranks explicitly supplied candidate future states against the
current StructuralSignature and an explicit set of influence-plane factors.

Scientific boundary
-------------------
The forecaster does not invent future events. Every candidate must provide a
forward-predicted StructuralSignature.

The multiplicative plane operator is:

    M = product(multiplier_i ** (weight_i * confidence_i))

It is a configurable model component, not proof that all real influences are
independent or exactly multiplicative.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .structural_signature import (
    SignatureComparison,
    SignatureDistanceWeights,
    StructuralSignature,
)


class HistoryForecastError(ValueError):
    """Raised when forecast data are invalid."""


class ForecastStatus(str, Enum):
    COMPLETED = "completed"
    AMBIGUOUS = "ambiguous"
    NO_CANDIDATES = "no_candidates"
    NO_VALID_CANDIDATES = "no_valid_candidates"


class ForecastCandidateStatus(str, Enum):
    READY = "ready"
    FORWARD_VALIDATED = "forward_validated"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    FAILED = "failed"


class ForecastDirection(str, Enum):
    DETERIORATION = "deterioration"
    ADAPTATION = "adaptation"
    MIXED = "mixed"
    STABLE = "stable"
    UNKNOWN = "unknown"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise HistoryForecastError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise HistoryForecastError(f"{name} must not be empty")
    return value


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoryForecastError(
            f"{name} must be a real number"
        ) from exc
    if not isfinite(result):
        raise HistoryForecastError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise HistoryForecastError(
            f"{name} must be non-negative"
        )
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise HistoryForecastError(
            f"{name} must be greater than zero"
        )
    return result


def _unit(value: float, name: str) -> float:
    result = _nonnegative(value, name)
    if result > 1.0:
        raise HistoryForecastError(
            f"{name} must be in [0, 1]"
        )
    return result


def _ids(
    values: Iterable[str],
    name: str,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for raw in values:
        value = _text(raw, name)
        if value not in seen:
            seen.add(value)
            result.append(value)

    return tuple(result)


def _mapping(
    value: Mapping[str, Any] | None,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise HistoryForecastError(
            f"{name} must be a mapping"
        )

    normalized = {
        _text(key, f"{name} key"): item
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class ForecastHorizon:
    start: float
    end: float
    units: str = "seconds"
    label: str | None = None

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        start = _nonnegative(self.start, "start")
        end = _positive(self.end, "end")
        if end <= start:
            raise HistoryForecastError(
                "end must be greater than start"
            )

        units = _text(self.units, "units")
        label = (
            None
            if self.label is None
            else _text(self.label, "label")
        )

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "label", label)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "units": self.units,
            "label": self.label,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ForecastHorizon":
        if not isinstance(data, Mapping):
            raise HistoryForecastError(
                "forecast-horizon data must be a mapping"
            )
        if data.get("version", cls.VERSION) != cls.VERSION:
            raise HistoryForecastError(
                "unsupported forecast horizon version"
            )

        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            units=str(data.get("units", "seconds")),
            label=data.get("label"),
        )


@dataclass(frozen=True, slots=True)
class PlaneInfluence:
    plane_id: str
    multiplier: float = 1.0
    weight: float = 1.0
    confidence: float = 1.0
    observed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        plane_id = _text(self.plane_id, "plane_id")
        multiplier = _positive(self.multiplier, "multiplier")
        weight = _nonnegative(self.weight, "weight")
        confidence = _unit(self.confidence, "confidence")

        if not isinstance(self.observed, bool):
            raise HistoryForecastError(
                "observed must be a boolean"
            )

        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "metadata",
            _mapping(self.metadata, "metadata"),
        )

    @property
    def exponent(self) -> float:
        return self.weight * self.confidence

    @property
    def effective_multiplier(self) -> float:
        return exp(self.exponent * log(self.multiplier))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "plane_id": self.plane_id,
            "multiplier": self.multiplier,
            "weight": self.weight,
            "confidence": self.confidence,
            "observed": self.observed,
            "effective_multiplier": self.effective_multiplier,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "PlaneInfluence":
        if not isinstance(data, Mapping):
            raise HistoryForecastError(
                "plane-influence data must be a mapping"
            )
        if data.get("version", cls.VERSION) != cls.VERSION:
            raise HistoryForecastError(
                "unsupported plane influence version"
            )

        return cls(
            plane_id=str(data["plane_id"]),
            multiplier=float(data.get("multiplier", 1.0)),
            weight=float(data.get("weight", 1.0)),
            confidence=float(data.get("confidence", 1.0)),
            observed=bool(data.get("observed", True)),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class PlaneInfluenceSet:
    influences: tuple[PlaneInfluence, ...] = ()
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    influence_set_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        influences = tuple(self.influences)
        if not all(
            isinstance(item, PlaneInfluence)
            for item in influences
        ):
            raise HistoryForecastError(
                "all influences must be PlaneInfluence objects"
            )

        ids = [item.plane_id for item in influences]
        if len(ids) != len(set(ids)):
            raise HistoryForecastError(
                "plane_id values must be unique"
            )

        object.__setattr__(
            self,
            "influences",
            tuple(sorted(influences, key=lambda x: x.plane_id)),
        )
        object.__setattr__(
            self,
            "label",
            None if self.label is None else _text(self.label, "label"),
        )
        object.__setattr__(
            self,
            "influence_set_id",
            _text(self.influence_set_id, "influence_set_id"),
        )
        object.__setattr__(
            self,
            "metadata",
            _mapping(self.metadata, "metadata"),
        )

    def __len__(self) -> int:
        return len(self.influences)

    def __iter__(self):
        return iter(self.influences)

    @property
    def combined_multiplier(self) -> float:
        return exp(
            sum(
                item.exponent * log(item.multiplier)
                for item in self.influences
            )
        )

    @property
    def observed_fraction(self) -> float:
        if not self.influences:
            return 1.0
        return (
            sum(item.observed for item in self.influences)
            / len(self.influences)
        )

    def subset(
        self,
        plane_ids: Iterable[str],
    ) -> "PlaneInfluenceSet":
        selected_ids = set(_ids(plane_ids, "plane_ids"))
        return PlaneInfluenceSet(
            influences=tuple(
                item
                for item in self.influences
                if item.plane_id in selected_ids
            ),
            label="candidate-specific subset",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "influence_set_id": self.influence_set_id,
            "label": self.label,
            "influences": [
                item.to_dict()
                for item in self.influences
            ],
            "combined_multiplier": self.combined_multiplier,
            "observed_fraction": self.observed_fraction,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "PlaneInfluenceSet":
        if not isinstance(data, Mapping):
            raise HistoryForecastError(
                "plane-influence-set data must be a mapping"
            )
        if data.get("version", cls.VERSION) != cls.VERSION:
            raise HistoryForecastError(
                "unsupported plane influence-set version"
            )

        raw = data.get("influences", ())
        if not isinstance(raw, (list, tuple)):
            raise HistoryForecastError(
                "influences must be a sequence"
            )

        return cls(
            influence_set_id=str(data["influence_set_id"]),
            label=data.get("label"),
            influences=tuple(
                PlaneInfluence.from_dict(item)
                for item in raw
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ForecastWeights:
    continuity: float = 1.0
    prior: float = 0.25
    validation: float = 0.50
    plane_support: float = 0.75
    complexity: float = 0.20
    custom_penalty: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            "continuity",
            "prior",
            "validation",
            "plane_support",
            "complexity",
            "custom_penalty",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )

        if self.positive_total <= 0.0:
            raise HistoryForecastError(
                "at least one positive forecast weight is required"
            )

    @property
    def positive_total(self) -> float:
        return (
            self.continuity
            + self.prior
            + self.validation
            + self.plane_support
        )

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "continuity",
                "prior",
                "validation",
                "plane_support",
                "complexity",
                "custom_penalty",
            )
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ForecastWeights":
        if not isinstance(data, Mapping):
            raise HistoryForecastError(
                "forecast-weight data must be a mapping"
            )

        return cls(
            continuity=float(data.get("continuity", 1.0)),
            prior=float(data.get("prior", 0.25)),
            validation=float(data.get("validation", 0.50)),
            plane_support=float(data.get("plane_support", 0.75)),
            complexity=float(data.get("complexity", 0.20)),
            custom_penalty=float(
                data.get("custom_penalty", 0.20)
            ),
        )


@dataclass(frozen=True, slots=True)
class ForecastCandidate:
    name: str
    horizon: ForecastHorizon
    predicted_signature: StructuralSignature

    description: str | None = None
    direction: ForecastDirection = ForecastDirection.UNKNOWN
    affected_plane_ids: tuple[str, ...] = ()
    affected_target_ids: tuple[str, ...] = ()
    predicted_change_kinds: tuple[str, ...] = ()

    prior_probability: float = 0.5
    validation_score: float = 1.0
    complexity: float = 0.0
    custom_penalty: float = 0.0

    status: ForecastCandidateStatus = ForecastCandidateStatus.READY
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = field(default_factory=lambda: uuid4().hex)

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, ForecastHorizon):
            raise HistoryForecastError(
                "horizon must be a ForecastHorizon"
            )
        if not isinstance(
            self.predicted_signature,
            StructuralSignature,
        ):
            raise HistoryForecastError(
                "predicted_signature must be a StructuralSignature"
            )

        try:
            direction = ForecastDirection(self.direction)
            status = ForecastCandidateStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise HistoryForecastError(
                "unsupported forecast enum value"
            ) from exc

        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(
            self,
            "candidate_id",
            _text(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "description",
            None
            if self.description is None
            else _text(self.description, "description"),
        )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "affected_plane_ids",
            _ids(self.affected_plane_ids, "affected_plane_ids"),
        )
        object.__setattr__(
            self,
            "affected_target_ids",
            _ids(self.affected_target_ids, "affected_target_ids"),
        )
        object.__setattr__(
            self,
            "predicted_change_kinds",
            _ids(
                self.predicted_change_kinds,
                "predicted_change_kinds",
            ),
        )
        object.__setattr__(
            self,
            "prior_probability",
            _unit(self.prior_probability, "prior_probability"),
        )
        object.__setattr__(
            self,
            "validation_score",
            _unit(self.validation_score, "validation_score"),
        )
        object.__setattr__(
            self,
            "complexity",
            _nonnegative(self.complexity, "complexity"),
        )
        object.__setattr__(
            self,
            "custom_penalty",
            _unit(self.custom_penalty, "custom_penalty"),
        )
        object.__setattr__(
            self,
            "metadata",
            _mapping(self.metadata, "metadata"),
        )

    @property
    def is_eligible(self) -> bool:
        return self.status in {
            ForecastCandidateStatus.READY,
            ForecastCandidateStatus.FORWARD_VALIDATED,
            ForecastCandidateStatus.INCOMPLETE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "candidate_id": self.candidate_id,
            "name": self.name,
            "description": self.description,
            "direction": self.direction.value,
            "horizon": self.horizon.to_dict(),
            "predicted_signature": (
                self.predicted_signature.to_dict()
            ),
            "affected_plane_ids": list(self.affected_plane_ids),
            "affected_target_ids": list(self.affected_target_ids),
            "predicted_change_kinds": list(
                self.predicted_change_kinds
            ),
            "prior_probability": self.prior_probability,
            "validation_score": self.validation_score,
            "complexity": self.complexity,
            "custom_penalty": self.custom_penalty,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ForecastCandidate":
        if not isinstance(data, Mapping):
            raise HistoryForecastError(
                "forecast-candidate data must be a mapping"
            )
        if data.get("version", cls.VERSION) != cls.VERSION:
            raise HistoryForecastError(
                "unsupported forecast candidate version"
            )

        horizon = data.get("horizon")
        signature = data.get("predicted_signature")

        if not isinstance(horizon, Mapping):
            raise HistoryForecastError(
                "horizon is missing or invalid"
            )
        if not isinstance(signature, Mapping):
            raise HistoryForecastError(
                "predicted_signature is missing or invalid"
            )

        return cls(
            candidate_id=str(data["candidate_id"]),
            name=str(data["name"]),
            description=data.get("description"),
            direction=ForecastDirection(
                data.get("direction", "unknown")
            ),
            horizon=ForecastHorizon.from_dict(horizon),
            predicted_signature=StructuralSignature.from_dict(
                signature
            ),
            affected_plane_ids=tuple(
                data.get("affected_plane_ids", ())
            ),
            affected_target_ids=tuple(
                data.get("affected_target_ids", ())
            ),
            predicted_change_kinds=tuple(
                data.get("predicted_change_kinds", ())
            ),
            prior_probability=float(
                data.get("prior_probability", 0.5)
            ),
            validation_score=float(
                data.get("validation_score", 1.0)
            ),
            complexity=float(data.get("complexity", 0.0)),
            custom_penalty=float(
                data.get("custom_penalty", 0.0)
            ),
            status=ForecastCandidateStatus(
                data.get("status", "ready")
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ForecastScore:
    candidate: ForecastCandidate
    continuity_comparison: SignatureComparison
    continuity_score: float
    prior_score: float
    validation_score: float
    plane_support_score: float
    plane_multiplier: float
    complexity_penalty: float
    custom_penalty: float
    raw_score: float
    normalized_score: float
    rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ForecastCandidate):
            raise HistoryForecastError(
                "candidate must be a ForecastCandidate"
            )
        if not isinstance(
            self.continuity_comparison,
            SignatureComparison,
        ):
            raise HistoryForecastError(
                "continuity_comparison must be a SignatureComparison"
            )

        for name in (
            "continuity_score",
            "prior_score",
            "validation_score",
            "plane_support_score",
            "complexity_penalty",
            "custom_penalty",
            "normalized_score",
        ):
            object.__setattr__(
                self,
                name,
                _unit(getattr(self, name), name),
            )

        object.__setattr__(
            self,
            "plane_multiplier",
            _positive(self.plane_multiplier, "plane_multiplier"),
        )
        object.__setattr__(
            self,
            "raw_score",
            _finite(self.raw_score, "raw_score"),
        )

        if not isinstance(self.rank, int) or self.rank < 0:
            raise HistoryForecastError(
                "rank must be a non-negative integer"
            )

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def name(self) -> str:
        return self.candidate.name

    def with_rank(self, rank: int) -> "ForecastScore":
        if not isinstance(rank, int) or rank <= 0:
            raise HistoryForecastError(
                "rank must be a positive integer"
            )

        return ForecastScore(
            candidate=self.candidate,
            continuity_comparison=self.continuity_comparison,
            continuity_score=self.continuity_score,
            prior_score=self.prior_score,
            validation_score=self.validation_score,
            plane_support_score=self.plane_support_score,
            plane_multiplier=self.plane_multiplier,
            complexity_penalty=self.complexity_penalty,
            custom_penalty=self.custom_penalty,
            raw_score=self.raw_score,
            normalized_score=self.normalized_score,
            rank=rank,
        )


@dataclass(frozen=True, slots=True)
class HistoryForecastResult:
    current_signature_id: str
    influence_set_id: str
    status: ForecastStatus
    rankings: tuple[ForecastScore, ...] = ()
    confidence: float = 0.0
    ambiguity: float = 1.0
    selection_margin: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: uuid4().hex)

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        rankings = tuple(self.rankings)
        if not all(
            isinstance(item, ForecastScore)
            for item in rankings
        ):
            raise HistoryForecastError(
                "all rankings must be ForecastScore objects"
            )

        actual = tuple(item.rank for item in rankings)
        expected = tuple(range(1, len(rankings) + 1))
        if rankings and actual != expected:
            raise HistoryForecastError(
                "ranking positions must be consecutive from one"
            )

        try:
            status = ForecastStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise HistoryForecastError(
                "unsupported forecast status"
            ) from exc

        object.__setattr__(
            self,
            "current_signature_id",
            _text(
                self.current_signature_id,
                "current_signature_id",
            ),
        )
        object.__setattr__(
            self,
            "influence_set_id",
            _text(self.influence_set_id, "influence_set_id"),
        )
        object.__setattr__(
            self,
            "result_id",
            _text(self.result_id, "result_id"),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "rankings", rankings)
        object.__setattr__(
            self,
            "confidence",
            _unit(self.confidence, "confidence"),
        )
        object.__setattr__(
            self,
            "ambiguity",
            _unit(self.ambiguity, "ambiguity"),
        )
        object.__setattr__(
            self,
            "selection_margin",
            _unit(self.selection_margin, "selection_margin"),
        )
        object.__setattr__(
            self,
            "metadata",
            _mapping(self.metadata, "metadata"),
        )

    @property
    def best(self) -> ForecastScore | None:
        return self.rankings[0] if self.rankings else None

    @property
    def alternatives(self) -> tuple[ForecastScore, ...]:
        return self.rankings[1:]

    @property
    def is_ambiguous(self) -> bool:
        return self.status is ForecastStatus.AMBIGUOUS

    def top(self, count: int) -> tuple[ForecastScore, ...]:
        if not isinstance(count, int):
            raise HistoryForecastError(
                "count must be an integer"
            )
        if count < 0:
            raise HistoryForecastError(
                "count must be non-negative"
            )
        return self.rankings[:count]


@dataclass(frozen=True, slots=True)
class HistoryForecaster:
    forecast_weights: ForecastWeights = field(
        default_factory=ForecastWeights
    )
    signature_weights: SignatureDistanceWeights = field(
        default_factory=SignatureDistanceWeights
    )
    ambiguity_threshold: float = 0.05
    complexity_scale: float = 5.0
    plane_scale: float = 1.0
    minimum_score: float = 0.0

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(
            self.forecast_weights,
            ForecastWeights,
        ):
            raise HistoryForecastError(
                "forecast_weights must be ForecastWeights"
            )
        if not isinstance(
            self.signature_weights,
            SignatureDistanceWeights,
        ):
            raise HistoryForecastError(
                "signature_weights must be SignatureDistanceWeights"
            )

        object.__setattr__(
            self,
            "ambiguity_threshold",
            _unit(
                self.ambiguity_threshold,
                "ambiguity_threshold",
            ),
        )
        object.__setattr__(
            self,
            "complexity_scale",
            _positive(
                self.complexity_scale,
                "complexity_scale",
            ),
        )
        object.__setattr__(
            self,
            "plane_scale",
            _positive(self.plane_scale, "plane_scale"),
        )
        object.__setattr__(
            self,
            "minimum_score",
            _unit(self.minimum_score, "minimum_score"),
        )

    def _complexity_penalty(self, complexity: float) -> float:
        return complexity / (
            complexity + self.complexity_scale
        )

    def _plane_support(self, multiplier: float) -> float:
        return 1.0 / (
            1.0
            + exp(
                -log(multiplier) / self.plane_scale
            )
        )

    def score_candidate(
        self,
        current: StructuralSignature,
        candidate: ForecastCandidate,
        influences: PlaneInfluenceSet,
    ) -> ForecastScore:
        if not isinstance(current, StructuralSignature):
            raise HistoryForecastError(
                "current must be a StructuralSignature"
            )
        if not isinstance(candidate, ForecastCandidate):
            raise HistoryForecastError(
                "candidate must be a ForecastCandidate"
            )
        if not isinstance(influences, PlaneInfluenceSet):
            raise HistoryForecastError(
                "influences must be a PlaneInfluenceSet"
            )
        if not candidate.is_eligible:
            raise HistoryForecastError(
                "rejected or failed candidates cannot be scored"
            )

        comparison = current.compare(
            candidate.predicted_signature,
            weights=self.signature_weights,
        )

        selected = (
            influences.subset(candidate.affected_plane_ids)
            if candidate.affected_plane_ids
            else influences
        )
        multiplier = selected.combined_multiplier
        plane_support = self._plane_support(multiplier)
        complexity_penalty = self._complexity_penalty(
            candidate.complexity
        )

        positive = (
            comparison.similarity
            * self.forecast_weights.continuity
            + candidate.prior_probability
            * self.forecast_weights.prior
            + candidate.validation_score
            * self.forecast_weights.validation
            + plane_support
            * self.forecast_weights.plane_support
        ) / self.forecast_weights.positive_total

        penalty = (
            complexity_penalty
            * self.forecast_weights.complexity
            + candidate.custom_penalty
            * self.forecast_weights.custom_penalty
        )

        raw = positive - penalty
        normalized = max(min(raw, 1.0), 0.0)

        return ForecastScore(
            candidate=candidate,
            continuity_comparison=comparison,
            continuity_score=comparison.similarity,
            prior_score=candidate.prior_probability,
            validation_score=candidate.validation_score,
            plane_support_score=plane_support,
            plane_multiplier=multiplier,
            complexity_penalty=complexity_penalty,
            custom_penalty=candidate.custom_penalty,
            raw_score=raw,
            normalized_score=normalized,
        )

    def forecast(
        self,
        current: StructuralSignature,
        candidates: Iterable[ForecastCandidate],
        influences: PlaneInfluenceSet,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> HistoryForecastResult:
        if not isinstance(current, StructuralSignature):
            raise HistoryForecastError(
                "current must be a StructuralSignature"
            )
        if not isinstance(influences, PlaneInfluenceSet):
            raise HistoryForecastError(
                "influences must be a PlaneInfluenceSet"
            )

        candidates = tuple(candidates)
        if not all(
            isinstance(item, ForecastCandidate)
            for item in candidates
        ):
            raise HistoryForecastError(
                "all candidates must be ForecastCandidate objects"
            )

        ids = [item.candidate_id for item in candidates]
        if len(ids) != len(set(ids)):
            raise HistoryForecastError(
                "candidate_id values must be unique"
            )

        if not candidates:
            return HistoryForecastResult(
                current_signature_id=current.signature_id,
                influence_set_id=influences.influence_set_id,
                status=ForecastStatus.NO_CANDIDATES,
                metadata=metadata or {},
            )

        eligible = tuple(
            item for item in candidates if item.is_eligible
        )
        if not eligible:
            return HistoryForecastResult(
                current_signature_id=current.signature_id,
                influence_set_id=influences.influence_set_id,
                status=ForecastStatus.NO_VALID_CANDIDATES,
                metadata=metadata or {},
            )

        scored = [
            self.score_candidate(
                current,
                candidate,
                influences,
            )
            for candidate in eligible
        ]
        scored.sort(
            key=lambda item: (
                -item.normalized_score,
                -item.continuity_score,
                -item.validation_score,
                item.candidate.complexity,
                item.candidate_id,
            )
        )

        ranked = tuple(
            score.with_rank(index)
            for index, score in enumerate(scored, start=1)
            if score.normalized_score >= self.minimum_score
        )

        if not ranked:
            return HistoryForecastResult(
                current_signature_id=current.signature_id,
                influence_set_id=influences.influence_set_id,
                status=ForecastStatus.NO_VALID_CANDIDATES,
                metadata=metadata or {},
            )

        best = ranked[0].normalized_score
        margin = (
            best
            if len(ranked) == 1
            else max(
                best - ranked[1].normalized_score,
                0.0,
            )
        )
        ambiguity = 1.0 - margin
        confidence = min(
            max(
                best
                * (0.5 + 0.5 * margin)
                * (
                    0.5
                    + 0.5 * influences.observed_fraction
                ),
                0.0,
            ),
            1.0,
        )

        status = (
            ForecastStatus.AMBIGUOUS
            if (
                len(ranked) > 1
                and margin < self.ambiguity_threshold
            )
            else ForecastStatus.COMPLETED
        )

        merged_metadata = {
            "forecaster_version": self.VERSION,
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "ranked_count": len(ranked),
            "plane_count": len(influences),
            "combined_plane_multiplier": (
                influences.combined_multiplier
            ),
            "observed_plane_fraction": (
                influences.observed_fraction
            ),
        }
        if metadata is not None:
            merged_metadata.update(dict(metadata))

        return HistoryForecastResult(
            current_signature_id=current.signature_id,
            influence_set_id=influences.influence_set_id,
            status=status,
            rankings=ranked,
            confidence=confidence,
            ambiguity=ambiguity,
            selection_margin=margin,
            metadata=merged_metadata,
        )

    def softmax_probabilities(
        self,
        result: HistoryForecastResult,
        *,
        temperature: float = 1.0,
    ) -> Mapping[str, float]:
        if not isinstance(result, HistoryForecastResult):
            raise HistoryForecastError(
                "result must be a HistoryForecastResult"
            )

        temperature = _positive(
            temperature,
            "temperature",
        )

        if not result.rankings:
            return MappingProxyType({})

        maximum = max(
            item.raw_score
            for item in result.rankings
        )
        values = {
            item.candidate_id: exp(
                (item.raw_score - maximum) / temperature
            )
            for item in result.rankings
        }
        total = sum(values.values())

        return MappingProxyType(
            dict(
                sorted(
                    (
                        candidate_id,
                        value / total,
                    )
                    for candidate_id, value
                    in values.items()
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "forecast_weights": (
                self.forecast_weights.to_dict()
            ),
            "signature_weights": (
                self.signature_weights.to_dict()
            ),
            "ambiguity_threshold": self.ambiguity_threshold,
            "complexity_scale": self.complexity_scale,
            "plane_scale": self.plane_scale,
            "minimum_score": self.minimum_score,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "HistoryForecaster":
        if not isinstance(data, Mapping):
            raise HistoryForecastError(
                "history-forecaster data must be a mapping"
            )
        if data.get("version", cls.VERSION) != cls.VERSION:
            raise HistoryForecastError(
                "unsupported history forecaster version"
            )

        forecast_weights = data.get(
            "forecast_weights",
            {},
        )
        signature_weights = data.get(
            "signature_weights",
            {},
        )

        if not isinstance(forecast_weights, Mapping):
            raise HistoryForecastError(
                "forecast_weights must be a mapping"
            )
        if not isinstance(signature_weights, Mapping):
            raise HistoryForecastError(
                "signature_weights must be a mapping"
            )

        return cls(
            forecast_weights=ForecastWeights.from_dict(
                forecast_weights
            ),
            signature_weights=(
                SignatureDistanceWeights.from_dict(
                    signature_weights
                )
            ),
            ambiguity_threshold=float(
                data.get("ambiguity_threshold", 0.05)
            ),
            complexity_scale=float(
                data.get("complexity_scale", 5.0)
            ),
            plane_scale=float(
                data.get("plane_scale", 1.0)
            ),
            minimum_score=float(
                data.get("minimum_score", 0.0)
            ),
        )


__all__ = [
    "ForecastCandidate",
    "ForecastCandidateStatus",
    "ForecastDirection",
    "ForecastHorizon",
    "ForecastScore",
    "ForecastStatus",
    "ForecastWeights",
    "HistoryForecastError",
    "HistoryForecastResult",
    "HistoryForecaster",
    "PlaneInfluence",
    "PlaneInfluenceSet",
]
