from __future__ import annotations

"""
Future-plane analysis for the ROIF Structural Memory Engine.

Purpose
-------
FuturePlaneAnalyzer estimates how explicitly supplied future influence planes
may modify candidate structural futures before intervention ranking.

It sits between:

    HistoryForecaster
        -> FuturePlaneAnalyzer
        -> CounterfactualEngine

The analyzer does not invent external events and does not claim certainty.
Every future plane must be supplied explicitly with:

- a time window;
- a signed influence direction;
- an intensity;
- a confidence estimate;
- a temporal profile;
- optional targets and candidate restrictions;
- an explicit Non-Fonit safety classification.

Scientific boundary
-------------------
This module implements a configurable multiplicative operator:

    M_candidate = product(m_i ** e_i)

where each effective exponent e_i depends on confidence, temporal overlap,
target overlap, and candidate applicability.

This is a model component. It is not proof that all real-world influences are
independent or exactly multiplicative. Correlated influences should be
represented as explicit joint planes or handled by a domain-specific forward
model.

Safety boundary
---------------
The analyzer includes an explicit Non-Fonit Gate. Unsafe, system-wide,
poorly bounded, or cascade-amplifying planes can be vetoed before they affect
ranking.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .history_forecast import (
    ForecastCandidate,
    ForecastHorizon,
    PlaneInfluence,
    PlaneInfluenceSet,
)


class FuturePlaneAnalyzerError(ValueError):
    """Raised when future-plane analysis data are invalid."""


class FuturePlaneStatus(str, Enum):
    """Computational and governance status of a future plane."""

    ACTIVE = "active"
    INFERRED = "inferred"
    INCOMPLETE = "incomplete"
    DISABLED = "disabled"
    FAILED = "failed"
    VETOED = "vetoed"


class FuturePlaneDirection(str, Enum):
    """High-level direction of a future-plane effect."""

    AMPLIFY = "amplify"
    ATTENUATE = "attenuate"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class TemporalProfile(str, Enum):
    """Temporal shape of a future-plane influence."""

    CONSTANT = "constant"
    IMPULSE = "impulse"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    WINDOWED = "windowed"


class FuturePlaneScope(str, Enum):
    """Declared scope of influence."""

    TARGET = "target"
    SUBSYSTEM = "subsystem"
    SYSTEM = "system"
    EXTERNAL = "external"


class FuturePlaneAnalysisStatus(str, Enum):
    """Overall result status of one future-plane analysis."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    NO_PLANES = "no_planes"
    NO_APPLICABLE_PLANES = "no_applicable_planes"
    ALL_VETOED = "all_vetoed"


def _text(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise FuturePlaneAnalyzerError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise FuturePlaneAnalyzerError(
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
        raise FuturePlaneAnalyzerError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise FuturePlaneAnalyzerError(
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
        raise FuturePlaneAnalyzerError(
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
        raise FuturePlaneAnalyzerError(
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
        raise FuturePlaneAnalyzerError(
            f"{field_name} must be in [0, 1]"
        )

    return numeric


def _signed_unit(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _finite(
        value,
        field_name=field_name,
    )

    if not -1.0 <= numeric <= 1.0:
        raise FuturePlaneAnalyzerError(
            f"{field_name} must be in [-1, 1]"
        )

    return numeric


def _ids(
    values: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_value in values:
        value = _text(
            raw_value,
            field_name=field_name,
        )

        if value not in seen:
            seen.add(value)
            normalized.append(value)

    return tuple(normalized)


def _mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise FuturePlaneAnalyzerError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, Any] = {}

    for raw_key, item in value.items():
        key = _text(
            raw_key,
            field_name=f"{field_name} key",
        )
        normalized[key] = item

    return MappingProxyType(
        dict(sorted(normalized.items()))
    )


def _interval_overlap_fraction(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    overlap = max(
        0.0,
        min(left_end, right_end)
        - max(left_start, right_start),
    )

    duration = max(
        right_end - right_start,
        1e-12,
    )

    return min(
        overlap / duration,
        1.0,
    )


@dataclass(frozen=True, slots=True)
class FuturePlane:
    """
    One explicitly supplied future influence plane.

    signed_effect:
        Direction and normalized effect strength in [-1, 1].

        Positive values amplify the candidate transition.
        Negative values attenuate it.
        Zero is neutral.

    intensity:
        Non-negative magnitude multiplier.

    confidence:
        Confidence in the future-plane estimate.

    multiplier_scale:
        Controls conversion from signed effect to a positive multiplier:

            multiplier = exp(
                signed_effect
                * intensity
                * multiplier_scale
            )
    """

    plane_id: str
    name: str
    horizon: ForecastHorizon

    signed_effect: float
    intensity: float = 1.0
    confidence: float = 1.0
    multiplier_scale: float = 1.0

    direction: FuturePlaneDirection = (
        FuturePlaneDirection.MIXED
    )
    temporal_profile: TemporalProfile = (
        TemporalProfile.CONSTANT
    )
    scope: FuturePlaneScope = FuturePlaneScope.EXTERNAL

    target_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    incompatible_candidate_ids: tuple[str, ...] = ()

    operational_risk: float = 0.0
    cascade_risk: float = 0.0
    observability: float = 1.0
    reversibility: float = 1.0

    non_fonit_allowed: bool = True
    status: FuturePlaneStatus = FuturePlaneStatus.ACTIVE

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    future_plane_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(
            self.horizon,
            ForecastHorizon,
        ):
            raise FuturePlaneAnalyzerError(
                "horizon must be a ForecastHorizon"
            )

        try:
            direction = FuturePlaneDirection(
                self.direction
            )
            temporal_profile = TemporalProfile(
                self.temporal_profile
            )
            scope = FuturePlaneScope(
                self.scope
            )
            status = FuturePlaneStatus(
                self.status
            )
        except (TypeError, ValueError) as exc:
            raise FuturePlaneAnalyzerError(
                "unsupported future-plane enum value"
            ) from exc

        if not isinstance(
            self.non_fonit_allowed,
            bool,
        ):
            raise FuturePlaneAnalyzerError(
                "non_fonit_allowed must be a boolean"
            )

        object.__setattr__(
            self,
            "plane_id",
            _text(
                self.plane_id,
                field_name="plane_id",
            ),
        )
        object.__setattr__(
            self,
            "name",
            _text(
                self.name,
                field_name="name",
            ),
        )
        object.__setattr__(
            self,
            "future_plane_id",
            _text(
                self.future_plane_id,
                field_name="future_plane_id",
            ),
        )
        object.__setattr__(
            self,
            "signed_effect",
            _signed_unit(
                self.signed_effect,
                field_name="signed_effect",
            ),
        )
        object.__setattr__(
            self,
            "intensity",
            _nonnegative(
                self.intensity,
                field_name="intensity",
            ),
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
            "multiplier_scale",
            _positive(
                self.multiplier_scale,
                field_name="multiplier_scale",
            ),
        )
        object.__setattr__(
            self,
            "direction",
            direction,
        )
        object.__setattr__(
            self,
            "temporal_profile",
            temporal_profile,
        )
        object.__setattr__(
            self,
            "scope",
            scope,
        )
        object.__setattr__(
            self,
            "status",
            status,
        )
        object.__setattr__(
            self,
            "target_ids",
            _ids(
                self.target_ids,
                field_name="target_ids",
            ),
        )
        object.__setattr__(
            self,
            "candidate_ids",
            _ids(
                self.candidate_ids,
                field_name="candidate_ids",
            ),
        )
        object.__setattr__(
            self,
            "incompatible_candidate_ids",
            _ids(
                self.incompatible_candidate_ids,
                field_name=(
                    "incompatible_candidate_ids"
                ),
            ),
        )
        object.__setattr__(
            self,
            "operational_risk",
            _unit(
                self.operational_risk,
                field_name="operational_risk",
            ),
        )
        object.__setattr__(
            self,
            "cascade_risk",
            _unit(
                self.cascade_risk,
                field_name="cascade_risk",
            ),
        )
        object.__setattr__(
            self,
            "observability",
            _unit(
                self.observability,
                field_name="observability",
            ),
        )
        object.__setattr__(
            self,
            "reversibility",
            _unit(
                self.reversibility,
                field_name="reversibility",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _mapping(
                self.metadata,
                field_name="metadata",
            ),
        )

    @property
    def is_eligible(self) -> bool:
        return (
            self.non_fonit_allowed
            and self.status
            in {
                FuturePlaneStatus.ACTIVE,
                FuturePlaneStatus.INFERRED,
                FuturePlaneStatus.INCOMPLETE,
            }
        )

    @property
    def safety_penalty(self) -> float:
        return min(
            (
                self.operational_risk
                + self.cascade_risk
                + (1.0 - self.reversibility)
            )
            / 3.0,
            1.0,
        )

    @property
    def base_multiplier(self) -> float:
        return exp(
            self.signed_effect
            * self.intensity
            * self.multiplier_scale
        )

    def temporal_weight(
        self,
        candidate_horizon: ForecastHorizon,
    ) -> float:
        """
        Return temporal applicability in [0, 1].
        """

        if not isinstance(
            candidate_horizon,
            ForecastHorizon,
        ):
            raise FuturePlaneAnalyzerError(
                "candidate_horizon must be a ForecastHorizon"
            )

        overlap = _interval_overlap_fraction(
            self.horizon.start,
            self.horizon.end,
            candidate_horizon.start,
            candidate_horizon.end,
        )

        if overlap <= 0.0:
            return 0.0

        if (
            self.temporal_profile
            is TemporalProfile.CONSTANT
        ):
            return overlap

        if (
            self.temporal_profile
            is TemporalProfile.WINDOWED
        ):
            return overlap

        midpoint = (
            max(
                self.horizon.start,
                candidate_horizon.start,
            )
            + min(
                self.horizon.end,
                candidate_horizon.end,
            )
        ) / 2.0

        plane_duration = max(
            self.horizon.duration,
            1e-12,
        )
        position = min(
            max(
                (
                    midpoint
                    - self.horizon.start
                )
                / plane_duration,
                0.0,
            ),
            1.0,
        )

        if (
            self.temporal_profile
            is TemporalProfile.RAMP_UP
        ):
            return overlap * position

        if (
            self.temporal_profile
            is TemporalProfile.RAMP_DOWN
        ):
            return overlap * (
                1.0 - position
            )

        if (
            self.temporal_profile
            is TemporalProfile.IMPULSE
        ):
            center = (
                self.horizon.start
                + self.horizon.end
            ) / 2.0
            half_width = max(
                self.horizon.duration / 2.0,
                1e-12,
            )
            impulse_weight = max(
                1.0
                - abs(midpoint - center)
                / half_width,
                0.0,
            )
            return overlap * impulse_weight

        return overlap

    def applies_to(
        self,
        candidate: ForecastCandidate,
    ) -> bool:
        if not isinstance(
            candidate,
            ForecastCandidate,
        ):
            raise FuturePlaneAnalyzerError(
                "candidate must be a ForecastCandidate"
            )

        if not self.is_eligible:
            return False

        if (
            candidate.candidate_id
            in self.incompatible_candidate_ids
        ):
            return False

        if (
            self.candidate_ids
            and candidate.candidate_id
            not in self.candidate_ids
        ):
            return False

        if (
            self.target_ids
            and candidate.affected_target_ids
            and not (
                set(self.target_ids)
                & set(candidate.affected_target_ids)
            )
        ):
            return False

        return (
            self.temporal_weight(
                candidate.horizon
            )
            > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "future_plane_id": (
                self.future_plane_id
            ),
            "plane_id": self.plane_id,
            "name": self.name,
            "horizon": self.horizon.to_dict(),
            "signed_effect": self.signed_effect,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "multiplier_scale": (
                self.multiplier_scale
            ),
            "direction": self.direction.value,
            "temporal_profile": (
                self.temporal_profile.value
            ),
            "scope": self.scope.value,
            "target_ids": list(self.target_ids),
            "candidate_ids": list(
                self.candidate_ids
            ),
            "incompatible_candidate_ids": list(
                self.incompatible_candidate_ids
            ),
            "operational_risk": (
                self.operational_risk
            ),
            "cascade_risk": self.cascade_risk,
            "observability": self.observability,
            "reversibility": self.reversibility,
            "non_fonit_allowed": (
                self.non_fonit_allowed
            ),
            "status": self.status.value,
            "base_multiplier": (
                self.base_multiplier
            ),
            "safety_penalty": (
                self.safety_penalty
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> FuturePlane:
        if not isinstance(data, Mapping):
            raise FuturePlaneAnalyzerError(
                "future-plane data must be a mapping"
            )

        version = data.get(
            "version",
            cls.VERSION,
        )

        if version != cls.VERSION:
            raise FuturePlaneAnalyzerError(
                "unsupported future plane version: "
                f"{version!r}"
            )

        horizon_data = data.get("horizon")

        if not isinstance(
            horizon_data,
            Mapping,
        ):
            raise FuturePlaneAnalyzerError(
                "horizon is missing or invalid"
            )

        return cls(
            future_plane_id=str(
                data["future_plane_id"]
            ),
            plane_id=str(data["plane_id"]),
            name=str(data["name"]),
            horizon=ForecastHorizon.from_dict(
                horizon_data
            ),
            signed_effect=float(
                data["signed_effect"]
            ),
            intensity=float(
                data.get("intensity", 1.0)
            ),
            confidence=float(
                data.get("confidence", 1.0)
            ),
            multiplier_scale=float(
                data.get(
                    "multiplier_scale",
                    1.0,
                )
            ),
            direction=FuturePlaneDirection(
                data.get(
                    "direction",
                    FuturePlaneDirection.MIXED.value,
                )
            ),
            temporal_profile=TemporalProfile(
                data.get(
                    "temporal_profile",
                    TemporalProfile.CONSTANT.value,
                )
            ),
            scope=FuturePlaneScope(
                data.get(
                    "scope",
                    FuturePlaneScope.EXTERNAL.value,
                )
            ),
            target_ids=tuple(
                data.get("target_ids", ())
            ),
            candidate_ids=tuple(
                data.get("candidate_ids", ())
            ),
            incompatible_candidate_ids=tuple(
                data.get(
                    "incompatible_candidate_ids",
                    (),
                )
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
            observability=float(
                data.get(
                    "observability",
                    1.0,
                )
            ),
            reversibility=float(
                data.get(
                    "reversibility",
                    1.0,
                )
            ),
            non_fonit_allowed=bool(
                data.get(
                    "non_fonit_allowed",
                    True,
                )
            ),
            status=FuturePlaneStatus(
                data.get(
                    "status",
                    FuturePlaneStatus.ACTIVE.value,
                )
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class FuturePlaneContribution:
    """
    Effective contribution of one plane to one forecast candidate.
    """

    future_plane: FuturePlane
    candidate_id: str

    temporal_weight: float
    target_weight: float
    confidence_weight: float
    observability_weight: float
    safety_weight: float

    effective_exponent: float
    effective_multiplier: float
    applied: bool
    veto_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.future_plane,
            FuturePlane,
        ):
            raise FuturePlaneAnalyzerError(
                "future_plane must be a FuturePlane"
            )

        candidate_id = _text(
            self.candidate_id,
            field_name="candidate_id",
        )

        for field_name in (
            "temporal_weight",
            "target_weight",
            "confidence_weight",
            "observability_weight",
            "safety_weight",
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

        effective_exponent = _finite(
            self.effective_exponent,
            field_name="effective_exponent",
        )
        effective_multiplier = _positive(
            self.effective_multiplier,
            field_name="effective_multiplier",
        )

        if not isinstance(
            self.applied,
            bool,
        ):
            raise FuturePlaneAnalyzerError(
                "applied must be a boolean"
            )

        veto_reason = self.veto_reason

        if veto_reason is not None:
            veto_reason = _text(
                veto_reason,
                field_name="veto_reason",
            )

        object.__setattr__(
            self,
            "candidate_id",
            candidate_id,
        )
        object.__setattr__(
            self,
            "effective_exponent",
            effective_exponent,
        )
        object.__setattr__(
            self,
            "effective_multiplier",
            effective_multiplier,
        )
        object.__setattr__(
            self,
            "veto_reason",
            veto_reason,
        )

    def to_plane_influence(
        self,
    ) -> PlaneInfluence:
        """
        Convert the contribution to HistoryForecaster input.
        """

        return PlaneInfluence(
            plane_id=self.future_plane.plane_id,
            multiplier=self.effective_multiplier,
            weight=1.0,
            confidence=1.0,
            observed=(
                self.future_plane.status
                is FuturePlaneStatus.ACTIVE
            ),
            metadata={
                "future_plane_id": (
                    self.future_plane.future_plane_id
                ),
                "candidate_id": self.candidate_id,
                "applied": self.applied,
                "effective_exponent": (
                    self.effective_exponent
                ),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "future_plane_id": (
                self.future_plane.future_plane_id
            ),
            "plane_id": (
                self.future_plane.plane_id
            ),
            "candidate_id": self.candidate_id,
            "temporal_weight": (
                self.temporal_weight
            ),
            "target_weight": self.target_weight,
            "confidence_weight": (
                self.confidence_weight
            ),
            "observability_weight": (
                self.observability_weight
            ),
            "safety_weight": self.safety_weight,
            "effective_exponent": (
                self.effective_exponent
            ),
            "effective_multiplier": (
                self.effective_multiplier
            ),
            "applied": self.applied,
            "veto_reason": self.veto_reason,
        }


@dataclass(frozen=True, slots=True)
class CandidatePlaneAnalysis:
    """
    Complete future-plane analysis for one forecast candidate.
    """

    candidate_id: str
    contributions: tuple[
        FuturePlaneContribution,
        ...
    ] = ()

    combined_multiplier: float = 1.0
    combined_log_multiplier: float = 0.0
    applied_plane_count: int = 0
    vetoed_plane_count: int = 0
    mean_observability: float = 1.0
    mean_safety: float = 1.0

    analysis_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        candidate_id = _text(
            self.candidate_id,
            field_name="candidate_id",
        )
        analysis_id = _text(
            self.analysis_id,
            field_name="analysis_id",
        )

        contributions = tuple(
            self.contributions
        )

        if not all(
            isinstance(
                item,
                FuturePlaneContribution,
            )
            for item in contributions
        ):
            raise FuturePlaneAnalyzerError(
                "all contributions must be FuturePlaneContribution objects"
            )

        contribution_ids = [
            item.future_plane.future_plane_id
            for item in contributions
        ]

        if len(
            contribution_ids
        ) != len(
            set(contribution_ids)
        ):
            raise FuturePlaneAnalyzerError(
                "future_plane_id values must be unique per candidate"
            )

        if not isinstance(
            self.applied_plane_count,
            int,
        ) or self.applied_plane_count < 0:
            raise FuturePlaneAnalyzerError(
                "applied_plane_count must be a non-negative integer"
            )

        if not isinstance(
            self.vetoed_plane_count,
            int,
        ) or self.vetoed_plane_count < 0:
            raise FuturePlaneAnalyzerError(
                "vetoed_plane_count must be a non-negative integer"
            )

        object.__setattr__(
            self,
            "candidate_id",
            candidate_id,
        )
        object.__setattr__(
            self,
            "analysis_id",
            analysis_id,
        )
        object.__setattr__(
            self,
            "contributions",
            contributions,
        )
        object.__setattr__(
            self,
            "combined_multiplier",
            _positive(
                self.combined_multiplier,
                field_name="combined_multiplier",
            ),
        )
        object.__setattr__(
            self,
            "combined_log_multiplier",
            _finite(
                self.combined_log_multiplier,
                field_name=(
                    "combined_log_multiplier"
                ),
            ),
        )
        object.__setattr__(
            self,
            "mean_observability",
            _unit(
                self.mean_observability,
                field_name="mean_observability",
            ),
        )
        object.__setattr__(
            self,
            "mean_safety",
            _unit(
                self.mean_safety,
                field_name="mean_safety",
            ),
        )

    @property
    def plane_influence_set(
        self,
    ) -> PlaneInfluenceSet:
        return PlaneInfluenceSet(
            influence_set_id=(
                f"future-{self.analysis_id}"
            ),
            label=(
                f"Future planes for "
                f"{self.candidate_id}"
            ),
            influences=tuple(
                item.to_plane_influence()
                for item in self.contributions
                if item.applied
            ),
            metadata={
                "candidate_id": (
                    self.candidate_id
                ),
                "combined_multiplier": (
                    self.combined_multiplier
                ),
                "applied_plane_count": (
                    self.applied_plane_count
                ),
                "vetoed_plane_count": (
                    self.vetoed_plane_count
                ),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "analysis_id": self.analysis_id,
            "candidate_id": self.candidate_id,
            "combined_multiplier": (
                self.combined_multiplier
            ),
            "combined_log_multiplier": (
                self.combined_log_multiplier
            ),
            "applied_plane_count": (
                self.applied_plane_count
            ),
            "vetoed_plane_count": (
                self.vetoed_plane_count
            ),
            "mean_observability": (
                self.mean_observability
            ),
            "mean_safety": self.mean_safety,
            "contributions": [
                item.to_dict()
                for item in self.contributions
            ],
        }


@dataclass(frozen=True, slots=True)
class FuturePlaneAnalysisResult:
    """
    Immutable result for all analyzed forecast candidates.
    """

    status: FuturePlaneAnalysisStatus
    candidate_analyses: tuple[
        CandidatePlaneAnalysis,
        ...
    ] = ()

    plane_count: int = 0
    eligible_plane_count: int = 0
    vetoed_plane_count: int = 0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    result_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        try:
            status = FuturePlaneAnalysisStatus(
                self.status
            )
        except (TypeError, ValueError) as exc:
            raise FuturePlaneAnalyzerError(
                "unsupported analysis status"
            ) from exc

        candidate_analyses = tuple(
            self.candidate_analyses
        )

        if not all(
            isinstance(
                item,
                CandidatePlaneAnalysis,
            )
            for item in candidate_analyses
        ):
            raise FuturePlaneAnalyzerError(
                "all candidate analyses must be CandidatePlaneAnalysis objects"
            )

        candidate_ids = [
            item.candidate_id
            for item in candidate_analyses
        ]

        if len(
            candidate_ids
        ) != len(
            set(candidate_ids)
        ):
            raise FuturePlaneAnalyzerError(
                "candidate_id values must be unique"
            )

        for field_name in (
            "plane_count",
            "eligible_plane_count",
            "vetoed_plane_count",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                not isinstance(value, int)
                or value < 0
            ):
                raise FuturePlaneAnalyzerError(
                    f"{field_name} must be a non-negative integer"
                )

        object.__setattr__(
            self,
            "status",
            status,
        )
        object.__setattr__(
            self,
            "candidate_analyses",
            candidate_analyses,
        )
        object.__setattr__(
            self,
            "result_id",
            _text(
                self.result_id,
                field_name="result_id",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _mapping(
                self.metadata,
                field_name="metadata",
            ),
        )

    def for_candidate(
        self,
        candidate_id: str,
    ) -> CandidatePlaneAnalysis | None:
        normalized = _text(
            candidate_id,
            field_name="candidate_id",
        )

        for analysis in self.candidate_analyses:
            if (
                analysis.candidate_id
                == normalized
            ):
                return analysis

        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "result_id": self.result_id,
            "status": self.status.value,
            "plane_count": self.plane_count,
            "eligible_plane_count": (
                self.eligible_plane_count
            ),
            "vetoed_plane_count": (
                self.vetoed_plane_count
            ),
            "candidate_analyses": [
                item.to_dict()
                for item in self.candidate_analyses
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FuturePlaneAnalyzer:
    """
    Analyze future planes for forecast candidates.

    target_overlap_floor:
        Weight used when a plane has no explicit targets or the candidate has
        no explicit affected targets.

    safety_threshold:
        Maximum allowed safety penalty before the plane is vetoed.

    missing_observation_penalty:
        Multiplier applied to inferred or incomplete planes.

    non_fonit_required:
        When True, planes with non_fonit_allowed=False are vetoed.
    """

    target_overlap_floor: float = 1.0
    safety_threshold: float = 0.75
    missing_observation_penalty: float = 0.75
    non_fonit_required: bool = True

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_overlap_floor",
            _unit(
                self.target_overlap_floor,
                field_name="target_overlap_floor",
            ),
        )
        object.__setattr__(
            self,
            "safety_threshold",
            _unit(
                self.safety_threshold,
                field_name="safety_threshold",
            ),
        )
        object.__setattr__(
            self,
            "missing_observation_penalty",
            _unit(
                self.missing_observation_penalty,
                field_name=(
                    "missing_observation_penalty"
                ),
            ),
        )

        if not isinstance(
            self.non_fonit_required,
            bool,
        ):
            raise FuturePlaneAnalyzerError(
                "non_fonit_required must be a boolean"
            )

    def _target_weight(
        self,
        plane: FuturePlane,
        candidate: ForecastCandidate,
    ) -> float:
        if (
            not plane.target_ids
            or not candidate.affected_target_ids
        ):
            return self.target_overlap_floor

        overlap = (
            set(plane.target_ids)
            & set(
                candidate.affected_target_ids
            )
        )

        if not overlap:
            return 0.0

        return min(
            len(overlap)
            / max(
                len(plane.target_ids),
                len(
                    candidate.affected_target_ids
                ),
                1,
            ),
            1.0,
        )

    def analyze_contribution(
        self,
        plane: FuturePlane,
        candidate: ForecastCandidate,
    ) -> FuturePlaneContribution:
        if not isinstance(
            plane,
            FuturePlane,
        ):
            raise FuturePlaneAnalyzerError(
                "plane must be a FuturePlane"
            )

        if not isinstance(
            candidate,
            ForecastCandidate,
        ):
            raise FuturePlaneAnalyzerError(
                "candidate must be a ForecastCandidate"
            )

        veto_reason: str | None = None

        if (
            self.non_fonit_required
            and not plane.non_fonit_allowed
        ):
            veto_reason = "non_fonit_gate"

        elif (
            plane.status
            is FuturePlaneStatus.VETOED
        ):
            veto_reason = "plane_status_vetoed"

        elif (
            plane.status
            in {
                FuturePlaneStatus.DISABLED,
                FuturePlaneStatus.FAILED,
            }
        ):
            veto_reason = (
                f"plane_status_{plane.status.value}"
            )

        elif (
            plane.safety_penalty
            > self.safety_threshold
        ):
            veto_reason = "safety_threshold"

        elif not plane.applies_to(candidate):
            veto_reason = "not_applicable"

        temporal_weight = (
            plane.temporal_weight(
                candidate.horizon
            )
            if veto_reason is None
            else 0.0
        )

        target_weight = (
            self._target_weight(
                plane,
                candidate,
            )
            if veto_reason is None
            else 0.0
        )

        confidence_weight = (
            plane.confidence
        )
        observability_weight = (
            plane.observability
        )

        if (
            plane.status
            in {
                FuturePlaneStatus.INFERRED,
                FuturePlaneStatus.INCOMPLETE,
            }
        ):
            observability_weight *= (
                self.missing_observation_penalty
            )

        safety_weight = (
            1.0
            - plane.safety_penalty
        )

        applied = (
            veto_reason is None
            and temporal_weight > 0.0
            and target_weight > 0.0
        )

        if applied:
            effective_exponent = (
                plane.signed_effect
                * plane.intensity
                * plane.multiplier_scale
                * temporal_weight
                * target_weight
                * confidence_weight
                * observability_weight
                * safety_weight
            )
            effective_multiplier = exp(
                effective_exponent
            )
        else:
            effective_exponent = 0.0
            effective_multiplier = 1.0

        return FuturePlaneContribution(
            future_plane=plane,
            candidate_id=(
                candidate.candidate_id
            ),
            temporal_weight=(
                temporal_weight
            ),
            target_weight=target_weight,
            confidence_weight=(
                confidence_weight
            ),
            observability_weight=(
                observability_weight
            ),
            safety_weight=safety_weight,
            effective_exponent=(
                effective_exponent
            ),
            effective_multiplier=(
                effective_multiplier
            ),
            applied=applied,
            veto_reason=veto_reason,
        )

    def analyze_candidate(
        self,
        candidate: ForecastCandidate,
        planes: Iterable[FuturePlane],
    ) -> CandidatePlaneAnalysis:
        if not isinstance(
            candidate,
            ForecastCandidate,
        ):
            raise FuturePlaneAnalyzerError(
                "candidate must be a ForecastCandidate"
            )

        plane_tuple = tuple(planes)

        if not all(
            isinstance(item, FuturePlane)
            for item in plane_tuple
        ):
            raise FuturePlaneAnalyzerError(
                "all planes must be FuturePlane objects"
            )

        future_plane_ids = [
            item.future_plane_id
            for item in plane_tuple
        ]

        if len(
            future_plane_ids
        ) != len(
            set(future_plane_ids)
        ):
            raise FuturePlaneAnalyzerError(
                "future_plane_id values must be unique"
            )

        contributions = tuple(
            self.analyze_contribution(
                plane,
                candidate,
            )
            for plane in plane_tuple
        )

        applied = tuple(
            item
            for item in contributions
            if item.applied
        )

        combined_log_multiplier = sum(
            log(
                item.effective_multiplier
            )
            for item in applied
        )
        combined_multiplier = exp(
            combined_log_multiplier
        )

        applied_plane_count = len(
            applied
        )
        vetoed_plane_count = sum(
            item.veto_reason is not None
            and item.veto_reason
            != "not_applicable"
            for item in contributions
        )

        if applied:
            mean_observability = (
                sum(
                    item.observability_weight
                    for item in applied
                )
                / len(applied)
            )
            mean_safety = (
                sum(
                    item.safety_weight
                    for item in applied
                )
                / len(applied)
            )
        else:
            mean_observability = 1.0
            mean_safety = 1.0

        return CandidatePlaneAnalysis(
            candidate_id=(
                candidate.candidate_id
            ),
            contributions=contributions,
            combined_multiplier=(
                combined_multiplier
            ),
            combined_log_multiplier=(
                combined_log_multiplier
            ),
            applied_plane_count=(
                applied_plane_count
            ),
            vetoed_plane_count=(
                vetoed_plane_count
            ),
            mean_observability=(
                mean_observability
            ),
            mean_safety=mean_safety,
        )

    def analyze(
        self,
        candidates: Iterable[
            ForecastCandidate
        ],
        planes: Iterable[FuturePlane],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> FuturePlaneAnalysisResult:
        candidate_tuple = tuple(candidates)
        plane_tuple = tuple(planes)

        if not all(
            isinstance(
                item,
                ForecastCandidate,
            )
            for item in candidate_tuple
        ):
            raise FuturePlaneAnalyzerError(
                "all candidates must be ForecastCandidate objects"
            )

        if not all(
            isinstance(
                item,
                FuturePlane,
            )
            for item in plane_tuple
        ):
            raise FuturePlaneAnalyzerError(
                "all planes must be FuturePlane objects"
            )

        candidate_ids = [
            item.candidate_id
            for item in candidate_tuple
        ]

        if len(
            candidate_ids
        ) != len(
            set(candidate_ids)
        ):
            raise FuturePlaneAnalyzerError(
                "candidate_id values must be unique"
            )

        future_plane_ids = [
            item.future_plane_id
            for item in plane_tuple
        ]

        if len(
            future_plane_ids
        ) != len(
            set(future_plane_ids)
        ):
            raise FuturePlaneAnalyzerError(
                "future_plane_id values must be unique"
            )

        if not plane_tuple:
            return FuturePlaneAnalysisResult(
                status=(
                    FuturePlaneAnalysisStatus
                    .NO_PLANES
                ),
                plane_count=0,
                eligible_plane_count=0,
                vetoed_plane_count=0,
                metadata=metadata or {},
            )

        analyses = tuple(
            self.analyze_candidate(
                candidate,
                plane_tuple,
            )
            for candidate in candidate_tuple
        )

        eligible_plane_count = sum(
            plane.is_eligible
            for plane in plane_tuple
        )
        vetoed_plane_count = sum(
            (
                not plane.non_fonit_allowed
                or plane.status
                is FuturePlaneStatus.VETOED
                or plane.safety_penalty
                > self.safety_threshold
            )
            for plane in plane_tuple
        )

        total_applied = sum(
            item.applied_plane_count
            for item in analyses
        )

        if (
            vetoed_plane_count
            == len(plane_tuple)
        ):
            status = (
                FuturePlaneAnalysisStatus
                .ALL_VETOED
            )
        elif total_applied == 0:
            status = (
                FuturePlaneAnalysisStatus
                .NO_APPLICABLE_PLANES
            )
        elif (
            vetoed_plane_count > 0
            or any(
                plane.status
                in {
                    FuturePlaneStatus.INFERRED,
                    FuturePlaneStatus.INCOMPLETE,
                }
                for plane in plane_tuple
            )
        ):
            status = (
                FuturePlaneAnalysisStatus
                .PARTIAL
            )
        else:
            status = (
                FuturePlaneAnalysisStatus
                .COMPLETED
            )

        merged_metadata = {
            "analyzer_version": self.VERSION,
            "candidate_count": len(
                candidate_tuple
            ),
            "plane_count": len(
                plane_tuple
            ),
            "eligible_plane_count": (
                eligible_plane_count
            ),
            "vetoed_plane_count": (
                vetoed_plane_count
            ),
            "total_applied_contributions": (
                total_applied
            ),
            "non_fonit_required": (
                self.non_fonit_required
            ),
        }

        if metadata is not None:
            merged_metadata.update(
                dict(metadata)
            )

        return FuturePlaneAnalysisResult(
            status=status,
            candidate_analyses=analyses,
            plane_count=len(plane_tuple),
            eligible_plane_count=(
                eligible_plane_count
            ),
            vetoed_plane_count=(
                vetoed_plane_count
            ),
            metadata=merged_metadata,
        )

    def build_influence_sets(
        self,
        result: FuturePlaneAnalysisResult,
    ) -> Mapping[str, PlaneInfluenceSet]:
        """
        Return candidate_id -> PlaneInfluenceSet.
        """

        if not isinstance(
            result,
            FuturePlaneAnalysisResult,
        ):
            raise FuturePlaneAnalyzerError(
                "result must be a FuturePlaneAnalysisResult"
            )

        return MappingProxyType(
            {
                analysis.candidate_id: (
                    analysis.plane_influence_set
                )
                for analysis
                in result.candidate_analyses
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "target_overlap_floor": (
                self.target_overlap_floor
            ),
            "safety_threshold": (
                self.safety_threshold
            ),
            "missing_observation_penalty": (
                self.missing_observation_penalty
            ),
            "non_fonit_required": (
                self.non_fonit_required
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> FuturePlaneAnalyzer:
        if not isinstance(data, Mapping):
            raise FuturePlaneAnalyzerError(
                "future-plane-analyzer data must be a mapping"
            )

        version = data.get(
            "version",
            cls.VERSION,
        )

        if version != cls.VERSION:
            raise FuturePlaneAnalyzerError(
                "unsupported future-plane analyzer version: "
                f"{version!r}"
            )

        return cls(
            target_overlap_floor=float(
                data.get(
                    "target_overlap_floor",
                    1.0,
                )
            ),
            safety_threshold=float(
                data.get(
                    "safety_threshold",
                    0.75,
                )
            ),
            missing_observation_penalty=float(
                data.get(
                    "missing_observation_penalty",
                    0.75,
                )
            ),
            non_fonit_required=bool(
                data.get(
                    "non_fonit_required",
                    True,
                )
            ),
        )


__all__ = [
    "CandidatePlaneAnalysis",
    "FuturePlane",
    "FuturePlaneAnalysisResult",
    "FuturePlaneAnalysisStatus",
    "FuturePlaneAnalyzer",
    "FuturePlaneAnalyzerError",
    "FuturePlaneContribution",
    "FuturePlaneDirection",
    "FuturePlaneScope",
    "FuturePlaneStatus",
    "TemporalProfile",
]
