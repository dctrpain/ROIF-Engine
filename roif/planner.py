"""Deterministic and explainable intervention planning for ROIF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

from .analyzer import AnalysisRisk, CascadeAnalysis, CascadePhase, CoherenceState
from .predictor import CascadePrediction, PredictionStatus, StabilityForecast


class PlannerError(ValueError):
    """Raised when planner data or configuration is invalid."""


_EPS = 1e-12


def _number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise PlannerError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PlannerError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise PlannerError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise PlannerError(f"{name} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise PlannerError(f"{name} must be <= {maximum}.")
    return result


def _text(value: Any, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise PlannerError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise PlannerError(f"{name} cannot be empty.")
    return result


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise PlannerError("metadata must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        copied[_text(key, name="metadata key")] = item
    return MappingProxyType(copied)


def _tags(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise PlannerError("tags must be a sequence of strings.")
    try:
        source = tuple(value)
    except TypeError as exc:
        raise PlannerError("tags must be a sequence of strings.") from exc
    result: list[str] = []
    for item in source:
        normalized = _text(item, name="tag")
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


class PlanStatus(str, Enum):
    AVAILABLE = "available"
    NO_CANDIDATES = "no_candidates"
    ALL_VETOED = "all_vetoed"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class CandidateDecision(str, Enum):
    RECOMMENDED = "recommended"
    ALTERNATIVE = "alternative"
    REJECTED = "rejected"
    VETOED = "vetoed"


class VetoReason(str, Enum):
    CASCADE_RISK = "cascade_risk"
    EXTERNAL_SYSTEM_RISK = "external_system_risk"
    IRREVERSIBLE_HIGH_RISK = "irreversible_high_risk"
    BELOW_MINIMUM_EVIDENCE = "below_minimum_evidence"
    NON_POSITIVE_GAIN = "non_positive_gain"


class TargetRole(str, Enum):
    NODE_STAR = "node_star"
    D_FAST = "d_fast"
    D_ROOT = "d_root"
    SUPPORT = "support"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    benefit_weight: float = 0.35
    urgency_weight: float = 0.15
    target_weight: float = 0.15
    evidence_weight: float = 0.10
    reversibility_weight: float = 0.05
    cost_weight: float = 0.08
    risk_weight: float = 0.08
    delay_weight: float = 0.04
    maximum_cascade_risk: float = 0.70
    maximum_external_system_risk: float = 0.20
    minimum_evidence_strength: float = 0.0
    minimum_expected_eta_gain: float = 0.0
    irreversible_risk_threshold: float = 0.35
    max_alternatives: int = 3
    require_prediction: bool = False
    non_fonit_gate_enabled: bool = True

    def __post_init__(self) -> None:
        weights = (
            "benefit_weight", "urgency_weight", "target_weight",
            "evidence_weight", "reversibility_weight", "cost_weight",
            "risk_weight", "delay_weight",
        )
        for name in weights:
            object.__setattr__(
                self, name, _number(getattr(self, name), name=name, minimum=0.0)
            )
        if self.weight_sum <= _EPS:
            raise PlannerError("At least one planner weight must be positive.")

        for name in (
            "maximum_cascade_risk", "maximum_external_system_risk",
            "minimum_evidence_strength", "irreversible_risk_threshold",
        ):
            object.__setattr__(
                self, name,
                _number(getattr(self, name), name=name, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(
            self, "minimum_expected_eta_gain",
            _number(self.minimum_expected_eta_gain, name="minimum_expected_eta_gain"),
        )
        if (
            isinstance(self.max_alternatives, bool)
            or not isinstance(self.max_alternatives, int)
            or self.max_alternatives < 0
        ):
            raise PlannerError("max_alternatives must be a non-negative integer.")
        for name in ("require_prediction", "non_fonit_gate_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise PlannerError(f"{name} must be a bool.")

    @property
    def weight_sum(self) -> float:
        return sum((
            self.benefit_weight, self.urgency_weight, self.target_weight,
            self.evidence_weight, self.reversibility_weight, self.cost_weight,
            self.risk_weight, self.delay_weight,
        ))


@dataclass(frozen=True, slots=True)
class InterventionCandidate:
    candidate_id: str
    target_id: str
    target_role: TargetRole
    expected_eta_gain: float
    cost: float = 0.0
    risk: float = 0.0
    delay: float = 0.0
    evidence_strength: float = 0.0
    reversibility: float = 1.0
    cascade_risk: float = 0.0
    external_system_risk: float = 0.0
    rationale: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, name="candidate_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, name="target_id"))
        if not isinstance(self.target_role, TargetRole):
            raise PlannerError("target_role must be a TargetRole.")
        object.__setattr__(
            self, "expected_eta_gain",
            _number(self.expected_eta_gain, name="expected_eta_gain"),
        )
        for name in ("cost", "delay"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name=name, minimum=0.0)
            )
        for name in (
            "risk", "evidence_strength", "reversibility",
            "cascade_risk", "external_system_risk",
        ):
            object.__setattr__(
                self, name,
                _number(getattr(self, name), name=name, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(
            self, "rationale", _text(self.rationale, name="rationale", optional=True)
        )
        object.__setattr__(self, "tags", _tags(self.tags))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def is_reversible(self) -> bool:
        return self.reversibility >= 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_id": self.target_id,
            "target_role": self.target_role.value,
            "expected_eta_gain": self.expected_eta_gain,
            "cost": self.cost,
            "risk": self.risk,
            "delay": self.delay,
            "evidence_strength": self.evidence_strength,
            "reversibility": self.reversibility,
            "cascade_risk": self.cascade_risk,
            "external_system_risk": self.external_system_risk,
            "rationale": self.rationale,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    benefit: float
    urgency: float
    target_alignment: float
    evidence: float
    reversibility: float
    cost_penalty: float
    risk_penalty: float
    delay_penalty: float
    utility: float

    def __post_init__(self) -> None:
        for name in (
            "benefit", "urgency", "target_alignment", "evidence",
            "reversibility", "cost_penalty", "risk_penalty", "delay_penalty",
        ):
            object.__setattr__(
                self, name,
                _number(getattr(self, name), name=name, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(self, "utility", _number(self.utility, name="utility"))

    def as_dict(self) -> dict[str, float]:
        return {
            "benefit": self.benefit,
            "urgency": self.urgency,
            "target_alignment": self.target_alignment,
            "evidence": self.evidence,
            "reversibility": self.reversibility,
            "cost_penalty": self.cost_penalty,
            "risk_penalty": self.risk_penalty,
            "delay_penalty": self.delay_penalty,
            "utility": self.utility,
        }


@dataclass(frozen=True, slots=True)
class RankedIntervention:
    rank: int | None
    decision: CandidateDecision
    candidate: InterventionCandidate
    score: ScoreBreakdown | None
    veto_reasons: tuple[VetoReason, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise PlannerError("rank must be None or a positive integer.")
        if not isinstance(self.decision, CandidateDecision):
            raise PlannerError("decision must be a CandidateDecision.")
        if not isinstance(self.candidate, InterventionCandidate):
            raise PlannerError("candidate must be an InterventionCandidate.")
        if self.score is not None and not isinstance(self.score, ScoreBreakdown):
            raise PlannerError("score must be a ScoreBreakdown or None.")
        reasons = tuple(self.veto_reasons)
        if any(not isinstance(item, VetoReason) for item in reasons):
            raise PlannerError("veto_reasons must contain VetoReason values.")
        object.__setattr__(self, "veto_reasons", reasons)
        object.__setattr__(self, "explanation", _text(self.explanation, name="explanation"))

        if self.decision is CandidateDecision.VETOED:
            if not reasons or self.rank is not None or self.score is not None:
                raise PlannerError("Vetoed items require reasons and no rank or score.")
        elif reasons or self.rank is None or self.score is None:
            raise PlannerError("Non-vetoed items require rank and score only.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "decision": self.decision.value,
            "candidate": self.candidate.as_dict(),
            "score": None if self.score is None else self.score.as_dict(),
            "veto_reasons": [item.value for item in self.veto_reasons],
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class InterventionPlan:
    status: PlanStatus
    analysis_phase: CascadePhase
    analysis_risk: AnalysisRisk
    prediction_stability: StabilityForecast | None
    recommended: RankedIntervention | None
    alternatives: tuple[RankedIntervention, ...]
    rejected: tuple[RankedIntervention, ...]
    vetoed: tuple[RankedIntervention, ...]
    candidate_count: int
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PlanStatus):
            raise PlannerError("status must be a PlanStatus.")
        if not isinstance(self.analysis_phase, CascadePhase):
            raise PlannerError("analysis_phase must be a CascadePhase.")
        if not isinstance(self.analysis_risk, AnalysisRisk):
            raise PlannerError("analysis_risk must be an AnalysisRisk.")
        if self.prediction_stability is not None and not isinstance(
            self.prediction_stability, StabilityForecast
        ):
            raise PlannerError("prediction_stability must be a StabilityForecast or None.")
        if self.recommended is not None and not isinstance(
            self.recommended, RankedIntervention
        ):
            raise PlannerError("recommended must be a RankedIntervention or None.")
        for name in ("alternatives", "rejected", "vetoed"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, RankedIntervention) for item in values):
                raise PlannerError(f"{name} must contain RankedIntervention values.")
            object.__setattr__(self, name, values)
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
        ):
            raise PlannerError("candidate_count must be a non-negative integer.")
        if self.recommended is not None and (
            self.recommended.decision is not CandidateDecision.RECOMMENDED
        ):
            raise PlannerError("recommended must have RECOMMENDED decision.")
        if any(x.decision is not CandidateDecision.ALTERNATIVE for x in self.alternatives):
            raise PlannerError("alternatives must have ALTERNATIVE decision.")
        if any(x.decision is not CandidateDecision.REJECTED for x in self.rejected):
            raise PlannerError("rejected must have REJECTED decision.")
        if any(x.decision is not CandidateDecision.VETOED for x in self.vetoed):
            raise PlannerError("vetoed must have VETOED decision.")
        object.__setattr__(self, "summary", _text(self.summary, name="summary"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def is_available(self) -> bool:
        return self.status is PlanStatus.AVAILABLE

    @property
    def node_star(self) -> str | None:
        return None if self.recommended is None else self.recommended.candidate.target_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "analysis_phase": self.analysis_phase.value,
            "analysis_risk": self.analysis_risk.value,
            "prediction_stability": (
                None if self.prediction_stability is None
                else self.prediction_stability.value
            ),
            "recommended": (
                None if self.recommended is None else self.recommended.as_dict()
            ),
            "alternatives": [x.as_dict() for x in self.alternatives],
            "rejected": [x.as_dict() for x in self.rejected],
            "vetoed": [x.as_dict() for x in self.vetoed],
            "candidate_count": self.candidate_count,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


def _urgency(
    analysis: CascadeAnalysis,
    prediction: CascadePrediction | None,
) -> float:
    scores = {
        CascadePhase.QUIESCENT: 0.0,
        CascadePhase.COMPENSATING: 0.35,
        CascadePhase.DESTABILIZING: 0.75,
        CascadePhase.SUPERCRITICAL: 1.0,
        CascadePhase.RECOVERING: 0.20,
        CascadePhase.INDETERMINATE: 0.25,
    }
    value = scores[analysis.phase]
    if analysis.coherence_state is CoherenceState.CRITICAL:
        value = 1.0
    elif analysis.coherence_state is CoherenceState.WARNING:
        value = max(value, 0.7)
    if prediction is not None:
        if prediction.stability is StabilityForecast.CRITICAL:
            value = 1.0
        elif prediction.stability is StabilityForecast.WARNING:
            value = max(value, 0.8)
        if prediction.critical_probability is not None:
            value = max(value, prediction.critical_probability)
    return min(1.0, value)


def _alignment(candidate: InterventionCandidate, analysis: CascadeAnalysis) -> float:
    if candidate.target_role is TargetRole.NODE_STAR:
        return 1.0
    if candidate.target_role is TargetRole.D_ROOT:
        return 0.8
    if candidate.target_role is TargetRole.SUPPORT:
        return 0.5
    if candidate.target_role is TargetRole.D_FAST:
        dominant = analysis.dominant_d_fast
        if dominant is not None and candidate.target_id == dominant.plane_id:
            return 0.9
        return 0.35
    return 0.2


def _veto(candidate: InterventionCandidate, config: PlannerConfig) -> tuple[VetoReason, ...]:
    result: list[VetoReason] = []
    if candidate.evidence_strength < config.minimum_evidence_strength:
        result.append(VetoReason.BELOW_MINIMUM_EVIDENCE)
    if candidate.expected_eta_gain <= config.minimum_expected_eta_gain:
        result.append(VetoReason.NON_POSITIVE_GAIN)
    if config.non_fonit_gate_enabled:
        if candidate.cascade_risk > config.maximum_cascade_risk:
            result.append(VetoReason.CASCADE_RISK)
        if candidate.external_system_risk > config.maximum_external_system_risk:
            result.append(VetoReason.EXTERNAL_SYSTEM_RISK)
        if (
            candidate.reversibility < 0.5
            and candidate.risk > config.irreversible_risk_threshold
        ):
            result.append(VetoReason.IRREVERSIBLE_HIGH_RISK)
    return tuple(result)


class CascadePlanner:
    """Rank supplied interventions and select an operational Node*."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        if config is None:
            config = PlannerConfig()
        if not isinstance(config, PlannerConfig):
            raise PlannerError("config must be a PlannerConfig.")
        self._config = config

    @property
    def config(self) -> PlannerConfig:
        return self._config

    def plan(
        self,
        analysis: CascadeAnalysis,
        candidates: Sequence[InterventionCandidate],
        *,
        prediction: CascadePrediction | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> InterventionPlan:
        if not isinstance(analysis, CascadeAnalysis):
            raise PlannerError("analysis must be a CascadeAnalysis.")
        if prediction is not None and not isinstance(prediction, CascadePrediction):
            raise PlannerError("prediction must be a CascadePrediction or None.")
        if isinstance(candidates, (str, bytes)):
            raise PlannerError("candidates must be a sequence.")
        try:
            candidates = tuple(candidates)
        except TypeError as exc:
            raise PlannerError("candidates must be a sequence.") from exc
        if any(not isinstance(x, InterventionCandidate) for x in candidates):
            raise PlannerError("candidates must contain InterventionCandidate values.")
        ids = [x.candidate_id for x in candidates]
        if len(ids) != len(set(ids)):
            raise PlannerError("candidate_id values must be unique.")

        stability = None if prediction is None else prediction.stability
        base = dict(
            analysis_phase=analysis.phase,
            analysis_risk=analysis.risk,
            prediction_stability=stability,
            candidate_count=len(candidates),
            metadata=metadata,
        )

        if not candidates:
            return InterventionPlan(
                status=PlanStatus.NO_CANDIDATES,
                recommended=None, alternatives=(), rejected=(), vetoed=(),
                summary="No intervention candidates were supplied.", **base
            )

        if self.config.require_prediction and (
            prediction is None
            or prediction.status is not PredictionStatus.AVAILABLE
        ):
            return InterventionPlan(
                status=PlanStatus.INSUFFICIENT_CONTEXT,
                recommended=None, alternatives=(), rejected=(), vetoed=(),
                summary="Planning context is insufficient because an available prediction is required.",
                **base,
            )

        accepted: list[InterventionCandidate] = []
        vetoed: list[RankedIntervention] = []
        for candidate in candidates:
            reasons = _veto(candidate, self.config)
            if reasons:
                vetoed.append(RankedIntervention(
                    rank=None,
                    decision=CandidateDecision.VETOED,
                    candidate=candidate,
                    score=None,
                    veto_reasons=reasons,
                    explanation=(
                        f"Candidate {candidate.candidate_id} was vetoed by the "
                        f"Non-Fonit/safety gate: "
                        f"{', '.join(x.value for x in reasons)}."
                    ),
                ))
            else:
                accepted.append(candidate)

        if not accepted:
            return InterventionPlan(
                status=PlanStatus.ALL_VETOED,
                recommended=None, alternatives=(), rejected=(),
                vetoed=tuple(vetoed),
                summary=f"All {len(vetoed)} intervention candidates were vetoed.",
                **base,
            )

        max_gain = max(max(x.expected_eta_gain, 0.0) for x in accepted)
        max_cost = max(x.cost for x in accepted)
        max_delay = max(x.delay for x in accepted)
        urgency = _urgency(analysis, prediction)
        scored: list[tuple[InterventionCandidate, ScoreBreakdown]] = []

        for candidate in accepted:
            benefit = max(candidate.expected_eta_gain, 0.0) / max_gain if max_gain > _EPS else 0.0
            cost = candidate.cost / max_cost if max_cost > _EPS else 0.0
            delay = candidate.delay / max_delay if max_delay > _EPS else 0.0
            alignment = _alignment(candidate, analysis)
            utility = (
                self.config.benefit_weight * benefit
                + self.config.urgency_weight * urgency
                + self.config.target_weight * alignment
                + self.config.evidence_weight * candidate.evidence_strength
                + self.config.reversibility_weight * candidate.reversibility
                - self.config.cost_weight * cost
                - self.config.risk_weight * candidate.risk
                - self.config.delay_weight * delay
            ) / self.config.weight_sum
            scored.append((candidate, ScoreBreakdown(
                benefit=benefit,
                urgency=urgency,
                target_alignment=alignment,
                evidence=candidate.evidence_strength,
                reversibility=candidate.reversibility,
                cost_penalty=cost,
                risk_penalty=candidate.risk,
                delay_penalty=delay,
                utility=utility,
            )))

        scored.sort(key=lambda x: (
            -x[1].utility,
            -x[0].expected_eta_gain,
            x[0].risk,
            x[0].cost,
            x[0].candidate_id,
        ))

        ranked: list[RankedIntervention] = []
        for rank, (candidate, score) in enumerate(scored, start=1):
            decision = (
                CandidateDecision.RECOMMENDED if rank == 1
                else CandidateDecision.ALTERNATIVE
                if rank <= self.config.max_alternatives + 1
                else CandidateDecision.REJECTED
            )
            ranked.append(RankedIntervention(
                rank=rank,
                decision=decision,
                candidate=candidate,
                score=score,
                explanation=(
                    f"Rank {rank}: {candidate.candidate_id} targets "
                    f"{candidate.target_role.value} '{candidate.target_id}'. "
                    f"Expected eta gain {candidate.expected_eta_gain:.6g}; "
                    f"utility {score.utility:.6g}; evidence "
                    f"{candidate.evidence_strength:.3f}; risk "
                    f"{candidate.risk:.3f}; reversibility "
                    f"{candidate.reversibility:.3f}."
                ),
            ))

        recommended = ranked[0]
        alternatives = tuple(x for x in ranked[1:] if x.decision is CandidateDecision.ALTERNATIVE)
        rejected = tuple(x for x in ranked[1:] if x.decision is CandidateDecision.REJECTED)
        return InterventionPlan(
            status=PlanStatus.AVAILABLE,
            recommended=recommended,
            alternatives=alternatives,
            rejected=rejected,
            vetoed=tuple(vetoed),
            summary=(
                f"Recommended candidate: {recommended.candidate.candidate_id} "
                f"targeting {recommended.candidate.target_id}. "
                f"{len(accepted)} candidates passed the safety gate; "
                f"{len(vetoed)} were vetoed."
            ),
            **base,
        )


def plan_interventions(
    analysis: CascadeAnalysis,
    candidates: Sequence[InterventionCandidate],
    *,
    prediction: CascadePrediction | None = None,
    config: PlannerConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionPlan:
    return CascadePlanner(config).plan(
        analysis, candidates, prediction=prediction, metadata=metadata
    )


__all__ = [
    "CandidateDecision", "CascadePlanner", "InterventionCandidate",
    "InterventionPlan", "PlanStatus", "PlannerConfig", "PlannerError",
    "RankedIntervention", "ScoreBreakdown", "TargetRole", "VetoReason",
    "plan_interventions",
]
