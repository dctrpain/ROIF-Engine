"""
ROIF Engine
===========

Universal planner for the Active Probe Engine.

The planner ranks admissible Probe definitions and selects Probe* — the next
controlled experiment expected to provide the greatest useful information
under current policy constraints.

Architectural boundaries
------------------------

ProbePlanner:

- does not execute Probes;
- does not mutate graphs;
- does not calculate D_origin, D_fast, D_root, or Node*;
- does not call Solver;
- does not infer domain-specific measurements;
- does not estimate Probe effects internally.

Expected information and operational estimates are supplied by the caller,
knowledge layer, simulation layer, or a future graph-uncertainty model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .probe_entities import ProbeDefinition
from .probe_policy import (
    ProbeAssessment,
    ProbePolicy,
    ProbePolicyContext,
    ProbePolicyDecision,
    ProbePolicyResult,
)
from .probe_registry import (
    ProbeQuery,
    ProbeRegistry,
    ProbeRegistrySnapshot,
)


# ============================================================================
# Errors
# ============================================================================


class ProbePlannerError(Exception):
    """Base exception for Probe Planner failures."""


class ProbeEstimateNotFoundError(ProbePlannerError):
    """Raised when a Probe has no required planning estimate."""


class NoAdmissibleProbeError(ProbePlannerError):
    """Raised when no Probe can be selected under current constraints."""


# ============================================================================
# Planner taxonomy
# ============================================================================


class ProbeCandidateStatus(str, Enum):
    """Planning status assigned to one Probe candidate."""

    ADMISSIBLE = "admissible"
    AUTHORIZATION_REQUIRED = "authorization_required"
    REJECTED = "rejected"
    MISSING_ESTIMATE = "missing_estimate"


class ProbeSelectionStatus(str, Enum):
    """Overall outcome of a planning operation."""

    SELECTED = "selected"
    NO_CANDIDATES = "no_candidates"
    NO_ADMISSIBLE_PROBES = "no_admissible_probes"
    AUTHORIZATION_REQUIRED = "authorization_required"


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_probability(
    value: float,
    field_name: str,
) -> float:
    """Validate and normalize a value constrained to [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0.0 and 1.0"
        )

    return normalized


def _validate_nonnegative(
    value: float,
    field_name: str,
) -> float:
    """Validate and normalize a finite non-negative value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")

    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative")

    return normalized


def _validate_finite(
    value: float,
    field_name: str,
) -> float:
    """Validate and normalize a finite numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")

    return normalized


def _freeze_mapping(
    mapping: Mapping[str, Any],
    field_name: str = "metadata",
) -> Mapping[str, Any]:
    """Return an immutable shallow mapping copy."""

    if not isinstance(mapping, Mapping):
        raise TypeError(f"{field_name} must be a mapping")

    return MappingProxyType(dict(mapping))


def _normalize_identifier(identifier: str) -> str:
    """Validate and normalize a Probe identifier."""

    if not isinstance(identifier, str):
        raise TypeError("probe identifier must be a string")

    normalized = identifier.strip()

    if not normalized:
        raise ValueError("probe identifier must not be empty")

    return normalized


# ============================================================================
# Information estimate
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbeInformationEstimate:
    """
    Expected informational and operational value of one Probe.

    All normalized components belong to [0, 1].

    information_gain:
        Expected total information gained about the system.

    uncertainty_reduction:
        Expected reduction of graph or model uncertainty.

    hypothesis_discrimination:
        Ability to distinguish competing causal hypotheses.

    graph_coverage:
        Fraction of currently uncertain graph structure informed by the Probe.

    novelty:
        Amount of non-redundant evidence expected from the Probe.

    feasibility:
        Practical probability that the Probe can be completed as intended.

    confidence:
        Confidence in this estimate itself.

    redundancy:
        Expected overlap with information already available.

    disruption:
        Expected non-hazardous disturbance introduced into the system.
        Safety vetoes remain the responsibility of ProbePolicy.
    """

    probe_identifier: str

    information_gain: float = 0.0
    uncertainty_reduction: float = 0.0
    hypothesis_discrimination: float = 0.0
    graph_coverage: float = 0.0
    novelty: float = 0.0

    feasibility: float = 1.0
    confidence: float = 1.0

    redundancy: float = 0.0
    disruption: float = 0.0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probe_identifier",
            _normalize_identifier(self.probe_identifier),
        )

        for field_name in (
            "information_gain",
            "uncertainty_reduction",
            "hypothesis_discrimination",
            "graph_coverage",
            "novelty",
            "feasibility",
            "confidence",
            "redundancy",
            "disruption",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_probability(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


# ============================================================================
# Planner weights and configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbePlannerWeights:
    """Weights used to calculate Probe planning utility."""

    information_gain: float = 1.0
    uncertainty_reduction: float = 1.0
    hypothesis_discrimination: float = 0.80
    graph_coverage: float = 0.60
    novelty: float = 0.50

    expected_cost: float = 0.35
    expected_duration: float = 0.20
    uncertainty: float = 0.40
    cascade_risk: float = 1.0

    redundancy: float = 0.50
    disruption: float = 0.40

    def __post_init__(self) -> None:
        for field_name in (
            "information_gain",
            "uncertainty_reduction",
            "hypothesis_discrimination",
            "graph_coverage",
            "novelty",
            "expected_cost",
            "expected_duration",
            "uncertainty",
            "cascade_risk",
            "redundancy",
            "disruption",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_nonnegative(
                    getattr(self, field_name),
                    field_name,
                ),
            )


@dataclass(frozen=True, slots=True)
class ProbePlannerConfig:
    """Configuration controlling candidate planning behavior."""

    weights: ProbePlannerWeights = field(
        default_factory=ProbePlannerWeights
    )

    include_authorization_required: bool = True
    allow_missing_estimates: bool = False
    raise_when_no_admissible_probe: bool = False

    maximum_results: int | None = None

    cost_normalization: float = 1.0
    duration_normalization_seconds: float = 60.0

    minimum_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.weights, ProbePlannerWeights):
            raise TypeError(
                "weights must be ProbePlannerWeights"
            )

        for field_name in (
            "include_authorization_required",
            "allow_missing_estimates",
            "raise_when_no_admissible_probe",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        if self.maximum_results is not None:
            if (
                isinstance(self.maximum_results, bool)
                or not isinstance(self.maximum_results, int)
            ):
                raise TypeError(
                    "maximum_results must be int or None"
                )

            if self.maximum_results < 1:
                raise ValueError(
                    "maximum_results must be at least 1"
                )

        object.__setattr__(
            self,
            "cost_normalization",
            _validate_nonnegative(
                self.cost_normalization,
                "cost_normalization",
            ),
        )

        object.__setattr__(
            self,
            "duration_normalization_seconds",
            _validate_nonnegative(
                self.duration_normalization_seconds,
                "duration_normalization_seconds",
            ),
        )

        if self.cost_normalization == 0.0:
            raise ValueError(
                "cost_normalization must be greater than zero"
            )

        if self.duration_normalization_seconds == 0.0:
            raise ValueError(
                "duration_normalization_seconds must be greater than zero"
            )

        if self.minimum_score is not None:
            object.__setattr__(
                self,
                "minimum_score",
                _validate_finite(
                    self.minimum_score,
                    "minimum_score",
                ),
            )


# ============================================================================
# Score components
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbeScoreComponents:
    """Auditable components of one Probe planning score."""

    information_value: float
    feasibility_factor: float
    confidence_factor: float

    cost_penalty: float
    duration_penalty: float
    uncertainty_penalty: float
    cascade_risk_penalty: float
    redundancy_penalty: float
    disruption_penalty: float

    gross_score: float
    total_penalty: float
    final_score: float

    def __post_init__(self) -> None:
        for field_name in (
            "information_value",
            "feasibility_factor",
            "confidence_factor",
            "cost_penalty",
            "duration_penalty",
            "uncertainty_penalty",
            "cascade_risk_penalty",
            "redundancy_penalty",
            "disruption_penalty",
            "gross_score",
            "total_penalty",
            "final_score",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_finite(
                    getattr(self, field_name),
                    field_name,
                ),
            )


# ============================================================================
# Candidate and result
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbePlanCandidate:
    """Complete planning evaluation of one ProbeDefinition."""

    definition: ProbeDefinition
    status: ProbeCandidateStatus

    policy_result: ProbePolicyResult
    estimate: ProbeInformationEstimate | None

    assessment: ProbeAssessment

    score: float | None = None
    components: ProbeScoreComponents | None = None

    rank: int | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ProbeDefinition):
            raise TypeError(
                "definition must be ProbeDefinition"
            )

        if not isinstance(self.status, ProbeCandidateStatus):
            raise TypeError(
                "status must be ProbeCandidateStatus"
            )

        if not isinstance(self.policy_result, ProbePolicyResult):
            raise TypeError(
                "policy_result must be ProbePolicyResult"
            )

        if (
            self.estimate is not None
            and not isinstance(
                self.estimate,
                ProbeInformationEstimate,
            )
        ):
            raise TypeError(
                "estimate must be ProbeInformationEstimate or None"
            )

        if not isinstance(self.assessment, ProbeAssessment):
            raise TypeError(
                "assessment must be ProbeAssessment"
            )

        if self.score is not None:
            object.__setattr__(
                self,
                "score",
                _validate_finite(self.score, "score"),
            )

        if (
            self.components is not None
            and not isinstance(
                self.components,
                ProbeScoreComponents,
            )
        ):
            raise TypeError(
                "components must be ProbeScoreComponents or None"
            )

        if self.rank is not None:
            if (
                isinstance(self.rank, bool)
                or not isinstance(self.rank, int)
            ):
                raise TypeError("rank must be int or None")

            if self.rank < 1:
                raise ValueError("rank must be at least 1")

        if (
            self.status
            in {
                ProbeCandidateStatus.ADMISSIBLE,
                ProbeCandidateStatus.AUTHORIZATION_REQUIRED,
            }
            and self.estimate is not None
            and self.score is None
        ):
            raise ValueError(
                "scored candidate must contain score"
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )

    @property
    def selectable(self) -> bool:
        """Return whether the candidate may participate in selection."""

        return self.status in {
            ProbeCandidateStatus.ADMISSIBLE,
            ProbeCandidateStatus.AUTHORIZATION_REQUIRED,
        }


@dataclass(frozen=True, slots=True)
class ProbePlanResult:
    """Complete deterministic output of one planning operation."""

    status: ProbeSelectionStatus
    selected: ProbePlanCandidate | None

    ranked_candidates: tuple[ProbePlanCandidate, ...]
    rejected_candidates: tuple[ProbePlanCandidate, ...]
    missing_estimate_candidates: tuple[ProbePlanCandidate, ...]

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProbeSelectionStatus):
            raise TypeError(
                "status must be ProbeSelectionStatus"
            )

        for field_name in (
            "ranked_candidates",
            "rejected_candidates",
            "missing_estimate_candidates",
        ):
            candidates = tuple(getattr(self, field_name))

            for candidate in candidates:
                if not isinstance(candidate, ProbePlanCandidate):
                    raise TypeError(
                        f"{field_name} must contain "
                        "ProbePlanCandidate instances"
                    )

            object.__setattr__(
                self,
                field_name,
                candidates,
            )

        if (
            self.selected is not None
            and not isinstance(
                self.selected,
                ProbePlanCandidate,
            )
        ):
            raise TypeError(
                "selected must be ProbePlanCandidate or None"
            )

        if (
            self.status is ProbeSelectionStatus.SELECTED
            and self.selected is None
        ):
            raise ValueError(
                "selected candidate is required for SELECTED status"
            )

        if (
            self.status is not ProbeSelectionStatus.SELECTED
            and self.selected is not None
        ):
            raise ValueError(
                "selected must be None unless status is SELECTED"
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )

    @property
    def probe_star(self) -> ProbeDefinition | None:
        """Return the selected ProbeDefinition, if any."""

        if self.selected is None:
            return None

        return self.selected.definition


# ============================================================================
# Score calculation
# ============================================================================


def calculate_probe_score(
    estimate: ProbeInformationEstimate,
    assessment: ProbeAssessment,
    config: ProbePlannerConfig,
) -> ProbeScoreComponents:
    """Calculate an auditable planning utility for one Probe."""

    if not isinstance(estimate, ProbeInformationEstimate):
        raise TypeError(
            "estimate must be ProbeInformationEstimate"
        )

    if not isinstance(assessment, ProbeAssessment):
        raise TypeError(
            "assessment must be ProbeAssessment"
        )

    if not isinstance(config, ProbePlannerConfig):
        raise TypeError(
            "config must be ProbePlannerConfig"
        )

    weights = config.weights

    information_value = (
        weights.information_gain
        * estimate.information_gain
        + weights.uncertainty_reduction
        * estimate.uncertainty_reduction
        + weights.hypothesis_discrimination
        * estimate.hypothesis_discrimination
        + weights.graph_coverage
        * estimate.graph_coverage
        + weights.novelty
        * estimate.novelty
    )

    feasibility_factor = estimate.feasibility
    confidence_factor = estimate.confidence

    gross_score = (
        information_value
        * feasibility_factor
        * confidence_factor
    )

    normalized_cost = min(
        1.0,
        assessment.expected_cost
        / config.cost_normalization,
    )

    normalized_duration = min(
        1.0,
        assessment.expected_duration_seconds
        / config.duration_normalization_seconds,
    )

    cost_penalty = (
        weights.expected_cost
        * normalized_cost
    )

    duration_penalty = (
        weights.expected_duration
        * normalized_duration
    )

    uncertainty_penalty = (
        weights.uncertainty
        * assessment.uncertainty
    )

    cascade_risk_penalty = (
        weights.cascade_risk
        * assessment.cascade_risk
    )

    redundancy_penalty = (
        weights.redundancy
        * estimate.redundancy
    )

    disruption_penalty = (
        weights.disruption
        * estimate.disruption
    )

    total_penalty = (
        cost_penalty
        + duration_penalty
        + uncertainty_penalty
        + cascade_risk_penalty
        + redundancy_penalty
        + disruption_penalty
    )

    final_score = gross_score - total_penalty

    return ProbeScoreComponents(
        information_value=information_value,
        feasibility_factor=feasibility_factor,
        confidence_factor=confidence_factor,
        cost_penalty=cost_penalty,
        duration_penalty=duration_penalty,
        uncertainty_penalty=uncertainty_penalty,
        cascade_risk_penalty=cascade_risk_penalty,
        redundancy_penalty=redundancy_penalty,
        disruption_penalty=disruption_penalty,
        gross_score=gross_score,
        total_penalty=total_penalty,
        final_score=final_score,
    )


# ============================================================================
# Planner
# ============================================================================


class ProbePlanner:
    """
    Deterministic planner selecting Probe* from registered definitions.

    Tie-breaking order:

    1. higher final score;
    2. immediate admissibility before authorization requirement;
    3. higher estimate confidence;
    4. lexicographically smaller Probe identifier.

    This makes planning reproducible across runs.
    """

    __slots__ = (
        "_registry",
        "_policy",
        "_config",
    )

    def __init__(
        self,
        registry: ProbeRegistry | ProbeRegistrySnapshot,
        policy: ProbePolicy | None = None,
        config: ProbePlannerConfig | None = None,
    ) -> None:
        if not isinstance(
            registry,
            (ProbeRegistry, ProbeRegistrySnapshot),
        ):
            raise TypeError(
                "registry must be ProbeRegistry "
                "or ProbeRegistrySnapshot"
            )

        if policy is None:
            policy = ProbePolicy()
        elif not isinstance(policy, ProbePolicy):
            raise TypeError(
                "policy must be ProbePolicy or None"
            )

        if config is None:
            config = ProbePlannerConfig()
        elif not isinstance(config, ProbePlannerConfig):
            raise TypeError(
                "config must be ProbePlannerConfig or None"
            )

        self._registry = registry
        self._policy = policy
        self._config = config

    @property
    def registry(
        self,
    ) -> ProbeRegistry | ProbeRegistrySnapshot:
        return self._registry

    @property
    def policy(self) -> ProbePolicy:
        return self._policy

    @property
    def config(self) -> ProbePlannerConfig:
        return self._config

    def plan(
        self,
        estimates: (
            Mapping[str, ProbeInformationEstimate]
            | Iterable[ProbeInformationEstimate]
        ),
        *,
        assessments: Mapping[str, ProbeAssessment] | None = None,
        context: ProbePolicyContext | None = None,
        query: ProbeQuery | None = None,
    ) -> ProbePlanResult:
        """
        Rank admissible Probe candidates and select Probe*.

        Parameters
        ----------
        estimates:
            Expected information estimates keyed by Probe identifier, or an
            iterable of ProbeInformationEstimate objects.

        assessments:
            Optional operational and safety estimates keyed by identifier.
            Missing entries use ProbeAssessment defaults.

        context:
            Runtime policy constraints.

        query:
            Optional registry filter applied before policy evaluation.
        """

        estimate_map = self._normalize_estimates(estimates)
        assessment_map = self._normalize_assessments(assessments)

        if context is None:
            context = ProbePolicyContext()
        elif not isinstance(context, ProbePolicyContext):
            raise TypeError(
                "context must be ProbePolicyContext or None"
            )

        if query is not None and not isinstance(query, ProbeQuery):
            raise TypeError(
                "query must be ProbeQuery or None"
            )

        definitions = self._definitions(query)

        if not definitions:
            return ProbePlanResult(
                status=ProbeSelectionStatus.NO_CANDIDATES,
                selected=None,
                ranked_candidates=(),
                rejected_candidates=(),
                missing_estimate_candidates=(),
            )

        selectable: list[ProbePlanCandidate] = []
        rejected: list[ProbePlanCandidate] = []
        missing: list[ProbePlanCandidate] = []

        for definition in definitions:
            identifier = definition.identifier

            assessment = assessment_map.get(
                identifier,
                ProbeAssessment(),
            )

            policy_result = self._policy.evaluate(
                definition,
                assessment,
                context,
            )

            estimate = estimate_map.get(identifier)

            if policy_result.decision is ProbePolicyDecision.REJECT:
                rejected.append(
                    ProbePlanCandidate(
                        definition=definition,
                        status=ProbeCandidateStatus.REJECTED,
                        policy_result=policy_result,
                        estimate=estimate,
                        assessment=assessment,
                    )
                )
                continue

            if estimate is None:
                missing_candidate = ProbePlanCandidate(
                    definition=definition,
                    status=ProbeCandidateStatus.MISSING_ESTIMATE,
                    policy_result=policy_result,
                    estimate=None,
                    assessment=assessment,
                )

                missing.append(missing_candidate)

                if not self._config.allow_missing_estimates:
                    continue

                estimate = ProbeInformationEstimate(
                    probe_identifier=identifier,
                    confidence=0.0,
                )

            if estimate.probe_identifier != identifier:
                raise ProbePlannerError(
                    "estimate identifier does not match "
                    f"ProbeDefinition: {identifier!r}"
                )

            components = calculate_probe_score(
                estimate,
                assessment,
                self._config,
            )

            if (
                self._config.minimum_score is not None
                and components.final_score
                < self._config.minimum_score
            ):
                rejected.append(
                    ProbePlanCandidate(
                        definition=definition,
                        status=ProbeCandidateStatus.REJECTED,
                        policy_result=policy_result,
                        estimate=estimate,
                        assessment=assessment,
                        score=components.final_score,
                        components=components,
                        metadata={
                            "rejection": "minimum_score",
                            "minimum_score": (
                                self._config.minimum_score
                            ),
                        },
                    )
                )
                continue

            if (
                policy_result.decision
                is ProbePolicyDecision.REQUIRE_AUTHORIZATION
            ):
                status = (
                    ProbeCandidateStatus.AUTHORIZATION_REQUIRED
                )
            else:
                status = ProbeCandidateStatus.ADMISSIBLE

            if (
                status
                is ProbeCandidateStatus.AUTHORIZATION_REQUIRED
                and not self._config.include_authorization_required
            ):
                rejected.append(
                    ProbePlanCandidate(
                        definition=definition,
                        status=status,
                        policy_result=policy_result,
                        estimate=estimate,
                        assessment=assessment,
                        score=components.final_score,
                        components=components,
                        metadata={
                            "excluded": (
                                "authorization_required"
                            ),
                        },
                    )
                )
                continue

            selectable.append(
                ProbePlanCandidate(
                    definition=definition,
                    status=status,
                    policy_result=policy_result,
                    estimate=estimate,
                    assessment=assessment,
                    score=components.final_score,
                    components=components,
                )
            )

        ranked = self._rank(selectable)

        if self._config.maximum_results is not None:
            ranked = ranked[: self._config.maximum_results]

        if ranked:
            best = ranked[0]

            if (
                best.status
                is ProbeCandidateStatus.AUTHORIZATION_REQUIRED
            ):
                status = (
                    ProbeSelectionStatus.AUTHORIZATION_REQUIRED
                )
                selected = None
            else:
                status = ProbeSelectionStatus.SELECTED
                selected = best

            return ProbePlanResult(
                status=status,
                selected=selected,
                ranked_candidates=ranked,
                rejected_candidates=tuple(rejected),
                missing_estimate_candidates=tuple(missing),
            )

        if self._config.raise_when_no_admissible_probe:
            raise NoAdmissibleProbeError(
                "no admissible Probe is available"
            )

        return ProbePlanResult(
            status=ProbeSelectionStatus.NO_ADMISSIBLE_PROBES,
            selected=None,
            ranked_candidates=(),
            rejected_candidates=tuple(rejected),
            missing_estimate_candidates=tuple(missing),
        )

    def _definitions(
        self,
        query: ProbeQuery | None,
    ) -> tuple[ProbeDefinition, ...]:
        """Return candidate definitions from registry or snapshot."""

        if isinstance(self._registry, ProbeRegistry):
            return self._registry.search(query)

        definitions = self._registry.definitions

        if query is None:
            return definitions

        return tuple(
            definition
            for definition in definitions
            if ProbeRegistry._matches(definition, query)
        )

    @staticmethod
    def _normalize_estimates(
        estimates: (
            Mapping[str, ProbeInformationEstimate]
            | Iterable[ProbeInformationEstimate]
        ),
    ) -> dict[str, ProbeInformationEstimate]:
        """Normalize planning estimates into an identifier mapping."""

        if isinstance(estimates, Mapping):
            normalized: dict[str, ProbeInformationEstimate] = {}

            for identifier, estimate in estimates.items():
                normalized_identifier = _normalize_identifier(
                    identifier
                )

                if not isinstance(
                    estimate,
                    ProbeInformationEstimate,
                ):
                    raise TypeError(
                        "estimate mapping values must be "
                        "ProbeInformationEstimate"
                    )

                if (
                    estimate.probe_identifier
                    != normalized_identifier
                ):
                    raise ProbePlannerError(
                        "estimate mapping key does not match "
                        "estimate.probe_identifier"
                    )

                normalized[normalized_identifier] = estimate

            return normalized

        if isinstance(estimates, (str, bytes)):
            raise TypeError(
                "estimates must be a mapping or iterable "
                "of ProbeInformationEstimate"
            )

        normalized = {}

        for estimate in estimates:
            if not isinstance(
                estimate,
                ProbeInformationEstimate,
            ):
                raise TypeError(
                    "estimates must contain only "
                    "ProbeInformationEstimate instances"
                )

            identifier = estimate.probe_identifier

            if identifier in normalized:
                raise ProbePlannerError(
                    f"duplicate estimate identifier: {identifier!r}"
                )

            normalized[identifier] = estimate

        return normalized

    @staticmethod
    def _normalize_assessments(
        assessments: Mapping[str, ProbeAssessment] | None,
    ) -> dict[str, ProbeAssessment]:
        """Normalize optional assessments into a plain dictionary."""

        if assessments is None:
            return {}

        if not isinstance(assessments, Mapping):
            raise TypeError(
                "assessments must be a mapping or None"
            )

        normalized: dict[str, ProbeAssessment] = {}

        for identifier, assessment in assessments.items():
            normalized_identifier = _normalize_identifier(
                identifier
            )

            if not isinstance(assessment, ProbeAssessment):
                raise TypeError(
                    "assessment mapping values must be "
                    "ProbeAssessment"
                )

            normalized[normalized_identifier] = assessment

        return normalized

    @staticmethod
    def _rank(
        candidates: Sequence[ProbePlanCandidate],
    ) -> tuple[ProbePlanCandidate, ...]:
        """Return candidates sorted deterministically with rank values."""

        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.score),
                (
                    0
                    if candidate.status
                    is ProbeCandidateStatus.ADMISSIBLE
                    else 1
                ),
                -(
                    candidate.estimate.confidence
                    if candidate.estimate is not None
                    else 0.0
                ),
                candidate.definition.identifier,
            ),
        )

        return tuple(
            ProbePlanCandidate(
                definition=candidate.definition,
                status=candidate.status,
                policy_result=candidate.policy_result,
                estimate=candidate.estimate,
                assessment=candidate.assessment,
                score=candidate.score,
                components=candidate.components,
                rank=index,
                metadata=candidate.metadata,
            )
            for index, candidate in enumerate(
                ordered,
                start=1,
            )
        )


__all__ = [
    "NoAdmissibleProbeError",
    "ProbeCandidateStatus",
    "ProbeEstimateNotFoundError",
    "ProbeInformationEstimate",
    "ProbePlanCandidate",
    "ProbePlanner",
    "ProbePlannerConfig",
    "ProbePlannerError",
    "ProbePlannerWeights",
    "ProbePlanResult",
    "ProbeScoreComponents",
    "ProbeSelectionStatus",
    "calculate_probe_score",
]
