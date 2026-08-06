"""
ROIF Engine
===========

Universal policy layer for the Active Probe Engine.

This module evaluates whether a ProbeDefinition is admissible under a given
system context.

Architectural boundaries
------------------------

Probe Policy:

- does not select the best Probe;
- does not calculate expected information gain;
- does not execute Probes;
- does not mutate graphs;
- does not call Solver;
- does not contain medical or other domain-specific logic.

Probe ranking belongs to probe_planner.py.
Probe execution belongs to active_probe_engine.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .probe_entities import (
    PerturbationType,
    ProbeDefinition,
    ProbeMethod,
    ProbePurpose,
)


# ============================================================================
# Errors
# ============================================================================


class ProbePolicyError(Exception):
    """Base exception for Probe Policy failures."""


class InvalidProbePolicyError(ProbePolicyError):
    """Raised when a Probe policy configuration is invalid."""


# ============================================================================
# Policy taxonomy
# ============================================================================


class ProbePolicyDecision(str, Enum):
    """Final admissibility decision for a Probe."""

    ALLOW = "allow"
    REQUIRE_AUTHORIZATION = "require_authorization"
    REJECT = "reject"


class ProbeRiskLevel(str, Enum):
    """Qualitative risk classification."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ProbeReversibility(str, Enum):
    """Expected reversibility of a Probe perturbation."""

    FULLY_REVERSIBLE = "fully_reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class PolicyReasonCode(str, Enum):
    """Machine-readable policy evaluation reason."""

    ALLOWED = "allowed"

    METHOD_NOT_ALLOWED = "method_not_allowed"
    METHOD_FORBIDDEN = "method_forbidden"

    PERTURBATION_NOT_ALLOWED = "perturbation_not_allowed"
    PERTURBATION_FORBIDDEN = "perturbation_forbidden"

    PURPOSE_NOT_ALLOWED = "purpose_not_allowed"

    COST_EXCEEDED = "cost_exceeded"
    DURATION_EXCEEDED = "duration_exceeded"

    RISK_EXCEEDED = "risk_exceeded"
    CASCADE_RISK_EXCEEDED = "cascade_risk_exceeded"

    UNCERTAINTY_EXCEEDED = "uncertainty_exceeded"

    IRREVERSIBLE_PROBE_REJECTED = "irreversible_probe_rejected"
    UNKNOWN_REVERSIBILITY = "unknown_reversibility"

    RESOURCE_MISSING = "resource_missing"
    REQUIRED_TAG_MISSING = "required_tag_missing"
    FORBIDDEN_TAG_PRESENT = "forbidden_tag_present"

    HUMAN_AUTHORIZATION_REQUIRED = "human_authorization_required"
    HUMAN_AUTHORIZATION_MISSING = "human_authorization_missing"

    NON_FONIT_VETO = "non_fonit_veto"

    CUSTOM_RULE_REJECTED = "custom_rule_rejected"
    CUSTOM_RULE_AUTHORIZATION_REQUIRED = (
        "custom_rule_authorization_required"
    )


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_probability(
    value: float,
    field_name: str,
) -> float:
    """Validate and normalize a probability-like value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    normalized = float(value)

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0.0 and 1.0"
        )

    return normalized


def _validate_nonnegative(
    value: float,
    field_name: str,
) -> float:
    """Validate and normalize a non-negative numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    normalized = float(value)

    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative")

    return normalized


def _normalize_text_set(
    values: Iterable[str],
    field_name: str,
) -> frozenset[str]:
    """Normalize a collection of non-empty strings."""

    if isinstance(values, str):
        raise TypeError(
            f"{field_name} must be an iterable of strings, not a string"
        )

    normalized: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must contain only strings"
            )

        stripped = value.strip()

        if not stripped:
            raise ValueError(
                f"{field_name} must not contain empty strings"
            )

        normalized.add(stripped)

    return frozenset(normalized)


def _freeze_mapping(
    mapping: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return an immutable shallow mapping copy."""

    if not isinstance(mapping, Mapping):
        raise TypeError("metadata must be a mapping")

    return MappingProxyType(dict(mapping))


def _risk_rank(level: ProbeRiskLevel) -> int:
    """Return an ordered numeric rank for risk comparison."""

    ranks = {
        ProbeRiskLevel.NEGLIGIBLE: 0,
        ProbeRiskLevel.LOW: 1,
        ProbeRiskLevel.MODERATE: 2,
        ProbeRiskLevel.HIGH: 3,
        ProbeRiskLevel.CRITICAL: 4,
    }

    return ranks[level]


# ============================================================================
# Probe assessment input
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbeAssessment:
    """
    Estimated execution properties of one Probe.

    These values are supplied by the caller, knowledge layer, instrumentation
    model, or future Probe Planner. ProbePolicy does not estimate them itself.
    """

    expected_cost: float = 0.0
    expected_duration_seconds: float = 0.0

    risk_level: ProbeRiskLevel = ProbeRiskLevel.NEGLIGIBLE
    cascade_risk: float = 0.0
    uncertainty: float = 0.0

    reversibility: ProbeReversibility = (
        ProbeReversibility.FULLY_REVERSIBLE
    )

    required_resources: frozenset[str] = field(
        default_factory=frozenset
    )

    affects_external_systems: bool = False
    large_scale_effect_possible: bool = False
    uncontrolled_propagation_possible: bool = False

    requires_human_authorization: bool = False

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_cost",
            _validate_nonnegative(
                self.expected_cost,
                "expected_cost",
            ),
        )

        object.__setattr__(
            self,
            "expected_duration_seconds",
            _validate_nonnegative(
                self.expected_duration_seconds,
                "expected_duration_seconds",
            ),
        )

        if not isinstance(self.risk_level, ProbeRiskLevel):
            raise TypeError(
                "risk_level must be ProbeRiskLevel"
            )

        object.__setattr__(
            self,
            "cascade_risk",
            _validate_probability(
                self.cascade_risk,
                "cascade_risk",
            ),
        )

        object.__setattr__(
            self,
            "uncertainty",
            _validate_probability(
                self.uncertainty,
                "uncertainty",
            ),
        )

        if not isinstance(
            self.reversibility,
            ProbeReversibility,
        ):
            raise TypeError(
                "reversibility must be ProbeReversibility"
            )

        object.__setattr__(
            self,
            "required_resources",
            _normalize_text_set(
                self.required_resources,
                "required_resources",
            ),
        )

        for field_name in (
            "affects_external_systems",
            "large_scale_effect_possible",
            "uncontrolled_propagation_possible",
            "requires_human_authorization",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


# ============================================================================
# Runtime policy context
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbePolicyContext:
    """
    Runtime constraints under which a Probe is being considered.
    """

    available_resources: frozenset[str] = field(
        default_factory=frozenset
    )

    human_authorized: bool = False

    maximum_cost: float = float("inf")
    maximum_duration_seconds: float = float("inf")

    maximum_risk_level: ProbeRiskLevel = ProbeRiskLevel.LOW
    maximum_cascade_risk: float = 0.25
    maximum_uncertainty: float = 1.0

    allow_irreversible: bool = False
    allow_unknown_reversibility: bool = False

    non_fonit_gate_enabled: bool = True

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "available_resources",
            _normalize_text_set(
                self.available_resources,
                "available_resources",
            ),
        )

        if not isinstance(self.human_authorized, bool):
            raise TypeError("human_authorized must be bool")

        object.__setattr__(
            self,
            "maximum_cost",
            _validate_nonnegative(
                self.maximum_cost,
                "maximum_cost",
            ),
        )

        object.__setattr__(
            self,
            "maximum_duration_seconds",
            _validate_nonnegative(
                self.maximum_duration_seconds,
                "maximum_duration_seconds",
            ),
        )

        if not isinstance(
            self.maximum_risk_level,
            ProbeRiskLevel,
        ):
            raise TypeError(
                "maximum_risk_level must be ProbeRiskLevel"
            )

        object.__setattr__(
            self,
            "maximum_cascade_risk",
            _validate_probability(
                self.maximum_cascade_risk,
                "maximum_cascade_risk",
            ),
        )

        object.__setattr__(
            self,
            "maximum_uncertainty",
            _validate_probability(
                self.maximum_uncertainty,
                "maximum_uncertainty",
            ),
        )

        for field_name in (
            "allow_irreversible",
            "allow_unknown_reversibility",
            "non_fonit_gate_enabled",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


# ============================================================================
# Policy result
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbePolicyReason:
    """One auditable reason produced during policy evaluation."""

    code: PolicyReasonCode
    message: str
    blocking: bool

    def __post_init__(self) -> None:
        if not isinstance(self.code, PolicyReasonCode):
            raise TypeError("code must be PolicyReasonCode")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string")

        if not self.message.strip():
            raise ValueError("message must not be empty")

        if not isinstance(self.blocking, bool):
            raise TypeError("blocking must be bool")


@dataclass(frozen=True, slots=True)
class ProbePolicyResult:
    """Complete auditable policy decision for one Probe."""

    probe_identifier: str
    decision: ProbePolicyDecision
    reasons: tuple[ProbePolicyReason, ...]

    assessment: ProbeAssessment
    context: ProbePolicyContext

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.probe_identifier, str):
            raise TypeError("probe_identifier must be a string")

        if not self.probe_identifier.strip():
            raise ValueError(
                "probe_identifier must not be empty"
            )

        if not isinstance(
            self.decision,
            ProbePolicyDecision,
        ):
            raise TypeError(
                "decision must be ProbePolicyDecision"
            )

        reasons = tuple(self.reasons)

        for reason in reasons:
            if not isinstance(reason, ProbePolicyReason):
                raise TypeError(
                    "reasons must contain ProbePolicyReason instances"
                )

        if not isinstance(self.assessment, ProbeAssessment):
            raise TypeError(
                "assessment must be ProbeAssessment"
            )

        if not isinstance(self.context, ProbePolicyContext):
            raise TypeError(
                "context must be ProbePolicyContext"
            )

        object.__setattr__(self, "reasons", reasons)

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )

    @property
    def allowed(self) -> bool:
        """Return whether execution may proceed immediately."""

        return self.decision is ProbePolicyDecision.ALLOW

    @property
    def rejected(self) -> bool:
        """Return whether the Probe is forbidden."""

        return self.decision is ProbePolicyDecision.REJECT

    @property
    def authorization_required(self) -> bool:
        """Return whether human authorization is required."""

        return (
            self.decision
            is ProbePolicyDecision.REQUIRE_AUTHORIZATION
        )

    @property
    def blocking_reasons(self) -> tuple[ProbePolicyReason, ...]:
        """Return blocking reasons only."""

        return tuple(
            reason
            for reason in self.reasons
            if reason.blocking
        )


# ============================================================================
# Optional custom rules
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbePolicyRuleResult:
    """Result returned by a custom Probe policy rule."""

    decision: ProbePolicyDecision
    reason: ProbePolicyReason

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision,
            ProbePolicyDecision,
        ):
            raise TypeError(
                "decision must be ProbePolicyDecision"
            )

        if not isinstance(self.reason, ProbePolicyReason):
            raise TypeError(
                "reason must be ProbePolicyReason"
            )


class ProbePolicyRule:
    """
    Base interface for custom policy rules.

    Subclasses may add domain-specific restrictions outside the universal
    core without modifying ProbePolicy itself.
    """

    def evaluate(
        self,
        definition: ProbeDefinition,
        assessment: ProbeAssessment,
        context: ProbePolicyContext,
    ) -> ProbePolicyRuleResult | None:
        """
        Evaluate one Probe.

        Return ``None`` when the rule has no opinion.
        """

        raise NotImplementedError


# ============================================================================
# Main policy
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbePolicy:
    """
    Universal admissibility policy for Active Probe Engine.

    Empty ``allowed_*`` sets mean that no positive whitelist is applied.
    Explicit forbidden sets always take precedence.
    """

    allowed_methods: frozenset[ProbeMethod] = field(
        default_factory=frozenset
    )

    forbidden_methods: frozenset[ProbeMethod] = field(
        default_factory=frozenset
    )

    allowed_perturbations: frozenset[PerturbationType] = field(
        default_factory=frozenset
    )

    forbidden_perturbations: frozenset[PerturbationType] = field(
        default_factory=frozenset
    )

    allowed_purposes: frozenset[ProbePurpose] = field(
        default_factory=frozenset
    )

    required_tags: frozenset[str] = field(
        default_factory=frozenset
    )

    forbidden_tags: frozenset[str] = field(
        default_factory=frozenset
    )

    require_authorization_for_methods: frozenset[ProbeMethod] = field(
        default_factory=frozenset
    )

    require_authorization_for_perturbations: frozenset[
        PerturbationType
    ] = field(default_factory=frozenset)

    custom_rules: tuple[ProbePolicyRule, ...] = ()

    def __post_init__(self) -> None:
        for field_name, enum_type in (
            ("allowed_methods", ProbeMethod),
            ("forbidden_methods", ProbeMethod),
            ("allowed_perturbations", PerturbationType),
            ("forbidden_perturbations", PerturbationType),
            ("allowed_purposes", ProbePurpose),
            (
                "require_authorization_for_methods",
                ProbeMethod,
            ),
            (
                "require_authorization_for_perturbations",
                PerturbationType,
            ),
        ):
            values = frozenset(getattr(self, field_name))

            for value in values:
                if not isinstance(value, enum_type):
                    raise TypeError(
                        f"{field_name} must contain only "
                        f"{enum_type.__name__} values"
                    )

            object.__setattr__(
                self,
                field_name,
                values,
            )

        object.__setattr__(
            self,
            "required_tags",
            _normalize_text_set(
                self.required_tags,
                "required_tags",
            ),
        )

        object.__setattr__(
            self,
            "forbidden_tags",
            _normalize_text_set(
                self.forbidden_tags,
                "forbidden_tags",
            ),
        )

        method_overlap = (
            self.allowed_methods
            & self.forbidden_methods
        )

        if method_overlap:
            raise InvalidProbePolicyError(
                "allowed_methods and forbidden_methods overlap: "
                f"{sorted(value.value for value in method_overlap)!r}"
            )

        perturbation_overlap = (
            self.allowed_perturbations
            & self.forbidden_perturbations
        )

        if perturbation_overlap:
            raise InvalidProbePolicyError(
                "allowed_perturbations and forbidden_perturbations "
                "overlap: "
                f"{sorted(value.value for value in perturbation_overlap)!r}"
            )

        tag_overlap = (
            self.required_tags
            & self.forbidden_tags
        )

        if tag_overlap:
            raise InvalidProbePolicyError(
                "required_tags and forbidden_tags overlap: "
                f"{sorted(tag_overlap)!r}"
            )

        rules = tuple(self.custom_rules)

        for rule in rules:
            if not isinstance(rule, ProbePolicyRule):
                raise TypeError(
                    "custom_rules must contain ProbePolicyRule instances"
                )

        object.__setattr__(
            self,
            "custom_rules",
            rules,
        )

    def evaluate(
        self,
        definition: ProbeDefinition,
        assessment: ProbeAssessment | None = None,
        context: ProbePolicyContext | None = None,
    ) -> ProbePolicyResult:
        """Evaluate whether a Probe is admissible."""

        if not isinstance(definition, ProbeDefinition):
            raise TypeError(
                "definition must be ProbeDefinition"
            )

        if assessment is None:
            assessment = ProbeAssessment()
        elif not isinstance(assessment, ProbeAssessment):
            raise TypeError(
                "assessment must be ProbeAssessment or None"
            )

        if context is None:
            context = ProbePolicyContext()
        elif not isinstance(context, ProbePolicyContext):
            raise TypeError(
                "context must be ProbePolicyContext or None"
            )

        reasons: list[ProbePolicyReason] = []
        authorization_required = False

        method = definition.method
        perturbation = definition.perturbation.kind
        tags = frozenset(definition.tags)

        # --------------------------------------------------------------------
        # Method restrictions
        # --------------------------------------------------------------------

        if (
            self.allowed_methods
            and method not in self.allowed_methods
        ):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.METHOD_NOT_ALLOWED,
                    message=(
                        f"Probe method {method.value!r} is not allowed."
                    ),
                    blocking=True,
                )
            )

        if method in self.forbidden_methods:
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.METHOD_FORBIDDEN,
                    message=(
                        f"Probe method {method.value!r} is forbidden."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Perturbation restrictions
        # --------------------------------------------------------------------

        if (
            self.allowed_perturbations
            and perturbation not in self.allowed_perturbations
        ):
            reasons.append(
                ProbePolicyReason(
                    code=(
                        PolicyReasonCode.PERTURBATION_NOT_ALLOWED
                    ),
                    message=(
                        f"Perturbation {perturbation.value!r} "
                        "is not allowed."
                    ),
                    blocking=True,
                )
            )

        if perturbation in self.forbidden_perturbations:
            reasons.append(
                ProbePolicyReason(
                    code=(
                        PolicyReasonCode.PERTURBATION_FORBIDDEN
                    ),
                    message=(
                        f"Perturbation {perturbation.value!r} "
                        "is forbidden."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Purpose restrictions
        # --------------------------------------------------------------------

        if (
            self.allowed_purposes
            and definition.purpose not in self.allowed_purposes
        ):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.PURPOSE_NOT_ALLOWED,
                    message=(
                        f"Probe purpose "
                        f"{definition.purpose.value!r} is not allowed."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Tag restrictions
        # --------------------------------------------------------------------

        missing_tags = self.required_tags - tags

        for tag in sorted(missing_tags):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.REQUIRED_TAG_MISSING,
                    message=f"Required Probe tag is missing: {tag!r}.",
                    blocking=True,
                )
            )

        present_forbidden_tags = (
            self.forbidden_tags & tags
        )

        for tag in sorted(present_forbidden_tags):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.FORBIDDEN_TAG_PRESENT,
                    message=(
                        f"Forbidden Probe tag is present: {tag!r}."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Quantitative constraints
        # --------------------------------------------------------------------

        if assessment.expected_cost > context.maximum_cost:
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.COST_EXCEEDED,
                    message=(
                        f"Expected cost {assessment.expected_cost} "
                        f"exceeds maximum {context.maximum_cost}."
                    ),
                    blocking=True,
                )
            )

        if (
            assessment.expected_duration_seconds
            > context.maximum_duration_seconds
        ):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.DURATION_EXCEEDED,
                    message=(
                        "Expected duration "
                        f"{assessment.expected_duration_seconds} "
                        "seconds exceeds maximum "
                        f"{context.maximum_duration_seconds} seconds."
                    ),
                    blocking=True,
                )
            )

        if (
            _risk_rank(assessment.risk_level)
            > _risk_rank(context.maximum_risk_level)
        ):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.RISK_EXCEEDED,
                    message=(
                        f"Risk level {assessment.risk_level.value!r} "
                        "exceeds allowed level "
                        f"{context.maximum_risk_level.value!r}."
                    ),
                    blocking=True,
                )
            )

        if (
            assessment.cascade_risk
            > context.maximum_cascade_risk
        ):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.CASCADE_RISK_EXCEEDED,
                    message=(
                        f"Cascade risk {assessment.cascade_risk} "
                        "exceeds maximum "
                        f"{context.maximum_cascade_risk}."
                    ),
                    blocking=True,
                )
            )

        if (
            assessment.uncertainty
            > context.maximum_uncertainty
        ):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.UNCERTAINTY_EXCEEDED,
                    message=(
                        f"Uncertainty {assessment.uncertainty} "
                        "exceeds maximum "
                        f"{context.maximum_uncertainty}."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Reversibility
        # --------------------------------------------------------------------

        if (
            assessment.reversibility
            is ProbeReversibility.IRREVERSIBLE
            and not context.allow_irreversible
        ):
            reasons.append(
                ProbePolicyReason(
                    code=(
                        PolicyReasonCode.IRREVERSIBLE_PROBE_REJECTED
                    ),
                    message=(
                        "Irreversible Probe is not allowed "
                        "in the current context."
                    ),
                    blocking=True,
                )
            )

        if (
            assessment.reversibility
            is ProbeReversibility.UNKNOWN
            and not context.allow_unknown_reversibility
        ):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.UNKNOWN_REVERSIBILITY,
                    message=(
                        "Probe reversibility is unknown and is not "
                        "allowed in the current context."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Resources
        # --------------------------------------------------------------------

        missing_resources = (
            assessment.required_resources
            - context.available_resources
        )

        for resource in sorted(missing_resources):
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.RESOURCE_MISSING,
                    message=(
                        f"Required resource is unavailable: "
                        f"{resource!r}."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Non-Fonit Gate
        # --------------------------------------------------------------------

        non_fonit_veto = (
            context.non_fonit_gate_enabled
            and (
                assessment.affects_external_systems
                or assessment.large_scale_effect_possible
                or assessment.uncontrolled_propagation_possible
            )
        )

        if non_fonit_veto:
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.NON_FONIT_VETO,
                    message=(
                        "Probe rejected by Non-Fonit Gate because "
                        "external, large-scale, or uncontrolled cascade "
                        "effects are possible."
                    ),
                    blocking=True,
                )
            )

        # --------------------------------------------------------------------
        # Authorization
        # --------------------------------------------------------------------

        if (
            assessment.requires_human_authorization
            or method in self.require_authorization_for_methods
            or perturbation
            in self.require_authorization_for_perturbations
        ):
            authorization_required = True

        # --------------------------------------------------------------------
        # Custom rules
        # --------------------------------------------------------------------

        for rule in self.custom_rules:
            rule_result = rule.evaluate(
                definition,
                assessment,
                context,
            )

            if rule_result is None:
                continue

            if not isinstance(
                rule_result,
                ProbePolicyRuleResult,
            ):
                raise ProbePolicyError(
                    "custom rule must return "
                    "ProbePolicyRuleResult or None"
                )

            reasons.append(rule_result.reason)

            if (
                rule_result.decision
                is ProbePolicyDecision.REQUIRE_AUTHORIZATION
            ):
                authorization_required = True

        # --------------------------------------------------------------------
        # Final decision
        # --------------------------------------------------------------------

        if any(reason.blocking for reason in reasons):
            decision = ProbePolicyDecision.REJECT

        elif authorization_required and not context.human_authorized:
            reasons.append(
                ProbePolicyReason(
                    code=(
                        PolicyReasonCode.HUMAN_AUTHORIZATION_REQUIRED
                    ),
                    message=(
                        "Human authorization is required before "
                        "Probe execution."
                    ),
                    blocking=False,
                )
            )

            decision = (
                ProbePolicyDecision.REQUIRE_AUTHORIZATION
            )

        else:
            reasons.append(
                ProbePolicyReason(
                    code=PolicyReasonCode.ALLOWED,
                    message="Probe is allowed by the current policy.",
                    blocking=False,
                )
            )

            decision = ProbePolicyDecision.ALLOW

        return ProbePolicyResult(
            probe_identifier=definition.identifier,
            decision=decision,
            reasons=tuple(reasons),
            assessment=assessment,
            context=context,
        )


__all__ = [
    "InvalidProbePolicyError",
    "PolicyReasonCode",
    "ProbeAssessment",
    "ProbePolicy",
    "ProbePolicyContext",
    "ProbePolicyDecision",
    "ProbePolicyError",
    "ProbePolicyReason",
    "ProbePolicyResult",
    "ProbePolicyRule",
    "ProbePolicyRuleResult",
    "ProbeReversibility",
    "ProbeRiskLevel",
]
