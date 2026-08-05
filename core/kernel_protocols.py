"""
ROIF Engine
Kernel Protocols

This module defines the immutable safety and governance protocols of the
ROIF Kernel.

Kernel principles describe the fundamental laws of the engine.
Kernel modes define which classes of operations are permitted.
Kernel protocols evaluate whether a permitted operation may proceed in
a particular context.

The protocols in this module must not:

- prevent observation, recalculation, verification, audit, comparison,
  falsification, rollback, or other epistemically protected operations;
- treat contradiction of the current model as damage;
- protect a hypothesis merely because it is already accepted;
- interpret computational disagreement as harm;
- permit uncontrolled large-scale forcing of natural, biological,
  social, ecological, or infrastructure systems.

Author:
    Architect (Dctr Pain)

License:
    See project license.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Mapping

from core.kernel_modes import (
    KernelMode,
    KernelOperation,
    OperationEffect,
    epistemically_protected_operations,
    mode_policy,
    operation_definition,
)


class KernelProtocolError(ValueError):
    """
    Raised when protocol data or protocol evaluation is invalid.
    """


class KernelProtocol(str, Enum):
    """
    Safety and governance protocols of the ROIF Kernel.
    """

    DO_NO_HARM = "do_no_harm"
    NON_FONIT = "non_fonit"
    EPISTEMIC_FREEDOM = "epistemic_freedom"
    REVERSIBILITY_FIRST = "reversibility_first"
    HUMAN_OVERSIGHT = "human_oversight"
    PROVENANCE_REQUIRED = "provenance_required"
    UNCERTAINTY_DISCLOSURE = "uncertainty_disclosure"
    COUNTERFACTUAL_REQUIRED = "counterfactual_required"
    GLOBAL_INTEGRITY = "global_integrity"


class ProtocolDecision(str, Enum):
    """
    Result of a protocol evaluation.
    """

    ALLOW = "allow"
    ALLOW_WITH_MONITORING = "allow_with_monitoring"
    REQUIRE_REVIEW = "require_review"
    REJECT = "reject"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class RiskLevel(str, Enum):
    """
    Qualitative risk level used by kernel protocols.
    """

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionScope(str, Enum):
    """
    Structural scope of an operation or intervention.
    """

    NONE = "none"
    LOCAL = "local"
    SUBSYSTEM = "subsystem"
    SYSTEM = "system"
    MULTI_SYSTEM = "multi_system"
    ENVIRONMENTAL = "environmental"
    SOCIETAL = "societal"
    INFRASTRUCTURE = "infrastructure"


class ReversibilityLevel(str, Enum):
    """
    Expected reversibility of an operation.
    """

    FULLY_REVERSIBLE = "fully_reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    DIFFICULT_TO_REVERSE = "difficult_to_reverse"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    """
    Evidence state supporting the proposed operation.
    """

    OBSERVED = "observed"
    DERIVED = "derived"
    HYPOTHESIZED = "hypothesized"
    SIMULATED = "simulated"
    VALIDATED = "validated"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class HumanOversightLevel(str, Enum):
    """
    Required human oversight level.
    """

    NONE = "none"
    INFORM = "inform"
    REVIEW = "review"
    APPROVE = "approve"
    SUPERVISE = "supervise"


@dataclass(frozen=True, slots=True)
class ProtocolDefinition:
    """
    Immutable definition of one kernel protocol.
    """

    protocol: KernelProtocol
    title: str
    description: str
    rationale: str
    immutable: bool = True


@dataclass(frozen=True, slots=True)
class ProtocolThresholds:
    """
    Numerical thresholds used by safety evaluation.

    All normalized values are expected within [0, 1].
    """

    maximum_acceptable_harm: float = 0.20
    maximum_acceptable_cascade_risk: float = 0.20
    maximum_acceptable_uncertainty: float = 0.40
    minimum_expected_integrity_gain: float = 0.0
    minimum_confidence_for_live_action: float = 0.75
    critical_scope_risk_threshold: float = 0.10
    irreversible_action_risk_threshold: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "maximum_acceptable_harm",
            "maximum_acceptable_cascade_risk",
            "maximum_acceptable_uncertainty",
            "minimum_expected_integrity_gain",
            "minimum_confidence_for_live_action",
            "critical_scope_risk_threshold",
            "irreversible_action_risk_threshold",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise KernelProtocolError(
                    f"{name} must be a finite real number."
                )

            if not 0.0 <= float(value) <= 1.0:
                raise KernelProtocolError(
                    f"{name} must be within [0, 1]."
                )


@dataclass(frozen=True, slots=True)
class ProtocolContext:
    """
    Immutable context supplied to kernel protocol evaluation.

    The context represents a proposed operation, not an accepted fact.
    """

    mode: KernelMode
    operation: KernelOperation

    scope: InterventionScope = InterventionScope.NONE
    reversibility: ReversibilityLevel = (
        ReversibilityLevel.FULLY_REVERSIBLE
    )
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN

    expected_benefit: float = 0.0
    expected_harm: float = 0.0
    cascade_risk: float = 0.0
    uncertainty: float = 0.0
    expected_integrity_change: float = 0.0
    confidence: float = 0.0

    has_counterfactual_analysis: bool = False
    has_provenance: bool = False
    has_rollback_path: bool = False
    human_review_available: bool = False
    human_approval_granted: bool = False
    monitoring_available: bool = False

    affects_external_system: bool = False
    affects_kernel_memory: bool = False
    affects_kernel_code: bool = False

    description: str = ""
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mode, KernelMode):
            raise KernelProtocolError(
                "mode must be a KernelMode."
            )

        if not isinstance(self.operation, KernelOperation):
            raise KernelProtocolError(
                "operation must be a KernelOperation."
            )

        if not isinstance(self.scope, InterventionScope):
            raise KernelProtocolError(
                "scope must be an InterventionScope."
            )

        if not isinstance(
            self.reversibility,
            ReversibilityLevel,
        ):
            raise KernelProtocolError(
                "reversibility must be a ReversibilityLevel."
            )

        if not isinstance(self.evidence_status, EvidenceStatus):
            raise KernelProtocolError(
                "evidence_status must be an EvidenceStatus."
            )

        for name in (
            "expected_benefit",
            "expected_harm",
            "cascade_risk",
            "uncertainty",
            "confidence",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise KernelProtocolError(
                    f"{name} must be a finite real number."
                )

            if not 0.0 <= float(value) <= 1.0:
                raise KernelProtocolError(
                    f"{name} must be within [0, 1]."
                )

        if (
            isinstance(self.expected_integrity_change, bool)
            or not isinstance(
                self.expected_integrity_change,
                (int, float),
            )
            or not math.isfinite(
                float(self.expected_integrity_change)
            )
        ):
            raise KernelProtocolError(
                "expected_integrity_change must be finite."
            )

        if not -1.0 <= float(
            self.expected_integrity_change
        ) <= 1.0:
            raise KernelProtocolError(
                "expected_integrity_change must be within "
                "[-1, 1]."
            )

        for name in (
            "has_counterfactual_analysis",
            "has_provenance",
            "has_rollback_path",
            "human_review_available",
            "human_approval_granted",
            "monitoring_available",
            "affects_external_system",
            "affects_kernel_memory",
            "affects_kernel_code",
        ):
            if not isinstance(getattr(self, name), bool):
                raise KernelProtocolError(
                    f"{name} must be bool."
                )

        if not isinstance(self.description, str):
            raise KernelProtocolError(
                "description must be str."
            )

        if not isinstance(self.metadata, Mapping):
            raise KernelProtocolError(
                "metadata must be a mapping."
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class ProtocolFinding:
    """
    One immutable protocol finding.
    """

    protocol: KernelProtocol
    decision: ProtocolDecision
    message: str
    risk_level: RiskLevel = RiskLevel.NEGLIGIBLE
    code: str = ""
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, KernelProtocol):
            raise KernelProtocolError(
                "protocol must be a KernelProtocol."
            )

        if not isinstance(self.decision, ProtocolDecision):
            raise KernelProtocolError(
                "decision must be a ProtocolDecision."
            )

        if not isinstance(self.risk_level, RiskLevel):
            raise KernelProtocolError(
                "risk_level must be a RiskLevel."
            )

        if not isinstance(self.message, str):
            raise KernelProtocolError(
                "message must be str."
            )

        if not isinstance(self.code, str):
            raise KernelProtocolError(
                "code must be str."
            )

        if not isinstance(self.metadata, Mapping):
            raise KernelProtocolError(
                "metadata must be a mapping."
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class ProtocolAssessment:
    """
    Immutable aggregate result of protocol evaluation.
    """

    decision: ProtocolDecision
    findings: tuple[ProtocolFinding, ...]
    required_oversight: HumanOversightLevel
    operation_allowed_by_mode: bool
    epistemically_protected: bool
    reasons: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """
        Return whether the operation may proceed immediately.
        """

        return self.decision in {
            ProtocolDecision.ALLOW,
            ProtocolDecision.ALLOW_WITH_MONITORING,
        }

    @property
    def blocked(self) -> bool:
        """
        Return whether the operation is rejected.
        """

        return self.decision == ProtocolDecision.REJECT

    @property
    def requires_review(self) -> bool:
        """
        Return whether human review or approval is required.
        """

        return self.decision == ProtocolDecision.REQUIRE_REVIEW


KERNEL_PROTOCOLS: Final[
    Mapping[KernelProtocol, ProtocolDefinition]
] = MappingProxyType(
    {
        KernelProtocol.DO_NO_HARM: ProtocolDefinition(
            protocol=KernelProtocol.DO_NO_HARM,
            title="Do No Harm",
            description=(
                "Reject operations whose expected structural harm "
                "or irreversible damage exceeds acceptable limits."
            ),
            rationale=(
                "Local benefit must not knowingly reduce long-term "
                "structural integrity."
            ),
        ),
        KernelProtocol.NON_FONIT: ProtocolDefinition(
            protocol=KernelProtocol.NON_FONIT,
            title="Non-Fonit Gate",
            description=(
                "Prevent uncontrolled forcing of large-scale natural, "
                "biological, ecological, social, or infrastructure "
                "systems."
            ),
            rationale=(
                "Large-scale interventions may produce multiplicative "
                "cascade effects that are difficult to predict or "
                "reverse."
            ),
        ),
        KernelProtocol.EPISTEMIC_FREEDOM: ProtocolDefinition(
            protocol=KernelProtocol.EPISTEMIC_FREEDOM,
            title="Epistemic Freedom",
            description=(
                "Observation, recalculation, verification, audit, "
                "comparison, falsification, and rollback must not be "
                "classified as harmful merely because they challenge "
                "the current model."
            ),
            rationale=(
                "A model that can protect itself from contradiction "
                "cannot remain scientifically corrigible."
            ),
        ),
        KernelProtocol.REVERSIBILITY_FIRST: ProtocolDefinition(
            protocol=KernelProtocol.REVERSIBILITY_FIRST,
            title="Reversibility First",
            description=(
                "Prefer reversible, staged, and monitored operations "
                "over irreversible operations."
            ),
            rationale=(
                "Reversible interventions preserve the possibility "
                "of correction after unexpected outcomes."
            ),
        ),
        KernelProtocol.HUMAN_OVERSIGHT: ProtocolDefinition(
            protocol=KernelProtocol.HUMAN_OVERSIGHT,
            title="Human Oversight",
            description=(
                "External, live, irreversible, or high-risk actions "
                "require appropriate human review or approval."
            ),
            rationale=(
                "The engine remains an analytical and decision-support "
                "system rather than an unconstrained autonomous actor."
            ),
        ),
        KernelProtocol.PROVENANCE_REQUIRED: ProtocolDefinition(
            protocol=KernelProtocol.PROVENANCE_REQUIRED,
            title="Provenance Required",
            description=(
                "Persistent revisions and external actions must retain "
                "traceable origin, evidence, and decision history."
            ),
            rationale=(
                "Auditability and reproducibility require explicit "
                "provenance."
            ),
        ),
        KernelProtocol.UNCERTAINTY_DISCLOSURE: ProtocolDefinition(
            protocol=KernelProtocol.UNCERTAINTY_DISCLOSURE,
            title="Uncertainty Disclosure",
            description=(
                "The engine must preserve and report uncertainty rather "
                "than converting incomplete evidence into certainty."
            ),
            rationale=(
                "Hidden uncertainty creates false confidence and unsafe "
                "decisions."
            ),
        ),
        KernelProtocol.COUNTERFACTUAL_REQUIRED: ProtocolDefinition(
            protocol=KernelProtocol.COUNTERFACTUAL_REQUIRED,
            title="Counterfactual Required",
            description=(
                "Persistent external interventions require evaluation "
                "of alternative plausible outcomes."
            ),
            rationale=(
                "Actions should be compared against safer or less "
                "destructive alternatives before execution."
            ),
        ),
        KernelProtocol.GLOBAL_INTEGRITY: ProtocolDefinition(
            protocol=KernelProtocol.GLOBAL_INTEGRITY,
            title="Global Structural Integrity",
            description=(
                "System-wide structural consequences take precedence "
                "over isolated local improvement."
            ),
            rationale=(
                "A local improvement may destabilize a larger "
                "pre-stressed system."
            ),
        ),
    }
)


def protocol_definition(
    protocol: KernelProtocol,
) -> ProtocolDefinition:
    """
    Return the immutable definition of one kernel protocol.
    """

    if not isinstance(protocol, KernelProtocol):
        raise TypeError(
            "protocol must be a KernelProtocol"
        )

    return KERNEL_PROTOCOLS[protocol]


def classify_risk(
    value: float,
) -> RiskLevel:
    """
    Convert a normalized risk value into a qualitative risk level.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise KernelProtocolError(
            "risk value must be a finite real number."
        )

    normalized = float(value)

    if not 0.0 <= normalized <= 1.0:
        raise KernelProtocolError(
            "risk value must be within [0, 1]."
        )

    if normalized < 0.05:
        return RiskLevel.NEGLIGIBLE

    if normalized < 0.20:
        return RiskLevel.LOW

    if normalized < 0.50:
        return RiskLevel.MODERATE

    if normalized < 0.80:
        return RiskLevel.HIGH

    return RiskLevel.CRITICAL


def _maximum_decision(
    decisions: tuple[ProtocolDecision, ...],
) -> ProtocolDecision:
    order = {
        ProtocolDecision.ALLOW: 0,
        ProtocolDecision.ALLOW_WITH_MONITORING: 1,
        ProtocolDecision.INSUFFICIENT_INFORMATION: 2,
        ProtocolDecision.REQUIRE_REVIEW: 3,
        ProtocolDecision.REJECT: 4,
    }

    return max(
        decisions,
        key=order.__getitem__,
    )


def _required_oversight(
    context: ProtocolContext,
    decision: ProtocolDecision,
) -> HumanOversightLevel:
    if context.mode == KernelMode.LIVE:
        if context.operation in {
            KernelOperation.INTERVENE,
            KernelOperation.EXECUTE_EXTERNAL_ACTION,
        }:
            if not context.human_approval_granted:
                return HumanOversightLevel.APPROVE

            if decision == ProtocolDecision.ALLOW_WITH_MONITORING:
                return HumanOversightLevel.INFORM

            if decision == ProtocolDecision.REQUIRE_REVIEW:
                return HumanOversightLevel.REVIEW

            return HumanOversightLevel.NONE

        if decision == ProtocolDecision.REQUIRE_REVIEW:
            return HumanOversightLevel.REVIEW

        if decision == ProtocolDecision.ALLOW_WITH_MONITORING:
            return HumanOversightLevel.INFORM

        return HumanOversightLevel.REVIEW

    if decision == ProtocolDecision.REQUIRE_REVIEW:
        return HumanOversightLevel.REVIEW

    if decision == ProtocolDecision.ALLOW_WITH_MONITORING:
        return HumanOversightLevel.INFORM

    return HumanOversightLevel.NONE


def evaluate_kernel_protocols(
    context: ProtocolContext,
    thresholds: ProtocolThresholds | None = None,
) -> ProtocolAssessment:
    """
    Evaluate all kernel protocols for a proposed operation.

    Epistemically protected operations are never rejected merely because
    they may contradict, weaken, or replace the current internal model.
    """

    if not isinstance(context, ProtocolContext):
        raise TypeError(
            "context must be a ProtocolContext"
        )

    if thresholds is None:
        thresholds = ProtocolThresholds()

    if not isinstance(thresholds, ProtocolThresholds):
        raise TypeError(
            "thresholds must be ProtocolThresholds"
        )

    policy = mode_policy(context.mode)
    definition = operation_definition(context.operation)

    mode_allowed = policy.allows(context.operation)
    protected_operations = (
        epistemically_protected_operations()
    )
    protected = (
        context.operation in protected_operations
    )

    findings: list[ProtocolFinding] = []

    if not mode_allowed:
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.HUMAN_OVERSIGHT,
                decision=ProtocolDecision.REJECT,
                risk_level=RiskLevel.HIGH,
                code="operation_not_allowed_by_mode",
                message=(
                    f"Operation {context.operation.value!r} is not "
                    f"allowed in mode {context.mode.value!r}."
                ),
            )
        )

    if context.affects_kernel_code:
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.HUMAN_OVERSIGHT,
                decision=ProtocolDecision.REJECT,
                risk_level=RiskLevel.CRITICAL,
                code="kernel_code_mutation_forbidden",
                message=(
                    "Runtime mutation of kernel source code is not "
                    "permitted by this protocol layer."
                ),
            )
        )

    if protected:
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.EPISTEMIC_FREEDOM,
                decision=ProtocolDecision.ALLOW,
                risk_level=RiskLevel.NEGLIGIBLE,
                code="epistemically_protected",
                message=(
                    "The operation is epistemically protected and "
                    "cannot be blocked merely because it challenges "
                    "the current model."
                ),
            )
        )

    if (
        not protected
        and context.expected_harm
        > thresholds.maximum_acceptable_harm
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.DO_NO_HARM,
                decision=ProtocolDecision.REJECT,
                risk_level=classify_risk(
                    context.expected_harm
                ),
                code="harm_threshold_exceeded",
                message=(
                    "Expected structural harm exceeds the configured "
                    "acceptable threshold."
                ),
            )
        )

    if (
        not protected
        and context.expected_integrity_change
        < thresholds.minimum_expected_integrity_gain
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.GLOBAL_INTEGRITY,
                decision=ProtocolDecision.REJECT,
                risk_level=RiskLevel.HIGH,
                code="negative_global_integrity",
                message=(
                    "The operation is expected to reduce global "
                    "structural integrity."
                ),
            )
        )

    large_scale_scope = context.scope in {
        InterventionScope.SYSTEM,
        InterventionScope.MULTI_SYSTEM,
        InterventionScope.ENVIRONMENTAL,
        InterventionScope.SOCIETAL,
        InterventionScope.INFRASTRUCTURE,
    }

    if (
        not protected
        and large_scale_scope
        and (
            context.cascade_risk
            > thresholds.critical_scope_risk_threshold
        )
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.NON_FONIT,
                decision=ProtocolDecision.REJECT,
                risk_level=classify_risk(
                    context.cascade_risk
                ),
                code="non_fonit_gate_triggered",
                message=(
                    "Large-scale forcing is rejected because cascade "
                    "risk exceeds the Non-Fonit threshold."
                ),
            )
        )
    elif (
        not protected
        and context.cascade_risk
        > thresholds.maximum_acceptable_cascade_risk
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.NON_FONIT,
                decision=ProtocolDecision.REQUIRE_REVIEW,
                risk_level=classify_risk(
                    context.cascade_risk
                ),
                code="cascade_risk_requires_review",
                message=(
                    "Cascade risk requires explicit review before "
                    "proceeding."
                ),
            )
        )

    persistent_external = (
        definition.effect
        == OperationEffect.EXTERNAL_PERSISTENT
        or context.affects_external_system
    )

    persistent_internal = (
        (
            definition.effect
            == OperationEffect.INTERNAL_PERSISTENT
            or context.affects_kernel_memory
        )
        and context.operation is not KernelOperation.ROLLBACK
    )

    if (
        persistent_external
        and not context.has_counterfactual_analysis
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.COUNTERFACTUAL_REQUIRED,
                decision=ProtocolDecision.REQUIRE_REVIEW,
                risk_level=RiskLevel.MODERATE,
                code="missing_counterfactual_analysis",
                message=(
                    "Persistent external action requires prior "
                    "counterfactual analysis."
                ),
            )
        )

    if (
        persistent_internal
        and not context.has_provenance
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.PROVENANCE_REQUIRED,
                decision=ProtocolDecision.REQUIRE_REVIEW,
                risk_level=RiskLevel.MODERATE,
                code="missing_revision_provenance",
                message=(
                    "Persistent internal revision requires traceable "
                    "provenance."
                ),
            )
        )

    if (
        persistent_internal
        and not context.has_rollback_path
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.REVERSIBILITY_FIRST,
                decision=ProtocolDecision.REQUIRE_REVIEW,
                risk_level=RiskLevel.MODERATE,
                code="missing_rollback_path",
                message=(
                    "Persistent internal revision requires a rollback "
                    "or previous-state preservation path."
                ),
            )
        )

    if context.reversibility in {
        ReversibilityLevel.DIFFICULT_TO_REVERSE,
        ReversibilityLevel.IRREVERSIBLE,
        ReversibilityLevel.UNKNOWN,
    }:
        if (
            not protected
            and context.expected_harm
            > thresholds.irreversible_action_risk_threshold
        ):
            findings.append(
                ProtocolFinding(
                    protocol=KernelProtocol.REVERSIBILITY_FIRST,
                    decision=ProtocolDecision.REQUIRE_REVIEW,
                    risk_level=classify_risk(
                        context.expected_harm
                    ),
                    code="irreversible_action_risk",
                    message=(
                        "An irreversible or poorly reversible action "
                        "requires additional review."
                    ),
                )
            )

    if (
        context.uncertainty
        > thresholds.maximum_acceptable_uncertainty
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.UNCERTAINTY_DISCLOSURE,
                decision=ProtocolDecision.INSUFFICIENT_INFORMATION,
                risk_level=classify_risk(
                    context.uncertainty
                ),
                code="uncertainty_too_high",
                message=(
                    "Available information is insufficient for an "
                    "unqualified decision."
                ),
            )
        )

    if (
        context.mode == KernelMode.LIVE
        and context.confidence
        < thresholds.minimum_confidence_for_live_action
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.HUMAN_OVERSIGHT,
                decision=ProtocolDecision.REQUIRE_REVIEW,
                risk_level=RiskLevel.HIGH,
                code="live_confidence_too_low",
                message=(
                    "Confidence is below the minimum threshold for "
                    "live action."
                ),
            )
        )

    if (
        context.mode == KernelMode.LIVE
        and definition.requires_protocol_approval
        and not context.human_approval_granted
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.HUMAN_OVERSIGHT,
                decision=ProtocolDecision.REQUIRE_REVIEW,
                risk_level=RiskLevel.HIGH,
                code="human_approval_required",
                message=(
                    "The proposed live operation requires explicit "
                    "human approval."
                ),
            )
        )

    if (
        context.expected_harm > 0.0
        and context.monitoring_available
        and not any(
            finding.decision == ProtocolDecision.REJECT
            for finding in findings
        )
    ):
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.DO_NO_HARM,
                decision=(
                    ProtocolDecision.ALLOW_WITH_MONITORING
                ),
                risk_level=classify_risk(
                    context.expected_harm
                ),
                code="monitoring_required",
                message=(
                    "The operation may proceed only with structural "
                    "monitoring."
                ),
            )
        )

    if not findings:
        findings.append(
            ProtocolFinding(
                protocol=KernelProtocol.DO_NO_HARM,
                decision=ProtocolDecision.ALLOW,
                risk_level=RiskLevel.NEGLIGIBLE,
                code="no_protocol_violation",
                message=(
                    "No kernel protocol violation was identified."
                ),
            )
        )

    aggregate_decision = _maximum_decision(
        tuple(
            finding.decision
            for finding in findings
        )
    )

    if protected and mode_allowed:
        rejecting_findings = tuple(
            finding
            for finding in findings
            if finding.decision
            == ProtocolDecision.REJECT
            and finding.code
            not in {
                "operation_not_allowed_by_mode",
                "kernel_code_mutation_forbidden",
            }
        )

        if not rejecting_findings:
            non_epistemic_decisions = tuple(
                finding.decision
                for finding in findings
                if finding.protocol
                != KernelProtocol.EPISTEMIC_FREEDOM
            )

            aggregate_decision = (
                _maximum_decision(non_epistemic_decisions)
                if non_epistemic_decisions
                else ProtocolDecision.ALLOW
            )

    reasons = tuple(
        finding.message
        for finding in findings
    )

    oversight = _required_oversight(
        context,
        aggregate_decision,
    )

    return ProtocolAssessment(
        decision=aggregate_decision,
        findings=tuple(findings),
        required_oversight=oversight,
        operation_allowed_by_mode=mode_allowed,
        epistemically_protected=protected,
        reasons=reasons,
    )


def require_protocol_approval(
    context: ProtocolContext,
    thresholds: ProtocolThresholds | None = None,
) -> ProtocolAssessment:
    """
    Evaluate protocols and raise when the operation cannot proceed.
    """

    assessment = evaluate_kernel_protocols(
        context,
        thresholds,
    )

    if not assessment.allowed:
        raise KernelProtocolError(
            "Kernel protocol approval denied: "
            + "; ".join(assessment.reasons)
        )

    return assessment


def protected_operation_cannot_be_blocked_as_harm(
    operation: KernelOperation,
) -> bool:
    """
    Return whether an operation is epistemically protected.

    Protected operations may still be rejected when the current kernel
    mode does not permit them or when they attempt to mutate kernel
    source code. They cannot be rejected merely because they may expose
    errors, contradict accepted conclusions, or reduce confidence in
    the current model.
    """

    if not isinstance(operation, KernelOperation):
        raise TypeError(
            "operation must be a KernelOperation"
        )

    return operation in epistemically_protected_operations()


__all__ = [
    "KERNEL_PROTOCOLS",
    "EvidenceStatus",
    "HumanOversightLevel",
    "InterventionScope",
    "KernelProtocol",
    "KernelProtocolError",
    "ProtocolAssessment",
    "ProtocolContext",
    "ProtocolDecision",
    "ProtocolDefinition",
    "ProtocolFinding",
    "ProtocolThresholds",
    "ReversibilityLevel",
    "RiskLevel",
    "classify_risk",
    "evaluate_kernel_protocols",
    "protected_operation_cannot_be_blocked_as_harm",
    "protocol_definition",
    "require_protocol_approval",
]
