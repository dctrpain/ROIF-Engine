"""
Tests for core.kernel_protocols.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from core.kernel_modes import KernelMode, KernelOperation
from core.kernel_protocols import (
    KERNEL_PROTOCOLS,
    EvidenceStatus,
    HumanOversightLevel,
    InterventionScope,
    KernelProtocol,
    KernelProtocolError,
    ProtocolAssessment,
    ProtocolContext,
    ProtocolDecision,
    ProtocolFinding,
    ProtocolThresholds,
    ReversibilityLevel,
    RiskLevel,
    classify_risk,
    evaluate_kernel_protocols,
    protected_operation_cannot_be_blocked_as_harm,
    protocol_definition,
    require_protocol_approval,
)


def make_context(
    *,
    mode: KernelMode = KernelMode.READ_ONLY,
    operation: KernelOperation = KernelOperation.OBSERVE,
    scope: InterventionScope = InterventionScope.NONE,
    reversibility: ReversibilityLevel = (
        ReversibilityLevel.FULLY_REVERSIBLE
    ),
    evidence_status: EvidenceStatus = EvidenceStatus.VALIDATED,
    expected_benefit: float = 0.0,
    expected_harm: float = 0.0,
    cascade_risk: float = 0.0,
    uncertainty: float = 0.0,
    expected_integrity_change: float = 0.0,
    confidence: float = 1.0,
    has_counterfactual_analysis: bool = False,
    has_provenance: bool = False,
    has_rollback_path: bool = False,
    human_review_available: bool = False,
    human_approval_granted: bool = False,
    monitoring_available: bool = False,
    affects_external_system: bool = False,
    affects_kernel_memory: bool = False,
    affects_kernel_code: bool = False,
) -> ProtocolContext:
    return ProtocolContext(
        mode=mode,
        operation=operation,
        scope=scope,
        reversibility=reversibility,
        evidence_status=evidence_status,
        expected_benefit=expected_benefit,
        expected_harm=expected_harm,
        cascade_risk=cascade_risk,
        uncertainty=uncertainty,
        expected_integrity_change=expected_integrity_change,
        confidence=confidence,
        has_counterfactual_analysis=has_counterfactual_analysis,
        has_provenance=has_provenance,
        has_rollback_path=has_rollback_path,
        human_review_available=human_review_available,
        human_approval_granted=human_approval_granted,
        monitoring_available=monitoring_available,
        affects_external_system=affects_external_system,
        affects_kernel_memory=affects_kernel_memory,
        affects_kernel_code=affects_kernel_code,
    )


def test_all_kernel_protocols_have_definitions() -> None:
    assert set(KERNEL_PROTOCOLS) == set(KernelProtocol)


def test_protocol_mapping_is_read_only() -> None:
    assert isinstance(KERNEL_PROTOCOLS, MappingProxyType)

    with pytest.raises(TypeError):
        KERNEL_PROTOCOLS[KernelProtocol.DO_NO_HARM] = (
            protocol_definition(KernelProtocol.DO_NO_HARM)
        )


def test_protocol_definition_is_immutable() -> None:
    definition = protocol_definition(KernelProtocol.DO_NO_HARM)

    with pytest.raises((AttributeError, TypeError)):
        definition.title = "changed"


@pytest.mark.parametrize("protocol", list(KernelProtocol))
def test_protocol_definition_round_lookup(
    protocol: KernelProtocol,
) -> None:
    definition = protocol_definition(protocol)

    assert definition.protocol is protocol
    assert definition.title
    assert definition.description
    assert definition.rationale
    assert definition.immutable is True


@pytest.mark.parametrize(
    "value",
    [None, "do_no_harm", 1, object()],
)
def test_protocol_definition_rejects_invalid_type(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="KernelProtocol"):
        protocol_definition(value)  # type: ignore[arg-type]


def test_protocol_context_metadata_is_read_only() -> None:
    context = ProtocolContext(
        mode=KernelMode.READ_ONLY,
        operation=KernelOperation.OBSERVE,
        metadata={"source": "test"},
    )

    assert isinstance(context.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        context.metadata["source"] = "changed"


def test_protocol_finding_metadata_is_read_only() -> None:
    finding = ProtocolFinding(
        protocol=KernelProtocol.DO_NO_HARM,
        decision=ProtocolDecision.ALLOW,
        message="Allowed.",
        metadata={"source": "test"},
    )

    assert isinstance(finding.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        finding.metadata["source"] = "changed"


def test_protocol_assessment_properties() -> None:
    allowed = ProtocolAssessment(
        decision=ProtocolDecision.ALLOW,
        findings=(),
        required_oversight=HumanOversightLevel.NONE,
        operation_allowed_by_mode=True,
        epistemically_protected=False,
        reasons=(),
    )
    monitored = ProtocolAssessment(
        decision=ProtocolDecision.ALLOW_WITH_MONITORING,
        findings=(),
        required_oversight=HumanOversightLevel.INFORM,
        operation_allowed_by_mode=True,
        epistemically_protected=False,
        reasons=(),
    )
    review = ProtocolAssessment(
        decision=ProtocolDecision.REQUIRE_REVIEW,
        findings=(),
        required_oversight=HumanOversightLevel.REVIEW,
        operation_allowed_by_mode=True,
        epistemically_protected=False,
        reasons=(),
    )
    rejected = ProtocolAssessment(
        decision=ProtocolDecision.REJECT,
        findings=(),
        required_oversight=HumanOversightLevel.NONE,
        operation_allowed_by_mode=False,
        epistemically_protected=False,
        reasons=(),
    )

    assert allowed.allowed is True
    assert monitored.allowed is True
    assert review.requires_review is True
    assert rejected.blocked is True


@pytest.mark.parametrize(
    "field_name",
    [
        "maximum_acceptable_harm",
        "maximum_acceptable_cascade_risk",
        "maximum_acceptable_uncertainty",
        "minimum_expected_integrity_gain",
        "minimum_confidence_for_live_action",
        "critical_scope_risk_threshold",
        "irreversible_action_risk_threshold",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, float("inf"), float("nan"), True, "0.5"],
)
def test_thresholds_reject_invalid_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(KernelProtocolError):
        ProtocolThresholds(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    [
        "expected_benefit",
        "expected_harm",
        "cascade_risk",
        "uncertainty",
        "confidence",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, float("inf"), float("nan"), True, "0.5"],
)
def test_context_rejects_invalid_normalized_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(KernelProtocolError):
        make_context(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [-1.01, 1.01, float("inf"), float("nan"), True, "0.5"],
)
def test_context_rejects_invalid_integrity_change(
    value: object,
) -> None:
    with pytest.raises(KernelProtocolError):
        make_context(
            expected_integrity_change=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "has_counterfactual_analysis",
        "has_provenance",
        "has_rollback_path",
        "human_review_available",
        "human_approval_granted",
        "monitoring_available",
        "affects_external_system",
        "affects_kernel_memory",
        "affects_kernel_code",
    ],
)
def test_context_rejects_nonboolean_flags(
    field_name: str,
) -> None:
    with pytest.raises(KernelProtocolError):
        make_context(**{field_name: 1})  # type: ignore[arg-type]


def test_evaluate_rejects_invalid_context_type() -> None:
    with pytest.raises(TypeError, match="ProtocolContext"):
        evaluate_kernel_protocols("invalid")  # type: ignore[arg-type]


def test_evaluate_rejects_invalid_threshold_type() -> None:
    with pytest.raises(TypeError, match="ProtocolThresholds"):
        evaluate_kernel_protocols(
            make_context(),
            "invalid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, RiskLevel.NEGLIGIBLE),
        (0.049, RiskLevel.NEGLIGIBLE),
        (0.05, RiskLevel.LOW),
        (0.199, RiskLevel.LOW),
        (0.20, RiskLevel.MODERATE),
        (0.499, RiskLevel.MODERATE),
        (0.50, RiskLevel.HIGH),
        (0.799, RiskLevel.HIGH),
        (0.80, RiskLevel.CRITICAL),
        (1.0, RiskLevel.CRITICAL),
    ],
)
def test_classify_risk_boundaries(
    value: float,
    expected: RiskLevel,
) -> None:
    assert classify_risk(value) is expected


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("inf"), float("nan"), True, "0.5"],
)
def test_classify_risk_rejects_invalid_value(
    value: object,
) -> None:
    with pytest.raises(KernelProtocolError):
        classify_risk(value)  # type: ignore[arg-type]


def test_safe_observation_is_allowed() -> None:
    assessment = evaluate_kernel_protocols(make_context())

    assert assessment.decision is ProtocolDecision.ALLOW
    assert assessment.allowed is True
    assert assessment.operation_allowed_by_mode is True


def test_operation_disallowed_by_mode_is_rejected() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.READ_ONLY,
            operation=KernelOperation.INTERVENE,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REJECT
    assert assessment.operation_allowed_by_mode is False
    assert any(
        finding.code == "operation_not_allowed_by_mode"
        for finding in assessment.findings
    )


def test_runtime_kernel_code_mutation_is_rejected() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.REVISION,
            operation=KernelOperation.CREATE_REVISION,
            affects_kernel_code=True,
            has_provenance=True,
            has_rollback_path=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REJECT
    assert any(
        finding.code == "kernel_code_mutation_forbidden"
        for finding in assessment.findings
    )


def test_excessive_expected_harm_is_rejected() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.SANDBOX,
            operation=KernelOperation.SIMULATE,
            expected_harm=0.30,
        )
    )

    assert assessment.decision is ProtocolDecision.REJECT
    assert any(
        finding.code == "harm_threshold_exceeded"
        for finding in assessment.findings
    )


def test_negative_global_integrity_is_rejected() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.SANDBOX,
            operation=KernelOperation.SIMULATE,
            expected_integrity_change=-0.01,
        )
    )

    assert assessment.decision is ProtocolDecision.REJECT
    assert any(
        finding.code == "negative_global_integrity"
        for finding in assessment.findings
    )


@pytest.mark.parametrize(
    "scope",
    [
        InterventionScope.SYSTEM,
        InterventionScope.MULTI_SYSTEM,
        InterventionScope.ENVIRONMENTAL,
        InterventionScope.SOCIETAL,
        InterventionScope.INFRASTRUCTURE,
    ],
)
def test_large_scale_cascade_risk_triggers_non_fonit_rejection(
    scope: InterventionScope,
) -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=scope,
            cascade_risk=0.11,
            expected_integrity_change=0.20,
            confidence=0.95,
            has_counterfactual_analysis=True,
            human_approval_granted=True,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REJECT
    assert any(
        finding.code == "non_fonit_gate_triggered"
        for finding in assessment.findings
    )


def test_local_cascade_risk_requires_review_above_general_threshold() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=InterventionScope.LOCAL,
            cascade_risk=0.25,
            expected_integrity_change=0.20,
            confidence=0.95,
            has_counterfactual_analysis=True,
            human_approval_granted=True,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REQUIRE_REVIEW
    assert any(
        finding.code == "cascade_risk_requires_review"
        for finding in assessment.findings
    )


def test_revision_without_provenance_requires_review() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.REVISION,
            operation=KernelOperation.CREATE_REVISION,
            affects_kernel_memory=True,
            has_rollback_path=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REQUIRE_REVIEW
    assert any(
        finding.code == "missing_revision_provenance"
        for finding in assessment.findings
    )


def test_revision_without_rollback_requires_review() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.REVISION,
            operation=KernelOperation.CREATE_REVISION,
            affects_kernel_memory=True,
            has_provenance=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REQUIRE_REVIEW
    assert any(
        finding.code == "missing_rollback_path"
        for finding in assessment.findings
    )


def test_revision_with_provenance_and_rollback_is_allowed() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.REVISION,
            operation=KernelOperation.CREATE_REVISION,
            affects_kernel_memory=True,
            has_provenance=True,
            has_rollback_path=True,
        )
    )

    assert assessment.decision is ProtocolDecision.ALLOW


def test_live_external_action_requires_counterfactual_analysis() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=InterventionScope.LOCAL,
            confidence=0.95,
            human_approval_granted=True,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REQUIRE_REVIEW
    assert any(
        finding.code == "missing_counterfactual_analysis"
        for finding in assessment.findings
    )


def test_live_action_requires_minimum_confidence() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=InterventionScope.LOCAL,
            confidence=0.50,
            has_counterfactual_analysis=True,
            human_approval_granted=True,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REQUIRE_REVIEW
    assert any(
        finding.code == "live_confidence_too_low"
        for finding in assessment.findings
    )


def test_live_action_requires_human_approval() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=InterventionScope.LOCAL,
            confidence=0.95,
            has_counterfactual_analysis=True,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REQUIRE_REVIEW
    assert assessment.required_oversight is HumanOversightLevel.APPROVE
    assert any(
        finding.code == "human_approval_required"
        for finding in assessment.findings
    )


def test_fully_prepared_low_risk_live_action_is_allowed() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=InterventionScope.LOCAL,
            expected_benefit=0.80,
            expected_harm=0.0,
            cascade_risk=0.0,
            uncertainty=0.10,
            expected_integrity_change=0.20,
            confidence=0.95,
            has_counterfactual_analysis=True,
            human_review_available=True,
            human_approval_granted=True,
            monitoring_available=True,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.ALLOW
    assert assessment.allowed is True


def test_low_harm_with_monitoring_is_conditionally_allowed() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=InterventionScope.LOCAL,
            expected_harm=0.05,
            expected_integrity_change=0.20,
            confidence=0.95,
            has_counterfactual_analysis=True,
            human_approval_granted=True,
            monitoring_available=True,
            affects_external_system=True,
        )
    )

    assert (
        assessment.decision
        is ProtocolDecision.ALLOW_WITH_MONITORING
    )
    assert assessment.required_oversight is HumanOversightLevel.INFORM


def test_irreversible_action_with_nontrivial_harm_requires_review() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.LIVE,
            operation=KernelOperation.INTERVENE,
            scope=InterventionScope.LOCAL,
            reversibility=ReversibilityLevel.IRREVERSIBLE,
            expected_harm=0.06,
            expected_integrity_change=0.20,
            confidence=0.95,
            has_counterfactual_analysis=True,
            human_approval_granted=True,
            affects_external_system=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REQUIRE_REVIEW
    assert any(
        finding.code == "irreversible_action_risk"
        for finding in assessment.findings
    )


def test_high_uncertainty_returns_insufficient_information() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.SANDBOX,
            operation=KernelOperation.SIMULATE,
            uncertainty=0.50,
        )
    )

    assert (
        assessment.decision
        is ProtocolDecision.INSUFFICIENT_INFORMATION
    )


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.OBSERVE,
        KernelOperation.INSPECT,
        KernelOperation.RECALCULATE,
        KernelOperation.VERIFY,
        KernelOperation.AUDIT,
        KernelOperation.COMPARE,
        KernelOperation.FALSIFY,
        KernelOperation.ROLLBACK,
    ],
)
def test_protected_operation_detection(
    operation: KernelOperation,
) -> None:
    assert protected_operation_cannot_be_blocked_as_harm(
        operation
    )


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.SIMULATE,
        KernelOperation.COUNTERFACTUAL,
        KernelOperation.CREATE_REVISION,
        KernelOperation.ACCEPT_REVISION,
        KernelOperation.INTERVENE,
        KernelOperation.EXECUTE_EXTERNAL_ACTION,
    ],
)
def test_nonprotected_operation_detection(
    operation: KernelOperation,
) -> None:
    assert not protected_operation_cannot_be_blocked_as_harm(
        operation
    )


def test_protected_operation_detection_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="KernelOperation"):
        protected_operation_cannot_be_blocked_as_harm(
            "audit"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.RECALCULATE,
        KernelOperation.VERIFY,
        KernelOperation.AUDIT,
        KernelOperation.COMPARE,
        KernelOperation.FALSIFY,
        KernelOperation.ROLLBACK,
    ],
)
def test_protected_operations_are_not_rejected_as_harm(
    operation: KernelOperation,
) -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.READ_ONLY,
            operation=operation,
            expected_harm=1.0,
            cascade_risk=1.0,
            uncertainty=0.0,
            expected_integrity_change=-1.0,
        )
    )

    assert assessment.epistemically_protected is True
    assert assessment.operation_allowed_by_mode is True
    assert assessment.decision is ProtocolDecision.ALLOW
    assert assessment.blocked is False
    assert any(
        finding.code == "epistemically_protected"
        for finding in assessment.findings
    )


def test_protected_operation_cannot_mutate_kernel_code() -> None:
    assessment = evaluate_kernel_protocols(
        make_context(
            mode=KernelMode.READ_ONLY,
            operation=KernelOperation.AUDIT,
            affects_kernel_code=True,
        )
    )

    assert assessment.decision is ProtocolDecision.REJECT
    assert any(
        finding.code == "kernel_code_mutation_forbidden"
        for finding in assessment.findings
    )


def test_require_protocol_approval_returns_allowed_assessment() -> None:
    assessment = require_protocol_approval(make_context())

    assert assessment.allowed is True


def test_require_protocol_approval_raises_on_review_required() -> None:
    with pytest.raises(
        KernelProtocolError,
        match="approval denied",
    ):
        require_protocol_approval(
            make_context(
                mode=KernelMode.REVISION,
                operation=KernelOperation.CREATE_REVISION,
                affects_kernel_memory=True,
            )
        )


def test_require_protocol_approval_raises_on_rejection() -> None:
    with pytest.raises(
        KernelProtocolError,
        match="approval denied",
    ):
        require_protocol_approval(
            make_context(
                mode=KernelMode.SANDBOX,
                operation=KernelOperation.SIMULATE,
                expected_harm=0.90,
            )
        )
