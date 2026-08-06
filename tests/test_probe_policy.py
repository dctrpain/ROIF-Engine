"""
Tests for the universal Active Probe policy layer.

The policy evaluates admissibility only. It must not rank, execute, simulate,
or mutate the investigated system.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from roif.probe_entities import (
    Perturbation,
    PerturbationType,
    ProbeDefinition,
    ProbeMethod,
    ProbePurpose,
    ProbeRegime,
)
from roif.probe_policy import (
    InvalidProbePolicyError,
    PolicyReasonCode,
    ProbeAssessment,
    ProbePolicy,
    ProbePolicyContext,
    ProbePolicyDecision,
    ProbePolicyError,
    ProbePolicyReason,
    ProbePolicyResult,
    ProbePolicyRule,
    ProbePolicyRuleResult,
    ProbeReversibility,
    ProbeRiskLevel,
)


# ============================================================================
# Helpers
# ============================================================================


def make_probe(
    identifier: str = "universal.controlled_probe",
    *,
    method: ProbeMethod = ProbeMethod.INSTRUMENTAL,
    purpose: ProbePurpose = ProbePurpose.REDUCE_UNCERTAINTY,
    perturbation: PerturbationType = PerturbationType.LOAD_CHANGE,
    tags: tuple[str, ...] = ("universal", "controlled"),
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=identifier,
        name="Controlled Probe",
        method=method,
        regime=ProbeRegime.STATIC,
        purpose=purpose,
        perturbation=Perturbation(
            kind=perturbation,
            magnitude=1.0,
            magnitude_units="relative",
            duration_seconds=1.0,
        ),
        description="Universal controlled experiment.",
        tags=tags,
    )


def reason_codes(
    result: ProbePolicyResult,
) -> set[PolicyReasonCode]:
    return {reason.code for reason in result.reasons}


class RejectingRule(ProbePolicyRule):
    def evaluate(
        self,
        definition: ProbeDefinition,
        assessment: ProbeAssessment,
        context: ProbePolicyContext,
    ) -> ProbePolicyRuleResult:
        return ProbePolicyRuleResult(
            decision=ProbePolicyDecision.REJECT,
            reason=ProbePolicyReason(
                code=PolicyReasonCode.CUSTOM_RULE_REJECTED,
                message="Rejected by custom rule.",
                blocking=True,
            ),
        )


class AuthorizationRule(ProbePolicyRule):
    def evaluate(
        self,
        definition: ProbeDefinition,
        assessment: ProbeAssessment,
        context: ProbePolicyContext,
    ) -> ProbePolicyRuleResult:
        return ProbePolicyRuleResult(
            decision=ProbePolicyDecision.REQUIRE_AUTHORIZATION,
            reason=ProbePolicyReason(
                code=(
                    PolicyReasonCode
                    .CUSTOM_RULE_AUTHORIZATION_REQUIRED
                ),
                message="Authorization required by custom rule.",
                blocking=False,
            ),
        )


class NoOpinionRule(ProbePolicyRule):
    def evaluate(
        self,
        definition: ProbeDefinition,
        assessment: ProbeAssessment,
        context: ProbePolicyContext,
    ) -> None:
        return None


class InvalidReturnRule(ProbePolicyRule):
    def evaluate(
        self,
        definition: ProbeDefinition,
        assessment: ProbeAssessment,
        context: ProbePolicyContext,
    ) -> object:
        return object()


# ============================================================================
# ProbeAssessment
# ============================================================================


def test_assessment_defaults_are_conservative_and_valid() -> None:
    assessment = ProbeAssessment()

    assert assessment.expected_cost == 0.0
    assert assessment.expected_duration_seconds == 0.0
    assert assessment.risk_level is ProbeRiskLevel.NEGLIGIBLE
    assert assessment.cascade_risk == 0.0
    assert assessment.uncertainty == 0.0
    assert (
        assessment.reversibility
        is ProbeReversibility.FULLY_REVERSIBLE
    )
    assert assessment.required_resources == frozenset()
    assert assessment.affects_external_systems is False
    assert assessment.large_scale_effect_possible is False
    assert assessment.uncontrolled_propagation_possible is False


@pytest.mark.parametrize(
    "field_name",
    (
        "expected_cost",
        "expected_duration_seconds",
    ),
)
def test_assessment_rejects_negative_values(
    field_name: str,
) -> None:
    with pytest.raises(ValueError):
        ProbeAssessment(**{field_name: -1.0})


@pytest.mark.parametrize(
    "field_name",
    (
        "expected_cost",
        "expected_duration_seconds",
    ),
)
def test_assessment_rejects_bool_as_numeric(
    field_name: str,
) -> None:
    with pytest.raises(TypeError):
        ProbeAssessment(**{field_name: True})


@pytest.mark.parametrize(
    "field_name",
    (
        "cascade_risk",
        "uncertainty",
    ),
)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_assessment_rejects_probability_outside_range(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        ProbeAssessment(**{field_name: value})


def test_assessment_rejects_wrong_risk_level() -> None:
    with pytest.raises(TypeError):
        ProbeAssessment(risk_level="low")  # type: ignore[arg-type]


def test_assessment_rejects_wrong_reversibility() -> None:
    with pytest.raises(TypeError):
        ProbeAssessment(
            reversibility="reversible",  # type: ignore[arg-type]
        )


def test_assessment_normalizes_resources() -> None:
    assessment = ProbeAssessment(
        required_resources=frozenset(
            {" sensor ", "operator"}
        )
    )

    assert assessment.required_resources == frozenset(
        {"sensor", "operator"}
    )


def test_assessment_rejects_string_resource_collection() -> None:
    with pytest.raises(TypeError):
        ProbeAssessment(
            required_resources="sensor",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "affects_external_systems",
        "large_scale_effect_possible",
        "uncontrolled_propagation_possible",
        "requires_human_authorization",
    ),
)
def test_assessment_boolean_fields_require_bool(
    field_name: str,
) -> None:
    with pytest.raises(TypeError):
        ProbeAssessment(**{field_name: 1})


def test_assessment_metadata_is_immutable() -> None:
    assessment = ProbeAssessment(
        metadata={"source": "model"}
    )

    assert isinstance(assessment.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        assessment.metadata["source"] = "changed"  # type: ignore[index]


# ============================================================================
# ProbePolicyContext
# ============================================================================


def test_context_defaults() -> None:
    context = ProbePolicyContext()

    assert context.available_resources == frozenset()
    assert context.human_authorized is False
    assert context.maximum_risk_level is ProbeRiskLevel.LOW
    assert context.maximum_cascade_risk == pytest.approx(0.25)
    assert context.maximum_uncertainty == pytest.approx(1.0)
    assert context.allow_irreversible is False
    assert context.allow_unknown_reversibility is False
    assert context.non_fonit_gate_enabled is True


def test_context_normalizes_resources() -> None:
    context = ProbePolicyContext(
        available_resources=frozenset(
            {" sensor ", "operator"}
        )
    )

    assert context.available_resources == frozenset(
        {"sensor", "operator"}
    )


def test_context_rejects_negative_maximum_cost() -> None:
    with pytest.raises(ValueError):
        ProbePolicyContext(maximum_cost=-1.0)


def test_context_rejects_negative_maximum_duration() -> None:
    with pytest.raises(ValueError):
        ProbePolicyContext(
            maximum_duration_seconds=-1.0
        )


def test_context_rejects_wrong_maximum_risk_level() -> None:
    with pytest.raises(TypeError):
        ProbePolicyContext(
            maximum_risk_level="low",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "human_authorized",
        "allow_irreversible",
        "allow_unknown_reversibility",
        "non_fonit_gate_enabled",
    ),
)
def test_context_boolean_fields_require_bool(
    field_name: str,
) -> None:
    with pytest.raises(TypeError):
        ProbePolicyContext(**{field_name: 1})


def test_context_metadata_is_immutable() -> None:
    context = ProbePolicyContext(
        metadata={"environment": "test"}
    )

    assert isinstance(context.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        context.metadata["environment"] = "changed"  # type: ignore[index]


# ============================================================================
# Policy construction
# ============================================================================


def test_empty_policy_allows_unrestricted_safe_probe() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(),
        ProbePolicyContext(),
    )

    assert result.decision is ProbePolicyDecision.ALLOW
    assert result.allowed
    assert PolicyReasonCode.ALLOWED in reason_codes(result)


def test_policy_rejects_method_overlap() -> None:
    with pytest.raises(InvalidProbePolicyError):
        ProbePolicy(
            allowed_methods=frozenset(
                {ProbeMethod.INSTRUMENTAL}
            ),
            forbidden_methods=frozenset(
                {ProbeMethod.INSTRUMENTAL}
            ),
        )


def test_policy_rejects_perturbation_overlap() -> None:
    with pytest.raises(InvalidProbePolicyError):
        ProbePolicy(
            allowed_perturbations=frozenset(
                {PerturbationType.LOAD_CHANGE}
            ),
            forbidden_perturbations=frozenset(
                {PerturbationType.LOAD_CHANGE}
            ),
        )


def test_policy_rejects_tag_overlap() -> None:
    with pytest.raises(InvalidProbePolicyError):
        ProbePolicy(
            required_tags=frozenset({"safe"}),
            forbidden_tags=frozenset({"safe"}),
        )


def test_policy_rejects_wrong_enum_collection_member() -> None:
    with pytest.raises(TypeError):
        ProbePolicy(
            allowed_methods=frozenset(
                {"instrumental"}  # type: ignore[arg-type]
            )
        )


def test_policy_rejects_non_rule_custom_member() -> None:
    with pytest.raises(TypeError):
        ProbePolicy(
            custom_rules=(object(),)  # type: ignore[arg-type]
        )


# ============================================================================
# Method, perturbation and purpose restrictions
# ============================================================================


def test_allowed_methods_act_as_whitelist() -> None:
    policy = ProbePolicy(
        allowed_methods=frozenset(
            {ProbeMethod.SIMULATION}
        )
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.METHOD_NOT_ALLOWED
        in reason_codes(result)
    )


def test_allowed_method_passes_whitelist() -> None:
    probe = make_probe(method=ProbeMethod.SIMULATION)

    policy = ProbePolicy(
        allowed_methods=frozenset(
            {ProbeMethod.SIMULATION}
        )
    )

    assert policy.evaluate(probe).allowed


def test_forbidden_method_rejects_probe() -> None:
    policy = ProbePolicy(
        forbidden_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.METHOD_FORBIDDEN
        in reason_codes(result)
    )


def test_allowed_perturbations_act_as_whitelist() -> None:
    policy = ProbePolicy(
        allowed_perturbations=frozenset(
            {PerturbationType.SIMULATED_INTERVENTION}
        )
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.PERTURBATION_NOT_ALLOWED
        in reason_codes(result)
    )


def test_forbidden_perturbation_rejects_probe() -> None:
    policy = ProbePolicy(
        forbidden_perturbations=frozenset(
            {PerturbationType.LOAD_CHANGE}
        )
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.PERTURBATION_FORBIDDEN
        in reason_codes(result)
    )


def test_allowed_purpose_rejects_other_purpose() -> None:
    policy = ProbePolicy(
        allowed_purposes=frozenset(
            {ProbePurpose.CONFIRM}
        )
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.PURPOSE_NOT_ALLOWED
        in reason_codes(result)
    )


# ============================================================================
# Tags
# ============================================================================


def test_missing_required_tag_rejects_probe() -> None:
    policy = ProbePolicy(
        required_tags=frozenset({"reversible"})
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.REQUIRED_TAG_MISSING
        in reason_codes(result)
    )


def test_required_tag_allows_matching_probe() -> None:
    policy = ProbePolicy(
        required_tags=frozenset({"controlled"})
    )

    assert policy.evaluate(make_probe()).allowed


def test_forbidden_tag_rejects_probe() -> None:
    policy = ProbePolicy(
        forbidden_tags=frozenset({"controlled"})
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.FORBIDDEN_TAG_PRESENT
        in reason_codes(result)
    )


# ============================================================================
# Quantitative limits
# ============================================================================


def test_cost_above_limit_rejects_probe() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(expected_cost=11.0),
        ProbePolicyContext(maximum_cost=10.0),
    )

    assert result.rejected
    assert PolicyReasonCode.COST_EXCEEDED in reason_codes(result)


def test_cost_equal_to_limit_is_allowed() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(expected_cost=10.0),
        ProbePolicyContext(maximum_cost=10.0),
    )

    assert result.allowed


def test_duration_above_limit_rejects_probe() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            expected_duration_seconds=11.0
        ),
        ProbePolicyContext(
            maximum_duration_seconds=10.0
        ),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.DURATION_EXCEEDED
        in reason_codes(result)
    )


@pytest.mark.parametrize(
    ("risk", "allowed"),
    (
        (ProbeRiskLevel.NEGLIGIBLE, True),
        (ProbeRiskLevel.LOW, True),
        (ProbeRiskLevel.MODERATE, False),
        (ProbeRiskLevel.HIGH, False),
        (ProbeRiskLevel.CRITICAL, False),
    ),
)
def test_risk_level_ordering(
    risk: ProbeRiskLevel,
    allowed: bool,
) -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(risk_level=risk),
        ProbePolicyContext(
            maximum_risk_level=ProbeRiskLevel.LOW
        ),
    )

    assert result.allowed is allowed
    assert result.rejected is not allowed


def test_cascade_risk_above_limit_rejects_probe() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(cascade_risk=0.51),
        ProbePolicyContext(maximum_cascade_risk=0.50),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.CASCADE_RISK_EXCEEDED
        in reason_codes(result)
    )


def test_uncertainty_above_limit_rejects_probe() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(uncertainty=0.81),
        ProbePolicyContext(maximum_uncertainty=0.80),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.UNCERTAINTY_EXCEEDED
        in reason_codes(result)
    )


# ============================================================================
# Reversibility
# ============================================================================


def test_irreversible_probe_rejected_by_default() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            reversibility=ProbeReversibility.IRREVERSIBLE
        ),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.IRREVERSIBLE_PROBE_REJECTED
        in reason_codes(result)
    )


def test_irreversible_probe_can_be_explicitly_allowed() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            reversibility=ProbeReversibility.IRREVERSIBLE
        ),
        ProbePolicyContext(allow_irreversible=True),
    )

    assert result.allowed


def test_unknown_reversibility_rejected_by_default() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            reversibility=ProbeReversibility.UNKNOWN
        ),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.UNKNOWN_REVERSIBILITY
        in reason_codes(result)
    )


def test_unknown_reversibility_can_be_explicitly_allowed() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            reversibility=ProbeReversibility.UNKNOWN
        ),
        ProbePolicyContext(
            allow_unknown_reversibility=True
        ),
    )

    assert result.allowed


# ============================================================================
# Resources
# ============================================================================


def test_missing_resource_rejects_probe() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            required_resources=frozenset(
                {"sensor", "operator"}
            )
        ),
        ProbePolicyContext(
            available_resources=frozenset({"sensor"})
        ),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.RESOURCE_MISSING
        in reason_codes(result)
    )

    messages = tuple(
        reason.message
        for reason in result.reasons
        if reason.code is PolicyReasonCode.RESOURCE_MISSING
    )

    assert len(messages) == 1
    assert "operator" in messages[0]


def test_all_required_resources_allow_probe() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            required_resources=frozenset(
                {"sensor", "operator"}
            )
        ),
        ProbePolicyContext(
            available_resources=frozenset(
                {"sensor", "operator", "simulator"}
            )
        ),
    )

    assert result.allowed


# ============================================================================
# Authorization
# ============================================================================


def test_assessment_can_require_authorization() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            requires_human_authorization=True
        ),
        ProbePolicyContext(human_authorized=False),
    )

    assert (
        result.decision
        is ProbePolicyDecision.REQUIRE_AUTHORIZATION
    )
    assert result.authorization_required
    assert not result.allowed
    assert not result.rejected


def test_human_authorization_satisfies_requirement() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            requires_human_authorization=True
        ),
        ProbePolicyContext(human_authorized=True),
    )

    assert result.allowed


def test_method_can_require_authorization() -> None:
    policy = ProbePolicy(
        require_authorization_for_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    result = policy.evaluate(make_probe())

    assert result.authorization_required


def test_perturbation_can_require_authorization() -> None:
    policy = ProbePolicy(
        require_authorization_for_perturbations=frozenset(
            {PerturbationType.LOAD_CHANGE}
        )
    )

    result = policy.evaluate(make_probe())

    assert result.authorization_required


def test_rejection_has_priority_over_authorization() -> None:
    policy = ProbePolicy(
        forbidden_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        ),
        require_authorization_for_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        ),
    )

    result = policy.evaluate(
        make_probe(),
        context=ProbePolicyContext(
            human_authorized=False
        ),
    )

    assert result.decision is ProbePolicyDecision.REJECT
    assert result.rejected
    assert not result.authorization_required


def test_authorization_does_not_override_blocking_rejection() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            risk_level=ProbeRiskLevel.CRITICAL,
            requires_human_authorization=True,
        ),
        ProbePolicyContext(
            maximum_risk_level=ProbeRiskLevel.LOW,
            human_authorized=True,
        ),
    )

    assert result.rejected
    assert PolicyReasonCode.RISK_EXCEEDED in reason_codes(result)


# ============================================================================
# Non-Fonit Gate
# ============================================================================


@pytest.mark.parametrize(
    "assessment",
    (
        ProbeAssessment(affects_external_systems=True),
        ProbeAssessment(large_scale_effect_possible=True),
        ProbeAssessment(
            uncontrolled_propagation_possible=True
        ),
    ),
)
def test_non_fonit_gate_vetoes_cascade_risk(
    assessment: ProbeAssessment,
) -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        assessment,
        ProbePolicyContext(
            maximum_risk_level=ProbeRiskLevel.CRITICAL,
            maximum_cascade_risk=1.0,
            human_authorized=True,
        ),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.NON_FONIT_VETO
        in reason_codes(result)
    )


def test_non_fonit_gate_has_priority_over_authorization() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            affects_external_systems=True,
            requires_human_authorization=True,
        ),
        ProbePolicyContext(human_authorized=True),
    )

    assert result.rejected
    assert (
        PolicyReasonCode.NON_FONIT_VETO
        in reason_codes(result)
    )


def test_non_fonit_gate_can_be_disabled_explicitly() -> None:
    result = ProbePolicy().evaluate(
        make_probe(),
        ProbeAssessment(
            affects_external_systems=True
        ),
        ProbePolicyContext(
            non_fonit_gate_enabled=False
        ),
    )

    assert result.allowed


# ============================================================================
# Custom rules
# ============================================================================


def test_custom_rejecting_rule_rejects_probe() -> None:
    policy = ProbePolicy(
        custom_rules=(RejectingRule(),)
    )

    result = policy.evaluate(make_probe())

    assert result.rejected
    assert (
        PolicyReasonCode.CUSTOM_RULE_REJECTED
        in reason_codes(result)
    )


def test_custom_authorization_rule_requires_authorization() -> None:
    policy = ProbePolicy(
        custom_rules=(AuthorizationRule(),)
    )

    result = policy.evaluate(make_probe())

    assert result.authorization_required
    assert (
        PolicyReasonCode
        .CUSTOM_RULE_AUTHORIZATION_REQUIRED
        in reason_codes(result)
    )


def test_custom_authorization_rule_is_satisfied_by_context() -> None:
    policy = ProbePolicy(
        custom_rules=(AuthorizationRule(),)
    )

    result = policy.evaluate(
        make_probe(),
        context=ProbePolicyContext(
            human_authorized=True
        ),
    )

    assert result.allowed


def test_custom_rule_with_no_opinion_does_not_block() -> None:
    policy = ProbePolicy(
        custom_rules=(NoOpinionRule(),)
    )

    assert policy.evaluate(make_probe()).allowed


def test_invalid_custom_rule_return_raises_policy_error() -> None:
    policy = ProbePolicy(
        custom_rules=(InvalidReturnRule(),)
    )

    with pytest.raises(ProbePolicyError):
        policy.evaluate(make_probe())


def test_custom_rejection_beats_custom_authorization() -> None:
    policy = ProbePolicy(
        custom_rules=(
            AuthorizationRule(),
            RejectingRule(),
        )
    )

    result = policy.evaluate(make_probe())

    assert result.rejected


# ============================================================================
# Reasons and results
# ============================================================================


def test_reason_requires_valid_code() -> None:
    with pytest.raises(TypeError):
        ProbePolicyReason(
            code="allowed",  # type: ignore[arg-type]
            message="Allowed.",
            blocking=False,
        )


def test_reason_requires_nonempty_message() -> None:
    with pytest.raises(ValueError):
        ProbePolicyReason(
            code=PolicyReasonCode.ALLOWED,
            message=" ",
            blocking=False,
        )


def test_reason_requires_bool_blocking() -> None:
    with pytest.raises(TypeError):
        ProbePolicyReason(
            code=PolicyReasonCode.ALLOWED,
            message="Allowed.",
            blocking=0,  # type: ignore[arg-type]
        )


def test_allowed_result_properties() -> None:
    result = ProbePolicy().evaluate(make_probe())

    assert result.allowed
    assert not result.rejected
    assert not result.authorization_required
    assert result.blocking_reasons == ()


def test_rejected_result_exposes_blocking_reasons() -> None:
    result = ProbePolicy(
        forbidden_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    ).evaluate(make_probe())

    assert result.rejected
    assert result.blocking_reasons
    assert all(
        reason.blocking
        for reason in result.blocking_reasons
    )


def test_result_metadata_is_immutable() -> None:
    definition = make_probe()
    assessment = ProbeAssessment()
    context = ProbePolicyContext()

    result = ProbePolicyResult(
        probe_identifier=definition.identifier,
        decision=ProbePolicyDecision.ALLOW,
        reasons=(
            ProbePolicyReason(
                code=PolicyReasonCode.ALLOWED,
                message="Allowed.",
                blocking=False,
            ),
        ),
        assessment=assessment,
        context=context,
        metadata={"source": "test"},
    )

    assert isinstance(result.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        result.metadata["source"] = "changed"  # type: ignore[index]


# ============================================================================
# Evaluation input validation
# ============================================================================


def test_evaluate_rejects_wrong_definition_type() -> None:
    with pytest.raises(TypeError):
        ProbePolicy().evaluate(object())  # type: ignore[arg-type]


def test_evaluate_rejects_wrong_assessment_type() -> None:
    with pytest.raises(TypeError):
        ProbePolicy().evaluate(
            make_probe(),
            assessment=object(),  # type: ignore[arg-type]
        )


def test_evaluate_rejects_wrong_context_type() -> None:
    with pytest.raises(TypeError):
        ProbePolicy().evaluate(
            make_probe(),
            context=object(),  # type: ignore[arg-type]
        )


def test_evaluate_uses_default_assessment_and_context() -> None:
    result = ProbePolicy().evaluate(make_probe())

    assert isinstance(result.assessment, ProbeAssessment)
    assert isinstance(result.context, ProbePolicyContext)


# ============================================================================
# Architectural boundaries
# ============================================================================


def test_policy_has_no_planning_execution_or_solver_methods() -> None:
    policy = ProbePolicy()

    forbidden_methods = (
        "rank",
        "plan",
        "select",
        "execute",
        "solve",
        "simulate",
        "update_graph",
        "apply_graph_update",
    )

    for method_name in forbidden_methods:
        assert not hasattr(policy, method_name)


def test_policy_evaluation_does_not_mutate_definition() -> None:
    definition = make_probe()
    original_tags = definition.tags
    original_perturbation = definition.perturbation

    ProbePolicy().evaluate(definition)

    assert definition.tags == original_tags
    assert definition.perturbation is original_perturbation


def test_policy_evaluation_is_deterministic() -> None:
    definition = make_probe()
    assessment = ProbeAssessment(
        expected_cost=1.0,
        cascade_risk=0.1,
    )
    context = ProbePolicyContext(
        maximum_cost=2.0,
        maximum_cascade_risk=0.2,
    )
    policy = ProbePolicy()

    first = policy.evaluate(
        definition,
        assessment,
        context,
    )
    second = policy.evaluate(
        definition,
        assessment,
        context,
    )

    assert first == second
