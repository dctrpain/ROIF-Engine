from __future__ import annotations

from roif.development.expression_contract import (
    ExpressionClaimKind,
    ExpressionContractBuilder,
)
from roif.development.expression_gate import (
    ExpressionDecision,
    ExpressionEvidence,
    ExpressionKind,
)


def _evidence(
    *,
    deviation_score=None,
    maximum_absolute_deviation=None,
    l2_deviation_norm=None,
    group_count=0,
    strongest_group_strength=None,
    represented_dimension=None,
    residual_variance_fraction=None,
    persistent_residual_dimension_count=None,
    growth_supported=None,
) -> ExpressionEvidence:
    return ExpressionEvidence(
        sequence_index=None,
        timestamp=None,
        deviation_score=deviation_score,
        maximum_absolute_deviation=maximum_absolute_deviation,
        l2_deviation_norm=l2_deviation_norm,
        group_count=group_count,
        strongest_group_strength=strongest_group_strength,
        represented_dimension=represented_dimension,
        residual_variance_fraction=residual_variance_fraction,
        persistent_residual_dimension_count=(
            persistent_residual_dimension_count
        ),
        growth_supported=growth_supported,
    )


def test_no_expression_produces_empty_contract():
    builder = ExpressionContractBuilder()

    decision = ExpressionDecision(
        kind=ExpressionKind.NO_EXPRESSION,
        evidence=_evidence(),
    )

    contract = builder.build(decision)

    assert contract.expression_kind is ExpressionKind.NO_EXPRESSION
    assert contract.claims == ()


def test_state_change_contract_preserves_deviation_claim():
    builder = ExpressionContractBuilder()

    decision = ExpressionDecision(
        kind=ExpressionKind.STATE_CHANGE,
        evidence=_evidence(
            deviation_score=12.0,
        ),
    )

    contract = builder.build(decision)

    assert contract.has(
        ExpressionClaimKind.STATE_CHANGE_PRESENT
    )

    assert (
        contract.value(
            ExpressionClaimKind.STATE_CHANGE_PRESENT
        )
        is True
    )

    assert (
        contract.value(
            ExpressionClaimKind.DEVIATION_SCORE
        )
        == 12.0
    )


def test_relational_structure_contract_preserves_group_claims():
    builder = ExpressionContractBuilder()

    decision = ExpressionDecision(
        kind=ExpressionKind.RELATIONAL_STRUCTURE,
        evidence=_evidence(
            group_count=2,
            strongest_group_strength=0.97,
        ),
    )

    contract = builder.build(decision)

    assert contract.has(
        ExpressionClaimKind.RELATIONAL_STRUCTURE_PRESENT
    )

    assert (
        contract.value(
            ExpressionClaimKind.GROUP_COUNT
        )
        == 2
    )

    assert (
        contract.value(
            ExpressionClaimKind.STRONGEST_GROUP_STRENGTH
        )
        == 0.97
    )


def test_representational_insufficiency_contract_preserves_all_claims():
    builder = ExpressionContractBuilder()

    decision = ExpressionDecision(
        kind=ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY,
        evidence=_evidence(
            represented_dimension=3,
            residual_variance_fraction=0.45759965414789505,
            persistent_residual_dimension_count=2,
            growth_supported=True,
        ),
    )

    contract = builder.build(decision)

    assert contract.has(
        ExpressionClaimKind
        .REPRESENTATIONAL_INSUFFICIENCY_PRESENT
    )

    assert (
        contract.value(
            ExpressionClaimKind
            .REPRESENTATIONAL_INSUFFICIENCY_PRESENT
        )
        is True
    )

    assert contract.has(
        ExpressionClaimKind
        .REPEATING_EXPERIENCE_STRUCTURE_PRESENT
    )

    assert (
        contract.value(
            ExpressionClaimKind
            .REPEATING_EXPERIENCE_STRUCTURE_PRESENT
        )
        is True
    )

    assert (
        contract.value(
            ExpressionClaimKind.REPRESENTED_DIMENSION
        )
        == 3
    )

    assert (
        contract.value(
            ExpressionClaimKind
            .PERSISTENT_RESIDUAL_DIMENSION_COUNT
        )
        == 2
    )

    assert (
        contract.value(
            ExpressionClaimKind.RESIDUAL_VARIANCE_FRACTION
        )
        == 0.45759965414789505
    )


def test_missing_optional_numeric_evidence_does_not_create_claims():
    builder = ExpressionContractBuilder()

    decision = ExpressionDecision(
        kind=ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY,
        evidence=_evidence(
            growth_supported=True,
        ),
    )

    contract = builder.build(decision)

    assert contract.has(
        ExpressionClaimKind
        .REPRESENTATIONAL_INSUFFICIENCY_PRESENT
    )

    assert contract.has(
        ExpressionClaimKind
        .REPEATING_EXPERIENCE_STRUCTURE_PRESENT
    )

    assert not contract.has(
        ExpressionClaimKind.REPRESENTED_DIMENSION
    )

    assert not contract.has(
        ExpressionClaimKind
        .PERSISTENT_RESIDUAL_DIMENSION_COUNT
    )

    assert not contract.has(
        ExpressionClaimKind.RESIDUAL_VARIANCE_FRACTION
    )


def test_difference_structure_contract_preserves_profile_claims():
    builder = ExpressionContractBuilder()

    decision = ExpressionDecision(
        kind=ExpressionKind.DIFFERENCE_STRUCTURE,
        evidence=_evidence(
            maximum_absolute_deviation=999999999.9999944,
            l2_deviation_norm=1000000021.0185108,
        ),
    )

    contract = builder.build(decision)

    assert contract.has(
        ExpressionClaimKind.DIFFERENCE_STRUCTURE_PRESENT
    )

    assert (
        contract.value(
            ExpressionClaimKind.DIFFERENCE_STRUCTURE_PRESENT
        )
        is True
    )

    assert (
        contract.value(
            ExpressionClaimKind.MAXIMUM_ABSOLUTE_DEVIATION
        )
        == 999999999.9999944
    )

    assert (
        contract.value(
            ExpressionClaimKind.L2_DEVIATION_NORM
        )
        == 1000000021.0185108
    )
