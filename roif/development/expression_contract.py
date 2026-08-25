from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

from .expression_gate import (
    ExpressionDecision,
    ExpressionKind,
)


ClaimValue = Union[bool, int, float]


class ExpressionClaimKind(str, Enum):
    """
    Machine-readable claims already supported by an ExpressionDecision.

    These claims do not introduce new interpretation.
    They make explicit which parts of the admitted RDA expression
    must survive linguistic verbalization.
    """

    STATE_CHANGE_PRESENT = "state_change_present"

    DIFFERENCE_STRUCTURE_PRESENT = (
        "difference_structure_present"
    )
    MAXIMUM_ABSOLUTE_DEVIATION = (
        "maximum_absolute_deviation"
    )
    L2_DEVIATION_NORM = "l2_deviation_norm"

    RELATIONAL_STRUCTURE_PRESENT = (
        "relational_structure_present"
    )
    GROUP_COUNT = "group_count"
    STRONGEST_GROUP_STRENGTH = (
        "strongest_group_strength"
    )

    REPRESENTATIONAL_INSUFFICIENCY_PRESENT = (
        "representational_insufficiency_present"
    )
    REPEATING_EXPERIENCE_STRUCTURE_PRESENT = (
        "repeating_experience_structure_present"
    )
    REPRESENTED_DIMENSION = "represented_dimension"
    PERSISTENT_RESIDUAL_DIMENSION_COUNT = (
        "persistent_residual_dimension_count"
    )
    RESIDUAL_VARIANCE_FRACTION = (
        "residual_variance_fraction"
    )

    DEVIATION_SCORE = "deviation_score"


@dataclass(frozen=True, slots=True)
class ExpressionClaim:
    """
    One claim that is already supported before natural-language
    generation.

    The claim is not a sentence and does not prescribe wording.
    """

    kind: ExpressionClaimKind
    value: ClaimValue


@dataclass(frozen=True, slots=True)
class ExpressionContract:
    """
    Semantic preservation contract for one ExpressionDecision.

    Fundamental invariant:

        ExpressionContract does not create new RDA content.

    It only makes explicit which supported claims must not be lost
    during optional generative verbalization.
    """

    expression_kind: ExpressionKind
    claims: tuple[ExpressionClaim, ...]

    def has(
        self,
        kind: ExpressionClaimKind,
    ) -> bool:
        return any(
            claim.kind is kind
            for claim in self.claims
        )

    def get(
        self,
        kind: ExpressionClaimKind,
    ) -> ExpressionClaim | None:
        for claim in self.claims:
            if claim.kind is kind:
                return claim

        return None

    def value(
        self,
        kind: ExpressionClaimKind,
    ) -> ClaimValue | None:
        claim = self.get(kind)

        if claim is None:
            return None

        return claim.value


class ExpressionContractBuilder:
    """
    Build a preservation contract directly from ExpressionDecision.

    No world state, hidden simulator state, ground truth, causal
    interpretation, reward, goal, valence, or LLM output is used.
    """

    def build(
        self,
        decision: ExpressionDecision,
    ) -> ExpressionContract:
        evidence = decision.evidence
        claims: list[ExpressionClaim] = []

        if decision.kind is ExpressionKind.NO_EXPRESSION:
            return ExpressionContract(
                expression_kind=decision.kind,
                claims=(),
            )

        if decision.kind is ExpressionKind.STATE_CHANGE:
            claims.append(
                ExpressionClaim(
                    kind=(
                        ExpressionClaimKind
                        .STATE_CHANGE_PRESENT
                    ),
                    value=True,
                )
            )

            if evidence.deviation_score is not None:
                claims.append(
                    ExpressionClaim(
                        kind=(
                            ExpressionClaimKind
                            .DEVIATION_SCORE
                        ),
                        value=evidence.deviation_score,
                    )
                )

        elif (
            decision.kind
            is ExpressionKind.DIFFERENCE_STRUCTURE
        ):
            claims.append(
                ExpressionClaim(
                    kind=(
                        ExpressionClaimKind
                        .DIFFERENCE_STRUCTURE_PRESENT
                    ),
                    value=True,
                )
            )

            if (
                evidence.maximum_absolute_deviation
                is not None
            ):
                claims.append(
                    ExpressionClaim(
                        kind=(
                            ExpressionClaimKind
                            .MAXIMUM_ABSOLUTE_DEVIATION
                        ),
                        value=(
                            evidence
                            .maximum_absolute_deviation
                        ),
                    )
                )

            if evidence.l2_deviation_norm is not None:
                claims.append(
                    ExpressionClaim(
                        kind=(
                            ExpressionClaimKind
                            .L2_DEVIATION_NORM
                        ),
                        value=evidence.l2_deviation_norm,
                    )
                )

        elif (
            decision.kind
            is ExpressionKind.RELATIONAL_STRUCTURE
        ):
            claims.append(
                ExpressionClaim(
                    kind=(
                        ExpressionClaimKind
                        .RELATIONAL_STRUCTURE_PRESENT
                    ),
                    value=True,
                )
            )

            claims.append(
                ExpressionClaim(
                    kind=ExpressionClaimKind.GROUP_COUNT,
                    value=evidence.group_count,
                )
            )

            if (
                evidence.strongest_group_strength
                is not None
            ):
                claims.append(
                    ExpressionClaim(
                        kind=(
                            ExpressionClaimKind
                            .STRONGEST_GROUP_STRENGTH
                        ),
                        value=(
                            evidence
                            .strongest_group_strength
                        ),
                    )
                )

        elif (
            decision.kind
            is ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY
        ):
            claims.append(
                ExpressionClaim(
                    kind=(
                        ExpressionClaimKind
                        .REPRESENTATIONAL_INSUFFICIENCY_PRESENT
                    ),
                    value=True,
                )
            )

            # The dimensional-growth assessment concerns a
            # structured, repeatable residual in experience.
            # This is already part of the admitted RDA claim and
            # must not disappear during verbalization.
            claims.append(
                ExpressionClaim(
                    kind=(
                        ExpressionClaimKind
                        .REPEATING_EXPERIENCE_STRUCTURE_PRESENT
                    ),
                    value=True,
                )
            )

            if evidence.represented_dimension is not None:
                claims.append(
                    ExpressionClaim(
                        kind=(
                            ExpressionClaimKind
                            .REPRESENTED_DIMENSION
                        ),
                        value=(
                            evidence.represented_dimension
                        ),
                    )
                )

            if (
                evidence
                .persistent_residual_dimension_count
                is not None
            ):
                claims.append(
                    ExpressionClaim(
                        kind=(
                            ExpressionClaimKind
                            .PERSISTENT_RESIDUAL_DIMENSION_COUNT
                        ),
                        value=(
                            evidence
                            .persistent_residual_dimension_count
                        ),
                    )
                )

            if (
                evidence.residual_variance_fraction
                is not None
            ):
                claims.append(
                    ExpressionClaim(
                        kind=(
                            ExpressionClaimKind
                            .RESIDUAL_VARIANCE_FRACTION
                        ),
                        value=(
                            evidence
                            .residual_variance_fraction
                        ),
                    )
                )

        else:
            raise ValueError(
                f"Unsupported ExpressionKind: {decision.kind!r}"
            )

        return ExpressionContract(
            expression_kind=decision.kind,
            claims=tuple(claims),
        )
