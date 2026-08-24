from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .familiar_state_change import ChangeDetection
from .endogenous_difference import DifferenceProfile
from .endogenous_grouping import GroupingResult
from .dimensional_growth import DimensionalGrowthAssessment


class ExpressionKind(str, Enum):
    """
    Structural expression categories.

    These are not emotions, semantic interpretations, goals,
    rewards, causes, or world-model claims.

    They describe only what the developmental machinery currently
    supports saying about its own internally available state.
    """

    NO_EXPRESSION = "no_expression"
    STATE_CHANGE = "state_change"
    RELATIONAL_STRUCTURE = "relational_structure"
    REPRESENTATIONAL_INSUFFICIENCY = "representational_insufficiency"


@dataclass(frozen=True, slots=True)
class ExpressionEvidence:
    """
    Numerical evidence available to the expression layer.

    No additional semantic interpretation is introduced here.
    """

    sequence_index: Optional[int]
    timestamp: Optional[float]

    deviation_score: Optional[float]
    maximum_absolute_deviation: Optional[float]
    l2_deviation_norm: Optional[float]

    group_count: int
    strongest_group_strength: Optional[float]

    represented_dimension: Optional[int]
    residual_variance_fraction: Optional[float]
    persistent_residual_dimension_count: Optional[int]
    growth_supported: Optional[bool]


@dataclass(frozen=True, slots=True)
class ExpressionDecision:
    """
    A decision about whether the current internal state supports
    expression.

    Fundamental invariant:

        ExpressionDecision != NaturalLanguageMeaning

    This object does not decide what the state "means".
    It exposes only which structural claim is supported and the
    evidence from which that claim was derived.
    """

    kind: ExpressionKind
    evidence: ExpressionEvidence

    @property
    def should_express(self) -> bool:
        return self.kind is not ExpressionKind.NO_EXPRESSION


class ExpressionGate:
    """
    Minimal gate between developmental state and a future
    language adapter.

    Priority is intentionally conservative:

        representational insufficiency
            >
        relational structure
            >
        state change
            >
        no expression

    The gate introduces no external semantics.
    """

    def decide(
        self,
        *,
        change: ChangeDetection | None = None,
        difference: DifferenceProfile | None = None,
        grouping: GroupingResult | None = None,
        dimensional_growth: DimensionalGrowthAssessment | None = None,
    ) -> ExpressionDecision:

        sequence_index: int | None = None
        timestamp: float | None = None

        deviation_score: float | None = None
        maximum_absolute_deviation: float | None = None
        l2_deviation_norm: float | None = None

        group_count = 0
        strongest_group_strength: float | None = None

        represented_dimension: int | None = None
        residual_variance_fraction: float | None = None
        persistent_residual_dimension_count: int | None = None
        growth_supported: bool | None = None

        if change is not None:
            sequence_index = change.sequence_index
            timestamp = change.timestamp
            deviation_score = change.deviation_score

        if difference is not None:
            if sequence_index is None:
                sequence_index = difference.sequence_index

            if timestamp is None:
                timestamp = difference.timestamp

            maximum_absolute_deviation = (
                difference.maximum_absolute_deviation
            )
            l2_deviation_norm = difference.l2_deviation_norm

        if grouping is not None:
            group_count = len(grouping.groups)

            if grouping.groups:
                strongest_group_strength = max(
                    group.internal_relation_strength
                    for group in grouping.groups
                )

        if dimensional_growth is not None:
            represented_dimension = (
                dimensional_growth.represented_dimension
            )
            residual_variance_fraction = (
                dimensional_growth.residual_variance_fraction
            )
            persistent_residual_dimension_count = (
                dimensional_growth.persistent_residual_dimension_count
            )
            growth_supported = dimensional_growth.growth_supported

        evidence = ExpressionEvidence(
            sequence_index=sequence_index,
            timestamp=timestamp,
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

        if (
            dimensional_growth is not None
            and dimensional_growth.growth_supported
        ):
            kind = ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY

        elif grouping is not None and grouping.groups:
            kind = ExpressionKind.RELATIONAL_STRUCTURE

        elif change is not None and change.changed:
            kind = ExpressionKind.STATE_CHANGE

        else:
            kind = ExpressionKind.NO_EXPRESSION

        return ExpressionDecision(
            kind=kind,
            evidence=evidence,
        )