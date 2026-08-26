from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .developmental_state import DevelopmentalState


class DevelopmentalCapability(str, Enum):
    """
    Computations that may become available from accumulated
    developmental evidence.

    These are capabilities, not stages.

    Their declaration does not imply:
        - temporal order,
        - a transition graph,
        - a required predecessor,
        - a required successor,
        - a developmental goal.
    """

    DETECT_FAMILIAR_STATE_CHANGE = (
        "detect_familiar_state_change"
    )
    ANALYZE_ENDOGENOUS_DIFFERENCE = (
        "analyze_endogenous_difference"
    )
    DISCOVER_ENDOGENOUS_GROUPING = (
        "discover_endogenous_grouping"
    )
    ASSESS_DIMENSIONAL_GROWTH = (
        "assess_dimensional_growth"
    )


@dataclass(frozen=True, slots=True)
class DevelopmentalAffordance:
    """
    One capability supported or unsupported by the currently
    accumulated developmental evidence.

    This object is descriptive only.

    It does not execute a computation and does not prescribe
    what computation should happen next.
    """

    capability: DevelopmentalCapability
    available: bool
    evidence_count: int
    reason: str

    def __post_init__(self) -> None:
        if self.evidence_count < 0:
            raise ValueError(
                "evidence_count must be non-negative."
            )

        if not self.reason:
            raise ValueError(
                "reason must not be empty."
            )


class DevelopmentalAffordanceInspector:
    """
    Inspect what computations are supported by accumulated
    developmental memory.

    Fundamental invariant:

        affordance != transition

    The inspector contains no:
        - current_stage,
        - next_stage,
        - transition table,
        - RDA stage ordering,
        - reward,
        - goal,
        - world-ground-truth instruction.

    It only examines evidence already present in
    DevelopmentalState.
    """

    def inspect(
        self,
        state: DevelopmentalState,
    ) -> tuple[DevelopmentalAffordance, ...]:
        if not isinstance(state, DevelopmentalState):
            raise TypeError(
                "state must be a DevelopmentalState."
            )

        observation_count = len(state.observations)
        difference_count = len(state.difference_profiles)

        return (
            DevelopmentalAffordance(
                capability=(
                    DevelopmentalCapability
                    .DETECT_FAMILIAR_STATE_CHANGE
                ),
                available=observation_count >= 3,
                evidence_count=observation_count,
                reason=(
                    "At least three accumulated observations "
                    "permit prior observations to define a "
                    "familiar baseline while leaving a current "
                    "observation for comparison."
                    if observation_count >= 3
                    else
                    "More accumulated observations are required "
                    "to define a prior familiar baseline and a "
                    "separate current observation."
                ),
            ),
            DevelopmentalAffordance(
                capability=(
                    DevelopmentalCapability
                    .ANALYZE_ENDOGENOUS_DIFFERENCE
                ),
                available=observation_count >= 3,
                evidence_count=observation_count,
                reason=(
                    "Accumulated observations permit a baseline "
                    "to be fitted from prior experience and a "
                    "current observation to be analyzed."
                    if observation_count >= 3
                    else
                    "More accumulated observations are required "
                    "for baseline fitting and current difference "
                    "analysis."
                ),
            ),
            DevelopmentalAffordance(
                capability=(
                    DevelopmentalCapability
                    .DISCOVER_ENDOGENOUS_GROUPING
                ),
                available=difference_count >= 2,
                evidence_count=difference_count,
                reason=(
                    "Multiple accumulated endogenous difference "
                    "profiles are available for temporal "
                    "co-deviation analysis."
                    if difference_count >= 2
                    else
                    "Multiple endogenous difference profiles are "
                    "required before temporal relations can be "
                    "examined."
                ),
            ),
            DevelopmentalAffordance(
                capability=(
                    DevelopmentalCapability
                    .ASSESS_DIMENSIONAL_GROWTH
                ),
                available=False,
                evidence_count=observation_count,
                reason=(
                    "DevelopmentalState does not yet expose an "
                    "explicit current representation together "
                    "with the experience matrix required for "
                    "dimensional-growth assessment."
                ),
            ),
        )
