from __future__ import annotations

from dataclasses import dataclass

from .developmental_affordance import (
    DevelopmentalAffordance,
    DevelopmentalAffordanceInspector,
    DevelopmentalCapability,
)
from .developmental_state import DevelopmentalState


@dataclass(frozen=True, slots=True)
class DevelopmentalActivation:
    """
    Describes whether one developmental capability is currently
    activatable from accumulated endogenous state.

    Fundamental invariant:

        activation != transition
        activation != scheduling
        activation != execution

    An activation does not identify a next developmental stage and
    does not prescribe which available computation should run.

    Multiple capabilities may be activatable simultaneously.
    """

    capability: DevelopmentalCapability
    activatable: bool
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


class DevelopmentalActivationInspector:
    """
    Inspect which developmental capabilities are independently
    activatable from the current developmental state.

    This object contains no:
        - current_stage,
        - next_stage,
        - transition table,
        - priority ordering,
        - scheduler,
        - executor,
        - reward,
        - goal.

    Availability is evaluated independently for every capability.

    Therefore zero, one, or multiple capabilities may be
    activatable at the same time.
    """

    def __init__(
        self,
        affordance_inspector: (
            DevelopmentalAffordanceInspector | None
        ) = None,
    ) -> None:
        self._affordance_inspector = (
            affordance_inspector
            if affordance_inspector is not None
            else DevelopmentalAffordanceInspector()
        )

    def inspect(
        self,
        state: DevelopmentalState,
    ) -> tuple[DevelopmentalActivation, ...]:
        if not isinstance(state, DevelopmentalState):
            raise TypeError(
                "state must be a DevelopmentalState."
            )

        affordances = self._affordance_inspector.inspect(
            state
        )

        return tuple(
            self._from_affordance(affordance)
            for affordance in affordances
        )

    @staticmethod
    def _from_affordance(
        affordance: DevelopmentalAffordance,
    ) -> DevelopmentalActivation:
        return DevelopmentalActivation(
            capability=affordance.capability,
            activatable=affordance.available,
            evidence_count=affordance.evidence_count,
            reason=affordance.reason,
        )

    def activatable_capabilities(
        self,
        state: DevelopmentalState,
    ) -> tuple[DevelopmentalCapability, ...]:
        """
        Return every currently activatable capability.

        No capability is selected as preferred or next.
        """

        return tuple(
            activation.capability
            for activation in self.inspect(state)
            if activation.activatable
        )
