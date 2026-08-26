from __future__ import annotations

from dataclasses import dataclass

from .developmental_activation import (
    DevelopmentalActivationInspector,
)
from .developmental_affordance import (
    DevelopmentalCapability,
)
from .developmental_state import DevelopmentalState
from .endogenous_grouping import (
    EndogenousGroupingAnalyzer,
    GroupingResult,
)


@dataclass(frozen=True, slots=True)
class DevelopmentalProcessResult:
    """
    Result of one independently activated developmental process.

    This object describes what happened during one process execution.
    It does not encode stage progression or prescribe any successor.
    """

    capability: DevelopmentalCapability
    executed: bool
    produced_object: object | None
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(
                "reason must not be empty."
            )


class EndogenousGroupingProcess:
    """
    Independently executable endogenous grouping process.

    Fundamental invariant:

        process != stage
        process != transition
        process != scheduler

    The process knows only:
        - its own capability,
        - whether current state makes it activatable,
        - how to read DifferenceProfile memory,
        - how to produce and record GroupingResult.

    It does not know:
        - RDA stage numbers,
        - previous or next developmental stages,
        - any transition table,
        - any priority ordering,
        - any developmental target.
    """

    capability = (
        DevelopmentalCapability
        .DISCOVER_ENDOGENOUS_GROUPING
    )

    def __init__(
        self,
        *,
        analyzer: EndogenousGroupingAnalyzer | None = None,
        activation_inspector: (
            DevelopmentalActivationInspector | None
        ) = None,
    ) -> None:
        self._analyzer = (
            analyzer
            if analyzer is not None
            else EndogenousGroupingAnalyzer()
        )

        self._activation_inspector = (
            activation_inspector
            if activation_inspector is not None
            else DevelopmentalActivationInspector()
        )

    def can_run(
        self,
        state: DevelopmentalState,
    ) -> bool:
        if not isinstance(state, DevelopmentalState):
            raise TypeError(
                "state must be a DevelopmentalState."
            )

        return self.capability in (
            self._activation_inspector
            .activatable_capabilities(state)
        )

    def run(
        self,
        state: DevelopmentalState,
    ) -> DevelopmentalProcessResult:
        if not isinstance(state, DevelopmentalState):
            raise TypeError(
                "state must be a DevelopmentalState."
            )

        if not self.can_run(state):
            return DevelopmentalProcessResult(
                capability=self.capability,
                executed=False,
                produced_object=None,
                reason=(
                    "Current developmental state does not "
                    "support endogenous grouping."
                ),
            )

        grouping = self._analyzer.analyze(
            tuple(state.difference_profiles)
        )

        state.record_grouping_result(
            grouping
        )

        return DevelopmentalProcessResult(
            capability=self.capability,
            executed=True,
            produced_object=grouping,
            reason=(
                "Endogenous grouping was produced from "
                "accumulated DifferenceProfile memory."
            ),
        )
