from __future__ import annotations

from typing import Iterable, Protocol

from .developmental_affordance import (
    DevelopmentalCapability,
)
from .developmental_state import DevelopmentalState


class DevelopmentalProcessLike(Protocol):
    """
    Minimal structural contract for an independent developmental process.

    A process exposes its capability and can report whether the current
    developmental state makes that computation runnable.

    This protocol does not define execution order.
    """

    capability: DevelopmentalCapability

    def can_run(
        self,
        state: DevelopmentalState,
    ) -> bool:
        ...


class DevelopmentalProcessSet:
    """
    Unordered collection of independent developmental processes.

    Fundamental invariant:

        process set != queue
        process set != scheduler
        process set != developmental path

    The set may report which capabilities are currently runnable.

    It does not:
        - execute processes,
        - choose a preferred process,
        - define priority,
        - define first or next,
        - define stage progression,
        - define transition order.

    Capability identity is unique inside one process set.
    """

    __slots__ = (
        "_processes_by_capability",
    )

    def __init__(
        self,
        processes: Iterable[DevelopmentalProcessLike],
    ) -> None:
        by_capability: dict[
            DevelopmentalCapability,
            DevelopmentalProcessLike,
        ] = {}

        for process in processes:
            capability = process.capability

            if capability in by_capability:
                raise ValueError(
                    "Duplicate developmental capability: "
                    f"{capability.value}"
                )

            by_capability[capability] = process

        self._processes_by_capability = by_capability

    @property
    def capabilities(
        self,
    ) -> frozenset[DevelopmentalCapability]:
        """
        Return registered capabilities with no ordering semantics.
        """

        return frozenset(
            self._processes_by_capability
        )

    def contains(
        self,
        capability: DevelopmentalCapability,
    ) -> bool:
        return capability in self._processes_by_capability

    def process_for(
        self,
        capability: DevelopmentalCapability,
    ) -> DevelopmentalProcessLike:
        """
        Return the process implementing one explicit capability.

        This lookup does not imply developmental precedence.
        """

        try:
            return self._processes_by_capability[
                capability
            ]
        except KeyError as exc:
            raise KeyError(
                "No process registered for capability "
                f"{capability.value!r}."
            ) from exc

    def runnable_capabilities(
        self,
        state: DevelopmentalState,
    ) -> frozenset[DevelopmentalCapability]:
        """
        Return every currently runnable capability as an unordered set.

        No runnable capability is designated first, preferred, or next.
        """

        if not isinstance(state, DevelopmentalState):
            raise TypeError(
                "state must be a DevelopmentalState."
            )

        return frozenset(
            capability
            for capability, process
            in self._processes_by_capability.items()
            if process.can_run(state)
        )
