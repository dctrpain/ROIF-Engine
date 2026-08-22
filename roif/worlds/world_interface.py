from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import math

from roif.development.body_interface import BodyInterface


Scalar = int | float | bool


@dataclass(frozen=True, slots=True)
class WorldState:
    """
    Complete experimenter-side state of a developmental world.

    WorldState is ground truth belonging to the simulator / benchmark layer.

    It may contain information that:
        - no body can directly sense,
        - no ROIF process should receive directly,
        - exists only for reproducibility and scientific auditing.

    Fundamental invariant:

        WorldState != BodyGroundTruth != Observation
    """

    timestamp: float
    values: Mapping[str, Scalar]
    metadata: Mapping[str, Scalar | str | None]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("WorldState.timestamp must be finite.")

        if not self.values:
            raise ValueError("WorldState.values must not be empty.")

        normalized_values: dict[str, Scalar] = {}

        for name, value in self.values.items():
            if not name or not name.strip():
                raise ValueError(
                    "WorldState value names must be non-empty."
                )

            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"WorldState value {name!r} must be finite."
                )

            normalized_values[name] = value

        object.__setattr__(
            self,
            "values",
            MappingProxyType(normalized_values),
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


class WorldInterface(ABC):
    """
    Abstract developmental world.

    A WorldInterface owns:
        - simulator time,
        - world ground truth,
        - one or more embodied interfaces,
        - progression of physical state.

    It does NOT define:
        - meaning,
        - reward,
        - novelty,
        - goals,
        - causal labels,
        - semantic event categories.

    Those must not be injected by the world into ROIF.
    """

    @property
    @abstractmethod
    def world_id(self) -> str:
        """
        Stable identifier for the world implementation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def time(self) -> float:
        """
        Current simulator time.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float) -> None:
        """
        Advance the physical world by dt seconds.

        This performs world dynamics only.
        It must not interpret the resulting state.
        """
        raise NotImplementedError

    @abstractmethod
    def world_state(self) -> WorldState:
        """
        Return complete simulator-side world state.

        This is available to experimental infrastructure, not to ROIF.
        """
        raise NotImplementedError

    @abstractmethod
    def bodies(self) -> tuple[BodyInterface, ...]:
        """
        Return embodied interfaces present in the world.
        """
        raise NotImplementedError

    def body(self, body_id: str) -> BodyInterface:
        """
        Return one body by stable identifier.
        """
        for body in self.bodies():
            if body.body_id == body_id:
                return body

        raise KeyError(f"Unknown body_id: {body_id!r}")