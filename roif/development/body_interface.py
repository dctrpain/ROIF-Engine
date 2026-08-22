from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import math

from .observation import Observation


Scalar = int | float | bool


@dataclass(frozen=True, slots=True)
class BodyGroundTruth:
    """
    Complete simulator-side physical state of a body.

    This object belongs to the experimental world / simulator boundary.
    It MUST NOT be passed directly into the ROIF developmental layer.

    RDA-0 invariant:

        BodyGroundTruth != Observation

    Ground truth may contain information unavailable to the body itself,
    including world-space coordinates, externally applied forces, or
    simulator-only state.
    """

    timestamp: float
    values: Mapping[str, Scalar]
    metadata: Mapping[str, Scalar | str | None]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("BodyGroundTruth.timestamp must be finite.")

        if not self.values:
            raise ValueError("BodyGroundTruth.values must not be empty.")

        normalized_values: dict[str, Scalar] = {}

        for name, value in self.values.items():
            if not name or not name.strip():
                raise ValueError(
                    "BodyGroundTruth value names must be non-empty."
                )

            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"BodyGroundTruth value {name!r} must be finite."
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


class BodyInterface(ABC):
    """
    Boundary between a physical/simulated body and ROIF developmental input.

    A BodyInterface is responsible for exposing only information that the
    body is permitted to sense.

    It must not leak simulator ground truth into Observation.
    """

    @property
    @abstractmethod
    def body_id(self) -> str:
        """
        Stable identifier of the body interface.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Observation:
        """
        Return the body's current directly available observation.

        The returned Observation must contain only exposed sensory channels,
        never simulator interpretation or hidden world ground truth.
        """
        raise NotImplementedError

    @abstractmethod
    def ground_truth(self) -> BodyGroundTruth:
        """
        Return complete physical state for experimental auditing.

        This method exists for the experimenter and benchmark infrastructure,
        not for developmental reasoning.
        """
        raise NotImplementedError