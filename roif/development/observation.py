from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional
import math


Scalar = int | float | bool


@dataclass(frozen=True, slots=True)
class ObservationChannel:
    """
    One directly available observation channel.

    This object carries only measured / exposed values and quality metadata.
    It must not encode semantic interpretation such as:
        - cause
        - novelty
        - reward
        - good / bad
        - external object identity
        - world-space meaning
    """

    name: str
    value: Scalar
    unit: Optional[str] = None
    available: bool = True
    quality: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ObservationChannel.name must be non-empty.")

        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(
                f"ObservationChannel.value for {self.name!r} must be finite."
            )

        if self.quality is not None:
            if not math.isfinite(self.quality):
                raise ValueError(
                    f"ObservationChannel.quality for {self.name!r} must be finite."
                )
            if not 0.0 <= self.quality <= 1.0:
                raise ValueError(
                    f"ObservationChannel.quality for {self.name!r} "
                    "must be in [0, 1]."
                )


@dataclass(frozen=True, slots=True)
class ObservationProvenance:
    """
    Describes where an observation came from without interpreting what it means.
    """

    source_id: str
    source_type: str
    sequence_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("ObservationProvenance.source_id must be non-empty.")

        if not self.source_type or not self.source_type.strip():
            raise ValueError(
                "ObservationProvenance.source_type must be non-empty."
            )

        if self.sequence_index is not None and self.sequence_index < 0:
            raise ValueError(
                "ObservationProvenance.sequence_index must be >= 0."
            )


@dataclass(frozen=True, slots=True)
class Observation:
    """
    Raw developmental observation available to ROIF.

    Fundamental RDA-0 invariant:

        Observation != Interpretation != Meaning

    Observation represents only what is available through the body's exposed
    channels at a particular moment.

    World ground truth must remain outside this object.
    """

    timestamp: float
    channels: Mapping[str, ObservationChannel]
    provenance: ObservationProvenance
    metadata: Mapping[str, Scalar | str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("Observation.timestamp must be finite.")

        if not self.channels:
            raise ValueError("Observation.channels must not be empty.")

        normalized_channels: dict[str, ObservationChannel] = {}

        for key, channel in self.channels.items():
            if not key or not key.strip():
                raise ValueError("Observation channel keys must be non-empty.")

            if key != channel.name:
                raise ValueError(
                    f"Channel key {key!r} does not match "
                    f"ObservationChannel.name {channel.name!r}."
                )

            if key in normalized_channels:
                raise ValueError(f"Duplicate observation channel: {key!r}")

            normalized_channels[key] = channel

        object.__setattr__(
            self,
            "channels",
            MappingProxyType(normalized_channels),
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    def get(self, name: str) -> ObservationChannel:
        """
        Return a named observation channel.

        Raises KeyError if the channel is not present.
        """
        return self.channels[name]

    def available_channels(self) -> tuple[ObservationChannel, ...]:
        """
        Return only channels currently marked as available.
        """
        return tuple(
            channel
            for channel in self.channels.values()
            if channel.available
        )

    def unavailable_channels(self) -> tuple[ObservationChannel, ...]:
        """
        Return channels currently marked as unavailable.
        """
        return tuple(
            channel
            for channel in self.channels.values()
            if not channel.available
        )

    def numeric_vector(
        self,
        *,
        include_unavailable: bool = False,
    ) -> tuple[float, ...]:
        """
        Return numeric channel values in deterministic channel-name order.

        This is a structural convenience only.
        No semantic meaning is inferred from channel order or values.
        """

        values: list[float] = []

        for name in sorted(self.channels):
            channel = self.channels[name]

            if not include_unavailable and not channel.available:
                continue

            value = channel.value

            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            else:
                values.append(float(value))

        return tuple(values)