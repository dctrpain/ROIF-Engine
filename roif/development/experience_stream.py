from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .observation import Observation


@dataclass(frozen=True, slots=True)
class ExperienceSample:
    """
    One position in an uninterrupted developmental experience stream.

    A sample records occurrence, not interpretation.

    It does not state whether the observation is:
        - novel,
        - important,
        - causal,
        - good or bad,
        - an event,
        - worthy of memory.
    """

    index: int
    observation: Observation

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("ExperienceSample.index must be >= 0.")


class ExperienceStream:
    """
    Ordered stream of observations experienced by one embodied interface.

    Fundamental invariant:

        ExperienceStream != StructuredMemory

    The stream preserves temporal occurrence and continuity only.
    Semantic segmentation and memory formation belong to later ROIF layers.
    """

    __slots__ = (
        "_body_id",
        "_samples",
        "_last_timestamp",
    )

    def __init__(self, body_id: str) -> None:
        if not body_id or not body_id.strip():
            raise ValueError("body_id must be non-empty.")

        self._body_id = body_id
        self._samples: list[ExperienceSample] = []
        self._last_timestamp: float | None = None

    @property
    def body_id(self) -> str:
        return self._body_id

    def append(self, observation: Observation) -> ExperienceSample:
        """
        Append the next directly experienced observation.

        Requirements:
            - observations must belong to the same body,
            - timestamps must be strictly increasing after the first sample.

        No semantic interpretation is performed.
        """

        if observation.provenance.source_id != self._body_id:
            raise ValueError(
                "Observation source_id does not match ExperienceStream body_id."
            )

        if (
            self._last_timestamp is not None
            and observation.timestamp <= self._last_timestamp
        ):
            raise ValueError(
                "ExperienceStream timestamps must be strictly increasing."
            )

        sample = ExperienceSample(
            index=len(self._samples),
            observation=observation,
        )

        self._samples.append(sample)
        self._last_timestamp = observation.timestamp

        return sample

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[ExperienceSample]:
        return iter(tuple(self._samples))

    def __getitem__(self, index: int) -> ExperienceSample:
        return self._samples[index]

    @property
    def first(self) -> ExperienceSample | None:
        if not self._samples:
            return None
        return self._samples[0]

    @property
    def latest(self) -> ExperienceSample | None:
        if not self._samples:
            return None
        return self._samples[-1]

    @property
    def duration(self) -> float:
        if len(self._samples) < 2:
            return 0.0

        return (
            self._samples[-1].observation.timestamp
            - self._samples[0].observation.timestamp
        )

    def snapshot(self) -> tuple[ExperienceSample, ...]:
        """
        Return an immutable snapshot of the experienced sequence.
        """
        return tuple(self._samples)