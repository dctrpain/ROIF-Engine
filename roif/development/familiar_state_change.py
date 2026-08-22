from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .observation import Observation


@dataclass(frozen=True, slots=True)
class ChangeDetection:
    """
    Result of comparing one experienced observation with a familiar-state
    regularity estimated only from earlier experienced observations.

    No event semantics, causal label, reward, goal, world coordinates,
    perturbation identity, or valence are represented here.
    """

    sequence_index: int | None
    timestamp: float
    deviation_score: float
    changed: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite.")

        if not np.isfinite(self.deviation_score):
            raise ValueError("deviation_score must be finite.")

        if self.deviation_score < 0.0:
            raise ValueError("deviation_score must be >= 0.")


class FamiliarStateChangeDetector:
    """
    Minimal developmental primitive for detecting departure from an
    experienced familiar state.

    Fundamental invariant:

        current Observation
        vs.
        regularity learned from previous Observations

    The detector receives no simulator ground truth and has no knowledge of:
        - perturbation time,
        - perturbation node,
        - force vector,
        - world-space coordinates,
        - reward,
        - goal,
        - semantic event labels,
        - meaning or valence.

    It does not answer "what happened?".
    It only estimates "how unlike the familiar baseline is this observation?".
    """

    __slots__ = (
        "_threshold",
        "_absolute_scale_floor",
        "_channel_names",
        "_mean",
        "_scale",
    )

    def __init__(
        self,
        *,
        threshold: float = 8.0,
        absolute_scale_floor: float = 1e-9,
    ) -> None:
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be finite and > 0.")

        if (
            not np.isfinite(absolute_scale_floor)
            or absolute_scale_floor <= 0.0
        ):
            raise ValueError(
                "absolute_scale_floor must be finite and > 0."
            )

        self._threshold = float(threshold)
        self._absolute_scale_floor = float(
            absolute_scale_floor
        )

        self._channel_names: tuple[str, ...] | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return (
            self._channel_names is not None
            and self._mean is not None
            and self._scale is not None
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def channel_names(self) -> tuple[str, ...]:
        if self._channel_names is None:
            raise RuntimeError(
                "Detector has not been fitted."
            )
        return self._channel_names

    def _observation_vector(
        self,
        observation: Observation,
        *,
        expected_names: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        names = tuple(
            sorted(
                name
                for name, channel
                in observation.channels.items()
                if channel.available
                and isinstance(
                    channel.value,
                    (int, float),
                )
                and not isinstance(
                    channel.value,
                    bool,
                )
            )
        )

        if not names:
            raise ValueError(
                "Observation has no available numeric channels."
            )

        if (
            expected_names is not None
            and names != expected_names
        ):
            raise ValueError(
                "Observation channel schema changed."
            )

        values = np.asarray(
            [
                float(
                    observation.channels[name].value
                )
                for name in names
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Observation contains non-finite values."
            )

        return names, values

    def fit(
        self,
        observations: Sequence[Observation],
    ) -> None:
        """
        Learn the familiar-state regularity from prior experience only.
        """

        if len(observations) < 2:
            raise ValueError(
                "At least two baseline observations are required."
            )

        first_names, first_vector = (
            self._observation_vector(
                observations[0]
            )
        )

        vectors = [first_vector]

        for observation in observations[1:]:
            _, vector = self._observation_vector(
                observation,
                expected_names=first_names,
            )
            vectors.append(vector)

        matrix = np.vstack(vectors)

        mean = np.mean(
            matrix,
            axis=0,
        )

        scale = np.std(
            matrix,
            axis=0,
            ddof=1,
        )

        scale = np.maximum(
            scale,
            self._absolute_scale_floor,
        )

        self._channel_names = first_names
        self._mean = mean
        self._scale = scale

    def score(
        self,
        observation: Observation,
    ) -> float:
        """
        Return the maximum standardized deviation across familiar channels.
        """

        if not self.fitted:
            raise RuntimeError(
                "Detector must be fitted before scoring."
            )

        assert self._channel_names is not None
        assert self._mean is not None
        assert self._scale is not None

        _, vector = self._observation_vector(
            observation,
            expected_names=self._channel_names,
        )

        standardized = np.abs(
            (vector - self._mean)
            / self._scale
        )

        return float(
            np.max(standardized)
        )

    def detect(
        self,
        observation: Observation,
    ) -> ChangeDetection:
        """
        Evaluate one current observation against the familiar baseline.
        """

        score = self.score(
            observation
        )

        return ChangeDetection(
            sequence_index=(
                observation.provenance.sequence_index
            ),
            timestamp=float(
                observation.timestamp
            ),
            deviation_score=score,
            changed=bool(
                score >= self._threshold
            ),
        )
