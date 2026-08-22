from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .observation import Observation


@dataclass(frozen=True, slots=True)
class ChannelDeviation:
    """
    Deviation of one experienced channel from its familiar baseline.

    This object describes only the structure of experienced difference.

    It does not encode:
        - cause,
        - external object,
        - body-part semantics,
        - direction in world space,
        - reward,
        - goal,
        - event identity,
        - valence,
        - damage,
        - meaning.
    """

    channel_name: str
    current_value: float
    familiar_mean: float
    familiar_scale: float
    signed_deviation: float
    absolute_deviation: float

    def __post_init__(self) -> None:
        if not self.channel_name:
            raise ValueError(
                "channel_name must be non-empty."
            )

        numeric_values = (
            self.current_value,
            self.familiar_mean,
            self.familiar_scale,
            self.signed_deviation,
            self.absolute_deviation,
        )

        if not all(
            np.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "ChannelDeviation values must be finite."
            )

        if self.familiar_scale <= 0.0:
            raise ValueError(
                "familiar_scale must be > 0."
            )

        if self.absolute_deviation < 0.0:
            raise ValueError(
                "absolute_deviation must be >= 0."
            )

        if not np.isclose(
            self.absolute_deviation,
            abs(self.signed_deviation),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "absolute_deviation must equal "
                "abs(signed_deviation)."
            )


@dataclass(frozen=True, slots=True)
class DifferenceProfile:
    """
    Multichannel structure of one experienced departure from familiarity.

    Fundamental invariant:

        DifferenceProfile != EventInterpretation

    The profile answers only:

        "How did the current experienced state differ from the familiar one?"

    It does not answer:

        "Why did it happen?"
        "What caused it?"
        "Where in world coordinates did it happen?"
        "Was it good or bad?"
    """

    timestamp: float
    sequence_index: int | None
    channel_deviations: tuple[ChannelDeviation, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError(
                "timestamp must be finite."
            )

        if not self.channel_deviations:
            raise ValueError(
                "channel_deviations must not be empty."
            )

        names = [
            deviation.channel_name
            for deviation in self.channel_deviations
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "channel_deviations contain duplicate channel names."
            )

    def by_name(
        self,
    ) -> Mapping[str, ChannelDeviation]:
        return {
            deviation.channel_name: deviation
            for deviation in self.channel_deviations
        }

    def strongest(
        self,
        *,
        n: int | None = None,
    ) -> tuple[ChannelDeviation, ...]:
        """
        Return deviations ordered by descending absolute magnitude.

        Channel identity remains descriptive only; no semantic grouping
        is inferred here.
        """

        ordered = tuple(
            sorted(
                self.channel_deviations,
                key=lambda deviation: (
                    -deviation.absolute_deviation,
                    deviation.channel_name,
                ),
            )
        )

        if n is None:
            return ordered

        if n <= 0:
            raise ValueError(
                "n must be > 0 when provided."
            )

        return ordered[:n]

    @property
    def maximum_absolute_deviation(
        self,
    ) -> float:
        return max(
            deviation.absolute_deviation
            for deviation in self.channel_deviations
        )

    @property
    def l2_deviation_norm(
        self,
    ) -> float:
        return float(
            np.linalg.norm(
                np.asarray(
                    [
                        deviation.signed_deviation
                        for deviation
                        in self.channel_deviations
                    ],
                    dtype=float,
                )
            )
        )


class EndogenousDifferenceAnalyzer:
    """
    Analyze the internal structure of departure from an experienced baseline.

    The analyzer receives only prior and current Observation objects.

    It has no access to:
        - WorldState,
        - BodyGroundTruth,
        - perturbation time,
        - perturbation node,
        - external force vector,
        - world coordinates,
        - reward,
        - goal,
        - semantic event labels,
        - body-part labels,
        - valence.

    It does not classify an event.

    It produces a DifferenceProfile over the same experienced channels that
    constituted the familiar baseline.
    """

    __slots__ = (
        "_absolute_scale_floor",
        "_channel_names",
        "_mean",
        "_scale",
    )

    def __init__(
        self,
        *,
        absolute_scale_floor: float = 1e-9,
    ) -> None:
        if (
            not np.isfinite(absolute_scale_floor)
            or absolute_scale_floor <= 0.0
        ):
            raise ValueError(
                "absolute_scale_floor must be finite and > 0."
            )

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
    def channel_names(
        self,
    ) -> tuple[str, ...]:
        if self._channel_names is None:
            raise RuntimeError(
                "Analyzer has not been fitted."
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
                if (
                    channel.available
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

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Observation contains non-finite numeric values."
            )

        return names, values

    def fit(
        self,
        observations: Sequence[Observation],
    ) -> None:
        """
        Learn familiar per-channel statistics from prior experience only.
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

        vectors = [
            first_vector
        ]

        for observation in observations[1:]:
            _, vector = self._observation_vector(
                observation,
                expected_names=first_names,
            )

            vectors.append(
                vector
            )

        matrix = np.vstack(
            vectors
        )

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

        self._channel_names = (
            first_names
        )
        self._mean = mean
        self._scale = scale

    def analyze(
        self,
        observation: Observation,
    ) -> DifferenceProfile:
        """
        Construct a multichannel difference profile for one current observation.
        """

        if not self.fitted:
            raise RuntimeError(
                "Analyzer must be fitted before analysis."
            )

        assert self._channel_names is not None
        assert self._mean is not None
        assert self._scale is not None

        _, vector = self._observation_vector(
            observation,
            expected_names=self._channel_names,
        )

        signed = (
            vector - self._mean
        ) / self._scale

        deviations = tuple(
            ChannelDeviation(
                channel_name=name,
                current_value=float(
                    vector[index]
                ),
                familiar_mean=float(
                    self._mean[index]
                ),
                familiar_scale=float(
                    self._scale[index]
                ),
                signed_deviation=float(
                    signed[index]
                ),
                absolute_deviation=float(
                    abs(
                        signed[index]
                    )
                ),
            )
            for index, name in enumerate(
                self._channel_names
            )
        )

        return DifferenceProfile(
            timestamp=float(
                observation.timestamp
            ),
            sequence_index=(
                observation.provenance.sequence_index
            ),
            channel_deviations=deviations,
        )
