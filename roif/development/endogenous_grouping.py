from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .endogenous_difference import DifferenceProfile


@dataclass(frozen=True, slots=True)
class ChannelRelation:
    """
    Symmetric relation between two experienced channels.

    The relation is derived only from the temporal co-variation of their
    endogenous difference signals.

    It does not encode:
        - body-part identity,
        - topology supplied by the simulator,
        - node membership,
        - member identity,
        - world-space distance,
        - causal direction,
        - reward,
        - goal,
        - event type,
        - valence,
        - semantic meaning.
    """

    channel_a: str
    channel_b: str
    similarity: float
    sample_count: int

    def __post_init__(self) -> None:
        if not self.channel_a:
            raise ValueError(
                "channel_a must be non-empty."
            )

        if not self.channel_b:
            raise ValueError(
                "channel_b must be non-empty."
            )

        if self.channel_a == self.channel_b:
            raise ValueError(
                "ChannelRelation requires two different channels."
            )

        if not np.isfinite(self.similarity):
            raise ValueError(
                "similarity must be finite."
            )

        if not -1.0 <= self.similarity <= 1.0:
            raise ValueError(
                "similarity must lie in [-1, 1]."
            )

        if self.sample_count < 2:
            raise ValueError(
                "sample_count must be >= 2."
            )

    @property
    def absolute_similarity(self) -> float:
        return abs(
            self.similarity
        )

    @property
    def key(
        self,
    ) -> tuple[str, str]:
        return tuple(
            sorted(
                (
                    self.channel_a,
                    self.channel_b,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class EndogenousGroup:
    """
    Group of channels discovered from shared experienced dynamics.

    Fundamental invariant:

        EndogenousGroup != BodyPartLabel

    A group is an internally discovered relational cluster only.
    """

    group_id: int
    channel_names: tuple[str, ...]
    internal_relation_strength: float

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError(
                "group_id must be >= 0."
            )

        if len(self.channel_names) < 2:
            raise ValueError(
                "An EndogenousGroup requires at least two channels."
            )

        if tuple(
            sorted(
                self.channel_names
            )
        ) != self.channel_names:
            raise ValueError(
                "channel_names must be sorted deterministically."
            )

        if len(
            self.channel_names
        ) != len(
            set(
                self.channel_names
            )
        ):
            raise ValueError(
                "channel_names must be unique."
            )

        if not np.isfinite(
            self.internal_relation_strength
        ):
            raise ValueError(
                "internal_relation_strength must be finite."
            )

        if not (
            0.0
            <= self.internal_relation_strength
            <= 1.0
        ):
            raise ValueError(
                "internal_relation_strength must lie in [0, 1]."
            )


@dataclass(frozen=True, slots=True)
class GroupingResult:
    """
    Complete relational grouping result over one temporal profile sequence.
    """

    channel_names: tuple[str, ...]
    relations: tuple[ChannelRelation, ...]
    groups: tuple[EndogenousGroup, ...]

    def __post_init__(self) -> None:
        if not self.channel_names:
            raise ValueError(
                "channel_names must not be empty."
            )

        if tuple(
            sorted(
                self.channel_names
            )
        ) != self.channel_names:
            raise ValueError(
                "channel_names must be sorted deterministically."
            )

        if len(
            self.channel_names
        ) != len(
            set(
                self.channel_names
            )
        ):
            raise ValueError(
                "channel_names must be unique."
            )

    def relation_map(
        self,
    ) -> Mapping[
        tuple[str, str],
        ChannelRelation,
    ]:
        return {
            relation.key: relation
            for relation in self.relations
        }

    def group_for_channel(
        self,
        channel_name: str,
    ) -> EndogenousGroup | None:
        for group in self.groups:
            if channel_name in group.channel_names:
                return group

        return None


class EndogenousGroupingAnalyzer:
    """
    Discover stable channel groups from temporal co-deviation patterns.

    Input:
        a sequence of DifferenceProfile objects.

    Output:
        pairwise channel relations and emergent groups.

    No externally supplied body topology is used.

    The analyzer has no access to:
        - WorldState,
        - BodyGroundTruth,
        - tensegrity node topology,
        - member-to-node mapping,
        - world coordinates,
        - perturbation node,
        - perturbation force,
        - reward,
        - goal,
        - event labels,
        - body-part labels,
        - valence.

    Grouping rule in v1:
        1. Build one signed-deviation time series per channel.
        2. Compute pairwise Pearson correlation.
        3. Convert correlation into an undirected relation.
        4. Connect channels whose absolute correlation is greater than
           or equal to relation_threshold.
        5. Connected components with at least two channels become
           EndogenousGroup objects.

    This intentionally remains simple and auditable.
    """

    __slots__ = (
        "_relation_threshold",
        "_minimum_temporal_variation",
    )

    def __init__(
        self,
        *,
        relation_threshold: float = 0.95,
        minimum_temporal_variation: float = 1e-12,
    ) -> None:
        if not np.isfinite(
            relation_threshold
        ):
            raise ValueError(
                "relation_threshold must be finite."
            )

        if not (
            0.0
            <= relation_threshold
            <= 1.0
        ):
            raise ValueError(
                "relation_threshold must lie in [0, 1]."
            )

        if (
            not np.isfinite(
                minimum_temporal_variation
            )
            or minimum_temporal_variation < 0.0
        ):
            raise ValueError(
                "minimum_temporal_variation must be finite and >= 0."
            )

        self._relation_threshold = float(
            relation_threshold
        )

        self._minimum_temporal_variation = float(
            minimum_temporal_variation
        )

    @property
    def relation_threshold(
        self,
    ) -> float:
        return self._relation_threshold

    @property
    def minimum_temporal_variation(
        self,
    ) -> float:
        return self._minimum_temporal_variation

    @staticmethod
    def _profile_channel_names(
        profile: DifferenceProfile,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                deviation.channel_name
                for deviation
                in profile.channel_deviations
            )
        )

    def _validate_profiles(
        self,
        profiles: Sequence[DifferenceProfile],
    ) -> tuple[str, ...]:
        if len(profiles) < 2:
            raise ValueError(
                "At least two DifferenceProfile objects are required."
            )

        expected_names = (
            self._profile_channel_names(
                profiles[0]
            )
        )

        if not expected_names:
            raise ValueError(
                "Profiles contain no channels."
            )

        previous_timestamp = (
            profiles[0].timestamp
        )

        for profile in profiles[1:]:
            names = (
                self._profile_channel_names(
                    profile
                )
            )

            if names != expected_names:
                raise ValueError(
                    "DifferenceProfile channel schema changed."
                )

            if (
                profile.timestamp
                <= previous_timestamp
            ):
                raise ValueError(
                    "DifferenceProfile timestamps must be strictly increasing."
                )

            previous_timestamp = (
                profile.timestamp
            )

        return expected_names

    @staticmethod
    def _profile_map(
        profile: DifferenceProfile,
    ) -> Mapping[str, float]:
        return {
            deviation.channel_name: (
                deviation.signed_deviation
            )
            for deviation
            in profile.channel_deviations
        }

    def _series_matrix(
        self,
        profiles: Sequence[DifferenceProfile],
        channel_names: tuple[str, ...],
    ) -> np.ndarray:
        matrix = np.empty(
            (
                len(profiles),
                len(channel_names),
            ),
            dtype=float,
        )

        for row_index, profile in enumerate(
            profiles
        ):
            values = self._profile_map(
                profile
            )

            for column_index, channel_name in enumerate(
                channel_names
            ):
                matrix[
                    row_index,
                    column_index,
                ] = float(
                    values[channel_name]
                )

        if not np.all(
            np.isfinite(
                matrix
            )
        ):
            raise ValueError(
                "Difference profiles contain non-finite deviations."
            )

        return matrix

    def _pairwise_relation(
        self,
        *,
        channel_a: str,
        channel_b: str,
        series_a: np.ndarray,
        series_b: np.ndarray,
    ) -> ChannelRelation:
        variation_a = float(
            np.std(
                series_a,
                ddof=1,
            )
        )

        variation_b = float(
            np.std(
                series_b,
                ddof=1,
            )
        )

        if (
            variation_a
            <= self._minimum_temporal_variation
            or variation_b
            <= self._minimum_temporal_variation
        ):
            similarity = 0.0
        else:
            correlation = np.corrcoef(
                series_a,
                series_b,
            )[0, 1]

            if not np.isfinite(
                correlation
            ):
                similarity = 0.0
            else:
                similarity = float(
                    np.clip(
                        correlation,
                        -1.0,
                        1.0,
                    )
                )

        return ChannelRelation(
            channel_a=channel_a,
            channel_b=channel_b,
            similarity=similarity,
            sample_count=int(
                series_a.shape[0]
            ),
        )

    def _relations(
        self,
        *,
        channel_names: tuple[str, ...],
        matrix: np.ndarray,
    ) -> tuple[ChannelRelation, ...]:
        relations: list[
            ChannelRelation
        ] = []

        for index_a in range(
            len(channel_names)
        ):
            for index_b in range(
                index_a + 1,
                len(channel_names),
            ):
                relations.append(
                    self._pairwise_relation(
                        channel_a=(
                            channel_names[
                                index_a
                            ]
                        ),
                        channel_b=(
                            channel_names[
                                index_b
                            ]
                        ),
                        series_a=matrix[
                            :,
                            index_a,
                        ],
                        series_b=matrix[
                            :,
                            index_b,
                        ],
                    )
                )

        return tuple(
            relations
        )

    def _adjacency(
        self,
        *,
        channel_names: tuple[str, ...],
        relations: Sequence[ChannelRelation],
    ) -> dict[str, set[str]]:
        adjacency = {
            channel_name: set()
            for channel_name in channel_names
        }

        for relation in relations:
            if (
                relation.absolute_similarity
                >= self._relation_threshold
            ):
                adjacency[
                    relation.channel_a
                ].add(
                    relation.channel_b
                )

                adjacency[
                    relation.channel_b
                ].add(
                    relation.channel_a
                )

        return adjacency

    @staticmethod
    def _connected_components(
        adjacency: Mapping[
            str,
            set[str],
        ],
    ) -> tuple[
        tuple[str, ...],
        ...
    ]:
        visited: set[str] = set()
        components: list[
            tuple[str, ...]
        ] = []

        for start in sorted(
            adjacency
        ):
            if start in visited:
                continue

            stack = [
                start
            ]

            component: set[str] = set()

            while stack:
                current = stack.pop()

                if current in visited:
                    continue

                visited.add(
                    current
                )

                component.add(
                    current
                )

                for neighbor in sorted(
                    adjacency[current],
                    reverse=True,
                ):
                    if neighbor not in visited:
                        stack.append(
                            neighbor
                        )

            if len(component) >= 2:
                components.append(
                    tuple(
                        sorted(
                            component
                        )
                    )
                )

        return tuple(
            sorted(
                components
            )
        )

    @staticmethod
    def _internal_strength(
        *,
        component: tuple[str, ...],
        relation_map: Mapping[
            tuple[str, str],
            ChannelRelation,
        ],
    ) -> float:
        strengths: list[
            float
        ] = []

        for index_a in range(
            len(component)
        ):
            for index_b in range(
                index_a + 1,
                len(component),
            ):
                key = tuple(
                    sorted(
                        (
                            component[
                                index_a
                            ],
                            component[
                                index_b
                            ],
                        )
                    )
                )

                relation = relation_map[
                    key
                ]

                strengths.append(
                    relation.absolute_similarity
                )

        if not strengths:
            return 0.0

        return float(
            np.mean(
                strengths
            )
        )

    def analyze(
        self,
        profiles: Sequence[DifferenceProfile],
    ) -> GroupingResult:
        """
        Discover endogenous channel relations and groups.
        """

        channel_names = (
            self._validate_profiles(
                profiles
            )
        )

        matrix = self._series_matrix(
            profiles,
            channel_names,
        )

        relations = self._relations(
            channel_names=channel_names,
            matrix=matrix,
        )

        adjacency = self._adjacency(
            channel_names=channel_names,
            relations=relations,
        )

        components = (
            self._connected_components(
                adjacency
            )
        )

        relation_map = {
            relation.key: relation
            for relation in relations
        }

        groups = tuple(
            EndogenousGroup(
                group_id=group_id,
                channel_names=component,
                internal_relation_strength=(
                    self._internal_strength(
                        component=component,
                        relation_map=relation_map,
                    )
                ),
            )
            for group_id, component in enumerate(
                components
            )
        )

        return GroupingResult(
            channel_names=channel_names,
            relations=relations,
            groups=groups,
        )
