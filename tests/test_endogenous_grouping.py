from __future__ import annotations

import pytest

from roif.development.endogenous_difference import (
    ChannelDeviation,
    DifferenceProfile,
)
from roif.development.endogenous_grouping import (
    ChannelRelation,
    EndogenousGroup,
    EndogenousGroupingAnalyzer,
    GroupingResult,
)


def profile(
    timestamp: float,
    *,
    sequence_index: int,
    values: dict[str, float],
) -> DifferenceProfile:
    deviations = tuple(
        ChannelDeviation(
            channel_name=name,
            current_value=value,
            familiar_mean=0.0,
            familiar_scale=1.0,
            signed_deviation=value,
            absolute_deviation=abs(value),
        )
        for name, value in sorted(
            values.items()
        )
    )

    return DifferenceProfile(
        timestamp=timestamp,
        sequence_index=sequence_index,
        channel_deviations=deviations,
    )


def correlated_profiles() -> list[DifferenceProfile]:
    """
    Synthetic pattern:

        a and b move together,
        c moves in the opposite direction,
        d varies independently.

    With absolute correlation, a/b/c should form one relational group.
    """

    rows = [
        {
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0,
        },
        {
            "a": 1.0,
            "b": 2.0,
            "c": -1.0,
            "d": 0.5,
        },
        {
            "a": 2.0,
            "b": 4.0,
            "c": -2.0,
            "d": -0.5,
        },
        {
            "a": 3.0,
            "b": 6.0,
            "c": -3.0,
            "d": 0.25,
        },
        {
            "a": 4.0,
            "b": 8.0,
            "c": -4.0,
            "d": -0.25,
        },
    ]

    return [
        profile(
            index * 0.01,
            sequence_index=index,
            values=row,
        )
        for index, row in enumerate(
            rows
        )
    ]


def test_channel_relation_rejects_same_channel():
    with pytest.raises(ValueError):
        ChannelRelation(
            channel_a="a",
            channel_b="a",
            similarity=1.0,
            sample_count=3,
        )


def test_channel_relation_requires_similarity_in_unit_interval():
    with pytest.raises(ValueError):
        ChannelRelation(
            channel_a="a",
            channel_b="b",
            similarity=1.1,
            sample_count=3,
        )


def test_channel_relation_requires_multiple_samples():
    with pytest.raises(ValueError):
        ChannelRelation(
            channel_a="a",
            channel_b="b",
            similarity=1.0,
            sample_count=1,
        )


def test_channel_relation_key_is_deterministic():
    relation = ChannelRelation(
        channel_a="b",
        channel_b="a",
        similarity=0.5,
        sample_count=3,
    )

    assert relation.key == (
        "a",
        "b",
    )


def test_endogenous_group_requires_at_least_two_channels():
    with pytest.raises(ValueError):
        EndogenousGroup(
            group_id=0,
            channel_names=("a",),
            internal_relation_strength=1.0,
        )


def test_endogenous_group_requires_sorted_channels():
    with pytest.raises(ValueError):
        EndogenousGroup(
            group_id=0,
            channel_names=(
                "b",
                "a",
            ),
            internal_relation_strength=1.0,
        )


def test_grouping_result_requires_channels():
    with pytest.raises(ValueError):
        GroupingResult(
            channel_names=(),
            relations=(),
            groups=(),
        )


def test_analyzer_rejects_invalid_relation_threshold():
    with pytest.raises(ValueError):
        EndogenousGroupingAnalyzer(
            relation_threshold=1.1
        )


def test_analyzer_rejects_negative_minimum_variation():
    with pytest.raises(ValueError):
        EndogenousGroupingAnalyzer(
            minimum_temporal_variation=-1.0
        )


def test_analyzer_requires_multiple_profiles():
    analyzer = EndogenousGroupingAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(
            correlated_profiles()[:1]
        )


def test_analyzer_requires_stable_channel_schema():
    analyzer = EndogenousGroupingAnalyzer()

    profiles = correlated_profiles()

    changed = profile(
        0.05,
        sequence_index=5,
        values={
            "a": 1.0,
            "b": 2.0,
            "different": 3.0,
            "d": 0.0,
        },
    )

    with pytest.raises(ValueError):
        analyzer.analyze(
            profiles[:2]
            + [changed]
        )


def test_analyzer_requires_strictly_increasing_timestamps():
    analyzer = EndogenousGroupingAnalyzer()

    profiles = [
        profile(
            0.0,
            sequence_index=0,
            values={
                "a": 0.0,
                "b": 0.0,
            },
        ),
        profile(
            0.0,
            sequence_index=1,
            values={
                "a": 1.0,
                "b": 1.0,
            },
        ),
    ]

    with pytest.raises(ValueError):
        analyzer.analyze(
            profiles
        )


def test_analyzer_computes_all_pairwise_relations():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    # Four channels -> 4 choose 2 = 6 relations.
    assert len(
        result.relations
    ) == 6


def test_perfect_positive_correlation_is_detected():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    relation = result.relation_map()[
        (
            "a",
            "b",
        )
    ]

    assert relation.similarity == pytest.approx(
        1.0
    )


def test_perfect_negative_correlation_is_preserved():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    relation = result.relation_map()[
        (
            "a",
            "c",
        )
    ]

    assert relation.similarity == pytest.approx(
        -1.0
    )

    assert relation.absolute_similarity == pytest.approx(
        1.0
    )


def test_absolute_correlation_can_place_oppositely_signed_channels_together():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    group = result.group_for_channel(
        "a"
    )

    assert group is not None

    assert {
        "a",
        "b",
        "c",
    }.issubset(
        set(
            group.channel_names
        )
    )


def test_independent_channel_is_not_forced_into_correlated_group():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    group = result.group_for_channel(
        "d"
    )

    if group is not None:
        assert not {
            "a",
            "b",
            "c",
        }.issubset(
            set(
                group.channel_names
            )
        )


def test_constant_channel_relations_are_zero():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95,
        minimum_temporal_variation=1e-12,
    )

    profiles = [
        profile(
            0.00,
            sequence_index=0,
            values={
                "a": 1.0,
                "b": 0.0,
            },
        ),
        profile(
            0.01,
            sequence_index=1,
            values={
                "a": 1.0,
                "b": 1.0,
            },
        ),
        profile(
            0.02,
            sequence_index=2,
            values={
                "a": 1.0,
                "b": 2.0,
            },
        ),
    ]

    result = analyzer.analyze(
        profiles
    )

    relation = result.relation_map()[
        (
            "a",
            "b",
        )
    ]

    assert relation.similarity == pytest.approx(
        0.0
    )


def test_connected_components_form_deterministic_groups():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    first = analyzer.analyze(
        correlated_profiles()
    )
    second = analyzer.analyze(
        correlated_profiles()
    )

    assert first.groups == second.groups


def test_group_ids_are_deterministic_and_zero_based():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    ids = [
        group.group_id
        for group in result.groups
    ]

    assert ids == list(
        range(
            len(ids)
        )
    )


def test_internal_group_strength_is_bounded():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    for group in result.groups:
        assert (
            0.0
            <= group.internal_relation_strength
            <= 1.0
        )


def test_group_for_unknown_channel_returns_none():
    analyzer = EndogenousGroupingAnalyzer(
        relation_threshold=0.95
    )

    result = analyzer.analyze(
        correlated_profiles()
    )

    assert (
        result.group_for_channel(
            "unknown"
        )
        is None
    )


def test_analyzer_has_no_ground_truth_or_world_interface():
    analyzer = EndogenousGroupingAnalyzer()

    assert not hasattr(
        analyzer,
        "ground_truth",
    )
    assert not hasattr(
        analyzer,
        "world_state",
    )
    assert not hasattr(
        analyzer,
        "body_ground_truth",
    )


def test_analyzer_has_no_topology_interface():
    analyzer = EndogenousGroupingAnalyzer()

    assert not hasattr(
        analyzer,
        "topology",
    )
    assert not hasattr(
        analyzer,
        "node_mapping",
    )
    assert not hasattr(
        analyzer,
        "member_mapping",
    )


def test_analyzer_has_no_perturbation_interface():
    analyzer = EndogenousGroupingAnalyzer()

    assert not hasattr(
        analyzer,
        "perturbation_node",
    )
    assert not hasattr(
        analyzer,
        "perturbation_time",
    )
    assert not hasattr(
        analyzer,
        "external_force",
    )


def test_analyzer_has_no_semantic_body_part_interface():
    analyzer = EndogenousGroupingAnalyzer()

    assert not hasattr(
        analyzer,
        "body_part",
    )
    assert not hasattr(
        analyzer,
        "anatomy",
    )
    assert not hasattr(
        analyzer,
        "left",
    )
    assert not hasattr(
        analyzer,
        "right",
    )


def test_analyzer_has_no_reward_goal_event_or_valence_interface():
    analyzer = EndogenousGroupingAnalyzer()

    assert not hasattr(
        analyzer,
        "reward",
    )
    assert not hasattr(
        analyzer,
        "goal",
    )
    assert not hasattr(
        analyzer,
        "event",
    )
    assert not hasattr(
        analyzer,
        "event_type",
    )
    assert not hasattr(
        analyzer,
        "valence",
    )
