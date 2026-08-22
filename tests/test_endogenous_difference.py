from __future__ import annotations

import pytest

from roif.development.endogenous_difference import (
    ChannelDeviation,
    DifferenceProfile,
    EndogenousDifferenceAnalyzer,
)
from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)


def observation(
    timestamp: float,
    *,
    a: float,
    b: float,
    c: float,
    sequence_index: int,
) -> Observation:
    return Observation(
        timestamp=timestamp,
        channels={
            "channel_a": ObservationChannel(
                name="channel_a",
                value=a,
            ),
            "channel_b": ObservationChannel(
                name="channel_b",
                value=b,
            ),
            "channel_c": ObservationChannel(
                name="channel_c",
                value=c,
            ),
        },
        provenance=ObservationProvenance(
            source_id="abstract_body",
            source_type="abstract_proprioception",
            sequence_index=sequence_index,
        ),
    )


def baseline() -> list[Observation]:
    return [
        observation(
            0.00,
            a=1.000,
            b=2.000,
            c=3.000,
            sequence_index=0,
        ),
        observation(
            0.01,
            a=1.001,
            b=1.999,
            c=3.001,
            sequence_index=1,
        ),
        observation(
            0.02,
            a=0.999,
            b=2.001,
            c=2.999,
            sequence_index=2,
        ),
        observation(
            0.03,
            a=1.000,
            b=2.000,
            c=3.000,
            sequence_index=3,
        ),
    ]


def test_channel_deviation_rejects_empty_name():
    with pytest.raises(ValueError):
        ChannelDeviation(
            channel_name="",
            current_value=1.0,
            familiar_mean=1.0,
            familiar_scale=1.0,
            signed_deviation=0.0,
            absolute_deviation=0.0,
        )


def test_channel_deviation_requires_positive_scale():
    with pytest.raises(ValueError):
        ChannelDeviation(
            channel_name="a",
            current_value=1.0,
            familiar_mean=1.0,
            familiar_scale=0.0,
            signed_deviation=0.0,
            absolute_deviation=0.0,
        )


def test_difference_profile_rejects_duplicate_channels():
    deviation = ChannelDeviation(
        channel_name="a",
        current_value=1.0,
        familiar_mean=1.0,
        familiar_scale=1.0,
        signed_deviation=0.0,
        absolute_deviation=0.0,
    )

    with pytest.raises(ValueError):
        DifferenceProfile(
            timestamp=0.0,
            sequence_index=0,
            channel_deviations=(
                deviation,
                deviation,
            ),
        )


def test_analyzer_is_initially_unfitted():
    analyzer = EndogenousDifferenceAnalyzer()

    assert analyzer.fitted is False


def test_analyzer_requires_multiple_baseline_observations():
    analyzer = EndogenousDifferenceAnalyzer()

    with pytest.raises(ValueError):
        analyzer.fit(
            baseline()[:1]
        )


def test_analyzer_becomes_fitted():
    analyzer = EndogenousDifferenceAnalyzer()

    analyzer.fit(
        baseline()
    )

    assert analyzer.fitted is True
    assert analyzer.channel_names == (
        "channel_a",
        "channel_b",
        "channel_c",
    )


def test_analyzer_requires_fit_before_analysis():
    analyzer = EndogenousDifferenceAnalyzer()

    with pytest.raises(RuntimeError):
        analyzer.analyze(
            baseline()[0]
        )


def test_familiar_state_produces_small_difference_profile():
    analyzer = EndogenousDifferenceAnalyzer()

    analyzer.fit(
        baseline()
    )

    profile = analyzer.analyze(
        observation(
            0.04,
            a=1.0,
            b=2.0,
            c=3.0,
            sequence_index=4,
        )
    )

    assert (
        profile.maximum_absolute_deviation
        < 1.0
    )


def test_profile_preserves_signed_direction_of_difference():
    analyzer = EndogenousDifferenceAnalyzer()

    analyzer.fit(
        baseline()
    )

    profile = analyzer.analyze(
        observation(
            0.04,
            a=1.1,
            b=1.9,
            c=3.0,
            sequence_index=4,
        )
    )

    deviations = profile.by_name()

    assert (
        deviations[
            "channel_a"
        ].signed_deviation
        > 0.0
    )

    assert (
        deviations[
            "channel_b"
        ].signed_deviation
        < 0.0
    )


def test_strongest_orders_channels_by_absolute_difference():
    analyzer = EndogenousDifferenceAnalyzer()

    analyzer.fit(
        baseline()
    )

    profile = analyzer.analyze(
        observation(
            0.04,
            a=1.02,
            b=2.20,
            c=3.001,
            sequence_index=4,
        )
    )

    strongest = profile.strongest()

    assert (
        strongest[0].absolute_deviation
        >= strongest[1].absolute_deviation
        >= strongest[2].absolute_deviation
    )


def test_strongest_can_return_top_n():
    analyzer = EndogenousDifferenceAnalyzer()

    analyzer.fit(
        baseline()
    )

    profile = analyzer.analyze(
        observation(
            0.04,
            a=1.1,
            b=2.1,
            c=3.1,
            sequence_index=4,
        )
    )

    assert len(
        profile.strongest(
            n=2
        )
    ) == 2


def test_profile_has_nonnegative_l2_norm():
    analyzer = EndogenousDifferenceAnalyzer()

    analyzer.fit(
        baseline()
    )

    profile = analyzer.analyze(
        observation(
            0.04,
            a=1.1,
            b=2.0,
            c=3.0,
            sequence_index=4,
        )
    )

    assert (
        profile.l2_deviation_norm
        >= 0.0
    )


def test_analyzer_rejects_schema_change():
    analyzer = EndogenousDifferenceAnalyzer()

    analyzer.fit(
        baseline()
    )

    changed = Observation(
        timestamp=0.04,
        channels={
            "different": ObservationChannel(
                name="different",
                value=1.0,
            )
        },
        provenance=ObservationProvenance(
            source_id="abstract_body",
            source_type="abstract_proprioception",
            sequence_index=4,
        ),
    )

    with pytest.raises(ValueError):
        analyzer.analyze(
            changed
        )


def test_analyzer_supports_constant_baseline_via_scale_floor():
    analyzer = EndogenousDifferenceAnalyzer(
        absolute_scale_floor=1e-9
    )

    constant = [
        observation(
            0.00,
            a=1.0,
            b=2.0,
            c=3.0,
            sequence_index=0,
        ),
        observation(
            0.01,
            a=1.0,
            b=2.0,
            c=3.0,
            sequence_index=1,
        ),
        observation(
            0.02,
            a=1.0,
            b=2.0,
            c=3.0,
            sequence_index=2,
        ),
    ]

    analyzer.fit(
        constant
    )

    profile = analyzer.analyze(
        observation(
            0.03,
            a=1.0,
            b=2.0,
            c=3.0,
            sequence_index=3,
        )
    )

    assert (
        profile.maximum_absolute_deviation
        == pytest.approx(0.0)
    )


def test_analyzer_has_no_ground_truth_or_world_interface():
    analyzer = EndogenousDifferenceAnalyzer()

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


def test_analyzer_has_no_event_reward_goal_or_valence_interface():
    analyzer = EndogenousDifferenceAnalyzer()

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
        "reward",
    )
    assert not hasattr(
        analyzer,
        "goal",
    )
    assert not hasattr(
        analyzer,
        "valence",
    )


def test_analyzer_has_no_external_perturbation_interface():
    analyzer = EndogenousDifferenceAnalyzer()

    assert not hasattr(
        analyzer,
        "perturbation_time",
    )
    assert not hasattr(
        analyzer,
        "perturbation_node",
    )
    assert not hasattr(
        analyzer,
        "external_force",
    )
