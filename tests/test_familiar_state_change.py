import pytest

from roif.development.familiar_state_change import (
    ChangeDetection,
    FamiliarStateChangeDetector,
)
from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)


def observation(
    timestamp: float,
    *,
    value_a: float,
    value_b: float,
    sequence_index: int,
) -> Observation:
    return Observation(
        timestamp=timestamp,
        channels={
            "a": ObservationChannel(
                name="a",
                value=value_a,
            ),
            "b": ObservationChannel(
                name="b",
                value=value_b,
            ),
        },
        provenance=ObservationProvenance(
            source_id="body_0",
            source_type="proprioceptive_body",
            sequence_index=sequence_index,
        ),
    )


def familiar_baseline() -> list[Observation]:
    return [
        observation(
            0.00,
            value_a=1.000,
            value_b=2.000,
            sequence_index=0,
        ),
        observation(
            0.01,
            value_a=1.001,
            value_b=2.001,
            sequence_index=1,
        ),
        observation(
            0.02,
            value_a=0.999,
            value_b=1.999,
            sequence_index=2,
        ),
        observation(
            0.03,
            value_a=1.000,
            value_b=2.000,
            sequence_index=3,
        ),
    ]


def test_change_detection_requires_nonnegative_score():
    with pytest.raises(ValueError):
        ChangeDetection(
            sequence_index=0,
            timestamp=0.0,
            deviation_score=-1.0,
            changed=False,
        )


def test_detector_rejects_nonpositive_threshold():
    with pytest.raises(ValueError):
        FamiliarStateChangeDetector(
            threshold=0.0
        )


def test_detector_rejects_nonpositive_scale_floor():
    with pytest.raises(ValueError):
        FamiliarStateChangeDetector(
            absolute_scale_floor=0.0
        )


def test_detector_is_initially_unfitted():
    detector = FamiliarStateChangeDetector()

    assert detector.fitted is False


def test_detector_requires_at_least_two_baseline_observations():
    detector = FamiliarStateChangeDetector()

    with pytest.raises(ValueError):
        detector.fit(
            familiar_baseline()[:1]
        )


def test_detector_becomes_fitted_after_baseline():
    detector = FamiliarStateChangeDetector()

    detector.fit(
        familiar_baseline()
    )

    assert detector.fitted is True
    assert detector.channel_names == (
        "a",
        "b",
    )


def test_detector_requires_fit_before_scoring():
    detector = FamiliarStateChangeDetector()

    with pytest.raises(RuntimeError):
        detector.score(
            familiar_baseline()[0]
        )


def test_familiar_observation_has_low_deviation():
    detector = FamiliarStateChangeDetector(
        threshold=8.0
    )

    detector.fit(
        familiar_baseline()
    )

    result = detector.detect(
        observation(
            0.04,
            value_a=1.000,
            value_b=2.000,
            sequence_index=4,
        )
    )

    assert result.changed is False
    assert result.deviation_score < detector.threshold


def test_large_departure_is_detected_as_change():
    detector = FamiliarStateChangeDetector(
        threshold=8.0
    )

    detector.fit(
        familiar_baseline()
    )

    result = detector.detect(
        observation(
            0.04,
            value_a=1.1,
            value_b=2.0,
            sequence_index=4,
        )
    )

    assert result.changed is True
    assert result.deviation_score >= detector.threshold


def test_detection_preserves_observation_time_and_sequence():
    detector = FamiliarStateChangeDetector()

    detector.fit(
        familiar_baseline()
    )

    result = detector.detect(
        observation(
            0.25,
            value_a=1.0,
            value_b=2.0,
            sequence_index=25,
        )
    )

    assert result.timestamp == pytest.approx(
        0.25
    )
    assert result.sequence_index == 25


def test_detector_rejects_schema_change():
    detector = FamiliarStateChangeDetector()

    detector.fit(
        familiar_baseline()
    )

    changed_schema = Observation(
        timestamp=0.04,
        channels={
            "different": ObservationChannel(
                name="different",
                value=1.0,
            )
        },
        provenance=ObservationProvenance(
            source_id="body_0",
            source_type="proprioceptive_body",
            sequence_index=4,
        ),
    )

    with pytest.raises(ValueError):
        detector.detect(
            changed_schema
        )


def test_constant_baseline_is_supported_by_scale_floor():
    detector = FamiliarStateChangeDetector(
        threshold=8.0,
        absolute_scale_floor=1e-9,
    )

    baseline = [
        observation(
            0.00,
            value_a=1.0,
            value_b=2.0,
            sequence_index=0,
        ),
        observation(
            0.01,
            value_a=1.0,
            value_b=2.0,
            sequence_index=1,
        ),
        observation(
            0.02,
            value_a=1.0,
            value_b=2.0,
            sequence_index=2,
        ),
    ]

    detector.fit(
        baseline
    )

    unchanged = detector.detect(
        observation(
            0.03,
            value_a=1.0,
            value_b=2.0,
            sequence_index=3,
        )
    )

    assert unchanged.changed is False
    assert unchanged.deviation_score == pytest.approx(
        0.0
    )


def test_detector_has_no_ground_truth_interface():
    detector = FamiliarStateChangeDetector()

    assert not hasattr(
        detector,
        "ground_truth",
    )
    assert not hasattr(
        detector,
        "world_state",
    )
    assert not hasattr(
        detector,
        "perturbation_time",
    )
    assert not hasattr(
        detector,
        "external_force",
    )


def test_detector_has_no_reward_goal_or_event_interface():
    detector = FamiliarStateChangeDetector()

    assert not hasattr(
        detector,
        "reward",
    )
    assert not hasattr(
        detector,
        "goal",
    )
    assert not hasattr(
        detector,
        "event",
    )
    assert not hasattr(
        detector,
        "event_type",
    )
    assert not hasattr(
        detector,
        "valence",
    )
