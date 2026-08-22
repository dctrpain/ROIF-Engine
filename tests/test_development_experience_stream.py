import pytest

from roif.development.experience_stream import (
    ExperienceSample,
    ExperienceStream,
)
from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)


def make_observation(
    timestamp: float,
    *,
    body_id: str = "body_0",
    sequence_index: int = 0,
    value: float = 1.0,
) -> Observation:
    return Observation(
        timestamp=timestamp,
        channels={
            "tension_0": ObservationChannel(
                name="tension_0",
                value=value,
                unit="N",
            )
        },
        provenance=ObservationProvenance(
            source_id=body_id,
            source_type="proprioceptive_body",
            sequence_index=sequence_index,
        ),
    )


def test_experience_sample_rejects_negative_index():
    with pytest.raises(ValueError):
        ExperienceSample(
            index=-1,
            observation=make_observation(0.0),
        )


def test_stream_requires_body_identity():
    with pytest.raises(ValueError):
        ExperienceStream("")


def test_empty_stream_has_no_first_or_latest_sample():
    stream = ExperienceStream("body_0")

    assert len(stream) == 0
    assert stream.first is None
    assert stream.latest is None
    assert stream.duration == 0.0


def test_first_observation_can_be_appended():
    stream = ExperienceStream("body_0")

    sample = stream.append(
        make_observation(0.0)
    )

    assert sample.index == 0
    assert len(stream) == 1
    assert stream.first is sample
    assert stream.latest is sample


def test_samples_receive_monotonic_indices():
    stream = ExperienceStream("body_0")

    first = stream.append(
        make_observation(
            0.0,
            sequence_index=0,
        )
    )
    second = stream.append(
        make_observation(
            0.1,
            sequence_index=1,
        )
    )

    assert first.index == 0
    assert second.index == 1


def test_stream_requires_strictly_increasing_time():
    stream = ExperienceStream("body_0")

    stream.append(make_observation(1.0))

    with pytest.raises(ValueError):
        stream.append(make_observation(1.0))

    with pytest.raises(ValueError):
        stream.append(make_observation(0.5))


def test_stream_rejects_observation_from_other_body():
    stream = ExperienceStream("body_0")

    with pytest.raises(ValueError):
        stream.append(
            make_observation(
                0.0,
                body_id="body_1",
            )
        )


def test_stream_preserves_observation_order():
    stream = ExperienceStream("body_0")

    stream.append(
        make_observation(
            0.0,
            value=1.0,
        )
    )
    stream.append(
        make_observation(
            0.1,
            value=2.0,
        )
    )
    stream.append(
        make_observation(
            0.2,
            value=3.0,
        )
    )

    values = [
        sample.observation.get("tension_0").value
        for sample in stream
    ]

    assert values == [1.0, 2.0, 3.0]


def test_duration_is_elapsed_experienced_time():
    stream = ExperienceStream("body_0")

    stream.append(make_observation(2.0))
    stream.append(make_observation(2.25))
    stream.append(make_observation(3.0))

    assert stream.duration == pytest.approx(1.0)


def test_snapshot_is_immutable_tuple():
    stream = ExperienceStream("body_0")

    stream.append(make_observation(0.0))
    stream.append(make_observation(0.1))

    snapshot = stream.snapshot()

    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 2


def test_stream_does_not_create_event_labels():
    stream = ExperienceStream("body_0")

    stream.append(make_observation(0.0))

    assert not hasattr(stream.latest, "event")
    assert not hasattr(stream.latest, "event_type")


def test_stream_does_not_create_novelty_labels():
    stream = ExperienceStream("body_0")

    stream.append(make_observation(0.0))

    assert not hasattr(stream.latest, "novelty")
    assert not hasattr(stream.latest, "is_novel")


def test_stream_does_not_create_reward_or_valence():
    stream = ExperienceStream("body_0")

    stream.append(make_observation(0.0))

    assert not hasattr(stream.latest, "reward")
    assert not hasattr(stream.latest, "valence")


def test_stream_does_not_decide_what_becomes_memory():
    stream = ExperienceStream("body_0")

    stream.append(make_observation(0.0))

    assert not hasattr(stream.latest, "memory")
    assert not hasattr(stream.latest, "remember")
    assert not hasattr(stream, "structured_memory")