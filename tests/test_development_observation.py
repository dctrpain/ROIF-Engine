import pytest

from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)


def _provenance() -> ObservationProvenance:
    return ObservationProvenance(
        source_id="body_0",
        source_type="proprioceptive_body",
        sequence_index=0,
    )


def test_observation_channel_accepts_finite_numeric_value():
    channel = ObservationChannel(
        name="tension_0",
        value=1.25,
        unit="N",
    )

    assert channel.name == "tension_0"
    assert channel.value == 1.25
    assert channel.unit == "N"
    assert channel.available is True


def test_observation_channel_rejects_empty_name():
    with pytest.raises(ValueError):
        ObservationChannel(
            name="",
            value=1.0,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_observation_channel_rejects_nonfinite_float(value):
    with pytest.raises(ValueError):
        ObservationChannel(
            name="tension_0",
            value=value,
        )


def test_observation_channel_quality_must_be_unit_interval():
    with pytest.raises(ValueError):
        ObservationChannel(
            name="strain_0",
            value=0.1,
            quality=1.1,
        )


def test_provenance_rejects_negative_sequence_index():
    with pytest.raises(ValueError):
        ObservationProvenance(
            source_id="body_0",
            source_type="proprioceptive_body",
            sequence_index=-1,
        )


def test_observation_requires_channels():
    with pytest.raises(ValueError):
        Observation(
            timestamp=0.0,
            channels={},
            provenance=_provenance(),
        )


def test_observation_rejects_channel_key_name_mismatch():
    with pytest.raises(ValueError):
        Observation(
            timestamp=0.0,
            channels={
                "tension_1": ObservationChannel(
                    name="tension_0",
                    value=1.0,
                )
            },
            provenance=_provenance(),
        )


def test_observation_is_immutable_at_mapping_level():
    observation = Observation(
        timestamp=0.0,
        channels={
            "tension_0": ObservationChannel(
                name="tension_0",
                value=1.0,
            )
        },
        provenance=_provenance(),
    )

    with pytest.raises(TypeError):
        observation.channels["tension_1"] = ObservationChannel(
            name="tension_1",
            value=2.0,
        )


def test_numeric_vector_is_deterministic_by_channel_name():
    observation = Observation(
        timestamp=0.0,
        channels={
            "strain_1": ObservationChannel(
                name="strain_1",
                value=0.4,
            ),
            "strain_0": ObservationChannel(
                name="strain_0",
                value=0.2,
            ),
        },
        provenance=_provenance(),
    )

    assert observation.numeric_vector() == (0.2, 0.4)


def test_numeric_vector_excludes_unavailable_by_default():
    observation = Observation(
        timestamp=0.0,
        channels={
            "a": ObservationChannel(
                name="a",
                value=1.0,
                available=True,
            ),
            "b": ObservationChannel(
                name="b",
                value=2.0,
                available=False,
            ),
        },
        provenance=_provenance(),
    )

    assert observation.numeric_vector() == (1.0,)
    assert observation.numeric_vector(
        include_unavailable=True
    ) == (1.0, 2.0)


def test_boolean_channel_is_encoded_without_semantic_interpretation():
    observation = Observation(
        timestamp=0.0,
        channels={
            "contact_0": ObservationChannel(
                name="contact_0",
                value=True,
            )
        },
        provenance=_provenance(),
    )

    assert observation.numeric_vector() == (1.0,)


def test_available_and_unavailable_channels_are_distinguished():
    observation = Observation(
        timestamp=0.0,
        channels={
            "a": ObservationChannel(
                name="a",
                value=1.0,
                available=True,
            ),
            "b": ObservationChannel(
                name="b",
                value=2.0,
                available=False,
            ),
        },
        provenance=_provenance(),
    )

    assert tuple(
        channel.name
        for channel in observation.available_channels()
    ) == ("a",)

    assert tuple(
        channel.name
        for channel in observation.unavailable_channels()
    ) == ("b",)


def test_observation_metadata_is_immutable():
    observation = Observation(
        timestamp=0.0,
        channels={
            "a": ObservationChannel(
                name="a",
                value=1.0,
            )
        },
        provenance=_provenance(),
        metadata={
            "sampling_mode": "continuous",
        },
    )

    with pytest.raises(TypeError):
        observation.metadata["sampling_mode"] = "changed"