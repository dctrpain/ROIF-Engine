import pytest

from roif.development.body_interface import (
    BodyGroundTruth,
    BodyInterface,
)
from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)


class FakeBody(BodyInterface):
    def __init__(self) -> None:
        self._timestamp = 1.0

    @property
    def body_id(self) -> str:
        return "fake_body"

    def observe(self) -> Observation:
        return Observation(
            timestamp=self._timestamp,
            channels={
                "tension_0": ObservationChannel(
                    name="tension_0",
                    value=2.0,
                    unit="N",
                ),
                "strain_0": ObservationChannel(
                    name="strain_0",
                    value=0.1,
                ),
            },
            provenance=ObservationProvenance(
                source_id=self.body_id,
                source_type="proprioceptive_body",
                sequence_index=0,
            ),
        )

    def ground_truth(self) -> BodyGroundTruth:
        return BodyGroundTruth(
            timestamp=self._timestamp,
            values={
                "world_x": 10.0,
                "world_y": 20.0,
                "world_z": 30.0,
                "external_force_x": 5.0,
                "tension_0": 2.0,
                "strain_0": 0.1,
            },
            metadata={
                "simulator": "fake",
            },
        )


def test_ground_truth_accepts_complete_physical_state():
    state = BodyGroundTruth(
        timestamp=0.0,
        values={
            "world_x": 1.0,
            "world_y": 2.0,
            "world_z": 3.0,
        },
        metadata={},
    )

    assert state.values["world_x"] == 1.0
    assert state.values["world_y"] == 2.0
    assert state.values["world_z"] == 3.0


def test_ground_truth_rejects_nonfinite_timestamp():
    with pytest.raises(ValueError):
        BodyGroundTruth(
            timestamp=float("nan"),
            values={"world_x": 1.0},
            metadata={},
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_ground_truth_rejects_nonfinite_float_values(value):
    with pytest.raises(ValueError):
        BodyGroundTruth(
            timestamp=0.0,
            values={"world_x": value},
            metadata={},
        )


def test_ground_truth_requires_values():
    with pytest.raises(ValueError):
        BodyGroundTruth(
            timestamp=0.0,
            values={},
            metadata={},
        )


def test_ground_truth_mapping_is_immutable():
    state = BodyGroundTruth(
        timestamp=0.0,
        values={"world_x": 1.0},
        metadata={},
    )

    with pytest.raises(TypeError):
        state.values["world_x"] = 2.0


def test_ground_truth_metadata_is_immutable():
    state = BodyGroundTruth(
        timestamp=0.0,
        values={"world_x": 1.0},
        metadata={"simulator": "fake"},
    )

    with pytest.raises(TypeError):
        state.metadata["simulator"] = "changed"


def test_body_interface_exposes_observation():
    body = FakeBody()

    observation = body.observe()

    assert isinstance(observation, Observation)
    assert observation.get("tension_0").value == 2.0
    assert observation.get("strain_0").value == 0.1


def test_body_interface_exposes_ground_truth_separately():
    body = FakeBody()

    state = body.ground_truth()

    assert state.values["world_x"] == 10.0
    assert state.values["world_y"] == 20.0
    assert state.values["world_z"] == 30.0


def test_observation_does_not_contain_hidden_world_coordinates():
    body = FakeBody()

    observation = body.observe()

    assert "world_x" not in observation.channels
    assert "world_y" not in observation.channels
    assert "world_z" not in observation.channels


def test_observation_does_not_contain_external_force_ground_truth():
    body = FakeBody()

    observation = body.observe()

    assert "external_force_x" not in observation.channels


def test_ground_truth_and_observation_are_not_same_representation():
    body = FakeBody()

    observation = body.observe()
    state = body.ground_truth()

    assert set(observation.channels) != set(state.values)


def test_body_identity_is_stable():
    body = FakeBody()

    assert body.body_id == "fake_body"
    assert body.body_id == "fake_body"