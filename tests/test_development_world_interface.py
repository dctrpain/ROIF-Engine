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
from roif.worlds.world_interface import (
    WorldInterface,
    WorldState,
)


class FakeBody(BodyInterface):
    def __init__(self) -> None:
        self._time = 0.0

    @property
    def body_id(self) -> str:
        return "body_0"

    def set_time(self, value: float) -> None:
        self._time = value

    def observe(self) -> Observation:
        return Observation(
            timestamp=self._time,
            channels={
                "tension_0": ObservationChannel(
                    name="tension_0",
                    value=2.0,
                    unit="N",
                )
            },
            provenance=ObservationProvenance(
                source_id=self.body_id,
                source_type="proprioceptive_body",
                sequence_index=0,
            ),
        )

    def ground_truth(self) -> BodyGroundTruth:
        return BodyGroundTruth(
            timestamp=self._time,
            values={
                "world_x": 1.0,
                "world_y": 2.0,
                "world_z": 3.0,
                "tension_0": 2.0,
            },
            metadata={
                "body": self.body_id,
            },
        )


class FakeWorld(WorldInterface):
    def __init__(self) -> None:
        self._time = 0.0
        self._body = FakeBody()

    @property
    def world_id(self) -> str:
        return "fake_world"

    @property
    def time(self) -> float:
        return self._time

    def step(self, dt: float) -> None:
        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be numeric.")

        dt = float(dt)

        if not dt > 0.0:
            raise ValueError("dt must be > 0.")

        self._time += dt
        self._body.set_time(self._time)

    def world_state(self) -> WorldState:
        return WorldState(
            timestamp=self._time,
            values={
                "hidden_world_variable": 42.0,
                "body_0_world_x": 1.0,
                "body_0_world_y": 2.0,
                "body_0_world_z": 3.0,
            },
            metadata={
                "simulator": "fake",
            },
        )

    def bodies(self) -> tuple[BodyInterface, ...]:
        return (self._body,)


def test_world_state_accepts_ground_truth():
    state = WorldState(
        timestamp=0.0,
        values={
            "hidden_q": 3.0,
        },
        metadata={},
    )

    assert state.values["hidden_q"] == 3.0


def test_world_state_rejects_empty_values():
    with pytest.raises(ValueError):
        WorldState(
            timestamp=0.0,
            values={},
            metadata={},
        )


def test_world_state_rejects_nonfinite_timestamp():
    with pytest.raises(ValueError):
        WorldState(
            timestamp=float("nan"),
            values={"x": 1.0},
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
def test_world_state_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        WorldState(
            timestamp=0.0,
            values={"x": value},
            metadata={},
        )


def test_world_state_values_are_immutable():
    state = WorldState(
        timestamp=0.0,
        values={"x": 1.0},
        metadata={},
    )

    with pytest.raises(TypeError):
        state.values["x"] = 2.0


def test_world_state_metadata_is_immutable():
    state = WorldState(
        timestamp=0.0,
        values={"x": 1.0},
        metadata={"simulator": "fake"},
    )

    with pytest.raises(TypeError):
        state.metadata["simulator"] = "changed"


def test_world_has_stable_identity():
    world = FakeWorld()

    assert world.world_id == "fake_world"
    assert world.world_id == "fake_world"


def test_world_time_starts_at_zero():
    world = FakeWorld()

    assert world.time == 0.0


def test_world_step_advances_time():
    world = FakeWorld()

    world.step(0.01)

    assert world.time == pytest.approx(0.01)


def test_world_step_rejects_nonpositive_dt():
    world = FakeWorld()

    with pytest.raises(ValueError):
        world.step(0.0)

    with pytest.raises(ValueError):
        world.step(-0.01)


def test_body_time_tracks_world_time():
    world = FakeWorld()

    world.step(0.125)

    observation = world.body("body_0").observe()

    assert observation.timestamp == pytest.approx(0.125)


def test_world_exposes_body_interface():
    world = FakeWorld()

    body = world.body("body_0")

    assert isinstance(body, BodyInterface)
    assert body.body_id == "body_0"


def test_unknown_body_id_raises_key_error():
    world = FakeWorld()

    with pytest.raises(KeyError):
        world.body("missing_body")


def test_world_ground_truth_contains_hidden_information():
    world = FakeWorld()

    state = world.world_state()

    assert "hidden_world_variable" in state.values


def test_body_observation_does_not_receive_world_hidden_information():
    world = FakeWorld()

    observation = world.body("body_0").observe()

    assert "hidden_world_variable" not in observation.channels


def test_world_body_ground_truth_and_observation_remain_distinct():
    world = FakeWorld()

    world_state = world.world_state()
    body_ground_truth = world.body("body_0").ground_truth()
    observation = world.body("body_0").observe()

    assert set(world_state.values) != set(body_ground_truth.values)
    assert set(body_ground_truth.values) != set(observation.channels)


def test_world_contains_no_required_reward_interface():
    world = FakeWorld()

    assert not hasattr(world, "reward")


def test_world_contains_no_required_goal_interface():
    world = FakeWorld()

    assert not hasattr(world, "goal")