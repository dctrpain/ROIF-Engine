from __future__ import annotations

from roif.development.developmental_process import (
    EndogenousDifferenceProcess,
    FamiliarStateChangeProcess,
)
from roif.development.developmental_state import (
    DevelopmentalState,
)
from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)


def _observation(
    sequence_index: int,
    value: float,
) -> Observation:
    return Observation(
        timestamp=float(sequence_index),
        channels={
            "sensor": ObservationChannel(
                name="sensor",
                value=value,
            ),
        },
        provenance=ObservationProvenance(
            source_id="order-invariance-test",
            source_type="synthetic-test",
            sequence_index=sequence_index,
        ),
    )


def _state() -> DevelopmentalState:
    state = DevelopmentalState()

    state.record_observation(
        _observation(0, 0.0)
    )
    state.record_observation(
        _observation(1, 0.1)
    )
    state.record_observation(
        _observation(2, 10.0)
    )

    return state


def test_familiar_and_difference_results_are_order_invariant():
    state_a = _state()
    state_b = _state()

    familiar_a = FamiliarStateChangeProcess()
    difference_a = EndogenousDifferenceProcess()

    familiar_b = FamiliarStateChangeProcess()
    difference_b = EndogenousDifferenceProcess()

    familiar_result_a = familiar_a.run(state_a)
    difference_result_a = difference_a.run(state_a)

    difference_result_b = difference_b.run(state_b)
    familiar_result_b = familiar_b.run(state_b)

    assert familiar_result_a.executed is True
    assert familiar_result_b.executed is True

    assert difference_result_a.executed is True
    assert difference_result_b.executed is True

    detection_a = familiar_result_a.produced_object
    detection_b = familiar_result_b.produced_object

    profile_a = difference_result_a.produced_object
    profile_b = difference_result_b.produced_object

    assert detection_a.sequence_index == detection_b.sequence_index
    assert detection_a.timestamp == detection_b.timestamp
    assert detection_a.changed == detection_b.changed
    assert detection_a.deviation_score == detection_b.deviation_score

    assert profile_a.sequence_index == profile_b.sequence_index
    assert profile_a.timestamp == profile_b.timestamp

    assert (
        profile_a.maximum_absolute_deviation
        == profile_b.maximum_absolute_deviation
    )

    assert (
        profile_a.l2_deviation_norm
        == profile_b.l2_deviation_norm
    )


def test_order_only_changes_recording_order_not_computed_content():
    state_a = _state()
    state_b = _state()

    familiar_a = FamiliarStateChangeProcess()
    difference_a = EndogenousDifferenceProcess()

    familiar_b = FamiliarStateChangeProcess()
    difference_b = EndogenousDifferenceProcess()

    familiar_a.run(state_a)
    difference_a.run(state_a)

    difference_b.run(state_b)
    familiar_b.run(state_b)

    assert state_a.change_detection_count == 1
    assert state_b.change_detection_count == 1

    assert state_a.difference_profile_count == 1
    assert state_b.difference_profile_count == 1

    assert (
        state_a.latest_change_detection.deviation_score
        == state_b.latest_change_detection.deviation_score
    )

    assert (
        state_a.latest_difference_profile.l2_deviation_norm
        == state_b.latest_difference_profile.l2_deviation_norm
    )


def test_parallel_processes_do_not_mutate_observation_memory():
    state = _state()

    before = tuple(state.observations)

    FamiliarStateChangeProcess().run(state)
    EndogenousDifferenceProcess().run(state)

    assert tuple(state.observations) == before
    assert state.observation_count == 3
