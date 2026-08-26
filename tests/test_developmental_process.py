from __future__ import annotations

from roif.development.developmental_affordance import (
    DevelopmentalCapability,
)
from roif.development.developmental_process import (
    DevelopmentalProcessResult,
    EndogenousGroupingProcess,
)
from roif.development.developmental_state import (
    DevelopmentalState,
)
from roif.development.endogenous_difference import (
    ChannelDeviation,
    DifferenceProfile,
)
from roif.development.endogenous_grouping import (
    GroupingResult,
)


def _profile(
    sequence_index: int,
    a: float,
    b: float,
) -> DifferenceProfile:
    return DifferenceProfile(
        timestamp=float(sequence_index),
        sequence_index=sequence_index,
        channel_deviations=(
            ChannelDeviation(
                channel_name="a",
                current_value=a,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=a,
                absolute_deviation=abs(a),
            ),
            ChannelDeviation(
                channel_name="b",
                current_value=b,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=b,
                absolute_deviation=abs(b),
            ),
        ),
    )


def test_grouping_process_does_not_run_without_affordance():
    state = DevelopmentalState()
    process = EndogenousGroupingProcess()

    result = process.run(state)

    assert isinstance(
        result,
        DevelopmentalProcessResult,
    )

    assert result.executed is False
    assert result.produced_object is None

    assert (
        result.capability
        == DevelopmentalCapability
        .DISCOVER_ENDOGENOUS_GROUPING
    )

    assert state.grouping_result_count == 0


def test_grouping_process_produces_real_grouping_result():
    state = DevelopmentalState()

    state.record_difference_profile(
        _profile(0, 1.0, 1.0)
    )
    state.record_difference_profile(
        _profile(1, 2.0, 2.0)
    )
    state.record_difference_profile(
        _profile(2, 3.0, 3.0)
    )

    process = EndogenousGroupingProcess()

    assert process.can_run(state) is True

    result = process.run(state)

    assert result.executed is True

    assert isinstance(
        result.produced_object,
        GroupingResult,
    )

    assert state.grouping_result_count == 1

    assert (
        state.latest_grouping_result
        is result.produced_object
    )


def test_grouping_process_reads_shared_state_memory():
    state = DevelopmentalState()

    profiles = (
        _profile(0, 1.0, 1.0),
        _profile(1, 2.0, 2.0),
        _profile(2, 3.0, 3.0),
    )

    for profile in profiles:
        state.record_difference_profile(profile)

    process = EndogenousGroupingProcess()

    result = process.run(state)

    assert result.executed is True

    grouping = result.produced_object

    assert isinstance(grouping, GroupingResult)

    assert grouping.channel_names == (
        "a",
        "b",
    )


def test_grouping_process_has_no_stage_transition_interface():
    process = EndogenousGroupingProcess()

    forbidden = (
        "current_stage",
        "next_stage",
        "expected_stage",
        "transition",
        "transition_map",
        "advance",
        "advance_to",
        "run_rda0",
        "run_rda1",
        "run_rda2",
        "run_rda3",
    )

    for name in forbidden:
        assert not hasattr(process, name)


def test_process_execution_does_not_consume_source_memory():
    state = DevelopmentalState()

    profiles = (
        _profile(0, 1.0, 1.0),
        _profile(1, 2.0, 2.0),
        _profile(2, 3.0, 3.0),
    )

    for profile in profiles:
        state.record_difference_profile(profile)

    before = tuple(state.difference_profiles)

    process = EndogenousGroupingProcess()
    result = process.run(state)

    assert result.executed is True

    assert tuple(state.difference_profiles) == before
    assert state.difference_profile_count == 3
    assert state.grouping_result_count == 1


def test_difference_process_does_not_run_without_enough_observations():
    from roif.development.developmental_process import (
        EndogenousDifferenceProcess,
    )

    state = DevelopmentalState()
    process = EndogenousDifferenceProcess()

    result = process.run(state)

    assert result.executed is False
    assert result.produced_object is None
    assert state.difference_profile_count == 0


def test_difference_process_produces_real_difference_profile():
    from roif.development.developmental_process import (
        EndogenousDifferenceProcess,
    )
    from roif.development.endogenous_difference import (
        DifferenceProfile,
    )
    from roif.development.observation import (
        Observation,
        ObservationChannel,
        ObservationProvenance,
    )

    def observation(
        sequence_index: int,
        a: float,
        b: float,
    ) -> Observation:
        return Observation(
            timestamp=float(sequence_index),
            channels={
                "a": ObservationChannel(
                    name="a",
                    value=a,
                ),
                "b": ObservationChannel(
                    name="b",
                    value=b,
                ),
            },
            provenance=ObservationProvenance(
                source_id="difference-process-test",
                source_type="synthetic-test",
                sequence_index=sequence_index,
            ),
        )

    state = DevelopmentalState()

    state.record_observation(
        observation(0, 0.0, 0.0)
    )
    state.record_observation(
        observation(1, 0.1, 0.0)
    )
    state.record_observation(
        observation(2, 1.0, 0.5)
    )

    process = EndogenousDifferenceProcess()

    assert process.can_run(state) is True

    result = process.run(state)

    assert result.executed is True

    assert isinstance(
        result.produced_object,
        DifferenceProfile,
    )

    assert state.difference_profile_count == 1

    assert (
        state.latest_difference_profile
        is result.produced_object
    )

    assert (
        result.produced_object.sequence_index
        == 2
    )

    assert (
        result.produced_object.timestamp
        == 2.0
    )


def test_difference_process_does_not_consume_observation_memory():
    from roif.development.developmental_process import (
        EndogenousDifferenceProcess,
    )
    from roif.development.observation import (
        Observation,
        ObservationChannel,
        ObservationProvenance,
    )

    def observation(
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
                source_id="difference-memory-test",
                source_type="synthetic-test",
                sequence_index=sequence_index,
            ),
        )

    state = DevelopmentalState()

    state.record_observation(
        observation(0, 0.0)
    )
    state.record_observation(
        observation(1, 0.1)
    )
    state.record_observation(
        observation(2, 1.0)
    )

    before = tuple(
        state.observations
    )

    process = EndogenousDifferenceProcess()
    result = process.run(state)

    assert result.executed is True

    assert tuple(state.observations) == before
    assert state.observation_count == 3
    assert state.difference_profile_count == 1


def test_difference_and_grouping_processes_do_not_reference_each_other():
    from roif.development.developmental_process import (
        EndogenousDifferenceProcess,
    )

    difference = EndogenousDifferenceProcess()
    grouping = EndogenousGroupingProcess()

    forbidden = (
        "next_process",
        "previous_process",
        "successor",
        "predecessor",
        "transition",
        "advance",
        "schedule",
    )

    for name in forbidden:
        assert not hasattr(
            difference,
            name,
        )
        assert not hasattr(
            grouping,
            name,
        )


def test_familiar_change_process_does_not_run_without_enough_observations():
    from roif.development.developmental_process import (
        FamiliarStateChangeProcess,
    )

    state = DevelopmentalState()
    process = FamiliarStateChangeProcess()

    result = process.run(state)

    assert result.executed is False
    assert result.produced_object is None
    assert state.change_detection_count == 0


def test_familiar_change_process_produces_real_change_detection():
    from roif.development.developmental_process import (
        FamiliarStateChangeProcess,
    )
    from roif.development.familiar_state_change import (
        ChangeDetection,
    )
    from roif.development.observation import (
        Observation,
        ObservationChannel,
        ObservationProvenance,
    )

    def observation(
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
                source_id="familiar-change-test",
                source_type="synthetic-test",
                sequence_index=sequence_index,
            ),
        )

    state = DevelopmentalState()

    state.record_observation(
        observation(0, 0.0)
    )
    state.record_observation(
        observation(1, 0.1)
    )
    state.record_observation(
        observation(2, 10.0)
    )

    process = FamiliarStateChangeProcess()

    assert process.can_run(state) is True

    result = process.run(state)

    assert result.executed is True

    assert isinstance(
        result.produced_object,
        ChangeDetection,
    )

    assert state.change_detection_count == 1

    assert (
        state.latest_change_detection
        is result.produced_object
    )

    assert (
        result.produced_object.sequence_index
        == 2
    )

    assert (
        result.produced_object.timestamp
        == 2.0
    )


def test_familiar_change_and_difference_can_run_from_same_state():
    from roif.development.developmental_process import (
        EndogenousDifferenceProcess,
        FamiliarStateChangeProcess,
    )
    from roif.development.observation import (
        Observation,
        ObservationChannel,
        ObservationProvenance,
    )

    def observation(
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
                source_id="parallel-process-test",
                source_type="synthetic-test",
                sequence_index=sequence_index,
            ),
        )

    state = DevelopmentalState()

    state.record_observation(
        observation(0, 0.0)
    )
    state.record_observation(
        observation(1, 0.1)
    )
    state.record_observation(
        observation(2, 10.0)
    )

    familiar = FamiliarStateChangeProcess()
    difference = EndogenousDifferenceProcess()

    assert familiar.can_run(state) is True
    assert difference.can_run(state) is True

    familiar_result = familiar.run(state)
    difference_result = difference.run(state)

    assert familiar_result.executed is True
    assert difference_result.executed is True

    assert state.change_detection_count == 1
    assert state.difference_profile_count == 1


def test_parallel_processes_have_no_prescribed_order():
    from roif.development.developmental_process import (
        EndogenousDifferenceProcess,
        FamiliarStateChangeProcess,
    )

    familiar = FamiliarStateChangeProcess()
    difference = EndogenousDifferenceProcess()

    forbidden = (
        "priority",
        "order",
        "next_process",
        "previous_process",
        "successor",
        "predecessor",
        "transition",
        "advance",
        "schedule",
    )

    for name in forbidden:
        assert not hasattr(familiar, name)
        assert not hasattr(difference, name)
