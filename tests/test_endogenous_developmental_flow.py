from __future__ import annotations

from roif.development.developmental_activation import (
    DevelopmentalActivationInspector,
)
from roif.development.developmental_affordance import (
    DevelopmentalCapability,
)
from roif.development.developmental_process import (
    EndogenousDifferenceProcess,
    EndogenousGroupingProcess,
)
from roif.development.developmental_state import (
    DevelopmentalState,
)
from roif.development.endogenous_difference import (
    DifferenceProfile,
)
from roif.development.endogenous_grouping import (
    GroupingResult,
)
from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)


def _observation(
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
            source_id="endogenous-flow-test",
            source_type="synthetic-test",
            sequence_index=sequence_index,
        ),
    )


def test_endogenous_flow_produces_grouping_from_observation_memory():
    state = DevelopmentalState()

    activation = DevelopmentalActivationInspector()
    difference_process = EndogenousDifferenceProcess()
    grouping_process = EndogenousGroupingProcess()

    # ---------------------------------------------------------
    # Initial state.
    #
    # No developmental computation is yet activatable.
    # ---------------------------------------------------------

    assert (
        activation.activatable_capabilities(state)
        == ()
    )

    assert state.observation_count == 0
    assert state.difference_profile_count == 0
    assert state.grouping_result_count == 0

    # ---------------------------------------------------------
    # Accumulate prior experience plus one current observation.
    #
    # No stage is advanced.
    # The state simply acquires experience.
    # ---------------------------------------------------------

    state.record_observation(
        _observation(
            sequence_index=0,
            a=0.0,
            b=0.0,
        )
    )

    state.record_observation(
        _observation(
            sequence_index=1,
            a=1.0,
            b=2.0,
        )
    )

    state.record_observation(
        _observation(
            sequence_index=2,
            a=2.0,
            b=4.0,
        )
    )

    active_after_three = set(
        activation.activatable_capabilities(state)
    )

    assert (
        DevelopmentalCapability
        .ANALYZE_ENDOGENOUS_DIFFERENCE
        in active_after_three
    )

    assert (
        DevelopmentalCapability
        .DISCOVER_ENDOGENOUS_GROUPING
        not in active_after_three
    )

    # ---------------------------------------------------------
    # First real endogenous difference production.
    # ---------------------------------------------------------

    first_difference = difference_process.run(state)

    assert first_difference.executed is True

    assert isinstance(
        first_difference.produced_object,
        DifferenceProfile,
    )

    assert state.difference_profile_count == 1

    assert (
        state.latest_difference_profile
        is first_difference.produced_object
    )

    active_after_first_profile = set(
        activation.activatable_capabilities(state)
    )

    assert (
        DevelopmentalCapability
        .DISCOVER_ENDOGENOUS_GROUPING
        not in active_after_first_profile
    )

    assert grouping_process.can_run(state) is False

    # ---------------------------------------------------------
    # More experience arrives.
    #
    # Again, no "next stage" is invoked.
    # ---------------------------------------------------------

    state.record_observation(
        _observation(
            sequence_index=3,
            a=3.0,
            b=6.0,
        )
    )

    # ---------------------------------------------------------
    # Second real endogenous difference production.
    # ---------------------------------------------------------

    second_difference = difference_process.run(state)

    assert second_difference.executed is True

    assert isinstance(
        second_difference.produced_object,
        DifferenceProfile,
    )

    assert state.difference_profile_count == 2

    assert (
        second_difference.produced_object
        is not first_difference.produced_object
    )

    # ---------------------------------------------------------
    # The state has changed.
    #
    # Grouping is now activatable solely because sufficient
    # endogenous difference memory exists.
    # ---------------------------------------------------------

    active_after_second_profile = set(
        activation.activatable_capabilities(state)
    )

    assert (
        DevelopmentalCapability
        .DISCOVER_ENDOGENOUS_GROUPING
        in active_after_second_profile
    )

    assert grouping_process.can_run(state) is True

    # ---------------------------------------------------------
    # Real endogenous grouping production.
    # ---------------------------------------------------------

    grouping = grouping_process.run(state)

    assert grouping.executed is True

    assert isinstance(
        grouping.produced_object,
        GroupingResult,
    )

    assert state.grouping_result_count == 1

    assert (
        state.latest_grouping_result
        is grouping.produced_object
    )

    assert grouping.produced_object.channel_names == (
        "a",
        "b",
    )

    # Both channels changed together across experience.
    assert len(grouping.produced_object.groups) >= 1

    # ---------------------------------------------------------
    # Source memory remains present.
    # Development does not consume its own history.
    # ---------------------------------------------------------

    assert state.observation_count == 4
    assert state.difference_profile_count == 2
    assert state.grouping_result_count == 1

    # ---------------------------------------------------------
    # Architectural invariant.
    #
    # Nothing in the flow exposes a prescribed developmental
    # successor or RDA transition controller.
    # ---------------------------------------------------------

    objects = (
        state,
        activation,
        difference_process,
        grouping_process,
    )

    forbidden = (
        "current_stage",
        "next_stage",
        "expected_stage",
        "next_process",
        "transition",
        "transition_map",
        "advance",
        "advance_to",
        "run_rda0",
        "run_rda1",
        "run_rda2",
        "run_rda3",
    )

    for obj in objects:
        for name in forbidden:
            assert not hasattr(obj, name)


def test_grouping_activation_emerges_from_state_not_process_linkage():
    state = DevelopmentalState()

    difference_process = EndogenousDifferenceProcess()
    grouping_process = EndogenousGroupingProcess()

    # The processes have no direct developmental linkage.
    assert not hasattr(
        difference_process,
        "next_process",
    )
    assert not hasattr(
        grouping_process,
        "previous_process",
    )

    state.record_observation(
        _observation(0, 0.0, 0.0)
    )
    state.record_observation(
        _observation(1, 1.0, 2.0)
    )
    state.record_observation(
        _observation(2, 2.0, 4.0)
    )

    first = difference_process.run(state)

    assert first.executed is True
    assert grouping_process.can_run(state) is False

    state.record_observation(
        _observation(3, 3.0, 6.0)
    )

    second = difference_process.run(state)

    assert second.executed is True

    # No process told grouping to become active.
    #
    # Its eligibility emerged because shared memory now contains
    # enough DifferenceProfile evidence.
    assert grouping_process.can_run(state) is True
