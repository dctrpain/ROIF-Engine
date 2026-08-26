from __future__ import annotations

import pytest

from roif.development.developmental_affordance import (
    DevelopmentalCapability,
)
from roif.development.developmental_process import (
    EndogenousDifferenceProcess,
    EndogenousGroupingProcess,
    FamiliarStateChangeProcess,
)
from roif.development.developmental_process_set import (
    DevelopmentalProcessSet,
)
from roif.development.developmental_state import (
    DevelopmentalState,
)
from roif.development.endogenous_difference import (
    ChannelDeviation,
    DifferenceProfile,
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
            source_id="process-set-test",
            source_type="synthetic-test",
            sequence_index=sequence_index,
        ),
    )


def _profile(
    sequence_index: int,
    deviation: float,
) -> DifferenceProfile:
    return DifferenceProfile(
        timestamp=float(sequence_index),
        sequence_index=sequence_index,
        channel_deviations=(
            ChannelDeviation(
                channel_name="sensor",
                current_value=deviation,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=deviation,
                absolute_deviation=abs(deviation),
            ),
        ),
    )


def _process_set() -> DevelopmentalProcessSet:
    return DevelopmentalProcessSet(
        (
            FamiliarStateChangeProcess(),
            EndogenousDifferenceProcess(),
            EndogenousGroupingProcess(),
        )
    )


def test_process_set_exposes_capabilities_as_frozenset():
    processes = _process_set()

    capabilities = processes.capabilities

    assert isinstance(
        capabilities,
        frozenset,
    )

    assert capabilities == frozenset(
        {
            DevelopmentalCapability
            .DETECT_FAMILIAR_STATE_CHANGE,
            DevelopmentalCapability
            .ANALYZE_ENDOGENOUS_DIFFERENCE,
            DevelopmentalCapability
            .DISCOVER_ENDOGENOUS_GROUPING,
        }
    )


def test_empty_state_has_no_runnable_capabilities():
    state = DevelopmentalState()
    processes = _process_set()

    assert (
        processes.runnable_capabilities(state)
        == frozenset()
    )


def test_same_state_can_make_two_processes_runnable():
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

    processes = _process_set()

    runnable = processes.runnable_capabilities(
        state
    )

    assert runnable == frozenset(
        {
            DevelopmentalCapability
            .DETECT_FAMILIAR_STATE_CHANGE,
            DevelopmentalCapability
            .ANALYZE_ENDOGENOUS_DIFFERENCE,
        }
    )


def test_grouping_joins_runnable_set_from_shared_memory():
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

    state.record_difference_profile(
        _profile(0, 1.0)
    )
    state.record_difference_profile(
        _profile(1, 2.0)
    )

    processes = _process_set()

    runnable = processes.runnable_capabilities(
        state
    )

    assert runnable == frozenset(
        {
            DevelopmentalCapability
            .DETECT_FAMILIAR_STATE_CHANGE,
            DevelopmentalCapability
            .ANALYZE_ENDOGENOUS_DIFFERENCE,
            DevelopmentalCapability
            .DISCOVER_ENDOGENOUS_GROUPING,
        }
    )


def test_registration_order_does_not_change_runnable_set():
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

    first = DevelopmentalProcessSet(
        (
            FamiliarStateChangeProcess(),
            EndogenousDifferenceProcess(),
            EndogenousGroupingProcess(),
        )
    )

    second = DevelopmentalProcessSet(
        (
            EndogenousGroupingProcess(),
            EndogenousDifferenceProcess(),
            FamiliarStateChangeProcess(),
        )
    )

    assert (
        first.runnable_capabilities(state)
        == second.runnable_capabilities(state)
    )


def test_duplicate_capability_is_rejected():
    with pytest.raises(
        ValueError,
        match="Duplicate developmental capability",
    ):
        DevelopmentalProcessSet(
            (
                EndogenousDifferenceProcess(),
                EndogenousDifferenceProcess(),
            )
        )


def test_process_set_has_no_scheduler_interface():
    processes = _process_set()

    forbidden = (
        "run",
        "run_all",
        "execute",
        "execute_all",
        "schedule",
        "queue",
        "priority",
        "first",
        "next",
        "next_process",
        "current_stage",
        "next_stage",
        "advance",
        "transition",
        "transition_map",
    )

    for name in forbidden:
        assert not hasattr(
            processes,
            name,
        )
