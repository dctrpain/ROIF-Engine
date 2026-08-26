from __future__ import annotations

from roif.development.developmental_activation import (
    DevelopmentalActivationInspector,
)
from roif.development.developmental_affordance import (
    DevelopmentalCapability,
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
            source_id="developmental-test",
            source_type="synthetic-test",
            sequence_index=sequence_index,
        ),
    )


def _difference_profile(
    sequence_index: int,
    signed_deviation: float,
) -> DifferenceProfile:
    return DifferenceProfile(
        timestamp=float(sequence_index),
        sequence_index=sequence_index,
        channel_deviations=(
            ChannelDeviation(
                channel_name="sensor",
                current_value=signed_deviation,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=signed_deviation,
                absolute_deviation=abs(signed_deviation),
            ),
        ),
    )


def _activatable(state: DevelopmentalState):
    inspector = DevelopmentalActivationInspector()

    return set(
        inspector.activatable_capabilities(state)
    )


def test_three_observations_activate_two_capabilities_together():
    state = DevelopmentalState()

    state.record_observation(_observation(0, 0.0))
    state.record_observation(_observation(1, 0.1))
    state.record_observation(_observation(2, 1.0))

    active = _activatable(state)

    assert (
        DevelopmentalCapability.DETECT_FAMILIAR_STATE_CHANGE
        in active
    )
    assert (
        DevelopmentalCapability.ANALYZE_ENDOGENOUS_DIFFERENCE
        in active
    )

    assert len(active) == 2


def test_difference_profiles_activate_grouping_independently():
    state = DevelopmentalState()

    state.record_difference_profile(
        _difference_profile(0, 1.0)
    )
    state.record_difference_profile(
        _difference_profile(1, -1.0)
    )

    active = _activatable(state)

    assert (
        DevelopmentalCapability.DISCOVER_ENDOGENOUS_GROUPING
        in active
    )

    assert (
        DevelopmentalCapability.DETECT_FAMILIAR_STATE_CHANGE
        not in active
    )
    assert (
        DevelopmentalCapability.ANALYZE_ENDOGENOUS_DIFFERENCE
        not in active
    )


def test_observations_and_profiles_enable_multiple_capabilities():
    state = DevelopmentalState()

    state.record_observation(_observation(0, 0.0))
    state.record_observation(_observation(1, 0.1))
    state.record_observation(_observation(2, 1.0))

    state.record_difference_profile(
        _difference_profile(0, 1.0)
    )
    state.record_difference_profile(
        _difference_profile(1, -1.0)
    )

    active = _activatable(state)

    assert active == {
        DevelopmentalCapability.DETECT_FAMILIAR_STATE_CHANGE,
        DevelopmentalCapability.ANALYZE_ENDOGENOUS_DIFFERENCE,
        DevelopmentalCapability.DISCOVER_ENDOGENOUS_GROUPING,
    }


def test_dimensional_growth_does_not_activate_from_counts_alone():
    state = DevelopmentalState()

    for index in range(10):
        state.record_observation(
            _observation(
                index,
                float(index),
            )
        )

    for index in range(10):
        state.record_difference_profile(
            _difference_profile(
                index,
                float(index + 1),
            )
        )

    active = _activatable(state)

    assert (
        DevelopmentalCapability.ASSESS_DIMENSIONAL_GROWTH
        not in active
    )


def test_activation_exposes_no_next_capability():
    inspector = DevelopmentalActivationInspector()

    forbidden = {
        "current_stage",
        "next_stage",
        "next_capability",
        "advance",
        "advance_to",
        "transition",
        "transition_map",
        "schedule",
        "execute",
    }

    for name in forbidden:
        assert not hasattr(inspector, name)
