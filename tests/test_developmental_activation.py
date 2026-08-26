from __future__ import annotations

from roif.development.developmental_activation import (
    DevelopmentalActivation,
    DevelopmentalActivationInspector,
)
from roif.development.developmental_affordance import (
    DevelopmentalCapability,
)
from roif.development.developmental_state import (
    DevelopmentalState,
)


def test_empty_state_has_no_activatable_capabilities():
    state = DevelopmentalState()
    inspector = DevelopmentalActivationInspector()

    assert inspector.activatable_capabilities(state) == ()


def test_activation_vocabulary_contains_no_rda_stages():
    values = tuple(
        capability.value
        for capability in DevelopmentalCapability
    )

    joined = " ".join(values).lower()

    assert "rda-0" not in joined
    assert "rda-1" not in joined
    assert "rda-2" not in joined
    assert "rda-3" not in joined


def test_activation_has_no_linear_stage_controller():
    inspector = DevelopmentalActivationInspector()

    forbidden = (
        "current_stage",
        "next_stage",
        "expected_stage",
        "transition_map",
        "transition_table",
        "advance",
        "advance_to",
        "run_rda0",
        "run_rda1",
        "run_rda2",
        "run_rda3",
    )

    for name in forbidden:
        assert not hasattr(inspector, name)


def test_activation_is_descriptive_not_executable():
    activation = DevelopmentalActivation(
        capability=(
            DevelopmentalCapability
            .DETECT_FAMILIAR_STATE_CHANGE
        ),
        activatable=True,
        evidence_count=3,
        reason="Enough endogenous evidence is available.",
    )

    forbidden = (
        "execute",
        "run",
        "advance",
        "transition",
        "schedule",
    )

    for name in forbidden:
        assert not hasattr(activation, name)


def test_multiple_capabilities_can_be_activatable_together():
    class StubState:
        pass

    state = DevelopmentalState()

    # The current affordance contract makes both of these
    # capabilities available from three accumulated observations.
    #
    # We do not need to construct synthetic Observation objects here:
    # this test isolates activation semantics by supplying explicit
    # affordances through a minimal inspector stub.

    class StubAffordanceInspector:
        def inspect(self, supplied_state):
            assert supplied_state is state

            from roif.development.developmental_affordance import (
                DevelopmentalAffordance,
            )

            return (
                DevelopmentalAffordance(
                    capability=(
                        DevelopmentalCapability
                        .DETECT_FAMILIAR_STATE_CHANGE
                    ),
                    available=True,
                    evidence_count=3,
                    reason="Available independently.",
                ),
                DevelopmentalAffordance(
                    capability=(
                        DevelopmentalCapability
                        .ANALYZE_ENDOGENOUS_DIFFERENCE
                    ),
                    available=True,
                    evidence_count=3,
                    reason="Available independently.",
                ),
                DevelopmentalAffordance(
                    capability=(
                        DevelopmentalCapability
                        .DISCOVER_ENDOGENOUS_GROUPING
                    ),
                    available=False,
                    evidence_count=0,
                    reason="Not enough evidence.",
                ),
            )

    inspector = DevelopmentalActivationInspector(
        affordance_inspector=StubAffordanceInspector()
    )

    active = inspector.activatable_capabilities(state)

    assert active == (
        DevelopmentalCapability.DETECT_FAMILIAR_STATE_CHANGE,
        DevelopmentalCapability.ANALYZE_ENDOGENOUS_DIFFERENCE,
    )


def test_activation_preserves_affordance_evidence():
    state = DevelopmentalState()

    class StubAffordanceInspector:
        def inspect(self, supplied_state):
            assert supplied_state is state

            from roif.development.developmental_affordance import (
                DevelopmentalAffordance,
            )

            return (
                DevelopmentalAffordance(
                    capability=(
                        DevelopmentalCapability
                        .DISCOVER_ENDOGENOUS_GROUPING
                    ),
                    available=True,
                    evidence_count=7,
                    reason="Seven endogenous profiles are available.",
                ),
            )

    inspector = DevelopmentalActivationInspector(
        affordance_inspector=StubAffordanceInspector()
    )

    activations = inspector.inspect(state)

    assert len(activations) == 1

    activation = activations[0]

    assert (
        activation.capability
        == DevelopmentalCapability.DISCOVER_ENDOGENOUS_GROUPING
    )
    assert activation.activatable is True
    assert activation.evidence_count == 7
    assert (
        activation.reason
        == "Seven endogenous profiles are available."
    )


def test_shared_state_changes_multiple_activations_without_stage_transition():
    from roif.development.observation import (
        Observation,
        ObservationChannel,
        ObservationProvenance,
    )
    from roif.development.endogenous_difference import (
        ChannelDeviation,
        DifferenceProfile,
    )

    state = DevelopmentalState()
    inspector = DevelopmentalActivationInspector()

    initial = inspector.activatable_capabilities(state)

    assert initial == ()

    # Three observations make two independent computations
    # available at the same time.
    observations = (
        Observation(
            timestamp=0.0,
            channels={
                "a": ObservationChannel(name="a", value=0.0),
                "b": ObservationChannel(name="b", value=0.0),
            },
            provenance=ObservationProvenance(
                source_id="activation-test",
                source_type="synthetic-test",
                sequence_index=0,
            ),
        ),
        Observation(
            timestamp=1.0,
            channels={
                "a": ObservationChannel(name="a", value=0.1),
                "b": ObservationChannel(name="b", value=0.0),
            },
            provenance=ObservationProvenance(
                source_id="activation-test",
                source_type="synthetic-test",
                sequence_index=1,
            ),
        ),
        Observation(
            timestamp=2.0,
            channels={
                "a": ObservationChannel(name="a", value=0.2),
                "b": ObservationChannel(name="b", value=0.1),
            },
            provenance=ObservationProvenance(
                source_id="activation-test",
                source_type="synthetic-test",
                sequence_index=2,
            ),
        ),
    )

    for observation in observations:
        state.record_observation(observation)

    after_observations = (
        inspector.activatable_capabilities(state)
    )

    assert (
        DevelopmentalCapability
        .DETECT_FAMILIAR_STATE_CHANGE
        in after_observations
    )
    assert (
        DevelopmentalCapability
        .ANALYZE_ENDOGENOUS_DIFFERENCE
        in after_observations
    )
    assert (
        DevelopmentalCapability
        .DISCOVER_ENDOGENOUS_GROUPING
        not in after_observations
    )

    # Two DifferenceProfile objects independently make
    # endogenous grouping available.
    profile_1 = DifferenceProfile(
        timestamp=2.0,
        sequence_index=2,
        channel_deviations=(
            ChannelDeviation(
                channel_name="a",
                current_value=0.2,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=0.2,
                absolute_deviation=0.2,
            ),
            ChannelDeviation(
                channel_name="b",
                current_value=0.1,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=0.1,
                absolute_deviation=0.1,
            ),
        ),
    )

    profile_2 = DifferenceProfile(
        timestamp=3.0,
        sequence_index=3,
        channel_deviations=(
            ChannelDeviation(
                channel_name="a",
                current_value=0.4,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=0.4,
                absolute_deviation=0.4,
            ),
            ChannelDeviation(
                channel_name="b",
                current_value=0.2,
                familiar_mean=0.0,
                familiar_scale=1.0,
                signed_deviation=0.2,
                absolute_deviation=0.2,
            ),
        ),
    )

    state.record_difference_profile(profile_1)
    state.record_difference_profile(profile_2)

    after_profiles = (
        inspector.activatable_capabilities(state)
    )

    assert (
        DevelopmentalCapability
        .DETECT_FAMILIAR_STATE_CHANGE
        in after_profiles
    )
    assert (
        DevelopmentalCapability
        .ANALYZE_ENDOGENOUS_DIFFERENCE
        in after_profiles
    )
    assert (
        DevelopmentalCapability
        .DISCOVER_ENDOGENOUS_GROUPING
        in after_profiles
    )
    assert (
        DevelopmentalCapability
        .ASSESS_DIMENSIONAL_GROWTH
        not in after_profiles
    )

    # Availability changed because evidence changed,
    # not because a developmental stage advanced.
    forbidden = (
        "current_stage",
        "next_stage",
        "advance",
        "transition_map",
    )

    for name in forbidden:
        assert not hasattr(state, name)
        assert not hasattr(inspector, name)

