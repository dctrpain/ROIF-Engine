from __future__ import annotations

from roif.development.developmental_affordance import (
    DevelopmentalAffordanceInspector,
    DevelopmentalCapability,
)
from roif.development.developmental_state import (
    DevelopmentalState,
)


def _by_capability(state):
    inspector = DevelopmentalAffordanceInspector()

    return {
        item.capability: item
        for item in inspector.inspect(state)
    }


def test_empty_state_prescribes_no_available_computation():
    state = DevelopmentalState()

    affordances = _by_capability(state)

    assert all(
        item.available is False
        for item in affordances.values()
    )


def test_affordance_vocabulary_contains_no_rda_stages():
    values = tuple(
        capability.value
        for capability in DevelopmentalCapability
    )

    assert all(
        "rda-" not in value.lower()
        for value in values
    )


def test_inspector_has_no_linear_stage_controller():
    inspector = DevelopmentalAffordanceInspector()

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
        assert not hasattr(inspector, name)


def test_dimensional_growth_is_not_inferred_from_observation_count():
    state = DevelopmentalState()

    affordances = _by_capability(state)

    dimensional = affordances[
        DevelopmentalCapability.ASSESS_DIMENSIONAL_GROWTH
    ]

    assert dimensional.available is False
    assert "representation" in dimensional.reason.lower()


def test_affordances_are_descriptive_not_executable():
    state = DevelopmentalState()

    affordances = _by_capability(state)

    for item in affordances.values():
        assert not hasattr(item, "execute")
        assert not hasattr(item, "run")
        assert not hasattr(item, "advance")
