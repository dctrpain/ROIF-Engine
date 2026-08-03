from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.event import (
    HistoryEvent,
    HistoryEventKind,
    HistoryTarget,
    HistoryTargetKind,
    ReversibilityClass,
    StateDelta,
    TimeScale,
)
from roif.history.irreversible_change import (
    FunctionalEffect,
    IrreversibleChange,
    IrreversibleChangeError,
    IrreversibleChangeKind,
    TracePersistence,
)


def make_target() -> HistoryTarget:
    return HistoryTarget(
        kind=HistoryTargetKind.ELEMENT,
        target_id="PALM_FRONT_RAY",
        label="Palm front ray",
        metadata={
            "region": "hand",
        },
    )


def make_delta() -> StateDelta:
    return StateDelta(
        quantity="damage",
        before=0.10,
        after=0.30,
        units="normalized",
        channel="material",
        metadata={
            "mechanism": "impact",
        },
    )


def make_change(
    **overrides,
) -> IrreversibleChange:
    data = {
        "change_id": "change-damage-001",
        "kind": IrreversibleChangeKind.DAMAGE,
        "source_event_id": "event-impact-001",
        "target": make_target(),
        "delta": make_delta(),
        "onset_time": 0.42,
        "recorded_time": 0.60,
        "retained_fraction": 0.75,
        "permanence": TracePersistence.LONG_LIVED,
        "characteristic_time": 3600.0,
        "time_scale": TimeScale.MEDIUM,
        "memory_strength": 0.80,
        "capacity_effect": -0.15,
        "functional_effect": FunctionalEffect.HARMFUL,
        "reversibility": (
            ReversibilityClass.PARTIALLY_REVERSIBLE
        ),
        "cause_change_ids": (
            "change-overload-001",
        ),
        "plane_ids": (
            "mechanical",
            "material",
        ),
        "agent_ids": (
            "steel_object",
        ),
        "description": (
            "Persistent local damage after mechanical impact."
        ),
        "metadata": {
            "model": "example",
        },
    }

    data.update(overrides)

    return IrreversibleChange(**data)


def make_event(
    **overrides,
) -> HistoryEvent:
    data = {
        "event_id": "event-impact-001",
        "kind": HistoryEventKind.DAMAGE,
        "time": 0.42,
        "target": make_target(),
        "deltas": (
            StateDelta(
                quantity="damage",
                before=0.10,
                after=0.30,
                units="normalized",
                channel="material",
            ),
            StateDelta(
                quantity="capacity",
                before=0.90,
                after=0.70,
                units="normalized",
                channel="capacity",
            ),
        ),
        "plane_ids": (
            "mechanical",
            "material",
        ),
        "agent_ids": (
            "steel_object",
        ),
        "time_scale": TimeScale.MEDIUM,
        "characteristic_time": 3600.0,
        "persistence": 0.70,
        "reversibility": (
            ReversibilityClass.PARTIALLY_REVERSIBLE
        ),
        "capacity_effect": -0.20,
        "confidence": 0.95,
        "description": (
            "Mechanical impact produced persistent damage."
        ),
        "metadata": {
            "impact_force": 140.0,
        },
    }

    data.update(overrides)

    return HistoryEvent(**data)


def test_irreversible_change_creation() -> None:
    change = make_change()

    assert change.kind is IrreversibleChangeKind.DAMAGE
    assert change.change_id == "change-damage-001"
    assert change.source_event_id == "event-impact-001"
    assert change.target.target_id == "PALM_FRONT_RAY"
    assert change.delta.quantity == "damage"

    assert change.onset_time == pytest.approx(0.42)
    assert change.recorded_time == pytest.approx(0.60)
    assert change.retained_fraction == pytest.approx(0.75)

    assert (
        change.permanence
        is TracePersistence.LONG_LIVED
    )
    assert change.characteristic_time == pytest.approx(
        3600.0
    )
    assert change.time_scale is TimeScale.MEDIUM
    assert change.memory_strength == pytest.approx(0.80)
    assert change.capacity_effect == pytest.approx(-0.15)

    assert (
        change.functional_effect
        is FunctionalEffect.HARMFUL
    )
    assert (
        change.reversibility
        is ReversibilityClass.PARTIALLY_REVERSIBLE
    )


def test_change_accepts_string_enum_values() -> None:
    change = make_change(
        kind="damage",
        permanence="long_lived",
        time_scale="medium",
        functional_effect="harmful",
        reversibility="partially_reversible",
    )

    assert change.kind is IrreversibleChangeKind.DAMAGE
    assert (
        change.permanence
        is TracePersistence.LONG_LIVED
    )
    assert change.time_scale is TimeScale.MEDIUM
    assert (
        change.functional_effect
        is FunctionalEffect.HARMFUL
    )


def test_irreversible_change_is_immutable() -> None:
    change = make_change()

    with pytest.raises(FrozenInstanceError):
        change.retained_fraction = 0.90


def test_change_metadata_is_read_only() -> None:
    change = make_change()

    with pytest.raises(TypeError):
        change.metadata["new"] = "value"


def test_change_formation_delay() -> None:
    change = make_change()

    assert change.formation_delay == pytest.approx(0.18)


def test_permanent_change_property() -> None:
    change = make_change(
        permanence=TracePersistence.PERMANENT,
        retained_fraction=1.0,
        reversibility=ReversibilityClass.IRREVERSIBLE,
    )

    assert change.is_permanent is True


def test_long_lived_change_is_not_permanent() -> None:
    change = make_change(
        permanence=TracePersistence.LONG_LIVED,
    )

    assert change.is_permanent is False


def test_partially_reversible_property() -> None:
    change = make_change(
        reversibility=(
            ReversibilityClass.PARTIALLY_REVERSIBLE
        )
    )

    assert change.is_partially_reversible is True


def test_irreversible_is_not_partially_reversible() -> None:
    change = make_change(
        reversibility=ReversibilityClass.IRREVERSIBLE,
    )

    assert change.is_partially_reversible is False


def test_capacity_reduction_property() -> None:
    change = make_change(
        capacity_effect=-0.25,
    )

    assert change.reduces_capacity is True
    assert change.improves_capacity is False


def test_capacity_improvement_property() -> None:
    change = make_change(
        kind=IrreversibleChangeKind.REMODELING,
        capacity_effect=0.18,
        functional_effect=FunctionalEffect.BENEFICIAL,
    )

    assert change.improves_capacity is True
    assert change.reduces_capacity is False
    assert change.is_beneficial is True


def test_zero_capacity_effect() -> None:
    change = make_change(
        capacity_effect=0.0,
        functional_effect=FunctionalEffect.NEUTRAL,
    )

    assert change.reduces_capacity is False
    assert change.improves_capacity is False


def test_harmful_property() -> None:
    change = make_change(
        functional_effect=FunctionalEffect.HARMFUL,
    )

    assert change.is_harmful is True
    assert change.is_beneficial is False


def test_beneficial_property() -> None:
    change = make_change(
        functional_effect=FunctionalEffect.BENEFICIAL,
    )

    assert change.is_beneficial is True
    assert change.is_harmful is False


def test_numeric_total_change() -> None:
    change = make_change()

    assert change.numeric_total_change == pytest.approx(0.20)


def test_retained_numeric_change() -> None:
    change = make_change(
        retained_fraction=0.75,
    )

    assert change.retained_numeric_change == pytest.approx(
        0.15
    )


def test_non_numeric_retained_change_returns_none() -> None:
    change = make_change(
        delta=StateDelta(
            quantity="material_state",
            before="elastic",
            after="damaged",
        )
    )

    assert change.numeric_total_change is None
    assert change.retained_numeric_change is None


def test_signature_weight() -> None:
    change = make_change(
        retained_fraction=0.75,
        memory_strength=0.80,
    )

    assert change.signature_weight == pytest.approx(
        0.60
    )


def test_has_cause_change() -> None:
    change = make_change()

    assert (
        change.has_cause_change("change-overload-001")
        is True
    )
    assert change.has_cause_change("missing") is False


def test_involves_plane() -> None:
    change = make_change()

    assert change.involves_plane("mechanical") is True
    assert change.involves_plane("thermal") is False


def test_involves_agent() -> None:
    change = make_change()

    assert change.involves_agent("steel_object") is True
    assert change.involves_agent("wood_object") is False


def test_duplicate_ids_are_removed_preserving_order() -> None:
    change = make_change(
        cause_change_ids=(
            "cause-a",
            "cause-b",
            "cause-a",
        ),
        plane_ids=(
            "mechanical",
            "thermal",
            "mechanical",
        ),
        agent_ids=(
            "steel",
            "wood",
            "steel",
        ),
    )

    assert change.cause_change_ids == (
        "cause-a",
        "cause-b",
    )
    assert change.plane_ids == (
        "mechanical",
        "thermal",
    )
    assert change.agent_ids == (
        "steel",
        "wood",
    )


def test_change_rejects_self_causation() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="cannot directly cause itself",
    ):
        make_change(
            change_id="same-id",
            cause_change_ids=("same-id",),
        )


def test_change_rejects_unchanged_delta() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="actual state change",
    ):
        make_change(
            delta=StateDelta(
                quantity="damage",
                before=0.10,
                after=0.10,
            )
        )


def test_change_rejects_invalid_target() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="target must be a HistoryTarget",
    ):
        make_change(
            target="PALM_FRONT_RAY",
        )


def test_change_rejects_invalid_delta() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="delta must be a StateDelta",
    ):
        make_change(
            delta="damage",
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_change_rejects_invalid_onset_time(
    bad_value: float,
) -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="onset_time",
    ):
        make_change(
            onset_time=bad_value,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_change_rejects_invalid_recorded_time(
    bad_value: float,
) -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="recorded_time",
    ):
        make_change(
            recorded_time=bad_value,
        )


def test_change_rejects_recorded_before_onset() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="must not precede onset_time",
    ):
        make_change(
            onset_time=2.0,
            recorded_time=1.0,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        -0.01,
        0.0,
        1.01,
        float("inf"),
        float("nan"),
    ),
)
def test_change_rejects_invalid_retained_fraction(
    bad_value: float,
) -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="retained_fraction",
    ):
        make_change(
            retained_fraction=bad_value,
        )


def test_transient_change_cannot_be_fully_retained() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="fully retained trace cannot be transient",
    ):
        make_change(
            permanence=TracePersistence.TRANSIENT,
            retained_fraction=1.0,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        -0.01,
        1.01,
        float("inf"),
        float("nan"),
    ),
)
def test_change_rejects_invalid_memory_strength(
    bad_value: float,
) -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="memory_strength",
    ):
        make_change(
            memory_strength=bad_value,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        -1.01,
        1.01,
        float("inf"),
        float("nan"),
    ),
)
def test_change_rejects_invalid_capacity_effect(
    bad_value: float,
) -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="capacity_effect",
    ):
        make_change(
            capacity_effect=bad_value,
        )


def test_change_rejects_negative_characteristic_time() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="characteristic_time",
    ):
        make_change(
            characteristic_time=-0.1,
        )


def test_change_rejects_reversible_class() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="cannot be fully reversible",
    ):
        make_change(
            reversibility=ReversibilityClass.REVERSIBLE,
        )


def test_change_rejects_invalid_kind() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="unsupported irreversible change kind",
    ):
        make_change(
            kind="unsupported",
        )


def test_change_rejects_invalid_permanence() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="unsupported trace persistence",
    ):
        make_change(
            permanence="unsupported",
        )


def test_change_rejects_invalid_time_scale() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="unsupported time scale",
    ):
        make_change(
            time_scale="unsupported",
        )


def test_change_rejects_invalid_functional_effect() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="unsupported functional effect",
    ):
        make_change(
            functional_effect="unsupported",
        )


def test_change_rejects_invalid_reversibility() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="unsupported reversibility class",
    ):
        make_change(
            reversibility="unsupported",
        )


def test_change_round_trip() -> None:
    change = make_change()

    serialized = change.to_dict()
    restored = IrreversibleChange.from_dict(
        serialized
    )

    assert restored == change
    assert restored is not change
    assert restored.target is not change.target
    assert restored.delta is not change.delta


def test_change_serialized_version() -> None:
    serialized = make_change().to_dict()

    assert serialized["version"] == "1.0.0"
    assert serialized["change_id"] == "change-damage-001"
    assert serialized["kind"] == "damage"
    assert serialized["permanence"] == "long_lived"
    assert serialized["time_scale"] == "medium"


def test_change_rejects_unsupported_version() -> None:
    data = make_change().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        IrreversibleChangeError,
        match="unsupported irreversible change version",
    ):
        IrreversibleChange.from_dict(data)


def test_change_from_dict_rejects_missing_target() -> None:
    data = make_change().to_dict()
    data.pop("target")

    with pytest.raises(
        IrreversibleChangeError,
        match="target is missing or invalid",
    ):
        IrreversibleChange.from_dict(data)


def test_change_from_dict_rejects_missing_delta() -> None:
    data = make_change().to_dict()
    data.pop("delta")

    with pytest.raises(
        IrreversibleChangeError,
        match="delta is missing or invalid",
    ):
        IrreversibleChange.from_dict(data)


def test_change_generated_id_is_nonempty() -> None:
    change = IrreversibleChange(
        kind=IrreversibleChangeKind.DAMAGE,
        source_event_id="event-1",
        target=make_target(),
        delta=make_delta(),
        onset_time=0.0,
        recorded_time=0.0,
        retained_fraction=0.5,
        permanence=TracePersistence.LONG_LIVED,
    )

    assert isinstance(change.change_id, str)
    assert len(change.change_id) > 0


def test_from_event_creation() -> None:
    event = make_event()

    change = IrreversibleChange.from_event(
        event,
        kind=IrreversibleChangeKind.DAMAGE,
        quantity="damage",
        retained_fraction=0.75,
        permanence=TracePersistence.LONG_LIVED,
        recorded_time=1.0,
        functional_effect=FunctionalEffect.HARMFUL,
        change_id="change-from-event-001",
    )

    assert change.change_id == "change-from-event-001"
    assert change.source_event_id == event.event_id
    assert change.target == event.target
    assert change.delta.quantity == "damage"

    assert change.onset_time == pytest.approx(
        event.time
    )
    assert change.recorded_time == pytest.approx(1.0)
    assert change.retained_fraction == pytest.approx(
        0.75
    )

    assert change.time_scale is event.time_scale
    assert change.characteristic_time == pytest.approx(
        event.characteristic_time
    )

    assert change.plane_ids == event.plane_ids
    assert change.agent_ids == event.agent_ids

    assert change.capacity_effect == pytest.approx(
        -0.15
    )
    assert change.memory_strength == pytest.approx(
        0.75
    )

    assert (
        change.metadata["source_event_kind"]
        == "damage"
    )
    assert (
        change.metadata["source_event_confidence"]
        == pytest.approx(0.95)
    )


def test_from_event_uses_event_time_by_default() -> None:
    event = make_event()

    change = IrreversibleChange.from_event(
        event,
        kind=IrreversibleChangeKind.DAMAGE,
        quantity="damage",
        retained_fraction=0.5,
        permanence=TracePersistence.LONG_LIVED,
    )

    assert change.recorded_time == pytest.approx(
        event.time
    )


def test_from_event_default_memory_strength() -> None:
    event = make_event(
        persistence=0.90,
    )

    change = IrreversibleChange.from_event(
        event,
        kind=IrreversibleChangeKind.DAMAGE,
        quantity="damage",
        retained_fraction=0.50,
        permanence=TracePersistence.LONG_LIVED,
    )

    assert change.memory_strength == pytest.approx(
        0.90
    )


def test_from_event_explicit_values_override_defaults() -> None:
    event = make_event()

    change = IrreversibleChange.from_event(
        event,
        kind=IrreversibleChangeKind.DAMAGE,
        quantity="damage",
        retained_fraction=0.50,
        permanence=TracePersistence.LONG_LIVED,
        memory_strength=0.25,
        capacity_effect=-0.60,
        reversibility=ReversibilityClass.IRREVERSIBLE,
        description="Explicit description",
        metadata={
            "custom": "value",
        },
    )

    assert change.memory_strength == pytest.approx(
        0.25
    )
    assert change.capacity_effect == pytest.approx(
        -0.60
    )
    assert (
        change.reversibility
        is ReversibilityClass.IRREVERSIBLE
    )
    assert change.description == "Explicit description"
    assert change.metadata["custom"] == "value"


def test_from_reversible_event_becomes_partially_reversible() -> None:
    event = make_event(
        reversibility=ReversibilityClass.REVERSIBLE,
    )

    change = IrreversibleChange.from_event(
        event,
        kind=IrreversibleChangeKind.RESIDUAL_STRAIN,
        quantity="damage",
        retained_fraction=0.25,
        permanence=TracePersistence.LONG_LIVED,
    )

    assert (
        change.reversibility
        is ReversibilityClass.PARTIALLY_REVERSIBLE
    )


def test_from_event_rejects_invalid_event() -> None:
    with pytest.raises(
        IrreversibleChangeError,
        match="event must be a HistoryEvent",
    ):
        IrreversibleChange.from_event(
            "event",
            kind=IrreversibleChangeKind.DAMAGE,
            quantity="damage",
            retained_fraction=0.5,
            permanence=TracePersistence.LONG_LIVED,
        )


def test_from_event_rejects_unknown_quantity() -> None:
    event = make_event()

    with pytest.raises(
        IrreversibleChangeError,
        match="contains no delta",
    ):
        IrreversibleChange.from_event(
            event,
            kind=IrreversibleChangeKind.DAMAGE,
            quantity="unknown",
            retained_fraction=0.5,
            permanence=TracePersistence.LONG_LIVED,
        )


def test_from_event_rejects_unchanged_quantity() -> None:
    event = make_event(
        deltas=(
            StateDelta(
                quantity="temperature",
                before=20.0,
                after=20.0,
                units="degC",
            ),
        )
    )

    with pytest.raises(
        IrreversibleChangeError,
        match="contains no state change",
    ):
        IrreversibleChange.from_event(
            event,
            kind=IrreversibleChangeKind.CUSTOM,
            quantity="temperature",
            retained_fraction=0.5,
            permanence=TracePersistence.LONG_LIVED,
        )


def test_from_event_inherits_description() -> None:
    event = make_event(
        description="Inherited event description",
    )

    change = IrreversibleChange.from_event(
        event,
        kind=IrreversibleChangeKind.DAMAGE,
        quantity="damage",
        retained_fraction=0.5,
        permanence=TracePersistence.LONG_LIVED,
    )

    assert change.description == (
        "Inherited event description"
    )


def test_from_event_merges_metadata() -> None:
    event = make_event()

    change = IrreversibleChange.from_event(
        event,
        kind=IrreversibleChangeKind.DAMAGE,
        quantity="damage",
        retained_fraction=0.5,
        permanence=TracePersistence.LONG_LIVED,
        metadata={
            "decoder_label": "impact_trace",
        },
    )

    assert (
        change.metadata["source_event_kind"]
        == "damage"
    )
    assert (
        change.metadata["decoder_label"]
        == "impact_trace"
    )