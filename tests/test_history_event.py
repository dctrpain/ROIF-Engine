from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.event import (
    HistoryEvent,
    HistoryEventKind,
    HistoryTarget,
    HistoryTargetKind,
    HistoryValidationError,
    ReversibilityClass,
    StateDelta,
    TimeScale,
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


def make_event(
    **overrides,
) -> HistoryEvent:
    data = {
        "event_id": "event-impact-001",
        "kind": HistoryEventKind.EXTERNAL_PERTURBATION,
        "time": 0.42,
        "target": make_target(),
        "deltas": (
            StateDelta(
                quantity="impact_force",
                before=0.0,
                after=140.0,
                units="N",
                channel="mechanical",
            ),
            StateDelta(
                quantity="damage",
                before=0.0,
                after=0.03,
                units="normalized",
                channel="capacity",
            ),
        ),
        "cause_event_ids": (
            "event-contact-001",
        ),
        "plane_ids": (
            "mechanical",
        ),
        "agent_ids": (
            "steel_object",
        ),
        "time_scale": TimeScale.FAST,
        "characteristic_time": 0.015,
        "persistence": 0.12,
        "reversibility": (
            ReversibilityClass.PARTIALLY_REVERSIBLE
        ),
        "capacity_effect": -0.08,
        "confidence": 0.95,
        "description": (
            "Distributed impact on the palm structure."
        ),
        "metadata": {
            "force_units": "N",
        },
    }

    data.update(overrides)
    return HistoryEvent(**data)


def test_history_target_creation() -> None:
    target = make_target()

    assert target.kind is HistoryTargetKind.ELEMENT
    assert target.target_id == "PALM_FRONT_RAY"
    assert target.label == "Palm front ray"
    assert target.metadata["region"] == "hand"


def test_history_target_strips_strings() -> None:
    target = HistoryTarget(
        kind="element",
        target_id="  ELEMENT_1  ",
        label="  Main element  ",
    )

    assert target.target_id == "ELEMENT_1"
    assert target.label == "Main element"


def test_history_target_rejects_empty_id() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="target_id must not be empty",
    ):
        HistoryTarget(
            kind=HistoryTargetKind.NODE,
            target_id="   ",
        )


def test_history_target_rejects_invalid_kind() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="unsupported target kind",
    ):
        HistoryTarget(
            kind="unsupported",
            target_id="A",
        )


def test_history_target_metadata_is_read_only() -> None:
    target = make_target()

    with pytest.raises(TypeError):
        target.metadata["new"] = "value"


def test_history_target_round_trip() -> None:
    target = make_target()

    restored = HistoryTarget.from_dict(
        target.to_dict()
    )

    assert restored == target
    assert restored is not target


def test_state_delta_creation() -> None:
    delta = StateDelta(
        quantity="capacity",
        before=0.92,
        after=0.71,
        units="normalized",
        channel="capacity",
    )

    assert delta.quantity == "capacity"
    assert delta.numeric_change == pytest.approx(
        -0.21
    )
    assert delta.changed is True


def test_state_delta_unchanged() -> None:
    delta = StateDelta(
        quantity="temperature",
        before=20.0,
        after=20.0,
        units="degC",
    )

    assert delta.changed is False
    assert delta.numeric_change == pytest.approx(0.0)


def test_state_delta_non_numeric_change_returns_none() -> None:
    delta = StateDelta(
        quantity="material_state",
        before="elastic",
        after="damaged",
    )

    assert delta.numeric_change is None
    assert delta.changed is True


def test_state_delta_rejects_empty_quantity() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="quantity must not be empty",
    ):
        StateDelta(
            quantity="",
            before=0,
            after=1,
        )


def test_state_delta_metadata_is_read_only() -> None:
    delta = StateDelta(
        quantity="damage",
        before=0.0,
        after=0.1,
        metadata={
            "source": "impact",
        },
    )

    with pytest.raises(TypeError):
        delta.metadata["source"] = "other"


def test_state_delta_round_trip() -> None:
    delta = StateDelta(
        quantity="damage",
        before=0.0,
        after=0.1,
        units="normalized",
        channel="material",
        metadata={
            "location": "surface",
        },
    )

    restored = StateDelta.from_dict(
        delta.to_dict()
    )

    assert restored == delta
    assert restored is not delta


def test_history_event_creation() -> None:
    event = make_event()

    assert (
        event.kind
        is HistoryEventKind.EXTERNAL_PERTURBATION
    )
    assert event.time == pytest.approx(0.42)
    assert event.target.target_id == "PALM_FRONT_RAY"
    assert event.time_scale is TimeScale.FAST
    assert event.characteristic_time == pytest.approx(
        0.015
    )
    assert event.persistence == pytest.approx(0.12)
    assert (
        event.reversibility
        is ReversibilityClass.PARTIALLY_REVERSIBLE
    )
    assert event.capacity_effect == pytest.approx(
        -0.08
    )
    assert event.confidence == pytest.approx(0.95)


def test_history_event_is_immutable() -> None:
    event = make_event()

    with pytest.raises(FrozenInstanceError):
        event.time = 1.0


def test_history_event_metadata_is_read_only() -> None:
    event = make_event()

    with pytest.raises(TypeError):
        event.metadata["new"] = "value"


def test_history_event_properties() -> None:
    event = make_event()

    assert event.is_persistent is True
    assert event.is_irreversible is False
    assert event.reduces_capacity is True
    assert event.improves_capacity is False


def test_irreversible_event_property() -> None:
    event = make_event(
        reversibility=ReversibilityClass.IRREVERSIBLE,
        persistence=1.0,
    )

    assert event.is_irreversible is True
    assert event.is_persistent is True


def test_capacity_improvement_property() -> None:
    event = make_event(
        kind=HistoryEventKind.REMODELING,
        capacity_effect=0.12,
    )

    assert event.improves_capacity is True
    assert event.reduces_capacity is False


def test_zero_capacity_effect_properties() -> None:
    event = make_event(
        capacity_effect=0.0,
    )

    assert event.improves_capacity is False
    assert event.reduces_capacity is False


def test_changed_quantities() -> None:
    event = make_event(
        deltas=(
            StateDelta(
                quantity="force",
                before=0.0,
                after=10.0,
            ),
            StateDelta(
                quantity="temperature",
                before=20.0,
                after=20.0,
            ),
            StateDelta(
                quantity="damage",
                before=0.0,
                after=0.2,
            ),
        )
    )

    assert event.changed_quantities == (
        "force",
        "damage",
    )


def test_delta_for_existing_quantity() -> None:
    event = make_event()

    delta = event.delta_for("damage")

    assert delta is not None
    assert delta.after == pytest.approx(0.03)


def test_delta_for_unknown_quantity() -> None:
    event = make_event()

    assert event.delta_for("unknown") is None


def test_has_cause() -> None:
    event = make_event()

    assert event.has_cause("event-contact-001") is True
    assert event.has_cause("missing") is False


def test_involves_plane() -> None:
    event = make_event()

    assert event.involves_plane("mechanical") is True
    assert event.involves_plane("thermal") is False


def test_involves_agent() -> None:
    event = make_event()

    assert event.involves_agent("steel_object") is True
    assert event.involves_agent("wood_object") is False


def test_duplicate_ids_are_removed_preserving_order() -> None:
    event = make_event(
        cause_event_ids=(
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

    assert event.cause_event_ids == (
        "cause-a",
        "cause-b",
    )
    assert event.plane_ids == (
        "mechanical",
        "thermal",
    )
    assert event.agent_ids == (
        "steel",
        "wood",
    )


def test_event_rejects_self_causation() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="cannot directly cause itself",
    ):
        make_event(
            event_id="same-id",
            cause_event_ids=("same-id",),
        )


@pytest.mark.parametrize(
    "bad_time",
    (
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_event_rejects_invalid_time(
    bad_time: float,
) -> None:
    with pytest.raises(
        HistoryValidationError,
        match="time",
    ):
        make_event(
            time=bad_time,
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
def test_event_rejects_invalid_persistence(
    bad_value: float,
) -> None:
    with pytest.raises(
        HistoryValidationError,
        match="persistence",
    ):
        make_event(
            persistence=bad_value,
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
def test_event_rejects_invalid_confidence(
    bad_value: float,
) -> None:
    with pytest.raises(
        HistoryValidationError,
        match="confidence",
    ):
        make_event(
            confidence=bad_value,
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
def test_event_rejects_invalid_capacity_effect(
    bad_value: float,
) -> None:
    with pytest.raises(
        HistoryValidationError,
        match="capacity_effect",
    ):
        make_event(
            capacity_effect=bad_value,
        )


def test_event_rejects_negative_characteristic_time() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="characteristic_time",
    ):
        make_event(
            characteristic_time=-0.1,
        )


def test_event_rejects_invalid_delta_object() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="all deltas must be StateDelta",
    ):
        make_event(
            deltas=("not-a-delta",),
        )


def test_event_rejects_invalid_target() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="target must be a HistoryTarget",
    ):
        make_event(
            target="PALM_FRONT_RAY",
        )


def test_event_rejects_invalid_kind() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="unsupported history event kind",
    ):
        make_event(
            kind="unsupported",
        )


def test_event_rejects_invalid_time_scale() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="unsupported time scale",
    ):
        make_event(
            time_scale="unsupported",
        )


def test_event_rejects_invalid_reversibility() -> None:
    with pytest.raises(
        HistoryValidationError,
        match="unsupported reversibility class",
    ):
        make_event(
            reversibility="unsupported",
        )


def test_event_round_trip() -> None:
    event = make_event()

    serialized = event.to_dict()
    restored = HistoryEvent.from_dict(serialized)

    assert restored == event
    assert restored is not event
    assert restored.target is not event.target
    assert restored.deltas[0] is not event.deltas[0]


def test_event_serialized_version() -> None:
    event = make_event()

    serialized = event.to_dict()

    assert serialized["version"] == "1.0.0"
    assert serialized["event_id"] == "event-impact-001"
    assert serialized["kind"] == "external_perturbation"
    assert serialized["time_scale"] == "fast"


def test_event_rejects_unsupported_serialized_version() -> None:
    data = make_event().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        HistoryValidationError,
        match="unsupported history event version",
    ):
        HistoryEvent.from_dict(data)


def test_event_from_dict_rejects_missing_target() -> None:
    data = make_event().to_dict()
    data.pop("target")

    with pytest.raises(
        HistoryValidationError,
        match="target is missing or invalid",
    ):
        HistoryEvent.from_dict(data)


def test_event_from_dict_rejects_invalid_deltas() -> None:
    data = make_event().to_dict()
    data["deltas"] = "invalid"

    with pytest.raises(
        HistoryValidationError,
        match="deltas must be a sequence",
    ):
        HistoryEvent.from_dict(data)


def test_event_generated_id_is_nonempty() -> None:
    event = HistoryEvent(
        kind=HistoryEventKind.CUSTOM,
        time=0.0,
        target=HistoryTarget(
            kind=HistoryTargetKind.SYSTEM,
            target_id="SYSTEM",
        ),
    )

    assert isinstance(event.event_id, str)
    assert len(event.event_id) > 0