"""Tests for discrete ROIF cascade events."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from roif.cascade_event import (
    CascadeEvent,
    CascadeEventBatch,
    CascadeEventError,
    CascadeEventKind,
    CascadeEventPolicy,
    CascadeEventSeverity,
    detect_cascade_events,
    link_event_chain,
)
from roif.plane_kernel import PlaneKernelResult


def make_event(
    *,
    time: float = 1.0,
    plane_id: str = "a",
    kind: CascadeEventKind | str = CascadeEventKind.AMPLIFICATION,
    severity: CascadeEventSeverity | str = CascadeEventSeverity.LOW,
    activation_before: float = 0.2,
    activation_after: float = 0.3,
    event_id: str | None = None,
    source_plane_id: str | None = None,
    parent_event_id: str | None = None,
    clipped: bool = False,
    retained_by_hysteresis: bool = False,
    metadata: dict[str, Any] | None = None,
) -> CascadeEvent:
    return CascadeEvent(
        time=time,
        plane_id=plane_id,
        kind=kind,
        severity=severity,
        activation_before=activation_before,
        activation_after=activation_after,
        event_id=event_id,
        source_plane_id=source_plane_id,
        parent_event_id=parent_event_id,
        clipped=clipped,
        retained_by_hysteresis=retained_by_hysteresis,
        metadata={} if metadata is None else metadata,
    )


def make_result(
    *,
    plane_ids: tuple[str, ...] = ("a", "b"),
    time: float = 1.0,
    dt: float = 1.0,
    activation_before: Any = (0.0, 0.0),
    interaction: Any = (0.0, 0.0),
    rate: Any = (0.0, 0.0),
    activation_after: Any = (0.0, 0.0),
    clipped: Any = (False, False),
    retained_by_hysteresis: Any = (False, False),
) -> PlaneKernelResult:
    return PlaneKernelResult(
        plane_ids=plane_ids,
        time=time,
        dt=dt,
        activation_before=activation_before,
        interaction=interaction,
        rate=rate,
        activation_after=activation_after,
        clipped=clipped,
        retained_by_hysteresis=retained_by_hysteresis,
    )


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (CascadeEventKind.ACTIVATION, "activation"),
        (CascadeEventKind.DEACTIVATION, "deactivation"),
        (CascadeEventKind.AMPLIFICATION, "amplification"),
        (CascadeEventKind.ATTENUATION, "attenuation"),
        (CascadeEventKind.SATURATION, "saturation"),
        (CascadeEventKind.FLOOR_CONTACT, "floor_contact"),
        (
            CascadeEventKind.HYSTERESIS_RETENTION,
            "hysteresis_retention",
        ),
        (CascadeEventKind.PROPAGATION, "propagation"),
    ],
)
def test_event_kind_values(
    member: CascadeEventKind,
    value: str,
) -> None:
    assert member.value == value
    assert isinstance(member, str)


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (CascadeEventSeverity.TRACE, "trace"),
        (CascadeEventSeverity.LOW, "low"),
        (CascadeEventSeverity.MODERATE, "moderate"),
        (CascadeEventSeverity.HIGH, "high"),
        (CascadeEventSeverity.CRITICAL, "critical"),
    ],
)
def test_event_severity_values(
    member: CascadeEventSeverity,
    value: str,
) -> None:
    assert member.value == value
    assert isinstance(member, str)


# ---------------------------------------------------------------------------
# CascadeEvent construction and properties
# ---------------------------------------------------------------------------


def test_event_accepts_enum_values() -> None:
    event = make_event()

    assert event.kind is CascadeEventKind.AMPLIFICATION
    assert event.severity is CascadeEventSeverity.LOW


def test_event_coerces_string_values_to_enums() -> None:
    event = make_event(
        kind="activation",
        severity="high",
    )

    assert event.kind is CascadeEventKind.ACTIVATION
    assert event.severity is CascadeEventSeverity.HIGH


def test_event_trims_identifiers() -> None:
    event = make_event(
        plane_id="  target  ",
        event_id="  explicit-id  ",
        source_plane_id="  source  ",
        parent_event_id="  parent  ",
    )

    assert event.plane_id == "target"
    assert event.event_id == "explicit-id"
    assert event.source_plane_id == "source"
    assert event.parent_event_id == "parent"


def test_source_plane_equal_to_target_is_removed() -> None:
    event = make_event(
        plane_id="a",
        source_plane_id="a",
    )

    assert event.source_plane_id is None


def test_event_generates_id() -> None:
    event = make_event()

    assert event.event_id is not None
    assert event.event_id.startswith("evt_")
    assert len(event.event_id) == 24


def test_generated_event_id_is_deterministic() -> None:
    first = make_event()
    second = make_event()

    assert first.event_id == second.event_id


def test_generated_event_id_changes_with_plane() -> None:
    first = make_event(plane_id="a")
    second = make_event(plane_id="b")

    assert first.event_id != second.event_id


def test_generated_event_id_changes_with_time() -> None:
    first = make_event(time=1.0)
    second = make_event(time=2.0)

    assert first.event_id != second.event_id


def test_generated_event_id_changes_with_kind() -> None:
    first = make_event(kind=CascadeEventKind.AMPLIFICATION)
    second = make_event(kind=CascadeEventKind.ATTENUATION)

    assert first.event_id != second.event_id


def test_generated_event_id_changes_with_activation() -> None:
    first = make_event(activation_after=0.3)
    second = make_event(activation_after=0.4)

    assert first.event_id != second.event_id


def test_explicit_event_id_is_preserved() -> None:
    event = make_event(event_id="event-1")

    assert event.event_id == "event-1"


def test_event_delta_for_increase() -> None:
    event = make_event(
        activation_before=0.2,
        activation_after=0.7,
    )

    assert event.delta == pytest.approx(0.5)


def test_event_delta_for_decrease() -> None:
    event = make_event(
        activation_before=0.8,
        activation_after=0.3,
    )

    assert event.delta == pytest.approx(-0.5)


def test_event_magnitude_is_absolute_delta() -> None:
    event = make_event(
        activation_before=0.8,
        activation_after=0.3,
    )

    assert event.magnitude == pytest.approx(0.5)


def test_event_increase_and_decrease_flags() -> None:
    increased = make_event(
        activation_before=0.2,
        activation_after=0.4,
    )
    decreased = make_event(
        activation_before=0.4,
        activation_after=0.2,
    )
    unchanged = make_event(
        activation_before=0.4,
        activation_after=0.4,
    )

    assert increased.is_increase is True
    assert increased.is_decrease is False
    assert decreased.is_increase is False
    assert decreased.is_decrease is True
    assert unchanged.is_increase is False
    assert unchanged.is_decrease is False


def test_event_retention_flag() -> None:
    event = make_event(retained_by_hysteresis=True)

    assert event.is_retained is True


def test_event_root_flag() -> None:
    root = make_event(parent_event_id=None)
    child = make_event(parent_event_id="parent")

    assert root.is_root is True
    assert child.is_root is False


def test_event_is_frozen() -> None:
    event = make_event()

    with pytest.raises(FrozenInstanceError):
        event.time = 2.0  # type: ignore[misc]


def test_event_metadata_is_trimmed_and_read_only() -> None:
    event = make_event(metadata={"  source  ": "kernel"})

    assert dict(event.metadata) == {"source": "kernel"}

    with pytest.raises(TypeError):
        event.metadata["source"] = "changed"  # type: ignore[index]


def test_event_metadata_input_is_copied() -> None:
    metadata = {"source": "original"}
    event = make_event(metadata=metadata)

    metadata["source"] = "changed"

    assert event.metadata["source"] == "original"


def test_event_with_parent_returns_independent_event() -> None:
    event = make_event()
    linked = event.with_parent(
        "parent-event",
        source_plane_id="source",
    )

    assert linked is not event
    assert linked.parent_event_id == "parent-event"
    assert linked.source_plane_id == "source"
    assert linked.event_id != event.event_id
    assert event.parent_event_id is None


def test_event_with_metadata_merges_values() -> None:
    event = make_event(metadata={"a": 1})
    updated = event.with_metadata(b=2, a=3)

    assert dict(event.metadata) == {"a": 1}
    assert dict(updated.metadata) == {"a": 3, "b": 2}


def test_event_as_dict() -> None:
    event = make_event(
        event_id="event-1",
        source_plane_id="source",
        parent_event_id="parent",
        clipped=True,
        metadata={"origin": "test"},
    )

    data = event.as_dict()

    assert data == {
        "event_id": "event-1",
        "time": 1.0,
        "plane_id": "a",
        "source_plane_id": "source",
        "parent_event_id": "parent",
        "kind": "amplification",
        "severity": "low",
        "activation_before": 0.2,
        "activation_after": 0.3,
        "delta": pytest.approx(0.1),
        "magnitude": pytest.approx(0.1),
        "clipped": True,
        "retained_by_hysteresis": False,
        "metadata": {"origin": "test"},
    }


# ---------------------------------------------------------------------------
# CascadeEvent validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_event_time_must_be_finite_and_non_negative(value: float) -> None:
    with pytest.raises(CascadeEventError, match="time"):
        make_event(time=value)


@pytest.mark.parametrize("value", ["", " ", "\t", "\n"])
def test_event_plane_id_cannot_be_empty(value: str) -> None:
    with pytest.raises(CascadeEventError, match="plane_id cannot be empty"):
        make_event(plane_id=value)


def test_event_plane_id_must_be_string() -> None:
    with pytest.raises(CascadeEventError, match="plane_id must be a string"):
        make_event(plane_id=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_activation_before_must_be_in_unit_interval(value: float) -> None:
    with pytest.raises(CascadeEventError, match="activation_before"):
        make_event(activation_before=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_activation_after_must_be_in_unit_interval(value: float) -> None:
    with pytest.raises(CascadeEventError, match="activation_after"):
        make_event(activation_after=value)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(CascadeEventError, match="Unknown cascade event kind"):
        make_event(kind="unknown")


def test_unknown_severity_is_rejected() -> None:
    with pytest.raises(
        CascadeEventError,
        match="Unknown cascade event severity",
    ):
        make_event(severity="unknown")


@pytest.mark.parametrize("field_name", ["event_id", "source_plane_id", "parent_event_id"])
def test_optional_identifier_cannot_be_empty(field_name: str) -> None:
    kwargs = {field_name: " "}

    with pytest.raises(CascadeEventError, match="cannot be empty"):
        make_event(**kwargs)


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_clipped_must_be_bool(value: object) -> None:
    with pytest.raises(CascadeEventError, match="clipped must be a bool"):
        make_event(clipped=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_retained_by_hysteresis_must_be_bool(value: object) -> None:
    with pytest.raises(
        CascadeEventError,
        match="retained_by_hysteresis must be a bool",
    ):
        make_event(retained_by_hysteresis=value)  # type: ignore[arg-type]


def test_event_metadata_must_be_mapping() -> None:
    with pytest.raises(CascadeEventError, match="metadata must be a mapping"):
        make_event(metadata=["invalid"])  # type: ignore[arg-type]


def test_event_metadata_key_must_be_string() -> None:
    with pytest.raises(CascadeEventError, match="metadata key must be a string"):
        make_event(metadata={1: "invalid"})  # type: ignore[dict-item]


def test_event_metadata_key_cannot_be_empty() -> None:
    with pytest.raises(CascadeEventError, match="metadata key cannot be empty"):
        make_event(metadata={" ": "invalid"})


# ---------------------------------------------------------------------------
# CascadeEventPolicy construction and severity
# ---------------------------------------------------------------------------


def test_policy_defaults() -> None:
    policy = CascadeEventPolicy()

    assert policy.minimum_delta == pytest.approx(1e-12)
    assert policy.activation_threshold == pytest.approx(0.5)
    assert policy.deactivation_threshold == pytest.approx(0.5)
    assert policy.low_threshold == pytest.approx(0.05)
    assert policy.moderate_threshold == pytest.approx(0.15)
    assert policy.high_threshold == pytest.approx(0.35)
    assert policy.critical_threshold == pytest.approx(0.65)
    assert policy.emit_hysteresis_events is True
    assert policy.emit_clipping_events is True


def test_policy_is_frozen() -> None:
    policy = CascadeEventPolicy()

    with pytest.raises(FrozenInstanceError):
        policy.minimum_delta = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("magnitude", "expected"),
    [
        (0.0, CascadeEventSeverity.TRACE),
        (0.049, CascadeEventSeverity.TRACE),
        (0.05, CascadeEventSeverity.LOW),
        (0.149, CascadeEventSeverity.LOW),
        (0.15, CascadeEventSeverity.MODERATE),
        (0.349, CascadeEventSeverity.MODERATE),
        (0.35, CascadeEventSeverity.HIGH),
        (0.649, CascadeEventSeverity.HIGH),
        (0.65, CascadeEventSeverity.CRITICAL),
        (1.0, CascadeEventSeverity.CRITICAL),
    ],
)
def test_policy_severity_for(
    magnitude: float,
    expected: CascadeEventSeverity,
) -> None:
    assert CascadeEventPolicy().severity_for(magnitude) is expected


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, -np.inf])
def test_severity_magnitude_must_be_finite_and_non_negative(
    value: float,
) -> None:
    with pytest.raises(CascadeEventError, match="magnitude"):
        CascadeEventPolicy().severity_for(value)


@pytest.mark.parametrize(
    "field_name",
    [
        "minimum_delta",
        "activation_threshold",
        "deactivation_threshold",
        "low_threshold",
        "moderate_threshold",
        "high_threshold",
        "critical_threshold",
    ],
)
@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_policy_numeric_fields_must_be_in_unit_interval(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(CascadeEventError, match=field_name):
        CascadeEventPolicy(**{field_name: value})


def test_policy_severity_thresholds_must_be_non_decreasing() -> None:
    with pytest.raises(
        CascadeEventError,
        match="Severity thresholds must be non-decreasing",
    ):
        CascadeEventPolicy(
            low_threshold=0.2,
            moderate_threshold=0.1,
        )


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_emit_hysteresis_events_must_be_bool(value: object) -> None:
    with pytest.raises(CascadeEventError, match="must be a bool"):
        CascadeEventPolicy(
            emit_hysteresis_events=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_emit_clipping_events_must_be_bool(value: object) -> None:
    with pytest.raises(CascadeEventError, match="must be a bool"):
        CascadeEventPolicy(
            emit_clipping_events=value,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Event-kind classification
# ---------------------------------------------------------------------------


def test_kind_for_activation_crossing() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.4,
        activation_after=0.5,
    )

    assert kind is CascadeEventKind.ACTIVATION


def test_kind_for_deactivation_crossing() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.5,
        activation_after=0.4,
    )

    assert kind is CascadeEventKind.DEACTIVATION


def test_kind_for_amplification_without_crossing() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.1,
        activation_after=0.2,
    )

    assert kind is CascadeEventKind.AMPLIFICATION


def test_kind_for_attenuation_without_crossing() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.4,
        activation_after=0.3,
    )

    assert kind is CascadeEventKind.ATTENUATION


def test_kind_for_propagated_change() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.1,
        activation_after=0.2,
        propagated=True,
    )

    assert kind is CascadeEventKind.PROPAGATION


def test_activation_crossing_has_priority_over_propagation() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.4,
        activation_after=0.6,
        propagated=True,
    )

    assert kind is CascadeEventKind.ACTIVATION


def test_kind_for_upper_clipping() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.4,
        activation_after=1.0,
        clipped=True,
    )

    assert kind is CascadeEventKind.SATURATION


def test_kind_for_lower_clipping() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.4,
        activation_after=0.0,
        clipped=True,
    )

    assert kind is CascadeEventKind.FLOOR_CONTACT


def test_clipping_events_can_be_disabled() -> None:
    policy = CascadeEventPolicy(emit_clipping_events=False)

    kind = policy.kind_for(
        activation_before=0.4,
        activation_after=1.0,
        clipped=True,
    )

    assert kind is CascadeEventKind.ACTIVATION


def test_kind_for_hysteresis_retention() -> None:
    kind = CascadeEventPolicy().kind_for(
        activation_before=0.4,
        activation_after=0.4,
        retained_by_hysteresis=True,
    )

    assert kind is CascadeEventKind.HYSTERESIS_RETENTION


def test_hysteresis_events_can_be_disabled() -> None:
    policy = CascadeEventPolicy(emit_hysteresis_events=False)

    kind = policy.kind_for(
        activation_before=0.4,
        activation_after=0.4,
        retained_by_hysteresis=True,
    )

    assert kind is None


def test_kind_for_change_below_minimum_delta_is_none() -> None:
    policy = CascadeEventPolicy(minimum_delta=0.1)

    kind = policy.kind_for(
        activation_before=0.2,
        activation_after=0.25,
    )

    assert kind is None


def test_kind_for_change_equal_to_minimum_delta_is_emitted() -> None:
    policy = CascadeEventPolicy(minimum_delta=0.1)

    kind = policy.kind_for(
        activation_before=0.2,
        activation_after=0.3,
    )

    assert kind is CascadeEventKind.AMPLIFICATION


# ---------------------------------------------------------------------------
# CascadeEventBatch
# ---------------------------------------------------------------------------


def test_empty_batch() -> None:
    batch = CascadeEventBatch(time=1.0)

    assert len(batch) == 0
    assert bool(batch) is False
    assert tuple(batch) == ()
    assert batch.plane_ids == ()
    assert batch.critical_events == ()


def test_batch_iteration_and_plane_ids() -> None:
    first = make_event(event_id="e1", plane_id="a")
    second = make_event(event_id="e2", plane_id="b")
    batch = CascadeEventBatch(time=1.0, events=(first, second))

    assert tuple(batch) == (first, second)
    assert batch.plane_ids == ("a", "b")
    assert bool(batch) is True


def test_batch_critical_events() -> None:
    low = make_event(
        event_id="low",
        severity=CascadeEventSeverity.LOW,
    )
    critical = make_event(
        event_id="critical",
        plane_id="b",
        severity=CascadeEventSeverity.CRITICAL,
    )
    batch = CascadeEventBatch(time=1.0, events=(low, critical))

    assert batch.critical_events == (critical,)


def test_batch_for_plane() -> None:
    first = make_event(event_id="e1", plane_id="a")
    second = make_event(event_id="e2", plane_id="b")
    third = make_event(event_id="e3", plane_id="a")
    batch = CascadeEventBatch(
        time=1.0,
        events=(first, second, third),
    )

    assert batch.for_plane(" a ") == (first, third)


def test_batch_by_kind_accepts_enum() -> None:
    first = make_event(
        event_id="e1",
        kind=CascadeEventKind.AMPLIFICATION,
    )
    second = make_event(
        event_id="e2",
        plane_id="b",
        kind=CascadeEventKind.ATTENUATION,
    )
    batch = CascadeEventBatch(time=1.0, events=(first, second))

    assert batch.by_kind(CascadeEventKind.ATTENUATION) == (second,)


def test_batch_by_kind_accepts_string() -> None:
    event = make_event(event_id="e1")
    batch = CascadeEventBatch(time=1.0, events=(event,))

    assert batch.by_kind("amplification") == (event,)


def test_batch_as_dict() -> None:
    event = make_event(event_id="e1")
    batch = CascadeEventBatch(time=1.0, events=(event,))

    data = batch.as_dict()

    assert data["time"] == pytest.approx(1.0)
    assert data["event_count"] == 1
    assert data["events"] == [event.as_dict()]


def test_batch_is_frozen() -> None:
    batch = CascadeEventBatch(time=1.0)

    with pytest.raises(FrozenInstanceError):
        batch.time = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_batch_time_must_be_finite_and_non_negative(value: float) -> None:
    with pytest.raises(CascadeEventError, match="time"):
        CascadeEventBatch(time=value)


def test_batch_rejects_non_event() -> None:
    with pytest.raises(
        CascadeEventError,
        match=r"events\[0\] must be a CascadeEvent",
    ):
        CascadeEventBatch(time=1.0, events=(object(),))  # type: ignore[arg-type]


def test_batch_rejects_mismatched_event_time() -> None:
    event = make_event(time=2.0)

    with pytest.raises(
        CascadeEventError,
        match="must have the batch time",
    ):
        CascadeEventBatch(time=1.0, events=(event,))


def test_batch_rejects_duplicate_event_ids() -> None:
    first = make_event(event_id="same", plane_id="a")
    second = make_event(event_id="same", plane_id="b")

    with pytest.raises(
        CascadeEventError,
        match="Event IDs must be unique",
    ):
        CascadeEventBatch(time=1.0, events=(first, second))


def test_batch_for_plane_rejects_empty_id() -> None:
    batch = CascadeEventBatch(time=1.0)

    with pytest.raises(CascadeEventError, match="plane_id cannot be empty"):
        batch.for_plane(" ")


def test_batch_by_kind_rejects_unknown_kind() -> None:
    batch = CascadeEventBatch(time=1.0)

    with pytest.raises(CascadeEventError, match="Unknown cascade event kind"):
        batch.by_kind("unknown")


# ---------------------------------------------------------------------------
# detect_cascade_events
# ---------------------------------------------------------------------------


def test_detect_requires_plane_kernel_result() -> None:
    with pytest.raises(
        CascadeEventError,
        match="result must be a PlaneKernelResult",
    ):
        detect_cascade_events(object())  # type: ignore[arg-type]


def test_detect_rejects_invalid_policy() -> None:
    with pytest.raises(
        CascadeEventError,
        match="policy must be a CascadeEventPolicy",
    ):
        detect_cascade_events(
            make_result(),
            policy=object(),  # type: ignore[arg-type]
        )


def test_detect_returns_empty_batch_for_no_change() -> None:
    result = make_result()
    batch = detect_cascade_events(result)

    assert isinstance(batch, CascadeEventBatch)
    assert batch.time == pytest.approx(1.0)
    assert len(batch) == 0


def test_detect_amplification_event() -> None:
    result = make_result(
        activation_before=(0.1, 0.0),
        activation_after=(0.2, 0.0),
        interaction=(0.2, 0.0),
        rate=(0.1, 0.0),
    )

    batch = detect_cascade_events(result)

    assert len(batch) == 1
    event = batch.events[0]
    assert event.plane_id == "a"
    assert event.kind is CascadeEventKind.AMPLIFICATION
    assert event.severity is CascadeEventSeverity.LOW
    assert event.delta == pytest.approx(0.1)


def test_detect_attenuation_event() -> None:
    result = make_result(
        activation_before=(0.4, 0.0),
        activation_after=(0.2, 0.0),
        interaction=(0.2, 0.0),
        rate=(-0.2, 0.0),
    )

    event = detect_cascade_events(result).events[0]

    assert event.kind is CascadeEventKind.ATTENUATION
    assert event.severity is CascadeEventSeverity.MODERATE


def test_detect_activation_crossing() -> None:
    result = make_result(
        activation_before=(0.4, 0.0),
        activation_after=(0.6, 0.0),
    )

    event = detect_cascade_events(result).events[0]

    assert event.kind is CascadeEventKind.ACTIVATION


def test_detect_deactivation_crossing() -> None:
    result = make_result(
        activation_before=(0.6, 0.0),
        activation_after=(0.4, 0.0),
    )

    event = detect_cascade_events(result).events[0]

    assert event.kind is CascadeEventKind.DEACTIVATION


def test_detect_saturation_event() -> None:
    result = make_result(
        activation_before=(0.2, 0.0),
        activation_after=(1.0, 0.0),
        clipped=(True, False),
    )

    event = detect_cascade_events(result).events[0]

    assert event.kind is CascadeEventKind.SATURATION
    assert event.clipped is True
    assert event.severity is CascadeEventSeverity.CRITICAL


def test_detect_floor_contact_event() -> None:
    result = make_result(
        activation_before=(0.8, 0.0),
        activation_after=(0.0, 0.0),
        clipped=(True, False),
    )

    event = detect_cascade_events(result).events[0]

    assert event.kind is CascadeEventKind.FLOOR_CONTACT
    assert event.clipped is True


def test_detect_hysteresis_event() -> None:
    result = make_result(
        activation_before=(0.4, 0.0),
        activation_after=(0.4, 0.0),
        retained_by_hysteresis=(True, False),
    )

    event = detect_cascade_events(result).events[0]

    assert event.kind is CascadeEventKind.HYSTERESIS_RETENTION
    assert event.retained_by_hysteresis is True
    assert event.severity is CascadeEventSeverity.TRACE


def test_detect_can_suppress_hysteresis_event() -> None:
    result = make_result(
        activation_before=(0.4, 0.0),
        activation_after=(0.4, 0.0),
        retained_by_hysteresis=(True, False),
    )
    policy = CascadeEventPolicy(emit_hysteresis_events=False)

    batch = detect_cascade_events(result, policy=policy)

    assert len(batch) == 0


def test_detect_multiple_events_preserves_plane_order() -> None:
    result = make_result(
        activation_before=(0.1, 0.8),
        activation_after=(0.2, 0.6),
    )

    batch = detect_cascade_events(result)

    assert batch.plane_ids == ("a", "b")


def test_detect_attaches_kernel_metadata() -> None:
    result = make_result(
        activation_before=(0.1, 0.0),
        activation_after=(0.2, 0.0),
        interaction=(0.3, 0.0),
        rate=(0.1, 0.0),
        dt=0.5,
    )

    event = detect_cascade_events(
        result,
        metadata={"experiment": "demo"},
    ).events[0]

    assert event.metadata["experiment"] == "demo"
    assert event.metadata["interaction"] == pytest.approx(0.3)
    assert event.metadata["rate"] == pytest.approx(0.1)
    assert event.metadata["dt"] == pytest.approx(0.5)


def test_detect_event_time_matches_result_time() -> None:
    result = make_result(
        time=3.5,
        activation_before=(0.1, 0.0),
        activation_after=(0.2, 0.0),
    )

    batch = detect_cascade_events(result)

    assert batch.time == pytest.approx(3.5)
    assert batch.events[0].time == pytest.approx(3.5)


def test_detect_uses_custom_minimum_delta() -> None:
    result = make_result(
        activation_before=(0.1, 0.0),
        activation_after=(0.15, 0.0),
    )
    policy = CascadeEventPolicy(minimum_delta=0.1)

    assert len(detect_cascade_events(result, policy=policy)) == 0


def test_detect_dominant_source_plane() -> None:
    result = make_result(
        plane_ids=("a", "b", "c"),
        activation_before=(0.8, 0.4, 0.0),
        activation_after=(0.8, 0.4, 0.3),
        interaction=(0.0, 0.0, 0.3),
        rate=(0.0, 0.0, 0.3),
        clipped=(False, False, False),
        retained_by_hysteresis=(False, False, False),
    )
    coupling = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 0.0],
    ]

    event = detect_cascade_events(
        result,
        coupling_matrix=coupling,
    ).for_plane("c")[0]

    assert event.source_plane_id == "a"
    assert event.kind is CascadeEventKind.PROPAGATION


def test_detect_dominant_source_uses_weight_times_activation() -> None:
    result = make_result(
        plane_ids=("a", "b", "c"),
        activation_before=(0.2, 0.9, 0.0),
        activation_after=(0.2, 0.9, 0.3),
        interaction=(0.0, 0.0, 0.3),
        rate=(0.0, 0.0, 0.3),
        clipped=(False, False, False),
        retained_by_hysteresis=(False, False, False),
    )
    coupling = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
    ]

    event = detect_cascade_events(
        result,
        coupling_matrix=coupling,
    ).for_plane("c")[0]

    assert event.source_plane_id == "b"


def test_detect_ignores_self_coupling_as_source() -> None:
    result = make_result(
        activation_before=(0.2, 0.0),
        activation_after=(0.3, 0.0),
    )
    coupling = [
        [10.0, 0.0],
        [0.0, 0.0],
    ]

    event = detect_cascade_events(
        result,
        coupling_matrix=coupling,
    ).events[0]

    assert event.source_plane_id is None
    assert event.kind is CascadeEventKind.AMPLIFICATION


def test_detect_respects_minimum_source_influence() -> None:
    result = make_result(
        activation_before=(0.2, 0.0),
        activation_after=(0.2, 0.1),
    )
    coupling = [
        [0.0, 0.0],
        [0.1, 0.0],
    ]

    event = detect_cascade_events(
        result,
        coupling_matrix=coupling,
        minimum_source_influence=0.05,
    ).for_plane("b")[0]

    assert event.source_plane_id is None
    assert event.kind is CascadeEventKind.AMPLIFICATION


def test_detect_links_parent_from_source_event_ids() -> None:
    result = make_result(
        activation_before=(0.8, 0.0),
        activation_after=(0.8, 0.2),
    )
    coupling = [
        [0.0, 0.0],
        [0.5, 0.0],
    ]

    event = detect_cascade_events(
        result,
        coupling_matrix=coupling,
        source_event_ids={"a": "source-event"},
    ).for_plane("b")[0]

    assert event.source_plane_id == "a"
    assert event.parent_event_id == "source-event"


def test_detect_without_source_id_leaves_parent_empty() -> None:
    result = make_result(
        activation_before=(0.8, 0.0),
        activation_after=(0.8, 0.2),
    )
    coupling = [
        [0.0, 0.0],
        [0.5, 0.0],
    ]

    event = detect_cascade_events(
        result,
        coupling_matrix=coupling,
        source_event_ids={},
    ).for_plane("b")[0]

    assert event.source_plane_id == "a"
    assert event.parent_event_id is None


def test_detect_rejects_non_mapping_source_event_ids() -> None:
    with pytest.raises(
        CascadeEventError,
        match="source_event_ids must be a mapping",
    ):
        detect_cascade_events(
            make_result(),
            source_event_ids=[],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, -np.inf])
def test_detect_minimum_source_influence_must_be_valid(
    value: float,
) -> None:
    with pytest.raises(
        CascadeEventError,
        match="minimum_source_influence",
    ):
        detect_cascade_events(
            make_result(),
            minimum_source_influence=value,
        )


def test_detect_rejects_wrong_coupling_row_count() -> None:
    with pytest.raises(
        CascadeEventError,
        match="row count must match",
    ):
        detect_cascade_events(
            make_result(
                activation_before=(0.1, 0.0),
                activation_after=(0.2, 0.0),
            ),
            coupling_matrix=[[0.0, 0.0]],
        )


def test_detect_rejects_non_square_coupling_matrix() -> None:
    with pytest.raises(
        CascadeEventError,
        match="must be square",
    ):
        detect_cascade_events(
            make_result(
                activation_before=(0.1, 0.0),
                activation_after=(0.2, 0.0),
            ),
            coupling_matrix=[
                [0.0],
                [0.0],
            ],
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_detect_rejects_non_finite_source_influence(value: float) -> None:
    result = make_result(
        activation_before=(0.5, 0.0),
        activation_after=(0.5, 0.2),
    )

    with pytest.raises(
        CascadeEventError,
        match="must be finite",
    ):
        detect_cascade_events(
            result,
            coupling_matrix=[
                [0.0, 0.0],
                [value, 0.0],
            ],
        )


def test_detect_metadata_must_be_mapping() -> None:
    with pytest.raises(CascadeEventError, match="metadata must be a mapping"):
        detect_cascade_events(
            make_result(),
            metadata=[],  # type: ignore[arg-type]
        )


def test_detect_source_event_id_cannot_be_empty() -> None:
    result = make_result(
        activation_before=(0.8, 0.0),
        activation_after=(0.8, 0.2),
    )

    with pytest.raises(CascadeEventError, match="cannot be empty"):
        detect_cascade_events(
            result,
            coupling_matrix=[
                [0.0, 0.0],
                [0.5, 0.0],
            ],
            source_event_ids={"a": " "},
        )


# ---------------------------------------------------------------------------
# link_event_chain
# ---------------------------------------------------------------------------


def test_link_empty_chain() -> None:
    assert link_event_chain(()) == ()


def test_link_single_event_keeps_root() -> None:
    event = make_event(event_id="root")

    linked = link_event_chain((event,))

    assert linked == (event,)
    assert linked[0].parent_event_id is None


def test_link_event_chain_assigns_parents() -> None:
    first = make_event(event_id="e1", plane_id="a")
    second = make_event(event_id="e2", plane_id="b")
    third = make_event(event_id="e3", plane_id="c")

    linked = link_event_chain((first, second, third))

    assert linked[0] is first
    assert linked[1].parent_event_id == first.event_id
    assert linked[1].source_plane_id == "a"
    assert linked[2].parent_event_id == linked[1].event_id
    assert linked[2].source_plane_id == "b"


def test_link_event_chain_does_not_mutate_inputs() -> None:
    first = make_event(event_id="e1", plane_id="a")
    second = make_event(event_id="e2", plane_id="b")

    linked = link_event_chain((first, second))

    assert second.parent_event_id is None
    assert linked[1] is not second


def test_link_event_chain_regenerates_child_ids() -> None:
    first = make_event(event_id="e1", plane_id="a")
    second = make_event(event_id="e2", plane_id="b")

    linked = link_event_chain((first, second))

    assert linked[1].event_id != second.event_id


def test_link_event_chain_accepts_generator() -> None:
    events = (
        make_event(event_id=f"e{index}", plane_id=f"p{index}")
        for index in range(3)
    )

    linked = link_event_chain(events)

    assert len(linked) == 3
    assert linked[2].parent_event_id == linked[1].event_id


def test_link_event_chain_rejects_non_event() -> None:
    with pytest.raises(
        CascadeEventError,
        match=r"events\[1\] must be a CascadeEvent",
    ):
        link_event_chain(
            (
                make_event(event_id="e1"),
                object(),  # type: ignore[arg-type]
            )
        )
