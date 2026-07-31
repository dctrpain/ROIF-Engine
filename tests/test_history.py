"""Tests for immutable ROIF cascade history."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from roif.cascade_event import (
    CascadeEvent,
    CascadeEventBatch,
    CascadeEventKind,
    CascadeEventSeverity,
)
from roif.history import (
    CascadeHistory,
    CascadeHistoryError,
    build_history,
    merge_histories,
)


def make_event(
    *,
    event_id: str = "e1",
    time: float = 1.0,
    plane_id: str = "a",
    kind: CascadeEventKind | str = CascadeEventKind.AMPLIFICATION,
    severity: CascadeEventSeverity | str = CascadeEventSeverity.LOW,
    activation_before: float = 0.1,
    activation_after: float = 0.2,
    source_plane_id: str | None = None,
    parent_event_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CascadeEvent:
    return CascadeEvent(
        event_id=event_id,
        time=time,
        plane_id=plane_id,
        kind=kind,
        severity=severity,
        activation_before=activation_before,
        activation_after=activation_after,
        source_plane_id=source_plane_id,
        parent_event_id=parent_event_id,
        metadata={} if metadata is None else metadata,
    )


def make_chain() -> tuple[CascadeEvent, CascadeEvent, CascadeEvent]:
    root = make_event(
        event_id="root",
        time=1.0,
        plane_id="a",
        severity=CascadeEventSeverity.LOW,
    )
    child = make_event(
        event_id="child",
        time=2.0,
        plane_id="b",
        parent_event_id="root",
        source_plane_id="a",
        severity=CascadeEventSeverity.MODERATE,
    )
    grandchild = make_event(
        event_id="grandchild",
        time=3.0,
        plane_id="c",
        parent_event_id="child",
        source_plane_id="b",
        severity=CascadeEventSeverity.CRITICAL,
    )
    return root, child, grandchild


# ---------------------------------------------------------------------------
# Construction and basic protocol
# ---------------------------------------------------------------------------


def test_empty_history_defaults() -> None:
    history = CascadeHistory()

    assert len(history) == 0
    assert bool(history) is False
    assert tuple(history) == ()
    assert history.event_ids == ()
    assert history.plane_ids == ()
    assert history.start_time is None
    assert history.end_time is None
    assert history.duration == pytest.approx(0.0)
    assert history.root_events == ()
    assert history.leaf_events == ()
    assert history.critical_events == ()


def test_history_accepts_ordered_events() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert len(history) == 3
    assert tuple(history) == (root, child, grandchild)
    assert bool(history) is True


def test_history_is_frozen() -> None:
    history = CascadeHistory()

    with pytest.raises(FrozenInstanceError):
        history.events = ()  # type: ignore[misc]


def test_history_events_are_tuple_copied() -> None:
    root, child, _ = make_chain()
    source = [root, child]
    history = CascadeHistory(events=source)  # type: ignore[arg-type]

    source.clear()

    assert history.events == (root, child)


def test_history_metadata_is_trimmed_and_read_only() -> None:
    history = CascadeHistory(metadata={"  source  ": "test"})

    assert dict(history.metadata) == {"source": "test"}

    with pytest.raises(TypeError):
        history.metadata["source"] = "changed"  # type: ignore[index]


def test_history_metadata_input_is_copied() -> None:
    metadata = {"source": "original"}
    history = CascadeHistory(metadata=metadata)

    metadata["source"] = "changed"

    assert history.metadata["source"] == "original"


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_allow_external_parents_must_be_bool(value: object) -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="allow_external_parents must be a bool",
    ):
        CascadeHistory(
            allow_external_parents=value,  # type: ignore[arg-type]
        )


def test_history_rejects_non_event() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match=r"events\[0\] must be a CascadeEvent",
    ):
        CascadeHistory(events=(object(),))  # type: ignore[arg-type]


def test_history_rejects_decreasing_time() -> None:
    first = make_event(event_id="e1", time=2.0)
    second = make_event(event_id="e2", time=1.0)

    with pytest.raises(
        CascadeHistoryError,
        match="ordered by non-decreasing time",
    ):
        CascadeHistory(events=(first, second))


def test_history_accepts_equal_times() -> None:
    first = make_event(event_id="e1", time=1.0)
    second = make_event(event_id="e2", time=1.0)

    history = CascadeHistory(events=(first, second))

    assert history.events == (first, second)


def test_history_rejects_duplicate_event_ids() -> None:
    first = make_event(event_id="same", time=1.0)
    second = make_event(event_id="same", time=2.0)

    with pytest.raises(
        CascadeHistoryError,
        match="Duplicate event ID",
    ):
        CascadeHistory(events=(first, second))


def test_history_rejects_unknown_parent_by_default() -> None:
    event = make_event(
        event_id="child",
        parent_event_id="missing",
    )

    with pytest.raises(
        CascadeHistoryError,
        match="unknown or future parent",
    ):
        CascadeHistory(events=(event,))


def test_history_rejects_future_parent() -> None:
    child = make_event(
        event_id="child",
        time=1.0,
        parent_event_id="parent",
    )
    parent = make_event(
        event_id="parent",
        time=2.0,
    )

    with pytest.raises(
        CascadeHistoryError,
        match="unknown or future parent",
    ):
        CascadeHistory(events=(child, parent))


def test_history_accepts_external_parent_when_enabled() -> None:
    event = make_event(
        event_id="child",
        parent_event_id="external",
    )

    history = CascadeHistory(
        events=(event,),
        allow_external_parents=True,
    )

    assert history.events == (event,)


def test_history_contains_event_id() -> None:
    event = make_event(event_id=" e1 ")
    history = CascadeHistory(events=(event,))

    assert "e1" in history
    assert " e1 " in history
    assert "missing" not in history
    assert "" not in history
    assert object() not in history


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


def test_event_ids_property() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.event_ids == ("root", "child", "grandchild")


def test_plane_ids_are_unique_in_first_seen_order() -> None:
    events = (
        make_event(event_id="e1", time=1.0, plane_id="b"),
        make_event(event_id="e2", time=2.0, plane_id="a"),
        make_event(event_id="e3", time=3.0, plane_id="b"),
        make_event(event_id="e4", time=4.0, plane_id="c"),
    )
    history = CascadeHistory(events=events)

    assert history.plane_ids == ("b", "a", "c")


def test_start_end_and_duration() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.start_time == pytest.approx(1.0)
    assert history.end_time == pytest.approx(3.0)
    assert history.duration == pytest.approx(2.0)


def test_single_event_duration_is_zero() -> None:
    history = CascadeHistory(events=(make_event(),))

    assert history.duration == pytest.approx(0.0)


def test_root_events_for_internal_chain() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.root_events == (root,)


def test_external_parent_event_is_treated_as_root() -> None:
    event = make_event(
        event_id="child",
        parent_event_id="external",
    )
    history = CascadeHistory(
        events=(event,),
        allow_external_parents=True,
    )

    assert history.root_events == (event,)


def test_leaf_events_for_chain() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.leaf_events == (grandchild,)


def test_leaf_events_for_branching_tree() -> None:
    root = make_event(event_id="root", time=1.0)
    left = make_event(
        event_id="left",
        time=2.0,
        parent_event_id="root",
    )
    right = make_event(
        event_id="right",
        time=2.0,
        parent_event_id="root",
    )
    history = CascadeHistory(events=(root, left, right))

    assert history.leaf_events == (left, right)


def test_critical_events_property() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.critical_events == (grandchild,)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_get_known_event() -> None:
    event = make_event(event_id="event")
    history = CascadeHistory(events=(event,))

    assert history.get(" event ") is event


def test_get_unknown_event_raises() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="Unknown event ID",
    ):
        CascadeHistory().get("missing")


@pytest.mark.parametrize("value", ["", " ", "\t", "\n"])
def test_get_rejects_empty_event_id(value: str) -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="event_id cannot be empty",
    ):
        CascadeHistory().get(value)


def test_get_rejects_non_string_event_id() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="event_id must be a string",
    ):
        CascadeHistory().get(1)  # type: ignore[arg-type]


def test_index_of_known_event() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.index_of("child") == 1


def test_index_of_unknown_event_raises() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="Unknown event ID",
    ):
        CascadeHistory().index_of("missing")


# ---------------------------------------------------------------------------
# Immutable writes
# ---------------------------------------------------------------------------


def test_append_returns_new_history() -> None:
    first = make_event(event_id="e1", time=1.0)
    second = make_event(event_id="e2", time=2.0)
    history = CascadeHistory(events=(first,))

    updated = history.append(second)

    assert updated is not history
    assert history.events == (first,)
    assert updated.events == (first, second)


def test_append_preserves_configuration_and_metadata() -> None:
    event = make_event(
        event_id="e1",
        parent_event_id="external",
    )
    history = CascadeHistory(
        allow_external_parents=True,
        metadata={"source": "test"},
    )

    updated = history.append(event)

    assert updated.allow_external_parents is True
    assert dict(updated.metadata) == {"source": "test"}


def test_append_rejects_non_event() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="event must be a CascadeEvent",
    ):
        CascadeHistory().append(object())  # type: ignore[arg-type]


def test_append_revalidates_time_order() -> None:
    history = CascadeHistory(
        events=(make_event(event_id="e1", time=2.0),)
    )

    with pytest.raises(
        CascadeHistoryError,
        match="ordered by non-decreasing time",
    ):
        history.append(make_event(event_id="e2", time=1.0))


def test_extend_accepts_generator() -> None:
    history = CascadeHistory()
    events = (
        make_event(event_id=f"e{i}", time=float(i))
        for i in range(1, 4)
    )

    updated = history.extend(events)

    assert updated.event_ids == ("e1", "e2", "e3")


def test_extend_rejects_non_iterable() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="events must be iterable",
    ):
        CascadeHistory().extend(1)  # type: ignore[arg-type]


def test_extend_rejects_invalid_member() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match=r"events\[0\] must be a CascadeEvent",
    ):
        CascadeHistory().extend((object(),))  # type: ignore[arg-type]


def test_append_batch() -> None:
    first = make_event(event_id="e1", time=1.0)
    second = make_event(event_id="e2", time=2.0)
    batch = CascadeEventBatch(time=2.0, events=(second,))
    history = CascadeHistory(events=(first,))

    updated = history.append_batch(batch)

    assert updated.events == (first, second)


def test_append_batch_rejects_wrong_type() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="batch must be a CascadeEventBatch",
    ):
        CascadeHistory().append_batch(object())  # type: ignore[arg-type]


def test_with_metadata_merges_without_mutation() -> None:
    history = CascadeHistory(metadata={"a": 1})

    updated = history.with_metadata(a=2, b=3)

    assert dict(history.metadata) == {"a": 1}
    assert dict(updated.metadata) == {"a": 2, "b": 3}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_for_plane() -> None:
    events = (
        make_event(event_id="e1", time=1.0, plane_id="a"),
        make_event(event_id="e2", time=2.0, plane_id="b"),
        make_event(event_id="e3", time=3.0, plane_id="a"),
    )
    history = CascadeHistory(events=events)

    assert history.for_plane(" a ") == (events[0], events[2])


def test_for_plane_rejects_empty_id() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="plane_id cannot be empty",
    ):
        CascadeHistory().for_plane(" ")


def test_by_kind_accepts_enum() -> None:
    amplification = make_event(
        event_id="e1",
        kind=CascadeEventKind.AMPLIFICATION,
    )
    attenuation = make_event(
        event_id="e2",
        time=2.0,
        kind=CascadeEventKind.ATTENUATION,
    )
    history = CascadeHistory(
        events=(amplification, attenuation)
    )

    assert history.by_kind(
        CascadeEventKind.ATTENUATION
    ) == (attenuation,)


def test_by_kind_accepts_string() -> None:
    event = make_event(
        kind=CascadeEventKind.ACTIVATION,
    )
    history = CascadeHistory(events=(event,))

    assert history.by_kind("activation") == (event,)


def test_by_kind_rejects_unknown_kind() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="Unknown cascade event kind",
    ):
        CascadeHistory().by_kind("unknown")


def test_by_severity_accepts_enum_and_string() -> None:
    low = make_event(
        event_id="low",
        severity=CascadeEventSeverity.LOW,
    )
    high = make_event(
        event_id="high",
        time=2.0,
        severity=CascadeEventSeverity.HIGH,
    )
    history = CascadeHistory(events=(low, high))

    assert history.by_severity(
        CascadeEventSeverity.HIGH
    ) == (high,)
    assert history.by_severity("low") == (low,)


def test_by_severity_rejects_unknown_value() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="Unknown cascade event severity",
    ):
        CascadeHistory().by_severity("unknown")


def test_at_or_above_severity() -> None:
    trace = make_event(
        event_id="trace",
        severity=CascadeEventSeverity.TRACE,
    )
    moderate = make_event(
        event_id="moderate",
        time=2.0,
        severity=CascadeEventSeverity.MODERATE,
    )
    critical = make_event(
        event_id="critical",
        time=3.0,
        severity=CascadeEventSeverity.CRITICAL,
    )
    history = CascadeHistory(
        events=(trace, moderate, critical)
    )

    assert history.at_or_above_severity(
        CascadeEventSeverity.MODERATE
    ) == (moderate, critical)


def test_between_inclusive() -> None:
    events = (
        make_event(event_id="e1", time=1.0),
        make_event(event_id="e2", time=2.0),
        make_event(event_id="e3", time=3.0),
    )
    history = CascadeHistory(events=events)

    assert history.between(1.0, 2.0) == (
        events[0],
        events[1],
    )


def test_between_exclusive() -> None:
    events = (
        make_event(event_id="e1", time=1.0),
        make_event(event_id="e2", time=2.0),
        make_event(event_id="e3", time=3.0),
    )
    history = CascadeHistory(events=events)

    assert history.between(
        1.0,
        3.0,
        inclusive=False,
    ) == (events[1],)


@pytest.mark.parametrize("field_name", ["start_time", "end_time"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_between_times_must_be_finite_and_non_negative(
    field_name: str,
    value: float,
) -> None:
    kwargs = {
        "start_time": 0.0,
        "end_time": 1.0,
    }
    kwargs[field_name] = value

    with pytest.raises(
        CascadeHistoryError,
        match=field_name,
    ):
        CascadeHistory().between(**kwargs)


def test_between_rejects_reversed_range() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="start_time cannot exceed end_time",
    ):
        CascadeHistory().between(2.0, 1.0)


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_between_inclusive_must_be_bool(value: object) -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="inclusive must be a bool",
    ):
        CascadeHistory().between(
            0.0,
            1.0,
            inclusive=value,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Causal traversal
# ---------------------------------------------------------------------------


def test_children_of() -> None:
    root = make_event(event_id="root", time=1.0)
    left = make_event(
        event_id="left",
        time=2.0,
        parent_event_id="root",
    )
    right = make_event(
        event_id="right",
        time=2.0,
        parent_event_id="root",
    )
    history = CascadeHistory(events=(root, left, right))

    assert history.children_of("root") == (left, right)


def test_children_of_unknown_event_raises() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="Unknown event ID",
    ):
        CascadeHistory().children_of("missing")


def test_parent_of_root_is_none() -> None:
    root = make_event(event_id="root")
    history = CascadeHistory(events=(root,))

    assert history.parent_of("root") is None


def test_parent_of_child() -> None:
    root, child, _ = make_chain()
    history = CascadeHistory(events=(root, child))

    assert history.parent_of("child") is root


def test_parent_of_external_parent_returns_none() -> None:
    event = make_event(
        event_id="child",
        parent_event_id="external",
    )
    history = CascadeHistory(
        events=(event,),
        allow_external_parents=True,
    )

    assert history.parent_of("child") is None


def test_ancestors_of_root_is_empty() -> None:
    root = make_event(event_id="root")
    history = CascadeHistory(events=(root,))

    assert history.ancestors_of("root") == ()


def test_ancestors_of_grandchild_are_root_first() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.ancestors_of("grandchild") == (
        root,
        child,
    )


def test_ancestors_stop_at_external_parent() -> None:
    child = make_event(
        event_id="child",
        parent_event_id="external",
    )
    history = CascadeHistory(
        events=(child,),
        allow_external_parents=True,
    )

    assert history.ancestors_of("child") == ()


def test_descendants_of_root_are_breadth_first() -> None:
    root = make_event(event_id="root", time=1.0)
    left = make_event(
        event_id="left",
        time=2.0,
        parent_event_id="root",
    )
    right = make_event(
        event_id="right",
        time=2.0,
        parent_event_id="root",
    )
    leaf = make_event(
        event_id="leaf",
        time=3.0,
        parent_event_id="left",
    )
    history = CascadeHistory(
        events=(root, left, right, leaf)
    )

    assert history.descendants_of("root") == (
        left,
        right,
        leaf,
    )


def test_descendants_of_leaf_is_empty() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.descendants_of("grandchild") == ()


def test_lineage_of_includes_target() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.lineage_of("grandchild") == (
        root,
        child,
        grandchild,
    )


def test_causal_depth() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.causal_depth("root") == 0
    assert history.causal_depth("child") == 1
    assert history.causal_depth("grandchild") == 2


def test_maximum_causal_depth_empty_history() -> None:
    assert CascadeHistory().maximum_causal_depth() == 0


def test_maximum_causal_depth() -> None:
    root, child, grandchild = make_chain()
    history = CascadeHistory(events=(root, child, grandchild))

    assert history.maximum_causal_depth() == 2


# ---------------------------------------------------------------------------
# Counts and serialization
# ---------------------------------------------------------------------------


def test_count_by_plane() -> None:
    events = (
        make_event(event_id="e1", time=1.0, plane_id="a"),
        make_event(event_id="e2", time=2.0, plane_id="b"),
        make_event(event_id="e3", time=3.0, plane_id="a"),
    )
    history = CascadeHistory(events=events)

    counts = history.count_by_plane()

    assert dict(counts) == {"a": 2, "b": 1}

    with pytest.raises(TypeError):
        counts["a"] = 3  # type: ignore[index]


def test_count_by_kind() -> None:
    events = (
        make_event(
            event_id="e1",
            time=1.0,
            kind=CascadeEventKind.AMPLIFICATION,
        ),
        make_event(
            event_id="e2",
            time=2.0,
            kind=CascadeEventKind.AMPLIFICATION,
        ),
        make_event(
            event_id="e3",
            time=3.0,
            kind=CascadeEventKind.ATTENUATION,
        ),
    )
    history = CascadeHistory(events=events)

    assert dict(history.count_by_kind()) == {
        CascadeEventKind.AMPLIFICATION: 2,
        CascadeEventKind.ATTENUATION: 1,
    }


def test_count_by_severity() -> None:
    events = (
        make_event(
            event_id="e1",
            time=1.0,
            severity=CascadeEventSeverity.LOW,
        ),
        make_event(
            event_id="e2",
            time=2.0,
            severity=CascadeEventSeverity.LOW,
        ),
        make_event(
            event_id="e3",
            time=3.0,
            severity=CascadeEventSeverity.HIGH,
        ),
    )
    history = CascadeHistory(events=events)

    assert dict(history.count_by_severity()) == {
        CascadeEventSeverity.LOW: 2,
        CascadeEventSeverity.HIGH: 1,
    }


def test_history_as_dict() -> None:
    event = make_event(event_id="e1")
    history = CascadeHistory(
        events=(event,),
        metadata={"source": "test"},
    )

    data = history.as_dict()

    assert data == {
        "event_count": 1,
        "start_time": 1.0,
        "end_time": 1.0,
        "duration": 0.0,
        "allow_external_parents": False,
        "metadata": {"source": "test"},
        "events": [event.as_dict()],
    }


# ---------------------------------------------------------------------------
# build_history
# ---------------------------------------------------------------------------


def test_build_history_preserves_order_by_default() -> None:
    first = make_event(event_id="e1", time=1.0)
    second = make_event(event_id="e2", time=2.0)

    history = build_history((first, second))

    assert history.events == (first, second)


def test_build_history_sorts_by_time_stably() -> None:
    late = make_event(event_id="late", time=2.0)
    first_equal = make_event(event_id="first", time=1.0)
    second_equal = make_event(event_id="second", time=1.0)

    history = build_history(
        (late, first_equal, second_equal),
        sort_by_time=True,
    )

    assert history.events == (
        first_equal,
        second_equal,
        late,
    )


def test_build_history_accepts_generator() -> None:
    events = (
        make_event(event_id=f"e{i}", time=float(i))
        for i in range(1, 4)
    )

    history = build_history(events)

    assert history.event_ids == ("e1", "e2", "e3")


def test_build_history_rejects_non_iterable() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="events must be iterable",
    ):
        build_history(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_build_history_sort_by_time_must_be_bool(value: object) -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="sort_by_time must be a bool",
    ):
        build_history(
            (),
            sort_by_time=value,  # type: ignore[arg-type]
        )


def test_build_history_passes_external_parent_option() -> None:
    event = make_event(
        event_id="child",
        parent_event_id="external",
    )

    history = build_history(
        (event,),
        allow_external_parents=True,
    )

    assert history.allow_external_parents is True


def test_build_history_passes_metadata() -> None:
    history = build_history(
        (),
        metadata={"source": "builder"},
    )

    assert dict(history.metadata) == {"source": "builder"}


def test_build_history_rejects_invalid_member_after_sorting() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match=r"events\[0\] must be a CascadeEvent",
    ):
        build_history(
            (object(),),  # type: ignore[arg-type]
            sort_by_time=True,
        )


# ---------------------------------------------------------------------------
# merge_histories
# ---------------------------------------------------------------------------


def test_merge_histories_empty() -> None:
    merged = merge_histories()

    assert isinstance(merged, CascadeHistory)
    assert len(merged) == 0


def test_merge_histories_orders_events_by_time() -> None:
    late = CascadeHistory(
        events=(make_event(event_id="late", time=3.0),)
    )
    early = CascadeHistory(
        events=(make_event(event_id="early", time=1.0),)
    )
    middle = CascadeHistory(
        events=(make_event(event_id="middle", time=2.0),)
    )

    merged = merge_histories(late, early, middle)

    assert merged.event_ids == ("early", "middle", "late")


def test_merge_histories_preserves_equal_time_history_order() -> None:
    first = CascadeHistory(
        events=(
            make_event(event_id="a1", time=1.0),
            make_event(event_id="a2", time=1.0),
        )
    )
    second = CascadeHistory(
        events=(
            make_event(event_id="b1", time=1.0),
            make_event(event_id="b2", time=1.0),
        )
    )

    merged = merge_histories(first, second)

    assert merged.event_ids == ("a1", "a2", "b1", "b2")


def test_merge_histories_combines_metadata_last_wins() -> None:
    first = CascadeHistory(metadata={"a": 1, "shared": "first"})
    second = CascadeHistory(metadata={"b": 2, "shared": "second"})

    merged = merge_histories(first, second)

    assert dict(merged.metadata) == {
        "a": 1,
        "b": 2,
        "shared": "second",
    }


def test_merge_histories_explicit_metadata_overrides_source_merge() -> None:
    first = CascadeHistory(metadata={"a": 1})

    merged = merge_histories(
        first,
        metadata={"explicit": True},
    )

    assert dict(merged.metadata) == {"explicit": True}


def test_merge_histories_enables_external_parents_if_any_source_does() -> None:
    first = CascadeHistory()
    second = CascadeHistory(allow_external_parents=True)

    merged = merge_histories(first, second)

    assert merged.allow_external_parents is True


@pytest.mark.parametrize("value", [0, 1, "yes"])
def test_merge_histories_external_parent_override_must_be_bool(
    value: object,
) -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="allow_external_parents must be a bool",
    ):
        merge_histories(
            allow_external_parents=value,  # type: ignore[arg-type]
        )


def test_merge_histories_accepts_explicit_external_parent_override() -> None:
    history = CascadeHistory(allow_external_parents=True)

    merged = merge_histories(
        history,
        allow_external_parents=False,
    )

    assert merged.allow_external_parents is False


def test_merge_histories_rejects_non_history() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match=r"histories\[1\] must be a CascadeHistory",
    ):
        merge_histories(
            CascadeHistory(),
            object(),  # type: ignore[arg-type]
        )


def test_merge_histories_rejects_duplicate_event_ids() -> None:
    first = CascadeHistory(
        events=(make_event(event_id="same", time=1.0),)
    )
    second = CascadeHistory(
        events=(make_event(event_id="same", time=2.0),)
    )

    with pytest.raises(
        CascadeHistoryError,
        match="Duplicate event ID",
    ):
        merge_histories(first, second)


def test_merge_histories_validates_parent_order_after_sorting() -> None:
    child = make_event(
        event_id="child",
        time=2.0,
        parent_event_id="parent",
    )
    parent = make_event(
        event_id="parent",
        time=1.0,
    )

    child_history = CascadeHistory(
        events=(child,),
        allow_external_parents=True,
    )
    parent_history = CascadeHistory(events=(parent,))

    merged = merge_histories(
        child_history,
        parent_history,
        allow_external_parents=False,
    )

    assert merged.event_ids == ("parent", "child")


def test_merge_histories_rejects_unresolved_parent_when_forced_strict() -> None:
    child = make_event(
        event_id="child",
        parent_event_id="missing",
    )
    history = CascadeHistory(
        events=(child,),
        allow_external_parents=True,
    )

    with pytest.raises(
        CascadeHistoryError,
        match="unknown or future parent",
    ):
        merge_histories(
            history,
            allow_external_parents=False,
        )


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def test_history_metadata_must_be_mapping() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="metadata must be a mapping",
    ):
        CascadeHistory(metadata=[])  # type: ignore[arg-type]


def test_history_metadata_key_must_be_string() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="metadata key must be a string",
    ):
        CascadeHistory(
            metadata={1: "invalid"},  # type: ignore[dict-item]
        )


def test_history_metadata_key_cannot_be_empty() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="metadata key cannot be empty",
    ):
        CascadeHistory(metadata={" ": "invalid"})


def test_merge_metadata_must_be_mapping() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="metadata must be a mapping",
    ):
        merge_histories(
            metadata=[],  # type: ignore[arg-type]
        )


def test_build_metadata_must_be_mapping() -> None:
    with pytest.raises(
        CascadeHistoryError,
        match="metadata must be a mapping",
    ):
        build_history(
            (),
            metadata=[],  # type: ignore[arg-type]
        )
