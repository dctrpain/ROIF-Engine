"""Tests for immutable ROIF state snapshots."""

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
from roif.history import CascadeHistory
from roif.plane_kernel import PlaneKernelResult
from roif.state_snapshot import (
    StateSnapshot,
    StateSnapshotError,
    create_state_snapshot,
)


def make_result(
    *,
    time: float = 1.0,
    plane_ids: tuple[str, ...] = ("a", "b"),
    activation_before: tuple[float, ...] = (0.1, 0.2),
    interaction: tuple[float, ...] = (0.1, -0.1),
    rate: tuple[float, ...] = (0.1, -0.1),
    activation_after: tuple[float, ...] = (0.2, 0.1),
    clipped: tuple[bool, ...] = (False, False),
    retained_by_hysteresis: tuple[bool, ...] = (False, False),
    dt: float = 1.0,
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


def make_event(
    *,
    event_id: str = "e1",
    time: float = 1.0,
    plane_id: str = "a",
    kind: CascadeEventKind | str = CascadeEventKind.AMPLIFICATION,
    severity: CascadeEventSeverity | str = CascadeEventSeverity.LOW,
    activation_before: float = 0.1,
    activation_after: float = 0.2,
    parent_event_id: str | None = None,
) -> CascadeEvent:
    return CascadeEvent(
        event_id=event_id,
        time=time,
        plane_id=plane_id,
        kind=kind,
        severity=severity,
        activation_before=activation_before,
        activation_after=activation_after,
        parent_event_id=parent_event_id,
    )


def make_snapshot(
    *,
    result: PlaneKernelResult | None = None,
    events: tuple[CascadeEvent, ...] | None = None,
    history: CascadeHistory | None = None,
    time: float | None = None,
    step_index: int | None = 1,
    snapshot_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StateSnapshot:
    kernel_result = make_result() if result is None else result

    if events is None:
        events = (
            make_event(
                time=kernel_result.time,
                activation_before=float(
                    kernel_result.activation_before[0]
                ),
                activation_after=float(
                    kernel_result.activation_after[0]
                ),
            ),
        )

    batch = CascadeEventBatch(
        time=kernel_result.time,
        events=events,
    )

    if history is None:
        history = CascadeHistory(events=events)

    return StateSnapshot(
        kernel_result=kernel_result,
        event_batch=batch,
        history=history,
        time=time,
        step_index=step_index,
        snapshot_id=snapshot_id,
        metadata={} if metadata is None else metadata,
    )


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_snapshot_accepts_coherent_components() -> None:
    snapshot = make_snapshot()

    assert isinstance(snapshot, StateSnapshot)
    assert snapshot.time == pytest.approx(1.0)
    assert snapshot.step_index == 1


def test_time_defaults_to_kernel_result_time() -> None:
    snapshot = make_snapshot(time=None)

    assert snapshot.time == pytest.approx(
        snapshot.kernel_result.time
    )


def test_explicit_equal_time_is_accepted() -> None:
    snapshot = make_snapshot(time=1.0)

    assert snapshot.time == pytest.approx(1.0)


def test_time_within_tolerance_is_accepted() -> None:
    snapshot = make_snapshot(time=1.0 + 5e-13)

    assert snapshot.time == pytest.approx(1.0 + 5e-13)


def test_snapshot_is_frozen() -> None:
    snapshot = make_snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.time = 2.0  # type: ignore[misc]


def test_rejects_wrong_kernel_result_type() -> None:
    event = make_event()
    batch = CascadeEventBatch(time=1.0, events=(event,))
    history = CascadeHistory(events=(event,))

    with pytest.raises(
        StateSnapshotError,
        match="kernel_result must be a PlaneKernelResult",
    ):
        StateSnapshot(
            kernel_result=object(),  # type: ignore[arg-type]
            event_batch=batch,
            history=history,
        )


def test_rejects_wrong_event_batch_type() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="event_batch must be a CascadeEventBatch",
    ):
        StateSnapshot(
            kernel_result=make_result(),
            event_batch=object(),  # type: ignore[arg-type]
            history=CascadeHistory(),
        )


def test_rejects_wrong_history_type() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="history must be a CascadeHistory",
    ):
        StateSnapshot(
            kernel_result=make_result(),
            event_batch=CascadeEventBatch(time=1.0),
            history=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_explicit_time_must_be_finite_and_non_negative(
    value: float,
) -> None:
    with pytest.raises(StateSnapshotError, match="time"):
        make_snapshot(time=value)


def test_explicit_time_must_be_numeric() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="time must be a real number",
    ):
        make_snapshot(time="bad")  # type: ignore[arg-type]


def test_rejects_time_different_from_kernel_result() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="time must match kernel_result.time",
    ):
        make_snapshot(time=2.0)


def test_rejects_batch_time_different_from_snapshot() -> None:
    result = make_result(time=1.0)
    batch = CascadeEventBatch(time=2.0)
    history = CascadeHistory()

    with pytest.raises(
        StateSnapshotError,
        match="time must match event_batch.time",
    ):
        StateSnapshot(
            kernel_result=result,
            event_batch=batch,
            history=history,
        )


@pytest.mark.parametrize("value", [True, False, 1.5, "1", object()])
def test_step_index_must_be_integer_or_none(value: object) -> None:
    with pytest.raises(
        StateSnapshotError,
        match="step_index must be an integer or None",
    ):
        make_snapshot(step_index=value)  # type: ignore[arg-type]


def test_step_index_none_is_accepted() -> None:
    snapshot = make_snapshot(step_index=None)

    assert snapshot.step_index is None


def test_step_index_zero_is_accepted() -> None:
    snapshot = make_snapshot(step_index=0)

    assert snapshot.step_index == 0


def test_negative_step_index_is_rejected() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="step_index must be non-negative",
    ):
        make_snapshot(step_index=-1)


def test_history_cannot_extend_beyond_snapshot_time() -> None:
    future = make_event(event_id="future", time=2.0)
    history = CascadeHistory(events=(future,))

    with pytest.raises(
        StateSnapshotError,
        match="history cannot contain events later",
    ):
        make_snapshot(events=(), history=history)


def test_current_event_must_be_present_in_history() -> None:
    event = make_event(event_id="current")
    history = CascadeHistory()

    with pytest.raises(
        StateSnapshotError,
        match="is missing from history",
    ):
        make_snapshot(events=(event,), history=history)


def test_current_event_must_equal_historical_event() -> None:
    current = make_event(
        event_id="same",
        severity=CascadeEventSeverity.LOW,
    )
    historical = make_event(
        event_id="same",
        severity=CascadeEventSeverity.HIGH,
    )
    history = CascadeHistory(events=(historical,))

    with pytest.raises(
        StateSnapshotError,
        match="contains a different event",
    ):
        make_snapshot(events=(current,), history=history)


def test_empty_batch_and_empty_history_are_valid() -> None:
    snapshot = make_snapshot(
        events=(),
        history=CascadeHistory(),
    )

    assert snapshot.event_count == 0
    assert snapshot.history_event_count == 0


def test_history_may_contain_earlier_events() -> None:
    earlier = make_event(
        event_id="earlier",
        time=0.5,
        activation_before=0.0,
        activation_after=0.1,
    )
    current = make_event(event_id="current", time=1.0)
    history = CascadeHistory(events=(earlier, current))

    snapshot = make_snapshot(
        events=(current,),
        history=history,
    )

    assert snapshot.history.events == (earlier, current)


# ---------------------------------------------------------------------------
# Snapshot identifiers and metadata
# ---------------------------------------------------------------------------


def test_generated_snapshot_id_has_prefix() -> None:
    snapshot = make_snapshot(snapshot_id=None)

    assert snapshot.snapshot_id.startswith("snap_")


def test_generated_snapshot_id_is_deterministic() -> None:
    first = make_snapshot(snapshot_id=None)
    second = make_snapshot(snapshot_id=None)

    assert first.snapshot_id == second.snapshot_id


def test_generated_snapshot_id_changes_with_step_index() -> None:
    first = make_snapshot(step_index=1, snapshot_id=None)
    second = make_snapshot(step_index=2, snapshot_id=None)

    assert first.snapshot_id != second.snapshot_id


def test_generated_snapshot_id_changes_with_events() -> None:
    first_event = make_event(event_id="e1")
    second_event = make_event(event_id="e2")

    first = make_snapshot(
        events=(first_event,),
        history=CascadeHistory(events=(first_event,)),
    )
    second = make_snapshot(
        events=(second_event,),
        history=CascadeHistory(events=(second_event,)),
    )

    assert first.snapshot_id != second.snapshot_id


def test_explicit_snapshot_id_is_trimmed() -> None:
    snapshot = make_snapshot(snapshot_id="  custom  ")

    assert snapshot.snapshot_id == "custom"


@pytest.mark.parametrize("value", ["", " ", "\t", "\n"])
def test_snapshot_id_cannot_be_empty(value: str) -> None:
    with pytest.raises(
        StateSnapshotError,
        match="snapshot_id cannot be empty",
    ):
        make_snapshot(snapshot_id=value)


def test_snapshot_id_must_be_string() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="snapshot_id must be a string",
    ):
        make_snapshot(snapshot_id=123)  # type: ignore[arg-type]


def test_metadata_is_trimmed_and_read_only() -> None:
    snapshot = make_snapshot(
        metadata={"  source  ": "test"}
    )

    assert dict(snapshot.metadata) == {"source": "test"}

    with pytest.raises(TypeError):
        snapshot.metadata["source"] = "changed"  # type: ignore[index]


def test_metadata_input_is_copied() -> None:
    metadata = {"source": "original"}
    snapshot = make_snapshot(metadata=metadata)

    metadata["source"] = "changed"

    assert snapshot.metadata["source"] == "original"


def test_metadata_must_be_mapping() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="metadata must be a mapping",
    ):
        make_snapshot(metadata=[])  # type: ignore[arg-type]


def test_metadata_key_must_be_string() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="metadata key must be a string",
    ):
        make_snapshot(
            metadata={1: "bad"},  # type: ignore[dict-item]
        )


def test_metadata_key_cannot_be_empty() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="metadata key cannot be empty",
    ):
        make_snapshot(metadata={" ": "bad"})


def test_with_metadata_returns_independent_snapshot() -> None:
    snapshot = make_snapshot(metadata={"a": 1})

    updated = snapshot.with_metadata(a=2, b=3)

    assert updated is not snapshot
    assert dict(snapshot.metadata) == {"a": 1}
    assert dict(updated.metadata) == {"a": 2, "b": 3}


def test_with_metadata_preserves_snapshot_id() -> None:
    snapshot = make_snapshot(snapshot_id="fixed")

    updated = snapshot.with_metadata(note="added")

    assert updated.snapshot_id == "fixed"


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


def test_plane_ids_delegate_to_kernel_result() -> None:
    snapshot = make_snapshot()

    assert snapshot.plane_ids == ("a", "b")


def test_plane_count_delegate_to_kernel_result() -> None:
    snapshot = make_snapshot()

    assert snapshot.plane_count == 2


def test_event_count() -> None:
    first = make_event(event_id="e1", plane_id="a")
    second = make_event(
        event_id="e2",
        plane_id="b",
        activation_before=0.2,
        activation_after=0.1,
    )
    history = CascadeHistory(events=(first, second))
    snapshot = make_snapshot(
        events=(first, second),
        history=history,
    )

    assert snapshot.event_count == 2


def test_history_event_count_includes_earlier_events() -> None:
    earlier = make_event(event_id="earlier", time=0.5)
    current = make_event(event_id="current", time=1.0)
    history = CascadeHistory(events=(earlier, current))
    snapshot = make_snapshot(
        events=(current,),
        history=history,
    )

    assert snapshot.history_event_count == 2


def test_changed_true() -> None:
    assert make_snapshot().changed is True


def test_changed_false() -> None:
    result = make_result(
        activation_after=(0.1, 0.2),
        interaction=(0.0, 0.0),
        rate=(0.0, 0.0),
    )
    snapshot = make_snapshot(
        result=result,
        events=(),
        history=CascadeHistory(),
    )

    assert snapshot.changed is False


def test_changed_plane_ids() -> None:
    snapshot = make_snapshot()

    assert snapshot.changed_plane_ids == ("a", "b")


def test_has_events_true() -> None:
    assert make_snapshot().has_events is True


def test_has_events_false() -> None:
    snapshot = make_snapshot(
        events=(),
        history=CascadeHistory(),
    )

    assert snapshot.has_events is False


def test_current_events_returns_batch_tuple() -> None:
    snapshot = make_snapshot()

    assert snapshot.current_events is snapshot.event_batch.events


def test_activation_before_mapping() -> None:
    snapshot = make_snapshot()

    assert dict(snapshot.activation_before) == {
        "a": pytest.approx(0.1),
        "b": pytest.approx(0.2),
    }


def test_activation_after_mapping() -> None:
    snapshot = make_snapshot()

    assert dict(snapshot.activation_after) == {
        "a": pytest.approx(0.2),
        "b": pytest.approx(0.1),
    }


def test_activation_mappings_are_read_only() -> None:
    snapshot = make_snapshot()
    before = snapshot.activation_before
    after = snapshot.activation_after

    with pytest.raises(TypeError):
        before["a"] = 0.5  # type: ignore[index]

    with pytest.raises(TypeError):
        after["a"] = 0.5  # type: ignore[index]


def test_activation_for_known_plane() -> None:
    snapshot = make_snapshot()

    assert snapshot.activation_for("a") == pytest.approx(0.2)


def test_activation_for_unknown_plane_delegates_error() -> None:
    with pytest.raises(ValueError, match="Unknown plane_id"):
        make_snapshot().activation_for("missing")


def test_events_for_plane() -> None:
    first = make_event(event_id="e1", plane_id="a")
    second = make_event(
        event_id="e2",
        plane_id="b",
        activation_before=0.2,
        activation_after=0.1,
    )
    history = CascadeHistory(events=(first, second))
    snapshot = make_snapshot(
        events=(first, second),
        history=history,
    )

    assert snapshot.events_for_plane(" a ") == (first,)


def test_events_for_plane_returns_empty_tuple() -> None:
    snapshot = make_snapshot()

    assert snapshot.events_for_plane("b") == ()


def test_events_for_plane_requires_string() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="plane_id must be a string",
    ):
        make_snapshot().events_for_plane(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", " ", "\t", "\n"])
def test_events_for_plane_rejects_empty_id(value: str) -> None:
    with pytest.raises(
        StateSnapshotError,
        match="plane_id cannot be empty",
    ):
        make_snapshot().events_for_plane(value)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_as_dict_top_level_fields() -> None:
    snapshot = make_snapshot(
        snapshot_id="snap_custom",
        metadata={"source": "test"},
    )

    data = snapshot.as_dict()

    assert data["snapshot_id"] == "snap_custom"
    assert data["time"] == pytest.approx(1.0)
    assert data["step_index"] == 1
    assert data["plane_count"] == 2
    assert data["event_count"] == 1
    assert data["history_event_count"] == 1
    assert data["changed"] is True
    assert data["changed_plane_ids"] == ["a", "b"]
    assert data["metadata"] == {"source": "test"}


def test_as_dict_contains_component_serializations() -> None:
    snapshot = make_snapshot()
    data = snapshot.as_dict()

    assert data["kernel_result"] == (
        snapshot.kernel_result.as_dict()
    )
    assert data["event_batch"] == (
        snapshot.event_batch.as_dict()
    )
    assert data["history"] == snapshot.history.as_dict()


def test_as_dict_returns_independent_metadata_dict() -> None:
    snapshot = make_snapshot(metadata={"source": "test"})
    data = snapshot.as_dict()

    data["metadata"]["source"] = "changed"

    assert snapshot.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# create_state_snapshot
# ---------------------------------------------------------------------------


def test_create_snapshot_with_empty_prior_history() -> None:
    result = make_result()
    event = make_event()
    batch = CascadeEventBatch(time=1.0, events=(event,))

    snapshot = create_state_snapshot(result, batch)

    assert snapshot.history.events == (event,)
    assert snapshot.current_events == (event,)


def test_create_snapshot_appends_to_prior_history() -> None:
    earlier = make_event(event_id="earlier", time=0.5)
    current = make_event(event_id="current", time=1.0)
    prior = CascadeHistory(events=(earlier,))
    batch = CascadeEventBatch(time=1.0, events=(current,))

    snapshot = create_state_snapshot(
        make_result(),
        batch,
        prior,
    )

    assert snapshot.history.events == (earlier, current)
    assert prior.events == (earlier,)


def test_create_snapshot_accepts_empty_batch() -> None:
    snapshot = create_state_snapshot(
        make_result(),
        CascadeEventBatch(time=1.0),
    )

    assert len(snapshot.history) == 0


def test_create_snapshot_passes_optional_fields() -> None:
    snapshot = create_state_snapshot(
        make_result(),
        CascadeEventBatch(time=1.0),
        step_index=3,
        snapshot_id=" custom ",
        metadata={"source": "factory"},
    )

    assert snapshot.step_index == 3
    assert snapshot.snapshot_id == "custom"
    assert dict(snapshot.metadata) == {"source": "factory"}


def test_create_snapshot_rejects_wrong_result_type() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="kernel_result must be a PlaneKernelResult",
    ):
        create_state_snapshot(
            object(),  # type: ignore[arg-type]
            CascadeEventBatch(time=1.0),
        )


def test_create_snapshot_rejects_wrong_batch_type() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="event_batch must be a CascadeEventBatch",
    ):
        create_state_snapshot(
            make_result(),
            object(),  # type: ignore[arg-type]
        )


def test_create_snapshot_rejects_wrong_history_type() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="history_before must be a CascadeHistory or None",
    ):
        create_state_snapshot(
            make_result(),
            CascadeEventBatch(time=1.0),
            object(),  # type: ignore[arg-type]
        )


def test_create_snapshot_rejects_future_prior_history() -> None:
    future = make_event(event_id="future", time=2.0)
    prior = CascadeHistory(events=(future,))

    with pytest.raises(
        StateSnapshotError,
        match="history_before cannot contain events later",
    ):
        create_state_snapshot(
            make_result(),
            CascadeEventBatch(time=1.0),
            prior,
        )


def test_create_snapshot_rejects_batch_time_mismatch() -> None:
    with pytest.raises(
        StateSnapshotError,
        match="time must match event_batch.time",
    ):
        create_state_snapshot(
            make_result(time=1.0),
            CascadeEventBatch(time=2.0),
        )


def test_create_snapshot_wraps_history_append_failure() -> None:
    existing = make_event(event_id="same", time=0.5)
    duplicate = make_event(event_id="same", time=1.0)
    prior = CascadeHistory(events=(existing,))
    batch = CascadeEventBatch(time=1.0, events=(duplicate,))

    with pytest.raises(
        StateSnapshotError,
        match="Cannot append event_batch to history_before",
    ):
        create_state_snapshot(
            make_result(),
            batch,
            prior,
        )


def test_create_snapshot_rejects_parent_missing_from_strict_history() -> None:
    child = make_event(
        event_id="child",
        parent_event_id="missing",
    )
    batch = CascadeEventBatch(time=1.0, events=(child,))

    with pytest.raises(
        StateSnapshotError,
        match="Cannot append event_batch to history_before",
    ):
        create_state_snapshot(
            make_result(),
            batch,
            CascadeHistory(),
        )


def test_create_snapshot_allows_external_parent_when_history_allows_it() -> None:
    child = make_event(
        event_id="child",
        parent_event_id="external",
    )
    batch = CascadeEventBatch(time=1.0, events=(child,))
    prior = CascadeHistory(allow_external_parents=True)

    snapshot = create_state_snapshot(
        make_result(),
        batch,
        prior,
    )

    assert snapshot.history.events == (child,)
    assert snapshot.history.allow_external_parents is True
