"""
ROIF Memory 2.0 — Memory State Core.

Represents accumulated committed memory M_t as an immutable snapshot.

Boundary:
    M_t + committed trace -> M_{t+1}

No merge, delete, rewrite, policy selection, diagnosis, or biological claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from roif.history.memory_commitment import (
    CommittedMemoryTrace,
    memory_commitment_is_policy_free,
)
from roif.history.memory_integration import (
    MemoryIntegrationResult,
    memory_integration_is_policy_free,
)

SCHEMA_VERSION = "memory_state_v1"


class MemoryStateError(ValueError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _id(name: str, value: str) -> str:
    value = str(value).strip()
    if not value:
        raise MemoryStateError(f"{name} must not be empty")
    return value


def _trace(trace: CommittedMemoryTrace) -> CommittedMemoryTrace:
    if not isinstance(trace, CommittedMemoryTrace):
        raise MemoryStateError(
            "memory state accepts only CommittedMemoryTrace objects"
        )
    if not memory_commitment_is_policy_free(trace):
        raise MemoryStateError("committed trace must be policy-free")
    if trace.commitment_applied is not True:
        raise MemoryStateError("trace must be committed")
    if trace.source_memory_mutated is not False:
        raise MemoryStateError("trace must not mutate source memory")
    if trace.existing_memory_rewritten is not False:
        raise MemoryStateError("trace must not rewrite existing memory")
    return trace


@dataclass(frozen=True, slots=True)
class MemoryState:
    state_id: str
    revision: int
    traces: tuple[CommittedMemoryTrace, ...]
    parent_state_id: str | None = None
    appended_trace_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _id("state_id", self.state_id))

        revision = int(self.revision)
        if revision < 0:
            raise MemoryStateError("revision must be non-negative")
        object.__setattr__(self, "revision", revision)

        traces = tuple(_trace(trace) for trace in self.traces)
        ids = tuple(trace.trace_id for trace in traces)
        if len(set(ids)) != len(ids):
            raise MemoryStateError("memory state trace IDs must be unique")
        object.__setattr__(self, "traces", traces)

        if self.parent_state_id is not None:
            object.__setattr__(
                self,
                "parent_state_id",
                _id("parent_state_id", self.parent_state_id),
            )

        if self.appended_trace_id is not None:
            appended = _id("appended_trace_id", self.appended_trace_id)
            if appended not in ids:
                raise MemoryStateError(
                    "appended_trace_id must reference a trace in the state"
                )
            object.__setattr__(self, "appended_trace_id", appended)

        object.__setattr__(self, "metadata", _readonly(self.metadata))

    @property
    def trace_count(self) -> int:
        return len(self.traces)

    @property
    def trace_ids(self) -> tuple[str, ...]:
        return tuple(trace.trace_id for trace in self.traces)


@dataclass(frozen=True, slots=True)
class MemoryStateTransition:
    transition_id: str
    source_state_id: str
    target_state_id: str
    source_revision: int
    target_revision: int
    appended_trace_id: str
    source_trace_count: int
    target_trace_count: int
    integration_relation: str | None
    cluster_hint: str | None
    source_state_mutated: bool
    existing_trace_rewritten: bool
    trace_deleted: bool
    trace_merged: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "transition_id",
            "source_state_id",
            "target_state_id",
            "appended_trace_id",
        ):
            object.__setattr__(self, name, _id(name, getattr(self, name)))

        for name in (
            "source_revision",
            "target_revision",
            "source_trace_count",
            "target_trace_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise MemoryStateError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

        if self.target_revision != self.source_revision + 1:
            raise MemoryStateError(
                "target_revision must equal source_revision + 1"
            )
        if self.target_trace_count != self.source_trace_count + 1:
            raise MemoryStateError(
                "target_trace_count must equal source_trace_count + 1"
            )

        if self.integration_relation is not None:
            object.__setattr__(
                self,
                "integration_relation",
                _id("integration_relation", self.integration_relation),
            )
        if self.cluster_hint is not None:
            object.__setattr__(
                self,
                "cluster_hint",
                _id("cluster_hint", self.cluster_hint),
            )

        if self.source_state_mutated is not False:
            raise MemoryStateError("source state must not be mutated")
        if self.existing_trace_rewritten is not False:
            raise MemoryStateError("existing traces must not be rewritten")
        if self.trace_deleted is not False:
            raise MemoryStateError("traces must not be deleted")
        if self.trace_merged is not False:
            raise MemoryStateError("traces must not be merged")

        object.__setattr__(self, "metadata", _readonly(self.metadata))


def build_memory_state(
    *,
    state_id: str,
    traces: Iterable[CommittedMemoryTrace] = (),
    revision: int = 0,
    parent_state_id: str | None = None,
    appended_trace_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MemoryState:
    return MemoryState(
        state_id=state_id,
        revision=revision,
        traces=tuple(traces),
        parent_state_id=parent_state_id,
        appended_trace_id=appended_trace_id,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "memory_state_mode": "immutable_snapshot",
            "memory_mutated": False,
            "existing_trace_rewritten": False,
            "trace_deleted": False,
            "trace_merged": False,
            "action_selected": False,
            "policy_modified": False,
            "diagnosis_generated": False,
            "biological_memory_claimed": False,
            "causal_truth_inferred": False,
            **({} if metadata is None else dict(metadata)),
        },
    )


def empty_memory_state(
    *,
    state_id: str = "memory_state_0",
    metadata: Mapping[str, Any] | None = None,
) -> MemoryState:
    return build_memory_state(
        state_id=state_id,
        traces=(),
        revision=0,
        metadata=metadata,
    )


def get_memory_trace(
    state: MemoryState,
    trace_id: str,
) -> CommittedMemoryTrace | None:
    target = _id("trace_id", trace_id)
    for trace in state.traces:
        if trace.trace_id == target:
            return trace
    return None


def contains_memory_trace(state: MemoryState, trace_id: str) -> bool:
    return get_memory_trace(state, trace_id) is not None


def append_committed_trace(
    *,
    transition_id: str,
    target_state_id: str,
    state: MemoryState,
    trace: CommittedMemoryTrace,
    integration: MemoryIntegrationResult | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[MemoryState, MemoryStateTransition]:
    trace = _trace(trace)

    if contains_memory_trace(state, trace.trace_id):
        raise MemoryStateError("trace already exists in memory state")

    integration_relation = None
    cluster_hint = None

    if integration is not None:
        if not memory_integration_is_policy_free(integration):
            raise MemoryStateError("integration result must be policy-free")
        if integration.new_trace_id != trace.trace_id:
            raise MemoryStateError(
                "integration result must refer to the appended trace"
            )
        if integration.existing_trace_count != state.trace_count:
            raise MemoryStateError(
                "integration existing_trace_count must match source state"
            )
        integration_relation = integration.integration_relation
        cluster_hint = integration.cluster_hint

    target = build_memory_state(
        state_id=target_state_id,
        traces=state.traces + (trace,),
        revision=state.revision + 1,
        parent_state_id=state.state_id,
        appended_trace_id=trace.trace_id,
        metadata={
            "transition_id": transition_id,
            "source_state_id": state.state_id,
            "integration_relation": integration_relation,
            "cluster_hint": cluster_hint,
            **({} if metadata is None else dict(metadata)),
        },
    )

    transition = MemoryStateTransition(
        transition_id=transition_id,
        source_state_id=state.state_id,
        target_state_id=target.state_id,
        source_revision=state.revision,
        target_revision=target.revision,
        appended_trace_id=trace.trace_id,
        source_trace_count=state.trace_count,
        target_trace_count=target.trace_count,
        integration_relation=integration_relation,
        cluster_hint=cluster_hint,
        source_state_mutated=False,
        existing_trace_rewritten=False,
        trace_deleted=False,
        trace_merged=False,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "memory_state_transition_mode": "append_only",
            "source_state_mutated": False,
            "existing_trace_rewritten": False,
            "trace_deleted": False,
            "trace_merged": False,
            "action_selected": False,
            "policy_modified": False,
            "diagnosis_generated": False,
            "biological_memory_claimed": False,
            "causal_truth_inferred": False,
        },
    )
    return target, transition


def memory_state_is_policy_free(state: MemoryState) -> bool:
    return (
        state.metadata.get("action_selected") is False
        and state.metadata.get("policy_modified") is False
        and state.metadata.get("memory_mutated") is False
        and state.metadata.get("existing_trace_rewritten") is False
        and state.metadata.get("trace_deleted") is False
        and state.metadata.get("trace_merged") is False
        and all(memory_commitment_is_policy_free(t) for t in state.traces)
    )


def memory_state_transition_is_policy_free(
    transition: MemoryStateTransition,
) -> bool:
    return (
        transition.source_state_mutated is False
        and transition.existing_trace_rewritten is False
        and transition.trace_deleted is False
        and transition.trace_merged is False
        and transition.metadata.get("action_selected") is False
        and transition.metadata.get("policy_modified") is False
    )


def memory_state_signature(
    state: MemoryState,
) -> tuple[str, int, tuple[str, ...]]:
    return state.state_id, state.revision, state.trace_ids


__all__ = [
    "MemoryState",
    "MemoryStateError",
    "MemoryStateTransition",
    "SCHEMA_VERSION",
    "append_committed_trace",
    "build_memory_state",
    "contains_memory_trace",
    "empty_memory_state",
    "get_memory_trace",
    "memory_state_is_policy_free",
    "memory_state_signature",
    "memory_state_transition_is_policy_free",
]
