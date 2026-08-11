
"""
ROIF Controller History Adapter

Bridges predictive-control ExperienceTrace records into the existing
roif.history subsystem.

Safe integration path:

    ExperienceTrace
        -> HistoryEvent[]
        -> StructuralSignature

Architectural boundaries:

    ExperienceTrace != HistoryPattern
    MemoryScar != StructuralSignature
    StructuralSignature != PredictivePreload
    PredictionError != IrreversibleChange

This adapter deliberately does not infer IrreversibleChange and does not mutate
PredictiveState, controller policy, ControllerMemory, or graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from roif.experience_trace import (
    ExperienceActionKind,
    ExperienceTrace,
    StabilizationOutcome,
    trace_is_learning_free,
)
from roif.history.event import (
    HistoryEvent,
    HistoryEventKind,
    HistoryTarget,
    HistoryTargetKind,
    ReversibilityClass,
    StateDelta,
    TimeScale,
)
from roif.history.structural_signature import StructuralSignature


class ControllerHistoryAdapterError(RuntimeError):
    pass


class InvalidControllerHistoryRecordError(ControllerHistoryAdapterError):
    pass


class ControllerHistoryEventRole(str, Enum):
    INITIAL_STATE = "initial_state"
    DISTURBANCE = "disturbance"
    DECISION = "decision"
    PREDICTION = "prediction"
    OBSERVATION = "observation"
    PREDICTION_ERROR = "prediction_error"
    STABILIZATION = "stabilization"


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _enum_member(enum_type: type, names: Sequence[str]):
    for name in names:
        if hasattr(enum_type, name):
            return getattr(enum_type, name)
    members = tuple(enum_type)
    if not members:
        raise ControllerHistoryAdapterError(
            f"{enum_type.__name__} has no enum members"
        )
    return members[0]


def _event_kind(role: ControllerHistoryEventRole) -> HistoryEventKind:
    preferences = {
        ControllerHistoryEventRole.INITIAL_STATE:
            ("STATE_CHANGE", "OBSERVATION", "OTHER"),
        ControllerHistoryEventRole.DISTURBANCE:
            ("LOAD_CHANGE", "PERTURBATION", "EXTERNAL", "STATE_CHANGE", "OTHER"),
        ControllerHistoryEventRole.DECISION:
            ("CONTROL", "ACTION", "INTERVENTION", "STATE_CHANGE", "OTHER"),
        ControllerHistoryEventRole.PREDICTION:
            ("FORECAST", "PREDICTION", "OBSERVATION", "STATE_CHANGE", "OTHER"),
        ControllerHistoryEventRole.OBSERVATION:
            ("OBSERVATION", "STATE_CHANGE", "OTHER"),
        ControllerHistoryEventRole.PREDICTION_ERROR:
            ("ERROR", "OBSERVATION", "STATE_CHANGE", "OTHER"),
        ControllerHistoryEventRole.STABILIZATION:
            ("ADAPTATION", "STATE_CHANGE", "OTHER"),
    }
    return _enum_member(HistoryEventKind, preferences[role])


def _target_kind() -> HistoryTargetKind:
    return _enum_member(
        HistoryTargetKind,
        ("SYSTEM", "CONTROLLER", "CHANNEL", "NODE", "OTHER"),
    )


def _unknown_reversibility() -> ReversibilityClass:
    return _enum_member(
        ReversibilityClass,
        ("UNKNOWN", "REVERSIBLE", "PARTIAL", "IRREVERSIBLE"),
    )


def _fast_time_scale() -> TimeScale:
    return _enum_member(
        TimeScale,
        ("FAST", "SHORT", "IMMEDIATE", "EVENT"),
    )


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _make_target(trace: ExperienceTrace) -> HistoryTarget:
    """
    Construct HistoryTarget against the current public dataclass API.

    HistoryTarget's exact field names are discovered at runtime so this adapter
    stays compatible with the existing history subsystem instead of inventing
    a second target model.
    """
    import inspect

    params = inspect.signature(HistoryTarget).parameters
    kwargs: dict[str, Any] = {}

    if "kind" in params:
        kwargs["kind"] = _target_kind()
    elif "target_kind" in params:
        kwargs["target_kind"] = _target_kind()

    target_id = "controller:" + (
        trace.identity.sequence_id or trace.identity.trace_id
    )

    for name in ("target_id", "id", "entity_id", "object_id", "reference_id"):
        if name in params:
            kwargs[name] = target_id
            break

    if "label" in params:
        kwargs["label"] = "predictive_controller"

    if "metadata" in params:
        kwargs["metadata"] = {
            "trace_id": trace.identity.trace_id,
            "sequence_id": trace.identity.sequence_id,
            "controller_history_adapter": True,
            "external_expected_label_used": False,
        }

    try:
        return HistoryTarget(**kwargs)
    except TypeError as exc:
        raise ControllerHistoryAdapterError(
            "Unable to construct HistoryTarget from current public API."
        ) from exc


@dataclass(frozen=True, slots=True)
class ControllerHistoryRecord:
    trace_id: str
    sequence_id: str | None
    pattern_id: str | None
    events: tuple[HistoryEvent, ...]
    signature: StructuralSignature
    source_trace: ExperienceTrace
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise InvalidControllerHistoryRecordError(
                "trace_id must not be empty"
            )
        object.__setattr__(self, "events", tuple(self.events))
        if not self.events:
            raise InvalidControllerHistoryRecordError(
                "record requires at least one HistoryEvent"
            )
        if self.source_trace.identity.trace_id != self.trace_id:
            raise InvalidControllerHistoryRecordError(
                "source_trace does not match trace_id"
            )
        object.__setattr__(self, "metadata", _readonly(self.metadata))


def _state_deltas(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    role: ControllerHistoryEventRole,
) -> tuple[StateDelta, ...]:
    return tuple(
        StateDelta(
            quantity=str(key),
            before=before.get(key),
            after=after.get(key),
            channel=str(key),
            metadata={
                "controller_history_role": role.value,
                "external_expected_label_used": False,
            },
        )
        for key in sorted(set(before) | set(after))
    )


def _scalar_delta(
    quantity: str,
    before: Any,
    after: Any,
    role: ControllerHistoryEventRole,
    channel: str | None = None,
) -> StateDelta:
    return StateDelta(
        quantity=quantity,
        before=before,
        after=after,
        channel=channel,
        metadata={
            "controller_history_role": role.value,
            "external_expected_label_used": False,
        },
    )


def _event_time(trace: ExperienceTrace, offset: float) -> float:
    base = (
        trace.identity.timestamp_start
        if trace.identity.timestamp_start is not None
        else float(trace.identity.episode_index)
    )
    return _finite(base + offset, "history_event_time")


def _build_event(
    trace: ExperienceTrace,
    role: ControllerHistoryEventRole,
    target: HistoryTarget,
    offset: float,
    deltas: Sequence[StateDelta],
    description: str,
    *,
    confidence: float = 1.0,
    cause_event_ids: Sequence[str] = (),
) -> HistoryEvent:
    return HistoryEvent(
        kind=_event_kind(role),
        time=_event_time(trace, offset),
        target=target,
        deltas=tuple(deltas),
        cause_event_ids=tuple(cause_event_ids),
        plane_ids=("controller",),
        agent_ids=("predictive_controller",),
        time_scale=_fast_time_scale(),
        characteristic_time=0.0,
        persistence=0.0,
        reversibility=_unknown_reversibility(),
        capacity_effect=0.0,
        confidence=max(0.0, min(1.0, float(confidence))),
        description=description,
        metadata={
            "controller_history_role": role.value,
            "trace_id": trace.identity.trace_id,
            "sequence_id": trace.identity.sequence_id,
            "learning_applied": False,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def experience_trace_to_history_events(
    trace: ExperienceTrace,
) -> tuple[HistoryEvent, ...]:
    if not trace_is_learning_free(trace):
        raise ControllerHistoryAdapterError(
            "adapter accepts only learning-free ExperienceTrace"
        )

    target = _make_target(trace)
    events: list[HistoryEvent] = []

    initial = _build_event(
        trace,
        ControllerHistoryEventRole.INITIAL_STATE,
        target,
        0.00,
        _state_deltas(
            {},
            trace.initial_state.values,
            role=ControllerHistoryEventRole.INITIAL_STATE,
        ),
        "Predictive-control episode initial state.",
        confidence=1.0 - trace.initial_state.uncertainty,
    )
    events.append(initial)

    disturbance = _build_event(
        trace,
        ControllerHistoryEventRole.DISTURBANCE,
        target,
        0.10,
        tuple(
            _scalar_delta(
                f"disturbance:{key}",
                0.0,
                value,
                ControllerHistoryEventRole.DISTURBANCE,
                str(key),
            )
            for key, value in sorted(trace.disturbance.components.items())
        ),
        "Disturbance estimate available to predictive controller.",
        confidence=trace.disturbance.confidence,
        cause_event_ids=(initial.event_id,),
    )
    events.append(disturbance)

    selected_id = (
        None
        if trace.selected_candidate is None
        else trace.selected_candidate.candidate_id
    )
    decision = _build_event(
        trace,
        ControllerHistoryEventRole.DECISION,
        target,
        0.20,
        (
            _scalar_delta(
                "decision_kind",
                None,
                trace.action_kind.value,
                ControllerHistoryEventRole.DECISION,
            ),
            _scalar_delta(
                "selected_candidate_id",
                None,
                selected_id,
                ControllerHistoryEventRole.DECISION,
            ),
            _scalar_delta(
                "stabilization_demand",
                None,
                trace.demand.total,
                ControllerHistoryEventRole.DECISION,
            ),
            _scalar_delta(
                "control_reserve_available",
                None,
                trace.reserve.available,
                ControllerHistoryEventRole.DECISION,
            ),
        ),
        "Predictive controller decision.",
        cause_event_ids=(disturbance.event_id,),
    )
    events.append(decision)

    prediction: HistoryEvent | None = None
    if trace.selected_prediction is not None:
        prediction = _build_event(
            trace,
            ControllerHistoryEventRole.PREDICTION,
            target,
            0.30,
            _state_deltas(
                trace.initial_state.values,
                trace.selected_prediction.predicted_state.values,
                role=ControllerHistoryEventRole.PREDICTION,
            ),
            "Selected candidate predicted outcome.",
            confidence=trace.selected_prediction.confidence,
            cause_event_ids=(decision.event_id,),
        )
        events.append(prediction)

    observation = _build_event(
        trace,
        ControllerHistoryEventRole.OBSERVATION,
        target,
        0.40,
        _state_deltas(
            trace.initial_state.values,
            trace.observed_state.values,
            role=ControllerHistoryEventRole.OBSERVATION,
        ),
        "Observed post-action system state.",
        confidence=1.0 - trace.observed_state.uncertainty,
        cause_event_ids=(
            (prediction.event_id,)
            if prediction is not None
            else (decision.event_id,)
        ),
    )
    events.append(observation)

    if trace.prediction_error is not None:
        error = _build_event(
            trace,
            ControllerHistoryEventRole.PREDICTION_ERROR,
            target,
            0.50,
            tuple(
                _scalar_delta(
                    f"prediction_error:{key}",
                    0.0,
                    value,
                    ControllerHistoryEventRole.PREDICTION_ERROR,
                    str(key),
                )
                for key, value
                in sorted(trace.prediction_error.variable_errors.items())
            )
            + (
                _scalar_delta(
                    "prediction_error_l2",
                    0.0,
                    trace.prediction_error.l2_error,
                    ControllerHistoryEventRole.PREDICTION_ERROR,
                ),
            ),
            "Observed-minus-predicted controller error.",
            confidence=trace.prediction_error.confidence,
            cause_event_ids=(observation.event_id,),
        )
        events.append(error)
        stabilization_causes = (error.event_id,)
    else:
        stabilization_causes = (observation.event_id,)

    stabilization = _build_event(
        trace,
        ControllerHistoryEventRole.STABILIZATION,
        target,
        0.60,
        (
            _scalar_delta(
                "residual_before",
                None,
                trace.stabilization.residual_before,
                ControllerHistoryEventRole.STABILIZATION,
            ),
            _scalar_delta(
                "residual_observed",
                None,
                trace.stabilization.residual_observed,
                ControllerHistoryEventRole.STABILIZATION,
            ),
            _scalar_delta(
                "residual_change",
                0.0,
                trace.stabilization.residual_change,
                ControllerHistoryEventRole.STABILIZATION,
            ),
            _scalar_delta(
                "stabilization_outcome",
                None,
                trace.stabilization.outcome.value,
                ControllerHistoryEventRole.STABILIZATION,
            ),
        ),
        "Completed predictive-control stabilization outcome.",
        cause_event_ids=stabilization_causes,
    )
    events.append(stabilization)

    return tuple(events)


def _normalized(values: Mapping[str, float]) -> Mapping[str, float]:
    total = sum(abs(float(value)) for value in values.values())
    if total <= 1e-12:
        return MappingProxyType(
            {str(key): 0.0 for key in values}
        )
    return MappingProxyType(
        {
            str(key): abs(float(value)) / total
            for key, value in values.items()
        }
    )


def _outcome_fractions(
    trace: ExperienceTrace,
) -> tuple[float, float, float]:
    outcome = trace.stabilization.outcome
    if outcome is StabilizationOutcome.IMPROVED:
        return 0.0, 1.0, 0.0
    if outcome is StabilizationOutcome.WORSENED:
        return 1.0, 0.0, 0.0
    return 0.0, 0.0, 1.0


def build_controller_structural_signature(
    trace: ExperienceTrace,
    *,
    source_pattern_id: str | None = None,
) -> StructuralSignature:
    if not trace_is_learning_free(trace):
        raise ControllerHistoryAdapterError(
            "cannot build signature from learning-mutated trace"
        )

    harmful, beneficial, mixed = _outcome_fractions(trace)

    target_profile = _normalized(
        {
            str(key): abs(
                float(trace.observed_state.values.get(key, 0.0))
                - float(trace.observed_state.target_values.get(key, 0.0))
            )
            for key in trace.observed_state.target_values
        }
    )

    error_norm = (
        0.0
        if trace.prediction_error is None
        else trace.prediction_error.l2_error
    )

    action_profile = {
        "action": 1.0 if trace.action_kind is ExperienceActionKind.ACTION else 0.0,
        "probe": 1.0 if trace.action_kind is ExperienceActionKind.PROBE else 0.0,
        "hold": 1.0 if trace.action_kind is ExperienceActionKind.HOLD else 0.0,
        "no_safe_action": (
            1.0
            if trace.action_kind is ExperienceActionKind.NO_SAFE_ACTION
            else 0.0
        ),
        "prediction_error": error_norm,
    }

    start = (
        0.0
        if trace.identity.timestamp_start is None
        else trace.identity.timestamp_start
    )
    end = (
        start
        if trace.identity.timestamp_end is None
        else trace.identity.timestamp_end
    )
    duration = max(0.0, end - start)

    residual_before = trace.stabilization.residual_before
    progression = min(
        1.0,
        abs(trace.stabilization.residual_change)
        / (residual_before + 1e-12),
    )

    source_id = (
        source_pattern_id
        or trace.metadata.get("pattern_family")
        or trace.identity.metadata.get("pattern_family")
        or trace.disturbance.metadata.get("pattern_family")
        or f"controller_trace:{trace.identity.trace_id}"
    )

    total_changes = (
        7
        if trace.prediction_error is not None
        else 6
    )

    mean_signature_weight = (
        error_norm
        / total_changes
    )

    return StructuralSignature(
        source_pattern_id=str(source_id),
        plane_profile={
            "controller": 0.5,
            "observed_system": 0.5,
        },
        agent_profile={
            "predictive_controller": 1.0,
        },
        kind_profile=action_profile,
        time_scale_profile={
            "fast": 1.0,
        },
        target_profile=target_profile,
        rheology_profile={},
        total_changes=total_changes,
        root_count=1,
        leaf_count=1,
        causal_depth=6 if trace.prediction_error is not None else 5,
        start_time=start,
        end_time=end,
        duration=duration,
        total_capacity_loss=max(
            0.0,
            trace.demand.total - trace.reserve.available,
        ),
        total_capacity_gain=max(
            0.0,
            trace.reserve.available - trace.demand.total,
        ),
        net_capacity_effect=(
            trace.reserve.available - trace.demand.total
        ),
        cumulative_signature_weight=error_norm,
        mean_signature_weight=mean_signature_weight,
        persistence_index=0.0,
        irreversibility_index=0.0,
        progression_index=progression,
        adaptation_index=0.0,
        harmful_fraction=harmful,
        beneficial_fraction=beneficial,
        mixed_fraction=mixed,
        rheology_change_count=0,
        total_abs_creep_strain=0.0,
        mean_abs_creep_strain=0.0,
        max_abs_creep_strain=0.0,
        mean_abs_residual_strain=0.0,
        max_abs_residual_strain=0.0,
        rheological_memory_index=0.0,
        rheological_retention_index=0.0,
        rheological_relaxation_index=0.0,
        label="controller_experience",
        metadata={
            "controller_history_signature": True,
            "trace_id": trace.identity.trace_id,
            "sequence_id": trace.identity.sequence_id,
            "pattern_id": source_id,
            "learning_applied": False,
            "controller_memory_mutated": False,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "rheological_claim": False,
            "irreversible_change_claim": False,
            "external_expected_label_used": False,
        },
    )


def adapt_experience_trace(
    trace: ExperienceTrace,
    *,
    pattern_id: str | None = None,
) -> ControllerHistoryRecord:
    events = experience_trace_to_history_events(trace)
    signature = build_controller_structural_signature(
        trace,
        source_pattern_id=pattern_id,
    )

    return ControllerHistoryRecord(
        trace_id=trace.identity.trace_id,
        sequence_id=trace.identity.sequence_id,
        pattern_id=pattern_id or signature.source_pattern_id,
        events=events,
        signature=signature,
        source_trace=trace,
        metadata={
            "controller_history_adapter": True,
            "history_event_count": len(events),
            "structural_signature_generated": True,
            "history_pattern_generated": False,
            "irreversible_change_generated": False,
            "learning_applied": False,
            "predictive_state_mutated": False,
            "predictive_preload_applied": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


def adapt_experience_traces(
    traces: Sequence[ExperienceTrace],
    *,
    pattern_id_resolver=None,
) -> tuple[ControllerHistoryRecord, ...]:
    return tuple(
        adapt_experience_trace(
            trace,
            pattern_id=(
                None
                if pattern_id_resolver is None
                else pattern_id_resolver(trace)
            ),
        )
        for trace in traces
    )


def controller_signature_distance(
    left: ControllerHistoryRecord,
    right: ControllerHistoryRecord,
) -> float:
    return float(
        left.signature.euclidean_scalar_distance(
            right.signature
        )
    )


def controller_signatures_compare(
    left: ControllerHistoryRecord,
    right: ControllerHistoryRecord,
):
    return left.signature.compare(
        right.signature
    )


def record_is_preload_free(
    record: ControllerHistoryRecord,
) -> bool:
    return (
        record.metadata.get("predictive_preload_applied") is False
        and record.metadata.get("predictive_state_mutated") is False
        and record.metadata.get("graph_mutated") is False
        and record.signature.metadata.get("predictive_preload_applied") is False
        and record.signature.metadata.get("learning_applied") is False
    )


__all__ = [
    "ControllerHistoryAdapterError",
    "ControllerHistoryEventRole",
    "ControllerHistoryRecord",
    "InvalidControllerHistoryRecordError",
    "adapt_experience_trace",
    "adapt_experience_traces",
    "build_controller_structural_signature",
    "controller_signature_distance",
    "controller_signatures_compare",
    "experience_trace_to_history_events",
    "record_is_preload_free",
]

