"""
ROIF Memory 2.0
Experience Transformation Core

Stores not merely what happened, but what happened to the system because it happened:

    state_before
      + event
      + response
      + body_response
      -> state_after
      -> scar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "experience_transformation_v1"


class ExperienceTransformationError(ValueError):
    pass


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _validate_id(name: str, value: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ExperienceTransformationError(f"{name} must not be empty")
    return cleaned


def _validate_scalar(name: str, value: float) -> float:
    x = float(value)
    if not isfinite(x):
        raise ExperienceTransformationError(f"{name} must be finite")
    return x


def _validate_vector(name: str, values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(
        _validate_scalar(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )
    if not vector:
        raise ExperienceTransformationError(f"{name} must not be empty")
    return vector


def _vector_subtract(
    after: Sequence[float],
    before: Sequence[float],
) -> tuple[float, ...]:
    if len(after) != len(before):
        raise ExperienceTransformationError("state vector dimensions must match")
    return tuple(float(a) - float(b) for a, b in zip(after, before))


def _l2_norm(vector: Sequence[float]) -> float:
    return sqrt(sum(float(value) ** 2 for value in vector))


@dataclass(frozen=True, slots=True)
class SystemState:
    state_id: str
    cognitive: tuple[float, ...]
    physiological: tuple[float, ...]
    contextual: tuple[float, ...]
    reserve: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _validate_id("state_id", self.state_id))
        object.__setattr__(self, "cognitive", _validate_vector("cognitive", self.cognitive))
        object.__setattr__(self, "physiological", _validate_vector("physiological", self.physiological))
        object.__setattr__(self, "contextual", _validate_vector("contextual", self.contextual))

        reserve = _validate_scalar("reserve", self.reserve)
        if not 0.0 <= reserve <= 1.0:
            raise ExperienceTransformationError("reserve must be within [0, 1]")
        object.__setattr__(self, "reserve", reserve)
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    event_id: str
    event_type: str
    structural_vector: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _validate_id("event_id", self.event_id))
        object.__setattr__(self, "event_type", _validate_id("event_type", self.event_type))
        object.__setattr__(self, "structural_vector", _validate_vector("structural_vector", self.structural_vector))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class SystemResponse:
    response_id: str
    cognitive_response: tuple[float, ...]
    physiological_response: tuple[float, ...]
    behavioral_response: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "response_id", _validate_id("response_id", self.response_id))
        object.__setattr__(self, "cognitive_response", _validate_vector("cognitive_response", self.cognitive_response))
        object.__setattr__(self, "physiological_response", _validate_vector("physiological_response", self.physiological_response))
        object.__setattr__(self, "behavioral_response", _validate_vector("behavioral_response", self.behavioral_response))
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class BodyResponse:
    autonomic: tuple[float, ...]
    endocrine: tuple[float, ...]
    immune: tuple[float, ...]
    motor: tuple[float, ...]
    interoceptive: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("autonomic", "endocrine", "immune", "motor", "interoceptive"):
            object.__setattr__(
                self,
                name,
                _validate_vector(name, getattr(self, name)),
            )
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ScarSignature:
    cognitive_delta: tuple[float, ...]
    physiological_delta: tuple[float, ...]
    contextual_delta: tuple[float, ...]
    reserve_delta: float

    cognitive_magnitude: float
    physiological_magnitude: float
    contextual_magnitude: float
    total_magnitude: float

    sensitization_index: float
    adaptation_index: float

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("cognitive_delta", "physiological_delta", "contextual_delta"):
            object.__setattr__(self, name, _validate_vector(name, getattr(self, name)))

        for name in (
            "reserve_delta",
            "cognitive_magnitude",
            "physiological_magnitude",
            "contextual_magnitude",
            "total_magnitude",
            "sensitization_index",
            "adaptation_index",
        ):
            object.__setattr__(self, name, _validate_scalar(name, getattr(self, name)))

        for name in (
            "cognitive_magnitude",
            "physiological_magnitude",
            "contextual_magnitude",
            "total_magnitude",
        ):
            if getattr(self, name) < 0.0:
                raise ExperienceTransformationError(f"{name} must be non-negative")

        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True, slots=True)
class ExperienceTransformation:
    transformation_id: str
    state_before: SystemState
    event: ExperienceEvent
    response: SystemResponse
    body_response: BodyResponse
    state_after: SystemState
    scar: ScarSignature
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transformation_id",
            _validate_id("transformation_id", self.transformation_id),
        )
        object.__setattr__(self, "metadata", _readonly(self.metadata))


def derive_scar_signature(
    state_before: SystemState,
    state_after: SystemState,
) -> ScarSignature:
    cognitive_delta = _vector_subtract(state_after.cognitive, state_before.cognitive)
    physiological_delta = _vector_subtract(
        state_after.physiological,
        state_before.physiological,
    )
    contextual_delta = _vector_subtract(state_after.contextual, state_before.contextual)
    reserve_delta = state_after.reserve - state_before.reserve

    cognitive_magnitude = _l2_norm(cognitive_delta)
    physiological_magnitude = _l2_norm(physiological_delta)
    contextual_magnitude = _l2_norm(contextual_delta)

    total_magnitude = sqrt(
        cognitive_magnitude ** 2
        + physiological_magnitude ** 2
        + contextual_magnitude ** 2
        + reserve_delta ** 2
    )

    # Magnitudes above are direction-agnostic: they describe HOW MUCH
    # the system changed. Sensitization/adaptation describe the DIRECTION
    # of that change.
    #
    # Convention for this compact state representation:
    # positive cognitive/physiological delta -> increased activation/load
    # negative cognitive/physiological delta -> unloading/recovery
    cognitive_load_delta = sum(cognitive_delta) / len(cognitive_delta)
    physiological_load_delta = (
        sum(physiological_delta) / len(physiological_delta)
    )

    sensitization_index = (
        max(0.0, -reserve_delta)
        + max(0.0, physiological_load_delta)
        + 0.5 * max(0.0, cognitive_load_delta)
    )

    adaptation_index = (
        max(0.0, reserve_delta)
        + max(0.0, -physiological_load_delta)
        + 0.5 * max(0.0, -cognitive_load_delta)
    )

    return ScarSignature(
        cognitive_delta=cognitive_delta,
        physiological_delta=physiological_delta,
        contextual_delta=contextual_delta,
        reserve_delta=reserve_delta,
        cognitive_magnitude=cognitive_magnitude,
        physiological_magnitude=physiological_magnitude,
        contextual_magnitude=contextual_magnitude,
        total_magnitude=total_magnitude,
        sensitization_index=sensitization_index,
        adaptation_index=adaptation_index,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "scar_is_damage_claim": False,
            "scar_is_transition_descriptor": True,
            "causal_truth_inferred": False,
        },
    )


def build_experience_transformation(
    *,
    transformation_id: str,
    state_before: SystemState,
    event: ExperienceEvent,
    response: SystemResponse,
    body_response: BodyResponse,
    state_after: SystemState,
    metadata: Mapping[str, Any] | None = None,
) -> ExperienceTransformation:
    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "causal_truth_inferred": False,
    }
    if metadata:
        merged_metadata.update(dict(metadata))

    return ExperienceTransformation(
        transformation_id=transformation_id,
        state_before=state_before,
        event=event,
        response=response,
        body_response=body_response,
        state_after=state_after,
        scar=derive_scar_signature(state_before, state_after),
        metadata=merged_metadata,
    )


def transition_vector(
    transformation: ExperienceTransformation,
) -> tuple[float, ...]:
    return (
        *transformation.scar.cognitive_delta,
        *transformation.scar.physiological_delta,
        *transformation.scar.contextual_delta,
        transformation.scar.reserve_delta,
    )


def transition_magnitude(
    transformation: ExperienceTransformation,
) -> float:
    return transformation.scar.total_magnitude


def physiological_scar_magnitude(
    transformation: ExperienceTransformation,
) -> float:
    return transformation.scar.physiological_magnitude


def cognitive_scar_magnitude(
    transformation: ExperienceTransformation,
) -> float:
    return transformation.scar.cognitive_magnitude


def transformation_is_policy_free(
    transformation: ExperienceTransformation,
) -> bool:
    return (
        transformation.metadata.get("action_selected") is False
        and transformation.metadata.get("policy_modified") is False
    )


def same_event_different_state(
    left: ExperienceTransformation,
    right: ExperienceTransformation,
) -> bool:
    same_event = (
        left.event.event_type == right.event.event_type
        and left.event.structural_vector == right.event.structural_vector
    )
    return same_event and left.state_before != right.state_before


def repeated_event_response_changed(
    left: ExperienceTransformation,
    right: ExperienceTransformation,
) -> bool:
    if not same_event_different_state(left, right):
        return False

    return (
        left.response != right.response
        or left.body_response != right.body_response
        or left.state_after != right.state_after
    )


__all__ = [
    "BodyResponse",
    "ExperienceEvent",
    "ExperienceTransformation",
    "ExperienceTransformationError",
    "SCHEMA_VERSION",
    "ScarSignature",
    "SystemResponse",
    "SystemState",
    "build_experience_transformation",
    "cognitive_scar_magnitude",
    "derive_scar_signature",
    "physiological_scar_magnitude",
    "repeated_event_response_changed",
    "same_event_different_state",
    "transition_magnitude",
    "transition_vector",
    "transformation_is_policy_free",
]

