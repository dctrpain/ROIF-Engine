"""
ROIF Memory 2.0
Experience Sequence Core

Builds on ExperienceTransformation by chaining transitions:

    S0 --E0--> S1 --E1--> S2 --E2--> S3 ...

Core invariant:
    transformation[n].state_after == transformation[n+1].state_before

The sequence layer measures how repeated or related experiences reshape
the system over time:
- accumulated scar magnitude
- physiological / cognitive scar accumulation
- reserve trajectory
- sensitization / adaptation trajectory
- repeated-event response change
- continuity of state propagation

This module remains:
- policy-free
- diagnosis-free
- causally agnostic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from roif.history.experience_transformation import (
    ExperienceTransformation,
    ExperienceTransformationError,
    repeated_event_response_changed,
    same_event_different_state,
    transformation_is_policy_free,
)


SCHEMA_VERSION = "experience_sequence_v1"


class ExperienceSequenceError(ValueError):
    pass


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _validate_id(
    name: str,
    value: str,
) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ExperienceSequenceError(
            f"{name} must not be empty"
        )
    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    x = float(value)
    if not isfinite(x):
        raise ExperienceSequenceError(
            f"{name} must be finite"
        )
    return x


@dataclass(frozen=True, slots=True)
class SequenceStepMetrics:
    step_index: int
    transformation_id: str
    event_type: str

    reserve_before: float
    reserve_after: float
    reserve_delta: float

    cognitive_scar_magnitude: float
    physiological_scar_magnitude: float
    total_scar_magnitude: float

    sensitization_index: float
    adaptation_index: float

    repeated_event_from_previous: bool
    response_changed_from_previous: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ExperienceSequenceError(
                "step_index must be non-negative"
            )

        object.__setattr__(
            self,
            "transformation_id",
            _validate_id(
                "transformation_id",
                self.transformation_id,
            ),
        )

        object.__setattr__(
            self,
            "event_type",
            _validate_id(
                "event_type",
                self.event_type,
            ),
        )

        for name in (
            "reserve_before",
            "reserve_after",
            "reserve_delta",
            "cognitive_scar_magnitude",
            "physiological_scar_magnitude",
            "total_scar_magnitude",
            "sensitization_index",
            "adaptation_index",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        for name in (
            "cognitive_scar_magnitude",
            "physiological_scar_magnitude",
            "total_scar_magnitude",
            "sensitization_index",
            "adaptation_index",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise ExperienceSequenceError(
                    f"{name} must be non-negative"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperienceSequenceSummary:
    transformation_count: int

    initial_reserve: float
    final_reserve: float
    reserve_change: float

    cumulative_cognitive_scar: float
    cumulative_physiological_scar: float
    cumulative_total_scar: float

    cumulative_sensitization: float
    cumulative_adaptation: float

    repeated_event_pair_count: int
    repeated_event_response_change_count: int

    all_transitions_policy_free: bool
    continuity_preserved: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.transformation_count < 1:
            raise ExperienceSequenceError(
                "transformation_count must be >= 1"
            )

        if self.repeated_event_pair_count < 0:
            raise ExperienceSequenceError(
                "repeated_event_pair_count must be non-negative"
            )

        if self.repeated_event_response_change_count < 0:
            raise ExperienceSequenceError(
                "repeated_event_response_change_count must be non-negative"
            )

        if (
            self.repeated_event_response_change_count
            > self.repeated_event_pair_count
        ):
            raise ExperienceSequenceError(
                "response change count cannot exceed repeated-event pair count"
            )

        for name in (
            "initial_reserve",
            "final_reserve",
            "reserve_change",
            "cumulative_cognitive_scar",
            "cumulative_physiological_scar",
            "cumulative_total_scar",
            "cumulative_sensitization",
            "cumulative_adaptation",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(
                        self,
                        name,
                    ),
                ),
            )

        for name in (
            "cumulative_cognitive_scar",
            "cumulative_physiological_scar",
            "cumulative_total_scar",
            "cumulative_sensitization",
            "cumulative_adaptation",
        ):
            if getattr(
                self,
                name,
            ) < 0.0:
                raise ExperienceSequenceError(
                    f"{name} must be non-negative"
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperienceSequence:
    sequence_id: str
    transformations: tuple[
        ExperienceTransformation,
        ...,
    ]
    step_metrics: tuple[
        SequenceStepMetrics,
        ...,
    ]
    summary: ExperienceSequenceSummary
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_id",
            _validate_id(
                "sequence_id",
                self.sequence_id,
            ),
        )

        if not self.transformations:
            raise ExperienceSequenceError(
                "transformations must not be empty"
            )

        if len(
            self.transformations
        ) != len(
            self.step_metrics
        ):
            raise ExperienceSequenceError(
                "transformations and step_metrics must have equal length"
            )

        object.__setattr__(
            self,
            "transformations",
            tuple(
                self.transformations
            ),
        )

        object.__setattr__(
            self,
            "step_metrics",
            tuple(
                self.step_metrics
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def continuity_break_indices(
    transformations: Iterable[
        ExperienceTransformation
    ],
) -> tuple[int, ...]:
    """
    Return indices i for which transition i does NOT connect to i+1.
    """

    items = tuple(
        transformations
    )

    breaks = []

    for index in range(
        len(
            items
        ) - 1
    ):
        if (
            items[
                index
            ].state_after
            != items[
                index + 1
            ].state_before
        ):
            breaks.append(
                index
            )

    return tuple(
        breaks
    )


def sequence_is_continuous(
    transformations: Iterable[
        ExperienceTransformation
    ],
) -> bool:
    return not continuity_break_indices(
        transformations
    )


def _build_step_metrics(
    transformations: tuple[
        ExperienceTransformation,
        ...,
    ],
) -> tuple[
    SequenceStepMetrics,
    ...,
]:
    metrics = []

    for index, transformation in enumerate(
        transformations
    ):
        previous = (
            transformations[
                index - 1
            ]
            if index > 0
            else None
        )

        repeated_event = (
            same_event_different_state(
                previous,
                transformation,
            )
            if previous is not None
            else False
        )

        response_changed = (
            repeated_event_response_changed(
                previous,
                transformation,
            )
            if previous is not None
            else False
        )

        metrics.append(
            SequenceStepMetrics(
                step_index=index,
                transformation_id=(
                    transformation.transformation_id
                ),
                event_type=(
                    transformation.event.event_type
                ),
                reserve_before=(
                    transformation.state_before.reserve
                ),
                reserve_after=(
                    transformation.state_after.reserve
                ),
                reserve_delta=(
                    transformation.scar.reserve_delta
                ),
                cognitive_scar_magnitude=(
                    transformation.scar.cognitive_magnitude
                ),
                physiological_scar_magnitude=(
                    transformation.scar.physiological_magnitude
                ),
                total_scar_magnitude=(
                    transformation.scar.total_magnitude
                ),
                sensitization_index=(
                    transformation.scar.sensitization_index
                ),
                adaptation_index=(
                    transformation.scar.adaptation_index
                ),
                repeated_event_from_previous=(
                    repeated_event
                ),
                response_changed_from_previous=(
                    response_changed
                ),
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "action_selected": False,
                    "policy_modified": False,
                    "causal_truth_inferred": False,
                },
            )
        )

    return tuple(
        metrics
    )


def summarize_experience_sequence(
    transformations: Iterable[
        ExperienceTransformation
    ],
) -> ExperienceSequenceSummary:
    items = tuple(
        transformations
    )

    if not items:
        raise ExperienceSequenceError(
            "cannot summarize empty transformation sequence"
        )

    continuous = sequence_is_continuous(
        items
    )

    repeated_pairs = 0
    changed_pairs = 0

    for previous, current in zip(
        items,
        items[
            1:
        ],
    ):
        if same_event_different_state(
            previous,
            current,
        ):
            repeated_pairs += 1

            if repeated_event_response_changed(
                previous,
                current,
            ):
                changed_pairs += 1

    return ExperienceSequenceSummary(
        transformation_count=len(
            items
        ),
        initial_reserve=(
            items[
                0
            ].state_before.reserve
        ),
        final_reserve=(
            items[
                -1
            ].state_after.reserve
        ),
        reserve_change=(
            items[
                -1
            ].state_after.reserve
            - items[
                0
            ].state_before.reserve
        ),
        cumulative_cognitive_scar=sum(
            item.scar.cognitive_magnitude
            for item in items
        ),
        cumulative_physiological_scar=sum(
            item.scar.physiological_magnitude
            for item in items
        ),
        cumulative_total_scar=sum(
            item.scar.total_magnitude
            for item in items
        ),
        cumulative_sensitization=sum(
            item.scar.sensitization_index
            for item in items
        ),
        cumulative_adaptation=sum(
            item.scar.adaptation_index
            for item in items
        ),
        repeated_event_pair_count=(
            repeated_pairs
        ),
        repeated_event_response_change_count=(
            changed_pairs
        ),
        all_transitions_policy_free=all(
            transformation_is_policy_free(
                item
            )
            for item in items
        ),
        continuity_preserved=continuous,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "summary_is_evaluator_only": True,
            "causal_truth_inferred": False,
        },
    )


def build_experience_sequence(
    *,
    sequence_id: str,
    transformations: Iterable[
        ExperienceTransformation
    ],
    require_continuity: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ExperienceSequence:
    items = tuple(
        transformations
    )

    if not items:
        raise ExperienceSequenceError(
            "transformations must not be empty"
        )

    breaks = continuity_break_indices(
        items
    )

    if (
        require_continuity
        and breaks
    ):
        raise ExperienceSequenceError(
            "sequence continuity violated at indices "
            + ", ".join(
                str(
                    index
                )
                for index in breaks
            )
        )

    summary = summarize_experience_sequence(
        items
    )

    step_metrics = _build_step_metrics(
        items
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "require_continuity": bool(
            require_continuity
        ),
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return ExperienceSequence(
        sequence_id=sequence_id,
        transformations=items,
        step_metrics=step_metrics,
        summary=summary,
        metadata=merged_metadata,
    )


def reserve_trajectory(
    sequence: ExperienceSequence,
) -> tuple[float, ...]:
    """
    Return S0 reserve followed by each state_after reserve.
    """

    first = sequence.transformations[
        0
    ].state_before.reserve

    return (
        first,
        *tuple(
            transformation.state_after.reserve
            for transformation
            in sequence.transformations
        ),
    )


def total_scar_series(
    sequence: ExperienceSequence,
) -> tuple[float, ...]:
    return tuple(
        step.total_scar_magnitude
        for step in sequence.step_metrics
    )


def physiological_scar_series(
    sequence: ExperienceSequence,
) -> tuple[float, ...]:
    return tuple(
        step.physiological_scar_magnitude
        for step in sequence.step_metrics
    )


def cognitive_scar_series(
    sequence: ExperienceSequence,
) -> tuple[float, ...]:
    return tuple(
        step.cognitive_scar_magnitude
        for step in sequence.step_metrics
    )


def sensitization_series(
    sequence: ExperienceSequence,
) -> tuple[float, ...]:
    return tuple(
        step.sensitization_index
        for step in sequence.step_metrics
    )


def adaptation_series(
    sequence: ExperienceSequence,
) -> tuple[float, ...]:
    return tuple(
        step.adaptation_index
        for step in sequence.step_metrics
    )


def repeated_event_flags(
    sequence: ExperienceSequence,
) -> tuple[bool, ...]:
    return tuple(
        step.repeated_event_from_previous
        for step in sequence.step_metrics
    )


def response_change_flags(
    sequence: ExperienceSequence,
) -> tuple[bool, ...]:
    return tuple(
        step.response_changed_from_previous
        for step in sequence.step_metrics
    )


def sequence_is_policy_free(
    sequence: ExperienceSequence,
) -> bool:
    return (
        sequence.metadata.get(
            "action_selected"
        )
        is False
        and sequence.metadata.get(
            "policy_modified"
        )
        is False
        and sequence.summary.all_transitions_policy_free
    )


__all__ = [
    "ExperienceSequence",
    "ExperienceSequenceError",
    "ExperienceSequenceSummary",
    "SCHEMA_VERSION",
    "SequenceStepMetrics",
    "adaptation_series",
    "build_experience_sequence",
    "cognitive_scar_series",
    "continuity_break_indices",
    "physiological_scar_series",
    "repeated_event_flags",
    "reserve_trajectory",
    "response_change_flags",
    "sensitization_series",
    "sequence_is_continuous",
    "sequence_is_policy_free",
    "summarize_experience_sequence",
    "total_scar_series",
]
