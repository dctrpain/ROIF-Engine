"""
Operator abstractions for the ROIF evolution layer.

The state layer represents an immutable system snapshot:

    x_k = SystemState(...)

The operator layer represents a transformation of that snapshot:

    x_{k+1} = R(x_k)

Operators are intentionally:

- immutable;
- composable;
- noncommutative;
- explicit about validation;
- independent of the future history/event subsystem.

Composition order follows mathematical function composition:

    composed = outer @ inner

means:

    composed(state) == outer(inner(state))

Sequential pipeline notation is also supported:

    pipeline = first >> second >> third

means:

    pipeline(state) == third(second(first(state)))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

from .state import SystemState


StateTransform: TypeAlias = Callable[[SystemState], SystemState]


class OperatorError(ValueError):
    """Raised when an operator or operator composition is invalid."""


class OperatorStage(str, Enum):
    """
    Canonical stages of the ROIF evolution operator.

    The complete evolution may later be assembled as:

        R = L ∘ C ∘ P ∘ D

    The enum records architectural placement only. It does not impose
    concrete physical semantics on future implementations.
    """

    D = "D"
    P = "P"
    C = "C"
    L = "L"
    CUSTOM = "custom"


def _normalize_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise OperatorError(f"{field_name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise OperatorError(f"{field_name} cannot be empty.")

    return normalized


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise OperatorError(f"{field_name} must be a mapping.")

    copied: dict[str, Any] = {}

    for key, item in value.items():
        normalized_key = _normalize_identifier(
            key,
            field_name=f"{field_name} key",
        )
        copied[normalized_key] = item

    return MappingProxyType(copied)


def _coerce_stage(value: OperatorStage | str) -> OperatorStage:
    if isinstance(value, OperatorStage):
        return value

    try:
        return OperatorStage(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(stage.value for stage in OperatorStage)
        raise OperatorError(
            f"Unknown operator stage {value!r}. Allowed values: {allowed}."
        ) from exc


@dataclass(frozen=True, slots=True)
class OperatorApplication:
    """
    Immutable description of one operator application.

    This object is deliberately lightweight. Persistent temporal history
    belongs to the future cascade_history module.
    """

    operator_id: str
    operator_name: str
    stage: OperatorStage
    input_state: SystemState
    output_state: SystemState
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator_id",
            _normalize_identifier(
                self.operator_id,
                field_name="operator_id",
            ),
        )
        object.__setattr__(
            self,
            "operator_name",
            _normalize_identifier(
                self.operator_name,
                field_name="operator_name",
            ),
        )
        object.__setattr__(
            self,
            "stage",
            _coerce_stage(self.stage),
        )

        if not isinstance(self.input_state, SystemState):
            raise OperatorError("input_state must be a SystemState.")

        if not isinstance(self.output_state, SystemState):
            raise OperatorError("output_state must be a SystemState.")

        if self.output_state.time < self.input_state.time:
            raise OperatorError(
                "An operator application cannot move system time backwards."
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(
                self.metadata,
                field_name="metadata",
            ),
        )

    @property
    def time_delta(self) -> float:
        """Return output time minus input time."""

        return self.output_state.time - self.input_state.time

    @property
    def changed(self) -> bool:
        """Return True when the resulting snapshot differs from the input."""

        return self.output_state != self.input_state

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return {
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "stage": self.stage.value,
            "input_time": self.input_state.time,
            "output_time": self.output_state.time,
            "time_delta": self.time_delta,
            "changed": self.changed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Operator(ABC):
    """
    Abstract immutable transformation of SystemState.

    Subclasses implement `_apply(state)`. The public `apply(state)` method
    performs common input/output validation and must normally not be
    overridden.
    """

    operator_id: str
    name: str | None = None
    stage: OperatorStage = OperatorStage.CUSTOM
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_id = _normalize_identifier(
            self.operator_id,
            field_name="operator_id",
        )

        normalized_name = (
            normalized_id
            if self.name is None
            else _normalize_identifier(
                self.name,
                field_name="name",
            )
        )

        if not isinstance(self.enabled, bool):
            raise OperatorError("enabled must be a bool.")

        object.__setattr__(self, "operator_id", normalized_id)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "stage", _coerce_stage(self.stage))
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(
                self.metadata,
                field_name="metadata",
            ),
        )

    def apply(self, state: SystemState) -> SystemState:
        """
        Apply the operator to one immutable state snapshot.

        Disabled operators behave as identity transformations.
        """

        self.validate_input(state)

        if not self.enabled:
            return state

        result = self._apply(state)
        self.validate_output(state, result)
        return result

    def apply_with_record(
        self,
        state: SystemState,
    ) -> OperatorApplication:
        """Apply the operator and return a lightweight application record."""

        result = self.apply(state)

        return OperatorApplication(
            operator_id=self.operator_id,
            operator_name=self.name or self.operator_id,
            stage=self.stage,
            input_state=state,
            output_state=result,
            metadata=self.metadata,
        )

    def __call__(self, state: SystemState) -> SystemState:
        return self.apply(state)

    def validate_input(self, state: SystemState) -> None:
        """Validate a state before transformation."""

        if not isinstance(state, SystemState):
            raise OperatorError(
                f"{self.operator_id!r} requires a SystemState input."
            )

    def validate_output(
        self,
        input_state: SystemState,
        output_state: SystemState,
    ) -> None:
        """Validate the transformation result."""

        if not isinstance(output_state, SystemState):
            raise OperatorError(
                f"{self.operator_id!r} must return a SystemState."
            )

        if output_state.time < input_state.time:
            raise OperatorError(
                f"{self.operator_id!r} moved system time backwards: "
                f"{input_state.time} -> {output_state.time}."
            )

    @abstractmethod
    def _apply(self, state: SystemState) -> SystemState:
        """Implement the concrete state transformation."""

    def then(self, next_operator: Operator) -> CompositeOperator:
        """
        Compose operators in execution order.

        `first.then(second)` applies first, then second.
        """

        if not isinstance(next_operator, Operator):
            raise OperatorError("next_operator must be an Operator.")

        return CompositeOperator.from_operators(
            self,
            next_operator,
        )

    def compose(self, inner_operator: Operator) -> CompositeOperator:
        """
        Compose using mathematical function order.

        `outer.compose(inner)` means outer(inner(state)).
        """

        if not isinstance(inner_operator, Operator):
            raise OperatorError("inner_operator must be an Operator.")

        return CompositeOperator.from_operators(
            inner_operator,
            self,
        )

    def __rshift__(self, next_operator: Operator) -> CompositeOperator:
        """`first >> second` means apply first, then second."""

        return self.then(next_operator)

    def __matmul__(self, inner_operator: Operator) -> CompositeOperator:
        """`outer @ inner` means outer(inner(state))."""

        return self.compose(inner_operator)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly operator description."""

        return {
            "operator_id": self.operator_id,
            "name": self.name,
            "stage": self.stage.value,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
            "type": type(self).__name__,
        }


@dataclass(frozen=True, slots=True)
class IdentityOperator(Operator):
    """Operator that returns the input state unchanged."""

    operator_id: str = "identity"
    name: str | None = "Identity"
    stage: OperatorStage = OperatorStage.CUSTOM
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def _apply(self, state: SystemState) -> SystemState:
        return state


@dataclass(frozen=True, slots=True)
class FunctionalOperator(Operator):
    """
    Operator backed by a Python callable.

    This class is useful for experiments and tests. Domain operators should
    normally receive dedicated subclasses with explicit parameters.
    """

    transform: StateTransform = field(
        default=lambda state: state,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        super(FunctionalOperator, self).__post_init__()

        if not callable(self.transform):
            raise OperatorError("transform must be callable.")

    def _apply(self, state: SystemState) -> SystemState:
        return self.transform(state)


@dataclass(frozen=True, slots=True)
class CompositeOperator(Operator):
    """
    Ordered, noncommutative sequence of operators.

    The sequence:

        (A, B, C)

    is evaluated as:

        C(B(A(state)))
    """

    operators: tuple[Operator, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super(CompositeOperator, self).__post_init__()

        normalized = tuple(self.operators)

        if not normalized:
            raise OperatorError(
                "CompositeOperator requires at least one operator."
            )

        flattened: list[Operator] = []

        for index, operator in enumerate(normalized):
            if not isinstance(operator, Operator):
                raise OperatorError(
                    f"operators[{index}] must be an Operator."
                )

            if isinstance(operator, CompositeOperator):
                flattened.extend(operator.operators)
            else:
                flattened.append(operator)

        object.__setattr__(
            self,
            "operators",
            tuple(flattened),
        )

    @classmethod
    def from_operators(
        cls,
        *operators: Operator,
        operator_id: str | None = None,
        name: str | None = None,
        stage: OperatorStage = OperatorStage.CUSTOM,
        enabled: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> CompositeOperator:
        """Construct and flatten an ordered operator pipeline."""

        if not operators:
            raise OperatorError(
                "At least one operator is required."
            )

        flattened: list[Operator] = []

        for index, operator in enumerate(operators):
            if not isinstance(operator, Operator):
                raise OperatorError(
                    f"operators[{index}] must be an Operator."
                )

            if isinstance(operator, CompositeOperator):
                flattened.extend(operator.operators)
            else:
                flattened.append(operator)

        generated_id = operator_id or "__then__".join(
            operator.operator_id
            for operator in flattened
        )

        generated_name = name or " → ".join(
            operator.name or operator.operator_id
            for operator in flattened
        )

        return cls(
            operator_id=generated_id,
            name=generated_name,
            stage=stage,
            enabled=enabled,
            metadata={} if metadata is None else metadata,
            operators=tuple(flattened),
        )

    def _apply(self, state: SystemState) -> SystemState:
        current = state

        for operator in self.operators:
            current = operator.apply(current)

        return current

    @property
    def operator_count(self) -> int:
        return len(self.operators)

    @property
    def operator_ids(self) -> tuple[str, ...]:
        return tuple(
            operator.operator_id
            for operator in self.operators
        )

    def application_records(
        self,
        state: SystemState,
    ) -> tuple[OperatorApplication, ...]:
        """
        Apply each child operator and return one record per transformation.

        The composite itself is not included as a separate record.
        """

        self.validate_input(state)

        if not self.enabled:
            return ()

        records: list[OperatorApplication] = []
        current = state

        for operator in self.operators:
            record = operator.apply_with_record(current)
            records.append(record)
            current = record.output_state

        return tuple(records)

    def as_dict(self) -> dict[str, Any]:
        data = super(CompositeOperator, self).as_dict()
        data["operator_count"] = self.operator_count
        data["operators"] = [
            operator.as_dict()
            for operator in self.operators
        ]
        return data


def compose_operators(
    operators: Iterable[Operator],
    *,
    operator_id: str | None = None,
    name: str | None = None,
    stage: OperatorStage = OperatorStage.CUSTOM,
    enabled: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> CompositeOperator:
    """
    Build a CompositeOperator from an iterable in execution order.

    The first iterable element is applied first.
    """

    normalized = tuple(operators)

    return CompositeOperator.from_operators(
        *normalized,
        operator_id=operator_id,
        name=name,
        stage=stage,
        enabled=enabled,
        metadata=metadata,
    )


def apply_operators(
    state: SystemState,
    operators: Iterable[Operator],
) -> SystemState:
    """Apply an iterable of operators without constructing a named pipeline."""

    if not isinstance(state, SystemState):
        raise OperatorError("state must be a SystemState.")

    current = state

    for index, operator in enumerate(operators):
        if not isinstance(operator, Operator):
            raise OperatorError(
                f"operators[{index}] must be an Operator."
            )

        current = operator.apply(current)

    return current


__all__ = [
    "CompositeOperator",
    "FunctionalOperator",
    "IdentityOperator",
    "Operator",
    "OperatorApplication",
    "OperatorError",
    "OperatorStage",
    "StateTransform",
    "apply_operators",
    "compose_operators",
]
