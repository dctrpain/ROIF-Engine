"""Tests for the ROIF operator abstraction layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from roif.operator import (
    CompositeOperator,
    FunctionalOperator,
    IdentityOperator,
    Operator,
    OperatorApplication,
    OperatorError,
    OperatorStage,
    apply_operators,
    compose_operators,
)
from roif.state import SystemState


def make_state(time: float = 0.0) -> SystemState:
    """Create the smallest valid immutable system snapshot."""

    return SystemState(time=time)


def make_time_shift(
    operator_id: str,
    delta: float,
    *,
    name: str | None = None,
    stage: OperatorStage | str = OperatorStage.CUSTOM,
    enabled: bool = True,
    metadata: dict[str, Any] | None = None,
) -> FunctionalOperator:
    """Create an operator that advances time by a fixed nonnegative delta."""

    return FunctionalOperator(
        operator_id=operator_id,
        name=name,
        stage=stage,
        enabled=enabled,
        metadata={} if metadata is None else metadata,
        transform=lambda state: state.at_time(state.time + delta),
    )


class WrongReturnOperator(Operator):
    """Test helper returning an invalid output object."""

    def _apply(self, state: SystemState) -> SystemState:
        return "not a state"  # type: ignore[return-value]


class BackwardTimeOperator(Operator):
    """Test helper attempting to move simulation time backwards."""

    def _apply(self, state: SystemState) -> SystemState:
        return SystemState(time=state.time - 1.0)


class CountingOperator(Operator):
    """Test helper recording whether its concrete transformation ran."""

    calls = 0

    def _apply(self, state: SystemState) -> SystemState:
        type(self).calls += 1
        return state


# ---------------------------------------------------------------------------
# OperatorStage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("D", OperatorStage.D),
        ("P", OperatorStage.P),
        ("C", OperatorStage.C),
        ("L", OperatorStage.L),
        ("custom", OperatorStage.CUSTOM),
    ],
)
def test_operator_stage_values(raw: str, expected: OperatorStage) -> None:
    assert OperatorStage(raw) is expected


def test_operator_stage_is_string_enum() -> None:
    assert OperatorStage.D.value == "D"
    assert isinstance(OperatorStage.D, str)


# ---------------------------------------------------------------------------
# IdentityOperator and common Operator behavior
# ---------------------------------------------------------------------------


def test_identity_operator_construction() -> None:
    operator = IdentityOperator()

    assert operator.operator_id == "identity"
    assert operator.name == "Identity"
    assert operator.stage is OperatorStage.CUSTOM
    assert operator.enabled is True


def test_operator_identifier_is_trimmed() -> None:
    operator = IdentityOperator(operator_id="  identity_test  ")

    assert operator.operator_id == "identity_test"


def test_operator_name_is_trimmed() -> None:
    operator = IdentityOperator(name="  No change  ")

    assert operator.name == "No change"


def test_missing_operator_name_defaults_to_identifier() -> None:
    operator = IdentityOperator(
        operator_id="same",
        name=None,
    )

    assert operator.name == "same"


def test_operator_stage_can_be_created_from_string() -> None:
    operator = IdentityOperator(stage="D")

    assert operator.stage is OperatorStage.D


def test_identity_returns_same_state_object() -> None:
    state = make_state()
    result = IdentityOperator().apply(state)

    assert result is state


def test_operator_is_callable() -> None:
    state = make_state()
    operator = make_time_shift("advance", 2.0)

    result = operator(state)

    assert result.time == pytest.approx(2.0)


def test_disabled_operator_returns_same_state_without_transforming() -> None:
    CountingOperator.calls = 0
    state = make_state()

    operator = CountingOperator(
        operator_id="disabled",
        enabled=False,
    )

    result = operator.apply(state)

    assert result is state
    assert CountingOperator.calls == 0


def test_operator_rejects_non_system_state_input() -> None:
    operator = IdentityOperator()

    with pytest.raises(OperatorError, match="requires a SystemState"):
        operator.apply(object())  # type: ignore[arg-type]


def test_operator_rejects_non_state_output() -> None:
    operator = WrongReturnOperator(operator_id="wrong")

    with pytest.raises(OperatorError, match="must return a SystemState"):
        operator.apply(make_state())


def test_operator_rejects_backward_time_output() -> None:
    operator = BackwardTimeOperator(operator_id="backward")

    with pytest.raises(OperatorError, match="moved system time backwards"):
        operator.apply(make_state(time=2.0))


@pytest.mark.parametrize("operator_id", ["", " ", "\t", "\n"])
def test_operator_rejects_empty_identifier(operator_id: str) -> None:
    with pytest.raises(OperatorError, match="operator_id cannot be empty"):
        IdentityOperator(operator_id=operator_id)


@pytest.mark.parametrize("operator_id", [1, 1.5, None, object()])
def test_operator_rejects_non_string_identifier(operator_id: object) -> None:
    with pytest.raises(OperatorError, match="operator_id must be a string"):
        IdentityOperator(operator_id=operator_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", " ", "\t", "\n"])
def test_operator_rejects_empty_name(name: str) -> None:
    with pytest.raises(OperatorError, match="name cannot be empty"):
        IdentityOperator(name=name)


@pytest.mark.parametrize("name", [1, 1.5, object()])
def test_operator_rejects_non_string_name(name: object) -> None:
    with pytest.raises(OperatorError, match="name must be a string"):
        IdentityOperator(name=name)  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", ["unknown", "", "CUSTOM", 123])
def test_operator_rejects_invalid_stage(stage: object) -> None:
    with pytest.raises(OperatorError, match="Unknown operator stage"):
        IdentityOperator(stage=stage)  # type: ignore[arg-type]


@pytest.mark.parametrize("enabled", [0, 1, "yes", None])
def test_operator_rejects_non_boolean_enabled(enabled: object) -> None:
    with pytest.raises(OperatorError, match="enabled must be a bool"):
        IdentityOperator(enabled=enabled)  # type: ignore[arg-type]


def test_operator_rejects_non_mapping_metadata() -> None:
    with pytest.raises(OperatorError, match="metadata must be a mapping"):
        IdentityOperator(metadata=["invalid"])  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["", " ", "\t"])
def test_operator_rejects_empty_metadata_key(key: str) -> None:
    with pytest.raises(OperatorError, match="metadata key cannot be empty"):
        IdentityOperator(metadata={key: 1})


def test_operator_rejects_non_string_metadata_key() -> None:
    with pytest.raises(OperatorError, match="metadata key must be a string"):
        IdentityOperator(metadata={1: "invalid"})  # type: ignore[dict-item]


def test_operator_metadata_key_is_trimmed() -> None:
    operator = IdentityOperator(metadata={"  source  ": "test"})

    assert dict(operator.metadata) == {"source": "test"}


def test_operator_metadata_mapping_is_read_only() -> None:
    operator = IdentityOperator(metadata={"source": "test"})

    with pytest.raises(TypeError):
        operator.metadata["source"] = "changed"  # type: ignore[index]


def test_operator_input_metadata_is_copied() -> None:
    metadata = {"source": "original"}
    operator = IdentityOperator(metadata=metadata)

    metadata["source"] = "mutated"

    assert operator.metadata["source"] == "original"


def test_operator_dataclass_is_frozen() -> None:
    operator = IdentityOperator()

    with pytest.raises(FrozenInstanceError):
        operator.enabled = False  # type: ignore[misc]


def test_operator_as_dict() -> None:
    operator = IdentityOperator(
        operator_id="same",
        name="Same state",
        stage=OperatorStage.P,
        metadata={"source": "test"},
    )

    result = operator.as_dict()

    assert result == {
        "operator_id": "same",
        "name": "Same state",
        "stage": "P",
        "enabled": True,
        "metadata": {"source": "test"},
        "type": "IdentityOperator",
    }


# ---------------------------------------------------------------------------
# FunctionalOperator
# ---------------------------------------------------------------------------


def test_functional_operator_applies_callable() -> None:
    operator = make_time_shift("advance", 3.0)

    result = operator.apply(make_state(time=2.0))

    assert result.time == pytest.approx(5.0)


def test_functional_operator_does_not_mutate_input_state() -> None:
    state = make_state(time=1.0)
    operator = make_time_shift("advance", 2.0)

    result = operator(state)

    assert state.time == pytest.approx(1.0)
    assert result.time == pytest.approx(3.0)
    assert result is not state


def test_functional_operator_can_return_same_state() -> None:
    state = make_state()
    operator = FunctionalOperator(
        operator_id="noop",
        transform=lambda current: current,
    )

    assert operator(state) is state


def test_functional_operator_rejects_non_callable_transform() -> None:
    with pytest.raises(OperatorError, match="transform must be callable"):
        FunctionalOperator(
            operator_id="invalid",
            transform=123,  # type: ignore[arg-type]
        )


def test_functional_operator_validates_callable_output() -> None:
    operator = FunctionalOperator(
        operator_id="invalid_output",
        transform=lambda state: None,  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(OperatorError, match="must return a SystemState"):
        operator(make_state())


# ---------------------------------------------------------------------------
# OperatorApplication
# ---------------------------------------------------------------------------


def test_apply_with_record() -> None:
    state = make_state(time=1.0)
    operator = make_time_shift(
        "advance",
        2.0,
        name="Advance time",
        stage=OperatorStage.D,
        metadata={"reason": "test"},
    )

    record = operator.apply_with_record(state)

    assert isinstance(record, OperatorApplication)
    assert record.operator_id == "advance"
    assert record.operator_name == "Advance time"
    assert record.stage is OperatorStage.D
    assert record.input_state is state
    assert record.output_state.time == pytest.approx(3.0)
    assert record.metadata["reason"] == "test"


def test_application_time_delta() -> None:
    record = OperatorApplication(
        operator_id="advance",
        operator_name="Advance",
        stage=OperatorStage.CUSTOM,
        input_state=make_state(2.0),
        output_state=make_state(5.5),
    )

    assert record.time_delta == pytest.approx(3.5)


def test_application_changed_is_false_for_identity() -> None:
    state = make_state()
    record = IdentityOperator().apply_with_record(state)

    assert record.changed is False


def test_application_changed_is_true_for_new_snapshot() -> None:
    record = make_time_shift("advance", 1.0).apply_with_record(make_state())

    assert record.changed is True


def test_application_identifiers_are_trimmed() -> None:
    record = OperatorApplication(
        operator_id="  op  ",
        operator_name="  Operator  ",
        stage="C",
        input_state=make_state(),
        output_state=make_state(),
    )

    assert record.operator_id == "op"
    assert record.operator_name == "Operator"
    assert record.stage is OperatorStage.C


def test_application_rejects_non_state_input() -> None:
    with pytest.raises(OperatorError, match="input_state must be a SystemState"):
        OperatorApplication(
            operator_id="op",
            operator_name="Operator",
            stage=OperatorStage.CUSTOM,
            input_state=object(),  # type: ignore[arg-type]
            output_state=make_state(),
        )


def test_application_rejects_non_state_output() -> None:
    with pytest.raises(OperatorError, match="output_state must be a SystemState"):
        OperatorApplication(
            operator_id="op",
            operator_name="Operator",
            stage=OperatorStage.CUSTOM,
            input_state=make_state(),
            output_state=object(),  # type: ignore[arg-type]
        )


def test_application_rejects_backward_time() -> None:
    with pytest.raises(OperatorError, match="cannot move system time backwards"):
        OperatorApplication(
            operator_id="op",
            operator_name="Operator",
            stage=OperatorStage.CUSTOM,
            input_state=make_state(2.0),
            output_state=make_state(1.0),
        )


def test_application_metadata_is_read_only() -> None:
    record = IdentityOperator(metadata={"source": "test"}).apply_with_record(
        make_state()
    )

    with pytest.raises(TypeError):
        record.metadata["source"] = "changed"  # type: ignore[index]


def test_application_as_dict() -> None:
    record = OperatorApplication(
        operator_id="advance",
        operator_name="Advance",
        stage=OperatorStage.L,
        input_state=make_state(1.0),
        output_state=make_state(4.0),
        metadata={"source": "test"},
    )

    assert record.as_dict() == {
        "operator_id": "advance",
        "operator_name": "Advance",
        "stage": "L",
        "input_time": 1.0,
        "output_time": 4.0,
        "time_delta": 3.0,
        "changed": True,
        "metadata": {"source": "test"},
    }


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_then_applies_operators_in_execution_order() -> None:
    first = make_time_shift("first", 2.0)
    second = FunctionalOperator(
        operator_id="second",
        transform=lambda state: state.at_time(state.time * 3.0),
    )

    pipeline = first.then(second)
    result = pipeline(make_state(1.0))

    assert result.time == pytest.approx(9.0)


def test_right_shift_applies_left_operator_first() -> None:
    first = make_time_shift("first", 2.0)
    second = FunctionalOperator(
        operator_id="second",
        transform=lambda state: state.at_time(state.time * 3.0),
    )

    result = (first >> second)(make_state(1.0))

    assert result.time == pytest.approx(9.0)


def test_compose_applies_inner_operator_first() -> None:
    outer = FunctionalOperator(
        operator_id="outer",
        transform=lambda state: state.at_time(state.time * 3.0),
    )
    inner = make_time_shift("inner", 2.0)

    result = outer.compose(inner)(make_state(1.0))

    assert result.time == pytest.approx(9.0)


def test_matmul_uses_mathematical_composition_order() -> None:
    outer = FunctionalOperator(
        operator_id="outer",
        transform=lambda state: state.at_time(state.time * 3.0),
    )
    inner = make_time_shift("inner", 2.0)

    result = (outer @ inner)(make_state(1.0))

    assert result.time == pytest.approx(9.0)


def test_operator_composition_is_noncommutative() -> None:
    add_two = make_time_shift("add_two", 2.0)
    multiply_three = FunctionalOperator(
        operator_id="multiply_three",
        transform=lambda state: state.at_time(state.time * 3.0),
    )
    state = make_state(1.0)

    add_then_multiply = (add_two >> multiply_three)(state)
    multiply_then_add = (multiply_three >> add_two)(state)

    assert add_then_multiply.time == pytest.approx(9.0)
    assert multiply_then_add.time == pytest.approx(5.0)
    assert add_then_multiply != multiply_then_add


def test_then_rejects_non_operator() -> None:
    with pytest.raises(OperatorError, match="next_operator must be an Operator"):
        IdentityOperator().then(object())  # type: ignore[arg-type]


def test_compose_rejects_non_operator() -> None:
    with pytest.raises(OperatorError, match="inner_operator must be an Operator"):
        IdentityOperator().compose(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CompositeOperator
# ---------------------------------------------------------------------------


def test_composite_operator_construction() -> None:
    first = make_time_shift("first", 1.0)
    second = make_time_shift("second", 2.0)

    pipeline = CompositeOperator(
        operator_id="pipeline",
        operators=(first, second),
    )

    assert pipeline.operator_count == 2
    assert pipeline.operator_ids == ("first", "second")


def test_composite_operator_applies_all_children() -> None:
    pipeline = CompositeOperator(
        operator_id="pipeline",
        operators=(
            make_time_shift("first", 1.0),
            make_time_shift("second", 2.0),
            make_time_shift("third", 3.0),
        ),
    )

    result = pipeline(make_state())

    assert result.time == pytest.approx(6.0)


def test_composite_operator_flattens_nested_composites() -> None:
    first = make_time_shift("first", 1.0)
    second = make_time_shift("second", 1.0)
    third = make_time_shift("third", 1.0)

    nested = CompositeOperator.from_operators(first, second)
    pipeline = CompositeOperator.from_operators(nested, third)

    assert pipeline.operator_ids == ("first", "second", "third")
    assert pipeline.operator_count == 3


def test_chained_right_shift_is_flattened() -> None:
    pipeline = (
        make_time_shift("first", 1.0)
        >> make_time_shift("second", 1.0)
        >> make_time_shift("third", 1.0)
    )

    assert pipeline.operator_ids == ("first", "second", "third")
    assert pipeline.operator_count == 3


def test_composite_generated_identifier_and_name() -> None:
    first = make_time_shift("first", 1.0, name="First")
    second = make_time_shift("second", 1.0, name="Second")

    pipeline = CompositeOperator.from_operators(first, second)

    assert pipeline.operator_id == "first__then__second"
    assert pipeline.name == "First → Second"


def test_composite_accepts_explicit_descriptors() -> None:
    pipeline = CompositeOperator.from_operators(
        IdentityOperator(),
        operator_id="custom_pipeline",
        name="Custom pipeline",
        stage=OperatorStage.C,
        metadata={"source": "test"},
    )

    assert pipeline.operator_id == "custom_pipeline"
    assert pipeline.name == "Custom pipeline"
    assert pipeline.stage is OperatorStage.C
    assert pipeline.metadata["source"] == "test"


def test_disabled_composite_returns_input_unchanged() -> None:
    state = make_state()
    pipeline = CompositeOperator.from_operators(
        make_time_shift("advance", 10.0),
        enabled=False,
    )

    assert pipeline(state) is state


def test_composite_requires_at_least_one_operator() -> None:
    with pytest.raises(
        OperatorError,
        match="requires at least one operator",
    ):
        CompositeOperator(
            operator_id="empty",
            operators=(),
        )


def test_composite_constructor_rejects_non_operator_child() -> None:
    with pytest.raises(OperatorError, match=r"operators\[0\] must be an Operator"):
        CompositeOperator(
            operator_id="invalid",
            operators=(object(),),  # type: ignore[arg-type]
        )


def test_from_operators_requires_at_least_one_operator() -> None:
    with pytest.raises(OperatorError, match="At least one operator is required"):
        CompositeOperator.from_operators()


def test_from_operators_rejects_non_operator() -> None:
    with pytest.raises(OperatorError, match=r"operators\[1\] must be an Operator"):
        CompositeOperator.from_operators(
            IdentityOperator(),
            object(),  # type: ignore[arg-type]
        )


def test_composite_operators_tuple_is_immutable() -> None:
    pipeline = CompositeOperator.from_operators(IdentityOperator())

    assert isinstance(pipeline.operators, tuple)

    with pytest.raises(FrozenInstanceError):
        pipeline.operators = ()  # type: ignore[misc]


def test_composite_application_records() -> None:
    state = make_state()
    pipeline = CompositeOperator.from_operators(
        make_time_shift("first", 1.0),
        make_time_shift("second", 2.0),
    )

    records = pipeline.application_records(state)

    assert len(records) == 2
    assert records[0].operator_id == "first"
    assert records[0].input_state.time == pytest.approx(0.0)
    assert records[0].output_state.time == pytest.approx(1.0)
    assert records[1].operator_id == "second"
    assert records[1].input_state.time == pytest.approx(1.0)
    assert records[1].output_state.time == pytest.approx(3.0)


def test_disabled_composite_has_no_application_records() -> None:
    pipeline = CompositeOperator.from_operators(
        IdentityOperator(),
        enabled=False,
    )

    assert pipeline.application_records(make_state()) == ()


def test_composite_application_records_reject_invalid_state() -> None:
    pipeline = CompositeOperator.from_operators(IdentityOperator())

    with pytest.raises(OperatorError, match="requires a SystemState"):
        pipeline.application_records(object())  # type: ignore[arg-type]


def test_composite_as_dict_contains_children() -> None:
    pipeline = CompositeOperator.from_operators(
        IdentityOperator(operator_id="first"),
        IdentityOperator(operator_id="second"),
        operator_id="pipeline",
    )

    data = pipeline.as_dict()

    assert data["operator_id"] == "pipeline"
    assert data["type"] == "CompositeOperator"
    assert data["operator_count"] == 2
    assert [item["operator_id"] for item in data["operators"]] == [
        "first",
        "second",
    ]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_compose_operators_builds_pipeline_from_iterable() -> None:
    pipeline = compose_operators(
        [
            make_time_shift("first", 1.0),
            make_time_shift("second", 2.0),
        ],
        operator_id="pipeline",
    )

    assert isinstance(pipeline, CompositeOperator)
    assert pipeline.operator_id == "pipeline"
    assert pipeline.operator_ids == ("first", "second")
    assert pipeline(make_state()).time == pytest.approx(3.0)


def test_compose_operators_rejects_empty_iterable() -> None:
    with pytest.raises(OperatorError, match="At least one operator is required"):
        compose_operators([])


def test_apply_operators_applies_iterable_without_pipeline() -> None:
    result = apply_operators(
        make_state(),
        [
            make_time_shift("first", 1.0),
            make_time_shift("second", 2.0),
        ],
    )

    assert result.time == pytest.approx(3.0)


def test_apply_operators_accepts_empty_iterable() -> None:
    state = make_state()

    assert apply_operators(state, []) is state


def test_apply_operators_rejects_invalid_state() -> None:
    with pytest.raises(OperatorError, match="state must be a SystemState"):
        apply_operators(object(), [])  # type: ignore[arg-type]


def test_apply_operators_rejects_non_operator_item() -> None:
    with pytest.raises(OperatorError, match=r"operators\[1\] must be an Operator"):
        apply_operators(
            make_state(),
            [
                IdentityOperator(),
                object(),  # type: ignore[list-item]
            ],
        )
