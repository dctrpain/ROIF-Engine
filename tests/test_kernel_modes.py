"""
Tests for core.kernel_modes.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from core.kernel_modes import (
    KERNEL_MODE_POLICIES,
    OPERATION_DEFINITIONS,
    KernelMode,
    KernelModeError,
    KernelModePolicy,
    KernelOperation,
    OperationDefinition,
    OperationEffect,
    epistemically_protected_operations,
    mode_policy,
    operation_definition,
    operation_is_allowed,
    require_operation,
)


def test_all_kernel_modes_have_policies() -> None:
    assert set(KERNEL_MODE_POLICIES) == set(KernelMode)


def test_all_kernel_operations_have_definitions() -> None:
    assert set(OPERATION_DEFINITIONS) == set(KernelOperation)


def test_mode_policy_mapping_is_read_only() -> None:
    assert isinstance(KERNEL_MODE_POLICIES, MappingProxyType)

    with pytest.raises(TypeError):
        KERNEL_MODE_POLICIES[KernelMode.READ_ONLY] = mode_policy(
            KernelMode.READ_ONLY
        )


def test_operation_definition_mapping_is_read_only() -> None:
    assert isinstance(OPERATION_DEFINITIONS, MappingProxyType)

    with pytest.raises(TypeError):
        OPERATION_DEFINITIONS[KernelOperation.OBSERVE] = (
            operation_definition(KernelOperation.OBSERVE)
        )


def test_operation_definition_creation() -> None:
    definition = OperationDefinition(
        operation=KernelOperation.OBSERVE,
        effect=OperationEffect.NONE,
        description="Read state.",
        epistemically_protected=True,
    )

    assert definition.operation is KernelOperation.OBSERVE
    assert definition.effect is OperationEffect.NONE
    assert definition.description == "Read state."
    assert definition.epistemically_protected is True
    assert definition.requires_protocol_approval is False


def test_operation_definition_is_immutable() -> None:
    definition = operation_definition(KernelOperation.OBSERVE)

    with pytest.raises((AttributeError, TypeError)):
        definition.description = "changed"


def test_mode_policy_creation() -> None:
    policy = KernelModePolicy(
        mode=KernelMode.READ_ONLY,
        allowed_operations=frozenset({KernelOperation.OBSERVE}),
        persistent_internal_changes=False,
        external_changes=False,
        requires_revision_history=False,
        requires_protocol_approval=False,
    )

    assert policy.mode is KernelMode.READ_ONLY
    assert policy.allows(KernelOperation.OBSERVE) is True
    assert policy.allows(KernelOperation.SIMULATE) is False


def test_mode_policy_is_immutable() -> None:
    policy = mode_policy(KernelMode.READ_ONLY)

    with pytest.raises((AttributeError, TypeError)):
        policy.external_changes = True


@pytest.mark.parametrize(
    ("operation", "effect"),
    [
        (KernelOperation.OBSERVE, OperationEffect.NONE),
        (KernelOperation.INSPECT, OperationEffect.NONE),
        (KernelOperation.RECALCULATE, OperationEffect.NONE),
        (KernelOperation.VERIFY, OperationEffect.NONE),
        (KernelOperation.AUDIT, OperationEffect.NONE),
        (KernelOperation.COMPARE, OperationEffect.NONE),
        (KernelOperation.FALSIFY, OperationEffect.NONE),
        (KernelOperation.SIMULATE, OperationEffect.TEMPORARY),
        (KernelOperation.COUNTERFACTUAL, OperationEffect.TEMPORARY),
        (
            KernelOperation.CREATE_REVISION,
            OperationEffect.INTERNAL_PERSISTENT,
        ),
        (
            KernelOperation.ACCEPT_REVISION,
            OperationEffect.INTERNAL_PERSISTENT,
        ),
        (
            KernelOperation.ROLLBACK,
            OperationEffect.INTERNAL_PERSISTENT,
        ),
        (
            KernelOperation.INTERVENE,
            OperationEffect.EXTERNAL_PERSISTENT,
        ),
        (
            KernelOperation.EXECUTE_EXTERNAL_ACTION,
            OperationEffect.EXTERNAL_PERSISTENT,
        ),
    ],
)
def test_operation_effect_contract(
    operation: KernelOperation,
    effect: OperationEffect,
) -> None:
    assert operation_definition(operation).effect is effect


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.ACCEPT_REVISION,
        KernelOperation.INTERVENE,
        KernelOperation.EXECUTE_EXTERNAL_ACTION,
    ],
)
def test_sensitive_operations_require_protocol_approval(
    operation: KernelOperation,
) -> None:
    assert (
        operation_definition(operation).requires_protocol_approval
        is True
    )


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.OBSERVE,
        KernelOperation.INSPECT,
        KernelOperation.RECALCULATE,
        KernelOperation.VERIFY,
        KernelOperation.AUDIT,
        KernelOperation.COMPARE,
        KernelOperation.FALSIFY,
        KernelOperation.ROLLBACK,
    ],
)
def test_epistemically_protected_definition_flags(
    operation: KernelOperation,
) -> None:
    assert (
        operation_definition(operation).epistemically_protected
        is True
    )


def test_epistemically_protected_operations_match_definitions() -> None:
    expected = frozenset(
        operation
        for operation, definition in OPERATION_DEFINITIONS.items()
        if definition.epistemically_protected
    )

    assert epistemically_protected_operations() == expected


def test_epistemically_protected_operations_are_immutable() -> None:
    operations = epistemically_protected_operations()

    assert isinstance(operations, frozenset)

    with pytest.raises(AttributeError):
        operations.add(KernelOperation.SIMULATE)


@pytest.mark.parametrize(
    "operation",
    sorted(
        epistemically_protected_operations(),
        key=lambda item: item.value,
    ),
)
def test_read_only_allows_all_epistemic_operations(
    operation: KernelOperation,
) -> None:
    assert operation_is_allowed(KernelMode.READ_ONLY, operation)


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.SIMULATE,
        KernelOperation.COUNTERFACTUAL,
        KernelOperation.CREATE_REVISION,
        KernelOperation.ACCEPT_REVISION,
        KernelOperation.INTERVENE,
        KernelOperation.EXECUTE_EXTERNAL_ACTION,
    ],
)
def test_read_only_rejects_non_epistemic_operations(
    operation: KernelOperation,
) -> None:
    assert not operation_is_allowed(KernelMode.READ_ONLY, operation)


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.SIMULATE,
        KernelOperation.COUNTERFACTUAL,
    ],
)
def test_sandbox_allows_temporary_operations(
    operation: KernelOperation,
) -> None:
    assert operation_is_allowed(KernelMode.SANDBOX, operation)


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.CREATE_REVISION,
        KernelOperation.ACCEPT_REVISION,
        KernelOperation.INTERVENE,
        KernelOperation.EXECUTE_EXTERNAL_ACTION,
    ],
)
def test_sandbox_rejects_persistent_operations(
    operation: KernelOperation,
) -> None:
    assert not operation_is_allowed(KernelMode.SANDBOX, operation)


@pytest.mark.parametrize("operation", list(KernelOperation))
def test_revision_mode_allows_everything_except_live_actions(
    operation: KernelOperation,
) -> None:
    expected = operation not in {
        KernelOperation.INTERVENE,
        KernelOperation.EXECUTE_EXTERNAL_ACTION,
    }

    assert operation_is_allowed(KernelMode.REVISION, operation) is expected


@pytest.mark.parametrize("operation", list(KernelOperation))
def test_live_mode_allows_all_declared_operations(
    operation: KernelOperation,
) -> None:
    assert operation_is_allowed(KernelMode.LIVE, operation)


def test_mode_capabilities_are_monotonic() -> None:
    read_only = mode_policy(KernelMode.READ_ONLY).allowed_operations
    sandbox = mode_policy(KernelMode.SANDBOX).allowed_operations
    revision = mode_policy(KernelMode.REVISION).allowed_operations
    live = mode_policy(KernelMode.LIVE).allowed_operations

    assert read_only < sandbox
    assert sandbox < revision
    assert revision < live


def test_read_only_policy_has_no_persistent_effects() -> None:
    policy = mode_policy(KernelMode.READ_ONLY)

    assert policy.persistent_internal_changes is False
    assert policy.external_changes is False
    assert policy.requires_revision_history is False
    assert policy.requires_protocol_approval is False


def test_sandbox_policy_has_no_persistent_effects() -> None:
    policy = mode_policy(KernelMode.SANDBOX)

    assert policy.persistent_internal_changes is False
    assert policy.external_changes is False
    assert policy.requires_revision_history is False
    assert policy.requires_protocol_approval is False


def test_revision_policy_requires_history_and_approval() -> None:
    policy = mode_policy(KernelMode.REVISION)

    assert policy.persistent_internal_changes is True
    assert policy.external_changes is False
    assert policy.requires_revision_history is True
    assert policy.requires_protocol_approval is True


def test_live_policy_allows_external_changes() -> None:
    policy = mode_policy(KernelMode.LIVE)

    assert policy.persistent_internal_changes is True
    assert policy.external_changes is True
    assert policy.requires_revision_history is True
    assert policy.requires_protocol_approval is True


@pytest.mark.parametrize("value", [None, "read_only", 1, object()])
def test_mode_policy_rejects_invalid_type(value: object) -> None:
    with pytest.raises(TypeError, match="KernelMode"):
        mode_policy(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "observe", 1, object()])
def test_operation_definition_rejects_invalid_type(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="KernelOperation"):
        operation_definition(value)  # type: ignore[arg-type]


def test_policy_allows_rejects_invalid_operation_type() -> None:
    policy = mode_policy(KernelMode.READ_ONLY)

    with pytest.raises(TypeError, match="KernelOperation"):
        policy.allows("observe")  # type: ignore[arg-type]


def test_require_operation_passes_for_allowed_operation() -> None:
    assert (
        require_operation(
            KernelMode.READ_ONLY,
            KernelOperation.RECALCULATE,
        )
        is None
    )


def test_require_operation_raises_for_disallowed_operation() -> None:
    with pytest.raises(KernelModeError, match="not allowed"):
        require_operation(
            KernelMode.READ_ONLY,
            KernelOperation.INTERVENE,
        )


def test_policy_require_raises_with_mode_and_operation_names() -> None:
    policy = mode_policy(KernelMode.SANDBOX)

    with pytest.raises(KernelModeError) as exc_info:
        policy.require(KernelOperation.ACCEPT_REVISION)

    message = str(exc_info.value)

    assert KernelOperation.ACCEPT_REVISION.value in message
    assert KernelMode.SANDBOX.value in message


@pytest.mark.parametrize(
    "operation",
    [
        KernelOperation.RECALCULATE,
        KernelOperation.VERIFY,
        KernelOperation.AUDIT,
        KernelOperation.COMPARE,
        KernelOperation.FALSIFY,
        KernelOperation.ROLLBACK,
    ],
)
def test_critical_scientific_operations_remain_available(
    operation: KernelOperation,
) -> None:
    assert operation in epistemically_protected_operations()

    for mode in KernelMode:
        assert operation_is_allowed(mode, operation)
