"""
ROIF Engine
Kernel Execution Modes

This module defines the execution modes and permission boundaries of the
ROIF Kernel.

Kernel modes determine whether an operation may only observe, simulate,
create an internal revision, or affect a live external system.

The modes do not evaluate whether an action is beneficial or harmful.
That responsibility belongs to kernel protocols and future structural
evaluation modules.

Author:
    Architect (Dctr Pain)

License:
    See project license.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


class KernelModeError(ValueError):
    """
    Raised when a kernel mode or permission request is invalid.
    """


class KernelMode(str, Enum):
    """
    Execution modes supported by the ROIF Kernel.

    READ_ONLY
        Observation, inspection, comparison, and verification only.
        No persistent state modification is allowed.

    SANDBOX
        Temporary simulation and counterfactual exploration are allowed.
        Results must not modify the accepted persistent state.

    REVISION
        A new immutable internal revision may be created.
        Previous accepted states must remain available.

    LIVE
        A validated operation may affect an external or operational
        system. This mode requires additional protocol approval.
    """

    READ_ONLY = "read_only"
    SANDBOX = "sandbox"
    REVISION = "revision"
    LIVE = "live"


class KernelOperation(str, Enum):
    """
    Operation classes understood by the kernel permission system.
    """

    OBSERVE = "observe"
    INSPECT = "inspect"
    RECALCULATE = "recalculate"
    VERIFY = "verify"
    AUDIT = "audit"
    COMPARE = "compare"
    FALSIFY = "falsify"

    SIMULATE = "simulate"
    COUNTERFACTUAL = "counterfactual"

    CREATE_REVISION = "create_revision"
    ACCEPT_REVISION = "accept_revision"
    ROLLBACK = "rollback"

    INTERVENE = "intervene"
    EXECUTE_EXTERNAL_ACTION = "execute_external_action"


class OperationEffect(str, Enum):
    """
    Maximum structural effect associated with an operation.
    """

    NONE = "none"
    TEMPORARY = "temporary"
    INTERNAL_PERSISTENT = "internal_persistent"
    EXTERNAL_PERSISTENT = "external_persistent"


@dataclass(frozen=True, slots=True)
class OperationDefinition:
    """
    Immutable definition of one kernel operation.
    """

    operation: KernelOperation
    effect: OperationEffect
    description: str
    epistemically_protected: bool = False
    requires_protocol_approval: bool = False


@dataclass(frozen=True, slots=True)
class KernelModePolicy:
    """
    Immutable permission policy for one kernel mode.
    """

    mode: KernelMode
    allowed_operations: frozenset[KernelOperation]
    persistent_internal_changes: bool
    external_changes: bool
    requires_revision_history: bool
    requires_protocol_approval: bool

    def allows(
        self,
        operation: KernelOperation,
    ) -> bool:
        """
        Return True when the operation is allowed in this mode.
        """

        if not isinstance(operation, KernelOperation):
            raise TypeError(
                "operation must be a KernelOperation"
            )

        return operation in self.allowed_operations

    def require(
        self,
        operation: KernelOperation,
    ) -> None:
        """
        Raise KernelModeError when an operation is not allowed.
        """

        if not self.allows(operation):
            raise KernelModeError(
                f"Operation {operation.value!r} is not allowed "
                f"in kernel mode {self.mode.value!r}."
            )


OPERATION_DEFINITIONS: Final[
    Mapping[KernelOperation, OperationDefinition]
] = MappingProxyType(
    {
        KernelOperation.OBSERVE: OperationDefinition(
            operation=KernelOperation.OBSERVE,
            effect=OperationEffect.NONE,
            description=(
                "Read observations without modifying accepted state."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.INSPECT: OperationDefinition(
            operation=KernelOperation.INSPECT,
            effect=OperationEffect.NONE,
            description=(
                "Inspect internal structures and recorded state."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.RECALCULATE: OperationDefinition(
            operation=KernelOperation.RECALCULATE,
            effect=OperationEffect.NONE,
            description=(
                "Repeat a calculation without modifying accepted state."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.VERIFY: OperationDefinition(
            operation=KernelOperation.VERIFY,
            effect=OperationEffect.NONE,
            description=(
                "Verify a result, invariant, or structural claim."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.AUDIT: OperationDefinition(
            operation=KernelOperation.AUDIT,
            effect=OperationEffect.NONE,
            description=(
                "Audit data, calculations, decisions, or provenance."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.COMPARE: OperationDefinition(
            operation=KernelOperation.COMPARE,
            effect=OperationEffect.NONE,
            description=(
                "Compare structures, states, revisions, or hypotheses."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.FALSIFY: OperationDefinition(
            operation=KernelOperation.FALSIFY,
            effect=OperationEffect.NONE,
            description=(
                "Attempt to disprove an existing hypothesis or model."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.SIMULATE: OperationDefinition(
            operation=KernelOperation.SIMULATE,
            effect=OperationEffect.TEMPORARY,
            description=(
                "Execute a temporary isolated simulation."
            ),
        ),
        KernelOperation.COUNTERFACTUAL: OperationDefinition(
            operation=KernelOperation.COUNTERFACTUAL,
            effect=OperationEffect.TEMPORARY,
            description=(
                "Evaluate alternative possible futures without "
                "changing accepted state."
            ),
        ),
        KernelOperation.CREATE_REVISION: OperationDefinition(
            operation=KernelOperation.CREATE_REVISION,
            effect=OperationEffect.INTERNAL_PERSISTENT,
            description=(
                "Create a new immutable internal state revision."
            ),
        ),
        KernelOperation.ACCEPT_REVISION: OperationDefinition(
            operation=KernelOperation.ACCEPT_REVISION,
            effect=OperationEffect.INTERNAL_PERSISTENT,
            description=(
                "Mark a validated revision as the accepted state."
            ),
            requires_protocol_approval=True,
        ),
        KernelOperation.ROLLBACK: OperationDefinition(
            operation=KernelOperation.ROLLBACK,
            effect=OperationEffect.INTERNAL_PERSISTENT,
            description=(
                "Restore a previous accepted internal revision."
            ),
            epistemically_protected=True,
        ),
        KernelOperation.INTERVENE: OperationDefinition(
            operation=KernelOperation.INTERVENE,
            effect=OperationEffect.EXTERNAL_PERSISTENT,
            description=(
                "Apply a structural intervention to an observed system."
            ),
            requires_protocol_approval=True,
        ),
        KernelOperation.EXECUTE_EXTERNAL_ACTION: OperationDefinition(
            operation=KernelOperation.EXECUTE_EXTERNAL_ACTION,
            effect=OperationEffect.EXTERNAL_PERSISTENT,
            description=(
                "Execute an action outside the internal ROIF model."
            ),
            requires_protocol_approval=True,
        ),
    }
)


_EPISTEMIC_OPERATIONS: Final[frozenset[KernelOperation]] = (
    frozenset(
        operation
        for operation, definition
        in OPERATION_DEFINITIONS.items()
        if definition.epistemically_protected
    )
)

_SIMULATION_OPERATIONS: Final[frozenset[KernelOperation]] = (
    frozenset(
        {
            KernelOperation.SIMULATE,
            KernelOperation.COUNTERFACTUAL,
        }
    )
)

_REVISION_OPERATIONS: Final[frozenset[KernelOperation]] = (
    frozenset(
        {
            KernelOperation.CREATE_REVISION,
            KernelOperation.ACCEPT_REVISION,
            KernelOperation.ROLLBACK,
        }
    )
)

_LIVE_OPERATIONS: Final[frozenset[KernelOperation]] = (
    frozenset(
        {
            KernelOperation.INTERVENE,
            KernelOperation.EXECUTE_EXTERNAL_ACTION,
        }
    )
)


KERNEL_MODE_POLICIES: Final[
    Mapping[KernelMode, KernelModePolicy]
] = MappingProxyType(
    {
        KernelMode.READ_ONLY: KernelModePolicy(
            mode=KernelMode.READ_ONLY,
            allowed_operations=_EPISTEMIC_OPERATIONS,
            persistent_internal_changes=False,
            external_changes=False,
            requires_revision_history=False,
            requires_protocol_approval=False,
        ),
        KernelMode.SANDBOX: KernelModePolicy(
            mode=KernelMode.SANDBOX,
            allowed_operations=(
                _EPISTEMIC_OPERATIONS
                | _SIMULATION_OPERATIONS
            ),
            persistent_internal_changes=False,
            external_changes=False,
            requires_revision_history=False,
            requires_protocol_approval=False,
        ),
        KernelMode.REVISION: KernelModePolicy(
            mode=KernelMode.REVISION,
            allowed_operations=(
                _EPISTEMIC_OPERATIONS
                | _SIMULATION_OPERATIONS
                | _REVISION_OPERATIONS
            ),
            persistent_internal_changes=True,
            external_changes=False,
            requires_revision_history=True,
            requires_protocol_approval=True,
        ),
        KernelMode.LIVE: KernelModePolicy(
            mode=KernelMode.LIVE,
            allowed_operations=(
                _EPISTEMIC_OPERATIONS
                | _SIMULATION_OPERATIONS
                | _REVISION_OPERATIONS
                | _LIVE_OPERATIONS
            ),
            persistent_internal_changes=True,
            external_changes=True,
            requires_revision_history=True,
            requires_protocol_approval=True,
        ),
    }
)


def operation_definition(
    operation: KernelOperation,
) -> OperationDefinition:
    """
    Return the immutable definition of an operation.
    """

    if not isinstance(operation, KernelOperation):
        raise TypeError(
            "operation must be a KernelOperation"
        )

    return OPERATION_DEFINITIONS[operation]


def mode_policy(
    mode: KernelMode,
) -> KernelModePolicy:
    """
    Return the immutable permission policy for a kernel mode.
    """

    if not isinstance(mode, KernelMode):
        raise TypeError(
            "mode must be a KernelMode"
        )

    return KERNEL_MODE_POLICIES[mode]


def operation_is_allowed(
    mode: KernelMode,
    operation: KernelOperation,
) -> bool:
    """
    Return whether an operation is permitted in a kernel mode.
    """

    return mode_policy(mode).allows(operation)


def require_operation(
    mode: KernelMode,
    operation: KernelOperation,
) -> None:
    """
    Require an operation to be permitted in a kernel mode.
    """

    mode_policy(mode).require(operation)


def epistemically_protected_operations(
) -> frozenset[KernelOperation]:
    """
    Return operations that must remain available for verification.

    These operations must not later be classified as harmful merely
    because they may contradict or weaken the current internal model.
    """

    return _EPISTEMIC_OPERATIONS


__all__ = [
    "KERNEL_MODE_POLICIES",
    "OPERATION_DEFINITIONS",
    "KernelMode",
    "KernelModeError",
    "KernelModePolicy",
    "KernelOperation",
    "OperationDefinition",
    "OperationEffect",
    "epistemically_protected_operations",
    "mode_policy",
    "operation_definition",
    "operation_is_allowed",
    "require_operation",
]