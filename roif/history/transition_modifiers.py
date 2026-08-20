"""
ROIF Memory-to-Transition Integration Layer — transition modifier contract.

This module defines the typed boundary between structured ROIF memory and
physical/SystemImage transition dynamics.

Architecture
------------

Structured memory is NOT reduced to a scalar history gain.

Instead:

    structured memory / associative context / conditioned dynamics
        ->
    explicit bounded transition modifiers
        ->
    SystemEvolution transition channels

The module intentionally does NOT apply modifiers to SystemEvolution yet.

Its role is to provide:

    1. typed transition channels;
    2. bounded modifier values;
    3. explicit target identity;
    4. ordered structured provenance;
    5. immutable modifier sets;
    6. zero-modifier equivalence contract;
    7. deterministic signatures and lookup helpers.

Core invariant
--------------

    zero TransitionModifierSet
        ==
    no change to the existing transition path

when integration is added later.

A non-zero modifier must retain explicit structured provenance. It cannot be
created as an anonymous scalar accumulation of historical traces.
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
import math
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = "transition_modifiers_v1"

MIN_MODIFIER_VALUE = -1.0
MAX_MODIFIER_VALUE = 1.0

ZERO_TOLERANCE = 1e-15


# =============================================================================
# ERRORS
# =============================================================================


class TransitionModifierError(ValueError):
    """
    Raised when a transition-modifier contract is invalid.
    """


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_id(
    name: str,
    value: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TransitionModifierError(
            f"{name} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise TransitionModifierError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_optional_id(
    name: str,
    value: str | None,
) -> str | None:
    if value is None:
        return None

    return _validate_id(
        name,
        value,
    )


def _validate_finite(
    name: str,
    value: float,
) -> float:
    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TransitionModifierError(
            f"{name} must be a finite number"
        ) from exc

    if not math.isfinite(
        number
    ):
        raise TransitionModifierError(
            f"{name} must be finite"
        )

    return number


def _validate_modifier_value(
    value: float,
) -> float:
    number = _validate_finite(
        "value",
        value,
    )

    if not (
        MIN_MODIFIER_VALUE
        <= number
        <= MAX_MODIFIER_VALUE
    ):
        raise TransitionModifierError(
            "value must lie in [-1, 1]"
        )

    return number


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    return MappingProxyType(
        dict(value)
    )


# =============================================================================
# TRANSITION CHANNELS
# =============================================================================


class TransitionChannel(
    str,
    Enum,
):
    """
    Explicit transition channels that structured memory may eventually modify.

    These channels describe *where* a memory-derived modifier is permitted to
    enter transition dynamics.

    They do not prescribe the numerical application rule yet.
    """

    CONNECTION_STIFFNESS_UPDATE = (
        "connection_stiffness_update"
    )

    CONNECTION_CONTRACTILE_CAPACITY_UPDATE = (
        "connection_contractile_capacity_update"
    )

    CONNECTION_REFLEX_GAIN_UPDATE = (
        "connection_reflex_gain_update"
    )

    CONNECTION_FATIGUE_UPDATE = (
        "connection_fatigue_update"
    )

    CONNECTION_REMODELING_BIAS_UPDATE = (
        "connection_remodeling_bias_update"
    )

    PRESTRESS_TRANSFER = (
        "prestress_transfer"
    )

    RESERVE_SENSITIVITY = (
        "reserve_sensitivity"
    )


# =============================================================================
# MEMORY PROVENANCE
# =============================================================================


class MemorySourceKind(
    str,
    Enum,
):
    """
    Structured source categories for transition modifiers.
    """

    HISTORICAL_TRACE = (
        "historical_trace"
    )

    CONDITIONING_MEMORY = (
        "conditioning_memory"
    )

    CONDITIONING_CUE = (
        "conditioning_cue"
    )

    EXPERIENCE_ATTRACTOR = (
        "experience_attractor"
    )

    ASSOCIATIVE_CONTEXT = (
        "associative_context"
    )

    CONTEXTUAL_DYNAMICS = (
        "contextual_dynamics"
    )

    CONDITIONED_DYNAMICS = (
        "conditioned_dynamics"
    )

    RECONSTRUCTION = (
        "reconstruction"
    )


@dataclass(
    frozen=True,
    slots=True,
)
class MemoryEvidenceRef:
    """
    One structured piece of provenance supporting a transition modifier.

    order_index
        Optional position inside an ordered history or memory sequence.

    cue_id / attractor_id / context_id
        Optional semantic links preserving Memory 2.0 structure.

    relation
        Optional semantic relation such as reinforcement, conflict,
        association, reconstruction, or contextual match.

    No numeric aggregation is performed by this object.
    """

    source_kind: MemorySourceKind
    source_id: str

    order_index: int | None = None

    cue_id: str | None = None
    attractor_id: str | None = None
    context_id: str | None = None

    relation: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.source_kind,
            MemorySourceKind,
        ):
            raise TransitionModifierError(
                "source_kind must be a MemorySourceKind"
            )

        source_id = _validate_id(
            "source_id",
            self.source_id,
        )

        cue_id = _validate_optional_id(
            "cue_id",
            self.cue_id,
        )

        attractor_id = _validate_optional_id(
            "attractor_id",
            self.attractor_id,
        )

        context_id = _validate_optional_id(
            "context_id",
            self.context_id,
        )

        relation = _validate_optional_id(
            "relation",
            self.relation,
        )

        order_index = (
            self.order_index
        )

        if order_index is not None:
            if not isinstance(
                order_index,
                int,
            ):
                raise TransitionModifierError(
                    "order_index must be an integer or None"
                )

            if order_index < 0:
                raise TransitionModifierError(
                    "order_index must be >= 0"
                )

        object.__setattr__(
            self,
            "source_id",
            source_id,
        )

        object.__setattr__(
            self,
            "cue_id",
            cue_id,
        )

        object.__setattr__(
            self,
            "attractor_id",
            attractor_id,
        )

        object.__setattr__(
            self,
            "context_id",
            context_id,
        )

        object.__setattr__(
            self,
            "relation",
            relation,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# TRANSITION MODIFIER
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TransitionModifier:
    """
    One bounded modifier for one explicit transition channel and target.

    value
        Normalized modifier signal in [-1, 1].

        The value is intentionally dimensionless at this contract layer.
        Its physical meaning will be defined by the future integration policy
        for the specific TransitionChannel.

    target_id
        Identity of the affected transition target, for example an adaptive
        connection ID or prestress-transfer path ID.

    evidence
        Ordered structured provenance from Memory 2.0 / history.

    Important
    ---------

    A non-zero modifier MUST contain evidence.

    This prevents creation of anonymous values such as:

        history_gain = sum(trace.magnitude)

    without preserving which history, cue, attractor, context, or relation
    generated the transition effect.
    """

    modifier_id: str

    channel: TransitionChannel

    target_id: str

    value: float

    evidence: tuple[
        MemoryEvidenceRef,
        ...,
    ] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        modifier_id = _validate_id(
            "modifier_id",
            self.modifier_id,
        )

        if not isinstance(
            self.channel,
            TransitionChannel,
        ):
            raise TransitionModifierError(
                "channel must be a TransitionChannel"
            )

        target_id = _validate_id(
            "target_id",
            self.target_id,
        )

        value = _validate_modifier_value(
            self.value
        )

        evidence = tuple(
            self.evidence
        )

        if not all(
            isinstance(
                item,
                MemoryEvidenceRef,
            )
            for item in evidence
        ):
            raise TransitionModifierError(
                "evidence must contain only MemoryEvidenceRef objects"
            )

        if (
            abs(value)
            > ZERO_TOLERANCE
            and not evidence
        ):
            raise TransitionModifierError(
                "non-zero modifier requires structured evidence"
            )

        object.__setattr__(
            self,
            "modifier_id",
            modifier_id,
        )

        object.__setattr__(
            self,
            "target_id",
            target_id,
        )

        object.__setattr__(
            self,
            "value",
            value,
        )

        object.__setattr__(
            self,
            "evidence",
            evidence,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )

    @property
    def is_zero(
        self,
    ) -> bool:
        return (
            abs(
                self.value
            )
            <= ZERO_TOLERANCE
        )

    @property
    def key(
        self,
    ) -> tuple[
        TransitionChannel,
        str,
    ]:
        return (
            self.channel,
            self.target_id,
        )


# =============================================================================
# TRANSITION MODIFIER SET
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TransitionModifierSet:
    """
    Immutable set of explicit memory-derived transition modifiers.

    No implicit composition
    -----------------------

    Two modifiers targeting the same:

        (channel, target_id)

    are rejected.

    The derivation layer must resolve reinforcement, conflict, associative
    context, attractor competition, or other memory semantics BEFORE creating
    the final TransitionModifierSet.

    This deliberately prevents the transition layer from silently summing
    heterogeneous historical evidence.
    """

    modifier_set_id: str

    modifiers: tuple[
        TransitionModifier,
        ...,
    ] = ()

    source_memory_ids: tuple[
        str,
        ...,
    ] = ()

    source_context_ids: tuple[
        str,
        ...,
    ] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        modifier_set_id = _validate_id(
            "modifier_set_id",
            self.modifier_set_id,
        )

        modifiers = tuple(
            self.modifiers
        )

        if not all(
            isinstance(
                item,
                TransitionModifier,
            )
            for item in modifiers
        ):
            raise TransitionModifierError(
                "modifiers must contain only TransitionModifier objects"
            )

        keys = tuple(
            modifier.key
            for modifier in modifiers
        )

        if len(
            set(keys)
        ) != len(
            keys
        ):
            raise TransitionModifierError(
                "duplicate modifier channel/target pairs are not allowed"
            )

        modifier_ids = tuple(
            modifier.modifier_id
            for modifier in modifiers
        )

        if len(
            set(modifier_ids)
        ) != len(
            modifier_ids
        ):
            raise TransitionModifierError(
                "modifier_id values must be unique"
            )

        source_memory_ids = tuple(
            _validate_id(
                "source_memory_id",
                value,
            )
            for value
            in self.source_memory_ids
        )

        source_context_ids = tuple(
            _validate_id(
                "source_context_id",
                value,
            )
            for value
            in self.source_context_ids
        )

        if len(
            set(source_memory_ids)
        ) != len(
            source_memory_ids
        ):
            raise TransitionModifierError(
                "source_memory_ids must be unique"
            )

        if len(
            set(source_context_ids)
        ) != len(
            source_context_ids
        ):
            raise TransitionModifierError(
                "source_context_ids must be unique"
            )

        object.__setattr__(
            self,
            "modifier_set_id",
            modifier_set_id,
        )

        object.__setattr__(
            self,
            "modifiers",
            modifiers,
        )

        object.__setattr__(
            self,
            "source_memory_ids",
            source_memory_ids,
        )

        object.__setattr__(
            self,
            "source_context_ids",
            source_context_ids,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )

    @property
    def is_zero(
        self,
    ) -> bool:
        return all(
            modifier.is_zero
            for modifier
            in self.modifiers
        )

    @property
    def has_effect(
        self,
    ) -> bool:
        return not self.is_zero

    @property
    def active_modifiers(
        self,
    ) -> tuple[
        TransitionModifier,
        ...,
    ]:
        return tuple(
            modifier
            for modifier
            in self.modifiers
            if not modifier.is_zero
        )

    @property
    def channels(
        self,
    ) -> tuple[
        TransitionChannel,
        ...,
    ]:
        seen: set[
            TransitionChannel
        ] = set()

        ordered = []

        for modifier in self.modifiers:
            if modifier.channel in seen:
                continue

            seen.add(
                modifier.channel
            )

            ordered.append(
                modifier.channel
            )

        return tuple(
            ordered
        )

    def modifier_for(
        self,
        *,
        channel: TransitionChannel,
        target_id: str,
    ) -> TransitionModifier | None:
        if not isinstance(
            channel,
            TransitionChannel,
        ):
            raise TransitionModifierError(
                "channel must be a TransitionChannel"
            )

        target = _validate_id(
            "target_id",
            target_id,
        )

        for modifier in self.modifiers:
            if (
                modifier.channel
                == channel
                and modifier.target_id
                == target
            ):
                return modifier

        return None

    def modifiers_for_channel(
        self,
        channel: TransitionChannel,
    ) -> tuple[
        TransitionModifier,
        ...,
    ]:
        if not isinstance(
            channel,
            TransitionChannel,
        ):
            raise TransitionModifierError(
                "channel must be a TransitionChannel"
            )

        return tuple(
            modifier
            for modifier
            in self.modifiers
            if modifier.channel
            == channel
        )

    def modifiers_for_target(
        self,
        target_id: str,
    ) -> tuple[
        TransitionModifier,
        ...,
    ]:
        target = _validate_id(
            "target_id",
            target_id,
        )

        return tuple(
            modifier
            for modifier
            in self.modifiers
            if modifier.target_id
            == target
        )


# =============================================================================
# CONSTRUCTION HELPERS
# =============================================================================


def zero_transition_modifier_set(
    *,
    modifier_set_id: str = (
        "zero_transition_modifiers"
    ),
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
) -> TransitionModifierSet:
    """
    Canonical no-effect modifier set.

    Future SystemEvolution integration must satisfy:

        evolve(..., modifiers=zero_transition_modifier_set())
            ==
        current evolve(...) behavior
    """

    return TransitionModifierSet(
        modifier_set_id=(
            modifier_set_id
        ),
        modifiers=(),
        source_memory_ids=(),
        source_context_ids=(),
        metadata=(
            {}
            if metadata is None
            else metadata
        ),
    )


def build_transition_modifier_set(
    *,
    modifier_set_id: str,
    modifiers: Iterable[
        TransitionModifier
    ],
    source_memory_ids: Iterable[
        str
    ] = (),
    source_context_ids: Iterable[
        str
    ] = (),
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
) -> TransitionModifierSet:
    """
    Deterministic constructor from iterable inputs.
    """

    return TransitionModifierSet(
        modifier_set_id=(
            modifier_set_id
        ),
        modifiers=tuple(
            modifiers
        ),
        source_memory_ids=tuple(
            source_memory_ids
        ),
        source_context_ids=tuple(
            source_context_ids
        ),
        metadata=(
            {}
            if metadata is None
            else metadata
        ),
    )


# =============================================================================
# SIGNATURES
# =============================================================================


def memory_evidence_signature(
    evidence: MemoryEvidenceRef,
) -> tuple[
    str,
    str,
    int | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """
    Semantic evidence signature excluding metadata.
    """

    if not isinstance(
        evidence,
        MemoryEvidenceRef,
    ):
        raise TransitionModifierError(
            "evidence must be a MemoryEvidenceRef"
        )

    return (
        evidence.source_kind.value,
        evidence.source_id,
        evidence.order_index,
        evidence.cue_id,
        evidence.attractor_id,
        evidence.context_id,
        evidence.relation,
    )


def transition_modifier_signature(
    modifier: TransitionModifier,
) -> tuple[
    str,
    str,
    str,
    float,
    tuple[
        tuple[
            str,
            str,
            int | None,
            str | None,
            str | None,
            str | None,
            str | None,
        ],
        ...,
    ],
]:
    """
    Deterministic semantic signature excluding metadata.
    """

    if not isinstance(
        modifier,
        TransitionModifier,
    ):
        raise TransitionModifierError(
            "modifier must be a TransitionModifier"
        )

    return (
        modifier.modifier_id,
        modifier.channel.value,
        modifier.target_id,
        modifier.value,
        tuple(
            memory_evidence_signature(
                item
            )
            for item
            in modifier.evidence
        ),
    )


def transition_modifier_set_signature(
    modifier_set: TransitionModifierSet,
) -> tuple[
    str,
    tuple[Any, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """
    Deterministic semantic signature of a complete modifier set.
    """

    if not isinstance(
        modifier_set,
        TransitionModifierSet,
    ):
        raise TransitionModifierError(
            "modifier_set must be a TransitionModifierSet"
        )

    return (
        modifier_set.modifier_set_id,
        tuple(
            transition_modifier_signature(
                modifier
            )
            for modifier
            in modifier_set.modifiers
        ),
        modifier_set.source_memory_ids,
        modifier_set.source_context_ids,
    )


# =============================================================================
# ARCHITECTURAL GUARDS
# =============================================================================


def transition_modifier_set_is_policy_free(
    modifier_set: TransitionModifierSet,
) -> bool:
    """
    Return whether this object remains a descriptive transition contract.

    The contract itself does not:
        - select an intervention;
        - modify a SystemImage;
        - execute a transition;
        - mutate memory;
        - optimize an objective.

    Derivation and application live in separate layers.
    """

    if not isinstance(
        modifier_set,
        TransitionModifierSet,
    ):
        raise TransitionModifierError(
            "modifier_set must be a TransitionModifierSet"
        )

    return (
        modifier_set.metadata.get(
            "action_selected",
            False,
        )
        is False
        and modifier_set.metadata.get(
            "policy_modified",
            False,
        )
        is False
        and modifier_set.metadata.get(
            "memory_mutated",
            False,
        )
        is False
        and modifier_set.metadata.get(
            "transition_executed",
            False,
        )
        is False
    )


def modifier_set_preserves_structured_provenance(
    modifier_set: TransitionModifierSet,
) -> bool:
    """
    Guard against anonymous non-zero history gains.

    Every active modifier must retain at least one structured MemoryEvidenceRef.
    """

    if not isinstance(
        modifier_set,
        TransitionModifierSet,
    ):
        raise TransitionModifierError(
            "modifier_set must be a TransitionModifierSet"
        )

    return all(
        bool(
            modifier.evidence
        )
        for modifier
        in modifier_set.active_modifiers
    )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "MAX_MODIFIER_VALUE",
    "MIN_MODIFIER_VALUE",
    "MemoryEvidenceRef",
    "MemorySourceKind",
    "SCHEMA_VERSION",
    "TransitionChannel",
    "TransitionModifier",
    "TransitionModifierError",
    "TransitionModifierSet",
    "ZERO_TOLERANCE",
    "build_transition_modifier_set",
    "memory_evidence_signature",
    "modifier_set_preserves_structured_provenance",
    "transition_modifier_set_is_policy_free",
    "transition_modifier_set_signature",
    "transition_modifier_signature",
    "zero_transition_modifier_set",
]
