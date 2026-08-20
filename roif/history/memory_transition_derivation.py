"""
ROIF Memory-to-Transition Integration Layer
Structured memory -> TransitionModifierSet derivation.

Purpose
-------

Convert already-computed structured ROIF memory semantics into explicit,
bounded transition modifiers without allowing the memory layer to silently
choose physical topology or transition targets.

Architecture
------------

Conditioning branch:

    ConditioningMemory
        + query ConditioningCue
        -> ConditionedResponse
        -> predicted attractor + conditioned response strength
        -> explicit TransitionTargetBinding
        -> TransitionModifierSet


Associative-context branch:

    AssociativeContext
        -> attractor-specific context prestress
        -> explicit TransitionTargetBinding(s)
        -> TransitionModifierSet


Critical boundary
-----------------

Memory semantics do NOT determine physical target identity automatically.

The mapping:

    semantic attractor
        -> physical transition target

must be supplied explicitly as TransitionTargetBinding.

Therefore this module does not:

    - infer topology;
    - create new physical connections;
    - select an intervention;
    - optimize an objective;
    - mutate memory;
    - execute SystemEvolution;
    - collapse all history into one global history_gain.

The first executable physical channel remains:

    TransitionChannel.PRESTRESS_TRANSFER

but the binding object is typed generally enough for future explicitly
supported channels.
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
import math
from types import MappingProxyType
from typing import Any


from roif.history.associative_context import (
    AssociativeContext,
)

from roif.history.experience_conditioning import (
    ConditioningCue,
    ConditioningMemory,
    conditioned_response,
)

from roif.history.transition_modifiers import (
    MAX_MODIFIER_VALUE,
    MIN_MODIFIER_VALUE,
    MemoryEvidenceRef,
    MemorySourceKind,
    TransitionChannel,
    TransitionModifier,
    TransitionModifierSet,
)


SCHEMA_VERSION = (
    "memory_transition_derivation_v1"
)

DEFAULT_MAX_ABS_MODIFIER = 1.0

ZERO_TOLERANCE = 1e-15


# =============================================================================
# ERRORS
# =============================================================================


class MemoryTransitionDerivationError(
    ValueError
):
    """
    Raised when structured memory cannot be safely mapped to transition
    modifiers under the explicit derivation contract.
    """


# =============================================================================
# VALIDATION
# =============================================================================


def _validate_id(
    name: str,
    value: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise MemoryTransitionDerivationError(
            f"{name} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise MemoryTransitionDerivationError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise MemoryTransitionDerivationError(
            f"{name} must be a finite number"
        ) from exc

    if not math.isfinite(
        result
    ):
        raise MemoryTransitionDerivationError(
            f"{name} must be finite"
        )

    return result


def _validate_unit_interval(
    name: str,
    value: float,
) -> float:
    number = _validate_finite(
        name,
        value,
    )

    if not (
        0.0
        <= number
        <= 1.0
    ):
        raise MemoryTransitionDerivationError(
            f"{name} must lie in [0, 1]"
        )

    return number


def _validate_polarity(
    value: float,
) -> float:
    number = _validate_finite(
        "polarity",
        value,
    )

    if number not in (
        -1.0,
        1.0,
    ):
        raise MemoryTransitionDerivationError(
            "polarity must be exactly -1.0 or 1.0"
        )

    return number


def _readonly_mapping(
    value: Mapping[
        str,
        Any,
    ] | None,
) -> Mapping[
    str,
    Any,
]:
    return MappingProxyType(
        {}
        if value is None
        else dict(
            value
        )
    )


def _clamp_modifier(
    value: float,
    max_abs_modifier: float,
) -> float:
    limit = _validate_unit_interval(
        "max_abs_modifier",
        max_abs_modifier,
    )

    number = _validate_finite(
        "modifier value",
        value,
    )

    bounded = max(
        -limit,
        min(
            limit,
            number,
        ),
    )

    return max(
        MIN_MODIFIER_VALUE,
        min(
            MAX_MODIFIER_VALUE,
            bounded,
        ),
    )


# =============================================================================
# EXPLICIT SEMANTIC -> PHYSICAL BINDING
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TransitionTargetBinding:
    """
    Explicit mapping from one semantic memory target to one physical transition
    target.

    semantic_target_id
        Currently an attractor ID generated by structured Memory 2.0.

    physical_target_id
        Explicit SystemEvolution target. For PRESTRESS_TRANSFER this is an
        adaptive connection ID such as "A_B".

    channel
        Transition channel to modify.

    polarity
        Explicit directional semantics:

            +1 -> memory signal increases the channel
            -1 -> memory signal decreases the channel

        Polarity is NOT inferred from attractor names.

    scale
        Bounded conversion strength in [0, 1].

        This is a declared mapping coefficient, not a learned or hidden
        history gain.

    Important
    ---------

    The binding itself contains no memory state and performs no transition.
    """

    binding_id: str

    semantic_target_id: str

    physical_target_id: str

    channel: TransitionChannel = (
        TransitionChannel.PRESTRESS_TRANSFER
    )

    polarity: float = 1.0

    scale: float = 1.0

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        binding_id = _validate_id(
            "binding_id",
            self.binding_id,
        )

        semantic_target_id = (
            _validate_id(
                "semantic_target_id",
                self.semantic_target_id,
            )
        )

        physical_target_id = (
            _validate_id(
                "physical_target_id",
                self.physical_target_id,
            )
        )

        if not isinstance(
            self.channel,
            TransitionChannel,
        ):
            raise MemoryTransitionDerivationError(
                "channel must be a TransitionChannel"
            )

        polarity = _validate_polarity(
            self.polarity
        )

        scale = _validate_unit_interval(
            "scale",
            self.scale,
        )

        object.__setattr__(
            self,
            "binding_id",
            binding_id,
        )

        object.__setattr__(
            self,
            "semantic_target_id",
            semantic_target_id,
        )

        object.__setattr__(
            self,
            "physical_target_id",
            physical_target_id,
        )

        object.__setattr__(
            self,
            "polarity",
            polarity,
        )

        object.__setattr__(
            self,
            "scale",
            scale,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )

    @property
    def transition_key(
        self,
    ) -> tuple[
        TransitionChannel,
        str,
    ]:
        return (
            self.channel,
            self.physical_target_id,
        )


# =============================================================================
# DERIVATION RESULT
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class MemoryTransitionDerivationResult:
    """
    Descriptive result of one memory-to-transition derivation.

    modifier_set
        Final explicit transition contract.

    semantic_signal_by_target
        Memory-derived signal BEFORE physical binding polarity/scale.

    applied_binding_ids
        Explicit bindings that participated in the result.

    No SystemEvolution transition is executed here.
    """

    derivation_id: str

    derivation_mode: str

    modifier_set: TransitionModifierSet

    semantic_signal_by_target: Mapping[
        str,
        float,
    ]

    applied_binding_ids: tuple[
        str,
        ...,
    ]

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        derivation_id = _validate_id(
            "derivation_id",
            self.derivation_id,
        )

        derivation_mode = _validate_id(
            "derivation_mode",
            self.derivation_mode,
        )

        if not isinstance(
            self.modifier_set,
            TransitionModifierSet,
        ):
            raise MemoryTransitionDerivationError(
                "modifier_set must be a TransitionModifierSet"
            )

        signals = {}

        for target_id, value in (
            self.semantic_signal_by_target.items()
        ):
            target = _validate_id(
                "semantic target id",
                target_id,
            )

            signal = _validate_unit_interval(
                "semantic signal",
                value,
            )

            signals[
                target
            ] = signal

        binding_ids = tuple(
            _validate_id(
                "binding_id",
                value,
            )
            for value
            in self.applied_binding_ids
        )

        if len(
            set(binding_ids)
        ) != len(
            binding_ids
        ):
            raise MemoryTransitionDerivationError(
                "applied_binding_ids must be unique"
            )

        object.__setattr__(
            self,
            "derivation_id",
            derivation_id,
        )

        object.__setattr__(
            self,
            "derivation_mode",
            derivation_mode,
        )

        object.__setattr__(
            self,
            "semantic_signal_by_target",
            MappingProxyType(
                signals
            ),
        )

        object.__setattr__(
            self,
            "applied_binding_ids",
            binding_ids,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )


# =============================================================================
# BINDING VALIDATION
# =============================================================================


def _validate_bindings(
    bindings: Iterable[
        TransitionTargetBinding
    ],
) -> tuple[
    TransitionTargetBinding,
    ...,
]:
    items = tuple(
        bindings
    )

    if not items:
        raise MemoryTransitionDerivationError(
            "bindings must not be empty"
        )

    if not all(
        isinstance(
            item,
            TransitionTargetBinding,
        )
        for item
        in items
    ):
        raise MemoryTransitionDerivationError(
            "bindings must contain only TransitionTargetBinding objects"
        )

    binding_ids = tuple(
        item.binding_id
        for item
        in items
    )

    if len(
        set(binding_ids)
    ) != len(
        binding_ids
    ):
        raise MemoryTransitionDerivationError(
            "binding_id values must be unique"
        )

    transition_keys = tuple(
        item.transition_key
        for item
        in items
    )

    if len(
        set(transition_keys)
    ) != len(
        transition_keys
    ):
        raise MemoryTransitionDerivationError(
            "multiple bindings may not resolve to the same "
            "(channel, physical_target_id) in one derivation"
        )

    return items


def _binding_by_semantic_target(
    bindings: tuple[
        TransitionTargetBinding,
        ...,
    ],
) -> dict[
    str,
    TransitionTargetBinding,
]:
    indexed = {}

    for binding in bindings:
        semantic_id = (
            binding.semantic_target_id
        )

        if semantic_id in indexed:
            raise MemoryTransitionDerivationError(
                "multiple bindings for the same semantic_target_id "
                "are not supported in v1"
            )

        indexed[
            semantic_id
        ] = binding

    return indexed


# =============================================================================
# CONDITIONING MEMORY -> TRANSITION MODIFIERS
# =============================================================================


def derive_conditioning_transition_modifiers(
    *,
    derivation_id: str,
    query_cue: ConditioningCue,
    memory: ConditioningMemory,
    bindings: Iterable[
        TransitionTargetBinding
    ],
    generalization_scale: float = 1.0,
    max_abs_modifier: float = (
        DEFAULT_MAX_ABS_MODIFIER
    ),
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
) -> MemoryTransitionDerivationResult:
    """
    Derive one explicit transition modifier from ConditioningMemory.

    Existing Memory 2.0 semantics are used unchanged:

        conditioned_response(
            query_cue,
            memory,
        )

    provides:

        predicted_attractor_id
        conditioned_response_strength
        cue_distance
        generalization_weight

    The predicted attractor MUST have an explicit TransitionTargetBinding.

    Physical modifier:

        modifier
            =
        conditioned_response_strength
        * binding.scale
        * binding.polarity

    bounded by max_abs_modifier.

    No target is inferred from the attractor name.
    """

    derivation_id = _validate_id(
        "derivation_id",
        derivation_id,
    )

    if not isinstance(
        query_cue,
        ConditioningCue,
    ):
        raise MemoryTransitionDerivationError(
            "query_cue must be a ConditioningCue"
        )

    if not isinstance(
        memory,
        ConditioningMemory,
    ):
        raise MemoryTransitionDerivationError(
            "memory must be a ConditioningMemory"
        )

    binding_items = _validate_bindings(
        bindings
    )

    binding_index = (
        _binding_by_semantic_target(
            binding_items
        )
    )

    response = conditioned_response(
        query_cue=query_cue,
        memory=memory,
        generalization_scale=(
            generalization_scale
        ),
    )

    semantic_target = (
        response.predicted_attractor_id
    )

    if semantic_target not in (
        binding_index
    ):
        raise MemoryTransitionDerivationError(
            "no explicit physical transition binding for "
            f"predicted attractor: {semantic_target}"
        )

    binding = (
        binding_index[
            semantic_target
        ]
    )

    semantic_signal = (
        _validate_unit_interval(
            "conditioned_response_strength",
            response.conditioned_response_strength,
        )
    )

    raw_modifier = (
        semantic_signal
        * binding.scale
        * binding.polarity
    )

    value = _clamp_modifier(
        raw_modifier,
        max_abs_modifier,
    )

    evidence = (
        MemoryEvidenceRef(
            source_kind=(
                MemorySourceKind.CONDITIONING_MEMORY
            ),
            source_id=(
                memory.memory_id
            ),
            cue_id=(
                query_cue.cue_id
            ),
            attractor_id=(
                semantic_target
            ),
            relation=(
                "conditioned_response"
            ),
            metadata={
                "schema_version": (
                    SCHEMA_VERSION
                ),
                "matched_memory_cue_id": (
                    response.matched_memory_cue_id
                ),
                "cue_distance": (
                    response.cue_distance
                ),
                "generalization_weight": (
                    response.generalization_weight
                ),
                "conditioned_response_strength": (
                    semantic_signal
                ),
                "binding_id": (
                    binding.binding_id
                ),
            },
        ),
    )

    modifier = TransitionModifier(
        modifier_id=(
            f"{derivation_id}::"
            f"{binding.binding_id}::modifier"
        ),
        channel=(
            binding.channel
        ),
        target_id=(
            binding.physical_target_id
        ),
        value=value,
        evidence=evidence,
        metadata={
            "schema_version": (
                SCHEMA_VERSION
            ),
            "derivation_mode": (
                "conditioning_memory"
            ),
            "binding_id": (
                binding.binding_id
            ),
            "semantic_target_id": (
                semantic_target
            ),
            "binding_scale": (
                binding.scale
            ),
            "binding_polarity": (
                binding.polarity
            ),
        },
    )

    modifier_set = (
        TransitionModifierSet(
            modifier_set_id=(
                f"{derivation_id}::modifier_set"
            ),
            modifiers=(
                modifier,
            ),
            source_memory_ids=(
                memory.memory_id,
            ),
            source_context_ids=(),
            metadata={
                "schema_version": (
                    SCHEMA_VERSION
                ),
                "derivation_mode": (
                    "conditioning_memory"
                ),
                "action_selected": False,
                "policy_modified": False,
                "memory_mutated": False,
                "transition_executed": False,
            },
        )
    )

    merged_metadata = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "source_memory_id": (
            memory.memory_id
        ),
        "query_cue_id": (
            query_cue.cue_id
        ),
        "predicted_attractor_id": (
            semantic_target
        ),
        "memory_mutated": False,
        "action_selected": False,
        "policy_modified": False,
        "transition_executed": False,
        "biological_truth_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return (
        MemoryTransitionDerivationResult(
            derivation_id=(
                derivation_id
            ),
            derivation_mode=(
                "conditioning_memory"
            ),
            modifier_set=(
                modifier_set
            ),
            semantic_signal_by_target={
                semantic_target: (
                    semantic_signal
                ),
            },
            applied_binding_ids=(
                binding.binding_id,
            ),
            metadata=(
                merged_metadata
            ),
        )
    )


# =============================================================================
# ASSOCIATIVE CONTEXT -> TRANSITION MODIFIERS
# =============================================================================


def derive_context_transition_modifiers(
    *,
    derivation_id: str,
    context: AssociativeContext,
    bindings: Iterable[
        TransitionTargetBinding
    ],
    max_abs_modifier: float = (
        DEFAULT_MAX_ABS_MODIFIER
    ),
    include_zero_signals: bool = False,
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
) -> MemoryTransitionDerivationResult:
    """
    Derive explicit modifiers from attractor-specific AssociativeContext bias.

    Existing context.prestress.bias_by_attractor_id is treated as the semantic
    signal map.

    For each explicitly bound attractor:

        modifier
            =
        context_bias(attractor)
        * binding.scale
        * binding.polarity

    bounded by max_abs_modifier.

    Important
    ---------

    Different attractors remain separate semantic targets.

    They are NOT summed into one context/history gain.

    v1 intentionally rejects multiple semantic targets resolving to the same
    physical (channel, target) pair. Conflict/composition semantics belong in a
    future explicit resolution layer rather than silent addition here.
    """

    derivation_id = _validate_id(
        "derivation_id",
        derivation_id,
    )

    if not isinstance(
        context,
        AssociativeContext,
    ):
        raise MemoryTransitionDerivationError(
            "context must be an AssociativeContext"
        )

    if not isinstance(
        include_zero_signals,
        bool,
    ):
        raise MemoryTransitionDerivationError(
            "include_zero_signals must be bool"
        )

    binding_items = _validate_bindings(
        bindings
    )

    binding_index = (
        _binding_by_semantic_target(
            binding_items
        )
    )

    semantic_signals = {}

    for attractor_id, raw_value in (
        context.prestress
        .bias_by_attractor_id
        .items()
    ):
        semantic_signals[
            _validate_id(
                "attractor_id",
                attractor_id,
            )
        ] = _validate_unit_interval(
            "context attractor bias",
            raw_value,
        )

    unknown_context_targets = tuple(
        sorted(
            semantic_target
            for semantic_target
            in semantic_signals
            if (
                semantic_target
                not in binding_index
            )
        )
    )

    if unknown_context_targets:
        raise MemoryTransitionDerivationError(
            "missing explicit physical transition bindings for "
            "context attractor(s): "
            + ", ".join(
                unknown_context_targets
            )
        )

    modifiers = []
    applied_bindings = []

    for semantic_target in sorted(
        semantic_signals
    ):
        signal = (
            semantic_signals[
                semantic_target
            ]
        )

        if (
            signal
            <= ZERO_TOLERANCE
            and not include_zero_signals
        ):
            continue

        binding = (
            binding_index[
                semantic_target
            ]
        )

        raw_modifier = (
            signal
            * binding.scale
            * binding.polarity
        )

        value = _clamp_modifier(
            raw_modifier,
            max_abs_modifier,
        )

        evidence = (
            MemoryEvidenceRef(
                source_kind=(
                    MemorySourceKind.ASSOCIATIVE_CONTEXT
                ),
                source_id=(
                    context.context_id
                ),
                attractor_id=(
                    semantic_target
                ),
                context_id=(
                    context.context_id
                ),
                relation=(
                    "context_attractor_bias"
                ),
                metadata={
                    "schema_version": (
                        SCHEMA_VERSION
                    ),
                    "semantic_bias": (
                        signal
                    ),
                    "context_confidence": (
                        context.context_confidence
                    ),
                    "competition_margin": (
                        context.competition_margin
                    ),
                    "dominant_attractor_id": (
                        context.dominant_attractor_id
                    ),
                    "binding_id": (
                        binding.binding_id
                    ),
                },
            ),
        )

        modifiers.append(
            TransitionModifier(
                modifier_id=(
                    f"{derivation_id}::"
                    f"{binding.binding_id}::modifier"
                ),
                channel=(
                    binding.channel
                ),
                target_id=(
                    binding.physical_target_id
                ),
                value=value,
                evidence=evidence,
                metadata={
                    "schema_version": (
                        SCHEMA_VERSION
                    ),
                    "derivation_mode": (
                        "associative_context"
                    ),
                    "binding_id": (
                        binding.binding_id
                    ),
                    "semantic_target_id": (
                        semantic_target
                    ),
                    "binding_scale": (
                        binding.scale
                    ),
                    "binding_polarity": (
                        binding.polarity
                    ),
                },
            )
        )

        applied_bindings.append(
            binding.binding_id
        )

    source_memory_ids = tuple(
        sorted(
            {
                contribution.memory_id
                for contribution
                in context.contributions
            }
        )
    )

    modifier_set = (
        TransitionModifierSet(
            modifier_set_id=(
                f"{derivation_id}::modifier_set"
            ),
            modifiers=tuple(
                modifiers
            ),
            source_memory_ids=(
                source_memory_ids
            ),
            source_context_ids=(
                context.context_id,
            ),
            metadata={
                "schema_version": (
                    SCHEMA_VERSION
                ),
                "derivation_mode": (
                    "associative_context"
                ),
                "action_selected": False,
                "policy_modified": False,
                "memory_mutated": False,
                "transition_executed": False,
            },
        )
    )

    merged_metadata = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "source_context_id": (
            context.context_id
        ),
        "dominant_attractor_id": (
            context.dominant_attractor_id
        ),
        "context_confidence": (
            context.context_confidence
        ),
        "competition_margin": (
            context.competition_margin
        ),
        "memory_mutated": False,
        "action_selected": False,
        "policy_modified": False,
        "transition_executed": False,
        "biological_truth_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return (
        MemoryTransitionDerivationResult(
            derivation_id=(
                derivation_id
            ),
            derivation_mode=(
                "associative_context"
            ),
            modifier_set=(
                modifier_set
            ),
            semantic_signal_by_target=(
                semantic_signals
            ),
            applied_binding_ids=tuple(
                applied_bindings
            ),
            metadata=(
                merged_metadata
            ),
        )
    )


# =============================================================================
# SIGNATURES
# =============================================================================


def transition_target_binding_signature(
    binding: TransitionTargetBinding,
) -> tuple[
    str,
    str,
    str,
    str,
    float,
    float,
]:
    if not isinstance(
        binding,
        TransitionTargetBinding,
    ):
        raise MemoryTransitionDerivationError(
            "binding must be a TransitionTargetBinding"
        )

    return (
        binding.binding_id,
        binding.semantic_target_id,
        binding.physical_target_id,
        binding.channel.value,
        binding.polarity,
        binding.scale,
    )


def memory_transition_derivation_signature(
    result: MemoryTransitionDerivationResult,
) -> tuple[
    str,
    str,
    tuple[
        tuple[
            str,
            float,
        ],
        ...,
    ],
    tuple[
        str,
        ...,
    ],
    tuple[
        tuple[
            str,
            str,
            str,
            float,
        ],
        ...,
    ],
]:
    """
    Deterministic semantic signature excluding metadata.
    """

    if not isinstance(
        result,
        MemoryTransitionDerivationResult,
    ):
        raise MemoryTransitionDerivationError(
            "result must be a MemoryTransitionDerivationResult"
        )

    modifier_signature = tuple(
        (
            modifier.channel.value,
            modifier.target_id,
            modifier.modifier_id,
            modifier.value,
        )
        for modifier
        in result.modifier_set.modifiers
    )

    return (
        result.derivation_id,
        result.derivation_mode,
        tuple(
            sorted(
                result
                .semantic_signal_by_target
                .items()
            )
        ),
        result.applied_binding_ids,
        modifier_signature,
    )


# =============================================================================
# ARCHITECTURAL GUARDS
# =============================================================================


def derivation_is_policy_free(
    result: MemoryTransitionDerivationResult,
) -> bool:
    """
    Derivation maps memory semantics into a transition contract only.

    It must not claim to:
        - select an action;
        - modify a policy;
        - mutate memory;
        - execute the physical transition.
    """

    if not isinstance(
        result,
        MemoryTransitionDerivationResult,
    ):
        raise MemoryTransitionDerivationError(
            "result must be a MemoryTransitionDerivationResult"
        )

    return (
        result.metadata.get(
            "action_selected",
            False,
        )
        is False
        and result.metadata.get(
            "policy_modified",
            False,
        )
        is False
        and result.metadata.get(
            "memory_mutated",
            False,
        )
        is False
        and result.metadata.get(
            "transition_executed",
            False,
        )
        is False
    )


def derivation_preserves_explicit_binding(
    result: MemoryTransitionDerivationResult,
) -> bool:
    """
    Every produced modifier must remain traceable to one explicit binding.
    """

    if not isinstance(
        result,
        MemoryTransitionDerivationResult,
    ):
        raise MemoryTransitionDerivationError(
            "result must be a MemoryTransitionDerivationResult"
        )

    if not result.modifier_set.modifiers:
        return True

    binding_ids = set(
        result.applied_binding_ids
    )

    return all(
        modifier.metadata.get(
            "binding_id"
        )
        in binding_ids
        for modifier
        in result.modifier_set.modifiers
    )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "DEFAULT_MAX_ABS_MODIFIER",
    "MemoryTransitionDerivationError",
    "MemoryTransitionDerivationResult",
    "SCHEMA_VERSION",
    "TransitionTargetBinding",
    "ZERO_TOLERANCE",
    "derivation_is_policy_free",
    "derivation_preserves_explicit_binding",
    "derive_conditioning_transition_modifiers",
    "derive_context_transition_modifiers",
    "memory_transition_derivation_signature",
    "transition_target_binding_signature",
]
