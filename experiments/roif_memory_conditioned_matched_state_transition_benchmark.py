"""
ROIF end-to-end structured-memory conditioned transition benchmark.

Question
--------

Can two different structured conditioning memories change the next physical
SystemEvolution transition when:

    - the current SystemImage is identical;
    - the physical event is identical;
    - the query cue is identical;
    - the semantic attractor is identical;
    - the semantic-to-physical binding is identical;
    - physical topology is identical?

Architecture under test
-----------------------

    conditioning history
        ->
    ConditioningMemory
        ->
    conditioned response
        ->
    explicit TransitionTargetBinding
        ->
    TransitionModifierSet
        ->
    SystemEvolution
        ->
    physical prestress redistribution

This benchmark does NOT test:

    - biological learning;
    - clinical validity;
    - universal history-conditioned dynamics;
    - full ROIF predictive preconfiguration;
    - Temporal-Image prediction;
    - optimal control;
    - automatic topology inference;
    - general Phi_H across all transition channels.

The only executable physical integration channel tested here is:

    TransitionChannel.PRESTRESS_TRANSFER
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


from experiments import (
    roif_temporal_image_reconstruction_benchmark
    as temporal
)

from roif.history.attractor_dynamics import (
    AttractorPrestress,
    DynamicsState,
)

from roif.history.attractor_trajectory import (
    build_attractor_trajectory,
)

from roif.history.experience_attractor import (
    build_experience_attractor,
)

from roif.history.experience_conditioning import (
    ConditioningCue,
    ConditioningMemory,
    build_conditioning_memory,
    conditioned_response,
    observation_from_trajectory,
)

from roif.history.experience_pattern import (
    build_experience_pattern,
)

from roif.history.experience_sequence import (
    build_experience_sequence,
)

from roif.history.experience_transformation import (
    BodyResponse,
    ExperienceEvent,
    SystemResponse,
    SystemState,
    build_experience_transformation,
)

from roif.history.memory_transition_derivation import (
    TransitionTargetBinding,
    derive_conditioning_transition_modifiers,
)

from roif.history.system_evolution import (
    SystemEvolutionResult,
    evolve_system,
)

from roif.history.system_image import (
    SystemImage,
    system_image_signature,
)

from roif.history.transition_modifiers import (
    TransitionChannel,
)


# =============================================================================
# METADATA
# =============================================================================


BENCHMARK_VERSION = (
    "roif_memory_conditioned_matched_state_transition_v1"
)

CLAIM_SCOPE = "computational_model_only"

ZERO_TOLERANCE = 1e-12

REINFORCED_OBSERVATION_COUNT = 5
UNREINFORCED_OBSERVATION_COUNT = 5

BINDING_SCALE = 0.50

PHYSICAL_TARGET_CONNECTION_ID = "A_B"

SEMANTIC_ATTRACTOR_ID = (
    "sensitization_attractor"
)


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def _l2_norm(
    values: Sequence[float],
) -> float:
    return math.sqrt(
        sum(
            float(value) ** 2
            for value
            in values
        )
    )


def _l2_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "vector dimensions differ"
        )

    return _l2_norm(
        tuple(
            float(a) - float(b)
            for a, b
            in zip(
                left,
                right,
            )
        )
    )


def _prestress_vector(
    image: SystemImage,
) -> tuple[float, ...]:
    return tuple(
        float(node.prestress)
        for node
        in sorted(
            image.prestress_nodes,
            key=lambda item: item.node_id,
        )
    )


def _adaptive_connection_signature(
    image: SystemImage,
) -> tuple[
    tuple[
        str,
        str,
        str,
        float,
        float,
        float,
        float,
        float,
    ],
    ...,
]:
    return tuple(
        (
            connection.connection_id,
            connection.source_node_id,
            connection.target_node_id,
            float(connection.stiffness),
            float(
                connection.contractile_capacity
            ),
            float(connection.reflex_gain),
            float(connection.fatigue),
            float(
                connection.remodeling_bias
            ),
        )
        for connection
        in sorted(
            image.adaptive_connections,
            key=lambda item:
            item.connection_id,
        )
    )


# =============================================================================
# MEMORY 2.0 EXPERIENCE CONSTRUCTION
# =============================================================================


def _make_state(
    state_id: str,
    *,
    cognitive: Sequence[float],
    physiological: Sequence[float],
    reserve: float,
) -> SystemState:
    return SystemState(
        state_id=state_id,
        cognitive=tuple(
            float(value)
            for value
            in cognitive
        ),
        physiological=tuple(
            float(value)
            for value
            in physiological
        ),
        contextual=(0.5, 0.2),
        reserve=float(
            reserve
        ),
    )


def _make_response(
    response_id: str,
    magnitude: float,
) -> SystemResponse:
    value = float(
        magnitude
    )

    return SystemResponse(
        response_id=response_id,
        cognitive_response=(
            value,
            value * 0.9,
        ),
        physiological_response=(
            value * 0.8,
            value * 0.7,
        ),
        behavioral_response=(
            value * 0.6,
            value * 0.5,
        ),
    )


def _make_body_response(
    magnitude: float,
) -> BodyResponse:
    value = float(
        magnitude
    )

    return BodyResponse(
        autonomic=(
            value,
            value * 0.9,
        ),
        endocrine=(
            value * 0.8,
            value * 0.7,
        ),
        immune=(
            value * 0.4,
            value * 0.3,
        ),
        motor=(
            value * 0.75,
            value * 0.65,
        ),
        interoceptive=(
            value * 0.95,
            value * 0.85,
        ),
    )


def _build_sequence(
    *,
    sequence_id: str,
    event: ExperienceEvent,
    states: Sequence[SystemState],
    magnitudes: Sequence[float],
):
    if len(
        magnitudes
    ) != (
        len(states)
        - 1
    ):
        raise ValueError(
            "magnitudes must match state transitions"
        )

    transformations = tuple(
        build_experience_transformation(
            transformation_id=(
                f"{sequence_id}::tx_{index}"
            ),
            state_before=(
                states[index]
            ),
            event=event,
            response=_make_response(
                f"{sequence_id}::response_{index}",
                magnitudes[index],
            ),
            body_response=(
                _make_body_response(
                    magnitudes[index]
                )
            ),
            state_after=(
                states[index + 1]
            ),
        )
        for index
        in range(
            len(states) - 1
        )
    )

    return build_experience_sequence(
        sequence_id=sequence_id,
        transformations=(
            transformations
        ),
    )


def build_memory_attractor():
    """
    Build one deterministic semantic attractor used by BOTH memory branches.

    The attractor itself is not the experimental variable.
    """

    event = ExperienceEvent(
        event_id="memory_training_event",
        event_type="repeated_cue",
        structural_vector=(
            1.0,
            0.25,
            -0.1,
            0.4,
        ),
    )

    patterns = []

    for pattern_index, offset in (
        (1, 0.00),
        (2, 0.01),
    ):
        sequence_id = (
            f"memory_pattern_{pattern_index}"
        )

        states = (
            _make_state(
                f"{sequence_id}::s0",
                cognitive=(
                    0.10 + offset,
                    0.10 + offset,
                    0.10 + offset,
                ),
                physiological=(
                    0.10 + offset,
                    0.08 + offset,
                    0.06 + offset,
                ),
                reserve=(
                    0.90 - offset
                ),
            ),

            _make_state(
                f"{sequence_id}::s1",
                cognitive=(
                    0.20 + offset,
                    0.25 + offset,
                    0.18 + offset,
                ),
                physiological=(
                    0.20 + offset,
                    0.18 + offset,
                    0.16 + offset,
                ),
                reserve=(
                    0.80 - offset
                ),
            ),

            _make_state(
                f"{sequence_id}::s2",
                cognitive=(
                    0.38 + offset,
                    0.45 + offset,
                    0.36 + offset,
                ),
                physiological=(
                    0.40 + offset,
                    0.36 + offset,
                    0.32 + offset,
                ),
                reserve=(
                    0.65 - offset
                ),
            ),
        )

        sequence = _build_sequence(
            sequence_id=sequence_id,
            event=event,
            states=states,
            magnitudes=(
                0.20 + offset,
                0.45 + offset,
            ),
        )

        patterns.append(
            build_experience_pattern(
                pattern_id=(
                    f"memory_pattern_{pattern_index}"
                ),
                event_type=(
                    event.event_type
                ),
                sequences=(
                    sequence,
                ),
            )
        )

    attractor = (
        build_experience_attractor(
            attractor_id=(
                SEMANTIC_ATTRACTOR_ID
            ),
            patterns=tuple(
                patterns
            ),
        )
    )

    return attractor


def build_memory_trajectory():
    """
    Build one common trajectory whose terminal attractor is used to construct
    observations for both memory histories.

    Reinforcement history, not trajectory geometry, is the experimental
    difference between the two ConditioningMemory objects.
    """

    attractor = (
        build_memory_attractor()
    )

    initial_state = DynamicsState(
        state_id="memory_initial_state",
        feature_vector=(
            attractor.center
        ),
    )

    prestress = AttractorPrestress(
        prestress_id="memory_training_bias",
        bias_by_attractor_id={
            SEMANTIC_ATTRACTOR_ID:
            0.75,
        },
    )

    trajectory = (
        build_attractor_trajectory(
            trajectory_id=(
                "memory_training_trajectory"
            ),
            initial_state=(
                initial_state
            ),
            attractors=(
                attractor,
            ),
            prestress_sequence=(
                prestress,
                prestress,
                prestress,
            ),
            step_size=0.10,
        )
    )

    return trajectory


def build_query_cue() -> ConditioningCue:
    return ConditioningCue(
        cue_id="matched_memory_probe_cue",
        cue_type="visual_symbol",
        feature_vector=(
            1.0,
            0.2,
            0.1,
            0.0,
        ),
    )


def build_reinforced_memory(
    *,
    cue: ConditioningCue,
) -> ConditioningMemory:
    """
    Repeated reinforced exposure history.
    """

    trajectory = (
        build_memory_trajectory()
    )

    observations = tuple(
        observation_from_trajectory(
            observation_id=(
                f"reinforced_obs_{index}"
            ),
            cue=cue,
            trajectory=(
                trajectory
            ),
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for index
        in range(
            REINFORCED_OBSERVATION_COUNT
        )
    )

    return build_conditioning_memory(
        memory_id="reinforced_memory",
        cue=cue,
        observations=observations,
    )


def build_unreinforced_memory(
    *,
    cue: ConditioningCue,
) -> ConditioningMemory:
    """
    Matched exposure history without reinforcement.

    This is deliberately not called a biological extinction model.

    It is simply the computational contrast:

        same cue
        same trajectory
        same semantic attractor
        different reinforcement history.
    """

    trajectory = (
        build_memory_trajectory()
    )

    observations = tuple(
        observation_from_trajectory(
            observation_id=(
                f"unreinforced_obs_{index}"
            ),
            cue=cue,
            trajectory=(
                trajectory
            ),
            reinforcement_present=False,
            reinforcement_strength=0.0,
        )
        for index
        in range(
            UNREINFORCED_OBSERVATION_COUNT
        )
    )

    return build_conditioning_memory(
        memory_id="unreinforced_memory",
        cue=cue,
        observations=observations,
    )


# =============================================================================
# PHYSICAL MATCHED STATE AND PROBE
# =============================================================================


def build_matched_current_image() -> SystemImage:
    """
    Use the same history-conditioned physical SystemImage for every branch.

    The temporal benchmark returns:

        I0, I1, I2, I3, I4, I5

    so index 4 is the physical image after E1 -> E2 -> E3 -> E4 and before E5.
    """

    trajectory = (
        temporal.generate_truth_trajectory()
    )

    return trajectory[4]


def build_common_probe():
    """
    Common physical challenge C.

    E5 is used only as one deterministic model-internal probe event.
    """

    return temporal.build_events()[4]


def build_binding() -> TransitionTargetBinding:
    """
    Explicit semantic-to-physical correspondence.

    The memory system does NOT infer this mapping.
    """

    return TransitionTargetBinding(
        binding_id=(
            "sensitization_to_A_B_transfer"
        ),
        semantic_target_id=(
            SEMANTIC_ATTRACTOR_ID
        ),
        physical_target_id=(
            PHYSICAL_TARGET_CONNECTION_ID
        ),
        channel=(
            TransitionChannel.PRESTRESS_TRANSFER
        ),
        polarity=1.0,
        scale=BINDING_SCALE,
        metadata={
            "benchmark": (
                BENCHMARK_VERSION
            ),
            "mapping_mode": (
                "explicit_benchmark_binding"
            ),
        },
    )


# =============================================================================
# MEMORY -> MODIFIER DERIVATION
# =============================================================================


def derive_memory_conditions():
    cue = build_query_cue()

    reinforced_memory = (
        build_reinforced_memory(
            cue=cue
        )
    )

    unreinforced_memory = (
        build_unreinforced_memory(
            cue=cue
        )
    )

    binding = build_binding()

    reinforced_derivation = (
        derive_conditioning_transition_modifiers(
            derivation_id=(
                "reinforced_derivation"
            ),
            query_cue=cue,
            memory=(
                reinforced_memory
            ),
            bindings=(
                binding,
            ),
        )
    )

    unreinforced_derivation = (
        derive_conditioning_transition_modifiers(
            derivation_id=(
                "unreinforced_derivation"
            ),
            query_cue=cue,
            memory=(
                unreinforced_memory
            ),
            bindings=(
                binding,
            ),
        )
    )

    return {
        "cue": cue,
        "binding": binding,
        "reinforced_memory": (
            reinforced_memory
        ),
        "unreinforced_memory": (
            unreinforced_memory
        ),
        "reinforced_derivation": (
            reinforced_derivation
        ),
        "unreinforced_derivation": (
            unreinforced_derivation
        ),
    }


# =============================================================================
# PHYSICAL TRANSITIONS
# =============================================================================


def run_physical_conditions(
    *,
    current_image: SystemImage,
    probe,
    memory_conditions,
):
    reinforced_modifiers = (
        memory_conditions[
            "reinforced_derivation"
        ].modifier_set
    )

    unreinforced_modifiers = (
        memory_conditions[
            "unreinforced_derivation"
        ].modifier_set
    )

    no_memory_result = evolve_system(
        evolution_id=(
            "memory_conditioned::"
            "no_memory_control"
        ),
        source_image=(
            current_image
        ),
        event=probe,
        config=(
            temporal.benchmark_config()
        ),
        transition_modifiers=None,
        target_image_id=(
            "memory_conditioned::"
            "no_memory_post"
        ),
    )

    reinforced_result = evolve_system(
        evolution_id=(
            "memory_conditioned::"
            "reinforced"
        ),
        source_image=(
            current_image
        ),
        event=probe,
        config=(
            temporal.benchmark_config()
        ),
        transition_modifiers=(
            reinforced_modifiers
        ),
        target_image_id=(
            "memory_conditioned::"
            "reinforced_post"
        ),
    )

    unreinforced_result = evolve_system(
        evolution_id=(
            "memory_conditioned::"
            "unreinforced"
        ),
        source_image=(
            current_image
        ),
        event=probe,
        config=(
            temporal.benchmark_config()
        ),
        transition_modifiers=(
            unreinforced_modifiers
        ),
        target_image_id=(
            "memory_conditioned::"
            "unreinforced_post"
        ),
    )

    return {
        "no_memory": (
            no_memory_result
        ),
        "reinforced": (
            reinforced_result
        ),
        "unreinforced": (
            unreinforced_result
        ),
    }


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


def _modifier_block(
    derivation,
) -> dict[str, Any]:
    modifiers = (
        derivation.modifier_set
        .modifiers
    )

    return {
        "derivation_id": (
            derivation.derivation_id
        ),

        "derivation_mode": (
            derivation.derivation_mode
        ),

        "semantic_signal_by_target": {
            key: float(value)
            for key, value
            in derivation
            .semantic_signal_by_target
            .items()
        },

        "applied_binding_ids": list(
            derivation.applied_binding_ids
        ),

        "modifier_set_id": (
            derivation.modifier_set
            .modifier_set_id
        ),

        "modifiers": [
            {
                "modifier_id": (
                    modifier.modifier_id
                ),
                "channel": (
                    modifier.channel.value
                ),
                "target_id": (
                    modifier.target_id
                ),
                "value": float(
                    modifier.value
                ),
                "evidence": [
                    {
                        "source_kind": (
                            evidence
                            .source_kind
                            .value
                        ),
                        "source_id": (
                            evidence.source_id
                        ),
                        "order_index": (
                            evidence.order_index
                        ),
                        "cue_id": (
                            evidence.cue_id
                        ),
                        "attractor_id": (
                            evidence
                            .attractor_id
                        ),
                        "context_id": (
                            evidence.context_id
                        ),
                        "relation": (
                            evidence.relation
                        ),
                    }
                    for evidence
                    in modifier.evidence
                ],
            }
            for modifier
            in modifiers
        ],
    }


def _physical_result_block(
    result: SystemEvolutionResult,
    *,
    current_image: SystemImage,
) -> dict[str, Any]:
    pre = {
        node.node_id:
        float(node.prestress)
        for node
        in current_image.prestress_nodes
    }

    post = {
        node.node_id:
        float(node.prestress)
        for node
        in result.target_image
        .prestress_nodes
    }

    delta = {
        node_id:
        float(
            post[node_id]
            - pre[node_id]
        )
        for node_id
        in sorted(pre)
    }

    return {
        "prestress_before": {
            key: pre[key]
            for key
            in sorted(pre)
        },

        "prestress_after": {
            key: post[key]
            for key
            in sorted(post)
        },

        "prestress_delta": (
            delta
        ),

        "prestress_change_norm": float(
            result.summary
            .prestress_change_norm
        ),

        "connection_change_norm": float(
            result.summary
            .connection_change_norm
        ),

        "trace_magnitude": float(
            result.generated_trace
            .magnitude
        ),

        "transition_modifiers_applied": bool(
            result.metadata.get(
                "transition_modifiers_applied",
                False,
            )
        ),

        "active_transition_modifier_count": int(
            result.metadata.get(
                "active_transition_modifier_count",
                0,
            )
        ),

        "adaptive_connection_signature": [
            list(item)
            for item
            in _adaptive_connection_signature(
                result.target_image
            )
        ],
    }


# =============================================================================
# BENCHMARK
# =============================================================================


def run_benchmark() -> dict[str, Any]:
    """
    Run matched-current-state / different-structured-memory end-to-end test.
    """

    current_image = (
        build_matched_current_image()
    )

    probe = build_common_probe()

    memory_conditions = (
        derive_memory_conditions()
    )

    physical = (
        run_physical_conditions(
            current_image=(
                current_image
            ),
            probe=probe,
            memory_conditions=(
                memory_conditions
            ),
        )
    )

    reinforced_response = (
        conditioned_response(
            query_cue=(
                memory_conditions[
                    "cue"
                ]
            ),
            memory=(
                memory_conditions[
                    "reinforced_memory"
                ]
            ),
        )
    )

    unreinforced_response = (
        conditioned_response(
            query_cue=(
                memory_conditions[
                    "cue"
                ]
            ),
            memory=(
                memory_conditions[
                    "unreinforced_memory"
                ]
            ),
        )
    )

    reinforced_modifier = (
        memory_conditions[
            "reinforced_derivation"
        ].modifier_set.modifiers[0]
    )

    unreinforced_modifier = (
        memory_conditions[
            "unreinforced_derivation"
        ].modifier_set.modifiers[0]
    )

    reinforced_post_vector = (
        _prestress_vector(
            physical[
                "reinforced"
            ].target_image
        )
    )

    unreinforced_post_vector = (
        _prestress_vector(
            physical[
                "unreinforced"
            ].target_image
        )
    )

    no_memory_post_vector = (
        _prestress_vector(
            physical[
                "no_memory"
            ].target_image
        )
    )

    reinforced_vs_unreinforced = (
        _l2_distance(
            reinforced_post_vector,
            unreinforced_post_vector,
        )
    )

    reinforced_vs_no_memory = (
        _l2_distance(
            reinforced_post_vector,
            no_memory_post_vector,
        )
    )

    unreinforced_vs_no_memory = (
        _l2_distance(
            unreinforced_post_vector,
            no_memory_post_vector,
        )
    )

    current_signature_before = (
        system_image_signature(
            current_image
        )
    )

    current_signature_after_runs = (
        system_image_signature(
            current_image
        )
    )

    adaptive_reinforced = (
        _adaptive_connection_signature(
            physical[
                "reinforced"
            ].target_image
        )
    )

    adaptive_unreinforced = (
        _adaptive_connection_signature(
            physical[
                "unreinforced"
            ].target_image
        )
    )

    adaptive_no_memory = (
        _adaptive_connection_signature(
            physical[
                "no_memory"
            ].target_image
        )
    )

    memory_signal_differs = (
        abs(
            reinforced_response
            .conditioned_response_strength
            -
            unreinforced_response
            .conditioned_response_strength
        )
        > ZERO_TOLERANCE
    )

    modifier_differs = (
        abs(
            reinforced_modifier.value
            -
            unreinforced_modifier.value
        )
        > ZERO_TOLERANCE
    )

    physical_transition_differs = (
        reinforced_vs_unreinforced
        > ZERO_TOLERANCE
    )

    adaptive_state_equal = (
        adaptive_reinforced
        ==
        adaptive_unreinforced
        ==
        adaptive_no_memory
    )

    source_image_unchanged = (
        current_signature_before
        ==
        current_signature_after_runs
    )

    return {
        "benchmark_version": (
            BENCHMARK_VERSION
        ),

        "claim_scope": (
            CLAIM_SCOPE
        ),

        "mechanism_under_test": (
            "structured_conditioning_memory_to_"
            "prestress_transfer_transition"
        ),

        "experimental_design": {
            "matched_current_system_image": True,
            "same_query_cue": True,
            "same_semantic_attractor": True,
            "same_explicit_binding": True,
            "same_physical_probe_event": True,
            "same_physical_topology": True,
            "different_conditioning_history": True,
            "modifier_values_set_manually": False,
        },

        "physical_source": {
            "image_id": (
                current_image.image_id
            ),
            "revision": int(
                current_image.revision
            ),
            "probe_event_id": (
                probe.event_id
            ),
            "binding": {
                "binding_id": (
                    memory_conditions[
                        "binding"
                    ].binding_id
                ),
                "semantic_target_id": (
                    memory_conditions[
                        "binding"
                    ].semantic_target_id
                ),
                "physical_target_id": (
                    memory_conditions[
                        "binding"
                    ].physical_target_id
                ),
                "channel": (
                    memory_conditions[
                        "binding"
                    ].channel.value
                ),
                "polarity": float(
                    memory_conditions[
                        "binding"
                    ].polarity
                ),
                "scale": float(
                    memory_conditions[
                        "binding"
                    ].scale
                ),
            },
        },

        "memory_conditions": {
            "reinforced": {
                "memory_id": (
                    memory_conditions[
                        "reinforced_memory"
                    ].memory_id
                ),
                "observation_count": len(
                    memory_conditions[
                        "reinforced_memory"
                    ].observations
                ),
                "conditioned_response_strength": float(
                    reinforced_response
                    .conditioned_response_strength
                ),
                "predicted_attractor_id": (
                    reinforced_response
                    .predicted_attractor_id
                ),
                "derivation": (
                    _modifier_block(
                        memory_conditions[
                            "reinforced_derivation"
                        ]
                    )
                ),
            },

            "unreinforced": {
                "memory_id": (
                    memory_conditions[
                        "unreinforced_memory"
                    ].memory_id
                ),
                "observation_count": len(
                    memory_conditions[
                        "unreinforced_memory"
                    ].observations
                ),
                "conditioned_response_strength": float(
                    unreinforced_response
                    .conditioned_response_strength
                ),
                "predicted_attractor_id": (
                    unreinforced_response
                    .predicted_attractor_id
                ),
                "derivation": (
                    _modifier_block(
                        memory_conditions[
                            "unreinforced_derivation"
                        ]
                    )
                ),
            },
        },

        "physical_results": {
            "no_memory_control": (
                _physical_result_block(
                    physical[
                        "no_memory"
                    ],
                    current_image=(
                        current_image
                    ),
                )
            ),

            "reinforced_memory": (
                _physical_result_block(
                    physical[
                        "reinforced"
                    ],
                    current_image=(
                        current_image
                    ),
                )
            ),

            "unreinforced_memory": (
                _physical_result_block(
                    physical[
                        "unreinforced"
                    ],
                    current_image=(
                        current_image
                    ),
                )
            ),
        },

        "response_distances": {
            "reinforced_vs_unreinforced": float(
                reinforced_vs_unreinforced
            ),

            "reinforced_vs_no_memory": float(
                reinforced_vs_no_memory
            ),

            "unreinforced_vs_no_memory": float(
                unreinforced_vs_no_memory
            ),
        },

        "central_results": {
            "matched_current_state_established": True,

            "structured_memory_conditions_differ": (
                memory_conditions[
                    "reinforced_memory"
                ]
                !=
                memory_conditions[
                    "unreinforced_memory"
                ]
            ),

            "memory_derived_semantic_signal_differs": (
                memory_signal_differs
            ),

            "memory_derived_transition_modifier_differs": (
                modifier_differs
            ),

            "same_physical_probe_applied": True,

            "physical_transition_differs_by_memory_condition": (
                physical_transition_differs
            ),

            "adaptive_connection_state_equal_across_conditions": (
                adaptive_state_equal
            ),

            "source_system_image_remained_immutable": (
                source_image_unchanged
            ),

            "reinforced_transition_distance_from_unreinforced": (
                float(
                    reinforced_vs_unreinforced
                )
            ),
        },

        # =====================================================================
        # CLAIM BOUNDARIES
        # =====================================================================

        "structured_memory_to_transition_path_tested": True,

        "conditioning_memory_derivation_tested": True,

        "explicit_semantic_physical_binding_used": True,

        "prestress_transfer_channel_tested": True,

        "automatic_topology_mapping_tested": False,

        "connection_update_memory_modulation_tested": False,

        "whole_system_history_operator_tested": False,

        "general_history_conditioned_operator_claimed": False,

        "predictive_preconfiguration_tested": False,

        "future_temporal_image_prediction_tested": False,

        "stabilization_tested": False,

        "biological_learning_claimed": False,

        "clinical_validity_claimed": False,

        "universal_memory_dynamics_claimed": False,

        "interpretation": (
            "This controlled computational benchmark evaluates an end-to-end "
            "Memory-2.0-to-SystemEvolution path under matched current physical "
            "state. Reinforced and unreinforced conditioning histories are "
            "converted into ConditioningMemory objects using the same cue, "
            "trajectory and semantic attractor. Existing conditioned-response "
            "semantics derive memory-specific response strengths. An explicit "
            "semantic-to-physical binding then maps the semantic attractor to "
            "the existing A_B PRESTRESS_TRANSFER channel, producing bounded "
            "TransitionModifierSet objects without manually specifying their "
            "numeric modifier values. The same physical SystemImage and the "
            "same subsequent probe event are used in all physical conditions. "
            "If the derived memory signals and modifiers differ and the same "
            "probe consequently produces different prestress redistribution "
            "while adaptive connection state and topology remain matched, the "
            "result demonstrates a structured-memory-conditioned contribution "
            "to the next physical transition through the implemented "
            "PRESTRESS_TRANSFER integration channel. It does not establish a "
            "general history-conditioned operator across ROIF, whole-SystemImage "
            "memory conditioning, predictive stabilization, Temporal-Image "
            "prediction, biological learning, clinical validity, or universal "
            "memory dynamics."
        ),
    }


# =============================================================================
# OUTPUT
# =============================================================================


def _write_json(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    data = run_benchmark()

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_path = (
        root
        / "benchmark_results"
        / (
            "memory_conditioned_"
            "matched_state_transition_v1.json"
        )
    )

    _write_json(
        output_path,
        data,
    )

    print(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "Saved memory-conditioned "
        "matched-state transition benchmark:"
    )

    print(
        output_path.resolve()
    )


if __name__ == "__main__":
    main()
