from __future__ import annotations

import inspect

import pytest

from experiments import (
    roif_q8_history_conditioned_redistribution_matched_state_benchmark
    as q8_v2
)
from experiments import (
    roif_matched_state_history_operator_identifiability_benchmark
    as matched_operator
)

import roif.history.system_evolution as system_evolution_module

from roif.history.system_evolution import (
    evolve_system,
)

from roif.history.experience_transformation import (
    BodyResponse,
    ExperienceEvent,
    SystemResponse,
    SystemState,
    build_experience_transformation,
)

from roif.history.experience_sequence import (
    build_experience_sequence,
)

from roif.history.experience_pattern import (
    build_experience_pattern,
)

from roif.history.experience_attractor import (
    build_experience_attractor,
)

from roif.history.attractor_dynamics import (
    AttractorPrestress,
    DynamicsState,
)

from roif.history.attractor_trajectory import (
    build_attractor_trajectory,
)

from roif.history.experience_conditioning import (
    ConditioningCue,
    build_conditioning_memory,
    observation_from_trajectory,
)

from roif.history.conditioned_prestress import (
    ConditionedPrestressConfig,
)

from roif.history.conditioned_dynamics import (
    conditioned_target_capture_gain,
    conditioned_target_weight_gain,
    run_conditioned_dynamics,
)

from roif.history.associative_context import (
    AssociativeCueInput,
    build_associative_context,
)

from roif.history.contextual_dynamics import (
    context_target_capture_gain,
    context_target_weight_gain,
    run_contextual_dynamics,
)


# =============================================================================
# HELPERS — MEMORY 2.0 STACK
# =============================================================================


def make_state(
    state_id: str,
    *,
    cognitive,
    physiological,
    reserve: float,
):
    return SystemState(
        state_id=state_id,
        cognitive=tuple(cognitive),
        physiological=tuple(
            physiological
        ),
        contextual=(0.5, 0.2),
        reserve=reserve,
    )


def make_response(
    response_id: str,
    magnitude: float,
) -> SystemResponse:
    return SystemResponse(
        response_id=response_id,
        cognitive_response=(
            magnitude,
            magnitude * 0.9,
        ),
        physiological_response=(
            magnitude * 0.8,
            magnitude * 0.7,
        ),
        behavioral_response=(
            magnitude * 0.6,
            magnitude * 0.5,
        ),
    )


def make_body(
    magnitude: float,
) -> BodyResponse:
    return BodyResponse(
        autonomic=(
            magnitude,
            magnitude * 0.9,
        ),
        endocrine=(
            magnitude * 0.8,
            magnitude * 0.7,
        ),
        immune=(
            magnitude * 0.4,
            magnitude * 0.3,
        ),
        motor=(
            magnitude * 0.75,
            magnitude * 0.65,
        ),
        interoceptive=(
            magnitude * 0.95,
            magnitude * 0.85,
        ),
    )


def build_sequence_from_states(
    sequence_id: str,
    event: ExperienceEvent,
    states,
    magnitudes,
):
    transformations = []

    for index in range(
        len(states) - 1
    ):
        transformations.append(
            build_experience_transformation(
                transformation_id=(
                    f"{sequence_id}_tx{index}"
                ),
                state_before=(
                    states[index]
                ),
                event=event,
                response=make_response(
                    f"{sequence_id}_r{index}",
                    magnitudes[index],
                ),
                body_response=make_body(
                    magnitudes[index]
                ),
                state_after=(
                    states[index + 1]
                ),
            )
        )

    return build_experience_sequence(
        sequence_id=sequence_id,
        transformations=tuple(
            transformations
        ),
    )


def build_sensitizing_pattern(
    pattern_id: str,
    sequence_id: str,
    event: ExperienceEvent,
    offset: float,
):
    states = (
        make_state(
            f"{sequence_id}_s0",
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

        make_state(
            f"{sequence_id}_s1",
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

        make_state(
            f"{sequence_id}_s2",
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

    sequence = (
        build_sequence_from_states(
            sequence_id,
            event,
            states,
            magnitudes=(
                0.20 + offset,
                0.45 + offset,
            ),
        )
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(sequence,),
    )


def build_adaptation_pattern(
    pattern_id: str,
    sequence_id: str,
    event: ExperienceEvent,
):
    states = (
        make_state(
            f"{sequence_id}_s0",
            cognitive=(
                0.50,
                0.50,
                0.50,
            ),
            physiological=(
                0.50,
                0.50,
                0.50,
            ),
            reserve=0.40,
        ),

        make_state(
            f"{sequence_id}_s1",
            cognitive=(
                0.30,
                0.30,
                0.30,
            ),
            physiological=(
                0.28,
                0.28,
                0.28,
            ),
            reserve=0.65,
        ),

        make_state(
            f"{sequence_id}_s2",
            cognitive=(
                0.18,
                0.18,
                0.18,
            ),
            physiological=(
                0.16,
                0.16,
                0.16,
            ),
            reserve=0.85,
        ),
    )

    sequence = (
        build_sequence_from_states(
            sequence_id,
            event,
            states,
            magnitudes=(
                0.40,
                0.30,
            ),
        )
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(sequence,),
    )


def build_attractors(
    event: ExperienceEvent,
):
    sensitizing_1 = (
        build_sensitizing_pattern(
            "sens_1",
            "sens_seq_1",
            event,
            0.00,
        )
    )

    sensitizing_2 = (
        build_sensitizing_pattern(
            "sens_2",
            "sens_seq_2",
            event,
            0.01,
        )
    )

    adaptation_1 = (
        build_adaptation_pattern(
            "adapt_1",
            "adapt_seq_1",
            event,
        )
    )

    adaptation_2 = (
        build_adaptation_pattern(
            "adapt_2",
            "adapt_seq_2",
            event,
        )
    )

    return (
        build_experience_attractor(
            attractor_id=(
                "sensitization_attractor"
            ),
            patterns=(
                sensitizing_1,
                sensitizing_2,
            ),
        ),

        build_experience_attractor(
            attractor_id=(
                "adaptation_attractor"
            ),
            patterns=(
                adaptation_1,
                adaptation_2,
            ),
        ),
    )


def midpoint_state(
    attractors,
) -> DynamicsState:
    vector = tuple(
        (
            left
            + right
        )
        / 2.0
        for left, right
        in zip(
            attractors[
                0
            ].center,
            attractors[
                1
            ].center,
        )
    )

    return DynamicsState(
        state_id="midpoint",
        feature_vector=vector,
    )


# =============================================================================
# FIXTURES — PHYSICAL HISTORY
# =============================================================================


@pytest.fixture(scope="module")
def q8_result():
    return q8_v2.run_benchmark()


@pytest.fixture(scope="module")
def matched_result():
    return (
        matched_operator.run_benchmark()
    )


# =============================================================================
# FIXTURES — MEMORY 2.0
# =============================================================================


@pytest.fixture(scope="module")
def experience_event():
    return ExperienceEvent(
        event_id="event_repeat",
        event_type="repeated_cue",
        structural_vector=(
            1.0,
            0.25,
            -0.1,
            0.4,
        ),
    )


@pytest.fixture(scope="module")
def attractors(
    experience_event,
):
    return build_attractors(
        experience_event
    )


@pytest.fixture(scope="module")
def initial_state(
    attractors,
):
    return midpoint_state(
        attractors
    )


@pytest.fixture(scope="module")
def sensitization_bias():
    return AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={
            "sensitization_attractor":
            0.75,
        },
    )


@pytest.fixture(scope="module")
def adaptation_bias():
    return AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor":
            0.75,
        },
    )


@pytest.fixture(scope="module")
def sensitization_trajectory(
    attractors,
    initial_state,
    sensitization_bias,
):
    return build_attractor_trajectory(
        trajectory_id=(
            "sens_trajectory"
        ),
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            sensitization_bias,
            sensitization_bias,
            sensitization_bias,
        ),
        step_size=0.10,
    )


@pytest.fixture(scope="module")
def adaptation_trajectory(
    attractors,
    initial_state,
    adaptation_bias,
):
    return build_attractor_trajectory(
        trajectory_id=(
            "adapt_trajectory"
        ),
        initial_state=initial_state,
        attractors=attractors,
        prestress_sequence=(
            adaptation_bias,
            adaptation_bias,
            adaptation_bias,
        ),
        step_size=0.10,
    )


@pytest.fixture(scope="module")
def cue_door():
    return ConditioningCue(
        cue_id="cue_door",
        cue_type="visual_symbol",
        feature_vector=(
            1.0,
            0.2,
            0.1,
            0.0,
        ),
    )


@pytest.fixture(scope="module")
def cue_room():
    return ConditioningCue(
        cue_id="cue_room",
        cue_type="visual_symbol",
        feature_vector=(
            0.9,
            0.3,
            0.15,
            0.05,
        ),
    )


@pytest.fixture(scope="module")
def cue_safe():
    return ConditioningCue(
        cue_id="cue_safe",
        cue_type="visual_symbol",
        feature_vector=(
            0.1,
            0.9,
            0.8,
            0.7,
        ),
    )


@pytest.fixture(scope="module")
def distant_cue():
    return ConditioningCue(
        cue_id="cue_far",
        cue_type="visual_symbol",
        feature_vector=(
            0.0,
            1.0,
            1.0,
            1.0,
        ),
    )


@pytest.fixture(scope="module")
def sensitization_memory(
    cue_door,
    sensitization_trajectory,
):
    observations = tuple(
        observation_from_trajectory(
            observation_id=(
                f"sens_obs_{index}"
            ),
            cue=cue_door,
            trajectory=(
                sensitization_trajectory
            ),
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for index in range(5)
    )

    return build_conditioning_memory(
        memory_id="sens_memory",
        cue=cue_door,
        observations=observations,
    )


@pytest.fixture(scope="module")
def adaptation_memory(
    cue_safe,
    adaptation_trajectory,
):
    observations = tuple(
        observation_from_trajectory(
            observation_id=(
                f"adapt_obs_{index}"
            ),
            cue=cue_safe,
            trajectory=(
                adaptation_trajectory
            ),
            reinforcement_present=True,
            reinforcement_strength=1.0,
        )
        for index in range(5)
    )

    return build_conditioning_memory(
        memory_id="adapt_memory",
        cue=cue_safe,
        observations=observations,
    )


@pytest.fixture(scope="module")
def conditioned_result(
    initial_state,
    attractors,
    cue_door,
    sensitization_memory,
):
    return run_conditioned_dynamics(
        result_id=(
            "architecture_conditioned"
        ),
        state=initial_state,
        attractors=attractors,
        query_cue=cue_door,
        conditioning_memory=(
            sensitization_memory
        ),
        step_size=0.25,
    )


@pytest.fixture(scope="module")
def distant_conditioned_result(
    initial_state,
    attractors,
    distant_cue,
    sensitization_memory,
):
    return run_conditioned_dynamics(
        result_id=(
            "architecture_distant"
        ),
        state=initial_state,
        attractors=attractors,
        query_cue=distant_cue,
        conditioning_memory=(
            sensitization_memory
        ),
        step_size=0.25,
    )


@pytest.fixture(scope="module")
def suppressed_conditioned_result(
    initial_state,
    attractors,
    cue_door,
    sensitization_memory,
):
    return run_conditioned_dynamics(
        result_id=(
            "architecture_suppressed"
        ),
        state=initial_state,
        attractors=attractors,
        query_cue=cue_door,
        conditioning_memory=(
            sensitization_memory
        ),
        prestress_config=(
            ConditionedPrestressConfig(
                minimum_response_strength=(
                    1.0
                ),
            )
        ),
        step_size=0.25,
    )


@pytest.fixture(scope="module")
def sensitization_context(
    cue_door,
    cue_room,
    sensitization_memory,
):
    return build_associative_context(
        context_id="sens_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_door,
                memory=(
                    sensitization_memory
                ),
                salience=1.0,
            ),
            AssociativeCueInput(
                cue=cue_room,
                memory=(
                    sensitization_memory
                ),
                salience=0.8,
            ),
        ),
    )


@pytest.fixture(scope="module")
def adaptation_context(
    cue_safe,
    adaptation_memory,
):
    return build_associative_context(
        context_id="adapt_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_safe,
                memory=adaptation_memory,
                salience=1.0,
            ),
        ),
    )


@pytest.fixture(scope="module")
def zero_context(
    cue_door,
    sensitization_memory,
):
    return build_associative_context(
        context_id="zero_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_door,
                memory=(
                    sensitization_memory
                ),
                salience=0.0,
            ),
        ),
    )


@pytest.fixture(scope="module")
def sensitization_context_result(
    initial_state,
    attractors,
    sensitization_context,
):
    return run_contextual_dynamics(
        result_id=(
            "architecture_sens_context"
        ),
        state=initial_state,
        attractors=attractors,
        context=(
            sensitization_context
        ),
        step_size=0.25,
    )


@pytest.fixture(scope="module")
def adaptation_context_result(
    initial_state,
    attractors,
    adaptation_context,
):
    return run_contextual_dynamics(
        result_id=(
            "architecture_adapt_context"
        ),
        state=initial_state,
        attractors=attractors,
        context=(
            adaptation_context
        ),
        step_size=0.25,
    )


@pytest.fixture(scope="module")
def zero_context_result(
    initial_state,
    attractors,
    zero_context,
):
    return run_contextual_dynamics(
        result_id=(
            "architecture_zero_context"
        ),
        state=initial_state,
        attractors=attractors,
        context=zero_context,
        step_size=0.25,
    )


# =============================================================================
# PHYSICAL-HISTORY BRANCH
# =============================================================================


def test_physical_history_changes_adaptive_connection_state(
    q8_result,
):
    assert q8_result[
        "state_control_audit"
    ][
        "baseline_vs_conditioned_connections_differ"
    ] is True


def test_physical_history_changes_response_at_matched_prestress(
    q8_result,
):
    assert q8_result[
        "state_control_audit"
    ][
        "baseline_prestress_matches_connection_only_control"
    ] is True

    assert q8_result[
        "central_results"
    ][
        "connection_state_changes_response_at_matched_prestress"
    ] is True


def test_physical_history_connection_effect_is_nonzero(
    q8_result,
):
    assert q8_result[
        "central_results"
    ][
        "connection_state_effect_norm_at_matched_prestress"
    ] > 0.0


# =============================================================================
# MATCHED CURRENT STATE / DIFFERENT ORDERED HISTORY
# =============================================================================


def test_matched_history_control_really_matches_current_state(
    matched_result,
):
    assert matched_result[
        "central_results"
    ][
        "matched_current_state_established"
    ] is True


def test_matched_history_control_really_preserves_different_history(
    matched_result,
):
    assert matched_result[
        "central_results"
    ][
        "different_ordered_history_established"
    ] is True

    assert matched_result[
        "central_results"
    ][
        "different_canonical_event_order_established"
    ] is True


def test_raw_ordered_history_does_not_change_tested_system_evolution_transition(
    matched_result,
):
    assert matched_result[
        "central_results"
    ][
        "history_specific_transition_effect_detected"
    ] is False

    assert matched_result[
        "central_results"
    ][
        "transition_response_distance"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_matched_histories_remain_distinct_after_identical_probe(
    matched_result,
):
    assert matched_result[
        "central_results"
    ][
        "post_probe_histories_remain_different"
    ] is True

    assert matched_result[
        "central_results"
    ][
        "post_probe_current_state_equal"
    ] is True


# =============================================================================
# CODE-PATH AUDIT — SYSTEM EVOLUTION
# =============================================================================


def test_system_evolution_documented_order_places_trace_after_physical_transition():
    source = inspect.getsource(
        evolve_system
    )

    connection_position = (
        source.index(
            "_evolve_connections"
        )
    )

    redistribution_position = (
        source.index(
            "redistribute_prestress"
        )
    )

    generated_trace_position = (
        source.index(
            "generated_trace ="
        )
    )

    inherited_history_position = (
        source.index(
            "source_image.historical_traces"
        )
    )

    assert (
        connection_position
        < redistribution_position
        < generated_trace_position
        < inherited_history_position
    )


def test_system_evolution_does_not_read_raw_history_before_transition():
    source = inspect.getsource(
        evolve_system
    )

    transition_prefix = source.split(
        "generated_trace =",
        maxsplit=1,
    )[0]

    assert (
        "source_image.historical_traces"
        not in transition_prefix
    )


def test_system_evolution_raw_history_is_used_when_building_next_image():
    source = inspect.getsource(
        evolve_system
    )

    suffix = source.split(
        "generated_trace =",
        maxsplit=1,
    )[1]

    assert (
        "source_image.historical_traces"
        in suffix
    )


def test_system_evolution_module_does_not_import_conditioned_dynamics():
    source = inspect.getsource(
        system_evolution_module
    )

    assert (
        "conditioned_dynamics"
        not in source
    )


def test_system_evolution_module_does_not_import_associative_context():
    source = inspect.getsource(
        system_evolution_module
    )

    assert (
        "associative_context"
        not in source
    )


def test_system_evolution_module_does_not_import_contextual_dynamics():
    source = inspect.getsource(
        system_evolution_module
    )

    assert (
        "contextual_dynamics"
        not in source
    )


# =============================================================================
# MEMORY 2.0 — CONDITIONING MEMORY CHANGES DYNAMICS
# =============================================================================


def test_conditioned_dynamics_uses_same_initial_state_for_control_and_memory_branch(
    conditioned_result,
):
    assert (
        conditioned_result.baseline.state_before
        ==
        conditioned_result.conditioned.state_before
        ==
        conditioned_result.state_before
    )


def test_conditioning_memory_generates_conditioned_prestress(
    conditioned_result,
):
    bias = dict(
        conditioned_result
        .conditioned_prestress
        .prestress
        .bias_by_attractor_id
    )

    assert bias
    assert (
        "sensitization_attractor"
        in bias
    )

    assert (
        bias[
            "sensitization_attractor"
        ]
        > 0.0
    )


def test_conditioning_memory_changes_attractor_competition(
    conditioned_result,
):
    assert conditioned_target_weight_gain(
        conditioned_result
    ) > 0.0

    assert conditioned_target_capture_gain(
        conditioned_result
    ) > 0.0


def test_conditioning_memory_changes_state_after(
    conditioned_result,
):
    assert (
        conditioned_result
        .state_after_distance
        > 0.0
    )

    assert (
        conditioned_result
        .baseline
        .state_after
        .feature_vector
        !=
        conditioned_result
        .conditioned
        .state_after
        .feature_vector
    )


def test_conditioned_effect_depends_on_cue_structure(
    conditioned_result,
    distant_conditioned_result,
):
    assert (
        distant_conditioned_result
        .state_after_distance
        <
        conditioned_result
        .state_after_distance
    )

    assert (
        conditioned_target_weight_gain(
            distant_conditioned_result
        )
        <
        conditioned_target_weight_gain(
            conditioned_result
        )
    )


def test_conditioned_channel_can_be_suppressed_without_changing_base_state(
    suppressed_conditioned_result,
):
    assert (
        suppressed_conditioned_result
        .conditioned_prestress
        .suppressed_by_threshold
        is True
    )

    assert (
        suppressed_conditioned_result
        .state_after_distance
        ==
        pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_suppressing_conditioned_prestress_removes_conditioned_weight_gain(
    suppressed_conditioned_result,
):
    assert conditioned_target_weight_gain(
        suppressed_conditioned_result
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )


# =============================================================================
# MEMORY 2.0 — ASSOCIATIVE CONTEXT
# =============================================================================


def test_associative_context_changes_competition(
    sensitization_context_result,
):
    assert context_target_weight_gain(
        sensitization_context_result
    ) > 0.0

    assert context_target_capture_gain(
        sensitization_context_result
    ) > 0.0


def test_associative_context_changes_state_after(
    sensitization_context_result,
):
    assert (
        sensitization_context_result
        .state_after_distance
        > 0.0
    )

    assert (
        sensitization_context_result
        .baseline
        .state_after
        .feature_vector
        !=
        sensitization_context_result
        .contextual
        .state_after
        .feature_vector
    )


def test_different_structured_contexts_produce_different_state_after(
    sensitization_context_result,
    adaptation_context_result,
):
    assert (
        sensitization_context_result
        .contextual
        .state_after
        .feature_vector
        !=
        adaptation_context_result
        .contextual
        .state_after
        .feature_vector
    )


def test_different_structured_contexts_can_select_different_attractors(
    sensitization_context_result,
    adaptation_context_result,
):
    assert (
        sensitization_context_result
        .contextual
        .dominant_attractor_id
        !=
        adaptation_context_result
        .contextual
        .dominant_attractor_id
    )


def test_sensitization_context_targets_sensitization_attractor(
    sensitization_context_result,
):
    assert (
        sensitization_context_result
        .contextual
        .dominant_attractor_id
        ==
        "sensitization_attractor"
    )


def test_adaptation_context_targets_adaptation_attractor(
    adaptation_context_result,
):
    assert (
        adaptation_context_result
        .contextual
        .dominant_attractor_id
        ==
        "adaptation_attractor"
    )


def test_zero_salience_context_has_no_dynamic_state_effect(
    zero_context_result,
):
    assert (
        zero_context_result
        .state_after_distance
        ==
        pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_zero_salience_context_has_no_target_weight_gain(
    zero_context_result,
):
    assert context_target_weight_gain(
        zero_context_result
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )


# =============================================================================
# STRUCTURED MEMORY ANTI-REDUCTION GUARDS
# =============================================================================


def test_associative_contexts_preserve_attractor_specific_bias_maps(
    sensitization_context,
    adaptation_context,
):
    sensitization_bias = dict(
        sensitization_context
        .prestress
        .bias_by_attractor_id
    )

    adaptation_bias = dict(
        adaptation_context
        .prestress
        .bias_by_attractor_id
    )

    assert sensitization_bias
    assert adaptation_bias

    assert (
        sensitization_bias
        != adaptation_bias
    )

    assert (
        "sensitization_attractor"
        in sensitization_bias
    )

    assert (
        "adaptation_attractor"
        in adaptation_bias
    )


def test_structured_context_effect_is_not_equivalent_to_global_scalar_bias(
    sensitization_context,
    adaptation_context,
):
    sensitization_targets = set(
        sensitization_context
        .prestress
        .bias_by_attractor_id
    )

    adaptation_targets = set(
        adaptation_context
        .prestress
        .bias_by_attractor_id
    )

    assert (
        sensitization_targets
        != adaptation_targets
    )


def test_conditioned_memory_preserves_cue_specific_transfer(
    conditioned_result,
    distant_conditioned_result,
):
    exact_gain = (
        conditioned_target_weight_gain(
            conditioned_result
        )
    )

    distant_gain = (
        conditioned_target_weight_gain(
            distant_conditioned_result
        )
    )

    assert exact_gain > 0.0
    assert distant_gain >= 0.0

    assert (
        exact_gain
        != pytest.approx(
            distant_gain,
            abs=1e-12,
        )
    )


# =============================================================================
# ARCHITECTURAL SEPARATION BETWEEN THE TWO BRANCHES
# =============================================================================


def test_physical_history_branch_and_memory2_branch_are_both_active(
    q8_result,
    conditioned_result,
):
    physical_effect = q8_result[
        "central_results"
    ][
        "connection_state_changes_response_at_matched_prestress"
    ]

    memory_effect = (
        conditioned_result
        .state_after_distance
        > 0.0
    )

    assert physical_effect is True
    assert memory_effect is True


def test_physical_history_null_operator_result_does_not_imply_memory2_null(
    matched_result,
    conditioned_result,
):
    assert matched_result[
        "central_results"
    ][
        "history_specific_transition_effect_detected"
    ] is False

    assert (
        conditioned_result
        .state_after_distance
        > 0.0
    )


def test_system_evolution_and_memory2_are_not_the_same_transition_path():
    source = inspect.getsource(
        system_evolution_module
    )

    assert (
        "run_conditioned_dynamics"
        not in source
    )

    assert (
        "run_contextual_dynamics"
        not in source
    )

    assert (
        "build_associative_context"
        not in source
    )


# =============================================================================
# CENTRAL MECHANISM REGRESSION GUARD
# =============================================================================


def test_roif_memory_transition_architecture_holds_together(
    q8_result,
    matched_result,
    conditioned_result,
    sensitization_context_result,
    adaptation_context_result,
):
    """
    Central mechanism-level regression guard.

    Current implemented picture:

        physical history
            -> changed current connection state
            -> changed subsequent redistribution

    while:

        matched current state
        + different raw ordered HistoricalTrace
            -> no additional transition difference
               in the audited SystemEvolution path

    but separately:

        structured conditioning memory
            -> conditioned prestress
            -> changed attractor dynamics

        associative context
            -> context-specific prestress
            -> changed attractor dynamics

    Therefore a null matched-history result in SystemEvolution must NOT be
    generalized into a claim that ROIF Engine lacks structured memory-conditioned
    dynamics.
    """

    # -------------------------------------------------------------------------
    # PHYSICAL STATE-MEDIATED HISTORY
    # -------------------------------------------------------------------------

    assert q8_result[
        "state_control_audit"
    ][
        "baseline_vs_conditioned_connections_differ"
    ] is True

    assert q8_result[
        "central_results"
    ][
        "connection_state_changes_response_at_matched_prestress"
    ] is True

    # -------------------------------------------------------------------------
    # NO INDEPENDENT RAW-TRACE EFFECT IN AUDITED SYSTEM EVOLUTION
    # -------------------------------------------------------------------------

    assert matched_result[
        "central_results"
    ][
        "matched_current_state_established"
    ] is True

    assert matched_result[
        "central_results"
    ][
        "different_ordered_history_established"
    ] is True

    assert matched_result[
        "central_results"
    ][
        "history_specific_transition_effect_detected"
    ] is False

    assert matched_result[
        "central_results"
    ][
        "transition_response_distance"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    # -------------------------------------------------------------------------
    # STRUCTURED MEMORY 2.0 DOES CHANGE DYNAMICS
    # -------------------------------------------------------------------------

    assert (
        conditioned_result
        .state_after_distance
        > 0.0
    )

    assert conditioned_target_weight_gain(
        conditioned_result
    ) > 0.0

    assert (
        sensitization_context_result
        .state_after_distance
        > 0.0
    )

    assert (
        adaptation_context_result
        .state_after_distance
        > 0.0
    )

    assert (
        sensitization_context_result
        .contextual
        .state_after
        .feature_vector
        !=
        adaptation_context_result
        .contextual
        .state_after
        .feature_vector
    )
