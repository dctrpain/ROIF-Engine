from __future__ import annotations

import pytest

from experiments import (
    roif_memory_conditioned_matched_state_transition_benchmark
    as bench
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def result():
    return bench.run_benchmark()


@pytest.fixture(scope="module")
def central(
    result,
):
    return result[
        "central_results"
    ]


@pytest.fixture(scope="module")
def design(
    result,
):
    return result[
        "experimental_design"
    ]


@pytest.fixture(scope="module")
def memories(
    result,
):
    return result[
        "memory_conditions"
    ]


@pytest.fixture(scope="module")
def physical(
    result,
):
    return result[
        "physical_results"
    ]


@pytest.fixture(scope="module")
def distances(
    result,
):
    return result[
        "response_distances"
    ]


# =============================================================================
# BENCHMARK IDENTITY
# =============================================================================


def test_benchmark_version(
    result,
):
    assert result[
        "benchmark_version"
    ] == (
        "roif_memory_conditioned_"
        "matched_state_transition_v1"
    )


def test_claim_scope_is_computational_only(
    result,
):
    assert result[
        "claim_scope"
    ] == (
        "computational_model_only"
    )


def test_mechanism_under_test_is_explicit(
    result,
):
    assert result[
        "mechanism_under_test"
    ] == (
        "structured_conditioning_memory_to_"
        "prestress_transfer_transition"
    )


# =============================================================================
# MATCHED-STATE EXPERIMENTAL DESIGN
# =============================================================================


def test_current_system_image_is_matched(
    design,
):
    assert design[
        "matched_current_system_image"
    ] is True


def test_query_cue_is_matched(
    design,
):
    assert design[
        "same_query_cue"
    ] is True


def test_semantic_attractor_is_matched(
    design,
):
    assert design[
        "same_semantic_attractor"
    ] is True


def test_explicit_binding_is_matched(
    design,
):
    assert design[
        "same_explicit_binding"
    ] is True


def test_physical_probe_is_matched(
    design,
):
    assert design[
        "same_physical_probe_event"
    ] is True


def test_physical_topology_is_matched(
    design,
):
    assert design[
        "same_physical_topology"
    ] is True


def test_conditioning_history_differs(
    design,
):
    assert design[
        "different_conditioning_history"
    ] is True


def test_modifier_values_are_not_manually_set(
    design,
):
    assert design[
        "modifier_values_set_manually"
    ] is False


# =============================================================================
# STRUCTURED MEMORY DIFFERENCE
# =============================================================================


def test_memory_ids_differ(
    memories,
):
    assert (
        memories[
            "reinforced"
        ][
            "memory_id"
        ]
        !=
        memories[
            "unreinforced"
        ][
            "memory_id"
        ]
    )


def test_observation_counts_are_matched(
    memories,
):
    assert (
        memories[
            "reinforced"
        ][
            "observation_count"
        ]
        ==
        memories[
            "unreinforced"
        ][
            "observation_count"
        ]
    )


def test_reinforced_memory_has_positive_conditioned_response(
    memories,
):
    assert (
        memories[
            "reinforced"
        ][
            "conditioned_response_strength"
        ]
        > 0.0
    )


def test_unreinforced_memory_has_zero_conditioned_response(
    memories,
):
    assert memories[
        "unreinforced"
    ][
        "conditioned_response_strength"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_memory_semantic_signals_differ(
    memories,
):
    reinforced = memories[
        "reinforced"
    ][
        "conditioned_response_strength"
    ]

    unreinforced = memories[
        "unreinforced"
    ][
        "conditioned_response_strength"
    ]

    assert (
        reinforced
        != pytest.approx(
            unreinforced,
            abs=1e-12,
        )
    )


def test_predicted_semantic_attractor_is_same(
    memories,
):
    assert (
        memories[
            "reinforced"
        ][
            "predicted_attractor_id"
        ]
        ==
        memories[
            "unreinforced"
        ][
            "predicted_attractor_id"
        ]
        ==
        bench.SEMANTIC_ATTRACTOR_ID
    )


# =============================================================================
# MEMORY -> MODIFIER DERIVATION
# =============================================================================


def test_reinforced_derivation_contains_modifier(
    memories,
):
    modifiers = memories[
        "reinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ]

    assert len(
        modifiers
    ) == 1


def test_unreinforced_derivation_contains_zero_modifier(
    memories,
):
    modifiers = memories[
        "unreinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ]

    assert len(
        modifiers
    ) == 1

    assert modifiers[
        0
    ][
        "value"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_reinforced_modifier_is_positive(
    memories,
):
    value = memories[
        "reinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ][
        0
    ][
        "value"
    ]

    assert value > 0.0


def test_memory_derived_modifier_values_differ(
    memories,
):
    reinforced = memories[
        "reinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ][
        0
    ][
        "value"
    ]

    unreinforced = memories[
        "unreinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ][
        0
    ][
        "value"
    ]

    assert (
        reinforced
        != pytest.approx(
            unreinforced,
            abs=1e-12,
        )
    )


def test_modifier_channel_is_prestress_transfer(
    memories,
):
    assert memories[
        "reinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ][
        0
    ][
        "channel"
    ] == (
        "prestress_transfer"
    )


def test_modifier_physical_target_is_A_B(
    memories,
):
    assert memories[
        "reinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ][
        0
    ][
        "target_id"
    ] == (
        bench.PHYSICAL_TARGET_CONNECTION_ID
    )


def test_memory_provenance_preserved(
    memories,
):
    evidence = memories[
        "reinforced"
    ][
        "derivation"
    ][
        "modifiers"
    ][
        0
    ][
        "evidence"
    ]

    assert len(
        evidence
    ) == 1

    assert evidence[
        0
    ][
        "source_kind"
    ] == (
        "conditioning_memory"
    )

    assert evidence[
        0
    ][
        "source_id"
    ] == (
        "reinforced_memory"
    )


def test_same_binding_id_used_for_both_memory_conditions(
    memories,
):
    reinforced = memories[
        "reinforced"
    ][
        "derivation"
    ][
        "applied_binding_ids"
    ]

    unreinforced = memories[
        "unreinforced"
    ][
        "derivation"
    ][
        "applied_binding_ids"
    ]

    assert reinforced == (
        unreinforced
    )

    assert reinforced == [
        "sensitization_to_A_B_transfer"
    ]


# =============================================================================
# PHYSICAL SOURCE / BINDING
# =============================================================================


def test_binding_is_explicit(
    result,
):
    binding = result[
        "physical_source"
    ][
        "binding"
    ]

    assert binding[
        "semantic_target_id"
    ] == (
        bench.SEMANTIC_ATTRACTOR_ID
    )

    assert binding[
        "physical_target_id"
    ] == (
        bench.PHYSICAL_TARGET_CONNECTION_ID
    )


def test_binding_channel_is_prestress_transfer(
    result,
):
    assert result[
        "physical_source"
    ][
        "binding"
    ][
        "channel"
    ] == (
        "prestress_transfer"
    )


def test_binding_scale_is_bounded(
    result,
):
    scale = result[
        "physical_source"
    ][
        "binding"
    ][
        "scale"
    ]

    assert 0.0 <= scale <= 1.0


# =============================================================================
# SAME PHYSICAL STATE / SAME CONNECTION EVOLUTION
# =============================================================================


def test_adaptive_connection_signatures_equal_across_all_conditions(
    physical,
):
    no_memory = physical[
        "no_memory_control"
    ][
        "adaptive_connection_signature"
    ]

    reinforced = physical[
        "reinforced_memory"
    ][
        "adaptive_connection_signature"
    ]

    unreinforced = physical[
        "unreinforced_memory"
    ][
        "adaptive_connection_signature"
    ]

    assert (
        no_memory
        ==
        reinforced
        ==
        unreinforced
    )


def test_connection_change_norm_equal_across_conditions(
    physical,
):
    no_memory = physical[
        "no_memory_control"
    ][
        "connection_change_norm"
    ]

    reinforced = physical[
        "reinforced_memory"
    ][
        "connection_change_norm"
    ]

    unreinforced = physical[
        "unreinforced_memory"
    ][
        "connection_change_norm"
    ]

    assert reinforced == pytest.approx(
        no_memory
    )

    assert unreinforced == pytest.approx(
        no_memory
    )


def test_prestress_before_equal_across_conditions(
    physical,
):
    assert (
        physical[
            "no_memory_control"
        ][
            "prestress_before"
        ]
        ==
        physical[
            "reinforced_memory"
        ][
            "prestress_before"
        ]
        ==
        physical[
            "unreinforced_memory"
        ][
            "prestress_before"
        ]
    )


# =============================================================================
# PHYSICAL TRANSITION EFFECT
# =============================================================================


def test_reinforced_transition_differs_from_unreinforced(
    distances,
):
    assert distances[
        "reinforced_vs_unreinforced"
    ] > 0.0


def test_reinforced_transition_differs_from_no_memory(
    distances,
):
    assert distances[
        "reinforced_vs_no_memory"
    ] > 0.0


def test_unreinforced_transition_matches_no_memory(
    distances,
):
    assert distances[
        "unreinforced_vs_no_memory"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_reinforced_memory_changes_downstream_B_response(
    physical,
):
    reinforced = abs(
        physical[
            "reinforced_memory"
        ][
            "prestress_delta"
        ][
            "B"
        ]
    )

    unreinforced = abs(
        physical[
            "unreinforced_memory"
        ][
            "prestress_delta"
        ][
            "B"
        ]
    )

    assert (
        reinforced
        >
        unreinforced
    )


def test_reinforced_memory_changes_downstream_C_response(
    physical,
):
    reinforced = abs(
        physical[
            "reinforced_memory"
        ][
            "prestress_delta"
        ][
            "C"
        ]
    )

    unreinforced = abs(
        physical[
            "unreinforced_memory"
        ][
            "prestress_delta"
        ][
            "C"
        ]
    )

    assert (
        reinforced
        >
        unreinforced
    )


def test_reinforced_memory_changes_downstream_D_response(
    physical,
):
    reinforced = abs(
        physical[
            "reinforced_memory"
        ][
            "prestress_delta"
        ][
            "D"
        ]
    )

    unreinforced = abs(
        physical[
            "unreinforced_memory"
        ][
            "prestress_delta"
        ][
            "D"
        ]
    )

    assert (
        reinforced
        >
        unreinforced
    )


def test_source_node_A_event_delta_remains_same(
    physical,
):
    no_memory = physical[
        "no_memory_control"
    ][
        "prestress_delta"
    ][
        "A"
    ]

    reinforced = physical[
        "reinforced_memory"
    ][
        "prestress_delta"
    ][
        "A"
    ]

    unreinforced = physical[
        "unreinforced_memory"
    ][
        "prestress_delta"
    ][
        "A"
    ]

    assert reinforced == pytest.approx(
        no_memory
    )

    assert unreinforced == pytest.approx(
        no_memory
    )


# =============================================================================
# MODIFIER APPLICATION STATE
# =============================================================================


def test_reinforced_condition_applies_active_modifier(
    physical,
):
    assert physical[
        "reinforced_memory"
    ][
        "transition_modifiers_applied"
    ] is True

    assert physical[
        "reinforced_memory"
    ][
        "active_transition_modifier_count"
    ] == 1


def test_unreinforced_condition_has_no_active_modifier(
    physical,
):
    assert physical[
        "unreinforced_memory"
    ][
        "transition_modifiers_applied"
    ] is False

    assert physical[
        "unreinforced_memory"
    ][
        "active_transition_modifier_count"
    ] == 0


def test_no_memory_control_has_no_modifier(
    physical,
):
    assert physical[
        "no_memory_control"
    ][
        "transition_modifiers_applied"
    ] is False

    assert physical[
        "no_memory_control"
    ][
        "active_transition_modifier_count"
    ] == 0


# =============================================================================
# CENTRAL RESULT FLAGS
# =============================================================================


def test_central_matched_current_state(
    central,
):
    assert central[
        "matched_current_state_established"
    ] is True


def test_central_structured_memory_conditions_differ(
    central,
):
    assert central[
        "structured_memory_conditions_differ"
    ] is True


def test_central_semantic_signal_differs(
    central,
):
    assert central[
        "memory_derived_semantic_signal_differs"
    ] is True


def test_central_transition_modifier_differs(
    central,
):
    assert central[
        "memory_derived_transition_modifier_differs"
    ] is True


def test_central_same_probe_applied(
    central,
):
    assert central[
        "same_physical_probe_applied"
    ] is True


def test_central_physical_transition_differs(
    central,
):
    assert central[
        "physical_transition_differs_by_memory_condition"
    ] is True


def test_central_adaptive_state_equal(
    central,
):
    assert central[
        "adaptive_connection_state_equal_across_conditions"
    ] is True


def test_central_source_image_immutable(
    central,
):
    assert central[
        "source_system_image_remained_immutable"
    ] is True


def test_central_transition_distance_positive(
    central,
):
    assert central[
        "reinforced_transition_distance_from_unreinforced"
    ] > 0.0


# =============================================================================
# CLAIM BOUNDARIES
# =============================================================================


def test_structured_memory_to_transition_path_is_tested(
    result,
):
    assert result[
        "structured_memory_to_transition_path_tested"
    ] is True


def test_conditioning_memory_derivation_is_tested(
    result,
):
    assert result[
        "conditioning_memory_derivation_tested"
    ] is True


def test_explicit_semantic_physical_binding_is_used(
    result,
):
    assert result[
        "explicit_semantic_physical_binding_used"
    ] is True


def test_prestress_transfer_channel_is_tested(
    result,
):
    assert result[
        "prestress_transfer_channel_tested"
    ] is True


def test_automatic_topology_mapping_is_not_tested(
    result,
):
    assert result[
        "automatic_topology_mapping_tested"
    ] is False


def test_connection_update_memory_modulation_is_not_tested(
    result,
):
    assert result[
        "connection_update_memory_modulation_tested"
    ] is False


def test_whole_system_history_operator_is_not_tested(
    result,
):
    assert result[
        "whole_system_history_operator_tested"
    ] is False


def test_general_history_conditioned_operator_not_claimed(
    result,
):
    assert result[
        "general_history_conditioned_operator_claimed"
    ] is False


def test_predictive_preconfiguration_not_tested(
    result,
):
    assert result[
        "predictive_preconfiguration_tested"
    ] is False


def test_future_temporal_image_prediction_not_tested(
    result,
):
    assert result[
        "future_temporal_image_prediction_tested"
    ] is False


def test_stabilization_not_tested(
    result,
):
    assert result[
        "stabilization_tested"
    ] is False


def test_biological_learning_not_claimed(
    result,
):
    assert result[
        "biological_learning_claimed"
    ] is False


def test_clinical_validity_not_claimed(
    result,
):
    assert result[
        "clinical_validity_claimed"
    ] is False


def test_universal_memory_dynamics_not_claimed(
    result,
):
    assert result[
        "universal_memory_dynamics_claimed"
    ] is False


# =============================================================================
# INTERPRETATION BOUNDARIES
# =============================================================================


def test_interpretation_mentions_matched_current_state(
    result,
):
    text = result[
        "interpretation"
    ].lower()

    assert (
        "matched current physical state"
        in text
    )


def test_interpretation_mentions_explicit_binding(
    result,
):
    text = result[
        "interpretation"
    ].lower()

    assert (
        "explicit semantic-to-physical binding"
        in text
    )


def test_interpretation_mentions_prestress_transfer(
    result,
):
    text = result[
        "interpretation"
    ].lower()

    assert (
        "prestress_transfer"
        in text
    )


def test_interpretation_rejects_general_history_operator_claim(
    result,
):
    text = result[
        "interpretation"
    ].lower()

    assert (
        "does not establish a general "
        "history-conditioned operator"
        in text
    )


def test_interpretation_rejects_predictive_stabilization_claim(
    result,
):
    text = result[
        "interpretation"
    ].lower()

    assert (
        "predictive stabilization"
        in text
    )


def test_interpretation_rejects_temporal_image_prediction_claim(
    result,
):
    text = result[
        "interpretation"
    ].lower()

    assert (
        "temporal-image prediction"
        in text
    )


# =============================================================================
# DETERMINISM
# =============================================================================


def test_benchmark_is_deterministic():
    left = bench.run_benchmark()
    right = bench.run_benchmark()

    assert left == right


def test_memory_derivation_is_deterministic():
    left = (
        bench.derive_memory_conditions()
    )

    right = (
        bench.derive_memory_conditions()
    )

    assert (
        left[
            "reinforced_derivation"
        ].modifier_set.modifiers[
            0
        ].value
        ==
        pytest.approx(
            right[
                "reinforced_derivation"
            ].modifier_set.modifiers[
                0
            ].value
        )
    )


def test_matched_current_image_is_deterministic():
    left = (
        bench.build_matched_current_image()
    )

    right = (
        bench.build_matched_current_image()
    )

    assert (
        bench.system_image_signature(
            left
        )
        ==
        bench.system_image_signature(
            right
        )
    )


# =============================================================================
# END-TO-END CAUSAL-PATH REGRESSION GUARD
# =============================================================================


def test_memory_difference_propagates_through_full_implemented_path(
    result,
    central,
):
    """
    End-to-end implemented path:

        different conditioning history
            ->
        different ConditioningMemory
            ->
        different conditioned semantic signal
            ->
        different derived TransitionModifierSet
            ->
        same physical SystemImage
        + same physical probe
            ->
        different physical prestress transition

    while adaptive connection state remains matched.

    This is a computational path test only.
    """

    assert result[
        "experimental_design"
    ][
        "different_conditioning_history"
    ] is True

    assert central[
        "structured_memory_conditions_differ"
    ] is True

    assert central[
        "memory_derived_semantic_signal_differs"
    ] is True

    assert central[
        "memory_derived_transition_modifier_differs"
    ] is True

    assert central[
        "matched_current_state_established"
    ] is True

    assert central[
        "same_physical_probe_applied"
    ] is True

    assert central[
        "physical_transition_differs_by_memory_condition"
    ] is True

    assert central[
        "adaptive_connection_state_equal_across_conditions"
    ] is True


# =============================================================================
# CENTRAL CLAIM / MECHANISM GUARD
# =============================================================================


def test_memory_conditioned_matched_state_transition_result_holds_together(
    result,
    central,
    distances,
):
    """
    Central scientific regression guard.

    Demonstrated:

        structured memory
            -> derived bounded transition modifier
            -> different next physical transition

    under matched current physical state and same event.

    Not demonstrated:

        general Phi_H,
        whole-SystemImage memory conditioning,
        predictive stabilization,
        Temporal-Image prediction,
        biological learning,
        clinical validity,
        universal memory dynamics.
    """

    assert central[
        "matched_current_state_established"
    ] is True

    assert central[
        "structured_memory_conditions_differ"
    ] is True

    assert central[
        "memory_derived_semantic_signal_differs"
    ] is True

    assert central[
        "memory_derived_transition_modifier_differs"
    ] is True

    assert central[
        "physical_transition_differs_by_memory_condition"
    ] is True

    assert central[
        "adaptive_connection_state_equal_across_conditions"
    ] is True

    assert distances[
        "reinforced_vs_unreinforced"
    ] > 0.0

    assert result[
        "general_history_conditioned_operator_claimed"
    ] is False

    assert result[
        "whole_system_history_operator_tested"
    ] is False

    assert result[
        "predictive_preconfiguration_tested"
    ] is False

    assert result[
        "future_temporal_image_prediction_tested"
    ] is False

    assert result[
        "stabilization_tested"
    ] is False

    assert result[
        "biological_learning_claimed"
    ] is False

    assert result[
        "clinical_validity_claimed"
    ] is False
