"""
Tests for Scenario 04K — Contextual Prestress / Associative Transition.

The suite validates:

    SAME physical tensor node
    SAME stored memory
    SAME query input
    NO learning
    NO memory mutation
    NO policy override

        +

    DIFFERENT structural prestress

        -> different response geometry
        -> different dominant association
        -> different next-state attractor
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from validation.sailing.sailing_yacht_crew_contextual_prestress_memory import (
    CONTEXT_IDS,
    DEFAULT_CONTEXT_ALIGNMENT_GAIN,
    DEFAULT_HISTORY_PER_IMAGE,
    DEFAULT_IMAGE_COUNT,
    DEFAULT_PRESTRESS_GAIN,
    DEFAULT_QUERY_INDEX,
    SCENARIO_ID,
    ContextPrestressState,
    ContextualAssociationResponse,
    ContextualPrestressBenchmarkResult,
    ContextualPrestressError,
    build_context_prestress_states,
    context_ids,
    contextual_prestress_is_policy_free,
    dominant_association_ids,
    next_state_attractor_ids,
    respond_with_contextual_prestress,
    response_ambiguities,
    response_energies,
    response_margins,
    run_contextual_prestress_benchmark,
    unique_attractor_count,
    unique_dominant_association_count,
)
from validation.sailing.sailing_yacht_crew_tensor_node_multiplex_memory import (
    SHARED_NODE_ID,
    build_tensor_memory_node,
    build_tensor_query_record,
)


@pytest.fixture(scope="module")
def result() -> ContextualPrestressBenchmarkResult:
    return run_contextual_prestress_benchmark()


@pytest.fixture(scope="module")
def node():
    return build_tensor_memory_node(
        history_per_image=DEFAULT_HISTORY_PER_IMAGE,
        image_count=DEFAULT_IMAGE_COUNT,
    )


# =============================================================================
# Identity / defaults
# =============================================================================


def test_scenario_id_is_04k() -> None:
    assert SCENARIO_ID == "sailing_04K_contextual_prestress_memory"


def test_context_ids_regression() -> None:
    assert CONTEXT_IDS == (
        "office",
        "home",
        "son_room",
        "basement",
    )


def test_default_history_per_image_is_16() -> None:
    assert DEFAULT_HISTORY_PER_IMAGE == 16


def test_default_image_count_is_four() -> None:
    assert DEFAULT_IMAGE_COUNT == 4


def test_default_query_index_is_65() -> None:
    assert DEFAULT_QUERY_INDEX == 65


def test_default_prestress_gain_is_positive() -> None:
    assert DEFAULT_PRESTRESS_GAIN > 0.0


def test_default_context_alignment_gain_is_positive() -> None:
    assert DEFAULT_CONTEXT_ALIGNMENT_GAIN > 0.0


# =============================================================================
# Context construction
# =============================================================================


def test_context_builder_returns_four_contexts(node) -> None:
    contexts = build_context_prestress_states(node)
    assert len(contexts) == 4


def test_context_builder_returns_context_states(node) -> None:
    contexts = build_context_prestress_states(node)
    assert all(
        isinstance(context, ContextPrestressState)
        for context in contexts
    )


def test_context_builder_preserves_context_ids(node) -> None:
    contexts = build_context_prestress_states(node)
    assert tuple(context.context_id for context in contexts) == CONTEXT_IDS


def test_context_builder_uses_four_distinct_target_prototypes(node) -> None:
    contexts = build_context_prestress_states(node)
    targets = tuple(context.target_prototype_id for context in contexts)
    assert len(set(targets)) == 4


def test_context_prestress_vectors_have_node_dimension(node) -> None:
    contexts = build_context_prestress_states(node)
    assert all(
        len(context.prestress_vector) == len(node.centroid)
        for context in contexts
    )


def test_context_vectors_are_not_all_equal(node) -> None:
    contexts = build_context_prestress_states(node)
    assert len(
        set(context.prestress_vector for context in contexts)
    ) > 1


def test_context_builder_declares_structural_vector_encoding(node) -> None:
    contexts = build_context_prestress_states(node)
    assert all(
        context.metadata["context_encoded_as_structural_vector"] is True
        for context in contexts
    )


def test_context_builder_declares_no_label_selection(node) -> None:
    contexts = build_context_prestress_states(node)
    assert all(
        context.metadata["context_label_used_for_response_selection"] is False
        for context in contexts
    )


def test_context_builder_declares_no_external_label_usage(node) -> None:
    contexts = build_context_prestress_states(node)
    assert all(
        context.metadata["external_expected_label_used"] is False
        for context in contexts
    )


def test_context_builder_rejects_negative_prestress_gain(node) -> None:
    with pytest.raises(ContextualPrestressError):
        build_context_prestress_states(
            node,
            prestress_gain=-1.0,
        )


def test_context_metadata_is_read_only(node) -> None:
    context = build_context_prestress_states(node)[0]
    assert isinstance(context.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        context.metadata["x"] = 1  # type: ignore[index]


def test_context_state_is_frozen(node) -> None:
    context = build_context_prestress_states(node)[0]
    with pytest.raises(FrozenInstanceError):
        context.context_id = "changed"  # type: ignore[misc]


# =============================================================================
# Single contextual response
# =============================================================================


def test_contextual_response_type(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]

    response = respond_with_contextual_prestress(
        node,
        query,
        context,
    )

    assert isinstance(response, ContextualAssociationResponse)


def test_contextual_response_uses_same_physical_node(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert response.node_id == node.node_id


def test_contextual_response_preserves_query_identity(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert response.query_record_id == query.trace_id


def test_contextual_response_has_nonempty_profile(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert len(response.response_profile) == 4


def test_contextual_response_declares_prestress_only_difference(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert response.metadata["prestress_only_difference"] is True


def test_contextual_response_declares_no_learning(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert response.metadata["learning_applied"] is False


def test_contextual_response_declares_no_memory_mutation(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert response.metadata["memory_mutated"] is False


def test_contextual_response_declares_no_policy_modification(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert response.metadata["policy_modified"] is False


def test_contextual_response_declares_no_label_selection(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]
    response = respond_with_contextual_prestress(node, query, context)

    assert response.metadata["context_label_used_for_response_selection"] is False


def test_contextual_response_rejects_negative_alignment_gain(node) -> None:
    query = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    context = build_context_prestress_states(node)[0]

    with pytest.raises(ContextualPrestressError):
        respond_with_contextual_prestress(
            node,
            query,
            context,
            context_alignment_gain=-1.0,
        )


# =============================================================================
# Complete benchmark regression
# =============================================================================


def test_complete_benchmark_returns_result(result) -> None:
    assert isinstance(result, ContextualPrestressBenchmarkResult)


def test_complete_benchmark_uses_shared_tensor_node(result) -> None:
    assert result.node.node_id == SHARED_NODE_ID


def test_complete_benchmark_context_ids(result) -> None:
    assert context_ids(result) == CONTEXT_IDS


def test_complete_benchmark_has_four_responses(result) -> None:
    assert len(result.responses) == 4


def test_dominant_association_regression(result) -> None:
    assert dominant_association_ids(result) == (
        "image_00",
        "image_03",
        "image_02",
        "image_00",
    )


def test_next_state_attractor_regression(result) -> None:
    assert next_state_attractor_ids(result) == (
        "attractor::image_00",
        "attractor::image_03",
        "attractor::image_02",
        "attractor::image_00",
    )


def test_unique_dominant_count_regression(result) -> None:
    assert unique_dominant_association_count(result) == 3


def test_unique_attractor_count_regression(result) -> None:
    assert unique_attractor_count(result) == 3


def test_response_margin_regression(result) -> None:
    assert response_margins(result) == pytest.approx(
        (
            0.23850874292769886,
            0.2631393721267151,
            0.6681903771538918,
            0.17218860847630846,
        )
    )


def test_response_ambiguity_regression(result) -> None:
    assert response_ambiguities(result) == pytest.approx(
        (
            0.6348702518636233,
            0.5725217559113261,
            0.0,
            0.7389096763363507,
        )
    )


def test_response_energy_regression(result) -> None:
    assert response_energies(result) == pytest.approx(
        (
            0.5986741254672125,
            0.5913080616134126,
            0.4464783801210602,
            0.6724086014899706,
        )
    )


def test_same_query_plus_different_prestress_changes_dominant_association(result) -> None:
    assert len(set(dominant_association_ids(result))) > 1


def test_same_query_plus_different_prestress_changes_attractor(result) -> None:
    assert len(set(next_state_attractor_ids(result))) > 1


def test_same_query_plus_different_prestress_changes_response_geometry(result) -> None:
    transformed = tuple(
        response.transformed_direction
        for response in result.responses
    )
    assert len(set(transformed)) > 1


def test_same_query_plus_different_prestress_changes_energy(result) -> None:
    assert len(
        set(round(value, 12) for value in response_energies(result))
    ) > 1


def test_same_query_plus_different_prestress_changes_ambiguity(result) -> None:
    assert len(
        set(round(value, 12) for value in response_ambiguities(result))
    ) > 1


def test_one_context_can_produce_zero_ambiguity(result) -> None:
    assert any(
        value == pytest.approx(0.0)
        for value in response_ambiguities(result)
    )


# =============================================================================
# Architectural boundaries
# =============================================================================


def test_result_metadata_same_node(result) -> None:
    assert result.metadata["same_physical_node"] is True


def test_result_metadata_same_query(result) -> None:
    assert result.metadata["same_query_record"] is True


def test_result_metadata_same_memory(result) -> None:
    assert result.metadata["same_stored_memory"] is True


def test_result_metadata_no_learning_between_queries(result) -> None:
    assert result.metadata["learning_between_queries"] is False


def test_result_metadata_no_memory_mutation_between_queries(result) -> None:
    assert result.metadata["memory_mutated_between_queries"] is False


def test_result_metadata_no_label_selection(result) -> None:
    assert result.metadata["context_label_used_for_response_selection"] is False


def test_result_metadata_memory_does_not_select_action(result) -> None:
    assert result.metadata["memory_selects_action"] is False


def test_result_metadata_policy_override_disabled(result) -> None:
    assert result.metadata["policy_override_enabled"] is False


def test_result_metadata_no_external_label_usage(result) -> None:
    assert result.metadata["external_expected_label_used"] is False


def test_contextual_prestress_is_policy_free(result) -> None:
    assert contextual_prestress_is_policy_free(result) is True


def test_every_response_same_node(result) -> None:
    assert all(
        response.node_id == result.node.node_id
        for response in result.responses
    )


def test_every_response_same_query(result) -> None:
    assert all(
        response.query_record_id == result.query_record.trace_id
        for response in result.responses
    )


def test_every_response_no_learning(result) -> None:
    assert all(
        response.metadata["learning_applied"] is False
        for response in result.responses
    )


def test_every_response_no_memory_mutation(result) -> None:
    assert all(
        response.metadata["memory_mutated"] is False
        for response in result.responses
    )


def test_every_response_policy_free(result) -> None:
    assert all(
        response.metadata["action_selected"] is False
        and response.metadata["policy_modified"] is False
        for response in result.responses
    )


# =============================================================================
# Immutability
# =============================================================================


def test_result_contexts_are_tuple(result) -> None:
    assert isinstance(result.contexts, tuple)


def test_result_responses_are_tuple(result) -> None:
    assert isinstance(result.responses, tuple)


def test_result_metadata_is_read_only(result) -> None:
    assert isinstance(result.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_result_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.scenario_id = "changed"  # type: ignore[misc]


def test_response_metadata_is_read_only(result) -> None:
    response = result.responses[0]
    assert isinstance(response.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        response.metadata["x"] = 1  # type: ignore[index]


def test_response_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.responses[0].context_id = "changed"  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def _response_signature(response: ContextualAssociationResponse):
    return (
        response.query_record_id,
        response.node_id,
        response.context_id,
        response.raw_query_direction,
        response.prestressed_query_direction,
        response.transformed_direction,
        response.response_profile,
        response.dominant_association_id,
        response.dominant_activation,
        response.second_activation,
        response.dominant_margin,
        response.ambiguity,
        response.response_energy,
        response.next_state_attractor_id,
        tuple(sorted(response.metadata.items())),
    )


def _result_signature(result: ContextualPrestressBenchmarkResult):
    return (
        result.scenario_id,
        result.node.node_id,
        result.node.centroid,
        result.node.second_moment,
        result.query_record.trace_id,
        tuple(
            (
                context.context_id,
                context.prestress_vector,
                context.target_prototype_id,
                context.prestress_gain,
                tuple(sorted(context.metadata.items())),
            )
            for context in result.contexts
        ),
        tuple(
            _response_signature(response)
            for response in result.responses
        ),
        tuple(sorted(result.metadata.items())),
    )


def test_context_builder_is_deterministic(node) -> None:
    left = build_context_prestress_states(node)
    right = build_context_prestress_states(node)

    assert left == right


def test_single_contextual_response_is_deterministic(node) -> None:
    query_left = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )
    query_right = build_tensor_query_record(
        query_index=DEFAULT_QUERY_INDEX,
        image_index=0,
    )

    context = build_context_prestress_states(node)[1]

    left = respond_with_contextual_prestress(
        node,
        query_left,
        context,
    )
    right = respond_with_contextual_prestress(
        node,
        query_right,
        context,
    )

    assert _response_signature(left) == _response_signature(right)


def test_complete_04k_benchmark_is_deterministic() -> None:
    left = run_contextual_prestress_benchmark()
    right = run_contextual_prestress_benchmark()

    assert _result_signature(left) == _result_signature(right)


def test_same_query_raw_direction_is_identical_across_contexts(result) -> None:
    directions = tuple(
        response.raw_query_direction
        for response in result.responses
    )
    assert len(set(directions)) == 1


def test_prestressed_query_directions_are_context_dependent(result) -> None:
    directions = tuple(
        response.prestressed_query_direction
        for response in result.responses
    )
    assert len(set(directions)) > 1
