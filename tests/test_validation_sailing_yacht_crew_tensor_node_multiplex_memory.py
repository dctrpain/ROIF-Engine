"""
Tests for Scenario 04I — Tensor Node / Multiplex Memory.

The benchmark validates the stronger memory architecture:

    many structural memory images
        -> one shared physical tensor-like node
        -> different query directions
        -> different response profiles

The tests intentionally avoid assuming:
    one image == one node
    evaluator label == retrieval target
    memory == action policy
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from validation.sailing.sailing_yacht_crew_tensor_node_multiplex_memory import (
    DEFAULT_HISTORY_PER_IMAGE,
    DEFAULT_IMAGE_COUNT,
    DEFAULT_QUERY_REPETITIONS,
    SCENARIO_ID,
    SHARED_NODE_ID,
    MultiplexImagePrototype,
    TensorMemoryNode,
    TensorNodeMemoryError,
    TensorNodeMultiplexResult,
    TensorNodeResponse,
    build_multiplex_history_record,
    build_tensor_memory_node,
    build_tensor_query_record,
    dominant_image_ids,
    multiplex_node_is_policy_free,
    respond_tensor_memory_node,
    response_energies,
    response_node_ids,
    response_profiles,
    run_tensor_node_multiplex_benchmark,
    unique_response_profile_count,
)


@pytest.fixture(scope="module")
def result() -> TensorNodeMultiplexResult:
    return run_tensor_node_multiplex_benchmark()


@pytest.fixture(scope="module")
def node() -> TensorMemoryNode:
    return build_tensor_memory_node()


# =============================================================================
# Identity / defaults
# =============================================================================


def test_scenario_id_is_04i() -> None:
    assert SCENARIO_ID == "sailing_04I_tensor_node_multiplex_memory"


def test_shared_node_id_regression() -> None:
    assert SHARED_NODE_ID == "tensor_node_shared_0001"


def test_default_history_per_image_is_16() -> None:
    assert DEFAULT_HISTORY_PER_IMAGE == 16


def test_default_image_count_is_four() -> None:
    assert DEFAULT_IMAGE_COUNT == 4


def test_default_query_repetitions_is_two() -> None:
    assert DEFAULT_QUERY_REPETITIONS == 2


# =============================================================================
# History record generation
# =============================================================================


def test_build_multiplex_history_record_returns_record() -> None:
    record = build_multiplex_history_record(
        event_index=1,
        image_index=0,
    )
    assert record.trace_id


def test_history_record_rejects_nonpositive_event_index() -> None:
    with pytest.raises(TensorNodeMemoryError):
        build_multiplex_history_record(
            event_index=0,
            image_index=0,
        )


def test_history_record_rejects_negative_image_index() -> None:
    with pytest.raises(TensorNodeMemoryError):
        build_multiplex_history_record(
            event_index=1,
            image_index=-1,
        )


def test_history_record_semantics_are_deterministic() -> None:
    left = build_multiplex_history_record(
        event_index=12,
        image_index=2,
    )
    right = build_multiplex_history_record(
        event_index=12,
        image_index=2,
    )

    assert left.trace_id == right.trace_id
    assert left.sequence_id == right.sequence_id
    assert left.pattern_id == right.pattern_id
    assert left.source_trace == right.source_trace
    assert left.signature.compact_vector() == right.signature.compact_vector()


def test_different_image_indices_change_structural_signature() -> None:
    left = build_multiplex_history_record(
        event_index=12,
        image_index=0,
    )
    right = build_multiplex_history_record(
        event_index=12,
        image_index=3,
    )

    assert left.signature.compact_vector() != right.signature.compact_vector()


# =============================================================================
# Shared tensor node
# =============================================================================


def test_tensor_node_type(node) -> None:
    assert isinstance(node, TensorMemoryNode)


def test_tensor_node_uses_shared_node_id(node) -> None:
    assert node.node_id == SHARED_NODE_ID


def test_tensor_node_contains_64_source_records(node) -> None:
    assert node.source_record_count == 64


def test_tensor_node_contains_four_image_prototypes(node) -> None:
    assert len(node.image_prototypes) == 4


def test_all_prototypes_are_multiplex_prototypes(node) -> None:
    assert all(
        isinstance(item, MultiplexImagePrototype)
        for item in node.image_prototypes
    )


def test_prototype_ids_are_unique(node) -> None:
    ids = tuple(item.image_id for item in node.image_prototypes)
    assert len(ids) == len(set(ids))


def test_each_prototype_has_16_source_records(node) -> None:
    assert all(
        len(item.source_record_ids) == 16
        for item in node.image_prototypes
    )


def test_each_prototype_prestress_weight_is_positive(node) -> None:
    assert all(
        item.prestress_weight > 0.0
        for item in node.image_prototypes
    )


def test_tensor_node_centroid_is_nonempty(node) -> None:
    assert len(node.centroid) > 0


def test_tensor_node_second_moment_is_square(node) -> None:
    width = len(node.centroid)
    assert len(node.second_moment) == width
    assert all(
        len(row) == width
        for row in node.second_moment
    )


def test_tensor_node_declares_shared_physical_node(node) -> None:
    assert node.metadata["shared_physical_node"] is True


def test_tensor_node_declares_multiplex_images(node) -> None:
    assert node.metadata["multiplex_structural_images"] is True


def test_tensor_node_declares_prestressed_response(node) -> None:
    assert node.metadata["prestressed_response_enabled"] is True


def test_tensor_node_declares_no_external_label_usage(node) -> None:
    assert node.metadata["external_expected_label_used"] is False


def test_tensor_node_is_frozen(node) -> None:
    with pytest.raises(FrozenInstanceError):
        node.node_id = "changed"  # type: ignore[misc]


def test_tensor_node_metadata_is_read_only(node) -> None:
    assert isinstance(node.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        node.metadata["x"] = 1  # type: ignore[index]


def test_prototype_metadata_is_read_only(node) -> None:
    prototype = node.image_prototypes[0]
    assert isinstance(prototype.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        prototype.metadata["x"] = 1  # type: ignore[index]


# =============================================================================
# Query / response
# =============================================================================


def test_tensor_query_record_builds() -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    assert query.trace_id


def test_response_returns_tensor_node_response(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert isinstance(response, TensorNodeResponse)


def test_response_uses_same_node_id(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert response.node_id == node.node_id


def test_response_energy_is_nonnegative(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert response.response_energy >= 0.0


def test_response_profile_contains_all_four_images(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert len(response.image_response_profile) == 4


def test_response_metadata_declares_structural_geometry(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert (
        response.metadata[
            "response_derived_from_structural_geometry"
        ]
        is True
    )


def test_response_declares_no_action_selection(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert response.metadata["action_selected"] is False


def test_response_declares_no_policy_modification(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert response.metadata["policy_modified"] is False


def test_response_declares_no_external_label_usage(node) -> None:
    query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    response = respond_tensor_memory_node(
        node,
        query,
    )
    assert response.metadata["external_expected_label_used"] is False


# =============================================================================
# Complete benchmark
# =============================================================================


def test_complete_benchmark_returns_result(result) -> None:
    assert isinstance(result, TensorNodeMultiplexResult)


def test_complete_benchmark_uses_one_shared_node(result) -> None:
    assert result.node.node_id == SHARED_NODE_ID


def test_complete_benchmark_contains_64_history_records(result) -> None:
    assert result.node.source_record_count == 64


def test_complete_benchmark_contains_four_images(result) -> None:
    assert len(result.node.image_prototypes) == 4


def test_complete_benchmark_contains_eight_queries(result) -> None:
    assert len(result.query_records) == 8
    assert len(result.responses) == 8


def test_all_responses_use_same_physical_node(result) -> None:
    assert set(response_node_ids(result)) == {SHARED_NODE_ID}


def test_metadata_confirms_all_queries_use_same_physical_node(result) -> None:
    assert result.metadata["all_queries_use_same_physical_node"] is True


def test_response_node_id_regression(result) -> None:
    assert response_node_ids(result) == (
        SHARED_NODE_ID,
        SHARED_NODE_ID,
        SHARED_NODE_ID,
        SHARED_NODE_ID,
        SHARED_NODE_ID,
        SHARED_NODE_ID,
        SHARED_NODE_ID,
        SHARED_NODE_ID,
    )


# =============================================================================
# Multiplex response behavior
# =============================================================================


def test_benchmark_has_multiple_dominant_images(result) -> None:
    dominant = tuple(
        item
        for item in dominant_image_ids(result)
        if item is not None
    )
    assert len(set(dominant)) >= 3


def test_all_queries_produce_distinct_response_profiles(result) -> None:
    assert unique_response_profile_count(result) == 8


def test_unique_response_profiles_exceed_image_count(result) -> None:
    assert unique_response_profile_count(result) > len(
        result.node.image_prototypes
    )


def test_response_profiles_are_not_all_equal(result) -> None:
    profiles = response_profiles(result)
    assert len(set(profiles)) > 1


def test_response_energies_are_not_all_equal(result) -> None:
    energies = response_energies(result)
    assert len(set(round(value, 12) for value in energies)) > 1


def test_response_energy_regression(result) -> None:
    assert response_energies(result) == pytest.approx(
        (
            0.5537120716969836,
            0.6883580163158082,
            0.8912006695057908,
            0.8451224105404027,
            0.736039506655064,
            0.8168799043336022,
            0.8561906731033577,
            0.548279618600366,
        )
    )


def test_dominant_image_regression(result) -> None:
    assert dominant_image_ids(result) == (
        "image_00",
        "image_03",
        "image_02",
        "image_02",
        "image_03",
        "image_01",
        "image_02",
        "image_00",
    )


def test_same_node_can_switch_dominant_response(result) -> None:
    ids = response_node_ids(result)
    dominant = dominant_image_ids(result)

    assert len(set(ids)) == 1
    assert len(set(dominant)) > 1


def test_same_node_can_emit_multiple_response_energies(result) -> None:
    assert len(set(response_node_ids(result))) == 1
    assert len(
        set(round(value, 12) for value in response_energies(result))
    ) > 1


# =============================================================================
# Policy / label boundary
# =============================================================================


def test_multiplex_node_is_policy_free(result) -> None:
    assert multiplex_node_is_policy_free(result) is True


def test_result_declares_memory_does_not_select_action(result) -> None:
    assert result.metadata["memory_selects_action"] is False


def test_result_declares_policy_override_disabled(result) -> None:
    assert result.metadata["policy_override_enabled"] is False


def test_result_declares_no_external_expected_label_usage(result) -> None:
    assert result.metadata["external_expected_label_used"] is False


def test_every_response_is_policy_free(result) -> None:
    assert all(
        response.metadata["action_selected"] is False
        and response.metadata["policy_modified"] is False
        for response in result.responses
    )


def test_every_response_declares_no_external_label_usage(result) -> None:
    assert all(
        response.metadata["external_expected_label_used"] is False
        for response in result.responses
    )


# =============================================================================
# Immutability
# =============================================================================


def test_result_query_records_are_tuple(result) -> None:
    assert isinstance(result.query_records, tuple)


def test_result_responses_are_tuple(result) -> None:
    assert isinstance(result.responses, tuple)


def test_result_metadata_is_read_only(result) -> None:
    assert isinstance(result.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_result_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.scenario_id = "changed"  # type: ignore[misc]


def test_response_is_frozen(result) -> None:
    with pytest.raises(FrozenInstanceError):
        result.responses[0].node_id = "changed"  # type: ignore[misc]


def test_response_metadata_is_read_only(result) -> None:
    response = result.responses[0]
    assert isinstance(response.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        response.metadata["x"] = 1  # type: ignore[index]


# =============================================================================
# Guards
# =============================================================================


def test_tensor_node_rejects_history_per_image_below_two() -> None:
    with pytest.raises(TensorNodeMemoryError):
        build_tensor_memory_node(
            history_per_image=1,
            image_count=4,
        )


def test_tensor_node_rejects_image_count_below_two() -> None:
    with pytest.raises(TensorNodeMemoryError):
        build_tensor_memory_node(
            history_per_image=2,
            image_count=1,
        )


def test_complete_benchmark_rejects_zero_query_repetitions() -> None:
    with pytest.raises(TensorNodeMemoryError):
        run_tensor_node_multiplex_benchmark(
            query_repetitions=0
        )


# =============================================================================
# Determinism
# =============================================================================


def _semantic_node_signature(node: TensorMemoryNode):
    return (
        node.node_id,
        node.source_record_count,
        node.centroid,
        node.second_moment,
        tuple(
            (
                prototype.image_id,
                prototype.source_record_ids,
                prototype.centroid,
                prototype.direction,
                prototype.prestress_weight,
                tuple(sorted(prototype.metadata.items())),
            )
            for prototype in node.image_prototypes
        ),
        tuple(sorted(node.metadata.items())),
    )


def _response_signature(response: TensorNodeResponse):
    return (
        response.query_record_id,
        response.node_id,
        response.query_direction,
        response.transformed_direction,
        response.image_response_profile,
        response.dominant_image_id,
        response.response_energy,
        tuple(sorted(response.metadata.items())),
    )


def test_tensor_node_construction_is_semantically_deterministic() -> None:
    left = build_tensor_memory_node()
    right = build_tensor_memory_node()

    assert _semantic_node_signature(left) == _semantic_node_signature(right)


def test_tensor_response_is_semantically_deterministic() -> None:
    node = build_tensor_memory_node()

    left_query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )
    right_query = build_tensor_query_record(
        query_index=65,
        image_index=0,
    )

    left = respond_tensor_memory_node(
        node,
        left_query,
    )
    right = respond_tensor_memory_node(
        node,
        right_query,
    )

    assert _response_signature(left) == _response_signature(right)


def test_complete_04i_benchmark_is_semantically_deterministic() -> None:
    left = run_tensor_node_multiplex_benchmark()
    right = run_tensor_node_multiplex_benchmark()

    assert _semantic_node_signature(left.node) == _semantic_node_signature(
        right.node
    )
    assert tuple(
        _response_signature(item)
        for item in left.responses
    ) == tuple(
        _response_signature(item)
        for item in right.responses
    )


def test_response_profiles_helper_is_deterministic() -> None:
    left = run_tensor_node_multiplex_benchmark()
    right = run_tensor_node_multiplex_benchmark()

    assert response_profiles(left) == response_profiles(right)
