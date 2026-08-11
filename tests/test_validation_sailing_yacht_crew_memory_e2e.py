"""
Tests for Scenario 04F — Autonomous Recursive Memory Retrieval E2E Benchmark

Updated for explicit RecursiveMemoryTree.

Pipeline:

    past ExperienceTrace records
        -> ControllerHistoryAdapter
        -> HierarchicalMemory
        -> RecursiveMemoryTree
        -> automatic nearest-cluster retrieval
        -> real ancestry: local -> parent -> ...
        -> PredictivePreload
        -> Predictive Control
        -> independent observation
        -> reduced prediction error

Core invariants:

    Retrieval != Action Selection
    PredictivePreload != Policy Override
    Cluster Match != Evaluator Label
    Recursive Tree != Ground Truth
    Observation != Preloaded Prediction
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.hierarchical_memory_retrieval import (
    retrieval_is_policy_free,
)
from roif.predictive_preload import (
    hierarchy_levels_used,
    preload_is_policy_free,
)
from roif.recursive_hierarchical_memory import (
    RecursiveMemoryTree,
    ancestry,
    tree_is_policy_free,
)

from validation.sailing.sailing_yacht_crew_memory_e2e import (
    DEFAULT_HISTORY_EVENT_COUNT,
    DEFAULT_QUERY_INDEX,
    SCENARIO_ID,
    SailingMemoryE2EError,
    SailingMemoryE2EHistory,
    SailingMemoryE2EQueryResult,
    SailingMemoryE2EResult,
    build_e2e_history,
    retrieval_level_ids,
    retrieval_tree_levels,
    retrieval_tree_node_ids,
    run_e2e_query,
    run_sailing_memory_e2e_benchmark,
    selected_action_id,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def result() -> SailingMemoryE2EResult:
    return run_sailing_memory_e2e_benchmark()


@pytest.fixture(scope="module")
def history() -> SailingMemoryE2EHistory:
    return build_e2e_history()


# =============================================================================
# Identity
# =============================================================================


def test_scenario_id_is_04f() -> None:
    assert SCENARIO_ID == "sailing_04F_memory_e2e"


def test_default_history_event_count_is_100() -> None:
    assert DEFAULT_HISTORY_EVENT_COUNT == 100


def test_default_query_index_is_101() -> None:
    assert DEFAULT_QUERY_INDEX == 101


def test_complete_run_returns_e2e_result(
    result,
) -> None:
    assert isinstance(
        result,
        SailingMemoryE2EResult,
    )


# =============================================================================
# History construction
# =============================================================================


def test_history_result_type(
    history,
) -> None:
    assert isinstance(
        history,
        SailingMemoryE2EHistory,
    )


def test_history_contains_100_events(
    history,
) -> None:
    assert history.event_count == 100


def test_history_contains_100_records(
    history,
) -> None:
    assert len(
        history.records
    ) == 100


def test_history_memory_contains_100_records(
    history,
) -> None:
    assert (
        history.hierarchical_memory.record_count
        == 100
    )


def test_history_memory_has_two_clusters(
    history,
) -> None:
    assert (
        history.hierarchical_memory.cluster_count
        == 2
    )


def test_history_contains_recursive_tree(
    history,
) -> None:
    assert isinstance(
        history.recursive_tree,
        RecursiveMemoryTree,
    )


def test_history_recursive_tree_has_three_nodes(
    history,
) -> None:
    assert history.recursive_tree.node_count == 3


def test_history_recursive_tree_has_one_root(
    history,
) -> None:
    assert history.recursive_tree.root_count == 1


def test_history_recursive_tree_max_level_is_two(
    history,
) -> None:
    assert history.recursive_tree.max_level == 2


def test_history_recursive_tree_preserves_two_leaf_clusters(
    history,
) -> None:
    assert len(
        history.recursive_tree.leaf_cluster_ids
    ) == 2


def test_history_recursive_tree_is_policy_free(
    history,
) -> None:
    assert tree_is_policy_free(
        history.recursive_tree
    ) is True


def test_history_declares_no_predictive_preload(
    history,
) -> None:
    assert (
        history.metadata[
            "history_built_without_predictive_preload"
        ]
        is True
    )


def test_history_declares_recursive_tree_built(
    history,
) -> None:
    assert (
        history.metadata[
            "recursive_tree_built"
        ]
        is True
    )


def test_history_metadata_preserves_recursive_tree_node_count(
    history,
) -> None:
    assert (
        history.metadata[
            "recursive_tree_node_count"
        ]
        == history.recursive_tree.node_count
    )


def test_history_metadata_preserves_recursive_tree_max_level(
    history,
) -> None:
    assert (
        history.metadata[
            "recursive_tree_max_level"
        ]
        == history.recursive_tree.max_level
    )


def test_history_cluster_assignment_is_label_free(
    history,
) -> None:
    assert (
        history.metadata[
            "cluster_assignment_uses_external_pattern_label"
        ]
        is False
    )


def test_history_declares_no_external_label_usage(
    history,
) -> None:
    assert (
        history.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Query shape
# =============================================================================


def test_query_result_type(
    result,
) -> None:
    assert isinstance(
        result.query,
        SailingMemoryE2EQueryResult,
    )


def test_query_uses_event_101(
    result,
) -> None:
    assert (
        result.query.event.event_id
        == "04C_event_00101_A"
    )


def test_query_reporting_family_is_a(
    result,
) -> None:
    assert (
        result.query.metadata[
            "query_pattern_family_for_reporting_only"
        ]
        == "A"
    )


def test_query_cluster_assignment_does_not_use_reporting_family(
    result,
) -> None:
    assert (
        result.query.metadata[
            "cluster_assignment_uses_external_pattern_label"
        ]
        is False
    )


def test_query_declares_recursive_tree_retrieval(
    result,
) -> None:
    assert (
        result.query.metadata[
            "recursive_tree_retrieval"
        ]
        is True
    )


# =============================================================================
# Automatic recursive retrieval
# =============================================================================


def test_query_retrieval_finds_cluster(
    result,
) -> None:
    assert (
        result.query.retrieval.nearest_cluster_id
        is not None
    )


def test_query_nearest_cluster_matches_regression(
    result,
) -> None:
    assert (
        result.query.retrieval.nearest_cluster_id
        == "cluster_0001"
    )


def test_query_nearest_distance_is_small(
    result,
) -> None:
    assert (
        result.query.retrieval.nearest_distance
        < 0.01
    )


def test_query_nearest_distance_matches_regression(
    result,
) -> None:
    assert (
        result.query.retrieval.nearest_distance
        == pytest.approx(
            0.002610248286973784
        )
    )


def test_retrieval_levels_are_local_and_parent(
    result,
) -> None:
    assert retrieval_level_ids(
        result
    ) == (
        0,
        1,
    )


def test_retrieval_tree_levels_are_one_two(
    result,
) -> None:
    assert retrieval_tree_levels(
        result
    ) == (
        1,
        2,
    )


def test_retrieval_uses_real_leaf_and_root_nodes(
    result,
) -> None:
    assert retrieval_tree_node_ids(
        result
    ) == (
        "node_l1_cluster_0001",
        "node_l2_0001",
    )


def test_query_metadata_preserves_retrieval_tree_levels(
    result,
) -> None:
    assert (
        result.query.metadata[
            "retrieval_tree_levels"
        ]
        == (
            1,
            2,
        )
    )


def test_query_metadata_preserves_retrieval_tree_node_ids(
    result,
) -> None:
    assert (
        result.query.metadata[
            "retrieval_tree_node_ids"
        ]
        == (
            "node_l1_cluster_0001",
            "node_l2_0001",
        )
    )


def test_retrieval_has_two_real_contributions(
    result,
) -> None:
    assert len(
        result.query.retrieval.contributions
    ) == 2


def test_retrieval_does_not_invent_grandparent(
    result,
) -> None:
    assert 2 not in retrieval_level_ids(
        result
    )


def test_retrieval_nodes_follow_real_tree_ancestry(
    result,
) -> None:
    chain = ancestry(
        result.history.recursive_tree,
        result.query.retrieval.leaf_node_id,
    )

    assert retrieval_tree_node_ids(
        result
    ) == tuple(
        node.node_id
        for node in chain
    )


def test_retrieval_is_policy_free(
    result,
) -> None:
    assert retrieval_is_policy_free(
        result.query.retrieval
    ) is True


def test_retrieval_declares_no_external_label_usage(
    result,
) -> None:
    assert (
        result.query.retrieval.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Predictive preload
# =============================================================================


def test_query_uses_memory(
    result,
) -> None:
    assert result.query.preload.used_memory is True


def test_preload_uses_real_available_nested_levels(
    result,
) -> None:
    assert hierarchy_levels_used(
        result.query.preload
    ) == (
        0,
        1,
    )


def test_preload_is_policy_free(
    result,
) -> None:
    assert preload_is_policy_free(
        result.query.preload
    ) is True


def test_preload_reduces_uncertainty(
    result,
) -> None:
    assert (
        result.query.preload.uncertainty_after
        <
        result.query.preload.uncertainty_before
    )


def test_preload_changes_course_error_expectation(
    result,
) -> None:
    assert (
        result.query.preload.preloaded_state.values[
            "course_error"
        ]
        != pytest.approx(
            result.query.preload.original_state.values[
                "course_error"
            ]
        )
    )


# =============================================================================
# Prediction error improvement
# =============================================================================


def test_baseline_error_matches_regression(
    result,
) -> None:
    assert (
        result.query.baseline_error
        == pytest.approx(
            0.08944271909999162
        )
    )


def test_recursive_memory_error_matches_regression(
    result,
) -> None:
    assert (
        result.query.memory_error
        == pytest.approx(
            0.029668658677675822
        )
    )


def test_recursive_memory_error_is_lower_than_baseline(
    result,
) -> None:
    assert (
        result.query.memory_error
        <
        result.query.baseline_error
    )


def test_recursive_improvement_is_positive(
    result,
) -> None:
    assert (
        result.query.improvement_fraction
        > 0.0
    )


def test_recursive_improvement_exceeds_65_percent(
    result,
) -> None:
    assert (
        result.query.improvement_fraction
        > 0.65
    )


def test_recursive_improvement_matches_regression(
    result,
) -> None:
    assert (
        result.query.improvement_fraction
        == pytest.approx(
            0.6682943119773893
        )
    )


def test_recursive_tree_improvement_exceeds_previous_synthetic_04f_regression(
    result,
) -> None:
    previous_synthetic_improvement = 0.6509598793634674

    assert (
        result.query.improvement_fraction
        >
        previous_synthetic_improvement
    )


# =============================================================================
# Action policy boundary
# =============================================================================


def test_selected_action_remains_coupled_helm_trim(
    result,
) -> None:
    assert (
        selected_action_id(
            result
        )
        == "coupled_helm_trim_action"
    )


def test_result_declares_policy_override_disabled(
    result,
) -> None:
    assert (
        result.metadata[
            "policy_override_enabled"
        ]
        is False
    )


def test_result_declares_memory_does_not_select_action(
    result,
) -> None:
    assert (
        result.metadata[
            "action_selected_by_memory"
        ]
        is False
    )


def test_query_declares_memory_not_used_for_action_selection(
    result,
) -> None:
    assert (
        result.query.metadata[
            "memory_retrieval_used_for_action_selection"
        ]
        is False
    )


def test_query_declares_policy_not_modified_by_memory(
    result,
) -> None:
    assert (
        result.query.metadata[
            "policy_modified_by_memory"
        ]
        is False
    )


# =============================================================================
# Independent observation
# =============================================================================


def test_observation_not_generated_from_preloaded_prediction(
    result,
) -> None:
    assert (
        result.query.observed_state.metadata[
            "memory_preload_used_to_generate_observation"
        ]
        is False
    )


def test_observation_source_is_raw_state_evaluator(
    result,
) -> None:
    assert (
        result.query.observed_state.metadata[
            "observation_source"
        ]
        == "independent_raw_state_plant_evaluator"
    )


# =============================================================================
# Label-leakage boundary
# =============================================================================


def test_result_cluster_assignment_declares_label_free(
    result,
) -> None:
    assert (
        result.metadata[
            "cluster_assignment_uses_external_pattern_label"
        ]
        is False
    )


def test_result_declares_recursive_memory_tree_enabled(
    result,
) -> None:
    assert (
        result.metadata[
            "recursive_memory_tree_enabled"
        ]
        is True
    )


def test_result_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_query_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.query.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_memory_trace_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.query.memory_trace.metadata[
            "external_expected_label_used"
        ]
        is False
    )


# =============================================================================
# Guards
# =============================================================================


def test_history_rejects_less_than_two_events() -> None:
    with pytest.raises(
        SailingMemoryE2EError
    ):
        build_e2e_history(
            event_count=1
        )


# =============================================================================
# Immutability
# =============================================================================


def test_history_records_are_tuple(
    history,
) -> None:
    assert isinstance(
        history.records,
        tuple,
    )


def test_history_metadata_is_read_only(
    history,
) -> None:
    assert isinstance(
        history.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        history.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_query_candidates_are_tuple(
    result,
) -> None:
    assert isinstance(
        result.query.candidates,
        tuple,
    )


def test_query_evaluations_are_tuple(
    result,
) -> None:
    assert isinstance(
        result.query.evaluations,
        tuple,
    )


def test_query_metadata_is_read_only(
    result,
) -> None:
    assert isinstance(
        result.query.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.query.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_result_metadata_is_read_only(
    result,
) -> None:
    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_history_is_frozen(
    history,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        history.event_count = 1  # type: ignore[misc]


def test_query_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.query.metadata = {}  # type: ignore[misc]


def test_result_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.scenario_id = "changed"  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def _result_signature(
    result: SailingMemoryE2EResult,
):
    return (
        result.scenario_id,
        result.history.event_count,
        result.history.hierarchical_memory.record_count,
        result.history.hierarchical_memory.cluster_count,
        result.history.recursive_tree.node_count,
        result.history.recursive_tree.root_ids,
        result.history.recursive_tree.max_level,
        result.query.event.event_id,
        result.query.retrieval.nearest_cluster_id,
        result.query.retrieval.nearest_distance,
        result.query.retrieval.leaf_node_id,
        retrieval_level_ids(result),
        retrieval_tree_levels(result),
        retrieval_tree_node_ids(result),
        tuple(
            (
                contribution.scar.scar_id,
                contribution.hierarchy_level,
                contribution.explicit_weight,
                contribution.source_cluster_id,
            )
            for contribution
            in result.query.retrieval.contributions
        ),
        result.query.preload.used_memory,
        tuple(
            sorted(
                result.query.preload.applied_bias_by_variable.items()
            )
        ),
        result.query.baseline_error,
        result.query.memory_error,
        result.query.improvement_fraction,
        selected_action_id(result),
        tuple(
            sorted(
                result.metadata.items()
            )
        ),
    )


def test_complete_recursive_e2e_benchmark_is_deterministic() -> None:
    left = run_sailing_memory_e2e_benchmark()
    right = run_sailing_memory_e2e_benchmark()

    assert _result_signature(
        left
    ) == _result_signature(
        right
    )


def test_query_against_same_recursive_history_is_deterministic(
    history,
) -> None:
    left = run_e2e_query(
        history
    )

    right = run_e2e_query(
        history
    )

    assert (
        left.retrieval.nearest_cluster_id
        == right.retrieval.nearest_cluster_id
    )

    assert (
        left.retrieval.nearest_distance
        == right.retrieval.nearest_distance
    )

    assert tuple(
        level.tree_node_id
        for level in left.retrieval.levels
    ) == tuple(
        level.tree_node_id
        for level in right.retrieval.levels
    )

    assert left.memory_error == right.memory_error
    assert left.baseline_error == right.baseline_error

    assert (
        left.decision.selected_candidate_id
        == right.decision.selected_candidate_id
    )
