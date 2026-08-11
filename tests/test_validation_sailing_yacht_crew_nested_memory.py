"""
Tests for Scenario 04E — Nested / Matryoshka Memory Benchmark

The suite fixes the first nested-memory contract.

Hierarchy under test:

    local MemoryScar
        -> parent MemoryScar
            -> grandparent MemoryScar

Core invariants:

    Nested Memory != Action Selection
    Parent Prior != Local Override
    Higher-Order Context != Ground Truth
    PredictivePreload != Policy Mutation

Validated cases:
- aligned hierarchy;
- conflicting parent/grandparent;
- weak local memory supported by parent context;
- geometric hierarchy decay;
- policy-free preload;
- immutable benchmark result;
- deterministic execution.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.controller_memory import (
    MemoryScar,
)
from roif.predictive_preload import (
    PredictivePreload,
    hierarchy_levels_used,
    preload_is_policy_free,
)

from validation.sailing.sailing_yacht_crew_nested_memory import (
    SCENARIO_ID,
    NestedMemoryBenchmarkResult,
    NestedMemoryCaseResult,
    build_aligned_scars,
    build_conflicting_scars,
    build_preload_config,
    build_reference_state,
    build_weak_local_scars,
    case_bias,
    case_weight_series,
    run_nested_memory_benchmark,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def result() -> NestedMemoryBenchmarkResult:
    return run_nested_memory_benchmark()


# =============================================================================
# Identity
# =============================================================================


def test_scenario_id_is_04e() -> None:
    assert (
        SCENARIO_ID
        == "sailing_04E_nested_matryoshka_memory"
    )


def test_complete_run_returns_nested_memory_result(
    result,
) -> None:
    assert isinstance(
        result,
        NestedMemoryBenchmarkResult,
    )


def test_result_contains_three_cases(
    result,
) -> None:
    assert isinstance(
        result.aligned,
        NestedMemoryCaseResult,
    )
    assert isinstance(
        result.conflicting_parent,
        NestedMemoryCaseResult,
    )
    assert isinstance(
        result.weak_local,
        NestedMemoryCaseResult,
    )


# =============================================================================
# Reference state / config
# =============================================================================


def test_reference_state_is_deterministic() -> None:
    left = build_reference_state()
    right = build_reference_state()

    assert left == right


def test_reference_state_has_expected_values() -> None:
    state = build_reference_state()

    assert state.values[
        "course_error"
    ] == pytest.approx(
        1.0
    )

    assert state.values[
        "heel_error"
    ] == pytest.approx(
        0.5
    )


def test_reference_state_uncertainty_is_point_four() -> None:
    state = build_reference_state()

    assert state.uncertainty == pytest.approx(
        0.40
    )


def test_preload_config_uses_half_level_decay() -> None:
    config = build_preload_config()

    assert config.local_level_decay == pytest.approx(
        0.50
    )


def test_preload_config_requires_systematic_memory() -> None:
    config = build_preload_config()

    assert config.require_systematic_scar is True


# =============================================================================
# Scar factories
# =============================================================================


def test_aligned_factory_returns_three_scars() -> None:
    scars = build_aligned_scars()

    assert len(
        scars
    ) == 3

    assert all(
        isinstance(
            scar,
            MemoryScar,
        )
        for scar
        in scars
    )


def test_conflicting_factory_returns_three_scars() -> None:
    scars = build_conflicting_scars()

    assert len(
        scars
    ) == 3


def test_weak_local_factory_returns_three_scars() -> None:
    scars = build_weak_local_scars()

    assert len(
        scars
    ) == 3


# =============================================================================
# Aligned hierarchy
# =============================================================================


def test_aligned_case_uses_predictive_preload(
    result,
) -> None:
    assert isinstance(
        result.aligned.preload,
        PredictivePreload,
    )


def test_aligned_case_uses_all_three_hierarchy_levels(
    result,
) -> None:
    assert hierarchy_levels_used(
        result.aligned.preload
    ) == (
        0,
        1,
        2,
    )


def test_aligned_weights_decrease_with_depth(
    result,
) -> None:
    local, parent, grandparent = (
        case_weight_series(
            result.aligned
        )
    )

    assert (
        local
        >
        parent
        >
        grandparent
    )


def test_aligned_local_weight_matches_regression(
    result,
) -> None:
    assert result.aligned.local_weight == pytest.approx(
        0.75
    )


def test_aligned_parent_weight_matches_regression(
    result,
) -> None:
    assert result.aligned.parent_weight == pytest.approx(
        0.40375
    )


def test_aligned_grandparent_weight_matches_regression(
    result,
) -> None:
    assert result.aligned.grandparent_weight == pytest.approx(
        0.2025
    )


def test_aligned_bias_is_positive(
    result,
) -> None:
    course, heel = case_bias(
        result.aligned
    )

    assert course > 0.0
    assert heel > 0.0


def test_aligned_bias_matches_regression(
    result,
) -> None:
    course, heel = case_bias(
        result.aligned
    )

    assert course == pytest.approx(
        0.0510552995391705
    )

    assert heel == pytest.approx(
        0.02552764976958525
    )


def test_aligned_preload_reduces_uncertainty(
    result,
) -> None:
    preload = result.aligned.preload

    assert (
        preload.uncertainty_after
        <
        preload.uncertainty_before
    )


# =============================================================================
# Conflicting hierarchy
# =============================================================================


def test_conflicting_case_uses_all_three_levels(
    result,
) -> None:
    assert hierarchy_levels_used(
        result.conflicting_parent.preload
    ) == (
        0,
        1,
        2,
    )


def test_conflicting_weights_still_decrease_with_depth(
    result,
) -> None:
    local, parent, grandparent = (
        case_weight_series(
            result.conflicting_parent
        )
    )

    assert (
        local
        >
        parent
        >
        grandparent
    )


def test_conflicting_bias_remains_on_local_positive_side(
    result,
) -> None:
    course, heel = case_bias(
        result.conflicting_parent
    )

    assert course > 0.0
    assert heel > 0.0


def test_conflicting_bias_matches_regression(
    result,
) -> None:
    course, heel = case_bias(
        result.conflicting_parent
    )

    assert course == pytest.approx(
        0.022009216589861748
    )

    assert heel == pytest.approx(
        0.011004608294930874
    )


def test_conflicting_parent_reduces_but_does_not_reverse_local_bias(
    result,
) -> None:
    aligned_course, aligned_heel = case_bias(
        result.aligned
    )

    conflict_course, conflict_heel = case_bias(
        result.conflicting_parent
    )

    assert (
        0.0
        <
        conflict_course
        <
        aligned_course
    )

    assert (
        0.0
        <
        conflict_heel
        <
        aligned_heel
    )


# =============================================================================
# Weak local memory
# =============================================================================


def test_weak_local_case_uses_all_three_levels(
    result,
) -> None:
    assert hierarchy_levels_used(
        result.weak_local.preload
    ) == (
        0,
        1,
        2,
    )


def test_weak_local_weight_is_smaller_than_parent_weight(
    result,
) -> None:
    local, parent, _grandparent = (
        case_weight_series(
            result.weak_local
        )
    )

    assert parent > local


def test_weak_local_parent_is_strongest_contribution(
    result,
) -> None:
    local, parent, grandparent = (
        case_weight_series(
            result.weak_local
        )
    )

    assert (
        parent
        >
        grandparent
        >
        local
    )


def test_weak_local_weights_match_regression(
    result,
) -> None:
    assert case_weight_series(
        result.weak_local
    ) == (
        pytest.approx(
            0.08925
        ),
        pytest.approx(
            0.4275
        ),
        pytest.approx(
            0.21375
        ),
    )


def test_weak_local_bias_is_positive(
    result,
) -> None:
    course, heel = case_bias(
        result.weak_local
    )

    assert course > 0.0
    assert heel > 0.0


def test_weak_local_bias_matches_regression(
    result,
) -> None:
    course, heel = case_bias(
        result.weak_local
    )

    assert course == pytest.approx(
        0.039140143737166316
    )

    assert heel == pytest.approx(
        0.019570071868583158
    )


def test_parent_context_amplifies_weak_local_prior(
    result,
) -> None:
    weak_local_scar = build_weak_local_scars()[
        0
    ]

    local_only_expected_course = (
        weak_local_scar.bias_by_variable[
            "course_error"
        ]
        * build_preload_config().max_bias_fraction
    )

    course, _heel = case_bias(
        result.weak_local
    )

    assert course > local_only_expected_course


# =============================================================================
# Cross-case structure
# =============================================================================


def test_aligned_bias_exceeds_conflict_bias(
    result,
) -> None:
    aligned_course, _ = case_bias(
        result.aligned
    )

    conflict_course, _ = case_bias(
        result.conflicting_parent
    )

    assert aligned_course > conflict_course


def test_weak_local_supported_bias_exceeds_conflict_bias(
    result,
) -> None:
    weak_course, _ = case_bias(
        result.weak_local
    )

    conflict_course, _ = case_bias(
        result.conflicting_parent
    )

    assert weak_course > conflict_course


def test_all_cases_use_memory(
    result,
) -> None:
    assert result.aligned.preload.used_memory is True
    assert result.conflicting_parent.preload.used_memory is True
    assert result.weak_local.preload.used_memory is True


# =============================================================================
# Policy-free boundary
# =============================================================================


def test_aligned_preload_is_policy_free(
    result,
) -> None:
    assert preload_is_policy_free(
        result.aligned.preload
    ) is True


def test_conflicting_preload_is_policy_free(
    result,
) -> None:
    assert preload_is_policy_free(
        result.conflicting_parent.preload
    ) is True


def test_weak_local_preload_is_policy_free(
    result,
) -> None:
    assert preload_is_policy_free(
        result.weak_local.preload
    ) is True


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


def test_each_case_declares_policy_override_disabled(
    result,
) -> None:
    for case in (
        result.aligned,
        result.conflicting_parent,
        result.weak_local,
    ):
        assert (
            case.metadata[
                "policy_override_enabled"
            ]
            is False
        )


def test_each_case_declares_memory_does_not_select_action(
    result,
) -> None:
    for case in (
        result.aligned,
        result.conflicting_parent,
        result.weak_local,
    ):
        assert (
            case.metadata[
                "action_selected_by_memory"
            ]
            is False
        )


# =============================================================================
# External-label boundary
# =============================================================================


def test_result_declares_no_external_expected_label_usage(
    result,
) -> None:
    assert (
        result.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_each_case_declares_no_external_expected_label_usage(
    result,
) -> None:
    for case in (
        result.aligned,
        result.conflicting_parent,
        result.weak_local,
    ):
        assert (
            case.metadata[
                "external_expected_label_used"
            ]
            is False
        )


# =============================================================================
# Metadata / immutability
# =============================================================================


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


def test_case_metadata_is_read_only(
    result,
) -> None:
    assert isinstance(
        result.aligned.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.aligned.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_result_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.scenario_id = "changed"  # type: ignore[misc]


def test_case_is_frozen(
    result,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        result.aligned.local_weight = 0.0  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def _signature(
    result: NestedMemoryBenchmarkResult,
):
    return (
        result.scenario_id,
        case_weight_series(
            result.aligned
        ),
        case_bias(
            result.aligned
        ),
        case_weight_series(
            result.conflicting_parent
        ),
        case_bias(
            result.conflicting_parent
        ),
        case_weight_series(
            result.weak_local
        ),
        case_bias(
            result.weak_local
        ),
        result.aligned.preload.uncertainty_after,
        result.conflicting_parent.preload.uncertainty_after,
        result.weak_local.preload.uncertainty_after,
        tuple(
            sorted(
                result.metadata.items()
            )
        ),
    )


def test_complete_nested_memory_benchmark_is_deterministic() -> None:
    left = run_nested_memory_benchmark()
    right = run_nested_memory_benchmark()

    assert _signature(
        left
    ) == _signature(
        right
    )


def test_aligned_scar_factory_is_deterministic() -> None:
    assert (
        build_aligned_scars()
        == build_aligned_scars()
    )


def test_conflicting_scar_factory_is_deterministic() -> None:
    assert (
        build_conflicting_scars()
        == build_conflicting_scars()
    )


def test_weak_local_scar_factory_is_deterministic() -> None:
    assert (
        build_weak_local_scars()
        == build_weak_local_scars()
    )
