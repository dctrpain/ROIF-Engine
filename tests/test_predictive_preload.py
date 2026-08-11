"""
Tests for roif.predictive_preload

The suite fixes the first memory-to-prediction bridge contract.

Core invariants:

    Memory Retrieval != Action Selection
    PredictivePreload != Policy Override
    MemoryScar != Direct State Mutation
    Hierarchical Context != Ground Truth

The tests validate:
- scar eligibility;
- systematic-memory gating;
- nested / Matryoshka hierarchy weighting;
- local-memory dominance over parent memory;
- bounded weighted bias aggregation;
- uncertainty reduction;
- state immutability;
- no new variable invention;
- policy-free audit metadata;
- deterministic preload behavior.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from roif.controller_memory import (
    MemoryPattern,
    MemoryScar,
)
from roif.predictive_control import (
    PredictiveState,
)
from roif.predictive_preload import (
    AppliedPreloadContribution,
    InvalidPredictivePreloadError,
    MemoryPreloadContribution,
    PredictivePreload,
    PredictivePreloadConfig,
    aggregate_memory_bias,
    apply_predictive_preload,
    contribution_is_eligible,
    effective_contribution_weight,
    hierarchy_levels_used,
    preload_changed_variable,
    preload_is_policy_free,
)


# =============================================================================
# Helpers
# =============================================================================


def state(
    *,
    course: float = 1.0,
    heel: float = 0.5,
    uncertainty: float = 0.4,
) -> PredictiveState:
    return PredictiveState(
        values={
            "course_error": course,
            "heel_error": heel,
        },
        target_values={
            "course_error": 0.0,
            "heel_error": 0.0,
        },
        timestamp=1.0,
        uncertainty=uncertainty,
        metadata={
            "external_expected_label_used": False,
        },
    )


def scar(
    *,
    scar_id: str = "scar_A",
    pattern_id: str = "A",
    course_bias: float = 0.08,
    heel_bias: float = 0.04,
    recurrence_count: int = 3,
    consistency: float = 1.0,
    confidence: float = 0.6,
) -> MemoryScar:
    improved = recurrence_count
    unchanged = 0
    worsened = 0

    return MemoryScar(
        scar_id=scar_id,
        pattern=MemoryPattern(
            pattern_id=pattern_id
        ),
        trace_ids=tuple(
            f"{scar_id}_{index}"
            for index
            in range(
                recurrence_count
            )
        ),
        bias_by_variable={
            "course_error": course_bias,
            "heel_error": heel_bias,
        },
        error_norm_mean=(
            course_bias ** 2
            + heel_bias ** 2
        ) ** 0.5,
        error_norm_last=(
            course_bias ** 2
            + heel_bias ** 2
        ) ** 0.5,
        recurrence_count=recurrence_count,
        consistency=consistency,
        confidence=confidence,
        improved_count=improved,
        unchanged_count=unchanged,
        worsened_count=worsened,
        metadata={
            "predictive_preload_applied": False,
            "predictive_state_mutated": False,
            "policy_modified": False,
            "graph_mutated": False,
            "external_expected_label_used": False,
        },
    )


# =============================================================================
# Config
# =============================================================================


def test_default_config_is_valid() -> None:
    config = PredictivePreloadConfig()

    assert config.max_bias_fraction == pytest.approx(
        0.75
    )
    assert config.local_level_decay == pytest.approx(
        0.50
    )


def test_config_clamps_bias_fraction() -> None:
    config = PredictivePreloadConfig(
        max_bias_fraction=2.0
    )

    assert config.max_bias_fraction == 1.0


def test_config_clamps_uncertainty_scale() -> None:
    config = PredictivePreloadConfig(
        uncertainty_reduction_scale=-1.0
    )

    assert config.uncertainty_reduction_scale == 0.0


def test_config_is_frozen() -> None:
    config = PredictivePreloadConfig()

    with pytest.raises(
        FrozenInstanceError
    ):
        config.max_bias_fraction = 0.1  # type: ignore[misc]


# =============================================================================
# Contribution validation
# =============================================================================


def test_contribution_accepts_local_level() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(),
        hierarchy_level=0,
    )

    assert contribution.hierarchy_level == 0


def test_contribution_rejects_negative_hierarchy_level() -> None:
    with pytest.raises(
        InvalidPredictivePreloadError
    ):
        MemoryPreloadContribution(
            scar=scar(),
            hierarchy_level=-1,
        )


def test_contribution_rejects_negative_weight() -> None:
    with pytest.raises(
        InvalidPredictivePreloadError
    ):
        MemoryPreloadContribution(
            scar=scar(),
            explicit_weight=-1.0,
        )


def test_contribution_metadata_is_read_only() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(),
        metadata={
            "x": 1,
        },
    )

    assert isinstance(
        contribution.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        contribution.metadata[
            "x"
        ] = 2  # type: ignore[index]


def test_contribution_is_frozen() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(),
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        contribution.hierarchy_level = 1  # type: ignore[misc]


# =============================================================================
# Eligibility
# =============================================================================


def test_consistent_repeated_scar_is_eligible() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar()
    )

    assert contribution_is_eligible(
        contribution,
        config=PredictivePreloadConfig(),
    ) is True


def test_low_confidence_scar_is_not_eligible() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(
            confidence=0.10
        )
    )

    assert contribution_is_eligible(
        contribution,
        config=PredictivePreloadConfig(),
    ) is False


def test_low_consistency_scar_is_not_eligible() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(
            consistency=0.50
        )
    )

    assert contribution_is_eligible(
        contribution,
        config=PredictivePreloadConfig(),
    ) is False


def test_single_recurrence_scar_is_not_systematic_by_default() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(
            recurrence_count=1,
            confidence=0.6,
        )
    )

    assert contribution_is_eligible(
        contribution,
        config=PredictivePreloadConfig(),
    ) is False


def test_zero_explicit_weight_is_not_eligible() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(),
        explicit_weight=0.0,
    )

    assert contribution_is_eligible(
        contribution,
        config=PredictivePreloadConfig(),
    ) is False


def test_systematic_requirement_can_be_disabled() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(
            recurrence_count=1,
            confidence=0.6,
        )
    )

    config = PredictivePreloadConfig(
        require_systematic_scar=False,
    )

    assert contribution_is_eligible(
        contribution,
        config=config,
    ) is True


# =============================================================================
# Matryoshka weighting
# =============================================================================


def test_local_level_has_larger_weight_than_parent() -> None:
    config = PredictivePreloadConfig()

    local = MemoryPreloadContribution(
        scar=scar(),
        hierarchy_level=0,
    )

    parent = MemoryPreloadContribution(
        scar=scar(
            scar_id="parent"
        ),
        hierarchy_level=1,
    )

    assert (
        effective_contribution_weight(
            local,
            config=config,
        )
        >
        effective_contribution_weight(
            parent,
            config=config,
        )
    )


def test_parent_has_larger_weight_than_grandparent() -> None:
    config = PredictivePreloadConfig()

    parent = MemoryPreloadContribution(
        scar=scar(
            scar_id="parent"
        ),
        hierarchy_level=1,
    )

    grandparent = MemoryPreloadContribution(
        scar=scar(
            scar_id="grandparent"
        ),
        hierarchy_level=2,
    )

    assert (
        effective_contribution_weight(
            parent,
            config=config,
        )
        >
        effective_contribution_weight(
            grandparent,
            config=config,
        )
    )


def test_level_decay_is_geometric() -> None:
    config = PredictivePreloadConfig(
        local_level_decay=0.5
    )

    local = MemoryPreloadContribution(
        scar=scar(),
        hierarchy_level=0,
    )
    parent = MemoryPreloadContribution(
        scar=scar(
            scar_id="parent"
        ),
        hierarchy_level=1,
    )
    grandparent = MemoryPreloadContribution(
        scar=scar(
            scar_id="grandparent"
        ),
        hierarchy_level=2,
    )

    w0 = effective_contribution_weight(
        local,
        config=config,
    )
    w1 = effective_contribution_weight(
        parent,
        config=config,
    )
    w2 = effective_contribution_weight(
        grandparent,
        config=config,
    )

    assert w1 == pytest.approx(
        w0 * 0.5
    )

    assert w2 == pytest.approx(
        w0 * 0.25
    )


def test_higher_explicit_weight_can_strengthen_parent() -> None:
    config = PredictivePreloadConfig()

    parent = MemoryPreloadContribution(
        scar=scar(
            scar_id="parent"
        ),
        hierarchy_level=1,
        explicit_weight=2.0,
    )

    plain_parent = MemoryPreloadContribution(
        scar=scar(
            scar_id="plain_parent"
        ),
        hierarchy_level=1,
        explicit_weight=1.0,
    )

    assert (
        effective_contribution_weight(
            parent,
            config=config,
        )
        >
        effective_contribution_weight(
            plain_parent,
            config=config,
        )
    )


# =============================================================================
# Bias aggregation
# =============================================================================


def test_single_scar_bias_is_fractionally_applied() -> None:
    config = PredictivePreloadConfig(
        max_bias_fraction=0.75
    )

    bias, applied = aggregate_memory_bias(
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
        config=config,
    )

    assert bias[
        "course_error"
    ] == pytest.approx(
        0.08 * 0.75
    )

    assert bias[
        "heel_error"
    ] == pytest.approx(
        0.04 * 0.75
    )

    assert len(
        applied
    ) == 1


def test_ineligible_scar_contributes_no_bias() -> None:
    config = PredictivePreloadConfig()

    bias, applied = aggregate_memory_bias(
        (
            MemoryPreloadContribution(
                scar=scar(
                    confidence=0.1
                )
            ),
        ),
        config=config,
    )

    assert dict(
        bias
    ) == {}

    assert applied == ()


def test_local_and_parent_bias_are_weighted_average() -> None:
    config = PredictivePreloadConfig(
        max_bias_fraction=1.0,
        local_level_decay=0.5,
    )

    local = MemoryPreloadContribution(
        scar=scar(
            scar_id="local",
            course_bias=0.10,
            heel_bias=0.0,
        ),
        hierarchy_level=0,
    )

    parent = MemoryPreloadContribution(
        scar=scar(
            scar_id="parent",
            course_bias=0.02,
            heel_bias=0.0,
        ),
        hierarchy_level=1,
    )

    bias, _ = aggregate_memory_bias(
        (
            local,
            parent,
        ),
        config=config,
    )

    local_w = effective_contribution_weight(
        local,
        config=config,
    )

    parent_w = effective_contribution_weight(
        parent,
        config=config,
    )

    expected = (
        0.10 * local_w
        + 0.02 * parent_w
    ) / (
        local_w
        + parent_w
    )

    assert bias[
        "course_error"
    ] == pytest.approx(
        expected
    )


def test_local_memory_dominates_opposite_parent_bias() -> None:
    config = PredictivePreloadConfig(
        max_bias_fraction=1.0,
        local_level_decay=0.5,
    )

    local = MemoryPreloadContribution(
        scar=scar(
            scar_id="local",
            course_bias=0.10,
            heel_bias=0.0,
        ),
        hierarchy_level=0,
    )

    parent = MemoryPreloadContribution(
        scar=scar(
            scar_id="parent",
            course_bias=-0.10,
            heel_bias=0.0,
        ),
        hierarchy_level=1,
    )

    bias, _ = aggregate_memory_bias(
        (
            local,
            parent,
        ),
        config=config,
    )

    assert bias[
        "course_error"
    ] > 0.0


def test_multiple_levels_do_not_sum_unboundedly() -> None:
    config = PredictivePreloadConfig(
        max_bias_fraction=1.0,
    )

    contributions = tuple(
        MemoryPreloadContribution(
            scar=scar(
                scar_id=f"scar_{level}",
                course_bias=0.08,
                heel_bias=0.04,
            ),
            hierarchy_level=level,
        )
        for level
        in range(
            5
        )
    )

    bias, _ = aggregate_memory_bias(
        contributions,
        config=config,
    )

    assert bias[
        "course_error"
    ] == pytest.approx(
        0.08
    )

    assert bias[
        "heel_error"
    ] == pytest.approx(
        0.04
    )


# =============================================================================
# Applying preload
# =============================================================================


def test_apply_preload_returns_predictive_preload() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert isinstance(
        preload,
        PredictivePreload,
    )


def test_apply_preload_does_not_mutate_original_state() -> None:
    original = state()

    preload = apply_predictive_preload(
        original,
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert original.values[
        "course_error"
    ] == pytest.approx(
        1.0
    )

    assert (
        preload.preloaded_state
        is not original
    )


def test_preload_shifts_course_in_remembered_bias_direction() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert (
        preload.preloaded_state.values[
            "course_error"
        ]
        >
        preload.original_state.values[
            "course_error"
        ]
    )


def test_preload_shifts_heel_in_remembered_bias_direction() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert (
        preload.preloaded_state.values[
            "heel_error"
        ]
        >
        preload.original_state.values[
            "heel_error"
        ]
    )


def test_preload_applied_bias_matches_default_fraction() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert preload.applied_bias_by_variable[
        "course_error"
    ] == pytest.approx(
        0.06
    )

    assert preload.applied_bias_by_variable[
        "heel_error"
    ] == pytest.approx(
        0.03
    )


def test_preload_reduces_uncertainty_when_memory_used() -> None:
    preload = apply_predictive_preload(
        state(
            uncertainty=0.4
        ),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert (
        preload.uncertainty_after
        <
        preload.uncertainty_before
    )


def test_preload_preserves_uncertainty_when_no_memory_used() -> None:
    preload = apply_predictive_preload(
        state(
            uncertainty=0.4
        ),
        (),
    )

    assert preload.uncertainty_after == pytest.approx(
        0.4
    )


def test_preload_used_memory_is_true_when_bias_applied() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert preload.used_memory is True


def test_preload_used_memory_is_false_without_contributions() -> None:
    preload = apply_predictive_preload(
        state(),
        (),
    )

    assert preload.used_memory is False


# =============================================================================
# No variable invention
# =============================================================================


def test_preload_does_not_invent_unknown_state_variable() -> None:
    strange_scar = MemoryScar(
        scar_id="strange",
        pattern=MemoryPattern(
            pattern_id="strange"
        ),
        trace_ids=(
            "x1",
            "x2",
        ),
        bias_by_variable={
            "unknown_variable": 100.0,
        },
        error_norm_mean=100.0,
        error_norm_last=100.0,
        recurrence_count=2,
        consistency=1.0,
        confidence=0.5,
        improved_count=2,
        unchanged_count=0,
        worsened_count=0,
    )

    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=strange_scar
            ),
        ),
    )

    assert (
        "unknown_variable"
        not in preload.preloaded_state.values
    )


def test_unknown_only_bias_does_not_mark_memory_used() -> None:
    strange_scar = MemoryScar(
        scar_id="strange",
        pattern=MemoryPattern(
            pattern_id="strange"
        ),
        trace_ids=(
            "x1",
            "x2",
        ),
        bias_by_variable={
            "unknown_variable": 100.0,
        },
        error_norm_mean=100.0,
        error_norm_last=100.0,
        recurrence_count=2,
        consistency=1.0,
        confidence=0.5,
        improved_count=2,
        unchanged_count=0,
        worsened_count=0,
    )

    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=strange_scar
            ),
        ),
    )

    assert preload.used_memory is False


# =============================================================================
# Policy-free boundary
# =============================================================================


def test_preload_does_not_select_action() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert (
        preload.metadata[
            "action_selected"
        ]
        is False
    )


def test_preload_does_not_modify_policy() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert (
        preload.metadata[
            "policy_modified"
        ]
        is False
    )


def test_preload_does_not_mutate_graph() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert (
        preload.metadata[
            "graph_mutated"
        ]
        is False
    )


def test_preload_state_declares_no_action_selected() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert (
        preload.preloaded_state.metadata[
            "action_selected"
        ]
        is False
    )


def test_preload_is_policy_free_returns_true() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert preload_is_policy_free(
        preload
    ) is True


def test_policy_free_detects_forbidden_flag() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    altered = replace(
        preload,
        metadata={
            **dict(
                preload.metadata
            ),
            "policy_modified": True,
        },
    )

    assert preload_is_policy_free(
        altered
    ) is False


# =============================================================================
# Matryoshka inspection
# =============================================================================


def test_hierarchy_levels_used_reports_local_parent_grandparent() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar(
                    scar_id="local"
                ),
                hierarchy_level=0,
            ),
            MemoryPreloadContribution(
                scar=scar(
                    scar_id="parent"
                ),
                hierarchy_level=1,
            ),
            MemoryPreloadContribution(
                scar=scar(
                    scar_id="grandparent"
                ),
                hierarchy_level=2,
            ),
        ),
    )

    assert hierarchy_levels_used(
        preload
    ) == (
        0,
        1,
        2,
    )


def test_applied_contribution_weights_decrease_with_depth() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar(
                    scar_id="local"
                ),
                hierarchy_level=0,
            ),
            MemoryPreloadContribution(
                scar=scar(
                    scar_id="parent"
                ),
                hierarchy_level=1,
            ),
            MemoryPreloadContribution(
                scar=scar(
                    scar_id="grandparent"
                ),
                hierarchy_level=2,
            ),
        ),
    )

    weights = {
        item.hierarchy_level:
            item.effective_weight
        for item
        in preload.applied_contributions
    }

    assert (
        weights[
            0
        ]
        >
        weights[
            1
        ]
        >
        weights[
            2
        ]
    )


def test_preload_changed_variable_helper() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert preload_changed_variable(
        preload,
        "course_error",
    ) is True


def test_preload_changed_variable_false_for_unknown() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert preload_changed_variable(
        preload,
        "unknown",
    ) is False


# =============================================================================
# Immutability
# =============================================================================


def test_applied_contribution_bias_is_read_only() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    item = preload.applied_contributions[
        0
    ]

    assert isinstance(
        item,
        AppliedPreloadContribution,
    )

    with pytest.raises(TypeError):
        item.bias_by_variable[
            "course_error"
        ] = 1.0  # type: ignore[index]


def test_preload_applied_bias_is_read_only() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert isinstance(
        preload.applied_bias_by_variable,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        preload.applied_bias_by_variable[
            "course_error"
        ] = 1.0  # type: ignore[index]


def test_preload_contributions_are_tuple() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    assert isinstance(
        preload.applied_contributions,
        tuple,
    )


def test_preload_metadata_is_read_only() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    with pytest.raises(TypeError):
        preload.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_preload_is_frozen() -> None:
    preload = apply_predictive_preload(
        state(),
        (
            MemoryPreloadContribution(
                scar=scar()
            ),
        ),
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        preload.used_memory = False  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_effective_weight_is_deterministic() -> None:
    contribution = MemoryPreloadContribution(
        scar=scar(),
        hierarchy_level=2,
    )

    config = PredictivePreloadConfig()

    left = effective_contribution_weight(
        contribution,
        config=config,
    )

    right = effective_contribution_weight(
        contribution,
        config=config,
    )

    assert left == right


def test_bias_aggregation_is_deterministic() -> None:
    contributions = (
        MemoryPreloadContribution(
            scar=scar(
                scar_id="local"
            ),
            hierarchy_level=0,
        ),
        MemoryPreloadContribution(
            scar=scar(
                scar_id="parent"
            ),
            hierarchy_level=1,
        ),
    )

    config = PredictivePreloadConfig()

    left = aggregate_memory_bias(
        contributions,
        config=config,
    )

    right = aggregate_memory_bias(
        contributions,
        config=config,
    )

    assert left == right


def test_complete_predictive_preload_is_deterministic() -> None:
    original = state()

    contributions = (
        MemoryPreloadContribution(
            scar=scar(
                scar_id="local"
            ),
            hierarchy_level=0,
        ),
        MemoryPreloadContribution(
            scar=scar(
                scar_id="parent"
            ),
            hierarchy_level=1,
        ),
        MemoryPreloadContribution(
            scar=scar(
                scar_id="grandparent"
            ),
            hierarchy_level=2,
        ),
    )

    left = apply_predictive_preload(
        original,
        contributions,
    )

    right = apply_predictive_preload(
        original,
        contributions,
    )

    assert left == right
