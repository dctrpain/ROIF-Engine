"""
Tests for roif.controller_memory

The suite fixes the first persistent controller-memory contract.

It validates:
- immutable ControllerMemory snapshots;
- append-as-copy semantics;
- bounded memory;
- duplicate trace protection;
- pattern retrieval;
- MemoryScar derivation from repeated systematic prediction error;
- bias aggregation;
- directional consistency;
- recurrence-based confidence;
- stabilization outcome counts;
- no PredictivePreload application;
- no predictive-state mutation;
- no graph mutation;
- deterministic memory behavior.

Core invariants:

    ExperienceTrace != Learning
    ControllerMemory != PredictivePreload
    MemoryScar != PredictiveState Mutation
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import sqrt
from types import MappingProxyType

import pytest

from roif.controller_memory import (
    ControllerMemory,
    DuplicateExperienceTraceError,
    InvalidControllerMemoryError,
    MemoryPattern,
    MemoryScar,
    MemoryScarError,
    append_experience_trace,
    append_experience_traces,
    derive_memory_scar,
    derive_memory_scar_from_memory,
    filter_traces,
    memory_is_preload_free,
    most_recent_trace,
    scar_bias_norm,
    scar_is_preload_free,
    scar_is_systematic,
    traces_for_pattern,
)
from roif.experience_trace import (
    ExperienceTrace,
    ExperienceTraceIdentity,
    StabilizationOutcome,
    build_experience_trace,
)
from roif.predictive_control import (
    CandidateEvaluation,
    ControlCandidate,
    ControlCandidateKind,
    ControlReserve,
    DisturbanceEstimate,
    PredictedOutcome,
    PredictiveControlDecision,
    PredictiveControlDecisionKind,
    PredictiveState,
    StabilizationDemand,
)


# =============================================================================
# Helpers
# =============================================================================


def _initial_state(
    *,
    residual: float = 1.0,
    uncertainty: float = 0.3,
) -> PredictiveState:
    return PredictiveState(
        values={
            "course_error": residual,
            "heel_error": 0.5,
        },
        target_values={
            "course_error": 0.0,
            "heel_error": 0.0,
        },
        timestamp=0.0,
        uncertainty=uncertainty,
        metadata={
            "external_expected_label_used": False,
        },
    )


def _observed_state(
    *,
    course: float,
    heel: float,
    uncertainty: float = 0.2,
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


def _candidate(
    candidate_id: str = "coupled",
) -> ControlCandidate:
    return ControlCandidate(
        candidate_id=candidate_id,
        kind=ControlCandidateKind.CORRECTIVE,
        control_delta={
            "rudder": -0.2,
            "trim": -0.2,
        },
        estimated_cost=0.2,
        reversibility=0.9,
        safety_risk=0.05,
        uncertainty=0.1,
        metadata={
            "external_expected_label_used": False,
        },
    )


def _evaluation(
    *,
    candidate_id: str = "coupled",
    predicted_course: float = 0.4,
    predicted_heel: float = 0.2,
) -> CandidateEvaluation:
    candidate = _candidate(
        candidate_id
    )

    predicted_state = PredictiveState(
        values={
            "course_error": predicted_course,
            "heel_error": predicted_heel,
        },
        target_values={
            "course_error": 0.0,
            "heel_error": 0.0,
        },
        timestamp=1.0,
        uncertainty=0.15,
    )

    outcome = PredictedOutcome(
        candidate_id=candidate_id,
        predicted_state=predicted_state,
        predicted_residual_error=predicted_state.residual_norm(),
        predicted_control_cost=0.2,
        predicted_uncertainty=0.15,
        confidence=0.85,
    )

    return CandidateEvaluation(
        candidate=candidate,
        outcome=outcome,
        reserve_after_action=1.0,
        stabilization_margin=-0.3,
        score=1.0,
        safe=True,
        viable=True,
    )


def _trace(
    *,
    trace_id: str,
    episode_index: int,
    pattern_family: str,
    course_bias: float,
    heel_bias: float,
    observed_residual_override: tuple[float, float] | None = None,
) -> ExperienceTrace:
    evaluation = _evaluation()

    decision = PredictiveControlDecision(
        kind=PredictiveControlDecisionKind.ACTION,
        selected_candidate_id=evaluation.candidate.candidate_id,
        selected_evaluation=evaluation,
        ranked_evaluations=(
            evaluation,
        ),
        reason="test",
        metadata={
            "external_expected_label_used": False,
        },
    )

    predicted = evaluation.outcome.predicted_state

    if observed_residual_override is None:
        course = (
            predicted.values[
                "course_error"
            ]
            + course_bias
        )

        heel = (
            predicted.values[
                "heel_error"
            ]
            + heel_bias
        )
    else:
        course, heel = observed_residual_override

    observed = _observed_state(
        course=course,
        heel=heel,
    )

    return build_experience_trace(
        identity=ExperienceTraceIdentity(
            trace_id=trace_id,
            episode_index=episode_index,
            sequence_id="sequence_A",
            timestamp_start=float(
                episode_index
            ),
            timestamp_end=float(
                episode_index + 1
            ),
            metadata={
                "pattern_family": pattern_family,
                "external_expected_label_used": False,
            },
        ),
        initial_state=_initial_state(),
        disturbance=DisturbanceEstimate(
            components={
                "wind": 0.3,
                "wave": 0.2,
            },
            confidence=0.9,
            metadata={
                "pattern_family": pattern_family,
                "external_expected_label_used": False,
            },
        ),
        demand=StabilizationDemand(
            state_error=1.0,
            disturbance_load=0.5,
            uncertainty_load=0.1,
        ),
        reserve=ControlReserve(
            capacity=2.0,
            committed=0.5,
        ),
        candidates=(
            evaluation.candidate,
        ),
        evaluations=(
            evaluation,
        ),
        decision=decision,
        observed_state=observed,
        metadata={
            "pattern_family": pattern_family,
            "external_expected_label_used": False,
        },
    )


def _pattern_a_trace(
    index: int,
) -> ExperienceTrace:
    return _trace(
        trace_id=f"A_{index}",
        episode_index=index,
        pattern_family="A",
        course_bias=0.08,
        heel_bias=0.04,
    )


def _pattern_b_trace(
    index: int,
) -> ExperienceTrace:
    return _trace(
        trace_id=f"B_{index}",
        episode_index=index,
        pattern_family="B",
        course_bias=-0.03,
        heel_bias=0.09,
    )


# =============================================================================
# MemoryPattern
# =============================================================================


def test_memory_pattern_accepts_valid_id() -> None:
    pattern = MemoryPattern(
        pattern_id="A"
    )

    assert pattern.pattern_id == "A"


def test_memory_pattern_rejects_empty_id() -> None:
    with pytest.raises(
        InvalidControllerMemoryError
    ):
        MemoryPattern(
            pattern_id=""
        )


def test_memory_pattern_features_are_read_only() -> None:
    pattern = MemoryPattern(
        pattern_id="A",
        features={
            "wind": 0.3,
        },
    )

    assert isinstance(
        pattern.features,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        pattern.features[
            "wind"
        ] = 1.0  # type: ignore[index]


def test_memory_pattern_metadata_is_read_only() -> None:
    pattern = MemoryPattern(
        pattern_id="A",
        metadata={
            "source": "test",
        },
    )

    with pytest.raises(TypeError):
        pattern.metadata[
            "source"
        ] = "x"  # type: ignore[index]


def test_memory_pattern_is_frozen() -> None:
    pattern = MemoryPattern(
        pattern_id="A"
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        pattern.pattern_id = "B"  # type: ignore[misc]


# =============================================================================
# Empty ControllerMemory
# =============================================================================


def test_empty_memory_is_valid() -> None:
    memory = ControllerMemory()

    assert memory.size == 0
    assert memory.trace_ids == ()


def test_empty_memory_is_preload_free() -> None:
    memory = ControllerMemory()

    assert memory_is_preload_free(
        memory
    ) is True


def test_memory_rejects_nonpositive_max_traces() -> None:
    with pytest.raises(
        InvalidControllerMemoryError
    ):
        ControllerMemory(
            max_traces=0
        )


def test_memory_metadata_is_read_only() -> None:
    memory = ControllerMemory(
        metadata={
            "x": 1,
        }
    )

    with pytest.raises(TypeError):
        memory.metadata[
            "x"
        ] = 2  # type: ignore[index]


def test_memory_is_frozen() -> None:
    memory = ControllerMemory()

    with pytest.raises(
        FrozenInstanceError
    ):
        memory.max_traces = 10  # type: ignore[misc]


# =============================================================================
# Append-as-copy semantics
# =============================================================================


def test_append_returns_new_memory_snapshot() -> None:
    memory = ControllerMemory()

    trace = _pattern_a_trace(
        1
    )

    updated = append_experience_trace(
        memory,
        trace,
    )

    assert updated is not memory


def test_append_does_not_mutate_original_memory() -> None:
    memory = ControllerMemory()

    updated = append_experience_trace(
        memory,
        _pattern_a_trace(
            1
        ),
    )

    assert memory.size == 0
    assert updated.size == 1


def test_append_preserves_trace_identity() -> None:
    memory = append_experience_trace(
        ControllerMemory(),
        _pattern_a_trace(
            1
        ),
    )

    assert memory.trace_ids == (
        "A_1",
    )


def test_append_multiple_traces_preserves_order() -> None:
    memory = append_experience_traces(
        ControllerMemory(),
        (
            _pattern_a_trace(
                1
            ),
            _pattern_b_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert memory.trace_ids == (
        "A_1",
        "B_2",
        "A_3",
    )


def test_append_rejects_duplicate_trace_id() -> None:
    trace = _pattern_a_trace(
        1
    )

    memory = append_experience_trace(
        ControllerMemory(),
        trace,
    )

    with pytest.raises(
        DuplicateExperienceTraceError
    ):
        append_experience_trace(
            memory,
            trace,
        )


def test_memory_constructor_rejects_duplicate_trace_ids() -> None:
    trace = _pattern_a_trace(
        1
    )

    with pytest.raises(
        DuplicateExperienceTraceError
    ):
        ControllerMemory(
            traces=(
                trace,
                trace,
            )
        )


# =============================================================================
# Bounded memory
# =============================================================================


def test_bounded_memory_keeps_last_n_traces() -> None:
    memory = append_experience_traces(
        ControllerMemory(
            max_traces=2
        ),
        (
            _pattern_a_trace(
                1
            ),
            _pattern_b_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert memory.trace_ids == (
        "B_2",
        "A_3",
    )


def test_bounded_append_does_not_mutate_previous_snapshot() -> None:
    base = append_experience_traces(
        ControllerMemory(
            max_traces=2
        ),
        (
            _pattern_a_trace(
                1
            ),
            _pattern_b_trace(
                2
            ),
        ),
    )

    updated = append_experience_trace(
        base,
        _pattern_a_trace(
            3
        ),
    )

    assert base.trace_ids == (
        "A_1",
        "B_2",
    )

    assert updated.trace_ids == (
        "B_2",
        "A_3",
    )


# =============================================================================
# Memory audit boundary
# =============================================================================


def test_appended_memory_marks_predictive_state_unmutated() -> None:
    memory = append_experience_trace(
        ControllerMemory(),
        _pattern_a_trace(
            1
        ),
    )

    assert (
        memory.metadata[
            "predictive_state_mutated"
        ]
        is False
    )


def test_appended_memory_marks_predictive_preload_not_applied() -> None:
    memory = append_experience_trace(
        ControllerMemory(),
        _pattern_a_trace(
            1
        ),
    )

    assert (
        memory.metadata[
            "predictive_preload_applied"
        ]
        is False
    )


def test_appended_memory_marks_graph_unmutated() -> None:
    memory = append_experience_trace(
        ControllerMemory(),
        _pattern_a_trace(
            1
        ),
    )

    assert (
        memory.metadata[
            "graph_mutated"
        ]
        is False
    )


def test_appended_memory_declares_no_external_label_usage() -> None:
    memory = append_experience_trace(
        ControllerMemory(),
        _pattern_a_trace(
            1
        ),
    )

    assert (
        memory.metadata[
            "external_expected_label_used"
        ]
        is False
    )


def test_appended_memory_remains_preload_free() -> None:
    memory = append_experience_trace(
        ControllerMemory(),
        _pattern_a_trace(
            1
        ),
    )

    assert memory_is_preload_free(
        memory
    ) is True


# =============================================================================
# Query helpers
# =============================================================================


def _mixed_memory() -> ControllerMemory:
    return append_experience_traces(
        ControllerMemory(),
        (
            _pattern_a_trace(
                1
            ),
            _pattern_b_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
            _pattern_a_trace(
                4
            ),
        ),
    )


def test_filter_traces_returns_matching_subset() -> None:
    memory = _mixed_memory()

    result = filter_traces(
        memory,
        lambda trace: (
            trace.identity.trace_id.startswith(
                "A_"
            )
        ),
    )

    assert tuple(
        trace.identity.trace_id
        for trace
        in result
    ) == (
        "A_1",
        "A_3",
        "A_4",
    )


def test_traces_for_pattern_finds_trace_metadata() -> None:
    memory = _mixed_memory()

    traces = traces_for_pattern(
        memory,
        "A",
    )

    assert tuple(
        trace.identity.trace_id
        for trace
        in traces
    ) == (
        "A_1",
        "A_3",
        "A_4",
    )


def test_traces_for_pattern_returns_empty_for_unknown_pattern() -> None:
    memory = _mixed_memory()

    assert traces_for_pattern(
        memory,
        "Z",
    ) == ()


def test_most_recent_trace_returns_last_trace() -> None:
    memory = _mixed_memory()

    trace = most_recent_trace(
        memory
    )

    assert trace is not None
    assert trace.identity.trace_id == "A_4"


def test_most_recent_trace_returns_none_for_empty_memory() -> None:
    assert most_recent_trace(
        ControllerMemory()
    ) is None


# =============================================================================
# MemoryScar basic derivation
# =============================================================================


def test_derive_memory_scar_returns_memory_scar() -> None:
    traces = (
        _pattern_a_trace(
            1
        ),
        _pattern_a_trace(
            2
        ),
    )

    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=traces,
    )

    assert isinstance(
        scar,
        MemoryScar,
    )


def test_memory_scar_default_id_uses_pattern_id() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
        ),
    )

    assert (
        scar.scar_id
        == "memory_scar:A"
    )


def test_memory_scar_custom_id_is_preserved() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
        ),
        scar_id="scar_custom",
    )

    assert scar.scar_id == "scar_custom"


def test_memory_scar_records_trace_ids() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert scar.trace_ids == (
        "A_1",
        "A_2",
        "A_3",
    )


def test_memory_scar_recurrence_count_matches_trace_count() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert scar.recurrence_count == 3


# =============================================================================
# Bias aggregation
# =============================================================================


def test_pattern_a_mean_course_bias_is_exact() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert (
        scar.bias_by_variable[
            "course_error"
        ]
        == pytest.approx(
            0.08
        )
    )


def test_pattern_a_mean_heel_bias_is_exact() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert (
        scar.bias_by_variable[
            "heel_error"
        ]
        == pytest.approx(
            0.04
        )
    )


def test_pattern_a_error_norm_mean_matches_bias_norm() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    expected = sqrt(
        0.08 ** 2
        + 0.04 ** 2
    )

    assert scar.error_norm_mean == pytest.approx(
        expected
    )


def test_pattern_a_error_norm_last_matches_expected() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    expected = sqrt(
        0.08 ** 2
        + 0.04 ** 2
    )

    assert scar.error_norm_last == pytest.approx(
        expected
    )


def test_scar_bias_norm_matches_euclidean_norm() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert scar_bias_norm(
        scar
    ) == pytest.approx(
        sqrt(
            0.08 ** 2
            + 0.04 ** 2
        )
    )


# =============================================================================
# Consistency / confidence
# =============================================================================


def test_identical_repeated_bias_has_maximal_consistency() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert scar.consistency == pytest.approx(
        1.0
    )


def test_confidence_increases_with_recurrence_for_consistent_pattern() -> None:
    one = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
        ),
    )

    three = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert three.confidence > one.confidence


def test_three_consistent_recurrences_have_expected_confidence() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    # recurrence_factor = 3 / (3 + 2) = 0.6
    # consistency = 1.0
    assert scar.confidence == pytest.approx(
        0.6
    )


def test_opposite_bias_reduces_consistency() -> None:
    forward = _trace(
        trace_id="forward",
        episode_index=1,
        pattern_family="X",
        course_bias=0.08,
        heel_bias=0.04,
    )

    reverse = _trace(
        trace_id="reverse",
        episode_index=2,
        pattern_family="X",
        course_bias=-0.08,
        heel_bias=-0.04,
    )

    # Add a third weak forward sample so mean bias is nonzero.
    weak_forward = _trace(
        trace_id="weak_forward",
        episode_index=3,
        pattern_family="X",
        course_bias=0.02,
        heel_bias=0.01,
    )

    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="X"
        ),
        traces=(
            forward,
            reverse,
            weak_forward,
        ),
    )

    assert scar.consistency < 1.0


def test_consistent_repeated_pattern_is_systematic() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert scar_is_systematic(
        scar
    ) is True


def test_single_recurrence_is_not_systematic_by_default() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
        ),
    )

    assert scar_is_systematic(
        scar
    ) is False


# =============================================================================
# Stabilization outcome aggregation
# =============================================================================


def test_scar_outcome_counts_sum_to_recurrence() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
            _pattern_a_trace(
                3
            ),
        ),
    )

    assert (
        scar.improved_count
        + scar.unchanged_count
        + scar.worsened_count
        == scar.recurrence_count
    )


def test_scar_improved_count_matches_source_traces() -> None:
    traces = (
        _pattern_a_trace(
            1
        ),
        _pattern_a_trace(
            2
        ),
        _pattern_a_trace(
            3
        ),
    )

    expected = sum(
        1
        for trace
        in traces
        if (
            trace.stabilization.outcome
            is StabilizationOutcome.IMPROVED
        )
    )

    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=traces,
    )

    assert scar.improved_count == expected


# =============================================================================
# Scar audit boundary
# =============================================================================


def test_scar_metadata_marks_memory_scar_generated() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert (
        scar.metadata[
            "memory_scar_generated"
        ]
        is True
    )


def test_scar_metadata_marks_predictive_state_unmutated() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert (
        scar.metadata[
            "predictive_state_mutated"
        ]
        is False
    )


def test_scar_metadata_marks_predictive_preload_not_applied() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert (
        scar.metadata[
            "predictive_preload_applied"
        ]
        is False
    )


def test_scar_metadata_marks_policy_unmodified() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert (
        scar.metadata[
            "policy_modified"
        ]
        is False
    )


def test_scar_metadata_marks_graph_unmutated() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert (
        scar.metadata[
            "graph_mutated"
        ]
        is False
    )


def test_scar_is_preload_free_returns_true() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert scar_is_preload_free(
        scar
    ) is True


def test_scar_is_preload_free_detects_forbidden_flag() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    altered = replace(
        scar,
        metadata={
            **dict(
                scar.metadata
            ),
            "predictive_preload_applied": True,
        },
    )

    assert scar_is_preload_free(
        altered
    ) is False


# =============================================================================
# Derivation guards
# =============================================================================


def test_derive_memory_scar_rejects_zero_traces() -> None:
    with pytest.raises(
        MemoryScarError
    ):
        derive_memory_scar(
            pattern=MemoryPattern(
                pattern_id="A"
            ),
            traces=(),
        )


def test_memory_scar_constructor_rejects_empty_scar_id() -> None:
    with pytest.raises(
        MemoryScarError
    ):
        MemoryScar(
            scar_id="",
            pattern=MemoryPattern(
                pattern_id="A"
            ),
            trace_ids=(
                "x",
            ),
            bias_by_variable={
                "x": 0.1,
            },
            error_norm_mean=0.1,
            error_norm_last=0.1,
            recurrence_count=1,
            consistency=1.0,
            confidence=0.3,
            improved_count=1,
            unchanged_count=0,
            worsened_count=0,
        )


def test_memory_scar_constructor_rejects_trace_count_mismatch() -> None:
    with pytest.raises(
        MemoryScarError
    ):
        MemoryScar(
            scar_id="scar",
            pattern=MemoryPattern(
                pattern_id="A"
            ),
            trace_ids=(
                "x",
            ),
            bias_by_variable={
                "x": 0.1,
            },
            error_norm_mean=0.1,
            error_norm_last=0.1,
            recurrence_count=2,
            consistency=1.0,
            confidence=0.3,
            improved_count=1,
            unchanged_count=1,
            worsened_count=0,
        )


def test_memory_scar_constructor_rejects_outcome_count_mismatch() -> None:
    with pytest.raises(
        MemoryScarError
    ):
        MemoryScar(
            scar_id="scar",
            pattern=MemoryPattern(
                pattern_id="A"
            ),
            trace_ids=(
                "x",
            ),
            bias_by_variable={
                "x": 0.1,
            },
            error_norm_mean=0.1,
            error_norm_last=0.1,
            recurrence_count=1,
            consistency=1.0,
            confidence=0.3,
            improved_count=0,
            unchanged_count=0,
            worsened_count=0,
        )


# =============================================================================
# Derive from memory
# =============================================================================


def test_derive_memory_scar_from_memory_uses_pattern_matches() -> None:
    memory = _mixed_memory()

    scar = derive_memory_scar_from_memory(
        memory,
        pattern=MemoryPattern(
            pattern_id="A"
        ),
    )

    assert scar.trace_ids == (
        "A_1",
        "A_3",
        "A_4",
    )


def test_derive_memory_scar_from_memory_rejects_unknown_pattern() -> None:
    memory = _mixed_memory()

    with pytest.raises(
        MemoryScarError
    ):
        derive_memory_scar_from_memory(
            memory,
            pattern=MemoryPattern(
                pattern_id="Z"
            ),
        )


# =============================================================================
# Immutability
# =============================================================================


def test_memory_traces_are_tuple() -> None:
    memory = _mixed_memory()

    assert isinstance(
        memory.traces,
        tuple,
    )


def test_memory_scar_trace_ids_are_tuple() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert isinstance(
        scar.trace_ids,
        tuple,
    )


def test_memory_scar_bias_mapping_is_read_only() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    assert isinstance(
        scar.bias_by_variable,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        scar.bias_by_variable[
            "course_error"
        ] = 1.0  # type: ignore[index]


def test_memory_scar_metadata_is_read_only() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    with pytest.raises(TypeError):
        scar.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_memory_scar_is_frozen() -> None:
    scar = derive_memory_scar(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=(
            _pattern_a_trace(
                1
            ),
            _pattern_a_trace(
                2
            ),
        ),
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        scar.confidence = 0.0  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_append_is_deterministic() -> None:
    trace = _pattern_a_trace(
        1
    )

    left = append_experience_trace(
        ControllerMemory(),
        trace,
    )

    right = append_experience_trace(
        ControllerMemory(),
        trace,
    )

    assert left == right


def test_pattern_query_is_deterministic() -> None:
    memory = _mixed_memory()

    left = traces_for_pattern(
        memory,
        "A",
    )

    right = traces_for_pattern(
        memory,
        "A",
    )

    assert left == right


def test_memory_scar_derivation_is_deterministic() -> None:
    traces = (
        _pattern_a_trace(
            1
        ),
        _pattern_a_trace(
            2
        ),
        _pattern_a_trace(
            3
        ),
    )

    kwargs = dict(
        pattern=MemoryPattern(
            pattern_id="A"
        ),
        traces=traces,
    )

    left = derive_memory_scar(
        **kwargs
    )

    right = derive_memory_scar(
        **kwargs
    )

    assert left == right


def test_complete_controller_memory_pipeline_is_deterministic() -> None:
    traces = (
        _pattern_a_trace(
            1
        ),
        _pattern_b_trace(
            2
        ),
        _pattern_a_trace(
            3
        ),
        _pattern_a_trace(
            4
        ),
    )

    left_memory = append_experience_traces(
        ControllerMemory(),
        traces,
    )

    right_memory = append_experience_traces(
        ControllerMemory(),
        traces,
    )

    left_scar = derive_memory_scar_from_memory(
        left_memory,
        pattern=MemoryPattern(
            pattern_id="A"
        ),
    )

    right_scar = derive_memory_scar_from_memory(
        right_memory,
        pattern=MemoryPattern(
            pattern_id="A"
        ),
    )

    assert left_memory == right_memory
    assert left_scar == right_scar
