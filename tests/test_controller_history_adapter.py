"""
Tests for roif.history.controller_history_adapter

The suite fixes the integration contract between the predictive-control
experience layer and the existing ROIF History subsystem.

Pipeline under test:

    ExperienceTrace
        -> HistoryEvent[]
        -> StructuralSignature

Core architectural boundaries:

    ExperienceTrace != HistoryPattern
    PredictionError != IrreversibleChange
    MemoryScar != StructuralSignature
    StructuralSignature != PredictivePreload

The suite validates:
- conversion of one real 04B ExperienceTrace into HistoryEvent objects;
- exact controller-history event ordering;
- causal chaining between adapted events;
- preservation of provenance;
- StructuralSignature construction;
- existing StructuralSignature invariants;
- no rheological or irreversible-change claim;
- no PredictivePreload or learning side effects;
- same-pattern signature proximity;
- cross-pattern signature separation;
- deterministic adaptation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from roif.experience_trace import (
    ExperienceTrace,
)
from roif.history.controller_history_adapter import (
    ControllerHistoryAdapterError,
    ControllerHistoryEventRole,
    ControllerHistoryRecord,
    InvalidControllerHistoryRecordError,
    adapt_experience_trace,
    adapt_experience_traces,
    build_controller_structural_signature,
    controller_signature_distance,
    controller_signatures_compare,
    experience_trace_to_history_events,
    record_is_preload_free,
)
from roif.history.event import (
    HistoryEvent,
)
from roif.history.structural_signature import (
    StructuralSignature,
)

from validation.sailing.sailing_yacht_crew_repeated_event_benchmark import (
    RepeatedEventBenchmarkResult,
    run_repeated_event_benchmark,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def benchmark() -> RepeatedEventBenchmarkResult:
    return run_repeated_event_benchmark()


@pytest.fixture(scope="module")
def trace_a1(
    benchmark: RepeatedEventBenchmarkResult,
) -> ExperienceTrace:
    return benchmark.episodes[0].trace


@pytest.fixture(scope="module")
def trace_b(
    benchmark: RepeatedEventBenchmarkResult,
) -> ExperienceTrace:
    return benchmark.episodes[1].trace


@pytest.fixture(scope="module")
def trace_a2(
    benchmark: RepeatedEventBenchmarkResult,
) -> ExperienceTrace:
    return benchmark.episodes[2].trace


@pytest.fixture(scope="module")
def trace_c(
    benchmark: RepeatedEventBenchmarkResult,
) -> ExperienceTrace:
    return benchmark.episodes[3].trace


@pytest.fixture(scope="module")
def trace_a3(
    benchmark: RepeatedEventBenchmarkResult,
) -> ExperienceTrace:
    return benchmark.episodes[4].trace


@pytest.fixture(scope="module")
def record_a1(
    trace_a1: ExperienceTrace,
) -> ControllerHistoryRecord:
    return adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )


# =============================================================================
# Basic adaptation
# =============================================================================


def test_adapt_experience_trace_returns_record(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert isinstance(
        record_a1,
        ControllerHistoryRecord,
    )


def test_record_preserves_trace_id(
    record_a1: ControllerHistoryRecord,
    trace_a1: ExperienceTrace,
) -> None:
    assert (
        record_a1.trace_id
        == trace_a1.identity.trace_id
    )


def test_record_preserves_sequence_id(
    record_a1: ControllerHistoryRecord,
    trace_a1: ExperienceTrace,
) -> None:
    assert (
        record_a1.sequence_id
        == trace_a1.identity.sequence_id
    )


def test_record_preserves_explicit_pattern_id(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert record_a1.pattern_id == "A"


def test_record_preserves_source_trace(
    record_a1: ControllerHistoryRecord,
    trace_a1: ExperienceTrace,
) -> None:
    assert record_a1.source_trace == trace_a1


def test_record_generates_structural_signature(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert isinstance(
        record_a1.signature,
        StructuralSignature,
    )


# =============================================================================
# HistoryEvent sequence
# =============================================================================


def test_experience_trace_generates_seven_history_events(
    trace_a1: ExperienceTrace,
) -> None:
    events = experience_trace_to_history_events(
        trace_a1
    )

    assert len(
        events
    ) == 7


def test_all_adapted_events_are_history_events(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert all(
        isinstance(
            event,
            HistoryEvent,
        )
        for event
        in record_a1.events
    )


def test_event_roles_are_exactly_ordered(
    record_a1: ControllerHistoryRecord,
) -> None:
    roles = tuple(
        event.metadata[
            "controller_history_role"
        ]
        for event
        in record_a1.events
    )

    assert roles == (
        ControllerHistoryEventRole.INITIAL_STATE.value,
        ControllerHistoryEventRole.DISTURBANCE.value,
        ControllerHistoryEventRole.DECISION.value,
        ControllerHistoryEventRole.PREDICTION.value,
        ControllerHistoryEventRole.OBSERVATION.value,
        ControllerHistoryEventRole.PREDICTION_ERROR.value,
        ControllerHistoryEventRole.STABILIZATION.value,
    )


def test_event_times_are_strictly_increasing(
    record_a1: ControllerHistoryRecord,
) -> None:
    times = tuple(
        event.time
        for event
        in record_a1.events
    )

    assert all(
        left < right
        for left, right
        in zip(
            times,
            times[1:],
        )
    )


def test_all_events_preserve_trace_id(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert all(
        event.metadata[
            "trace_id"
        ]
        == record_a1.trace_id
        for event
        in record_a1.events
    )


def test_all_events_declare_no_learning(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert all(
        event.metadata[
            "learning_applied"
        ]
        is False
        for event
        in record_a1.events
    )


def test_all_events_declare_no_predictive_preload(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert all(
        event.metadata[
            "predictive_preload_applied"
        ]
        is False
        for event
        in record_a1.events
    )


def test_all_events_declare_no_graph_mutation(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert all(
        event.metadata[
            "graph_mutated"
        ]
        is False
        for event
        in record_a1.events
    )


def test_all_events_declare_no_external_expected_label_usage(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert all(
        event.metadata[
            "external_expected_label_used"
        ]
        is False
        for event
        in record_a1.events
    )


# =============================================================================
# Causal event chain
# =============================================================================


def test_disturbance_event_is_caused_by_initial_state(
    record_a1: ControllerHistoryRecord,
) -> None:
    initial, disturbance = record_a1.events[
        0
    ], record_a1.events[
        1
    ]

    assert disturbance.cause_event_ids == (
        initial.event_id,
    )


def test_decision_event_is_caused_by_disturbance(
    record_a1: ControllerHistoryRecord,
) -> None:
    disturbance, decision = record_a1.events[
        1
    ], record_a1.events[
        2
    ]

    assert decision.cause_event_ids == (
        disturbance.event_id,
    )


def test_prediction_event_is_caused_by_decision(
    record_a1: ControllerHistoryRecord,
) -> None:
    decision, prediction = record_a1.events[
        2
    ], record_a1.events[
        3
    ]

    assert prediction.cause_event_ids == (
        decision.event_id,
    )


def test_observation_event_is_caused_by_prediction(
    record_a1: ControllerHistoryRecord,
) -> None:
    prediction, observation = record_a1.events[
        3
    ], record_a1.events[
        4
    ]

    assert observation.cause_event_ids == (
        prediction.event_id,
    )


def test_prediction_error_event_is_caused_by_observation(
    record_a1: ControllerHistoryRecord,
) -> None:
    observation, error = record_a1.events[
        4
    ], record_a1.events[
        5
    ]

    assert error.cause_event_ids == (
        observation.event_id,
    )


def test_stabilization_event_is_caused_by_prediction_error(
    record_a1: ControllerHistoryRecord,
) -> None:
    error, stabilization = record_a1.events[
        5
    ], record_a1.events[
        6
    ]

    assert stabilization.cause_event_ids == (
        error.event_id,
    )


# =============================================================================
# Event payloads
# =============================================================================


def test_prediction_error_event_contains_l2_delta(
    record_a1: ControllerHistoryRecord,
) -> None:
    error_event = record_a1.events[
        5
    ]

    quantities = tuple(
        delta.quantity
        for delta
        in error_event.deltas
    )

    assert (
        "prediction_error_l2"
        in quantities
    )


def test_decision_event_contains_selected_candidate(
    record_a1: ControllerHistoryRecord,
) -> None:
    decision_event = record_a1.events[
        2
    ]

    selected = tuple(
        delta
        for delta
        in decision_event.deltas
        if (
            delta.quantity
            == "selected_candidate_id"
        )
    )

    assert len(
        selected
    ) == 1

    assert (
        selected[
            0
        ].after
        == "coupled_helm_trim_action"
    )


def test_decision_event_contains_stabilization_demand(
    record_a1: ControllerHistoryRecord,
) -> None:
    decision_event = record_a1.events[
        2
    ]

    assert any(
        delta.quantity
        == "stabilization_demand"
        for delta
        in decision_event.deltas
    )


def test_decision_event_contains_available_control_reserve(
    record_a1: ControllerHistoryRecord,
) -> None:
    decision_event = record_a1.events[
        2
    ]

    assert any(
        delta.quantity
        == "control_reserve_available"
        for delta
        in decision_event.deltas
    )


# =============================================================================
# StructuralSignature
# =============================================================================


def test_signature_source_pattern_is_explicit_pattern(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.source_pattern_id
        == "A"
    )


def test_signature_label_marks_controller_experience(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.label
        == "controller_experience"
    )


def test_signature_contains_controller_plane(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        "controller"
        in record_a1.signature.plane_profile
    )


def test_signature_contains_predictive_controller_agent(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        "predictive_controller"
        in record_a1.signature.agent_profile
    )


def test_signature_contains_action_kind(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.kind_profile[
            "action"
        ]
        > 0.0
    )

    assert (
        record_a1.signature.kind_profile[
            "action"
        ]
        >
        record_a1.signature.kind_profile[
            "prediction_error"
        ]
    )


def test_signature_contains_prediction_error_weight(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.kind_profile[
            "prediction_error"
        ]
        > 0.0
    )


def test_signature_has_seven_total_changes(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.total_changes
        == 7
    )


def test_signature_mean_weight_respects_existing_history_invariant(
    record_a1: ControllerHistoryRecord,
) -> None:
    signature = record_a1.signature

    assert (
        signature.mean_signature_weight
        == pytest.approx(
            signature.cumulative_signature_weight
            / signature.total_changes
        )
    )


def test_signature_has_zero_irreversibility_index(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.irreversibility_index
        == pytest.approx(
            0.0
        )
    )


def test_signature_has_zero_adaptation_index_before_learning(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.adaptation_index
        == pytest.approx(
            0.0
        )
    )


def test_signature_has_no_rheology_changes(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.rheology_change_count
        == 0
    )


def test_signature_does_not_claim_rheological_memory(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.metadata[
            "rheological_claim"
        ]
        is False
    )


def test_signature_does_not_claim_irreversible_change(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.metadata[
            "irreversible_change_claim"
        ]
        is False
    )


def test_signature_declares_no_predictive_preload(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.metadata[
            "predictive_preload_applied"
        ]
        is False
    )


def test_signature_declares_no_learning(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.signature.metadata[
            "learning_applied"
        ]
        is False
    )


# =============================================================================
# Record audit boundary
# =============================================================================


def test_record_metadata_marks_adapter(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.metadata[
            "controller_history_adapter"
        ]
        is True
    )


def test_record_metadata_records_event_count(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.metadata[
            "history_event_count"
        ]
        == 7
    )


def test_record_metadata_declares_history_pattern_not_generated(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.metadata[
            "history_pattern_generated"
        ]
        is False
    )


def test_record_metadata_declares_irreversible_change_not_generated(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert (
        record_a1.metadata[
            "irreversible_change_generated"
        ]
        is False
    )


def test_record_is_preload_free_returns_true(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert record_is_preload_free(
        record_a1
    ) is True


def test_record_is_preload_free_detects_forbidden_flag(
    record_a1: ControllerHistoryRecord,
) -> None:
    altered = replace(
        record_a1,
        metadata={
            **dict(
                record_a1.metadata
            ),
            "predictive_preload_applied": True,
        },
    )

    assert record_is_preload_free(
        altered
    ) is False


# =============================================================================
# Pattern-distance validation
# =============================================================================


def test_repeated_a_signatures_are_close(
    trace_a1: ExperienceTrace,
    trace_a2: ExperienceTrace,
) -> None:
    a1 = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )

    a2 = adapt_experience_trace(
        trace_a2,
        pattern_id="A",
    )

    assert controller_signature_distance(
        a1,
        a2,
    ) < 0.02


def test_first_and_third_a_signatures_are_close(
    trace_a1: ExperienceTrace,
    trace_a3: ExperienceTrace,
) -> None:
    a1 = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )

    a3 = adapt_experience_trace(
        trace_a3,
        pattern_id="A",
    )

    assert controller_signature_distance(
        a1,
        a3,
    ) < 0.02


def test_a_to_b_distance_is_large_relative_to_a_to_a(
    trace_a1: ExperienceTrace,
    trace_a2: ExperienceTrace,
    trace_b: ExperienceTrace,
) -> None:
    a1 = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    a2 = adapt_experience_trace(
        trace_a2,
        pattern_id="A",
    )
    b = adapt_experience_trace(
        trace_b,
        pattern_id="B",
    )

    same = controller_signature_distance(
        a1,
        a2,
    )

    cross = controller_signature_distance(
        a1,
        b,
    )

    assert cross > same * 50.0


def test_a_to_c_distance_is_large_relative_to_a_to_a(
    trace_a1: ExperienceTrace,
    trace_a2: ExperienceTrace,
    trace_c: ExperienceTrace,
) -> None:
    a1 = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    a2 = adapt_experience_trace(
        trace_a2,
        pattern_id="A",
    )
    c = adapt_experience_trace(
        trace_c,
        pattern_id="C",
    )

    same = controller_signature_distance(
        a1,
        a2,
    )

    cross = controller_signature_distance(
        a1,
        c,
    )

    assert cross > same * 50.0


def test_b_and_c_are_more_similar_than_a_and_b(
    trace_a1: ExperienceTrace,
    trace_b: ExperienceTrace,
    trace_c: ExperienceTrace,
) -> None:
    a = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    b = adapt_experience_trace(
        trace_b,
        pattern_id="B",
    )
    c = adapt_experience_trace(
        trace_c,
        pattern_id="C",
    )

    assert (
        controller_signature_distance(
            b,
            c,
        )
        <
        controller_signature_distance(
            a,
            b,
        )
    )


def test_exact_04b_a1_a2_distance_regression(
    trace_a1: ExperienceTrace,
    trace_a2: ExperienceTrace,
) -> None:
    a1 = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    a2 = adapt_experience_trace(
        trace_a2,
        pattern_id="A",
    )

    assert controller_signature_distance(
        a1,
        a2,
    ) == pytest.approx(
        0.011038025558232283
    )


def test_exact_04b_a1_a3_distance_regression(
    trace_a1: ExperienceTrace,
    trace_a3: ExperienceTrace,
) -> None:
    a1 = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    a3 = adapt_experience_trace(
        trace_a3,
        pattern_id="A",
    )

    assert controller_signature_distance(
        a1,
        a3,
    ) == pytest.approx(
        0.007433776184188691
    )


def test_exact_04b_a1_b_distance_regression(
    trace_a1: ExperienceTrace,
    trace_b: ExperienceTrace,
) -> None:
    a = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    b = adapt_experience_trace(
        trace_b,
        pattern_id="B",
    )

    assert controller_signature_distance(
        a,
        b,
    ) == pytest.approx(
        1.429569871958611
    )


def test_exact_04b_a1_c_distance_regression(
    trace_a1: ExperienceTrace,
    trace_c: ExperienceTrace,
) -> None:
    a = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    c = adapt_experience_trace(
        trace_c,
        pattern_id="C",
    )

    assert controller_signature_distance(
        a,
        c,
    ) == pytest.approx(
        1.4498262459398883
    )


def test_exact_04b_b_c_distance_regression(
    trace_b: ExperienceTrace,
    trace_c: ExperienceTrace,
) -> None:
    b = adapt_experience_trace(
        trace_b,
        pattern_id="B",
    )
    c = adapt_experience_trace(
        trace_c,
        pattern_id="C",
    )

    assert controller_signature_distance(
        b,
        c,
    ) == pytest.approx(
        0.11736418754754356
    )


# =============================================================================
# Existing StructuralSignature compare API
# =============================================================================


def test_controller_signatures_compare_returns_comparison(
    trace_a1: ExperienceTrace,
    trace_a2: ExperienceTrace,
) -> None:
    a1 = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    a2 = adapt_experience_trace(
        trace_a2,
        pattern_id="A",
    )

    comparison = controller_signatures_compare(
        a1,
        a2,
    )

    assert comparison is not None


def test_structural_signature_compact_vectors_have_equal_length(
    trace_a1: ExperienceTrace,
    trace_b: ExperienceTrace,
) -> None:
    a = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )
    b = adapt_experience_trace(
        trace_b,
        pattern_id="B",
    )

    assert len(
        a.signature.compact_vector()
    ) == len(
        b.signature.compact_vector()
    )


# =============================================================================
# Batch adaptation
# =============================================================================


def test_adapt_experience_traces_returns_tuple(
    benchmark: RepeatedEventBenchmarkResult,
) -> None:
    traces = tuple(
        episode.trace
        for episode
        in benchmark.episodes
    )

    records = adapt_experience_traces(
        traces
    )

    assert isinstance(
        records,
        tuple,
    )


def test_batch_adaptation_preserves_trace_count(
    benchmark: RepeatedEventBenchmarkResult,
) -> None:
    traces = tuple(
        episode.trace
        for episode
        in benchmark.episodes
    )

    records = adapt_experience_traces(
        traces
    )

    assert len(
        records
    ) == len(
        traces
    )


def test_batch_pattern_resolver_assigns_families(
    benchmark: RepeatedEventBenchmarkResult,
) -> None:
    traces = tuple(
        episode.trace
        for episode
        in benchmark.episodes
    )

    family_by_trace_id = {
        episode.trace.identity.trace_id:
            episode.event.metadata[
                "pattern_family"
            ]
        for episode
        in benchmark.episodes
    }

    records = adapt_experience_traces(
        traces,
        pattern_id_resolver=lambda trace: (
            family_by_trace_id[
                trace.identity.trace_id
            ]
        ),
    )

    assert tuple(
        record.pattern_id
        for record
        in records
    ) == (
        "A",
        "B",
        "A",
        "C",
        "A",
    )


# =============================================================================
# Record validation / immutability
# =============================================================================


def test_record_rejects_empty_trace_id(
    record_a1: ControllerHistoryRecord,
) -> None:
    with pytest.raises(
        InvalidControllerHistoryRecordError
    ):
        replace(
            record_a1,
            trace_id="",
        )


def test_record_rejects_empty_event_sequence(
    record_a1: ControllerHistoryRecord,
) -> None:
    with pytest.raises(
        InvalidControllerHistoryRecordError
    ):
        replace(
            record_a1,
            events=(),
        )


def test_record_rejects_source_trace_identity_mismatch(
    record_a1: ControllerHistoryRecord,
    trace_b: ExperienceTrace,
) -> None:
    with pytest.raises(
        InvalidControllerHistoryRecordError
    ):
        replace(
            record_a1,
            source_trace=trace_b,
        )


def test_record_events_are_tuple(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert isinstance(
        record_a1.events,
        tuple,
    )


def test_record_metadata_is_read_only(
    record_a1: ControllerHistoryRecord,
) -> None:
    assert isinstance(
        record_a1.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        record_a1.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_record_is_frozen(
    record_a1: ControllerHistoryRecord,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        record_a1.pattern_id = "B"  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_history_event_sequence_is_deterministic(
    trace_a1: ExperienceTrace,
) -> None:
    left = experience_trace_to_history_events(
        trace_a1
    )

    right = experience_trace_to_history_events(
        trace_a1
    )

    # event_id may be generated internally, so compare stable semantic payload.
    left_payload = tuple(
        (
            event.kind,
            event.time,
            event.target,
            event.deltas,
            event.plane_ids,
            event.agent_ids,
            event.time_scale,
            event.characteristic_time,
            event.persistence,
            event.reversibility,
            event.capacity_effect,
            event.confidence,
            event.description,
            tuple(
                sorted(
                    event.metadata.items()
                )
            ),
        )
        for event
        in left
    )

    right_payload = tuple(
        (
            event.kind,
            event.time,
            event.target,
            event.deltas,
            event.plane_ids,
            event.agent_ids,
            event.time_scale,
            event.characteristic_time,
            event.persistence,
            event.reversibility,
            event.capacity_effect,
            event.confidence,
            event.description,
            tuple(
                sorted(
                    event.metadata.items()
                )
            ),
        )
        for event
        in right
    )

    assert left_payload == right_payload


def test_controller_signature_is_deterministic(
    trace_a1: ExperienceTrace,
) -> None:
    left = build_controller_structural_signature(
        trace_a1,
        source_pattern_id="A",
    )

    right = build_controller_structural_signature(
        trace_a1,
        source_pattern_id="A",
    )

    assert left.compact_vector() == right.compact_vector()
    assert left.source_pattern_id == right.source_pattern_id
    assert left.metadata == right.metadata


def test_complete_adapter_semantics_are_deterministic(
    trace_a1: ExperienceTrace,
) -> None:
    left = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )

    right = adapt_experience_trace(
        trace_a1,
        pattern_id="A",
    )

    assert left.trace_id == right.trace_id
    assert left.sequence_id == right.sequence_id
    assert left.pattern_id == right.pattern_id
    assert (
        left.signature.compact_vector()
        == right.signature.compact_vector()
    )
    assert left.metadata == right.metadata

    assert tuple(
        event.metadata[
            "controller_history_role"
        ]
        for event
        in left.events
    ) == tuple(
        event.metadata[
            "controller_history_role"
        ]
        for event
        in right.events
    )

