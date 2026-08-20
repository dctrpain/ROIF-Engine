from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import roif_predictive_stabilization_benchmark as q7
from experiments import roif_q7_off_nominal_transferability_audit as off_nominal
from experiments import roif_multilayer_temporal_image_trajectory_benchmark as temporal

from roif.history.system_image import (
    revise_system_image,
)


# =============================================================================
# METADATA
# =============================================================================

BENCHMARK_VERSION = (
    "roif_matched_state_history_operator_identifiability_v1"
)

CLAIM_SCOPE = "computational_model_only"

ZERO_TOL = 1e-12


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def l2_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "vector dimensions differ"
        )

    return math.sqrt(
        sum(
            (
                float(a)
                - float(b)
            ) ** 2
            for a, b in zip(
                left,
                right,
            )
        )
    )


# =============================================================================
# CANONICAL HISTORY SEMANTICS
# =============================================================================


def canonical_event_label(
    event_id: str,
) -> str:
    """
    Map technical run-specific event IDs to semantic event identities.

    The purpose is to distinguish:

        A -> D

    from:

        D -> A

    without treating technical labels such as
    'audit_exact_nominal_E5' versus 'audit_second_event_A'
    as different physical/event semantics.
    """

    mapping = {
        "audit_exact_nominal_E5": "A",
        "audit_second_event_A": "A",

        "audit_first_event_D": "D",
        "audit_second_event_D": "D",
    }

    return mapping.get(
        event_id,
        event_id,
    )


def canonical_history_order(
    image,
) -> tuple[str, ...]:
    return tuple(
        canonical_event_label(
            trace.source_event_id
        )
        for trace
        in image.historical_traces
    )


def trace_semantic_signature(
    image,
) -> tuple[
    tuple[
        str,
        str,
        float,
        float,
        int,
    ],
    ...,
]:
    """
    Ordered history representation using canonical event semantics.

    trace_id is intentionally excluded because it contains execution-specific
    labels rather than historical meaning.
    """

    return tuple(
        (
            canonical_event_label(
                trace.source_event_id
            ),
            trace.trace_type,
            float(trace.magnitude),
            float(trace.persistence),
            int(trace.relation_count),
        )
        for trace
        in image.historical_traces
    )


# =============================================================================
# CURRENT-STATE SIGNATURE
# =============================================================================


def current_state_signature(
    image,
) -> tuple[Any, ...]:
    """
    Signature of current SystemImage layers EXCLUDING historical_traces.

    This is deliberate.

    The benchmark asks whether two different ordered histories change the next
    transition when the represented current state is explicitly matched.

    image_id, revision, summary, metadata, and historical_traces are therefore
    excluded from the matching signature.

    Included current layers:
        - heterogeneous measures, independently represented;
        - prestress node state;
        - adaptive connection topology and state;
        - context.
    """

    measures = tuple(
        sorted(
            (
                measure.measure_id,
                measure.measure_type,
                float(measure.value),
                measure.unit,
                measure.domain,
                (
                    None
                    if measure.normalized_value
                    is None
                    else float(
                        measure.normalized_value
                    )
                ),
            )
            for measure in image.measures
        )
    )

    prestress = tuple(
        sorted(
            (
                node.node_id,
                float(node.prestress),
                float(node.reserve),
                float(node.min_prestress),
                float(node.max_prestress),
            )
            for node
            in image.prestress_nodes
        )
    )

    connections = tuple(
        sorted(
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
            in image.adaptive_connections
        )
    )

    context = (
        image.context.context_id,
        image.context.timestamp_label,
    )

    return (
        measures,
        prestress,
        connections,
        context,
    )


# =============================================================================
# RESPONSE SIGNATURE
# =============================================================================


def transition_response_signature(
    source,
    target,
) -> dict[str, Any]:
    """
    Describe the transition in current-state layers.

    Historical traces are kept separate from the response signature so that an
    inherited difference in H cannot by itself be misreported as a different
    transition response.
    """

    source_nodes = {
        node.node_id: node
        for node
        in source.prestress_nodes
    }

    target_nodes = {
        node.node_id: node
        for node
        in target.prestress_nodes
    }

    node_ids = tuple(
        sorted(
            source_nodes
        )
    )

    prestress_delta = tuple(
        (
            node_id,
            float(
                target_nodes[
                    node_id
                ].prestress
                -
                source_nodes[
                    node_id
                ].prestress
            ),
        )
        for node_id in node_ids
    )

    source_connections = {
        connection.connection_id:
        connection
        for connection
        in source.adaptive_connections
    }

    target_connections = {
        connection.connection_id:
        connection
        for connection
        in target.adaptive_connections
    }

    connection_ids = tuple(
        sorted(
            source_connections
        )
    )

    connection_delta = tuple(
        (
            connection_id,

            float(
                target_connections[
                    connection_id
                ].stiffness
                -
                source_connections[
                    connection_id
                ].stiffness
            ),

            float(
                target_connections[
                    connection_id
                ].contractile_capacity
                -
                source_connections[
                    connection_id
                ].contractile_capacity
            ),

            float(
                target_connections[
                    connection_id
                ].reflex_gain
                -
                source_connections[
                    connection_id
                ].reflex_gain
            ),

            float(
                target_connections[
                    connection_id
                ].fatigue
                -
                source_connections[
                    connection_id
                ].fatigue
            ),

            float(
                target_connections[
                    connection_id
                ].remodeling_bias
                -
                source_connections[
                    connection_id
                ].remodeling_bias
            ),
        )
        for connection_id
        in connection_ids
    )

    newest_trace = (
        target.historical_traces[-1]
        if target.historical_traces
        else None
    )

    generated_trace_semantics = (
        None
        if newest_trace is None
        else {
            "source_event": (
                canonical_event_label(
                    newest_trace.source_event_id
                )
            ),
            "trace_type": (
                newest_trace.trace_type
            ),
            "magnitude": float(
                newest_trace.magnitude
            ),
            "persistence": float(
                newest_trace.persistence
            ),
            "relation_count": int(
                newest_trace.relation_count
            ),
        }
    )

    return {
        "prestress_delta": [
            [
                node_id,
                value,
            ]
            for node_id, value
            in prestress_delta
        ],

        "connection_delta": [
            list(item)
            for item
            in connection_delta
        ],

        "generated_trace_semantics": (
            generated_trace_semantics
        ),

        "target_current_state_signature": (
            current_state_signature(
                target
            )
        ),
    }


# =============================================================================
# BUILD NATURALLY CONVERGED DIFFERENT-HISTORY SOURCES
# =============================================================================


def build_history_sources():
    """
    Reproduce the two realised futures:

        A -> D
        D -> A

    from the previous multilayer benchmark.

    The prior experiment showed that these branches converge in current
    structural layers while retaining different ordered histories.
    """

    pre_event_image = (
        q7.build_history_conditioned_pre_event_image()
    )

    scenarios = (
        off_nominal.build_future_scenarios()
    )

    trajectory_A_D = (
        temporal.evolve_multilayer_trajectory(
            pre_event_image=(
                pre_event_image
            ),
            events=scenarios[
                "sequence_A_then_D"
            ],
            action=None,
            label=(
                "operator_source_A_then_D"
            ),
        )
    )

    trajectory_D_A = (
        temporal.evolve_multilayer_trajectory(
            pre_event_image=(
                pre_event_image
            ),
            events=scenarios[
                "sequence_D_then_A"
            ],
            action=None,
            label=(
                "operator_source_D_then_A"
            ),
        )
    )

    return (
        trajectory_A_D[
            "images"
        ][-1],
        trajectory_D_A[
            "images"
        ][-1],
    )


# =============================================================================
# EXPLICIT MATCHED-CURRENT-STATE CONSTRUCTION
# =============================================================================


def build_matched_state_different_history_pair():
    """
    Build two source SystemImages that have EXACTLY the same represented current
    layers but retain different ordered historical traces.

    A single common current-state template is used for both branches.

    Therefore any subsequent difference in transition response cannot be
    attributed to:
        - measures,
        - prestress,
        - reserve,
        - adaptive connection state,
        - adaptive connection topology,
        - context.

    The deliberately unmatched layer is historical_traces.
    """

    natural_A_D, natural_D_A = (
        build_history_sources()
    )

    history_A_D = (
        natural_A_D.historical_traces
    )

    history_D_A = (
        natural_D_A.historical_traces
    )

    # Use exactly the same current-state template
    # for both matched branches.
    template = natural_A_D

    matched_A_D = revise_system_image(
        source=template,
        target_image_id=(
            "matched_state_history_A_then_D"
        ),
        historical_traces=(
            history_A_D
        ),
        metadata={
            "benchmark": (
                BENCHMARK_VERSION
            ),
            "matched_current_state": True,
            "history_branch": (
                "A_then_D"
            ),
        },
    )

    matched_D_A = revise_system_image(
        source=template,
        target_image_id=(
            "matched_state_history_D_then_A"
        ),
        historical_traces=(
            history_D_A
        ),
        metadata={
            "benchmark": (
                BENCHMARK_VERSION
            ),
            "matched_current_state": True,
            "history_branch": (
                "D_then_A"
            ),
        },
    )

    return (
        matched_A_D,
        matched_D_A,
        natural_A_D,
        natural_D_A,
    )


# =============================================================================
# COMMON PROBE EVENT C
# =============================================================================


def build_common_probe_event():
    """
    Common event C applied to both matched states.

    The probe uses the existing nominal E5 mechanics but receives a new event
    identity so it is clearly separated from the histories that generated the
    matched source states.
    """

    nominal = (
        q7.build_events()[4]
    )

    return replace(
        nominal,
        event_id=(
            "matched_history_operator_probe_C"
        ),
    )


# =============================================================================
# SEMANTIC RESPONSE COMPARISON
# =============================================================================


def response_numeric_vector(
    response: Mapping[str, Any],
) -> tuple[float, ...]:

    values = []

    for _, value in response[
        "prestress_delta"
    ]:
        values.append(
            float(value)
        )

    for item in response[
        "connection_delta"
    ]:
        values.extend(
            float(value)
            for value
            in item[1:]
        )

    return tuple(
        values
    )


def compare_transition_responses(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:

    left_vector = (
        response_numeric_vector(
            left
        )
    )

    right_vector = (
        response_numeric_vector(
            right
        )
    )

    numeric_distance = (
        l2_distance(
            left_vector,
            right_vector,
        )
    )

    generated_trace_equal = (
        left[
            "generated_trace_semantics"
        ]
        ==
        right[
            "generated_trace_semantics"
        ]
    )

    target_current_state_equal = (
        left[
            "target_current_state_signature"
        ]
        ==
        right[
            "target_current_state_signature"
        ]
    )

    return {
        "numeric_response_distance": (
            numeric_distance
        ),

        "numeric_response_differs": (
            numeric_distance
            > ZERO_TOL
        ),

        "generated_trace_semantics_equal": (
            generated_trace_equal
        ),

        "target_current_state_equal": (
            target_current_state_equal
        ),

        "history_specific_transition_effect_detected": (
            numeric_distance
            > ZERO_TOL
            or not target_current_state_equal
        ),
    }


# =============================================================================
# FULL BENCHMARK
# =============================================================================


def run_benchmark() -> dict[str, Any]:
    """
    Matched-current-state / different-history operator-identifiability test.

    Architecture-level question
    ---------------------------

        Given:

            current_state_1 == current_state_2

        but:

            H_1 != H_2

        does the same subsequent event C produce:

            R_C(current_state, H_1)
                !=
            R_C(current_state, H_2) ?

    This is the experimental form required to separate:

        state-mediated historical effects

    from:

        an independently history-conditioned transition effect.

    IMPORTANT
    ---------

    A null result does NOT establish that history cannot condition transitions
    in the ROIF architecture.

    It establishes only that the tested implementation did not exhibit an
    additional transition difference once represented current-state layers were
    explicitly matched.
    """

    (
        source_A_D,
        source_D_A,
        natural_A_D,
        natural_D_A,
    ) = (
        build_matched_state_different_history_pair()
    )

    source_state_equal = (
        current_state_signature(
            source_A_D
        )
        ==
        current_state_signature(
            source_D_A
        )
    )

    history_A = (
        trace_semantic_signature(
            source_A_D
        )
    )

    history_D = (
        trace_semantic_signature(
            source_D_A
        )
    )

    histories_differ = (
        history_A
        != history_D
    )

    canonical_order_A = (
        canonical_history_order(
            source_A_D
        )
    )

    canonical_order_D = (
        canonical_history_order(
            source_D_A
        )
    )

    canonical_orders_differ = (
        canonical_order_A
        != canonical_order_D
    )

    probe = (
        build_common_probe_event()
    )

    post_A_D = q7.apply_event(
        source_A_D,
        probe,
        run_id=(
            "matched_operator::"
            "history_A_then_D"
        ),
    )

    post_D_A = q7.apply_event(
        source_D_A,
        probe,
        run_id=(
            "matched_operator::"
            "history_D_then_A"
        ),
    )

    response_A_D = (
        transition_response_signature(
            source_A_D,
            post_A_D,
        )
    )

    response_D_A = (
        transition_response_signature(
            source_D_A,
            post_D_A,
        )
    )

    response_comparison = (
        compare_transition_responses(
            response_A_D,
            response_D_A,
        )
    )

    post_histories_differ = (
        trace_semantic_signature(
            post_A_D
        )
        !=
        trace_semantic_signature(
            post_D_A
        )
    )

    natural_current_state_distance = {
        "prestress": l2_distance(
            q7.prestress_vector(
                natural_A_D
            ),
            q7.prestress_vector(
                natural_D_A
            ),
        ),
    }

    operator_effect_detected = (
        response_comparison[
            "history_specific_transition_effect_detected"
        ]
    )

    return {
        "benchmark_version": (
            BENCHMARK_VERSION
        ),

        "claim_scope": (
            CLAIM_SCOPE
        ),

        "experimental_question": (
            "same_current_state_different_history_same_event"
        ),

        "common_probe_event_id": (
            probe.event_id
        ),

        "source_control": {
            "current_state_matched": (
                source_state_equal
            ),

            "ordered_histories_differ": (
                histories_differ
            ),

            "canonical_event_orders_differ": (
                canonical_orders_differ
            ),

            "history_A_then_D": list(
                canonical_order_A
            ),

            "history_D_then_A": list(
                canonical_order_D
            ),

            "natural_pre_matching_prestress_distance": (
                natural_current_state_distance[
                    "prestress"
                ]
            ),
        },

        "responses": {
            "history_A_then_D": {
                "prestress_delta": (
                    response_A_D[
                        "prestress_delta"
                    ]
                ),

                "connection_delta": (
                    response_A_D[
                        "connection_delta"
                    ]
                ),

                "generated_trace_semantics": (
                    response_A_D[
                        "generated_trace_semantics"
                    ]
                ),
            },

            "history_D_then_A": {
                "prestress_delta": (
                    response_D_A[
                        "prestress_delta"
                    ]
                ),

                "connection_delta": (
                    response_D_A[
                        "connection_delta"
                    ]
                ),

                "generated_trace_semantics": (
                    response_D_A[
                        "generated_trace_semantics"
                    ]
                ),
            },
        },

        "response_comparison": (
            response_comparison
        ),

        "central_results": {
            "matched_current_state_established": (
                source_state_equal
            ),

            "different_ordered_history_established": (
                histories_differ
            ),

            "different_canonical_event_order_established": (
                canonical_orders_differ
            ),

            "same_probe_applied": True,

            "history_specific_transition_effect_detected": (
                operator_effect_detected
            ),

            "post_probe_current_state_equal": (
                response_comparison[
                    "target_current_state_equal"
                ]
            ),

            "transition_response_distance": (
                response_comparison[
                    "numeric_response_distance"
                ]
            ),

            "post_probe_histories_remain_different": (
                post_histories_differ
            ),
        },

        # =====================================================================
        # CLAIM BOUNDARIES
        # =====================================================================

        "matched_state_operator_identifiability_test_performed": True,

        "state_mediated_history_confound_removed": True,

        "history_conditioned_operator_effect_claimed": (
            operator_effect_detected
        ),

        "absence_of_architectural_history_effect_claimed": False,

        "full_predictive_roif_tested": False,

        "future_temporal_image_prediction_tested": False,

        "predictive_preconfiguration_tested": False,

        "biological_history_mechanism_tested": False,

        "external_validity_tested": False,

        "interpretation": (
            (
                "The matched-current-state control detected a difference in "
                "the subsequent transition under different ordered histories. "
                "Within this computational implementation, this is evidence "
                "for a history-specific transition contribution beyond the "
                "explicitly matched current-state layers."
            )
            if operator_effect_detected
            else
            (
                "The two source SystemImages had explicitly matched current "
                "measures, prestress, reserve, adaptive connection topology "
                "and state, and context, while retaining different ordered "
                "historical traces. Applying the same probe event produced "
                "the same represented current-state transition within numeric "
                "tolerance. Therefore the tested implementation does not yet "
                "show an independently history-conditioned transition effect "
                "beyond state-mediated historical reconfiguration. This null "
                "result is a limitation/property of the current implementation "
                "and does not reject the broader ROIF architectural concept "
                "of a history-conditioned effective transition operator."
            )
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
        / "matched_state_history_operator_identifiability_v1.json"
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
        "Saved matched-state history/operator benchmark:"
    )
    print(
        output_path.resolve()
    )


if __name__ == "__main__":
    main()
