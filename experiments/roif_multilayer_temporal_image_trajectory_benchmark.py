from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import roif_predictive_stabilization_benchmark as q7
from experiments import roif_q7_off_nominal_transferability_audit as off_nominal


# =============================================================================
# METADATA
# =============================================================================

BENCHMARK_VERSION = (
    "roif_multilayer_temporal_image_trajectory_v1"
)

CLAIM_SCOPE = "computational_model_only"

BENCHMARK_PURPOSE = (
    "multilayer_systemimage_trajectory_comparison"
)


# =============================================================================
# BASIC HELPERS
# =============================================================================


def l2_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "layer vectors must have equal dimensions"
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


def sorted_prestress_nodes(image):
    return tuple(
        sorted(
            image.prestress_nodes,
            key=lambda item: item.node_id,
        )
    )


def sorted_connections(image):
    return tuple(
        sorted(
            image.adaptive_connections,
            key=lambda item: item.connection_id,
        )
    )


def sorted_measures(image):
    return tuple(
        sorted(
            image.measures,
            key=lambda item: item.measure_id,
        )
    )


# =============================================================================
# SYSTEMIMAGE LAYER SNAPSHOT
# =============================================================================


def system_image_layer_snapshot(
    image,
) -> dict[str, Any]:
    """
    Represent one SystemImage without collapsing heterogeneous layers.

    IMPORTANT
    ---------

    This function intentionally does NOT concatenate every field into one
    Euclidean vector.

    Prestress, reserve, adaptive-connection variables, normalized measures,
    traces, and context remain distinguishable.
    """

    measures = {}

    for measure in sorted_measures(image):
        measures[measure.measure_id] = {
            "measure_type": measure.measure_type,
            "unit": measure.unit,
            "raw_value": float(
                measure.value
            ),
            "normalized_value": (
                None
                if measure.normalized_value is None
                else float(
                    measure.normalized_value
                )
            ),
        }

    prestress = {}

    for node in sorted_prestress_nodes(image):
        prestress[node.node_id] = {
            "prestress": float(
                node.prestress
            ),
            "reserve": float(
                node.reserve
            ),
            "min_prestress": float(
                node.min_prestress
            ),
            "max_prestress": float(
                node.max_prestress
            ),
        }

    adaptive = {}

    for connection in sorted_connections(image):
        adaptive[
            connection.connection_id
        ] = {
            "source_node_id": (
                connection.source_node_id
            ),
            "target_node_id": (
                connection.target_node_id
            ),
            "stiffness": float(
                connection.stiffness
            ),
            "contractile_capacity": float(
                connection.contractile_capacity
            ),
            "reflex_gain": float(
                connection.reflex_gain
            ),
            "fatigue": float(
                connection.fatigue
            ),
            "remodeling_bias": float(
                connection.remodeling_bias
            ),
        }

    traces = [
        {
            "trace_id": trace.trace_id,
            "source_event_id": (
                trace.source_event_id
            ),
            "trace_type": trace.trace_type,
            "magnitude": float(
                trace.magnitude
            ),
            "persistence": float(
                trace.persistence
            ),
            "relation_count": int(
                trace.relation_count
            ),
        }
        for trace in image.historical_traces
    ]

    trace_magnitudes = tuple(
        float(trace.magnitude)
        for trace
        in image.historical_traces
    )

    return {
        "image_id": image.image_id,
        "revision": int(
            image.revision
        ),

        "measures": measures,

        "prestress": prestress,

        "adaptive_connections": adaptive,

        "historical_traces": traces,

        "trace_summary": {
            "count": len(
                image.historical_traces
            ),
            "magnitude_sum": sum(
                trace_magnitudes
            ),
            "magnitude_l2": math.sqrt(
                sum(
                    value * value
                    for value
                    in trace_magnitudes
                )
            ),
            "newest_trace_magnitude": (
                trace_magnitudes[-1]
                if trace_magnitudes
                else 0.0
            ),
        },

        "context": {
            "context_id": (
                image.context.context_id
            ),
            "timestamp_label": (
                image.context.timestamp_label
            ),
        },
    }


# =============================================================================
# LAYER-SPECIFIC COMPARISON
# =============================================================================


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
    Ordered semantic representation of HistoricalTrace.

    trace_id is intentionally excluded because it contains execution/run labels.
    The historical meaning is represented by event identity, trace type,
    magnitude, persistence, and relation count.
    """

    return tuple(
        (
            trace.source_event_id,
            trace.trace_type,
            float(trace.magnitude),
            float(trace.persistence),
            int(trace.relation_count),
        )
        for trace in image.historical_traces
    )


def trace_event_order(
    image,
) -> tuple[str, ...]:
    """
    Explicit ordered event history encoded by HistoricalTrace.
    """

    return tuple(
        trace.source_event_id
        for trace in image.historical_traces
    )


def connection_topology_signature(
    image,
) -> tuple[
    tuple[
        str,
        str,
        str,
    ],
    ...,
]:
    return tuple(
        (
            connection.connection_id,
            connection.source_node_id,
            connection.target_node_id,
        )
        for connection in sorted_connections(
            image
        )
    )


def measure_identity_signature(
    image,
) -> tuple[str, ...]:
    return tuple(
        measure.measure_id
        for measure in sorted_measures(
            image
        )
    )


def measure_semantic_comparison(
    left,
    right,
) -> dict[str, Any]:
    """
    Compare heterogeneous measures individually.

    Raw values from incompatible physical units are never pooled into a single
    distance. Each measure is compared only with the same measure_id.
    """

    left_by_id = {
        measure.measure_id: measure
        for measure in sorted_measures(
            left
        )
    }

    right_by_id = {
        measure.measure_id: measure
        for measure in sorted_measures(
            right
        )
    }

    left_ids = set(
        left_by_id
    )

    right_ids = set(
        right_by_id
    )

    shared_ids = tuple(
        sorted(
            left_ids
            & right_ids
        )
    )

    only_left = tuple(
        sorted(
            left_ids
            - right_ids
        )
    )

    only_right = tuple(
        sorted(
            right_ids
            - left_ids
        )
    )

    per_measure = {}

    any_raw_difference = False
    any_normalized_difference = False
    any_metadata_difference = False

    for measure_id in shared_ids:
        left_measure = (
            left_by_id[
                measure_id
            ]
        )

        right_measure = (
            right_by_id[
                measure_id
            ]
        )

        raw_difference = (
            float(
                left_measure.value
            )
            != float(
                right_measure.value
            )
        )

        left_normalized = (
            left_measure.normalized_value
        )

        right_normalized = (
            right_measure.normalized_value
        )

        normalized_difference = (
            left_normalized
            != right_normalized
        )

        metadata_difference = (
            left_measure.measure_type
            != right_measure.measure_type
            or left_measure.unit
            != right_measure.unit
            or left_measure.domain
            != right_measure.domain
        )

        any_raw_difference = (
            any_raw_difference
            or raw_difference
        )

        any_normalized_difference = (
            any_normalized_difference
            or normalized_difference
        )

        any_metadata_difference = (
            any_metadata_difference
            or metadata_difference
        )

        per_measure[
            measure_id
        ] = {
            "left_raw_value": float(
                left_measure.value
            ),
            "right_raw_value": float(
                right_measure.value
            ),
            "left_normalized_value": (
                None
                if left_normalized is None
                else float(
                    left_normalized
                )
            ),
            "right_normalized_value": (
                None
                if right_normalized is None
                else float(
                    right_normalized
                )
            ),
            "raw_value_differs": (
                raw_difference
            ),
            "normalized_value_differs": (
                normalized_difference
            ),
            "metadata_differs": (
                metadata_difference
            ),
        }

    return {
        "shared_measure_ids": (
            list(shared_ids)
        ),

        "measure_ids_only_left": (
            list(only_left)
        ),

        "measure_ids_only_right": (
            list(only_right)
        ),

        "measure_identity_set_differs": (
            bool(
                only_left
                or only_right
            )
        ),

        "raw_measure_values_differ": (
            any_raw_difference
        ),

        "normalized_measure_values_differ": (
            any_normalized_difference
        ),

        "measure_metadata_differs": (
            any_metadata_difference
        ),

        "per_measure": (
            per_measure
        ),
    }


def compare_system_images(
    left,
    right,
) -> dict[str, Any]:
    """
    Compare two complete SystemImages layer by layer.

    Architectural invariants
    ------------------------

    1. Heterogeneous layers are NOT concatenated into one Euclidean vector.

    2. HistoricalTrace is treated as an ordered sequence, not reduced to trace
       count or magnitude summaries.

    3. Heterogeneous raw SystemMeasure values are compared individually by
       measure identity and are never summed across incompatible units.

    4. Adaptive connection topology is distinguished from adaptive connection
       state.

    Therefore this comparator asks whether corresponding SystemImages differ
    and identifies the layer(s) responsible, without asserting one universal
    whole-image scalar distance.
    """

    # =========================================================================
    # PRESTRESS / RESERVE
    # =========================================================================

    left_nodes = sorted_prestress_nodes(
        left
    )

    right_nodes = sorted_prestress_nodes(
        right
    )

    left_node_ids = tuple(
        node.node_id
        for node in left_nodes
    )

    right_node_ids = tuple(
        node.node_id
        for node in right_nodes
    )

    prestress_node_identity_differs = (
        left_node_ids
        != right_node_ids
    )

    if prestress_node_identity_differs:
        prestress_distance = None
        reserve_distance = None
    else:
        prestress_distance = (
            l2_distance(
                tuple(
                    node.prestress
                    for node
                    in left_nodes
                ),
                tuple(
                    node.prestress
                    for node
                    in right_nodes
                ),
            )
        )

        reserve_distance = (
            l2_distance(
                tuple(
                    node.reserve
                    for node
                    in left_nodes
                ),
                tuple(
                    node.reserve
                    for node
                    in right_nodes
                ),
            )
        )

    # =========================================================================
    # ADAPTIVE CONNECTION TOPOLOGY + STATE
    # =========================================================================

    left_topology = (
        connection_topology_signature(
            left
        )
    )

    right_topology = (
        connection_topology_signature(
            right
        )
    )

    connection_topology_differs = (
        left_topology
        != right_topology
    )

    left_connections = (
        sorted_connections(
            left
        )
    )

    right_connections = (
        sorted_connections(
            right
        )
    )

    left_connection_ids = tuple(
        connection.connection_id
        for connection
        in left_connections
    )

    right_connection_ids = tuple(
        connection.connection_id
        for connection
        in right_connections
    )

    adaptive_connection_identity_differs = (
        left_connection_ids
        != right_connection_ids
    )

    adaptive_distances: dict[
        str,
        float | None,
    ] = {}

    if adaptive_connection_identity_differs:
        for field_name in (
            "stiffness",
            "contractile_capacity",
            "reflex_gain",
            "fatigue",
            "remodeling_bias",
        ):
            adaptive_distances[
                field_name
            ] = None

        adaptive_connection_state_differs = True

    else:
        for field_name in (
            "stiffness",
            "contractile_capacity",
            "reflex_gain",
            "fatigue",
            "remodeling_bias",
        ):
            adaptive_distances[
                field_name
            ] = l2_distance(
                tuple(
                    getattr(
                        connection,
                        field_name,
                    )
                    for connection
                    in left_connections
                ),
                tuple(
                    getattr(
                        connection,
                        field_name,
                    )
                    for connection
                    in right_connections
                ),
            )

        adaptive_connection_state_differs = any(
            (
                value is not None
                and value > 1e-12
            )
            for value
            in adaptive_distances.values()
        )

    # =========================================================================
    # HETEROGENEOUS MEASURES
    # =========================================================================

    measures = (
        measure_semantic_comparison(
            left,
            right,
        )
    )

    # =========================================================================
    # ORDERED HISTORICAL STATE
    # =========================================================================

    left_trace_signature = (
        trace_semantic_signature(
            left
        )
    )

    right_trace_signature = (
        trace_semantic_signature(
            right
        )
    )

    left_event_order = (
        trace_event_order(
            left
        )
    )

    right_event_order = (
        trace_event_order(
            right
        )
    )

    trace_sequence_differs = (
        left_trace_signature
        != right_trace_signature
    )

    trace_event_order_differs = (
        left_event_order
        != right_event_order
    )

    left_trace_magnitudes = tuple(
        float(
            trace.magnitude
        )
        for trace
        in left.historical_traces
    )

    right_trace_magnitudes = tuple(
        float(
            trace.magnitude
        )
        for trace
        in right.historical_traces
    )

    trace_magnitude_sequence_differs = (
        left_trace_magnitudes
        != right_trace_magnitudes
    )

    trace_count_difference = (
        len(
            left_trace_signature
        )
        - len(
            right_trace_signature
        )
    )

    trace_magnitude_sum_difference = (
        sum(
            left_trace_magnitudes
        )
        - sum(
            right_trace_magnitudes
        )
    )

    newest_left = (
        left_trace_magnitudes[-1]
        if left_trace_magnitudes
        else 0.0
    )

    newest_right = (
        right_trace_magnitudes[-1]
        if right_trace_magnitudes
        else 0.0
    )

    newest_trace_magnitude_difference = (
        newest_left
        - newest_right
    )

    # =========================================================================
    # CONTEXT
    # =========================================================================

    context_differs = (
        left.context.context_id
        != right.context.context_id
        or
        left.context.timestamp_label
        != right.context.timestamp_label
    )

    # =========================================================================
    # LAYER FLAGS
    # =========================================================================

    layer_difference_flags = {
        "measure_identity_set_differs": (
            measures[
                "measure_identity_set_differs"
            ]
        ),

        "raw_measure_values_differ": (
            measures[
                "raw_measure_values_differ"
            ]
        ),

        "normalized_measure_values_differ": (
            measures[
                "normalized_measure_values_differ"
            ]
        ),

        "measure_metadata_differs": (
            measures[
                "measure_metadata_differs"
            ]
        ),

        "prestress_node_identity_differs": (
            prestress_node_identity_differs
        ),

        "prestress_differs": (
            (
                prestress_distance is None
            )
            or (
                prestress_distance
                > 1e-12
            )
        ),

        "reserve_differs": (
            (
                reserve_distance is None
            )
            or (
                reserve_distance
                > 1e-12
            )
        ),

        "adaptive_connection_identity_differs": (
            adaptive_connection_identity_differs
        ),

        "connection_topology_differs": (
            connection_topology_differs
        ),

        "adaptive_connection_state_differs": (
            adaptive_connection_state_differs
        ),

        "trace_sequence_differs": (
            trace_sequence_differs
        ),

        "trace_event_order_differs": (
            trace_event_order_differs
        ),

        "trace_magnitude_sequence_differs": (
            trace_magnitude_sequence_differs
        ),

        "context_differs": (
            context_differs
        ),
    }

    any_layer_differs = any(
        layer_difference_flags.values()
    )

    return {
        # ---------------------------------------------------------------------
        # MEASURES
        # ---------------------------------------------------------------------

        "measure_comparison": (
            measures
        ),

        # ---------------------------------------------------------------------
        # PRESTRESS / RESERVE
        # ---------------------------------------------------------------------

        "prestress_node_ids_left": list(
            left_node_ids
        ),

        "prestress_node_ids_right": list(
            right_node_ids
        ),

        "prestress_distance": (
            prestress_distance
        ),

        "reserve_distance": (
            reserve_distance
        ),

        # ---------------------------------------------------------------------
        # CONNECTIONS
        # ---------------------------------------------------------------------

        "connection_topology_left": [
            list(item)
            for item
            in left_topology
        ],

        "connection_topology_right": [
            list(item)
            for item
            in right_topology
        ],

        "adaptive_connection_distances": (
            adaptive_distances
        ),

        # ---------------------------------------------------------------------
        # ORDERED HISTORY
        # ---------------------------------------------------------------------

        "trace_event_order_left": list(
            left_event_order
        ),

        "trace_event_order_right": list(
            right_event_order
        ),

        "trace_semantic_sequence_left": [
            list(item)
            for item
            in left_trace_signature
        ],

        "trace_semantic_sequence_right": [
            list(item)
            for item
            in right_trace_signature
        ],

        # Summaries retained only as diagnostics.
        "trace_count_difference": (
            trace_count_difference
        ),

        "trace_magnitude_sum_difference": (
            trace_magnitude_sum_difference
        ),

        "newest_trace_magnitude_difference": (
            newest_trace_magnitude_difference
        ),

        # ---------------------------------------------------------------------
        # CONTEXT
        # ---------------------------------------------------------------------

        "context_equal": (
            not context_differs
        ),

        # ---------------------------------------------------------------------
        # DIFFERENCE FLAGS
        # ---------------------------------------------------------------------

        "layer_difference_flags": (
            layer_difference_flags
        ),

        "any_layer_differs": (
            any_layer_differs
        ),
    }

# =============================================================================
# TRAJECTORY EXECUTION
# =============================================================================


def evolve_multilayer_trajectory(
    *,
    pre_event_image,
    events,
    action,
    label: str,
) -> dict[str, Any]:
    """
    Produce an ordered sequence of complete SystemImages.

    The returned trajectory is the actual sequence:

        I_0, I_1, ..., I_k

    generated by the Engine.

    This function does not collapse that sequence into a single scalar.
    """

    if action is None:
        current = pre_event_image
    else:
        current = (
            q7.apply_action_to_system_image(
                pre_event_image,
                action,
                image_id=(
                    f"{label}::prepared"
                ),
                policy_name=(
                    f"multilayer_audit::{label}"
                ),
            )
        )

    initial_image = current

    images = [
        initial_image
    ]

    event_ids = []

    for index, event in enumerate(
        events,
        start=1,
    ):
        current = q7.apply_event(
            current,
            event,
            run_id=(
                f"{label}::step_{index}"
            ),
        )

        images.append(
            current
        )

        event_ids.append(
            event.event_id
        )

    return {
        "event_ids": event_ids,

        "images": images,

        "snapshots": [
            system_image_layer_snapshot(
                image
            )
            for image in images
        ],
    }


# =============================================================================
# TEMPORAL-IMAGE COMPARISON
# =============================================================================


def compare_trajectories(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Compare two ordered SystemImage trajectories slice by slice.

    Temporal difference means that at least one corresponding SystemImage
    differs in at least one represented layer.

    No scalar Temporal-Image distance is asserted.
    """

    left_images = tuple(
        left["images"]
    )

    right_images = tuple(
        right["images"]
    )

    if len(left_images) != len(
        right_images
    ):
        raise ValueError(
            "trajectory lengths differ"
        )

    per_slice = []

    for index, (
        left_image,
        right_image,
    ) in enumerate(
        zip(
            left_images,
            right_images,
        )
    ):
        comparison = (
            compare_system_images(
                left_image,
                right_image,
            )
        )

        per_slice.append(
            {
                "slice_index": index,
                **comparison,
            }
        )

    differing_slice_indices = [
        block["slice_index"]
        for block in per_slice
        if block[
            "any_layer_differs"
        ]
    ]

    return {
        "slice_count": len(
            per_slice
        ),

        "per_slice": per_slice,

        "differing_slice_indices": (
            differing_slice_indices
        ),

        "trajectory_differs": bool(
            differing_slice_indices
        ),

        "final_slice_differs": (
            per_slice[-1][
                "any_layer_differs"
            ]
            if per_slice
            else False
        ),

        "intermediate_slice_differs": (
            any(
                block[
                    "any_layer_differs"
                ]
                for block
                in per_slice[:-1]
            )
            if len(per_slice) > 1
            else False
        ),
    }


# =============================================================================
# FINAL PRESTRESS ENDPOINT COMPARISON
# =============================================================================


def final_prestress_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    left_final = (
        left["images"][-1]
    )

    right_final = (
        right["images"][-1]
    )

    return l2_distance(
        q7.prestress_vector(
            left_final
        ),
        q7.prestress_vector(
            right_final
        ),
    )


# =============================================================================
# FULL BENCHMARK
# =============================================================================


def run_benchmark() -> dict[str, Any]:
    """
    Multilayer SystemImage trajectory benchmark.

    Experimental question
    ---------------------

    Can two realised futures have the same or nearly identical final prestress
    endpoint while differing as ordered sequences of complete SystemImages?

    The benchmark uses the already-defined off-nominal future pair:

        A -> D
        D -> A

    under:

        1. no preconfiguration
        2. the same Q7 nominal-E5-derived prestress preconfiguration

    Each future is represented as:

        I_0, I_1, I_2

    and each SystemImage is inspected through distinct layers:

        normalized measures,
        prestress,
        reserve,
        adaptive connections,
        historical traces,
        context.

    IMPORTANT
    ---------

    This benchmark does NOT claim to reconstruct a predicted future Temporal
    Image.

    It evaluates realised multilayer trajectories and establishes whether a
    scalar final prestress endpoint is sufficient to characterize them.

    It also does NOT combine heterogeneous SystemImage fields into one global
    Euclidean metric.
    """

    pre_event_image = (
        q7.build_history_conditioned_pre_event_image()
    )

    nominal_event = (
        q7.build_events()[4]
    )

    nominal_prediction = (
        q7.predict_prestress_response(
            pre_event_image,
            nominal_event,
            run_id=(
                "multilayer_temporal_image::"
                "nominal_prediction"
            ),
        )
    )

    history_action = (
        q7.bounded_predictive_action(
            nominal_prediction
        )
    )

    scenarios = (
        off_nominal.build_future_scenarios()
    )

    events_A_D = scenarios[
        "sequence_A_then_D"
    ]

    events_D_A = scenarios[
        "sequence_D_then_A"
    ]

    # -------------------------------------------------------------------------
    # NO PRECONFIGURATION
    # -------------------------------------------------------------------------

    no_control_A_D = (
        evolve_multilayer_trajectory(
            pre_event_image=(
                pre_event_image
            ),
            events=events_A_D,
            action=None,
            label=(
                "no_control_A_then_D"
            ),
        )
    )

    no_control_D_A = (
        evolve_multilayer_trajectory(
            pre_event_image=(
                pre_event_image
            ),
            events=events_D_A,
            action=None,
            label=(
                "no_control_D_then_A"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Q7 PRECONFIGURATION
    # -------------------------------------------------------------------------

    controlled_A_D = (
        evolve_multilayer_trajectory(
            pre_event_image=(
                pre_event_image
            ),
            events=events_A_D,
            action=history_action,
            label=(
                "q7_action_A_then_D"
            ),
        )
    )

    controlled_D_A = (
        evolve_multilayer_trajectory(
            pre_event_image=(
                pre_event_image
            ),
            events=events_D_A,
            action=history_action,
            label=(
                "q7_action_D_then_A"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # ORDER COMPARISONS
    # -------------------------------------------------------------------------

    no_control_order_comparison = (
        compare_trajectories(
            no_control_A_D,
            no_control_D_A,
        )
    )

    controlled_order_comparison = (
        compare_trajectories(
            controlled_A_D,
            controlled_D_A,
        )
    )

    no_control_final_prestress_distance = (
        final_prestress_distance(
            no_control_A_D,
            no_control_D_A,
        )
    )

    controlled_final_prestress_distance = (
        final_prestress_distance(
            controlled_A_D,
            controlled_D_A,
        )
    )

    # -------------------------------------------------------------------------
    # CENTRAL OBSERVATION
    #
    # Do not pre-assume the answer.
    # -------------------------------------------------------------------------

    no_control_same_final_prestress = (
        no_control_final_prestress_distance
        <= 1e-12
    )

    controlled_same_final_prestress = (
        controlled_final_prestress_distance
        <= 1e-12
    )

    no_control_endpoint_insufficient = (
        no_control_same_final_prestress
        and no_control_order_comparison[
            "trajectory_differs"
        ]
    )

    controlled_endpoint_insufficient = (
        controlled_same_final_prestress
        and controlled_order_comparison[
            "trajectory_differs"
        ]
    )

    return {
        "benchmark_version": (
            BENCHMARK_VERSION
        ),

        "claim_scope": (
            CLAIM_SCOPE
        ),

        "benchmark_purpose": (
            BENCHMARK_PURPOSE
        ),

        "future_sequences": {
            "A_then_D": [
                event.event_id
                for event
                in events_A_D
            ],

            "D_then_A": [
                event.event_id
                for event
                in events_D_A
            ],
        },

        "nominal_prediction": list(
            nominal_prediction
        ),

        "q7_history_action": list(
            history_action
        ),

        "trajectories": {
            "no_control_A_then_D": (
                no_control_A_D[
                    "snapshots"
                ]
            ),

            "no_control_D_then_A": (
                no_control_D_A[
                    "snapshots"
                ]
            ),

            "q7_action_A_then_D": (
                controlled_A_D[
                    "snapshots"
                ]
            ),

            "q7_action_D_then_A": (
                controlled_D_A[
                    "snapshots"
                ]
            ),
        },

        "order_comparisons": {
            "no_control": (
                no_control_order_comparison
            ),

            "q7_action": (
                controlled_order_comparison
            ),
        },

        "central_results": {
            "no_control_final_prestress_distance_between_orders": (
                no_control_final_prestress_distance
            ),

            "q7_action_final_prestress_distance_between_orders": (
                controlled_final_prestress_distance
            ),

            "no_control_same_final_prestress_within_tolerance": (
                no_control_same_final_prestress
            ),

            "q7_action_same_final_prestress_within_tolerance": (
                controlled_same_final_prestress
            ),

            "no_control_multilayer_trajectory_differs_by_order": (
                no_control_order_comparison[
                    "trajectory_differs"
                ]
            ),

            "q7_action_multilayer_trajectory_differs_by_order": (
                controlled_order_comparison[
                    "trajectory_differs"
                ]
            ),

            "no_control_final_prestress_endpoint_insufficient": (
                no_control_endpoint_insufficient
            ),

            "q7_action_final_prestress_endpoint_insufficient": (
                controlled_endpoint_insufficient
            ),
        },

        # ---------------------------------------------------------------------
        # CLAIM BOUNDARIES
        # ---------------------------------------------------------------------

        "realised_multilayer_systemimage_sequence_evaluated": True,

        "heterogeneous_layers_collapsed_to_single_metric": False,

        "predicted_temporal_image_reconstructed": False,

        "future_temporal_image_prediction_validated": False,

        "general_preconfiguration_operator_tested": False,

        "whole_system_stabilization_tested": False,

        "objective_independent_stabilization_tested": False,

        "external_predictive_validity_tested": False,

        "interpretation": (
            "This benchmark compares ordered realised sequences of complete "
            "ROIF SystemImages rather than reducing future evolution to a "
            "single prestress endpoint or a sum of prestress norms. "
            "SystemImage layers remain distinct: normalized measures, "
            "prestress, reserve, adaptive connection state, historical "
            "traces, and context are evaluated separately. If A->D and D->A "
            "share the same final prestress endpoint while their multilayer "
            "trajectories differ, the result establishes only that final "
            "prestress is insufficient to identify the realised SystemImage "
            "trajectory in this implementation. The benchmark does not yet "
            "reconstruct or predict a future Temporal Image and does not test "
            "the full ROIF reconstruction operator Q or general "
            "preconfiguration operator Pi."
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

    root = Path(
        __file__
    ).resolve().parents[1]

    output_path = (
        root
        / "benchmark_results"
        / "multilayer_temporal_image_trajectory_v1.json"
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
        "Saved multilayer Temporal-Image trajectory benchmark:"
    )
    print(
        output_path.resolve()
    )


if __name__ == "__main__":
    main()
