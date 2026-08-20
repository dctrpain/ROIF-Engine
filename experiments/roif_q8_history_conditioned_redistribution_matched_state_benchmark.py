from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.roif_q8_history_conditioned_redistribution_benchmark import (
    BENCHMARK_VERSION as Q8_V1_BENCHMARK_VERSION,
    build_source_image,
    connection_metrics,
    evolve_conditioning_history,
    redistribution_metrics,
    run_matched_challenge,
)

from roif.history.system_image import (
    SystemImage,
    revise_system_image,
)


# =============================================================================
# Q8-v2 METADATA
# =============================================================================

BENCHMARK_VERSION = (
    "roif_q8_history_conditioned_redistribution_matched_state_v2"
)

CLAIM_SCOPE = "computational_model_only"

MECHANISM_UNDER_TEST = (
    "history_conditioned_redistribution_reconfiguration"
)


# =============================================================================
# ERRORS
# =============================================================================


class Q8MatchedStateBenchmarkError(RuntimeError):
    """Raised when Q8-v2 matched-state construction is invalid."""


# =============================================================================
# LAYER SIGNATURES
# =============================================================================


def prestress_signature(
    image: SystemImage,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (
                node.node_id,
                float(node.prestress),
            )
            for node in image.prestress_nodes
        )
    )


def connection_state_signature(
    image: SystemImage,
) -> tuple[
    tuple[
        str,
        float,
        float,
        float,
        float,
        float,
    ],
    ...,
]:
    return tuple(
        sorted(
            (
                connection.connection_id,
                float(connection.stiffness),
                float(connection.contractile_capacity),
                float(connection.reflex_gain),
                float(connection.fatigue),
                float(connection.remodeling_bias),
            )
            for connection in image.adaptive_connections
        )
    )


def transfer_gain_signature(
    image: SystemImage,
) -> tuple[tuple[str, float], ...]:
    metrics = connection_metrics(image)

    return tuple(
        sorted(
            (
                connection_id,
                float(values["effective_transfer_gain"]),
            )
            for connection_id, values
            in metrics["per_connection"].items()
        )
    )


# =============================================================================
# FACTORIAL MATCHED-STATE CONSTRUCTION
# =============================================================================


def build_factorial_states(
    baseline: SystemImage,
    conditioned: SystemImage,
) -> dict[str, SystemImage]:
    """
    Construct a 2 x 2 factorial state decomposition.

    Factors
    -------

    Prestress layer:
        p0 = baseline prestress
        pH = history-conditioned prestress

    Adaptive connection layer:
        theta0 = baseline adaptive connections
        thetaH = history-conditioned adaptive connections

    Four states
    -----------

        B00:
            p0, theta0

        C01:
            p0, thetaH
            conditioned connections with baseline prestress

        C10:
            pH, theta0
            conditioned prestress with baseline connections

        H11:
            pH, thetaH

    This isolates the effects of the two current SystemImage layers without
    claiming that the resulting response decomposition is globally additive.
    """

    baseline_prestress = baseline.prestress_nodes
    baseline_connections = baseline.adaptive_connections

    conditioned_prestress = conditioned.prestress_nodes
    conditioned_connections = conditioned.adaptive_connections

    b00 = baseline

    c01 = revise_system_image(
        source=baseline,
        target_image_id="q8_v2_p0_thetaH",
        prestress_nodes=baseline_prestress,
        adaptive_connections=conditioned_connections,
        metadata={
            "benchmark": BENCHMARK_VERSION,
            "factorial_state": "p0_thetaH",
            "prestress_source": "baseline",
            "connection_source": "conditioned",
            "matched_prestress_control": True,
            "topology_modified": False,
            "learning_applied": False,
            "biological_truth_claimed": False,
        },
    )

    c10 = revise_system_image(
        source=baseline,
        target_image_id="q8_v2_pH_theta0",
        prestress_nodes=conditioned_prestress,
        adaptive_connections=baseline_connections,
        metadata={
            "benchmark": BENCHMARK_VERSION,
            "factorial_state": "pH_theta0",
            "prestress_source": "conditioned",
            "connection_source": "baseline",
            "matched_connection_control": True,
            "topology_modified": False,
            "learning_applied": False,
            "biological_truth_claimed": False,
        },
    )

    h11 = revise_system_image(
        source=baseline,
        target_image_id="q8_v2_pH_thetaH",
        prestress_nodes=conditioned_prestress,
        adaptive_connections=conditioned_connections,
        metadata={
            "benchmark": BENCHMARK_VERSION,
            "factorial_state": "pH_thetaH",
            "prestress_source": "conditioned",
            "connection_source": "conditioned",
            "full_history_conditioned_state": True,
            "topology_modified": False,
            "learning_applied": False,
            "biological_truth_claimed": False,
        },
    )

    return {
        "B00_baseline_p0_theta0": b00,
        "C01_baseline_prestress_conditioned_connections": c01,
        "C10_conditioned_prestress_baseline_connections": c10,
        "H11_conditioned_prestress_conditioned_connections": h11,
    }


# =============================================================================
# RESPONSE VECTOR HELPERS
# =============================================================================


NODE_ORDER = (
    "A",
    "B",
    "C",
    "D",
)


def response_vector(
    response_metrics: Mapping[str, Any],
) -> tuple[float, ...]:
    node_deltas = response_metrics[
        "node_deltas"
    ]

    return tuple(
        float(node_deltas[node_id])
        for node_id in NODE_ORDER
    )


def vector_subtract(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    if len(left) != len(right):
        raise Q8MatchedStateBenchmarkError(
            "vector dimensions differ"
        )

    return tuple(
        float(a) - float(b)
        for a, b in zip(left, right)
    )


def vector_add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, ...]:
    if len(left) != len(right):
        raise Q8MatchedStateBenchmarkError(
            "vector dimensions differ"
        )

    return tuple(
        float(a) + float(b)
        for a, b in zip(left, right)
    )


def vector_norm(
    values: Sequence[float],
) -> float:
    return math.sqrt(
        sum(
            float(value) ** 2
            for value in values
        )
    )


def vector_by_node(
    values: Sequence[float],
) -> dict[str, float]:
    if len(values) != len(NODE_ORDER):
        raise Q8MatchedStateBenchmarkError(
            "unexpected response vector dimension"
        )

    return {
        node_id: float(values[index])
        for index, node_id
        in enumerate(NODE_ORDER)
    }


# =============================================================================
# FACTORIAL RESPONSE DECOMPOSITION
# =============================================================================


def factorial_response_analysis(
    responses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Compute controlled response contrasts.

    Let:

        R00 = response(p0, theta0)
        R01 = response(p0, thetaH)
        R10 = response(pH, theta0)
        R11 = response(pH, thetaH)

    Then:

        connection contrast at matched baseline prestress:
            D_theta_p0 = R01 - R00

        prestress contrast at matched baseline connections:
            D_p_theta0 = R10 - R00

        connection contrast at conditioned prestress:
            D_theta_pH = R11 - R10

        prestress contrast at conditioned connections:
            D_p_thetaH = R11 - R01

        factorial interaction residual:
            I = R11 - R10 - R01 + R00

    The interaction residual is reported descriptively.

    A non-zero value means the two layer effects are not captured by a simple
    additive decomposition in this benchmark. It is not a universal law.
    """

    r00 = response_vector(
        responses[
            "B00_baseline_p0_theta0"
        ]
    )

    r01 = response_vector(
        responses[
            "C01_baseline_prestress_conditioned_connections"
        ]
    )

    r10 = response_vector(
        responses[
            "C10_conditioned_prestress_baseline_connections"
        ]
    )

    r11 = response_vector(
        responses[
            "H11_conditioned_prestress_conditioned_connections"
        ]
    )

    connection_effect_at_p0 = (
        vector_subtract(
            r01,
            r00,
        )
    )

    prestress_effect_at_theta0 = (
        vector_subtract(
            r10,
            r00,
        )
    )

    connection_effect_at_pH = (
        vector_subtract(
            r11,
            r10,
        )
    )

    prestress_effect_at_thetaH = (
        vector_subtract(
            r11,
            r01,
        )
    )

    interaction = tuple(
        float(a)
        - float(b)
        - float(c)
        + float(d)
        for a, b, c, d
        in zip(
            r11,
            r10,
            r01,
            r00,
        )
    )

    return {
        "response_vectors": {
            "R00": vector_by_node(r00),
            "R01": vector_by_node(r01),
            "R10": vector_by_node(r10),
            "R11": vector_by_node(r11),
        },

        "connection_state_effect_at_matched_baseline_prestress": {
            "vector": vector_by_node(
                connection_effect_at_p0
            ),
            "norm": vector_norm(
                connection_effect_at_p0
            ),
        },

        "prestress_state_effect_at_matched_baseline_connections": {
            "vector": vector_by_node(
                prestress_effect_at_theta0
            ),
            "norm": vector_norm(
                prestress_effect_at_theta0
            ),
        },

        "connection_state_effect_at_conditioned_prestress": {
            "vector": vector_by_node(
                connection_effect_at_pH
            ),
            "norm": vector_norm(
                connection_effect_at_pH
            ),
        },

        "prestress_state_effect_at_conditioned_connections": {
            "vector": vector_by_node(
                prestress_effect_at_thetaH
            ),
            "norm": vector_norm(
                prestress_effect_at_thetaH
            ),
        },

        "factorial_interaction_residual": {
            "vector": vector_by_node(
                interaction
            ),
            "norm": vector_norm(
                interaction
            ),
        },

        "connection_effect_detected_at_matched_prestress": (
            vector_norm(
                connection_effect_at_p0
            )
            > 1e-12
        ),

        "prestress_effect_detected_at_matched_connections": (
            vector_norm(
                prestress_effect_at_theta0
            )
            > 1e-12
        ),

        "factorial_interaction_detected": (
            vector_norm(interaction)
            > 1e-12
        ),
    }


# =============================================================================
# STATE CONTROL AUDIT
# =============================================================================


def state_control_audit(
    states: Mapping[str, SystemImage],
) -> dict[str, Any]:
    b00 = states[
        "B00_baseline_p0_theta0"
    ]

    c01 = states[
        "C01_baseline_prestress_conditioned_connections"
    ]

    c10 = states[
        "C10_conditioned_prestress_baseline_connections"
    ]

    h11 = states[
        "H11_conditioned_prestress_conditioned_connections"
    ]

    p00 = prestress_signature(b00)
    p01 = prestress_signature(c01)
    p10 = prestress_signature(c10)
    p11 = prestress_signature(h11)

    theta00 = connection_state_signature(b00)
    theta01 = connection_state_signature(c01)
    theta10 = connection_state_signature(c10)
    theta11 = connection_state_signature(h11)

    gain00 = transfer_gain_signature(b00)
    gain01 = transfer_gain_signature(c01)
    gain10 = transfer_gain_signature(c10)
    gain11 = transfer_gain_signature(h11)

    return {
        "baseline_prestress_matches_connection_only_control": (
            p00 == p01
        ),

        "conditioned_prestress_matches_full_conditioned": (
            p10 == p11
        ),

        "baseline_connections_match_prestress_only_control": (
            theta00 == theta10
        ),

        "conditioned_connections_match_full_conditioned": (
            theta01 == theta11
        ),

        "baseline_transfer_gains_match_prestress_only_control": (
            gain00 == gain10
        ),

        "conditioned_transfer_gains_match_full_conditioned": (
            gain01 == gain11
        ),

        "baseline_vs_conditioned_prestress_differ": (
            p00 != p10
        ),

        "baseline_vs_conditioned_connections_differ": (
            theta00 != theta01
        ),
    }


# =============================================================================
# FULL Q8-v2
# =============================================================================


def run_benchmark() -> dict[str, Any]:
    """
    Q8-v2:
    Matched-state decomposition of history-conditioned redistribution.

    This benchmark does NOT ask whether the conditioned structure is more
    stable.

    It asks a narrower mechanistic question:

        Does the history-conditioned adaptive connection layer change the
        redistribution of the same perturbation when pre-challenge prestress
        is explicitly matched?

    A second control performs the complementary isolation:

        Does the history-conditioned prestress layer change redistribution
        when adaptive connection state is held at baseline?

    The full 2 x 2 design is:

                         theta0              thetaH

        p0               R00                 R01

        pH               R10                 R11

    where p is prestress state and theta is adaptive connection state.
    """

    baseline = build_source_image()

    conditioned, conditioning_trajectory = (
        evolve_conditioning_history(
            baseline
        )
    )

    states = build_factorial_states(
        baseline,
        conditioned,
    )

    audit = state_control_audit(
        states
    )

    responses: dict[str, Any] = {}

    structures: dict[str, Any] = {}

    prestress_states: dict[str, Any] = {}

    for name, image in states.items():
        structures[name] = (
            connection_metrics(image)
        )

        prestress_states[name] = {
            node_id: prestress
            for node_id, prestress
            in prestress_signature(image)
        }

        challenge_result = run_matched_challenge(
            image,
            run_id=f"q8_v2::{name}",
        )

        responses[name] = (
            redistribution_metrics(
                challenge_result
            )
        )

    factorial = factorial_response_analysis(
        responses
    )

    matched_prestress_connection_effect = (
        factorial[
            "connection_state_effect_at_matched_baseline_prestress"
        ]["norm"]
    )

    matched_connection_prestress_effect = (
        factorial[
            "prestress_state_effect_at_matched_baseline_connections"
        ]["norm"]
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "q8_v1_parent_version": (
            Q8_V1_BENCHMARK_VERSION
        ),

        "claim_scope": CLAIM_SCOPE,

        "mechanism_under_test": (
            MECHANISM_UNDER_TEST
        ),

        "experimental_design": (
            "2x2_matched_current_state_layer_decomposition"
        ),

        "factors": {
            "prestress": [
                "baseline_p0",
                "conditioned_pH",
            ],
            "adaptive_connections": [
                "baseline_theta0",
                "conditioned_thetaH",
            ],
        },

        "conditioning_history_trajectory": list(
            conditioning_trajectory
        ),

        "state_control_audit": audit,

        "prestress_states": (
            prestress_states
        ),

        "connection_structures": (
            structures
        ),

        "matched_challenge_responses": (
            responses
        ),

        "factorial_response_analysis": (
            factorial
        ),

        "central_results": {
            "connection_state_changes_response_at_matched_prestress": (
                matched_prestress_connection_effect
                > 1e-12
            ),

            "connection_state_effect_norm_at_matched_prestress": (
                matched_prestress_connection_effect
            ),

            "prestress_state_changes_response_at_matched_connections": (
                matched_connection_prestress_effect
                > 1e-12
            ),

            "prestress_state_effect_norm_at_matched_connections": (
                matched_connection_prestress_effect
            ),

            "factorial_interaction_detected": (
                factorial[
                    "factorial_interaction_detected"
                ]
            ),

            "factorial_interaction_norm": (
                factorial[
                    "factorial_interaction_residual"
                ]["norm"]
            ),
        },

        # =====================================================================
        # CLAIM BOUNDARIES
        # =====================================================================

        "matched_prestress_connection_effect_tested": True,
        "matched_connection_prestress_effect_tested": True,

        "stabilization_tested": False,
        "energy_dissipation_tested": False,
        "literal_absorption_tested": False,
        "injury_or_failure_threshold_tested": False,
        "predictive_preconfiguration_tested": False,
        "biological_adaptation_claimed": False,
        "human_foot_model_claimed": False,
        "universal_tensegrity_claimed": False,

        "interpretation": (
            "Q8-v2 isolates two current-state contributions to matched-event "
            "redistribution in the implemented ROIF model. The connection-only "
            "control uses history-conditioned adaptive connections with baseline "
            "pre-challenge prestress. The complementary prestress-only control "
            "uses history-conditioned prestress with baseline adaptive "
            "connections. A response difference between R01 and R00 therefore "
            "establishes a connection-state contribution under matched prestress "
            "within this computational implementation. A response difference "
            "between R10 and R00 establishes a prestress-state contribution under "
            "matched adaptive connections. The R11-R10-R01+R00 residual reports "
            "interaction between these two current-state layers. Q8-v2 does not "
            "interpret increased effective transfer gain, downstream response, "
            "or total response as improved stabilization, absorption, or "
            "dissipation."
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

    root = Path(__file__).resolve().parents[1]

    output_path = (
        root
        / "benchmark_results"
        / "q8_history_conditioned_redistribution_matched_state_v2.json"
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
    print("Saved Q8-v2 benchmark:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
