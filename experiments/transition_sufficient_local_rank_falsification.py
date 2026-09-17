"""
Local transition-sufficiency rank falsification benchmark.

Tests whether the preregistered future mechanical-interface response map
locally distinguishes all five dynamic coordinates of one adaptive connection.

The observable is restricted to the production mechanical interface:
    (effective_transfer_gain, contractile_capacity)

Hidden coordinates are never used as output observables.

This benchmark does NOT establish global minimality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from roif.history.adaptive_connection import (
    AdaptiveConnectionConfig,
    AdaptiveConnectionState,
    ConnectionExposure,
    effective_transfer_gain,
    update_adaptive_connection,
)


PREREGISTRATION_PATH = Path(
    "experiments/transition_sufficient_local_rank_preregistration.json"
)

RESULTS_DIR = Path("benchmark_results")
RESULT_PATH = RESULTS_DIR / (
    "transition_sufficient_local_rank_falsification.json"
)

STATE_NAMES = (
    "stiffness",
    "contractile_capacity",
    "reflex_gain",
    "fatigue",
    "remodeling_bias",
)


def _load_preregistration() -> dict[str, Any]:
    with PREREGISTRATION_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def _base_vector(
    prereg: dict[str, Any],
) -> np.ndarray:
    state = prereg["base_state"]

    return np.asarray(
        [
            state["stiffness"],
            state["contractile_capacity"],
            state["reflex_gain"],
            state["fatigue"],
            state["remodeling_bias"],
        ],
        dtype=float,
    )


def _build_state(
    vector: np.ndarray,
    prereg: dict[str, Any],
) -> AdaptiveConnectionState:
    bounds = prereg["fixed_bounds"]

    return AdaptiveConnectionState(
        connection_id="A_B",
        source_node_id="A",
        target_node_id="B",
        stiffness=float(vector[0]),
        contractile_capacity=float(vector[1]),
        reflex_gain=float(vector[2]),
        fatigue=float(vector[3]),
        remodeling_bias=float(vector[4]),
        min_stiffness=float(
            bounds["min_stiffness"]
        ),
        max_stiffness=float(
            bounds["max_stiffness"]
        ),
        min_contractile_capacity=float(
            bounds["min_contractile_capacity"]
        ),
        max_contractile_capacity=float(
            bounds["max_contractile_capacity"]
        ),
        min_reflex_gain=float(
            bounds["min_reflex_gain"]
        ),
        max_reflex_gain=float(
            bounds["max_reflex_gain"]
        ),
    )


def _build_exposure(
    *,
    name: str,
    values: dict[str, float],
) -> ConnectionExposure:
    return ConnectionExposure(
        exposure_id=f"rank_probe::{name}",
        connection_id="A_B",
        load=float(values["load"]),
        strain=float(values["strain"]),
        activation=float(values["activation"]),
        damage=float(values["damage"]),
        recovery=float(values["recovery"]),
    )


def _single_probe_response(
    *,
    state: AdaptiveConnectionState,
    probe_name: str,
    probe_values: dict[str, float],
) -> np.ndarray:
    exposure = _build_exposure(
        name=probe_name,
        values=probe_values,
    )

    result = update_adaptive_connection(
        update_id=f"rank_update::{probe_name}",
        state=state,
        exposure=exposure,
        config=AdaptiveConnectionConfig(),
    )

    target = result.target_state

    return np.asarray(
        [
            float(
                effective_transfer_gain(target)
            ),
            float(
                target.contractile_capacity
            ),
        ],
        dtype=float,
    )


def response_map(
    vector: np.ndarray,
    prereg: dict[str, Any],
) -> np.ndarray:
    state = _build_state(
        vector,
        prereg,
    )

    responses: list[float] = []

    for probe_name, probe_values in (
        prereg["probes"].items()
    ):
        response = _single_probe_response(
            state=state,
            probe_name=probe_name,
            probe_values=probe_values,
        )

        responses.extend(
            float(value)
            for value in response
        )

    return np.asarray(
        responses,
        dtype=float,
    )


def numerical_jacobian(
    *,
    vector: np.ndarray,
    prereg: dict[str, Any],
    step: float,
) -> np.ndarray:
    baseline_response = response_map(
        vector,
        prereg,
    )

    jacobian = np.zeros(
        (
            baseline_response.size,
            vector.size,
        ),
        dtype=float,
    )

    for column in range(vector.size):
        plus = vector.copy()
        minus = vector.copy()

        plus[column] += step
        minus[column] -= step

        response_plus = response_map(
            plus,
            prereg,
        )
        response_minus = response_map(
            minus,
            prereg,
        )

        jacobian[:, column] = (
            response_plus
            - response_minus
        ) / (2.0 * step)

    return jacobian


def _rank_from_singular_values(
    singular_values: np.ndarray,
    *,
    relative_threshold: float,
) -> tuple[int, float]:
    if singular_values.size == 0:
        return 0, 0.0

    sigma_max = float(
        np.max(singular_values)
    )

    threshold = (
        relative_threshold
        * sigma_max
    )

    rank = int(
        np.sum(
            singular_values
            > threshold
        )
    )

    return rank, threshold


def _inside_registered_bounds(
    vector: np.ndarray,
    prereg: dict[str, Any],
    *,
    step: float,
) -> bool:
    bounds = prereg["fixed_bounds"]

    s, c, r, f, b = (
        float(value)
        for value in vector
    )

    return bool(
        s - step
        > float(bounds["min_stiffness"])
        and s + step
        < float(bounds["max_stiffness"])
        and c - step
        > float(
            bounds[
                "min_contractile_capacity"
            ]
        )
        and c + step
        < float(
            bounds[
                "max_contractile_capacity"
            ]
        )
        and r - step
        > float(bounds["min_reflex_gain"])
        and r + step
        < float(bounds["max_reflex_gain"])
        and f - step > 0.0
        and f + step < 1.0
        and b - step > -1.0
        and b + step < 1.0
    )


def _evaluate_step(
    *,
    vector: np.ndarray,
    prereg: dict[str, Any],
    step: float,
) -> dict[str, Any]:
    jacobian = numerical_jacobian(
        vector=vector,
        prereg=prereg,
        step=step,
    )

    singular_values = np.linalg.svd(
        jacobian,
        compute_uv=False,
    )

    relative_threshold = float(
        prereg["rank_relative_threshold"]
    )

    rank, absolute_threshold = (
        _rank_from_singular_values(
            singular_values,
            relative_threshold=relative_threshold,
        )
    )

    condition_number = (
        float(
            singular_values[0]
            / singular_values[-1]
        )
        if singular_values[-1] > 0.0
        else float("inf")
    )

    return {
        "finite_difference_step": step,
        "jacobian": jacobian.tolist(),
        "singular_values": (
            singular_values.tolist()
        ),
        "rank": rank,
        "relative_rank_threshold": (
            relative_threshold
        ),
        "absolute_rank_threshold": (
            absolute_threshold
        ),
        "condition_number": (
            condition_number
        ),
        "finite_difference_states_inside_bounds": (
            _inside_registered_bounds(
                vector,
                prereg,
                step=step,
            )
        ),
    }


def run_benchmark() -> dict[str, Any]:
    prereg = _load_preregistration()

    vector = _base_vector(
        prereg
    )

    primary_step = float(
        prereg["finite_difference_step"]
    )

    stability_steps = (
        1e-5,
        1e-6,
        1e-7,
    )

    baseline_1 = response_map(
        vector,
        prereg,
    )

    baseline_2 = response_map(
        vector,
        prereg,
    )

    deterministic_repeat_distance = float(
        np.linalg.norm(
            baseline_1
            - baseline_2
        )
    )

    evaluations = {
        f"{step:.0e}": _evaluate_step(
            vector=vector,
            prereg=prereg,
            step=step,
        )
        for step in stability_steps
    }

    primary_key = (
        f"{primary_step:.0e}"
    )

    if primary_key not in evaluations:
        evaluations[primary_key] = (
            _evaluate_step(
                vector=vector,
                prereg=prereg,
                step=primary_step,
            )
        )

    primary = evaluations[
        primary_key
    ]

    primary_rank = int(
        primary["rank"]
    )

    ranks = [
        int(item["rank"])
        for item in evaluations.values()
    ]

    rank_stable = (
        len(set(ranks)) == 1
    )

    all_inside_bounds = all(
        bool(
            item[
                "finite_difference_states_inside_bounds"
            ]
        )
        for item in evaluations.values()
    )

    deterministic = (
        deterministic_repeat_distance
        == 0.0
    )

    primary_full_rank = (
        primary_rank
        == len(STATE_NAMES)
    )

    numerical_controls_valid = (
        deterministic
        and all_inside_bounds
        and rank_stable
    )

    local_lower_bound_supported = (
        numerical_controls_valid
        and primary_full_rank
    )

    if not deterministic:
        decision = (
            "INVALID_NONDETERMINISTIC_RESPONSE_MAP"
        )
    elif not all_inside_bounds:
        decision = (
            "INVALID_FINITE_DIFFERENCE_BOUNDARY_CROSSING"
        )
    elif not rank_stable:
        decision = (
            "INCONCLUSIVE_RANK_NOT_STABLE_ACROSS_REGISTERED_STEPS"
        )
    elif primary_full_rank:
        decision = (
            "LOCAL_SMOOTH_EXACT_REDUCTION_BELOW_5D_EXCLUDED"
        )
    else:
        decision = (
            "FIVE_DIMENSIONAL_LOCAL_LOWER_BOUND_NOT_ESTABLISHED"
        )

    return {
        "benchmark": (
            "transition_sufficient_local_rank_falsification"
        ),
        "preregistration": (
            str(PREREGISTRATION_PATH)
        ),
        "preregistration_version": (
            prereg["version"]
        ),
        "hidden_state_names": list(
            STATE_NAMES
        ),
        "hidden_state_dimension": len(
            STATE_NAMES
        ),
        "base_state_vector": (
            vector.tolist()
        ),
        "observable_per_probe": (
            prereg[
                "observable_after_each_probe"
            ]
        ),
        "probe_names": list(
            prereg["probes"].keys()
        ),
        "response_dimension": int(
            baseline_1.size
        ),
        "baseline_response": (
            baseline_1.tolist()
        ),
        "deterministic_repeat_distance": (
            deterministic_repeat_distance
        ),
        "deterministic": deterministic,
        "primary_finite_difference_step": (
            primary_step
        ),
        "primary_rank": (
            primary_rank
        ),
        "rank_stable_across_registered_steps": (
            rank_stable
        ),
        "all_finite_difference_states_inside_bounds": (
            all_inside_bounds
        ),
        "evaluations": evaluations,
        "registered_decision": {
            "numerical_controls_valid": (
                numerical_controls_valid
            ),
            "primary_full_rank": (
                primary_full_rank
            ),
            "local_lower_bound_supported": (
                local_lower_bound_supported
            ),
            "decision": decision,
        },
        "claim_scope": {
            "local_only": True,
            "smooth_exact_factorizations_only": True,
            "global_minimality_proved": False,
            "non_smooth_reductions_excluded": False,
            "discontinuous_encodings_excluded": False,
            "biological_validity_tested": False,
            "clinical_validity_tested": False,
        },
    }


def save_result(
    result: dict[str, Any],
) -> Path:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return RESULT_PATH


def main() -> None:
    result = run_benchmark()

    path = save_result(
        result
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print()
    print(
        f"Saved: {path}"
    )


if __name__ == "__main__":
    main()
