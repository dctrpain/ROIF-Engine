"""
ROIF Engine v1.0 — Separability / Recursive Non-Separability Benchmark

Purpose
-------
Test a deeper distinction than "additive vs multiplicative":

    separable
    vs
    state-dependent non-separable
    vs
    recursive history-dependent

The benchmark asks:

Q5.1 Separability
    Can the joint effect of A and B be reconstructed from independent effects?

    Interaction residual:
        I(A,B) =
            F(A,B)
            - F(A,0)
            - F(0,B)
            + F(0,0)

    For a strictly separable additive model:
        I(A,B) = 0

    For a state-dependent interaction model:
        I(A,B) != 0

Q5.2 Recursive path dependence
    If two systems receive the same current input but have different prior
    histories, do they converge to the same result?

    Non-recursive separable controls should not retain history once state is
    reset to the same current condition.

    Recursive ROIF dynamics may retain history because previous reconstruction
    becomes part of the next step.

Important:
- This is a computational benchmark only.
- It does not establish biological truth.
- It does not claim all natural systems are multiplicative.
- Multiplicative terms are treated as one implementation of non-separability.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from experiments.roif_end_to_end_benchmark import build_world
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    build_feedback_trajectory,
)
from roif.history.reconstruction_feedback import (
    ReconstructionFeedbackConfig,
)


BENCHMARK_VERSION = "roif_separability_benchmark_v1"


# ---------------------------------------------------------------------------
# Scalar interaction models
# ---------------------------------------------------------------------------

def separable_additive(
    a: float,
    b: float,
    *,
    bias: float = 0.0,
) -> float:
    return bias + a + b


def nonlinear_but_separable(
    a: float,
    b: float,
    *,
    bias: float = 0.0,
) -> float:
    """
    Nonlinear in each component but still separable:
        f(a) + g(b)
    """
    return (
        bias
        + a
        + 0.5 * a * a
        + b
        + 0.25 * b * b
    )


def multiplicative_interaction(
    a: float,
    b: float,
    *,
    gamma: float = 1.0,
    bias: float = 0.0,
) -> float:
    return (
        bias
        + a
        + b
        + gamma * a * b
    )


def state_modulated_interaction(
    a: float,
    b: float,
    *,
    state: float,
    gamma: float = 1.0,
    bias: float = 0.0,
) -> float:
    """
    Coupling strength depends on current state.
    """
    effective_gamma = gamma * (
        1.0
        + state
    )

    return (
        bias
        + a
        + b
        + effective_gamma * a * b
    )


def interaction_residual(
    fn,
    *,
    a: float,
    b: float,
    **kwargs,
) -> float:
    """
    I(A,B) = F(A,B)-F(A,0)-F(0,B)+F(0,0)
    """
    return (
        fn(a, b, **kwargs)
        - fn(a, 0.0, **kwargs)
        - fn(0.0, b, **kwargs)
        + fn(0.0, 0.0, **kwargs)
    )


@dataclass(frozen=True)
class SeparabilityRow:
    model: str
    a: float
    b: float
    state: float
    output_ab: float
    interaction_residual: float
    separable_within_tolerance: bool


def run_separability_grid(
    *,
    a_values: Sequence[float] = (0.1, 0.3, 0.6, 1.0),
    b_values: Sequence[float] = (0.1, 0.3, 0.6, 1.0),
    states: Sequence[float] = (0.0, 0.5, 1.0),
    gamma: float = 1.0,
    tolerance: float = 1e-12,
) -> tuple[SeparabilityRow, ...]:
    rows = []

    for a in a_values:
        for b in b_values:
            for state in states:
                models = (
                    (
                        "separable_additive",
                        separable_additive,
                        {},
                    ),
                    (
                        "nonlinear_but_separable",
                        nonlinear_but_separable,
                        {},
                    ),
                    (
                        "multiplicative_interaction",
                        multiplicative_interaction,
                        {"gamma": gamma},
                    ),
                    (
                        "state_modulated_interaction",
                        state_modulated_interaction,
                        {
                            "state": state,
                            "gamma": gamma,
                        },
                    ),
                )

                for model_name, fn, kwargs in models:
                    output = fn(
                        a,
                        b,
                        **kwargs,
                    )

                    residual = interaction_residual(
                        fn,
                        a=a,
                        b=b,
                        **kwargs,
                    )

                    rows.append(
                        SeparabilityRow(
                            model=model_name,
                            a=float(a),
                            b=float(b),
                            state=float(state),
                            output_ab=float(output),
                            interaction_residual=float(residual),
                            separable_within_tolerance=(
                                abs(residual)
                                <= tolerance
                            ),
                        )
                    )

    return tuple(rows)


# ---------------------------------------------------------------------------
# Recursive history dependence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecursiveHistoryRow:
    feedback_gain: float
    history_label: str
    final_vector: tuple[float, ...]
    final_vector_norm: float
    final_dominant: str


def _norm(values: Sequence[float]) -> float:
    return sum(
        float(value) ** 2
        for value in values
    ) ** 0.5


def _distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(
            left,
            right,
        )
    ) ** 0.5


def run_recursive_history_branch(
    *,
    feedback_gains: Sequence[float] = (0.0, 1.0, 5.0, 20.0),
    repeated_steps: int = 4,
) -> tuple[RecursiveHistoryRow, ...]:
    world = build_world()

    frames = tuple(
        FeedbackFrame(
            cue=world.cue_door,
            context=world.adapt_context,
        )
        for _ in range(repeated_steps)
    )

    rows = []

    for gain in feedback_gains:
        config = ReconstructionFeedbackConfig(
            feedback_gain=float(gain),
            max_feedback_multiplier=100.0,
        )

        for history_label, seed in (
            (
                "sens_history",
                world.attractors[0].center,
            ),
            (
                "adapt_history",
                world.attractors[1].center,
            ),
        ):
            trajectory = build_feedback_trajectory(
                trajectory_id=f"q5::{gain}::{history_label}",
                frames=frames,
                attractors=world.attractors,
                config=config,
                seed_from_first_baseline=False,
                initial_reconstructed_vector=seed,
            )

            final_vector = tuple(
                trajectory.final_reconstructed_vector
            )

            rows.append(
                RecursiveHistoryRow(
                    feedback_gain=float(gain),
                    history_label=history_label,
                    final_vector=final_vector,
                    final_vector_norm=_norm(
                        final_vector
                    ),
                    final_dominant=(
                        trajectory.steps[-1]
                        .feedback_dominant_attractor_id
                    ),
                )
            )

    return tuple(rows)


def history_pair_distances(
    rows: Sequence[RecursiveHistoryRow],
) -> dict[str, float]:
    grouped: dict[
        float,
        dict[str, RecursiveHistoryRow],
    ] = {}

    for row in rows:
        grouped.setdefault(
            row.feedback_gain,
            {},
        )[row.history_label] = row

    out = {}

    for gain, pair in sorted(
        grouped.items()
    ):
        if (
            "sens_history" in pair
            and "adapt_history" in pair
        ):
            out[str(gain)] = _distance(
                pair["sens_history"].final_vector,
                pair["adapt_history"].final_vector,
            )

    return out


# ---------------------------------------------------------------------------
# Recursive scalar demonstrator
# ---------------------------------------------------------------------------

def recursive_state_update(
    *,
    x: float,
    current_input: float,
    history_state: float,
    coupling: float,
    interaction_gain: float,
) -> float:
    """
    Minimal recursive non-separable state update.

    Current input does not act independently:
        effect(current_input)
        depends on history_state.

    x_(t+1) =
        x_t
        + current_input
        + coupling * x_t
        + interaction_gain * current_input * history_state
    """
    return (
        x
        + current_input
        + coupling * x
        + interaction_gain
        * current_input
        * history_state
    )


@dataclass(frozen=True)
class RecursiveScalarRow:
    history_label: str
    step: int
    current_input: float
    history_state: float
    value: float


def run_recursive_scalar_demo(
    *,
    steps: int = 8,
    current_input: float = 0.2,
    coupling: float = 0.10,
    interaction_gain: float = 0.80,
) -> tuple[RecursiveScalarRow, ...]:
    rows = []

    histories = {
        "low_history": 0.1,
        "high_history": 0.9,
    }

    for label, history_state in histories.items():
        x = 0.0

        for step in range(
            1,
            steps + 1,
        ):
            x = recursive_state_update(
                x=x,
                current_input=current_input,
                history_state=history_state,
                coupling=coupling,
                interaction_gain=interaction_gain,
            )

            rows.append(
                RecursiveScalarRow(
                    history_label=label,
                    step=step,
                    current_input=current_input,
                    history_state=history_state,
                    value=x,
                )
            )

            history_state = x

    return tuple(rows)


def scalar_final_distance(
    rows: Sequence[RecursiveScalarRow],
) -> float:
    grouped: dict[
        str,
        list[RecursiveScalarRow],
    ] = {}

    for row in rows:
        grouped.setdefault(
            row.history_label,
            [],
        ).append(row)

    low = grouped["low_history"][-1].value
    high = grouped["high_history"][-1].value

    return abs(
        high
        - low
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_separability(
    rows: Sequence[SeparabilityRow],
) -> dict:
    models = sorted(
        {
            row.model
            for row in rows
        }
    )

    summary = {}

    for model in models:
        subset = [
            row
            for row in rows
            if row.model == model
        ]

        absolute_residuals = [
            abs(
                row.interaction_residual
            )
            for row in subset
        ]

        summary[model] = {
            "mean_abs_interaction_residual": mean(
                absolute_residuals
            ),
            "max_abs_interaction_residual": max(
                absolute_residuals
            ),
            "separable_fraction": (
                sum(
                    1
                    for row in subset
                    if row.separable_within_tolerance
                )
                / len(subset)
            ),
        }

    return summary


def summarize_benchmark(
    separability_rows: Sequence[SeparabilityRow],
    history_rows: Sequence[RecursiveHistoryRow],
    scalar_rows: Sequence[RecursiveScalarRow],
) -> dict:
    pair_distances = history_pair_distances(
        history_rows
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": "computational_model_only",
        "clinical_validation_claimed": False,
        "biological_multiplicativity_claimed": False,
        "core_hypothesis": (
            "ROIF is modeled as recursive, state-dependent, "
            "history-dependent, and potentially non-separable."
        ),
        "separability": summarize_separability(
            separability_rows
        ),
        "roif_recursive_history_dependence": {
            "pair_distances_by_feedback_gain": pair_distances,
            "nonzero_history_effect_count": sum(
                1
                for value in pair_distances.values()
                if value > 1e-12
            ),
        },
        "recursive_scalar_demo": {
            "final_history_distance": scalar_final_distance(
                scalar_rows
            ),
            "same_current_input_used": True,
        },
        "interpretation_note": (
            "A zero interaction residual characterizes separability under the "
            "chosen decomposition. Nonzero residual demonstrates interaction "
            "but does not by itself prove multiplicativity. Recursive ROIF "
            "history dependence is evaluated separately."
        ),
    }


def save_csv(
    rows: Sequence[dict],
    path: Path,
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark() -> dict:
    separability_rows = (
        run_separability_grid()
    )

    history_rows = (
        run_recursive_history_branch()
    )

    scalar_rows = (
        run_recursive_scalar_demo()
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "separability_rows": [
            asdict(row)
            for row in separability_rows
        ],
        "history_rows": [
            asdict(row)
            for row in history_rows
        ],
        "recursive_scalar_rows": [
            asdict(row)
            for row in scalar_rows
        ],
        "summary": summarize_benchmark(
            separability_rows,
            history_rows,
            scalar_rows,
        ),
    }


def main() -> None:
    data = run_benchmark()

    output_dir = Path(
        "benchmark_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "separability_benchmark_v1.json"
    ).write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    save_csv(
        data["separability_rows"],
        output_dir
        / "separability_grid_v1.csv",
    )

    save_csv(
        data["history_rows"],
        output_dir
        / "recursive_history_v1.csv",
    )

    save_csv(
        data["recursive_scalar_rows"],
        output_dir
        / "recursive_scalar_v1.csv",
    )

    print(
        json.dumps(
            data["summary"],
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "\nSaved Q5 outputs to: "
        + str(
            output_dir.resolve()
        )
    )


if __name__ == "__main__":
    main()
