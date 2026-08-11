"""
ROIF Engine v1.0 — Additive vs Multiplicative Control Benchmark v2

Purpose
-------
Replace the original simplistic additive-vs-multiplicative comparison with a
matched synthetic control that directly addresses reviewer criticism.

Models:
1. linear_additive
2. nonlinear_additive_threshold
3. multiplicative
4. roif_recursive_history_dependent

Design principles:
- match local one-step response at a reference state
- sweep the same gain parameter across models
- record threshold crossing, final magnitude, peak magnitude
- record sensitivity to history for ROIF
- do NOT claim biological multiplicativity
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path
from statistics import mean
from typing import Sequence

from experiments.roif_end_to_end_benchmark import (
    BENCHMARK_VERSION,
    build_world,
)
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    build_feedback_trajectory,
)
from roif.history.reconstruction_feedback import ReconstructionFeedbackConfig


CONTROL_VERSION = "roif_additive_multiplicative_control_v2"


@dataclass(frozen=True)
class MatchedControlRow:
    model: str
    gain: float
    reference_state: float
    local_increment: float
    final_total: float
    peak_total: float
    threshold_crossed: bool
    first_crossing_step: int | None


@dataclass(frozen=True)
class RoifHistoryRow:
    feedback_gain: float
    history_label: str
    final_dominant: str
    final_vector_norm: float
    final_vector: tuple[float, ...]


def vector_norm(values: Sequence[float]) -> float:
    return sum(float(v) ** 2 for v in values) ** 0.5


def threshold_summary(
    values: Sequence[float],
    threshold: float,
) -> tuple[bool, int | None]:
    for index, value in enumerate(values):
        if float(value) >= threshold:
            return True, index
    return False, None


def linear_additive_series(
    *,
    initial: float,
    input_drive: float,
    local_increment: float,
    steps: int,
) -> tuple[float, ...]:
    x = float(initial)
    out = [x]

    for _ in range(steps):
        x = max(
            0.0,
            x
            + input_drive
            + local_increment,
        )
        out.append(x)

    return tuple(out)


def nonlinear_additive_threshold_series(
    *,
    initial: float,
    input_drive: float,
    local_increment: float,
    reference_state: float,
    threshold_state: float,
    nonlinear_gain: float,
    steps: int,
) -> tuple[float, ...]:
    """
    Additive system with a nonlinear threshold-like state term.

    Below threshold, response is near matched local additive response.
    Above threshold, a state-dependent additive acceleration appears.
    """
    x = float(initial)
    out = [x]

    for _ in range(steps):
        excess = max(
            0.0,
            x - threshold_state,
        )

        nonlinear_term = (
            nonlinear_gain
            * excess ** 2
        )

        x = max(
            0.0,
            x
            + input_drive
            + local_increment
            + nonlinear_term,
        )
        out.append(x)

    return tuple(out)


def multiplicative_series(
    *,
    initial: float,
    input_drive: float,
    multiplier: float,
    steps: int,
) -> tuple[float, ...]:
    x = float(initial)
    out = [x]

    for _ in range(steps):
        x = max(
            0.0,
            (x + input_drive)
            * multiplier,
        )
        out.append(x)

    return tuple(out)


def matched_parameters(
    *,
    gain: float,
    reference_state: float,
    input_drive: float,
) -> tuple[float, float]:
    """
    Match the one-step increment at reference_state.

    Multiplicative:
        x1 = (x0 + u) * exp(gain)

    Desired local increment above x0 + u:
        delta = (x0 + u) * (exp(gain) - 1)

    The additive models receive that same local increment.
    """
    multiplier = exp(
        float(gain)
    )

    local_increment = (
        reference_state
        + input_drive
    ) * (
        multiplier
        - 1.0
    )

    return (
        local_increment,
        multiplier,
    )


def run_matched_synthetic_control(
    *,
    gains: Sequence[float] = (0.02, 0.05, 0.10, 0.20),
    initial: float = 0.05,
    reference_state: float = 0.25,
    input_drive: float = 0.02,
    threshold_state: float = 0.50,
    nonlinear_gain: float = 0.80,
    steps: int = 12,
    threshold: float = 1.0,
) -> tuple[MatchedControlRow, ...]:
    rows = []

    for gain in gains:
        local_increment, multiplier = matched_parameters(
            gain=float(gain),
            reference_state=reference_state,
            input_drive=input_drive,
        )

        models = (
            (
                "linear_additive",
                linear_additive_series(
                    initial=initial,
                    input_drive=input_drive,
                    local_increment=local_increment,
                    steps=steps,
                ),
            ),
            (
                "nonlinear_additive_threshold",
                nonlinear_additive_threshold_series(
                    initial=initial,
                    input_drive=input_drive,
                    local_increment=local_increment,
                    reference_state=reference_state,
                    threshold_state=threshold_state,
                    nonlinear_gain=nonlinear_gain,
                    steps=steps,
                ),
            ),
            (
                "multiplicative",
                multiplicative_series(
                    initial=initial,
                    input_drive=input_drive,
                    multiplier=multiplier,
                    steps=steps,
                ),
            ),
        )

        for model_name, series in models:
            crossed, first_crossing = threshold_summary(
                series,
                threshold,
            )

            rows.append(
                MatchedControlRow(
                    model=model_name,
                    gain=float(gain),
                    reference_state=reference_state,
                    local_increment=local_increment,
                    final_total=series[-1],
                    peak_total=max(series),
                    threshold_crossed=crossed,
                    first_crossing_step=first_crossing,
                )
            )

    return tuple(rows)


def run_roif_history_control(
    *,
    feedback_gains: Sequence[float] = (0.0, 1.0, 5.0, 20.0),
) -> tuple[RoifHistoryRow, ...]:
    world = build_world()

    frames = tuple(
        FeedbackFrame(
            cue=world.cue_door,
            context=world.adapt_context,
        )
        for _ in range(4)
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
                trajectory_id=f"q4v2::{gain}::{history_label}",
                frames=frames,
                attractors=world.attractors,
                config=config,
                seed_from_first_baseline=False,
                initial_reconstructed_vector=seed,
            )

            final_vector = trajectory.final_reconstructed_vector

            rows.append(
                RoifHistoryRow(
                    feedback_gain=float(gain),
                    history_label=history_label,
                    final_dominant=(
                        trajectory.steps[-1].feedback_dominant_attractor_id
                    ),
                    final_vector_norm=vector_norm(final_vector),
                    final_vector=final_vector,
                )
            )

    return tuple(rows)


def pairwise_history_distance(
    rows: Sequence[RoifHistoryRow],
) -> dict[str, float]:
    grouped: dict[float, dict[str, RoifHistoryRow]] = {}

    for row in rows:
        grouped.setdefault(
            row.feedback_gain,
            {},
        )[row.history_label] = row

    out = {}

    for gain, pair in sorted(grouped.items()):
        if (
            "sens_history" in pair
            and "adapt_history" in pair
        ):
            left = pair["sens_history"].final_vector
            right = pair["adapt_history"].final_vector

            distance = sum(
                (float(a) - float(b)) ** 2
                for a, b in zip(left, right)
            ) ** 0.5

            out[str(gain)] = distance

    return out


def summarize_control(
    synthetic_rows: Sequence[MatchedControlRow],
    roif_rows: Sequence[RoifHistoryRow],
) -> dict:
    models = sorted(
        {
            row.model
            for row in synthetic_rows
        }
    )

    model_summary = {}

    for model in models:
        subset = [
            row
            for row in synthetic_rows
            if row.model == model
        ]

        model_summary[model] = {
            "mean_final_total": mean(
                row.final_total
                for row in subset
            ),
            "threshold_cross_count": sum(
                1
                for row in subset
                if row.threshold_crossed
            ),
            "mean_peak_total": mean(
                row.peak_total
                for row in subset
            ),
        }

    history_distances = pairwise_history_distance(
        roif_rows
    )

    return {
        "control_version": CONTROL_VERSION,
        "parent_benchmark_version": BENCHMARK_VERSION,
        "claim_scope": "computational_model_only",
        "clinical_validation_claimed": False,
        "biological_multiplicativity_claimed": False,
        "synthetic_models": model_summary,
        "roif_history_dependence": {
            "pair_distances_by_feedback_gain": history_distances,
            "nonzero_history_effect_count": sum(
                1
                for value in history_distances.values()
                if value > 1e-12
            ),
        },
        "interpretation_note": (
            "Matched local response removes the trivial advantage created by "
            "adding a fixed coupling term every step. The nonlinear additive "
            "baseline tests whether threshold-like abrupt transitions can arise "
            "without multiplicative propagation. ROIF history dependence is "
            "reported separately and is not treated as proof of biological "
            "multiplicativity."
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


def run_control_benchmark() -> dict:
    synthetic_rows = run_matched_synthetic_control()
    roif_rows = run_roif_history_control()

    return {
        "control_version": CONTROL_VERSION,
        "synthetic_rows": [
            asdict(row)
            for row in synthetic_rows
        ],
        "roif_history_rows": [
            asdict(row)
            for row in roif_rows
        ],
        "summary": summarize_control(
            synthetic_rows,
            roif_rows,
        ),
    }


def main() -> None:
    data = run_control_benchmark()

    output_dir = Path(
        "benchmark_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "additive_multiplicative_control_v2.json"
    ).write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    save_csv(
        data["synthetic_rows"],
        output_dir
        / "additive_multiplicative_control_v2.csv",
    )

    save_csv(
        data["roif_history_rows"],
        output_dir
        / "roif_history_control_v2.csv",
    )

    print(
        json.dumps(
            data["summary"],
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "\nSaved Q4 v2 outputs to: "
        + str(
            output_dir.resolve()
        )
    )


if __name__ == "__main__":
    main()
