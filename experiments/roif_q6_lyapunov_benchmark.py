"""
ROIF Engine v1.0 — Q6 Perturbation / Lyapunov-Like Benchmark

Purpose
-------
Test whether the recursive ROIF architecture exhibits sensitive dependence on
small perturbations of internal state/history.

This benchmark does NOT claim deterministic chaos by default.

It measures:
- perturbation size epsilon
- trajectory separation delta(t)
- finite-time Lyapunov-like estimate
- gain dependence
- boundedness of trajectory separation
- whether perturbations decay, persist, or amplify

A positive finite-time Lyapunov-like estimate over a finite interval is treated
as evidence of local perturbation amplification, not as proof of a chaotic
attractor.

Chaos would require stronger evidence, including persistent positive maximal
Lyapunov exponent over sufficiently long trajectories, bounded dynamics, and
appropriate robustness checks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from math import isfinite, log
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from experiments.roif_end_to_end_benchmark import build_world
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    build_feedback_trajectory,
    reconstructed_vector_series,
)
from roif.history.reconstruction_feedback import (
    ReconstructionFeedbackConfig,
)


BENCHMARK_VERSION = "roif_q6_lyapunov_like_v1"


def _norm(values: Sequence[float]) -> float:
    return sum(float(v) ** 2 for v in values) ** 0.5


def _distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")

    return sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(left, right)
    ) ** 0.5


def _perturb(
    vector: Sequence[float],
    *,
    epsilon: float,
    axis: int = 0,
) -> tuple[float, ...]:
    result = list(float(v) for v in vector)

    if not result:
        raise ValueError("vector must not be empty")

    if axis < 0 or axis >= len(result):
        raise ValueError("axis out of bounds")

    result[axis] += float(epsilon)

    return tuple(result)


def _finite_time_lambda(
    *,
    delta_0: float,
    delta_t: float,
    t: int,
) -> float | None:
    if t <= 0:
        return None

    if delta_0 <= 0.0:
        return None

    if delta_t <= 0.0:
        return float("-inf")

    return (
        log(delta_t / delta_0)
        / float(t)
    )


@dataclass(frozen=True)
class PerturbationStepRow:
    feedback_gain: float
    epsilon: float
    axis: int
    step: int
    delta_0: float
    delta_t: float
    amplification_ratio: float
    lyapunov_like: float | None


@dataclass(frozen=True)
class PerturbationSummaryRow:
    feedback_gain: float
    epsilon: float
    axis: int
    steps: int
    delta_0: float
    final_delta: float
    peak_delta: float
    final_amplification_ratio: float
    peak_amplification_ratio: float
    final_lyapunov_like: float | None
    max_finite_lyapunov_like: float | None
    regime: str
    bounded_under_limit: bool


def _trajectory_vector_series(trajectory) -> tuple[tuple[float, ...], ...]:
    """
    Extract reconstructed vectors through the public feedback-trajectory API.

    Q6 intentionally does not depend on FeedbackTrajectoryStep internals.
    """
    return tuple(
        tuple(vector)
        for vector in reconstructed_vector_series(
            trajectory
        )
    )


def classify_perturbation_regime(
    *,
    delta_0: float,
    final_delta: float,
    peak_delta: float,
    tolerance: float = 1e-12,
) -> str:
    if delta_0 <= 0.0:
        return "undefined"

    if final_delta < delta_0 - tolerance:
        return "decaying"

    if abs(final_delta - delta_0) <= tolerance:
        if peak_delta > delta_0 + tolerance:
            return "transient_amplification"
        return "neutral"

    if final_delta > delta_0 + tolerance:
        return "amplifying"

    return "mixed"


def run_perturbation_pair(
    *,
    feedback_gain: float,
    epsilon: float,
    steps: int,
    axis: int = 0,
    bounded_limit: float = 100.0,
) -> tuple[
    tuple[PerturbationStepRow, ...],
    PerturbationSummaryRow,
]:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    if steps < 2:
        raise ValueError("steps must be >= 2")

    world = build_world()

    frames = tuple(
        FeedbackFrame(
            cue=world.cue_door,
            context=world.adapt_context,
        )
        for _ in range(steps)
    )

    baseline_seed = tuple(
        world.attractors[0].center
    )

    perturbed_seed = _perturb(
        baseline_seed,
        epsilon=epsilon,
        axis=axis,
    )

    delta_0 = _distance(
        baseline_seed,
        perturbed_seed,
    )

    config = ReconstructionFeedbackConfig(
        feedback_gain=float(feedback_gain),
        max_feedback_multiplier=100.0,
    )

    baseline = build_feedback_trajectory(
        trajectory_id=(
            f"q6::baseline::{feedback_gain}::{epsilon}::{axis}"
        ),
        frames=frames,
        attractors=world.attractors,
        config=config,
        seed_from_first_baseline=False,
        initial_reconstructed_vector=baseline_seed,
    )

    perturbed = build_feedback_trajectory(
        trajectory_id=(
            f"q6::perturbed::{feedback_gain}::{epsilon}::{axis}"
        ),
        frames=frames,
        attractors=world.attractors,
        config=config,
        seed_from_first_baseline=False,
        initial_reconstructed_vector=perturbed_seed,
    )

    baseline_series = _trajectory_vector_series(
        baseline
    )

    perturbed_series = _trajectory_vector_series(
        perturbed
    )

    if len(baseline_series) != len(perturbed_series):
        raise RuntimeError("trajectory lengths must match")

    step_rows = []

    deltas = []

    for index, (
        left,
        right,
    ) in enumerate(
        zip(
            baseline_series,
            perturbed_series,
        ),
        start=1,
    ):
        delta_t = _distance(
            left,
            right,
        )

        deltas.append(
            delta_t
        )

        ratio = (
            delta_t / delta_0
            if delta_0 > 0.0
            else float("nan")
        )

        lam = _finite_time_lambda(
            delta_0=delta_0,
            delta_t=delta_t,
            t=index,
        )

        step_rows.append(
            PerturbationStepRow(
                feedback_gain=float(feedback_gain),
                epsilon=float(epsilon),
                axis=int(axis),
                step=index,
                delta_0=delta_0,
                delta_t=delta_t,
                amplification_ratio=ratio,
                lyapunov_like=lam,
            )
        )

    final_delta = deltas[-1]
    peak_delta = max(deltas)

    finite_lambdas = [
        row.lyapunov_like
        for row in step_rows
        if row.lyapunov_like is not None
        and isfinite(row.lyapunov_like)
    ]

    final_lambda = step_rows[-1].lyapunov_like

    max_lambda = (
        max(finite_lambdas)
        if finite_lambdas
        else None
    )

    regime = classify_perturbation_regime(
        delta_0=delta_0,
        final_delta=final_delta,
        peak_delta=peak_delta,
    )

    bounded = (
        peak_delta <= bounded_limit
    )

    summary = PerturbationSummaryRow(
        feedback_gain=float(feedback_gain),
        epsilon=float(epsilon),
        axis=int(axis),
        steps=int(steps),
        delta_0=delta_0,
        final_delta=final_delta,
        peak_delta=peak_delta,
        final_amplification_ratio=(
            final_delta / delta_0
        ),
        peak_amplification_ratio=(
            peak_delta / delta_0
        ),
        final_lyapunov_like=final_lambda,
        max_finite_lyapunov_like=max_lambda,
        regime=regime,
        bounded_under_limit=bounded,
    )

    return (
        tuple(step_rows),
        summary,
    )


def run_q6_grid(
    *,
    feedback_gains: Sequence[float] = (
        0.0,
        1.0,
        5.0,
        20.0,
    ),
    epsilons: Sequence[float] = (
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
    ),
    axes: Sequence[int] = (0,),
    steps: int = 12,
) -> tuple[
    tuple[PerturbationStepRow, ...],
    tuple[PerturbationSummaryRow, ...],
]:
    all_steps = []
    all_summaries = []

    for gain in feedback_gains:
        for epsilon in epsilons:
            for axis in axes:
                step_rows, summary = run_perturbation_pair(
                    feedback_gain=float(gain),
                    epsilon=float(epsilon),
                    axis=int(axis),
                    steps=steps,
                )

                all_steps.extend(
                    step_rows
                )

                all_summaries.append(
                    summary
                )

    return (
        tuple(all_steps),
        tuple(all_summaries),
    )


def summarize_q6(
    summaries: Sequence[PerturbationSummaryRow],
) -> dict:
    by_gain = {}

    gains = sorted(
        {
            row.feedback_gain
            for row in summaries
        }
    )

    for gain in gains:
        subset = [
            row
            for row in summaries
            if row.feedback_gain == gain
        ]

        finite_final = [
            row.final_lyapunov_like
            for row in subset
            if row.final_lyapunov_like is not None
            and isfinite(row.final_lyapunov_like)
        ]

        finite_max = [
            row.max_finite_lyapunov_like
            for row in subset
            if row.max_finite_lyapunov_like is not None
            and isfinite(row.max_finite_lyapunov_like)
        ]

        by_gain[str(gain)] = {
            "mean_final_amplification_ratio": mean(
                row.final_amplification_ratio
                for row in subset
            ),
            "max_peak_amplification_ratio": max(
                row.peak_amplification_ratio
                for row in subset
            ),
            "mean_final_lyapunov_like": (
                mean(finite_final)
                if finite_final
                else None
            ),
            "max_finite_lyapunov_like": (
                max(finite_max)
                if finite_max
                else None
            ),
            "positive_final_lambda_count": sum(
                1
                for row in subset
                if row.final_lyapunov_like is not None
                and isfinite(row.final_lyapunov_like)
                and row.final_lyapunov_like > 0.0
            ),
            "positive_max_lambda_count": sum(
                1
                for row in subset
                if row.max_finite_lyapunov_like is not None
                and isfinite(row.max_finite_lyapunov_like)
                and row.max_finite_lyapunov_like > 0.0
            ),
            "regimes": sorted(
                {
                    row.regime
                    for row in subset
                }
            ),
            "all_bounded_under_limit": all(
                row.bounded_under_limit
                for row in subset
            ),
        }

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": "computational_model_only",
        "chaos_claimed": False,
        "butterfly_effect_claimed": False,
        "clinical_validation_claimed": False,
        "interpretation": (
            "Positive finite-time Lyapunov-like values indicate local "
            "perturbation amplification over the measured interval. "
            "They are not sufficient to establish deterministic chaos."
        ),
        "by_feedback_gain": by_gain,
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
        writer.writerows(
            rows
        )


def run_benchmark() -> dict:
    step_rows, summary_rows = run_q6_grid()

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "step_rows": [
            asdict(row)
            for row in step_rows
        ],
        "summary_rows": [
            asdict(row)
            for row in summary_rows
        ],
        "summary": summarize_q6(
            summary_rows
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
        / "q6_lyapunov_like_v1.json"
    ).write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    save_csv(
        data["step_rows"],
        output_dir
        / "q6_lyapunov_steps_v1.csv",
    )

    save_csv(
        data["summary_rows"],
        output_dir
        / "q6_lyapunov_summary_v1.csv",
    )

    print(
        json.dumps(
            data["summary"],
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "\nSaved Q6 outputs to: "
        + str(
            output_dir.resolve()
        )
    )


if __name__ == "__main__":
    main()


