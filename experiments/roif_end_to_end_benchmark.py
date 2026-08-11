"""
ROIF Engine v1.0 — End-to-End Benchmark

Purpose
-------
Reproducible computational benchmark for the first algorithmic ROIF paper.

This benchmark is intentionally NOT a clinical validation study.

It evaluates four computational questions:

Q1. History dependence
    Does the same current cue/context produce different trajectories when the
    recursive initial state/history differs?

Q2. Recursive stabilization
    Does repeated exposure produce decreasing recursive drift and a stable
    feedback regime?

Q3. Memory-state transition
    Does an eligible repeated experience pass consolidation + explicit
    commitment and create a new immutable M_(t+1) without rewriting M_t?

Q4. Additive vs multiplicative control
    Can a matched synthetic additive and multiplicative propagation model be
    compared directly instead of assuming that abrupt change uniquely implies
    multiplicative dynamics?

The benchmark records:
- exact parameter values
- deterministic run identifiers
- per-run outcomes
- summary statistics
- sensitivity across feedback gain
- explicit non-clinical / non-causal claims

No external packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import exp, isfinite, sqrt
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from roif.history.experience_transformation import (
    BodyResponse,
    ExperienceEvent,
    SystemResponse,
    SystemState,
    build_experience_transformation,
)
from roif.history.experience_sequence import build_experience_sequence
from roif.history.experience_pattern import build_experience_pattern
from roif.history.experience_attractor import build_experience_attractor
from roif.history.attractor_dynamics import (
    AttractorPrestress,
    DynamicsState,
)
from roif.history.attractor_trajectory import build_attractor_trajectory
from roif.history.experience_conditioning import (
    ConditioningCue,
    build_conditioning_memory,
    observation_from_trajectory,
)
from roif.history.associative_context import (
    AssociativeCueInput,
    build_associative_context,
)
from roif.history.feedback_trajectory import (
    FeedbackFrame,
    build_feedback_trajectory,
    feedback_dominant_series,
    recursive_drift_series,
)
from roif.history.feedback_stability import analyze_feedback_stability
from roif.history.experience_consolidation import ExperienceConsolidationConfig
from roif.history.memory_commitment import MemoryCommitmentConfig
from roif.history.memory_integration import MemoryIntegrationConfig
from roif.history.memory_state import empty_memory_state
from roif.history.reconstruction_feedback import ReconstructionFeedbackConfig
from roif.history.experience_cycle import (
    ExperienceCycleConfig,
    run_experience_cycle,
)


BENCHMARK_VERSION = "roif_e2e_benchmark_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")
    return sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right)
        )
    )


def _finite_tuple(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if not result:
        raise ValueError("vector must not be empty")
    if not all(isfinite(v) for v in result):
        raise ValueError("vector values must be finite")
    return result


def make_state(
    state_id: str,
    *,
    cognitive: Sequence[float],
    physiological: Sequence[float],
    reserve: float,
) -> SystemState:
    return SystemState(
        state_id=state_id,
        cognitive=tuple(cognitive),
        physiological=tuple(physiological),
        contextual=(0.5, 0.2),
        reserve=reserve,
    )


def make_response(response_id: str, magnitude: float) -> SystemResponse:
    return SystemResponse(
        response_id=response_id,
        cognitive_response=(magnitude, magnitude * 0.9),
        physiological_response=(magnitude * 0.8, magnitude * 0.7),
        behavioral_response=(magnitude * 0.6, magnitude * 0.5),
    )


def make_body(magnitude: float) -> BodyResponse:
    return BodyResponse(
        autonomic=(magnitude, magnitude * 0.9),
        endocrine=(magnitude * 0.8, magnitude * 0.7),
        immune=(magnitude * 0.4, magnitude * 0.3),
        motor=(magnitude * 0.75, magnitude * 0.65),
        interoceptive=(magnitude * 0.95, magnitude * 0.85),
    )


def build_sequence_from_states(
    sequence_id: str,
    event: ExperienceEvent,
    states: Sequence[SystemState],
    magnitudes: Sequence[float],
):
    transformations = []

    for index in range(len(states) - 1):
        transformations.append(
            build_experience_transformation(
                transformation_id=f"{sequence_id}_tx{index}",
                state_before=states[index],
                event=event,
                response=make_response(
                    f"{sequence_id}_r{index}",
                    magnitudes[index],
                ),
                body_response=make_body(magnitudes[index]),
                state_after=states[index + 1],
            )
        )

    return build_experience_sequence(
        sequence_id=sequence_id,
        transformations=tuple(transformations),
    )


def build_sensitizing_pattern(
    pattern_id: str,
    sequence_id: str,
    event: ExperienceEvent,
    offset: float,
):
    states = (
        make_state(
            f"{sequence_id}_s0",
            cognitive=(0.10 + offset, 0.10 + offset, 0.10 + offset),
            physiological=(0.10 + offset, 0.08 + offset, 0.06 + offset),
            reserve=0.90 - offset,
        ),
        make_state(
            f"{sequence_id}_s1",
            cognitive=(0.20 + offset, 0.25 + offset, 0.18 + offset),
            physiological=(0.20 + offset, 0.18 + offset, 0.16 + offset),
            reserve=0.80 - offset,
        ),
        make_state(
            f"{sequence_id}_s2",
            cognitive=(0.38 + offset, 0.45 + offset, 0.36 + offset),
            physiological=(0.40 + offset, 0.36 + offset, 0.32 + offset),
            reserve=0.65 - offset,
        ),
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(
            build_sequence_from_states(
                sequence_id,
                event,
                states,
                magnitudes=(0.20 + offset, 0.45 + offset),
            ),
        ),
    )


def build_adaptation_pattern(
    pattern_id: str,
    sequence_id: str,
    event: ExperienceEvent,
):
    states = (
        make_state(
            f"{sequence_id}_s0",
            cognitive=(0.50, 0.50, 0.50),
            physiological=(0.50, 0.50, 0.50),
            reserve=0.40,
        ),
        make_state(
            f"{sequence_id}_s1",
            cognitive=(0.30, 0.30, 0.30),
            physiological=(0.28, 0.28, 0.28),
            reserve=0.65,
        ),
        make_state(
            f"{sequence_id}_s2",
            cognitive=(0.18, 0.18, 0.18),
            physiological=(0.16, 0.16, 0.16),
            reserve=0.85,
        ),
    )

    return build_experience_pattern(
        pattern_id=pattern_id,
        event_type=event.event_type,
        sequences=(
            build_sequence_from_states(
                sequence_id,
                event,
                states,
                magnitudes=(0.40, 0.30),
            ),
        ),
    )


@dataclass(frozen=True)
class BenchmarkWorld:
    event: ExperienceEvent
    attractors: tuple
    cue_door: ConditioningCue
    cue_safe: ConditioningCue
    sens_context: object
    adapt_context: object


def build_world() -> BenchmarkWorld:
    event = ExperienceEvent(
        event_id="event_repeat",
        event_type="repeated_cue",
        structural_vector=(1.0, 0.25, -0.1, 0.4),
    )

    sens_1 = build_sensitizing_pattern(
        "sens_1",
        "sens_seq_1",
        event,
        0.00,
    )
    sens_2 = build_sensitizing_pattern(
        "sens_2",
        "sens_seq_2",
        event,
        0.01,
    )
    adapt_1 = build_adaptation_pattern(
        "adapt_1",
        "adapt_seq_1",
        event,
    )
    adapt_2 = build_adaptation_pattern(
        "adapt_2",
        "adapt_seq_2",
        event,
    )

    attractors = (
        build_experience_attractor(
            attractor_id="sensitization_attractor",
            patterns=(sens_1, sens_2),
        ),
        build_experience_attractor(
            attractor_id="adaptation_attractor",
            patterns=(adapt_1, adapt_2),
        ),
    )

    midpoint = DynamicsState(
        state_id="midpoint",
        feature_vector=tuple(
            (a + b) / 2.0
            for a, b in zip(
                attractors[0].center,
                attractors[1].center,
            )
        ),
    )

    sens_bias = AttractorPrestress(
        prestress_id="sens_bias",
        bias_by_attractor_id={
            "sensitization_attractor": 0.75,
        },
    )
    adapt_bias = AttractorPrestress(
        prestress_id="adapt_bias",
        bias_by_attractor_id={
            "adaptation_attractor": 0.75,
        },
    )

    sens_training = build_attractor_trajectory(
        trajectory_id="sens_training",
        initial_state=midpoint,
        attractors=attractors,
        prestress_sequence=(sens_bias, sens_bias, sens_bias),
        step_size=0.10,
    )
    adapt_training = build_attractor_trajectory(
        trajectory_id="adapt_training",
        initial_state=midpoint,
        attractors=attractors,
        prestress_sequence=(adapt_bias, adapt_bias, adapt_bias),
        step_size=0.10,
    )

    cue_door = ConditioningCue(
        cue_id="cue_door",
        cue_type="visual_symbol",
        feature_vector=(1.0, 0.2, 0.1, 0.0),
    )
    cue_safe = ConditioningCue(
        cue_id="cue_safe",
        cue_type="visual_symbol",
        feature_vector=(0.1, 0.9, 0.8, 0.7),
    )

    sens_memory = build_conditioning_memory(
        memory_id="sens_memory",
        cue=cue_door,
        observations=tuple(
            observation_from_trajectory(
                observation_id=f"sens_obs_{i}",
                cue=cue_door,
                trajectory=sens_training,
                reinforcement_present=True,
                reinforcement_strength=1.0,
            )
            for i in range(5)
        ),
    )

    adapt_memory = build_conditioning_memory(
        memory_id="adapt_memory",
        cue=cue_safe,
        observations=tuple(
            observation_from_trajectory(
                observation_id=f"adapt_obs_{i}",
                cue=cue_safe,
                trajectory=adapt_training,
                reinforcement_present=True,
                reinforcement_strength=1.0,
            )
            for i in range(5)
        ),
    )

    sens_context = build_associative_context(
        context_id="sens_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_door,
                memory=sens_memory,
                salience=1.0,
            ),
        ),
    )

    adapt_context = build_associative_context(
        context_id="adapt_context",
        cue_inputs=(
            AssociativeCueInput(
                cue=cue_safe,
                memory=adapt_memory,
                salience=1.0,
            ),
        ),
    )

    return BenchmarkWorld(
        event=event,
        attractors=attractors,
        cue_door=cue_door,
        cue_safe=cue_safe,
        sens_context=sens_context,
        adapt_context=adapt_context,
    )


# ---------------------------------------------------------------------------
# Q1 — history dependence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryDependenceRow:
    feedback_gain: float
    history_label: str
    final_dominant: str
    final_vector: tuple[float, ...]
    terminal_drift: float
    regime: str
    convergence_score: float
    divergence_score: float


def run_history_dependence(
    world: BenchmarkWorld,
    feedback_gains: Sequence[float],
) -> tuple[HistoryDependenceRow, ...]:
    rows = []

    frames = tuple(
        FeedbackFrame(
            cue=world.cue_door,
            context=world.adapt_context,
        )
        for _ in range(4)
    )

    for gain in feedback_gains:
        config = ReconstructionFeedbackConfig(
            feedback_gain=float(gain),
            max_feedback_multiplier=100.0,
        )

        for label, seed in (
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
                trajectory_id=f"history::{gain}::{label}",
                frames=frames,
                attractors=world.attractors,
                config=config,
                seed_from_first_baseline=False,
                initial_reconstructed_vector=seed,
            )

            stability = analyze_feedback_stability(
                analysis_id=f"history_stability::{gain}::{label}",
                trajectory=trajectory,
            )

            rows.append(
                HistoryDependenceRow(
                    feedback_gain=float(gain),
                    history_label=label,
                    final_dominant=feedback_dominant_series(
                        trajectory
                    )[-1],
                    final_vector=trajectory.final_reconstructed_vector,
                    terminal_drift=stability.terminal_recursive_drift,
                    regime=stability.regime,
                    convergence_score=stability.convergence_score,
                    divergence_score=stability.divergence_score,
                )
            )

    return tuple(rows)


# ---------------------------------------------------------------------------
# Q2 — recursive stabilization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StabilizationRow:
    exposure_count: int
    regime: str
    switch_rate: float
    terminal_drift: float
    mean_recursive_drift: float
    recursive_drift_slope: float
    convergence_score: float
    final_dominant: str


def run_recursive_stabilization(
    world: BenchmarkWorld,
    exposure_counts: Sequence[int],
    feedback_gain: float = 1.0,
) -> tuple[StabilizationRow, ...]:
    rows = []

    for count in exposure_counts:
        if count < 2:
            raise ValueError("exposure_count must be >= 2")

        frames = tuple(
            FeedbackFrame(
                cue=world.cue_door,
                context=world.sens_context,
            )
            for _ in range(count)
        )

        trajectory = build_feedback_trajectory(
            trajectory_id=f"stabilization::{count}",
            frames=frames,
            attractors=world.attractors,
            config=ReconstructionFeedbackConfig(
                feedback_gain=feedback_gain,
            ),
        )

        stability = analyze_feedback_stability(
            analysis_id=f"stabilization_stability::{count}",
            trajectory=trajectory,
        )

        rows.append(
            StabilizationRow(
                exposure_count=count,
                regime=stability.regime,
                switch_rate=stability.switch_rate,
                terminal_drift=stability.terminal_recursive_drift,
                mean_recursive_drift=stability.mean_recursive_drift,
                recursive_drift_slope=stability.recursive_drift_slope,
                convergence_score=stability.convergence_score,
                final_dominant=feedback_dominant_series(
                    trajectory
                )[-1],
            )
        )

    return tuple(rows)


# ---------------------------------------------------------------------------
# Q3 — memory-state transition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryCycleRow:
    cycle_index: int
    source_revision: int
    target_revision: int
    source_trace_count: int
    target_trace_count: int
    committed_trace_id: str
    integration_relation: str
    cluster_hint: str
    consolidation_score: float
    commitment_confidence: float


def paper_cycle_config(
    feedback_gain: float = 1.0,
) -> ExperienceCycleConfig:
    return ExperienceCycleConfig(
        feedback=ReconstructionFeedbackConfig(
            feedback_gain=feedback_gain,
        ),
        consolidation=ExperienceConsolidationConfig(
            minimum_recursive_steps=3,
            minimum_stability_score=0.45,
            minimum_repetition_score=0.60,
            minimum_consistency_score=0.50,
            minimum_total_score=0.50,
            maximum_mean_ambiguity=0.60,
            maximum_terminal_drift=0.10,
        ),
        commitment=MemoryCommitmentConfig(
            minimum_candidate_score=0.50,
        ),
        integration=MemoryIntegrationConfig(),
    )


def run_memory_cycles(
    world: BenchmarkWorld,
    cycle_count: int = 3,
    feedback_gain: float = 1.0,
) -> tuple[MemoryCycleRow, ...]:
    if cycle_count < 1:
        raise ValueError("cycle_count must be >= 1")

    state = empty_memory_state(
        state_id="M0",
    )

    rows = []

    frames = tuple(
        FeedbackFrame(
            cue=world.cue_door,
            context=world.sens_context,
        )
        for _ in range(6)
    )

    config = paper_cycle_config(
        feedback_gain=feedback_gain,
    )

    for index in range(1, cycle_count + 1):
        result = run_experience_cycle(
            cycle_id=f"cycle_{index:03d}",
            source_memory_state=state,
            target_memory_state_id=f"M{index}",
            frames=frames,
            attractors=world.attractors,
            explicit_commit_requested=True,
            trace_id=f"trace_{index:03d}",
            candidate_id=f"candidate_{index:03d}",
            config=config,
        )

        rows.append(
            MemoryCycleRow(
                cycle_index=index,
                source_revision=result.source_memory_state.revision,
                target_revision=result.target_memory_state.revision,
                source_trace_count=result.source_memory_state.trace_count,
                target_trace_count=result.target_memory_state.trace_count,
                committed_trace_id=result.committed_trace.trace_id,
                integration_relation=result.integration.integration_relation,
                cluster_hint=result.integration.cluster_hint,
                consolidation_score=(
                    result.consolidation_candidate.consolidation_score
                ),
                commitment_confidence=(
                    result.committed_trace.commitment_confidence
                ),
            )
        )

        state = result.target_memory_state

    return tuple(rows)


# ---------------------------------------------------------------------------
# Q4 — additive vs multiplicative synthetic control
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropagationRow:
    model: str
    coupling: float
    steps: int
    final_total: float
    peak_total: float
    threshold_crossed: bool
    first_crossing_step: int | None


def additive_propagation(
    *,
    initial: float,
    coupling: float,
    input_drive: float,
    steps: int,
) -> tuple[float, ...]:
    x = float(initial)
    out = [x]

    for _ in range(steps):
        x = max(
            0.0,
            x + input_drive + coupling,
        )
        out.append(x)

    return tuple(out)


def multiplicative_propagation(
    *,
    initial: float,
    coupling: float,
    input_drive: float,
    steps: int,
) -> tuple[float, ...]:
    x = float(initial)
    out = [x]

    for _ in range(steps):
        x = max(
            0.0,
            (x + input_drive)
            * exp(coupling),
        )
        out.append(x)

    return tuple(out)


def threshold_summary(
    values: Sequence[float],
    threshold: float,
) -> tuple[bool, int | None]:
    for index, value in enumerate(values):
        if value >= threshold:
            return True, index
    return False, None


def run_additive_multiplicative_control(
    couplings: Sequence[float],
    *,
    initial: float = 0.05,
    input_drive: float = 0.02,
    steps: int = 12,
    threshold: float = 1.0,
) -> tuple[PropagationRow, ...]:
    rows = []

    for coupling in couplings:
        for model_name, fn in (
            ("additive", additive_propagation),
            ("multiplicative", multiplicative_propagation),
        ):
            series = fn(
                initial=initial,
                coupling=float(coupling),
                input_drive=input_drive,
                steps=steps,
            )

            crossed, first_crossing = threshold_summary(
                series,
                threshold,
            )

            rows.append(
                PropagationRow(
                    model=model_name,
                    coupling=float(coupling),
                    steps=steps,
                    final_total=series[-1],
                    peak_total=max(series),
                    threshold_crossed=crossed,
                    first_crossing_step=first_crossing,
                )
            )

    return tuple(rows)


# ---------------------------------------------------------------------------
# Aggregate benchmark
# ---------------------------------------------------------------------------

def history_pair_distances(
    rows: Sequence[HistoryDependenceRow],
) -> dict[str, float]:
    grouped: dict[float, dict[str, HistoryDependenceRow]] = {}

    for row in rows:
        grouped.setdefault(
            row.feedback_gain,
            {},
        )[row.history_label] = row

    result = {}

    for gain, pair in sorted(grouped.items()):
        if {
            "sens_history",
            "adapt_history",
        }.issubset(pair):
            result[str(gain)] = _distance(
                pair["sens_history"].final_vector,
                pair["adapt_history"].final_vector,
            )

    return result


def benchmark_summary(
    history_rows: Sequence[HistoryDependenceRow],
    stabilization_rows: Sequence[StabilizationRow],
    memory_rows: Sequence[MemoryCycleRow],
    propagation_rows: Sequence[PropagationRow],
) -> dict:
    history_distances = history_pair_distances(
        history_rows
    )

    additive = [
        row.final_total
        for row in propagation_rows
        if row.model == "additive"
    ]
    multiplicative = [
        row.final_total
        for row in propagation_rows
        if row.model == "multiplicative"
    ]

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "claim_scope": "computational_model_only",
        "clinical_validation_claimed": False,
        "causal_truth_claimed": False,
        "history_dependence": {
            "pair_distances_by_feedback_gain": history_distances,
            "nonzero_history_effect_count": sum(
                1
                for value in history_distances.values()
                if value > 1e-12
            ),
        },
        "recursive_stabilization": {
            "exposure_counts": [
                row.exposure_count
                for row in stabilization_rows
            ],
            "terminal_drifts": [
                row.terminal_drift
                for row in stabilization_rows
            ],
            "convergence_scores": [
                row.convergence_score
                for row in stabilization_rows
            ],
            "final_regimes": [
                row.regime
                for row in stabilization_rows
            ],
        },
        "memory_state_transition": {
            "cycles": len(memory_rows),
            "final_revision": (
                memory_rows[-1].target_revision
                if memory_rows
                else 0
            ),
            "final_trace_count": (
                memory_rows[-1].target_trace_count
                if memory_rows
                else 0
            ),
            "integration_relations": [
                row.integration_relation
                for row in memory_rows
            ],
        },
        "additive_vs_multiplicative": {
            "mean_final_additive": (
                mean(additive)
                if additive
                else 0.0
            ),
            "mean_final_multiplicative": (
                mean(multiplicative)
                if multiplicative
                else 0.0
            ),
            "note": (
                "This control is a matched synthetic comparison; "
                "it does not establish that biological cascades are multiplicative."
            ),
        },
    }


def run_benchmark() -> dict:
    world = build_world()

    history_rows = run_history_dependence(
        world,
        feedback_gains=(0.0, 1.0, 5.0, 20.0),
    )

    stabilization_rows = run_recursive_stabilization(
        world,
        exposure_counts=(2, 3, 4, 6, 8),
        feedback_gain=1.0,
    )

    memory_rows = run_memory_cycles(
        world,
        cycle_count=3,
        feedback_gain=1.0,
    )

    propagation_rows = run_additive_multiplicative_control(
        couplings=(0.02, 0.05, 0.10, 0.20),
        steps=12,
    )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "history_dependence_rows": [
            asdict(row)
            for row in history_rows
        ],
        "stabilization_rows": [
            asdict(row)
            for row in stabilization_rows
        ],
        "memory_cycle_rows": [
            asdict(row)
            for row in memory_rows
        ],
        "propagation_rows": [
            asdict(row)
            for row in propagation_rows
        ],
        "summary": benchmark_summary(
            history_rows,
            stabilization_rows,
            memory_rows,
            propagation_rows,
        ),
    }


def write_json(
    data: dict,
    path: Path,
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


def write_csv_rows(
    rows: Sequence[dict],
    path: Path,
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def save_benchmark(
    data: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        data,
        output_dir / "benchmark_results.json",
    )

    write_csv_rows(
        data["history_dependence_rows"],
        output_dir / "history_dependence.csv",
    )

    write_csv_rows(
        data["stabilization_rows"],
        output_dir / "recursive_stabilization.csv",
    )

    write_csv_rows(
        data["memory_cycle_rows"],
        output_dir / "memory_cycles.csv",
    )

    write_csv_rows(
        data["propagation_rows"],
        output_dir / "additive_vs_multiplicative.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ROIF Engine v1.0 end-to-end benchmark."
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for JSON/CSV benchmark outputs.",
    )

    args = parser.parse_args()

    data = run_benchmark()

    output_dir = Path(
        args.output_dir
    )

    save_benchmark(
        data,
        output_dir,
    )

    print(
        json.dumps(
            data["summary"],
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"\nSaved benchmark outputs to: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
