from __future__ import annotations

"""
ROIF Engine - Example 26
Inverse history reconstruction under noise and partial observability.

This experiment extends Example 25.

Instead of decoding an exact StructuralSignature, it deliberately perturbs the
observation by:

- adding deterministic profile noise;
- changing measured Capacity loss and gain;
- removing agent information;
- removing target information;
- masking part of the change-kind profile;
- combining noise with missing observations.

The hidden ground-truth class remains:

    Chronic repeated overload

The decoder receives only the degraded observed signature and the forward-
predicted signatures of admissible candidate histories.

This is a controlled synthetic robustness experiment. It measures internal
pipeline behavior under known perturbations. It is not evidence that arbitrary
real histories are uniquely identifiable.
"""

from dataclasses import dataclass, replace
from random import Random
from types import MappingProxyType
from typing import Final

from example_25_inverse_history_reconstruction import (
    build_forward_predictor,
    build_true_chronic_overload_pattern,
    make_candidate_rules,
)

from roif.history import (
    DecodeStatus,
    HistoryDecodeResult,
    HistoryDecoder,
    HypothesisGenerationResult,
    HypothesisGenerator,
    StructuralSignature,
)


SEPARATOR: Final[str] = "=" * 120
SUBSEPARATOR: Final[str] = "-" * 120

TRUE_HYPOTHESIS_NAME: Final[str] = "Chronic repeated overload"
RANDOM_SEED: Final[int] = 260826


@dataclass(frozen=True, slots=True)
class ObservationCondition:
    """One deterministic observation-degradation condition."""

    name: str
    description: str

    profile_noise: float = 0.0
    capacity_noise: float = 0.0

    drop_agent_profile: bool = False
    drop_target_profile: bool = False
    mask_remodeling_kind: bool = False
    mask_very_slow_time: bool = False


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Compact result of one robustness trial."""

    condition: ObservationCondition
    observed_signature: StructuralSignature
    generation: HypothesisGenerationResult
    decode_result: HistoryDecodeResult

    @property
    def recovered(self) -> bool:
        best = self.decode_result.best

        return (
            best is not None
            and best.name == TRUE_HYPOTHESIS_NAME
        )


def renormalize_profile(
    values: dict[str, float],
) -> dict[str, float]:
    """Remove non-positive values and normalize the remaining profile."""

    positive = {
        key: float(value)
        for key, value in values.items()
        if float(value) > 0.0
    }

    total = sum(positive.values())

    if total <= 0.0:
        return {}

    return {
        key: value / total
        for key, value in sorted(positive.items())
    }


def perturb_profile(
    profile: dict[str, float],
    *,
    noise_fraction: float,
    rng: Random,
) -> dict[str, float]:
    """
    Add bounded multiplicative noise and renormalize.

    Each profile component is multiplied by:

        1 + uniform(-noise_fraction, +noise_fraction)
    """

    if noise_fraction <= 0.0:
        return dict(profile)

    perturbed = {
        key: max(
            value
            * (
                1.0
                + rng.uniform(
                    -noise_fraction,
                    noise_fraction,
                )
            ),
            0.0,
        )
        for key, value in profile.items()
    }

    return renormalize_profile(perturbed)


def mask_profile_key(
    profile: dict[str, float],
    key: str,
) -> dict[str, float]:
    """Remove one observed category and renormalize the remaining profile."""

    masked = dict(profile)
    masked.pop(key, None)

    return renormalize_profile(masked)


def perturb_nonnegative(
    value: float,
    *,
    noise_fraction: float,
    rng: Random,
) -> float:
    """Apply bounded multiplicative noise to a non-negative scalar."""

    if noise_fraction <= 0.0:
        return value

    return max(
        value
        * (
            1.0
            + rng.uniform(
                -noise_fraction,
                noise_fraction,
            )
        ),
        0.0,
    )


def build_degraded_observation(
    base: StructuralSignature,
    condition: ObservationCondition,
    *,
    seed: int,
) -> StructuralSignature:
    """
    Build one degraded but internally valid StructuralSignature.

    The observation remains immutable and satisfies all StructuralSignature
    invariants.
    """

    rng = Random(seed)

    plane_profile = perturb_profile(
        dict(base.plane_profile),
        noise_fraction=condition.profile_noise,
        rng=rng,
    )
    agent_profile = perturb_profile(
        dict(base.agent_profile),
        noise_fraction=condition.profile_noise,
        rng=rng,
    )
    kind_profile = perturb_profile(
        dict(base.kind_profile),
        noise_fraction=condition.profile_noise,
        rng=rng,
    )
    time_scale_profile = perturb_profile(
        dict(base.time_scale_profile),
        noise_fraction=condition.profile_noise,
        rng=rng,
    )
    target_profile = perturb_profile(
        dict(base.target_profile),
        noise_fraction=condition.profile_noise,
        rng=rng,
    )

    if condition.drop_agent_profile:
        agent_profile = {}

    if condition.drop_target_profile:
        target_profile = {}

    if condition.mask_remodeling_kind:
        kind_profile = mask_profile_key(
            kind_profile,
            "remodeling",
        )

    if condition.mask_very_slow_time:
        time_scale_profile = mask_profile_key(
            time_scale_profile,
            "very_slow",
        )

    total_capacity_loss = perturb_nonnegative(
        base.total_capacity_loss,
        noise_fraction=condition.capacity_noise,
        rng=rng,
    )
    total_capacity_gain = perturb_nonnegative(
        base.total_capacity_gain,
        noise_fraction=condition.capacity_noise,
        rng=rng,
    )
    net_capacity_effect = (
        total_capacity_gain
        - total_capacity_loss
    )

    metadata = {
        **dict(base.metadata),
        "example": 26,
        "condition": condition.name,
        "condition_description": condition.description,
        "profile_noise": condition.profile_noise,
        "capacity_noise": condition.capacity_noise,
        "drop_agent_profile": condition.drop_agent_profile,
        "drop_target_profile": condition.drop_target_profile,
        "mask_remodeling_kind": condition.mask_remodeling_kind,
        "mask_very_slow_time": condition.mask_very_slow_time,
        "deterministic_seed": seed,
    }

    return replace(
        base,
        signature_id=f"observed-{condition.name}",
        label=f"Observed: {condition.name}",
        plane_profile=plane_profile,
        agent_profile=agent_profile,
        kind_profile=kind_profile,
        time_scale_profile=time_scale_profile,
        target_profile=target_profile,
        total_capacity_loss=total_capacity_loss,
        total_capacity_gain=total_capacity_gain,
        net_capacity_effect=net_capacity_effect,
        metadata=MappingProxyType(metadata),
    )


def make_conditions() -> tuple[ObservationCondition, ...]:
    """Return the deterministic robustness matrix."""

    return (
        ObservationCondition(
            name="baseline",
            description="Exact observation from Example 25.",
        ),
        ObservationCondition(
            name="moderate_noise",
            description=(
                "Ten-percent profile noise and eight-percent "
                "Capacity measurement noise."
            ),
            profile_noise=0.10,
            capacity_noise=0.08,
        ),
        ObservationCondition(
            name="high_noise",
            description=(
                "Twenty-five-percent profile noise and twenty-percent "
                "Capacity measurement noise."
            ),
            profile_noise=0.25,
            capacity_noise=0.20,
        ),
        ObservationCondition(
            name="unknown_agents",
            description=(
                "The complete agent profile is unavailable."
            ),
            drop_agent_profile=True,
        ),
        ObservationCondition(
            name="unknown_targets",
            description=(
                "The complete target-location profile is unavailable."
            ),
            drop_target_profile=True,
        ),
        ObservationCondition(
            name="masked_secondary_history",
            description=(
                "Remodeling and very-slow observations are not recorded."
            ),
            mask_remodeling_kind=True,
            mask_very_slow_time=True,
        ),
        ObservationCondition(
            name="combined_degradation",
            description=(
                "Fifteen-percent profile noise, twelve-percent Capacity "
                "noise, unknown agents, and unknown targets."
            ),
            profile_noise=0.15,
            capacity_noise=0.12,
            drop_agent_profile=True,
            drop_target_profile=True,
        ),
    )


def run_trial(
    observed: StructuralSignature,
    condition: ObservationCondition,
    *,
    generator: HypothesisGenerator,
    decoder: HistoryDecoder,
) -> TrialResult:
    """Generate, predict, and decode one degraded observation."""

    generation = generator.generate(
        observed,
        build_forward_predictor(),
        metadata={
            "example": 26,
            "condition": condition.name,
        },
    )

    decode_result = decoder.decode(
        observed,
        generation.hypotheses,
        metadata={
            "example": 26,
            "condition": condition.name,
            "ground_truth_not_supplied_to_decoder": True,
        },
    )

    return TrialResult(
        condition=condition,
        observed_signature=observed,
        generation=generation,
        decode_result=decode_result,
    )


def main_difference_text(
    trial: TrialResult,
) -> str:
    """Return the largest signature mismatch of the winning hypothesis."""

    best = trial.decode_result.best

    if best is None:
        return "none"

    difference = (
        best.comparison.most_different_component
    )

    if (
        difference is None
        or difference[1] <= 1e-12
    ):
        return "none"

    return (
        f"{difference[0]}="
        f"{difference[1]:.3f}"
    )


def print_condition_details(
    trial: TrialResult,
) -> None:
    """Print a detailed report for one trial."""

    observed = trial.observed_signature
    result = trial.decode_result
    best = result.best

    print()
    print(f"Condition: {trial.condition.name}")
    print(SUBSEPARATOR)
    print(trial.condition.description)
    print()

    print(
        "Observed profiles  : "
        f"planes={len(observed.plane_profile)}, "
        f"agents={len(observed.agent_profile)}, "
        f"kinds={len(observed.kind_profile)}, "
        f"times={len(observed.time_scale_profile)}, "
        f"targets={len(observed.target_profile)}"
    )
    print(
        "Observed Capacity  : "
        f"loss={observed.total_capacity_loss:.6f}, "
        f"gain={observed.total_capacity_gain:.6f}, "
        f"net={observed.net_capacity_effect:+.6f}"
    )
    print(
        "Generated          : "
        f"{trial.generation.predicted_count} predicted, "
        f"{trial.generation.failed_count} failed"
    )
    print(f"Decode status      : {result.status.value}")
    print(f"Confidence         : {result.confidence:.6f}")
    print(
        "Selection margin   : "
        f"{result.selection_margin:.6f}"
    )
    print(
        "Alternative prox.  : "
        f"{result.ambiguity:.6f}"
    )

    if best is None:
        print("Best hypothesis    : none")
        print("Recovered          : NO")
        return

    print(f"Best hypothesis    : {best.name}")
    print(f"Fit                : {best.fit_score:.6f}")
    print(
        "Main difference    : "
        f"{main_difference_text(trial)}"
    )
    print(
        "Recovered          : "
        f"{'YES' if trial.recovered else 'NO'}"
    )


def print_summary(
    trials: tuple[TrialResult, ...],
) -> None:
    """Print a compact robustness table."""

    print()
    print("Robustness summary")
    print(SUBSEPARATOR)
    print(
        f"{'Condition':<28}  "
        f"{'Candidates':>12}  "
        f"{'Best hypothesis':<32}  "
        f"{'Fit':>10}  "
        f"{'Margin':>10}  "
        f"{'Status':>14}  "
        f"{'Recovered':>12}"
    )
    print(SUBSEPARATOR)

    for trial in trials:
        result = trial.decode_result
        best = result.best

        if best is None:
            best_name = "none"
            fit = 0.0
        else:
            best_name = best.name
            fit = best.fit_score

        print(
            f"{trial.condition.name:<28}  "
            f"{len(result.rankings):>12}  "
            f"{best_name:<32}  "
            f"{fit:>10.6f}  "
            f"{result.selection_margin:>10.6f}  "
            f"{result.status.value:>14}  "
            f"{('YES' if trial.recovered else 'NO'):>12}"
        )

    recovered_count = sum(
        trial.recovered
        for trial in trials
    )
    total_count = len(trials)
    accuracy = (
        recovered_count / total_count
        if total_count
        else 0.0
    )

    print(SUBSEPARATOR)
    print(
        "Top-1 recovery     : "
        f"{recovered_count}/{total_count} "
        f"({accuracy:.1%})"
    )

    mean_fit = (
        sum(
            (
                trial.decode_result.best.fit_score
                if trial.decode_result.best is not None
                else 0.0
            )
            for trial in trials
        )
        / total_count
    )

    mean_margin = (
        sum(
            trial.decode_result.selection_margin
            for trial in trials
        )
        / total_count
    )

    print(f"Mean winning fit    : {mean_fit:.6f}")
    print(f"Mean select. margin : {mean_margin:.6f}")


def main() -> None:
    print(SEPARATOR)
    print("ROIF Engine - Example 26")
    print(
        "Inverse history reconstruction under noise "
        "and partial observability"
    )
    print(SEPARATOR)
    print()

    print("Experiment design")
    print(SUBSEPARATOR)
    print(
        "1. Build the known chronic-overload ground-truth history.\n"
        "2. Convert it into a StructuralSignature.\n"
        "3. Degrade the observation with deterministic noise or missing data.\n"
        "4. Generate admissible candidate histories for each observation.\n"
        "5. Reconstruct every candidate forward.\n"
        "6. Decode and record whether the true history class remains top-ranked."
    )
    print()
    print(
        "Metric note: confidence, alternative proximity, and softmax-derived "
        "values are internal decoder measures. They are not calibrated "
        "probabilities of historical truth."
    )

    true_pattern = (
        build_true_chronic_overload_pattern()
    )
    exact_signature = StructuralSignature.from_pattern(
        true_pattern,
        label="Exact hidden-history signature",
        metadata={
            "ground_truth_hidden_from_decoder": True,
        },
        signature_id="exact-hidden-history",
    )

    generator = HypothesisGenerator(
        rules=make_candidate_rules(),
        max_candidates=10,
        fail_fast=True,
    )
    decoder = HistoryDecoder(
        ambiguity_threshold=0.05,
        complexity_scale=5.0,
        minimum_score=0.0,
    )

    trials: list[TrialResult] = []

    for index, condition in enumerate(
        make_conditions(),
        start=1,
    ):
        observed = build_degraded_observation(
            exact_signature,
            condition,
            seed=RANDOM_SEED + index,
        )

        trial = run_trial(
            observed,
            condition,
            generator=generator,
            decoder=decoder,
        )
        trials.append(trial)
        print_condition_details(trial)

    trial_tuple = tuple(trials)
    print_summary(trial_tuple)

    failed_conditions = [
        trial.condition.name
        for trial in trial_tuple
        if not trial.recovered
    ]

    print()
    print("Interpretation")
    print(SUBSEPARATOR)

    if failed_conditions:
        print(
            "The true class was not top-ranked under every condition."
        )
        print(
            "Failed conditions   : "
            + ", ".join(failed_conditions)
        )
        print(
            "These failures are informative robustness limits, not "
            "software errors."
        )
    else:
        print(
            "The true history class remained top-ranked in all configured "
            "noise and missing-data conditions."
        )

    print(
        "This controlled result measures pipeline robustness only within "
        "the explicitly defined candidate library and perturbation model."
    )
    print(
        "A publication-grade claim requires repeated randomized trials, "
        "held-out scenarios, calibrated uncertainty, baselines, and "
        "confidence intervals."
    )
    print()
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()



