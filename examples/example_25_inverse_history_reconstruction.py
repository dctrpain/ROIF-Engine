from __future__ import annotations

"""
ROIF Engine - Example 25
Inverse reconstruction of structural history.

The experiment demonstrates the complete Structural Memory pipeline:

    known hidden history
        -> irreversible changes
        -> history pattern
        -> observed structural signature
        -> causal-rule matching
        -> candidate generation
        -> forward prediction
        -> history decoding
        -> ranked causal alternatives

This is a controlled synthetic experiment. The true history is known to the
experiment but is not passed directly to HistoryDecoder.

The purpose is not to prove that arbitrary real histories are uniquely
recoverable. The purpose is to verify that the pipeline ranks the candidate
whose forward-predicted structural signature best matches the observation.
"""

from collections.abc import Callable
from typing import Final

from roif.history import (
    CausalRule,
    DecodeStatus,
    FunctionalEffect,
    HistoryDecoder,
    HistoryPattern,
    HistoryTarget,
    HistoryTargetKind,
    HypothesisGenerator,
    HypothesisSeed,
    IrreversibleChange,
    IrreversibleChangeKind,
    ReversibilityClass,
    RuleMatchMode,
    StateDelta,
    StructuralSignature,
    TimeScale,
    TracePersistence,
)


SEPARATOR: Final[str] = "=" * 112
SUBSEPARATOR: Final[str] = "-" * 112


def make_target(
    target_id: str,
    label: str,
) -> HistoryTarget:
    """Create a stable structural target."""

    return HistoryTarget(
        kind=HistoryTargetKind.ELEMENT,
        target_id=target_id,
        label=label,
        metadata={
            "model": "synthetic upper-limb chain",
        },
    )


def make_change(
    *,
    change_id: str,
    source_event_id: str,
    kind: IrreversibleChangeKind,
    target: HistoryTarget,
    quantity: str,
    before: float,
    after: float,
    onset_time: float,
    recorded_time: float,
    retained_fraction: float,
    permanence: TracePersistence,
    characteristic_time: float,
    time_scale: TimeScale,
    memory_strength: float,
    capacity_effect: float,
    functional_effect: FunctionalEffect,
    reversibility: ReversibilityClass,
    cause_change_ids: tuple[str, ...] = (),
    plane_ids: tuple[str, ...] = (),
    agent_ids: tuple[str, ...] = (),
    description: str | None = None,
) -> IrreversibleChange:
    """Create one persistent structural trace."""

    return IrreversibleChange(
        change_id=change_id,
        kind=kind,
        source_event_id=source_event_id,
        target=target,
        delta=StateDelta(
            quantity=quantity,
            before=before,
            after=after,
            units="normalized",
        ),
        onset_time=onset_time,
        recorded_time=recorded_time,
        retained_fraction=retained_fraction,
        permanence=permanence,
        characteristic_time=characteristic_time,
        time_scale=time_scale,
        memory_strength=memory_strength,
        capacity_effect=capacity_effect,
        functional_effect=functional_effect,
        reversibility=reversibility,
        cause_change_ids=cause_change_ids,
        plane_ids=plane_ids,
        agent_ids=agent_ids,
        description=description,
    )


def build_true_chronic_overload_pattern() -> HistoryPattern:
    """
    Build the hidden ground-truth history.

    Scenario:
        repeated mechanical overload
            -> accumulated fatigue
            -> local damage
            -> compensatory remodeling
    """

    tendon = make_target(
        "TENDON_DISTAL",
        "Distal tendon analogue",
    )
    joint = make_target(
        "JOINT_SUPPORT",
        "Joint-support analogue",
    )
    compensator = make_target(
        "COMPENSATOR",
        "Compensating soft-tissue element",
    )

    overload = make_change(
        change_id="true-overload",
        source_event_id="event-repeated-load",
        kind=IrreversibleChangeKind.FATIGUE,
        target=tendon,
        quantity="fatigue",
        before=0.05,
        after=0.42,
        onset_time=10.0,
        recorded_time=100.0,
        retained_fraction=0.90,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=86_400.0,
        time_scale=TimeScale.SLOW,
        memory_strength=0.92,
        capacity_effect=-0.12,
        functional_effect=FunctionalEffect.HARMFUL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        plane_ids=(
            "mechanical",
            "metabolic",
        ),
        agent_ids=(
            "repeated_load",
        ),
        description=(
            "Repeated subcritical loading produced accumulated fatigue."
        ),
    )

    damage = make_change(
        change_id="true-damage",
        source_event_id="event-fatigue-damage",
        kind=IrreversibleChangeKind.DAMAGE,
        target=joint,
        quantity="damage",
        before=0.02,
        after=0.31,
        onset_time=100.0,
        recorded_time=180.0,
        retained_fraction=0.82,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=172_800.0,
        time_scale=TimeScale.SLOW,
        memory_strength=0.88,
        capacity_effect=-0.18,
        functional_effect=FunctionalEffect.HARMFUL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        cause_change_ids=(
            "true-overload",
        ),
        plane_ids=(
            "mechanical",
            "material",
        ),
        agent_ids=(
            "repeated_load",
        ),
        description=(
            "Fatigue lowered reserve until persistent local damage formed."
        ),
    )

    remodeling = make_change(
        change_id="true-remodeling",
        source_event_id="event-compensation",
        kind=IrreversibleChangeKind.REMODELING,
        target=compensator,
        quantity="stiffness",
        before=0.50,
        after=0.68,
        onset_time=180.0,
        recorded_time=420.0,
        retained_fraction=0.76,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=604_800.0,
        time_scale=TimeScale.VERY_SLOW,
        memory_strength=0.84,
        capacity_effect=0.05,
        functional_effect=FunctionalEffect.MIXED,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        cause_change_ids=(
            "true-damage",
        ),
        plane_ids=(
            "mechanical",
            "biological",
        ),
        agent_ids=(
            "internal_adaptation",
        ),
        description=(
            "Compensatory remodeling improved local support but preserved "
            "the historical load redistribution."
        ),
    )

    return HistoryPattern(
        pattern_id="hidden-true-history",
        changes=(
            remodeling,
            overload,
            damage,
        ),
        label="Hidden chronic overload history",
        description=(
            "Ground-truth history used only to generate the observation."
        ),
        metadata={
            "ground_truth": True,
            "scenario": "chronic_overload",
        },
    )


def build_single_impact_pattern() -> HistoryPattern:
    """Build a competing single-impact hypothesis."""

    contact = make_target(
        "JOINT_SUPPORT",
        "Joint-support analogue",
    )
    adjacent = make_target(
        "COMPENSATOR",
        "Compensating soft-tissue element",
    )

    impact_damage = make_change(
        change_id="impact-damage",
        source_event_id="event-single-impact",
        kind=IrreversibleChangeKind.DAMAGE,
        target=contact,
        quantity="damage",
        before=0.0,
        after=0.38,
        onset_time=10.0,
        recorded_time=10.2,
        retained_fraction=0.88,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=0.05,
        time_scale=TimeScale.FAST,
        memory_strength=0.90,
        capacity_effect=-0.27,
        functional_effect=FunctionalEffect.HARMFUL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        plane_ids=(
            "mechanical",
        ),
        agent_ids=(
            "steel_object",
        ),
        description=(
            "One strong mechanical impact produced immediate local damage."
        ),
    )

    residual = make_change(
        change_id="impact-residual",
        source_event_id="event-residual-strain",
        kind=IrreversibleChangeKind.RESIDUAL_STRAIN,
        target=adjacent,
        quantity="residual_strain",
        before=0.0,
        after=0.18,
        onset_time=10.2,
        recorded_time=15.0,
        retained_fraction=0.68,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=3600.0,
        time_scale=TimeScale.MEDIUM,
        memory_strength=0.74,
        capacity_effect=-0.06,
        functional_effect=FunctionalEffect.HARMFUL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        cause_change_ids=(
            "impact-damage",
        ),
        plane_ids=(
            "mechanical",
        ),
        agent_ids=(
            "steel_object",
        ),
        description=(
            "Residual strain remained after the isolated impact."
        ),
    )

    return HistoryPattern(
        pattern_id="candidate-single-impact",
        changes=(
            residual,
            impact_damage,
        ),
        label="Single mechanical impact",
        metadata={
            "scenario": "single_impact",
        },
    )


def build_thermal_degradation_pattern() -> HistoryPattern:
    """Build a competing thermal-degradation hypothesis."""

    tissue = make_target(
        "TENDON_DISTAL",
        "Distal tendon analogue",
    )
    support = make_target(
        "JOINT_SUPPORT",
        "Joint-support analogue",
    )

    thermal_change = make_change(
        change_id="thermal-aging",
        source_event_id="event-thermal-cycle",
        kind=IrreversibleChangeKind.AGING,
        target=tissue,
        quantity="material_integrity",
        before=0.95,
        after=0.70,
        onset_time=10.0,
        recorded_time=220.0,
        retained_fraction=0.80,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=259_200.0,
        time_scale=TimeScale.SLOW,
        memory_strength=0.82,
        capacity_effect=-0.14,
        functional_effect=FunctionalEffect.HARMFUL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        plane_ids=(
            "thermal",
            "material",
        ),
        agent_ids=(
            "temperature_cycle",
        ),
        description=(
            "Repeated thermal cycling degraded material integrity."
        ),
    )

    stiffness_change = make_change(
        change_id="thermal-stiffness",
        source_event_id="event-stiffness-drift",
        kind=IrreversibleChangeKind.STIFFNESS_CHANGE,
        target=support,
        quantity="stiffness",
        before=0.65,
        after=0.48,
        onset_time=220.0,
        recorded_time=400.0,
        retained_fraction=0.72,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=604_800.0,
        time_scale=TimeScale.VERY_SLOW,
        memory_strength=0.78,
        capacity_effect=-0.10,
        functional_effect=FunctionalEffect.HARMFUL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        cause_change_ids=(
            "thermal-aging",
        ),
        plane_ids=(
            "thermal",
            "material",
        ),
        agent_ids=(
            "temperature_cycle",
        ),
        description=(
            "Long-term thermal exposure changed effective stiffness."
        ),
    )

    return HistoryPattern(
        pattern_id="candidate-thermal-degradation",
        changes=(
            stiffness_change,
            thermal_change,
        ),
        label="Thermal degradation",
        metadata={
            "scenario": "thermal_degradation",
        },
    )


def build_adaptive_training_pattern() -> HistoryPattern:
    """Build a competing beneficial-adaptation hypothesis."""

    muscle = make_target(
        "COMPENSATOR",
        "Compensating soft-tissue element",
    )
    tendon = make_target(
        "TENDON_DISTAL",
        "Distal tendon analogue",
    )

    hypertrophy = make_change(
        change_id="training-hypertrophy",
        source_event_id="event-training-load",
        kind=IrreversibleChangeKind.HYPERTROPHY,
        target=muscle,
        quantity="capacity",
        before=0.60,
        after=0.78,
        onset_time=10.0,
        recorded_time=200.0,
        retained_fraction=0.84,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=604_800.0,
        time_scale=TimeScale.VERY_SLOW,
        memory_strength=0.86,
        capacity_effect=0.16,
        functional_effect=FunctionalEffect.BENEFICIAL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        plane_ids=(
            "mechanical",
            "metabolic",
            "biological",
        ),
        agent_ids=(
            "training_load",
        ),
        description=(
            "Progressive training produced beneficial capacity adaptation."
        ),
    )

    tendon_adaptation = make_change(
        change_id="training-remodeling",
        source_event_id="event-tendon-adaptation",
        kind=IrreversibleChangeKind.REMODELING,
        target=tendon,
        quantity="stiffness",
        before=0.55,
        after=0.66,
        onset_time=200.0,
        recorded_time=420.0,
        retained_fraction=0.78,
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=1_209_600.0,
        time_scale=TimeScale.VERY_SLOW,
        memory_strength=0.80,
        capacity_effect=0.09,
        functional_effect=FunctionalEffect.BENEFICIAL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        cause_change_ids=(
            "training-hypertrophy",
        ),
        plane_ids=(
            "mechanical",
            "biological",
        ),
        agent_ids=(
            "training_load",
        ),
        description=(
            "Tendon remodeling followed repeated adaptive loading."
        ),
    )

    return HistoryPattern(
        pattern_id="candidate-adaptive-training",
        changes=(
            tendon_adaptation,
            hypertrophy,
        ),
        label="Adaptive training history",
        metadata={
            "scenario": "adaptive_training",
        },
    )


def make_candidate_rules() -> tuple[CausalRule, ...]:
    """Create explicit admissibility rules for competing histories."""

    return (
        CausalRule(
            rule_id="rule-chronic-overload",
            name="Chronic overload admissibility rule",
            hypothesis_name="Chronic repeated overload",
            description=(
                "Repeated loading with fatigue, damage, and later "
                "compensatory remodeling."
            ),
            initiating_plane_ids=(
                "mechanical",
                "metabolic",
            ),
            initiating_agent_ids=(
                "repeated_load",
            ),
            candidate_root_ids=(
                "TENDON_DISTAL",
            ),
            plane_thresholds={
                "mechanical": 0.20,
            },
            kind_thresholds={
                "fatigue": 0.05,
            },
            time_scale_thresholds={
                "slow": 0.05,
            },
            minimum_total_changes=3,
            minimum_causal_depth=3,
            minimum_capacity_loss=0.20,
            minimum_persistence=0.50,
            match_mode=RuleMatchMode.ALL,
            prior_probability=0.55,
            validation_score=0.95,
            complexity=3.0,
        ),
        CausalRule(
            rule_id="rule-single-impact",
            name="Single impact admissibility rule",
            hypothesis_name="Single mechanical impact",
            description=(
                "One short, strong mechanical event with residual strain."
            ),
            initiating_plane_ids=(
                "mechanical",
            ),
            initiating_agent_ids=(
                "steel_object",
            ),
            candidate_root_ids=(
                "JOINT_SUPPORT",
            ),
            plane_thresholds={
                "mechanical": 0.20,
            },
            kind_thresholds={
                "damage": 0.05,
            },
            minimum_total_changes=1,
            minimum_capacity_loss=0.10,
            match_mode=RuleMatchMode.ALL,
            prior_probability=0.60,
            validation_score=0.90,
            complexity=1.0,
        ),
        CausalRule(
            rule_id="rule-thermal-degradation",
            name="Thermal degradation admissibility rule",
            hypothesis_name="Thermal degradation",
            description=(
                "Slow material degradation caused by repeated thermal cycles."
            ),
            initiating_plane_ids=(
                "thermal",
            ),
            initiating_agent_ids=(
                "temperature_cycle",
            ),
            candidate_root_ids=(
                "TENDON_DISTAL",
            ),
            plane_thresholds={
                "thermal": 0.05,
            },
            minimum_total_changes=2,
            minimum_persistence=0.40,
            match_mode=RuleMatchMode.ALL,
            prior_probability=0.30,
            validation_score=0.85,
            complexity=2.0,
        ),
        CausalRule(
            rule_id="rule-adaptive-training",
            name="Adaptive training admissibility rule",
            hypothesis_name="Adaptive training",
            description=(
                "Beneficial structural adaptation to progressive loading."
            ),
            initiating_plane_ids=(
                "mechanical",
                "biological",
            ),
            initiating_agent_ids=(
                "training_load",
            ),
            candidate_root_ids=(
                "COMPENSATOR",
            ),
            plane_thresholds={
                "mechanical": 0.10,
            },
            minimum_total_changes=2,
            match_mode=RuleMatchMode.ANY,
            prior_probability=0.45,
            validation_score=0.90,
            complexity=2.0,
        ),
    )


def build_forward_predictor() -> Callable[
    [HypothesisSeed, StructuralSignature],
    StructuralSignature,
]:
    """
    Return a controlled forward predictor.

    In a later experiment each branch will run a real ROIF simulation.
    Here each candidate rule maps to an explicit independently constructed
    HistoryPattern, from which its predicted signature is calculated.
    """

    builders: dict[str, Callable[[], HistoryPattern]] = {
        "rule-chronic-overload": (
            build_true_chronic_overload_pattern
        ),
        "rule-single-impact": build_single_impact_pattern,
        "rule-thermal-degradation": (
            build_thermal_degradation_pattern
        ),
        "rule-adaptive-training": (
            build_adaptive_training_pattern
        ),
    }

    def predictor(
        seed: HypothesisSeed,
        observed: StructuralSignature,
    ) -> StructuralSignature:
        try:
            pattern_builder = builders[seed.rule_id]
        except KeyError as exc:
            raise ValueError(
                f"No forward model registered for rule "
                f"{seed.rule_id!r}"
            ) from exc

        pattern = pattern_builder()

        return StructuralSignature.from_pattern(
            pattern,
            label=f"Predicted: {seed.name}",
            metadata={
                "forward_model": (
                    "controlled_pattern_reconstruction"
                ),
                "seed_id": seed.seed_id,
                "rule_id": seed.rule_id,
                "observation_used_only_for_context": (
                    observed.signature_id
                ),
            },
            signature_id=f"predicted-{seed.rule_id}",
        )

    return predictor


def print_signature(
    title: str,
    signature: StructuralSignature,
) -> None:
    """Print the most relevant signature fields."""

    print(title)
    print(SUBSEPARATOR)
    print(f"Signature ID       : {signature.signature_id}")
    print(f"Source pattern     : {signature.source_pattern_id}")
    print(f"Changes            : {signature.total_changes}")
    print(f"Causal depth       : {signature.causal_depth}")
    print(f"Duration           : {signature.duration:.3f}")
    print(
        "Capacity loss      : "
        f"{signature.total_capacity_loss:.6f}"
    )
    print(
        "Capacity gain      : "
        f"{signature.total_capacity_gain:.6f}"
    )
    print(
        "Net capacity       : "
        f"{signature.net_capacity_effect:+.6f}"
    )
    print(
        "Persistence index  : "
        f"{signature.persistence_index:.6f}"
    )
    print(
        "Irreversibility    : "
        f"{signature.irreversibility_index:.6f}"
    )
    print(
        "Progression index  : "
        f"{signature.progression_index:.6f}"
    )
    print(
        "Adaptation index   : "
        f"{signature.adaptation_index:.6f}"
    )
    print(
        "Dominant plane     : "
        f"{signature.dominant_plane}"
    )
    print(
        "Dominant kind      : "
        f"{signature.dominant_kind}"
    )
    print(
        "Dominant time      : "
        f"{signature.dominant_time_scale}"
    )


def main() -> None:
    print(SEPARATOR)
    print("ROIF Engine - Example 25")
    print("Inverse structural-history reconstruction")
    print(SEPARATOR)
    print()

    print("Experiment design")
    print(SUBSEPARATOR)
    print(
        "1. Create a known chronic-overload history.\n"
        "2. Convert it into an observed StructuralSignature.\n"
        "3. Do not pass its identity to HistoryDecoder.\n"
        "4. Generate several competing causal hypotheses.\n"
        "5. Reconstruct every hypothesis forward.\n"
        "6. Rank predicted signatures against the observation."
    )
    print()

    true_pattern = build_true_chronic_overload_pattern()

    observed_signature = StructuralSignature.from_pattern(
        true_pattern,
        label="Observed anonymous structural signature",
        metadata={
            "ground_truth_hidden_from_decoder": True,
        },
        signature_id="observed-unknown-history",
    )

    print_signature(
        "Observed structural signature",
        observed_signature,
    )
    print()

    generator = HypothesisGenerator(
        rules=make_candidate_rules(),
        max_candidates=10,
        fail_fast=True,
    )

    matched_rules = generator.matching_rules(
        observed_signature,
    )

    print("Admissible causal rules")
    print(SUBSEPARATOR)

    for index, rule in enumerate(
        matched_rules,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{rule.hypothesis_name:<32} "
            f"prior={rule.prior_probability:.3f} "
            f"complexity={rule.complexity:.3f}"
        )

    print()

    generation = generator.generate(
        observed_signature,
        build_forward_predictor(),
        metadata={
            "example": 25,
            "experiment": (
                "inverse_history_reconstruction"
            ),
        },
    )

    print("Forward reconstruction")
    print(SUBSEPARATOR)
    print(f"Matched rules      : {len(matched_rules)}")
    print(f"Predicted histories: {generation.predicted_count}")
    print(f"Failed predictions : {generation.failed_count}")
    print()

    decoder = HistoryDecoder(
        ambiguity_threshold=0.05,
        complexity_scale=5.0,
        minimum_score=0.0,
    )

    result = decoder.decode(
        observed_signature,
        generation.hypotheses,
        metadata={
            "ground_truth_not_supplied_to_decoder": True,
        },
    )

    probabilities = decoder.softmax_probabilities(
        result,
        temperature=0.25,
    )

    print("Decoder ranking")
    print(SUBSEPARATOR)
    print(
        f"{'Rank':<6}"
        f"{'Hypothesis':<34}"
        f"{'Fit':>10}"
        f"{'Score':>10}"
        f"{'Probability':>14}  "
        f"{'Main difference':>30}"
    )
    print(SUBSEPARATOR)

    for ranked in result.rankings:
        most_different = (
            ranked.comparison.most_different_component
        )

        if (
            most_different is None
            or most_different[1] <= 1e-12
        ):
            difference_text = "none"
        else:
            difference_text = (
                f"{most_different[0]}="
                f"{most_different[1]:.3f}"
            )

        probability = probabilities.get(
            ranked.hypothesis_id,
            0.0,
        )

        print(
            f"{ranked.rank:<6}"
            f"{ranked.name:<34}"
            f"{ranked.fit_score:>10.6f}"
            f"{ranked.normalized_score:>10.6f}"
            f"{probability:>14.6f}  "
            f"{difference_text:>30}"
        )

    print()
    print("Decode result")
    print(SUBSEPARATOR)
    print(f"Status               : {result.status.value}")
    print(
        "Decoder confidence   : "
        f"{result.confidence:.6f}"
    )
    print(
        "Selection margin     : "
        f"{result.selection_margin:.6f}"
    )
    print(
        "Alternative proximity: "
        f"{result.ambiguity:.6f}"
    )

    if result.best is None:
        print("Best hypothesis      : none")
        raise RuntimeError(
            "Decoder returned no ranked hypothesis."
        )

    print(f"Best hypothesis      : {result.best.name}")
    print(
        "Best hypothesis ID   : "
        f"{result.best.hypothesis_id}"
    )
    print()
    print(
        "Metric note          : confidence and alternative "
        "proximity are internal decoder scores, not calibrated "
        "probabilities of historical truth."
    )
    print()

    expected_name = "Chronic repeated overload"
    recovered = result.best.name == expected_name

    print("Ground-truth check")
    print(SUBSEPARATOR)
    print(f"Hidden true history: {expected_name}")
    print(f"Decoder selection  : {result.best.name}")
    print(
        "Recovered correctly: "
        f"{'YES' if recovered else 'NO'}"
    )

    if not recovered:
        raise RuntimeError(
            "The controlled inverse-history experiment did not "
            "rank the true scenario first."
        )

    if result.status is DecodeStatus.AMBIGUOUS:
        print(
            "\nWarning: the true candidate ranked first, but the "
            "decoder considers the alternatives insufficiently "
            "separated."
        )

    print()
    print("Interpretation")
    print(SUBSEPARATOR)
    print(
        "The decoder did not receive the true scenario label. "
        "It received only the observed StructuralSignature and "
        "forward-predicted signatures for admissible candidate "
        "histories."
    )
    print(
        "Correct recovery in this controlled experiment demonstrates "
        "pipeline consistency, not unique identifiability of arbitrary "
        "real-world history."
    )
    print()
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()