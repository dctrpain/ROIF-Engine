from __future__ import annotations

"""
ROIF Engine - Example 27
Unified history-to-action pipeline demonstration.

This example connects the complete Structural Memory workflow:

    observed StructuralSignature
        -> inverse history reconstruction
        -> future candidate generation
        -> future-plane analysis
        -> forecast ranking
        -> counterfactual intervention ranking
        -> Node*

The historical ground truth is the chronic repeated-overload pattern used in
Examples 25 and 26. The decoder does not receive its label.

The future states, future planes, and interventions are controlled synthetic
scenarios. The example demonstrates pipeline consistency and orchestration. It
does not establish unique real-world historical identification or guaranteed
intervention effectiveness.
"""

from dataclasses import replace
from types import MappingProxyType
from typing import Final

from example_25_inverse_history_reconstruction import (
    build_forward_predictor,
    build_true_chronic_overload_pattern,
    make_candidate_rules,
)

from roif.history import (
    CounterfactualEngine,
    CounterfactualScenario,
    ForecastCandidate,
    ForecastCandidateStatus,
    ForecastDirection,
    ForecastHorizon,
    FuturePlane,
    FuturePlaneAnalyzer,
    FuturePlaneDirection,
    FuturePlaneScope,
    FuturePlaneStatus,
    HistoryDecoder,
    HistoryForecaster,
    HistoryPipeline,
    HistoryPipelineConfig,
    HypothesisGenerator,
    Intervention,
    InterventionScope,
    InterventionStatus,
    PlaneInfluenceSet,
    StructuralSignature,
    TemporalProfile,
)


SEPARATOR: Final[str] = "=" * 118
SUBSEPARATOR: Final[str] = "-" * 118

TRUE_HISTORY_NAME: Final[str] = "Chronic repeated overload"

FUTURE_CONTINUED: Final[str] = "future-continued-overload"
FUTURE_STABLE: Final[str] = "future-load-stabilization"
FUTURE_ADAPTIVE: Final[str] = "future-adaptive-recovery"


def updated_signature(
    base: StructuralSignature,
    *,
    signature_id: str,
    label: str,
    total_capacity_loss: float,
    total_capacity_gain: float,
    progression_index: float,
    adaptation_index: float,
    persistence_index: float | None = None,
    irreversibility_index: float | None = None,
    metadata: dict[str, object] | None = None,
) -> StructuralSignature:
    """Create an internally consistent future signature."""

    merged_metadata = {
        **dict(base.metadata),
        "example": 27,
        "synthetic_future": True,
        **(metadata or {}),
    }

    return replace(
        base,
        signature_id=signature_id,
        source_pattern_id=base.source_pattern_id,
        label=label,
        total_capacity_loss=total_capacity_loss,
        total_capacity_gain=total_capacity_gain,
        net_capacity_effect=(
            total_capacity_gain
            - total_capacity_loss
        ),
        progression_index=progression_index,
        adaptation_index=adaptation_index,
        persistence_index=(
            base.persistence_index
            if persistence_index is None
            else persistence_index
        ),
        irreversibility_index=(
            base.irreversibility_index
            if irreversibility_index is None
            else irreversibility_index
        ),
        metadata=MappingProxyType(merged_metadata),
    )


def build_future_candidates(
    observed: StructuralSignature,
    generation,
    decode_result,
) -> tuple[ForecastCandidate, ...]:
    """
    Build three explicit synthetic futures.

    The decoded history is available to the factory, but candidate construction
    remains explicit and forward-defined.
    """

    horizon = ForecastHorizon(
        start=0.0,
        end=90.0,
        units="days",
        label="Ninety-day future",
    )

    continued_signature = updated_signature(
        observed,
        signature_id="signature-future-continued",
        label="Continued repeated overload",
        total_capacity_loss=0.43,
        total_capacity_gain=0.04,
        progression_index=0.93,
        adaptation_index=0.07,
        persistence_index=0.86,
        irreversibility_index=0.64,
        metadata={
            "future_class": "continued_overload",
        },
    )

    stable_signature = updated_signature(
        observed,
        signature_id="signature-future-stable",
        label="Load stabilization",
        total_capacity_loss=0.25,
        total_capacity_gain=0.08,
        progression_index=0.66,
        adaptation_index=0.34,
        persistence_index=0.66,
        irreversibility_index=0.50,
        metadata={
            "future_class": "stabilization",
        },
    )

    adaptive_signature = updated_signature(
        observed,
        signature_id="signature-future-adaptive",
        label="Adaptive recovery",
        total_capacity_loss=0.16,
        total_capacity_gain=0.17,
        progression_index=0.39,
        adaptation_index=0.61,
        persistence_index=0.48,
        irreversibility_index=0.42,
        metadata={
            "future_class": "adaptive_recovery",
        },
    )

    decoded_best = decode_result.best
    decoded_name = (
        decoded_best.name
        if decoded_best is not None
        else "none"
    )

    common_metadata = {
        "example": 27,
        "decoded_history": decoded_name,
        "history_candidate_count": (
            generation.predicted_count
        ),
    }

    return (
        ForecastCandidate(
            candidate_id=FUTURE_CONTINUED,
            name="Continued repeated overload",
            description=(
                "The source loading pattern remains active and "
                "Capacity loss continues."
            ),
            horizon=horizon,
            predicted_signature=continued_signature,
            direction=ForecastDirection.DETERIORATION,
            affected_plane_ids=(
                "workload",
                "cold-exposure",
            ),
            affected_target_ids=(
                "node:source",
                "node:compensator",
            ),
            predicted_change_kinds=(
                "fatigue",
                "remodeling",
            ),
            prior_probability=0.62,
            validation_score=0.92,
            complexity=1.0,
            status=(
                ForecastCandidateStatus
                .FORWARD_VALIDATED
            ),
            metadata=common_metadata,
        ),
        ForecastCandidate(
            candidate_id=FUTURE_STABLE,
            name="Load stabilization",
            description=(
                "The repeated load is reduced enough to stop "
                "further rapid Capacity loss."
            ),
            horizon=horizon,
            predicted_signature=stable_signature,
            direction=ForecastDirection.STABLE,
            affected_plane_ids=(
                "workload",
                "recovery-window",
            ),
            affected_target_ids=(
                "node:source",
            ),
            predicted_change_kinds=(
                "stabilization",
            ),
            prior_probability=0.54,
            validation_score=0.90,
            complexity=1.5,
            status=(
                ForecastCandidateStatus
                .FORWARD_VALIDATED
            ),
            metadata=common_metadata,
        ),
        ForecastCandidate(
            candidate_id=FUTURE_ADAPTIVE,
            name="Adaptive recovery",
            description=(
                "Load reduction and recovery resources produce "
                "a net-positive Capacity trajectory."
            ),
            horizon=horizon,
            predicted_signature=adaptive_signature,
            direction=ForecastDirection.ADAPTATION,
            affected_plane_ids=(
                "recovery-window",
                "adaptive-support",
            ),
            affected_target_ids=(
                "node:source",
                "node:reserve",
            ),
            predicted_change_kinds=(
                "adaptation",
                "remodeling",
            ),
            prior_probability=0.42,
            validation_score=0.86,
            complexity=2.0,
            status=(
                ForecastCandidateStatus
                .FORWARD_VALIDATED
            ),
            metadata=common_metadata,
        ),
    )


def build_future_planes() -> tuple[FuturePlane, ...]:
    """Build explicit future influences with different temporal profiles."""

    return (
        FuturePlane(
            future_plane_id="plane-future-workload",
            plane_id="workload",
            name="Expected workload surge",
            horizon=ForecastHorizon(
                start=0.0,
                end=45.0,
                units="days",
            ),
            signed_effect=0.62,
            intensity=1.0,
            confidence=0.90,
            multiplier_scale=1.0,
            direction=FuturePlaneDirection.AMPLIFY,
            temporal_profile=TemporalProfile.RAMP_UP,
            scope=FuturePlaneScope.SUBSYSTEM,
            target_ids=(
                "node:source",
                "node:compensator",
            ),
            candidate_ids=(
                FUTURE_CONTINUED,
                FUTURE_STABLE,
            ),
            operational_risk=0.08,
            cascade_risk=0.14,
            observability=0.92,
            reversibility=0.88,
            non_fonit_allowed=True,
            status=FuturePlaneStatus.ACTIVE,
            metadata={
                "example": 27,
                "interpretation": (
                    "Known near-term increase in repeated loading."
                ),
            },
        ),
        FuturePlane(
            future_plane_id="plane-future-cold",
            plane_id="cold-exposure",
            name="Intermittent cold exposure",
            horizon=ForecastHorizon(
                start=10.0,
                end=35.0,
                units="days",
            ),
            signed_effect=0.28,
            intensity=0.70,
            confidence=0.70,
            multiplier_scale=1.0,
            direction=FuturePlaneDirection.AMPLIFY,
            temporal_profile=TemporalProfile.WINDOWED,
            scope=FuturePlaneScope.EXTERNAL,
            target_ids=(
                "node:compensator",
            ),
            candidate_ids=(
                FUTURE_CONTINUED,
            ),
            operational_risk=0.04,
            cascade_risk=0.05,
            observability=0.78,
            reversibility=0.95,
            non_fonit_allowed=True,
            status=FuturePlaneStatus.INFERRED,
            metadata={
                "example": 27,
            },
        ),
        FuturePlane(
            future_plane_id="plane-future-recovery",
            plane_id="recovery-window",
            name="Protected recovery window",
            horizon=ForecastHorizon(
                start=0.0,
                end=90.0,
                units="days",
            ),
            signed_effect=0.48,
            intensity=1.0,
            confidence=0.86,
            multiplier_scale=1.0,
            direction=FuturePlaneDirection.AMPLIFY,
            temporal_profile=TemporalProfile.CONSTANT,
            scope=FuturePlaneScope.SUBSYSTEM,
            target_ids=(
                "node:source",
                "node:reserve",
            ),
            candidate_ids=(
                FUTURE_STABLE,
                FUTURE_ADAPTIVE,
            ),
            operational_risk=0.02,
            cascade_risk=0.03,
            observability=0.88,
            reversibility=0.98,
            non_fonit_allowed=True,
            status=FuturePlaneStatus.ACTIVE,
            metadata={
                "example": 27,
            },
        ),
        FuturePlane(
            future_plane_id="plane-future-adaptive",
            plane_id="adaptive-support",
            name="Adaptive support",
            horizon=ForecastHorizon(
                start=15.0,
                end=90.0,
                units="days",
            ),
            signed_effect=0.54,
            intensity=0.90,
            confidence=0.76,
            multiplier_scale=1.0,
            direction=FuturePlaneDirection.AMPLIFY,
            temporal_profile=TemporalProfile.RAMP_UP,
            scope=FuturePlaneScope.TARGET,
            target_ids=(
                "node:source",
                "node:reserve",
            ),
            candidate_ids=(
                FUTURE_ADAPTIVE,
            ),
            operational_risk=0.03,
            cascade_risk=0.04,
            observability=0.72,
            reversibility=0.96,
            non_fonit_allowed=True,
            status=FuturePlaneStatus.INFERRED,
            metadata={
                "example": 27,
            },
        ),
        FuturePlane(
            future_plane_id="plane-future-unsafe-global",
            plane_id="unsafe-global",
            name="Unsafe global forcing",
            horizon=ForecastHorizon(
                start=0.0,
                end=90.0,
                units="days",
            ),
            signed_effect=0.95,
            intensity=1.0,
            confidence=0.40,
            multiplier_scale=1.0,
            direction=FuturePlaneDirection.MIXED,
            temporal_profile=TemporalProfile.CONSTANT,
            scope=FuturePlaneScope.SYSTEM,
            operational_risk=0.95,
            cascade_risk=0.98,
            observability=0.35,
            reversibility=0.05,
            non_fonit_allowed=False,
            status=FuturePlaneStatus.VETOED,
            metadata={
                "example": 27,
                "veto": "Non-Fonit Gate",
            },
        ),
    )


def intervention_forecast(
    observed: StructuralSignature,
    *,
    candidate_id: str,
    name: str,
    signature: StructuralSignature,
    prior_probability: float,
    validation_score: float,
) -> object:
    """Build a one-candidate forward forecast for an intervention."""

    candidate = ForecastCandidate(
        candidate_id=candidate_id,
        name=name,
        description=(
            "Synthetic post-intervention future."
        ),
        horizon=ForecastHorizon(
            start=0.0,
            end=90.0,
            units="days",
        ),
        predicted_signature=signature,
        direction=ForecastDirection.ADAPTATION,
        affected_plane_ids=(),
        affected_target_ids=(
            "node:source",
        ),
        predicted_change_kinds=(
            "stabilization",
            "adaptation",
        ),
        prior_probability=prior_probability,
        validation_score=validation_score,
        complexity=1.0,
        status=(
            ForecastCandidateStatus
            .FORWARD_VALIDATED
        ),
        metadata={
            "example": 27,
            "intervention_forecast": True,
        },
    )

    return HistoryForecaster().forecast(
        observed,
        (candidate,),
        PlaneInfluenceSet(
            label="Post-intervention neutral plane set"
        ),
        metadata={
            "example": 27,
        },
    )


def build_intervention_scenarios(
    observed: StructuralSignature,
    decode_result,
    forecast_result,
    plane_analysis,
) -> tuple[CounterfactualScenario, ...]:
    """Build bounded counterfactual interventions for Node* selection."""

    source_signature = updated_signature(
        observed,
        signature_id="signature-intervention-source",
        label="Source-load reduction future",
        total_capacity_loss=0.11,
        total_capacity_gain=0.18,
        progression_index=0.31,
        adaptation_index=0.69,
        persistence_index=0.40,
        irreversibility_index=0.39,
        metadata={
            "intervention": "source_load_reduction",
        },
    )

    compensator_signature = updated_signature(
        observed,
        signature_id="signature-intervention-compensator",
        label="Compensator-only future",
        total_capacity_loss=0.23,
        total_capacity_gain=0.09,
        progression_index=0.62,
        adaptation_index=0.38,
        persistence_index=0.64,
        irreversibility_index=0.51,
        metadata={
            "intervention": "compensator_only",
        },
    )

    broad_signature = updated_signature(
        observed,
        signature_id="signature-intervention-broad",
        label="Broad forcing future",
        total_capacity_loss=0.15,
        total_capacity_gain=0.13,
        progression_index=0.46,
        adaptation_index=0.54,
        persistence_index=0.52,
        irreversibility_index=0.46,
        metadata={
            "intervention": "broad_system_forcing",
        },
    )

    return (
        CounterfactualScenario(
            scenario_id="scenario-source-node",
            intervention=Intervention(
                intervention_id="intervention-source-node",
                name="Reduce load at source node",
                target_id="node:source",
                scope=InterventionScope.NODE,
                description=(
                    "Small, bounded reduction of the repeated "
                    "source load."
                ),
                affected_plane_ids=(
                    "workload",
                ),
                affected_target_ids=(
                    "node:source",
                ),
                magnitude=0.30,
                cost=0.18,
                operational_risk=0.05,
                cascade_risk=0.04,
                reversibility=0.96,
                complexity=1.0,
                non_fonit_allowed=True,
                status=(
                    InterventionStatus
                    .FORWARD_VALIDATED
                ),
                metadata={
                    "example": 27,
                },
            ),
            forecast_result=intervention_forecast(
                observed,
                candidate_id="post-source-intervention",
                name="Future after source-load reduction",
                signature=source_signature,
                prior_probability=0.66,
                validation_score=0.93,
            ),
            validation_score=0.92,
            model_penalty=0.04,
            metadata={
                "example": 27,
            },
        ),
        CounterfactualScenario(
            scenario_id="scenario-compensator-node",
            intervention=Intervention(
                intervention_id="intervention-compensator-node",
                name="Support compensator only",
                target_id="node:compensator",
                scope=InterventionScope.NODE,
                description=(
                    "Local support of the visible compensator "
                    "without changing the source load."
                ),
                affected_plane_ids=(
                    "recovery-window",
                ),
                affected_target_ids=(
                    "node:compensator",
                ),
                magnitude=0.40,
                cost=0.25,
                operational_risk=0.08,
                cascade_risk=0.10,
                reversibility=0.92,
                complexity=1.2,
                non_fonit_allowed=True,
                status=(
                    InterventionStatus
                    .FORWARD_VALIDATED
                ),
                metadata={
                    "example": 27,
                },
            ),
            forecast_result=intervention_forecast(
                observed,
                candidate_id="post-compensator-intervention",
                name="Future after compensator support",
                signature=compensator_signature,
                prior_probability=0.58,
                validation_score=0.88,
            ),
            validation_score=0.86,
            model_penalty=0.06,
            metadata={
                "example": 27,
            },
        ),
        CounterfactualScenario(
            scenario_id="scenario-unsafe-global",
            intervention=Intervention(
                intervention_id="intervention-unsafe-global",
                name="Broad system forcing",
                target_id="system:global",
                scope=InterventionScope.SYSTEM,
                description=(
                    "High-scope intervention rejected by the "
                    "Non-Fonit Gate."
                ),
                affected_plane_ids=(
                    "unsafe-global",
                ),
                affected_target_ids=(),
                magnitude=1.0,
                cost=0.80,
                operational_risk=0.95,
                cascade_risk=0.98,
                reversibility=0.08,
                complexity=5.0,
                non_fonit_allowed=False,
                status=InterventionStatus.VETOED,
                metadata={
                    "example": 27,
                    "veto": "Non-Fonit Gate",
                },
            ),
            forecast_result=intervention_forecast(
                observed,
                candidate_id="post-unsafe-global",
                name="Future after broad forcing",
                signature=broad_signature,
                prior_probability=0.30,
                validation_score=0.45,
            ),
            validation_score=0.40,
            model_penalty=0.50,
            metadata={
                "example": 27,
            },
        ),
    )


def print_history_section(result) -> None:
    print()
    print("1. Inverse history reconstruction")
    print(SUBSEPARATOR)

    generation = result.history_generation
    decode = result.history_decode
    best = result.best_history

    print(
        "Generated histories : "
        f"{result.metrics.generated_hypotheses}"
    )
    print(
        "Failed predictions  : "
        f"{result.metrics.failed_history_predictions}"
    )
    print(
        "Decoded candidates  : "
        f"{result.metrics.decoded_rankings}"
    )
    print(
        "Decoder confidence  : "
        f"{result.metrics.history_confidence:.6f}"
    )

    if best is None:
        print("Best history        : none")
        return

    print(f"Best history        : {best.name}")
    print(
        "Ground-truth check  : "
        f"{'YES' if best.name == TRUE_HISTORY_NAME else 'NO'}"
    )

    if generation is not None:
        print(
            "Pipeline input rule : true scenario label was not "
            "passed to HistoryDecoder"
        )


def print_future_planes(result) -> None:
    print()
    print("2. Future-plane analysis")
    print(SUBSEPARATOR)

    analysis = result.future_plane_analysis

    if analysis is None:
        print("Future-plane analysis: unavailable")
        return

    print(f"Analysis status     : {analysis.status.value}")
    print(f"Declared planes     : {analysis.plane_count}")
    print(
        "Eligible planes     : "
        f"{analysis.eligible_plane_count}"
    )
    print(
        "Vetoed planes       : "
        f"{analysis.vetoed_plane_count}"
    )
    print(
        "Applied contributions: "
        f"{result.metrics.applied_future_plane_contributions}"
    )
    print()

    print(
        f"{'Future candidate':<34}"
        f"{'Applied':>10}"
        f"{'Vetoed':>10}"
        f"{'Multiplier':>16}"
        f"{'Observability':>17}"
    )
    print(SUBSEPARATOR)

    for item in analysis.candidate_analyses:
        print(
            f"{item.candidate_id:<34}"
            f"{item.applied_plane_count:>10}"
            f"{item.vetoed_plane_count:>10}"
            f"{item.combined_multiplier:>16.6f}"
            f"{item.mean_observability:>17.6f}"
        )


def print_forecast_section(result) -> None:
    print()
    print("3. Ranked futures")
    print(SUBSEPARATOR)

    forecast = result.forecast_result

    if forecast is None:
        print("Forecast: unavailable")
        return

    print(f"Forecast status     : {forecast.status.value}")
    print(f"Forecast confidence : {forecast.confidence:.6f}")
    print(f"Selection margin    : {forecast.selection_margin:.6f}")
    print()

    print(
        f"{'Rank':<6}"
        f"{'Future':<34}"
        f"{'Score':>12}"
        f"{'Plane M':>14}"
        f"{'Net Capacity':>16}"
    )
    print(SUBSEPARATOR)

    for ranking in forecast.rankings:
        signature = (
            ranking.candidate.predicted_signature
        )
        print(
            f"{ranking.rank:<6}"
            f"{ranking.name:<34}"
            f"{ranking.normalized_score:>12.6f}"
            f"{ranking.plane_multiplier:>14.6f}"
            f"{signature.net_capacity_effect:>+16.6f}"
        )


def print_intervention_section(result) -> None:
    print()
    print("4. Counterfactual intervention ranking")
    print(SUBSEPARATOR)

    counterfactual = result.counterfactual_result

    if counterfactual is None:
        print("Counterfactual result: unavailable")
        return

    print(
        "Counterfactual status: "
        f"{counterfactual.status.value}"
    )
    print(
        "Intervention confidence: "
        f"{counterfactual.confidence:.6f}"
    )
    print(
        "Selection margin      : "
        f"{counterfactual.selection_margin:.6f}"
    )
    print(f"Node*                 : {counterfactual.node_star}")
    print()

    print(
        f"{'Rank':<6}"
        f"{'Intervention':<34}"
        f"{'Target':<22}"
        f"{'Score':>12}"
        f"{'Delta Capacity':>17}"
    )
    print(SUBSEPARATOR)

    for ranking in counterfactual.rankings:
        print(
            f"{ranking.rank:<6}"
            f"{ranking.intervention_name:<34}"
            f"{ranking.target_id:<22}"
            f"{ranking.normalized_score:>12.6f}"
            f"{ranking.net_capacity_delta:>+17.6f}"
        )

    vetoed = (
        counterfactual.metadata.get(
            "vetoed_count",
            0,
        )
    )
    print()
    print(
        "Non-Fonit vetoed scenarios: "
        f"{vetoed}"
    )


def print_pipeline_trace(result) -> None:
    print()
    print("5. Pipeline execution trace")
    print(SUBSEPARATOR)
    print(
        f"{'Stage':<34}"
        f"{'Status':<14}"
        f"{'Input':>10}"
        f"{'Output':>10}"
    )
    print(SUBSEPARATOR)

    for record in result.stage_records:
        print(
            f"{record.stage.value:<34}"
            f"{record.status.value:<14}"
            f"{record.input_count:>10}"
            f"{record.output_count:>10}"
        )


def main() -> None:
    print(SEPARATOR)
    print("ROIF Engine - Example 27")
    print("Unified history-to-action pipeline")
    print(SEPARATOR)
    print()
    print("Experiment design")
    print(SUBSEPARATOR)
    print(
        "1. Build a hidden chronic repeated-overload history.\n"
        "2. Convert it into an observed StructuralSignature.\n"
        "3. Generate and forward-predict competing histories.\n"
        "4. Decode the most consistent past history.\n"
        "5. Build three explicit future structural candidates.\n"
        "6. Apply candidate-specific future influence planes.\n"
        "7. Rank future states.\n"
        "8. Compare bounded counterfactual interventions.\n"
        "9. Select Node* while vetoing unsafe global forcing."
    )
    print()
    print(
        "Metric note: decoder, forecast, counterfactual, and "
        "softmax-derived scores are internal model measures. "
        "They are not calibrated probabilities of real-world truth."
    )

    true_pattern = build_true_chronic_overload_pattern()

    observed = StructuralSignature.from_pattern(
        true_pattern,
        label="Observed chronic-overload signature",
        metadata={
            "example": 27,
            "ground_truth_hidden_from_decoder": True,
        },
        signature_id="observed-example-27",
    )

    pipeline = HistoryPipeline(
        hypothesis_generator=HypothesisGenerator(
            rules=make_candidate_rules(),
            max_candidates=10,
            fail_fast=True,
        ),
        history_decoder=HistoryDecoder(
            ambiguity_threshold=0.05,
            complexity_scale=5.0,
            minimum_score=0.0,
        ),
        history_forecaster=HistoryForecaster(
            ambiguity_threshold=0.05,
            complexity_scale=5.0,
            plane_scale=1.0,
            minimum_score=0.0,
        ),
        future_plane_analyzer=FuturePlaneAnalyzer(
            target_overlap_floor=1.0,
            safety_threshold=0.75,
            missing_observation_penalty=0.75,
            non_fonit_required=True,
        ),
        counterfactual_engine=CounterfactualEngine(
            ambiguity_threshold=0.05,
            cost_scale=1.0,
            complexity_scale=5.0,
            minimum_score=0.0,
            require_benefit=True,
        ),
        config=HistoryPipelineConfig(
            strict=True,
            require_forecast=True,
            require_intervention=True,
            max_future_candidates=10,
            max_intervention_scenarios=10,
            forecast_ambiguity_threshold=0.05,
            forecast_minimum_score=0.0,
        ),
    )

    result = pipeline.run(
        observed,
        history_forward_predictor=(
            build_forward_predictor()
        ),
        future_candidate_factory=(
            build_future_candidates
        ),
        future_planes=build_future_planes(),
        intervention_scenario_factory=(
            build_intervention_scenarios
        ),
        metadata={
            "example": 27,
            "experiment": (
                "unified-history-to-action-pipeline"
            ),
            "controlled_synthetic": True,
        },
    )

    print()
    print("Pipeline result")
    print(SUBSEPARATOR)
    print(f"Status              : {result.status.value}")
    print(f"Observed signature  : {result.observed_signature_id}")
    print(f"Pipeline result ID  : {result.result_id}")

    print_history_section(result)
    print_future_planes(result)
    print_forecast_section(result)
    print_intervention_section(result)
    print_pipeline_trace(result)

    print()
    print("Interpretation")
    print(SUBSEPARATOR)
    print(
        "The pipeline reconstructed the controlled historical class, "
        "ranked explicit future states under candidate-specific future "
        "planes, compared bounded interventions, and selected Node*."
    )
    print(
        "The unsafe system-wide intervention was excluded by the "
        "Non-Fonit Gate before ranking."
    )
    print(
        "This demonstrates end-to-end computational consistency. "
        "It does not prove unique real-world historical identifiability "
        "or guaranteed intervention efficacy."
    )

    print()
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
