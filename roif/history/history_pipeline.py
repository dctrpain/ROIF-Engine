from __future__ import annotations

"""
Unified orchestration pipeline for the ROIF Structural Memory Engine.

The pipeline connects:
    StructuralSignature
        -> HypothesisGenerator
        -> HistoryDecoder
        -> FuturePlaneAnalyzer
        -> HistoryForecaster
        -> CounterfactualEngine

The pipeline is an orchestrator. It does not invent causal rules, predicted
histories, future states, future planes, or interventions. Domain-dependent
objects are supplied by caller-provided forward models and factories.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import uuid4

from .counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    CounterfactualScenario,
    CounterfactualStatus,
)
from .future_plane_analyzer import (
    FuturePlane,
    FuturePlaneAnalysisResult,
    FuturePlaneAnalysisStatus,
    FuturePlaneAnalyzer,
)
from .history_decoder import HistoryDecodeResult, HistoryDecoder
from .history_forecast import (
    ForecastCandidate,
    ForecastScore,
    ForecastStatus,
    HistoryForecastResult,
    HistoryForecaster,
    PlaneInfluenceSet,
)
from .hypothesis_generator import (
    HypothesisGenerationResult,
    HypothesisGenerator,
    HypothesisSeed,
)
from .structural_signature import StructuralSignature


class HistoryPipelineError(ValueError):
    """Raised when pipeline inputs or results are invalid."""


class PipelineStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_HYPOTHESES = "no_hypotheses"
    NO_FORECASTS = "no_forecasts"
    NO_INTERVENTIONS = "no_interventions"


class PipelineStage(str, Enum):
    GENERATE_HISTORY = "generate_history"
    DECODE_HISTORY = "decode_history"
    BUILD_FUTURES = "build_futures"
    ANALYZE_FUTURE_PLANES = "analyze_future_planes"
    RANK_FUTURES = "rank_futures"
    BUILD_INTERVENTIONS = "build_interventions"
    RANK_INTERVENTIONS = "rank_interventions"


class StageStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    EMPTY = "empty"


HistoryForwardPredictor: TypeAlias = Callable[
    [HypothesisSeed, StructuralSignature],
    StructuralSignature,
]

FutureCandidateFactory: TypeAlias = Callable[
    [StructuralSignature, HypothesisGenerationResult, HistoryDecodeResult],
    Iterable[ForecastCandidate],
]

InterventionScenarioFactory: TypeAlias = Callable[
    [
        StructuralSignature,
        HistoryDecodeResult,
        HistoryForecastResult,
        FuturePlaneAnalysisResult,
    ],
    Iterable[CounterfactualScenario],
]


def _text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise HistoryPipelineError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise HistoryPipelineError(f"{field_name} must not be empty")
    return normalized


def _finite(value: float, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoryPipelineError(
            f"{field_name} must be a real number"
        ) from exc
    if not isfinite(numeric):
        raise HistoryPipelineError(f"{field_name} must be finite")
    return numeric


def _nonnegative(value: float, *, field_name: str) -> float:
    numeric = _finite(value, field_name=field_name)
    if numeric < 0.0:
        raise HistoryPipelineError(
            f"{field_name} must be non-negative"
        )
    return numeric


def _unit(value: float, *, field_name: str) -> float:
    numeric = _nonnegative(value, field_name=field_name)
    if numeric > 1.0:
        raise HistoryPipelineError(
            f"{field_name} must be in [0, 1]"
        )
    return numeric


def _mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise HistoryPipelineError(
            f"{field_name} must be a mapping"
        )
    normalized = {
        _text(key, field_name=f"{field_name} key"): item
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class PipelineStageRecord:
    stage: PipelineStage
    status: StageStatus
    input_count: int = 0
    output_count: int = 0
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        try:
            stage = PipelineStage(self.stage)
            status = StageStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise HistoryPipelineError(
                "unsupported pipeline-stage enum value"
            ) from exc

        for name in ("input_count", "output_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise HistoryPipelineError(
                    f"{name} must be a non-negative integer"
                )

        message = (
            None
            if self.message is None
            else _text(self.message, field_name="message")
        )

        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "metadata",
            _mapping(self.metadata, field_name="metadata"),
        )

    @property
    def succeeded(self) -> bool:
        return self.status in {
            StageStatus.COMPLETED,
            StageStatus.SKIPPED,
            StageStatus.EMPTY,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "stage": self.stage.value,
            "status": self.status.value,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HistoryPipelineConfig:
    strict: bool = True
    require_forecast: bool = True
    require_intervention: bool = False
    max_future_candidates: int | None = None
    max_intervention_scenarios: int | None = None
    forecast_ambiguity_threshold: float = 0.05
    forecast_minimum_score: float = 0.0

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        for name in (
            "strict",
            "require_forecast",
            "require_intervention",
        ):
            if not isinstance(getattr(self, name), bool):
                raise HistoryPipelineError(
                    f"{name} must be a boolean"
                )

        for name in (
            "max_future_candidates",
            "max_intervention_scenarios",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or value <= 0
            ):
                raise HistoryPipelineError(
                    f"{name} must be a positive integer or None"
                )

        object.__setattr__(
            self,
            "forecast_ambiguity_threshold",
            _unit(
                self.forecast_ambiguity_threshold,
                field_name="forecast_ambiguity_threshold",
            ),
        )
        object.__setattr__(
            self,
            "forecast_minimum_score",
            _unit(
                self.forecast_minimum_score,
                field_name="forecast_minimum_score",
            ),
        )


@dataclass(frozen=True, slots=True)
class HistoryPipelineMetrics:
    generated_hypotheses: int = 0
    failed_history_predictions: int = 0
    decoded_rankings: int = 0
    future_candidates: int = 0
    applied_future_plane_contributions: int = 0
    vetoed_future_planes: int = 0
    ranked_futures: int = 0
    intervention_scenarios: int = 0
    ranked_interventions: int = 0
    history_confidence: float = 0.0
    forecast_confidence: float = 0.0
    intervention_confidence: float = 0.0

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        for name in (
            "generated_hypotheses",
            "failed_history_predictions",
            "decoded_rankings",
            "future_candidates",
            "applied_future_plane_contributions",
            "vetoed_future_planes",
            "ranked_futures",
            "intervention_scenarios",
            "ranked_interventions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise HistoryPipelineError(
                    f"{name} must be a non-negative integer"
                )

        for name in (
            "history_confidence",
            "forecast_confidence",
            "intervention_confidence",
        ):
            object.__setattr__(
                self,
                name,
                _unit(getattr(self, name), field_name=name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "generated_hypotheses",
                "failed_history_predictions",
                "decoded_rankings",
                "future_candidates",
                "applied_future_plane_contributions",
                "vetoed_future_planes",
                "ranked_futures",
                "intervention_scenarios",
                "ranked_interventions",
                "history_confidence",
                "forecast_confidence",
                "intervention_confidence",
            )
        } | {"version": self.VERSION}


@dataclass(frozen=True, slots=True)
class HistoryPipelineResult:
    observed_signature_id: str
    status: PipelineStatus
    history_generation: HypothesisGenerationResult | None = None
    history_decode: HistoryDecodeResult | None = None
    future_plane_analysis: FuturePlaneAnalysisResult | None = None
    forecast_result: HistoryForecastResult | None = None
    counterfactual_result: CounterfactualResult | None = None
    stage_records: tuple[PipelineStageRecord, ...] = ()
    metrics: HistoryPipelineMetrics = field(
        default_factory=HistoryPipelineMetrics
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: uuid4().hex)

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        try:
            status = PipelineStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise HistoryPipelineError(
                "unsupported pipeline status"
            ) from exc

        records = tuple(self.stage_records)
        if not all(
            isinstance(item, PipelineStageRecord)
            for item in records
        ):
            raise HistoryPipelineError(
                "all stage_records must be PipelineStageRecord objects"
            )
        if not isinstance(self.metrics, HistoryPipelineMetrics):
            raise HistoryPipelineError(
                "metrics must be HistoryPipelineMetrics"
            )

        object.__setattr__(
            self,
            "observed_signature_id",
            _text(
                self.observed_signature_id,
                field_name="observed_signature_id",
            ),
        )
        object.__setattr__(
            self,
            "result_id",
            _text(self.result_id, field_name="result_id"),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "stage_records", records)
        object.__setattr__(
            self,
            "metadata",
            _mapping(self.metadata, field_name="metadata"),
        )

    @property
    def best_history(self):
        return (
            self.history_decode.best
            if self.history_decode is not None
            else None
        )

    @property
    def best_future(self) -> ForecastScore | None:
        return (
            self.forecast_result.best
            if self.forecast_result is not None
            else None
        )

    @property
    def best_intervention(self):
        return (
            self.counterfactual_result.best
            if self.counterfactual_result is not None
            else None
        )

    @property
    def node_star(self) -> str | None:
        return (
            self.counterfactual_result.node_star
            if self.counterfactual_result is not None
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "result_id": self.result_id,
            "observed_signature_id": self.observed_signature_id,
            "status": self.status.value,
            "node_star": self.node_star,
            "metrics": self.metrics.to_dict(),
            "stage_records": [
                item.to_dict() for item in self.stage_records
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HistoryPipeline:
    hypothesis_generator: HypothesisGenerator
    history_decoder: HistoryDecoder
    history_forecaster: HistoryForecaster
    future_plane_analyzer: FuturePlaneAnalyzer
    counterfactual_engine: CounterfactualEngine
    config: HistoryPipelineConfig = field(
        default_factory=HistoryPipelineConfig
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        expected = (
            ("hypothesis_generator", self.hypothesis_generator, HypothesisGenerator),
            ("history_decoder", self.history_decoder, HistoryDecoder),
            ("history_forecaster", self.history_forecaster, HistoryForecaster),
            ("future_plane_analyzer", self.future_plane_analyzer, FuturePlaneAnalyzer),
            ("counterfactual_engine", self.counterfactual_engine, CounterfactualEngine),
            ("config", self.config, HistoryPipelineConfig),
        )

        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise HistoryPipelineError(
                    f"{name} must be {expected_type.__name__}"
                )

    def _aggregate_forecast(
        self,
        current: StructuralSignature,
        candidates: tuple[ForecastCandidate, ...],
        plane_analysis: FuturePlaneAnalysisResult,
    ) -> HistoryForecastResult:
        scored: list[ForecastScore] = []

        for candidate in candidates:
            analysis = plane_analysis.for_candidate(
                candidate.candidate_id
            )
            influences = (
                analysis.plane_influence_set
                if analysis is not None
                else PlaneInfluenceSet(
                    label="No applicable future planes"
                )
            )
            score = self.history_forecaster.score_candidate(
                current,
                candidate,
                influences,
            )
            if (
                score.normalized_score
                >= self.config.forecast_minimum_score
            ):
                scored.append(score)

        scored.sort(
            key=lambda item: (
                -item.normalized_score,
                -item.continuity_score,
                -item.validation_score,
                item.candidate.complexity,
                item.candidate_id,
            )
        )

        ranked = tuple(
            item.with_rank(index)
            for index, item in enumerate(scored, start=1)
        )

        influence_set_id = (
            f"pipeline-{plane_analysis.result_id}"
        )

        if not ranked:
            return HistoryForecastResult(
                current_signature_id=current.signature_id,
                influence_set_id=influence_set_id,
                status=ForecastStatus.NO_VALID_CANDIDATES,
            )

        best_score = ranked[0].normalized_score
        margin = (
            best_score
            if len(ranked) == 1
            else max(
                best_score - ranked[1].normalized_score,
                0.0,
            )
        )
        ambiguity = 1.0 - margin

        best_analysis = plane_analysis.for_candidate(
            ranked[0].candidate_id
        )
        observed_fraction = (
            best_analysis.mean_observability
            if best_analysis is not None
            else 1.0
        )

        confidence = min(
            max(
                best_score
                * (0.5 + 0.5 * margin)
                * (0.5 + 0.5 * observed_fraction),
                0.0,
            ),
            1.0,
        )

        status = (
            ForecastStatus.AMBIGUOUS
            if (
                len(ranked) > 1
                and margin
                < self.config.forecast_ambiguity_threshold
            )
            else ForecastStatus.COMPLETED
        )

        return HistoryForecastResult(
            current_signature_id=current.signature_id,
            influence_set_id=influence_set_id,
            status=status,
            rankings=ranked,
            confidence=confidence,
            ambiguity=ambiguity,
            selection_margin=margin,
            metadata={
                "pipeline_version": self.VERSION,
                "candidate_specific_future_planes": True,
            },
        )

    def _metrics(
        self,
        generation: HypothesisGenerationResult | None,
        decode: HistoryDecodeResult | None,
        planes: FuturePlaneAnalysisResult | None,
        forecast: HistoryForecastResult | None,
        counterfactual: CounterfactualResult | None,
        *,
        future_candidate_count: int,
        intervention_scenario_count: int,
    ) -> HistoryPipelineMetrics:
        return HistoryPipelineMetrics(
            generated_hypotheses=(
                generation.predicted_count
                if generation is not None
                else 0
            ),
            failed_history_predictions=(
                generation.failed_count
                if generation is not None
                else 0
            ),
            decoded_rankings=(
                len(decode.rankings)
                if decode is not None
                else 0
            ),
            future_candidates=future_candidate_count,
            applied_future_plane_contributions=(
                sum(
                    item.applied_plane_count
                    for item in planes.candidate_analyses
                )
                if planes is not None
                else 0
            ),
            vetoed_future_planes=(
                planes.vetoed_plane_count
                if planes is not None
                else 0
            ),
            ranked_futures=(
                len(forecast.rankings)
                if forecast is not None
                else 0
            ),
            intervention_scenarios=intervention_scenario_count,
            ranked_interventions=(
                len(counterfactual.rankings)
                if counterfactual is not None
                else 0
            ),
            history_confidence=(
                decode.confidence if decode is not None else 0.0
            ),
            forecast_confidence=(
                forecast.confidence if forecast is not None else 0.0
            ),
            intervention_confidence=(
                counterfactual.confidence
                if counterfactual is not None
                else 0.0
            ),
        )

    def run(
        self,
        observed: StructuralSignature,
        *,
        history_forward_predictor: HistoryForwardPredictor,
        future_candidate_factory: FutureCandidateFactory,
        future_planes: Iterable[FuturePlane] = (),
        intervention_scenario_factory: (
            InterventionScenarioFactory | None
        ) = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> HistoryPipelineResult:
        if not isinstance(observed, StructuralSignature):
            raise HistoryPipelineError(
                "observed must be a StructuralSignature"
            )
        if not callable(history_forward_predictor):
            raise HistoryPipelineError(
                "history_forward_predictor must be callable"
            )
        if not callable(future_candidate_factory):
            raise HistoryPipelineError(
                "future_candidate_factory must be callable"
            )
        if (
            intervention_scenario_factory is not None
            and not callable(intervention_scenario_factory)
        ):
            raise HistoryPipelineError(
                "intervention_scenario_factory must be callable or None"
            )

        plane_tuple = tuple(future_planes)
        if not all(
            isinstance(item, FuturePlane)
            for item in plane_tuple
        ):
            raise HistoryPipelineError(
                "all future_planes must be FuturePlane objects"
            )

        records: list[PipelineStageRecord] = []

        try:
            generation = self.hypothesis_generator.generate(
                observed,
                history_forward_predictor,
                metadata={
                    "pipeline_version": self.VERSION,
                },
            )
            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.GENERATE_HISTORY,
                    status=(
                        StageStatus.COMPLETED
                        if generation.hypotheses
                        else StageStatus.EMPTY
                    ),
                    input_count=1,
                    output_count=len(generation.hypotheses),
                )
            )

            if not generation.hypotheses:
                return HistoryPipelineResult(
                    observed_signature_id=observed.signature_id,
                    status=PipelineStatus.NO_HYPOTHESES,
                    history_generation=generation,
                    stage_records=tuple(records),
                    metrics=self._metrics(
                        generation,
                        None,
                        None,
                        None,
                        None,
                        future_candidate_count=0,
                        intervention_scenario_count=0,
                    ),
                    metadata=metadata or {},
                )

            decode = self.history_decoder.decode(
                observed,
                generation.hypotheses,
                metadata={"pipeline_version": self.VERSION},
            )
            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.DECODE_HISTORY,
                    status=StageStatus.COMPLETED,
                    input_count=len(generation.hypotheses),
                    output_count=len(decode.rankings),
                )
            )

            candidates = tuple(
                future_candidate_factory(
                    observed,
                    generation,
                    decode,
                )
            )
            if not all(
                isinstance(item, ForecastCandidate)
                for item in candidates
            ):
                raise HistoryPipelineError(
                    "future_candidate_factory must return "
                    "ForecastCandidate objects"
                )

            if self.config.max_future_candidates is not None:
                candidates = candidates[
                    : self.config.max_future_candidates
                ]

            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.BUILD_FUTURES,
                    status=(
                        StageStatus.COMPLETED
                        if candidates
                        else StageStatus.EMPTY
                    ),
                    input_count=len(decode.rankings),
                    output_count=len(candidates),
                )
            )

            if not candidates:
                return HistoryPipelineResult(
                    observed_signature_id=observed.signature_id,
                    status=(
                        PipelineStatus.NO_FORECASTS
                        if self.config.require_forecast
                        else PipelineStatus.PARTIAL
                    ),
                    history_generation=generation,
                    history_decode=decode,
                    stage_records=tuple(records),
                    metrics=self._metrics(
                        generation,
                        decode,
                        None,
                        None,
                        None,
                        future_candidate_count=0,
                        intervention_scenario_count=0,
                    ),
                    metadata=metadata or {},
                )

            plane_analysis = self.future_plane_analyzer.analyze(
                candidates,
                plane_tuple,
                metadata={"pipeline_version": self.VERSION},
            )
            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.ANALYZE_FUTURE_PLANES,
                    status=(
                        StageStatus.COMPLETED
                        if plane_analysis.status
                        not in {
                            FuturePlaneAnalysisStatus.NO_PLANES,
                            FuturePlaneAnalysisStatus.NO_APPLICABLE_PLANES,
                            FuturePlaneAnalysisStatus.ALL_VETOED,
                        }
                        else StageStatus.EMPTY
                    ),
                    input_count=len(plane_tuple),
                    output_count=len(
                        plane_analysis.candidate_analyses
                    ),
                )
            )

            forecast = self._aggregate_forecast(
                observed,
                candidates,
                plane_analysis,
            )
            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.RANK_FUTURES,
                    status=(
                        StageStatus.COMPLETED
                        if forecast.rankings
                        else StageStatus.EMPTY
                    ),
                    input_count=len(candidates),
                    output_count=len(forecast.rankings),
                )
            )

            if not forecast.rankings:
                return HistoryPipelineResult(
                    observed_signature_id=observed.signature_id,
                    status=PipelineStatus.NO_FORECASTS,
                    history_generation=generation,
                    history_decode=decode,
                    future_plane_analysis=plane_analysis,
                    forecast_result=forecast,
                    stage_records=tuple(records),
                    metrics=self._metrics(
                        generation,
                        decode,
                        plane_analysis,
                        forecast,
                        None,
                        future_candidate_count=len(candidates),
                        intervention_scenario_count=0,
                    ),
                    metadata=metadata or {},
                )

            if intervention_scenario_factory is None:
                records.extend(
                    (
                        PipelineStageRecord(
                            stage=PipelineStage.BUILD_INTERVENTIONS,
                            status=StageStatus.SKIPPED,
                            message=(
                                "No intervention scenario factory supplied."
                            ),
                        ),
                        PipelineStageRecord(
                            stage=PipelineStage.RANK_INTERVENTIONS,
                            status=StageStatus.SKIPPED,
                            message=(
                                "Counterfactual ranking was not requested."
                            ),
                        ),
                    )
                )
                return HistoryPipelineResult(
                    observed_signature_id=observed.signature_id,
                    status=(
                        PipelineStatus.NO_INTERVENTIONS
                        if self.config.require_intervention
                        else PipelineStatus.COMPLETED
                    ),
                    history_generation=generation,
                    history_decode=decode,
                    future_plane_analysis=plane_analysis,
                    forecast_result=forecast,
                    stage_records=tuple(records),
                    metrics=self._metrics(
                        generation,
                        decode,
                        plane_analysis,
                        forecast,
                        None,
                        future_candidate_count=len(candidates),
                        intervention_scenario_count=0,
                    ),
                    metadata=metadata or {},
                )

            scenarios = tuple(
                intervention_scenario_factory(
                    observed,
                    decode,
                    forecast,
                    plane_analysis,
                )
            )
            if not all(
                isinstance(item, CounterfactualScenario)
                for item in scenarios
            ):
                raise HistoryPipelineError(
                    "intervention_scenario_factory must return "
                    "CounterfactualScenario objects"
                )

            if (
                self.config.max_intervention_scenarios
                is not None
            ):
                scenarios = scenarios[
                    : self.config.max_intervention_scenarios
                ]

            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.BUILD_INTERVENTIONS,
                    status=(
                        StageStatus.COMPLETED
                        if scenarios
                        else StageStatus.EMPTY
                    ),
                    input_count=len(forecast.rankings),
                    output_count=len(scenarios),
                )
            )

            if not scenarios:
                return HistoryPipelineResult(
                    observed_signature_id=observed.signature_id,
                    status=(
                        PipelineStatus.NO_INTERVENTIONS
                        if self.config.require_intervention
                        else PipelineStatus.PARTIAL
                    ),
                    history_generation=generation,
                    history_decode=decode,
                    future_plane_analysis=plane_analysis,
                    forecast_result=forecast,
                    stage_records=tuple(records),
                    metrics=self._metrics(
                        generation,
                        decode,
                        plane_analysis,
                        forecast,
                        None,
                        future_candidate_count=len(candidates),
                        intervention_scenario_count=0,
                    ),
                    metadata=metadata or {},
                )

            counterfactual = self.counterfactual_engine.evaluate(
                forecast,
                scenarios,
                metadata={"pipeline_version": self.VERSION},
            )
            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.RANK_INTERVENTIONS,
                    status=(
                        StageStatus.COMPLETED
                        if counterfactual.rankings
                        else StageStatus.EMPTY
                    ),
                    input_count=len(scenarios),
                    output_count=len(
                        counterfactual.rankings
                    ),
                )
            )

            return HistoryPipelineResult(
                observed_signature_id=observed.signature_id,
                status=(
                    PipelineStatus.COMPLETED
                    if counterfactual.status
                    in {
                        CounterfactualStatus.COMPLETED,
                        CounterfactualStatus.AMBIGUOUS,
                    }
                    else PipelineStatus.NO_INTERVENTIONS
                ),
                history_generation=generation,
                history_decode=decode,
                future_plane_analysis=plane_analysis,
                forecast_result=forecast,
                counterfactual_result=counterfactual,
                stage_records=tuple(records),
                metrics=self._metrics(
                    generation,
                    decode,
                    plane_analysis,
                    forecast,
                    counterfactual,
                    future_candidate_count=len(candidates),
                    intervention_scenario_count=len(scenarios),
                ),
                metadata=metadata or {},
            )

        except Exception as exc:
            if self.config.strict:
                raise

            records.append(
                PipelineStageRecord(
                    stage=PipelineStage.RANK_INTERVENTIONS,
                    status=StageStatus.FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            return HistoryPipelineResult(
                observed_signature_id=observed.signature_id,
                status=PipelineStatus.FAILED,
                stage_records=tuple(records),
                metadata={
                    **dict(metadata or {}),
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                },
            )


__all__ = [
    "FutureCandidateFactory",
    "HistoryForwardPredictor",
    "HistoryPipeline",
    "HistoryPipelineConfig",
    "HistoryPipelineError",
    "HistoryPipelineMetrics",
    "HistoryPipelineResult",
    "InterventionScenarioFactory",
    "PipelineStage",
    "PipelineStageRecord",
    "PipelineStatus",
    "StageStatus",
]
