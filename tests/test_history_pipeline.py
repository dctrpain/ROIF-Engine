from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType, SimpleNamespace

import pytest

from roif.history.counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    CounterfactualScenario,
    CounterfactualStatus,
)
from roif.history.future_plane_analyzer import (
    FuturePlane,
    FuturePlaneAnalysisResult,
    FuturePlaneAnalysisStatus,
    FuturePlaneAnalyzer,
)
from roif.history.history_decoder import (
    HistoryDecodeResult,
    HistoryDecoder,
)
from roif.history.history_forecast import (
    ForecastCandidate,
    ForecastScore,
    ForecastStatus,
    HistoryForecastResult,
    HistoryForecaster,
    PlaneInfluenceSet,
)
from roif.history.history_pipeline import (
    HistoryPipeline,
    HistoryPipelineConfig,
    HistoryPipelineError,
    HistoryPipelineMetrics,
    HistoryPipelineResult,
    PipelineStage,
    PipelineStageRecord,
    PipelineStatus,
    StageStatus,
)
from roif.history.hypothesis_generator import (
    HypothesisGenerationResult,
    HypothesisGenerator,
)
from roif.history.structural_signature import (
    StructuralSignature,
)


# ---------------------------------------------------------------------------
# Raw-object helpers
# ---------------------------------------------------------------------------


def raw_instance(cls, **attributes):
    """
    Build an instance without invoking a domain constructor.

    These tests isolate HistoryPipeline orchestration. Domain modules already
    have their own constructor and validation test suites.
    """

    instance = object.__new__(cls)

    for name, value in attributes.items():
        object.__setattr__(instance, name, value)

    return instance


def make_observed(
    signature_id: str = "observed-signature",
) -> StructuralSignature:
    return raw_instance(
        StructuralSignature,
        signature_id=signature_id,
    )


def make_generation(
    *,
    hypothesis_count: int = 2,
    failed_count: int = 0,
):
    """
    Lightweight orchestration stub.

    HypothesisGenerationResult exposes hypotheses through a read-only
    property, so bypassing its constructor with object.__new__ is invalid.
    The pipeline only consumes these three public attributes.
    """

    hypotheses = tuple(
        SimpleNamespace(
            hypothesis_id=f"hypothesis-{index}",
            name=f"Hypothesis {index}",
        )
        for index in range(hypothesis_count)
    )

    return SimpleNamespace(
        hypotheses=hypotheses,
        predicted_count=hypothesis_count,
        failed_count=failed_count,
    )


def make_decode(
    *,
    ranking_count: int = 2,
    confidence: float = 0.80,
) -> HistoryDecodeResult:
    rankings = tuple(
        SimpleNamespace(
            hypothesis_id=f"hypothesis-{index}",
            name=f"Hypothesis {index}",
        )
        for index in range(ranking_count)
    )

    return raw_instance(
        HistoryDecodeResult,
        rankings=rankings,
        confidence=confidence,
    )


def make_candidate(
    candidate_id: str,
    *,
    complexity: float = 1.0,
    validation_score: float = 0.90,
) -> ForecastCandidate:
    return raw_instance(
        ForecastCandidate,
        candidate_id=candidate_id,
        complexity=complexity,
        validation_score=validation_score,
    )


def make_forecast_score(
    candidate: ForecastCandidate,
    *,
    normalized_score: float,
    continuity_score: float = 0.80,
    validation_score: float = 0.90,
    raw_score: float | None = None,
    rank: int = 0,
) -> ForecastScore:
    return raw_instance(
        ForecastScore,
        candidate=candidate,
        normalized_score=normalized_score,
        continuity_score=continuity_score,
        validation_score=validation_score,
        raw_score=(
            normalized_score
            if raw_score is None
            else raw_score
        ),
        rank=rank,
    )


def clone_score_with_rank(
    self: ForecastScore,
    rank: int,
) -> ForecastScore:
    return make_forecast_score(
        self.candidate,
        normalized_score=self.normalized_score,
        continuity_score=self.continuity_score,
        validation_score=self.validation_score,
        raw_score=self.raw_score,
        rank=rank,
    )


def make_plane_analysis(
    candidates: tuple[ForecastCandidate, ...],
    *,
    status: FuturePlaneAnalysisStatus = (
        FuturePlaneAnalysisStatus.COMPLETED
    ),
    multiplier_by_candidate: dict[str, float] | None = None,
    vetoed_plane_count: int = 0,
) -> FuturePlaneAnalysisResult:
    multipliers = multiplier_by_candidate or {}

    analyses = tuple(
        SimpleNamespace(
            candidate_id=candidate.candidate_id,
            plane_influence_set=PlaneInfluenceSet(
                influence_set_id=(
                    f"planes-{candidate.candidate_id}"
                ),
                label=f"Planes for {candidate.candidate_id}",
            ),
            mean_observability=0.90,
            applied_plane_count=(
                1
                if candidate.candidate_id in multipliers
                else 0
            ),
            combined_multiplier=multipliers.get(
                candidate.candidate_id,
                1.0,
            ),
        )
        for candidate in candidates
    )

    return raw_instance(
        FuturePlaneAnalysisResult,
        result_id="future-plane-result",
        status=status,
        candidate_analyses=analyses,
        vetoed_plane_count=vetoed_plane_count,
    )


def plane_analysis_for_candidate(
    self: FuturePlaneAnalysisResult,
    candidate_id: str,
):
    for analysis in self.candidate_analyses:
        if analysis.candidate_id == candidate_id:
            return analysis

    return None


def make_forecast_result(
    candidates: tuple[ForecastCandidate, ...],
    *,
    confidence: float = 0.75,
    status: ForecastStatus = ForecastStatus.COMPLETED,
) -> HistoryForecastResult:
    rankings = tuple(
        make_forecast_score(
            candidate,
            normalized_score=0.90 - index * 0.10,
            rank=index + 1,
        )
        for index, candidate in enumerate(candidates)
    )

    return raw_instance(
        HistoryForecastResult,
        result_id="baseline-forecast",
        status=status,
        rankings=rankings,
        confidence=confidence,
    )


def make_counterfactual_result(
    *,
    ranking_count: int = 1,
    confidence: float = 0.85,
    status: CounterfactualStatus = (
        CounterfactualStatus.COMPLETED
    ),
    target_id: str = "node:STAR",
) -> CounterfactualResult:
    rankings = tuple(
        SimpleNamespace(
            intervention_id=f"intervention-{index}",
            target_id=target_id if index == 0 else f"node:{index}",
        )
        for index in range(ranking_count)
    )

    return raw_instance(
        CounterfactualResult,
        result_id="counterfactual-result",
        status=status,
        rankings=rankings,
        confidence=confidence,
    )


def make_scenario(
    scenario_id: str,
) -> CounterfactualScenario:
    return raw_instance(
        CounterfactualScenario,
        scenario_id=scenario_id,
    )


def make_pipeline(
    *,
    config: HistoryPipelineConfig | None = None,
) -> HistoryPipeline:
    return HistoryPipeline(
        hypothesis_generator=object.__new__(
            HypothesisGenerator
        ),
        history_decoder=object.__new__(HistoryDecoder),
        history_forecaster=object.__new__(
            HistoryForecaster
        ),
        future_plane_analyzer=object.__new__(
            FuturePlaneAnalyzer
        ),
        counterfactual_engine=object.__new__(
            CounterfactualEngine
        ),
        config=config or HistoryPipelineConfig(),
    )


def install_common_stage_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation: HypothesisGenerationResult | None = None,
    decode: HistoryDecodeResult | None = None,
    candidates: tuple[ForecastCandidate, ...] | None = None,
    plane_analysis: FuturePlaneAnalysisResult | None = None,
    forecast: HistoryForecastResult | None = None,
    counterfactual: CounterfactualResult | None = None,
) -> tuple[
    HypothesisGenerationResult,
    HistoryDecodeResult,
    tuple[ForecastCandidate, ...],
    FuturePlaneAnalysisResult,
    HistoryForecastResult,
    CounterfactualResult,
]:
    generation = generation or make_generation()
    decode = decode or make_decode()
    candidates = candidates or (
        make_candidate("future-a"),
        make_candidate("future-b"),
    )
    plane_analysis = plane_analysis or make_plane_analysis(
        candidates
    )
    forecast = forecast or make_forecast_result(
        candidates
    )
    counterfactual = (
        counterfactual
        or make_counterfactual_result()
    )

    monkeypatch.setattr(
        HypothesisGenerator,
        "generate",
        lambda self, *args, **kwargs: generation,
    )
    monkeypatch.setattr(
        HistoryDecoder,
        "decode",
        lambda self, *args, **kwargs: decode,
    )
    monkeypatch.setattr(
        FuturePlaneAnalyzer,
        "analyze",
        lambda self, *args, **kwargs: plane_analysis,
    )
    monkeypatch.setattr(
        HistoryPipeline,
        "_aggregate_forecast",
        lambda self, *args, **kwargs: forecast,
    )
    monkeypatch.setattr(
        CounterfactualEngine,
        "evaluate",
        lambda self, *args, **kwargs: counterfactual,
    )

    return (
        generation,
        decode,
        candidates,
        plane_analysis,
        forecast,
        counterfactual,
    )


# ---------------------------------------------------------------------------
# Config, records, metrics, and result
# ---------------------------------------------------------------------------


def test_pipeline_stage_record_creation() -> None:
    record = PipelineStageRecord(
        stage=PipelineStage.GENERATE_HISTORY,
        status=StageStatus.COMPLETED,
        input_count=1,
        output_count=3,
        metadata={"source": "test"},
    )

    assert record.succeeded is True
    assert record.metadata["source"] == "test"


def test_pipeline_stage_record_is_immutable() -> None:
    record = PipelineStageRecord(
        stage=PipelineStage.GENERATE_HISTORY,
        status=StageStatus.COMPLETED,
    )

    with pytest.raises(FrozenInstanceError):
        record.status = StageStatus.FAILED


def test_pipeline_stage_record_metadata_is_read_only() -> None:
    record = PipelineStageRecord(
        stage=PipelineStage.GENERATE_HISTORY,
        status=StageStatus.COMPLETED,
    )

    with pytest.raises(TypeError):
        record.metadata["new"] = "value"


@pytest.mark.parametrize(
    "bad_value",
    (-1, 1.5),
)
def test_pipeline_stage_record_rejects_invalid_counts(
    bad_value,
) -> None:
    with pytest.raises(
        HistoryPipelineError,
        match="input_count",
    ):
        PipelineStageRecord(
            stage=PipelineStage.GENERATE_HISTORY,
            status=StageStatus.COMPLETED,
            input_count=bad_value,
        )


def test_pipeline_config_defaults() -> None:
    config = HistoryPipelineConfig()

    assert config.strict is True
    assert config.require_forecast is True
    assert config.require_intervention is False
    assert config.max_future_candidates is None


def test_pipeline_config_is_immutable() -> None:
    config = HistoryPipelineConfig()

    with pytest.raises(FrozenInstanceError):
        config.strict = False


@pytest.mark.parametrize(
    "field_name",
    (
        "strict",
        "require_forecast",
        "require_intervention",
    ),
)
def test_pipeline_config_rejects_nonboolean_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        HistoryPipelineError,
        match=field_name,
    ):
        HistoryPipelineConfig(
            **{field_name: 1}
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "max_future_candidates",
        "max_intervention_scenarios",
    ),
)
def test_pipeline_config_rejects_nonpositive_limits(
    field_name: str,
) -> None:
    with pytest.raises(
        HistoryPipelineError,
        match=field_name,
    ):
        HistoryPipelineConfig(
            **{field_name: 0}
        )


def test_pipeline_metrics_creation() -> None:
    metrics = HistoryPipelineMetrics(
        generated_hypotheses=3,
        ranked_futures=2,
        ranked_interventions=1,
        history_confidence=0.8,
        forecast_confidence=0.7,
        intervention_confidence=0.6,
    )

    assert metrics.generated_hypotheses == 3
    assert metrics.ranked_futures == 2
    assert metrics.to_dict()["version"] == "1.0.0"


def test_pipeline_metrics_rejects_negative_count() -> None:
    with pytest.raises(
        HistoryPipelineError,
        match="ranked_futures",
    ):
        HistoryPipelineMetrics(
            ranked_futures=-1,
        )


def test_pipeline_result_properties() -> None:
    decode = make_decode()
    candidates = (make_candidate("future-a"),)
    forecast = make_forecast_result(candidates)
    counterfactual = make_counterfactual_result(
        target_id="node:OPTIMAL"
    )

    result = HistoryPipelineResult(
        observed_signature_id="observed",
        status=PipelineStatus.COMPLETED,
        history_decode=decode,
        forecast_result=forecast,
        counterfactual_result=counterfactual,
    )

    assert result.best_history == decode.rankings[0]
    assert result.best_future == forecast.rankings[0]
    assert result.best_intervention == (
        counterfactual.rankings[0]
    )
    assert result.node_star == "node:OPTIMAL"


def test_pipeline_result_metadata_is_read_only() -> None:
    result = HistoryPipelineResult(
        observed_signature_id="observed",
        status=PipelineStatus.COMPLETED,
    )

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


def test_pipeline_result_to_dict() -> None:
    result = HistoryPipelineResult(
        result_id="pipeline-result",
        observed_signature_id="observed",
        status=PipelineStatus.COMPLETED,
    )

    data = result.to_dict()

    assert data["version"] == "1.0.0"
    assert data["result_id"] == "pipeline-result"
    assert data["status"] == "completed"


# ---------------------------------------------------------------------------
# Pipeline construction and validation
# ---------------------------------------------------------------------------


def test_pipeline_creation() -> None:
    pipeline = make_pipeline()

    assert pipeline.VERSION == "1.0.0"
    assert isinstance(
        pipeline.config,
        HistoryPipelineConfig,
    )


def test_pipeline_is_immutable() -> None:
    pipeline = make_pipeline()

    with pytest.raises(FrozenInstanceError):
        pipeline.config = HistoryPipelineConfig(
            strict=False
        )


@pytest.mark.parametrize(
    ("field_name", "value", "expected_name"),
    (
        (
            "hypothesis_generator",
            "generator",
            "HypothesisGenerator",
        ),
        (
            "history_decoder",
            "decoder",
            "HistoryDecoder",
        ),
        (
            "history_forecaster",
            "forecaster",
            "HistoryForecaster",
        ),
        (
            "future_plane_analyzer",
            "analyzer",
            "FuturePlaneAnalyzer",
        ),
        (
            "counterfactual_engine",
            "counterfactual",
            "CounterfactualEngine",
        ),
    ),
)
def test_pipeline_rejects_invalid_components(
    field_name: str,
    value,
    expected_name: str,
) -> None:
    values = {
        "hypothesis_generator": object.__new__(
            HypothesisGenerator
        ),
        "history_decoder": object.__new__(
            HistoryDecoder
        ),
        "history_forecaster": object.__new__(
            HistoryForecaster
        ),
        "future_plane_analyzer": object.__new__(
            FuturePlaneAnalyzer
        ),
        "counterfactual_engine": object.__new__(
            CounterfactualEngine
        ),
    }
    values[field_name] = value

    with pytest.raises(
        HistoryPipelineError,
        match=expected_name,
    ):
        HistoryPipeline(
            **values,
        )


def test_run_rejects_invalid_observed() -> None:
    pipeline = make_pipeline()

    with pytest.raises(
        HistoryPipelineError,
        match="observed must be",
    ):
        pipeline.run(
            "observed",
            history_forward_predictor=lambda *_: None,
            future_candidate_factory=lambda *_: (),
        )


def test_run_rejects_noncallable_predictor() -> None:
    pipeline = make_pipeline()

    with pytest.raises(
        HistoryPipelineError,
        match="history_forward_predictor",
    ):
        pipeline.run(
            make_observed(),
            history_forward_predictor="predictor",
            future_candidate_factory=lambda *_: (),
        )


def test_run_rejects_noncallable_future_factory() -> None:
    pipeline = make_pipeline()

    with pytest.raises(
        HistoryPipelineError,
        match="future_candidate_factory",
    ):
        pipeline.run(
            make_observed(),
            history_forward_predictor=lambda *_: None,
            future_candidate_factory="factory",
        )


def test_run_rejects_noncallable_intervention_factory() -> None:
    pipeline = make_pipeline()

    with pytest.raises(
        HistoryPipelineError,
        match="intervention_scenario_factory",
    ):
        pipeline.run(
            make_observed(),
            history_forward_predictor=lambda *_: None,
            future_candidate_factory=lambda *_: (),
            intervention_scenario_factory="factory",
        )


def test_run_rejects_invalid_future_plane() -> None:
    pipeline = make_pipeline()

    with pytest.raises(
        HistoryPipelineError,
        match="all future_planes",
    ):
        pipeline.run(
            make_observed(),
            history_forward_predictor=lambda *_: None,
            future_candidate_factory=lambda *_: (),
            future_planes=("plane",),
        )


# ---------------------------------------------------------------------------
# Pipeline execution paths
# ---------------------------------------------------------------------------


def test_complete_pipeline_without_interventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        generation,
        decode,
        candidates,
        plane_analysis,
        forecast,
        _,
    ) = install_common_stage_mocks(monkeypatch)

    result = make_pipeline().run(
        make_observed(),
        history_forward_predictor=lambda *_: make_observed(
            "predicted-history"
        ),
        future_candidate_factory=lambda *_: candidates,
        future_planes=(),
        metadata={"experiment": "pipeline-test"},
    )

    assert result.status is PipelineStatus.COMPLETED
    assert result.history_generation is generation
    assert result.history_decode is decode
    assert result.future_plane_analysis is plane_analysis
    assert result.forecast_result is forecast
    assert result.counterfactual_result is None
    assert result.metadata["experiment"] == "pipeline-test"
    assert result.stage_records[-1].status is StageStatus.SKIPPED


def test_complete_pipeline_with_interventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        candidates,
        _,
        _,
        counterfactual,
    ) = install_common_stage_mocks(monkeypatch)

    scenarios = (
        make_scenario("scenario-a"),
        make_scenario("scenario-b"),
    )

    result = make_pipeline().run(
        make_observed(),
        history_forward_predictor=lambda *_: make_observed(
            "predicted-history"
        ),
        future_candidate_factory=lambda *_: candidates,
        intervention_scenario_factory=lambda *_: scenarios,
    )

    assert result.status is PipelineStatus.COMPLETED
    assert result.counterfactual_result is counterfactual
    assert result.node_star == "node:STAR"
    assert result.metrics.intervention_scenarios == 2
    assert result.metrics.ranked_interventions == 1


def test_no_hypotheses_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = make_generation(
        hypothesis_count=0
    )

    monkeypatch.setattr(
        HypothesisGenerator,
        "generate",
        lambda self, *args, **kwargs: generation,
    )

    result = make_pipeline().run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: (),
    )

    assert result.status is PipelineStatus.NO_HYPOTHESES
    assert result.history_generation is generation
    assert result.history_decode is None
    assert len(result.stage_records) == 1
    assert result.stage_records[0].status is StageStatus.EMPTY


def test_no_future_candidates_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = make_generation()
    decode = make_decode()

    monkeypatch.setattr(
        HypothesisGenerator,
        "generate",
        lambda self, *args, **kwargs: generation,
    )
    monkeypatch.setattr(
        HistoryDecoder,
        "decode",
        lambda self, *args, **kwargs: decode,
    )

    result = make_pipeline().run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: (),
    )

    assert result.status is PipelineStatus.NO_FORECASTS
    assert result.history_decode is decode
    assert result.stage_records[-1].stage is (
        PipelineStage.BUILD_FUTURES
    )
    assert result.stage_records[-1].status is StageStatus.EMPTY


def test_no_future_candidates_optional_returns_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = make_generation()
    decode = make_decode()

    monkeypatch.setattr(
        HypothesisGenerator,
        "generate",
        lambda self, *args, **kwargs: generation,
    )
    monkeypatch.setattr(
        HistoryDecoder,
        "decode",
        lambda self, *args, **kwargs: decode,
    )

    pipeline = make_pipeline(
        config=HistoryPipelineConfig(
            require_forecast=False
        )
    )

    result = pipeline.run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: (),
    )

    assert result.status is PipelineStatus.PARTIAL


def test_required_intervention_without_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        candidates,
        _,
        _,
        _,
    ) = install_common_stage_mocks(monkeypatch)

    pipeline = make_pipeline(
        config=HistoryPipelineConfig(
            require_intervention=True
        )
    )

    result = pipeline.run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: candidates,
    )

    assert result.status is PipelineStatus.NO_INTERVENTIONS
    assert result.counterfactual_result is None


def test_empty_intervention_factory_returns_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        candidates,
        _,
        _,
        _,
    ) = install_common_stage_mocks(monkeypatch)

    result = make_pipeline().run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: candidates,
        intervention_scenario_factory=lambda *_: (),
    )

    assert result.status is PipelineStatus.PARTIAL
    assert result.stage_records[-1].stage is (
        PipelineStage.BUILD_INTERVENTIONS
    )
    assert result.stage_records[-1].status is StageStatus.EMPTY


def test_empty_intervention_factory_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        candidates,
        _,
        _,
        _,
    ) = install_common_stage_mocks(monkeypatch)

    pipeline = make_pipeline(
        config=HistoryPipelineConfig(
            require_intervention=True
        )
    )

    result = pipeline.run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: candidates,
        intervention_scenario_factory=lambda *_: (),
    )

    assert result.status is PipelineStatus.NO_INTERVENTIONS


def test_future_candidate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        make_candidate("future-a"),
        make_candidate("future-b"),
        make_candidate("future-c"),
    )
    captured: dict[str, int] = {}

    (
        _,
        _,
        _,
        _,
        _,
        _,
    ) = install_common_stage_mocks(
        monkeypatch,
        candidates=candidates,
    )

    def analyze(self, received, *args, **kwargs):
        received = tuple(received)
        captured["count"] = len(received)
        return make_plane_analysis(received)

    monkeypatch.setattr(
        FuturePlaneAnalyzer,
        "analyze",
        analyze,
    )

    pipeline = make_pipeline(
        config=HistoryPipelineConfig(
            max_future_candidates=2
        )
    )

    result = pipeline.run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: candidates,
    )

    assert result.status is PipelineStatus.COMPLETED
    assert captured["count"] == 2
    assert result.metrics.future_candidates == 2


def test_intervention_scenario_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        candidates,
        _,
        _,
        counterfactual,
    ) = install_common_stage_mocks(monkeypatch)

    captured: dict[str, int] = {}

    def evaluate(self, baseline, scenarios, **kwargs):
        captured["count"] = len(tuple(scenarios))
        return counterfactual

    monkeypatch.setattr(
        CounterfactualEngine,
        "evaluate",
        evaluate,
    )

    scenarios = tuple(
        make_scenario(f"scenario-{index}")
        for index in range(4)
    )

    pipeline = make_pipeline(
        config=HistoryPipelineConfig(
            max_intervention_scenarios=2
        )
    )

    result = pipeline.run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: candidates,
        intervention_scenario_factory=lambda *_: scenarios,
    )

    assert result.status is PipelineStatus.COMPLETED
    assert captured["count"] == 2
    assert result.metrics.intervention_scenarios == 2


def test_invalid_future_factory_output_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = make_generation()
    decode = make_decode()

    monkeypatch.setattr(
        HypothesisGenerator,
        "generate",
        lambda self, *args, **kwargs: generation,
    )
    monkeypatch.setattr(
        HistoryDecoder,
        "decode",
        lambda self, *args, **kwargs: decode,
    )

    with pytest.raises(
        HistoryPipelineError,
        match="ForecastCandidate",
    ):
        make_pipeline().run(
            make_observed(),
            history_forward_predictor=lambda *_: None,
            future_candidate_factory=lambda *_: ("future",),
        )


def test_invalid_future_factory_output_non_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = make_generation()
    decode = make_decode()

    monkeypatch.setattr(
        HypothesisGenerator,
        "generate",
        lambda self, *args, **kwargs: generation,
    )
    monkeypatch.setattr(
        HistoryDecoder,
        "decode",
        lambda self, *args, **kwargs: decode,
    )

    result = make_pipeline(
        config=HistoryPipelineConfig(
            strict=False
        )
    ).run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: ("future",),
    )

    assert result.status is PipelineStatus.FAILED
    assert result.metadata["failure_type"] == (
        "HistoryPipelineError"
    )
    assert result.stage_records[-1].status is StageStatus.FAILED


def test_invalid_intervention_factory_output_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        candidates,
        _,
        _,
        _,
    ) = install_common_stage_mocks(monkeypatch)

    with pytest.raises(
        HistoryPipelineError,
        match="CounterfactualScenario",
    ):
        make_pipeline().run(
            make_observed(),
            history_forward_predictor=lambda *_: None,
            future_candidate_factory=lambda *_: candidates,
            intervention_scenario_factory=lambda *_: (
                "scenario",
            ),
        )


# ---------------------------------------------------------------------------
# Candidate-specific future-plane aggregation
# ---------------------------------------------------------------------------


def test_aggregate_forecast_uses_candidate_specific_influences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        make_candidate("future-a"),
        make_candidate("future-b"),
    )
    plane_analysis = make_plane_analysis(
        candidates,
        multiplier_by_candidate={
            "future-a": 1.20,
            "future-b": 0.80,
        },
    )

    monkeypatch.setattr(
        FuturePlaneAnalysisResult,
        "for_candidate",
        plane_analysis_for_candidate,
    )
    monkeypatch.setattr(
        ForecastScore,
        "with_rank",
        clone_score_with_rank,
    )

    received_ids: list[str] = []

    def score_candidate(
        self,
        current,
        candidate,
        influences,
    ):
        received_ids.append(
            influences.influence_set_id
        )

        score = (
            0.90
            if candidate.candidate_id == "future-a"
            else 0.70
        )

        return make_forecast_score(
            candidate,
            normalized_score=score,
        )

    monkeypatch.setattr(
        HistoryForecaster,
        "score_candidate",
        score_candidate,
    )

    result = make_pipeline()._aggregate_forecast(
        make_observed(),
        candidates,
        plane_analysis,
    )

    assert received_ids == [
        "planes-future-a",
        "planes-future-b",
    ]
    assert result.rankings[0].candidate_id == "future-a"
    assert result.rankings[1].candidate_id == "future-b"


def test_aggregate_forecast_applies_minimum_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        make_candidate("future-a"),
        make_candidate("future-b"),
    )
    plane_analysis = make_plane_analysis(
        candidates
    )

    monkeypatch.setattr(
        FuturePlaneAnalysisResult,
        "for_candidate",
        plane_analysis_for_candidate,
    )
    monkeypatch.setattr(
        ForecastScore,
        "with_rank",
        clone_score_with_rank,
    )

    def score_candidate(
        self,
        current,
        candidate,
        influences,
    ):
        return make_forecast_score(
            candidate,
            normalized_score=(
                0.80
                if candidate.candidate_id == "future-a"
                else 0.20
            ),
        )

    monkeypatch.setattr(
        HistoryForecaster,
        "score_candidate",
        score_candidate,
    )

    pipeline = make_pipeline(
        config=HistoryPipelineConfig(
            forecast_minimum_score=0.50
        )
    )

    result = pipeline._aggregate_forecast(
        make_observed(),
        candidates,
        plane_analysis,
    )

    assert len(result.rankings) == 1
    assert result.best.candidate_id == "future-a"


def test_aggregate_forecast_detects_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        make_candidate("future-a"),
        make_candidate("future-b"),
    )
    plane_analysis = make_plane_analysis(
        candidates
    )

    monkeypatch.setattr(
        FuturePlaneAnalysisResult,
        "for_candidate",
        plane_analysis_for_candidate,
    )
    monkeypatch.setattr(
        ForecastScore,
        "with_rank",
        clone_score_with_rank,
    )

    def score_candidate(
        self,
        current,
        candidate,
        influences,
    ):
        return make_forecast_score(
            candidate,
            normalized_score=(
                0.80
                if candidate.candidate_id == "future-a"
                else 0.79
            ),
        )

    monkeypatch.setattr(
        HistoryForecaster,
        "score_candidate",
        score_candidate,
    )

    result = make_pipeline(
        config=HistoryPipelineConfig(
            forecast_ambiguity_threshold=0.05
        )
    )._aggregate_forecast(
        make_observed(),
        candidates,
        plane_analysis,
    )

    assert result.status is ForecastStatus.AMBIGUOUS
    assert result.selection_margin == pytest.approx(
        0.01
    )


def test_pipeline_metrics_are_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = make_generation(
        hypothesis_count=3,
        failed_count=1,
    )
    decode = make_decode(
        ranking_count=3,
        confidence=0.81,
    )
    candidates = (
        make_candidate("future-a"),
        make_candidate("future-b"),
    )
    plane_analysis = make_plane_analysis(
        candidates,
        multiplier_by_candidate={
            "future-a": 1.2,
            "future-b": 0.8,
        },
        vetoed_plane_count=1,
    )
    forecast = make_forecast_result(
        candidates,
        confidence=0.72,
    )
    counterfactual = make_counterfactual_result(
        ranking_count=2,
        confidence=0.63,
    )

    install_common_stage_mocks(
        monkeypatch,
        generation=generation,
        decode=decode,
        candidates=candidates,
        plane_analysis=plane_analysis,
        forecast=forecast,
        counterfactual=counterfactual,
    )

    scenarios = (
        make_scenario("scenario-a"),
        make_scenario("scenario-b"),
    )

    result = make_pipeline().run(
        make_observed(),
        history_forward_predictor=lambda *_: None,
        future_candidate_factory=lambda *_: candidates,
        intervention_scenario_factory=lambda *_: scenarios,
    )

    metrics = result.metrics

    assert metrics.generated_hypotheses == 3
    assert metrics.failed_history_predictions == 1
    assert metrics.decoded_rankings == 3
    assert metrics.future_candidates == 2
    assert (
        metrics.applied_future_plane_contributions
        == 2
    )
    assert metrics.vetoed_future_planes == 1
    assert metrics.ranked_futures == 2
    assert metrics.intervention_scenarios == 2
    assert metrics.ranked_interventions == 2
    assert metrics.history_confidence == pytest.approx(
        0.81
    )
    assert metrics.forecast_confidence == pytest.approx(
        0.72
    )
    assert (
        metrics.intervention_confidence
        == pytest.approx(0.63)
    )
