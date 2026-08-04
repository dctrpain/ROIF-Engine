"""
ROIF history subsystem.

This package preserves the original cascade-history public API and extends it
with the Structural Memory event model.
"""

from .cascade_history import *
from .event import (
    HistoryEvent,
    HistoryEventKind,
    HistoryTarget,
    HistoryTargetKind,
    HistoryValidationError,
    ReversibilityClass,
    StateDelta,
    TimeScale,
)

from .irreversible_change import (
    FunctionalEffect,
    IrreversibleChange,
    IrreversibleChangeError,
    IrreversibleChangeKind,
    TracePersistence,
)
from .history_pattern import (
    HistoryPattern,
    HistoryPatternError,
    RankedContribution,
)
from .structural_signature import (
    SignatureComparison,
    SignatureDistanceWeights,
    StructuralSignature,
    StructuralSignatureError,
)
from .history_decoder import (
    DecodeStatus,
    DecoderWeights,
    HistoryDecodeResult,
    HistoryDecoder,
    HistoryDecoderError,
    HistoryHypothesis,
    HypothesisScore,
    HypothesisStatus,
)
from .hypothesis_generator import (
    CandidateStatus,
    CausalRule,
    GeneratedCandidate,
    HypothesisGenerationResult,
    HypothesisGenerator,
    HypothesisGeneratorError,
    HypothesisSeed,
    RuleMatchMode,
    SignaturePredictor,
)
from .history_forecast import (
    ForecastCandidate,
    ForecastCandidateStatus,
    ForecastDirection,
    ForecastHorizon,
    ForecastScore,
    ForecastStatus,
    ForecastWeights,
    HistoryForecastError,
    HistoryForecastResult,
    HistoryForecaster,
    PlaneInfluence,
    PlaneInfluenceSet,
)
from .counterfactual import (
    CounterfactualEngine,
    CounterfactualError,
    CounterfactualResult,
    CounterfactualScenario,
    CounterfactualScore,
    CounterfactualStatus,
    CounterfactualWeights,
    Intervention,
    InterventionScope,
    InterventionStatus,
)
from .future_plane_analyzer import (
    CandidatePlaneAnalysis,
    FuturePlane,
    FuturePlaneAnalysisResult,
    FuturePlaneAnalysisStatus,
    FuturePlaneAnalyzer,
    FuturePlaneAnalyzerError,
    FuturePlaneContribution,
    FuturePlaneDirection,
    FuturePlaneScope,
    FuturePlaneStatus,
    TemporalProfile,
)
from .history_pipeline import (
    FutureCandidateFactory,
    HistoryForwardPredictor,
    HistoryPipeline,
    HistoryPipelineConfig,
    HistoryPipelineError,
    HistoryPipelineMetrics,
    HistoryPipelineResult,
    InterventionScenarioFactory,
    PipelineStage,
    PipelineStageRecord,
    PipelineStatus,
    StageStatus,
)