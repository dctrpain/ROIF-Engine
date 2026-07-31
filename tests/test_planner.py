"""Tests for ROIF planner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest

from roif.analyzer import (
    AnalysisRisk, CascadeAnalysis, CascadePhase, CoherenceState,
    PlaneDominance, ScalarTrend, TrendDirection,
)
from roif.planner import (
    CandidateDecision, CascadePlanner, InterventionCandidate,
    InterventionPlan, PlanStatus, PlannerConfig, PlannerError,
    RankedIntervention, ScoreBreakdown, TargetRole, VetoReason,
    plan_interventions,
)
from roif.predictor import (
    CascadePrediction, DFastForecast, ForecastInterval,
    PredictionStatus, StabilityForecast,
)


def trend() -> ScalarTrend:
    return ScalarTrend(TrendDirection.STABLE, 0.5, 0.5, 0.0, 0.0, 2)


def analysis(
    phase: CascadePhase = CascadePhase.QUIESCENT,
    risk: AnalysisRisk = AnalysisRisk.LOW,
    coherence: CoherenceState = CoherenceState.STABLE,
    dominant: str | None = "A",
) -> CascadeAnalysis:
    return CascadeAnalysis(
        3, phase, risk, coherence, trend(), trend(), trend(),
        None if dominant is None else PlaneDominance(dominant, 2, 2 / 3),
        1, 0, 0, 0, "s1", "s3", "Analysis summary.",
    )


def prediction(
    status: PredictionStatus = PredictionStatus.AVAILABLE,
    stability: StabilityForecast = StabilityForecast.STABLE,
    probability: float | None = 0.1,
) -> CascadePrediction:
    available = status is PredictionStatus.AVAILABLE
    return CascadePrediction(
        status=status,
        stability=stability,
        source_observation_count=3,
        source_point_count=3 if available else 1,
        last_time=3.0,
        forecast_time=4.0 if available else None,
        eta=ForecastInterval(0.5, 0.4, 0.6, 0.05) if available else None,
        eta_slope=0.0 if available else None,
        critical_probability=probability if available else None,
        first_crossing=None,
        d_fast=DFastForecast("A", 1.0, 0.0, 3),
        source_phase=CascadePhase.QUIESCENT,
        summary="Prediction summary.",
    )


def candidate(
    candidate_id: str = "c1",
    *,
    target_id: str = "A",
    target_role: TargetRole = TargetRole.NODE_STAR,
    expected_eta_gain: float = 0.2,
    cost: float = 1.0,
    risk: float = 0.1,
    delay: float = 1.0,
    evidence_strength: float = 0.8,
    reversibility: float = 0.9,
    cascade_risk: float = 0.1,
    external_system_risk: float = 0.0,
) -> InterventionCandidate:
    return InterventionCandidate(
        candidate_id, target_id, target_role, expected_eta_gain,
        cost, risk, delay, evidence_strength, reversibility,
        cascade_risk, external_system_risk, "Test candidate.",
        ("test",), {"source": "unit"},
    )


def score(utility: float = 0.5) -> ScoreBreakdown:
    return ScoreBreakdown(1.0, 0.5, 1.0, 0.8, 0.9, 0.2, 0.1, 0.2, utility)


def ranked(
    decision: CandidateDecision = CandidateDecision.RECOMMENDED,
    rank: int | None = 1,
    *,
    reasons: tuple[VetoReason, ...] = (),
    with_score: bool = True,
) -> RankedIntervention:
    return RankedIntervention(
        rank, decision, candidate(),
        score() if with_score else None,
        reasons, "Explanation.",
    )


def test_enum_values() -> None:
    assert [x.value for x in PlanStatus] == [
        "available", "no_candidates", "all_vetoed", "insufficient_context"
    ]
    assert [x.value for x in CandidateDecision] == [
        "recommended", "alternative", "rejected", "vetoed"
    ]
    assert [x.value for x in TargetRole] == [
        "node_star", "d_fast", "d_root", "support", "unknown"
    ]


def test_config_defaults() -> None:
    config = PlannerConfig()
    assert config.weight_sum == pytest.approx(1.0)
    assert config.non_fonit_gate_enabled
    assert config.max_alternatives == 3


def test_config_normalization_and_immutability() -> None:
    config = PlannerConfig(benefit_weight="1", maximum_cascade_risk="0.5")
    assert config.benefit_weight == pytest.approx(1.0)
    with pytest.raises(FrozenInstanceError):
        config.benefit_weight = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("field", [
    "benefit_weight", "urgency_weight", "target_weight", "evidence_weight",
    "reversibility_weight", "cost_weight", "risk_weight", "delay_weight",
])
@pytest.mark.parametrize("value", [-0.1, float("nan"), True, "bad"])
def test_invalid_weights(field: str, value: Any) -> None:
    with pytest.raises(PlannerError):
        PlannerConfig(**{field: value})


def test_all_zero_weights_rejected() -> None:
    with pytest.raises(PlannerError):
        PlannerConfig(
            benefit_weight=0, urgency_weight=0, target_weight=0,
            evidence_weight=0, reversibility_weight=0, cost_weight=0,
            risk_weight=0, delay_weight=0,
        )


@pytest.mark.parametrize("field", [
    "maximum_cascade_risk", "maximum_external_system_risk",
    "minimum_evidence_strength", "irreversible_risk_threshold",
])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "bad"])
def test_invalid_config_unit_values(field: str, value: Any) -> None:
    with pytest.raises(PlannerError):
        PlannerConfig(**{field: value})


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_invalid_max_alternatives(value: Any) -> None:
    with pytest.raises(PlannerError):
        PlannerConfig(max_alternatives=value)


@pytest.mark.parametrize("field", ["require_prediction", "non_fonit_gate_enabled"])
@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_boolean_flags_required(field: str, value: Any) -> None:
    with pytest.raises(PlannerError):
        PlannerConfig(**{field: value})


def test_candidate_properties_and_dict() -> None:
    item = InterventionCandidate(
        " c1 ", " A ", TargetRole.NODE_STAR, "0.2",
        tags=("x", "x", " y "), metadata={"source": "test"},
    )
    assert item.candidate_id == "c1"
    assert item.target_id == "A"
    assert item.tags == ("x", "y")
    assert item.is_reversible
    assert item.as_dict()["target_role"] == "node_star"
    assert isinstance(item.metadata, MappingProxyType)


def test_candidate_metadata_copied() -> None:
    source = {"x": 1}
    item = InterventionCandidate("c", "A", TargetRole.NODE_STAR, 0.1, metadata=source)
    source["x"] = 2
    assert item.metadata["x"] == 1
    with pytest.raises(TypeError):
        item.metadata["y"] = 2  # type: ignore[index]


@pytest.mark.parametrize("field", ["candidate_id", "target_id"])
@pytest.mark.parametrize("value", ["", "   ", None, 1])
def test_invalid_candidate_ids(field: str, value: Any) -> None:
    data = dict(candidate_id="c", target_id="A", target_role=TargetRole.NODE_STAR, expected_eta_gain=0.1)
    data[field] = value
    with pytest.raises(PlannerError):
        InterventionCandidate(**data)


@pytest.mark.parametrize("value", ["node_star", None, 1])
def test_invalid_target_role(value: Any) -> None:
    with pytest.raises(PlannerError):
        InterventionCandidate("c", "A", value, 0.1)


@pytest.mark.parametrize("field", [
    "risk", "evidence_strength", "reversibility",
    "cascade_risk", "external_system_risk",
])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "bad"])
def test_invalid_candidate_unit_fields(field: str, value: Any) -> None:
    data = dict(candidate_id="c", target_id="A", target_role=TargetRole.NODE_STAR, expected_eta_gain=0.1)
    data[field] = value
    with pytest.raises(PlannerError):
        InterventionCandidate(**data)


@pytest.mark.parametrize("field", ["cost", "delay"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), True, "bad"])
def test_invalid_candidate_nonnegative_fields(field: str, value: Any) -> None:
    data = dict(candidate_id="c", target_id="A", target_role=TargetRole.NODE_STAR, expected_eta_gain=0.1)
    data[field] = value
    with pytest.raises(PlannerError):
        InterventionCandidate(**data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "bad"])
def test_invalid_eta_gain(value: Any) -> None:
    with pytest.raises(PlannerError):
        InterventionCandidate("c", "A", TargetRole.NODE_STAR, value)


@pytest.mark.parametrize("value", ["tag", (1,), ("",)])
def test_invalid_tags(value: Any) -> None:
    with pytest.raises(PlannerError):
        InterventionCandidate("c", "A", TargetRole.NODE_STAR, 0.1, tags=value)


def test_score_properties_and_dict() -> None:
    item = score()
    assert item.utility == pytest.approx(0.5)
    assert item.as_dict()["benefit"] == pytest.approx(1.0)


@pytest.mark.parametrize("field", [
    "benefit", "urgency", "target_alignment", "evidence",
    "reversibility", "cost_penalty", "risk_penalty", "delay_penalty",
])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "bad"])
def test_invalid_score_components(field: str, value: Any) -> None:
    data = score().as_dict()
    data[field] = value
    with pytest.raises(PlannerError):
        ScoreBreakdown(**data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "bad"])
def test_invalid_utility(value: Any) -> None:
    data = score().as_dict()
    data["utility"] = value
    with pytest.raises(PlannerError):
        ScoreBreakdown(**data)


def test_ranked_properties_and_dict() -> None:
    item = ranked()
    assert item.rank == 1
    assert item.as_dict()["decision"] == "recommended"


def test_valid_vetoed_ranked_item() -> None:
    item = ranked(
        CandidateDecision.VETOED, None,
        reasons=(VetoReason.CASCADE_RISK,), with_score=False,
    )
    assert item.score is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_invalid_rank(value: Any) -> None:
    with pytest.raises(PlannerError):
        ranked(rank=value)


def test_vetoed_requires_reasons() -> None:
    with pytest.raises(PlannerError):
        ranked(CandidateDecision.VETOED, None, with_score=False)


def test_vetoed_disallows_rank_and_score() -> None:
    with pytest.raises(PlannerError):
        ranked(CandidateDecision.VETOED, 1, reasons=(VetoReason.CASCADE_RISK,))


def test_non_vetoed_requires_rank_score_and_no_reasons() -> None:
    with pytest.raises(PlannerError):
        ranked(CandidateDecision.ALTERNATIVE, None, with_score=False)
    with pytest.raises(PlannerError):
        ranked(reasons=(VetoReason.CASCADE_RISK,))


def test_plan_object_properties_and_dict() -> None:
    item = ranked()
    plan = InterventionPlan(
        PlanStatus.AVAILABLE, CascadePhase.QUIESCENT, AnalysisRisk.LOW,
        StabilityForecast.STABLE, item, (), (), (), 1, " Plan summary. ",
        {"source": "test"},
    )
    assert plan.is_available
    assert plan.node_star == "A"
    assert plan.summary == "Plan summary."
    assert plan.as_dict()["recommended"]["candidate"]["candidate_id"] == "c1"


def test_planner_validation() -> None:
    with pytest.raises(PlannerError):
        CascadePlanner({})
    with pytest.raises(PlannerError):
        CascadePlanner().plan(None, ())
    with pytest.raises(PlannerError):
        CascadePlanner().plan(analysis(), (), prediction="bad")
    with pytest.raises(PlannerError):
        CascadePlanner().plan(analysis(), "bad")
    with pytest.raises(PlannerError):
        CascadePlanner().plan(analysis(), ("bad",))
    with pytest.raises(PlannerError):
        CascadePlanner().plan(analysis(), (candidate("c"), candidate("c")))


def test_no_candidates() -> None:
    plan = CascadePlanner().plan(analysis(), ())
    assert plan.status is PlanStatus.NO_CANDIDATES
    assert plan.node_star is None


def test_prediction_requirement() -> None:
    planner = CascadePlanner(PlannerConfig(require_prediction=True))
    assert planner.plan(analysis(), (candidate(),)).status is PlanStatus.INSUFFICIENT_CONTEXT
    unavailable = prediction(
        PredictionStatus.INSUFFICIENT_DATA,
        StabilityForecast.UNKNOWN,
        None,
    )
    assert planner.plan(
        analysis(), (candidate(),), prediction=unavailable
    ).status is PlanStatus.INSUFFICIENT_CONTEXT


def test_available_plan() -> None:
    plan = CascadePlanner().plan(analysis(), (candidate(),))
    assert plan.status is PlanStatus.AVAILABLE
    assert plan.recommended.candidate.candidate_id == "c1"
    assert "Recommended candidate: c1" in plan.summary


def test_higher_gain_wins() -> None:
    plan = CascadePlanner().plan(analysis(), (
        candidate("low", expected_eta_gain=0.1),
        candidate("high", expected_eta_gain=0.3),
    ))
    assert plan.recommended.candidate.candidate_id == "high"


def test_lower_risk_wins_equal_case() -> None:
    plan = CascadePlanner().plan(analysis(), (
        candidate("high-risk", risk=0.4),
        candidate("low-risk", risk=0.1),
    ))
    assert plan.recommended.candidate.candidate_id == "low-risk"


def test_node_star_alignment_wins() -> None:
    plan = CascadePlanner().plan(analysis(), (
        candidate("support", target_role=TargetRole.SUPPORT),
        candidate("node", target_role=TargetRole.NODE_STAR),
    ))
    assert plan.recommended.candidate.candidate_id == "node"


def test_matching_d_fast_alignment() -> None:
    config = PlannerConfig(
        benefit_weight=0, urgency_weight=0, target_weight=1,
        evidence_weight=0, reversibility_weight=0, cost_weight=0,
        risk_weight=0, delay_weight=0,
    )
    plan = CascadePlanner(config).plan(analysis(dominant="A"), (
        candidate("match", target_id="A", target_role=TargetRole.D_FAST),
        candidate("miss", target_id="B", target_role=TargetRole.D_FAST),
    ))
    assert plan.recommended.candidate.candidate_id == "match"


def test_tie_breaker_uses_id() -> None:
    plan = CascadePlanner().plan(analysis(), (candidate("b"), candidate("a")))
    assert plan.recommended.candidate.candidate_id == "a"


def test_alternative_limit_and_rejection() -> None:
    plan = CascadePlanner(PlannerConfig(max_alternatives=1)).plan(analysis(), (
        candidate("c1", expected_eta_gain=0.4),
        candidate("c2", expected_eta_gain=0.3),
        candidate("c3", expected_eta_gain=0.2),
    ))
    assert len(plan.alternatives) == 1
    assert len(plan.rejected) == 1
    assert plan.alternatives[0].rank == 2
    assert plan.rejected[0].rank == 3


def test_veto_rules() -> None:
    cases = [
        (candidate(cascade_risk=0.9), VetoReason.CASCADE_RISK),
        (candidate(external_system_risk=0.9), VetoReason.EXTERNAL_SYSTEM_RISK),
        (candidate(risk=0.9, reversibility=0.1), VetoReason.IRREVERSIBLE_HIGH_RISK),
        (candidate(expected_eta_gain=0.0), VetoReason.NON_POSITIVE_GAIN),
    ]
    for item, reason in cases:
        plan = CascadePlanner().plan(analysis(), (item,))
        assert plan.status is PlanStatus.ALL_VETOED
        assert reason in plan.vetoed[0].veto_reasons


def test_low_evidence_veto() -> None:
    plan = CascadePlanner(
        PlannerConfig(minimum_evidence_strength=0.9)
    ).plan(analysis(), (candidate(evidence_strength=0.2),))
    assert VetoReason.BELOW_MINIMUM_EVIDENCE in plan.vetoed[0].veto_reasons


def test_non_fonit_gate_can_be_disabled() -> None:
    plan = CascadePlanner(
        PlannerConfig(non_fonit_gate_enabled=False)
    ).plan(analysis(), (
        candidate(
            cascade_risk=1.0, external_system_risk=1.0,
            risk=1.0, reversibility=0.0,
        ),
    ))
    assert plan.status is PlanStatus.AVAILABLE


def test_mixed_safe_and_vetoed() -> None:
    plan = CascadePlanner().plan(analysis(), (
        candidate("safe"),
        candidate("unsafe", cascade_risk=0.9),
    ))
    assert plan.recommended.candidate.candidate_id == "safe"
    assert len(plan.vetoed) == 1
    assert plan.candidate_count == 2


def test_critical_context_has_full_urgency() -> None:
    plan = CascadePlanner().plan(
        analysis(CascadePhase.SUPERCRITICAL, AnalysisRisk.CRITICAL, CoherenceState.CRITICAL),
        (candidate(),),
        prediction=prediction(stability=StabilityForecast.CRITICAL, probability=0.99),
    )
    assert plan.recommended.score.urgency == pytest.approx(1.0)


def test_metadata_forwarding() -> None:
    source = {"experiment": "E1"}
    plan = CascadePlanner().plan(analysis(), (candidate(),), metadata=source)
    source["experiment"] = "changed"
    assert plan.metadata["experiment"] == "E1"
    with pytest.raises(TypeError):
        plan.metadata["x"] = 1  # type: ignore[index]


def test_explanations() -> None:
    safe = CascadePlanner().plan(analysis(), (candidate(),))
    assert "Expected eta gain" in safe.recommended.explanation
    vetoed = CascadePlanner().plan(analysis(), (candidate(cascade_risk=1.0),))
    assert "Non-Fonit/safety gate" in vetoed.vetoed[0].explanation


def test_wrapper() -> None:
    plan = plan_interventions(
        analysis(), (candidate(),),
        prediction=prediction(),
        metadata={"wrapper": True},
    )
    assert isinstance(plan, InterventionPlan)
    assert plan.metadata["wrapper"] is True


def test_wrapper_forwards_config() -> None:
    plan = plan_interventions(
        analysis(),
        (candidate("c1", expected_eta_gain=0.3), candidate("c2", expected_eta_gain=0.2)),
        config=PlannerConfig(max_alternatives=0),
    )
    assert plan.alternatives == ()
    assert len(plan.rejected) == 1
