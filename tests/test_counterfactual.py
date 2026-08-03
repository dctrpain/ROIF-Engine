from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.counterfactual import (
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
from roif.history.history_forecast import (
    ForecastCandidate,
    ForecastCandidateStatus,
    ForecastDirection,
    ForecastHorizon,
    ForecastScore,
    ForecastStatus,
    HistoryForecastResult,
)
from roif.history.structural_signature import (
    SignatureComparison,
    SignatureDistanceWeights,
    StructuralSignature,
)


def make_signature(
    signature_id: str = "signature-baseline",
    *,
    total_capacity_loss: float = 0.30,
    total_capacity_gain: float = 0.05,
    **overrides,
) -> StructuralSignature:
    data = {
        "signature_id": signature_id,
        "source_pattern_id": f"pattern-{signature_id}",
        "label": signature_id,
        "plane_profile": {
            "mechanical": 0.70,
            "biological": 0.30,
        },
        "agent_profile": {
            "repeated_load": 0.75,
            "adaptation": 0.25,
        },
        "kind_profile": {
            "damage": 0.70,
            "remodeling": 0.30,
        },
        "time_scale_profile": {
            "slow": 0.70,
            "very_slow": 0.30,
        },
        "target_profile": {
            "node:A": 0.60,
            "node:B": 0.40,
        },
        "total_changes": 3,
        "root_count": 1,
        "leaf_count": 1,
        "causal_depth": 3,
        "start_time": 0.0,
        "end_time": 10.0,
        "duration": 10.0,
        "total_capacity_loss": total_capacity_loss,
        "total_capacity_gain": total_capacity_gain,
        "net_capacity_effect": (
            total_capacity_gain
            - total_capacity_loss
        ),
        "cumulative_signature_weight": 1.80,
        "mean_signature_weight": 0.60,
        "persistence_index": 0.75,
        "irreversibility_index": 0.60,
        "progression_index": 0.80,
        "adaptation_index": 0.20,
        "harmful_fraction": 0.75,
        "beneficial_fraction": 0.20,
        "mixed_fraction": 0.05,
        "metadata": {
            "source": "counterfactual-test",
        },
    }
    data.update(overrides)
    return StructuralSignature(**data)


def make_forecast_candidate(
    candidate_id: str,
    *,
    signature: StructuralSignature,
    name: str = "Predicted future",
) -> ForecastCandidate:
    return ForecastCandidate(
        candidate_id=candidate_id,
        name=name,
        horizon=ForecastHorizon(
            start=0.0,
            end=30.0,
            units="days",
        ),
        predicted_signature=signature,
        direction=ForecastDirection.MIXED,
        affected_plane_ids=("mechanical",),
        affected_target_ids=("node:A",),
        predicted_change_kinds=("remodeling",),
        prior_probability=0.60,
        validation_score=0.90,
        complexity=1.0,
        status=ForecastCandidateStatus.FORWARD_VALIDATED,
    )


def make_forecast_result(
    result_id: str,
    *,
    signature: StructuralSignature,
    confidence: float = 0.80,
    score: float = 0.80,
) -> HistoryForecastResult:
    candidate = make_forecast_candidate(
        f"candidate-{result_id}",
        signature=signature,
    )

    comparison = SignatureComparison(
        left_signature_id="current",
        right_signature_id=signature.signature_id,
        distance=0.20,
        similarity=0.80,
        components={
            "capacity": 0.20,
        },
        weights=SignatureDistanceWeights(),
    )

    forecast_score = ForecastScore(
        candidate=candidate,
        continuity_comparison=comparison,
        continuity_score=0.80,
        prior_score=0.60,
        validation_score=0.90,
        plane_support_score=0.70,
        plane_multiplier=1.20,
        complexity_penalty=1.0 / 6.0,
        custom_penalty=0.0,
        raw_score=score,
        normalized_score=score,
        rank=1,
    )

    return HistoryForecastResult(
        result_id=result_id,
        current_signature_id="current-signature",
        influence_set_id="influence-set",
        status=ForecastStatus.COMPLETED,
        rankings=(forecast_score,),
        confidence=confidence,
        ambiguity=0.20,
        selection_margin=0.80,
        metadata={
            "source": "test",
        },
    )


def make_intervention(
    intervention_id: str = "intervention-a",
    **overrides,
) -> Intervention:
    data = {
        "intervention_id": intervention_id,
        "name": f"Intervention {intervention_id}",
        "target_id": "node:A",
        "scope": InterventionScope.NODE,
        "description": "Bounded test intervention.",
        "affected_plane_ids": (
            "mechanical",
            "biological",
        ),
        "affected_target_ids": (
            "node:A",
        ),
        "magnitude": 0.25,
        "cost": 0.20,
        "operational_risk": 0.10,
        "cascade_risk": 0.05,
        "reversibility": 0.90,
        "complexity": 1.0,
        "non_fonit_allowed": True,
        "status": InterventionStatus.FORWARD_VALIDATED,
        "metadata": {
            "domain": "test",
        },
    }
    data.update(overrides)
    return Intervention(**data)


def make_scenario(
    scenario_id: str = "scenario-a",
    *,
    intervention: Intervention | None = None,
    signature: StructuralSignature | None = None,
    validation_score: float = 0.90,
    model_penalty: float = 0.05,
) -> CounterfactualScenario:
    if intervention is None:
        intervention = make_intervention(
            intervention_id=f"intervention-{scenario_id}"
        )

    if signature is None:
        signature = make_signature(
            signature_id=f"signature-{scenario_id}",
            total_capacity_loss=0.20,
            total_capacity_gain=0.10,
        )

    return CounterfactualScenario(
        scenario_id=scenario_id,
        intervention=intervention,
        forecast_result=make_forecast_result(
            f"forecast-{scenario_id}",
            signature=signature,
        ),
        validation_score=validation_score,
        model_penalty=model_penalty,
        metadata={
            "scenario": scenario_id,
        },
    )


def make_score(
    *,
    rank: int = 1,
    normalized_score: float = 0.80,
) -> CounterfactualScore:
    scenario = make_scenario()

    return CounterfactualScore(
        scenario=scenario,
        baseline_net_capacity=-0.25,
        intervention_net_capacity=-0.10,
        net_capacity_delta=0.15,
        capacity_benefit_score=0.80,
        loss_reduction_score=0.70,
        gain_increase_score=0.75,
        forecast_quality_score=0.80,
        validation_score=0.90,
        cost_penalty=0.20,
        safety_penalty=0.10,
        complexity_penalty=1.0 / 6.0,
        model_penalty=0.05,
        raw_score=normalized_score,
        normalized_score=normalized_score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------


def test_intervention_creation() -> None:
    intervention = make_intervention()

    assert intervention.intervention_id == "intervention-a"
    assert intervention.target_id == "node:A"
    assert intervention.scope is InterventionScope.NODE
    assert intervention.status is InterventionStatus.FORWARD_VALIDATED
    assert intervention.is_eligible is True


def test_intervention_accepts_string_enums() -> None:
    intervention = make_intervention(
        scope="edge",
        status="ready",
    )

    assert intervention.scope is InterventionScope.EDGE
    assert intervention.status is InterventionStatus.READY


def test_intervention_is_immutable() -> None:
    intervention = make_intervention()

    with pytest.raises(FrozenInstanceError):
        intervention.cost = 2.0


def test_intervention_metadata_is_read_only() -> None:
    intervention = make_intervention()

    with pytest.raises(TypeError):
        intervention.metadata["new"] = "value"


def test_intervention_ids_are_deduplicated() -> None:
    intervention = make_intervention(
        affected_plane_ids=(
            "mechanical",
            "biological",
            "mechanical",
        ),
        affected_target_ids=(
            "node:A",
            "node:B",
            "node:A",
        ),
    )

    assert intervention.affected_plane_ids == (
        "mechanical",
        "biological",
    )
    assert intervention.affected_target_ids == (
        "node:A",
        "node:B",
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        (InterventionStatus.READY, True),
        (InterventionStatus.FORWARD_VALIDATED, True),
        (InterventionStatus.INCOMPLETE, True),
        (InterventionStatus.REJECTED, False),
        (InterventionStatus.FAILED, False),
        (InterventionStatus.VETOED, False),
    ),
)
def test_intervention_eligibility_by_status(
    status: InterventionStatus,
    expected: bool,
) -> None:
    intervention = make_intervention(
        status=status,
    )

    assert intervention.is_eligible is expected


def test_non_fonit_gate_vetoes_intervention() -> None:
    intervention = make_intervention(
        non_fonit_allowed=False,
    )

    assert intervention.is_eligible is False


def test_intervention_safety_penalty() -> None:
    intervention = make_intervention(
        operational_risk=0.30,
        cascade_risk=0.60,
        reversibility=0.60,
    )

    assert intervention.safety_penalty == pytest.approx(
        (0.30 + 0.60 + 0.40) / 3.0
    )


def test_intervention_round_trip() -> None:
    intervention = make_intervention()

    restored = Intervention.from_dict(
        intervention.to_dict()
    )

    assert restored == intervention
    assert restored is not intervention


def test_intervention_serialized_version() -> None:
    data = make_intervention().to_dict()

    assert data["version"] == "1.0.0"
    assert data["scope"] == "node"
    assert data["status"] == "forward_validated"


def test_intervention_from_dict_rejects_version() -> None:
    data = make_intervention().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        CounterfactualError,
        match="unsupported intervention version",
    ):
        Intervention.from_dict(data)


@pytest.mark.parametrize(
    "field_name",
    (
        "magnitude",
        "cost",
        "complexity",
    ),
)
def test_intervention_rejects_negative_nonnegative_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        CounterfactualError,
        match=field_name,
    ):
        make_intervention(
            **{
                field_name: -0.01,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "operational_risk",
        "cascade_risk",
        "reversibility",
    ),
)
def test_intervention_rejects_invalid_unit_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        CounterfactualError,
        match=field_name,
    ):
        make_intervention(
            **{
                field_name: 1.01,
            }
        )


def test_intervention_rejects_nonboolean_gate() -> None:
    with pytest.raises(
        CounterfactualError,
        match="non_fonit_allowed must be a boolean",
    ):
        make_intervention(
            non_fonit_allowed=1,
        )


def test_intervention_rejects_invalid_scope() -> None:
    with pytest.raises(
        CounterfactualError,
        match="unsupported intervention scope",
    ):
        make_intervention(
            scope="unsupported",
        )


def test_intervention_rejects_invalid_status() -> None:
    with pytest.raises(
        CounterfactualError,
        match="unsupported intervention status",
    ):
        make_intervention(
            status="unsupported",
        )


# ---------------------------------------------------------------------------
# CounterfactualScenario and weights
# ---------------------------------------------------------------------------


def test_scenario_creation() -> None:
    scenario = make_scenario()

    assert scenario.scenario_id == "scenario-a"
    assert scenario.validation_score == pytest.approx(0.90)
    assert scenario.model_penalty == pytest.approx(0.05)
    assert scenario.is_eligible is True


def test_scenario_is_immutable() -> None:
    scenario = make_scenario()

    with pytest.raises(FrozenInstanceError):
        scenario.validation_score = 0.1


def test_scenario_metadata_is_read_only() -> None:
    scenario = make_scenario()

    with pytest.raises(TypeError):
        scenario.metadata["new"] = "value"


def test_scenario_ineligible_when_intervention_vetoed() -> None:
    scenario = make_scenario(
        intervention=make_intervention(
            non_fonit_allowed=False,
        )
    )

    assert scenario.is_eligible is False


def test_scenario_ineligible_without_forecast_best() -> None:
    empty_forecast = HistoryForecastResult(
        current_signature_id="current",
        influence_set_id="planes",
        status=ForecastStatus.NO_CANDIDATES,
    )

    scenario = CounterfactualScenario(
        intervention=make_intervention(),
        forecast_result=empty_forecast,
    )

    assert scenario.is_eligible is False


def test_scenario_rejects_invalid_intervention() -> None:
    with pytest.raises(
        CounterfactualError,
        match="intervention must be an Intervention",
    ):
        CounterfactualScenario(
            intervention="intervention",
            forecast_result=make_forecast_result(
                "forecast",
                signature=make_signature(),
            ),
        )


def test_scenario_rejects_invalid_forecast() -> None:
    with pytest.raises(
        CounterfactualError,
        match="forecast_result must be",
    ):
        CounterfactualScenario(
            intervention=make_intervention(),
            forecast_result="forecast",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "validation_score",
        "model_penalty",
    ),
)
def test_scenario_rejects_invalid_unit_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        CounterfactualError,
        match=field_name,
    ):
        make_scenario(
            **{
                field_name: 1.01,
            }
        )


def test_default_counterfactual_weights() -> None:
    weights = CounterfactualWeights()

    assert weights.positive_weight == pytest.approx(3.25)
    assert weights.capacity_benefit == pytest.approx(1.0)
    assert weights.safety == pytest.approx(0.75)


def test_weights_are_immutable() -> None:
    weights = CounterfactualWeights()

    with pytest.raises(FrozenInstanceError):
        weights.safety = 1.0


def test_weights_to_dict() -> None:
    data = CounterfactualWeights().to_dict()

    assert data["capacity_benefit"] == pytest.approx(1.0)
    assert data["model_penalty"] == pytest.approx(0.30)


def test_weights_reject_negative_values() -> None:
    with pytest.raises(
        CounterfactualError,
        match="safety must be non-negative",
    ):
        CounterfactualWeights(
            safety=-0.1,
        )


def test_weights_require_positive_benefit_weight() -> None:
    with pytest.raises(
        CounterfactualError,
        match="at least one positive benefit weight",
    ):
        CounterfactualWeights(
            capacity_benefit=0.0,
            loss_reduction=0.0,
            gain_increase=0.0,
            forecast_quality=0.0,
            validation=0.0,
        )


# ---------------------------------------------------------------------------
# CounterfactualScore and result
# ---------------------------------------------------------------------------


def test_score_creation() -> None:
    score = make_score()

    assert score.intervention_id == "intervention-scenario-a"
    assert score.intervention_name == (
        "Intervention intervention-scenario-a"
    )
    assert score.target_id == "node:A"
    assert score.is_beneficial is True


def test_score_is_immutable() -> None:
    score = make_score()

    with pytest.raises(FrozenInstanceError):
        score.rank = 2


def test_score_with_rank() -> None:
    score = make_score(rank=0)
    ranked = score.with_rank(2)

    assert score.rank == 0
    assert ranked.rank == 2
    assert ranked.scenario == score.scenario


@pytest.mark.parametrize(
    "bad_rank",
    (
        0,
        -1,
        1.5,
    ),
)
def test_score_with_rank_rejects_invalid(
    bad_rank,
) -> None:
    score = make_score(rank=0)

    with pytest.raises(
        CounterfactualError,
        match="rank must be a positive integer",
    ):
        score.with_rank(bad_rank)


def test_score_to_dict() -> None:
    data = make_score().to_dict()

    assert data["rank"] == 1
    assert data["target_id"] == "node:A"
    assert data["is_beneficial"] is True


def test_result_creation_and_node_star() -> None:
    first = make_score(
        rank=1,
        normalized_score=0.90,
    )
    second = make_score(
        rank=2,
        normalized_score=0.70,
    )

    result = CounterfactualResult(
        result_id="result-001",
        baseline_forecast_id="baseline",
        status=CounterfactualStatus.COMPLETED,
        rankings=(first, second),
        confidence=0.80,
        ambiguity=0.80,
        selection_margin=0.20,
    )

    assert result.best == first
    assert result.alternatives == (second,)
    assert result.node_star == "node:A"
    assert result.is_ambiguous is False


def test_empty_result_has_no_node_star() -> None:
    result = CounterfactualResult(
        baseline_forecast_id="baseline",
        status=CounterfactualStatus.NO_SCENARIOS,
    )

    assert result.best is None
    assert result.node_star is None


def test_result_is_immutable() -> None:
    result = CounterfactualResult(
        baseline_forecast_id="baseline",
        status=CounterfactualStatus.NO_SCENARIOS,
    )

    with pytest.raises(FrozenInstanceError):
        result.status = CounterfactualStatus.COMPLETED


def test_result_metadata_is_read_only() -> None:
    result = CounterfactualResult(
        baseline_forecast_id="baseline",
        status=CounterfactualStatus.NO_SCENARIOS,
    )

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


def test_result_top() -> None:
    rankings = (
        make_score(rank=1),
        make_score(rank=2),
    )

    result = CounterfactualResult(
        baseline_forecast_id="baseline",
        status=CounterfactualStatus.COMPLETED,
        rankings=rankings,
    )

    assert result.top(1) == rankings[:1]
    assert result.top(0) == ()


def test_result_rejects_nonconsecutive_ranks() -> None:
    with pytest.raises(
        CounterfactualError,
        match="ranking positions must be consecutive",
    ):
        CounterfactualResult(
            baseline_forecast_id="baseline",
            status=CounterfactualStatus.COMPLETED,
            rankings=(
                make_score(rank=1),
                make_score(rank=3),
            ),
        )


def test_result_to_dict() -> None:
    result = CounterfactualResult(
        result_id="result-001",
        baseline_forecast_id="baseline",
        status=CounterfactualStatus.NO_SCENARIOS,
    )

    data = result.to_dict()

    assert data["version"] == "1.0.0"
    assert data["result_id"] == "result-001"
    assert data["node_star"] is None
    assert data["rankings"] == []


# ---------------------------------------------------------------------------
# CounterfactualEngine
# ---------------------------------------------------------------------------


def test_engine_creation() -> None:
    engine = CounterfactualEngine()

    assert engine.ambiguity_threshold == pytest.approx(0.05)
    assert engine.cost_scale == pytest.approx(1.0)
    assert engine.complexity_scale == pytest.approx(5.0)
    assert engine.minimum_score == pytest.approx(0.0)


def test_engine_is_immutable() -> None:
    engine = CounterfactualEngine()

    with pytest.raises(FrozenInstanceError):
        engine.minimum_score = 0.5


def test_engine_rejects_invalid_weights() -> None:
    with pytest.raises(
        CounterfactualError,
        match="weights must be CounterfactualWeights",
    ):
        CounterfactualEngine(
            weights="weights",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "cost_scale",
        "complexity_scale",
    ),
)
def test_engine_rejects_nonpositive_scales(
    field_name: str,
) -> None:
    with pytest.raises(
        CounterfactualError,
        match=field_name,
    ):
        CounterfactualEngine(
            **{
                field_name: 0.0,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "ambiguity_threshold",
        "minimum_score",
    ),
)
def test_engine_rejects_invalid_unit_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        CounterfactualError,
        match=field_name,
    ):
        CounterfactualEngine(
            **{
                field_name: 1.01,
            }
        )


def test_engine_rejects_nonboolean_require_benefit() -> None:
    with pytest.raises(
        CounterfactualError,
        match="require_benefit must be a boolean",
    ):
        CounterfactualEngine(
            require_benefit=1,
        )


def test_score_scenario_detects_capacity_improvement() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(
            total_capacity_loss=0.30,
            total_capacity_gain=0.05,
        ),
    )
    scenario = make_scenario(
        signature=make_signature(
            signature_id="improved",
            total_capacity_loss=0.15,
            total_capacity_gain=0.12,
        )
    )

    score = CounterfactualEngine().score_scenario(
        baseline,
        scenario,
    )

    assert score.net_capacity_delta > 0.0
    assert score.is_beneficial is True
    assert score.loss_reduction_score > 0.5
    assert score.gain_increase_score > 0.5


def test_higher_cost_reduces_score() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    low_cost = make_scenario(
        "low-cost",
        intervention=make_intervention(
            intervention_id="low-cost",
            cost=0.0,
        ),
    )
    high_cost = make_scenario(
        "high-cost",
        intervention=make_intervention(
            intervention_id="high-cost",
            cost=100.0,
        ),
    )

    engine = CounterfactualEngine()

    assert (
        engine.score_scenario(
            baseline,
            low_cost,
        ).normalized_score
        >
        engine.score_scenario(
            baseline,
            high_cost,
        ).normalized_score
    )


def test_higher_safety_risk_reduces_score() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    safe = make_scenario(
        "safe",
        intervention=make_intervention(
            intervention_id="safe",
            operational_risk=0.0,
            cascade_risk=0.0,
            reversibility=1.0,
        ),
    )
    risky = make_scenario(
        "risky",
        intervention=make_intervention(
            intervention_id="risky",
            operational_risk=1.0,
            cascade_risk=1.0,
            reversibility=0.0,
        ),
    )

    engine = CounterfactualEngine()

    assert (
        engine.score_scenario(
            baseline,
            safe,
        ).normalized_score
        >
        engine.score_scenario(
            baseline,
            risky,
        ).normalized_score
    )


def test_score_rejects_ineligible_scenario() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )
    scenario = make_scenario(
        intervention=make_intervention(
            non_fonit_allowed=False,
        )
    )

    with pytest.raises(
        CounterfactualError,
        match="ineligible intervention scenarios",
    ):
        CounterfactualEngine().score_scenario(
            baseline,
            scenario,
        )


def test_evaluate_no_scenarios() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    result = CounterfactualEngine().evaluate(
        baseline,
        (),
    )

    assert result.status is CounterfactualStatus.NO_SCENARIOS
    assert result.rankings == ()
    assert result.node_star is None


def test_evaluate_no_valid_scenarios() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )
    vetoed = make_scenario(
        intervention=make_intervention(
            status=InterventionStatus.VETOED,
        )
    )

    result = CounterfactualEngine().evaluate(
        baseline,
        (vetoed,),
    )

    assert (
        result.status
        is CounterfactualStatus.NO_VALID_SCENARIOS
    )


def test_evaluate_ranks_best_intervention_first() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(
            total_capacity_loss=0.30,
            total_capacity_gain=0.05,
        ),
    )

    strong = make_scenario(
        "strong",
        intervention=make_intervention(
            intervention_id="strong",
            target_id="node:STRONG",
            cost=0.05,
            operational_risk=0.02,
            cascade_risk=0.02,
        ),
        signature=make_signature(
            signature_id="strong-future",
            total_capacity_loss=0.10,
            total_capacity_gain=0.15,
        ),
    )

    weak = make_scenario(
        "weak",
        intervention=make_intervention(
            intervention_id="weak",
            target_id="node:WEAK",
            cost=0.50,
            operational_risk=0.20,
            cascade_risk=0.20,
        ),
        signature=make_signature(
            signature_id="weak-future",
            total_capacity_loss=0.25,
            total_capacity_gain=0.06,
        ),
    )

    result = CounterfactualEngine().evaluate(
        baseline,
        (weak, strong),
    )

    assert result.best is not None
    assert result.best.intervention_id == "strong"
    assert result.node_star == "node:STRONG"
    assert result.rankings[0].rank == 1
    assert result.rankings[1].rank == 2


def test_evaluate_excludes_non_fonit_veto() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    allowed = make_scenario(
        "allowed",
        intervention=make_intervention(
            intervention_id="allowed",
        ),
    )
    vetoed = make_scenario(
        "vetoed",
        intervention=make_intervention(
            intervention_id="vetoed",
            non_fonit_allowed=False,
        ),
    )

    result = CounterfactualEngine().evaluate(
        baseline,
        (vetoed, allowed),
    )

    assert len(result.rankings) == 1
    assert result.best is not None
    assert result.best.intervention_id == "allowed"
    assert result.metadata["vetoed_count"] == 1


def test_evaluate_require_benefit_filters_harmful_scenario() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(
            total_capacity_loss=0.20,
            total_capacity_gain=0.10,
        ),
    )

    harmful = make_scenario(
        "harmful",
        signature=make_signature(
            signature_id="harmful-future",
            total_capacity_loss=0.40,
            total_capacity_gain=0.02,
        ),
    )

    result = CounterfactualEngine(
        require_benefit=True,
    ).evaluate(
        baseline,
        (harmful,),
    )

    assert (
        result.status
        is CounterfactualStatus.NO_VALID_SCENARIOS
    )


def test_evaluate_identical_scenarios_is_ambiguous() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    first = make_scenario(
        "first",
        intervention=make_intervention(
            intervention_id="first",
        ),
    )
    second = make_scenario(
        "second",
        intervention=make_intervention(
            intervention_id="second",
        ),
    )

    result = CounterfactualEngine(
        ambiguity_threshold=0.05,
    ).evaluate(
        baseline,
        (first, second),
    )

    assert result.status is CounterfactualStatus.AMBIGUOUS
    assert result.is_ambiguous is True
    assert result.selection_margin == pytest.approx(0.0)


def test_evaluate_rejects_duplicate_scenario_ids() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    first = make_scenario("duplicate")
    second = make_scenario(
        "duplicate",
        intervention=make_intervention(
            intervention_id="other",
        ),
    )

    with pytest.raises(
        CounterfactualError,
        match="scenario_id values must be unique",
    ):
        CounterfactualEngine().evaluate(
            baseline,
            (first, second),
        )


def test_evaluate_rejects_invalid_scenario_object() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    with pytest.raises(
        CounterfactualError,
        match="all scenarios must be",
    ):
        CounterfactualEngine().evaluate(
            baseline,
            ("scenario",),
        )


def test_evaluate_metadata_is_merged() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    result = CounterfactualEngine().evaluate(
        baseline,
        (make_scenario(),),
        metadata={
            "experiment_id": "exp-001",
        },
    )

    assert result.metadata["engine_version"] == "1.0.0"
    assert result.metadata["scenario_count"] == 1
    assert result.metadata["experiment_id"] == "exp-001"


def test_softmax_probabilities_sum_to_one() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )

    result = CounterfactualEngine().evaluate(
        baseline,
        (
            make_scenario("a"),
            make_scenario(
                "b",
                intervention=make_intervention(
                    intervention_id="b",
                    cost=1.0,
                ),
            ),
        ),
    )

    probabilities = (
        CounterfactualEngine()
        .softmax_probabilities(result)
    )

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert set(probabilities) == {
        "intervention-a",
        "b",
    }


def test_softmax_probabilities_are_read_only() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )
    engine = CounterfactualEngine()
    result = engine.evaluate(
        baseline,
        (make_scenario(),),
    )

    probabilities = engine.softmax_probabilities(
        result
    )

    with pytest.raises(TypeError):
        probabilities["new"] = 0.5


def test_softmax_empty_result() -> None:
    result = CounterfactualResult(
        baseline_forecast_id="baseline",
        status=CounterfactualStatus.NO_SCENARIOS,
    )

    assert (
        CounterfactualEngine()
        .softmax_probabilities(result)
        == {}
    )


def test_softmax_rejects_invalid_temperature() -> None:
    baseline = make_forecast_result(
        "baseline",
        signature=make_signature(),
    )
    engine = CounterfactualEngine()
    result = engine.evaluate(
        baseline,
        (make_scenario(),),
    )

    with pytest.raises(
        CounterfactualError,
        match="temperature must be greater than zero",
    ):
        engine.softmax_probabilities(
            result,
            temperature=0.0,
        )


def test_engine_round_trip() -> None:
    engine = CounterfactualEngine(
        weights=CounterfactualWeights(
            capacity_benefit=2.0,
            loss_reduction=1.0,
            gain_increase=0.8,
            forecast_quality=0.7,
            validation=0.9,
            cost=0.4,
            safety=1.2,
            complexity=0.3,
            model_penalty=0.5,
        ),
        ambiguity_threshold=0.10,
        cost_scale=2.0,
        complexity_scale=8.0,
        minimum_score=0.20,
        require_benefit=True,
    )

    restored = CounterfactualEngine.from_dict(
        engine.to_dict()
    )

    assert restored == engine


def test_engine_from_dict_rejects_version() -> None:
    data = CounterfactualEngine().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        CounterfactualError,
        match="unsupported counterfactual engine version",
    ):
        CounterfactualEngine.from_dict(data)


def test_engine_from_dict_rejects_invalid_weights() -> None:
    data = CounterfactualEngine().to_dict()
    data["weights"] = "invalid"

    with pytest.raises(
        CounterfactualError,
        match="weights must be a mapping",
    ):
        CounterfactualEngine.from_dict(data)
