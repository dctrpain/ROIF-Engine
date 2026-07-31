"""Tests for ROIF counterfactual intervention optimization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from roif.optimization import (
    InterventionOptimizer,
    InterventionScore,
    InterventionSimulation,
    InterventionSimulationStatus,
    OptimizationConfig,
    OptimizationError,
    OptimizationResult,
    OptimizationStatus,
    OptimizationVetoReason,
    ReserveAggregation,
    aggregate_reserve,
    optimize_interventions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_simulation(
    candidate_id: Any = "A",
    *,
    reserve_after: tuple[float, ...] = (0.6, 0.7),
    cost: float = 1.0,
    risk: float = 0.1,
    delay: float = 1.0,
    evidence_strength: float = 0.8,
    reversibility: float = 0.9,
    cascade_risk: float = 0.1,
    external_system_risk: float = 0.0,
    converged: bool = True,
    iterations: int | None = 5,
) -> InterventionSimulation:
    return InterventionSimulation(
        candidate_id=candidate_id,
        reserve_after=np.asarray(reserve_after, dtype=float),
        cost=cost,
        risk=risk,
        delay=delay,
        evidence_strength=evidence_strength,
        reversibility=reversibility,
        cascade_risk=cascade_risk,
        external_system_risk=external_system_risk,
        converged=converged,
        iterations=iterations,
        metadata={"source": "test"},
    )


def make_score(
    *,
    candidate_id: Any = "A",
    rank: int = 1,
    status: InterventionSimulationStatus = (
        InterventionSimulationStatus.SUCCEEDED
    ),
    global_reserve_before: float = 1.0,
    global_reserve_after: float | None = 1.3,
    recovery_gain: float | None = 0.3,
    utility: float | None = 0.3,
    tied_for_best: bool = False,
    converged: bool | None = True,
    cost: float | None = 1.0,
    risk: float | None = 0.1,
    delay: float | None = 1.0,
    evidence_strength: float | None = 0.8,
    reversibility: float | None = 0.9,
    veto_reasons: tuple[OptimizationVetoReason, ...] = (),
    error_type: str | None = None,
    error_message: str | None = None,
) -> InterventionScore:
    return InterventionScore(
        candidate_id=candidate_id,
        rank=rank,
        status=status,
        global_reserve_before=global_reserve_before,
        global_reserve_after=global_reserve_after,
        recovery_gain=recovery_gain,
        utility=utility,
        tied_for_best=tied_for_best,
        converged=converged,
        cost=cost,
        risk=risk,
        delay=delay,
        evidence_strength=evidence_strength,
        reversibility=reversibility,
        veto_reasons=veto_reasons,
        error_type=error_type,
        error_message=error_message,
        metadata={"source": "test"},
    )


def make_result(
    *,
    status: OptimizationStatus = OptimizationStatus.OPTIMIZED,
    node_star: Any = "A",
    best_utility: float | None = 0.3,
    best_recovery_gain: float | None = 0.3,
    tied_node_ids: tuple[Any, ...] = ("A",),
    scores: tuple[InterventionScore, ...] | None = None,
    successful_candidate_count: int = 1,
    failed_candidate_count: int = 0,
    vetoed_candidate_count: int = 0,
) -> OptimizationResult:
    if scores is None:
        scores = (make_score(),)

    return OptimizationResult(
        status=status,
        node_star=node_star,
        best_utility=best_utility,
        best_recovery_gain=best_recovery_gain,
        tied_node_ids=tied_node_ids,
        scores=scores,
        aggregation=ReserveAggregation.SUM,
        global_reserve_before=1.0,
        successful_candidate_count=successful_candidate_count,
        failed_candidate_count=failed_candidate_count,
        vetoed_candidate_count=vetoed_candidate_count,
        summary=" Optimization summary. ",
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


def test_reserve_aggregation_values_are_stable() -> None:
    assert tuple(item.value for item in ReserveAggregation) == (
        "sum",
        "mean",
        "minimum",
        "weighted_sum",
        "custom",
    )


def test_optimization_status_values_are_stable() -> None:
    assert tuple(item.value for item in OptimizationStatus) == (
        "optimized",
        "tied",
        "partial",
        "failed",
        "all_vetoed",
    )


def test_simulation_status_values_are_stable() -> None:
    assert tuple(
        item.value for item in InterventionSimulationStatus
    ) == (
        "succeeded",
        "failed",
        "vetoed",
    )


def test_veto_reason_values_are_stable() -> None:
    assert tuple(item.value for item in OptimizationVetoReason) == (
        "cascade_risk",
        "external_system_risk",
        "irreversible_high_risk",
        "low_evidence",
    )


# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    config = OptimizationConfig()

    assert config.aggregation is ReserveAggregation.SUM
    assert config.weights is None
    assert config.gain_weight == pytest.approx(1.0)
    assert config.weight_sum == pytest.approx(1.0)
    assert config.top_k == 5
    assert config.non_fonit_gate_enabled is True


def test_config_normalizes_numbers() -> None:
    config = OptimizationConfig(
        gain_weight="2",
        tie_tolerance="0.1",
        minimum_recovery_gain="-0.2",
        maximum_cascade_risk="0.5",
    )

    assert config.gain_weight == pytest.approx(2.0)
    assert config.tie_tolerance == pytest.approx(0.1)
    assert config.minimum_recovery_gain == pytest.approx(-0.2)
    assert config.maximum_cascade_risk == pytest.approx(0.5)


def test_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        OptimizationConfig().top_k = 2  # type: ignore[misc]


@pytest.mark.parametrize("value", ("sum", None, 1))
def test_config_rejects_invalid_aggregation(value: Any) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(aggregation=value)


@pytest.mark.parametrize(
    "field",
    (
        "gain_weight",
        "cost_weight",
        "risk_weight",
        "delay_weight",
        "evidence_weight",
        "reversibility_weight",
    ),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), float("inf"), True, "bad"),
)
def test_config_rejects_invalid_weights(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(**{field: value})


def test_config_rejects_all_zero_weights() -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(
            gain_weight=0.0,
            cost_weight=0.0,
            risk_weight=0.0,
            delay_weight=0.0,
            evidence_weight=0.0,
            reversibility_weight=0.0,
        )


@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), float("inf"), True, "bad"),
)
def test_config_rejects_invalid_tie_tolerance(value: Any) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(tie_tolerance=value)


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "5"))
def test_config_rejects_invalid_top_k(value: Any) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(top_k=value)


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_config_rejects_invalid_minimum_recovery_gain(
    value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(minimum_recovery_gain=value)


@pytest.mark.parametrize(
    "field",
    (
        "allow_partial_results",
        "capture_simulation_errors",
        "non_fonit_gate_enabled",
    ),
)
@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_config_requires_boolean_flags(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "maximum_cascade_risk",
        "maximum_external_system_risk",
        "irreversible_risk_threshold",
        "minimum_evidence_strength",
    ),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, 1.1, float("nan"), True, "bad"),
)
def test_config_rejects_invalid_unit_interval_values(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(**{field: value})


def test_weighted_sum_requires_weights() -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(
            aggregation=ReserveAggregation.WEIGHTED_SUM
        )


def test_weights_only_allowed_for_weighted_sum() -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(weights=(1.0, 1.0))


def test_weighted_sum_accepts_weights() -> None:
    config = OptimizationConfig(
        aggregation=ReserveAggregation.WEIGHTED_SUM,
        weights=(1.0, 2.0),
    )

    assert config.weights == (1.0, 2.0)


@pytest.mark.parametrize(
    "weights",
    (
        (),
        (-1.0, 1.0),
        (0.0, 0.0),
        (float("nan"), 1.0),
        "bad",
    ),
)
def test_config_rejects_invalid_reserve_weights(
    weights: Any,
) -> None:
    with pytest.raises(OptimizationError):
        OptimizationConfig(
            aggregation=ReserveAggregation.WEIGHTED_SUM,
            weights=weights,
        )


# ---------------------------------------------------------------------------
# InterventionSimulation
# ---------------------------------------------------------------------------


def test_simulation_properties() -> None:
    simulation = make_simulation(candidate_id=" A ")

    assert simulation.candidate_id == "A"
    assert np.allclose(simulation.reserve_after, (0.6, 0.7))
    assert simulation.reserve_after.flags.writeable is False
    assert simulation.cost == pytest.approx(1.0)
    assert isinstance(simulation.metadata, MappingProxyType)


def test_simulation_as_dict() -> None:
    payload = make_simulation().as_dict()

    assert payload["candidate_id"] == "A"
    assert payload["reserve_after"] == [0.6, 0.7]
    assert payload["evidence_strength"] == pytest.approx(0.8)


def test_simulation_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_simulation().cost = 5.0  # type: ignore[misc]


def test_simulation_reserve_is_read_only() -> None:
    simulation = make_simulation()

    with pytest.raises(ValueError):
        simulation.reserve_after[0] = 99.0


@pytest.mark.parametrize("candidate_id", (None, "", "   ", []))
def test_simulation_rejects_invalid_candidate_id(
    candidate_id: Any,
) -> None:
    with pytest.raises(OptimizationError):
        make_simulation(candidate_id=candidate_id)


@pytest.mark.parametrize(
    "reserve_after",
    (
        (),
        ((1.0, 2.0),),
        (1.0, float("nan")),
        "bad",
        True,
    ),
)
def test_simulation_rejects_invalid_reserve_after(
    reserve_after: Any,
) -> None:
    with pytest.raises(OptimizationError):
        InterventionSimulation("A", reserve_after)


@pytest.mark.parametrize("field", ("cost", "delay"))
@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), True, "bad"),
)
def test_simulation_rejects_invalid_nonnegative_values(
    field: str,
    value: Any,
) -> None:
    values = {
        "candidate_id": "A",
        "reserve_after": (1.0,),
        field: value,
    }
    with pytest.raises(OptimizationError):
        InterventionSimulation(**values)


@pytest.mark.parametrize(
    "field",
    (
        "risk",
        "evidence_strength",
        "reversibility",
        "cascade_risk",
        "external_system_risk",
    ),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, 1.1, float("nan"), True, "bad"),
)
def test_simulation_rejects_invalid_unit_values(
    field: str,
    value: Any,
) -> None:
    values = {
        "candidate_id": "A",
        "reserve_after": (1.0,),
        field: value,
    }
    with pytest.raises(OptimizationError):
        InterventionSimulation(**values)


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_simulation_rejects_invalid_converged(value: Any) -> None:
    with pytest.raises(OptimizationError):
        InterventionSimulation(
            "A",
            (1.0,),
            converged=value,
        )


@pytest.mark.parametrize("value", (-1, True, 1.5, "1"))
def test_simulation_rejects_invalid_iterations(value: Any) -> None:
    with pytest.raises(OptimizationError):
        InterventionSimulation(
            "A",
            (1.0,),
            iterations=value,
        )


# ---------------------------------------------------------------------------
# InterventionScore
# ---------------------------------------------------------------------------


def test_successful_score_properties() -> None:
    score = make_score()

    assert score.candidate_id == "A"
    assert score.status is InterventionSimulationStatus.SUCCEEDED
    assert score.recovery_gain == pytest.approx(0.3)
    assert isinstance(score.metadata, MappingProxyType)


def test_successful_score_as_dict() -> None:
    payload = make_score().as_dict()

    assert payload["candidate_id"] == "A"
    assert payload["status"] == "succeeded"
    assert payload["utility"] == pytest.approx(0.3)


def test_vetoed_score_is_valid() -> None:
    score = make_score(
        status=InterventionSimulationStatus.VETOED,
        utility=None,
        veto_reasons=(OptimizationVetoReason.CASCADE_RISK,),
    )

    assert score.status is InterventionSimulationStatus.VETOED
    assert score.veto_reasons == (
        OptimizationVetoReason.CASCADE_RISK,
    )


def test_failed_score_is_valid() -> None:
    score = make_score(
        status=InterventionSimulationStatus.FAILED,
        global_reserve_after=None,
        recovery_gain=None,
        utility=None,
        converged=None,
        cost=None,
        risk=None,
        delay=None,
        evidence_strength=None,
        reversibility=None,
        error_type="RuntimeError",
        error_message="boom",
    )

    assert score.status is InterventionSimulationStatus.FAILED
    assert score.error_type == "RuntimeError"


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "1"))
def test_score_rejects_invalid_rank(value: Any) -> None:
    with pytest.raises(OptimizationError):
        make_score(rank=value)


@pytest.mark.parametrize("value", ("succeeded", None, 1))
def test_score_rejects_invalid_status(value: Any) -> None:
    with pytest.raises(OptimizationError):
        make_score(status=value)


@pytest.mark.parametrize(
    "field",
    (
        "global_reserve_before",
        "global_reserve_after",
        "recovery_gain",
        "utility",
    ),
)
@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_score_rejects_invalid_general_numbers(
    field: str,
    value: Any,
) -> None:
    values = {
        "candidate_id": "A",
        "rank": 1,
        "status": InterventionSimulationStatus.SUCCEEDED,
        "global_reserve_before": 1.0,
        "global_reserve_after": 1.3,
        "recovery_gain": 0.3,
        "utility": 0.3,
        "tied_for_best": False,
        "converged": True,
        "cost": 1.0,
        "risk": 0.1,
        "delay": 1.0,
        "evidence_strength": 0.8,
        "reversibility": 0.9,
    }
    values[field] = value
    with pytest.raises(OptimizationError):
        InterventionScore(**values)


@pytest.mark.parametrize("field", ("cost", "delay"))
@pytest.mark.parametrize("value", (-0.1, float("nan"), True, "bad"))
def test_score_rejects_invalid_nonnegative_numbers(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        make_score(**{field: value})


@pytest.mark.parametrize(
    "field",
    ("risk", "evidence_strength", "reversibility"),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, 1.1, float("nan"), True, "bad"),
)
def test_score_rejects_invalid_unit_numbers(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        make_score(**{field: value})


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_score_rejects_invalid_tied_flag(value: Any) -> None:
    with pytest.raises(OptimizationError):
        make_score(tied_for_best=value)


def test_successful_score_requires_all_numeric_fields() -> None:
    with pytest.raises(OptimizationError):
        make_score(cost=None)


def test_successful_score_disallows_veto_reasons() -> None:
    with pytest.raises(OptimizationError):
        make_score(
            veto_reasons=(OptimizationVetoReason.CASCADE_RISK,)
        )


def test_successful_score_disallows_errors() -> None:
    with pytest.raises(OptimizationError):
        make_score(
            error_type="Unexpected",
            error_message="Unexpected",
        )


def test_vetoed_score_requires_reasons() -> None:
    with pytest.raises(OptimizationError):
        make_score(
            status=InterventionSimulationStatus.VETOED,
            utility=None,
        )


def test_vetoed_score_disallows_utility() -> None:
    with pytest.raises(OptimizationError):
        make_score(
            status=InterventionSimulationStatus.VETOED,
            veto_reasons=(OptimizationVetoReason.CASCADE_RISK,),
        )


def test_failed_score_requires_errors() -> None:
    with pytest.raises(OptimizationError):
        make_score(
            status=InterventionSimulationStatus.FAILED,
            global_reserve_after=None,
            recovery_gain=None,
            utility=None,
            converged=None,
            cost=None,
            risk=None,
            delay=None,
            evidence_strength=None,
            reversibility=None,
        )


def test_failed_score_disallows_veto_reasons() -> None:
    with pytest.raises(OptimizationError):
        make_score(
            status=InterventionSimulationStatus.FAILED,
            global_reserve_after=None,
            recovery_gain=None,
            utility=None,
            converged=None,
            cost=None,
            risk=None,
            delay=None,
            evidence_strength=None,
            reversibility=None,
            veto_reasons=(OptimizationVetoReason.CASCADE_RISK,),
            error_type="RuntimeError",
            error_message="boom",
        )


def test_score_rejects_duplicate_veto_reasons() -> None:
    with pytest.raises(OptimizationError):
        make_score(
            status=InterventionSimulationStatus.VETOED,
            utility=None,
            veto_reasons=(
                OptimizationVetoReason.CASCADE_RISK,
                OptimizationVetoReason.CASCADE_RISK,
            ),
        )


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


def test_result_properties() -> None:
    result = make_result()

    assert result.node_star == "A"
    assert result.best_utility == pytest.approx(0.3)
    assert result.is_tied is False
    assert result.summary == "Optimization summary."
    assert isinstance(result.metadata, MappingProxyType)


def test_result_as_dict() -> None:
    payload = make_result().as_dict()

    assert payload["status"] == "optimized"
    assert payload["node_star"] == "A"
    assert payload["aggregation"] == "sum"


def test_failed_result_is_valid_without_node_star() -> None:
    failed_score = make_score(
        status=InterventionSimulationStatus.FAILED,
        global_reserve_after=None,
        recovery_gain=None,
        utility=None,
        converged=None,
        cost=None,
        risk=None,
        delay=None,
        evidence_strength=None,
        reversibility=None,
        error_type="RuntimeError",
        error_message="boom",
    )
    result = make_result(
        status=OptimizationStatus.FAILED,
        node_star=None,
        best_utility=None,
        best_recovery_gain=None,
        tied_node_ids=(),
        scores=(failed_score,),
        successful_candidate_count=0,
        failed_candidate_count=1,
    )

    assert result.node_star is None


def test_all_vetoed_result_is_valid_without_node_star() -> None:
    vetoed_score = make_score(
        status=InterventionSimulationStatus.VETOED,
        utility=None,
        veto_reasons=(OptimizationVetoReason.CASCADE_RISK,),
    )
    result = make_result(
        status=OptimizationStatus.ALL_VETOED,
        node_star=None,
        best_utility=None,
        best_recovery_gain=None,
        tied_node_ids=(),
        scores=(vetoed_score,),
        successful_candidate_count=0,
        vetoed_candidate_count=1,
    )

    assert result.status is OptimizationStatus.ALL_VETOED


def test_successful_result_requires_node_star() -> None:
    with pytest.raises(OptimizationError):
        make_result(node_star=None)


def test_failed_result_disallows_node_star() -> None:
    with pytest.raises(OptimizationError):
        make_result(status=OptimizationStatus.FAILED)


def test_result_rejects_duplicate_tied_ids() -> None:
    with pytest.raises(OptimizationError):
        make_result(tied_node_ids=("A", "A"))


def test_result_rejects_count_mismatch() -> None:
    with pytest.raises(OptimizationError):
        make_result(successful_candidate_count=2)


@pytest.mark.parametrize("value", ("optimized", None, 1))
def test_result_rejects_invalid_status(value: Any) -> None:
    with pytest.raises(OptimizationError):
        make_result(status=value)


# ---------------------------------------------------------------------------
# aggregate_reserve
# ---------------------------------------------------------------------------


def test_aggregate_sum() -> None:
    assert aggregate_reserve((1.0, 2.0, 3.0)) == pytest.approx(6.0)


def test_aggregate_mean() -> None:
    value = aggregate_reserve(
        (1.0, 2.0, 3.0),
        config=OptimizationConfig(
            aggregation=ReserveAggregation.MEAN
        ),
    )

    assert value == pytest.approx(2.0)


def test_aggregate_minimum() -> None:
    value = aggregate_reserve(
        (1.0, -2.0, 3.0),
        config=OptimizationConfig(
            aggregation=ReserveAggregation.MINIMUM
        ),
    )

    assert value == pytest.approx(-2.0)


def test_aggregate_weighted_sum() -> None:
    value = aggregate_reserve(
        (1.0, 2.0),
        config=OptimizationConfig(
            aggregation=ReserveAggregation.WEIGHTED_SUM,
            weights=(2.0, 3.0),
        ),
    )

    assert value == pytest.approx(8.0)


def test_aggregate_custom() -> None:
    value = aggregate_reserve(
        (1.0, 2.0, 3.0),
        config=OptimizationConfig(
            aggregation=ReserveAggregation.CUSTOM
        ),
        aggregator=lambda reserve: float(np.max(reserve)),
    )

    assert value == pytest.approx(3.0)


def test_aggregate_accepts_scalar() -> None:
    assert aggregate_reserve(2.5) == pytest.approx(2.5)


def test_aggregate_rejects_invalid_config() -> None:
    with pytest.raises(OptimizationError):
        aggregate_reserve((1.0,), config={})


def test_custom_aggregation_requires_aggregator() -> None:
    with pytest.raises(OptimizationError):
        aggregate_reserve(
            (1.0,),
            config=OptimizationConfig(
                aggregation=ReserveAggregation.CUSTOM
            ),
        )


def test_builtin_aggregation_disallows_aggregator() -> None:
    with pytest.raises(OptimizationError):
        aggregate_reserve(
            (1.0,),
            aggregator=lambda reserve: 0.0,
        )


def test_weighted_sum_rejects_dimension_mismatch() -> None:
    with pytest.raises(OptimizationError):
        aggregate_reserve(
            (1.0,),
            config=OptimizationConfig(
                aggregation=ReserveAggregation.WEIGHTED_SUM,
                weights=(1.0, 1.0),
            ),
        )


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_custom_aggregator_rejects_invalid_result(
    value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        aggregate_reserve(
            (1.0,),
            config=OptimizationConfig(
                aggregation=ReserveAggregation.CUSTOM
            ),
            aggregator=lambda reserve: value,
        )


# ---------------------------------------------------------------------------
# InterventionOptimizer
# ---------------------------------------------------------------------------


def test_optimizer_defaults() -> None:
    optimizer = InterventionOptimizer()

    assert isinstance(optimizer.config, OptimizationConfig)


def test_optimizer_accepts_config() -> None:
    config = OptimizationConfig(
        aggregation=ReserveAggregation.MEAN
    )
    optimizer = InterventionOptimizer(config)

    assert optimizer.config is config


@pytest.mark.parametrize("config", ({}, "bad", 1))
def test_optimizer_rejects_invalid_config(config: Any) -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer(config)


def test_custom_optimizer_requires_aggregator() -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer(
            OptimizationConfig(
                aggregation=ReserveAggregation.CUSTOM
            )
        )


def test_builtin_optimizer_disallows_aggregator() -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer(
            aggregator=lambda reserve: 0.0
        )


@pytest.mark.parametrize("utility", ("bad", 1))
def test_optimizer_rejects_invalid_utility(utility: Any) -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer(utility=utility)


@pytest.mark.parametrize("simulate", (None, "bad", 1))
def test_optimize_rejects_invalid_simulator(simulate: Any) -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer().optimize(
            (0.5,),
            ("A",),
            simulate,
        )


@pytest.mark.parametrize(
    "candidates",
    (
        (),
        "A",
        ("A", "A"),
        (None,),
        ([],),
    ),
)
def test_optimize_rejects_invalid_candidates(
    candidates: Any,
) -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer().optimize(
            (0.5,),
            candidates,
            lambda _: (0.6,),
        )


@pytest.mark.parametrize(
    "reserve_before",
    (
        (),
        ((1.0, 2.0),),
        (1.0, float("nan")),
        "bad",
        True,
    ),
)
def test_optimize_rejects_invalid_reserve_before(
    reserve_before: Any,
) -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer().optimize(
            reserve_before,
            ("A",),
            lambda _: (1.0,),
        )


def test_optimize_selects_maximum_recovery_gain() -> None:
    reserves = {
        "A": (0.6, 0.6),
        "B": (0.8, 0.9),
        "C": (0.5, 0.7),
    }

    result = InterventionOptimizer().optimize(
        (0.5, 0.5),
        ("A", "B", "C"),
        lambda node_id: reserves[node_id],
    )

    assert result.status is OptimizationStatus.OPTIMIZED
    assert result.node_star == "B"
    assert result.best_recovery_gain == pytest.approx(0.7)
    assert result.successful_candidate_count == 3


def test_optimize_accepts_simulation_objects() -> None:
    result = InterventionOptimizer().optimize(
        (0.5, 0.5),
        ("A", "B"),
        lambda node_id: InterventionSimulation(
            node_id,
            (0.9, 0.9) if node_id == "A" else (0.6, 0.6),
            iterations=3,
        ),
    )

    assert result.node_star == "A"
    assert result.scores[0].metadata["iterations"] == 3


def test_optimize_accepts_mapping_results() -> None:
    result = InterventionOptimizer().optimize(
        (0.5, 0.5),
        ("A",),
        lambda _: {
            "reserve_after": (0.8, 0.8),
            "cost": 2.0,
            "risk": 0.2,
            "evidence_strength": 0.9,
        },
    )

    assert result.node_star == "A"
    assert result.scores[0].cost == pytest.approx(2.0)


def test_optimize_rejects_mapping_without_reserve() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: {"cost": 1.0},
    )

    assert result.status is OptimizationStatus.FAILED
    assert result.failed_candidate_count == 1


def test_optimize_rejects_mapping_unknown_fields() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: {
            "reserve_after": (0.6,),
            "unknown": 1,
        },
    )

    assert result.status is OptimizationStatus.FAILED


def test_optimize_rejects_wrong_simulation_candidate_id() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: InterventionSimulation("wrong", (0.6,)),
    )

    assert result.status is OptimizationStatus.FAILED


def test_optimize_rejects_wrong_reserve_dimension() -> None:
    result = InterventionOptimizer().optimize(
        (0.5, 0.5),
        ("A",),
        lambda _: (0.6,),
    )

    assert result.status is OptimizationStatus.FAILED


def test_optimize_detects_tie() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(tie_tolerance=1e-6)
    ).optimize(
        (0.5,),
        ("A", "B"),
        lambda _: (0.8,),
    )

    assert result.status is OptimizationStatus.TIED
    assert result.tied_node_ids == ("A", "B")
    assert result.node_star == "A"
    assert all(score.tied_for_best for score in result.scores)


def test_optimize_ranking_is_deterministic() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("B", "A"),
        lambda _: (0.8,),
    )

    assert tuple(score.candidate_id for score in result.scores) == (
        "A",
        "B",
    )


def test_optimize_marks_partial_when_candidate_fails() -> None:
    def simulate(node_id: str):
        if node_id == "B":
            raise RuntimeError("boom")
        return (0.8,) if node_id == "A" else (0.7,)

    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A", "B", "C"),
        simulate,
    )

    assert result.status is OptimizationStatus.PARTIAL
    assert result.node_star == "A"
    assert result.failed_candidate_count == 1
    assert any(
        score.status is InterventionSimulationStatus.FAILED
        for score in result.scores
    )


def test_optimize_rejects_partial_when_disabled() -> None:
    def simulate(node_id: str):
        if node_id == "B":
            raise RuntimeError("boom")
        return (0.8,)

    result = InterventionOptimizer(
        OptimizationConfig(allow_partial_results=False)
    ).optimize(
        (0.5,),
        ("A", "B"),
        simulate,
    )

    assert result.status is OptimizationStatus.FAILED
    assert result.node_star is None


def test_optimize_can_propagate_simulation_error() -> None:
    optimizer = InterventionOptimizer(
        OptimizationConfig(capture_simulation_errors=False)
    )

    with pytest.raises(RuntimeError, match="boom"):
        optimizer.optimize(
            (0.5,),
            ("A",),
            lambda _: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )


def test_optimize_all_failures() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A", "B"),
        lambda _: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    assert result.status is OptimizationStatus.FAILED
    assert result.node_star is None
    assert result.failed_candidate_count == 2


def test_non_fonit_gate_vetoes_cascade_risk() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: make_simulation(
            "A",
            reserve_after=(0.9,),
            cascade_risk=0.9,
        ),
    )

    assert result.status is OptimizationStatus.ALL_VETOED
    assert result.vetoed_candidate_count == 1
    assert result.scores[0].veto_reasons == (
        OptimizationVetoReason.CASCADE_RISK,
    )


def test_non_fonit_gate_vetoes_external_system_risk() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: make_simulation(
            "A",
            reserve_after=(0.9,),
            external_system_risk=0.9,
        ),
    )

    assert OptimizationVetoReason.EXTERNAL_SYSTEM_RISK in (
        result.scores[0].veto_reasons
    )


def test_non_fonit_gate_vetoes_irreversible_high_risk() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: make_simulation(
            "A",
            reserve_after=(0.9,),
            risk=0.9,
            reversibility=0.1,
        ),
    )

    assert OptimizationVetoReason.IRREVERSIBLE_HIGH_RISK in (
        result.scores[0].veto_reasons
    )


def test_non_fonit_gate_vetoes_low_evidence() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(minimum_evidence_strength=0.9)
    ).optimize(
        (0.5,),
        ("A",),
        lambda _: make_simulation(
            "A",
            reserve_after=(0.9,),
            evidence_strength=0.2,
        ),
    )

    assert OptimizationVetoReason.LOW_EVIDENCE in (
        result.scores[0].veto_reasons
    )


def test_non_fonit_gate_can_be_disabled() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(non_fonit_gate_enabled=False)
    ).optimize(
        (0.5,),
        ("A",),
        lambda _: make_simulation(
            "A",
            reserve_after=(0.9,),
            cascade_risk=1.0,
            external_system_risk=1.0,
            risk=1.0,
            reversibility=0.0,
        ),
    )

    assert result.status is OptimizationStatus.OPTIMIZED
    assert result.node_star == "A"


def test_mixed_success_and_veto_is_partial() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A", "B"),
        lambda node_id: (
            make_simulation(
                "A",
                reserve_after=(0.8,),
            )
            if node_id == "A"
            else make_simulation(
                "B",
                reserve_after=(0.9,),
                cascade_risk=0.9,
            )
        ),
    )

    assert result.status is OptimizationStatus.PARTIAL
    assert result.node_star == "A"
    assert result.vetoed_candidate_count == 1


def test_minimum_recovery_gain_can_reject_all_successes() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(minimum_recovery_gain=0.5)
    ).optimize(
        (0.5,),
        ("A", "B"),
        lambda node_id: (0.7,) if node_id == "A" else (0.6,),
    )

    assert result.status is OptimizationStatus.FAILED
    assert result.node_star is None


def test_mean_aggregation_changes_gain_scale() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(
            aggregation=ReserveAggregation.MEAN
        )
    ).optimize(
        (0.4, 0.6),
        ("A",),
        lambda _: (0.8, 0.8),
    )

    assert result.global_reserve_before == pytest.approx(0.5)
    assert result.best_recovery_gain == pytest.approx(0.3)


def test_weighted_sum_can_change_winner() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(
            aggregation=ReserveAggregation.WEIGHTED_SUM,
            weights=(10.0, 1.0),
        )
    ).optimize(
        (0.0, 0.0),
        ("A", "B"),
        lambda node_id: (
            (0.2, 1.0)
            if node_id == "A"
            else (0.5, 0.0)
        ),
    )

    assert result.node_star == "B"


def test_custom_aggregation() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(
            aggregation=ReserveAggregation.CUSTOM
        ),
        aggregator=lambda reserve: float(np.min(reserve)),
    ).optimize(
        (0.2, 0.5),
        ("A", "B"),
        lambda node_id: (
            (0.4, 0.4)
            if node_id == "A"
            else (0.3, 0.8)
        ),
    )

    assert result.node_star == "A"


def test_cost_penalty_can_change_winner() -> None:
    config = OptimizationConfig(
        gain_weight=1.0,
        cost_weight=1.0,
    )
    result = InterventionOptimizer(config).optimize(
        (0.5,),
        ("A", "B"),
        lambda node_id: (
            make_simulation(
                "A",
                reserve_after=(0.9,),
                cost=100.0,
            )
            if node_id == "A"
            else make_simulation(
                "B",
                reserve_after=(0.85,),
                cost=1.0,
            )
        ),
    )

    assert result.node_star == "B"


def test_risk_penalty_can_change_winner() -> None:
    config = OptimizationConfig(
        gain_weight=1.0,
        risk_weight=1.0,
    )
    result = InterventionOptimizer(config).optimize(
        (0.5,),
        ("A", "B"),
        lambda node_id: (
            make_simulation(
                "A",
                reserve_after=(0.9,),
                risk=0.7,
            )
            if node_id == "A"
            else make_simulation(
                "B",
                reserve_after=(0.85,),
                risk=0.0,
            )
        ),
    )

    assert result.node_star == "B"


def test_evidence_bonus_can_change_winner() -> None:
    config = OptimizationConfig(
        gain_weight=1.0,
        evidence_weight=1.0,
    )
    result = InterventionOptimizer(config).optimize(
        (0.5,),
        ("A", "B"),
        lambda node_id: (
            make_simulation(
                "A",
                reserve_after=(0.85,),
                evidence_strength=0.0,
            )
            if node_id == "A"
            else make_simulation(
                "B",
                reserve_after=(0.80,),
                evidence_strength=1.0,
            )
        ),
    )

    assert result.node_star == "B"


def test_custom_utility_can_change_winner() -> None:
    optimizer = InterventionOptimizer(
        utility=lambda simulation, gain, config: (
            -simulation.cost
        )
    )
    result = optimizer.optimize(
        (0.5,),
        ("A", "B"),
        lambda node_id: (
            make_simulation(
                "A",
                reserve_after=(0.9,),
                cost=10.0,
            )
            if node_id == "A"
            else make_simulation(
                "B",
                reserve_after=(0.7,),
                cost=1.0,
            )
        ),
    )

    assert result.node_star == "B"


@pytest.mark.parametrize(
    "utility_value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_custom_utility_rejects_invalid_result(
    utility_value: Any,
) -> None:
    with pytest.raises(OptimizationError):
        InterventionOptimizer(
            utility=lambda simulation, gain, config: utility_value
        ).optimize(
            (0.5,),
            ("A",),
            lambda _: (0.8,),
        )


def test_optimizer_metadata_is_forwarded() -> None:
    metadata = {"experiment": "E1"}
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: (0.8,),
        metadata=metadata,
    )
    metadata["experiment"] = "changed"

    assert result.metadata["experiment"] == "E1"


def test_summary_contains_node_star() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A",),
        lambda _: (0.8,),
    )

    assert "Node*: A." in result.summary
    assert "recovery gain" in result.summary


def test_tied_summary_contains_candidates() -> None:
    result = InterventionOptimizer().optimize(
        (0.5,),
        ("A", "B"),
        lambda _: (0.8,),
    )

    assert "Node* is tied" in result.summary
    assert "A, B" in result.summary


def test_top_k_configuration_does_not_discard_full_scores() -> None:
    result = InterventionOptimizer(
        OptimizationConfig(top_k=2)
    ).optimize(
        (0.5,),
        ("A", "B", "C"),
        lambda node_id: {
            "A": (0.9,),
            "B": (0.8,),
            "C": (0.7,),
        }[node_id],
    )

    assert len(result.scores) == 3
    assert tuple(
        score.candidate_id
        for score in result.scores[:2]
    ) == ("A", "B")


def test_optimize_interventions_wrapper() -> None:
    result = optimize_interventions(
        (0.5,),
        ("A", "B"),
        lambda node_id: (
            (0.9,)
            if node_id == "A"
            else (0.7,)
        ),
        metadata={"wrapper": True},
    )

    assert isinstance(result, OptimizationResult)
    assert result.node_star == "A"
    assert result.metadata["wrapper"] is True


def test_wrapper_forwards_custom_aggregation() -> None:
    result = optimize_interventions(
        (0.2, 0.5),
        ("A", "B"),
        lambda node_id: (
            (0.4, 0.4)
            if node_id == "A"
            else (0.3, 0.8)
        ),
        config=OptimizationConfig(
            aggregation=ReserveAggregation.CUSTOM
        ),
        aggregator=lambda reserve: float(np.min(reserve)),
    )

    assert result.node_star == "A"


def test_wrapper_forwards_custom_utility() -> None:
    result = optimize_interventions(
        (0.5,),
        ("A", "B"),
        lambda node_id: (
            make_simulation(
                "A",
                reserve_after=(0.9,),
                cost=10.0,
            )
            if node_id == "A"
            else make_simulation(
                "B",
                reserve_after=(0.7,),
                cost=1.0,
            )
        ),
        utility=lambda simulation, gain, config: (
            -simulation.cost
        ),
    )

    assert result.node_star == "B"


