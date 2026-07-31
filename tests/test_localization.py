"""Tests for ROIF inverse cascade localization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from roif.localization import (
    FastFailureResult,
    FastFailureStatus,
    LocalizationConfig,
    LocalizationError,
    LocalizationMetric,
    LocalizationScore,
    LocalizationSimulation,
    LocalizationStatus,
    RootLocalizationEngine,
    RootLocalizationResult,
    SimulationStatus,
    estimate_d_fast,
    localization_discrepancy,
    localize_root,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_simulation(
    candidate_id: Any = "A",
    *,
    predicted_state: tuple[float, ...] = (1.0, 2.0),
    converged: bool = True,
    iterations: int | None = 5,
    propagation_time: float | None = 1.5,
) -> LocalizationSimulation:
    return LocalizationSimulation(
        candidate_id=candidate_id,
        predicted_state=np.asarray(predicted_state, dtype=float),
        converged=converged,
        iterations=iterations,
        propagation_time=propagation_time,
        reserve_history={
            "A": ((0.0, 1.0), (1.0, 0.0)),
        },
        metadata={"source": "test"},
    )


def make_score(
    *,
    candidate_id: Any = "A",
    discrepancy: float = 0.1,
    rank: int = 1,
    tied_for_best: bool = False,
    simulation_status: SimulationStatus = SimulationStatus.SUCCEEDED,
    converged: bool | None = True,
    predicted_state: tuple[float, ...] | None = (1.0, 2.0),
    error_type: str | None = None,
    error_message: str | None = None,
) -> LocalizationScore:
    return LocalizationScore(
        candidate_id=candidate_id,
        discrepancy=discrepancy,
        rank=rank,
        tied_for_best=tied_for_best,
        simulation_status=simulation_status,
        converged=converged,
        predicted_state=(
            None
            if predicted_state is None
            else np.asarray(predicted_state, dtype=float)
        ),
        error_type=error_type,
        error_message=error_message,
        metadata={"source": "test"},
    )


def make_fast_result(
    *,
    status: FastFailureStatus = FastFailureStatus.IDENTIFIED,
    node_id: Any = "A",
    crossing_time: float | None = 1.0,
    tied_node_ids: tuple[Any, ...] = ("A",),
    crossing_times: dict[Any, float] | None = None,
) -> FastFailureResult:
    if crossing_times is None:
        crossing_times = {"A": 1.0}

    return FastFailureResult(
        status=status,
        node_id=node_id,
        crossing_time=crossing_time,
        tied_node_ids=tied_node_ids,
        critical_reserve=0.0,
        crossing_times=crossing_times,
        metadata={"source": "test"},
    )


def make_root_result(
    *,
    status: LocalizationStatus = LocalizationStatus.LOCALIZED,
    root_id: Any = "A",
    best_discrepancy: float | None = 0.1,
    tied_root_ids: tuple[Any, ...] = ("A",),
    scores: tuple[LocalizationScore, ...] | None = None,
    successful_candidate_count: int = 1,
    failed_candidate_count: int = 0,
) -> RootLocalizationResult:
    if scores is None:
        scores = (make_score(),)

    return RootLocalizationResult(
        status=status,
        root_id=root_id,
        best_discrepancy=best_discrepancy,
        tied_root_ids=tied_root_ids,
        scores=scores,
        observed_state=np.asarray((1.0, 2.0), dtype=float),
        metric=LocalizationMetric.L2,
        successful_candidate_count=successful_candidate_count,
        failed_candidate_count=failed_candidate_count,
        summary=" Localization summary. ",
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


def test_localization_metric_values_are_stable() -> None:
    assert tuple(item.value for item in LocalizationMetric) == (
        "l1",
        "l2",
        "linf",
        "weighted_l2",
        "custom",
    )


def test_localization_status_values_are_stable() -> None:
    assert tuple(item.value for item in LocalizationStatus) == (
        "localized",
        "tied",
        "partial",
        "failed",
    )


def test_simulation_status_values_are_stable() -> None:
    assert tuple(item.value for item in SimulationStatus) == (
        "succeeded",
        "failed",
    )


def test_fast_failure_status_values_are_stable() -> None:
    assert tuple(item.value for item in FastFailureStatus) == (
        "identified",
        "tied",
        "not_reached",
    )


# ---------------------------------------------------------------------------
# LocalizationConfig
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    config = LocalizationConfig()

    assert config.metric is LocalizationMetric.L2
    assert config.weights is None
    assert config.tie_tolerance == pytest.approx(1e-9)
    assert config.top_k == 5
    assert config.critical_reserve == pytest.approx(0.0)
    assert config.allow_partial_results is True
    assert config.capture_simulation_errors is True
    assert config.normalize_by_dimension is False


def test_config_normalizes_values() -> None:
    config = LocalizationConfig(
        tie_tolerance="0.1",
        top_k=3,
        critical_reserve="-0.2",
    )

    assert config.tie_tolerance == pytest.approx(0.1)
    assert config.critical_reserve == pytest.approx(-0.2)


def test_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        LocalizationConfig().top_k = 10  # type: ignore[misc]


@pytest.mark.parametrize("value", ("l2", None, 1))
def test_config_rejects_invalid_metric(value: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(metric=value)


@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), float("inf"), True, "bad"),
)
def test_config_rejects_invalid_tie_tolerance(value: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(tie_tolerance=value)


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "5"))
def test_config_rejects_invalid_top_k(value: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(top_k=value)


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_config_rejects_invalid_critical_reserve(value: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(critical_reserve=value)


@pytest.mark.parametrize(
    "field",
    (
        "allow_partial_results",
        "capture_simulation_errors",
        "normalize_by_dimension",
    ),
)
@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_config_requires_boolean_flags(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(**{field: value})


def test_weighted_l2_requires_weights() -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(metric=LocalizationMetric.WEIGHTED_L2)


def test_weights_only_allowed_for_weighted_l2() -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(weights=(1.0, 1.0))


def test_weighted_l2_accepts_weights() -> None:
    config = LocalizationConfig(
        metric=LocalizationMetric.WEIGHTED_L2,
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
def test_config_rejects_invalid_weights(weights: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationConfig(
            metric=LocalizationMetric.WEIGHTED_L2,
            weights=weights,
        )


# ---------------------------------------------------------------------------
# LocalizationSimulation
# ---------------------------------------------------------------------------


def test_simulation_properties() -> None:
    simulation = make_simulation(candidate_id=" A ")

    assert simulation.candidate_id == "A"
    assert np.allclose(simulation.predicted_state, (1.0, 2.0))
    assert simulation.converged is True
    assert simulation.iterations == 5
    assert simulation.propagation_time == pytest.approx(1.5)
    assert isinstance(simulation.metadata, MappingProxyType)
    assert isinstance(simulation.reserve_history, MappingProxyType)


def test_simulation_state_is_read_only() -> None:
    simulation = make_simulation()

    assert simulation.predicted_state.flags.writeable is False
    with pytest.raises(ValueError):
        simulation.predicted_state[0] = 99.0


def test_simulation_as_dict() -> None:
    payload = make_simulation().as_dict()

    assert payload["candidate_id"] == "A"
    assert payload["predicted_state"] == [1.0, 2.0]
    assert payload["reserve_history"]["A"] == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]


@pytest.mark.parametrize("candidate_id", (None, "", "   ", []))
def test_simulation_rejects_invalid_candidate_id(
    candidate_id: Any,
) -> None:
    with pytest.raises(LocalizationError):
        make_simulation(candidate_id=candidate_id)


@pytest.mark.parametrize(
    "state",
    (
        (),
        ((1.0, 2.0),),
        (1.0, float("nan")),
        "bad",
    ),
)
def test_simulation_rejects_invalid_predicted_state(state: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationSimulation("A", state)


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_simulation_rejects_invalid_converged(value: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationSimulation("A", (1.0,), converged=value)


@pytest.mark.parametrize("value", (-1, True, 1.5, "1"))
def test_simulation_rejects_invalid_iterations(value: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationSimulation("A", (1.0,), iterations=value)


@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), True, "bad"),
)
def test_simulation_rejects_invalid_propagation_time(value: Any) -> None:
    with pytest.raises(LocalizationError):
        LocalizationSimulation(
            "A",
            (1.0,),
            propagation_time=value,
        )


def test_simulation_rejects_non_chronological_history() -> None:
    with pytest.raises(LocalizationError):
        LocalizationSimulation(
            "A",
            (1.0,),
            reserve_history={
                "A": ((2.0, 1.0), (1.0, 0.0)),
            },
        )


# ---------------------------------------------------------------------------
# LocalizationScore
# ---------------------------------------------------------------------------


def test_score_properties() -> None:
    score = make_score(candidate_id=" A ")

    assert score.candidate_id == "A"
    assert score.discrepancy == pytest.approx(0.1)
    assert score.rank == 1
    assert score.simulation_status is SimulationStatus.SUCCEEDED
    assert isinstance(score.metadata, MappingProxyType)


def test_score_as_dict() -> None:
    payload = make_score().as_dict()

    assert payload["candidate_id"] == "A"
    assert payload["simulation_status"] == "succeeded"
    assert payload["predicted_state"] == [1.0, 2.0]


def test_failed_score_is_valid_with_error_details() -> None:
    score = make_score(
        discrepancy=np.finfo(float).max,
        simulation_status=SimulationStatus.FAILED,
        converged=None,
        predicted_state=None,
        error_type="RuntimeError",
        error_message="boom",
    )

    assert score.error_type == "RuntimeError"
    assert score.predicted_state is None


@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), float("inf"), True, "bad"),
)
def test_score_rejects_invalid_discrepancy(value: Any) -> None:
    with pytest.raises(LocalizationError):
        make_score(discrepancy=value)


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "1"))
def test_score_rejects_invalid_rank(value: Any) -> None:
    with pytest.raises(LocalizationError):
        make_score(rank=value)


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_score_rejects_invalid_tied_flag(value: Any) -> None:
    with pytest.raises(LocalizationError):
        make_score(tied_for_best=value)


@pytest.mark.parametrize("value", ("succeeded", None, 1))
def test_score_rejects_invalid_simulation_status(value: Any) -> None:
    with pytest.raises(LocalizationError):
        make_score(simulation_status=value)


def test_failed_score_requires_error_details() -> None:
    with pytest.raises(LocalizationError):
        make_score(
            simulation_status=SimulationStatus.FAILED,
            converged=None,
            predicted_state=None,
        )


def test_successful_score_disallows_error_details() -> None:
    with pytest.raises(LocalizationError):
        make_score(
            error_type="Unexpected",
            error_message="Unexpected",
        )


# ---------------------------------------------------------------------------
# FastFailureResult and estimate_d_fast
# ---------------------------------------------------------------------------


def test_fast_failure_result_properties() -> None:
    result = make_fast_result()

    assert result.status is FastFailureStatus.IDENTIFIED
    assert result.node_id == "A"
    assert result.crossing_time == pytest.approx(1.0)
    assert isinstance(result.crossing_times, MappingProxyType)


def test_fast_failure_result_as_dict() -> None:
    payload = make_fast_result().as_dict()

    assert payload["status"] == "identified"
    assert payload["node_id"] == "A"
    assert payload["crossing_times"] == {"A": 1.0}


def test_not_reached_result_is_valid() -> None:
    result = make_fast_result(
        status=FastFailureStatus.NOT_REACHED,
        node_id=None,
        crossing_time=None,
        tied_node_ids=(),
        crossing_times={},
    )

    assert result.node_id is None


def test_not_reached_disallows_result_values() -> None:
    with pytest.raises(LocalizationError):
        make_fast_result(
            status=FastFailureStatus.NOT_REACHED,
        )


def test_identified_requires_node_and_time() -> None:
    with pytest.raises(LocalizationError):
        make_fast_result(node_id=None)


def test_fast_result_rejects_duplicate_tied_ids() -> None:
    with pytest.raises(LocalizationError):
        make_fast_result(tied_node_ids=("A", "A"))


def test_estimate_d_fast_identifies_earliest_crossing() -> None:
    result = estimate_d_fast(
        {
            "A": ((0.0, 1.0), (2.0, 0.0)),
            "B": ((0.0, 1.0), (1.0, 0.0)),
        }
    )

    assert result.status is FastFailureStatus.IDENTIFIED
    assert result.node_id == "B"
    assert result.crossing_time == pytest.approx(1.0)


def test_estimate_d_fast_detects_tie() -> None:
    result = estimate_d_fast(
        {
            "A": ((0.0, 1.0), (1.0, 0.0)),
            "B": ((0.0, 1.0), (1.0, -0.1)),
        }
    )

    assert result.status is FastFailureStatus.TIED
    assert result.tied_node_ids == ("A", "B")
    assert result.node_id == "A"


def test_estimate_d_fast_respects_threshold() -> None:
    result = estimate_d_fast(
        {
            "A": ((0.0, 1.0), (1.0, 0.4)),
            "B": ((0.0, 1.0), (2.0, 0.6)),
        },
        critical_reserve=0.5,
    )

    assert result.node_id == "A"


def test_estimate_d_fast_returns_not_reached() -> None:
    result = estimate_d_fast(
        {
            "A": ((0.0, 1.0), (1.0, 0.5)),
        },
        critical_reserve=0.0,
    )

    assert result.status is FastFailureStatus.NOT_REACHED


def test_estimate_d_fast_uses_first_crossing_only() -> None:
    result = estimate_d_fast(
        {
            "A": (
                (0.0, 1.0),
                (1.0, 0.0),
                (2.0, -1.0),
            ),
        }
    )

    assert result.crossing_times["A"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "history",
    (
        "bad",
        {"A": "bad"},
        {"A": ((1.0,),)},
        {"A": ((2.0, 1.0), (1.0, 0.0))},
    ),
)
def test_estimate_d_fast_rejects_invalid_history(history: Any) -> None:
    with pytest.raises(LocalizationError):
        estimate_d_fast(history)


# ---------------------------------------------------------------------------
# localization_discrepancy
# ---------------------------------------------------------------------------


def test_l1_discrepancy() -> None:
    value = localization_discrepancy(
        (1.0, 2.0),
        (2.0, 4.0),
        config=LocalizationConfig(metric=LocalizationMetric.L1),
    )

    assert value == pytest.approx(3.0)


def test_l2_discrepancy() -> None:
    value = localization_discrepancy(
        (0.0, 0.0),
        (3.0, 4.0),
    )

    assert value == pytest.approx(5.0)


def test_linf_discrepancy() -> None:
    value = localization_discrepancy(
        (1.0, 2.0),
        (4.0, 0.0),
        config=LocalizationConfig(metric=LocalizationMetric.LINF),
    )

    assert value == pytest.approx(3.0)


def test_weighted_l2_discrepancy() -> None:
    value = localization_discrepancy(
        (0.0, 0.0),
        (1.0, 2.0),
        config=LocalizationConfig(
            metric=LocalizationMetric.WEIGHTED_L2,
            weights=(4.0, 1.0),
        ),
    )

    assert value == pytest.approx(np.sqrt(8.0))


def test_custom_discrepancy() -> None:
    value = localization_discrepancy(
        (1.0, 2.0),
        (4.0, 6.0),
        config=LocalizationConfig(metric=LocalizationMetric.CUSTOM),
        distance=lambda observed, predicted: float(
            np.sum(np.abs(predicted - observed))
        ),
    )

    assert value == pytest.approx(7.0)


def test_dimension_normalization() -> None:
    value = localization_discrepancy(
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        config=LocalizationConfig(normalize_by_dimension=True),
    )

    assert value == pytest.approx(1.0)


def test_discrepancy_rejects_dimension_mismatch() -> None:
    with pytest.raises(LocalizationError):
        localization_discrepancy((1.0, 2.0), (1.0,))


def test_custom_metric_requires_distance() -> None:
    with pytest.raises(LocalizationError):
        localization_discrepancy(
            (1.0,),
            (1.0,),
            config=LocalizationConfig(
                metric=LocalizationMetric.CUSTOM
            ),
        )


def test_builtin_metric_disallows_distance() -> None:
    with pytest.raises(LocalizationError):
        localization_discrepancy(
            (1.0,),
            (1.0,),
            distance=lambda a, b: 0.0,
        )


@pytest.mark.parametrize(
    "value",
    (
        -1.0,
        float("nan"),
        float("inf"),
        True,
        "bad",
    ),
)
def test_custom_metric_rejects_invalid_result(value: Any) -> None:
    with pytest.raises(LocalizationError):
        localization_discrepancy(
            (1.0,),
            (1.0,),
            config=LocalizationConfig(
                metric=LocalizationMetric.CUSTOM
            ),
            distance=lambda a, b: value,
        )


# ---------------------------------------------------------------------------
# RootLocalizationResult
# ---------------------------------------------------------------------------


def test_root_result_properties() -> None:
    result = make_root_result()

    assert result.root_id == "A"
    assert result.d_root == "A"
    assert result.is_tied is False
    assert result.summary == "Localization summary."
    assert result.observed_state.flags.writeable is False


def test_root_result_as_dict() -> None:
    payload = make_root_result().as_dict()

    assert payload["status"] == "localized"
    assert payload["d_root"] == "A"
    assert payload["metric"] == "l2"


def test_failed_root_result_is_valid_without_root() -> None:
    failed_score = make_score(
        discrepancy=np.finfo(float).max,
        simulation_status=SimulationStatus.FAILED,
        converged=None,
        predicted_state=None,
        error_type="RuntimeError",
        error_message="boom",
    )
    result = make_root_result(
        status=LocalizationStatus.FAILED,
        root_id=None,
        best_discrepancy=None,
        tied_root_ids=(),
        scores=(failed_score,),
        successful_candidate_count=0,
        failed_candidate_count=1,
    )

    assert result.d_root is None


def test_failed_root_result_disallows_root() -> None:
    with pytest.raises(LocalizationError):
        make_root_result(status=LocalizationStatus.FAILED)


def test_successful_root_result_requires_root() -> None:
    with pytest.raises(LocalizationError):
        make_root_result(root_id=None)


def test_root_result_rejects_count_mismatch() -> None:
    with pytest.raises(LocalizationError):
        make_root_result(
            successful_candidate_count=2,
        )


def test_root_result_rejects_duplicate_tied_ids() -> None:
    with pytest.raises(LocalizationError):
        make_root_result(tied_root_ids=("A", "A"))


# ---------------------------------------------------------------------------
# RootLocalizationEngine and localize_root
# ---------------------------------------------------------------------------


def test_engine_defaults() -> None:
    engine = RootLocalizationEngine()

    assert isinstance(engine.config, LocalizationConfig)


def test_engine_accepts_config() -> None:
    config = LocalizationConfig(metric=LocalizationMetric.L1)
    engine = RootLocalizationEngine(config)

    assert engine.config is config


@pytest.mark.parametrize("config", ({}, "bad", 1))
def test_engine_rejects_invalid_config(config: Any) -> None:
    with pytest.raises(LocalizationError):
        RootLocalizationEngine(config)


def test_custom_engine_requires_distance() -> None:
    with pytest.raises(LocalizationError):
        RootLocalizationEngine(
            LocalizationConfig(metric=LocalizationMetric.CUSTOM)
        )


def test_builtin_engine_disallows_distance() -> None:
    with pytest.raises(LocalizationError):
        RootLocalizationEngine(
            distance=lambda a, b: 0.0
        )


@pytest.mark.parametrize("simulate", (None, "bad", 1))
def test_localize_rejects_invalid_simulator(simulate: Any) -> None:
    with pytest.raises(LocalizationError):
        RootLocalizationEngine().localize(
            (1.0,),
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
def test_localize_rejects_invalid_candidates(
    candidates: Any,
) -> None:
    with pytest.raises(LocalizationError):
        RootLocalizationEngine().localize(
            (1.0,),
            candidates,
            lambda _: (1.0,),
        )


def test_localize_selects_minimum_discrepancy() -> None:
    predicted = {
        "A": (1.0, 1.0),
        "B": (2.0, 2.0),
        "C": (4.0, 4.0),
    }

    result = RootLocalizationEngine().localize(
        (2.1, 1.9),
        ("A", "B", "C"),
        lambda node_id: predicted[node_id],
    )

    assert result.status is LocalizationStatus.LOCALIZED
    assert result.root_id == "B"
    assert result.best_discrepancy == pytest.approx(
        np.sqrt(0.02)
    )
    assert result.successful_candidate_count == 3
    assert result.failed_candidate_count == 0


def test_localize_accepts_simulation_objects() -> None:
    result = RootLocalizationEngine().localize(
        (1.0, 2.0),
        ("A", "B"),
        lambda node_id: LocalizationSimulation(
            node_id,
            (1.0, 2.0) if node_id == "A" else (3.0, 4.0),
            converged=node_id == "A",
            iterations=2,
        ),
    )

    assert result.root_id == "A"
    assert result.scores[0].metadata["iterations"] == 2


def test_localize_detects_tie() -> None:
    result = RootLocalizationEngine(
        LocalizationConfig(tie_tolerance=1e-6)
    ).localize(
        (0.0,),
        ("A", "B"),
        lambda node_id: (
            (1.0,)
            if node_id == "A"
            else (-1.0,)
        ),
    )

    assert result.status is LocalizationStatus.TIED
    assert result.tied_root_ids == ("A", "B")
    assert result.root_id == "A"
    assert all(score.tied_for_best for score in result.scores)


def test_localize_honors_top_k() -> None:
    result = RootLocalizationEngine(
        LocalizationConfig(top_k=2)
    ).localize(
        (0.0,),
        ("A", "B", "C"),
        lambda node_id: {
            "A": (1.0,),
            "B": (2.0,),
            "C": (3.0,),
        }[node_id],
    )

    assert len(result.scores) == 3
    assert tuple(
        score.candidate_id
        for score in result.scores[:2]
    ) == (
        "A",
        "B",
    )


def test_localize_marks_partial_when_one_candidate_fails() -> None:
    def simulate(node_id: str):
        if node_id == "B":
            raise RuntimeError("boom")
        return (0.0,) if node_id == "A" else (2.0,)

    result = RootLocalizationEngine().localize(
        (0.0,),
        ("A", "B", "C"),
        simulate,
    )

    assert result.status is LocalizationStatus.PARTIAL
    assert result.root_id == "A"
    assert result.successful_candidate_count == 2
    assert result.failed_candidate_count == 1
    assert any(
        score.simulation_status is SimulationStatus.FAILED
        for score in result.scores
    )


def test_localize_rejects_partial_when_disabled() -> None:
    def simulate(node_id: str):
        if node_id == "B":
            raise RuntimeError("boom")
        return (0.0,)

    result = RootLocalizationEngine(
        LocalizationConfig(allow_partial_results=False)
    ).localize(
        (0.0,),
        ("A", "B"),
        simulate,
    )

    assert result.status is LocalizationStatus.FAILED
    assert result.root_id is None


def test_localize_can_propagate_simulation_error() -> None:
    engine = RootLocalizationEngine(
        LocalizationConfig(capture_simulation_errors=False)
    )

    with pytest.raises(RuntimeError, match="boom"):
        engine.localize(
            (0.0,),
            ("A",),
            lambda _: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )


def test_localize_rejects_wrong_simulation_candidate_id() -> None:
    result = RootLocalizationEngine().localize(
        (0.0,),
        ("A", "B"),
        lambda node_id: LocalizationSimulation(
            "wrong",
            (0.0,),
        ),
    )

    assert result.status is LocalizationStatus.FAILED
    assert result.failed_candidate_count == 2


def test_localize_rejects_wrong_predicted_dimension() -> None:
    result = RootLocalizationEngine().localize(
        (0.0, 1.0),
        ("A",),
        lambda _: (0.0,),
    )

    assert result.status is LocalizationStatus.FAILED


def test_localize_all_failures_returns_failed_result() -> None:
    result = RootLocalizationEngine().localize(
        (0.0,),
        ("A", "B"),
        lambda _: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    assert result.status is LocalizationStatus.FAILED
    assert result.root_id is None
    assert result.successful_candidate_count == 0
    assert result.failed_candidate_count == 2
    assert len(result.scores) == 2


def test_localize_custom_metric() -> None:
    result = RootLocalizationEngine(
        LocalizationConfig(metric=LocalizationMetric.CUSTOM),
        distance=lambda observed, predicted: float(
            abs(predicted[0] - observed[0])
        ),
    ).localize(
        (5.0,),
        ("A", "B"),
        lambda node_id: (4.0,) if node_id == "A" else (10.0,),
    )

    assert result.root_id == "A"


def test_localize_weighted_l2_changes_winner() -> None:
    config = LocalizationConfig(
        metric=LocalizationMetric.WEIGHTED_L2,
        weights=(100.0, 1.0),
    )
    result = RootLocalizationEngine(config).localize(
        (0.0, 0.0),
        ("A", "B"),
        lambda node_id: (
            (0.2, 10.0)
            if node_id == "A"
            else (1.0, 0.0)
        ),
    )

    assert result.root_id == "B"


def test_localize_metadata_is_forwarded() -> None:
    metadata = {"experiment": "E1"}
    result = RootLocalizationEngine().localize(
        (0.0,),
        ("A",),
        lambda _: (0.0,),
        metadata=metadata,
    )
    metadata["experiment"] = "changed"

    assert result.metadata["experiment"] == "E1"


def test_localize_root_wrapper() -> None:
    result = localize_root(
        (0.0,),
        ("A", "B"),
        lambda node_id: (0.0,) if node_id == "A" else (1.0,),
        metadata={"wrapper": True},
    )

    assert isinstance(result, RootLocalizationResult)
    assert result.root_id == "A"
    assert result.metadata["wrapper"] is True


def test_localize_root_forwards_config() -> None:
    result = localize_root(
        (0.0,),
        ("A", "B"),
        lambda node_id: (1.0,) if node_id == "A" else (-1.0,),
        config=LocalizationConfig(tie_tolerance=1e-6),
    )

    assert result.status is LocalizationStatus.TIED


def test_result_ranking_is_deterministic() -> None:
    result = RootLocalizationEngine().localize(
        (0.0,),
        ("B", "A"),
        lambda _: (1.0,),
    )

    assert tuple(score.candidate_id for score in result.scores) == (
        "A",
        "B",
    )


def test_localization_summary_contains_root() -> None:
    result = RootLocalizationEngine().localize(
        (0.0,),
        ("A",),
        lambda _: (0.0,),
    )

    assert "D_root candidate: A." in result.summary
    assert "Minimum localization discrepancy" in result.summary


def test_tied_summary_contains_candidates() -> None:
    result = RootLocalizationEngine().localize(
        (0.0,),
        ("A", "B"),
        lambda node_id: (1.0,) if node_id == "A" else (-1.0,),
    )

    assert "D_root is tied" in result.summary
    assert "A, B" in result.summary

