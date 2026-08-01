"""Tests for roif.validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest

from roif.validation import (
    GraphFamily,
    GraphSpec,
    GroupSummary,
    MetricName,
    MetricSummary,
    TrialMetrics,
    TrialResult,
    TrialStatus,
    ValidationConfig,
    ValidationError,
    ValidationOutcome,
    ValidationResult,
    ValidationRunner,
    ValidationStatus,
    ValidationTrial,
    compute_trial_metrics,
    derive_seed,
    group_trial_summaries,
    make_trials,
    run_validation,
    summarize_metric,
    summarize_trials,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def graph_spec(
    *,
    family: GraphFamily = GraphFamily.LINEAR_CHAIN,
    node_count: int = 5,
    edge_count: int | None = 4,
    name: str | None = None,
) -> GraphSpec:
    return GraphSpec(
        family=family,
        node_count=node_count,
        edge_count=edge_count,
        parameters={"p": 0.2},
        name=name,
        metadata={"source": "test"},
    )


def trial(
    *,
    trial_id: str = "trial-1",
    repetition: int = 0,
    seed: int = 42,
    true_root: Any = "A",
    true_node_star: Any = "B",
    graph: GraphSpec | None = None,
) -> ValidationTrial:
    return ValidationTrial(
        trial_id=trial_id,
        repetition=repetition,
        seed=seed,
        graph=graph or graph_spec(),
        true_root=true_root,
        true_node_star=true_node_star,
        metadata={"source": "test"},
    )


def outcome(
    *,
    predicted_root: Any = "A",
    root_ranking: tuple[Any, ...] = ("A", "C", "B"),
    root_localization_distance: float | None = 0.0,
    predicted_node_star: Any = "B",
    node_star_ranking: tuple[Any, ...] = ("B", "A", "C"),
    selected_gain: float | None = 0.8,
    optimal_gain: float | None = 1.0,
    converged: bool = True,
    runtime_seconds: float | None = 0.5,
) -> ValidationOutcome:
    return ValidationOutcome(
        predicted_root=predicted_root,
        root_ranking=root_ranking,
        root_localization_distance=root_localization_distance,
        predicted_node_star=predicted_node_star,
        node_star_ranking=node_star_ranking,
        selected_gain=selected_gain,
        optimal_gain=optimal_gain,
        converged=converged,
        runtime_seconds=runtime_seconds,
        metadata={"source": "test"},
    )


def metrics() -> TrialMetrics:
    return TrialMetrics(
        root_correct=True,
        root_top_k={1: True, 3: True},
        root_localization_distance=0.0,
        node_star_correct=True,
        node_star_top_k={1: True, 3: True},
        node_star_gain_regret=0.2,
        converged=True,
        runtime_seconds=0.5,
        runtime_per_node=0.1,
    )


def successful_result(
    *,
    graph: GraphSpec | None = None,
    trial_id: str = "trial-1",
) -> TrialResult:
    return TrialResult(
        trial=trial(graph=graph, trial_id=trial_id),
        status=TrialStatus.SUCCEEDED,
        outcome=outcome(),
        metrics=metrics(),
        metadata={"source": "test"},
    )


def failed_result(
    *,
    graph: GraphSpec | None = None,
    trial_id: str = "trial-failed",
) -> TrialResult:
    return TrialResult(
        trial=trial(graph=graph, trial_id=trial_id),
        status=TrialStatus.FAILED,
        outcome=None,
        metrics=None,
        error_type="RuntimeError",
        error_message="boom",
    )


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


def test_enum_values_are_stable() -> None:
    assert tuple(item.value for item in GraphFamily) == (
        "linear_chain",
        "tree",
        "lattice",
        "erdos_renyi",
        "watts_strogatz",
        "barabasi_albert",
        "custom",
    )
    assert tuple(item.value for item in TrialStatus) == (
        "succeeded",
        "failed",
        "skipped",
    )
    assert tuple(item.value for item in ValidationStatus) == (
        "completed",
        "partial",
        "failed",
    )
    assert tuple(item.value for item in MetricName) == (
        "root_accuracy",
        "root_top_k_accuracy",
        "root_localization_distance",
        "node_star_accuracy",
        "node_star_top_k_accuracy",
        "node_star_gain_regret",
        "convergence_rate",
        "success_rate",
        "runtime_seconds",
        "runtime_per_node",
    )


# ---------------------------------------------------------------------------
# GraphSpec
# ---------------------------------------------------------------------------


def test_graph_spec_properties() -> None:
    spec = graph_spec(name="chain")

    assert spec.label == "chain"
    assert isinstance(spec.parameters, MappingProxyType)
    assert isinstance(spec.metadata, MappingProxyType)
    assert spec.as_dict()["family"] == "linear_chain"


def test_graph_spec_default_label() -> None:
    assert graph_spec(name=None).label == "linear_chain:5"


def test_graph_spec_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        graph_spec().node_count = 10  # type: ignore[misc]


@pytest.mark.parametrize("value", ("linear_chain", None, 1))
def test_graph_spec_rejects_invalid_family(value: Any) -> None:
    with pytest.raises(ValidationError):
        GraphSpec(value, 5)


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "5"))
def test_graph_spec_rejects_invalid_node_count(value: Any) -> None:
    with pytest.raises(ValidationError):
        GraphSpec(GraphFamily.TREE, value)


@pytest.mark.parametrize("value", (-1, True, 1.5, "4"))
def test_graph_spec_rejects_invalid_edge_count(value: Any) -> None:
    with pytest.raises(ValidationError):
        GraphSpec(GraphFamily.TREE, 5, edge_count=value)


# ---------------------------------------------------------------------------
# ValidationConfig
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    config = ValidationConfig()

    assert config.repetitions == 10
    assert config.base_seed == 0
    assert config.top_k == (1, 3, 5)
    assert config.capture_trial_errors is True
    assert config.measure_runtime is True


def test_config_sorts_top_k() -> None:
    config = ValidationConfig(top_k=(5, 1, 3))

    assert config.top_k == (1, 3, 5)


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "10"))
def test_config_rejects_invalid_repetitions(value: Any) -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(repetitions=value)


@pytest.mark.parametrize("value", (True, 1.5, "seed"))
def test_config_rejects_invalid_base_seed(value: Any) -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(base_seed=value)


@pytest.mark.parametrize(
    "value",
    ((), "bad", (1, 1), (0, 1), (True,)),
)
def test_config_rejects_invalid_top_k(value: Any) -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(top_k=value)


@pytest.mark.parametrize(
    "field",
    ("capture_trial_errors", "allow_partial_results", "measure_runtime"),
)
@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_config_rejects_invalid_boolean_flags(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    ("minimum_success_rate", "minimum_convergence_rate"),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, 1.1, float("nan"), True, "bad"),
)
def test_config_rejects_invalid_rates(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(**{field: value})


# ---------------------------------------------------------------------------
# ValidationTrial and ValidationOutcome
# ---------------------------------------------------------------------------


def test_trial_properties() -> None:
    item = trial(trial_id=" trial-1 ")

    assert item.trial_id == "trial-1"
    assert item.true_root == "A"
    assert isinstance(item.metadata, MappingProxyType)
    assert item.as_dict()["seed"] == 42


@pytest.mark.parametrize("value", ("", "   ", None, 1))
def test_trial_rejects_invalid_id(value: Any) -> None:
    with pytest.raises(ValidationError):
        trial(trial_id=value)


@pytest.mark.parametrize("value", (-1, True, 1.5, "0"))
def test_trial_rejects_invalid_repetition(value: Any) -> None:
    with pytest.raises(ValidationError):
        trial(repetition=value)


@pytest.mark.parametrize("value", (True, 1.5, "42"))
def test_trial_rejects_invalid_seed(value: Any) -> None:
    with pytest.raises(ValidationError):
        trial(seed=value)


def test_outcome_properties() -> None:
    item = outcome()

    assert item.gain_regret == pytest.approx(0.2)
    assert isinstance(item.metadata, MappingProxyType)
    assert item.as_dict()["predicted_root"] == "A"


def test_outcome_gain_regret_none() -> None:
    item = outcome(selected_gain=None)

    assert item.gain_regret is None


def test_outcome_rejects_duplicate_rankings() -> None:
    with pytest.raises(ValidationError):
        outcome(root_ranking=("A", "A"))


@pytest.mark.parametrize(
    "field",
    (
        "root_localization_distance",
        "runtime_seconds",
    ),
)
@pytest.mark.parametrize(
    "value",
    (-0.1, float("nan"), True, "bad"),
)
def test_outcome_rejects_invalid_nonnegative_values(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        outcome(**{field: value})


@pytest.mark.parametrize(
    "field",
    ("selected_gain", "optimal_gain"),
)
@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), True, "bad"),
)
def test_outcome_rejects_invalid_gain_values(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        outcome(**{field: value})


@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_outcome_rejects_invalid_converged(value: Any) -> None:
    with pytest.raises(ValidationError):
        outcome(converged=value)


# ---------------------------------------------------------------------------
# TrialMetrics and TrialResult
# ---------------------------------------------------------------------------


def test_trial_metrics_properties() -> None:
    item = metrics()

    assert item.root_correct is True
    assert item.runtime_per_node == pytest.approx(0.1)
    assert isinstance(item.root_top_k, MappingProxyType)
    assert item.as_dict()["node_star_gain_regret"] == pytest.approx(0.2)


@pytest.mark.parametrize("field", ("root_correct", "node_star_correct"))
@pytest.mark.parametrize("value", (0, 1, "yes"))
def test_trial_metrics_rejects_invalid_optional_flags(
    field: str,
    value: Any,
) -> None:
    kwargs = metrics().as_dict()
    kwargs[field] = value
    with pytest.raises(ValidationError):
        TrialMetrics(**kwargs)


def test_trial_metrics_rejects_invalid_top_k_mapping() -> None:
    with pytest.raises(ValidationError):
        TrialMetrics(
            root_correct=True,
            root_top_k={0: True},
            root_localization_distance=0.0,
            node_star_correct=True,
            node_star_top_k={1: True},
            node_star_gain_regret=0.0,
            converged=True,
            runtime_seconds=0.1,
            runtime_per_node=0.01,
        )


def test_successful_trial_result() -> None:
    result = successful_result()

    assert result.status is TrialStatus.SUCCEEDED
    assert result.outcome is not None
    assert result.as_dict()["status"] == "succeeded"


def test_failed_trial_result() -> None:
    result = failed_result()

    assert result.error_type == "RuntimeError"
    assert result.outcome is None


def test_successful_trial_requires_outputs() -> None:
    with pytest.raises(ValidationError):
        TrialResult(
            trial=trial(),
            status=TrialStatus.SUCCEEDED,
            outcome=None,
            metrics=None,
        )


def test_failed_trial_requires_error_details() -> None:
    with pytest.raises(ValidationError):
        TrialResult(
            trial=trial(),
            status=TrialStatus.FAILED,
            outcome=None,
            metrics=None,
        )


# ---------------------------------------------------------------------------
# MetricSummary and GroupSummary
# ---------------------------------------------------------------------------


def test_summarize_metric() -> None:
    summary = summarize_metric(
        MetricName.RUNTIME_SECONDS,
        (1.0, 2.0, 3.0),
    )

    assert summary.count == 3
    assert summary.mean == pytest.approx(2.0)
    assert summary.median == pytest.approx(2.0)
    assert summary.minimum == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(3.0)


def test_summarize_empty_metric() -> None:
    summary = summarize_metric(
        MetricName.RUNTIME_SECONDS,
        (),
    )

    assert summary.count == 0
    assert summary.mean is None


def test_metric_summary_rejects_partial_statistics() -> None:
    with pytest.raises(ValidationError):
        MetricSummary(
            name=MetricName.RUNTIME_SECONDS,
            count=1,
            mean=1.0,
            median=None,
            standard_deviation=0.0,
            minimum=1.0,
            maximum=1.0,
        )


def test_group_summary() -> None:
    group = GroupSummary(
        graph_family=GraphFamily.TREE,
        node_count=10,
        trial_count=2,
        successful_count=1,
        failed_count=1,
        metrics={
            "success_rate": summarize_metric(
                MetricName.SUCCESS_RATE,
                (0.5,),
            )
        },
    )

    assert group.graph_family is GraphFamily.TREE
    assert isinstance(group.metrics, MappingProxyType)


def test_group_summary_rejects_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        GroupSummary(
            graph_family=GraphFamily.TREE,
            node_count=10,
            trial_count=2,
            successful_count=2,
            failed_count=1,
            metrics={},
        )


# ---------------------------------------------------------------------------
# Seed and trial generation
# ---------------------------------------------------------------------------


def test_derive_seed_is_deterministic() -> None:
    first = derive_seed(123, 0, 0)
    second = derive_seed(123, 0, 0)

    assert first == second


def test_derive_seed_changes_with_indices() -> None:
    seeds = {
        derive_seed(123, graph_index, repetition)
        for graph_index in range(2)
        for repetition in range(3)
    }

    assert len(seeds) == 6


@pytest.mark.parametrize(
    ("graph_index", "repetition"),
    ((-1, 0), (0, -1), (True, 0)),
)
def test_derive_seed_rejects_invalid_indices(
    graph_index: Any,
    repetition: Any,
) -> None:
    with pytest.raises(ValidationError):
        derive_seed(0, graph_index, repetition)


def test_make_trials() -> None:
    specs = (
        graph_spec(node_count=5),
        graph_spec(
            family=GraphFamily.TREE,
            node_count=7,
            edge_count=6,
        ),
    )
    config = ValidationConfig(repetitions=2, base_seed=10)

    trials = make_trials(
        specs,
        config=config,
        true_root_factory=lambda spec, repetition, seed: (
            f"root-{spec.node_count}-{repetition}"
        ),
        true_node_star_factory=lambda spec, repetition, seed: (
            f"star-{spec.node_count}-{repetition}"
        ),
    )

    assert len(trials) == 4
    assert trials[0].trial_id.startswith("000-linear_chain-n5")
    assert trials[0].true_root == "root-5-0"
    assert len({item.seed for item in trials}) == 4


def test_make_trials_rejects_empty_specs() -> None:
    with pytest.raises(ValidationError):
        make_trials(())


# ---------------------------------------------------------------------------
# Trial metrics
# ---------------------------------------------------------------------------


def test_compute_trial_metrics() -> None:
    result = compute_trial_metrics(
        trial(),
        outcome(),
        top_k=(1, 3),
    )

    assert result.root_correct is True
    assert result.root_top_k == {1: True, 3: True}
    assert result.node_star_correct is True
    assert result.node_star_gain_regret == pytest.approx(0.2)
    assert result.runtime_per_node == pytest.approx(0.1)


def test_compute_trial_metrics_uses_default_distance() -> None:
    result = compute_trial_metrics(
        trial(true_root="A"),
        outcome(
            predicted_root="B",
            root_ranking=("B", "A"),
            root_localization_distance=None,
        ),
        top_k=(1, 2),
    )

    assert result.root_correct is False
    assert result.root_localization_distance == pytest.approx(1.0)
    assert result.root_top_k == {1: False, 2: True}


def test_compute_trial_metrics_uses_custom_distance() -> None:
    result = compute_trial_metrics(
        trial(true_root="A"),
        outcome(
            predicted_root="C",
            root_localization_distance=None,
        ),
        distance=lambda true, predicted, fixture: 2.5,
    )

    assert result.root_localization_distance == pytest.approx(2.5)


def test_compute_trial_metrics_rejects_negative_regret() -> None:
    with pytest.raises(ValidationError):
        compute_trial_metrics(
            trial(),
            outcome(selected_gain=2.0, optimal_gain=1.0),
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_summarize_trials() -> None:
    results = (successful_result(), failed_result())

    summary = summarize_trials(results, top_k=(1, 3))

    assert summary["success_rate"].mean == pytest.approx(0.5)
    assert summary["root_accuracy"].mean == pytest.approx(1.0)
    assert summary["root_top_1_accuracy"].mean == pytest.approx(1.0)
    assert summary["node_star_gain_regret"].mean == pytest.approx(0.2)


def test_group_trial_summaries() -> None:
    chain = graph_spec(node_count=5)
    tree = graph_spec(
        family=GraphFamily.TREE,
        node_count=7,
        edge_count=6,
    )
    results = (
        successful_result(graph=chain, trial_id="c1"),
        failed_result(graph=chain, trial_id="c2"),
        successful_result(graph=tree, trial_id="t1"),
    )

    groups = group_trial_summaries(results, top_k=(1,))

    assert len(groups) == 2
    assert groups[0].trial_count in (1, 2)
    assert sum(group.trial_count for group in groups) == 3


# ---------------------------------------------------------------------------
# ValidationRunner
# ---------------------------------------------------------------------------


class DeterministicClock:
    def __init__(self) -> None:
        self.values = iter((1.0, 1.5, 2.0, 2.25, 3.0, 3.75))

    def __call__(self) -> float:
        return next(self.values)


def test_runner_completed() -> None:
    runner = ValidationRunner(
        ValidationConfig(repetitions=2, top_k=(1, 2)),
        clock=DeterministicClock(),
    )

    result = runner.run(
        (graph_spec(),),
        trial_factory=lambda trial: {"trial": trial},
        execute_trial=lambda trial, fixture: {
            "predicted_root": "A",
            "root_ranking": ("A", "B"),
            "predicted_node_star": "B",
            "node_star_ranking": ("B", "A"),
            "selected_gain": 0.8,
            "optimal_gain": 1.0,
            "converged": True,
        },
        true_root_factory=lambda spec, repetition, seed: "A",
        true_node_star_factory=lambda spec, repetition, seed: "B",
    )

    assert result.status is ValidationStatus.COMPLETED
    assert result.successful_count == 2
    assert result.failed_count == 0
    assert result.metrics["root_accuracy"].mean == pytest.approx(1.0)
    assert result.trials[0].outcome.runtime_seconds == pytest.approx(0.5)


def test_runner_partial_on_trial_failure() -> None:
    runner = ValidationRunner(
        ValidationConfig(repetitions=2),
        clock=DeterministicClock(),
    )

    def execute(item: ValidationTrial, fixture: Any):
        if item.repetition == 1:
            raise RuntimeError("boom")
        return ValidationOutcome(
            predicted_root="A",
            predicted_node_star="B",
            converged=True,
        )

    result = runner.run(
        (graph_spec(),),
        trial_factory=lambda item: object(),
        execute_trial=execute,
        true_root_factory=lambda spec, repetition, seed: "A",
        true_node_star_factory=lambda spec, repetition, seed: "B",
    )

    assert result.status is ValidationStatus.PARTIAL
    assert result.successful_count == 1
    assert result.failed_count == 1


def test_runner_failed_when_partial_disallowed() -> None:
    runner = ValidationRunner(
        ValidationConfig(
            repetitions=2,
            allow_partial_results=False,
        ),
        clock=DeterministicClock(),
    )

    result = runner.run(
        (graph_spec(),),
        trial_factory=lambda item: object(),
        execute_trial=lambda item, fixture: (
            (_ for _ in ()).throw(RuntimeError("boom"))
            if item.repetition == 1
            else ValidationOutcome()
        ),
    )

    assert result.status is ValidationStatus.FAILED


def test_runner_can_propagate_trial_error() -> None:
    runner = ValidationRunner(
        ValidationConfig(
            repetitions=1,
            capture_trial_errors=False,
        )
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(
            (graph_spec(),),
            trial_factory=lambda item: object(),
            execute_trial=lambda item, fixture: (
                (_ for _ in ()).throw(RuntimeError("boom"))
            ),
        )


def test_runner_respects_minimum_rates() -> None:
    runner = ValidationRunner(
        ValidationConfig(
            repetitions=1,
            minimum_convergence_rate=1.0,
        ),
        clock=DeterministicClock(),
    )

    result = runner.run(
        (graph_spec(),),
        trial_factory=lambda item: object(),
        execute_trial=lambda item, fixture: ValidationOutcome(
            converged=False
        ),
    )

    assert result.status is ValidationStatus.FAILED


def test_runner_rejects_invalid_callbacks() -> None:
    runner = ValidationRunner()

    with pytest.raises(ValidationError):
        runner.run(
            (graph_spec(),),
            trial_factory=None,  # type: ignore[arg-type]
            execute_trial=lambda item, fixture: ValidationOutcome(),
        )

    with pytest.raises(ValidationError):
        runner.run(
            (graph_spec(),),
            trial_factory=lambda item: object(),
            execute_trial=None,  # type: ignore[arg-type]
        )


def test_run_validation_wrapper() -> None:
    result = run_validation(
        (graph_spec(),),
        config=ValidationConfig(
            repetitions=1,
            measure_runtime=False,
        ),
        trial_factory=lambda item: object(),
        execute_trial=lambda item, fixture: {
            "predicted_root": "A",
            "predicted_node_star": "B",
            "converged": True,
        },
        true_root_factory=lambda spec, repetition, seed: "A",
        true_node_star_factory=lambda spec, repetition, seed: "B",
        metadata={"experiment": "E1"},
    )

    assert isinstance(result, ValidationResult)
    assert result.status is ValidationStatus.COMPLETED
    assert result.metadata["experiment"] == "E1"
    assert "Validation completed" in result.summary


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


def test_validation_result_serialization() -> None:
    config = ValidationConfig(repetitions=1)
    spec = graph_spec()
    item = successful_result(graph=spec)

    result = ValidationResult(
        status=ValidationStatus.COMPLETED,
        config=config,
        graph_specs=(spec,),
        trials=(item,),
        metrics=summarize_trials((item,)),
        groups=group_trial_summaries((item,)),
        summary="Complete.",
        metadata={"source": "test"},
    )

    payload = result.as_dict()

    assert payload["status"] == "completed"
    assert payload["successful_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["graph_specs"][0]["family"] == "linear_chain"
    assert isinstance(result.metrics, MappingProxyType)
