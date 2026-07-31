"""Tests for roif.benchmark."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest

from roif.benchmark import (
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkMethod,
    BenchmarkMethodKind,
    BenchmarkResult,
    BenchmarkRunStatus,
    BenchmarkRunner,
    BenchmarkStatus,
    MethodRunResult,
    PairedDelta,
    paired_deltas,
    run_benchmark,
    summarize_methods,
)
from roif.validation import (
    GraphFamily,
    GraphSpec,
    TrialMetrics,
    ValidationOutcome,
    ValidationTrial,
)


def spec() -> GraphSpec:
    return GraphSpec(
        GraphFamily.LINEAR_CHAIN,
        node_count=5,
        edge_count=4,
    )


def trial(trial_id: str = "t1") -> ValidationTrial:
    return ValidationTrial(
        trial_id=trial_id,
        repetition=0,
        seed=42,
        graph=spec(),
        true_root="A",
        true_node_star="B",
    )


def metrics(
    *,
    root_correct: bool = True,
    star_correct: bool = True,
    distance: float = 0.0,
    regret: float = 0.1,
    runtime: float = 0.5,
) -> TrialMetrics:
    return TrialMetrics(
        root_correct=root_correct,
        root_top_k={1: root_correct, 3: True},
        root_localization_distance=distance,
        node_star_correct=star_correct,
        node_star_top_k={1: star_correct, 3: True},
        node_star_gain_regret=regret,
        converged=True,
        runtime_seconds=runtime,
        runtime_per_node=runtime / 5,
    )


def outcome(
    *,
    root: str = "A",
    star: str = "B",
    runtime: float | None = 0.5,
) -> ValidationOutcome:
    return ValidationOutcome(
        predicted_root=root,
        root_ranking=(root, "B", "C"),
        predicted_node_star=star,
        node_star_ranking=(star, "A", "C"),
        selected_gain=0.9 if star == "B" else 0.6,
        optimal_gain=1.0,
        converged=True,
        runtime_seconds=runtime,
    )


def method(
    method_id: str = "roif",
    *,
    kind: BenchmarkMethodKind = BenchmarkMethodKind.ROIF,
):
    return BenchmarkMethod(
        method_id=method_id,
        label=method_id.upper(),
        kind=kind,
        execute=lambda validation_trial, fixture: outcome(),
        metadata={"source": "test"},
    )


def run_result(
    method_id: str,
    trial_id: str,
    *,
    root_correct: bool = True,
    star_correct: bool = True,
    distance: float = 0.0,
    regret: float = 0.1,
    runtime: float = 0.5,
) -> MethodRunResult:
    return MethodRunResult(
        method_id=method_id,
        trial_id=trial_id,
        status=BenchmarkRunStatus.SUCCEEDED,
        outcome=outcome(runtime=runtime),
        metrics=metrics(
            root_correct=root_correct,
            star_correct=star_correct,
            distance=distance,
            regret=regret,
            runtime=runtime,
        ),
        runtime_seconds=runtime,
    )


def test_enum_values() -> None:
    assert tuple(x.value for x in BenchmarkMethodKind) == (
        "roif", "baseline"
    )
    assert tuple(x.value for x in BenchmarkRunStatus) == (
        "succeeded", "failed"
    )
    assert tuple(x.value for x in BenchmarkStatus) == (
        "completed", "partial", "failed"
    )


def test_method_properties() -> None:
    item = method()
    assert item.method_id == "roif"
    assert item.kind is BenchmarkMethodKind.ROIF
    assert isinstance(item.metadata, MappingProxyType)
    assert item.as_dict()["kind"] == "roif"


def test_method_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        method().label = "X"  # type: ignore[misc]


@pytest.mark.parametrize("value", ("", "   ", None, 1))
def test_method_rejects_bad_id(value: Any) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkMethod(
            value,
            "Label",
            BenchmarkMethodKind.ROIF,
            lambda t, f: outcome(),
        )


def test_method_rejects_non_callable_executor() -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkMethod(
            "x",
            "X",
            BenchmarkMethodKind.BASELINE,
            None,  # type: ignore[arg-type]
        )


def test_config_defaults_and_sorting() -> None:
    config = BenchmarkConfig(top_k=(5, 1, 3))
    assert config.repetitions == 10
    assert config.top_k == (1, 3, 5)
    assert config.reference_method_id == "roif"


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "10"))
def test_config_rejects_bad_repetitions(value: Any) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkConfig(repetitions=value)


@pytest.mark.parametrize(
    "value", ((), "bad", (1, 1), (0, 1), (True,))
)
def test_config_rejects_bad_top_k(value: Any) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkConfig(top_k=value)


@pytest.mark.parametrize(
    "field",
    (
        "capture_method_errors",
        "measure_runtime",
        "allow_partial_results",
    ),
)
@pytest.mark.parametrize("value", (0, 1, "yes", None))
def test_config_rejects_bad_flags(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkConfig(**{field: value})


def test_method_run_success() -> None:
    result = run_result("roif", "t1")
    assert result.status is BenchmarkRunStatus.SUCCEEDED
    assert result.metrics is not None
    assert result.as_dict()["status"] == "succeeded"


def test_method_run_failure() -> None:
    result = MethodRunResult(
        method_id="baseline",
        trial_id="t1",
        status=BenchmarkRunStatus.FAILED,
        outcome=None,
        metrics=None,
        runtime_seconds=None,
        error_type="RuntimeError",
        error_message="boom",
    )
    assert result.error_type == "RuntimeError"


def test_successful_run_requires_outputs() -> None:
    with pytest.raises(BenchmarkError):
        MethodRunResult(
            method_id="x",
            trial_id="t1",
            status=BenchmarkRunStatus.SUCCEEDED,
            outcome=None,
            metrics=None,
            runtime_seconds=None,
        )


def test_failed_run_requires_error() -> None:
    with pytest.raises(BenchmarkError):
        MethodRunResult(
            method_id="x",
            trial_id="t1",
            status=BenchmarkRunStatus.FAILED,
            outcome=None,
            metrics=None,
            runtime_seconds=None,
        )


def test_paired_delta_validation() -> None:
    item = PairedDelta(
        method_id="baseline",
        reference_method_id="roif",
        metric="root_accuracy",
        count=2,
        mean_delta=-0.5,
        median_delta=-0.5,
        win_rate=0.0,
    )
    assert item.as_dict()["count"] == 2


def test_empty_paired_delta() -> None:
    item = PairedDelta(
        method_id="baseline",
        reference_method_id="roif",
        metric="root_accuracy",
        count=0,
        mean_delta=None,
        median_delta=None,
        win_rate=None,
    )
    assert item.count == 0


def test_summarize_methods() -> None:
    methods = (
        method("roif"),
        method(
            "degree",
            kind=BenchmarkMethodKind.BASELINE,
        ),
    )
    runs = (
        run_result("roif", "t1"),
        run_result(
            "degree",
            "t1",
            root_correct=False,
            star_correct=False,
            distance=2.0,
            regret=0.4,
        ),
    )

    summaries = summarize_methods(
        methods,
        runs,
        top_k=(1, 3),
    )

    assert len(summaries) == 2
    assert summaries[0].metrics["root_accuracy"].mean == pytest.approx(1.0)
    assert summaries[1].metrics["root_accuracy"].mean == pytest.approx(0.0)


def test_paired_deltas() -> None:
    runs = (
        run_result("roif", "t1", runtime=1.0),
        run_result(
            "degree",
            "t1",
            root_correct=False,
            star_correct=False,
            distance=2.0,
            regret=0.5,
            runtime=0.2,
        ),
    )

    deltas = paired_deltas(
        runs,
        reference_method_id="roif",
        method_ids=("roif", "degree"),
    )

    root = next(
        item for item in deltas
        if item.method_id == "degree"
        and item.metric == "root_accuracy"
    )
    runtime = next(
        item for item in deltas
        if item.method_id == "degree"
        and item.metric == "runtime_seconds"
    )

    assert root.mean_delta == pytest.approx(-1.0)
    assert root.win_rate == pytest.approx(0.0)
    assert runtime.mean_delta == pytest.approx(-0.8)
    assert runtime.win_rate == pytest.approx(1.0)


class Clock:
    def __init__(self) -> None:
        self.values = iter(
            (1.0, 1.5, 2.0, 2.2, 3.0, 3.4, 4.0, 4.1)
        )

    def __call__(self) -> float:
        return next(self.values)


def test_runner_completed() -> None:
    roif = BenchmarkMethod(
        "roif",
        "ROIF",
        BenchmarkMethodKind.ROIF,
        lambda t, f: {
            "predicted_root": "A",
            "root_ranking": ("A", "B"),
            "predicted_node_star": "B",
            "node_star_ranking": ("B", "A"),
            "selected_gain": 0.9,
            "optimal_gain": 1.0,
            "converged": True,
        },
    )
    baseline = BenchmarkMethod(
        "degree",
        "Degree",
        BenchmarkMethodKind.BASELINE,
        lambda t, f: {
            "predicted_root": "C",
            "root_ranking": ("C", "A"),
            "predicted_node_star": "A",
            "node_star_ranking": ("A", "B"),
            "selected_gain": 0.5,
            "optimal_gain": 1.0,
            "converged": True,
        },
    )

    result = BenchmarkRunner(
        BenchmarkConfig(repetitions=2, top_k=(1, 2)),
        clock=Clock(),
    ).run(
        (spec(),),
        (roif, baseline),
        fixture_factory=lambda t: {"seed": t.seed},
        true_root_factory=lambda s, r, seed: "A",
        true_node_star_factory=lambda s, r, seed: "B",
    )

    assert result.status is BenchmarkStatus.COMPLETED
    assert result.successful_run_count == 4
    assert len(result.method_summaries) == 2
    roif_summary = next(
        x for x in result.method_summaries
        if x.method.method_id == "roif"
    )
    assert roif_summary.metrics["root_accuracy"].mean == pytest.approx(1.0)
    assert result.paired_deltas


def test_runner_partial() -> None:
    good = method("roif")

    bad = BenchmarkMethod(
        "bad",
        "Bad",
        BenchmarkMethodKind.BASELINE,
        lambda t, f: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = BenchmarkRunner(
        BenchmarkConfig(repetitions=1)
    ).run(
        (spec(),),
        (good, bad),
        fixture_factory=lambda t: object(),
        true_root_factory=lambda s, r, seed: "A",
        true_node_star_factory=lambda s, r, seed: "B",
    )

    assert result.status is BenchmarkStatus.PARTIAL
    assert result.failed_run_count == 1


def test_runner_failed_when_partial_disallowed() -> None:
    result = BenchmarkRunner(
        BenchmarkConfig(
            repetitions=1,
            allow_partial_results=False,
        )
    ).run(
        (spec(),),
        (
            method("roif"),
            BenchmarkMethod(
                "bad",
                "Bad",
                BenchmarkMethodKind.BASELINE,
                lambda t, f: (_ for _ in ()).throw(
                    RuntimeError("boom")
                ),
            ),
        ),
        fixture_factory=lambda t: object(),
    )

    assert result.status is BenchmarkStatus.FAILED


def test_runner_propagates_error_when_configured() -> None:
    runner = BenchmarkRunner(
        BenchmarkConfig(
            repetitions=1,
            capture_method_errors=False,
        )
    )
    with pytest.raises(RuntimeError, match="boom"):
        runner.run(
            (spec(),),
            (
                BenchmarkMethod(
                    "roif",
                    "ROIF",
                    BenchmarkMethodKind.ROIF,
                    lambda t, f: (_ for _ in ()).throw(
                        RuntimeError("boom")
                    ),
                ),
            ),
            fixture_factory=lambda t: object(),
        )


def test_runner_rejects_duplicate_method_ids() -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkRunner(
            BenchmarkConfig(repetitions=1)
        ).run(
            (spec(),),
            (method("roif"), method("roif")),
            fixture_factory=lambda t: object(),
        )


def test_runner_requires_reference_method() -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkRunner(
            BenchmarkConfig(
                repetitions=1,
                reference_method_id="roif",
            )
        ).run(
            (spec(),),
            (
                method(
                    "degree",
                    kind=BenchmarkMethodKind.BASELINE,
                ),
            ),
            fixture_factory=lambda t: object(),
        )


def test_run_benchmark_wrapper() -> None:
    result = run_benchmark(
        (spec(),),
        (
            method("roif"),
            method(
                "degree",
                kind=BenchmarkMethodKind.BASELINE,
            ),
        ),
        config=BenchmarkConfig(
            repetitions=1,
            measure_runtime=False,
        ),
        fixture_factory=lambda t: object(),
        true_root_factory=lambda s, r, seed: "A",
        true_node_star_factory=lambda s, r, seed: "B",
        metadata={"experiment": "E1"},
    )

    assert isinstance(result, BenchmarkResult)
    assert result.status is BenchmarkStatus.COMPLETED
    assert result.metadata["experiment"] == "E1"
    assert "Benchmark completed" in result.summary


def test_result_serialization() -> None:
    result = run_benchmark(
        (spec(),),
        (
            method("roif"),
            method(
                "degree",
                kind=BenchmarkMethodKind.BASELINE,
            ),
        ),
        config=BenchmarkConfig(
            repetitions=1,
            measure_runtime=False,
        ),
        fixture_factory=lambda t: object(),
        true_root_factory=lambda s, r, seed: "A",
        true_node_star_factory=lambda s, r, seed: "B",
    )

    payload = result.as_dict()
    assert payload["status"] == "completed"
    assert payload["successful_run_count"] == 2
    assert payload["methods"][0]["method_id"] == "roif"


