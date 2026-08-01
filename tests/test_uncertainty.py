"""Tests for roif.uncertainty."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from roif.uncertainty import (
    CalibrationResult,
    CalibrationStatus,
    ConfidenceGrade,
    ConfidenceInterval,
    EvidenceItem,
    IntervalMethod,
    PropagationMethod,
    PropagationResult,
    UncertainValue,
    UncertaintyConfig,
    UncertaintyError,
    calibrate_intervals,
    combine_evidence,
    confidence_grade,
    interval_from_mean_sd,
    monte_carlo_propagate,
    normal_sampler,
    propagate_difference,
    propagate_product,
    propagate_ratio,
    propagate_sum,
    summarize_samples,
    uncertain_value,
    uniform_sampler,
)


def interval(
    lower: float = 0.8,
    upper: float = 1.2,
    level: float = 0.95,
) -> ConfidenceInterval:
    return ConfidenceInterval(lower, upper, level)


def value(
    estimate: float = 1.0,
    sd: float = 0.1,
    size: int = 100,
) -> UncertainValue:
    return UncertainValue(
        estimate=estimate,
        standard_deviation=sd,
        interval=interval(),
        sample_size=size,
        grade=ConfidenceGrade.MODERATE,
        label="eta",
        units="a.u.",
        method="test",
        metadata={"source": "test"},
    )


def evidence(
    estimate: float = 1.0,
    sd: float = 0.1,
    weight: float = 1.0,
) -> EvidenceItem:
    return EvidenceItem(
        estimate=estimate,
        standard_deviation=sd,
        weight=weight,
        source="study",
        metadata={"source": "test"},
    )


def test_enum_values() -> None:
    assert tuple(x.value for x in ConfidenceGrade) == (
        "very_low", "low", "moderate", "high", "very_high"
    )
    assert tuple(x.value for x in IntervalMethod) == (
        "normal", "empirical", "bootstrap", "provided"
    )
    assert tuple(x.value for x in PropagationMethod) == (
        "analytic", "monte_carlo", "custom"
    )
    assert tuple(x.value for x in CalibrationStatus) == (
        "underconfident", "calibrated", "overconfident",
        "insufficient_data",
    )


def test_config_defaults_and_normalization() -> None:
    config = UncertaintyConfig()
    assert config.confidence_level == pytest.approx(0.95)
    assert config.sample_count == 10_000
    assert config.bootstrap_count == 2_000
    assert config.seed == 0
    assert config.interval_method is IntervalMethod.EMPIRICAL

    normalized = UncertaintyConfig(
        confidence_level="0.9",
        minimum_standard_deviation="0.01",
        calibration_tolerance="0.1",
    )
    assert normalized.confidence_level == pytest.approx(0.9)
    assert normalized.minimum_standard_deviation == pytest.approx(0.01)


def test_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        UncertaintyConfig().sample_count = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "value_", (0.0, 1.0, -0.1, 1.1, float("nan"), True, "bad")
)
def test_config_rejects_bad_confidence(value_: Any) -> None:
    with pytest.raises(UncertaintyError):
        UncertaintyConfig(confidence_level=value_)


@pytest.mark.parametrize("field", ("sample_count", "bootstrap_count"))
@pytest.mark.parametrize("value_", (0, -1, True, 1.5, "10"))
def test_config_rejects_bad_counts(field: str, value_: Any) -> None:
    with pytest.raises(UncertaintyError):
        UncertaintyConfig(**{field: value_})


@pytest.mark.parametrize("value_", (True, 1.5, "seed"))
def test_config_rejects_bad_seed(value_: Any) -> None:
    with pytest.raises(UncertaintyError):
        UncertaintyConfig(seed=value_)


@pytest.mark.parametrize("value_", ("empirical", None, 1))
def test_config_rejects_bad_method(value_: Any) -> None:
    with pytest.raises(UncertaintyError):
        UncertaintyConfig(interval_method=value_)


@pytest.mark.parametrize(
    "field",
    ("minimum_standard_deviation", "calibration_tolerance"),
)
@pytest.mark.parametrize(
    "value_", (-0.1, float("nan"), float("inf"), True, "bad")
)
def test_config_rejects_bad_nonnegative_values(
    field: str,
    value_: Any,
) -> None:
    with pytest.raises(UncertaintyError):
        UncertaintyConfig(**{field: value_})


def test_interval_properties_and_serialization() -> None:
    item = interval()
    assert item.width == pytest.approx(0.4)
    assert item.midpoint == pytest.approx(1.0)
    assert item.contains(1.0)
    assert not item.contains(1.3)
    assert item.as_dict()["method"] == "provided"


def test_interval_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        interval().lower = 0.0  # type: ignore[misc]


def test_interval_rejects_reversed_bounds() -> None:
    with pytest.raises(UncertaintyError):
        ConfidenceInterval(2.0, 1.0, 0.95)


@pytest.mark.parametrize(
    "value_", (float("nan"), float("inf"), True, "bad")
)
def test_interval_rejects_bad_bounds(value_: Any) -> None:
    with pytest.raises(UncertaintyError):
        ConfidenceInterval(value_, 1.0, 0.95)


@pytest.mark.parametrize(
    "value_", (0.0, 1.0, -0.1, 1.1, float("nan"), True, "bad")
)
def test_interval_rejects_bad_level(value_: Any) -> None:
    with pytest.raises(UncertaintyError):
        ConfidenceInterval(0.0, 1.0, value_)


def test_uncertain_value_properties() -> None:
    item = value()
    assert item.variance == pytest.approx(0.01)
    assert item.relative_uncertainty == pytest.approx(0.1)
    assert isinstance(item.metadata, MappingProxyType)
    assert item.as_dict()["grade"] == "moderate"


def test_zero_estimate_relative_uncertainty_is_none() -> None:
    assert value(estimate=0.0).relative_uncertainty is None


@pytest.mark.parametrize(
    "field", ("estimate", "standard_deviation")
)
@pytest.mark.parametrize(
    "value_", (float("nan"), float("inf"), True, "bad")
)
def test_uncertain_value_rejects_bad_numbers(
    field: str,
    value_: Any,
) -> None:
    kwargs = {
        "estimate": 1.0,
        "standard_deviation": 0.1,
        "interval": interval(),
        "sample_size": 10,
        "grade": ConfidenceGrade.MODERATE,
    }
    kwargs[field] = value_
    with pytest.raises(UncertaintyError):
        UncertainValue(**kwargs)


def test_uncertain_value_rejects_negative_sd() -> None:
    with pytest.raises(UncertaintyError):
        value(sd=-0.1)


@pytest.mark.parametrize("value_", (0, -1, True, 1.5, "10"))
def test_uncertain_value_rejects_bad_sample_size(value_: Any) -> None:
    with pytest.raises(UncertaintyError):
        value(size=value_)


def test_evidence_item() -> None:
    item = evidence()
    assert item.weight == pytest.approx(1.0)
    assert isinstance(item.metadata, MappingProxyType)


@pytest.mark.parametrize(
    "field", ("estimate", "standard_deviation", "weight")
)
@pytest.mark.parametrize(
    "value_", (float("nan"), float("inf"), True, "bad")
)
def test_evidence_rejects_bad_numbers(
    field: str,
    value_: Any,
) -> None:
    kwargs = {"estimate": 1.0, "sd": 0.1, "weight": 1.0}
    kwargs[field if field != "standard_deviation" else "sd"] = value_
    with pytest.raises(UncertaintyError):
        evidence(**kwargs)


def test_propagation_result() -> None:
    result = PropagationResult(
        value=value(),
        method=PropagationMethod.ANALYTIC,
        input_count=2,
        sample_count=None,
        seed=None,
        metadata={"x": 1},
    )
    assert result.as_dict()["method"] == "analytic"
    assert isinstance(result.metadata, MappingProxyType)


@pytest.mark.parametrize("value_", (0, -1, True, 1.5, "10"))
def test_propagation_result_rejects_bad_input_count(
    value_: Any,
) -> None:
    with pytest.raises(UncertaintyError):
        PropagationResult(
            value=value(),
            method=PropagationMethod.ANALYTIC,
            input_count=value_,
            sample_count=None,
            seed=None,
        )


def test_calibration_result() -> None:
    result = CalibrationResult(
        status=CalibrationStatus.CALIBRATED,
        expected_coverage=0.95,
        observed_coverage=0.94,
        calibration_error=-0.01,
        sample_count=100,
        covered_count=94,
        tolerance=0.05,
    )
    assert result.as_dict()["status"] == "calibrated"


def test_insufficient_calibration() -> None:
    result = CalibrationResult(
        status=CalibrationStatus.INSUFFICIENT_DATA,
        expected_coverage=0.0,
        observed_coverage=None,
        calibration_error=None,
        sample_count=0,
        covered_count=0,
        tolerance=0.05,
    )
    assert result.observed_coverage is None


def test_insufficient_calibration_rejects_values() -> None:
    with pytest.raises(UncertaintyError):
        CalibrationResult(
            status=CalibrationStatus.INSUFFICIENT_DATA,
            expected_coverage=0.0,
            observed_coverage=0.0,
            calibration_error=0.0,
            sample_count=0,
            covered_count=0,
            tolerance=0.05,
        )


def test_confidence_grades() -> None:
    assert confidence_grade(
        standard_deviation=0.01, estimate=1.0, sample_size=1000
    ) is ConfidenceGrade.VERY_HIGH
    assert confidence_grade(
        standard_deviation=0.04, estimate=1.0, sample_size=100
    ) is ConfidenceGrade.HIGH
    assert confidence_grade(
        standard_deviation=0.1, estimate=1.0, sample_size=20
    ) is ConfidenceGrade.MODERATE
    assert confidence_grade(
        standard_deviation=0.3, estimate=1.0, sample_size=2
    ) is ConfidenceGrade.LOW
    assert confidence_grade(
        standard_deviation=1.0, estimate=1.0, sample_size=1
    ) is ConfidenceGrade.VERY_LOW


def test_interval_from_mean_sd() -> None:
    result = interval_from_mean_sd(10.0, 2.0)
    assert result.method is IntervalMethod.NORMAL
    assert result.midpoint == pytest.approx(10.0)
    assert result.lower == pytest.approx(10.0 - 1.959963985 * 2.0)


def test_summarize_samples_empirical() -> None:
    result = summarize_samples(
        (1.0, 2.0, 3.0, 4.0, 5.0),
        config=UncertaintyConfig(
            confidence_level=0.8,
            interval_method=IntervalMethod.EMPIRICAL,
        ),
        label="reserve",
    )
    assert result.estimate == pytest.approx(3.0)
    assert result.interval.method is IntervalMethod.EMPIRICAL
    assert result.sample_size == 5


def test_summarize_samples_normal() -> None:
    result = summarize_samples(
        (1.0, 2.0, 3.0),
        config=UncertaintyConfig(
            interval_method=IntervalMethod.NORMAL
        ),
    )
    assert result.interval.method is IntervalMethod.NORMAL


def test_summarize_samples_bootstrap_reproducible() -> None:
    config = UncertaintyConfig(
        interval_method=IntervalMethod.BOOTSTRAP,
        bootstrap_count=100,
        seed=42,
    )
    first = summarize_samples((1.0, 2.0, 3.0), config=config)
    second = summarize_samples((1.0, 2.0, 3.0), config=config)
    assert first.interval.lower == pytest.approx(second.interval.lower)
    assert first.interval.upper == pytest.approx(second.interval.upper)


def test_single_sample_uses_minimum_sd() -> None:
    result = summarize_samples(
        (2.0,),
        config=UncertaintyConfig(
            minimum_standard_deviation=0.25
        ),
    )
    assert result.standard_deviation == pytest.approx(0.25)


def test_summarize_rejects_provided_method() -> None:
    with pytest.raises(UncertaintyError):
        summarize_samples(
            (1.0, 2.0),
            config=UncertaintyConfig(
                interval_method=IntervalMethod.PROVIDED
            ),
        )


@pytest.mark.parametrize(
    "samples",
    ((), ((1.0, 2.0),), (1.0, float("nan")), "bad"),
)
def test_summarize_rejects_bad_samples(samples: Any) -> None:
    with pytest.raises(UncertaintyError):
        summarize_samples(samples)


def test_uncertain_value_factory() -> None:
    result = uncertain_value(
        2.0, 0.2, sample_size=50, label="gain"
    )
    assert result.method == "provided_mean_sd"
    assert result.estimate == pytest.approx(2.0)


def test_combine_evidence_precision_weighting() -> None:
    result = combine_evidence(
        (
            evidence(1.0, 0.1),
            evidence(2.0, 1.0),
        )
    )
    assert result.estimate == pytest.approx(102.0 / 101.0)
    assert result.standard_deviation == pytest.approx(
        np.sqrt(1.0 / 101.0)
    )


def test_combine_evidence_explicit_weights() -> None:
    result = combine_evidence(
        (
            evidence(1.0, 1.0, 1.0),
            evidence(3.0, 1.0, 3.0),
        )
    )
    assert result.estimate == pytest.approx(2.5)


@pytest.mark.parametrize("items", ((), "bad", (1,)))
def test_combine_evidence_rejects_bad_items(items: Any) -> None:
    with pytest.raises(UncertaintyError):
        combine_evidence(items)


def test_combine_evidence_rejects_zero_precision() -> None:
    with pytest.raises(UncertaintyError):
        combine_evidence(
            (evidence(weight=0.0), evidence(weight=0.0))
        )


def test_propagate_sum() -> None:
    result = propagate_sum(
        (
            uncertain_value(2.0, 0.3, sample_size=20),
            uncertain_value(3.0, 0.4, sample_size=10),
        )
    )
    assert result.value.estimate == pytest.approx(5.0)
    assert result.value.standard_deviation == pytest.approx(0.5)
    assert result.value.sample_size == 10


def test_propagate_weighted_sum_and_difference() -> None:
    left = uncertain_value(2.0, 0.3)
    right = uncertain_value(3.0, 0.4)
    weighted = propagate_sum(
        (left, right), coefficients=(2.0, -1.0)
    )
    assert weighted.value.estimate == pytest.approx(1.0)

    difference = propagate_difference(
        uncertain_value(5.0, 0.3),
        uncertain_value(2.0, 0.4),
    )
    assert difference.value.estimate == pytest.approx(3.0)
    assert difference.value.standard_deviation == pytest.approx(0.5)


def test_propagate_product() -> None:
    result = propagate_product(
        (
            uncertain_value(2.0, 0.2),
            uncertain_value(3.0, 0.3),
        )
    )
    assert result.value.estimate == pytest.approx(6.0)
    assert result.value.standard_deviation == pytest.approx(
        6.0 * np.sqrt(0.02)
    )


def test_propagate_product_with_zero() -> None:
    result = propagate_product(
        (
            uncertain_value(0.0, 0.2),
            uncertain_value(3.0, 0.3),
        )
    )
    assert result.value.estimate == pytest.approx(0.0)
    assert result.value.standard_deviation == pytest.approx(
    0.608276253029822
)


def test_propagate_ratio() -> None:
    result = propagate_ratio(
        uncertain_value(4.0, 0.4),
        uncertain_value(2.0, 0.2),
    )
    assert result.value.estimate == pytest.approx(2.0)
    assert result.value.standard_deviation == pytest.approx(
        2.0 * np.sqrt(0.02)
    )


def test_ratio_rejects_zero_denominator() -> None:
    with pytest.raises(UncertaintyError):
        propagate_ratio(
            uncertain_value(1.0, 0.1),
            uncertain_value(0.0, 0.1),
        )


@pytest.mark.parametrize("values", ((), (1,)))
def test_sum_and_product_reject_bad_values(values: Any) -> None:
    with pytest.raises(UncertaintyError):
        propagate_sum(values)
    with pytest.raises(UncertaintyError):
        propagate_product(values)


def test_sum_rejects_coefficient_mismatch() -> None:
    with pytest.raises(UncertaintyError):
        propagate_sum(
            (
                uncertain_value(1.0, 0.1),
                uncertain_value(2.0, 0.1),
            ),
            coefficients=(1.0,),
        )


def test_normal_and_uniform_samplers() -> None:
    rng = np.random.default_rng(42)
    normal = normal_sampler(2.0, 0.5)(rng, 50_000)
    assert np.mean(normal) == pytest.approx(2.0, abs=0.01)
    assert np.std(normal) == pytest.approx(0.5, abs=0.01)

    rng = np.random.default_rng(42)
    uniform = uniform_sampler(-1.0, 3.0)(rng, 50_000)
    assert np.mean(uniform) == pytest.approx(1.0, abs=0.02)
    assert np.min(uniform) >= -1.0
    assert np.max(uniform) <= 3.0


def test_uniform_rejects_reversed_bounds() -> None:
    with pytest.raises(UncertaintyError):
        uniform_sampler(2.0, 1.0)


def test_monte_carlo_is_reproducible() -> None:
    config = UncertaintyConfig(sample_count=2_000, seed=123)
    first = monte_carlo_propagate(
        (normal_sampler(1.0, 0.1), normal_sampler(2.0, 0.2)),
        lambda row: float(row[0] + row[1]),
        config=config,
    )
    second = monte_carlo_propagate(
        (normal_sampler(1.0, 0.1), normal_sampler(2.0, 0.2)),
        lambda row: float(row[0] + row[1]),
        config=config,
    )
    assert first.value.estimate == pytest.approx(second.value.estimate)
    assert first.value.interval.lower == pytest.approx(
        second.value.interval.lower
    )


def test_monte_carlo_matches_sum() -> None:
    result = monte_carlo_propagate(
        (normal_sampler(1.0, 0.1), normal_sampler(2.0, 0.2)),
        lambda row: float(row[0] + row[1]),
        config=UncertaintyConfig(
            sample_count=20_000,
            seed=42,
            interval_method=IntervalMethod.NORMAL,
        ),
    )
    assert result.value.estimate == pytest.approx(3.0, abs=0.02)
    assert result.value.standard_deviation == pytest.approx(
        np.sqrt(0.05), abs=0.01
    )
    assert result.method is PropagationMethod.MONTE_CARLO


@pytest.mark.parametrize("samplers", ((), (1,)))
def test_monte_carlo_rejects_bad_samplers(samplers: Any) -> None:
    with pytest.raises(UncertaintyError):
        monte_carlo_propagate(samplers, lambda row: 0.0)


def test_monte_carlo_rejects_wrong_sampler_length() -> None:
    with pytest.raises(UncertaintyError):
        monte_carlo_propagate(
            (lambda rng, count: np.zeros(count - 1),),
            lambda row: 0.0,
            config=UncertaintyConfig(sample_count=10),
        )


@pytest.mark.parametrize(
    "output", (float("nan"), float("inf"), True, "bad")
)
def test_monte_carlo_rejects_bad_function_output(
    output: Any,
) -> None:
    with pytest.raises(UncertaintyError):
        monte_carlo_propagate(
            (normal_sampler(0.0, 1.0),),
            lambda row: output,
            config=UncertaintyConfig(sample_count=10),
        )


def test_calibration_states() -> None:
    intervals = tuple(
        ConfidenceInterval(0.0, 1.0, 0.8)
        for _ in range(10)
    )

    calibrated = calibrate_intervals(
        intervals,
        (0.5,) * 8 + (2.0,) * 2,
        tolerance=0.01,
    )
    assert calibrated.status is CalibrationStatus.CALIBRATED

    under = calibrate_intervals(
        intervals,
        (0.5,) * 10,
        tolerance=0.01,
    )
    assert under.status is CalibrationStatus.UNDERCONFIDENT

    over = calibrate_intervals(
        intervals,
        (0.5,) * 2 + (2.0,) * 8,
        tolerance=0.01,
    )
    assert over.status is CalibrationStatus.OVERCONFIDENT


def test_calibration_rejects_length_mismatch() -> None:
    with pytest.raises(UncertaintyError):
        calibrate_intervals(
            (ConfidenceInterval(0.0, 1.0, 0.95),),
            (0.5, 0.6),
        )


def test_calibration_rejects_bad_interval_member() -> None:
    with pytest.raises(UncertaintyError):
        calibrate_intervals(("bad",), (0.5,))  # type: ignore[arg-type]
