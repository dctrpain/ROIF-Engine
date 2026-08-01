"""
Uncertainty quantification for ROIF Engine v1.

This module provides a domain-independent uncertainty layer for analytical,
localization, prediction, optimization, and reporting results.

It supports:

- point estimates with standard deviation and confidence intervals;
- deterministic or sample-based uncertainty summaries;
- weighted evidence aggregation;
- uncertainty propagation for sums, differences, products, ratios,
  and arbitrary scalar functions;
- Monte Carlo propagation with reproducible random seeds;
- confidence grading;
- calibration diagnostics;
- immutable, serializable result objects.

The module deliberately does not prescribe a probabilistic interpretation for
all ROIF quantities. It only provides reusable numerical structures and
operations. Domain-specific meaning remains the responsibility of the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable
import math

import numpy as np


class UncertaintyError(ValueError):
    """Raised when uncertainty input or configuration is invalid."""


ScalarFunction: TypeAlias = Callable[[np.ndarray], float]
SampleGenerator: TypeAlias = Callable[[np.random.Generator, int], np.ndarray]

_EPSILON = 1e-12


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise UncertaintyError(f"{name} must be a real number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UncertaintyError(f"{name} must be a real number.") from exc

    if not math.isfinite(result):
        raise UncertaintyError(f"{name} must be finite.")

    if minimum is not None and result < minimum:
        raise UncertaintyError(
            f"{name} must be greater than or equal to {minimum}."
        )

    if maximum is not None and result > maximum:
        raise UncertaintyError(
            f"{name} must be less than or equal to {maximum}."
        )

    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UncertaintyError(
            f"{name} must be an integer greater than or equal to 1."
        )
    return value


def _normalize_text(
    value: Any,
    *,
    name: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None

    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise UncertaintyError(f"{name} must be a string{suffix}.")

    normalized = value.strip()
    if not normalized:
        raise UncertaintyError(f"{name} cannot be empty.")

    return normalized


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise UncertaintyError("metadata must be a mapping.")

    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized = _normalize_text(key, name="metadata key")
        copied[normalized] = value

    return MappingProxyType(copied)


def _sample_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    minimum_size: int = 1,
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise UncertaintyError(
            f"{name} must be a one-dimensional numeric sequence."
        )

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise UncertaintyError(
            f"{name} must be a one-dimensional numeric sequence."
        ) from exc

    if array.ndim != 1:
        raise UncertaintyError(f"{name} must be one-dimensional.")

    if array.size < minimum_size:
        raise UncertaintyError(
            f"{name} must contain at least {minimum_size} values."
        )

    if not np.all(np.isfinite(array)):
        raise UncertaintyError(
            f"{name} must contain only finite values."
        )

    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _weights_vector(
    values: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
) -> np.ndarray:
    weights = _sample_vector(
        values,
        name="weights",
        minimum_size=expected_size,
    )
    if weights.size != expected_size:
        raise UncertaintyError(
            f"weights has size {weights.size}; expected {expected_size}."
        )
    if np.any(weights < 0.0):
        raise UncertaintyError("weights cannot contain negative values.")
    if not np.any(weights > 0.0):
        raise UncertaintyError(
            "weights must contain at least one positive value."
        )

    normalized = np.array(weights / np.sum(weights), copy=True)
    normalized.setflags(write=False)
    return normalized


class ConfidenceGrade(str, Enum):
    """Qualitative confidence grade."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class IntervalMethod(str, Enum):
    """Method used to construct an uncertainty interval."""

    NORMAL = "normal"
    EMPIRICAL = "empirical"
    BOOTSTRAP = "bootstrap"
    PROVIDED = "provided"


class PropagationMethod(str, Enum):
    """Method used to propagate uncertainty."""

    ANALYTIC = "analytic"
    MONTE_CARLO = "monte_carlo"
    CUSTOM = "custom"


class CalibrationStatus(str, Enum):
    """Interpretation of empirical interval coverage."""

    UNDERCONFIDENT = "underconfident"
    CALIBRATED = "calibrated"
    OVERCONFIDENT = "overconfident"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class UncertaintyConfig:
    """Configuration for uncertainty estimation and propagation."""

    confidence_level: float = 0.95
    sample_count: int = 10_000
    bootstrap_count: int = 2_000
    seed: int | None = 0
    interval_method: IntervalMethod = IntervalMethod.EMPIRICAL
    minimum_standard_deviation: float = 0.0
    calibration_tolerance: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_level",
            _finite_float(
                self.confidence_level,
                name="confidence_level",
                minimum=_EPSILON,
                maximum=1.0 - _EPSILON,
            ),
        )
        object.__setattr__(
            self,
            "sample_count",
            _positive_int(self.sample_count, name="sample_count"),
        )
        object.__setattr__(
            self,
            "bootstrap_count",
            _positive_int(
                self.bootstrap_count,
                name="bootstrap_count",
            ),
        )

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise UncertaintyError("seed must be an integer or None.")

        if not isinstance(self.interval_method, IntervalMethod):
            raise UncertaintyError(
                "interval_method must be an IntervalMethod."
            )

        object.__setattr__(
            self,
            "minimum_standard_deviation",
            _finite_float(
                self.minimum_standard_deviation,
                name="minimum_standard_deviation",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "calibration_tolerance",
            _finite_float(
                self.calibration_tolerance,
                name="calibration_tolerance",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Closed confidence interval."""

    lower: float
    upper: float
    confidence_level: float
    method: IntervalMethod = IntervalMethod.PROVIDED

    def __post_init__(self) -> None:
        lower = _finite_float(self.lower, name="lower")
        upper = _finite_float(self.upper, name="upper")
        if lower > upper:
            raise UncertaintyError(
                "lower cannot be greater than upper."
            )

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self,
            "confidence_level",
            _finite_float(
                self.confidence_level,
                name="confidence_level",
                minimum=_EPSILON,
                maximum=1.0 - _EPSILON,
            ),
        )

        if not isinstance(self.method, IntervalMethod):
            raise UncertaintyError(
                "method must be an IntervalMethod."
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    def contains(self, value: float) -> bool:
        numeric = _finite_float(value, name="value")
        return self.lower <= numeric <= self.upper

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "midpoint": self.midpoint,
            "confidence_level": self.confidence_level,
            "method": self.method.value,
        }


@dataclass(frozen=True, slots=True)
class UncertainValue:
    """Scalar estimate with uncertainty information."""

    estimate: float
    standard_deviation: float
    interval: ConfidenceInterval
    sample_size: int
    grade: ConfidenceGrade
    label: str | None = None
    units: str | None = None
    method: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "estimate",
            _finite_float(self.estimate, name="estimate"),
        )
        object.__setattr__(
            self,
            "standard_deviation",
            _finite_float(
                self.standard_deviation,
                name="standard_deviation",
                minimum=0.0,
            ),
        )

        if not isinstance(self.interval, ConfidenceInterval):
            raise UncertaintyError(
                "interval must be a ConfidenceInterval."
            )

        object.__setattr__(
            self,
            "sample_size",
            _positive_int(self.sample_size, name="sample_size"),
        )

        if not isinstance(self.grade, ConfidenceGrade):
            raise UncertaintyError(
                "grade must be a ConfidenceGrade."
            )

        for name in ("label", "units", "method"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _normalize_text(
                        value,
                        name=name,
                        optional=True,
                    ),
                )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def variance(self) -> float:
        return self.standard_deviation ** 2

    @property
    def relative_uncertainty(self) -> float | None:
        if abs(self.estimate) <= _EPSILON:
            return None
        return self.standard_deviation / abs(self.estimate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "standard_deviation": self.standard_deviation,
            "variance": self.variance,
            "relative_uncertainty": self.relative_uncertainty,
            "interval": self.interval.as_dict(),
            "sample_size": self.sample_size,
            "grade": self.grade.value,
            "label": self.label,
            "units": self.units,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One weighted piece of evidence."""

    estimate: float
    standard_deviation: float
    weight: float = 1.0
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "estimate",
            _finite_float(self.estimate, name="estimate"),
        )
        object.__setattr__(
            self,
            "standard_deviation",
            _finite_float(
                self.standard_deviation,
                name="standard_deviation",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "weight",
            _finite_float(
                self.weight,
                name="weight",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_text(
                self.source,
                name="source",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Result of uncertainty propagation."""

    value: UncertainValue
    method: PropagationMethod
    input_count: int
    sample_count: int | None
    seed: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, UncertainValue):
            raise UncertaintyError(
                "value must be an UncertainValue."
            )

        if not isinstance(self.method, PropagationMethod):
            raise UncertaintyError(
                "method must be a PropagationMethod."
            )

        object.__setattr__(
            self,
            "input_count",
            _positive_int(self.input_count, name="input_count"),
        )

        if self.sample_count is not None:
            object.__setattr__(
                self,
                "sample_count",
                _positive_int(
                    self.sample_count,
                    name="sample_count",
                ),
            )

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise UncertaintyError("seed must be an integer or None.")

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value.as_dict(),
            "method": self.method.value,
            "input_count": self.input_count,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Empirical confidence-interval calibration summary."""

    status: CalibrationStatus
    expected_coverage: float
    observed_coverage: float | None
    calibration_error: float | None
    sample_count: int
    covered_count: int
    tolerance: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, CalibrationStatus):
            raise UncertaintyError(
                "status must be a CalibrationStatus."
            )

        object.__setattr__(
            self,
            "expected_coverage",
            _finite_float(
                self.expected_coverage,
                name="expected_coverage",
                minimum=0.0,
                maximum=1.0,
            ),
        )

        for name in ("observed_coverage", "calibration_error"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_float(
                        value,
                        name=name,
                        minimum=0.0 if name == "observed_coverage" else None,
                        maximum=1.0 if name == "observed_coverage" else None,
                    ),
                )

        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise UncertaintyError(
                "sample_count must be a non-negative integer."
            )

        if (
            isinstance(self.covered_count, bool)
            or not isinstance(self.covered_count, int)
            or self.covered_count < 0
            or self.covered_count > self.sample_count
        ):
            raise UncertaintyError(
                "covered_count must be between zero and sample_count."
            )

        object.__setattr__(
            self,
            "tolerance",
            _finite_float(
                self.tolerance,
                name="tolerance",
                minimum=0.0,
                maximum=1.0,
            ),
        )

        if self.status is CalibrationStatus.INSUFFICIENT_DATA:
            if (
                self.observed_coverage is not None
                or self.calibration_error is not None
            ):
                raise UncertaintyError(
                    "Insufficient calibration cannot contain coverage values."
                )
        else:
            if (
                self.observed_coverage is None
                or self.calibration_error is None
            ):
                raise UncertaintyError(
                    "Calibration result requires coverage values."
                )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "expected_coverage": self.expected_coverage,
            "observed_coverage": self.observed_coverage,
            "calibration_error": self.calibration_error,
            "sample_count": self.sample_count,
            "covered_count": self.covered_count,
            "tolerance": self.tolerance,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class DistributionSampler(Protocol):
    """Callable sampler contract for Monte Carlo propagation."""

    def __call__(
        self,
        rng: np.random.Generator,
        sample_count: int,
    ) -> np.ndarray:
        ...


def confidence_grade(
    *,
    standard_deviation: float,
    estimate: float,
    sample_size: int,
) -> ConfidenceGrade:
    """Convert numerical uncertainty into a qualitative grade."""

    sd = _finite_float(
        standard_deviation,
        name="standard_deviation",
        minimum=0.0,
    )
    point = _finite_float(estimate, name="estimate")
    count = _positive_int(sample_size, name="sample_size")

    if abs(point) <= _EPSILON:
        relative = math.inf if sd > _EPSILON else 0.0
    else:
        relative = sd / abs(point)

    sample_factor = min(1.0, math.log10(count + 1.0) / 3.0)

    if relative <= 0.02 and sample_factor >= 0.8:
        return ConfidenceGrade.VERY_HIGH
    if relative <= 0.05 and sample_factor >= 0.6:
        return ConfidenceGrade.HIGH
    if relative <= 0.15 and sample_factor >= 0.4:
        return ConfidenceGrade.MODERATE
    if relative <= 0.35:
        return ConfidenceGrade.LOW
    return ConfidenceGrade.VERY_LOW


def _normal_quantile(confidence_level: float) -> float:
    # Acklam approximation is unnecessary here; for common confidence levels
    # linear interpolation over standard z-values is accurate enough.
    levels = np.asarray(
        (0.80, 0.90, 0.95, 0.98, 0.99, 0.995),
        dtype=float,
    )
    z_values = np.asarray(
        (1.281551566, 1.644853627, 1.959963985, 2.326347874,
         2.575829304, 2.807033768),
        dtype=float,
    )
    return float(np.interp(confidence_level, levels, z_values))


def interval_from_mean_sd(
    mean: float,
    standard_deviation: float,
    *,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Construct a symmetric normal interval."""

    point = _finite_float(mean, name="mean")
    sd = _finite_float(
        standard_deviation,
        name="standard_deviation",
        minimum=0.0,
    )
    level = _finite_float(
        confidence_level,
        name="confidence_level",
        minimum=_EPSILON,
        maximum=1.0 - _EPSILON,
    )

    z = _normal_quantile(level)
    margin = z * sd
    return ConfidenceInterval(
        lower=point - margin,
        upper=point + margin,
        confidence_level=level,
        method=IntervalMethod.NORMAL,
    )


def summarize_samples(
    samples: Sequence[float] | np.ndarray,
    *,
    config: UncertaintyConfig | None = None,
    label: str | None = None,
    units: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> UncertainValue:
    """Create an uncertainty summary from scalar samples."""

    if config is None:
        config = UncertaintyConfig()

    if not isinstance(config, UncertaintyConfig):
        raise UncertaintyError(
            "config must be an UncertaintyConfig."
        )

    values = _sample_vector(
        samples,
        name="samples",
        minimum_size=1,
    )
    estimate = float(np.mean(values))
    sd = (
        float(np.std(values, ddof=1))
        if values.size > 1
        else config.minimum_standard_deviation
    )
    sd = max(sd, config.minimum_standard_deviation)

    alpha = 1.0 - config.confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    if config.interval_method is IntervalMethod.NORMAL:
        interval = interval_from_mean_sd(
            estimate,
            sd,
            confidence_level=config.confidence_level,
        )

    elif config.interval_method is IntervalMethod.EMPIRICAL:
        interval = ConfidenceInterval(
            lower=float(np.quantile(values, lower_q)),
            upper=float(np.quantile(values, upper_q)),
            confidence_level=config.confidence_level,
            method=IntervalMethod.EMPIRICAL,
        )

    elif config.interval_method is IntervalMethod.BOOTSTRAP:
        rng = np.random.default_rng(config.seed)
        means = np.empty(config.bootstrap_count, dtype=float)
        for index in range(config.bootstrap_count):
            draw = rng.choice(
                values,
                size=values.size,
                replace=True,
            )
            means[index] = float(np.mean(draw))

        interval = ConfidenceInterval(
            lower=float(np.quantile(means, lower_q)),
            upper=float(np.quantile(means, upper_q)),
            confidence_level=config.confidence_level,
            method=IntervalMethod.BOOTSTRAP,
        )

    else:
        raise UncertaintyError(
            "PROVIDED interval method cannot summarize raw samples."
        )

    return UncertainValue(
        estimate=estimate,
        standard_deviation=sd,
        interval=interval,
        sample_size=int(values.size),
        grade=confidence_grade(
            standard_deviation=sd,
            estimate=estimate,
            sample_size=int(values.size),
        ),
        label=label,
        units=units,
        method=f"sample_summary:{config.interval_method.value}",
        metadata=metadata,
    )


def uncertain_value(
    estimate: float,
    standard_deviation: float,
    *,
    confidence_level: float = 0.95,
    sample_size: int = 1,
    label: str | None = None,
    units: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> UncertainValue:
    """Construct an uncertain scalar from estimate and standard deviation."""

    point = _finite_float(estimate, name="estimate")
    sd = _finite_float(
        standard_deviation,
        name="standard_deviation",
        minimum=0.0,
    )
    count = _positive_int(sample_size, name="sample_size")

    return UncertainValue(
        estimate=point,
        standard_deviation=sd,
        interval=interval_from_mean_sd(
            point,
            sd,
            confidence_level=confidence_level,
        ),
        sample_size=count,
        grade=confidence_grade(
            standard_deviation=sd,
            estimate=point,
            sample_size=count,
        ),
        label=label,
        units=units,
        method="provided_mean_sd",
        metadata=metadata,
    )


def combine_evidence(
    items: Iterable[EvidenceItem],
    *,
    confidence_level: float = 0.95,
    metadata: Mapping[str, Any] | None = None,
) -> UncertainValue:
    """Combine independent weighted estimates using precision weighting."""

    if isinstance(items, (str, bytes)):
        raise UncertaintyError(
            "items must be an iterable of EvidenceItem values."
        )

    try:
        evidence = tuple(items)
    except TypeError as exc:
        raise UncertaintyError(
            "items must be an iterable of EvidenceItem values."
        ) from exc

    if not evidence:
        raise UncertaintyError("items cannot be empty.")

    if any(not isinstance(item, EvidenceItem) for item in evidence):
        raise UncertaintyError(
            "items must contain EvidenceItem values."
        )

    precision_weights: list[float] = []
    estimates: list[float] = []

    for item in evidence:
        variance = max(item.standard_deviation ** 2, _EPSILON)
        precision_weights.append(item.weight / variance)
        estimates.append(item.estimate)

    total_precision = float(np.sum(precision_weights))
    if total_precision <= _EPSILON:
        raise UncertaintyError(
            "Combined evidence has zero total precision."
        )

    normalized = np.asarray(precision_weights) / total_precision
    estimate = float(np.dot(normalized, np.asarray(estimates)))
    sd = math.sqrt(1.0 / total_precision)

    return uncertain_value(
        estimate,
        sd,
        confidence_level=confidence_level,
        sample_size=len(evidence),
        label="combined_evidence",
        metadata={
            "source_count": len(evidence),
            **({} if metadata is None else dict(metadata)),
        },
    )


def propagate_sum(
    values: Sequence[UncertainValue],
    *,
    coefficients: Sequence[float] | None = None,
    confidence_level: float = 0.95,
) -> PropagationResult:
    """Analytically propagate independent uncertainty through a linear sum."""

    items = tuple(values)
    if not items:
        raise UncertaintyError("values cannot be empty.")
    if any(not isinstance(item, UncertainValue) for item in items):
        raise UncertaintyError(
            "values must contain UncertainValue objects."
        )

    if coefficients is None:
        coeff = np.ones(len(items), dtype=float)
    else:
        coeff = _sample_vector(
            coefficients,
            name="coefficients",
            minimum_size=len(items),
        )
        if coeff.size != len(items):
            raise UncertaintyError(
                "coefficients must match values length."
            )

    means = np.asarray([item.estimate for item in items])
    variances = np.asarray([item.variance for item in items])

    estimate = float(np.dot(coeff, means))
    variance = float(np.dot(coeff * coeff, variances))
    sd = math.sqrt(max(variance, 0.0))

    return PropagationResult(
        value=uncertain_value(
            estimate,
            sd,
            confidence_level=confidence_level,
            sample_size=min(item.sample_size for item in items),
            label="propagated_sum",
        ),
        method=PropagationMethod.ANALYTIC,
        input_count=len(items),
        sample_count=None,
        seed=None,
    )


def propagate_difference(
    left: UncertainValue,
    right: UncertainValue,
    *,
    confidence_level: float = 0.95,
) -> PropagationResult:
    """Analytically propagate independent uncertainty through subtraction."""

    return propagate_sum(
        (left, right),
        coefficients=(1.0, -1.0),
        confidence_level=confidence_level,
    )


def propagate_product(
    values: Sequence[UncertainValue],
    *,
    confidence_level: float = 0.95,
) -> PropagationResult:
    """First-order analytic propagation for a product of independent values."""

    items = tuple(values)
    if not items:
        raise UncertaintyError("values cannot be empty.")
    if any(not isinstance(item, UncertainValue) for item in items):
        raise UncertaintyError(
            "values must contain UncertainValue objects."
        )

    estimate = float(np.prod([item.estimate for item in items]))

    relative_variance = 0.0
    for item in items:
        if abs(item.estimate) <= _EPSILON:
            # Fall back to direct derivative formula.
            derivative = 1.0
            for other in items:
                if other is not item:
                    derivative *= other.estimate
            relative_variance += (derivative * item.standard_deviation) ** 2
        else:
            relative_variance += (
                item.standard_deviation / item.estimate
            ) ** 2

    if all(abs(item.estimate) > _EPSILON for item in items):
        sd = abs(estimate) * math.sqrt(relative_variance)
    else:
        sd = math.sqrt(relative_variance)

    return PropagationResult(
        value=uncertain_value(
            estimate,
            sd,
            confidence_level=confidence_level,
            sample_size=min(item.sample_size for item in items),
            label="propagated_product",
        ),
        method=PropagationMethod.ANALYTIC,
        input_count=len(items),
        sample_count=None,
        seed=None,
    )


def propagate_ratio(
    numerator: UncertainValue,
    denominator: UncertainValue,
    *,
    confidence_level: float = 0.95,
) -> PropagationResult:
    """First-order analytic propagation for a ratio."""

    if not isinstance(numerator, UncertainValue):
        raise UncertaintyError(
            "numerator must be an UncertainValue."
        )
    if not isinstance(denominator, UncertainValue):
        raise UncertaintyError(
            "denominator must be an UncertainValue."
        )
    if abs(denominator.estimate) <= _EPSILON:
        raise UncertaintyError(
            "denominator estimate cannot be zero."
        )

    estimate = numerator.estimate / denominator.estimate
    relative_variance = 0.0

    if abs(numerator.estimate) > _EPSILON:
        relative_variance += (
            numerator.standard_deviation / numerator.estimate
        ) ** 2
    else:
        relative_variance += (
            numerator.standard_deviation / denominator.estimate
        ) ** 2

    relative_variance += (
        denominator.standard_deviation / denominator.estimate
    ) ** 2

    sd = abs(estimate) * math.sqrt(relative_variance)

    return PropagationResult(
        value=uncertain_value(
            estimate,
            sd,
            confidence_level=confidence_level,
            sample_size=min(
                numerator.sample_size,
                denominator.sample_size,
            ),
            label="propagated_ratio",
        ),
        method=PropagationMethod.ANALYTIC,
        input_count=2,
        sample_count=None,
        seed=None,
    )


def monte_carlo_propagate(
    samplers: Sequence[DistributionSampler | SampleGenerator],
    function: ScalarFunction,
    *,
    config: UncertaintyConfig | None = None,
    label: str | None = None,
    units: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PropagationResult:
    """Propagate uncertainty through an arbitrary scalar function."""

    if config is None:
        config = UncertaintyConfig()

    if not isinstance(config, UncertaintyConfig):
        raise UncertaintyError(
            "config must be an UncertaintyConfig."
        )

    sampler_items = tuple(samplers)
    if not sampler_items:
        raise UncertaintyError("samplers cannot be empty.")
    if any(not callable(item) for item in sampler_items):
        raise UncertaintyError("all samplers must be callable.")
    if not callable(function):
        raise UncertaintyError("function must be callable.")

    rng = np.random.default_rng(config.seed)
    columns: list[np.ndarray] = []

    for index, sampler in enumerate(sampler_items):
        raw = sampler(rng, config.sample_count)
        values = _sample_vector(
            raw,
            name=f"sampler[{index}] output",
            minimum_size=config.sample_count,
        )
        if values.size != config.sample_count:
            raise UncertaintyError(
                f"sampler[{index}] returned {values.size} values; "
                f"expected {config.sample_count}."
            )
        columns.append(values)

    matrix = np.column_stack(columns)
    outputs = np.empty(config.sample_count, dtype=float)

    for index, row in enumerate(matrix):
        outputs[index] = _finite_float(
            function(row),
            name="propagated function output",
        )

    value = summarize_samples(
        outputs,
        config=config,
        label=label or "monte_carlo_propagation",
        units=units,
        metadata=metadata,
    )

    return PropagationResult(
        value=value,
        method=PropagationMethod.MONTE_CARLO,
        input_count=len(sampler_items),
        sample_count=config.sample_count,
        seed=config.seed,
        metadata=metadata,
    )


def normal_sampler(
    mean: float,
    standard_deviation: float,
) -> SampleGenerator:
    """Create a reproducible normal-distribution sampler."""

    point = _finite_float(mean, name="mean")
    sd = _finite_float(
        standard_deviation,
        name="standard_deviation",
        minimum=0.0,
    )

    def sample(
        rng: np.random.Generator,
        sample_count: int,
    ) -> np.ndarray:
        return rng.normal(point, sd, size=sample_count)

    return sample


def uniform_sampler(
    lower: float,
    upper: float,
) -> SampleGenerator:
    """Create a uniform-distribution sampler."""

    low = _finite_float(lower, name="lower")
    high = _finite_float(upper, name="upper")
    if low > high:
        raise UncertaintyError(
            "lower cannot be greater than upper."
        )

    def sample(
        rng: np.random.Generator,
        sample_count: int,
    ) -> np.ndarray:
        return rng.uniform(low, high, size=sample_count)

    return sample


def calibrate_intervals(
    intervals: Sequence[ConfidenceInterval],
    observations: Sequence[float],
    *,
    tolerance: float = 0.05,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Evaluate empirical coverage of predicted confidence intervals."""

    interval_items = tuple(intervals)
    observed = _sample_vector(
        observations,
        name="observations",
        minimum_size=1,
    )

    if len(interval_items) != observed.size:
        raise UncertaintyError(
            "intervals and observations must have equal length."
        )

    if any(
        not isinstance(item, ConfidenceInterval)
        for item in interval_items
    ):
        raise UncertaintyError(
            "intervals must contain ConfidenceInterval values."
        )

    allowed_error = _finite_float(
        tolerance,
        name="tolerance",
        minimum=0.0,
        maximum=1.0,
    )

    if not interval_items:
        return CalibrationResult(
            status=CalibrationStatus.INSUFFICIENT_DATA,
            expected_coverage=0.0,
            observed_coverage=None,
            calibration_error=None,
            sample_count=0,
            covered_count=0,
            tolerance=allowed_error,
            metadata=metadata,
        )

    expected = float(
        np.mean([item.confidence_level for item in interval_items])
    )
    covered = sum(
        interval.contains(float(value))
        for interval, value in zip(interval_items, observed)
    )
    actual = covered / len(interval_items)
    error = actual - expected

    if abs(error) <= allowed_error:
        status = CalibrationStatus.CALIBRATED
    elif actual > expected:
        status = CalibrationStatus.UNDERCONFIDENT
    else:
        status = CalibrationStatus.OVERCONFIDENT

    return CalibrationResult(
        status=status,
        expected_coverage=expected,
        observed_coverage=actual,
        calibration_error=error,
        sample_count=len(interval_items),
        covered_count=covered,
        tolerance=allowed_error,
        metadata=metadata,
    )


__all__ = [
    "CalibrationResult",
    "CalibrationStatus",
    "ConfidenceGrade",
    "ConfidenceInterval",
    "DistributionSampler",
    "EvidenceItem",
    "IntervalMethod",
    "PropagationMethod",
    "PropagationResult",
    "SampleGenerator",
    "ScalarFunction",
    "UncertainValue",
    "UncertaintyConfig",
    "UncertaintyError",
    "calibrate_intervals",
    "combine_evidence",
    "confidence_grade",
    "interval_from_mean_sd",
    "monte_carlo_propagate",
    "normal_sampler",
    "propagate_difference",
    "propagate_product",
    "propagate_ratio",
    "propagate_sum",
    "summarize_samples",
    "uncertain_value",
    "uniform_sampler",
]
