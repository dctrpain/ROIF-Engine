from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ResidualDimension:
    """
    One empirically supported direction that remains outside the
    currently represented subspace.

    This object carries no semantic interpretation.  It does not say
    what the direction means, what caused it, or whether it should
    become a permanent internal coordinate.
    """

    residual_index: int
    eigenvalue: float
    explained_residual_fraction: float
    cumulative_residual_fraction: float
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.residual_index < 0:
            raise ValueError(
                "residual_index must be non-negative."
            )

        if not math.isfinite(self.eigenvalue):
            raise ValueError(
                "eigenvalue must be finite."
            )

        if self.eigenvalue < 0.0:
            raise ValueError(
                "eigenvalue must be non-negative."
            )

        for name, value in (
            (
                "explained_residual_fraction",
                self.explained_residual_fraction,
            ),
            (
                "cumulative_residual_fraction",
                self.cumulative_residual_fraction,
            ),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1]."
                )

        if not self.direction:
            raise ValueError(
                "direction must not be empty."
            )

        if not all(
            math.isfinite(value)
            for value in self.direction
        ):
            raise ValueError(
                "direction must contain only finite values."
            )


@dataclass(frozen=True, slots=True)
class DimensionalGrowthAssessment:
    """
    Result of asking whether the current representation leaves a
    structured, repeatable residual.

    `growth_supported` does NOT itself create a new dimension.

    It means only that the supplied experience contains residual
    structure which satisfies the frozen evidential criteria.
    """

    sample_count: int
    observed_dimension: int
    represented_dimension: int

    residual_numerical_rank: int
    residual_effective_rank: float
    residual_participation_ratio: float

    total_variance: float
    represented_variance: float
    residual_variance: float
    residual_variance_fraction: float

    reconstruction_error_rms: float
    normalized_reconstruction_error: float

    persistent_residual_dimension_count: int
    growth_supported: bool

    residual_dimensions: tuple[
        ResidualDimension,
        ...
    ]

    def __post_init__(self) -> None:
        if self.sample_count < 2:
            raise ValueError(
                "sample_count must be at least 2."
            )

        if self.observed_dimension < 1:
            raise ValueError(
                "observed_dimension must be positive."
            )

        if not (
            0
            <= self.represented_dimension
            <= self.observed_dimension
        ):
            raise ValueError(
                "represented_dimension is outside valid bounds."
            )

        if self.residual_numerical_rank < 0:
            raise ValueError(
                "residual_numerical_rank must be non-negative."
            )

        if self.persistent_residual_dimension_count < 0:
            raise ValueError(
                "persistent_residual_dimension_count "
                "must be non-negative."
            )

        finite_values = (
            self.residual_effective_rank,
            self.residual_participation_ratio,
            self.total_variance,
            self.represented_variance,
            self.residual_variance,
            self.residual_variance_fraction,
            self.reconstruction_error_rms,
            self.normalized_reconstruction_error,
        )

        if not all(
            math.isfinite(value)
            for value in finite_values
        ):
            raise ValueError(
                "Assessment metrics must be finite."
            )

        if not 0.0 <= self.residual_variance_fraction <= 1.0:
            raise ValueError(
                "residual_variance_fraction must lie in [0, 1]."
            )

        if (
            self.persistent_residual_dimension_count
            != len(self.residual_dimensions)
        ):
            raise ValueError(
                "persistent residual dimension count "
                "does not match residual_dimensions."
            )


class DimensionalGrowthAnalyzer:
    """
    Detect whether a current endogenous representation is insufficient.

    The analyzer receives:

        experience:
            samples x observed coordinates

        representation:
            directions already available to the internal model

    It then projects experience onto the represented subspace and
    studies what remains.

    The analyzer is intentionally blind to:

        - world coordinates,
        - body topology,
        - node identity,
        - member identity,
        - perturbation identity,
        - external force,
        - reward,
        - goal,
        - event type,
        - valence,
        - semantic meaning.

    It also does not mutate the representation and does not create
    dimensions.

    Its only role is evidential:

        "Does structured variation remain outside what is currently
        represented?"
    """

    def __init__(
        self,
        *,
        minimum_residual_variance_fraction: float = 0.01,
        minimum_dimension_fraction: float = 0.01,
        minimum_singular_value_ratio: float = 1e-8,
        numerical_floor: float = 1e-12,
    ) -> None:
        for name, value in (
            (
                "minimum_residual_variance_fraction",
                minimum_residual_variance_fraction,
            ),
            (
                "minimum_dimension_fraction",
                minimum_dimension_fraction,
            ),
        ):
            if not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{name} must be numeric."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1]."
                )

        if not isinstance(
            minimum_singular_value_ratio,
            (int, float),
        ):
            raise TypeError(
                "minimum_singular_value_ratio must be numeric."
            )

        if not math.isfinite(
            minimum_singular_value_ratio
        ):
            raise ValueError(
                "minimum_singular_value_ratio must be finite."
            )

        if minimum_singular_value_ratio < 0.0:
            raise ValueError(
                "minimum_singular_value_ratio "
                "must be non-negative."
            )

        if not isinstance(
            numerical_floor,
            (int, float),
        ):
            raise TypeError(
                "numerical_floor must be numeric."
            )

        if not math.isfinite(numerical_floor):
            raise ValueError(
                "numerical_floor must be finite."
            )

        if numerical_floor <= 0.0:
            raise ValueError(
                "numerical_floor must be positive."
            )

        self._minimum_residual_variance_fraction = float(
            minimum_residual_variance_fraction
        )

        self._minimum_dimension_fraction = float(
            minimum_dimension_fraction
        )

        self._minimum_singular_value_ratio = float(
            minimum_singular_value_ratio
        )

        self._numerical_floor = float(
            numerical_floor
        )

    @property
    def minimum_residual_variance_fraction(
        self,
    ) -> float:
        return self._minimum_residual_variance_fraction

    @property
    def minimum_dimension_fraction(
        self,
    ) -> float:
        return self._minimum_dimension_fraction

    @property
    def minimum_singular_value_ratio(
        self,
    ) -> float:
        return self._minimum_singular_value_ratio

    @property
    def numerical_floor(
        self,
    ) -> float:
        return self._numerical_floor

    def assess(
        self,
        *,
        experience: np.ndarray,
        representation: np.ndarray,
    ) -> DimensionalGrowthAssessment:
        experience = self._validate_experience(
            experience
        )

        representation = self._validate_representation(
            representation=representation,
            observed_dimension=experience.shape[1],
        )

        centered = (
            experience
            - np.mean(
                experience,
                axis=0,
                keepdims=True,
            )
        )

        basis = self._orthonormal_basis(
            representation
        )

        represented_dimension = int(
            basis.shape[1]
        )

        if represented_dimension == 0:
            reconstructed = np.zeros_like(
                centered
            )
        else:
            reconstructed = (
                centered
                @ basis
                @ basis.T
            )

        residual = (
            centered
            - reconstructed
        )

        total_variance = self._total_variance(
            centered
        )

        represented_variance = self._total_variance(
            reconstructed
        )

        residual_variance = self._total_variance(
            residual
        )

        if total_variance <= self._numerical_floor:
            residual_variance_fraction = 0.0
        else:
            residual_variance_fraction = float(
                residual_variance
                / total_variance
            )

        reconstruction_error_rms = float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        )

        experience_rms = float(
            np.sqrt(
                np.mean(
                    centered ** 2
                )
            )
        )

        if experience_rms <= self._numerical_floor:
            normalized_reconstruction_error = 0.0
        else:
            normalized_reconstruction_error = float(
                reconstruction_error_rms
                / experience_rms
            )

        (
            residual_numerical_rank,
            residual_effective_rank,
            residual_participation_ratio,
            residual_dimensions,
        ) = self._analyze_residual(
            residual
        )

        persistent_count = len(
            residual_dimensions
        )

        growth_supported = bool(
            residual_variance_fraction
            >= self._minimum_residual_variance_fraction
            and persistent_count > 0
        )

        return DimensionalGrowthAssessment(
            sample_count=int(
                experience.shape[0]
            ),
            observed_dimension=int(
                experience.shape[1]
            ),
            represented_dimension=(
                represented_dimension
            ),
            residual_numerical_rank=(
                residual_numerical_rank
            ),
            residual_effective_rank=(
                residual_effective_rank
            ),
            residual_participation_ratio=(
                residual_participation_ratio
            ),
            total_variance=(
                total_variance
            ),
            represented_variance=(
                represented_variance
            ),
            residual_variance=(
                residual_variance
            ),
            residual_variance_fraction=(
                residual_variance_fraction
            ),
            reconstruction_error_rms=(
                reconstruction_error_rms
            ),
            normalized_reconstruction_error=(
                normalized_reconstruction_error
            ),
            persistent_residual_dimension_count=(
                persistent_count
            ),
            growth_supported=(
                growth_supported
            ),
            residual_dimensions=tuple(
                residual_dimensions
            ),
        )

    def _analyze_residual(
        self,
        residual: np.ndarray,
    ) -> tuple[
        int,
        float,
        float,
        list[ResidualDimension],
    ]:
        if np.allclose(
            residual,
            0.0,
            atol=self._numerical_floor,
            rtol=0.0,
        ):
            return (
                0,
                0.0,
                0.0,
                [],
            )

        _u, singular_values, vt = np.linalg.svd(
            residual,
            full_matrices=False,
        )

        singular_values = np.asarray(
            singular_values,
            dtype=float,
        )

        if singular_values.size == 0:
            return (
                0,
                0.0,
                0.0,
                [],
            )

        maximum_singular_value = float(
            singular_values[0]
        )

        machine_tolerance = (
            max(residual.shape)
            * np.finfo(float).eps
            * maximum_singular_value
        )

        ratio_tolerance = (
            self._minimum_singular_value_ratio
            * maximum_singular_value
        )

        tolerance = max(
            machine_tolerance,
            ratio_tolerance,
            self._numerical_floor,
        )

        numerical_rank = int(
            np.sum(
                singular_values > tolerance
            )
        )

        eigenvalues = (
            singular_values ** 2
        ) / max(
            residual.shape[0] - 1,
            1,
        )

        total = float(
            np.sum(
                eigenvalues
            )
        )

        if total <= self._numerical_floor:
            return (
                numerical_rank,
                0.0,
                0.0,
                [],
            )

        probabilities = (
            eigenvalues
            / total
        )

        positive = probabilities[
            probabilities > 0.0
        ]

        entropy = float(
            -np.sum(
                positive
                * np.log(
                    positive
                )
            )
        )

        effective_rank = float(
            np.exp(entropy)
        )

        squared_sum = float(
            np.sum(
                eigenvalues ** 2
            )
        )

        participation_ratio = (
            float(
                total ** 2
                / squared_sum
            )
            if squared_sum
            > self._numerical_floor
            else 0.0
        )

        cumulative = 0.0

        residual_dimensions: list[
            ResidualDimension
        ] = []

        for index, eigenvalue in enumerate(
            eigenvalues
        ):
            fraction = float(
                eigenvalue
                / total
            )

            cumulative += fraction

            if index >= numerical_rank:
                continue

            if (
                fraction
                < self._minimum_dimension_fraction
            ):
                continue

            direction = tuple(
                float(value)
                for value in vt[index]
            )

            residual_dimensions.append(
                ResidualDimension(
                    residual_index=index,
                    eigenvalue=float(
                        eigenvalue
                    ),
                    explained_residual_fraction=(
                        fraction
                    ),
                    cumulative_residual_fraction=(
                        min(
                            cumulative,
                            1.0,
                        )
                    ),
                    direction=direction,
                )
            )

        return (
            numerical_rank,
            effective_rank,
            participation_ratio,
            residual_dimensions,
        )

    @staticmethod
    def _total_variance(
        matrix: np.ndarray,
    ) -> float:
        if matrix.shape[0] < 2:
            return 0.0

        return float(
            np.sum(
                np.var(
                    matrix,
                    axis=0,
                    ddof=1,
                )
            )
        )

    @staticmethod
    def _orthonormal_basis(
        representation: np.ndarray,
    ) -> np.ndarray:
        observed_dimension = int(
            representation.shape[0]
        )

        if representation.shape[1] == 0:
            return np.empty(
                (
                    observed_dimension,
                    0,
                ),
                dtype=float,
            )

        u, singular_values, _vt = np.linalg.svd(
            representation,
            full_matrices=False,
        )

        if singular_values.size == 0:
            return np.empty(
                (
                    observed_dimension,
                    0,
                ),
                dtype=float,
            )

        tolerance = (
            max(
                representation.shape
            )
            * np.finfo(float).eps
            * float(
                singular_values[0]
            )
        )

        rank = int(
            np.sum(
                singular_values > tolerance
            )
        )

        return np.asarray(
            u[:, :rank],
            dtype=float,
        )

    @staticmethod
    def _validate_experience(
        experience: np.ndarray,
    ) -> np.ndarray:
        experience = np.asarray(
            experience,
            dtype=float,
        )

        if experience.ndim != 2:
            raise ValueError(
                "experience must be two-dimensional."
            )

        if experience.shape[0] < 2:
            raise ValueError(
                "experience must contain at least two samples."
            )

        if experience.shape[1] < 1:
            raise ValueError(
                "experience must contain at least one dimension."
            )

        if not np.all(
            np.isfinite(
                experience
            )
        ):
            raise ValueError(
                "experience must contain only finite values."
            )

        return experience

    @staticmethod
    def _validate_representation(
        *,
        representation: np.ndarray,
        observed_dimension: int,
    ) -> np.ndarray:
        representation = np.asarray(
            representation,
            dtype=float,
        )

        if representation.ndim != 2:
            raise ValueError(
                "representation must be two-dimensional."
            )

        if (
            representation.shape[0]
            != observed_dimension
        ):
            raise ValueError(
                "representation first dimension must equal "
                "the observed dimension."
            )

        if not np.all(
            np.isfinite(
                representation
            )
        ):
            raise ValueError(
                "representation must contain only finite values."
            )

        return representation
