from __future__ import annotations

import inspect

import numpy as np
import pytest

from roif.development.dimensional_growth import (
    DimensionalGrowthAnalyzer,
    ResidualDimension,
)


def _orthonormal_matrix(
    *,
    rows: int,
    columns: int,
    seed: int,
) -> np.ndarray:
    """
    Construct a deterministic orthonormal basis without encoding any
    coordinate-axis preference into the test data.
    """
    if columns > rows:
        raise ValueError("columns must not exceed rows.")

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        size=(rows, columns)
    )

    q, _r = np.linalg.qr(matrix)

    return np.asarray(
        q[:, :columns],
        dtype=float,
    )


def _latent_experience(
    *,
    sample_count: int,
    observed_dimension: int,
    latent_dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build deterministic observations whose true linear dimension is
    known by construction.

    Returns:
        experience:
            samples x observed dimensions

        latent_basis:
            observed dimensions x latent dimensions
    """
    if latent_dimension > observed_dimension:
        raise ValueError(
            "latent_dimension must not exceed observed_dimension."
        )

    rng = np.random.default_rng(seed)

    latent_basis = _orthonormal_matrix(
        rows=observed_dimension,
        columns=latent_dimension,
        seed=seed + 1,
    )

    latent_state = rng.normal(
        size=(
            sample_count,
            latent_dimension,
        )
    )

    experience = (
        latent_state
        @ latent_basis.T
    )

    return (
        np.asarray(
            experience,
            dtype=float,
        ),
        latent_basis,
    )


def test_residual_dimension_rejects_negative_index():
    with pytest.raises(ValueError):
        ResidualDimension(
            residual_index=-1,
            eigenvalue=1.0,
            explained_residual_fraction=1.0,
            cumulative_residual_fraction=1.0,
            direction=(1.0,),
        )


def test_analyzer_rejects_non_matrix_experience():
    analyzer = DimensionalGrowthAnalyzer()

    with pytest.raises(ValueError):
        analyzer.assess(
            experience=np.arange(10.0),
            representation=np.empty(
                (1, 0)
            ),
        )


def test_analyzer_rejects_single_sample():
    analyzer = DimensionalGrowthAnalyzer()

    with pytest.raises(ValueError):
        analyzer.assess(
            experience=np.zeros(
                (1, 3)
            ),
            representation=np.empty(
                (3, 0)
            ),
        )


def test_analyzer_rejects_representation_with_wrong_observed_dimension():
    analyzer = DimensionalGrowthAnalyzer()

    experience = np.zeros(
        (10, 3)
    )

    representation = np.zeros(
        (4, 1)
    )

    with pytest.raises(ValueError):
        analyzer.assess(
            experience=experience,
            representation=representation,
        )


def test_analyzer_rejects_nonfinite_experience():
    analyzer = DimensionalGrowthAnalyzer()

    experience = np.zeros(
        (10, 3)
    )

    experience[5, 1] = np.nan

    with pytest.raises(ValueError):
        analyzer.assess(
            experience=experience,
            representation=np.empty(
                (3, 0)
            ),
        )


def test_complete_two_dimensional_representation_requires_no_growth():
    experience, latent_basis = _latent_experience(
        sample_count=1000,
        observed_dimension=2,
        latent_dimension=2,
        seed=100,
    )

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=latent_basis,
    )

    assert result.observed_dimension == 2
    assert result.represented_dimension == 2

    assert result.residual_numerical_rank == 0
    assert result.persistent_residual_dimension_count == 0

    assert result.residual_variance_fraction == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result.growth_supported is False


def test_three_dimensional_experience_with_two_dimensions_finds_one_missing_dimension():
    experience, latent_basis = _latent_experience(
        sample_count=2000,
        observed_dimension=3,
        latent_dimension=3,
        seed=200,
    )

    representation = latent_basis[:, :2]

    analyzer = DimensionalGrowthAnalyzer(
        minimum_residual_variance_fraction=0.01,
        minimum_dimension_fraction=0.01,
    )

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert result.observed_dimension == 3
    assert result.represented_dimension == 2

    assert result.residual_numerical_rank == 1

    assert (
        result.persistent_residual_dimension_count
        == 1
    )

    assert result.growth_supported is True

    assert (
        result.residual_dimensions[0]
        .explained_residual_fraction
        == pytest.approx(
            1.0,
            abs=1e-10,
        )
    )


def test_five_dimensional_experience_with_two_dimensions_finds_three_missing_dimensions():
    experience, latent_basis = _latent_experience(
        sample_count=4000,
        observed_dimension=5,
        latent_dimension=5,
        seed=300,
    )

    representation = latent_basis[:, :2]

    analyzer = DimensionalGrowthAnalyzer(
        minimum_residual_variance_fraction=0.01,
        minimum_dimension_fraction=0.01,
    )

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert result.observed_dimension == 5
    assert result.represented_dimension == 2

    assert result.residual_numerical_rank == 3

    assert (
        result.persistent_residual_dimension_count
        == 3
    )

    assert result.growth_supported is True


def test_thirty_observed_channels_with_true_rank_twelve_and_seven_represented_find_five_missing_dimensions():
    """
    Central synthetic control.

    The analyzer sees thirty observable coordinates.

    The experience actually occupies a twelve-dimensional latent
    subspace.

    Seven dimensions of that subspace are supplied as already
    represented.

    The expected residual dimension is therefore five, but that number
    is not supplied to the analyzer.
    """
    experience, latent_basis = _latent_experience(
        sample_count=6000,
        observed_dimension=30,
        latent_dimension=12,
        seed=400,
    )

    representation = latent_basis[:, :7]

    analyzer = DimensionalGrowthAnalyzer(
        minimum_residual_variance_fraction=0.01,
        minimum_dimension_fraction=0.01,
        minimum_singular_value_ratio=1e-8,
    )

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert result.observed_dimension == 30
    assert result.represented_dimension == 7

    assert result.residual_numerical_rank == 5

    assert (
        result.persistent_residual_dimension_count
        == 5
    )

    assert result.growth_supported is True

    assert len(
        result.residual_dimensions
    ) == 5


def test_observed_channel_count_does_not_determine_missing_dimension_count():
    """
    Thirty channels must not be interpreted as thirty dimensions.
    """
    experience, latent_basis = _latent_experience(
        sample_count=3000,
        observed_dimension=30,
        latent_dimension=4,
        seed=500,
    )

    representation = latent_basis[:, :3]

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert result.observed_dimension == 30
    assert result.represented_dimension == 3

    assert result.residual_numerical_rank == 1

    assert (
        result.persistent_residual_dimension_count
        == 1
    )


def test_rotated_latent_space_preserves_missing_dimension_count():
    """
    Growth detection must depend on dimensional structure rather than
    privileged observable axes.
    """
    experience, latent_basis = _latent_experience(
        sample_count=3000,
        observed_dimension=17,
        latent_dimension=8,
        seed=600,
    )

    representation = latent_basis[:, :5]

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert result.residual_numerical_rank == 3

    assert (
        result.persistent_residual_dimension_count
        == 3
    )

    assert result.growth_supported is True


def test_redundant_representation_vectors_do_not_inflate_represented_dimension():
    experience, latent_basis = _latent_experience(
        sample_count=2000,
        observed_dimension=6,
        latent_dimension=4,
        seed=700,
    )

    first = latent_basis[:, 0]
    second = latent_basis[:, 1]

    representation = np.column_stack(
        (
            first,
            second,
            2.0 * first,
            -3.0 * second,
        )
    )

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert result.represented_dimension == 2
    assert result.residual_numerical_rank == 2

    assert (
        result.persistent_residual_dimension_count
        == 2
    )


def test_small_unstructured_noise_does_not_force_dimensional_growth():
    """
    A complete representation with very small measurement noise should
    not automatically cause the system to invent dimensions.
    """
    rng = np.random.default_rng(800)

    sample_count = 4000
    observed_dimension = 10
    latent_dimension = 3

    clean, latent_basis = _latent_experience(
        sample_count=sample_count,
        observed_dimension=observed_dimension,
        latent_dimension=latent_dimension,
        seed=801,
    )

    noise = rng.normal(
        scale=1e-5,
        size=clean.shape,
    )

    experience = clean + noise

    analyzer = DimensionalGrowthAnalyzer(
        minimum_residual_variance_fraction=1e-4,
        minimum_dimension_fraction=0.01,
        minimum_singular_value_ratio=1e-8,
    )

    result = analyzer.assess(
        experience=experience,
        representation=latent_basis,
    )

    assert (
        result.residual_variance_fraction
        < analyzer.minimum_residual_variance_fraction
    )

    assert result.growth_supported is False


def test_large_structured_missing_component_supports_growth():
    """
    A coherent missing direction should survive the evidential gate.
    """
    rng = np.random.default_rng(900)

    sample_count = 3000
    observed_dimension = 8

    basis = _orthonormal_matrix(
        rows=observed_dimension,
        columns=4,
        seed=901,
    )

    represented_latent = rng.normal(
        size=(sample_count, 3)
    )

    missing_latent = rng.normal(
        size=(sample_count, 1)
    )

    experience = (
        represented_latent
        @ basis[:, :3].T
        + missing_latent
        @ basis[:, 3:4].T
    )

    analyzer = DimensionalGrowthAnalyzer(
        minimum_residual_variance_fraction=0.01,
        minimum_dimension_fraction=0.01,
    )

    result = analyzer.assess(
        experience=experience,
        representation=basis[:, :3],
    )

    assert result.residual_numerical_rank == 1

    assert (
        result.persistent_residual_dimension_count
        == 1
    )

    assert result.growth_supported is True


def test_growth_is_not_supported_when_residual_variance_gate_is_not_met():
    rng = np.random.default_rng(1000)

    sample_count = 3000

    basis = _orthonormal_matrix(
        rows=5,
        columns=3,
        seed=1001,
    )

    represented = rng.normal(
        size=(sample_count, 2)
    )

    tiny_missing = rng.normal(
        scale=1e-4,
        size=(sample_count, 1),
    )

    experience = (
        represented
        @ basis[:, :2].T
        + tiny_missing
        @ basis[:, 2:3].T
    )

    analyzer = DimensionalGrowthAnalyzer(
        minimum_residual_variance_fraction=0.01,
        minimum_dimension_fraction=0.01,
    )

    result = analyzer.assess(
        experience=experience,
        representation=basis[:, :2],
    )

    assert result.residual_numerical_rank == 1

    assert (
        result.residual_variance_fraction
        < 0.01
    )

    assert result.growth_supported is False


def test_residual_directions_are_orthogonal_to_existing_representation():
    experience, latent_basis = _latent_experience(
        sample_count=3000,
        observed_dimension=9,
        latent_dimension=6,
        seed=1100,
    )

    representation = latent_basis[:, :4]

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert (
        result.persistent_residual_dimension_count
        == 2
    )

    for residual_dimension in result.residual_dimensions:
        direction = np.asarray(
            residual_dimension.direction,
            dtype=float,
        )

        overlaps = (
            representation.T
            @ direction
        )

        assert np.allclose(
            overlaps,
            0.0,
            atol=1e-10,
            rtol=0.0,
        )


def test_residual_directions_are_mutually_orthogonal():
    experience, latent_basis = _latent_experience(
        sample_count=3000,
        observed_dimension=10,
        latent_dimension=7,
        seed=1200,
    )

    representation = latent_basis[:, :3]

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=representation,
    )

    directions = np.column_stack(
        [
            np.asarray(
                item.direction,
                dtype=float,
            )
            for item in result.residual_dimensions
        ]
    )

    gram = directions.T @ directions

    assert np.allclose(
        gram,
        np.eye(
            directions.shape[1]
        ),
        atol=1e-10,
        rtol=0.0,
    )


def test_residual_explained_fractions_are_nonincreasing():
    experience, latent_basis = _latent_experience(
        sample_count=4000,
        observed_dimension=12,
        latent_dimension=8,
        seed=1300,
    )

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=latent_basis[:, :2],
    )

    fractions = [
        item.explained_residual_fraction
        for item in result.residual_dimensions
    ]

    assert fractions == sorted(
        fractions,
        reverse=True,
    )


def test_cumulative_residual_fraction_is_monotonic():
    experience, latent_basis = _latent_experience(
        sample_count=4000,
        observed_dimension=12,
        latent_dimension=8,
        seed=1400,
    )

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=latent_basis[:, :2],
    )

    cumulative = [
        item.cumulative_residual_fraction
        for item in result.residual_dimensions
    ]

    assert all(
        later >= earlier
        for earlier, later in zip(
            cumulative,
            cumulative[1:],
        )
    )


def test_effective_rank_is_positive_for_nonzero_residual():
    experience, latent_basis = _latent_experience(
        sample_count=2000,
        observed_dimension=7,
        latent_dimension=5,
        seed=1500,
    )

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=latent_basis[:, :2],
    )

    assert result.residual_effective_rank > 0.0
    assert result.residual_participation_ratio > 0.0


def test_complete_representation_has_zero_normalized_reconstruction_error():
    experience, latent_basis = _latent_experience(
        sample_count=2000,
        observed_dimension=6,
        latent_dimension=4,
        seed=1600,
    )

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=latent_basis,
    )

    assert (
        result.normalized_reconstruction_error
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_empty_representation_can_detect_all_latent_dimensions():
    experience, _latent_basis = _latent_experience(
        sample_count=4000,
        observed_dimension=12,
        latent_dimension=5,
        seed=1700,
    )

    analyzer = DimensionalGrowthAnalyzer(
        minimum_dimension_fraction=0.01,
    )

    result = analyzer.assess(
        experience=experience,
        representation=np.empty(
            (12, 0)
        ),
    )

    assert result.represented_dimension == 0
    assert result.residual_numerical_rank == 5

    assert (
        result.persistent_residual_dimension_count
        == 5
    )

    assert result.growth_supported is True


def test_constant_experience_does_not_support_growth():
    experience = np.ones(
        (1000, 8),
        dtype=float,
    )

    analyzer = DimensionalGrowthAnalyzer()

    result = analyzer.assess(
        experience=experience,
        representation=np.empty(
            (8, 0)
        ),
    )

    assert result.residual_numerical_rank == 0
    assert result.residual_effective_rank == 0.0
    assert result.residual_participation_ratio == 0.0

    assert (
        result.persistent_residual_dimension_count
        == 0
    )

    assert result.growth_supported is False


def test_analyzer_has_no_ground_truth_or_world_interface():
    forbidden = {
        "ground_truth",
        "world_state",
        "body_ground_truth",
        "world",
        "coordinates",
    }

    parameters = set(
        inspect.signature(
            DimensionalGrowthAnalyzer.assess
        ).parameters
    )

    assert forbidden.isdisjoint(
        parameters
    )


def test_analyzer_has_no_topology_interface():
    forbidden = {
        "topology",
        "node_mapping",
        "member_mapping",
        "node_index",
        "member_index",
        "body_part",
        "anatomy",
    }

    parameters = set(
        inspect.signature(
            DimensionalGrowthAnalyzer.assess
        ).parameters
    )

    assert forbidden.isdisjoint(
        parameters
    )


def test_analyzer_has_no_external_perturbation_interface():
    forbidden = {
        "perturbation_time",
        "perturbation_step",
        "perturbation_node",
        "external_force",
        "force",
    }

    parameters = set(
        inspect.signature(
            DimensionalGrowthAnalyzer.assess
        ).parameters
    )

    assert forbidden.isdisjoint(
        parameters
    )


def test_analyzer_has_no_reward_goal_event_or_valence_interface():
    forbidden = {
        "reward",
        "goal",
        "event_type",
        "event_label",
        "valence",
        "meaning",
        "target",
    }

    parameters = set(
        inspect.signature(
            DimensionalGrowthAnalyzer.assess
        ).parameters
    )

    assert forbidden.isdisjoint(
        parameters
    )


def test_analyzer_does_not_mutate_experience_or_representation():
    experience, latent_basis = _latent_experience(
        sample_count=1000,
        observed_dimension=6,
        latent_dimension=4,
        seed=1800,
    )

    representation = latent_basis[:, :2].copy()

    original_experience = experience.copy()
    original_representation = representation.copy()

    analyzer = DimensionalGrowthAnalyzer()

    analyzer.assess(
        experience=experience,
        representation=representation,
    )

    assert np.array_equal(
        experience,
        original_experience,
    )

    assert np.array_equal(
        representation,
        original_representation,
    )


def test_analyzer_does_not_contain_dimension_creation_interface():
    forbidden_names = {
        "grow",
        "create_dimension",
        "add_dimension",
        "append_dimension",
        "expand",
        "mutate",
        "develop",
    }

    public_names = {
        name
        for name in dir(
            DimensionalGrowthAnalyzer
        )
        if not name.startswith("_")
    }

    assert forbidden_names.isdisjoint(
        public_names
    )
