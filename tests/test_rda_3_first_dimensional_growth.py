from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np

from experiments.roif_rda_3_first_dimensional_growth import (
    BENCHMARK_VERSION,
    FROZEN_DIFFERENCE_ANALYZER_COMMIT,
    FROZEN_DIMENSIONAL_GROWTH_COMMIT,
    FirstDimensionalGrowthConfig,
    _build_endogenous_representation,
    run_first_dimensional_growth,
)


def short_config() -> FirstDimensionalGrowthConfig:
    return FirstDimensionalGrowthConfig(
        duration=0.10,
        dt=0.001,
        baseline_duration=0.02,
        representation_start_time=0.02,
        representation_end_time=0.04,
        perturbation_time=0.05,
        assessment_start_time=0.05,
        assessment_end_time=0.08,
        perturbation_node=0,
        perturbation_force_x=1.0,
        perturbation_force_y=0.0,
        perturbation_force_z=0.0,
        representation_variance_fraction=0.99,
    )


def test_benchmark_identity_is_stable():
    assert (
        BENCHMARK_VERSION
        == "roif_rda_3_first_dimensional_growth_v1"
    )


def test_frozen_component_commits_are_explicit():
    assert (
        FROZEN_DIMENSIONAL_GROWTH_COMMIT
        == "31292ec"
    )
    assert (
        FROZEN_DIFFERENCE_ANALYZER_COMMIT
        == "0aee759"
    )


def test_representation_is_built_before_perturbation():
    config = short_config()

    assert (
        config.representation_start_time
        < config.representation_end_time
        <= config.perturbation_time
    )


def test_assessment_occurs_at_or_after_perturbation():
    config = short_config()

    assert (
        config.perturbation_time
        <= config.assessment_start_time
        < config.assessment_end_time
    )


def test_configuration_does_not_accept_expected_dimension_count():
    parameters = set(
        inspect.signature(
            FirstDimensionalGrowthConfig
        ).parameters
    )

    forbidden = {
        "expected_dimension_count",
        "target_dimension",
        "world_dimension",
        "latent_dimension",
        "represented_dimension",
    }

    assert forbidden.isdisjoint(
        parameters
    )


def test_representation_builder_selects_dimension_from_experience():
    rng = np.random.default_rng(100)

    latent = rng.normal(
        size=(1000, 2)
    )

    mixing = rng.normal(
        size=(2, 7)
    )

    experience = latent @ mixing

    representation, summary = (
        _build_endogenous_representation(
            experience,
            variance_fraction=0.99,
        )
    )

    assert (
        representation.shape[0]
        == experience.shape[1]
    )

    assert (
        representation.shape[1]
        == summary["selected_dimension_count"]
    )

    assert (
        summary["selected_variance_fraction"]
        >= 0.99
    )


def test_representation_builder_has_no_expected_dimension_input():
    parameters = set(
        inspect.signature(
            _build_endogenous_representation
        ).parameters
    )

    forbidden = {
        "expected_dimension_count",
        "target_dimension",
        "world_dimension",
        "latent_dimension",
    }

    assert forbidden.isdisjoint(
        parameters
    )


def test_run_identifies_rda_3():
    result = run_first_dimensional_growth(
        short_config()
    )

    assert (
        result["developmental_stage"]
        == "RDA-3"
    )

    assert (
        result["experience"]
        == "first_dimensional_growth_assessment"
    )


def test_run_preserves_frozen_component_identity():
    result = run_first_dimensional_growth(
        short_config()
    )

    frozen = result[
        "frozen_components"
    ]

    assert (
        frozen[
            "dimensional_growth_analyzer_commit"
        ]
        == "31292ec"
    )

    assert (
        frozen[
            "difference_analyzer_commit"
        ]
        == "0aee759"
    )

    assert (
        frozen[
            "parameters_modified_after_freeze"
        ]
        is False
    )


def test_epistemic_boundary_contains_only_false_values():
    result = run_first_dimensional_growth(
        short_config()
    )

    boundary = result[
        "epistemic_boundary"
    ]

    assert boundary

    assert all(
        value is False
        for value in boundary.values()
    )


def test_expected_dimension_is_explicitly_not_supplied():
    result = run_first_dimensional_growth(
        short_config()
    )

    boundary = result[
        "epistemic_boundary"
    ]

    assert (
        boundary[
            "expected_dimension_count_supplied"
        ]
        is False
    )

    assert (
        boundary[
            "world_dimension_count_supplied"
        ]
        is False
    )


def test_initial_representation_returns_unknown_dimension_count():
    result = run_first_dimensional_growth(
        short_config()
    )

    count = result[
        "endogenous_initial_representation"
    ][
        "selected_dimension_count"
    ]

    assert isinstance(
        count,
        int,
    )

    assert count >= 0


def test_assessment_returns_unknown_residual_dimension_count():
    result = run_first_dimensional_growth(
        short_config()
    )

    count = result[
        "dimensional_growth_assessment"
    ][
        "persistent_residual_dimension_count"
    ]

    assert isinstance(
        count,
        int,
    )

    assert count >= 0


def test_contract_does_not_require_growth():
    result = run_first_dimensional_growth(
        short_config()
    )

    assert isinstance(
        result[
            "dimensional_growth_assessment"
        ][
            "growth_supported"
        ],
        bool,
    )


def test_contract_does_not_require_specific_residual_rank():
    result = run_first_dimensional_growth(
        short_config()
    )

    rank = result[
        "dimensional_growth_assessment"
    ][
        "residual_numerical_rank"
    ]

    assert isinstance(
        rank,
        int,
    )

    assert rank >= 0


def test_representation_window_precedes_assessment_window():
    result = run_first_dimensional_growth(
        short_config()
    )

    representation_steps = result[
        "representation_step_indices"
    ]

    assessment_steps = result[
        "assessment_step_indices"
    ]

    assert representation_steps
    assert assessment_steps

    assert (
        max(representation_steps)
        < min(assessment_steps)
    )


def test_representation_window_excludes_perturbation_step():
    result = run_first_dimensional_growth(
        short_config()
    )

    perturbation_step = result[
        "ground_truth_evaluation_only"
    ][
        "perturbation_step"
    ]

    assert all(
        step < perturbation_step
        for step in result[
            "representation_step_indices"
        ]
    )


def test_assessment_window_starts_at_or_after_perturbation():
    result = run_first_dimensional_growth(
        short_config()
    )

    perturbation_step = result[
        "ground_truth_evaluation_only"
    ][
        "perturbation_step"
    ]

    assert all(
        step >= perturbation_step
        for step in result[
            "assessment_step_indices"
        ]
    )


def test_ground_truth_is_separated_for_evaluation_only():
    result = run_first_dimensional_growth(
        short_config()
    )

    evaluation = result[
        "ground_truth_evaluation_only"
    ]

    assert (
        evaluation[
            "perturbation_node"
        ]
        == 0
    )

    assert (
        evaluation[
            "perturbation_force"
        ]
        == [
            1.0,
            0.0,
            0.0,
        ]
    )


def test_assessment_has_no_ground_truth_fields():
    result = run_first_dimensional_growth(
        short_config()
    )

    assessment = result[
        "dimensional_growth_assessment"
    ]

    forbidden = {
        "ground_truth",
        "world_state",
        "topology",
        "node_mapping",
        "member_mapping",
        "perturbation_node",
        "perturbation_force",
        "external_force",
        "world_dimension",
        "expected_dimension_count",
        "reward",
        "goal",
        "event_type",
        "valence",
    }

    assert forbidden.isdisjoint(
        assessment.keys()
    )


def test_residual_dimensions_have_no_semantic_labels():
    result = run_first_dimensional_growth(
        short_config()
    )

    residuals = result[
        "dimensional_growth_assessment"
    ][
        "residual_dimensions"
    ]

    forbidden = {
        "body_part",
        "anatomy",
        "meaning",
        "semantic_label",
        "world_axis",
        "node",
        "member",
        "cause",
        "event",
    }

    for item in residuals:
        assert forbidden.isdisjoint(
            item.keys()
        )


def test_residual_directions_are_only_hashed_in_result():
    result = run_first_dimensional_growth(
        short_config()
    )

    residuals = result[
        "dimensional_growth_assessment"
    ][
        "residual_dimensions"
    ]

    for item in residuals:
        digest = item[
            "direction_sha256"
        ]

        assert isinstance(
            digest,
            str,
        )

        assert len(digest) == 64


def test_hashes_have_sha256_shape():
    result = run_first_dimensional_growth(
        short_config()
    )

    for key in (
        "representation_experience_sha256",
        "assessment_experience_sha256",
        "representation_basis_sha256",
        "run_sha256",
    ):
        digest = result[key]

        assert isinstance(
            digest,
            str,
        )

        assert len(digest) == 64


def test_run_is_deterministic():
    config = short_config()

    first = run_first_dimensional_growth(
        config
    )

    second = run_first_dimensional_growth(
        config
    )

    assert (
        first[
            "representation_experience_sha256"
        ]
        == second[
            "representation_experience_sha256"
        ]
    )

    assert (
        first[
            "assessment_experience_sha256"
        ]
        == second[
            "assessment_experience_sha256"
        ]
    )

    assert (
        first[
            "representation_basis_sha256"
        ]
        == second[
            "representation_basis_sha256"
        ]
    )

    assert (
        first["run_sha256"]
        == second["run_sha256"]
    )


def test_force_change_does_not_change_preperturbation_representation():
    config = short_config()

    changed = replace(
        config,
        perturbation_force_x=0.5,
    )

    first = run_first_dimensional_growth(
        config
    )

    second = run_first_dimensional_growth(
        changed
    )

    assert (
        first[
            "representation_experience_sha256"
        ]
        == second[
            "representation_experience_sha256"
        ]
    )

    assert (
        first[
            "representation_basis_sha256"
        ]
        == second[
            "representation_basis_sha256"
        ]
    )


def test_force_change_may_change_postperturbation_experience():
    config = short_config()

    changed = replace(
        config,
        perturbation_force_x=0.5,
    )

    first = run_first_dimensional_growth(
        config
    )

    second = run_first_dimensional_growth(
        changed
    )

    assert isinstance(
        first[
            "assessment_experience_sha256"
        ],
        str,
    )

    assert isinstance(
        second[
            "assessment_experience_sha256"
        ],
        str,
    )


def test_top_level_result_has_no_assigned_meaning():
    result = run_first_dimensional_growth(
        short_config()
    )

    forbidden = {
        "meaning",
        "purpose",
        "goal",
        "reward",
        "valence",
        "diagnosis",
        "body_part",
        "world_dimension_label",
    }

    assert forbidden.isdisjoint(
        result.keys()
    )


def test_contract_never_asserts_specific_dimension_number():
    """
    Здесь специально нет проверки ответа на 3, 5, 7, 12, 18
    или любое другое желаемое число.

    Проверяется только то, что эксперимент самостоятельно
    вернул целочисленные оценки размерности.
    """

    result = run_first_dimensional_growth(
        short_config()
    )

    assessment = result[
        "dimensional_growth_assessment"
    ]

    values = (
        assessment[
            "represented_dimension"
        ],
        assessment[
            "residual_numerical_rank"
        ],
        assessment[
            "persistent_residual_dimension_count"
        ],
    )

    assert all(
        isinstance(
            value,
            int,
        )
        for value in values
    )
