from __future__ import annotations

from experiments.roif_rda_3_first_dimensional_growth import (
    run_first_dimensional_growth,
)
from roif.development.dimensional_growth import (
    DimensionalGrowthAssessment,
)


EXPECTED_RUN_SHA256 = (
    "3423ebb7346656f6336aed9a68d05bbbf"
    "5913763c0612f5c91dee656b47ab24a"
)


def test_default_rda3_run_remains_unchanged():
    result = run_first_dimensional_growth()

    assert isinstance(result, dict)
    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    assessment_record = result[
        "dimensional_growth_assessment"
    ]

    assert assessment_record["growth_supported"] is True
    assert assessment_record["represented_dimension"] == 3

    assert (
        assessment_record[
            "persistent_residual_dimension_count"
        ]
        == 2
    )

    assert (
        assessment_record["residual_variance_fraction"]
        == 0.45759965414789505
    )


def test_rda3_can_return_real_dimensional_growth_assessment():
    result, assessment = run_first_dimensional_growth(
        return_internal=True,
    )

    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    assert isinstance(
        assessment,
        DimensionalGrowthAssessment,
    )

    assert assessment.growth_supported is True
    assert assessment.represented_dimension == 3

    assert (
        assessment.persistent_residual_dimension_count
        == 2
    )

    assert (
        assessment.residual_variance_fraction
        == 0.45759965414789505
    )


def test_internal_assessment_matches_serialized_result():
    result, assessment = run_first_dimensional_growth(
        return_internal=True,
    )

    assessment_record = result[
        "dimensional_growth_assessment"
    ]

    assert (
        assessment_record["growth_supported"]
        == assessment.growth_supported
    )

    assert (
        assessment_record["represented_dimension"]
        == assessment.represented_dimension
    )

    assert (
        assessment_record[
            "persistent_residual_dimension_count"
        ]
        == assessment.persistent_residual_dimension_count
    )

    assert (
        assessment_record["residual_variance_fraction"]
        == assessment.residual_variance_fraction
    )

    assert (
        assessment_record["residual_effective_rank"]
        == assessment.residual_effective_rank
    )

    assert (
        assessment_record["residual_numerical_rank"]
        == assessment.residual_numerical_rank
    )

    assert (
        assessment_record["residual_participation_ratio"]
        == assessment.residual_participation_ratio
    )
