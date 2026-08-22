from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.roif_rda_2_constitutive_coupling_audit import (
    BENCHMARK_VERSION,
    RDA_2_RESULT_COMMIT,
    ConstitutiveCouplingAuditConfig,
    _linear_fit_metrics,
    run_constitutive_coupling_audit,
)


def short_config() -> ConstitutiveCouplingAuditConfig:
    return ConstitutiveCouplingAuditConfig(
        duration=0.08,
        dt=0.001,
        perturbation_time=0.04,
        perturbation_node=0,
        perturbation_force_x=1.0,
        perturbation_force_y=0.0,
        perturbation_force_z=0.0,
        fit_start_time=0.04,
        fit_end_time=0.06,
    )


def test_benchmark_identity_is_stable():
    assert (
        BENCHMARK_VERSION
        == "roif_rda_2_constitutive_coupling_audit_v1"
    )


def test_audited_rda_2_commit_is_explicit():
    assert RDA_2_RESULT_COMMIT == "8e182d3"


def test_configuration_requires_positive_duration():
    with pytest.raises(ValueError):
        ConstitutiveCouplingAuditConfig(
            duration=0.0
        )


def test_configuration_requires_positive_dt():
    with pytest.raises(ValueError):
        ConstitutiveCouplingAuditConfig(
            dt=0.0
        )


def test_configuration_requires_perturbation_inside_run():
    with pytest.raises(ValueError):
        ConstitutiveCouplingAuditConfig(
            duration=1.0,
            perturbation_time=1.0,
        )


def test_configuration_requires_valid_fit_interval():
    with pytest.raises(ValueError):
        ConstitutiveCouplingAuditConfig(
            duration=1.0,
            fit_start_time=0.8,
            fit_end_time=0.7,
        )


def test_configuration_rejects_negative_perturbation_node():
    with pytest.raises(ValueError):
        ConstitutiveCouplingAuditConfig(
            perturbation_node=-1
        )


def test_linear_fit_recovers_exact_linear_relation():
    lengths = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        dtype=float,
    )

    forces = (
        3.0 * lengths
        + 2.0
    )

    metrics = _linear_fit_metrics(
        lengths=lengths,
        forces=forces,
    )

    assert metrics["slope"] == pytest.approx(
        3.0
    )
    assert metrics["intercept"] == pytest.approx(
        2.0
    )
    assert metrics["r_squared"] == pytest.approx(
        1.0
    )
    assert metrics["residual_rms"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_linear_fit_preserves_negative_correlation():
    lengths = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        dtype=float,
    )

    forces = (
        -2.0 * lengths
        + 5.0
    )

    metrics = _linear_fit_metrics(
        lengths=lengths,
        forces=forces,
    )

    assert (
        metrics["pearson_correlation"]
        == pytest.approx(-1.0)
    )


def test_linear_fit_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        _linear_fit_metrics(
            lengths=np.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=float,
            ),
            forces=np.asarray(
                [
                    1.0,
                    2.0,
                ],
                dtype=float,
            ),
        )


def test_linear_fit_requires_multiple_samples():
    with pytest.raises(ValueError):
        _linear_fit_metrics(
            lengths=np.asarray(
                [
                    1.0,
                    2.0,
                ],
                dtype=float,
            ),
            forces=np.asarray(
                [
                    2.0,
                    4.0,
                ],
                dtype=float,
            ),
        )


def test_linear_fit_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        _linear_fit_metrics(
            lengths=np.asarray(
                [
                    1.0,
                    np.nan,
                    3.0,
                ],
                dtype=float,
            ),
            forces=np.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=float,
            ),
        )


def test_audit_identifies_itself_as_external():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    assert (
        result["audit_type"]
        == "external_constitutive_coupling_audit"
    )

    assert (
        result["developmental_input_to_roif"]
        is False
    )


def test_audit_does_not_modify_rda_analyzer():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    assert result["analyzer_modified"] is False


def test_audit_preserves_audited_commit_identity():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    assert (
        result["audited_rda_2_commit"]
        == "8e182d3"
    )


def test_audit_records_member_count():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    assert (
        result["configuration"][
            "member_count"
        ]
        > 0
    )


def test_member_result_count_matches_member_count():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    assert len(
        result["member_results"]
    ) == result["configuration"][
        "member_count"
    ]


def test_each_member_result_has_expected_schema():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    expected_keys = {
        "member_index",
        "length_channel",
        "force_channel",
        "slope",
        "intercept",
        "r_squared",
        "pearson_correlation",
        "residual_rms",
        "force_rms",
        "force_variation_rms",
        "relative_residual_to_force_rms",
        "relative_residual_to_force_variation",
        "length_standard_deviation",
        "force_standard_deviation",
        "near_exact_linear_explanation",
    }

    for item in result["member_results"]:
        assert set(item) == expected_keys


def test_member_indices_are_deterministic_and_zero_based():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    indices = [
        item["member_index"]
        for item in result[
            "member_results"
        ]
    ]

    assert indices == list(
        range(
            len(indices)
        )
    )


def test_member_channel_names_follow_index_only():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    for item in result["member_results"]:
        index = item["member_index"]

        assert (
            item["length_channel"]
            == f"member_{index}_length"
        )

        assert (
            item["force_channel"]
            == f"member_{index}_axial_force"
        )


def test_r_squared_values_are_finite():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    for item in result["member_results"]:
        assert np.isfinite(
            item["r_squared"]
        )


def test_correlations_are_bounded():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    for item in result["member_results"]:
        correlation = item[
            "pearson_correlation"
        ]

        assert (
            -1.0
            <= correlation
            <= 1.0
        )


def test_residual_metrics_are_nonnegative():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    for item in result["member_results"]:
        assert item[
            "residual_rms"
        ] >= 0.0

        assert item[
            "relative_residual_to_force_rms"
        ] >= 0.0

        assert item[
            "relative_residual_to_force_variation"
        ] >= 0.0


def test_near_exact_count_is_within_member_count():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    summary = result[
        "audit_summary"
    ]

    count = summary[
        "near_exact_linear_member_count"
    ]

    member_count = result[
        "configuration"
    ]["member_count"]

    assert (
        0
        <= count
        <= member_count
    )


def test_near_exact_fraction_is_bounded():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    fraction = result[
        "audit_summary"
    ][
        "near_exact_linear_fraction"
    ]

    assert (
        0.0
        <= fraction
        <= 1.0
    )


def test_audit_does_not_require_near_exact_result():
    """
    Научный контракт не требует, чтобы локальная линейная
    зависимость обязательно объясняла результат RDA-2.
    """

    result = run_constitutive_coupling_audit(
        short_config()
    )

    assert isinstance(
        result[
            "audit_summary"
        ][
            "near_exact_linear_member_count"
        ],
        int,
    )


def test_force_change_changes_only_external_audit_condition():
    config = short_config()

    changed = replace(
        config,
        perturbation_force_x=0.5,
    )

    assert (
        changed.perturbation_force_x
        != config.perturbation_force_x
    )


def test_member_results_hash_has_sha256_shape():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    digest = result[
        "member_results_sha256"
    ]

    assert isinstance(
        digest,
        str,
    )
    assert len(digest) == 64


def test_run_hash_has_sha256_shape():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    digest = result[
        "run_sha256"
    ]

    assert isinstance(
        digest,
        str,
    )
    assert len(digest) == 64


def test_audit_is_deterministic():
    config = short_config()

    first = run_constitutive_coupling_audit(
        config
    )

    second = run_constitutive_coupling_audit(
        config
    )

    assert (
        first[
            "member_results_sha256"
        ]
        == second[
            "member_results_sha256"
        ]
    )

    assert (
        first["run_sha256"]
        == second["run_sha256"]
    )


def test_audit_result_does_not_assign_developmental_semantics():
    result = run_constitutive_coupling_audit(
        short_config()
    )

    forbidden = {
        "reward",
        "goal",
        "event_type",
        "valence",
        "meaning",
        "damage",
        "diagnosis",
        "body_part",
    }

    assert forbidden.isdisjoint(
        result.keys()
    )
