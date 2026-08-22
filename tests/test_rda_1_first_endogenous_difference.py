from __future__ import annotations

from dataclasses import replace

from experiments.roif_rda_1_first_endogenous_difference import (
    BENCHMARK_VERSION,
    FROZEN_ABSOLUTE_SCALE_FLOOR,
    FROZEN_ANALYZER_COMMIT,
    FirstEndogenousDifferenceConfig,
    run_first_endogenous_difference,
)


def short_config() -> FirstEndogenousDifferenceConfig:
    """
    Small deterministic contract run.

    This configuration preserves the ordering:

        baseline
            <
        perturbation
            <
        end

    without asserting anything about which experienced channels
    should become strongest.
    """

    return FirstEndogenousDifferenceConfig(
        duration=0.06,
        dt=0.001,
        baseline_duration=0.02,
        perturbation_time=0.04,
        perturbation_node=0,
        perturbation_force_x=1.0,
        perturbation_force_y=0.0,
        perturbation_force_z=0.0,
    )


def test_benchmark_identity_is_stable():
    assert (
        BENCHMARK_VERSION
        == "roif_rda_1_first_endogenous_difference_v1"
    )


def test_frozen_analyzer_commit_is_explicit():
    assert FROZEN_ANALYZER_COMMIT == "0aee759"


def test_frozen_scale_floor_is_explicit():
    assert FROZEN_ABSOLUTE_SCALE_FLOOR == 1e-9


def test_configuration_requires_baseline_before_perturbation():
    config = short_config()

    assert (
        config.baseline_duration
        < config.perturbation_time
        < config.duration
    )


def test_run_identifies_rda_1():
    result = run_first_endogenous_difference(
        short_config()
    )

    assert result["developmental_stage"] == "RDA-1"
    assert (
        result["experience"]
        == "first_endogenous_difference"
    )


def test_run_uses_frozen_analyzer_configuration():
    result = run_first_endogenous_difference(
        short_config()
    )

    frozen = result["frozen_analyzer"]

    assert frozen["commit"] == "0aee759"
    assert (
        frozen["absolute_scale_floor"]
        == 1e-9
    )
    assert (
        frozen["parameters_modified_after_freeze"]
        is False
    )


def test_analyzer_epistemic_boundary_is_explicit():
    result = run_first_endogenous_difference(
        short_config()
    )

    boundary = result["epistemic_boundary"]

    assert boundary
    assert all(
        value is False
        for value in boundary.values()
    )


def test_ground_truth_is_separated_as_evaluation_only():
    result = run_first_endogenous_difference(
        short_config()
    )

    evaluation = result[
        "ground_truth_evaluation_only"
    ]

    assert evaluation["perturbation_step"] == 40
    assert evaluation["perturbation_node"] == 0
    assert evaluation["perturbation_force"] == [
        1.0,
        0.0,
        0.0,
    ]


def test_profile_count_matches_post_baseline_steps():
    config = short_config()

    result = run_first_endogenous_difference(
        config
    )

    total_steps = round(
        config.duration / config.dt
    )
    baseline_steps = round(
        config.baseline_duration / config.dt
    )

    assert (
        result["configuration"]["profile_count"]
        == total_steps - baseline_steps
    )


def test_baseline_contains_only_preperturbation_experience():
    config = short_config()

    result = run_first_endogenous_difference(
        config
    )

    baseline_last_step = round(
        config.baseline_duration / config.dt
    )
    perturbation_step = result[
        "ground_truth_evaluation_only"
    ]["perturbation_step"]

    assert baseline_last_step < perturbation_step


def test_perturbation_profile_exists():
    result = run_first_endogenous_difference(
        short_config()
    )

    summary = result[
        "perturbation_profile_summary"
    ]

    assert summary["channel_count"] > 0
    assert (
        summary["maximum_absolute_deviation"]
        >= 0.0
    )
    assert summary["l2_deviation_norm"] >= 0.0


def test_perturbation_profile_accounts_for_all_channels():
    result = run_first_endogenous_difference(
        short_config()
    )

    summary = result[
        "perturbation_profile_summary"
    ]

    accounted = (
        summary["positive_signed_channel_count"]
        + summary["negative_signed_channel_count"]
        + summary[
            "approximately_unchanged_channel_count"
        ]
    )

    assert accounted == summary["channel_count"]


def test_top_channels_are_not_predeclared_by_contract():
    result = run_first_endogenous_difference(
        short_config()
    )

    top = result[
        "perturbation_profile_summary"
    ]["top_10_strongest_channels"]

    assert len(top) > 0

    for item in top:
        assert "channel_name" in item
        assert "signed_deviation" in item
        assert "absolute_deviation" in item


def test_profile_records_preserve_channel_level_structure():
    result = run_first_endogenous_difference(
        short_config()
    )

    records = result["profile_records"]

    assert records

    first = records[0]

    assert first["channel_deviations"]
    assert first["strongest_channels"]

    channel_names = {
        item["channel_name"]
        for item in first["channel_deviations"]
    }

    strongest_names = {
        item["channel_name"]
        for item in first["strongest_channels"]
    }

    assert channel_names == strongest_names


def test_strongest_channels_are_sorted_by_absolute_deviation():
    result = run_first_endogenous_difference(
        short_config()
    )

    for record in result["profile_records"]:
        values = [
            item["absolute_deviation"]
            for item in record["strongest_channels"]
        ]

        assert values == sorted(
            values,
            reverse=True,
        )


def test_profile_records_have_stable_hash():
    result = run_first_endogenous_difference(
        short_config()
    )

    digest = result["profile_records_sha256"]

    assert isinstance(digest, str)
    assert len(digest) == 64


def test_run_has_stable_hash():
    result = run_first_endogenous_difference(
        short_config()
    )

    digest = result["run_sha256"]

    assert isinstance(digest, str)
    assert len(digest) == 64


def test_run_is_deterministic():
    config = short_config()

    first = run_first_endogenous_difference(
        config
    )
    second = run_first_endogenous_difference(
        config
    )

    assert (
        first["profile_records_sha256"]
        == second["profile_records_sha256"]
    )

    assert (
        first["run_sha256"]
        == second["run_sha256"]
    )


def test_force_change_changes_experimenter_condition():
    config = short_config()

    changed = replace(
        config,
        perturbation_force_x=0.5,
    )

    assert (
        changed.perturbation_force_x
        != config.perturbation_force_x
    )


def test_result_does_not_assign_event_semantics():
    result = run_first_endogenous_difference(
        short_config()
    )

    forbidden_top_level_keys = {
        "reward",
        "goal",
        "event_type",
        "valence",
        "meaning",
        "damage",
        "diagnosis",
    }

    assert forbidden_top_level_keys.isdisjoint(
        result.keys()
    )


def test_profile_records_do_not_contain_ground_truth_fields():
    result = run_first_endogenous_difference(
        short_config()
    )

    forbidden = {
        "ground_truth",
        "world_state",
        "perturbation_node",
        "perturbation_force",
        "external_force",
        "world_x",
        "world_y",
        "world_z",
        "reward",
        "goal",
        "event_type",
        "valence",
    }

    for record in result["profile_records"]:
        assert forbidden.isdisjoint(
            record.keys()
        )


def test_no_expected_channel_identity_is_encoded_in_contract():
    """
    The contract must not assert that any named proprioceptive channel
    is supposed to dominate the first endogenous difference.
    """

    result = run_first_endogenous_difference(
        short_config()
    )

    top = result[
        "perturbation_profile_summary"
    ]["top_10_strongest_channels"]

    assert all(
        isinstance(
            item["channel_name"],
            str,
        )
        and item["channel_name"]
        for item in top
    )
