import copy

import pytest

from experiments.roif_rda_0_baseline_existence import (
    BENCHMARK_VERSION,
    BaselineExistenceConfig,
    run_baseline_existence,
)


def small_config() -> BaselineExistenceConfig:
    return BaselineExistenceConfig(
        duration=0.1,
        dt=0.001,
        observation_interval_steps=10,
    )


def test_config_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        BaselineExistenceConfig(
            duration=0.0,
        )


def test_config_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        BaselineExistenceConfig(
            dt=0.0,
        )


def test_config_rejects_nonpositive_observation_interval():
    with pytest.raises(ValueError):
        BaselineExistenceConfig(
            observation_interval_steps=0,
        )


def test_duration_must_be_integer_multiple_of_dt():
    config = BaselineExistenceConfig(
        duration=0.1005,
        dt=0.001,
        observation_interval_steps=10,
    )

    with pytest.raises(ValueError):
        run_baseline_existence(config)


def test_baseline_existence_has_expected_identity():
    result = run_baseline_existence(
        small_config()
    )

    assert result["benchmark"] == BENCHMARK_VERSION
    assert result["developmental_stage"] == "RDA-0"
    assert result["experience"] == "baseline_existence"


def test_no_reward_is_supplied():
    result = run_baseline_existence(
        small_config()
    )

    assert result["reward_supplied"] is False


def test_no_goal_is_supplied():
    result = run_baseline_existence(
        small_config()
    )

    assert result["goal_supplied"] is False


def test_no_external_perturbation_is_supplied():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["external_perturbation_supplied"]
        is False
    )


def test_no_interpretation_is_supplied():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["interpretation_supplied_to_roif"]
        is False
    )


def test_body_is_prestressed():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["physical_summary"][
            "initial_internal_force_norm"
        ]
        > 0.0
    )


def test_body_is_initially_self_equilibrated():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["physical_summary"][
            "maximum_initial_nodal_force_norm"
        ]
        < 1e-8
    )


def test_baseline_body_does_not_drift():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["physical_summary"][
            "maximum_node_displacement"
        ]
        < 1e-10
    )


def test_baseline_member_forces_do_not_drift():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["physical_summary"][
            "maximum_member_force_change"
        ]
        < 1e-10
    )


def test_experience_stream_contains_multiple_samples():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["experience_summary"]["sample_count"]
        > 1
    )


def test_experience_duration_matches_requested_duration():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["experience_summary"]["duration"]
        == pytest.approx(0.1)
    )


def test_experience_contains_proprioceptive_channels():
    result = run_baseline_existence(
        small_config()
    )

    first_record = result["experience_records"][0]

    assert "member_0_length" in first_record["channels"]
    assert (
        "member_0_axial_force"
        in first_record["channels"]
    )


def test_experience_does_not_contain_world_coordinates():
    result = run_baseline_existence(
        small_config()
    )

    for record in result["experience_records"]:
        channel_names = set(record["channels"])

        assert "node_0_x" not in channel_names
        assert "node_0_y" not in channel_names
        assert "node_0_z" not in channel_names
        assert "world_x" not in channel_names
        assert "world_y" not in channel_names
        assert "world_z" not in channel_names


def test_ground_truth_contains_world_coordinates():
    result = run_baseline_existence(
        small_config()
    )

    first_record = result["ground_truth_records"][0]

    values = first_record["world_state"]["values"]

    assert "node_0_x" in values
    assert "node_0_y" in values
    assert "node_0_z" in values


def test_no_ground_truth_leak_is_detected():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["experience_summary"][
            "ground_truth_leak_detected"
        ]
        is False
    )


def test_experience_and_ground_truth_hashes_are_distinct():
    result = run_baseline_existence(
        small_config()
    )

    assert (
        result["experience_sha256"]
        != result["ground_truth_sha256"]
    )


def test_run_is_deterministic():
    first = run_baseline_existence(
        small_config()
    )
    second = run_baseline_existence(
        small_config()
    )

    assert first["experience_sha256"] == (
        second["experience_sha256"]
    )

    assert first["ground_truth_sha256"] == (
        second["ground_truth_sha256"]
    )

    assert first["run_sha256"] == second["run_sha256"]


def test_experience_records_preserve_temporal_order():
    result = run_baseline_existence(
        small_config()
    )

    timestamps = [
        record["timestamp"]
        for record in result["experience_records"]
    ]

    assert timestamps == sorted(timestamps)

    assert len(set(timestamps)) == len(timestamps)


def test_run_hash_changes_if_result_content_changes():
    result = run_baseline_existence(
        small_config()
    )

    changed = copy.deepcopy(result)

    changed["experience_records"][0]["channels"][
        "member_0_length"
    ]["value"] += 0.001

    assert changed != result