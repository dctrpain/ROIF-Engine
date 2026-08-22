from experiments.roif_rda_0_first_perturbation import (
    FirstPerturbationConfig,
    run_first_perturbation,
)


def small_config() -> FirstPerturbationConfig:
    return FirstPerturbationConfig(
        duration=0.5,
        dt=0.001,
        observation_interval_steps=1,
        perturbation_time=0.2,
        perturbation_force_x=1.0,
        perturbation_force_y=0.0,
        perturbation_force_z=0.0,
    )


def test_first_perturbation_changes_experienced_channels():
    result = run_first_perturbation(
        small_config()
    )

    assert (
        result["experience_summary"][
            "changed_channel_count"
        ]
        > 0
    )


def test_first_perturbation_does_not_leak_ground_truth():
    result = run_first_perturbation(
        small_config()
    )

    assert (
        result["experience_summary"][
            "ground_truth_or_semantic_leak_detected"
        ]
        is False
    )


def test_first_perturbation_has_no_reward_goal_or_event_label():
    result = run_first_perturbation(
        small_config()
    )

    assert result["reward_supplied"] is False
    assert result["goal_supplied"] is False
    assert (
        result["semantic_event_label_supplied"]
        is False
    )


def test_first_perturbation_is_deterministic():
    first = run_first_perturbation(
        small_config()
    )
    second = run_first_perturbation(
        small_config()
    )

    assert (
        first["experience_sha256"]
        == second["experience_sha256"]
    )

    assert first["run_sha256"] == second["run_sha256"]
