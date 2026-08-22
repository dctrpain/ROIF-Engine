from __future__ import annotations

import pytest

from experiments.roif_rda_0_first_self_detected_change import (
    FROZEN_ABSOLUTE_SCALE_FLOOR,
    FROZEN_DETECTOR_COMMIT,
    FROZEN_THRESHOLD,
    FirstSelfDetectedChangeConfig,
    run_first_self_detected_change,
)


def test_default_configuration_places_baseline_before_perturbation():
    config = FirstSelfDetectedChangeConfig()

    assert (
        0.0
        < config.baseline_duration
        < config.perturbation_time
        < config.duration
    )


def test_configuration_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        FirstSelfDetectedChangeConfig(
            dt=0.0
        )


def test_configuration_rejects_perturbation_during_baseline():
    with pytest.raises(ValueError):
        FirstSelfDetectedChangeConfig(
            baseline_duration=1.0,
            perturbation_time=0.5,
        )


def test_configuration_rejects_perturbation_at_baseline_boundary():
    with pytest.raises(ValueError):
        FirstSelfDetectedChangeConfig(
            baseline_duration=1.0,
            perturbation_time=1.0,
        )


def test_configuration_rejects_negative_node():
    with pytest.raises(ValueError):
        FirstSelfDetectedChangeConfig(
            perturbation_node=-1
        )


def test_frozen_detector_identity_is_explicit():
    assert FROZEN_DETECTOR_COMMIT == "28deb00"
    assert FROZEN_THRESHOLD == pytest.approx(8.0)
    assert FROZEN_ABSOLUTE_SCALE_FLOOR == pytest.approx(
        1e-9
    )


def test_experiment_preserves_frozen_detector_parameters():
    result = run_first_self_detected_change(
        FirstSelfDetectedChangeConfig(
            duration=0.05,
            dt=0.001,
            baseline_duration=0.01,
            perturbation_time=0.03,
        )
    )

    frozen = result["frozen_detector"]

    assert frozen["commit"] == "28deb00"
    assert frozen["threshold"] == pytest.approx(8.0)
    assert frozen["absolute_scale_floor"] == pytest.approx(
        1e-9
    )
    assert frozen["parameters_modified_after_freeze"] is False


def test_detector_epistemic_boundary_excludes_ground_truth():
    result = run_first_self_detected_change(
        FirstSelfDetectedChangeConfig(
            duration=0.05,
            dt=0.001,
            baseline_duration=0.01,
            perturbation_time=0.03,
        )
    )

    boundary = result["epistemic_boundary"]

    assert boundary == {
        "detector_received_ground_truth": False,
        "detector_received_perturbation_time": False,
        "detector_received_perturbation_node": False,
        "detector_received_force_vector": False,
        "detector_received_world_coordinates": False,
        "detector_received_reward": False,
        "detector_received_goal": False,
        "detector_received_event_label": False,
        "detector_received_valence": False,
    }


def test_ground_truth_is_separated_as_evaluation_only():
    config = FirstSelfDetectedChangeConfig(
        duration=0.05,
        dt=0.001,
        baseline_duration=0.01,
        perturbation_time=0.03,
    )

    result = run_first_self_detected_change(
        config
    )

    ground_truth = result[
        "ground_truth_evaluation_only"
    ]

    assert ground_truth["perturbation_time"] == pytest.approx(
        config.perturbation_time
    )
    assert ground_truth["perturbation_node"] == (
        config.perturbation_node
    )


def test_experiment_is_deterministic():
    config = FirstSelfDetectedChangeConfig(
        duration=0.05,
        dt=0.001,
        baseline_duration=0.01,
        perturbation_time=0.03,
    )

    first = run_first_self_detected_change(
        config
    )
    second = run_first_self_detected_change(
        config
    )

    assert (
        first["detection_records_sha256"]
        == second["detection_records_sha256"]
    )

    assert (
        first["run_sha256"]
        == second["run_sha256"]
    )


def test_experiment_records_observations_after_baseline():
    config = FirstSelfDetectedChangeConfig(
        duration=0.05,
        dt=0.001,
        baseline_duration=0.01,
        perturbation_time=0.03,
    )

    result = run_first_self_detected_change(
        config
    )

    records = result["detection_records"]

    assert records
    assert records[0]["timestamp"] > (
        config.baseline_duration
    )


def test_detection_records_contain_only_detector_outputs():
    config = FirstSelfDetectedChangeConfig(
        duration=0.05,
        dt=0.001,
        baseline_duration=0.01,
        perturbation_time=0.03,
    )

    result = run_first_self_detected_change(
        config
    )

    allowed_keys = {
        "sequence_index",
        "timestamp",
        "deviation_score",
        "changed",
    }

    for record in result["detection_records"]:
        assert set(record) == allowed_keys


def test_contract_does_not_require_successful_detection():
    """
    Pre-exposure scientific contract:

    This test deliberately does NOT assert that a change must be detected.
    A positive detection, false positive, or failure to detect remains an
    empirical result of the first full exposure.
    """

    config = FirstSelfDetectedChangeConfig(
        duration=0.05,
        dt=0.001,
        baseline_duration=0.01,
        perturbation_time=0.03,
    )

    result = run_first_self_detected_change(
        config
    )

    detected = result[
        "detection_summary"
    ]["detected_any_change"]

    assert isinstance(
        detected,
        bool,
    )
