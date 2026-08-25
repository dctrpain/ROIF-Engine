from __future__ import annotations

from experiments.roif_rda_1_first_detected_difference import (
    run_first_detected_difference,
)
from roif.development.endogenous_difference import (
    DifferenceProfile,
)
from roif.development.familiar_state_change import (
    ChangeDetection,
)


EXPECTED_RUN_SHA256 = (
    "d8732f63d1eae4b991dce64d90a8affe"
    "447f47ae34110e4917ccea0eb4dcf6eb"
)


def test_detected_difference_bridge_is_reproducible():
    result = run_first_detected_difference()

    assert isinstance(result, dict)
    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    identity = result["identity_check"]

    assert identity["same_sequence_index"] is True
    assert identity["same_timestamp"] is True


def test_bridge_returns_real_internal_objects():
    result, detection, profile = (
        run_first_detected_difference(
            return_internal=True,
        )
    )

    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    assert isinstance(
        detection,
        ChangeDetection,
    )

    assert isinstance(
        profile,
        DifferenceProfile,
    )

    assert detection.changed is True


def test_detection_and_profile_refer_to_same_observation():
    result, detection, profile = (
        run_first_detected_difference(
            return_internal=True,
        )
    )

    assert (
        detection.sequence_index
        == profile.sequence_index
    )

    assert (
        detection.timestamp
        == profile.timestamp
    )

    matched = result["matched_internal_state"]

    assert (
        matched["sequence_index"]
        == detection.sequence_index
        == profile.sequence_index
    )

    assert (
        matched["timestamp"]
        == detection.timestamp
        == profile.timestamp
    )


def test_selection_rule_does_not_use_ground_truth():
    result = run_first_detected_difference()

    selection = result["selection_rule"]

    assert (
        selection["rule"]
        == "first_detection_changed_true"
    )

    forbidden_keys = (
        "uses_perturbation_time",
        "uses_perturbation_node",
        "uses_force_vector",
        "uses_world_coordinates",
        "uses_reward",
        "uses_goal",
        "uses_event_label",
        "uses_valence",
    )

    for key in forbidden_keys:
        assert selection[key] is False
