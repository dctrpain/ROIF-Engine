from __future__ import annotations

from experiments.roif_rda_0_first_self_detected_change import (
    run_first_self_detected_change,
)
from roif.development.familiar_state_change import (
    ChangeDetection,
)


EXPECTED_RUN_SHA256 = (
    "9b46bf18374b069c6d671f6ec32e167be"
    "477e3b9c8324dec64b5a14c8ceeeb8c"
)


def test_default_rda0_run_remains_unchanged():
    result = run_first_self_detected_change()

    assert isinstance(result, dict)
    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    summary = result["detection_summary"]

    assert summary["detected_any_change"] is True


def test_rda0_can_return_real_first_change_detection():
    result, detection = run_first_self_detected_change(
        return_internal=True,
    )

    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    assert isinstance(
        detection,
        ChangeDetection,
    )

    assert detection.changed is True
    assert detection.sequence_index == 2000
    assert detection.timestamp == 1.9999999999998905
    assert detection.deviation_score == 999999999.9999944


def test_internal_detection_matches_serialized_first_detection():
    result, detection = run_first_self_detected_change(
        return_internal=True,
    )

    serialized = result[
        "detection_summary"
    ]["first_detection"]

    assert serialized is not None

    assert (
        serialized["sequence_index"]
        == detection.sequence_index
    )

    assert (
        serialized["timestamp"]
        == detection.timestamp
    )

    assert (
        serialized["deviation_score"]
        == detection.deviation_score
    )

    assert (
        serialized["changed"]
        == detection.changed
    )
