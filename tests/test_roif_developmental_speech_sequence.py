from __future__ import annotations

import pytest
from experiments.roif_developmental_speech import (
    build_developmental_speech_sequence,
)


EXPECTED_STAGES = (
    "RDA-0",
    "RDA-1",
    "RDA-2",
    "RDA-3",
)

EXPECTED_EXPRESSION_KINDS = (
    "state_change",
    "difference_structure",
    "relational_structure",
    "representational_insufficiency",
)


@pytest.fixture(scope="module")
def developmental_audit():
    """
    Build one real developmental speech sequence for this module.

    All regression assertions inspect the same audit object
    produced from the frozen developmental benchmarks.
    """

    return build_developmental_speech_sequence()


def test_developmental_speech_contains_four_stages(developmental_audit):
    result = developmental_audit

    assert result["stage_count"] == 4
    assert len(result["stages"]) == 4


def test_developmental_stage_order_is_preserved(developmental_audit):
    result = developmental_audit

    stages = tuple(
        stage["stage"]
        for stage in result["stages"]
    )

    assert stages == EXPECTED_STAGES


def test_expression_kind_progression_is_preserved(developmental_audit):
    result = developmental_audit

    kinds = tuple(
        stage["speech"]["expression_kind"]
        for stage in result["stages"]
    )

    assert kinds == EXPECTED_EXPRESSION_KINDS


def test_all_developmental_expressions_are_accepted(developmental_audit):
    result = developmental_audit

    for stage in result["stages"]:
        speech = stage["speech"]

        assert speech["validation_status"] == "accepted"
        assert speech["validation_reasons"] == []
        assert speech["final_expression"]


def test_rda1_preserves_detection_profile_identity(developmental_audit):
    result = developmental_audit

    rda1 = result["stages"][1]

    assert rda1["stage"] == "RDA-1"

    assert (
        rda1["identity_check"]["same_sequence_index"]
        is True
    )
    assert (
        rda1["identity_check"]["same_timestamp"]
        is True
    )


def test_rda1_uses_endogenous_selection_only(developmental_audit):
    result = developmental_audit

    rda1 = result["stages"][1]
    rule = rda1["selection_rule"]

    assert rule["uses_event_label"] is False
    assert rule["uses_force_vector"] is False
    assert rule["uses_goal"] is False
    assert rule["uses_perturbation_node"] is False
    assert rule["uses_perturbation_time"] is False
    assert rule["uses_reward"] is False
    assert rule["uses_valence"] is False
    assert rule["uses_world_coordinates"] is False


def test_llm_does_not_create_developmental_state(developmental_audit):
    result = developmental_audit

    boundary = result["architectural_boundary"]

    assert boundary["llm_creates_developmental_state"] is False
    assert (
        boundary["llm_role"]
        == "validated_verbalization_only"
    )
    assert (
        boundary["speech_reads_benchmark_ground_truth"]
        is False
    )
    assert (
        boundary["speech_reads_real_internal_objects"]
        is True
    )


def test_continuity_claim_remains_conservative(developmental_audit):
    result = developmental_audit

    assert (
        result["continuity_claim"]
        == "independent_frozen_developmental_benchmarks"
    )


def test_builder_does_not_write_to_stdout(capsys):
    audit = build_developmental_speech_sequence()

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert audit["stage_count"] == 4
