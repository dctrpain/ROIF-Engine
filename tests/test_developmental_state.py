from __future__ import annotations

from roif.development.developmental_state import (
    DevelopmentalState,
)


def test_new_developmental_state_is_empty():
    state = DevelopmentalState()

    assert state.observation_count == 0
    assert state.change_detection_count == 0
    assert state.difference_profile_count == 0
    assert state.grouping_result_count == 0
    assert state.dimensional_growth_assessment_count == 0

    assert state.latest_change_detection is None
    assert state.latest_difference_profile is None
    assert state.latest_grouping_result is None
    assert state.latest_dimensional_growth_assessment is None


def test_state_has_no_linear_stage_controller():
    state = DevelopmentalState()

    forbidden = (
        "current_stage",
        "next_stage",
        "expected_stage",
        "transition_map",
        "advance",
        "run_rda0",
        "run_rda1",
        "run_rda2",
        "run_rda3",
    )

    for name in forbidden:
        assert not hasattr(state, name)


def test_state_exposes_memory_not_stage_progression():
    state = DevelopmentalState()

    assert hasattr(state, "observations")
    assert hasattr(state, "change_detections")
    assert hasattr(state, "difference_profiles")
    assert hasattr(state, "grouping_results")
    assert hasattr(
        state,
        "dimensional_growth_assessments",
    )


def test_record_methods_exist_without_transition_logic():
    state = DevelopmentalState()

    assert callable(state.record_observation)
    assert callable(state.record_change_detection)
    assert callable(state.record_difference_profile)
    assert callable(state.record_grouping_result)
    assert callable(
        state.record_dimensional_growth_assessment
    )


def test_state_can_hold_real_rda0_internal_change():
    from experiments.roif_rda_0_first_self_detected_change import (
        run_first_self_detected_change,
    )

    state = DevelopmentalState()

    result, detection = run_first_self_detected_change(
        return_internal=True,
    )

    state.record_change_detection(
        detection
    )

    assert state.change_detection_count == 1
    assert state.latest_change_detection is detection

    assert detection.changed is True
    assert (
        result["run_sha256"]
        == "9b46bf18374b069c6d671f6ec32e167be477e3b9c8324dec64b5a14c8ceeeb8c"
    )


def test_state_can_hold_real_rda0_and_rda1_internal_objects_together():
    from experiments.roif_rda_0_first_self_detected_change import (
        run_first_self_detected_change,
    )
    from experiments.roif_rda_1_first_detected_difference import (
        run_first_detected_difference,
    )

    state = DevelopmentalState()

    _, rda0_detection = run_first_self_detected_change(
        return_internal=True,
    )

    (
        rda1_result,
        rda1_detection,
        rda1_profile,
    ) = run_first_detected_difference(
        return_internal=True,
    )

    state.record_change_detection(
        rda0_detection
    )

    state.record_change_detection(
        rda1_detection
    )

    state.record_difference_profile(
        rda1_profile
    )

    assert state.change_detection_count == 2
    assert state.difference_profile_count == 1

    assert (
        state.latest_change_detection
        is rda1_detection
    )

    assert (
        state.latest_difference_profile
        is rda1_profile
    )

    assert (
        rda1_detection.sequence_index
        == rda1_profile.sequence_index
    )

    assert (
        rda1_detection.timestamp
        == rda1_profile.timestamp
    )

    assert (
        rda1_result["identity_check"][
            "same_sequence_index"
        ]
        is True
    )

    assert (
        rda1_result["identity_check"][
            "same_timestamp"
        ]
        is True
    )

    assert not hasattr(
        state,
        "next_stage",
    )

    assert not hasattr(
        state,
        "advance",
    )


def test_state_can_hold_real_rda2_grouping_result():
    from experiments.roif_rda_2_first_endogenous_grouping import (
        run_first_endogenous_grouping,
    )

    state = DevelopmentalState()

    result, grouping = run_first_endogenous_grouping(
        return_internal=True,
    )

    state.record_grouping_result(
        grouping
    )

    assert state.grouping_result_count == 1
    assert state.latest_grouping_result is grouping

    assert len(grouping.channel_names) > 0
    assert len(grouping.groups) > 0

    assert (
        result["run_sha256"]
        == "e42b11ca50350ef9cd250ce9eeafc9a8d051d2080dfbbb41a1ba45331c7f5a4c"
    )


def test_state_can_hold_real_rda3_growth_assessment():
    from experiments.roif_rda_3_first_dimensional_growth import (
        run_first_dimensional_growth,
    )

    state = DevelopmentalState()

    result, assessment = run_first_dimensional_growth(
        return_internal=True,
    )

    state.record_dimensional_growth_assessment(
        assessment
    )

    assert (
        state.dimensional_growth_assessment_count
        == 1
    )

    assert (
        state.latest_dimensional_growth_assessment
        is assessment
    )

    assert assessment.growth_supported is True

    assert (
        result["run_sha256"]
        == "3423ebb7346656f6336aed9a68d05bbbf5913763c0612f5c91dee656b47ab24a"
    )


def test_one_state_can_hold_all_existing_developmental_artifacts():
    from experiments.roif_rda_0_first_self_detected_change import (
        run_first_self_detected_change,
    )
    from experiments.roif_rda_1_first_detected_difference import (
        run_first_detected_difference,
    )
    from experiments.roif_rda_2_first_endogenous_grouping import (
        run_first_endogenous_grouping,
    )
    from experiments.roif_rda_3_first_dimensional_growth import (
        run_first_dimensional_growth,
    )

    state = DevelopmentalState()

    _, rda0_detection = run_first_self_detected_change(
        return_internal=True,
    )

    (
        _,
        rda1_detection,
        rda1_profile,
    ) = run_first_detected_difference(
        return_internal=True,
    )

    _, rda2_grouping = run_first_endogenous_grouping(
        return_internal=True,
    )

    _, rda3_assessment = run_first_dimensional_growth(
        return_internal=True,
    )

    state.record_change_detection(
        rda0_detection
    )

    state.record_change_detection(
        rda1_detection
    )

    state.record_difference_profile(
        rda1_profile
    )

    state.record_grouping_result(
        rda2_grouping
    )

    state.record_dimensional_growth_assessment(
        rda3_assessment
    )

    assert state.change_detection_count == 2
    assert state.difference_profile_count == 1
    assert state.grouping_result_count == 1
    assert (
        state.dimensional_growth_assessment_count
        == 1
    )

    assert state.latest_change_detection is rda1_detection
    assert state.latest_difference_profile is rda1_profile
    assert state.latest_grouping_result is rda2_grouping
    assert (
        state.latest_dimensional_growth_assessment
        is rda3_assessment
    )

    forbidden = (
        "current_stage",
        "next_stage",
        "expected_stage",
        "transition_map",
        "advance",
        "run_rda0",
        "run_rda1",
        "run_rda2",
        "run_rda3",
    )

    for name in forbidden:
        assert not hasattr(state, name)
