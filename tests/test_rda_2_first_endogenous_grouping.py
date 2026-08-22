from __future__ import annotations

from dataclasses import replace

from experiments.roif_rda_2_first_endogenous_grouping import (
    BENCHMARK_VERSION,
    FROZEN_DIFFERENCE_ANALYZER_COMMIT,
    FROZEN_DIFFERENCE_SCALE_FLOOR,
    FROZEN_GROUPING_ANALYZER_COMMIT,
    FROZEN_MINIMUM_TEMPORAL_VARIATION,
    FROZEN_RELATION_THRESHOLD,
    FirstEndogenousGroupingConfig,
    run_first_endogenous_grouping,
)


def short_config() -> FirstEndogenousGroupingConfig:
    return FirstEndogenousGroupingConfig(
        duration=0.08,
        dt=0.001,
        baseline_duration=0.02,
        perturbation_time=0.04,
        grouping_window_duration=0.02,
        perturbation_node=0,
        perturbation_force_x=1.0,
        perturbation_force_y=0.0,
        perturbation_force_z=0.0,
    )


def test_benchmark_identity_is_stable():
    assert (
        BENCHMARK_VERSION
        == "roif_rda_2_first_endogenous_grouping_v1"
    )


def test_frozen_component_commits_are_explicit():
    assert (
        FROZEN_DIFFERENCE_ANALYZER_COMMIT
        == "0aee759"
    )
    assert (
        FROZEN_GROUPING_ANALYZER_COMMIT
        == "88ab5cf"
    )


def test_frozen_parameters_are_explicit():
    assert (
        FROZEN_DIFFERENCE_SCALE_FLOOR
        == 1e-9
    )
    assert (
        FROZEN_RELATION_THRESHOLD
        == 0.95
    )
    assert (
        FROZEN_MINIMUM_TEMPORAL_VARIATION
        == 1e-12
    )


def test_configuration_orders_baseline_perturbation_and_end():
    config = short_config()

    assert (
        0.0
        < config.baseline_duration
        < config.perturbation_time
        < config.duration
    )


def test_grouping_window_fits_inside_run():
    config = short_config()

    assert (
        config.perturbation_time
        + config.grouping_window_duration
        <= config.duration
    )


def test_run_identifies_rda_2():
    result = run_first_endogenous_grouping(
        short_config()
    )

    assert (
        result["developmental_stage"]
        == "RDA-2"
    )
    assert (
        result["experience"]
        == "first_endogenous_grouping"
    )


def test_run_preserves_frozen_component_identity():
    result = run_first_endogenous_grouping(
        short_config()
    )

    frozen = result["frozen_components"]

    assert (
        frozen["difference_analyzer_commit"]
        == "0aee759"
    )
    assert (
        frozen["grouping_analyzer_commit"]
        == "88ab5cf"
    )
    assert (
        frozen["parameters_modified_after_freeze"]
        is False
    )


def test_epistemic_boundary_contains_only_false_values():
    result = run_first_endogenous_grouping(
        short_config()
    )

    boundary = result[
        "epistemic_boundary"
    ]

    assert boundary
    assert all(
        value is False
        for value in boundary.values()
    )


def test_ground_truth_is_separated_for_evaluation_only():
    result = run_first_endogenous_grouping(
        short_config()
    )

    evaluation = result[
        "ground_truth_evaluation_only"
    ]

    assert (
        evaluation["perturbation_step"]
        == 40
    )
    assert (
        evaluation["perturbation_node"]
        == 0
    )
    assert (
        evaluation["perturbation_force"]
        == [
            1.0,
            0.0,
            0.0,
        ]
    )


def test_grouping_profile_count_matches_window():
    config = short_config()

    result = run_first_endogenous_grouping(
        config
    )

    expected_window_steps = round(
        config.grouping_window_duration
        / config.dt
    )

    assert (
        result["configuration"][
            "grouping_profile_count"
        ]
        == expected_window_steps + 1
    )


def test_grouping_window_starts_on_perturbation_step():
    result = run_first_endogenous_grouping(
        short_config()
    )

    evaluation = result[
        "ground_truth_evaluation_only"
    ]

    assert (
        evaluation[
            "grouping_window_start_step"
        ]
        == evaluation[
            "perturbation_step"
        ]
    )


def test_grouping_window_step_indices_are_monotonic():
    result = run_first_endogenous_grouping(
        short_config()
    )

    indices = result[
        "grouping_profile_step_indices"
    ]

    assert indices == sorted(
        indices
    )
    assert len(indices) == len(
        set(indices)
    )


def test_relation_count_matches_complete_pairwise_graph():
    result = run_first_endogenous_grouping(
        short_config()
    )

    channel_count = result[
        "grouping_summary"
    ]["channel_count"]

    expected_relations = (
        channel_count
        * (channel_count - 1)
        // 2
    )

    assert (
        result[
            "grouping_summary"
        ]["relation_count"]
        == expected_relations
    )


def test_group_sizes_match_group_records():
    result = run_first_endogenous_grouping(
        short_config()
    )

    recorded_sizes = [
        group["channel_count"]
        for group in result["groups"]
    ]

    assert (
        result[
            "grouping_summary"
        ]["group_sizes"]
        == recorded_sizes
    )


def test_grouped_and_ungrouped_channels_account_for_all_channels():
    result = run_first_endogenous_grouping(
        short_config()
    )

    summary = result[
        "grouping_summary"
    ]

    assert (
        summary["grouped_channel_count"]
        + summary["ungrouped_channel_count"]
        == summary["channel_count"]
    )


def test_group_records_have_no_body_part_semantics():
    result = run_first_endogenous_grouping(
        short_config()
    )

    forbidden = {
        "body_part",
        "anatomy",
        "node_id",
        "world_x",
        "world_y",
        "world_z",
        "event_type",
        "reward",
        "goal",
        "valence",
    }

    for group in result["groups"]:
        assert forbidden.isdisjoint(
            group.keys()
        )


def test_relation_records_have_no_ground_truth_fields():
    result = run_first_endogenous_grouping(
        short_config()
    )

    forbidden = {
        "ground_truth",
        "world_state",
        "topology",
        "node_mapping",
        "member_mapping",
        "perturbation_node",
        "perturbation_force",
        "external_force",
        "reward",
        "goal",
        "event_type",
        "valence",
    }

    for relation in result["relations"]:
        assert forbidden.isdisjoint(
            relation.keys()
        )


def test_relation_records_preserve_sign_and_absolute_similarity():
    result = run_first_endogenous_grouping(
        short_config()
    )

    for relation in result["relations"]:
        assert (
            relation[
                "absolute_similarity"
            ]
            == abs(
                relation[
                    "similarity"
                ]
            )
        )


def test_groups_are_deterministically_numbered():
    result = run_first_endogenous_grouping(
        short_config()
    )

    ids = [
        group["group_id"]
        for group in result["groups"]
    ]

    assert ids == list(
        range(
            len(ids)
        )
    )


def test_group_channel_names_are_sorted_and_unique():
    result = run_first_endogenous_grouping(
        short_config()
    )

    for group in result["groups"]:
        names = group[
            "channel_names"
        ]

        assert names == sorted(
            names
        )
        assert len(names) == len(
            set(names)
        )


def test_internal_group_strength_is_bounded():
    result = run_first_endogenous_grouping(
        short_config()
    )

    for group in result["groups"]:
        assert (
            0.0
            <= group[
                "internal_relation_strength"
            ]
            <= 1.0
        )


def test_strongest_relation_summaries_are_schema_only():
    result = run_first_endogenous_grouping(
        short_config()
    )

    summary = result[
        "grouping_summary"
    ]

    for key in (
        "strongest_positive_relations",
        "strongest_negative_relations",
    ):
        for item in summary[key]:
            assert set(item) == {
                "channel_a",
                "channel_b",
                "similarity",
                "absolute_similarity",
                "sample_count",
            }


def test_hashes_have_expected_sha256_shape():
    result = run_first_endogenous_grouping(
        short_config()
    )

    for key in (
        "relations_sha256",
        "groups_sha256",
        "grouping_window_sha256",
        "run_sha256",
    ):
        digest = result[key]

        assert isinstance(
            digest,
            str,
        )
        assert len(digest) == 64


def test_run_is_deterministic():
    config = short_config()

    first = run_first_endogenous_grouping(
        config
    )
    second = run_first_endogenous_grouping(
        config
    )

    assert (
        first["relations_sha256"]
        == second["relations_sha256"]
    )
    assert (
        first["groups_sha256"]
        == second["groups_sha256"]
    )
    assert (
        first["grouping_window_sha256"]
        == second["grouping_window_sha256"]
    )
    assert (
        first["run_sha256"]
        == second["run_sha256"]
    )


def test_force_change_changes_only_experimenter_condition():
    config = short_config()

    changed = replace(
        config,
        perturbation_force_x=0.5,
    )

    assert (
        changed.perturbation_force_x
        != config.perturbation_force_x
    )


def test_contract_does_not_predeclare_expected_group_membership():
    result = run_first_endogenous_grouping(
        short_config()
    )

    for group in result["groups"]:
        for channel_name in group[
            "channel_names"
        ]:
            assert isinstance(
                channel_name,
                str,
            )
            assert channel_name


def test_contract_does_not_require_any_group_to_exist():
    """
    A first full RDA-2 exposure is allowed to produce zero groups.

    The scientific contract validates the pipeline and epistemic boundary,
    not a desired grouping outcome.
    """

    result = run_first_endogenous_grouping(
        short_config()
    )

    assert isinstance(
        result[
            "grouping_summary"
        ]["group_count"],
        int,
    )


def test_top_level_result_does_not_assign_event_semantics():
    result = run_first_endogenous_grouping(
        short_config()
    )

    forbidden = {
        "meaning",
        "damage",
        "diagnosis",
        "body_part",
        "event_type",
        "reward",
        "goal",
        "valence",
    }

    assert forbidden.isdisjoint(
        result.keys()
    )
