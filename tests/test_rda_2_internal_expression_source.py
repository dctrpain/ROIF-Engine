from __future__ import annotations

from experiments.roif_rda_2_first_endogenous_grouping import (
    run_first_endogenous_grouping,
)
from roif.development.endogenous_grouping import (
    GroupingResult,
)


EXPECTED_RUN_SHA256 = (
    "e42b11ca50350ef9cd250ce9eeafc9a8d"
    "051d2080dfbbb41a1ba45331c7f5a4c"
)


def test_default_rda2_run_remains_unchanged():
    result = run_first_endogenous_grouping()

    assert isinstance(result, dict)
    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    summary = result["grouping_summary"]

    assert summary["group_count"] == 12
    assert summary["relation_count"] == 435
    assert summary["channel_count"] == 30


def test_rda2_can_return_real_grouping_result():
    result, grouping = run_first_endogenous_grouping(
        return_internal=True,
    )

    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    assert isinstance(
        grouping,
        GroupingResult,
    )

    assert len(grouping.groups) == 12
    assert len(grouping.relations) == 435
    assert len(grouping.channel_names) == 30

    strongest = max(
        group.internal_relation_strength
        for group in grouping.groups
    )

    assert strongest == 1.0


def test_internal_grouping_matches_serialized_summary():
    result, grouping = run_first_endogenous_grouping(
        return_internal=True,
    )

    summary = result["grouping_summary"]

    assert (
        summary["group_count"]
        == len(grouping.groups)
    )

    assert (
        summary["relation_count"]
        == len(grouping.relations)
    )

    assert (
        summary["channel_count"]
        == len(grouping.channel_names)
    )
