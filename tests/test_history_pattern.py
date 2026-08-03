from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.event import (
    HistoryTarget,
    HistoryTargetKind,
    ReversibilityClass,
    StateDelta,
    TimeScale,
)
from roif.history.history_pattern import (
    HistoryPattern,
    HistoryPatternError,
    RankedContribution,
)
from roif.history.irreversible_change import (
    FunctionalEffect,
    IrreversibleChange,
    IrreversibleChangeKind,
    TracePersistence,
)


def make_target(
    target_id: str = "ELEMENT_A",
    *,
    kind: HistoryTargetKind = HistoryTargetKind.ELEMENT,
) -> HistoryTarget:
    return HistoryTarget(
        kind=kind,
        target_id=target_id,
        label=target_id,
    )


def make_change(
    change_id: str,
    *,
    kind: IrreversibleChangeKind = (
        IrreversibleChangeKind.DAMAGE
    ),
    target_id: str = "ELEMENT_A",
    onset_time: float = 0.0,
    recorded_time: float | None = None,
    retained_fraction: float = 0.80,
    memory_strength: float = 0.75,
    permanence: TracePersistence = (
        TracePersistence.LONG_LIVED
    ),
    time_scale: TimeScale = TimeScale.MEDIUM,
    capacity_effect: float = -0.10,
    functional_effect: FunctionalEffect = (
        FunctionalEffect.HARMFUL
    ),
    reversibility: ReversibilityClass = (
        ReversibilityClass.PARTIALLY_REVERSIBLE
    ),
    cause_change_ids: tuple[str, ...] = (),
    plane_ids: tuple[str, ...] = ("mechanical",),
    agent_ids: tuple[str, ...] = ("external_load",),
    quantity: str = "damage",
    before: float = 0.0,
    after: float = 0.2,
) -> IrreversibleChange:
    if recorded_time is None:
        recorded_time = onset_time

    return IrreversibleChange(
        change_id=change_id,
        kind=kind,
        source_event_id=f"event-{change_id}",
        target=make_target(target_id),
        delta=StateDelta(
            quantity=quantity,
            before=before,
            after=after,
            units="normalized",
        ),
        onset_time=onset_time,
        recorded_time=recorded_time,
        retained_fraction=retained_fraction,
        permanence=permanence,
        characteristic_time=60.0,
        time_scale=time_scale,
        memory_strength=memory_strength,
        capacity_effect=capacity_effect,
        functional_effect=functional_effect,
        reversibility=reversibility,
        cause_change_ids=cause_change_ids,
        plane_ids=plane_ids,
        agent_ids=agent_ids,
    )


def make_chain() -> tuple[
    IrreversibleChange,
    IrreversibleChange,
    IrreversibleChange,
]:
    root = make_change(
        "change-a",
        onset_time=1.0,
        recorded_time=1.1,
        target_id="ELEMENT_A",
        plane_ids=("mechanical",),
        agent_ids=("steel_object",),
        capacity_effect=-0.10,
    )

    middle = make_change(
        "change-b",
        onset_time=2.0,
        recorded_time=2.2,
        target_id="ELEMENT_B",
        cause_change_ids=("change-a",),
        plane_ids=("mechanical", "thermal"),
        agent_ids=("steel_object",),
        capacity_effect=-0.15,
    )

    leaf = make_change(
        "change-c",
        onset_time=3.0,
        recorded_time=3.4,
        target_id="ELEMENT_C",
        cause_change_ids=("change-b",),
        plane_ids=("thermal",),
        agent_ids=("environment",),
        capacity_effect=-0.20,
    )

    return root, middle, leaf


def make_pattern() -> HistoryPattern:
    root, middle, leaf = make_chain()

    return HistoryPattern(
        pattern_id="pattern-001",
        changes=(
            leaf,
            root,
            middle,
        ),
        label="Progressive damage pattern",
        description="Three-stage causal history.",
        metadata={
            "system": "test-network",
        },
    )


def test_empty_pattern_creation() -> None:
    pattern = HistoryPattern(
        pattern_id="empty-pattern",
    )

    assert pattern.is_empty is True
    assert pattern.total_changes == 0
    assert pattern.first_change is None
    assert pattern.last_change is None
    assert pattern.start_time == pytest.approx(0.0)
    assert pattern.end_time == pytest.approx(0.0)
    assert pattern.duration == pytest.approx(0.0)


def test_pattern_creation() -> None:
    pattern = make_pattern()

    assert pattern.pattern_id == "pattern-001"
    assert pattern.label == "Progressive damage pattern"
    assert pattern.total_changes == 3
    assert pattern.metadata["system"] == "test-network"


def test_pattern_is_immutable() -> None:
    pattern = make_pattern()

    with pytest.raises(FrozenInstanceError):
        pattern.label = "Changed"


def test_pattern_metadata_is_read_only() -> None:
    pattern = make_pattern()

    with pytest.raises(TypeError):
        pattern.metadata["new"] = "value"


def test_changes_are_sorted_chronologically() -> None:
    pattern = make_pattern()

    assert tuple(
        change.change_id
        for change in pattern.changes
    ) == (
        "change-a",
        "change-b",
        "change-c",
    )


def test_len_and_iteration() -> None:
    pattern = make_pattern()

    assert len(pattern) == 3
    assert tuple(
        change.change_id
        for change in pattern
    ) == (
        "change-a",
        "change-b",
        "change-c",
    )


def test_contains_change_id() -> None:
    pattern = make_pattern()

    assert "change-a" in pattern
    assert "missing" not in pattern
    assert 123 not in pattern


def test_first_and_last_change() -> None:
    pattern = make_pattern()

    assert pattern.first_change is not None
    assert pattern.last_change is not None
    assert pattern.first_change.change_id == "change-a"
    assert pattern.last_change.change_id == "change-c"


def test_pattern_time_range() -> None:
    pattern = make_pattern()

    assert pattern.start_time == pytest.approx(1.0)
    assert pattern.end_time == pytest.approx(3.4)
    assert pattern.duration == pytest.approx(2.4)


def test_chronology_returns_ordered_changes() -> None:
    pattern = make_pattern()

    assert pattern.chronology() == pattern.changes


def test_change_by_id() -> None:
    pattern = make_pattern()

    found = pattern.change_by_id("change-b")

    assert found is not None
    assert found.target.target_id == "ELEMENT_B"
    assert pattern.change_by_id("missing") is None


def test_root_changes() -> None:
    pattern = make_pattern()

    assert tuple(
        change.change_id
        for change in pattern.root_changes
    ) == ("change-a",)


def test_leaf_changes() -> None:
    pattern = make_pattern()

    assert tuple(
        change.change_id
        for change in pattern.leaf_changes
    ) == ("change-c",)


def test_causal_depth() -> None:
    pattern = make_pattern()

    assert pattern.causal_depth == 3


def test_direct_causes_of() -> None:
    pattern = make_pattern()

    assert tuple(
        change.change_id
        for change in pattern.direct_causes_of(
            "change-c"
        )
    ) == ("change-b",)


def test_direct_effects_of() -> None:
    pattern = make_pattern()

    assert tuple(
        change.change_id
        for change in pattern.direct_effects_of(
            "change-a"
        )
    ) == ("change-b",)


def test_ancestors_of() -> None:
    pattern = make_pattern()

    assert tuple(
        change.change_id
        for change in pattern.ancestors_of(
            "change-c"
        )
    ) == (
        "change-a",
        "change-b",
    )


def test_descendants_of() -> None:
    pattern = make_pattern()

    assert tuple(
        change.change_id
        for change in pattern.descendants_of(
            "change-a"
        )
    ) == (
        "change-b",
        "change-c",
    )


@pytest.mark.parametrize(
    "method_name",
    (
        "direct_causes_of",
        "direct_effects_of",
        "ancestors_of",
        "descendants_of",
    ),
)
def test_causal_queries_reject_unknown_change(
    method_name: str,
) -> None:
    pattern = make_pattern()
    method = getattr(pattern, method_name)

    with pytest.raises(
        HistoryPatternError,
        match="unknown change_id",
    ):
        method("missing")


def test_capacity_summaries() -> None:
    pattern = make_pattern()

    assert pattern.total_capacity_loss == pytest.approx(
        0.45
    )
    assert pattern.total_capacity_gain == pytest.approx(
        0.0
    )
    assert pattern.net_capacity_effect == pytest.approx(
        -0.45
    )


def test_capacity_gain_and_loss_are_separate() -> None:
    harmful = make_change(
        "harmful",
        capacity_effect=-0.30,
    )
    beneficial = make_change(
        "beneficial",
        onset_time=1.0,
        capacity_effect=0.20,
        functional_effect=FunctionalEffect.BENEFICIAL,
        kind=IrreversibleChangeKind.REMODELING,
    )

    pattern = HistoryPattern(
        changes=(harmful, beneficial),
    )

    assert pattern.total_capacity_loss == pytest.approx(
        0.30
    )
    assert pattern.total_capacity_gain == pytest.approx(
        0.20
    )
    assert pattern.net_capacity_effect == pytest.approx(
        -0.10
    )


def test_signature_weight_summaries() -> None:
    pattern = make_pattern()

    # Each default change:
    # retained_fraction=0.80
    # memory_strength=0.75
    # signature_weight=0.60
    assert (
        pattern.cumulative_signature_weight
        == pytest.approx(1.80)
    )
    assert pattern.mean_signature_weight == pytest.approx(
        0.60
    )


def test_persistence_index_is_bounded() -> None:
    pattern = make_pattern()

    assert 0.0 <= pattern.persistence_index <= 1.0


def test_irreversibility_index_is_bounded() -> None:
    pattern = make_pattern()

    assert 0.0 <= pattern.irreversibility_index <= 1.0


def test_fully_irreversible_pattern_index() -> None:
    change = make_change(
        "permanent",
        permanence=TracePersistence.PERMANENT,
        retained_fraction=1.0,
        memory_strength=1.0,
        reversibility=ReversibilityClass.IRREVERSIBLE,
    )

    pattern = HistoryPattern(
        changes=(change,),
    )

    assert pattern.persistence_index == pytest.approx(
        1.0
    )
    assert (
        pattern.irreversibility_index
        == pytest.approx(1.0)
    )


def test_progressive_pattern() -> None:
    pattern = make_pattern()

    assert pattern.progression_index == pytest.approx(
        1.0
    )
    assert pattern.adaptation_index == pytest.approx(
        0.0
    )
    assert pattern.is_progressive is True
    assert pattern.is_adaptive is False
    assert pattern.is_balanced is False


def test_adaptive_pattern() -> None:
    change_a = make_change(
        "adapt-a",
        kind=IrreversibleChangeKind.REMODELING,
        functional_effect=FunctionalEffect.BENEFICIAL,
        capacity_effect=0.20,
    )
    change_b = make_change(
        "adapt-b",
        onset_time=2.0,
        kind=IrreversibleChangeKind.HYPERTROPHY,
        functional_effect=FunctionalEffect.BENEFICIAL,
        capacity_effect=0.15,
    )

    pattern = HistoryPattern(
        changes=(change_a, change_b),
    )

    assert pattern.progression_index == pytest.approx(
        0.0
    )
    assert pattern.adaptation_index == pytest.approx(
        1.0
    )
    assert pattern.is_adaptive is True
    assert pattern.is_progressive is False


def test_empty_pattern_is_balanced() -> None:
    pattern = HistoryPattern()

    assert pattern.is_balanced is True
    assert pattern.is_progressive is False
    assert pattern.is_adaptive is False


def test_mixed_functional_effect_contributes_half() -> None:
    change = make_change(
        "mixed",
        functional_effect=FunctionalEffect.MIXED,
        capacity_effect=0.0,
    )

    pattern = HistoryPattern(
        changes=(change,),
    )

    assert pattern.progression_index == pytest.approx(
        0.5
    )
    assert pattern.adaptation_index == pytest.approx(
        0.5
    )
    assert pattern.is_balanced is True


def test_plane_profile() -> None:
    pattern = make_pattern()
    profile = pattern.plane_profile()

    assert tuple(
        item.key
        for item in profile
    ) == (
        "mechanical",
        "thermal",
    )
    assert profile[0].count == 2
    assert profile[1].count == 2
    assert sum(
        item.score
        for item in profile
    ) == pytest.approx(1.0)


def test_agent_profile() -> None:
    pattern = make_pattern()
    profile = pattern.agent_profile()

    assert profile[0].key == "steel_object"
    assert profile[0].count == 2
    assert profile[1].key == "environment"


def test_kind_profile() -> None:
    pattern = make_pattern()
    profile = pattern.kind_profile()

    assert len(profile) == 1
    assert profile[0].key == "damage"
    assert profile[0].count == 3
    assert profile[0].score == pytest.approx(1.0)


def test_time_scale_profile() -> None:
    pattern = make_pattern()
    profile = pattern.time_scale_profile()

    assert len(profile) == 1
    assert profile[0].key == "medium"
    assert profile[0].count == 3


def test_target_profile() -> None:
    pattern = make_pattern()
    profile = pattern.target_profile()

    assert {
        item.key
        for item in profile
    } == {
        "element:ELEMENT_A",
        "element:ELEMENT_B",
        "element:ELEMENT_C",
    }


def test_dominant_contributions() -> None:
    pattern = make_pattern()

    assert pattern.dominant_plane is not None
    assert pattern.dominant_agent is not None
    assert pattern.dominant_kind is not None
    assert pattern.dominant_time_scale is not None

    # Equal plane weights are ordered alphabetically.
    assert pattern.dominant_plane.key == "mechanical"
    assert pattern.dominant_agent.key == "steel_object"
    assert pattern.dominant_kind.key == "damage"
    assert pattern.dominant_time_scale.key == "medium"


def test_empty_pattern_has_no_dominant_values() -> None:
    pattern = HistoryPattern()

    assert pattern.dominant_plane is None
    assert pattern.dominant_agent is None
    assert pattern.dominant_kind is None
    assert pattern.dominant_time_scale is None


def test_ranked_contribution_validation() -> None:
    contribution = RankedContribution(
        key="mechanical",
        count=2,
        raw_weight=1.2,
        score=0.6,
    )

    assert contribution.to_dict() == {
        "key": "mechanical",
        "count": 2,
        "raw_weight": 1.2,
        "score": 0.6,
    }


def test_ranked_contribution_rejects_bad_score() -> None:
    with pytest.raises(
        HistoryPatternError,
        match="score must be in",
    ):
        RankedContribution(
            key="mechanical",
            count=1,
            raw_weight=1.0,
            score=1.1,
        )


def test_filter_by_plane() -> None:
    pattern = make_pattern()
    filtered = pattern.filter_by_plane("thermal")

    assert tuple(
        change.change_id
        for change in filtered
    ) == (
        "change-b",
        "change-c",
    )

    # change-a is no longer included but remains an external cause.
    assert "change-a" in filtered.external_cause_ids


def test_filter_by_agent() -> None:
    pattern = make_pattern()
    filtered = pattern.filter_by_agent("steel_object")

    assert tuple(
        change.change_id
        for change in filtered
    ) == (
        "change-a",
        "change-b",
    )


def test_filter_by_kind() -> None:
    pattern = make_pattern()
    filtered = pattern.filter_by_kind("damage")

    assert len(filtered) == 3


def test_filter_by_time_scale() -> None:
    pattern = make_pattern()
    filtered = pattern.filter_by_time_scale(
        TimeScale.MEDIUM
    )

    assert len(filtered) == 3


def test_filter_by_target_kind() -> None:
    pattern = make_pattern()
    filtered = pattern.filter_by_target_kind(
        HistoryTargetKind.ELEMENT
    )

    assert len(filtered) == 3


def test_filter_by_time() -> None:
    pattern = make_pattern()
    filtered = pattern.filter_by_time(
        start=1.5,
        end=3.0,
    )

    assert tuple(
        change.change_id
        for change in filtered
    ) == ("change-b",)


def test_filter_by_time_rejects_invalid_range() -> None:
    pattern = make_pattern()

    with pytest.raises(
        HistoryPatternError,
        match="end must not precede start",
    ):
        pattern.filter_by_time(
            start=5.0,
            end=2.0,
        )


def test_append_returns_new_pattern() -> None:
    first = make_change(
        "first",
        onset_time=1.0,
    )
    second = make_change(
        "second",
        onset_time=2.0,
        cause_change_ids=("first",),
    )

    pattern = HistoryPattern(
        pattern_id="append-pattern",
        changes=(first,),
    )

    extended = pattern.append(second)

    assert len(pattern) == 1
    assert len(extended) == 2
    assert extended.pattern_id == "append-pattern"
    assert extended is not pattern


def test_append_rejects_invalid_object() -> None:
    pattern = HistoryPattern()

    with pytest.raises(
        HistoryPatternError,
        match="must be an IrreversibleChange",
    ):
        pattern.append("change")


def test_merge_patterns() -> None:
    change_a = make_change(
        "merge-a",
        onset_time=1.0,
    )
    change_b = make_change(
        "merge-b",
        onset_time=2.0,
        cause_change_ids=("merge-a",),
    )

    pattern_a = HistoryPattern(
        changes=(change_a,),
        metadata={"domain": "mechanical"},
    )
    pattern_b = HistoryPattern(
        changes=(change_b,),
        external_cause_ids=("merge-a",),
        metadata={"model": "test"},
    )

    merged = pattern_a.merge(
        pattern_b,
        label="Merged",
    )

    assert len(merged) == 2
    assert merged.label == "Merged"
    assert merged.external_cause_ids == ()
    assert merged.metadata["domain"] == "mechanical"
    assert merged.metadata["model"] == "test"
    assert merged.causal_depth == 2


def test_merge_rejects_invalid_pattern() -> None:
    pattern = make_pattern()

    with pytest.raises(
        HistoryPatternError,
        match="other must be a HistoryPattern",
    ):
        pattern.merge("pattern")


def test_merge_rejects_conflicting_changes() -> None:
    original = make_change(
        "same-id",
        capacity_effect=-0.10,
    )
    conflicting = make_change(
        "same-id",
        capacity_effect=-0.50,
    )

    pattern_a = HistoryPattern(
        changes=(original,),
    )
    pattern_b = HistoryPattern(
        changes=(conflicting,),
    )

    with pytest.raises(
        HistoryPatternError,
        match="conflicting changes",
    ):
        pattern_a.merge(pattern_b)


def test_merge_rejects_conflicting_metadata() -> None:
    pattern_a = HistoryPattern(
        metadata={"domain": "mechanical"},
    )
    pattern_b = HistoryPattern(
        metadata={"domain": "biological"},
    )

    with pytest.raises(
        HistoryPatternError,
        match="conflicting metadata",
    ):
        pattern_a.merge(pattern_b)


def test_summary() -> None:
    pattern = make_pattern()
    summary = pattern.summary()

    assert summary["pattern_id"] == "pattern-001"
    assert summary["total_changes"] == 3
    assert summary["root_change_ids"] == [
        "change-a"
    ]
    assert summary["leaf_change_ids"] == [
        "change-c"
    ]
    assert summary["causal_depth"] == 3
    assert summary["is_progressive"] is True
    assert (
        summary["dominant_plane"]["key"]
        == "mechanical"
    )


def test_pattern_round_trip() -> None:
    pattern = make_pattern()

    restored = HistoryPattern.from_dict(
        pattern.to_dict()
    )

    assert restored == pattern
    assert restored is not pattern
    assert restored.changes[0] is not pattern.changes[0]


def test_serialized_version() -> None:
    data = make_pattern().to_dict()

    assert data["version"] == "1.0.0"
    assert data["pattern_id"] == "pattern-001"
    assert len(data["changes"]) == 3


def test_from_dict_rejects_unsupported_version() -> None:
    data = make_pattern().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        HistoryPatternError,
        match="unsupported history pattern version",
    ):
        HistoryPattern.from_dict(data)


def test_from_dict_rejects_invalid_changes() -> None:
    data = make_pattern().to_dict()
    data["changes"] = "invalid"

    with pytest.raises(
        HistoryPatternError,
        match="changes must be a sequence",
    ):
        HistoryPattern.from_dict(data)


def test_pattern_rejects_non_change_objects() -> None:
    with pytest.raises(
        HistoryPatternError,
        match="all changes must be",
    ):
        HistoryPattern(
            changes=("invalid",),
        )


def test_pattern_rejects_duplicate_change_ids() -> None:
    first = make_change(
        "duplicate",
        onset_time=1.0,
    )
    second = make_change(
        "duplicate",
        onset_time=2.0,
    )

    with pytest.raises(
        HistoryPatternError,
        match="change_id values must be unique",
    ):
        HistoryPattern(
            changes=(first, second),
        )


def test_pattern_rejects_unknown_causal_reference() -> None:
    change = make_change(
        "effect",
        cause_change_ids=("unknown-cause",),
    )

    with pytest.raises(
        HistoryPatternError,
        match="unknown causal reference",
    ):
        HistoryPattern(
            changes=(change,),
        )


def test_external_cause_is_allowed() -> None:
    change = make_change(
        "observed-effect",
        cause_change_ids=("unobserved-cause",),
    )

    pattern = HistoryPattern(
        changes=(change,),
        external_cause_ids=("unobserved-cause",),
    )

    assert pattern.external_cause_ids == (
        "unobserved-cause",
    )
    assert pattern.root_changes == (change,)


def test_internal_id_cannot_also_be_external() -> None:
    change = make_change(
        "change-a",
    )

    with pytest.raises(
        HistoryPatternError,
        match="internal change cannot also be",
    ):
        HistoryPattern(
            changes=(change,),
            external_cause_ids=("change-a",),
        )


def test_cause_cannot_begin_after_effect() -> None:
    cause = make_change(
        "cause",
        onset_time=5.0,
    )
    effect = make_change(
        "effect",
        onset_time=2.0,
        cause_change_ids=("cause",),
    )

    with pytest.raises(
        HistoryPatternError,
        match="cause cannot begin after",
    ):
        HistoryPattern(
            changes=(cause, effect),
        )


def test_cause_at_same_onset_cannot_be_recorded_later() -> None:
    cause = make_change(
        "cause",
        onset_time=1.0,
        recorded_time=2.0,
    )
    effect = make_change(
        "effect",
        onset_time=1.0,
        recorded_time=1.5,
        cause_change_ids=("cause",),
    )

    with pytest.raises(
        HistoryPatternError,
        match="cause cannot be recorded after",
    ):
        HistoryPattern(
            changes=(cause, effect),
        )


def test_pattern_rejects_causal_cycle() -> None:
    change_a = make_change(
        "cycle-a",
        onset_time=1.0,
        recorded_time=1.0,
        cause_change_ids=("cycle-b",),
    )
    change_b = make_change(
        "cycle-b",
        onset_time=1.0,
        recorded_time=1.0,
        cause_change_ids=("cycle-a",),
    )

    with pytest.raises(
        HistoryPatternError,
        match="causal graph contains a cycle",
    ):
        HistoryPattern(
            changes=(change_a, change_b),
        )


def test_external_ids_are_deduplicated() -> None:
    change = make_change(
        "effect",
        cause_change_ids=("external-a",),
    )

    pattern = HistoryPattern(
        changes=(change,),
        external_cause_ids=(
            "external-a",
            "external-b",
            "external-a",
        ),
    )

    assert pattern.external_cause_ids == (
        "external-a",
        "external-b",
    )


def test_generated_pattern_id_is_nonempty() -> None:
    pattern = HistoryPattern()

    assert isinstance(pattern.pattern_id, str)
    assert len(pattern.pattern_id) > 0