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
from roif.history.history_pattern import HistoryPattern
from roif.history.irreversible_change import (
    FunctionalEffect,
    IrreversibleChange,
    IrreversibleChangeKind,
    TracePersistence,
)
from roif.history.rheology_memory import (
    RheologicalMemory,
    RheologyModel,
    RheologyPhase,
)
from roif.history.structural_signature import (
    SignatureComparison,
    SignatureDistanceWeights,
    StructuralSignature,
    StructuralSignatureError,
)


def make_target(
    target_id: str = "ELEMENT_A",
) -> HistoryTarget:
    return HistoryTarget(
        kind=HistoryTargetKind.ELEMENT,
        target_id=target_id,
        label=target_id,
    )



def make_rheology_memory(
    *,
    target_id: str = "ELEMENT_A",
    memory_id: str = "rheology-memory-a",
    creep_strain: float = 0.08,
    peak_creep_strain: float = 0.10,
    residual_strain: float = 0.03,
    retained_fraction: float = 0.80,
    phase: RheologyPhase = RheologyPhase.CREEPING,
) -> RheologicalMemory:
    return RheologicalMemory(
        memory_id=memory_id,
        target_id=target_id,
        observation_time=3.0,
        rheology_time=10.0,
        creep_strain=creep_strain,
        peak_creep_strain=peak_creep_strain,
        elastic_strain=0.04,
        residual_strain=residual_strain,
        recoverable_strain=0.05,
        total_strain=0.12,
        instantaneous_stiffness=10.0,
        relaxed_stiffness=5.0,
        creep_time_constant=2.0,
        applied_stress=1.0,
        relaxation_fraction=0.50,
        recovery_fraction=0.20,
        retained_fraction=retained_fraction,
        confidence=0.90,
        phase=phase,
        model=RheologyModel.STANDARD_LINEAR_SOLID,
    )


def make_change(
    change_id: str,
    *,
    onset_time: float = 0.0,
    recorded_time: float | None = None,
    target_id: str = "ELEMENT_A",
    kind: IrreversibleChangeKind = (
        IrreversibleChangeKind.DAMAGE
    ),
    capacity_effect: float = -0.10,
    functional_effect: FunctionalEffect = (
        FunctionalEffect.HARMFUL
    ),
    permanence: TracePersistence = (
        TracePersistence.LONG_LIVED
    ),
    reversibility: ReversibilityClass = (
        ReversibilityClass.PARTIALLY_REVERSIBLE
    ),
    retained_fraction: float = 0.80,
    memory_strength: float = 0.75,
    time_scale: TimeScale = TimeScale.MEDIUM,
    cause_change_ids: tuple[str, ...] = (),
    plane_ids: tuple[str, ...] = ("mechanical",),
    agent_ids: tuple[str, ...] = ("external_load",),
    rheology_memory: RheologicalMemory | None = None,
) -> IrreversibleChange:
    if recorded_time is None:
        recorded_time = onset_time

    return IrreversibleChange(
        change_id=change_id,
        kind=kind,
        source_event_id=f"event-{change_id}",
        target=make_target(target_id),
        delta=StateDelta(
            quantity="damage",
            before=0.0,
            after=0.2,
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
        rheology_memory=rheology_memory,
    )


def make_pattern() -> HistoryPattern:
    root = make_change(
        "change-a",
        onset_time=1.0,
        recorded_time=1.2,
        target_id="ELEMENT_A",
        capacity_effect=-0.10,
        plane_ids=("mechanical",),
        agent_ids=("steel_object",),
    )

    middle = make_change(
        "change-b",
        onset_time=2.0,
        recorded_time=2.3,
        target_id="ELEMENT_B",
        capacity_effect=-0.15,
        cause_change_ids=("change-a",),
        plane_ids=("mechanical", "thermal"),
        agent_ids=("steel_object",),
    )

    leaf = make_change(
        "change-c",
        onset_time=3.0,
        recorded_time=3.5,
        target_id="ELEMENT_C",
        kind=IrreversibleChangeKind.REMODELING,
        capacity_effect=0.05,
        functional_effect=FunctionalEffect.BENEFICIAL,
        cause_change_ids=("change-b",),
        plane_ids=("thermal",),
        agent_ids=("environment",),
        time_scale=TimeScale.SLOW,
    )

    return HistoryPattern(
        pattern_id="pattern-001",
        changes=(leaf, root, middle),
        label="Mixed structural history",
        description="Damage followed by partial adaptation.",
        external_cause_ids=("external-trigger",),
        metadata={
            "domain": "test-network",
        },
    )


def make_signature(
    **overrides,
) -> StructuralSignature:
    data = {
        "signature_id": "signature-001",
        "source_pattern_id": "pattern-001",
        "label": "Reference signature",
        "plane_profile": {
            "mechanical": 0.70,
            "thermal": 0.30,
        },
        "agent_profile": {
            "steel_object": 0.75,
            "environment": 0.25,
        },
        "kind_profile": {
            "damage": 0.80,
            "remodeling": 0.20,
        },
        "time_scale_profile": {
            "medium": 0.80,
            "slow": 0.20,
        },
        "target_profile": {
            "element:ELEMENT_A": 0.40,
            "element:ELEMENT_B": 0.35,
            "element:ELEMENT_C": 0.25,
        },
        "total_changes": 3,
        "root_count": 1,
        "leaf_count": 1,
        "causal_depth": 3,
        "start_time": 1.0,
        "end_time": 3.5,
        "duration": 2.5,
        "total_capacity_loss": 0.25,
        "total_capacity_gain": 0.05,
        "net_capacity_effect": -0.20,
        "cumulative_signature_weight": 1.80,
        "mean_signature_weight": 0.60,
        "persistence_index": 0.70,
        "irreversibility_index": 0.50,
        "progression_index": 0.80,
        "adaptation_index": 0.20,
        "harmful_fraction": 0.80,
        "beneficial_fraction": 0.20,
        "mixed_fraction": 0.0,
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)
    return StructuralSignature(**data)


def test_signature_creation() -> None:
    signature = make_signature()

    assert signature.signature_id == "signature-001"
    assert signature.source_pattern_id == "pattern-001"
    assert signature.total_changes == 3
    assert signature.root_count == 1
    assert signature.leaf_count == 1
    assert signature.causal_depth == 3
    assert signature.duration == pytest.approx(2.5)


def test_signature_is_immutable() -> None:
    signature = make_signature()

    with pytest.raises(FrozenInstanceError):
        signature.total_changes = 10


def test_signature_metadata_is_read_only() -> None:
    signature = make_signature()

    with pytest.raises(TypeError):
        signature.metadata["new"] = "value"


def test_signature_profiles_are_read_only() -> None:
    signature = make_signature()

    with pytest.raises(TypeError):
        signature.plane_profile["chemical"] = 0.2


def test_profiles_are_normalized() -> None:
    signature = make_signature(
        plane_profile={
            "mechanical": 7.0,
            "thermal": 3.0,
        }
    )

    assert signature.plane_profile["mechanical"] == (
        pytest.approx(0.7)
    )
    assert signature.plane_profile["thermal"] == (
        pytest.approx(0.3)
    )
    assert sum(
        signature.plane_profile.values()
    ) == pytest.approx(1.0)


def test_zero_profile_values_are_removed() -> None:
    signature = make_signature(
        plane_profile={
            "mechanical": 1.0,
            "thermal": 0.0,
        }
    )

    assert dict(signature.plane_profile) == {
        "mechanical": 1.0,
    }


def test_empty_signature_creation() -> None:
    signature = StructuralSignature(
        source_pattern_id="empty-pattern",
        signature_id="empty-signature",
    )

    assert signature.is_empty is True
    assert signature.total_changes == 0
    assert signature.root_count == 0
    assert signature.leaf_count == 0
    assert signature.causal_depth == 0
    assert signature.duration == pytest.approx(0.0)


def test_progressive_signature() -> None:
    signature = make_signature()

    assert signature.is_progressive is True
    assert signature.is_adaptive is False
    assert signature.is_balanced is False


def test_adaptive_signature() -> None:
    signature = make_signature(
        total_capacity_loss=0.05,
        total_capacity_gain=0.30,
        net_capacity_effect=0.25,
        progression_index=0.10,
        adaptation_index=0.90,
        harmful_fraction=0.10,
        beneficial_fraction=0.90,
    )

    assert signature.is_adaptive is True
    assert signature.is_progressive is False
    assert signature.is_balanced is False


def test_empty_signature_is_balanced() -> None:
    signature = StructuralSignature(
        source_pattern_id="empty-pattern",
    )

    assert signature.is_balanced is True


def test_dominant_values() -> None:
    signature = make_signature()

    assert signature.dominant_plane == (
        "mechanical",
        pytest.approx(0.70),
    )
    assert signature.dominant_agent == (
        "steel_object",
        pytest.approx(0.75),
    )
    assert signature.dominant_kind == (
        "damage",
        pytest.approx(0.80),
    )
    assert signature.dominant_time_scale == (
        "medium",
        pytest.approx(0.80),
    )


def test_empty_profiles_have_no_dominant_values() -> None:
    signature = StructuralSignature(
        source_pattern_id="empty-pattern",
    )

    assert signature.dominant_plane is None
    assert signature.dominant_agent is None
    assert signature.dominant_kind is None
    assert signature.dominant_time_scale is None


def test_profile_value() -> None:
    signature = make_signature()

    assert signature.profile_value(
        "plane",
        "mechanical",
    ) == pytest.approx(0.70)

    assert signature.profile_value(
        "plane",
        "missing",
    ) == pytest.approx(0.0)


def test_profile_value_rejects_unknown_profile() -> None:
    signature = make_signature()

    with pytest.raises(
        StructuralSignatureError,
        match="unsupported profile name",
    ):
        signature.profile_value(
            "unsupported",
            "mechanical",
        )


def test_capacity_burden() -> None:
    signature = make_signature()

    assert signature.capacity_burden == pytest.approx(
        0.25 / 0.30
    )


def test_capacity_adaptation() -> None:
    signature = make_signature()

    assert signature.capacity_adaptation == pytest.approx(
        0.05 / 0.30
    )


def test_zero_capacity_changes_have_zero_ratios() -> None:
    signature = StructuralSignature(
        source_pattern_id="empty-pattern",
    )

    assert signature.capacity_burden == pytest.approx(
        0.0
    )
    assert signature.capacity_adaptation == pytest.approx(
        0.0
    )


def test_normalized_causal_depth() -> None:
    signature = make_signature()

    assert (
        signature.normalized_causal_depth
        == pytest.approx(1.0)
    )


def test_causal_branching_index_for_chain() -> None:
    signature = make_signature(
        root_count=1,
        leaf_count=1,
    )

    assert signature.causal_branching_index == (
        pytest.approx(0.0)
    )


def test_causal_branching_index_for_branch() -> None:
    signature = make_signature(
        total_changes=4,
        root_count=1,
        leaf_count=3,
        causal_depth=2,
        cumulative_signature_weight=2.4,
        mean_signature_weight=0.6,
    )

    assert signature.causal_branching_index == (
        pytest.approx(2.0 / 3.0)
    )


def test_compact_vector_is_stable() -> None:
    signature = make_signature()
    vector = signature.compact_vector()

    assert isinstance(vector, tuple)
    assert len(vector) == 21
    assert vector[0] == pytest.approx(3.0)
    assert vector[4] == pytest.approx(2.5)


def test_from_pattern_creation() -> None:
    pattern = make_pattern()

    signature = StructuralSignature.from_pattern(
        pattern,
        signature_id="derived-signature",
    )

    assert signature.signature_id == "derived-signature"
    assert signature.source_pattern_id == "pattern-001"
    assert signature.label == pattern.label
    assert signature.total_changes == 3
    assert signature.root_count == 1
    assert signature.leaf_count == 1
    assert signature.causal_depth == 3

    assert signature.start_time == pytest.approx(1.0)
    assert signature.end_time == pytest.approx(3.5)
    assert signature.duration == pytest.approx(2.5)

    assert signature.total_capacity_loss == pytest.approx(
        0.25
    )
    assert signature.total_capacity_gain == pytest.approx(
        0.05
    )
    assert signature.net_capacity_effect == pytest.approx(
        -0.20
    )


def test_from_pattern_profiles_are_created() -> None:
    signature = StructuralSignature.from_pattern(
        make_pattern()
    )

    assert set(signature.plane_profile) == {
        "mechanical",
        "thermal",
    }
    assert set(signature.agent_profile) == {
        "steel_object",
        "environment",
    }
    assert set(signature.kind_profile) == {
        "damage",
        "remodeling",
    }
    assert set(signature.time_scale_profile) == {
        "medium",
        "slow",
    }


def test_from_pattern_functional_fractions() -> None:
    signature = StructuralSignature.from_pattern(
        make_pattern()
    )

    assert signature.harmful_fraction == pytest.approx(
        2.0 / 3.0
    )
    assert signature.beneficial_fraction == pytest.approx(
        1.0 / 3.0
    )
    assert signature.mixed_fraction == pytest.approx(
        0.0
    )


def test_from_pattern_metadata() -> None:
    pattern = make_pattern()

    signature = StructuralSignature.from_pattern(
        pattern,
        metadata={
            "decoder_class": "reference",
        },
    )

    assert (
        signature.metadata["source_pattern_label"]
        == pattern.label
    )
    assert (
        signature.metadata[
            "source_pattern_description"
        ]
        == pattern.description
    )
    assert (
        signature.metadata[
            "source_pattern_external_causes"
        ]
        == ["external-trigger"]
    )
    assert (
        signature.metadata["decoder_class"]
        == "reference"
    )


def test_from_pattern_explicit_label() -> None:
    signature = StructuralSignature.from_pattern(
        make_pattern(),
        label="Explicit label",
    )

    assert signature.label == "Explicit label"


def test_from_pattern_rejects_invalid_pattern() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="pattern must be a HistoryPattern",
    ):
        StructuralSignature.from_pattern("pattern")


def test_empty_pattern_signature() -> None:
    pattern = HistoryPattern(
        pattern_id="empty-pattern",
    )

    signature = StructuralSignature.from_pattern(
        pattern
    )

    assert signature.is_empty is True
    assert signature.plane_profile == {}
    assert signature.cumulative_signature_weight == (
        pytest.approx(0.0)
    )


def test_signature_round_trip() -> None:
    signature = make_signature()

    restored = StructuralSignature.from_dict(
        signature.to_dict()
    )

    assert restored == signature
    assert restored is not signature


def test_serialized_version() -> None:
    data = make_signature().to_dict()

    assert data["version"] == "1.0.0"
    assert data["signature_id"] == "signature-001"
    assert data["source_pattern_id"] == "pattern-001"


def test_from_dict_rejects_unsupported_version() -> None:
    data = make_signature().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        StructuralSignatureError,
        match="unsupported structural signature version",
    ):
        StructuralSignature.from_dict(data)


def test_summary() -> None:
    signature = make_signature()
    summary = signature.summary()

    assert summary["signature_id"] == "signature-001"
    assert summary["source_pattern_id"] == "pattern-001"
    assert summary["total_changes"] == 3
    assert summary["causal_depth"] == 3
    assert summary["is_progressive"] is True
    assert summary["dominant_plane"][0] == "mechanical"


def test_default_distance_weights() -> None:
    weights = SignatureDistanceWeights()

    assert weights.total_weight > 0.0
    assert weights.plane_profile == pytest.approx(
        1.0
    )
    assert weights.capacity == pytest.approx(1.0)


def test_distance_weights_are_immutable() -> None:
    weights = SignatureDistanceWeights()

    with pytest.raises(FrozenInstanceError):
        weights.capacity = 2.0


def test_distance_weights_round_trip() -> None:
    weights = SignatureDistanceWeights(
        plane_profile=2.0,
        capacity=3.0,
    )

    restored = SignatureDistanceWeights.from_dict(
        weights.to_dict()
    )

    assert restored == weights


def test_distance_weights_reject_negative_value() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="capacity must be non-negative",
    ):
        SignatureDistanceWeights(
            capacity=-1.0,
        )


def test_distance_weights_reject_all_zero() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="at least one distance weight",
    ):
        SignatureDistanceWeights(
            plane_profile=0.0,
            agent_profile=0.0,
            kind_profile=0.0,
            time_scale_profile=0.0,
            target_profile=0.0,
            capacity=0.0,
            persistence=0.0,
            irreversibility=0.0,
            progression=0.0,
            adaptation=0.0,
            causality=0.0,
            chronology=0.0,
        )


def test_identical_signature_comparison() -> None:
    signature = make_signature()

    comparison = signature.compare(signature)

    assert comparison.distance == pytest.approx(0.0)
    assert comparison.similarity == pytest.approx(1.0)
    assert all(
        value == pytest.approx(0.0)
        for value in comparison.components.values()
    )


def test_comparison_is_symmetric() -> None:
    left = make_signature(
        signature_id="left"
    )
    right = make_signature(
        signature_id="right",
        plane_profile={
            "chemical": 1.0,
        },
        progression_index=0.2,
        adaptation_index=0.8,
        total_capacity_loss=0.05,
        total_capacity_gain=0.20,
        net_capacity_effect=0.15,
        harmful_fraction=0.20,
        beneficial_fraction=0.80,
    )

    left_to_right = left.compare(right)
    right_to_left = right.compare(left)

    assert left_to_right.distance == pytest.approx(
        right_to_left.distance
    )
    assert left_to_right.similarity == pytest.approx(
        right_to_left.similarity
    )
    assert dict(
        left_to_right.components
    ) == pytest.approx(
        dict(right_to_left.components)
    )


def test_different_profiles_increase_distance() -> None:
    left = make_signature(
        signature_id="left",
        plane_profile={
            "mechanical": 1.0,
        },
    )
    right = make_signature(
        signature_id="right",
        plane_profile={
            "biological": 1.0,
        },
    )

    comparison = left.compare(right)

    assert comparison.components[
        "plane_profile"
    ] == pytest.approx(1.0)
    assert comparison.distance > 0.0
    assert comparison.similarity < 1.0


def test_custom_comparison_weights() -> None:
    left = make_signature(
        signature_id="left",
        plane_profile={
            "mechanical": 1.0,
        },
    )
    right = make_signature(
        signature_id="right",
        plane_profile={
            "thermal": 1.0,
        },
    )

    weights = SignatureDistanceWeights(
        plane_profile=10.0,
        agent_profile=0.0,
        kind_profile=0.0,
        time_scale_profile=0.0,
        target_profile=0.0,
        capacity=0.0,
        persistence=0.0,
        irreversibility=0.0,
        progression=0.0,
        adaptation=0.0,
        causality=0.0,
        chronology=0.0,
    )

    comparison = left.compare(
        right,
        weights=weights,
    )

    assert comparison.distance == pytest.approx(1.0)
    assert comparison.similarity == pytest.approx(0.0)


def test_compare_rejects_invalid_other() -> None:
    signature = make_signature()

    with pytest.raises(
        StructuralSignatureError,
        match="other must be a StructuralSignature",
    ):
        signature.compare("signature")


def test_compare_rejects_invalid_weights() -> None:
    signature = make_signature()

    with pytest.raises(
        StructuralSignatureError,
        match="weights must be SignatureDistanceWeights",
    ):
        signature.compare(
            signature,
            weights="weights",
        )


def test_comparison_properties() -> None:
    left = make_signature(
        signature_id="left"
    )
    right = make_signature(
        signature_id="right",
        plane_profile={
            "chemical": 1.0,
        },
    )

    comparison = left.compare(right)

    assert isinstance(
        comparison,
        SignatureComparison,
    )
    assert comparison.left_signature_id == "left"
    assert comparison.right_signature_id == "right"
    assert comparison.similarity == pytest.approx(
        1.0 - comparison.distance
    )


def test_comparison_components_are_read_only() -> None:
    comparison = make_signature().compare(
        make_signature(
            signature_id="other"
        )
    )

    with pytest.raises(TypeError):
        comparison.components["new"] = 0.5


def test_most_different_component() -> None:
    left = make_signature(
        signature_id="left",
        plane_profile={
            "mechanical": 1.0,
        },
    )
    right = make_signature(
        signature_id="right",
        plane_profile={
            "chemical": 1.0,
        },
    )

    comparison = left.compare(right)

    most_different = (
        comparison.most_different_component
    )

    assert most_different is not None
    assert most_different[1] == pytest.approx(1.0)


def test_comparison_to_dict() -> None:
    comparison = make_signature().compare(
        make_signature(
            signature_id="other"
        )
    )

    data = comparison.to_dict()

    assert data["version"] == "1.0.0"
    assert data["left_signature_id"] == (
        "signature-001"
    )
    assert data["right_signature_id"] == "other"
    assert "components" in data
    assert "weights" in data


def test_euclidean_distance_identical() -> None:
    signature = make_signature()

    assert signature.euclidean_scalar_distance(
        signature
    ) == pytest.approx(0.0)


def test_euclidean_distance_is_symmetric() -> None:
    left = make_signature(
        signature_id="left"
    )
    right = make_signature(
        signature_id="right",
        total_capacity_loss=0.50,
        total_capacity_gain=0.05,
        net_capacity_effect=-0.45,
    )

    assert left.euclidean_scalar_distance(
        right
    ) == pytest.approx(
        right.euclidean_scalar_distance(left)
    )


def test_euclidean_distance_rejects_invalid_other() -> None:
    signature = make_signature()

    with pytest.raises(
        StructuralSignatureError,
        match="other must be a StructuralSignature",
    ):
        signature.euclidean_scalar_distance(
            "signature"
        )


def test_signature_rejects_empty_source_pattern_id() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="source_pattern_id must not be empty",
    ):
        make_signature(
            source_pattern_id="   "
        )


def test_signature_rejects_empty_signature_id() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="signature_id must not be empty",
    ):
        make_signature(
            signature_id=""
        )


def test_signature_rejects_negative_profile_value() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="plane_profile",
    ):
        make_signature(
            plane_profile={
                "mechanical": -0.1,
            }
        )


def test_empty_signature_rejects_nonzero_roots() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="empty signature cannot have roots",
    ):
        StructuralSignature(
            source_pattern_id="empty-pattern",
            total_changes=0,
            root_count=1,
        )


def test_root_count_cannot_exceed_total() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="root_count cannot exceed",
    ):
        make_signature(
            total_changes=2,
            root_count=3,
            leaf_count=1,
            causal_depth=2,
            cumulative_signature_weight=1.2,
            mean_signature_weight=0.6,
        )


def test_leaf_count_cannot_exceed_total() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="leaf_count cannot exceed",
    ):
        make_signature(
            total_changes=2,
            root_count=1,
            leaf_count=3,
            causal_depth=2,
            cumulative_signature_weight=1.2,
            mean_signature_weight=0.6,
        )


def test_causal_depth_cannot_exceed_total() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="causal_depth cannot exceed",
    ):
        make_signature(
            total_changes=2,
            root_count=1,
            leaf_count=1,
            causal_depth=3,
            cumulative_signature_weight=1.2,
            mean_signature_weight=0.6,
        )


def test_signature_rejects_end_before_start() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="end_time must not precede",
    ):
        make_signature(
            start_time=5.0,
            end_time=3.0,
            duration=0.0,
        )


def test_signature_rejects_wrong_duration() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="duration must equal",
    ):
        make_signature(
            duration=10.0,
        )


def test_signature_rejects_wrong_net_capacity() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="net_capacity_effect must equal",
    ):
        make_signature(
            net_capacity_effect=-0.99,
        )


def test_signature_rejects_wrong_mean_weight() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="mean_signature_weight must equal",
    ):
        make_signature(
            mean_signature_weight=0.90,
        )


def test_signature_rejects_invalid_index() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="progression_index must be in",
    ):
        make_signature(
            progression_index=1.1,
        )


def test_signature_rejects_functional_sum_above_one() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="functional fractions cannot sum above",
    ):
        make_signature(
            harmful_fraction=0.7,
            beneficial_fraction=0.5,
            mixed_fraction=0.1,
        )


def test_comparison_rejects_inconsistent_similarity() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="similarity must equal",
    ):
        SignatureComparison(
            left_signature_id="left",
            right_signature_id="right",
            distance=0.4,
            similarity=0.8,
            components={
                "capacity": 0.4,
            },
            weights=SignatureDistanceWeights(),
        )


def test_comparison_rejects_invalid_component() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="components",
    ):
        SignatureComparison(
            left_signature_id="left",
            right_signature_id="right",
            distance=0.4,
            similarity=0.6,
            components={
                "capacity": 1.2,
            },
            weights=SignatureDistanceWeights(),
        )


# ---------------------------------------------------------------------------
# Rheological signature integration
# ---------------------------------------------------------------------------


def make_rheological_pattern() -> HistoryPattern:
    first_memory = make_rheology_memory(
        target_id="ELEMENT_A",
        memory_id="rheology-memory-a",
        creep_strain=0.08,
        peak_creep_strain=0.10,
        residual_strain=0.03,
        retained_fraction=0.80,
    )
    second_memory = make_rheology_memory(
        target_id="ELEMENT_B",
        memory_id="rheology-memory-b",
        creep_strain=0.04,
        peak_creep_strain=0.06,
        residual_strain=0.02,
        retained_fraction=0.60,
        phase=RheologyPhase.RECOVERING,
    )

    first = make_change(
        "change-creep-a",
        onset_time=1.0,
        recorded_time=1.2,
        target_id="ELEMENT_A",
        kind=IrreversibleChangeKind.CREEP,
        plane_ids=("mechanical", "rheological"),
        rheology_memory=first_memory,
    )
    second = make_change(
        "change-creep-b",
        onset_time=2.0,
        recorded_time=2.3,
        target_id="ELEMENT_B",
        kind=IrreversibleChangeKind.CREEP,
        cause_change_ids=("change-creep-a",),
        plane_ids=("mechanical", "rheological"),
        rheology_memory=second_memory,
    )
    ordinary = make_change(
        "change-damage-c",
        onset_time=3.0,
        recorded_time=3.2,
        target_id="ELEMENT_C",
        cause_change_ids=("change-creep-b",),
    )

    return HistoryPattern(
        pattern_id="rheological-pattern",
        changes=(ordinary, second, first),
    )


def test_signature_without_rheology_is_backward_compatible() -> None:
    signature = StructuralSignature.from_pattern(
        make_pattern()
    )

    assert signature.has_rheological_memory is False
    assert signature.rheology_change_count == 0
    assert dict(signature.rheology_profile) == {}
    assert signature.total_abs_creep_strain == pytest.approx(0.0)
    assert signature.rheological_memory_index == pytest.approx(0.0)


def test_from_pattern_aggregates_rheological_memory() -> None:
    signature = StructuralSignature.from_pattern(
        make_rheological_pattern()
    )

    assert signature.has_rheological_memory is True
    assert signature.rheology_change_count == 2
    assert signature.rheology_fraction == pytest.approx(2.0 / 3.0)
    assert signature.total_abs_creep_strain == pytest.approx(0.12)
    assert signature.mean_abs_creep_strain == pytest.approx(0.06)
    assert signature.max_abs_creep_strain == pytest.approx(0.08)
    assert signature.mean_abs_residual_strain == pytest.approx(0.025)
    assert signature.max_abs_residual_strain == pytest.approx(0.03)
    assert 0.0 < signature.rheological_memory_index <= 1.0
    assert 0.0 < signature.rheological_retention_index <= 1.0
    assert 0.0 < signature.rheological_relaxation_index <= 1.0


def test_rheology_profile_is_normalized_and_read_only() -> None:
    signature = StructuralSignature.from_pattern(
        make_rheological_pattern()
    )

    assert sum(signature.rheology_profile.values()) == pytest.approx(1.0)

    with pytest.raises(TypeError):
        signature.rheology_profile["new"] = 1.0


def test_signature_reports_dominant_rheology() -> None:
    signature = StructuralSignature.from_pattern(
        make_rheological_pattern()
    )

    assert signature.dominant_rheology is not None

    name, weight = signature.dominant_rheology

    assert name in signature.rheology_profile
    assert signature.rheology_profile[name] == pytest.approx(weight)

def test_rheology_vector_is_stable() -> None:
    signature = StructuralSignature.from_pattern(
        make_rheological_pattern()
    )

    first = signature.rheology_vector()
    second = signature.rheology_vector()

    assert first == second
    assert len(first) > 0
    assert all(isinstance(value, float) for value in first)


def test_compact_vector_remains_backward_compatible() -> None:
    plain = make_signature()
    rheological = make_signature(
        rheology_change_count=1,
        rheology_profile={
            "standard_linear_solid": 1.0,
        },
        total_abs_creep_strain=0.08,
        mean_abs_creep_strain=0.08,
        max_abs_creep_strain=0.08,
        mean_abs_residual_strain=0.03,
        max_abs_residual_strain=0.03,
        rheological_memory_index=0.60,
        rheological_retention_index=0.80,
        rheological_relaxation_index=0.50,
    )

    assert len(plain.compact_vector()) == len(
        rheological.compact_vector()
    )


def test_rheological_signature_round_trip() -> None:
    signature = StructuralSignature.from_pattern(
        make_rheological_pattern()
    )

    restored = StructuralSignature.from_dict(
        signature.to_dict()
    )

    assert restored.to_dict() == signature.to_dict()
    assert restored.has_rheological_memory is True


def test_rheology_changes_signature_comparison() -> None:
    without_memory = make_signature(
        signature_id="without-memory",
    )
    with_memory = make_signature(
        signature_id="with-memory",
        rheology_change_count=1,
        rheology_profile={
            "standard_linear_solid": 1.0,
        },
        total_abs_creep_strain=0.08,
        mean_abs_creep_strain=0.08,
        max_abs_creep_strain=0.08,
        mean_abs_residual_strain=0.03,
        max_abs_residual_strain=0.03,
        rheological_memory_index=0.60,
        rheological_retention_index=0.80,
        rheological_relaxation_index=0.50,
    )

    comparison = without_memory.compare(with_memory)

    assert comparison.distance > 0.0
    assert comparison.similarity < 1.0
    assert comparison.components["rheology"] > 0.0
    assert comparison.components["rheology_profile"] > 0.0


def test_identical_rheological_signatures_compare_equal() -> None:
    signature = StructuralSignature.from_pattern(
        make_rheological_pattern()
    )

    comparison = signature.compare(signature)

    assert comparison.distance == pytest.approx(0.0)
    assert comparison.similarity == pytest.approx(1.0)
    assert comparison.components["rheology"] == pytest.approx(0.0)


def test_signature_rejects_rheology_count_above_total() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="rheology_change_count",
    ):
        make_signature(
            rheology_change_count=4,
        )


def test_signature_rejects_nonzero_rheology_without_count() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="rheology",
    ):
        make_signature(
            rheology_profile={
                "standard_linear_solid": 1.0,
            },
        )


def test_signature_rejects_wrong_mean_creep() -> None:
    with pytest.raises(
        StructuralSignatureError,
        match="mean_abs_creep_strain",
    ):
        make_signature(
            rheology_change_count=2,
            rheology_profile={
                "standard_linear_solid": 1.0,
            },
            total_abs_creep_strain=0.12,
            mean_abs_creep_strain=0.05,
            max_abs_creep_strain=0.08,
        )

