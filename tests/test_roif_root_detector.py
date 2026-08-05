"""
Tests for roif.roif_root_detector.

The suite verifies the causal-role contract of ROIF:
D_origin, D_fast, D_root, and Node* are evaluated independently.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
import math

import numpy as np
import pytest

from roif.roif_entities import (
    ActivationAvailability,
    CapacityState,
    ChannelState,
    EntityKind,
    FunctionalChannel,
    FunctionalRole,
    GeometryState,
    InfluenceOperator,
    InfluencePlane,
    OperatorKind,
    PlaneKind,
    ROIFSystem,
    StructuralEntity,
    TemporalMode,
    Vector3,
)
from roif.roif_cascade import (
    CascadeConfig,
    CascadeDirection,
    CascadeUpdateMode,
    run_cascade,
)
from roif.roif_tensor import (
    SelfCouplingMode,
    TensorBuildConfig,
    build_capacity_tensor,
)
from roif.roif_root_detector import (
    CandidateIntervention,
    CandidateStatus,
    FastDetectionMode,
    InterventionKind,
    InterventionPolicy,
    OriginDetectionMode,
    ROIFRole,
    ROIFRootDetectorError,
    RoleEvidence,
    RoleRanking,
    RootCandidate,
    RootDetectionMode,
    RootDetectorConfig,
    RootDetectorThresholds,
    RootDetectorWeights,
    ScoreNormalization,
    aggregate_result_confidence,
    candidate_by_id,
    candidate_channels,
    candidate_confidence,
    candidate_status,
    channel_exposure,
    channel_index_map,
    channel_peak,
    collateral_effect_score,
    detect_d_fast,
    detect_d_origin,
    detect_d_root,
    detect_node_star,
    detect_roif_roles,
    downstream_reach,
    evaluate_candidate,
    evaluate_intervention,
    first_activity_index,
    first_event_index,
    first_threshold_index,
    intervention_is_safe,
    matrix_affected_channel_count,
    matrix_trajectory_burden,
    normalize_scores,
    normalized_early_activity,
    plane_contribution,
    rank_candidates,
    reachable_nodes,
    role_assignments,
    root_result_summary,
    simulate_matrix_cascade,
    tensor_path_influence,
    tensor_sensitivity,
    trajectory_burden,
    validate_detector_inputs,
    virtual_intervention_matrix,
)


def make_channel(
    channel_id: str,
    *,
    capacity: float = 10.0,
    load: float = 0.0,
    activation: float = 1.0,
    mobility: float = 1.0,
    history_factor: float = 1.0,
    state: ChannelState = ChannelState.AVAILABLE,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=f"entity_{channel_id}",
        name=channel_id,
        role=FunctionalRole.TRANSMITTER,
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(capacity=capacity, load=load),
        activation=ActivationAvailability(command=activation),
        geometry=GeometryState(mobility=mobility),
        history_factor=history_factor,
        state=state,
    )


def make_operator(
    operator_id: str,
    source: str,
    target: str,
    gain: float,
    *,
    plane_id: str = "mechanical",
) -> InfluenceOperator:
    return InfluenceOperator(
        operator_id=operator_id,
        name=operator_id,
        plane_id=plane_id,
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=(source,),
        target_ids=(target,),
        gain=gain,
    )


def make_system(
    *,
    channels: tuple[FunctionalChannel, ...] | None = None,
    operators: tuple[InfluenceOperator, ...] | None = None,
) -> ROIFSystem:
    channels = channels or (
        make_channel("a"),
        make_channel("b"),
        make_channel("c"),
    )
    entities = tuple(
        StructuralEntity(
            entity_id=channel.entity_id,
            name=channel.entity_id,
            kind=EntityKind.GENERIC,
            channels=(channel,),
        )
        for channel in channels
    )
    operators = operators or (
        make_operator("a_to_b", "a", "b", 0.8),
        make_operator("b_to_c", "b", "c", 0.6),
    )
    return ROIFSystem(
        system_id="root_detector_test",
        name="Root detector test",
        entities=entities,
        planes=(
            InfluencePlane(
                plane_id="mechanical",
                name="Mechanical",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.STATIC,
            ),
        ),
        operators=operators,
    )


def neutral_tensor_config() -> TensorBuildConfig:
    return TensorBuildConfig(
        tensor_ceiling=100.0,
        source_reserve_exponent=0.0,
        target_reserve_exponent=0.0,
        availability_exponent=0.0,
        geometry_exponent=0.0,
        material_exponent=0.0,
        history_exponent=0.0,
        state_dependent=False,
        self_coupling_mode=SelfCouplingMode.NONE,
    )


def cascade_config(*, steps: int = 4) -> CascadeConfig:
    return CascadeConfig(
        steps=steps,
        update_mode=CascadeUpdateMode.LINEAR,
        direction=CascadeDirection.GENERIC,
        retention=0.0,
        dissipation=0.0,
        lower_bound=0.0,
        upper_bound=100.0,
        clip_state=False,
        convergence_tolerance=1e-12,
        convergence_patience=steps + 2,
        record_transmission_events=True,
    )


def build_fixture(
    *,
    system: ROIFSystem | None = None,
    initial_state: dict[str, float] | None = None,
    steps: int = 4,
):
    system = system or make_system()
    tensor = build_capacity_tensor(system, config=neutral_tensor_config())
    trajectory = run_cascade(
        system,
        initial_state=initial_state or {
            channel_id: 1.0 if channel_id == "a" else 0.0
            for channel_id in system.channel_ids
        },
        tensor_config=neutral_tensor_config(),
        cascade_config=cascade_config(steps=steps),
    )
    return system, tensor, trajectory


def dummy_intervention(candidate_id: str = "a") -> CandidateIntervention:
    return CandidateIntervention(
        candidate_id=candidate_id,
        kind=InterventionKind.RESTORE_CHANNEL,
        baseline_burden=10.0,
        counterfactual_burden=5.0,
        cascade_reduction=5.0,
        relative_reduction=0.5,
        baseline_spectral_radius=0.8,
        counterfactual_spectral_radius=0.4,
        spectral_gain=0.4,
        affected_channel_count=2,
        collateral_effect=0.1,
        intervention_cost=0.1,
        safety_risk=0.1,
        uncertainty=0.1,
        irreversibility=0.0,
        safe=True,
    )


def dummy_candidate(
    channel_id: str,
    *,
    origin: float = 0.0,
    fast: float = 0.0,
    root: float = 0.0,
    node: float = 0.0,
    status: CandidateStatus = CandidateStatus.VALID,
) -> RootCandidate:
    return RootCandidate(
        channel_id=channel_id,
        entity_id=f"entity_{channel_id}",
        status=status,
        d_origin_score=origin,
        d_fast_score=fast,
        d_root_score=root,
        node_star_score=node,
        reserve_factor=1.0,
        reserve_deficit=0.0,
        activation_factor=1.0,
        activation_deficit=0.0,
        geometry_factor=1.0,
        geometry_deficit=0.0,
        material_factor=1.0,
        material_deficit=0.0,
        history_factor=1.0,
        history_effect=0.0,
        first_activity_index=None,
        first_threshold_index=None,
        peak_value=0.0,
        total_exposure=0.0,
        outgoing_strength=0.0,
        incoming_strength=0.0,
        downstream_reach=0.0,
        tensor_sensitivity=0.0,
        plane_contribution=0.0,
        confidence=1.0,
        intervention=dummy_intervention(channel_id),
    )


def test_default_config_is_valid_and_immutable() -> None:
    config = RootDetectorConfig()
    assert config.root_mode is RootDetectionMode.HYBRID
    assert config.fast_mode is FastDetectionMode.HYBRID
    assert config.origin_mode is OriginDetectionMode.HYBRID
    assert isinstance(config.metadata, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        config.root_mode = RootDetectionMode.HYBRID  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("restoration_fraction", -0.1),
        ("restoration_fraction", 1.1),
        ("outgoing_reduction_fraction", -0.1),
        ("incoming_reduction_fraction", 1.1),
    ],
)
def test_config_rejects_invalid_fractions(field_name: str, value: float) -> None:
    with pytest.raises(ROIFRootDetectorError):
        RootDetectorConfig(**{field_name: value})


def test_config_rejects_overlapping_candidate_sets() -> None:
    with pytest.raises(ROIFRootDetectorError):
        RootDetectorConfig(candidate_ids=("a",), excluded_ids=("a",))


def test_config_mappings_are_read_only() -> None:
    config = RootDetectorConfig(
        plane_weights={"mechanical": 2.0},
        intervention_costs={"a": 0.5},
    )
    assert isinstance(config.plane_weights, MappingProxyType)
    assert isinstance(config.intervention_costs, MappingProxyType)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("activity_threshold", -0.1),
        ("failure_threshold", 0.0),
        ("minimum_candidate_confidence", 1.1),
        ("irreversible_action_threshold", -0.1),
        ("numerical_epsilon", 0.0),
    ],
)
def test_thresholds_reject_invalid_values(field_name: str, value: float) -> None:
    with pytest.raises(ROIFRootDetectorError):
        RootDetectorThresholds(**{field_name: value})


def test_weights_reject_negative_values() -> None:
    with pytest.raises(ROIFRootDetectorError):
        RootDetectorWeights(cascade_reduction=-1.0)


def test_role_evidence_is_immutable() -> None:
    evidence = RoleEvidence(
        evidence_id="a:reserve",
        role=ROIFRole.D_FAST,
        value=0.5,
        weight=1.0,
        weighted_value=0.5,
        description="Reserve deficit.",
    )
    assert isinstance(evidence.metadata, MappingProxyType)


def test_candidate_intervention_is_immutable() -> None:
    intervention = dummy_intervention()
    assert intervention.safe is True
    assert isinstance(intervention.metadata, MappingProxyType)


def test_root_candidate_score_for_each_role() -> None:
    candidate = dummy_candidate("a", origin=0.1, fast=0.2, root=0.3, node=0.4)
    assert candidate.score_for(ROIFRole.D_ORIGIN) == pytest.approx(0.1)
    assert candidate.score_for(ROIFRole.D_FAST) == pytest.approx(0.2)
    assert candidate.score_for(ROIFRole.D_ROOT) == pytest.approx(0.3)
    assert candidate.score_for(ROIFRole.NODE_STAR) == pytest.approx(0.4)


def test_role_ranking_requires_descending_scores() -> None:
    with pytest.raises(ROIFRootDetectorError):
        RoleRanking(
            role=ROIFRole.D_ROOT,
            candidates=(
                dummy_candidate("a", root=0.1),
                dummy_candidate("b", root=0.9),
            ),
        )


def test_role_ranking_winner_and_position() -> None:
    ranking = RoleRanking(
        role=ROIFRole.D_ROOT,
        candidates=(
            dummy_candidate("b", root=0.9),
            dummy_candidate("a", root=0.1),
        ),
    )
    assert ranking.winner.channel_id == "b"
    assert ranking.position("a") == 1


def test_candidate_channels_defaults_to_all_channels() -> None:
    result = candidate_channels(make_system(), RootDetectorConfig())
    assert tuple(channel.channel_id for channel in result) == ("a", "b", "c")


def test_candidate_channels_respects_selection() -> None:
    system = make_system()
    config = RootDetectorConfig(
        candidate_ids=("c", "a"),
    )

    result = candidate_channels(system, config)

    assert tuple(channel.channel_id for channel in result) == ("c", "a")


def test_candidate_channels_respects_exclusion() -> None:
    system = make_system()
    config = RootDetectorConfig(
        excluded_ids=("a",),
    )

    result = candidate_channels(system, config)

    assert tuple(
        channel.channel_id
        for channel in result
    ) == ("b", "c")


def test_candidate_channels_rejects_unknown_id() -> None:
    with pytest.raises(ROIFRootDetectorError):
        candidate_channels(
            make_system(),
            RootDetectorConfig(candidate_ids=("missing",)),
        )


def test_channel_index_map_is_read_only() -> None:
    mapping = channel_index_map(("a", "b"))
    assert isinstance(mapping, MappingProxyType)
    assert mapping["b"] == 1


def test_validate_detector_inputs_accepts_matching_axes() -> None:
    system, tensor, trajectory = build_fixture()
    validate_detector_inputs(system, tensor, trajectory)


def test_trajectory_burden_is_total_absolute_exposure() -> None:
    _, _, trajectory = build_fixture(steps=2)
    assert trajectory_burden(trajectory) == pytest.approx(
        float(np.sum(np.abs(trajectory.state_history)))
    )


def test_first_activity_index_detects_initial_activity() -> None:
    _, _, trajectory = build_fixture()
    assert first_activity_index(trajectory, "a") == 0


def test_first_threshold_index() -> None:
    _, _, trajectory = build_fixture()
    assert first_threshold_index(trajectory, "a", threshold=0.5) == 0


def test_first_event_index_finds_transmission_event() -> None:
    _, _, trajectory = build_fixture()
    assert first_event_index(trajectory, "a") is not None


def test_channel_peak_and_exposure() -> None:
    _, _, trajectory = build_fixture()
    assert channel_peak(trajectory, "a") == pytest.approx(1.0)
    assert channel_exposure(trajectory, "a") >= 1.0


@pytest.mark.parametrize(
    ("first_index", "steps", "expected"),
    [(None, 4, 0.0), (0, 4, 1.0), (2, 4, 0.5), (4, 4, 0.0)],
)
def test_normalized_early_activity(first_index, steps, expected) -> None:
    assert normalized_early_activity(first_index, steps) == pytest.approx(expected)


def test_reachable_nodes_follow_tensor_direction() -> None:
    _, tensor, _ = build_fixture()
    assert reachable_nodes(tensor, "a") == frozenset({"b", "c"})
    assert reachable_nodes(tensor, "b") == frozenset({"c"})
    assert reachable_nodes(tensor, "c") == frozenset()


def test_downstream_reach_is_fraction_of_other_nodes() -> None:
    _, tensor, _ = build_fixture()
    assert downstream_reach(tensor, "a") == pytest.approx(1.0)
    assert downstream_reach(tensor, "b") == pytest.approx(0.5)
    assert downstream_reach(tensor, "c") == pytest.approx(0.0)


def test_tensor_path_influence_is_largest_for_upstream_source() -> None:
    _, tensor, _ = build_fixture()
    assert tensor_path_influence(tensor, "a") > tensor_path_influence(tensor, "c")


def test_tensor_sensitivity_is_nonnegative() -> None:
    _, tensor, _ = build_fixture()
    assert tensor_sensitivity(tensor, "a") >= 0.0
    assert tensor_sensitivity(tensor, "b") >= 0.0


def test_plane_contribution_uses_plane_weight() -> None:
    _, tensor, _ = build_fixture()
    base = plane_contribution(tensor, "a", {"mechanical": 1.0})
    doubled = plane_contribution(tensor, "a", {"mechanical": 2.0})
    assert doubled == pytest.approx(2.0 * base)


def test_min_max_normalization() -> None:
    result = normalize_scores({"a": 1.0, "b": 3.0}, ScoreNormalization.MIN_MAX)
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx(1.0)


def test_equal_min_max_scores_become_ones() -> None:
    result = normalize_scores({"a": 2.0, "b": 2.0}, ScoreNormalization.MIN_MAX)
    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(1.0)


def test_l1_normalization() -> None:
    result = normalize_scores({"a": 1.0, "b": 3.0}, ScoreNormalization.L1)
    assert result["a"] == pytest.approx(0.25)
    assert result["b"] == pytest.approx(0.75)


def test_max_abs_normalization() -> None:
    result = normalize_scores({"a": -2.0, "b": 4.0}, ScoreNormalization.MAX_ABS)
    assert result["a"] == pytest.approx(-0.5)
    assert result["b"] == pytest.approx(1.0)


def test_rank_candidates_sorts_by_role_then_channel_id() -> None:
    ranking = rank_candidates(
        (
            dummy_candidate("b", root=0.5),
            dummy_candidate("a", root=0.5),
            dummy_candidate("c", root=0.1),
        ),
        ROIFRole.D_ROOT,
    )
    assert tuple(candidate.channel_id for candidate in ranking.candidates) == (
        "a",
        "b",
        "c",
    )


def test_remove_outgoing_intervention_zeroes_candidate_column() -> None:
    _, tensor, _ = build_fixture()
    matrix = virtual_intervention_matrix(
        tensor,
        "a",
        RootDetectorConfig(
            intervention_kind=InterventionKind.REMOVE_OUTGOING_INFLUENCE
        ),
    )
    assert np.allclose(matrix[:, tensor.index("a")], 0.0)


def test_reduce_outgoing_intervention_scales_column() -> None:
    _, tensor, _ = build_fixture()
    config = RootDetectorConfig(
        intervention_kind=InterventionKind.REDUCE_OUTGOING_INFLUENCE,
        outgoing_reduction_fraction=0.5,
    )
    matrix = virtual_intervention_matrix(tensor, "a", config)
    assert np.allclose(
        matrix[:, tensor.index("a")],
        0.5 * tensor.matrix[:, tensor.index("a")],
    )


def test_reduce_incoming_intervention_scales_row() -> None:
    _, tensor, _ = build_fixture()
    config = RootDetectorConfig(
        intervention_kind=InterventionKind.REDUCE_INCOMING_LOAD,
        incoming_reduction_fraction=0.5,
    )
    matrix = virtual_intervention_matrix(tensor, "b", config)
    assert np.allclose(
        matrix[tensor.index("b"), :],
        0.5 * tensor.matrix[tensor.index("b"), :],
    )


def test_custom_intervention_requires_adapter() -> None:
    _, tensor, _ = build_fixture()
    with pytest.raises(ROIFRootDetectorError):
        virtual_intervention_matrix(
            tensor,
            "a",
            RootDetectorConfig(intervention_kind=InterventionKind.CUSTOM),
        )


def test_simulate_matrix_cascade_returns_initial_plus_steps() -> None:
    history = simulate_matrix_cascade(
        np.array([[0.0, 0.0], [0.5, 0.0]]),
        np.array([2.0, 0.0]),
        steps=2,
    )
    assert history.shape == (3, 2)
    assert np.array_equal(history[0], np.array([2.0, 0.0]))
    assert np.array_equal(history[1], np.array([0.0, 1.0]))


def test_matrix_trajectory_burden() -> None:
    history = np.array([[1.0, 0.0], [0.0, 0.5]])
    assert matrix_trajectory_burden(history) == pytest.approx(1.5)


def test_matrix_affected_channel_count() -> None:
    baseline = np.zeros((3, 2))
    counterfactual = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    assert matrix_affected_channel_count(baseline, counterfactual) == 2


def test_collateral_effect_excludes_candidate_channel() -> None:
    baseline = np.array([[1.0, 1.0], [1.0, 1.0]])
    counterfactual = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert collateral_effect_score(
        baseline,
        counterfactual,
        candidate_index=0,
    ) == pytest.approx(0.0)


def test_evaluate_intervention_reduces_upstream_chain_burden() -> None:
    _, tensor, trajectory = build_fixture()
    intervention = evaluate_intervention(
        "a",
        tensor,
        trajectory,
        RootDetectorConfig(
            intervention_kind=InterventionKind.REMOVE_OUTGOING_INFLUENCE
        ),
    )
    assert intervention.cascade_reduction >= 0.0
    assert intervention.relative_reduction >= 0.0


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (InterventionPolicy.OBSERVATIONAL, True),
        (InterventionPolicy.STRICT_LINEAR, False),
    ],
)
def test_intervention_policy_changes_safety(policy, expected) -> None:
    config = RootDetectorConfig(
        intervention_policy=policy,
        thresholds=RootDetectorThresholds(
            failure_threshold=1.0,
            irreversible_action_threshold=1.0,
        ),
    )
    assert intervention_is_safe(
        collateral_effect=0.0,
        safety_risk=0.0,
        uncertainty=0.0,
        irreversibility=0.0,
        counterfactual_history=np.array([[2.0]]),
        config=config,
    ) is expected


def test_candidate_confidence_is_bounded() -> None:
    system, tensor, trajectory = build_fixture()
    confidence = candidate_confidence(
        "a",
        tensor,
        trajectory,
        RootDetectorConfig(),
    )
    assert 0.0 <= confidence <= 1.0


def test_failed_candidate_can_be_excluded() -> None:
    channel = make_channel("a", state=ChannelState.FAILED)
    status = candidate_status(
        channel,
        confidence=1.0,
        config=RootDetectorConfig(exclude_failed_candidates=True),
    )
    assert status is CandidateStatus.EXCLUDED


def test_evaluate_candidate_returns_all_four_scores() -> None:
    system, tensor, trajectory = build_fixture()
    candidate = evaluate_candidate(
        system.channel("a"),
        system,
        tensor,
        trajectory,
        RootDetectorConfig(),
    )
    assert math.isfinite(candidate.d_origin_score)
    assert math.isfinite(candidate.d_fast_score)
    assert math.isfinite(candidate.d_root_score)
    assert math.isfinite(candidate.node_star_score)
    assert candidate.intervention is not None
    assert candidate.evidence


def test_detect_roif_roles_returns_complete_rankings() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(system, tensor, trajectory)
    assert result.candidate_count == 3
    assert len(result.origin_ranking.candidates) == 3
    assert len(result.fast_ranking.candidates) == 3
    assert len(result.root_ranking.candidates) == 3
    assert len(result.node_star_ranking.candidates) == 3


def test_result_role_ids_match_ranking_winners() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(system, tensor, trajectory)
    assert result.d_origin == result.origin_ranking.winner.channel_id
    assert result.d_fast == result.fast_ranking.winner.channel_id
    assert result.d_root == result.root_ranking.winner.channel_id
    assert result.node_star == result.node_star_ranking.winner.channel_id


def test_d_fast_minimum_reserve_mode_prefers_low_reserve_channel() -> None:
    system = make_system(
        channels=(
            make_channel("a", capacity=10.0, load=1.0),
            make_channel("b", capacity=10.0, load=9.0),
            make_channel("c", capacity=10.0, load=2.0),
        )
    )
    tensor = build_capacity_tensor(system)
    trajectory = run_cascade(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        tensor_config=TensorBuildConfig(
            tensor_ceiling=100.0,
            self_coupling_mode=SelfCouplingMode.NONE,
        ),
        cascade_config=cascade_config(),
    )
    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        RootDetectorConfig(fast_mode=FastDetectionMode.MINIMUM_RESERVE),
    )
    assert result.d_fast == "b"


def test_d_origin_earliest_activity_prefers_initially_active_channel() -> None:
    system, tensor, trajectory = build_fixture(
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0}
    )
    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        RootDetectorConfig(origin_mode=OriginDetectionMode.EARLIEST_ACTIVITY),
    )
    assert result.d_origin == "a"


def test_d_root_virtual_restoration_returns_valid_candidate() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        RootDetectorConfig(
            root_mode=RootDetectionMode.VIRTUAL_RESTORATION,
            intervention_kind=InterventionKind.REMOVE_OUTGOING_INFLUENCE,
        ),
    )
    assert result.d_root in {"a", "b", "c"}
    assert result.root_ranking.winner.intervention is not None


def test_node_star_can_be_penalized_by_cost() -> None:
    system, tensor, trajectory = build_fixture()
    baseline = detect_roif_roles(system, tensor, trajectory)
    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        RootDetectorConfig(intervention_costs={baseline.d_root: 1.0}),
    )
    assert result.node_star in system.channel_ids


def test_distinct_role_policy_uses_available_nodes() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        RootDetectorConfig(allow_same_role_node=False),
    )
    assert result.d_origin in system.channel_ids
    assert result.d_fast in system.channel_ids
    assert result.d_root in system.channel_ids
    assert result.node_star in system.channel_ids


def test_detection_is_deterministic() -> None:
    system, tensor, trajectory = build_fixture()
    left = detect_roif_roles(system, tensor, trajectory)
    right = detect_roif_roles(system, tensor, trajectory)
    assert left.d_origin == right.d_origin
    assert left.d_fast == right.d_fast
    assert left.d_root == right.d_root
    assert left.node_star == right.node_star
    assert left.root_ranking.candidates == right.root_ranking.candidates


def test_convenience_detectors_match_complete_result() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(system, tensor, trajectory)
    assert detect_d_origin(system, tensor, trajectory).channel_id == result.d_origin
    assert detect_d_fast(system, tensor, trajectory).channel_id == result.d_fast
    assert detect_d_root(system, tensor, trajectory).channel_id == result.d_root
    assert detect_node_star(system, tensor, trajectory).channel_id == result.node_star


def test_candidate_by_id_and_role_assignments() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(system, tensor, trajectory)
    candidate = candidate_by_id(result, result.d_root)
    assignments = role_assignments(result)
    assert candidate.channel_id == result.d_root
    assert assignments[ROIFRole.D_ROOT] == result.d_root
    assert isinstance(assignments, MappingProxyType)


def test_root_result_summary_is_read_only() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(system, tensor, trajectory)
    summary = root_result_summary(result)
    assert isinstance(summary, MappingProxyType)
    assert summary["d_root"] == result.d_root
    assert summary["node_star"] == result.node_star


def test_aggregate_result_confidence() -> None:
    candidates = (dummy_candidate("a"), dummy_candidate("b"))
    assert aggregate_result_confidence(candidates) == pytest.approx(1.0)


def test_detector_does_not_require_hidden_truth_labels() -> None:
    system, tensor, trajectory = build_fixture()
    result = detect_roif_roles(system, tensor, trajectory)
    assert result.d_root in system.channel_ids
    assert result.node_star in system.channel_ids

