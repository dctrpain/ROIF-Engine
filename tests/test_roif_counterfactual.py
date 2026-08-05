"""
Tests for roif.roif_counterfactual.

The suite verifies the ROIF counterfactual contract:

- immutable configuration and scenario objects;
- single-channel and multi-channel interventions;
- matrix-only, tensor-rebuild, and full-cascade execution;
- adaptive, strict-linear, and observational policies;
- cascade burden, spectral gain, collateral effect, cost, risk,
  uncertainty, irreversibility, and utility;
- deterministic batch ranking and Pareto-front extraction.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from roif.roif_entities import (
    ActivationAvailability,
    CapacityState,
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
    InterventionPolicy,
    RootDetectorThresholds,
)
from roif.roif_counterfactual import (
    CounterfactualAction,
    CounterfactualBatchResult,
    CounterfactualConfig,
    CounterfactualExecutionMode,
    CounterfactualIntervention,
    CounterfactualMetrics,
    CounterfactualObjective,
    CounterfactualResult,
    CounterfactualScenario,
    CounterfactualStatus,
    CounterfactualWeights,
    ROIFCounterfactualError,
    apply_intervention_to_matrix,
    apply_scenario_to_matrix,
    apply_scenario_to_system,
    batch_summary,
    counterfactual_summary,
    dominates,
    evaluate_counterfactual,
    evaluate_counterfactual_batch,
    generate_single_channel_scenarios,
    mark_dominated_results,
    pareto_front,
    result_by_scenario_id,
    scenario_costs,
    validate_counterfactual_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_channel(
    channel_id: str,
    *,
    capacity: float = 10.0,
    load: float = 0.0,
    activation: float = 1.0,
    mobility: float = 1.0,
    history_factor: float = 1.0,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=f"entity_{channel_id}",
        name=channel_id,
        role=FunctionalRole.TRANSMITTER,
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(
            capacity=capacity,
            load=load,
        ),
        activation=ActivationAvailability(
            command=activation,
        ),
        geometry=GeometryState(
            mobility=mobility,
        ),
        history_factor=history_factor,
    )


def make_operator(
    operator_id: str,
    source: str,
    target: str,
    gain: float,
) -> InfluenceOperator:
    return InfluenceOperator(
        operator_id=operator_id,
        name=operator_id,
        plane_id="mechanical",
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=(source,),
        target_ids=(target,),
        gain=gain,
    )


def make_system() -> ROIFSystem:
    channels = (
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

    return ROIFSystem(
        system_id="counterfactual_test",
        name="Counterfactual test",
        entities=entities,
        planes=(
            InfluencePlane(
                plane_id="mechanical",
                name="Mechanical",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.STATIC,
            ),
        ),
        operators=(
            make_operator("a_to_b", "a", "b", 0.8),
            make_operator("b_to_c", "b", "c", 0.6),
        ),
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


def simple_cascade_config(steps: int = 4) -> CascadeConfig:
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
        record_transmission_events=False,
    )


def build_fixture():
    system = make_system()
    tensor_config = neutral_tensor_config()
    tensor = build_capacity_tensor(
        system,
        config=tensor_config,
    )
    trajectory = run_cascade(
        system,
        initial_state={"a": 1.0, "b": 0.0, "c": 0.0},
        tensor_config=tensor_config,
        cascade_config=simple_cascade_config(),
    )
    return system, tensor, trajectory


def make_intervention(
    channel_id: str = "a",
    *,
    action: CounterfactualAction = (
        CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
    ),
    magnitude: float = 1.0,
    target_value: float | None = None,
    cost: float | None = None,
    safety_risk: float | None = None,
    uncertainty: float | None = None,
    irreversibility: float | None = None,
) -> CounterfactualIntervention:
    return CounterfactualIntervention(
        intervention_id=f"{action.value}:{channel_id}",
        channel_id=channel_id,
        action=action,
        magnitude=magnitude,
        target_value=target_value,
        cost=cost,
        safety_risk=safety_risk,
        uncertainty=uncertainty,
        irreversibility=irreversibility,
    )


def make_scenario(
    scenario_id: str = "scenario_a",
    *,
    interventions: tuple[CounterfactualIntervention, ...] | None = None,
) -> CounterfactualScenario:
    return CounterfactualScenario(
        scenario_id=scenario_id,
        name=scenario_id,
        interventions=interventions or (make_intervention(),),
    )


def dummy_metrics(
    *,
    reduction: float,
    spectral_gain: float,
    cost: float,
    risk: float,
    utility: float,
    safe: bool = True,
) -> CounterfactualMetrics:
    return CounterfactualMetrics(
        baseline_burden=10.0,
        scenario_burden=10.0 * (1.0 - reduction),
        cascade_reduction=10.0 * reduction,
        relative_reduction=reduction,
        baseline_final_burden=2.0,
        scenario_final_burden=1.0,
        final_state_reduction=0.5,
        baseline_peak_burden=3.0,
        scenario_peak_burden=2.0,
        peak_reduction=1.0 / 3.0,
        baseline_spectral_radius=0.8,
        scenario_spectral_radius=0.8 - spectral_gain,
        spectral_gain=spectral_gain,
        affected_channel_count=2,
        collateral_effect=0.1,
        cost=cost,
        safety_risk=risk,
        uncertainty=0.1,
        irreversibility=0.0,
        complexity=0.2,
        utility=utility,
        safe=safe,
    )


def dummy_result(
    scenario_id: str,
    *,
    reduction: float,
    spectral_gain: float,
    cost: float,
    risk: float,
    utility: float,
) -> CounterfactualResult:
    matrix = np.zeros((2, 2))
    history = np.zeros((2, 2))
    return CounterfactualResult(
        scenario=CounterfactualScenario(
            scenario_id=scenario_id,
            name=scenario_id,
            interventions=(
                make_intervention(
                    channel_id="a",
                ),
            ),
        ),
        status=CounterfactualStatus.VALID,
        metrics=dummy_metrics(
            reduction=reduction,
            spectral_gain=spectral_gain,
            cost=cost,
            risk=risk,
            utility=utility,
        ),
        baseline_matrix=matrix,
        scenario_matrix=matrix,
        baseline_history=history,
        scenario_history=history,
        channel_ids=("a", "b"),
    )


# ---------------------------------------------------------------------------
# Configuration and immutable objects
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_immutable() -> None:
    config = CounterfactualConfig()

    assert config.execution_mode is (
        CounterfactualExecutionMode.MATRIX_ONLY
    )
    assert config.objective is (
        CounterfactualObjective.BALANCED_UTILITY
    )
    assert isinstance(config.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        config.steps = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("steps", 0),
        ("dissipation", -0.1),
        ("dissipation", 1.1),
        ("retention", -0.1),
        ("default_cost", -0.1),
        ("default_safety_risk", 1.1),
        ("default_uncertainty", -0.1),
        ("default_irreversibility", 1.1),
    ],
)
def test_config_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ROIFCounterfactualError):
        CounterfactualConfig(**{field_name: value})


def test_config_rejects_inverted_clip_bounds() -> None:
    with pytest.raises(ROIFCounterfactualError):
        CounterfactualConfig(
            clip_min=2.0,
            clip_max=1.0,
        )


def test_weights_reject_negative_values() -> None:
    with pytest.raises(ROIFCounterfactualError):
        CounterfactualWeights(cost=-1.0)


def test_intervention_is_immutable() -> None:
    intervention = make_intervention()

    assert isinstance(intervention.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        intervention.magnitude = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("intervention_id", ""),
        ("channel_id", ""),
        ("magnitude", -0.1),
        ("magnitude", 1.1),
        ("cost", 1.1),
    ],
)
def test_intervention_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "intervention_id": "x",
        "channel_id": "a",
        "action": CounterfactualAction.RESTORE_CHANNEL,
    }
    kwargs[field_name] = value

    with pytest.raises(ROIFCounterfactualError):
        CounterfactualIntervention(**kwargs)


def test_scenario_requires_interventions() -> None:
    with pytest.raises(ROIFCounterfactualError):
        CounterfactualScenario(
            scenario_id="empty",
            name="Empty",
            interventions=(),
        )


def test_scenario_rejects_duplicate_intervention_ids() -> None:
    first = make_intervention("a")
    second = CounterfactualIntervention(
        intervention_id=first.intervention_id,
        channel_id="b",
        action=CounterfactualAction.RESTORE_CHANNEL,
    )

    with pytest.raises(ROIFCounterfactualError):
        CounterfactualScenario(
            scenario_id="duplicate",
            name="Duplicate",
            interventions=(first, second),
        )


def test_result_arrays_are_read_only() -> None:
    result = dummy_result(
        "x",
        reduction=0.5,
        spectral_gain=0.1,
        cost=0.1,
        risk=0.1,
        utility=1.0,
    )

    assert result.baseline_matrix.flags.writeable is False
    assert result.scenario_history.flags.writeable is False


# ---------------------------------------------------------------------------
# Input validation and cost aggregation
# ---------------------------------------------------------------------------


def test_validate_inputs_accepts_matching_axes() -> None:
    system, tensor, trajectory = build_fixture()

    validate_counterfactual_inputs(
        system,
        tensor,
        trajectory,
    )


def test_scenario_costs_average_interventions() -> None:
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                cost=0.2,
                safety_risk=0.4,
                uncertainty=0.6,
                irreversibility=0.8,
            ),
            make_intervention(
                "b",
                cost=0.4,
                safety_risk=0.2,
                uncertainty=0.2,
                irreversibility=0.0,
            ),
        )
    )

    cost, risk, uncertainty, irreversibility, complexity = (
        scenario_costs(
            scenario,
            CounterfactualConfig(),
        )
    )

    assert cost == pytest.approx(0.3)
    assert risk == pytest.approx(0.3)
    assert uncertainty == pytest.approx(0.4)
    assert irreversibility == pytest.approx(0.4)
    assert 0.0 <= complexity <= 1.0


# ---------------------------------------------------------------------------
# Matrix interventions
# ---------------------------------------------------------------------------


def test_remove_outgoing_zeroes_column() -> None:
    _, tensor, _ = build_fixture()
    matrix = apply_intervention_to_matrix(
        tensor.matrix,
        tensor.channel_ids,
        make_intervention(
            "a",
            action=(
                CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
            ),
        ),
    )

    assert np.allclose(
        matrix[:, tensor.index("a")],
        0.0,
    )


def test_reduce_outgoing_scales_column() -> None:
    _, tensor, _ = build_fixture()
    matrix = apply_intervention_to_matrix(
        tensor.matrix,
        tensor.channel_ids,
        make_intervention(
            "a",
            action=(
                CounterfactualAction.REDUCE_OUTGOING_INFLUENCE
            ),
            magnitude=0.5,
        ),
    )

    assert np.allclose(
        matrix[:, tensor.index("a")],
        0.5 * tensor.matrix[:, tensor.index("a")],
    )


def test_reduce_incoming_scales_row() -> None:
    _, tensor, _ = build_fixture()
    matrix = apply_intervention_to_matrix(
        tensor.matrix,
        tensor.channel_ids,
        make_intervention(
            "b",
            action=CounterfactualAction.REDUCE_INCOMING_LOAD,
            magnitude=0.5,
        ),
    )

    assert np.allclose(
        matrix[tensor.index("b"), :],
        0.5 * tensor.matrix[tensor.index("b"), :],
    )


def test_restore_channel_scales_row_and_column() -> None:
    _, tensor, _ = build_fixture()
    matrix = apply_intervention_to_matrix(
        tensor.matrix,
        tensor.channel_ids,
        make_intervention(
            "b",
            action=CounterfactualAction.RESTORE_CHANNEL,
            magnitude=0.5,
        ),
    )

    index = tensor.index("b")
    assert np.allclose(
        matrix[:, index],
        0.5 * tensor.matrix[:, index],
    )
    assert np.allclose(
        matrix[index, :],
        0.5 * tensor.matrix[index, :],
    )


def test_state_change_action_scales_source_column() -> None:
    _, tensor, _ = build_fixture()
    matrix = apply_intervention_to_matrix(
        tensor.matrix,
        tensor.channel_ids,
        make_intervention(
            "a",
            action=CounterfactualAction.CHANGE_ACTIVATION,
            target_value=0.25,
        ),
    )

    assert np.allclose(
        matrix[:, tensor.index("a")],
        0.25 * tensor.matrix[:, tensor.index("a")],
    )


def test_custom_matrix_action_requires_adapter() -> None:
    _, tensor, _ = build_fixture()

    with pytest.raises(ROIFCounterfactualError):
        apply_intervention_to_matrix(
            tensor.matrix,
            tensor.channel_ids,
            make_intervention(
                "a",
                action=CounterfactualAction.CUSTOM,
            ),
        )


def test_unknown_matrix_channel_is_rejected() -> None:
    _, tensor, _ = build_fixture()

    with pytest.raises(ROIFCounterfactualError):
        apply_intervention_to_matrix(
            tensor.matrix,
            tensor.channel_ids,
            make_intervention("missing"),
        )


def test_scenario_interventions_are_applied_in_order() -> None:
    _, tensor, _ = build_fixture()
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                action=(
                    CounterfactualAction.REDUCE_OUTGOING_INFLUENCE
                ),
                magnitude=0.5,
            ),
            CounterfactualIntervention(
                intervention_id="second:a",
                channel_id="a",
                action=(
                    CounterfactualAction.REDUCE_OUTGOING_INFLUENCE
                ),
                magnitude=0.5,
            ),
        )
    )

    matrix = apply_scenario_to_matrix(tensor, scenario)

    assert np.allclose(
        matrix[:, tensor.index("a")],
        0.25 * tensor.matrix[:, tensor.index("a")],
    )


# ---------------------------------------------------------------------------
# System interventions
# ---------------------------------------------------------------------------


def test_restore_channel_updates_channel_state() -> None:
    system = make_system()
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                action=CounterfactualAction.RESTORE_CHANNEL,
                magnitude=1.0,
            ),
        )
    )

    updated = apply_scenario_to_system(system, scenario)
    channel = updated.channel("a")

    assert channel.capacity_state.load == pytest.approx(0.0)
    assert channel.activation.command == pytest.approx(1.0)
    assert channel.geometry.mobility == pytest.approx(1.0)
    assert channel.history_factor == pytest.approx(1.0)


def test_change_activation_updates_command() -> None:
    system = make_system()
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                action=CounterfactualAction.CHANGE_ACTIVATION,
                target_value=0.25,
            ),
        )
    )

    updated = apply_scenario_to_system(system, scenario)

    assert updated.channel("a").activation.command == pytest.approx(0.25)


def test_change_geometry_updates_mobility() -> None:
    system = make_system()
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                action=CounterfactualAction.CHANGE_GEOMETRY,
                target_value=0.4,
            ),
        )
    )

    updated = apply_scenario_to_system(system, scenario)

    assert updated.channel("a").geometry.mobility == pytest.approx(0.4)


def test_change_history_updates_history_factor() -> None:
    system = make_system()
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                action=CounterfactualAction.CHANGE_HISTORY,
                target_value=0.3,
            ),
        )
    )

    updated = apply_scenario_to_system(system, scenario)

    assert updated.channel("a").history_factor == pytest.approx(0.3)


def test_unknown_system_channel_is_rejected() -> None:
    with pytest.raises(ROIFCounterfactualError):
        apply_scenario_to_system(
            make_system(),
            make_scenario(
                interventions=(make_intervention("missing"),),
            ),
        )


# ---------------------------------------------------------------------------
# Complete evaluation
# ---------------------------------------------------------------------------


def test_matrix_only_evaluation_returns_valid_result() -> None:
    system, tensor, trajectory = build_fixture()

    result = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        make_scenario(),
    )

    assert result.status in {
        CounterfactualStatus.VALID,
        CounterfactualStatus.UNSAFE,
    }
    assert result.metrics.cascade_reduction >= 0.0
    assert result.channel_ids == tensor.channel_ids


def test_rebuild_tensor_execution_runs() -> None:
    system, tensor, trajectory = build_fixture()
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                action=CounterfactualAction.CHANGE_ACTIVATION,
                target_value=0.5,
            ),
        )
    )

    result = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        scenario,
        CounterfactualConfig(
            execution_mode=(
                CounterfactualExecutionMode.REBUILD_TENSOR
            ),
        ),
        tensor_config=neutral_tensor_config(),
    )

    assert result.scenario_matrix.shape == tensor.matrix.shape


def test_full_cascade_execution_runs() -> None:
    system, tensor, trajectory = build_fixture()
    scenario = make_scenario()

    result = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        scenario,
        CounterfactualConfig(
            execution_mode=(
                CounterfactualExecutionMode.FULL_CASCADE
            ),
            steps=4,
        ),
        tensor_config=neutral_tensor_config(),
        cascade_config=simple_cascade_config(),
    )

    assert result.scenario_history.shape == result.baseline_history.shape


def test_strict_linear_policy_can_mark_scenario_unsafe() -> None:
    system, tensor, trajectory = build_fixture()
    result = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        make_scenario(),
        CounterfactualConfig(
            policy=InterventionPolicy.STRICT_LINEAR,
            thresholds=RootDetectorThresholds(
                failure_threshold=0.1,
                irreversible_action_threshold=1.0,
            ),
        ),
    )

    assert result.metrics.safe is False
    assert result.status is CounterfactualStatus.UNSAFE


def test_reject_unsafe_marks_invalid() -> None:
    system, tensor, trajectory = build_fixture()
    result = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        make_scenario(),
        CounterfactualConfig(
            policy=InterventionPolicy.STRICT_LINEAR,
            reject_unsafe=True,
            thresholds=RootDetectorThresholds(
                failure_threshold=0.1,
                irreversible_action_threshold=1.0,
            ),
        ),
    )

    assert result.status is CounterfactualStatus.INVALID


def test_high_irreversibility_can_make_scenario_unsafe() -> None:
    system, tensor, trajectory = build_fixture()
    scenario = make_scenario(
        interventions=(
            make_intervention(
                "a",
                irreversibility=1.0,
            ),
        )
    )
    result = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        scenario,
        CounterfactualConfig(
            thresholds=RootDetectorThresholds(
                irreversible_action_threshold=0.05,
            ),
        ),
    )

    assert result.metrics.safe is False


def test_counterfactual_evaluation_is_deterministic() -> None:
    system, tensor, trajectory = build_fixture()
    scenario = make_scenario()

    left = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        scenario,
    )
    right = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        scenario,
    )

    assert left.metrics == right.metrics
    assert np.array_equal(
        left.scenario_history,
        right.scenario_history,
    )


def test_counterfactual_summary_is_read_only() -> None:
    system, tensor, trajectory = build_fixture()
    result = evaluate_counterfactual(
        system,
        tensor,
        trajectory,
        make_scenario(),
    )

    summary = counterfactual_summary(result)

    assert isinstance(summary, MappingProxyType)
    assert summary["scenario_id"] == "scenario_a"
    assert summary["utility"] == pytest.approx(
        result.metrics.utility
    )


# ---------------------------------------------------------------------------
# Pareto dominance and batch ranking
# ---------------------------------------------------------------------------


def test_dominates_when_no_worse_and_strictly_better() -> None:
    left = dummy_result(
        "left",
        reduction=0.8,
        spectral_gain=0.5,
        cost=0.1,
        risk=0.1,
        utility=2.0,
    )
    right = dummy_result(
        "right",
        reduction=0.5,
        spectral_gain=0.3,
        cost=0.2,
        risk=0.2,
        utility=1.0,
    )

    assert dominates(left, right) is True
    assert dominates(right, left) is False


def test_pareto_front_excludes_dominated_result() -> None:
    best = dummy_result(
        "best",
        reduction=0.8,
        spectral_gain=0.5,
        cost=0.1,
        risk=0.1,
        utility=2.0,
    )
    dominated = dummy_result(
        "dominated",
        reduction=0.5,
        spectral_gain=0.3,
        cost=0.2,
        risk=0.2,
        utility=1.0,
    )

    front = pareto_front((dominated, best))

    assert tuple(
        item.scenario.scenario_id
        for item in front
    ) == ("best",)


def test_mark_dominated_results_changes_status() -> None:
    best = dummy_result(
        "best",
        reduction=0.8,
        spectral_gain=0.5,
        cost=0.1,
        risk=0.1,
        utility=2.0,
    )
    dominated = dummy_result(
        "dominated",
        reduction=0.5,
        spectral_gain=0.3,
        cost=0.2,
        risk=0.2,
        utility=1.0,
    )

    marked = mark_dominated_results((best, dominated))
    by_id = {
        item.scenario.scenario_id: item
        for item in marked
    }

    assert by_id["best"].status is CounterfactualStatus.VALID
    assert (
        by_id["dominated"].status
        is CounterfactualStatus.DOMINATED
    )


def test_batch_requires_scenarios() -> None:
    system, tensor, trajectory = build_fixture()

    with pytest.raises(ROIFCounterfactualError):
        evaluate_counterfactual_batch(
            system,
            tensor,
            trajectory,
            (),
        )


def test_batch_rejects_duplicate_scenario_ids() -> None:
    system, tensor, trajectory = build_fixture()
    scenario = make_scenario("same")

    with pytest.raises(ROIFCounterfactualError):
        evaluate_counterfactual_batch(
            system,
            tensor,
            trajectory,
            (scenario, scenario),
        )


def test_batch_ranks_by_balanced_utility() -> None:
    system, tensor, trajectory = build_fixture()
    scenarios = (
        make_scenario(
            "a",
            interventions=(
                make_intervention(
                    "a",
                    cost=0.0,
                ),
            ),
        ),
        make_scenario(
            "b",
            interventions=(
                make_intervention(
                    "b",
                    cost=1.0,
                ),
            ),
        ),
    )

    batch = evaluate_counterfactual_batch(
        system,
        tensor,
        trajectory,
        scenarios,
        CounterfactualConfig(
            mark_dominated=False,
            objective=(
                CounterfactualObjective.BALANCED_UTILITY
            ),
        ),
    )

    utilities = [
        result.metrics.utility
        for result in batch.results
    ]
    assert utilities == sorted(utilities, reverse=True)


def test_batch_winner_property() -> None:
    system, tensor, trajectory = build_fixture()
    scenarios = generate_single_channel_scenarios(
        system.channel_ids
    )

    batch = evaluate_counterfactual_batch(
        system,
        tensor,
        trajectory,
        scenarios,
    )

    assert batch.winner is batch.results[0]


def test_result_by_scenario_id() -> None:
    system, tensor, trajectory = build_fixture()
    scenarios = generate_single_channel_scenarios(
        system.channel_ids
    )
    batch = evaluate_counterfactual_batch(
        system,
        tensor,
        trajectory,
        scenarios,
    )

    scenario_id = scenarios[0].scenario_id
    result = result_by_scenario_id(
        batch,
        scenario_id,
    )

    assert result.scenario.scenario_id == scenario_id


def test_result_by_scenario_id_rejects_missing_id() -> None:
    batch = CounterfactualBatchResult(
        results=(
            dummy_result(
                "a",
                reduction=0.5,
                spectral_gain=0.2,
                cost=0.1,
                risk=0.1,
                utility=1.0,
            ),
        ),
        objective=CounterfactualObjective.BALANCED_UTILITY,
        pareto_front_ids=("a",),
    )

    with pytest.raises(KeyError):
        result_by_scenario_id(batch, "missing")


def test_batch_summary_is_read_only() -> None:
    system, tensor, trajectory = build_fixture()
    scenarios = generate_single_channel_scenarios(
        system.channel_ids
    )
    batch = evaluate_counterfactual_batch(
        system,
        tensor,
        trajectory,
        scenarios,
    )

    summary = batch_summary(batch)

    assert isinstance(summary, MappingProxyType)
    assert summary["scenario_count"] == 3
    assert summary["winner"] == batch.winner.scenario.scenario_id


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


def test_generate_single_channel_scenarios() -> None:
    scenarios = generate_single_channel_scenarios(
        ("a", "b"),
        action=CounterfactualAction.RESTORE_CHANNEL,
        magnitude=0.5,
        cost_by_channel={"a": 0.2},
    )

    assert len(scenarios) == 2
    assert scenarios[0].interventions[0].channel_id == "a"
    assert scenarios[0].interventions[0].cost == pytest.approx(0.2)
    assert scenarios[1].interventions[0].cost is None


def test_generated_scenario_ids_are_unique() -> None:
    scenarios = generate_single_channel_scenarios(
        ("a", "b", "c")
    )

    ids = tuple(item.scenario_id for item in scenarios)

    assert len(ids) == len(set(ids))
