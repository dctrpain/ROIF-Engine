"""
Validation tests for Clinical Scenario 01:
Forefoot restriction -> tibial rotation -> knee dysfunction.

The scenario contains external clinical validation labels, but those labels are
never passed into the ROIF solver. The tests compare the engine output with the
labels only after inference.

Important:
    A failed role-matching or intervention-ranking assertion is a validation
    result, not automatically a software defect. The diagnostic message shows
    what the engine actually inferred.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from roif.roif_cascade import run_cascade
from roif.roif_solver import (
    ROIFSolution,
    SolverStatus,
    VetoReason,
    solution_is_actionable,
    solution_summary,
    solve_roif,
)
from roif.roif_tensor import build_capacity_tensor
from validation.clinical.foot_knee_case import (
    FootKneeValidationCase,
    build_foot_knee_case,
)


@pytest.fixture(scope="module")
def case() -> FootKneeValidationCase:
    return build_foot_knee_case()


@pytest.fixture(scope="module")
def solution(
    case: FootKneeValidationCase,
) -> ROIFSolution:
    return solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )


def result_by_id(
    solution: ROIFSolution,
    scenario_id: str,
):
    for result in solution.counterfactual_batch.results:
        if result.scenario.scenario_id == scenario_id:
            return result
    raise KeyError(scenario_id)


def inferred_roles(solution: ROIFSolution) -> dict[str, str]:
    return {
        "d_origin": solution.root_result.d_origin,
        "d_fast": solution.root_result.d_fast,
        "d_root": solution.root_result.d_root,
        "node_star": solution.root_result.node_star,
    }


def expected_roles(case: FootKneeValidationCase) -> dict[str, str]:
    return dict(case.expected.as_mapping())


def metrics_snapshot(solution: ROIFSolution) -> dict[str, dict[str, float]]:
    return {
        result.scenario.scenario_id: {
            "relative_reduction": result.metrics.relative_reduction,
            "cascade_reduction": result.metrics.cascade_reduction,
            "utility": result.metrics.utility,
            "spectral_gain": result.metrics.spectral_gain,
            "collateral_effect": result.metrics.collateral_effect,
            "safety_risk": result.metrics.safety_risk,
            "uncertainty": result.metrics.uncertainty,
        }
        for result in solution.counterfactual_batch.results
    }


def test_case_contract_is_complete(
    case: FootKneeValidationCase,
) -> None:
    assert case.case_id == "clinical_01_foot_knee"
    assert case.title
    assert case.pathological_system.channel_ids == (
        "forefoot_fibrosis",
        "forefoot_compliance",
        "tibial_rotation_control",
        "knee_tracking",
        "knee_flexion_tolerance",
    )
    assert case.reference_system.channel_ids == (
        case.pathological_system.channel_ids
    )
    assert set(case.initial_state) == set(case.channel_ids)
    assert len(case.scenarios) == 4
    assert case.metadata["hidden_labels_used_by_solver"] is False


def test_validation_labels_are_external_and_read_only(
    case: FootKneeValidationCase,
) -> None:
    labels = case.expected.as_mapping()

    assert isinstance(labels, MappingProxyType)
    assert labels == {
        "d_origin": "forefoot_fibrosis",
        "d_fast": "forefoot_compliance",
        "d_root": "tibial_rotation_control",
        "node_star": "forefoot_compliance",
    }

    with pytest.raises(TypeError):
        labels["d_root"] = "knee_tracking"  # type: ignore[index]


def test_pathological_state_has_lower_reserve_than_reference(
    case: FootKneeValidationCase,
) -> None:
    for channel_id in case.channel_ids:
        pathological = case.pathological_system.channel(channel_id)
        reference = case.reference_system.channel(channel_id)

        assert (
            pathological.capacity_state.reserve
            <= reference.capacity_state.reserve
        ), channel_id


def test_pathological_state_has_reduced_distal_mobility(
    case: FootKneeValidationCase,
) -> None:
    pathological = case.pathological_system
    reference = case.reference_system

    for channel_id in (
        "forefoot_compliance",
        "tibial_rotation_control",
        "knee_tracking",
        "knee_flexion_tolerance",
    ):
        assert (
            pathological.channel(channel_id).geometry.mobility
            < reference.channel(channel_id).geometry.mobility
        ), channel_id


def test_case_generates_recursive_cascade(
    case: FootKneeValidationCase,
) -> None:
    trajectory = run_cascade(
        case.pathological_system,
        initial_state=case.initial_state,
        tensor_config=case.tensor_config,
        cascade_config=case.cascade_config,
    )

    source_series = trajectory.series("forefoot_fibrosis")
    tibial_series = trajectory.series("tibial_rotation_control")
    knee_series = trajectory.series("knee_tracking")
    flexion_series = trajectory.series("knee_flexion_tolerance")

    assert source_series[0] > 0.0
    assert max(tibial_series) > 0.0
    assert max(knee_series) > 0.0
    assert max(flexion_series) > 0.0


def test_reference_tensor_has_greater_functional_transmission(
    case: FootKneeValidationCase,
) -> None:
    pathological = build_capacity_tensor(
        case.pathological_system,
        config=case.tensor_config,
    )
    reference = build_capacity_tensor(
        case.reference_system,
        config=case.tensor_config,
    )

    source = pathological.index("forefoot_compliance")
    target = pathological.index("tibial_rotation_control")

    assert (
        reference.matrix[target, source]
        > pathological.matrix[target, source]
    )


def test_complete_validation_pipeline_runs(
    case: FootKneeValidationCase,
    solution: ROIFSolution,
) -> None:
    assert isinstance(solution, ROIFSolution)
    assert solution.status is SolverStatus.OBSERVATION_ONLY
    assert solution.decision.safe is True
    assert solution.decision.executable is False
    assert solution.decision.veto_reason is VetoReason.NONE
    assert solution_is_actionable(solution) is False

    assert solution.tensor.channel_ids == case.channel_ids
    assert solution.trajectory.channel_ids == case.channel_ids
    assert len(solution.counterfactual_batch.results) == len(
        case.scenarios
    )


def test_all_provided_scenarios_are_evaluated_once(
    case: FootKneeValidationCase,
    solution: ROIFSolution,
) -> None:
    provided_ids = tuple(
        scenario.scenario_id
        for scenario in case.scenarios
    )
    evaluated_ids = tuple(
        result.scenario.scenario_id
        for result in solution.counterfactual_batch.results
    )

    assert set(evaluated_ids) == set(provided_ids)
    assert len(evaluated_ids) == len(set(evaluated_ids))


def test_forefoot_intervention_outperforms_local_knee_interventions(
    solution: ROIFSolution,
) -> None:
    """
    Central counterfactual validation claim.

    Restoring the upstream forefoot compliance channel should reduce the
    complete cascade more effectively than locally unloading either knee
    channel.
    """

    forefoot = result_by_id(
        solution,
        "release_forefoot_fibrosis",
    )
    knee_tracking = result_by_id(
        solution,
        "reduce_knee_tracking_load",
    )
    knee_flexion = result_by_id(
        solution,
        "reduce_knee_flexion_load",
    )

    diagnostic = metrics_snapshot(solution)

    assert (
        forefoot.metrics.relative_reduction
        > knee_tracking.metrics.relative_reduction
    ), diagnostic

    assert (
        forefoot.metrics.relative_reduction
        > knee_flexion.metrics.relative_reduction
    ), diagnostic


def test_upstream_intervention_has_better_utility_than_knee_only(
    solution: ROIFSolution,
) -> None:
    forefoot = result_by_id(
        solution,
        "release_forefoot_fibrosis",
    )
    knee_tracking = result_by_id(
        solution,
        "reduce_knee_tracking_load",
    )
    knee_flexion = result_by_id(
        solution,
        "reduce_knee_flexion_load",
    )

    diagnostic = metrics_snapshot(solution)

    assert (
        forefoot.metrics.utility
        > knee_tracking.metrics.utility
    ), diagnostic

    assert (
        forefoot.metrics.utility
        > knee_flexion.metrics.utility
    ), diagnostic


def test_solver_selects_forefoot_restoration(
    solution: ROIFSolution,
) -> None:
    diagnostic = metrics_snapshot(solution)

    assert (
        solution.decision.scenario_id
        == "release_forefoot_fibrosis"
    ), {
        "selected": solution.decision.scenario_id,
        "metrics": diagnostic,
    }


def test_inferred_roles_match_clinical_validation_labels(
    case: FootKneeValidationCase,
    solution: ROIFSolution,
) -> None:
    """
    Compare engine-derived roles with external clinical labels.

    No expected role was supplied to solve_roif().
    """

    actual = inferred_roles(solution)
    expected = expected_roles(case)

    assert actual == expected, {
        "expected": expected,
        "inferred": actual,
        "scenario_metrics": metrics_snapshot(solution),
    }


def test_knee_is_not_inferred_as_origin_or_root(
    solution: ROIFSolution,
) -> None:
    assert solution.root_result.d_origin not in {
        "knee_tracking",
        "knee_flexion_tolerance",
    }
    assert solution.root_result.d_root not in {
        "knee_tracking",
        "knee_flexion_tolerance",
    }


def test_solution_summary_is_consistent(
    solution: ROIFSolution,
) -> None:
    summary = solution_summary(solution)

    assert isinstance(summary, MappingProxyType)
    assert summary["status"] == solution.status.value
    assert summary["scenario_id"] == solution.decision.scenario_id
    assert summary["d_origin"] == solution.root_result.d_origin
    assert summary["d_fast"] == solution.root_result.d_fast
    assert summary["d_root"] == solution.root_result.d_root
    assert summary["node_star"] == solution.root_result.node_star


def test_validation_pipeline_is_deterministic(
    case: FootKneeValidationCase,
    solution: ROIFSolution,
) -> None:
    repeated = solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=case.scenarios,
        config=case.solver_config,
    )

    assert repeated.status is solution.status
    assert repeated.decision == solution.decision
    assert repeated.root_result == solution.root_result
    assert metrics_snapshot(repeated) == metrics_snapshot(solution)
