"""
Tests for roif.reviewer_roif_method.

These tests define the reviewer-facing ROIF contract:

- only public fixture fields are used;
- hidden true_root, true_node_star, and intervention_gain are not needed;
- identical input produces identical D_fast, D_root, Node*, and rankings;
- every candidate origin is evaluated by forward reconstruction;
- every intervention node is evaluated independently;
- complete rankings are returned;
- the method works across all declared graph families;
- noise and partial observation do not crash the algorithm;
- the adapter is compatible with BenchmarkRunner data structures.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any
import math

import networkx as nx
import numpy as np
import pytest

from roif.benchmark import (
    BenchmarkMethod,
    BenchmarkMethodKind,
)
from roif.reviewer_graph_factory import (
    ReviewerFixture,
    build_reviewer_fixture,
    clean_reviewer_config,
    high_noise_reviewer_config,
    moderate_noise_reviewer_config,
    reviewer_graph_specs,
)
from roif.reviewer_roif_method import (
    DEFAULT_REVIEWER_ROIF_CONFIG,
    CandidateScore,
    ForwardState,
    InterventionScore,
    LocalizationLoss,
    PublicFixtureView,
    ReviewerROIFConfig,
    ReviewerROIFError,
    ReviewerROIFResult,
    evaluate_reviewer_roif,
    execute_reviewer_roif,
    fit_candidate_root,
    localization_error,
    make_reviewer_roif_method,
    public_fixture_view,
    rank_candidate_roots,
    rank_interventions,
    score_intervention,
    simulate_forward,
)
from roif.validation import (
    GraphFamily,
    GraphSpec,
    ValidationOutcome,
    ValidationTrial,
)


ALL_FAMILIES = (
    GraphFamily.LINEAR_CHAIN,
    GraphFamily.TREE,
    GraphFamily.LATTICE,
    GraphFamily.ERDOS_RENYI,
    GraphFamily.WATTS_STROGATZ,
    GraphFamily.BARABASI_ALBERT,
)


def make_spec(
    family: GraphFamily = GraphFamily.LINEAR_CHAIN,
    *,
    node_count: int = 8,
) -> GraphSpec:
    parameters: dict[str, Any]

    if family is GraphFamily.TREE:
        parameters = {"branching": 2}
    elif family is GraphFamily.ERDOS_RENYI:
        parameters = {"p": 0.35}
    elif family is GraphFamily.WATTS_STROGATZ:
        parameters = {"k": 4, "p": 0.20}
    elif family is GraphFamily.BARABASI_ALBERT:
        parameters = {"m": 2}
    else:
        parameters = {}

    return GraphSpec(
        family=family,
        node_count=node_count,
        edge_count=None,
        parameters=parameters,
        name=f"{family.value}-{node_count}",
    )


def make_fixture(
    family: GraphFamily = GraphFamily.LINEAR_CHAIN,
    *,
    node_count: int = 8,
    seed: int = 42,
    noisy: str = "clean",
) -> ReviewerFixture:
    if noisy == "clean":
        graph_config = clean_reviewer_config()
    elif noisy == "moderate":
        graph_config = moderate_noise_reviewer_config()
    elif noisy == "high":
        graph_config = high_noise_reviewer_config()
    else:
        raise ValueError(noisy)

    return build_reviewer_fixture(
        make_spec(family, node_count=node_count),
        seed,
        graph_config,
    )


def make_trial(
    fixture: ReviewerFixture,
) -> ValidationTrial:
    return ValidationTrial(
        trial_id=(
            f"{fixture.graph_family.value}-"
            f"{len(fixture.node_order)}-"
            f"{fixture.trial_seed}"
        ),
        repetition=0,
        seed=fixture.trial_seed,
        graph=GraphSpec(
            family=fixture.graph_family,
            node_count=len(fixture.node_order),
            edge_count=fixture.graph.number_of_edges(),
            parameters={},
            name=fixture.graph_name,
        ),
        true_root=fixture.true_root,
        true_node_star=fixture.true_node_star,
    )


def result_signature(
    result: ReviewerROIFResult,
) -> tuple[Any, ...]:
    return (
        result.d_fast,
        result.d_root,
        result.node_star,
        result.root_ranking,
        result.node_star_ranking,
        tuple(
            (
                item.node,
                round(item.error, 14),
                round(item.fitted_disturbance, 14),
            )
            for item in result.candidate_scores
        ),
        tuple(
            (
                item.node,
                round(item.gain, 14),
                round(item.post_intervention_deficit, 14),
            )
            for item in result.intervention_scores
        ),
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_immutable() -> None:
    config = ReviewerROIFConfig()

    assert config.use_state_dependent_tensor is True
    assert (
        config.localization_loss
        is LocalizationLoss.WEIGHTED_RMSE
    )
    assert isinstance(config.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        config.dissipation = 0.5  # type: ignore[misc]


def test_exported_default_config_is_reviewer_config() -> None:
    assert isinstance(
        DEFAULT_REVIEWER_ROIF_CONFIG,
        ReviewerROIFConfig,
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("propagation_steps", 0),
        ("intervention_steps", 0),
        ("disturbance_grid_size", 0),
        ("disturbance_refinement_rounds", -1),
        ("convergence_tolerance", 0.0),
        ("reserve_floor", 0.0),
        ("huber_delta", 0.0),
        ("dissipation", -0.1),
        ("dissipation", 1.0),
        ("intervention_fraction", 0.0),
        ("intervention_fraction", 1.1),
        ("missing_value_weight", -0.1),
        ("missing_value_weight", 1.1),
        ("disturbance_min", -0.1),
        ("tensor_floor", 0.0),
        ("source_reserve_exponent", -0.1),
        ("target_reserve_exponent", -0.1),
        ("overload_amplification", -0.1),
        ("near_failure_weight", -0.1),
    ],
)
def test_config_rejects_invalid_values(
    field_name: str,
    value: Any,
) -> None:
    with pytest.raises(ReviewerROIFError):
        ReviewerROIFConfig(
            **{field_name: value},
        )


def test_config_rejects_invalid_disturbance_range() -> None:
    with pytest.raises(ReviewerROIFError):
        ReviewerROIFConfig(
            disturbance_min=0.4,
            disturbance_max=0.2,
        )


def test_config_rejects_tensor_floor_above_ceiling() -> None:
    with pytest.raises(ReviewerROIFError):
        ReviewerROIFConfig(
            tensor_floor=1.0,
            tensor_ceiling=0.5,
        )


def test_config_rejects_invalid_loss_type() -> None:
    with pytest.raises(ReviewerROIFError):
        ReviewerROIFConfig(
            localization_loss="rmse",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Public fixture boundary
# ---------------------------------------------------------------------------


def test_public_fixture_view_contains_only_public_fields() -> None:
    fixture = make_fixture()
    view = public_fixture_view(fixture)

    assert isinstance(view, PublicFixtureView)
    assert not hasattr(view, "true_root")
    assert not hasattr(view, "true_node_star")
    assert not hasattr(view, "intervention_gain")


def test_public_fixture_view_copies_graph() -> None:
    fixture = make_fixture()
    view = public_fixture_view(fixture)

    assert view.graph is not fixture.graph
    assert set(view.graph.nodes) == set(fixture.graph.nodes)
    assert set(view.graph.edges) == set(fixture.graph.edges)


def test_public_fixture_view_mappings_are_read_only() -> None:
    view = public_fixture_view(make_fixture())

    assert isinstance(view.capacity, MappingProxyType)
    assert isinstance(view.initial_load, MappingProxyType)
    assert isinstance(
        view.observed_reserve_fraction,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        view.capacity[view.nodes[0]] = 99.0


def test_public_fixture_view_works_without_hidden_truth_fields() -> None:
    fixture = make_fixture()

    public_only = {
        "graph": fixture.graph,
        "node_order": fixture.node_order,
        "capacity": fixture.capacity,
        "initial_load": fixture.initial_load,
        "load": fixture.load,
        "observed_state": fixture.observed_state,
        "observed_mask": fixture.observed_mask,
        "symptom_node": fixture.symptom_node,
        "metadata": fixture.metadata,
    }

    view = public_fixture_view(public_only)

    assert view.nodes == fixture.node_order
    assert dict(view.capacity) == dict(fixture.capacity)


def test_evaluation_works_without_hidden_truth_fields() -> None:
    fixture = make_fixture()

    public_only = {
        "graph": fixture.graph,
        "node_order": fixture.node_order,
        "capacity": fixture.capacity,
        "initial_load": fixture.initial_load,
        "load": fixture.load,
        "observed_state": fixture.observed_state,
        "observed_mask": fixture.observed_mask,
        "symptom_node": fixture.symptom_node,
        "metadata": fixture.metadata,
    }

    result = evaluate_reviewer_roif(public_only)

    assert result.d_root in fixture.graph
    assert result.node_star in fixture.graph


def test_hidden_truth_changes_do_not_change_roif_result() -> None:
    fixture = make_fixture()

    left = evaluate_reviewer_roif(fixture)

    public_with_false_labels = {
        "graph": fixture.graph,
        "node_order": fixture.node_order,
        "capacity": fixture.capacity,
        "initial_load": fixture.initial_load,
        "load": fixture.load,
        "observed_state": fixture.observed_state,
        "observed_mask": fixture.observed_mask,
        "symptom_node": fixture.symptom_node,
        "true_root": -999,
        "true_node_star": -998,
        "intervention_gain": {
            node: 999.0 - node
            for node in fixture.node_order
        },
        "metadata": fixture.metadata,
    }

    right = evaluate_reviewer_roif(
        public_with_false_labels
    )

    assert result_signature(left) == result_signature(right)


def test_public_fixture_view_rejects_missing_graph() -> None:
    with pytest.raises(ReviewerROIFError):
        public_fixture_view({})


def test_public_fixture_view_rejects_missing_initial_load() -> None:
    fixture = make_fixture()

    with pytest.raises(ReviewerROIFError):
        public_fixture_view(
            {
                "graph": fixture.graph,
                "node_order": fixture.node_order,
                "capacity": fixture.capacity,
                "observed_state": fixture.observed_state,
            }
        )


# ---------------------------------------------------------------------------
# Forward solver
# ---------------------------------------------------------------------------


def test_forward_simulation_is_deterministic() -> None:
    view = public_fixture_view(make_fixture())
    config = ReviewerROIFConfig()

    left = simulate_forward(
        view,
        initial_load=view.initial_load,
        disturbance_node=view.nodes[2],
        disturbance=0.20,
        config=config,
    )
    right = simulate_forward(
        view,
        initial_load=view.initial_load,
        disturbance_node=view.nodes[2],
        disturbance=0.20,
        config=config,
    )

    assert dict(left.load) == pytest.approx(dict(right.load))
    assert dict(left.reserve_fraction) == pytest.approx(
        dict(right.reserve_fraction)
    )
    assert left.global_deficit == pytest.approx(
        right.global_deficit
    )


def test_zero_disturbance_preserves_initial_load() -> None:
    view = public_fixture_view(make_fixture())

    state = simulate_forward(
        view,
        initial_load=view.initial_load,
        disturbance_node=None,
        disturbance=0.0,
        config=ReviewerROIFConfig(),
    )

    assert dict(state.load) == pytest.approx(
        dict(view.initial_load)
    )


def test_positive_disturbance_changes_state() -> None:
    view = public_fixture_view(make_fixture())

    state = simulate_forward(
        view,
        initial_load=view.initial_load,
        disturbance_node=view.nodes[1],
        disturbance=0.25,
        config=ReviewerROIFConfig(),
    )

    assert any(
        state.load[node] > view.initial_load[node]
        for node in view.nodes
    )


def test_forward_simulation_rejects_unknown_node() -> None:
    view = public_fixture_view(make_fixture())

    with pytest.raises(ReviewerROIFError):
        simulate_forward(
            view,
            initial_load=view.initial_load,
            disturbance_node="unknown",
            disturbance=0.10,
            config=ReviewerROIFConfig(),
        )


def test_forward_simulation_rejects_negative_disturbance() -> None:
    view = public_fixture_view(make_fixture())

    with pytest.raises(ReviewerROIFError):
        simulate_forward(
            view,
            initial_load=view.initial_load,
            disturbance_node=view.nodes[0],
            disturbance=-0.10,
            config=ReviewerROIFConfig(),
        )


def test_state_dependent_and_static_tensor_can_differ() -> None:
    view = public_fixture_view(make_fixture())

    dynamic = simulate_forward(
        view,
        initial_load=view.initial_load,
        disturbance_node=view.nodes[3],
        disturbance=0.30,
        config=ReviewerROIFConfig(
            use_state_dependent_tensor=True,
        ),
    )
    static = simulate_forward(
        view,
        initial_load=view.initial_load,
        disturbance_node=view.nodes[3],
        disturbance=0.30,
        config=ReviewerROIFConfig(
            use_state_dependent_tensor=False,
        ),
    )

    assert any(
        not math.isclose(
            dynamic.reserve_fraction[node],
            static.reserve_fraction[node],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for node in view.nodes
    )


# ---------------------------------------------------------------------------
# Localization functional and candidate fitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss",
    list(LocalizationLoss),
)
def test_localization_error_is_zero_for_identical_state(
    loss: LocalizationLoss,
) -> None:
    view = public_fixture_view(make_fixture())
    config = ReviewerROIFConfig(
        localization_loss=loss,
    )

    error = localization_error(
        view.observed_reserve_fraction,
        view,
        config,
    )

    assert error == pytest.approx(0.0)


def test_localization_error_increases_for_perturbed_state() -> None:
    view = public_fixture_view(make_fixture())
    config = ReviewerROIFConfig()

    perturbed = {
        node: value + 0.10
        for node, value in (
            view.observed_reserve_fraction.items()
        )
    }

    assert localization_error(
        perturbed,
        view,
        config,
    ) > 0.0


def test_fit_candidate_root_returns_candidate_score() -> None:
    view = public_fixture_view(make_fixture())
    candidate = view.nodes[0]

    score = fit_candidate_root(
        view,
        candidate,
        ReviewerROIFConfig(
            disturbance_grid_size=9,
            disturbance_refinement_rounds=1,
        ),
    )

    assert isinstance(score, CandidateScore)
    assert score.node == candidate
    assert score.error >= 0.0
    assert set(score.predicted_reserve_fraction) == set(
        view.nodes
    )


def test_candidate_ranking_is_complete_and_sorted() -> None:
    view = public_fixture_view(make_fixture())
    config = ReviewerROIFConfig(
        disturbance_grid_size=9,
        disturbance_refinement_rounds=1,
    )

    scores = rank_candidate_roots(view, config)

    assert len(scores) == len(view.nodes)
    assert {score.node for score in scores} == set(view.nodes)
    assert [score.error for score in scores] == sorted(
        score.error for score in scores
    )


# ---------------------------------------------------------------------------
# Intervention optimization
# ---------------------------------------------------------------------------


def test_score_intervention_returns_valid_score() -> None:
    view = public_fixture_view(make_fixture())

    score = score_intervention(
        view,
        view.nodes[0],
        ReviewerROIFConfig(),
    )

    assert isinstance(score, InterventionScore)
    assert score.node == view.nodes[0]
    assert np.isfinite(score.gain)
    assert np.isfinite(score.post_intervention_deficit)


def test_intervention_ranking_is_complete() -> None:
    view = public_fixture_view(make_fixture())

    scores = rank_interventions(
        view,
        ReviewerROIFConfig(),
    )

    assert len(scores) == len(view.nodes)
    assert {score.node for score in scores} == set(view.nodes)


def test_intervention_ranking_is_sorted_by_gain() -> None:
    view = public_fixture_view(make_fixture())

    scores = rank_interventions(
        view,
        ReviewerROIFConfig(),
    )

    gains = [score.gain for score in scores]

    assert gains == sorted(gains, reverse=True)


# ---------------------------------------------------------------------------
# Complete ROIF evaluation
# ---------------------------------------------------------------------------


def test_complete_evaluation_returns_three_roles() -> None:
    result = evaluate_reviewer_roif(make_fixture())

    assert result.d_fast is not None
    assert result.d_root is not None
    assert result.node_star is not None


def test_complete_rankings_cover_every_node() -> None:
    fixture = make_fixture()
    result = evaluate_reviewer_roif(fixture)

    assert len(result.root_ranking) == len(fixture.node_order)
    assert set(result.root_ranking) == set(fixture.node_order)

    assert len(result.node_star_ranking) == len(
        fixture.node_order
    )
    assert set(result.node_star_ranking) == set(
        fixture.node_order
    )


def test_root_and_node_star_match_first_ranking_entries() -> None:
    result = evaluate_reviewer_roif(make_fixture())

    assert result.d_root == result.root_ranking[0]
    assert result.node_star == result.node_star_ranking[0]


def test_d_fast_is_minimum_observed_reserve_node() -> None:
    fixture = make_fixture()
    result = evaluate_reviewer_roif(fixture)

    observed_nodes = [
        node
        for node in fixture.node_order
        if fixture.observed_mask[node]
    ]
    expected = min(
        observed_nodes,
        key=lambda node: (
            fixture.observed_state[node],
            (type(node).__name__, repr(node)),
        ),
    )

    assert result.d_fast == expected


def test_evaluation_is_deterministic() -> None:
    fixture = make_fixture(
        GraphFamily.WATTS_STROGATZ,
        node_count=8,
        seed=123,
    )

    left = evaluate_reviewer_roif(fixture)
    right = evaluate_reviewer_roif(fixture)

    assert result_signature(left) == result_signature(right)


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_roif_runs_on_all_graph_families(
    family: GraphFamily,
) -> None:
    fixture = make_fixture(
        family,
        node_count=8,
        seed=42,
    )

    result = evaluate_reviewer_roif(
        fixture,
        ReviewerROIFConfig(
            disturbance_grid_size=9,
            disturbance_refinement_rounds=1,
        ),
    )

    assert result.d_root in fixture.graph
    assert result.node_star in fixture.graph
    assert len(result.root_ranking) == 8


@pytest.mark.parametrize(
    "noise_level",
    ["moderate", "high"],
)
def test_roif_runs_under_noise_and_partial_observation(
    noise_level: str,
) -> None:
    fixture = make_fixture(
        GraphFamily.WATTS_STROGATZ,
        node_count=8,
        seed=42,
        noisy=noise_level,
    )

    result = evaluate_reviewer_roif(
        fixture,
        ReviewerROIFConfig(
            disturbance_grid_size=9,
            disturbance_refinement_rounds=1,
        ),
    )

    assert result.d_root in fixture.graph
    assert result.node_star in fixture.graph
    assert len(result.root_ranking) == 8


def test_roles_are_selected_from_independent_rankings() -> None:
    fixture = make_fixture()

    result = evaluate_reviewer_roif(
        fixture,
        ReviewerROIFConfig(
            disturbance_grid_size=9,
            disturbance_refinement_rounds=1,
        ),
    )

    observed_nodes = [
        node
        for node in fixture.node_order
        if fixture.observed_mask[node]
    ]

    expected_fast = min(
        observed_nodes,
        key=lambda node: (
            fixture.observed_state[node],
            (type(node).__name__, repr(node)),
        ),
    )

    assert result.d_fast == expected_fast
    assert result.d_root == result.root_ranking[0]
    assert result.node_star == result.node_star_ranking[0]

    assert len(result.candidate_scores) == len(
        fixture.node_order
    )
    assert len(result.intervention_scores) == len(
        fixture.node_order
    )

def test_make_method_returns_benchmark_method() -> None:
    method = make_reviewer_roif_method(
        ReviewerROIFConfig(
            disturbance_grid_size=9,
            disturbance_refinement_rounds=1,
        )
    )

    assert isinstance(method, BenchmarkMethod)
    assert method.method_id == "roif"
    assert method.kind is BenchmarkMethodKind.ROIF


def test_execute_returns_validation_outcome() -> None:
    fixture = make_fixture()
    trial = make_trial(fixture)

    outcome = execute_reviewer_roif(
        trial,
        fixture,
        ReviewerROIFConfig(
            disturbance_grid_size=9,
            disturbance_refinement_rounds=1,
        ),
    )

    assert isinstance(outcome, ValidationOutcome)
    assert outcome.predicted_root in fixture.graph
    assert outcome.predicted_node_star in fixture.graph
    assert set(outcome.root_ranking) == set(fixture.graph.nodes)
    assert set(outcome.node_star_ranking) == set(
        fixture.graph.nodes
    )


def test_execute_does_not_depend_on_trial_truth_labels() -> None:
    fixture = make_fixture()

    correct_trial = make_trial(fixture)
    false_trial = ValidationTrial(
        trial_id="false-labels",
        repetition=0,
        seed=fixture.trial_seed,
        graph=correct_trial.graph,
        true_root=-999,
        true_node_star=-998,
    )

    config = ReviewerROIFConfig(
        disturbance_grid_size=9,
        disturbance_refinement_rounds=1,
    )

    left = execute_reviewer_roif(
        correct_trial,
        fixture,
        config,
    )
    right = execute_reviewer_roif(
        false_trial,
        fixture,
        config,
    )

    assert left.predicted_root == right.predicted_root
    assert (
        left.predicted_node_star
        == right.predicted_node_star
    )
    assert left.root_ranking == right.root_ranking
    assert (
        left.node_star_ranking
        == right.node_star_ranking
    )


def test_method_executor_runs_on_fixture() -> None:
    fixture = make_fixture()
    trial = make_trial(fixture)
    method = make_reviewer_roif_method(
        ReviewerROIFConfig(
            disturbance_grid_size=9,
            disturbance_refinement_rounds=1,
        )
    )

    outcome = method.execute(trial, fixture)

    assert outcome.predicted_root in fixture.graph
    assert outcome.predicted_node_star in fixture.graph


def test_execute_rejects_invalid_trial_type() -> None:
    with pytest.raises(TypeError, match="ValidationTrial"):
        execute_reviewer_roif(
            "invalid",  # type: ignore[arg-type]
            make_fixture(),
        )

