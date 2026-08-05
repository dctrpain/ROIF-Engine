"""
Tests for roif.reviewer_graph_factory.

These tests define the reproducibility and fairness contract of the
reviewer-facing synthetic benchmark:

- identical graph specification and seed produce identical fixtures;
- different seeds alter the hidden trial;
- all six declared graph families are supported;
- public fixture fields are internally consistent;
- hidden labels are excluded from ordinary serialization;
- graph degradation is deterministic;
- the fixture is compatible with reviewer baseline methods;
- the generated benchmark contains non-trivial separation between
  cascade origin, symptom node, and optimal intervention node.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import networkx as nx
import pytest

from roif.baselines import (
    fixture_view,
    reviewer_baselines,
)
from roif.reviewer_graph_factory import (
    DEFAULT_REVIEWER_NODE_COUNTS,
    ObservationNoiseMode,
    ReviewerFixture,
    ReviewerGraphConfig,
    ReviewerGraphFactoryError,
    build_reviewer_fixture,
    build_reviewer_graph,
    clean_reviewer_config,
    high_noise_reviewer_config,
    moderate_noise_reviewer_config,
    reviewer_fixture_factory,
    reviewer_graph_distance,
    reviewer_graph_specs,
    reviewer_true_node_star,
    reviewer_true_root,
)
from roif.validation import (
    GraphFamily,
    GraphSpec,
    ValidationTrial,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spec(
    family: GraphFamily = GraphFamily.LINEAR_CHAIN,
    *,
    node_count: int = 12,
) -> GraphSpec:
    parameters: dict[str, Any]

    if family is GraphFamily.TREE:
        parameters = {"branching": 2}
    elif family is GraphFamily.ERDOS_RENYI:
        parameters = {"p": 0.30}
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


def make_trial(
    *,
    family: GraphFamily = GraphFamily.LINEAR_CHAIN,
    node_count: int = 12,
    seed: int = 42,
) -> ValidationTrial:
    return ValidationTrial(
        trial_id=f"{family.value}-{node_count}-{seed}",
        repetition=0,
        seed=seed,
        graph=make_spec(
            family,
            node_count=node_count,
        ),
        true_root=None,
        true_node_star=None,
    )


def edge_signature(
    graph: nx.DiGraph,
) -> tuple[tuple[int, int, float], ...]:
    return tuple(
        sorted(
            (
                int(source),
                int(target),
                round(
                    float(data["transmission"]),
                    12,
                ),
            )
            for source, target, data in graph.edges(data=True)
        )
    )


def fixture_signature(
    fixture: ReviewerFixture,
) -> tuple[Any, ...]:
    return (
        edge_signature(fixture.graph),
        tuple(fixture.capacity.items()),
        tuple(fixture.initial_load.items()),
        tuple(fixture.load.items()),
        tuple(fixture.reserve.items()),
        tuple(fixture.observed_state.items()),
        tuple(fixture.observed_mask.items()),
        tuple(fixture.intervention_gain.items()),
        fixture.true_root,
        fixture.symptom_node,
        fixture.true_node_star,
        fixture.disturbance,
        fixture.global_deficit,
    )


ALL_SYNTHETIC_FAMILIES = (
    GraphFamily.LINEAR_CHAIN,
    GraphFamily.TREE,
    GraphFamily.LATTICE,
    GraphFamily.ERDOS_RENYI,
    GraphFamily.WATTS_STROGATZ,
    GraphFamily.BARABASI_ALBERT,
)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_immutable() -> None:
    config = ReviewerGraphConfig()

    assert config.directed is True
    assert config.observation_noise_mode is ObservationNoiseMode.NONE
    assert isinstance(config.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        config.directed = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("capacity_min", 0.0),
        ("capacity_max", float("inf")),
        ("prestress_min", -0.1),
        ("prestress_max", 1.0),
        ("transmission_min", -0.1),
        ("transmission_max", 1.1),
        ("dissipation", -0.1),
        ("dissipation", 1.0),
        ("intervention_fraction", 0.0),
        ("intervention_fraction", 1.1),
        ("observation_noise_std", -0.1),
        ("edge_dropout_probability", -0.1),
        ("edge_dropout_probability", 1.1),
        ("capacity_noise_std", -0.1),
        ("transmission_noise_std", -0.1),
        ("missing_observation_probability", 1.1),
        ("propagation_steps", 0),
        ("intervention_steps", 0),
        ("minimum_root_degree", 0),
    ],
)
def test_config_rejects_invalid_values(
    field_name: str,
    value: Any,
) -> None:
    with pytest.raises(ReviewerGraphFactoryError):
        ReviewerGraphConfig(
            **{field_name: value},
        )


def test_config_rejects_reversed_ranges() -> None:
    with pytest.raises(ReviewerGraphFactoryError):
        ReviewerGraphConfig(
            capacity_min=1.2,
            capacity_max=1.0,
        )


def test_clean_moderate_and_high_noise_presets() -> None:
    clean = clean_reviewer_config()
    moderate = moderate_noise_reviewer_config()
    high = high_noise_reviewer_config()

    assert clean.observation_noise_std == 0.0
    assert clean.edge_dropout_probability == 0.0

    assert moderate.observation_noise_std > clean.observation_noise_std
    assert high.observation_noise_std > moderate.observation_noise_std

    assert (
        high.missing_observation_probability
        > moderate.missing_observation_probability
    )


# ---------------------------------------------------------------------------
# Graph specification set
# ---------------------------------------------------------------------------


def test_default_node_counts_are_stable() -> None:
    assert DEFAULT_REVIEWER_NODE_COUNTS == (16, 32, 64)


def test_reviewer_graph_specs_cover_every_family_and_size() -> None:
    specs = reviewer_graph_specs((8, 16))

    assert len(specs) == 12
    assert {
        (spec.family, spec.node_count)
        for spec in specs
    } == {
        (family, node_count)
        for family in ALL_SYNTHETIC_FAMILIES
        for node_count in (8, 16)
    }


def test_reviewer_graph_specs_have_unique_labels() -> None:
    specs = reviewer_graph_specs((8, 16))

    labels = [spec.label for spec in specs]

    assert len(labels) == len(set(labels))


@pytest.mark.parametrize(
    "node_counts",
    [
        (),
        (0,),
        (-1,),
        (8, 8),
    ],
)
def test_reviewer_graph_specs_reject_invalid_node_counts(
    node_counts: tuple[int, ...],
) -> None:
    with pytest.raises(ReviewerGraphFactoryError):
        reviewer_graph_specs(node_counts)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ALL_SYNTHETIC_FAMILIES)
def test_build_graph_supports_all_synthetic_families(
    family: GraphFamily,
) -> None:
    spec = make_spec(family, node_count=12)
    graph = build_reviewer_graph(spec, 42)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 12
    assert set(graph.nodes) == set(range(12))
    assert nx.is_connected(graph.to_undirected())
    assert graph.number_of_edges() > 0


@pytest.mark.parametrize("family", ALL_SYNTHETIC_FAMILIES)
def test_every_edge_has_valid_transmission_metadata(
    family: GraphFamily,
) -> None:
    graph = build_reviewer_graph(
        make_spec(family, node_count=12),
        42,
    )

    for _, _, data in graph.edges(data=True):
        transmission = data["transmission"]

        assert 0.0 <= transmission <= 0.95
        assert data["weight"] == transmission
        assert data["distance"] > 0.0


def test_graph_generation_is_deterministic() -> None:
    spec = make_spec(
        GraphFamily.WATTS_STROGATZ,
        node_count=16,
    )

    left = build_reviewer_graph(spec, 123)
    right = build_reviewer_graph(spec, 123)

    assert edge_signature(left) == edge_signature(right)


def test_different_graph_seeds_change_weighted_graph() -> None:
    spec = make_spec(
        GraphFamily.WATTS_STROGATZ,
        node_count=16,
    )

    left = build_reviewer_graph(spec, 123)
    right = build_reviewer_graph(spec, 124)

    assert edge_signature(left) != edge_signature(right)


def test_custom_graph_family_is_rejected() -> None:
    spec = GraphSpec(
        family=GraphFamily.CUSTOM,
        node_count=5,
    )

    with pytest.raises(
        ReviewerGraphFactoryError,
        match="CUSTOM",
    ):
        build_reviewer_graph(spec, 42)


def test_build_graph_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError, match="GraphSpec"):
        build_reviewer_graph("invalid", 42)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="seed"):
        build_reviewer_graph(
            make_spec(),
            True,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Fixture generation and physical consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ALL_SYNTHETIC_FAMILIES)
def test_fixture_supports_every_graph_family(
    family: GraphFamily,
) -> None:
    fixture = build_reviewer_fixture(
        make_spec(family, node_count=10),
        42,
    )

    assert fixture.graph_family is family
    assert len(fixture.node_order) == 10
    assert fixture.true_root in fixture.graph
    assert fixture.symptom_node in fixture.graph
    assert fixture.true_node_star in fixture.graph


def test_fixture_is_deterministic_for_same_seed() -> None:
    spec = make_spec(
        GraphFamily.ERDOS_RENYI,
        node_count=14,
    )

    left = build_reviewer_fixture(spec, 771)
    right = build_reviewer_fixture(spec, 771)

    assert fixture_signature(left) == fixture_signature(right)


def test_different_seeds_change_fixture() -> None:
    spec = make_spec(
        GraphFamily.ERDOS_RENYI,
        node_count=14,
    )

    left = build_reviewer_fixture(spec, 771)
    right = build_reviewer_fixture(spec, 772)

    assert fixture_signature(left) != fixture_signature(right)


def test_capacity_load_and_reserve_are_consistent() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=12),
        42,
    )

    for node in fixture.node_order:
        assert fixture.initial_reserve[node] == pytest.approx(
            fixture.capacity[node] - fixture.initial_load[node]
        )
        assert fixture.reserve[node] == pytest.approx(
            fixture.capacity[node] - fixture.load[node]
        )
        assert fixture.reserve_fraction[node] == pytest.approx(
            fixture.reserve[node] / fixture.capacity[node]
        )


def test_symptom_node_is_minimum_clean_reserve_fraction() -> None:
    fixture = build_reviewer_fixture(
        make_spec(
            GraphFamily.TREE,
            node_count=15,
        ),
        42,
    )

    expected = min(
        fixture.reserve_fraction,
        key=lambda node: (
            fixture.reserve_fraction[node],
            node,
        ),
    )

    assert fixture.symptom_node == expected


def test_true_node_star_matches_maximum_intervention_gain() -> None:
    fixture = build_reviewer_fixture(
        make_spec(
            GraphFamily.LATTICE,
            node_count=16,
        ),
        42,
    )

    expected = min(
        fixture.intervention_gain,
        key=lambda node: (
            -fixture.intervention_gain[node],
            node,
        ),
    )

    assert fixture.true_node_star == expected


def test_fixture_mappings_are_read_only() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=8),
        42,
    )

    assert isinstance(fixture.capacity, MappingProxyType)
    assert isinstance(fixture.reserve, MappingProxyType)
    assert isinstance(fixture.observed_state, MappingProxyType)
    assert isinstance(fixture.intervention_gain, MappingProxyType)

    with pytest.raises(TypeError):
        fixture.capacity[0] = 99.0


def test_fixture_aliases_support_baseline_adapter() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=8),
        42,
    )

    assert fixture.nodes == fixture.node_order
    assert fixture.capacities is fixture.capacity
    assert fixture.loads is fixture.load
    assert fixture.reserves is fixture.reserve
    assert fixture.intervention_gains is fixture.intervention_gain


def test_fixture_view_accepts_reviewer_fixture() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=8),
        42,
    )

    view = fixture_view(fixture)

    assert view.graph is fixture.graph
    assert view.nodes == fixture.node_order
    assert dict(view.capacity) == dict(fixture.capacity)
    assert dict(view.reserve) == dict(fixture.reserve)
    assert view.symptom_node == fixture.symptom_node


# ---------------------------------------------------------------------------
# Hidden truth and serialization
# ---------------------------------------------------------------------------


def test_default_serialization_hides_true_labels() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=8),
        42,
    )

    payload = fixture.as_dict()

    assert "true_root" not in payload
    assert "true_node_star" not in payload


def test_explicit_serialization_can_include_hidden_truth() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=8),
        42,
    )

    payload = fixture.as_dict(include_hidden_truth=True)

    assert payload["true_root"] == fixture.true_root
    assert payload["true_node_star"] == fixture.true_node_star


def test_public_metadata_does_not_expose_hidden_node_ids() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=8),
        42,
    )

    assert "true_root" not in fixture.metadata
    assert "true_node_star" not in fixture.metadata


# ---------------------------------------------------------------------------
# Noise and partial observability
# ---------------------------------------------------------------------------


def test_noise_condition_is_reproducible() -> None:
    spec = make_spec(
        GraphFamily.WATTS_STROGATZ,
        node_count=16,
    )
    config = moderate_noise_reviewer_config()

    left = build_reviewer_fixture(spec, 1234, config)
    right = build_reviewer_fixture(spec, 1234, config)

    assert fixture_signature(left) == fixture_signature(right)


def test_clean_and_noisy_conditions_share_hidden_clean_labels() -> None:
    spec = make_spec(
        GraphFamily.WATTS_STROGATZ,
        node_count=16,
    )

    clean = build_reviewer_fixture(
        spec,
        1234,
        clean_reviewer_config(),
    )
    noisy = build_reviewer_fixture(
        spec,
        1234,
        moderate_noise_reviewer_config(),
    )

    assert noisy.true_root == clean.true_root
    assert noisy.true_node_star == clean.true_node_star
    assert noisy.symptom_node == clean.symptom_node


def test_noisy_condition_changes_public_observation() -> None:
    spec = make_spec(
        GraphFamily.WATTS_STROGATZ,
        node_count=16,
    )

    clean = build_reviewer_fixture(
        spec,
        1234,
        clean_reviewer_config(),
    )
    noisy = build_reviewer_fixture(
        spec,
        1234,
        high_noise_reviewer_config(),
    )

    assert (
        dict(noisy.observed_state)
        != dict(clean.observed_state)
        or edge_signature(noisy.graph)
        != edge_signature(clean.graph)
        or dict(noisy.capacity)
        != dict(clean.capacity)
    )


def test_at_least_one_observation_remains_available() -> None:
    config = ReviewerGraphConfig(
        missing_observation_probability=1.0,
    )
    fixture = build_reviewer_fixture(
        make_spec(node_count=8),
        42,
        config,
    )

    assert any(fixture.observed_mask.values())


# ---------------------------------------------------------------------------
# BenchmarkRunner-compatible factories
# ---------------------------------------------------------------------------


def test_fixture_factory_is_reproducible() -> None:
    trial = make_trial(seed=42)

    left = reviewer_fixture_factory(trial)
    right = reviewer_fixture_factory(trial)

    assert fixture_signature(left) == fixture_signature(right)


def test_truth_factories_match_fixture() -> None:
    spec = make_spec(
        GraphFamily.BARABASI_ALBERT,
        node_count=14,
    )
    seed = 991
    fixture = build_reviewer_fixture(spec, seed)

    assert reviewer_true_root(
        spec,
        0,
        seed,
    ) == fixture.true_root

    assert reviewer_true_node_star(
        spec,
        0,
        seed,
    ) == fixture.true_node_star


def test_fixture_factory_ignores_trial_truth_fields() -> None:
    spec = make_spec(node_count=10)
    trial = ValidationTrial(
        trial_id="truth-independence",
        repetition=0,
        seed=42,
        graph=spec,
        true_root=999,
        true_node_star=998,
    )

    fixture = reviewer_fixture_factory(trial)
    direct = build_reviewer_fixture(spec, 42)

    assert fixture_signature(fixture) == fixture_signature(direct)


def test_reviewer_graph_distance_is_zero_for_same_node() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=10),
        42,
    )

    assert reviewer_graph_distance(
        fixture.true_root,
        fixture.true_root,
        fixture,
    ) == 0.0


def test_reviewer_graph_distance_is_symmetric() -> None:
    fixture = build_reviewer_fixture(
        make_spec(
            GraphFamily.TREE,
            node_count=15,
        ),
        42,
    )

    left = fixture.node_order[0]
    right = fixture.node_order[-1]

    assert reviewer_graph_distance(
        left,
        right,
        fixture,
    ) == reviewer_graph_distance(
        right,
        left,
        fixture,
    )


def test_reviewer_graph_distance_penalizes_unknown_node() -> None:
    fixture = build_reviewer_fixture(
        make_spec(node_count=10),
        42,
    )

    assert reviewer_graph_distance(
        "unknown",
        fixture.true_root,
        fixture,
    ) == float(fixture.graph.number_of_nodes())


# ---------------------------------------------------------------------------
# Compatibility with baseline suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_id",
    [
        "random",
        "symptom_only",
        "minimum_reserve",
        "maximum_utilization",
        "degree",
        "betweenness",
        "upstream_symptom",
        "in_degree",
        "out_degree",
        "closeness",
        "eigenvector",
        "pagerank",
    ],
)
def test_fixture_runs_through_each_reviewer_baseline(
    method_id: str,
) -> None:
    trial = make_trial(
        family=GraphFamily.WATTS_STROGATZ,
        node_count=12,
        seed=42,
    )
    fixture = reviewer_fixture_factory(trial)

    methods = {
        method.method_id: method
        for method in reviewer_baselines()
    }
    outcome = methods[method_id].execute(
        trial,
        fixture,
    )

    assert outcome.converged is True
    assert outcome.predicted_root in fixture.graph
    assert len(outcome.root_ranking) == fixture.graph.number_of_nodes()
    assert set(outcome.root_ranking) == set(fixture.graph.nodes)


# ---------------------------------------------------------------------------
# Non-trivial role-separation contract
# ---------------------------------------------------------------------------


def test_seeded_suite_contains_root_node_star_separation() -> None:
    """
    The benchmark must contain trials in which causal origin and optimal
    intervention are not the same node.
    """

    separated = False
    spec = make_spec(
        GraphFamily.LINEAR_CHAIN,
        node_count=16,
    )

    for seed in range(16):
        fixture = build_reviewer_fixture(spec, seed)
        if fixture.true_root != fixture.true_node_star:
            separated = True
            break

    assert separated


def test_seeded_suite_contains_at_least_two_role_patterns() -> None:
    """
    A useful reviewer benchmark must not collapse every trial into one
    identical relation between root, symptom, and Node*.
    """

    patterns: set[tuple[bool, bool, bool]] = set()

    specs = (
        make_spec(GraphFamily.LINEAR_CHAIN, node_count=12),
        make_spec(GraphFamily.TREE, node_count=15),
        make_spec(GraphFamily.WATTS_STROGATZ, node_count=16),
    )

    for spec in specs:
        for seed in range(8):
            fixture = build_reviewer_fixture(spec, seed)
            patterns.add(
                (
                    fixture.true_root == fixture.symptom_node,
                    fixture.true_root == fixture.true_node_star,
                    fixture.symptom_node == fixture.true_node_star,
                )
            )

    assert len(patterns) >= 2
