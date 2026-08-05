"""
ROIF Engine reviewer benchmark baselines.

Transparent baseline methods for comparison with ROIF on identical seeded
ValidationTrial fixtures. The module never reads hidden true_root or
true_node_star labels from the fixture.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Final
import hashlib
import math
import random

import numpy as np

from .benchmark import BenchmarkMethod, BenchmarkMethodKind
from .validation import ValidationOutcome, ValidationTrial


class BaselineError(ValueError):
    """Raised when baseline input or fixture data are invalid."""


class BaselineKind(str, Enum):
    RANDOM = "random"
    SYMPTOM_ONLY = "symptom_only"
    MINIMUM_RESERVE = "minimum_reserve"
    MAXIMUM_UTILIZATION = "maximum_utilization"
    DEGREE_CENTRALITY = "degree_centrality"
    IN_DEGREE_CENTRALITY = "in_degree_centrality"
    OUT_DEGREE_CENTRALITY = "out_degree_centrality"
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    CLOSENESS_CENTRALITY = "closeness_centrality"
    EIGENVECTOR_CENTRALITY = "eigenvector_centrality"
    PAGERANK = "pagerank"
    UPSTREAM_OF_SYMPTOM = "upstream_of_symptom"
    STATIC_FORWARD_MATCHING = "static_forward_matching"


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    directed: bool | None = None
    weighted: bool = True
    edge_weight_key: str = "weight"
    pagerank_alpha: float = 0.85
    eigenvector_max_iter: int = 1000
    eigenvector_tolerance: float = 1e-10
    random_salt: str = "roif-reviewer-baseline-v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.directed is not None and not isinstance(self.directed, bool):
            raise BaselineError("directed must be bool or None.")
        if not isinstance(self.weighted, bool):
            raise BaselineError("weighted must be bool.")
        for name in ("edge_weight_key", "random_salt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BaselineError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if (
            isinstance(self.pagerank_alpha, bool)
            or not isinstance(self.pagerank_alpha, (int, float))
            or not math.isfinite(float(self.pagerank_alpha))
            or not 0.0 < float(self.pagerank_alpha) < 1.0
        ):
            raise BaselineError("pagerank_alpha must be finite and within (0, 1).")
        if (
            isinstance(self.eigenvector_max_iter, bool)
            or not isinstance(self.eigenvector_max_iter, int)
            or self.eigenvector_max_iter < 1
        ):
            raise BaselineError("eigenvector_max_iter must be positive.")
        if (
            isinstance(self.eigenvector_tolerance, bool)
            or not isinstance(self.eigenvector_tolerance, (int, float))
            or not math.isfinite(float(self.eigenvector_tolerance))
            or float(self.eigenvector_tolerance) <= 0.0
        ):
            raise BaselineError("eigenvector_tolerance must be positive and finite.")
        if not isinstance(self.metadata, Mapping):
            raise BaselineError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class BaselineFixtureView:
    graph: Any
    nodes: tuple[Any, ...]
    reserve: Mapping[Any, float]
    capacity: Mapping[Any, float]
    load: Mapping[Any, float]
    symptom_node: Any | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def reserve_fraction(self, node: Any) -> float:
        reserve = self.reserve.get(node)
        capacity = self.capacity.get(node)
        if reserve is None:
            if capacity is not None and node in self.load:
                reserve = capacity - self.load[node]
            else:
                raise BaselineError(
                    "minimum-reserve baseline requires reserve values or both capacity and load."
                )
        if capacity is None:
            return reserve
        if capacity <= 0.0:
            raise BaselineError(f"capacity[{node!r}] must be greater than zero.")
        return reserve / capacity

    def utilization(self, node: Any) -> float:
        capacity = self.capacity.get(node)
        load = self.load.get(node)
        if load is None:
            reserve = self.reserve.get(node)
            if capacity is None or reserve is None:
                raise BaselineError(
                    "maximum-utilization baseline requires capacity and load or reserve."
                )
            load = capacity - reserve
        if capacity is None or capacity <= 0.0:
            raise BaselineError("maximum-utilization baseline requires positive capacity.")
        return load / capacity


@dataclass(frozen=True, slots=True)
class BaselineRanking:
    kind: BaselineKind
    ranking: tuple[Any, ...]
    scores: Mapping[Any, float]
    symptom_node: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


StaticPredictionFunction = Callable[
    [ValidationTrial, BaselineFixtureView, Any],
    Mapping[Any, float] | Sequence[float],
]


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise BaselineError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise BaselineError(f"{name} must be finite.")
    return result


def _field(value: Any, names: Sequence[str], *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _graph_nodes(graph: Any) -> tuple[Any, ...]:
    if graph is None:
        raise BaselineError("fixture must provide graph/network/G.")
    if isinstance(graph, Mapping):
        result: list[Any] = []
        seen: set[Any] = set()
        for source, neighbours in graph.items():
            if source not in seen:
                seen.add(source)
                result.append(source)
            iterator = neighbours.keys() if isinstance(neighbours, Mapping) else neighbours
            for target in iterator:
                if target not in seen:
                    seen.add(target)
                    result.append(target)
        return tuple(result)
    nodes_attr = getattr(graph, "nodes", None)
    if nodes_attr is None:
        raise BaselineError("graph must be an adjacency mapping or expose nodes.")
    return tuple(nodes_attr() if callable(nodes_attr) else nodes_attr)


def _node_mapping(value: Any, nodes: Sequence[Any], *, name: str) -> Mapping[Any, float]:
    if value is None:
        return MappingProxyType({})
    node_set = set(nodes)
    if isinstance(value, Mapping):
        return MappingProxyType({
            node: _finite_float(item, name=f"{name}[{node!r}]")
            for node, item in value.items()
            if node in node_set
        })
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(nodes):
            raise BaselineError(f"{name} sequence length must equal node count.")
        return MappingProxyType({
            node: _finite_float(item, name=f"{name}[{node!r}]")
            for node, item in zip(nodes, value, strict=True)
        })
    raise BaselineError(f"{name} must be a mapping or node-aligned sequence.")


def fixture_view(fixture: Any) -> BaselineFixtureView:
    if isinstance(fixture, BaselineFixtureView):
        return fixture
    graph = _field(fixture, ("graph", "network", "G"))
    explicit_nodes = _field(fixture, ("node_order", "nodes"), default=None)
    nodes = tuple(explicit_nodes) if explicit_nodes is not None else _graph_nodes(graph)
    if not nodes:
        raise BaselineError("fixture graph must contain at least one node.")
    reserve = _node_mapping(
        _field(fixture, ("reserve", "reserves", "reserve_by_node")),
        nodes,
        name="reserve",
    )
    capacity = _node_mapping(
        _field(fixture, ("capacity", "capacities", "capacity_by_node")),
        nodes,
        name="capacity",
    )
    load = _node_mapping(
        _field(fixture, ("load", "loads", "load_by_node")),
        nodes,
        name="load",
    )
    symptom_node = _field(
        fixture,
        ("symptom_node", "observed_node", "manifestation_node"),
    )
    if symptom_node is not None and symptom_node not in set(nodes):
        raise BaselineError("symptom_node must be present in nodes.")
    metadata = _field(fixture, ("metadata",), default={})
    if not isinstance(metadata, Mapping):
        raise BaselineError("fixture metadata must be a mapping.")
    return BaselineFixtureView(
        graph=graph,
        nodes=nodes,
        reserve=reserve,
        capacity=capacity,
        load=load,
        symptom_node=symptom_node,
        metadata=MappingProxyType(dict(metadata)),
    )


def _stable_key(node: Any) -> tuple[str, str]:
    return type(node).__name__, repr(node)


def _rank(nodes: Sequence[Any], scores: Mapping[Any, float]) -> tuple[Any, ...]:
    missing = [node for node in nodes if node not in scores]
    if missing:
        raise BaselineError(f"score mapping misses {len(missing)} nodes.")
    return tuple(sorted(nodes, key=lambda node: (-scores[node], _stable_key(node))))


def _symptom(view: BaselineFixtureView) -> Any:
    if view.symptom_node is not None:
        return view.symptom_node
    values = {node: view.reserve_fraction(node) for node in view.nodes}
    return min(view.nodes, key=lambda node: (values[node], _stable_key(node)))


def _trial_seed(trial: ValidationTrial, salt: str) -> int:
    payload = f"{getattr(trial, 'seed', 0)}|{getattr(trial, 'trial_id', '')}|{salt}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _is_directed(graph: Any, config: BaselineConfig) -> bool:
    if config.directed is not None:
        return config.directed
    method = getattr(graph, "is_directed", None)
    return bool(method()) if callable(method) else bool(getattr(graph, "directed", False))


def _networkx() -> Any:
    try:
        import networkx as nx
    except ImportError as exc:
        raise BaselineError("NetworkX is required for centrality baselines.") from exc
    return nx


def _to_networkx(view: BaselineFixtureView, config: BaselineConfig) -> Any:
    nx = _networkx()
    if isinstance(view.graph, (nx.Graph, nx.DiGraph)):
        return view.graph.copy()
    graph = nx.DiGraph() if _is_directed(view.graph, config) else nx.Graph()
    graph.add_nodes_from(view.nodes)
    if isinstance(view.graph, Mapping):
        for source, neighbours in view.graph.items():
            if isinstance(neighbours, Mapping):
                for target, data in neighbours.items():
                    weight = data.get(config.edge_weight_key, 1.0) if isinstance(data, Mapping) else data
                    graph.add_edge(source, target, **{config.edge_weight_key: float(weight)})
            else:
                for target in neighbours:
                    graph.add_edge(source, target, **{config.edge_weight_key: 1.0})
        return graph
    edges_attr = getattr(view.graph, "edges", None)
    if edges_attr is None:
        raise BaselineError("graph does not expose edges.")
    try:
        edges = edges_attr(data=True) if callable(edges_attr) else edges_attr
    except TypeError:
        edges = edges_attr()
    for item in edges:
        if len(item) == 2:
            source, target = item
            weight = 1.0
        else:
            source, target, data = item[:3]
            weight = data.get(config.edge_weight_key, 1.0) if isinstance(data, Mapping) else data
        graph.add_edge(source, target, **{config.edge_weight_key: float(weight)})
    return graph


def rank_random(trial: ValidationTrial, view: BaselineFixtureView, config: BaselineConfig) -> BaselineRanking:
    rng = random.Random(_trial_seed(trial, config.random_salt))
    ranking = list(view.nodes)
    rng.shuffle(ranking)
    scores = {node: float(len(ranking) - index) for index, node in enumerate(ranking)}
    return BaselineRanking(BaselineKind.RANDOM, tuple(ranking), MappingProxyType(scores))


def rank_symptom_only(view: BaselineFixtureView) -> BaselineRanking:
    symptom = _symptom(view)
    ranking = (symptom, *sorted((n for n in view.nodes if n != symptom), key=_stable_key))
    scores = {node: 1.0 if node == symptom else 0.0 for node in view.nodes}
    return BaselineRanking(BaselineKind.SYMPTOM_ONLY, ranking, MappingProxyType(scores), symptom)


def rank_minimum_reserve(view: BaselineFixtureView) -> BaselineRanking:
    scores = {node: -view.reserve_fraction(node) for node in view.nodes}
    ranking = _rank(view.nodes, scores)
    return BaselineRanking(BaselineKind.MINIMUM_RESERVE, ranking, MappingProxyType(scores), ranking[0])


def rank_maximum_utilization(view: BaselineFixtureView) -> BaselineRanking:
    scores = {node: view.utilization(node) for node in view.nodes}
    ranking = _rank(view.nodes, scores)
    return BaselineRanking(BaselineKind.MAXIMUM_UTILIZATION, ranking, MappingProxyType(scores), _symptom(view))


def rank_degree(view: BaselineFixtureView, config: BaselineConfig, mode: str = "total") -> BaselineRanking:
    graph = _to_networkx(view, config)
    if mode == "in":
        values = dict(graph.in_degree()) if graph.is_directed() else dict(graph.degree())
        kind = BaselineKind.IN_DEGREE_CENTRALITY
    elif mode == "out":
        values = dict(graph.out_degree()) if graph.is_directed() else dict(graph.degree())
        kind = BaselineKind.OUT_DEGREE_CENTRALITY
    elif mode == "total":
        values = dict(graph.degree())
        kind = BaselineKind.DEGREE_CENTRALITY
    else:
        raise BaselineError("degree mode must be total, in, or out.")
    scores = {node: float(values.get(node, 0.0)) for node in view.nodes}
    return BaselineRanking(kind, _rank(view.nodes, scores), MappingProxyType(scores))


def rank_betweenness(view: BaselineFixtureView, config: BaselineConfig) -> BaselineRanking:
    nx = _networkx()
    graph = _to_networkx(view, config)
    scores = nx.betweenness_centrality(
        graph,
        normalized=True,
        weight=config.edge_weight_key if config.weighted else None,
    )
    return BaselineRanking(BaselineKind.BETWEENNESS_CENTRALITY, _rank(view.nodes, scores), MappingProxyType(scores))


def rank_closeness(view: BaselineFixtureView, config: BaselineConfig) -> BaselineRanking:
    nx = _networkx()
    graph = _to_networkx(view, config)
    scores = nx.closeness_centrality(
        graph,
        distance=config.edge_weight_key if config.weighted else None,
    )
    return BaselineRanking(BaselineKind.CLOSENESS_CENTRALITY, _rank(view.nodes, scores), MappingProxyType(scores))


def rank_eigenvector(view: BaselineFixtureView, config: BaselineConfig) -> BaselineRanking:
    nx = _networkx()
    graph = _to_networkx(view, config)
    weight = config.edge_weight_key if config.weighted else None
    try:
        scores = nx.eigenvector_centrality(
            graph,
            max_iter=config.eigenvector_max_iter,
            tol=config.eigenvector_tolerance,
            weight=weight,
        )
    except nx.PowerIterationFailedConvergence:
        scores = nx.eigenvector_centrality_numpy(graph, weight=weight)
    return BaselineRanking(BaselineKind.EIGENVECTOR_CENTRALITY, _rank(view.nodes, scores), MappingProxyType(scores))


def rank_pagerank(view: BaselineFixtureView, config: BaselineConfig) -> BaselineRanking:
    nx = _networkx()
    graph = _to_networkx(view, config)
    scores = nx.pagerank(
        graph,
        alpha=float(config.pagerank_alpha),
        weight=config.edge_weight_key if config.weighted else None,
    )
    return BaselineRanking(BaselineKind.PAGERANK, _rank(view.nodes, scores), MappingProxyType(scores))


def rank_upstream_of_symptom(view: BaselineFixtureView, config: BaselineConfig | None = None) -> BaselineRanking:
    config = config or BaselineConfig()
    nx = _networkx()
    graph = _to_networkx(view, config)
    symptom = _symptom(view)
    if graph.is_directed():
        graph = graph.reverse(copy=False)
    lengths = nx.single_source_shortest_path_length(graph, symptom)
    unreachable = len(view.nodes) + 1
    scores = {node: -float(lengths.get(node, unreachable)) for node in view.nodes}
    return BaselineRanking(BaselineKind.UPSTREAM_OF_SYMPTOM, _rank(view.nodes, scores), MappingProxyType(scores), symptom)


def rank_static_forward_matching(
    trial: ValidationTrial,
    view: BaselineFixtureView,
    predictor: StaticPredictionFunction,
) -> BaselineRanking:
    observed = _field(view.metadata, ("observed_state", "observation"))
    if observed is None:
        raise BaselineError("fixture metadata must contain observed_state.")
    observed_vector = np.asarray(
        [observed[node] for node in view.nodes] if isinstance(observed, Mapping) else observed,
        dtype=float,
    )
    if observed_vector.shape != (len(view.nodes),) or not np.all(np.isfinite(observed_vector)):
        raise BaselineError("observed_state must be one finite scalar per node.")
    errors: dict[Any, float] = {}
    for candidate in view.nodes:
        prediction = predictor(trial, view, candidate)
        vector = np.asarray(
            [prediction[node] for node in view.nodes] if isinstance(prediction, Mapping) else prediction,
            dtype=float,
        )
        if vector.shape != observed_vector.shape or not np.all(np.isfinite(vector)):
            raise BaselineError("static predictor returned invalid state vector.")
        errors[candidate] = float(np.linalg.norm(vector - observed_vector))
    scores = {node: -error for node, error in errors.items()}
    return BaselineRanking(
        BaselineKind.STATIC_FORWARD_MATCHING,
        _rank(view.nodes, scores),
        MappingProxyType(scores),
        metadata=MappingProxyType({"errors": errors}),
    )


def _outcome(result: BaselineRanking) -> ValidationOutcome:
    ranking = tuple(result.ranking)
    return ValidationOutcome(
        predicted_root=ranking[0],
        root_ranking=ranking,
        root_localization_distance=None,
        predicted_node_star=None,
        node_star_ranking=(),
        selected_gain=None,
        optimal_gain=None,
        converged=True,
        runtime_seconds=None,
        metadata={
            "baseline_kind": result.kind.value,
            "scores": {repr(node): score for node, score in result.scores.items()},
            "symptom_node": None if result.symptom_node is None else repr(result.symptom_node),
            **dict(result.metadata),
        },
    )


def _method(method_id: str, label: str, kind: BaselineKind, ranker: Callable[[ValidationTrial, BaselineFixtureView], BaselineRanking], metadata: Mapping[str, Any] | None = None) -> BenchmarkMethod:
    def execute(trial: ValidationTrial, fixture: Any) -> ValidationOutcome:
        return _outcome(ranker(trial, fixture_view(fixture)))
    return BenchmarkMethod(
        method_id=method_id,
        label=label,
        kind=BenchmarkMethodKind.BASELINE,
        execute=execute,
        metadata={"baseline_kind": kind.value, **dict(metadata or {})},
    )


def make_random_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("random", "Seeded random ranking", BaselineKind.RANDOM, lambda trial, view: rank_random(trial, view, config), config.metadata)


def make_symptom_only_baseline() -> BenchmarkMethod:
    return _method("symptom_only", "Observed symptom node", BaselineKind.SYMPTOM_ONLY, lambda _trial, view: rank_symptom_only(view))


def make_minimum_reserve_baseline() -> BenchmarkMethod:
    return _method("minimum_reserve", "Minimum reserve fraction", BaselineKind.MINIMUM_RESERVE, lambda _trial, view: rank_minimum_reserve(view))


def make_maximum_utilization_baseline() -> BenchmarkMethod:
    return _method("maximum_utilization", "Maximum utilization", BaselineKind.MAXIMUM_UTILIZATION, lambda _trial, view: rank_maximum_utilization(view))


def make_degree_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("degree", "Degree centrality", BaselineKind.DEGREE_CENTRALITY, lambda _trial, view: rank_degree(view, config, "total"), config.metadata)


def make_in_degree_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("in_degree", "In-degree centrality", BaselineKind.IN_DEGREE_CENTRALITY, lambda _trial, view: rank_degree(view, config, "in"), config.metadata)


def make_out_degree_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("out_degree", "Out-degree centrality", BaselineKind.OUT_DEGREE_CENTRALITY, lambda _trial, view: rank_degree(view, config, "out"), config.metadata)


def make_betweenness_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("betweenness", "Betweenness centrality", BaselineKind.BETWEENNESS_CENTRALITY, lambda _trial, view: rank_betweenness(view, config), config.metadata)


def make_closeness_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("closeness", "Closeness centrality", BaselineKind.CLOSENESS_CENTRALITY, lambda _trial, view: rank_closeness(view, config), config.metadata)


def make_eigenvector_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("eigenvector", "Eigenvector centrality", BaselineKind.EIGENVECTOR_CENTRALITY, lambda _trial, view: rank_eigenvector(view, config), config.metadata)


def make_pagerank_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("pagerank", "PageRank", BaselineKind.PAGERANK, lambda _trial, view: rank_pagerank(view, config), config.metadata)


def make_upstream_symptom_baseline(config: BaselineConfig | None = None) -> BenchmarkMethod:
    config = config or BaselineConfig()
    return _method("upstream_symptom", "Shortest upstream path to symptom", BaselineKind.UPSTREAM_OF_SYMPTOM, lambda _trial, view: rank_upstream_of_symptom(view, config), config.metadata)


def make_static_forward_matching_baseline(predictor: StaticPredictionFunction) -> BenchmarkMethod:
    if not callable(predictor):
        raise BaselineError("predictor must be callable.")
    return _method(
        "static_forward_matching",
        "Static forward-state matching",
        BaselineKind.STATIC_FORWARD_MATCHING,
        lambda trial, view: rank_static_forward_matching(trial, view, predictor),
        {"state_dependent": False},
    )


def reviewer_baselines(
    config: BaselineConfig | None = None,
    *,
    include_directed_degree: bool = True,
    include_closeness: bool = True,
    include_eigenvector: bool = True,
    include_pagerank: bool = True,
) -> tuple[BenchmarkMethod, ...]:
    config = config or BaselineConfig()
    methods: list[BenchmarkMethod] = [
        make_random_baseline(config),
        make_symptom_only_baseline(),
        make_minimum_reserve_baseline(),
        make_maximum_utilization_baseline(),
        make_degree_baseline(config),
        make_betweenness_baseline(config),
        make_upstream_symptom_baseline(config),
    ]
    if include_directed_degree:
        methods.extend((make_in_degree_baseline(config), make_out_degree_baseline(config)))
    if include_closeness:
        methods.append(make_closeness_baseline(config))
    if include_eigenvector:
        methods.append(make_eigenvector_baseline(config))
    if include_pagerank:
        methods.append(make_pagerank_baseline(config))
    return tuple(methods)


STANDARD_BASELINE_IDS: Final[tuple[str, ...]] = (
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
)


__all__ = [
    "STANDARD_BASELINE_IDS",
    "BaselineConfig",
    "BaselineError",
    "BaselineFixtureView",
    "BaselineKind",
    "BaselineRanking",
    "StaticPredictionFunction",
    "fixture_view",
    "make_betweenness_baseline",
    "make_closeness_baseline",
    "make_degree_baseline",
    "make_eigenvector_baseline",
    "make_in_degree_baseline",
    "make_maximum_utilization_baseline",
    "make_minimum_reserve_baseline",
    "make_out_degree_baseline",
    "make_pagerank_baseline",
    "make_random_baseline",
    "make_static_forward_matching_baseline",
    "make_symptom_only_baseline",
    "make_upstream_symptom_baseline",
    "rank_betweenness",
    "rank_closeness",
    "rank_degree",
    "rank_eigenvector",
    "rank_maximum_utilization",
    "rank_minimum_reserve",
    "rank_pagerank",
    "rank_random",
    "rank_static_forward_matching",
    "rank_symptom_only",
    "rank_upstream_of_symptom",
    "reviewer_baselines",
]
