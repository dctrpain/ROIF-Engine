"""
ROIF Engine
Reviewer Graph Factory

This module creates deterministic synthetic graph fixtures for the
reviewer-facing benchmark. Every method receives the same graph, the same
pre-stressed capacity landscape, and the same final observed state.

The hidden labels are generated independently from every tested method:

    true_root
        The seeded node at which the external disturbance is introduced.

    symptom_node
        The node with the smallest final reserve fraction. It may differ
        from true_root.

    true_node_star
        The node whose standardized local intervention produces the
        greatest reduction of global reserve deficit in a counterfactual
        simulation.

Supported graph families:

    linear chain
    balanced tree
    two-dimensional lattice
    Erdos-Renyi random graph
    Watts-Strogatz small-world graph
    Barabasi-Albert scale-free graph

This module is intentionally domain-independent. It provides controlled
synthetic evidence for algorithmic comparison; it does not establish
clinical or tissue-specific validity.

Typical use
-----------

    specs = reviewer_graph_specs()

    runner.run(
        specs,
        methods,
        fixture_factory=reviewer_fixture_factory,
        true_root_factory=reviewer_true_root,
        true_node_star_factory=reviewer_true_node_star,
        distance=reviewer_graph_distance,
    )

Author:
    Architect (Dctr Pain)

License:
    See project license.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Final
import math

import networkx as nx
import numpy as np

from .validation import (
    GraphFamily,
    GraphSpec,
    ValidationTrial,
)


class ReviewerGraphFactoryError(ValueError):
    """Raised when a reviewer graph fixture cannot be generated."""


class ObservationNoiseMode(str, Enum):
    """Noise model applied after generation of the hidden clean state."""

    NONE = "none"
    GAUSSIAN = "gaussian"


def _freeze_cache_value(value: Any) -> Any:
    """Convert nested metadata into a deterministic hashable value.

    The helper preserves semantic distinctions between mappings, sequences,
    sets, enums, and scalar values. It is intentionally local to cache-key
    construction; public metadata remains a read-only mapping.
    """

    if isinstance(value, Mapping):
        return tuple(sorted(
            (str(key), _freeze_cache_value(item))
            for key, item in value.items()
        ))
    if isinstance(value, tuple):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, list):
        return (
            "__list__",
            tuple(_freeze_cache_value(item) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        frozen_items = [_freeze_cache_value(item) for item in value]
        return (
            "__set__",
            tuple(sorted(frozen_items, key=repr)),
        )
    if isinstance(value, Enum):
        return (
            value.__class__.__module__,
            value.__class__.__qualname__,
            value.value,
        )
    try:
        hash(value)
    except TypeError:
        return (
            "__repr__",
            value.__class__.__module__,
            value.__class__.__qualname__,
            repr(value),
        )
    return value


@dataclass(frozen=True, slots=True)
class ReviewerGraphConfig:
    """
    Configuration for deterministic reviewer benchmark fixtures.

    The propagation model represents redistribution of structural load
    over a pre-stressed network. It is deliberately compact and fully
    reproducible rather than biologically parameterized.
    """

    directed: bool = True

    capacity_min: float = 0.80
    capacity_max: float = 1.20

    prestress_min: float = 0.18
    prestress_max: float = 0.48

    transmission_min: float = 0.12
    transmission_max: float = 0.42
    dissipation: float = 0.18

    disturbance_min: float = 0.16
    disturbance_max: float = 0.34

    propagation_steps: int = 18
    convergence_tolerance: float = 1e-10

    intervention_fraction: float = 0.16
    intervention_steps: int = 12

    observation_noise_mode: ObservationNoiseMode = (
        ObservationNoiseMode.NONE
    )
    observation_noise_std: float = 0.0

    edge_dropout_probability: float = 0.0
    capacity_noise_std: float = 0.0
    transmission_noise_std: float = 0.0
    missing_observation_probability: float = 0.0

    ensure_connected: bool = True
    minimum_root_degree: int = 1

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.directed, bool):
            raise ReviewerGraphFactoryError("directed must be bool.")

        for lower_name, upper_name in (
            ("capacity_min", "capacity_max"),
            ("prestress_min", "prestress_max"),
            ("transmission_min", "transmission_max"),
            ("disturbance_min", "disturbance_max"),
        ):
            lower = _finite_float(
                getattr(self, lower_name),
                name=lower_name,
            )
            upper = _finite_float(
                getattr(self, upper_name),
                name=upper_name,
            )
            if lower >= upper:
                raise ReviewerGraphFactoryError(
                    f"{lower_name} must be less than {upper_name}."
                )
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, upper_name, upper)

        if self.capacity_min <= 0.0:
            raise ReviewerGraphFactoryError(
                "capacity_min must be greater than zero."
            )

        if not 0.0 <= self.prestress_min < 1.0:
            raise ReviewerGraphFactoryError(
                "prestress_min must be within [0, 1)."
            )
        if not 0.0 < self.prestress_max < 1.0:
            raise ReviewerGraphFactoryError(
                "prestress_max must be within (0, 1)."
            )

        if not 0.0 <= self.transmission_min <= 1.0:
            raise ReviewerGraphFactoryError(
                "transmission_min must be within [0, 1]."
            )
        if not 0.0 <= self.transmission_max <= 1.0:
            raise ReviewerGraphFactoryError(
                "transmission_max must be within [0, 1]."
            )

        object.__setattr__(
            self,
            "dissipation",
            _unit_interval(self.dissipation, name="dissipation"),
        )
        if self.dissipation >= 1.0:
            raise ReviewerGraphFactoryError(
                "dissipation must be less than one."
            )

        for name in (
            "propagation_steps",
            "intervention_steps",
            "minimum_root_degree",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ReviewerGraphFactoryError(
                    f"{name} must be a positive integer."
                )

        tolerance = _finite_float(
            self.convergence_tolerance,
            name="convergence_tolerance",
        )
        if tolerance <= 0.0:
            raise ReviewerGraphFactoryError(
                "convergence_tolerance must be positive."
            )
        object.__setattr__(
            self,
            "convergence_tolerance",
            tolerance,
        )

        intervention_fraction = _finite_float(
            self.intervention_fraction,
            name="intervention_fraction",
        )
        if not 0.0 < intervention_fraction <= 1.0:
            raise ReviewerGraphFactoryError(
                "intervention_fraction must be within (0, 1]."
            )
        object.__setattr__(
            self,
            "intervention_fraction",
            intervention_fraction,
        )

        if not isinstance(
            self.observation_noise_mode,
            ObservationNoiseMode,
        ):
            raise ReviewerGraphFactoryError(
                "observation_noise_mode must be ObservationNoiseMode."
            )

        for name in (
            "observation_noise_std",
            "capacity_noise_std",
            "transmission_noise_std",
        ):
            value = _finite_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ReviewerGraphFactoryError(
                    f"{name} cannot be negative."
                )
            object.__setattr__(self, name, value)

        for name in (
            "edge_dropout_probability",
            "missing_observation_probability",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), name=name),
            )

        if not isinstance(self.ensure_connected, bool):
            raise ReviewerGraphFactoryError(
                "ensure_connected must be bool."
            )

        if not isinstance(self.metadata, Mapping):
            raise ReviewerGraphFactoryError(
                "metadata must be a mapping."
            )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    def __hash__(self) -> int:
        """Return a stable hash suitable for the fixture cache.

        ``MappingProxyType`` protects metadata from mutation but does not
        provide a hash. ReviewerGraphConfig is used as an ``lru_cache`` key,
        so metadata must be converted recursively into a deterministic,
        hashable representation.
        """

        return hash((
            self.directed,
            self.capacity_min,
            self.capacity_max,
            self.prestress_min,
            self.prestress_max,
            self.transmission_min,
            self.transmission_max,
            self.dissipation,
            self.disturbance_min,
            self.disturbance_max,
            self.propagation_steps,
            self.convergence_tolerance,
            self.intervention_fraction,
            self.intervention_steps,
            self.observation_noise_mode,
            self.observation_noise_std,
            self.edge_dropout_probability,
            self.capacity_noise_std,
            self.transmission_noise_std,
            self.missing_observation_probability,
            self.ensure_connected,
            self.minimum_root_degree,
            _freeze_cache_value(self.metadata),
        ))


@dataclass(frozen=True, slots=True)
class CascadeState:
    """Immutable result of one synthetic cascade simulation."""

    load: Mapping[int, float]
    reserve: Mapping[int, float]
    reserve_fraction: Mapping[int, float]
    global_deficit: float
    steps_completed: int
    converged: bool

    def __post_init__(self) -> None:
        for name in ("load", "reserve", "reserve_fraction"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ReviewerGraphFactoryError(
                    f"{name} must be a mapping."
                )
            object.__setattr__(
                self,
                name,
                MappingProxyType({
                    int(node): _finite_float(
                        item,
                        name=f"{name}[{node!r}]",
                    )
                    for node, item in value.items()
                }),
            )

        object.__setattr__(
            self,
            "global_deficit",
            _finite_float(
                self.global_deficit,
                name="global_deficit",
            ),
        )

        if (
            isinstance(self.steps_completed, bool)
            or not isinstance(self.steps_completed, int)
            or self.steps_completed < 0
        ):
            raise ReviewerGraphFactoryError(
                "steps_completed must be a nonnegative integer."
            )

        if not isinstance(self.converged, bool):
            raise ReviewerGraphFactoryError(
                "converged must be bool."
            )


@dataclass(frozen=True, slots=True)
class ReviewerFixture:
    """
    Immutable fixture shared by ROIF and every baseline method.

    Hidden truth is stored separately for evaluation. Baseline adapters
    must only inspect graph, capacity, load, reserve, symptom_node, and
    intervention_gain.
    """

    graph: nx.DiGraph
    node_order: tuple[int, ...]

    capacity: Mapping[int, float]
    initial_load: Mapping[int, float]
    initial_reserve: Mapping[int, float]

    load: Mapping[int, float]
    reserve: Mapping[int, float]
    reserve_fraction: Mapping[int, float]

    observed_state: Mapping[int, float]
    observed_mask: Mapping[int, bool]

    symptom_node: int
    intervention_gain: Mapping[int, float]

    true_root: int
    true_node_star: int

    disturbance: float
    global_deficit: float
    clean_global_deficit: float

    trial_seed: int
    graph_family: GraphFamily
    graph_name: str

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, nx.DiGraph):
            raise ReviewerGraphFactoryError(
                "graph must be a networkx.DiGraph."
            )

        node_order = tuple(self.node_order)
        graph_nodes = set(self.graph.nodes)
        if not node_order:
            raise ReviewerGraphFactoryError(
                "node_order cannot be empty."
            )
        if set(node_order) != graph_nodes:
            raise ReviewerGraphFactoryError(
                "node_order must contain every graph node exactly once."
            )
        object.__setattr__(self, "node_order", node_order)

        for name in (
            "capacity",
            "initial_load",
            "initial_reserve",
            "load",
            "reserve",
            "reserve_fraction",
            "observed_state",
            "intervention_gain",
        ):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping):
                raise ReviewerGraphFactoryError(
                    f"{name} must be a mapping."
                )
            if set(mapping) != graph_nodes:
                raise ReviewerGraphFactoryError(
                    f"{name} must contain every graph node."
                )
            object.__setattr__(
                self,
                name,
                MappingProxyType({
                    int(node): _finite_float(
                        value,
                        name=f"{name}[{node!r}]",
                    )
                    for node, value in mapping.items()
                }),
            )

        if not isinstance(self.observed_mask, Mapping):
            raise ReviewerGraphFactoryError(
                "observed_mask must be a mapping."
            )
        if set(self.observed_mask) != graph_nodes:
            raise ReviewerGraphFactoryError(
                "observed_mask must contain every graph node."
            )
        object.__setattr__(
            self,
            "observed_mask",
            MappingProxyType({
                int(node): bool(value)
                for node, value in self.observed_mask.items()
            }),
        )

        for name in (
            "symptom_node",
            "true_root",
            "true_node_star",
        ):
            value = getattr(self, name)
            if value not in graph_nodes:
                raise ReviewerGraphFactoryError(
                    f"{name} must be a graph node."
                )

        if not isinstance(self.graph_family, GraphFamily):
            raise ReviewerGraphFactoryError(
                "graph_family must be a GraphFamily."
            )

        if not isinstance(self.graph_name, str) or not self.graph_name:
            raise ReviewerGraphFactoryError(
                "graph_name must be a non-empty string."
            )

        if (
            isinstance(self.trial_seed, bool)
            or not isinstance(self.trial_seed, int)
        ):
            raise ReviewerGraphFactoryError(
                "trial_seed must be an integer."
            )

        for name in (
            "disturbance",
            "global_deficit",
            "clean_global_deficit",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )

        if not isinstance(self.metadata, Mapping):
            raise ReviewerGraphFactoryError(
                "metadata must be a mapping."
            )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def nodes(self) -> tuple[int, ...]:
        """Alias used by baseline adapters."""

        return self.node_order

    @property
    def capacities(self) -> Mapping[int, float]:
        """Alias used by baseline adapters."""

        return self.capacity

    @property
    def loads(self) -> Mapping[int, float]:
        """Alias used by baseline adapters."""

        return self.load

    @property
    def reserves(self) -> Mapping[int, float]:
        """Alias used by baseline adapters."""

        return self.reserve

    @property
    def intervention_gains(self) -> Mapping[int, float]:
        """Alias used by baseline adapters."""

        return self.intervention_gain

    def as_dict(
        self,
        *,
        include_hidden_truth: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-serializable fixture representation."""

        result = {
            "graph_family": self.graph_family.value,
            "graph_name": self.graph_name,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "nodes": list(self.node_order),
            "edges": [
                {
                    "source": int(source),
                    "target": int(target),
                    "weight": float(data.get("weight", 1.0)),
                    "transmission": float(
                        data.get("transmission", 1.0)
                    ),
                }
                for source, target, data in self.graph.edges(data=True)
            ],
            "capacity": dict(self.capacity),
            "initial_load": dict(self.initial_load),
            "initial_reserve": dict(self.initial_reserve),
            "load": dict(self.load),
            "reserve": dict(self.reserve),
            "reserve_fraction": dict(self.reserve_fraction),
            "observed_state": dict(self.observed_state),
            "observed_mask": dict(self.observed_mask),
            "symptom_node": self.symptom_node,
            "intervention_gain": dict(self.intervention_gain),
            "disturbance": self.disturbance,
            "global_deficit": self.global_deficit,
            "clean_global_deficit": self.clean_global_deficit,
            "trial_seed": self.trial_seed,
            "metadata": dict(self.metadata),
        }

        if include_hidden_truth:
            result["true_root"] = self.true_root
            result["true_node_star"] = self.true_node_star

        return result


DEFAULT_REVIEWER_NODE_COUNTS: Final[tuple[int, ...]] = (
    16,
    32,
    64,
)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ReviewerGraphFactoryError(
            f"{name} must be a real number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewerGraphFactoryError(
            f"{name} must be a real number."
        ) from exc
    if not math.isfinite(result):
        raise ReviewerGraphFactoryError(
            f"{name} must be finite."
        )
    return result


def _unit_interval(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ReviewerGraphFactoryError(
            f"{name} must be within [0, 1]."
        )
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ReviewerGraphFactoryError(
            f"{name} must be a positive integer."
        )
    return value


def _rng(seed: int, stream: int = 0) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReviewerGraphFactoryError(
            "seed must be an integer."
        )
    sequence = np.random.SeedSequence([seed, stream])
    return np.random.default_rng(sequence)


def _parameter(
    spec: GraphSpec,
    name: str,
    default: Any,
) -> Any:
    return spec.parameters.get(name, default)


def _balanced_tree(
    node_count: int,
    *,
    branching: int,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    for child in range(1, node_count):
        parent = (child - 1) // branching
        graph.add_edge(parent, child)

    return graph


def _grid_dimensions(node_count: int) -> tuple[int, int]:
    rows = int(math.floor(math.sqrt(node_count)))
    rows = max(rows, 1)
    columns = int(math.ceil(node_count / rows))
    return rows, columns


def _lattice_graph(node_count: int) -> nx.Graph:
    rows, columns = _grid_dimensions(node_count)
    full = nx.grid_2d_graph(rows, columns)

    selected = list(full.nodes)[:node_count]
    graph = full.subgraph(selected).copy()

    mapping = {
        node: index
        for index, node in enumerate(selected)
    }
    return nx.relabel_nodes(graph, mapping)


def _connected_erdos_renyi(
    node_count: int,
    *,
    probability: float,
    seed: int,
    ensure_connected: bool,
) -> nx.Graph:
    probability = _unit_interval(
        probability,
        name="erdos_renyi probability",
    )
    if node_count == 1:
        return nx.empty_graph(1)

    for attempt in range(64):
        graph_seed = int(
            _rng(seed, 100 + attempt).integers(0, 2**32 - 1)
        )
        graph = nx.erdos_renyi_graph(
            node_count,
            probability,
            seed=graph_seed,
            directed=False,
        )
        if not ensure_connected or nx.is_connected(graph):
            return graph

    graph = nx.erdos_renyi_graph(
        node_count,
        probability,
        seed=seed,
        directed=False,
    )

    components = [
        tuple(component)
        for component in nx.connected_components(graph)
    ]
    for left, right in zip(components, components[1:], strict=False):
        graph.add_edge(left[0], right[0])
    return graph


def _watts_strogatz(
    node_count: int,
    *,
    neighbour_count: int,
    rewiring_probability: float,
    seed: int,
) -> nx.Graph:
    if node_count < 3:
        return nx.path_graph(node_count)

    neighbour_count = min(neighbour_count, node_count - 1)
    if neighbour_count % 2 == 1:
        neighbour_count -= 1
    neighbour_count = max(neighbour_count, 2)

    rewiring_probability = _unit_interval(
        rewiring_probability,
        name="rewiring_probability",
    )

    return nx.connected_watts_strogatz_graph(
        node_count,
        neighbour_count,
        rewiring_probability,
        tries=100,
        seed=seed,
    )


def _barabasi_albert(
    node_count: int,
    *,
    attachment_count: int,
    seed: int,
) -> nx.Graph:
    if node_count < 2:
        return nx.empty_graph(node_count)

    attachment_count = max(
        1,
        min(attachment_count, node_count - 1),
    )
    return nx.barabasi_albert_graph(
        node_count,
        attachment_count,
        seed=seed,
    )


def _orient_graph(
    graph: nx.Graph,
    *,
    seed: int,
    directed: bool,
) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(sorted(graph.nodes))

    rng = _rng(seed, 200)

    for left, right in sorted(
        graph.edges,
        key=lambda edge: (int(edge[0]), int(edge[1])),
    ):
        if not directed:
            result.add_edge(left, right)
            result.add_edge(right, left)
            continue

        if float(rng.random()) < 0.5:
            result.add_edge(left, right)
        else:
            result.add_edge(right, left)

        # Preserve a weaker reverse compensation route. This keeps the
        # directed graph connected while retaining anisotropy.
        result.add_edge(right, left)
        result.add_edge(left, right)

    return result


def build_reviewer_graph(
    spec: GraphSpec,
    seed: int,
    config: ReviewerGraphConfig | None = None,
) -> nx.DiGraph:
    """Construct one deterministic graph from a GraphSpec."""

    if not isinstance(spec, GraphSpec):
        raise TypeError("spec must be a GraphSpec.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    config = config or ReviewerGraphConfig()
    if not isinstance(config, ReviewerGraphConfig):
        raise TypeError(
            "config must be a ReviewerGraphConfig."
        )

    node_count = spec.node_count

    if spec.family is GraphFamily.LINEAR_CHAIN:
        undirected = nx.path_graph(node_count)

    elif spec.family is GraphFamily.TREE:
        branching = _positive_int(
            int(_parameter(spec, "branching", 2)),
            name="branching",
        )
        undirected = _balanced_tree(
            node_count,
            branching=branching,
        )

    elif spec.family is GraphFamily.LATTICE:
        undirected = _lattice_graph(node_count)

    elif spec.family is GraphFamily.ERDOS_RENYI:
        default_probability = min(
            0.35,
            max(0.08, 3.0 / max(node_count - 1, 1)),
        )
        probability = float(
            _parameter(spec, "p", default_probability)
        )
        undirected = _connected_erdos_renyi(
            node_count,
            probability=probability,
            seed=seed,
            ensure_connected=config.ensure_connected,
        )

    elif spec.family is GraphFamily.WATTS_STROGATZ:
        neighbour_count = int(_parameter(spec, "k", 4))
        rewiring_probability = float(
            _parameter(spec, "p", 0.18)
        )
        undirected = _watts_strogatz(
            node_count,
            neighbour_count=neighbour_count,
            rewiring_probability=rewiring_probability,
            seed=seed,
        )

    elif spec.family is GraphFamily.BARABASI_ALBERT:
        attachment_count = int(_parameter(spec, "m", 2))
        undirected = _barabasi_albert(
            node_count,
            attachment_count=attachment_count,
            seed=seed,
        )

    else:
        raise ReviewerGraphFactoryError(
            "CUSTOM GraphSpec requires a caller-owned graph factory."
        )

    graph = _orient_graph(
        undirected,
        seed=seed,
        directed=config.directed,
    )

    edge_rng = _rng(seed, 300)
    for source, target in sorted(graph.edges):
        transmission = float(
            edge_rng.uniform(
                config.transmission_min,
                config.transmission_max,
            )
        )

        # Directional anisotropy is deterministic but asymmetric.
        direction_factor = float(
            edge_rng.uniform(0.75, 1.25)
        )
        transmission = float(
            np.clip(
                transmission * direction_factor,
                0.0,
                0.95,
            )
        )

        graph[source][target]["transmission"] = transmission
        graph[source][target]["weight"] = transmission
        graph[source][target]["distance"] = (
            1.0 / max(transmission, 1e-12)
        )

    return graph


def _capacity_landscape(
    graph: nx.DiGraph,
    seed: int,
    config: ReviewerGraphConfig,
) -> tuple[
    Mapping[int, float],
    Mapping[int, float],
    Mapping[int, float],
]:
    rng = _rng(seed, 400)
    nodes = tuple(sorted(graph.nodes))

    capacity_array = rng.uniform(
        config.capacity_min,
        config.capacity_max,
        size=len(nodes),
    )

    prestress_fraction = rng.uniform(
        config.prestress_min,
        config.prestress_max,
        size=len(nodes),
    )

    # Degree-dependent pre-stress prevents the landscape from being an
    # independent random vector while avoiding a direct centrality oracle.
    degrees = np.asarray(
        [
            graph.in_degree(node) + graph.out_degree(node)
            for node in nodes
        ],
        dtype=float,
    )
    if np.max(degrees) > 0.0:
        degree_component = degrees / np.max(degrees)
        prestress_fraction = np.clip(
            prestress_fraction
            + 0.06 * degree_component,
            config.prestress_min,
            min(config.prestress_max + 0.08, 0.88),
        )

    load_array = capacity_array * prestress_fraction
    reserve_array = capacity_array - load_array

    capacity = MappingProxyType({
        node: float(value)
        for node, value in zip(
            nodes,
            capacity_array,
            strict=True,
        )
    })
    load = MappingProxyType({
        node: float(value)
        for node, value in zip(
            nodes,
            load_array,
            strict=True,
        )
    })
    reserve = MappingProxyType({
        node: float(value)
        for node, value in zip(
            nodes,
            reserve_array,
            strict=True,
        )
    })

    return capacity, load, reserve


def _eligible_roots(
    graph: nx.DiGraph,
    *,
    minimum_degree: int,
) -> tuple[int, ...]:
    candidates = tuple(
        int(node)
        for node in sorted(graph.nodes)
        if (
            graph.in_degree(node) + graph.out_degree(node)
            >= minimum_degree
        )
    )
    return candidates or tuple(int(node) for node in sorted(graph.nodes))


def _select_true_root(
    graph: nx.DiGraph,
    seed: int,
    config: ReviewerGraphConfig,
) -> int:
    candidates = _eligible_roots(
        graph,
        minimum_degree=config.minimum_root_degree,
    )
    rng = _rng(seed, 500)
    index = int(rng.integers(0, len(candidates)))
    return int(candidates[index])


def _disturbance(
    seed: int,
    config: ReviewerGraphConfig,
) -> float:
    rng = _rng(seed, 600)
    return float(
        rng.uniform(
            config.disturbance_min,
            config.disturbance_max,
        )
    )


def _transition_matrix(
    graph: nx.DiGraph,
    nodes: Sequence[int],
    *,
    dissipation: float,
) -> np.ndarray:
    index = {
        node: position
        for position, node in enumerate(nodes)
    }
    matrix = np.zeros(
        (len(nodes), len(nodes)),
        dtype=float,
    )

    for source in nodes:
        outgoing = list(graph.out_edges(source, data=True))
        if not outgoing:
            continue

        raw_weights = np.asarray(
            [
                max(
                    float(data.get("transmission", 0.0)),
                    0.0,
                )
                for _, _, data in outgoing
            ],
            dtype=float,
        )
        total = float(np.sum(raw_weights))
        if total <= 0.0:
            continue

        retained_fraction = 1.0 - dissipation
        scaled = retained_fraction * raw_weights / total

        for (_, target, _), value in zip(
            outgoing,
            scaled,
            strict=True,
        ):
            matrix[index[target], index[source]] += float(value)

    return matrix


def _simulate(
    graph: nx.DiGraph,
    capacity: Mapping[int, float],
    initial_load: Mapping[int, float],
    *,
    disturbance_node: int | None,
    disturbance: float,
    config: ReviewerGraphConfig,
    steps: int | None = None,
) -> CascadeState:
    nodes = tuple(sorted(graph.nodes))
    index = {
        node: position
        for position, node in enumerate(nodes)
    }

    capacity_vector = np.asarray(
        [capacity[node] for node in nodes],
        dtype=float,
    )
    base_load = np.asarray(
        [initial_load[node] for node in nodes],
        dtype=float,
    )
    propagated = np.zeros(len(nodes), dtype=float)

    if disturbance_node is not None:
        propagated[index[disturbance_node]] = disturbance

    transition = _transition_matrix(
        graph,
        nodes,
        dissipation=config.dissipation,
    )

    limit = (
        config.propagation_steps
        if steps is None
        else _positive_int(steps, name="steps")
    )
    converged = False
    completed = 0

    cumulative = propagated.copy()
    current = propagated.copy()

    for completed in range(1, limit + 1):
        next_vector = transition @ current
        cumulative += next_vector

        if (
            float(np.linalg.norm(next_vector, ord=1))
            <= config.convergence_tolerance
        ):
            converged = True
            break

        current = next_vector

    final_load = np.maximum(base_load + cumulative, 0.0)
    reserve = capacity_vector - final_load
    reserve_fraction = reserve / capacity_vector

    deficit = np.maximum(-reserve, 0.0)
    utilization_excess = np.maximum(
        final_load / capacity_vector - 0.75,
        0.0,
    )

    # The objective rewards preservation of positive reserve and also
    # penalizes broad near-capacity loading before outright failure.
    global_deficit = float(
        np.sum(deficit / capacity_vector)
        + 0.20 * np.sum(utilization_excess)
    )

    return CascadeState(
        load={
            node: float(final_load[index[node]])
            for node in nodes
        },
        reserve={
            node: float(reserve[index[node]])
            for node in nodes
        },
        reserve_fraction={
            node: float(reserve_fraction[index[node]])
            for node in nodes
        },
        global_deficit=global_deficit,
        steps_completed=completed,
        converged=converged,
    )


def _intervention_gains(
    graph: nx.DiGraph,
    capacity: Mapping[int, float],
    initial_load: Mapping[int, float],
    final_state: CascadeState,
    config: ReviewerGraphConfig,
) -> Mapping[int, float]:
    nodes = tuple(sorted(graph.nodes))
    baseline_load = dict(final_state.load)
    baseline_deficit = final_state.global_deficit

    gains: dict[int, float] = {}

    for node in nodes:
        intervention_amount = (
            config.intervention_fraction * capacity[node]
        )
        intervened_load = dict(baseline_load)
        intervened_load[node] = max(
            0.0,
            intervened_load[node] - intervention_amount,
        )

        counterfactual = _simulate(
            graph,
            capacity,
            intervened_load,
            disturbance_node=None,
            disturbance=0.0,
            config=config,
            steps=config.intervention_steps,
        )

        gain = baseline_deficit - counterfactual.global_deficit
        gains[int(node)] = float(gain)

    return MappingProxyType(gains)


def _select_best_gain(
    gains: Mapping[int, float],
) -> int:
    if not gains:
        raise ReviewerGraphFactoryError(
            "intervention gains cannot be empty."
        )

    return int(
        min(
            gains,
            key=lambda node: (
                -gains[node],
                int(node),
            ),
        )
    )


def _apply_observation_degradation(
    state: CascadeState,
    capacity: Mapping[int, float],
    seed: int,
    config: ReviewerGraphConfig,
) -> tuple[
    Mapping[int, float],
    Mapping[int, bool],
]:
    nodes = tuple(sorted(state.reserve_fraction))
    rng = _rng(seed, 700)

    observed = np.asarray(
        [state.reserve_fraction[node] for node in nodes],
        dtype=float,
    )

    if (
        config.observation_noise_mode
        is ObservationNoiseMode.GAUSSIAN
        and config.observation_noise_std > 0.0
    ):
        observed += rng.normal(
            0.0,
            config.observation_noise_std,
            size=len(nodes),
        )

    mask_array = rng.random(len(nodes)) >= (
        config.missing_observation_probability
    )

    # At least one observation must remain available.
    if not bool(np.any(mask_array)):
        mask_array[int(rng.integers(0, len(nodes)))] = True

    observed_mapping = MappingProxyType({
        node: float(observed[index])
        for index, node in enumerate(nodes)
    })
    mask_mapping = MappingProxyType({
        node: bool(mask_array[index])
        for index, node in enumerate(nodes)
    })

    return observed_mapping, mask_mapping


def _degraded_graph(
    graph: nx.DiGraph,
    seed: int,
    config: ReviewerGraphConfig,
) -> nx.DiGraph:
    result = graph.copy()
    rng = _rng(seed, 800)

    if config.edge_dropout_probability > 0.0:
        removable = sorted(result.edges)
        for edge in removable:
            if (
                float(rng.random())
                < config.edge_dropout_probability
            ):
                result.remove_edge(*edge)

    if config.transmission_noise_std > 0.0:
        for source, target, data in result.edges(data=True):
            transmission = float(
                data.get("transmission", 0.0)
            )
            transmission += float(
                rng.normal(
                    0.0,
                    config.transmission_noise_std,
                )
            )
            transmission = float(
                np.clip(transmission, 0.0, 0.95)
            )
            data["transmission"] = transmission
            data["weight"] = transmission
            data["distance"] = (
                1.0 / max(transmission, 1e-12)
            )

    return result


def _degraded_capacity(
    capacity: Mapping[int, float],
    seed: int,
    config: ReviewerGraphConfig,
) -> Mapping[int, float]:
    if config.capacity_noise_std <= 0.0:
        return MappingProxyType(dict(capacity))

    rng = _rng(seed, 900)
    nodes = tuple(sorted(capacity))
    values = np.asarray(
        [capacity[node] for node in nodes],
        dtype=float,
    )
    values *= 1.0 + rng.normal(
        0.0,
        config.capacity_noise_std,
        size=len(nodes),
    )
    values = np.maximum(values, 1e-6)

    return MappingProxyType({
        node: float(values[index])
        for index, node in enumerate(nodes)
    })


def build_reviewer_fixture(
    spec: GraphSpec,
    seed: int,
    config: ReviewerGraphConfig | None = None,
) -> ReviewerFixture:
    """Build one deterministic hidden cascade and public fixture."""

    if not isinstance(spec, GraphSpec):
        raise TypeError("spec must be a GraphSpec.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    config = config or ReviewerGraphConfig()
    if not isinstance(config, ReviewerGraphConfig):
        raise TypeError(
            "config must be a ReviewerGraphConfig."
        )

    clean_graph = build_reviewer_graph(spec, seed, config)
    capacity, initial_load, initial_reserve = (
        _capacity_landscape(clean_graph, seed, config)
    )

    true_root = _select_true_root(
        clean_graph,
        seed,
        config,
    )
    disturbance = _disturbance(seed, config)

    clean_state = _simulate(
        clean_graph,
        capacity,
        initial_load,
        disturbance_node=true_root,
        disturbance=disturbance,
        config=config,
    )

    gains = _intervention_gains(
        clean_graph,
        capacity,
        initial_load,
        clean_state,
        config,
    )
    true_node_star = _select_best_gain(gains)

    symptom_node = int(
        min(
            clean_state.reserve_fraction,
            key=lambda node: (
                clean_state.reserve_fraction[node],
                int(node),
            ),
        )
    )

    observed_state, observed_mask = (
        _apply_observation_degradation(
            clean_state,
            capacity,
            seed,
            config,
        )
    )

    public_graph = _degraded_graph(
        clean_graph,
        seed,
        config,
    )
    public_capacity = _degraded_capacity(
        capacity,
        seed,
        config,
    )

    public_reserve = MappingProxyType({
        node: float(
            observed_state[node] * public_capacity[node]
        )
        for node in sorted(public_graph.nodes)
    })
    public_load = MappingProxyType({
        node: float(
            public_capacity[node] - public_reserve[node]
        )
        for node in sorted(public_graph.nodes)
    })

    return ReviewerFixture(
        graph=public_graph,
        node_order=tuple(sorted(public_graph.nodes)),
        capacity=public_capacity,
        initial_load=initial_load,
        initial_reserve=initial_reserve,
        load=public_load,
        reserve=public_reserve,
        reserve_fraction=observed_state,
        observed_state=observed_state,
        observed_mask=observed_mask,
        symptom_node=symptom_node,
        intervention_gain=gains,
        true_root=true_root,
        true_node_star=true_node_star,
        disturbance=disturbance,
        global_deficit=clean_state.global_deficit,
        clean_global_deficit=clean_state.global_deficit,
        trial_seed=seed,
        graph_family=spec.family,
        graph_name=spec.label,
        metadata={
            "project_stage": "research_prototype",
            "evidence_type": "controlled_synthetic_computation",
            "clinical_evidence": False,
            "state_dependent_propagation": True,
            "directed": config.directed,
            "propagation_steps": clean_state.steps_completed,
            "propagation_converged": clean_state.converged,
            "root_equals_symptom": true_root == symptom_node,
            "root_equals_node_star": (
                true_root == true_node_star
            ),
            "symptom_equals_node_star": (
                symptom_node == true_node_star
            ),
            "config": {
                "dissipation": config.dissipation,
                "observation_noise_mode": (
                    config.observation_noise_mode.value
                ),
                "observation_noise_std": (
                    config.observation_noise_std
                ),
                "edge_dropout_probability": (
                    config.edge_dropout_probability
                ),
                "capacity_noise_std": (
                    config.capacity_noise_std
                ),
                "transmission_noise_std": (
                    config.transmission_noise_std
                ),
                "missing_observation_probability": (
                    config.missing_observation_probability
                ),
            },
            **dict(config.metadata),
            "observed_state": dict(observed_state),
        },
    )


def _cached_fixture(
    spec_family: GraphFamily,
    node_count: int,
    edge_count: int | None,
    parameters_tuple: tuple[tuple[str, Any], ...],
    name: str | None,
    seed: int,
    config: ReviewerGraphConfig,
) -> ReviewerFixture:
    spec = GraphSpec(
        family=spec_family,
        node_count=node_count,
        edge_count=edge_count,
        parameters=dict(parameters_tuple),
        name=name,
    )
    return build_reviewer_fixture(spec, seed, config)


def reviewer_fixture_factory(
    trial: ValidationTrial,
    config: ReviewerGraphConfig | None = None,
) -> ReviewerFixture:
    """BenchmarkRunner-compatible fixture factory."""

    if not isinstance(trial, ValidationTrial):
        raise TypeError(
            "trial must be a ValidationTrial."
        )

    config = config or ReviewerGraphConfig()

    # Avoid exposing trial.true_root to generation. The fixture is rebuilt
    # solely from graph specification and seed.
    return _cached_fixture(
        trial.graph.family,
        trial.graph.node_count,
        trial.graph.edge_count,
        tuple(sorted(trial.graph.parameters.items())),
        trial.graph.name,
        trial.seed,
        config,
    )


def reviewer_true_root(
    spec: GraphSpec,
    repetition: int,
    seed: int,
    config: ReviewerGraphConfig | None = None,
) -> int:
    """BenchmarkRunner-compatible hidden-root factory."""

    del repetition
    fixture = build_reviewer_fixture(
        spec,
        seed,
        config or ReviewerGraphConfig(),
    )
    return fixture.true_root


def reviewer_true_node_star(
    spec: GraphSpec,
    repetition: int,
    seed: int,
    config: ReviewerGraphConfig | None = None,
) -> int:
    """BenchmarkRunner-compatible hidden Node* factory."""

    del repetition
    fixture = build_reviewer_fixture(
        spec,
        seed,
        config or ReviewerGraphConfig(),
    )
    return fixture.true_node_star


def reviewer_graph_distance(
    predicted: Any,
    truth: Any,
    fixture: Any,
) -> float:
    """Shortest-path distance used by validation metrics."""

    if not isinstance(fixture, ReviewerFixture):
        raise TypeError(
            "fixture must be a ReviewerFixture."
        )

    if predicted not in fixture.graph or truth not in fixture.graph:
        return float(fixture.graph.number_of_nodes())

    undirected = fixture.graph.to_undirected()

    try:
        return float(
            nx.shortest_path_length(
                undirected,
                source=predicted,
                target=truth,
            )
        )
    except nx.NetworkXNoPath:
        return float(fixture.graph.number_of_nodes())


def reviewer_graph_specs(
    node_counts: Sequence[int] = DEFAULT_REVIEWER_NODE_COUNTS,
) -> tuple[GraphSpec, ...]:
    """Create the standard graph specification set for reviewers."""

    counts = tuple(
        _positive_int(value, name="node_count")
        for value in node_counts
    )
    if not counts:
        raise ReviewerGraphFactoryError(
            "node_counts cannot be empty."
        )
    if len(set(counts)) != len(counts):
        raise ReviewerGraphFactoryError(
            "node_counts cannot contain duplicates."
        )

    specs: list[GraphSpec] = []

    for node_count in counts:
        lattice_rows, lattice_columns = _grid_dimensions(
            node_count
        )

        specs.extend(
            (
                GraphSpec(
                    family=GraphFamily.LINEAR_CHAIN,
                    node_count=node_count,
                    edge_count=max(node_count - 1, 0),
                    parameters={},
                    name=f"chain-{node_count}",
                    metadata={
                        "reviewer_benchmark": True,
                    },
                ),
                GraphSpec(
                    family=GraphFamily.TREE,
                    node_count=node_count,
                    edge_count=max(node_count - 1, 0),
                    parameters={"branching": 2},
                    name=f"tree-{node_count}",
                    metadata={
                        "reviewer_benchmark": True,
                    },
                ),
                GraphSpec(
                    family=GraphFamily.LATTICE,
                    node_count=node_count,
                    edge_count=None,
                    parameters={
                        "rows": lattice_rows,
                        "columns": lattice_columns,
                    },
                    name=f"lattice-{node_count}",
                    metadata={
                        "reviewer_benchmark": True,
                    },
                ),
                GraphSpec(
                    family=GraphFamily.ERDOS_RENYI,
                    node_count=node_count,
                    edge_count=None,
                    parameters={
                        "p": min(
                            0.35,
                            max(
                                0.08,
                                3.0 / max(node_count - 1, 1),
                            ),
                        ),
                    },
                    name=f"erdos-renyi-{node_count}",
                    metadata={
                        "reviewer_benchmark": True,
                    },
                ),
                GraphSpec(
                    family=GraphFamily.WATTS_STROGATZ,
                    node_count=node_count,
                    edge_count=None,
                    parameters={
                        "k": min(4, max(node_count - 1, 2)),
                        "p": 0.18,
                    },
                    name=f"watts-strogatz-{node_count}",
                    metadata={
                        "reviewer_benchmark": True,
                    },
                ),
                GraphSpec(
                    family=GraphFamily.BARABASI_ALBERT,
                    node_count=node_count,
                    edge_count=None,
                    parameters={
                        "m": min(2, max(node_count - 1, 1)),
                    },
                    name=f"barabasi-albert-{node_count}",
                    metadata={
                        "reviewer_benchmark": True,
                    },
                ),
            )
        )

    return tuple(specs)


def clean_reviewer_config() -> ReviewerGraphConfig:
    """Configuration for the primary clean benchmark."""

    return ReviewerGraphConfig(
        observation_noise_mode=ObservationNoiseMode.NONE,
        observation_noise_std=0.0,
        edge_dropout_probability=0.0,
        capacity_noise_std=0.0,
        transmission_noise_std=0.0,
        missing_observation_probability=0.0,
        metadata={"condition": "clean"},
    )


def moderate_noise_reviewer_config() -> ReviewerGraphConfig:
    """Configuration for moderate observation degradation."""

    return ReviewerGraphConfig(
        observation_noise_mode=ObservationNoiseMode.GAUSSIAN,
        observation_noise_std=0.035,
        edge_dropout_probability=0.03,
        capacity_noise_std=0.025,
        transmission_noise_std=0.020,
        missing_observation_probability=0.05,
        metadata={"condition": "moderate_noise"},
    )


def high_noise_reviewer_config() -> ReviewerGraphConfig:
    """Configuration for the high-noise sensitivity condition."""

    return ReviewerGraphConfig(
        observation_noise_mode=ObservationNoiseMode.GAUSSIAN,
        observation_noise_std=0.075,
        edge_dropout_probability=0.08,
        capacity_noise_std=0.060,
        transmission_noise_std=0.050,
        missing_observation_probability=0.12,
        metadata={"condition": "high_noise"},
    )


__all__ = [
    "DEFAULT_REVIEWER_NODE_COUNTS",
    "CascadeState",
    "ObservationNoiseMode",
    "ReviewerFixture",
    "ReviewerGraphConfig",
    "ReviewerGraphFactoryError",
    "build_reviewer_fixture",
    "build_reviewer_graph",
    "clean_reviewer_config",
    "high_noise_reviewer_config",
    "moderate_noise_reviewer_config",
    "reviewer_fixture_factory",
    "reviewer_graph_distance",
    "reviewer_graph_specs",
    "reviewer_true_node_star",
    "reviewer_true_root",
]
