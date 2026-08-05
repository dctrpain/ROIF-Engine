"""
ROIF Engine
Reviewer-Facing ROIF Method

This module exposes the ROIF inverse-cascade localization algorithm as a
BenchmarkMethod compatible with roif.benchmark.BenchmarkRunner.

The method receives only public fixture fields and never reads true_root,
true_node_star, or intervention_gain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final
import math

import networkx as nx
import numpy as np

from .benchmark import BenchmarkMethod, BenchmarkMethodKind
from .validation import ValidationOutcome, ValidationTrial


class ReviewerROIFError(ValueError):
    """Raised when reviewer-facing ROIF evaluation cannot proceed."""


class LocalizationLoss(str, Enum):
    WEIGHTED_RMSE = "weighted_rmse"
    WEIGHTED_MAE = "weighted_mae"
    HUBER = "huber"


@dataclass(frozen=True, slots=True)
class ReviewerROIFConfig:
    propagation_steps: int = 18
    intervention_steps: int = 12
    convergence_tolerance: float = 1e-10
    dissipation: float = 0.18
    disturbance_min: float = 0.02
    disturbance_max: float = 0.60
    disturbance_grid_size: int = 41
    disturbance_refinement_rounds: int = 2
    intervention_fraction: float = 0.16
    reserve_floor: float = 1e-8
    tensor_floor: float = 0.05
    tensor_ceiling: float = 1.50
    source_reserve_exponent: float = 0.50
    target_reserve_exponent: float = 1.00
    overload_amplification: float = 0.75
    localization_loss: LocalizationLoss = LocalizationLoss.WEIGHTED_RMSE
    huber_delta: float = 0.08
    missing_value_weight: float = 0.0
    near_failure_weight: float = 2.0
    use_state_dependent_tensor: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "propagation_steps",
            "intervention_steps",
            "disturbance_grid_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ReviewerROIFError(f"{name} must be a positive integer.")

        if (
            isinstance(self.disturbance_refinement_rounds, bool)
            or not isinstance(self.disturbance_refinement_rounds, int)
            or self.disturbance_refinement_rounds < 0
        ):
            raise ReviewerROIFError(
                "disturbance_refinement_rounds must be nonnegative."
            )

        for name in ("convergence_tolerance", "reserve_floor", "huber_delta"):
            value = _finite_float(getattr(self, name), name=name)
            if value <= 0.0:
                raise ReviewerROIFError(f"{name} must be positive.")
            object.__setattr__(self, name, value)

        for name in ("dissipation", "intervention_fraction", "missing_value_weight"):
            value = _finite_float(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ReviewerROIFError(f"{name} must be within [0, 1].")
            object.__setattr__(self, name, value)

        if self.dissipation >= 1.0:
            raise ReviewerROIFError("dissipation must be less than one.")
        if self.intervention_fraction <= 0.0:
            raise ReviewerROIFError(
                "intervention_fraction must be greater than zero."
            )

        for name in (
            "disturbance_min",
            "disturbance_max",
            "tensor_floor",
            "tensor_ceiling",
            "source_reserve_exponent",
            "target_reserve_exponent",
            "overload_amplification",
            "near_failure_weight",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )

        if self.disturbance_min < 0.0:
            raise ReviewerROIFError("disturbance_min cannot be negative.")
        if self.disturbance_min >= self.disturbance_max:
            raise ReviewerROIFError(
                "disturbance_min must be less than disturbance_max."
            )
        if self.tensor_floor <= 0.0:
            raise ReviewerROIFError("tensor_floor must be positive.")
        if self.tensor_floor > self.tensor_ceiling:
            raise ReviewerROIFError(
                "tensor_floor cannot exceed tensor_ceiling."
            )
        if min(
            self.source_reserve_exponent,
            self.target_reserve_exponent,
            self.overload_amplification,
            self.near_failure_weight,
        ) < 0.0:
            raise ReviewerROIFError(
                "tensor exponents and weights cannot be negative."
            )
        if not isinstance(self.localization_loss, LocalizationLoss):
            raise ReviewerROIFError(
                "localization_loss must be LocalizationLoss."
            )
        if not isinstance(self.use_state_dependent_tensor, bool):
            raise ReviewerROIFError(
                "use_state_dependent_tensor must be bool."
            )
        if not isinstance(self.metadata, Mapping):
            raise ReviewerROIFError("metadata must be a mapping.")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class PublicFixtureView:
    graph: nx.DiGraph
    nodes: tuple[Any, ...]
    capacity: Mapping[Any, float]
    initial_load: Mapping[Any, float]
    observed_load: Mapping[Any, float]
    observed_reserve_fraction: Mapping[Any, float]
    observed_mask: Mapping[Any, bool]
    symptom_node: Any | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, nx.DiGraph):
            raise ReviewerROIFError("graph must be networkx.DiGraph.")

        nodes = tuple(self.nodes)
        if not nodes or len(nodes) != len(set(nodes)):
            raise ReviewerROIFError("fixture nodes must be nonempty and unique.")
        if set(nodes) != set(self.graph.nodes):
            raise ReviewerROIFError("fixture nodes must match graph nodes.")
        object.__setattr__(self, "nodes", nodes)

        node_set = set(nodes)
        for name in (
            "capacity",
            "initial_load",
            "observed_load",
            "observed_reserve_fraction",
        ):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping) or set(mapping) != node_set:
                raise ReviewerROIFError(
                    f"{name} must contain every graph node."
                )
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        node: _finite_float(
                            value,
                            name=f"{name}[{node!r}]",
                        )
                        for node, value in mapping.items()
                    }
                ),
            )

        if (
            not isinstance(self.observed_mask, Mapping)
            or set(self.observed_mask) != node_set
        ):
            raise ReviewerROIFError(
                "observed_mask must contain every graph node."
            )
        object.__setattr__(
            self,
            "observed_mask",
            MappingProxyType(
                {
                    node: bool(value)
                    for node, value in self.observed_mask.items()
                }
            ),
        )

        if self.symptom_node is not None and self.symptom_node not in node_set:
            raise ReviewerROIFError("symptom_node must be a graph node.")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class ForwardState:
    load: Mapping[Any, float]
    reserve_fraction: Mapping[Any, float]
    global_deficit: float
    steps_completed: int
    converged: bool


@dataclass(frozen=True, slots=True)
class CandidateScore:
    node: Any
    error: float
    fitted_disturbance: float
    converged: bool
    predicted_reserve_fraction: Mapping[Any, float]


@dataclass(frozen=True, slots=True)
class InterventionScore:
    node: Any
    gain: float
    post_intervention_deficit: float
    converged: bool


@dataclass(frozen=True, slots=True)
class ReviewerROIFResult:
    d_fast: Any
    d_root: Any
    node_star: Any
    root_ranking: tuple[Any, ...]
    node_star_ranking: tuple[Any, ...]
    candidate_scores: tuple[CandidateScore, ...]
    intervention_scores: tuple[InterventionScore, ...]
    selected_gain: float
    optimal_gain: float
    converged: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ReviewerROIFError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewerROIFError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise ReviewerROIFError(f"{name} must be finite.")
    return result


def _read_field(
    fixture: Any,
    names: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    if isinstance(fixture, Mapping):
        for name in names:
            if name in fixture:
                return fixture[name]
        return default

    for name in names:
        if hasattr(fixture, name):
            return getattr(fixture, name)
    return default


def _node_mapping(
    value: Any,
    nodes: Sequence[Any],
    *,
    name: str,
) -> Mapping[Any, float]:
    if value is None:
        raise ReviewerROIFError(f"fixture must provide {name}.")

    if isinstance(value, Mapping):
        if set(value) != set(nodes):
            raise ReviewerROIFError(
                f"{name} must contain every graph node."
            )
        return MappingProxyType(
            {
                node: _finite_float(value[node], name=f"{name}[{node!r}]")
                for node in nodes
            }
        )

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(nodes):
            raise ReviewerROIFError(
                f"{name} length must equal node count."
            )
        return MappingProxyType(
            {
                node: _finite_float(item, name=f"{name}[{node!r}]")
                for node, item in zip(nodes, value, strict=True)
            }
        )

    raise ReviewerROIFError(
        f"{name} must be a mapping or node-aligned sequence."
    )


def public_fixture_view(fixture: Any) -> PublicFixtureView:
    graph = _read_field(fixture, ("graph", "network", "G"))
    if not isinstance(graph, nx.DiGraph):
        raise ReviewerROIFError(
            "fixture must expose graph as networkx.DiGraph."
        )

    nodes = tuple(
        _read_field(
            fixture,
            ("node_order", "nodes"),
            default=tuple(graph.nodes),
        )
    )

    capacity = _node_mapping(
        _read_field(fixture, ("capacity", "capacities")),
        nodes,
        name="capacity",
    )
    initial_load = _node_mapping(
        _read_field(fixture, ("initial_load",)),
        nodes,
        name="initial_load",
    )

    observed_load_raw = _read_field(
        fixture,
        ("load", "loads", "observed_load"),
    )
    observed_reserve_raw = _read_field(
        fixture,
        (
            "observed_state",
            "reserve_fraction",
            "observed_reserve_fraction",
        ),
    )

    if observed_load_raw is None and observed_reserve_raw is None:
        raise ReviewerROIFError(
            "fixture must provide observed load or reserve fraction."
        )

    if observed_reserve_raw is None:
        observed_load = _node_mapping(
            observed_load_raw,
            nodes,
            name="observed_load",
        )
        observed_reserve = MappingProxyType(
            {
                node: (
                    capacity[node] - observed_load[node]
                )
                / capacity[node]
                for node in nodes
            }
        )
    else:
        observed_reserve = _node_mapping(
            observed_reserve_raw,
            nodes,
            name="observed_reserve_fraction",
        )
        if observed_load_raw is None:
            observed_load = MappingProxyType(
                {
                    node: capacity[node] * (1.0 - observed_reserve[node])
                    for node in nodes
                }
            )
        else:
            observed_load = _node_mapping(
                observed_load_raw,
                nodes,
                name="observed_load",
            )

    mask_raw = _read_field(
        fixture,
        ("observed_mask",),
        default={node: True for node in nodes},
    )
    if not isinstance(mask_raw, Mapping):
        raise ReviewerROIFError("observed_mask must be a mapping.")

    return PublicFixtureView(
        graph=graph.copy(),
        nodes=nodes,
        capacity=capacity,
        initial_load=initial_load,
        observed_load=observed_load,
        observed_reserve_fraction=observed_reserve,
        observed_mask=MappingProxyType(
            {node: bool(mask_raw.get(node, True)) for node in nodes}
        ),
        symptom_node=_read_field(fixture, ("symptom_node",)),
        metadata=_read_field(fixture, ("metadata",), default={}),
    )


def _node_key(node: Any) -> tuple[str, str]:
    return type(node).__name__, repr(node)


def _base_edge_matrix(view: PublicFixtureView) -> np.ndarray:
    index = {node: position for position, node in enumerate(view.nodes)}
    matrix = np.zeros((len(view.nodes), len(view.nodes)), dtype=float)

    for source, target, data in view.graph.edges(data=True):
        transmission = _finite_float(
            data.get("transmission", data.get("weight", 1.0)),
            name=f"edge transmission {source!r}->{target!r}",
        )
        if transmission < 0.0:
            raise ReviewerROIFError(
                "edge transmission cannot be negative."
            )
        matrix[index[target], index[source]] += transmission

    return matrix


def _capacity_tensor(
    base_matrix: np.ndarray,
    reserve_fraction: np.ndarray,
    utilization: np.ndarray,
    config: ReviewerROIFConfig,
) -> np.ndarray:
    if not config.use_state_dependent_tensor:
        tensor = base_matrix.copy()
    else:
        positive_reserve = np.maximum(
            reserve_fraction,
            config.reserve_floor,
        )
        source_factor = np.power(
            positive_reserve,
            config.source_reserve_exponent,
        )
        target_factor = np.power(
            positive_reserve,
            config.target_reserve_exponent,
        )
        overload = np.maximum(utilization - 1.0, 0.0)
        overload_factor = (
            1.0 + config.overload_amplification * overload
        )
        tensor = (
            base_matrix
            * target_factor[:, np.newaxis]
            * source_factor[np.newaxis, :]
            * overload_factor[np.newaxis, :]
        )

    tensor = np.clip(tensor, 0.0, config.tensor_ceiling)
    target_total = 1.0 - config.dissipation

    for column in range(tensor.shape[1]):
        total = float(np.sum(tensor[:, column]))
        if total <= 0.0:
            continue
        tensor[:, column] *= target_total / total

        nonzero = tensor[:, column] > 0.0
        if np.any(nonzero):
            tensor[nonzero, column] = np.maximum(
                tensor[nonzero, column],
                config.tensor_floor
                * target_total
                / max(int(np.sum(nonzero)), 1),
            )
            renormalized = float(np.sum(tensor[:, column]))
            if renormalized > 0.0:
                tensor[:, column] *= target_total / renormalized

    return tensor


def simulate_forward(
    view: PublicFixtureView,
    *,
    initial_load: Mapping[Any, float],
    disturbance_node: Any | None,
    disturbance: float,
    config: ReviewerROIFConfig,
    steps: int | None = None,
) -> ForwardState:
    if disturbance_node is not None and disturbance_node not in view.nodes:
        raise ReviewerROIFError(
            "disturbance_node must be a graph node."
        )

    disturbance = _finite_float(disturbance, name="disturbance")
    if disturbance < 0.0:
        raise ReviewerROIFError("disturbance cannot be negative.")

    nodes = view.nodes
    index = {node: position for position, node in enumerate(nodes)}
    capacity = np.asarray(
        [view.capacity[node] for node in nodes],
        dtype=float,
    )
    base_load = np.asarray(
        [initial_load[node] for node in nodes],
        dtype=float,
    )

    current_packet = np.zeros(len(nodes), dtype=float)
    if disturbance_node is not None:
        current_packet[index[disturbance_node]] = disturbance

    cumulative = current_packet.copy()
    base_matrix = _base_edge_matrix(view)
    limit = config.propagation_steps if steps is None else steps

    converged = False
    completed = 0

    for completed in range(1, limit + 1):
        current_load = np.maximum(base_load + cumulative, 0.0)
        reserve_fraction = (capacity - current_load) / capacity
        utilization = current_load / capacity

        tensor = _capacity_tensor(
            base_matrix,
            reserve_fraction,
            utilization,
            config,
        )
        next_packet = tensor @ current_packet
        cumulative += next_packet

        if float(np.linalg.norm(next_packet, ord=1)) <= (
            config.convergence_tolerance
        ):
            converged = True
            break

        current_packet = next_packet

    final_load = np.maximum(base_load + cumulative, 0.0)
    reserve_fraction = (capacity - final_load) / capacity

    deficit = np.maximum(-reserve_fraction, 0.0)
    near_failure = np.maximum(0.25 - reserve_fraction, 0.0)
    global_deficit = float(
        np.sum(deficit) + 0.20 * np.sum(near_failure)
    )

    return ForwardState(
        load={
            node: float(final_load[index[node]])
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


def _observation_weights(
    view: PublicFixtureView,
    config: ReviewerROIFConfig,
) -> np.ndarray:
    observed = np.asarray(
        [view.observed_reserve_fraction[node] for node in view.nodes],
        dtype=float,
    )
    mask = np.asarray(
        [view.observed_mask[node] for node in view.nodes],
        dtype=bool,
    )
    weights = np.where(
        mask,
        1.0,
        config.missing_value_weight,
    ).astype(float)
    weights *= (
        1.0
        + config.near_failure_weight
        * np.clip(1.0 - observed, 0.0, 2.0)
    )
    if float(np.sum(weights)) <= 0.0:
        raise ReviewerROIFError(
            "at least one observation must have positive weight."
        )
    return weights


def localization_error(
    predicted: Mapping[Any, float],
    view: PublicFixtureView,
    config: ReviewerROIFConfig,
) -> float:
    predicted_vector = np.asarray(
        [predicted[node] for node in view.nodes],
        dtype=float,
    )
    observed_vector = np.asarray(
        [view.observed_reserve_fraction[node] for node in view.nodes],
        dtype=float,
    )
    weights = _observation_weights(view, config)
    residual = predicted_vector - observed_vector

    if config.localization_loss is LocalizationLoss.WEIGHTED_RMSE:
        return math.sqrt(
            float(
                np.sum(weights * residual**2) / np.sum(weights)
            )
        )
    if config.localization_loss is LocalizationLoss.WEIGHTED_MAE:
        return float(
            np.sum(weights * np.abs(residual)) / np.sum(weights)
        )

    absolute = np.abs(residual)
    quadratic = np.minimum(absolute, config.huber_delta)
    linear = absolute - quadratic
    huber = 0.5 * quadratic**2 + config.huber_delta * linear
    return float(np.sum(weights * huber) / np.sum(weights))


def fit_candidate_root(
    view: PublicFixtureView,
    candidate: Any,
    config: ReviewerROIFConfig,
) -> CandidateScore:
    if candidate not in view.nodes:
        raise ReviewerROIFError(
            "candidate must be a graph node."
        )

    lower = config.disturbance_min
    upper = config.disturbance_max
    best_error = float("inf")
    best_disturbance = lower
    best_state: ForwardState | None = None

    for refinement in range(
        config.disturbance_refinement_rounds + 1
    ):
        grid = np.linspace(
            lower,
            upper,
            num=config.disturbance_grid_size,
            dtype=float,
        )
        round_best_index = 0

        for index, disturbance in enumerate(grid):
            state = simulate_forward(
                view,
                initial_load=view.initial_load,
                disturbance_node=candidate,
                disturbance=float(disturbance),
                config=config,
            )
            error = localization_error(
                state.reserve_fraction,
                view,
                config,
            )

            if (
                error < best_error
                or (
                    math.isclose(
                        error,
                        best_error,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    and float(disturbance) < best_disturbance
                )
            ):
                best_error = error
                best_disturbance = float(disturbance)
                best_state = state
                round_best_index = index

        if refinement < config.disturbance_refinement_rounds:
            lower = float(grid[max(round_best_index - 1, 0)])
            upper = float(
                grid[min(round_best_index + 1, len(grid) - 1)]
            )
            if math.isclose(lower, upper):
                break

    if best_state is None:
        raise ReviewerROIFError(
            "candidate fitting produced no state."
        )

    return CandidateScore(
        node=candidate,
        error=best_error,
        fitted_disturbance=best_disturbance,
        converged=best_state.converged,
        predicted_reserve_fraction=MappingProxyType(
            dict(best_state.reserve_fraction)
        ),
    )


def rank_candidate_roots(
    view: PublicFixtureView,
    config: ReviewerROIFConfig,
) -> tuple[CandidateScore, ...]:
    return tuple(
        sorted(
            (
                fit_candidate_root(view, node, config)
                for node in view.nodes
            ),
            key=lambda item: (
                item.error,
                item.fitted_disturbance,
                _node_key(item.node),
            ),
        )
    )


def _observed_global_deficit(
    view: PublicFixtureView,
) -> float:
    reserve = np.asarray(
        [view.observed_reserve_fraction[node] for node in view.nodes],
        dtype=float,
    )
    return float(
        np.sum(np.maximum(-reserve, 0.0))
        + 0.20 * np.sum(np.maximum(0.25 - reserve, 0.0))
    )


def score_intervention(
    view: PublicFixtureView,
    node: Any,
    config: ReviewerROIFConfig,
) -> InterventionScore:
    if node not in view.nodes:
        raise ReviewerROIFError(
            "intervention node must be a graph node."
        )

    baseline_deficit = _observed_global_deficit(view)
    intervened_load = dict(view.observed_load)
    intervened_load[node] = max(
        0.0,
        intervened_load[node]
        - config.intervention_fraction * view.capacity[node],
    )

    state = simulate_forward(
        view,
        initial_load=intervened_load,
        disturbance_node=None,
        disturbance=0.0,
        config=config,
        steps=config.intervention_steps,
    )

    return InterventionScore(
        node=node,
        gain=float(baseline_deficit - state.global_deficit),
        post_intervention_deficit=state.global_deficit,
        converged=state.converged,
    )


def rank_interventions(
    view: PublicFixtureView,
    config: ReviewerROIFConfig,
) -> tuple[InterventionScore, ...]:
    return tuple(
        sorted(
            (
                score_intervention(view, node, config)
                for node in view.nodes
            ),
            key=lambda item: (
                -item.gain,
                item.post_intervention_deficit,
                _node_key(item.node),
            ),
        )
    )


def evaluate_reviewer_roif(
    fixture: Any,
    config: ReviewerROIFConfig | None = None,
) -> ReviewerROIFResult:
    config = config or ReviewerROIFConfig()
    view = public_fixture_view(fixture)

    observed_nodes = tuple(
        node for node in view.nodes if view.observed_mask[node]
    )
    if not observed_nodes:
        raise ReviewerROIFError(
            "fixture contains no observed nodes."
        )

    d_fast = min(
        observed_nodes,
        key=lambda node: (
            view.observed_reserve_fraction[node],
            _node_key(node),
        ),
    )
    candidates = rank_candidate_roots(view, config)
    interventions = rank_interventions(view, config)

    root_ranking = tuple(item.node for item in candidates)
    node_star_ranking = tuple(item.node for item in interventions)

    return ReviewerROIFResult(
        d_fast=d_fast,
        d_root=root_ranking[0],
        node_star=node_star_ranking[0],
        root_ranking=root_ranking,
        node_star_ranking=node_star_ranking,
        candidate_scores=candidates,
        intervention_scores=interventions,
        selected_gain=interventions[0].gain,
        optimal_gain=interventions[0].gain,
        converged=all(item.converged for item in candidates)
        and all(item.converged for item in interventions),
        metadata=MappingProxyType(
            {
                "algorithm": "roif_inverse_capacity_localization",
                "capacity_tensor": (
                    "state_dependent"
                    if config.use_state_dependent_tensor
                    else "static"
                ),
                "localization_loss": config.localization_loss.value,
                "candidate_count": len(candidates),
                "observed_node_count": len(observed_nodes),
                "d_fast": repr(d_fast),
                "best_localization_error": candidates[0].error,
                "second_localization_error": (
                    candidates[1].error
                    if len(candidates) > 1
                    else None
                ),
                "fitted_disturbance": (
                    candidates[0].fitted_disturbance
                ),
                **dict(config.metadata),
            }
        ),
    )


def execute_reviewer_roif(
    trial: ValidationTrial,
    fixture: Any,
    config: ReviewerROIFConfig | None = None,
) -> ValidationOutcome:
    if not isinstance(trial, ValidationTrial):
        raise TypeError(
            "trial must be a ValidationTrial."
        )

    result = evaluate_reviewer_roif(fixture, config)

    return ValidationOutcome(
        predicted_root=result.d_root,
        root_ranking=result.root_ranking,
        root_localization_distance=None,
        predicted_node_star=result.node_star,
        node_star_ranking=result.node_star_ranking,
        selected_gain=result.selected_gain,
        optimal_gain=result.optimal_gain,
        converged=result.converged,
        runtime_seconds=None,
        metadata={
            **dict(result.metadata),
            "method_role": "reviewer_facing_roif",
        },
    )


def make_reviewer_roif_method(
    config: ReviewerROIFConfig | None = None,
) -> BenchmarkMethod:
    config = config or ReviewerROIFConfig()

    return BenchmarkMethod(
        method_id="roif",
        label="ROIF inverse capacity localization",
        kind=BenchmarkMethodKind.ROIF,
        execute=lambda trial, fixture: execute_reviewer_roif(
            trial,
            fixture,
            config,
        ),
        metadata={
            "algorithm": "roif",
            "state_dependent_capacity_tensor": (
                config.use_state_dependent_tensor
            ),
            "localization_loss": config.localization_loss.value,
            "project_stage": "research_prototype",
            "evidence_type": "controlled_synthetic_computation",
            "clinical_evidence": False,
            **dict(config.metadata),
        },
    )


DEFAULT_REVIEWER_ROIF_CONFIG: Final[
    ReviewerROIFConfig
] = ReviewerROIFConfig()


__all__ = [
    "DEFAULT_REVIEWER_ROIF_CONFIG",
    "CandidateScore",
    "ForwardState",
    "InterventionScore",
    "LocalizationLoss",
    "PublicFixtureView",
    "ReviewerROIFConfig",
    "ReviewerROIFError",
    "ReviewerROIFResult",
    "evaluate_reviewer_roif",
    "execute_reviewer_roif",
    "fit_candidate_root",
    "localization_error",
    "make_reviewer_roif_method",
    "public_fixture_view",
    "rank_candidate_roots",
    "rank_interventions",
    "score_intervention",
    "simulate_forward",
]
