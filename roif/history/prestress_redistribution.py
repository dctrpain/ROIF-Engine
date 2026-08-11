"""
ROIF Engine
Prestress Redistribution Core

Purpose
-------
Model how a local prestress change propagates through an already prestressed
interaction network.

The module is domain-agnostic. A node may represent a rigid structural element,
region, anchor, joint, material domain, or other system component. A connection
represents a path through which a change in prestress may be transmitted.

Core idea
---------
A local event changes prestress at one or more nodes:

    P_t -> P_t + dP

That change may alter the loads transmitted through the connected network:

    local change
        -> transmitted contribution
        -> capacity-limited redistribution
        -> new global prestress state

Boundaries
----------
- explicit topology only
- explicit gains only
- explicit attenuation only
- capacity limits enforced
- no hidden learning
- no topology mutation
- deterministic update order
- immutable result objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "prestress_redistribution_v1"


class PrestressRedistributionError(ValueError):
    """Raised when redistribution inputs are invalid."""


def _readonly(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else dict(value)
    )


def _validate_id(
    name: str,
    value: str,
) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise PrestressRedistributionError(
            f"{name} must not be empty"
        )
    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    number = float(value)
    if not isfinite(number):
        raise PrestressRedistributionError(
            f"{name} must be finite"
        )
    return number


def _clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return min(upper, max(lower, value))


@dataclass(frozen=True, slots=True)
class PrestressNodeState:
    node_id: str
    prestress: float
    min_prestress: float = -1.0
    max_prestress: float = 1.0
    reserve: float = 1.0
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        node_id = _validate_id(
            "node_id",
            self.node_id,
        )
        prestress = _validate_finite(
            "prestress",
            self.prestress,
        )
        min_prestress = _validate_finite(
            "min_prestress",
            self.min_prestress,
        )
        max_prestress = _validate_finite(
            "max_prestress",
            self.max_prestress,
        )
        reserve = _validate_finite(
            "reserve",
            self.reserve,
        )

        if min_prestress > max_prestress:
            raise PrestressRedistributionError(
                "min_prestress must be <= max_prestress"
            )

        if not (
            min_prestress
            <= prestress
            <= max_prestress
        ):
            raise PrestressRedistributionError(
                "prestress must lie within node bounds"
            )

        if not (
            0.0
            <= reserve
            <= 1.0
        ):
            raise PrestressRedistributionError(
                "reserve must lie in [0, 1]"
            )

        object.__setattr__(
            self,
            "node_id",
            node_id,
        )
        object.__setattr__(
            self,
            "prestress",
            prestress,
        )
        object.__setattr__(
            self,
            "min_prestress",
            min_prestress,
        )
        object.__setattr__(
            self,
            "max_prestress",
            max_prestress,
        )
        object.__setattr__(
            self,
            "reserve",
            reserve,
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class PrestressConnection:
    connection_id: str
    source_node_id: str
    target_node_id: str
    transfer_gain: float = 1.0
    attenuation: float = 1.0
    capacity: float = 1.0
    enabled: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        connection_id = _validate_id(
            "connection_id",
            self.connection_id,
        )
        source_node_id = _validate_id(
            "source_node_id",
            self.source_node_id,
        )
        target_node_id = _validate_id(
            "target_node_id",
            self.target_node_id,
        )

        if source_node_id == target_node_id:
            raise PrestressRedistributionError(
                "self-connections are not allowed"
            )

        transfer_gain = _validate_finite(
            "transfer_gain",
            self.transfer_gain,
        )
        attenuation = _validate_finite(
            "attenuation",
            self.attenuation,
        )
        capacity = _validate_finite(
            "capacity",
            self.capacity,
        )

        if not (
            0.0
            <= attenuation
            <= 1.0
        ):
            raise PrestressRedistributionError(
                "attenuation must lie in [0, 1]"
            )

        if capacity < 0.0:
            raise PrestressRedistributionError(
                "capacity must be nonnegative"
            )

        object.__setattr__(
            self,
            "connection_id",
            connection_id,
        )
        object.__setattr__(
            self,
            "source_node_id",
            source_node_id,
        )
        object.__setattr__(
            self,
            "target_node_id",
            target_node_id,
        )
        object.__setattr__(
            self,
            "transfer_gain",
            transfer_gain,
        )
        object.__setattr__(
            self,
            "attenuation",
            attenuation,
        )
        object.__setattr__(
            self,
            "capacity",
            capacity,
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class PrestressPerturbation:
    perturbation_id: str
    node_id: str
    delta: float
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "perturbation_id",
            _validate_id(
                "perturbation_id",
                self.perturbation_id,
            ),
        )
        object.__setattr__(
            self,
            "node_id",
            _validate_id(
                "node_id",
                self.node_id,
            ),
        )
        object.__setattr__(
            self,
            "delta",
            _validate_finite(
                "delta",
                self.delta,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class PrestressRedistributionConfig:
    propagation_steps: int = 4
    propagation_decay: float = 0.75
    reserve_modulation: bool = True
    minimum_signal: float = 1e-12
    max_total_abs_delta_per_node: float | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if int(
            self.propagation_steps
        ) < 0:
            raise PrestressRedistributionError(
                "propagation_steps must be >= 0"
            )

        propagation_decay = _validate_finite(
            "propagation_decay",
            self.propagation_decay,
        )
        minimum_signal = _validate_finite(
            "minimum_signal",
            self.minimum_signal,
        )

        if not (
            0.0
            <= propagation_decay
            <= 1.0
        ):
            raise PrestressRedistributionError(
                "propagation_decay must lie in [0, 1]"
            )

        if minimum_signal < 0.0:
            raise PrestressRedistributionError(
                "minimum_signal must be nonnegative"
            )

        cap = self.max_total_abs_delta_per_node

        if cap is not None:
            cap = _validate_finite(
                "max_total_abs_delta_per_node",
                cap,
            )
            if cap < 0.0:
                raise PrestressRedistributionError(
                    "max_total_abs_delta_per_node must be nonnegative"
                )

        object.__setattr__(
            self,
            "propagation_steps",
            int(self.propagation_steps),
        )
        object.__setattr__(
            self,
            "propagation_decay",
            propagation_decay,
        )
        object.__setattr__(
            self,
            "minimum_signal",
            minimum_signal,
        )
        object.__setattr__(
            self,
            "max_total_abs_delta_per_node",
            cap,
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class PrestressContribution:
    step_index: int
    connection_id: str
    source_node_id: str
    target_node_id: str
    source_signal: float
    raw_transmitted: float
    accepted_delta: float
    capacity_limited: bool
    node_bound_limited: bool
    reserve_limited: bool
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class PrestressRedistributionResult:
    redistribution_id: str
    source_states: tuple[PrestressNodeState, ...]
    target_states: tuple[PrestressNodeState, ...]
    perturbations: tuple[PrestressPerturbation, ...]
    contributions: tuple[PrestressContribution, ...]
    node_delta_by_id: Mapping[str, float]
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "redistribution_id",
            _validate_id(
                "redistribution_id",
                self.redistribution_id,
            ),
        )
        object.__setattr__(
            self,
            "source_states",
            tuple(self.source_states),
        )
        object.__setattr__(
            self,
            "target_states",
            tuple(self.target_states),
        )
        object.__setattr__(
            self,
            "perturbations",
            tuple(self.perturbations),
        )
        object.__setattr__(
            self,
            "contributions",
            tuple(self.contributions),
        )
        object.__setattr__(
            self,
            "node_delta_by_id",
            MappingProxyType(
                dict(self.node_delta_by_id)
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly(self.metadata),
        )


def _index_nodes(
    nodes: Sequence[PrestressNodeState],
) -> dict[str, PrestressNodeState]:
    indexed: dict[str, PrestressNodeState] = {}

    for node in nodes:
        if node.node_id in indexed:
            raise PrestressRedistributionError(
                f"duplicate node_id: {node.node_id}"
            )
        indexed[node.node_id] = node

    return indexed


def _validate_connections(
    connections: Sequence[PrestressConnection],
    node_ids: set[str],
) -> None:
    seen = set()

    for connection in connections:
        if connection.connection_id in seen:
            raise PrestressRedistributionError(
                f"duplicate connection_id: {connection.connection_id}"
            )

        seen.add(connection.connection_id)

        if connection.source_node_id not in node_ids:
            raise PrestressRedistributionError(
                f"unknown source node: {connection.source_node_id}"
            )

        if connection.target_node_id not in node_ids:
            raise PrestressRedistributionError(
                f"unknown target node: {connection.target_node_id}"
            )


def _apply_node_delta(
    *,
    original: PrestressNodeState,
    current_value: float,
    requested_delta: float,
    cumulative_delta: float,
    config: PrestressRedistributionConfig,
) -> tuple[float, float, bool, bool]:
    requested_value = (
        current_value
        + requested_delta
    )

    bounded_value = _clamp(
        requested_value,
        original.min_prestress,
        original.max_prestress,
    )

    node_bound_limited = (
        bounded_value
        != requested_value
    )

    accepted_delta = (
        bounded_value
        - current_value
    )

    total_delta_limited = False

    cap = (
        config.max_total_abs_delta_per_node
    )

    if cap is not None:
        desired_total_delta = (
            cumulative_delta
            + accepted_delta
        )

        capped_total_delta = _clamp(
            desired_total_delta,
            -cap,
            cap,
        )

        if capped_total_delta != desired_total_delta:
            total_delta_limited = True

        accepted_delta = (
            capped_total_delta
            - cumulative_delta
        )

        bounded_value = (
            current_value
            + accepted_delta
        )

    return (
        bounded_value,
        accepted_delta,
        node_bound_limited,
        total_delta_limited,
    )


def redistribute_prestress(
    *,
    redistribution_id: str,
    nodes: Iterable[PrestressNodeState],
    connections: Iterable[PrestressConnection],
    perturbations: Iterable[PrestressPerturbation],
    config: PrestressRedistributionConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PrestressRedistributionResult:
    redistribution_id = _validate_id(
        "redistribution_id",
        redistribution_id,
    )

    node_items = tuple(nodes)

    if not node_items:
        raise PrestressRedistributionError(
            "nodes must not be empty"
        )

    connection_items = tuple(connections)
    perturbation_items = tuple(perturbations)

    actual_config = (
        config
        if config is not None
        else PrestressRedistributionConfig()
    )

    node_index = _index_nodes(
        node_items
    )

    node_ids = set(node_index)

    _validate_connections(
        connection_items,
        node_ids,
    )

    for perturbation in perturbation_items:
        if perturbation.node_id not in node_ids:
            raise PrestressRedistributionError(
                f"unknown perturbation node: {perturbation.node_id}"
            )

    current = {
        node.node_id: node.prestress
        for node in node_items
    }

    cumulative_delta = {
        node.node_id: 0.0
        for node in node_items
    }

    frontier = {
        node.node_id: 0.0
        for node in node_items
    }

    for perturbation in perturbation_items:
        node = node_index[
            perturbation.node_id
        ]

        (
            new_value,
            accepted_delta,
            _,
            _,
        ) = _apply_node_delta(
            original=node,
            current_value=current[node.node_id],
            requested_delta=perturbation.delta,
            cumulative_delta=cumulative_delta[node.node_id],
            config=actual_config,
        )

        current[node.node_id] = new_value
        cumulative_delta[node.node_id] += accepted_delta
        frontier[node.node_id] += accepted_delta

    sorted_connections = tuple(
        sorted(
            (
                connection
                for connection in connection_items
                if connection.enabled
            ),
            key=lambda item: (
                item.source_node_id,
                item.target_node_id,
                item.connection_id,
            ),
        )
    )

    contributions = []

    for step_index in range(
        1,
        actual_config.propagation_steps + 1,
    ):
        next_frontier = {
            node_id: 0.0
            for node_id in node_ids
        }

        for connection in sorted_connections:
            source_signal = frontier[
                connection.source_node_id
            ]

            if abs(
                source_signal
            ) < actual_config.minimum_signal:
                continue

            raw = (
                source_signal
                * connection.transfer_gain
                * connection.attenuation
                * actual_config.propagation_decay
            )

            capacity_limited = False

            if abs(raw) > connection.capacity:
                capacity_limited = True
                raw = (
                    connection.capacity
                    if raw >= 0.0
                    else -connection.capacity
                )

            target_node = node_index[
                connection.target_node_id
            ]

            reserve_limited = False
            transmitted = raw

            if actual_config.reserve_modulation:
                reserve_factor = target_node.reserve

                if reserve_factor < 1.0:
                    reserve_limited = True

                transmitted *= reserve_factor

            (
                new_value,
                accepted_delta,
                node_bound_limited,
                total_delta_limited,
            ) = _apply_node_delta(
                original=target_node,
                current_value=current[target_node.node_id],
                requested_delta=transmitted,
                cumulative_delta=cumulative_delta[target_node.node_id],
                config=actual_config,
            )

            current[
                target_node.node_id
            ] = new_value

            cumulative_delta[
                target_node.node_id
            ] += accepted_delta

            if abs(
                accepted_delta
            ) >= actual_config.minimum_signal:
                next_frontier[
                    target_node.node_id
                ] += accepted_delta

            contributions.append(
                PrestressContribution(
                    step_index=step_index,
                    connection_id=connection.connection_id,
                    source_node_id=connection.source_node_id,
                    target_node_id=connection.target_node_id,
                    source_signal=source_signal,
                    raw_transmitted=raw,
                    accepted_delta=accepted_delta,
                    capacity_limited=capacity_limited,
                    node_bound_limited=(
                        node_bound_limited
                        or total_delta_limited
                    ),
                    reserve_limited=reserve_limited,
                    metadata={
                        "schema_version": SCHEMA_VERSION,
                        "learning_applied": False,
                        "topology_modified": False,
                    },
                )
            )

        frontier = next_frontier

        if all(
            abs(signal)
            < actual_config.minimum_signal
            for signal in frontier.values()
        ):
            break

    target_states = tuple(
        PrestressNodeState(
            node_id=node.node_id,
            prestress=current[node.node_id],
            min_prestress=node.min_prestress,
            max_prestress=node.max_prestress,
            reserve=node.reserve,
            metadata=dict(node.metadata),
        )
        for node in node_items
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "redistribution_mode": "deterministic_explicit_graph",
        "source_node_count": len(node_items),
        "connection_count": len(connection_items),
        "perturbation_count": len(perturbation_items),
        "memory_mutated": False,
        "learning_applied": False,
        "topology_modified": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_truth_claimed": False,
        "causal_truth_inferred": False,
    }

    if metadata:
        merged_metadata.update(
            dict(metadata)
        )

    return PrestressRedistributionResult(
        redistribution_id=redistribution_id,
        source_states=node_items,
        target_states=target_states,
        perturbations=perturbation_items,
        contributions=tuple(contributions),
        node_delta_by_id={
            node_id: (
                current[node_id]
                - node_index[node_id].prestress
            )
            for node_id in sorted(node_ids)
        },
        metadata=merged_metadata,
    )


def prestress_redistribution_is_policy_free(
    result: PrestressRedistributionResult,
) -> bool:
    return (
        result.metadata.get(
            "action_selected"
        )
        is False
        and result.metadata.get(
            "policy_modified"
        )
        is False
        and result.metadata.get(
            "learning_applied"
        )
        is False
        and result.metadata.get(
            "topology_modified"
        )
        is False
    )


def prestress_signature(
    result: PrestressRedistributionResult,
) -> tuple[
    str,
    tuple[tuple[str, float], ...],
]:
    return (
        result.redistribution_id,
        tuple(
            (
                node.node_id,
                node.prestress,
            )
            for node in result.target_states
        ),
    )


def node_delta(
    result: PrestressRedistributionResult,
    node_id: str,
) -> float | None:
    return result.node_delta_by_id.get(
        node_id
    )


def target_state(
    result: PrestressRedistributionResult,
    node_id: str,
) -> PrestressNodeState | None:
    for node in result.target_states:
        if node.node_id == node_id:
            return node
    return None


__all__ = [
    "PrestressConnection",
    "PrestressContribution",
    "PrestressNodeState",
    "PrestressPerturbation",
    "PrestressRedistributionConfig",
    "PrestressRedistributionError",
    "PrestressRedistributionResult",
    "SCHEMA_VERSION",
    "node_delta",
    "prestress_redistribution_is_policy_free",
    "prestress_signature",
    "redistribute_prestress",
    "target_state",
]
