"""
State objects for the ROIF analytical layer.

This module defines immutable snapshots of a pre-stressed, multi-plane system.

The principal hierarchy is:

    NodeState
        Local state of one graph node.

    PlaneState
        Dynamic state of one functional plane.

    SystemState
        Complete system snapshot at a particular moment.

The objects in this module do not propagate cascades and do not choose
interventions. They only describe the state upon which future operators act.

A future ROIF operator will follow the pattern:

    next_state = operator.apply(current_state)

Using immutable snapshots provides several important properties:

1. Cascade history can safely store previous states.
2. Counterfactual simulations cannot silently mutate the observed state.
3. The order of non-commuting operators can be reconstructed.
4. Reproducibility and testing become considerably simpler.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ScalarMap: TypeAlias = Mapping[str, float]
MetadataMap: TypeAlias = Mapping[str, Any]

_EPSILON = 1e-12


class StateError(ValueError):
    """Raised when a ROIF state object is physically or structurally invalid."""


class NodeStatus(str, Enum):
    """
    Discrete functional condition of a node.

    The status is descriptive. It does not replace continuous quantities such
    as reserve, utilization, deficit or hysteresis.
    """

    QUIESCENT = "quiescent"
    ACTIVE = "active"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"
    RECOVERING = "recovering"
    DISABLED = "disabled"


class PlaneStatus(str, Enum):
    """Discrete activation condition of a functional plane."""

    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    RELAXING = "relaxing"
    SATURATED = "saturated"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class NodeState:
    """
    Immutable state of one node in the ROIF graph.

    Parameters
    ----------
    node_id:
        Unique identifier of the node.

    plane_ids:
        Functional planes acting on this node.

        A node may participate in several planes simultaneously. This is more
        general than assigning every node to exactly one plane.

    capacity:
        Current scalar capacity after all local modifiers have been applied.

        Direction-dependent capacity remains the responsibility of
        ``CapacityTensor``. This value represents the capacity relevant to the
        current state or service direction.

    load:
        Current signed load or effort.

        Absolute load is used for utilization and reserve calculations, while
        the sign remains available for tension/compression or directional
        interpretation.

    status:
        Current discrete node condition.

    activation_time:
        Time at which the node most recently became active.

        ``None`` means that no activation time has been recorded.

    last_update_time:
        Simulation time at which this state was produced.

    relaxation_time:
        Characteristic node-level relaxation time.

    hysteresis:
        Current memory / hysteresis value.

        The base state layer does not prescribe its physical meaning. A future
        plane kernel may interpret it as accumulated memory, delayed recovery,
        plasticity or another path-dependent variable.

    variables:
        Additional named continuous state variables.

        Examples:

        - temperature
        - metabolite concentration
        - vascular tone
        - local stiffness
        - damage
        - neural activation

    metadata:
        Non-numerical descriptive data not used directly by the mathematical
        core.
    """

    node_id: str
    plane_ids: tuple[str, ...] = ()
    capacity: float = 0.0
    load: float = 0.0
    status: NodeStatus = NodeStatus.QUIESCENT
    activation_time: float | None = None
    last_update_time: float = 0.0
    relaxation_time: float = 0.0
    hysteresis: float = 0.0
    variables: ScalarMap = field(default_factory=dict)
    metadata: MetadataMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_id = self._validate_identifier(self.node_id, name="node_id")
        plane_ids = self._validate_plane_ids(self.plane_ids)

        capacity = self._validate_non_negative_finite(
            self.capacity,
            name="capacity",
        )
        load = self._validate_finite(
            self.load,
            name="load",
        )
        last_update_time = self._validate_non_negative_finite(
            self.last_update_time,
            name="last_update_time",
        )
        relaxation_time = self._validate_non_negative_finite(
            self.relaxation_time,
            name="relaxation_time",
        )
        hysteresis = self._validate_finite(
            self.hysteresis,
            name="hysteresis",
        )

        activation_time: float | None

        if self.activation_time is None:
            activation_time = None
        else:
            activation_time = self._validate_non_negative_finite(
                self.activation_time,
                name="activation_time",
            )

            if activation_time > last_update_time + _EPSILON:
                raise StateError(
                    "activation_time cannot be later than last_update_time."
                )

        status = self._coerce_node_status(self.status)
        variables = self._freeze_scalar_map(
            self.variables,
            name="variables",
        )
        metadata = self._freeze_metadata(self.metadata)

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "plane_ids", plane_ids)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "load", load)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "activation_time", activation_time)
        object.__setattr__(self, "last_update_time", last_update_time)
        object.__setattr__(self, "relaxation_time", relaxation_time)
        object.__setattr__(self, "hysteresis", hysteresis)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "metadata", metadata)

    @property
    def absolute_load(self) -> float:
        """Absolute magnitude of the current node load."""

        return abs(self.load)

    @property
    def reserve(self) -> float:
        """
        Signed remaining reserve.

        Positive:
            Capacity remains available.

        Zero:
            Load exactly equals capacity.

        Negative:
            Capacity is exceeded.
        """

        return self.capacity - self.absolute_load

    @property
    def available_reserve(self) -> float:
        """Non-negative reserve still available for additional loading."""

        return max(self.reserve, 0.0)

    @property
    def deficit(self) -> float:
        """Positive amount by which absolute load exceeds capacity."""

        return max(-self.reserve, 0.0)

    @property
    def utilization(self) -> float:
        """
        Non-negative utilization ratio ``|load| / capacity``.

        Rules
        -----
        - capacity > 0:
            utilization = |load| / capacity

        - capacity == 0 and load == 0:
            utilization = 0

        - capacity == 0 and load != 0:
            utilization = infinity
        """

        if self.capacity > _EPSILON:
            return self.absolute_load / self.capacity

        if self.absolute_load <= _EPSILON:
            return 0.0

        return float("inf")

    @property
    def signed_utilization(self) -> float:
        """Signed utilization preserving the sign of the node load."""

        if self.capacity > _EPSILON:
            return self.load / self.capacity

        if self.absolute_load <= _EPSILON:
            return 0.0

        return float(np.copysign(np.inf, self.load))

    @property
    def is_loaded(self) -> bool:
        """Whether the node currently carries a non-negligible load."""

        return self.absolute_load > _EPSILON

    @property
    def is_critical(self) -> bool:
        """Whether the node has reached or exceeded its current capacity."""

        return self.utilization >= 1.0

    @property
    def is_exhausted(self) -> bool:
        """Whether the node load strictly exceeds its current capacity."""

        return self.deficit > _EPSILON

    @property
    def is_available(self) -> bool:
        """Whether the node has non-zero capacity and is not disabled."""

        return (
            self.capacity > _EPSILON
            and self.status is not NodeStatus.DISABLED
        )

    @property
    def active_duration(self) -> float:
        """
        Time elapsed since the latest recorded activation.

        Returns zero when the node has no activation timestamp.
        """

        if self.activation_time is None:
            return 0.0

        return max(self.last_update_time - self.activation_time, 0.0)

    def belongs_to(self, plane_id: str) -> bool:
        """Return whether the node participates in the requested plane."""

        normalized = self._validate_identifier(
            plane_id,
            name="plane_id",
        )
        return normalized in self.plane_ids

    def variable(self, name: str, default: float | None = None) -> float:
        """
        Read an additional continuous variable.

        Raises
        ------
        StateError
            If the variable does not exist and no default was supplied.
        """

        key = self._validate_identifier(name, name="variable name")

        if key in self.variables:
            return float(self.variables[key])

        if default is not None:
            return self._validate_finite(
                default,
                name=f"default for variable {key!r}",
            )

        raise StateError(
            f"Node {self.node_id!r} has no variable named {key!r}."
        )

    def with_load(
        self,
        load: float,
        *,
        time: float | None = None,
        status: NodeStatus | str | None = None,
    ) -> NodeState:
        """Return a new snapshot with an updated signed load."""

        updates: dict[str, Any] = {
            "load": load,
        }

        if time is not None:
            updates["last_update_time"] = time

        if status is not None:
            updates["status"] = status

        return replace(self, **updates)

    def with_capacity(
        self,
        capacity: float,
        *,
        time: float | None = None,
    ) -> NodeState:
        """Return a new snapshot with an updated current capacity."""

        updates: dict[str, Any] = {
            "capacity": capacity,
        }

        if time is not None:
            updates["last_update_time"] = time

        return replace(self, **updates)

    def with_hysteresis(
        self,
        hysteresis: float,
        *,
        time: float | None = None,
    ) -> NodeState:
        """Return a new snapshot with an updated memory value."""

        updates: dict[str, Any] = {
            "hysteresis": hysteresis,
        }

        if time is not None:
            updates["last_update_time"] = time

        return replace(self, **updates)

    def with_variable(
        self,
        name: str,
        value: float,
        *,
        time: float | None = None,
    ) -> NodeState:
        """Return a new snapshot with one continuous variable updated."""

        key = self._validate_identifier(name, name="variable name")
        numeric_value = self._validate_finite(
            value,
            name=f"variable {key!r}",
        )

        variables = dict(self.variables)
        variables[key] = numeric_value

        updates: dict[str, Any] = {
            "variables": variables,
        }

        if time is not None:
            updates["last_update_time"] = time

        return replace(self, **updates)

    def activate(
        self,
        time: float,
        *,
        status: NodeStatus | str = NodeStatus.ACTIVE,
    ) -> NodeState:
        """Return a new activated node snapshot."""

        activation_time = self._validate_non_negative_finite(
            time,
            name="time",
        )

        if activation_time + _EPSILON < self.last_update_time:
            raise StateError(
                "Activation time cannot be earlier than last_update_time."
            )

        return replace(
            self,
            activation_time=activation_time,
            last_update_time=activation_time,
            status=status,
        )

    def deactivate(
        self,
        time: float,
        *,
        status: NodeStatus | str = NodeStatus.QUIESCENT,
    ) -> NodeState:
        """Return a new inactive node snapshot."""

        update_time = self._validate_non_negative_finite(
            time,
            name="time",
        )

        if update_time + _EPSILON < self.last_update_time:
            raise StateError(
                "Deactivation time cannot be earlier than last_update_time."
            )

        return replace(
            self,
            activation_time=None,
            last_update_time=update_time,
            status=status,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "node_id": self.node_id,
            "plane_ids": list(self.plane_ids),
            "capacity": self.capacity,
            "load": self.load,
            "absolute_load": self.absolute_load,
            "reserve": self.reserve,
            "available_reserve": self.available_reserve,
            "deficit": self.deficit,
            "utilization": self.utilization,
            "signed_utilization": self.signed_utilization,
            "status": self.status.value,
            "activation_time": self.activation_time,
            "last_update_time": self.last_update_time,
            "relaxation_time": self.relaxation_time,
            "hysteresis": self.hysteresis,
            "variables": dict(self.variables),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _validate_plane_ids(
        plane_ids: Iterable[str],
    ) -> tuple[str, ...]:
        try:
            normalized = tuple(
                NodeState._validate_identifier(
                    plane_id,
                    name="plane_id",
                )
                for plane_id in plane_ids
            )
        except TypeError as error:
            raise StateError(
                "plane_ids must be an iterable of strings."
            ) from error

        if len(normalized) != len(set(normalized)):
            raise StateError("plane_ids cannot contain duplicates.")

        return normalized

    @staticmethod
    def _coerce_node_status(
        status: NodeStatus | str,
    ) -> NodeStatus:
        if isinstance(status, NodeStatus):
            return status

        try:
            return NodeStatus(status)
        except (TypeError, ValueError) as error:
            valid = ", ".join(item.value for item in NodeStatus)
            raise StateError(
                f"Unknown node status {status!r}. Expected one of: {valid}."
            ) from error

    @staticmethod
    def _validate_identifier(
        value: str,
        *,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise StateError(f"{name} must be a string.")

        normalized = value.strip()

        if not normalized:
            raise StateError(f"{name} cannot be empty.")

        return normalized

    @staticmethod
    def _validate_finite(
        value: float,
        *,
        name: str,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise StateError(f"{name} must be numeric.") from error

        if not isfinite(result):
            raise StateError(f"{name} must be finite.")

        return result

    @staticmethod
    def _validate_non_negative_finite(
        value: float,
        *,
        name: str,
    ) -> float:
        result = NodeState._validate_finite(
            value,
            name=name,
        )

        if result < 0.0:
            raise StateError(
                f"{name} must be non-negative, received {result}."
            )

        return result

    @staticmethod
    def _freeze_scalar_map(
        values: ScalarMap,
        *,
        name: str,
    ) -> ScalarMap:
        if not isinstance(values, Mapping):
            raise StateError(f"{name} must be a mapping.")

        frozen: dict[str, float] = {}

        for key, value in values.items():
            normalized_key = NodeState._validate_identifier(
                key,
                name=f"{name} key",
            )
            frozen[normalized_key] = NodeState._validate_finite(
                value,
                name=f"{name}[{normalized_key!r}]",
            )

        return MappingProxyType(frozen)

    @staticmethod
    def _freeze_metadata(
        metadata: MetadataMap,
    ) -> MetadataMap:
        if not isinstance(metadata, Mapping):
            raise StateError("metadata must be a mapping.")

        return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True)
class PlaneState:
    """
    Immutable dynamic state of one functional plane.

    ``PlaneState`` stores the current activation and memory of a plane.
    The actual transformation law will be implemented later by ``PlaneKernel``.

    Parameters
    ----------
    plane_id:
        Unique functional-plane identifier.

    status:
        Current activation condition.

    activation:
        Continuous activation level.

        The state layer requires a non-negative value but does not force an
        upper bound of one. Values above one may represent amplification or
        saturation, depending on the future plane kernel.

    activation_time:
        Time at which the current activation episode began.

    last_update_time:
        Time at which this snapshot was produced.

    characteristic_time:
        Main response timescale of the plane.

    relaxation_time:
        Characteristic decay or recovery timescale.

    hysteresis:
        Current path-dependent memory.

    variables:
        Additional continuous plane-level variables.
    """

    plane_id: str
    status: PlaneStatus = PlaneStatus.INACTIVE
    activation: float = 0.0
    activation_time: float | None = None
    last_update_time: float = 0.0
    characteristic_time: float = 1.0
    relaxation_time: float = 1.0
    hysteresis: float = 0.0
    variables: ScalarMap = field(default_factory=dict)
    metadata: MetadataMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        plane_id = NodeState._validate_identifier(
            self.plane_id,
            name="plane_id",
        )
        status = self._coerce_plane_status(self.status)
        activation = NodeState._validate_non_negative_finite(
            self.activation,
            name="activation",
        )
        last_update_time = NodeState._validate_non_negative_finite(
            self.last_update_time,
            name="last_update_time",
        )
        characteristic_time = self._validate_positive_finite(
            self.characteristic_time,
            name="characteristic_time",
        )
        relaxation_time = self._validate_positive_finite(
            self.relaxation_time,
            name="relaxation_time",
        )
        hysteresis = NodeState._validate_finite(
            self.hysteresis,
            name="hysteresis",
        )

        activation_time: float | None

        if self.activation_time is None:
            activation_time = None
        else:
            activation_time = NodeState._validate_non_negative_finite(
                self.activation_time,
                name="activation_time",
            )

            if activation_time > last_update_time + _EPSILON:
                raise StateError(
                    "activation_time cannot be later than last_update_time."
                )

        variables = NodeState._freeze_scalar_map(
            self.variables,
            name="variables",
        )
        metadata = NodeState._freeze_metadata(self.metadata)

        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "activation_time", activation_time)
        object.__setattr__(self, "last_update_time", last_update_time)
        object.__setattr__(
            self,
            "characteristic_time",
            characteristic_time,
        )
        object.__setattr__(
            self,
            "relaxation_time",
            relaxation_time,
        )
        object.__setattr__(self, "hysteresis", hysteresis)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "metadata", metadata)

    @property
    def is_active(self) -> bool:
        """Whether the plane currently has non-zero activation."""

        return (
            self.activation > _EPSILON
            and self.status is not PlaneStatus.DISABLED
        )

    @property
    def active_duration(self) -> float:
        """Time elapsed since the current activation began."""

        if self.activation_time is None:
            return 0.0

        return max(self.last_update_time - self.activation_time, 0.0)

    def with_activation(
        self,
        activation: float,
        *,
        time: float | None = None,
        status: PlaneStatus | str | None = None,
    ) -> PlaneState:
        """Return a new plane snapshot with updated activation."""

        updates: dict[str, Any] = {
            "activation": activation,
        }

        if time is not None:
            updates["last_update_time"] = time

        if status is not None:
            updates["status"] = status

        return replace(self, **updates)

    def with_hysteresis(
        self,
        hysteresis: float,
        *,
        time: float | None = None,
    ) -> PlaneState:
        """Return a new plane snapshot with updated memory."""

        updates: dict[str, Any] = {
            "hysteresis": hysteresis,
        }

        if time is not None:
            updates["last_update_time"] = time

        return replace(self, **updates)

    def with_variable(
        self,
        name: str,
        value: float,
        *,
        time: float | None = None,
    ) -> PlaneState:
        """Return a new plane snapshot with one variable updated."""

        key = NodeState._validate_identifier(
            name,
            name="variable name",
        )
        numeric_value = NodeState._validate_finite(
            value,
            name=f"variable {key!r}",
        )

        variables = dict(self.variables)
        variables[key] = numeric_value

        updates: dict[str, Any] = {
            "variables": variables,
        }

        if time is not None:
            updates["last_update_time"] = time

        return replace(self, **updates)

    def activate(
        self,
        time: float,
        *,
        activation: float = 1.0,
    ) -> PlaneState:
        """Begin or restart a plane activation episode."""

        activation_time = NodeState._validate_non_negative_finite(
            time,
            name="time",
        )

        if activation_time + _EPSILON < self.last_update_time:
            raise StateError(
                "Activation time cannot be earlier than last_update_time."
            )

        return replace(
            self,
            status=PlaneStatus.ACTIVE,
            activation=activation,
            activation_time=activation_time,
            last_update_time=activation_time,
        )

    def begin_relaxation(
        self,
        time: float,
    ) -> PlaneState:
        """Mark the plane as entering its relaxation phase."""

        update_time = NodeState._validate_non_negative_finite(
            time,
            name="time",
        )

        if update_time + _EPSILON < self.last_update_time:
            raise StateError(
                "Relaxation time cannot be earlier than last_update_time."
            )

        return replace(
            self,
            status=PlaneStatus.RELAXING,
            last_update_time=update_time,
        )

    def deactivate(
        self,
        time: float,
    ) -> PlaneState:
        """Return a fully inactive plane snapshot."""

        update_time = NodeState._validate_non_negative_finite(
            time,
            name="time",
        )

        if update_time + _EPSILON < self.last_update_time:
            raise StateError(
                "Deactivation time cannot be earlier than last_update_time."
            )

        return replace(
            self,
            status=PlaneStatus.INACTIVE,
            activation=0.0,
            activation_time=None,
            last_update_time=update_time,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "plane_id": self.plane_id,
            "status": self.status.value,
            "activation": self.activation,
            "activation_time": self.activation_time,
            "last_update_time": self.last_update_time,
            "characteristic_time": self.characteristic_time,
            "relaxation_time": self.relaxation_time,
            "hysteresis": self.hysteresis,
            "variables": dict(self.variables),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _coerce_plane_status(
        status: PlaneStatus | str,
    ) -> PlaneStatus:
        if isinstance(status, PlaneStatus):
            return status

        try:
            return PlaneStatus(status)
        except (TypeError, ValueError) as error:
            valid = ", ".join(item.value for item in PlaneStatus)
            raise StateError(
                f"Unknown plane status {status!r}. Expected one of: {valid}."
            ) from error

    @staticmethod
    def _validate_positive_finite(
        value: float,
        *,
        name: str,
    ) -> float:
        result = NodeState._validate_finite(
            value,
            name=name,
        )

        if result <= 0.0:
            raise StateError(
                f"{name} must be positive, received {result}."
            )

        return result


@dataclass(frozen=True, slots=True)
class SystemState:
    """
    Immutable snapshot of the complete ROIF system.

    A system state contains:

    - current simulation time;
    - all node states;
    - all functional-plane states;
    - optional global continuous variables;
    - optional descriptive metadata.

    Future cascade solvers should produce a new ``SystemState`` for every
    accepted event or operator application.
    """

    time: float = 0.0
    nodes: Mapping[str, NodeState] = field(default_factory=dict)
    planes: Mapping[str, PlaneState] = field(default_factory=dict)
    variables: ScalarMap = field(default_factory=dict)
    metadata: MetadataMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = NodeState._validate_non_negative_finite(
            self.time,
            name="time",
        )
        nodes = self._freeze_nodes(self.nodes)
        planes = self._freeze_planes(self.planes)
        variables = NodeState._freeze_scalar_map(
            self.variables,
            name="variables",
        )
        metadata = NodeState._freeze_metadata(self.metadata)

        self._validate_references(
            nodes=nodes,
            planes=planes,
        )
        self._validate_component_times(
            system_time=time,
            nodes=nodes,
            planes=planes,
        )

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "planes", planes)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "metadata", metadata)

    @property
    def node_count(self) -> int:
        """Number of nodes in the system."""

        return len(self.nodes)

    @property
    def plane_count(self) -> int:
        """Number of functional planes in the system."""

        return len(self.planes)

    @property
    def total_capacity(self) -> float:
        """Sum of current scalar capacities over all nodes."""

        return float(sum(node.capacity for node in self.nodes.values()))

    @property
    def total_absolute_load(self) -> float:
        """Sum of absolute node loads."""

        return float(
            sum(node.absolute_load for node in self.nodes.values())
        )

    @property
    def total_available_reserve(self) -> float:
        """Sum of non-negative available reserves."""

        return float(
            sum(node.available_reserve for node in self.nodes.values())
        )

    @property
    def total_deficit(self) -> float:
        """Sum of node deficits."""

        return float(
            sum(node.deficit for node in self.nodes.values())
        )

    @property
    def critical_node_ids(self) -> tuple[str, ...]:
        """Identifiers of nodes at or above full utilization."""

        return tuple(
            node_id
            for node_id, node in self.nodes.items()
            if node.is_critical
        )

    @property
    def exhausted_node_ids(self) -> tuple[str, ...]:
        """Identifiers of nodes whose load exceeds capacity."""

        return tuple(
            node_id
            for node_id, node in self.nodes.items()
            if node.is_exhausted
        )

    @property
    def active_plane_ids(self) -> tuple[str, ...]:
        """Identifiers of currently active planes."""

        return tuple(
            plane_id
            for plane_id, plane in self.planes.items()
            if plane.is_active
        )

    @property
    def d_fast_node_id(self) -> str | None:
        """
        Node with the greatest instantaneous utilization.

        This property is a snapshot-level convenience definition.

        Later, event history may define D_fast more strictly as the first node
        that lost reserve in time. That temporal definition belongs to
        ``CascadeHistory`` rather than to a single state snapshot.
        """

        if not self.nodes:
            return None

        return max(
            self.nodes,
            key=lambda node_id: self.nodes[node_id].utilization,
        )

    def node(self, node_id: str) -> NodeState:
        """Return one node state by identifier."""

        key = NodeState._validate_identifier(
            node_id,
            name="node_id",
        )

        try:
            return self.nodes[key]
        except KeyError as error:
            raise StateError(
                f"Unknown node_id {key!r}."
            ) from error

    def plane(self, plane_id: str) -> PlaneState:
        """Return one plane state by identifier."""

        key = NodeState._validate_identifier(
            plane_id,
            name="plane_id",
        )

        try:
            return self.planes[key]
        except KeyError as error:
            raise StateError(
                f"Unknown plane_id {key!r}."
            ) from error

    def nodes_in_plane(
        self,
        plane_id: str,
    ) -> tuple[NodeState, ...]:
        """Return all nodes participating in a functional plane."""

        key = NodeState._validate_identifier(
            plane_id,
            name="plane_id",
        )

        if key not in self.planes:
            raise StateError(
                f"Unknown plane_id {key!r}."
            )

        return tuple(
            node
            for node in self.nodes.values()
            if key in node.plane_ids
        )

    def with_node(
        self,
        node: NodeState,
    ) -> SystemState:
        """Return a new system snapshot containing the supplied node state."""

        if not isinstance(node, NodeState):
            raise StateError("node must be a NodeState instance.")

        nodes = dict(self.nodes)
        nodes[node.node_id] = node

        new_time = max(
            self.time,
            node.last_update_time,
        )

        return replace(
            self,
            time=new_time,
            nodes=nodes,
        )

    def without_node(
        self,
        node_id: str,
    ) -> SystemState:
        """Return a new system snapshot without the requested node."""

        key = NodeState._validate_identifier(
            node_id,
            name="node_id",
        )

        if key not in self.nodes:
            raise StateError(
                f"Unknown node_id {key!r}."
            )

        nodes = dict(self.nodes)
        del nodes[key]

        return replace(
            self,
            nodes=nodes,
        )

    def with_plane(
        self,
        plane: PlaneState,
    ) -> SystemState:
        """Return a new system snapshot containing the supplied plane state."""

        if not isinstance(plane, PlaneState):
            raise StateError("plane must be a PlaneState instance.")

        planes = dict(self.planes)
        planes[plane.plane_id] = plane

        new_time = max(
            self.time,
            plane.last_update_time,
        )

        return replace(
            self,
            time=new_time,
            planes=planes,
        )

    def without_plane(
        self,
        plane_id: str,
        *,
        detach_nodes: bool = False,
    ) -> SystemState:
        """
        Return a system snapshot without the requested plane.

        Parameters
        ----------
        detach_nodes:
            If ``False``, removal is rejected while nodes reference the plane.

            If ``True``, the plane identifier is also removed from every
            affected node.
        """

        key = NodeState._validate_identifier(
            plane_id,
            name="plane_id",
        )

        if key not in self.planes:
            raise StateError(
                f"Unknown plane_id {key!r}."
            )

        referencing_nodes = [
            node
            for node in self.nodes.values()
            if key in node.plane_ids
        ]

        if referencing_nodes and not detach_nodes:
            node_ids = ", ".join(
                node.node_id for node in referencing_nodes
            )
            raise StateError(
                f"Cannot remove plane {key!r}; it is referenced by nodes: "
                f"{node_ids}."
            )

        planes = dict(self.planes)
        del planes[key]

        nodes = dict(self.nodes)

        if detach_nodes:
            for node in referencing_nodes:
                nodes[node.node_id] = replace(
                    node,
                    plane_ids=tuple(
                        item
                        for item in node.plane_ids
                        if item != key
                    ),
                )

        return replace(
            self,
            nodes=nodes,
            planes=planes,
        )

    def with_variable(
        self,
        name: str,
        value: float,
    ) -> SystemState:
        """Return a new system snapshot with one global variable updated."""

        key = NodeState._validate_identifier(
            name,
            name="variable name",
        )
        numeric_value = NodeState._validate_finite(
            value,
            name=f"variable {key!r}",
        )

        variables = dict(self.variables)
        variables[key] = numeric_value

        return replace(
            self,
            variables=variables,
        )

    def at_time(
        self,
        time: float,
    ) -> SystemState:
        """
        Return the same structural state at a later simulation time.

        This method does not automatically relax nodes or planes. Temporal
        evolution belongs to future operators and plane kernels.
        """

        new_time = NodeState._validate_non_negative_finite(
            time,
            name="time",
        )

        if new_time + _EPSILON < self.time:
            raise StateError(
                "System time cannot move backwards."
            )

        return replace(
            self,
            time=new_time,
        )

    def node_capacity_vector(
        self,
        node_ids: Iterable[str] | None = None,
    ) -> FloatArray:
        """Return node capacities as a read-only NumPy vector."""

        selected = self._resolve_node_ids(node_ids)

        return self._readonly_array(
            self.nodes[node_id].capacity
            for node_id in selected
        )

    def node_load_vector(
        self,
        node_ids: Iterable[str] | None = None,
    ) -> FloatArray:
        """Return signed node loads as a read-only NumPy vector."""

        selected = self._resolve_node_ids(node_ids)

        return self._readonly_array(
            self.nodes[node_id].load
            for node_id in selected
        )

    def node_reserve_vector(
        self,
        node_ids: Iterable[str] | None = None,
    ) -> FloatArray:
        """Return signed node reserves as a read-only NumPy vector."""

        selected = self._resolve_node_ids(node_ids)

        return self._readonly_array(
            self.nodes[node_id].reserve
            for node_id in selected
        )

    def node_utilization_vector(
        self,
        node_ids: Iterable[str] | None = None,
    ) -> FloatArray:
        """Return node utilizations as a read-only NumPy vector."""

        selected = self._resolve_node_ids(node_ids)

        return self._readonly_array(
            self.nodes[node_id].utilization
            for node_id in selected
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "time": self.time,
            "nodes": {
                node_id: node.as_dict()
                for node_id, node in self.nodes.items()
            },
            "planes": {
                plane_id: plane.as_dict()
                for plane_id, plane in self.planes.items()
            },
            "variables": dict(self.variables),
            "metadata": dict(self.metadata),
            "summary": {
                "node_count": self.node_count,
                "plane_count": self.plane_count,
                "total_capacity": self.total_capacity,
                "total_absolute_load": self.total_absolute_load,
                "total_available_reserve": self.total_available_reserve,
                "total_deficit": self.total_deficit,
                "critical_node_ids": list(self.critical_node_ids),
                "exhausted_node_ids": list(self.exhausted_node_ids),
                "active_plane_ids": list(self.active_plane_ids),
                "d_fast_node_id": self.d_fast_node_id,
            },
        }

    def _resolve_node_ids(
        self,
        node_ids: Iterable[str] | None,
    ) -> tuple[str, ...]:
        if node_ids is None:
            return tuple(self.nodes.keys())

        resolved = tuple(
            NodeState._validate_identifier(
                node_id,
                name="node_id",
            )
            for node_id in node_ids
        )

        missing = [
            node_id
            for node_id in resolved
            if node_id not in self.nodes
        ]

        if missing:
            raise StateError(
                "Unknown node identifiers: "
                + ", ".join(repr(item) for item in missing)
                + "."
            )

        return resolved

    @staticmethod
    def _freeze_nodes(
        nodes: Mapping[str, NodeState],
    ) -> Mapping[str, NodeState]:
        if not isinstance(nodes, Mapping):
            raise StateError("nodes must be a mapping.")

        frozen: dict[str, NodeState] = {}

        for key, node in nodes.items():
            normalized_key = NodeState._validate_identifier(
                key,
                name="node mapping key",
            )

            if not isinstance(node, NodeState):
                raise StateError(
                    f"nodes[{normalized_key!r}] must be a NodeState."
                )

            if normalized_key != node.node_id:
                raise StateError(
                    f"Node mapping key {normalized_key!r} does not match "
                    f"NodeState.node_id {node.node_id!r}."
                )

            frozen[normalized_key] = node

        return MappingProxyType(frozen)

    @staticmethod
    def _freeze_planes(
        planes: Mapping[str, PlaneState],
    ) -> Mapping[str, PlaneState]:
        if not isinstance(planes, Mapping):
            raise StateError("planes must be a mapping.")

        frozen: dict[str, PlaneState] = {}

        for key, plane in planes.items():
            normalized_key = NodeState._validate_identifier(
                key,
                name="plane mapping key",
            )

            if not isinstance(plane, PlaneState):
                raise StateError(
                    f"planes[{normalized_key!r}] must be a PlaneState."
                )

            if normalized_key != plane.plane_id:
                raise StateError(
                    f"Plane mapping key {normalized_key!r} does not match "
                    f"PlaneState.plane_id {plane.plane_id!r}."
                )

            frozen[normalized_key] = plane

        return MappingProxyType(frozen)

    @staticmethod
    def _validate_references(
        *,
        nodes: Mapping[str, NodeState],
        planes: Mapping[str, PlaneState],
    ) -> None:
        for node in nodes.values():
            missing_planes = [
                plane_id
                for plane_id in node.plane_ids
                if plane_id not in planes
            ]

            if missing_planes:
                raise StateError(
                    f"Node {node.node_id!r} references unknown planes: "
                    + ", ".join(repr(item) for item in missing_planes)
                    + "."
                )

    @staticmethod
    def _validate_component_times(
        *,
        system_time: float,
        nodes: Mapping[str, NodeState],
        planes: Mapping[str, PlaneState],
    ) -> None:
        for node in nodes.values():
            if node.last_update_time > system_time + _EPSILON:
                raise StateError(
                    f"Node {node.node_id!r} has last_update_time "
                    f"{node.last_update_time}, which is later than system "
                    f"time {system_time}."
                )

        for plane in planes.values():
            if plane.last_update_time > system_time + _EPSILON:
                raise StateError(
                    f"Plane {plane.plane_id!r} has last_update_time "
                    f"{plane.last_update_time}, which is later than system "
                    f"time {system_time}."
                )

    @staticmethod
    def _readonly_array(
        values: Iterable[float],
    ) -> FloatArray:
        result = np.asarray(
            list(values),
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result