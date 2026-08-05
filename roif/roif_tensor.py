"""
ROIF Engine
State-Dependent Capacity Tensor

This module assembles the universal ROIF representation into a numerical
operator that answers:

    How much influence can each functional channel transmit to every other
    channel in the current state of the system?

The tensor is domain-independent. A channel may describe:

- a muscle fascicle, nerve pathway, vessel, ligament, or organ function;
- a steel cable, concrete section, timber fibre direction, fluid pathway,
  electrical branch, or software component;
- a memory process, information channel, behavioural pattern, or control loop.

Mathematical role
-----------------

For a directed source channel j and target channel i, the effective coupling is
represented as a product of state-dependent factors:

    W[i, j] =
        base_transmission
        * operator_gain
        * plane_activity
        * source_capacity
        * source_availability
        * target_acceptance
        * geometry
        * vector_alignment
        * prestress
        * material
        * history
        * confidence

The factors inside one connection are multiplicative. Parallel operators may
then be aggregated additively, multiplicatively, by maximum, or by a custom
future adapter.

The module stores both:

1. plane-resolved tensor slices ``T[p, i, j]``;
2. the aggregated cascade operator ``W[i, j]``.

The current implementation is a reduced-order tensor layer. It does not yet
perform full cascade propagation, root detection, or counterfactual search.
Those responsibilities belong to:

- ``roif_cascade.py``;
- ``roif_root_detector.py``;
- ``roif_counterfactual.py``;
- ``roif_solver.py``.

This is a research-prototype computation layer and not a clinical decision
system.

Author:
    Architect (Dctr Pain)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from .roif_entities import (
    FunctionalChannel,
    InfluenceOperator,
    InfluencePlane,
    ROIFSystem,
    Vector3,
)
from .roif_influence import (
    InfluenceContext,
    effective_operator_gain,
    plane_activity,
)
from .roif_materials import (
    MaterialModel,
    material_plane_factors,
)


class ROIFTensorError(ValueError):
    """Raised when a ROIF tensor object or operation violates an invariant."""


class TensorAggregation(str, Enum):
    """How parallel operator contributions are combined."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


class TensorNormalization(str, Enum):
    """Optional normalization applied after tensor assembly."""

    NONE = "none"
    ROW = "row"
    COLUMN = "column"
    SPECTRAL = "spectral"
    GLOBAL_MAX = "global_max"


class CapacityFactorMode(str, Enum):
    """How reserve contributes to transmissible capacity."""

    CLAMPED_RESERVE = "clamped_reserve"
    POSITIVE_RESERVE = "positive_reserve"
    EXPONENTIAL_RESERVE = "exponential_reserve"
    UTILIZATION_COMPLEMENT = "utilization_complement"


class SelfCouplingMode(str, Enum):
    """How diagonal tensor entries are handled."""

    NONE = "none"
    IDENTITY = "identity"
    RETENTION = "retention"
    EXPLICIT_ONLY = "explicit_only"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFTensorError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFTensorError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ROIFTensorError(f"{name} must be finite.")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ROIFTensorError(f"{name} cannot be negative.")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ROIFTensorError(f"{name} must be positive.")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ROIFTensorError(f"{name} must be within [0, 1].")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFTensorError(f"{name} must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFTensorError("metadata must be a mapping.")
    return MappingProxyType(dict(value))


def _readonly_array(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ROIFTensorError(f"{name} must be numeric.") from exc
    if array.ndim != ndim:
        raise ROIFTensorError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(array)):
        raise ROIFTensorError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool_array(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim != ndim:
        raise ROIFTensorError(f"{name} must have {ndim} dimensions.")
    result = np.array(array, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


def _safe_product(values: Sequence[float]) -> float:
    result = 1.0
    for value in values:
        result *= _finite(value, "factor")
    return result


@dataclass(frozen=True, slots=True)
class TensorBuildConfig:
    """Configuration for constructing a state-dependent Capacity Tensor."""

    aggregation: TensorAggregation = TensorAggregation.ADDITIVE
    normalization: TensorNormalization = TensorNormalization.NONE
    capacity_factor_mode: CapacityFactorMode = (
        CapacityFactorMode.CLAMPED_RESERVE
    )
    self_coupling_mode: SelfCouplingMode = SelfCouplingMode.RETENTION

    tensor_floor: float = 0.0
    tensor_ceiling: float = 1.0
    reserve_floor: float = 1e-9
    direction_floor: float = 0.0
    material_floor: float = 0.0

    default_base_transmission: float = 1.0
    default_prestress_factor: float = 1.0
    default_confidence: float = 1.0
    self_retention: float = 0.0
    spectral_target: float = 0.95

    source_reserve_exponent: float = 1.0
    target_reserve_exponent: float = 0.0
    availability_exponent: float = 1.0
    geometry_exponent: float = 1.0
    material_exponent: float = 1.0
    history_exponent: float = 1.0

    state_dependent: bool = True
    include_inactive_operators: bool = False
    include_inactive_planes: bool = False
    allow_negative_operator_gain: bool = False
    clip_final_tensor: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        enum_fields = (
            ("aggregation", self.aggregation, TensorAggregation),
            ("normalization", self.normalization, TensorNormalization),
            (
                "capacity_factor_mode",
                self.capacity_factor_mode,
                CapacityFactorMode,
            ),
            (
                "self_coupling_mode",
                self.self_coupling_mode,
                SelfCouplingMode,
            ),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise ROIFTensorError(
                    f"{name} must be {enum_type.__name__}."
                )

        for name in (
            "tensor_floor",
            "reserve_floor",
            "direction_floor",
            "material_floor",
            "default_base_transmission",
            "default_prestress_factor",
            "default_confidence",
            "self_retention",
            "source_reserve_exponent",
            "target_reserve_exponent",
            "availability_exponent",
            "geometry_exponent",
            "material_exponent",
            "history_exponent",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )

        object.__setattr__(
            self,
            "tensor_ceiling",
            _positive(self.tensor_ceiling, "tensor_ceiling"),
        )
        if self.tensor_floor > self.tensor_ceiling:
            raise ROIFTensorError(
                "tensor_floor cannot exceed tensor_ceiling."
            )
        object.__setattr__(
            self,
            "spectral_target",
            _positive(self.spectral_target, "spectral_target"),
        )
        object.__setattr__(
            self,
            "default_confidence",
            _unit(self.default_confidence, "default_confidence"),
        )

        for name in (
            "state_dependent",
            "include_inactive_operators",
            "include_inactive_planes",
            "allow_negative_operator_gain",
            "clip_final_tensor",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ROIFTensorError(f"{name} must be bool.")

        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ChannelTensorState:
    """Numerical tensor-facing state of one functional channel."""

    channel_id: str
    reserve_factor: float
    availability_factor: float
    geometry_factor: float
    history_factor: float
    material_factor: float
    acceptance_factor: float
    effective_output: float
    direction: Vector3
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_id",
            _text(self.channel_id, "channel_id"),
        )
        for name in (
            "reserve_factor",
            "availability_factor",
            "geometry_factor",
            "history_factor",
            "material_factor",
            "acceptance_factor",
            "effective_output",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        if not isinstance(self.direction, Vector3):
            raise ROIFTensorError("direction must be Vector3.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TensorEntryFactors:
    """Auditable multiplicative factors for one directed tensor entry."""

    source_id: str
    target_id: str
    operator_id: str
    plane_id: str

    base_transmission: float
    operator_gain: float
    plane_factor: float
    source_capacity_factor: float
    source_availability_factor: float
    target_acceptance_factor: float
    geometry_factor: float
    vector_alignment_factor: float
    prestress_factor: float
    material_factor: float
    history_factor: float
    confidence_factor: float

    raw_value: float
    final_value: float
    active: bool = True
    reason: str = "active"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "target_id",
            "operator_id",
            "plane_id",
            "reason",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name),
            )
        numeric = (
            "base_transmission",
            "operator_gain",
            "plane_factor",
            "source_capacity_factor",
            "source_availability_factor",
            "target_acceptance_factor",
            "geometry_factor",
            "vector_alignment_factor",
            "prestress_factor",
            "material_factor",
            "history_factor",
            "confidence_factor",
            "raw_value",
            "final_value",
        )
        for name in numeric:
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        if not isinstance(self.active, bool):
            raise ROIFTensorError("active must be bool.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def multiplicative_product(self) -> float:
        return _safe_product(
            (
                self.base_transmission,
                self.operator_gain,
                self.plane_factor,
                self.source_capacity_factor,
                self.source_availability_factor,
                self.target_acceptance_factor,
                self.geometry_factor,
                self.vector_alignment_factor,
                self.prestress_factor,
                self.material_factor,
                self.history_factor,
                self.confidence_factor,
            )
        )


@dataclass(frozen=True, slots=True)
class CapacityTensor:
    """
    Plane-resolved ROIF Capacity Tensor.

    Axis convention:
        plane_tensor[p, i, j]
            influence from source channel j to target channel i through plane p;

        matrix[i, j]
            aggregated influence from source j to target i.
    """

    channel_ids: tuple[str, ...]
    plane_ids: tuple[str, ...]
    plane_tensor: np.ndarray
    matrix: np.ndarray
    adjacency_mask: np.ndarray
    entries: tuple[TensorEntryFactors, ...] = ()
    channel_states: tuple[ChannelTensorState, ...] = ()
    config: TensorBuildConfig = field(default_factory=TensorBuildConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channel_ids = tuple(
            _text(value, "channel_id") for value in self.channel_ids
        )
        plane_ids = tuple(
            _text(value, "plane_id") for value in self.plane_ids
        )
        if len(channel_ids) != len(set(channel_ids)):
            raise ROIFTensorError("channel_ids must be unique.")
        if len(plane_ids) != len(set(plane_ids)):
            raise ROIFTensorError("plane_ids must be unique.")

        tensor = _readonly_array(
            self.plane_tensor,
            name="plane_tensor",
            ndim=3,
        )
        matrix = _readonly_array(
            self.matrix,
            name="matrix",
            ndim=2,
        )
        mask = _readonly_bool_array(
            self.adjacency_mask,
            name="adjacency_mask",
            ndim=2,
        )

        node_count = len(channel_ids)
        plane_count = len(plane_ids)
        if tensor.shape != (plane_count, node_count, node_count):
            raise ROIFTensorError(
                "plane_tensor shape must be "
                "(len(plane_ids), len(channel_ids), len(channel_ids))."
            )
        if matrix.shape != (node_count, node_count):
            raise ROIFTensorError(
                "matrix shape must be square and match channel_ids."
            )
        if mask.shape != matrix.shape:
            raise ROIFTensorError(
                "adjacency_mask shape must match matrix."
            )

        object.__setattr__(self, "channel_ids", channel_ids)
        object.__setattr__(self, "plane_ids", plane_ids)
        object.__setattr__(self, "plane_tensor", tensor)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "adjacency_mask", mask)
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(
            self,
            "channel_states",
            tuple(self.channel_states),
        )
        if not isinstance(self.config, TensorBuildConfig):
            raise ROIFTensorError(
                "config must be TensorBuildConfig."
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.plane_tensor.shape

    @property
    def node_count(self) -> int:
        return len(self.channel_ids)

    @property
    def plane_count(self) -> int:
        return len(self.plane_ids)

    @property
    def nonzero_count(self) -> int:
        return int(np.count_nonzero(self.matrix))

    @property
    def density(self) -> float:
        if self.node_count == 0:
            return 0.0
        return self.nonzero_count / float(self.node_count**2)

    def index(self, channel_id: str) -> int:
        try:
            return self.channel_ids.index(channel_id)
        except ValueError as exc:
            raise KeyError(channel_id) from exc

    def plane_index(self, plane_id: str) -> int:
        try:
            return self.plane_ids.index(plane_id)
        except ValueError as exc:
            raise KeyError(plane_id) from exc

    def coupling(self, source_id: str, target_id: str) -> float:
        source = self.index(source_id)
        target = self.index(target_id)
        return float(self.matrix[target, source])

    def plane_coupling(
        self,
        plane_id: str,
        source_id: str,
        target_id: str,
    ) -> float:
        plane = self.plane_index(plane_id)
        source = self.index(source_id)
        target = self.index(target_id)
        return float(self.plane_tensor[plane, target, source])

    def propagate(
        self,
        state: Sequence[float] | Mapping[str, float],
        *,
        include_current_state: bool = False,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> np.ndarray:
        """Apply the aggregated cascade operator to one channel-state vector."""

        vector = state_vector(
            state,
            self.channel_ids,
        )
        result = self.matrix @ vector
        if include_current_state:
            result = result + vector
        if clip_min is not None or clip_max is not None:
            lower = -math.inf if clip_min is None else _finite(
                clip_min,
                "clip_min",
            )
            upper = math.inf if clip_max is None else _finite(
                clip_max,
                "clip_max",
            )
            if lower > upper:
                raise ROIFTensorError(
                    "clip_min cannot exceed clip_max."
                )
            result = np.clip(result, lower, upper)
        result = np.asarray(result, dtype=float)
        result.setflags(write=False)
        return result

    def spectral_radius(self) -> float:
        return spectral_radius(self.matrix)

    def spectral_coherence(
        self,
        lambda_critical: float = 1.0,
    ) -> float:
        return spectral_coherence(
            self.matrix,
            lambda_critical=lambda_critical,
        )

    def row(self, target_id: str) -> np.ndarray:
        row = np.array(
            self.matrix[self.index(target_id), :],
            copy=True,
        )
        row.setflags(write=False)
        return row

    def column(self, source_id: str) -> np.ndarray:
        column = np.array(
            self.matrix[:, self.index(source_id)],
            copy=True,
        )
        column.setflags(write=False)
        return column


def channel_reserve_factor(
    channel: FunctionalChannel,
    config: TensorBuildConfig,
) -> float:
    """Convert Capacity / Load / Reserve into a nonnegative tensor factor."""

    reserve_fraction = channel.capacity_state.reserve_fraction
    utilization = channel.capacity_state.utilization

    if config.capacity_factor_mode is CapacityFactorMode.CLAMPED_RESERVE:
        value = max(0.0, min(1.0, reserve_fraction))
    elif config.capacity_factor_mode is CapacityFactorMode.POSITIVE_RESERVE:
        value = max(config.reserve_floor, reserve_fraction)
    elif config.capacity_factor_mode is CapacityFactorMode.EXPONENTIAL_RESERVE:
        value = math.exp(min(0.0, reserve_fraction - 1.0))
    elif (
        config.capacity_factor_mode
        is CapacityFactorMode.UTILIZATION_COMPLEMENT
    ):
        value = max(0.0, 1.0 - utilization)
    else:
        raise ROIFTensorError(
            "unsupported capacity_factor_mode."
        )

    return value ** config.source_reserve_exponent


def channel_acceptance_factor(
    channel: FunctionalChannel,
    config: TensorBuildConfig,
) -> float:
    """
    Estimate the target's remaining ability to accept transmitted influence.

    ``target_reserve_exponent == 0`` makes this factor neutral.
    """

    if config.target_reserve_exponent == 0.0:
        return 1.0
    reserve = max(
        config.reserve_floor,
        min(1.0, channel.capacity_state.reserve_fraction),
    )
    return reserve ** config.target_reserve_exponent


def channel_material_factor(
    channel_id: str,
    materials: Mapping[str, MaterialModel],
    config: TensorBuildConfig,
) -> float:
    """Return the multiplicative material capacity factor for a channel."""

    model = materials.get(channel_id)
    if model is None:
        return 1.0

    factors = material_plane_factors(model)
    value = float(
        factors.get(
            "material.available_capacity",
            factors.get("material.integrity", 1.0),
        )
    )
    value = max(config.material_floor, value)
    return value ** config.material_exponent


def channel_tensor_state(
    channel: FunctionalChannel,
    *,
    materials: Mapping[str, MaterialModel] | None = None,
    config: TensorBuildConfig | None = None,
) -> ChannelTensorState:
    """Create the complete tensor-facing state of one channel."""

    config = config or TensorBuildConfig()
    materials = materials or {}

    reserve = channel_reserve_factor(channel, config)
    availability = (
        channel.activation.effective ** config.availability_exponent
        if config.state_dependent
        else 1.0
    )
    geometry = (
        channel.geometry.geometric_efficiency
        ** config.geometry_exponent
        if config.state_dependent
        else 1.0
    )
    history = (
        max(0.0, channel.history_factor)
        ** config.history_exponent
        if config.state_dependent
        else 1.0
    )
    material = channel_material_factor(
        channel.channel_id,
        materials,
        config,
    )
    acceptance = channel_acceptance_factor(channel, config)

    effective_output = _safe_product(
        (
            reserve,
            availability,
            geometry,
            history,
            material,
        )
    )

    return ChannelTensorState(
        channel_id=channel.channel_id,
        reserve_factor=reserve,
        availability_factor=availability,
        geometry_factor=geometry,
        history_factor=history,
        material_factor=material,
        acceptance_factor=acceptance,
        effective_output=effective_output,
        direction=channel.direction,
        metadata={
            "entity_id": channel.entity_id,
            "channel_state": channel.state.value,
        },
    )


def vector_alignment_factor(
    source: FunctionalChannel,
    target: FunctionalChannel,
    operator: InfluenceOperator,
    config: TensorBuildConfig,
) -> float:
    """
    Compute directional compatibility in [direction_floor, 1].

    If the operator declares an explicit direction, it is compared with the
    target channel. Otherwise source and target channel directions are used.
    Zero vectors are treated as direction-neutral.
    """

    source_direction = (
        operator.direction
        if not math.isclose(
            operator.direction.magnitude,
            0.0,
            abs_tol=1e-15,
        )
        else source.direction
    )
    target_direction = target.direction

    if (
        math.isclose(source_direction.magnitude, 0.0, abs_tol=1e-15)
        or math.isclose(target_direction.magnitude, 0.0, abs_tol=1e-15)
    ):
        return 1.0

    source_unit = source_direction.normalized()
    target_unit = target_direction.normalized()

    dot = (
        source_unit.x * target_unit.x
        + source_unit.y * target_unit.y
        + source_unit.z * target_unit.z
    )

    mode = str(
        operator.metadata.get(
            "alignment_mode",
            "absolute",
        )
    )
    if mode == "signed":
        alignment = max(0.0, dot)
    elif mode == "opposed":
        alignment = max(0.0, -dot)
    elif mode == "absolute":
        alignment = abs(dot)
    elif mode == "neutral":
        alignment = 1.0
    else:
        raise ROIFTensorError(
            f"unsupported alignment_mode {mode!r}."
        )

    return max(config.direction_floor, min(1.0, alignment))


def geometry_pair_factor(
    source_state: ChannelTensorState,
    target_state: ChannelTensorState,
    operator: InfluenceOperator,
) -> float:
    """Combine source and target geometric accessibility."""

    mode = str(
        operator.metadata.get(
            "geometry_mode",
            "geometric_mean",
        )
    )
    source = source_state.geometry_factor
    target = target_state.geometry_factor

    if mode == "source":
        return source
    if mode == "target":
        return target
    if mode == "minimum":
        return min(source, target)
    if mode == "maximum":
        return max(source, target)
    if mode == "product":
        return source * target
    if mode == "geometric_mean":
        return math.sqrt(max(0.0, source * target))
    if mode == "neutral":
        return 1.0

    raise ROIFTensorError(
        f"unsupported geometry_mode {mode!r}."
    )


def prestress_factor(
    operator: InfluenceOperator,
    config: TensorBuildConfig,
    context: InfluenceContext,
) -> float:
    """
    Return the prestress multiplier.

    Supported metadata:
        prestress_factor:
            explicit neutral-centred multiplier;

        prestress:
            scalar prestress level;

        prestress_sensitivity:
            multiplier applied as ``1 + sensitivity * prestress``;

        prestress_state_key:
            read prestress from ``context.scalar_state``.
    """

    if "prestress_factor" in operator.metadata:
        return _nonnegative(
            operator.metadata["prestress_factor"],
            "prestress_factor",
        )

    state_key = operator.metadata.get("prestress_state_key")
    if state_key is not None:
        prestress = context.scalar(str(state_key), 0.0)
    else:
        prestress = _finite(
            operator.metadata.get("prestress", 0.0),
            "prestress",
        )

    sensitivity = _finite(
        operator.metadata.get("prestress_sensitivity", 0.0),
        "prestress_sensitivity",
    )
    value = (
        config.default_prestress_factor
        * (1.0 + sensitivity * prestress)
    )
    return max(0.0, value)


def history_pair_factor(
    source_state: ChannelTensorState,
    target_state: ChannelTensorState,
    operator: InfluenceOperator,
) -> float:
    """Combine source and target structural-history factors."""

    mode = str(
        operator.metadata.get(
            "history_mode",
            "source",
        )
    )
    source = source_state.history_factor
    target = target_state.history_factor

    if mode == "source":
        return source
    if mode == "target":
        return target
    if mode == "product":
        return source * target
    if mode == "geometric_mean":
        return math.sqrt(max(0.0, source * target))
    if mode == "minimum":
        return min(source, target)
    if mode == "maximum":
        return max(source, target)
    if mode == "neutral":
        return 1.0

    raise ROIFTensorError(
        f"unsupported history_mode {mode!r}."
    )


def material_pair_factor(
    source_state: ChannelTensorState,
    target_state: ChannelTensorState,
    operator: InfluenceOperator,
) -> float:
    """Combine source and target material capacity factors."""

    mode = str(
        operator.metadata.get(
            "material_mode",
            "geometric_mean",
        )
    )
    source = source_state.material_factor
    target = target_state.material_factor

    if mode == "source":
        return source
    if mode == "target":
        return target
    if mode == "product":
        return source * target
    if mode == "geometric_mean":
        return math.sqrt(max(0.0, source * target))
    if mode == "minimum":
        return min(source, target)
    if mode == "maximum":
        return max(source, target)
    if mode == "neutral":
        return 1.0

    raise ROIFTensorError(
        f"unsupported material_mode {mode!r}."
    )


def tensor_entry_factors(
    source: FunctionalChannel,
    target: FunctionalChannel,
    operator: InfluenceOperator,
    plane: InfluencePlane,
    context: InfluenceContext,
    *,
    source_state: ChannelTensorState,
    target_state: ChannelTensorState,
    config: TensorBuildConfig,
) -> TensorEntryFactors:
    """Calculate the complete factorization of one operator connection."""

    if operator.plane_id != plane.plane_id:
        raise ROIFTensorError(
            "operator.plane_id must match plane.plane_id."
        )

    if not operator.active and not config.include_inactive_operators:
        return TensorEntryFactors(
            source_id=source.channel_id,
            target_id=target.channel_id,
            operator_id=operator.operator_id,
            plane_id=plane.plane_id,
            base_transmission=0.0,
            operator_gain=0.0,
            plane_factor=0.0,
            source_capacity_factor=0.0,
            source_availability_factor=0.0,
            target_acceptance_factor=0.0,
            geometry_factor=0.0,
            vector_alignment_factor=0.0,
            prestress_factor=0.0,
            material_factor=0.0,
            history_factor=0.0,
            confidence_factor=0.0,
            raw_value=0.0,
            final_value=0.0,
            active=False,
            reason="operator inactive",
        )

    plane_factor = plane_activity(plane, context)
    if plane_factor <= 0.0 and not config.include_inactive_planes:
        return TensorEntryFactors(
            source_id=source.channel_id,
            target_id=target.channel_id,
            operator_id=operator.operator_id,
            plane_id=plane.plane_id,
            base_transmission=0.0,
            operator_gain=0.0,
            plane_factor=0.0,
            source_capacity_factor=0.0,
            source_availability_factor=0.0,
            target_acceptance_factor=0.0,
            geometry_factor=0.0,
            vector_alignment_factor=0.0,
            prestress_factor=0.0,
            material_factor=0.0,
            history_factor=0.0,
            confidence_factor=0.0,
            raw_value=0.0,
            final_value=0.0,
            active=False,
            reason="plane inactive",
        )

    if context.time < operator.delay:
        return TensorEntryFactors(
            source_id=source.channel_id,
            target_id=target.channel_id,
            operator_id=operator.operator_id,
            plane_id=plane.plane_id,
            base_transmission=0.0,
            operator_gain=0.0,
            plane_factor=0.0,
            source_capacity_factor=0.0,
            source_availability_factor=0.0,
            target_acceptance_factor=0.0,
            geometry_factor=0.0,
            vector_alignment_factor=0.0,
            prestress_factor=0.0,
            material_factor=0.0,
            history_factor=0.0,
            confidence_factor=0.0,
            raw_value=0.0,
            final_value=0.0,
            active=False,
            reason="operator delay not reached",
        )

    if not context.condition(operator.condition_key):
        return TensorEntryFactors(
            source_id=source.channel_id,
            target_id=target.channel_id,
            operator_id=operator.operator_id,
            plane_id=plane.plane_id,
            base_transmission=0.0,
            operator_gain=0.0,
            plane_factor=0.0,
            source_capacity_factor=0.0,
            source_availability_factor=0.0,
            target_acceptance_factor=0.0,
            geometry_factor=0.0,
            vector_alignment_factor=0.0,
            prestress_factor=0.0,
            material_factor=0.0,
            history_factor=0.0,
            confidence_factor=0.0,
            raw_value=0.0,
            final_value=0.0,
            active=False,
            reason="operator condition false",
        )

    base = _nonnegative(
        operator.metadata.get(
            "base_transmission",
            config.default_base_transmission,
        ),
        "base_transmission",
    )

    gain = effective_operator_gain(
        operator,
        plane,
        context,
    )
    if not config.allow_negative_operator_gain:
        gain = max(0.0, gain)

    source_capacity = source_state.reserve_factor
    source_availability = source_state.availability_factor
    target_acceptance = target_state.acceptance_factor
    geometry = geometry_pair_factor(
        source_state,
        target_state,
        operator,
    )
    alignment = vector_alignment_factor(
        source,
        target,
        operator,
        config,
    )
    prestress = prestress_factor(
        operator,
        config,
        context,
    )
    material = material_pair_factor(
        source_state,
        target_state,
        operator,
    )
    history = history_pair_factor(
        source_state,
        target_state,
        operator,
    )
    confidence = _unit(
        operator.metadata.get(
            "confidence",
            config.default_confidence,
        ),
        "confidence",
    )

    # effective_operator_gain already includes plane_activity. Divide it out
    # so plane influence appears exactly once in the audited factorization.
    if plane_factor > 0.0:
        gain_without_plane = gain / plane_factor
    else:
        gain_without_plane = 0.0

    raw = _safe_product(
        (
            base,
            gain_without_plane,
            plane_factor,
            source_capacity,
            source_availability,
            target_acceptance,
            geometry,
            alignment,
            prestress,
            material,
            history,
            confidence,
        )
    )

    final = raw
    if config.clip_final_tensor:
        final = min(
            config.tensor_ceiling,
            max(config.tensor_floor, final),
        )

    return TensorEntryFactors(
        source_id=source.channel_id,
        target_id=target.channel_id,
        operator_id=operator.operator_id,
        plane_id=plane.plane_id,
        base_transmission=base,
        operator_gain=gain_without_plane,
        plane_factor=plane_factor,
        source_capacity_factor=source_capacity,
        source_availability_factor=source_availability,
        target_acceptance_factor=target_acceptance,
        geometry_factor=geometry,
        vector_alignment_factor=alignment,
        prestress_factor=prestress,
        material_factor=material,
        history_factor=history,
        confidence_factor=confidence,
        raw_value=raw,
        final_value=final,
        active=True,
        reason="active",
        metadata={
            "source_entity_id": source.entity_id,
            "target_entity_id": target.entity_id,
            "operator_kind": operator.kind.value,
            "plane_kind": plane.kind.value,
        },
    )


def aggregate_values(
    current: float,
    contribution: float,
    aggregation: TensorAggregation,
    *,
    has_existing: bool,
) -> float:
    """Combine one parallel contribution with an existing tensor entry."""

    if not has_existing:
        return contribution
    if aggregation is TensorAggregation.ADDITIVE:
        return current + contribution
    if aggregation is TensorAggregation.MULTIPLICATIVE:
        return current * contribution
    if aggregation is TensorAggregation.MAXIMUM:
        return max(current, contribution)
    if aggregation is TensorAggregation.MINIMUM:
        return min(current, contribution)
    raise ROIFTensorError("unsupported tensor aggregation.")


def normalize_matrix(
    matrix: np.ndarray,
    mode: TensorNormalization,
    *,
    spectral_target: float = 0.95,
) -> np.ndarray:
    """Return a normalized copy of a square cascade matrix."""

    result = np.array(matrix, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ROIFTensorError("matrix must be square.")

    if mode is TensorNormalization.NONE:
        return result

    if mode is TensorNormalization.ROW:
        sums = np.sum(np.abs(result), axis=1)
        nonzero = sums > 1e-15
        result[nonzero, :] = result[nonzero, :] / sums[nonzero, None]
        return result

    if mode is TensorNormalization.COLUMN:
        sums = np.sum(np.abs(result), axis=0)
        nonzero = sums > 1e-15
        result[:, nonzero] = result[:, nonzero] / sums[None, nonzero]
        return result

    if mode is TensorNormalization.GLOBAL_MAX:
        maximum = float(np.max(np.abs(result))) if result.size else 0.0
        if maximum > 1e-15:
            result /= maximum
        return result

    if mode is TensorNormalization.SPECTRAL:
        radius = spectral_radius(result)
        if radius > spectral_target and radius > 1e-15:
            result *= spectral_target / radius
        return result

    raise ROIFTensorError("unsupported tensor normalization.")


def spectral_radius(matrix: Any) -> float:
    """Return the maximum absolute eigenvalue of a finite square matrix."""

    array = np.asarray(matrix, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or array.shape[0] == 0
    ):
        raise ROIFTensorError(
            "matrix must be a non-empty square matrix."
        )
    if not np.all(np.isfinite(array)):
        raise ROIFTensorError(
            "matrix must contain only finite values."
        )
    eigenvalues = np.linalg.eigvals(array)
    return float(np.max(np.abs(eigenvalues)))


def spectral_coherence(
    matrix: Any,
    *,
    lambda_critical: float = 1.0,
    clamp: bool = False,
) -> float:
    """
    Compute ROIF spectral coherence:

        eta = 1 - spectral_radius(W) / lambda_critical
    """

    critical = _positive(lambda_critical, "lambda_critical")
    if not isinstance(clamp, bool):
        raise ROIFTensorError("clamp must be bool.")

    eta = 1.0 - spectral_radius(matrix) / critical
    if clamp:
        eta = max(-1.0, min(1.0, eta))
    return eta


def state_vector(
    state: Sequence[float] | Mapping[str, float],
    channel_ids: Sequence[str],
) -> np.ndarray:
    """Convert a named or positional state into an immutable numeric vector."""

    ids = tuple(channel_ids)
    if isinstance(state, Mapping):
        values = [
            _finite(state.get(channel_id, 0.0), f"state[{channel_id!r}]")
            for channel_id in ids
        ]
    else:
        values = [_finite(value, "state value") for value in state]
        if len(values) != len(ids):
            raise ROIFTensorError(
                "state length must match channel_ids."
            )

    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _system_channels(
    system: ROIFSystem,
) -> tuple[FunctionalChannel, ...]:
    channels = tuple(
        channel
        for entity in system.entities
        for channel in entity.channels
    )
    if not channels:
        raise ROIFTensorError(
            "system must contain at least one FunctionalChannel."
        )
    return channels


def _resolve_endpoint_channels(
    endpoint_id: str,
    channels_by_id: Mapping[str, FunctionalChannel],
    entity_channels: Mapping[str, tuple[FunctionalChannel, ...]],
) -> tuple[FunctionalChannel, ...]:
    """
    Resolve an operator endpoint.

    Endpoints may reference either an exact channel_id or an entity_id. Entity
    references expand to all channels owned by that entity.
    """

    if endpoint_id in channels_by_id:
        return (channels_by_id[endpoint_id],)
    if endpoint_id in entity_channels:
        return entity_channels[endpoint_id]
    return ()


def build_capacity_tensor(
    system: ROIFSystem,
    *,
    context: InfluenceContext | None = None,
    materials: Mapping[str, MaterialModel] | None = None,
    config: TensorBuildConfig | None = None,
) -> CapacityTensor:
    """
    Assemble a plane-resolved, state-dependent Capacity Tensor from ROIFSystem.

    Material mappings are keyed by channel_id. A future adapter may map one
    material model to several channels before calling this function.
    """

    if not isinstance(system, ROIFSystem):
        raise TypeError("system must be ROIFSystem.")

    context = context or InfluenceContext()
    config = config or TensorBuildConfig()
    materials = materials or {}

    channels = _system_channels(system)
    channel_ids = tuple(channel.channel_id for channel in channels)
    channel_index = {
        channel_id: index
        for index, channel_id in enumerate(channel_ids)
    }
    channels_by_id = {
        channel.channel_id: channel for channel in channels
    }

    entity_channels: dict[str, tuple[FunctionalChannel, ...]] = {}
    for entity in system.entities:
        entity_channels[entity.entity_id] = tuple(entity.channels)

    planes_by_id = {
        plane.plane_id: plane for plane in system.planes
    }
    if len(planes_by_id) != len(system.planes):
        raise ROIFTensorError("system contains duplicate plane_id values.")

    operator_plane_ids = tuple(
        dict.fromkeys(
            operator.plane_id for operator in system.operators
        )
    )
    missing_planes = [
        plane_id
        for plane_id in operator_plane_ids
        if plane_id not in planes_by_id
    ]
    if missing_planes:
        raise ROIFTensorError(
            f"operators reference unknown planes: {missing_planes!r}."
        )

    plane_ids = tuple(
        plane.plane_id for plane in system.planes
    )
    plane_index = {
        plane_id: index for index, plane_id in enumerate(plane_ids)
    }

    node_count = len(channel_ids)
    plane_count = len(plane_ids)
    tensor = np.zeros(
        (plane_count, node_count, node_count),
        dtype=float,
    )
    matrix = np.zeros((node_count, node_count), dtype=float)
    mask = np.zeros((node_count, node_count), dtype=bool)

    channel_states = tuple(
        channel_tensor_state(
            channel,
            materials=materials,
            config=config,
        )
        for channel in channels
    )
    states_by_id = {
        item.channel_id: item for item in channel_states
    }

    entries: list[TensorEntryFactors] = []
    plane_has_value = np.zeros_like(tensor, dtype=bool)
    matrix_has_value = np.zeros_like(matrix, dtype=bool)

    for operator in system.operators:
        plane = planes_by_id[operator.plane_id]
        sources: list[FunctionalChannel] = []
        targets: list[FunctionalChannel] = []

        for source_id in operator.source_ids:
            sources.extend(
                _resolve_endpoint_channels(
                    source_id,
                    channels_by_id,
                    entity_channels,
                )
            )
        for target_id in operator.target_ids:
            targets.extend(
                _resolve_endpoint_channels(
                    target_id,
                    channels_by_id,
                    entity_channels,
                )
            )

        # Operators without explicit endpoints are retained in the system but
        # do not contribute to the pairwise tensor.
        if not sources or not targets:
            continue

        for source in sources:
            for target in targets:
                entry = tensor_entry_factors(
                    source,
                    target,
                    operator,
                    plane,
                    context,
                    source_state=states_by_id[source.channel_id],
                    target_state=states_by_id[target.channel_id],
                    config=config,
                )
                entries.append(entry)
                if not entry.active:
                    continue

                p = plane_index[plane.plane_id]
                i = channel_index[target.channel_id]
                j = channel_index[source.channel_id]

                tensor[p, i, j] = aggregate_values(
                    tensor[p, i, j],
                    entry.final_value,
                    config.aggregation,
                    has_existing=bool(plane_has_value[p, i, j]),
                )
                plane_has_value[p, i, j] = True

                matrix[i, j] = aggregate_values(
                    matrix[i, j],
                    entry.final_value,
                    config.aggregation,
                    has_existing=bool(matrix_has_value[i, j]),
                )
                matrix_has_value[i, j] = True
                mask[i, j] = True

    if config.self_coupling_mode is SelfCouplingMode.IDENTITY:
        np.fill_diagonal(matrix, 1.0)
        np.fill_diagonal(mask, True)
    elif config.self_coupling_mode is SelfCouplingMode.RETENTION:
        retention = min(
            config.tensor_ceiling,
            max(config.tensor_floor, config.self_retention),
        )
        if retention > 0.0:
            for index in range(node_count):
                matrix[index, index] = aggregate_values(
                    matrix[index, index],
                    retention,
                    config.aggregation,
                    has_existing=bool(matrix_has_value[index, index]),
                )
                mask[index, index] = True
    elif config.self_coupling_mode in (
        SelfCouplingMode.NONE,
        SelfCouplingMode.EXPLICIT_ONLY,
    ):
        pass
    else:
        raise ROIFTensorError(
            "unsupported self_coupling_mode."
        )

    matrix = normalize_matrix(
        matrix,
        config.normalization,
        spectral_target=config.spectral_target,
    )

    if config.clip_final_tensor:
        tensor = np.clip(
            tensor,
            config.tensor_floor,
            config.tensor_ceiling,
        )
        matrix = np.clip(
            matrix,
            config.tensor_floor,
            config.tensor_ceiling,
        )

    radius = (
        spectral_radius(matrix)
        if node_count > 0
        else 0.0
    )
    eta = 1.0 - radius

    return CapacityTensor(
        channel_ids=channel_ids,
        plane_ids=plane_ids,
        plane_tensor=tensor,
        matrix=matrix,
        adjacency_mask=mask,
        entries=tuple(entries),
        channel_states=channel_states,
        config=config,
        metadata={
            "system_id": system.system_id,
            "operator_count": len(system.operators),
            "active_entry_count": sum(
                1 for entry in entries if entry.active
            ),
            "spectral_radius": radius,
            "spectral_coherence_at_1": eta,
            "state_dependent": config.state_dependent,
        },
    )


def build_static_capacity_tensor(
    system: ROIFSystem,
    *,
    context: InfluenceContext | None = None,
    materials: Mapping[str, MaterialModel] | None = None,
    config: TensorBuildConfig | None = None,
) -> CapacityTensor:
    """Build a topology/operator tensor with state-dependent factors disabled."""

    source = config or TensorBuildConfig()
    static_config = TensorBuildConfig(
        aggregation=source.aggregation,
        normalization=source.normalization,
        capacity_factor_mode=source.capacity_factor_mode,
        self_coupling_mode=source.self_coupling_mode,
        tensor_floor=source.tensor_floor,
        tensor_ceiling=source.tensor_ceiling,
        reserve_floor=source.reserve_floor,
        direction_floor=source.direction_floor,
        material_floor=source.material_floor,
        default_base_transmission=source.default_base_transmission,
        default_prestress_factor=source.default_prestress_factor,
        default_confidence=source.default_confidence,
        self_retention=source.self_retention,
        spectral_target=source.spectral_target,
        source_reserve_exponent=0.0,
        target_reserve_exponent=0.0,
        availability_exponent=0.0,
        geometry_exponent=0.0,
        material_exponent=0.0,
        history_exponent=0.0,
        state_dependent=False,
        include_inactive_operators=source.include_inactive_operators,
        include_inactive_planes=source.include_inactive_planes,
        allow_negative_operator_gain=source.allow_negative_operator_gain,
        clip_final_tensor=source.clip_final_tensor,
        metadata={
            **dict(source.metadata),
            "static_tensor": True,
        },
    )
    return build_capacity_tensor(
        system,
        context=context,
        materials=materials,
        config=static_config,
    )


def tensor_difference(
    left: CapacityTensor,
    right: CapacityTensor,
) -> np.ndarray:
    """Return ``left.matrix - right.matrix`` for matching tensor axes."""

    if left.channel_ids != right.channel_ids:
        raise ROIFTensorError(
            "capacity tensors must use the same channel order."
        )
    result = np.array(left.matrix - right.matrix, copy=True)
    result.setflags(write=False)
    return result


def tensor_relative_change(
    before: CapacityTensor,
    after: CapacityTensor,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return element-wise relative change between matching tensors."""

    if before.channel_ids != after.channel_ids:
        raise ROIFTensorError(
            "capacity tensors must use the same channel order."
        )
    epsilon = _positive(epsilon, "epsilon")
    denominator = np.maximum(np.abs(before.matrix), epsilon)
    result = (after.matrix - before.matrix) / denominator
    result = np.asarray(result, dtype=float)
    result.setflags(write=False)
    return result


def tensor_summary(
    tensor: CapacityTensor,
) -> Mapping[str, Any]:
    """Return a compact immutable numerical summary."""

    return MappingProxyType(
        {
            "node_count": tensor.node_count,
            "plane_count": tensor.plane_count,
            "shape": tensor.shape,
            "nonzero_count": tensor.nonzero_count,
            "density": tensor.density,
            "spectral_radius": tensor.spectral_radius(),
            "spectral_coherence_at_1": tensor.spectral_coherence(1.0),
            "maximum_coupling": (
                float(np.max(tensor.matrix))
                if tensor.matrix.size
                else 0.0
            ),
            "mean_coupling": (
                float(np.mean(tensor.matrix))
                if tensor.matrix.size
                else 0.0
            ),
        }
    )


__all__ = [
    "CapacityFactorMode",
    "CapacityTensor",
    "ChannelTensorState",
    "ROIFTensorError",
    "SelfCouplingMode",
    "TensorAggregation",
    "TensorBuildConfig",
    "TensorEntryFactors",
    "TensorNormalization",
    "aggregate_values",
    "build_capacity_tensor",
    "build_static_capacity_tensor",
    "channel_acceptance_factor",
    "channel_material_factor",
    "channel_reserve_factor",
    "channel_tensor_state",
    "geometry_pair_factor",
    "history_pair_factor",
    "material_pair_factor",
    "normalize_matrix",
    "prestress_factor",
    "spectral_coherence",
    "spectral_radius",
    "state_vector",
    "tensor_difference",
    "tensor_entry_factors",
    "tensor_relative_change",
    "tensor_summary",
    "vector_alignment_factor",
]

