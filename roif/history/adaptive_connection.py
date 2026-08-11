"""
ROIF Engine
Adaptive Connection Core

Purpose
-------
Represent a soft, stateful, contractile connection in a prestressed network.

This layer models the idea that a connection is not a passive fixed spring.
After an event, its effective mechanical/response properties may change, and
future redistribution therefore occurs through a different connection state.

The module is domain-agnostic.

Core state
----------
An adaptive connection stores:
- stiffness
- contractile_capacity
- reflex_gain
- fatigue
- remodeling_bias

An event may produce:
- load exposure
- strain exposure
- activation demand
- damage
- recovery

The update is deterministic and immutable.

This module does NOT:
- infer biological truth
- diagnose pathology
- learn hidden parameters
- change topology
- select treatment/action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "adaptive_connection_v1"


class AdaptiveConnectionError(ValueError):
    """Raised when adaptive-connection inputs are invalid."""


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
        raise AdaptiveConnectionError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    number = float(value)

    if not isfinite(number):
        raise AdaptiveConnectionError(
            f"{name} must be finite"
        )

    return number


def _validate_unit_interval(
    name: str,
    value: float,
) -> float:
    number = _validate_finite(
        name,
        value,
    )

    if not (
        0.0
        <= number
        <= 1.0
    ):
        raise AdaptiveConnectionError(
            f"{name} must lie in [0, 1]"
        )

    return number


def _clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return min(
        upper,
        max(
            lower,
            value,
        ),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveConnectionState:
    """
    Stateful soft connection.

    stiffness
        Effective resistance to deformation.

    contractile_capacity
        Available active shortening / force-generating capacity.

    reflex_gain
        Sensitivity of reflexive/automatic response.

    fatigue
        Accumulated short-term loss of available response, in [0, 1].

    remodeling_bias
        Slow structural tendency in [-1, 1].
        Positive values bias toward strengthening/stiffening,
        negative values toward weakening/softening.

    min/max fields define hard admissible bounds.
    """

    connection_id: str
    source_node_id: str
    target_node_id: str

    stiffness: float
    contractile_capacity: float
    reflex_gain: float

    fatigue: float = 0.0
    remodeling_bias: float = 0.0

    min_stiffness: float = 0.0
    max_stiffness: float = 10.0

    min_contractile_capacity: float = 0.0
    max_contractile_capacity: float = 10.0

    min_reflex_gain: float = 0.0
    max_reflex_gain: float = 10.0

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
            raise AdaptiveConnectionError(
                "self-connections are not allowed"
            )

        stiffness = _validate_finite(
            "stiffness",
            self.stiffness,
        )

        contractile_capacity = _validate_finite(
            "contractile_capacity",
            self.contractile_capacity,
        )

        reflex_gain = _validate_finite(
            "reflex_gain",
            self.reflex_gain,
        )

        fatigue = _validate_unit_interval(
            "fatigue",
            self.fatigue,
        )

        remodeling_bias = _validate_finite(
            "remodeling_bias",
            self.remodeling_bias,
        )

        if not (
            -1.0
            <= remodeling_bias
            <= 1.0
        ):
            raise AdaptiveConnectionError(
                "remodeling_bias must lie in [-1, 1]"
            )

        min_stiffness = _validate_finite(
            "min_stiffness",
            self.min_stiffness,
        )

        max_stiffness = _validate_finite(
            "max_stiffness",
            self.max_stiffness,
        )

        min_contractile_capacity = _validate_finite(
            "min_contractile_capacity",
            self.min_contractile_capacity,
        )

        max_contractile_capacity = _validate_finite(
            "max_contractile_capacity",
            self.max_contractile_capacity,
        )

        min_reflex_gain = _validate_finite(
            "min_reflex_gain",
            self.min_reflex_gain,
        )

        max_reflex_gain = _validate_finite(
            "max_reflex_gain",
            self.max_reflex_gain,
        )

        if min_stiffness > max_stiffness:
            raise AdaptiveConnectionError(
                "min_stiffness must be <= max_stiffness"
            )

        if (
            min_contractile_capacity
            > max_contractile_capacity
        ):
            raise AdaptiveConnectionError(
                "min_contractile_capacity must be <= "
                "max_contractile_capacity"
            )

        if min_reflex_gain > max_reflex_gain:
            raise AdaptiveConnectionError(
                "min_reflex_gain must be <= max_reflex_gain"
            )

        if not (
            min_stiffness
            <= stiffness
            <= max_stiffness
        ):
            raise AdaptiveConnectionError(
                "stiffness must lie within bounds"
            )

        if not (
            min_contractile_capacity
            <= contractile_capacity
            <= max_contractile_capacity
        ):
            raise AdaptiveConnectionError(
                "contractile_capacity must lie within bounds"
            )

        if not (
            min_reflex_gain
            <= reflex_gain
            <= max_reflex_gain
        ):
            raise AdaptiveConnectionError(
                "reflex_gain must lie within bounds"
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
            "stiffness",
            stiffness,
        )

        object.__setattr__(
            self,
            "contractile_capacity",
            contractile_capacity,
        )

        object.__setattr__(
            self,
            "reflex_gain",
            reflex_gain,
        )

        object.__setattr__(
            self,
            "fatigue",
            fatigue,
        )

        object.__setattr__(
            self,
            "remodeling_bias",
            remodeling_bias,
        )

        object.__setattr__(
            self,
            "min_stiffness",
            min_stiffness,
        )

        object.__setattr__(
            self,
            "max_stiffness",
            max_stiffness,
        )

        object.__setattr__(
            self,
            "min_contractile_capacity",
            min_contractile_capacity,
        )

        object.__setattr__(
            self,
            "max_contractile_capacity",
            max_contractile_capacity,
        )

        object.__setattr__(
            self,
            "min_reflex_gain",
            min_reflex_gain,
        )

        object.__setattr__(
            self,
            "max_reflex_gain",
            max_reflex_gain,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ConnectionExposure:
    """
    Explicit event exposure applied to a connection.

    load
        External/mechanical load demand, nonnegative.

    strain
        Magnitude of deformation demand, nonnegative.

    activation
        Active contraction demand, in [0, 1].

    damage
        Event-linked structural loss, in [0, 1].

    recovery
        Recovery input available during the step, in [0, 1].
    """

    exposure_id: str
    connection_id: str

    load: float = 0.0
    strain: float = 0.0
    activation: float = 0.0
    damage: float = 0.0
    recovery: float = 0.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exposure_id",
            _validate_id(
                "exposure_id",
                self.exposure_id,
            ),
        )

        object.__setattr__(
            self,
            "connection_id",
            _validate_id(
                "connection_id",
                self.connection_id,
            ),
        )

        load = _validate_finite(
            "load",
            self.load,
        )

        strain = _validate_finite(
            "strain",
            self.strain,
        )

        if load < 0.0:
            raise AdaptiveConnectionError(
                "load must be nonnegative"
            )

        if strain < 0.0:
            raise AdaptiveConnectionError(
                "strain must be nonnegative"
            )

        activation = _validate_unit_interval(
            "activation",
            self.activation,
        )

        damage = _validate_unit_interval(
            "damage",
            self.damage,
        )

        recovery = _validate_unit_interval(
            "recovery",
            self.recovery,
        )

        object.__setattr__(
            self,
            "load",
            load,
        )

        object.__setattr__(
            self,
            "strain",
            strain,
        )

        object.__setattr__(
            self,
            "activation",
            activation,
        )

        object.__setattr__(
            self,
            "damage",
            damage,
        )

        object.__setattr__(
            self,
            "recovery",
            recovery,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveConnectionConfig:
    """
    Deterministic update coefficients.

    fatigue_load_gain
        How strongly load raises fatigue.

    fatigue_strain_gain
        How strongly strain raises fatigue.

    fatigue_activation_gain
        How strongly activation raises fatigue.

    recovery_gain
        How strongly recovery reduces fatigue.

    damage_capacity_loss_gain
        How strongly damage reduces contractile capacity.

    fatigue_capacity_loss_gain
        How strongly fatigue reduces contractile capacity.

    reflex_activation_gain
        Activation-linked reflex increase.

    reflex_damage_gain
        Damage-linked reflex change.

    stiffness_load_gain
        Load-linked stiffness increase.

    stiffness_damage_gain
        Damage-linked stiffness loss.

    remodeling_rate
        Slow update speed for remodeling_bias.
    """

    fatigue_load_gain: float = 0.10
    fatigue_strain_gain: float = 0.10
    fatigue_activation_gain: float = 0.10
    recovery_gain: float = 0.20

    damage_capacity_loss_gain: float = 0.50
    fatigue_capacity_loss_gain: float = 0.25

    reflex_activation_gain: float = 0.20
    reflex_damage_gain: float = 0.10

    stiffness_load_gain: float = 0.10
    stiffness_damage_gain: float = 0.20

    remodeling_rate: float = 0.05

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        numeric_fields = (
            "fatigue_load_gain",
            "fatigue_strain_gain",
            "fatigue_activation_gain",
            "recovery_gain",
            "damage_capacity_loss_gain",
            "fatigue_capacity_loss_gain",
            "reflex_activation_gain",
            "reflex_damage_gain",
            "stiffness_load_gain",
            "stiffness_damage_gain",
            "remodeling_rate",
        )

        for name in numeric_fields:
            value = _validate_finite(
                name,
                getattr(self, name),
            )

            if value < 0.0:
                raise AdaptiveConnectionError(
                    f"{name} must be nonnegative"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveConnectionDelta:
    connection_id: str

    stiffness_delta: float
    contractile_capacity_delta: float
    reflex_gain_delta: float
    fatigue_delta: float
    remodeling_bias_delta: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "connection_id",
            _validate_id(
                "connection_id",
                self.connection_id,
            ),
        )

        for name in (
            "stiffness_delta",
            "contractile_capacity_delta",
            "reflex_gain_delta",
            "fatigue_delta",
            "remodeling_bias_delta",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(self, name),
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveConnectionResult:
    update_id: str

    source_state: AdaptiveConnectionState
    exposure: ConnectionExposure
    target_state: AdaptiveConnectionState
    delta: AdaptiveConnectionDelta

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "update_id",
            _validate_id(
                "update_id",
                self.update_id,
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _fatigue_after_exposure(
    *,
    state: AdaptiveConnectionState,
    exposure: ConnectionExposure,
    config: AdaptiveConnectionConfig,
) -> float:
    increase = (
        config.fatigue_load_gain
        * exposure.load
        + config.fatigue_strain_gain
        * exposure.strain
        + config.fatigue_activation_gain
        * exposure.activation
    )

    recovery = (
        config.recovery_gain
        * exposure.recovery
    )

    return _clamp(
        state.fatigue
        + increase
        - recovery,
        0.0,
        1.0,
    )


def _remodeling_bias_after_exposure(
    *,
    state: AdaptiveConnectionState,
    exposure: ConnectionExposure,
    config: AdaptiveConnectionConfig,
) -> float:
    strengthening_signal = (
        exposure.load
        + exposure.activation
    )

    weakening_signal = (
        exposure.damage
        + exposure.strain
    )

    delta = (
        config.remodeling_rate
        * (
            strengthening_signal
            - weakening_signal
        )
    )

    return _clamp(
        state.remodeling_bias
        + delta,
        -1.0,
        1.0,
    )


def update_adaptive_connection(
    *,
    update_id: str,
    state: AdaptiveConnectionState,
    exposure: ConnectionExposure,
    config: AdaptiveConnectionConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AdaptiveConnectionResult:
    """
    Apply one explicit event exposure to one adaptive connection.

    The update is state-dependent:
    the same exposure can produce different outcomes when the source connection
    has different fatigue/remodeling states.
    """
    update_id = _validate_id(
        "update_id",
        update_id,
    )

    if (
        exposure.connection_id
        != state.connection_id
    ):
        raise AdaptiveConnectionError(
            "exposure.connection_id must match state.connection_id"
        )

    actual_config = (
        config
        if config is not None
        else AdaptiveConnectionConfig()
    )

    next_fatigue = _fatigue_after_exposure(
        state=state,
        exposure=exposure,
        config=actual_config,
    )

    next_remodeling_bias = (
        _remodeling_bias_after_exposure(
            state=state,
            exposure=exposure,
            config=actual_config,
        )
    )

    stiffness_raw = (
        state.stiffness
        + actual_config.stiffness_load_gain
        * exposure.load
        - actual_config.stiffness_damage_gain
        * exposure.damage
        + next_remodeling_bias
        * actual_config.remodeling_rate
    )

    next_stiffness = _clamp(
        stiffness_raw,
        state.min_stiffness,
        state.max_stiffness,
    )

    capacity_loss = (
        actual_config.damage_capacity_loss_gain
        * exposure.damage
        + actual_config.fatigue_capacity_loss_gain
        * next_fatigue
    )

    capacity_recovery = (
        actual_config.recovery_gain
        * exposure.recovery
    )

    capacity_raw = (
        state.contractile_capacity
        - capacity_loss
        + capacity_recovery
        + max(
            0.0,
            next_remodeling_bias,
        )
        * actual_config.remodeling_rate
    )

    next_contractile_capacity = _clamp(
        capacity_raw,
        state.min_contractile_capacity,
        state.max_contractile_capacity,
    )

    reflex_raw = (
        state.reflex_gain
        + actual_config.reflex_activation_gain
        * exposure.activation
        + actual_config.reflex_damage_gain
        * exposure.damage
        - actual_config.recovery_gain
        * exposure.recovery
        + next_remodeling_bias
        * actual_config.remodeling_rate
    )

    next_reflex_gain = _clamp(
        reflex_raw,
        state.min_reflex_gain,
        state.max_reflex_gain,
    )

    target_state = AdaptiveConnectionState(
        connection_id=state.connection_id,
        source_node_id=state.source_node_id,
        target_node_id=state.target_node_id,
        stiffness=next_stiffness,
        contractile_capacity=next_contractile_capacity,
        reflex_gain=next_reflex_gain,
        fatigue=next_fatigue,
        remodeling_bias=next_remodeling_bias,
        min_stiffness=state.min_stiffness,
        max_stiffness=state.max_stiffness,
        min_contractile_capacity=state.min_contractile_capacity,
        max_contractile_capacity=state.max_contractile_capacity,
        min_reflex_gain=state.min_reflex_gain,
        max_reflex_gain=state.max_reflex_gain,
        metadata=dict(
            state.metadata
        ),
    )

    delta = AdaptiveConnectionDelta(
        connection_id=state.connection_id,
        stiffness_delta=(
            target_state.stiffness
            - state.stiffness
        ),
        contractile_capacity_delta=(
            target_state.contractile_capacity
            - state.contractile_capacity
        ),
        reflex_gain_delta=(
            target_state.reflex_gain
            - state.reflex_gain
        ),
        fatigue_delta=(
            target_state.fatigue
            - state.fatigue
        ),
        remodeling_bias_delta=(
            target_state.remodeling_bias
            - state.remodeling_bias
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "learning_applied": False,
            "topology_modified": False,
        },
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "adaptive_connection_mode": "deterministic_state_update",
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
            dict(
                metadata
            )
        )

    return AdaptiveConnectionResult(
        update_id=update_id,
        source_state=state,
        exposure=exposure,
        target_state=target_state,
        delta=delta,
        metadata=merged_metadata,
    )


def adaptive_connection_is_policy_free(
    result: AdaptiveConnectionResult,
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


def adaptive_connection_signature(
    result: AdaptiveConnectionResult,
) -> tuple[
    str,
    str,
    float,
    float,
    float,
    float,
    float,
]:
    state = result.target_state

    return (
        result.update_id,
        state.connection_id,
        state.stiffness,
        state.contractile_capacity,
        state.reflex_gain,
        state.fatigue,
        state.remodeling_bias,
    )


def effective_transfer_gain(
    state: AdaptiveConnectionState,
) -> float:
    """
    Domain-agnostic transfer proxy for later integration with prestress
    redistribution.

    The proxy increases with stiffness, active capacity and reflex gain, and is
    reduced by fatigue.

    It is not a biological law; it is an explicit computational bridge.
    """
    return (
        state.stiffness
        * state.contractile_capacity
        * state.reflex_gain
        * (
            1.0
            - state.fatigue
        )
    )


__all__ = [
    "AdaptiveConnectionConfig",
    "AdaptiveConnectionDelta",
    "AdaptiveConnectionError",
    "AdaptiveConnectionResult",
    "AdaptiveConnectionState",
    "ConnectionExposure",
    "SCHEMA_VERSION",
    "adaptive_connection_is_policy_free",
    "adaptive_connection_signature",
    "effective_transfer_gain",
    "update_adaptive_connection",
]
