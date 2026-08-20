"""
ROIF Engine
System Evolution Core

Purpose
-------
Integrate the new physical-history branch into one deterministic event cycle:

    source SystemImage
        -> connection exposures
        -> adaptive connection updates
        -> prestress perturbations / redistribution
        -> historical trace creation
        -> revised SystemImage

The source image is never mutated.

The module is domain-agnostic and deliberately conservative:
- explicit inputs only
- explicit topology only
- no hidden learning
- no diagnosis
- no biological truth claim
- no policy/action selection
- deterministic ordering
- immutable result objects

This module does not assume that the whole system is additive or
multiplicative. It preserves the multilayer system image and updates each
layer through its own explicit rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.history.adaptive_connection import (
    AdaptiveConnectionConfig,
    AdaptiveConnectionResult,
    AdaptiveConnectionState,
    ConnectionExposure,
    effective_transfer_gain,
    update_adaptive_connection,
)
from roif.history.prestress_redistribution import (
    PrestressConnection,
    PrestressPerturbation,
    PrestressRedistributionConfig,
    PrestressRedistributionResult,
    redistribute_prestress,
)
from roif.history.system_image import (
    HistoricalTrace,
    SystemImage,
    SystemImageContext,
    SystemMeasure,
    build_system_image,
)
from roif.history.transition_modifiers import (
    TransitionChannel,
    TransitionModifierSet,
)


SCHEMA_VERSION = "system_evolution_v1"


class SystemEvolutionError(ValueError):
    """Raised when system-evolution inputs are invalid."""


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
        raise SystemEvolutionError(
            f"{name} must not be empty"
        )

    return cleaned


def _validate_finite(
    name: str,
    value: float,
) -> float:
    number = float(value)

    if not isfinite(number):
        raise SystemEvolutionError(
            f"{name} must be finite"
        )

    return number


@dataclass(frozen=True, slots=True)
class SystemEvent:
    """
    Explicit event applied to a SystemImage.

    connection_exposures
        Stateful soft-connection exposures.

    prestress_perturbations
        Explicit local prestress changes introduced by the event.

    replacement_measures
        Optional new measurement layer observed after the event.
        If empty, source measures are preserved.

    target_context
        Optional new observation context.
    """

    event_id: str
    event_type: str

    connection_exposures: tuple[
        ConnectionExposure,
        ...
    ] = ()

    prestress_perturbations: tuple[
        PrestressPerturbation,
        ...
    ] = ()

    replacement_measures: tuple[
        SystemMeasure,
        ...
    ] = ()

    target_context: SystemImageContext | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_id",
            _validate_id(
                "event_id",
                self.event_id,
            ),
        )

        object.__setattr__(
            self,
            "event_type",
            _validate_id(
                "event_type",
                self.event_type,
            ),
        )

        object.__setattr__(
            self,
            "connection_exposures",
            tuple(
                self.connection_exposures
            ),
        )

        object.__setattr__(
            self,
            "prestress_perturbations",
            tuple(
                self.prestress_perturbations
            ),
        )

        object.__setattr__(
            self,
            "replacement_measures",
            tuple(
                self.replacement_measures
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
class SystemEvolutionConfig:
    adaptive_config: AdaptiveConnectionConfig = field(
        default_factory=AdaptiveConnectionConfig
    )

    prestress_config: PrestressRedistributionConfig = field(
        default_factory=PrestressRedistributionConfig
    )

    trace_type: str = "system_evolution"
    trace_persistence: float = 1.0

    include_connection_change_in_trace: bool = True
    include_prestress_change_in_trace: bool = True

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        trace_type = _validate_id(
            "trace_type",
            self.trace_type,
        )

        trace_persistence = _validate_finite(
            "trace_persistence",
            self.trace_persistence,
        )

        if not (
            0.0
            <= trace_persistence
            <= 1.0
        ):
            raise SystemEvolutionError(
                "trace_persistence must lie in [0, 1]"
            )

        object.__setattr__(
            self,
            "trace_type",
            trace_type,
        )

        object.__setattr__(
            self,
            "trace_persistence",
            trace_persistence,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


@dataclass(frozen=True, slots=True)
class ConnectionEvolutionRecord:
    connection_id: str
    exposure_id: str
    source_transfer_gain: float
    target_transfer_gain: float
    transfer_gain_delta: float
    update_result: AdaptiveConnectionResult
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

        object.__setattr__(
            self,
            "exposure_id",
            _validate_id(
                "exposure_id",
                self.exposure_id,
            ),
        )

        for name in (
            "source_transfer_gain",
            "target_transfer_gain",
            "transfer_gain_delta",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite(
                    name,
                    getattr(
                        self,
                        name,
                    ),
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
class SystemEvolutionSummary:
    adaptive_update_count: int
    prestress_contribution_count: int
    changed_connection_count: int
    changed_prestress_node_count: int

    connection_change_norm: float
    prestress_change_norm: float
    trace_magnitude: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "adaptive_update_count",
            "prestress_contribution_count",
            "changed_connection_count",
            "changed_prestress_node_count",
        ):
            value = int(
                getattr(
                    self,
                    name,
                )
            )

            if value < 0:
                raise SystemEvolutionError(
                    f"{name} must be nonnegative"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        for name in (
            "connection_change_norm",
            "prestress_change_norm",
            "trace_magnitude",
        ):
            value = _validate_finite(
                name,
                getattr(
                    self,
                    name,
                ),
            )

            if value < 0.0:
                raise SystemEvolutionError(
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
class SystemEvolutionResult:
    evolution_id: str
    source_image: SystemImage
    event: SystemEvent

    connection_records: tuple[
        ConnectionEvolutionRecord,
        ...
    ]

    prestress_result: PrestressRedistributionResult

    generated_trace: HistoricalTrace

    target_image: SystemImage
    summary: SystemEvolutionSummary

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evolution_id",
            _validate_id(
                "evolution_id",
                self.evolution_id,
            ),
        )

        object.__setattr__(
            self,
            "connection_records",
            tuple(
                self.connection_records
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly(
                self.metadata
            ),
        )


def _index_adaptive_connections(
    connections: Sequence[
        AdaptiveConnectionState
    ],
) -> dict[
    str,
    AdaptiveConnectionState,
]:
    result = {}

    for connection in connections:
        if connection.connection_id in result:
            raise SystemEvolutionError(
                f"duplicate adaptive connection_id: "
                f"{connection.connection_id}"
            )

        result[
            connection.connection_id
        ] = connection

    return result


def _index_exposures(
    exposures: Sequence[
        ConnectionExposure
    ],
) -> dict[
    str,
    list[ConnectionExposure],
]:
    result: dict[
        str,
        list[ConnectionExposure],
    ] = {}

    for exposure in exposures:
        result.setdefault(
            exposure.connection_id,
            [],
        ).append(
            exposure
        )

    return result


def _evolve_connections(
    *,
    evolution_id: str,
    source_connections: Sequence[
        AdaptiveConnectionState
    ],
    exposures: Sequence[
        ConnectionExposure
    ],
    config: AdaptiveConnectionConfig,
) -> tuple[
    tuple[AdaptiveConnectionState, ...],
    tuple[ConnectionEvolutionRecord, ...],
]:
    indexed = _index_adaptive_connections(
        source_connections
    )

    grouped_exposures = _index_exposures(
        exposures
    )

    for connection_id in grouped_exposures:
        if connection_id not in indexed:
            raise SystemEvolutionError(
                f"unknown exposure connection_id: "
                f"{connection_id}"
            )

    target_by_id = dict(
        indexed
    )

    records = []

    for connection_id in sorted(
        grouped_exposures
    ):
        current = target_by_id[
            connection_id
        ]

        for exposure_index, exposure in enumerate(
            grouped_exposures[
                connection_id
            ],
            start=1,
        ):
            source_gain = (
                effective_transfer_gain(
                    current
                )
            )

            update_result = (
                update_adaptive_connection(
                    update_id=(
                        f"{evolution_id}"
                        f"::connection::{connection_id}"
                        f"::{exposure_index}"
                        f"::{exposure.exposure_id}"
                    ),
                    state=current,
                    exposure=exposure,
                    config=config,
                )
            )

            target = (
                update_result.target_state
            )

            target_gain = (
                effective_transfer_gain(
                    target
                )
            )

            records.append(
                ConnectionEvolutionRecord(
                    connection_id=connection_id,
                    exposure_id=exposure.exposure_id,
                    source_transfer_gain=source_gain,
                    target_transfer_gain=target_gain,
                    transfer_gain_delta=(
                        target_gain
                        - source_gain
                    ),
                    update_result=update_result,
                    metadata={
                        "schema_version": SCHEMA_VERSION,
                        "learning_applied": False,
                        "topology_modified": False,
                    },
                )
            )

            current = target

        target_by_id[
            connection_id
        ] = current

    target_connections = tuple(
        target_by_id[
            connection.connection_id
        ]
        for connection in source_connections
    )

    return (
        target_connections,
        tuple(
            records
        ),
    )


def _build_prestress_connections(
    *,
    adaptive_connections: Sequence[
        AdaptiveConnectionState
    ],
) -> tuple[
    PrestressConnection,
    ...,
]:
    """
    Convert current adaptive-connection state into explicit prestress transfer
    connections.

    The effective adaptive transfer gain is used as the transmission gain.
    This is a computational bridge, not a biological law.
    """
    return tuple(
        PrestressConnection(
            connection_id=(
                f"prestress::{connection.connection_id}"
            ),
            source_node_id=connection.source_node_id,
            target_node_id=connection.target_node_id,
            transfer_gain=effective_transfer_gain(
                connection
            ),
            attenuation=1.0,
            capacity=max(
                0.0,
                connection.contractile_capacity,
            ),
            enabled=True,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "source_adaptive_connection_id": (
                    connection.connection_id
                ),
            },
        )
        for connection in adaptive_connections
    )


def _build_modified_prestress_connections(
    *,
    adaptive_connections: Sequence[
        AdaptiveConnectionState
    ],
    transition_modifiers: TransitionModifierSet,
) -> tuple[
    PrestressConnection,
    ...,
]:
    """
    Build prestress-transfer connections with explicit bounded memory-derived
    modifiers.

    Current integration scope
    -------------------------

    Only:

        TransitionChannel.PRESTRESS_TRANSFER

    is executable in SystemEvolution at this stage.

    For one adaptive connection with ordinary transfer gain g and modifier m:

        g_modified = g * (1 + m)

    where:

        m in [-1, 1]

    Therefore:

        m =  0  -> unchanged transfer
        m = -1  -> zero transfer through the existing path
        m = +1  -> twice the existing transfer gain

    This is a local integration rule for the PRESTRESS_TRANSFER channel only.
    It is NOT a general definition of the ROIF history-conditioned transition
    operator.

    Active modifiers for unsupported channels are rejected rather than silently
    ignored.
    """

    if not isinstance(
        transition_modifiers,
        TransitionModifierSet,
    ):
        raise SystemEvolutionError(
            "transition_modifiers must be a TransitionModifierSet"
        )

    active = (
        transition_modifiers.active_modifiers
    )

    unsupported = tuple(
        modifier
        for modifier in active
        if modifier.channel
        != TransitionChannel.PRESTRESS_TRANSFER
    )

    if unsupported:
        unsupported_channels = ", ".join(
            sorted(
                {
                    modifier.channel.value
                    for modifier
                    in unsupported
                }
            )
        )

        raise SystemEvolutionError(
            "active transition modifier channels are not yet "
            f"supported by SystemEvolution: {unsupported_channels}"
        )

    adaptive_by_id = {
        connection.connection_id:
        connection
        for connection
        in adaptive_connections
    }

    for modifier in active:
        if (
            modifier.target_id
            not in adaptive_by_id
        ):
            raise SystemEvolutionError(
                "unknown PRESTRESS_TRANSFER modifier target: "
                f"{modifier.target_id}"
            )

    modifier_by_target = {
        modifier.target_id:
        modifier
        for modifier
        in active
    }

    connections = []

    for connection in adaptive_connections:
        base_gain = (
            effective_transfer_gain(
                connection
            )
        )

        modifier = (
            modifier_by_target.get(
                connection.connection_id
            )
        )

        if modifier is None:
            modified_gain = (
                base_gain
            )

            modifier_metadata = {}

        else:
            modified_gain = (
                base_gain
                * (
                    1.0
                    + modifier.value
                )
            )

            modifier_metadata = {
                "transition_modifier_set_id": (
                    transition_modifiers.modifier_set_id
                ),
                "transition_modifier_id": (
                    modifier.modifier_id
                ),
                "transition_modifier_channel": (
                    modifier.channel.value
                ),
                "transition_modifier_value": (
                    modifier.value
                ),
                "memory_source_ids": (
                    transition_modifiers.source_memory_ids
                ),
                "context_source_ids": (
                    transition_modifiers.source_context_ids
                ),
            }

        connections.append(
            PrestressConnection(
                connection_id=(
                    f"prestress::{connection.connection_id}"
                ),
                source_node_id=(
                    connection.source_node_id
                ),
                target_node_id=(
                    connection.target_node_id
                ),
                transfer_gain=(
                    modified_gain
                ),
                attenuation=1.0,
                capacity=max(
                    0.0,
                    connection.contractile_capacity,
                ),
                enabled=True,
                metadata={
                    "schema_version": (
                        SCHEMA_VERSION
                    ),
                    "source_adaptive_connection_id": (
                        connection.connection_id
                    ),
                    **modifier_metadata,
                },
            )
        )

    return tuple(
        connections
    )


def _connection_change_norm(
    records: Sequence[
        ConnectionEvolutionRecord
    ],
) -> float:
    if not records:
        return 0.0

    return sqrt(
        sum(
            record.transfer_gain_delta
            * record.transfer_gain_delta
            for record in records
        )
    )


def _prestress_change_norm(
    result: PrestressRedistributionResult,
) -> float:
    return sqrt(
        sum(
            delta * delta
            for delta in result.node_delta_by_id.values()
        )
    )


def _trace_magnitude(
    *,
    connection_change_norm: float,
    prestress_change_norm: float,
    config: SystemEvolutionConfig,
) -> float:
    parts = []

    if config.include_connection_change_in_trace:
        parts.append(
            connection_change_norm
        )

    if config.include_prestress_change_in_trace:
        parts.append(
            prestress_change_norm
        )

    if not parts:
        return 0.0

    return sqrt(
        sum(
            value * value
            for value in parts
        )
    )


def evolve_system(
    *,
    evolution_id: str,
    source_image: SystemImage,
    event: SystemEvent,
    config: SystemEvolutionConfig | None = None,
    transition_modifiers: TransitionModifierSet | None = None,
    target_image_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SystemEvolutionResult:
    """
    Apply one event to one immutable SystemImage.

    Order of operations:
    1. evolve adaptive connections from explicit connection exposures;
    2. convert updated connections into current prestress-transfer paths;
    3. apply event prestress perturbations through the updated network;
    4. generate a persistent historical trace;
    5. build the next SystemImage revision.
    """
    evolution_id = _validate_id(
        "evolution_id",
        evolution_id,
    )

    actual_config = (
        config
        if config is not None
        else SystemEvolutionConfig()
    )

    (
        target_connections,
        connection_records,
    ) = _evolve_connections(
        evolution_id=evolution_id,
        source_connections=(
            source_image.adaptive_connections
        ),
        exposures=(
            event.connection_exposures
        ),
        config=(
            actual_config.adaptive_config
        ),
    )

    if (
        transition_modifiers is None
        or transition_modifiers.is_zero
    ):
        # Exact backward-compatible path.
        #
        # None and the canonical zero modifier set deliberately use the
        # pre-integration construction unchanged.
        prestress_connections = (
            _build_prestress_connections(
                adaptive_connections=target_connections,
            )
        )

        active_transition_modifiers = None

    else:
        prestress_connections = (
            _build_modified_prestress_connections(
                adaptive_connections=target_connections,
                transition_modifiers=transition_modifiers,
            )
        )

        active_transition_modifiers = (
            transition_modifiers
        )

    prestress_result = (
        redistribute_prestress(
            redistribution_id=(
                f"{evolution_id}::prestress"
            ),
            nodes=source_image.prestress_nodes,
            connections=prestress_connections,
            perturbations=(
                event.prestress_perturbations
            ),
            config=(
                actual_config.prestress_config
            ),
            metadata={
                "source_system_image_id": (
                    source_image.image_id
                ),
                "source_event_id": (
                    event.event_id
                ),
            },
        )
    )

    connection_norm = (
        _connection_change_norm(
            connection_records
        )
    )

    prestress_norm = (
        _prestress_change_norm(
            prestress_result
        )
    )

    trace_magnitude = _trace_magnitude(
        connection_change_norm=connection_norm,
        prestress_change_norm=prestress_norm,
        config=actual_config,
    )

    generated_trace = HistoricalTrace(
        trace_id=(
            f"{evolution_id}::trace"
        ),
        source_event_id=event.event_id,
        trace_type=(
            actual_config.trace_type
        ),
        magnitude=trace_magnitude,
        persistence=(
            actual_config.trace_persistence
        ),
        relation_count=(
            len(
                connection_records
            )
            + len(
                prestress_result.contributions
            )
        ),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "source_image_id": (
                source_image.image_id
            ),
            "source_revision": (
                source_image.revision
            ),
            "connection_change_norm": (
                connection_norm
            ),
            "prestress_change_norm": (
                prestress_norm
            ),
        },
    )

    next_image_id = (
        target_image_id
        if target_image_id is not None
        else f"{source_image.image_id}::r{source_image.revision + 1}"
    )

    measures = (
        event.replacement_measures
        if event.replacement_measures
        else source_image.measures
    )

    context = (
        event.target_context
        if event.target_context is not None
        else source_image.context
    )

    target_image = build_system_image(
        image_id=next_image_id,
        revision=source_image.revision + 1,
        measures=measures,
        prestress_nodes=(
            prestress_result.target_states
        ),
        adaptive_connections=(
            target_connections
        ),
        historical_traces=(
            *source_image.historical_traces,
            generated_trace,
        ),
        context=context,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "source_image_id": (
                source_image.image_id
            ),
            "source_event_id": (
                event.event_id
            ),
            "evolved_from_previous_image": True,
            "memory_mutated": False,
            "learning_applied": False,
            "topology_modified": False,
            "action_selected": False,
            "policy_modified": False,
            "diagnosis_generated": False,
            "biological_truth_claimed": False,
            "causal_truth_inferred": False,
        },
    )

    summary = SystemEvolutionSummary(
        adaptive_update_count=len(
            connection_records
        ),
        prestress_contribution_count=len(
            prestress_result.contributions
        ),
        changed_connection_count=sum(
            1
            for record in connection_records
            if abs(
                record.transfer_gain_delta
            ) > 0.0
        ),
        changed_prestress_node_count=sum(
            1
            for delta in prestress_result.node_delta_by_id.values()
            if abs(
                delta
            ) > 0.0
        ),
        connection_change_norm=connection_norm,
        prestress_change_norm=prestress_norm,
        trace_magnitude=trace_magnitude,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "summary_mode": "explicit_multilayer_evolution",
        },
    )

    merged_metadata = {
        "schema_version": SCHEMA_VERSION,
        "system_evolution_mode": "event_to_revised_system_image",
        "source_image_id": source_image.image_id,
        "target_image_id": target_image.image_id,
        "source_event_id": event.event_id,
        "memory_mutated": False,
        "learning_applied": False,
        "topology_modified": False,
        "action_selected": False,
        "policy_modified": False,
        "diagnosis_generated": False,
        "biological_truth_claimed": False,
        "causal_truth_inferred": False,
    }

    if active_transition_modifiers is not None:
        merged_metadata.update(
            {
                "transition_modifiers_applied": True,
                "transition_modifier_set_id": (
                    active_transition_modifiers.modifier_set_id
                ),
                "active_transition_modifier_count": (
                    len(
                        active_transition_modifiers.active_modifiers
                    )
                ),
                "transition_modifier_channels": tuple(
                    channel.value
                    for channel
                    in active_transition_modifiers.channels
                ),
                "transition_modifier_source_memory_ids": (
                    active_transition_modifiers.source_memory_ids
                ),
                "transition_modifier_source_context_ids": (
                    active_transition_modifiers.source_context_ids
                ),
            }
        )

    if metadata:
        merged_metadata.update(
            dict(
                metadata
            )
        )

    return SystemEvolutionResult(
        evolution_id=evolution_id,
        source_image=source_image,
        event=event,
        connection_records=connection_records,
        prestress_result=prestress_result,
        generated_trace=generated_trace,
        target_image=target_image,
        summary=summary,
        metadata=merged_metadata,
    )


def evolve_sequence(
    *,
    sequence_id: str,
    source_image: SystemImage,
    events: Iterable[SystemEvent],
    config: SystemEvolutionConfig | None = None,
    transition_modifiers: TransitionModifierSet | None = None,
) -> tuple[
    SystemEvolutionResult,
    ...,
]:
    """
    Apply an ordered sequence of events.

    Each event acts on the image produced by the previous event.
    This is the explicit system-level realization of path dependence.
    """
    sequence_id = _validate_id(
        "sequence_id",
        sequence_id,
    )

    event_items = tuple(
        events
    )

    if not event_items:
        raise SystemEvolutionError(
            "events must not be empty"
        )

    current = source_image
    results = []

    for index, event in enumerate(
        event_items,
        start=1,
    ):
        result = evolve_system(
            evolution_id=(
                f"{sequence_id}::step_{index}::{event.event_id}"
            ),
            source_image=current,
            event=event,
            config=config,
            transition_modifiers=(
                transition_modifiers
            ),
            target_image_id=(
                f"{sequence_id}::image_{index}"
            ),
        )

        results.append(
            result
        )

        current = (
            result.target_image
        )

    return tuple(
        results
    )


def final_system_image(
    results: Sequence[
        SystemEvolutionResult
    ],
) -> SystemImage:
    if not results:
        raise SystemEvolutionError(
            "results must not be empty"
        )

    return results[
        -1
    ].target_image


def system_evolution_is_policy_free(
    result: SystemEvolutionResult,
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


def system_evolution_signature(
    result: SystemEvolutionResult,
) -> tuple[
    str,
    str,
    int,
    str,
    float,
    float,
    float,
]:
    return (
        result.evolution_id,
        result.target_image.image_id,
        result.target_image.revision,
        result.generated_trace.trace_id,
        result.summary.connection_change_norm,
        result.summary.prestress_change_norm,
        result.summary.trace_magnitude,
    )


__all__ = [
    "ConnectionEvolutionRecord",
    "SCHEMA_VERSION",
    "SystemEvent",
    "SystemEvolutionConfig",
    "SystemEvolutionError",
    "SystemEvolutionResult",
    "SystemEvolutionSummary",
    "evolve_sequence",
    "evolve_system",
    "final_system_image",
    "system_evolution_is_policy_free",
    "system_evolution_signature",
]
