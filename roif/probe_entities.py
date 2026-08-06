"""
ROIF Engine
===========

Universal domain entities for the Active Probe Engine.

This module defines domain-independent structures describing controlled
experiments over incomplete graphs of complex systems.

A Probe follows the invariant:

    BASELINE
        ->
    PERTURBATION
        ->
    REASSESSMENT
        ->
    OBSERVATION DELTA
        ->
    GRAPH UPDATE

The module contains no medical logic, planning policy, graph mutation logic,
or Solver integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_confidence(value: float, field_name: str = "confidence") -> None:
    """Validate that a confidence value belongs to the closed interval [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


def _validate_non_negative_optional(
    value: float | None,
    field_name: str,
) -> None:
    """Validate an optional non-negative numeric value."""

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number or None")

    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Return an immutable shallow copy of a mapping.

    The copy prevents later mutation through a dictionary originally supplied
    by the caller.
    """

    return MappingProxyType(dict(mapping))


def _validate_non_empty_string(value: str, field_name: str) -> None:
    """Validate a required non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


# ============================================================================
# Probe taxonomy
# ============================================================================


class ProbeMethod(str, Enum):
    """Method through which information is acquired."""

    INTERVIEW = "interview"
    OBSERVATION = "observation"
    PALPATION = "palpation"

    MANUAL_MUSCLE_TEST = "manual_muscle_test"
    MANUAL_FIXATION = "manual_fixation"
    MANUAL_MOBILIZATION = "manual_mobilization"

    FUNCTIONAL_MOVEMENT = "functional_movement"
    INSTRUMENTAL = "instrumental"
    SIMULATION = "simulation"


class ProbeRegime(str, Enum):
    """System regime during Probe execution."""

    REST = "rest"
    STATIC = "static"
    DYNAMIC = "dynamic"
    TRANSITION = "transition"
    REPETITIVE = "repetitive"


class ProbePurpose(str, Enum):
    """Primary information objective of a Probe."""

    OBSERVE = "observe"
    CONFIRM = "confirm"
    REJECT_HYPOTHESIS = "reject_hypothesis"
    LOCALIZE_SOURCE = "localize_source"
    REDUCE_UNCERTAINTY = "reduce_uncertainty"
    VALIDATE_GRAPH = "validate_graph"


class ProbePhase(str, Enum):
    """Execution phase in which an observation was acquired."""

    BASELINE = "baseline"
    PERTURBATION = "perturbation"
    REASSESSMENT = "reassessment"


class PerturbationType(str, Enum):
    """Controlled intervention applied to a system."""

    NONE = "none"

    MANUAL_STABILIZATION = "manual_stabilization"
    LOAD_CHANGE = "load_change"
    ROTATION = "rotation"
    TRACTION = "traction"
    COMPRESSION = "compression"
    EXTERNAL_SUPPORT = "external_support"
    SIMULATED_INTERVENTION = "simulated_intervention"


class ObservationChannel(str, Enum):
    """Channel through which a system response is observed."""

    PAIN = "pain"
    STABILITY = "stability"
    RANGE_OF_MOTION = "range_of_motion"
    COMPENSATION = "compensation"
    TEMPERATURE = "temperature"
    STRAIN = "strain"
    DISPLACEMENT = "displacement"
    VIBRATION = "vibration"
    SIGNAL = "signal"
    SUBJECTIVE_TENSION = "subjective_tension"


# ============================================================================
# Core entities
# ============================================================================


@dataclass(slots=True, frozen=True)
class Perturbation:
    """
    Controlled action applied to a system.

    Parameters
    ----------
    kind:
        Type of controlled intervention.
    magnitude:
        Optional intervention magnitude.
    magnitude_units:
        Units associated with magnitude.
    duration_seconds:
        Optional intervention duration expressed in seconds.
    parameters:
        Additional domain-independent intervention parameters.
    """

    kind: PerturbationType
    magnitude: float | None = None
    magnitude_units: str | None = None
    duration_seconds: float | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative_optional(self.duration_seconds, "duration_seconds")

        if self.magnitude is not None:
            if isinstance(self.magnitude, bool) or not isinstance(
                self.magnitude,
                (int, float),
            ):
                raise TypeError("magnitude must be a real number or None")

        if self.magnitude_units is not None:
            _validate_non_empty_string(
                self.magnitude_units,
                "magnitude_units",
            )

        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters),
        )


@dataclass(slots=True, frozen=True)
class Observation:
    """
    Single observation acquired during a specific Probe phase.

    Parameters
    ----------
    phase:
        Probe phase during which the observation was acquired.
    channel:
        Observable response channel.
    target:
        Node, edge, subsystem, component, region, or other observed target.
    value:
        Raw measured or reported value.
    units:
        Optional measurement units.
    confidence:
        Confidence in the observation, constrained to [0, 1].
    metadata:
        Additional immutable observation context.
    """

    phase: ProbePhase
    channel: ObservationChannel
    target: str
    value: Any
    units: str | None = None
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.target, "target")
        _validate_confidence(self.confidence)

        if self.units is not None:
            _validate_non_empty_string(self.units, "units")

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


@dataclass(slots=True, frozen=True)
class ObservationDelta:
    """
    Difference between compatible baseline and reassessment observations.

    The delta is stored explicitly because not all observation values are
    numeric. It may represent a numerical difference, category transition,
    logical state change, vector difference, or another domain-specific result.
    """

    baseline: Observation
    reassessment: Observation
    delta: Any

    def __post_init__(self) -> None:
        if self.baseline.phase is not ProbePhase.BASELINE:
            raise ValueError(
                "baseline observation must have phase ProbePhase.BASELINE"
            )

        if self.reassessment.phase is not ProbePhase.REASSESSMENT:
            raise ValueError(
                "reassessment observation must have "
                "phase ProbePhase.REASSESSMENT"
            )

        if self.baseline.channel is not self.reassessment.channel:
            raise ValueError(
                "baseline and reassessment must use the same "
                "observation channel"
            )

        if self.baseline.target != self.reassessment.target:
            raise ValueError(
                "baseline and reassessment must refer to the same target"
            )

        if self.baseline.units != self.reassessment.units:
            raise ValueError(
                "baseline and reassessment must use compatible units"
            )


@dataclass(slots=True, frozen=True)
class ProbeDefinition:
    """
    Static and reusable description of a Probe.

    A ProbeDefinition describes what experiment may be performed. It does not
    contain execution observations or planning decisions.
    """

    identifier: str
    name: str
    method: ProbeMethod
    regime: ProbeRegime
    purpose: ProbePurpose
    perturbation: Perturbation
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.identifier, "identifier")
        _validate_non_empty_string(self.name, "name")

        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

        normalized_tags: list[str] = []

        for tag in self.tags:
            _validate_non_empty_string(tag, "tag")
            normalized_tags.append(tag.strip())

        object.__setattr__(self, "tags", tuple(normalized_tags))


@dataclass(slots=True)
class ProbeExecution:
    """
    Runtime record of a Probe execution.

    Observations are grouped explicitly by phase. The class stores execution
    data only and does not calculate deltas or mutate the graph.
    """

    definition: ProbeDefinition
    baseline_observations: list[Observation] = field(default_factory=list)
    perturbation_observations: list[Observation] = field(default_factory=list)
    reassessment_observations: list[Observation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_phase_collection(
            self.baseline_observations,
            ProbePhase.BASELINE,
            "baseline_observations",
        )
        self._validate_phase_collection(
            self.perturbation_observations,
            ProbePhase.PERTURBATION,
            "perturbation_observations",
        )
        self._validate_phase_collection(
            self.reassessment_observations,
            ProbePhase.REASSESSMENT,
            "reassessment_observations",
        )

    @staticmethod
    def _validate_phase_collection(
        observations: Sequence[Observation],
        expected_phase: ProbePhase,
        field_name: str,
    ) -> None:
        for observation in observations:
            if not isinstance(observation, Observation):
                raise TypeError(
                    f"{field_name} must contain only Observation instances"
                )

            if observation.phase is not expected_phase:
                raise ValueError(
                    f"all observations in {field_name} must have phase "
                    f"{expected_phase.value!r}"
                )

    def add_observation(self, observation: Observation) -> None:
        """Add an observation to the collection matching its phase."""

        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation instance")

        if observation.phase is ProbePhase.BASELINE:
            self.baseline_observations.append(observation)
            return

        if observation.phase is ProbePhase.PERTURBATION:
            self.perturbation_observations.append(observation)
            return

        if observation.phase is ProbePhase.REASSESSMENT:
            self.reassessment_observations.append(observation)
            return

        raise ValueError(f"unsupported Probe phase: {observation.phase!r}")


@dataclass(slots=True, frozen=True)
class GraphUpdate:
    """
    Evidence package proposed for later graph refinement.

    This entity records which graph components may be affected and which
    ObservationDelta objects support the proposal.

    It intentionally does not apply graph mutations.
    """

    affected_nodes: tuple[str, ...] = ()
    affected_edges: tuple[tuple[str, str], ...] = ()
    evidence: tuple[ObservationDelta, ...] = ()
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)

        normalized_nodes: list[str] = []

        for node in self.affected_nodes:
            _validate_non_empty_string(node, "affected node")
            normalized_nodes.append(node.strip())

        normalized_edges: list[tuple[str, str]] = []

        for edge in self.affected_edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise TypeError(
                    "each affected edge must be a tuple of two node identifiers"
                )

            source, target = edge

            _validate_non_empty_string(source, "edge source")
            _validate_non_empty_string(target, "edge target")

            normalized_edges.append((source.strip(), target.strip()))

        for delta in self.evidence:
            if not isinstance(delta, ObservationDelta):
                raise TypeError(
                    "evidence must contain only ObservationDelta instances"
                )

        object.__setattr__(
            self,
            "affected_nodes",
            tuple(normalized_nodes),
        )
        object.__setattr__(
            self,
            "affected_edges",
            tuple(normalized_edges),
        )
        object.__setattr__(
            self,
            "evidence",
            tuple(self.evidence),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


@dataclass(slots=True)
class ProbeResult:
    """
    Final structured outcome of a completed Probe.

    ProbeResult combines execution data, derived observation deltas, and an
    optional graph-update proposal.
    """

    execution: ProbeExecution
    deltas: list[ObservationDelta] = field(default_factory=list)
    graph_update: GraphUpdate | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ProbeExecution):
            raise TypeError("execution must be a ProbeExecution instance")

        for delta in self.deltas:
            if not isinstance(delta, ObservationDelta):
                raise TypeError(
                    "deltas must contain only ObservationDelta instances"
                )

        if self.graph_update is not None and not isinstance(
            self.graph_update,
            GraphUpdate,
        ):
            raise TypeError(
                "graph_update must be a GraphUpdate instance or None"
            )

        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
