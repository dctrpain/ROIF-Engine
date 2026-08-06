"""
ROIF Engine
===========

Universal adapter between incomplete graph uncertainty and Active Probe Engine.

The adapter performs two transformations:

    incomplete graph uncertainty
        ->
    ProbeInformationEstimate

and:

    ProbeResult
        ->
    GraphUpdate proposal

Architectural boundaries
------------------------

The adapter:

- does not apply graph mutations;
- does not call Solver;
- does not calculate D_origin, D_fast, D_root, or Node*;
- does not execute Probes;
- does not contain medical or other domain-specific logic;
- does not depend on a concrete Network implementation.

A graph owner must explicitly review and apply every GraphUpdate proposal.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .probe_entities import (
    GraphUpdate,
    ObservationDelta,
    ProbeDefinition,
    ProbeResult,
)
from .probe_planner import ProbeInformationEstimate
from .probe_registry import (
    ProbeRegistry,
    ProbeRegistrySnapshot,
)


# ============================================================================
# Errors
# ============================================================================


class ProbeGraphAdapterError(Exception):
    """Base exception for Probe Graph Adapter failures."""


class GraphTargetNotFoundError(ProbeGraphAdapterError):
    """Raised when a Probe mapping references an unknown graph target."""


class ProbeGraphLinkNotFoundError(ProbeGraphAdapterError):
    """Raised when no graph link exists for a Probe."""


class GraphVersionMismatchError(ProbeGraphAdapterError):
    """Raised when a result targets an incompatible graph version."""


# ============================================================================
# Taxonomy
# ============================================================================


class GraphTargetKind(str, Enum):
    """Kind of graph entity referenced by uncertainty or evidence."""

    NODE = "node"
    EDGE = "edge"


class GraphEvidenceDirection(str, Enum):
    """Direction in which Probe evidence changes graph uncertainty."""

    REDUCE_UNCERTAINTY = "reduce_uncertainty"
    INCREASE_UNCERTAINTY = "increase_uncertainty"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_nonempty_text(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    return normalized


def _validate_probability(
    value: float,
    field_name: str,
) -> float:
    """Validate and normalize a finite value in [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0.0 and 1.0"
        )

    return normalized


def _validate_nonnegative(
    value: float,
    field_name: str,
) -> float:
    """Validate and normalize a finite non-negative value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")

    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative")

    return normalized


def _freeze_mapping(
    value: Mapping[str, Any],
    field_name: str = "metadata",
) -> Mapping[str, Any]:
    """Return an immutable shallow mapping copy."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")

    return MappingProxyType(dict(value))


# ============================================================================
# Graph references
# ============================================================================


@dataclass(frozen=True, slots=True)
class GraphTarget:
    """
    Universal reference to one node or directed edge.

    Node target
    -----------

    kind = NODE
    identifier = node identifier

    Edge target
    -----------

    kind = EDGE
    identifier = stable edge identifier
    source = source node identifier
    target = target node identifier
    """

    kind: GraphTargetKind
    identifier: str

    source: str | None = None
    target: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GraphTargetKind):
            raise TypeError("kind must be GraphTargetKind")

        object.__setattr__(
            self,
            "identifier",
            _validate_nonempty_text(
                self.identifier,
                "identifier",
            ),
        )

        if self.kind is GraphTargetKind.NODE:
            if self.source is not None or self.target is not None:
                raise ValueError(
                    "node target must not define source or target"
                )

        elif self.kind is GraphTargetKind.EDGE:
            if self.source is None or self.target is None:
                raise ValueError(
                    "edge target must define source and target"
                )

            object.__setattr__(
                self,
                "source",
                _validate_nonempty_text(
                    self.source,
                    "source",
                ),
            )
            object.__setattr__(
                self,
                "target",
                _validate_nonempty_text(
                    self.target,
                    "target",
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class GraphUncertainty:
    """Uncertainty state associated with one graph target."""

    target: GraphTarget

    uncertainty: float
    importance: float = 1.0
    confidence: float = 1.0

    hypothesis_count: int = 1

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.target, GraphTarget):
            raise TypeError("target must be GraphTarget")

        object.__setattr__(
            self,
            "uncertainty",
            _validate_probability(
                self.uncertainty,
                "uncertainty",
            ),
        )

        object.__setattr__(
            self,
            "importance",
            _validate_nonnegative(
                self.importance,
                "importance",
            ),
        )

        object.__setattr__(
            self,
            "confidence",
            _validate_probability(
                self.confidence,
                "confidence",
            ),
        )

        if (
            isinstance(self.hypothesis_count, bool)
            or not isinstance(self.hypothesis_count, int)
        ):
            raise TypeError("hypothesis_count must be an integer")

        if self.hypothesis_count < 1:
            raise ValueError(
                "hypothesis_count must be at least 1"
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


# ============================================================================
# Incomplete graph snapshot
# ============================================================================


@dataclass(frozen=True, slots=True)
class IncompleteGraphSnapshot:
    """
    Immutable domain-independent uncertainty snapshot.

    This is not the graph itself. It is only the uncertainty view required
    by the Active Probe planning layer.
    """

    graph_id: str
    version: str

    uncertainties: tuple[GraphUncertainty, ...]

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_id",
            _validate_nonempty_text(
                self.graph_id,
                "graph_id",
            ),
        )

        object.__setattr__(
            self,
            "version",
            _validate_nonempty_text(
                self.version,
                "version",
            ),
        )

        uncertainties = tuple(self.uncertainties)

        identifiers: list[str] = []

        for uncertainty in uncertainties:
            if not isinstance(uncertainty, GraphUncertainty):
                raise TypeError(
                    "uncertainties must contain "
                    "GraphUncertainty instances"
                )

            identifiers.append(
                uncertainty.target.identifier
            )

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "uncertainties contain duplicate target identifiers"
            )

        object.__setattr__(
            self,
            "uncertainties",
            uncertainties,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )

    def __len__(self) -> int:
        return len(self.uncertainties)

    def get(
        self,
        target_identifier: str,
    ) -> GraphUncertainty:
        """Return uncertainty for one graph target."""

        normalized = _validate_nonempty_text(
            target_identifier,
            "target_identifier",
        )

        for uncertainty in self.uncertainties:
            if uncertainty.target.identifier == normalized:
                return uncertainty

        raise GraphTargetNotFoundError(
            f"graph target not found: {normalized!r}"
        )

    @property
    def total_weight(self) -> float:
        """Return total importance weight of the snapshot."""

        return sum(
            uncertainty.importance
            for uncertainty in self.uncertainties
        )

    @property
    def total_uncertainty_burden(self) -> float:
        """Return weighted graph uncertainty burden."""

        return sum(
            uncertainty.uncertainty
            * uncertainty.importance
            for uncertainty in self.uncertainties
        )


# ============================================================================
# Probe-to-graph mapping
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbeGraphTarget:
    """One graph target informed by a particular Probe."""

    target_identifier: str

    sensitivity: float = 1.0
    discrimination: float = 1.0
    expected_uncertainty_reduction: float = 1.0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_identifier",
            _validate_nonempty_text(
                self.target_identifier,
                "target_identifier",
            ),
        )

        for field_name in (
            "sensitivity",
            "discrimination",
            "expected_uncertainty_reduction",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_probability(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ProbeGraphLink:
    """
    Static mapping between one Probe and graph uncertainty targets.
    """

    probe_identifier: str
    targets: tuple[ProbeGraphTarget, ...]

    novelty: float = 0.5
    feasibility: float = 1.0
    estimate_confidence: float = 1.0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probe_identifier",
            _validate_nonempty_text(
                self.probe_identifier,
                "probe_identifier",
            ),
        )

        targets = tuple(self.targets)

        identifiers: list[str] = []

        for target in targets:
            if not isinstance(target, ProbeGraphTarget):
                raise TypeError(
                    "targets must contain ProbeGraphTarget instances"
                )

            identifiers.append(target.target_identifier)

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "ProbeGraphLink contains duplicate target identifiers"
            )

        object.__setattr__(self, "targets", targets)

        for field_name in (
            "novelty",
            "feasibility",
            "estimate_confidence",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_probability(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


# ============================================================================
# Adapter configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbeGraphAdapterConfig:
    """Configuration controlling estimate and update generation."""

    strict_target_resolution: bool = True
    require_graph_version_match: bool = True

    default_link_novelty: float = 0.0
    maximum_hypothesis_count: int = 5

    delta_magnitude_scale: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "strict_target_resolution",
            "require_graph_version_match",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        object.__setattr__(
            self,
            "default_link_novelty",
            _validate_probability(
                self.default_link_novelty,
                "default_link_novelty",
            ),
        )

        if (
            isinstance(self.maximum_hypothesis_count, bool)
            or not isinstance(self.maximum_hypothesis_count, int)
        ):
            raise TypeError(
                "maximum_hypothesis_count must be an integer"
            )

        if self.maximum_hypothesis_count < 2:
            raise ValueError(
                "maximum_hypothesis_count must be at least 2"
            )

        object.__setattr__(
            self,
            "delta_magnitude_scale",
            _validate_nonnegative(
                self.delta_magnitude_scale,
                "delta_magnitude_scale",
            ),
        )

        if self.delta_magnitude_scale == 0.0:
            raise ValueError(
                "delta_magnitude_scale must be greater than zero"
            )


# ============================================================================
# Evidence impact
# ============================================================================


@dataclass(frozen=True, slots=True)
class GraphEvidenceImpact:
    """Auditable interpretation of one Probe delta."""

    delta: ObservationDelta
    direction: GraphEvidenceDirection

    normalized_magnitude: float
    confidence: float

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.delta, ObservationDelta):
            raise TypeError(
                "delta must be ObservationDelta"
            )

        if not isinstance(
            self.direction,
            GraphEvidenceDirection,
        ):
            raise TypeError(
                "direction must be GraphEvidenceDirection"
            )

        object.__setattr__(
            self,
            "normalized_magnitude",
            _validate_probability(
                self.normalized_magnitude,
                "normalized_magnitude",
            ),
        )

        object.__setattr__(
            self,
            "confidence",
            _validate_probability(
                self.confidence,
                "confidence",
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class GraphUpdateProposal:
    """
    Auditable graph-update proposal created from Probe evidence.

    The proposal wraps GraphUpdate but does not apply it.
    """

    graph_id: str
    base_version: str

    probe_identifier: str
    graph_update: GraphUpdate

    impacts: tuple[GraphEvidenceImpact, ...]

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_id",
            _validate_nonempty_text(
                self.graph_id,
                "graph_id",
            ),
        )

        object.__setattr__(
            self,
            "base_version",
            _validate_nonempty_text(
                self.base_version,
                "base_version",
            ),
        )

        object.__setattr__(
            self,
            "probe_identifier",
            _validate_nonempty_text(
                self.probe_identifier,
                "probe_identifier",
            ),
        )

        if not isinstance(self.graph_update, GraphUpdate):
            raise TypeError(
                "graph_update must be GraphUpdate"
            )

        impacts = tuple(self.impacts)

        for impact in impacts:
            if not isinstance(impact, GraphEvidenceImpact):
                raise TypeError(
                    "impacts must contain GraphEvidenceImpact instances"
                )

        object.__setattr__(self, "impacts", impacts)

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


# ============================================================================
# Main adapter
# ============================================================================


class ProbeGraphAdapter:
    """
    Universal adapter between graph uncertainty and Active Probe structures.
    """

    __slots__ = (
        "_links",
        "_config",
    )

    def __init__(
        self,
        links: (
            Iterable[ProbeGraphLink]
            | Mapping[str, ProbeGraphLink]
            | None
        ) = None,
        *,
        config: ProbeGraphAdapterConfig | None = None,
    ) -> None:
        if config is None:
            config = ProbeGraphAdapterConfig()
        elif not isinstance(config, ProbeGraphAdapterConfig):
            raise TypeError(
                "config must be ProbeGraphAdapterConfig or None"
            )

        self._config = config
        self._links = self._normalize_links(links)

    @property
    def config(self) -> ProbeGraphAdapterConfig:
        return self._config

    @property
    def links(self) -> Mapping[str, ProbeGraphLink]:
        return MappingProxyType(dict(self._links))

    def get_link(
        self,
        probe_identifier: str,
    ) -> ProbeGraphLink:
        """Return graph mapping for one Probe."""

        normalized = _validate_nonempty_text(
            probe_identifier,
            "probe_identifier",
        )

        try:
            return self._links[normalized]
        except KeyError as error:
            raise ProbeGraphLinkNotFoundError(
                f"Probe graph link not found: {normalized!r}"
            ) from error

    def estimate_probe(
        self,
        definition: ProbeDefinition,
        snapshot: IncompleteGraphSnapshot,
    ) -> ProbeInformationEstimate:
        """Create one planning estimate from graph uncertainty."""

        if not isinstance(definition, ProbeDefinition):
            raise TypeError(
                "definition must be ProbeDefinition"
            )

        if not isinstance(snapshot, IncompleteGraphSnapshot):
            raise TypeError(
                "snapshot must be IncompleteGraphSnapshot"
            )

        link = self._links.get(definition.identifier)

        if link is None:
            return ProbeInformationEstimate(
                probe_identifier=definition.identifier,
                novelty=self._config.default_link_novelty,
                confidence=0.0,
                metadata={
                    "graph_id": snapshot.graph_id,
                    "graph_version": snapshot.version,
                    "mapping_status": "missing",
                },
            )

        resolved = self._resolve_targets(
            link,
            snapshot,
        )

        if not resolved:
            return ProbeInformationEstimate(
                probe_identifier=definition.identifier,
                novelty=link.novelty,
                feasibility=link.feasibility,
                confidence=0.0,
                metadata={
                    "graph_id": snapshot.graph_id,
                    "graph_version": snapshot.version,
                    "mapping_status": "empty",
                },
            )

        total_graph_weight = snapshot.total_weight

        if total_graph_weight <= 0.0:
            total_graph_weight = 1.0

        information_sum = 0.0
        reduction_sum = 0.0
        discrimination_sum = 0.0
        coverage_weight = 0.0
        confidence_sum = 0.0
        target_weight_sum = 0.0

        for mapping, uncertainty in resolved:
            target_weight = uncertainty.importance

            information_sum += (
                uncertainty.uncertainty
                * target_weight
                * mapping.sensitivity
            )

            reduction_sum += (
                uncertainty.uncertainty
                * target_weight
                * mapping.expected_uncertainty_reduction
            )

            normalized_hypotheses = min(
                1.0,
                max(
                    0.0,
                    (
                        uncertainty.hypothesis_count - 1
                    )
                    / (
                        self._config.maximum_hypothesis_count - 1
                    ),
                ),
            )

            discrimination_sum += (
                normalized_hypotheses
                * target_weight
                * mapping.discrimination
            )

            coverage_weight += target_weight
            confidence_sum += (
                uncertainty.confidence
                * target_weight
            )
            target_weight_sum += target_weight

        denominator = max(
            snapshot.total_uncertainty_burden,
            1e-12,
        )

        information_gain = min(
            1.0,
            information_sum / denominator,
        )

        uncertainty_reduction = min(
            1.0,
            reduction_sum / denominator,
        )

        hypothesis_discrimination = min(
            1.0,
            discrimination_sum
            / max(target_weight_sum, 1e-12),
        )

        graph_coverage = min(
            1.0,
            coverage_weight / total_graph_weight,
        )

        target_confidence = min(
            1.0,
            confidence_sum
            / max(target_weight_sum, 1e-12),
        )

        estimate_confidence = (
            target_confidence
            * link.estimate_confidence
        )

        return ProbeInformationEstimate(
            probe_identifier=definition.identifier,
            information_gain=information_gain,
            uncertainty_reduction=uncertainty_reduction,
            hypothesis_discrimination=(
                hypothesis_discrimination
            ),
            graph_coverage=graph_coverage,
            novelty=link.novelty,
            feasibility=link.feasibility,
            confidence=estimate_confidence,
            metadata={
                "graph_id": snapshot.graph_id,
                "graph_version": snapshot.version,
                "mapped_target_count": len(resolved),
                "mapping_status": "resolved",
            },
        )

    def estimate_all(
        self,
        definitions: (
            Iterable[ProbeDefinition]
            | ProbeRegistry
            | ProbeRegistrySnapshot
        ),
        snapshot: IncompleteGraphSnapshot,
    ) -> tuple[ProbeInformationEstimate, ...]:
        """Create deterministic estimates for multiple Probe definitions."""

        if not isinstance(snapshot, IncompleteGraphSnapshot):
            raise TypeError(
                "snapshot must be IncompleteGraphSnapshot"
            )

        if isinstance(definitions, ProbeRegistry):
            normalized = definitions.definitions()

        elif isinstance(definitions, ProbeRegistrySnapshot):
            normalized = definitions.definitions

        else:
            if isinstance(definitions, (str, bytes)):
                raise TypeError(
                    "definitions must be an iterable "
                    "of ProbeDefinition instances"
                )

            normalized = tuple(definitions)

        for definition in normalized:
            if not isinstance(definition, ProbeDefinition):
                raise TypeError(
                    "definitions must contain only "
                    "ProbeDefinition instances"
                )

        return tuple(
            self.estimate_probe(
                definition,
                snapshot,
            )
            for definition in normalized
        )

    def propose_graph_update(
        self,
        result: ProbeResult,
        snapshot: IncompleteGraphSnapshot,
        *,
        graph_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GraphUpdateProposal:
        """
        Convert ProbeResult into an auditable GraphUpdate proposal.

        The returned proposal is never applied automatically.
        """

        if not isinstance(result, ProbeResult):
            raise TypeError("result must be ProbeResult")

        if not isinstance(snapshot, IncompleteGraphSnapshot):
            raise TypeError(
                "snapshot must be IncompleteGraphSnapshot"
            )

        if graph_version is not None:
            normalized_version = _validate_nonempty_text(
                graph_version,
                "graph_version",
            )

            if (
                self._config.require_graph_version_match
                and normalized_version != snapshot.version
            ):
                raise GraphVersionMismatchError(
                    "graph version does not match uncertainty snapshot"
                )

        definition = result.execution.definition
        link = self.get_link(definition.identifier)

        resolved = self._resolve_targets(
            link,
            snapshot,
        )

        affected_nodes: list[str] = []
        affected_edges: list[tuple[str, str]] = []

        target_confidences: list[float] = []

        for _, uncertainty in resolved:
            target = uncertainty.target

            target_confidences.append(
                uncertainty.confidence
            )

            if target.kind is GraphTargetKind.NODE:
                affected_nodes.append(target.identifier)

            else:
                assert target.source is not None
                assert target.target is not None

                affected_edges.append(
                    (
                        target.source,
                        target.target,
                    )
                )

        impacts = tuple(
            self._interpret_delta(delta)
            for delta in result.deltas
        )

        evidence_confidence = self._evidence_confidence(
            impacts
        )

        if target_confidences:
            target_confidence = (
                sum(target_confidences)
                / len(target_confidences)
            )
        else:
            target_confidence = 0.0

        proposal_confidence = min(
            1.0,
            evidence_confidence
            * target_confidence
            * link.estimate_confidence,
        )

        graph_update = GraphUpdate(
            affected_nodes=tuple(dict.fromkeys(affected_nodes)),
            affected_edges=tuple(dict.fromkeys(affected_edges)),
            evidence=tuple(result.deltas),
            confidence=proposal_confidence,
            metadata={
                "graph_id": snapshot.graph_id,
                "base_version": snapshot.version,
                "probe_identifier": definition.identifier,
                "application_status": "proposal_only",
            },
        )

        proposal_metadata: dict[str, Any] = {
            "automatic_application": False,
            "mapped_target_count": len(resolved),
        }

        if metadata is not None:
            proposal_metadata.update(
                dict(_freeze_mapping(metadata))
            )

        return GraphUpdateProposal(
            graph_id=snapshot.graph_id,
            base_version=snapshot.version,
            probe_identifier=definition.identifier,
            graph_update=graph_update,
            impacts=impacts,
            metadata=proposal_metadata,
        )

    def _resolve_targets(
        self,
        link: ProbeGraphLink,
        snapshot: IncompleteGraphSnapshot,
    ) -> tuple[
        tuple[ProbeGraphTarget, GraphUncertainty],
        ...,
    ]:
        """Resolve one Probe link against an uncertainty snapshot."""

        resolved: list[
            tuple[ProbeGraphTarget, GraphUncertainty]
        ] = []

        for target_mapping in link.targets:
            try:
                uncertainty = snapshot.get(
                    target_mapping.target_identifier
                )
            except GraphTargetNotFoundError:
                if self._config.strict_target_resolution:
                    raise

                continue

            resolved.append(
                (
                    target_mapping,
                    uncertainty,
                )
            )

        return tuple(resolved)

    def _interpret_delta(
        self,
        delta: ObservationDelta,
    ) -> GraphEvidenceImpact:
        """Interpret one delta without domain-specific assumptions."""

        value = delta.delta

        direction = GraphEvidenceDirection.UNKNOWN
        magnitude = 0.0

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            numeric_value = float(value)

            magnitude = min(
                1.0,
                abs(numeric_value)
                / self._config.delta_magnitude_scale,
            )

            if numeric_value == 0.0:
                direction = GraphEvidenceDirection.NEUTRAL
            else:
                direction = (
                    GraphEvidenceDirection.REDUCE_UNCERTAINTY
                )

        baseline_confidence = delta.baseline.confidence
        reassessment_confidence = delta.reassessment.confidence

        confidence = (
            baseline_confidence
            + reassessment_confidence
        ) / 2.0

        return GraphEvidenceImpact(
            delta=delta,
            direction=direction,
            normalized_magnitude=magnitude,
            confidence=confidence,
        )

    @staticmethod
    def _evidence_confidence(
        impacts: tuple[GraphEvidenceImpact, ...],
    ) -> float:
        """Aggregate confidence across Probe evidence impacts."""

        if not impacts:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for impact in impacts:
            weight = max(
                impact.normalized_magnitude,
                1e-6,
            )

            weighted_sum += impact.confidence * weight
            total_weight += weight

        return min(
            1.0,
            weighted_sum / total_weight,
        )

    @staticmethod
    def _normalize_links(
        links: (
            Iterable[ProbeGraphLink]
            | Mapping[str, ProbeGraphLink]
            | None
        ),
    ) -> dict[str, ProbeGraphLink]:
        """Normalize Probe graph links into a deterministic mapping."""

        if links is None:
            return {}

        normalized: dict[str, ProbeGraphLink] = {}

        if isinstance(links, Mapping):
            for identifier, link in links.items():
                normalized_identifier = _validate_nonempty_text(
                    identifier,
                    "link identifier",
                )

                if not isinstance(link, ProbeGraphLink):
                    raise TypeError(
                        "link mapping values must be ProbeGraphLink"
                    )

                if link.probe_identifier != normalized_identifier:
                    raise ProbeGraphAdapterError(
                        "link mapping key does not match "
                        "link.probe_identifier"
                    )

                normalized[normalized_identifier] = link

            return normalized

        if isinstance(links, (str, bytes)):
            raise TypeError(
                "links must be a mapping or iterable "
                "of ProbeGraphLink instances"
            )

        for link in links:
            if not isinstance(link, ProbeGraphLink):
                raise TypeError(
                    "links must contain only ProbeGraphLink instances"
                )

            identifier = link.probe_identifier

            if identifier in normalized:
                raise ProbeGraphAdapterError(
                    f"duplicate Probe graph link: {identifier!r}"
                )

            normalized[identifier] = link

        return normalized


__all__ = [
    "GraphEvidenceDirection",
    "GraphEvidenceImpact",
    "GraphTarget",
    "GraphTargetKind",
    "GraphTargetNotFoundError",
    "GraphUncertainty",
    "GraphUpdateProposal",
    "GraphVersionMismatchError",
    "IncompleteGraphSnapshot",
    "ProbeGraphAdapter",
    "ProbeGraphAdapterConfig",
    "ProbeGraphAdapterError",
    "ProbeGraphLink",
    "ProbeGraphLinkNotFoundError",
    "ProbeGraphTarget",
]
