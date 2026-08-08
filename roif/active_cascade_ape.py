"""
ROIF Active Cascade в†” Active Probe Engine Bridge

Purpose
-------
This module connects the generic orchestration layer in:

    roif.active_cascade

with the existing Active Probe Engine (APE) stack.

The bridge implements two responsibilities required by ActiveCascadeExplorer:

1. ProbePlannerBridge
       unresolved local cascade transition
           в†“
       candidate outgoing relations
           в†“
       APE information estimation
           в†“
       ProbePlanner
           в†“
       ProbePlanResult / Probe*

2. GraphUpdateResolver
       authorized Probe result / graph update evidence
           в†“
       match evidence against current candidate relations
           в†“
       confirm / reject candidate edge(s)
           в†“
       ActiveCascadeExplorer continues from the same node or advances

Important architectural boundary
--------------------------------
This module does NOT:
- invent Probe definitions;
- mutate the ROIF graph;
- authorize GraphUpdateProposal;
- fabricate observations;
- use hidden validation labels;
- silently choose a cascade edge from passive score alone.

GraphUpdateProposal != authorization.

The caller remains responsible for:
- executing the selected Probe;
- collecting observations;
- completing the Probe lifecycle;
- authorizing any proposed graph update;
- applying the authorized graph update to the real graph/snapshot.

The bridge then interprets the already-authorized result for the local
cascade-transition problem.

Design goal
-----------
Keep the bridge tolerant of current APE API evolution by depending on
public behavior rather than private implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from roif.active_cascade import (
    ActiveCascadeError,
    ActiveCascadeState,
    CandidateTransition,
    EdgeEvidenceState,
)

from roif.probe_entities import (
    ProbeDefinition,
)

from roif.probe_registry import (
    ProbeRegistry,
)

from roif.probe_graph_adapter import (
    ProbeGraphAdapter,
)

from roif.probe_planner import (
    ProbePlanner,
)

from roif.active_probe_engine import (
    ActiveProbeEngine,
)


# =============================================================================
# Errors
# =============================================================================


class ActiveCascadeAPEError(ActiveCascadeError):
    """Raised when the Active Cascade в†” APE bridge cannot resolve a step."""


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ActiveCascadeAPEConfig:
    """
    Bridge configuration.

    min_confirmation_confidence
        Minimum evidence confidence required before a Probe result may
        confirm a candidate cascade relation.

    reject_on_explicit_negative
        If True, explicit negative/rejected graph-update evidence may
        reject a candidate relation.

    require_exact_relation_target
        If True, only graph-update evidence explicitly naming the current
        candidate relation/edge may resolve it.

    allow_probe_without_target_link
        If False, Probe planning fails when none of the estimated Probes
        map to the unresolved candidate relations.

    metadata
        Read-only bridge metadata.
    """

    min_confirmation_confidence: float = 0.80
    reject_on_explicit_negative: bool = True
    require_exact_relation_target: bool = True
    allow_probe_without_target_link: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = float(
            self.min_confirmation_confidence
        )

        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ActiveCascadeAPEError(
                "min_confirmation_confidence must be in [0, 1]."
            )

        object.__setattr__(
            self,
            "min_confirmation_confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


# =============================================================================
# Probe planning result wrapper
# =============================================================================


@dataclass(frozen=True, slots=True)
class ActiveCascadeProbePlan:
    """
    Audit wrapper returned to ActiveCascadeExplorer/caller.

    The original APE objects remain available unchanged.
    """

    current_node_id: str
    candidate_relation_ids: tuple[str, ...]
    estimates: tuple[Any, ...]
    plan_result: Any
    selected_probe_identifier: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node = str(
            self.current_node_id
        ).strip()

        if not node:
            raise ActiveCascadeAPEError(
                "current_node_id must not be empty."
            )

        object.__setattr__(
            self,
            "current_node_id",
            node,
        )
        object.__setattr__(
            self,
            "candidate_relation_ids",
            tuple(self.candidate_relation_ids),
        )
        object.__setattr__(
            self,
            "estimates",
            tuple(self.estimates),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


# =============================================================================
# Public evidence model
# =============================================================================


@dataclass(frozen=True, slots=True)
class AuthorizedRelationEvidence:
    """
    Explicit authorized evidence for one candidate relation.

    This small neutral object is useful when the caller wants to avoid
    coupling GraphUpdateResolver to one specific GraphUpdateProposal schema.

    relation_id
        Candidate relation identifier used by Active Cascade.

    confirmed
        True  -> evidence supports the relation.
        False -> evidence rejects the relation.

    confidence
        Confidence of the authorized evidence.

    source_id / target_id
        Optional edge identity guard.

    metadata
        Read-only audit information.
    """

    relation_id: str
    confirmed: bool
    confidence: float
    source_id: str | None = None
    target_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relation_id = str(
            self.relation_id
        ).strip()

        if not relation_id:
            raise ActiveCascadeAPEError(
                "relation_id must not be empty."
            )

        confidence = float(
            self.confidence
        )

        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ActiveCascadeAPEError(
                "confidence must be in [0, 1]."
            )

        object.__setattr__(
            self,
            "relation_id",
            relation_id,
        )
        object.__setattr__(
            self,
            "confidence",
            confidence,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


# =============================================================================
# Helpers
# =============================================================================


def _definitions_from_registry(
    registry: ProbeRegistry,
) -> tuple[ProbeDefinition, ...]:
    """
    Use only public ProbeRegistry API.
    """

    definitions = registry.definitions()

    return tuple(definitions)


def _candidate_relation_ids(
    candidates: Iterable[CandidateTransition],
) -> tuple[str, ...]:
    return tuple(
        candidate.relation_id
        for candidate in candidates
        if candidate.evidence_state
        is not EdgeEvidenceState.REJECTED
    )


def _extract_probe_identifier(
    plan_result: Any,
) -> str | None:
    """
    Best-effort public-result extraction without touching private fields.

    Supports likely public shapes:
    - result.selected.identifier
    - result.selected_probe.identifier
    - result.probe.identifier
    - result.selected_identifier
    - result.probe_identifier
    """

    direct_names = (
        "selected_identifier",
        "probe_identifier",
    )

    for name in direct_names:
        value = getattr(
            plan_result,
            name,
            None,
        )

        if isinstance(value, str) and value:
            return value

    object_names = (
        "selected",
        "selected_probe",
        "probe",
    )

    for name in object_names:
        value = getattr(
            plan_result,
            name,
            None,
        )

        identifier = getattr(
            value,
            "identifier",
            None,
        )

        if isinstance(identifier, str) and identifier:
            return identifier

        # ProbePlanner currently exposes the selected Probe as:
        #
        #     ProbePlanResult.selected
        #         -> ProbePlanCandidate.definition
        #         -> ProbeDefinition.identifier
        #
        # Keep support for the simpler public result shapes above,
        # while also handling the real planner result without touching
        # private implementation fields.
        definition = getattr(
            value,
            "definition",
            None,
        )

        definition_identifier = getattr(
            definition,
            "identifier",
            None,
        )

        if (
            isinstance(definition_identifier, str)
            and definition_identifier
        ):
            return definition_identifier

    return None


def _estimate_probe_identifier(
    estimate: Any,
) -> str | None:
    for name in (
        "probe_identifier",
        "identifier",
    ):
        value = getattr(
            estimate,
            name,
            None,
        )

        if isinstance(value, str) and value:
            return value

    definition = getattr(
        estimate,
        "definition",
        None,
    )

    identifier = getattr(
        definition,
        "identifier",
        None,
    )

    if isinstance(identifier, str) and identifier:
        return identifier

    return None


def _estimate_target_ids(
    adapter: ProbeGraphAdapter,
    estimate: Any,
) -> tuple[str, ...]:
    """
    Resolve graph targets through public ProbeGraphAdapter.get_link().
    """

    probe_id = _estimate_probe_identifier(
        estimate
    )

    if probe_id is None:
        return ()

    try:
        link = adapter.get_link(
            probe_id
        )
    except Exception:
        return ()

    targets = getattr(
        link,
        "targets",
        (),
    )

    result: list[str] = []

    for target in targets:
        identifier = getattr(
            target,
            "target_identifier",
            None,
        )

        if isinstance(identifier, str) and identifier:
            result.append(identifier)

    return tuple(result)


def _filter_estimates_for_candidates(
    *,
    adapter: ProbeGraphAdapter,
    estimates: Sequence[Any],
    candidate_relation_ids: tuple[str, ...],
) -> tuple[Any, ...]:
    """
    Retain estimates whose public ProbeGraphLink targets at least one
    unresolved candidate relation.
    """

    candidate_ids = set(
        candidate_relation_ids
    )

    matched: list[Any] = []

    for estimate in estimates:
        target_ids = set(
            _estimate_target_ids(
                adapter,
                estimate,
            )
        )

        if target_ids & candidate_ids:
            matched.append(
                estimate
            )

    return tuple(matched)


def _coerce_authorized_evidence(
    graph_update: Any,
) -> tuple[AuthorizedRelationEvidence, ...]:
    """
    Convert several safe public representations into relation evidence.

    Supported inputs:
    - AuthorizedRelationEvidence
    - iterable of AuthorizedRelationEvidence
    - object with `relation_evidence`
    - mapping containing `relation_evidence`

    This function intentionally does NOT infer confirmation merely because
    an edge appears in `affected_edges`. Affected != causally confirmed.
    """

    if isinstance(
        graph_update,
        AuthorizedRelationEvidence,
    ):
        return (graph_update,)

    if isinstance(
        graph_update,
        Mapping,
    ):
        raw = graph_update.get(
            "relation_evidence"
        )

        if raw is None:
            return ()

        return _coerce_authorized_evidence(
            raw
        )

    raw = getattr(
        graph_update,
        "relation_evidence",
        None,
    )

    if raw is not None:
        return _coerce_authorized_evidence(
            raw
        )

    if isinstance(
        graph_update,
        Iterable,
    ) and not isinstance(
        graph_update,
        (str, bytes),
    ):
        result: list[
            AuthorizedRelationEvidence
        ] = []

        for item in graph_update:
            if not isinstance(
                item,
                AuthorizedRelationEvidence,
            ):
                raise ActiveCascadeAPEError(
                    "iterable graph_update evidence must contain only "
                    "AuthorizedRelationEvidence objects."
                )

            result.append(item)

        return tuple(result)

    return ()


# =============================================================================
# Real APE planner bridge
# =============================================================================


class APEProbePlannerBridge:
    """
    Real ProbePlannerBridge implementation backed by existing APE objects.
    """

    def __init__(
        self,
        *,
        registry: ProbeRegistry,
        adapter: ProbeGraphAdapter,
        planner: ProbePlanner,
        engine: ActiveProbeEngine | None = None,
        config: ActiveCascadeAPEConfig | None = None,
    ) -> None:
        self._registry = registry
        self._adapter = adapter
        self._planner = planner
        self._engine = engine
        self._config = (
            config
            or ActiveCascadeAPEConfig()
        )

    @property
    def registry(
        self,
    ) -> ProbeRegistry:
        return self._registry

    @property
    def adapter(
        self,
    ) -> ProbeGraphAdapter:
        return self._adapter

    @property
    def planner(
        self,
    ) -> ProbePlanner:
        return self._planner

    @property
    def engine(
        self,
    ) -> ActiveProbeEngine | None:
        return self._engine

    @property
    def config(
        self,
    ) -> ActiveCascadeAPEConfig:
        return self._config

    def estimate_candidates(
        self,
        *,
        candidates: tuple[CandidateTransition, ...],
        graph_snapshot: Any,
    ) -> tuple[Any, ...]:
        """
        Estimate all registered Probes, then retain those targeting the
        unresolved local cascade relations.
        """

        definitions = _definitions_from_registry(
            self._registry
        )

        estimates = tuple(
            self._adapter.estimate_all(
                definitions,
                graph_snapshot,
            )
        )

        relation_ids = _candidate_relation_ids(
            candidates
        )

        matched = _filter_estimates_for_candidates(
            adapter=self._adapter,
            estimates=estimates,
            candidate_relation_ids=relation_ids,
        )

        if (
            not matched
            and self._config.allow_probe_without_target_link
        ):
            return estimates

        return matched

    def plan_probe(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> ActiveCascadeProbePlan:
        """
        Estimate and plan Probe* for the unresolved local transition.
        """

        relation_ids = _candidate_relation_ids(
            candidates
        )

        if not relation_ids:
            raise ActiveCascadeAPEError(
                "no viable candidate relations available for Probe planning."
            )

        estimates = self.estimate_candidates(
            candidates=candidates,
            graph_snapshot=graph_snapshot,
        )

        if not estimates:
            raise ActiveCascadeAPEError(
                "no registered Probe targets the unresolved candidate "
                "relations."
            )

        plan_result = self._planner.plan(
            estimates
        )

        selected_id = _extract_probe_identifier(
            plan_result
        )

        return ActiveCascadeProbePlan(
            current_node_id=current_node_id,
            candidate_relation_ids=relation_ids,
            estimates=estimates,
            plan_result=plan_result,
            selected_probe_identifier=selected_id,
            metadata={
                "graph_version": (
                    state.graph_version
                ),
                "candidate_count": len(
                    relation_ids
                ),
                "estimate_count": len(
                    estimates
                ),
            },
        )

    def start_selected_probe(
        self,
        plan: ActiveCascadeProbePlan,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """
        Optional convenience wrapper around real ActiveProbeEngine.

        Planning and execution remain separate operations.
        """

        if self._engine is None:
            raise ActiveCascadeAPEError(
                "ActiveProbeEngine was not supplied to this bridge."
            )

        return self._engine.start_selected(
            plan.plan_result,
            metadata=dict(
                metadata or {}
            ),
        )


# =============================================================================
# Authorized graph-update resolver
# =============================================================================


class APEGraphUpdateResolver:
    """
    Resolve authorized relation evidence against local cascade candidates.

    It does not authorize evidence and does not infer causality from mere
    graph mutation.
    """

    def __init__(
        self,
        *,
        config: ActiveCascadeAPEConfig | None = None,
    ) -> None:
        self._config = (
            config
            or ActiveCascadeAPEConfig()
        )

    @property
    def config(
        self,
    ) -> ActiveCascadeAPEConfig:
        return self._config

    def resolve_update(
        self,
        *,
        current_node_id: str,
        candidates: tuple[CandidateTransition, ...],
        graph_update: Any,
        state: ActiveCascadeState,
    ) -> tuple[
        CandidateTransition | None,
        tuple[str, ...],
    ]:
        """
        Return one confirmed candidate, if uniquely supported, plus rejected
        relation identifiers.
        """

        evidence = _coerce_authorized_evidence(
            graph_update
        )

        if not evidence:
            return None, ()

        by_relation = {
            candidate.relation_id: candidate
            for candidate in candidates
        }

        confirmed: list[
            CandidateTransition
        ] = []
        rejected: list[str] = []

        for item in evidence:
            candidate = by_relation.get(
                item.relation_id
            )

            if candidate is None:
                # Evidence may legitimately refer to another graph region.
                continue

            if (
                self._config.require_exact_relation_target
                and item.source_id is not None
                and item.source_id != candidate.source_id
            ):
                continue

            if (
                self._config.require_exact_relation_target
                and item.target_id is not None
                and item.target_id != candidate.target_id
            ):
                continue

            if item.confirmed:
                if (
                    item.confidence
                    >= self._config.min_confirmation_confidence
                ):
                    confirmed.append(
                        CandidateTransition(
                            source_id=candidate.source_id,
                            target_id=candidate.target_id,
                            relation_id=candidate.relation_id,
                            confidence=max(
                                candidate.confidence,
                                item.confidence,
                            ),
                            uncertainty=min(
                                candidate.uncertainty,
                                1.0 - item.confidence,
                            ),
                            score=candidate.score,
                            evidence_state=EdgeEvidenceState.CONFIRMED,
                            metadata={
                                **dict(candidate.metadata),
                                "ape_confirmed": True,
                                "ape_evidence_confidence": item.confidence,
                            },
                        )
                    )
            elif self._config.reject_on_explicit_negative:
                rejected.append(
                    candidate.relation_id
                )

        rejected_tuple = tuple(
            dict.fromkeys(
                rejected
            )
        )

        # A Probe is allowed to narrow uncertainty without resolving the
        # transition. More than one supported candidate therefore remains
        # unresolved instead of being silently tie-broken.
        unique_confirmed = {
            item.relation_id: item
            for item in confirmed
        }

        if len(unique_confirmed) != 1:
            return None, rejected_tuple

        selected = next(
            iter(
                unique_confirmed.values()
            )
        )

        if (
            selected.relation_id
            in rejected_tuple
        ):
            raise ActiveCascadeAPEError(
                "authorized evidence both confirms and rejects the same "
                "candidate relation."
            )

        return selected, rejected_tuple


# =============================================================================
# Composite convenience object
# =============================================================================


@dataclass(frozen=True, slots=True)
class ActiveCascadeAPEBridge:
    """
    Convenience bundle for ActiveCascadeExplorer construction.

    Example
    -------
    bridge = ActiveCascadeAPEBridge.create(
        registry=registry,
        adapter=adapter,
        planner=planner,
        engine=engine,
    )

    explorer = ActiveCascadeExplorer(
        candidate_provider=provider,
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )
    """

    probe_planner: APEProbePlannerBridge
    graph_update_resolver: APEGraphUpdateResolver

    @classmethod
    def create(
        cls,
        *,
        registry: ProbeRegistry,
        adapter: ProbeGraphAdapter,
        planner: ProbePlanner,
        engine: ActiveProbeEngine | None = None,
        config: ActiveCascadeAPEConfig | None = None,
    ) -> "ActiveCascadeAPEBridge":
        shared_config = (
            config
            or ActiveCascadeAPEConfig()
        )

        return cls(
            probe_planner=APEProbePlannerBridge(
                registry=registry,
                adapter=adapter,
                planner=planner,
                engine=engine,
                config=shared_config,
            ),
            graph_update_resolver=APEGraphUpdateResolver(
                config=shared_config,
            ),
        )


__all__ = [
    "APEGraphUpdateResolver",
    "APEProbePlannerBridge",
    "ActiveCascadeAPEBridge",
    "ActiveCascadeAPEConfig",
    "ActiveCascadeAPEError",
    "ActiveCascadeProbePlan",
    "AuthorizedRelationEvidence",
]


# =============================================================================
# ROIF ACTIVE CASCADE VECTOR EXTENSION
# =============================================================================
#
# Adds vector-aware Probe evidence without rewriting the existing
# APE planning and graph-update implementation.
#
# VectorProbeEvidence != authorization.
# VectorEvidenceAssessment != authorization.
# RelationEvidenceProposal != authorization.
# Only AuthorizedRelationEvidence may resolve a cascade relation.
# =============================================================================


from roif.vector_probe import (
    VectorEvidenceAssessment,
    VectorEvidenceDecision,
    VectorEvidencePolicy,
    VectorProbeEvidence,
    assess_vector_evidence,
)


@dataclass(frozen=True, slots=True)
class ActiveCascadeAPEConfig:
    """
    Active Cascade в†” APE configuration with vector evidence support.
    """

    min_confirmation_confidence: float = 0.80
    reject_on_explicit_negative: bool = True
    require_exact_relation_target: bool = True
    allow_probe_without_target_link: bool = False

    allow_vector_support_to_propose_confirmation: bool = True
    allow_vector_contradiction_to_propose_rejection: bool = True

    vector_policy: VectorEvidencePolicy = field(
        default_factory=VectorEvidencePolicy
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        confidence = float(
            self.min_confirmation_confidence
        )

        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ActiveCascadeAPEError(
                "min_confirmation_confidence must be in [0, 1]."
            )

        if not isinstance(
            self.vector_policy,
            VectorEvidencePolicy,
        ):
            raise ActiveCascadeAPEError(
                "vector_policy must be VectorEvidencePolicy."
            )

        object.__setattr__(
            self,
            "min_confirmation_confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


@dataclass(frozen=True, slots=True)
class RelationEvidenceProposal:
    """
    Non-authorized interpretation of physical vector Probe evidence.
    """

    relation_id: str
    source_id: str
    target_id: str

    decision: VectorEvidenceDecision

    proposed_confirmed: bool
    proposed_rejected: bool

    confidence: float

    assessment: VectorEvidenceAssessment

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "source_id",
            "target_id",
        ):
            value = str(
                getattr(
                    self,
                    name,
                )
            ).strip()

            if not value:
                raise ActiveCascadeAPEError(
                    f"{name} must not be empty."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        confidence = float(
            self.confidence
        )

        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ActiveCascadeAPEError(
                "confidence must be in [0, 1]."
            )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "decision",
            VectorEvidenceDecision(
                self.decision
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


@dataclass(frozen=True, slots=True)
class AuthorizedRelationEvidence:
    """
    Explicitly authorized relation evidence.

    Vector evidence is preserved for audit, but does not authorize itself.
    """

    relation_id: str
    confirmed: bool
    confidence: float

    source_id: str | None = None
    target_id: str | None = None

    vector_assessment: VectorEvidenceAssessment | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        relation_id = str(
            self.relation_id
        ).strip()

        if not relation_id:
            raise ActiveCascadeAPEError(
                "relation_id must not be empty."
            )

        confidence = float(
            self.confidence
        )

        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ActiveCascadeAPEError(
                "confidence must be in [0, 1]."
            )

        object.__setattr__(
            self,
            "relation_id",
            relation_id,
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


def _vector_assessment_confidence(
    assessment: VectorEvidenceAssessment,
) -> float:
    """
    Derive conservative confidence from explicit directional evidence.

    No opaque weighted aggregate is used.
    """

    if assessment.decision in {
        VectorEvidenceDecision.INSUFFICIENT,
        VectorEvidenceDecision.NO_RESPONSE,
    }:
        return 0.0

    evidence = assessment.evidence

    channels = tuple(
        channel
        for channel in (
            evidence.tension,
            evidence.displacement,
        )
        if channel is not None
    )

    if not channels:
        return 0.0

    if (
        assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    ):
        values = [
            max(
                0.0,
                channel.alignment,
            )
            for channel in channels
            if channel.alignment > 0.0
        ]

    else:
        values = [
            abs(
                min(
                    0.0,
                    channel.alignment,
                )
            )
            for channel in channels
            if channel.alignment < 0.0
        ]

    if not values:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            max(values),
        ),
    )


class APEVectorEvidenceBridge:
    """
    Converts VectorProbeEvidence into a non-authorized relation proposal.
    """

    def __init__(
        self,
        *,
        config: ActiveCascadeAPEConfig | None = None,
    ) -> None:
        self._config = (
            config
            or ActiveCascadeAPEConfig()
        )

    @property
    def config(
        self,
    ) -> ActiveCascadeAPEConfig:
        return self._config

    def assess(
        self,
        evidence: VectorProbeEvidence,
    ) -> RelationEvidenceProposal:

        assessment = assess_vector_evidence(
            evidence,
            policy=self._config.vector_policy,
        )

        confidence = (
            _vector_assessment_confidence(
                assessment
            )
        )

        proposed_confirmed = (
            assessment.decision
            is VectorEvidenceDecision.SUPPORTS
            and
            self._config
            .allow_vector_support_to_propose_confirmation
        )

        proposed_rejected = (
            assessment.decision
            is VectorEvidenceDecision.CONTRADICTS
            and
            self._config
            .allow_vector_contradiction_to_propose_rejection
        )

        return RelationEvidenceProposal(
            relation_id=evidence.relation_id,
            source_id=evidence.source_id,
            target_id=evidence.target_id,
            decision=assessment.decision,
            proposed_confirmed=proposed_confirmed,
            proposed_rejected=proposed_rejected,
            confidence=confidence,
            assessment=assessment,
            metadata={
                "supporting_channels":
                    assessment.supporting_channels,

                "contradicting_channels":
                    assessment.contradicting_channels,

                "reasons":
                    assessment.reasons,
            },
        )

    def assess_many(
        self,
        evidence: Iterable[
            VectorProbeEvidence
        ],
    ) -> tuple[
        RelationEvidenceProposal,
        ...,
    ]:
        return tuple(
            self.assess(item)
            for item in evidence
        )

    def authorize(
        self,
        proposal: RelationEvidenceProposal,
        *,
        approve: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuthorizedRelationEvidence | None:
        """
        Explicit authorization boundary.
        """

        if not approve:
            return None

        if proposal.proposed_confirmed:
            confirmed = True

        elif proposal.proposed_rejected:
            confirmed = False

        else:
            raise ActiveCascadeAPEError(
                "proposal is not eligible for authorization."
            )

        return AuthorizedRelationEvidence(
            relation_id=proposal.relation_id,
            confirmed=confirmed,
            confidence=proposal.confidence,
            source_id=proposal.source_id,
            target_id=proposal.target_id,
            vector_assessment=proposal.assessment,
            metadata={
                **dict(proposal.metadata),
                **dict(
                    metadata or {}
                ),
                "authorized_from_vector_evidence": True,
            },
        )


@dataclass(frozen=True, slots=True)
class ActiveCascadeAPEBridge:
    """
    Composite Active Cascade в†” APE bridge with vector evidence support.
    """

    probe_planner: APEProbePlannerBridge
    vector_evidence: APEVectorEvidenceBridge
    graph_update_resolver: APEGraphUpdateResolver

    @classmethod
    def create(
        cls,
        *,
        registry: ProbeRegistry,
        adapter: ProbeGraphAdapter,
        planner: ProbePlanner,
        engine: ActiveProbeEngine | None = None,
        config: ActiveCascadeAPEConfig | None = None,
    ) -> "ActiveCascadeAPEBridge":

        shared_config = (
            config
            or ActiveCascadeAPEConfig()
        )

        return cls(
            probe_planner=APEProbePlannerBridge(
                registry=registry,
                adapter=adapter,
                planner=planner,
                engine=engine,
                config=shared_config,
            ),

            vector_evidence=APEVectorEvidenceBridge(
                config=shared_config,
            ),

            graph_update_resolver=APEGraphUpdateResolver(
                config=shared_config,
            ),
        )


# Extend public exports after the original module definitions.
for _name in (
    "APEVectorEvidenceBridge",
    "RelationEvidenceProposal",
):
    if _name not in __all__:
        __all__.append(_name)

# =============================================================================
# END ROIF ACTIVE CASCADE VECTOR EXTENSION
# =============================================================================

