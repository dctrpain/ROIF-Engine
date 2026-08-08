"""
ROIF Mechanical Validation
Scenario 02A вЂ” Active Probe Phase 2

Phase 2 tests whether load_sharing_junction is actively supported as the
common convergent mediator of both pre-stressed branches:

    primary_branch_stiffness -> load_sharing_junction
    bypass_branch_stiffness  -> load_sharing_junction

No external expected D_root label is used during candidate construction,
Probe planning, response generation, evidence assessment, or authorization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from roif.active_cascade import (
    ActiveCascadeExplorer,
    ActiveCascadeState,
    CandidateTransition,
    start_active_cascade,
)
from roif.active_cascade_ape import (
    ActiveCascadeAPEBridge,
    ActiveCascadeAPEConfig,
    AuthorizedRelationEvidence,
)
from roif.active_probe_engine import ActiveProbeEngine
from roif.probe_entities import (
    Perturbation,
    PerturbationType,
    ProbeDefinition,
    ProbeMethod,
    ProbePurpose,
    ProbeRegime,
)
from roif.probe_graph_adapter import (
    GraphTarget,
    GraphTargetKind,
    GraphUncertainty,
    IncompleteGraphSnapshot,
    ProbeGraphAdapter,
    ProbeGraphLink,
    ProbeGraphTarget,
)
from roif.probe_planner import ProbePlanner
from roif.probe_registry import ProbeRegistry
from roif.vector_probe import (
    CandidateEdgeGeometry,
    VectorEvidenceDecision,
    VectorEvidencePolicy,
    VectorProbeEvidence,
    build_vector_probe_evidence,
)
from validation.mechanical.prestressed_branching_case import (
    PrestressedBranchingValidationCase,
    build_prestressed_branching_case,
)

PRIMARY_CHANNEL = "primary_branch_stiffness"
BYPASS_CHANNEL = "bypass_branch_stiffness"
JUNCTION_CHANNEL = "load_sharing_junction"

PRIMARY_TO_JUNCTION = "primary_to_junction"
BYPASS_TO_JUNCTION = "bypass_to_junction"

PRIMARY_PROBE_ID = "probe_02A_primary_to_junction"
BYPASS_PROBE_ID = "probe_02A_bypass_to_junction"

PROBE_MAGNITUDE = 0.20
MIN_JUNCTION_DELTA_UTILIZATION = 0.02


class Phase2ValidationError(ValueError):
    """Raised when Scenario 02A Phase 2 cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class BranchJunctionProbeResult:
    source_channel_id: str
    relation_id: str
    probe_identifier: str
    evidence: VectorProbeEvidence
    decision: VectorEvidenceDecision
    authorized_evidence: AuthorizedRelationEvidence
    reached_junction: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class JunctionConvergenceResult:
    primary_result: BranchJunctionProbeResult
    bypass_result: BranchJunctionProbeResult
    junction_candidate_id: str | None
    both_relations_supported: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def build_case() -> PrestressedBranchingValidationCase:
    return build_prestressed_branching_case()


def channel_lookup(
    case: PrestressedBranchingValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            channel.channel_id: channel
            for entity in case.pathological_system.entities
            for channel in entity.channels
        }
    )


def operator_lookup(
    case: PrestressedBranchingValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            operator.operator_id: operator
            for operator in case.pathological_system.operators
        }
    )


def _vector3_tuple(vector: Any) -> tuple[float, float, float]:
    return float(vector.x), float(vector.y), float(vector.z)


def _normalized_direction(channel: Any) -> tuple[float, float, float]:
    return _vector3_tuple(channel.direction.normalized())


def candidate_for_relation(
    case: PrestressedBranchingValidationCase,
    relation_id: str,
) -> CandidateTransition:
    operator = operator_lookup(case)[relation_id]

    return CandidateTransition(
        source_id=operator.source_ids[0],
        target_id=operator.target_ids[0],
        relation_id=operator.operator_id,
        confidence=0.50,
        uncertainty=0.50,
        score=abs(float(operator.gain)),
        metadata={
            "validation_case": "mechanical_02A",
            "phase": "active_probe_2",
            "expected_d_root_label_used": False,
        },
    )


class Phase2CandidateProvider:
    def __init__(
        self,
        case: PrestressedBranchingValidationCase,
    ) -> None:
        self._case = case

    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> tuple[CandidateTransition, ...]:
        if current_node_id == PRIMARY_CHANNEL:
            return (
                candidate_for_relation(
                    self._case,
                    PRIMARY_TO_JUNCTION,
                ),
            )
        if current_node_id == BYPASS_CHANNEL:
            return (
                candidate_for_relation(
                    self._case,
                    BYPASS_TO_JUNCTION,
                ),
            )
        return ()


def _probe_definition(
    identifier: str,
    source_channel_id: str,
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=identifier,
        name=f"02A {source_channel_id} -> junction Probe",
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.DYNAMIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=PerturbationType.SIMULATED_INTERVENTION,
            magnitude=PROBE_MAGNITUDE,
            magnitude_units="normalized",
            duration_seconds=1.0,
        ),
        tags=(
            "validation",
            "mechanical_02A",
            "phase2",
            "junction_convergence",
        ),
    )


def build_probe_registry() -> ProbeRegistry:
    return ProbeRegistry(
        (
            _probe_definition(
                PRIMARY_PROBE_ID,
                PRIMARY_CHANNEL,
            ),
            _probe_definition(
                BYPASS_PROBE_ID,
                BYPASS_CHANNEL,
            ),
        )
    )


def build_probe_adapter() -> ProbeGraphAdapter:
    return ProbeGraphAdapter(
        (
            ProbeGraphLink(
                probe_identifier=PRIMARY_PROBE_ID,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=PRIMARY_TO_JUNCTION,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.90,
                    ),
                ),
                novelty=0.80,
                feasibility=1.0,
                estimate_confidence=0.95,
            ),
            ProbeGraphLink(
                probe_identifier=BYPASS_PROBE_ID,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=BYPASS_TO_JUNCTION,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.90,
                    ),
                ),
                novelty=0.80,
                feasibility=1.0,
                estimate_confidence=0.95,
            ),
        )
    )


def build_snapshot(
    source_channel_id: str,
) -> IncompleteGraphSnapshot:
    if source_channel_id not in {
        PRIMARY_CHANNEL,
        BYPASS_CHANNEL,
    }:
        raise Phase2ValidationError(
            "source_channel_id must be primary or bypass."
        )

    return IncompleteGraphSnapshot(
        graph_id=f"mechanical_02A_phase2_{source_channel_id}",
        version="v0",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PRIMARY_TO_JUNCTION,
                    source=PRIMARY_CHANNEL,
                    target=JUNCTION_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
                metadata={
                    "expected_d_root_label_used": False,
                },
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=BYPASS_TO_JUNCTION,
                    source=BYPASS_CHANNEL,
                    target=JUNCTION_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
                metadata={
                    "expected_d_root_label_used": False,
                },
            ),
        ),
        metadata={
            "validation_case": "mechanical_02A",
            "phase": "active_probe_2",
            "active_source_channel_id": source_channel_id,
            "expected_d_root_label_used": False,
        },
    )


def build_bridge() -> ActiveCascadeAPEBridge:
    registry = build_probe_registry()

    return ActiveCascadeAPEBridge.create(
        registry=registry,
        adapter=build_probe_adapter(),
        planner=ProbePlanner(registry=registry),
        engine=ActiveProbeEngine(registry=registry),
        config=ActiveCascadeAPEConfig(
            min_confirmation_confidence=0.80,
            vector_policy=VectorEvidencePolicy(
                min_forward_alignment=0.70,
                contradiction_alignment=-0.50,
                require_tension_or_displacement=True,
                min_delta_utilization=(
                    MIN_JUNCTION_DELTA_UTILIZATION
                ),
                require_capacity_relevance=True,
            ),
            metadata={
                "validation_case": "mechanical_02A",
                "phase": "active_probe_2",
            },
        ),
    )


def build_branch_junction_evidence(
    case: PrestressedBranchingValidationCase,
    relation_id: str,
) -> VectorProbeEvidence:
    operators = operator_lookup(case)
    channels = channel_lookup(case)

    if relation_id not in {
        PRIMARY_TO_JUNCTION,
        BYPASS_TO_JUNCTION,
    }:
        raise Phase2ValidationError(
            "relation_id is not a Phase 2 branch relation."
        )

    operator = operators[relation_id]
    junction = channels[JUNCTION_CHANNEL]
    direction = _normalized_direction(junction)

    transmitted_increment = (
        PROBE_MAGNITUDE * abs(float(operator.gain))
    )

    probe_tension = tuple(
        component * transmitted_increment
        for component in direction
    )

    capacity = float(junction.capacity_state.capacity)
    baseline_load = float(junction.capacity_state.load)

    geometry = CandidateEdgeGeometry(
        source_id=operator.source_ids[0],
        target_id=operator.target_ids[0],
        source_position=(0.0, 0.0, 0.0),
        target_position=direction,
    )

    return build_vector_probe_evidence(
        relation_id=relation_id,
        geometry=geometry,
        baseline_tension=(0.0, 0.0, 0.0),
        probe_tension=probe_tension,
        baseline_demand=baseline_load,
        baseline_capacity=capacity,
        probe_demand=baseline_load + transmitted_increment,
        probe_capacity=capacity,
        metadata={
            "operator_gain": float(operator.gain),
            "probe_magnitude": PROBE_MAGNITUDE,
            "expected_d_root_label_used": False,
        },
    )


def _relation_probe_pair(
    source_channel_id: str,
) -> tuple[str, str]:
    if source_channel_id == PRIMARY_CHANNEL:
        return PRIMARY_TO_JUNCTION, PRIMARY_PROBE_ID
    if source_channel_id == BYPASS_CHANNEL:
        return BYPASS_TO_JUNCTION, BYPASS_PROBE_ID
    raise Phase2ValidationError(
        "source_channel_id must be primary or bypass."
    )


def run_branch_probe(
    case: PrestressedBranchingValidationCase,
    source_channel_id: str,
) -> BranchJunctionProbeResult:
    relation_id, probe_id = _relation_probe_pair(
        source_channel_id
    )

    bridge = build_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Phase2CandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        source_channel_id,
        graph_version="v0",
    )
    snapshot = build_snapshot(source_channel_id)

    explorer.plan_required_probe(
        state,
        graph_snapshot=snapshot,
    )

    evidence = build_branch_junction_evidence(
        case,
        relation_id,
    )
    proposal = bridge.vector_evidence.assess(evidence)

    if proposal.decision is not VectorEvidenceDecision.SUPPORTS:
        raise Phase2ValidationError(
            f"{relation_id} was not supported."
        )

    authorized = bridge.vector_evidence.authorize(
        proposal,
        approve=True,
    )

    if authorized is None:
        raise Phase2ValidationError(
            f"{relation_id} was not authorized."
        )

    next_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=snapshot,
        graph_update=authorized,
        probe_identifier=probe_id,
        graph_version="v1",
    )

    return BranchJunctionProbeResult(
        source_channel_id=source_channel_id,
        relation_id=relation_id,
        probe_identifier=probe_id,
        evidence=evidence,
        decision=proposal.decision,
        authorized_evidence=authorized,
        reached_junction=(
            next_state.current_node_id == JUNCTION_CHANNEL
        ),
        metadata={
            "final_node_id": next_state.current_node_id,
            "expected_d_root_label_used": False,
        },
    )


def reconstruct_junction_candidate(
    case: PrestressedBranchingValidationCase | None = None,
) -> JunctionConvergenceResult:
    case = case or build_case()

    primary = run_branch_probe(
        case,
        PRIMARY_CHANNEL,
    )
    bypass = run_branch_probe(
        case,
        BYPASS_CHANNEL,
    )

    both_supported = (
        primary.decision is VectorEvidenceDecision.SUPPORTS
        and bypass.decision is VectorEvidenceDecision.SUPPORTS
        and primary.reached_junction
        and bypass.reached_junction
    )

    primary_target = primary.authorized_evidence.target_id
    bypass_target = bypass.authorized_evidence.target_id

    same_target = (
        primary_target is not None
        and primary_target == bypass_target
    )

    junction_candidate = (
        primary_target
        if both_supported and same_target
        else None
    )

    return JunctionConvergenceResult(
        primary_result=primary,
        bypass_result=bypass,
        junction_candidate_id=junction_candidate,
        both_relations_supported=(
            both_supported and same_target
        ),
        metadata={
            "validation_case": "mechanical_02A",
            "phase": "active_probe_2",
            "evidence_basis": "dual_branch_authorized_vector_probe",
            "expected_d_root_label_used": False,
        },
    )


__all__ = [
    "BYPASS_CHANNEL",
    "BYPASS_PROBE_ID",
    "BYPASS_TO_JUNCTION",
    "BranchJunctionProbeResult",
    "JUNCTION_CHANNEL",
    "JunctionConvergenceResult",
    "MIN_JUNCTION_DELTA_UTILIZATION",
    "PRIMARY_CHANNEL",
    "PRIMARY_PROBE_ID",
    "PRIMARY_TO_JUNCTION",
    "PROBE_MAGNITUDE",
    "Phase2CandidateProvider",
    "Phase2ValidationError",
    "build_branch_junction_evidence",
    "build_bridge",
    "build_case",
    "build_probe_adapter",
    "build_probe_registry",
    "build_snapshot",
    "candidate_for_relation",
    "channel_lookup",
    "operator_lookup",
    "reconstruct_junction_candidate",
    "run_branch_probe",
]

