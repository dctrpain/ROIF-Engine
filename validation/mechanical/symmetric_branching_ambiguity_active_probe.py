"""
ROIF Mechanical Validation
Scenario 03B — Symmetric Branching Ambiguity Active Probe

Purpose
-------
Scenario 03B tests the opposite failure mode to Scenario 02A.

02A:
    Probe is discriminative enough to support one branch and leave the other
    insufficient.

03B:
    A deliberately weak/non-discriminative Probe produces nearly equivalent
    directed responses in both branches.

Canonical control rule
----------------------
If more than one candidate remains supportable after the Probe, Active Cascade
must not silently select a winner.

The system should remain unresolved at the shared preload node until additional
discriminative evidence becomes available.

This module:
- uses the real ActiveCascadeExplorer;
- uses the real ActiveCascadeAPEBridge;
- uses real VectorProbeEvidence construction;
- does not inject a preferred branch;
- does not consult evaluator-side branch_truth during inference;
- never authorizes two supported branches into a unique transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from roif.active_cascade import (
    ActiveCascadeExplorer,
    ActiveCascadeState,
    CandidateTransition,
    TransitionStatus,
    start_active_cascade,
)

from roif.active_cascade_ape import (
    ActiveCascadeAPEBridge,
    ActiveCascadeAPEConfig,
    ActiveCascadeAPEError,
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

from validation.mechanical.symmetric_branching_ambiguity_case import (
    BRANCH_A_CHANNEL,
    BRANCH_B_CHANNEL,
    PRELOAD_CHANNEL,
    PRELOAD_TO_A,
    PRELOAD_TO_B,
    SymmetricBranchingValidationCase,
    build_symmetric_branching_ambiguity_case,
)


# =============================================================================
# Configuration
# =============================================================================


AMBIGUITY_PROBE_ID = "probe_03B_shared_branch_response"

PROBE_MAGNITUDE = 0.10

# The two branches are intentionally allowed to differ only weakly.
MAX_BRANCH_RESPONSE_GAP = 0.02

MIN_MEANINGFUL_DELTA_UTILIZATION = 0.02


class SymmetricBranchingActiveProbeError(
    ActiveCascadeAPEError
):
    """Raised when Scenario 03B violates its ambiguity-preservation contract."""


# =============================================================================
# Immutable audit objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class AmbiguousBranchAssessment:
    relation_id: str
    decision: VectorEvidenceDecision
    proposed_confirmed: bool
    proposed_rejected: bool
    confidence: float
    delta_utilization: float | None
    alignment: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision",
            VectorEvidenceDecision(self.decision),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class SymmetricAmbiguityProbeResult:
    initial_state: ActiveCascadeState
    decision_before_probe: Any
    probe_plan: Any
    branch_a: AmbiguousBranchAssessment
    branch_b: AmbiguousBranchAssessment
    authorized_evidence: tuple[AuthorizedRelationEvidence, ...]
    final_state: ActiveCascadeState
    decision_after_probe: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authorized_evidence",
            tuple(self.authorized_evidence),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def remains_on_preload(self) -> bool:
        return self.final_state.current_node_id == PRELOAD_CHANNEL

    @property
    def advanced(self) -> bool:
        return self.final_state.current_node_id != PRELOAD_CHANNEL


# =============================================================================
# Case helpers
# =============================================================================


def build_case() -> SymmetricBranchingValidationCase:
    return build_symmetric_branching_ambiguity_case()


def channel_lookup(
    case: SymmetricBranchingValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            channel.channel_id: channel
            for entity in case.pathological_system.entities
            for channel in entity.channels
        }
    )


def operator_lookup(
    case: SymmetricBranchingValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            operator.operator_id: operator
            for operator in case.pathological_system.operators
        }
    )


def outgoing_branch_operators(
    case: SymmetricBranchingValidationCase,
) -> tuple[Any, ...]:
    operators = tuple(
        operator
        for operator in case.pathological_system.operators
        if operator.source_ids == (PRELOAD_CHANNEL,)
    )

    if len(operators) != 2:
        raise SymmetricBranchingActiveProbeError(
            "Scenario 03B requires exactly two outgoing preload branches."
        )

    return operators


def vector3_tuple(
    vector: Any,
) -> tuple[float, float, float]:
    return (
        float(vector.x),
        float(vector.y),
        float(vector.z),
    )


def normalized_vector3_tuple(
    vector: Any,
) -> tuple[float, float, float]:
    return vector3_tuple(
        vector.normalized()
    )


def candidate_from_operator(
    operator: Any,
) -> CandidateTransition:
    if len(operator.source_ids) != 1:
        raise SymmetricBranchingActiveProbeError(
            "03B branch operator must have one source."
        )

    if len(operator.target_ids) != 1:
        raise SymmetricBranchingActiveProbeError(
            "03B branch operator must have one target."
        )

    return CandidateTransition(
        source_id=operator.source_ids[0],
        target_id=operator.target_ids[0],
        relation_id=operator.operator_id,
        confidence=0.50,
        uncertainty=0.50,
        score=abs(float(operator.gain)),
        metadata={
            "validation_case": "mechanical_03B",
            "evidence_source": "symmetric_branch_candidate",
            "preferred_branch_label_used": False,
            "expected_role_label_used": False,
        },
    )


class Scenario03BAmbiguousCandidateProvider:
    def __init__(
        self,
        case: SymmetricBranchingValidationCase,
    ) -> None:
        self._case = case
        self.calls: list[str] = []

    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> tuple[CandidateTransition, ...]:
        self.calls.append(current_node_id)

        return tuple(
            candidate_from_operator(operator)
            for operator in self._case.pathological_system.operators
            if operator.source_ids == (current_node_id,)
        )


# =============================================================================
# Probe definition / APE bridge
# =============================================================================


def build_probe_definition() -> ProbeDefinition:
    return ProbeDefinition(
        identifier=AMBIGUITY_PROBE_ID,
        name="03B symmetric branch ambiguity Probe",
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.DYNAMIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=PerturbationType.SIMULATED_INTERVENTION,
            magnitude=PROBE_MAGNITUDE,
            magnitude_units="normalized",
            duration_seconds=1.0,
        ),
        description=(
            "Small reversible synthetic perturbation of the shared preload "
            "node. The Probe is intentionally weakly discriminative so both "
            "parallel branches may remain supportable."
        ),
        tags=(
            "validation",
            "mechanical_03B",
            "active_cascade",
            "vector_probe",
            "ambiguity_control",
        ),
    )


def build_probe_registry() -> ProbeRegistry:
    return ProbeRegistry(
        (
            build_probe_definition(),
        )
    )


def build_probe_adapter() -> ProbeGraphAdapter:
    return ProbeGraphAdapter(
        (
            ProbeGraphLink(
                probe_identifier=AMBIGUITY_PROBE_ID,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=PRELOAD_TO_A,
                        sensitivity=1.0,
                        discrimination=0.20,
                        expected_uncertainty_reduction=0.30,
                    ),
                    ProbeGraphTarget(
                        target_identifier=PRELOAD_TO_B,
                        sensitivity=1.0,
                        discrimination=0.20,
                        expected_uncertainty_reduction=0.30,
                    ),
                ),
                novelty=0.50,
                feasibility=1.0,
                estimate_confidence=0.90,
                metadata={
                    "validation_case": "mechanical_03B",
                    "preferred_branch_label_used": False,
                    "weak_discrimination": True,
                },
            ),
        )
    )


def build_snapshot() -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id="mechanical_03B_ambiguous_preload",
        version="v0",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PRELOAD_TO_A,
                    source=PRELOAD_CHANNEL,
                    target=BRANCH_A_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PRELOAD_TO_B,
                    source=PRELOAD_CHANNEL,
                    target=BRANCH_B_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
        ),
        metadata={
            "validation_case": "mechanical_03B",
            "expected_unique_branch": False,
            "preferred_branch_label_used": False,
            "external_branch_truth_included": False,
        },
    )


def build_bridge() -> ActiveCascadeAPEBridge:
    registry = build_probe_registry()
    adapter = build_probe_adapter()
    planner = ProbePlanner(
        registry=registry
    )
    engine = ActiveProbeEngine(
        registry=registry
    )

    config = ActiveCascadeAPEConfig(
        min_confirmation_confidence=0.80,
        vector_policy=VectorEvidencePolicy(
            min_forward_alignment=0.70,
            contradiction_alignment=-0.50,
            require_tension_or_displacement=True,
            min_delta_utilization=MIN_MEANINGFUL_DELTA_UTILIZATION,
            require_capacity_relevance=False,
        ),
        metadata={
            "validation_case": "mechanical_03B",
            "phase": "ambiguity_probe",
            "preferred_branch_label_used": False,
        },
    )

    return ActiveCascadeAPEBridge.create(
        registry=registry,
        adapter=adapter,
        planner=planner,
        engine=engine,
        config=config,
    )


# =============================================================================
# Vector Probe evidence
# =============================================================================


def _target_channel_for_relation(
    case: SymmetricBranchingValidationCase,
    relation_id: str,
) -> Any:
    operators = operator_lookup(case)

    if relation_id not in {
        PRELOAD_TO_A,
        PRELOAD_TO_B,
    }:
        raise SymmetricBranchingActiveProbeError(
            f"unknown 03B branch relation: {relation_id!r}"
        )

    operator = operators[relation_id]
    target_id = operator.target_ids[0]

    return channel_lookup(case)[
        target_id
    ]


def local_probe_evidence(
    case: SymmetricBranchingValidationCase,
    relation_id: str,
) -> VectorProbeEvidence:
    """
    Build nearly symmetric directed responses for both branch candidates.

    The Probe follows the target branch direction and produces a small
    utilization change. Because branch parameters are near-symmetric, both
    relations should receive comparable supportive evidence.
    """

    operators = operator_lookup(case)
    operator = operators[relation_id]

    target_channel = _target_channel_for_relation(
        case,
        relation_id,
    )

    target_direction = normalized_vector3_tuple(
        target_channel.direction
    )

    transmitted_increment = (
        PROBE_MAGNITUDE
        * abs(float(operator.gain))
    )

    probe_tension = tuple(
        component * transmitted_increment
        for component in target_direction
    )

    capacity_state = target_channel.capacity_state

    geometry = CandidateEdgeGeometry(
        source_id=PRELOAD_CHANNEL,
        target_id=target_channel.channel_id,
        source_position=(0.0, 0.0, 0.0),
        target_position=target_direction,
        metadata={
            "geometry_mode": "03B_local_target_direction",
            "preferred_branch_label_used": False,
        },
    )

    return build_vector_probe_evidence(
        relation_id=relation_id,
        geometry=geometry,
        baseline_tension=(0.0, 0.0, 0.0),
        probe_tension=probe_tension,
        baseline_demand=float(
            capacity_state.load
        ),
        baseline_capacity=float(
            capacity_state.capacity
        ),
        probe_demand=(
            float(capacity_state.load)
            + transmitted_increment
        ),
        probe_capacity=float(
            capacity_state.capacity
        ),
        metadata={
            "validation_case": "mechanical_03B",
            "probe_magnitude": PROBE_MAGNITUDE,
            "operator_gain": float(operator.gain),
            "preferred_branch_label_used": False,
            "expected_role_label_used": False,
        },
    )


def branch_response_gap(
    case: SymmetricBranchingValidationCase,
) -> float:
    left = local_probe_evidence(
        case,
        PRELOAD_TO_A,
    )
    right = local_probe_evidence(
        case,
        PRELOAD_TO_B,
    )

    if (
        left.utilization is None
        or right.utilization is None
    ):
        raise SymmetricBranchingActiveProbeError(
            "03B requires utilization evidence for both branches."
        )

    return abs(
        float(
            left.utilization.delta_utilization
        )
        - float(
            right.utilization.delta_utilization
        )
    )


# =============================================================================
# Assessment helpers
# =============================================================================


def assess_relation(
    bridge: ActiveCascadeAPEBridge,
    case: SymmetricBranchingValidationCase,
    relation_id: str,
) -> AmbiguousBranchAssessment:
    evidence = local_probe_evidence(
        case,
        relation_id,
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    utilization_delta = (
        float(
            evidence.utilization.delta_utilization
        )
        if evidence.utilization is not None
        else None
    )

    alignment = (
        float(
            evidence.tension.alignment
        )
        if evidence.tension is not None
        else None
    )

    return AmbiguousBranchAssessment(
        relation_id=relation_id,
        decision=proposal.decision,
        proposed_confirmed=proposal.proposed_confirmed,
        proposed_rejected=proposal.proposed_rejected,
        confidence=float(proposal.confidence),
        delta_utilization=utilization_delta,
        alignment=alignment,
        metadata={
            "preferred_branch_label_used": False,
        },
    )


def authorized_support(
    bridge: ActiveCascadeAPEBridge,
    case: SymmetricBranchingValidationCase,
    relation_id: str,
) -> AuthorizedRelationEvidence | None:
    """
    Authorize an individual proposal only if the vector policy considers it
    supportable.

    This function does not choose between branches.
    """

    evidence = local_probe_evidence(
        case,
        relation_id,
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    if not proposal.proposed_confirmed:
        return None

    return bridge.vector_evidence.authorize(
        proposal,
        approve=True,
        metadata={
            "validation_case": "mechanical_03B",
            "preferred_branch_label_used": False,
        },
    )


# =============================================================================
# Complete ambiguity-preservation run
# =============================================================================


def run_ambiguity_probe(
    case: SymmetricBranchingValidationCase | None = None,
) -> SymmetricAmbiguityProbeResult:
    case = case or build_case()
    bridge = build_bridge()
    snapshot = build_snapshot()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario03BAmbiguousCandidateProvider(
            case
        ),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        PRELOAD_CHANNEL,
        graph_version=snapshot.version,
        metadata={
            "validation_case": "mechanical_03B",
            "preferred_branch_label_used": False,
        },
    )

    before = explorer.inspect_transition(
        state,
        graph_snapshot=snapshot,
    )

    if before.status is not TransitionStatus.REQUIRES_PROBE:
        raise SymmetricBranchingActiveProbeError(
            "03B preload transition must require Probe before evidence."
        )

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=snapshot,
        decision=before,
    )

    branch_a = assess_relation(
        bridge,
        case,
        PRELOAD_TO_A,
    )
    branch_b = assess_relation(
        bridge,
        case,
        PRELOAD_TO_B,
    )

    if branch_response_gap(case) > MAX_BRANCH_RESPONSE_GAP:
        raise SymmetricBranchingActiveProbeError(
            "03B Probe became too discriminative for the ambiguity control."
        )

    authorized = tuple(
        item
        for item in (
            authorized_support(
                bridge,
                case,
                PRELOAD_TO_A,
            ),
            authorized_support(
                bridge,
                case,
                PRELOAD_TO_B,
            ),
        )
        if item is not None
    )

    # Feed all individually supportable evidence together.
    # APEGraphUpdateResolver must refuse to choose a unique winner when more
    # than one candidate remains positively supported.
    final_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=snapshot,
        graph_update=authorized,
        probe_identifier=AMBIGUITY_PROBE_ID,
        graph_version="v1",
    )

    after = explorer.inspect_transition(
        final_state,
        graph_snapshot=snapshot,
    )

    if final_state.current_node_id != PRELOAD_CHANNEL:
        raise SymmetricBranchingActiveProbeError(
            "03B advanced despite non-discriminative multi-branch support."
        )

    if final_state.confirmed_edges:
        raise SymmetricBranchingActiveProbeError(
            "03B incorrectly added a confirmed branch edge."
        )

    if after.status is not TransitionStatus.REQUIRES_PROBE:
        raise SymmetricBranchingActiveProbeError(
            "03B should remain unresolved after a non-discriminative Probe."
        )

    return SymmetricAmbiguityProbeResult(
        initial_state=state,
        decision_before_probe=before,
        probe_plan=plan,
        branch_a=branch_a,
        branch_b=branch_b,
        authorized_evidence=authorized,
        final_state=final_state,
        decision_after_probe=after,
        metadata={
            "validation_case": "mechanical_03B",
            "control_question": (
                "does_non_discriminative_probe_preserve_ambiguity"
            ),
            "branch_response_gap": branch_response_gap(
                case
            ),
            "max_allowed_branch_response_gap": MAX_BRANCH_RESPONSE_GAP,
            "preferred_branch_label_used": False,
            "external_branch_truth_used": False,
            "advanced": False,
        },
    )


__all__ = [
    "AMBIGUITY_PROBE_ID",
    "AmbiguousBranchAssessment",
    "MAX_BRANCH_RESPONSE_GAP",
    "MIN_MEANINGFUL_DELTA_UTILIZATION",
    "PROBE_MAGNITUDE",
    "Scenario03BAmbiguousCandidateProvider",
    "SymmetricAmbiguityProbeResult",
    "SymmetricBranchingActiveProbeError",
    "assess_relation",
    "authorized_support",
    "branch_response_gap",
    "build_bridge",
    "build_case",
    "build_probe_adapter",
    "build_probe_definition",
    "build_probe_registry",
    "build_snapshot",
    "candidate_from_operator",
    "channel_lookup",
    "local_probe_evidence",
    "operator_lookup",
    "outgoing_branch_operators",
    "run_ambiguity_probe",
]
