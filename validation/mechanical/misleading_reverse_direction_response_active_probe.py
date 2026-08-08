"""
ROIF Mechanical Validation
Scenario 03D — Misleading Reverse-Direction Response Active Probe

Purpose
-------
Scenario 03D tests whether Active Probe inference can reject a louder
reverse-directed branch while advancing through a quieter aligned branch.

Passive scalar mechanics
------------------------
The passive CapacityTensor intentionally gives:

    loud_reverse_branch > aligned_forward_branch

in scalar transfer amplitude.

Active Probe evidence
---------------------
Probe-response directions are evaluated separately:

    reverse response -> strictly opposite to candidate edge
    aligned response -> parallel to candidate edge

Expected inference
------------------
Before Probe:
    shared_probe_node has two candidates -> REQUIRES_PROBE

After vector assessment:
    reverse relation -> CONTRADICTS / rejectable
    aligned relation -> SUPPORTS / confirmable

After explicit authorization:
    reverse relation -> AuthorizedRelationEvidence(confirmed=False)
    aligned relation -> AuthorizedRelationEvidence(confirmed=True)

After graph update:
    reverse relation is added to rejected_relations
    aligned relation is added to confirmed_edges
    Active Cascade advances only to aligned_forward_branch

Architectural boundaries
------------------------
This module does not use evaluator-side amplitude_winner,
contradiction_branch, or directional_winner labels during inference.

It uses:
- real ActiveCascadeExplorer;
- real ActiveCascadeAPEBridge;
- real ProbeRegistry / ProbeGraphAdapter / ProbePlanner;
- real VectorProbeEvidence;
- explicit authorization before graph-update resolution.
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

from validation.mechanical.misleading_reverse_direction_response_case import (
    ALIGNED_BRANCH_CHANNEL,
    ALIGNED_GAIN,
    CANDIDATE_EDGE_DIRECTION,
    MisleadingReverseDirectionValidationCase,
    PROBE_NODE_CHANNEL,
    PROBE_TO_ALIGNED,
    PROBE_TO_REVERSE,
    REVERSE_BRANCH_CHANNEL,
    REVERSE_GAIN,
    build_misleading_reverse_direction_case,
    probe_response_direction,
)


# =============================================================================
# Configuration
# =============================================================================


REVERSE_PROBE_ID = "probe_03D_reverse_direction_discrimination"

PROBE_MAGNITUDE = 0.10

MIN_MEANINGFUL_DELTA_UTILIZATION = 0.02

MIN_REVERSE_TO_ALIGNED_AMPLITUDE_RATIO = 1.40


class MisleadingReverseDirectionActiveProbeError(
    ActiveCascadeAPEError
):
    """Raised when Scenario 03D violates its reverse-direction contract."""


# =============================================================================
# Immutable audit objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class ReverseBranchProbeAssessment:
    relation_id: str
    target_channel_id: str
    scalar_amplitude: float
    alignment: float | None
    delta_utilization: float | None
    decision: VectorEvidenceDecision
    proposed_confirmed: bool
    proposed_rejected: bool
    confidence: float
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
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
                dict(
                    self.metadata
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class MisleadingReverseDirectionProbeResult:
    initial_state: ActiveCascadeState
    decision_before_probe: Any
    probe_plan: Any
    reverse_assessment: ReverseBranchProbeAssessment
    aligned_assessment: ReverseBranchProbeAssessment
    authorized_evidence: tuple[
        AuthorizedRelationEvidence,
        ...,
    ]
    final_state: ActiveCascadeState
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "authorized_evidence",
            tuple(
                self.authorized_evidence
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(
                    self.metadata
                )
            ),
        )

    @property
    def advanced(
        self,
    ) -> bool:
        return (
            self.final_state.current_node_id
            != PROBE_NODE_CHANNEL
        )

    @property
    def selected_relation_id(
        self,
    ) -> str | None:
        if not self.final_state.confirmed_edges:
            return None

        return self.final_state.confirmed_edges[
            -1
        ].relation_id


# =============================================================================
# Case helpers
# =============================================================================


def build_case(
) -> MisleadingReverseDirectionValidationCase:
    return build_misleading_reverse_direction_case()


def channel_lookup(
    case: MisleadingReverseDirectionValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            channel.channel_id: channel
            for entity
            in case.pathological_system.entities
            for channel
            in entity.channels
        }
    )


def operator_lookup(
    case: MisleadingReverseDirectionValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            operator.operator_id: operator
            for operator
            in case.pathological_system.operators
        }
    )


def branch_target_id(
    relation_id: str,
) -> str:
    if relation_id == PROBE_TO_REVERSE:
        return REVERSE_BRANCH_CHANNEL

    if relation_id == PROBE_TO_ALIGNED:
        return ALIGNED_BRANCH_CHANNEL

    raise MisleadingReverseDirectionActiveProbeError(
        f"unknown 03D branch relation: {relation_id!r}"
    )


def vector_tuple(
    vector: Any,
) -> tuple[
    float,
    float,
    float,
]:
    return (
        float(
            vector.x
        ),
        float(
            vector.y
        ),
        float(
            vector.z
        ),
    )


# =============================================================================
# Candidate provider
# =============================================================================


def candidate_from_operator(
    operator: Any,
) -> CandidateTransition:
    if len(
        operator.source_ids
    ) != 1:
        raise MisleadingReverseDirectionActiveProbeError(
            "03D branch operator must have exactly one source."
        )

    if len(
        operator.target_ids
    ) != 1:
        raise MisleadingReverseDirectionActiveProbeError(
            "03D branch operator must have exactly one target."
        )

    return CandidateTransition(
        source_id=operator.source_ids[
            0
        ],
        target_id=operator.target_ids[
            0
        ],
        relation_id=operator.operator_id,
        confidence=0.50,
        uncertainty=0.50,
        score=abs(
            float(
                operator.gain
            )
        ),
        metadata={
            "validation_case": "mechanical_03D",
            "evidence_source": (
                "passive_scalar_candidate"
            ),
            "scalar_score_only": True,
            "preferred_branch_label_used": False,
            "expected_contradiction_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


class Scenario03DCandidateProvider:
    def __init__(
        self,
        case: MisleadingReverseDirectionValidationCase,
    ) -> None:
        self._case = case
        self.calls: list[str] = []

    def candidates_for(
        self,
        current_node_id: str,
        *,
        graph_snapshot: Any,
        state: ActiveCascadeState,
    ) -> tuple[
        CandidateTransition,
        ...,
    ]:
        self.calls.append(
            current_node_id
        )

        return tuple(
            candidate_from_operator(
                operator
            )
            for operator
            in self._case.pathological_system.operators
            if (
                operator.source_ids
                == (
                    current_node_id,
                )
            )
        )


# =============================================================================
# Probe / graph uncertainty
# =============================================================================


def build_probe_definition(
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=REVERSE_PROBE_ID,
        name=(
            "03D reverse-direction discrimination Probe"
        ),
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.DYNAMIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=(
                PerturbationType.SIMULATED_INTERVENTION
            ),
            magnitude=PROBE_MAGNITUDE,
            magnitude_units="normalized",
            duration_seconds=1.0,
        ),
        description=(
            "Small reversible synthetic perturbation of the shared probe "
            "node. One candidate produces a larger but reverse-directed "
            "response and must be explicitly rejected."
        ),
        tags=(
            "validation",
            "mechanical_03D",
            "active_cascade",
            "vector_probe",
            "reverse_direction_conflict",
        ),
    )


def build_probe_registry(
) -> ProbeRegistry:
    return ProbeRegistry(
        (
            build_probe_definition(),
        )
    )


def build_probe_adapter(
) -> ProbeGraphAdapter:
    return ProbeGraphAdapter(
        (
            ProbeGraphLink(
                probe_identifier=REVERSE_PROBE_ID,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=PROBE_TO_REVERSE,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.90,
                    ),
                    ProbeGraphTarget(
                        target_identifier=PROBE_TO_ALIGNED,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.90,
                    ),
                ),
                novelty=0.90,
                feasibility=1.0,
                estimate_confidence=0.95,
                metadata={
                    "validation_case": "mechanical_03D",
                    "reverse_direction_conflict": True,
                    "preferred_branch_label_used": False,
                },
            ),
        )
    )


def build_snapshot(
) -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id=(
            "mechanical_03D_reverse_direction_conflict"
        ),
        version="v0",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PROBE_TO_REVERSE,
                    source=PROBE_NODE_CHANNEL,
                    target=REVERSE_BRANCH_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PROBE_TO_ALIGNED,
                    source=PROBE_NODE_CHANNEL,
                    target=ALIGNED_BRANCH_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
        ),
        metadata={
            "validation_case": "mechanical_03D",
            "scalar_amplitude_winner_included": False,
            "contradiction_branch_included": False,
            "directional_winner_included": False,
            "preferred_branch_label_used": False,
        },
    )


def build_bridge(
) -> ActiveCascadeAPEBridge:
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
            min_delta_utilization=(
                MIN_MEANINGFUL_DELTA_UTILIZATION
            ),
            require_capacity_relevance=False,
        ),
        metadata={
            "validation_case": "mechanical_03D",
            "phase": "reverse_direction_discrimination",
            "preferred_branch_label_used": False,
            "expected_contradiction_label_used": False,
            "expected_directional_winner_used": False,
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


def scalar_probe_amplitude(
    case: MisleadingReverseDirectionValidationCase,
    relation_id: str,
) -> float:
    operator = operator_lookup(
        case
    )[
        relation_id
    ]

    return (
        PROBE_MAGNITUDE
        * abs(
            float(
                operator.gain
            )
        )
    )


def local_probe_evidence(
    case: MisleadingReverseDirectionValidationCase,
    relation_id: str,
) -> VectorProbeEvidence:
    """
    Build 03D scalar-vector conflict evidence.

    Scalar amplitude comes from operator gain.
    Direction comes from probe_response_direction(), not from channel transport.

    For the reverse relation, utilization evidence is intentionally omitted so
    the explicit opposite vector remains the decisive evidence channel.
    """

    if relation_id not in {
        PROBE_TO_REVERSE,
        PROBE_TO_ALIGNED,
    }:
        raise MisleadingReverseDirectionActiveProbeError(
            f"unknown 03D relation: {relation_id!r}"
        )

    channels = channel_lookup(
        case
    )

    target_id = branch_target_id(
        relation_id
    )

    target_channel = channels[
        target_id
    ]

    scalar_amplitude = scalar_probe_amplitude(
        case,
        relation_id,
    )

    response_direction = vector_tuple(
        probe_response_direction(
            relation_id
        )
    )

    probe_tension = tuple(
        component
        * scalar_amplitude
        for component
        in response_direction
    )

    geometry = CandidateEdgeGeometry(
        source_id=PROBE_NODE_CHANNEL,
        target_id=target_id,
        source_position=(
            0.0,
            0.0,
            0.0,
        ),
        target_position=vector_tuple(
            CANDIDATE_EDGE_DIRECTION
        ),
        metadata={
            "validation_case": "mechanical_03D",
            "geometry_role": "candidate_edge_direction",
            "transport_direction_used": False,
            "preferred_branch_label_used": False,
        },
    )

    if relation_id == PROBE_TO_REVERSE:
        return build_vector_probe_evidence(
            relation_id=relation_id,
            geometry=geometry,
            baseline_tension=(
                0.0,
                0.0,
                0.0,
            ),
            probe_tension=probe_tension,
            metadata={
                "validation_case": "mechanical_03D",
                "probe_magnitude": PROBE_MAGNITUDE,
                "scalar_amplitude": scalar_amplitude,
                "response_direction": response_direction,
                "utilization_evidence_included": False,
                "preferred_branch_label_used": False,
                "expected_contradiction_label_used": False,
                "expected_directional_winner_used": False,
            },
        )

    capacity_state = target_channel.capacity_state

    return build_vector_probe_evidence(
        relation_id=relation_id,
        geometry=geometry,
        baseline_tension=(
            0.0,
            0.0,
            0.0,
        ),
        probe_tension=probe_tension,
        baseline_demand=float(
            capacity_state.load
        ),
        baseline_capacity=float(
            capacity_state.capacity
        ),
        probe_demand=(
            float(
                capacity_state.load
            )
            + scalar_amplitude
        ),
        probe_capacity=float(
            capacity_state.capacity
        ),
        metadata={
            "validation_case": "mechanical_03D",
            "probe_magnitude": PROBE_MAGNITUDE,
            "scalar_amplitude": scalar_amplitude,
            "response_direction": response_direction,
            "utilization_evidence_included": True,
            "preferred_branch_label_used": False,
            "expected_contradiction_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


def amplitude_ratio(
    case: MisleadingReverseDirectionValidationCase,
) -> float:
    reverse = scalar_probe_amplitude(
        case,
        PROBE_TO_REVERSE,
    )

    aligned = scalar_probe_amplitude(
        case,
        PROBE_TO_ALIGNED,
    )

    if aligned <= 0.0:
        raise MisleadingReverseDirectionActiveProbeError(
            "aligned scalar amplitude must be positive."
        )

    return reverse / aligned


# =============================================================================
# Vector assessments / authorization
# =============================================================================


def assess_relation(
    bridge: ActiveCascadeAPEBridge,
    case: MisleadingReverseDirectionValidationCase,
    relation_id: str,
) -> ReverseBranchProbeAssessment:
    evidence = local_probe_evidence(
        case,
        relation_id,
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    alignment = (
        float(
            evidence.tension.alignment
        )
        if evidence.tension is not None
        else None
    )

    delta_utilization = (
        float(
            evidence.utilization.delta_utilization
        )
        if evidence.utilization is not None
        else None
    )

    return ReverseBranchProbeAssessment(
        relation_id=relation_id,
        target_channel_id=branch_target_id(
            relation_id
        ),
        scalar_amplitude=scalar_probe_amplitude(
            case,
            relation_id,
        ),
        alignment=alignment,
        delta_utilization=delta_utilization,
        decision=proposal.decision,
        proposed_confirmed=proposal.proposed_confirmed,
        proposed_rejected=proposal.proposed_rejected,
        confidence=float(
            proposal.confidence
        ),
        metadata={
            "validation_case": "mechanical_03D",
            "preferred_branch_label_used": False,
            "expected_contradiction_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


def authorize_relation(
    bridge: ActiveCascadeAPEBridge,
    case: MisleadingReverseDirectionValidationCase,
    relation_id: str,
) -> AuthorizedRelationEvidence | None:
    evidence = local_probe_evidence(
        case,
        relation_id,
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    if (
        not proposal.proposed_confirmed
        and not proposal.proposed_rejected
    ):
        return None

    return bridge.vector_evidence.authorize(
        proposal,
        approve=True,
        metadata={
            "validation_case": "mechanical_03D",
            "preferred_branch_label_used": False,
            "expected_contradiction_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


# =============================================================================
# Complete Active Probe run
# =============================================================================


def run_misleading_reverse_direction_probe(
    case: MisleadingReverseDirectionValidationCase | None = None,
) -> MisleadingReverseDirectionProbeResult:
    case = (
        case
        or build_case()
    )

    if (
        amplitude_ratio(
            case
        )
        < MIN_REVERSE_TO_ALIGNED_AMPLITUDE_RATIO
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D reverse branch is not sufficiently louder than aligned branch."
        )

    bridge = build_bridge()
    snapshot = build_snapshot()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario03DCandidateProvider(
            case
        ),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        PROBE_NODE_CHANNEL,
        graph_version=snapshot.version,
        metadata={
            "validation_case": "mechanical_03D",
            "preferred_branch_label_used": False,
            "expected_contradiction_label_used": False,
            "expected_directional_winner_used": False,
        },
    )

    before = explorer.inspect_transition(
        state,
        graph_snapshot=snapshot,
    )

    if (
        before.status
        is not TransitionStatus.REQUIRES_PROBE
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D initial branch decision must require Probe."
        )

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=snapshot,
        decision=before,
    )

    reverse_assessment = assess_relation(
        bridge,
        case,
        PROBE_TO_REVERSE,
    )

    aligned_assessment = assess_relation(
        bridge,
        case,
        PROBE_TO_ALIGNED,
    )

    if (
        reverse_assessment.scalar_amplitude
        <= aligned_assessment.scalar_amplitude
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D scalar conflict disappeared: reverse branch is not louder."
        )

    if (
        reverse_assessment.decision
        is not VectorEvidenceDecision.CONTRADICTS
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D reverse branch was not classified as CONTRADICTS."
        )

    if (
        not reverse_assessment.proposed_rejected
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D reverse contradiction was not eligible for rejection."
        )

    if (
        aligned_assessment.decision
        is not VectorEvidenceDecision.SUPPORTS
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D failed to support aligned forward branch."
        )

    if (
        not aligned_assessment.proposed_confirmed
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D aligned support was not eligible for confirmation."
        )

    reverse_authorized = authorize_relation(
        bridge,
        case,
        PROBE_TO_REVERSE,
    )

    aligned_authorized = authorize_relation(
        bridge,
        case,
        PROBE_TO_ALIGNED,
    )

    if reverse_authorized is None:
        raise MisleadingReverseDirectionActiveProbeError(
            "03D reverse contradiction was not authorized."
        )

    if reverse_authorized.confirmed is not False:
        raise MisleadingReverseDirectionActiveProbeError(
            "03D reverse relation authorization must be negative."
        )

    if aligned_authorized is None:
        raise MisleadingReverseDirectionActiveProbeError(
            "03D aligned support was not authorized."
        )

    if aligned_authorized.confirmed is not True:
        raise MisleadingReverseDirectionActiveProbeError(
            "03D aligned relation authorization must be positive."
        )

    authorized = (
        reverse_authorized,
        aligned_authorized,
    )

    final_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=snapshot,
        graph_update=authorized,
        probe_identifier=REVERSE_PROBE_ID,
        graph_version="v1",
    )

    if (
        final_state.current_node_id
        != ALIGNED_BRANCH_CHANNEL
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D did not advance to aligned forward branch."
        )

    if len(
        final_state.confirmed_edges
    ) != 1:
        raise MisleadingReverseDirectionActiveProbeError(
            "03D must add exactly one confirmed edge."
        )

    if (
        final_state.confirmed_edges[
            0
        ].relation_id
        != PROBE_TO_ALIGNED
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D confirmed the wrong branch relation."
        )

    if (
        PROBE_TO_REVERSE
        not in final_state.rejected_relations
    ):
        raise MisleadingReverseDirectionActiveProbeError(
            "03D reverse relation was not recorded as rejected."
        )

    return MisleadingReverseDirectionProbeResult(
        initial_state=state,
        decision_before_probe=before,
        probe_plan=plan,
        reverse_assessment=reverse_assessment,
        aligned_assessment=aligned_assessment,
        authorized_evidence=authorized,
        final_state=final_state,
        metadata={
            "validation_case": "mechanical_03D",
            "control_question": (
                "does_reverse_vector_evidence_trigger_explicit_rejection"
            ),
            "reverse_scalar_amplitude": (
                reverse_assessment.scalar_amplitude
            ),
            "aligned_scalar_amplitude": (
                aligned_assessment.scalar_amplitude
            ),
            "amplitude_ratio": amplitude_ratio(
                case
            ),
            "rejected_relation_id": PROBE_TO_REVERSE,
            "selected_relation_id": PROBE_TO_ALIGNED,
            "selected_channel_id": ALIGNED_BRANCH_CHANNEL,
            "preferred_branch_label_used": False,
            "external_amplitude_winner_used": False,
            "external_contradiction_label_used": False,
            "external_directional_winner_used": False,
        },
    )


__all__ = [
    "MIN_MEANINGFUL_DELTA_UTILIZATION",
    "MIN_REVERSE_TO_ALIGNED_AMPLITUDE_RATIO",
    "PROBE_MAGNITUDE",
    "REVERSE_PROBE_ID",
    "MisleadingReverseDirectionActiveProbeError",
    "MisleadingReverseDirectionProbeResult",
    "ReverseBranchProbeAssessment",
    "Scenario03DCandidateProvider",
    "amplitude_ratio",
    "assess_relation",
    "authorize_relation",
    "branch_target_id",
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
    "run_misleading_reverse_direction_probe",
    "scalar_probe_amplitude",
]
