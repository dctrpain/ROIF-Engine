"""
ROIF Mechanical Validation
Scenario 03C вЂ” Misleading High-Amplitude Branch Active Probe

Purpose
-------
Scenario 03C tests whether Active Probe inference can reject a louder but
directionally irrelevant branch in favor of a quieter aligned branch.

Passive scalar mechanics
------------------------
The passive CapacityTensor intentionally gives:

    loud_orthogonal_branch > aligned_quiet_branch

in scalar transfer amplitude.

Active Probe evidence
---------------------
Probe-response directions are evaluated separately:

    loud response    -> orthogonal to candidate edge
    aligned response -> parallel to candidate edge

Expected inference
------------------
Before Probe:
    shared_probe_node has two candidates -> REQUIRES_PROBE

After vector assessment:
    loud relation    -> ORTHOGONAL / not authorizable
    aligned relation -> SUPPORTS / authorizable

After authorized graph update:
    Active Cascade advances only to aligned_quiet_branch.

Architectural boundaries
------------------------
This module does not use evaluator-side amplitude_winner or directional_winner
labels during inference. Those labels remain external validation targets.

The module uses:
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

from validation.mechanical.misleading_high_amplitude_branch_case import (
    ALIGNED_BRANCH_CHANNEL,
    ALIGNED_GAIN,
    CANDIDATE_EDGE_DIRECTION,
    LOUD_BRANCH_CHANNEL,
    LOUD_GAIN,
    PROBE_NODE_CHANNEL,
    PROBE_TO_ALIGNED,
    PROBE_TO_LOUD,
    MisleadingHighAmplitudeValidationCase,
    build_misleading_high_amplitude_case,
    probe_response_direction,
)


# =============================================================================
# Configuration
# =============================================================================


MISLEADING_PROBE_ID = "probe_03C_scalar_vector_discrimination"

PROBE_MAGNITUDE = 0.10

MIN_MEANINGFUL_DELTA_UTILIZATION = 0.02

# Loud branch must remain materially larger in scalar response.
MIN_LOUD_TO_ALIGNED_AMPLITUDE_RATIO = 1.40


class MisleadingHighAmplitudeActiveProbeError(
    ActiveCascadeAPEError
):
    """Raised when Scenario 03C violates its scalar-vector conflict contract."""


# =============================================================================
# Immutable audit objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class BranchProbeAssessment:
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
class MisleadingHighAmplitudeProbeResult:
    initial_state: ActiveCascadeState
    decision_before_probe: Any
    probe_plan: Any
    loud_assessment: BranchProbeAssessment
    aligned_assessment: BranchProbeAssessment
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
) -> MisleadingHighAmplitudeValidationCase:
    return build_misleading_high_amplitude_case()


def channel_lookup(
    case: MisleadingHighAmplitudeValidationCase,
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
    case: MisleadingHighAmplitudeValidationCase,
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
    if relation_id == PROBE_TO_LOUD:
        return LOUD_BRANCH_CHANNEL

    if relation_id == PROBE_TO_ALIGNED:
        return ALIGNED_BRANCH_CHANNEL

    raise MisleadingHighAmplitudeActiveProbeError(
        f"unknown 03C branch relation: {relation_id!r}"
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
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C branch operator must have exactly one source."
        )

    if len(
        operator.target_ids
    ) != 1:
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C branch operator must have exactly one target."
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
            "validation_case": "mechanical_03C",
            "evidence_source": (
                "passive_scalar_candidate"
            ),
            "scalar_score_only": True,
            "preferred_branch_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


class Scenario03CCandidateProvider:
    def __init__(
        self,
        case: MisleadingHighAmplitudeValidationCase,
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
        identifier=MISLEADING_PROBE_ID,
        name=(
            "03C scalar-vector discrimination Probe"
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
            "node. Scalar amplitude is intentionally misleading; vector "
            "alignment must determine relation support."
        ),
        tags=(
            "validation",
            "mechanical_03C",
            "active_cascade",
            "vector_probe",
            "scalar_vector_conflict",
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
                probe_identifier=MISLEADING_PROBE_ID,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=PROBE_TO_LOUD,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.80,
                    ),
                    ProbeGraphTarget(
                        target_identifier=PROBE_TO_ALIGNED,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.80,
                    ),
                ),
                novelty=0.80,
                feasibility=1.0,
                estimate_confidence=0.95,
                metadata={
                    "validation_case": "mechanical_03C",
                    "scalar_vector_conflict": True,
                    "preferred_branch_label_used": False,
                },
            ),
        )
    )


def build_snapshot(
) -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id=(
            "mechanical_03C_scalar_vector_conflict"
        ),
        version="v0",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PROBE_TO_LOUD,
                    source=PROBE_NODE_CHANNEL,
                    target=LOUD_BRANCH_CHANNEL,
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
            "validation_case": "mechanical_03C",
            "scalar_amplitude_winner_included": False,
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
            "validation_case": "mechanical_03C",
            "phase": "scalar_vector_discrimination",
            "preferred_branch_label_used": False,
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
    case: MisleadingHighAmplitudeValidationCase,
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
    case: MisleadingHighAmplitudeValidationCase,
    relation_id: str,
) -> VectorProbeEvidence:
    """
    Build scalar-vector conflict evidence.

    Scalar amplitude comes from operator gain.
    Direction comes from probe_response_direction(), not from channel transport.
    """

    if relation_id not in {
        PROBE_TO_LOUD,
        PROBE_TO_ALIGNED,
    }:
        raise MisleadingHighAmplitudeActiveProbeError(
            f"unknown 03C relation: {relation_id!r}"
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
            "validation_case": "mechanical_03C",
            "geometry_role": "candidate_edge_direction",
            "transport_direction_used": False,
            "preferred_branch_label_used": False,
        },
    )

    capacity_state = target_channel.capacity_state

    # ---------------------------------------------------------
    # 03C loud branch:
    #
    # Preserve its HIGH scalar response amplitude in the vector
    # magnitude, but do not add utilization as an independent
    # supporting evidence channel.
    #
    # This mirrors the canonical ROIF orthogonal-response test:
    # high vector magnitude alone cannot overcome wrong direction.
    # ---------------------------------------------------------

    if relation_id == PROBE_TO_LOUD:
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
                "validation_case": "mechanical_03C",
                "probe_magnitude": PROBE_MAGNITUDE,
                "scalar_amplitude": scalar_amplitude,
                "response_direction": response_direction,
                "utilization_evidence_included": False,
                "preferred_branch_label_used": False,
                "expected_directional_winner_used": False,
            },
        )

    # Aligned branch may additionally carry utilization evidence.
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
            "validation_case": "mechanical_03C",
            "probe_magnitude": PROBE_MAGNITUDE,
            "scalar_amplitude": scalar_amplitude,
            "response_direction": response_direction,
            "preferred_branch_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


def amplitude_ratio(
    case: MisleadingHighAmplitudeValidationCase,
) -> float:
    loud = scalar_probe_amplitude(
        case,
        PROBE_TO_LOUD,
    )

    aligned = scalar_probe_amplitude(
        case,
        PROBE_TO_ALIGNED,
    )

    if aligned <= 0.0:
        raise MisleadingHighAmplitudeActiveProbeError(
            "aligned scalar amplitude must be positive."
        )

    return loud / aligned


# =============================================================================
# Vector assessments / authorization
# =============================================================================


def assess_relation(
    bridge: ActiveCascadeAPEBridge,
    case: MisleadingHighAmplitudeValidationCase,
    relation_id: str,
) -> BranchProbeAssessment:
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

    return BranchProbeAssessment(
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
            "validation_case": "mechanical_03C",
            "preferred_branch_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


def authorize_if_supported(
    bridge: ActiveCascadeAPEBridge,
    case: MisleadingHighAmplitudeValidationCase,
    relation_id: str,
) -> AuthorizedRelationEvidence | None:
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
            "validation_case": "mechanical_03C",
            "preferred_branch_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


# =============================================================================
# Complete Active Probe run
# =============================================================================


def run_misleading_high_amplitude_probe(
    case: MisleadingHighAmplitudeValidationCase | None = None,
) -> MisleadingHighAmplitudeProbeResult:
    case = (
        case
        or build_case()
    )

    if (
        amplitude_ratio(
            case
        )
        < MIN_LOUD_TO_ALIGNED_AMPLITUDE_RATIO
    ):
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C loud branch is not sufficiently louder than aligned branch."
        )

    bridge = build_bridge()
    snapshot = build_snapshot()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario03CCandidateProvider(
            case
        ),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        PROBE_NODE_CHANNEL,
        graph_version=snapshot.version,
        metadata={
            "validation_case": "mechanical_03C",
            "preferred_branch_label_used": False,
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
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C initial branch decision must require Probe."
        )

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=snapshot,
        decision=before,
    )

    loud_assessment = assess_relation(
        bridge,
        case,
        PROBE_TO_LOUD,
    )

    aligned_assessment = assess_relation(
        bridge,
        case,
        PROBE_TO_ALIGNED,
    )

    if (
        loud_assessment.scalar_amplitude
        <= aligned_assessment.scalar_amplitude
    ):
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C scalar conflict disappeared: loud branch is not louder."
        )

    if (
        loud_assessment.decision
        is VectorEvidenceDecision.SUPPORTS
    ):
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C incorrectly supported the orthogonal loud branch."
        )

    if (
        aligned_assessment.decision
        is not VectorEvidenceDecision.SUPPORTS
    ):
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C failed to support the aligned quiet branch."
        )

    authorized = tuple(
        item
        for item in (
            authorize_if_supported(
                bridge,
                case,
                PROBE_TO_LOUD,
            ),
            authorize_if_supported(
                bridge,
                case,
                PROBE_TO_ALIGNED,
            ),
        )
        if item is not None
    )

    if len(
        authorized
    ) != 1:
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C must produce exactly one authorized supportive relation."
        )

    if (
        authorized[
            0
        ].relation_id
        != PROBE_TO_ALIGNED
    ):
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C authorized the wrong relation."
        )

    final_state = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=snapshot,
        graph_update=authorized,
        probe_identifier=MISLEADING_PROBE_ID,
        graph_version="v1",
    )

    if (
        final_state.current_node_id
        != ALIGNED_BRANCH_CHANNEL
    ):
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C did not advance to aligned quiet branch."
        )

    if len(
        final_state.confirmed_edges
    ) != 1:
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C must add exactly one confirmed edge."
        )

    if (
        final_state.confirmed_edges[
            0
        ].relation_id
        != PROBE_TO_ALIGNED
    ):
        raise MisleadingHighAmplitudeActiveProbeError(
            "03C confirmed the wrong branch relation."
        )

    return MisleadingHighAmplitudeProbeResult(
        initial_state=state,
        decision_before_probe=before,
        probe_plan=plan,
        loud_assessment=loud_assessment,
        aligned_assessment=aligned_assessment,
        authorized_evidence=authorized,
        final_state=final_state,
        metadata={
            "validation_case": "mechanical_03C",
            "control_question": (
                "does_vector_evidence_reject_louder_orthogonal_response"
            ),
            "loud_scalar_amplitude": (
                loud_assessment.scalar_amplitude
            ),
            "aligned_scalar_amplitude": (
                aligned_assessment.scalar_amplitude
            ),
            "amplitude_ratio": amplitude_ratio(
                case
            ),
            "selected_relation_id": PROBE_TO_ALIGNED,
            "selected_channel_id": ALIGNED_BRANCH_CHANNEL,
            "preferred_branch_label_used": False,
            "external_amplitude_winner_used": False,
            "external_directional_winner_used": False,
        },
    )


__all__ = [
    "BranchProbeAssessment",
    "MIN_LOUD_TO_ALIGNED_AMPLITUDE_RATIO",
    "MIN_MEANINGFUL_DELTA_UTILIZATION",
    "MISLEADING_PROBE_ID",
    "MisleadingHighAmplitudeActiveProbeError",
    "MisleadingHighAmplitudeProbeResult",
    "PROBE_MAGNITUDE",
    "Scenario03CCandidateProvider",
    "amplitude_ratio",
    "assess_relation",
    "authorize_if_supported",
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
    "run_misleading_high_amplitude_probe",
    "scalar_probe_amplitude",
]

