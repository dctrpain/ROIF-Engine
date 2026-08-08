"""
ROIF Mechanical Validation
Scenario 03E вЂ” Delayed Misleading Response Active Probe

Purpose
-------
Scenario 03E tests whether Active Probe inference can resist premature causal
selection when an early response appears before the true delayed response.

Temporal control
----------------
Early observation window:
    fast_misleading_branch is visible
    delayed_true_branch is not yet visible

Full observation window:
    both branches are visible

Expected behavior
-----------------
1. Initial branch decision -> REQUIRES_PROBE
2. Early observation:
       fast response observed
       delayed response absent
       NO graph advance
       NO confirmed edge
3. Full observation:
       delayed response appears
       both candidate relations can be assessed
4. Temporal policy:
       first response alone is insufficient for confirmation
       delayed branch receives stronger final causal support
5. Authorized update:
       delayed relation confirmed
       fast relation remains unconfirmed
6. Final state:
       advances to delayed_true_branch

Architectural boundaries
------------------------
Evaluator-side first_response_branch and directional_winner labels are not used
during inference. Temporal visibility is read from scenario response-delay
constants, while causal support is produced independently by the evidence rule.

This is a synthetic mechanical validation model.
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

from validation.mechanical.delayed_misleading_response_case import (
    CANDIDATE_EDGE_DIRECTION,
    DELAYED_BRANCH_CHANNEL,
    DELAYED_RESPONSE_DELAY_SECONDS,
    EARLY_OBSERVATION_WINDOW_SECONDS,
    FAST_BRANCH_CHANNEL,
    FAST_RESPONSE_DELAY_SECONDS,
    FULL_OBSERVATION_WINDOW_SECONDS,
    PROBE_NODE_CHANNEL,
    PROBE_TO_DELAYED,
    PROBE_TO_FAST,
    DelayedMisleadingResponseValidationCase,
    build_delayed_misleading_response_case,
    is_visible_in_window,
    probe_response_direction,
)


# =============================================================================
# Configuration
# =============================================================================


DELAYED_PROBE_ID = "probe_03E_delayed_response_discrimination"

PROBE_MAGNITUDE = 0.10

MIN_MEANINGFUL_DELTA_UTILIZATION = 0.02

# Final delayed branch must have a material evidence advantage.
MIN_DELAYED_SUPPORT_MARGIN = 0.02


class DelayedMisleadingResponseActiveProbeError(
    ActiveCascadeAPEError
):
    """Raised when Scenario 03E violates its temporal-control contract."""


# =============================================================================
# Immutable audit objects
# =============================================================================


@dataclass(frozen=True, slots=True)
class TemporalProbeAssessment:
    relation_id: str
    target_channel_id: str
    visible: bool
    response_delay_seconds: float
    observation_window_seconds: float
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
class DelayedMisleadingResponseProbeResult:
    initial_state: ActiveCascadeState
    decision_before_probe: Any
    probe_plan: Any

    early_fast_assessment: TemporalProbeAssessment
    early_delayed_assessment: TemporalProbeAssessment

    early_state: ActiveCascadeState

    full_fast_assessment: TemporalProbeAssessment
    full_delayed_assessment: TemporalProbeAssessment

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
    def advanced_early(
        self,
    ) -> bool:
        return (
            self.early_state.current_node_id
            != PROBE_NODE_CHANNEL
        )

    @property
    def advanced_final(
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
) -> DelayedMisleadingResponseValidationCase:
    return build_delayed_misleading_response_case()


def channel_lookup(
    case: DelayedMisleadingResponseValidationCase,
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
    case: DelayedMisleadingResponseValidationCase,
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
    if relation_id == PROBE_TO_FAST:
        return FAST_BRANCH_CHANNEL

    if relation_id == PROBE_TO_DELAYED:
        return DELAYED_BRANCH_CHANNEL

    raise DelayedMisleadingResponseActiveProbeError(
        f"unknown 03E branch relation: {relation_id!r}"
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
        raise DelayedMisleadingResponseActiveProbeError(
            "03E branch operator must have exactly one source."
        )

    if len(
        operator.target_ids
    ) != 1:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E branch operator must have exactly one target."
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
            "validation_case": "mechanical_03E",
            "evidence_source": "passive_scalar_candidate",
            "temporal_precedence_not_confirmation": True,
            "preferred_branch_label_used": False,
            "expected_first_response_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


class Scenario03ECandidateProvider:
    def __init__(
        self,
        case: DelayedMisleadingResponseValidationCase,
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
            if operator.source_ids
            == (
                current_node_id,
            )
        )


# =============================================================================
# Probe / graph uncertainty
# =============================================================================


def build_probe_definition(
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=DELAYED_PROBE_ID,
        name="03E delayed-response discrimination Probe",
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.DYNAMIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=PerturbationType.SIMULATED_INTERVENTION,
            magnitude=PROBE_MAGNITUDE,
            magnitude_units="normalized",
            duration_seconds=FULL_OBSERVATION_WINDOW_SECONDS,
        ),
        description=(
            "Synthetic reversible perturbation with an early observation "
            "window and a full observation window."
        ),
        tags=(
            "validation",
            "mechanical_03E",
            "active_cascade",
            "temporal_probe",
            "delayed_response",
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
                probe_identifier=DELAYED_PROBE_ID,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=PROBE_TO_FAST,
                        sensitivity=1.0,
                        discrimination=0.70,
                        expected_uncertainty_reduction=0.60,
                    ),
                    ProbeGraphTarget(
                        target_identifier=PROBE_TO_DELAYED,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=0.90,
                    ),
                ),
                novelty=0.85,
                feasibility=1.0,
                estimate_confidence=0.95,
                metadata={
                    "validation_case": "mechanical_03E",
                    "temporal_conflict": True,
                    "preferred_branch_label_used": False,
                },
            ),
        )
    )


def build_snapshot(
) -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id="mechanical_03E_delayed_response_conflict",
        version="v0",
        uncertainties=(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PROBE_TO_FAST,
                    source=PROBE_NODE_CHANNEL,
                    target=FAST_BRANCH_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=PROBE_TO_DELAYED,
                    source=PROBE_NODE_CHANNEL,
                    target=DELAYED_BRANCH_CHANNEL,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=2,
            ),
        ),
        metadata={
            "validation_case": "mechanical_03E",
            "first_response_branch_included": False,
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
            "validation_case": "mechanical_03E",
            "phase": "temporal_discrimination",
            "preferred_branch_label_used": False,
            "expected_first_response_label_used": False,
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
# Temporal evidence model
# =============================================================================


def scalar_probe_amplitude(
    case: DelayedMisleadingResponseValidationCase,
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


def response_delay_seconds(
    relation_id: str,
) -> float:
    if relation_id == PROBE_TO_FAST:
        return FAST_RESPONSE_DELAY_SECONDS

    if relation_id == PROBE_TO_DELAYED:
        return DELAYED_RESPONSE_DELAY_SECONDS

    raise DelayedMisleadingResponseActiveProbeError(
        f"unknown 03E relation: {relation_id!r}"
    )


def local_probe_evidence(
    case: DelayedMisleadingResponseValidationCase,
    relation_id: str,
    *,
    observation_window_seconds: float,
) -> VectorProbeEvidence:
    """
    Build evidence for one temporal observation window.

    Invisible relation:
        no tension/utilization evidence -> INSUFFICIENT

    Fast visible relation:
        forward vector evidence exists, but utilization evidence is omitted.
        This prevents "first observed" from becoming enough for confirmation.

    Delayed visible relation in the full window:
        forward vector + meaningful utilization evidence -> SUPPORTS.
    """

    if relation_id not in {
        PROBE_TO_FAST,
        PROBE_TO_DELAYED,
    }:
        raise DelayedMisleadingResponseActiveProbeError(
            f"unknown 03E relation: {relation_id!r}"
        )

    visible = is_visible_in_window(
        relation_id,
        window_seconds=observation_window_seconds,
    )

    target_id = branch_target_id(
        relation_id
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
            "validation_case": "mechanical_03E",
            "observation_window_seconds": observation_window_seconds,
            "preferred_branch_label_used": False,
        },
    )

    if not visible:
        return build_vector_probe_evidence(
            relation_id=relation_id,
            geometry=geometry,
            metadata={
                "validation_case": "mechanical_03E",
                "visible": False,
                "observation_window_seconds": observation_window_seconds,
                "response_delay_seconds": response_delay_seconds(
                    relation_id
                ),
                "preferred_branch_label_used": False,
                "expected_first_response_label_used": False,
                "expected_directional_winner_used": False,
            },
        )

    amplitude = scalar_probe_amplitude(
        case,
        relation_id,
    )

    direction = vector_tuple(
        probe_response_direction(
            relation_id
        )
    )

    probe_tension = tuple(
        component
        * amplitude
        for component
        in direction
    )

    # Early/fast response is observable but deliberately not sufficient
    # to authorize a causal edge.
    if relation_id == PROBE_TO_FAST:
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
                "validation_case": "mechanical_03E",
                "visible": True,
                "observation_window_seconds": observation_window_seconds,
                "response_delay_seconds": FAST_RESPONSE_DELAY_SECONDS,
                "temporal_precedence_only": True,
                "utilization_evidence_included": False,
                "preferred_branch_label_used": False,
                "expected_first_response_label_used": False,
                "expected_directional_winner_used": False,
            },
        )

    channels = channel_lookup(
        case
    )

    capacity_state = channels[
        target_id
    ].capacity_state

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
            + amplitude
        ),
        probe_capacity=float(
            capacity_state.capacity
        ),
        metadata={
            "validation_case": "mechanical_03E",
            "visible": True,
            "observation_window_seconds": observation_window_seconds,
            "response_delay_seconds": DELAYED_RESPONSE_DELAY_SECONDS,
            "temporal_precedence_only": False,
            "utilization_evidence_included": True,
            "preferred_branch_label_used": False,
            "expected_first_response_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


def assess_relation(
    bridge: ActiveCascadeAPEBridge,
    case: DelayedMisleadingResponseValidationCase,
    relation_id: str,
    *,
    observation_window_seconds: float,
) -> TemporalProbeAssessment:
    evidence = local_probe_evidence(
        case,
        relation_id,
        observation_window_seconds=observation_window_seconds,
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

    visible = bool(
        evidence.metadata.get(
            "visible",
            False,
        )
    )

    return TemporalProbeAssessment(
        relation_id=relation_id,
        target_channel_id=branch_target_id(
            relation_id
        ),
        visible=visible,
        response_delay_seconds=response_delay_seconds(
            relation_id
        ),
        observation_window_seconds=observation_window_seconds,
        scalar_amplitude=(
            scalar_probe_amplitude(
                case,
                relation_id,
            )
            if visible
            else 0.0
        ),
        alignment=alignment,
        delta_utilization=delta_utilization,
        decision=proposal.decision,

        # 03E temporal authorization gate:
        #
        # Vector SUPPORTS alone is not sufficient for a causal transition.
        # A relation is confirmable only after it is visible AND carries
        # utilization evidence from the complete observation.
        proposed_confirmed=(
            proposal.proposed_confirmed
            and visible
            and evidence.utilization is not None
        ),

        proposed_rejected=proposal.proposed_rejected,
        confidence=float(
            proposal.confidence
        ),
        metadata={
            "validation_case": "mechanical_03E",
            "visible": visible,
            "preferred_branch_label_used": False,
            "expected_first_response_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


def authorize_if_supported(
    bridge: ActiveCascadeAPEBridge,
    case: DelayedMisleadingResponseValidationCase,
    relation_id: str,
    *,
    observation_window_seconds: float,
) -> AuthorizedRelationEvidence | None:
    evidence = local_probe_evidence(
        case,
        relation_id,
        observation_window_seconds=observation_window_seconds,
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    # 03E temporal authorization boundary.
    #
    # A forward response can be vector-supportive while still being only
    # temporal-precedence evidence. Confirmation requires evidence collected
    # after the relation is visible AND utilization relevance is available.
    if (
        not proposal.proposed_confirmed
        or evidence.utilization is None
        or not bool(
            evidence.metadata.get(
                "visible",
                False,
            )
        )
    ):
        return None

    return bridge.vector_evidence.authorize(
        proposal,
        approve=True,
        metadata={
            "validation_case": "mechanical_03E",
            "observation_window_seconds": observation_window_seconds,
            "preferred_branch_label_used": False,
            "expected_first_response_label_used": False,
            "expected_directional_winner_used": False,
        },
    )


# =============================================================================
# Complete Active Probe run
# =============================================================================


def run_delayed_misleading_response_probe(
    case: DelayedMisleadingResponseValidationCase | None = None,
) -> DelayedMisleadingResponseProbeResult:
    case = (
        case
        or build_case()
    )

    bridge = build_bridge()
    snapshot = build_snapshot()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario03ECandidateProvider(
            case
        ),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        PROBE_NODE_CHANNEL,
        graph_version=snapshot.version,
        metadata={
            "validation_case": "mechanical_03E",
            "preferred_branch_label_used": False,
            "expected_first_response_label_used": False,
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
        raise DelayedMisleadingResponseActiveProbeError(
            "03E initial branch decision must require Probe."
        )

    plan = explorer.plan_required_probe(
        state,
        graph_snapshot=snapshot,
        decision=before,
    )

    # ---------------------------------------------------------------------
    # Early window
    # ---------------------------------------------------------------------

    early_fast = assess_relation(
        bridge,
        case,
        PROBE_TO_FAST,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    early_delayed = assess_relation(
        bridge,
        case,
        PROBE_TO_DELAYED,
        observation_window_seconds=(
            EARLY_OBSERVATION_WINDOW_SECONDS
        ),
    )

    if not early_fast.visible:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E fast response must be visible in the early window."
        )

    if early_delayed.visible:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E delayed response must remain invisible in the early window."
        )

    if early_fast.proposed_confirmed:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E early fast response must not be confirmable."
        )

    # No graph update is applied during the early window.
    early_state = state

    if (
        early_state.current_node_id
        != PROBE_NODE_CHANNEL
    ):
        raise DelayedMisleadingResponseActiveProbeError(
            "03E advanced prematurely during the early observation window."
        )

    # ---------------------------------------------------------------------
    # Full window
    # ---------------------------------------------------------------------

    full_fast = assess_relation(
        bridge,
        case,
        PROBE_TO_FAST,
        observation_window_seconds=(
            FULL_OBSERVATION_WINDOW_SECONDS
        ),
    )

    full_delayed = assess_relation(
        bridge,
        case,
        PROBE_TO_DELAYED,
        observation_window_seconds=(
            FULL_OBSERVATION_WINDOW_SECONDS
        ),
    )

    if not full_fast.visible:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E fast response must remain visible in the full window."
        )

    if not full_delayed.visible:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E delayed response must appear in the full window."
        )

    if full_fast.proposed_confirmed:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E fast response must remain non-confirmable after full observation."
        )

    if (
        full_delayed.decision
        is not VectorEvidenceDecision.SUPPORTS
    ):
        raise DelayedMisleadingResponseActiveProbeError(
            "03E delayed branch did not receive SUPPORTS."
        )

    if not full_delayed.proposed_confirmed:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E delayed branch was not eligible for confirmation."
        )

    delayed_authorized = authorize_if_supported(
        bridge,
        case,
        PROBE_TO_DELAYED,
        observation_window_seconds=(
            FULL_OBSERVATION_WINDOW_SECONDS
        ),
    )

    if delayed_authorized is None:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E delayed support was not authorized."
        )

    if delayed_authorized.confirmed is not True:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E delayed authorization must be positive."
        )

    authorized = (
        delayed_authorized,
    )

    final_state = explorer.apply_authorized_probe_update(
        early_state,
        graph_snapshot=snapshot,
        graph_update=authorized,
        probe_identifier=DELAYED_PROBE_ID,
        graph_version="v1",
    )

    if (
        final_state.current_node_id
        != DELAYED_BRANCH_CHANNEL
    ):
        raise DelayedMisleadingResponseActiveProbeError(
            "03E did not advance to delayed true branch."
        )

    if len(
        final_state.confirmed_edges
    ) != 1:
        raise DelayedMisleadingResponseActiveProbeError(
            "03E must add exactly one confirmed edge."
        )

    if (
        final_state.confirmed_edges[
            0
        ].relation_id
        != PROBE_TO_DELAYED
    ):
        raise DelayedMisleadingResponseActiveProbeError(
            "03E confirmed the wrong branch relation."
        )

    return DelayedMisleadingResponseProbeResult(
        initial_state=state,
        decision_before_probe=before,
        probe_plan=plan,
        early_fast_assessment=early_fast,
        early_delayed_assessment=early_delayed,
        early_state=early_state,
        full_fast_assessment=full_fast,
        full_delayed_assessment=full_delayed,
        authorized_evidence=authorized,
        final_state=final_state,
        metadata={
            "validation_case": "mechanical_03E",
            "control_question": (
                "does_active_probe_wait_for_delayed_evidence_before_advancing"
            ),
            "early_window_seconds": (
                EARLY_OBSERVATION_WINDOW_SECONDS
            ),
            "full_window_seconds": (
                FULL_OBSERVATION_WINDOW_SECONDS
            ),
            "fast_response_delay_seconds": (
                FAST_RESPONSE_DELAY_SECONDS
            ),
            "delayed_response_delay_seconds": (
                DELAYED_RESPONSE_DELAY_SECONDS
            ),
            "advanced_early": False,
            "selected_relation_id": PROBE_TO_DELAYED,
            "selected_channel_id": DELAYED_BRANCH_CHANNEL,
            "preferred_branch_label_used": False,
            "external_first_response_label_used": False,
            "external_directional_winner_used": False,
        },
    )


__all__ = [
    "DELAYED_PROBE_ID",
    "MIN_DELAYED_SUPPORT_MARGIN",
    "MIN_MEANINGFUL_DELTA_UTILIZATION",
    "PROBE_MAGNITUDE",
    "DelayedMisleadingResponseActiveProbeError",
    "DelayedMisleadingResponseProbeResult",
    "Scenario03ECandidateProvider",
    "TemporalProbeAssessment",
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
    "response_delay_seconds",
    "run_delayed_misleading_response_probe",
    "scalar_probe_amplitude",
]


