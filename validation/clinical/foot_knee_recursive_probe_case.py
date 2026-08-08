"""
ROIF Validation Suite
Clinical Scenario 01C:
Foot-Knee Recursive Active Probe Reconstruction

Purpose
-------
Scenario 01A:
    Known graph -> ROIF causal-role inference.

Scenario 01B:
    Incomplete graph -> one blind Probe* -> GraphUpdateProposal.

Scenario 01C:
    Incomplete graph
        -> Probe*
        -> observation
        -> authorized graph update
        -> reduced uncertainty
        -> next Probe*
        -> observation
        -> authorized graph update
        -> reconstructed graph

The final reconstructed graph can then be passed back to the normal
ROIF Solver and compared with the external Scenario 01A labels:

    D_origin
    D_fast
    D_root
    Node*

Scientific boundary
-------------------
Hidden reference structure is evaluator-side truth.

It may be queried only after a Probe has been independently selected,
to simulate the resulting observation. A GraphUpdateProposal never
mutates the graph automatically. Structural mutation requires explicit
authorization by this validation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from roif.probe_graph_adapter import (
    GraphTarget,
    GraphUncertainty,
    IncompleteGraphSnapshot,
)
from roif.roif_entities import (
    InfluenceOperator,
    ROIFSystem,
)

from validation.clinical.foot_knee_case import (
    ClinicalValidationLabels,
    build_foot_knee_case,
)
from validation.clinical.foot_knee_probe_case import (
    FootKneeProbeValidationCase,
    HiddenEdgeTruth,
    UncertainRelation,
    build_foot_knee_probe_case,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FootKneeRecursiveProbeValidationError(ValueError):
    """Raised when Scenario 01C violates its validation contract."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _operator_map(
    system: ROIFSystem,
) -> Mapping[str, InfluenceOperator]:
    return MappingProxyType(
        {
            operator.operator_id: operator
            for operator in system.operators
        }
    )


def _relation_map(
    case: FootKneeProbeValidationCase,
) -> Mapping[str, UncertainRelation]:
    return MappingProxyType(
        {
            relation.relation_id: relation
            for relation in case.uncertain_relations
        }
    )


def _hidden_edge_map(
    case: FootKneeProbeValidationCase,
) -> Mapping[str, HiddenEdgeTruth]:
    return MappingProxyType(
        {
            edge.operator_id: edge
            for edge in case.hidden_edges
        }
    )


# ---------------------------------------------------------------------------
# Recursive reconstruction state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthorizedGraphUpdate:
    """
    One explicitly authorized structural update.

    proposal_identifier
        Identifier of the Probe that produced the evidence.

    relation_id
        Public uncertain relation resolved by that Probe.

    operator_id
        Operator inserted after authorization.

    graph_version_before / graph_version_after
        Explicit reconstruction version transition.

    authorization
        Must be True. Scenario 01C never applies an implicit update.
    """

    proposal_identifier: str
    relation_id: str
    operator_id: str
    graph_version_before: str
    graph_version_after: str
    authorization: bool
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_identifier:
            raise FootKneeRecursiveProbeValidationError(
                "proposal_identifier must not be empty."
            )

        if not self.relation_id:
            raise FootKneeRecursiveProbeValidationError(
                "relation_id must not be empty."
            )

        if not self.operator_id:
            raise FootKneeRecursiveProbeValidationError(
                "operator_id must not be empty."
            )

        if not self.graph_version_before:
            raise FootKneeRecursiveProbeValidationError(
                "graph_version_before must not be empty."
            )

        if not self.graph_version_after:
            raise FootKneeRecursiveProbeValidationError(
                "graph_version_after must not be empty."
            )

        if self.graph_version_before == self.graph_version_after:
            raise FootKneeRecursiveProbeValidationError(
                "authorized update must advance graph version."
            )

        if self.authorization is not True:
            raise FootKneeRecursiveProbeValidationError(
                "Scenario 01C requires explicit authorization."
            )

        confidence = float(self.confidence)

        if not 0.0 <= confidence <= 1.0:
            raise FootKneeRecursiveProbeValidationError(
                "confidence must be in [0, 1]."
            )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class RecursiveProbeState:
    """
    Immutable state of Scenario 01C after zero or more authorized Probes.
    """

    case_id: str
    graph_version: str
    system: ROIFSystem
    unresolved_relations: tuple[UncertainRelation, ...]
    resolved_relation_ids: tuple[str, ...] = ()
    authorized_updates: tuple[AuthorizedGraphUpdate, ...] = ()
    expected_roles: ClinicalValidationLabels | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise FootKneeRecursiveProbeValidationError(
                "case_id must not be empty."
            )

        if not self.graph_version:
            raise FootKneeRecursiveProbeValidationError(
                "graph_version must not be empty."
            )

        object.__setattr__(
            self,
            "unresolved_relations",
            tuple(self.unresolved_relations),
        )
        object.__setattr__(
            self,
            "resolved_relation_ids",
            tuple(self.resolved_relation_ids),
        )
        object.__setattr__(
            self,
            "authorized_updates",
            tuple(self.authorized_updates),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )

        unresolved_ids = tuple(
            relation.relation_id
            for relation in self.unresolved_relations
        )

        if len(set(unresolved_ids)) != len(unresolved_ids):
            raise FootKneeRecursiveProbeValidationError(
                "unresolved relation IDs must be unique."
            )

        if len(set(self.resolved_relation_ids)) != len(
            self.resolved_relation_ids
        ):
            raise FootKneeRecursiveProbeValidationError(
                "resolved relation IDs must be unique."
            )

        overlap = set(unresolved_ids).intersection(
            self.resolved_relation_ids
        )

        if overlap:
            raise FootKneeRecursiveProbeValidationError(
                "a relation cannot be both resolved and unresolved."
            )

    @property
    def operator_ids(self) -> tuple[str, ...]:
        return tuple(
            operator.operator_id
            for operator in self.system.operators
        )

    @property
    def unresolved_relation_ids(self) -> tuple[str, ...]:
        return tuple(
            relation.relation_id
            for relation in self.unresolved_relations
        )

    @property
    def is_fully_reconstructed(self) -> bool:
        return not self.unresolved_relations

    @property
    def probe_count(self) -> int:
        return len(self.authorized_updates)


# ---------------------------------------------------------------------------
# Scenario 01C initialization
# ---------------------------------------------------------------------------


def build_recursive_probe_initial_state() -> RecursiveProbeState:
    """
    Start Scenario 01C from the same intentionally incomplete graph as 01B.
    """

    case = build_foot_knee_probe_case()

    return RecursiveProbeState(
        case_id="clinical_01C_foot_knee_recursive_probe",
        graph_version="clinical-01C-v0",
        system=case.public_system,
        unresolved_relations=case.uncertain_relations,
        resolved_relation_ids=(),
        authorized_updates=(),
        expected_roles=case.expected_roles,
        metadata={
            "validation_family": "clinical_01_foot_knee",
            "validation_stage": "01C",
            "parent_case": case.case_id,
            "purpose": "recursive_active_probe_reconstruction",
            "hidden_graph_exposed_to_planner": False,
            "hidden_labels_exposed_to_planner": False,
            "automatic_graph_mutation": False,
            "explicit_authorization_required": True,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


# ---------------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------------


def _target_for_relation(
    relation: UncertainRelation,
) -> GraphTarget:
    """
    Convert one public uncertainty into the real APE graph target model.

    GraphTargetKind is resolved lazily to avoid duplicating enum spelling.
    """

    from roif.probe_graph_adapter import GraphTargetKind

    edge_kind = next(
        (
            member
            for member in GraphTargetKind
            if member.value == "edge"
        ),
        None,
    )

    if edge_kind is None:
        raise FootKneeRecursiveProbeValidationError(
            "GraphTargetKind must support edge targets."
        )

    return GraphTarget(
        kind=edge_kind,
        identifier=relation.relation_id,
        source=relation.source_id,
        target=relation.target_id,
        metadata={
            "validation_stage": "01C",
            "ground_truth_included": False,
        },
    )


def build_recursive_snapshot(
    state: RecursiveProbeState,
) -> IncompleteGraphSnapshot:
    """
    Build the next APE snapshot from unresolved public uncertainty only.
    """

    uncertainties = tuple(
        GraphUncertainty(
            target=_target_for_relation(relation),
            uncertainty=relation.uncertainty,
            importance=relation.importance,
            confidence=relation.confidence,
            hypothesis_count=relation.hypothesis_count,
            metadata={
                "validation_stage": "01C",
                "ground_truth_included": False,
            },
        )
        for relation in state.unresolved_relations
    )

    return IncompleteGraphSnapshot(
        graph_id=state.system.system_id,
        version=state.graph_version,
        uncertainties=uncertainties,
        metadata={
            "validation_case": state.case_id,
            "resolved_relation_ids": state.resolved_relation_ids,
            "hidden_graph_exposed": False,
            "hidden_labels_exposed": False,
            "automatic_graph_mutation": False,
        },
    )


# ---------------------------------------------------------------------------
# Evaluator-side truth access
# ---------------------------------------------------------------------------


def hidden_reference_operator(
    relation_id: str,
    *,
    case: FootKneeProbeValidationCase | None = None,
) -> InfluenceOperator:
    """
    Return one hidden reference operator.

    VALIDATION HARNESS ONLY.

    This function must be called only after Probe* selection and
    observation evaluation.
    """

    validation_case = case or build_foot_knee_probe_case()

    hidden = _hidden_edge_map(validation_case)

    if relation_id not in hidden:
        raise FootKneeRecursiveProbeValidationError(
            f"{relation_id!r} is not a hidden validation relation."
        )

    reference_operators = _operator_map(
        validation_case.hidden_reference_system
    )

    try:
        return reference_operators[relation_id]
    except KeyError as exc:
        raise FootKneeRecursiveProbeValidationError(
            f"hidden reference operator {relation_id!r} is missing."
        ) from exc


# ---------------------------------------------------------------------------
# Explicit authorization and graph reconstruction
# ---------------------------------------------------------------------------


def authorize_relation_update(
    state: RecursiveProbeState,
    *,
    probe_identifier: str,
    relation_id: str,
    observation_confidence: float,
    authorized: bool,
) -> RecursiveProbeState:
    """
    Apply one evidence-backed relation only after explicit authorization.

    The inserted operator is copied from hidden evaluator truth only after
    the selected Probe has produced supporting evidence.

    This function intentionally does not accept expected D_* labels.
    """

    if not authorized:
        raise FootKneeRecursiveProbeValidationError(
            "graph update rejected: explicit authorization is required."
        )

    relation_by_id = {
        relation.relation_id: relation
        for relation in state.unresolved_relations
    }

    if relation_id not in relation_by_id:
        raise FootKneeRecursiveProbeValidationError(
            f"{relation_id!r} is not currently unresolved."
        )

    confidence = float(observation_confidence)

    if not 0.0 <= confidence <= 1.0:
        raise FootKneeRecursiveProbeValidationError(
            "observation_confidence must be in [0, 1]."
        )

    validation_case = build_foot_knee_probe_case()

    operator = hidden_reference_operator(
        relation_id,
        case=validation_case,
    )

    current_ids = {
        item.operator_id
        for item in state.system.operators
    }

    if operator.operator_id in current_ids:
        raise FootKneeRecursiveProbeValidationError(
            f"operator {operator.operator_id!r} is already present."
        )

    next_operators = (
        tuple(state.system.operators)
        + (operator,)
    )

    next_version_index = state.probe_count + 1
    next_version = f"clinical-01C-v{next_version_index}"

    next_metadata = dict(state.system.metadata)
    next_metadata.update(
        {
            "validation_case": "foot_knee_recursive_probe",
            "validation_stage": "01C",
            "graph_completeness": (
                "complete"
                if len(state.unresolved_relations) == 1
                else "incomplete"
            ),
            "last_authorized_relation": relation_id,
            "automatic_graph_mutation": False,
        }
    )

    next_system = ROIFSystem(
        system_id=state.system.system_id,
        name=state.system.name,
        entities=state.system.entities,
        planes=state.system.planes,
        operators=next_operators,
        metadata=next_metadata,
    )

    remaining = tuple(
        relation
        for relation in state.unresolved_relations
        if relation.relation_id != relation_id
    )

    update = AuthorizedGraphUpdate(
        proposal_identifier=probe_identifier,
        relation_id=relation_id,
        operator_id=operator.operator_id,
        graph_version_before=state.graph_version,
        graph_version_after=next_version,
        authorization=True,
        confidence=confidence,
        metadata={
            "validation_stage": "01C",
            "source": "post_probe_observation",
            "hidden_truth_used_before_selection": False,
        },
    )

    return RecursiveProbeState(
        case_id=state.case_id,
        graph_version=next_version,
        system=next_system,
        unresolved_relations=remaining,
        resolved_relation_ids=(
            *state.resolved_relation_ids,
            relation_id,
        ),
        authorized_updates=(
            *state.authorized_updates,
            update,
        ),
        expected_roles=state.expected_roles,
        metadata=state.metadata,
    )


# ---------------------------------------------------------------------------
# Reconstruction checks
# ---------------------------------------------------------------------------


def reference_operator_ids() -> tuple[str, ...]:
    return tuple(
        operator.operator_id
        for operator in build_foot_knee_case().pathological_system.operators
    )


def reconstruction_matches_reference_topology(
    state: RecursiveProbeState,
) -> bool:
    """
    Compare operator identity, not object identity or hidden labels.
    """

    return set(state.operator_ids) == set(
        reference_operator_ids()
    )


def assert_complete_reconstruction(
    state: RecursiveProbeState,
) -> None:
    if not state.is_fully_reconstructed:
        raise FootKneeRecursiveProbeValidationError(
            "graph still contains unresolved relations."
        )

    if not reconstruction_matches_reference_topology(state):
        raise FootKneeRecursiveProbeValidationError(
            "reconstructed topology does not match Scenario 01A."
        )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "AuthorizedGraphUpdate",
    "FootKneeRecursiveProbeValidationError",
    "RecursiveProbeState",
    "assert_complete_reconstruction",
    "authorize_relation_update",
    "build_recursive_probe_initial_state",
    "build_recursive_snapshot",
    "hidden_reference_operator",
    "reconstruction_matches_reference_topology",
    "reference_operator_ids",
]
