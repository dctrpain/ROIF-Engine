"""
ROIF Validation Suite
Clinical Scenario 01B:
Foot-Knee Incomplete Graph / Active Probe Validation

Purpose
-------
This validation case extends Clinical Scenario 01A.

Scenario 01A asks:

    Given a sufficiently known graph, can ROIF infer
    D_origin, D_fast, D_root and Node* correctly?

Scenario 01B asks:

    Given an intentionally incomplete version of the same graph,
    can the Active Probe Engine identify informative missing
    structure, reduce uncertainty, and recover enough evidence
    for the causal roles to converge toward the external labels
    defined by Scenario 01A?

Important
---------
The full graph is retained only as hidden validation ground truth.

It must never be supplied to:

- Probe Registry
- Probe Policy
- Probe Planner
- Active Probe Engine
- Cascade Solver operating on the incomplete graph

The hidden graph may only be used by the validation harness to
simulate observations and to evaluate reconstruction accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from roif.roif_entities import (
    InfluenceOperator,
    ROIFSystem,
)

from validation.clinical.foot_knee_case import (
    ClinicalValidationLabels,
    FootKneeValidationCase,
    build_foot_knee_case,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FootKneeProbeValidationError(ValueError):
    """Raised when Clinical Scenario 01B is internally inconsistent."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    return MappingProxyType(dict(value))


def _unit_interval(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = float(value)

    if not 0.0 <= numeric <= 1.0:
        raise FootKneeProbeValidationError(
            f"{field_name} must be in [0, 1]."
        )

    return numeric


def _operator_map(
    system: ROIFSystem,
) -> Mapping[str, InfluenceOperator]:
    return MappingProxyType(
        {
            operator.operator_id: operator
            for operator in system.operators
        }
    )


# ---------------------------------------------------------------------------
# Hidden structural truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HiddenEdgeTruth:
    """
    One edge intentionally hidden from the observable graph.

    This object belongs to the validation harness only.

    APE must not receive this object.
    """

    operator_id: str
    source_id: str
    target_id: str
    gain: float
    expected_present: bool = True

    def __post_init__(self) -> None:
        if not self.operator_id:
            raise FootKneeProbeValidationError(
                "operator_id must not be empty."
            )

        if not self.source_id:
            raise FootKneeProbeValidationError(
                "source_id must not be empty."
            )

        if not self.target_id:
            raise FootKneeProbeValidationError(
                "target_id must not be empty."
            )

        if self.source_id == self.target_id:
            raise FootKneeProbeValidationError(
                "hidden edge cannot be self-referential."
            )

        if self.gain < 0.0:
            raise FootKneeProbeValidationError(
                "gain must be non-negative."
            )


# ---------------------------------------------------------------------------
# Observable uncertainty specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UncertainRelation:
    """
    Relation visible to APE only as an uncertainty hypothesis.

    It deliberately does not reveal whether the edge truly exists.
    """

    relation_id: str
    source_id: str
    target_id: str
    uncertainty: float
    importance: float
    confidence: float
    hypothesis_count: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.relation_id:
            raise FootKneeProbeValidationError(
                "relation_id must not be empty."
            )

        if not self.source_id or not self.target_id:
            raise FootKneeProbeValidationError(
                "source_id and target_id must not be empty."
            )

        if self.source_id == self.target_id:
            raise FootKneeProbeValidationError(
                "uncertain relation cannot be self-referential."
            )

        object.__setattr__(
            self,
            "uncertainty",
            _unit_interval(
                self.uncertainty,
                field_name="uncertainty",
            ),
        )

        object.__setattr__(
            self,
            "importance",
            _unit_interval(
                self.importance,
                field_name="importance",
            ),
        )

        object.__setattr__(
            self,
            "confidence",
            _unit_interval(
                self.confidence,
                field_name="confidence",
            ),
        )

        if self.hypothesis_count < 2:
            raise FootKneeProbeValidationError(
                "hypothesis_count must be at least 2."
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


# ---------------------------------------------------------------------------
# Probe expectation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeValidationExpectation:
    """
    External validation expectation for one uncertain relation.

    This is an evaluator-side label.

    It must not be passed into Probe planning.
    """

    relation_id: str
    expected_source_id: str
    expected_target_id: str
    expected_edge_present: bool

    minimum_uncertainty_reduction: float

    should_be_high_priority: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_uncertainty_reduction",
            _unit_interval(
                self.minimum_uncertainty_reduction,
                field_name="minimum_uncertainty_reduction",
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


# ---------------------------------------------------------------------------
# Complete Clinical Scenario 01B contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FootKneeProbeValidationCase:
    """
    Complete contract for Clinical Scenario 01B.

    public_system
        Graph visible to APE and downstream inference.

    hidden_reference_system
        Full Scenario 01A graph. Evaluator only.

    uncertain_relations
        Structural questions available to APE.

    hidden_edges
        Answers retained by the validation harness.

    expected_roles
        Final external labels inherited from Scenario 01A.
    """

    case_id: str
    title: str

    public_system: ROIFSystem
    hidden_reference_system: ROIFSystem

    uncertain_relations: tuple[UncertainRelation, ...]
    hidden_edges: tuple[HiddenEdgeTruth, ...]

    probe_expectations: tuple[ProbeValidationExpectation, ...]

    expected_roles: ClinicalValidationLabels

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "uncertain_relations",
            tuple(self.uncertain_relations),
        )

        object.__setattr__(
            self,
            "hidden_edges",
            tuple(self.hidden_edges),
        )

        object.__setattr__(
            self,
            "probe_expectations",
            tuple(self.probe_expectations),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )

        uncertain_ids = {
            relation.relation_id
            for relation in self.uncertain_relations
        }

        hidden_ids = {
            edge.operator_id
            for edge in self.hidden_edges
        }

        expectation_ids = {
            expectation.relation_id
            for expectation in self.probe_expectations
        }

        if len(uncertain_ids) != len(self.uncertain_relations):
            raise FootKneeProbeValidationError(
                "uncertain relation IDs must be unique."
            )

        if len(hidden_ids) != len(self.hidden_edges):
            raise FootKneeProbeValidationError(
                "hidden operator IDs must be unique."
            )

        if expectation_ids != uncertain_ids:
            raise FootKneeProbeValidationError(
                "every uncertain relation must have exactly one "
                "validation expectation."
            )

    @property
    def public_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            operator.operator_id
            for operator in self.public_system.operators
        )

    @property
    def hidden_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            edge.operator_id
            for edge in self.hidden_edges
        )

    def hidden_edge(
        self,
        operator_id: str,
    ) -> HiddenEdgeTruth:
        for edge in self.hidden_edges:
            if edge.operator_id == operator_id:
                return edge

        raise KeyError(operator_id)

    def uncertain_relation(
        self,
        relation_id: str,
    ) -> UncertainRelation:
        for relation in self.uncertain_relations:
            if relation.relation_id == relation_id:
                return relation

        raise KeyError(relation_id)


# ---------------------------------------------------------------------------
# Hidden-edge selection
# ---------------------------------------------------------------------------


PRIMARY_HIDDEN_OPERATOR_IDS: tuple[str, ...] = (
    "fibrosis_to_compliance",
    "compliance_to_tibial_rotation",
)


def build_hidden_edge_truth(
    reference_case: FootKneeValidationCase | None = None,
) -> tuple[HiddenEdgeTruth, ...]:
    """
    Extract hidden truth from Clinical Scenario 01A.

    This function is evaluator-side only.
    """

    case = reference_case or build_foot_knee_case()

    operators = _operator_map(case.pathological_system)

    hidden: list[HiddenEdgeTruth] = []

    for operator_id in PRIMARY_HIDDEN_OPERATOR_IDS:
        try:
            operator = operators[operator_id]
        except KeyError as exc:
            raise FootKneeProbeValidationError(
                f"reference graph is missing required operator "
                f"{operator_id!r}."
            ) from exc

        if len(operator.source_ids) != 1:
            raise FootKneeProbeValidationError(
                f"{operator_id!r} must have exactly one source."
            )

        if len(operator.target_ids) != 1:
            raise FootKneeProbeValidationError(
                f"{operator_id!r} must have exactly one target."
            )

        hidden.append(
            HiddenEdgeTruth(
                operator_id=operator.operator_id,
                source_id=operator.source_ids[0],
                target_id=operator.target_ids[0],
                gain=float(operator.gain),
                expected_present=True,
            )
        )

    return tuple(hidden)


# ---------------------------------------------------------------------------
# Incomplete graph construction
# ---------------------------------------------------------------------------


def build_incomplete_pathological_system(
    reference_case: FootKneeValidationCase | None = None,
) -> ROIFSystem:
    """
    Return a copy of Scenario 01A with critical operators removed.

    Hidden edges:

        forefoot_fibrosis
            ?
        forefoot_compliance
            ?
        tibial_rotation_control

    Downstream knee topology remains observable.

    The function deliberately does not encode the missing edges
    anywhere in the returned ROIFSystem.
    """

    case = reference_case or build_foot_knee_case()

    source = case.pathological_system

    hidden_ids = frozenset(PRIMARY_HIDDEN_OPERATOR_IDS)

    visible_operators = tuple(
        operator
        for operator in source.operators
        if operator.operator_id not in hidden_ids
    )

    metadata = dict(source.metadata)

    metadata.update(
        {
            "validation_case": "foot_knee_probe",
            "validation_stage": "01B",
            "graph_completeness": "incomplete",
            "hidden_ground_truth_exposed": False,
        }
    )

    return ROIFSystem(
        system_id="foot_knee_probe_incomplete",
        name="Foot-knee incomplete graph for Active Probe validation",
        entities=source.entities,
        planes=source.planes,
        operators=visible_operators,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Observable uncertainty
# ---------------------------------------------------------------------------


def build_uncertain_relations() -> tuple[UncertainRelation, ...]:
    """
    Build the uncertainty view exposed to APE.

    Important:
    These objects describe questions, not answers.
    """

    return (
        UncertainRelation(
            relation_id="fibrosis_to_compliance",
            source_id="forefoot_fibrosis",
            target_id="forefoot_compliance",
            uncertainty=0.95,
            importance=1.00,
            confidence=0.05,
            hypothesis_count=2,
            metadata={
                "question": (
                    "Does forefoot fibrosis materially alter "
                    "forefoot compliance?"
                ),
                "domain_label": "mechanical",
                "ground_truth_included": False,
            },
        ),
        UncertainRelation(
            relation_id="compliance_to_tibial_rotation",
            source_id="forefoot_compliance",
            target_id="tibial_rotation_control",
            uncertainty=0.90,
            importance=0.95,
            confidence=0.10,
            hypothesis_count=2,
            metadata={
                "question": (
                    "Does altered forefoot compliance propagate "
                    "into tibial rotation control?"
                ),
                "domain_label": "mechanical",
                "ground_truth_included": False,
            },
        ),
    )


# ---------------------------------------------------------------------------
# External expectations
# ---------------------------------------------------------------------------


def build_probe_expectations() -> tuple[
    ProbeValidationExpectation,
    ...
]:
    """
    External evaluator labels.

    These labels must not be provided to ProbePlanner.
    """

    return (
        ProbeValidationExpectation(
            relation_id="fibrosis_to_compliance",
            expected_source_id="forefoot_fibrosis",
            expected_target_id="forefoot_compliance",
            expected_edge_present=True,
            minimum_uncertainty_reduction=0.50,
            should_be_high_priority=True,
            metadata={
                "validation_only": True,
                "hidden_from_ape": True,
            },
        ),
        ProbeValidationExpectation(
            relation_id="compliance_to_tibial_rotation",
            expected_source_id="forefoot_compliance",
            expected_target_id="tibial_rotation_control",
            expected_edge_present=True,
            minimum_uncertainty_reduction=0.40,
            should_be_high_priority=True,
            metadata={
                "validation_only": True,
                "hidden_from_ape": True,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Validation case factory
# ---------------------------------------------------------------------------


def build_foot_knee_probe_case() -> FootKneeProbeValidationCase:
    """
    Build Clinical Scenario 01B.

    Full graph:
        retained as hidden evaluator truth.

    Public graph:
        contains the same entities and state but omits the two
        upstream causal operators.

    Expected final roles:
        inherited unchanged from Scenario 01A.
    """

    reference_case = build_foot_knee_case()

    public_system = build_incomplete_pathological_system(
        reference_case
    )

    hidden_edges = build_hidden_edge_truth(reference_case)

    return FootKneeProbeValidationCase(
        case_id="clinical_01B_foot_knee_probe",
        title=(
            "Forefoot-to-knee incomplete graph "
            "Active Probe validation"
        ),
        public_system=public_system,
        hidden_reference_system=reference_case.pathological_system,
        uncertain_relations=build_uncertain_relations(),
        hidden_edges=hidden_edges,
        probe_expectations=build_probe_expectations(),
        expected_roles=reference_case.expected,
        metadata={
            "domain": "clinical_biomechanics",
            "validation_family": "clinical_01_foot_knee",
            "validation_stage": "01B",
            "parent_case": reference_case.case_id,
            "purpose": "active_probe_graph_reconstruction",
            "hidden_graph_used_by_ape": False,
            "hidden_labels_used_by_ape": False,
            "hidden_labels_used_by_solver": False,
            "automatic_graph_mutation": False,
            "requires_explicit_graph_update_authorization": True,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


# ---------------------------------------------------------------------------
# Ground-truth evaluator helpers
# ---------------------------------------------------------------------------


def evaluate_edge_hypothesis(
    case: FootKneeProbeValidationCase,
    *,
    source_id: str,
    target_id: str,
) -> bool:
    """
    Return the hidden truth for an edge hypothesis.

    VALIDATION HARNESS ONLY.

    This simulates an observation source for deterministic tests.
    It must never be called from ProbePlanner or ActiveProbeEngine.
    """

    for edge in case.hidden_edges:
        if (
            edge.source_id == source_id
            and edge.target_id == target_id
        ):
            return edge.expected_present

    return False


def reference_operator_for_relation(
    case: FootKneeProbeValidationCase,
    relation_id: str,
) -> InfluenceOperator:
    """
    Retrieve the hidden reference operator.

    VALIDATION HARNESS ONLY.
    """

    edge = case.hidden_edge(relation_id)

    operators = _operator_map(case.hidden_reference_system)

    try:
        return operators[edge.operator_id]
    except KeyError as exc:
        raise FootKneeProbeValidationError(
            f"hidden reference operator "
            f"{edge.operator_id!r} is unavailable."
        ) from exc


# ---------------------------------------------------------------------------
# Scientific validation invariants
# ---------------------------------------------------------------------------


def assert_hidden_truth_is_isolated(
    case: FootKneeProbeValidationCase,
) -> None:
    """
    Verify that hidden structural truth is absent from public graph.
    """

    public_ids = set(case.public_operator_ids)

    leaked = public_ids.intersection(case.hidden_operator_ids)

    if leaked:
        raise FootKneeProbeValidationError(
            "hidden operators leaked into public graph: "
            + ", ".join(sorted(leaked))
        )


def assert_reference_roles_preserved(
    case: FootKneeProbeValidationCase,
) -> None:
    """
    Verify that Scenario 01B inherits the external labels of 01A.
    """

    reference = build_foot_knee_case()

    if case.expected_roles != reference.expected:
        raise FootKneeProbeValidationError(
            "Scenario 01B role labels diverged from Scenario 01A."
        )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "FootKneeProbeValidationCase",
    "FootKneeProbeValidationError",
    "HiddenEdgeTruth",
    "ProbeValidationExpectation",
    "UncertainRelation",
    "PRIMARY_HIDDEN_OPERATOR_IDS",
    "assert_hidden_truth_is_isolated",
    "assert_reference_roles_preserved",
    "build_foot_knee_probe_case",
    "build_hidden_edge_truth",
    "build_incomplete_pathological_system",
    "build_probe_expectations",
    "build_uncertain_relations",
    "evaluate_edge_hypothesis",
    "reference_operator_for_relation",
]