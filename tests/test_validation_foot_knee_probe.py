"""
ROIF Validation Suite
Clinical Scenario 01B:
Foot-Knee Incomplete Graph / Active Probe Validation.

Scientific purpose
------------------
Scenario 01A validates causal-role inference when the graph is known.

Scenario 01B validates blind Active Probe exploration when critical
parts of the graph are deliberately hidden.

Validation layers
-----------------
Layer 1:
    Verify that the incomplete clinical graph is scientifically valid
    and that hidden ground truth cannot leak into APE inputs.

Layer 2:
    Convert the clinical uncertainty specification into the real
    ROIF Active Probe data model:

        UncertainRelation
            ->
        GraphTarget
            ->
        GraphUncertainty
            ->
        IncompleteGraphSnapshot

    and verify compatibility with:

        ProbeGraphTarget
        ProbeGraphLink
        ProbeInformationEstimate
        ProbeRegistry
        ProbePlanner
        ActiveProbeEngine

Important
---------
The complete graph and expected causal roles are evaluator-side truth.

They must never be supplied to ProbePlanner or ActiveProbeEngine.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.active_probe_engine import (
    ActiveProbeEngine,
)

from roif.probe_graph_adapter import (
    GraphTarget,
    GraphTargetKind,
    GraphUncertainty,
    IncompleteGraphSnapshot,
    ProbeGraphLink,
    ProbeGraphTarget,
)

from roif.probe_planner import (
    ProbeInformationEstimate,
    ProbePlanner,
)

from roif.probe_registry import (
    ProbeRegistry,
)

from validation.clinical.foot_knee_case import (
    build_foot_knee_case,
)

from validation.clinical.foot_knee_probe_case import (
    PRIMARY_HIDDEN_OPERATOR_IDS,
    FootKneeProbeValidationCase,
    assert_hidden_truth_is_isolated,
    assert_reference_roles_preserved,
    build_foot_knee_probe_case,
    build_hidden_edge_truth,
    build_incomplete_pathological_system,
    build_probe_expectations,
    build_uncertain_relations,
    evaluate_edge_hypothesis,
    reference_operator_for_relation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def case() -> FootKneeProbeValidationCase:
    return build_foot_knee_probe_case()


@pytest.fixture()
def reference_case():
    return build_foot_knee_case()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge_target_kind() -> GraphTargetKind:
    """
    Resolve the public graph-target kind by semantic value.

    This avoids coupling the validation case to Enum member spelling
    while still requiring the APE graph model to support edge targets.
    """

    for member in GraphTargetKind:
        if member.value == "edge":
            return member

    raise AssertionError(
        "GraphTargetKind must provide an 'edge' target kind."
    )


def _graph_target_for_relation(
    relation,
) -> GraphTarget:
    return GraphTarget(
        kind=_edge_target_kind(),
        identifier=relation.relation_id,
        source=relation.source_id,
        target=relation.target_id,
        metadata={
            "validation_case": "clinical_01B_foot_knee_probe",
            "ground_truth_included": False,
        },
    )


def _graph_uncertainty_for_relation(
    relation,
) -> GraphUncertainty:
    return GraphUncertainty(
        target=_graph_target_for_relation(relation),
        uncertainty=relation.uncertainty,
        importance=relation.importance,
        confidence=relation.confidence,
        hypothesis_count=relation.hypothesis_count,
        metadata={
            "validation_only": False,
            "ground_truth_included": False,
        },
    )


def _build_ape_snapshot(
    case: FootKneeProbeValidationCase,
) -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id=case.public_system.system_id,
        version="clinical-01B-v1",
        uncertainties=tuple(
            _graph_uncertainty_for_relation(relation)
            for relation in case.uncertain_relations
        ),
        metadata={
            "validation_case": case.case_id,
            "hidden_graph_exposed": False,
            "hidden_labels_exposed": False,
        },
    )


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_case_contract_is_complete(
    case: FootKneeProbeValidationCase,
) -> None:
    assert case.case_id == "clinical_01B_foot_knee_probe"

    assert case.public_system is not None
    assert case.hidden_reference_system is not None

    assert case.uncertain_relations
    assert case.hidden_edges
    assert case.probe_expectations

    assert case.expected_roles is not None


def test_case_metadata_declares_blind_validation(
    case: FootKneeProbeValidationCase,
) -> None:
    assert case.metadata["validation_stage"] == "01B"

    assert (
        case.metadata["purpose"]
        == "active_probe_graph_reconstruction"
    )

    assert case.metadata["hidden_graph_used_by_ape"] is False
    assert case.metadata["hidden_labels_used_by_ape"] is False
    assert case.metadata["hidden_labels_used_by_solver"] is False

    assert case.metadata["automatic_graph_mutation"] is False

    assert (
        case.metadata[
            "requires_explicit_graph_update_authorization"
        ]
        is True
    )


def test_case_metadata_is_read_only(
    case: FootKneeProbeValidationCase,
) -> None:
    assert isinstance(case.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        case.metadata["leak"] = True  # type: ignore[index]


# ---------------------------------------------------------------------------
# Relationship with Scenario 01A
# ---------------------------------------------------------------------------


def test_probe_case_uses_same_hidden_reference_as_01a(
    case: FootKneeProbeValidationCase,
    reference_case,
) -> None:
    assert (
        case.hidden_reference_system.channel_ids
        == reference_case.pathological_system.channel_ids
    )

    assert {
        operator.operator_id
        for operator in case.hidden_reference_system.operators
    } == {
        operator.operator_id
        for operator in reference_case.pathological_system.operators
    }


def test_expected_roles_are_inherited_from_01a(
    case: FootKneeProbeValidationCase,
    reference_case,
) -> None:
    assert case.expected_roles == reference_case.expected

    assert (
        case.expected_roles.d_origin
        == "forefoot_fibrosis"
    )

    assert (
        case.expected_roles.d_fast
        == "forefoot_compliance"
    )

    assert (
        case.expected_roles.d_root
        == "tibial_rotation_control"
    )

    assert (
        case.expected_roles.node_star
        == "forefoot_compliance"
    )


def test_reference_role_invariant(
    case: FootKneeProbeValidationCase,
) -> None:
    assert_reference_roles_preserved(case)


# ---------------------------------------------------------------------------
# Public incomplete graph
# ---------------------------------------------------------------------------


def test_public_graph_preserves_all_channels(
    case: FootKneeProbeValidationCase,
    reference_case,
) -> None:
    """
    Scenario 01B hides relations, not entities.

    Missing nodes would make graph reconstruction artificially easy.
    """

    assert (
        case.public_system.channel_ids
        == reference_case.pathological_system.channel_ids
    )


def test_public_graph_is_structurally_incomplete(
    case: FootKneeProbeValidationCase,
    reference_case,
) -> None:
    public_count = len(case.public_system.operators)

    reference_count = len(
        reference_case.pathological_system.operators
    )

    assert public_count < reference_count

    assert (
        reference_count - public_count
        == len(PRIMARY_HIDDEN_OPERATOR_IDS)
    )


def test_only_declared_upstream_edges_are_hidden(
    case: FootKneeProbeValidationCase,
    reference_case,
) -> None:
    public_ids = {
        operator.operator_id
        for operator in case.public_system.operators
    }

    reference_ids = {
        operator.operator_id
        for operator
        in reference_case.pathological_system.operators
    }

    missing = reference_ids - public_ids

    assert missing == set(PRIMARY_HIDDEN_OPERATOR_IDS)


def test_downstream_graph_remains_observable(
    case: FootKneeProbeValidationCase,
) -> None:
    assert set(case.public_operator_ids) == {
        "tibial_rotation_to_knee_tracking",
        "knee_tracking_to_flexion",
        "flexion_feedback_to_tibial_rotation",
    }


def test_hidden_truth_is_not_present_in_public_graph(
    case: FootKneeProbeValidationCase,
) -> None:
    assert_hidden_truth_is_isolated(case)

    assert not (
        set(case.public_operator_ids)
        & set(case.hidden_operator_ids)
    )


# ---------------------------------------------------------------------------
# Hidden ground truth
# ---------------------------------------------------------------------------


def test_hidden_edges_are_exactly_the_two_upstream_relations(
    case: FootKneeProbeValidationCase,
) -> None:
    assert case.hidden_operator_ids == (
        "fibrosis_to_compliance",
        "compliance_to_tibial_rotation",
    )


def test_hidden_edge_truth_matches_reference_graph(
    case: FootKneeProbeValidationCase,
) -> None:
    first = case.hidden_edge(
        "fibrosis_to_compliance"
    )

    assert first.source_id == "forefoot_fibrosis"
    assert first.target_id == "forefoot_compliance"
    assert first.expected_present is True
    assert first.gain == pytest.approx(0.95)

    second = case.hidden_edge(
        "compliance_to_tibial_rotation"
    )

    assert second.source_id == "forefoot_compliance"

    assert (
        second.target_id
        == "tibial_rotation_control"
    )

    assert second.expected_present is True
    assert second.gain == pytest.approx(0.90)


def test_hidden_truth_factory_is_deterministic() -> None:
    left = build_hidden_edge_truth()
    right = build_hidden_edge_truth()

    assert left == right


def test_reference_operator_lookup_matches_hidden_truth(
    case: FootKneeProbeValidationCase,
) -> None:
    for edge in case.hidden_edges:
        operator = reference_operator_for_relation(
            case,
            edge.operator_id,
        )

        assert operator.operator_id == edge.operator_id

        assert (
            operator.source_ids
            == (edge.source_id,)
        )

        assert (
            operator.target_ids
            == (edge.target_id,)
        )

        assert operator.gain == pytest.approx(
            edge.gain
        )


# ---------------------------------------------------------------------------
# APE-visible uncertainty
# ---------------------------------------------------------------------------


def test_uncertain_relations_match_hidden_questions(
    case: FootKneeProbeValidationCase,
) -> None:
    identifiers = {
        relation.relation_id
        for relation in case.uncertain_relations
    }

    assert identifiers == set(
        PRIMARY_HIDDEN_OPERATOR_IDS
    )


def test_uncertain_relations_do_not_encode_answer_flag(
    case: FootKneeProbeValidationCase,
) -> None:
    """
    APE-visible uncertainty describes questions, not answers.
    """

    for relation in case.uncertain_relations:
        assert not hasattr(
            relation,
            "expected_present",
        )

        assert not hasattr(
            relation,
            "gain",
        )

        assert (
            relation.metadata[
                "ground_truth_included"
            ]
            is False
        )


def test_uncertainty_is_high_before_probe(
    case: FootKneeProbeValidationCase,
) -> None:
    by_id = {
        relation.relation_id: relation
        for relation in case.uncertain_relations
    }

    assert (
        by_id[
            "fibrosis_to_compliance"
        ].uncertainty
        == pytest.approx(0.95)
    )

    assert (
        by_id[
            "compliance_to_tibial_rotation"
        ].uncertainty
        == pytest.approx(0.90)
    )


def test_confidence_is_low_before_probe(
    case: FootKneeProbeValidationCase,
) -> None:
    for relation in case.uncertain_relations:
        assert relation.confidence <= 0.10
        assert relation.uncertainty >= 0.90


def test_uncertain_relation_metadata_is_read_only(
    case: FootKneeProbeValidationCase,
) -> None:
    relation = case.uncertain_relations[0]

    assert isinstance(
        relation.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        relation.metadata["answer"] = True  # type: ignore[index]


# ---------------------------------------------------------------------------
# External Probe expectations
# ---------------------------------------------------------------------------


def test_every_uncertain_relation_has_external_expectation(
    case: FootKneeProbeValidationCase,
) -> None:
    uncertain_ids = {
        relation.relation_id
        for relation in case.uncertain_relations
    }

    expectation_ids = {
        expectation.relation_id
        for expectation in case.probe_expectations
    }

    assert expectation_ids == uncertain_ids


def test_probe_expectations_are_validation_only(
    case: FootKneeProbeValidationCase,
) -> None:
    for expectation in case.probe_expectations:
        assert (
            expectation.metadata[
                "validation_only"
            ]
            is True
        )

        assert (
            expectation.metadata[
                "hidden_from_ape"
            ]
            is True
        )


def test_probe_expectations_require_meaningful_uncertainty_reduction(
    case: FootKneeProbeValidationCase,
) -> None:
    for expectation in case.probe_expectations:
        assert (
            expectation.minimum_uncertainty_reduction
            > 0.0
        )


# ---------------------------------------------------------------------------
# Validation-harness observation oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "source_id",
        "target_id",
    ),
    (
        (
            "forefoot_fibrosis",
            "forefoot_compliance",
        ),
        (
            "forefoot_compliance",
            "tibial_rotation_control",
        ),
    ),
)
def test_hidden_oracle_confirms_true_hidden_edges(
    case: FootKneeProbeValidationCase,
    source_id: str,
    target_id: str,
) -> None:
    assert (
        evaluate_edge_hypothesis(
            case,
            source_id=source_id,
            target_id=target_id,
        )
        is True
    )


@pytest.mark.parametrize(
    (
        "source_id",
        "target_id",
    ),
    (
        (
            "forefoot_fibrosis",
            "tibial_rotation_control",
        ),
        (
            "forefoot_fibrosis",
            "knee_tracking",
        ),
        (
            "knee_tracking",
            "forefoot_compliance",
        ),
    ),
)
def test_hidden_oracle_rejects_false_edge_hypotheses(
    case: FootKneeProbeValidationCase,
    source_id: str,
    target_id: str,
) -> None:
    assert (
        evaluate_edge_hypothesis(
            case,
            source_id=source_id,
            target_id=target_id,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_hidden_edge_truth_is_immutable(
    case: FootKneeProbeValidationCase,
) -> None:
    edge = case.hidden_edges[0]

    with pytest.raises(FrozenInstanceError):
        edge.gain = 0.0  # type: ignore[misc]


def test_uncertain_relation_is_immutable(
    case: FootKneeProbeValidationCase,
) -> None:
    relation = case.uncertain_relations[0]

    with pytest.raises(FrozenInstanceError):
        relation.uncertainty = 0.0  # type: ignore[misc]


def test_probe_expectation_is_immutable(
    case: FootKneeProbeValidationCase,
) -> None:
    expectation = case.probe_expectations[0]

    with pytest.raises(FrozenInstanceError):
        expectation.expected_edge_present = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Factory determinism
# ---------------------------------------------------------------------------


def test_incomplete_system_factory_is_deterministic() -> None:
    left = build_incomplete_pathological_system()
    right = build_incomplete_pathological_system()

    assert left.channel_ids == right.channel_ids

    assert tuple(
        operator.operator_id
        for operator in left.operators
    ) == tuple(
        operator.operator_id
        for operator in right.operators
    )


def test_uncertainty_factory_is_deterministic() -> None:
    assert (
        build_uncertain_relations()
        == build_uncertain_relations()
    )


def test_expectation_factory_is_deterministic() -> None:
    assert (
        build_probe_expectations()
        == build_probe_expectations()
    )


def test_complete_probe_case_is_deterministic() -> None:
    left = build_foot_knee_probe_case()
    right = build_foot_knee_probe_case()

    assert left.case_id == right.case_id

    assert (
        left.public_operator_ids
        == right.public_operator_ids
    )

    assert left.hidden_edges == right.hidden_edges

    assert (
        left.uncertain_relations
        == right.uncertain_relations
    )

    assert (
        left.probe_expectations
        == right.probe_expectations
    )

    assert (
        left.expected_roles
        == right.expected_roles
    )


# ---------------------------------------------------------------------------
# Scientific separation invariants
# ---------------------------------------------------------------------------


def test_ground_truth_and_public_graph_are_distinct_objects(
    case: FootKneeProbeValidationCase,
) -> None:
    assert (
        case.public_system
        is not case.hidden_reference_system
    )


def test_public_graph_does_not_claim_completeness(
    case: FootKneeProbeValidationCase,
) -> None:
    assert (
        case.public_system.metadata[
            "graph_completeness"
        ]
        == "incomplete"
    )

    assert (
        case.public_system.metadata[
            "hidden_ground_truth_exposed"
        ]
        is False
    )


def test_validation_has_no_diagnostic_or_treatment_claim(
    case: FootKneeProbeValidationCase,
) -> None:
    assert case.metadata["diagnostic_claim"] is False
    assert case.metadata["treatment_claim"] is False


# ---------------------------------------------------------------------------
# APE module availability
# ---------------------------------------------------------------------------


def test_active_probe_stack_is_importable() -> None:
    import roif.active_probe_engine as active_probe_engine
    import roif.probe_entities as probe_entities
    import roif.probe_graph_adapter as probe_graph_adapter
    import roif.probe_planner as probe_planner
    import roif.probe_policy as probe_policy
    import roif.probe_registry as probe_registry

    assert active_probe_engine is not None
    assert probe_entities is not None
    assert probe_graph_adapter is not None
    assert probe_planner is not None
    assert probe_policy is not None
    assert probe_registry is not None


# ===========================================================================
# REAL APE DATA-MODEL INTEGRATION
# ===========================================================================


def test_graph_target_kind_supports_edges() -> None:
    assert _edge_target_kind().value == "edge"


def test_clinical_relation_maps_to_real_graph_target(
    case: FootKneeProbeValidationCase,
) -> None:
    relation = case.uncertain_relation(
        "fibrosis_to_compliance"
    )

    target = _graph_target_for_relation(
        relation
    )

    assert (
        target.identifier
        == "fibrosis_to_compliance"
    )

    assert (
        target.source
        == "forefoot_fibrosis"
    )

    assert (
        target.target
        == "forefoot_compliance"
    )

    assert target.kind.value == "edge"

    assert (
        target.metadata[
            "ground_truth_included"
        ]
        is False
    )


def test_second_clinical_relation_maps_to_real_graph_target(
    case: FootKneeProbeValidationCase,
) -> None:
    relation = case.uncertain_relation(
        "compliance_to_tibial_rotation"
    )

    target = _graph_target_for_relation(
        relation
    )

    assert (
        target.source
        == "forefoot_compliance"
    )

    assert (
        target.target
        == "tibial_rotation_control"
    )


def test_clinical_uncertainty_maps_to_real_graph_uncertainty(
    case: FootKneeProbeValidationCase,
) -> None:
    relation = case.uncertain_relation(
        "fibrosis_to_compliance"
    )

    uncertainty = _graph_uncertainty_for_relation(
        relation
    )

    assert (
        uncertainty.target.identifier
        == relation.relation_id
    )

    assert (
        uncertainty.uncertainty
        == pytest.approx(
            relation.uncertainty
        )
    )

    assert (
        uncertainty.importance
        == pytest.approx(
            relation.importance
        )
    )

    assert (
        uncertainty.confidence
        == pytest.approx(
            relation.confidence
        )
    )

    assert (
        uncertainty.hypothesis_count
        == relation.hypothesis_count
    )


def test_real_ape_snapshot_contains_both_uncertain_edges(
    case: FootKneeProbeValidationCase,
) -> None:
    snapshot = _build_ape_snapshot(case)

    assert (
        snapshot.graph_id
        == case.public_system.system_id
    )

    assert snapshot.version == "clinical-01B-v1"

    assert len(snapshot.uncertainties) == 2

    identifiers = {
        uncertainty.target.identifier
        for uncertainty in snapshot.uncertainties
    }

    assert identifiers == {
        "fibrosis_to_compliance",
        "compliance_to_tibial_rotation",
    }


def test_real_ape_snapshot_contains_no_hidden_answer(
    case: FootKneeProbeValidationCase,
) -> None:
    snapshot = _build_ape_snapshot(case)

    assert (
        snapshot.metadata[
            "hidden_graph_exposed"
        ]
        is False
    )

    assert (
        snapshot.metadata[
            "hidden_labels_exposed"
        ]
        is False
    )

    for uncertainty in snapshot.uncertainties:
        assert (
            uncertainty.metadata[
                "ground_truth_included"
            ]
            is False
        )

        assert not hasattr(
            uncertainty,
            "expected_present",
        )


def test_real_ape_snapshot_does_not_expose_hidden_gain(
    case: FootKneeProbeValidationCase,
) -> None:
    snapshot = _build_ape_snapshot(case)

    hidden_gains = {
        edge.operator_id: edge.gain
        for edge in case.hidden_edges
    }

    for uncertainty in snapshot.uncertainties:
        assert not hasattr(
            uncertainty,
            "gain",
        )

        assert (
            uncertainty.target.identifier
            in hidden_gains
        )


def test_real_ape_snapshot_is_deterministic(
    case: FootKneeProbeValidationCase,
) -> None:
    left = _build_ape_snapshot(case)
    right = _build_ape_snapshot(case)

    assert left.graph_id == right.graph_id
    assert left.version == right.version

    assert (
        left.uncertainties
        == right.uncertainties
    )


# ---------------------------------------------------------------------------
# Probe graph-link compatibility
# ---------------------------------------------------------------------------


def test_real_probe_graph_target_can_reference_first_unknown_edge() -> None:
    target = ProbeGraphTarget(
        target_identifier="fibrosis_to_compliance",
        sensitivity=1.0,
        discrimination=1.0,
        expected_uncertainty_reduction=0.80,
        metadata={
            "validation_case": "clinical_01B_foot_knee_probe",
        },
    )

    assert (
        target.target_identifier
        == "fibrosis_to_compliance"
    )

    assert target.sensitivity == pytest.approx(1.0)

    assert (
        target.expected_uncertainty_reduction
        == pytest.approx(0.80)
    )


def test_real_probe_graph_target_can_reference_second_unknown_edge() -> None:
    target = ProbeGraphTarget(
        target_identifier="compliance_to_tibial_rotation",
        sensitivity=0.95,
        discrimination=1.0,
        expected_uncertainty_reduction=0.75,
    )

    assert (
        target.target_identifier
        == "compliance_to_tibial_rotation"
    )

    assert target.sensitivity == pytest.approx(0.95)


def test_real_probe_graph_link_can_target_clinical_uncertainty() -> None:
    graph_target = ProbeGraphTarget(
        target_identifier="fibrosis_to_compliance",
        sensitivity=1.0,
        discrimination=1.0,
        expected_uncertainty_reduction=0.80,
    )

    link = ProbeGraphLink(
        probe_identifier="probe_forefoot_compliance_response",
        targets=(graph_target,),
        novelty=0.90,
        feasibility=0.95,
        estimate_confidence=0.90,
        metadata={
            "validation_case": "clinical_01B_foot_knee_probe",
            "ground_truth_included": False,
        },
    )

    assert (
        link.probe_identifier
        == "probe_forefoot_compliance_response"
    )

    assert len(link.targets) == 1

    assert (
        link.targets[0].target_identifier
        == "fibrosis_to_compliance"
    )

    assert (
        link.metadata[
            "ground_truth_included"
        ]
        is False
    )


# ---------------------------------------------------------------------------
# Real ProbeInformationEstimate compatibility
# ---------------------------------------------------------------------------


def test_real_information_estimate_can_represent_first_probe() -> None:
    estimate = ProbeInformationEstimate(
        probe_identifier="probe_forefoot_compliance_response",
        information_gain=0.90,
        uncertainty_reduction=0.80,
        hypothesis_discrimination=0.90,
        graph_coverage=0.50,
        novelty=0.90,
        feasibility=0.95,
        confidence=0.90,
        redundancy=0.05,
        disruption=0.10,
        metadata={
            "validation_case": "clinical_01B_foot_knee_probe",
            "ground_truth_included": False,
        },
    )

    assert (
        estimate.probe_identifier
        == "probe_forefoot_compliance_response"
    )

    assert estimate.information_gain == pytest.approx(0.90)

    assert (
        estimate.uncertainty_reduction
        == pytest.approx(0.80)
    )

    assert (
        estimate.metadata[
            "ground_truth_included"
        ]
        is False
    )


def test_real_information_estimate_can_represent_second_probe() -> None:
    estimate = ProbeInformationEstimate(
        probe_identifier="probe_tibial_rotation_response",
        information_gain=0.82,
        uncertainty_reduction=0.75,
        hypothesis_discrimination=0.88,
        graph_coverage=0.45,
        novelty=0.85,
        feasibility=0.95,
        confidence=0.88,
        redundancy=0.05,
        disruption=0.10,
    )

    assert (
        estimate.probe_identifier
        == "probe_tibial_rotation_response"
    )

    assert estimate.information_gain > 0.0
    assert estimate.uncertainty_reduction > 0.0


def test_information_estimates_do_not_contain_hidden_edge_truth() -> None:
    estimates = (
        ProbeInformationEstimate(
            probe_identifier="probe_forefoot_compliance_response",
            information_gain=0.90,
            uncertainty_reduction=0.80,
            metadata={
                "ground_truth_included": False,
            },
        ),
        ProbeInformationEstimate(
            probe_identifier="probe_tibial_rotation_response",
            information_gain=0.82,
            uncertainty_reduction=0.75,
            metadata={
                "ground_truth_included": False,
            },
        ),
    )

    for estimate in estimates:
        assert not hasattr(
            estimate,
            "expected_present",
        )

        assert not hasattr(
            estimate,
            "hidden_gain",
        )

        assert (
            estimate.metadata[
                "ground_truth_included"
            ]
            is False
        )


# ---------------------------------------------------------------------------
# Real APE object construction
# ---------------------------------------------------------------------------


def test_real_probe_registry_can_be_created_empty() -> None:
    registry = ProbeRegistry()

    assert registry is not None


def test_real_probe_planner_accepts_real_registry() -> None:
    registry = ProbeRegistry()

    planner = ProbePlanner(
        registry,
    )

    assert planner is not None


def test_real_active_probe_engine_accepts_real_registry() -> None:
    registry = ProbeRegistry()

    engine = ActiveProbeEngine(
        registry,
    )

    assert engine is not None


def test_planner_and_engine_can_share_same_registry() -> None:
    registry = ProbeRegistry()

    planner = ProbePlanner(
        registry,
    )

    engine = ActiveProbeEngine(
        registry,
    )

    assert planner is not None
    assert engine is not None


# ---------------------------------------------------------------------------
# Scientific boundary before the blind execution trial
# ---------------------------------------------------------------------------


def test_01b_is_ready_for_blind_probe_selection(
    case: FootKneeProbeValidationCase,
) -> None:
    """
    This is the boundary condition for the next validation layer.

    At this point:

    - the complete reference graph exists;
    - two critical edges are hidden;
    - APE receives only graph uncertainty;
    - APE-compatible graph objects can be constructed;
    - the Probe Registry / Planner / Engine can be instantiated;
    - hidden expected answers remain evaluator-side.

    No claim is made here that Probe* has already been selected.
    """

    snapshot = _build_ape_snapshot(case)

    registry = ProbeRegistry()
    planner = ProbePlanner(registry)
    engine = ActiveProbeEngine(registry)

    assert len(snapshot.uncertainties) == 2

    assert (
        set(case.public_operator_ids)
        .isdisjoint(
            case.hidden_operator_ids
        )
    )

    assert registry is not None
    assert planner is not None
    assert engine is not None

    assert (
        case.metadata[
            "hidden_graph_used_by_ape"
        ]
        is False
    )

    assert (
        case.metadata[
            "hidden_labels_used_by_ape"
        ]
        is False
    )
# ===========================================================================
# BLIND ACTIVE PROBE SELECTION TRIAL
# ===========================================================================

from roif.probe_entities import (
    GraphUpdate,
    Observation,
    ObservationChannel,
    Perturbation,
    PerturbationType,
    ProbeDefinition,
    ProbeMethod,
    ProbePhase,
    ProbePurpose,
    ProbeRegime,
)
from roif.probe_graph_adapter import ProbeGraphAdapter
from roif.probe_policy import ProbeAssessment, ProbePolicyContext


FIRST_PROBE_ID = "probe_forefoot_compliance_response"
SECOND_PROBE_ID = "probe_tibial_rotation_response"


def _candidate_identifier(candidate) -> str:
    """Return a public Probe identifier from a planner candidate."""
    definition = getattr(candidate, "definition", None)
    if definition is not None:
        identifier = getattr(definition, "identifier", None)
        if identifier is not None:
            return identifier

    for name in ("probe_identifier", "identifier"):
        identifier = getattr(candidate, name, None)
        if identifier is not None:
            return identifier

    raise AssertionError("Planner candidate exposes no Probe identifier.")


def _build_probe_definitions() -> tuple[ProbeDefinition, ProbeDefinition]:
    """Build two safe synthetic candidate Probes without hidden answers."""
    return (
        ProbeDefinition(
            identifier=FIRST_PROBE_ID,
            name="Forefoot compliance response probe",
            method=ProbeMethod.SIMULATION,
            regime=ProbeRegime.DYNAMIC,
            purpose=ProbePurpose.REDUCE_UNCERTAINTY,
            perturbation=Perturbation(
                kind=PerturbationType.SIMULATED_INTERVENTION,
                magnitude=1.0,
                magnitude_units="normalized",
                duration_seconds=1.0,
                parameters={"validation": True},
            ),
            description=(
                "Synthetic reversible probe of the public uncertainty "
                "between forefoot fibrosis and forefoot compliance."
            ),
            tags=("validation", "clinical_01B", "synthetic"),
        ),
        ProbeDefinition(
            identifier=SECOND_PROBE_ID,
            name="Tibial rotation response probe",
            method=ProbeMethod.SIMULATION,
            regime=ProbeRegime.DYNAMIC,
            purpose=ProbePurpose.REDUCE_UNCERTAINTY,
            perturbation=Perturbation(
                kind=PerturbationType.SIMULATED_INTERVENTION,
                magnitude=1.0,
                magnitude_units="normalized",
                duration_seconds=1.0,
                parameters={"validation": True},
            ),
            description=(
                "Synthetic reversible probe of the public uncertainty "
                "between forefoot compliance and tibial rotation control."
            ),
            tags=("validation", "clinical_01B", "synthetic"),
        ),
    )


def _build_probe_links() -> tuple[ProbeGraphLink, ProbeGraphLink]:
    """Connect candidate Probes only to APE-visible uncertain relations."""
    return (
        ProbeGraphLink(
            probe_identifier=FIRST_PROBE_ID,
            targets=(
                ProbeGraphTarget(
                    target_identifier="fibrosis_to_compliance",
                    sensitivity=1.0,
                    discrimination=1.0,
                    expected_uncertainty_reduction=0.80,
                    metadata={"ground_truth_included": False},
                ),
            ),
            novelty=0.90,
            feasibility=0.95,
            estimate_confidence=0.90,
            metadata={"ground_truth_included": False},
        ),
        ProbeGraphLink(
            probe_identifier=SECOND_PROBE_ID,
            targets=(
                ProbeGraphTarget(
                    target_identifier="compliance_to_tibial_rotation",
                    sensitivity=0.85,
                    discrimination=0.90,
                    expected_uncertainty_reduction=0.55,
                    metadata={"ground_truth_included": False},
                ),
            ),
            novelty=0.50,
            feasibility=0.95,
            estimate_confidence=0.85,
            metadata={"ground_truth_included": False},
        ),
    )


def _build_blind_registry() -> ProbeRegistry:
    return ProbeRegistry(_build_probe_definitions())


def _build_blind_adapter() -> ProbeGraphAdapter:
    return ProbeGraphAdapter(_build_probe_links())


def _safe_probe_assessments() -> dict[str, ProbeAssessment]:
    common = dict(
        expected_cost=0.05,
        expected_duration_seconds=1.0,
        cascade_risk=0.0,
        uncertainty=0.05,
        affects_external_systems=False,
        large_scale_effect_possible=False,
        uncontrolled_propagation_possible=False,
        requires_human_authorization=False,
        metadata={"validation": True},
    )
    return {
        FIRST_PROBE_ID: ProbeAssessment(**common),
        SECOND_PROBE_ID: ProbeAssessment(**common),
    }


def _safe_probe_context() -> ProbePolicyContext:
    return ProbePolicyContext(
        human_authorized=False,
        maximum_cost=1.0,
        maximum_duration_seconds=60.0,
        maximum_cascade_risk=0.25,
        maximum_uncertainty=1.0,
        non_fonit_gate_enabled=True,
        metadata={"validation_case": "clinical_01B_foot_knee_probe"},
    )


def _estimate_blind_probes(case: FootKneeProbeValidationCase):
    registry = _build_blind_registry()
    adapter = _build_blind_adapter()
    snapshot = _build_ape_snapshot(case)
    estimates = adapter.estimate_all(registry, snapshot)
    return registry, adapter, snapshot, estimates


def _plan_blind_probe(case: FootKneeProbeValidationCase):
    registry, adapter, snapshot, estimates = _estimate_blind_probes(case)
    engine = ActiveProbeEngine(registry)
    plan = engine.plan(
        estimates,
        assessments=_safe_probe_assessments(),
        context=_safe_probe_context(),
    )
    return registry, adapter, snapshot, estimates, engine, plan


def _relation_id_for_probe(probe_identifier: str) -> str:
    return {
        FIRST_PROBE_ID: "fibrosis_to_compliance",
        SECOND_PROBE_ID: "compliance_to_tibial_rotation",
    }[probe_identifier]


def _observation_pair_from_hidden_oracle(
    case: FootKneeProbeValidationCase,
    probe_identifier: str,
) -> tuple[Observation, Observation]:
    """
    Query hidden truth only after Probe selection and convert it to evidence.
    """
    relation_id = _relation_id_for_probe(probe_identifier)
    relation = case.uncertain_relation(relation_id)
    edge_present = evaluate_edge_hypothesis(
        case,
        source_id=relation.source_id,
        target_id=relation.target_id,
    )

    baseline = Observation(
        phase=ProbePhase.BASELINE,
        channel=ObservationChannel.SIGNAL,
        target=relation_id,
        value=0.0,
        units="normalized",
        confidence=0.95,
        metadata={
            "validation_case": case.case_id,
            "hidden_truth_exposed_to_planner": False,
        },
    )
    reassessment = Observation(
        phase=ProbePhase.REASSESSMENT,
        channel=ObservationChannel.SIGNAL,
        target=relation_id,
        value=1.0 if edge_present else 0.0,
        units="normalized",
        confidence=0.95,
        metadata={
            "validation_case": case.case_id,
            "synthetic_oracle_observation": True,
            "hidden_truth_exposed_to_planner": False,
        },
    )
    return baseline, reassessment


def _complete_selected_probe(
    case: FootKneeProbeValidationCase,
):
    registry, adapter, graph_snapshot, estimates, engine, plan = (
        _plan_blind_probe(case)
    )
    assert plan.selected is not None
    selected_id = _candidate_identifier(plan.selected)
    relation = case.uncertain_relation(_relation_id_for_probe(selected_id))

    run = engine.start_selected(
        plan,
        metadata={"validation_case": case.case_id},
    )
    baseline, reassessment = _observation_pair_from_hidden_oracle(
        case,
        selected_id,
    )
    engine.add_observation(baseline)
    engine.add_observation(reassessment)

    result = engine.complete(
        graph_update=GraphUpdate(
            affected_nodes=(relation.source_id, relation.target_id),
            affected_edges=((relation.source_id, relation.target_id),),
            confidence=0.95,
            metadata={
                "validation_case": case.case_id,
                "proposal_only": True,
                "automatic_mutation": False,
            },
        ),
        notes="Clinical Scenario 01B synthetic blind observation.",
        metadata={"validation_case": case.case_id},
    )
    return (
        registry,
        adapter,
        graph_snapshot,
        estimates,
        engine,
        plan,
        run,
        result,
        relation,
    )


# ---------------------------------------------------------------------------
# Blind candidate construction and estimation
# ---------------------------------------------------------------------------


def test_blind_trial_defines_two_candidate_probes() -> None:
    definitions = _build_probe_definitions()
    assert {item.identifier for item in definitions} == {
        FIRST_PROBE_ID,
        SECOND_PROBE_ID,
    }


def test_blind_probe_definitions_contain_no_hidden_answers() -> None:
    for definition in _build_probe_definitions():
        text = repr(
            (
                definition.identifier,
                definition.name,
                definition.description,
                definition.tags,
            )
        ).lower()
        for forbidden in (
            "expected_present",
            "hidden_gain",
            "d_origin",
            "d_fast",
            "d_root",
            "node_star",
        ):
            assert forbidden not in text


def test_blind_probes_use_safe_simulation_method() -> None:
    for definition in _build_probe_definitions():
        assert definition.method is ProbeMethod.SIMULATION
        assert definition.purpose is ProbePurpose.REDUCE_UNCERTAINTY
        assert (
            definition.perturbation.kind
            is PerturbationType.SIMULATED_INTERVENTION
        )


def test_adapter_estimates_both_blind_candidates(
    case: FootKneeProbeValidationCase,
) -> None:
    *_, estimates = _estimate_blind_probes(case)
    assert {item.probe_identifier for item in estimates} == {
        FIRST_PROBE_ID,
        SECOND_PROBE_ID,
    }
    assert all(item.information_gain > 0.0 for item in estimates)
    assert all(item.uncertainty_reduction > 0.0 for item in estimates)


def test_adapter_estimates_are_deterministic(
    case: FootKneeProbeValidationCase,
) -> None:
    *_, left = _estimate_blind_probes(case)
    *_, right = _estimate_blind_probes(case)
    assert left == right


# ---------------------------------------------------------------------------
# Real blind Probe selection
# ---------------------------------------------------------------------------


def test_engine_produces_blind_probe_plan(
    case: FootKneeProbeValidationCase,
) -> None:
    *_, plan = _plan_blind_probe(case)
    assert plan.selected is not None
    assert plan.ranked_candidates


def test_blind_plan_contains_both_candidates(
    case: FootKneeProbeValidationCase,
) -> None:
    *_, plan = _plan_blind_probe(case)
    identifiers = {
        _candidate_identifier(candidate)
        for candidate in plan.ranked_candidates
    }
    assert identifiers == {FIRST_PROBE_ID, SECOND_PROBE_ID}


def test_blind_planner_selects_admissible_probe_star(
    case: FootKneeProbeValidationCase,
) -> None:
    """
    Core Probe* assertion before the hidden oracle is queried.

    The validation must not prescribe which candidate wins.
    It verifies that Planner independently selects one of the public,
    admissible candidates and that the selected candidate is ranked first.
    """
    *_, plan = _plan_blind_probe(case)

    assert plan.selected is not None

    selected_id = _candidate_identifier(plan.selected)

    assert selected_id in {
        FIRST_PROBE_ID,
        SECOND_PROBE_ID,
    }

    assert plan.ranked_candidates

    assert (
        _candidate_identifier(plan.ranked_candidates[0])
        == selected_id
    )


def test_blind_selection_is_deterministic(
    case: FootKneeProbeValidationCase,
) -> None:
    *_, left = _plan_blind_probe(case)
    *_, right = _plan_blind_probe(case)
    assert left.selected is not None
    assert right.selected is not None
    assert _candidate_identifier(left.selected) == _candidate_identifier(
        right.selected
    )


def test_selected_probe_targets_public_uncertainty(
    case: FootKneeProbeValidationCase,
) -> None:
    _, adapter, _, _, _, plan = _plan_blind_probe(case)
    assert plan.selected is not None
    selected_id = _candidate_identifier(plan.selected)
    link = adapter.get_link(selected_id)
    public_ids = {item.relation_id for item in case.uncertain_relations}
    assert link.targets
    assert all(
        target.target_identifier in public_ids
        for target in link.targets
    )


# ---------------------------------------------------------------------------
# Real Probe lifecycle and GraphUpdateProposal
# ---------------------------------------------------------------------------


def test_hidden_oracle_is_used_only_after_probe_selection(
    case: FootKneeProbeValidationCase,
) -> None:
    *_, plan = _plan_blind_probe(case)
    assert plan.selected is not None
    selected_id = _candidate_identifier(plan.selected)
    baseline, reassessment = _observation_pair_from_hidden_oracle(
        case,
        selected_id,
    )
    assert baseline.phase is ProbePhase.BASELINE
    assert reassessment.phase is ProbePhase.REASSESSMENT
    assert baseline.target == reassessment.target


def test_selected_probe_enters_real_execution_lifecycle(
    case: FootKneeProbeValidationCase,
) -> None:
    (
        _,
        _,
        _,
        _,
        _,
        plan,
        run,
        _,
        _,
    ) = _complete_selected_probe(case)

    assert plan.selected is not None

    assert (
        run.definition.identifier
        == _candidate_identifier(plan.selected)
    )


def test_completed_blind_probe_returns_result(
    case: FootKneeProbeValidationCase,
) -> None:
    *_, result, _ = _complete_selected_probe(case)
    assert result is not None


def test_completed_probe_generates_graph_update_proposal(
    case: FootKneeProbeValidationCase,
) -> None:
    (
        _,
        adapter,
        graph_snapshot,
        _,
        _,
        plan,
        _,
        result,
        relation,
    ) = _complete_selected_probe(case)
    assert plan.selected is not None
    selected_id = _candidate_identifier(plan.selected)

    proposal = adapter.propose_graph_update(
        result,
        graph_snapshot,
        metadata={
            "validation_case": case.case_id,
            "authorization_required": True,
        },
    )
    assert proposal.graph_id == graph_snapshot.graph_id
    assert proposal.base_version == graph_snapshot.version
    assert proposal.probe_identifier == selected_id
    assert (
        relation.source_id,
        relation.target_id,
    ) in proposal.graph_update.affected_edges


def test_graph_update_proposal_does_not_mutate_public_graph(
    case: FootKneeProbeValidationCase,
) -> None:
    original_operator_ids = tuple(case.public_operator_ids)
    (
        _,
        adapter,
        graph_snapshot,
        _,
        _,
        _,
        _,
        result,
        _,
    ) = _complete_selected_probe(case)

    proposal = adapter.propose_graph_update(result, graph_snapshot)
    assert proposal is not None
    assert case.public_operator_ids == original_operator_ids
    assert "fibrosis_to_compliance" not in case.public_operator_ids
    assert "compliance_to_tibial_rotation" not in case.public_operator_ids


def test_clinical_01b_blind_probe_trial_end_to_end(
    case: FootKneeProbeValidationCase,
) -> None:
    """
    Complete controlled Scenario 01B blind Active Probe trial.

    Hidden truth is absent during estimation and planning. It is queried
    only after Probe* is selected, to simulate the resulting observation.
    """
    (
        _,
        adapter,
        graph_snapshot,
        estimates,
        _,
        plan,
        run,
        result,
        relation,
    ) = _complete_selected_probe(case)

    assert len(estimates) == 2
    assert plan.selected is not None

    selected_id = _candidate_identifier(plan.selected)

    assert selected_id in {
        FIRST_PROBE_ID,
        SECOND_PROBE_ID,
    }

    assert plan.ranked_candidates

    assert (
        _candidate_identifier(plan.ranked_candidates[0])
        == selected_id
    )

    assert graph_snapshot.metadata["hidden_graph_exposed"] is False
    assert graph_snapshot.metadata["hidden_labels_exposed"] is False
    assert run.definition.identifier == selected_id

    proposal = adapter.propose_graph_update(
        result,
        graph_snapshot,
        metadata={
            "validation_case": case.case_id,
            "authorization_required": True,
        },
    )
    assert proposal.probe_identifier == selected_id
    assert proposal.graph_id == case.public_system.system_id
    assert (
        relation.source_id,
        relation.target_id,
    ) in proposal.graph_update.affected_edges
    assert case.metadata["automatic_graph_mutation"] is False
    assert "fibrosis_to_compliance" not in case.public_operator_ids
    assert "compliance_to_tibial_rotation" not in case.public_operator_ids
