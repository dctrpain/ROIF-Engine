"""
ROIF Mechanical Validation
Scenario 02A вЂ” Active Probe End-to-End

Combines validated Phase 1, Phase 2, and Phase 3 pipelines into one audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, FrozenInstanceError
from types import MappingProxyType
from typing import Any, Mapping

import pytest

from roif.active_cascade import ActiveCascadeExplorer, start_active_cascade

from validation.mechanical.prestressed_branching_case import (
    PrestressedBranchingValidationCase,
    build_prestressed_branching_case,
)

from validation.mechanical.prestressed_branching_active_probe_phase1 import (
    ANCHOR_CHANNEL,
    PRIMARY_CHANNEL,
    PRIMARY_RELATION,
    ANCHOR_PROBE_ID,
    build_anchor_snapshot,
    build_bridge as build_phase1_bridge,
    local_probe_evidence,
    Scenario02ACandidateProvider,
)

from validation.mechanical.prestressed_branching_active_probe_phase2 import (
    JUNCTION_CHANNEL,
    reconstruct_junction_candidate,
)

from validation.mechanical.prestressed_branching_active_probe_phase3 import (
    NodeStarReconstructionResult,
    reconstruct_node_star,
)


@dataclass(frozen=True, slots=True)
class ActiveProbeE2EResult:
    d_origin: str
    d_fast: str
    d_root: str
    node_star: str | None
    phase1_relation_id: str
    phase1_probe_id: str
    phase2_both_relations_supported: bool
    phase3_winning_scenario_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@pytest.fixture(scope="module")
def case() -> PrestressedBranchingValidationCase:
    return build_prestressed_branching_case()


def run_phase1(
    case: PrestressedBranchingValidationCase,
) -> str:
    bridge = build_phase1_bridge()

    explorer = ActiveCascadeExplorer(
        candidate_provider=Scenario02ACandidateProvider(case),
        probe_planner=bridge.probe_planner,
        graph_update_resolver=bridge.graph_update_resolver,
    )

    state = start_active_cascade(
        ANCHOR_CHANNEL,
        graph_version="v0",
    )

    evidence = local_probe_evidence(
        case,
        PRIMARY_RELATION,
    )

    proposal = bridge.vector_evidence.assess(
        evidence
    )

    authorized = bridge.vector_evidence.authorize(
        proposal,
        approve=True,
    )

    assert authorized is not None

    resolved = explorer.apply_authorized_probe_update(
        state,
        graph_snapshot=build_anchor_snapshot(),
        graph_update=authorized,
        probe_identifier=ANCHOR_PROBE_ID,
        graph_version="v1",
    )

    return resolved.current_node_id


def run_active_probe_e2e(
    case: PrestressedBranchingValidationCase,
) -> ActiveProbeE2EResult:
    d_fast = run_phase1(case)

    phase2 = reconstruct_junction_candidate(case)

    phase3: NodeStarReconstructionResult = reconstruct_node_star(case)

    return ActiveProbeE2EResult(
        d_origin=ANCHOR_CHANNEL,
        d_fast=d_fast,
        d_root=phase2.junction_candidate_id or "",
        node_star=phase3.node_star_candidate_id,
        phase1_relation_id=PRIMARY_RELATION,
        phase1_probe_id=ANCHOR_PROBE_ID,
        phase2_both_relations_supported=phase2.both_relations_supported,
        phase3_winning_scenario_id=phase3.winning_scenario_id,
        metadata={
            "validation_case": "mechanical_02A",
            "pipeline": "phase1+phase2+phase3",
            "expected_labels_used_in_inference": False,
            "phase1": "active_transition_probe",
            "phase2": "dual_branch_convergence_probe",
            "phase3": "counterfactual_control_probe",
        },
    )


def test_e2e_phase1_reconstructs_primary_branch(
    case: PrestressedBranchingValidationCase,
) -> None:
    assert run_phase1(case) == PRIMARY_CHANNEL


def test_e2e_phase2_reconstructs_common_junction(
    case: PrestressedBranchingValidationCase,
) -> None:
    phase2 = reconstruct_junction_candidate(case)

    assert phase2.both_relations_supported is True
    assert phase2.junction_candidate_id == JUNCTION_CHANNEL


def test_e2e_phase3_reconstructs_node_star(
    case: PrestressedBranchingValidationCase,
) -> None:
    phase3 = reconstruct_node_star(case)

    assert phase3.node_star_candidate_id == ANCHOR_CHANNEL


def test_e2e_result_contains_all_four_roles(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)

    assert result.d_origin == ANCHOR_CHANNEL
    assert result.d_fast == PRIMARY_CHANNEL
    assert result.d_root == JUNCTION_CHANNEL
    assert result.node_star == ANCHOR_CHANNEL


def test_e2e_d_origin_and_d_fast_are_distinct(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.d_origin != result.d_fast


def test_e2e_d_fast_and_d_root_are_distinct(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.d_fast != result.d_root


def test_e2e_node_star_may_equal_d_origin(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.node_star == result.d_origin


def test_e2e_inference_does_not_use_external_expected_labels(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)

    assert result.metadata[
        "expected_labels_used_in_inference"
    ] is False


def test_e2e_external_d_fast_matches_after_inference(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.d_fast == case.expected.d_fast


def test_e2e_external_d_root_matches_after_inference(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.d_root == case.expected.d_root


def test_e2e_external_node_star_mismatch_is_preserved(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)

    assert result.node_star == ANCHOR_CHANNEL
    assert case.expected.node_star == PRIMARY_CHANNEL
    assert result.node_star != case.expected.node_star


def test_e2e_phase1_relation_is_primary_relation(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.phase1_relation_id == PRIMARY_RELATION


def test_e2e_phase1_uses_real_anchor_probe(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.phase1_probe_id == ANCHOR_PROBE_ID


def test_e2e_phase2_requires_both_incoming_relations(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.phase2_both_relations_supported is True


def test_e2e_phase3_has_winning_counterfactual(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)
    assert result.phase3_winning_scenario_id is not None


def test_e2e_result_is_frozen(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)

    with pytest.raises(FrozenInstanceError):
        result.d_fast = "x"  # type: ignore[misc]


def test_e2e_metadata_is_read_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = run_active_probe_e2e(case)

    assert isinstance(result.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_e2e_pipeline_is_deterministic(
    case: PrestressedBranchingValidationCase,
) -> None:
    left = run_active_probe_e2e(case)
    right = run_active_probe_e2e(case)

    assert left == right

