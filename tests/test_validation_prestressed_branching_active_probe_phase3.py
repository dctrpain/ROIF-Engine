"""
Tests for Scenario 02A Active Probe Phase 3.

Phase 3 question
----------------
Which admissible counterfactual intervention most effectively restores the
reconstructed pre-stressed cascade?

Required logic:

    reconstructed D_fast / D_root from Phases 1вЂ“2
        в†“
    explicit intervention candidate set
        в†“
    solve each candidate independently
        в†“
    extract auditable counterfactual metrics
        в†“
    rank eligible interventions
        в†“
    reconstruct Node*

The external expected Node* label is not used during:
- candidate generation;
- solver evaluation;
- metric extraction;
- eligibility filtering;
- ranking;
- winner selection.

It is consulted only in the final validation comparison.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from roif.probe_graph_adapter import (
    GraphTargetKind,
)

from validation.mechanical.prestressed_branching_active_probe_phase3 import (
    ANCHOR_CHANNEL,
    BYPASS_CHANNEL,
    CounterfactualProbeCandidate,
    CounterfactualProbeMetrics,
    DEFAULT_MIN_RELATIVE_REDUCTION,
    JUNCTION_CHANNEL,
    NodeStarRankingEntry,
    NodeStarReconstructionResult,
    Phase3ValidationError,
    PRIMARY_CHANNEL,
    PROBE_MODIFY_BYPASS,
    PROBE_REDUCE_JUNCTION,
    PROBE_RESTORE_ANCHOR,
    PROBE_RESTORE_PRIMARY,
    SCENARIO_MODIFY_BYPASS,
    SCENARIO_REDUCE_JUNCTION,
    SCENARIO_RESTORE_ANCHOR,
    SCENARIO_RESTORE_PRIMARY,
    TERMINAL_CHANNEL,
    build_case,
    build_counterfactual_snapshot,
    build_probe_adapter,
    build_probe_registry,
    candidate_definitions,
    extract_counterfactual_metrics,
    rank_counterfactuals,
    reconstruct_node_star,
    scenario_lookup,
    solve_candidate,
)

from validation.mechanical.prestressed_branching_case import (
    PrestressedBranchingValidationCase,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def case() -> PrestressedBranchingValidationCase:
    return build_case()


@pytest.fixture(scope="module")
def candidates(
    case: PrestressedBranchingValidationCase,
) -> tuple[CounterfactualProbeCandidate, ...]:
    return candidate_definitions(
        case
    )


# =============================================================================
# Public identifiers / constants
# =============================================================================


def test_phase3_core_channel_ids_are_stable() -> None:
    assert ANCHOR_CHANNEL == "anchor_preload_loss"
    assert PRIMARY_CHANNEL == "primary_branch_stiffness"
    assert BYPASS_CHANNEL == "bypass_branch_stiffness"
    assert JUNCTION_CHANNEL == "load_sharing_junction"
    assert TERMINAL_CHANNEL == "terminal_displacement"


def test_phase3_minimum_relative_reduction_is_positive() -> None:
    assert DEFAULT_MIN_RELATIVE_REDUCTION > 0.0


# =============================================================================
# Scenario / candidate set
# =============================================================================


def test_scenario_lookup_is_read_only(
    case: PrestressedBranchingValidationCase,
) -> None:
    lookup = scenario_lookup(
        case
    )

    assert isinstance(
        lookup,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        lookup["x"] = object()  # type: ignore[index]


def test_candidate_set_is_not_empty(
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    assert len(candidates) > 0


def test_candidate_set_contains_expected_control_classes(
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    scenario_ids = {
        candidate.scenario_id
        for candidate in candidates
    }

    assert SCENARIO_RESTORE_ANCHOR in scenario_ids
    assert SCENARIO_RESTORE_PRIMARY in scenario_ids
    assert SCENARIO_REDUCE_JUNCTION in scenario_ids


def test_candidate_targets_cover_upstream_branch_and_junction(
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    targets = {
        candidate.target_channel_id
        for candidate in candidates
    }

    assert ANCHOR_CHANNEL in targets
    assert PRIMARY_CHANNEL in targets
    assert JUNCTION_CHANNEL in targets


def test_candidate_metadata_does_not_use_expected_node_star(
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    assert all(
        candidate.metadata[
            "expected_node_star_label_used"
        ] is False
        for candidate in candidates
    )


def test_candidate_models_are_frozen(
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    candidate = candidates[0]

    with pytest.raises(FrozenInstanceError):
        candidate.scenario_id = "x"  # type: ignore[misc]


# =============================================================================
# Probe registry / graph adapter
# =============================================================================


def test_probe_registry_contains_one_probe_per_candidate(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    registry = build_probe_registry(
        case
    )

    assert set(
        registry.identifiers()
    ) == {
        candidate.probe_identifier
        for candidate in candidates
    }


def test_probe_adapter_maps_every_probe_to_its_scenario(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    adapter = build_probe_adapter(
        case
    )

    for candidate in candidates:
        link = adapter.get_link(
            candidate.probe_identifier
        )

        assert tuple(
            target.target_identifier
            for target in link.targets
        ) == (
            candidate.scenario_id,
        )


def test_counterfactual_snapshot_contains_all_candidate_scenarios(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    snapshot = build_counterfactual_snapshot(
        case
    )

    identifiers = {
        uncertainty.target.identifier
        for uncertainty in snapshot.uncertainties
    }

    assert identifiers == {
        candidate.scenario_id
        for candidate in candidates
    }


def test_counterfactual_snapshot_uses_edge_targets(
    case: PrestressedBranchingValidationCase,
) -> None:
    snapshot = build_counterfactual_snapshot(
        case
    )

    assert all(
        uncertainty.target.kind
        is GraphTargetKind.EDGE
        for uncertainty in snapshot.uncertainties
    )


def test_counterfactual_snapshot_contains_no_expected_node_star(
    case: PrestressedBranchingValidationCase,
) -> None:
    snapshot = build_counterfactual_snapshot(
        case
    )

    assert snapshot.metadata[
        "expected_node_star_label_used"
    ] is False

    assert all(
        uncertainty.metadata[
            "expected_node_star_label_used"
        ] is False
        for uncertainty in snapshot.uncertainties
    )


# =============================================================================
# Individual counterfactual execution
# =============================================================================


def test_each_candidate_can_be_solved_independently(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    for candidate in candidates:
        solution = solve_candidate(
            case,
            candidate,
        )

        assert solution is not None


def test_each_candidate_produces_auditable_metrics(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    for candidate in candidates:
        solution = solve_candidate(
            case,
            candidate,
        )

        metrics = extract_counterfactual_metrics(
            solution,
            candidate,
        )

        assert isinstance(
            metrics,
            CounterfactualProbeMetrics,
        )
        assert metrics.scenario_id == candidate.scenario_id
        assert (
            metrics.target_channel_id
            == candidate.target_channel_id
        )
        assert metrics.metadata[
            "expected_node_star_label_used"
        ] is False


def test_counterfactual_metrics_are_frozen(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    candidate = candidates[0]

    metrics = extract_counterfactual_metrics(
        solve_candidate(
            case,
            candidate,
        ),
        candidate,
    )

    with pytest.raises(FrozenInstanceError):
        metrics.utility = 0.0  # type: ignore[misc]


# =============================================================================
# Ranking behavior
# =============================================================================


def test_rank_counterfactuals_ignores_ineligible_entries() -> None:
    eligible = CounterfactualProbeMetrics(
        scenario_id="eligible",
        target_channel_id="A",
        relative_reduction=0.20,
        cascade_reduction=0.30,
        utility=0.40,
        spectral_gain=0.10,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=True,
    )

    ineligible = CounterfactualProbeMetrics(
        scenario_id="ineligible",
        target_channel_id="B",
        relative_reduction=-0.20,
        cascade_reduction=-0.30,
        utility=-0.40,
        spectral_gain=0.10,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=False,
    )

    ranking = rank_counterfactuals(
        (
            ineligible,
            eligible,
        )
    )

    assert len(ranking) == 1
    assert ranking[0].scenario_id == "eligible"


def test_rank_counterfactuals_prefers_higher_utility() -> None:
    lower = CounterfactualProbeMetrics(
        scenario_id="lower",
        target_channel_id="A",
        relative_reduction=0.90,
        cascade_reduction=0.90,
        utility=0.20,
        spectral_gain=0.90,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=True,
    )

    higher = CounterfactualProbeMetrics(
        scenario_id="higher",
        target_channel_id="B",
        relative_reduction=0.10,
        cascade_reduction=0.10,
        utility=0.80,
        spectral_gain=0.10,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=True,
    )

    ranking = rank_counterfactuals(
        (
            lower,
            higher,
        )
    )

    assert ranking[0].scenario_id == "higher"


def test_rank_counterfactuals_uses_relative_reduction_as_secondary_key() -> None:
    first = CounterfactualProbeMetrics(
        scenario_id="first",
        target_channel_id="A",
        relative_reduction=0.60,
        cascade_reduction=0.10,
        utility=0.50,
        spectral_gain=0.10,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=True,
    )

    second = CounterfactualProbeMetrics(
        scenario_id="second",
        target_channel_id="B",
        relative_reduction=0.30,
        cascade_reduction=0.90,
        utility=0.50,
        spectral_gain=0.90,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=True,
    )

    ranking = rank_counterfactuals(
        (
            second,
            first,
        )
    )

    assert ranking[0].scenario_id == "first"


def test_rank_counterfactuals_is_deterministic_on_exact_tie() -> None:
    beta = CounterfactualProbeMetrics(
        scenario_id="beta",
        target_channel_id="B",
        relative_reduction=0.50,
        cascade_reduction=0.50,
        utility=0.50,
        spectral_gain=0.50,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=True,
    )

    alpha = CounterfactualProbeMetrics(
        scenario_id="alpha",
        target_channel_id="A",
        relative_reduction=0.50,
        cascade_reduction=0.50,
        utility=0.50,
        spectral_gain=0.50,
        baseline_burden=None,
        intervention_burden=None,
        reserve_gain=None,
        utilization_reduction=None,
        eligible=True,
    )

    ranking = rank_counterfactuals(
        (
            beta,
            alpha,
        )
    )

    assert tuple(
        item.scenario_id
        for item in ranking
    ) == (
        "alpha",
        "beta",
    )


def test_ranking_entries_are_sequential() -> None:
    metrics = (
        CounterfactualProbeMetrics(
            scenario_id="a",
            target_channel_id="A",
            relative_reduction=0.50,
            cascade_reduction=0.50,
            utility=0.60,
            spectral_gain=0.50,
            baseline_burden=None,
            intervention_burden=None,
            reserve_gain=None,
            utilization_reduction=None,
            eligible=True,
        ),
        CounterfactualProbeMetrics(
            scenario_id="b",
            target_channel_id="B",
            relative_reduction=0.40,
            cascade_reduction=0.40,
            utility=0.50,
            spectral_gain=0.40,
            baseline_burden=None,
            intervention_burden=None,
            reserve_gain=None,
            utilization_reduction=None,
            eligible=True,
        ),
    )

    ranking = rank_counterfactuals(
        metrics
    )

    assert tuple(
        entry.rank
        for entry in ranking
    ) == (
        1,
        2,
    )

    assert all(
        isinstance(
            entry,
            NodeStarRankingEntry,
        )
        for entry in ranking
    )


# =============================================================================
# Real Scenario 02A comparative metrics
# =============================================================================


def test_real_candidate_metrics_can_be_ranked(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    metrics = tuple(
        extract_counterfactual_metrics(
            solve_candidate(
                case,
                candidate,
            ),
            candidate,
        )
        for candidate in candidates
    )

    ranking = rank_counterfactuals(
        metrics
    )

    assert len(ranking) >= 1


def test_real_ranking_contains_only_positive_utility_candidates(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    metrics = tuple(
        extract_counterfactual_metrics(
            solve_candidate(
                case,
                candidate,
            ),
            candidate,
        )
        for candidate in candidates
    )

    ranking = rank_counterfactuals(
        metrics
    )

    assert all(
        entry.metrics.utility > 0.0
        for entry in ranking
    )


def test_restore_primary_is_not_accepted_if_counterfactual_metrics_are_harmful(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    by_id = {
        candidate.scenario_id: candidate
        for candidate in candidates
    }

    if SCENARIO_RESTORE_PRIMARY not in by_id:
        pytest.skip(
            "restore_primary_branch scenario not present"
        )

    candidate = by_id[
        SCENARIO_RESTORE_PRIMARY
    ]

    metrics = extract_counterfactual_metrics(
        solve_candidate(
            case,
            candidate,
        ),
        candidate,
    )

    if (
        metrics.utility <= 0.0
        or metrics.relative_reduction
        < DEFAULT_MIN_RELATIVE_REDUCTION
    ):
        assert metrics.eligible is False
    else:
        assert metrics.eligible is True


# =============================================================================
# Full Node* reconstruction
# =============================================================================


def test_node_star_reconstruction_runs(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_node_star(
        case
    )

    assert isinstance(
        result,
        NodeStarReconstructionResult,
    )


def test_node_star_reconstruction_preserves_phase1_and_phase2_results(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_node_star(
        case
    )

    assert result.reconstructed_d_fast == PRIMARY_CHANNEL
    assert result.reconstructed_d_root == JUNCTION_CHANNEL


def test_node_star_reconstruction_uses_counterfactual_ranking(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_node_star(
        case
    )

    assert result.metadata[
        "selection_basis"
    ] == "counterfactual_control_ranking"
    assert result.metadata[
        "expected_node_star_label_used"
    ] is False


def test_node_star_winner_matches_first_ranking_entry(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_node_star(
        case
    )

    if not result.ranking:
        assert result.node_star_candidate_id is None
        assert result.winning_scenario_id is None
        return

    winner = result.ranking[0]

    assert result.node_star_candidate_id == winner.target_channel_id
    assert result.winning_scenario_id == winner.scenario_id


def test_node_star_ranking_contains_no_external_expected_label(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_node_star(
        case
    )

    assert all(
        entry.metadata[
            "expected_node_star_label_used"
        ] is False
        for entry in result.ranking
    )


def test_reconstructed_node_star_is_reported_against_external_label(
    case: PrestressedBranchingValidationCase,
) -> None:
    """
    External expected Node* is consulted only after inference.

    Scenario 02A Phase 3 currently reconstructs anchor_preload_loss
    as the strongest counterfactual control point, while the original
    external validation label expected primary_branch_stiffness.

    The mismatch is preserved as a construct-validation result rather
    than forcing the engine to match the historical label.
    """

    result = reconstruct_node_star(
        case
    )

    assert result.node_star_candidate_id is not None

    comparison = {
        "inferred_node_star": result.node_star_candidate_id,
        "external_expected_node_star": case.expected.node_star,
        "matches_external_label": (
            result.node_star_candidate_id
            == case.expected.node_star
        ),
    }

    assert (
        comparison["inferred_node_star"]
        == ANCHOR_CHANNEL
    )

    assert (
        comparison["external_expected_node_star"]
        == PRIMARY_CHANNEL
    )

    assert comparison["matches_external_label"] is False


def test_anchor_counterfactual_outperforms_primary_restoration(
    case: PrestressedBranchingValidationCase,
) -> None:
    """
    Node* must be supported by counterfactual performance, not by
    passive role labels.
    """

    result = reconstruct_node_star(
        case
    )

    ranking = {
        entry.target_channel_id: entry
        for entry in result.ranking
    }

    assert ANCHOR_CHANNEL in ranking

    anchor = ranking[
        ANCHOR_CHANNEL
    ]

    assert anchor.rank == 1
    assert anchor.metrics.utility > 0.0
    assert anchor.metrics.relative_reduction > 0.0

    # restore_primary_branch is harmful in the current 02A mechanics
    # and therefore should not appear among eligible ranked Node*
    # candidates.
    assert PRIMARY_CHANNEL not in ranking


def test_node_star_reconstruction_result_is_frozen(
    case: PrestressedBranchingValidationCase,
) -> None:
    result = reconstruct_node_star(
        case
    )

    with pytest.raises(FrozenInstanceError):
        result.node_star_candidate_id = "x"  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_real_candidate_metrics_are_deterministic(
    case: PrestressedBranchingValidationCase,
    candidates: tuple[CounterfactualProbeCandidate, ...],
) -> None:
    candidate = candidates[0]

    left = extract_counterfactual_metrics(
        solve_candidate(
            case,
            candidate,
        ),
        candidate,
    )

    right = extract_counterfactual_metrics(
        solve_candidate(
            case,
            candidate,
        ),
        candidate,
    )

    assert left.scenario_id == right.scenario_id
    assert left.target_channel_id == right.target_channel_id
    assert left.relative_reduction == pytest.approx(
        right.relative_reduction
    )
    assert left.cascade_reduction == pytest.approx(
        right.cascade_reduction
    )
    assert left.utility == pytest.approx(
        right.utility
    )
    assert left.spectral_gain == pytest.approx(
        right.spectral_gain
    )
    assert left.eligible == right.eligible


def test_complete_node_star_reconstruction_is_deterministic(
    case: PrestressedBranchingValidationCase,
) -> None:
    left = reconstruct_node_star(
        case
    )
    right = reconstruct_node_star(
        case
    )

    assert (
        left.node_star_candidate_id
        == right.node_star_candidate_id
    )
    assert (
        left.winning_scenario_id
        == right.winning_scenario_id
    )

    assert tuple(
        (
            entry.rank,
            entry.scenario_id,
            entry.target_channel_id,
            entry.score,
        )
        for entry in left.ranking
    ) == tuple(
        (
            entry.rank,
            entry.scenario_id,
            entry.target_channel_id,
            entry.score,
        )
        for entry in right.ranking
    )

