"""
ROIF Mechanical Validation
Scenario 02A вЂ” Active Probe Phase 3

Purpose
-------
Phase 1 actively reconstructed D_fast.
Phase 2 actively reconstructed the convergent D_root candidate.

Phase 3 asks a different question:

    Which admissible counterfactual intervention produces the strongest
    confirmed restoration of the reconstructed pre-stressed cascade?

This is the Node* search layer.

Important distinction
---------------------
Propagation Probe:
    "Where does the cascade go?"

Counterfactual control Probe:
    "Which intervention most effectively restores system function?"

Node* is therefore NOT selected from passive node score alone.

It is selected from comparative counterfactual response across admissible
interventions.

Safety / audit boundary
-----------------------
- expected Node* label is not used in ranking;
- no hidden validation role labels are injected into the Probe engine;
- all intervention candidates are explicit and auditable;
- non-fonit / cascade-risk limits remain part of Probe policy;
- this module produces validation evidence, not treatment advice.

Physical model
--------------
This Phase 3 module uses Scenario 02A's existing ROIF system and solver
counterfactual machinery. Each intervention candidate is evaluated by running
the normal ROIF solver over an explicitly supplied scenario subset and
extracting the resulting metrics.

The active Probe layer then interprets those counterfactual outcomes as
control evidence.

This is construct/integration validation, not independent FEA or multibody
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import math

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
from roif.active_probe_engine import ActiveProbeEngine

from roif.active_cascade_ape import (
    ActiveCascadeAPEConfig,
)

from roif.roif_solver import (
    ROIFSolution,
    solve_roif,
)

from validation.mechanical.prestressed_branching_case import (
    PrestressedBranchingValidationCase,
    build_prestressed_branching_case,
)

from validation.mechanical.prestressed_branching_active_probe_phase2 import (
    JUNCTION_CHANNEL,
    PRIMARY_CHANNEL,
    BYPASS_CHANNEL,
    reconstruct_junction_candidate,
)


# =============================================================================
# Public identifiers
# =============================================================================


ANCHOR_CHANNEL = "anchor_preload_loss"
TERMINAL_CHANNEL = "terminal_displacement"

SCENARIO_RESTORE_ANCHOR = "remove_anchor_defect_influence"
SCENARIO_RESTORE_PRIMARY = "restore_primary_branch"
SCENARIO_MODIFY_BYPASS = "modify_bypass_branch"
SCENARIO_REDUCE_JUNCTION = "reduce_junction_load"

PROBE_RESTORE_ANCHOR = "probe_02A_restore_anchor"
PROBE_RESTORE_PRIMARY = "probe_02A_restore_primary"
PROBE_MODIFY_BYPASS = "probe_02A_modify_bypass"
PROBE_REDUCE_JUNCTION = "probe_02A_reduce_junction"

DEFAULT_MIN_RELATIVE_REDUCTION = 0.01


# =============================================================================
# Errors
# =============================================================================


class Phase3ValidationError(ValueError):
    """Raised when Scenario 02A Phase 3 cannot be evaluated safely."""


# =============================================================================
# Read-only result models
# =============================================================================


@dataclass(frozen=True, slots=True)
class CounterfactualProbeCandidate:
    """
    One admissible Node* control candidate.
    """

    probe_identifier: str
    scenario_id: str
    target_channel_id: str
    rationale: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "probe_identifier",
            "scenario_id",
            "target_channel_id",
            "rationale",
        ):
            value = str(
                getattr(self, name)
            ).strip()

            if not value:
                raise Phase3ValidationError(
                    f"{name} must not be empty."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualProbeMetrics:
    """
    Normalized audit metrics for one intervention scenario.
    """

    scenario_id: str
    target_channel_id: str

    relative_reduction: float
    cascade_reduction: float
    utility: float
    spectral_gain: float

    baseline_burden: float | None
    intervention_burden: float | None

    reserve_gain: float | None
    utilization_reduction: float | None

    eligible: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "relative_reduction",
            "cascade_reduction",
            "utility",
            "spectral_gain",
        ):
            value = float(
                getattr(self, name)
            )

            if not math.isfinite(value):
                raise Phase3ValidationError(
                    f"{name} must be finite."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class NodeStarRankingEntry:
    """
    Ranked intervention evidence.

    score is intentionally transparent:
        primary key   = utility
        secondary     = relative_reduction
        tertiary      = cascade_reduction
        quaternary    = spectral_gain

    No expected Node* label is used.
    """

    rank: int
    scenario_id: str
    target_channel_id: str
    score: tuple[float, float, float, float]
    metrics: CounterfactualProbeMetrics
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise Phase3ValidationError(
                "rank must be >= 1."
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class NodeStarReconstructionResult:
    """
    Final Phase 3 result.
    """

    reconstructed_d_fast: str
    reconstructed_d_root: str
    node_star_candidate_id: str | None
    winning_scenario_id: str | None
    ranking: tuple[NodeStarRankingEntry, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ranking",
            tuple(self.ranking),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


# =============================================================================
# Scenario helpers
# =============================================================================


def build_case() -> PrestressedBranchingValidationCase:
    return build_prestressed_branching_case()


def scenario_lookup(
    case: PrestressedBranchingValidationCase,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            scenario.scenario_id: scenario
            for scenario in case.scenarios
        }
    )


def candidate_definitions(
    case: PrestressedBranchingValidationCase,
) -> tuple[CounterfactualProbeCandidate, ...]:
    """
    Build explicit Node* candidate set.

    Expected role labels are not consulted here.
    """

    available = scenario_lookup(case)

    mapping = (
        (
            PROBE_RESTORE_ANCHOR,
            SCENARIO_RESTORE_ANCHOR,
            ANCHOR_CHANNEL,
            "Restore upstream anchor preload influence.",
        ),
        (
            PROBE_RESTORE_PRIMARY,
            SCENARIO_RESTORE_PRIMARY,
            PRIMARY_CHANNEL,
            "Restore the primary pre-stressed branch.",
        ),
        (
            PROBE_MODIFY_BYPASS,
            SCENARIO_MODIFY_BYPASS,
            BYPASS_CHANNEL,
            "Modify the bypass compensatory branch.",
        ),
        (
            PROBE_REDUCE_JUNCTION,
            SCENARIO_REDUCE_JUNCTION,
            JUNCTION_CHANNEL,
            "Reduce load at the convergent junction.",
        ),
    )

    result: list[
        CounterfactualProbeCandidate
    ] = []

    for (
        probe_id,
        scenario_id,
        target_channel_id,
        rationale,
    ) in mapping:
        if scenario_id not in available:
            continue

        result.append(
            CounterfactualProbeCandidate(
                probe_identifier=probe_id,
                scenario_id=scenario_id,
                target_channel_id=target_channel_id,
                rationale=rationale,
                metadata={
                    "validation_case": "mechanical_02A",
                    "phase": "active_probe_3",
                    "expected_node_star_label_used": False,
                },
            )
        )

    if not result:
        raise Phase3ValidationError(
            "no Phase 3 intervention candidates are available."
        )

    return tuple(result)


# =============================================================================
# Probe registry / graph mapping
# =============================================================================


def _probe_definition(
    candidate: CounterfactualProbeCandidate,
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=candidate.probe_identifier,
        name=(
            f"02A counterfactual control Probe: "
            f"{candidate.target_channel_id}"
        ),
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.DYNAMIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=PerturbationType.SIMULATED_INTERVENTION,
            magnitude=1.0,
            magnitude_units="scenario",
            duration_seconds=1.0,
        ),
        description=candidate.rationale,
        tags=(
            "validation",
            "mechanical_02A",
            "active_probe",
            "phase3",
            "node_star",
            "counterfactual",
        ),
    )


def build_probe_registry(
    case: PrestressedBranchingValidationCase,
) -> ProbeRegistry:
    return ProbeRegistry(
        tuple(
            _probe_definition(candidate)
            for candidate in candidate_definitions(case)
        )
    )


def build_probe_adapter(
    case: PrestressedBranchingValidationCase,
) -> ProbeGraphAdapter:
    links: list[
        ProbeGraphLink
    ] = []

    for candidate in candidate_definitions(case):
        links.append(
            ProbeGraphLink(
                probe_identifier=candidate.probe_identifier,
                targets=(
                    ProbeGraphTarget(
                        target_identifier=candidate.scenario_id,
                        sensitivity=1.0,
                        discrimination=1.0,
                        expected_uncertainty_reduction=1.0,
                    ),
                ),
                novelty=0.75,
                feasibility=1.0,
                estimate_confidence=0.95,
                metadata={
                    "validation_case": "mechanical_02A",
                    "phase": "active_probe_3",
                    "expected_node_star_label_used": False,
                },
            )
        )

    return ProbeGraphAdapter(
        tuple(links)
    )


def build_counterfactual_snapshot(
    case: PrestressedBranchingValidationCase,
) -> IncompleteGraphSnapshot:
    uncertainties: list[
        GraphUncertainty
    ] = []

    for candidate in candidate_definitions(case):
        uncertainties.append(
            GraphUncertainty(
                target=GraphTarget(
                    kind=GraphTargetKind.EDGE,
                    identifier=candidate.scenario_id,
                    source="counterfactual_control",
                    target=candidate.target_channel_id,
                ),
                uncertainty=0.90,
                importance=1.0,
                confidence=0.50,
                hypothesis_count=len(
                    candidate_definitions(case)
                ),
                metadata={
                    "expected_node_star_label_used": False,
                },
            )
        )

    return IncompleteGraphSnapshot(
        graph_id="mechanical_02A_phase3_node_star",
        version="v0",
        uncertainties=tuple(uncertainties),
        metadata={
            "validation_case": "mechanical_02A",
            "phase": "active_probe_3",
            "expected_node_star_label_used": False,
        },
    )


# =============================================================================
# Solver metric extraction
# =============================================================================


def solve_candidate(
    case: PrestressedBranchingValidationCase,
    candidate: CounterfactualProbeCandidate,
) -> ROIFSolution:
    """
    Evaluate exactly one explicit counterfactual intervention.
    """

    scenarios = scenario_lookup(case)

    scenario = scenarios[
        candidate.scenario_id
    ]

    return solve_roif(
        case.pathological_system,
        initial_state=case.initial_state,
        scenarios=(
            scenario,
        ),
        config=case.solver_config,
    )


def _metric_from_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    default: float = 0.0,
) -> float:
    value = mapping.get(
        key,
        default,
    )

    return float(value)


def _scenario_metrics_mapping(
    solution: ROIFSolution,
    scenario_id: str,
) -> Mapping[str, Any]:
    """
    Read the evaluated counterfactual metrics for one scenario.

    Phase 3 must inspect the counterfactual evaluation itself rather
    than only SolverDecision. SolverDecision represents the final
    selection policy and may intentionally contain scenario_id=None
    when no scenario is selected, even though the counterfactual was
    evaluated successfully.

    CounterfactualBatchResult therefore forms the primary audit source
    for Phase 3 Node* reconstruction.
    """

    batch = getattr(
        solution,
        "counterfactual_batch",
        None,
    )

    results = getattr(
        batch,
        "results",
        (),
    )

    for result in results:
        scenario = getattr(
            result,
            "scenario",
            None,
        )

        result_scenario_id = getattr(
            scenario,
            "scenario_id",
            None,
        )

        if result_scenario_id != scenario_id:
            continue

        metrics = getattr(
            result,
            "metrics",
            None,
        )

        if metrics is None:
            raise Phase3ValidationError(
                f"counterfactual metrics missing for {scenario_id!r}."
            )

        return MappingProxyType(
            {
                "baseline_burden": float(
                    metrics.baseline_burden
                ),
                "scenario_burden": float(
                    metrics.scenario_burden
                ),
                "cascade_reduction": float(
                    metrics.cascade_reduction
                ),
                "relative_reduction": float(
                    metrics.relative_reduction
                ),
                "final_state_reduction": float(
                    metrics.final_state_reduction
                ),
                "peak_reduction": float(
                    metrics.peak_reduction
                ),
                "spectral_gain": float(
                    metrics.spectral_gain
                ),
                "collateral_effect": float(
                    metrics.collateral_effect
                ),
                "cost": float(
                    metrics.cost
                ),
                "safety_risk": float(
                    metrics.safety_risk
                ),
                "uncertainty": float(
                    metrics.uncertainty
                ),
                "irreversibility": float(
                    metrics.irreversibility
                ),
                "complexity": float(
                    metrics.complexity
                ),
                "utility": float(
                    metrics.utility
                ),
                "safe": bool(
                    metrics.safe
                ),
                "solver_metric_source": "counterfactual_batch",
            }
        )

    raise Phase3ValidationError(
        f"counterfactual result unavailable for {scenario_id!r}."
    )

def extract_counterfactual_metrics(
    solution: ROIFSolution,
    candidate: CounterfactualProbeCandidate,
) -> CounterfactualProbeMetrics:
    metrics = _scenario_metrics_mapping(
        solution,
        candidate.scenario_id,
    )

    relative_reduction = _metric_from_mapping(
        metrics,
        "relative_reduction",
    )
    cascade_reduction = _metric_from_mapping(
        metrics,
        "cascade_reduction",
    )
    utility = _metric_from_mapping(
        metrics,
        "utility",
    )
    spectral_gain = _metric_from_mapping(
        metrics,
        "spectral_gain",
    )

    baseline_burden_raw = metrics.get(
        "baseline_burden"
    )
    intervention_burden_raw = metrics.get(
        "intervention_burden"
    )
    reserve_gain_raw = metrics.get(
        "reserve_gain"
    )
    utilization_reduction_raw = metrics.get(
        "utilization_reduction"
    )

    baseline_burden = (
        None
        if baseline_burden_raw is None
        else float(baseline_burden_raw)
    )
    intervention_burden = (
        None
        if intervention_burden_raw is None
        else float(intervention_burden_raw)
    )
    reserve_gain = (
        None
        if reserve_gain_raw is None
        else float(reserve_gain_raw)
    )
    utilization_reduction = (
        None
        if utilization_reduction_raw is None
        else float(utilization_reduction_raw)
    )

    eligible = (
        utility > 0.0
        and relative_reduction
        >= DEFAULT_MIN_RELATIVE_REDUCTION
    )

    return CounterfactualProbeMetrics(
        scenario_id=candidate.scenario_id,
        target_channel_id=candidate.target_channel_id,
        relative_reduction=relative_reduction,
        cascade_reduction=cascade_reduction,
        utility=utility,
        spectral_gain=spectral_gain,
        baseline_burden=baseline_burden,
        intervention_burden=intervention_burden,
        reserve_gain=reserve_gain,
        utilization_reduction=utilization_reduction,
        eligible=eligible,
        metadata={
            "expected_node_star_label_used": False,
        },
    )


# =============================================================================
# Comparative ranking
# =============================================================================


def rank_counterfactuals(
    metrics: Iterable[
        CounterfactualProbeMetrics
    ],
) -> tuple[NodeStarRankingEntry, ...]:
    """
    Rank only eligible interventions.

    Ordering:
        utility
        relative_reduction
        cascade_reduction
        spectral_gain
        scenario_id (deterministic tie break)
    """

    eligible = [
        item
        for item in metrics
        if item.eligible
    ]

    ordered = sorted(
        eligible,
        key=lambda item: (
            -item.utility,
            -item.relative_reduction,
            -item.cascade_reduction,
            -item.spectral_gain,
            item.scenario_id,
        ),
    )

    return tuple(
        NodeStarRankingEntry(
            rank=index + 1,
            scenario_id=item.scenario_id,
            target_channel_id=item.target_channel_id,
            score=(
                item.utility,
                item.relative_reduction,
                item.cascade_reduction,
                item.spectral_gain,
            ),
            metrics=item,
            metadata={
                "expected_node_star_label_used": False,
            },
        )
        for index, item in enumerate(ordered)
    )


# =============================================================================
# Full Phase 3 reconstruction
# =============================================================================


def reconstruct_node_star(
    case: PrestressedBranchingValidationCase | None = None,
) -> NodeStarReconstructionResult:
    case = (
        case
        or build_case()
    )

    phase2 = reconstruct_junction_candidate(
        case
    )

    if (
        not phase2.both_relations_supported
        or phase2.junction_candidate_id
        != JUNCTION_CHANNEL
    ):
        raise Phase3ValidationError(
            "Phase 2 convergence must be resolved before Node* search."
        )

    candidates = candidate_definitions(
        case
    )

    collected: list[
        CounterfactualProbeMetrics
    ] = []

    for candidate in candidates:
        solution = solve_candidate(
            case,
            candidate,
        )

        collected.append(
            extract_counterfactual_metrics(
                solution,
                candidate,
            )
        )

    ranking = rank_counterfactuals(
        collected
    )

    if ranking:
        winner = ranking[0]
        node_star_candidate_id = (
            winner.target_channel_id
        )
        winning_scenario_id = (
            winner.scenario_id
        )
    else:
        node_star_candidate_id = None
        winning_scenario_id = None

    return NodeStarReconstructionResult(
        reconstructed_d_fast=PRIMARY_CHANNEL,
        reconstructed_d_root=JUNCTION_CHANNEL,
        node_star_candidate_id=node_star_candidate_id,
        winning_scenario_id=winning_scenario_id,
        ranking=ranking,
        metadata={
            "validation_case": "mechanical_02A",
            "phase": "active_probe_3",
            "selection_basis": "counterfactual_control_ranking",
            "expected_node_star_label_used": False,
            "candidate_count": len(candidates),
        },
    )


__all__ = [
    "ANCHOR_CHANNEL",
    "BYPASS_CHANNEL",
    "CounterfactualProbeCandidate",
    "CounterfactualProbeMetrics",
    "DEFAULT_MIN_RELATIVE_REDUCTION",
    "JUNCTION_CHANNEL",
    "NodeStarRankingEntry",
    "NodeStarReconstructionResult",
    "Phase3ValidationError",
    "PRIMARY_CHANNEL",
    "PROBE_MODIFY_BYPASS",
    "PROBE_REDUCE_JUNCTION",
    "PROBE_RESTORE_ANCHOR",
    "PROBE_RESTORE_PRIMARY",
    "SCENARIO_MODIFY_BYPASS",
    "SCENARIO_REDUCE_JUNCTION",
    "SCENARIO_RESTORE_ANCHOR",
    "SCENARIO_RESTORE_PRIMARY",
    "TERMINAL_CHANNEL",
    "build_case",
    "build_counterfactual_snapshot",
    "build_probe_adapter",
    "build_probe_registry",
    "candidate_definitions",
    "extract_counterfactual_metrics",
    "rank_counterfactuals",
    "reconstruct_node_star",
    "scenario_lookup",
    "solve_candidate",
]


