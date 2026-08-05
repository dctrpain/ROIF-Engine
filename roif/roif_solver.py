"""
ROIF Engine
Decision Solver and Pipeline Orchestrator

Pipeline:
    ROIFSystem
    -> CapacityTensor
    -> CascadeTrajectory
    -> D_origin / D_fast / D_root / Node*
    -> Counterfactual scenarios
    -> Safety gates
    -> ROIFSolution

The solver coordinates existing modules and does not replace them. It is
domain-independent and does not authorize real-world interventions.

Author:
    Architect (Dctr Pain)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

from .roif_entities import ROIFSystem
from .roif_influence import InfluenceContext
from .roif_materials import MaterialModel
from .roif_tensor import (
    CapacityTensor,
    TensorBuildConfig,
    build_capacity_tensor,
)
from .roif_cascade import (
    CascadeConfig,
    CascadeTrajectory,
    run_cascade,
)
from .roif_root_detector import (
    InterventionPolicy,
    ROIFRootResult,
    RootDetectorConfig,
    detect_roif_roles,
)
from .roif_counterfactual import (
    CounterfactualAction,
    CounterfactualBatchResult,
    CounterfactualConfig,
    CounterfactualResult,
    CounterfactualScenario,
    evaluate_counterfactual_batch,
    generate_single_channel_scenarios,
)


class ROIFSolverError(ValueError):
    """Raised when the solver cannot complete safely."""


class SolverStatus(str, Enum):
    SOLVED = "solved"
    NO_SAFE_SOLUTION = "no_safe_solution"
    OBSERVATION_ONLY = "observation_only"
    VETOED = "vetoed"


class SolverMode(str, Enum):
    ANALYSIS_ONLY = "analysis_only"
    RECOMMENDATION = "recommendation"
    STRICT_SAFETY = "strict_safety"


class ScenarioGenerationMode(str, Enum):
    PROVIDED_ONLY = "provided_only"
    NODE_STAR_ONLY = "node_star_only"
    ALL_CHANNELS = "all_channels"
    ROOT_AND_NODE_STAR = "root_and_node_star"
    PROVIDED_PLUS_GENERATED = "provided_plus_generated"


class SelectionPolicy(str, Enum):
    BEST_RANKED = "best_ranked"
    BEST_SAFE = "best_safe"
    PARETO_SAFE = "pareto_safe"
    MAXIMUM_REDUCTION = "maximum_reduction"
    MINIMUM_RISK = "minimum_risk"


class VetoReason(str, Enum):
    NONE = "none"
    NON_FONIT_SCOPE = "non_fonit_scope"
    CASCADE_RISK = "cascade_risk"
    COLLATERAL_EFFECT = "collateral_effect"
    UNCERTAINTY = "uncertainty"
    IRREVERSIBILITY = "irreversibility"
    STRICT_POLICY = "strict_policy"
    NO_SAFE_SCENARIO = "no_safe_scenario"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFSolverError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFSolverError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ROIFSolverError(f"{name} must be finite.")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ROIFSolverError(f"{name} must be within [0, 1].")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFSolverError(f"{name} must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFSolverError("metadata must be a mapping.")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class NonFonitGate:
    """Conservative veto for broad or uncontrolled interventions."""

    enabled: bool = True
    scope_risk: float = 0.0
    externality_risk: float = 0.0
    propagation_uncertainty: float = 0.0
    human_authorization: bool = False
    bounded_scope: bool = True
    reversible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "human_authorization",
            "bounded_scope",
            "reversible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ROIFSolverError(f"{name} must be bool.")

        for name in (
            "scope_risk",
            "externality_risk",
            "propagation_uncertainty",
        ):
            object.__setattr__(
                self,
                name,
                _unit(getattr(self, name), name),
            )

        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SolverThresholds:
    minimum_relative_reduction: float = 0.0
    minimum_utility: float = -1e12
    maximum_safety_risk: float = 1.0
    maximum_collateral_effect: float = 1.0
    maximum_uncertainty: float = 1.0
    maximum_irreversibility: float = 0.05
    maximum_scope_risk: float = 0.10
    maximum_externality_risk: float = 0.10
    maximum_propagation_uncertainty: float = 0.20

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_relative_reduction",
            _finite(
                self.minimum_relative_reduction,
                "minimum_relative_reduction",
            ),
        )
        object.__setattr__(
            self,
            "minimum_utility",
            _finite(self.minimum_utility, "minimum_utility"),
        )
        for name in (
            "maximum_safety_risk",
            "maximum_collateral_effect",
            "maximum_uncertainty",
            "maximum_irreversibility",
            "maximum_scope_risk",
            "maximum_externality_risk",
            "maximum_propagation_uncertainty",
        ):
            object.__setattr__(
                self,
                name,
                _unit(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class SolverConfig:
    mode: SolverMode = SolverMode.ANALYSIS_ONLY
    scenario_generation: ScenarioGenerationMode = (
        ScenarioGenerationMode.ALL_CHANNELS
    )
    selection_policy: SelectionPolicy = SelectionPolicy.BEST_SAFE

    generated_action: CounterfactualAction = (
        CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
    )
    generated_magnitude: float = 1.0

    tensor_config: TensorBuildConfig = field(
        default_factory=TensorBuildConfig
    )
    cascade_config: CascadeConfig = field(
        default_factory=CascadeConfig
    )
    root_config: RootDetectorConfig = field(
        default_factory=RootDetectorConfig
    )
    counterfactual_config: CounterfactualConfig = field(
        default_factory=CounterfactualConfig
    )
    thresholds: SolverThresholds = field(
        default_factory=SolverThresholds
    )
    non_fonit_gate: NonFonitGate = field(
        default_factory=NonFonitGate
    )

    require_safe_solution: bool = True
    require_positive_reduction: bool = False
    include_all_ranked_alternatives: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        enum_checks = (
            ("mode", self.mode, SolverMode),
            (
                "scenario_generation",
                self.scenario_generation,
                ScenarioGenerationMode,
            ),
            (
                "selection_policy",
                self.selection_policy,
                SelectionPolicy,
            ),
            (
                "generated_action",
                self.generated_action,
                CounterfactualAction,
            ),
        )
        for name, value, enum_type in enum_checks:
            if not isinstance(value, enum_type):
                raise ROIFSolverError(
                    f"{name} must be {enum_type.__name__}."
                )

        object.__setattr__(
            self,
            "generated_magnitude",
            _unit(
                self.generated_magnitude,
                "generated_magnitude",
            ),
        )

        type_checks = (
            ("tensor_config", self.tensor_config, TensorBuildConfig),
            ("cascade_config", self.cascade_config, CascadeConfig),
            ("root_config", self.root_config, RootDetectorConfig),
            (
                "counterfactual_config",
                self.counterfactual_config,
                CounterfactualConfig,
            ),
            ("thresholds", self.thresholds, SolverThresholds),
            ("non_fonit_gate", self.non_fonit_gate, NonFonitGate),
        )
        for name, value, expected_type in type_checks:
            if not isinstance(value, expected_type):
                raise ROIFSolverError(
                    f"{name} must be {expected_type.__name__}."
                )

        for name in (
            "require_safe_solution",
            "require_positive_reduction",
            "include_all_ranked_alternatives",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ROIFSolverError(f"{name} must be bool.")

        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SolverEvidence:
    evidence_id: str
    category: str
    value: float
    threshold: float | None
    passed: bool
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _text(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(
            self,
            "category",
            _text(self.category, "category"),
        )
        object.__setattr__(self, "value", _finite(self.value, "value"))
        if self.threshold is not None:
            object.__setattr__(
                self,
                "threshold",
                _finite(self.threshold, "threshold"),
            )
        if not isinstance(self.passed, bool):
            raise ROIFSolverError("passed must be bool.")
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SolverDecision:
    scenario_id: str | None
    node_ids: tuple[str, ...]
    action_ids: tuple[str, ...]

    expected_relative_reduction: float
    expected_utility: float
    expected_spectral_gain: float
    expected_cost: float
    expected_collateral_effect: float
    expected_safety_risk: float
    expected_uncertainty: float
    expected_irreversibility: float

    safe: bool
    executable: bool
    veto_reason: VetoReason

    def __post_init__(self) -> None:
        if self.scenario_id is not None:
            object.__setattr__(
                self,
                "scenario_id",
                _text(self.scenario_id, "scenario_id"),
            )
        object.__setattr__(
            self,
            "node_ids",
            tuple(_text(item, "node_id") for item in self.node_ids),
        )
        object.__setattr__(
            self,
            "action_ids",
            tuple(_text(item, "action_id") for item in self.action_ids),
        )
        for name in (
            "expected_relative_reduction",
            "expected_utility",
            "expected_spectral_gain",
            "expected_cost",
            "expected_collateral_effect",
            "expected_safety_risk",
            "expected_uncertainty",
            "expected_irreversibility",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        if not isinstance(self.safe, bool):
            raise ROIFSolverError("safe must be bool.")
        if not isinstance(self.executable, bool):
            raise ROIFSolverError("executable must be bool.")
        if not isinstance(self.veto_reason, VetoReason):
            raise ROIFSolverError(
                "veto_reason must be VetoReason."
            )


@dataclass(frozen=True, slots=True)
class ROIFSolution:
    status: SolverStatus
    decision: SolverDecision
    tensor: CapacityTensor
    trajectory: CascadeTrajectory
    root_result: ROIFRootResult
    counterfactual_batch: CounterfactualBatchResult
    alternatives: tuple[CounterfactualResult, ...]
    evidence: tuple[SolverEvidence, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, SolverStatus):
            raise ROIFSolverError("status must be SolverStatus.")
        if not isinstance(self.decision, SolverDecision):
            raise ROIFSolverError(
                "decision must be SolverDecision."
            )
        if not isinstance(self.tensor, CapacityTensor):
            raise ROIFSolverError("tensor must be CapacityTensor.")
        if not isinstance(self.trajectory, CascadeTrajectory):
            raise ROIFSolverError(
                "trajectory must be CascadeTrajectory."
            )
        if not isinstance(self.root_result, ROIFRootResult):
            raise ROIFSolverError(
                "root_result must be ROIFRootResult."
            )
        if not isinstance(
            self.counterfactual_batch,
            CounterfactualBatchResult,
        ):
            raise ROIFSolverError(
                "counterfactual_batch must be CounterfactualBatchResult."
            )
        object.__setattr__(
            self,
            "alternatives",
            tuple(self.alternatives),
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


def validate_solver_inputs(
    system: ROIFSystem,
    tensor: CapacityTensor | None,
    trajectory: CascadeTrajectory | None,
) -> None:
    if not isinstance(system, ROIFSystem):
        raise TypeError("system must be ROIFSystem.")

    channel_ids = tuple(system.channel_ids)
    if not channel_ids:
        raise ROIFSolverError(
            "system must contain at least one channel."
        )

    if tensor is not None:
        if not isinstance(tensor, CapacityTensor):
            raise TypeError("tensor must be CapacityTensor or None.")
        if tensor.channel_ids != channel_ids:
            raise ROIFSolverError(
                "tensor channel order must match system.channel_ids."
            )

    if trajectory is not None:
        if not isinstance(trajectory, CascadeTrajectory):
            raise TypeError(
                "trajectory must be CascadeTrajectory or None."
            )
        if trajectory.channel_ids != channel_ids:
            raise ROIFSolverError(
                "trajectory channel order must match system.channel_ids."
            )


def build_solver_tensor(
    system: ROIFSystem,
    config: SolverConfig,
    *,
    context: InfluenceContext | None,
    materials: Mapping[str, MaterialModel],
) -> CapacityTensor:
    return build_capacity_tensor(
        system,
        context=context,
        materials=materials,
        config=config.tensor_config,
    )


def build_solver_trajectory(
    system: ROIFSystem,
    config: SolverConfig,
    *,
    initial_state: Sequence[float] | Mapping[str, float] | None,
    context: InfluenceContext | None,
    materials: Mapping[str, MaterialModel],
) -> CascadeTrajectory:
    return run_cascade(
        system,
        initial_state=initial_state,
        context=context,
        materials=materials,
        tensor_config=config.tensor_config,
        cascade_config=config.cascade_config,
    )


def _deduplicate_scenarios(
    scenarios: Sequence[CounterfactualScenario],
) -> tuple[CounterfactualScenario, ...]:
    result = []
    seen: set[str] = set()

    for scenario in scenarios:
        if scenario.scenario_id in seen:
            continue
        seen.add(scenario.scenario_id)
        result.append(scenario)

    return tuple(result)


def generate_solver_scenarios(
    system: ROIFSystem,
    root_result: ROIFRootResult,
    provided: Sequence[CounterfactualScenario],
    config: SolverConfig,
) -> tuple[CounterfactualScenario, ...]:
    provided = tuple(provided)

    if (
        config.scenario_generation
        is ScenarioGenerationMode.PROVIDED_ONLY
    ):
        if not provided:
            raise ROIFSolverError(
                "PROVIDED_ONLY requires scenarios."
            )
        return _deduplicate_scenarios(provided)

    if (
        config.scenario_generation
        is ScenarioGenerationMode.NODE_STAR_ONLY
    ):
        generated_ids = (root_result.node_star,)

    elif (
        config.scenario_generation
        is ScenarioGenerationMode.ALL_CHANNELS
    ):
        generated_ids = tuple(system.channel_ids)

    elif (
        config.scenario_generation
        is ScenarioGenerationMode.ROOT_AND_NODE_STAR
    ):
        generated_ids = tuple(
            dict.fromkeys(
                (
                    root_result.d_root,
                    root_result.node_star,
                )
            )
        )

    elif (
        config.scenario_generation
        is ScenarioGenerationMode.PROVIDED_PLUS_GENERATED
    ):
        generated_ids = tuple(system.channel_ids)

    else:
        raise ROIFSolverError(
            "unsupported ScenarioGenerationMode."
        )

    generated = generate_single_channel_scenarios(
        generated_ids,
        action=config.generated_action,
        magnitude=config.generated_magnitude,
    )

    if (
        config.scenario_generation
        is ScenarioGenerationMode.PROVIDED_PLUS_GENERATED
    ):
        return _deduplicate_scenarios(
            tuple(provided) + tuple(generated)
        )

    return _deduplicate_scenarios(generated)


def scenario_passes_thresholds(
    result: CounterfactualResult,
    thresholds: SolverThresholds,
    *,
    require_positive_reduction: bool,
) -> bool:
    metrics = result.metrics

    if not metrics.safe:
        return False
    if (
        metrics.relative_reduction
        < thresholds.minimum_relative_reduction
    ):
        return False
    if require_positive_reduction and metrics.relative_reduction <= 0.0:
        return False
    if metrics.utility < thresholds.minimum_utility:
        return False
    if metrics.safety_risk > thresholds.maximum_safety_risk:
        return False
    if (
        metrics.collateral_effect
        > thresholds.maximum_collateral_effect
    ):
        return False
    if metrics.uncertainty > thresholds.maximum_uncertainty:
        return False
    if (
        metrics.irreversibility
        > thresholds.maximum_irreversibility
    ):
        return False

    return True


def safe_candidates(
    batch: CounterfactualBatchResult,
    config: SolverConfig,
) -> tuple[CounterfactualResult, ...]:
    return tuple(
        result
        for result in batch.results
        if scenario_passes_thresholds(
            result,
            config.thresholds,
            require_positive_reduction=(
                config.require_positive_reduction
            ),
        )
    )


def select_solver_result(
    batch: CounterfactualBatchResult,
    config: SolverConfig,
) -> CounterfactualResult | None:
    accepted = safe_candidates(batch, config)

    if config.selection_policy is SelectionPolicy.BEST_RANKED:
        winner = batch.winner
        if config.require_safe_solution and winner not in accepted:
            return accepted[0] if accepted else None
        return winner

    if config.selection_policy is SelectionPolicy.BEST_SAFE:
        return accepted[0] if accepted else None

    if config.selection_policy is SelectionPolicy.PARETO_SAFE:
        front = set(batch.pareto_front_ids)
        for result in accepted:
            if result.scenario.scenario_id in front:
                return result
        return accepted[0] if accepted else None

    if config.selection_policy is SelectionPolicy.MAXIMUM_REDUCTION:
        if not accepted:
            return None
        return max(
            accepted,
            key=lambda item: (
                item.metrics.relative_reduction,
                item.metrics.utility,
                -item.metrics.safety_risk,
                item.scenario.scenario_id,
            ),
        )

    if config.selection_policy is SelectionPolicy.MINIMUM_RISK:
        if not accepted:
            return None
        return min(
            accepted,
            key=lambda item: (
                item.metrics.safety_risk,
                item.metrics.collateral_effect,
                item.metrics.uncertainty,
                -item.metrics.relative_reduction,
                item.scenario.scenario_id,
            ),
        )

    raise ROIFSolverError("unsupported SelectionPolicy.")


def evaluate_non_fonit_gate(
    gate: NonFonitGate,
    thresholds: SolverThresholds,
) -> tuple[bool, VetoReason, tuple[SolverEvidence, ...]]:
    if not gate.enabled:
        return True, VetoReason.NONE, ()

    evidence = (
        SolverEvidence(
            evidence_id="non_fonit:scope",
            category="non_fonit",
            value=gate.scope_risk,
            threshold=thresholds.maximum_scope_risk,
            passed=gate.scope_risk <= thresholds.maximum_scope_risk,
            description="Scope risk remains bounded.",
        ),
        SolverEvidence(
            evidence_id="non_fonit:externality",
            category="non_fonit",
            value=gate.externality_risk,
            threshold=thresholds.maximum_externality_risk,
            passed=(
                gate.externality_risk
                <= thresholds.maximum_externality_risk
            ),
            description="Externality risk remains bounded.",
        ),
        SolverEvidence(
            evidence_id="non_fonit:propagation",
            category="non_fonit",
            value=gate.propagation_uncertainty,
            threshold=(
                thresholds.maximum_propagation_uncertainty
            ),
            passed=(
                gate.propagation_uncertainty
                <= thresholds.maximum_propagation_uncertainty
            ),
            description=(
                "Propagation uncertainty remains bounded."
            ),
        ),
    )

    if not gate.bounded_scope:
        return False, VetoReason.NON_FONIT_SCOPE, evidence
    if gate.scope_risk > thresholds.maximum_scope_risk:
        return False, VetoReason.NON_FONIT_SCOPE, evidence
    if gate.externality_risk > thresholds.maximum_externality_risk:
        return False, VetoReason.NON_FONIT_SCOPE, evidence
    if (
        gate.propagation_uncertainty
        > thresholds.maximum_propagation_uncertainty
    ):
        return False, VetoReason.NON_FONIT_SCOPE, evidence
    if not gate.reversible and not gate.human_authorization:
        return False, VetoReason.IRREVERSIBILITY, evidence

    return True, VetoReason.NONE, evidence


def scenario_veto_reason(
    result: CounterfactualResult,
    config: SolverConfig,
) -> VetoReason:
    metrics = result.metrics
    thresholds = config.thresholds

    if not metrics.safe:
        if (
            config.counterfactual_config.policy
            is InterventionPolicy.STRICT_LINEAR
        ):
            return VetoReason.STRICT_POLICY
        return VetoReason.CASCADE_RISK
    if metrics.safety_risk > thresholds.maximum_safety_risk:
        return VetoReason.CASCADE_RISK
    if (
        metrics.collateral_effect
        > thresholds.maximum_collateral_effect
    ):
        return VetoReason.COLLATERAL_EFFECT
    if metrics.uncertainty > thresholds.maximum_uncertainty:
        return VetoReason.UNCERTAINTY
    if (
        metrics.irreversibility
        > thresholds.maximum_irreversibility
    ):
        return VetoReason.IRREVERSIBILITY

    return VetoReason.NONE


def build_solver_evidence(
    selected: CounterfactualResult | None,
    config: SolverConfig,
    gate_evidence: Sequence[SolverEvidence],
) -> tuple[SolverEvidence, ...]:
    evidence = list(gate_evidence)

    if selected is None:
        evidence.append(
            SolverEvidence(
                evidence_id="selection:none",
                category="selection",
                value=0.0,
                threshold=None,
                passed=False,
                description=(
                    "No scenario satisfied solver constraints."
                ),
            )
        )
        return tuple(evidence)

    metrics = selected.metrics
    thresholds = config.thresholds

    evidence.extend(
        (
            SolverEvidence(
                evidence_id="scenario:reduction",
                category="effect",
                value=metrics.relative_reduction,
                threshold=thresholds.minimum_relative_reduction,
                passed=(
                    metrics.relative_reduction
                    >= thresholds.minimum_relative_reduction
                ),
                description="Expected relative cascade reduction.",
            ),
            SolverEvidence(
                evidence_id="scenario:utility",
                category="effect",
                value=metrics.utility,
                threshold=thresholds.minimum_utility,
                passed=metrics.utility >= thresholds.minimum_utility,
                description="Expected balanced utility.",
            ),
            SolverEvidence(
                evidence_id="scenario:risk",
                category="safety",
                value=metrics.safety_risk,
                threshold=thresholds.maximum_safety_risk,
                passed=(
                    metrics.safety_risk
                    <= thresholds.maximum_safety_risk
                ),
                description="Expected safety risk.",
            ),
            SolverEvidence(
                evidence_id="scenario:collateral",
                category="safety",
                value=metrics.collateral_effect,
                threshold=thresholds.maximum_collateral_effect,
                passed=(
                    metrics.collateral_effect
                    <= thresholds.maximum_collateral_effect
                ),
                description="Expected collateral effect.",
            ),
            SolverEvidence(
                evidence_id="scenario:uncertainty",
                category="safety",
                value=metrics.uncertainty,
                threshold=thresholds.maximum_uncertainty,
                passed=(
                    metrics.uncertainty
                    <= thresholds.maximum_uncertainty
                ),
                description="Expected uncertainty.",
            ),
            SolverEvidence(
                evidence_id="scenario:irreversibility",
                category="safety",
                value=metrics.irreversibility,
                threshold=thresholds.maximum_irreversibility,
                passed=(
                    metrics.irreversibility
                    <= thresholds.maximum_irreversibility
                ),
                description="Expected irreversibility.",
            ),
        )
    )

    return tuple(evidence)


def empty_solver_decision(
    veto_reason: VetoReason,
) -> SolverDecision:
    return SolverDecision(
        scenario_id=None,
        node_ids=(),
        action_ids=(),
        expected_relative_reduction=0.0,
        expected_utility=0.0,
        expected_spectral_gain=0.0,
        expected_cost=0.0,
        expected_collateral_effect=0.0,
        expected_safety_risk=0.0,
        expected_uncertainty=0.0,
        expected_irreversibility=0.0,
        safe=False,
        executable=False,
        veto_reason=veto_reason,
    )


def decision_from_result(
    result: CounterfactualResult,
    config: SolverConfig,
    *,
    gate_passed: bool,
    gate_reason: VetoReason,
) -> SolverDecision:
    metrics = result.metrics
    scenario_reason = scenario_veto_reason(result, config)

    veto_reason = gate_reason if not gate_passed else scenario_reason
    safe = (
        gate_passed
        and scenario_reason is VetoReason.NONE
        and metrics.safe
    )

    executable = (
        safe
        and config.mode is not SolverMode.ANALYSIS_ONLY
        and config.non_fonit_gate.human_authorization
    )

    return SolverDecision(
        scenario_id=result.scenario.scenario_id,
        node_ids=tuple(
            item.channel_id
            for item in result.scenario.interventions
        ),
        action_ids=tuple(
            item.action.value
            for item in result.scenario.interventions
        ),
        expected_relative_reduction=metrics.relative_reduction,
        expected_utility=metrics.utility,
        expected_spectral_gain=metrics.spectral_gain,
        expected_cost=metrics.cost,
        expected_collateral_effect=metrics.collateral_effect,
        expected_safety_risk=metrics.safety_risk,
        expected_uncertainty=metrics.uncertainty,
        expected_irreversibility=metrics.irreversibility,
        safe=safe,
        executable=executable,
        veto_reason=veto_reason,
    )


def _status_from_decision(
    decision: SolverDecision,
    config: SolverConfig,
) -> SolverStatus:
    if decision.scenario_id is None:
        return SolverStatus.NO_SAFE_SOLUTION
    if decision.veto_reason is not VetoReason.NONE:
        return SolverStatus.VETOED
    if config.mode is SolverMode.ANALYSIS_ONLY:
        return SolverStatus.OBSERVATION_ONLY
    if decision.safe:
        return SolverStatus.SOLVED
    return SolverStatus.NO_SAFE_SOLUTION


def solve_roif(
    system: ROIFSystem,
    *,
    initial_state: Sequence[float] | Mapping[str, float] | None = None,
    scenarios: Sequence[CounterfactualScenario] = (),
    config: SolverConfig | None = None,
    tensor: CapacityTensor | None = None,
    trajectory: CascadeTrajectory | None = None,
    context: InfluenceContext | None = None,
    materials: Mapping[str, MaterialModel] | None = None,
) -> ROIFSolution:
    """Run the complete deterministic ROIF decision pipeline."""

    config = config or SolverConfig()
    materials = materials or {}

    validate_solver_inputs(system, tensor, trajectory)

    if tensor is None:
        tensor = build_solver_tensor(
            system,
            config,
            context=context,
            materials=materials,
        )

    if trajectory is None:
        trajectory = build_solver_trajectory(
            system,
            config,
            initial_state=initial_state,
            context=context,
            materials=materials,
        )

    root_result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        config.root_config,
    )

    resolved_scenarios = generate_solver_scenarios(
        system,
        root_result,
        scenarios,
        config,
    )

    batch = evaluate_counterfactual_batch(
        system,
        tensor,
        trajectory,
        resolved_scenarios,
        config.counterfactual_config,
        tensor_config=config.tensor_config,
        cascade_config=config.cascade_config,
        context=context,
        materials=materials,
    )

    selected = select_solver_result(batch, config)

    gate_passed, gate_reason, gate_evidence = (
        evaluate_non_fonit_gate(
            config.non_fonit_gate,
            config.thresholds,
        )
    )

    if selected is None:
        decision = empty_solver_decision(
            VetoReason.NO_SAFE_SCENARIO
        )
    else:
        decision = decision_from_result(
            selected,
            config,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
        )

    status = _status_from_decision(decision, config)
    evidence = build_solver_evidence(
        selected,
        config,
        gate_evidence,
    )

    alternatives = (
        batch.results
        if config.include_all_ranked_alternatives
        else (() if selected is None else (selected,))
    )

    return ROIFSolution(
        status=status,
        decision=decision,
        tensor=tensor,
        trajectory=trajectory,
        root_result=root_result,
        counterfactual_batch=batch,
        alternatives=tuple(alternatives),
        evidence=evidence,
        metadata={
            "system_id": system.system_id,
            "solver_mode": config.mode.value,
            "scenario_generation": (
                config.scenario_generation.value
            ),
            "selection_policy": (
                config.selection_policy.value
            ),
            "scenario_count": len(batch.results),
            "d_origin": root_result.d_origin,
            "d_fast": root_result.d_fast,
            "d_root": root_result.d_root,
            "node_star": root_result.node_star,
            "human_authorization": (
                config.non_fonit_gate.human_authorization
            ),
        },
    )


def solve_analysis_only(
    system: ROIFSystem,
    **kwargs: Any,
) -> ROIFSolution:
    config = kwargs.pop("config", None) or SolverConfig()
    config = replace(
        config,
        mode=SolverMode.ANALYSIS_ONLY,
    )
    return solve_roif(
        system,
        config=config,
        **kwargs,
    )


def solution_summary(
    solution: ROIFSolution,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "status": solution.status.value,
            "scenario_id": solution.decision.scenario_id,
            "safe": solution.decision.safe,
            "executable": solution.decision.executable,
            "veto_reason": (
                solution.decision.veto_reason.value
            ),
            "d_origin": solution.root_result.d_origin,
            "d_fast": solution.root_result.d_fast,
            "d_root": solution.root_result.d_root,
            "node_star": solution.root_result.node_star,
            "expected_relative_reduction": (
                solution.decision.expected_relative_reduction
            ),
            "expected_utility": (
                solution.decision.expected_utility
            ),
            "expected_spectral_gain": (
                solution.decision.expected_spectral_gain
            ),
            "expected_cost": (
                solution.decision.expected_cost
            ),
            "expected_collateral_effect": (
                solution.decision.expected_collateral_effect
            ),
            "alternative_count": len(solution.alternatives),
            "evidence_count": len(solution.evidence),
        }
    )


def solution_is_actionable(
    solution: ROIFSolution,
) -> bool:
    return (
        solution.status is SolverStatus.SOLVED
        and solution.decision.safe
        and solution.decision.executable
        and solution.decision.veto_reason is VetoReason.NONE
    )


__all__ = [
    "NonFonitGate",
    "ROIFSolution",
    "ROIFSolverError",
    "ScenarioGenerationMode",
    "SelectionPolicy",
    "SolverConfig",
    "SolverDecision",
    "SolverEvidence",
    "SolverMode",
    "SolverStatus",
    "SolverThresholds",
    "VetoReason",
    "build_solver_evidence",
    "build_solver_tensor",
    "build_solver_trajectory",
    "decision_from_result",
    "empty_solver_decision",
    "evaluate_non_fonit_gate",
    "generate_solver_scenarios",
    "safe_candidates",
    "scenario_passes_thresholds",
    "scenario_veto_reason",
    "select_solver_result",
    "solution_is_actionable",
    "solution_summary",
    "solve_analysis_only",
    "solve_roif",
    "validate_solver_inputs",
]
