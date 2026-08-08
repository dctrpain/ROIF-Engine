"""
ROIF Validation Suite
Mechanical Scenario 03B:
Symmetric Branching Ambiguity — Indistinguishable Branch Control

Purpose
-------
Scenario 03B is a deliberately ambiguous branching control case.

It asks:

    If a pre-stressed system contains two nearly symmetric outgoing branches
    with comparable gain, reserve, geometry, and downstream effect, will ROIF
    preserve uncertainty instead of inventing a unique causal branch?

Topology
--------

                         /-> symmetric_branch_a --\
    shared_preload_node                         -> common_terminal
                         +-> symmetric_branch_b --/

There is:

- one shared upstream pre-load node;
- two parallel branches;
- near-symmetric mechanical properties;
- one common downstream terminal;
- no feedback;
- no evaluator-side preferred branch in Solver input.

Validation role
---------------
03B is a negative/ambiguity benchmark for Active Probe logic.

The ground-truth contract is intentionally different from 02A:

- both branches are physically admissible;
- neither branch should be privileged from passive structure alone;
- a weak or non-discriminative Probe should be allowed to remain unresolved;
- the system should prefer explicit uncertainty over false certainty.

This is a synthetic mechanical validation model. It carries no biological,
diagnostic, or treatment claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from roif.roif_entities import (
    ActivationAvailability,
    CapacityState,
    EntityKind,
    FunctionalChannel,
    FunctionalRole,
    GeometryState,
    InfluenceOperator,
    InfluencePlane,
    OperatorKind,
    PlaneKind,
    ROIFSystem,
    StructuralEntity,
    TemporalMode,
    Vector3,
)
from roif.roif_tensor import (
    SelfCouplingMode,
    TensorBuildConfig,
)
from roif.roif_cascade import (
    CascadeConfig,
    CascadeDirection,
    CascadeUpdateMode,
)
from roif.roif_root_detector import (
    RootDetectorConfig,
)
from roif.roif_counterfactual import (
    CounterfactualAction,
    CounterfactualConfig,
    CounterfactualExecutionMode,
    CounterfactualIntervention,
    CounterfactualObjective,
    CounterfactualScenario,
)
from roif.roif_solver import (
    NonFonitGate,
    ScenarioGenerationMode,
    SelectionPolicy,
    SolverConfig,
    SolverMode,
)


# =============================================================================
# Public identifiers
# =============================================================================


PRELOAD_CHANNEL = "shared_preload_node"
BRANCH_A_CHANNEL = "symmetric_branch_a"
BRANCH_B_CHANNEL = "symmetric_branch_b"
TERMINAL_CHANNEL = "common_terminal"

PRELOAD_TO_A = "preload_to_branch_a"
PRELOAD_TO_B = "preload_to_branch_b"
A_TO_TERMINAL = "branch_a_to_terminal"
B_TO_TERMINAL = "branch_b_to_terminal"

SCENARIO_REMOVE_PRELOAD = "remove_shared_preload"
SCENARIO_REDUCE_BRANCH_A = "reduce_branch_a_load"
SCENARIO_REDUCE_BRANCH_B = "reduce_branch_b_load"
SCENARIO_REDUCE_TERMINAL = "reduce_common_terminal_load"


# =============================================================================
# Immutable validation value objects
# =============================================================================


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class SymmetricBranchingValidationLabels:
    """
    Evaluator-side labels.

    Scenario 03B intentionally does NOT declare one branch as the uniquely
    correct passive continuation.

    branch_truth
        Tuple of equally admissible branch identifiers.

    ambiguity_expected
        Whether the passive local branch decision is expected to remain
        unresolved without discriminative evidence.
    """

    d_origin: str
    branch_truth: tuple[str, str]
    downstream_terminal: str
    ambiguity_expected: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "branch_truth",
            tuple(self.branch_truth),
        )

    def as_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "d_origin": self.d_origin,
                "branch_truth": self.branch_truth,
                "downstream_terminal": self.downstream_terminal,
                "ambiguity_expected": self.ambiguity_expected,
            }
        )


@dataclass(frozen=True, slots=True)
class SymmetricBranchingValidationCase:
    case_id: str
    title: str
    pathological_system: ROIFSystem
    reference_system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: SymmetricBranchingValidationLabels
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_state",
            _readonly_mapping(self.initial_state),
        )
        object.__setattr__(
            self,
            "scenarios",
            tuple(self.scenarios),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(self.pathological_system.channel_ids)

    def scenario(
        self,
        scenario_id: str,
    ) -> CounterfactualScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(scenario_id)


# =============================================================================
# ROIF entity builders
# =============================================================================


def _channel(
    channel_id: str,
    *,
    entity_id: str,
    name: str,
    capacity: float,
    load: float,
    activation: float,
    mobility: float,
    history_factor: float = 1.0,
    role: FunctionalRole = FunctionalRole.TRANSMITTER,
    direction: Vector3 = Vector3(1.0, 0.0, 0.0),
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id,
        name=name,
        role=role,
        direction=direction,
        capacity_state=CapacityState(
            capacity=capacity,
            load=load,
        ),
        activation=ActivationAvailability(
            command=activation,
        ),
        geometry=GeometryState(
            mobility=mobility,
        ),
        history_factor=history_factor,
    )


def _entity(
    entity_id: str,
    *,
    name: str,
    channel: FunctionalChannel,
) -> StructuralEntity:
    return StructuralEntity(
        entity_id=entity_id,
        name=name,
        kind=EntityKind.GENERIC,
        channels=(channel,),
    )


def _operator(
    operator_id: str,
    *,
    source: str,
    target: str,
    gain: float,
) -> InfluenceOperator:
    return InfluenceOperator(
        operator_id=operator_id,
        name=operator_id,
        plane_id="mechanical",
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=(source,),
        target_ids=(target,),
        gain=gain,
    )


# =============================================================================
# Mechanical graph
# =============================================================================


def _build_system(
    *,
    pathological: bool,
) -> ROIFSystem:
    """
    Build a nearly symmetric two-branch pre-stressed graph.

    The pathological model contains one upstream disturbance feeding two
    deliberately near-equivalent branches. Tiny numerical asymmetry is kept
    below any intended validation-level discriminatory threshold.
    """

    if pathological:
        values = {
            "preload_capacity": 1.50,
            "preload_load": 1.20,
            "preload_activation": 1.00,
            "preload_mobility": 1.00,
            "preload_history": 1.00,

            "a_capacity": 2.40,
            "a_load": 1.44,
            "a_activation": 0.86,
            "a_mobility": 0.80,
            "a_history": 0.90,

            "b_capacity": 2.42,
            "b_load": 1.45,
            "b_activation": 0.855,
            "b_mobility": 0.805,
            "b_history": 0.895,

            "terminal_capacity": 3.40,
            "terminal_load": 2.10,
            "terminal_activation": 0.84,
            "terminal_mobility": 0.76,
            "terminal_history": 0.86,
        }
        gains = {
            "preload_a": 0.900,
            "preload_b": 0.895,
            "a_terminal": 0.820,
            "b_terminal": 0.818,
        }
    else:
        values = {
            "preload_capacity": 1.50,
            "preload_load": 0.15,
            "preload_activation": 1.00,
            "preload_mobility": 1.00,
            "preload_history": 1.00,

            "a_capacity": 3.00,
            "a_load": 0.75,
            "a_activation": 0.96,
            "a_mobility": 0.95,
            "a_history": 1.00,

            "b_capacity": 3.00,
            "b_load": 0.75,
            "b_activation": 0.96,
            "b_mobility": 0.95,
            "b_history": 1.00,

            "terminal_capacity": 3.80,
            "terminal_load": 0.80,
            "terminal_activation": 0.96,
            "terminal_mobility": 0.95,
            "terminal_history": 1.00,
        }
        gains = {
            "preload_a": 0.90,
            "preload_b": 0.90,
            "a_terminal": 0.82,
            "b_terminal": 0.82,
        }

    preload = _channel(
        PRELOAD_CHANNEL,
        entity_id="shared_preload",
        name="Shared preload disturbance",
        capacity=values["preload_capacity"],
        load=values["preload_load"],
        activation=values["preload_activation"],
        mobility=values["preload_mobility"],
        history_factor=values["preload_history"],
        role=FunctionalRole.MODULATOR,
        direction=Vector3(1.0, 0.0, 0.0),
    )

    branch_a = _channel(
        BRANCH_A_CHANNEL,
        entity_id="branch_a",
        name="Symmetric branch A",
        capacity=values["a_capacity"],
        load=values["a_load"],
        activation=values["a_activation"],
        mobility=values["a_mobility"],
        history_factor=values["a_history"],
        role=FunctionalRole.TRANSMITTER,
        direction=Vector3(1.0, 0.05, 0.0),
    )

    branch_b = _channel(
        BRANCH_B_CHANNEL,
        entity_id="branch_b",
        name="Symmetric branch B",
        capacity=values["b_capacity"],
        load=values["b_load"],
        activation=values["b_activation"],
        mobility=values["b_mobility"],
        history_factor=values["b_history"],
        role=FunctionalRole.TRANSMITTER,
        direction=Vector3(1.0, -0.05, 0.0),
    )

    terminal = _channel(
        TERMINAL_CHANNEL,
        entity_id="common_terminal",
        name="Common downstream terminal",
        capacity=values["terminal_capacity"],
        load=values["terminal_load"],
        activation=values["terminal_activation"],
        mobility=values["terminal_mobility"],
        history_factor=values["terminal_history"],
        role=FunctionalRole.SENSOR,
        direction=Vector3(1.0, 0.0, 0.0),
    )

    return ROIFSystem(
        system_id=(
            "symmetric_branching_ambiguity_pathological"
            if pathological
            else "symmetric_branching_ambiguity_reference"
        ),
        name=(
            "Symmetric branching ambiguity mechanical system"
            if pathological
            else "Restored symmetric branching reference"
        ),
        entities=(
            _entity(
                "shared_preload",
                name="Shared preload node",
                channel=preload,
            ),
            _entity(
                "branch_a",
                name="Symmetric branch A",
                channel=branch_a,
            ),
            _entity(
                "branch_b",
                name="Symmetric branch B",
                channel=branch_b,
            ),
            _entity(
                "common_terminal",
                name="Common terminal",
                channel=terminal,
            ),
        ),
        planes=(
            InfluencePlane(
                plane_id="mechanical",
                name="Mechanical load-transfer plane",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.STATIC,
            ),
        ),
        operators=(
            _operator(
                PRELOAD_TO_A,
                source=PRELOAD_CHANNEL,
                target=BRANCH_A_CHANNEL,
                gain=gains["preload_a"],
            ),
            _operator(
                PRELOAD_TO_B,
                source=PRELOAD_CHANNEL,
                target=BRANCH_B_CHANNEL,
                gain=gains["preload_b"],
            ),
            _operator(
                A_TO_TERMINAL,
                source=BRANCH_A_CHANNEL,
                target=TERMINAL_CHANNEL,
                gain=gains["a_terminal"],
            ),
            _operator(
                B_TO_TERMINAL,
                source=BRANCH_B_CHANNEL,
                target=TERMINAL_CHANNEL,
                gain=gains["b_terminal"],
            ),
        ),
        metadata={
            "validation_case": "mechanical_03B",
            "domain": "mechanical",
            "topology": "symmetric_branching_ambiguity",
            "pre_stressed": True,
            "branching": True,
            "feedback": False,
            "near_symmetric_branches": True,
            "state": (
                "pathological"
                if pathological
                else "reference"
            ),
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
        },
    )


# =============================================================================
# Public system factories
# =============================================================================


def build_pathological_system() -> ROIFSystem:
    return _build_system(
        pathological=True,
    )


def build_reference_system() -> ROIFSystem:
    return _build_system(
        pathological=False,
    )


def build_initial_state() -> Mapping[str, float]:
    return MappingProxyType(
        {
            PRELOAD_CHANNEL: 1.0,
            BRANCH_A_CHANNEL: 0.0,
            BRANCH_B_CHANNEL: 0.0,
            TERMINAL_CHANNEL: 0.0,
        }
    )


# =============================================================================
# Tensor / cascade configuration
# =============================================================================


def build_tensor_config() -> TensorBuildConfig:
    return TensorBuildConfig(
        tensor_ceiling=100.0,
        source_reserve_exponent=0.0,
        target_reserve_exponent=0.0,
        availability_exponent=1.0,
        geometry_exponent=1.0,
        material_exponent=0.0,
        history_exponent=1.0,
        state_dependent=True,
        self_coupling_mode=SelfCouplingMode.NONE,
    )


def build_cascade_config() -> CascadeConfig:
    return CascadeConfig(
        steps=10,
        update_mode=CascadeUpdateMode.LEAKY,
        direction=CascadeDirection.GENERIC,
        retention=0.10,
        dissipation=0.10,
        lower_bound=0.0,
        upper_bound=100.0,
        clip_state=False,
        convergence_tolerance=1e-10,
        convergence_patience=12,
        record_transmission_events=True,
        stop_on_failure=False,
    )


# =============================================================================
# Counterfactual scenarios
# =============================================================================


def build_validation_scenarios() -> tuple[
    CounterfactualScenario,
    ...,
]:
    return (
        CounterfactualScenario(
            scenario_id=SCENARIO_REMOVE_PRELOAD,
            name="Remove shared preload influence",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="remove:shared_preload",
                    channel_id=PRELOAD_CHANNEL,
                    action=CounterfactualAction.REMOVE_OUTGOING_INFLUENCE,
                    magnitude=1.0,
                    cost=0.20,
                    safety_risk=0.05,
                    uncertainty=0.10,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_REDUCE_BRANCH_A,
            name="Reduce branch A incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:symmetric_branch_a",
                    channel_id=BRANCH_A_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.30,
                    safety_risk=0.05,
                    uncertainty=0.15,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_REDUCE_BRANCH_B,
            name="Reduce branch B incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:symmetric_branch_b",
                    channel_id=BRANCH_B_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.30,
                    safety_risk=0.05,
                    uncertainty=0.15,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_REDUCE_TERMINAL,
            name="Reduce common terminal incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:common_terminal",
                    channel_id=TERMINAL_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.40,
                    safety_risk=0.08,
                    uncertainty=0.20,
                    irreversibility=0.0,
                ),
            ),
        ),
    )


# =============================================================================
# Solver configuration
# =============================================================================


def build_solver_config() -> SolverConfig:
    return SolverConfig(
        mode=SolverMode.ANALYSIS_ONLY,
        scenario_generation=ScenarioGenerationMode.PROVIDED_ONLY,
        selection_policy=SelectionPolicy.BEST_SAFE,
        generated_action=CounterfactualAction.REMOVE_OUTGOING_INFLUENCE,
        generated_magnitude=1.0,
        tensor_config=build_tensor_config(),
        cascade_config=build_cascade_config(),
        root_config=RootDetectorConfig(),
        counterfactual_config=CounterfactualConfig(
            execution_mode=CounterfactualExecutionMode.FULL_CASCADE,
            objective=CounterfactualObjective.BALANCED_UTILITY,
            steps=10,
            dissipation=0.10,
            retention=0.10,
            mark_dominated=True,
            include_baseline=True,
            deterministic=True,
        ),
        non_fonit_gate=NonFonitGate(
            enabled=True,
            scope_risk=0.0,
            externality_risk=0.0,
            propagation_uncertainty=0.10,
            human_authorization=False,
            bounded_scope=True,
            reversible=True,
        ),
        require_safe_solution=True,
        require_positive_reduction=True,
        include_all_ranked_alternatives=True,
        metadata={
            "validation_case": "mechanical_03B",
            "domain": "mechanical",
            "navigator_only": True,
        },
    )


# =============================================================================
# External validation contract
# =============================================================================


def build_expected_labels() -> SymmetricBranchingValidationLabels:
    """
    Evaluator-side ambiguity hypothesis.

    No single branch is declared the uniquely correct passive continuation.
    """

    return SymmetricBranchingValidationLabels(
        d_origin=PRELOAD_CHANNEL,
        branch_truth=(
            BRANCH_A_CHANNEL,
            BRANCH_B_CHANNEL,
        ),
        downstream_terminal=TERMINAL_CHANNEL,
        ambiguity_expected=True,
    )


# =============================================================================
# Complete factory
# =============================================================================


def build_symmetric_branching_ambiguity_case() -> (
    SymmetricBranchingValidationCase
):
    return SymmetricBranchingValidationCase(
        case_id="mechanical_03B_symmetric_branching_ambiguity",
        title="Symmetric branching ambiguity mechanical control",
        pathological_system=build_pathological_system(),
        reference_system=build_reference_system(),
        initial_state=build_initial_state(),
        tensor_config=build_tensor_config(),
        cascade_config=build_cascade_config(),
        solver_config=build_solver_config(),
        scenarios=build_validation_scenarios(),
        expected=build_expected_labels(),
        metadata={
            "domain": "mechanical",
            "validation_family": "mechanical_03_branch_controls",
            "validation_stage": "03B",
            "evidence_type": "controlled synthetic mechanical ambiguity",
            "topology": "symmetric_branching_ambiguity",
            "pre_stressed": True,
            "branching": True,
            "feedback": False,
            "near_symmetric_branches": True,
            "expected_unique_passive_branch": False,
            "expected_probe_discrimination": "insufficient_or_unresolved",
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


__all__ = [
    "A_TO_TERMINAL",
    "B_TO_TERMINAL",
    "BRANCH_A_CHANNEL",
    "BRANCH_B_CHANNEL",
    "PRELOAD_CHANNEL",
    "PRELOAD_TO_A",
    "PRELOAD_TO_B",
    "SCENARIO_REDUCE_BRANCH_A",
    "SCENARIO_REDUCE_BRANCH_B",
    "SCENARIO_REDUCE_TERMINAL",
    "SCENARIO_REMOVE_PRELOAD",
    "SymmetricBranchingValidationCase",
    "SymmetricBranchingValidationLabels",
    "TERMINAL_CHANNEL",
    "build_cascade_config",
    "build_expected_labels",
    "build_initial_state",
    "build_pathological_system",
    "build_reference_system",
    "build_solver_config",
    "build_symmetric_branching_ambiguity_case",
    "build_tensor_config",
    "build_validation_scenarios",
]
