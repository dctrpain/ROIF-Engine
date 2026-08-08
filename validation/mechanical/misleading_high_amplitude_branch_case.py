"""
ROIF Validation Suite
Mechanical Scenario 03C:
Misleading High-Amplitude Branch

Purpose
-------
Scenario 03C tests whether ROIF can distinguish response magnitude from
directional causal relevance.

Topology
--------

                         /-> loud_orthogonal_branch --\
    shared_probe_node                               -> common_terminal
                         +-> aligned_quiet_branch ----/

The deliberately misleading branch produces a larger response amplitude but
its declared mechanical direction is approximately orthogonal to the candidate
edge direction used by the later Vector Probe layer.

The aligned branch produces a smaller response but follows the candidate edge
direction.

Validation hypothesis
---------------------
A causal-inference layer that uses vector evidence should NOT prefer the
high-amplitude branch merely because its scalar magnitude is larger.

The ground-truth contract therefore declares:

    amplitude_winner    = loud_orthogonal_branch
    directional_winner  = aligned_quiet_branch

These labels are evaluator-side only and must never be supplied to Solver,
ActiveCascadeExplorer, ProbePlanner, or ActiveProbeEngine.

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
from roif.roif_tensor import SelfCouplingMode, TensorBuildConfig
from roif.roif_cascade import CascadeConfig, CascadeDirection, CascadeUpdateMode
from roif.roif_root_detector import RootDetectorConfig
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

PROBE_NODE_CHANNEL = "shared_probe_node"
LOUD_BRANCH_CHANNEL = "loud_orthogonal_branch"
ALIGNED_BRANCH_CHANNEL = "aligned_quiet_branch"
TERMINAL_CHANNEL = "common_terminal"

PROBE_TO_LOUD = "probe_to_loud_orthogonal"
PROBE_TO_ALIGNED = "probe_to_aligned_quiet"
LOUD_TO_TERMINAL = "loud_to_terminal"
ALIGNED_TO_TERMINAL = "aligned_to_terminal"

SCENARIO_REMOVE_PROBE_NODE = "remove_shared_probe_node"
SCENARIO_REDUCE_LOUD = "reduce_loud_branch_load"
SCENARIO_REDUCE_ALIGNED = "reduce_aligned_branch_load"
SCENARIO_REDUCE_TERMINAL = "reduce_common_terminal_load"

LOUD_GAIN = 1.20
ALIGNED_GAIN = 0.72

LOUD_RESPONSE_DIRECTION = Vector3(0.0, 1.0, 0.0)
ALIGNED_RESPONSE_DIRECTION = Vector3(1.0, 0.0, 0.0)
CANDIDATE_EDGE_DIRECTION = Vector3(1.0, 0.0, 0.0)

# Passive CapacityTensor transport directions.
# Probe-response direction is intentionally kept separate.
LOUD_TRANSPORT_DIRECTION = Vector3(1.0, 0.0, 0.0)
ALIGNED_TRANSPORT_DIRECTION = Vector3(1.0, 0.0, 0.0)


def _readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class MisleadingAmplitudeValidationLabels:
    amplitude_winner: str
    directional_winner: str
    terminal: str
    amplitude_must_not_decide_direction: bool = True

    def as_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "amplitude_winner": self.amplitude_winner,
                "directional_winner": self.directional_winner,
                "terminal": self.terminal,
                "amplitude_must_not_decide_direction": self.amplitude_must_not_decide_direction,
            }
        )


@dataclass(frozen=True, slots=True)
class MisleadingHighAmplitudeValidationCase:
    case_id: str
    title: str
    pathological_system: ROIFSystem
    reference_system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: MisleadingAmplitudeValidationLabels
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_state", _readonly_mapping(self.initial_state))
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(self.pathological_system.channel_ids)

    def scenario(self, scenario_id: str) -> CounterfactualScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(scenario_id)


def _channel(
    channel_id: str,
    *,
    entity_id: str,
    name: str,
    capacity: float,
    load: float,
    activation: float,
    mobility: float,
    history_factor: float,
    role: FunctionalRole,
    direction: Vector3,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id,
        name=name,
        role=role,
        direction=direction,
        capacity_state=CapacityState(capacity=capacity, load=load),
        activation=ActivationAvailability(command=activation),
        geometry=GeometryState(mobility=mobility),
        history_factor=history_factor,
    )


def _entity(entity_id: str, *, name: str, channel: FunctionalChannel) -> StructuralEntity:
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


def _build_system(*, pathological: bool) -> ROIFSystem:
    if pathological:
        values = {
            "probe_capacity": 1.60,
            "probe_load": 1.10,
            "probe_activation": 1.00,
            "probe_mobility": 1.00,
            "probe_history": 1.00,
            "loud_capacity": 2.80,
            "loud_load": 1.55,
            "loud_activation": 0.96,
            "loud_mobility": 0.95,
            "loud_history": 0.96,
            "aligned_capacity": 2.50,
            "aligned_load": 1.25,
            "aligned_activation": 0.84,
            "aligned_mobility": 0.82,
            "aligned_history": 0.88,
            "terminal_capacity": 3.80,
            "terminal_load": 2.05,
            "terminal_activation": 0.88,
            "terminal_mobility": 0.82,
            "terminal_history": 0.90,
        }
        gains = {
            "probe_loud": LOUD_GAIN,
            "probe_aligned": ALIGNED_GAIN,
            "loud_terminal": 0.86,
            "aligned_terminal": 0.84,
        }
    else:
        values = {
            "probe_capacity": 1.60,
            "probe_load": 0.20,
            "probe_activation": 1.00,
            "probe_mobility": 1.00,
            "probe_history": 1.00,
            "loud_capacity": 3.00,
            "loud_load": 0.70,
            "loud_activation": 0.95,
            "loud_mobility": 0.95,
            "loud_history": 1.00,
            "aligned_capacity": 3.00,
            "aligned_load": 0.70,
            "aligned_activation": 0.95,
            "aligned_mobility": 0.95,
            "aligned_history": 1.00,
            "terminal_capacity": 4.00,
            "terminal_load": 0.80,
            "terminal_activation": 0.95,
            "terminal_mobility": 0.95,
            "terminal_history": 1.00,
        }
        gains = {
            "probe_loud": 0.90,
            "probe_aligned": 0.90,
            "loud_terminal": 0.85,
            "aligned_terminal": 0.85,
        }

    probe_node = _channel(
        PROBE_NODE_CHANNEL,
        entity_id="shared_probe",
        name="Shared probe node",
        capacity=values["probe_capacity"],
        load=values["probe_load"],
        activation=values["probe_activation"],
        mobility=values["probe_mobility"],
        history_factor=values["probe_history"],
        role=FunctionalRole.MODULATOR,
        direction=CANDIDATE_EDGE_DIRECTION,
    )
    loud_branch = _channel(
        LOUD_BRANCH_CHANNEL,
        entity_id="loud_branch",
        name="Loud orthogonal branch",
        capacity=values["loud_capacity"],
        load=values["loud_load"],
        activation=values["loud_activation"],
        mobility=values["loud_mobility"],
        history_factor=values["loud_history"],
        role=FunctionalRole.TRANSMITTER,
        direction=LOUD_TRANSPORT_DIRECTION,
    )
    aligned_branch = _channel(
        ALIGNED_BRANCH_CHANNEL,
        entity_id="aligned_branch",
        name="Aligned quiet branch",
        capacity=values["aligned_capacity"],
        load=values["aligned_load"],
        activation=values["aligned_activation"],
        mobility=values["aligned_mobility"],
        history_factor=values["aligned_history"],
        role=FunctionalRole.TRANSMITTER,
        direction=ALIGNED_TRANSPORT_DIRECTION,
    )
    terminal = _channel(
        TERMINAL_CHANNEL,
        entity_id="common_terminal",
        name="Common terminal",
        capacity=values["terminal_capacity"],
        load=values["terminal_load"],
        activation=values["terminal_activation"],
        mobility=values["terminal_mobility"],
        history_factor=values["terminal_history"],
        role=FunctionalRole.SENSOR,
        direction=CANDIDATE_EDGE_DIRECTION,
    )

    return ROIFSystem(
        system_id=(
            "misleading_high_amplitude_pathological"
            if pathological
            else "misleading_high_amplitude_reference"
        ),
        name=(
            "Misleading high-amplitude branch mechanical system"
            if pathological
            else "Balanced reference for misleading amplitude control"
        ),
        entities=(
            _entity("shared_probe", name="Shared probe node", channel=probe_node),
            _entity("loud_branch", name="Loud orthogonal branch", channel=loud_branch),
            _entity("aligned_branch", name="Aligned quiet branch", channel=aligned_branch),
            _entity("common_terminal", name="Common terminal", channel=terminal),
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
            _operator(PROBE_TO_LOUD, source=PROBE_NODE_CHANNEL, target=LOUD_BRANCH_CHANNEL, gain=gains["probe_loud"]),
            _operator(PROBE_TO_ALIGNED, source=PROBE_NODE_CHANNEL, target=ALIGNED_BRANCH_CHANNEL, gain=gains["probe_aligned"]),
            _operator(LOUD_TO_TERMINAL, source=LOUD_BRANCH_CHANNEL, target=TERMINAL_CHANNEL, gain=gains["loud_terminal"]),
            _operator(ALIGNED_TO_TERMINAL, source=ALIGNED_BRANCH_CHANNEL, target=TERMINAL_CHANNEL, gain=gains["aligned_terminal"]),
        ),
        metadata={
            "validation_case": "mechanical_03C",
            "domain": "mechanical",
            "topology": "misleading_high_amplitude_branch",
            "branching": True,
            "feedback": False,
            "pre_stressed": True,
            "scalar_amplitude_conflict": pathological,
            "preferred_branch_label_used": False,
            "expected_directional_winner_used_by_solver": False,
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
        },
    )



# =============================================================================
# Direction-layer helpers
# =============================================================================


def probe_response_direction(
    relation_id: str,
) -> Vector3:
    """
    Synthetic local response direction used by Vector Probe evidence.

    This is deliberately separate from passive CapacityTensor transport.
    """
    if relation_id == PROBE_TO_LOUD:
        return LOUD_RESPONSE_DIRECTION

    if relation_id == PROBE_TO_ALIGNED:
        return ALIGNED_RESPONSE_DIRECTION

    raise KeyError(relation_id)


def transport_direction(
    channel_id: str,
) -> Vector3:
    """
    Passive mechanical transport direction used by CapacityTensor.
    """
    if channel_id == LOUD_BRANCH_CHANNEL:
        return LOUD_TRANSPORT_DIRECTION

    if channel_id == ALIGNED_BRANCH_CHANNEL:
        return ALIGNED_TRANSPORT_DIRECTION

    if channel_id in {
        PROBE_NODE_CHANNEL,
        TERMINAL_CHANNEL,
    }:
        return CANDIDATE_EDGE_DIRECTION

    raise KeyError(channel_id)


# =============================================================================
# Public system factories
# =============================================================================

def build_pathological_system() -> ROIFSystem:
    return _build_system(pathological=True)



def build_reference_system() -> ROIFSystem:
    return _build_system(pathological=False)



def build_initial_state() -> Mapping[str, float]:
    return MappingProxyType(
        {
            PROBE_NODE_CHANNEL: 1.0,
            LOUD_BRANCH_CHANNEL: 0.0,
            ALIGNED_BRANCH_CHANNEL: 0.0,
            TERMINAL_CHANNEL: 0.0,
        }
    )



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



def build_validation_scenarios() -> tuple[CounterfactualScenario, ...]:
    return (
        CounterfactualScenario(
            scenario_id=SCENARIO_REMOVE_PROBE_NODE,
            name="Remove shared probe-node outgoing influence",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="remove:shared_probe_node",
                    channel_id=PROBE_NODE_CHANNEL,
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
            scenario_id=SCENARIO_REDUCE_LOUD,
            name="Reduce loud branch incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:loud_orthogonal_branch",
                    channel_id=LOUD_BRANCH_CHANNEL,
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
            scenario_id=SCENARIO_REDUCE_ALIGNED,
            name="Reduce aligned branch incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:aligned_quiet_branch",
                    channel_id=ALIGNED_BRANCH_CHANNEL,
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
            "validation_case": "mechanical_03C",
            "domain": "mechanical",
            "navigator_only": True,
            "preferred_branch_label_used": False,
        },
    )



def build_expected_labels() -> MisleadingAmplitudeValidationLabels:
    return MisleadingAmplitudeValidationLabels(
        amplitude_winner=LOUD_BRANCH_CHANNEL,
        directional_winner=ALIGNED_BRANCH_CHANNEL,
        terminal=TERMINAL_CHANNEL,
        amplitude_must_not_decide_direction=True,
    )



def build_misleading_high_amplitude_case() -> MisleadingHighAmplitudeValidationCase:
    return MisleadingHighAmplitudeValidationCase(
        case_id="mechanical_03C_misleading_high_amplitude_branch",
        title="Misleading high-amplitude branch mechanical control",
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
            "validation_stage": "03C",
            "evidence_type": "controlled synthetic scalar-vector conflict",
            "topology": "misleading_high_amplitude_branch",
            "pre_stressed": True,
            "branching": True,
            "feedback": False,
            "amplitude_winner": "evaluator_side_only",
            "directional_winner": "evaluator_side_only",
            "expected_scalar_vector_conflict": True,
            "expected_high_amplitude_branch_directional_support": False,
            "expected_low_amplitude_branch_directional_support": True,
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


__all__ = [
    "ALIGNED_BRANCH_CHANNEL",
    "ALIGNED_GAIN",
    "ALIGNED_RESPONSE_DIRECTION",
    "ALIGNED_TRANSPORT_DIRECTION",
    "ALIGNED_TO_TERMINAL",
    "CANDIDATE_EDGE_DIRECTION",
    "LOUD_BRANCH_CHANNEL",
    "LOUD_GAIN",
    "LOUD_RESPONSE_DIRECTION",
    "LOUD_TRANSPORT_DIRECTION",
    "LOUD_TO_TERMINAL",
    "MisleadingAmplitudeValidationLabels",
    "MisleadingHighAmplitudeValidationCase",
    "PROBE_NODE_CHANNEL",
    "PROBE_TO_ALIGNED",
    "PROBE_TO_LOUD",
    "SCENARIO_REDUCE_ALIGNED",
    "SCENARIO_REDUCE_LOUD",
    "SCENARIO_REDUCE_TERMINAL",
    "SCENARIO_REMOVE_PROBE_NODE",
    "TERMINAL_CHANNEL",
    "build_cascade_config",
    "build_expected_labels",
    "build_initial_state",
    "build_misleading_high_amplitude_case",
    "build_pathological_system",
    "build_reference_system",
    "build_solver_config",
    "build_tensor_config",
    "build_validation_scenarios",
    "probe_response_direction",
    "transport_direction",
]



