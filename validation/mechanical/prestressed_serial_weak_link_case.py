"""
ROIF Validation Suite
Mechanical Scenario 03A:
Pre-Stressed Serial Weak-Link Chain вЂ” Control Validation

A deliberately simple non-branching mechanical control case.

Topology:

    preload_disturbance
            |
            v
    proximal_transmission
            |
            v
      serial_weak_link
            |
            v
    distal_transmission
            |
            v
    terminal_displacement

The case is designed to test whether Active Cascade can avoid inventing
branch ambiguity when every internal node has only one outgoing relation.

External validation labels are evaluator-side only and are never supplied
to the Solver.
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
from roif.roif_cascade import (
    CascadeConfig,
    CascadeDirection,
    CascadeUpdateMode,
)
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

PRELOAD_CHANNEL = "preload_disturbance"
PROXIMAL_CHANNEL = "proximal_transmission"
WEAK_LINK_CHANNEL = "serial_weak_link"
DISTAL_CHANNEL = "distal_transmission"
TERMINAL_CHANNEL = "terminal_displacement"

PRELOAD_TO_PROXIMAL = "preload_to_proximal"
PROXIMAL_TO_WEAK_LINK = "proximal_to_weak_link"
WEAK_LINK_TO_DISTAL = "weak_link_to_distal"
DISTAL_TO_TERMINAL = "distal_to_terminal"

SCENARIO_REMOVE_PRELOAD = "remove_preload_influence"
SCENARIO_REDUCE_WEAK_LINK = "reduce_weak_link_load"
SCENARIO_REDUCE_DISTAL = "reduce_distal_load"
SCENARIO_REDUCE_TERMINAL = "reduce_terminal_load"


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class SerialWeakLinkValidationLabels:
    d_origin: str
    d_fast: str
    d_root: str
    node_star: str

    def as_mapping(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "d_origin": self.d_origin,
                "d_fast": self.d_fast,
                "d_root": self.d_root,
                "node_star": self.node_star,
            }
        )


@dataclass(frozen=True, slots=True)
class PrestressedSerialWeakLinkValidationCase:
    case_id: str
    title: str
    pathological_system: ROIFSystem
    reference_system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: SerialWeakLinkValidationLabels
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
    history_factor: float = 1.0,
    role: FunctionalRole = FunctionalRole.TRANSMITTER,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id,
        name=name,
        role=role,
        direction=Vector3(1.0, 0.0, 0.0),
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


def _build_system(*, pathological: bool) -> ROIFSystem:
    if pathological:
        values = {
            "preload_capacity": 1.20,
            "preload_load": 1.00,
            "preload_activation": 1.00,
            "preload_mobility": 1.00,
            "preload_history": 1.00,
            "proximal_capacity": 3.20,
            "proximal_load": 1.40,
            "proximal_activation": 0.92,
            "proximal_mobility": 0.90,
            "proximal_history": 0.96,
            "weak_capacity": 1.60,
            "weak_load": 1.46,
            "weak_activation": 0.52,
            "weak_mobility": 0.42,
            "weak_history": 0.64,
            "distal_capacity": 3.00,
            "distal_load": 1.95,
            "distal_activation": 0.82,
            "distal_mobility": 0.72,
            "distal_history": 0.84,
            "terminal_capacity": 3.20,
            "terminal_load": 2.05,
            "terminal_activation": 0.80,
            "terminal_mobility": 0.68,
            "terminal_history": 0.82,
        }
    else:
        values = {
            "preload_capacity": 1.20,
            "preload_load": 0.12,
            "preload_activation": 1.00,
            "preload_mobility": 1.00,
            "preload_history": 1.00,
            "proximal_capacity": 3.40,
            "proximal_load": 0.75,
            "proximal_activation": 0.97,
            "proximal_mobility": 0.96,
            "proximal_history": 1.00,
            "weak_capacity": 3.20,
            "weak_load": 0.82,
            "weak_activation": 0.96,
            "weak_mobility": 0.95,
            "weak_history": 1.00,
            "distal_capacity": 3.40,
            "distal_load": 0.78,
            "distal_activation": 0.96,
            "distal_mobility": 0.95,
            "distal_history": 1.00,
            "terminal_capacity": 3.50,
            "terminal_load": 0.72,
            "terminal_activation": 0.96,
            "terminal_mobility": 0.95,
            "terminal_history": 1.00,
        }

    preload = _channel(
        PRELOAD_CHANNEL,
        entity_id="preload_source",
        name="Upstream preload disturbance",
        capacity=values["preload_capacity"],
        load=values["preload_load"],
        activation=values["preload_activation"],
        mobility=values["preload_mobility"],
        history_factor=values["preload_history"],
        role=FunctionalRole.MODULATOR,
    )
    proximal = _channel(
        PROXIMAL_CHANNEL,
        entity_id="proximal_stage",
        name="Proximal serial transmission",
        capacity=values["proximal_capacity"],
        load=values["proximal_load"],
        activation=values["proximal_activation"],
        mobility=values["proximal_mobility"],
        history_factor=values["proximal_history"],
    )
    weak_link = _channel(
        WEAK_LINK_CHANNEL,
        entity_id="weak_link",
        name="Serial weak-link reserve",
        capacity=values["weak_capacity"],
        load=values["weak_load"],
        activation=values["weak_activation"],
        mobility=values["weak_mobility"],
        history_factor=values["weak_history"],
    )
    distal = _channel(
        DISTAL_CHANNEL,
        entity_id="distal_stage",
        name="Distal serial transmission",
        capacity=values["distal_capacity"],
        load=values["distal_load"],
        activation=values["distal_activation"],
        mobility=values["distal_mobility"],
        history_factor=values["distal_history"],
    )
    terminal = _channel(
        TERMINAL_CHANNEL,
        entity_id="terminal",
        name="Terminal displacement tolerance",
        capacity=values["terminal_capacity"],
        load=values["terminal_load"],
        activation=values["terminal_activation"],
        mobility=values["terminal_mobility"],
        history_factor=values["terminal_history"],
        role=FunctionalRole.SENSOR,
    )

    return ROIFSystem(
        system_id=(
            "prestressed_serial_weak_link_pathological"
            if pathological
            else "prestressed_serial_weak_link_reference"
        ),
        name=(
            "Pre-stressed serial weak-link mechanical chain"
            if pathological
            else "Restored serial mechanical reference"
        ),
        entities=(
            _entity(
                "preload_source",
                name="Preload source",
                channel=preload,
            ),
            _entity(
                "proximal_stage",
                name="Proximal transmission stage",
                channel=proximal,
            ),
            _entity(
                "weak_link",
                name="Serial weak link",
                channel=weak_link,
            ),
            _entity(
                "distal_stage",
                name="Distal transmission stage",
                channel=distal,
            ),
            _entity(
                "terminal",
                name="Terminal response",
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
                PRELOAD_TO_PROXIMAL,
                source=PRELOAD_CHANNEL,
                target=PROXIMAL_CHANNEL,
                gain=0.92,
            ),
            _operator(
                PROXIMAL_TO_WEAK_LINK,
                source=PROXIMAL_CHANNEL,
                target=WEAK_LINK_CHANNEL,
                gain=0.96,
            ),
            _operator(
                WEAK_LINK_TO_DISTAL,
                source=WEAK_LINK_CHANNEL,
                target=DISTAL_CHANNEL,
                gain=0.82,
            ),
            _operator(
                DISTAL_TO_TERMINAL,
                source=DISTAL_CHANNEL,
                target=TERMINAL_CHANNEL,
                gain=0.76,
            ),
        ),
        metadata={
            "validation_case": "mechanical_03A",
            "domain": "mechanical",
            "topology": "serial_weak_link",
            "pre_stressed": True,
            "branching": False,
            "feedback": False,
            "state": (
                "pathological"
                if pathological
                else "reference"
            ),
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
        },
    )


def build_pathological_system() -> ROIFSystem:
    return _build_system(pathological=True)


def build_reference_system() -> ROIFSystem:
    return _build_system(pathological=False)


def build_initial_state() -> Mapping[str, float]:
    return MappingProxyType(
        {
            PRELOAD_CHANNEL: 1.0,
            PROXIMAL_CHANNEL: 0.0,
            WEAK_LINK_CHANNEL: 0.0,
            DISTAL_CHANNEL: 0.0,
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
            scenario_id=SCENARIO_REMOVE_PRELOAD,
            name="Remove upstream preload influence",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="remove:preload_disturbance",
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
            scenario_id=SCENARIO_REDUCE_WEAK_LINK,
            name="Reduce weak-link incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:serial_weak_link",
                    channel_id=WEAK_LINK_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.25,
                    safety_risk=0.05,
                    uncertainty=0.10,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_REDUCE_DISTAL,
            name="Reduce distal transmission load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:distal_transmission",
                    channel_id=DISTAL_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.35,
                    safety_risk=0.08,
                    uncertainty=0.15,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_REDUCE_TERMINAL,
            name="Reduce terminal displacement load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:terminal_displacement",
                    channel_id=TERMINAL_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.45,
                    safety_risk=0.10,
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
            "validation_case": "mechanical_03A",
            "domain": "mechanical",
            "navigator_only": True,
        },
    )


def build_expected_labels() -> SerialWeakLinkValidationLabels:
    return SerialWeakLinkValidationLabels(
        d_origin=PRELOAD_CHANNEL,
        d_fast=WEAK_LINK_CHANNEL,
        d_root=WEAK_LINK_CHANNEL,
        node_star=PRELOAD_CHANNEL,
    )


def build_prestressed_serial_weak_link_case() -> (
    PrestressedSerialWeakLinkValidationCase
):
    return PrestressedSerialWeakLinkValidationCase(
        case_id="mechanical_03A_prestressed_serial_weak_link",
        title="Pre-stressed serial weak-link mechanical chain",
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
            "validation_family": "mechanical_03_serial_controls",
            "validation_stage": "03A",
            "evidence_type": "controlled synthetic mechanical ground truth",
            "topology": "serial_weak_link",
            "pre_stressed": True,
            "branching": False,
            "feedback": False,
            "active_probe_control_case": True,
            "expected_branch_ambiguity": False,
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


__all__ = [
    "DISTAL_CHANNEL",
    "DISTAL_TO_TERMINAL",
    "PRELOAD_CHANNEL",
    "PRELOAD_TO_PROXIMAL",
    "PROXIMAL_CHANNEL",
    "PROXIMAL_TO_WEAK_LINK",
    "PrestressedSerialWeakLinkValidationCase",
    "SCENARIO_REDUCE_DISTAL",
    "SCENARIO_REDUCE_TERMINAL",
    "SCENARIO_REDUCE_WEAK_LINK",
    "SCENARIO_REMOVE_PRELOAD",
    "SerialWeakLinkValidationLabels",
    "TERMINAL_CHANNEL",
    "WEAK_LINK_CHANNEL",
    "WEAK_LINK_TO_DISTAL",
    "build_cascade_config",
    "build_expected_labels",
    "build_initial_state",
    "build_pathological_system",
    "build_prestressed_serial_weak_link_case",
    "build_reference_system",
    "build_solver_config",
    "build_tensor_config",
    "build_validation_scenarios",
]

