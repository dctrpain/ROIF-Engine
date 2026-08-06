"""
ROIF Validation Suite
Clinical Scenario 01: Forefoot Restriction -> Tibial Rotation -> Knee
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


def _readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ClinicalValidationLabels:
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
class FootKneeValidationCase:
    case_id: str
    title: str
    pathological_system: ROIFSystem
    reference_system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: ClinicalValidationLabels
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
    history_factor: float = 1.0,
    role: FunctionalRole = FunctionalRole.TRANSMITTER,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id,
        name=name,
        role=role,
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(capacity=capacity, load=load),
        activation=ActivationAvailability(command=activation),
        geometry=GeometryState(mobility=mobility),
        history_factor=history_factor,
    )


def _entity(
    entity_id: str,
    *,
    name: str,
    channels: tuple[FunctionalChannel, ...],
) -> StructuralEntity:
    return StructuralEntity(
        entity_id=entity_id,
        name=name,
        kind=EntityKind.GENERIC,
        channels=channels,
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
            "fibrosis_load": 0.90,
            "compliance_capacity": 1.40,
            "compliance_load": 1.20,
            "compliance_activation": 0.35,
            "compliance_mobility": 0.20,
            "compliance_history": 0.55,
            "tibial_capacity": 2.50,
            "tibial_load": 2.10,
            "tibial_activation": 0.75,
            "tibial_mobility": 0.55,
            "tibial_history": 0.80,
            "knee_capacity": 3.00,
            "knee_load": 2.45,
            "knee_activation": 0.80,
            "knee_mobility": 0.60,
            "knee_history": 0.85,
        }
    else:
        values = {
            "fibrosis_load": 0.10,
            "compliance_capacity": 3.50,
            "compliance_load": 0.40,
            "compliance_activation": 0.95,
            "compliance_mobility": 0.95,
            "compliance_history": 1.00,
            "tibial_capacity": 3.50,
            "tibial_load": 0.80,
            "tibial_activation": 0.95,
            "tibial_mobility": 0.90,
            "tibial_history": 1.00,
            "knee_capacity": 4.00,
            "knee_load": 0.90,
            "knee_activation": 0.95,
            "knee_mobility": 0.90,
            "knee_history": 1.00,
        }

    forefoot_fibrosis = _channel(
        "forefoot_fibrosis",
        entity_id="lateral_forefoot",
        name="Lateral forefoot fibrosis",
        capacity=1.0,
        load=values["fibrosis_load"],
        activation=1.0,
        mobility=1.0,
        role=FunctionalRole.MODULATOR,
    )
    forefoot_compliance = _channel(
        "forefoot_compliance",
        entity_id="lateral_forefoot",
        name="Forefoot compliance",
        capacity=values["compliance_capacity"],
        load=values["compliance_load"],
        activation=values["compliance_activation"],
        mobility=values["compliance_mobility"],
        history_factor=values["compliance_history"],
    )
    tibial_rotation_control = _channel(
        "tibial_rotation_control",
        entity_id="tibia",
        name="Tibial rotation control",
        capacity=values["tibial_capacity"],
        load=values["tibial_load"],
        activation=values["tibial_activation"],
        mobility=values["tibial_mobility"],
        history_factor=values["tibial_history"],
        role=FunctionalRole.STABILIZER,
    )
    knee_tracking = _channel(
        "knee_tracking",
        entity_id="knee",
        name="Knee tracking",
        capacity=values["knee_capacity"],
        load=values["knee_load"],
        activation=values["knee_activation"],
        mobility=values["knee_mobility"],
        history_factor=values["knee_history"],
        role=FunctionalRole.COMPENSATOR,
    )
    knee_flexion_tolerance = _channel(
        "knee_flexion_tolerance",
        entity_id="knee",
        name="Knee flexion tolerance",
        capacity=3.0 if pathological else 4.0,
        load=2.20 if pathological else 0.70,
        activation=0.70 if pathological else 0.95,
        mobility=0.50 if pathological else 0.95,
        history_factor=0.80 if pathological else 1.00,
        role=FunctionalRole.BUFFER,
    )

    return ROIFSystem(
        system_id="foot_knee_pathological" if pathological else "foot_knee_reference",
        name="Foot-knee pathological cascade" if pathological else "Foot-knee restored reference",
        entities=(
            _entity(
                "lateral_forefoot",
                name="Lateral forefoot",
                channels=(forefoot_fibrosis, forefoot_compliance),
            ),
            _entity("tibia", name="Tibia", channels=(tibial_rotation_control,)),
            _entity(
                "knee",
                name="Knee",
                channels=(knee_tracking, knee_flexion_tolerance),
            ),
        ),
        planes=(
            InfluencePlane(
                plane_id="mechanical",
                name="Mechanical",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.STATIC,
            ),
        ),
        operators=(
            _operator(
                "fibrosis_to_compliance",
                source="forefoot_fibrosis",
                target="forefoot_compliance",
                gain=0.95,
            ),
            _operator(
                "compliance_to_tibial_rotation",
                source="forefoot_compliance",
                target="tibial_rotation_control",
                gain=0.90,
            ),
            _operator(
                "tibial_rotation_to_knee_tracking",
                source="tibial_rotation_control",
                target="knee_tracking",
                gain=0.85,
            ),
            _operator(
                "knee_tracking_to_flexion",
                source="knee_tracking",
                target="knee_flexion_tolerance",
                gain=0.80,
            ),
            _operator(
                "flexion_feedback_to_tibial_rotation",
                source="knee_flexion_tolerance",
                target="tibial_rotation_control",
                gain=0.20,
            ),
        ),
        metadata={
            "validation_case": "foot_knee",
            "state": "pathological" if pathological else "reference",
            "clinical_scope": "local",
        },
    )


def build_pathological_system() -> ROIFSystem:
    return _build_system(pathological=True)


def build_reference_system() -> ROIFSystem:
    return _build_system(pathological=False)


def build_initial_state() -> Mapping[str, float]:
    return MappingProxyType(
        {
            "forefoot_fibrosis": 1.0,
            "forefoot_compliance": 0.0,
            "tibial_rotation_control": 0.0,
            "knee_tracking": 0.0,
            "knee_flexion_tolerance": 0.0,
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
        steps=8,
        update_mode=CascadeUpdateMode.LEAKY,
        direction=CascadeDirection.GENERIC,
        retention=0.10,
        dissipation=0.10,
        lower_bound=0.0,
        upper_bound=100.0,
        clip_state=False,
        convergence_tolerance=1e-10,
        convergence_patience=10,
        record_transmission_events=True,
        stop_on_failure=False,
    )


def build_validation_scenarios() -> tuple[CounterfactualScenario, ...]:
    return (
        CounterfactualScenario(
            scenario_id="release_forefoot_fibrosis",
            name="Release forefoot fibrosis influence",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="remove_outgoing:forefoot_fibrosis",
                    channel_id="forefoot_fibrosis",
                    action=CounterfactualAction.REMOVE_OUTGOING_INFLUENCE,
                    magnitude=1.0,
                    cost=0.20,
                    safety_risk=0.05,
                    uncertainty=0.15,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="restore_tibial_rotation_control",
            name="Restore tibial rotation control",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="restore:tibial_rotation_control",
                    channel_id="tibial_rotation_control",
                    action=CounterfactualAction.REMOVE_OUTGOING_INFLUENCE,
                    magnitude=1.0,
                    cost=0.35,
                    safety_risk=0.10,
                    uncertainty=0.20,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="reduce_knee_tracking_load",
            name="Reduce knee tracking load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:knee_tracking",
                    channel_id="knee_tracking",
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.45,
                    safety_risk=0.10,
                    uncertainty=0.25,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="reduce_knee_flexion_load",
            name="Reduce knee flexion load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:knee_flexion_tolerance",
                    channel_id="knee_flexion_tolerance",
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.40,
                    safety_risk=0.10,
                    uncertainty=0.25,
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
            steps=8,
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
            "validation_case": "foot_knee",
            "navigator_only": True,
        },
    )


def build_expected_labels() -> ClinicalValidationLabels:
    return ClinicalValidationLabels(
        d_origin="forefoot_fibrosis",
        d_fast="forefoot_compliance",
        d_root="tibial_rotation_control",
        node_star="forefoot_compliance",
    )


def build_foot_knee_case() -> FootKneeValidationCase:
    return FootKneeValidationCase(
        case_id="clinical_01_foot_knee",
        title="Forefoot restriction to knee dysfunction",
        pathological_system=build_pathological_system(),
        reference_system=build_reference_system(),
        initial_state=build_initial_state(),
        tensor_config=build_tensor_config(),
        cascade_config=build_cascade_config(),
        solver_config=build_solver_config(),
        scenarios=build_validation_scenarios(),
        expected=build_expected_labels(),
        metadata={
            "domain": "clinical_biomechanics",
            "evidence_type": "repeated clinical counterfactual observation",
            "observed_response": (
                "Local correction of forefoot restriction restored knee flexion immediately."
            ),
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


__all__ = [
    "ClinicalValidationLabels",
    "FootKneeValidationCase",
    "build_cascade_config",
    "build_expected_labels",
    "build_foot_knee_case",
    "build_initial_state",
    "build_pathological_system",
    "build_reference_system",
    "build_solver_config",
    "build_tensor_config",
    "build_validation_scenarios",
]


