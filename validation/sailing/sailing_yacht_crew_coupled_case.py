"""
ROIF External-Domain Validation
Scenario 04A вЂ” Sailing Yacht + Crew:
Coupled Pre-Stressed HumanвЂ“Mechanical Control System

Research basis
--------------
This validation case is structurally grounded in published yachtвЂ“crew
simulation work by Scarponi, Shenoi, Turnock, Conti and collaborators:

- Scarponi et al. (2007), "Including Human Performance in the Dynamic
  Model of a Sailing Yacht: A MATLAB-Simulink Based Tool"
- Scarponi et al. (2008), "Interactions Between Yacht-Crew Systems and
  Racing Scenarios Combining Behavioural Models With VPPs"
- Scarponi et al. (2007), "Robo-Yacht: a human behaviour-based tool to
  predict the performances of yacht-crew systems"

The published system treats yacht performance as a coupled yachtвЂ“crew
problem. Yacht equations of motion are solved in the time domain, while crew
inputs include yacht steering and sail trim; helmsman and sail-trimmer
behaviour are modeled explicitly.

Important methodological boundary
---------------------------------
The topology and role separation in this benchmark are research-grounded.

The normalized ROIF capacities, loads, gains and costs below are NOT claimed
to be numerical values reported by those papers. They are transparent,
deterministic benchmark parameters used to instantiate the published
humanвЂ“mechanical coupling inside the ROIF engine.

Conceptual model
----------------

Environment:
    wind / wind-shift disturbance

Mechanical pre-stressed system:
    sail load
        -> heel / yaw tendency
        -> hull / appendage response
        -> course error

Human pre-stressed control system:
    perceived course / heel state
        -> helmsman control demand
        -> sail-trimmer control demand
        -> rudder / sail-trim actions

Closed loop:
    yacht state
        -> crew perception / demand
        -> crew action
        -> yacht state

Key ROIF hypothesis
-------------------
A yacht may remain close to the desired course while crew control demand rises.
Therefore low observable course error does NOT imply large remaining system
reserve.

This benchmark is designed to distinguish:
    observable state deviation
from:
    compensatory control effort / reserve consumption

External validation labels are evaluator-side only and must not be passed to
the solver.

No biological, diagnostic, or treatment claim is made.
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


# =============================================================================
# Published-source identifiers
# =============================================================================


SOURCE_SCARPONI_2007 = "scarponi_et_al_2007_modern_yacht"
SOURCE_SCARPONI_2008 = "scarponi_et_al_2008_yacht_crew_vpp"
SOURCE_ROBO_YACHT_2007 = "scarponi_et_al_2007_robo_yacht"


# =============================================================================
# Channel identifiers
# =============================================================================

# Environment / disturbance
WIND_DISTURBANCE_CHANNEL = "wind_shift_disturbance"

# Yacht / mechanical plant
SAIL_LOAD_CHANNEL = "sail_load"
HEEL_YAW_CHANNEL = "heel_yaw_response"
COURSE_ERROR_CHANNEL = "course_error"

# Crew / living controller
HELM_CONTROL_DEMAND_CHANNEL = "helmsman_control_demand"
TRIM_CONTROL_DEMAND_CHANNEL = "sail_trimmer_control_demand"

# Actuation back into yacht
RUDDER_ACTION_CHANNEL = "rudder_action"
SAIL_TRIM_ACTION_CHANNEL = "sail_trim_action"


# =============================================================================
# Operator identifiers
# =============================================================================


WIND_TO_SAIL = "wind_to_sail_load"
SAIL_TO_HEEL_YAW = "sail_load_to_heel_yaw"
HEEL_YAW_TO_COURSE = "heel_yaw_to_course_error"

COURSE_TO_HELM = "course_error_to_helmsman_demand"
HEEL_YAW_TO_TRIMMER = "heel_yaw_to_sail_trimmer_demand"

HELM_TO_RUDDER = "helmsman_demand_to_rudder_action"
TRIMMER_TO_SAIL_TRIM = "trimmer_demand_to_sail_trim_action"

RUDDER_TO_COURSE = "rudder_action_to_course_error"
SAIL_TRIM_TO_SAIL = "sail_trim_action_to_sail_load"


# =============================================================================
# Counterfactual scenario identifiers
# =============================================================================


SCENARIO_REDUCE_WIND_DISTURBANCE = "reduce_wind_disturbance"
SCENARIO_REDUCE_SAIL_LOAD = "reduce_sail_load"
SCENARIO_RELIEVE_HELM_DEMAND = "relieve_helmsman_control_demand"
SCENARIO_RELIEVE_TRIMMER_DEMAND = "relieve_sail_trimmer_control_demand"


# =============================================================================
# Geometry / direction conventions
# =============================================================================


FORWARD_DIRECTION = Vector3(
    1.0,
    0.0,
    0.0,
)

CONTROL_DIRECTION = Vector3(
    0.0,
    1.0,
    0.0,
)

CORRECTIVE_DIRECTION = Vector3(
    -1.0,
    0.0,
    0.0,
)


# =============================================================================
# Read-only helpers
# =============================================================================


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {}
        if value is None
        else dict(value)
    )


# =============================================================================
# External validation contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class SailingYachtCrewExpectedLabels:
    """
    Evaluator-side labels only.

    They are intentionally not passed into the solver.

    control_effort_indicator:
        channel expected to expose compensation before large course deviation

    regulated_output:
        yacht observable kept near target by human control

    living_controller_channels:
        explicit human-controller state variables

    mechanical_plant_channels:
        yacht/environment state variables
    """

    control_effort_indicator: str
    regulated_output: str
    living_controller_channels: tuple[str, ...]
    mechanical_plant_channels: tuple[str, ...]
    coupled_control_expected: bool = True

    def as_mapping(
        self,
    ) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "control_effort_indicator": (
                    self.control_effort_indicator
                ),
                "regulated_output": self.regulated_output,
                "living_controller_channels": (
                    self.living_controller_channels
                ),
                "mechanical_plant_channels": (
                    self.mechanical_plant_channels
                ),
                "coupled_control_expected": (
                    self.coupled_control_expected
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class SailingYachtCrewValidationCase:
    case_id: str
    title: str
    pathological_system: ROIFSystem
    reference_system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: SailingYachtCrewExpectedLabels
    source_metadata: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "initial_state",
            _readonly_mapping(
                self.initial_state
            ),
        )

        object.__setattr__(
            self,
            "scenarios",
            tuple(
                self.scenarios
            ),
        )

        object.__setattr__(
            self,
            "source_metadata",
            _readonly_mapping(
                self.source_metadata
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(
                self.metadata
            ),
        )

    @property
    def channel_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            self.pathological_system.channel_ids
        )

    def scenario(
        self,
        scenario_id: str,
    ) -> CounterfactualScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario

        raise KeyError(
            scenario_id
        )


# =============================================================================
# Entity / channel builders
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
    history_factor: float,
    role: FunctionalRole,
    direction: Vector3,
    metadata: Mapping[str, Any] | None = None,
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
        metadata=(
            {}
            if metadata is None
            else dict(metadata)
        ),
    )


def _entity(
    entity_id: str,
    *,
    name: str,
    channel: FunctionalChannel,
    subsystem: str,
) -> StructuralEntity:
    return StructuralEntity(
        entity_id=entity_id,
        name=name,
        kind=EntityKind.GENERIC,
        channels=(
            channel,
        ),
        metadata={
            "validation_case": "sailing_04A",
            "subsystem": subsystem,
        },
    )


def _operator(
    operator_id: str,
    *,
    source: str,
    target: str,
    gain: float,
    metadata: Mapping[str, Any] | None = None,
) -> InfluenceOperator:
    return InfluenceOperator(
        operator_id=operator_id,
        name=operator_id,
        plane_id="coupled_yacht_crew",
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=(
            source,
        ),
        target_ids=(
            target,
        ),
        gain=gain,
        metadata=(
            {}
            if metadata is None
            else dict(metadata)
        ),
    )


# =============================================================================
# Published-topology benchmark system
# =============================================================================


def _build_system(
    *,
    stressed: bool,
) -> ROIFSystem:
    """
    Build a normalized ROIF instantiation of the published yachtвЂ“crew topology.

    NOTE:
    The values below are benchmark-normalized and are not claimed to be direct
    numerical measurements from the cited papers.
    """

    if stressed:
        values = {
            "wind": (1.50, 1.10, 1.00, 1.00, 1.00),
            "sail": (2.20, 1.55, 0.97, 0.95, 0.96),
            "heel_yaw": (2.20, 1.48, 0.94, 0.91, 0.94),
            # Regulated output remains relatively contained:
            "course": (2.50, 0.62, 0.96, 0.95, 0.98),
            # Human controller absorbs hidden cost:
            "helm": (1.65, 1.28, 0.84, 0.82, 0.86),
            "trimmer": (1.80, 1.34, 0.86, 0.84, 0.88),
            "rudder": (2.10, 1.30, 0.92, 0.90, 0.93),
            "sail_trim": (2.05, 1.32, 0.91, 0.89, 0.92),
        }

        gains = {
            WIND_TO_SAIL: 1.00,
            SAIL_TO_HEEL_YAW: 0.92,
            HEEL_YAW_TO_COURSE: 0.58,
            COURSE_TO_HELM: 1.12,
            HEEL_YAW_TO_TRIMMER: 1.05,
            HELM_TO_RUDDER: 0.98,
            TRIMMER_TO_SAIL_TRIM: 0.96,
            # corrective feedback
            RUDDER_TO_COURSE: -0.78,
            SAIL_TRIM_TO_SAIL: -0.70,
        }

    else:
        values = {
            "wind": (1.50, 0.35, 1.00, 1.00, 1.00),
            "sail": (2.50, 0.70, 0.98, 0.98, 1.00),
            "heel_yaw": (2.50, 0.62, 0.98, 0.98, 1.00),
            "course": (2.50, 0.35, 0.98, 0.98, 1.00),
            "helm": (2.20, 0.55, 0.95, 0.95, 1.00),
            "trimmer": (2.20, 0.55, 0.95, 0.95, 1.00),
            "rudder": (2.30, 0.60, 0.96, 0.96, 1.00),
            "sail_trim": (2.30, 0.60, 0.96, 0.96, 1.00),
        }

        gains = {
            WIND_TO_SAIL: 0.80,
            SAIL_TO_HEEL_YAW: 0.80,
            HEEL_YAW_TO_COURSE: 0.50,
            COURSE_TO_HELM: 0.85,
            HEEL_YAW_TO_TRIMMER: 0.82,
            HELM_TO_RUDDER: 0.88,
            TRIMMER_TO_SAIL_TRIM: 0.88,
            RUDDER_TO_COURSE: -0.60,
            SAIL_TRIM_TO_SAIL: -0.58,
        }

    def unpack(
        key: str,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
    ]:
        return values[key]

    wind = _channel(
        WIND_DISTURBANCE_CHANNEL,
        entity_id="environment",
        name="Wind / wind-shift disturbance",
        capacity=unpack("wind")[0],
        load=unpack("wind")[1],
        activation=unpack("wind")[2],
        mobility=unpack("wind")[3],
        history_factor=unpack("wind")[4],
        role=FunctionalRole.MODULATOR,
        direction=FORWARD_DIRECTION,
        metadata={
            "subsystem": "environment",
            "research_grounded_role": True,
        },
    )

    sail = _channel(
        SAIL_LOAD_CHANNEL,
        entity_id="sail_system",
        name="Sail aerodynamic load",
        capacity=unpack("sail")[0],
        load=unpack("sail")[1],
        activation=unpack("sail")[2],
        mobility=unpack("sail")[3],
        history_factor=unpack("sail")[4],
        role=FunctionalRole.TRANSMITTER,
        direction=FORWARD_DIRECTION,
        metadata={
            "subsystem": "yacht_mechanical",
            "research_grounded_role": True,
        },
    )

    heel_yaw = _channel(
        HEEL_YAW_CHANNEL,
        entity_id="yacht_motion",
        name="Heel / yaw response",
        capacity=unpack("heel_yaw")[0],
        load=unpack("heel_yaw")[1],
        activation=unpack("heel_yaw")[2],
        mobility=unpack("heel_yaw")[3],
        history_factor=unpack("heel_yaw")[4],
        role=FunctionalRole.TRANSMITTER,
        direction=FORWARD_DIRECTION,
        metadata={
            "subsystem": "yacht_mechanical",
            "research_grounded_role": True,
        },
    )

    course = _channel(
        COURSE_ERROR_CHANNEL,
        entity_id="course_state",
        name="Course error",
        capacity=unpack("course")[0],
        load=unpack("course")[1],
        activation=unpack("course")[2],
        mobility=unpack("course")[3],
        history_factor=unpack("course")[4],
        role=FunctionalRole.SENSOR,
        direction=FORWARD_DIRECTION,
        metadata={
            "subsystem": "regulated_output",
            "research_grounded_role": True,
            "can_remain_small_under_compensation": True,
        },
    )

    helm = _channel(
        HELM_CONTROL_DEMAND_CHANNEL,
        entity_id="helmsman",
        name="Helmsman control demand",
        capacity=unpack("helm")[0],
        load=unpack("helm")[1],
        activation=unpack("helm")[2],
        mobility=unpack("helm")[3],
        history_factor=unpack("helm")[4],
        role=FunctionalRole.MODULATOR,
        direction=CONTROL_DIRECTION,
        metadata={
            "subsystem": "crew_living_controller",
            "research_grounded_role": True,
            "finite_control_reserve": True,
        },
    )

    trimmer = _channel(
        TRIM_CONTROL_DEMAND_CHANNEL,
        entity_id="sail_trimmer",
        name="Sail-trimmer control demand",
        capacity=unpack("trimmer")[0],
        load=unpack("trimmer")[1],
        activation=unpack("trimmer")[2],
        mobility=unpack("trimmer")[3],
        history_factor=unpack("trimmer")[4],
        role=FunctionalRole.MODULATOR,
        direction=CONTROL_DIRECTION,
        metadata={
            "subsystem": "crew_living_controller",
            "research_grounded_role": True,
            "finite_control_reserve": True,
        },
    )

    rudder = _channel(
        RUDDER_ACTION_CHANNEL,
        entity_id="rudder_actuation",
        name="Rudder action",
        capacity=unpack("rudder")[0],
        load=unpack("rudder")[1],
        activation=unpack("rudder")[2],
        mobility=unpack("rudder")[3],
        history_factor=unpack("rudder")[4],
        role=FunctionalRole.TRANSMITTER,
        direction=CORRECTIVE_DIRECTION,
        metadata={
            "subsystem": "crew_to_yacht_actuation",
            "research_grounded_role": True,
        },
    )

    sail_trim = _channel(
        SAIL_TRIM_ACTION_CHANNEL,
        entity_id="sail_trim_actuation",
        name="Sail-trim action",
        capacity=unpack("sail_trim")[0],
        load=unpack("sail_trim")[1],
        activation=unpack("sail_trim")[2],
        mobility=unpack("sail_trim")[3],
        history_factor=unpack("sail_trim")[4],
        role=FunctionalRole.TRANSMITTER,
        direction=CORRECTIVE_DIRECTION,
        metadata={
            "subsystem": "crew_to_yacht_actuation",
            "research_grounded_role": True,
        },
    )

    channels = (
        _entity(
            "environment",
            name="Wind environment",
            channel=wind,
            subsystem="environment",
        ),
        _entity(
            "sail_system",
            name="Sail system",
            channel=sail,
            subsystem="yacht",
        ),
        _entity(
            "yacht_motion",
            name="Yacht motion",
            channel=heel_yaw,
            subsystem="yacht",
        ),
        _entity(
            "course_state",
            name="Course state",
            channel=course,
            subsystem="yacht",
        ),
        _entity(
            "helmsman",
            name="Helmsman",
            channel=helm,
            subsystem="crew",
        ),
        _entity(
            "sail_trimmer",
            name="Sail trimmer",
            channel=trimmer,
            subsystem="crew",
        ),
        _entity(
            "rudder_actuation",
            name="Rudder actuation",
            channel=rudder,
            subsystem="actuation",
        ),
        _entity(
            "sail_trim_actuation",
            name="Sail trim actuation",
            channel=sail_trim,
            subsystem="actuation",
        ),
    )

    operators = (
        _operator(
            WIND_TO_SAIL,
            source=WIND_DISTURBANCE_CHANNEL,
            target=SAIL_LOAD_CHANNEL,
            gain=gains[WIND_TO_SAIL],
            metadata={
                "coupling_class": "environment_to_yacht",
            },
        ),
        _operator(
            SAIL_TO_HEEL_YAW,
            source=SAIL_LOAD_CHANNEL,
            target=HEEL_YAW_CHANNEL,
            gain=gains[SAIL_TO_HEEL_YAW],
            metadata={
                "coupling_class": "yacht_internal",
            },
        ),
        _operator(
            HEEL_YAW_TO_COURSE,
            source=HEEL_YAW_CHANNEL,
            target=COURSE_ERROR_CHANNEL,
            gain=gains[HEEL_YAW_TO_COURSE],
            metadata={
                "coupling_class": "yacht_internal",
            },
        ),
        _operator(
            COURSE_TO_HELM,
            source=COURSE_ERROR_CHANNEL,
            target=HELM_CONTROL_DEMAND_CHANNEL,
            gain=gains[COURSE_TO_HELM],
            metadata={
                "coupling_class": "yacht_to_crew",
                "human_controller_link": True,
            },
        ),
        _operator(
            HEEL_YAW_TO_TRIMMER,
            source=HEEL_YAW_CHANNEL,
            target=TRIM_CONTROL_DEMAND_CHANNEL,
            gain=gains[HEEL_YAW_TO_TRIMMER],
            metadata={
                "coupling_class": "yacht_to_crew",
                "human_controller_link": True,
            },
        ),
        _operator(
            HELM_TO_RUDDER,
            source=HELM_CONTROL_DEMAND_CHANNEL,
            target=RUDDER_ACTION_CHANNEL,
            gain=gains[HELM_TO_RUDDER],
            metadata={
                "coupling_class": "crew_to_yacht",
                "human_controller_link": True,
            },
        ),
        _operator(
            TRIMMER_TO_SAIL_TRIM,
            source=TRIM_CONTROL_DEMAND_CHANNEL,
            target=SAIL_TRIM_ACTION_CHANNEL,
            gain=gains[TRIMMER_TO_SAIL_TRIM],
            metadata={
                "coupling_class": "crew_to_yacht",
                "human_controller_link": True,
            },
        ),
        _operator(
            RUDDER_TO_COURSE,
            source=RUDDER_ACTION_CHANNEL,
            target=COURSE_ERROR_CHANNEL,
            gain=gains[RUDDER_TO_COURSE],
            metadata={
                "coupling_class": "corrective_feedback",
                "negative_feedback": True,
            },
        ),
        _operator(
            SAIL_TRIM_TO_SAIL,
            source=SAIL_TRIM_ACTION_CHANNEL,
            target=SAIL_LOAD_CHANNEL,
            gain=gains[SAIL_TRIM_TO_SAIL],
            metadata={
                "coupling_class": "corrective_feedback",
                "negative_feedback": True,
            },
        ),
    )

    return ROIFSystem(
        system_id=(
            "sailing_04A_coupled_yacht_crew_stressed"
            if stressed
            else "sailing_04A_coupled_yacht_crew_reference"
        ),
        name=(
            "Sailing Yacht + Crew Coupled Pre-Stressed System"
        ),
        entities=channels,
        planes=(
            InfluencePlane(
                plane_id="coupled_yacht_crew",
                name="Coupled yachtвЂ“crew control plane",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.CONTINUOUS,
            ),
        ),
        operators=operators,
        metadata={
            "validation_case": "sailing_04A",
            "domain": "coupled_human_mechanical",
            "external_domain_validation": True,
            "published_topology_basis": True,
            "benchmark_numeric_parameters": True,
            "pre_stressed_yacht": True,
            "pre_stressed_crew": True,
            "living_controller": True,
            "closed_loop": True,
            "negative_feedback": True,
            "human_control_inputs": (
                "steering",
                "sail_trim",
            ),
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


# =============================================================================
# Public system factories
# =============================================================================


def build_stressed_system(
) -> ROIFSystem:
    return _build_system(
        stressed=True,
    )


def build_reference_system(
) -> ROIFSystem:
    return _build_system(
        stressed=False,
    )


def build_initial_state(
) -> Mapping[str, float]:
    return MappingProxyType(
        {
            WIND_DISTURBANCE_CHANNEL: 1.0,
            SAIL_LOAD_CHANNEL: 0.0,
            HEEL_YAW_CHANNEL: 0.0,
            COURSE_ERROR_CHANNEL: 0.0,
            HELM_CONTROL_DEMAND_CHANNEL: 0.0,
            TRIM_CONTROL_DEMAND_CHANNEL: 0.0,
            RUDDER_ACTION_CHANNEL: 0.0,
            SAIL_TRIM_ACTION_CHANNEL: 0.0,
        }
    )


# =============================================================================
# Tensor / cascade
# =============================================================================


def build_tensor_config(
) -> TensorBuildConfig:
    return TensorBuildConfig(
        tensor_ceiling=100.0,
        source_reserve_exponent=0.0,
        target_reserve_exponent=0.0,
        availability_exponent=1.0,
        geometry_exponent=0.0,
        material_exponent=0.0,
        history_exponent=1.0,
        state_dependent=True,
        self_coupling_mode=SelfCouplingMode.NONE,
    )


def build_cascade_config(
) -> CascadeConfig:
    return CascadeConfig(
        steps=16,
        update_mode=CascadeUpdateMode.LEAKY,
        direction=CascadeDirection.GENERIC,
        retention=0.16,
        dissipation=0.12,
        lower_bound=-100.0,
        upper_bound=100.0,
        clip_state=False,
        convergence_tolerance=1e-10,
        convergence_patience=18,
        record_transmission_events=True,
        stop_on_failure=False,
    )


# =============================================================================
# Counterfactual benchmark scenarios
# =============================================================================


def build_validation_scenarios(
) -> tuple[
    CounterfactualScenario,
    ...,
]:
    return (
        CounterfactualScenario(
            scenario_id=SCENARIO_REDUCE_WIND_DISTURBANCE,
            name="Reduce environmental disturbance",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:wind_disturbance",
                    channel_id=WIND_DISTURBANCE_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=0.50,
                    cost=0.10,
                    safety_risk=0.00,
                    uncertainty=0.05,
                    irreversibility=0.00,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_REDUCE_SAIL_LOAD,
            name="Reduce sail load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:sail_load",
                    channel_id=SAIL_LOAD_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=0.50,
                    cost=0.20,
                    safety_risk=0.05,
                    uncertainty=0.10,
                    irreversibility=0.00,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_RELIEVE_HELM_DEMAND,
            name="Relieve helmsman control demand",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:helmsman_control_demand",
                    channel_id=HELM_CONTROL_DEMAND_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=0.50,
                    cost=0.25,
                    safety_risk=0.05,
                    uncertainty=0.10,
                    irreversibility=0.00,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id=SCENARIO_RELIEVE_TRIMMER_DEMAND,
            name="Relieve sail-trimmer control demand",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:sail_trimmer_control_demand",
                    channel_id=TRIM_CONTROL_DEMAND_CHANNEL,
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=0.50,
                    cost=0.25,
                    safety_risk=0.05,
                    uncertainty=0.10,
                    irreversibility=0.00,
                ),
            ),
        ),
    )


# =============================================================================
# Solver config
# =============================================================================


def build_solver_config(
) -> SolverConfig:
    return SolverConfig(
        mode=SolverMode.ANALYSIS_ONLY,
        scenario_generation=ScenarioGenerationMode.PROVIDED_ONLY,
        selection_policy=SelectionPolicy.BEST_SAFE,
        generated_action=CounterfactualAction.REDUCE_INCOMING_LOAD,
        generated_magnitude=0.50,
        tensor_config=build_tensor_config(),
        cascade_config=build_cascade_config(),
        root_config=RootDetectorConfig(),
        counterfactual_config=CounterfactualConfig(
            execution_mode=CounterfactualExecutionMode.FULL_CASCADE,
            objective=CounterfactualObjective.BALANCED_UTILITY,
            steps=16,
            dissipation=0.12,
            retention=0.16,
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
        require_positive_reduction=False,
        include_all_ranked_alternatives=True,
        metadata={
            "validation_case": "sailing_04A",
            "external_domain_validation": True,
            "published_topology_basis": True,
            "benchmark_numeric_parameters": True,
            "hidden_labels_used": False,
        },
    )


# =============================================================================
# Source / expected metadata
# =============================================================================


def build_source_metadata(
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "primary_sources": (
                SOURCE_SCARPONI_2007,
                SOURCE_SCARPONI_2008,
                SOURCE_ROBO_YACHT_2007,
            ),
            "published_claims_used": (
                "yacht_crew_system_modeled_as_whole",
                "yacht_equations_solved_in_time_domain",
                "crew_inputs_include_steering",
                "crew_inputs_include_sail_trim",
                "helmsman_modeled_explicitly",
                "sail_trimmers_modeled_explicitly",
                "human_decision_process_can_be_recorded_and_analyzed",
            ),
            "not_claimed_from_sources": (
                "roif_capacity_values",
                "roif_load_values",
                "roif_operator_gains",
                "roif_counterfactual_costs",
                "roif_role_labels",
            ),
        }
    )


def build_expected_labels(
) -> SailingYachtCrewExpectedLabels:
    return SailingYachtCrewExpectedLabels(
        control_effort_indicator=HELM_CONTROL_DEMAND_CHANNEL,
        regulated_output=COURSE_ERROR_CHANNEL,
        living_controller_channels=(
            HELM_CONTROL_DEMAND_CHANNEL,
            TRIM_CONTROL_DEMAND_CHANNEL,
        ),
        mechanical_plant_channels=(
            WIND_DISTURBANCE_CHANNEL,
            SAIL_LOAD_CHANNEL,
            HEEL_YAW_CHANNEL,
            COURSE_ERROR_CHANNEL,
            RUDDER_ACTION_CHANNEL,
            SAIL_TRIM_ACTION_CHANNEL,
        ),
        coupled_control_expected=True,
    )


# =============================================================================
# Complete case
# =============================================================================


def build_sailing_yacht_crew_coupled_case(
) -> SailingYachtCrewValidationCase:
    return SailingYachtCrewValidationCase(
        case_id="sailing_04A_coupled_yacht_crew",
        title=(
            "Sailing Yacht + Crew: Coupled Pre-Stressed "
            "HumanвЂ“Mechanical Control System"
        ),
        pathological_system=build_stressed_system(),
        reference_system=build_reference_system(),
        initial_state=build_initial_state(),
        tensor_config=build_tensor_config(),
        cascade_config=build_cascade_config(),
        solver_config=build_solver_config(),
        scenarios=build_validation_scenarios(),
        expected=build_expected_labels(),
        source_metadata=build_source_metadata(),
        metadata={
            "validation_stage": "04A",
            "validation_family": "external_domain_coupled_systems",
            "domain": "sailing_yacht",
            "external_domain_validation": True,
            "research_grounded_topology": True,
            "normalized_benchmark_parameters": True,
            "yacht_is_pre_stressed_dynamic_system": True,
            "crew_is_pre_stressed_dynamic_system": True,
            "crew_is_living_controller": True,
            "closed_loop_human_mechanical_coupling": True,
            "control_effort_may_precede_large_output_error": True,
            "external_expected_labels_used_by_solver": False,
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


__all__ = [
    "CONTROL_DIRECTION",
    "CORRECTIVE_DIRECTION",
    "COURSE_ERROR_CHANNEL",
    "COURSE_TO_HELM",
    "FORWARD_DIRECTION",
    "HEEL_YAW_CHANNEL",
    "HEEL_YAW_TO_COURSE",
    "HEEL_YAW_TO_TRIMMER",
    "HELM_CONTROL_DEMAND_CHANNEL",
    "HELM_TO_RUDDER",
    "RUDDER_ACTION_CHANNEL",
    "RUDDER_TO_COURSE",
    "SAIL_LOAD_CHANNEL",
    "SAIL_TO_HEEL_YAW",
    "SAIL_TRIM_ACTION_CHANNEL",
    "SAIL_TRIM_TO_SAIL",
    "SCENARIO_REDUCE_SAIL_LOAD",
    "SCENARIO_REDUCE_WIND_DISTURBANCE",
    "SCENARIO_RELIEVE_HELM_DEMAND",
    "SCENARIO_RELIEVE_TRIMMER_DEMAND",
    "SOURCE_ROBO_YACHT_2007",
    "SOURCE_SCARPONI_2007",
    "SOURCE_SCARPONI_2008",
    "SailingYachtCrewExpectedLabels",
    "SailingYachtCrewValidationCase",
    "TRIMMER_TO_SAIL_TRIM",
    "TRIM_CONTROL_DEMAND_CHANNEL",
    "WIND_DISTURBANCE_CHANNEL",
    "WIND_TO_SAIL",
    "build_cascade_config",
    "build_expected_labels",
    "build_initial_state",
    "build_reference_system",
    "build_sailing_yacht_crew_coupled_case",
    "build_solver_config",
    "build_source_metadata",
    "build_stressed_system",
    "build_tensor_config",
    "build_validation_scenarios",
]

