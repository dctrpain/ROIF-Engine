"""
Mechanical Validation вЂ” Scenario 02R
ROIF Representation of the Pre-Stressed Parallel Spring Physics Benchmark

Purpose
-------
This module maps the independently validated mechanical ground truth from:

    validation.mechanical.prestressed_spring_ground_truth

into a ROIFSystem.

Important boundary
------------------
The physics benchmark is the source of truth.

ROIF does NOT generate:
- stiffness;
- force split;
- allowable force;
- failure order.

Those are computed analytically first.

ROIF receives a representation derived from those values and is then asked
to infer causal roles.

Physical system
---------------

                    в”Њв”Ђв”Ђ primary_branch в”Ђв”Ђв”ђ
load_source в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”¤                    в”њв”Ђв”Ђ junction в”Ђв”Ђ downstream_member
                    в””в”Ђв”Ђ bypass_branch в”Ђв”Ђв”Ђв”

Defective physical state:
    primary stiffness: 250 N/m
    bypass stiffness: 500 N/m
    total force: 240 N

Analytical load split:
    primary force: 80 N
    bypass force: 160 N

Allowable forces:
    primary: 190 N
    bypass: 130 N
    downstream: 300 N

Therefore:
    primary utilization = 80 / 190
    bypass utilization = 160 / 130 > 1
    downstream utilization = 240 / 300

The first physical overload is bypass_branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

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
    FastDetectionMode,
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

from validation.mechanical.prestressed_spring_ground_truth import (
    MechanicalGroundTruth,
    build_physics_ground_truth,
)


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class PhysicsDerivedLabels:
    """
    Evaluator-side labels derived from the analytical mechanics.

    Only d_fast is a strict physics-ground-truth label in Scenario 02R.

    d_origin is also structurally explicit: the imposed stiffness defect is
    introduced at primary_branch_stiffness_loss.

    d_root and node_star are deliberately left as hypotheses because their
    ROIF definitions are structural/counterfactual rather than standard
    mechanical failure quantities.
    """

    d_origin: str
    d_fast: str
    d_root_hypothesis: str | None = None
    node_star_hypothesis: str | None = None

    def as_mapping(self) -> Mapping[str, str | None]:
        return MappingProxyType(
            {
                "d_origin": self.d_origin,
                "d_fast": self.d_fast,
                "d_root_hypothesis": self.d_root_hypothesis,
                "node_star_hypothesis": self.node_star_hypothesis,
            }
        )


@dataclass(frozen=True, slots=True)
class PrestressedSpringROIFCase:
    case_id: str
    title: str
    physics: MechanicalGroundTruth
    system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: PhysicsDerivedLabels
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
        return tuple(self.system.channel_ids)


def _channel(
    channel_id: str,
    *,
    entity_id: str,
    name: str,
    capacity_n: float,
    load_n: float,
    activation: float,
    mobility: float,
    history_factor: float,
    role: FunctionalRole,
) -> FunctionalChannel:
    """
    Map physical force capacity/load directly into ROIF CapacityState.

    Capacity and load remain in Newtons.
    """
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id,
        name=name,
        role=role,
        direction=Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(
            capacity=capacity_n,
            load=load_n,
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


def _stiffness_fraction(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        raise ValueError(
            "denominator must be positive."
        )
    return numerator / denominator


def build_roif_system_from_physics(
    physics: MechanicalGroundTruth | None = None,
) -> ROIFSystem:
    """
    Build ROIF channels from the defective analytical state.

    No arbitrary force capacities are introduced:
        capacity = physical allowable force
        load = analytical physical force

    Transfer gains for the two parallel branches are derived from the
    classical stiffness-weighted force split.
    """

    gt = physics or build_physics_ground_truth()

    k_primary = (
        gt.defective_primary.axial_stiffness_n_per_m
    )
    k_bypass = (
        gt.bypass.axial_stiffness_n_per_m
    )
    k_total = k_primary + k_bypass

    primary_share = _stiffness_fraction(
        k_primary,
        k_total,
    )
    bypass_share = _stiffness_fraction(
        k_bypass,
        k_total,
    )

    primary_utilization = (
        gt.defective_state.primary_utilization
    )
    bypass_utilization = (
        gt.defective_state.bypass_utilization
    )
    downstream_utilization = (
        gt.defective_state.downstream_utilization
    )

    primary = _channel(
        "primary_branch",
        entity_id="primary",
        name="Defective primary axial member",
        capacity_n=(
            gt.defective_primary.allowable_force_n
        ),
        load_n=(
            gt.defective_state.primary_force_n
        ),
        activation=1.0,
        mobility=1.0,
        history_factor=1.0,
        role=FunctionalRole.TRANSMITTER,
    )

    bypass = _channel(
        "bypass_branch",
        entity_id="bypass",
        name="Parallel bypass axial member",
        capacity_n=(
            gt.bypass.allowable_force_n
        ),
        load_n=(
            gt.defective_state.bypass_force_n
        ),
        activation=1.0,
        mobility=1.0,
        history_factor=1.0,
        role=FunctionalRole.COMPENSATOR,
    )

    junction = _channel(
        "load_sharing_junction",
        entity_id="junction",
        name="Parallel load-sharing junction",
        capacity_n=gt.load_case.total_force_n,
        load_n=gt.load_case.total_force_n,
        activation=1.0,
        mobility=1.0,
        history_factor=1.0,
        role=FunctionalRole.STABILIZER,
    )

    downstream = _channel(
        "downstream_member",
        entity_id="downstream",
        name="Downstream series member",
        capacity_n=(
            gt.downstream.allowable_force_n
        ),
        load_n=(
            gt.defective_state.downstream_force_n
        ),
        activation=1.0,
        mobility=1.0,
        history_factor=1.0,
        role=FunctionalRole.TRANSMITTER,
    )

    stiffness_loss = _channel(
        "primary_branch_stiffness_loss",
        entity_id="defect",
        name="Primary stiffness-loss driver",
        capacity_n=1.0,
        load_n=1.0,
        activation=1.0,
        mobility=1.0,
        history_factor=1.0,
        role=FunctionalRole.PRIME_MOVER,
    )

    return ROIFSystem(
        system_id="mechanical_02R_prestressed_spring_roif",
        name=(
            "ROIF representation of pre-stressed "
            "parallel spring benchmark"
        ),
        entities=(
            _entity(
                "defect",
                name="Primary stiffness defect",
                channels=(stiffness_loss,),
            ),
            _entity(
                "primary",
                name="Primary axial member",
                channels=(primary,),
            ),
            _entity(
                "bypass",
                name="Bypass axial member",
                channels=(bypass,),
            ),
            _entity(
                "junction",
                name="Load-sharing junction",
                channels=(junction,),
            ),
            _entity(
                "downstream",
                name="Downstream axial member",
                channels=(downstream,),
            ),
        ),
        planes=(
            InfluencePlane(
                plane_id="mechanical",
                name="Mechanical axial load plane",
                kind=PlaneKind.MECHANICAL,
                temporal_mode=TemporalMode.STATIC,
            ),
        ),
        operators=(
            # Stiffness loss primarily affects the primary branch.
            _operator(
                "defect_to_primary",
                source="primary_branch_stiffness_loss",
                target="primary_branch",
                gain=1.0,
            ),

            # Classical parallel force fractions:
            # primary 1/3, bypass 2/3 in defective state.
            _operator(
                "primary_to_junction",
                source="primary_branch",
                target="load_sharing_junction",
                gain=primary_share,
            ),
            _operator(
                "bypass_to_junction",
                source="bypass_branch",
                target="load_sharing_junction",
                gain=bypass_share,
            ),

            # Series member carries total junction load.
            _operator(
                "junction_to_downstream",
                source="load_sharing_junction",
                target="downstream_member",
                gain=1.0,
            ),
        ),
        metadata={
            "validation_case": "mechanical_02R",
            "domain": "mechanical",
            "physics_source": (
                "mechanical_02P_parallel_spring_ground_truth"
            ),
            "roif_generated_physics": False,
            "biological_semantics": False,
            "force_units": "N",
            "stiffness_units": "N/m",
            "primary_stiffness_n_per_m": k_primary,
            "bypass_stiffness_n_per_m": k_bypass,
            "primary_force_share": primary_share,
            "bypass_force_share": bypass_share,
            "primary_utilization": primary_utilization,
            "bypass_utilization": bypass_utilization,
            "downstream_utilization": downstream_utilization,
        },
    )


def build_initial_state(
    physics: MechanicalGroundTruth | None = None,
) -> Mapping[str, float]:
    """
    Seed the known stiffness-loss disturbance.

    This does not encode the expected D_fast answer.
    """
    _ = physics or build_physics_ground_truth()

    return MappingProxyType(
        {
            "primary_branch_stiffness_loss": 1.0,
            "primary_branch": 0.0,
            "bypass_branch": 0.0,
            "load_sharing_junction": 0.0,
            "downstream_member": 0.0,
        }
    )


def build_tensor_config() -> TensorBuildConfig:
    """
    Keep the tensor focused on state and geometry.

    Material exponent remains zero here because the physical material
    response has already been converted into the measured stiffness and
    force state of the benchmark. This avoids double-counting stiffness.
    """
    return TensorBuildConfig(
        tensor_ceiling=100.0,
        source_reserve_exponent=0.0,
        target_reserve_exponent=1.0,
        availability_exponent=0.0,
        geometry_exponent=0.0,
        material_exponent=0.0,
        history_exponent=0.0,
        state_dependent=True,
        self_coupling_mode=SelfCouplingMode.NONE,
    )


def build_cascade_config() -> CascadeConfig:
    return CascadeConfig(
        steps=8,
        update_mode=CascadeUpdateMode.LEAKY,
        direction=CascadeDirection.GENERIC,
        retention=0.0,
        dissipation=0.0,
        lower_bound=0.0,
        upper_bound=100.0,
        clip_state=False,
        convergence_tolerance=1e-12,
        convergence_patience=10,
        record_transmission_events=True,
        stop_on_failure=False,
    )


def build_validation_scenarios() -> tuple[
    CounterfactualScenario,
    ...
]:
    """
    Counterfactuals are physical hypotheses, not expected answers.
    """

    return (
        CounterfactualScenario(
            scenario_id="remove_stiffness_defect",
            name="Remove primary stiffness defect",
            interventions=(
                CounterfactualIntervention(
                    intervention_id=(
                        "remove:primary_branch_stiffness_loss"
                    ),
                    channel_id=(
                        "primary_branch_stiffness_loss"
                    ),
                    action=(
                        CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
                    ),
                    magnitude=1.0,
                    cost=0.10,
                    safety_risk=0.0,
                    uncertainty=0.0,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="restore_primary_branch",
            name="Restore primary branch",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="restore:primary_branch",
                    channel_id="primary_branch",
                    action=CounterfactualAction.RESTORE_CHANNEL,
                    magnitude=1.0,
                    cost=0.20,
                    safety_risk=0.0,
                    uncertainty=0.0,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="restore_bypass_branch",
            name="Restore bypass branch",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="restore:bypass_branch",
                    channel_id="bypass_branch",
                    action=CounterfactualAction.RESTORE_CHANNEL,
                    magnitude=1.0,
                    cost=0.20,
                    safety_risk=0.0,
                    uncertainty=0.0,
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
        generated_action=(
            CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
        ),
        generated_magnitude=1.0,
        tensor_config=build_tensor_config(),
        cascade_config=build_cascade_config(),
        root_config=RootDetectorConfig(
            fast_mode=FastDetectionMode.CAPACITY_EXCEEDANCE,
        ),
        counterfactual_config=CounterfactualConfig(
            execution_mode=CounterfactualExecutionMode.FULL_CASCADE,
            objective=CounterfactualObjective.BALANCED_UTILITY,
            steps=8,
            dissipation=0.0,
            retention=0.0,
            mark_dominated=True,
            include_baseline=True,
            deterministic=True,
        ),
        non_fonit_gate=NonFonitGate(
            enabled=True,
            scope_risk=0.0,
            externality_risk=0.0,
            propagation_uncertainty=0.0,
            human_authorization=False,
            bounded_scope=True,
            reversible=True,
        ),
        require_safe_solution=True,
        require_positive_reduction=False,
        include_all_ranked_alternatives=True,
        metadata={
            "validation_case": "mechanical_02R",
            "physics_grounded": True,
        },
    )


def build_expected_labels() -> PhysicsDerivedLabels:
    """
    Strict labels derived before ROIF inference.

    D_origin:
        imposed physical stiffness defect.

    D_fast:
        first physical overload in the analytical benchmark = bypass.

    D_root / Node*:
        intentionally not hard-coded as physics facts.
    """
    return PhysicsDerivedLabels(
        d_origin="primary_branch_stiffness_loss",
        d_fast="bypass_branch",
        d_root_hypothesis=None,
        node_star_hypothesis=None,
    )


def build_prestressed_spring_roif_case() -> PrestressedSpringROIFCase:
    physics = build_physics_ground_truth()

    return PrestressedSpringROIFCase(
        case_id="mechanical_02R_prestressed_spring_roif",
        title=(
            "ROIF representation of the pre-stressed "
            "parallel spring physics benchmark"
        ),
        physics=physics,
        system=build_roif_system_from_physics(
            physics
        ),
        initial_state=build_initial_state(
            physics
        ),
        tensor_config=build_tensor_config(),
        cascade_config=build_cascade_config(),
        solver_config=build_solver_config(),
        scenarios=build_validation_scenarios(),
        expected=build_expected_labels(),
        metadata={
            "domain": "mechanical",
            "validation_family": (
                "mechanical_02_parallel_spring"
            ),
            "validation_stage": "02R",
            "physics_ground_truth_case": physics.case_id,
            "roif_used_to_generate_ground_truth": False,
            "biological_semantics": False,
            "strict_physics_labels": (
                "d_origin",
                "d_fast",
            ),
            "exploratory_roif_labels": (
                "d_root",
                "node_star",
            ),
        },
    )


__all__ = [
    "PhysicsDerivedLabels",
    "PrestressedSpringROIFCase",
    "build_cascade_config",
    "build_expected_labels",
    "build_initial_state",
    "build_prestressed_spring_roif_case",
    "build_roif_system_from_physics",
    "build_solver_config",
    "build_tensor_config",
    "build_validation_scenarios",
]

