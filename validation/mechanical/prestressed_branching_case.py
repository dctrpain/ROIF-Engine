r"""
ROIF Validation Suite
Mechanical Scenario 02A:
Pre-Stressed Branching Load Network вЂ” Known Graph Validation

Purpose
-------
This scenario is the first explicitly non-biological validation family
for ROIF.

The system is a synthetic pre-stressed mechanical load-transfer network
with:

- one upstream preload defect;
- two parallel transmission branches;
- asymmetric branch reserve;
- a downstream load-sharing junction;
- an observable terminal displacement;
- a weak feedback path.

Topology
--------

    anchor_preload_loss
          /       \
         v         v
 primary_branch   bypass_branch
         \         /
          v       v
        load_sharing_junction
                |
                v
         output_alignment
                |
                v
      terminal_displacement
                |
                +------ feedback ------> load_sharing_junction

This is intentionally not an anatomical model.

Scenario 02A asks:

    Given a sufficiently known branching mechanical graph,
    can ROIF separate D_origin, D_fast, D_root, and Node*?

Later Scenario 02B/02C cases may deliberately hide structural relations
and test Active Probe reconstruction without changing the ROIF kernel.
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


# ---------------------------------------------------------------------------
# Immutable validation value objects
# ---------------------------------------------------------------------------


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class MechanicalValidationLabels:
    """
    External evaluator-side causal-role labels.

    These labels must never be supplied to the ROIF Solver.
    They are used only after inference for validation comparison.
    """

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
class PrestressedBranchingValidationCase:
    case_id: str
    title: str
    pathological_system: ROIFSystem
    reference_system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: MechanicalValidationLabels
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

        raise KeyError(scenario_id)


# ---------------------------------------------------------------------------
# ROIF entity builders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mechanical graph
# ---------------------------------------------------------------------------


def _build_system(
    *,
    pathological: bool,
) -> ROIFSystem:
    """
    Build the pre-stressed branching network.

    The pathological state represents partial loss of anchor preload.
    The primary branch loses reserve first, the bypass branch carries
    compensatory load, and the convergent junction becomes the dominant
    structural mediator of downstream amplification.
    """

    if pathological:
        values = {
            # Upstream defect.
            "anchor_load": 0.92,

            # Primary path: low reserve / early degradation.
            "primary_capacity": 1.55,
            "primary_load": 1.34,
            "primary_activation": 0.40,
            "primary_mobility": 0.30,
            "primary_history": 0.60,

            # Parallel bypass: compensation remains available.
            "bypass_capacity": 2.60,
            "bypass_load": 1.72,
            "bypass_activation": 0.82,
            "bypass_mobility": 0.72,
            "bypass_history": 0.88,

            # Merge node: high transmitted burden.
            "junction_capacity": 3.10,
            "junction_load": 2.72,
            "junction_activation": 0.76,
            "junction_mobility": 0.58,
            "junction_history": 0.78,

            # Downstream observable chain.
            "alignment_capacity": 3.30,
            "alignment_load": 2.54,
            "alignment_activation": 0.80,
            "alignment_mobility": 0.60,
            "alignment_history": 0.84,

            "terminal_capacity": 3.00,
            "terminal_load": 2.48,
            "terminal_activation": 0.78,
            "terminal_mobility": 0.52,
            "terminal_history": 0.82,
        }
    else:
        values = {
            "anchor_load": 0.10,

            "primary_capacity": 3.00,
            "primary_load": 0.72,
            "primary_activation": 0.96,
            "primary_mobility": 0.95,
            "primary_history": 1.00,

            "bypass_capacity": 3.00,
            "bypass_load": 0.70,
            "bypass_activation": 0.95,
            "bypass_mobility": 0.94,
            "bypass_history": 1.00,

            "junction_capacity": 4.20,
            "junction_load": 1.10,
            "junction_activation": 0.96,
            "junction_mobility": 0.94,
            "junction_history": 1.00,

            "alignment_capacity": 4.00,
            "alignment_load": 0.90,
            "alignment_activation": 0.96,
            "alignment_mobility": 0.95,
            "alignment_history": 1.00,

            "terminal_capacity": 4.00,
            "terminal_load": 0.75,
            "terminal_activation": 0.96,
            "terminal_mobility": 0.95,
            "terminal_history": 1.00,
        }

    anchor_preload_loss = _channel(
        "anchor_preload_loss",
        entity_id="anchor",
        name="Anchor preload loss",
        capacity=1.0,
        load=values["anchor_load"],
        activation=1.0,
        mobility=1.0,
        role=FunctionalRole.PRIME_MOVER,
        direction=Vector3(1.0, 0.0, 0.0),
    )

    primary_branch_stiffness = _channel(
        "primary_branch_stiffness",
        entity_id="primary_branch",
        name="Primary branch stiffness reserve",
        capacity=values["primary_capacity"],
        load=values["primary_load"],
        activation=values["primary_activation"],
        mobility=values["primary_mobility"],
        history_factor=values["primary_history"],
        role=FunctionalRole.TRANSMITTER,
        direction=Vector3(1.0, 0.25, 0.0),
    )

    bypass_branch_stiffness = _channel(
        "bypass_branch_stiffness",
        entity_id="bypass_branch",
        name="Bypass branch stiffness reserve",
        capacity=values["bypass_capacity"],
        load=values["bypass_load"],
        activation=values["bypass_activation"],
        mobility=values["bypass_mobility"],
        history_factor=values["bypass_history"],
        role=FunctionalRole.COMPENSATOR,
        direction=Vector3(1.0, -0.25, 0.0),
    )

    load_sharing_junction = _channel(
        "load_sharing_junction",
        entity_id="junction",
        name="Load-sharing junction",
        capacity=values["junction_capacity"],
        load=values["junction_load"],
        activation=values["junction_activation"],
        mobility=values["junction_mobility"],
        history_factor=values["junction_history"],
        role=FunctionalRole.STABILIZER,
        direction=Vector3(1.0, 0.0, 0.0),
    )

    output_alignment = _channel(
        "output_alignment",
        entity_id="output_stage",
        name="Output alignment",
        capacity=values["alignment_capacity"],
        load=values["alignment_load"],
        activation=values["alignment_activation"],
        mobility=values["alignment_mobility"],
        history_factor=values["alignment_history"],
        role=FunctionalRole.COMPENSATOR,
        direction=Vector3(1.0, 0.0, 0.0),
    )

    terminal_displacement = _channel(
        "terminal_displacement",
        entity_id="terminal",
        name="Terminal displacement tolerance",
        capacity=values["terminal_capacity"],
        load=values["terminal_load"],
        activation=values["terminal_activation"],
        mobility=values["terminal_mobility"],
        history_factor=values["terminal_history"],
        role=FunctionalRole.SENSOR,
        direction=Vector3(1.0, 0.0, 0.0),
    )

    system_id = (
        "prestressed_branching_pathological"
        if pathological
        else "prestressed_branching_reference"
    )

    return ROIFSystem(
        system_id=system_id,
        name=(
            "Pre-stressed branching mechanical network"
            if pathological
            else "Restored branching mechanical reference"
        ),
        entities=(
            _entity(
                "anchor",
                name="Preload anchor",
                channels=(anchor_preload_loss,),
            ),
            _entity(
                "primary_branch",
                name="Primary transmission branch",
                channels=(primary_branch_stiffness,),
            ),
            _entity(
                "bypass_branch",
                name="Parallel bypass branch",
                channels=(bypass_branch_stiffness,),
            ),
            _entity(
                "junction",
                name="Load-sharing junction",
                channels=(load_sharing_junction,),
            ),
            _entity(
                "output_stage",
                name="Output alignment stage",
                channels=(output_alignment,),
            ),
            _entity(
                "terminal",
                name="Terminal response",
                channels=(terminal_displacement,),
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
            # Upstream branching.
            _operator(
                "anchor_to_primary",
                source="anchor_preload_loss",
                target="primary_branch_stiffness",
                gain=0.95,
            ),
            _operator(
                "anchor_to_bypass",
                source="anchor_preload_loss",
                target="bypass_branch_stiffness",
                gain=0.45,
            ),

            # Parallel paths converge.
            _operator(
                "primary_to_junction",
                source="primary_branch_stiffness",
                target="load_sharing_junction",
                gain=0.90,
            ),
            _operator(
                "bypass_to_junction",
                source="bypass_branch_stiffness",
                target="load_sharing_junction",
                gain=0.60,
            ),

            # Downstream propagation.
            _operator(
                "junction_to_alignment",
                source="load_sharing_junction",
                target="output_alignment",
                gain=0.85,
            ),
            _operator(
                "alignment_to_terminal",
                source="output_alignment",
                target="terminal_displacement",
                gain=0.80,
            ),

            # Weak recurrence.
            _operator(
                "terminal_feedback_to_junction",
                source="terminal_displacement",
                target="load_sharing_junction",
                gain=0.20,
            ),
        ),
        metadata={
            "validation_case": "mechanical_02A",
            "domain": "mechanical",
            "topology": "branching_convergent_feedback",
            "pre_stressed": True,
            "state": (
                "pathological"
                if pathological
                else "reference"
            ),
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
        },
    )


# ---------------------------------------------------------------------------
# Public system factories
# ---------------------------------------------------------------------------


def build_pathological_system() -> ROIFSystem:
    return _build_system(
        pathological=True,
    )


def build_reference_system() -> ROIFSystem:
    return _build_system(
        pathological=False,
    )


def build_initial_state() -> Mapping[str, float]:
    """
    External disturbance seed.

    The upstream preload-loss channel is activated first. All other
    channels start without injected cascade activity.
    """

    return MappingProxyType(
        {
            "anchor_preload_loss": 1.0,
            "primary_branch_stiffness": 0.0,
            "bypass_branch_stiffness": 0.0,
            "load_sharing_junction": 0.0,
            "output_alignment": 0.0,
            "terminal_displacement": 0.0,
        }
    )


# ---------------------------------------------------------------------------
# Tensor and cascade configuration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Counterfactual validation scenarios
# ---------------------------------------------------------------------------


def build_validation_scenarios() -> tuple[
    CounterfactualScenario,
    ...
]:
    """
    Provide evaluator-neutral intervention alternatives.

    The scenarios deliberately include upstream, branch-level,
    junction-level, and terminal actions so Node* and counterfactual
    ranking are not forced by a single available choice.
    """

    return (
        CounterfactualScenario(
            scenario_id="remove_anchor_defect_influence",
            name="Remove anchor defect influence",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="remove:anchor_preload_loss",
                    channel_id="anchor_preload_loss",
                    action=(
                        CounterfactualAction.REMOVE_OUTGOING_INFLUENCE
                    ),
                    magnitude=1.0,
                    cost=0.20,
                    safety_risk=0.05,
                    uncertainty=0.10,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="restore_primary_branch",
            name="Restore primary branch reserve",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="restore:primary_branch_stiffness",
                    channel_id="primary_branch_stiffness",
                    action=CounterfactualAction.RESTORE_CHANNEL,
                    magnitude=1.0,
                    cost=0.25,
                    safety_risk=0.05,
                    uncertainty=0.15,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="reduce_junction_load",
            name="Reduce junction load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:load_sharing_junction",
                    channel_id="load_sharing_junction",
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.40,
                    safety_risk=0.10,
                    uncertainty=0.20,
                    irreversibility=0.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="reduce_terminal_load",
            name="Reduce terminal displacement load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:terminal_displacement",
                    channel_id="terminal_displacement",
                    action=CounterfactualAction.REDUCE_INCOMING_LOAD,
                    magnitude=1.0,
                    cost=0.45,
                    safety_risk=0.10,
                    uncertainty=0.25,
                    irreversibility=0.0,
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


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
            "validation_case": "mechanical_02A",
            "domain": "mechanical",
            "navigator_only": True,
        },
    )


# ---------------------------------------------------------------------------
# External validation labels
# ---------------------------------------------------------------------------


def build_expected_labels() -> MechanicalValidationLabels:
    """
    Independent structural hypothesis for Scenario 02A.

    These are validation labels, not Solver inputs.

    D_origin
        Upstream defect from which the cascade originates.

    D_fast
        Primary branch whose reserve is designed to collapse first.

    D_root
        Convergent junction mediating both parallel paths and downstream
        propagation.

    Node*
        Primary branch reserve is the intended local control hypothesis;
        this remains subject to counterfactual validation by the engine.
    """

    return MechanicalValidationLabels(
        d_origin="anchor_preload_loss",
        d_fast="primary_branch_stiffness",
        d_root="load_sharing_junction",
        node_star="primary_branch_stiffness",
    )


# ---------------------------------------------------------------------------
# Complete Scenario 02A factory
# ---------------------------------------------------------------------------


def build_prestressed_branching_case() -> (
    PrestressedBranchingValidationCase
):
    return PrestressedBranchingValidationCase(
        case_id="mechanical_02A_prestressed_branching",
        title=(
            "Pre-stressed branching mechanical load network"
        ),
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
            "validation_family": (
                "mechanical_02_prestressed_branching"
            ),
            "validation_stage": "02A",
            "evidence_type": (
                "controlled synthetic mechanical ground truth"
            ),
            "topology": "branching_convergent_feedback",
            "pre_stressed": True,
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


__all__ = [
    "MechanicalValidationLabels",
    "PrestressedBranchingValidationCase",
    "build_cascade_config",
    "build_expected_labels",
    "build_initial_state",
    "build_pathological_system",
    "build_prestressed_branching_case",
    "build_reference_system",
    "build_solver_config",
    "build_tensor_config",
    "build_validation_scenarios",
]

