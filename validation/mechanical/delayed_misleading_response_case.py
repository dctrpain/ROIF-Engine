"""
ROIF Validation Suite
Mechanical Scenario 03E:
Delayed Misleading Response

Purpose
-------
Scenario 03E tests whether ROIF can distinguish "first response observed"
from "causally correct branch".

Topology
--------

                         /-> fast_misleading_branch ---\
    shared_probe_node                               -> common_terminal
                         +-> delayed_true_branch ------/

Core temporal conflict
----------------------
The fast branch responds earlier after a shared perturbation, but its response
is deliberately weaker as causal evidence.

The delayed branch responds later, but is the evaluator-side true branch.

This first file defines only:
- topology,
- passive mechanical transport,
- temporal response constants,
- external validation contract,
- solver/cascade configuration.

It does NOT implement Active Probe selection.

Validation hypothesis
---------------------
Observed first response:

    fast_misleading_branch

Causally correct branch:

    delayed_true_branch

Therefore:

    first_response_branch != directional_winner

The future Active Probe layer must not equate temporal precedence with causal
truth.

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
# Public identifiers
# =============================================================================


PROBE_NODE_CHANNEL = "shared_probe_node"
FAST_BRANCH_CHANNEL = "fast_misleading_branch"
DELAYED_BRANCH_CHANNEL = "delayed_true_branch"
TERMINAL_CHANNEL = "common_terminal"

PROBE_TO_FAST = "probe_to_fast_misleading"
PROBE_TO_DELAYED = "probe_to_delayed_true"
FAST_TO_TERMINAL = "fast_to_terminal"
DELAYED_TO_TERMINAL = "delayed_to_terminal"

SCENARIO_REMOVE_PROBE_NODE = "remove_shared_probe_node"
SCENARIO_REDUCE_FAST = "reduce_fast_branch_load"
SCENARIO_REDUCE_DELAYED = "reduce_delayed_branch_load"
SCENARIO_REDUCE_TERMINAL = "reduce_common_terminal_load"


# =============================================================================
# Passive scalar constants
# =============================================================================


FAST_GAIN = 0.92
DELAYED_GAIN = 0.88


# =============================================================================
# Temporal response contract
# =============================================================================

# Synthetic local Probe-response onset, seconds.
FAST_RESPONSE_DELAY_SECONDS = 0.05
DELAYED_RESPONSE_DELAY_SECONDS = 0.35

# Observation windows used later by Active Probe validation.
EARLY_OBSERVATION_WINDOW_SECONDS = 0.10
FULL_OBSERVATION_WINDOW_SECONDS = 0.50

# Both branches use the same transport axis.
CANDIDATE_EDGE_DIRECTION = Vector3(
    1.0,
    0.0,
    0.0,
)

FAST_TRANSPORT_DIRECTION = Vector3(
    1.0,
    0.0,
    0.0,
)

DELAYED_TRANSPORT_DIRECTION = Vector3(
    1.0,
    0.0,
    0.0,
)

# Response directions remain forward for this temporal control.
FAST_RESPONSE_DIRECTION = Vector3(
    1.0,
    0.0,
    0.0,
)

DELAYED_RESPONSE_DIRECTION = Vector3(
    1.0,
    0.0,
    0.0,
)


# =============================================================================
# Immutable validation value objects
# =============================================================================


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    return MappingProxyType(
        dict(value)
    )


@dataclass(frozen=True, slots=True)
class DelayedResponseValidationLabels:
    first_response_branch: str
    directional_winner: str
    terminal: str
    first_response_must_not_decide: bool = True

    def as_mapping(
        self,
    ) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "first_response_branch": self.first_response_branch,
                "directional_winner": self.directional_winner,
                "terminal": self.terminal,
                "first_response_must_not_decide": (
                    self.first_response_must_not_decide
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class DelayedMisleadingResponseValidationCase:
    case_id: str
    title: str
    pathological_system: ROIFSystem
    reference_system: ROIFSystem
    initial_state: Mapping[str, float]
    tensor_config: TensorBuildConfig
    cascade_config: CascadeConfig
    solver_config: SolverConfig
    scenarios: tuple[CounterfactualScenario, ...]
    expected: DelayedResponseValidationLabels
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
            if (
                scenario.scenario_id
                == scenario_id
            ):
                return scenario

        raise KeyError(
            scenario_id
        )


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
) -> StructuralEntity:
    return StructuralEntity(
        entity_id=entity_id,
        name=name,
        kind=EntityKind.GENERIC,
        channels=(
            channel,
        ),
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
        source_ids=(
            source,
        ),
        target_ids=(
            target,
        ),
        gain=gain,
    )


# =============================================================================
# Mechanical graph
# =============================================================================


def _build_system(
    *,
    pathological: bool,
) -> ROIFSystem:
    if pathological:
        values = {
            "probe_capacity": 1.60,
            "probe_load": 1.10,
            "probe_activation": 1.00,
            "probe_mobility": 1.00,
            "probe_history": 1.00,

            "fast_capacity": 2.70,
            "fast_load": 1.30,
            "fast_activation": 0.92,
            "fast_mobility": 0.90,
            "fast_history": 0.92,

            "delayed_capacity": 2.80,
            "delayed_load": 1.35,
            "delayed_activation": 0.91,
            "delayed_mobility": 0.89,
            "delayed_history": 0.93,

            "terminal_capacity": 3.90,
            "terminal_load": 2.00,
            "terminal_activation": 0.89,
            "terminal_mobility": 0.84,
            "terminal_history": 0.91,
        }

        gains = {
            "probe_fast": FAST_GAIN,
            "probe_delayed": DELAYED_GAIN,
            "fast_terminal": 0.83,
            "delayed_terminal": 0.88,
        }

    else:
        values = {
            "probe_capacity": 1.60,
            "probe_load": 0.20,
            "probe_activation": 1.00,
            "probe_mobility": 1.00,
            "probe_history": 1.00,

            "fast_capacity": 3.00,
            "fast_load": 0.70,
            "fast_activation": 0.95,
            "fast_mobility": 0.95,
            "fast_history": 1.00,

            "delayed_capacity": 3.00,
            "delayed_load": 0.70,
            "delayed_activation": 0.95,
            "delayed_mobility": 0.95,
            "delayed_history": 1.00,

            "terminal_capacity": 4.00,
            "terminal_load": 0.80,
            "terminal_activation": 0.95,
            "terminal_mobility": 0.95,
            "terminal_history": 1.00,
        }

        gains = {
            "probe_fast": 0.90,
            "probe_delayed": 0.90,
            "fast_terminal": 0.85,
            "delayed_terminal": 0.85,
        }

    probe_node = _channel(
        PROBE_NODE_CHANNEL,
        entity_id="shared_probe",
        name="Shared probe node",
        capacity=values[
            "probe_capacity"
        ],
        load=values[
            "probe_load"
        ],
        activation=values[
            "probe_activation"
        ],
        mobility=values[
            "probe_mobility"
        ],
        history_factor=values[
            "probe_history"
        ],
        role=FunctionalRole.MODULATOR,
        direction=CANDIDATE_EDGE_DIRECTION,
        metadata={
            "validation_case": "mechanical_03E",
        },
    )

    fast_branch = _channel(
        FAST_BRANCH_CHANNEL,
        entity_id="fast_branch",
        name="Fast misleading branch",
        capacity=values[
            "fast_capacity"
        ],
        load=values[
            "fast_load"
        ],
        activation=values[
            "fast_activation"
        ],
        mobility=values[
            "fast_mobility"
        ],
        history_factor=values[
            "fast_history"
        ],
        role=FunctionalRole.TRANSMITTER,
        direction=FAST_TRANSPORT_DIRECTION,
        metadata={
            "validation_case": "mechanical_03E",
            "probe_response_delay_seconds": (
                FAST_RESPONSE_DELAY_SECONDS
            ),
            "probe_response_direction": (
                1.0,
                0.0,
                0.0,
            ),
            "expected_first_response_label_used_by_solver": False,
        },
    )

    delayed_branch = _channel(
        DELAYED_BRANCH_CHANNEL,
        entity_id="delayed_branch",
        name="Delayed true branch",
        capacity=values[
            "delayed_capacity"
        ],
        load=values[
            "delayed_load"
        ],
        activation=values[
            "delayed_activation"
        ],
        mobility=values[
            "delayed_mobility"
        ],
        history_factor=values[
            "delayed_history"
        ],
        role=FunctionalRole.TRANSMITTER,
        direction=DELAYED_TRANSPORT_DIRECTION,
        metadata={
            "validation_case": "mechanical_03E",
            "probe_response_delay_seconds": (
                DELAYED_RESPONSE_DELAY_SECONDS
            ),
            "probe_response_direction": (
                1.0,
                0.0,
                0.0,
            ),
            "expected_directional_winner_used_by_solver": False,
        },
    )

    terminal = _channel(
        TERMINAL_CHANNEL,
        entity_id="common_terminal",
        name="Common terminal",
        capacity=values[
            "terminal_capacity"
        ],
        load=values[
            "terminal_load"
        ],
        activation=values[
            "terminal_activation"
        ],
        mobility=values[
            "terminal_mobility"
        ],
        history_factor=values[
            "terminal_history"
        ],
        role=FunctionalRole.SENSOR,
        direction=CANDIDATE_EDGE_DIRECTION,
        metadata={
            "validation_case": "mechanical_03E",
        },
    )

    return ROIFSystem(
        system_id=(
            "delayed_misleading_response_pathological"
            if pathological
            else "delayed_misleading_response_reference"
        ),
        name=(
            "Delayed misleading response mechanical system"
            if pathological
            else "Balanced reference for delayed-response control"
        ),
        entities=(
            _entity(
                "shared_probe",
                name="Shared probe node",
                channel=probe_node,
            ),
            _entity(
                "fast_branch",
                name="Fast misleading branch",
                channel=fast_branch,
            ),
            _entity(
                "delayed_branch",
                name="Delayed true branch",
                channel=delayed_branch,
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
                PROBE_TO_FAST,
                source=PROBE_NODE_CHANNEL,
                target=FAST_BRANCH_CHANNEL,
                gain=gains[
                    "probe_fast"
                ],
            ),
            _operator(
                PROBE_TO_DELAYED,
                source=PROBE_NODE_CHANNEL,
                target=DELAYED_BRANCH_CHANNEL,
                gain=gains[
                    "probe_delayed"
                ],
            ),
            _operator(
                FAST_TO_TERMINAL,
                source=FAST_BRANCH_CHANNEL,
                target=TERMINAL_CHANNEL,
                gain=gains[
                    "fast_terminal"
                ],
            ),
            _operator(
                DELAYED_TO_TERMINAL,
                source=DELAYED_BRANCH_CHANNEL,
                target=TERMINAL_CHANNEL,
                gain=gains[
                    "delayed_terminal"
                ],
            ),
        ),
        metadata={
            "validation_case": "mechanical_03E",
            "domain": "mechanical",
            "topology": "delayed_misleading_response",
            "branching": True,
            "feedback": False,
            "pre_stressed": True,
            "temporal_conflict": pathological,
            "preferred_branch_label_used": False,
            "external_first_response_label_used_by_solver": False,
            "external_directional_winner_used_by_solver": False,
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
        },
    )


# =============================================================================
# Public factories
# =============================================================================


def build_pathological_system(
) -> ROIFSystem:
    return _build_system(
        pathological=True,
    )


def build_reference_system(
) -> ROIFSystem:
    return _build_system(
        pathological=False,
    )


def build_initial_state(
) -> Mapping[str, float]:
    return MappingProxyType(
        {
            PROBE_NODE_CHANNEL: 1.0,
            FAST_BRANCH_CHANNEL: 0.0,
            DELAYED_BRANCH_CHANNEL: 0.0,
            TERMINAL_CHANNEL: 0.0,
        }
    )


# =============================================================================
# Temporal helpers
# =============================================================================


def response_delay_seconds(
    relation_id: str,
) -> float:
    if relation_id == PROBE_TO_FAST:
        return FAST_RESPONSE_DELAY_SECONDS

    if relation_id == PROBE_TO_DELAYED:
        return DELAYED_RESPONSE_DELAY_SECONDS

    raise KeyError(
        relation_id
    )


def probe_response_direction(
    relation_id: str,
) -> Vector3:
    if relation_id == PROBE_TO_FAST:
        return FAST_RESPONSE_DIRECTION

    if relation_id == PROBE_TO_DELAYED:
        return DELAYED_RESPONSE_DIRECTION

    raise KeyError(
        relation_id
    )


def is_visible_in_window(
    relation_id: str,
    *,
    window_seconds: float,
) -> bool:
    return (
        response_delay_seconds(
            relation_id
        )
        <= window_seconds
    )


# =============================================================================
# Tensor / cascade configuration
# =============================================================================


def build_tensor_config(
) -> TensorBuildConfig:
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


def build_cascade_config(
) -> CascadeConfig:
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


def build_validation_scenarios(
) -> tuple[
    CounterfactualScenario,
    ...,
]:
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
            scenario_id=SCENARIO_REDUCE_FAST,
            name="Reduce fast branch incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:fast_misleading_branch",
                    channel_id=FAST_BRANCH_CHANNEL,
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
            scenario_id=SCENARIO_REDUCE_DELAYED,
            name="Reduce delayed branch incoming load",
            interventions=(
                CounterfactualIntervention(
                    intervention_id="reduce:delayed_true_branch",
                    channel_id=DELAYED_BRANCH_CHANNEL,
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


def build_solver_config(
) -> SolverConfig:
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
            "validation_case": "mechanical_03E",
            "domain": "mechanical",
            "navigator_only": True,
            "preferred_branch_label_used": False,
            "external_first_response_label_used": False,
            "external_directional_winner_used": False,
        },
    )


# =============================================================================
# External validation contract
# =============================================================================


def build_expected_labels(
) -> DelayedResponseValidationLabels:
    return DelayedResponseValidationLabels(
        first_response_branch=FAST_BRANCH_CHANNEL,
        directional_winner=DELAYED_BRANCH_CHANNEL,
        terminal=TERMINAL_CHANNEL,
        first_response_must_not_decide=True,
    )


# =============================================================================
# Complete factory
# =============================================================================


def build_delayed_misleading_response_case(
) -> DelayedMisleadingResponseValidationCase:
    return DelayedMisleadingResponseValidationCase(
        case_id=(
            "mechanical_03E_"
            "delayed_misleading_response"
        ),
        title=(
            "Delayed misleading response "
            "mechanical control"
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
                "mechanical_03_branch_controls"
            ),
            "validation_stage": "03E",
            "evidence_type": (
                "controlled synthetic temporal-precedence conflict"
            ),
            "topology": (
                "delayed_misleading_response"
            ),
            "pre_stressed": True,
            "branching": True,
            "feedback": False,
            "temporal_conflict": True,
            "first_response_branch": "evaluator_side_only",
            "directional_winner": "evaluator_side_only",
            "fast_response_delay_seconds": (
                FAST_RESPONSE_DELAY_SECONDS
            ),
            "delayed_response_delay_seconds": (
                DELAYED_RESPONSE_DELAY_SECONDS
            ),
            "early_observation_window_seconds": (
                EARLY_OBSERVATION_WINDOW_SECONDS
            ),
            "full_observation_window_seconds": (
                FULL_OBSERVATION_WINDOW_SECONDS
            ),
            "expected_first_response_not_final_winner": True,
            "probe_response_ground_truth_used_by_solver": False,
            "biological_semantics": False,
            "hidden_labels_used_by_solver": False,
            "diagnostic_claim": False,
            "treatment_claim": False,
        },
    )


__all__ = [
    "ALIGNED_GAIN",
    "CANDIDATE_EDGE_DIRECTION",
    "DELAYED_BRANCH_CHANNEL",
    "DELAYED_GAIN",
    "DELAYED_RESPONSE_DELAY_SECONDS",
    "DELAYED_RESPONSE_DIRECTION",
    "DELAYED_TO_TERMINAL",
    "DELAYED_TRANSPORT_DIRECTION",
    "DelayedMisleadingResponseValidationCase",
    "DelayedResponseValidationLabels",
    "EARLY_OBSERVATION_WINDOW_SECONDS",
    "FAST_BRANCH_CHANNEL",
    "FAST_GAIN",
    "FAST_RESPONSE_DELAY_SECONDS",
    "FAST_RESPONSE_DIRECTION",
    "FAST_TO_TERMINAL",
    "FAST_TRANSPORT_DIRECTION",
    "FULL_OBSERVATION_WINDOW_SECONDS",
    "PROBE_NODE_CHANNEL",
    "PROBE_TO_DELAYED",
    "PROBE_TO_FAST",
    "SCENARIO_REDUCE_DELAYED",
    "SCENARIO_REDUCE_FAST",
    "SCENARIO_REDUCE_TERMINAL",
    "SCENARIO_REMOVE_PROBE_NODE",
    "TERMINAL_CHANNEL",
    "build_cascade_config",
    "build_delayed_misleading_response_case",
    "build_expected_labels",
    "build_initial_state",
    "build_pathological_system",
    "build_reference_system",
    "build_solver_config",
    "build_tensor_config",
    "build_validation_scenarios",
    "is_visible_in_window",
    "probe_response_direction",
    "response_delay_seconds",
]
