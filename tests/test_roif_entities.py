"""
Tests for roif.roif_entities.

The suite fixes the representation-layer contract of canonical ROIF
entities:

- immutable value objects;
- vector and geometry calculations;
- multiplicative functional output;
- Capacity / Load / Reserve invariants;
- influence planes and operators;
- phasic afferent events;
- compensation and compensator lock-in;
- repetition memory;
- cycle roots;
- distinct root roles and root migration;
- counterfactual Node* scoring;
- immutable ROIFSystem aggregation.

The module under test contains entities only. These tests do not assume a
solver, clinical diagnosis, or treatment logic.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
import math

import pytest

from roif.roif_entities import (
    ActivationAvailability,
    CapacityState,
    CascadeEvent,
    ChannelState,
    CompensationLink,
    CompensatorLockIn,
    CompositionKind,
    CounterfactualIntervention,
    CounterfactualResult,
    CycleRoot,
    EntityKind,
    FunctionalChannel,
    FunctionalRole,
    GeometryState,
    InfluenceOperator,
    InfluencePlane,
    InterventionKind,
    OperatorComposition,
    OperatorKind,
    PhasicAfferentEvent,
    PlaneKind,
    ROIFEntityError,
    ROIFSystem,
    RepetitionMemory,
    RootAssignment,
    RootMigration,
    RootRole,
    RootTargetKind,
    StructuralEntity,
    TemporalMode,
    Vector3,
    new_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_channel(
    *,
    channel_id: str = "glute_max_extension",
    entity_id: str = "glute_max",
    name: str = "Gluteus maximus extension channel",
    capacity: float = 10.0,
    load: float = 2.0,
    direction: Vector3 | None = None,
    geometry: GeometryState | None = None,
    activation: ActivationAvailability | None = None,
    baseline_output: float = 1.0,
    history_factor: float = 1.0,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id,
        name=name,
        role=FunctionalRole.PRIME_MOVER,
        state=ChannelState.AVAILABLE,
        direction=direction or Vector3(1.0, 0.0, 0.0),
        capacity_state=CapacityState(
            capacity=capacity,
            load=load,
        ),
        geometry=geometry or GeometryState(),
        activation=activation or ActivationAvailability(),
        baseline_output=baseline_output,
        history_factor=history_factor,
        metadata={"domain": "biomechanics"},
    )


def make_entity() -> StructuralEntity:
    return StructuralEntity(
        entity_id="glute_max",
        name="Gluteus maximus",
        kind=EntityKind.MUSCLE,
        channels=(make_channel(),),
        metadata={"side": "right"},
    )


# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------


def test_vector_creation_and_magnitude() -> None:
    vector = Vector3(3.0, 4.0, 0.0)

    assert vector.magnitude == pytest.approx(5.0)


def test_vector_normalization() -> None:
    vector = Vector3(3.0, 4.0, 0.0).normalized()

    assert vector.magnitude == pytest.approx(1.0)
    assert vector.x == pytest.approx(0.6)
    assert vector.y == pytest.approx(0.8)


def test_zero_vector_normalization_is_safe() -> None:
    assert Vector3().normalized() == Vector3()


def test_vector_scaling() -> None:
    assert Vector3(1.0, -2.0, 3.0).scaled(2.0) == Vector3(
        2.0,
        -4.0,
        6.0,
    )


def test_vector_addition() -> None:
    assert Vector3(1.0, 2.0, 3.0) + Vector3(
        4.0,
        5.0,
        6.0,
    ) == Vector3(5.0, 7.0, 9.0)


def test_vector_is_immutable() -> None:
    vector = Vector3(1.0, 2.0, 3.0)

    with pytest.raises(FrozenInstanceError):
        vector.x = 9.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), True],
)
def test_vector_rejects_invalid_components(value: object) -> None:
    with pytest.raises(ROIFEntityError):
        Vector3(value, 0.0, 0.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GeometryState
# ---------------------------------------------------------------------------


def test_default_geometry_has_full_efficiency() -> None:
    geometry = GeometryState()

    assert geometry.geometric_efficiency == pytest.approx(1.0)


def test_nonoptimal_length_reduces_geometric_efficiency() -> None:
    geometry = GeometryState(
        length=0.5,
        optimal_length=1.0,
    )

    assert 0.0 < geometry.geometric_efficiency < 1.0


def test_reduced_moment_arm_reduces_efficiency() -> None:
    geometry = GeometryState(
        moment_arm=0.5,
        optimal_moment_arm=1.0,
    )

    assert geometry.geometric_efficiency == pytest.approx(0.5)


def test_mobility_and_stability_multiply_geometry() -> None:
    geometry = GeometryState(
        mobility=0.5,
        stability=0.4,
    )

    assert geometry.geometric_efficiency == pytest.approx(0.2)


def test_geometry_metadata_is_read_only() -> None:
    geometry = GeometryState(metadata={"joint": "hip"})

    assert isinstance(geometry.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        geometry.metadata["joint"] = "knee"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("length", 0.0),
        ("optimal_length", -1.0),
        ("moment_arm", 0.0),
        ("optimal_moment_arm", -1.0),
        ("mobility", -0.1),
        ("mobility", 1.1),
        ("stability", -0.1),
        ("stability", 1.1),
    ],
)
def test_geometry_rejects_invalid_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ROIFEntityError):
        GeometryState(**{field_name: value})


# ---------------------------------------------------------------------------
# ActivationAvailability
# ---------------------------------------------------------------------------


def test_default_activation_is_fully_available() -> None:
    activation = ActivationAvailability()

    assert activation.effective == pytest.approx(1.0)


def test_activation_factors_are_multiplicative() -> None:
    activation = ActivationAvailability(
        command=0.8,
        neural_drive=0.5,
        afferent_gate=0.5,
        timing=1.0,
        coordination=1.0,
        geometric_access=1.0,
        task_compatibility=1.0,
    )

    assert activation.effective == pytest.approx(0.2)


def test_zero_activation_factor_blocks_effective_output() -> None:
    activation = ActivationAvailability(
        afferent_gate=0.0,
    )

    assert activation.effective == pytest.approx(0.0)


@pytest.mark.parametrize(
    "field_name",
    [
        "command",
        "neural_drive",
        "afferent_gate",
        "timing",
        "coordination",
        "geometric_access",
        "task_compatibility",
    ],
)
def test_activation_rejects_values_outside_unit_interval(
    field_name: str,
) -> None:
    with pytest.raises(ROIFEntityError):
        ActivationAvailability(**{field_name: 1.1})


# ---------------------------------------------------------------------------
# CapacityState
# ---------------------------------------------------------------------------


def test_capacity_reserve_and_utilization() -> None:
    state = CapacityState(
        capacity=10.0,
        load=4.0,
    )

    assert state.reserve == pytest.approx(6.0)
    assert state.reserve_fraction == pytest.approx(0.6)
    assert state.utilization == pytest.approx(0.4)
    assert state.is_critical is False


def test_capacity_becomes_critical_at_threshold() -> None:
    state = CapacityState(
        capacity=10.0,
        load=9.0,
        critical_reserve_fraction=0.1,
    )

    assert state.reserve_fraction == pytest.approx(0.1)
    assert state.is_critical is True


def test_overload_produces_negative_reserve() -> None:
    state = CapacityState(
        capacity=10.0,
        load=12.0,
    )

    assert state.reserve == pytest.approx(-2.0)
    assert state.reserve_fraction == pytest.approx(-0.2)
    assert state.utilization == pytest.approx(1.2)
    assert state.is_critical is True


@pytest.mark.parametrize(
    ("capacity", "load"),
    [
        (0.0, 0.0),
        (-1.0, 0.0),
        (1.0, -0.1),
    ],
)
def test_capacity_rejects_invalid_values(
    capacity: float,
    load: float,
) -> None:
    with pytest.raises(ROIFEntityError):
        CapacityState(
            capacity=capacity,
            load=load,
        )


# ---------------------------------------------------------------------------
# FunctionalChannel
# ---------------------------------------------------------------------------


def test_functional_channel_creation() -> None:
    channel = make_channel()

    assert channel.role is FunctionalRole.PRIME_MOVER
    assert channel.state is ChannelState.AVAILABLE
    assert channel.capacity_state.reserve_fraction == pytest.approx(0.8)


def test_functional_output_is_multiplicative() -> None:
    channel = make_channel(
        capacity=10.0,
        load=5.0,
        activation=ActivationAvailability(
            command=0.8,
            neural_drive=0.5,
        ),
        geometry=GeometryState(
            mobility=0.5,
            stability=1.0,
        ),
        baseline_output=2.0,
        history_factor=0.5,
    )

    expected = 2.0 * 0.4 * 0.5 * 0.5 * 0.5

    assert channel.effective_scalar_output == pytest.approx(
        expected
    )


def test_channel_vector_output_follows_direction() -> None:
    channel = make_channel(
        direction=Vector3(0.0, 3.0, 4.0),
    )
    output = channel.effective_vector_output

    assert output.magnitude == pytest.approx(
        channel.effective_scalar_output
    )
    assert output.y / output.z == pytest.approx(3.0 / 4.0)


def test_negative_reserve_clamps_output_to_zero() -> None:
    channel = make_channel(
        capacity=10.0,
        load=12.0,
    )

    assert channel.effective_scalar_output == pytest.approx(0.0)


def test_channel_metadata_is_read_only() -> None:
    channel = make_channel()

    assert isinstance(channel.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        channel.metadata["domain"] = "changed"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("channel_id", ""),
        ("entity_id", " "),
        ("name", ""),
    ],
)
def test_channel_rejects_empty_identifiers(
    field_name: str,
    value: str,
) -> None:
    kwargs = {
        "channel_id": "channel",
        "entity_id": "entity",
        "name": "name",
    }
    kwargs[field_name] = value

    with pytest.raises(ROIFEntityError):
        FunctionalChannel(**kwargs)


def test_channel_rejects_negative_output_factor() -> None:
    with pytest.raises(ROIFEntityError):
        make_channel(
            baseline_output=-1.0,
        )


# ---------------------------------------------------------------------------
# StructuralEntity
# ---------------------------------------------------------------------------


def test_structural_entity_contains_channels() -> None:
    entity = make_entity()

    assert entity.kind is EntityKind.MUSCLE
    assert entity.channels[0].entity_id == entity.entity_id


def test_structural_entity_channels_are_tuple_normalized() -> None:
    channel = make_channel()
    entity = StructuralEntity(
        entity_id="glute_max",
        name="Gluteus maximus",
        channels=[channel],  # type: ignore[arg-type]
    )

    assert isinstance(entity.channels, tuple)


def test_structural_entity_rejects_duplicate_channel_ids() -> None:
    channel = make_channel()

    with pytest.raises(ROIFEntityError):
        StructuralEntity(
            entity_id="glute_max",
            name="Gluteus maximus",
            channels=(channel, channel),
        )


def test_structural_entity_rejects_foreign_channel() -> None:
    channel = make_channel(
        entity_id="other_entity",
    )

    with pytest.raises(ROIFEntityError):
        StructuralEntity(
            entity_id="glute_max",
            name="Gluteus maximus",
            channels=(channel,),
        )


# ---------------------------------------------------------------------------
# Influence planes and operators
# ---------------------------------------------------------------------------


def test_influence_plane_creation() -> None:
    plane = InfluencePlane(
        plane_id="mechanical",
        name="Mechanical plane",
        kind=PlaneKind.MECHANICAL,
        temporal_mode=TemporalMode.CONTINUOUS,
        intensity=0.8,
        timescale=2.0,
    )

    assert plane.kind is PlaneKind.MECHANICAL
    assert plane.intensity == pytest.approx(0.8)


def test_plane_metadata_is_read_only() -> None:
    plane = InfluencePlane(
        plane_id="afferent",
        name="Afferent plane",
        kind=PlaneKind.AFFERENT,
        metadata={"source": "Ib"},
    )

    assert isinstance(plane.metadata, MappingProxyType)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("plane_id", ""),
        ("name", ""),
        ("intensity", -0.1),
        ("timescale", 0.0),
    ],
)
def test_plane_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "plane_id": "plane",
        "name": "Plane",
        "kind": PlaneKind.MECHANICAL,
    }
    kwargs[field_name] = value

    with pytest.raises(ROIFEntityError):
        InfluencePlane(**kwargs)  # type: ignore[arg-type]


def test_influence_operator_creation() -> None:
    operator = InfluenceOperator(
        operator_id="hamstring_shortening",
        name="Hamstring shortening",
        plane_id="mechanical",
        kind=OperatorKind.SHORTEN,
        composition=CompositionKind.MULTIPLICATIVE,
        source_ids=("hamstrings",),
        target_ids=("pelvis_geometry",),
        gain=0.7,
        delay=1.0,
    )

    assert operator.kind is OperatorKind.SHORTEN
    assert operator.source_ids == ("hamstrings",)
    assert operator.target_ids == ("pelvis_geometry",)


def test_operator_sequences_are_tuple_normalized() -> None:
    operator = InfluenceOperator(
        operator_id="load_transfer",
        name="Load transfer",
        plane_id="mechanical",
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=["glute"],  # type: ignore[arg-type]
        target_ids=["hamstrings"],  # type: ignore[arg-type]
    )

    assert isinstance(operator.source_ids, tuple)
    assert isinstance(operator.target_ids, tuple)


def test_operator_rejects_negative_delay() -> None:
    with pytest.raises(ROIFEntityError):
        InfluenceOperator(
            operator_id="operator",
            name="Operator",
            plane_id="plane",
            kind=OperatorKind.DELAY,
            delay=-1.0,
        )


def test_operator_composition_requires_operators() -> None:
    with pytest.raises(ROIFEntityError):
        OperatorComposition(
            composition_id="composition",
            operator_ids=(),
            kind=CompositionKind.SEQUENTIAL,
        )


def test_operator_composition_preserves_order() -> None:
    composition = OperatorComposition(
        composition_id="cascade",
        operator_ids=(
            "shorten",
            "rotate",
            "inhibit",
        ),
        kind=CompositionKind.SEQUENTIAL,
        ordered=True,
    )

    assert composition.operator_ids == (
        "shorten",
        "rotate",
        "inhibit",
    )


# ---------------------------------------------------------------------------
# Phasic afferent events
# ---------------------------------------------------------------------------


def test_phasic_afferent_event_gate() -> None:
    event = PhasicAfferentEvent(
        event_id="attempt_1",
        channel_id="glute_max_extension",
        time=1.0,
        activation_attempt=0.8,
        active_tension=1.0,
        ib_drive=0.4,
        joint_afferent_drive=0.2,
        spindle_drive=0.1,
        inhibitory_gain=0.5,
    )

    assert event.effective_gate == pytest.approx(0.65)


def test_afferent_gate_is_clamped_at_zero() -> None:
    event = PhasicAfferentEvent(
        event_id="attempt_1",
        channel_id="channel",
        time=0.0,
        activation_attempt=1.0,
        active_tension=1.0,
        ib_drive=2.0,
        inhibitory_gain=1.0,
    )

    assert event.effective_gate == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("time", -1.0),
        ("activation_attempt", 1.1),
        ("active_tension", -1.0),
        ("ib_drive", -1.0),
    ],
)
def test_afferent_event_rejects_invalid_values(
    field_name: str,
    value: float,
) -> None:
    kwargs = {
        "event_id": "event",
        "channel_id": "channel",
        "time": 0.0,
        "activation_attempt": 1.0,
        "active_tension": 1.0,
    }
    kwargs[field_name] = value

    with pytest.raises(ROIFEntityError):
        PhasicAfferentEvent(**kwargs)


# ---------------------------------------------------------------------------
# Repetition and compensation
# ---------------------------------------------------------------------------


def test_repetition_memory_creation() -> None:
    memory = RepetitionMemory(
        memory_id="memory",
        pattern_id="gait_pattern",
        repetitions=1000,
        cumulative_exposure=500.0,
        adaptation_level=0.7,
        lock_in_level=0.8,
        reversibility=0.4,
    )

    assert memory.repetitions == 1000
    assert memory.lock_in_level == pytest.approx(0.8)


def test_repetition_memory_rejects_negative_repetitions() -> None:
    with pytest.raises(ROIFEntityError):
        RepetitionMemory(
            memory_id="memory",
            pattern_id="pattern",
            repetitions=-1,
        )


def test_compensation_link_creation() -> None:
    link = CompensationLink(
        link_id="link",
        deficient_channel_id="glute",
        compensator_channel_id="hamstrings",
        transferred_fraction=0.7,
    )

    assert link.transferred_fraction == pytest.approx(0.7)


def test_compensation_link_rejects_self_compensation() -> None:
    with pytest.raises(ROIFEntityError):
        CompensationLink(
            link_id="link",
            deficient_channel_id="same",
            compensator_channel_id="same",
            transferred_fraction=0.5,
        )


def test_compensator_lock_in_creation() -> None:
    lock = CompensatorLockIn(
        lock_id="posterior_chain_lock",
        deficient_channel_id="glute",
        compensator_channel_ids=(
            "hamstrings",
            "paravertebral",
        ),
        lock_strength=0.8,
        self_reinforcement=0.9,
        original_channel_availability=0.2,
    )

    assert lock.compensator_channel_ids == (
        "hamstrings",
        "paravertebral",
    )
    assert lock.self_reinforcement == pytest.approx(0.9)


def test_compensator_lock_in_requires_compensator() -> None:
    with pytest.raises(ROIFEntityError):
        CompensatorLockIn(
            lock_id="lock",
            deficient_channel_id="glute",
            compensator_channel_ids=(),
        )


# ---------------------------------------------------------------------------
# Cascade, cycles, roots, migration
# ---------------------------------------------------------------------------


def test_cascade_event_creation() -> None:
    event = CascadeEvent(
        event_id="event",
        time=3.0,
        operator_id="transfer_load",
        source_ids=("glute",),
        target_ids=("hamstrings",),
        magnitude=0.5,
    )

    assert event.source_ids == ("glute",)
    assert event.target_ids == ("hamstrings",)


def test_cycle_root_requires_two_members() -> None:
    with pytest.raises(ROIFEntityError):
        CycleRoot(
            cycle_id="cycle",
            member_ids=("single",),
            operator_ids=(),
            gain=1.0,
            persistence=0.5,
            closure=0.5,
        )


def test_cycle_root_creation() -> None:
    cycle = CycleRoot(
        cycle_id="posterior_chain_cycle",
        member_ids=(
            "hamstrings",
            "pelvis",
            "paravertebral",
        ),
        operator_ids=(
            "fixate_pelvis",
            "increase_lumbar_rotation",
        ),
        gain=1.2,
        persistence=0.9,
        closure=0.8,
    )

    assert cycle.self_sustaining is True
    assert cycle.persistence == pytest.approx(0.9)


@pytest.mark.parametrize(
    "role",
    list(RootRole),
)
def test_all_root_roles_can_be_assigned(role: RootRole) -> None:
    assignment = RootAssignment(
        assignment_id=f"assignment_{role.value}",
        role=role,
        target_kind=RootTargetKind.CHANNEL,
        target_id="channel",
        time=1.0,
        confidence=0.8,
    )

    assert assignment.role is role


def test_root_assignment_confidence_is_bounded() -> None:
    with pytest.raises(ROIFEntityError):
        RootAssignment(
            assignment_id="root",
            role=RootRole.D_ROOT_CURRENT,
            target_kind=RootTargetKind.CYCLE,
            target_id="cycle",
            time=1.0,
            confidence=1.1,
        )


def test_root_migration_creation() -> None:
    migration = RootMigration(
        migration_id="migration",
        from_assignment_id="origin_glute",
        to_assignment_id="current_hamstring_cycle",
        time=100.0,
        trigger_event_ids=("lock_in",),
        mechanism="compensator_lock_in",
        confidence=0.9,
    )

    assert migration.mechanism == "compensator_lock_in"


def test_root_migration_requires_different_assignments() -> None:
    with pytest.raises(ROIFEntityError):
        RootMigration(
            migration_id="migration",
            from_assignment_id="same",
            to_assignment_id="same",
            time=1.0,
        )


# ---------------------------------------------------------------------------
# Counterfactual Node*
# ---------------------------------------------------------------------------


def test_counterfactual_intervention_creation() -> None:
    intervention = CounterfactualIntervention(
        intervention_id="mobilize_hamstrings",
        target_kind=RootTargetKind.CHANNEL,
        target_id="hamstrings",
        kind=InterventionKind.MOBILIZE,
        magnitude=0.5,
        plane_id="mechanical",
        reversible=True,
    )

    assert intervention.kind is InterventionKind.MOBILIZE
    assert intervention.reversible is True


def test_counterfactual_result_unlock_score() -> None:
    result = CounterfactualResult(
        result_id="result",
        intervention_id="mobilize_hamstrings",
        restored_reserve=1.0,
        restored_availability=2.0,
        cycle_break_score=3.0,
        geometry_restoration=4.0,
        global_integrity_gain=5.0,
        harm_risk=0.2,
        uncertainty=0.25,
    )

    expected = 15.0 * 0.8 * 0.75

    assert result.unlock_score == pytest.approx(expected)


def test_harm_or_uncertainty_can_zero_unlock_score() -> None:
    harmful = CounterfactualResult(
        result_id="harmful",
        intervention_id="intervention",
        global_integrity_gain=10.0,
        harm_risk=1.0,
    )
    uncertain = CounterfactualResult(
        result_id="uncertain",
        intervention_id="intervention",
        global_integrity_gain=10.0,
        uncertainty=1.0,
    )

    assert harmful.unlock_score == pytest.approx(0.0)
    assert uncertain.unlock_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ROIFSystem
# ---------------------------------------------------------------------------


def test_roif_system_creation() -> None:
    entity = make_entity()
    plane = InfluencePlane(
        plane_id="mechanical",
        name="Mechanical plane",
        kind=PlaneKind.MECHANICAL,
    )
    operator = InfluenceOperator(
        operator_id="load_transfer",
        name="Load transfer",
        plane_id="mechanical",
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=("glute_max_extension",),
        target_ids=("hamstrings",),
    )
    assignment = RootAssignment(
        assignment_id="origin",
        role=RootRole.D_ORIGIN,
        target_kind=RootTargetKind.CHANNEL,
        target_id="glute_max_extension",
        time=0.0,
    )

    system = ROIFSystem(
        system_id="posterior_chain",
        name="Posterior chain model",
        entities=(entity,),
        planes=(plane,),
        operators=(operator,),
        root_assignments=(assignment,),
        metadata={"development_level": "research_prototype"},
    )

    assert system.channel_ids == ("glute_max_extension",)
    assert system.channel("glute_max_extension") == entity.channels[0]
    assert system.assignments_for(
        RootRole.D_ORIGIN
    ) == (assignment,)


def test_roif_system_normalizes_collections_to_tuples() -> None:
    system = ROIFSystem(
        system_id="system",
        name="System",
        entities=[make_entity()],  # type: ignore[arg-type]
    )

    assert isinstance(system.entities, tuple)
    assert isinstance(system.planes, tuple)
    assert isinstance(system.events, tuple)


def test_roif_system_metadata_is_read_only() -> None:
    system = ROIFSystem(
        system_id="system",
        name="System",
        metadata={"version": 1},
    )

    assert isinstance(system.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        system.metadata["version"] = 2


def test_roif_system_unknown_channel_raises_key_error() -> None:
    system = ROIFSystem(
        system_id="system",
        name="System",
        entities=(make_entity(),),
    )

    with pytest.raises(KeyError):
        system.channel("unknown")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def test_new_id_has_requested_prefix_and_is_unique() -> None:
    left = new_id("channel")
    right = new_id("channel")

    assert left.startswith("channel_")
    assert right.startswith("channel_")
    assert left != right


def test_new_id_rejects_empty_prefix() -> None:
    with pytest.raises(ROIFEntityError):
        new_id("")
