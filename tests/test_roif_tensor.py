"""
Tests for roif.roif_tensor.

This suite verifies the mathematical contract of the ROIF Capacity Tensor:

- channel-facing reserve, availability, geometry, material, history, and
  acceptance factors;
- vector alignment and prestress;
- multiplicative factorization of directed couplings;
- plane-resolved tensor assembly;
- additive, multiplicative, minimum, and maximum aggregation;
- row, column, global, and spectral normalization;
- static versus state-dependent tensors;
- spectral radius and ROIF spectral coherence eta;
- propagation, tensor comparison, summaries, immutability, and determinism.

The tests are domain-independent. Fixtures use generic functional channels,
but the same tensor contract applies to biological, engineering,
informational, and control systems.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

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
from roif.roif_influence import InfluenceContext
from roif.roif_materials import (
    MaterialDescriptor,
    MaterialFamily,
    MaterialModel,
    MaterialState,
)
from roif.roif_tensor import (
    CapacityFactorMode,
    CapacityTensor,
    ChannelTensorState,
    ROIFTensorError,
    SelfCouplingMode,
    TensorAggregation,
    TensorBuildConfig,
    TensorEntryFactors,
    TensorNormalization,
    aggregate_values,
    build_capacity_tensor,
    build_static_capacity_tensor,
    channel_acceptance_factor,
    channel_material_factor,
    channel_reserve_factor,
    channel_tensor_state,
    geometry_pair_factor,
    history_pair_factor,
    material_pair_factor,
    normalize_matrix,
    prestress_factor,
    spectral_coherence,
    spectral_radius,
    state_vector,
    tensor_difference,
    tensor_entry_factors,
    tensor_relative_change,
    tensor_summary,
    vector_alignment_factor,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_channel(
    channel_id: str,
    *,
    entity_id: str | None = None,
    capacity: float = 10.0,
    load: float = 2.0,
    direction: Vector3 = Vector3(1.0, 0.0, 0.0),
    activation: ActivationAvailability | None = None,
    geometry: GeometryState | None = None,
    history_factor: float = 1.0,
) -> FunctionalChannel:
    return FunctionalChannel(
        channel_id=channel_id,
        entity_id=entity_id or f"entity_{channel_id}",
        name=channel_id,
        role=FunctionalRole.TRANSMITTER,
        direction=direction,
        capacity_state=CapacityState(capacity=capacity, load=load),
        activation=activation or ActivationAvailability(),
        geometry=geometry or GeometryState(),
        history_factor=history_factor,
    )


def make_plane(
    *,
    plane_id: str = "mechanical",
    intensity: float = 1.0,
    active: bool = True,
    temporal_mode: TemporalMode = TemporalMode.STATIC,
) -> InfluencePlane:
    return InfluencePlane(
        plane_id=plane_id,
        name=plane_id,
        kind=PlaneKind.MECHANICAL,
        intensity=intensity,
        active=active,
        temporal_mode=temporal_mode,
    )


def make_operator(
    *,
    operator_id: str = "a_to_b",
    plane_id: str = "mechanical",
    source_ids: tuple[str, ...] = ("a",),
    target_ids: tuple[str, ...] = ("b",),
    gain: float = 0.5,
    direction: Vector3 = Vector3(),
    active: bool = True,
    delay: float = 0.0,
    condition_key: str | None = None,
    metadata: dict[str, object] | None = None,
) -> InfluenceOperator:
    return InfluenceOperator(
        operator_id=operator_id,
        name=operator_id,
        plane_id=plane_id,
        kind=OperatorKind.TRANSFER_LOAD,
        source_ids=source_ids,
        target_ids=target_ids,
        gain=gain,
        direction=direction,
        active=active,
        delay=delay,
        condition_key=condition_key,
        metadata=metadata or {},
    )


def make_system(
    *,
    channels: tuple[FunctionalChannel, ...] | None = None,
    planes: tuple[InfluencePlane, ...] | None = None,
    operators: tuple[InfluenceOperator, ...] | None = None,
) -> ROIFSystem:
    channels = channels or (
        make_channel("a", entity_id="entity_a"),
        make_channel("b", entity_id="entity_b"),
    )
    entities = tuple(
        StructuralEntity(
            entity_id=channel.entity_id,
            name=channel.entity_id,
            kind=EntityKind.GENERIC,
            channels=(channel,),
        )
        for channel in channels
    )
    return ROIFSystem(
        system_id="tensor_test_system",
        name="Tensor test system",
        entities=entities,
        planes=planes or (make_plane(),),
        operators=operators or (make_operator(),),
    )


def neutral_config(**overrides: object) -> TensorBuildConfig:
    values: dict[str, object] = {
        "tensor_ceiling": 100.0,
        "source_reserve_exponent": 0.0,
        "target_reserve_exponent": 0.0,
        "availability_exponent": 0.0,
        "geometry_exponent": 0.0,
        "material_exponent": 0.0,
        "history_exponent": 0.0,
        "self_coupling_mode": SelfCouplingMode.NONE,
        "state_dependent": False,
    }
    values.update(overrides)
    return TensorBuildConfig(**values)


# ---------------------------------------------------------------------------
# TensorBuildConfig
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_immutable() -> None:
    config = TensorBuildConfig()

    assert config.aggregation is TensorAggregation.ADDITIVE
    assert config.normalization is TensorNormalization.NONE
    assert isinstance(config.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        config.tensor_floor = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tensor_floor", -0.1),
        ("tensor_ceiling", 0.0),
        ("reserve_floor", -0.1),
        ("direction_floor", -0.1),
        ("material_floor", -0.1),
        ("default_confidence", 1.1),
        ("spectral_target", 0.0),
        ("source_reserve_exponent", -0.1),
        ("target_reserve_exponent", -0.1),
    ],
)
def test_config_rejects_invalid_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ROIFTensorError):
        TensorBuildConfig(**{field_name: value})


def test_config_rejects_floor_above_ceiling() -> None:
    with pytest.raises(ROIFTensorError):
        TensorBuildConfig(
            tensor_floor=2.0,
            tensor_ceiling=1.0,
        )


# ---------------------------------------------------------------------------
# Channel factors
# ---------------------------------------------------------------------------


def test_clamped_reserve_factor() -> None:
    channel = make_channel("a", capacity=10.0, load=2.0)

    assert channel_reserve_factor(
        channel,
        TensorBuildConfig(),
    ) == pytest.approx(0.8)


def test_overloaded_channel_has_zero_clamped_reserve() -> None:
    channel = make_channel("a", capacity=10.0, load=12.0)

    assert channel_reserve_factor(
        channel,
        TensorBuildConfig(),
    ) == pytest.approx(0.0)


def test_positive_reserve_mode_uses_floor() -> None:
    channel = make_channel("a", capacity=10.0, load=12.0)
    config = TensorBuildConfig(
        capacity_factor_mode=CapacityFactorMode.POSITIVE_RESERVE,
        reserve_floor=0.05,
    )

    assert channel_reserve_factor(channel, config) == pytest.approx(0.05)


def test_utilization_complement_mode() -> None:
    channel = make_channel("a", capacity=10.0, load=3.0)
    config = TensorBuildConfig(
        capacity_factor_mode=(
            CapacityFactorMode.UTILIZATION_COMPLEMENT
        )
    )

    assert channel_reserve_factor(channel, config) == pytest.approx(0.7)


def test_source_reserve_exponent() -> None:
    channel = make_channel("a", capacity=10.0, load=2.0)
    config = TensorBuildConfig(source_reserve_exponent=2.0)

    assert channel_reserve_factor(channel, config) == pytest.approx(0.64)


def test_target_acceptance_is_neutral_at_zero_exponent() -> None:
    channel = make_channel("a", capacity=10.0, load=9.0)

    assert channel_acceptance_factor(
        channel,
        TensorBuildConfig(target_reserve_exponent=0.0),
    ) == pytest.approx(1.0)


def test_target_acceptance_uses_reserve() -> None:
    channel = make_channel("a", capacity=10.0, load=5.0)

    assert channel_acceptance_factor(
        channel,
        TensorBuildConfig(target_reserve_exponent=2.0),
    ) == pytest.approx(0.25)


def test_channel_material_factor_defaults_to_neutral() -> None:
    assert channel_material_factor(
        "a",
        {},
        TensorBuildConfig(),
    ) == pytest.approx(1.0)


def test_channel_material_factor_uses_available_capacity() -> None:
    from roif.roif_materials import MaterialParameters

    material = MaterialModel(
        descriptor=MaterialDescriptor(
            material_id="a",
            name="A",
            family=MaterialFamily.METAL,
        ),
        parameters=MaterialParameters(
            yield_stress=1e12,
            tensile_strength=1e12,
            compressive_strength=1e12,
            shear_strength=1e12,
            failure_strain=1e12,
        ),
        state=MaterialState(
            integrity=1.0,
            damage=0.5,
        ),
    )

    assert channel_material_factor(
        "a",
        {"a": material},
        TensorBuildConfig(),
    ) == pytest.approx(0.5)


def test_channel_tensor_state_is_multiplicative() -> None:
    channel = make_channel(
        "a",
        capacity=10.0,
        load=5.0,
        activation=ActivationAvailability(command=0.5),
        geometry=GeometryState(mobility=0.5),
        history_factor=0.5,
    )

    state = channel_tensor_state(channel)

    expected = (
        state.reserve_factor
        * state.availability_factor
        * state.geometry_factor
        * state.history_factor
        * state.material_factor
    )
    assert state.effective_output == pytest.approx(expected)


def test_static_channel_state_neutralizes_dynamic_factors() -> None:
    channel = make_channel(
        "a",
        capacity=10.0,
        load=9.0,
        activation=ActivationAvailability(command=0.1),
        geometry=GeometryState(mobility=0.1),
        history_factor=0.1,
    )

    state = channel_tensor_state(
        channel,
        config=neutral_config(),
    )

    assert state.reserve_factor == pytest.approx(1.0)
    assert state.availability_factor == pytest.approx(1.0)
    assert state.geometry_factor == pytest.approx(1.0)
    assert state.history_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pair factors
# ---------------------------------------------------------------------------


def test_parallel_vectors_have_full_alignment() -> None:
    source = make_channel("a", direction=Vector3(1.0, 0.0, 0.0))
    target = make_channel("b", direction=Vector3(2.0, 0.0, 0.0))

    assert vector_alignment_factor(
        source,
        target,
        make_operator(),
        TensorBuildConfig(),
    ) == pytest.approx(1.0)


def test_orthogonal_vectors_use_direction_floor() -> None:
    source = make_channel("a", direction=Vector3(1.0, 0.0, 0.0))
    target = make_channel("b", direction=Vector3(0.0, 1.0, 0.0))
    config = TensorBuildConfig(direction_floor=0.2)

    assert vector_alignment_factor(
        source,
        target,
        make_operator(),
        config,
    ) == pytest.approx(0.2)


def test_opposed_alignment_mode() -> None:
    source = make_channel("a", direction=Vector3(1.0, 0.0, 0.0))
    target = make_channel("b", direction=Vector3(-1.0, 0.0, 0.0))
    operator = make_operator(
        metadata={"alignment_mode": "opposed"},
    )

    assert vector_alignment_factor(
        source,
        target,
        operator,
        TensorBuildConfig(),
    ) == pytest.approx(1.0)


def test_zero_vector_is_direction_neutral() -> None:
    source = make_channel("a", direction=Vector3())
    target = make_channel("b", direction=Vector3(0.0, 1.0, 0.0))

    assert vector_alignment_factor(
        source,
        target,
        make_operator(),
        TensorBuildConfig(),
    ) == pytest.approx(1.0)


def test_geometry_pair_geometric_mean() -> None:
    source = ChannelTensorState(
        channel_id="a",
        reserve_factor=1.0,
        availability_factor=1.0,
        geometry_factor=0.25,
        history_factor=1.0,
        material_factor=1.0,
        acceptance_factor=1.0,
        effective_output=0.25,
        direction=Vector3(),
    )
    target = ChannelTensorState(
        channel_id="b",
        reserve_factor=1.0,
        availability_factor=1.0,
        geometry_factor=1.0,
        history_factor=1.0,
        material_factor=1.0,
        acceptance_factor=1.0,
        effective_output=1.0,
        direction=Vector3(),
    )

    assert geometry_pair_factor(
        source,
        target,
        make_operator(),
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("source", 0.25),
        ("target", 0.8),
        ("minimum", 0.25),
        ("maximum", 0.8),
        ("product", 0.2),
        ("neutral", 1.0),
    ],
)
def test_geometry_pair_modes(
    mode: str,
    expected: float,
) -> None:
    source = ChannelTensorState(
        "a", 1.0, 1.0, 0.25, 1.0, 1.0, 1.0, 0.25, Vector3()
    )
    target = ChannelTensorState(
        "b", 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, Vector3()
    )

    assert geometry_pair_factor(
        source,
        target,
        make_operator(metadata={"geometry_mode": mode}),
    ) == pytest.approx(expected)


def test_prestress_explicit_factor() -> None:
    operator = make_operator(
        metadata={"prestress_factor": 1.5},
    )

    assert prestress_factor(
        operator,
        TensorBuildConfig(),
        InfluenceContext(),
    ) == pytest.approx(1.5)


def test_prestress_from_context() -> None:
    operator = make_operator(
        metadata={
            "prestress_state_key": "prestress.a",
            "prestress_sensitivity": 2.0,
        }
    )
    context = InfluenceContext(
        scalar_state={"prestress.a": 0.25},
    )

    assert prestress_factor(
        operator,
        TensorBuildConfig(),
        context,
    ) == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("helper", "metadata_key", "mode", "expected"),
    [
        (history_pair_factor, "history_mode", "source", 0.4),
        (history_pair_factor, "history_mode", "target", 0.9),
        (history_pair_factor, "history_mode", "product", 0.36),
        (material_pair_factor, "material_mode", "minimum", 0.4),
        (material_pair_factor, "material_mode", "maximum", 0.9),
        (material_pair_factor, "material_mode", "neutral", 1.0),
    ],
)
def test_history_and_material_pair_modes(
    helper,
    metadata_key: str,
    mode: str,
    expected: float,
) -> None:
    source = ChannelTensorState(
        "a", 1.0, 1.0, 1.0, 0.4, 0.4, 1.0, 1.0, Vector3()
    )
    target = ChannelTensorState(
        "b", 1.0, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0, Vector3()
    )

    assert helper(
        source,
        target,
        make_operator(metadata={metadata_key: mode}),
    ) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Entry factorization
# ---------------------------------------------------------------------------


def test_tensor_entry_is_product_of_all_factors() -> None:
    source = make_channel("a")
    target = make_channel("b")
    plane = make_plane()
    operator = make_operator(
        gain=0.5,
        metadata={
            "base_transmission": 0.8,
            "confidence": 0.5,
            "geometry_mode": "neutral",
            "material_mode": "neutral",
            "history_mode": "neutral",
            "alignment_mode": "neutral",
        },
    )
    config = neutral_config()
    source_state = channel_tensor_state(source, config=config)
    target_state = channel_tensor_state(target, config=config)

    entry = tensor_entry_factors(
        source,
        target,
        operator,
        plane,
        InfluenceContext(),
        source_state=source_state,
        target_state=target_state,
        config=config,
    )

    assert entry.raw_value == pytest.approx(0.8 * 0.5 * 0.5)
    assert entry.multiplicative_product == pytest.approx(entry.raw_value)


def test_inactive_operator_produces_inactive_zero_entry() -> None:
    source = make_channel("a")
    target = make_channel("b")
    config = neutral_config()

    entry = tensor_entry_factors(
        source,
        target,
        make_operator(active=False),
        make_plane(),
        InfluenceContext(),
        source_state=channel_tensor_state(source, config=config),
        target_state=channel_tensor_state(target, config=config),
        config=config,
    )

    assert entry.active is False
    assert entry.final_value == pytest.approx(0.0)
    assert entry.reason == "operator inactive"


def test_inactive_plane_produces_inactive_zero_entry() -> None:
    source = make_channel("a")
    target = make_channel("b")
    config = neutral_config()

    entry = tensor_entry_factors(
        source,
        target,
        make_operator(),
        make_plane(active=False),
        InfluenceContext(),
        source_state=channel_tensor_state(source, config=config),
        target_state=channel_tensor_state(target, config=config),
        config=config,
    )

    assert entry.active is False
    assert entry.reason == "plane inactive"


def test_operator_delay_blocks_entry() -> None:
    source = make_channel("a")
    target = make_channel("b")
    config = neutral_config()

    entry = tensor_entry_factors(
        source,
        target,
        make_operator(delay=2.0),
        make_plane(),
        InfluenceContext(time=1.0),
        source_state=channel_tensor_state(source, config=config),
        target_state=channel_tensor_state(target, config=config),
        config=config,
    )

    assert entry.active is False
    assert "delay" in entry.reason


def test_operator_condition_blocks_entry() -> None:
    source = make_channel("a")
    target = make_channel("b")
    config = neutral_config()

    entry = tensor_entry_factors(
        source,
        target,
        make_operator(condition_key="enabled"),
        make_plane(),
        InfluenceContext(conditions={"enabled": False}),
        source_state=channel_tensor_state(source, config=config),
        target_state=channel_tensor_state(target, config=config),
        config=config,
    )

    assert entry.active is False
    assert "condition" in entry.reason


def test_tensor_entry_clipping() -> None:
    source = make_channel("a")
    target = make_channel("b")
    config = neutral_config(
        tensor_ceiling=0.3,
        clip_final_tensor=True,
    )

    entry = tensor_entry_factors(
        source,
        target,
        make_operator(gain=10.0),
        make_plane(),
        InfluenceContext(),
        source_state=channel_tensor_state(source, config=config),
        target_state=channel_tensor_state(target, config=config),
        config=config,
    )

    assert entry.raw_value > 0.3
    assert entry.final_value == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Aggregation and normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        (TensorAggregation.ADDITIVE, 0.7),
        (TensorAggregation.MULTIPLICATIVE, 0.1),
        (TensorAggregation.MAXIMUM, 0.5),
        (TensorAggregation.MINIMUM, 0.2),
    ],
)
def test_aggregate_values(
    aggregation: TensorAggregation,
    expected: float,
) -> None:
    assert aggregate_values(
        0.5,
        0.2,
        aggregation,
        has_existing=True,
    ) == pytest.approx(expected)


def test_first_aggregate_value_uses_contribution_directly() -> None:
    assert aggregate_values(
        99.0,
        0.2,
        TensorAggregation.ADDITIVE,
        has_existing=False,
    ) == pytest.approx(0.2)


def test_row_normalization() -> None:
    matrix = normalize_matrix(
        np.array([[1.0, 1.0], [0.0, 2.0]]),
        TensorNormalization.ROW,
    )

    assert np.sum(np.abs(matrix[0])) == pytest.approx(1.0)
    assert np.sum(np.abs(matrix[1])) == pytest.approx(1.0)


def test_column_normalization() -> None:
    matrix = normalize_matrix(
        np.array([[1.0, 0.0], [1.0, 2.0]]),
        TensorNormalization.COLUMN,
    )

    assert np.sum(np.abs(matrix[:, 0])) == pytest.approx(1.0)
    assert np.sum(np.abs(matrix[:, 1])) == pytest.approx(1.0)


def test_global_max_normalization() -> None:
    matrix = normalize_matrix(
        np.array([[1.0, 4.0], [2.0, 0.0]]),
        TensorNormalization.GLOBAL_MAX,
    )

    assert np.max(np.abs(matrix)) == pytest.approx(1.0)


def test_spectral_normalization_targets_radius() -> None:
    matrix = normalize_matrix(
        np.array([[2.0, 0.0], [0.0, 1.0]]),
        TensorNormalization.SPECTRAL,
        spectral_target=0.8,
    )

    assert spectral_radius(matrix) == pytest.approx(0.8)


def test_normalization_rejects_non_square_matrix() -> None:
    with pytest.raises(ROIFTensorError):
        normalize_matrix(
            np.zeros((2, 3)),
            TensorNormalization.NONE,
        )


# ---------------------------------------------------------------------------
# CapacityTensor assembly
# ---------------------------------------------------------------------------


def test_build_capacity_tensor_shape_and_axis_convention() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    assert tensor.channel_ids == ("a", "b")
    assert tensor.plane_ids == ("mechanical",)
    assert tensor.shape == (1, 2, 2)
    assert tensor.coupling("a", "b") == pytest.approx(0.5)
    assert tensor.matrix[1, 0] == pytest.approx(0.5)


def test_plane_resolved_coupling_matches_aggregate_for_one_plane() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    assert tensor.plane_coupling(
        "mechanical",
        "a",
        "b",
    ) == pytest.approx(tensor.coupling("a", "b"))


def test_entity_endpoint_expands_to_its_channels() -> None:
    channel_a = make_channel("a", entity_id="source_entity")
    channel_b = make_channel("b", entity_id="target_entity")
    system = make_system(
        channels=(channel_a, channel_b),
        operators=(
            make_operator(
                source_ids=("source_entity",),
                target_ids=("target_entity",),
            ),
        ),
    )

    tensor = build_capacity_tensor(system, config=neutral_config())

    assert tensor.coupling("a", "b") == pytest.approx(0.5)


def test_operator_without_resolvable_endpoints_is_ignored() -> None:
    system = make_system(
        operators=(
            make_operator(
                source_ids=("missing",),
                target_ids=("b",),
            ),
        )
    )

    tensor = build_capacity_tensor(system, config=neutral_config())

    assert tensor.nonzero_count == 0


def test_additive_parallel_operators_are_summed() -> None:
    system = make_system(
        operators=(
            make_operator(operator_id="one", gain=0.2),
            make_operator(operator_id="two", gain=0.3),
        )
    )

    tensor = build_capacity_tensor(
        system,
        config=neutral_config(
            aggregation=TensorAggregation.ADDITIVE,
        ),
    )

    assert tensor.coupling("a", "b") == pytest.approx(0.5)


def test_maximum_parallel_operator_is_selected() -> None:
    system = make_system(
        operators=(
            make_operator(operator_id="one", gain=0.2),
            make_operator(operator_id="two", gain=0.3),
        )
    )

    tensor = build_capacity_tensor(
        system,
        config=neutral_config(
            aggregation=TensorAggregation.MAXIMUM,
        ),
    )

    assert tensor.coupling("a", "b") == pytest.approx(0.3)


def test_identity_self_coupling() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(
            self_coupling_mode=SelfCouplingMode.IDENTITY,
        ),
    )

    assert tensor.coupling("a", "a") == pytest.approx(1.0)
    assert tensor.coupling("b", "b") == pytest.approx(1.0)


def test_retention_self_coupling() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(
            self_coupling_mode=SelfCouplingMode.RETENTION,
            self_retention=0.25,
        ),
    )

    assert tensor.coupling("a", "a") == pytest.approx(0.25)
    assert tensor.coupling("b", "b") == pytest.approx(0.25)


def test_state_dependent_tensor_differs_from_static_tensor() -> None:
    channel_a = make_channel(
        "a",
        entity_id="entity_a",
        load=9.0,
        activation=ActivationAvailability(command=0.5),
    )
    system = make_system(
        channels=(
            channel_a,
            make_channel("b", entity_id="entity_b"),
        )
    )

    dynamic = build_capacity_tensor(system)
    static = build_static_capacity_tensor(system)

    assert dynamic.coupling("a", "b") < static.coupling("a", "b")


def test_tensor_build_is_deterministic() -> None:
    system = make_system()
    config = neutral_config()

    left = build_capacity_tensor(system, config=config)
    right = build_capacity_tensor(system, config=config)

    assert np.array_equal(left.matrix, right.matrix)
    assert np.array_equal(left.plane_tensor, right.plane_tensor)
    assert left.entries == right.entries


def test_capacity_tensor_arrays_are_read_only() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    assert tensor.matrix.flags.writeable is False
    assert tensor.plane_tensor.flags.writeable is False
    assert tensor.adjacency_mask.flags.writeable is False

    with pytest.raises(ValueError):
        tensor.matrix[0, 0] = 1.0


def test_capacity_tensor_rejects_bad_shape() -> None:
    with pytest.raises(ROIFTensorError):
        CapacityTensor(
            channel_ids=("a", "b"),
            plane_ids=("p",),
            plane_tensor=np.zeros((1, 2, 2)),
            matrix=np.zeros((3, 3)),
            adjacency_mask=np.zeros((3, 3), dtype=bool),
        )


def test_build_rejects_system_without_channels() -> None:
    system = ROIFSystem(
        system_id="empty",
        name="Empty",
    )

    with pytest.raises(ROIFTensorError):
        build_capacity_tensor(system)


def test_unknown_operator_plane_is_rejected_by_tensor_builder() -> None:
    system = make_system(
        planes=(make_plane(plane_id="mechanical"),),
        operators=(make_operator(plane_id="missing"),),
    )

    with pytest.raises(ROIFTensorError):
        build_capacity_tensor(system)


# ---------------------------------------------------------------------------
# Propagation and spectral metrics
# ---------------------------------------------------------------------------


def test_state_vector_from_mapping_uses_channel_order() -> None:
    vector = state_vector(
        {"b": 2.0, "a": 1.0},
        ("a", "b"),
    )

    assert np.array_equal(vector, np.array([1.0, 2.0]))
    assert vector.flags.writeable is False


def test_state_vector_missing_mapping_value_defaults_to_zero() -> None:
    vector = state_vector({"a": 1.0}, ("a", "b"))

    assert np.array_equal(vector, np.array([1.0, 0.0]))


def test_state_vector_rejects_wrong_sequence_length() -> None:
    with pytest.raises(ROIFTensorError):
        state_vector((1.0,), ("a", "b"))


def test_tensor_propagation_uses_matrix_times_state() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    result = tensor.propagate({"a": 2.0, "b": 0.0})

    assert np.array_equal(result, np.array([0.0, 1.0]))


def test_tensor_propagation_can_include_current_state() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    result = tensor.propagate(
        {"a": 2.0, "b": 0.0},
        include_current_state=True,
    )

    assert np.array_equal(result, np.array([2.0, 1.0]))


def test_tensor_propagation_clipping() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    result = tensor.propagate(
        {"a": 10.0, "b": 0.0},
        clip_max=2.0,
    )

    assert np.max(result) == pytest.approx(2.0)


def test_spectral_radius_of_diagonal_matrix() -> None:
    assert spectral_radius(
        np.diag([0.2, 0.8, 0.5])
    ) == pytest.approx(0.8)


def test_spectral_coherence_formula() -> None:
    matrix = np.diag([0.2, 0.8])

    assert spectral_coherence(
        matrix,
        lambda_critical=1.0,
    ) == pytest.approx(0.2)


def test_spectral_coherence_can_be_negative() -> None:
    assert spectral_coherence(
        np.diag([1.5, 0.1]),
        lambda_critical=1.0,
    ) == pytest.approx(-0.5)


def test_spectral_radius_rejects_empty_or_non_square_matrix() -> None:
    with pytest.raises(ROIFTensorError):
        spectral_radius(np.zeros((0, 0)))
    with pytest.raises(ROIFTensorError):
        spectral_radius(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Tensor comparison and summary
# ---------------------------------------------------------------------------


def test_tensor_difference() -> None:
    system = make_system()
    left = build_capacity_tensor(
        system,
        config=neutral_config(),
    )
    right = build_capacity_tensor(
        make_system(
            operators=(make_operator(gain=0.2),),
        ),
        config=neutral_config(),
    )

    difference = tensor_difference(left, right)

    assert difference[1, 0] == pytest.approx(0.3)
    assert difference.flags.writeable is False


def test_tensor_relative_change() -> None:
    before = build_capacity_tensor(
        make_system(
            operators=(make_operator(gain=0.2),),
        ),
        config=neutral_config(),
    )
    after = build_capacity_tensor(
        make_system(
            operators=(make_operator(gain=0.3),),
        ),
        config=neutral_config(),
    )

    change = tensor_relative_change(before, after)

    assert change[1, 0] == pytest.approx(0.5)


def test_tensor_comparison_rejects_different_channel_order() -> None:
    left = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )
    right = build_capacity_tensor(
        make_system(
            channels=(
                make_channel("b", entity_id="entity_b"),
                make_channel("a", entity_id="entity_a"),
            ),
            operators=(
                make_operator(source_ids=("a",), target_ids=("b",)),
            ),
        ),
        config=neutral_config(),
    )

    with pytest.raises(ROIFTensorError):
        tensor_difference(left, right)
    with pytest.raises(ROIFTensorError):
        tensor_relative_change(left, right)


def test_tensor_summary_is_read_only_and_consistent() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )
    summary = tensor_summary(tensor)

    assert isinstance(summary, MappingProxyType)
    assert summary["node_count"] == 2
    assert summary["plane_count"] == 1
    assert summary["nonzero_count"] == 1
    assert summary["maximum_coupling"] == pytest.approx(0.5)


def test_tensor_row_and_column_are_read_only() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    row = tensor.row("b")
    column = tensor.column("a")

    assert row.flags.writeable is False
    assert column.flags.writeable is False
    assert row[0] == pytest.approx(0.5)
    assert column[1] == pytest.approx(0.5)


def test_unknown_channel_and_plane_raise_key_error() -> None:
    tensor = build_capacity_tensor(
        make_system(),
        config=neutral_config(),
    )

    with pytest.raises(KeyError):
        tensor.index("missing")
    with pytest.raises(KeyError):
        tensor.plane_index("missing")
