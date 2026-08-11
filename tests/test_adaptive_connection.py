"""
Tests for roif.history.adaptive_connection
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.adaptive_connection import (
    AdaptiveConnectionConfig,
    AdaptiveConnectionDelta,
    AdaptiveConnectionError,
    AdaptiveConnectionResult,
    AdaptiveConnectionState,
    ConnectionExposure,
    SCHEMA_VERSION,
    adaptive_connection_is_policy_free,
    adaptive_connection_signature,
    effective_transfer_gain,
    update_adaptive_connection,
)


@pytest.fixture
def state():
    return AdaptiveConnectionState(
        connection_id="soft_A_B",
        source_node_id="A",
        target_node_id="B",
        stiffness=2.0,
        contractile_capacity=3.0,
        reflex_gain=1.5,
        fatigue=0.10,
        remodeling_bias=0.0,
        min_stiffness=0.5,
        max_stiffness=5.0,
        min_contractile_capacity=0.5,
        max_contractile_capacity=5.0,
        min_reflex_gain=0.5,
        max_reflex_gain=4.0,
    )


@pytest.fixture
def exposure():
    return ConnectionExposure(
        exposure_id="load_1",
        connection_id="soft_A_B",
        load=0.6,
        strain=0.4,
        activation=0.7,
        damage=0.2,
        recovery=0.1,
    )


@pytest.fixture
def default_result(state, exposure):
    return update_adaptive_connection(
        update_id="update_1",
        state=state,
        exposure=exposure,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "adaptive_connection_v1"


def test_state_constructs(state):
    assert state.connection_id == "soft_A_B"
    assert state.stiffness == pytest.approx(2.0)


def test_exposure_constructs(exposure):
    assert exposure.activation == pytest.approx(0.7)


def test_default_config_constructs():
    config = AdaptiveConnectionConfig()
    assert config.remodeling_rate == pytest.approx(0.05)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_connection_id_rejected(value):
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionState(
            connection_id=value,
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
        )


def test_self_connection_rejected():
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionState(
            connection_id="x",
            source_node_id="A",
            target_node_id="A",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
        )


@pytest.mark.parametrize("fatigue", [-0.1, 1.1])
def test_invalid_fatigue_rejected(fatigue):
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionState(
            connection_id="x",
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
            fatigue=fatigue,
        )


@pytest.mark.parametrize("bias", [-1.1, 1.1])
def test_invalid_remodeling_bias_rejected(bias):
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionState(
            connection_id="x",
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
            remodeling_bias=bias,
        )


def test_stiffness_out_of_bounds_rejected():
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionState(
            connection_id="x",
            source_node_id="A",
            target_node_id="B",
            stiffness=11.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
        )


def test_capacity_out_of_bounds_rejected():
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionState(
            connection_id="x",
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=11.0,
            reflex_gain=1.0,
        )


def test_reflex_gain_out_of_bounds_rejected():
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionState(
            connection_id="x",
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=11.0,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_activation_rejected(value):
    with pytest.raises(AdaptiveConnectionError):
        ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            activation=value,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_damage_rejected(value):
    with pytest.raises(AdaptiveConnectionError):
        ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            damage=value,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_recovery_rejected(value):
    with pytest.raises(AdaptiveConnectionError):
        ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            recovery=value,
        )


def test_negative_load_rejected():
    with pytest.raises(AdaptiveConnectionError):
        ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            load=-0.1,
        )


def test_negative_strain_rejected():
    with pytest.raises(AdaptiveConnectionError):
        ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            strain=-0.1,
        )


def test_negative_config_coefficient_rejected():
    with pytest.raises(AdaptiveConnectionError):
        AdaptiveConnectionConfig(
            fatigue_load_gain=-0.1,
        )


def test_connection_id_mismatch_rejected(state):
    with pytest.raises(AdaptiveConnectionError):
        update_adaptive_connection(
            update_id="bad",
            state=state,
            exposure=ConnectionExposure(
                exposure_id="e",
                connection_id="other",
            ),
        )


def test_update_returns_expected_type(default_result):
    assert isinstance(
        default_result,
        AdaptiveConnectionResult,
    )


def test_update_preserves_connection_identity(default_result):
    assert (
        default_result.target_state.connection_id
        == default_result.source_state.connection_id
    )


def test_update_preserves_topology(default_result):
    assert (
        default_result.target_state.source_node_id
        == default_result.source_state.source_node_id
    )
    assert (
        default_result.target_state.target_node_id
        == default_result.source_state.target_node_id
    )


def test_load_and_activation_raise_fatigue(state):
    result = update_adaptive_connection(
        update_id="fatigue",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
            load=0.8,
            activation=0.8,
        ),
    )
    assert result.target_state.fatigue > state.fatigue


def test_recovery_reduces_fatigue():
    state = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=1.0,
        reflex_gain=1.0,
        fatigue=0.8,
    )
    result = update_adaptive_connection(
        update_id="recovery",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            recovery=1.0,
        ),
    )
    assert result.target_state.fatigue < state.fatigue


def test_damage_reduces_contractile_capacity(state):
    result = update_adaptive_connection(
        update_id="damage",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
            damage=0.8,
        ),
    )
    assert (
        result.target_state.contractile_capacity
        < state.contractile_capacity
    )


def test_activation_increases_reflex_gain(state):
    result = update_adaptive_connection(
        update_id="reflex",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
            activation=1.0,
        ),
    )
    assert (
        result.target_state.reflex_gain
        > state.reflex_gain
    )


def test_load_can_increase_stiffness(state):
    result = update_adaptive_connection(
        update_id="stiff",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
            load=1.0,
        ),
    )
    assert (
        result.target_state.stiffness
        > state.stiffness
    )


def test_damage_can_reduce_stiffness(state):
    result = update_adaptive_connection(
        update_id="damage_stiffness",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
            damage=1.0,
        ),
    )
    assert (
        result.target_state.stiffness
        < state.stiffness
    )


def test_strengthening_exposure_can_raise_remodeling_bias(state):
    result = update_adaptive_connection(
        update_id="remodel_plus",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
            load=1.0,
            activation=1.0,
        ),
    )
    assert (
        result.target_state.remodeling_bias
        > state.remodeling_bias
    )


def test_damage_and_strain_can_lower_remodeling_bias(state):
    result = update_adaptive_connection(
        update_id="remodel_minus",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
            damage=1.0,
            strain=1.0,
        ),
    )
    assert (
        result.target_state.remodeling_bias
        < state.remodeling_bias
    )


def test_fatigue_is_bounded_at_one():
    state = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=1.0,
        reflex_gain=1.0,
        fatigue=0.95,
    )
    result = update_adaptive_connection(
        update_id="fatigue_cap",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            load=10.0,
            strain=10.0,
            activation=1.0,
        ),
    )
    assert result.target_state.fatigue == pytest.approx(1.0)


def test_stiffness_respects_upper_bound():
    state = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=0.95,
        contractile_capacity=1.0,
        reflex_gain=1.0,
        max_stiffness=1.0,
    )
    result = update_adaptive_connection(
        update_id="bound",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            load=10.0,
        ),
    )
    assert result.target_state.stiffness == pytest.approx(1.0)


def test_capacity_respects_lower_bound():
    state = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=0.55,
        reflex_gain=1.0,
        min_contractile_capacity=0.5,
    )
    result = update_adaptive_connection(
        update_id="cap_lower",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            damage=1.0,
        ),
    )
    assert (
        result.target_state.contractile_capacity
        == pytest.approx(0.5)
    )


def test_reflex_gain_respects_upper_bound():
    state = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=1.0,
        reflex_gain=0.95,
        max_reflex_gain=1.0,
    )
    result = update_adaptive_connection(
        update_id="reflex_bound",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            activation=1.0,
            damage=1.0,
        ),
    )
    assert result.target_state.reflex_gain == pytest.approx(1.0)


def test_zero_exposure_preserves_or_nearly_preserves_state(state):
    result = update_adaptive_connection(
        update_id="zero",
        state=state,
        exposure=ConnectionExposure(
            exposure_id="e",
            connection_id=state.connection_id,
        ),
    )

    assert result.target_state.fatigue == pytest.approx(
        state.fatigue
    )
    assert result.target_state.remodeling_bias == pytest.approx(
        state.remodeling_bias
    )


def test_same_exposure_different_fatigue_produces_different_capacity():
    low = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=3.0,
        reflex_gain=1.0,
        fatigue=0.0,
    )
    high = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=3.0,
        reflex_gain=1.0,
        fatigue=0.8,
    )
    exposure = ConnectionExposure(
        exposure_id="e",
        connection_id="c",
        load=0.5,
    )

    a = update_adaptive_connection(
        update_id="a",
        state=low,
        exposure=exposure,
    )
    b = update_adaptive_connection(
        update_id="b",
        state=high,
        exposure=exposure,
    )

    assert (
        a.target_state.contractile_capacity
        != b.target_state.contractile_capacity
    )


def test_delta_matches_state_difference(default_result):
    assert (
        default_result.delta.stiffness_delta
        == pytest.approx(
            default_result.target_state.stiffness
            - default_result.source_state.stiffness
        )
    )
    assert (
        default_result.delta.contractile_capacity_delta
        == pytest.approx(
            default_result.target_state.contractile_capacity
            - default_result.source_state.contractile_capacity
        )
    )
    assert (
        default_result.delta.reflex_gain_delta
        == pytest.approx(
            default_result.target_state.reflex_gain
            - default_result.source_state.reflex_gain
        )
    )
    assert (
        default_result.delta.fatigue_delta
        == pytest.approx(
            default_result.target_state.fatigue
            - default_result.source_state.fatigue
        )
    )


def test_delta_type(default_result):
    assert isinstance(
        default_result.delta,
        AdaptiveConnectionDelta,
    )


def test_effective_transfer_gain_matches_formula(state):
    expected = (
        state.stiffness
        * state.contractile_capacity
        * state.reflex_gain
        * (1.0 - state.fatigue)
    )
    assert effective_transfer_gain(
        state
    ) == pytest.approx(expected)


def test_fatigue_reduces_effective_transfer_gain():
    low = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=2.0,
        contractile_capacity=2.0,
        reflex_gain=2.0,
        fatigue=0.1,
    )
    high = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=2.0,
        contractile_capacity=2.0,
        reflex_gain=2.0,
        fatigue=0.9,
    )

    assert (
        effective_transfer_gain(low)
        > effective_transfer_gain(high)
    )


def test_zero_fatigue_maximizes_same_parameter_transfer():
    a = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=1.0,
        reflex_gain=1.0,
        fatigue=0.0,
    )
    b = AdaptiveConnectionState(
        connection_id="c",
        source_node_id="A",
        target_node_id="B",
        stiffness=1.0,
        contractile_capacity=1.0,
        reflex_gain=1.0,
        fatigue=0.5,
    )

    assert effective_transfer_gain(a) > effective_transfer_gain(b)


def test_result_metadata_schema(default_result):
    assert default_result.metadata[
        "schema_version"
    ] == SCHEMA_VERSION


def test_result_declares_no_memory_mutation(default_result):
    assert default_result.metadata[
        "memory_mutated"
    ] is False


def test_result_declares_no_learning(default_result):
    assert default_result.metadata[
        "learning_applied"
    ] is False


def test_result_declares_no_topology_mutation(default_result):
    assert default_result.metadata[
        "topology_modified"
    ] is False


def test_result_declares_no_action(default_result):
    assert default_result.metadata[
        "action_selected"
    ] is False


def test_result_declares_no_policy_modified(default_result):
    assert default_result.metadata[
        "policy_modified"
    ] is False


def test_result_declares_no_diagnosis(default_result):
    assert default_result.metadata[
        "diagnosis_generated"
    ] is False


def test_result_declares_no_biological_truth(default_result):
    assert default_result.metadata[
        "biological_truth_claimed"
    ] is False


def test_result_declares_no_causal_truth(default_result):
    assert default_result.metadata[
        "causal_truth_inferred"
    ] is False


def test_delta_declares_no_learning(default_result):
    assert default_result.delta.metadata[
        "learning_applied"
    ] is False


def test_delta_declares_no_topology_mutation(default_result):
    assert default_result.delta.metadata[
        "topology_modified"
    ] is False


def test_policy_free_helper(default_result):
    assert adaptive_connection_is_policy_free(
        default_result
    ) is True


def test_update_does_not_mutate_source_state(state, exposure):
    before = (
        state.stiffness,
        state.contractile_capacity,
        state.reflex_gain,
        state.fatigue,
        state.remodeling_bias,
    )

    update_adaptive_connection(
        update_id="immutability",
        state=state,
        exposure=exposure,
    )

    after = (
        state.stiffness,
        state.contractile_capacity,
        state.reflex_gain,
        state.fatigue,
        state.remodeling_bias,
    )

    assert after == before


def test_update_does_not_mutate_exposure(state, exposure):
    before = (
        exposure.load,
        exposure.strain,
        exposure.activation,
        exposure.damage,
        exposure.recovery,
    )

    update_adaptive_connection(
        update_id="immutability_exposure",
        state=state,
        exposure=exposure,
    )

    after = (
        exposure.load,
        exposure.strain,
        exposure.activation,
        exposure.damage,
        exposure.recovery,
    )

    assert after == before


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AdaptiveConnectionState(
            connection_id="c",
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
        ),
        lambda: ConnectionExposure(
            exposure_id="e",
            connection_id="c",
        ),
        lambda: AdaptiveConnectionConfig(),
        lambda: AdaptiveConnectionDelta(
            connection_id="c",
            stiffness_delta=0.0,
            contractile_capacity_delta=0.0,
            reflex_gain_delta=0.0,
            fatigue_delta=0.0,
            remodeling_bias_delta=0.0,
        ),
    ],
)
def test_primary_objects_are_frozen(factory):
    obj = factory()

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        obj.metadata = {}


def test_result_is_frozen(default_result):
    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        default_result.update_id = "changed"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AdaptiveConnectionState(
            connection_id="c",
            source_node_id="A",
            target_node_id="B",
            stiffness=1.0,
            contractile_capacity=1.0,
            reflex_gain=1.0,
            metadata={"x": 1},
        ),
        lambda: ConnectionExposure(
            exposure_id="e",
            connection_id="c",
            metadata={"x": 1},
        ),
        lambda: AdaptiveConnectionConfig(
            metadata={"x": 1},
        ),
    ],
)
def test_primary_metadata_is_read_only(factory):
    obj = factory()

    with pytest.raises(TypeError):
        obj.metadata["x"] = 2


def test_result_metadata_is_read_only(default_result):
    with pytest.raises(TypeError):
        default_result.metadata[
            "schema_version"
        ] = "changed"


def test_delta_metadata_is_read_only(default_result):
    with pytest.raises(TypeError):
        default_result.delta.metadata[
            "learning_applied"
        ] = True


def test_default_update_is_deterministic(state, exposure):
    left = update_adaptive_connection(
        update_id="same",
        state=state,
        exposure=exposure,
    )
    right = update_adaptive_connection(
        update_id="same",
        state=state,
        exposure=exposure,
    )

    assert left == right


def test_custom_config_update_is_deterministic(state, exposure):
    config = AdaptiveConnectionConfig(
        fatigue_load_gain=0.2,
        fatigue_strain_gain=0.3,
        fatigue_activation_gain=0.4,
        recovery_gain=0.1,
        damage_capacity_loss_gain=0.6,
        fatigue_capacity_loss_gain=0.2,
        reflex_activation_gain=0.25,
        reflex_damage_gain=0.15,
        stiffness_load_gain=0.3,
        stiffness_damage_gain=0.4,
        remodeling_rate=0.07,
    )

    left = update_adaptive_connection(
        update_id="same_custom",
        state=state,
        exposure=exposure,
        config=config,
    )

    right = update_adaptive_connection(
        update_id="same_custom",
        state=state,
        exposure=exposure,
        config=config,
    )

    assert left == right


def test_reporting_helpers_are_deterministic(default_result):
    assert adaptive_connection_signature(
        default_result
    ) == adaptive_connection_signature(
        default_result
    )

    assert effective_transfer_gain(
        default_result.target_state
    ) == effective_transfer_gain(
        default_result.target_state
    )
