"""
Tests for roif.history.prestress_redistribution

The suite verifies:
- validation
- direct perturbation
- deterministic graph propagation
- directionality
- attenuation / decay
- connection capacity
- reserve modulation
- node bounds
- optional cumulative-delta cap
- disabled connections
- multi-step redistribution
- immutability / read-only metadata
- no hidden learning or topology mutation
- deterministic reporting helpers
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.prestress_redistribution import (
    PrestressConnection,
    PrestressContribution,
    PrestressNodeState,
    PrestressPerturbation,
    PrestressRedistributionConfig,
    PrestressRedistributionError,
    PrestressRedistributionResult,
    SCHEMA_VERSION,
    node_delta,
    prestress_redistribution_is_policy_free,
    prestress_signature,
    redistribute_prestress,
    target_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nodes():
    return (
        PrestressNodeState(
            node_id="A",
            prestress=0.20,
            min_prestress=-1.0,
            max_prestress=1.0,
            reserve=1.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=0.10,
            min_prestress=-1.0,
            max_prestress=1.0,
            reserve=1.0,
        ),
        PrestressNodeState(
            node_id="C",
            prestress=0.00,
            min_prestress=-1.0,
            max_prestress=1.0,
            reserve=1.0,
        ),
    )


@pytest.fixture
def chain_connections():
    return (
        PrestressConnection(
            connection_id="A_B",
            source_node_id="A",
            target_node_id="B",
            transfer_gain=1.0,
            attenuation=1.0,
            capacity=1.0,
        ),
        PrestressConnection(
            connection_id="B_C",
            source_node_id="B",
            target_node_id="C",
            transfer_gain=1.0,
            attenuation=1.0,
            capacity=1.0,
        ),
    )


@pytest.fixture
def perturb_a():
    return (
        PrestressPerturbation(
            perturbation_id="damage_A",
            node_id="A",
            delta=-0.40,
        ),
    )


@pytest.fixture
def default_result(
    nodes,
    chain_connections,
    perturb_a,
):
    return redistribute_prestress(
        redistribution_id="default",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
        config=PrestressRedistributionConfig(
            propagation_steps=2,
            propagation_decay=0.50,
            reserve_modulation=False,
        ),
    )


# ---------------------------------------------------------------------------
# Schema / construction
# ---------------------------------------------------------------------------

def test_schema_version():
    assert SCHEMA_VERSION == "prestress_redistribution_v1"


def test_node_constructs():
    node = PrestressNodeState(
        node_id="n",
        prestress=0.25,
    )
    assert node.node_id == "n"
    assert node.prestress == pytest.approx(0.25)


def test_connection_constructs():
    connection = PrestressConnection(
        connection_id="c",
        source_node_id="a",
        target_node_id="b",
    )
    assert connection.connection_id == "c"


def test_perturbation_constructs():
    perturbation = PrestressPerturbation(
        perturbation_id="p",
        node_id="a",
        delta=-0.1,
    )
    assert perturbation.delta == pytest.approx(-0.1)


def test_default_config_constructs():
    config = PrestressRedistributionConfig()
    assert config.propagation_steps == 4
    assert config.propagation_decay == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    ["", "   "],
)
def test_empty_node_id_rejected(value):
    with pytest.raises(PrestressRedistributionError):
        PrestressNodeState(
            node_id=value,
            prestress=0.0,
        )


def test_node_bounds_order_rejected():
    with pytest.raises(PrestressRedistributionError):
        PrestressNodeState(
            node_id="x",
            prestress=0.0,
            min_prestress=1.0,
            max_prestress=-1.0,
        )


def test_node_prestress_outside_bounds_rejected():
    with pytest.raises(PrestressRedistributionError):
        PrestressNodeState(
            node_id="x",
            prestress=2.0,
            min_prestress=-1.0,
            max_prestress=1.0,
        )


@pytest.mark.parametrize(
    "reserve",
    [-0.01, 1.01],
)
def test_invalid_reserve_rejected(reserve):
    with pytest.raises(PrestressRedistributionError):
        PrestressNodeState(
            node_id="x",
            prestress=0.0,
            reserve=reserve,
        )


def test_self_connection_rejected():
    with pytest.raises(PrestressRedistributionError):
        PrestressConnection(
            connection_id="self",
            source_node_id="a",
            target_node_id="a",
        )


@pytest.mark.parametrize(
    "attenuation",
    [-0.1, 1.1],
)
def test_invalid_attenuation_rejected(attenuation):
    with pytest.raises(PrestressRedistributionError):
        PrestressConnection(
            connection_id="c",
            source_node_id="a",
            target_node_id="b",
            attenuation=attenuation,
        )


def test_negative_capacity_rejected():
    with pytest.raises(PrestressRedistributionError):
        PrestressConnection(
            connection_id="c",
            source_node_id="a",
            target_node_id="b",
            capacity=-0.1,
        )


def test_negative_propagation_steps_rejected():
    with pytest.raises(PrestressRedistributionError):
        PrestressRedistributionConfig(
            propagation_steps=-1,
        )


@pytest.mark.parametrize(
    "decay",
    [-0.01, 1.01],
)
def test_invalid_propagation_decay_rejected(decay):
    with pytest.raises(PrestressRedistributionError):
        PrestressRedistributionConfig(
            propagation_decay=decay,
        )


def test_negative_minimum_signal_rejected():
    with pytest.raises(PrestressRedistributionError):
        PrestressRedistributionConfig(
            minimum_signal=-1e-6,
        )


def test_negative_total_delta_cap_rejected():
    with pytest.raises(PrestressRedistributionError):
        PrestressRedistributionConfig(
            max_total_abs_delta_per_node=-0.1,
        )


def test_empty_nodes_rejected():
    with pytest.raises(PrestressRedistributionError):
        redistribute_prestress(
            redistribution_id="x",
            nodes=(),
            connections=(),
            perturbations=(),
        )


def test_duplicate_node_id_rejected():
    with pytest.raises(PrestressRedistributionError):
        redistribute_prestress(
            redistribution_id="x",
            nodes=(
                PrestressNodeState(
                    node_id="A",
                    prestress=0.0,
                ),
                PrestressNodeState(
                    node_id="A",
                    prestress=0.1,
                ),
            ),
            connections=(),
            perturbations=(),
        )


def test_duplicate_connection_id_rejected(nodes):
    with pytest.raises(PrestressRedistributionError):
        redistribute_prestress(
            redistribution_id="x",
            nodes=nodes,
            connections=(
                PrestressConnection(
                    connection_id="same",
                    source_node_id="A",
                    target_node_id="B",
                ),
                PrestressConnection(
                    connection_id="same",
                    source_node_id="B",
                    target_node_id="C",
                ),
            ),
            perturbations=(),
        )


def test_unknown_connection_source_rejected(nodes):
    with pytest.raises(PrestressRedistributionError):
        redistribute_prestress(
            redistribution_id="x",
            nodes=nodes,
            connections=(
                PrestressConnection(
                    connection_id="bad",
                    source_node_id="Z",
                    target_node_id="A",
                ),
            ),
            perturbations=(),
        )


def test_unknown_connection_target_rejected(nodes):
    with pytest.raises(PrestressRedistributionError):
        redistribute_prestress(
            redistribution_id="x",
            nodes=nodes,
            connections=(
                PrestressConnection(
                    connection_id="bad",
                    source_node_id="A",
                    target_node_id="Z",
                ),
            ),
            perturbations=(),
        )


def test_unknown_perturbation_node_rejected(nodes):
    with pytest.raises(PrestressRedistributionError):
        redistribute_prestress(
            redistribution_id="x",
            nodes=nodes,
            connections=(),
            perturbations=(
                PrestressPerturbation(
                    perturbation_id="p",
                    node_id="Z",
                    delta=0.1,
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Direct perturbation
# ---------------------------------------------------------------------------

def test_zero_propagation_applies_only_direct_perturbation(
    nodes,
    chain_connections,
    perturb_a,
):
    result = redistribute_prestress(
        redistribution_id="direct",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
        config=PrestressRedistributionConfig(
            propagation_steps=0,
        ),
    )

    assert node_delta(
        result,
        "A",
    ) == pytest.approx(-0.40)

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.0)

    assert node_delta(
        result,
        "C",
    ) == pytest.approx(0.0)


def test_direct_perturbation_respects_node_lower_bound():
    nodes = (
        PrestressNodeState(
            node_id="A",
            prestress=0.0,
            min_prestress=-0.2,
            max_prestress=1.0,
        ),
    )

    result = redistribute_prestress(
        redistribution_id="bound",
        nodes=nodes,
        connections=(),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=-1.0,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=0,
        ),
    )

    assert target_state(
        result,
        "A",
    ).prestress == pytest.approx(-0.2)


def test_direct_perturbation_respects_total_delta_cap():
    result = redistribute_prestress(
        redistribution_id="cap",
        nodes=(
            PrestressNodeState(
                node_id="A",
                prestress=0.0,
            ),
        ),
        connections=(),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.8,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=0,
            max_total_abs_delta_per_node=0.25,
        ),
    )

    assert node_delta(
        result,
        "A",
    ) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Propagation semantics
# ---------------------------------------------------------------------------

def test_first_hop_uses_source_delta_not_absolute_prestress(
    nodes,
    chain_connections,
    perturb_a,
):
    result = redistribute_prestress(
        redistribution_id="hop",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=0.50,
            reserve_modulation=False,
        ),
    )

    # A delta = -0.4, so B receives -0.4 * 1 * 1 * 0.5 = -0.2.
    assert node_delta(
        result,
        "B",
    ) == pytest.approx(-0.20)


def test_second_hop_propagates_previous_round_signal(
    default_result,
):
    # Step 1: A -> B = -0.20
    # Step 2: B -> C = -0.10
    assert node_delta(
        default_result,
        "C",
    ) == pytest.approx(-0.10)


def test_directionality_prevents_reverse_propagation(
    nodes,
):
    result = redistribute_prestress(
        redistribution_id="directional",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="B",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=2,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "A",
    ) == pytest.approx(0.0)


def test_negative_transfer_gain_inverts_signal(
    nodes,
):
    result = redistribute_prestress(
        redistribution_id="invert",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                transfer_gain=-1.0,
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(-0.4)


def test_attenuation_reduces_transmission(
    nodes,
):
    result = redistribute_prestress(
        redistribution_id="attenuation",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                attenuation=0.25,
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.1)


def test_global_propagation_decay_reduces_transmission(
    nodes,
):
    result = redistribute_prestress(
        redistribution_id="decay",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=0.25,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.1)


def test_connection_capacity_caps_transmitted_signal(
    nodes,
):
    result = redistribute_prestress(
        redistribution_id="capacity",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                capacity=0.1,
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.8,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.1)

    assert result.contributions[0].capacity_limited is True


def test_disabled_connection_does_not_propagate(
    nodes,
):
    result = redistribute_prestress(
        redistribution_id="disabled",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                enabled=False,
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=2,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.0)

    assert result.contributions == ()


def test_minimum_signal_stops_small_propagation(
    nodes,
):
    result = redistribute_prestress(
        redistribution_id="minimum",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.01,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=2,
            propagation_decay=1.0,
            reserve_modulation=False,
            minimum_signal=0.02,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reserve semantics
# ---------------------------------------------------------------------------

def test_reserve_modulation_reduces_target_acceptance():
    nodes = (
        PrestressNodeState(
            node_id="A",
            prestress=0.0,
            reserve=1.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=0.0,
            reserve=0.25,
        ),
    )

    result = redistribute_prestress(
        redistribution_id="reserve",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=True,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.1)

    assert result.contributions[0].reserve_limited is True


def test_reserve_modulation_can_be_disabled():
    nodes = (
        PrestressNodeState(
            node_id="A",
            prestress=0.0,
            reserve=1.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=0.0,
            reserve=0.25,
        ),
    )

    result = redistribute_prestress(
        redistribution_id="reserve_off",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Node bounds / cumulative cap during propagation
# ---------------------------------------------------------------------------

def test_propagated_signal_respects_target_upper_bound():
    nodes = (
        PrestressNodeState(
            node_id="A",
            prestress=0.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=0.45,
            min_prestress=-1.0,
            max_prestress=0.50,
        ),
    )

    result = redistribute_prestress(
        redistribution_id="upper_bound",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.4,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert target_state(
        result,
        "B",
    ).prestress == pytest.approx(0.50)

    assert result.contributions[0].node_bound_limited is True


def test_propagated_signal_respects_total_delta_cap():
    nodes = (
        PrestressNodeState(
            node_id="A",
            prestress=0.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=0.0,
        ),
    )

    result = redistribute_prestress(
        redistribution_id="total_cap",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.8,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=False,
            max_total_abs_delta_per_node=0.15,
        ),
    )

    # The same global cap also limits the source direct perturbation.
    assert node_delta(
        result,
        "A",
    ) == pytest.approx(0.15)

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Branching / multiple perturbations
# ---------------------------------------------------------------------------

def test_branching_sends_signal_to_multiple_targets():
    nodes = (
        PrestressNodeState(
            node_id="A",
            prestress=0.0,
        ),
        PrestressNodeState(
            node_id="B",
            prestress=0.0,
        ),
        PrestressNodeState(
            node_id="C",
            prestress=0.0,
        ),
    )

    result = redistribute_prestress(
        redistribution_id="branch",
        nodes=nodes,
        connections=(
            PrestressConnection(
                connection_id="A_B",
                source_node_id="A",
                target_node_id="B",
                transfer_gain=0.5,
            ),
            PrestressConnection(
                connection_id="A_C",
                source_node_id="A",
                target_node_id="C",
                transfer_gain=0.25,
            ),
        ),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p",
                node_id="A",
                delta=0.8,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=1,
            propagation_decay=1.0,
            reserve_modulation=False,
        ),
    )

    assert node_delta(
        result,
        "B",
    ) == pytest.approx(0.4)

    assert node_delta(
        result,
        "C",
    ) == pytest.approx(0.2)


def test_multiple_direct_perturbations_accumulate_on_same_node():
    result = redistribute_prestress(
        redistribution_id="multi_direct",
        nodes=(
            PrestressNodeState(
                node_id="A",
                prestress=0.0,
            ),
        ),
        connections=(),
        perturbations=(
            PrestressPerturbation(
                perturbation_id="p1",
                node_id="A",
                delta=0.2,
            ),
            PrestressPerturbation(
                perturbation_id="p2",
                node_id="A",
                delta=-0.05,
            ),
        ),
        config=PrestressRedistributionConfig(
            propagation_steps=0,
        ),
    )

    assert node_delta(
        result,
        "A",
    ) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Result structure / helpers
# ---------------------------------------------------------------------------

def test_result_type(default_result):
    assert isinstance(
        default_result,
        PrestressRedistributionResult,
    )


def test_source_states_preserved(default_result):
    assert tuple(
        node.prestress
        for node in default_result.source_states
    ) == pytest.approx(
        (0.20, 0.10, 0.00)
    )


def test_target_state_helper(default_result):
    state = target_state(
        default_result,
        "B",
    )

    assert state is not None
    assert state.node_id == "B"


def test_target_state_helper_returns_none_for_unknown(default_result):
    assert target_state(
        default_result,
        "Z",
    ) is None


def test_node_delta_helper_returns_none_for_unknown(default_result):
    assert node_delta(
        default_result,
        "Z",
    ) is None


def test_signature_contains_target_states(default_result):
    signature = prestress_signature(
        default_result
    )

    assert signature[0] == "default"

    assert signature[1] == (
        ("A", pytest.approx(-0.20)),
        ("B", pytest.approx(-0.10)),
        ("C", pytest.approx(-0.10)),
    )


def test_contributions_are_tuple(default_result):
    assert isinstance(
        default_result.contributions,
        tuple,
    )


def test_contribution_type(default_result):
    assert all(
        isinstance(
            item,
            PrestressContribution,
        )
        for item in default_result.contributions
    )


def test_contribution_step_order_is_non_decreasing(default_result):
    steps = tuple(
        item.step_index
        for item in default_result.contributions
    )

    assert steps == tuple(
        sorted(steps)
    )


# ---------------------------------------------------------------------------
# Metadata safety semantics
# ---------------------------------------------------------------------------

def test_result_metadata_schema(default_result):
    assert (
        default_result.metadata[
            "schema_version"
        ]
        == SCHEMA_VERSION
    )


def test_result_declares_no_memory_mutation(default_result):
    assert (
        default_result.metadata[
            "memory_mutated"
        ]
        is False
    )


def test_result_declares_no_learning(default_result):
    assert (
        default_result.metadata[
            "learning_applied"
        ]
        is False
    )


def test_result_declares_no_topology_mutation(default_result):
    assert (
        default_result.metadata[
            "topology_modified"
        ]
        is False
    )


def test_result_declares_no_action(default_result):
    assert (
        default_result.metadata[
            "action_selected"
        ]
        is False
    )


def test_result_declares_no_policy_modified(default_result):
    assert (
        default_result.metadata[
            "policy_modified"
        ]
        is False
    )


def test_result_declares_no_diagnosis(default_result):
    assert (
        default_result.metadata[
            "diagnosis_generated"
        ]
        is False
    )


def test_result_declares_no_biological_truth(default_result):
    assert (
        default_result.metadata[
            "biological_truth_claimed"
        ]
        is False
    )


def test_result_declares_no_causal_truth(default_result):
    assert (
        default_result.metadata[
            "causal_truth_inferred"
        ]
        is False
    )


def test_every_contribution_declares_no_learning(default_result):
    assert all(
        contribution.metadata[
            "learning_applied"
        ]
        is False
        for contribution in default_result.contributions
    )


def test_every_contribution_declares_no_topology_mutation(default_result):
    assert all(
        contribution.metadata[
            "topology_modified"
        ]
        is False
        for contribution in default_result.contributions
    )


def test_policy_free_helper(default_result):
    assert (
        prestress_redistribution_is_policy_free(
            default_result
        )
        is True
    )


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------

def test_redistribution_does_not_mutate_source_nodes(
    nodes,
    chain_connections,
    perturb_a,
):
    before = tuple(
        (
            node.node_id,
            node.prestress,
            node.reserve,
        )
        for node in nodes
    )

    redistribute_prestress(
        redistribution_id="immutability",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
    )

    after = tuple(
        (
            node.node_id,
            node.prestress,
            node.reserve,
        )
        for node in nodes
    )

    assert after == before


def test_redistribution_does_not_mutate_connections(
    nodes,
    chain_connections,
    perturb_a,
):
    before = tuple(
        (
            connection.connection_id,
            connection.transfer_gain,
            connection.enabled,
        )
        for connection in chain_connections
    )

    redistribute_prestress(
        redistribution_id="connection_immutability",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
    )

    after = tuple(
        (
            connection.connection_id,
            connection.transfer_gain,
            connection.enabled,
        )
        for connection in chain_connections
    )

    assert after == before


# ---------------------------------------------------------------------------
# Frozen dataclasses / read-only metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factory",
    [
        lambda: PrestressNodeState(
            node_id="A",
            prestress=0.0,
        ),
        lambda: PrestressConnection(
            connection_id="A_B",
            source_node_id="A",
            target_node_id="B",
        ),
        lambda: PrestressPerturbation(
            perturbation_id="p",
            node_id="A",
            delta=0.1,
        ),
        lambda: PrestressRedistributionConfig(),
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
        default_result.redistribution_id = "changed"


def test_contribution_is_frozen(default_result):
    contribution = default_result.contributions[0]

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        contribution.accepted_delta = 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PrestressNodeState(
            node_id="A",
            prestress=0.0,
            metadata={"x": 1},
        ),
        lambda: PrestressConnection(
            connection_id="A_B",
            source_node_id="A",
            target_node_id="B",
            metadata={"x": 1},
        ),
        lambda: PrestressPerturbation(
            perturbation_id="p",
            node_id="A",
            delta=0.1,
            metadata={"x": 1},
        ),
        lambda: PrestressRedistributionConfig(
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


def test_node_delta_mapping_is_read_only(default_result):
    with pytest.raises(TypeError):
        default_result.node_delta_by_id[
            "A"
        ] = 999.0


def test_contribution_metadata_is_read_only(default_result):
    with pytest.raises(TypeError):
        default_result.contributions[
            0
        ].metadata[
            "learning_applied"
        ] = True


# ---------------------------------------------------------------------------
# Determinism / ordering
# ---------------------------------------------------------------------------

def test_default_redistribution_is_deterministic(
    nodes,
    chain_connections,
    perturb_a,
):
    kwargs = dict(
        redistribution_id="deterministic",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
        config=PrestressRedistributionConfig(
            propagation_steps=4,
            propagation_decay=0.75,
            reserve_modulation=True,
        ),
    )

    left = redistribute_prestress(
        **kwargs
    )
    right = redistribute_prestress(
        **kwargs
    )

    assert left == right


def test_connection_input_order_does_not_change_result(
    nodes,
    chain_connections,
    perturb_a,
):
    config = PrestressRedistributionConfig(
        propagation_steps=2,
        propagation_decay=0.5,
        reserve_modulation=False,
    )

    left = redistribute_prestress(
        redistribution_id="ordered",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
        config=config,
    )

    right = redistribute_prestress(
        redistribution_id="ordered",
        nodes=nodes,
        connections=tuple(
            reversed(
                chain_connections
            )
        ),
        perturbations=perturb_a,
        config=config,
    )

    assert (
        left.node_delta_by_id
        == right.node_delta_by_id
    )

    assert (
        prestress_signature(left)
        == prestress_signature(right)
    )


def test_reporting_helpers_are_deterministic(
    nodes,
    chain_connections,
    perturb_a,
):
    result = redistribute_prestress(
        redistribution_id="reporting",
        nodes=nodes,
        connections=chain_connections,
        perturbations=perturb_a,
    )

    assert (
        prestress_signature(result)
        == prestress_signature(result)
    )

    assert (
        node_delta(
            result,
            "B",
        )
        == node_delta(
            result,
            "B",
        )
    )

    assert (
        target_state(
            result,
            "C",
        )
        == target_state(
            result,
            "C",
        )
    )
