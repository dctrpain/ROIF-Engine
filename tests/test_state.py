"""
Tests for roif.state.

The state layer defines immutable snapshots of a multi-plane ROIF system:

    NodeState
    PlaneState
    SystemState

The tests verify:

- capacity, load, reserve and deficit;
- utilization and signed utilization;
- node and plane activation states;
- temporal consistency;
- multi-plane node membership;
- immutable mappings and snapshots;
- system aggregates and vectors;
- snapshot-level D_fast;
- structural validation and serialization.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from roif.state import (
    NodeState,
    NodeStatus,
    PlaneState,
    PlaneStatus,
    StateError,
    SystemState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plane(
    plane_id: str = "mechanical",
    *,
    status: PlaneStatus = PlaneStatus.INACTIVE,
    activation: float = 0.0,
    activation_time: float | None = None,
    last_update_time: float = 0.0,
    characteristic_time: float = 1.0,
    relaxation_time: float = 1.0,
    hysteresis: float = 0.0,
) -> PlaneState:
    """Create a valid PlaneState for tests."""

    return PlaneState(
        plane_id=plane_id,
        status=status,
        activation=activation,
        activation_time=activation_time,
        last_update_time=last_update_time,
        characteristic_time=characteristic_time,
        relaxation_time=relaxation_time,
        hysteresis=hysteresis,
    )


def make_node(
    node_id: str = "node_a",
    *,
    plane_ids: tuple[str, ...] = ("mechanical",),
    capacity: float = 100.0,
    load: float = 0.0,
    status: NodeStatus = NodeStatus.QUIESCENT,
    activation_time: float | None = None,
    last_update_time: float = 0.0,
    relaxation_time: float = 0.0,
    hysteresis: float = 0.0,
) -> NodeState:
    """Create a valid NodeState for tests."""

    return NodeState(
        node_id=node_id,
        plane_ids=plane_ids,
        capacity=capacity,
        load=load,
        status=status,
        activation_time=activation_time,
        last_update_time=last_update_time,
        relaxation_time=relaxation_time,
        hysteresis=hysteresis,
    )


def make_system(
    *,
    time: float = 0.0,
) -> SystemState:
    """Create a small valid two-plane system."""

    mechanical = make_plane("mechanical")
    vascular = make_plane("vascular")

    node_a = make_node(
        "node_a",
        plane_ids=("mechanical",),
        capacity=100.0,
        load=40.0,
    )
    node_b = make_node(
        "node_b",
        plane_ids=("mechanical", "vascular"),
        capacity=80.0,
        load=60.0,
    )

    return SystemState(
        time=time,
        nodes={
            node_a.node_id: node_a,
            node_b.node_id: node_b,
        },
        planes={
            mechanical.plane_id: mechanical,
            vascular.plane_id: vascular,
        },
    )


# ---------------------------------------------------------------------------
# NodeState: construction and basic quantities
# ---------------------------------------------------------------------------


def test_node_state_construction() -> None:
    """NodeState must preserve valid constructor values."""

    node = NodeState(
        node_id="diaphragm_right",
        plane_ids=("mechanical", "autonomic"),
        capacity=120.0,
        load=-30.0,
        status=NodeStatus.ACTIVE,
        activation_time=2.0,
        last_update_time=5.0,
        relaxation_time=10.0,
        hysteresis=0.25,
        variables={
            "stiffness": 1.4,
        },
        metadata={
            "region": "thorax",
        },
    )

    assert node.node_id == "diaphragm_right"
    assert node.plane_ids == ("mechanical", "autonomic")
    assert node.capacity == pytest.approx(120.0)
    assert node.load == pytest.approx(-30.0)
    assert node.status is NodeStatus.ACTIVE
    assert node.activation_time == pytest.approx(2.0)
    assert node.last_update_time == pytest.approx(5.0)
    assert node.relaxation_time == pytest.approx(10.0)
    assert node.hysteresis == pytest.approx(0.25)
    assert node.variables["stiffness"] == pytest.approx(1.4)
    assert node.metadata["region"] == "thorax"


def test_node_identifier_is_trimmed() -> None:
    """Leading and trailing whitespace must be removed."""

    node = NodeState(
        node_id="  node_a  ",
    )

    assert node.node_id == "node_a"


def test_node_plane_identifiers_are_trimmed() -> None:
    """Functional-plane identifiers must be normalized."""

    node = NodeState(
        node_id="node_a",
        plane_ids=(
            " mechanical ",
            " vascular ",
        ),
    )

    assert node.plane_ids == (
        "mechanical",
        "vascular",
    )


def test_node_status_can_be_created_from_string() -> None:
    """Declared NodeStatus values must support string coercion."""

    node = NodeState(
        node_id="node_a",
        status="active",
    )

    assert node.status is NodeStatus.ACTIVE


def test_node_absolute_load_ignores_sign() -> None:
    """absolute_load must return the magnitude of signed load."""

    node = make_node(
        capacity=100.0,
        load=-40.0,
    )

    assert node.absolute_load == pytest.approx(40.0)


def test_node_positive_reserve() -> None:
    """Reserve must equal capacity minus absolute load."""

    node = make_node(
        capacity=100.0,
        load=-40.0,
    )

    assert node.reserve == pytest.approx(60.0)
    assert node.available_reserve == pytest.approx(60.0)
    assert node.deficit == pytest.approx(0.0)


def test_node_zero_reserve_at_capacity() -> None:
    """A fully utilized node must have zero reserve and deficit."""

    node = make_node(
        capacity=100.0,
        load=-100.0,
    )

    assert node.reserve == pytest.approx(0.0)
    assert node.available_reserve == pytest.approx(0.0)
    assert node.deficit == pytest.approx(0.0)


def test_node_negative_reserve_and_positive_deficit() -> None:
    """Overload must produce negative reserve and positive deficit."""

    node = make_node(
        capacity=100.0,
        load=130.0,
    )

    assert node.reserve == pytest.approx(-30.0)
    assert node.available_reserve == pytest.approx(0.0)
    assert node.deficit == pytest.approx(30.0)


def test_node_utilization() -> None:
    """Utilization must equal absolute load divided by capacity."""

    node = make_node(
        capacity=80.0,
        load=-60.0,
    )

    assert node.utilization == pytest.approx(0.75)


def test_node_signed_utilization_preserves_positive_sign() -> None:
    """Positive load must produce positive signed utilization."""

    node = make_node(
        capacity=80.0,
        load=60.0,
    )

    assert node.signed_utilization == pytest.approx(0.75)


def test_node_signed_utilization_preserves_negative_sign() -> None:
    """Negative load must produce negative signed utilization."""

    node = make_node(
        capacity=80.0,
        load=-60.0,
    )

    assert node.signed_utilization == pytest.approx(-0.75)


def test_zero_load_and_zero_capacity_have_zero_utilization() -> None:
    """An unloaded unavailable node must not appear overloaded."""

    node = make_node(
        capacity=0.0,
        load=0.0,
    )

    assert node.utilization == pytest.approx(0.0)
    assert node.signed_utilization == pytest.approx(0.0)
    assert not node.is_loaded
    assert not node.is_critical
    assert not node.is_exhausted


def test_positive_load_and_zero_capacity_have_infinite_utilization() -> None:
    """A loaded node with zero capacity must have infinite utilization."""

    node = make_node(
        capacity=0.0,
        load=10.0,
    )

    assert np.isposinf(node.utilization)
    assert np.isposinf(node.signed_utilization)
    assert node.is_loaded
    assert node.is_critical
    assert node.is_exhausted


def test_negative_load_and_zero_capacity_have_negative_signed_infinity() -> None:
    """Signed utilization must preserve negative load direction."""

    node = make_node(
        capacity=0.0,
        load=-10.0,
    )

    assert np.isposinf(node.utilization)
    assert np.isneginf(node.signed_utilization)


def test_node_is_critical_at_exact_capacity() -> None:
    """A node at utilization one is critical but not exhausted."""

    node = make_node(
        capacity=100.0,
        load=100.0,
    )

    assert node.is_critical
    assert not node.is_exhausted


def test_node_is_exhausted_above_capacity() -> None:
    """A node above capacity must be exhausted."""

    node = make_node(
        capacity=100.0,
        load=100.01,
    )

    assert node.is_critical
    assert node.is_exhausted


def test_node_is_available_with_positive_capacity() -> None:
    """A non-disabled node with capacity must be available."""

    node = make_node(
        capacity=100.0,
        status=NodeStatus.ACTIVE,
    )

    assert node.is_available


def test_disabled_node_is_not_available() -> None:
    """Disabled state must override positive capacity."""

    node = make_node(
        capacity=100.0,
        status=NodeStatus.DISABLED,
    )

    assert not node.is_available


def test_zero_capacity_node_is_not_available() -> None:
    """A node with zero capacity must be unavailable."""

    node = make_node(
        capacity=0.0,
    )

    assert not node.is_available


# ---------------------------------------------------------------------------
# NodeState: planes, time and variables
# ---------------------------------------------------------------------------


def test_node_belongs_to_plane() -> None:
    """A node may participate in several functional planes."""

    node = make_node(
        plane_ids=(
            "mechanical",
            "vascular",
            "autonomic",
        ),
    )

    assert node.belongs_to("mechanical")
    assert node.belongs_to("vascular")
    assert node.belongs_to("autonomic")
    assert not node.belongs_to("metabolic")


def test_node_active_duration() -> None:
    """active_duration must use last_update_time minus activation_time."""

    node = make_node(
        activation_time=2.0,
        last_update_time=8.5,
        status=NodeStatus.ACTIVE,
    )

    assert node.active_duration == pytest.approx(6.5)


def test_node_without_activation_time_has_zero_active_duration() -> None:
    """Missing activation timestamp must produce zero duration."""

    node = make_node(
        activation_time=None,
        last_update_time=8.5,
    )

    assert node.active_duration == pytest.approx(0.0)


def test_node_reads_existing_variable() -> None:
    """variable() must return a stored continuous variable."""

    node = NodeState(
        node_id="node_a",
        variables={
            "damage": 0.2,
        },
    )

    assert node.variable("damage") == pytest.approx(0.2)


def test_node_reads_missing_variable_default() -> None:
    """variable() must return an explicitly supplied numeric default."""

    node = make_node()

    assert node.variable(
        "damage",
        default=0.0,
    ) == pytest.approx(0.0)


def test_node_missing_variable_without_default_raises() -> None:
    """Reading an absent variable without a default must fail."""

    node = make_node()

    with pytest.raises(
        StateError,
        match="has no variable",
    ):
        node.variable("damage")


def test_node_with_load_returns_new_snapshot() -> None:
    """with_load() must not mutate the source node."""

    original = make_node(
        load=20.0,
        last_update_time=1.0,
    )

    updated = original.with_load(
        -60.0,
        time=3.0,
        status=NodeStatus.ACTIVE,
    )

    assert original.load == pytest.approx(20.0)
    assert original.last_update_time == pytest.approx(1.0)
    assert original.status is NodeStatus.QUIESCENT

    assert updated.load == pytest.approx(-60.0)
    assert updated.last_update_time == pytest.approx(3.0)
    assert updated.status is NodeStatus.ACTIVE


def test_node_with_capacity_returns_new_snapshot() -> None:
    """with_capacity() must update capacity immutably."""

    original = make_node(
        capacity=100.0,
    )

    updated = original.with_capacity(
        75.0,
        time=4.0,
    )

    assert original.capacity == pytest.approx(100.0)
    assert updated.capacity == pytest.approx(75.0)
    assert updated.last_update_time == pytest.approx(4.0)


def test_node_with_hysteresis_returns_new_snapshot() -> None:
    """with_hysteresis() must update node memory immutably."""

    original = make_node(
        hysteresis=0.1,
    )

    updated = original.with_hysteresis(
        0.8,
        time=5.0,
    )

    assert original.hysteresis == pytest.approx(0.1)
    assert updated.hysteresis == pytest.approx(0.8)
    assert updated.last_update_time == pytest.approx(5.0)


def test_node_with_variable_returns_new_snapshot() -> None:
    """with_variable() must preserve existing variables."""

    original = NodeState(
        node_id="node_a",
        variables={
            "stiffness": 1.0,
        },
    )

    updated = original.with_variable(
        "damage",
        0.25,
        time=2.0,
    )

    assert "damage" not in original.variables
    assert updated.variables["stiffness"] == pytest.approx(1.0)
    assert updated.variables["damage"] == pytest.approx(0.25)
    assert updated.last_update_time == pytest.approx(2.0)


def test_node_activate() -> None:
    """activate() must record activation time and active status."""

    original = make_node(
        last_update_time=2.0,
    )

    updated = original.activate(5.0)

    assert updated.status is NodeStatus.ACTIVE
    assert updated.activation_time == pytest.approx(5.0)
    assert updated.last_update_time == pytest.approx(5.0)


def test_node_activate_accepts_alternative_status() -> None:
    """Activation may explicitly enter another active state."""

    node = make_node().activate(
        2.0,
        status=NodeStatus.CRITICAL,
    )

    assert node.status is NodeStatus.CRITICAL


def test_node_deactivate() -> None:
    """deactivate() must clear activation timestamp."""

    active = make_node(
        status=NodeStatus.ACTIVE,
        activation_time=2.0,
        last_update_time=5.0,
    )

    inactive = active.deactivate(8.0)

    assert inactive.status is NodeStatus.QUIESCENT
    assert inactive.activation_time is None
    assert inactive.last_update_time == pytest.approx(8.0)


def test_node_deactivate_accepts_recovering_status() -> None:
    """Deactivation may transition to recovering state."""

    active = make_node(
        status=NodeStatus.ACTIVE,
        activation_time=1.0,
        last_update_time=3.0,
    )

    recovering = active.deactivate(
        4.0,
        status=NodeStatus.RECOVERING,
    )

    assert recovering.status is NodeStatus.RECOVERING
    assert recovering.activation_time is None


def test_node_activate_rejects_backward_time() -> None:
    """Activation time cannot precede the existing snapshot time."""

    node = make_node(
        last_update_time=5.0,
    )

    with pytest.raises(
        StateError,
        match="earlier",
    ):
        node.activate(4.0)


def test_node_deactivate_rejects_backward_time() -> None:
    """Deactivation time cannot move backwards."""

    node = make_node(
        status=NodeStatus.ACTIVE,
        activation_time=2.0,
        last_update_time=5.0,
    )

    with pytest.raises(
        StateError,
        match="earlier",
    ):
        node.deactivate(4.0)


# ---------------------------------------------------------------------------
# NodeState validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_id",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_node_rejects_empty_identifier(
    node_id: str,
) -> None:
    """node_id must contain a non-whitespace identifier."""

    with pytest.raises(
        StateError,
        match="cannot be empty",
    ):
        NodeState(
            node_id=node_id,
        )


@pytest.mark.parametrize(
    "node_id",
    [
        1,
        1.5,
        None,
        [],
    ],
)
def test_node_rejects_non_string_identifier(
    node_id: object,
) -> None:
    """node_id must be a string."""

    with pytest.raises(
        StateError,
        match="must be a string",
    ):
        NodeState(
            node_id=node_id,  # type: ignore[arg-type]
        )


def test_node_rejects_duplicate_plane_ids() -> None:
    """A node cannot reference the same plane twice."""

    with pytest.raises(
        StateError,
        match="duplicates",
    ):
        NodeState(
            node_id="node_a",
            plane_ids=(
                "mechanical",
                "mechanical",
            ),
        )


def test_node_rejects_empty_plane_identifier() -> None:
    """Every plane identifier must be non-empty."""

    with pytest.raises(
        StateError,
        match="cannot be empty",
    ):
        NodeState(
            node_id="node_a",
            plane_ids=(
                "mechanical",
                "",
            ),
        )


@pytest.mark.parametrize(
    "capacity",
    [
        -1.0,
        -0.001,
    ],
)
def test_node_rejects_negative_capacity(
    capacity: float,
) -> None:
    """Capacity must be non-negative."""

    with pytest.raises(
        StateError,
        match="non-negative",
    ):
        make_node(
            capacity=capacity,
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_node_rejects_non_finite_capacity(
    value: float,
) -> None:
    """Capacity must be finite."""

    with pytest.raises(
        StateError,
        match="finite",
    ):
        make_node(
            capacity=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_node_rejects_non_finite_load(
    value: float,
) -> None:
    """Signed load must be finite."""

    with pytest.raises(
        StateError,
        match="finite",
    ):
        make_node(
            load=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_node_rejects_invalid_last_update_time(
    value: float,
) -> None:
    """last_update_time must be finite and non-negative."""

    with pytest.raises(StateError):
        make_node(
            last_update_time=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_node_rejects_invalid_relaxation_time(
    value: float,
) -> None:
    """relaxation_time must be finite and non-negative."""

    with pytest.raises(StateError):
        make_node(
            relaxation_time=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_node_rejects_non_finite_hysteresis(
    value: float,
) -> None:
    """Node hysteresis may be signed but must remain finite."""

    with pytest.raises(
        StateError,
        match="finite",
    ):
        make_node(
            hysteresis=value,
        )


def test_node_rejects_activation_time_after_update_time() -> None:
    """An activation timestamp cannot be in the snapshot future."""

    with pytest.raises(
        StateError,
        match="later",
    ):
        make_node(
            activation_time=5.0,
            last_update_time=4.0,
        )


@pytest.mark.parametrize(
    "status",
    [
        "unknown",
        "",
        "ACTIVE",
        123,
    ],
)
def test_node_rejects_invalid_status(
    status: object,
) -> None:
    """Only declared NodeStatus values are valid."""

    with pytest.raises(
        StateError,
        match="Unknown node status",
    ):
        NodeState(
            node_id="node_a",
            status=status,  # type: ignore[arg-type]
        )


def test_node_rejects_non_mapping_variables() -> None:
    """variables must be supplied as a mapping."""

    with pytest.raises(
        StateError,
        match="mapping",
    ):
        NodeState(
            node_id="node_a",
            variables=[("damage", 0.2)],  # type: ignore[arg-type]
        )


def test_node_rejects_non_finite_variable_value() -> None:
    """Every additional continuous variable must be finite."""

    with pytest.raises(
        StateError,
        match="finite",
    ):
        NodeState(
            node_id="node_a",
            variables={
                "damage": np.nan,
            },
        )


def test_node_rejects_empty_variable_name() -> None:
    """Continuous variable names must be non-empty."""

    with pytest.raises(
        StateError,
        match="cannot be empty",
    ):
        NodeState(
            node_id="node_a",
            variables={
                "": 0.2,
            },
        )


def test_node_rejects_non_mapping_metadata() -> None:
    """metadata must be a mapping."""

    with pytest.raises(
        StateError,
        match="mapping",
    ):
        NodeState(
            node_id="node_a",
            metadata=["region"],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# NodeState immutability and serialization
# ---------------------------------------------------------------------------


def test_node_dataclass_is_frozen() -> None:
    """Direct field mutation must be rejected."""

    node = make_node()

    with pytest.raises(FrozenInstanceError):
        node.load = 100.0  # type: ignore[misc]


def test_node_variables_mapping_is_read_only() -> None:
    """Stored continuous variables must not be mutable."""

    node = NodeState(
        node_id="node_a",
        variables={
            "damage": 0.2,
        },
    )

    with pytest.raises(TypeError):
        node.variables["damage"] = 0.8  # type: ignore[index]


def test_node_metadata_mapping_is_read_only() -> None:
    """Stored metadata mapping must not be mutable."""

    node = NodeState(
        node_id="node_a",
        metadata={
            "region": "thorax",
        },
    )

    with pytest.raises(TypeError):
        node.metadata["region"] = "pelvis"  # type: ignore[index]


def test_node_input_mappings_are_copied() -> None:
    """Changing source dictionaries must not change a NodeState."""

    variables = {
        "damage": 0.2,
    }
    metadata = {
        "region": "thorax",
    }

    node = NodeState(
        node_id="node_a",
        variables=variables,
        metadata=metadata,
    )

    variables["damage"] = 0.9
    metadata["region"] = "pelvis"

    assert node.variables["damage"] == pytest.approx(0.2)
    assert node.metadata["region"] == "thorax"


def test_node_as_dict_contains_derived_quantities() -> None:
    """Node serialization must include primary and derived state."""

    node = NodeState(
        node_id="node_a",
        plane_ids=("mechanical",),
        capacity=100.0,
        load=-120.0,
        status=NodeStatus.CRITICAL,
        activation_time=2.0,
        last_update_time=5.0,
        relaxation_time=10.0,
        hysteresis=0.4,
        variables={
            "damage": 0.2,
        },
        metadata={
            "region": "thorax",
        },
    )

    data = node.as_dict()

    assert data["node_id"] == "node_a"
    assert data["plane_ids"] == ["mechanical"]
    assert data["capacity"] == pytest.approx(100.0)
    assert data["load"] == pytest.approx(-120.0)
    assert data["absolute_load"] == pytest.approx(120.0)
    assert data["reserve"] == pytest.approx(-20.0)
    assert data["available_reserve"] == pytest.approx(0.0)
    assert data["deficit"] == pytest.approx(20.0)
    assert data["utilization"] == pytest.approx(1.2)
    assert data["signed_utilization"] == pytest.approx(-1.2)
    assert data["status"] == "critical"
    assert data["variables"] == {
        "damage": 0.2,
    }
    assert data["metadata"] == {
        "region": "thorax",
    }


# ---------------------------------------------------------------------------
# PlaneState: construction and dynamics
# ---------------------------------------------------------------------------


def test_plane_state_construction() -> None:
    """PlaneState must preserve valid constructor values."""

    plane = PlaneState(
        plane_id="vascular",
        status=PlaneStatus.ACTIVE,
        activation=1.25,
        activation_time=2.0,
        last_update_time=5.0,
        characteristic_time=4.0,
        relaxation_time=8.0,
        hysteresis=0.3,
        variables={
            "tone": 0.8,
        },
        metadata={
            "domain": "circulation",
        },
    )

    assert plane.plane_id == "vascular"
    assert plane.status is PlaneStatus.ACTIVE
    assert plane.activation == pytest.approx(1.25)
    assert plane.activation_time == pytest.approx(2.0)
    assert plane.last_update_time == pytest.approx(5.0)
    assert plane.characteristic_time == pytest.approx(4.0)
    assert plane.relaxation_time == pytest.approx(8.0)
    assert plane.hysteresis == pytest.approx(0.3)
    assert plane.variables["tone"] == pytest.approx(0.8)


def test_plane_identifier_is_trimmed() -> None:
    """plane_id must be normalized."""

    plane = PlaneState(
        plane_id="  mechanical  ",
    )

    assert plane.plane_id == "mechanical"


def test_plane_status_can_be_created_from_string() -> None:
    """Declared PlaneStatus values must support string coercion."""

    plane = PlaneState(
        plane_id="mechanical",
        status="active",
        activation=1.0,
    )

    assert plane.status is PlaneStatus.ACTIVE


def test_plane_is_active() -> None:
    """Positive activation in a non-disabled plane means active."""

    plane = make_plane(
        status=PlaneStatus.ACTIVE,
        activation=0.5,
    )

    assert plane.is_active


def test_plane_with_zero_activation_is_not_active() -> None:
    """Status alone must not create continuous activation."""

    plane = make_plane(
        status=PlaneStatus.ACTIVE,
        activation=0.0,
    )

    assert not plane.is_active


def test_disabled_plane_is_not_active() -> None:
    """Disabled status must override positive activation."""

    plane = make_plane(
        status=PlaneStatus.DISABLED,
        activation=2.0,
    )

    assert not plane.is_active


def test_plane_active_duration() -> None:
    """Plane duration must use its activation timestamp."""

    plane = make_plane(
        status=PlaneStatus.ACTIVE,
        activation=1.0,
        activation_time=3.0,
        last_update_time=9.0,
    )

    assert plane.active_duration == pytest.approx(6.0)


def test_plane_without_activation_time_has_zero_duration() -> None:
    """A plane without activation timestamp has no active duration."""

    plane = make_plane(
        status=PlaneStatus.ACTIVE,
        activation=1.0,
        activation_time=None,
        last_update_time=9.0,
    )

    assert plane.active_duration == pytest.approx(0.0)


def test_plane_with_activation_returns_new_snapshot() -> None:
    """with_activation() must not mutate the source plane."""

    original = make_plane(
        activation=0.2,
        last_update_time=1.0,
    )

    updated = original.with_activation(
        0.8,
        time=3.0,
        status=PlaneStatus.ACTIVE,
    )

    assert original.activation == pytest.approx(0.2)
    assert original.last_update_time == pytest.approx(1.0)

    assert updated.activation == pytest.approx(0.8)
    assert updated.last_update_time == pytest.approx(3.0)
    assert updated.status is PlaneStatus.ACTIVE


def test_plane_with_hysteresis_returns_new_snapshot() -> None:
    """Plane hysteresis must be updated immutably."""

    original = make_plane(
        hysteresis=0.1,
    )

    updated = original.with_hysteresis(
        0.7,
        time=4.0,
    )

    assert original.hysteresis == pytest.approx(0.1)
    assert updated.hysteresis == pytest.approx(0.7)
    assert updated.last_update_time == pytest.approx(4.0)


def test_plane_with_variable_preserves_existing_variables() -> None:
    """with_variable() must copy the variable mapping."""

    original = PlaneState(
        plane_id="vascular",
        variables={
            "tone": 0.4,
        },
    )

    updated = original.with_variable(
        "flow",
        1.5,
        time=2.0,
    )

    assert "flow" not in original.variables
    assert updated.variables["tone"] == pytest.approx(0.4)
    assert updated.variables["flow"] == pytest.approx(1.5)


def test_plane_activate() -> None:
    """activate() must begin a new activation episode."""

    original = make_plane(
        last_update_time=2.0,
    )

    active = original.activate(
        5.0,
        activation=1.4,
    )

    assert active.status is PlaneStatus.ACTIVE
    assert active.activation == pytest.approx(1.4)
    assert active.activation_time == pytest.approx(5.0)
    assert active.last_update_time == pytest.approx(5.0)


def test_plane_begin_relaxation() -> None:
    """begin_relaxation() must preserve activation and memory."""

    active = make_plane(
        status=PlaneStatus.ACTIVE,
        activation=1.2,
        activation_time=2.0,
        last_update_time=5.0,
        hysteresis=0.4,
    )

    relaxing = active.begin_relaxation(7.0)

    assert relaxing.status is PlaneStatus.RELAXING
    assert relaxing.activation == pytest.approx(1.2)
    assert relaxing.activation_time == pytest.approx(2.0)
    assert relaxing.hysteresis == pytest.approx(0.4)
    assert relaxing.last_update_time == pytest.approx(7.0)


def test_plane_deactivate() -> None:
    """deactivate() must clear activation and activation time."""

    active = make_plane(
        status=PlaneStatus.ACTIVE,
        activation=1.2,
        activation_time=2.0,
        last_update_time=5.0,
    )

    inactive = active.deactivate(8.0)

    assert inactive.status is PlaneStatus.INACTIVE
    assert inactive.activation == pytest.approx(0.0)
    assert inactive.activation_time is None
    assert inactive.last_update_time == pytest.approx(8.0)


def test_plane_activate_rejects_backward_time() -> None:
    """A plane activation cannot precede its current state time."""

    plane = make_plane(
        last_update_time=5.0,
    )

    with pytest.raises(
        StateError,
        match="earlier",
    ):
        plane.activate(4.0)


def test_plane_relaxation_rejects_backward_time() -> None:
    """Relaxation cannot begin before the current state time."""

    plane = make_plane(
        last_update_time=5.0,
    )

    with pytest.raises(
        StateError,
        match="earlier",
    ):
        plane.begin_relaxation(4.0)


def test_plane_deactivation_rejects_backward_time() -> None:
    """Plane deactivation cannot move time backwards."""

    plane = make_plane(
        last_update_time=5.0,
    )

    with pytest.raises(
        StateError,
        match="earlier",
    ):
        plane.deactivate(4.0)


# ---------------------------------------------------------------------------
# PlaneState validation and immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activation",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_plane_rejects_invalid_activation(
    activation: float,
) -> None:
    """Plane activation must be finite and non-negative."""

    with pytest.raises(StateError):
        make_plane(
            activation=activation,
        )


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_plane_rejects_invalid_characteristic_time(
    value: float,
) -> None:
    """characteristic_time must be finite and strictly positive."""

    with pytest.raises(StateError):
        make_plane(
            characteristic_time=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_plane_rejects_invalid_relaxation_time(
    value: float,
) -> None:
    """Plane relaxation_time must be finite and strictly positive."""

    with pytest.raises(StateError):
        make_plane(
            relaxation_time=value,
        )


def test_plane_rejects_activation_time_after_update_time() -> None:
    """Plane activation timestamp cannot be in the future."""

    with pytest.raises(
        StateError,
        match="later",
    ):
        make_plane(
            activation_time=5.0,
            last_update_time=4.0,
        )


@pytest.mark.parametrize(
    "status",
    [
        "unknown",
        "",
        "ACTIVE",
        123,
    ],
)
def test_plane_rejects_invalid_status(
    status: object,
) -> None:
    """Only declared PlaneStatus values are valid."""

    with pytest.raises(
        StateError,
        match="Unknown plane status",
    ):
        PlaneState(
            plane_id="mechanical",
            status=status,  # type: ignore[arg-type]
        )


def test_plane_dataclass_is_frozen() -> None:
    """Direct mutation of a PlaneState must fail."""

    plane = make_plane()

    with pytest.raises(FrozenInstanceError):
        plane.activation = 2.0  # type: ignore[misc]


def test_plane_variables_mapping_is_read_only() -> None:
    """Plane variables must be immutable."""

    plane = PlaneState(
        plane_id="vascular",
        variables={
            "tone": 0.5,
        },
    )

    with pytest.raises(TypeError):
        plane.variables["tone"] = 1.0  # type: ignore[index]


def test_plane_metadata_mapping_is_read_only() -> None:
    """Plane metadata must be immutable."""

    plane = PlaneState(
        plane_id="vascular",
        metadata={
            "domain": "circulation",
        },
    )

    with pytest.raises(TypeError):
        plane.metadata["domain"] = "mechanics"  # type: ignore[index]


def test_plane_as_dict() -> None:
    """Plane serialization must preserve its dynamic state."""

    plane = PlaneState(
        plane_id="vascular",
        status=PlaneStatus.ACTIVE,
        activation=1.5,
        activation_time=2.0,
        last_update_time=5.0,
        characteristic_time=4.0,
        relaxation_time=8.0,
        hysteresis=0.3,
        variables={
            "tone": 0.7,
        },
        metadata={
            "domain": "circulation",
        },
    )

    data = plane.as_dict()

    assert data == {
        "plane_id": "vascular",
        "status": "active",
        "activation": 1.5,
        "activation_time": 2.0,
        "last_update_time": 5.0,
        "characteristic_time": 4.0,
        "relaxation_time": 8.0,
        "hysteresis": 0.3,
        "variables": {
            "tone": 0.7,
        },
        "metadata": {
            "domain": "circulation",
        },
    }


# ---------------------------------------------------------------------------
# SystemState: construction and aggregate values
# ---------------------------------------------------------------------------


def test_empty_system_state() -> None:
    """An empty state must be a valid initial system snapshot."""

    state = SystemState()

    assert state.time == pytest.approx(0.0)
    assert state.node_count == 0
    assert state.plane_count == 0
    assert state.total_capacity == pytest.approx(0.0)
    assert state.total_absolute_load == pytest.approx(0.0)
    assert state.total_available_reserve == pytest.approx(0.0)
    assert state.total_deficit == pytest.approx(0.0)
    assert state.critical_node_ids == ()
    assert state.exhausted_node_ids == ()
    assert state.active_plane_ids == ()
    assert state.d_fast_node_id is None


def test_system_state_construction() -> None:
    """SystemState must preserve valid node and plane mappings."""

    state = make_system()

    assert state.node_count == 2
    assert state.plane_count == 2
    assert tuple(state.nodes) == (
        "node_a",
        "node_b",
    )
    assert tuple(state.planes) == (
        "mechanical",
        "vascular",
    )


def test_system_total_capacity() -> None:
    """Total capacity must sum node capacities."""

    state = make_system()

    assert state.total_capacity == pytest.approx(180.0)


def test_system_total_absolute_load() -> None:
    """Total load must sum absolute node loads."""

    state = make_system()

    assert state.total_absolute_load == pytest.approx(100.0)


def test_system_total_available_reserve() -> None:
    """System reserve must sum only non-negative available reserves."""

    state = make_system()

    assert state.total_available_reserve == pytest.approx(80.0)


def test_system_total_deficit() -> None:
    """System deficit must sum node overloads."""

    mechanical = make_plane("mechanical")

    state = SystemState(
        nodes={
            "node_a": make_node(
                "node_a",
                capacity=100.0,
                load=130.0,
            ),
            "node_b": make_node(
                "node_b",
                capacity=50.0,
                load=-70.0,
            ),
        },
        planes={
            "mechanical": mechanical,
        },
    )

    assert state.total_deficit == pytest.approx(50.0)


def test_system_critical_node_ids() -> None:
    """Critical node IDs must include utilization at or above one."""

    state = SystemState(
        nodes={
            "safe": make_node(
                "safe",
                capacity=100.0,
                load=50.0,
            ),
            "critical": make_node(
                "critical",
                capacity=100.0,
                load=100.0,
            ),
            "exhausted": make_node(
                "exhausted",
                capacity=100.0,
                load=120.0,
            ),
        },
        planes={
            "mechanical": make_plane("mechanical"),
        },
    )

    assert state.critical_node_ids == (
        "critical",
        "exhausted",
    )


def test_system_exhausted_node_ids() -> None:
    """Exhausted node IDs must exclude a node exactly at capacity."""

    state = SystemState(
        nodes={
            "critical": make_node(
                "critical",
                capacity=100.0,
                load=100.0,
            ),
            "exhausted": make_node(
                "exhausted",
                capacity=100.0,
                load=120.0,
            ),
        },
        planes={
            "mechanical": make_plane("mechanical"),
        },
    )

    assert state.exhausted_node_ids == (
        "exhausted",
    )


def test_system_active_plane_ids() -> None:
    """Only continuously active planes must be reported."""

    state = SystemState(
        nodes={},
        planes={
            "mechanical": make_plane(
                "mechanical",
                status=PlaneStatus.ACTIVE,
                activation=1.0,
            ),
            "vascular": make_plane(
                "vascular",
                status=PlaneStatus.INACTIVE,
                activation=0.0,
            ),
            "disabled": make_plane(
                "disabled",
                status=PlaneStatus.DISABLED,
                activation=2.0,
            ),
        },
    )

    assert state.active_plane_ids == (
        "mechanical",
    )


def test_system_snapshot_d_fast_uses_maximum_utilization() -> None:
    """Snapshot D_fast must identify maximum current utilization."""

    state = SystemState(
        nodes={
            "node_a": make_node(
                "node_a",
                capacity=100.0,
                load=20.0,
            ),
            "node_b": make_node(
                "node_b",
                capacity=100.0,
                load=80.0,
            ),
            "node_c": make_node(
                "node_c",
                capacity=100.0,
                load=40.0,
            ),
        },
        planes={
            "mechanical": make_plane("mechanical"),
        },
    )

    assert state.d_fast_node_id == "node_b"


def test_system_snapshot_d_fast_uses_first_node_when_tied() -> None:
    """A utilization tie must resolve deterministically by insertion order."""

    state = SystemState(
        nodes={
            "node_a": make_node(
                "node_a",
                capacity=100.0,
                load=80.0,
            ),
            "node_b": make_node(
                "node_b",
                capacity=50.0,
                load=-40.0,
            ),
        },
        planes={
            "mechanical": make_plane("mechanical"),
        },
    )

    assert state.d_fast_node_id == "node_a"


# ---------------------------------------------------------------------------
# SystemState access and plane membership
# ---------------------------------------------------------------------------


def test_system_node_lookup() -> None:
    """node() must return the requested NodeState."""

    state = make_system()

    assert state.node("node_a") is state.nodes["node_a"]


def test_system_plane_lookup() -> None:
    """plane() must return the requested PlaneState."""

    state = make_system()

    assert state.plane("mechanical") is state.planes["mechanical"]


def test_system_unknown_node_lookup_raises() -> None:
    """Unknown node identifiers must fail explicitly."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="Unknown node_id",
    ):
        state.node("missing")


def test_system_unknown_plane_lookup_raises() -> None:
    """Unknown plane identifiers must fail explicitly."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="Unknown plane_id",
    ):
        state.plane("missing")


def test_system_nodes_in_plane() -> None:
    """nodes_in_plane() must support overlapping plane membership."""

    state = make_system()

    mechanical_nodes = state.nodes_in_plane(
        "mechanical",
    )
    vascular_nodes = state.nodes_in_plane(
        "vascular",
    )

    assert tuple(
        node.node_id
        for node in mechanical_nodes
    ) == (
        "node_a",
        "node_b",
    )

    assert tuple(
        node.node_id
        for node in vascular_nodes
    ) == (
        "node_b",
    )


def test_system_nodes_in_unknown_plane_raises() -> None:
    """Membership lookup requires a registered plane."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="Unknown plane_id",
    ):
        state.nodes_in_plane("metabolic")


# ---------------------------------------------------------------------------
# SystemState immutable update methods
# ---------------------------------------------------------------------------


def test_system_with_node_replaces_existing_node() -> None:
    """with_node() must return a new state with one node replaced."""

    original = make_system()

    updated_node = original.node(
        "node_a"
    ).with_load(
        90.0,
        time=2.0,
    )

    updated = original.with_node(
        updated_node,
    )

    assert original.node("node_a").load == pytest.approx(40.0)
    assert updated.node("node_a").load == pytest.approx(90.0)
    assert updated.node("node_b") is original.node("node_b")
    assert updated.time == pytest.approx(2.0)


def test_system_with_node_adds_new_node() -> None:
    """with_node() must support adding a new valid node."""

    original = make_system()

    new_node = make_node(
        "node_c",
        plane_ids=("vascular",),
        capacity=50.0,
        load=10.0,
    )

    updated = original.with_node(
        new_node,
    )

    assert "node_c" not in original.nodes
    assert updated.node_count == 3
    assert updated.node("node_c") == new_node


def test_system_with_node_rejects_unknown_plane_reference() -> None:
    """A newly added node cannot reference an unregistered plane."""

    state = make_system()

    new_node = make_node(
        "node_c",
        plane_ids=("metabolic",),
    )

    with pytest.raises(
        StateError,
        match="unknown planes",
    ):
        state.with_node(new_node)


def test_system_with_node_rejects_non_node_object() -> None:
    """with_node() requires a NodeState instance."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="NodeState",
    ):
        state.with_node("node_a")  # type: ignore[arg-type]


def test_system_without_node() -> None:
    """without_node() must remove a node immutably."""

    original = make_system()
    updated = original.without_node("node_a")

    assert "node_a" in original.nodes
    assert "node_a" not in updated.nodes
    assert updated.node_count == 1


def test_system_without_unknown_node_raises() -> None:
    """Removing an unknown node must fail."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="Unknown node_id",
    ):
        state.without_node("missing")


def test_system_with_plane_replaces_existing_plane() -> None:
    """with_plane() must replace a plane immutably."""

    original = make_system()

    updated_plane = original.plane(
        "vascular"
    ).activate(
        3.0,
        activation=1.2,
    )

    updated = original.with_plane(
        updated_plane,
    )

    assert original.plane("vascular").activation == pytest.approx(0.0)
    assert updated.plane("vascular").activation == pytest.approx(1.2)
    assert updated.time == pytest.approx(3.0)


def test_system_with_plane_adds_new_plane() -> None:
    """with_plane() must support registering a new plane."""

    original = make_system()

    metabolic = make_plane(
        "metabolic",
    )

    updated = original.with_plane(
        metabolic,
    )

    assert "metabolic" not in original.planes
    assert updated.plane_count == 3
    assert updated.plane("metabolic") == metabolic


def test_system_with_plane_rejects_non_plane_object() -> None:
    """with_plane() requires a PlaneState instance."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="PlaneState",
    ):
        state.with_plane("vascular")  # type: ignore[arg-type]


def test_system_without_unreferenced_plane() -> None:
    """An unreferenced plane may be removed directly."""

    original = make_system().with_plane(
        make_plane("metabolic")
    )

    updated = original.without_plane(
        "metabolic",
    )

    assert "metabolic" in original.planes
    assert "metabolic" not in updated.planes


def test_system_rejects_removal_of_referenced_plane() -> None:
    """A referenced plane cannot be removed without detaching nodes."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="referenced by nodes",
    ):
        state.without_plane("mechanical")


def test_system_removes_plane_and_detaches_nodes() -> None:
    """detach_nodes=True must remove plane references from nodes."""

    original = make_system()

    updated = original.without_plane(
        "vascular",
        detach_nodes=True,
    )

    assert "vascular" not in updated.planes
    assert updated.node("node_a").plane_ids == (
        "mechanical",
    )
    assert updated.node("node_b").plane_ids == (
        "mechanical",
    )

    assert original.node("node_b").plane_ids == (
        "mechanical",
        "vascular",
    )


def test_system_without_unknown_plane_raises() -> None:
    """Removing an unknown plane must fail."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="Unknown plane_id",
    ):
        state.without_plane("missing")


def test_system_with_global_variable() -> None:
    """Global variables must update immutably."""

    original = SystemState(
        variables={
            "external_stress": 0.2,
        },
    )

    updated = original.with_variable(
        "temperature",
        37.0,
    )

    assert "temperature" not in original.variables
    assert updated.variables["external_stress"] == pytest.approx(0.2)
    assert updated.variables["temperature"] == pytest.approx(37.0)


def test_system_at_time() -> None:
    """at_time() must advance only the global clock."""

    original = make_system(
        time=2.0,
    )

    updated = original.at_time(10.0)

    assert original.time == pytest.approx(2.0)
    assert updated.time == pytest.approx(10.0)
    assert updated.nodes == original.nodes
    assert updated.planes == original.planes


def test_system_at_time_allows_same_time() -> None:
    """Producing another snapshot at the same time is valid."""

    state = make_system(
        time=2.0,
    )

    updated = state.at_time(2.0)

    assert updated.time == pytest.approx(2.0)


def test_system_at_time_rejects_backward_time() -> None:
    """The global system clock cannot move backwards."""

    state = make_system(
        time=5.0,
    )

    with pytest.raises(
        StateError,
        match="cannot move backwards",
    ):
        state.at_time(4.0)


# ---------------------------------------------------------------------------
# SystemState vectors
# ---------------------------------------------------------------------------


def test_system_capacity_vector() -> None:
    """Capacity vector must preserve node insertion order."""

    state = make_system()

    vector = state.node_capacity_vector()

    assert np.allclose(
        vector,
        [100.0, 80.0],
    )


def test_system_load_vector() -> None:
    """Load vector must contain signed node loads."""

    state = make_system()

    vector = state.node_load_vector()

    assert np.allclose(
        vector,
        [40.0, 60.0],
    )


def test_system_reserve_vector() -> None:
    """Reserve vector must contain signed reserves."""

    state = make_system()

    vector = state.node_reserve_vector()

    assert np.allclose(
        vector,
        [60.0, 20.0],
    )


def test_system_utilization_vector() -> None:
    """Utilization vector must contain non-negative ratios."""

    state = make_system()

    vector = state.node_utilization_vector()

    assert np.allclose(
        vector,
        [0.4, 0.75],
    )


def test_system_vector_accepts_explicit_node_order() -> None:
    """Explicit IDs must determine vector ordering."""

    state = make_system()

    vector = state.node_load_vector(
        [
            "node_b",
            "node_a",
        ]
    )

    assert np.allclose(
        vector,
        [60.0, 40.0],
    )


def test_system_vector_rejects_unknown_node() -> None:
    """Vector extraction must reject unknown identifiers."""

    state = make_system()

    with pytest.raises(
        StateError,
        match="Unknown node identifiers",
    ):
        state.node_load_vector(
            [
                "node_a",
                "missing",
            ]
        )


@pytest.mark.parametrize(
    "method_name",
    [
        "node_capacity_vector",
        "node_load_vector",
        "node_reserve_vector",
        "node_utilization_vector",
    ],
)
def test_system_vectors_are_read_only(
    method_name: str,
) -> None:
    """Numerical vectors returned by SystemState must be immutable."""

    state = make_system()
    method = getattr(
        state,
        method_name,
    )

    vector = method()

    assert not vector.flags.writeable

    with pytest.raises(ValueError):
        vector[0] = 999.0


# ---------------------------------------------------------------------------
# SystemState validation and immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "time",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_system_rejects_invalid_time(
    time: float,
) -> None:
    """System time must be finite and non-negative."""

    with pytest.raises(StateError):
        SystemState(
            time=time,
        )


def test_system_rejects_node_mapping_key_mismatch() -> None:
    """Mapping keys must equal NodeState.node_id."""

    node = make_node(
        "node_a",
    )

    with pytest.raises(
        StateError,
        match="does not match",
    ):
        SystemState(
            nodes={
                "wrong_id": node,
            },
            planes={
                "mechanical": make_plane("mechanical"),
            },
        )


def test_system_rejects_plane_mapping_key_mismatch() -> None:
    """Mapping keys must equal PlaneState.plane_id."""

    plane = make_plane(
        "mechanical",
    )

    with pytest.raises(
        StateError,
        match="does not match",
    ):
        SystemState(
            planes={
                "wrong_id": plane,
            },
        )


def test_system_rejects_non_node_mapping_value() -> None:
    """Every value in nodes must be a NodeState."""

    with pytest.raises(
        StateError,
        match="NodeState",
    ):
        SystemState(
            nodes={
                "node_a": "not a node",  # type: ignore[dict-item]
            },
        )


def test_system_rejects_non_plane_mapping_value() -> None:
    """Every value in planes must be a PlaneState."""

    with pytest.raises(
        StateError,
        match="PlaneState",
    ):
        SystemState(
            planes={
                "mechanical": "not a plane",  # type: ignore[dict-item]
            },
        )


def test_system_rejects_unknown_node_plane_reference() -> None:
    """Every plane referenced by a node must be registered."""

    node = make_node(
        "node_a",
        plane_ids=(
            "mechanical",
            "vascular",
        ),
    )

    with pytest.raises(
        StateError,
        match="unknown planes",
    ):
        SystemState(
            nodes={
                "node_a": node,
            },
            planes={
                "mechanical": make_plane("mechanical"),
            },
        )


def test_system_rejects_node_updated_after_system_time() -> None:
    """A component snapshot cannot come from the system future."""

    node = make_node(
        "node_a",
        last_update_time=5.0,
    )

    with pytest.raises(
        StateError,
        match="later than system time",
    ):
        SystemState(
            time=4.0,
            nodes={
                "node_a": node,
            },
            planes={
                "mechanical": make_plane("mechanical"),
            },
        )


def test_system_rejects_plane_updated_after_system_time() -> None:
    """A plane snapshot cannot come from the system future."""

    plane = make_plane(
        "mechanical",
        last_update_time=5.0,
    )

    with pytest.raises(
        StateError,
        match="later than system time",
    ):
        SystemState(
            time=4.0,
            planes={
                "mechanical": plane,
            },
        )


def test_system_dataclass_is_frozen() -> None:
    """Direct mutation of the global state must fail."""

    state = make_system()

    with pytest.raises(FrozenInstanceError):
        state.time = 10.0  # type: ignore[misc]


def test_system_nodes_mapping_is_read_only() -> None:
    """The stored node registry must be immutable."""

    state = make_system()

    with pytest.raises(TypeError):
        state.nodes["node_c"] = make_node(  # type: ignore[index]
            "node_c",
        )


def test_system_planes_mapping_is_read_only() -> None:
    """The stored plane registry must be immutable."""

    state = make_system()

    with pytest.raises(TypeError):
        state.planes["metabolic"] = make_plane(  # type: ignore[index]
            "metabolic"
        )


def test_system_variables_mapping_is_read_only() -> None:
    """Global continuous variables must be immutable."""

    state = SystemState(
        variables={
            "stress": 0.5,
        },
    )

    with pytest.raises(TypeError):
        state.variables["stress"] = 1.0  # type: ignore[index]


def test_system_metadata_mapping_is_read_only() -> None:
    """System metadata must be immutable."""

    state = SystemState(
        metadata={
            "model": "test",
        },
    )

    with pytest.raises(TypeError):
        state.metadata["model"] = "production"  # type: ignore[index]


def test_system_input_mappings_are_copied() -> None:
    """Changing constructor dictionaries must not alter SystemState."""

    plane = make_plane(
        "mechanical",
    )
    node = make_node(
        "node_a",
    )

    nodes = {
        "node_a": node,
    }
    planes = {
        "mechanical": plane,
    }
    variables = {
        "stress": 0.5,
    }
    metadata = {
        "model": "test",
    }

    state = SystemState(
        nodes=nodes,
        planes=planes,
        variables=variables,
        metadata=metadata,
    )

    nodes.clear()
    planes.clear()
    variables["stress"] = 2.0
    metadata["model"] = "changed"

    assert state.node_count == 1
    assert state.plane_count == 1
    assert state.variables["stress"] == pytest.approx(0.5)
    assert state.metadata["model"] == "test"


# ---------------------------------------------------------------------------
# SystemState serialization
# ---------------------------------------------------------------------------


def test_system_as_dict_contains_components_and_summary() -> None:
    """System serialization must include state and aggregate information."""

    mechanical = make_plane(
        "mechanical",
        status=PlaneStatus.ACTIVE,
        activation=1.0,
    )

    node_a = make_node(
        "node_a",
        capacity=100.0,
        load=120.0,
    )
    node_b = make_node(
        "node_b",
        capacity=80.0,
        load=40.0,
    )

    state = SystemState(
        nodes={
            "node_a": node_a,
            "node_b": node_b,
        },
        planes={
            "mechanical": mechanical,
        },
        variables={
            "external_stress": 0.4,
        },
        metadata={
            "scenario": "test",
        },
    )

    data = state.as_dict()

    assert data["time"] == pytest.approx(0.0)

    assert set(data["nodes"]) == {
        "node_a",
        "node_b",
    }
    assert set(data["planes"]) == {
        "mechanical",
    }

    assert data["variables"] == {
        "external_stress": 0.4,
    }
    assert data["metadata"] == {
        "scenario": "test",
    }

    summary = data["summary"]

    assert summary["node_count"] == 2
    assert summary["plane_count"] == 1
    assert summary["total_capacity"] == pytest.approx(180.0)
    assert summary["total_absolute_load"] == pytest.approx(160.0)
    assert summary["total_available_reserve"] == pytest.approx(40.0)
    assert summary["total_deficit"] == pytest.approx(20.0)
    assert summary["critical_node_ids"] == [
        "node_a",
    ]
    assert summary["exhausted_node_ids"] == [
        "node_a",
    ]
    assert summary["active_plane_ids"] == [
        "mechanical",
    ]
    assert summary["d_fast_node_id"] == "node_a"