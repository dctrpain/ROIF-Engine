from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core.recorder import RecordingFrame
from visualization import (
    ElementVisualRole,
    ViewerLayers,
    ViewerMode,
    ViewerSelection,
    ViewerState,
    viewer_states_from_frames,
)


class DummyNode:
    def __init__(
        self,
        node_id: str,
        position: list[float],
        *,
        fixed: bool = False,
    ) -> None:
        self.id = node_id
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.zeros_like(self.position)
        self.force = np.zeros_like(self.position)
        self.mass = 1.0
        self.fixed = fixed


class DummyMaterial:
    def __init__(
        self,
        *,
        name: str = "material",
        damage: float = 0.0,
        fatigue: float = 0.0,
        integrity: float = 1.0,
        failed: bool = False,
        activation: float = 0.0,
        reference_force: float = 10.0,
    ) -> None:
        self.name = name
        self.state = SimpleNamespace(
            damage=damage,
            fatigue=fatigue,
            integrity=integrity,
            failed=failed,
            activation=activation,
        )
        self.parameters = SimpleNamespace(
            reference_force=reference_force,
        )


class MuscleMaterial(DummyMaterial):
    pass


class DummyElement:
    def __init__(
        self,
        node_a: DummyNode,
        node_b: DummyNode,
        *,
        element_id: str = "E",
        material: Any | None = None,
        tension_only: bool = False,
        compression_only: bool = False,
        enabled: bool = True,
        force: float = 0.0,
    ) -> None:
        self.id = element_id
        self.name = f"Element_{element_id}"
        self.node_a = node_a
        self.node_b = node_b
        self.material = material or DummyMaterial()
        self.tension_only = tension_only
        self.compression_only = compression_only
        self.enabled = enabled
        self.rest_length = 1.0
        self.last_force = force
        self.force_components = SimpleNamespace(
            active=0.0,
            total=force,
        )

    def current_length(self) -> float:
        return float(
            np.linalg.norm(
                self.node_b.position
                - self.node_a.position
            )
        )

    def strain(self) -> float:
        return (
            self.current_length() - self.rest_length
        ) / self.rest_length


class DummyNetwork:
    def __init__(
        self,
        nodes: list[DummyNode],
        elements: list[DummyElement],
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.time = 0.5
        self.step_index = 25


def make_network() -> DummyNetwork:
    node_a = DummyNode(
        "A",
        [0.0, 0.0],
        fixed=True,
    )
    node_b = DummyNode(
        "B",
        [1.1, 0.0],
    )
    element = DummyElement(
        node_a,
        node_b,
        element_id="AB",
        force=5.0,
    )
    return DummyNetwork(
        [node_a, node_b],
        [element],
    )


def test_viewer_mode_values_are_stable() -> None:
    assert ViewerMode.GEOMETRY.value == "geometry"
    assert ViewerMode.FORCE.value == "force"
    assert ViewerMode.CAPACITY.value == "capacity"
    assert ViewerMode.RESERVE.value == "reserve"
    assert ViewerMode.CASCADE.value == "cascade"


def test_selection_defaults_to_empty() -> None:
    selection = ViewerSelection()

    assert selection.empty
    assert selection.node_id is None
    assert selection.element_id is None


def test_selection_rejects_node_and_element_together() -> None:
    with pytest.raises(ValueError):
        ViewerSelection(
            node_id="A",
            element_id="AB",
        )


def test_layers_defaults() -> None:
    layers = ViewerLayers()

    assert layers.geometry is True
    assert layers.reference_geometry is True
    assert layers.labels is True
    assert layers.capacity is False
    assert layers.reserve is False
    assert layers.cascade is False


def test_state_requires_source() -> None:
    with pytest.raises(ValueError):
        ViewerState()


def test_state_from_network() -> None:
    network = make_network()

    state = ViewerState.from_network(network)

    assert state.source_network is network
    assert state.is_recorded is False
    assert state.frame_index is None
    assert state.step_index == 25
    assert state.time == pytest.approx(0.5)
    assert state.dimension == 2
    assert len(state.nodes) == 2
    assert len(state.elements) == 1


def test_state_from_frame() -> None:
    network = make_network()
    frame = RecordingFrame(
        index=3,
        step_index=15,
        time=0.3,
        network=network,
        metadata={"source": "test"},
    )

    state = ViewerState.from_frame(frame)

    assert state.source_network is network
    assert state.is_recorded is True
    assert state.frame_index == 3
    assert state.step_index == 15
    assert state.time == pytest.approx(0.3)
    assert state.metadata == {"source": "test"}


def test_frame_metadata_can_be_extended() -> None:
    network = make_network()
    frame = RecordingFrame(
        index=0,
        step_index=0,
        time=0.0,
        network=network,
        metadata={"source": "frame"},
    )

    state = ViewerState.from_frame(
        frame,
        metadata={"purpose": "viewer"},
    )

    assert state.metadata == {
        "source": "frame",
        "purpose": "viewer",
    }


def test_state_rejects_conflicting_frame_network() -> None:
    first = make_network()
    second = make_network()
    frame = RecordingFrame(
        index=0,
        step_index=0,
        time=0.0,
        network=first,
    )

    with pytest.raises(ValueError):
        ViewerState(
            frame=frame,
            network=second,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("geometry", ViewerMode.GEOMETRY),
        ("force", ViewerMode.FORCE),
        ("capacity", ViewerMode.CAPACITY),
        ("reserve", ViewerMode.RESERVE),
        ("cascade", ViewerMode.CASCADE),
    ],
)
def test_mode_coercion(
    value: str,
    expected: ViewerMode,
) -> None:
    state = ViewerState.from_network(
        make_network(),
        mode=value,
    )

    assert state.mode is expected


def test_bad_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        ViewerState.from_network(
            make_network(),
            mode="unknown",
        )


def test_with_mode_returns_new_state() -> None:
    original = ViewerState.from_network(
        make_network()
    )

    changed = original.with_mode(
        ViewerMode.FORCE
    )

    assert original.mode is ViewerMode.GEOMETRY
    assert changed.mode is ViewerMode.FORCE
    assert changed is not original


def test_with_layers_returns_new_state() -> None:
    original = ViewerState.from_network(
        make_network()
    )

    changed = original.with_layers(
        capacity=True,
        reserve=True,
    )

    assert original.layers.capacity is False
    assert original.layers.reserve is False
    assert changed.layers.capacity is True
    assert changed.layers.reserve is True


def test_with_layers_rejects_unknown_field() -> None:
    state = ViewerState.from_network(
        make_network()
    )

    with pytest.raises(ValueError):
        state.with_layers(unknown=True)


@pytest.mark.parametrize(
    "bad_value",
    [1, 0, "yes", None],
)
def test_with_layers_requires_bool(
    bad_value: Any,
) -> None:
    state = ViewerState.from_network(
        make_network()
    )

    with pytest.raises(TypeError):
        state.with_layers(
            capacity=bad_value,
        )


def test_node_selection() -> None:
    state = ViewerState.from_network(
        make_network()
    )

    selected = state.select_node("A")

    assert selected.selection.node_id == "A"
    assert selected.selection.element_id is None
    assert selected.node_by_id("A").id == "A"


def test_element_selection() -> None:
    state = ViewerState.from_network(
        make_network()
    )

    selected = state.select_element("AB")

    assert selected.selection.node_id is None
    assert selected.selection.element_id == "AB"
    assert selected.element_by_id("AB").id == "AB"


def test_clear_selection() -> None:
    selected = ViewerState.from_network(
        make_network()
    ).select_node("A")

    cleared = selected.clear_selection()

    assert cleared.selection.empty


def test_unknown_node_is_rejected() -> None:
    state = ViewerState.from_network(
        make_network()
    )

    with pytest.raises(KeyError):
        state.select_node("missing")


def test_unknown_element_is_rejected() -> None:
    state = ViewerState.from_network(
        make_network()
    )

    with pytest.raises(KeyError):
        state.select_element("missing")


def test_generic_element_role() -> None:
    network = make_network()

    assert (
        ViewerState.element_role(
            network.elements[0]
        )
        is ElementVisualRole.GENERIC
    )


def test_tension_element_role() -> None:
    network = make_network()
    network.elements[0].tension_only = True

    assert (
        ViewerState.element_role(
            network.elements[0]
        )
        is ElementVisualRole.TENSION
    )


def test_compression_element_role() -> None:
    network = make_network()
    network.elements[0].compression_only = True

    assert (
        ViewerState.element_role(
            network.elements[0]
        )
        is ElementVisualRole.COMPRESSION
    )


def test_disabled_element_role() -> None:
    network = make_network()
    network.elements[0].enabled = False

    assert (
        ViewerState.element_role(
            network.elements[0]
        )
        is ElementVisualRole.DISABLED
    )


def test_failed_element_role_has_priority() -> None:
    network = make_network()
    element = network.elements[0]
    element.enabled = False
    element.material.state.failed = True

    assert (
        ViewerState.element_role(element)
        is ElementVisualRole.FAILED
    )


def test_muscle_material_role() -> None:
    network = make_network()
    network.elements[0].material = MuscleMaterial(
        name="muscle",
    )

    assert (
        ViewerState.element_role(
            network.elements[0]
        )
        is ElementVisualRole.ACTIVE_TENSION
    )


def test_explicit_rigid_role() -> None:
    network = make_network()
    network.elements[0].mechanical_role = "rigid"

    assert (
        ViewerState.element_role(
            network.elements[0]
        )
        is ElementVisualRole.RIGID
    )


def test_node_snapshot_is_independent() -> None:
    network = make_network()
    state = ViewerState.from_network(network)

    snapshot = state.node_snapshot(
        network.nodes[0]
    )
    snapshot["position"][0] = 100.0

    assert network.nodes[0].position[0] == 0.0


def test_element_snapshot() -> None:
    network = make_network()
    state = ViewerState.from_network(network)

    snapshot = state.element_snapshot(
        network.elements[0]
    )

    assert snapshot["id"] == "AB"
    assert snapshot["current_length"] == pytest.approx(1.1)
    assert snapshot["strain"] == pytest.approx(0.1)
    assert snapshot["force"] == pytest.approx(5.0)
    assert snapshot["load_ratio"] == pytest.approx(0.5)
    assert snapshot["capacity"] == pytest.approx(1.0)
    assert snapshot["reserve"] == pytest.approx(0.5)
    assert snapshot["role"] == "generic"


def test_damage_reduces_capacity_and_reserve() -> None:
    network = make_network()
    element = network.elements[0]
    element.material.state.damage = 0.25

    snapshot = ViewerState.from_network(
        network
    ).element_snapshot(element)

    assert snapshot["capacity"] == pytest.approx(0.75)
    assert snapshot["reserve"] == pytest.approx(0.25)


def test_network_snapshot() -> None:
    state = ViewerState.from_network(
        make_network(),
        metadata={"purpose": "test"},
    )

    snapshot = state.network_snapshot()

    assert snapshot["viewer_version"] == "3.0.0"
    assert snapshot["recorded"] is False
    assert snapshot["mode"] == "geometry"
    assert snapshot["dimension"] == 2
    assert len(snapshot["nodes"]) == 2
    assert len(snapshot["elements"]) == 1
    assert snapshot["metadata"] == {
        "purpose": "test"
    }


def test_viewer_states_from_frames() -> None:
    first_network = make_network()
    second_network = make_network()

    frames = [
        RecordingFrame(
            index=0,
            step_index=0,
            time=0.0,
            network=first_network,
        ),
        RecordingFrame(
            index=1,
            step_index=5,
            time=0.1,
            network=second_network,
        ),
    ]

    states = viewer_states_from_frames(
        frames,
        mode=ViewerMode.FORCE,
    )

    assert len(states) == 2
    assert states[0].frame_index == 0
    assert states[1].frame_index == 1
    assert all(
        state.mode is ViewerMode.FORCE
        for state in states
    )