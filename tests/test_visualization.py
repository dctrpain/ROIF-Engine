from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import to_hex
import pytest

from core.element import Element
from core.material import (
    Material,
    MaterialParameters,
    MaterialState,
)
from core.network import Network
from core.node import Node
from visualization import NetworkPlotter
from visualization.styles import (
    CURRENT_ELEMENT_STYLE,
    REFERENCE_ELEMENT_STYLE,
)


def make_material(
    name: str,
    *,
    stiffness: float = 10.0,
    reference_force: float = 1.0,
    damage: float = 0.0,
    fatigue: float = 0.0,
    remodeling: float = 1.0,
) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=stiffness,
            damping=0.0,
            failure_threshold=100.0,
            reference_force=reference_force,
            reference_strain=1.0,
        ),
        state=MaterialState(
            damage=damage,
            fatigue=fatigue,
            remodeling=remodeling,
        ),
        reference_length=1.0,
    )


def make_network() -> tuple[Network, Element, Element]:
    node_a = Node(position=[0.0, 0.0], fixed=True, node_id="A")
    node_b = Node(position=[1.2, 0.0], node_id="B")
    node_c = Node(position=[2.5, 0.0], node_id="C")

    first = Element(
        node_a=node_a,
        node_b=node_b,
        material=make_material(
            "first_material",
            stiffness=10.0,
            reference_force=1.0,
            damage=0.10,
            fatigue=0.20,
            remodeling=0.80,
        ),
        rest_length=1.0,
        element_id="AB",
    )
    second = Element(
        node_a=node_b,
        node_b=node_c,
        material=make_material(
            "second_material",
            stiffness=20.0,
            reference_force=1.0,
            damage=0.70,
            fatigue=0.60,
            remodeling=0.95,
        ),
        rest_length=1.0,
        element_id="BC",
    )

    network = Network()

    for node in (node_a, node_b, node_c):
        network.add_node(node)

    network.add_element(first)
    network.add_element(second)

    return network, first, second


def current_lines(axes):
    return {
        str(line.get_gid()).split(":", 1)[1]: line
        for line in axes.lines
        if str(line.get_gid()).startswith("current:")
    }


def test_network_plotter_draws_network_without_modifying_state():
    network, _, _ = make_network()

    initial_positions = [
        node.position.copy()
        for node in network.nodes
    ]

    plotter = NetworkPlotter(network)
    figure, axes = plotter.plot()

    assert figure is not None
    assert axes is not None
    assert len(axes.lines) == 4
    assert len(axes.collections) == 2

    for node, initial_position in zip(
        network.nodes,
        initial_positions,
    ):
        assert (node.position == initial_position).all()


def test_network_plotter_uses_consistent_geometry_styles():
    network, _, _ = make_network()
    plotter = NetworkPlotter(network)
    _, axes = plotter.plot(mode="geometry")

    reference_lines = [
        line
        for line in axes.lines
        if str(line.get_gid()).startswith("reference:")
    ]
    current_geometry_lines = [
        line
        for line in axes.lines
        if str(line.get_gid()).startswith("current:")
    ]

    assert {
        to_hex(line.get_color())
        for line in reference_lines
    } == {to_hex(REFERENCE_ELEMENT_STYLE["color"])}

    assert {
        to_hex(line.get_color())
        for line in current_geometry_lines
    } == {to_hex(CURRENT_ELEMENT_STYLE["color"])}


def test_force_mode_maps_force_to_width_and_color():
    network, first, second = make_network()
    network.clear_forces()
    network.assemble_element_forces(include_active=False)

    plotter = NetworkPlotter(network)
    figure, axes = plotter.plot(
        mode="force",
        show_reference=False,
    )
    lines = current_lines(axes)

    force_by_id = {
        "AB": abs(plotter.element_force(first)),
        "BC": abs(plotter.element_force(second)),
    }
    stronger_id = max(force_by_id, key=force_by_id.get)
    weaker_id = min(force_by_id, key=force_by_id.get)

    assert force_by_id[stronger_id] > force_by_id[weaker_id]
    assert (
        lines[stronger_id].get_linewidth()
        > lines[weaker_id].get_linewidth()
    )
    assert (
        to_hex(lines["AB"].get_color())
        != to_hex(lines["BC"].get_color())
    )
    assert len(figure.axes) == 2


@pytest.mark.parametrize(
    ("mode", "first_value", "second_value"),
    [
        ("damage", 0.10, 0.70),
        ("fatigue", 0.20, 0.60),
        ("remodeling", 0.80, 0.95),
    ],
)
def test_material_modes_map_values_to_colors(
    mode,
    first_value,
    second_value,
):
    network, first, second = make_network()
    plotter = NetworkPlotter(network)
    figure, axes = plotter.plot(
        mode=mode,
        show_reference=False,
    )
    lines = current_lines(axes)

    assert plotter.element_material_value(
        first,
        mode,
    ) == pytest.approx(first_value)

    assert plotter.element_material_value(
        second,
        mode,
    ) == pytest.approx(second_value)

    assert (
        to_hex(lines["AB"].get_color())
        != to_hex(lines["BC"].get_color())
    )
    assert len(figure.axes) == 2


def test_integrity_mode_uses_material_integrity_method():
    network, first, second = make_network()
    plotter = NetworkPlotter(network)

    first_integrity = plotter.element_material_value(
        first,
        "integrity",
    )
    second_integrity = plotter.element_material_value(
        second,
        "integrity",
    )

    assert first_integrity == pytest.approx(
        first.material.integrity()
    )
    assert second_integrity == pytest.approx(
        second.material.integrity()
    )
    assert first_integrity > second_integrity

    _, axes = plotter.plot(
        mode="integrity",
        show_reference=False,
    )
    lines = current_lines(axes)

    assert (
        to_hex(lines["AB"].get_color())
        != to_hex(lines["BC"].get_color())
    )


def test_network_plotter_rejects_unknown_mode():
    network, _, _ = make_network()
    plotter = NetworkPlotter(network)

    with pytest.raises(ValueError, match="unsupported plot mode"):
        plotter.plot(mode="energy")
