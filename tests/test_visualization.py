from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import to_hex
import pytest

from core.element import Element
from core.material import Material, MaterialParameters
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
        reference_length=1.0,
    )


def test_network_plotter_draws_network_without_modifying_state():
    fixed = Node(
        position=[0.0, 0.0],
        fixed=True,
        node_id="fixed",
    )
    free = Node(
        position=[1.0, 0.0],
        node_id="free",
    )

    element = Element(
        node_a=fixed,
        node_b=free,
        material=make_material("visualization_test_material"),
        rest_length=1.0,
        element_id="test_element",
    )

    network = Network()
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(element)

    initial_fixed = fixed.position.copy()
    initial_free = free.position.copy()

    plotter = NetworkPlotter(network)
    figure, axes = plotter.plot()

    assert figure is not None
    assert axes is not None
    assert len(axes.lines) == 2
    assert len(axes.collections) == 2

    assert (fixed.position == initial_fixed).all()
    assert (free.position == initial_free).all()


def test_network_plotter_uses_consistent_geometry_styles():
    node_a = Node(position=[0.0, 0.0], fixed=True, node_id="A")
    node_b = Node(position=[1.0, 0.0], node_id="B")
    node_c = Node(position=[2.0, 0.0], node_id="C")

    network = Network()

    for node in (node_a, node_b, node_c):
        network.add_node(node)

    network.add_element(
        Element(
            node_a=node_a,
            node_b=node_b,
            material=make_material("material_ab"),
            rest_length=1.0,
            element_id="AB",
        )
    )
    network.add_element(
        Element(
            node_a=node_b,
            node_b=node_c,
            material=make_material("material_bc"),
            rest_length=1.0,
            element_id="BC",
        )
    )

    plotter = NetworkPlotter(network)
    _, axes = plotter.plot(mode="geometry")

    reference_lines = [
        line
        for line in axes.lines
        if str(line.get_gid()).startswith("reference:")
    ]
    current_lines = [
        line
        for line in axes.lines
        if str(line.get_gid()).startswith("current:")
    ]

    assert len(reference_lines) == 2
    assert len(current_lines) == 2

    assert {
        to_hex(line.get_color())
        for line in reference_lines
    } == {to_hex(REFERENCE_ELEMENT_STYLE["color"])}

    assert {
        to_hex(line.get_color())
        for line in current_lines
    } == {to_hex(CURRENT_ELEMENT_STYLE["color"])}


def test_force_mode_maps_force_to_width_and_stimulus_to_color():
    fixed = Node(position=[0.0, 0.0], fixed=True, node_id="A")
    middle = Node(position=[1.2, 0.0], node_id="B")
    free = Node(position=[2.5, 0.0], node_id="C")

    first = Element(
        node_a=fixed,
        node_b=middle,
        material=make_material(
            "first_material",
            stiffness=10.0,
            reference_force=1.0,
        ),
        rest_length=1.0,
        element_id="AB",
    )
    second = Element(
        node_a=middle,
        node_b=free,
        material=make_material(
            "second_material",
            stiffness=20.0,
            reference_force=1.0,
        ),
        rest_length=1.0,
        element_id="BC",
    )

    network = Network()

    for node in (fixed, middle, free):
        network.add_node(node)

    network.add_element(first)
    network.add_element(second)

    network.clear_forces()
    network.assemble_element_forces(include_active=False)

    initial_positions = [
        node.position.copy()
        for node in network.nodes
    ]

    plotter = NetworkPlotter(network)
    figure, axes = plotter.plot(
        mode="force",
        show_reference=False,
        show_colorbar=True,
    )

    current_lines = {
        str(line.get_gid()).split(":", 1)[1]: line
        for line in axes.lines
        if str(line.get_gid()).startswith("current:")
    }

    assert set(current_lines) == {"AB", "BC"}

    force_ab = abs(plotter.element_force(first))
    force_bc = abs(plotter.element_force(second))

    assert force_bc > force_ab
    assert (
        current_lines["BC"].get_linewidth()
        > current_lines["AB"].get_linewidth()
    )

    assert (
        to_hex(current_lines["BC"].get_color())
        != to_hex(current_lines["AB"].get_color())
    )

    # Main axes plus one colorbar axes.
    assert len(figure.axes) == 2

    for node, initial_position in zip(
        network.nodes,
        initial_positions,
    ):
        assert (node.position == initial_position).all()


def test_network_plotter_rejects_unknown_mode():
    fixed = Node(position=[0.0, 0.0], fixed=True)
    free = Node(position=[1.0, 0.0])

    network = Network()
    network.add_node(fixed)
    network.add_node(free)
    network.add_element(
        Element(
            fixed,
            free,
            material=make_material("mode_test"),
            rest_length=1.0,
        )
    )

    plotter = NetworkPlotter(network)

    with pytest.raises(ValueError, match="unsupported plot mode"):
        plotter.plot(mode="damage")
