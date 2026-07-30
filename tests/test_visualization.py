from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import to_hex

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node
from visualization import NetworkPlotter
from visualization.styles import (
    CURRENT_ELEMENT_STYLE,
    REFERENCE_ELEMENT_STYLE,
)


def make_material(name: str) -> Material:
    return Material(
        name=name,
        parameters=MaterialParameters(
            stiffness=10.0,
            failure_threshold=100.0,
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


def test_network_plotter_uses_consistent_element_styles():
    node_a = Node(
        position=[0.0, 0.0],
        fixed=True,
        node_id="A",
    )
    node_b = Node(
        position=[1.0, 0.0],
        node_id="B",
    )
    node_c = Node(
        position=[2.0, 0.0],
        node_id="C",
    )

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
    _, axes = plotter.plot()

    reference_lines = [
        line
        for line in axes.lines
        if line.get_linestyle() == "--"
    ]
    current_lines = [
        line
        for line in axes.lines
        if line.get_linestyle() == "-"
    ]

    expected_reference_color = to_hex(
        REFERENCE_ELEMENT_STYLE["color"]
    )
    expected_current_color = to_hex(
        CURRENT_ELEMENT_STYLE["color"]
    )

    assert len(reference_lines) == 2
    assert len(current_lines) == 2

    assert {
        to_hex(line.get_color())
        for line in reference_lines
    } == {expected_reference_color}

    assert {
        to_hex(line.get_color())
        for line in current_lines
    } == {expected_current_color}

    assert {
        line.get_linestyle()
        for line in reference_lines
    } == {"--"}

    assert {
        line.get_linestyle()
        for line in current_lines
    } == {"-"}