from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from core.element import Element
from core.material import Material, MaterialParameters
from core.network import Network
from core.node import Node
from visualization import NetworkPlotter


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

    material = Material(
        name="visualization_test_material",
        parameters=MaterialParameters(
            stiffness=10.0,
            failure_threshold=100.0,
        ),
        reference_length=1.0,
    )

    element = Element(
        node_a=fixed,
        node_b=free,
        material=material,
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
