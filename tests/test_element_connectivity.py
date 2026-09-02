import numpy as np
import pytest

from core.element import Element
from core.network import Network
from core.node import Node


class ThreeNodeElement(Element):
    """
    Test-only element used to verify the generic
    Element.connected_nodes() topology contract.
    """

    def __init__(
        self,
        node_a: Node,
        node_b: Node,
        node_c: Node,
    ) -> None:
        self.node_c = node_c
        super().__init__(
            node_a,
            node_b,
        )

    def connected_nodes(self) -> tuple[Node, ...]:
        return (
            self.node_a,
            self.node_b,
            self.node_c,
        )


def make_node(
    x: float,
    y: float,
    z: float,
) -> Node:
    return Node(
        position=np.array(
            [x, y, z],
            dtype=float,
        ),
        mass=1.0,
    )


def test_base_element_reports_two_connected_nodes() -> None:
    node_a = make_node(0.0, 0.0, 0.0)
    node_b = make_node(1.0, 0.0, 0.0)

    element = Element(
        node_a,
        node_b,
    )

    assert element.connected_nodes() == (
        node_a,
        node_b,
    )


def test_network_respects_general_connected_nodes_contract() -> None:
    node_a = make_node(0.0, 0.0, 0.0)
    node_b = make_node(1.0, 0.0, 0.0)
    node_c = make_node(0.0, 1.0, 0.0)

    network = Network()
    network.add_node(node_a)
    network.add_node(node_b)

    element = ThreeNodeElement(
        node_a,
        node_b,
        node_c,
    )

    with pytest.raises(ValueError):
        network.add_element(element)

    network.add_node(node_c)
    network.add_element(element)

    assert element in network.elements

    network.validate()

    with pytest.raises(ValueError):
        network.remove_node(node_c)

    network.remove_node(
        node_c,
        remove_connected=True,
    )

    assert node_c not in network.nodes
    assert element not in network.elements
