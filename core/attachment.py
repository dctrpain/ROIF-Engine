from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .node import Node


class Attachment:
    """
    Body-relative mechanical attachment represented as an affine
    combination of existing ROIF nodes.

    The attachment has no independent mass, integration state,
    acceleration, or force accumulator.

    Its world position and velocity are derived from parent nodes:

        p = sum_i w_i * x_i
        v = sum_i w_i * v_i

    Applied attachment force is distributed back to the same nodes:

        F_i = w_i * F

    For affine attachment coordinates the weights must sum to one.
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        weights: Iterable[float],
        *,
        attachment_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.nodes = tuple(nodes)
        self.weights = np.asarray(
            tuple(weights),
            dtype=float,
        )

        self.id = attachment_id
        self.name = name

        self.validate()

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError(
                "attachment must reference at least one node"
            )

        for node in self.nodes:
            if not isinstance(node, Node):
                raise TypeError(
                    "all attachment nodes must be Node instances"
                )

        if self.weights.ndim != 1:
            raise ValueError(
                "attachment weights must be one-dimensional"
            )

        if len(self.nodes) != self.weights.size:
            raise ValueError(
                "attachment node and weight counts must match"
            )

        if not np.all(np.isfinite(self.weights)):
            raise ValueError(
                "attachment weights must be finite"
            )

        dimension = self.nodes[0].dimension

        for node in self.nodes[1:]:
            if node.dimension != dimension:
                raise ValueError(
                    "all attachment nodes must have the "
                    "same dimension"
                )

        weight_sum = float(
            np.sum(
                self.weights
            )
        )

        if not np.isclose(
            weight_sum,
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "affine attachment weights must sum to 1"
            )

    def connected_nodes(
        self,
    ) -> tuple[Node, ...]:
        return self.nodes

    @property
    def dimension(self) -> int:
        return self.nodes[0].dimension

    def position(
        self,
    ) -> np.ndarray:
        result = np.zeros(
            self.dimension,
            dtype=float,
        )

        for node, weight in zip(
            self.nodes,
            self.weights,
            strict=True,
        ):
            result += float(weight) * node.position

        return result

    def velocity(
        self,
    ) -> np.ndarray:
        result = np.zeros(
            self.dimension,
            dtype=float,
        )

        for node, weight in zip(
            self.nodes,
            self.weights,
            strict=True,
        ):
            result += float(weight) * node.velocity

        return result

    def apply_force(
        self,
        force: Iterable[float] | np.ndarray,
    ) -> None:
        force_vector = np.asarray(
            force,
            dtype=float,
        )

        if force_vector.ndim != 1:
            raise ValueError(
                "attachment force must be one-dimensional"
            )

        if force_vector.shape != (
            self.dimension,
        ):
            raise ValueError(
                "attachment force dimension must match "
                "attachment dimension"
            )

        if not np.all(np.isfinite(force_vector)):
            raise ValueError(
                "attachment force must be finite"
            )

        for node, weight in zip(
            self.nodes,
            self.weights,
            strict=True,
        ):
            node.apply_force(
                float(weight)
                * force_vector
            )

    def snapshot(
        self,
    ) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position(),
            "velocity": self.velocity(),
            "weights": self.weights.copy(),
            "node_ids": tuple(
                node.id
                for node in self.nodes
            ),
            "dimension": self.dimension,
        }

    def __repr__(self) -> str:
        return (
            "Attachment("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"nodes={len(self.nodes)}, "
            f"weights={self.weights.tolist()}"
            ")"
        )
