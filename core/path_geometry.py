from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .attachment import Attachment
from .node import Node


class PathGeometry:
    """
    Ordered multipoint mechanical path through body-relative
    attachments.

    This class owns geometry and kinematics only.

    It has no material state, force law, mass, integration,
    activation, or tissue-specific behavior.
    """

    epsilon = 1e-12

    def __init__(
        self,
        attachments: Sequence[Attachment],
    ) -> None:
        self.attachments = tuple(attachments)
        self.validate()

    def validate(self) -> None:
        if len(self.attachments) < 2:
            raise ValueError(
                "path geometry requires at least two attachments"
            )

        for attachment in self.attachments:
            if not isinstance(attachment, Attachment):
                raise TypeError(
                    "all path points must be Attachment instances"
                )

        dimension = self.attachments[0].dimension

        for attachment in self.attachments[1:]:
            if attachment.dimension != dimension:
                raise ValueError(
                    "all path attachments must have the "
                    "same dimension"
                )

    @property
    def dimension(self) -> int:
        return self.attachments[0].dimension

    def positions(
        self,
    ) -> tuple[np.ndarray, ...]:
        return tuple(
            attachment.position()
            for attachment in self.attachments
        )

    def velocities(
        self,
    ) -> tuple[np.ndarray, ...]:
        return tuple(
            attachment.velocity()
            for attachment in self.attachments
        )

    def segment_vectors(
        self,
    ) -> tuple[np.ndarray, ...]:
        positions = self.positions()

        return tuple(
            positions[index + 1] - positions[index]
            for index in range(len(positions) - 1)
        )

    def segment_lengths(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            float(np.linalg.norm(vector))
            for vector in self.segment_vectors()
        )

    def segment_directions(
        self,
    ) -> tuple[np.ndarray, ...]:
        directions: list[np.ndarray] = []

        for vector in self.segment_vectors():
            length = float(
                np.linalg.norm(vector)
            )

            if length <= self.epsilon:
                directions.append(
                    np.zeros(
                        self.dimension,
                        dtype=float,
                    )
                )
            else:
                directions.append(
                    vector / length
                )

        return tuple(directions)

    def current_length(
        self,
    ) -> float:
        return float(
            sum(
                self.segment_lengths()
            )
        )

    def length_velocity(
        self,
    ) -> float:
        velocities = self.velocities()
        directions = self.segment_directions()

        total = 0.0

        for index, direction in enumerate(directions):
            relative_velocity = (
                velocities[index + 1]
                - velocities[index]
            )

            total += float(
                np.dot(
                    relative_velocity,
                    direction,
                )
            )

        return float(total)

    def attachment_forces(
        self,
        tension: float,
    ) -> tuple[np.ndarray, ...]:
        """
        Return forces applied at path attachments by one scalar
        tension acting along the complete multipoint path.

        Positive tension pulls each path segment toward its
        neighboring attachment.
        """

        tension = float(tension)

        if not np.isfinite(tension):
            raise ValueError(
                "path tension must be finite"
            )

        directions = self.segment_directions()

        forces = [
            np.zeros(
                self.dimension,
                dtype=float,
            )
            for _ in self.attachments
        ]

        forces[0] = (
            tension
            * directions[0]
        )

        for index in range(
            1,
            len(self.attachments) - 1,
        ):
            forces[index] = (
                tension
                * (
                    directions[index]
                    - directions[index - 1]
                )
            )

        forces[-1] = (
            -tension
            * directions[-1]
        )

        return tuple(forces)

    def apply_tension(
        self,
        tension: float,
    ) -> tuple[np.ndarray, ...]:
        """
        Apply one scalar path tension through all attachments.

        Forces are distributed from each attachment to its
        connected ROIF nodes by Attachment.apply_force().
        """

        forces = self.attachment_forces(
            tension
        )

        for attachment, force in zip(
            self.attachments,
            forces,
            strict=True,
        ):
            attachment.apply_force(
                force
            )

        return forces

    def connected_nodes(
        self,
    ) -> tuple[Node, ...]:
        nodes: list[Node] = []
        seen: set[int] = set()

        for attachment in self.attachments:
            for node in attachment.connected_nodes():
                key = id(node)

                if key in seen:
                    continue

                seen.add(key)
                nodes.append(node)

        return tuple(nodes)

    def snapshot(
        self,
    ) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "attachment_count": len(self.attachments),
            "segment_count": len(self.attachments) - 1,
            "current_length": self.current_length(),
            "length_velocity": self.length_velocity(),
            "segment_lengths": self.segment_lengths(),
        }

    def __repr__(self) -> str:
        return (
            "PathGeometry("
            f"attachments={len(self.attachments)}, "
            f"segments={len(self.attachments) - 1}, "
            f"length={self.current_length():.6f}"
            ")"
        )

