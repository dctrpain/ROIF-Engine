from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from roif.development.body_interface import (
    BodyGroundTruth,
    BodyInterface,
)
from roif.development.observation import (
    Observation,
    ObservationChannel,
    ObservationProvenance,
)
from roif.worlds.world_interface import (
    WorldInterface,
    WorldState,
)


@dataclass(frozen=True, slots=True)
class TensegrityMember:
    """
    One structural member of the 3D tensegrity body.

    axial_force > 0  -> tension
    axial_force < 0  -> compression
    """

    name: str
    node_i: int
    node_j: int
    kind: str
    stiffness: float
    rest_length: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TensegrityMember.name must be non-empty.")

        if self.kind not in {"cable", "strut"}:
            raise ValueError(
                "TensegrityMember.kind must be 'cable' or 'strut'."
            )

        if self.node_i == self.node_j:
            raise ValueError("A member must connect two different nodes.")

        if self.stiffness <= 0.0:
            raise ValueError("Member stiffness must be > 0.")

        if self.rest_length <= 0.0:
            raise ValueError("Member rest_length must be > 0.")


class _TensegrityBody3D(BodyInterface):
    """
    Proprioceptive interface of the tensegrity body.

    Important:
        world-space x/y/z coordinates remain simulator ground truth.
        ROIF receives only body-accessible structural channels.
    """

    def __init__(self, world: "Tensegrity3DWorld") -> None:
        self._world = world
        self._sequence_index = 0

    @property
    def body_id(self) -> str:
        return "tensegrity_body_0"

    def observe(self) -> Observation:
        channels: dict[str, ObservationChannel] = {}

        lengths = self._world.member_lengths()
        axial_forces = self._world.member_axial_forces()

        for index, member in enumerate(self._world.members):
            channels[f"member_{index}_length"] = ObservationChannel(
                name=f"member_{index}_length",
                value=float(lengths[index]),
                unit="m",
            )

            channels[f"member_{index}_axial_force"] = ObservationChannel(
                name=f"member_{index}_axial_force",
                value=float(axial_forces[index]),
                unit="N",
            )

        accelerations = self._world.last_accelerations

        for node_index in range(self._world.node_count):
            accel_norm = float(
                np.linalg.norm(accelerations[node_index])
            )

            channels[f"node_{node_index}_accel_norm"] = (
                ObservationChannel(
                    name=f"node_{node_index}_accel_norm",
                    value=accel_norm,
                    unit="m/s^2",
                )
            )

        observation = Observation(
            timestamp=self._world.time,
            channels=channels,
            provenance=ObservationProvenance(
                source_id=self.body_id,
                source_type="proprioceptive_tensegrity_body",
                sequence_index=self._sequence_index,
            ),
            metadata={
                "body_family": "tensegrity",
            },
        )

        self._sequence_index += 1
        return observation

    def ground_truth(self) -> BodyGroundTruth:
        values: dict[str, float] = {}

        for node_index in range(self._world.node_count):
            position = self._world.positions[node_index]
            velocity = self._world.velocities[node_index]

            values[f"node_{node_index}_world_x"] = float(position[0])
            values[f"node_{node_index}_world_y"] = float(position[1])
            values[f"node_{node_index}_world_z"] = float(position[2])

            values[f"node_{node_index}_velocity_x"] = float(velocity[0])
            values[f"node_{node_index}_velocity_y"] = float(velocity[1])
            values[f"node_{node_index}_velocity_z"] = float(velocity[2])

        lengths = self._world.member_lengths()
        axial_forces = self._world.member_axial_forces()

        for index, member in enumerate(self._world.members):
            values[f"member_{index}_length"] = float(lengths[index])
            values[f"member_{index}_axial_force"] = float(
                axial_forces[index]
            )
            values[f"member_{index}_rest_length"] = float(
                member.rest_length
            )

        return BodyGroundTruth(
            timestamp=self._world.time,
            values=values,
            metadata={
                "body": self.body_id,
                "world": self._world.world_id,
            },
        )


class Tensegrity3DWorld(WorldInterface):
    """
    Minimal deterministic 3D developmental world.

    The body is a classical three-strut tensegrity prism:

        - 6 point masses,
        - 3 compression struts,
        - 9 tensile cables,
        - non-zero self-stress,
        - zero external gravity in RDA-0,
        - no reward,
        - no goals,
        - no semantic event labels.

    Initial geometry is constructed directly in a self-stressed equilibrium.
    """

    def __init__(
        self,
        *,
        radius: float = 1.0,
        height: float = 1.2,
        node_mass: float = 1.0,
        cable_stiffness: float = 100.0,
        strut_stiffness: float = 200.0,
        prestress_force_density: float = 5.0,
        damping: float = 2.0,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be > 0.")

        if height <= 0.0:
            raise ValueError("height must be > 0.")

        if node_mass <= 0.0:
            raise ValueError("node_mass must be > 0.")

        if cable_stiffness <= 0.0:
            raise ValueError("cable_stiffness must be > 0.")

        if strut_stiffness <= 0.0:
            raise ValueError("strut_stiffness must be > 0.")

        if prestress_force_density <= 0.0:
            raise ValueError(
                "prestress_force_density must be > 0."
            )

        if damping < 0.0:
            raise ValueError("damping must be >= 0.")

        self._time = 0.0
        self._node_mass = float(node_mass)
        self._damping = float(damping)

        self.positions = self._initial_positions(
            radius=radius,
            height=height,
        )

        self.velocities = np.zeros_like(self.positions)
        self.last_accelerations = np.zeros_like(self.positions)

        topology = self._member_topology()

        self_stress = self._compute_self_stress(
            positions=self.positions,
            topology=topology,
        )

        self_stress *= (
            prestress_force_density
            / np.max(np.abs(self_stress))
        )

        members: list[TensegrityMember] = []

        for index, (
            node_i,
            node_j,
            kind,
        ) in enumerate(topology):
            length = float(
                np.linalg.norm(
                    self.positions[node_j]
                    - self.positions[node_i]
                )
            )

            stiffness = (
                cable_stiffness
                if kind == "cable"
                else strut_stiffness
            )

            target_axial_force = (
                self_stress[index] * length
            )

            rest_length = (
                length
                - target_axial_force / stiffness
            )

            members.append(
                TensegrityMember(
                    name=f"{kind}_{index}",
                    node_i=node_i,
                    node_j=node_j,
                    kind=kind,
                    stiffness=float(stiffness),
                    rest_length=float(rest_length),
                )
            )

        self.members = tuple(members)
        self._body = _TensegrityBody3D(self)

        initial_forces = self.nodal_internal_forces()

        if np.max(np.linalg.norm(initial_forces, axis=1)) > 1e-8:
            raise RuntimeError(
                "Initial tensegrity configuration is not in "
                "self-stressed equilibrium."
            )

    @property
    def world_id(self) -> str:
        return "rda_tensegrity_3d_world_v1"

    @property
    def time(self) -> float:
        return self._time

    @property
    def node_count(self) -> int:
        return int(self.positions.shape[0])

    @staticmethod
    def _initial_positions(
        *,
        radius: float,
        height: float,
    ) -> np.ndarray:
        """
        Build a three-strut tensegrity prism.

        Bottom triangle:
            z = -height / 2

        Top triangle:
            z = +height / 2
            rotated by 30 degrees.

        The 30-degree twist admits the required self-stress state for
        the topology used below.
        """

        points: list[list[float]] = []

        for z, phase in (
            (-height / 2.0, 0.0),
            (+height / 2.0, math.pi / 6.0),
        ):
            for index in range(3):
                angle = (
                    phase
                    + 2.0 * math.pi * index / 3.0
                )

                points.append(
                    [
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                        z,
                    ]
                )

        return np.asarray(points, dtype=float)

    @staticmethod
    def _member_topology() -> tuple[
        tuple[int, int, str],
        ...
    ]:
        members: list[tuple[int, int, str]] = []

        # Three crossing compression struts.
        for index in range(3):
            members.append(
                (
                    index,
                    3 + ((index + 1) % 3),
                    "strut",
                )
            )

        # Bottom tensile triangle.
        for index in range(3):
            members.append(
                (
                    index,
                    (index + 1) % 3,
                    "cable",
                )
            )

        # Top tensile triangle.
        for index in range(3):
            members.append(
                (
                    3 + index,
                    3 + ((index + 1) % 3),
                    "cable",
                )
            )

        # Three side tensile cables.
        for index in range(3):
            members.append(
                (
                    index,
                    3 + index,
                    "cable",
                )
            )

        return tuple(members)

    @staticmethod
    def _compute_self_stress(
        *,
        positions: np.ndarray,
        topology: tuple[tuple[int, int, str], ...],
    ) -> np.ndarray:
        """
        Solve the equilibrium matrix null-space:

            A q = 0

        q is member force density.

        Positive q:
            tension

        Negative q:
            compression
        """

        node_count = positions.shape[0]
        member_count = len(topology)

        equilibrium = np.zeros(
            (3 * node_count, member_count),
            dtype=float,
        )

        for member_index, (
            node_i,
            node_j,
            _kind,
        ) in enumerate(topology):
            delta = (
                positions[node_j]
                - positions[node_i]
            )

            equilibrium[
                3 * node_i : 3 * node_i + 3,
                member_index,
            ] += delta

            equilibrium[
                3 * node_j : 3 * node_j + 3,
                member_index,
            ] -= delta

        _u, singular_values, vh = np.linalg.svd(
            equilibrium,
            full_matrices=True,
        )

        self_stress = vh[-1].copy()

        # Orient the null-space vector so that cables carry tension.
        cable_values = self_stress[3:]

        if float(np.mean(cable_values)) < 0.0:
            self_stress *= -1.0

        residual = float(
            np.linalg.norm(
                equilibrium @ self_stress
            )
        )

        if residual > 1e-10:
            raise RuntimeError(
                "Failed to construct tensegrity self-stress state."
            )

        if np.any(self_stress[:3] >= 0.0):
            raise RuntimeError(
                "Expected compression in all three struts."
            )

        if np.any(self_stress[3:] <= 0.0):
            raise RuntimeError(
                "Expected tension in all cables."
            )

        if singular_values[-1] > 1e-10:
            raise RuntimeError(
                "Tensegrity topology does not possess the expected "
                "self-stress null mode."
            )

        return self_stress

    def member_lengths(self) -> np.ndarray:
        lengths = np.zeros(len(self.members), dtype=float)

        for index, member in enumerate(self.members):
            lengths[index] = np.linalg.norm(
                self.positions[member.node_j]
                - self.positions[member.node_i]
            )

        return lengths

    def member_axial_forces(self) -> np.ndarray:
        lengths = self.member_lengths()

        forces = np.zeros(len(self.members), dtype=float)

        for index, member in enumerate(self.members):
            axial_force = (
                member.stiffness
                * (
                    lengths[index]
                    - member.rest_length
                )
            )

            # Cables cannot carry compression.
            if member.kind == "cable":
                axial_force = max(
                    0.0,
                    axial_force,
                )

            forces[index] = axial_force

        return forces

    def nodal_internal_forces(self) -> np.ndarray:
        forces = np.zeros_like(self.positions)
        axial_forces = self.member_axial_forces()

        for index, member in enumerate(self.members):
            delta = (
                self.positions[member.node_j]
                - self.positions[member.node_i]
            )

            length = float(np.linalg.norm(delta))

            if length <= 0.0:
                raise RuntimeError(
                    "Tensegrity member collapsed to zero length."
                )

            direction = delta / length

            member_force = (
                axial_forces[index] * direction
            )

            forces[member.node_i] += member_force
            forces[member.node_j] -= member_force

        return forces

    def step(self, dt: float) -> None:
        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be numeric.")

        dt = float(dt)

        if not math.isfinite(dt):
            raise ValueError("dt must be finite.")

        if dt <= 0.0:
            raise ValueError("dt must be > 0.")

        internal_forces = self.nodal_internal_forces()

        damping_forces = (
            -self._damping * self.velocities
        )

        total_forces = (
            internal_forces
            + damping_forces
        )

        accelerations = (
            total_forces
            / self._node_mass
        )

        # Semi-implicit Euler.
        self.velocities = (
            self.velocities
            + accelerations * dt
        )

        self.positions = (
            self.positions
            + self.velocities * dt
        )

        self.last_accelerations = accelerations.copy()
        self._time += dt

    def world_state(self) -> WorldState:
        values: dict[str, float] = {}

        for node_index in range(self.node_count):
            position = self.positions[node_index]

            values[f"node_{node_index}_x"] = float(position[0])
            values[f"node_{node_index}_y"] = float(position[1])
            values[f"node_{node_index}_z"] = float(position[2])

        forces = self.member_axial_forces()

        for index, force in enumerate(forces):
            values[f"member_{index}_axial_force"] = float(force)

        return WorldState(
            timestamp=self.time,
            values=values,
            metadata={
                "world": self.world_id,
                "dimension": 3,
                "ground_truth_only": True,
            },
        )

    def bodies(self) -> tuple[BodyInterface, ...]:
        return (self._body,)