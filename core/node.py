from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np


class Node:
    """
    Mechanical and biological state of one network point.

    Mechanical state
    ----------------
    position:
        Current spatial coordinates.

    velocity:
        Current linear velocity.

    acceleration:
        Current linear acceleration.

    force:
        Accumulated force for the current physical step.

    mass:
        Positive nodal mass.

    fixed:
        Fixed nodes have zero inverse mass and are not integrated.

    Biological state
    ----------------
    damage:
        Normalized structural damage in [0, 1].

    remodeling:
        Normalized remodeling state.

    activity:
        Normalized biological or neural activity in [0, 1].

    metabolic_state:
        Normalized metabolic reserve in [0, 1].

    energy:
        Normalized local energy reserve in [0, 1].
    """

    _next_id: int = 0

    def __init__(
        self,
        position: Iterable[float] | np.ndarray,
        *,
        velocity: Iterable[float] | np.ndarray | None = None,
        acceleration: Iterable[float] | np.ndarray | None = None,
        force: Iterable[float] | np.ndarray | None = None,
        mass: float = 1.0,
        fixed: bool = False,
        node_id: int | str | None = None,
        damage: float = 0.0,
        remodeling: float = 1.0,
        activity: float = 0.0,
        metabolic_state: float = 1.0,
        energy: float = 1.0,
        record_history: bool = True,
    ) -> None:
        """
        Create a simulation node.

        Parameters
        ----------
        position:
            Initial coordinates. Supports any spatial dimension.

        velocity, acceleration, force:
            Optional initial vectors. Missing vectors are
            initialized with zeros matching position.

        mass:
            Positive nodal mass. Fixed nodes still require a
            valid positive mass, but their inverse mass is zero.

        fixed:
            Prevent mechanical integration of the node.

        node_id:
            Optional unique identifier.

        record_history:
            Controls whether record() stores state history.
        """

        self.position = self._prepare_vector(
            position,
            name="position",
        )

        dimension = self.position.shape

        self.velocity = self._prepare_optional_vector(
            velocity,
            dimension=dimension,
            name="velocity",
        )

        self.acceleration = self._prepare_optional_vector(
            acceleration,
            dimension=dimension,
            name="acceleration",
        )

        self.force = self._prepare_optional_vector(
            force,
            dimension=dimension,
            name="force",
        )

        self.mass = float(mass)

        if not np.isfinite(self.mass):
            raise ValueError(
                "mass must be finite"
            )

        if self.mass <= 0.0:
            raise ValueError(
                "mass must be positive"
            )

        self.fixed = bool(fixed)

        if node_id is None:
            node_id = Node._next_id
            Node._next_id += 1

        self.id = node_id

        self.damage = float(damage)
        self.remodeling = float(remodeling)
        self.activity = float(activity)
        self.metabolic_state = float(
            metabolic_state
        )
        self.energy = float(energy)

        self.record_history = bool(
            record_history
        )

        self.positions: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.accelerations: list[np.ndarray] = []
        self.forces: list[np.ndarray] = []

        self.damage_history: list[float] = []
        self.remodeling_history: list[float] = []
        self.activity_history: list[float] = []
        self.metabolic_history: list[float] = []
        self.energy_history: list[float] = []

        self._initial_state = {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "acceleration": self.acceleration.copy(),
            "force": self.force.copy(),
            "mass": self.mass,
            "fixed": self.fixed,
            "damage": self.damage,
            "remodeling": self.remodeling,
            "activity": self.activity,
            "metabolic_state": self.metabolic_state,
            "energy": self.energy,
        }

        self.clamp_biological_state()
        self.validate()

        if self.fixed:
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)

    # =========================================================
    # Vector preparation
    # =========================================================

    @staticmethod
    def _prepare_vector(
        value: Iterable[float] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        vector = np.asarray(
            value,
            dtype=float,
        )

        if vector.ndim != 1:
            raise ValueError(
                f"{name} must be a one-dimensional vector"
            )

        if vector.size == 0:
            raise ValueError(
                f"{name} cannot be empty"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} must contain finite values"
            )

        return vector.copy()

    @classmethod
    def _prepare_optional_vector(
        cls,
        value: Iterable[float] | np.ndarray | None,
        *,
        dimension: tuple[int, ...],
        name: str,
    ) -> np.ndarray:
        if value is None:
            return np.zeros(
                dimension,
                dtype=float,
            )

        vector = cls._prepare_vector(
            value,
            name=name,
        )

        if vector.shape != dimension:
            raise ValueError(
                f"{name} dimension must match position"
            )

        return vector

    # =========================================================
    # Validation
    # =========================================================

    def validate(self) -> None:
        vectors = {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "force": self.force,
        }

        expected_shape = self.position.shape

        for name, value in vectors.items():
            vector = np.asarray(
                value,
                dtype=float,
            )

            if vector.ndim != 1:
                raise ValueError(
                    f"{name} must be one-dimensional"
                )

            if vector.shape != expected_shape:
                raise ValueError(
                    f"{name} dimension must match position"
                )

            if not np.all(np.isfinite(vector)):
                raise ValueError(
                    f"{name} must contain finite values"
                )

            setattr(
                self,
                name,
                vector.copy(),
            )

        self.mass = float(self.mass)

        if not np.isfinite(self.mass):
            raise ValueError(
                "mass must be finite"
            )

        if self.mass <= 0.0:
            raise ValueError(
                "mass must be positive"
            )

        self.fixed = bool(self.fixed)

        self.clamp_biological_state()

    def clamp_biological_state(self) -> None:
        biological_values = {
            "damage": self.damage,
            "remodeling": self.remodeling,
            "activity": self.activity,
            "metabolic_state": self.metabolic_state,
            "energy": self.energy,
        }

        for name, value in biological_values.items():
            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

            setattr(
                self,
                name,
                float(
                    np.clip(
                        value,
                        0.0,
                        1.0,
                    )
                ),
            )

    # =========================================================
    # Mechanical properties
    # =========================================================

    @property
    def dimension(self) -> int:
        return int(
            self.position.size
        )

    def inverse_mass(self) -> float:
        """
        Return zero for fixed nodes and 1 / mass otherwise.
        """

        if self.fixed:
            return 0.0

        return float(
            1.0 / self.mass
        )

    @property
    def inv_mass(self) -> float:
        """
        Compatibility alias used by some solvers.
        """

        return self.inverse_mass()

    def momentum(self) -> np.ndarray:
        return (
            self.mass
            * self.velocity
        )

    def kinetic_energy(self) -> float:
        return float(
            0.5
            * self.mass
            * np.dot(
                self.velocity,
                self.velocity,
            )
        )

    def speed(self) -> float:
        return float(
            np.linalg.norm(
                self.velocity
            )
        )

    def force_magnitude(self) -> float:
        return float(
            np.linalg.norm(
                self.force
            )
        )

    # =========================================================
    # Force accumulation
    # =========================================================

    def apply_force(
        self,
        force: Iterable[float] | np.ndarray,
    ) -> None:
        """
        Add force to the current nodal force accumulator.

        Fixed nodes may still accumulate forces so reaction-force
        diagnostics remain available. They are simply not moved
        during integration.
        """

        force_vector = self._prepare_vector(
            force,
            name="force",
        )

        if force_vector.shape != self.position.shape:
            raise ValueError(
                "applied force dimension must match node "
                "position dimension"
            )

        self.force += force_vector

    def add_force(
        self,
        force: Iterable[float] | np.ndarray,
    ) -> None:
        """
        Compatibility alias for apply_force().
        """

        self.apply_force(force)

    def set_force(
        self,
        force: Iterable[float] | np.ndarray,
    ) -> None:
        force_vector = self._prepare_vector(
            force,
            name="force",
        )

        if force_vector.shape != self.position.shape:
            raise ValueError(
                "force dimension must match node position"
            )

        self.force = force_vector

    def clear_force(self) -> None:
        self.force.fill(0.0)

    def reset_force(self) -> None:
        """
        Compatibility alias for clear_force().
        """

        self.clear_force()

    # =========================================================
    # Integration
    # =========================================================

    @staticmethod
    def _validate_dt(
        dt: float,
    ) -> float:
        dt = float(dt)

        if not np.isfinite(dt):
            raise ValueError(
                "dt must be finite"
            )

        if dt <= 0.0:
            raise ValueError(
                "dt must be positive"
            )

        return dt

    def update_acceleration(self) -> np.ndarray:
        """
        Calculate acceleration from accumulated force.

            a = F / m
        """

        if self.fixed:
            self.acceleration.fill(0.0)
        else:
            self.acceleration = (
                self.force / self.mass
            )

        return self.acceleration.copy()

    def integrate_velocity(
        self,
        dt: float,
    ) -> np.ndarray:
        """
        Semi-implicit Euler velocity stage.

            v(t + dt) = v(t) + a(t) * dt
        """

        dt = self._validate_dt(dt)

        if self.fixed:
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            return self.velocity.copy()

        self.update_acceleration()

        self.velocity += (
            self.acceleration * dt
        )

        self._validate_finite_mechanical_state()

        return self.velocity.copy()

    def integrate_position(
        self,
        dt: float,
    ) -> np.ndarray:
        """
        Semi-implicit Euler position stage.

            x(t + dt) = x(t) + v(t + dt) * dt
        """

        dt = self._validate_dt(dt)

        if self.fixed:
            return self.position.copy()

        self.position += (
            self.velocity * dt
        )

        self._validate_finite_mechanical_state()

        return self.position.copy()

    def integrate(
        self,
        dt: float,
    ) -> None:
        self.integrate_velocity(dt)
        self.integrate_position(dt)

    def _validate_finite_mechanical_state(
        self,
    ) -> None:
        for name in (
            "position",
            "velocity",
            "acceleration",
            "force",
        ):
            value = np.asarray(
                getattr(self, name),
                dtype=float,
            )

            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"Node {self.id!r} generated a "
                    f"non-finite {name}"
                )

    # =========================================================
    # Position and velocity control
    # =========================================================

    def set_position(
        self,
        position: Iterable[float] | np.ndarray,
    ) -> None:
        position_vector = self._prepare_vector(
            position,
            name="position",
        )

        if position_vector.shape != self.position.shape:
            raise ValueError(
                "new position dimension must match node "
                "dimension"
            )

        self.position = position_vector

    def translate(
        self,
        displacement: Iterable[float] | np.ndarray,
    ) -> None:
        displacement_vector = self._prepare_vector(
            displacement,
            name="displacement",
        )

        if displacement_vector.shape != self.position.shape:
            raise ValueError(
                "displacement dimension must match node "
                "dimension"
            )

        self.position += displacement_vector

    def set_velocity(
        self,
        velocity: Iterable[float] | np.ndarray,
    ) -> None:
        velocity_vector = self._prepare_vector(
            velocity,
            name="velocity",
        )

        if velocity_vector.shape != self.position.shape:
            raise ValueError(
                "velocity dimension must match node dimension"
            )

        if self.fixed:
            self.velocity.fill(0.0)
        else:
            self.velocity = velocity_vector

    def zero_velocity(self) -> None:
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)

    def set_fixed(
        self,
        fixed: bool,
        *,
        zero_velocity: bool = True,
    ) -> None:
        self.fixed = bool(fixed)

        if self.fixed and zero_velocity:
            self.zero_velocity()

    # =========================================================
    # Biological state
    # =========================================================

    def update_biology(
        self,
        *,
        damage: float | None = None,
        remodeling: float | None = None,
        activity: float | None = None,
        metabolic_state: float | None = None,
        energy: float | None = None,
    ) -> None:
        """
        Update selected biological variables.

        This method does not implement tissue biology. It only
        provides controlled state assignment for future node-level
        biological operators.
        """

        updates = {
            "damage": damage,
            "remodeling": remodeling,
            "activity": activity,
            "metabolic_state": metabolic_state,
            "energy": energy,
        }

        for name, value in updates.items():
            if value is not None:
                setattr(
                    self,
                    name,
                    float(value),
                )

        self.clamp_biological_state()

    def structural_integrity(self) -> float:
        return float(
            1.0 - self.damage
        )

    def available_energy(self) -> float:
        return float(
            np.clip(
                self.energy
                * self.metabolic_state,
                0.0,
                1.0,
            )
        )

    # =========================================================
    # History
    # =========================================================

    def record(self) -> None:
        if not self.record_history:
            return

        self.positions.append(
            self.position.copy()
        )

        self.velocities.append(
            self.velocity.copy()
        )

        self.accelerations.append(
            self.acceleration.copy()
        )

        self.forces.append(
            self.force.copy()
        )

        self.damage_history.append(
            float(self.damage)
        )

        self.remodeling_history.append(
            float(self.remodeling)
        )

        self.activity_history.append(
            float(self.activity)
        )

        self.metabolic_history.append(
            float(self.metabolic_state)
        )

        self.energy_history.append(
            float(self.energy)
        )

    def clear_history(self) -> None:
        self.positions.clear()
        self.velocities.clear()
        self.accelerations.clear()
        self.forces.clear()

        self.damage_history.clear()
        self.remodeling_history.clear()
        self.activity_history.clear()
        self.metabolic_history.clear()
        self.energy_history.clear()

    def history_length(self) -> int:
        return len(
            self.positions
        )

    # =========================================================
    # Snapshot and restoration
    # =========================================================

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "acceleration": self.acceleration.copy(),
            "force": self.force.copy(),
            "mass": float(self.mass),
            "inverse_mass": self.inverse_mass(),
            "fixed": bool(self.fixed),
            "dimension": self.dimension,
            "damage": float(self.damage),
            "remodeling": float(self.remodeling),
            "activity": float(self.activity),
            "metabolic_state": float(
                self.metabolic_state
            ),
            "energy": float(self.energy),
            "structural_integrity": (
                self.structural_integrity()
            ),
            "available_energy": (
                self.available_energy()
            ),
            "kinetic_energy": (
                self.kinetic_energy()
            ),
            "speed": self.speed(),
            "force_magnitude": (
                self.force_magnitude()
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        """
        Alias suitable for serialization and state transfer.
        """

        return self.snapshot()

    def restore(
        self,
        state: dict[str, Any],
    ) -> None:
        required = (
            "position",
            "velocity",
            "acceleration",
            "force",
        )

        for name in required:
            if name not in state:
                raise KeyError(
                    f"Missing node state field: {name}"
                )

        position = self._prepare_vector(
            state["position"],
            name="position",
        )

        if position.shape != self.position.shape:
            raise ValueError(
                "restored node dimension does not match"
            )

        self.position = position

        self.velocity = self._prepare_optional_vector(
            state["velocity"],
            dimension=position.shape,
            name="velocity",
        )

        self.acceleration = (
            self._prepare_optional_vector(
                state["acceleration"],
                dimension=position.shape,
                name="acceleration",
            )
        )

        self.force = self._prepare_optional_vector(
            state["force"],
            dimension=position.shape,
            name="force",
        )

        if "mass" in state:
            self.mass = float(
                state["mass"]
            )

        if "fixed" in state:
            self.fixed = bool(
                state["fixed"]
            )

        for name in (
            "damage",
            "remodeling",
            "activity",
            "metabolic_state",
            "energy",
        ):
            if name in state:
                setattr(
                    self,
                    name,
                    float(state[name]),
                )

        self.validate()

        if self.fixed:
            self.zero_velocity()

    def reset(
        self,
        *,
        clear_history: bool = False,
    ) -> None:
        """
        Restore the state captured at node construction.
        """

        self.position = self._initial_state[
            "position"
        ].copy()

        self.velocity = self._initial_state[
            "velocity"
        ].copy()

        self.acceleration = self._initial_state[
            "acceleration"
        ].copy()

        self.force = self._initial_state[
            "force"
        ].copy()

        self.mass = float(
            self._initial_state["mass"]
        )

        self.fixed = bool(
            self._initial_state["fixed"]
        )

        self.damage = float(
            self._initial_state["damage"]
        )

        self.remodeling = float(
            self._initial_state["remodeling"]
        )

        self.activity = float(
            self._initial_state["activity"]
        )

        self.metabolic_state = float(
            self._initial_state[
                "metabolic_state"
            ]
        )

        self.energy = float(
            self._initial_state["energy"]
        )

        self.validate()

        if clear_history:
            self.clear_history()

    def clone(self) -> Node:
        return deepcopy(self)

    # =========================================================
    # Representation
    # =========================================================

    def __repr__(self) -> str:
        position = np.array2string(
            self.position,
            precision=4,
            suppress_small=True,
        )

        velocity = np.array2string(
            self.velocity,
            precision=4,
            suppress_small=True,
        )

        return (
            f"Node("
            f"id={self.id!r}, "
            f"position={position}, "
            f"velocity={velocity}, "
            f"mass={self.mass:.4f}, "
            f"fixed={self.fixed}, "
            f"damage={self.damage:.3f}, "
            f"energy={self.energy:.3f})"
        )