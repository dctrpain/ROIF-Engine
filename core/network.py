from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .element import Element
from .node import Node


@dataclass
class NetworkStepStats:
    """
    Diagnostic information for one physical simulation step.
    """

    time: float = 0.0
    dt: float = 0.0

    node_count: int = 0
    element_count: int = 0
    constraint_count: int = 0

    kinetic_energy: float = 0.0
    elastic_energy: float = 0.0
    gravitational_energy: float = 0.0
    dissipated_energy: float = 0.0
    total_energy: float = 0.0

    maximum_force: float = 0.0
    maximum_velocity: float = 0.0
    maximum_constraint_violation: float = 0.0

    solver_iterations: int = 0
    solver_converged: bool = True

    failed_elements: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": float(self.time),
            "dt": float(self.dt),
            "node_count": int(self.node_count),
            "element_count": int(self.element_count),
            "constraint_count": int(self.constraint_count),
            "kinetic_energy": float(self.kinetic_energy),
            "elastic_energy": float(self.elastic_energy),
            "gravitational_energy": float(
                self.gravitational_energy
            ),
            "dissipated_energy": float(
                self.dissipated_energy
            ),
            "total_energy": float(self.total_energy),
            "maximum_force": float(self.maximum_force),
            "maximum_velocity": float(
                self.maximum_velocity
            ),
            "maximum_constraint_violation": float(
                self.maximum_constraint_violation
            ),
            "solver_iterations": int(
                self.solver_iterations
            ),
            "solver_converged": bool(
                self.solver_converged
            ),
            "failed_elements": int(
                self.failed_elements
            ),
        }


@dataclass
class NetworkState:
    """
    Serializable dynamic state of a network.
    """

    time: float
    step_index: int
    nodes: list[dict[str, Any]]
    elements: list[dict[str, Any]]
    statistics: dict[str, Any] = field(
        default_factory=dict
    )


class Network:
    """
    Simulation container for ROIF Engine.

    Responsibilities
    ----------------
    Network manages:

    - nodes;
    - axial elements;
    - constraints;
    - external forces;
    - biological material updates;
    - mechanical force assembly;
    - numerical integration;
    - constraint projection;
    - diagnostics;
    - history and snapshots.

    Important update rule
    ---------------------
    Material biology is updated exactly once per physical step.

    Mechanical force evaluation may be repeated by a solver
    without repeatedly accumulating biological changes.
    """

    def __init__(
        self,
        *,
        gravity: Iterable[float] | np.ndarray | None = None,
        global_damping: float = 0.0,
        solver: Any | None = None,
        record_history: bool = True,
    ) -> None:
        self.nodes: list[Node] = []
        self.elements: list[Element] = []
        self.constraints: list[Any] = []

        self.solver = solver

        self.gravity = self._prepare_gravity(
            gravity
        )

        self.global_damping = float(
            global_damping
        )

        if not np.isfinite(self.global_damping):
            raise ValueError(
                "global_damping must be finite"
            )

        if self.global_damping < 0.0:
            raise ValueError(
                "global_damping cannot be negative"
            )

        self.record_history = bool(
            record_history
        )

        self.time: float = 0.0
        self.step_index: int = 0

        self.last_step_stats = NetworkStepStats()

        self.history: list[NetworkState] = []

        self._node_ids: set[Any] = set()
        self._element_ids: set[Any] = set()

    # =========================================================
    # Construction
    # =========================================================

    @staticmethod
    def _prepare_gravity(
        gravity: Iterable[float] | np.ndarray | None,
    ) -> np.ndarray | None:
        if gravity is None:
            return None

        vector = np.asarray(
            gravity,
            dtype=float,
        )

        if vector.ndim != 1:
            raise ValueError(
                "gravity must be a one-dimensional vector"
            )

        if vector.size == 0:
            raise ValueError(
                "gravity cannot be empty"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "gravity must contain finite values"
            )

        return vector.copy()

    def add_node(
        self,
        node: Node,
    ) -> Node:
        if not isinstance(node, Node):
            raise TypeError(
                "node must be an instance of Node"
            )

        node_id = getattr(
            node,
            "id",
            None,
        )

        if (
            node_id is not None
            and node_id in self._node_ids
        ):
            raise ValueError(
                f"Duplicate node id: {node_id!r}"
            )

        self._validate_node(node)

        self.nodes.append(node)

        if node_id is not None:
            self._node_ids.add(node_id)

        self._validate_network_dimension()

        return node

    def add_element(
        self,
        element: Element,
    ) -> Element:
        if not isinstance(element, Element):
            raise TypeError(
                "element must be an instance of Element"
            )

        for connected_node in element.connected_nodes():
            if connected_node not in self.nodes:
                raise ValueError(
                    "all element-connected nodes must be "
                    "added to the network first"
                )

        element_id = getattr(
            element,
            "id",
            None,
        )

        if (
            element_id is not None
            and element_id in self._element_ids
        ):
            raise ValueError(
                f"Duplicate element id: {element_id!r}"
            )

        element.synchronize_material_length()

        self.elements.append(element)

        if element_id is not None:
            self._element_ids.add(element_id)

        return element

    def add_constraint(
        self,
        constraint: Any,
    ) -> Any:
        required_methods = (
            "evaluate",
            "gradients",
            "project",
        )

        for method_name in required_methods:
            if not callable(
                getattr(
                    constraint,
                    method_name,
                    None,
                )
            ):
                raise TypeError(
                    "constraint must implement "
                    f"{method_name}()"
                )

        self.constraints.append(
            constraint
        )

        return constraint

    def remove_node(
        self,
        node: Node,
        *,
        remove_connected: bool = False,
    ) -> None:
        if node not in self.nodes:
            raise ValueError(
                "node is not part of this network"
            )

        connected = [
            element
            for element in self.elements
            if node in element.connected_nodes()
        ]

        if connected and not remove_connected:
            raise ValueError(
                "node has connected elements; set "
                "remove_connected=True to remove them"
            )

        for element in connected:
            self.remove_element(element)

        self.nodes.remove(node)

        node_id = getattr(
            node,
            "id",
            None,
        )

        if node_id is not None:
            self._node_ids.discard(node_id)

    def remove_element(
        self,
        element: Element,
    ) -> None:
        if element not in self.elements:
            raise ValueError(
                "element is not part of this network"
            )

        self.elements.remove(element)

        element_id = getattr(
            element,
            "id",
            None,
        )

        if element_id is not None:
            self._element_ids.discard(
                element_id
            )

    def remove_constraint(
        self,
        constraint: Any,
    ) -> None:
        if constraint not in self.constraints:
            raise ValueError(
                "constraint is not part of this network"
            )

        self.constraints.remove(
            constraint
        )

    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def _node_vector(
        node: Node,
        attribute: str,
    ) -> np.ndarray:
        vector = np.asarray(
            getattr(node, attribute),
            dtype=float,
        )

        if vector.ndim != 1:
            raise ValueError(
                f"node.{attribute} must be one-dimensional"
            )

        if vector.size == 0:
            raise ValueError(
                f"node.{attribute} cannot be empty"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"node.{attribute} contains non-finite values"
            )

        return vector

    def _validate_node(
        self,
        node: Node,
    ) -> None:
        position = self._node_vector(
            node,
            "position",
        )

        velocity = self._node_vector(
            node,
            "velocity",
        )

        acceleration = self._node_vector(
            node,
            "acceleration",
        )

        force = self._node_vector(
            node,
            "force",
        )

        shapes = {
            position.shape,
            velocity.shape,
            acceleration.shape,
            force.shape,
        }

        if len(shapes) != 1:
            raise ValueError(
                "node position, velocity, acceleration "
                "and force must have matching dimensions"
            )

        mass = float(node.mass)

        if not np.isfinite(mass):
            raise ValueError(
                "node.mass must be finite"
            )

        if mass <= 0.0:
            raise ValueError(
                "node.mass must be positive"
            )

    def _validate_network_dimension(
        self,
    ) -> None:
        if not self.nodes:
            return

        dimensions = {
            self._node_vector(
                node,
                "position",
            ).shape
            for node in self.nodes
        }

        if len(dimensions) != 1:
            raise ValueError(
                "all nodes in a network must have "
                "the same dimension"
            )

        if self.gravity is not None:
            node_dimension = next(
                iter(dimensions)
            )

            if self.gravity.shape != node_dimension:
                raise ValueError(
                    "gravity dimension must match "
                    "node dimension"
                )

    def validate(self) -> None:
        self._validate_network_dimension()

        for node in self.nodes:
            self._validate_node(node)

        for element in self.elements:
            for connected_node in element.connected_nodes():
                if connected_node not in self.nodes:
                    raise ValueError(
                        "an element references a connected "
                        "node outside the network"
                    )

    # =========================================================
    # Force control
    # =========================================================

    def clear_forces(self) -> None:
        """
        Reset accumulated nodal forces.

        The method supports Node implementations with either
        clear_force(), reset_force(), or direct force storage.
        """

        for node in self.nodes:
            clear_method = getattr(
                node,
                "clear_force",
                None,
            )

            if callable(clear_method):
                clear_method()
                continue

            reset_method = getattr(
                node,
                "reset_force",
                None,
            )

            if callable(reset_method):
                reset_method()
                continue

            node.force = np.zeros_like(
                self._node_vector(
                    node,
                    "force",
                )
            )

    def apply_external_force(
        self,
        node: Node,
        force: Iterable[float] | np.ndarray,
    ) -> None:
        if node not in self.nodes:
            raise ValueError(
                "node is not part of this network"
            )

        force_vector = np.asarray(
            force,
            dtype=float,
        )

        if force_vector.shape != (
            self._node_vector(
                node,
                "position",
            ).shape
        ):
            raise ValueError(
                "external force dimension must match "
                "node dimension"
            )

        if not np.all(np.isfinite(force_vector)):
            raise ValueError(
                "external force must contain finite values"
            )

        node.apply_force(
            force_vector
        )

    def apply_gravity(self) -> None:
        if self.gravity is None:
            return

        for node in self.nodes:
            if bool(
                getattr(node, "fixed", False)
            ):
                continue

            gravitational_force = (
                float(node.mass)
                * self.gravity
            )

            node.apply_force(
                gravitational_force
            )

    def apply_global_damping(self) -> None:
        if self.global_damping <= 0.0:
            return

        for node in self.nodes:
            if bool(
                getattr(node, "fixed", False)
            ):
                continue

            velocity = self._node_vector(
                node,
                "velocity",
            )

            damping_force = (
                -self.global_damping
                * velocity
            )

            node.apply_force(
                damping_force
            )

    def assemble_element_forces(
        self,
        *,
        include_active: bool = True,
    ) -> None:
        """
        Evaluate mechanical element forces without changing
        biological states.
        """

        for element in self.elements:
            element.apply_forces(
                update_material=False,
                include_active=include_active,
            )

    def assemble_forces(
        self,
        *,
        include_active: bool = True,
        external_forces: dict[Node, Any] | None = None,
    ) -> None:
        """
        Clear and rebuild all forces acting on the nodes.
        """

        self.clear_forces()
        self.apply_gravity()
        self.apply_global_damping()

        if external_forces:
            for node, force in external_forces.items():
                self.apply_external_force(
                    node,
                    force,
                )

        self.assemble_element_forces(
            include_active=include_active
        )

    # =========================================================
    # Biological material update
    # =========================================================

    def update_materials(
        self,
        dt: float,
        *,
        include_active: bool = True,
    ) -> None:
        """
        Update every material exactly once per physical step.

        Important
        ---------
        This method changes biological states but does not apply
        the calculated forces to nodes.
        """

        dt = self._validate_dt(dt)

        for element in self.elements:
            if not element.enabled:
                continue

            element.update_material(
                dt=dt,
            )

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

    def integrate_velocities(
        self,
        dt: float,
    ) -> None:
        """
        Update acceleration and velocity.

        Semi-implicit Euler stage 1:

            a = F / m
            v(t + dt) = v(t) + a * dt
        """

        dt = self._validate_dt(dt)

        for node in self.nodes:
            if bool(
                getattr(node, "fixed", False)
            ):
                node.acceleration = np.zeros_like(
                    self._node_vector(
                        node,
                        "acceleration",
                    )
                )

                node.velocity = np.zeros_like(
                    self._node_vector(
                        node,
                        "velocity",
                    )
                )

                continue

            force = self._node_vector(
                node,
                "force",
            )

            acceleration = (
                force / float(node.mass)
            )

            velocity = (
                self._node_vector(
                    node,
                    "velocity",
                )
                + acceleration * dt
            )

            node.acceleration = acceleration
            node.velocity = velocity

    def integrate_positions(
        self,
        dt: float,
    ) -> None:
        """
        Semi-implicit Euler stage 2:

            x(t + dt) = x(t) + v(t + dt) * dt
        """

        dt = self._validate_dt(dt)

        for node in self.nodes:
            if bool(
                getattr(node, "fixed", False)
            ):
                continue

            node.position = (
                self._node_vector(
                    node,
                    "position",
                )
                + self._node_vector(
                    node,
                    "velocity",
                )
                * dt
            )

    def integrate(
        self,
        dt: float,
    ) -> None:
        self.integrate_velocities(dt)
        self.integrate_positions(dt)

    # =========================================================
    # Constraint solver
    # =========================================================

    def solve_constraints(
        self,
        dt: float,
    ) -> dict[str, Any]:
        """
        Project configured constraints.

        Preferred route:
            use self.solver.solve(...)

        Fallback route:
            call constraint.project(...) directly.
        """

        dt = self._validate_dt(dt)

        if not self.constraints:
            return {
                "iterations": 0,
                "converged": True,
                "maximum_violation": 0.0,
            }

        if self.solver is not None:
            solve_method = getattr(
                self.solver,
                "solve",
                None,
            )

            if not callable(solve_method):
                raise TypeError(
                    "solver must implement solve()"
                )

            result = self._call_solver(
                solve_method,
                dt,
            )

            return self._normalize_solver_result(
                result
            )

        maximum_violation = 0.0

        for constraint in self.constraints:
            if not bool(
                getattr(
                    constraint,
                    "enabled",
                    True,
                )
            ):
                continue

            begin_step = getattr(
                constraint,
                "begin_step",
                None,
            )

            if callable(begin_step):
                begin_step()

            project_method = getattr(
                constraint,
                "project",
            )

            self._call_constraint_project(
                project_method,
                dt,
            )

            violation = abs(
                float(
                    constraint.evaluate()
                )
            )

            maximum_violation = max(
                maximum_violation,
                violation,
            )

            damp_velocity = getattr(
                constraint,
                "damp_velocity",
                None,
            )

            if callable(damp_velocity):
                self._call_constraint_damping(
                    damp_velocity,
                    dt,
                )

            end_step = getattr(
                constraint,
                "end_step",
                None,
            )

            if callable(end_step):
                end_step()

        return {
            "iterations": 1,
            "converged": True,
            "maximum_violation": (
                maximum_violation
            ),
        }

    def _call_solver(
        self,
        solve_method: Any,
        dt: float,
    ) -> Any:
        """
        Support the most likely Solver signatures during
        architecture migration.
        """

        attempts = (
            lambda: solve_method(
                constraints=self.constraints,
                dt=dt,
            ),
            lambda: solve_method(
                self.constraints,
                dt,
            ),
            lambda: solve_method(
                network=self,
                dt=dt,
            ),
            lambda: solve_method(
                self,
                dt,
            ),
        )

        last_error: TypeError | None = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as error:
                last_error = error

        raise TypeError(
            "Unable to call solver.solve() with a supported "
            "signature"
        ) from last_error

    @staticmethod
    def _call_constraint_project(
        project_method: Any,
        dt: float,
    ) -> Any:
        attempts = (
            lambda: project_method(dt=dt),
            lambda: project_method(dt),
            lambda: project_method(),
        )

        last_error: TypeError | None = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as error:
                last_error = error

        raise TypeError(
            "Unable to call constraint.project()"
        ) from last_error

    @staticmethod
    def _call_constraint_damping(
        damp_method: Any,
        dt: float,
    ) -> Any:
        try:
            return damp_method(dt=dt)
        except TypeError:
            try:
                return damp_method(dt)
            except TypeError:
                return damp_method()

    @staticmethod
    def _normalize_solver_result(
        result: Any,
    ) -> dict[str, Any]:
        if result is None:
            return {
                "iterations": 0,
                "converged": True,
                "maximum_violation": 0.0,
            }

        if isinstance(result, dict):
            return {
                "iterations": int(
                    result.get(
                        "iterations",
                        result.get(
                            "iteration_count",
                            0,
                        ),
                    )
                ),
                "converged": bool(
                    result.get(
                        "converged",
                        True,
                    )
                ),
                "maximum_violation": float(
                    result.get(
                        "maximum_violation",
                        result.get(
                            "max_violation",
                            0.0,
                        ),
                    )
                ),
            }

        return {
            "iterations": int(
                getattr(
                    result,
                    "iterations",
                    getattr(
                        result,
                        "iteration_count",
                        0,
                    ),
                )
            ),
            "converged": bool(
                getattr(
                    result,
                    "converged",
                    True,
                )
            ),
            "maximum_violation": float(
                getattr(
                    result,
                    "maximum_violation",
                    getattr(
                        result,
                        "max_violation",
                        0.0,
                    ),
                )
            ),
        }

    # =========================================================
    # Velocity reconstruction
    # =========================================================

    def capture_positions(
        self,
    ) -> dict[int, np.ndarray]:
        return {
            id(node): self._node_vector(
                node,
                "position",
            ).copy()
            for node in self.nodes
        }

    def reconstruct_velocities(
        self,
        previous_positions: dict[int, np.ndarray],
        dt: float,
    ) -> None:
        """
        Reconstruct velocity after positional constraint
        projection.

        This is required because XPBD changes positions directly.
        """

        dt = self._validate_dt(dt)

        for node in self.nodes:
            if bool(
                getattr(node, "fixed", False)
            ):
                node.velocity = np.zeros_like(
                    self._node_vector(
                        node,
                        "velocity",
                    )
                )
                continue

            previous_position = previous_positions[
                id(node)
            ]

            current_position = self._node_vector(
                node,
                "position",
            )

            node.velocity = (
                current_position
                - previous_position
            ) / dt

    # =========================================================
    # Physical step
    # =========================================================

    def step(
        self,
        dt: float,
        *,
        external_forces: dict[Node, Any] | None = None,
        update_materials: bool = True,
        include_active: bool = True,
        solve_constraints: bool = True,
        record: bool | None = None,
    ) -> NetworkStepStats:
        """
        Advance the network by one physical time step.

        Execution order
        ---------------
        1. validate state;
        2. update material biology once;
        3. assemble mechanical forces;
        4. integrate velocities and positions;
        5. project constraints;
        6. reconstruct velocities;
        7. record histories;
        8. compute diagnostics.

        Material forces are evaluated again after biological
        updating, but material states are not advanced again.
        """

        dt = self._validate_dt(dt)
        self.validate()

        if record is None:
            record = self.record_history

        # -----------------------------------------------------
        # 1. Biological evolution — exactly once
        # -----------------------------------------------------

        if update_materials:
            self.update_materials(
                dt=dt,
                include_active=include_active,
            )

        # -----------------------------------------------------
        # 2. Mechanical force assembly
        # -----------------------------------------------------

        self.assemble_forces(
            include_active=include_active,
            external_forces=external_forces,
        )

        # -----------------------------------------------------
        # 3. Semi-implicit Euler integration
        # -----------------------------------------------------

        self.integrate_velocities(dt)

        positions_before_projection = (
            self.capture_positions()
        )

        self.integrate_positions(dt)

        # -----------------------------------------------------
        # 4. Constraint projection
        # -----------------------------------------------------

        solver_result = {
            "iterations": 0,
            "converged": True,
            "maximum_violation": 0.0,
        }

        if solve_constraints:
            solver_result = self.solve_constraints(
                dt
            )

            if self.constraints:
                self.reconstruct_velocities(
                    previous_positions=(
                        positions_before_projection
                    ),
                    dt=dt,
                )

        # -----------------------------------------------------
        # 5. Complete physical time step
        # -----------------------------------------------------

        self.time += dt
        self.step_index += 1

        if record:
            self.record()

        self.last_step_stats = (
            self.compute_step_statistics(
                dt=dt,
                solver_result=solver_result,
            )
        )

        return self.last_step_stats

    # =========================================================
    # Simulation
    # =========================================================

    def simulate(
        self,
        duration: float,
        dt: float,
        *,
        external_forces: (
            dict[Node, Any]
            | callable
            | None
        ) = None,
        update_materials: bool = True,
        include_active: bool = True,
        solve_constraints: bool = True,
        record: bool | None = None,
        maximum_steps: int | None = None,
    ) -> list[NetworkStepStats]:
        """
        Simulate the network for a given duration.

        external_forces may be:

        - None;
        - a static mapping {node: force};
        - a callable receiving (network, time, step_index).
        """

        duration = float(duration)
        dt = self._validate_dt(dt)

        if not np.isfinite(duration):
            raise ValueError(
                "duration must be finite"
            )

        if duration < 0.0:
            raise ValueError(
                "duration cannot be negative"
            )

        if duration == 0.0:
            return []

        required_steps = int(
            np.ceil(duration / dt)
        )

        if maximum_steps is not None:
            maximum_steps = int(
                maximum_steps
            )

            if maximum_steps <= 0:
                raise ValueError(
                    "maximum_steps must be positive"
                )

            required_steps = min(
                required_steps,
                maximum_steps,
            )

        statistics: list[NetworkStepStats] = []

        for _ in range(required_steps):
            remaining_time = (
                duration
                - (
                    self.time
                    - statistics[0].time
                    if statistics
                    else 0.0
                )
            )

            step_dt = min(
                dt,
                max(remaining_time, 0.0),
            )

            if step_dt <= 0.0:
                break

            if callable(external_forces):
                current_external_forces = (
                    external_forces(
                        self,
                        self.time,
                        self.step_index,
                    )
                )
            else:
                current_external_forces = (
                    external_forces
                )

            stats = self.step(
                dt=step_dt,
                external_forces=(
                    current_external_forces
                ),
                update_materials=update_materials,
                include_active=include_active,
                solve_constraints=solve_constraints,
                record=record,
            )

            statistics.append(stats)

        return statistics

    # =========================================================
    # Relaxation
    # =========================================================

    def relax(
        self,
        *,
        dt: float = 0.001,
        maximum_steps: int = 10_000,
        velocity_tolerance: float = 1e-5,
        force_tolerance: float = 1e-5,
        update_materials: bool = False,
        include_active: bool = False,
        solve_constraints: bool = True,
        record: bool = False,
    ) -> dict[str, Any]:
        """
        Advance the system until mechanical equilibrium.

        Biological updating is disabled by default so numerical
        relaxation does not simulate thousands of biological
        time steps.
        """

        dt = self._validate_dt(dt)
        maximum_steps = int(maximum_steps)

        if maximum_steps <= 0:
            raise ValueError(
                "maximum_steps must be positive"
            )

        if velocity_tolerance < 0.0:
            raise ValueError(
                "velocity_tolerance cannot be negative"
            )

        if force_tolerance < 0.0:
            raise ValueError(
                "force_tolerance cannot be negative"
            )

        converged = False
        steps = 0

        for steps in range(
            1,
            maximum_steps + 1,
        ):
            stats = self.step(
                dt=dt,
                update_materials=update_materials,
                include_active=include_active,
                solve_constraints=solve_constraints,
                record=record,
            )

            if (
                stats.maximum_velocity
                <= velocity_tolerance
                and stats.maximum_force
                <= force_tolerance
                and stats.maximum_constraint_violation
                <= force_tolerance
            ):
                converged = True
                break

        return {
            "converged": converged,
            "steps": steps,
            "time": self.time,
            "maximum_velocity": (
                self.last_step_stats.maximum_velocity
            ),
            "maximum_force": (
                self.last_step_stats.maximum_force
            ),
            "maximum_constraint_violation": (
                self.last_step_stats
                .maximum_constraint_violation
            ),
        }

    # =========================================================
    # Diagnostics
    # =========================================================

    def kinetic_energy(self) -> float:
        energy = 0.0

        for node in self.nodes:
            velocity = self._node_vector(
                node,
                "velocity",
            )

            energy += (
                0.5
                * float(node.mass)
                * float(
                    np.dot(
                        velocity,
                        velocity,
                    )
                )
            )

        return float(energy)

    def elastic_energy(self) -> float:
        return float(
            sum(
                element.elastic_energy()
                for element in self.elements
                if element.enabled
            )
        )

    def dissipated_energy(self) -> float:
        return float(
            sum(
                element.dissipated_energy()
                for element in self.elements
            )
        )

    def gravitational_energy(self) -> float:
        """
        Potential energy relative to the coordinate origin.

        U = -m * g·x
        """

        if self.gravity is None:
            return 0.0

        energy = 0.0

        for node in self.nodes:
            position = self._node_vector(
                node,
                "position",
            )

            energy += (
                -float(node.mass)
                * float(
                    np.dot(
                        self.gravity,
                        position,
                    )
                )
            )

        return float(energy)

    def total_energy(self) -> float:
        return float(
            self.kinetic_energy()
            + self.elastic_energy()
            + self.gravitational_energy()
        )

    def maximum_velocity(self) -> float:
        if not self.nodes:
            return 0.0

        return float(
            max(
                np.linalg.norm(
                    self._node_vector(
                        node,
                        "velocity",
                    )
                )
                for node in self.nodes
            )
        )

    def maximum_force(self) -> float:
        if not self.nodes:
            return 0.0

        return float(
            max(
                np.linalg.norm(
                    self._node_vector(
                        node,
                        "force",
                    )
                )
                for node in self.nodes
            )
        )

    def failed_elements(self) -> list[Element]:
        return [
            element
            for element in self.elements
            if element.is_failed()
        ]

    def tensions(self) -> np.ndarray:
        return np.asarray(
            [
                element.tension()
                for element in self.elements
            ],
            dtype=float,
        )

    def compressions(self) -> np.ndarray:
        return np.asarray(
            [
                element.compression()
                for element in self.elements
            ],
            dtype=float,
        )

    def positions(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(
                (0, 0),
                dtype=float,
            )

        return np.vstack(
            [
                self._node_vector(
                    node,
                    "position",
                )
                for node in self.nodes
            ]
        )

    def velocities(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(
                (0, 0),
                dtype=float,
            )

        return np.vstack(
            [
                self._node_vector(
                    node,
                    "velocity",
                )
                for node in self.nodes
            ]
        )

    def forces(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(
                (0, 0),
                dtype=float,
            )

        return np.vstack(
            [
                self._node_vector(
                    node,
                    "force",
                )
                for node in self.nodes
            ]
        )

    def compute_step_statistics(
        self,
        *,
        dt: float,
        solver_result: dict[str, Any],
    ) -> NetworkStepStats:
        kinetic = self.kinetic_energy()
        elastic = self.elastic_energy()
        gravitational = (
            self.gravitational_energy()
        )
        dissipated = self.dissipated_energy()

        return NetworkStepStats(
            time=self.time,
            dt=dt,
            node_count=len(self.nodes),
            element_count=len(self.elements),
            constraint_count=len(
                self.constraints
            ),
            kinetic_energy=kinetic,
            elastic_energy=elastic,
            gravitational_energy=gravitational,
            dissipated_energy=dissipated,
            total_energy=(
                kinetic
                + elastic
                + gravitational
            ),
            maximum_force=self.maximum_force(),
            maximum_velocity=(
                self.maximum_velocity()
            ),
            maximum_constraint_violation=float(
                solver_result.get(
                    "maximum_violation",
                    0.0,
                )
            ),
            solver_iterations=int(
                solver_result.get(
                    "iterations",
                    0,
                )
            ),
            solver_converged=bool(
                solver_result.get(
                    "converged",
                    True,
                )
            ),
            failed_elements=len(
                self.failed_elements()
            ),
        )

    def energies(self) -> dict[str, float]:
        kinetic = self.kinetic_energy()
        elastic = self.elastic_energy()
        gravitational = (
            self.gravitational_energy()
        )

        return {
            "kinetic": kinetic,
            "elastic": elastic,
            "gravitational": gravitational,
            "dissipated": self.dissipated_energy(),
            "total": (
                kinetic
                + elastic
                + gravitational
            ),
        }

    # =========================================================
    # History
    # =========================================================

    def record(self) -> None:
        for node in self.nodes:
            record_method = getattr(
                node,
                "record",
                None,
            )

            if callable(record_method):
                record_method()
            else:
                self._record_node_fallback(
                    node
                )

        for element in self.elements:
            element.record()

        self.history.append(
            self.snapshot()
        )

    def _record_node_fallback(
        self,
        node: Node,
    ) -> None:
        if hasattr(node, "positions"):
            node.positions.append(
                self._node_vector(
                    node,
                    "position",
                ).copy()
            )

        if hasattr(node, "velocities"):
            node.velocities.append(
                self._node_vector(
                    node,
                    "velocity",
                ).copy()
            )

        if hasattr(node, "forces"):
            node.forces.append(
                self._node_vector(
                    node,
                    "force",
                ).copy()
            )

    def clear_history(self) -> None:
        self.history.clear()

        for node in self.nodes:
            clear_method = getattr(
                node,
                "clear_history",
                None,
            )

            if callable(clear_method):
                clear_method()
            else:
                for attribute in (
                    "positions",
                    "velocities",
                    "forces",
                ):
                    history = getattr(
                        node,
                        attribute,
                        None,
                    )

                    if hasattr(
                        history,
                        "clear",
                    ):
                        history.clear()

        for element in self.elements:
            element.clear_history()

    # =========================================================
    # Snapshot and restoration
    # =========================================================

    def _node_snapshot(
        self,
        node: Node,
    ) -> dict[str, Any]:
        snapshot_method = getattr(
            node,
            "snapshot",
            None,
        )

        if callable(snapshot_method):
            return snapshot_method()

        return {
            "id": getattr(
                node,
                "id",
                None,
            ),
            "position": self._node_vector(
                node,
                "position",
            ).copy(),
            "velocity": self._node_vector(
                node,
                "velocity",
            ).copy(),
            "acceleration": self._node_vector(
                node,
                "acceleration",
            ).copy(),
            "force": self._node_vector(
                node,
                "force",
            ).copy(),
            "mass": float(node.mass),
            "fixed": bool(node.fixed),
            "damage": float(
                getattr(
                    node,
                    "damage",
                    0.0,
                )
            ),
            "remodeling": float(
                getattr(
                    node,
                    "remodeling",
                    1.0,
                )
            ),
            "activity": float(
                getattr(
                    node,
                    "activity",
                    0.0,
                )
            ),
            "metabolic_state": float(
                getattr(
                    node,
                    "metabolic_state",
                    1.0,
                )
            ),
            "energy": float(
                getattr(
                    node,
                    "energy",
                    1.0,
                )
            ),
        }

    def snapshot(self) -> NetworkState:
        return NetworkState(
            time=float(self.time),
            step_index=int(self.step_index),
            nodes=[
                self._node_snapshot(node)
                for node in self.nodes
            ],
            elements=[
                element.snapshot()
                for element in self.elements
            ],
            statistics=(
                self.last_step_stats.as_dict()
            ),
        )

    def clone(self) -> Network:
        return deepcopy(self)

    def restore(
        self,
        state: NetworkState | Network,
    ) -> None:
        """
        Restore either a complete cloned Network or a
        NetworkState snapshot.

        A complete clone restores all material internals.
        A lightweight NetworkState restores mechanical node
        values and basic timing information.
        """

        if isinstance(state, Network):
            restored = deepcopy(state)

            self.__dict__.clear()
            self.__dict__.update(
                restored.__dict__
            )

            return

        if not isinstance(state, NetworkState):
            raise TypeError(
                "state must be NetworkState or Network"
            )

        if len(state.nodes) != len(self.nodes):
            raise ValueError(
                "snapshot node count does not match network"
            )

        for node, node_state in zip(
            self.nodes,
            state.nodes,
        ):
            for attribute in (
                "position",
                "velocity",
                "acceleration",
                "force",
            ):
                if attribute in node_state:
                    setattr(
                        node,
                        attribute,
                        np.asarray(
                            node_state[attribute],
                            dtype=float,
                        ).copy(),
                    )

            for attribute in (
                "mass",
                "fixed",
                "damage",
                "remodeling",
                "activity",
                "metabolic_state",
                "energy",
            ):
                if attribute in node_state:
                    setattr(
                        node,
                        attribute,
                        deepcopy(
                            node_state[attribute]
                        ),
                    )

        self.time = float(state.time)
        self.step_index = int(
            state.step_index
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset_time(self) -> None:
        self.time = 0.0
        self.step_index = 0

    def reset_material_states(self) -> None:
        for element in self.elements:
            reset_method = getattr(
                element.material,
                "reset_state",
                None,
            )

            if callable(reset_method):
                reset_method()

    def zero_velocities(self) -> None:
        for node in self.nodes:
            node.velocity = np.zeros_like(
                self._node_vector(
                    node,
                    "velocity",
                )
            )

            node.acceleration = np.zeros_like(
                self._node_vector(
                    node,
                    "acceleration",
                )
            )

    # =========================================================
    # Lookup
    # =========================================================

    def get_node(
        self,
        node_id: Any,
    ) -> Node:
        for node in self.nodes:
            if getattr(
                node,
                "id",
                None,
            ) == node_id:
                return node

        raise KeyError(
            f"Node not found: {node_id!r}"
        )

    def get_element(
        self,
        element_id: Any,
    ) -> Element:
        for element in self.elements:
            if element.id == element_id:
                return element

        raise KeyError(
            f"Element not found: {element_id!r}"
        )

    # =========================================================
    # Representation
    # =========================================================

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"Network("
            f"nodes={len(self.nodes)}, "
            f"elements={len(self.elements)}, "
            f"constraints={len(self.constraints)}, "
            f"time={self.time:.6f}, "
            f"step={self.step_index})"
        )




