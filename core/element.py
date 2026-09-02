from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .material import Material, MaterialParameters


@dataclass
class ElementForce:
    """
    Diagnostic decomposition of the current axial force.

    Sign convention
    ---------------
    positive:
        tension;

    negative:
        compression.
    """

    elastic: float = 0.0
    damping: float = 0.0
    pretension: float = 0.0
    active: float = 0.0
    total: float = 0.0

    def validate(self) -> None:
        for name in (
            "elastic",
            "damping",
            "pretension",
            "active",
            "total",
        ):
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"ElementForce.{name} must be finite"
                )

            setattr(
                self,
                name,
                value,
            )

    def as_dict(self) -> dict[str, float]:
        self.validate()

        return {
            "elastic": float(self.elastic),
            "damping": float(self.damping),
            "pretension": float(self.pretension),
            "active": float(self.active),
            "total": float(self.total),
        }


class Element:
    """
    Axial mechanical connection between two nodes.

    Responsibilities
    ----------------
    Element manages:

    - node connectivity;
    - geometry;
    - axial relative motion;
    - conversion of scalar axial force into nodal vectors;
    - equal and opposite force application;
    - force and geometry diagnostics;
    - material lifecycle delegation.

    Element does not own constitutive biology.

    Damage, fatigue, activation, remodeling, production,
    pretension, creep and plasticity belong to Material.
    """

    _next_id: int = 0

    def __init__(
        self,
        node_a: Any,
        node_b: Any,
        *,
        material: Material | None = None,
        rest_length: float | None = None,
        element_id: int | str | None = None,
        name: str | None = None,
        stiffness: float = 1.0,
        damping: float = 0.0,
        pretension: float = 0.0,
        tension_only: bool = False,
        compression_only: bool = False,
        enabled: bool = True,
        record_history: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        if node_a is None:
            raise ValueError(
                "node_a must be provided"
            )

        if node_b is None:
            raise ValueError(
                "node_b must be provided"
            )

        if node_a is node_b:
            raise ValueError(
                "Element cannot connect a node to itself"
            )

        self.node_a = node_a
        self.node_b = node_b

        self._validate_node_dimensions()

        self._initialize_common(
            material=material,
            rest_length=rest_length,
            element_id=element_id,
            name=name,
            stiffness=stiffness,
            damping=damping,
            pretension=pretension,
            tension_only=tension_only,
            compression_only=compression_only,
            enabled=enabled,
            record_history=record_history,
            epsilon=epsilon,
        )

        self.validate()

    def _initialize_common(
        self,
        *,
        material: Material | None = None,
        rest_length: float | None = None,
        element_id: int | str | None = None,
        name: str | None = None,
        stiffness: float = 1.0,
        damping: float = 0.0,
        pretension: float = 0.0,
        tension_only: bool = False,
        compression_only: bool = False,
        enabled: bool = True,
        record_history: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        """
        Initialize geometry-independent element state.

        Subclasses with non-two-point geometry may call this
        after their own geometry has been initialized.

        This method does not validate node_a/node_b and does not
        call validate(); geometry-specific subclasses remain
        responsible for their own validation.
        """

        self.epsilon = float(epsilon)

        if not np.isfinite(self.epsilon):
            raise ValueError(
                "epsilon must be finite"
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive"
            )

        if rest_length is None:
            rest_length = self.current_length()

        self.rest_length = self._validate_length(
            rest_length,
            name="rest_length",
        )

        if material is None:
            parameters = MaterialParameters(
                stiffness=float(stiffness),
                damping=float(damping),
            )

            material = Material(
                name="default_axial_material",
                parameters=parameters,
            )

            if hasattr(
                material.state,
                "pretension",
            ):
                material.state.pretension = float(
                    pretension
                )

        if not isinstance(material, Material):
            raise TypeError(
                "material must inherit from Material"
            )

        self.material = material

        self.tension_only = bool(
            tension_only
        )

        self.compression_only = bool(
            compression_only
        )

        if (
            self.tension_only
            and self.compression_only
        ):
            raise ValueError(
                "element cannot be both tension-only "
                "and compression-only"
            )

        self.enabled = bool(enabled)

        if element_id is None:
            element_id = Element._next_id
            Element._next_id += 1

        self.id = element_id

        if name is None:
            name = f"Element_{self.id}"

        self.name = str(name)

        self.record_history = bool(
            record_history
        )

        self.last_force = ElementForce()

        self.length_history: list[float] = []
        self.extension_history: list[float] = []
        self.strain_history: list[float] = []
        self.length_velocity_history: list[float] = []

        self.force_history: list[float] = []
        self.elastic_force_history: list[float] = []
        self.damping_force_history: list[float] = []
        self.active_force_history: list[float] = []
        self.pretension_history: list[float] = []

        self.energy_history: list[float] = []
        self.dissipated_energy_history: list[float] = []

        self._initial_state = {
            "rest_length": float(
                self.rest_length
            ),
            "enabled": bool(
                self.enabled
            ),
            "tension_only": bool(
                self.tension_only
            ),
            "compression_only": bool(
                self.compression_only
            ),
            "material": self.material.clone(),
        }

        self.synchronize_material_length()
    # =========================================================
    # Validation
    # =========================================================

    # =========================================================
    # Connectivity
    # =========================================================

    def connected_nodes(
        self,
    ) -> tuple[Any, ...]:
        """
        Return all network nodes mechanically connected
        by this element.

        The base axial Element connects exactly node_a
        and node_b. More general element geometries may
        override this contract without changing Network.
        """
        return (
            self.node_a,
            self.node_b,
        )

    @staticmethod
    def _validate_length(
        value: float,
        *,
        name: str,
    ) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

        return value

    def _validate_node_dimensions(
        self,
    ) -> None:
        position_a = self._node_vector(
            self.node_a,
            "position",
        )

        position_b = self._node_vector(
            self.node_b,
            "position",
        )

        velocity_a = self._node_vector(
            self.node_a,
            "velocity",
        )

        velocity_b = self._node_vector(
            self.node_b,
            "velocity",
        )

        expected_shape = position_a.shape

        for name, vector in (
            ("node_b.position", position_b),
            ("node_a.velocity", velocity_a),
            ("node_b.velocity", velocity_b),
        ):
            if vector.shape != expected_shape:
                raise ValueError(
                    f"{name} dimension must match "
                    "node_a.position"
                )

    def validate(self) -> None:
        self._validate_node_dimensions()

        self.rest_length = self._validate_length(
            self.rest_length,
            name="rest_length",
        )

        if not isinstance(
            self.material,
            Material,
        ):
            raise TypeError(
                "material must inherit from Material"
            )

        if hasattr(
            self.material,
            "validate",
        ):
            self.material.validate()

        self.enabled = bool(
            self.enabled
        )

        self.tension_only = bool(
            self.tension_only
        )

        self.compression_only = bool(
            self.compression_only
        )

        if (
            self.tension_only
            and self.compression_only
        ):
            raise ValueError(
                "element cannot be both tension-only "
                "and compression-only"
            )

        self.last_force.validate()

    @staticmethod
    def _node_vector(
        node: Any,
        attribute: str,
    ) -> np.ndarray:
        if not hasattr(
            node,
            attribute,
        ):
            raise AttributeError(
                f"node must define {attribute!r}"
            )

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

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                f"node.{attribute} must contain "
                "finite values"
            )

        return vector

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

    # =========================================================
    # Geometry
    # =========================================================

    @property
    def dimension(self) -> int:
        return int(
            self._node_vector(
                self.node_a,
                "position",
            ).size
        )

    def displacement_vector(
        self,
    ) -> np.ndarray:
        return (
            self._node_vector(
                self.node_b,
                "position",
            )
            - self._node_vector(
                self.node_a,
                "position",
            )
        )

    def current_length(
        self,
    ) -> float:
        return float(
            np.linalg.norm(
                self.displacement_vector()
            )
        )

    def direction(
        self,
    ) -> np.ndarray:
        displacement = (
            self.displacement_vector()
        )

        length = float(
            np.linalg.norm(
                displacement
            )
        )

        if length > self.epsilon:
            return (
                displacement
                / length
            )

        relative_velocity = (
            self.relative_velocity_vector()
        )

        velocity_norm = float(
            np.linalg.norm(
                relative_velocity
            )
        )

        if velocity_norm > self.epsilon:
            return (
                relative_velocity
                / velocity_norm
            )

        fallback = np.zeros_like(
            displacement
        )

        fallback[0] = 1.0

        return fallback

    def extension(
        self,
    ) -> float:
        return float(
            self.current_length()
            - self.material_reference_length()
        )

    def strain(
        self,
    ) -> float:
        reference_length = (
            self.material_reference_length()
        )

        if reference_length <= self.epsilon:
            return float(
                self.current_length()
            )

        return float(
            self.extension()
            / reference_length
        )

    def relative_velocity_vector(
        self,
    ) -> np.ndarray:
        return (
            self._node_vector(
                self.node_b,
                "velocity",
            )
            - self._node_vector(
                self.node_a,
                "velocity",
            )
        )

    def length_velocity(
        self,
    ) -> float:
        """
        Positive when the element length increases.
        """

        return float(
            np.dot(
                self.relative_velocity_vector(),
                self.direction(),
            )
        )

    # =========================================================
    # Material reference length
    # =========================================================

    def material_reference_length(
        self,
    ) -> float:
        """
        Return the current constitutive reference length.

        Specialized materials may alter reference length through
        plastic deformation, creep or remodeling.
        """

        if hasattr(
            self.material,
            "effective_reference_length",
        ):
            method = getattr(
                self.material,
                "effective_reference_length",
            )

            if callable(method):
                try:
                    value = method(
                        self.rest_length
                    )
                except TypeError:
                    value = method()

                value = self._validate_length(
                    value,
                    name="material reference length",
                )

                return value

        for attribute in (
            "reference_length",
            "rest_length",
        ):
            if hasattr(
                self.material,
                attribute,
            ):
                value = getattr(
                    self.material,
                    attribute,
                )

                if value is not None:
                    return self._validate_length(
                        value,
                        name=(
                            f"material.{attribute}"
                        ),
                    )

        return float(
            self.rest_length
        )

    def synchronize_material_length(
        self,
    ) -> None:
        """
        Supply element rest length to materials that maintain
        their own reference-length field.
        """

        for attribute in (
            "reference_length",
            "rest_length",
        ):
            if hasattr(
                self.material,
                attribute,
            ):
                value = getattr(
                    self.material,
                    attribute,
                )

                if value is None:
                    setattr(
                        self.material,
                        attribute,
                        float(self.rest_length),
                    )

    # =========================================================
    # Material lifecycle
    # =========================================================

    def begin_material_step(
        self,
        dt: float,
    ) -> None:
        """
        Update pre-mechanical material state once.

        Intended for neural excitation, activation and other
        control variables that must be known before force
        evaluation.
        """

        dt = self._validate_dt(dt)

        method = getattr(
            self.material,
            "begin_step",
            None,
        )

        if callable(method):
            try:
                method(
                    dt=dt,
                    current_length=self.current_length(),
                    length_velocity=self.length_velocity(),
                )
            except TypeError:
                try:
                    method(dt=dt)
                except TypeError:
                    method(dt)

    def end_material_step(
        self,
        dt: float,
        *,
        force: float | None = None,
    ) -> None:
        """
        Update post-mechanical material biology once.

        Intended for damage, fatigue, creep, plasticity,
        remodeling and energy adaptation.
        """

        dt = self._validate_dt(dt)

        if force is None:
            force = self.last_force.total

        force = float(force)

        if not np.isfinite(force):
            raise ValueError(
                "force must be finite"
            )

        method = getattr(
            self.material,
            "end_step",
            None,
        )

        if callable(method):
            try:
                method(
                    dt=dt,
                    current_length=self.current_length(),
                    length_velocity=self.length_velocity(),
                    force=force,
                )
                return
            except TypeError:
                try:
                    method(
                        dt,
                        self.current_length(),
                        self.length_velocity(),
                        force,
                    )
                    return
                except TypeError:
                    pass

        self.update_material(
            dt,
            force=force,
        )

    def update_material(
        self,
        dt: float,
        *,
        force: float | None = None,
    ) -> None:
        """
        Compatibility lifecycle for current material classes.

        This method must be called no more than once per physical
        network step.
        """

        dt = self._validate_dt(dt)

        update = getattr(
            self.material,
            "update",
            None,
        )

        if not callable(update):
            return

        current_length = self.current_length()
        length_velocity = self.length_velocity()

        keyword_attempts = (
            {
                "dt": dt,
                "current_length": current_length,
                "length_velocity": length_velocity,
                "force": force,
            },
            {
                "dt": dt,
                "current_length": current_length,
                "length_velocity": length_velocity,
            },
            {
                "dt": dt,
                "length": current_length,
                "velocity": length_velocity,
            },
            {
                "dt": dt,
            },
        )

        for arguments in keyword_attempts:
            try:
                update(**arguments)
                return
            except TypeError:
                continue

        positional_attempts = (
            (
                dt,
                current_length,
                length_velocity,
                force,
            ),
            (
                dt,
                current_length,
                length_velocity,
            ),
            (
                dt,
            ),
        )

        for arguments in positional_attempts:
            try:
                update(*arguments)
                return
            except TypeError:
                continue

        raise TypeError(
            f"{type(self.material).__name__}.update() "
            "has an unsupported signature"
        )

    # =========================================================
    # Force evaluation helpers
    # =========================================================

    @staticmethod
    def _finite_force(
        value: Any,
        *,
        name: str,
    ) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise FloatingPointError(
                f"{name} must be finite"
            )

        return value

    def _material_pretension_force(
        self,
    ) -> float:
        if hasattr(
            self.material,
            "pretension_force",
        ):
            method = getattr(
                self.material,
                "pretension_force",
            )

            if callable(method):
                try:
                    return self._finite_force(
                        method(
                            self.current_length()
                        ),
                        name="pretension force",
                    )
                except TypeError:
                    return self._finite_force(
                        method(),
                        name="pretension force",
                    )

        if hasattr(
            self.material,
            "pretension",
        ):
            return self._finite_force(
                getattr(
                    self.material,
                    "pretension",
                ),
                name="material pretension",
            )

        state = getattr(
            self.material,
            "state",
            None,
        )

        if (
            state is not None
            and hasattr(
                state,
                "pretension",
            )
        ):
            return self._finite_force(
                state.pretension,
                name="material state pretension",
            )

        return 0.0

    def _material_active_force(
        self,
    ) -> float:
        current_length = self.current_length()
        length_velocity = self.length_velocity()

        method_names = (
            "active_muscle_force",
            "active_force",
        )

        for name in method_names:
            method = getattr(
                self.material,
                name,
                None,
            )

            if not callable(method):
                continue

            attempts = (
                (
                    current_length,
                    length_velocity,
                ),
                (
                    current_length,
                ),
                (),
            )

            for arguments in attempts:
                try:
                    return self._finite_force(
                        method(*arguments),
                        name="active force",
                    )
                except TypeError:
                    continue

        return 0.0

    def _material_passive_force(
        self,
    ) -> float:
        current_length = self.current_length()
        length_velocity = self.length_velocity()
        extension = self.extension()

        specialized_methods = (
            "passive_muscle_force",
            "passive_force",
        )

        for name in specialized_methods:
            method = getattr(
                self.material,
                name,
                None,
            )

            if not callable(method):
                continue

            attempts = (
                (
                    current_length,
                    length_velocity,
                ),
                (
                    extension,
                ),
                (
                    current_length,
                ),
                (),
            )

            for arguments in attempts:
                try:
                    return self._finite_force(
                        method(*arguments),
                        name="passive force",
                    )
                except TypeError:
                    continue

        return 0.0

    def _material_damping_force(
        self,
    ) -> float:
        method = getattr(
            self.material,
            "damping_force",
            None,
        )

        if callable(method):
            attempts = (
                (
                    self.length_velocity(),
                ),
                (
                    self.current_length(),
                    self.length_velocity(),
                ),
                (),
            )

            for arguments in attempts:
                try:
                    return self._finite_force(
                        method(*arguments),
                        name="damping force",
                    )
                except TypeError:
                    continue

        effective_damping = getattr(
            self.material,
            "effective_damping",
            None,
        )

        if callable(effective_damping):
            damping = self._finite_force(
                effective_damping(),
                name="effective damping",
            )
        else:
            damping = self._finite_force(
                getattr(
                    self.material,
                    "damping",
                    0.0,
                ),
                name="material damping",
            )

        return float(
            damping
            * self.length_velocity()
        )

    def _specialized_total_force(
        self,
    ) -> float | None:
        current_length = self.current_length()
        length_velocity = self.length_velocity()

        method_names = (
            "muscle_force",
            "ligament_force",
            "fascia_force",
            "tendon_force",
        )

        for name in method_names:
            method = getattr(
                self.material,
                name,
                None,
            )

            if not callable(method):
                continue

            attempts = (
                (
                    current_length,
                    length_velocity,
                ),
                (
                    current_length,
                ),
                (),
            )

            for arguments in attempts:
                try:
                    return self._finite_force(
                        method(*arguments),
                        name=f"{name} result",
                    )
                except TypeError:
                    continue

        return None

    def _generic_total_force(
        self,
    ) -> float:
        current_length = self.current_length()
        length_velocity = self.length_velocity()
        extension = self.extension()

        force_method = getattr(
            self.material,
            "force",
            None,
        )

        if callable(force_method):
            attempts = (
                {
                    "current_length": current_length,
                    "length_velocity": length_velocity,
                    "reference_length": (
                        self.material_reference_length()
                    ),
                },
                {
                    "current_length": current_length,
                    "length_velocity": length_velocity,
                },
                {
                    "extension": extension,
                    "relative_velocity": length_velocity,
                },
            )

            for arguments in attempts:
                try:
                    return self._finite_force(
                        force_method(**arguments),
                        name="material force",
                    )
                except TypeError:
                    continue

        total_force = getattr(
            self.material,
            "total_force",
            None,
        )

        if callable(total_force):
            attempts = (
                (
                    extension,
                    length_velocity,
                ),
                (
                    current_length,
                    length_velocity,
                ),
                (
                    extension,
                ),
                (),
            )

            for arguments in attempts:
                try:
                    return self._finite_force(
                        total_force(*arguments),
                        name="material total force",
                    )
                except TypeError:
                    continue

        raise TypeError(
            f"{type(self.material).__name__} does not expose "
            "a supported force interface"
        )

    # =========================================================
    # Force evaluation
    # =========================================================

    def compute_axial_force(
        self,
        *,
        dt: float | None = None,
        update_material: bool = False,
        include_active: bool = True,
    ) -> float:
        """
        Evaluate signed axial force.

        This method should normally be side-effect free.

        ``update_material=True`` exists only for compatibility.
        Network should call material evolution explicitly once
        per physical step.
        """

        if not self.enabled:
            self.last_force = ElementForce()
            return 0.0

        if update_material:
            if dt is None:
                raise ValueError(
                    "dt is required when update_material=True"
                )

            self.update_material(dt)

        specialized_total = (
            self._specialized_total_force()
        )

        if specialized_total is not None:
            total = specialized_total

            active = (
                self._material_active_force()
                if include_active
                else 0.0
            )

            damping = (
                self._material_damping_force()
            )

            pretension = (
                self._material_pretension_force()
            )

            elastic = (
                total
                - active
                - damping
                - pretension
            )

        else:
            passive = (
                self._material_passive_force()
            )

            damping = (
                self._material_damping_force()
            )

            pretension = (
                self._material_pretension_force()
            )

            active = (
                self._material_active_force()
                if include_active
                else 0.0
            )

            components_available = any(
                abs(value) > self.epsilon
                for value in (
                    passive,
                    damping,
                    pretension,
                    active,
                )
            )

            if components_available:
                elastic = passive

                total = (
                    elastic
                    + damping
                    + pretension
                    + active
                )

            else:
                total = (
                    self._generic_total_force()
                )

                elastic = total
                damping = 0.0
                pretension = 0.0
                active = 0.0

        total = self._apply_force_mode(
            total
        )

        if total == 0.0:
            elastic = 0.0
            damping = 0.0
            pretension = 0.0
            active = 0.0

        self.last_force = ElementForce(
            elastic=float(elastic),
            damping=float(damping),
            pretension=float(pretension),
            active=float(active),
            total=float(total),
        )

        self.last_force.validate()

        return float(total)

    def _apply_force_mode(
        self,
        force: float,
    ) -> float:
        force = self._finite_force(
            force,
            name="axial force",
        )

        if (
            self.tension_only
            and force < 0.0
        ):
            return 0.0

        if (
            self.compression_only
            and force > 0.0
        ):
            return 0.0

        return force

    # =========================================================
    # Force application
    # =========================================================

    @staticmethod
    def _apply_node_force(
        node: Any,
        force: np.ndarray,
    ) -> None:
        if hasattr(
            node,
            "apply_force",
        ):
            node.apply_force(force)
            return

        if hasattr(
            node,
            "add_force",
        ):
            node.add_force(force)
            return

        node_force = np.asarray(
            getattr(node, "force"),
            dtype=float,
        )

        if node_force.shape != force.shape:
            raise ValueError(
                "node force dimension does not match "
                "element force dimension"
            )

        node.force = (
            node_force + force
        )

    def force_vector(
        self,
        axial_force: float | None = None,
    ) -> np.ndarray:
        if axial_force is None:
            axial_force = self.last_force.total

        axial_force = self._finite_force(
            axial_force,
            name="axial force",
        )

        return (
            axial_force
            * self.direction()
        )

    def apply_forces(
        self,
        *,
        dt: float | None = None,
        update_material: bool = False,
        include_active: bool = True,
    ) -> float:
        """
        Apply equal and opposite nodal forces.

        Positive axial force pulls the nodes together.
        """

        axial_force = self.compute_axial_force(
            dt=dt,
            update_material=update_material,
            include_active=include_active,
        )

        force_vector = self.force_vector(
            axial_force
        )

        self._apply_node_force(
            self.node_a,
            force_vector,
        )

        self._apply_node_force(
            self.node_b,
            -force_vector,
        )

        return axial_force

    def apply_force(
        self,
        *,
        dt: float | None = None,
        update_material: bool = False,
        include_active: bool = True,
    ) -> float:
        """
        Compatibility alias for apply_forces().
        """

        return self.apply_forces(
            dt=dt,
            update_material=update_material,
            include_active=include_active,
        )

    # =========================================================
    # Energies
    # =========================================================

    def elastic_energy(
        self,
    ) -> float:
        method = getattr(
            self.material,
            "elastic_energy",
            None,
        )

        if callable(method):
            attempts = (
                (
                    self.current_length(),
                    self.material_reference_length(),
                ),
                (
                    self.extension(),
                ),
                (),
            )

            for arguments in attempts:
                try:
                    energy = float(
                        method(*arguments)
                    )

                    if not np.isfinite(energy):
                        raise FloatingPointError(
                            "elastic energy must be finite"
                        )

                    return max(
                        0.0,
                        energy,
                    )
                except TypeError:
                    continue

        stored_energy = getattr(
            getattr(
                self.material,
                "state",
                None,
            ),
            "stored_elastic_energy",
            None,
        )

        if stored_energy is not None:
            energy = float(
                stored_energy
            )

            if not np.isfinite(energy):
                raise FloatingPointError(
                    "stored elastic energy must be finite"
                )

            return max(
                0.0,
                energy,
            )

        stiffness_method = getattr(
            self.material,
            "effective_stiffness",
            None,
        )

        if callable(stiffness_method):
            stiffness = float(
                stiffness_method()
            )
        else:
            stiffness = float(
                getattr(
                    self.material,
                    "stiffness",
                    0.0,
                )
            )

        if not np.isfinite(stiffness):
            raise FloatingPointError(
                "material stiffness must be finite"
            )

        extension = self.extension()

        return float(
            max(
                0.0,
                0.5
                * stiffness
                * extension
                * extension,
            )
        )

    def dissipated_energy(
        self,
    ) -> float:
        state = getattr(
            self.material,
            "state",
            None,
        )

        for source in (
            state,
            self.material,
        ):
            if source is None:
                continue

            if hasattr(
                source,
                "dissipated_energy",
            ):
                value = float(
                    getattr(
                        source,
                        "dissipated_energy",
                    )
                )

                if not np.isfinite(value):
                    raise FloatingPointError(
                        "dissipated energy must be finite"
                    )

                return max(
                    0.0,
                    value,
                )

        return 0.0

    # =========================================================
    # Diagnostics
    # =========================================================

    def tension(self) -> float:
        return float(
            max(
                0.0,
                self.last_force.total,
            )
        )

    def compression(self) -> float:
        return float(
            max(
                0.0,
                -self.last_force.total,
            )
        )

    def is_slack(self) -> bool:
        if not self.enabled:
            return True

        if (
            self.tension_only
            and self.extension() <= 0.0
            and self.last_force.total <= 0.0
        ):
            return True

        if (
            self.compression_only
            and self.extension() >= 0.0
            and self.last_force.total >= 0.0
        ):
            return True

        return bool(
            abs(
                self.last_force.total
            )
            <= self.epsilon
        )

    def is_failed(self) -> bool:
        state = getattr(
            self.material,
            "state",
            None,
        )

        if (
            state is not None
            and hasattr(
                state,
                "failed",
            )
        ):
            return bool(
                state.failed
            )

        if hasattr(
            self.material,
            "failed",
        ):
            return bool(
                self.material.failed
            )

        return bool(
            self.damage >= 1.0
        )

    # =========================================================
    # Runtime control
    # =========================================================

    def set_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.enabled = bool(
            enabled
        )

        if not self.enabled:
            self.last_force = ElementForce()

    def set_rest_length(
        self,
        rest_length: float,
        *,
        synchronize_material: bool = True,
    ) -> None:
        self.rest_length = self._validate_length(
            rest_length,
            name="rest_length",
        )

        if synchronize_material:
            for attribute in (
                "reference_length",
                "rest_length",
            ):
                if hasattr(
                    self.material,
                    attribute,
                ):
                    setattr(
                        self.material,
                        attribute,
                        float(self.rest_length),
                    )

    def reset_rest_length_to_current(
        self,
        *,
        synchronize_material: bool = True,
    ) -> None:
        self.set_rest_length(
            self.current_length(),
            synchronize_material=(
                synchronize_material
            ),
        )

    # =========================================================
    # History
    # =========================================================

    def record(self) -> None:
        if not self.record_history:
            return

        self.length_history.append(
            self.current_length()
        )

        self.extension_history.append(
            self.extension()
        )

        self.strain_history.append(
            self.strain()
        )

        self.length_velocity_history.append(
            self.length_velocity()
        )

        self.force_history.append(
            float(
                self.last_force.total
            )
        )

        self.elastic_force_history.append(
            float(
                self.last_force.elastic
            )
        )

        self.damping_force_history.append(
            float(
                self.last_force.damping
            )
        )

        self.active_force_history.append(
            float(
                self.last_force.active
            )
        )

        self.pretension_history.append(
            float(
                self.last_force.pretension
            )
        )

        self.energy_history.append(
            self.elastic_energy()
        )

        self.dissipated_energy_history.append(
            self.dissipated_energy()
        )

    def clear_history(self) -> None:
        self.length_history.clear()
        self.extension_history.clear()
        self.strain_history.clear()
        self.length_velocity_history.clear()

        self.force_history.clear()
        self.elastic_force_history.clear()
        self.damping_force_history.clear()
        self.active_force_history.clear()
        self.pretension_history.clear()

        self.energy_history.clear()
        self.dissipated_energy_history.clear()

    def history_length(self) -> int:
        return len(
            self.length_history
        )

    # =========================================================
    # Snapshot and restoration
    # =========================================================

    def snapshot(self) -> dict[str, Any]:
        material_snapshot = getattr(
            self.material,
            "snapshot",
            None,
        )

        if callable(material_snapshot):
            material_state = (
                material_snapshot()
            )
        else:
            material_state = {
                "type": type(
                    self.material
                ).__name__,
            }

        return {
            "id": self.id,
            "name": self.name,
            "type": type(self).__name__,
            "node_a": getattr(
                self.node_a,
                "id",
                None,
            ),
            "node_b": getattr(
                self.node_b,
                "id",
                None,
            ),
            "enabled": bool(
                self.enabled
            ),
            "dimension": self.dimension,
            "tension_only": bool(
                self.tension_only
            ),
            "compression_only": bool(
                self.compression_only
            ),
            "rest_length": float(
                self.rest_length
            ),
            "material_reference_length": float(
                self.material_reference_length()
            ),
            "current_length": float(
                self.current_length()
            ),
            "extension": float(
                self.extension()
            ),
            "strain": float(
                self.strain()
            ),
            "length_velocity": float(
                self.length_velocity()
            ),
            "force": self.last_force.as_dict(),
            "tension": self.tension(),
            "compression": self.compression(),
            "slack": self.is_slack(),
            "failed": self.is_failed(),
            "elastic_energy": (
                self.elastic_energy()
            ),
            "dissipated_energy": (
                self.dissipated_energy()
            ),
            "material": material_state,
        }

    def restore_initial_state(
        self,
        *,
        clear_history: bool = False,
    ) -> None:
        self.rest_length = float(
            self._initial_state[
                "rest_length"
            ]
        )

        self.enabled = bool(
            self._initial_state[
                "enabled"
            ]
        )

        self.tension_only = bool(
            self._initial_state[
                "tension_only"
            ]
        )

        self.compression_only = bool(
            self._initial_state[
                "compression_only"
            ]
        )

        self.material = (
            self._initial_state[
                "material"
            ].clone()
        )

        self.last_force = (
            ElementForce()
        )

        if clear_history:
            self.clear_history()

        self.validate()

    def clone(self) -> Element:
        return deepcopy(self)

    # =========================================================
    # Material compatibility properties
    # =========================================================

    @property
    def stiffness(self) -> float:
        return float(
            self.material.stiffness
        )

    @stiffness.setter
    def stiffness(
        self,
        value: float,
    ) -> None:
        self.material.stiffness = float(
            value
        )

    @property
    def damping(self) -> float:
        return float(
            self.material.damping
        )

    @damping.setter
    def damping(
        self,
        value: float,
    ) -> None:
        self.material.damping = float(
            value
        )

    @property
    def pretension(self) -> float:
        if hasattr(
            self.material,
            "pretension",
        ):
            return float(
                self.material.pretension
            )

        state = getattr(
            self.material,
            "state",
            None,
        )

        if (
            state is not None
            and hasattr(
                state,
                "pretension",
            )
        ):
            return float(
                state.pretension
            )

        return 0.0

    @pretension.setter
    def pretension(
        self,
        value: float,
    ) -> None:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "pretension must be finite"
            )

        if hasattr(
            self.material,
            "pretension",
        ):
            self.material.pretension = value
            return

        state = getattr(
            self.material,
            "state",
            None,
        )

        if (
            state is not None
            and hasattr(
                state,
                "pretension",
            )
        ):
            state.pretension = value
            return

        raise AttributeError(
            "material does not expose pretension"
        )

    @property
    def damage(self) -> float:
        return float(
            self.material.damage
        )

    @damage.setter
    def damage(
        self,
        value: float,
    ) -> None:
        self.material.damage = float(
            value
        )

    @property
    def fatigue(self) -> float:
        return float(
            self.material.fatigue
        )

    @fatigue.setter
    def fatigue(
        self,
        value: float,
    ) -> None:
        self.material.fatigue = float(
            value
        )

    @property
    def remodeling(self) -> float:
        return float(
            self.material.remodeling
        )

    @remodeling.setter
    def remodeling(
        self,
        value: float,
    ) -> None:
        self.material.remodeling = float(
            value
        )

    @property
    def production(self) -> float:
        return float(
            self.material.production
        )

    @production.setter
    def production(
        self,
        value: float,
    ) -> None:
        self.material.production = float(
            value
        )

    @property
    def energy(self) -> float:
        return float(
            self.material.energy
        )

    @energy.setter
    def energy(
        self,
        value: float,
    ) -> None:
        self.material.energy = float(
            value
        )

    # =========================================================
    # Representation
    # =========================================================

    def __repr__(self) -> str:
        node_a_id = getattr(
            self.node_a,
            "id",
            "?",
        )

        node_b_id = getattr(
            self.node_b,
            "id",
            "?",
        )

        return (
            f"Element("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"nodes=({node_a_id!r}, "
            f"{node_b_id!r}), "
            f"material="
            f"{type(self.material).__name__}, "
            f"rest_length="
            f"{self.rest_length:.6f}, "
            f"current_length="
            f"{self.current_length():.6f}, "
            f"force="
            f"{self.last_force.total:.6f}, "
            f"enabled={self.enabled})"
        )



