from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class ConstraintState:
    """
    Runtime state shared by all constraints.

    Sign convention
    ---------------
    violation:
        Signed value of C(x).

    correction:
        Magnitude or vector representation of the most recent
        positional correction.

    impulse:
        Accumulated XPBD Lagrange multiplier.

        The historical field name ``impulse`` is retained for
        compatibility. Internally it represents lambda rather
        than a classical instantaneous impulse.

    active:
        Whether the constraint currently participates in solving.
    """

    violation: float = 0.0

    correction: float | np.ndarray = 0.0

    impulse: float = 0.0

    active: bool = True

    projected: bool = False

    projection_count: int = 0

    last_delta_lambda: float = 0.0

    maximum_correction: float = 0.0

    initial_violation: float = 0.0

    final_violation: float = 0.0

    def reset_step(
        self,
        *,
        reset_impulse: bool = True,
    ) -> None:
        """
        Reset values that belong to one physical time step.
        """

        self.violation = 0.0
        self.correction = 0.0
        self.projected = False
        self.projection_count = 0
        self.last_delta_lambda = 0.0
        self.maximum_correction = 0.0
        self.initial_violation = 0.0
        self.final_violation = 0.0

        if reset_impulse:
            self.impulse = 0.0

    def set_correction(
        self,
        correction: float | Iterable[float] | np.ndarray,
    ) -> None:
        """
        Store the most recent correction and update diagnostics.
        """

        array = np.asarray(
            correction,
            dtype=float,
        )

        if array.ndim == 0:
            value = float(array)

            if not np.isfinite(value):
                raise FloatingPointError(
                    "constraint correction must be finite"
                )

            self.correction = value
            magnitude = abs(value)

        else:
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(
                    "constraint correction must contain "
                    "finite values"
                )

            self.correction = array.copy()
            magnitude = float(
                np.linalg.norm(array)
            )

        self.maximum_correction = max(
            self.maximum_correction,
            magnitude,
        )

    def correction_magnitude(self) -> float:
        correction = np.asarray(
            self.correction,
            dtype=float,
        )

        if correction.size == 0:
            return 0.0

        return float(
            np.linalg.norm(correction)
        )

    def clamp(self) -> None:
        """
        Validate and sanitize scalar diagnostic values.
        """

        scalar_fields = (
            "violation",
            "impulse",
            "last_delta_lambda",
            "maximum_correction",
            "initial_violation",
            "final_violation",
        )

        for name in scalar_fields:
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"ConstraintState.{name} must be finite"
                )

            setattr(
                self,
                name,
                value,
            )

        self.projection_count = max(
            0,
            int(self.projection_count),
        )

        self.active = bool(self.active)
        self.projected = bool(self.projected)


@dataclass
class Constraint(ABC):
    """
    Abstract base class for XPBD constraints.

    Required subclass methods
    -------------------------
    evaluate()
        Return signed constraint violation C(x).

    gradients()
        Return node-gradient pairs for C(x).

    project()
        Apply one XPBD positional projection.

    Solver lifecycle
    ----------------
    The Solver calls:

        begin_step(dt)
        project(dt, relaxation)  # multiple times
        damp_velocity(dt)
        end_step(dt)

    The base class manages:

    - compliance;
    - damping;
    - enabled and active state;
    - accumulated XPBD lambda;
    - per-step diagnostics;
    - node mass utilities;
    - validation.
    """

    compliance: float = 0.0

    damping: float = 0.0

    enabled: bool = True

    warm_start: bool = False

    name: str | None = None

    state: ConstraintState = field(
        default_factory=ConstraintState
    )

    _current_dt: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.compliance = float(
            self.compliance
        )

        self.damping = float(
            self.damping
        )

        self.enabled = bool(
            self.enabled
        )

        self.warm_start = bool(
            self.warm_start
        )

        if self.name is None:
            self.name = type(self).__name__

        self.validate()

    # =========================================================
    # Required constraint interface
    # =========================================================

    @abstractmethod
    def evaluate(self) -> float:
        """
        Return signed constraint function C(x).

        A value of zero means the constraint is satisfied.
        """

        raise NotImplementedError

    @abstractmethod
    def gradients(
        self,
    ) -> tuple[
        tuple[Any, np.ndarray],
        ...,
    ]:
        """
        Return pairs of:

            (node, gradient_of_C_with_respect_to_node)

        Example for a distance constraint:

            (
                (node_a, -direction),
                (node_b, +direction),
            )
        """

        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        dt: float,
        relaxation: float = 1.0,
    ) -> float | np.ndarray | None:
        """
        Perform one positional XPBD projection.

        Subclasses should:

        1. evaluate C(x);
        2. calculate gradients;
        3. calculate delta lambda;
        4. modify node positions;
        5. update ConstraintState;
        6. return correction magnitude/vector or None.
        """

        raise NotImplementedError

    # =========================================================
    # Validation
    # =========================================================

    def validate(self) -> None:
        finite_fields = {
            "compliance": self.compliance,
            "damping": self.damping,
        }

        for name, value in finite_fields.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        if self.compliance < 0.0:
            raise ValueError(
                "compliance cannot be negative"
            )

        if self.damping < 0.0:
            raise ValueError(
                "damping cannot be negative"
            )

        self.state.clamp()

    @staticmethod
    def validate_dt(
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

    @staticmethod
    def validate_relaxation(
        relaxation: float,
    ) -> float:
        relaxation = float(
            relaxation
        )

        if not np.isfinite(relaxation):
            raise ValueError(
                "relaxation must be finite"
            )

        if relaxation <= 0.0:
            raise ValueError(
                "relaxation must be positive"
            )

        return relaxation

    # =========================================================
    # Constraint lifecycle
    # =========================================================

    def begin_step(
        self,
        dt: float,
    ) -> None:
        """
        Initialize the constraint once per physical step.

        By default XPBD lambda is reset every step. When
        warm_start=True, the accumulated value is preserved.
        """

        dt = self.validate_dt(dt)

        self._current_dt = dt

        self.state.reset_step(
            reset_impulse=not self.warm_start,
        )

        if not self.enabled:
            self.state.active = False
            return

        violation = float(
            self.evaluate()
        )

        self._validate_finite_scalar(
            violation,
            "initial violation",
        )

        self.state.violation = violation
        self.state.initial_violation = violation
        self.state.final_violation = violation

        self.update_active_state(
            violation
        )

        self.state.clamp()

    def end_step(
        self,
        dt: float,
    ) -> None:
        """
        Finalize diagnostics once after all projections.
        """

        self.validate_dt(dt)

        if self.enabled:
            violation = float(
                self.evaluate()
            )

            self._validate_finite_scalar(
                violation,
                "final violation",
            )

            self.state.violation = violation
            self.state.final_violation = violation

        self.state.clamp()
        self._current_dt = None

    def update_active_state(
        self,
        violation: float | None = None,
    ) -> bool:
        """
        Update whether the constraint should participate.

        Equality constraints are active whenever enabled.
        Inequality constraints can override this method.
        """

        self.state.active = bool(
            self.enabled
        )

        return self.state.active

    # =========================================================
    # XPBD utilities
    # =========================================================

    def alpha(
        self,
        dt: float,
    ) -> float:
        """
        Time-scaled XPBD compliance:

            alpha_tilde = compliance / dt²
        """

        dt = self.validate_dt(dt)

        return float(
            self.compliance / (dt * dt)
        )

    def effective_inverse_mass(
        self,
        gradients: (
            tuple[
                tuple[Any, np.ndarray],
                ...,
            ]
            | None
        ) = None,
    ) -> float:
        """
        Calculate:

            Σ w_i ||∇C_i||²

        where w_i is node inverse mass.
        """

        if gradients is None:
            gradients = self.gradients()

        denominator = 0.0

        for node, gradient in gradients:
            gradient_vector = self._vector(
                gradient,
                name="constraint gradient",
            )

            inverse_mass = self.node_inverse_mass(
                node
            )

            denominator += (
                inverse_mass
                * float(
                    np.dot(
                        gradient_vector,
                        gradient_vector,
                    )
                )
            )

        return float(denominator)

    def calculate_delta_lambda(
        self,
        *,
        violation: float,
        denominator: float,
        dt: float,
        relaxation: float = 1.0,
    ) -> float:
        """
        Standard XPBD multiplier increment:

            Δλ =
                (-C - α λ)
                / (Σ w_i ||∇C_i||² + α)

        followed by global relaxation.
        """

        dt = self.validate_dt(dt)
        relaxation = self.validate_relaxation(
            relaxation
        )

        violation = float(
            violation
        )

        denominator = float(
            denominator
        )

        self._validate_finite_scalar(
            violation,
            "violation",
        )

        self._validate_finite_scalar(
            denominator,
            "effective inverse mass",
        )

        if denominator < 0.0:
            raise ValueError(
                "effective inverse mass cannot be negative"
            )

        alpha = self.alpha(dt)

        full_denominator = (
            denominator + alpha
        )

        if full_denominator <= 1e-15:
            return 0.0

        delta_lambda = (
            -violation
            - alpha * self.state.impulse
        ) / full_denominator

        delta_lambda *= relaxation

        self._validate_finite_scalar(
            delta_lambda,
            "delta lambda",
        )

        return float(delta_lambda)

    def apply_position_corrections(
        self,
        *,
        gradients: tuple[
            tuple[Any, np.ndarray],
            ...,
        ],
        delta_lambda: float,
    ) -> float:
        """
        Apply:

            Δx_i = w_i ∇C_i Δλ

        Returns the largest nodal correction magnitude.
        """

        delta_lambda = float(
            delta_lambda
        )

        self._validate_finite_scalar(
            delta_lambda,
            "delta lambda",
        )

        correction_vectors: list[
            np.ndarray
        ] = []

        maximum_correction = 0.0

        for node, gradient in gradients:
            gradient_vector = self._vector(
                gradient,
                name="constraint gradient",
            )

            inverse_mass = self.node_inverse_mass(
                node
            )

            if inverse_mass <= 0.0:
                correction = np.zeros_like(
                    gradient_vector
                )
            else:
                correction = (
                    inverse_mass
                    * delta_lambda
                    * gradient_vector
                )

                position = self.node_position(
                    node
                )

                if position.shape != correction.shape:
                    raise ValueError(
                        "constraint gradient dimension must "
                        "match node position dimension"
                    )

                node.position = (
                    position + correction
                )

            correction_vectors.append(
                correction
            )

            maximum_correction = max(
                maximum_correction,
                float(
                    np.linalg.norm(
                        correction
                    )
                ),
            )

        if correction_vectors:
            combined = np.concatenate(
                [
                    correction.reshape(-1)
                    for correction in correction_vectors
                ]
            )

            self.state.set_correction(
                combined
            )
        else:
            self.state.set_correction(
                0.0
            )

        self.state.maximum_correction = max(
            self.state.maximum_correction,
            maximum_correction,
        )

        return float(
            maximum_correction
        )

    def perform_xpbd_projection(
        self,
        *,
        dt: float,
        relaxation: float = 1.0,
    ) -> float:
        """
        Generic equality-constraint XPBD projection.

        Simple subclasses may call this directly from project().
        """

        dt = self.validate_dt(dt)
        relaxation = self.validate_relaxation(
            relaxation
        )

        if not self.enabled:
            self.state.active = False
            return 0.0

        violation = float(
            self.evaluate()
        )

        self._validate_finite_scalar(
            violation,
            "constraint violation",
        )

        self.state.violation = violation

        if not self.update_active_state(
            violation
        ):
            self.state.set_correction(
                0.0
            )
            return 0.0

        gradients = self.gradients()

        self.validate_gradients(
            gradients
        )

        denominator = (
            self.effective_inverse_mass(
                gradients
            )
        )

        delta_lambda = (
            self.calculate_delta_lambda(
                violation=violation,
                denominator=denominator,
                dt=dt,
                relaxation=relaxation,
            )
        )

        maximum_correction = (
            self.apply_position_corrections(
                gradients=gradients,
                delta_lambda=delta_lambda,
            )
        )

        self.state.impulse += (
            delta_lambda
        )

        self.state.last_delta_lambda = (
            delta_lambda
        )

        self.state.projected = True

        self.state.projection_count += 1

        final_violation = float(
            self.evaluate()
        )

        self._validate_finite_scalar(
            final_violation,
            "post-projection violation",
        )

        self.state.violation = (
            final_violation
        )

        self.state.final_violation = (
            final_violation
        )

        self.state.clamp()

        return maximum_correction

    # =========================================================
    # Velocity damping
    # =========================================================

    def damp_velocity(
        self,
        dt: float,
    ) -> None:
        """
        Remove part of relative velocity along the constraint
        gradients.

        This is a general gradient-based damping operation:

            Jv = Σ ∇C_i · v_i

            impulse =
                -damping * Jv
                / Σ w_i ||∇C_i||²

        The damping parameter is interpreted as a dimensionless
        fraction per physical step and is clamped to [0, 1].
        """

        self.validate_dt(dt)

        if not self.enabled:
            return

        if not self.state.active:
            return

        if self.damping <= 0.0:
            return

        gradients = self.gradients()

        self.validate_gradients(
            gradients
        )

        denominator = (
            self.effective_inverse_mass(
                gradients
            )
        )

        if denominator <= 1e-15:
            return

        constraint_velocity = 0.0

        for node, gradient in gradients:
            gradient_vector = self._vector(
                gradient,
                name="constraint gradient",
            )

            velocity = self.node_velocity(
                node
            )

            if gradient_vector.shape != velocity.shape:
                raise ValueError(
                    "constraint gradient dimension must "
                    "match node velocity dimension"
                )

            constraint_velocity += float(
                np.dot(
                    gradient_vector,
                    velocity,
                )
            )

        damping_fraction = float(
            np.clip(
                self.damping,
                0.0,
                1.0,
            )
        )

        velocity_lambda = (
            -damping_fraction
            * constraint_velocity
            / denominator
        )

        self._validate_finite_scalar(
            velocity_lambda,
            "velocity damping multiplier",
        )

        for node, gradient in gradients:
            inverse_mass = self.node_inverse_mass(
                node
            )

            if inverse_mass <= 0.0:
                continue

            gradient_vector = self._vector(
                gradient,
                name="constraint gradient",
            )

            velocity = self.node_velocity(
                node
            )

            node.velocity = (
                velocity
                + inverse_mass
                * velocity_lambda
                * gradient_vector
            )

    # =========================================================
    # Gradient validation
    # =========================================================

    def validate_gradients(
        self,
        gradients: tuple[
            tuple[Any, np.ndarray],
            ...,
        ],
    ) -> None:
        if not isinstance(
            gradients,
            tuple,
        ):
            raise TypeError(
                "gradients() must return a tuple"
            )

        if len(gradients) == 0:
            raise ValueError(
                "gradients() cannot return an empty tuple"
            )

        for index, item in enumerate(
            gradients
        ):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
            ):
                raise TypeError(
                    "each gradient entry must be "
                    "(node, gradient)"
                )

            node, gradient = item

            gradient_vector = self._vector(
                gradient,
                name=f"gradient[{index}]",
            )

            position = self.node_position(
                node
            )

            if gradient_vector.shape != position.shape:
                raise ValueError(
                    f"gradient[{index}] dimension does not "
                    "match node position dimension"
                )

    # =========================================================
    # Node helpers
    # =========================================================

    @staticmethod
    def node_inverse_mass(
        node: Any,
    ) -> float:
        """
        Return zero for fixed nodes and 1 / mass otherwise.
        """

        if bool(
            getattr(
                node,
                "fixed",
                False,
            )
        ):
            return 0.0

        inverse_mass = getattr(
            node,
            "inverse_mass",
            None,
        )

        if callable(inverse_mass):
            value = float(
                inverse_mass()
            )

        elif inverse_mass is not None:
            value = float(
                inverse_mass
            )

        else:
            mass = float(
                getattr(
                    node,
                    "mass",
                )
            )

            if not np.isfinite(mass):
                raise ValueError(
                    "node mass must be finite"
                )

            if mass <= 0.0:
                raise ValueError(
                    "node mass must be positive"
                )

            value = 1.0 / mass

        if not np.isfinite(value):
            raise ValueError(
                "node inverse mass must be finite"
            )

        if value < 0.0:
            raise ValueError(
                "node inverse mass cannot be negative"
            )

        return value

    @classmethod
    def node_position(
        cls,
        node: Any,
    ) -> np.ndarray:
        return cls._vector(
            getattr(
                node,
                "position",
            ),
            name="node.position",
        )

    @classmethod
    def node_velocity(
        cls,
        node: Any,
    ) -> np.ndarray:
        return cls._vector(
            getattr(
                node,
                "velocity",
            ),
            name="node.velocity",
        )

    @staticmethod
    def _vector(
        value: Any,
        *,
        name: str,
    ) -> np.ndarray:
        vector = np.asarray(
            value,
            dtype=float,
        )

        if vector.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional"
            )

        if vector.size == 0:
            raise ValueError(
                f"{name} cannot be empty"
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                f"{name} must contain finite values"
            )

        return vector

    @staticmethod
    def _validate_finite_scalar(
        value: float,
        name: str,
    ) -> None:
        if not np.isfinite(value):
            raise FloatingPointError(
                f"{name} must be finite"
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

        self.state.active = (
            self.enabled
        )

        if not self.enabled:
            self.state.correction = 0.0
            self.state.last_delta_lambda = 0.0

    def set_compliance(
        self,
        compliance: float,
    ) -> None:
        previous = self.compliance
        self.compliance = float(
            compliance
        )

        try:
            self.validate()
        except (
            ValueError,
            FloatingPointError,
        ):
            self.compliance = previous
            raise

    def set_damping(
        self,
        damping: float,
    ) -> None:
        previous = self.damping
        self.damping = float(
            damping
        )

        try:
            self.validate()
        except (
            ValueError,
            FloatingPointError,
        ):
            self.damping = previous
            raise

    def reset_lambda(self) -> None:
        self.state.impulse = 0.0
        self.state.last_delta_lambda = 0.0

    # =========================================================
    # Diagnostics
    # =========================================================

    @property
    def lambda_value(self) -> float:
        """
        Explicit alias for the legacy ``state.impulse`` field.
        """

        return float(
            self.state.impulse
        )

    @lambda_value.setter
    def lambda_value(
        self,
        value: float,
    ) -> None:
        value = float(
            value
        )

        self._validate_finite_scalar(
            value,
            "lambda value",
        )

        self.state.impulse = value

    def relative_error(
        self,
        scale: float = 1.0,
    ) -> float:
        """
        Generic normalized violation diagnostic.
        """

        scale = abs(
            float(scale)
        )

        if scale <= 1e-15:
            scale = 1.0

        return float(
            abs(
                self.evaluate()
            )
            / scale
        )

    def snapshot(self) -> dict[str, Any]:
        correction = np.asarray(
            self.state.correction,
            dtype=float,
        )

        if correction.ndim == 0:
            correction_value: (
                float | list[float]
            ) = float(correction)
        else:
            correction_value = (
                correction.tolist()
            )

        return {
            "name": self.name,
            "type": type(self).__name__,
            "enabled": self.enabled,
            "warm_start": self.warm_start,
            "compliance": self.compliance,
            "damping": self.damping,
            "state": {
                "violation": (
                    self.state.violation
                ),
                "correction": correction_value,
                "correction_magnitude": (
                    self.state
                    .correction_magnitude()
                ),
                "impulse": (
                    self.state.impulse
                ),
                "lambda": (
                    self.state.impulse
                ),
                "active": (
                    self.state.active
                ),
                "projected": (
                    self.state.projected
                ),
                "projection_count": (
                    self.state.projection_count
                ),
                "last_delta_lambda": (
                    self.state
                    .last_delta_lambda
                ),
                "maximum_correction": (
                    self.state
                    .maximum_correction
                ),
                "initial_violation": (
                    self.state
                    .initial_violation
                ),
                "final_violation": (
                    self.state
                    .final_violation
                ),
            },
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"compliance={self.compliance:.3e}, "
            f"damping={self.damping:.3f}, "
            f"enabled={self.enabled}, "
            f"active={self.state.active}, "
            f"violation={self.state.violation:.3e}, "
            f"lambda={self.state.impulse:.3e})"
        )