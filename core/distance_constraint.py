from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constraint import Constraint


@dataclass
class DistanceConstraint(Constraint):
    """
    XPBD distance constraint between two nodes.

    Constraint function
    -------------------

        C(x) = ||x_b - x_a|| - target_length

    Interpretation
    --------------
    C(x) > 0:
        current distance is greater than target length;

    C(x) < 0:
        current distance is smaller than target length;

    C(x) = 0:
        target distance is satisfied.

    The constraint can operate in three modes:

    equality:
        preserve an exact distance;

    maximum:
        activate only when the distance exceeds target_length;

    minimum:
        activate only when the distance falls below target_length.
    """

    node_a: Any = None
    node_b: Any = None

    target_length: float | None = None

    mode: str = "equality"

    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.node_a is None:
            raise ValueError(
                "node_a must be provided"
            )

        if self.node_b is None:
            raise ValueError(
                "node_b must be provided"
            )

        if self.node_a is self.node_b:
            raise ValueError(
                "DistanceConstraint cannot connect "
                "a node to itself"
            )

        self.mode = str(
            self.mode
        ).lower()

        if self.mode not in {
            "equality",
            "maximum",
            "minimum",
        }:
            raise ValueError(
                "mode must be 'equality', "
                "'maximum', or 'minimum'"
            )

        self.epsilon = float(
            self.epsilon
        )

        if not np.isfinite(
            self.epsilon
        ):
            raise ValueError(
                "epsilon must be finite"
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive"
            )

        self._validate_node_dimensions()

        if self.target_length is None:
            self.target_length = (
                self.current_length()
            )

        self.target_length = float(
            self.target_length
        )

        if not np.isfinite(
            self.target_length
        ):
            raise ValueError(
                "target_length must be finite"
            )

        if self.target_length < 0.0:
            raise ValueError(
                "target_length cannot be negative"
            )

        super().__post_init__()

    # =========================================================
    # Node validation
    # =========================================================

    def _validate_node_dimensions(
        self,
    ) -> None:
        position_a = self.node_position(
            self.node_a
        )

        position_b = self.node_position(
            self.node_b
        )

        velocity_a = self.node_velocity(
            self.node_a
        )

        velocity_b = self.node_velocity(
            self.node_b
        )

        dimensions = {
            position_a.shape,
            position_b.shape,
            velocity_a.shape,
            velocity_b.shape,
        }

        if len(dimensions) != 1:
            raise ValueError(
                "connected node positions and velocities "
                "must have matching dimensions"
            )

    # =========================================================
    # Geometry
    # =========================================================

    def displacement_vector(
        self,
    ) -> np.ndarray:
        """
        Vector from node_a to node_b.
        """

        position_a = self.node_position(
            self.node_a
        )

        position_b = self.node_position(
            self.node_b
        )

        if (
            position_a.shape
            != position_b.shape
        ):
            raise ValueError(
                "connected node positions must have "
                "matching dimensions"
            )

        return (
            position_b
            - position_a
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
        """
        Unit vector from node_a to node_b.

        When both nodes occupy the same position, a stable
        fallback direction is selected.
        """

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
            self.node_velocity(
                self.node_b
            )
            - self.node_velocity(
                self.node_a
            )
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
        """
        Signed distance difference relative to target length.
        """

        return float(
            self.current_length()
            - self.target_length
        )

    def strain(
        self,
    ) -> float:
        """
        Engineering strain relative to target length.

        For target_length == 0, the current length is returned
        as an absolute geometric error.
        """

        if (
            self.target_length
            <= self.epsilon
        ):
            return float(
                self.current_length()
            )

        return float(
            self.extension()
            / self.target_length
        )

    # =========================================================
    # Constraint contract
    # =========================================================

    def evaluate(
        self,
    ) -> float:
        """
        Return signed constraint value C(x).
        """

        return self.extension()

    def gradients(
        self,
    ) -> tuple[
        tuple[Any, np.ndarray],
        tuple[Any, np.ndarray],
    ]:
        """
        Return gradients of C(x) for both nodes.

        For:

            C(x) = ||x_b - x_a|| - L

        gradients are:

            dC/dx_a = -n
            dC/dx_b = +n
        """

        axis = self.direction()

        return (
            (
                self.node_a,
                -axis,
            ),
            (
                self.node_b,
                axis,
            ),
        )

    def update_active_state(
        self,
        violation: float | None = None,
    ) -> bool:
        """
        Activate equality or unilateral distance constraints.
        """

        if not self.enabled:
            self.state.active = False
            return False

        if violation is None:
            violation = self.evaluate()

        violation = float(
            violation
        )

        self._validate_finite_scalar(
            violation,
            "constraint violation",
        )

        if self.mode == "equality":
            active = True

        elif self.mode == "maximum":
            active = (
                violation > 0.0
                or self.state.impulse < 0.0
            )

        elif self.mode == "minimum":
            active = (
                violation < 0.0
                or self.state.impulse > 0.0
            )

        else:
            raise RuntimeError(
                f"Unsupported constraint mode: "
                f"{self.mode!r}"
            )

        self.state.active = bool(
            active
        )

        return self.state.active

    # =========================================================
    # XPBD projection
    # =========================================================

    def calculate_delta_lambda(
        self,
        *,
        violation: float,
        denominator: float,
        dt: float,
        relaxation: float = 1.0,
    ) -> float:
        """
        Calculate and clamp XPBD lambda for unilateral modes.

        Equality mode uses the base implementation.

        Maximum-distance constraint:
            lambda <= 0

        Minimum-distance constraint:
            lambda >= 0
        """

        delta_lambda = (
            super().calculate_delta_lambda(
                violation=violation,
                denominator=denominator,
                dt=dt,
                relaxation=relaxation,
            )
        )

        if self.mode == "equality":
            return delta_lambda

        old_lambda = float(
            self.state.impulse
        )

        candidate_lambda = (
            old_lambda
            + delta_lambda
        )

        if self.mode == "maximum":
            clamped_lambda = min(
                0.0,
                candidate_lambda,
            )

        elif self.mode == "minimum":
            clamped_lambda = max(
                0.0,
                candidate_lambda,
            )

        else:
            raise RuntimeError(
                f"Unsupported constraint mode: "
                f"{self.mode!r}"
            )

        return float(
            clamped_lambda
            - old_lambda
        )

    def project(
        self,
        dt: float,
        relaxation: float = 1.0,
    ) -> float:
        """
        Perform one XPBD positional projection.
        """

        return self.perform_xpbd_projection(
            dt=dt,
            relaxation=relaxation,
        )

    # =========================================================
    # Velocity diagnostics
    # =========================================================

    def relative_velocity_vector(
        self,
    ) -> np.ndarray:
        return (
            self.node_velocity(
                self.node_b
            )
            - self.node_velocity(
                self.node_a
            )
        )

    def length_velocity(
        self,
    ) -> float:
        """
        Relative velocity along the constraint axis.

        Positive:
            nodes move farther apart;

        negative:
            nodes move closer together.
        """

        return float(
            np.dot(
                self.relative_velocity_vector(),
                self.direction(),
            )
        )

    # =========================================================
    # Constraint-specific damping
    # =========================================================

    def damp_velocity(
        self,
        dt: float,
    ) -> None:
        """
        Apply damping only along the distance axis.

        For unilateral modes, damping is only applied while the
        constraint is active.
        """

        self.validate_dt(dt)

        if not self.enabled:
            return

        if not self.state.active:
            return

        if self.damping <= 0.0:
            return

        direction = self.direction()

        inverse_mass_a = (
            self.node_inverse_mass(
                self.node_a
            )
        )

        inverse_mass_b = (
            self.node_inverse_mass(
                self.node_b
            )
        )

        denominator = (
            inverse_mass_a
            + inverse_mass_b
        )

        if denominator <= self.epsilon:
            return

        velocity_a = self.node_velocity(
            self.node_a
        )

        velocity_b = self.node_velocity(
            self.node_b
        )

        relative_axial_velocity = float(
            np.dot(
                velocity_b
                - velocity_a,
                direction,
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
            * relative_axial_velocity
            / denominator
        )

        if (
            self.mode == "maximum"
            and relative_axial_velocity <= 0.0
        ):
            return

        if (
            self.mode == "minimum"
            and relative_axial_velocity >= 0.0
        ):
            return

        velocity_correction_a = (
            -inverse_mass_a
            * velocity_lambda
            * direction
        )

        velocity_correction_b = (
            inverse_mass_b
            * velocity_lambda
            * direction
        )

        if inverse_mass_a > 0.0:
            self.node_a.velocity = (
                velocity_a
                + velocity_correction_a
            )

        if inverse_mass_b > 0.0:
            self.node_b.velocity = (
                velocity_b
                + velocity_correction_b
            )

    # =========================================================
    # Runtime control
    # =========================================================

    def set_target_length(
        self,
        target_length: float,
    ) -> None:
        target_length = float(
            target_length
        )

        if not np.isfinite(
            target_length
        ):
            raise ValueError(
                "target_length must be finite"
            )

        if target_length < 0.0:
            raise ValueError(
                "target_length cannot be negative"
            )

        self.target_length = (
            target_length
        )

    def reset_target_to_current(
        self,
    ) -> None:
        self.target_length = (
            self.current_length()
        )

        self.reset_lambda()

    def set_mode(
        self,
        mode: str,
    ) -> None:
        mode = str(
            mode
        ).lower()

        if mode not in {
            "equality",
            "maximum",
            "minimum",
        }:
            raise ValueError(
                "mode must be 'equality', "
                "'maximum', or 'minimum'"
            )

        self.mode = mode
        self.reset_lambda()

    # =========================================================
    # Diagnostics
    # =========================================================

    def relative_error(
        self,
        scale: float | None = None,
    ) -> float:
        """
        Normalized distance error.
        """

        if scale is None:
            scale = max(
                abs(self.target_length),
                self.epsilon,
            )

        return super().relative_error(
            scale=scale
        )

    def is_satisfied(
        self,
        tolerance: float = 1e-6,
    ) -> bool:
        tolerance = float(
            tolerance
        )

        if not np.isfinite(
            tolerance
        ):
            raise ValueError(
                "tolerance must be finite"
            )

        if tolerance < 0.0:
            raise ValueError(
                "tolerance cannot be negative"
            )

        violation = self.evaluate()

        if self.mode == "equality":
            return bool(
                abs(violation)
                <= tolerance
            )

        if self.mode == "maximum":
            return bool(
                violation
                <= tolerance
            )

        if self.mode == "minimum":
            return bool(
                violation
                >= -tolerance
            )

        raise RuntimeError(
            f"Unsupported constraint mode: "
            f"{self.mode!r}"
        )

    def snapshot(
        self,
    ) -> dict[str, Any]:
        snapshot = super().snapshot()

        snapshot.update(
            {
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
                "mode": self.mode,
                "target_length": (
                    self.target_length
                ),
                "current_length": (
                    self.current_length()
                ),
                "extension": (
                    self.extension()
                ),
                "strain": (
                    self.strain()
                ),
                "relative_error": (
                    self.relative_error()
                ),
                "length_velocity": (
                    self.length_velocity()
                ),
                "satisfied": (
                    self.is_satisfied()
                ),
            }
        )

        return snapshot

    def __repr__(
        self,
    ) -> str:
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
            f"DistanceConstraint("
            f"name={self.name!r}, "
            f"nodes=({node_a_id!r}, "
            f"{node_b_id!r}), "
            f"mode={self.mode!r}, "
            f"target={self.target_length:.6f}, "
            f"current={self.current_length():.6f}, "
            f"violation={self.evaluate():.3e}, "
            f"compliance={self.compliance:.3e}, "
            f"damping={self.damping:.3f}, "
            f"active={self.state.active})"
        )