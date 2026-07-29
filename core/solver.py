from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .constraint import Constraint


@dataclass
class SolverIterationStats:
    """
    Diagnostics for one XPBD solver iteration.
    """

    iteration: int = 0

    active_constraints: int = 0
    projected_constraints: int = 0

    maximum_violation: float = 0.0
    mean_violation: float = 0.0
    rms_violation: float = 0.0

    maximum_correction: float = 0.0
    mean_correction: float = 0.0

    converged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "active_constraints": int(
                self.active_constraints
            ),
            "projected_constraints": int(
                self.projected_constraints
            ),
            "maximum_violation": float(
                self.maximum_violation
            ),
            "mean_violation": float(
                self.mean_violation
            ),
            "rms_violation": float(
                self.rms_violation
            ),
            "maximum_correction": float(
                self.maximum_correction
            ),
            "mean_correction": float(
                self.mean_correction
            ),
            "converged": bool(self.converged),
        }


@dataclass
class SolverStatistics:
    """
    Diagnostics for one complete XPBD solve.
    """

    iterations: int = 0
    converged: bool = False

    active_constraints: int = 0
    projected_constraints: int = 0

    initial_maximum_violation: float = 0.0
    maximum_violation: float = 0.0
    mean_violation: float = 0.0
    rms_violation: float = 0.0

    maximum_correction: float = 0.0
    mean_correction: float = 0.0

    stagnated: bool = False
    divergence_detected: bool = False

    iteration_history: list[
        SolverIterationStats
    ] = field(
        default_factory=list
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "active_constraints": int(
                self.active_constraints
            ),
            "projected_constraints": int(
                self.projected_constraints
            ),
            "initial_maximum_violation": float(
                self.initial_maximum_violation
            ),
            "maximum_violation": float(
                self.maximum_violation
            ),
            "mean_violation": float(
                self.mean_violation
            ),
            "rms_violation": float(
                self.rms_violation
            ),
            "maximum_correction": float(
                self.maximum_correction
            ),
            "mean_correction": float(
                self.mean_correction
            ),
            "stagnated": bool(self.stagnated),
            "divergence_detected": bool(
                self.divergence_detected
            ),
            "iteration_history": [
                item.as_dict()
                for item in self.iteration_history
            ],
        }


class Solver:
    """
    Iterative XPBD constraint solver.

    Solver responsibilities
    -----------------------
    The solver:

    - validates the time step;
    - initializes constraints once per physical step;
    - iteratively projects enabled constraints;
    - measures convergence;
    - detects stagnation and divergence;
    - applies velocity damping after projection;
    - finalizes constraint states;
    - returns complete diagnostics.

    The solver does not:

    - integrate node forces;
    - update biological materials;
    - advance simulation time;
    - assemble element forces.

    Those responsibilities belong to Network.
    """

    def __init__(
        self,
        *,
        iterations: int = 20,
        tolerance: float = 1e-6,
        correction_tolerance: float = 1e-9,
        minimum_iterations: int = 1,
        relaxation: float = 1.0,
        enable_velocity_damping: bool = True,
        record_iteration_history: bool = False,
        detect_stagnation: bool = True,
        stagnation_window: int = 4,
        stagnation_tolerance: float = 1e-4,
        divergence_ratio: float = 10.0,
        fail_on_nonfinite: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        iterations:
            Maximum number of projection iterations.

        tolerance:
            Maximum absolute constraint violation required
            for convergence.

        correction_tolerance:
            Maximum positional correction below which the
            solution can also be considered stationary.

        minimum_iterations:
            Minimum iteration count before convergence testing
            may terminate the solve.

        relaxation:
            Global projection multiplier.

            1.0:
                normal projection;

            below 1.0:
                under-relaxation;

            above 1.0:
                over-relaxation.

        enable_velocity_damping:
            Apply constraint-specific velocity damping once
            after positional projection.

        record_iteration_history:
            Preserve diagnostics for every iteration.

        detect_stagnation:
            Detect when violation stops improving.

        stagnation_window:
            Number of recent iterations used for stagnation
            detection.

        stagnation_tolerance:
            Minimum relative improvement over the stagnation
            window.

        divergence_ratio:
            Mark divergence if violation exceeds the initial
            violation by this factor.

        fail_on_nonfinite:
            Raise an exception when a constraint produces NaN
            or infinity.
        """

        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.correction_tolerance = float(
            correction_tolerance
        )
        self.minimum_iterations = int(
            minimum_iterations
        )
        self.relaxation = float(relaxation)

        self.enable_velocity_damping = bool(
            enable_velocity_damping
        )

        self.record_iteration_history = bool(
            record_iteration_history
        )

        self.detect_stagnation = bool(
            detect_stagnation
        )

        self.stagnation_window = int(
            stagnation_window
        )

        self.stagnation_tolerance = float(
            stagnation_tolerance
        )

        self.divergence_ratio = float(
            divergence_ratio
        )

        self.fail_on_nonfinite = bool(
            fail_on_nonfinite
        )

        self._validate_configuration()

        self.last_statistics = SolverStatistics()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_configuration(self) -> None:
        if self.iterations <= 0:
            raise ValueError(
                "iterations must be positive"
            )

        if self.minimum_iterations <= 0:
            raise ValueError(
                "minimum_iterations must be positive"
            )

        if self.minimum_iterations > self.iterations:
            raise ValueError(
                "minimum_iterations cannot exceed iterations"
            )

        finite_values = {
            "tolerance": self.tolerance,
            "correction_tolerance": (
                self.correction_tolerance
            ),
            "relaxation": self.relaxation,
            "stagnation_tolerance": (
                self.stagnation_tolerance
            ),
            "divergence_ratio": (
                self.divergence_ratio
            ),
        }

        for name, value in finite_values.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        if self.tolerance < 0.0:
            raise ValueError(
                "tolerance cannot be negative"
            )

        if self.correction_tolerance < 0.0:
            raise ValueError(
                "correction_tolerance cannot be negative"
            )

        if self.relaxation <= 0.0:
            raise ValueError(
                "relaxation must be positive"
            )

        if self.stagnation_window < 2:
            raise ValueError(
                "stagnation_window must be at least 2"
            )

        if self.stagnation_tolerance < 0.0:
            raise ValueError(
                "stagnation_tolerance cannot be negative"
            )

        if self.divergence_ratio <= 1.0:
            raise ValueError(
                "divergence_ratio must be greater than 1"
            )

    @staticmethod
    def _validate_dt(dt: float) -> float:
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
    def _prepare_constraints(
        constraints: Iterable[Constraint],
    ) -> list[Constraint]:
        prepared = list(constraints)

        for index, constraint in enumerate(prepared):
            if not isinstance(
                constraint,
                Constraint,
            ):
                raise TypeError(
                    "all constraints must inherit from "
                    f"Constraint; item {index} has type "
                    f"{type(constraint).__name__}"
                )

        return prepared

    # =========================================================
    # Constraint state helpers
    # =========================================================

    @staticmethod
    def _is_enabled(
        constraint: Constraint,
    ) -> bool:
        return bool(
            getattr(
                constraint,
                "enabled",
                True,
            )
        )

    @staticmethod
    def _is_active(
        constraint: Constraint,
    ) -> bool:
        state = getattr(
            constraint,
            "state",
            None,
        )

        if state is None:
            return True

        return bool(
            getattr(
                state,
                "active",
                True,
            )
        )

    @staticmethod
    def _evaluate_constraint(
        constraint: Constraint,
    ) -> float:
        violation = float(
            constraint.evaluate()
        )

        return violation

    @staticmethod
    def _state_correction_magnitude(
        constraint: Constraint,
    ) -> float:
        state = getattr(
            constraint,
            "state",
            None,
        )

        if state is None:
            return 0.0

        correction = getattr(
            state,
            "correction",
            0.0,
        )

        correction_array = np.asarray(
            correction,
            dtype=float,
        )

        if correction_array.size == 0:
            return 0.0

        return float(
            np.linalg.norm(
                correction_array
            )
        )

    @staticmethod
    def _reset_constraint_correction(
        constraint: Constraint,
    ) -> None:
        state = getattr(
            constraint,
            "state",
            None,
        )

        if state is None:
            return

        if hasattr(state, "correction"):
            correction = getattr(
                state,
                "correction",
            )

            if isinstance(
                correction,
                np.ndarray,
            ):
                state.correction = np.zeros_like(
                    correction,
                    dtype=float,
                )
            else:
                state.correction = 0.0

    # =========================================================
    # Constraint lifecycle
    # =========================================================

    def begin_step(
        self,
        constraints: list[Constraint],
        dt: float,
    ) -> None:
        """
        Initialize constraints once before solver iterations.
        """

        for constraint in constraints:
            if not self._is_enabled(constraint):
                continue

            constraint.begin_step(
                dt=dt
            )

    def end_step(
        self,
        constraints: list[Constraint],
        dt: float,
    ) -> None:
        """
        Finalize constraints once after all iterations.
        """

        for constraint in constraints:
            if not self._is_enabled(constraint):
                continue

            constraint.end_step(
                dt=dt
            )

    def apply_velocity_damping(
        self,
        constraints: list[Constraint],
        dt: float,
    ) -> None:
        """
        Apply constraint velocity damping once per physical step.
        """

        if not self.enable_velocity_damping:
            return

        for constraint in constraints:
            if not self._is_enabled(constraint):
                continue

            if not self._is_active(constraint):
                continue

            constraint.damp_velocity(
                dt=dt
            )

    # =========================================================
    # Diagnostics
    # =========================================================

    def _validate_finite(
        self,
        value: float,
        *,
        constraint: Constraint,
        quantity: str,
    ) -> None:
        if np.isfinite(value):
            return

        if not self.fail_on_nonfinite:
            return

        constraint_name = getattr(
            constraint,
            "name",
            type(constraint).__name__,
        )

        raise FloatingPointError(
            f"Constraint {constraint_name!r} generated "
            f"a non-finite {quantity}: {value}"
        )

    def evaluate_violations(
        self,
        constraints: list[Constraint],
    ) -> np.ndarray:
        """
        Evaluate absolute violations of all enabled and active
        constraints.
        """

        violations: list[float] = []

        for constraint in constraints:
            if not self._is_enabled(constraint):
                continue

            violation = self._evaluate_constraint(
                constraint
            )

            self._validate_finite(
                violation,
                constraint=constraint,
                quantity="violation",
            )

            if not self._is_active(constraint):
                continue

            violations.append(
                abs(violation)
            )

        return np.asarray(
            violations,
            dtype=float,
        )

    @staticmethod
    def _aggregate_values(
        values: list[float] | np.ndarray,
    ) -> tuple[float, float, float]:
        array = np.asarray(
            values,
            dtype=float,
        )

        if array.size == 0:
            return 0.0, 0.0, 0.0

        maximum = float(
            np.max(array)
        )

        mean = float(
            np.mean(array)
        )

        rms = float(
            np.sqrt(
                np.mean(array ** 2)
            )
        )

        return maximum, mean, rms

    def _iteration_statistics(
        self,
        *,
        iteration: int,
        violations: list[float],
        corrections: list[float],
        active_constraints: int,
        projected_constraints: int,
    ) -> SolverIterationStats:
        (
            maximum_violation,
            mean_violation,
            rms_violation,
        ) = self._aggregate_values(
            violations
        )

        (
            maximum_correction,
            mean_correction,
            _,
        ) = self._aggregate_values(
            corrections
        )

        converged = (
            maximum_violation <= self.tolerance
            and maximum_correction
            <= self.correction_tolerance
        )

        return SolverIterationStats(
            iteration=iteration,
            active_constraints=active_constraints,
            projected_constraints=(
                projected_constraints
            ),
            maximum_violation=maximum_violation,
            mean_violation=mean_violation,
            rms_violation=rms_violation,
            maximum_correction=maximum_correction,
            mean_correction=mean_correction,
            converged=converged,
        )

    # =========================================================
    # Projection
    # =========================================================

    def project_iteration(
        self,
        constraints: list[Constraint],
        dt: float,
        iteration: int,
    ) -> SolverIterationStats:
        """
        Execute one Gauss-Seidel XPBD projection iteration.

        Constraints are projected sequentially. Each correction
        is immediately visible to subsequent constraints.
        """

        violations: list[float] = []
        corrections: list[float] = []

        active_constraints = 0
        projected_constraints = 0

        for constraint in constraints:
            if not self._is_enabled(constraint):
                continue

            violation_before = (
                self._evaluate_constraint(
                    constraint
                )
            )

            self._validate_finite(
                violation_before,
                constraint=constraint,
                quantity="violation",
            )

            if not self._is_active(constraint):
                continue

            active_constraints += 1

            self._reset_constraint_correction(
                constraint
            )

            correction = constraint.project(
                dt=dt,
                relaxation=self.relaxation,
            )

            projected_constraints += 1

            if correction is None:
                correction_magnitude = (
                    self._state_correction_magnitude(
                        constraint
                    )
                )
            else:
                correction_array = np.asarray(
                    correction,
                    dtype=float,
                )

                correction_magnitude = float(
                    np.linalg.norm(
                        correction_array
                    )
                )

            self._validate_finite(
                correction_magnitude,
                constraint=constraint,
                quantity="correction",
            )

            violation_after = (
                self._evaluate_constraint(
                    constraint
                )
            )

            self._validate_finite(
                violation_after,
                constraint=constraint,
                quantity="post-projection violation",
            )

            violations.append(
                abs(violation_after)
            )

            corrections.append(
                abs(correction_magnitude)
            )

        return self._iteration_statistics(
            iteration=iteration,
            violations=violations,
            corrections=corrections,
            active_constraints=active_constraints,
            projected_constraints=(
                projected_constraints
            ),
        )

    # =========================================================
    # Stagnation and divergence
    # =========================================================

    def _has_stagnated(
        self,
        violation_history: list[float],
    ) -> bool:
        if not self.detect_stagnation:
            return False

        if len(violation_history) < (
            self.stagnation_window
        ):
            return False

        recent = violation_history[
            -self.stagnation_window:
        ]

        start = recent[0]
        end = recent[-1]

        if start <= self.tolerance:
            return False

        relative_improvement = (
            start - end
        ) / max(
            start,
            1e-12,
        )

        return bool(
            relative_improvement
            < self.stagnation_tolerance
        )

    def _has_diverged(
        self,
        *,
        initial_violation: float,
        current_violation: float,
    ) -> bool:
        if initial_violation <= self.tolerance:
            return False

        return bool(
            current_violation
            > initial_violation
            * self.divergence_ratio
        )

    # =========================================================
    # Main solve
    # =========================================================

    def solve(
        self,
        constraints: Iterable[Constraint],
        dt: float,
    ) -> SolverStatistics:
        """
        Solve all constraints for one physical time step.

        Parameters
        ----------
        constraints:
            Iterable of Constraint objects.

        dt:
            Physical simulation time step.

        Returns
        -------
        SolverStatistics
            Diagnostics consumed directly by Network.
        """

        dt = self._validate_dt(dt)

        prepared_constraints = (
            self._prepare_constraints(
                constraints
            )
        )

        enabled_constraints = [
            constraint
            for constraint in prepared_constraints
            if self._is_enabled(constraint)
        ]

        if not enabled_constraints:
            statistics = SolverStatistics(
                iterations=0,
                converged=True,
            )

            self.last_statistics = statistics
            return statistics

        self.begin_step(
            enabled_constraints,
            dt,
        )

        initial_violations = (
            self.evaluate_violations(
                enabled_constraints
            )
        )

        (
            initial_maximum,
            _,
            _,
        ) = self._aggregate_values(
            initial_violations
        )

        iteration_history: list[
            SolverIterationStats
        ] = []

        violation_history: list[float] = []

        converged = False
        stagnated = False
        divergence_detected = False

        final_iteration = SolverIterationStats()

        try:
            for iteration in range(
                1,
                self.iterations + 1,
            ):
                iteration_stats = (
                    self.project_iteration(
                        constraints=enabled_constraints,
                        dt=dt,
                        iteration=iteration,
                    )
                )

                final_iteration = iteration_stats

                violation_history.append(
                    iteration_stats.maximum_violation
                )

                if self.record_iteration_history:
                    iteration_history.append(
                        iteration_stats
                    )

                if (
                    iteration
                    >= self.minimum_iterations
                    and iteration_stats.converged
                ):
                    converged = True
                    break

                divergence_detected = (
                    self._has_diverged(
                        initial_violation=(
                            initial_maximum
                        ),
                        current_violation=(
                            iteration_stats
                            .maximum_violation
                        ),
                    )
                )

                if divergence_detected:
                    break

                stagnated = self._has_stagnated(
                    violation_history
                )

                if (
                    stagnated
                    and iteration
                    >= self.minimum_iterations
                ):
                    break

            self.apply_velocity_damping(
                enabled_constraints,
                dt,
            )

        finally:
            self.end_step(
                enabled_constraints,
                dt,
            )

        statistics = SolverStatistics(
            iterations=int(
                final_iteration.iteration
            ),
            converged=converged,
            active_constraints=int(
                final_iteration.active_constraints
            ),
            projected_constraints=int(
                final_iteration
                .projected_constraints
            ),
            initial_maximum_violation=float(
                initial_maximum
            ),
            maximum_violation=float(
                final_iteration.maximum_violation
            ),
            mean_violation=float(
                final_iteration.mean_violation
            ),
            rms_violation=float(
                final_iteration.rms_violation
            ),
            maximum_correction=float(
                final_iteration.maximum_correction
            ),
            mean_correction=float(
                final_iteration.mean_correction
            ),
            stagnated=stagnated,
            divergence_detected=(
                divergence_detected
            ),
            iteration_history=iteration_history,
        )

        self.last_statistics = statistics

        return statistics

    # =========================================================
    # Runtime control
    # =========================================================

    def set_iterations(
        self,
        iterations: int,
    ) -> None:
        previous = self.iterations
        self.iterations = int(iterations)

        try:
            self._validate_configuration()
        except ValueError:
            self.iterations = previous
            raise

    def set_tolerance(
        self,
        tolerance: float,
    ) -> None:
        previous = self.tolerance
        self.tolerance = float(tolerance)

        try:
            self._validate_configuration()
        except ValueError:
            self.tolerance = previous
            raise

    def set_relaxation(
        self,
        relaxation: float,
    ) -> None:
        previous = self.relaxation
        self.relaxation = float(relaxation)

        try:
            self._validate_configuration()
        except ValueError:
            self.relaxation = previous
            raise

    def reset_statistics(self) -> None:
        self.last_statistics = SolverStatistics()

    # =========================================================
    # Representation
    # =========================================================

    def __repr__(self) -> str:
        return (
            f"Solver("
            f"iterations={self.iterations}, "
            f"tolerance={self.tolerance:.3e}, "
            f"correction_tolerance="
            f"{self.correction_tolerance:.3e}, "
            f"relaxation={self.relaxation:.3f}, "
            f"last_iterations="
            f"{self.last_statistics.iterations}, "
            f"last_converged="
            f"{self.last_statistics.converged})"
        )