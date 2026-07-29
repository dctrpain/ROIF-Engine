from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .material import (
    Material,
    MaterialParameters,
    MaterialState,
)


@dataclass
class MuscleParameters(MaterialParameters):
    """
    Mechanical and physiological parameters of an axial muscle.

    Contractile parameters
    ----------------------
    maximum_isometric_force:
        Maximum active force at full activation, optimal length
        and zero contraction velocity.

    optimal_length:
        Length at which the active force-length relation reaches
        its maximum.

    tendon_slack_length:
        Reserved for future musculotendon decomposition. The
        current axial muscle law operates on total element length.

    active_length_width:
        Width of the active force-length curve, normalized by
        optimal length.

    maximum_shortening_velocity:
        Maximum shortening velocity in optimal lengths per second.

    eccentric_force_multiplier:
        Maximum increase of active force during lengthening.

    hill_curvature:
        Controls the curvature of the shortening force-velocity
        relation.

    Passive parameters
    ------------------
    passive_slack_ratio:
        Passive muscle force starts when current length exceeds

            optimal_length * passive_slack_ratio

    passive_stiffness:
        Scale of the exponential passive force.

    passive_exponent:
        Curvature of the exponential passive response.

    Activation parameters
    ---------------------
    activation_time_constant:
        Time constant when excitation exceeds activation.

    deactivation_time_constant:
        Time constant when excitation is below activation.

    minimum_activation:
        Baseline activation or tone.

    activation_energy_rate:
        Additional energy consumption caused by activation.

    contraction_energy_rate:
        Additional energy consumption caused by active force and
        shortening velocity.

    Fatigue parameters
    ------------------
    active_fatigue_multiplier:
        Scales fatigue accumulation from activation.

    overload_fatigue_multiplier:
        Scales fatigue accumulation from normalized mechanical
        overload.
    """

    maximum_isometric_force: float = 1.0

    optimal_length: float = 1.0
    tendon_slack_length: float = 0.0

    active_length_width: float = 0.45

    maximum_shortening_velocity: float = 10.0
    eccentric_force_multiplier: float = 1.5
    hill_curvature: float = 0.25

    passive_slack_ratio: float = 1.0
    passive_stiffness: float = 0.05
    passive_exponent: float = 5.0

    activation_time_constant: float = 0.015
    deactivation_time_constant: float = 0.050
    minimum_activation: float = 0.0

    activation_energy_rate: float = 0.05
    contraction_energy_rate: float = 0.05

    active_fatigue_multiplier: float = 1.0
    overload_fatigue_multiplier: float = 1.0

    def validate(self) -> None:
        super().validate()

        finite_fields = (
            "maximum_isometric_force",
            "optimal_length",
            "tendon_slack_length",
            "active_length_width",
            "maximum_shortening_velocity",
            "eccentric_force_multiplier",
            "hill_curvature",
            "passive_slack_ratio",
            "passive_stiffness",
            "passive_exponent",
            "activation_time_constant",
            "deactivation_time_constant",
            "minimum_activation",
            "activation_energy_rate",
            "contraction_energy_rate",
            "active_fatigue_multiplier",
            "overload_fatigue_multiplier",
        )

        for name in finite_fields:
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise ValueError(
                    f"MuscleParameters.{name} must be finite"
                )

            setattr(
                self,
                name,
                value,
            )

        nonnegative_fields = (
            "maximum_isometric_force",
            "optimal_length",
            "tendon_slack_length",
            "active_length_width",
            "maximum_shortening_velocity",
            "eccentric_force_multiplier",
            "hill_curvature",
            "passive_slack_ratio",
            "passive_stiffness",
            "passive_exponent",
            "activation_time_constant",
            "deactivation_time_constant",
            "minimum_activation",
            "activation_energy_rate",
            "contraction_energy_rate",
            "active_fatigue_multiplier",
            "overload_fatigue_multiplier",
        )

        for name in nonnegative_fields:
            if getattr(self, name) < 0.0:
                raise ValueError(
                    f"MuscleParameters.{name} cannot be negative"
                )

        positive_fields = (
            "maximum_isometric_force",
            "optimal_length",
            "active_length_width",
            "maximum_shortening_velocity",
            "eccentric_force_multiplier",
            "hill_curvature",
            "passive_exponent",
            "activation_time_constant",
            "deactivation_time_constant",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"MuscleParameters.{name} must be positive"
                )

        if self.eccentric_force_multiplier < 1.0:
            raise ValueError(
                "eccentric_force_multiplier cannot be smaller "
                "than 1"
            )

        if self.minimum_activation > 1.0:
            raise ValueError(
                "minimum_activation cannot exceed 1"
            )

    def snapshot(self) -> dict[str, float]:
        self.validate()

        return {
            name: float(value)
            for name, value in asdict(self).items()
        }


@dataclass
class MuscleState(MaterialState):
    """
    Mutable muscle-specific state.

    excitation:
        Neural input in [0, 1].

    activation:
        Contractile activation inherited from MaterialState.

    neural_drive:
        Optional filtered or externally supplied control signal.

    normalized_length:
        Current length divided by optimal length.

    normalized_velocity:
        Current length velocity divided by maximum shortening
        velocity.

    active_length_factor:
        Current force-length multiplier.

    force_velocity_factor:
        Current force-velocity multiplier.

    metabolic_power:
        Current normalized rate of energy consumption.

    active_work_rate:
        Mechanical active power estimate.
    """

    excitation: float = 0.0
    neural_drive: float = 0.0

    normalized_length: float = 1.0
    normalized_velocity: float = 0.0

    active_length_factor: float = 1.0
    force_velocity_factor: float = 1.0

    metabolic_power: float = 0.0
    active_work_rate: float = 0.0

    activation_update_count: int = 0

    def clamp(self) -> None:
        super().clamp()

        normalized_fields = (
            "excitation",
            "neural_drive",
        )

        for name in normalized_fields:
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"MuscleState.{name} must be finite"
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

        finite_fields = (
            "normalized_length",
            "normalized_velocity",
            "active_length_factor",
            "force_velocity_factor",
            "metabolic_power",
            "active_work_rate",
        )

        for name in finite_fields:
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"MuscleState.{name} must be finite"
                )

            setattr(
                self,
                name,
                value,
            )

        self.normalized_length = max(
            0.0,
            self.normalized_length,
        )

        self.active_length_factor = max(
            0.0,
            self.active_length_factor,
        )

        self.force_velocity_factor = max(
            0.0,
            self.force_velocity_factor,
        )

        self.metabolic_power = max(
            0.0,
            self.metabolic_power,
        )

        self.activation_update_count = max(
            0,
            int(self.activation_update_count),
        )

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()

        snapshot.update(
            {
                "excitation": float(
                    self.excitation
                ),
                "neural_drive": float(
                    self.neural_drive
                ),
                "normalized_length": float(
                    self.normalized_length
                ),
                "normalized_velocity": float(
                    self.normalized_velocity
                ),
                "active_length_factor": float(
                    self.active_length_factor
                ),
                "force_velocity_factor": float(
                    self.force_velocity_factor
                ),
                "metabolic_power": float(
                    self.metabolic_power
                ),
                "active_work_rate": float(
                    self.active_work_rate
                ),
                "activation_update_count": int(
                    self.activation_update_count
                ),
            }
        )

        return snapshot


@dataclass
class MuscleMaterial(Material):
    """
    Active axial muscle material.

    Lifecycle
    ---------
    begin_step():
        Evolves excitation-to-activation dynamics exactly once
        before mechanical force evaluation.

    force():
        Evaluates active, passive, damping and pretension force
        without changing biological state.

    end_step():
        Evolves fatigue, damage, recovery, remodeling and energy
        exactly once after mechanics.

    Sign convention
    ---------------
    Positive force:
        tension.

    Positive length velocity:
        muscle lengthening.

    Negative length velocity:
        muscle shortening.
    """

    name: str = "muscle"

    parameters: MuscleParameters = field(
        default_factory=MuscleParameters
    )

    state: MuscleState = field(
        default_factory=MuscleState
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(
            self.parameters,
            MuscleParameters,
        ):
            raise TypeError(
                "MuscleMaterial.parameters must be "
                "MuscleParameters"
            )

        if not isinstance(
            self.state,
            MuscleState,
        ):
            raise TypeError(
                "MuscleMaterial.state must be MuscleState"
            )

        if self.reference_length is None:
            self.reference_length = float(
                self.parameters.optimal_length
            )

        self.state.activation = max(
            self.state.activation,
            self.parameters.minimum_activation,
        )

        self.state.clamp()
        self.validate()

    # =========================================================
    # Validation
    # =========================================================

    def validate(self) -> None:
        super().validate()

        if not isinstance(
            self.parameters,
            MuscleParameters,
        ):
            raise TypeError(
                "parameters must be MuscleParameters"
            )

        if not isinstance(
            self.state,
            MuscleState,
        ):
            raise TypeError(
                "state must be MuscleState"
            )

        self.parameters.validate()
        self.state.clamp()

    # =========================================================
    # Neural control
    # =========================================================

    def set_excitation(
        self,
        excitation: float,
    ) -> None:
        excitation = self._validate_scalar(
            excitation,
            name="excitation",
        )

        self.state.excitation = float(
            np.clip(
                excitation,
                0.0,
                1.0,
            )
        )

    def set_neural_drive(
        self,
        neural_drive: float,
        *,
        update_excitation: bool = True,
    ) -> None:
        neural_drive = self._validate_scalar(
            neural_drive,
            name="neural_drive",
        )

        self.state.neural_drive = float(
            np.clip(
                neural_drive,
                0.0,
                1.0,
            )
        )

        if update_excitation:
            self.state.excitation = (
                self.state.neural_drive
            )

    def activation_target(self) -> float:
        return float(
            np.clip(
                max(
                    self.state.excitation,
                    self.parameters.minimum_activation,
                ),
                0.0,
                1.0,
            )
        )

    def activation_time_constant(
        self,
        target: float | None = None,
    ) -> float:
        if target is None:
            target = self.activation_target()

        if target >= self.state.activation:
            return float(
                self.parameters.activation_time_constant
            )

        return float(
            self.parameters.deactivation_time_constant
        )

    def update_activation(
        self,
        dt: float,
    ) -> None:
        """
        Exact first-order excitation-to-activation update.

            da/dt = (u - a) / tau

        Exact discrete solution:

            a_next = u + (a - u) exp(-dt / tau)
        """

        dt = self._validate_dt(dt)

        target = self.activation_target()

        time_constant = (
            self.activation_time_constant(
                target
            )
        )

        decay = float(
            np.exp(
                -dt
                / max(
                    time_constant,
                    self.epsilon,
                )
            )
        )

        self.state.activation = (
            target
            + (
                self.state.activation
                - target
            )
            * decay
        )

        self.state.activation = float(
            np.clip(
                self.state.activation,
                self.parameters.minimum_activation,
                1.0,
            )
        )

        self.state.activation_update_count += 1
        self.state.clamp()

    # =========================================================
    # Step lifecycle
    # =========================================================

    def begin_step(
        self,
        dt: float,
        current_length: float | None = None,
        length_velocity: float = 0.0,
    ) -> None:
        """
        Update muscle activation once before force assembly.
        """

        super().begin_step(
            dt=dt,
            current_length=current_length,
            length_velocity=length_velocity,
        )

        self.update_activation(dt)

        if current_length is not None:
            self._store_normalized_kinematics(
                current_length=current_length,
                length_velocity=length_velocity,
            )

    def end_step(
        self,
        dt: float,
        current_length: float,
        length_velocity: float = 0.0,
        force: float | None = None,
        reference_length: float | None = None,
    ) -> None:
        """
        Update post-mechanical muscle biology exactly once.
        """

        self._store_normalized_kinematics(
            current_length=current_length,
            length_velocity=length_velocity,
        )

        if force is None:
            force = self.force(
                current_length=current_length,
                length_velocity=length_velocity,
                reference_length=reference_length,
                store_state=True,
            )

        self._update_metabolic_diagnostics(
            current_length=current_length,
            length_velocity=length_velocity,
        )

        super().end_step(
            dt=dt,
            current_length=current_length,
            length_velocity=length_velocity,
            force=force,
            reference_length=reference_length,
        )

    # =========================================================
    # Muscle kinematics
    # =========================================================

    def normalized_length(
        self,
        current_length: float,
    ) -> float:
        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        return float(
            current_length
            / max(
                self.parameters.optimal_length,
                self.epsilon,
            )
        )

    def normalized_velocity(
        self,
        length_velocity: float,
    ) -> float:
        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        velocity_scale = (
            self.parameters.maximum_shortening_velocity
            * self.parameters.optimal_length
        )

        return float(
            length_velocity
            / max(
                velocity_scale,
                self.epsilon,
            )
        )

    def _store_normalized_kinematics(
        self,
        *,
        current_length: float,
        length_velocity: float,
    ) -> None:
        self.state.normalized_length = (
            self.normalized_length(
                current_length
            )
        )

        self.state.normalized_velocity = (
            self.normalized_velocity(
                length_velocity
            )
        )

        self.state.active_length_factor = (
            self.active_force_length_factor(
                current_length
            )
        )

        self.state.force_velocity_factor = (
            self.force_velocity_factor(
                length_velocity
            )
        )

        self.state.clamp()

    # =========================================================
    # Active force-length relation
    # =========================================================

    def active_force_length_factor(
        self,
        current_length: float,
    ) -> float:
        """
        Gaussian approximation of the active force-length curve.

        Maximum force occurs at optimal_length.
        """

        normalized_length = (
            self.normalized_length(
                current_length
            )
        )

        width = max(
            self.parameters.active_length_width,
            self.epsilon,
        )

        deviation = (
            normalized_length
            - 1.0
        ) / width

        return float(
            np.exp(
                -(deviation * deviation)
            )
        )

    # =========================================================
    # Force-velocity relation
    # =========================================================

    def force_velocity_factor(
        self,
        length_velocity: float,
    ) -> float:
        """
        Hill-type force-velocity approximation.

        Shortening:
            length_velocity < 0
            active force decreases with shortening speed.

        Lengthening:
            length_velocity > 0
            active force increases toward
            eccentric_force_multiplier.
        """

        normalized_velocity = (
            self.normalized_velocity(
                length_velocity
            )
        )

        curvature = max(
            self.parameters.hill_curvature,
            self.epsilon,
        )

        if normalized_velocity < 0.0:
            shortening_speed = min(
                1.0,
                -normalized_velocity,
            )

            numerator = (
                1.0
                - shortening_speed
            )

            denominator = (
                1.0
                + shortening_speed
                / curvature
            )

            factor = (
                numerator
                / max(
                    denominator,
                    self.epsilon,
                )
            )

            return float(
                np.clip(
                    factor,
                    0.0,
                    1.0,
                )
            )

        eccentric_limit = (
            self.parameters.eccentric_force_multiplier
        )

        factor = (
            1.0
            + (
                eccentric_limit
                - 1.0
            )
            * (
                normalized_velocity
                / (
                    normalized_velocity
                    + curvature
                )
            )
        )

        return float(
            np.clip(
                factor,
                1.0,
                eccentric_limit,
            )
        )

    # =========================================================
    # Passive muscle relation
    # =========================================================

    def passive_slack_length(self) -> float:
        return float(
            self.parameters.optimal_length
            * self.parameters.passive_slack_ratio
        )

    def passive_muscle_force(
        self,
        current_length: float,
        length_velocity: float = 0.0,
    ) -> float:
        """
        Exponential passive tension above slack length.

        No passive compression is generated.
        """

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        slack_length = (
            self.passive_slack_length()
        )

        if current_length <= slack_length:
            return 0.0

        normalized_extension = (
            current_length
            - slack_length
        ) / max(
            self.parameters.optimal_length,
            self.epsilon,
        )

        exponent_argument = float(
            np.clip(
                self.parameters.passive_exponent
                * normalized_extension,
                0.0,
                50.0,
            )
        )

        normalized_force = (
            np.exp(
                exponent_argument
            )
            - 1.0
        )

        force = (
            self.parameters.maximum_isometric_force
            * self.parameters.passive_stiffness
            * normalized_force
            * self.integrity()
        )

        return float(
            max(
                0.0,
                force,
            )
        )

    def passive_force(
        self,
        extension: float,
    ) -> float:
        """
        Compatibility implementation used by Material.force().

        The extension is interpreted relative to the effective
        reference length.
        """

        extension = self._validate_scalar(
            extension,
            name="extension",
        )

        reference_length = (
            self.effective_reference_length(
                self.parameters.optimal_length
            )
        )

        current_length = max(
            0.0,
            reference_length
            + extension,
        )

        return self.passive_muscle_force(
            current_length
        )

    # =========================================================
    # Active muscle force
    # =========================================================

    def active_capacity(self) -> float:
        """
        Current maximum active capacity after structural,
        fatigue and energetic limitations.
        """

        if self.state.failed:
            return 0.0

        structural_capacity = (
            1.0
            - self.state.damage
        )

        fatigue_capacity = (
            1.0
            - self.state.fatigue
        )

        energy_capacity = float(
            np.clip(
                self.state.energy,
                0.0,
                1.0,
            )
        )

        remodeling_capacity = float(
            np.clip(
                self.state.remodeling,
                0.0,
                1.0,
            )
        )

        capacity = (
            self.parameters.maximum_isometric_force
            * structural_capacity
            * fatigue_capacity
            * energy_capacity
            * remodeling_capacity
        )

        return float(
            max(
                0.0,
                capacity,
            )
        )

    def active_force(
        self,
        current_length: float | None = None,
        length_velocity: float = 0.0,
    ) -> float:
        """
        Active contractile tension.

            F_active =
                activation
                * active_capacity
                * force_length_factor
                * force_velocity_factor
        """

        if self.state.failed:
            return 0.0

        if current_length is None:
            current_length = (
                self.state.current_length
            )

            if current_length <= self.epsilon:
                current_length = (
                    self.parameters.optimal_length
                )

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        length_factor = (
            self.active_force_length_factor(
                current_length
            )
        )

        velocity_factor = (
            self.force_velocity_factor(
                length_velocity
            )
        )

        activation = float(
            np.clip(
                self.state.activation,
                0.0,
                1.0,
            )
        )

        active_force = (
            activation
            * self.active_capacity()
            * length_factor
            * velocity_factor
        )

        return float(
            max(
                0.0,
                active_force,
            )
        )

    def active_muscle_force(
        self,
        current_length: float,
        length_velocity: float = 0.0,
    ) -> float:
        """
        Explicit compatibility alias used by Element.
        """

        return self.active_force(
            current_length=current_length,
            length_velocity=length_velocity,
        )

    # =========================================================
    # Total muscle force
    # =========================================================

    def muscle_force(
        self,
        current_length: float,
        length_velocity: float = 0.0,
        *,
        include_active: bool = True,
        store_state: bool = True,
    ) -> float:
        """
        Evaluate complete muscle tension.

        Components:

        - passive muscle tension;
        - viscous tension;
        - pretension;
        - active contractile tension.

        Muscle does not generate axial compression. The final
        force is therefore clamped to zero.
        """

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        passive = self.passive_muscle_force(
            current_length,
            length_velocity,
        )

        damping = self.damping_force(
            length_velocity
        )

        pretension = self.pretension_force()

        active = (
            self.active_force(
                current_length,
                length_velocity,
            )
            if include_active
            else 0.0
        )

        total = max(
            0.0,
            passive
            + damping
            + pretension
            + active,
        )

        if self.state.failed:
            passive = 0.0
            damping = 0.0
            pretension = 0.0
            active = 0.0
            total = 0.0

        if store_state:
            reference_length = (
                self.effective_reference_length(
                    self.parameters.optimal_length
                )
            )

            extension = (
                current_length
                - reference_length
            )

            strain = (
                extension
                / max(
                    reference_length,
                    self.epsilon,
                )
            )

            self.state.current_length = (
                current_length
            )

            self.state.length_velocity = (
                length_velocity
            )

            self.state.extension = extension
            self.state.strain = strain

            self.state.passive_force_component = (
                passive
            )

            self.state.damping_force_component = (
                damping
            )

            self.state.pretension_force_component = (
                pretension
            )

            self.state.active_force_component = (
                active
            )

            self.state.total_force_component = (
                total
            )

            self._store_normalized_kinematics(
                current_length=current_length,
                length_velocity=length_velocity,
            )

            self.state.clamp()

        return float(total)

    def force(
        self,
        *,
        current_length: float,
        length_velocity: float = 0.0,
        reference_length: float | None = None,
        include_active: bool = True,
        store_state: bool = True,
    ) -> float:
        """
        Strict Material.force() implementation.

        ``reference_length`` is accepted for interface
        compatibility. The muscle-specific active and passive
        curves use optimal_length and passive_slack_ratio.
        """

        if reference_length is not None:
            self._validate_length(
                reference_length,
                name="reference_length",
            )

        return self.muscle_force(
            current_length=current_length,
            length_velocity=length_velocity,
            include_active=include_active,
            store_state=store_state,
        )

    # =========================================================
    # Muscle fatigue
    # =========================================================

    def update_fatigue(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        """
        Muscle fatigue depends on:

        - activation;
        - normalized active force;
        - mechanical overload;
        - available energy.
        """

        dt = self._validate_dt(dt)

        stimulus = max(
            0.0,
            self._validate_scalar(
                stimulus,
                name="stimulus",
            ),
        )

        normalized_active_force = (
            abs(
                self.state.active_force_component
            )
            / max(
                self.parameters.maximum_isometric_force,
                self.epsilon,
            )
        )

        activation_load = (
            self.state.activation
            * normalized_active_force
            * self.parameters.active_fatigue_multiplier
        )

        overload = max(
            0.0,
            stimulus
            - self.parameters.overload_threshold,
        )

        overload_load = (
            overload
            * self.parameters.overload_fatigue_multiplier
        )

        total_fatigue_drive = (
            activation_load
            + overload_load
        )

        fatigue_gain = (
            self.parameters.fatigue_rate
            * total_fatigue_drive
            * (
                1.0
                - self.state.fatigue
            )
            * dt
        )

        recovery = (
            self.parameters.recovery_rate
            * self.state.energy
            * (
                1.0
                - self.state.activation
            )
            * self.state.fatigue
            * dt
        )

        self.state.fatigue += (
            fatigue_gain
            - recovery
        )

    # =========================================================
    # Muscle energy
    # =========================================================

    def _update_metabolic_diagnostics(
        self,
        *,
        current_length: float,
        length_velocity: float,
    ) -> None:
        active_force = abs(
            self.active_force(
                current_length=current_length,
                length_velocity=length_velocity,
            )
        )

        normalized_active_force = (
            active_force
            / max(
                self.parameters.maximum_isometric_force,
                self.epsilon,
            )
        )

        normalized_speed = abs(
            self.normalized_velocity(
                length_velocity
            )
        )

        activation_power = (
            self.parameters.activation_energy_rate
            * self.state.activation
        )

        contraction_power = (
            self.parameters.contraction_energy_rate
            * normalized_active_force
            * (
                1.0
                + normalized_speed
            )
        )

        self.state.metabolic_power = max(
            0.0,
            activation_power
            + contraction_power,
        )

        self.state.active_work_rate = float(
            active_force
            * length_velocity
        )

        self.state.clamp()

    def update_energy(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        stimulus = max(
            0.0,
            self._validate_scalar(
                stimulus,
                name="stimulus",
            ),
        )

        baseline_consumption = (
            self.parameters.energy_decay_rate
            * (
                1.0
                + stimulus
            )
        )

        muscle_consumption = (
            self.state.metabolic_power
        )

        fatigue_cost = (
            self.parameters.energy_decay_rate
            * self.state.fatigue
        )

        total_consumption = (
            baseline_consumption
            + muscle_consumption
            + fatigue_cost
        ) * dt

        recovery = (
            self.parameters.recovery_rate
            * (
                1.0
                - self.state.energy
            )
            * (
                1.0
                - self.state.activation
            )
            * dt
        )

        self.state.energy += (
            recovery
            - total_consumption
        )

    # =========================================================
    # Stimulus
    # =========================================================

    def normalized_stimulus(
        self,
        *,
        force: float | None = None,
        strain: float | None = None,
    ) -> float:
        """
        Muscle stimulus combines:

        - total force;
        - strain;
        - active recruitment;
        - energetic limitation.
        """

        base_stimulus = (
            super().normalized_stimulus(
                force=force,
                strain=strain,
            )
        )

        recruitment = (
            self.state.activation
        )

        energy_deficit = (
            1.0
            - self.state.energy
        )

        return float(
            max(
                base_stimulus,
                recruitment,
                energy_deficit,
            )
        )

    # =========================================================
    # Diagnostics
    # =========================================================

    def is_active(self) -> bool:
        return bool(
            self.state.activation
            > self.parameters.minimum_activation
            + self.epsilon
        )

    def is_shortening(self) -> bool:
        return bool(
            self.state.length_velocity
            < -self.epsilon
        )

    def is_lengthening(self) -> bool:
        return bool(
            self.state.length_velocity
            > self.epsilon
        )

    def is_isometric(self) -> bool:
        return bool(
            abs(
                self.state.length_velocity
            )
            <= self.epsilon
        )

    def contractile_efficiency(self) -> float:
        """
        Available active capacity relative to nominal maximum.
        """

        return float(
            np.clip(
                self.active_capacity()
                / max(
                    self.parameters.maximum_isometric_force,
                    self.epsilon,
                ),
                0.0,
                1.0,
            )
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset_state(self) -> None:
        self.state = MuscleState(
            activation=(
                self.parameters.minimum_activation
            ),
            excitation=(
                self.parameters.minimum_activation
            ),
            neural_drive=(
                self.parameters.minimum_activation
            ),
        )

        self.validate()

    # =========================================================
    # Snapshot
    # =========================================================

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()

        current_length = (
            self.state.current_length
        )

        if current_length <= self.epsilon:
            current_length = (
                self.parameters.optimal_length
            )

        snapshot.update(
            {
                "muscle": {
                    "excitation": float(
                        self.state.excitation
                    ),
                    "activation": float(
                        self.state.activation
                    ),
                    "active_capacity": (
                        self.active_capacity()
                    ),
                    "contractile_efficiency": (
                        self.contractile_efficiency()
                    ),
                    "active_force_length_factor": (
                        self.active_force_length_factor(
                            current_length
                        )
                    ),
                    "force_velocity_factor": (
                        self.force_velocity_factor(
                            self.state.length_velocity
                        )
                    ),
                    "passive_slack_length": (
                        self.passive_slack_length()
                    ),
                    "active": self.is_active(),
                    "shortening": (
                        self.is_shortening()
                    ),
                    "lengthening": (
                        self.is_lengthening()
                    ),
                    "isometric": (
                        self.is_isometric()
                    ),
                    "metabolic_power": float(
                        self.state.metabolic_power
                    ),
                    "active_work_rate": float(
                        self.state.active_work_rate
                    ),
                }
            }
        )

        return snapshot

    def __repr__(self) -> str:
        return (
            f"MuscleMaterial("
            f"name={self.name!r}, "
            f"excitation={self.state.excitation:.3f}, "
            f"activation={self.state.activation:.3f}, "
            f"active_capacity={self.active_capacity():.6g}, "
            f"fatigue={self.state.fatigue:.3f}, "
            f"damage={self.state.damage:.3f}, "
            f"energy={self.state.energy:.3f}, "
            f"failed={self.state.failed})"
        )