from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class MaterialParameters:
    """
    Immutable-style constitutive and biological coefficients.

    Mechanical parameters
    ---------------------
    stiffness:
        Base axial stiffness.

    damping:
        Base viscous coefficient.

    density:
        Material density for future mass-distribution models.

    Biological parameters
    ---------------------
    recovery_rate:
        Damage and fatigue recovery rate.

    fatigue_rate:
        Rate of fatigue accumulation under sustained loading.

    remodeling_rate:
        Rate of adaptation toward the current mechanical demand.

    production_rate:
        Rate of matrix or tissue production.

    damage_rate:
        Rate of structural damage accumulation.

    pretension_rate:
        Rate at which pretension adapts.

    energy_decay_rate:
        Baseline energy-consumption rate.
    """

    stiffness: float = 1.0
    damping: float = 0.0
    density: float = 1.0

    recovery_rate: float = 0.0
    fatigue_rate: float = 0.0
    remodeling_rate: float = 0.0
    production_rate: float = 0.0
    damage_rate: float = 0.0
    pretension_rate: float = 0.0
    energy_decay_rate: float = 0.0

    minimum_integrity: float = 0.0
    maximum_integrity: float = 1.0

    overload_threshold: float = 1.0
    failure_threshold: float = 1.0

    reference_force: float = 1.0
    reference_strain: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        finite_fields = (
            "stiffness",
            "damping",
            "density",
            "recovery_rate",
            "fatigue_rate",
            "remodeling_rate",
            "production_rate",
            "damage_rate",
            "pretension_rate",
            "energy_decay_rate",
            "minimum_integrity",
            "maximum_integrity",
            "overload_threshold",
            "failure_threshold",
            "reference_force",
            "reference_strain",
        )

        for name in finite_fields:
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise ValueError(
                    f"MaterialParameters.{name} must be finite"
                )

            setattr(
                self,
                name,
                value,
            )

        nonnegative_fields = (
            "stiffness",
            "damping",
            "density",
            "recovery_rate",
            "fatigue_rate",
            "remodeling_rate",
            "production_rate",
            "damage_rate",
            "pretension_rate",
            "energy_decay_rate",
            "minimum_integrity",
            "maximum_integrity",
            "overload_threshold",
            "failure_threshold",
            "reference_force",
            "reference_strain",
        )

        for name in nonnegative_fields:
            if getattr(self, name) < 0.0:
                raise ValueError(
                    f"MaterialParameters.{name} "
                    "cannot be negative"
                )

        if self.maximum_integrity < self.minimum_integrity:
            raise ValueError(
                "maximum_integrity cannot be smaller than "
                "minimum_integrity"
            )

        if self.maximum_integrity > 1.0:
            raise ValueError(
                "maximum_integrity cannot exceed 1"
            )

        if self.failure_threshold <= 0.0:
            raise ValueError(
                "failure_threshold must be positive"
            )

        if self.reference_force <= 0.0:
            raise ValueError(
                "reference_force must be positive"
            )

        if self.reference_strain <= 0.0:
            raise ValueError(
                "reference_strain must be positive"
            )

    def snapshot(self) -> dict[str, float]:
        self.validate()

        return {
            name: float(value)
            for name, value in asdict(self).items()
        }


@dataclass
class MaterialState:
    """
    Mutable biological and mechanical state.

    All normalized biological variables are constrained to [0, 1].

    pretension:
        Signed axial preload.

    history:
        Exponentially filtered loading memory.

    energy:
        Available normalized metabolic reserve.
    """

    damage: float = 0.0
    fatigue: float = 0.0
    remodeling: float = 1.0
    production: float = 0.0
    pretension: float = 0.0
    activation: float = 0.0
    history: float = 0.0
    energy: float = 1.0

    current_length: float = 0.0
    length_velocity: float = 0.0
    extension: float = 0.0
    strain: float = 0.0

    passive_force_component: float = 0.0
    damping_force_component: float = 0.0
    active_force_component: float = 0.0
    pretension_force_component: float = 0.0
    total_force_component: float = 0.0

    normalized_stimulus: float = 0.0
    overloaded: bool = False
    failed: bool = False

    update_count: int = 0

    def clamp(self) -> None:
        normalized_fields = (
            "damage",
            "fatigue",
            "remodeling",
            "production",
            "activation",
            "energy",
        )

        for name in normalized_fields:
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"MaterialState.{name} must be finite"
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
            "pretension",
            "history",
            "current_length",
            "length_velocity",
            "extension",
            "strain",
            "passive_force_component",
            "damping_force_component",
            "active_force_component",
            "pretension_force_component",
            "total_force_component",
            "normalized_stimulus",
        )

        for name in finite_fields:
            value = float(
                getattr(self, name)
            )

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"MaterialState.{name} must be finite"
                )

            setattr(
                self,
                name,
                value,
            )

        self.current_length = max(
            0.0,
            self.current_length,
        )

        self.history = max(
            0.0,
            self.history,
        )

        self.normalized_stimulus = max(
            0.0,
            self.normalized_stimulus,
        )

        self.overloaded = bool(
            self.overloaded
        )

        self.failed = bool(
            self.failed
        )

        self.update_count = max(
            0,
            int(self.update_count),
        )

    def reset_force_components(self) -> None:
        self.passive_force_component = 0.0
        self.damping_force_component = 0.0
        self.active_force_component = 0.0
        self.pretension_force_component = 0.0
        self.total_force_component = 0.0

    def snapshot(self) -> dict[str, Any]:
        self.clamp()

        return {
            "damage": float(self.damage),
            "fatigue": float(self.fatigue),
            "remodeling": float(self.remodeling),
            "production": float(self.production),
            "pretension": float(self.pretension),
            "activation": float(self.activation),
            "history": float(self.history),
            "energy": float(self.energy),
            "current_length": float(
                self.current_length
            ),
            "length_velocity": float(
                self.length_velocity
            ),
            "extension": float(self.extension),
            "strain": float(self.strain),
            "passive_force_component": float(
                self.passive_force_component
            ),
            "damping_force_component": float(
                self.damping_force_component
            ),
            "active_force_component": float(
                self.active_force_component
            ),
            "pretension_force_component": float(
                self.pretension_force_component
            ),
            "total_force_component": float(
                self.total_force_component
            ),
            "normalized_stimulus": float(
                self.normalized_stimulus
            ),
            "overloaded": bool(self.overloaded),
            "failed": bool(self.failed),
            "update_count": int(
                self.update_count
            ),
        }


@dataclass
class Material:
    """
    Base axial constitutive material.

    Lifecycle
    ---------
    begin_step(...)
        Updates only pre-mechanical control state.

    force(...)
        Evaluates force without biological evolution.

    end_step(...)
        Updates fatigue, damage, repair, production,
        remodeling, history, energy and pretension exactly once
        after the mechanical solution.

    update(...)
        Compatibility wrapper combining begin_step(), force()
        and end_step(). New Network code should not use it when
        the split lifecycle is available.
    """

    name: str = "material"

    parameters: MaterialParameters = field(
        default_factory=MaterialParameters
    )

    state: MaterialState = field(
        default_factory=MaterialState
    )

    reference_length: float | None = None

    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(
            self.parameters,
            MaterialParameters,
        ):
            raise TypeError(
                "parameters must be MaterialParameters"
            )

        if not isinstance(
            self.state,
            MaterialState,
        ):
            raise TypeError(
                "state must be MaterialState"
            )

        self.name = str(self.name)

        self.epsilon = float(
            self.epsilon
        )

        if not np.isfinite(self.epsilon):
            raise ValueError(
                "epsilon must be finite"
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive"
            )

        if self.reference_length is not None:
            self.reference_length = (
                self._validate_length(
                    self.reference_length,
                    name="reference_length",
                )
            )

        self.validate()

    # =========================================================
    # Validation
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

    @staticmethod
    def _validate_scalar(
        value: float,
        *,
        name: str,
    ) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        return value

    @classmethod
    def _validate_length(
        cls,
        value: float,
        *,
        name: str,
    ) -> float:
        value = cls._validate_scalar(
            value,
            name=name,
        )

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

        return value

    def validate(self) -> None:
        self.parameters.validate()
        self.state.clamp()

        if self.reference_length is not None:
            self.reference_length = (
                self._validate_length(
                    self.reference_length,
                    name="reference_length",
                )
            )

    # =========================================================
    # Mechanical integrity
    # =========================================================

    def integrity(self) -> float:
        """
        Remaining structural capacity.

        Damage and fatigue both reduce effective integrity.
        A failed material has no remaining structural capacity.
        """
        if self.state.failed:
            return 0.0

        damage_factor = (
            1.0 - self.state.damage
        )

        fatigue_factor = (
            1.0 - self.state.fatigue
        )

        integrity = (
            damage_factor
            * fatigue_factor
        )

        return float(
            np.clip(
                integrity,
                self.parameters.minimum_integrity,
                self.parameters.maximum_integrity,
            )
        )

    def effective_stiffness(self) -> float:
        if self.state.failed:
            return 0.0

        return float(
            self.parameters.stiffness
            * self.integrity()
            * max(
                self.state.remodeling,
                self.epsilon,
            )
        )

    def effective_damping(self) -> float:
        if self.state.failed:
            return 0.0

        return float(
            self.parameters.damping
            * max(
                self.integrity(),
                self.epsilon,
            )
        )

    # =========================================================
    # Geometry
    # =========================================================

    def effective_reference_length(
        self,
        fallback_length: float | None = None,
    ) -> float:
        if self.reference_length is not None:
            return float(
                self.reference_length
            )

        if fallback_length is not None:
            return self._validate_length(
                fallback_length,
                name="fallback_length",
            )

        return 0.0

    def calculate_extension(
        self,
        current_length: float,
        *,
        reference_length: float | None = None,
    ) -> float:
        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        if reference_length is None:
            reference_length = (
                self.effective_reference_length()
            )

        reference_length = self._validate_length(
            reference_length,
            name="reference_length",
        )

        return float(
            current_length
            - reference_length
        )

    def calculate_strain(
        self,
        current_length: float,
        *,
        reference_length: float | None = None,
    ) -> float:
        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        if reference_length is None:
            reference_length = (
                self.effective_reference_length()
            )

        reference_length = self._validate_length(
            reference_length,
            name="reference_length",
        )

        if reference_length <= self.epsilon:
            return float(
                current_length
            )

        return float(
            (
                current_length
                - reference_length
            )
            / reference_length
        )

    # =========================================================
    # Force components
    # =========================================================

    def passive_force(
        self,
        extension: float,
    ) -> float:
        """
        Linear elastic base law.

            F_elastic = k_eff * extension
        """

        extension = self._validate_scalar(
            extension,
            name="extension",
        )

        return float(
            self.effective_stiffness()
            * extension
        )

    def damping_force(
        self,
        relative_velocity: float,
    ) -> float:
        """
        Viscous force opposing length change.

        Positive length velocity means extension, therefore the
        damping contribution is positive under the established
        axial sign convention.
        """

        relative_velocity = (
            self._validate_scalar(
                relative_velocity,
                name="relative_velocity",
            )
        )

        return float(
            self.effective_damping()
            * relative_velocity
        )

    def pretension_force(self) -> float:
        if self.state.failed:
            return 0.0

        return float(
            self.state.pretension
            * self.integrity()
        )

    def active_force(
        self,
        current_length: float | None = None,
        length_velocity: float = 0.0,
    ) -> float:
        """
        Base Material has no active contractile force.

        MuscleMaterial overrides this behavior.
        """

        return 0.0

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
        Evaluate axial force without biological evolution.

        This method is safe to call repeatedly inside mechanical
        calculations because it does not advance fatigue,
        damage, remodeling or energy.
        """

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        length_velocity = (
            self._validate_scalar(
                length_velocity,
                name="length_velocity",
            )
        )

        if reference_length is None:
            reference_length = (
                self.effective_reference_length()
            )

        reference_length = self._validate_length(
            reference_length,
            name="reference_length",
        )

        extension = (
            current_length
            - reference_length
        )

        if reference_length <= self.epsilon:
            strain = current_length
        else:
            strain = (
                extension
                / reference_length
            )

        passive = self.passive_force(
            extension
        )

        damping = self.damping_force(
            length_velocity
        )

        pretension = (
            self.pretension_force()
        )

        active = (
            self.active_force(
                current_length,
                length_velocity,
            )
            if include_active
            else 0.0
        )

        total = (
            passive
            + damping
            + pretension
            + active
        )

        values = {
            "passive force": passive,
            "damping force": damping,
            "pretension force": pretension,
            "active force": active,
            "total force": total,
        }

        for name, value in values.items():
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"{name} must be finite"
                )

        if self.state.failed:
            passive = 0.0
            damping = 0.0
            pretension = 0.0
            active = 0.0
            total = 0.0

        if store_state:
            self.state.current_length = (
                current_length
            )

            self.state.length_velocity = (
                length_velocity
            )

            self.state.extension = (
                extension
            )

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

            self.state.clamp()

        return float(total)

    def total_force(
        self,
        extension: float,
        relative_velocity: float = 0.0,
    ) -> float:
        """
        Legacy force interface.

        Uses extension directly and therefore does not require a
        configured reference length.
        """

        extension = self._validate_scalar(
            extension,
            name="extension",
        )

        relative_velocity = (
            self._validate_scalar(
                relative_velocity,
                name="relative_velocity",
            )
        )

        passive = self.passive_force(
            extension
        )

        damping = self.damping_force(
            relative_velocity
        )

        pretension = (
            self.pretension_force()
        )

        active = self.active_force(
            None,
            relative_velocity,
        )

        total = (
            passive
            + damping
            + pretension
            + active
        )

        if self.state.failed:
            total = 0.0

        self.state.extension = extension
        self.state.length_velocity = (
            relative_velocity
        )

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

        self.state.clamp()

        return float(total)

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
        Pre-mechanical state update.

        Base Material has no neural or control state, so this
        method only validates and stores available kinematics.

        MuscleMaterial should override this method to evolve
        activation before mechanical force evaluation.
        """

        self._validate_dt(dt)

        if current_length is not None:
            self.state.current_length = (
                self._validate_length(
                    current_length,
                    name="current_length",
                )
            )

        self.state.length_velocity = (
            self._validate_scalar(
                length_velocity,
                name="length_velocity",
            )
        )

        self.state.clamp()

    def end_step(
        self,
        dt: float,
        current_length: float,
        length_velocity: float = 0.0,
        force: float | None = None,
        reference_length: float | None = None,
    ) -> None:
        """
        Post-mechanical biological evolution.

        This is the only base method that advances:

        - loading history;
        - fatigue;
        - damage;
        - recovery;
        - production;
        - remodeling;
        - pretension;
        - energy.

        It should be called exactly once per physical timestep.
        """

        dt = self._validate_dt(dt)

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        length_velocity = (
            self._validate_scalar(
                length_velocity,
                name="length_velocity",
            )
        )

        if force is None:
            force = self.force(
                current_length=current_length,
                length_velocity=length_velocity,
                reference_length=reference_length,
                store_state=True,
            )
        else:
            force = self._validate_scalar(
                force,
                name="force",
            )

            if reference_length is None:
                reference_length = (
                    self.effective_reference_length()
                )

            reference_length = (
                self._validate_length(
                    reference_length,
                    name="reference_length",
                )
            )

            extension = (
                current_length
                - reference_length
            )

            if reference_length <= self.epsilon:
                strain = current_length
            else:
                strain = (
                    extension
                    / reference_length
                )

            self.state.current_length = (
                current_length
            )

            self.state.length_velocity = (
                length_velocity
            )

            self.state.extension = extension
            self.state.strain = strain

            self.state.total_force_component = (
                force
            )

        stimulus = self.normalized_stimulus(
            force=force,
            strain=self.state.strain,
        )

        self.state.normalized_stimulus = (
            stimulus
        )

        self.state.overloaded = (
            self.overloaded(stimulus)
        )

        self.update_history(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_fatigue(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_damage(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_repair(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_production(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_remodeling(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_pretension(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_energy(
            dt=dt,
            stimulus=stimulus,
        )

        self.update_failure_state()

        self.state.update_count += 1
        self.state.clamp()

    def update(
        self,
        dt: float,
        current_length: float,
        length_velocity: float = 0.0,
        force: float | None = None,
        reference_length: float | None = None,
    ) -> float:
        """
        Compatibility lifecycle.

        New code should use:

            begin_step(...)
            force(...)
            end_step(...)

        separately.
        """

        self.begin_step(
            dt=dt,
            current_length=current_length,
            length_velocity=length_velocity,
        )

        evaluated_force = self.force(
            current_length=current_length,
            length_velocity=length_velocity,
            reference_length=reference_length,
            store_state=True,
        )

        if force is None:
            force = evaluated_force

        self.end_step(
            dt=dt,
            current_length=current_length,
            length_velocity=length_velocity,
            force=force,
            reference_length=reference_length,
        )

        return float(
            evaluated_force
        )

    # =========================================================
    # Stimulus
    # =========================================================

    def normalized_force_stimulus(
        self,
        *,
        force: float | None = None,
    ) -> float:
        """
        Normalized mechanical load transmitted by the material.

        This quantity depends only on force. A failed material normally
        has zero force stimulus because it can no longer transmit force.
        """
        if force is None:
            force = (
                self.state.total_force_component
            )

        force = self._validate_scalar(
            force,
            name="force",
        )

        return float(
            abs(force)
            / self.parameters.reference_force
        )

    def normalized_strain_stimulus(
        self,
        *,
        strain: float | None = None,
    ) -> float:
        """
        Normalized geometric deformation.

        This quantity depends only on strain. It can remain nonzero after
        failure because a failed element may stay geometrically stretched.
        """
        if strain is None:
            strain = self.state.strain

        strain = self._validate_scalar(
            strain,
            name="strain",
        )

        return float(
            abs(strain)
            / self.parameters.reference_strain
        )

    def normalized_stimulus(
        self,
        *,
        force: float | None = None,
        strain: float | None = None,
    ) -> float:
        """
        Combined stimulus used by the existing material lifecycle.

        The combined value is the larger of normalized transmitted force
        and normalized geometric strain. This preserves compatibility with
        overload, fatigue, damage, remodeling, and automatic failure.
        """
        force_stimulus = (
            self.normalized_force_stimulus(
                force=force,
            )
        )

        strain_stimulus = (
            self.normalized_strain_stimulus(
                strain=strain,
            )
        )

        return float(
            max(
                force_stimulus,
                strain_stimulus,
            )
        )


    def overloaded(
        self,
        stimulus: float | None = None,
    ) -> bool:
        if stimulus is None:
            stimulus = (
                self.state.normalized_stimulus
            )

        stimulus = self._validate_scalar(
            stimulus,
            name="stimulus",
        )

        return bool(
            stimulus
            > self.parameters.overload_threshold
        )

    # =========================================================
    # Biological operators
    # =========================================================

    def update_history(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        """
        First-order loading-memory operator.
        """

        dt = self._validate_dt(dt)

        stimulus = max(
            0.0,
            self._validate_scalar(
                stimulus,
                name="stimulus",
            ),
        )

        memory_rate = max(
            self.parameters.remodeling_rate,
            self.epsilon,
        )

        blend = float(
            np.clip(
                memory_rate * dt,
                0.0,
                1.0,
            )
        )

        self.state.history += (
            blend
            * (
                stimulus
                - self.state.history
            )
        )

    def update_fatigue(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        load_excess = max(
            0.0,
            stimulus,
        )

        fatigue_gain = (
            self.parameters.fatigue_rate
            * load_excess
            * (1.0 - self.state.fatigue)
            * dt
        )

        recovery = (
            self.parameters.recovery_rate
            * self.state.energy
            * self.state.fatigue
            * dt
        )

        self.state.fatigue += (
            fatigue_gain
            - recovery
        )

    def update_damage(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        overload = max(
            0.0,
            stimulus
            - self.parameters.overload_threshold,
        )

        fatigue_amplification = (
            1.0 + self.state.fatigue
        )

        damage_gain = (
            self.parameters.damage_rate
            * overload
            * fatigue_amplification
            * (1.0 - self.state.damage)
            * dt
        )

        self.state.damage += damage_gain

    def update_repair(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        overload = max(
            0.0,
            stimulus
            - self.parameters.overload_threshold,
        )

        repair_suppression = (
            1.0
            / (1.0 + overload)
        )

        repair = (
            self.parameters.recovery_rate
            * self.state.energy
            * repair_suppression
            * self.state.damage
            * dt
        )

        self.state.damage -= repair

    def update_production(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        adaptive_signal = float(
            np.clip(
                stimulus,
                0.0,
                1.0,
            )
        )

        production_gain = (
            self.parameters.production_rate
            * adaptive_signal
            * self.state.energy
            * (1.0 - self.state.production)
            * dt
        )

        production_decay = (
            self.parameters.recovery_rate
            * 0.25
            * self.state.production
            * dt
        )

        self.state.production += (
            production_gain
            - production_decay
        )

    def update_remodeling(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        target = float(
            np.clip(
                1.0
                + 0.25
                * (
                    stimulus
                    - 1.0
                ),
                0.0,
                1.0,
            )
        )

        adaptation_capacity = (
            self.state.energy
            * (1.0 - self.state.damage)
        )

        rate = (
            self.parameters.remodeling_rate
            * adaptation_capacity
        )

        self.state.remodeling += (
            rate
            * (
                target
                - self.state.remodeling
            )
            * dt
        )

    def update_pretension(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        if self.parameters.pretension_rate <= 0.0:
            return

        target_pretension = (
            self.parameters.reference_force
            * max(
                0.0,
                stimulus - 1.0,
            )
            * self.state.production
        )

        self.state.pretension += (
            self.parameters.pretension_rate
            * (
                target_pretension
                - self.state.pretension
            )
            * dt
        )

    def update_energy(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        consumption = (
            self.parameters.energy_decay_rate
            * (
                1.0
                + stimulus
                + self.state.fatigue
            )
            * dt
        )

        recovery = (
            self.parameters.recovery_rate
            * (
                1.0 - self.state.energy
            )
            * dt
        )

        self.state.energy += (
            recovery
            - consumption
        )

    def update_failure_state(self) -> None:
        failure_metric = max(
            self.state.damage,
            self.state.normalized_stimulus
            / self.parameters.failure_threshold,
        )

        if failure_metric >= 1.0:
            self.state.failed = True

    # =========================================================
    # Energy
    # =========================================================

    def elastic_energy(
        self,
        extension: float | None = None,
    ) -> float:
        if extension is None:
            extension = self.state.extension

        extension = self._validate_scalar(
            extension,
            name="extension",
        )

        return float(
            max(
                0.0,
                0.5
                * self.effective_stiffness()
                * extension
                * extension,
            )
        )

    # =========================================================
    # Runtime control
    # =========================================================

    def reset_state(self) -> None:
        state_type = type(self.state)

        self.state = state_type()

        self.validate()

    def set_failed(
        self,
        failed: bool,
    ) -> None:
        self.state.failed = bool(
            failed
        )

    def clone(self) -> Material:
        return deepcopy(self)

    # =========================================================
    # Compatibility properties
    # =========================================================

    @property
    def stiffness(self) -> float:
        return float(
            self.parameters.stiffness
        )

    @stiffness.setter
    def stiffness(
        self,
        value: float,
    ) -> None:
        previous = self.parameters.stiffness

        self.parameters.stiffness = float(
            value
        )

        try:
            self.parameters.validate()
        except ValueError:
            self.parameters.stiffness = previous
            raise

    @property
    def damping(self) -> float:
        return float(
            self.parameters.damping
        )

    @damping.setter
    def damping(
        self,
        value: float,
    ) -> None:
        previous = self.parameters.damping

        self.parameters.damping = float(
            value
        )

        try:
            self.parameters.validate()
        except ValueError:
            self.parameters.damping = previous
            raise

    @property
    def density(self) -> float:
        return float(
            self.parameters.density
        )

    @density.setter
    def density(
        self,
        value: float,
    ) -> None:
        previous = self.parameters.density

        self.parameters.density = float(
            value
        )

        try:
            self.parameters.validate()
        except ValueError:
            self.parameters.density = previous
            raise

    @property
    def damage(self) -> float:
        return float(
            self.state.damage
        )

    @damage.setter
    def damage(
        self,
        value: float,
    ) -> None:
        self.state.damage = float(
            value
        )

        self.state.clamp()

    @property
    def fatigue(self) -> float:
        return float(
            self.state.fatigue
        )

    @fatigue.setter
    def fatigue(
        self,
        value: float,
    ) -> None:
        self.state.fatigue = float(
            value
        )

        self.state.clamp()

    @property
    def remodeling(self) -> float:
        return float(
            self.state.remodeling
        )

    @remodeling.setter
    def remodeling(
        self,
        value: float,
    ) -> None:
        self.state.remodeling = float(
            value
        )

        self.state.clamp()

    @property
    def production(self) -> float:
        return float(
            self.state.production
        )

    @production.setter
    def production(
        self,
        value: float,
    ) -> None:
        self.state.production = float(
            value
        )

        self.state.clamp()

    @property
    def pretension(self) -> float:
        return float(
            self.state.pretension
        )

    @pretension.setter
    def pretension(
        self,
        value: float,
    ) -> None:
        value = self._validate_scalar(
            value,
            name="pretension",
        )

        self.state.pretension = value

    @property
    def activation(self) -> float:
        return float(
            self.state.activation
        )

    @activation.setter
    def activation(
        self,
        value: float,
    ) -> None:
        self.state.activation = float(
            value
        )

        self.state.clamp()

    @property
    def history(self) -> float:
        return float(
            self.state.history
        )

    @history.setter
    def history(
        self,
        value: float,
    ) -> None:
        self.state.history = float(
            value
        )

        self.state.clamp()

    @property
    def energy(self) -> float:
        return float(
            self.state.energy
        )

    @energy.setter
    def energy(
        self,
        value: float,
    ) -> None:
        self.state.energy = float(
            value
        )

        self.state.clamp()

    @property
    def failed(self) -> bool:
        return bool(
            self.state.failed
        )

    # =========================================================
    # Snapshot
    # =========================================================

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": type(self).__name__,
            "reference_length": (
                None
                if self.reference_length is None
                else float(
                    self.reference_length
                )
            ),
            "integrity": self.integrity(),
            "effective_stiffness": (
                self.effective_stiffness()
            ),
            "effective_damping": (
                self.effective_damping()
            ),
            "parameters": (
                self.parameters.snapshot()
            ),
            "state": self.state.snapshot(),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"stiffness={self.stiffness:.6g}, "
            f"damping={self.damping:.6g}, "
            f"damage={self.damage:.3f}, "
            f"fatigue={self.fatigue:.3f}, "
            f"remodeling={self.remodeling:.3f}, "
            f"energy={self.energy:.3f}, "
            f"failed={self.failed})"
        )