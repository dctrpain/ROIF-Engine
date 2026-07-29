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
class LigamentParameters(MaterialParameters):
    """
    Mechanical and biological parameters of an axial ligament.

    Mechanical regions
    ------------------
    slack_length:
        Length below which the ligament produces no tensile force.

    toe_strain:
        Strain at which the nonlinear toe region transitions
        into the approximately linear region.

    toe_stiffness:
        Initial stiffness scale in the toe region.

    linear_stiffness:
        Tangent stiffness after the toe region.

    transition_smoothness:
        Controls the numerical smoothness around the transition
        between toe and linear regions.

    maximum_strain:
        Strain at which structural failure may occur.

    rupture_strain:
        Strain at which the ligament is considered ruptured.

    Viscoelasticity
    ---------------
    loading_damping:
        Damping coefficient during lengthening.

    unloading_damping:
        Damping coefficient during shortening.

    hysteresis_ratio:
        Reduces unloading force relative to loading force.

    Creep and plasticity
    --------------------
    creep_rate:
        Rate of reversible reference-length adaptation under
        sustained tensile load.

    creep_recovery_rate:
        Rate at which creep recovers when load decreases.

    plasticity_rate:
        Rate of permanent reference-length increase after the
        plastic strain threshold is exceeded.

    plastic_strain_threshold:
        Strain above which permanent elongation may accumulate.

    maximum_plastic_strain:
        Upper bound for accumulated permanent elongation.

    Damage
    ------
    microdamage_strain:
        Strain above which ligament-specific damage begins.

    rupture_damage:
        Damage level treated as rupture.
    """

    slack_length: float = 1.0

    toe_strain: float = 0.04
    toe_stiffness: float = 100.0
    linear_stiffness: float = 1000.0
    transition_smoothness: float = 0.005

    maximum_strain: float = 0.12
    rupture_strain: float = 0.18

    loading_damping: float = 1.0
    unloading_damping: float = 0.5
    hysteresis_ratio: float = 0.05

    creep_rate: float = 0.0
    creep_recovery_rate: float = 0.0

    plasticity_rate: float = 0.0
    plastic_strain_threshold: float = 0.08
    maximum_plastic_strain: float = 0.20

    microdamage_strain: float = 0.06
    rupture_damage: float = 1.0

    compression_stiffness: float = 0.0

    def validate(self) -> None:
        super().validate()

        fields = (
            "slack_length",
            "toe_strain",
            "toe_stiffness",
            "linear_stiffness",
            "transition_smoothness",
            "maximum_strain",
            "rupture_strain",
            "loading_damping",
            "unloading_damping",
            "hysteresis_ratio",
            "creep_rate",
            "creep_recovery_rate",
            "plasticity_rate",
            "plastic_strain_threshold",
            "maximum_plastic_strain",
            "microdamage_strain",
            "rupture_damage",
            "compression_stiffness",
        )

        for name in fields:
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise ValueError(
                    f"LigamentParameters.{name} must be finite"
                )

            if value < 0.0:
                raise ValueError(
                    f"LigamentParameters.{name} cannot be negative"
                )

            setattr(self, name, value)

        positive_fields = (
            "slack_length",
            "toe_stiffness",
            "linear_stiffness",
            "transition_smoothness",
            "maximum_strain",
            "rupture_strain",
            "rupture_damage",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"LigamentParameters.{name} must be positive"
                )

        if self.linear_stiffness < self.toe_stiffness:
            raise ValueError(
                "linear_stiffness cannot be smaller than "
                "toe_stiffness"
            )

        if self.maximum_strain <= self.toe_strain:
            raise ValueError(
                "maximum_strain must exceed toe_strain"
            )

        if self.rupture_strain <= self.maximum_strain:
            raise ValueError(
                "rupture_strain must exceed maximum_strain"
            )

        if (
            self.plastic_strain_threshold
            < self.toe_strain
        ):
            raise ValueError(
                "plastic_strain_threshold cannot be smaller "
                "than toe_strain"
            )

        if (
            self.maximum_plastic_strain
            < self.plastic_strain_threshold
        ):
            raise ValueError(
                "maximum_plastic_strain cannot be smaller than "
                "plastic_strain_threshold"
            )

        if self.microdamage_strain > self.rupture_strain:
            raise ValueError(
                "microdamage_strain cannot exceed rupture_strain"
            )

        if self.hysteresis_ratio > 1.0:
            raise ValueError(
                "hysteresis_ratio cannot exceed 1"
            )

        if self.rupture_damage > 1.0:
            raise ValueError(
                "rupture_damage cannot exceed 1"
            )

    def snapshot(self) -> dict[str, float]:
        self.validate()

        return {
            name: float(value)
            for name, value in asdict(self).items()
        }


@dataclass
class LigamentState(MaterialState):
    """
    Mutable ligament-specific state.

    creep_strain:
        Reversible reference-length increase.

    plastic_strain:
        Permanent reference-length increase.

    effective_slack_length:
        Current constitutive slack length.

    tensile_strain:
        Strain relative to current effective slack length.

    tangent_stiffness:
        Current tangent stiffness of the constitutive curve.

    toe_region:
        True while tensile strain remains below toe_strain.

    loading:
        True while the ligament length is increasing.

    ruptured:
        Permanent rupture flag.
    """

    creep_strain: float = 0.0
    plastic_strain: float = 0.0

    effective_slack_length: float = 0.0
    tensile_strain: float = 0.0

    tangent_stiffness: float = 0.0

    loading: bool = False
    toe_region: bool = False
    taut: bool = False
    ruptured: bool = False

    peak_strain: float = 0.0
    peak_force: float = 0.0

    creep_update_count: int = 0
    plasticity_update_count: int = 0

    def clamp(self) -> None:
        super().clamp()

        nonnegative_fields = (
            "creep_strain",
            "plastic_strain",
            "effective_slack_length",
            "tangent_stiffness",
            "peak_strain",
            "peak_force",
        )

        for name in nonnegative_fields:
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"LigamentState.{name} must be finite"
                )

            setattr(
                self,
                name,
                max(0.0, value),
            )

        self.tensile_strain = float(
            self.tensile_strain
        )

        if not np.isfinite(self.tensile_strain):
            raise FloatingPointError(
                "LigamentState.tensile_strain must be finite"
            )

        self.loading = bool(self.loading)
        self.toe_region = bool(self.toe_region)
        self.taut = bool(self.taut)
        self.ruptured = bool(self.ruptured)

        self.creep_update_count = max(
            0,
            int(self.creep_update_count),
        )

        self.plasticity_update_count = max(
            0,
            int(self.plasticity_update_count),
        )

        if self.ruptured:
            self.failed = True

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()

        snapshot.update(
            {
                "creep_strain": float(
                    self.creep_strain
                ),
                "plastic_strain": float(
                    self.plastic_strain
                ),
                "effective_slack_length": float(
                    self.effective_slack_length
                ),
                "tensile_strain": float(
                    self.tensile_strain
                ),
                "tangent_stiffness": float(
                    self.tangent_stiffness
                ),
                "loading": bool(self.loading),
                "toe_region": bool(
                    self.toe_region
                ),
                "taut": bool(self.taut),
                "ruptured": bool(self.ruptured),
                "peak_strain": float(
                    self.peak_strain
                ),
                "peak_force": float(
                    self.peak_force
                ),
                "creep_update_count": int(
                    self.creep_update_count
                ),
                "plasticity_update_count": int(
                    self.plasticity_update_count
                ),
            }
        )

        return snapshot


@dataclass
class LigamentMaterial(Material):
    """
    Nonlinear, tension-dominant ligament material.

    Lifecycle
    ---------
    begin_step():
        Stores current ligament kinematics.

    force():
        Evaluates the nonlinear tensile response without advancing
        damage, creep or plasticity.

    end_step():
        Updates creep, plasticity, damage, remodeling and failure
        exactly once after the mechanical solve.

    Sign convention
    ---------------
    positive:
        tension;

    negative:
        compression.

    By default compression_stiffness is zero, therefore the
    ligament produces no compressive force.
    """

    name: str = "ligament"

    parameters: LigamentParameters = field(
        default_factory=LigamentParameters
    )

    state: LigamentState = field(
        default_factory=LigamentState
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(
            self.parameters,
            LigamentParameters,
        ):
            raise TypeError(
                "LigamentMaterial.parameters must be "
                "LigamentParameters"
            )

        if not isinstance(
            self.state,
            LigamentState,
        ):
            raise TypeError(
                "LigamentMaterial.state must be LigamentState"
            )

        if self.reference_length is None:
            self.reference_length = float(
                self.parameters.slack_length
            )

        self.state.effective_slack_length = (
            self.effective_slack_length()
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
            LigamentParameters,
        ):
            raise TypeError(
                "parameters must be LigamentParameters"
            )

        if not isinstance(
            self.state,
            LigamentState,
        ):
            raise TypeError(
                "state must be LigamentState"
            )

        self.parameters.validate()
        self.state.clamp()

        self.state.creep_strain = float(
            np.clip(
                self.state.creep_strain,
                0.0,
                self.parameters.maximum_plastic_strain,
            )
        )

        self.state.plastic_strain = float(
            np.clip(
                self.state.plastic_strain,
                0.0,
                self.parameters.maximum_plastic_strain,
            )
        )

    # =========================================================
    # Reference length
    # =========================================================

    def base_slack_length(self) -> float:
        if self.reference_length is not None:
            return float(self.reference_length)

        return float(
            self.parameters.slack_length
        )

    def effective_slack_length(self) -> float:
        """
        Slack length after reversible creep and permanent
        plastic elongation.
        """

        total_adaptation = (
            self.state.creep_strain
            + self.state.plastic_strain
        )

        return float(
            self.base_slack_length()
            * (1.0 + total_adaptation)
        )

    def effective_reference_length(
        self,
        fallback_length: float | None = None,
    ) -> float:
        return self.effective_slack_length()

    # =========================================================
    # Kinematics
    # =========================================================

    def tensile_strain(
        self,
        current_length: float,
    ) -> float:
        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        slack_length = (
            self.effective_slack_length()
        )

        if slack_length <= self.epsilon:
            return 0.0

        return float(
            (
                current_length
                - slack_length
            )
            / slack_length
        )

    def is_taut(
        self,
        current_length: float,
    ) -> bool:
        return bool(
            current_length
            > self.effective_slack_length()
            + self.epsilon
        )

    # =========================================================
    # Nonlinear constitutive law
    # =========================================================

    def toe_force(
        self,
        strain: float,
    ) -> float:
        """
        Quadratic toe-region force.

        The coefficient is selected so that the tangent stiffness
        reaches toe_stiffness at toe_strain.
        """

        strain = max(
            0.0,
            self._validate_scalar(
                strain,
                name="strain",
            ),
        )

        toe_strain = max(
            self.parameters.toe_strain,
            self.epsilon,
        )

        return float(
            0.5
            * self.parameters.toe_stiffness
            * strain
            * strain
            / toe_strain
        )

    def toe_transition_force(self) -> float:
        return self.toe_force(
            self.parameters.toe_strain
        )

    def linear_region_force(
        self,
        strain: float,
    ) -> float:
        strain = max(
            0.0,
            self._validate_scalar(
                strain,
                name="strain",
            ),
        )

        toe_strain = (
            self.parameters.toe_strain
        )

        return float(
            self.toe_transition_force()
            + self.parameters.linear_stiffness
            * (
                strain
                - toe_strain
            )
        )

    def tensile_elastic_force(
        self,
        strain: float,
    ) -> float:
        """
        Piecewise nonlinear ligament force.

        Toe region:
            quadratic;

        Linear region:
            continuous linear continuation.
        """

        strain = self._validate_scalar(
            strain,
            name="strain",
        )

        if strain <= 0.0:
            return 0.0

        if strain <= self.parameters.toe_strain:
            force = self.toe_force(strain)
        else:
            force = self.linear_region_force(strain)

        return float(
            max(
                0.0,
                force
                * self.integrity()
                * max(
                    self.state.remodeling,
                    self.epsilon,
                ),
            )
        )

    def tangent_stiffness(
        self,
        strain: float,
    ) -> float:
        strain = self._validate_scalar(
            strain,
            name="strain",
        )

        if strain <= 0.0:
            return 0.0

        if strain <= self.parameters.toe_strain:
            stiffness = (
                self.parameters.toe_stiffness
                * strain
                / max(
                    self.parameters.toe_strain,
                    self.epsilon,
                )
            )
        else:
            stiffness = (
                self.parameters.linear_stiffness
            )

        return float(
            max(
                0.0,
                stiffness
                * self.integrity()
                * max(
                    self.state.remodeling,
                    self.epsilon,
                ),
            )
        )

    def passive_force(
        self,
        extension: float,
    ) -> float:
        """
        Compatibility implementation for Material.total_force().
        """

        extension = self._validate_scalar(
            extension,
            name="extension",
        )

        slack_length = max(
            self.effective_slack_length(),
            self.epsilon,
        )

        if extension >= 0.0:
            strain = extension / slack_length
            return self.tensile_elastic_force(
                strain
            )

        return float(
            self.parameters.compression_stiffness
            * extension
        )

    # =========================================================
    # Viscosity and hysteresis
    # =========================================================

    def ligament_damping_force(
        self,
        length_velocity: float,
        *,
        taut: bool,
    ) -> float:
        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        if not taut:
            return 0.0

        if length_velocity >= 0.0:
            damping = (
                self.parameters.loading_damping
            )
        else:
            damping = (
                self.parameters.unloading_damping
            )

        return float(
            damping
            * length_velocity
            * self.integrity()
        )

    def hysteresis_multiplier(
        self,
        length_velocity: float,
    ) -> float:
        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        if length_velocity >= 0.0:
            return 1.0

        return float(
            max(
                0.0,
                1.0
                - self.parameters.hysteresis_ratio,
            )
        )

    # =========================================================
    # Total ligament force
    # =========================================================

    def ligament_force(
        self,
        current_length: float,
        length_velocity: float = 0.0,
        *,
        store_state: bool = True,
    ) -> float:
        """
        Evaluate total ligament force without advancing biology.
        """

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        slack_length = (
            self.effective_slack_length()
        )

        strain = self.tensile_strain(
            current_length
        )

        taut = current_length > (
            slack_length + self.epsilon
        )

        loading = (
            length_velocity >= 0.0
        )

        if self.state.failed or self.state.ruptured:
            elastic = 0.0
            damping = 0.0
            pretension = 0.0
            total = 0.0

        elif taut:
            elastic = (
                self.tensile_elastic_force(
                    strain
                )
                * self.hysteresis_multiplier(
                    length_velocity
                )
            )

            damping = (
                self.ligament_damping_force(
                    length_velocity,
                    taut=True,
                )
            )

            pretension = max(
                0.0,
                self.pretension_force(),
            )

            total = max(
                0.0,
                elastic
                + damping
                + pretension,
            )

        else:
            compression = (
                self.parameters.compression_stiffness
                * (
                    current_length
                    - slack_length
                )
            )

            elastic = compression
            damping = 0.0
            pretension = 0.0
            total = min(
                0.0,
                compression,
            )

        if store_state:
            self.state.current_length = (
                current_length
            )

            self.state.length_velocity = (
                length_velocity
            )

            self.state.extension = (
                current_length
                - slack_length
            )

            self.state.strain = strain
            self.state.tensile_strain = strain

            self.state.effective_slack_length = (
                slack_length
            )

            self.state.passive_force_component = (
                elastic
            )

            self.state.damping_force_component = (
                damping
            )

            self.state.pretension_force_component = (
                pretension
            )

            self.state.active_force_component = 0.0

            self.state.total_force_component = (
                total
            )

            self.state.loading = loading
            self.state.taut = taut

            self.state.toe_region = bool(
                taut
                and strain
                <= self.parameters.toe_strain
            )

            self.state.tangent_stiffness = (
                self.tangent_stiffness(
                    strain
                )
            )

            self.state.peak_strain = max(
                self.state.peak_strain,
                strain,
            )

            self.state.peak_force = max(
                self.state.peak_force,
                abs(total),
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

        ``reference_length`` and ``include_active`` are accepted
        for interface compatibility.
        """

        if reference_length is not None:
            self._validate_length(
                reference_length,
                name="reference_length",
            )

        return self.ligament_force(
            current_length=current_length,
            length_velocity=length_velocity,
            store_state=store_state,
        )

    # =========================================================
    # Step lifecycle
    # =========================================================

    def begin_step(
        self,
        dt: float,
        current_length: float | None = None,
        length_velocity: float = 0.0,
    ) -> None:
        super().begin_step(
            dt=dt,
            current_length=current_length,
            length_velocity=length_velocity,
        )

        if current_length is not None:
            self.state.effective_slack_length = (
                self.effective_slack_length()
            )

            self.state.tensile_strain = (
                self.tensile_strain(
                    current_length
                )
            )

            self.state.loading = bool(
                length_velocity >= 0.0
            )

            self.state.taut = self.is_taut(
                current_length
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
        Advance ligament biology once after mechanics.
        """

        dt = self._validate_dt(dt)

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        if force is None:
            force = self.ligament_force(
                current_length=current_length,
                length_velocity=length_velocity,
                store_state=True,
            )
        else:
            force = self._validate_scalar(
                force,
                name="force",
            )

            self.ligament_force(
                current_length=current_length,
                length_velocity=length_velocity,
                store_state=True,
            )

            self.state.total_force_component = (
                force
            )

        self.update_creep(
            dt=dt,
            current_length=current_length,
        )

        self.update_plasticity(
            dt=dt,
            current_length=current_length,
        )

        super().end_step(
            dt=dt,
            current_length=current_length,
            length_velocity=length_velocity,
            force=force,
            reference_length=(
                self.effective_slack_length()
            ),
        )

        self.update_rupture_state()

        self.state.effective_slack_length = (
            self.effective_slack_length()
        )

        self.state.clamp()

    # =========================================================
    # Creep
    # =========================================================

    def update_creep(
        self,
        *,
        dt: float,
        current_length: float,
    ) -> None:
        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self.tensile_strain(
                current_length
            ),
        )

        if strain > self.parameters.toe_strain:
            overload = (
                strain
                - self.parameters.toe_strain
            )

            creep_gain = (
                self.parameters.creep_rate
                * overload
                * (
                    1.0
                    - self.state.creep_strain
                    / max(
                        self.parameters.maximum_plastic_strain,
                        self.epsilon,
                    )
                )
                * dt
            )

            self.state.creep_strain += (
                creep_gain
            )

        else:
            recovery = (
                self.parameters.creep_recovery_rate
                * self.state.energy
                * self.state.creep_strain
                * dt
            )

            self.state.creep_strain -= recovery

        self.state.creep_strain = float(
            np.clip(
                self.state.creep_strain,
                0.0,
                self.parameters.maximum_plastic_strain,
            )
        )

        self.state.creep_update_count += 1

    # =========================================================
    # Plasticity
    # =========================================================

    def update_plasticity(
        self,
        *,
        dt: float,
        current_length: float,
    ) -> None:
        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self.tensile_strain(
                current_length
            ),
        )

        excess = max(
            0.0,
            strain
            - self.parameters.plastic_strain_threshold,
        )

        if excess > 0.0:
            remaining_capacity = (
                1.0
                - self.state.plastic_strain
                / max(
                    self.parameters.maximum_plastic_strain,
                    self.epsilon,
                )
            )

            plastic_gain = (
                self.parameters.plasticity_rate
                * excess
                * max(
                    0.0,
                    remaining_capacity,
                )
                * dt
            )

            self.state.plastic_strain += (
                plastic_gain
            )

        self.state.plastic_strain = float(
            np.clip(
                self.state.plastic_strain,
                0.0,
                self.parameters.maximum_plastic_strain,
            )
        )

        self.state.plasticity_update_count += 1

    # =========================================================
    # Ligament damage
    # =========================================================

    def update_damage(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self.state.tensile_strain,
        )

        strain_excess = max(
            0.0,
            strain
            - self.parameters.microdamage_strain,
        )

        normalized_excess = (
            strain_excess
            / max(
                self.parameters.rupture_strain
                - self.parameters.microdamage_strain,
                self.epsilon,
            )
        )

        overload = max(
            0.0,
            stimulus
            - self.parameters.overload_threshold,
        )

        fatigue_amplification = (
            1.0
            + self.state.fatigue
        )

        damage_gain = (
            self.parameters.damage_rate
            * max(
                normalized_excess,
                overload,
            )
            * fatigue_amplification
            * (
                1.0
                - self.state.damage
            )
            * dt
        )

        self.state.damage += damage_gain

        if strain >= self.parameters.rupture_strain:
            self.state.damage = max(
                self.state.damage,
                self.parameters.rupture_damage,
            )

    def update_repair(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        """
        Ligament repair is inhibited during substantial tension.
        """

        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self.state.tensile_strain,
        )

        mechanical_suppression = (
            1.0
            / (
                1.0
                + strain
                / max(
                    self.parameters.toe_strain,
                    self.epsilon,
                )
            )
        )

        repair = (
            self.parameters.recovery_rate
            * self.state.energy
            * mechanical_suppression
            * self.state.damage
            * dt
        )

        self.state.damage -= repair

    # =========================================================
    # Remodeling
    # =========================================================

    def update_remodeling(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        dt = self._validate_dt(dt)

        target = float(
            np.clip(
                0.75
                + 0.25
                * min(
                    stimulus,
                    1.0,
                ),
                0.0,
                1.0,
            )
        )

        capacity = (
            self.state.energy
            * (
                1.0
                - self.state.damage
            )
        )

        self.state.remodeling += (
            self.parameters.remodeling_rate
            * capacity
            * (
                target
                - self.state.remodeling
            )
            * dt
        )

    # =========================================================
    # Failure
    # =========================================================

    def update_rupture_state(self) -> None:
        strain = self.state.tensile_strain

        if (
            strain
            >= self.parameters.rupture_strain
        ):
            self.state.ruptured = True

        if (
            self.state.damage
            >= self.parameters.rupture_damage
        ):
            self.state.ruptured = True

        if self.state.ruptured:
            self.state.failed = True

    def update_failure_state(self) -> None:
        super().update_failure_state()
        self.update_rupture_state()

    # =========================================================
    # Stimulus
    # =========================================================

    def normalized_stimulus(
        self,
        *,
        force: float | None = None,
        strain: float | None = None,
    ) -> float:
        base_stimulus = (
            super().normalized_stimulus(
                force=force,
                strain=strain,
            )
        )

        ligament_strain = max(
            0.0,
            self.state.tensile_strain,
        )

        normalized_ligament_strain = (
            ligament_strain
            / max(
                self.parameters.maximum_strain,
                self.epsilon,
            )
        )

        return float(
            max(
                base_stimulus,
                normalized_ligament_strain,
            )
        )

    # =========================================================
    # Energy
    # =========================================================

    def elastic_energy(
        self,
        extension: float | None = None,
    ) -> float:
        """
        Approximate stored energy of the nonlinear ligament curve.
        """

        strain = max(
            0.0,
            self.state.tensile_strain,
        )

        slack_length = max(
            self.effective_slack_length(),
            self.epsilon,
        )

        if strain <= 0.0:
            return 0.0

        if strain <= self.parameters.toe_strain:
            energy_density = (
                self.parameters.toe_stiffness
                * strain**3
                / (
                    6.0
                    * max(
                        self.parameters.toe_strain,
                        self.epsilon,
                    )
                )
            )
        else:
            toe_strain = (
                self.parameters.toe_strain
            )

            toe_energy = (
                self.parameters.toe_stiffness
                * toe_strain**2
                / 6.0
            )

            transition_force = (
                self.toe_transition_force()
            )

            linear_extension = (
                strain
                - toe_strain
            )

            linear_energy = (
                transition_force
                * linear_extension
                + 0.5
                * self.parameters.linear_stiffness
                * linear_extension**2
            )

            energy_density = (
                toe_energy
                + linear_energy
            )

        return float(
            max(
                0.0,
                energy_density
                * slack_length
                * self.integrity(),
            )
        )

    # =========================================================
    # Diagnostics
    # =========================================================

    def is_loading(self) -> bool:
        return bool(
            self.state.loading
        )

    def is_unloading(self) -> bool:
        return bool(
            not self.state.loading
        )

    def is_in_toe_region(self) -> bool:
        return bool(
            self.state.toe_region
        )

    def is_linear_region(self) -> bool:
        return bool(
            self.state.taut
            and not self.state.toe_region
        )

    def is_ruptured(self) -> bool:
        return bool(
            self.state.ruptured
        )

    def strain_reserve(self) -> float:
        """
        Remaining strain reserve before rupture.
        """

        reserve = (
            self.parameters.rupture_strain
            - max(
                0.0,
                self.state.tensile_strain,
            )
        )

        return float(
            max(
                0.0,
                reserve,
            )
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset_state(self) -> None:
        self.state = LigamentState(
            effective_slack_length=(
                self.base_slack_length()
            )
        )

        self.validate()

    # =========================================================
    # Snapshot
    # =========================================================

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()

        snapshot.update(
            {
                "ligament": {
                    "base_slack_length": (
                        self.base_slack_length()
                    ),
                    "effective_slack_length": (
                        self.effective_slack_length()
                    ),
                    "tensile_strain": float(
                        self.state.tensile_strain
                    ),
                    "tangent_stiffness": float(
                        self.state.tangent_stiffness
                    ),
                    "creep_strain": float(
                        self.state.creep_strain
                    ),
                    "plastic_strain": float(
                        self.state.plastic_strain
                    ),
                    "strain_reserve": (
                        self.strain_reserve()
                    ),
                    "taut": bool(
                        self.state.taut
                    ),
                    "toe_region": bool(
                        self.state.toe_region
                    ),
                    "linear_region": (
                        self.is_linear_region()
                    ),
                    "loading": bool(
                        self.state.loading
                    ),
                    "ruptured": bool(
                        self.state.ruptured
                    ),
                    "peak_strain": float(
                        self.state.peak_strain
                    ),
                    "peak_force": float(
                        self.state.peak_force
                    ),
                }
            }
        )

        return snapshot

    def __repr__(self) -> str:
        return (
            f"LigamentMaterial("
            f"name={self.name!r}, "
            f"strain={self.state.tensile_strain:.6f}, "
            f"force={self.state.total_force_component:.6g}, "
            f"creep={self.state.creep_strain:.6f}, "
            f"plastic={self.state.plastic_strain:.6f}, "
            f"damage={self.state.damage:.3f}, "
            f"ruptured={self.state.ruptured})"
        )