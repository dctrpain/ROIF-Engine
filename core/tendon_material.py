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
class TendonParameters(MaterialParameters):
    """
    Parameters of a passive axial tendon material.

    The tendon model includes:

    - slack region;
    - short nonlinear toe region;
    - high-stiffness linear region;
    - asymmetric viscous response;
    - small hysteresis;
    - slow reversible creep;
    - limited permanent elongation;
    - strain-driven damage;
    - rupture.
    """

    slack_length: float = 1.0

    toe_strain: float = 0.025
    toe_stiffness: float = 500.0
    linear_stiffness: float = 5000.0

    loading_damping: float = 0.50
    unloading_damping: float = 0.25
    hysteresis_ratio: float = 0.05

    creep_rate: float = 0.00005
    creep_recovery_rate: float = 0.00002
    maximum_creep_strain: float = 0.05

    plasticity_rate: float = 0.00001
    plastic_strain_threshold: float = 0.06
    maximum_plastic_strain: float = 0.10

    microdamage_strain: float = 0.04
    maximum_strain: float = 0.08
    rupture_strain: float = 0.12
    rupture_damage: float = 1.0

    compression_stiffness: float = 0.0

    def validate(self) -> None:
        super().validate()

        fields = (
            "slack_length",
            "toe_strain",
            "toe_stiffness",
            "linear_stiffness",
            "loading_damping",
            "unloading_damping",
            "hysteresis_ratio",
            "creep_rate",
            "creep_recovery_rate",
            "maximum_creep_strain",
            "plasticity_rate",
            "plastic_strain_threshold",
            "maximum_plastic_strain",
            "microdamage_strain",
            "maximum_strain",
            "rupture_strain",
            "rupture_damage",
            "compression_stiffness",
        )

        for name in fields:
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise ValueError(
                    f"TendonParameters.{name} must be finite"
                )

            if value < 0.0:
                raise ValueError(
                    f"TendonParameters.{name} cannot be negative"
                )

            setattr(self, name, value)

        positive_fields = (
            "slack_length",
            "toe_strain",
            "toe_stiffness",
            "linear_stiffness",
            "maximum_strain",
            "rupture_strain",
            "rupture_damage",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"TendonParameters.{name} must be positive"
                )

        if self.linear_stiffness < self.toe_stiffness:
            raise ValueError(
                "linear_stiffness cannot be smaller than "
                "toe_stiffness"
            )

        if self.microdamage_strain < self.toe_strain:
            raise ValueError(
                "microdamage_strain cannot be smaller than "
                "toe_strain"
            )

        if self.maximum_strain <= self.microdamage_strain:
            raise ValueError(
                "maximum_strain must exceed microdamage_strain"
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

        if not 0.0 <= self.hysteresis_ratio <= 1.0:
            raise ValueError(
                "hysteresis_ratio must be in [0, 1]"
            )

        if not 0.0 < self.rupture_damage <= 1.0:
            raise ValueError(
                "rupture_damage must be in (0, 1]"
            )

    def snapshot(self) -> dict[str, float]:
        self.validate()

        return {
            name: float(value)
            for name, value in asdict(self).items()
        }


@dataclass
class TendonState(MaterialState):
    """
    Mutable tendon state.

    creep_strain:
        Reversible increase of the effective slack length.

    plastic_strain:
        Permanent increase of the effective slack length.

    tensile_strain:
        Current strain relative to effective slack length.

    tangent_stiffness:
        Current tangent stiffness.

    ruptured:
        Irreversible structural failure flag.
    """

    creep_strain: float = 0.0
    plastic_strain: float = 0.0

    effective_slack_length: float = 0.0
    tensile_strain: float = 0.0
    tangent_stiffness: float = 0.0

    loading: bool = False
    taut: bool = False
    toe_region: bool = False
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
                    f"TendonState.{name} must be finite"
                )

            setattr(self, name, max(0.0, value))

        self.tensile_strain = float(
            self.tensile_strain
        )

        if not np.isfinite(self.tensile_strain):
            raise FloatingPointError(
                "TendonState.tensile_strain must be finite"
            )

        self.loading = bool(self.loading)
        self.taut = bool(self.taut)
        self.toe_region = bool(self.toe_region)
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
                "taut": bool(self.taut),
                "toe_region": bool(
                    self.toe_region
                ),
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
class TendonMaterial(Material):
    """
    Passive nonlinear tendon material.

    Lifecycle:

        begin_step()
            -> stores pre-force kinematics

        force()
            -> pure mechanical evaluation

        end_step()
            -> creep, plasticity, damage, repair,
               remodeling and rupture

    Positive force means tension.
    """

    name: str = "tendon"

    parameters: TendonParameters = field(
        default_factory=TendonParameters
    )

    state: TendonState = field(
        default_factory=TendonState
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(
            self.parameters,
            TendonParameters,
        ):
            raise TypeError(
                "TendonMaterial.parameters must be "
                "TendonParameters"
            )

        if not isinstance(
            self.state,
            TendonState,
        ):
            raise TypeError(
                "TendonMaterial.state must be TendonState"
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
            TendonParameters,
        ):
            raise TypeError(
                "parameters must be TendonParameters"
            )

        if not isinstance(
            self.state,
            TendonState,
        ):
            raise TypeError(
                "state must be TendonState"
            )

        self.parameters.validate()

        self.state.creep_strain = float(
            np.clip(
                self.state.creep_strain,
                0.0,
                self.parameters.maximum_creep_strain,
            )
        )

        self.state.plastic_strain = float(
            np.clip(
                self.state.plastic_strain,
                0.0,
                self.parameters.maximum_plastic_strain,
            )
        )

        self.state.clamp()

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
        adaptation = (
            self.state.creep_strain
            + self.state.plastic_strain
        )

        return float(
            self.base_slack_length()
            * (1.0 + adaptation)
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

        slack_length = max(
            self.effective_slack_length(),
            self.epsilon,
        )

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
    # Constitutive response
    # =========================================================

    def toe_force(
        self,
        strain: float,
    ) -> float:
        """
        Quadratic collagen-uncrimping region.

        Tangent stiffness reaches toe_stiffness at toe_strain.
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
            * strain**2
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

        return float(
            self.toe_transition_force()
            + self.parameters.linear_stiffness
            * (
                strain
                - self.parameters.toe_strain
            )
        )

    def tensile_elastic_force(
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
            raw_force = self.toe_force(strain)
        else:
            raw_force = self.linear_region_force(
                strain
            )

        capacity = (
            self.integrity()
            * max(
                self.state.remodeling,
                self.epsilon,
            )
        )

        return float(
            max(
                0.0,
                raw_force * capacity,
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
            raw_stiffness = (
                self.parameters.toe_stiffness
                * strain
                / max(
                    self.parameters.toe_strain,
                    self.epsilon,
                )
            )
        else:
            raw_stiffness = (
                self.parameters.linear_stiffness
            )

        return float(
            max(
                0.0,
                raw_stiffness
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

    def active_force(
        self,
        current_length: float | None = None,
        length_velocity: float = 0.0,
    ) -> float:
        return 0.0

    def active_capacity(self) -> float:
        return 0.0

    # =========================================================
    # Viscosity and hysteresis
    # =========================================================

    def tendon_damping_force(
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

        coefficient = (
            self.parameters.loading_damping
            if length_velocity >= 0.0
            else self.parameters.unloading_damping
        )

        return float(
            coefficient
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
    # Total force
    # =========================================================

    def tendon_force(
        self,
        current_length: float,
        length_velocity: float = 0.0,
        *,
        store_state: bool = True,
    ) -> float:
        """
        Evaluate tendon force without changing biological state.
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

        loading = bool(
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

            damping = self.tendon_damping_force(
                length_velocity,
                taut=True,
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
        for the common Material interface.
        """

        if reference_length is not None:
            self._validate_length(
                reference_length,
                name="reference_length",
            )

        return self.tendon_force(
            current_length=current_length,
            length_velocity=length_velocity,
            store_state=store_state,
        )

    # =========================================================
    # Lifecycle
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

        if current_length is None:
            return

        strain = self.tensile_strain(
            current_length
        )

        self.state.effective_slack_length = (
            self.effective_slack_length()
        )

        self.state.tensile_strain = strain

        self.state.loading = bool(
            length_velocity >= 0.0
        )

        self.state.taut = self.is_taut(
            current_length
        )

        self.state.toe_region = bool(
            self.state.taut
            and strain
            <= self.parameters.toe_strain
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
        Advance tendon biology once after the mechanical solve.
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

        evaluated_force = self.tendon_force(
            current_length=current_length,
            length_velocity=length_velocity,
            store_state=True,
        )

        if force is None:
            force = evaluated_force
        else:
            force = self._validate_scalar(
                force,
                name="force",
            )

            self.state.total_force_component = (
                force
            )

        strain = max(
            0.0,
            self.state.tensile_strain,
        )

        self.update_creep(
            dt=dt,
            strain=strain,
        )

        self.update_plasticity(
            dt=dt,
            strain=strain,
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
        strain: float,
    ) -> None:
        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self._validate_scalar(
                strain,
                name="strain",
            ),
        )

        if strain > self.parameters.toe_strain:
            drive = (
                strain
                - self.parameters.toe_strain
            )

            remaining = (
                1.0
                - self.state.creep_strain
                / max(
                    self.parameters.maximum_creep_strain,
                    self.epsilon,
                )
            )

            self.state.creep_strain += (
                self.parameters.creep_rate
                * drive
                * max(0.0, remaining)
                * dt
            )

        else:
            self.state.creep_strain -= (
                self.parameters.creep_recovery_rate
                * self.state.energy
                * self.state.creep_strain
                * dt
            )

        self.state.creep_strain = float(
            np.clip(
                self.state.creep_strain,
                0.0,
                self.parameters.maximum_creep_strain,
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
        strain: float,
    ) -> None:
        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self._validate_scalar(
                strain,
                name="strain",
            ),
        )

        excess = max(
            0.0,
            strain
            - self.parameters.plastic_strain_threshold,
        )

        if excess > 0.0:
            remaining = (
                1.0
                - self.state.plastic_strain
                / max(
                    self.parameters.maximum_plastic_strain,
                    self.epsilon,
                )
            )

            self.state.plastic_strain += (
                self.parameters.plasticity_rate
                * excess
                * max(0.0, remaining)
                * dt
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
    # Damage and repair
    # =========================================================

    def update_damage(
        self,
        *,
        dt: float,
        stimulus: float,
    ) -> None:
        """
        Tendon damage is primarily strain-driven.
        """

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

        fatigue_amplification = (
            1.0
            + self.state.fatigue
        )

        damage_gain = (
            self.parameters.damage_rate
            * normalized_excess
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
        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self.state.tensile_strain,
        )

        load_suppression = (
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
            * load_suppression
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
                0.80
                + 0.20
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
    # Rupture
    # =========================================================

    def update_rupture_state(self) -> None:
        if (
            self.state.tensile_strain
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
        base_stimulus = super().normalized_stimulus(
            force=force,
            strain=strain,
        )

        tendon_strain = max(
            0.0,
            self.state.tensile_strain,
        )

        normalized_tendon_strain = (
            tendon_strain
            / max(
                self.parameters.maximum_strain,
                self.epsilon,
            )
        )

        return float(
            max(
                base_stimulus,
                normalized_tendon_strain,
            )
        )

    # =========================================================
    # Energy
    # =========================================================

    def elastic_energy(
        self,
        extension: float | None = None,
    ) -> float:
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

            linear_strain = (
                strain
                - toe_strain
            )

            linear_energy = (
                self.toe_transition_force()
                * linear_strain
                + 0.5
                * self.parameters.linear_stiffness
                * linear_strain**2
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
        return bool(self.state.loading)

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
        return float(
            max(
                0.0,
                self.parameters.rupture_strain
                - max(
                    0.0,
                    self.state.tensile_strain,
                ),
            )
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset_state(self) -> None:
        self.state = TendonState(
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
                "tendon": {
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
            f"TendonMaterial("
            f"name={self.name!r}, "
            f"strain={self.state.tensile_strain:.6f}, "
            f"force={self.state.total_force_component:.6g}, "
            f"creep={self.state.creep_strain:.6f}, "
            f"plastic={self.state.plastic_strain:.6f}, "
            f"damage={self.state.damage:.3f}, "
            f"ruptured={self.state.ruptured})"
        )