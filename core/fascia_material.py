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
class FasciaParameters(MaterialParameters):
    """
    Parameters of a passive viscoelastic fascial material.

    Mechanical response
    -------------------
    slack_length:
        Length below which tensile fascial force is absent.

    toe_strain:
        End of the low-stiffness recruitment region.

    toe_stiffness:
        Initial collagen recruitment stiffness.

    linear_stiffness:
        Tangent stiffness after fiber recruitment.

    compression_stiffness:
        Optional low compressive stiffness.

    loading_damping:
        Viscous coefficient during lengthening.

    unloading_damping:
        Viscous coefficient during shortening.

    hysteresis_ratio:
        Reduction of elastic force during unloading.

    Stress relaxation
    -----------------
    relaxation_rate:
        Rate of stress relaxation under sustained strain.

    minimum_relaxation_factor:
        Lower bound of retained elastic force after relaxation.

    Creep
    -----
    creep_rate:
        Rate of reversible reference-length increase under
        sustained tensile load.

    creep_recovery_rate:
        Recovery of creep after unloading.

    maximum_creep_strain:
        Upper bound for reversible fascial elongation.

    Plastic adaptation
    ------------------
    plasticity_rate:
        Rate of permanent elongation above threshold.

    plastic_strain_threshold:
        Tensile strain above which permanent elongation begins.

    maximum_plastic_strain:
        Maximum permanent elongation.

    Fiber recruitment
    -----------------
    recruitment_rate:
        Rate at which collagen recruitment follows load.

    derecruitment_rate:
        Rate at which recruitment decreases after unloading.

    minimum_recruitment:
        Baseline fraction of recruited fibers.

    Damage
    ------
    microdamage_strain:
        Strain above which fascial microdamage begins.

    failure_strain:
        Strain associated with structural failure.
    """

    slack_length: float = 1.0

    toe_strain: float = 0.05
    toe_stiffness: float = 25.0
    linear_stiffness: float = 250.0

    compression_stiffness: float = 0.0

    loading_damping: float = 2.0
    unloading_damping: float = 1.0
    hysteresis_ratio: float = 0.10

    relaxation_rate: float = 0.05
    minimum_relaxation_factor: float = 0.30

    creep_rate: float = 0.001
    creep_recovery_rate: float = 0.0002
    maximum_creep_strain: float = 0.20

    plasticity_rate: float = 0.0001
    plastic_strain_threshold: float = 0.12
    maximum_plastic_strain: float = 0.25

    recruitment_rate: float = 1.0
    derecruitment_rate: float = 0.25
    minimum_recruitment: float = 0.05

    microdamage_strain: float = 0.15
    failure_strain: float = 0.40

    def validate(self) -> None:
        super().validate()

        fields = (
            "slack_length",
            "toe_strain",
            "toe_stiffness",
            "linear_stiffness",
            "compression_stiffness",
            "loading_damping",
            "unloading_damping",
            "hysteresis_ratio",
            "relaxation_rate",
            "minimum_relaxation_factor",
            "creep_rate",
            "creep_recovery_rate",
            "maximum_creep_strain",
            "plasticity_rate",
            "plastic_strain_threshold",
            "maximum_plastic_strain",
            "recruitment_rate",
            "derecruitment_rate",
            "minimum_recruitment",
            "microdamage_strain",
            "failure_strain",
        )

        for name in fields:
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise ValueError(
                    f"FasciaParameters.{name} must be finite"
                )

            if value < 0.0:
                raise ValueError(
                    f"FasciaParameters.{name} cannot be negative"
                )

            setattr(self, name, value)

        positive_fields = (
            "slack_length",
            "toe_strain",
            "toe_stiffness",
            "linear_stiffness",
            "failure_strain",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"FasciaParameters.{name} must be positive"
                )

        if self.linear_stiffness < self.toe_stiffness:
            raise ValueError(
                "linear_stiffness cannot be smaller than "
                "toe_stiffness"
            )

        if not 0.0 <= self.hysteresis_ratio <= 1.0:
            raise ValueError(
                "hysteresis_ratio must be in [0, 1]"
            )

        if not 0.0 <= self.minimum_relaxation_factor <= 1.0:
            raise ValueError(
                "minimum_relaxation_factor must be in [0, 1]"
            )

        if not 0.0 <= self.minimum_recruitment <= 1.0:
            raise ValueError(
                "minimum_recruitment must be in [0, 1]"
            )

        if (
            self.maximum_plastic_strain
            < self.plastic_strain_threshold
        ):
            raise ValueError(
                "maximum_plastic_strain cannot be smaller than "
                "plastic_strain_threshold"
            )

        if self.failure_strain <= self.microdamage_strain:
            raise ValueError(
                "failure_strain must exceed microdamage_strain"
            )

    def snapshot(self) -> dict[str, float]:
        self.validate()

        return {
            name: float(value)
            for name, value in asdict(self).items()
        }


@dataclass
class FasciaState(MaterialState):
    """
    Mutable fascia-specific state.

    creep_strain:
        Reversible elongation of the effective reference length.

    plastic_strain:
        Permanent elongation.

    recruitment:
        Current collagen fiber recruitment in [0, 1].

    relaxation_factor:
        Fraction of elastic force retained after stress
        relaxation.

    effective_slack_length:
        Current constitutive slack length.

    tensile_strain:
        Strain relative to current effective slack length.

    loading:
        True when length velocity is nonnegative.

    taut:
        True when current length exceeds effective slack length.

    peak_strain / peak_force:
        Historical maxima.
    """

    creep_strain: float = 0.0
    plastic_strain: float = 0.0

    recruitment: float = 0.05
    relaxation_factor: float = 1.0

    effective_slack_length: float = 0.0
    tensile_strain: float = 0.0
    tangent_stiffness: float = 0.0

    loading: bool = False
    taut: bool = False
    toe_region: bool = False

    peak_strain: float = 0.0
    peak_force: float = 0.0

    creep_update_count: int = 0
    plasticity_update_count: int = 0
    relaxation_update_count: int = 0
    recruitment_update_count: int = 0

    def clamp(self) -> None:
        super().clamp()

        normalized_fields = (
            "recruitment",
            "relaxation_factor",
        )

        for name in normalized_fields:
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise FloatingPointError(
                    f"FasciaState.{name} must be finite"
                )

            setattr(
                self,
                name,
                float(np.clip(value, 0.0, 1.0)),
            )

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
                    f"FasciaState.{name} must be finite"
                )

            setattr(self, name, max(0.0, value))

        self.tensile_strain = float(
            self.tensile_strain
        )

        if not np.isfinite(self.tensile_strain):
            raise FloatingPointError(
                "FasciaState.tensile_strain must be finite"
            )

        self.loading = bool(self.loading)
        self.taut = bool(self.taut)
        self.toe_region = bool(self.toe_region)

        self.creep_update_count = max(
            0,
            int(self.creep_update_count),
        )

        self.plasticity_update_count = max(
            0,
            int(self.plasticity_update_count),
        )

        self.relaxation_update_count = max(
            0,
            int(self.relaxation_update_count),
        )

        self.recruitment_update_count = max(
            0,
            int(self.recruitment_update_count),
        )

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
                "recruitment": float(
                    self.recruitment
                ),
                "relaxation_factor": float(
                    self.relaxation_factor
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
                "loading": bool(
                    self.loading
                ),
                "taut": bool(
                    self.taut
                ),
                "toe_region": bool(
                    self.toe_region
                ),
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
                "relaxation_update_count": int(
                    self.relaxation_update_count
                ),
                "recruitment_update_count": int(
                    self.recruitment_update_count
                ),
            }
        )

        return snapshot


@dataclass
class FasciaMaterial(Material):
    """
    Passive nonlinear viscoelastic fascial material.

    The material supports:

    - collagen recruitment;
    - toe and linear tensile regions;
    - asymmetric loading and unloading damping;
    - hysteresis;
    - stress relaxation;
    - reversible creep;
    - permanent plastic elongation;
    - slow remodeling and damage.

    Positive force means tension.
    """

    name: str = "fascia"

    parameters: FasciaParameters = field(
        default_factory=FasciaParameters
    )

    state: FasciaState = field(
        default_factory=FasciaState
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(
            self.parameters,
            FasciaParameters,
        ):
            raise TypeError(
                "FasciaMaterial.parameters must be "
                "FasciaParameters"
            )

        if not isinstance(
            self.state,
            FasciaState,
        ):
            raise TypeError(
                "FasciaMaterial.state must be FasciaState"
            )

        if self.reference_length is None:
            self.reference_length = float(
                self.parameters.slack_length
            )

        self.state.recruitment = max(
            self.state.recruitment,
            self.parameters.minimum_recruitment,
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
            FasciaParameters,
        ):
            raise TypeError(
                "parameters must be FasciaParameters"
            )

        if not isinstance(
            self.state,
            FasciaState,
        ):
            raise TypeError(
                "state must be FasciaState"
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

        self.state.recruitment = float(
            np.clip(
                self.state.recruitment,
                self.parameters.minimum_recruitment,
                1.0,
            )
        )

        self.state.relaxation_factor = float(
            np.clip(
                self.state.relaxation_factor,
                self.parameters.minimum_relaxation_factor,
                1.0,
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
    # Recruitment
    # =========================================================

    def target_recruitment(
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

        normalized = strain / max(
            self.parameters.toe_strain,
            self.epsilon,
        )

        return float(
            np.clip(
                self.parameters.minimum_recruitment
                + (
                    1.0
                    - self.parameters.minimum_recruitment
                )
                * normalized,
                self.parameters.minimum_recruitment,
                1.0,
            )
        )

    def update_recruitment(
        self,
        *,
        dt: float,
        strain: float,
    ) -> None:
        dt = self._validate_dt(dt)

        target = self.target_recruitment(
            strain
        )

        if target >= self.state.recruitment:
            rate = self.parameters.recruitment_rate
        else:
            rate = self.parameters.derecruitment_rate

        blend = float(
            np.clip(
                rate * dt,
                0.0,
                1.0,
            )
        )

        self.state.recruitment += (
            blend
            * (
                target
                - self.state.recruitment
            )
        )

        self.state.recruitment = float(
            np.clip(
                self.state.recruitment,
                self.parameters.minimum_recruitment,
                1.0,
            )
        )

        self.state.recruitment_update_count += 1

    # =========================================================
    # Constitutive response
    # =========================================================

    def toe_force(
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

    def transition_force(self) -> float:
        return self.toe_force(
            self.parameters.toe_strain
        )

    def linear_force(
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
            self.transition_force()
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
            raw_force = self.linear_force(strain)

        capacity = (
            self.integrity()
            * max(
                self.state.remodeling,
                self.epsilon,
            )
            * self.state.recruitment
            * self.state.relaxation_factor
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

        capacity = (
            self.integrity()
            * max(
                self.state.remodeling,
                self.epsilon,
            )
            * self.state.recruitment
            * self.state.relaxation_factor
        )

        return float(
            max(
                0.0,
                raw_stiffness * capacity,
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
            return self.tensile_elastic_force(
                extension / slack_length
            )

        return float(
            self.parameters.compression_stiffness
            * extension
        )

    # =========================================================
    # Damping and hysteresis
    # =========================================================

    def fascia_damping_force(
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
            * self.state.recruitment
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

    def fascia_force(
        self,
        current_length: float,
        length_velocity: float = 0.0,
        *,
        store_state: bool = True,
    ) -> float:
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

        if self.state.failed:
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
                self.fascia_damping_force(
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
        if reference_length is not None:
            self._validate_length(
                reference_length,
                name="reference_length",
            )

        return self.fascia_force(
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
        dt = self._validate_dt(dt)

        current_length = self._validate_length(
            current_length,
            name="current_length",
        )

        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        evaluated_force = self.fascia_force(
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

        self.update_recruitment(
            dt=dt,
            strain=strain,
        )

        self.update_relaxation(
            dt=dt,
            strain=strain,
            length_velocity=length_velocity,
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

        self.update_fascial_failure()

        self.state.effective_slack_length = (
            self.effective_slack_length()
        )

        self.state.clamp()

    # =========================================================
    # Stress relaxation
    # =========================================================

    def update_relaxation(
        self,
        *,
        dt: float,
        strain: float,
        length_velocity: float,
    ) -> None:
        dt = self._validate_dt(dt)

        strain = max(
            0.0,
            self._validate_scalar(
                strain,
                name="strain",
            ),
        )

        length_velocity = self._validate_scalar(
            length_velocity,
            name="length_velocity",
        )

        sustained_loading = (
            strain > 0.0
            and abs(length_velocity)
            <= self.epsilon
        )

        if sustained_loading:
            target = (
                self.parameters.minimum_relaxation_factor
            )

            rate = (
                self.parameters.relaxation_rate
                * (
                    1.0 + strain
                )
            )
        else:
            target = 1.0

            rate = (
                self.parameters.recovery_rate
            )

        blend = float(
            np.clip(
                rate * dt,
                0.0,
                1.0,
            )
        )

        self.state.relaxation_factor += (
            blend
            * (
                target
                - self.state.relaxation_factor
            )
        )

        self.state.relaxation_factor = float(
            np.clip(
                self.state.relaxation_factor,
                self.parameters.minimum_relaxation_factor,
                1.0,
            )
        )

        self.state.relaxation_update_count += 1

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

        normalized_strain_excess = (
            strain_excess
            / max(
                self.parameters.failure_strain
                - self.parameters.microdamage_strain,
                self.epsilon,
            )
        )

        overload = max(
            0.0,
            stimulus
            - self.parameters.overload_threshold,
        )

        damage_drive = max(
            normalized_strain_excess,
            overload,
        )

        self.state.damage += (
            self.parameters.damage_rate
            * damage_drive
            * (
                1.0 + self.state.fatigue
            )
            * (
                1.0 - self.state.damage
            )
            * dt
        )

        if strain >= self.parameters.failure_strain:
            self.state.damage = 1.0

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
                0.60
                + 0.40
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

    def update_fascial_failure(self) -> None:
        if (
            self.state.tensile_strain
            >= self.parameters.failure_strain
        ):
            self.state.failed = True

        if self.state.damage >= 1.0:
            self.state.failed = True

    def update_failure_state(self) -> None:
        super().update_failure_state()
        self.update_fascial_failure()

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

        fascial_strain = max(
            0.0,
            self.state.tensile_strain,
        )

        normalized_fascial_strain = (
            fascial_strain
            / max(
                self.parameters.failure_strain,
                self.epsilon,
            )
        )

        relaxation_load = (
            1.0
            - self.state.relaxation_factor
        )

        return float(
            max(
                base_stimulus,
                normalized_fascial_strain,
                relaxation_load,
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

            linear_extension = (
                strain
                - toe_strain
            )

            linear_energy = (
                self.transition_force()
                * linear_extension
                + 0.5
                * self.parameters.linear_stiffness
                * linear_extension**2
            )

            energy_density = (
                toe_energy
                + linear_energy
            )

        capacity = (
            self.integrity()
            * self.state.recruitment
            * self.state.relaxation_factor
        )

        return float(
            max(
                0.0,
                energy_density
                * slack_length
                * capacity,
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

    def is_relaxed(self) -> bool:
        return bool(
            self.state.relaxation_factor
            < 1.0 - self.epsilon
        )

    def strain_reserve(self) -> float:
        return float(
            max(
                0.0,
                self.parameters.failure_strain
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
        self.state = FasciaState(
            recruitment=(
                self.parameters.minimum_recruitment
            ),
            relaxation_factor=1.0,
            effective_slack_length=(
                self.base_slack_length()
            ),
        )

        self.validate()

    # =========================================================
    # Snapshot
    # =========================================================

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()

        snapshot.update(
            {
                "fascia": {
                    "base_slack_length": (
                        self.base_slack_length()
                    ),
                    "effective_slack_length": (
                        self.effective_slack_length()
                    ),
                    "tensile_strain": float(
                        self.state.tensile_strain
                    ),
                    "recruitment": float(
                        self.state.recruitment
                    ),
                    "relaxation_factor": float(
                        self.state.relaxation_factor
                    ),
                    "creep_strain": float(
                        self.state.creep_strain
                    ),
                    "plastic_strain": float(
                        self.state.plastic_strain
                    ),
                    "tangent_stiffness": float(
                        self.state.tangent_stiffness
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
                    "relaxed": (
                        self.is_relaxed()
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
            f"FasciaMaterial("
            f"name={self.name!r}, "
            f"strain={self.state.tensile_strain:.6f}, "
            f"force={self.state.total_force_component:.6g}, "
            f"recruitment={self.state.recruitment:.3f}, "
            f"relaxation={self.state.relaxation_factor:.3f}, "
            f"creep={self.state.creep_strain:.6f}, "
            f"plastic={self.state.plastic_strain:.6f}, "
            f"damage={self.state.damage:.3f}, "
            f"failed={self.state.failed})"
        )