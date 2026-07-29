from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


# =============================================================
# Utility functions
# =============================================================


def _finite(
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


def _nonnegative(
    value: float,
    *,
    name: str,
) -> float:
    value = _finite(
        value,
        name=name,
    )

    if value < 0.0:
        raise ValueError(
            f"{name} cannot be negative"
        )

    return value


def _unit_interval(
    value: float,
    *,
    name: str,
) -> float:
    value = _finite(
        value,
        name=name,
    )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1]"
        )

    return value


def _clamp01(
    value: float,
) -> float:
    return float(
        np.clip(
            float(value),
            0.0,
            1.0,
        )
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    epsilon: float,
) -> float:
    return float(
        numerator
        / max(
            abs(denominator),
            epsilon,
        )
    )


# =============================================================
# Parameters
# =============================================================


@dataclass
class RemodelingParameters:
    """
    Generic mechanobiological remodeling parameters.

    The engine models five coupled processes:

    1. damage accumulation;
    2. biological repair;
    3. structural remodeling;
    4. production and degradation;
    5. pretension adaptation.

    Tissue-specific materials may either:

    - use these rules directly;
    - modify the input stimulus;
    - subclass RemodelingEngine;
    - override selected target calculations.
    """

    # ---------------------------------------------------------
    # Numerical
    # ---------------------------------------------------------

    epsilon: float = 1.0e-12

    # ---------------------------------------------------------
    # Homeostatic loading window
    # ---------------------------------------------------------

    homeostatic_stimulus: float = 0.50
    underload_threshold: float = 0.20
    overload_threshold: float = 1.00
    critical_stimulus: float = 2.00

    # ---------------------------------------------------------
    # Damage
    # ---------------------------------------------------------

    damage_rate: float = 0.01
    overload_damage_exponent: float = 2.0
    fatigue_damage_multiplier: float = 1.0
    energy_deficit_damage_multiplier: float = 1.0

    critical_damage: float = 0.95
    failure_damage: float = 1.00

    # ---------------------------------------------------------
    # Repair
    # ---------------------------------------------------------

    repair_rate: float = 0.002
    repair_energy_exponent: float = 1.0
    repair_load_suppression: float = 1.0
    minimum_repair_capacity: float = 0.0

    # ---------------------------------------------------------
    # Structural remodeling
    # ---------------------------------------------------------

    remodeling_rate: float = 0.001
    remodeling_loss_rate: float = 0.0005

    minimum_remodeling: float = 0.05
    maximum_remodeling: float = 1.50
    homeostatic_remodeling: float = 1.00

    overload_remodeling_gain: float = 0.25
    underload_remodeling_loss: float = 0.25

    # ---------------------------------------------------------
    # Material production
    # ---------------------------------------------------------

    production_rate: float = 0.001
    degradation_rate: float = 0.0005

    minimum_production: float = 0.0
    maximum_production: float = 1.50
    homeostatic_production: float = 1.00

    production_stimulus_gain: float = 0.50
    production_damage_gain: float = 0.50
    production_energy_exponent: float = 1.0

    # ---------------------------------------------------------
    # Pretension adaptation
    # ---------------------------------------------------------

    pretension_rate: float = 0.0
    pretension_decay_rate: float = 0.0

    minimum_pretension: float = 0.0
    maximum_pretension: float = 1.0

    pretension_stimulus_gain: float = 1.0
    pretension_damage_suppression: float = 1.0

    # ---------------------------------------------------------
    # Fatigue interaction
    # ---------------------------------------------------------

    fatigue_recovery_rate: float = 0.0
    fatigue_stimulus_rate: float = 0.0

    # ---------------------------------------------------------
    # Energy interaction
    # ---------------------------------------------------------

    energy_consumption_rate: float = 0.0
    energy_recovery_rate: float = 0.0

    # ---------------------------------------------------------
    # Temporal smoothing
    # ---------------------------------------------------------

    stimulus_filter_rate: float = 5.0
    history_weight: float = 0.0
    history_window: int = 32

    def validate(self) -> None:
        nonnegative_fields = (
            "epsilon",
            "homeostatic_stimulus",
            "underload_threshold",
            "overload_threshold",
            "critical_stimulus",
            "damage_rate",
            "overload_damage_exponent",
            "fatigue_damage_multiplier",
            "energy_deficit_damage_multiplier",
            "critical_damage",
            "failure_damage",
            "repair_rate",
            "repair_energy_exponent",
            "repair_load_suppression",
            "minimum_repair_capacity",
            "remodeling_rate",
            "remodeling_loss_rate",
            "minimum_remodeling",
            "maximum_remodeling",
            "homeostatic_remodeling",
            "overload_remodeling_gain",
            "underload_remodeling_loss",
            "production_rate",
            "degradation_rate",
            "minimum_production",
            "maximum_production",
            "homeostatic_production",
            "production_stimulus_gain",
            "production_damage_gain",
            "production_energy_exponent",
            "pretension_rate",
            "pretension_decay_rate",
            "minimum_pretension",
            "maximum_pretension",
            "pretension_stimulus_gain",
            "pretension_damage_suppression",
            "fatigue_recovery_rate",
            "fatigue_stimulus_rate",
            "energy_consumption_rate",
            "energy_recovery_rate",
            "stimulus_filter_rate",
            "history_weight",
        )

        for name in nonnegative_fields:
            value = _nonnegative(
                getattr(self, name),
                name=f"RemodelingParameters.{name}",
            )

            setattr(
                self,
                name,
                value,
            )

        self.history_window = int(
            self.history_window
        )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive"
            )

        if self.history_window <= 0:
            raise ValueError(
                "history_window must be positive"
            )

        if (
            self.underload_threshold
            > self.homeostatic_stimulus
        ):
            raise ValueError(
                "underload_threshold cannot exceed "
                "homeostatic_stimulus"
            )

        if (
            self.overload_threshold
            < self.homeostatic_stimulus
        ):
            raise ValueError(
                "overload_threshold cannot be smaller than "
                "homeostatic_stimulus"
            )

        if (
            self.critical_stimulus
            <= self.overload_threshold
        ):
            raise ValueError(
                "critical_stimulus must exceed "
                "overload_threshold"
            )

        if (
            self.critical_damage
            > self.failure_damage
        ):
            raise ValueError(
                "critical_damage cannot exceed failure_damage"
            )

        if self.failure_damage > 1.0:
            raise ValueError(
                "failure_damage cannot exceed 1"
            )

        if (
            self.minimum_remodeling
            > self.maximum_remodeling
        ):
            raise ValueError(
                "minimum_remodeling cannot exceed "
                "maximum_remodeling"
            )

        if not (
            self.minimum_remodeling
            <= self.homeostatic_remodeling
            <= self.maximum_remodeling
        ):
            raise ValueError(
                "homeostatic_remodeling must lie inside "
                "remodeling bounds"
            )

        if (
            self.minimum_production
            > self.maximum_production
        ):
            raise ValueError(
                "minimum_production cannot exceed "
                "maximum_production"
            )

        if not (
            self.minimum_production
            <= self.homeostatic_production
            <= self.maximum_production
        ):
            raise ValueError(
                "homeostatic_production must lie inside "
                "production bounds"
            )

        if (
            self.minimum_pretension
            > self.maximum_pretension
        ):
            raise ValueError(
                "minimum_pretension cannot exceed "
                "maximum_pretension"
            )

        if not 0.0 <= self.history_weight <= 1.0:
            raise ValueError(
                "history_weight must be in [0, 1]"
            )

    def snapshot(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


# =============================================================
# Input and output
# =============================================================


@dataclass(frozen=True)
class RemodelingInput:
    """
    Read-only mechanobiological state supplied to the engine.

    stimulus:
        Normalized current mechanical stimulus.

    damage:
        Structural damage in [0, 1].

    fatigue:
        Functional fatigue in [0, 1].

    energy:
        Available biological energy in [0, 1].

    remodeling:
        Current structural capacity multiplier.

    production:
        Current production or synthesis state.

    pretension:
        Current normalized pretension state.

    history:
        Optional previous normalized stimulus values.

    failed:
        Existing irreversible failure flag.
    """

    stimulus: float

    damage: float = 0.0
    fatigue: float = 0.0
    energy: float = 1.0

    remodeling: float = 1.0
    production: float = 1.0
    pretension: float = 0.0

    history: Sequence[float] = field(
        default_factory=tuple
    )

    failed: bool = False

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def validated(
        self,
    ) -> RemodelingInput:
        stimulus = _nonnegative(
            self.stimulus,
            name="RemodelingInput.stimulus",
        )

        damage = _unit_interval(
            self.damage,
            name="RemodelingInput.damage",
        )

        fatigue = _unit_interval(
            self.fatigue,
            name="RemodelingInput.fatigue",
        )

        energy = _unit_interval(
            self.energy,
            name="RemodelingInput.energy",
        )

        remodeling = _nonnegative(
            self.remodeling,
            name="RemodelingInput.remodeling",
        )

        production = _nonnegative(
            self.production,
            name="RemodelingInput.production",
        )

        pretension = _nonnegative(
            self.pretension,
            name="RemodelingInput.pretension",
        )

        history = tuple(
            _nonnegative(
                value,
                name="RemodelingInput.history value",
            )
            for value in self.history
        )

        return RemodelingInput(
            stimulus=stimulus,
            damage=damage,
            fatigue=fatigue,
            energy=energy,
            remodeling=remodeling,
            production=production,
            pretension=pretension,
            history=history,
            failed=bool(self.failed),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class RemodelingResult:
    """
    Result of exactly one biological update.

    Rates are retained for diagnostics and testing.
    """

    raw_stimulus: float
    effective_stimulus: float
    filtered_stimulus: float

    damage: float
    fatigue: float
    energy: float

    remodeling: float
    production: float
    pretension: float

    damage_rate: float
    repair_rate: float
    remodeling_rate: float
    production_rate: float
    pretension_rate: float

    overload: float
    underload: float
    energy_deficit: float

    critical: bool
    failed: bool

    update_count: int

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================
# Internal engine state
# =============================================================


@dataclass
class RemodelingState:
    """
    Internal temporal state of RemodelingEngine.

    Biological material state remains in MaterialState. This class
    stores only quantities required by the remodeling controller.
    """

    filtered_stimulus: float = 0.0
    cumulative_damage_drive: float = 0.0
    cumulative_repair: float = 0.0

    time: float = 0.0
    update_count: int = 0

    last_result: RemodelingResult | None = None

    stimulus_history: list[float] = field(
        default_factory=list
    )

    def clamp(
        self,
        *,
        history_window: int,
    ) -> None:
        self.filtered_stimulus = max(
            0.0,
            _finite(
                self.filtered_stimulus,
                name="RemodelingState.filtered_stimulus",
            ),
        )

        self.cumulative_damage_drive = max(
            0.0,
            _finite(
                self.cumulative_damage_drive,
                name=(
                    "RemodelingState."
                    "cumulative_damage_drive"
                ),
            ),
        )

        self.cumulative_repair = max(
            0.0,
            _finite(
                self.cumulative_repair,
                name="RemodelingState.cumulative_repair",
            ),
        )

        self.time = max(
            0.0,
            _finite(
                self.time,
                name="RemodelingState.time",
            ),
        )

        self.update_count = max(
            0,
            int(self.update_count),
        )

        clean_history = []

        for value in self.stimulus_history:
            clean_history.append(
                max(
                    0.0,
                    _finite(
                        value,
                        name=(
                            "RemodelingState."
                            "stimulus_history value"
                        ),
                    ),
                )
            )

        self.stimulus_history = clean_history[
            -history_window:
        ]

    def reset(self) -> None:
        self.filtered_stimulus = 0.0
        self.cumulative_damage_drive = 0.0
        self.cumulative_repair = 0.0

        self.time = 0.0
        self.update_count = 0

        self.last_result = None
        self.stimulus_history.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "filtered_stimulus": float(
                self.filtered_stimulus
            ),
            "cumulative_damage_drive": float(
                self.cumulative_damage_drive
            ),
            "cumulative_repair": float(
                self.cumulative_repair
            ),
            "time": float(self.time),
            "update_count": int(
                self.update_count
            ),
            "stimulus_history": list(
                self.stimulus_history
            ),
            "last_result": (
                None
                if self.last_result is None
                else self.last_result.snapshot()
            ),
        }


# =============================================================
# Remodeling engine
# =============================================================


@dataclass
class RemodelingEngine:
    """
    Generic mechanobiological adaptation engine.

    Important invariant
    -------------------
    ``step()`` must be called exactly once per physical simulation
    step, after the mechanical solution is complete.

    The engine does not access Nodes, Elements or Materials.
    It receives an immutable RemodelingInput and returns an
    immutable RemodelingResult.
    """

    parameters: RemodelingParameters = field(
        default_factory=RemodelingParameters
    )

    state: RemodelingState = field(
        default_factory=RemodelingState
    )

    def __post_init__(self) -> None:
        self.parameters.validate()

        self.state.clamp(
            history_window=(
                self.parameters.history_window
            )
        )

    # =========================================================
    # Main update
    # =========================================================

    def step(
        self,
        *,
        dt: float,
        input_state: RemodelingInput,
    ) -> RemodelingResult:
        """
        Advance all generic biological processes once.

        Order:

        stimulus filtering
            ↓
        overload and underload calculation
            ↓
        damage and repair
            ↓
        remodeling
            ↓
        production
            ↓
        pretension
            ↓
        fatigue and energy
            ↓
        failure state
        """

        dt = self._validate_dt(dt)
        data = input_state.validated()

        if data.failed:
            result = self._failed_result(
                data=data,
            )

            self._commit(
                dt=dt,
                result=result,
            )

            return result

        effective_stimulus = (
            self.effective_stimulus(
                raw_stimulus=data.stimulus,
                external_history=data.history,
            )
        )

        filtered_stimulus = (
            self.update_filtered_stimulus(
                dt=dt,
                target=effective_stimulus,
            )
        )

        overload = self.overload(
            filtered_stimulus
        )

        underload = self.underload(
            filtered_stimulus
        )

        energy_deficit = float(
            1.0 - data.energy
        )

        damage_rate = (
            self.damage_rate(
                stimulus=filtered_stimulus,
                overload=overload,
                fatigue=data.fatigue,
                energy_deficit=energy_deficit,
                damage=data.damage,
            )
        )

        repair_rate = (
            self.repair_rate(
                stimulus=filtered_stimulus,
                energy=data.energy,
                damage=data.damage,
            )
        )

        new_damage = _clamp01(
            data.damage
            + (
                damage_rate
                - repair_rate
            )
            * dt
        )

        remodeling_rate = (
            self.structural_remodeling_rate(
                stimulus=filtered_stimulus,
                overload=overload,
                underload=underload,
                damage=new_damage,
                energy=data.energy,
                remodeling=data.remodeling,
            )
        )

        new_remodeling = float(
            np.clip(
                data.remodeling
                + remodeling_rate * dt,
                self.parameters.minimum_remodeling,
                self.parameters.maximum_remodeling,
            )
        )

        production_rate = (
            self.material_production_rate(
                stimulus=filtered_stimulus,
                overload=overload,
                underload=underload,
                damage=new_damage,
                energy=data.energy,
                production=data.production,
            )
        )

        new_production = float(
            np.clip(
                data.production
                + production_rate * dt,
                self.parameters.minimum_production,
                self.parameters.maximum_production,
            )
        )

        pretension_rate = (
            self.adaptive_pretension_rate(
                stimulus=filtered_stimulus,
                overload=overload,
                underload=underload,
                damage=new_damage,
                energy=data.energy,
                pretension=data.pretension,
            )
        )

        new_pretension = float(
            np.clip(
                data.pretension
                + pretension_rate * dt,
                self.parameters.minimum_pretension,
                self.parameters.maximum_pretension,
            )
        )

        new_fatigue = self.updated_fatigue(
            dt=dt,
            fatigue=data.fatigue,
            stimulus=filtered_stimulus,
            energy=data.energy,
        )

        new_energy = self.updated_energy(
            dt=dt,
            energy=data.energy,
            stimulus=filtered_stimulus,
            damage=new_damage,
            production=new_production,
        )

        critical = bool(
            filtered_stimulus
            >= self.parameters.critical_stimulus
            or new_damage
            >= self.parameters.critical_damage
        )

        failed = bool(
            new_damage
            >= self.parameters.failure_damage
        )

        if failed:
            new_damage = max(
                new_damage,
                self.parameters.failure_damage,
            )

            new_damage = _clamp01(
                new_damage
            )

        result = RemodelingResult(
            raw_stimulus=float(
                data.stimulus
            ),
            effective_stimulus=float(
                effective_stimulus
            ),
            filtered_stimulus=float(
                filtered_stimulus
            ),
            damage=float(
                new_damage
            ),
            fatigue=float(
                new_fatigue
            ),
            energy=float(
                new_energy
            ),
            remodeling=float(
                new_remodeling
            ),
            production=float(
                new_production
            ),
            pretension=float(
                new_pretension
            ),
            damage_rate=float(
                damage_rate
            ),
            repair_rate=float(
                repair_rate
            ),
            remodeling_rate=float(
                remodeling_rate
            ),
            production_rate=float(
                production_rate
            ),
            pretension_rate=float(
                pretension_rate
            ),
            overload=float(
                overload
            ),
            underload=float(
                underload
            ),
            energy_deficit=float(
                energy_deficit
            ),
            critical=critical,
            failed=failed,
            update_count=(
                self.state.update_count + 1
            ),
        )

        self._commit(
            dt=dt,
            result=result,
        )

        return result

    # =========================================================
    # Stimulus processing
    # =========================================================

    def effective_stimulus(
        self,
        *,
        raw_stimulus: float,
        external_history: Sequence[float] = (),
    ) -> float:
        raw_stimulus = _nonnegative(
            raw_stimulus,
            name="raw_stimulus",
        )

        history = list(
            self.state.stimulus_history
        )

        history.extend(
            max(
                0.0,
                _finite(
                    value,
                    name="external history value",
                ),
            )
            for value in external_history
        )

        if not history:
            return raw_stimulus

        window = history[
            -self.parameters.history_window:
        ]

        history_mean = float(
            np.mean(window)
        )

        weight = (
            self.parameters.history_weight
        )

        return float(
            (
                1.0 - weight
            )
            * raw_stimulus
            + weight
            * history_mean
        )

    def update_filtered_stimulus(
        self,
        *,
        dt: float,
        target: float,
    ) -> float:
        target = _nonnegative(
            target,
            name="target",
        )

        rate = (
            self.parameters.stimulus_filter_rate
        )

        if self.state.update_count == 0:
            return target

        blend = float(
            np.clip(
                rate * dt,
                0.0,
                1.0,
            )
        )

        return float(
            self.state.filtered_stimulus
            + blend
            * (
                target
                - self.state.filtered_stimulus
            )
        )

    def overload(
        self,
        stimulus: float,
    ) -> float:
        stimulus = _nonnegative(
            stimulus,
            name="stimulus",
        )

        return max(
            0.0,
            stimulus
            - self.parameters.overload_threshold,
        )

    def underload(
        self,
        stimulus: float,
    ) -> float:
        stimulus = _nonnegative(
            stimulus,
            name="stimulus",
        )

        return max(
            0.0,
            self.parameters.underload_threshold
            - stimulus,
        )

    # =========================================================
    # Damage and repair
    # =========================================================

    def damage_rate(
        self,
        *,
        stimulus: float,
        overload: float,
        fatigue: float,
        energy_deficit: float,
        damage: float,
    ) -> float:
        if overload <= 0.0:
            return 0.0

        normalized_overload = (
            _safe_ratio(
                overload,
                (
                    self.parameters.critical_stimulus
                    - self.parameters.overload_threshold
                ),
                epsilon=self.parameters.epsilon,
            )
        )

        overload_drive = (
            normalized_overload
            ** self.parameters.overload_damage_exponent
        )

        fatigue_factor = (
            1.0
            + self.parameters.fatigue_damage_multiplier
            * fatigue
        )

        energy_factor = (
            1.0
            + (
                self.parameters
                .energy_deficit_damage_multiplier
                * energy_deficit
            )
        )

        remaining_integrity = max(
            0.0,
            1.0 - damage,
        )

        return float(
            self.parameters.damage_rate
            * overload_drive
            * fatigue_factor
            * energy_factor
            * remaining_integrity
        )

    def repair_rate(
        self,
        *,
        stimulus: float,
        energy: float,
        damage: float,
    ) -> float:
        if damage <= 0.0:
            return 0.0

        energy_capacity = max(
            self.parameters.minimum_repair_capacity,
            energy
            ** self.parameters.repair_energy_exponent,
        )

        load_suppression = (
            1.0
            / (
                1.0
                + self.parameters.repair_load_suppression
                * stimulus
            )
        )

        return float(
            self.parameters.repair_rate
            * energy_capacity
            * load_suppression
            * damage
        )

    # =========================================================
    # Structural remodeling
    # =========================================================

    def remodeling_target(
        self,
        *,
        overload: float,
        underload: float,
    ) -> float:
        target = (
            self.parameters.homeostatic_remodeling
        )

        target += (
            self.parameters.overload_remodeling_gain
            * overload
        )

        target -= (
            self.parameters.underload_remodeling_loss
            * underload
        )

        return float(
            np.clip(
                target,
                self.parameters.minimum_remodeling,
                self.parameters.maximum_remodeling,
            )
        )

    def structural_remodeling_rate(
        self,
        *,
        stimulus: float,
        overload: float,
        underload: float,
        damage: float,
        energy: float,
        remodeling: float,
    ) -> float:
        target = self.remodeling_target(
            overload=overload,
            underload=underload,
        )

        biological_capacity = (
            energy
            * max(
                0.0,
                1.0 - damage,
            )
        )

        adaptation = (
            self.parameters.remodeling_rate
            * biological_capacity
            * (
                target
                - remodeling
            )
        )

        degradation = (
            self.parameters.remodeling_loss_rate
            * (
                damage
                + underload
            )
            * max(
                0.0,
                remodeling
                - self.parameters.minimum_remodeling,
            )
        )

        return float(
            adaptation - degradation
        )

    # =========================================================
    # Production
    # =========================================================

    def production_target(
        self,
        *,
        overload: float,
        underload: float,
        damage: float,
    ) -> float:
        target = (
            self.parameters.homeostatic_production
        )

        target += (
            self.parameters.production_stimulus_gain
            * overload
        )

        target += (
            self.parameters.production_damage_gain
            * damage
        )

        target -= (
            self.parameters.production_stimulus_gain
            * underload
        )

        return float(
            np.clip(
                target,
                self.parameters.minimum_production,
                self.parameters.maximum_production,
            )
        )

    def material_production_rate(
        self,
        *,
        stimulus: float,
        overload: float,
        underload: float,
        damage: float,
        energy: float,
        production: float,
    ) -> float:
        target = self.production_target(
            overload=overload,
            underload=underload,
            damage=damage,
        )

        energy_capacity = (
            energy
            ** self.parameters.production_energy_exponent
        )

        synthesis = (
            self.parameters.production_rate
            * energy_capacity
            * (
                target
                - production
            )
        )

        degradation = (
            self.parameters.degradation_rate
            * (
                damage
                + underload
            )
            * max(
                0.0,
                production
                - self.parameters.minimum_production,
            )
        )

        return float(
            synthesis - degradation
        )

    # =========================================================
    # Pretension
    # =========================================================

    def pretension_target(
        self,
        *,
        overload: float,
        underload: float,
        damage: float,
        energy: float,
    ) -> float:
        target = (
            self.parameters.minimum_pretension
        )

        target += (
            self.parameters.pretension_stimulus_gain
            * overload
        )

        target -= (
            self.parameters.pretension_stimulus_gain
            * underload
        )

        target *= (
            max(
                0.0,
                1.0
                - (
                    self.parameters
                    .pretension_damage_suppression
                    * damage
                ),
            )
            * energy
        )

        return float(
            np.clip(
                target,
                self.parameters.minimum_pretension,
                self.parameters.maximum_pretension,
            )
        )

    def adaptive_pretension_rate(
        self,
        *,
        stimulus: float,
        overload: float,
        underload: float,
        damage: float,
        energy: float,
        pretension: float,
    ) -> float:
        target = self.pretension_target(
            overload=overload,
            underload=underload,
            damage=damage,
            energy=energy,
        )

        adaptation = (
            self.parameters.pretension_rate
            * (
                target
                - pretension
            )
        )

        decay = (
            self.parameters.pretension_decay_rate
            * underload
            * max(
                0.0,
                pretension
                - self.parameters.minimum_pretension,
            )
        )

        return float(
            adaptation - decay
        )

    # =========================================================
    # Fatigue and energy
    # =========================================================

    def updated_fatigue(
        self,
        *,
        dt: float,
        fatigue: float,
        stimulus: float,
        energy: float,
    ) -> float:
        fatigue_gain = (
            self.parameters.fatigue_stimulus_rate
            * stimulus
            * (
                1.0 + (
                    1.0 - energy
                )
            )
        )

        fatigue_recovery = (
            self.parameters.fatigue_recovery_rate
            * energy
            * fatigue
        )

        return _clamp01(
            fatigue
            + (
                fatigue_gain
                - fatigue_recovery
            )
            * dt
        )

    def updated_energy(
        self,
        *,
        dt: float,
        energy: float,
        stimulus: float,
        damage: float,
        production: float,
    ) -> float:
        consumption = (
            self.parameters.energy_consumption_rate
            * (
                stimulus
                + damage
                + production
            )
        )

        recovery = (
            self.parameters.energy_recovery_rate
            * (
                1.0 - energy
            )
        )

        return _clamp01(
            energy
            + (
                recovery
                - consumption
            )
            * dt
        )

    # =========================================================
    # Failure handling
    # =========================================================

    def _failed_result(
        self,
        *,
        data: RemodelingInput,
    ) -> RemodelingResult:
        return RemodelingResult(
            raw_stimulus=float(
                data.stimulus
            ),
            effective_stimulus=float(
                data.stimulus
            ),
            filtered_stimulus=float(
                data.stimulus
            ),
            damage=float(
                max(
                    data.damage,
                    self.parameters.failure_damage,
                )
            ),
            fatigue=float(
                data.fatigue
            ),
            energy=float(
                data.energy
            ),
            remodeling=float(
                data.remodeling
            ),
            production=float(
                data.production
            ),
            pretension=float(
                data.pretension
            ),
            damage_rate=0.0,
            repair_rate=0.0,
            remodeling_rate=0.0,
            production_rate=0.0,
            pretension_rate=0.0,
            overload=float(
                self.overload(
                    data.stimulus
                )
            ),
            underload=float(
                self.underload(
                    data.stimulus
                )
            ),
            energy_deficit=float(
                1.0 - data.energy
            ),
            critical=True,
            failed=True,
            update_count=(
                self.state.update_count + 1
            ),
        )

    # =========================================================
    # Commit
    # =========================================================

    def _commit(
        self,
        *,
        dt: float,
        result: RemodelingResult,
    ) -> None:
        self.state.filtered_stimulus = (
            result.filtered_stimulus
        )

        self.state.cumulative_damage_drive += (
            result.damage_rate * dt
        )

        self.state.cumulative_repair += (
            result.repair_rate * dt
        )

        self.state.time += dt
        self.state.update_count += 1

        self.state.stimulus_history.append(
            result.raw_stimulus
        )

        self.state.stimulus_history = (
            self.state.stimulus_history[
                -self.parameters.history_window:
            ]
        )

        self.state.last_result = result

        self.state.clamp(
            history_window=(
                self.parameters.history_window
            )
        )

    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def _validate_dt(
        dt: float,
    ) -> float:
        dt = _finite(
            dt,
            name="dt",
        )

        if dt <= 0.0:
            raise ValueError(
                "dt must be positive"
            )

        return dt

    # =========================================================
    # State management
    # =========================================================

    def reset(self) -> None:
        self.state.reset()

    def snapshot(self) -> dict[str, Any]:
        return {
            "parameters": (
                self.parameters.snapshot()
            ),
            "state": (
                self.state.snapshot()
            ),
        }

    def __repr__(self) -> str:
        result = self.state.last_result

        if result is None:
            return (
                "RemodelingEngine("
                "updates=0, "
                "stimulus=0.000, "
                "damage=unknown)"
            )

        return (
            "RemodelingEngine("
            f"updates={self.state.update_count}, "
            f"stimulus={result.filtered_stimulus:.3f}, "
            f"damage={result.damage:.3f}, "
            f"remodeling={result.remodeling:.3f}, "
            f"failed={result.failed})"
        )