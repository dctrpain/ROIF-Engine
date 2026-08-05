"""
ROIF Engine
Universal Material Representation Layer

This module connects the universal ROIF entity/influence model with material
behaviour.

A material is not restricted to biological tissue. The same representation
can describe:

- metals, ceramics, glass, concrete, timber, polymers, elastomers;
- fibre, particle, and layered composites;
- cables, membranes, foams, porous media, fluids, and granular media;
- muscle, tendon, ligament, fascia, bone, nerve, vessel, and organ tissue;
- software, information, and organisational substrates through custom laws.

The existing ``core.material.Material`` remains the axial constitutive solver
used by the mechanical engine. This module does not replace it. It provides a
domain-independent ROIF description that can:

1. classify a material and its active constitutive laws;
2. store stress, strain, integrity, fatigue, damage, creep, plasticity,
   anisotropy, temperature, moisture, corrosion, and biological adaptation;
3. evaluate simplified constitutive responses for ROIF cascade reasoning;
4. expose material factors to InfluencePlane and InfluenceOperator logic;
5. translate snapshots from ``core.material.Material`` without requiring the
   ROIF layer to depend on a particular material subclass;
6. identify failure modes and persistent structural traces.

The formulas implemented here are explicit research-prototype laws. They are
not substitutes for a validated finite-element constitutive model.

Author:
    Architect (Dctr Pain)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
import math

from .roif_entities import Vector3


class ROIFMaterialError(ValueError):
    """Raised when a ROIF material object violates an invariant."""


class MaterialFamily(str, Enum):
    """Broad material family, independent of geometry or element type."""

    GENERIC = "generic"

    METAL = "metal"
    CERAMIC = "ceramic"
    GLASS = "glass"
    CONCRETE = "concrete"
    MASONRY = "masonry"
    TIMBER = "timber"
    POLYMER = "polymer"
    ELASTOMER = "elastomer"
    FOAM = "foam"
    TEXTILE = "textile"
    CABLE = "cable"
    MEMBRANE = "membrane"
    GRANULAR = "granular"
    POROUS_MEDIUM = "porous_medium"
    FLUID = "fluid"
    GAS = "gas"

    COMPOSITE = "composite"
    FIBRE_COMPOSITE = "fibre_composite"
    PARTICLE_COMPOSITE = "particle_composite"
    LAYERED_COMPOSITE = "layered_composite"

    BIOLOGICAL_TISSUE = "biological_tissue"
    MUSCLE = "muscle"
    TENDON = "tendon"
    LIGAMENT = "ligament"
    FASCIA = "fascia"
    BONE = "bone"
    CARTILAGE = "cartilage"
    NERVE = "nerve"
    VESSEL = "vessel"
    ORGAN_TISSUE = "organ_tissue"

    SOFTWARE = "software"
    INFORMATION = "information"
    ORGANISATIONAL = "organisational"
    CUSTOM = "custom"


class ConstitutiveLawKind(str, Enum):
    """Canonical constitutive response categories."""

    LINEAR_ELASTIC = "linear_elastic"
    NONLINEAR_ELASTIC = "nonlinear_elastic"
    HYPERELASTIC = "hyperelastic"
    VISCOELASTIC = "viscoelastic"
    STANDARD_LINEAR_SOLID = "standard_linear_solid"
    KELVIN_VOIGT = "kelvin_voigt"
    MAXWELL = "maxwell"
    ELASTOPLASTIC = "elastoplastic"
    VISCOPLASTIC = "viscoplastic"
    PERFECTLY_PLASTIC = "perfectly_plastic"
    DAMAGE = "damage"
    FATIGUE = "fatigue"
    CREEP = "creep"
    STRESS_RELAXATION = "stress_relaxation"
    POROELASTIC = "poroelastic"
    FRICTIONAL = "frictional"
    GRANULAR = "granular"
    ACTIVE_CONTRACTILE = "active_contractile"
    REMODELING = "remodeling"
    GROWTH = "growth"
    AGING = "aging"
    CORROSION = "corrosion"
    THERMAL_EXPANSION = "thermal_expansion"
    MOISTURE_SWELLING = "moisture_swelling"
    CUSTOM = "custom"


class ResponseMode(str, Enum):
    TENSION = "tension"
    COMPRESSION = "compression"
    SHEAR = "shear"
    BENDING = "bending"
    TORSION = "torsion"
    HYDROSTATIC = "hydrostatic"
    VOLUMETRIC = "volumetric"
    ACTIVE = "active"
    INFORMATIONAL = "informational"
    CUSTOM = "custom"


class FailureMode(str, Enum):
    NONE = "none"
    TENSILE_RUPTURE = "tensile_rupture"
    COMPRESSIVE_CRUSHING = "compressive_crushing"
    SHEAR_FAILURE = "shear_failure"
    BUCKLING = "buckling"
    FATIGUE_CRACK = "fatigue_crack"
    BRITTLE_FRACTURE = "brittle_fracture"
    DUCTILE_YIELD = "ductile_yield"
    DELAMINATION = "delamination"
    DEBONDING = "debonding"
    FIBRE_BREAKAGE = "fibre_breakage"
    MATRIX_CRACKING = "matrix_cracking"
    CREEP_RUPTURE = "creep_rupture"
    CORROSION_LOSS = "corrosion_loss"
    THERMAL_DAMAGE = "thermal_damage"
    MOISTURE_DAMAGE = "moisture_damage"
    CAVITATION = "cavitation"
    TEARING = "tearing"
    BIOLOGICAL_FAILURE = "biological_failure"
    CONTROL_FAILURE = "control_failure"
    CUSTOM = "custom"


class MaterialPhase(str, Enum):
    INSTANTANEOUS = "instantaneous"
    LOADING = "loading"
    HOLD = "hold"
    UNLOADING = "unloading"
    RECOVERY = "recovery"
    RESIDUAL = "residual"
    FAILED = "failed"


class AnisotropyKind(str, Enum):
    ISOTROPIC = "isotropic"
    TRANSVERSE_ISOTROPIC = "transverse_isotropic"
    ORTHOTROPIC = "orthotropic"
    FULLY_ANISOTROPIC = "fully_anisotropic"
    EVOLVING = "evolving"


class MaterialEffect(str, Enum):
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"
    MIXED = "mixed"
    UNKNOWN = "unknown"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFMaterialError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFMaterialError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ROIFMaterialError(f"{name} must be finite.")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ROIFMaterialError(f"{name} must be positive.")
    return result


def _nonnegative_or_positive_infinity(
    value: Any,
    name: str,
) -> float:
    """
    Validate a nonnegative scalar or positive infinity.

    ``math.inf`` is used as the sentinel for an unspecified material
    strength or failure limit.
    """

    if isinstance(value, bool):
        raise ROIFMaterialError(f"{name} must be numeric.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFMaterialError(
            f"{name} must be numeric."
        ) from exc

    if math.isnan(result) or result == -math.inf:
        raise ROIFMaterialError(
            f"{name} must be nonnegative or positive infinity."
        )

    if result < 0.0:
        raise ROIFMaterialError(
            f"{name} cannot be negative."
        )

    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ROIFMaterialError(f"{name} cannot be negative.")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ROIFMaterialError(f"{name} must be within [0, 1].")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFMaterialError(f"{name} must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFMaterialError("metadata must be a mapping.")
    return MappingProxyType(dict(value))


def _float_mapping(
    value: Mapping[str, Any] | None,
    name: str,
) -> Mapping[str, float]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ROIFMaterialError(f"{name} must be a mapping.")
    return MappingProxyType(
        {
            str(key): _finite(item, f"{name}[{key!r}]")
            for key, item in value.items()
        }
    )


@dataclass(frozen=True, slots=True)
class MaterialDescriptor:
    """Stable material identity and broad classification."""

    material_id: str
    name: str
    family: MaterialFamily = MaterialFamily.GENERIC
    anisotropy: AnisotropyKind = AnisotropyKind.ISOTROPIC
    active_material: bool = False
    porous: bool = False
    composite: bool = False
    biological: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "material_id",
            _text(self.material_id, "material_id"),
        )
        object.__setattr__(self, "name", _text(self.name, "name"))
        if not isinstance(self.family, MaterialFamily):
            raise ROIFMaterialError("family must be MaterialFamily.")
        if not isinstance(self.anisotropy, AnisotropyKind):
            raise ROIFMaterialError(
                "anisotropy must be AnisotropyKind."
            )
        for name in (
            "active_material",
            "porous",
            "composite",
            "biological",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ROIFMaterialError(f"{name} must be bool.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MaterialParameters:
    """
    Universal scalar parameters used by reduced-order ROIF material laws.

    Domain adapters may add specialised coefficients through ``extra``.
    """

    density: float = 1.0
    elastic_modulus: float = 1.0
    relaxed_modulus: float | None = None
    shear_modulus: float = 1.0
    bulk_modulus: float = 1.0
    damping: float = 0.0
    poisson_ratio: float = 0.0

    yield_stress: float = math.inf
    tensile_strength: float = math.inf
    compressive_strength: float = math.inf
    shear_strength: float = math.inf
    failure_strain: float = math.inf

    thermal_expansion: float = 0.0
    moisture_expansion: float = 0.0
    reference_temperature: float = 20.0
    reference_moisture: float = 0.0

    creep_time_constant: float = 1.0
    relaxation_time_constant: float = 1.0
    viscosity: float = 0.0

    fatigue_rate: float = 0.0
    damage_rate: float = 0.0
    recovery_rate: float = 0.0
    remodeling_rate: float = 0.0
    corrosion_rate: float = 0.0
    aging_rate: float = 0.0

    active_stress_scale: float = 0.0
    activation_time_constant: float = 1.0

    reference_stress: float = 1.0
    reference_strain: float = 1.0

    extra: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive = (
            "density",
            "elastic_modulus",
            "shear_modulus",
            "bulk_modulus",
            "creep_time_constant",
            "relaxation_time_constant",
            "activation_time_constant",
            "reference_stress",
            "reference_strain",
        )
        for name in positive:
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), name),
            )

        if self.relaxed_modulus is not None:
            relaxed = _positive(
                self.relaxed_modulus,
                "relaxed_modulus",
            )
            if relaxed > self.elastic_modulus:
                raise ROIFMaterialError(
                    "relaxed_modulus cannot exceed elastic_modulus."
                )
            object.__setattr__(
                self,
                "relaxed_modulus",
                relaxed,
            )

        unlimited_nonnegative = (
            "yield_stress",
            "tensile_strength",
            "compressive_strength",
            "shear_strength",
            "failure_strain",
        )

        for name in unlimited_nonnegative:
            object.__setattr__(
                self,
                name,
                _nonnegative_or_positive_infinity(
                    getattr(self, name),
                    name,
                ),
            )

        nonnegative = (
            "damping",
            "thermal_expansion",
            "moisture_expansion",
            "reference_moisture",
            "viscosity",
            "fatigue_rate",
            "damage_rate",
            "recovery_rate",
            "remodeling_rate",
            "corrosion_rate",
            "aging_rate",
            "active_stress_scale",
        )
        for name in nonnegative:
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )

        poisson = _finite(self.poisson_ratio, "poisson_ratio")
        if not -1.0 < poisson < 0.5:
            raise ROIFMaterialError(
                "poisson_ratio must be within (-1, 0.5)."
            )
        object.__setattr__(self, "poisson_ratio", poisson)
        object.__setattr__(
            self,
            "reference_temperature",
            _finite(
                self.reference_temperature,
                "reference_temperature",
            ),
        )
        object.__setattr__(self, "extra", _float_mapping(self.extra, "extra"))


@dataclass(frozen=True, slots=True)
class MaterialState:
    """
    State shared by biological and non-biological materials.

    Stress and strain are scalar reduced-order values. ``direction`` and
    ``directional_state`` preserve anisotropic context for a future tensor
    constitutive solver.
    """

    time: float = 0.0
    phase: MaterialPhase = MaterialPhase.INSTANTANEOUS

    stress: float = 0.0
    strain: float = 0.0
    strain_rate: float = 0.0
    elastic_strain: float = 0.0
    plastic_strain: float = 0.0
    creep_strain: float = 0.0
    residual_strain: float = 0.0
    residual_stress: float = 0.0

    temperature: float = 20.0
    moisture: float = 0.0
    pressure: float = 0.0

    integrity: float = 1.0
    damage: float = 0.0
    fatigue: float = 0.0
    corrosion: float = 0.0
    aging: float = 0.0
    remodeling: float = 0.0
    activation: float = 0.0
    history: float = 0.0
    energy: float = 0.0
    dissipated_energy: float = 0.0

    direction: Vector3 = field(
        default_factory=lambda: Vector3(1.0, 0.0, 0.0)
    )
    directional_state: Mapping[str, float] = field(
        default_factory=dict
    )

    failed: bool = False
    failure_mode: FailureMode = FailureMode.NONE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _nonnegative(self.time, "time"))
        if not isinstance(self.phase, MaterialPhase):
            raise ROIFMaterialError("phase must be MaterialPhase.")

        signed = (
            "stress",
            "strain",
            "strain_rate",
            "elastic_strain",
            "plastic_strain",
            "creep_strain",
            "residual_strain",
            "residual_stress",
            "temperature",
            "pressure",
        )
        for name in signed:
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )

        object.__setattr__(
            self,
            "moisture",
            _nonnegative(self.moisture, "moisture"),
        )

        unit_values = (
            "integrity",
            "damage",
            "fatigue",
            "corrosion",
            "aging",
            "remodeling",
            "activation",
        )
        for name in unit_values:
            object.__setattr__(
                self,
                name,
                _unit(getattr(self, name), name),
            )

        for name in (
            "history",
            "energy",
            "dissipated_energy",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )

        if not isinstance(self.direction, Vector3):
            raise ROIFMaterialError("direction must be Vector3.")
        object.__setattr__(
            self,
            "directional_state",
            _float_mapping(
                self.directional_state,
                "directional_state",
            ),
        )
        if not isinstance(self.failed, bool):
            raise ROIFMaterialError("failed must be bool.")
        if not isinstance(self.failure_mode, FailureMode):
            raise ROIFMaterialError(
                "failure_mode must be FailureMode."
            )
        if self.failed and self.failure_mode is FailureMode.NONE:
            raise ROIFMaterialError(
                "failed material must specify a failure_mode."
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def recoverable_strain(self) -> float:
        return self.elastic_strain + self.creep_strain - self.residual_strain

    @property
    def retained_fraction(self) -> float:
        total = abs(self.strain)
        if total <= 1e-12:
            return 0.0
        return min(1.0, abs(self.residual_strain) / total)

    @property
    def available_capacity(self) -> float:
        return max(
            0.0,
            self.integrity
            * (1.0 - self.damage)
            * (1.0 - self.fatigue)
            * (1.0 - self.corrosion)
            * (1.0 - self.aging),
        )


@dataclass(frozen=True, slots=True)
class ConstitutiveLaw:
    """One active material law and its weighting in a hybrid model."""

    law_id: str
    kind: ConstitutiveLawKind
    weight: float = 1.0
    response_modes: tuple[ResponseMode, ...] = (
        ResponseMode.TENSION,
        ResponseMode.COMPRESSION,
    )
    active: bool = True
    parameters: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "law_id", _text(self.law_id, "law_id"))
        if not isinstance(self.kind, ConstitutiveLawKind):
            raise ROIFMaterialError(
                "kind must be ConstitutiveLawKind."
            )
        object.__setattr__(
            self,
            "weight",
            _nonnegative(self.weight, "weight"),
        )
        modes = tuple(self.response_modes)
        if not modes:
            raise ROIFMaterialError(
                "response_modes cannot be empty."
            )
        if any(not isinstance(mode, ResponseMode) for mode in modes):
            raise ROIFMaterialError(
                "response_modes must contain ResponseMode values."
            )
        object.__setattr__(self, "response_modes", modes)
        if not isinstance(self.active, bool):
            raise ROIFMaterialError("active must be bool.")
        object.__setattr__(
            self,
            "parameters",
            _float_mapping(self.parameters, "parameters"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MaterialModel:
    """Complete universal material definition."""

    descriptor: MaterialDescriptor
    parameters: MaterialParameters = field(
        default_factory=MaterialParameters
    )
    laws: tuple[ConstitutiveLaw, ...] = (
        ConstitutiveLaw(
            law_id="linear_elastic",
            kind=ConstitutiveLawKind.LINEAR_ELASTIC,
        ),
    )
    state: MaterialState = field(default_factory=MaterialState)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, MaterialDescriptor):
            raise ROIFMaterialError(
                "descriptor must be MaterialDescriptor."
            )
        if not isinstance(self.parameters, MaterialParameters):
            raise ROIFMaterialError(
                "parameters must be MaterialParameters."
            )
        laws = tuple(self.laws)
        if not laws:
            raise ROIFMaterialError("laws cannot be empty.")
        if len({law.law_id for law in laws}) != len(laws):
            raise ROIFMaterialError("law_id values must be unique.")
        object.__setattr__(self, "laws", laws)
        if not isinstance(self.state, MaterialState):
            raise ROIFMaterialError("state must be MaterialState.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def law(self, law_id: str) -> ConstitutiveLaw:
        for law in self.laws:
            if law.law_id == law_id:
                return law
        raise KeyError(law_id)


@dataclass(frozen=True, slots=True)
class MaterialInput:
    """Input for one reduced-order material evaluation."""

    strain: float
    strain_rate: float = 0.0
    delta_time: float = 1.0
    temperature: float | None = None
    moisture: float | None = None
    pressure: float | None = None
    activation: float | None = None
    response_mode: ResponseMode = ResponseMode.TENSION
    direction: Vector3 = field(
        default_factory=lambda: Vector3(1.0, 0.0, 0.0)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strain", _finite(self.strain, "strain"))
        object.__setattr__(
            self,
            "strain_rate",
            _finite(self.strain_rate, "strain_rate"),
        )
        object.__setattr__(
            self,
            "delta_time",
            _positive(self.delta_time, "delta_time"),
        )
        for name in ("temperature", "moisture", "pressure"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite(value, name),
                )
        if self.activation is not None:
            object.__setattr__(
                self,
                "activation",
                _unit(self.activation, "activation"),
            )
        if not isinstance(self.response_mode, ResponseMode):
            raise ROIFMaterialError(
                "response_mode must be ResponseMode."
            )
        if not isinstance(self.direction, Vector3):
            raise ROIFMaterialError("direction must be Vector3.")
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class LawContribution:
    """Contribution of one constitutive law to the total response."""

    law_id: str
    kind: ConstitutiveLawKind
    stress: float
    tangent_modulus: float
    dissipated_energy: float = 0.0
    state_delta: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "law_id", _text(self.law_id, "law_id"))
        if not isinstance(self.kind, ConstitutiveLawKind):
            raise ROIFMaterialError(
                "kind must be ConstitutiveLawKind."
            )
        for name in (
            "stress",
            "tangent_modulus",
            "dissipated_energy",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        if self.dissipated_energy < 0.0:
            raise ROIFMaterialError(
                "dissipated_energy cannot be negative."
            )
        object.__setattr__(
            self,
            "state_delta",
            _float_mapping(self.state_delta, "state_delta"),
        )


@dataclass(frozen=True, slots=True)
class MaterialResponse:
    """Evaluated material response and updated immutable state."""

    material_id: str
    stress: float
    tangent_modulus: float
    force_factor: float
    state_before: MaterialState
    state_after: MaterialState
    contributions: tuple[LawContribution, ...]
    failure_modes: tuple[FailureMode, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "material_id",
            _text(self.material_id, "material_id"),
        )
        for name in (
            "stress",
            "tangent_modulus",
            "force_factor",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "contributions",
            tuple(self.contributions),
        )
        modes = tuple(self.failure_modes)
        if any(not isinstance(mode, FailureMode) for mode in modes):
            raise ROIFMaterialError(
                "failure_modes must contain FailureMode values."
            )
        object.__setattr__(self, "failure_modes", modes)
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MaterialTrace:
    """Persistent or partially persistent structural material trace."""

    trace_id: str
    material_id: str
    time: float
    kind: ConstitutiveLawKind
    magnitude: float
    retained_fraction: float
    reversibility: float
    capacity_effect: float = 0.0
    stiffness_effect: float = 0.0
    geometry_effect: float = 0.0
    effect: MaterialEffect = MaterialEffect.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _text(self.trace_id, "trace_id"))
        object.__setattr__(
            self,
            "material_id",
            _text(self.material_id, "material_id"),
        )
        object.__setattr__(self, "time", _nonnegative(self.time, "time"))
        if not isinstance(self.kind, ConstitutiveLawKind):
            raise ROIFMaterialError(
                "kind must be ConstitutiveLawKind."
            )
        object.__setattr__(
            self,
            "magnitude",
            _finite(self.magnitude, "magnitude"),
        )
        object.__setattr__(
            self,
            "retained_fraction",
            _unit(self.retained_fraction, "retained_fraction"),
        )
        object.__setattr__(
            self,
            "reversibility",
            _unit(self.reversibility, "reversibility"),
        )
        for name in (
            "capacity_effect",
            "stiffness_effect",
            "geometry_effect",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        if not isinstance(self.effect, MaterialEffect):
            raise ROIFMaterialError(
                "effect must be MaterialEffect."
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@runtime_checkable
class SnapshotProvider(Protocol):
    """Protocol implemented by ``core.material.Material`` and adapters."""

    def snapshot(self) -> Mapping[str, Any]:
        ...


def _law_parameter(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    name: str,
    default: float,
) -> float:
    if name in law.parameters:
        return law.parameters[name]
    if name in parameters.extra:
        return parameters.extra[name]
    return default


def _linear_elastic(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    modulus = _law_parameter(
        law,
        parameters,
        "elastic_modulus",
        parameters.elastic_modulus,
    )
    stress = modulus * material_input.strain
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * stress,
        tangent_modulus=law.weight * modulus,
    )


def _nonlinear_elastic(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    modulus = _law_parameter(
        law,
        parameters,
        "elastic_modulus",
        parameters.elastic_modulus,
    )
    exponent = _law_parameter(law, parameters, "exponent", 2.0)
    strain = material_input.strain
    stress = modulus * math.copysign(abs(strain) ** exponent, strain)
    tangent = (
        modulus
        * exponent
        * abs(strain) ** max(exponent - 1.0, 0.0)
    )
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * stress,
        tangent_modulus=law.weight * tangent,
    )


def _viscoelastic(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    modulus = _law_parameter(
        law,
        parameters,
        "elastic_modulus",
        parameters.elastic_modulus,
    )
    viscosity = _law_parameter(
        law,
        parameters,
        "viscosity",
        parameters.viscosity or parameters.damping,
    )
    elastic = modulus * material_input.strain
    viscous = viscosity * material_input.strain_rate
    dissipated = (
        abs(viscous * material_input.strain_rate)
        * material_input.delta_time
    )
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * (elastic + viscous),
        tangent_modulus=law.weight * modulus,
        dissipated_energy=law.weight * dissipated,
    )


def _standard_linear_solid(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    instant = parameters.elastic_modulus
    relaxed = (
        parameters.relaxed_modulus
        if parameters.relaxed_modulus is not None
        else instant
    )
    tau = parameters.creep_time_constant
    alpha = 1.0 - math.exp(-material_input.delta_time / tau)
    effective = instant + (relaxed - instant) * alpha
    stress = effective * (
        material_input.strain - state.creep_strain
    )
    creep_increment = (
        material_input.strain - state.creep_strain
    ) * alpha
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * stress,
        tangent_modulus=law.weight * effective,
        state_delta={
            "creep_strain": law.weight * creep_increment,
        },
    )


def _elastoplastic(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    modulus = parameters.elastic_modulus
    trial = modulus * (
        material_input.strain - state.plastic_strain
    )
    yield_stress = parameters.yield_stress
    if abs(trial) <= yield_stress:
        return LawContribution(
            law_id=law.law_id,
            kind=law.kind,
            stress=law.weight * trial,
            tangent_modulus=law.weight * modulus,
        )

    hardening = _law_parameter(law, parameters, "hardening_modulus", 0.0)
    plastic_increment = (
        abs(trial) - yield_stress
    ) / max(modulus + hardening, 1e-12)
    stress = math.copysign(
        yield_stress + hardening * plastic_increment,
        trial,
    )
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * stress,
        tangent_modulus=law.weight * hardening,
        state_delta={
            "plastic_strain": (
                law.weight
                * math.copysign(plastic_increment, trial)
            )
        },
    )


def _active_contractile(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    activation = (
        material_input.activation
        if material_input.activation is not None
        else state.activation
    )
    length_optimum = _law_parameter(
        law,
        parameters,
        "optimal_strain",
        0.0,
    )
    width = max(
        _law_parameter(law, parameters, "length_width", 1.0),
        1e-12,
    )
    length_factor = math.exp(
        -((material_input.strain - length_optimum) / width) ** 2
    )
    velocity_factor = 1.0 / (
        1.0
        + abs(material_input.strain_rate)
        * _law_parameter(
            law,
            parameters,
            "velocity_sensitivity",
            1.0,
        )
    )
    stress = (
        parameters.active_stress_scale
        * activation
        * length_factor
        * velocity_factor
    )
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * stress,
        tangent_modulus=0.0,
    )


def _thermal_expansion(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    temperature = (
        material_input.temperature
        if material_input.temperature is not None
        else state.temperature
    )
    thermal_strain = (
        parameters.thermal_expansion
        * (temperature - parameters.reference_temperature)
    )
    stress = parameters.elastic_modulus * (
        material_input.strain - thermal_strain
    )
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * stress,
        tangent_modulus=law.weight * parameters.elastic_modulus,
        state_delta={"thermal_strain": thermal_strain},
    )


def _moisture_swelling(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    moisture = (
        material_input.moisture
        if material_input.moisture is not None
        else state.moisture
    )
    swelling_strain = (
        parameters.moisture_expansion
        * (moisture - parameters.reference_moisture)
    )
    stress = parameters.elastic_modulus * (
        material_input.strain - swelling_strain
    )
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight * stress,
        tangent_modulus=law.weight * parameters.elastic_modulus,
        state_delta={"swelling_strain": swelling_strain},
    )


def _damage_contribution(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    stimulus = abs(material_input.strain) / parameters.reference_strain
    increment = (
        parameters.damage_rate
        * max(stimulus - 1.0, 0.0)
        * material_input.delta_time
    )
    degraded_modulus = parameters.elastic_modulus * (
        1.0 - state.damage
    )
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight
        * degraded_modulus
        * material_input.strain,
        tangent_modulus=law.weight * degraded_modulus,
        state_delta={"damage": law.weight * increment},
    )


def _fatigue_contribution(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    stimulus = abs(material_input.strain) / parameters.reference_strain
    increment = (
        parameters.fatigue_rate
        * stimulus
        * material_input.delta_time
    )
    factor = max(0.0, 1.0 - state.fatigue)
    return LawContribution(
        law_id=law.law_id,
        kind=law.kind,
        stress=law.weight
        * parameters.elastic_modulus
        * factor
        * material_input.strain,
        tangent_modulus=law.weight
        * parameters.elastic_modulus
        * factor,
        state_delta={"fatigue": law.weight * increment},
    )


def evaluate_law(
    law: ConstitutiveLaw,
    parameters: MaterialParameters,
    state: MaterialState,
    material_input: MaterialInput,
) -> LawContribution:
    """Evaluate one reduced-order constitutive law."""

    if not law.active:
        return LawContribution(
            law_id=law.law_id,
            kind=law.kind,
            stress=0.0,
            tangent_modulus=0.0,
        )
    if material_input.response_mode not in law.response_modes:
        return LawContribution(
            law_id=law.law_id,
            kind=law.kind,
            stress=0.0,
            tangent_modulus=0.0,
        )

    kind = law.kind
    if kind is ConstitutiveLawKind.LINEAR_ELASTIC:
        return _linear_elastic(law, parameters, state, material_input)
    if kind in (
        ConstitutiveLawKind.NONLINEAR_ELASTIC,
        ConstitutiveLawKind.HYPERELASTIC,
    ):
        return _nonlinear_elastic(
            law,
            parameters,
            state,
            material_input,
        )
    if kind in (
        ConstitutiveLawKind.VISCOELASTIC,
        ConstitutiveLawKind.KELVIN_VOIGT,
        ConstitutiveLawKind.MAXWELL,
        ConstitutiveLawKind.STRESS_RELAXATION,
    ):
        return _viscoelastic(
            law,
            parameters,
            state,
            material_input,
        )
    if kind is ConstitutiveLawKind.STANDARD_LINEAR_SOLID:
        return _standard_linear_solid(
            law,
            parameters,
            state,
            material_input,
        )
    if kind in (
        ConstitutiveLawKind.ELASTOPLASTIC,
        ConstitutiveLawKind.VISCOPLASTIC,
        ConstitutiveLawKind.PERFECTLY_PLASTIC,
    ):
        return _elastoplastic(
            law,
            parameters,
            state,
            material_input,
        )
    if kind is ConstitutiveLawKind.ACTIVE_CONTRACTILE:
        return _active_contractile(
            law,
            parameters,
            state,
            material_input,
        )
    if kind is ConstitutiveLawKind.THERMAL_EXPANSION:
        return _thermal_expansion(
            law,
            parameters,
            state,
            material_input,
        )
    if kind is ConstitutiveLawKind.MOISTURE_SWELLING:
        return _moisture_swelling(
            law,
            parameters,
            state,
            material_input,
        )
    if kind is ConstitutiveLawKind.DAMAGE:
        return _damage_contribution(
            law,
            parameters,
            state,
            material_input,
        )
    if kind is ConstitutiveLawKind.FATIGUE:
        return _fatigue_contribution(
            law,
            parameters,
            state,
            material_input,
        )
    if kind in (
        ConstitutiveLawKind.CREEP,
        ConstitutiveLawKind.REMODELING,
        ConstitutiveLawKind.GROWTH,
        ConstitutiveLawKind.AGING,
        ConstitutiveLawKind.CORROSION,
        ConstitutiveLawKind.POROELASTIC,
        ConstitutiveLawKind.FRICTIONAL,
        ConstitutiveLawKind.GRANULAR,
        ConstitutiveLawKind.CUSTOM,
    ):
        modulus_factor = _law_parameter(
            law,
            parameters,
            "modulus_factor",
            1.0,
        )
        modulus = parameters.elastic_modulus * modulus_factor
        return LawContribution(
            law_id=law.law_id,
            kind=law.kind,
            stress=law.weight * modulus * material_input.strain,
            tangent_modulus=law.weight * modulus,
        )

    raise ROIFMaterialError(
        f"unsupported constitutive law {kind.value!r}."
    )


def _updated_state(
    state: MaterialState,
    material_input: MaterialInput,
    contributions: Sequence[LawContribution],
    stress: float,
) -> MaterialState:
    delta: dict[str, float] = {}
    dissipated = 0.0
    for contribution in contributions:
        dissipated += contribution.dissipated_energy
        for key, value in contribution.state_delta.items():
            delta[key] = delta.get(key, 0.0) + value

    damage = min(
        1.0,
        max(0.0, state.damage + delta.get("damage", 0.0)),
    )
    fatigue = min(
        1.0,
        max(0.0, state.fatigue + delta.get("fatigue", 0.0)),
    )
    creep_strain = (
        state.creep_strain + delta.get("creep_strain", 0.0)
    )
    plastic_strain = (
        state.plastic_strain + delta.get("plastic_strain", 0.0)
    )
    integrity = max(
        0.0,
        min(
            1.0,
            state.integrity
            * (1.0 - damage)
            * (1.0 - fatigue),
        ),
    )

    temperature = (
        material_input.temperature
        if material_input.temperature is not None
        else state.temperature
    )
    moisture = (
        material_input.moisture
        if material_input.moisture is not None
        else state.moisture
    )
    pressure = (
        material_input.pressure
        if material_input.pressure is not None
        else state.pressure
    )
    activation = (
        material_input.activation
        if material_input.activation is not None
        else state.activation
    )

    return replace(
        state,
        time=state.time + material_input.delta_time,
        phase=MaterialPhase.LOADING,
        stress=stress,
        strain=material_input.strain,
        strain_rate=material_input.strain_rate,
        elastic_strain=(
            material_input.strain
            - plastic_strain
            - creep_strain
        ),
        plastic_strain=plastic_strain,
        creep_strain=creep_strain,
        temperature=temperature,
        moisture=max(0.0, moisture),
        pressure=pressure,
        integrity=integrity,
        damage=damage,
        fatigue=fatigue,
        activation=activation,
        history=state.history + abs(material_input.strain),
        energy=state.energy
        + 0.5 * abs(stress * material_input.strain),
        dissipated_energy=state.dissipated_energy + dissipated,
        direction=material_input.direction,
    )


def detect_failure_modes(
    model: MaterialModel,
    material_input: MaterialInput,
    stress: float,
    state: MaterialState,
) -> tuple[FailureMode, ...]:
    """Return all reduced-order failure modes active after evaluation."""

    parameters = model.parameters
    modes: list[FailureMode] = []

    if stress >= parameters.tensile_strength:
        modes.append(FailureMode.TENSILE_RUPTURE)
    if -stress >= parameters.compressive_strength:
        modes.append(FailureMode.COMPRESSIVE_CRUSHING)
    if (
        material_input.response_mode is ResponseMode.SHEAR
        and abs(stress) >= parameters.shear_strength
    ):
        modes.append(FailureMode.SHEAR_FAILURE)
    if abs(material_input.strain) >= parameters.failure_strain:
        modes.append(FailureMode.TEARING)
    if state.fatigue >= 1.0:
        modes.append(FailureMode.FATIGUE_CRACK)
    if state.corrosion >= 1.0:
        modes.append(FailureMode.CORROSION_LOSS)

    return tuple(dict.fromkeys(modes))


def evaluate_material(
    model: MaterialModel,
    material_input: MaterialInput,
) -> MaterialResponse:
    """Evaluate all active constitutive laws and return an immutable response."""

    if not isinstance(model, MaterialModel):
        raise TypeError("model must be MaterialModel.")
    if not isinstance(material_input, MaterialInput):
        raise TypeError("material_input must be MaterialInput.")

    contributions = tuple(
        evaluate_law(
            law,
            model.parameters,
            model.state,
            material_input,
        )
        for law in model.laws
    )
    stress = sum(item.stress for item in contributions)
    tangent = sum(
        item.tangent_modulus for item in contributions
    )

    capacity_factor = model.state.available_capacity
    stress *= capacity_factor
    tangent *= capacity_factor

    state_after = _updated_state(
        model.state,
        material_input,
        contributions,
        stress,
    )
    modes = detect_failure_modes(
        model,
        material_input,
        stress,
        state_after,
    )
    if modes:
        state_after = replace(
            state_after,
            failed=True,
            failure_mode=modes[0],
            phase=MaterialPhase.FAILED,
        )

    force_factor = (
        stress / model.parameters.reference_stress
    )

    return MaterialResponse(
        material_id=model.descriptor.material_id,
        stress=stress,
        tangent_modulus=tangent,
        force_factor=force_factor,
        state_before=model.state,
        state_after=state_after,
        contributions=contributions,
        failure_modes=modes,
        metadata={
            "family": model.descriptor.family.value,
            "anisotropy": model.descriptor.anisotropy.value,
            "law_count": len(model.laws),
        },
    )


def advance_material(
    model: MaterialModel,
    material_input: MaterialInput,
) -> tuple[MaterialModel, MaterialResponse]:
    """Evaluate a material and return a new model carrying the updated state."""

    response = evaluate_material(model, material_input)
    return (
        replace(model, state=response.state_after),
        response,
    )


def material_plane_factors(
    model: MaterialModel,
) -> Mapping[str, float]:
    """
    Expose material state as multiplicative ROIF plane factors.

    These values are designed for the Influence layer and future Capacity
    Tensor. They do not replace the constitutive response itself.
    """

    state = model.state
    factors = {
        "material.integrity": state.integrity,
        "material.available_capacity": state.available_capacity,
        "material.damage_gate": 1.0 - state.damage,
        "material.fatigue_gate": 1.0 - state.fatigue,
        "material.corrosion_gate": 1.0 - state.corrosion,
        "material.aging_gate": 1.0 - state.aging,
        "material.activation": state.activation
        if model.descriptor.active_material
        else 1.0,
        "material.remodeling": 1.0 + state.remodeling,
        "material.history": 1.0 / (1.0 + state.history),
    }
    return MappingProxyType(factors)


def response_to_trace(
    response: MaterialResponse,
    *,
    trace_id: str,
    kind: ConstitutiveLawKind,
    effect: MaterialEffect = MaterialEffect.UNKNOWN,
) -> MaterialTrace:
    """Create a persistent material trace from an evaluated response."""

    before = response.state_before
    after = response.state_after

    if kind in (
        ConstitutiveLawKind.CREEP,
        ConstitutiveLawKind.STANDARD_LINEAR_SOLID,
        ConstitutiveLawKind.VISCOELASTIC,
    ):
        magnitude = after.creep_strain - before.creep_strain
    elif kind in (
        ConstitutiveLawKind.ELASTOPLASTIC,
        ConstitutiveLawKind.VISCOPLASTIC,
        ConstitutiveLawKind.PERFECTLY_PLASTIC,
    ):
        magnitude = after.plastic_strain - before.plastic_strain
    elif kind is ConstitutiveLawKind.DAMAGE:
        magnitude = after.damage - before.damage
    elif kind is ConstitutiveLawKind.FATIGUE:
        magnitude = after.fatigue - before.fatigue
    else:
        magnitude = after.residual_strain - before.residual_strain

    retained = max(
        0.0,
        min(
            1.0,
            after.retained_fraction,
        ),
    )

    return MaterialTrace(
        trace_id=trace_id,
        material_id=response.material_id,
        time=after.time,
        kind=kind,
        magnitude=magnitude,
        retained_fraction=retained,
        reversibility=1.0 - retained,
        capacity_effect=(
            after.available_capacity - before.available_capacity
        ),
        stiffness_effect=(
            response.tangent_modulus
        ),
        effect=effect,
        metadata={
            "failure_modes": [
                mode.value for mode in response.failure_modes
            ],
        },
    )


def material_state_from_snapshot(
    snapshot: Mapping[str, Any],
) -> MaterialState:
    """
    Translate a ``core.material.Material.snapshot()``-style mapping.

    Unknown fields are preserved in metadata. Missing universal fields use
    conservative defaults.
    """

    if not isinstance(snapshot, Mapping):
        raise ROIFMaterialError("snapshot must be a mapping.")

    known = {
        "damage",
        "fatigue",
        "remodeling",
        "activation",
        "history",
        "energy",
        "strain",
        "creep_strain",
        "residual_strain",
        "rheology_time",
        "dissipated_energy",
        "failed",
        "total_force_component",
        "length_velocity",
    }
    metadata = {
        str(key): value
        for key, value in snapshot.items()
        if key not in known
    }

    damage = _unit(snapshot.get("damage", 0.0), "damage")
    fatigue = _unit(snapshot.get("fatigue", 0.0), "fatigue")
    failed = bool(snapshot.get("failed", False))

    integrity = max(
        0.0,
        min(1.0, (1.0 - damage) * (1.0 - fatigue)),
    )
    failure_mode = (
        FailureMode.BIOLOGICAL_FAILURE
        if failed
        else FailureMode.NONE
    )

    return MaterialState(
        time=_nonnegative(
            snapshot.get("rheology_time", 0.0),
            "rheology_time",
        ),
        phase=(
            MaterialPhase.FAILED
            if failed
            else MaterialPhase.INSTANTANEOUS
        ),
        stress=_finite(
            snapshot.get("total_force_component", 0.0),
            "total_force_component",
        ),
        strain=_finite(snapshot.get("strain", 0.0), "strain"),
        strain_rate=_finite(
            snapshot.get("length_velocity", 0.0),
            "length_velocity",
        ),
        elastic_strain=_finite(
            snapshot.get("strain", 0.0),
            "elastic_strain",
        )
        - _finite(
            snapshot.get("creep_strain", 0.0),
            "creep_strain",
        ),
        creep_strain=_finite(
            snapshot.get("creep_strain", 0.0),
            "creep_strain",
        ),
        residual_strain=_finite(
            snapshot.get("residual_strain", 0.0),
            "residual_strain",
        ),
        integrity=integrity,
        damage=damage,
        fatigue=fatigue,
        remodeling=_unit(
            snapshot.get("remodeling", 0.0),
            "remodeling",
        ),
        activation=_unit(
            snapshot.get("activation", 0.0),
            "activation",
        ),
        history=_nonnegative(
            snapshot.get("history", 0.0),
            "history",
        ),
        energy=_nonnegative(
            snapshot.get("energy", 0.0),
            "energy",
        ),
        dissipated_energy=_nonnegative(
            snapshot.get("dissipated_energy", 0.0),
            "dissipated_energy",
        ),
        failed=failed,
        failure_mode=failure_mode,
        metadata=metadata,
    )


def model_from_core_material(
    material: SnapshotProvider,
    *,
    material_id: str,
    family: MaterialFamily = MaterialFamily.GENERIC,
    anisotropy: AnisotropyKind = AnisotropyKind.ISOTROPIC,
    laws: Sequence[ConstitutiveLaw] | None = None,
) -> MaterialModel:
    """
    Build a ROIF MaterialModel from an existing core material instance.

    Only the public ``snapshot()`` contract is required.
    """

    if not isinstance(material, SnapshotProvider):
        raise TypeError(
            "material must provide snapshot()."
        )

    snapshot = material.snapshot()
    name = str(snapshot.get("name", material_id))

    parameters_snapshot = snapshot.get("parameters", {})
    if not isinstance(parameters_snapshot, Mapping):
        parameters_snapshot = {}

    elastic_modulus = parameters_snapshot.get(
        "stiffness",
        snapshot.get("stiffness", 1.0),
    )
    damping = parameters_snapshot.get(
        "damping",
        snapshot.get("damping", 0.0),
    )
    density = parameters_snapshot.get(
        "density",
        snapshot.get("density", 1.0),
    )
    relaxed = parameters_snapshot.get(
        "relaxed_stiffness",
        snapshot.get("relaxed_stiffness"),
    )

    parameters = MaterialParameters(
        density=max(1e-12, float(density)),
        elastic_modulus=max(1e-12, float(elastic_modulus)),
        relaxed_modulus=(
            None
            if relaxed is None
            else max(1e-12, float(relaxed))
        ),
        shear_modulus=max(1e-12, float(elastic_modulus)),
        bulk_modulus=max(1e-12, float(elastic_modulus)),
        damping=max(0.0, float(damping)),
        viscosity=max(0.0, float(damping)),
        fatigue_rate=max(
            0.0,
            float(parameters_snapshot.get("fatigue_rate", 0.0)),
        ),
        damage_rate=max(
            0.0,
            float(parameters_snapshot.get("damage_rate", 0.0)),
        ),
        recovery_rate=max(
            0.0,
            float(parameters_snapshot.get("recovery_rate", 0.0)),
        ),
        remodeling_rate=max(
            0.0,
            float(
                parameters_snapshot.get("remodeling_rate", 0.0)
            ),
        ),
        active_stress_scale=max(
            0.0,
            float(
                parameters_snapshot.get("reference_force", 0.0)
            ),
        ),
    )

    if laws is None:
        inferred: list[ConstitutiveLaw] = [
            ConstitutiveLaw(
                law_id="linear_elastic",
                kind=ConstitutiveLawKind.LINEAR_ELASTIC,
            )
        ]
        if relaxed is not None:
            inferred.append(
                ConstitutiveLaw(
                    law_id="standard_linear_solid",
                    kind=(
                        ConstitutiveLawKind.STANDARD_LINEAR_SOLID
                    ),
                )
            )
        if parameters.active_stress_scale > 0.0:
            inferred.append(
                ConstitutiveLaw(
                    law_id="active_contractile",
                    kind=(
                        ConstitutiveLawKind.ACTIVE_CONTRACTILE
                    ),
                    response_modes=(ResponseMode.ACTIVE,),
                )
            )
        laws = inferred

    descriptor = MaterialDescriptor(
        material_id=material_id,
        name=name,
        family=family,
        anisotropy=anisotropy,
        active_material=(
            parameters.active_stress_scale > 0.0
        ),
        biological=family in {
            MaterialFamily.BIOLOGICAL_TISSUE,
            MaterialFamily.MUSCLE,
            MaterialFamily.TENDON,
            MaterialFamily.LIGAMENT,
            MaterialFamily.FASCIA,
            MaterialFamily.BONE,
            MaterialFamily.CARTILAGE,
            MaterialFamily.NERVE,
            MaterialFamily.VESSEL,
            MaterialFamily.ORGAN_TISSUE,
        },
        composite=family in {
            MaterialFamily.COMPOSITE,
            MaterialFamily.FIBRE_COMPOSITE,
            MaterialFamily.PARTICLE_COMPOSITE,
            MaterialFamily.LAYERED_COMPOSITE,
        },
    )

    return MaterialModel(
        descriptor=descriptor,
        parameters=parameters,
        laws=tuple(laws),
        state=material_state_from_snapshot(snapshot),
        metadata={
            "source": "core.material.Material.snapshot",
        },
    )


def default_material_model(
    material_id: str,
    *,
    name: str | None = None,
    family: MaterialFamily = MaterialFamily.GENERIC,
) -> MaterialModel:
    """Create a conservative universal linear-elastic material."""

    material_id = _text(material_id, "material_id")
    return MaterialModel(
        descriptor=MaterialDescriptor(
            material_id=material_id,
            name=name or material_id,
            family=family,
        )
    )


def material_models_by_id(
    models: Sequence[MaterialModel],
) -> Mapping[str, MaterialModel]:
    """Build an immutable lookup with duplicate protection."""

    lookup: dict[str, MaterialModel] = {}
    for model in models:
        material_id = model.descriptor.material_id
        if material_id in lookup:
            raise ROIFMaterialError(
                f"duplicate material_id {material_id!r}."
            )
        lookup[material_id] = model
    return MappingProxyType(lookup)


__all__ = [
    "AnisotropyKind",
    "ConstitutiveLaw",
    "ConstitutiveLawKind",
    "FailureMode",
    "LawContribution",
    "MaterialDescriptor",
    "MaterialEffect",
    "MaterialFamily",
    "MaterialInput",
    "MaterialModel",
    "MaterialParameters",
    "MaterialPhase",
    "MaterialResponse",
    "MaterialState",
    "MaterialTrace",
    "ROIFMaterialError",
    "ResponseMode",
    "SnapshotProvider",
    "advance_material",
    "default_material_model",
    "detect_failure_modes",
    "evaluate_law",
    "evaluate_material",
    "material_models_by_id",
    "material_plane_factors",
    "material_state_from_snapshot",
    "model_from_core_material",
    "response_to_trace",
]


