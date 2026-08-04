from __future__ import annotations

"""
Immutable rheological memory for the ROIF Structural Memory Engine.

MaterialState answers:

    What is the current physical rheological state?

RheologicalMemory answers:

    What rheological history has accumulated and what part of it remains?

The class is intentionally domain-independent. It can represent the retained
memory of a Standard Linear Solid, another internal-variable viscoelastic
model, or an externally measured creep/recovery process.

RheologicalMemory is a snapshot, not a live material model. It stores observed
or derived rheological quantities at one moment and exposes stable normalized
indices for IrreversibleChange, HistoryPattern, and StructuralSignature.

The object is immutable. Updating rheological history creates a new instance.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any
from uuid import uuid4


class RheologicalMemoryError(ValueError):
    """Raised when a rheological-memory snapshot is invalid."""


class RheologyPhase(str, Enum):
    """Current phase of the rheological trajectory."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    CREEPING = "creeping"
    RELAXING = "relaxing"
    RECOVERING = "recovering"
    RELAXED = "relaxed"
    RESIDUAL = "residual"
    UNKNOWN = "unknown"


class RheologyModel(str, Enum):
    """Constitutive-model family associated with the snapshot."""

    ELASTIC = "elastic"
    KELVIN_VOIGT = "kelvin_voigt"
    MAXWELL = "maxwell"
    STANDARD_LINEAR_SOLID = "standard_linear_solid"
    GENERALIZED_MAXWELL = "generalized_maxwell"
    GENERALIZED_KELVIN = "generalized_kelvin"
    FRACTIONAL = "fractional"
    EMPIRICAL = "empirical"
    UNKNOWN = "unknown"


def _require_nonempty_string(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise RheologicalMemoryError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise RheologicalMemoryError(
            f"{field_name} must not be empty"
        )

    return normalized


def _validate_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RheologicalMemoryError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise RheologicalMemoryError(
            f"{field_name} must be finite"
        )

    return numeric


def _validate_nonnegative_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _validate_finite(
        value,
        field_name=field_name,
    )

    if numeric < 0.0:
        raise RheologicalMemoryError(
            f"{field_name} must be non-negative"
        )

    return numeric


def _validate_unit_interval(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = _validate_nonnegative_finite(
        value,
        field_name=field_name,
    )

    if numeric > 1.0:
        raise RheologicalMemoryError(
            f"{field_name} must be in [0, 1]"
        )

    return numeric


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise RheologicalMemoryError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, Any] = {}

    for raw_key, item in value.items():
        key = _require_nonempty_string(
            raw_key,
            field_name=f"{field_name} key",
        )
        normalized[key] = item

    return MappingProxyType(
        dict(sorted(normalized.items()))
    )


@dataclass(frozen=True, slots=True)
class RheologicalMemory:
    """
    Immutable snapshot of accumulated rheological history.

    This object stores one revision of rheological memory. The next revision
    should be created with ``evolve`` so the parent-child chain remains
    explicit and previous knowledge is not overwritten.
    """

    target_id: str
    observation_time: float

    rheology_time: float = 0.0

    creep_strain: float = 0.0
    peak_creep_strain: float = 0.0
    elastic_strain: float = 0.0
    residual_strain: float = 0.0
    recoverable_strain: float = 0.0
    total_strain: float = 0.0

    instantaneous_stiffness: float = 0.0
    relaxed_stiffness: float = 0.0
    creep_time_constant: float = 0.0
    applied_stress: float = 0.0

    relaxation_fraction: float = 0.0
    recovery_fraction: float = 0.0
    retained_fraction: float = 0.0
    confidence: float = 1.0

    phase: RheologyPhase = RheologyPhase.UNKNOWN
    model: RheologyModel = RheologyModel.UNKNOWN

    source_snapshot_id: str | None = None
    parent_memory_id: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    memory_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        target_id = _require_nonempty_string(
            self.target_id,
            field_name="target_id",
        )
        memory_id = _require_nonempty_string(
            self.memory_id,
            field_name="memory_id",
        )

        observation_time = _validate_nonnegative_finite(
            self.observation_time,
            field_name="observation_time",
        )
        rheology_time = _validate_nonnegative_finite(
            self.rheology_time,
            field_name="rheology_time",
        )

        creep_strain = _validate_finite(
            self.creep_strain,
            field_name="creep_strain",
        )
        peak_creep_strain = _validate_nonnegative_finite(
            self.peak_creep_strain,
            field_name="peak_creep_strain",
        )
        elastic_strain = _validate_finite(
            self.elastic_strain,
            field_name="elastic_strain",
        )
        residual_strain = _validate_finite(
            self.residual_strain,
            field_name="residual_strain",
        )
        recoverable_strain = _validate_nonnegative_finite(
            self.recoverable_strain,
            field_name="recoverable_strain",
        )
        total_strain = _validate_finite(
            self.total_strain,
            field_name="total_strain",
        )

        if abs(creep_strain) > peak_creep_strain + 1e-12:
            raise RheologicalMemoryError(
                "peak_creep_strain must be at least "
                "abs(creep_strain)"
            )

        instantaneous_stiffness = (
            _validate_nonnegative_finite(
                self.instantaneous_stiffness,
                field_name="instantaneous_stiffness",
            )
        )
        relaxed_stiffness = _validate_nonnegative_finite(
            self.relaxed_stiffness,
            field_name="relaxed_stiffness",
        )
        creep_time_constant = (
            _validate_nonnegative_finite(
                self.creep_time_constant,
                field_name="creep_time_constant",
            )
        )
        applied_stress = _validate_finite(
            self.applied_stress,
            field_name="applied_stress",
        )

        if (
            instantaneous_stiffness > 0.0
            and relaxed_stiffness
            > instantaneous_stiffness + 1e-12
        ):
            raise RheologicalMemoryError(
                "relaxed_stiffness must not exceed "
                "instantaneous_stiffness"
            )

        relaxation_fraction = _validate_unit_interval(
            self.relaxation_fraction,
            field_name="relaxation_fraction",
        )
        recovery_fraction = _validate_unit_interval(
            self.recovery_fraction,
            field_name="recovery_fraction",
        )
        retained_fraction = _validate_unit_interval(
            self.retained_fraction,
            field_name="retained_fraction",
        )
        confidence = _validate_unit_interval(
            self.confidence,
            field_name="confidence",
        )

        try:
            phase = RheologyPhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise RheologicalMemoryError(
                f"unsupported rheology phase: {self.phase!r}"
            ) from exc

        try:
            model = RheologyModel(self.model)
        except (TypeError, ValueError) as exc:
            raise RheologicalMemoryError(
                f"unsupported rheology model: {self.model!r}"
            ) from exc

        source_snapshot_id = self.source_snapshot_id
        if source_snapshot_id is not None:
            source_snapshot_id = _require_nonempty_string(
                source_snapshot_id,
                field_name="source_snapshot_id",
            )

        parent_memory_id = self.parent_memory_id
        if parent_memory_id is not None:
            parent_memory_id = _require_nonempty_string(
                parent_memory_id,
                field_name="parent_memory_id",
            )

        if parent_memory_id == memory_id:
            raise RheologicalMemoryError(
                "a rheological memory cannot be its own parent"
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "memory_id", memory_id)
        object.__setattr__(
            self,
            "observation_time",
            observation_time,
        )
        object.__setattr__(
            self,
            "rheology_time",
            rheology_time,
        )
        object.__setattr__(
            self,
            "creep_strain",
            creep_strain,
        )
        object.__setattr__(
            self,
            "peak_creep_strain",
            peak_creep_strain,
        )
        object.__setattr__(
            self,
            "elastic_strain",
            elastic_strain,
        )
        object.__setattr__(
            self,
            "residual_strain",
            residual_strain,
        )
        object.__setattr__(
            self,
            "recoverable_strain",
            recoverable_strain,
        )
        object.__setattr__(
            self,
            "total_strain",
            total_strain,
        )
        object.__setattr__(
            self,
            "instantaneous_stiffness",
            instantaneous_stiffness,
        )
        object.__setattr__(
            self,
            "relaxed_stiffness",
            relaxed_stiffness,
        )
        object.__setattr__(
            self,
            "creep_time_constant",
            creep_time_constant,
        )
        object.__setattr__(
            self,
            "applied_stress",
            applied_stress,
        )
        object.__setattr__(
            self,
            "relaxation_fraction",
            relaxation_fraction,
        )
        object.__setattr__(
            self,
            "recovery_fraction",
            recovery_fraction,
        )
        object.__setattr__(
            self,
            "retained_fraction",
            retained_fraction,
        )
        object.__setattr__(
            self,
            "confidence",
            confidence,
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "model", model)
        object.__setattr__(
            self,
            "source_snapshot_id",
            source_snapshot_id,
        )
        object.__setattr__(
            self,
            "parent_memory_id",
            parent_memory_id,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @property
    def has_creep(self) -> bool:
        return (
            abs(self.creep_strain) > 1e-12
            or self.peak_creep_strain > 1e-12
        )

    @property
    def has_residual_memory(self) -> bool:
        return (
            abs(self.residual_strain) > 1e-12
            or self.retained_fraction > 1e-12
        )

    @property
    def is_recovering(self) -> bool:
        return (
            self.phase is RheologyPhase.RECOVERING
            or (
                self.recoverable_strain > 1e-12
                and self.recovery_fraction < 1.0
            )
        )

    @property
    def is_relaxed(self) -> bool:
        return (
            self.phase is RheologyPhase.RELAXED
            or self.relaxation_fraction >= 1.0 - 1e-9
        )

    @property
    def stiffness_ratio(self) -> float:
        if self.instantaneous_stiffness <= 0.0:
            return 0.0

        return min(
            max(
                self.relaxed_stiffness
                / self.instantaneous_stiffness,
                0.0,
            ),
            1.0,
        )

    @property
    def relaxable_fraction(self) -> float:
        if self.instantaneous_stiffness <= 0.0:
            return 0.0

        return 1.0 - self.stiffness_ratio

    @property
    def strain_retention_ratio(self) -> float:
        if self.peak_creep_strain <= 0.0:
            return 0.0

        retained = max(
            abs(self.creep_strain),
            abs(self.residual_strain),
        )

        return min(
            retained / self.peak_creep_strain,
            1.0,
        )

    @property
    def recovery_remaining(self) -> float:
        return 1.0 - self.recovery_fraction

    @property
    def normalized_age(self) -> float:
        if self.creep_time_constant <= 0.0:
            return 0.0

        ratio = (
            self.rheology_time
            / self.creep_time_constant
        )

        return ratio / (1.0 + ratio)

    @property
    def relaxation_index(self) -> float:
        return min(
            max(
                0.5 * self.relaxation_fraction
                + 0.5 * self.relaxable_fraction,
                0.0,
            ),
            1.0,
        )

    @property
    def memory_index(self) -> float:
        strain_memory = max(
            self.retained_fraction,
            self.strain_retention_ratio,
        )
        residual_presence = min(
            abs(self.residual_strain)
            / max(self.peak_creep_strain, 1e-12),
            1.0,
        )

        raw = (
            0.45 * strain_memory
            + 0.25 * residual_presence
            + 0.20 * self.normalized_age
            + 0.10 * self.relaxation_index
        )

        return min(
            max(raw * self.confidence, 0.0),
            1.0,
        )

    @property
    def signed_memory_strain(self) -> float:
        if abs(self.residual_strain) > 1e-12:
            return self.residual_strain

        return self.creep_strain

    def evolve(
        self,
        *,
        observation_time: float,
        rheology_time: float | None = None,
        creep_strain: float | None = None,
        peak_creep_strain: float | None = None,
        elastic_strain: float | None = None,
        residual_strain: float | None = None,
        recoverable_strain: float | None = None,
        total_strain: float | None = None,
        applied_stress: float | None = None,
        relaxation_fraction: float | None = None,
        recovery_fraction: float | None = None,
        retained_fraction: float | None = None,
        confidence: float | None = None,
        phase: RheologyPhase | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> RheologicalMemory:
        """
        Create the next immutable rheological-memory revision.
        """

        next_creep = (
            self.creep_strain
            if creep_strain is None
            else float(creep_strain)
        )
        next_peak = (
            self.peak_creep_strain
            if peak_creep_strain is None
            else float(peak_creep_strain)
        )
        next_peak = max(next_peak, abs(next_creep))

        merged_metadata = {
            **dict(self.metadata),
            **dict(metadata or {}),
        }

        return replace(
            self,
            memory_id=memory_id or uuid4().hex,
            parent_memory_id=self.memory_id,
            observation_time=observation_time,
            rheology_time=(
                self.rheology_time
                if rheology_time is None
                else rheology_time
            ),
            creep_strain=next_creep,
            peak_creep_strain=next_peak,
            elastic_strain=(
                self.elastic_strain
                if elastic_strain is None
                else elastic_strain
            ),
            residual_strain=(
                self.residual_strain
                if residual_strain is None
                else residual_strain
            ),
            recoverable_strain=(
                self.recoverable_strain
                if recoverable_strain is None
                else recoverable_strain
            ),
            total_strain=(
                self.total_strain
                if total_strain is None
                else total_strain
            ),
            applied_stress=(
                self.applied_stress
                if applied_stress is None
                else applied_stress
            ),
            relaxation_fraction=(
                self.relaxation_fraction
                if relaxation_fraction is None
                else relaxation_fraction
            ),
            recovery_fraction=(
                self.recovery_fraction
                if recovery_fraction is None
                else recovery_fraction
            ),
            retained_fraction=(
                self.retained_fraction
                if retained_fraction is None
                else retained_fraction
            ),
            confidence=(
                self.confidence
                if confidence is None
                else confidence
            ),
            phase=(
                self.phase
                if phase is None
                else phase
            ),
            metadata=merged_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "memory_id": self.memory_id,
            "parent_memory_id": self.parent_memory_id,
            "source_snapshot_id": self.source_snapshot_id,
            "target_id": self.target_id,
            "observation_time": self.observation_time,
            "rheology_time": self.rheology_time,
            "creep_strain": self.creep_strain,
            "peak_creep_strain": self.peak_creep_strain,
            "elastic_strain": self.elastic_strain,
            "residual_strain": self.residual_strain,
            "recoverable_strain": self.recoverable_strain,
            "total_strain": self.total_strain,
            "instantaneous_stiffness": (
                self.instantaneous_stiffness
            ),
            "relaxed_stiffness": self.relaxed_stiffness,
            "creep_time_constant": self.creep_time_constant,
            "applied_stress": self.applied_stress,
            "relaxation_fraction": self.relaxation_fraction,
            "recovery_fraction": self.recovery_fraction,
            "retained_fraction": self.retained_fraction,
            "confidence": self.confidence,
            "phase": self.phase.value,
            "model": self.model.value,
            "derived": {
                "has_creep": self.has_creep,
                "has_residual_memory": (
                    self.has_residual_memory
                ),
                "is_recovering": self.is_recovering,
                "is_relaxed": self.is_relaxed,
                "stiffness_ratio": self.stiffness_ratio,
                "relaxable_fraction": (
                    self.relaxable_fraction
                ),
                "strain_retention_ratio": (
                    self.strain_retention_ratio
                ),
                "normalized_age": self.normalized_age,
                "relaxation_index": (
                    self.relaxation_index
                ),
                "memory_index": self.memory_index,
                "signed_memory_strain": (
                    self.signed_memory_strain
                ),
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> RheologicalMemory:
        if not isinstance(data, Mapping):
            raise RheologicalMemoryError(
                "rheological-memory data must be a mapping"
            )

        return cls(
            memory_id=str(
                data.get("memory_id")
                or uuid4().hex
            ),
            parent_memory_id=(
                None
                if data.get("parent_memory_id") is None
                else str(data["parent_memory_id"])
            ),
            source_snapshot_id=(
                None
                if data.get("source_snapshot_id") is None
                else str(data["source_snapshot_id"])
            ),
            target_id=str(data["target_id"]),
            observation_time=float(
                data.get("observation_time", 0.0)
            ),
            rheology_time=float(
                data.get("rheology_time", 0.0)
            ),
            creep_strain=float(
                data.get("creep_strain", 0.0)
            ),
            peak_creep_strain=float(
                data.get("peak_creep_strain", 0.0)
            ),
            elastic_strain=float(
                data.get("elastic_strain", 0.0)
            ),
            residual_strain=float(
                data.get("residual_strain", 0.0)
            ),
            recoverable_strain=float(
                data.get("recoverable_strain", 0.0)
            ),
            total_strain=float(
                data.get("total_strain", 0.0)
            ),
            instantaneous_stiffness=float(
                data.get("instantaneous_stiffness", 0.0)
            ),
            relaxed_stiffness=float(
                data.get("relaxed_stiffness", 0.0)
            ),
            creep_time_constant=float(
                data.get("creep_time_constant", 0.0)
            ),
            applied_stress=float(
                data.get("applied_stress", 0.0)
            ),
            relaxation_fraction=float(
                data.get("relaxation_fraction", 0.0)
            ),
            recovery_fraction=float(
                data.get("recovery_fraction", 0.0)
            ),
            retained_fraction=float(
                data.get("retained_fraction", 0.0)
            ),
            confidence=float(
                data.get("confidence", 1.0)
            ),
            phase=data.get(
                "phase",
                RheologyPhase.UNKNOWN.value,
            ),
            model=data.get(
                "model",
                RheologyModel.UNKNOWN.value,
            ),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_material_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        target_id: str,
        observation_time: float,
        phase: RheologyPhase | str = (
            RheologyPhase.UNKNOWN
        ),
        source_snapshot_id: str | None = None,
        confidence: float = 1.0,
        residual_strain: float = 0.0,
        recoverable_strain: float = 0.0,
        recovery_fraction: float = 0.0,
        retained_fraction: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RheologicalMemory:
        """
        Build memory from ``Material.snapshot()`` output.

        Missing rheology fields default to zero for compatibility with older
        snapshots.
        """

        if not isinstance(snapshot, Mapping):
            raise RheologicalMemoryError(
                "material snapshot must be a mapping"
            )

        parameters = snapshot.get("parameters", {})
        state = snapshot.get("state", {})

        if not isinstance(parameters, Mapping):
            raise RheologicalMemoryError(
                "snapshot parameters must be a mapping"
            )

        if not isinstance(state, Mapping):
            raise RheologicalMemoryError(
                "snapshot state must be a mapping"
            )

        creep_strain = float(
            state.get("creep_strain", 0.0)
        )
        peak_creep_strain = float(
            state.get(
                "peak_creep_strain",
                abs(creep_strain),
            )
        )
        elastic_strain = float(
            state.get("elastic_strain", 0.0)
        )
        total_strain = float(
            state.get(
                "strain",
                elastic_strain
                + creep_strain
                + residual_strain,
            )
        )

        if retained_fraction is None:
            if peak_creep_strain > 0.0:
                retained_fraction = min(
                    max(
                        abs(creep_strain)
                        / peak_creep_strain,
                        0.0,
                    ),
                    1.0,
                )
            else:
                retained_fraction = 0.0

        instantaneous_stiffness = float(
            parameters.get("stiffness", 0.0)
        )
        relaxed_stiffness = float(
            parameters.get(
                "relaxed_stiffness",
                instantaneous_stiffness,
            )
        )

        if instantaneous_stiffness > 0.0:
            relaxation_fraction = min(
                max(
                    1.0
                    - (
                        relaxed_stiffness
                        / instantaneous_stiffness
                    ),
                    0.0,
                ),
                1.0,
            )
        else:
            relaxation_fraction = 0.0

        merged_metadata = {
            "source": "material_snapshot",
            **dict(metadata or {}),
        }

        return cls(
            target_id=target_id,
            observation_time=observation_time,
            rheology_time=float(
                state.get("rheology_time", 0.0)
            ),
            creep_strain=creep_strain,
            peak_creep_strain=max(
                peak_creep_strain,
                abs(creep_strain),
            ),
            elastic_strain=elastic_strain,
            residual_strain=residual_strain,
            recoverable_strain=recoverable_strain,
            total_strain=total_strain,
            instantaneous_stiffness=(
                instantaneous_stiffness
            ),
            relaxed_stiffness=relaxed_stiffness,
            creep_time_constant=float(
                parameters.get(
                    "creep_time_constant",
                    0.0,
                )
            ),
            applied_stress=float(
                state.get("stress", 0.0)
            ),
            relaxation_fraction=relaxation_fraction,
            recovery_fraction=recovery_fraction,
            retained_fraction=retained_fraction,
            confidence=confidence,
            phase=phase,
            model=(
                RheologyModel.STANDARD_LINEAR_SOLID
                if bool(
                    parameters.get(
                        "rheology_enabled",
                        False,
                    )
                )
                else RheologyModel.ELASTIC
            ),
            source_snapshot_id=source_snapshot_id,
            metadata=merged_metadata,
        )
