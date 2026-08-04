from __future__ import annotations

"""
Structural signatures for the ROIF Structural Memory Engine.

HistoryEvent answers:

    What happened?

IrreversibleChange answers:

    What remained after one event?

HistoryPattern answers:

    What causal pattern was formed by all retained changes?

StructuralSignature answers:

    What stable, comparable imprint does that pattern leave in the structure?

A StructuralSignature is an immutable numerical and categorical description
derived from a HistoryPattern. It supports:

- normalized influence-plane profiles;
- agent, material-change, time-scale, target, and rheology profiles;
- capacity-loss and capacity-gain descriptors;
- persistence, irreversibility, and rheological-memory descriptors;
- causal-depth and chronology descriptors;
- harmful/adaptive balance;
- comparison between structural histories;
- deterministic serialization;
- future use by HistoryDecoder and inverse-cascade reconstruction.

The signature does not claim that a history is uniquely identifiable.
Different histories may produce similar or indistinguishable signatures.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .history_pattern import (
    HistoryPattern,
    RankedContribution,
)


class StructuralSignatureError(ValueError):
    """Raised when a structural signature is invalid."""


def _require_nonempty_string(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise StructuralSignatureError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise StructuralSignatureError(
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
        raise StructuralSignatureError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise StructuralSignatureError(
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
        raise StructuralSignatureError(
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
        raise StructuralSignatureError(
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
        raise StructuralSignatureError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, Any] = {}

    for raw_key, item in value.items():
        key = _require_nonempty_string(
            raw_key,
            field_name=f"{field_name} key",
        )
        normalized[key] = item

    return MappingProxyType(normalized)


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0

    return numerator / denominator


def _normalize_profile(
    profile: Mapping[str, float] | None,
    *,
    field_name: str,
) -> Mapping[str, float]:
    if profile is None:
        return MappingProxyType({})

    if not isinstance(profile, Mapping):
        raise StructuralSignatureError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, float] = {}

    for raw_key, raw_value in profile.items():
        key = _require_nonempty_string(
            raw_key,
            field_name=f"{field_name} key",
        )
        value = _validate_nonnegative_finite(
            raw_value,
            field_name=f"{field_name}[{key!r}]",
        )

        if value > 0.0:
            normalized[key] = value

    total = sum(normalized.values())

    if total > 0.0:
        normalized = {
            key: value / total
            for key, value in normalized.items()
        }

    return MappingProxyType(
        dict(
            sorted(
                normalized.items(),
                key=lambda item: item[0],
            )
        )
    )


def _profile_from_ranked(
    contributions: Iterable[RankedContribution],
) -> Mapping[str, float]:
    return MappingProxyType(
        {
            contribution.key: contribution.score
            for contribution in contributions
            if contribution.score > 0.0
        }
    )


def _mapping_to_dict(
    mapping: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in mapping.items()
    }


@dataclass(frozen=True, slots=True)
class SignatureDistanceWeights:
    """
    Weights used when comparing two StructuralSignature objects.

    Profile distances use normalized L1 distance.
    Scalar distances use absolute normalized difference.
    """

    plane_profile: float = 1.0
    agent_profile: float = 0.6
    kind_profile: float = 0.8
    time_scale_profile: float = 0.8
    target_profile: float = 0.4
    capacity: float = 1.0
    persistence: float = 0.8
    irreversibility: float = 0.8
    progression: float = 0.7
    adaptation: float = 0.7
    causality: float = 0.7
    chronology: float = 0.4

    def __post_init__(self) -> None:
        for field_name in (
            "plane_profile",
            "agent_profile",
            "kind_profile",
            "time_scale_profile",
            "target_profile",
            "capacity",
            "persistence",
            "irreversibility",
            "progression",
            "adaptation",
            "causality",
            "chronology",
        ):
            value = _validate_nonnegative_finite(
                getattr(self, field_name),
                field_name=field_name,
            )
            object.__setattr__(
                self,
                field_name,
                value,
            )

        if self.total_weight <= 0.0:
            raise StructuralSignatureError(
                "at least one distance weight must be positive"
            )

    @property
    def total_weight(self) -> float:
        return sum(
            (
                self.plane_profile,
                self.agent_profile,
                self.kind_profile,
                self.time_scale_profile,
                self.target_profile,
                self.capacity,
                self.persistence,
                self.irreversibility,
                self.progression,
                self.adaptation,
                self.causality,
                self.chronology,
            )
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "plane_profile": self.plane_profile,
            "agent_profile": self.agent_profile,
            "kind_profile": self.kind_profile,
            "time_scale_profile": self.time_scale_profile,
            "target_profile": self.target_profile,
            "capacity": self.capacity,
            "persistence": self.persistence,
            "irreversibility": self.irreversibility,
            "progression": self.progression,
            "adaptation": self.adaptation,
            "causality": self.causality,
            "chronology": self.chronology,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> SignatureDistanceWeights:
        if not isinstance(data, Mapping):
            raise StructuralSignatureError(
                "distance-weight data must be a mapping"
            )

        return cls(
            plane_profile=float(
                data.get("plane_profile", 1.0)
            ),
            agent_profile=float(
                data.get("agent_profile", 0.6)
            ),
            kind_profile=float(
                data.get("kind_profile", 0.8)
            ),
            time_scale_profile=float(
                data.get("time_scale_profile", 0.8)
            ),
            target_profile=float(
                data.get("target_profile", 0.4)
            ),
            capacity=float(
                data.get("capacity", 1.0)
            ),
            persistence=float(
                data.get("persistence", 0.8)
            ),
            irreversibility=float(
                data.get("irreversibility", 0.8)
            ),
            progression=float(
                data.get("progression", 0.7)
            ),
            adaptation=float(
                data.get("adaptation", 0.7)
            ),
            causality=float(
                data.get("causality", 0.7)
            ),
            chronology=float(
                data.get("chronology", 0.4)
            ),
        )


@dataclass(frozen=True, slots=True)
class SignatureComparison:
    """
    Result of comparing two structural signatures.

    distance:
        Normalized distance in [0, 1].

    similarity:
        Equal to 1 - distance.

    components:
        Per-component normalized distances.
    """

    left_signature_id: str
    right_signature_id: str
    distance: float
    similarity: float
    components: Mapping[str, float]
    weights: SignatureDistanceWeights

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        left_signature_id = _require_nonempty_string(
            self.left_signature_id,
            field_name="left_signature_id",
        )
        right_signature_id = _require_nonempty_string(
            self.right_signature_id,
            field_name="right_signature_id",
        )
        distance = _validate_unit_interval(
            self.distance,
            field_name="distance",
        )
        similarity = _validate_unit_interval(
            self.similarity,
            field_name="similarity",
        )

        if abs(
            similarity - (1.0 - distance)
        ) > 1e-9:
            raise StructuralSignatureError(
                "similarity must equal 1 - distance"
            )

        if not isinstance(
            self.weights,
            SignatureDistanceWeights,
        ):
            raise StructuralSignatureError(
                "weights must be SignatureDistanceWeights"
            )

        raw_components = self.components

        if not isinstance(raw_components, Mapping):
            raise StructuralSignatureError(
                "components must be a mapping"
            )

        components: dict[str, float] = {}

        for raw_key, raw_value in raw_components.items():
            key = _require_nonempty_string(
                raw_key,
                field_name="component key",
            )
            value = _validate_unit_interval(
                raw_value,
                field_name=f"components[{key!r}]",
            )
            components[key] = value

        object.__setattr__(
            self,
            "left_signature_id",
            left_signature_id,
        )
        object.__setattr__(
            self,
            "right_signature_id",
            right_signature_id,
        )
        object.__setattr__(
            self,
            "distance",
            distance,
        )
        object.__setattr__(
            self,
            "similarity",
            similarity,
        )
        object.__setattr__(
            self,
            "components",
            MappingProxyType(
                dict(sorted(components.items()))
            ),
        )

    @property
    def most_different_component(
        self,
    ) -> tuple[str, float] | None:
        if not self.components:
            return None

        return max(
            self.components.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "left_signature_id": self.left_signature_id,
            "right_signature_id": self.right_signature_id,
            "distance": self.distance,
            "similarity": self.similarity,
            "components": dict(self.components),
            "weights": self.weights.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StructuralSignature:
    """
    Comparable imprint of accumulated structural history.

    Profiles are normalized independently. Every profile sums to 1 when
    non-empty.

    Scalar values preserve information about accumulated capacity effects,
    persistence, irreversibility, progression, adaptation, causal depth,
    and temporal extent.
    """

    source_pattern_id: str

    plane_profile: Mapping[str, float] = field(
        default_factory=dict
    )
    agent_profile: Mapping[str, float] = field(
        default_factory=dict
    )
    kind_profile: Mapping[str, float] = field(
        default_factory=dict
    )
    time_scale_profile: Mapping[str, float] = field(
        default_factory=dict
    )
    target_profile: Mapping[str, float] = field(
        default_factory=dict
    )
    rheology_profile: Mapping[str, float] = field(
        default_factory=dict
    )

    total_changes: int = 0
    root_count: int = 0
    leaf_count: int = 0
    causal_depth: int = 0

    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    total_capacity_loss: float = 0.0
    total_capacity_gain: float = 0.0
    net_capacity_effect: float = 0.0

    cumulative_signature_weight: float = 0.0
    mean_signature_weight: float = 0.0

    persistence_index: float = 0.0
    irreversibility_index: float = 0.0
    progression_index: float = 0.0
    adaptation_index: float = 0.0

    harmful_fraction: float = 0.0
    beneficial_fraction: float = 0.0
    mixed_fraction: float = 0.0

    rheology_change_count: int = 0
    total_abs_creep_strain: float = 0.0
    mean_abs_creep_strain: float = 0.0
    max_abs_creep_strain: float = 0.0
    mean_abs_residual_strain: float = 0.0
    max_abs_residual_strain: float = 0.0
    rheological_memory_index: float = 0.0
    rheological_retention_index: float = 0.0
    rheological_relaxation_index: float = 0.0

    label: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    signature_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        source_pattern_id = _require_nonempty_string(
            self.source_pattern_id,
            field_name="source_pattern_id",
        )
        signature_id = _require_nonempty_string(
            self.signature_id,
            field_name="signature_id",
        )

        plane_profile = _normalize_profile(
            self.plane_profile,
            field_name="plane_profile",
        )
        agent_profile = _normalize_profile(
            self.agent_profile,
            field_name="agent_profile",
        )
        kind_profile = _normalize_profile(
            self.kind_profile,
            field_name="kind_profile",
        )
        time_scale_profile = _normalize_profile(
            self.time_scale_profile,
            field_name="time_scale_profile",
        )
        target_profile = _normalize_profile(
            self.target_profile,
            field_name="target_profile",
        )
        rheology_profile = _normalize_profile(
            self.rheology_profile,
            field_name="rheology_profile",
        )

        integer_fields = (
            "total_changes",
            "root_count",
            "leaf_count",
            "causal_depth",
            "rheology_change_count",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if not isinstance(value, int):
                raise StructuralSignatureError(
                    f"{field_name} must be an integer"
                )

            if value < 0:
                raise StructuralSignatureError(
                    f"{field_name} must be non-negative"
                )

        if self.rheology_change_count > self.total_changes:
            raise StructuralSignatureError(
                "rheology_change_count cannot exceed total_changes"
            )

        if self.total_changes == 0:
            if (
                self.root_count != 0
                or self.leaf_count != 0
                or self.causal_depth != 0
            ):
                raise StructuralSignatureError(
                    "an empty signature cannot have roots, "
                    "leaves, or causal depth"
                )
        else:
            if self.root_count > self.total_changes:
                raise StructuralSignatureError(
                    "root_count cannot exceed total_changes"
                )

            if self.leaf_count > self.total_changes:
                raise StructuralSignatureError(
                    "leaf_count cannot exceed total_changes"
                )

            if self.causal_depth > self.total_changes:
                raise StructuralSignatureError(
                    "causal_depth cannot exceed total_changes"
                )

        start_time = _validate_nonnegative_finite(
            self.start_time,
            field_name="start_time",
        )
        end_time = _validate_nonnegative_finite(
            self.end_time,
            field_name="end_time",
        )
        duration = _validate_nonnegative_finite(
            self.duration,
            field_name="duration",
        )

        if end_time < start_time:
            raise StructuralSignatureError(
                "end_time must not precede start_time"
            )

        expected_duration = end_time - start_time

        if abs(
            duration - expected_duration
        ) > 1e-9:
            raise StructuralSignatureError(
                "duration must equal end_time - start_time"
            )

        total_capacity_loss = (
            _validate_nonnegative_finite(
                self.total_capacity_loss,
                field_name="total_capacity_loss",
            )
        )
        total_capacity_gain = (
            _validate_nonnegative_finite(
                self.total_capacity_gain,
                field_name="total_capacity_gain",
            )
        )
        net_capacity_effect = _validate_finite(
            self.net_capacity_effect,
            field_name="net_capacity_effect",
        )

        expected_net = (
            total_capacity_gain
            - total_capacity_loss
        )

        if abs(
            net_capacity_effect - expected_net
        ) > 1e-9:
            raise StructuralSignatureError(
                "net_capacity_effect must equal "
                "total_capacity_gain - total_capacity_loss"
            )

        cumulative_signature_weight = (
            _validate_nonnegative_finite(
                self.cumulative_signature_weight,
                field_name="cumulative_signature_weight",
            )
        )
        mean_signature_weight = (
            _validate_nonnegative_finite(
                self.mean_signature_weight,
                field_name="mean_signature_weight",
            )
        )

        if self.total_changes == 0:
            if (
                cumulative_signature_weight != 0.0
                or mean_signature_weight != 0.0
            ):
                raise StructuralSignatureError(
                    "empty signatures must have zero "
                    "signature weights"
                )
        else:
            expected_mean = (
                cumulative_signature_weight
                / self.total_changes
            )

            if abs(
                mean_signature_weight - expected_mean
            ) > 1e-9:
                raise StructuralSignatureError(
                    "mean_signature_weight must equal "
                    "cumulative_signature_weight / total_changes"
                )

        persistence_index = _validate_unit_interval(
            self.persistence_index,
            field_name="persistence_index",
        )
        irreversibility_index = _validate_unit_interval(
            self.irreversibility_index,
            field_name="irreversibility_index",
        )
        progression_index = _validate_unit_interval(
            self.progression_index,
            field_name="progression_index",
        )
        adaptation_index = _validate_unit_interval(
            self.adaptation_index,
            field_name="adaptation_index",
        )

        harmful_fraction = _validate_unit_interval(
            self.harmful_fraction,
            field_name="harmful_fraction",
        )
        beneficial_fraction = _validate_unit_interval(
            self.beneficial_fraction,
            field_name="beneficial_fraction",
        )
        mixed_fraction = _validate_unit_interval(
            self.mixed_fraction,
            field_name="mixed_fraction",
        )

        functional_total = (
            harmful_fraction
            + beneficial_fraction
            + mixed_fraction
        )

        if functional_total > 1.0 + 1e-9:
            raise StructuralSignatureError(
                "functional fractions cannot sum above 1"
            )

        total_abs_creep_strain = (
            _validate_nonnegative_finite(
                self.total_abs_creep_strain,
                field_name="total_abs_creep_strain",
            )
        )
        mean_abs_creep_strain = (
            _validate_nonnegative_finite(
                self.mean_abs_creep_strain,
                field_name="mean_abs_creep_strain",
            )
        )
        max_abs_creep_strain = (
            _validate_nonnegative_finite(
                self.max_abs_creep_strain,
                field_name="max_abs_creep_strain",
            )
        )
        mean_abs_residual_strain = (
            _validate_nonnegative_finite(
                self.mean_abs_residual_strain,
                field_name="mean_abs_residual_strain",
            )
        )
        max_abs_residual_strain = (
            _validate_nonnegative_finite(
                self.max_abs_residual_strain,
                field_name="max_abs_residual_strain",
            )
        )
        rheological_memory_index = _validate_unit_interval(
            self.rheological_memory_index,
            field_name="rheological_memory_index",
        )
        rheological_retention_index = _validate_unit_interval(
            self.rheological_retention_index,
            field_name="rheological_retention_index",
        )
        rheological_relaxation_index = _validate_unit_interval(
            self.rheological_relaxation_index,
            field_name="rheological_relaxation_index",
        )

        if self.rheology_change_count == 0:
            rheology_scalars = (
                total_abs_creep_strain,
                mean_abs_creep_strain,
                max_abs_creep_strain,
                mean_abs_residual_strain,
                max_abs_residual_strain,
                rheological_memory_index,
                rheological_retention_index,
                rheological_relaxation_index,
            )
            if any(value != 0.0 for value in rheology_scalars):
                raise StructuralSignatureError(
                    "rheology scalars must be zero when "
                    "rheology_change_count is zero"
                )
            if rheology_profile:
                raise StructuralSignatureError(
                    "rheology_profile must be empty when "
                    "rheology_change_count is zero"
                )
        else:
            expected_mean_creep = (
                total_abs_creep_strain
                / self.rheology_change_count
            )
            if abs(
                mean_abs_creep_strain
                - expected_mean_creep
            ) > 1e-9:
                raise StructuralSignatureError(
                    "mean_abs_creep_strain must equal "
                    "total_abs_creep_strain / "
                    "rheology_change_count"
                )
            if (
                max_abs_creep_strain
                + 1e-12
                < mean_abs_creep_strain
            ):
                raise StructuralSignatureError(
                    "max_abs_creep_strain cannot be below "
                    "mean_abs_creep_strain"
                )
            if (
                max_abs_residual_strain
                + 1e-12
                < mean_abs_residual_strain
            ):
                raise StructuralSignatureError(
                    "max_abs_residual_strain cannot be below "
                    "mean_abs_residual_strain"
                )

        label = self.label

        if label is not None:
            label = _require_nonempty_string(
                label,
                field_name="label",
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(
            self,
            "source_pattern_id",
            source_pattern_id,
        )
        object.__setattr__(
            self,
            "signature_id",
            signature_id,
        )
        object.__setattr__(
            self,
            "plane_profile",
            plane_profile,
        )
        object.__setattr__(
            self,
            "agent_profile",
            agent_profile,
        )
        object.__setattr__(
            self,
            "kind_profile",
            kind_profile,
        )
        object.__setattr__(
            self,
            "time_scale_profile",
            time_scale_profile,
        )
        object.__setattr__(
            self,
            "target_profile",
            target_profile,
        )
        object.__setattr__(
            self,
            "rheology_profile",
            rheology_profile,
        )
        object.__setattr__(
            self,
            "start_time",
            start_time,
        )
        object.__setattr__(
            self,
            "end_time",
            end_time,
        )
        object.__setattr__(
            self,
            "duration",
            duration,
        )
        object.__setattr__(
            self,
            "total_capacity_loss",
            total_capacity_loss,
        )
        object.__setattr__(
            self,
            "total_capacity_gain",
            total_capacity_gain,
        )
        object.__setattr__(
            self,
            "net_capacity_effect",
            net_capacity_effect,
        )
        object.__setattr__(
            self,
            "cumulative_signature_weight",
            cumulative_signature_weight,
        )
        object.__setattr__(
            self,
            "mean_signature_weight",
            mean_signature_weight,
        )
        object.__setattr__(
            self,
            "persistence_index",
            persistence_index,
        )
        object.__setattr__(
            self,
            "irreversibility_index",
            irreversibility_index,
        )
        object.__setattr__(
            self,
            "progression_index",
            progression_index,
        )
        object.__setattr__(
            self,
            "adaptation_index",
            adaptation_index,
        )
        object.__setattr__(
            self,
            "harmful_fraction",
            harmful_fraction,
        )
        object.__setattr__(
            self,
            "beneficial_fraction",
            beneficial_fraction,
        )
        object.__setattr__(
            self,
            "mixed_fraction",
            mixed_fraction,
        )
        object.__setattr__(
            self,
            "total_abs_creep_strain",
            total_abs_creep_strain,
        )
        object.__setattr__(
            self,
            "mean_abs_creep_strain",
            mean_abs_creep_strain,
        )
        object.__setattr__(
            self,
            "max_abs_creep_strain",
            max_abs_creep_strain,
        )
        object.__setattr__(
            self,
            "mean_abs_residual_strain",
            mean_abs_residual_strain,
        )
        object.__setattr__(
            self,
            "max_abs_residual_strain",
            max_abs_residual_strain,
        )
        object.__setattr__(
            self,
            "rheological_memory_index",
            rheological_memory_index,
        )
        object.__setattr__(
            self,
            "rheological_retention_index",
            rheological_retention_index,
        )
        object.__setattr__(
            self,
            "rheological_relaxation_index",
            rheological_relaxation_index,
        )
        object.__setattr__(
            self,
            "label",
            label,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @property
    def is_empty(self) -> bool:
        return self.total_changes == 0

    @property
    def is_progressive(self) -> bool:
        return (
            self.progression_index > 0.5
            and self.total_capacity_loss
            > self.total_capacity_gain
        )

    @property
    def is_adaptive(self) -> bool:
        return (
            self.adaptation_index > 0.5
            and self.total_capacity_gain
            > self.total_capacity_loss
        )

    @property
    def is_balanced(self) -> bool:
        if self.is_empty:
            return True

        return (
            not self.is_progressive
            and not self.is_adaptive
        )

    @property
    def has_rheological_memory(self) -> bool:
        return self.rheology_change_count > 0

    @property
    def rheology_fraction(self) -> float:
        return _safe_ratio(
            float(self.rheology_change_count),
            float(self.total_changes),
        )

    @property
    def dominant_rheology(
        self,
    ) -> tuple[str, float] | None:
        return self._dominant_from_profile(
            self.rheology_profile
        )

    def rheology_vector(self) -> tuple[float, ...]:
        """
        Return rheology-only scalar descriptors.

        Kept separate from compact_vector() so the legacy 21-value vector
        remains backward compatible.
        """
        return (
            float(self.rheology_change_count),
            self.total_abs_creep_strain,
            self.mean_abs_creep_strain,
            self.max_abs_creep_strain,
            self.mean_abs_residual_strain,
            self.max_abs_residual_strain,
            self.rheological_memory_index,
            self.rheological_retention_index,
            self.rheological_relaxation_index,
            self.rheology_fraction,
        )

    @property
    def dominant_plane(
        self,
    ) -> tuple[str, float] | None:
        return self._dominant_from_profile(
            self.plane_profile
        )

    @property
    def dominant_agent(
        self,
    ) -> tuple[str, float] | None:
        return self._dominant_from_profile(
            self.agent_profile
        )

    @property
    def dominant_kind(
        self,
    ) -> tuple[str, float] | None:
        return self._dominant_from_profile(
            self.kind_profile
        )

    @property
    def dominant_time_scale(
        self,
    ) -> tuple[str, float] | None:
        return self._dominant_from_profile(
            self.time_scale_profile
        )

    @staticmethod
    def _dominant_from_profile(
        profile: Mapping[str, float],
    ) -> tuple[str, float] | None:
        if not profile:
            return None

        return max(
            profile.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
        )

    @property
    def capacity_burden(self) -> float:
        """
        Normalized harmful share of total absolute capacity change.
        """

        denominator = (
            self.total_capacity_loss
            + self.total_capacity_gain
        )

        return _safe_ratio(
            self.total_capacity_loss,
            denominator,
        )

    @property
    def capacity_adaptation(self) -> float:
        """
        Normalized beneficial share of total absolute capacity change.
        """

        denominator = (
            self.total_capacity_loss
            + self.total_capacity_gain
        )

        return _safe_ratio(
            self.total_capacity_gain,
            denominator,
        )

    @property
    def causal_branching_index(self) -> float:
        """
        Normalized difference between leaves and roots.

        Zero means roots and leaves are equal in count.
        Values approaching one indicate broader causal branching.
        """

        if self.total_changes <= 1:
            return 0.0

        spread = max(
            self.leaf_count - self.root_count,
            0,
        )

        return min(
            spread / (self.total_changes - 1),
            1.0,
        )

    @property
    def normalized_causal_depth(self) -> float:
        if self.total_changes <= 0:
            return 0.0

        return self.causal_depth / self.total_changes

    def profile_value(
        self,
        profile_name: str,
        key: str,
    ) -> float:
        normalized_profile_name = (
            _require_nonempty_string(
                profile_name,
                field_name="profile_name",
            )
        )
        normalized_key = _require_nonempty_string(
            key,
            field_name="key",
        )

        profiles = {
            "plane": self.plane_profile,
            "agent": self.agent_profile,
            "kind": self.kind_profile,
            "time_scale": self.time_scale_profile,
            "target": self.target_profile,
            "rheology": self.rheology_profile,
        }

        try:
            profile = profiles[
                normalized_profile_name
            ]
        except KeyError as exc:
            raise StructuralSignatureError(
                "unsupported profile name: "
                f"{normalized_profile_name!r}"
            ) from exc

        return float(
            profile.get(
                normalized_key,
                0.0,
            )
        )

    def compact_vector(self) -> tuple[float, ...]:
        """
        Return a stable scalar vector independent of profile vocabulary.

        Categorical profiles remain separate because their dimensions depend
        on the modeled domain.
        """

        return (
            float(self.total_changes),
            float(self.root_count),
            float(self.leaf_count),
            float(self.causal_depth),
            self.duration,
            self.total_capacity_loss,
            self.total_capacity_gain,
            self.net_capacity_effect,
            self.cumulative_signature_weight,
            self.mean_signature_weight,
            self.persistence_index,
            self.irreversibility_index,
            self.progression_index,
            self.adaptation_index,
            self.harmful_fraction,
            self.beneficial_fraction,
            self.mixed_fraction,
            self.capacity_burden,
            self.capacity_adaptation,
            self.causal_branching_index,
            self.normalized_causal_depth,
        )

    @staticmethod
    def _profile_distance(
        left: Mapping[str, float],
        right: Mapping[str, float],
    ) -> float:
        """
        Return normalized L1 distance between probability profiles.

        The result is in [0, 1].
        """

        keys = set(left) | set(right)

        if not keys:
            return 0.0

        l1 = sum(
            abs(
                float(left.get(key, 0.0))
                - float(right.get(key, 0.0))
            )
            for key in keys
        )

        return min(
            l1 / 2.0,
            1.0,
        )

    @staticmethod
    def _bounded_scalar_distance(
        left: float,
        right: float,
    ) -> float:
        return min(
            abs(left - right),
            1.0,
        )

    @staticmethod
    def _relative_nonnegative_distance(
        left: float,
        right: float,
    ) -> float:
        denominator = max(
            abs(left),
            abs(right),
            1.0,
        )

        return min(
            abs(left - right) / denominator,
            1.0,
        )

    def compare(
        self,
        other: StructuralSignature,
        *,
        weights: SignatureDistanceWeights | None = None,
    ) -> SignatureComparison:
        """
        Compare this signature with another signature.

        The comparison is symmetric and normalized to [0, 1].
        """

        if not isinstance(
            other,
            StructuralSignature,
        ):
            raise StructuralSignatureError(
                "other must be a StructuralSignature"
            )

        if weights is None:
            weights = SignatureDistanceWeights()

        if not isinstance(
            weights,
            SignatureDistanceWeights,
        ):
            raise StructuralSignatureError(
                "weights must be SignatureDistanceWeights"
            )

        capacity_distance = (
            self._relative_nonnegative_distance(
                self.total_capacity_loss,
                other.total_capacity_loss,
            )
            + self._relative_nonnegative_distance(
                self.total_capacity_gain,
                other.total_capacity_gain,
            )
            + self._relative_nonnegative_distance(
                self.net_capacity_effect,
                other.net_capacity_effect,
            )
        ) / 3.0

        causality_distance = (
            self._relative_nonnegative_distance(
                float(self.root_count),
                float(other.root_count),
            )
            + self._relative_nonnegative_distance(
                float(self.leaf_count),
                float(other.leaf_count),
            )
            + self._relative_nonnegative_distance(
                float(self.causal_depth),
                float(other.causal_depth),
            )
            + self._bounded_scalar_distance(
                self.causal_branching_index,
                other.causal_branching_index,
            )
            + self._bounded_scalar_distance(
                self.normalized_causal_depth,
                other.normalized_causal_depth,
            )
        ) / 5.0

        chronology_distance = (
            self._relative_nonnegative_distance(
                self.duration,
                other.duration,
            )
            + self._relative_nonnegative_distance(
                float(self.total_changes),
                float(other.total_changes),
            )
        ) / 2.0

        rheology_scalar_distance = (
            self._relative_nonnegative_distance(
                float(self.rheology_change_count),
                float(other.rheology_change_count),
            )
            + self._relative_nonnegative_distance(
                self.total_abs_creep_strain,
                other.total_abs_creep_strain,
            )
            + self._relative_nonnegative_distance(
                self.max_abs_creep_strain,
                other.max_abs_creep_strain,
            )
            + self._relative_nonnegative_distance(
                self.max_abs_residual_strain,
                other.max_abs_residual_strain,
            )
            + self._bounded_scalar_distance(
                self.rheological_memory_index,
                other.rheological_memory_index,
            )
            + self._bounded_scalar_distance(
                self.rheological_retention_index,
                other.rheological_retention_index,
            )
            + self._bounded_scalar_distance(
                self.rheological_relaxation_index,
                other.rheological_relaxation_index,
            )
        ) / 7.0

        rheology_distance = (
            self._profile_distance(
                self.rheology_profile,
                other.rheology_profile,
            )
            + rheology_scalar_distance
        ) / 2.0

        components = {
            "plane_profile": self._profile_distance(
                self.plane_profile,
                other.plane_profile,
            ),
            "agent_profile": self._profile_distance(
                self.agent_profile,
                other.agent_profile,
            ),
            "kind_profile": self._profile_distance(
                self.kind_profile,
                other.kind_profile,
            ),
            "time_scale_profile": self._profile_distance(
                self.time_scale_profile,
                other.time_scale_profile,
            ),
            "target_profile": self._profile_distance(
                self.target_profile,
                other.target_profile,
            ),
            "rheology_profile": self._profile_distance(
    self.rheology_profile,
    other.rheology_profile,
),
            "rheology": rheology_distance,
            "capacity": capacity_distance,
            "persistence": (
                self._bounded_scalar_distance(
                    self.persistence_index,
                    other.persistence_index,
                )
            ),
            "irreversibility": (
                (
                    self._bounded_scalar_distance(
                        self.irreversibility_index,
                        other.irreversibility_index,
                    )
                    + rheology_distance
                ) / 2.0
                if (
                    self.has_rheological_memory
                    or other.has_rheological_memory
                )
                else self._bounded_scalar_distance(
                    self.irreversibility_index,
                    other.irreversibility_index,
                )
            ),
            "progression": (
                self._bounded_scalar_distance(
                    self.progression_index,
                    other.progression_index,
                )
            ),
            "adaptation": (
                self._bounded_scalar_distance(
                    self.adaptation_index,
                    other.adaptation_index,
                )
            ),
            "causality": causality_distance,
            "chronology": chronology_distance,
        }

        weighted_sum = (
            components["plane_profile"]
            * weights.plane_profile
            + components["agent_profile"]
            * weights.agent_profile
            + components["kind_profile"]
            * weights.kind_profile
            + components["time_scale_profile"]
            * weights.time_scale_profile
            + components["target_profile"]
            * weights.target_profile
            + components["capacity"]
            * weights.capacity
            + components["persistence"]
            * weights.persistence
            + components["irreversibility"]
            * weights.irreversibility
            + components["progression"]
            * weights.progression
            + components["adaptation"]
            * weights.adaptation
            + components["causality"]
            * weights.causality
            + components["chronology"]
            * weights.chronology
        )

        distance = min(
            weighted_sum / weights.total_weight,
            1.0,
        )

        return SignatureComparison(
            left_signature_id=self.signature_id,
            right_signature_id=other.signature_id,
            distance=distance,
            similarity=1.0 - distance,
            components=components,
            weights=weights,
        )

    def euclidean_scalar_distance(
        self,
        other: StructuralSignature,
    ) -> float:
        """
        Return Euclidean distance between compact scalar vectors.

        This distance is not normalized and is primarily intended for
        diagnostics, clustering experiments, and baseline comparisons.
        """

        if not isinstance(
            other,
            StructuralSignature,
        ):
            raise StructuralSignatureError(
                "other must be a StructuralSignature"
            )

        left_vector = self.compact_vector()
        right_vector = other.compact_vector()

        return sqrt(
            sum(
                (left - right) ** 2
                for left, right in zip(
                    left_vector,
                    right_vector,
                    strict=True,
                )
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "signature_id": self.signature_id,
            "source_pattern_id": self.source_pattern_id,
            "label": self.label,
            "total_changes": self.total_changes,
            "duration": self.duration,
            "causal_depth": self.causal_depth,
            "root_count": self.root_count,
            "leaf_count": self.leaf_count,
            "total_capacity_loss": (
                self.total_capacity_loss
            ),
            "total_capacity_gain": (
                self.total_capacity_gain
            ),
            "net_capacity_effect": (
                self.net_capacity_effect
            ),
            "persistence_index": (
                self.persistence_index
            ),
            "irreversibility_index": (
                self.irreversibility_index
            ),
            "progression_index": (
                self.progression_index
            ),
            "adaptation_index": (
                self.adaptation_index
            ),
            "capacity_burden": self.capacity_burden,
            "capacity_adaptation": (
                self.capacity_adaptation
            ),
            "causal_branching_index": (
                self.causal_branching_index
            ),
            "normalized_causal_depth": (
                self.normalized_causal_depth
            ),
            "is_progressive": self.is_progressive,
            "is_adaptive": self.is_adaptive,
            "is_balanced": self.is_balanced,
            "dominant_plane": self.dominant_plane,
            "dominant_agent": self.dominant_agent,
            "dominant_kind": self.dominant_kind,
            "dominant_time_scale": (
                self.dominant_time_scale
            ),
            "has_rheological_memory": (
                self.has_rheological_memory
            ),
            "rheology_change_count": (
                self.rheology_change_count
            ),
            "rheology_fraction": self.rheology_fraction,
            "rheological_memory_index": (
                self.rheological_memory_index
            ),
            "rheological_retention_index": (
                self.rheological_retention_index
            ),
            "rheological_relaxation_index": (
                self.rheological_relaxation_index
            ),
            "dominant_rheology": self.dominant_rheology,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "signature_id": self.signature_id,
            "source_pattern_id": self.source_pattern_id,
            "label": self.label,
            "plane_profile": _mapping_to_dict(
                self.plane_profile
            ),
            "agent_profile": _mapping_to_dict(
                self.agent_profile
            ),
            "kind_profile": _mapping_to_dict(
                self.kind_profile
            ),
            "time_scale_profile": _mapping_to_dict(
                self.time_scale_profile
            ),
            "target_profile": _mapping_to_dict(
                self.target_profile
            ),
            "rheology_profile": _mapping_to_dict(
                self.rheology_profile
            ),
            "total_changes": self.total_changes,
            "root_count": self.root_count,
            "leaf_count": self.leaf_count,
            "causal_depth": self.causal_depth,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "total_capacity_loss": (
                self.total_capacity_loss
            ),
            "total_capacity_gain": (
                self.total_capacity_gain
            ),
            "net_capacity_effect": (
                self.net_capacity_effect
            ),
            "cumulative_signature_weight": (
                self.cumulative_signature_weight
            ),
            "mean_signature_weight": (
                self.mean_signature_weight
            ),
            "persistence_index": (
                self.persistence_index
            ),
            "irreversibility_index": (
                self.irreversibility_index
            ),
            "progression_index": (
                self.progression_index
            ),
            "adaptation_index": (
                self.adaptation_index
            ),
            "harmful_fraction": (
                self.harmful_fraction
            ),
            "beneficial_fraction": (
                self.beneficial_fraction
            ),
            "mixed_fraction": self.mixed_fraction,
            "rheology_change_count": (
                self.rheology_change_count
            ),
            "total_abs_creep_strain": (
                self.total_abs_creep_strain
            ),
            "mean_abs_creep_strain": (
                self.mean_abs_creep_strain
            ),
            "max_abs_creep_strain": (
                self.max_abs_creep_strain
            ),
            "mean_abs_residual_strain": (
                self.mean_abs_residual_strain
            ),
            "max_abs_residual_strain": (
                self.max_abs_residual_strain
            ),
            "rheological_memory_index": (
                self.rheological_memory_index
            ),
            "rheological_retention_index": (
                self.rheological_retention_index
            ),
            "rheological_relaxation_index": (
                self.rheological_relaxation_index
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> StructuralSignature:
        if not isinstance(data, Mapping):
            raise StructuralSignatureError(
                "structural signature data must be a mapping"
            )

        version = data.get(
            "version",
            cls.VERSION,
        )

        if version != cls.VERSION:
            raise StructuralSignatureError(
                "unsupported structural signature version: "
                f"{version!r}"
            )

        return cls(
            signature_id=str(
                data["signature_id"]
            ),
            source_pattern_id=str(
                data["source_pattern_id"]
            ),
            label=data.get("label"),
            plane_profile=data.get(
                "plane_profile",
                {},
            ),
            agent_profile=data.get(
                "agent_profile",
                {},
            ),
            kind_profile=data.get(
                "kind_profile",
                {},
            ),
            time_scale_profile=data.get(
                "time_scale_profile",
                {},
            ),
            target_profile=data.get(
                "target_profile",
                {},
            ),
            rheology_profile=data.get(
                "rheology_profile",
                {},
            ),
            total_changes=int(
                data.get("total_changes", 0)
            ),
            root_count=int(
                data.get("root_count", 0)
            ),
            leaf_count=int(
                data.get("leaf_count", 0)
            ),
            causal_depth=int(
                data.get("causal_depth", 0)
            ),
            start_time=float(
                data.get("start_time", 0.0)
            ),
            end_time=float(
                data.get("end_time", 0.0)
            ),
            duration=float(
                data.get("duration", 0.0)
            ),
            total_capacity_loss=float(
                data.get(
                    "total_capacity_loss",
                    0.0,
                )
            ),
            total_capacity_gain=float(
                data.get(
                    "total_capacity_gain",
                    0.0,
                )
            ),
            net_capacity_effect=float(
                data.get(
                    "net_capacity_effect",
                    0.0,
                )
            ),
            cumulative_signature_weight=float(
                data.get(
                    "cumulative_signature_weight",
                    0.0,
                )
            ),
            mean_signature_weight=float(
                data.get(
                    "mean_signature_weight",
                    0.0,
                )
            ),
            persistence_index=float(
                data.get(
                    "persistence_index",
                    0.0,
                )
            ),
            irreversibility_index=float(
                data.get(
                    "irreversibility_index",
                    0.0,
                )
            ),
            progression_index=float(
                data.get(
                    "progression_index",
                    0.0,
                )
            ),
            adaptation_index=float(
                data.get(
                    "adaptation_index",
                    0.0,
                )
            ),
            harmful_fraction=float(
                data.get(
                    "harmful_fraction",
                    0.0,
                )
            ),
            beneficial_fraction=float(
                data.get(
                    "beneficial_fraction",
                    0.0,
                )
            ),
            mixed_fraction=float(
                data.get(
                    "mixed_fraction",
                    0.0,
                )
            ),
            rheology_change_count=int(
                data.get("rheology_change_count", 0)
            ),
            total_abs_creep_strain=float(
                data.get("total_abs_creep_strain", 0.0)
            ),
            mean_abs_creep_strain=float(
                data.get("mean_abs_creep_strain", 0.0)
            ),
            max_abs_creep_strain=float(
                data.get("max_abs_creep_strain", 0.0)
            ),
            mean_abs_residual_strain=float(
                data.get(
                    "mean_abs_residual_strain",
                    0.0,
                )
            ),
            max_abs_residual_strain=float(
                data.get(
                    "max_abs_residual_strain",
                    0.0,
                )
            ),
            rheological_memory_index=float(
                data.get(
                    "rheological_memory_index",
                    0.0,
                )
            ),
            rheological_retention_index=float(
                data.get(
                    "rheological_retention_index",
                    0.0,
                )
            ),
            rheological_relaxation_index=float(
                data.get(
                    "rheological_relaxation_index",
                    0.0,
                )
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    @classmethod
    def from_pattern(
        cls,
        pattern: HistoryPattern,
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        signature_id: str | None = None,
    ) -> StructuralSignature:
        """
        Build a StructuralSignature from a HistoryPattern.
        """

        if not isinstance(
            pattern,
            HistoryPattern,
        ):
            raise StructuralSignatureError(
                "pattern must be a HistoryPattern"
            )

        total_functional_weight = (
            pattern.harmful_weight
            + pattern.beneficial_weight
            + pattern.mixed_weight
        )

        harmful_fraction = _safe_ratio(
            pattern.harmful_weight,
            total_functional_weight,
        )
        beneficial_fraction = _safe_ratio(
            pattern.beneficial_weight,
            total_functional_weight,
        )
        mixed_fraction = _safe_ratio(
            pattern.mixed_weight,
            total_functional_weight,
        )

        rheology_memories = [
            change.rheology_memory
            for change in pattern.changes
            if getattr(
                change,
                "rheology_memory",
                None,
            ) is not None
        ]
        rheology_change_count = len(rheology_memories)

        abs_creep_values = [
            abs(memory.creep_strain)
            for memory in rheology_memories
        ]
        abs_residual_values = [
            abs(memory.residual_strain)
            for memory in rheology_memories
        ]
        total_abs_creep_strain = sum(abs_creep_values)
        mean_abs_creep_strain = _safe_ratio(
            total_abs_creep_strain,
            float(rheology_change_count),
        )
        max_abs_creep_strain = max(
            abs_creep_values,
            default=0.0,
        )
        mean_abs_residual_strain = _safe_ratio(
            sum(abs_residual_values),
            float(rheology_change_count),
        )
        max_abs_residual_strain = max(
            abs_residual_values,
            default=0.0,
        )

        rheological_memory_index = _safe_ratio(
            sum(
                memory.memory_index
                for memory in rheology_memories
            ),
            float(rheology_change_count),
        )
        rheological_retention_index = _safe_ratio(
            sum(
                memory.strain_retention_ratio
                for memory in rheology_memories
            ),
            float(rheology_change_count),
        )
        rheological_relaxation_index = _safe_ratio(
            sum(
                memory.relaxation_index
                for memory in rheology_memories
            ),
            float(rheology_change_count),
        )

        rheology_scores: dict[str, float] = {}
        for memory in rheology_memories:
            weight = max(
                memory.memory_index,
                memory.confidence * 1e-12,
            )
            model_key = f"model:{memory.model.value}"
            phase_key = f"phase:{memory.phase.value}"
            rheology_scores[model_key] = (
                rheology_scores.get(model_key, 0.0)
                + weight
            )
            rheology_scores[phase_key] = (
                rheology_scores.get(phase_key, 0.0)
                + weight
            )

        merged_metadata: dict[str, Any] = {
            "source_pattern_label": pattern.label,
            "source_pattern_description": (
                pattern.description
            ),
            "source_pattern_external_causes": list(
                pattern.external_cause_ids
            ),
        }

        if metadata is not None:
            merged_metadata.update(
                dict(metadata)
            )

        constructor_args: dict[str, Any] = {
            "source_pattern_id": pattern.pattern_id,
            "label": (
                label
                if label is not None
                else pattern.label
            ),
            "plane_profile": _profile_from_ranked(
                pattern.plane_profile()
            ),
            "agent_profile": _profile_from_ranked(
                pattern.agent_profile()
            ),
            "kind_profile": _profile_from_ranked(
                pattern.kind_profile()
            ),
            "time_scale_profile": _profile_from_ranked(
                pattern.time_scale_profile()
            ),
            "target_profile": _profile_from_ranked(
                pattern.target_profile()
            ),
            "rheology_profile": rheology_scores,
            "total_changes": pattern.total_changes,
            "root_count": len(
                pattern.root_changes
            ),
            "leaf_count": len(
                pattern.leaf_changes
            ),
            "causal_depth": pattern.causal_depth,
            "start_time": pattern.start_time,
            "end_time": pattern.end_time,
            "duration": pattern.duration,
            "total_capacity_loss": (
                pattern.total_capacity_loss
            ),
            "total_capacity_gain": (
                pattern.total_capacity_gain
            ),
            "net_capacity_effect": (
                pattern.net_capacity_effect
            ),
            "cumulative_signature_weight": (
                pattern.cumulative_signature_weight
            ),
            "mean_signature_weight": (
                pattern.mean_signature_weight
            ),
            "persistence_index": (
                pattern.persistence_index
            ),
            "irreversibility_index": (
                pattern.irreversibility_index
            ),
            "progression_index": (
                pattern.progression_index
            ),
            "adaptation_index": (
                pattern.adaptation_index
            ),
            "harmful_fraction": harmful_fraction,
            "beneficial_fraction": (
                beneficial_fraction
            ),
            "mixed_fraction": mixed_fraction,
            "rheology_change_count": (
                rheology_change_count
            ),
            "total_abs_creep_strain": (
                total_abs_creep_strain
            ),
            "mean_abs_creep_strain": (
                mean_abs_creep_strain
            ),
            "max_abs_creep_strain": (
                max_abs_creep_strain
            ),
            "mean_abs_residual_strain": (
                mean_abs_residual_strain
            ),
            "max_abs_residual_strain": (
                max_abs_residual_strain
            ),
            "rheological_memory_index": (
                rheological_memory_index
            ),
            "rheological_retention_index": (
                rheological_retention_index
            ),
            "rheological_relaxation_index": (
                rheological_relaxation_index
            ),
            "metadata": merged_metadata,
        }

        if signature_id is not None:
            constructor_args["signature_id"] = (
                signature_id
            )

        return cls(**constructor_args)


__all__ = [
    "SignatureComparison",
    "SignatureDistanceWeights",
    "StructuralSignature",
    "StructuralSignatureError",
]