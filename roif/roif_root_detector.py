"""
ROIF Engine
Root and Role Detector

This module separates four causally different roles in a recursive cascade:

D_origin
    Earliest identifiable origin of the observed cascade.

D_fast
    Functional channel with the earliest or greatest reserve depletion in the
    observed trajectory. It is the first visible failure or symptom role and
    is not automatically treated as the root.

D_root
    Channel whose virtual restoration produces the greatest reduction of the
    observed global cascade burden.

Node*
    Best intervention point after accounting for expected cascade reduction,
    intervention cost, collateral effects, uncertainty, and safety risk.

The detector operates on:

- ROIFSystem;
- CapacityTensor;
- CascadeTrajectory;
- FunctionalChannel state;
- plane-resolved tensor contributions;
- geometry, material, history, reserve, and activation factors.

The module is domain-independent. It can analyse biological, engineering,
informational, behavioural, organisational, or control systems.

This is a research-prototype computation layer. It does not diagnose,
prescribe treatment, or authorize real-world interventions.

Author:
    Architect (Dctr Pain)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
import math

import numpy as np

from roif.utilization import compute_utilization

from .roif_entities import (
    FunctionalChannel,
    ROIFSystem,
)
from .roif_tensor import (
    CapacityTensor,
    ChannelTensorState,
    TensorEntryFactors,
    spectral_radius,
)
from .roif_cascade import (
    CascadeTrajectory,
    EventKind,
)


class ROIFRootDetectorError(ValueError):
    """Raised when root detection cannot be performed safely."""


class ROIFRole(str, Enum):
    """Independent causal roles identified by the detector."""

    D_ORIGIN = "d_origin"
    D_FAST = "d_fast"
    D_ROOT = "d_root"
    NODE_STAR = "node_star"


class RootDetectionMode(str, Enum):
    """
    Main strategy used to estimate D_root.

    TRAJECTORY_SENSITIVITY
        Uses the observed trajectory and tensor influence.

    VIRTUAL_RESTORATION
        Removes or attenuates one candidate from the tensor and compares the
        resulting cascade burden.

    HYBRID
        Combines trajectory sensitivity and virtual restoration.
    """

    TRAJECTORY_SENSITIVITY = "trajectory_sensitivity"
    VIRTUAL_RESTORATION = "virtual_restoration"
    HYBRID = "hybrid"


class FastDetectionMode(str, Enum):
    """Strategy used to identify D_fast."""

    MINIMUM_RESERVE = "minimum_reserve"
    EARLIEST_THRESHOLD = "earliest_threshold"
    PEAK_DEFICIT = "peak_deficit"
    CAPACITY_EXCEEDANCE = "capacity_exceedance"
    HYBRID = "hybrid"


class OriginDetectionMode(str, Enum):
    """Strategy used to estimate D_origin."""

    EARLIEST_ACTIVITY = "earliest_activity"
    EARLIEST_EVENT = "earliest_event"
    UPSTREAM_ACTIVITY = "upstream_activity"
    HYBRID = "hybrid"


class InterventionPolicy(str, Enum):
    """
    Safety policy applied while ranking Node*.

    ADAPTIVE
        Allows temporary overload when the simulated system can redistribute
        the cascade and recover.

    STRICT_LINEAR
        Treats critical threshold crossing as a hard safety penalty. Intended
        for systems in which post-failure adaptation must not be assumed.

    OBSERVATIONAL
        Computes rankings but does not interpret them as executable actions.
    """

    ADAPTIVE = "adaptive"
    STRICT_LINEAR = "strict_linear"
    OBSERVATIONAL = "observational"


class CandidateStatus(str, Enum):
    """Status assigned to one candidate evaluation."""

    VALID = "valid"
    EXCLUDED = "excluded"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSAFE = "unsafe"


class ScoreNormalization(str, Enum):
    """Normalization used across candidate scores."""

    NONE = "none"
    MIN_MAX = "min_max"
    L1 = "l1"
    MAX_ABS = "max_abs"


class InterventionKind(str, Enum):
    """Reduced-order virtual intervention applied to a candidate."""

    RESTORE_CHANNEL = "restore_channel"
    REMOVE_OUTGOING_INFLUENCE = "remove_outgoing_influence"
    REDUCE_OUTGOING_INFLUENCE = "reduce_outgoing_influence"
    REDUCE_INCOMING_LOAD = "reduce_incoming_load"
    RESTORE_RESERVE = "restore_reserve"
    CUSTOM = "custom"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ROIFRootDetectorError(
            f"{name} must be numeric."
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIFRootDetectorError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(result):
        raise ROIFRootDetectorError(
            f"{name} must be finite."
        )

    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)

    if result < 0.0:
        raise ROIFRootDetectorError(
            f"{name} cannot be negative."
        )

    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)

    if result <= 0.0:
        raise ROIFRootDetectorError(
            f"{name} must be positive."
        )

    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)

    if not 0.0 <= result <= 1.0:
        raise ROIFRootDetectorError(
            f"{name} must be within [0, 1]."
        )

    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ROIFRootDetectorError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _readonly_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise ROIFRootDetectorError(
            "metadata must be a mapping."
        )

    return MappingProxyType(dict(value))


def _readonly_float_mapping(
    value: Mapping[str, Any] | None,
    name: str,
) -> Mapping[str, float]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise ROIFRootDetectorError(
            f"{name} must be a mapping."
        )

    return MappingProxyType(
        {
            str(key): _finite(
                item,
                f"{name}[{key!r}]",
            )
            for key, item in value.items()
        }
    )


def _readonly_vector(
    value: Any,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ROIFRootDetectorError(
            f"{name} must be numeric."
        ) from exc

    if array.ndim != 1:
        raise ROIFRootDetectorError(
            f"{name} must be one-dimensional."
        )

    if not np.all(np.isfinite(array)):
        raise ROIFRootDetectorError(
            f"{name} must contain finite values."
        )

    result = np.array(
        array,
        dtype=float,
        copy=True,
    )
    result.setflags(write=False)
    return result


def _readonly_matrix(
    value: Any,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ROIFRootDetectorError(
            f"{name} must be numeric."
        ) from exc

    if array.ndim != 2:
        raise ROIFRootDetectorError(
            f"{name} must be two-dimensional."
        )

    if not np.all(np.isfinite(array)):
        raise ROIFRootDetectorError(
            f"{name} must contain finite values."
        )

    result = np.array(
        array,
        dtype=float,
        copy=True,
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RootDetectorWeights:
    """
    Weights used for causal and intervention rankings.

    D_root is driven primarily by:

    - virtual cascade reduction;
    - tensor sensitivity;
    - downstream reach;
    - trajectory exposure;
    - plane contributions;
    - material, geometry, and history effects.

    Node* additionally includes:

    - intervention cost;
    - collateral effect;
    - uncertainty;
    - irreversible-action risk;
    - safety-policy penalty.
    """

    cascade_reduction: float = 1.0
    tensor_sensitivity: float = 0.80
    downstream_reach: float = 0.50
    trajectory_exposure: float = 0.50
    early_activity: float = 0.35

    reserve_deficit: float = 0.70
    activation_deficit: float = 0.30
    geometry_deficit: float = 0.45
    material_deficit: float = 0.45
    history_effect: float = 0.40
    plane_contribution: float = 0.60

    intervention_gain: float = 1.0
    intervention_cost: float = 0.40
    collateral_effect: float = 0.70
    uncertainty: float = 0.60
    safety_risk: float = 1.0
    irreversibility: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "cascade_reduction",
            "tensor_sensitivity",
            "downstream_reach",
            "trajectory_exposure",
            "early_activity",
            "reserve_deficit",
            "activation_deficit",
            "geometry_deficit",
            "material_deficit",
            "history_effect",
            "plane_contribution",
            "intervention_gain",
            "intervention_cost",
            "collateral_effect",
            "uncertainty",
            "safety_risk",
            "irreversibility",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(
                    getattr(self, name),
                    name,
                ),
            )


@dataclass(frozen=True, slots=True)
class RootDetectorThresholds:
    """Thresholds used by the detector."""

    activity_threshold: float = 1e-9
    fast_threshold: float = 0.50
    failure_threshold: float = 1.0
    minimum_candidate_confidence: float = 0.0

    maximum_acceptable_collateral: float = 1.0
    maximum_acceptable_safety_risk: float = 1.0
    maximum_acceptable_uncertainty: float = 1.0
    irreversible_action_threshold: float = 0.05

    numerical_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "activity_threshold",
            _nonnegative(
                self.activity_threshold,
                "activity_threshold",
            ),
        )
        object.__setattr__(
            self,
            "fast_threshold",
            _nonnegative(
                self.fast_threshold,
                "fast_threshold",
            ),
        )
        object.__setattr__(
            self,
            "failure_threshold",
            _positive(
                self.failure_threshold,
                "failure_threshold",
            ),
        )

        for name in (
            "minimum_candidate_confidence",
            "maximum_acceptable_collateral",
            "maximum_acceptable_safety_risk",
            "maximum_acceptable_uncertainty",
            "irreversible_action_threshold",
        ):
            object.__setattr__(
                self,
                name,
                _unit(
                    getattr(self, name),
                    name,
                ),
            )

        object.__setattr__(
            self,
            "numerical_epsilon",
            _positive(
                self.numerical_epsilon,
                "numerical_epsilon",
            ),
        )


@dataclass(frozen=True, slots=True)
class RootDetectorConfig:
    """Complete configuration for ROIF role detection."""

    root_mode: RootDetectionMode = RootDetectionMode.HYBRID
    fast_mode: FastDetectionMode = FastDetectionMode.HYBRID
    origin_mode: OriginDetectionMode = OriginDetectionMode.HYBRID

    intervention_policy: InterventionPolicy = (
        InterventionPolicy.OBSERVATIONAL
    )
    intervention_kind: InterventionKind = (
        InterventionKind.RESTORE_CHANNEL
    )
    normalization: ScoreNormalization = ScoreNormalization.MIN_MAX

    weights: RootDetectorWeights = field(
        default_factory=RootDetectorWeights
    )
    thresholds: RootDetectorThresholds = field(
        default_factory=RootDetectorThresholds
    )

    restoration_fraction: float = 1.0
    outgoing_reduction_fraction: float = 1.0
    incoming_reduction_fraction: float = 0.0

    include_plane_contributions: bool = True
    include_material_contribution: bool = True
    include_geometry_contribution: bool = True
    include_history_contribution: bool = True
    include_activation_contribution: bool = True
    include_tensor_sensitivity: bool = True

    allow_same_role_node: bool = True
    exclude_failed_candidates: bool = False
    strict_missing_data: bool = False

    candidate_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()
    plane_weights: Mapping[str, float] = field(
        default_factory=dict
    )
    intervention_costs: Mapping[str, float] = field(
        default_factory=dict
    )
    intervention_risks: Mapping[str, float] = field(
        default_factory=dict
    )
    intervention_uncertainties: Mapping[str, float] = field(
        default_factory=dict
    )
    intervention_irreversibility: Mapping[str, float] = field(
        default_factory=dict
    )

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        enum_fields = (
            ("root_mode", self.root_mode, RootDetectionMode),
            ("fast_mode", self.fast_mode, FastDetectionMode),
            ("origin_mode", self.origin_mode, OriginDetectionMode),
            (
                "intervention_policy",
                self.intervention_policy,
                InterventionPolicy,
            ),
            (
                "intervention_kind",
                self.intervention_kind,
                InterventionKind,
            ),
            (
                "normalization",
                self.normalization,
                ScoreNormalization,
            ),
        )

        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise ROIFRootDetectorError(
                    f"{name} must be {enum_type.__name__}."
                )

        if not isinstance(self.weights, RootDetectorWeights):
            raise ROIFRootDetectorError(
                "weights must be RootDetectorWeights."
            )

        if not isinstance(
            self.thresholds,
            RootDetectorThresholds,
        ):
            raise ROIFRootDetectorError(
                "thresholds must be RootDetectorThresholds."
            )

        for name in (
            "restoration_fraction",
            "outgoing_reduction_fraction",
            "incoming_reduction_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _unit(
                    getattr(self, name),
                    name,
                ),
            )

        for name in (
            "include_plane_contributions",
            "include_material_contribution",
            "include_geometry_contribution",
            "include_history_contribution",
            "include_activation_contribution",
            "include_tensor_sensitivity",
            "allow_same_role_node",
            "exclude_failed_candidates",
            "strict_missing_data",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ROIFRootDetectorError(
                    f"{name} must be bool."
                )

        candidate_ids = tuple(
            _text(value, "candidate_id")
            for value in self.candidate_ids
        )
        excluded_ids = tuple(
            _text(value, "excluded_id")
            for value in self.excluded_ids
        )

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ROIFRootDetectorError(
                "candidate_ids must be unique."
            )

        if len(excluded_ids) != len(set(excluded_ids)):
            raise ROIFRootDetectorError(
                "excluded_ids must be unique."
            )

        overlap = set(candidate_ids) & set(excluded_ids)
        if overlap:
            raise ROIFRootDetectorError(
                "candidate_ids and excluded_ids overlap: "
                f"{sorted(overlap)!r}."
            )

        object.__setattr__(
            self,
            "candidate_ids",
            candidate_ids,
        )
        object.__setattr__(
            self,
            "excluded_ids",
            excluded_ids,
        )

        object.__setattr__(
            self,
            "plane_weights",
            _readonly_float_mapping(
                self.plane_weights,
                "plane_weights",
            ),
        )
        object.__setattr__(
            self,
            "intervention_costs",
            _readonly_float_mapping(
                self.intervention_costs,
                "intervention_costs",
            ),
        )
        object.__setattr__(
            self,
            "intervention_risks",
            _readonly_float_mapping(
                self.intervention_risks,
                "intervention_risks",
            ),
        )
        object.__setattr__(
            self,
            "intervention_uncertainties",
            _readonly_float_mapping(
                self.intervention_uncertainties,
                "intervention_uncertainties",
            ),
        )
        object.__setattr__(
            self,
            "intervention_irreversibility",
            _readonly_float_mapping(
                self.intervention_irreversibility,
                "intervention_irreversibility",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class RoleEvidence:
    """One auditable evidence component supporting a candidate role."""

    evidence_id: str
    role: ROIFRole
    value: float
    weight: float
    weighted_value: float
    description: str
    plane_id: str | None = None
    time_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _text(self.evidence_id, "evidence_id"),
        )

        if not isinstance(self.role, ROIFRole):
            raise ROIFRootDetectorError(
                "role must be ROIFRole."
            )

        for name in (
            "value",
            "weight",
            "weighted_value",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )

        object.__setattr__(
            self,
            "description",
            _text(self.description, "description"),
        )

        if self.plane_id is not None:
            object.__setattr__(
                self,
                "plane_id",
                _text(self.plane_id, "plane_id"),
            )

        if self.time_index is not None:
            if (
                isinstance(self.time_index, bool)
                or not isinstance(self.time_index, int)
                or self.time_index < 0
            ):
                raise ROIFRootDetectorError(
                    "time_index must be a nonnegative integer."
                )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class CandidateIntervention:
    """Counterfactual intervention summary used for D_root and Node*."""

    candidate_id: str
    kind: InterventionKind

    baseline_burden: float
    counterfactual_burden: float
    cascade_reduction: float
    relative_reduction: float

    baseline_spectral_radius: float
    counterfactual_spectral_radius: float
    spectral_gain: float

    affected_channel_count: int
    collateral_effect: float
    intervention_cost: float
    safety_risk: float
    uncertainty: float
    irreversibility: float

    safe: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _text(self.candidate_id, "candidate_id"),
        )

        if not isinstance(self.kind, InterventionKind):
            raise ROIFRootDetectorError(
                "kind must be InterventionKind."
            )

        for name in (
            "baseline_burden",
            "counterfactual_burden",
            "cascade_reduction",
            "relative_reduction",
            "baseline_spectral_radius",
            "counterfactual_spectral_radius",
            "spectral_gain",
            "collateral_effect",
            "intervention_cost",
            "safety_risk",
            "uncertainty",
            "irreversibility",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )

        if (
            isinstance(self.affected_channel_count, bool)
            or not isinstance(self.affected_channel_count, int)
            or self.affected_channel_count < 0
        ):
            raise ROIFRootDetectorError(
                "affected_channel_count must be a "
                "nonnegative integer."
            )

        if not isinstance(self.safe, bool):
            raise ROIFRootDetectorError(
                "safe must be bool."
            )

        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class RootCandidate:
    """Complete role evaluation for one functional channel."""

    channel_id: str
    entity_id: str
    status: CandidateStatus

    d_origin_score: float
    d_fast_score: float
    d_root_score: float
    node_star_score: float

    reserve_factor: float
    reserve_deficit: float
    activation_factor: float
    activation_deficit: float
    geometry_factor: float
    geometry_deficit: float
    material_factor: float
    material_deficit: float
    history_factor: float
    history_effect: float

    first_activity_index: int | None
    first_threshold_index: int | None
    peak_value: float
    total_exposure: float

    outgoing_strength: float
    incoming_strength: float
    downstream_reach: float
    tensor_sensitivity: float
    plane_contribution: float

    confidence: float
    intervention: CandidateIntervention | None = None
    evidence: tuple[RoleEvidence, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_id",
            _text(self.channel_id, "channel_id"),
        )
        object.__setattr__(
            self,
            "entity_id",
            _text(self.entity_id, "entity_id"),
        )

        if not isinstance(self.status, CandidateStatus):
            raise ROIFRootDetectorError(
                "status must be CandidateStatus."
            )

        numeric_fields = (
            "d_origin_score",
            "d_fast_score",
            "d_root_score",
            "node_star_score",
            "reserve_factor",
            "reserve_deficit",
            "activation_factor",
            "activation_deficit",
            "geometry_factor",
            "geometry_deficit",
            "material_factor",
            "material_deficit",
            "history_factor",
            "history_effect",
            "peak_value",
            "total_exposure",
            "outgoing_strength",
            "incoming_strength",
            "downstream_reach",
            "tensor_sensitivity",
            "plane_contribution",
            "confidence",
        )

        for name in numeric_fields:
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )

        for name in (
            "first_activity_index",
            "first_threshold_index",
        ):
            value = getattr(self, name)

            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ROIFRootDetectorError(
                        f"{name} must be a nonnegative integer."
                    )

        if (
            self.intervention is not None
            and not isinstance(
                self.intervention,
                CandidateIntervention,
            )
        ):
            raise ROIFRootDetectorError(
                "intervention must be CandidateIntervention."
            )

        object.__setattr__(
            self,
            "evidence",
            tuple(self.evidence),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )

    def score_for(self, role: ROIFRole) -> float:
        if role is ROIFRole.D_ORIGIN:
            return self.d_origin_score

        if role is ROIFRole.D_FAST:
            return self.d_fast_score

        if role is ROIFRole.D_ROOT:
            return self.d_root_score

        if role is ROIFRole.NODE_STAR:
            return self.node_star_score

        raise ROIFRootDetectorError(
            f"unsupported role {role!r}."
        )


@dataclass(frozen=True, slots=True)
class RoleRanking:
    """Sorted candidate ranking for one ROIF role."""

    role: ROIFRole
    candidates: tuple[RootCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, ROIFRole):
            raise ROIFRootDetectorError(
                "role must be ROIFRole."
            )

        candidates = tuple(self.candidates)

        if not candidates:
            raise ROIFRootDetectorError(
                "candidates cannot be empty."
            )

        scores = [
            candidate.score_for(self.role)
            for candidate in candidates
        ]

        if any(
            scores[index] < scores[index + 1]
            for index in range(len(scores) - 1)
        ):
            raise ROIFRootDetectorError(
                "candidates must be sorted by descending score."
            )

        object.__setattr__(
            self,
            "candidates",
            candidates,
        )

    @property
    def winner(self) -> RootCandidate:
        return self.candidates[0]

    def position(self, channel_id: str) -> int:
        for index, candidate in enumerate(self.candidates):
            if candidate.channel_id == channel_id:
                return index

        raise KeyError(channel_id)


@dataclass(frozen=True, slots=True)
class ROIFRootResult:
    """Final independent ROIF role assignments."""

    d_origin: str
    d_fast: str
    d_root: str
    node_star: str

    origin_ranking: RoleRanking
    fast_ranking: RoleRanking
    root_ranking: RoleRanking
    node_star_ranking: RoleRanking

    baseline_burden: float
    baseline_spectral_radius: float
    candidate_count: int

    roles_are_distinct: bool
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "d_origin",
            "d_fast",
            "d_root",
            "node_star",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name),
            )

        expected_roles = (
            ("origin_ranking", self.origin_ranking, ROIFRole.D_ORIGIN),
            ("fast_ranking", self.fast_ranking, ROIFRole.D_FAST),
            ("root_ranking", self.root_ranking, ROIFRole.D_ROOT),
            (
                "node_star_ranking",
                self.node_star_ranking,
                ROIFRole.NODE_STAR,
            ),
        )

        for name, ranking, role in expected_roles:
            if not isinstance(ranking, RoleRanking):
                raise ROIFRootDetectorError(
                    f"{name} must be RoleRanking."
                )

            if ranking.role is not role:
                raise ROIFRootDetectorError(
                    f"{name} has incorrect role."
                )

        object.__setattr__(
            self,
            "baseline_burden",
            _nonnegative(
                self.baseline_burden,
                "baseline_burden",
            ),
        )
        object.__setattr__(
            self,
            "baseline_spectral_radius",
            _nonnegative(
                self.baseline_spectral_radius,
                "baseline_spectral_radius",
            ),
        )

        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 1
        ):
            raise ROIFRootDetectorError(
                "candidate_count must be a positive integer."
            )

        if not isinstance(self.roles_are_distinct, bool):
            raise ROIFRootDetectorError(
                "roles_are_distinct must be bool."
            )

        object.__setattr__(
            self,
            "confidence",
            _unit(self.confidence, "confidence"),
        )
        object.__setattr__(
            self,
            "metadata",
            _readonly_mapping(self.metadata),
        )

    def ranking_for(self, role: ROIFRole) -> RoleRanking:
        if role is ROIFRole.D_ORIGIN:
            return self.origin_ranking

        if role is ROIFRole.D_FAST:
            return self.fast_ranking

        if role is ROIFRole.D_ROOT:
            return self.root_ranking

        if role is ROIFRole.NODE_STAR:
            return self.node_star_ranking

        raise ROIFRootDetectorError(
            f"unsupported role {role!r}."
        )


def candidate_channels(
    system: ROIFSystem,
    config: RootDetectorConfig,
) -> tuple[FunctionalChannel, ...]:
    """
    Return channels eligible for role detection.

    If ``candidate_ids`` is empty, all system channels are considered.
    ``excluded_ids`` are always removed.
    """

    if not isinstance(system, ROIFSystem):
        raise TypeError("system must be ROIFSystem.")

    channels = tuple(
        channel
        for entity in system.entities
        for channel in entity.channels
    )

    if not channels:
        raise ROIFRootDetectorError(
            "system must contain at least one FunctionalChannel."
        )

    channels_by_id = {
        channel.channel_id: channel
        for channel in channels
    }

    if config.candidate_ids:
        unknown = [
            channel_id
            for channel_id in config.candidate_ids
            if channel_id not in channels_by_id
        ]

        if unknown:
            raise ROIFRootDetectorError(
                f"unknown candidate_ids: {unknown!r}."
            )

        selected = tuple(
            channels_by_id[channel_id]
            for channel_id in config.candidate_ids
        )
    else:
        selected = channels

    excluded = set(config.excluded_ids)

    result = tuple(
        channel
        for channel in selected
        if channel.channel_id not in excluded
    )

    if not result:
        raise ROIFRootDetectorError(
            "no candidate channels remain after filtering."
        )

    return result


def validate_detector_inputs(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
) -> None:
    """Validate consistency of detector inputs."""

    if not isinstance(system, ROIFSystem):
        raise TypeError("system must be ROIFSystem.")

    if not isinstance(tensor, CapacityTensor):
        raise TypeError("tensor must be CapacityTensor.")

    if not isinstance(trajectory, CascadeTrajectory):
        raise TypeError(
            "trajectory must be CascadeTrajectory."
        )

    system_ids = tuple(system.channel_ids)

    if tensor.channel_ids != system_ids:
        raise ROIFRootDetectorError(
            "tensor channel order must match system.channel_ids."
        )

    if trajectory.channel_ids != system_ids:
        raise ROIFRootDetectorError(
            "trajectory channel order must match system.channel_ids."
        )

    if tensor.matrix.shape != (
        len(system_ids),
        len(system_ids),
    ):
        raise ROIFRootDetectorError(
            "tensor matrix shape does not match system channels."
        )


def channel_index_map(
    channel_ids: Sequence[str],
) -> Mapping[str, int]:
    """Create an immutable channel index lookup."""

    ids = tuple(
        _text(channel_id, "channel_id")
        for channel_id in channel_ids
    )

    if len(ids) != len(set(ids)):
        raise ROIFRootDetectorError(
            "channel_ids must be unique."
        )

    return MappingProxyType(
        {
            channel_id: index
            for index, channel_id in enumerate(ids)
        }
    )


def trajectory_burden(
    trajectory: CascadeTrajectory,
    *,
    use_absolute_values: bool = True,
) -> float:
    """
    Return global cascade burden over the complete trajectory.

    Burden is the sum of state exposure across all channels and time steps.
    """

    history = trajectory.state_history

    if history.size == 0:
        return 0.0

    values = (
        np.abs(history)
        if use_absolute_values
        else history
    )

    return float(np.sum(values))


def final_state_burden(
    trajectory: CascadeTrajectory,
) -> float:
    """Return absolute burden in the final cascade state."""

    return float(
        np.sum(np.abs(trajectory.final_state))
    )


def peak_global_burden(
    trajectory: CascadeTrajectory,
) -> float:
    """Return maximum global L1 burden reached at any time."""

    history = trajectory.state_history

    if history.size == 0:
        return 0.0

    return float(
        np.max(
            np.sum(
                np.abs(history),
                axis=1,
            )
        )
    )


def first_activity_index(
    trajectory: CascadeTrajectory,
    channel_id: str,
    *,
    threshold: float = 1e-9,
) -> int | None:
    """Return first time index at which channel activity exceeds threshold."""

    threshold = _nonnegative(
        threshold,
        "threshold",
    )

    series = trajectory.series(channel_id)

    indices = np.flatnonzero(
        np.abs(series) > threshold
    )

    if indices.size == 0:
        return None

    return int(indices[0])


def first_threshold_index(
    trajectory: CascadeTrajectory,
    channel_id: str,
    *,
    threshold: float,
) -> int | None:
    """Return first index where a channel reaches a specified threshold."""

    threshold = _nonnegative(
        threshold,
        "threshold",
    )

    series = trajectory.series(channel_id)

    indices = np.flatnonzero(
        np.abs(series) >= threshold
    )

    if indices.size == 0:
        return None

    return int(indices[0])


def first_event_index(
    trajectory: CascadeTrajectory,
    channel_id: str,
) -> int | None:
    """
    Return earliest event step involving a channel.

    Initial-state events may use step ``-1``. They are mapped to index zero.
    """

    earliest: int | None = None

    for event in trajectory.events:
        involved = (
            channel_id in event.source_ids
            or channel_id in event.target_ids
        )

        if not involved:
            continue

        raw_step = event.metadata.get("step")

        if raw_step is None:
            step = 0
        else:
            try:
                step = int(raw_step)
            except (TypeError, ValueError):
                step = 0

        step = max(0, step)

        if earliest is None or step < earliest:
            earliest = step

    return earliest


def channel_peak(
    trajectory: CascadeTrajectory,
    channel_id: str,
) -> float:
    """Return maximum absolute channel value."""

    series = trajectory.series(channel_id)

    if series.size == 0:
        return 0.0

    return float(
        np.max(np.abs(series))
    )


def channel_exposure(
    trajectory: CascadeTrajectory,
    channel_id: str,
) -> float:
    """Return total absolute channel exposure."""

    return float(
        np.sum(
            np.abs(
                trajectory.series(channel_id)
            )
        )
    )


def normalized_early_activity(
    first_index: int | None,
    total_steps: int,
) -> float:
    """
    Convert first activity time to a score in [0, 1].

    Earlier activity receives a larger score.
    """

    if first_index is None:
        return 0.0

    denominator = max(1, total_steps)

    return max(
        0.0,
        min(
            1.0,
            1.0 - first_index / denominator,
        ),
    )


def channel_state_lookup(
    tensor: CapacityTensor,
) -> Mapping[str, ChannelTensorState]:
    """Return tensor-facing channel states keyed by channel id."""

    lookup = {
        state.channel_id: state
        for state in tensor.channel_states
    }

    missing = [
        channel_id
        for channel_id in tensor.channel_ids
        if channel_id not in lookup
    ]

    if missing:
        raise ROIFRootDetectorError(
            "tensor is missing ChannelTensorState values for "
            f"{missing!r}."
        )

    return MappingProxyType(lookup)


def reserve_deficit(
    state: ChannelTensorState,
) -> float:
    """Convert reserve factor to deficit."""

    return max(
        0.0,
        1.0 - state.reserve_factor,
    )


def activation_deficit(
    state: ChannelTensorState,
) -> float:
    """Convert activation availability to deficit."""

    return max(
        0.0,
        1.0 - state.availability_factor,
    )


def geometry_deficit(
    state: ChannelTensorState,
) -> float:
    """Convert geometric efficiency to deficit."""

    return max(
        0.0,
        1.0 - state.geometry_factor,
    )


def material_deficit(
    state: ChannelTensorState,
) -> float:
    """Convert material capacity to deficit."""

    return max(
        0.0,
        1.0 - state.material_factor,
    )


def history_effect(
    state: ChannelTensorState,
) -> float:
    """
    Return magnitude of the channel history effect.

    Values below one are treated as retained impairment.
    Values above one represent accumulated facilitation or reinforcement.
    """

    return abs(
        1.0 - state.history_factor
    )


def outgoing_strength(
    tensor: CapacityTensor,
    channel_id: str,
) -> float:
    """Return total absolute outgoing tensor coupling."""

    column = tensor.column(channel_id)

    return float(
        np.sum(np.abs(column))
    )


def incoming_strength(
    tensor: CapacityTensor,
    channel_id: str,
) -> float:
    """Return total absolute incoming tensor coupling."""

    row = tensor.row(channel_id)

    return float(
        np.sum(np.abs(row))
    )


def reachable_nodes(
    tensor: CapacityTensor,
    source_id: str,
    *,
    activity_threshold: float = 1e-12,
) -> frozenset[str]:
    """
    Return all channels reachable from source through nonzero tensor entries.

    The tensor convention is ``matrix[target, source]``.
    """

    threshold = _nonnegative(
        activity_threshold,
        "activity_threshold",
    )

    source_index = tensor.index(source_id)

    visited: set[int] = set()
    frontier: list[int] = [source_index]

    while frontier:
        current_source = frontier.pop()

        target_indices = np.flatnonzero(
            np.abs(
                tensor.matrix[:, current_source]
            ) > threshold
        )

        for target_index in target_indices:
            target = int(target_index)

            if target == source_index:
                continue

            if target not in visited:
                visited.add(target)
                frontier.append(target)

    return frozenset(
        tensor.channel_ids[index]
        for index in sorted(visited)
    )


def downstream_reach(
    tensor: CapacityTensor,
    source_id: str,
    *,
    activity_threshold: float = 1e-12,
) -> float:
    """Return fraction of other channels reachable from source."""

    if tensor.node_count <= 1:
        return 0.0

    reachable = reachable_nodes(
        tensor,
        source_id,
        activity_threshold=activity_threshold,
    )

    return len(reachable) / float(
        tensor.node_count - 1
    )


def tensor_path_influence(
    tensor: CapacityTensor,
    source_id: str,
    *,
    maximum_steps: int | None = None,
) -> float:
    """
    Estimate recursive influence emitted by one source.

    The source basis vector is propagated repeatedly through the tensor.
    """

    source_index = tensor.index(source_id)

    steps = (
        tensor.node_count
        if maximum_steps is None
        else maximum_steps
    )

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
    ):
        raise ROIFRootDetectorError(
            "maximum_steps must be a positive integer."
        )

    state = np.zeros(
        tensor.node_count,
        dtype=float,
    )
    state[source_index] = 1.0

    total = 0.0

    for _ in range(steps):
        state = tensor.matrix @ state
        total += float(
            np.sum(np.abs(state))
        )

    return total


def tensor_sensitivity(
    tensor: CapacityTensor,
    channel_id: str,
) -> float:
    """
    Estimate how strongly global tensor behaviour depends on one source.

    The candidate's outgoing column is removed and the spectral radius is
    compared with the baseline tensor.
    """

    index = tensor.index(channel_id)

    baseline_radius = spectral_radius(
        tensor.matrix
    )

    modified = np.array(
        tensor.matrix,
        dtype=float,
        copy=True,
    )
    modified[:, index] = 0.0

    modified_radius = spectral_radius(
        modified
    )

    radius_gain = max(
        0.0,
        baseline_radius - modified_radius,
    )

    recursive_influence = tensor_path_influence(
        tensor,
        channel_id,
    )

    return radius_gain + recursive_influence


def plane_slice_strength(
    tensor: CapacityTensor,
    plane_id: str,
    channel_id: str,
) -> float:
    """Return outgoing strength of one channel through one plane."""

    plane_index = tensor.plane_index(
        plane_id
    )
    channel_index = tensor.index(
        channel_id
    )

    values = tensor.plane_tensor[
        plane_index,
        :,
        channel_index,
    ]

    return float(
        np.sum(np.abs(values))
    )


def plane_contribution(
    tensor: CapacityTensor,
    channel_id: str,
    plane_weights: Mapping[str, float],
) -> float:
    """
    Return weighted outgoing contribution across influence planes.

    Missing plane weights default to one.
    """

    total = 0.0

    for plane_id in tensor.plane_ids:
        weight = _nonnegative(
            plane_weights.get(
                plane_id,
                1.0,
            ),
            f"plane_weights[{plane_id!r}]",
        )

        strength = plane_slice_strength(
            tensor,
            plane_id,
            channel_id,
        )

        total += weight * strength

    return total


def plane_contributions_by_id(
    tensor: CapacityTensor,
    channel_id: str,
    plane_weights: Mapping[str, float],
) -> Mapping[str, float]:
    """Return immutable weighted contribution for every plane."""

    values = {
        plane_id: (
            _nonnegative(
                plane_weights.get(
                    plane_id,
                    1.0,
                ),
                f"plane_weights[{plane_id!r}]",
            )
            * plane_slice_strength(
                tensor,
                plane_id,
                channel_id,
            )
        )
        for plane_id in tensor.plane_ids
    }

    return MappingProxyType(values)


def tensor_entries_for_source(
    tensor: CapacityTensor,
    channel_id: str,
) -> tuple[TensorEntryFactors, ...]:
    """Return all active tensor entries emitted by a source."""

    return tuple(
        entry
        for entry in tensor.entries
        if (
            entry.source_id == channel_id
            and entry.active
        )
    )


def tensor_entries_for_target(
    tensor: CapacityTensor,
    channel_id: str,
) -> tuple[TensorEntryFactors, ...]:
    """Return all active tensor entries received by a target."""

    return tuple(
        entry
        for entry in tensor.entries
        if (
            entry.target_id == channel_id
            and entry.active
        )
    )


def entry_factor_means(
    entries: Sequence[TensorEntryFactors],
) -> Mapping[str, float]:
    """
    Return mean audited tensor factors for a group of connections.

    Empty groups return neutral factors.
    """

    entries = tuple(entries)

    if not entries:
        return MappingProxyType(
            {
                "geometry": 1.0,
                "material": 1.0,
                "history": 1.0,
                "availability": 1.0,
                "capacity": 1.0,
                "plane": 0.0,
                "alignment": 1.0,
                "prestress": 1.0,
            }
        )

    count = float(len(entries))

    values = {
        "geometry": sum(
            entry.geometry_factor
            for entry in entries
        ) / count,
        "material": sum(
            entry.material_factor
            for entry in entries
        ) / count,
        "history": sum(
            entry.history_factor
            for entry in entries
        ) / count,
        "availability": sum(
            entry.source_availability_factor
            for entry in entries
        ) / count,
        "capacity": sum(
            entry.source_capacity_factor
            for entry in entries
        ) / count,
        "plane": sum(
            entry.plane_factor
            for entry in entries
        ) / count,
        "alignment": sum(
            entry.vector_alignment_factor
            for entry in entries
        ) / count,
        "prestress": sum(
            entry.prestress_factor
            for entry in entries
        ) / count,
    }

    return MappingProxyType(values)


def event_kind_count(
    trajectory: CascadeTrajectory,
    channel_id: str,
    event_kind: EventKind,
) -> int:
    """Count events of one kind involving a candidate channel."""

    count = 0

    for event in trajectory.events:
        if event.metadata.get(
            "event_kind"
        ) != event_kind.value:
            continue

        if (
            channel_id in event.source_ids
            or channel_id in event.target_ids
        ):
            count += 1

    return count


def earliest_event_score(
    trajectory: CascadeTrajectory,
    channel_id: str,
) -> float:
    """Return normalized early-event score."""

    index = first_event_index(
        trajectory,
        channel_id,
    )

    return normalized_early_activity(
        index,
        max(
            1,
            trajectory.step_count,
        ),
    )


def upstream_activity_score(
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    channel_id: str,
    threshold: float,
) -> float:
    """
    Score early activity combined with downstream causal reach.

    A channel receives a high score when it activates early and can reach a
    substantial portion of the system.
    """

    activity_index = first_activity_index(
        trajectory,
        channel_id,
        threshold=threshold,
    )

    early = normalized_early_activity(
        activity_index,
        max(
            1,
            trajectory.step_count,
        ),
    )

    reach = downstream_reach(
        tensor,
        channel_id,
        activity_threshold=threshold,
    )

    return early * reach


def candidate_confidence(
    channel_id: str,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig,
) -> float:
    """
    Estimate evidence completeness for one candidate.

    Confidence rises when the candidate has:

    - an available tensor-facing state;
    - trajectory observations;
    - active tensor connections;
    - plane-resolved evidence.
    """

    components: list[float] = []

    state_ids = {
        state.channel_id
        for state in tensor.channel_states
    }
    components.append(
        1.0 if channel_id in state_ids else 0.0
    )

    try:
        series = trajectory.series(
            channel_id
        )
        components.append(
            1.0 if series.size > 0 else 0.0
        )
    except KeyError:
        components.append(0.0)

    outgoing_entries = tensor_entries_for_source(
        tensor,
        channel_id,
    )
    incoming_entries = tensor_entries_for_target(
        tensor,
        channel_id,
    )

    components.append(
        1.0
        if outgoing_entries or incoming_entries
        else 0.5
    )

    if config.include_plane_contributions:
        components.append(
            1.0
            if tensor.plane_count > 0
            else 0.0
        )

    if not components:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            sum(components) / len(components),
        ),
    )


def candidate_status(
    channel: FunctionalChannel,
    confidence: float,
    config: RootDetectorConfig,
) -> CandidateStatus:
    """Determine whether a candidate can participate in rankings."""

    if channel.channel_id in config.excluded_ids:
        return CandidateStatus.EXCLUDED

    if (
        confidence
        < config.thresholds.minimum_candidate_confidence
    ):
        return CandidateStatus.INSUFFICIENT_DATA

    state_value = getattr(
        channel.state,
        "value",
        str(channel.state),
    )

    if (
        config.exclude_failed_candidates
        and state_value == "failed"
    ):
        return CandidateStatus.EXCLUDED

    return CandidateStatus.VALID


def normalize_scores(
    values: Mapping[str, float],
    mode: ScoreNormalization,
    *,
    epsilon: float = 1e-12,
) -> Mapping[str, float]:
    """Normalize candidate scores while preserving channel identifiers."""

    if not values:
        return MappingProxyType({})

    epsilon = _positive(
        epsilon,
        "epsilon",
    )

    keys = tuple(values)
    array = np.asarray(
        [
            _finite(
                values[key],
                f"values[{key!r}]",
            )
            for key in keys
        ],
        dtype=float,
    )

    if mode is ScoreNormalization.NONE:
        normalized = array

    elif mode is ScoreNormalization.MIN_MAX:
        minimum = float(
            np.min(array)
        )
        maximum = float(
            np.max(array)
        )
        span = maximum - minimum

        if span <= epsilon:
            normalized = np.ones_like(
                array
            )
        else:
            normalized = (
                array - minimum
            ) / span

    elif mode is ScoreNormalization.L1:
        denominator = float(
            np.sum(
                np.abs(array)
            )
        )

        if denominator <= epsilon:
            normalized = np.zeros_like(
                array
            )
        else:
            normalized = array / denominator

    elif mode is ScoreNormalization.MAX_ABS:
        denominator = float(
            np.max(
                np.abs(array)
            )
        )

        if denominator <= epsilon:
            normalized = np.zeros_like(
                array
            )
        else:
            normalized = array / denominator

    else:
        raise ROIFRootDetectorError(
            "unsupported ScoreNormalization."
        )

    return MappingProxyType(
        {
            key: float(normalized[index])
            for index, key in enumerate(keys)
        }
    )


def rank_candidates(
    candidates: Sequence[RootCandidate],
    role: ROIFRole,
) -> RoleRanking:
    """Sort candidates by descending role score and stable channel id."""

    candidates = tuple(candidates)

    if not candidates:
        raise ROIFRootDetectorError(
            "candidates cannot be empty."
        )

    valid = tuple(
        candidate
        for candidate in candidates
        if candidate.status
        in {
            CandidateStatus.VALID,
            CandidateStatus.UNSAFE,
        }
    )

    ranked_source = valid or candidates

    ranked = tuple(
        sorted(
            ranked_source,
            key=lambda candidate: (
                -candidate.score_for(role),
                candidate.channel_id,
            ),
        )
    )

    return RoleRanking(
        role=role,
        candidates=ranked,
    )
def virtual_intervention_matrix(
    tensor: CapacityTensor,
    candidate_id: str,
    config: RootDetectorConfig,
) -> np.ndarray:
    """
    Return a modified tensor matrix representing a virtual intervention.

    Tensor convention:
        matrix[target, source]

    Therefore:

    - outgoing influence is stored in the candidate column;
    - incoming influence is stored in the candidate row.
    """

    candidate_index = tensor.index(
        candidate_id
    )

    matrix = np.array(
        tensor.matrix,
        dtype=float,
        copy=True,
    )

    kind = config.intervention_kind

    if kind is InterventionKind.RESTORE_CHANNEL:
        outgoing_scale = max(
            0.0,
            1.0 - config.restoration_fraction,
        )
        incoming_scale = max(
            0.0,
            1.0 - config.incoming_reduction_fraction,
        )

        matrix[:, candidate_index] *= outgoing_scale
        matrix[candidate_index, :] *= incoming_scale

    elif (
        kind
        is InterventionKind.REMOVE_OUTGOING_INFLUENCE
    ):
        matrix[:, candidate_index] = 0.0

    elif (
        kind
        is InterventionKind.REDUCE_OUTGOING_INFLUENCE
    ):
        scale = max(
            0.0,
            1.0 - config.outgoing_reduction_fraction,
        )
        matrix[:, candidate_index] *= scale

    elif (
        kind
        is InterventionKind.REDUCE_INCOMING_LOAD
    ):
        scale = max(
            0.0,
            1.0 - config.incoming_reduction_fraction,
        )
        matrix[candidate_index, :] *= scale

    elif (
        kind
        is InterventionKind.RESTORE_RESERVE
    ):
        scale = max(
            0.0,
            1.0 - config.restoration_fraction,
        )

        matrix[:, candidate_index] *= scale
        matrix[candidate_index, :] *= scale

    elif kind is InterventionKind.CUSTOM:
        raise ROIFRootDetectorError(
            "CUSTOM intervention requires a domain adapter."
        )

    else:
        raise ROIFRootDetectorError(
            f"unsupported intervention kind {kind!r}."
        )

    result = np.asarray(
        matrix,
        dtype=float,
    )
    result.setflags(write=False)
    return result


def simulate_matrix_cascade(
    matrix: np.ndarray,
    initial_state: Sequence[float],
    *,
    steps: int,
    dissipation: float = 0.0,
    retention: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    """
    Simulate a reduced-order cascade directly through a matrix.

    The recurrence is:

        x(t + 1) =
            retention * x(t)
            + (1 - dissipation) * matrix @ x(t)

    The complete history, including the initial state, is returned.
    """

    matrix = _readonly_matrix(
        matrix,
        "matrix",
    )

    if matrix.shape[0] != matrix.shape[1]:
        raise ROIFRootDetectorError(
            "matrix must be square."
        )

    state = _readonly_vector(
        initial_state,
        "initial_state",
    )

    if state.shape[0] != matrix.shape[0]:
        raise ROIFRootDetectorError(
            "initial_state size must match matrix."
        )

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
    ):
        raise ROIFRootDetectorError(
            "steps must be a positive integer."
        )

    dissipation = _unit(
        dissipation,
        "dissipation",
    )
    retention = _nonnegative(
        retention,
        "retention",
    )

    lower = (
        None
        if clip_min is None
        else _finite(
            clip_min,
            "clip_min",
        )
    )
    upper = (
        None
        if clip_max is None
        else _finite(
            clip_max,
            "clip_max",
        )
    )

    if (
        lower is not None
        and upper is not None
        and lower > upper
    ):
        raise ROIFRootDetectorError(
            "clip_min cannot exceed clip_max."
        )

    current = np.array(
        state,
        dtype=float,
        copy=True,
    )

    history = [
        np.array(
            current,
            dtype=float,
            copy=True,
        )
    ]

    for _ in range(steps):
        propagated = matrix @ current

        current = (
            retention * current
            + (1.0 - dissipation) * propagated
        )

        if lower is not None or upper is not None:
            current = np.clip(
                current,
                -math.inf if lower is None else lower,
                math.inf if upper is None else upper,
            )

        history.append(
            np.array(
                current,
                dtype=float,
                copy=True,
            )
        )

    result = np.vstack(history)
    result.setflags(write=False)
    return result


def matrix_trajectory_burden(
    history: np.ndarray,
) -> float:
    """Return total absolute burden of a simulated matrix trajectory."""

    history = _readonly_matrix(
        history,
        "history",
    )

    return float(
        np.sum(np.abs(history))
    )


def matrix_affected_channel_count(
    baseline_history: np.ndarray,
    counterfactual_history: np.ndarray,
    *,
    threshold: float = 1e-9,
) -> int:
    """
    Count channels whose trajectory materially changes after intervention.
    """

    baseline = _readonly_matrix(
        baseline_history,
        "baseline_history",
    )
    counterfactual = _readonly_matrix(
        counterfactual_history,
        "counterfactual_history",
    )

    if baseline.shape != counterfactual.shape:
        raise ROIFRootDetectorError(
            "trajectory histories must have the same shape."
        )

    threshold = _nonnegative(
        threshold,
        "threshold",
    )

    difference = np.max(
        np.abs(
            baseline - counterfactual
        ),
        axis=0,
    )

    return int(
        np.count_nonzero(
            difference > threshold
        )
    )


def collateral_effect_score(
    baseline_history: np.ndarray,
    counterfactual_history: np.ndarray,
    candidate_index: int,
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Estimate collateral change outside the candidate channel.

    A large change in non-candidate channels is interpreted as a larger
    collateral footprint. The value is normalized relative to baseline burden.
    """

    baseline = _readonly_matrix(
        baseline_history,
        "baseline_history",
    )
    counterfactual = _readonly_matrix(
        counterfactual_history,
        "counterfactual_history",
    )

    if baseline.shape != counterfactual.shape:
        raise ROIFRootDetectorError(
            "trajectory histories must have the same shape."
        )

    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or not 0 <= candidate_index < baseline.shape[1]
    ):
        raise ROIFRootDetectorError(
            "candidate_index is out of range."
        )

    epsilon = _positive(
        epsilon,
        "epsilon",
    )

    difference = np.abs(
        baseline - counterfactual
    )

    difference[:, candidate_index] = 0.0

    collateral = float(
        np.sum(difference)
    )

    baseline_external = np.array(
        baseline,
        dtype=float,
        copy=True,
    )
    baseline_external[:, candidate_index] = 0.0

    denominator = max(
        float(
            np.sum(
                np.abs(
                    baseline_external
                )
            )
        ),
        epsilon,
    )

    return max(
        0.0,
        min(
            1.0,
            collateral / denominator,
        ),
    )


def intervention_cost(
    channel_id: str,
    config: RootDetectorConfig,
) -> float:
    """Return configured intervention cost for a candidate."""

    return _unit(
        config.intervention_costs.get(
            channel_id,
            0.0,
        ),
        f"intervention_costs[{channel_id!r}]",
    )


def intervention_risk(
    channel_id: str,
    config: RootDetectorConfig,
) -> float:
    """Return configured safety risk for a candidate."""

    return _unit(
        config.intervention_risks.get(
            channel_id,
            0.0,
        ),
        f"intervention_risks[{channel_id!r}]",
    )


def intervention_uncertainty(
    channel_id: str,
    config: RootDetectorConfig,
) -> float:
    """Return configured intervention uncertainty."""

    return _unit(
        config.intervention_uncertainties.get(
            channel_id,
            0.0,
        ),
        f"intervention_uncertainties[{channel_id!r}]",
    )


def intervention_irreversibility(
    channel_id: str,
    config: RootDetectorConfig,
) -> float:
    """Return configured intervention irreversibility."""

    return _unit(
        config.intervention_irreversibility.get(
            channel_id,
            0.0,
        ),
        f"intervention_irreversibility[{channel_id!r}]",
    )


def intervention_is_safe(
    *,
    collateral_effect: float,
    safety_risk: float,
    uncertainty: float,
    irreversibility: float,
    counterfactual_history: np.ndarray,
    config: RootDetectorConfig,
) -> bool:
    """Evaluate intervention safety under the selected policy."""

    thresholds = config.thresholds

    if (
        collateral_effect
        > thresholds.maximum_acceptable_collateral
    ):
        return False

    if (
        safety_risk
        > thresholds.maximum_acceptable_safety_risk
    ):
        return False

    if (
        uncertainty
        > thresholds.maximum_acceptable_uncertainty
    ):
        return False

    if (
        irreversibility
        > thresholds.irreversible_action_threshold
    ):
        return False

    if (
        config.intervention_policy
        is InterventionPolicy.STRICT_LINEAR
    ):
        if np.any(
            np.abs(counterfactual_history)
            >= thresholds.failure_threshold
        ):
            return False

    return True


def evaluate_intervention(
    candidate_id: str,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig,
) -> CandidateIntervention:
    """
    Evaluate a reduced-order virtual intervention on one candidate channel.
    """

    candidate_index = tensor.index(
        candidate_id
    )

    initial_state = trajectory.initial_state

    steps = max(
        1,
        trajectory.step_count,
    )

    baseline_history = simulate_matrix_cascade(
        tensor.matrix,
        initial_state,
        steps=steps,
    )

    modified_matrix = virtual_intervention_matrix(
        tensor,
        candidate_id,
        config,
    )

    counterfactual_history = simulate_matrix_cascade(
        modified_matrix,
        initial_state,
        steps=steps,
    )

    baseline_burden = matrix_trajectory_burden(
        baseline_history
    )
    counterfactual_burden = matrix_trajectory_burden(
        counterfactual_history
    )

    cascade_reduction = (
        baseline_burden
        - counterfactual_burden
    )

    relative_reduction = (
        cascade_reduction
        / max(
            baseline_burden,
            config.thresholds.numerical_epsilon,
        )
    )

    baseline_radius = spectral_radius(
        tensor.matrix
    )
    counterfactual_radius = spectral_radius(
        modified_matrix
    )

    spectral_gain = (
        baseline_radius
        - counterfactual_radius
    )

    affected_count = matrix_affected_channel_count(
        baseline_history,
        counterfactual_history,
        threshold=config.thresholds.activity_threshold,
    )

    collateral = collateral_effect_score(
        baseline_history,
        counterfactual_history,
        candidate_index,
        epsilon=config.thresholds.numerical_epsilon,
    )

    cost = intervention_cost(
        candidate_id,
        config,
    )
    risk = intervention_risk(
        candidate_id,
        config,
    )
    uncertainty = intervention_uncertainty(
        candidate_id,
        config,
    )
    irreversibility = intervention_irreversibility(
        candidate_id,
        config,
    )

    safe = intervention_is_safe(
        collateral_effect=collateral,
        safety_risk=risk,
        uncertainty=uncertainty,
        irreversibility=irreversibility,
        counterfactual_history=counterfactual_history,
        config=config,
    )

    return CandidateIntervention(
        candidate_id=candidate_id,
        kind=config.intervention_kind,
        baseline_burden=baseline_burden,
        counterfactual_burden=counterfactual_burden,
        cascade_reduction=cascade_reduction,
        relative_reduction=relative_reduction,
        baseline_spectral_radius=baseline_radius,
        counterfactual_spectral_radius=counterfactual_radius,
        spectral_gain=spectral_gain,
        affected_channel_count=affected_count,
        collateral_effect=collateral,
        intervention_cost=cost,
        safety_risk=risk,
        uncertainty=uncertainty,
        irreversibility=irreversibility,
        safe=safe,
        metadata={
            "candidate_index": candidate_index,
            "simulated_steps": steps,
            "intervention_policy": (
                config.intervention_policy.value
            ),
        },
    )


def origin_score(
    *,
    first_activity: int | None,
    first_event: int | None,
    upstream_score: float,
    trajectory_steps: int,
    mode: OriginDetectionMode,
) -> float:
    """Calculate D_origin score."""

    activity_score = normalized_early_activity(
        first_activity,
        max(1, trajectory_steps),
    )
    event_score = normalized_early_activity(
        first_event,
        max(1, trajectory_steps),
    )

    if mode is OriginDetectionMode.EARLIEST_ACTIVITY:
        return activity_score

    if mode is OriginDetectionMode.EARLIEST_EVENT:
        return event_score

    if mode is OriginDetectionMode.UPSTREAM_ACTIVITY:
        return upstream_score

    if mode is OriginDetectionMode.HYBRID:
        return (
            0.40 * activity_score
            + 0.25 * event_score
            + 0.35 * upstream_score
        )

    raise ROIFRootDetectorError(
        "unsupported OriginDetectionMode."
    )


def fast_score(
    *,
    reserve_deficit_value: float,
    utilization_value: float,
    peak_deficit: float,
    first_threshold: int | None,
    trajectory_steps: int,
    mode: FastDetectionMode,
) -> float:
    """Calculate D_fast score."""

    threshold_score = normalized_early_activity(
        first_threshold,
        max(1, trajectory_steps),
    )

    if mode is FastDetectionMode.MINIMUM_RESERVE:
        return reserve_deficit_value

    if mode is FastDetectionMode.EARLIEST_THRESHOLD:
        return threshold_score

    if mode is FastDetectionMode.PEAK_DEFICIT:
        return peak_deficit


    if mode is FastDetectionMode.CAPACITY_EXCEEDANCE:
        # Physical failure criterion:
        #
        #     utilization = |demand| / capacity
        #
        # A channel at or above capacity outranks a channel
        # that remains physically below capacity.
        #
        # RootCandidate requires finite scores.
        if math.isinf(utilization_value):
            return 2.0

        if utilization_value >= 1.0:
            return 1.0 + min(
                1.0,
                utilization_value - 1.0,
            )

        return max(
            0.0,
            min(
                1.0,
                utilization_value,
            ),
        )

    if mode is FastDetectionMode.HYBRID:
        return (
            0.45 * reserve_deficit_value
            + 0.30 * threshold_score
            + 0.25 * peak_deficit
        )

    raise ROIFRootDetectorError(
        "unsupported FastDetectionMode."
    )

def mediation_balance(
    incoming_strength_value: float,
    outgoing_strength_value: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Return the directional balance of cascade mediation.

    The value approaches zero for a pure source or terminal channel and
    reaches one when incoming and outgoing structural strengths are balanced.
    """

    incoming = max(0.0, float(incoming_strength_value))
    outgoing = max(0.0, float(outgoing_strength_value))

    total = incoming + outgoing

    if total <= epsilon:
        return 0.0

    return min(
        1.0,
        4.0 * incoming * outgoing / (total * total),
    )


def mediation_throughput(
    incoming_strength_value: float,
    outgoing_strength_value: float,
) -> float:
    """
    Return the structural throughput of a candidate channel.

    Both incoming and outgoing influence must be present.
    """

    incoming = max(0.0, float(incoming_strength_value))
    outgoing = max(0.0, float(outgoing_strength_value))

    return float((incoming * outgoing) ** 0.5)


def propagation_gain(
    incoming_strength_value: float,
    outgoing_strength_value: float,
    *,
    epsilon: float = 1e-12,
    maximum_gain: float = 4.0,
) -> float:
    """
    Return the local cascade propagation gain.

    Values:

        < 1.0  attenuation
        = 1.0  approximately neutral transmission
        > 1.0  amplification

    The upper bound protects the detector from numerical instability when
    incoming strength is very small. Pure source channels remain excluded by
    zero mediation balance and zero throughput.
    """

    incoming = max(0.0, float(incoming_strength_value))
    outgoing = max(0.0, float(outgoing_strength_value))

    if outgoing <= epsilon:
        return 0.0

    gain = outgoing / max(incoming, epsilon)

    return min(
        max(0.0, gain),
        max(1.0, float(maximum_gain)),
    )


def root_score(
    *,
    incoming_strength_value: float,
    outgoing_strength_value: float,
    tensor_sensitivity_value: float,
    downstream_reach_value: float,
    exposure_value: float,
    early_activity_value: float,
    plane_contribution_value: float,
    geometry_deficit_value: float,
    material_deficit_value: float,
    history_effect_value: float,
    weights: RootDetectorWeights,
    mode: RootDetectionMode,
) -> float:
    """
    Calculate D_root as an amplifying cascade-mediation channel.

    D_root is neither:

    - the earliest cascade origin;
    - the first channel that lost function;
    - the optimal intervention target.

    It is the channel that receives upstream influence, transmits it
    downstream, and locally amplifies cascade propagation.

    Several arguments remain in the signature for backward API compatibility,
    but origin-, failure-, and persistence-related features are intentionally
    excluded from D_root scoring.
    """

    del (
        exposure_value,
        early_activity_value,
        plane_contribution_value,
        geometry_deficit_value,
        material_deficit_value,
        history_effect_value,
    )

    balance = mediation_balance(
        incoming_strength_value,
        outgoing_strength_value,
    )

    throughput = mediation_throughput(
        incoming_strength_value,
        outgoing_strength_value,
    )

    gain = propagation_gain(
        incoming_strength_value,
        outgoing_strength_value,
    )

    mediation_component = (
        balance
        * throughput
        * gain
        * max(0.0, downstream_reach_value)
    )

    sensitivity_component = (
        weights.tensor_sensitivity
        * max(0.0, tensor_sensitivity_value)
    )

    if mode is RootDetectionMode.VIRTUAL_RESTORATION:
        return mediation_component

    if mode in {
        RootDetectionMode.TRAJECTORY_SENSITIVITY,
        RootDetectionMode.HYBRID,
    }:
        return (
            mediation_component
            + sensitivity_component
        )

    raise ROIFRootDetectorError(
        "unsupported RootDetectionMode."
    )


def node_star_score(
    *,
    intervention: CandidateIntervention,
    confidence: float,
    weights: RootDetectorWeights,
    policy: InterventionPolicy,
) -> float:
    """
    Calculate Node* as targeted counterfactual intervention utility.

    Node* is independent of D_root.

    The score rewards retained reduction of cascade burden after accounting
    for collateral influence, intervention cost, uncertainty, safety risk,
    and irreversibility.
    """

    confidence_factor = max(
        0.0,
        min(
            1.0,
            float(confidence),
        ),
    )

    relative_gain = max(
        0.0,
        intervention.relative_reduction,
    )

    collateral_fraction = max(
        0.0,
        min(
            1.0,
            intervention.collateral_effect,
        ),
    )

    retained_gain = (
        weights.intervention_gain
        * relative_gain
        * (1.0 - collateral_fraction)
    )

    penalties = (
        weights.intervention_cost
        * max(
            0.0,
            intervention.intervention_cost,
        )
        + weights.uncertainty
        * max(
            0.0,
            intervention.uncertainty,
        )
        + weights.safety_risk
        * max(
            0.0,
            intervention.safety_risk,
        )
        + weights.irreversibility
        * max(
            0.0,
            intervention.irreversibility,
        )
    )

    score = (
        retained_gain
        - penalties
    ) * confidence_factor

    if (
        policy is InterventionPolicy.STRICT_LINEAR
        and not intervention.safe
    ):
        return -abs(score) - 1.0

    if (
        policy is InterventionPolicy.ADAPTIVE
        and not intervention.safe
    ):
        return score - 0.50

    return score


def build_role_evidence(
    *,
    channel_id: str,
    state: ChannelTensorState,
    first_activity: int | None,
    first_threshold: int | None,
    peak_value: float,
    exposure_value: float,
    reach_value: float,
    tensor_sensitivity_value: float,
    plane_values: Mapping[str, float],
    intervention: CandidateIntervention,
    config: RootDetectorConfig,
) -> tuple[RoleEvidence, ...]:
    """Create auditable evidence records for one candidate."""

    weights = config.weights

    evidence: list[RoleEvidence] = []

    evidence.append(
        RoleEvidence(
            evidence_id=f"{channel_id}:reserve",
            role=ROIFRole.D_FAST,
            value=reserve_deficit(state),
            weight=weights.reserve_deficit,
            weighted_value=(
                reserve_deficit(state)
                * weights.reserve_deficit
            ),
            description=(
                "Functional reserve deficit of the candidate channel."
            ),
        )
    )

    evidence.append(
        RoleEvidence(
            evidence_id=f"{channel_id}:peak",
            role=ROIFRole.D_FAST,
            value=peak_value,
            weight=1.0,
            weighted_value=peak_value,
            description=(
                "Maximum absolute trajectory value of the candidate."
            ),
            time_index=first_threshold,
        )
    )

    evidence.append(
        RoleEvidence(
            evidence_id=f"{channel_id}:cascade_reduction",
            role=ROIFRole.D_ROOT,
            value=intervention.relative_reduction,
            weight=weights.cascade_reduction,
            weighted_value=(
                intervention.relative_reduction
                * weights.cascade_reduction
            ),
            description=(
                "Relative global cascade reduction after virtual restoration."
            ),
        )
    )

    evidence.append(
        RoleEvidence(
            evidence_id=f"{channel_id}:tensor_sensitivity",
            role=ROIFRole.D_ROOT,
            value=tensor_sensitivity_value,
            weight=weights.tensor_sensitivity,
            weighted_value=(
                tensor_sensitivity_value
                * weights.tensor_sensitivity
            ),
            description=(
                "Change in recursive tensor influence after candidate removal."
            ),
        )
    )

    evidence.append(
        RoleEvidence(
            evidence_id=f"{channel_id}:reach",
            role=ROIFRole.D_ROOT,
            value=reach_value,
            weight=weights.downstream_reach,
            weighted_value=(
                reach_value
                * weights.downstream_reach
            ),
            description=(
                "Fraction of system channels reachable downstream."
            ),
        )
    )

    evidence.append(
        RoleEvidence(
            evidence_id=f"{channel_id}:exposure",
            role=ROIFRole.D_ROOT,
            value=exposure_value,
            weight=weights.trajectory_exposure,
            weighted_value=(
                exposure_value
                * weights.trajectory_exposure
            ),
            description=(
                "Total observed trajectory exposure."
            ),
            time_index=first_activity,
        )
    )

    for plane_id, value in plane_values.items():
        weight = config.plane_weights.get(
            plane_id,
            1.0,
        )

        evidence.append(
            RoleEvidence(
                evidence_id=(
                    f"{channel_id}:plane:{plane_id}"
                ),
                role=ROIFRole.D_ROOT,
                value=value,
                weight=weight,
                weighted_value=value * weight,
                description=(
                    "Outgoing contribution through an influence plane."
                ),
                plane_id=plane_id,
            )
        )

    evidence.append(
        RoleEvidence(
            evidence_id=f"{channel_id}:node_star_gain",
            role=ROIFRole.NODE_STAR,
            value=intervention.relative_reduction,
            weight=weights.intervention_gain,
            weighted_value=(
                intervention.relative_reduction
                * weights.intervention_gain
            ),
            description=(
                "Expected cascade reduction from candidate intervention."
            ),
            metadata={
                "safe": intervention.safe,
                "cost": intervention.intervention_cost,
                "collateral": intervention.collateral_effect,
                "uncertainty": intervention.uncertainty,
                "irreversibility": intervention.irreversibility,
            },
        )
    )

    return tuple(evidence)
def evaluate_candidate(
    channel: FunctionalChannel,
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig,
) -> RootCandidate:
    """
    Evaluate one channel independently for all four ROIF roles.

    The candidate receives separate scores for:

    - D_origin;
    - D_fast;
    - D_root;
    - Node*.
    """

    channel_id = channel.channel_id

    state_lookup = channel_state_lookup(
        tensor
    )

    if channel_id not in state_lookup:
        if config.strict_missing_data:
            raise ROIFRootDetectorError(
                f"tensor state is missing for {channel_id!r}."
            )

        fallback_state = ChannelTensorState(
            channel_id=channel_id,
            reserve_factor=1.0,
            availability_factor=1.0,
            geometry_factor=1.0,
            history_factor=1.0,
            material_factor=1.0,
            acceptance_factor=1.0,
            effective_output=1.0,
            direction=channel.direction,
            metadata={
                "fallback": True,
            },
        )
        state = fallback_state
    else:
        state = state_lookup[channel_id]

    confidence = candidate_confidence(
        channel_id,
        tensor,
        trajectory,
        config,
    )

    status = candidate_status(
        channel,
        confidence,
        config,
    )

    first_activity = first_activity_index(
        trajectory,
        channel_id,
        threshold=config.thresholds.activity_threshold,
    )

    first_threshold = first_threshold_index(
        trajectory,
        channel_id,
        threshold=config.thresholds.fast_threshold,
    )

    first_event = first_event_index(
        trajectory,
        channel_id,
    )

    peak_value = channel_peak(
        trajectory,
        channel_id,
    )

    exposure_value = channel_exposure(
        trajectory,
        channel_id,
    )

    outgoing_value = outgoing_strength(
        tensor,
        channel_id,
    )

    incoming_value = incoming_strength(
        tensor,
        channel_id,
    )

    reach_value = downstream_reach(
        tensor,
        channel_id,
        activity_threshold=config.thresholds.activity_threshold,
    )

    sensitivity_value = (
        tensor_sensitivity(
            tensor,
            channel_id,
        )
        if config.include_tensor_sensitivity
        else 0.0
    )

    plane_values = (
        plane_contributions_by_id(
            tensor,
            channel_id,
            config.plane_weights,
        )
        if config.include_plane_contributions
        else MappingProxyType({})
    )

    plane_value = (
        sum(plane_values.values())
        if plane_values
        else 0.0
    )

    reserve_deficit_value = reserve_deficit(
        state
    )


    # Physical utilization:
    #
    #     u = |load| / capacity
    #
    # Independent from cascade trajectory amplitude.
    physical_utilization = compute_utilization(
        channel.capacity_state.load,
        channel.capacity_state.capacity,
    )
    activation_deficit_value = (
        activation_deficit(state)
        if config.include_activation_contribution
        else 0.0
    )

    geometry_deficit_value = (
        geometry_deficit(state)
        if config.include_geometry_contribution
        else 0.0
    )

    material_deficit_value = (
        material_deficit(state)
        if config.include_material_contribution
        else 0.0
    )

    history_effect_value = (
        history_effect(state)
        if config.include_history_contribution
        else 0.0
    )

    upstream_score = upstream_activity_score(
        tensor,
        trajectory,
        channel_id,
        config.thresholds.activity_threshold,
    )

    origin_raw = origin_score(
        first_activity=first_activity,
        first_event=first_event,
        upstream_score=upstream_score,
        trajectory_steps=trajectory.step_count,
        mode=config.origin_mode,
    )

    fast_raw = fast_score(
        reserve_deficit_value=reserve_deficit_value,
        utilization_value=physical_utilization,
        peak_deficit=peak_value,
        first_threshold=first_threshold,
        trajectory_steps=trajectory.step_count,
        mode=config.fast_mode,
    )

    intervention = evaluate_intervention(
        channel_id,
        tensor,
        trajectory,
        config,
    )

    early_activity_value = normalized_early_activity(
        first_activity,
        max(
            1,
            trajectory.step_count,
        ),
    )

    exposure_denominator = max(
        trajectory_burden(trajectory),
        config.thresholds.numerical_epsilon,
    )

    normalized_exposure = (
        exposure_value
        / exposure_denominator
    )

    root_raw = root_score(
        incoming_strength_value=incoming_value,
        outgoing_strength_value=outgoing_value,
        tensor_sensitivity_value=sensitivity_value,
        downstream_reach_value=reach_value,
        exposure_value=normalized_exposure,
        early_activity_value=early_activity_value,
        plane_contribution_value=plane_value,
        geometry_deficit_value=geometry_deficit_value,
        material_deficit_value=material_deficit_value,
        history_effect_value=history_effect_value,
        weights=config.weights,
        mode=config.root_mode,
    )

    node_raw = node_star_score(
        intervention=intervention,
        confidence=confidence,
        weights=config.weights,
        policy=config.intervention_policy,
    )

    if status is CandidateStatus.INSUFFICIENT_DATA:
        origin_raw = 0.0
        fast_raw = 0.0
        root_raw = 0.0
        node_raw = 0.0

    if status is CandidateStatus.EXCLUDED:
        origin_raw = -math.inf
        fast_raw = -math.inf
        root_raw = -math.inf
        node_raw = -math.inf

    if (
        not intervention.safe
        and status is CandidateStatus.VALID
    ):
        status = CandidateStatus.UNSAFE

    evidence = build_role_evidence(
        channel_id=channel_id,
        state=state,
        first_activity=first_activity,
        first_threshold=first_threshold,
        peak_value=peak_value,
        exposure_value=normalized_exposure,
        reach_value=reach_value,
        tensor_sensitivity_value=sensitivity_value,
        plane_values=plane_values,
        intervention=intervention,
        config=config,
    )

    return RootCandidate(
        channel_id=channel_id,
        entity_id=channel.entity_id,
        status=status,
        d_origin_score=origin_raw,
        d_fast_score=fast_raw,
        d_root_score=root_raw,
        node_star_score=node_raw,
        reserve_factor=state.reserve_factor,
        reserve_deficit=reserve_deficit_value,
        activation_factor=state.availability_factor,
        activation_deficit=activation_deficit_value,
        geometry_factor=state.geometry_factor,
        geometry_deficit=geometry_deficit_value,
        material_factor=state.material_factor,
        material_deficit=material_deficit_value,
        history_factor=state.history_factor,
        history_effect=history_effect_value,
        first_activity_index=first_activity,
        first_threshold_index=first_threshold,
        peak_value=peak_value,
        total_exposure=exposure_value,
        outgoing_strength=outgoing_value,
        incoming_strength=incoming_value,
        downstream_reach=reach_value,
        tensor_sensitivity=sensitivity_value,
        plane_contribution=plane_value,
        confidence=confidence,
        intervention=intervention,
        evidence=evidence,
        metadata={
            "first_event_index": first_event,
            "physical_utilization": physical_utilization,
            "capacity_exceeded": physical_utilization >= 1.0,
            "upstream_activity_score": upstream_score,
            "normalized_exposure": normalized_exposure,
            "event_counts": {
                kind.value: event_kind_count(
                    trajectory,
                    channel_id,
                    kind,
                )
                for kind in EventKind
            },
        },
    )


def replace_candidate_scores(
    candidate: RootCandidate,
    *,
    d_origin_score: float | None = None,
    d_fast_score: float | None = None,
    d_root_score: float | None = None,
    node_star_score: float | None = None,
) -> RootCandidate:
    """Return a candidate with normalized role scores."""

    return RootCandidate(
        channel_id=candidate.channel_id,
        entity_id=candidate.entity_id,
        status=candidate.status,
        d_origin_score=(
            candidate.d_origin_score
            if d_origin_score is None
            else d_origin_score
        ),
        d_fast_score=(
            candidate.d_fast_score
            if d_fast_score is None
            else d_fast_score
        ),
        d_root_score=(
            candidate.d_root_score
            if d_root_score is None
            else d_root_score
        ),
        node_star_score=(
            candidate.node_star_score
            if node_star_score is None
            else node_star_score
        ),
        reserve_factor=candidate.reserve_factor,
        reserve_deficit=candidate.reserve_deficit,
        activation_factor=candidate.activation_factor,
        activation_deficit=candidate.activation_deficit,
        geometry_factor=candidate.geometry_factor,
        geometry_deficit=candidate.geometry_deficit,
        material_factor=candidate.material_factor,
        material_deficit=candidate.material_deficit,
        history_factor=candidate.history_factor,
        history_effect=candidate.history_effect,
        first_activity_index=candidate.first_activity_index,
        first_threshold_index=candidate.first_threshold_index,
        peak_value=candidate.peak_value,
        total_exposure=candidate.total_exposure,
        outgoing_strength=candidate.outgoing_strength,
        incoming_strength=candidate.incoming_strength,
        downstream_reach=candidate.downstream_reach,
        tensor_sensitivity=candidate.tensor_sensitivity,
        plane_contribution=candidate.plane_contribution,
        confidence=candidate.confidence,
        intervention=candidate.intervention,
        evidence=candidate.evidence,
        metadata=candidate.metadata,
    )


def normalize_candidate_role_scores(
    candidates: Sequence[RootCandidate],
    config: RootDetectorConfig,
) -> tuple[RootCandidate, ...]:
    """Normalize each role independently across all valid candidates."""

    candidates = tuple(candidates)

    origin_values = {
        candidate.channel_id: candidate.d_origin_score
        for candidate in candidates
        if math.isfinite(candidate.d_origin_score)
    }
    fast_values = {
        candidate.channel_id: candidate.d_fast_score
        for candidate in candidates
        if math.isfinite(candidate.d_fast_score)
    }
    root_values = {
        candidate.channel_id: candidate.d_root_score
        for candidate in candidates
        if math.isfinite(candidate.d_root_score)
    }
    node_values = {
        candidate.channel_id: candidate.node_star_score
        for candidate in candidates
        if math.isfinite(candidate.node_star_score)
    }

    normalized_origin = normalize_scores(
        origin_values,
        config.normalization,
        epsilon=config.thresholds.numerical_epsilon,
    )
    normalized_fast = normalize_scores(
        fast_values,
        config.normalization,
        epsilon=config.thresholds.numerical_epsilon,
    )
    normalized_root = normalize_scores(
        root_values,
        config.normalization,
        epsilon=config.thresholds.numerical_epsilon,
    )
    normalized_node = normalize_scores(
        node_values,
        config.normalization,
        epsilon=config.thresholds.numerical_epsilon,
    )

    result: list[RootCandidate] = []

    for candidate in candidates:
        if candidate.status is CandidateStatus.EXCLUDED:
            result.append(candidate)
            continue

        result.append(
            replace_candidate_scores(
                candidate,
                d_origin_score=normalized_origin.get(
                    candidate.channel_id,
                    0.0,
                ),
                d_fast_score=normalized_fast.get(
                    candidate.channel_id,
                    0.0,
                ),
                d_root_score=normalized_root.get(
                    candidate.channel_id,
                    0.0,
                ),
                node_star_score=normalized_node.get(
                    candidate.channel_id,
                    0.0,
                ),
            )
        )

    return tuple(result)


def select_role_winner(
    ranking: RoleRanking,
    *,
    excluded_ids: frozenset[str] = frozenset(),
) -> RootCandidate:
    """Select the first ranked candidate not explicitly excluded."""

    for candidate in ranking.candidates:
        if candidate.channel_id not in excluded_ids:
            return candidate

    raise ROIFRootDetectorError(
        f"no eligible candidate remains for {ranking.role.value}."
    )


def enforce_distinct_roles(
    origin_ranking: RoleRanking,
    fast_ranking: RoleRanking,
    root_ranking: RoleRanking,
    node_ranking: RoleRanking,
) -> tuple[
    RootCandidate,
    RootCandidate,
    RootCandidate,
    RootCandidate,
]:
    """
    Select distinct winners where possible.

    Role order is:

    1. D_origin;
    2. D_fast;
    3. D_root;
    4. Node*.

    If the number of eligible channels is insufficient, duplication is allowed
    rather than failing the analysis.
    """

    selected: list[RootCandidate] = []

    rankings = (
        origin_ranking,
        fast_ranking,
        root_ranking,
        node_ranking,
    )

    for ranking in rankings:
        used = frozenset(
            candidate.channel_id
            for candidate in selected
        )

        try:
            winner = select_role_winner(
                ranking,
                excluded_ids=used,
            )
        except ROIFRootDetectorError:
            winner = ranking.winner

        selected.append(winner)

    return (
        selected[0],
        selected[1],
        selected[2],
        selected[3],
    )


def aggregate_result_confidence(
    winners: Sequence[RootCandidate],
) -> float:
    """Return mean confidence across selected role winners."""

    winners = tuple(winners)

    if not winners:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            sum(
                candidate.confidence
                for candidate in winners
            )
            / len(winners),
        ),
    )


def detect_roif_roles(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig | None = None,
) -> ROIFRootResult:
    """
    Detect D_origin, D_fast, D_root, and Node*.

    Hidden truth labels are neither required nor used.
    """

    config = config or RootDetectorConfig()

    validate_detector_inputs(
        system,
        tensor,
        trajectory,
    )

    channels = candidate_channels(
        system,
        config,
    )

    raw_candidates = tuple(
        evaluate_candidate(
            channel,
            system,
            tensor,
            trajectory,
            config,
        )
        for channel in channels
    )

    candidates = normalize_candidate_role_scores(
        raw_candidates,
        config,
    )

    origin_ranking = rank_candidates(
        candidates,
        ROIFRole.D_ORIGIN,
    )
    fast_ranking = rank_candidates(
        candidates,
        ROIFRole.D_FAST,
    )
    root_ranking = rank_candidates(
        candidates,
        ROIFRole.D_ROOT,
    )
    node_ranking = rank_candidates(
        candidates,
        ROIFRole.NODE_STAR,
    )

    if config.allow_same_role_node:
        origin_winner = origin_ranking.winner
        fast_winner = fast_ranking.winner
        root_winner = root_ranking.winner
        node_winner = node_ranking.winner
    else:
        (
            origin_winner,
            fast_winner,
            root_winner,
            node_winner,
        ) = enforce_distinct_roles(
            origin_ranking,
            fast_ranking,
            root_ranking,
            node_ranking,
        )

    winner_ids = (
        origin_winner.channel_id,
        fast_winner.channel_id,
        root_winner.channel_id,
        node_winner.channel_id,
    )

    confidence = aggregate_result_confidence(
        (
            origin_winner,
            fast_winner,
            root_winner,
            node_winner,
        )
    )

    baseline_burden = trajectory_burden(
        trajectory
    )

    baseline_radius = spectral_radius(
        tensor.matrix
    )

    return ROIFRootResult(
        d_origin=origin_winner.channel_id,
        d_fast=fast_winner.channel_id,
        d_root=root_winner.channel_id,
        node_star=node_winner.channel_id,
        origin_ranking=origin_ranking,
        fast_ranking=fast_ranking,
        root_ranking=root_ranking,
        node_star_ranking=node_ranking,
        baseline_burden=baseline_burden,
        baseline_spectral_radius=baseline_radius,
        candidate_count=len(candidates),
        roles_are_distinct=(
            len(set(winner_ids)) == len(winner_ids)
        ),
        confidence=confidence,
        metadata={
            "system_id": system.system_id,
            "root_mode": config.root_mode.value,
            "fast_mode": config.fast_mode.value,
            "origin_mode": config.origin_mode.value,
            "intervention_policy": (
                config.intervention_policy.value
            ),
            "intervention_kind": (
                config.intervention_kind.value
            ),
            "normalization": config.normalization.value,
            "allow_same_role_node": (
                config.allow_same_role_node
            ),
            "trajectory_steps": trajectory.step_count,
            "trajectory_failed": trajectory.failed,
            "trajectory_converged": trajectory.converged,
        },
    )


def detect_d_origin(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig | None = None,
) -> RootCandidate:
    """Return the highest-ranked D_origin candidate."""

    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        config,
    )

    return result.origin_ranking.winner


def detect_d_fast(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig | None = None,
) -> RootCandidate:
    """Return the highest-ranked D_fast candidate."""

    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        config,
    )

    return result.fast_ranking.winner


def detect_d_root(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig | None = None,
) -> RootCandidate:
    """Return the highest-ranked D_root candidate."""

    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        config,
    )

    return result.root_ranking.winner


def detect_node_star(
    system: ROIFSystem,
    tensor: CapacityTensor,
    trajectory: CascadeTrajectory,
    config: RootDetectorConfig | None = None,
) -> RootCandidate:
    """Return the highest-ranked Node* intervention candidate."""

    result = detect_roif_roles(
        system,
        tensor,
        trajectory,
        config,
    )

    return result.node_star_ranking.winner


def candidate_by_id(
    result: ROIFRootResult,
    channel_id: str,
) -> RootCandidate:
    """Return a candidate from the root ranking by channel identifier."""

    channel_id = _text(
        channel_id,
        "channel_id",
    )

    for candidate in result.root_ranking.candidates:
        if candidate.channel_id == channel_id:
            return candidate

    raise KeyError(channel_id)


def role_assignments(
    result: ROIFRootResult,
) -> Mapping[ROIFRole, str]:
    """Return immutable mapping of ROIF roles to channel identifiers."""

    return MappingProxyType(
        {
            ROIFRole.D_ORIGIN: result.d_origin,
            ROIFRole.D_FAST: result.d_fast,
            ROIFRole.D_ROOT: result.d_root,
            ROIFRole.NODE_STAR: result.node_star,
        }
    )


def root_result_summary(
    result: ROIFRootResult,
) -> Mapping[str, Any]:
    """Return a compact immutable result summary."""

    root_candidate = result.root_ranking.winner
    node_candidate = result.node_star_ranking.winner

    return MappingProxyType(
        {
            "d_origin": result.d_origin,
            "d_fast": result.d_fast,
            "d_root": result.d_root,
            "node_star": result.node_star,
            "roles_are_distinct": result.roles_are_distinct,
            "confidence": result.confidence,
            "candidate_count": result.candidate_count,
            "baseline_burden": result.baseline_burden,
            "baseline_spectral_radius": (
                result.baseline_spectral_radius
            ),
            "root_score": root_candidate.d_root_score,
            "root_relative_reduction": (
                root_candidate.intervention.relative_reduction
                if root_candidate.intervention is not None
                else 0.0
            ),
            "node_star_score": (
                node_candidate.node_star_score
            ),
            "node_star_safe": (
                node_candidate.intervention.safe
                if node_candidate.intervention is not None
                else False
            ),
        }
    )


__all__ = [
    "CandidateIntervention",
    "CandidateStatus",
    "FastDetectionMode",
    "InterventionKind",
    "InterventionPolicy",
    "OriginDetectionMode",
    "ROIFRole",
    "ROIFRootDetectorError",
    "ROIFRootResult",
    "RoleEvidence",
    "RoleRanking",
    "RootCandidate",
    "RootDetectionMode",
    "RootDetectorConfig",
    "RootDetectorThresholds",
    "RootDetectorWeights",
    "ScoreNormalization",
    "aggregate_result_confidence",
    "candidate_by_id",
    "candidate_channels",
    "candidate_confidence",
    "candidate_status",
    "channel_exposure",
    "channel_index_map",
    "channel_peak",
    "channel_state_lookup",
    "collateral_effect_score",
    "detect_d_fast",
    "detect_d_origin",
    "detect_d_root",
    "detect_node_star",
    "detect_roif_roles",
    "downstream_reach",
    "earliest_event_score",
    "entry_factor_means",
    "evaluate_candidate",
    "evaluate_intervention",
    "event_kind_count",
    "final_state_burden",
    "first_activity_index",
    "first_event_index",
    "first_threshold_index",
    "intervention_cost",
    "intervention_irreversibility",
    "intervention_is_safe",
    "intervention_risk",
    "intervention_uncertainty",
    "matrix_affected_channel_count",
    "matrix_trajectory_burden",
    "normalize_candidate_role_scores",
    "normalize_scores",
    "normalized_early_activity",
    "outgoing_strength",
    "incoming_strength",
    "peak_global_burden",
    "plane_contribution",
    "plane_contributions_by_id",
    "plane_slice_strength",
    "rank_candidates",
    "reachable_nodes",
    "reserve_deficit",
    "activation_deficit",
    "geometry_deficit",
    "history_effect",
    "material_deficit",
    "role_assignments",
    "root_result_summary",
    "select_role_winner",
    "simulate_matrix_cascade",
    "system_with_simulated_state",
    "tensor_entries_for_source",
    "tensor_entries_for_target",
    "tensor_path_influence",
    "tensor_sensitivity",
    "trajectory_burden",
    "upstream_activity_score",
    "validate_detector_inputs",
    "virtual_intervention_matrix",
]








