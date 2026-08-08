"""
ROIF Vector Probe Analysis

Purpose
-------
This module extracts physically interpretable transition evidence from
controlled Probe responses in a pre-stressed network.

The core question is not merely:

    "Which node changed the most?"

but:

    "Which candidate relation shows a directed, temporally plausible,
     capacity-relevant response consistent with transmission along the
     candidate edge?"

For a candidate relation i -> j, the module can evaluate:

- candidate edge direction e_ij;
- baseline -> Probe tension/load-vector change ΔT_j;
- baseline -> Probe displacement-vector change Δx_j;
- directional alignment of those changes with e_ij;
- signed projection onto e_ij;
- perpendicular residual;
- response latency;
- change in utilization Δu_j;
- whether capacity is crossed during the Probe;
- multidimensional evidence without collapsing it prematurely into one score.

Canonical quantities
--------------------

Candidate edge unit direction:

    e_ij = (x_j - x_i) / ||x_j - x_i||

Vector response:

    Δv = v_probe - v_baseline

Directional alignment:

    a = (Δv · e_ij) / ||Δv||

Signed axial projection:

    p = Δv · e_ij

Perpendicular residual magnitude:

    r_perp = ||Δv - p e_ij||

Utilization change:

    Δu = u_probe - u_baseline

where utilization is supplied explicitly or computed from load/capacity
using roif.utilization.

Important boundary
------------------
This module does NOT:
- decide D_fast / D_root / Node*;
- mutate the graph;
- authorize GraphUpdateProposal;
- declare an edge causally confirmed by itself;
- infer causality from response amplitude alone.

It produces typed physical evidence for higher-level APE / Active Cascade
policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from roif.utilization import (
    compute_utilization,
)


FloatVector = NDArray[np.float64]

_EPSILON = 1e-12


# =============================================================================
# Errors / enums
# =============================================================================


class VectorProbeError(ValueError):
    """Raised when vector Probe evidence cannot be evaluated safely."""


class ResponsePolarity(str, Enum):
    """
    Direction of response relative to the candidate edge.

    FORWARD
        Positive projection along source -> target.

    REVERSE
        Negative projection.

    ORTHOGONAL
        Projection is negligible relative to response magnitude.

    NONE
        Response magnitude is negligible.
    """

    FORWARD = "forward"
    REVERSE = "reverse"
    ORTHOGONAL = "orthogonal"
    NONE = "none"


class VectorEvidenceDecision(str, Enum):
    """
    Conservative interpretation of multidimensional evidence.

    SUPPORTS
        Physical evidence is consistent with the candidate relation.

    CONTRADICTS
        Physical evidence materially contradicts the candidate relation.

    INSUFFICIENT
        Evidence exists but is not strong enough to decide.

    NO_RESPONSE
        No meaningful vector/capacity response was observed.
    """

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"
    NO_RESPONSE = "no_response"


# =============================================================================
# Validation helpers
# =============================================================================


def _vector(
    value: Sequence[float] | FloatVector,
    *,
    name: str,
    allow_zero: bool = True,
) -> FloatVector:
    array = np.asarray(
        value,
        dtype=np.float64,
    ).copy()

    if array.ndim != 1:
        raise VectorProbeError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise VectorProbeError(
            f"{name} must not be empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise VectorProbeError(
            f"{name} must contain only finite values."
        )

    if (
        not allow_zero
        and float(np.linalg.norm(array)) <= _EPSILON
    ):
        raise VectorProbeError(
            f"{name} must have nonzero magnitude."
        )

    array.setflags(
        write=False
    )

    return array


def _same_dimension(
    *vectors: FloatVector,
) -> None:
    dimensions = {
        vector.size
        for vector in vectors
    }

    if len(dimensions) > 1:
        raise VectorProbeError(
            "all vectors must have the same dimension."
        )


def _finite_nonnegative(
    value: float,
    *,
    name: str,
) -> float:
    result = float(value)

    if (
        not math.isfinite(result)
        or result < 0.0
    ):
        raise VectorProbeError(
            f"{name} must be finite and non-negative."
        )

    return result


def _unit_interval(
    value: float,
    *,
    name: str,
) -> float:
    result = float(value)

    if (
        not math.isfinite(result)
        or not 0.0 <= result <= 1.0
    ):
        raise VectorProbeError(
            f"{name} must be in [0, 1]."
        )

    return result


# =============================================================================
# Geometry
# =============================================================================


@dataclass(frozen=True, slots=True)
class CandidateEdgeGeometry:
    """
    Geometry of one candidate directed relation source -> target.
    """

    source_id: str
    target_id: str
    source_position: FloatVector
    target_position: FloatVector
    direction: FloatVector = field(init=False)
    length: float = field(init=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = str(
            self.source_id
        ).strip()
        target_id = str(
            self.target_id
        ).strip()

        if not source_id:
            raise VectorProbeError(
                "source_id must not be empty."
            )

        if not target_id:
            raise VectorProbeError(
                "target_id must not be empty."
            )

        if source_id == target_id:
            raise VectorProbeError(
                "candidate edge must connect distinct nodes."
            )

        source = _vector(
            self.source_position,
            name="source_position",
        )
        target = _vector(
            self.target_position,
            name="target_position",
        )

        _same_dimension(
            source,
            target,
        )

        delta = target - source
        length = float(
            np.linalg.norm(delta)
        )

        if length <= _EPSILON:
            raise VectorProbeError(
                "candidate edge length must be nonzero."
            )

        direction = np.asarray(
            delta / length,
            dtype=np.float64,
        )
        direction.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "source_id",
            source_id,
        )
        object.__setattr__(
            self,
            "target_id",
            target_id,
        )
        object.__setattr__(
            self,
            "source_position",
            source,
        )
        object.__setattr__(
            self,
            "target_position",
            target,
        )
        object.__setattr__(
            self,
            "direction",
            direction,
        )
        object.__setattr__(
            self,
            "length",
            length,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


def edge_direction(
    source_position: Sequence[float] | FloatVector,
    target_position: Sequence[float] | FloatVector,
) -> FloatVector:
    """
    Return unit direction source -> target.
    """

    source = _vector(
        source_position,
        name="source_position",
    )
    target = _vector(
        target_position,
        name="target_position",
    )

    _same_dimension(
        source,
        target,
    )

    delta = target - source
    norm = float(
        np.linalg.norm(delta)
    )

    if norm <= _EPSILON:
        raise VectorProbeError(
            "source and target positions must be distinct."
        )

    result = np.asarray(
        delta / norm,
        dtype=np.float64,
    )
    result.setflags(
        write=False
    )

    return result


# =============================================================================
# Vector response
# =============================================================================


@dataclass(frozen=True, slots=True)
class VectorResponse:
    """
    Baseline -> Probe change of one vector quantity.
    """

    baseline: FloatVector
    probe: FloatVector
    delta: FloatVector = field(init=False)
    magnitude: float = field(init=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        baseline = _vector(
            self.baseline,
            name="baseline",
        )
        probe = _vector(
            self.probe,
            name="probe",
        )

        _same_dimension(
            baseline,
            probe,
        )

        delta = np.asarray(
            probe - baseline,
            dtype=np.float64,
        )
        magnitude = float(
            np.linalg.norm(delta)
        )

        delta.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "baseline",
            baseline,
        )
        object.__setattr__(
            self,
            "probe",
            probe,
        )
        object.__setattr__(
            self,
            "delta",
            delta,
        )
        object.__setattr__(
            self,
            "magnitude",
            magnitude,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


@dataclass(frozen=True, slots=True)
class DirectionalResponseEvidence:
    """
    Directional decomposition of one vector response relative to an edge.
    """

    alignment: float
    signed_projection: float
    perpendicular_magnitude: float
    response_magnitude: float
    polarity: ResponsePolarity

    def __post_init__(self) -> None:
        alignment = float(
            self.alignment
        )

        if (
            not math.isfinite(alignment)
            or not -1.0 <= alignment <= 1.0
        ):
            raise VectorProbeError(
                "alignment must be in [-1, 1]."
            )

        for name in (
            "signed_projection",
            "perpendicular_magnitude",
            "response_magnitude",
        ):
            value = float(
                getattr(self, name)
            )

            if not math.isfinite(value):
                raise VectorProbeError(
                    f"{name} must be finite."
                )

            if (
                name != "signed_projection"
                and value < 0.0
            ):
                raise VectorProbeError(
                    f"{name} must be non-negative."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        object.__setattr__(
            self,
            "alignment",
            alignment,
        )
        object.__setattr__(
            self,
            "polarity",
            ResponsePolarity(
                self.polarity
            ),
        )


def analyze_directional_response(
    response: VectorResponse,
    direction: Sequence[float] | FloatVector,
    *,
    zero_tolerance: float = _EPSILON,
    orthogonal_tolerance: float = 0.10,
) -> DirectionalResponseEvidence:
    """
    Decompose Δvector relative to candidate edge direction.
    """

    axis = _vector(
        direction,
        name="direction",
        allow_zero=False,
    )

    _same_dimension(
        response.delta,
        axis,
    )

    zero_tol = _finite_nonnegative(
        zero_tolerance,
        name="zero_tolerance",
    )
    ortho_tol = _unit_interval(
        orthogonal_tolerance,
        name="orthogonal_tolerance",
    )

    axis_norm = float(
        np.linalg.norm(axis)
    )
    axis_unit = axis / axis_norm

    magnitude = response.magnitude

    if magnitude <= zero_tol:
        return DirectionalResponseEvidence(
            alignment=0.0,
            signed_projection=0.0,
            perpendicular_magnitude=0.0,
            response_magnitude=magnitude,
            polarity=ResponsePolarity.NONE,
        )

    projection = float(
        np.dot(
            response.delta,
            axis_unit,
        )
    )

    alignment = projection / magnitude
    alignment = max(
        -1.0,
        min(
            1.0,
            alignment,
        ),
    )

    perpendicular = (
        response.delta
        - projection * axis_unit
    )

    perpendicular_magnitude = float(
        np.linalg.norm(
            perpendicular
        )
    )

    if abs(alignment) <= ortho_tol:
        polarity = (
            ResponsePolarity.ORTHOGONAL
        )
    elif alignment > 0.0:
        polarity = ResponsePolarity.FORWARD
    else:
        polarity = ResponsePolarity.REVERSE

    return DirectionalResponseEvidence(
        alignment=alignment,
        signed_projection=projection,
        perpendicular_magnitude=perpendicular_magnitude,
        response_magnitude=magnitude,
        polarity=polarity,
    )


# =============================================================================
# Temporal response
# =============================================================================


@dataclass(frozen=True, slots=True)
class ResponseLatency:
    """
    Temporal evidence for one observed response.
    """

    onset_index: int | None
    onset_time: float | None
    peak_index: int | None
    peak_time: float | None
    peak_magnitude: float

    def __post_init__(self) -> None:
        for name in (
            "onset_index",
            "peak_index",
        ):
            value = getattr(
                self,
                name,
            )

            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(
                        value,
                        int,
                    )
                    or value < 0
                ):
                    raise VectorProbeError(
                        f"{name} must be a nonnegative integer or None."
                    )

        for name in (
            "onset_time",
            "peak_time",
        ):
            value = getattr(
                self,
                name,
            )

            if value is not None:
                value = float(value)
                if (
                    not math.isfinite(value)
                    or value < 0.0
                ):
                    raise VectorProbeError(
                        f"{name} must be finite and non-negative or None."
                    )
                object.__setattr__(
                    self,
                    name,
                    value,
                )

        peak_magnitude = _finite_nonnegative(
            self.peak_magnitude,
            name="peak_magnitude",
        )

        object.__setattr__(
            self,
            "peak_magnitude",
            peak_magnitude,
        )


def response_latency(
    response_series: Sequence[
        Sequence[float]
    ]
    | NDArray[np.float64],
    *,
    baseline_vector: Sequence[float] | FloatVector,
    threshold: float,
    dt: float = 1.0,
) -> ResponseLatency:
    """
    Find first meaningful vector deviation from baseline and peak response.

    response_series:
        shape (time, dimensions)

    threshold:
        absolute Euclidean magnitude threshold for |v(t)-baseline|.
    """

    series = np.asarray(
        response_series,
        dtype=np.float64,
    )

    if series.ndim != 2:
        raise VectorProbeError(
            "response_series must have shape (time, dimensions)."
        )

    if not np.all(
        np.isfinite(series)
    ):
        raise VectorProbeError(
            "response_series must contain only finite values."
        )

    baseline = _vector(
        baseline_vector,
        name="baseline_vector",
    )

    if series.shape[1] != baseline.size:
        raise VectorProbeError(
            "response_series dimension does not match baseline_vector."
        )

    threshold_value = _finite_nonnegative(
        threshold,
        name="threshold",
    )

    dt_value = float(dt)

    if (
        not math.isfinite(dt_value)
        or dt_value <= 0.0
    ):
        raise VectorProbeError(
            "dt must be finite and positive."
        )

    if series.shape[0] == 0:
        return ResponseLatency(
            onset_index=None,
            onset_time=None,
            peak_index=None,
            peak_time=None,
            peak_magnitude=0.0,
        )

    deltas = (
        series
        - baseline[np.newaxis, :]
    )

    magnitudes = np.linalg.norm(
        deltas,
        axis=1,
    )

    peak_index = int(
        np.argmax(magnitudes)
    )
    peak_magnitude = float(
        magnitudes[
            peak_index
        ]
    )

    indices = np.flatnonzero(
        magnitudes >= threshold_value
    )

    if indices.size == 0:
        onset_index = None
        onset_time = None
    else:
        onset_index = int(
            indices[0]
        )
        onset_time = (
            onset_index
            * dt_value
        )

    return ResponseLatency(
        onset_index=onset_index,
        onset_time=onset_time,
        peak_index=peak_index,
        peak_time=(
            peak_index
            * dt_value
        ),
        peak_magnitude=peak_magnitude,
    )


# =============================================================================
# Utilization response
# =============================================================================


@dataclass(frozen=True, slots=True)
class UtilizationResponse:
    """
    Baseline -> Probe utilization evidence.
    """

    baseline_utilization: float
    probe_utilization: float
    delta_utilization: float
    baseline_capacity_exceeded: bool
    probe_capacity_exceeded: bool

    def __post_init__(self) -> None:
        for name in (
            "baseline_utilization",
            "probe_utilization",
            "delta_utilization",
        ):
            value = float(
                getattr(self, name)
            )

            if math.isnan(value):
                raise VectorProbeError(
                    f"{name} must not be NaN."
                )

            object.__setattr__(
                self,
                name,
                value,
            )


def analyze_utilization_response(
    *,
    baseline_demand: float,
    baseline_capacity: float,
    probe_demand: float,
    probe_capacity: float,
    threshold: float = 1.0,
) -> UtilizationResponse:
    """
    Compute baseline and Probe utilization with roif.utilization.
    """

    threshold_value = float(
        threshold
    )

    if (
        not math.isfinite(threshold_value)
        or threshold_value <= 0.0
    ):
        raise VectorProbeError(
            "threshold must be finite and positive."
        )

    baseline = compute_utilization(
        baseline_demand,
        baseline_capacity,
    )
    probe = compute_utilization(
        probe_demand,
        probe_capacity,
    )

    return UtilizationResponse(
        baseline_utilization=baseline,
        probe_utilization=probe,
        delta_utilization=(
            probe - baseline
        ),
        baseline_capacity_exceeded=(
            baseline >= threshold_value
        ),
        probe_capacity_exceeded=(
            probe >= threshold_value
        ),
    )


# =============================================================================
# Combined candidate evidence
# =============================================================================


@dataclass(frozen=True, slots=True)
class VectorProbeEvidence:
    """
    Multidimensional physical evidence for one candidate relation.
    """

    relation_id: str
    source_id: str
    target_id: str
    edge_direction: FloatVector
    tension: DirectionalResponseEvidence | None = None
    displacement: DirectionalResponseEvidence | None = None
    latency: ResponseLatency | None = None
    utilization: UtilizationResponse | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "source_id",
            "target_id",
        ):
            value = str(
                getattr(
                    self,
                    name,
                )
            ).strip()

            if not value:
                raise VectorProbeError(
                    f"{name} must not be empty."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        direction = _vector(
            self.edge_direction,
            name="edge_direction",
            allow_zero=False,
        )

        norm = float(
            np.linalg.norm(direction)
        )

        normalized = np.asarray(
            direction / norm,
            dtype=np.float64,
        )
        normalized.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "edge_direction",
            normalized,
        )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


@dataclass(frozen=True, slots=True)
class VectorEvidencePolicy:
    """
    Conservative policy for interpreting vector Probe evidence.

    min_forward_alignment
        Minimum positive alignment that supports a directed edge.

    contradiction_alignment
        Alignment at or below this negative threshold contradicts edge
        direction.

    require_tension_or_displacement
        Prevents utilization-only evidence from confirming a directed edge.

    max_support_latency
        Optional maximum onset time for direct-response support.

    min_delta_utilization
        Optional minimum Δu considered materially supportive.

    require_capacity_relevance
        If True, Probe must change utilization materially or cross capacity
        before evidence can SUPPORT a relation.
    """

    min_forward_alignment: float = 0.70
    contradiction_alignment: float = -0.50
    require_tension_or_displacement: bool = True
    max_support_latency: float | None = None
    min_delta_utilization: float = 0.0
    require_capacity_relevance: bool = False

    def __post_init__(self) -> None:
        min_alignment = float(
            self.min_forward_alignment
        )
        contradiction = float(
            self.contradiction_alignment
        )

        if (
            not math.isfinite(min_alignment)
            or not 0.0 <= min_alignment <= 1.0
        ):
            raise VectorProbeError(
                "min_forward_alignment must be in [0, 1]."
            )

        if (
            not math.isfinite(contradiction)
            or not -1.0 <= contradiction <= 0.0
        ):
            raise VectorProbeError(
                "contradiction_alignment must be in [-1, 0]."
            )

        object.__setattr__(
            self,
            "min_forward_alignment",
            min_alignment,
        )
        object.__setattr__(
            self,
            "contradiction_alignment",
            contradiction,
        )

        if self.max_support_latency is not None:
            latency = _finite_nonnegative(
                self.max_support_latency,
                name="max_support_latency",
            )
            object.__setattr__(
                self,
                "max_support_latency",
                latency,
            )

        delta = float(
            self.min_delta_utilization
        )

        if (
            not math.isfinite(delta)
        ):
            raise VectorProbeError(
                "min_delta_utilization must be finite."
            )

        object.__setattr__(
            self,
            "min_delta_utilization",
            delta,
        )


@dataclass(frozen=True, slots=True)
class VectorEvidenceAssessment:
    """
    Policy interpretation while preserving the raw typed evidence.
    """

    relation_id: str
    decision: VectorEvidenceDecision
    supporting_channels: tuple[str, ...]
    contradicting_channels: tuple[str, ...]
    reasons: tuple[str, ...]
    evidence: VectorProbeEvidence
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relation_id = str(
            self.relation_id
        ).strip()

        if not relation_id:
            raise VectorProbeError(
                "relation_id must not be empty."
            )

        object.__setattr__(
            self,
            "relation_id",
            relation_id,
        )
        object.__setattr__(
            self,
            "decision",
            VectorEvidenceDecision(
                self.decision
            ),
        )
        object.__setattr__(
            self,
            "supporting_channels",
            tuple(
                self.supporting_channels
            ),
        )
        object.__setattr__(
            self,
            "contradicting_channels",
            tuple(
                self.contradicting_channels
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(self.reasons),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                dict(self.metadata)
            ),
        )


def assess_vector_evidence(
    evidence: VectorProbeEvidence,
    *,
    policy: VectorEvidencePolicy | None = None,
) -> VectorEvidenceAssessment:
    """
    Conservatively interpret multidimensional Probe evidence.

    No weighted aggregate score is used.
    """

    policy = (
        policy
        or VectorEvidencePolicy()
    )

    supporting: list[str] = []
    contradicting: list[str] = []
    reasons: list[str] = []

    directional_channels = {
        "tension": evidence.tension,
        "displacement": evidence.displacement,
    }

    meaningful_directional_response = False

    for name, channel in directional_channels.items():
        if channel is None:
            continue

        if channel.polarity is ResponsePolarity.NONE:
            continue

        meaningful_directional_response = True

        if (
            channel.alignment
            >= policy.min_forward_alignment
        ):
            supporting.append(name)
            reasons.append(
                f"{name}:forward_alignment"
            )
        elif (
            channel.alignment
            <= policy.contradiction_alignment
        ):
            contradicting.append(name)
            reasons.append(
                f"{name}:reverse_alignment"
            )

    if contradicting:
        return VectorEvidenceAssessment(
            relation_id=evidence.relation_id,
            decision=VectorEvidenceDecision.CONTRADICTS,
            supporting_channels=tuple(
                supporting
            ),
            contradicting_channels=tuple(
                contradicting
            ),
            reasons=tuple(reasons),
            evidence=evidence,
        )

    if (
        policy.require_tension_or_displacement
        and not meaningful_directional_response
    ):
        return VectorEvidenceAssessment(
            relation_id=evidence.relation_id,
            decision=VectorEvidenceDecision.NO_RESPONSE,
            supporting_channels=(),
            contradicting_channels=(),
            reasons=(
                "no_meaningful_directional_response",
            ),
            evidence=evidence,
        )

    latency_ok = True

    if (
        policy.max_support_latency is not None
    ):
        if (
            evidence.latency is None
            or evidence.latency.onset_time is None
        ):
            latency_ok = False
            reasons.append(
                "latency_unavailable"
            )
        elif (
            evidence.latency.onset_time
            > policy.max_support_latency
        ):
            latency_ok = False
            reasons.append(
                "response_too_late"
            )
        else:
            reasons.append(
                "latency_supported"
            )

    capacity_relevant = True

    if policy.require_capacity_relevance:
        utilization = evidence.utilization

        if utilization is None:
            capacity_relevant = False
            reasons.append(
                "utilization_unavailable"
            )
        else:
            capacity_relevant = (
                utilization.delta_utilization
                >= policy.min_delta_utilization
                or (
                    not utilization.baseline_capacity_exceeded
                    and utilization.probe_capacity_exceeded
                )
            )

            reasons.append(
                "capacity_relevant"
                if capacity_relevant
                else "capacity_not_relevant"
            )

    elif evidence.utilization is not None:
        if (
            evidence.utilization.delta_utilization
            >= policy.min_delta_utilization
        ):
            supporting.append(
                "utilization"
            )
            reasons.append(
                "utilization_change_supported"
            )

    if (
        supporting
        and latency_ok
        and capacity_relevant
    ):
        decision = (
            VectorEvidenceDecision.SUPPORTS
        )
    elif not meaningful_directional_response:
        decision = (
            VectorEvidenceDecision.NO_RESPONSE
        )
    else:
        decision = (
            VectorEvidenceDecision.INSUFFICIENT
        )

    return VectorEvidenceAssessment(
        relation_id=evidence.relation_id,
        decision=decision,
        supporting_channels=tuple(
            dict.fromkeys(
                supporting
            )
        ),
        contradicting_channels=tuple(
            contradicting
        ),
        reasons=tuple(
            dict.fromkeys(
                reasons
            )
        ),
        evidence=evidence,
    )


# =============================================================================
# High-level builder
# =============================================================================


def build_vector_probe_evidence(
    *,
    relation_id: str,
    geometry: CandidateEdgeGeometry,
    baseline_tension: Sequence[float] | FloatVector | None = None,
    probe_tension: Sequence[float] | FloatVector | None = None,
    baseline_displacement: Sequence[float] | FloatVector | None = None,
    probe_displacement: Sequence[float] | FloatVector | None = None,
    response_series: Sequence[
        Sequence[float]
    ]
    | NDArray[np.float64]
    | None = None,
    response_baseline_vector: Sequence[float] | FloatVector | None = None,
    response_threshold: float | None = None,
    response_dt: float = 1.0,
    baseline_demand: float | None = None,
    baseline_capacity: float | None = None,
    probe_demand: float | None = None,
    probe_capacity: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> VectorProbeEvidence:
    """
    Build typed evidence for one candidate relation.

    Every evidence channel is optional, but paired inputs must be complete.
    """

    tension_evidence = None

    if (
        baseline_tension is not None
        or probe_tension is not None
    ):
        if (
            baseline_tension is None
            or probe_tension is None
        ):
            raise VectorProbeError(
                "baseline_tension and probe_tension must be supplied together."
            )

        tension_response = VectorResponse(
            baseline=_vector(
                baseline_tension,
                name="baseline_tension",
            ),
            probe=_vector(
                probe_tension,
                name="probe_tension",
            ),
        )

        tension_evidence = (
            analyze_directional_response(
                tension_response,
                geometry.direction,
            )
        )

    displacement_evidence = None

    if (
        baseline_displacement is not None
        or probe_displacement is not None
    ):
        if (
            baseline_displacement is None
            or probe_displacement is None
        ):
            raise VectorProbeError(
                "baseline_displacement and probe_displacement "
                "must be supplied together."
            )

        displacement_response = VectorResponse(
            baseline=_vector(
                baseline_displacement,
                name="baseline_displacement",
            ),
            probe=_vector(
                probe_displacement,
                name="probe_displacement",
            ),
        )

        displacement_evidence = (
            analyze_directional_response(
                displacement_response,
                geometry.direction,
            )
        )

    latency = None

    temporal_inputs = (
        response_series is not None,
        response_baseline_vector is not None,
        response_threshold is not None,
    )

    if any(temporal_inputs):
        if not all(temporal_inputs):
            raise VectorProbeError(
                "response_series, response_baseline_vector, and "
                "response_threshold must be supplied together."
            )

        latency = response_latency(
            response_series,
            baseline_vector=response_baseline_vector,
            threshold=float(
                response_threshold
            ),
            dt=response_dt,
        )

    utilization = None

    utilization_inputs = (
        baseline_demand,
        baseline_capacity,
        probe_demand,
        probe_capacity,
    )

    if any(
        value is not None
        for value in utilization_inputs
    ):
        if not all(
            value is not None
            for value in utilization_inputs
        ):
            raise VectorProbeError(
                "baseline/probe demand and capacity values "
                "must be supplied together."
            )

        utilization = (
            analyze_utilization_response(
                baseline_demand=float(
                    baseline_demand
                ),
                baseline_capacity=float(
                    baseline_capacity
                ),
                probe_demand=float(
                    probe_demand
                ),
                probe_capacity=float(
                    probe_capacity
                ),
            )
        )

    return VectorProbeEvidence(
        relation_id=relation_id,
        source_id=geometry.source_id,
        target_id=geometry.target_id,
        edge_direction=geometry.direction,
        tension=tension_evidence,
        displacement=displacement_evidence,
        latency=latency,
        utilization=utilization,
        metadata=metadata or {},
    )


__all__ = [
    "CandidateEdgeGeometry",
    "DirectionalResponseEvidence",
    "ResponseLatency",
    "ResponsePolarity",
    "UtilizationResponse",
    "VectorEvidenceAssessment",
    "VectorEvidenceDecision",
    "VectorEvidencePolicy",
    "VectorProbeError",
    "VectorProbeEvidence",
    "VectorResponse",
    "analyze_directional_response",
    "analyze_utilization_response",
    "assess_vector_evidence",
    "build_vector_probe_evidence",
    "edge_direction",
    "response_latency",
]
