"""
Tests for roif.vector_probe

The suite validates physically typed Probe evidence for candidate cascade
relations in pre-stressed systems.

Core principle:

    response amplitude alone is not causal confirmation.

Candidate-edge evidence is evaluated through:
- geometry;
- vector direction;
- signed projection;
- perpendicular residual;
- temporal onset;
- utilization change;
- conservative multidimensional policy.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from types import MappingProxyType

import numpy as np
import pytest

from roif.vector_probe import (
    CandidateEdgeGeometry,
    DirectionalResponseEvidence,
    ResponseLatency,
    ResponsePolarity,
    UtilizationResponse,
    VectorEvidenceAssessment,
    VectorEvidenceDecision,
    VectorEvidencePolicy,
    VectorProbeError,
    VectorProbeEvidence,
    VectorResponse,
    analyze_directional_response,
    analyze_utilization_response,
    assess_vector_evidence,
    build_vector_probe_evidence,
    edge_direction,
    response_latency,
)


# =============================================================================
# Geometry
# =============================================================================


def test_edge_direction_x_axis() -> None:
    direction = edge_direction(
        (0.0, 0.0),
        (2.0, 0.0),
    )

    assert direction.tolist() == pytest.approx(
        [1.0, 0.0]
    )


def test_edge_direction_is_normalized() -> None:
    direction = edge_direction(
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 0.0),
    )

    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction.tolist() == pytest.approx(
        [0.6, 0.8, 0.0]
    )


def test_edge_direction_arrays_are_read_only() -> None:
    direction = edge_direction(
        (0.0, 0.0),
        (1.0, 0.0),
    )

    assert direction.flags.writeable is False

    with pytest.raises(ValueError):
        direction[0] = 2.0


def test_edge_direction_rejects_identical_points() -> None:
    with pytest.raises(VectorProbeError):
        edge_direction(
            (1.0, 2.0),
            (1.0, 2.0),
        )


def test_edge_direction_rejects_dimension_mismatch() -> None:
    with pytest.raises(VectorProbeError):
        edge_direction(
            (0.0, 0.0),
            (1.0, 0.0, 0.0),
        )


def test_candidate_edge_geometry_contract() -> None:
    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(3.0, 4.0),
        metadata={"case": "test"},
    )

    assert geometry.source_id == "A"
    assert geometry.target_id == "B"
    assert geometry.length == pytest.approx(5.0)
    assert geometry.direction.tolist() == pytest.approx(
        [0.6, 0.8]
    )
    assert isinstance(
        geometry.metadata,
        MappingProxyType,
    )


def test_candidate_edge_geometry_is_frozen() -> None:
    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    with pytest.raises(FrozenInstanceError):
        geometry.length = 3.0  # type: ignore[misc]


def test_candidate_edge_geometry_rejects_self_edge() -> None:
    with pytest.raises(VectorProbeError):
        CandidateEdgeGeometry(
            source_id="A",
            target_id="A",
            source_position=(0.0, 0.0),
            target_position=(1.0, 0.0),
        )


# =============================================================================
# VectorResponse
# =============================================================================


def test_vector_response_delta_and_magnitude() -> None:
    response = VectorResponse(
        baseline=(1.0, 1.0),
        probe=(4.0, 5.0),
    )

    assert response.delta.tolist() == pytest.approx(
        [3.0, 4.0]
    )
    assert response.magnitude == pytest.approx(5.0)


def test_vector_response_supports_negative_change() -> None:
    response = VectorResponse(
        baseline=(2.0, 0.0),
        probe=(1.0, 0.0),
    )

    assert response.delta.tolist() == pytest.approx(
        [-1.0, 0.0]
    )
    assert response.magnitude == pytest.approx(1.0)


def test_vector_response_arrays_are_read_only() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(1.0, 0.0),
    )

    assert response.baseline.flags.writeable is False
    assert response.probe.flags.writeable is False
    assert response.delta.flags.writeable is False


def test_vector_response_rejects_dimension_mismatch() -> None:
    with pytest.raises(VectorProbeError):
        VectorResponse(
            baseline=(0.0, 0.0),
            probe=(1.0, 0.0, 0.0),
        )


# =============================================================================
# Directional evidence
# =============================================================================


def test_forward_response_has_alignment_one() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(5.0, 0.0),
    )

    evidence = analyze_directional_response(
        response,
        (1.0, 0.0),
    )

    assert isinstance(
        evidence,
        DirectionalResponseEvidence,
    )
    assert evidence.alignment == pytest.approx(1.0)
    assert evidence.signed_projection == pytest.approx(5.0)
    assert evidence.perpendicular_magnitude == pytest.approx(0.0)
    assert evidence.polarity is ResponsePolarity.FORWARD


def test_reverse_response_has_alignment_minus_one() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(-3.0, 0.0),
    )

    evidence = analyze_directional_response(
        response,
        (1.0, 0.0),
    )

    assert evidence.alignment == pytest.approx(-1.0)
    assert evidence.signed_projection == pytest.approx(-3.0)
    assert evidence.polarity is ResponsePolarity.REVERSE


def test_orthogonal_response_is_detected() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(0.0, 4.0),
    )

    evidence = analyze_directional_response(
        response,
        (1.0, 0.0),
    )

    assert evidence.alignment == pytest.approx(0.0)
    assert evidence.signed_projection == pytest.approx(0.0)
    assert evidence.perpendicular_magnitude == pytest.approx(4.0)
    assert evidence.polarity is ResponsePolarity.ORTHOGONAL


def test_diagonal_forward_response_has_expected_alignment() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(1.0, 1.0),
    )

    evidence = analyze_directional_response(
        response,
        (1.0, 0.0),
    )

    assert evidence.alignment == pytest.approx(
        1.0 / math.sqrt(2.0)
    )
    assert evidence.signed_projection == pytest.approx(1.0)
    assert evidence.perpendicular_magnitude == pytest.approx(1.0)


def test_zero_response_has_none_polarity() -> None:
    response = VectorResponse(
        baseline=(1.0, 2.0),
        probe=(1.0, 2.0),
    )

    evidence = analyze_directional_response(
        response,
        (1.0, 0.0),
    )

    assert evidence.response_magnitude == pytest.approx(0.0)
    assert evidence.alignment == pytest.approx(0.0)
    assert evidence.polarity is ResponsePolarity.NONE


def test_direction_vector_is_normalized_internally() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(5.0, 0.0),
    )

    evidence = analyze_directional_response(
        response,
        (100.0, 0.0),
    )

    assert evidence.alignment == pytest.approx(1.0)
    assert evidence.signed_projection == pytest.approx(5.0)


def test_directional_analysis_rejects_zero_direction() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(1.0, 0.0),
    )

    with pytest.raises(VectorProbeError):
        analyze_directional_response(
            response,
            (0.0, 0.0),
        )


def test_directional_analysis_rejects_dimension_mismatch() -> None:
    response = VectorResponse(
        baseline=(0.0, 0.0),
        probe=(1.0, 0.0),
    )

    with pytest.raises(VectorProbeError):
        analyze_directional_response(
            response,
            (1.0, 0.0, 0.0),
        )


# =============================================================================
# Temporal evidence
# =============================================================================


def test_response_latency_finds_first_threshold_crossing() -> None:
    series = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.6, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = response_latency(
        series,
        baseline_vector=(0.0, 0.0),
        threshold=0.5,
        dt=0.2,
    )

    assert isinstance(result, ResponseLatency)
    assert result.onset_index == 2
    assert result.onset_time == pytest.approx(0.4)
    assert result.peak_index == 3
    assert result.peak_time == pytest.approx(0.6)
    assert result.peak_magnitude == pytest.approx(1.0)


def test_response_latency_returns_none_on_no_threshold_crossing() -> None:
    series = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float64,
    )

    result = response_latency(
        series,
        baseline_vector=(0.0, 0.0),
        threshold=1.0,
        dt=1.0,
    )

    assert result.onset_index is None
    assert result.onset_time is None
    assert result.peak_index == 2
    assert result.peak_magnitude == pytest.approx(0.2)


def test_response_latency_supports_multidimensional_norm() -> None:
    series = np.asarray(
        [
            [0.0, 0.0],
            [0.3, 0.4],
        ],
        dtype=np.float64,
    )

    result = response_latency(
        series,
        baseline_vector=(0.0, 0.0),
        threshold=0.5,
    )

    assert result.onset_index == 1
    assert result.peak_magnitude == pytest.approx(0.5)


def test_response_latency_rejects_invalid_dt() -> None:
    with pytest.raises(VectorProbeError):
        response_latency(
            [[0.0, 0.0]],
            baseline_vector=(0.0, 0.0),
            threshold=0.1,
            dt=0.0,
        )


def test_response_latency_rejects_dimension_mismatch() -> None:
    with pytest.raises(VectorProbeError):
        response_latency(
            [[0.0, 0.0, 0.0]],
            baseline_vector=(0.0, 0.0),
            threshold=0.1,
        )


# =============================================================================
# Utilization evidence
# =============================================================================


def test_utilization_response_basic() -> None:
    result = analyze_utilization_response(
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=80.0,
        probe_capacity=100.0,
    )

    assert isinstance(result, UtilizationResponse)
    assert result.baseline_utilization == pytest.approx(0.5)
    assert result.probe_utilization == pytest.approx(0.8)
    assert result.delta_utilization == pytest.approx(0.3)
    assert result.baseline_capacity_exceeded is False
    assert result.probe_capacity_exceeded is False


def test_utilization_response_detects_capacity_crossing() -> None:
    result = analyze_utilization_response(
        baseline_demand=90.0,
        baseline_capacity=100.0,
        probe_demand=120.0,
        probe_capacity=100.0,
    )

    assert result.baseline_capacity_exceeded is False
    assert result.probe_capacity_exceeded is True
    assert result.delta_utilization == pytest.approx(0.3)


def test_utilization_response_detects_capacity_loss() -> None:
    result = analyze_utilization_response(
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=50.0,
        probe_capacity=40.0,
    )

    assert result.baseline_utilization == pytest.approx(0.5)
    assert result.probe_utilization == pytest.approx(1.25)
    assert result.probe_capacity_exceeded is True


def test_utilization_response_uses_zero_capacity_convention() -> None:
    result = analyze_utilization_response(
        baseline_demand=0.0,
        baseline_capacity=0.0,
        probe_demand=1.0,
        probe_capacity=0.0,
    )

    assert result.baseline_utilization == pytest.approx(0.0)
    assert math.isinf(result.probe_utilization)
    assert math.isinf(result.delta_utilization)
    assert result.probe_capacity_exceeded is True


# =============================================================================
# High-level evidence builder
# =============================================================================


@pytest.fixture
def horizontal_geometry() -> CandidateEdgeGeometry:
    return CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )


def test_build_vector_probe_evidence_tension_only(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(2.0, 0.0),
    )

    assert isinstance(
        evidence,
        VectorProbeEvidence,
    )
    assert evidence.tension is not None
    assert evidence.tension.alignment == pytest.approx(1.0)
    assert evidence.displacement is None
    assert evidence.latency is None
    assert evidence.utilization is None


def test_build_vector_probe_evidence_full(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        baseline_displacement=(0.0, 0.0),
        probe_displacement=(0.5, 0.0),
        response_series=[
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
        ],
        response_baseline_vector=(0.0, 0.0),
        response_threshold=0.5,
        response_dt=0.1,
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=80.0,
        probe_capacity=100.0,
        metadata={"probe": "p1"},
    )

    assert evidence.tension is not None
    assert evidence.displacement is not None
    assert evidence.latency is not None
    assert evidence.utilization is not None

    assert evidence.tension.polarity is ResponsePolarity.FORWARD
    assert evidence.displacement.polarity is ResponsePolarity.FORWARD
    assert evidence.latency.onset_index == 2
    assert evidence.latency.onset_time == pytest.approx(0.2)
    assert evidence.utilization.delta_utilization == pytest.approx(0.3)
    assert isinstance(
        evidence.metadata,
        MappingProxyType,
    )


def test_builder_requires_tension_pair(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    with pytest.raises(VectorProbeError):
        build_vector_probe_evidence(
            relation_id="A_to_B",
            geometry=horizontal_geometry,
            baseline_tension=(0.0, 0.0),
        )


def test_builder_requires_displacement_pair(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    with pytest.raises(VectorProbeError):
        build_vector_probe_evidence(
            relation_id="A_to_B",
            geometry=horizontal_geometry,
            probe_displacement=(1.0, 0.0),
        )


def test_builder_requires_complete_temporal_inputs(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    with pytest.raises(VectorProbeError):
        build_vector_probe_evidence(
            relation_id="A_to_B",
            geometry=horizontal_geometry,
            response_series=[[0.0, 0.0]],
            response_threshold=0.1,
        )


def test_builder_requires_complete_utilization_inputs(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    with pytest.raises(VectorProbeError):
        build_vector_probe_evidence(
            relation_id="A_to_B",
            geometry=horizontal_geometry,
            baseline_demand=10.0,
            baseline_capacity=20.0,
        )


# =============================================================================
# Policy assessment
# =============================================================================


def test_forward_tension_supports_candidate(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert isinstance(
        assessment,
        VectorEvidenceAssessment,
    )
    assert assessment.decision is VectorEvidenceDecision.SUPPORTS
    assert "tension" in assessment.supporting_channels


def test_reverse_tension_contradicts_candidate(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(-1.0, 0.0),
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert assessment.decision is VectorEvidenceDecision.CONTRADICTS
    assert "tension" in assessment.contradicting_channels


def test_orthogonal_response_is_insufficient_not_supportive(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(0.0, 1.0),
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert assessment.decision is VectorEvidenceDecision.INSUFFICIENT
    assert "tension" not in assessment.supporting_channels


def test_zero_response_returns_no_response(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(1.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert assessment.decision is VectorEvidenceDecision.NO_RESPONSE


def test_displacement_can_support_without_tension(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_displacement=(0.0, 0.0),
        probe_displacement=(1.0, 0.0),
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert assessment.decision is VectorEvidenceDecision.SUPPORTS
    assert assessment.supporting_channels == (
        "displacement",
    )


def test_contradiction_overrides_support() -> None:
    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        baseline_displacement=(0.0, 0.0),
        probe_displacement=(-1.0, 0.0),
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert assessment.decision is VectorEvidenceDecision.CONTRADICTS
    assert "tension" in assessment.supporting_channels
    assert "displacement" in assessment.contradicting_channels


def test_latency_gate_can_make_directional_support_insufficient(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        response_series=[
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        response_baseline_vector=(0.0, 0.0),
        response_threshold=0.5,
        response_dt=1.0,
    )

    assessment = assess_vector_evidence(
        evidence,
        policy=VectorEvidencePolicy(
            max_support_latency=1.0,
        ),
    )

    assert evidence.latency is not None
    assert evidence.latency.onset_time == pytest.approx(2.0)
    assert assessment.decision is VectorEvidenceDecision.INSUFFICIENT
    assert "response_too_late" in assessment.reasons


def test_latency_gate_supports_early_response(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        response_series=[
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        response_baseline_vector=(0.0, 0.0),
        response_threshold=0.5,
        response_dt=0.1,
    )

    assessment = assess_vector_evidence(
        evidence,
        policy=VectorEvidencePolicy(
            max_support_latency=0.2,
        ),
    )

    assert assessment.decision is VectorEvidenceDecision.SUPPORTS
    assert "latency_supported" in assessment.reasons


def test_capacity_relevance_can_be_required(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=51.0,
        probe_capacity=100.0,
    )

    assessment = assess_vector_evidence(
        evidence,
        policy=VectorEvidencePolicy(
            require_capacity_relevance=True,
            min_delta_utilization=0.10,
        ),
    )

    assert assessment.decision is VectorEvidenceDecision.INSUFFICIENT
    assert "capacity_not_relevant" in assessment.reasons


def test_capacity_crossing_can_make_response_relevant(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        baseline_demand=90.0,
        baseline_capacity=100.0,
        probe_demand=110.0,
        probe_capacity=100.0,
    )

    assessment = assess_vector_evidence(
        evidence,
        policy=VectorEvidencePolicy(
            require_capacity_relevance=True,
            min_delta_utilization=1.0,
        ),
    )

    assert evidence.utilization is not None
    assert evidence.utilization.probe_capacity_exceeded is True
    assert assessment.decision is VectorEvidenceDecision.SUPPORTS
    assert "capacity_relevant" in assessment.reasons


def test_utilization_alone_cannot_confirm_directed_edge_by_default(
    horizontal_geometry: CandidateEdgeGeometry,
) -> None:
    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=horizontal_geometry,
        baseline_demand=50.0,
        baseline_capacity=100.0,
        probe_demand=120.0,
        probe_capacity=100.0,
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert assessment.decision is VectorEvidenceDecision.NO_RESPONSE


# =============================================================================
# Pre-stressed branching examples
# =============================================================================


def test_larger_orthogonal_response_does_not_beat_smaller_aligned_response() -> None:
    """
    The strongest response is not automatically the next cascade edge.

    Candidate A->B:
        smaller but perfectly aligned response.

    Candidate A->C:
        much larger response, but orthogonal to its edge.
    """

    geometry_b = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    geometry_c = CandidateEdgeGeometry(
        source_id="A",
        target_id="C",
        source_position=(0.0, 0.0),
        target_position=(0.0, 1.0),
    )

    evidence_b = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=geometry_b,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
    )

    # Magnitude 10, but C edge is vertical while response is horizontal.
    evidence_c = build_vector_probe_evidence(
        relation_id="A_to_C",
        geometry=geometry_c,
        baseline_tension=(0.0, 0.0),
        probe_tension=(10.0, 0.0),
    )

    assessment_b = assess_vector_evidence(
        evidence_b
    )
    assessment_c = assess_vector_evidence(
        evidence_c
    )

    assert (
        evidence_c.tension.response_magnitude
        > evidence_b.tension.response_magnitude
    )

    assert assessment_b.decision is VectorEvidenceDecision.SUPPORTS
    assert assessment_c.decision is VectorEvidenceDecision.INSUFFICIENT


def test_reverse_high_amplitude_response_contradicts_edge() -> None:
    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(-100.0, 0.0),
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert evidence.tension is not None
    assert evidence.tension.response_magnitude == pytest.approx(100.0)
    assert assessment.decision is VectorEvidenceDecision.CONTRADICTS


def test_early_aligned_candidate_can_be_distinguished_from_late_aligned_candidate() -> None:
    geometry_b = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )
    geometry_c = CandidateEdgeGeometry(
        source_id="A",
        target_id="C",
        source_position=(0.0, 0.0),
        target_position=(0.0, 1.0),
    )

    evidence_b = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=geometry_b,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        response_series=[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        response_baseline_vector=(0.0, 0.0),
        response_threshold=0.5,
        response_dt=0.1,
    )

    evidence_c = build_vector_probe_evidence(
        relation_id="A_to_C",
        geometry=geometry_c,
        baseline_tension=(0.0, 0.0),
        probe_tension=(0.0, 2.0),
        response_series=[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 2.0],
        ],
        response_baseline_vector=(0.0, 0.0),
        response_threshold=0.5,
        response_dt=0.1,
    )

    policy = VectorEvidencePolicy(
        max_support_latency=0.15,
    )

    assessment_b = assess_vector_evidence(
        evidence_b,
        policy=policy,
    )
    assessment_c = assess_vector_evidence(
        evidence_c,
        policy=policy,
    )

    assert assessment_b.decision is VectorEvidenceDecision.SUPPORTS
    assert assessment_c.decision is VectorEvidenceDecision.INSUFFICIENT


def test_utilization_change_is_preserved_as_separate_evidence_channel() -> None:
    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(1.0, 0.0),
    )

    evidence = build_vector_probe_evidence(
        relation_id="A_to_B",
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(1.0, 0.0),
        baseline_demand=60.0,
        baseline_capacity=100.0,
        probe_demand=95.0,
        probe_capacity=100.0,
    )

    assessment = assess_vector_evidence(
        evidence
    )

    assert evidence.utilization is not None
    assert evidence.utilization.delta_utilization == pytest.approx(0.35)
    assert "tension" in assessment.supporting_channels
    assert "utilization" in assessment.supporting_channels


def test_vector_probe_evidence_is_deterministic() -> None:
    geometry = CandidateEdgeGeometry(
        source_id="A",
        target_id="B",
        source_position=(0.0, 0.0),
        target_position=(3.0, 4.0),
    )

    kwargs = dict(
        relation_id="A_to_B",
        geometry=geometry,
        baseline_tension=(0.0, 0.0),
        probe_tension=(0.6, 0.8),
        baseline_demand=40.0,
        baseline_capacity=100.0,
        probe_demand=70.0,
        probe_capacity=100.0,
    )

    left = build_vector_probe_evidence(
        **kwargs
    )
    right = build_vector_probe_evidence(
        **kwargs
    )

    assert left.relation_id == right.relation_id
    assert left.source_id == right.source_id
    assert left.target_id == right.target_id

    assert np.array_equal(
        left.edge_direction,
        right.edge_direction,
    )

    assert left.tension == right.tension
    assert left.displacement == right.displacement
    assert left.latency == right.latency
    assert left.utilization == right.utilization
    assert dict(left.metadata) == dict(right.metadata)

    left_assessment = assess_vector_evidence(left)
    right_assessment = assess_vector_evidence(right)

    assert left_assessment.relation_id == right_assessment.relation_id
    assert left_assessment.decision is right_assessment.decision
    assert (
        left_assessment.supporting_channels
        == right_assessment.supporting_channels
    )
    assert (
        left_assessment.contradicting_channels
        == right_assessment.contradicting_channels
    )
    assert left_assessment.reasons == right_assessment.reasons

